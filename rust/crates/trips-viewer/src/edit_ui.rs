//! The Regions / Inspector / Tools panels, and the session that drives them.
//!
//! Module: `trips_viewer::edit_ui` (binary only)
//! Purpose: `docs/EDITOR.md` §4's UI sketch, for milestones E1 and E2. This is
//!     the egui half of the editor; every number it computes comes from
//!     [`trips_viewer::edit`], which knows nothing about windows. The split is
//!     the same one `blend.rs`/`app.rs` already use, and it is what lets the
//!     region maths be unit-tested and the golden test run without a GPU.
//! Invariants:
//!     - Every mutation goes through [`EditDocument`]'s log
//!       (`add_region`/`remove_region`/`update_region`/`reorder`), never by
//!       writing `regions` directly, so undo and redo are cursor moves and
//!       nothing the UI does can drift from what `edits.json` records.
//!     - Region membership is recomputed **once per edit change**, not per
//!       frame: [`EditSession::needs_apply`] is set by the widgets and consumed
//!       once in [`EditSession::apply`], which is also where the timing the HUD
//!       reports comes from (`docs/EDITOR.md` §7 asks for that number).
//!     - New regions are sized and placed from `crate::bundle::SceneScale` and
//!       the camera's own look-at point — never from the point cloud's bounds,
//!       which a TRIPS export's environment sphere makes meaningless
//!       (`renderer.rs`'s `bounds` field says why).
//!     - The 3D drag gizmos are NOT built. E1 ships numeric fields plus
//!         keyboard nudge/resize instead, which the brief explicitly allows; the
//!       keys are listed in [`KEYS_HELP`] and in `docs/USER_GUIDE.md`.
//! Units: world units for every geometry field; `mix` is 0 = splat, 1 = TRIPS.
//! Related docs: `docs/EDITOR.md` §1, §3, §4; `docs/USER_GUIDE.md` "Editor".

use std::path::{Path, PathBuf};

use brush_pyramid::gpu::block_on;
use eframe::egui;
use serde_json::json;

use trips_viewer::edit::apply::{edited_points, gaussian_opacity_scale, tinted_points};
use trips_viewer::edit::brush::{self, BrushCells};
use trips_viewer::edit::cluster::{
    self, ClickCamera, ClickParams, ClickSelection, PointGrid, DEFAULT_MAX_POINTS,
};
use trips_viewer::edit::gizmo::{self, Drag, GizmoScreen};
use trips_viewer::edit::model::{
    auto_name_for, new_region_id, Op, Params, Region, KAREKARE_LID,
};
use trips_viewer::edit::sam::{self as sam_geom, nearest_view};
use trips_viewer::edit::shade::{self, ShadeSelection, ShadeViews, Thresholds};
use trips_viewer::edit::weights::{
    compose_gaussian_weights_solo, compose_trips_weights_solo, widen, RegionCount,
};
use trips_viewer::edit::{
    EditDocument, BRUSH_CELLS_PER_RADIUS, BRUSH_RADIUS_STEP, BRUSH_SAMPLE_PX,
    DEFAULT_BRUSH_SCENE_FRACTION, DEFAULT_REGION_SCENE_FRACTION, EDITS_FILENAME,
    NUDGE_SCENE_FRACTION, RESIZE_STEP,
};
use trips_viewer::renderer::Renderer;

use crate::sam_child::{
    build_command, command_line, imported_region, resolve_interpreter_from_env, SamJob, SamPrompt,
    SamRequest, SamState,
};

/// The editor's key bindings, shown in the panel and in `docs/USER_GUIDE.md`.
///
/// Chosen to avoid every key `app.rs::ViewerApp::handle_input` already binds
/// (`V X B Tab - = F R N P W A S D Q E`), per `docs/EDITOR.md` §4.
pub const KEYS_HELP: &str = "\
M edit mode | T cycle tool | H preview highlight | SHIFT-CLICK the render to select\n\
SAM tool: DRAG a box on the render (SHIFT-drag still orbits) or ALT-CLICK a point\n\
Brush tool: DRAG to paint, ALT-drag to erase (SHIFT-drag still orbits), [ / ] radius\n\
gizmo: DRAG a handle to move it, SHIFT-drag to resize, CTRL-drag to rotate a box\n\
arrows + PageUp/PageDown nudge the selected region | [ / ] shrink / grow (brush radius\n\
while the Brush tool has focus)\n\
Delete removes the selected region | Cmd-Z undo | Cmd-Shift-Z redo | Cmd-S save";

/// Which tool the Tools panel has focus on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tool {
    /// Place and shape box/sphere/lid regions by hand.
    #[default]
    Regions,
    /// Threshold the shade audit's own numbers into a `pointset` region.
    ShadeFinder,
    /// Shift-click the render; grow a `pointset` region from what was clicked.
    ClickCluster,
    /// Drag a box (or Alt-click) on the render; the local SAM 3 segments that
    /// photograph and the mask is lifted onto the points (E5).
    Sam,
    /// Drag on the render to paint a sparse-voxel `brush` region in 3D;
    /// Alt-drag erases (`docs/EDITOR.md` §1 "brush", §4).
    Brush,
}

impl Tool {
    /// Cycle order for the `T` key.
    const fn next(self) -> Self {
        match self {
            Self::Regions => Self::ShadeFinder,
            Self::ShadeFinder => Self::ClickCluster,
            Self::ClickCluster => Self::Sam,
            Self::Sam => Self::Brush,
            Self::Brush => Self::Regions,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Regions => "regions",
            Self::ShadeFinder => "shade-cloud finder",
            Self::ClickCluster => "click-to-cluster",
            Self::Sam => "SAM 3 lift",
            Self::Brush => "brush",
        }
    }
}

/// The shade finder's own state: the sliders, the frames, and the last selection.
pub struct ShadeUi {
    /// The frames the finder tests against, from `shade_views.json` or estimated.
    pub views: ShadeViews,
    /// The four slider values.
    pub thresholds: Thresholds,
    /// The last result of [`shade::find`].
    pub selection: ShadeSelection,
    /// Whether the render tints the selection (`H`).
    pub preview: bool,
    /// Set when a slider moved, cleared once the finder has re-run.
    dirty: bool,
    /// Why the finder cannot run, if it cannot.
    unavailable: Option<String>,
}

impl ShadeUi {
    fn new(views: Option<ShadeViews>, bundle_dir: &Path) -> Self {
        let (views, unavailable) = match views {
            Some(v) if !v.views.is_empty() => (v, None),
            _ => (
                ShadeViews::default(),
                Some(format!(
                    "no {} in {} -- run the precompute (docs/USER_GUIDE.md \"Shade finder\") \
                     to give the finder the shade frames' cameras and median depths",
                    shade::SHADE_VIEWS_FILENAME,
                    bundle_dir.display()
                )),
            ),
        };
        let thresholds = Thresholds {
            znear_frac: views.znear_frac,
            zfar_frac: views.zfar_frac,
            ..Thresholds::default()
        };
        Self {
            views,
            thresholds,
            selection: ShadeSelection::default(),
            preview: false,
            dirty: true,
            unavailable,
        }
    }
}

/// The neighbour index one click needs, and the widened arrays it queries.
///
/// Built on the first click and kept while the click tool has a live selection,
/// so moving a slider re-runs in milliseconds instead of rebuilding the index.
/// Dropped by [`EditSession::clear_click`] and whenever the Tools panel leaves
/// this tool — a `f64` copy of a multi-million-point cloud is not something to
/// hold onto for a tool nobody is using. The panel reports its size.
struct ClickCache {
    /// Flat `(N, 3)` world positions, widened from the renderer's `f32`.
    xyz: Vec<f64>,
    /// Flat `(N, 3)` base colour, `clip(feat[:, :3], 0, 1)`.
    rgb: Vec<f64>,
    /// The spatial hash over `xyz`.
    grid: PointGrid,
    /// Milliseconds the build took.
    build_ms: f64,
}

impl ClickCache {
    /// Widen the renderer's own cloud and index it.
    fn build(points: &brush_pyramid::scene::PointSet) -> Self {
        let started = std::time::Instant::now();
        let xyz = widen(&points.xyz);
        let channels = points.num_channels;
        let mut rgb = Vec::with_capacity(points.len() * 3);
        for row in 0..points.len() {
            for c in 0..3 {
                rgb.push(f64::from(points.feat[row * channels + c].clamp(0.0, 1.0)));
            }
        }
        let grid = PointGrid::build(&xyz);
        Self {
            xyz,
            rgb,
            grid,
            build_ms: started.elapsed().as_secs_f64() * 1e3,
        }
    }

    /// Roughly how much host memory this holds, mebibytes — the panel says so.
    fn megabytes(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let bytes = (self.xyz.len() * 8 + self.rgb.len() * 8 + self.grid.len() * 8) as f64;
        bytes / (1024.0 * 1024.0)
    }
}

/// The click tool's own state: the sliders, the last click, and what it selected.
pub struct ClickUi {
    /// The four sliders (`docs/EDITOR.md` §4's "size/tightness slider", grown
    /// into the same four numbers `trippy edits click` takes).
    pub params: ClickParams,
    /// The last result of [`cluster::click_to_cluster`].
    pub selection: ClickSelection,
    /// Whether the render tints the selection (`H`).
    pub preview: bool,
    /// The op "add as region" will give the new region.
    pub op: Op,
    /// The mix "add as region" will give it (ignored by `delete`).
    pub mix: f64,
    /// The camera and pixel of the last click, so a slider move re-runs it
    /// against the frame it was made in rather than wherever the camera is now.
    last: Option<(ClickCamera, (f64, f64))>,
    /// Set by a Shift-click; consumed by [`EditSession::resolve_click`].
    pending: Option<(f64, f64)>,
    /// Set by a slider; also consumed by `resolve_click`.
    dirty: bool,
    /// The neighbour index, while the tool is in use.
    cache: Option<ClickCache>,
    /// Milliseconds the last clustering took.
    ms: f64,
    /// A one-line "what just happened" for the panel.
    note: String,
}

impl Default for ClickUi {
    fn default() -> Self {
        Self {
            params: ClickParams::default(),
            selection: ClickSelection::default(),
            preview: true,
            op: Op::Fade,
            mix: cluster::DEFAULT_MIX,
            last: None,
            pending: None,
            dirty: false,
            cache: None,
            ms: 0.0,
            note: String::new(),
        }
    }
}

/// A gesture on the render, waiting for the frame's camera to resolve it.
///
/// Recorded in `app.rs` while the pointer is still in egui's coordinates and
/// consumed by [`EditSession::resolve_sam`] once the frame's camera exists —
/// exactly the two-step the Shift-click already uses, and for the same reason.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamGesture {
    /// A drag rectangle, both corners in RENDER pixels.
    Box((f64, f64), (f64, f64)),
    /// An Alt-click, in RENDER pixels.
    Point((f64, f64)),
}

/// The SAM tool's own state: the prompt, the child process, and what it found.
///
/// `docs/EDITOR.md` §3's "4. SAM 3 lift (E5)". The lift itself is Python and
/// runs as a child process ([`crate::sam_child`]); everything here is the
/// gesture, the settings that become its flags, and the region it hands back.
pub struct SamUi {
    /// The gesture waiting for a camera, set by `app.rs`.
    pending: Option<SamGesture>,
    /// The resolved prompt, in the capture VIEW's pixels, with its view name.
    prompt: Option<(String, SamPrompt)>,
    /// `--views-around`. 0 = the prompted view alone; `docs/EDITOR.md` §3 says
    /// to use 0 or >= 2, because 1 neighbour makes the vote an intersection.
    views_around: usize,
    /// The op the imported region gets.
    op: Op,
    /// The mix it gets (ignored by `delete`).
    mix: f64,
    /// `--device`. `mps` is Jordan's own interactive GPU use, which
    /// `AGENTS.md` §6 allows outside the queue; it is not the default.
    device: &'static str,
    /// `--fake`: synthesise the mask instead of loading SAM 3. The screenshot
    /// proof and the tests use it; the panel exposes it so a broken SAM
    /// install can be told apart from a broken lift.
    fake: bool,
    /// The running (or just-finished) child, if any.
    job: Option<SamJob>,
    /// The command line the last run used, shown so it can be re-run by hand.
    command: String,
    /// The temporary directory holding the child's throwaway `edits.json`.
    /// Kept alive for the run and replaced on the next one.
    work_dir: Option<PathBuf>,
    /// The point ids of the last imported region, for the tint.
    selection: Vec<u32>,
    /// Whether the render tints that selection (`H`).
    preview: bool,
    /// The finished run's own counts, for the panel.
    summary: Option<serde_json::Value>,
    /// A view position `app.rs` should snap the camera to, because the SAM
    /// lift needs a photograph and the camera was not on one.
    snap_request: Option<usize>,
    /// A one-line "what just happened".
    note: String,
}

impl Default for SamUi {
    fn default() -> Self {
        Self {
            pending: None,
            prompt: None,
            views_around: 0,
            op: Op::Fade,
            mix: 0.0,
            device: "cpu",
            fake: false,
            job: None,
            command: String::new(),
            work_dir: None,
            selection: Vec::new(),
            preview: true,
            summary: None,
            snap_request: None,
            note: String::new(),
        }
    }
}

/// The brush tool's own state: the settings, the live stroke, and the cloud it
/// anchors against.
///
/// `docs/EDITOR.md` §1 "brush" / §4. The tool paints a `brush`-kind region:
/// each sample of a drag becomes a world-space sphere at the depth of the
/// nearest point under the cursor, and the union of those spheres' voxels is
/// the region.
pub struct BrushUi {
    /// Sphere radius, world units. `[` / `]` change it while this tool has focus.
    radius: f64,
    /// The weight painted cells get, `[0, 1]`.
    weight: f64,
    /// The op a NEW brush region is created with.
    op: Op,
    /// The mix it is created with (ignored by `delete`).
    mix: f64,
    /// The region strokes go into. A fresh stroke starts a new region when this
    /// is `None` or names a region that is gone.
    region: Option<String>,
    /// The stroke in progress, if the button is down.
    stroke: Option<Stroke>,
    /// Render pixels waiting for the frame's camera, oldest first.
    pending: Vec<(f64, f64)>,
    /// A widened `(N, 3)` copy of the cloud, for the depth anchor. Built on the
    /// first stroke and dropped when the tool loses focus, exactly as
    /// [`ClickCache`] is and for the same reason.
    cache: Option<Vec<f64>>,
    /// The screen-space bucket index [`brush::depth_anchor`]'s `O(points)` scan
    /// used to cost every sample of a drag (measured at Karekare scale in
    /// `research/trips-metal.md`, `trippy-brush-anchor-perf-1`). Rebuilt only
    /// when the camera changes (`ClickCamera` is `PartialEq`, compared in
    /// [`EditSession::resolve_brush`]), so a still camera reuses it across
    /// every sample of a stroke — and across strokes, until the pointer
    /// actually orbits.
    screen_grid: Option<(ClickCamera, brush::ScreenGrid)>,
    /// The last depth the anchor found, world units. Kept so a sample that
    /// lands on empty space continues the stroke at the depth it started
    /// rather than inventing one.
    depth: Option<f64>,
    /// Where the cursor was last painted, render pixels, and the radius the
    /// stroke is painting at there, for the on-screen circle.
    cursor: Option<((f64, f64), f64)>,
    /// A one-line "what just happened".
    note: String,
}

/// One brush gesture, from button-down to button-up.
struct Stroke {
    /// The region being painted.
    region_id: String,
    /// The name a region this stroke CREATES gets. Chosen once, when the
    /// stroke starts: recomputing it per frame would see the region the
    /// stroke's own previous frame added and count it, so a single stroke
    /// would walk up "brush-1", "brush-2", "brush-3" as it was painted.
    name: String,
    /// The grid the region was created on (never changes once it exists).
    origin: [f64; 3],
    /// Voxel edge, world units.
    cell_size: f64,
    /// The cells as they stand mid-stroke.
    cells: BrushCells,
    /// Whether this stroke erases instead of painting (Alt).
    erasing: bool,
    /// Whether the region was CREATED by this stroke, so the log entry to
    /// coalesce into is an `add_region` rather than an `update_region`.
    creating: bool,
    /// Whether this stroke has already written to the document — the flag that
    /// turns the second and later samples into coalesced amendments of the
    /// first, so one stroke is one undo step.
    emitted: bool,
    /// The last sample's pixel, so samples are spaced by `BRUSH_SAMPLE_PX`.
    last_px: Option<(f64, f64)>,
    /// How many samples this stroke has painted, for the note.
    samples: usize,
}

impl Default for BrushUi {
    fn default() -> Self {
        Self {
            // Replaced by `init_brush_defaults` once the scene's scale is known;
            // a zero radius paints nothing, which is the safe sentinel.
            radius: 0.0,
            weight: 1.0,
            op: Op::Delete,
            mix: 0.0,
            region: None,
            stroke: None,
            pending: Vec::new(),
            cache: None,
            screen_grid: None,
            depth: None,
            cursor: None,
            note: String::new(),
        }
    }
}

/// The 3D drag gizmos' state: this frame's handles, and the drag in progress.
///
/// `docs/EDITOR.md` §6's E1/E3 row. The maths is in [`gizmo`]; this is the part
/// that remembers which handle the pointer grabbed and what the region looked
/// like before the drag started.
#[derive(Default)]
pub struct GizmoUi {
    /// The selected region's handles, projected with the last frame's camera.
    /// `None` when nothing is selected, the region has no shape, or it is
    /// behind the camera.
    screen: Option<GizmoScreen>,
    /// The drag in progress.
    drag: Option<GizmoDrag>,
    /// Whether the handles are drawn at all (the panel's checkbox).
    show: bool,
}

/// One gizmo gesture.
struct GizmoDrag {
    /// The region being dragged.
    region_id: String,
    /// Which world axis was grabbed.
    axis: usize,
    /// What the modifiers made of it.
    kind: Drag,
    /// The region's geometry when the drag started; every frame recomputes the
    /// result from THIS, so a long drag cannot accumulate rounding drift.
    base: Params,
    /// The handles as they were when the drag started, for the same reason.
    screen: GizmoScreen,
    /// Where the pointer went down, render pixels.
    start_px: (f64, f64),
    /// Whether a document entry has been written yet (the coalescing flag).
    emitted: bool,
}

/// Everything the editor holds for one open bundle.
pub struct EditSession {
    /// The document, and the only place regions live.
    pub doc: EditDocument,
    /// `<bundle>/edits.json`.
    pub path: PathBuf,
    /// Whether there are unsaved changes.
    pub dirty: bool,
    /// Whether the edit panels are shown and the edit keys are live (`M`).
    pub active: bool,
    /// Which region the Inspector is editing.
    selected: Option<String>,
    /// Which tool the Tools panel has focus on (`T`).
    tool: Tool,
    /// The shade finder.
    shade: ShadeUi,
    /// The click-to-cluster tool.
    click: ClickUi,
    /// The SAM 3 lift (E5).
    sam: SamUi,
    /// The brush tool.
    brush: BrushUi,
    /// The 3D drag gizmos.
    gizmo: GizmoUi,
    /// The Named Objects panel's "solo" region: while it is set, ONLY that
    /// region composes, so its own effect can be seen apart from every other
    /// (`docs/EDITOR.md` §4). Never saved — it is a way of looking, not an edit.
    solo: Option<String>,
    /// How many points each region claimed at the last [`Self::apply`], for the
    /// Named Objects panel's count column.
    counts: Vec<RegionCount>,
    /// The row the Named Objects panel is renaming, and its buffer.
    rename_for: Option<String>,
    /// See [`Self::rename_for`].
    rename_buffer: String,
    /// The directory holding `bundle.json`. NOT `self.path`'s parent: `--edits`
    /// can put the edit document anywhere, and the SAM child is given a bundle.
    bundle_dir: PathBuf,
    /// `bundle.json`'s own `trippy_root`, for resolving the child's python.
    trippy_root: Option<String>,
    /// Whether `bundle.json` records a `scene_root`. Without one the child
    /// cannot find the photographs and the SAM tool refuses to run.
    has_scene_root: bool,
    /// Set by any widget that changed the document; consumed by [`Self::apply`].
    needs_apply: bool,
    /// Milliseconds the last [`Self::apply`] took — `docs/EDITOR.md` §7's budget
    /// number, measured on the cloud actually loaded rather than estimated.
    last_apply_ms: f64,
    /// The last thing that went wrong, shown in the panel rather than swallowed.
    error: Option<String>,
    /// A one-line "what just happened", e.g. "saved edits.json".
    note: String,
    /// The name field's buffer, committed on focus loss so one keystroke is not
    /// one undo step.
    name_buffer: String,
    /// Which region `name_buffer` belongs to.
    name_buffer_for: Option<String>,
}

impl EditSession {
    /// Open (or start) the edit session for a bundle directory.
    ///
    /// A missing `edits.json` is the ordinary case and yields an empty document;
    /// a malformed one is reported in the panel and also yields an empty
    /// document, because refusing to open the scene over a sidecar would be a
    /// worse answer than opening it and saying so.
    ///
    /// # Arguments
    /// - `bundle_dir`: the directory holding `bundle.json`.
    /// - `bundle_format`: `bundle.json`'s own `"format"`, cross-checked against
    ///   the sidecar's `bundle_format` (`docs/EDITOR.md` §1).
    ///
    /// `edits` overrides the sidecar's location (the `--edits` flag), so one
    /// bundle can be opened against several edit documents without moving files
    /// around; `Cmd-S` writes back to whichever path was opened.
    #[must_use]
    pub fn open_with_path(bundle_dir: &Path, bundle_format: &str, edits: Option<&Path>) -> Self {
        let path = edits.map_or_else(|| bundle_dir.join(EDITS_FILENAME), Path::to_path_buf);
        let (doc, error) = if path.exists() {
            match EditDocument::load(&path) {
                Ok(doc) => match doc.validate(Some(bundle_format)) {
                    Ok(()) => (doc, None),
                    Err(e) => (EditDocument::new(bundle_format.to_owned()), Some(e)),
                },
                Err(e) => (EditDocument::new(bundle_format.to_owned()), Some(e)),
            }
        } else {
            (EditDocument::new(bundle_format.to_owned()), None)
        };
        let shade_views = match ShadeViews::load(bundle_dir) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("{e}");
                None
            }
        };
        let note = if path.exists() {
            format!("loaded {} regions from {}", doc.regions().len(), path.display())
        } else {
            "no edits.json yet -- Cmd-S writes one next to bundle.json".to_owned()
        };
        Self {
            doc,
            path,
            dirty: false,
            active: false,
            selected: None,
            tool: Tool::default(),
            shade: ShadeUi::new(shade_views, bundle_dir),
            click: ClickUi::default(),
            sam: SamUi::default(),
            brush: BrushUi::default(),
            gizmo: GizmoUi {
                show: true,
                ..GizmoUi::default()
            },
            solo: None,
            counts: Vec::new(),
            rename_for: None,
            rename_buffer: String::new(),
            bundle_dir: bundle_dir.to_path_buf(),
            trippy_root: None,
            has_scene_root: false,
            // The first apply is unconditional: a bundle reopened with a saved
            // `edits.json` must render edited on its very first frame.
            needs_apply: true,
            last_apply_ms: 0.0,
            error,
            note,
            name_buffer: String::new(),
            name_buffer_for: None,
        }
    }

    /// The point ids the render should tint this frame, if any.
    ///
    /// Both selection tools can ask for a highlight; where both do, the union is
    /// tinted the one colour (`edit::apply::PREVIEW_TINT`). A single flat list
    /// keeps `edited_points` doing exactly one tint pass, whichever tool asked.
    fn preview_ids(&self) -> Option<Vec<u32>> {
        let mut ids: Vec<u32> = Vec::new();
        if self.shade.preview {
            ids.extend_from_slice(&self.shade.selection.point_ids);
        }
        if self.click.preview {
            ids.extend_from_slice(&self.click.selection.point_ids);
        }
        if self.sam.preview {
            ids.extend_from_slice(&self.sam.selection);
        }
        (!ids.is_empty()).then_some(ids)
    }

    /// Give the click tool a `max_radius` before anyone has touched a slider.
    ///
    /// `trippy.edit.cluster.default_max_radius_from_bundle`: the bundle's own
    /// median nearest-CAMERA spacing, not point-cloud density. Called once, at
    /// open; a user-moved slider is never overwritten because this only fires
    /// while the radius is still the sentinel `0.0`.
    ///
    /// # Arguments
    /// - `views`: every capture view, for their camera centres.
    /// - `renderer`: only its `xyz`, and only for the single-view fallback.
    /// - `scene_diameter`: world units across the captured area, the last
    ///   resort when neither cameras nor points give any scale.
    pub fn init_click_defaults(
        &mut self,
        views: &[trips_viewer::bundle::BundleView],
        renderer: &Renderer,
        scene_diameter: f32,
    ) {
        if self.click.params.max_radius > 0.0 {
            return;
        }
        let mut centres = Vec::with_capacity(views.len() * 3);
        for view in views {
            let c = view.position();
            centres.extend_from_slice(&[f64::from(c.x), f64::from(c.y), f64::from(c.z)]);
        }
        let radius = if centres.len() / 3 >= 2 {
            cluster::default_max_radius(&centres, &[])
        } else {
            cluster::default_max_radius(&centres, &widen(&renderer.base_points().xyz))
        };
        // A bundle with one camera and one point gives no scale at all; a
        // fraction of the scene is a better answer than a radius of zero, which
        // would make every click select exactly its own seed.
        self.click.params.max_radius = if radius > 0.0 {
            radius
        } else {
            f64::from(scene_diameter * DEFAULT_REGION_SCENE_FRACTION)
        };
    }

    /// Record a Shift-click on the render, in the render camera's own pixels.
    ///
    /// Only stored here: the camera is not known until the frame is laid out,
    /// so [`Self::resolve_click`] is where the work happens.
    pub fn request_click(&mut self, px: (f64, f64)) {
        self.click.pending = Some(px);
        self.tool = Tool::ClickCluster;
    }

    /// Run a pending Shift-click (or a slider move) against `camera`.
    ///
    /// A no-op on every frame where nothing asked for it, exactly as
    /// [`Self::refresh_shade`] is.
    pub fn resolve_click(&mut self, camera: &ClickCamera, renderer: &Renderer) {
        let click = match (self.click.pending.take(), self.click.dirty) {
            (Some(px), _) => {
                self.click.last = Some((camera.clone(), px));
                self.click.last.clone()
            }
            (None, true) => self.click.last.clone(),
            (None, false) => return,
        };
        self.click.dirty = false;
        let Some((camera, px)) = click else {
            return;
        };
        self.run_click(&camera, px, renderer);
    }

    /// Cluster one click and keep the result. The headless `--click` path too.
    ///
    /// # Arguments
    /// - `camera`: the camera the frame was rendered with.
    /// - `px`: the clicked pixel in that camera's own coordinates.
    /// - `renderer`: the source of the cloud (its UNEDITED rows, so the ids the
    ///   selection carries index `points.npz` and not a delete-filtered copy).
    pub fn run_click(&mut self, camera: &ClickCamera, px: (f64, f64), renderer: &Renderer) {
        if self.click.cache.is_none() {
            self.click.cache = Some(ClickCache::build(renderer.base_points()));
        }
        let cache = self.click.cache.as_ref().expect("just built");
        let started = std::time::Instant::now();
        self.click.selection = cluster::click_to_cluster(
            &cache.grid,
            &cache.xyz,
            &cache.rgb,
            camera,
            px,
            &self.click.params,
        );
        self.click.ms = started.elapsed().as_secs_f64() * 1e3;
        self.click.last = Some((camera.clone(), px));
        self.click.note = match &self.click.selection.warning {
            Some(warning) => warning.clone(),
            None => format!(
                "{} points from ({:.0}, {:.0}) in {:.1} ms",
                self.click.selection.point_ids.len(),
                px.0,
                px.1,
                self.click.ms
            ),
        };
        if self.click.preview {
            self.needs_apply = true;
        }
    }

    /// Drop the click selection, its preview and its index.
    pub fn clear_click(&mut self) {
        let had_preview = self.click.preview && !self.click.selection.point_ids.is_empty();
        self.click.selection = ClickSelection::default();
        self.click.last = None;
        self.click.pending = None;
        self.click.dirty = false;
        self.click.cache = None;
        self.click.note = "selection cleared".to_owned();
        if had_preview {
            // The tint has to come off the render, which only happens on an
            // apply -- this is the "Clear restores the frame" half of the
            // screenshot proof in `docs/EDITOR.md` §6.
            self.needs_apply = true;
        }
    }

    /// Show or hide the click selection's tint (the headless `--click` path).
    pub fn set_click_preview(&mut self, on: bool) {
        if self.click.preview != on {
            self.click.preview = on;
            self.needs_apply = true;
        }
    }

    /// The current click selection, for the headless dump and the tests.
    #[must_use]
    pub const fn click_selection(&self) -> &ClickSelection {
        &self.click.selection
    }

    /// Override the click tool's sliders (the `--click-*` flags).
    pub fn set_click_params(&mut self, params: ClickParams) {
        self.click.params = params;
        self.click.dirty = self.click.last.is_some();
    }

    // --- the SAM 3 lift (E5) -------------------------------------------------

    /// Tell the session what `bundle.json` says about where things live.
    ///
    /// Called once at open, from `app.rs` and from the headless path, because
    /// [`Self::open_with_path`] takes a directory rather than a whole
    /// [`trips_viewer::bundle::Manifest`] and the SAM tool is the only thing
    /// that needs more of it than the format string.
    ///
    /// # Arguments
    /// - `trippy_root`: `bundle.json`'s `trippy_root`, if it has one.
    /// - `has_scene_root`: whether it records a `scene_root`.
    pub fn set_bundle_paths(&mut self, trippy_root: Option<&str>, has_scene_root: bool) {
        self.trippy_root = trippy_root.map(str::to_owned);
        self.has_scene_root = has_scene_root;
    }

    /// Whether the SAM tool has focus (so `app.rs` knows to draw a marquee).
    #[must_use]
    pub fn sam_tool_active(&self) -> bool {
        self.active && self.tool == Tool::Sam
    }

    /// Whether a child is running right now.
    #[must_use]
    pub fn sam_running(&self) -> bool {
        self.sam.job.as_ref().is_some_and(|j| j.state().is_running())
    }

    /// Record a gesture made on the render. Consumed by [`Self::resolve_sam`].
    pub fn request_sam(&mut self, gesture: SamGesture) {
        self.sam.pending = Some(gesture);
        self.tool = Tool::Sam;
    }

    /// The view position `app.rs` should snap to, taken once.
    pub fn take_sam_snap(&mut self) -> Option<usize> {
        self.sam.snap_request.take()
    }

    /// The point ids of the last imported SAM region (tests, headless dump).
    #[must_use]
    pub fn sam_selection(&self) -> &[u32] {
        &self.sam.selection
    }

    /// The finished run's summary object, if there is one.
    #[must_use]
    pub const fn sam_summary(&self) -> Option<&serde_json::Value> {
        self.sam.summary.as_ref()
    }

    /// Show or hide the imported region's tint.
    pub fn set_sam_preview(&mut self, on: bool) {
        if self.sam.preview != on {
            self.sam.preview = on;
            self.needs_apply = true;
        }
    }

    /// Override the SAM tool's settings (the `--sam-*` headless flags).
    pub fn set_sam_settings(
        &mut self,
        views_around: usize,
        op: Op,
        mix: f64,
        device: &'static str,
        fake: bool,
    ) {
        self.sam.views_around = views_around;
        self.sam.op = op;
        self.sam.mix = mix;
        self.sam.device = device;
        self.sam.fake = fake;
    }

    /// Turn a pending gesture into a prompt in `view`'s own pixel grid.
    ///
    /// The lift needs a PHOTOGRAPH, so it needs a capture view. When the camera
    /// is not pinned to one the gesture is DISCARDED and the camera snaps to
    /// the nearest capture view instead: the pixels were measured against a
    /// free-flying frame and mean nothing in any photo, so re-using them would
    /// segment the wrong part of the image while looking like it worked.
    ///
    /// # Arguments
    /// - `camera`: this frame's render camera.
    /// - `view`: the view the controller is pinned (or last snapped) to.
    /// - `pinned`: whether the camera really is reproducing `view`.
    /// - `views`: every capture view, for the snap.
    /// - `position`: the camera centre, world units, for the snap.
    pub fn resolve_sam(
        &mut self,
        camera: &brush_pyramid::scene::Camera,
        view: &trips_viewer::bundle::BundleView,
        pinned: bool,
        views: &[trips_viewer::bundle::BundleView],
        position: glam::Vec3,
    ) {
        let Some(gesture) = self.sam.pending.take() else {
            return;
        };
        if !pinned {
            let nearest = nearest_view(views, position);
            self.sam.snap_request = nearest;
            self.sam.note = match nearest.and_then(|i| views.get(i)) {
                Some(v) => format!(
                    "the SAM lift needs a photograph and the camera was not on one -- \
                     snapped to {}; drag the box again",
                    v.name
                ),
                None => "the SAM lift needs a capture view and this bundle has none".to_owned(),
            };
            return;
        }
        let prompt = match gesture {
            SamGesture::Box(a, b) => {
                SamPrompt::Box(sam_geom::view_box_from_render(camera, view, a, b))
            }
            SamGesture::Point(px) => {
                SamPrompt::Point(sam_geom::view_pixel_from_render(camera, view, px))
            }
        };
        self.sam.note = format!("{} on {} -- press run", prompt.label(), view.name);
        self.sam.prompt = Some((view.name.clone(), prompt));
    }

    /// Set the prompt directly, in VIEW pixels (the headless `--sam-box` path).
    pub fn set_sam_prompt(&mut self, view_name: &str, prompt: SamPrompt) {
        self.sam.prompt = Some((view_name.to_owned(), prompt));
        self.tool = Tool::Sam;
    }

    /// Spawn `trippy edits sam` for the current prompt.
    ///
    /// One child, only when asked, and never a second one while the first is
    /// alive (`crate::sam_child`'s invariants).
    ///
    /// # Arguments
    /// - `bundle_dir`: the bundle to segment against.
    ///
    /// # Errors
    /// Returns `Err` when there is no prompt, when the bundle records no
    /// `scene_root`, when no interpreter can be resolved, when the working
    /// directory cannot be made, or when the process will not start. Every one
    /// of those is shown in the panel rather than logged and swallowed.
    pub fn start_sam(&mut self, bundle_dir: &Path) -> Result<(), String> {
        if self.sam_running() {
            return Err("a SAM run is already going; cancel it first".to_owned());
        }
        let Some((view_name, prompt)) = self.sam.prompt.clone() else {
            return Err("drag a box (or Alt-click) on the render first".to_owned());
        };
        if !self.has_scene_root {
            return Err(format!(
                "{} records no `scene_root`, so `trippy edits sam` cannot find the \
                 photographs; re-export the bundle with `trippy export-bundle`",
                bundle_dir.join("bundle.json").display()
            ));
        }
        let interpreter = resolve_interpreter_from_env(self.trippy_root.as_deref())?;
        // A fresh directory per run: the child appends to whatever `--out`
        // names, and reading "the last region" only means "this run's" when the
        // file started empty.
        let work_dir = std::env::temp_dir().join(format!(
            "trips-sam-{}-{}",
            std::process::id(),
            new_region_id()
        ));
        std::fs::create_dir_all(&work_dir).map_err(|e| format!("{}: {e}", work_dir.display()))?;
        let request = SamRequest {
            bundle_dir: bundle_dir.to_path_buf(),
            view_name,
            prompt,
            views_around: self.sam.views_around,
            op: self.sam.op,
            mix: self.sam.mix,
            out: work_dir.join(EDITS_FILENAME),
            device: self.sam.device.to_owned(),
            fake: self.sam.fake,
        };
        self.sam.command = command_line(&interpreter, &request);
        let job = SamJob::spawn(build_command(&interpreter, &request))?;
        self.sam.job = Some(job);
        self.sam.work_dir = Some(work_dir);
        self.sam.summary = None;
        self.sam.note = format!("running {}...", prompt.label());
        Ok(())
    }

    /// Block until the child exits (the headless `--sam-box` path only).
    ///
    /// The window never calls this: blocking the UI thread is exactly what the
    /// Cancel button exists to make unnecessary. A headless run has no Cancel
    /// button and nothing else for the thread to do.
    pub fn wait_for_sam(&mut self) {
        if let Some(job) = self.sam.job.as_mut() {
            job.wait();
        }
    }

    /// The last error the panel would be showing, if any.
    #[must_use]
    pub fn last_error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Undo one step (the headless `--sam-undo` proof; `Cmd-Z` in the window).
    pub fn undo_once(&mut self) {
        self.undo();
    }

    /// Kill the running child, if there is one.
    pub fn cancel_sam(&mut self) {
        if let Some(job) = self.sam.job.as_mut() {
            job.cancel();
            self.sam.note = format!("cancelled after {:.1} s", job.elapsed());
        }
    }

    /// Advance the child's state machine, importing its region when it lands.
    ///
    /// A no-op on every frame with no child, exactly as [`Self::refresh_shade`]
    /// is. Call it once per frame, before [`Self::apply`].
    pub fn poll_sam(&mut self) {
        let Some(job) = self.sam.job.as_mut() else {
            return;
        };
        match job.poll().clone() {
            SamState::Running => {}
            SamState::Finished => {
                let summary = job.summary().cloned();
                let elapsed = job.elapsed();
                let out = self
                    .sam
                    .work_dir
                    .as_ref()
                    .map(|d| d.join(EDITS_FILENAME))
                    .unwrap_or_default();
                self.sam.job = None;
                self.sam.summary = summary;
                match self.import_sam_region(&out, elapsed) {
                    Ok(()) => {}
                    Err(e) => {
                        self.error = Some(e);
                        self.sam.note = "the run finished but its region could not be read".to_owned();
                    }
                }
            }
            SamState::Failed(message) => {
                self.sam.job = None;
                self.error = Some(message);
                self.sam.note = "the SAM run failed -- see the log below".to_owned();
            }
            SamState::Cancelled => {
                self.sam.job = None;
            }
        }
    }

    /// Add the child's region to the document and light it up.
    ///
    /// Through [`EditDocument::add_region`] like every other edit, so Cmd-Z
    /// takes it straight back out (`docs/EDITOR.md` §6's E5 row).
    fn import_sam_region(&mut self, out: &Path, elapsed: f64) -> Result<(), String> {
        let mut region = imported_region(out)?;
        // The child chose its own id in its own throwaway file; a fresh one here
        // keeps ids unique within THIS document, which is what undo keys off.
        region.id = new_region_id();
        let ids = match &region.params {
            Params::Pointset { point_ids } => point_ids.clone(),
            other => {
                return Err(format!(
                    "the SAM child wrote a {} region, not a pointset",
                    other.kind().as_str()
                ))
            }
        };
        let count = ids.len();
        self.sam.selection = ids;
        self.sam.preview = true;
        self.add(region);
        self.needs_apply = true;
        self.sam.note = format!("imported {count} points in {elapsed:.1} s -- Cmd-Z removes it");
        Ok(())
    }

    // --- the brush tool (paint a sparse-voxel region in 3D) ------------------

    /// Give the brush a radius before anyone has touched a slider.
    ///
    /// A fraction of the scene's own diameter, like a new box/sphere: the point
    /// cloud's bounds are meaningless on a TRIPS export (`renderer.rs`'s
    /// `bounds` field). Called once at open; never overwrites a user value,
    /// because it only fires while the radius is still the sentinel `0.0`.
    pub fn init_brush_defaults(&mut self, scene_diameter: f32) {
        if self.brush.radius > 0.0 {
            return;
        }
        self.brush.radius = f64::from(scene_diameter * DEFAULT_BRUSH_SCENE_FRACTION).max(1e-6);
    }

    /// Whether the Brush tool has focus (so `app.rs` knows the drag is a stroke).
    #[must_use]
    pub fn brush_tool_active(&self) -> bool {
        self.active && self.tool == Tool::Brush
    }

    /// Where to draw the brush cursor: `(render pixel, radius in render px)`.
    #[must_use]
    pub const fn brush_cursor(&self) -> Option<((f64, f64), f64)> {
        self.brush.cursor
    }

    /// The brush radius, world units (the headless dump and the tests).
    #[must_use]
    pub const fn brush_radius(&self) -> f64 {
        self.brush.radius
    }

    /// Override the brush's settings (the `--brush-*` headless flags).
    pub fn set_brush_settings(&mut self, radius: f64, weight: f64, op: Op, mix: f64) {
        self.brush.radius = radius;
        self.brush.weight = weight;
        self.brush.op = op;
        self.brush.mix = mix;
        self.tool = Tool::Brush;
    }

    /// Start a stroke. `erasing` is the Alt-drag.
    ///
    /// The region is chosen here, once: an existing brush region if the tool
    /// has one, otherwise a new one, created on a voxel grid fixed by the
    /// CURRENT radius. Changing the radius later paints bigger or smaller
    /// spheres into the same grid; it never re-grids a region, because the
    /// cells already painted were measured on the old one.
    pub fn begin_brush_stroke(&mut self, erasing: bool) {
        if self.brush.stroke.is_some() {
            return;
        }
        let existing = self
            .brush
            .region
            .as_ref()
            .and_then(|id| self.doc.region(id))
            .and_then(|r| match &r.params {
                Params::Brush {
                    origin,
                    cell_size,
                    cells,
                } => Some((r.id.clone(), *origin, *cell_size, cells.clone())),
                _ => None,
            });
        self.brush.stroke = Some(match existing {
            Some((region_id, origin, cell_size, cells)) => Stroke {
                region_id,
                // Unused on this branch: the region already has a name.
                name: String::new(),
                origin,
                cell_size,
                cells,
                erasing,
                creating: false,
                emitted: false,
                last_px: None,
                samples: 0,
            },
            None => Stroke {
                region_id: new_region_id(),
                name: auto_name_for(&self.doc, "brush", None),
                // The grid is anchored at the world origin rather than at the
                // first dab, so two regions painted in the same scene share a
                // lattice and a hand-merged `cells` list still means something.
                origin: [0.0, 0.0, 0.0],
                cell_size: (self.brush.radius / BRUSH_CELLS_PER_RADIUS).max(1e-9),
                cells: BrushCells::new(),
                erasing,
                creating: true,
                emitted: false,
                last_px: None,
                samples: 0,
            },
        });
    }

    /// Record one sample of the stroke, in the render camera's own pixels.
    ///
    /// Dropped when it is within `BRUSH_SAMPLE_PX` of the previous one: a
    /// pointer resting still would otherwise paint the same sphere every frame,
    /// and each sample costs one depth-anchor pass over the cloud.
    pub fn brush_sample(&mut self, px: (f64, f64)) {
        let Some(stroke) = self.brush.stroke.as_mut() else {
            return;
        };
        if let Some(last) = stroke.last_px {
            if (px.0 - last.0).hypot(px.1 - last.1) < BRUSH_SAMPLE_PX {
                return;
            }
        }
        stroke.last_px = Some(px);
        self.brush.pending.push(px);
    }

    /// Finish the stroke: the next drag starts a new undo step.
    pub fn end_brush_stroke(&mut self) {
        let Some(stroke) = self.brush.stroke.take() else {
            return;
        };
        if stroke.emitted {
            self.brush.region = Some(stroke.region_id);
            self.brush.note = format!(
                "{} stroke: {} samples, {} cells",
                if stroke.erasing { "erase" } else { "paint" },
                stroke.samples,
                stroke.cells.len()
            );
        }
        self.brush.cursor = None;
    }

    /// Whether a stroke is in progress (so `app.rs` keeps feeding it pixels).
    #[must_use]
    pub const fn brush_stroking(&self) -> bool {
        self.brush.stroke.is_some()
    }

    /// Paint every pending sample against this frame's camera.
    ///
    /// The two-step every gesture in this viewer uses: `app.rs` records pixels
    /// while the pointer is in egui's coordinates, and the work happens here,
    /// once the frame's camera exists (`docs/EDITOR.md` §4's last paragraph).
    /// A no-op on every frame with nothing pending.
    pub fn resolve_brush(&mut self, camera: &ClickCamera, renderer: &Renderer) {
        if self.brush.pending.is_empty() || self.brush.stroke.is_none() {
            // A pending sample with no stroke cannot happen through either
            // caller, and dropping it silently would be the wrong answer if it
            // ever could: leave it for the frame that has a stroke.
            return;
        }
        let samples: Vec<(f64, f64)> = std::mem::take(&mut self.brush.pending);
        if self.brush.cache.is_none() {
            self.brush.cache = Some(widen(&renderer.base_points().xyz));
        }
        let cache = self.brush.cache.as_ref().expect("just built");
        let radius = self.brush.radius;
        let weight = self.brush.weight;

        // Rebuild the screen-space anchor index only when the camera actually
        // moved (`ClickCamera` is `PartialEq`): a still camera during a drag,
        // or a second stroke from the same viewpoint, reuses it instead of
        // re-scanning the whole cloud (`brush::ScreenGrid`'s own doc comment).
        let stale = self
            .brush
            .screen_grid
            .as_ref()
            .is_none_or(|(built_with, _)| built_with != camera);
        if stale {
            self.brush.screen_grid =
                Some((camera.clone(), brush::ScreenGrid::build(camera, cache, brush::ANCHOR_RADIUS_PX)));
        }
        let grid = &self.brush.screen_grid.as_ref().expect("just built").1;

        // Every sample becomes a world centre first, so one stroke is ONE
        // `paint_along` over the path rather than N separate spheres — the same
        // shape the Python's own `paint_along` produces for a dragged stroke.
        let mut path: Vec<[f64; 3]> = Vec::with_capacity(samples.len());
        let mut last_pixel = None;
        for px in samples {
            let depth = grid.nearest(px, brush::ANCHOR_RADIUS_PX).or(self.brush.depth);
            let Some(depth) = depth else {
                // Nothing under the cursor and no earlier anchor: painting at an
                // invented depth would put cells somewhere Jordan cannot see.
                self.brush.note =
                    "no point under the cursor yet -- aim at the scene to anchor the stroke"
                        .to_owned();
                continue;
            };
            self.brush.depth = Some(depth);
            path.push(camera.unproject(px, depth));
            last_pixel = Some((px, depth));
        }
        if let Some((px, depth)) = last_pixel {
            // The stroke's own radius, in this frame's pixels, for the cursor
            // ring: `fx * r / z` is the projection of a sphere at that depth.
            self.brush.cursor = Some((px, camera.fx * radius / depth));
        }
        if path.is_empty() {
            return;
        }

        let Some(stroke) = self.brush.stroke.as_mut() else {
            return;
        };
        let (origin, cell_size, erasing) = (stroke.origin, stroke.cell_size, stroke.erasing);
        let result = if erasing {
            path.iter()
                .try_fold(0_usize, |n, centre| {
                    brush::erase(&mut stroke.cells, origin, cell_size, *centre, radius)
                        .map(|dropped| n + dropped)
                })
                .map(|_| ())
        } else {
            brush::paint_along(&mut stroke.cells, origin, cell_size, &path, radius, weight)
                .map(|_| ())
        };
        if let Err(e) = result {
            self.error = Some(e);
            return;
        }
        stroke.samples += path.len();
        self.commit_stroke();
    }

    /// Write the stroke so far into the document, as ONE undo entry.
    ///
    /// The first write of a stroke appends; every later write of the SAME
    /// stroke replaces it (`EditDocument::add_region_coalesced` /
    /// `update_region_coalesced`), so a stroke that took 200 frames is one
    /// `Cmd-Z` and `edits.json` carries one entry for it.
    fn commit_stroke(&mut self) {
        let Some(stroke) = self.brush.stroke.as_ref() else {
            return;
        };
        let params = Params::Brush {
            origin: stroke.origin,
            cell_size: stroke.cell_size,
            cells: stroke.cells.clone(),
        };
        let (region_id, creating, emitted) =
            (stroke.region_id.clone(), stroke.creating, stroke.emitted);
        let name = stroke.name.clone();
        let result = if creating {
            let region = Region::new(region_id.clone(), name, params, self.brush.mix, self.brush.op)
                .with_source(
                    "brush",
                    json!({
                        "radius": self.brush.radius,
                        "weight": self.brush.weight,
                        "cell_size": stroke.cell_size,
                    }),
                );
            self.doc.add_region_coalesced(&region, None, emitted)
        } else {
            self.doc
                .update_region_coalesced(&region_id, json!({ "params": params.to_json() }), emitted)
        };
        let ok = result.is_ok();
        self.record(result);
        if ok {
            if let Some(stroke) = self.brush.stroke.as_mut() {
                stroke.emitted = true;
            }
            self.selected = Some(region_id.clone());
            self.brush.region = Some(region_id);
        }
    }

    /// Forget the tool's current region, so the next stroke starts a new one.
    pub fn new_brush_region(&mut self) {
        self.brush.region = None;
        self.brush.note = "the next stroke starts a new region".to_owned();
    }

    // --- the 3D drag gizmos --------------------------------------------------

    /// Project the selected region's handles with this frame's camera.
    ///
    /// Called once per frame from `app.rs`, right after the camera is built and
    /// before the render, so the handles `app.rs` paints and the handles a drag
    /// hit-tests against are the same ones.
    pub fn update_gizmo(&mut self, camera: &ClickCamera) {
        self.gizmo.screen = if self.gizmo_available() {
            self.selected
                .as_ref()
                .and_then(|id| self.doc.region(id))
                .and_then(|region| GizmoScreen::project(camera, &region.params))
        } else {
            None
        };
    }

    /// Whether a gizmo drag is possible at all right now.
    ///
    /// Not while the SAM or Brush tool has focus: those two have taken the
    /// primary drag for their own gesture, and a handle under the pointer must
    /// not silently steal a brush stroke.
    #[must_use]
    pub fn gizmo_available(&self) -> bool {
        self.active && self.gizmo.show && !matches!(self.tool, Tool::Sam | Tool::Brush)
    }

    /// This frame's handles, for the egui painter (`app.rs`).
    #[must_use]
    pub const fn gizmo_screen(&self) -> Option<&GizmoScreen> {
        self.gizmo.screen.as_ref()
    }

    /// Whether a gizmo drag is in progress (so `app.rs` does not orbit).
    #[must_use]
    pub const fn gizmo_dragging(&self) -> bool {
        self.gizmo.drag.is_some()
    }

    /// Try to grab a handle at `px`. `true` when the drag belongs to the gizmo.
    ///
    /// A drag that grabs nothing returns `false` and `app.rs` orbits with it,
    /// which is what keeps navigation intact: the gizmo takes the drag only
    /// where a handle is (`docs/EDITOR.md` §4, "a drag is scoped to what it
    /// started on").
    pub fn begin_gizmo_drag(&mut self, px: (f64, f64), shift: bool, ctrl: bool) -> bool {
        if !self.gizmo_available() || self.gizmo.drag.is_some() {
            return false;
        }
        let Some(screen) = self.gizmo.screen.clone() else {
            return false;
        };
        let Some(axis) = screen.pick(px) else {
            return false;
        };
        let Some(region) = self
            .selected
            .as_ref()
            .and_then(|id| self.doc.region(id))
            .cloned()
        else {
            return false;
        };
        // The normal handle only ever tilts, whatever modifier is held: there
        // is no world axis to resize or rotate about, only a direction to
        // drag — Shift/Ctrl on this ONE handle would be a surprise, not a
        // feature, so they are ignored rather than refused.
        let kind = if axis == gizmo::NORMAL_AXIS {
            Drag::Tilt
        } else {
            match (shift, ctrl) {
                (true, _) => Drag::Resize,
                (false, true) => Drag::Rotate,
                (false, false) => Drag::Translate,
            }
        };
        if kind == Drag::Rotate && !matches!(region.params, Params::Box { .. }) {
            self.note = "only a box has an orientation to rotate".to_owned();
            return false;
        }
        self.gizmo.drag = Some(GizmoDrag {
            region_id: region.id,
            axis,
            kind,
            base: region.params,
            screen,
            start_px: px,
            emitted: false,
        });
        true
    }

    /// Apply the drag at `px`, recomputed from the gesture's own starting state.
    ///
    /// Absolute rather than incremental: the region is always
    /// `f(base, pointer - start)`, so a slow drag and a fast one over the same
    /// path end in the same place and nothing accumulates rounding error.
    pub fn gizmo_drag_to(&mut self, px: (f64, f64)) {
        let Some(drag) = self.gizmo.drag.as_ref() else {
            return;
        };
        let total = (px.0 - drag.start_px.0, px.1 - drag.start_px.1);
        if total.0.hypot(total.1) < gizmo::MIN_DRAG_PX {
            return;
        }
        let params = match drag.kind {
            Drag::Translate => {
                let along = drag.screen.translate_world(drag.axis, total);
                let mut delta = [0.0_f64; 3];
                delta[drag.axis] = along;
                gizmo::translated(&drag.base, delta)
            }
            Drag::Resize => {
                gizmo::resized(&drag.base, drag.screen.resize_factor(drag.axis, total))
            }
            Drag::Rotate => gizmo::rotated(
                &drag.base,
                drag.axis,
                drag.screen.rotate_angle(drag.axis, drag.start_px, px),
            ),
            Drag::Tilt => {
                let Params::Lid(lid) = &drag.base else {
                    return;
                };
                drag.screen
                    .tilted_up(lid.up, total)
                    .and_then(|up| gizmo::tilted(&drag.base, up))
            }
        };
        let Some(params) = params else {
            return;
        };
        let (id, emitted) = (drag.region_id.clone(), drag.emitted);
        let result =
            self.doc
                .update_region_coalesced(&id, json!({ "params": params.to_json() }), emitted);
        let ok = result.is_ok();
        self.record(result);
        if ok {
            if let Some(drag) = self.gizmo.drag.as_mut() {
                drag.emitted = true;
            }
            self.note = match self.gizmo.drag.as_ref().map(|d| d.kind) {
                Some(Drag::Translate) => format!("moved along {}", axis_name(self.gizmo_axis())),
                Some(Drag::Resize) => format!("resized on {}", axis_name(self.gizmo_axis())),
                Some(Drag::Rotate) => format!("rotated about {}", axis_name(self.gizmo_axis())),
                Some(Drag::Tilt) => "tilted the lid's normal".to_owned(),
                None => String::new(),
            };
        }
    }

    /// Which axis the current drag grabbed, for the note.
    fn gizmo_axis(&self) -> usize {
        self.gizmo.drag.as_ref().map_or(0, |d| d.axis)
    }

    /// Finish the gizmo drag: the next one is a new undo step.
    pub fn end_gizmo_drag(&mut self) {
        self.gizmo.drag = None;
    }

    /// Move the selected region along `axis` by `delta` world units, as ONE
    /// undo entry (the headless `--move-region` proof).
    ///
    /// The same `gizmo::translated` a drag goes through, so the headless twin
    /// exercises the shipped path and not a parallel one.
    pub fn move_selected_region(&mut self, delta: [f64; 3]) -> bool {
        self.nudge_selected(delta);
        self.error.is_none()
    }

    // --- the Named Objects panel --------------------------------------------

    /// Select a region by id, or by name when no id matches. `false` when
    /// neither does.
    ///
    /// The headless twin of clicking a row in the Named Objects panel; names
    /// are accepted because `--move-region brush-1` is what a proof script has
    /// (`docs/EDITOR.md` §1's auto names are stable, ids are random hex).
    pub fn select_region(&mut self, id_or_name: &str) -> bool {
        let found = self
            .doc
            .regions()
            .iter()
            .find(|r| r.id == id_or_name)
            .or_else(|| self.doc.regions().iter().find(|r| r.name == id_or_name))
            .map(|r| r.id.clone());
        match found {
            Some(id) => {
                self.selected = Some(id);
                true
            }
            None => false,
        }
    }

    /// The selected region's id, if any.
    #[must_use]
    pub fn selected_id(&self) -> Option<&str> {
        self.selected.as_deref()
    }

    /// Show only `id`'s effect, or `None` for all of them.
    pub fn set_solo(&mut self, id: Option<&str>) {
        let next = id.map(ToOwned::to_owned);
        if self.solo != next {
            self.solo = next;
            self.needs_apply = true;
        }
    }

    /// The soloed region, if any.
    #[must_use]
    pub fn solo(&self) -> Option<&str> {
        self.solo.as_deref()
    }

    /// How many points a region claimed at the last apply.
    #[must_use]
    fn count_for(&self, id: &str) -> Option<usize> {
        self.counts.iter().find(|c| c.id == id).map(|c| c.count)
    }

    /// Recompose the per-point weights and hand the renderer its point sets.
    ///
    /// A no-op unless a widget or a key asked for it. This is the one place
    /// `O(points x regions)` work happens; everything else is per frame and
    /// costs nothing.
    ///
    /// # Errors
    /// Returns `Err` when the composition does not fit the cloud, or an upload
    /// fails. The caller shows it; the previous frame's edit stays applied.
    pub fn apply(&mut self, renderer: &mut Renderer) -> Result<(), String> {
        if !self.needs_apply {
            return Ok(());
        }
        self.needs_apply = false;
        let started = std::time::Instant::now();

        let preview_ids = self.preview_ids();

        // Nothing enabled and no preview: clear the edit and return WITHOUT
        // composing. `edited_points` would otherwise clone the whole point set
        // (tens of MB on a real scene) only to be told it changed nothing, and
        // this is the path every unedited bundle takes on every open.
        if preview_ids.is_none() && !self.doc.regions().iter().any(|r| r.enabled) {
            renderer.set_edits(None);
            self.counts.clear();
            self.last_apply_ms = started.elapsed().as_secs_f64() * 1e3;
            return Ok(());
        }
        let solo = self.solo.clone();

        // The borrow of `renderer` ends with this block; `edited` owns its data.
        let (edited, gaussian_needed) = {
            let base = renderer.base_points();
            let composed = compose_trips_weights_solo(&self.doc, &widen(&base.xyz), solo.as_deref());
            let gaussian_needed = composed.delete_mask.iter().any(|d| *d);
            // The Named Objects panel's count column, observed on the way
            // through rather than recomputed by a second O(points x regions) pass.
            self.counts.clone_from(&composed.per_region);
            let edited = match preview_ids.as_deref() {
                Some(ids) => edited_points(&tinted_points(base, ids), &composed)?,
                None => edited_points(base, &composed)?,
            };
            (edited, gaussian_needed)
        };
        let identity = edited.num_deleted == 0 && edited.probe.is_none() && preview_ids.is_none();
        renderer.set_edits((!identity).then_some(edited));

        // The Gaussian half. Skipped entirely unless a `delete` region is
        // enabled: `blend`/`fade` never touch a Gaussian's opacity (see
        // `edit::apply::gaussian_opacity_scale`), and the means readback on a
        // multi-million-Gaussian ply is not free.
        #[cfg(not(target_family = "wasm"))]
        {
            let scale = if renderer.live_splat().is_some() {
                let means = {
                    let splat = renderer.live_splat().expect("checked above");
                    block_on(splat.means_host())?
                };
                if gaussian_needed {
                    let composed =
                        compose_gaussian_weights_solo(&self.doc, &widen(&means), solo.as_deref());
                    Some(gaussian_opacity_scale(&composed))
                } else {
                    Some(None)
                }
            } else {
                None
            };
            if let Some(scale) = scale {
                if let Some(splat) = renderer.live_splat_mut() {
                    match scale {
                        Some(values) => splat.set_opacity_scale(&values)?,
                        None => splat.clear_opacity_scale(),
                    }
                }
            }
        }
        #[cfg(target_family = "wasm")]
        let _ = gaussian_needed;

        self.last_apply_ms = started.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }

    /// Write `edits.json` next to `bundle.json`, undo history included.
    fn save(&mut self) {
        match self.doc.save(&self.path) {
            Ok(()) => {
                self.dirty = false;
                self.error = None;
                self.note = format!("saved {}", self.path.display());
            }
            Err(e) => self.error = Some(e),
        }
    }

    /// Write the document to an EXPLICIT path, sidecar externalisation
    /// included — `main.rs`'s `--save-edits`, the headless twin of [`Self::save`]
    /// for a script that wants the result somewhere other than this session's
    /// own `self.path`.
    ///
    /// # Errors
    /// Returns `Err` when `EditDocument::save` does (the directory cannot be
    /// created, a brush sidecar cannot be written, or the write itself fails).
    pub fn save_as(&mut self, path: &std::path::Path) -> Result<(), String> {
        self.doc.save(path)
    }

    /// Re-read `edits.json` from disk, discarding unsaved changes.
    fn reload(&mut self) {
        if !self.path.exists() {
            self.note = format!("{} does not exist yet", self.path.display());
            return;
        }
        match EditDocument::load(&self.path) {
            Ok(doc) => {
                self.selected = None;
                self.doc = doc;
                self.dirty = false;
                self.error = None;
                self.needs_apply = true;
                self.note = format!("reloaded {} regions", self.doc.regions().len());
            }
            Err(e) => self.error = Some(e),
        }
    }

    /// Record a document mutation's result, marking the session dirty.
    fn record(&mut self, result: Result<(), String>) {
        match result {
            Ok(()) => {
                self.dirty = true;
                self.needs_apply = true;
                self.error = None;
            }
            Err(e) => self.error = Some(e),
        }
    }

    /// Add a region and select it.
    fn add(&mut self, region: Region) {
        let id = region.id.clone();
        let result = self.doc.add_region(&region, None);
        let ok = result.is_ok();
        self.record(result);
        if ok {
            self.note = format!("added {}", region.name);
            self.selected = Some(id);
        }
    }

    /// The selected region's index in paint order, if any.
    fn selected_index(&self) -> Option<usize> {
        let id = self.selected.as_ref()?;
        self.doc.regions().iter().position(|r| &r.id == id)
    }

    /// Move the selected region up (`-1`) or down (`+1`) the paint order.
    fn move_selected(&mut self, delta: isize) {
        let Some(index) = self.selected_index() else {
            return;
        };
        let target = index as isize + delta;
        if target < 0 || target as usize >= self.doc.regions().len() {
            return;
        }
        let mut order: Vec<String> = self.doc.order().to_vec();
        order.swap(index, target as usize);
        let result = self.doc.reorder(order);
        self.record(result);
    }

    /// Delete the selected region.
    fn delete_selected(&mut self) {
        let Some(id) = self.selected.clone() else {
            return;
        };
        let result = self.doc.remove_region(&id);
        self.record(result);
        self.selected = None;
    }

    /// Move the selected region's centre by `delta` world units.
    ///
    /// The keyboard nudge and the translate gizmo share
    /// [`gizmo::translated`], so an arrow key and a drag can never disagree
    /// about what "move" means. A `pointset` has no centre and a `brush`'s
    /// cells are measured on its own grid, so neither moves.
    fn nudge_selected(&mut self, delta: [f64; 3]) {
        let Some(region) = self
            .selected
            .as_ref()
            .and_then(|id| self.doc.region(id))
            .cloned()
        else {
            return;
        };
        let Some(moved) = gizmo::translated(&region.params, delta) else {
            return;
        };
        self.set_params(&region.id, &moved);
    }

    /// Scale the selected region's size by `factor` (shares the resize gizmo's maths).
    fn resize_selected(&mut self, factor: f64) {
        let Some(region) = self
            .selected
            .as_ref()
            .and_then(|id| self.doc.region(id))
            .cloned()
        else {
            return;
        };
        let Some(resized) = gizmo::resized(&region.params, factor) else {
            return;
        };
        self.set_params(&region.id, &resized);
    }

    /// Push a mix change through the undo log, coalescing a slider drag.
    ///
    /// A slider emits a change per FRAME while it is dragged, and one undo step
    /// per frame of a drag is not an undo history anyone can use. Consecutive
    /// mix changes to the same region therefore collapse into one entry (the
    /// same `update_region_coalesced` the gizmo and the brush use); any other
    /// edit in between ends the run, so `Cmd-Z` after a fiddle returns the mix
    /// to what it was before the fiddle started.
    fn set_mix(&mut self, id: &str, mix: f64) {
        let coalesce = self.doc.log.get(self.doc.cursor.wrapping_sub(1)).is_some_and(|e| {
            self.doc.cursor == self.doc.log.len()
                && e.get("type").and_then(serde_json::Value::as_str) == Some("update_region")
                && e.get("id").and_then(serde_json::Value::as_str) == Some(id)
                && e.pointer("/changes/mix").is_some()
        });
        let result = self
            .doc
            .update_region_coalesced(id, json!({ "mix": mix }), coalesce);
        self.record(result);
    }

    /// Push a params change through the undo log.
    fn set_params(&mut self, id: &str, params: &Params) {
        let result = self
            .doc
            .update_region(id, json!({ "params": params.to_json() }));
        self.record(result);
    }

    /// Undo one entry, if there is one.
    fn undo(&mut self) {
        if self.doc.undo() {
            self.dirty = true;
            self.needs_apply = true;
            self.note = format!("undo ({}/{})", self.doc.cursor, self.doc.log.len());
            if self.selected_index().is_none() {
                self.selected = None;
            }
        }
    }

    /// Redo one entry, if there is one.
    fn redo(&mut self) {
        if self.doc.redo() {
            self.dirty = true;
            self.needs_apply = true;
            self.note = format!("redo ({}/{})", self.doc.cursor, self.doc.log.len());
        }
    }

    /// Consume this frame's keyboard, when edit mode is on.
    ///
    /// `M` is handled by the caller so it works with the panels hidden.
    ///
    /// # Arguments
    /// - `ctx`: egui's context, for `input`.
    /// - `scene_diameter`: world units across the captured area, the unit both
    ///   the nudge step and the default region size are quoted in.
    pub fn handle_keys(&mut self, ctx: &egui::Context, scene_diameter: f32) {
        if !self.active || ctx.text_edit_focused() {
            return;
        }
        let step = f64::from(scene_diameter * NUDGE_SCENE_FRACTION);
        let (mut undo, mut redo, mut save) = (false, false, false);
        let mut nudge = [0.0_f64; 3];
        let (mut resize, mut cycle_tool, mut toggle_preview, mut remove) = (0.0, false, false, false);
        ctx.input(|i| {
            let command = i.modifiers.command;
            if command && i.key_pressed(egui::Key::Z) {
                if i.modifiers.shift {
                    redo = true;
                } else {
                    undo = true;
                }
            }
            if command && i.key_pressed(egui::Key::S) {
                save = true;
            }
            if i.key_pressed(egui::Key::T) {
                cycle_tool = true;
            }
            if i.key_pressed(egui::Key::H) {
                toggle_preview = true;
            }
            if i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace) {
                remove = true;
            }
            if i.key_pressed(egui::Key::OpenBracket) {
                resize = 1.0 / f64::from(RESIZE_STEP);
            }
            if i.key_pressed(egui::Key::CloseBracket) {
                resize = f64::from(RESIZE_STEP);
            }
            // Arrows move in the world's X/Z plane and PageUp/PageDown in Y;
            // a screen-relative nudge would need the gizmo this milestone does
            // not ship, and a world-axis one is at least unambiguous.
            nudge[0] += f64::from(i.key_pressed(egui::Key::ArrowRight)) * step;
            nudge[0] -= f64::from(i.key_pressed(egui::Key::ArrowLeft)) * step;
            nudge[2] += f64::from(i.key_pressed(egui::Key::ArrowUp)) * step;
            nudge[2] -= f64::from(i.key_pressed(egui::Key::ArrowDown)) * step;
            nudge[1] += f64::from(i.key_pressed(egui::Key::PageDown)) * step;
            nudge[1] -= f64::from(i.key_pressed(egui::Key::PageUp)) * step;
        });
        if undo {
            self.undo();
        }
        if redo {
            self.redo();
        }
        if save {
            self.save();
        }
        if cycle_tool {
            self.tool = self.tool.next();
            if self.tool != Tool::ClickCluster {
                // Same rule the Tools panel's own radio applies: the click
                // tool's `f64` copy of the cloud is not held for a tool nobody
                // is using. The selection itself survives, so cycling back and
                // pressing a slider rebuilds the index and re-runs the click.
                self.click.cache = None;
            }
        }
        if toggle_preview {
            // `H` belongs to whichever tool has focus: both produce a tinted
            // `pointset` preview and there is no second highlight colour to
            // tell two of them apart with.
            match self.tool {
                Tool::ClickCluster => self.click.preview = !self.click.preview,
                Tool::Sam => self.sam.preview = !self.sam.preview,
                // The brush has no preview of its own -- what a stroke paints IS
                // the region, and the region is already in the frame -- so `H`
                // keeps the one meaning it has everywhere else. The 3D handles
                // have their own checkbox in the Regions panel.
                Tool::Regions | Tool::ShadeFinder | Tool::Brush => {
                    self.shade.preview = !self.shade.preview;
                }
            }
            self.needs_apply = true;
        }
        if remove {
            self.delete_selected();
        }
        if resize > 0.0 {
            if self.tool == Tool::Brush {
                // While painting there is no selected region to resize, and a
                // brush without a radius key is unusable (`docs/EDITOR.md` §4).
                let step = if resize > 1.0 {
                    BRUSH_RADIUS_STEP
                } else {
                    1.0 / BRUSH_RADIUS_STEP
                };
                self.brush.radius = (self.brush.radius * step).max(1e-9);
                self.brush.note = format!("brush radius {:.4} world units", self.brush.radius);
            } else {
                self.resize_selected(resize);
            }
        }
        if nudge.iter().any(|d| *d != 0.0) {
            self.nudge_selected(nudge);
        }
    }

    /// Draw the Regions / Inspector / Tools panels.
    ///
    /// # Arguments
    /// - `ui`: the Edit window's `Ui`.
    /// - `look_at`: the camera's current look-at point — where a new region is
    ///   born (`docs/EDITOR.md` §4).
    /// - `scene_diameter`: world units across the captured area.
    /// - `num_points`: the cloud's size, for the Tools panel's readout.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        look_at: glam::Vec3,
        scene_diameter: f32,
        num_points: usize,
    ) {
        self.regions_panel(ui, look_at, scene_diameter);
        ui.separator();
        self.named_objects_panel(ui);
        ui.separator();
        self.inspector_panel(ui, scene_diameter);
        ui.separator();
        self.tools_panel(ui, num_points);

        ui.separator();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(self.doc.can_undo(), egui::Button::new("undo"))
                .clicked()
            {
                self.undo();
            }
            if ui
                .add_enabled(self.doc.can_redo(), egui::Button::new("redo"))
                .clicked()
            {
                self.redo();
            }
            if ui.button("save").clicked() {
                self.save();
            }
            if ui.button("reload").clicked() {
                self.reload();
            }
        });
        ui.label(format!(
            "{}{}  |  history {}/{}  |  weights recomputed in {:.1} ms",
            self.path.display(),
            if self.dirty { " (unsaved)" } else { "" },
            self.doc.cursor,
            self.doc.log.len(),
            self.last_apply_ms
        ));
        if !self.note.is_empty() {
            ui.label(&self.note);
        }
        if let Some(error) = &self.error {
            ui.colored_label(egui::Color32::from_rgb(255, 120, 120), error);
        }
        ui.label(KEYS_HELP);
    }

    /// The Regions list: enable, select, reorder, delete, and the three "+" buttons.
    fn regions_panel(&mut self, ui: &mut egui::Ui, look_at: glam::Vec3, scene_diameter: f32) {
        ui.label(egui::RichText::new("Regions (paint order: later wins)").strong());
        let mut toggled: Option<(String, bool)> = None;
        let mut clicked: Option<String> = None;
        egui::ScrollArea::vertical()
            .max_height(160.0)
            .show(ui, |ui| {
                for region in self.doc.regions() {
                    ui.horizontal(|ui| {
                        let mut enabled = region.enabled;
                        if ui.checkbox(&mut enabled, "").changed() {
                            toggled = Some((region.id.clone(), enabled));
                        }
                        let selected = self.selected.as_deref() == Some(region.id.as_str());
                        let label = format!(
                            "{}  [{} {}{}]",
                            if region.name.is_empty() {
                                region.id.as_str()
                            } else {
                                region.name.as_str()
                            },
                            region.short_kind(),
                            region.op.as_str(),
                            if region.op == Op::Delete {
                                String::new()
                            } else {
                                format!(" {:.2}", region.mix)
                            }
                        );
                        if ui.selectable_label(selected, label).clicked() {
                            clicked = Some(region.id.clone());
                        }
                    });
                }
                if self.doc.regions().is_empty() {
                    ui.label("no regions yet -- add one with the buttons below");
                }
            });
        if let Some((id, enabled)) = toggled {
            let result = self.doc.update_region(&id, json!({ "enabled": enabled }));
            self.record(result);
        }
        if let Some(id) = clicked {
            self.selected = Some(id);
        }

        let half = f64::from(scene_diameter * DEFAULT_REGION_SCENE_FRACTION);
        let centre = [
            f64::from(look_at.x),
            f64::from(look_at.y),
            f64::from(look_at.z),
        ];
        ui.horizontal(|ui| {
            if ui.button("+ box").clicked() {
                self.add(Region::new(
                    new_region_id(),
                    "box".to_owned(),
                    Params::Box {
                        center: centre,
                        half_extents: [half, half, half],
                        quat: [1.0, 0.0, 0.0, 0.0],
                    },
                    0.0,
                    Op::Blend,
                ));
            }
            if ui.button("+ sphere").clicked() {
                self.add(Region::new(
                    new_region_id(),
                    "sphere".to_owned(),
                    Params::Sphere {
                        center: centre,
                        radius: half,
                    },
                    0.0,
                    Op::Blend,
                ));
            }
            if ui
                .button("+ lid")
                .on_hover_text(
                    "seeded with the Karekare pool plane already fitted in SURFACE_LID.md",
                )
                .clicked()
            {
                self.add(Region::new(
                    new_region_id(),
                    "pool lid".to_owned(),
                    Params::Lid(KAREKARE_LID),
                    0.0,
                    Op::Delete,
                ));
            }
            if ui
                .add_enabled(self.selected.is_some(), egui::Button::new("^"))
                .on_hover_text("move earlier in the paint order")
                .clicked()
            {
                self.move_selected(-1);
            }
            if ui
                .add_enabled(self.selected.is_some(), egui::Button::new("v"))
                .on_hover_text("move later in the paint order")
                .clicked()
            {
                self.move_selected(1);
            }
        });
        ui.label(format!(
            "new regions are born at the look-at point ({:.2}, {:.2}, {:.2}), {half:.2} u across",
            look_at.x, look_at.y, look_at.z
        ));
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.gizmo.show, "3D handles")
                .on_hover_text(
                    "drag a handle to move the selected region along that axis; SHIFT-drag \
                     resizes, CTRL-drag rotates a box. A drag anywhere else still orbits.",
                );
            if self.gizmo.show && self.gizmo.screen.is_none() && self.selected.is_some() {
                ui.label("(no handles: this kind has no shape, or it is behind the camera)");
            }
        });
    }

    /// The Inspector: name, op, mix, and the selected region's own numbers.
    #[allow(clippy::too_many_lines)]
    fn inspector_panel(&mut self, ui: &mut egui::Ui, scene_diameter: f32) {
        let Some(region) = self
            .selected
            .as_ref()
            .and_then(|id| self.doc.region(id))
            .cloned()
        else {
            ui.label(egui::RichText::new("Inspector").strong());
            ui.label("select a region above");
            return;
        };
        ui.label(egui::RichText::new(format!("Inspector -- {}", region.kind.as_str())).strong());

        // The name buffer is committed on focus loss, so typing a name is one
        // undo step rather than one per keystroke.
        if self.name_buffer_for.as_deref() != Some(region.id.as_str()) {
            self.name_buffer.clone_from(&region.name);
            self.name_buffer_for = Some(region.id.clone());
        }
        let response = ui.add(
            egui::TextEdit::singleline(&mut self.name_buffer)
                .hint_text("name")
                .desired_width(220.0),
        );
        if response.lost_focus() && self.name_buffer != region.name {
            let result = self
                .doc
                .update_region(&region.id, json!({ "name": self.name_buffer }));
            self.record(result);
        }

        ui.horizontal(|ui| {
            ui.label("op:");
            for op in Op::ALL {
                if ui.selectable_label(region.op == op, op.as_str()).clicked() && region.op != op {
                    let result = self
                        .doc
                        .update_region(&region.id, json!({ "op": op.as_str() }));
                    self.record(result);
                }
            }
        });
        if region.op == Op::Delete {
            ui.label("delete ignores mix: the points and Gaussians inside are removed");
        } else {
            let mut mix = region.mix;
            #[allow(clippy::cast_possible_truncation)]
            if ui
                .add(egui::Slider::new(&mut mix, 0.0..=1.0).text("mix (0 = splat, 1 = TRIPS)"))
                .changed()
            {
                self.set_mix(&region.id, mix);
            }
        }

        let speed = f64::from(scene_diameter) * 0.005;
        let mut params = region.params.clone();
        let mut changed = false;
        match &mut params {
            Params::Box {
                center,
                half_extents,
                quat,
            } => {
                changed |= vec3_row(ui, "centre", center, speed);
                changed |= vec3_row(ui, "half extents", half_extents, speed);
                changed |= vec4_row(ui, "quat (w x y z)", quat, 0.01);
            }
            Params::Sphere { center, radius } => {
                changed |= vec3_row(ui, "centre", center, speed);
                changed |= scalar_row(ui, "radius", radius, speed);
            }
            Params::Lid(lid) => {
                changed |= vec3_row(ui, "up", &mut lid.up, 0.01);
                changed |= scalar_row(ui, "height", &mut lid.height, speed);
                changed |= vec3_row(ui, "centre", &mut lid.center, speed);
                changed |= scalar_row(ui, "radius", &mut lid.radius, speed);
                changed |= scalar_row(ui, "falloff", &mut lid.falloff, speed);
                changed |= scalar_row(ui, "band", &mut lid.band, speed);
            }
            Params::Pointset { point_ids } => {
                ui.label(format!(
                    "{} point ids (from the shade finder or a Python selection); \
                     these index points.npz's own row order",
                    point_ids.len()
                ));
            }
            Params::Brush {
                origin,
                cell_size,
                cells,
            } => {
                ui.label(format!(
                    "{} painted voxel cells of {cell_size:.4} world units, on a grid anchored \
                     at ({:.2}, {:.2}, {:.2})",
                    cells.len(),
                    origin[0],
                    origin[1],
                    origin[2],
                ));
                ui.label(
                    "paint more of it with the Brush tool (T); the grid is fixed when the \
                     region is created, so the radius slider changes the STROKE, not the cells \
                     already painted",
                );
            }
        }
        if changed {
            self.set_params(&region.id, &params);
        }
        if ui.button("delete region").clicked() {
            self.delete_selected();
        }
    }

    /// The Tools panel: the shade-cloud finder's sliders, preview and "select".
    #[allow(clippy::too_many_lines)]
    fn tools_panel(&mut self, ui: &mut egui::Ui, num_points: usize) {
        ui.label(egui::RichText::new("Tools (T)").strong());
        let previous = self.tool;
        ui.horizontal(|ui| {
            for tool in [
                Tool::Regions,
                Tool::ShadeFinder,
                Tool::ClickCluster,
                Tool::Sam,
                Tool::Brush,
            ] {
                ui.selectable_value(&mut self.tool, tool, tool.label());
            }
        });
        if previous == Tool::ClickCluster && self.tool != previous {
            // Leaving the tool drops its index: a `f64` copy of the cloud is
            // not something to hold for a tool nobody is using (`ClickCache`).
            self.click.cache = None;
        }
        if previous == Tool::Brush && self.tool != previous {
            // Same rule for the brush's own widened copy of the cloud, and the
            // screen-space index built from it (stale the moment the points
            // it indexed might not be).
            self.brush.cache = None;
            self.brush.screen_grid = None;
        }
        match self.tool {
            Tool::Regions => {
                ui.label("place box/sphere/lid regions from the Regions panel above");
                return;
            }
            Tool::ClickCluster => {
                self.click_panel(ui, num_points);
                return;
            }
            Tool::Sam => {
                self.sam_panel(ui, num_points);
                return;
            }
            Tool::Brush => {
                self.brush_panel(ui, num_points);
                return;
            }
            Tool::ShadeFinder => {}
        }
        if let Some(reason) = &self.shade.unavailable {
            ui.colored_label(egui::Color32::from_rgb(255, 200, 120), reason);
            return;
        }

        ui.label(format!(
            "{} shade frames from {}{}",
            self.shade.views.views.len(),
            self.shade.views.source,
            if self.shade.views.depth_is_estimated {
                "  (median depth ESTIMATED from this bundle's own points, not COLMAP's sparse \
                 observations -- the selection will not match `trippy edits shade-find` exactly)"
            } else {
                ""
            }
        ));
        let t = &mut self.shade.thresholds;
        let mut moved = false;
        moved |= ui
            .add(egui::Slider::new(&mut t.lum_threshold, 0.0..=1.0).text("luminance <"))
            .changed();
        moved |= ui
            .add(egui::Slider::new(&mut t.conf_threshold, 0.0..=1.0).text("confidence <"))
            .changed();
        moved |= ui
            .add(egui::Slider::new(&mut t.znear_frac, 0.0..=1.0).text("znear / median depth"))
            .changed();
        moved |= ui
            .add(egui::Slider::new(&mut t.zfar_frac, 0.0..=2.0).text("zfar / median depth"))
            .changed();
        if moved {
            self.shade.dirty = true;
        }
        if ui.checkbox(&mut self.shade.preview, "preview highlight (H)").changed() {
            self.needs_apply = true;
        }
        ui.label(format!(
            "{} of {num_points} points selected  |  {} in the audit region  |  \
             dark mass fraction {:.4}",
            self.shade.selection.point_ids.len(),
            self.shade.selection.n_in_region,
            self.shade.selection.dark_mass_fraction
        ));
        if self.shade.preview {
            ui.label(
                "the selection is tinted magenta and everything else dimmed; press V for the \
                 raw level-0 view, where the tint is the rasteriser's own pixels rather than \
                 the network's",
            );
        }
        ui.horizontal(|ui| {
            let usable = !self.shade.selection.point_ids.is_empty();
            if ui
                .add_enabled(usable, egui::Button::new("add as region (fade)"))
                .clicked()
            {
                self.add_shade_region(Op::Fade);
            }
            if ui
                .add_enabled(usable, egui::Button::new("add as region (delete)"))
                .clicked()
            {
                self.add_shade_region(Op::Delete);
            }
        });
    }

    /// The SAM panel: the prompt, the run/cancel buttons, the child's own output.
    ///
    /// `docs/EDITOR.md` §3's "4. SAM 3 lift (E5)". Everything here either
    /// becomes a flag on the one child process or reports what that child said;
    /// no segmentation, projection or voting happens in this process.
    #[allow(clippy::too_many_lines)]
    fn sam_panel(&mut self, ui: &mut egui::Ui, num_points: usize) {
        ui.label(
            "DRAG a box on the render to segment the object inside it, or ALT-CLICK a point. \
             SHIFT-drag still orbits. The camera must be pinned to a capture view -- the lift \
             needs the photograph, not the render.",
        );
        if !self.has_scene_root {
            ui.colored_label(
                egui::Color32::from_rgb(255, 200, 120),
                "this bundle.json records no `scene_root`, so the lift cannot find the \
                 photographs -- re-export it with `trippy export-bundle`",
            );
        }

        match &self.sam.prompt {
            Some((view, prompt)) => ui.label(format!("prompt: {} on {view}", prompt.label())),
            None => ui.label("prompt: none yet"),
        };

        ui.horizontal(|ui| {
            ui.label("op:");
            for op in Op::ALL {
                ui.selectable_value(&mut self.sam.op, op, op.as_str());
            }
        });
        if self.sam.op == Op::Delete {
            ui.label("delete ignores mix: the lifted points are removed");
        } else {
            ui.add(
                egui::Slider::new(&mut self.sam.mix, 0.0..=1.0)
                    .text("mix (0 = splat, 1 = TRIPS)"),
            );
        }
        let mut views_around = self.sam.views_around as f64;
        if ui
            .add(
                egui::Slider::new(&mut views_around, 0.0..=8.0)
                    .integer()
                    .text("views around (0 = this view alone; 1 is an INTERSECTION, use 0 or >= 2)"),
            )
            .changed()
        {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                self.sam.views_around = views_around.round() as usize;
            }
        }
        ui.horizontal(|ui| {
            ui.label("device:");
            for device in ["cpu", "mps"] {
                ui.selectable_value(&mut self.sam.device, device, device);
            }
            ui.checkbox(&mut self.sam.fake, "fake (no SAM 3)")
                .on_hover_text(
                    "synthesise the mask from the prompt: proves the plumbing without \
                     loading the checkpoint",
                );
        });
        if self.sam.device == "mps" {
            ui.label(
                "mps is your own interactive GPU use, which is allowed outside the queue; \
                 a batch of lifts is not",
            );
        }

        let running = self.sam_running();
        let bundle_dir = self.bundle_dir.clone();
        ui.horizontal(|ui| {
            let can_run = !running && self.sam.prompt.is_some() && self.has_scene_root;
            if ui
                .add_enabled(can_run, egui::Button::new("run SAM lift"))
                .on_hover_text("spawns `trippy edits sam` once; this is the only child it starts")
                .clicked()
            {
                if let Err(e) = self.start_sam(&bundle_dir) {
                    self.error = Some(e);
                }
            }
            if ui
                .add_enabled(running, egui::Button::new("cancel"))
                .on_hover_text("kill the child process")
                .clicked()
            {
                self.cancel_sam();
            }
            if ui
                .add_enabled(
                    !self.sam.selection.is_empty(),
                    egui::Button::new("clear highlight"),
                )
                .clicked()
            {
                self.sam.selection.clear();
                self.needs_apply = true;
            }
        });
        if ui
            .checkbox(&mut self.sam.preview, "preview highlight (H)")
            .changed()
        {
            self.needs_apply = true;
        }

        if let Some(job) = &self.sam.job {
            ui.label(format!("running for {:.1} s", job.elapsed()));
            // The child is alive: the panel has to keep asking for frames or
            // its progress would only update when something else moved.
            ui.ctx().request_repaint();
        }
        if let Some(summary) = &self.sam.summary {
            ui.label(format!(
                "{} of {num_points} points lifted ({} in the prompted view, {} view(s) voted)",
                summary["n_points"].as_u64().unwrap_or(0),
                summary["n_points_primary_view"].as_u64().unwrap_or(0),
                summary["vote"]["n_views"].as_u64().unwrap_or(0),
            ));
        }
        if !self.sam.note.is_empty() {
            ui.label(&self.sam.note);
        }
        if let Some(job) = &self.sam.job {
            let lines = job.lines();
            egui::ScrollArea::vertical()
                .id_salt("sam-log")
                .max_height(120.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for line in &lines {
                        ui.label(egui::RichText::new(line).monospace().size(11.0));
                    }
                });
        }
        if !self.sam.command.is_empty() {
            ui.collapsing("command", |ui| {
                ui.label(egui::RichText::new(&self.sam.command).monospace().size(11.0));
            });
        }
    }

    /// The Brush panel: radius, weight, op/mix, and what the last stroke did.
    ///
    /// `docs/EDITOR.md` §1 "brush" / §4. The stroke itself is a drag on the
    /// render; everything here is the settings it uses and the state it leaves.
    fn brush_panel(&mut self, ui: &mut egui::Ui, num_points: usize) {
        ui.label(
            "DRAG on the render to paint a 3D brush region, ALT-drag to erase. \
             SHIFT-drag still orbits. Each dab is a sphere at the depth of the nearest \
             point under the cursor -- aim at the scene, not at empty sky.",
        );

        let mut radius = self.brush.radius;
        // Logarithmic, like the click tool's own radius: scene scales differ by
        // orders of magnitude and a linear slider is unusable on both.
        if ui
            .add(
                egui::Slider::new(&mut radius, 1e-4..=1e3)
                    .logarithmic(true)
                    .text("radius (world units)  [ / ]"),
            )
            .changed()
        {
            self.brush.radius = radius.max(1e-9);
        }
        ui.add(
            egui::Slider::new(&mut self.brush.weight, 0.0..=1.0)
                .text("weight: how strongly a painted cell claims a point"),
        );

        let active = self
            .brush
            .region
            .as_ref()
            .and_then(|id| self.doc.region(id))
            .map(|r| (r.id.clone(), r.name.clone(), r.op, r.mix));
        match &active {
            Some((id, name, _, _)) => {
                let cells = self
                    .doc
                    .region(id)
                    .and_then(|r| match &r.params {
                        Params::Brush { cells, .. } => Some(cells.len()),
                        _ => None,
                    })
                    .unwrap_or(0);
                ui.label(format!(
                    "painting into {name}: {cells} cells, {} of {num_points} points claimed",
                    self.count_for(id)
                        .map_or_else(|| "?".to_owned(), |n| n.to_string())
                ));
            }
            None => {
                ui.label("the next stroke creates a new brush region");
                ui.horizontal(|ui| {
                    ui.label("new region op:");
                    for op in Op::ALL {
                        ui.selectable_value(&mut self.brush.op, op, op.as_str());
                    }
                });
                if self.brush.op == Op::Delete {
                    ui.label("delete ignores mix: the painted points are removed");
                } else {
                    ui.add(
                        egui::Slider::new(&mut self.brush.mix, 0.0..=1.0)
                            .text("mix (0 = splat, 1 = TRIPS)"),
                    );
                }
            }
        }

        ui.horizontal(|ui| {
            if ui
                .add_enabled(active.is_some(), egui::Button::new("start a new region"))
                .on_hover_text("the next stroke paints into a fresh brush region")
                .clicked()
            {
                self.new_brush_region();
            }
            if ui
                .add_enabled(active.is_some(), egui::Button::new("select in Inspector"))
                .clicked()
            {
                self.selected.clone_from(&self.brush.region);
            }
        });
        if !self.brush.note.is_empty() {
            ui.label(&self.brush.note);
        }
    }

    /// The Named Objects panel: every region, grouped by the tool that made it.
    ///
    /// `docs/EDITOR.md` §4's Regions panel, grown into the list the brief asks
    /// for: name, source tool, kind, how many points it claims, an enable
    /// toggle, a mix slider, rename, remove, and "solo". Grouping is by
    /// `Region.source.tool` (`docs/EDITOR.md` §1 "Named regions"), with
    /// hand-authored regions in their own group — the same provenance the CLI's
    /// `trippy edits list` prints.
    #[allow(clippy::too_many_lines)]
    fn named_objects_panel(&mut self, ui: &mut egui::Ui) {
        let groups = group_by_tool(self.doc.regions());

        let header = format!(
            "Named Objects ({} region{}, {} group{})",
            self.doc.regions().len(),
            if self.doc.regions().len() == 1 { "" } else { "s" },
            groups.len(),
            if groups.len() == 1 { "" } else { "s" },
        );
        let mut toggled: Option<(String, bool)> = None;
        let mut mixed: Option<(String, f64)> = None;
        let mut removed: Option<String> = None;
        let mut renamed: Option<(String, String)> = None;
        let mut solo_click: Option<String> = None;
        let mut selected: Option<String> = None;

        egui::CollapsingHeader::new(egui::RichText::new(header).strong())
            .id_salt("named-objects")
            .default_open(true)
            .show(ui, |ui| {
                if groups.is_empty() {
                    ui.label("no regions yet -- paint one with the Brush tool, or add a box");
                    return;
                }
                if self.solo.is_some() {
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 200, 120),
                        "SOLO is on: the render shows one region's effect and no other",
                    );
                }
                egui::ScrollArea::vertical()
                    .id_salt("named-objects-scroll")
                    .max_height(260.0)
                    .show(ui, |ui| {
                        for (tool, regions) in &groups {
                            ui.label(egui::RichText::new(format!("{tool} ({})", regions.len())).italics());
                            for region in regions {
                                let id = region.id.as_str();
                                ui.horizontal(|ui| {
                                    let mut enabled = region.enabled;
                                    if ui
                                        .checkbox(&mut enabled, "")
                                        .on_hover_text("enable / disable without deleting")
                                        .changed()
                                    {
                                        toggled = Some((id.to_owned(), enabled));
                                    }
                                    if self.rename_for.as_deref() == Some(id) {
                                        let response = ui.add(
                                            egui::TextEdit::singleline(&mut self.rename_buffer)
                                                .desired_width(150.0),
                                        );
                                        response.request_focus();
                                        let done = ui.button("ok").clicked()
                                            || (response.lost_focus()
                                                && ui.input(|i| i.key_pressed(egui::Key::Enter)));
                                        if done {
                                            renamed = Some((
                                                id.to_owned(),
                                                self.rename_buffer.clone(),
                                            ));
                                        }
                                    } else {
                                        let label = if region.name.is_empty() {
                                            id.to_owned()
                                        } else {
                                            region.name.clone()
                                        };
                                        if ui
                                            .selectable_label(
                                                self.selected.as_deref() == Some(id),
                                                label,
                                            )
                                            .clicked()
                                        {
                                            selected = Some(id.to_owned());
                                        }
                                    }
                                    ui.label(format!("[{} {}]", region.short_kind(), region.op.as_str()));
                                    ui.label(match (region.enabled, self.count_for(id)) {
                                        (false, _) => "off".to_owned(),
                                        (true, Some(n)) => format!("{n} pts"),
                                        (true, None) => "-".to_owned(),
                                    });
                                });
                                ui.horizontal(|ui| {
                                    ui.add_space(24.0);
                                    if region.op == Op::Delete {
                                        ui.label("delete (no mix)");
                                    } else {
                                        let mut mix = region.mix;
                                        if ui
                                            .add(
                                                egui::Slider::new(&mut mix, 0.0..=1.0)
                                                    .show_value(true)
                                                    .text("mix"),
                                            )
                                            .changed()
                                        {
                                            mixed = Some((id.to_owned(), mix));
                                        }
                                    }
                                    let soloed = self.solo.as_deref() == Some(id);
                                    if ui
                                        .selectable_label(soloed, "solo")
                                        .on_hover_text(
                                            "show ONLY this region's effect (a disabled region \
                                             still shows nothing)",
                                        )
                                        .clicked()
                                    {
                                        solo_click = Some(id.to_owned());
                                    }
                                    if ui.button("rename").clicked() {
                                        self.rename_for = Some(id.to_owned());
                                        self.rename_buffer.clone_from(&region.name);
                                    }
                                    if ui.button("remove").clicked() {
                                        removed = Some(id.to_owned());
                                    }
                                });
                            }
                        }
                    });
            });

        // Every mutation is applied AFTER the loop: the rows borrow `self.doc`.
        if let Some(id) = selected {
            self.selected = Some(id);
        }
        if let Some((id, enabled)) = toggled {
            let result = self.doc.update_region(&id, json!({ "enabled": enabled }));
            self.record(result);
        }
        if let Some((id, mix)) = mixed {
            self.set_mix(&id, mix);
        }
        if let Some((id, name)) = renamed {
            let result = self.doc.update_region(&id, json!({ "name": name }));
            self.record(result);
            self.rename_for = None;
        }
        if let Some(id) = solo_click {
            let next = (self.solo.as_deref() != Some(id.as_str())).then_some(id);
            self.set_solo(next.as_deref());
        }
        if let Some(id) = removed {
            let result = self.doc.remove_region(&id);
            self.record(result);
            if self.selected.as_deref() == Some(id.as_str()) {
                self.selected = None;
            }
            if self.solo.as_deref() == Some(id.as_str()) {
                self.set_solo(None);
            }
            if self.brush.region.as_deref() == Some(id.as_str()) {
                self.brush.region = None;
            }
        }
    }

    /// The Selection panel: what the last Shift-click found, and what to do with it.
    ///
    /// `docs/EDITOR.md` §4's Tools panel for E4. The four sliders are the same
    /// four numbers `trippy edits click` takes on the command line, so a
    /// selection made here and one made there are the same selection.
    #[allow(clippy::too_many_lines)]
    fn click_panel(&mut self, ui: &mut egui::Ui, num_points: usize) {
        ui.label(
            "SHIFT-CLICK the render to select the object under the pointer. \
             A plain drag still orbits.",
        );

        let mut moved = false;
        let params = &mut self.click.params;
        moved |= ui
            .add(
                egui::Slider::new(&mut params.radius_px, 1.0..=64.0)
                    .text("radius (px): how far from the click a point may project"),
            )
            .changed();
        moved |= ui
            .add(
                egui::Slider::new(&mut params.colour_tol, 0.0..=2.0)
                    .text("colour tol: distance in [0,1]^3 from the seed's mean colour"),
            )
            .changed();
        // The radius slider is logarithmic because scene scales differ by
        // orders of magnitude and a linear one is unusable on both.
        moved |= ui
            .add(
                egui::Slider::new(&mut params.max_radius, 1e-3..=1e3)
                    .logarithmic(true)
                    .text("max radius (world units) from the seed centroid"),
            )
            .changed();
        let mut max_points = params.max_points as f64;
        if ui
            .add(
                egui::Slider::new(&mut max_points, 1.0..=(DEFAULT_MAX_POINTS as f64))
                    .logarithmic(true)
                    .text("max points"),
            )
            .changed()
        {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                params.max_points = (max_points.round() as usize).max(1);
            }
            moved = true;
        }
        if moved && self.click.last.is_some() {
            // Re-run the LAST click, not a new one: a slider is "show me more
            // or less of what I clicked", never "click somewhere else".
            self.click.dirty = true;
        }

        if ui
            .checkbox(&mut self.click.preview, "preview highlight (H)")
            .changed()
        {
            self.needs_apply = true;
        }

        let selection = &self.click.selection;
        ui.label(format!(
            "{} of {num_points} points selected  |  {} candidates within the radius  |  \
             {} seeded the nearest depth mode  |  clustered in {:.1} ms",
            selection.point_ids.len(),
            selection.n_candidates,
            selection.n_seed,
            self.click.ms
        ));
        if let Some(depth) = selection.seed_depth_mean {
            ui.label(format!(
                "seed depth {depth:.3} world units in front of the camera{}",
                if selection.hit_max_points {
                    "  |  STOPPED AT max points -- raise it to grow further"
                } else {
                    ""
                }
            ));
        }
        if let Some(cache) = &self.click.cache {
            ui.label(format!(
                "neighbour index: {} points, {:.1} MiB, built in {:.0} ms (dropped when you \
                 leave this tool or press clear)",
                cache.grid.len(),
                cache.megabytes(),
                cache.build_ms
            ));
        }
        if !self.click.note.is_empty() {
            ui.label(&self.click.note);
        }

        ui.horizontal(|ui| {
            ui.label("op:");
            for op in Op::ALL {
                ui.selectable_value(&mut self.click.op, op, op.as_str());
            }
        });
        if self.click.op == Op::Delete {
            ui.label("delete ignores mix: the selected points are removed");
        } else {
            ui.add(
                egui::Slider::new(&mut self.click.mix, 0.0..=1.0)
                    .text("mix (0 = splat, 1 = TRIPS)"),
            );
        }

        ui.horizontal(|ui| {
            let usable = !self.click.selection.point_ids.is_empty();
            if ui
                .add_enabled(usable, egui::Button::new("add as region"))
                .on_hover_text("commit this selection as a pointset region")
                .clicked()
            {
                self.add_click_region();
            }
            if ui
                .add_enabled(usable, egui::Button::new("clear"))
                .on_hover_text("drop the selection, its highlight and its neighbour index")
                .clicked()
            {
                self.clear_click();
            }
        });
    }

    /// Turn the click selection into a committed `pointset` region.
    fn add_click_region(&mut self) {
        let params = self.click.params;
        let (op, mix) = (self.click.op, self.click.mix);
        self.add(Region::new(
            new_region_id(),
            format!(
                "click cluster ({:.0} px, tol {:.2})",
                params.radius_px, params.colour_tol
            ),
            Params::Pointset {
                point_ids: self.click.selection.point_ids.clone(),
            },
            mix,
            op,
        ));
        // The committed region is its own thing now; leave the tint off so it
        // does not sit on top of the edit it just became (the shade finder's
        // "add as region" makes the same choice, for the same reason).
        self.click.preview = false;
        self.needs_apply = true;
    }

    /// Turn the live preview into a committed `pointset` region.
    fn add_shade_region(&mut self, op: Op) {
        let t = self.shade.thresholds;
        self.add(Region::new(
            new_region_id(),
            format!(
                "shade cloud (lum<{:.2} conf<{:.2})",
                t.lum_threshold, t.conf_threshold
            ),
            Params::Pointset {
                point_ids: self.shade.selection.point_ids.clone(),
            },
            0.0,
            op,
        ));
        // A committed region is its own thing now; leave the preview off so the
        // tint does not sit on top of the edit it just became.
        self.shade.preview = false;
        self.needs_apply = true;
    }

    /// Re-run the shade finder when a slider moved.
    ///
    /// Reads the cloud's own `feat[:, :3]` as base colour, exactly as
    /// `trippy.edit.shade_finder.find_shade_pointset_in_bundle` does.
    pub fn refresh_shade(&mut self, renderer: &Renderer) {
        if self.shade.unavailable.is_some() || !self.shade.dirty {
            return;
        }
        self.shade.dirty = false;
        let points = renderer.base_points();
        let xyz = widen(&points.xyz);
        let channels = points.num_channels;
        let mut rgb = Vec::with_capacity(points.len() * 3);
        for row in 0..points.len() {
            for c in 0..3 {
                rgb.push(f64::from(
                    points.feat[row * channels + c].clamp(0.0, 1.0),
                ));
            }
        }
        let conf: Vec<f64> = points.conf.iter().map(|c| f64::from(*c)).collect();
        self.shade.selection = shade::find(
            &self.shade.views.views,
            &xyz,
            &rgb,
            &conf,
            self.shade.thresholds,
        );
        if self.shade.preview {
            self.needs_apply = true;
        }
    }

    /// Fill in each shade frame's median depth from the bundle's own points,
    /// for a bundle with no `shade_views.json`.
    ///
    /// Used only by the fallback path; see `edit::shade`'s invariants for why
    /// this is announced rather than silent.
    pub fn estimate_shade_depths(&mut self, renderer: &Renderer) {
        if !self.shade.views.depth_is_estimated {
            return;
        }
        let xyz = widen(&renderer.base_points().xyz);
        for view in &mut self.shade.views.views {
            view.d = view.estimate_median_depth(&xyz);
        }
        self.shade.dirty = true;
    }
}

/// The label a region with no `source` is grouped under in Named Objects.
const HAND_AUTHORED_GROUP: &str = "hand-authored";

/// Regions grouped by the tool that made them (`Region.source.tool`).
///
/// Paint order within a group, and groups in the order they first appear, so
/// the list is stable while regions are added and removed — a list that
/// reordered itself under Jordan's pointer would be worse than an unsorted one.
fn group_by_tool(regions: &[Region]) -> Vec<(String, Vec<Region>)> {
    let mut groups: Vec<(String, Vec<Region>)> = Vec::new();
    for region in regions {
        let tool = region
            .source_tool()
            .map_or_else(|| HAND_AUTHORED_GROUP.to_owned(), ToOwned::to_owned);
        match groups.iter_mut().find(|(name, _)| *name == tool) {
            Some((_, list)) => list.push(region.clone()),
            None => groups.push((tool, vec![region.clone()])),
        }
    }
    groups
}

/// `0/1/2` -> `"world X"/"world Y"/"world Z"`, for the gizmo's own note.
fn axis_name(axis: usize) -> &'static str {
    match axis {
        0 => "world X",
        1 => "world Y",
        2 => "world Z",
        // Reachable only if a future caller passes `gizmo::NORMAL_AXIS` here;
        // `Drag::Tilt`'s own note branch never does (see `gizmo_drag_to`).
        _ => "the lid's normal",
    }
}

/// A labelled row of three drag boxes. Returns whether any changed.
fn vec3_row(ui: &mut egui::Ui, label: &str, value: &mut [f64; 3], speed: f64) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        for slot in value.iter_mut() {
            changed |= ui.add(egui::DragValue::new(slot).speed(speed)).changed();
        }
    });
    changed
}

/// A labelled row of four drag boxes.
fn vec4_row(ui: &mut egui::Ui, label: &str, value: &mut [f64; 4], speed: f64) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        for slot in value.iter_mut() {
            changed |= ui.add(egui::DragValue::new(slot).speed(speed)).changed();
        }
    });
    changed
}

/// A labelled single drag box.
fn scalar_row(ui: &mut egui::Ui, label: &str, value: &mut f64, speed: f64) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed |= ui.add(egui::DragValue::new(value).speed(speed)).changed();
    });
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use trips_viewer::edit::model::LidParams;

    #[test]
    fn the_tool_key_cycles_and_returns() {
        let mut tool = Tool::default();
        assert_eq!(tool, Tool::Regions);
        tool = tool.next();
        assert_eq!(tool, Tool::ShadeFinder);
        tool = tool.next();
        assert_eq!(tool, Tool::ClickCluster);
        tool = tool.next();
        assert_eq!(tool, Tool::Sam);
        tool = tool.next();
        assert_eq!(tool, Tool::Brush);
        tool = tool.next();
        assert_eq!(tool, Tool::Regions);
    }

    #[test]
    fn a_shift_click_is_recorded_and_focuses_the_click_tool() {
        let dir = std::env::temp_dir().join(format!("trips-edit-click-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut session = EditSession::open_with_path(&dir, "trippy-bundle-1", None);
        assert_eq!(session.tool, Tool::Regions);
        session.request_click((12.0, 34.0));
        assert_eq!(session.tool, Tool::ClickCluster, "the click picks its tool");
        assert_eq!(session.click.pending, Some((12.0, 34.0)));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn clearing_a_click_drops_the_selection_the_preview_and_the_index() {
        let dir = std::env::temp_dir().join(format!("trips-edit-clear-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut session = EditSession::open_with_path(&dir, "trippy-bundle-1", None);
        session.click.selection = ClickSelection {
            point_ids: vec![1, 2, 3],
            n_candidates: 9,
            n_seed: 3,
            ..ClickSelection::default()
        };
        session.click.preview = true;
        assert_eq!(session.preview_ids(), Some(vec![1, 2, 3]));

        session.needs_apply = false;
        session.clear_click();
        assert!(session.click.selection.point_ids.is_empty());
        assert!(session.click.cache.is_none());
        assert_eq!(session.preview_ids(), None);
        assert!(
            session.needs_apply,
            "clearing has to reach the render, or the tint stays on screen"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn adding_a_click_region_commits_the_ids_and_turns_the_tint_off() {
        let dir = std::env::temp_dir().join(format!("trips-edit-cadd-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut session = EditSession::open_with_path(&dir, "trippy-bundle-1", None);
        session.click.selection = ClickSelection {
            point_ids: vec![4, 7, 9],
            ..ClickSelection::default()
        };
        session.click.preview = true;
        session.click.op = Op::Delete;
        session.add_click_region();

        let region = session.doc.regions().last().expect("a region was added");
        assert_eq!(region.op, Op::Delete);
        assert!(matches!(
            &region.params,
            Params::Pointset { point_ids } if point_ids == &vec![4, 7, 9]
        ));
        assert!(!session.click.preview, "the tint comes off what it became");
        assert!(session.dirty, "an added region is an unsaved change");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_preview_tints_both_tools_selections_at_once() {
        let dir = std::env::temp_dir().join(format!("trips-edit-both-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut session = EditSession::open_with_path(&dir, "trippy-bundle-1", None);
        session.shade.selection.point_ids = vec![1, 2];
        session.shade.preview = true;
        session.click.selection.point_ids = vec![5];
        session.click.preview = true;
        assert_eq!(session.preview_ids(), Some(vec![1, 2, 5]));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_missing_edits_json_opens_an_empty_session_rather_than_failing() {
        let dir = std::env::temp_dir().join(format!("trips-edit-ui-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let session = EditSession::open_with_path(&dir, "trippy-bundle-1", None);
        assert!(session.doc.regions().is_empty());
        assert!(session.error.is_none());
        assert!(!session.dirty);
        assert!(session.needs_apply, "the first frame must still apply");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_malformed_edits_json_is_reported_and_does_not_stop_the_scene_opening() {
        let dir = std::env::temp_dir().join(format!("trips-edit-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(EDITS_FILENAME), "{ not json").unwrap();
        let session = EditSession::open_with_path(&dir, "trippy-bundle-1", None);
        assert!(session.error.is_some());
        assert!(session.doc.regions().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_session_saves_and_reopens_with_its_regions_and_its_history() {
        let dir = std::env::temp_dir().join(format!("trips-edit-rt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut session = EditSession::open_with_path(&dir, "trippy-bundle-1", None);
        session.add(Region::new(
            "r-a".to_owned(),
            "box a".to_owned(),
            Params::Box {
                center: [0.0, 0.0, 1.0],
                half_extents: [1.0, 1.0, 1.0],
                quat: [1.0, 0.0, 0.0, 0.0],
            },
            0.5,
            Op::Blend,
        ));
        session.add(Region::new(
            "r-b".to_owned(),
            "sphere b".to_owned(),
            Params::Sphere {
                center: [1.0, 0.0, 1.0],
                radius: 2.0,
            },
            0.0,
            Op::Delete,
        ));
        session.selected = Some("r-a".to_owned());
        session.move_selected(1);
        session.undo();
        assert!(session.dirty);
        session.save();
        assert!(!session.dirty);

        let reopened = EditSession::open_with_path(&dir, "trippy-bundle-1", None);
        assert_eq!(reopened.doc.regions().len(), 2);
        assert_eq!(reopened.doc.order(), ["r-a", "r-b"]);
        assert_eq!(reopened.doc.cursor, session.doc.cursor);
        assert!(reopened.doc.can_redo(), "the reorder is still redoable");
        assert!((reopened.doc.region("r-a").unwrap().mix - 0.5).abs() < 1e-12);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn nudging_and_resizing_go_through_the_undo_log() {
        let dir = std::env::temp_dir().join(format!("trips-edit-nudge-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut session = EditSession::open_with_path(&dir, "trippy-bundle-1", None);
        session.add(Region::new(
            "r-a".to_owned(),
            "sphere".to_owned(),
            Params::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 1.0,
            },
            0.0,
            Op::Blend,
        ));
        session.nudge_selected([0.5, 0.0, 0.0]);
        session.resize_selected(2.0);
        let region = session.doc.region("r-a").unwrap();
        assert!(matches!(
            region.params,
            Params::Sphere { center, radius }
                if (center[0] - 0.5).abs() < 1e-12 && (radius - 2.0).abs() < 1e-12
        ));
        // Both are undoable, one step each.
        session.undo();
        assert!(matches!(
            session.doc.region("r-a").unwrap().params,
            Params::Sphere { radius, .. } if (radius - 1.0).abs() < 1e-12
        ));
        session.undo();
        assert!(matches!(
            session.doc.region("r-a").unwrap().params,
            Params::Sphere { center, .. } if center[0] == 0.0
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    // --- the SAM tool (E5) --------------------------------------------------

    /// A session in its own throwaway directory, named after the test.
    fn a_session(tag: &str) -> (PathBuf, EditSession) {
        let dir = std::env::temp_dir().join(format!("trips-edit-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let session = EditSession::open_with_path(&dir, "trippy-bundle-1", None);
        (dir, session)
    }

    /// A view the mapping is the identity on, so these tests are about the
    /// SESSION's behaviour and not about the arithmetic `edit::sam` already
    /// has its own tests for.
    fn a_view() -> trips_viewer::bundle::BundleView {
        trips_viewer::bundle::BundleView {
            index: 0,
            name: "IMG_0000.jpg".to_owned(),
            width: 200,
            height: 100,
            fx: 100.0,
            fy: 100.0,
            cx: 100.0,
            cy: 50.0,
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            t: [0.0, 0.0, 0.0],
            distortion: [0.0; 8],
        }
    }

    #[test]
    fn the_tool_cycle_reaches_the_sam_tool_and_returns() {
        let mut tool = Tool::ClickCluster;
        tool = tool.next();
        assert_eq!(tool, Tool::Sam);
        assert_eq!(tool.next(), Tool::Brush);
    }

    #[test]
    fn a_pinned_gesture_becomes_a_prompt_in_the_views_own_pixels() {
        let (dir, mut session) = a_session("sam-resolve");
        let view = a_view();
        session.request_sam(SamGesture::Box((10.0, 20.0), (110.0, 80.0)));
        assert_eq!(session.tool, Tool::Sam, "the gesture picks its tool");
        session.resolve_sam(&view.camera(), &view, true, &[view.clone()], glam::Vec3::ZERO);

        match &session.sam.prompt {
            Some((name, SamPrompt::Box(b))) => {
                assert_eq!(name, "IMG_0000.jpg");
                assert_eq!(*b, [10.0, 20.0, 110.0, 80.0]);
            }
            other => panic!("expected a box prompt, got {other:?}"),
        }
        assert!(session.take_sam_snap().is_none(), "a pinned camera needs no snap");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unpinned_gesture_is_dropped_and_asks_for_a_snap_instead() {
        // The lift needs the PHOTOGRAPH: pixels measured against a free-flying
        // frame mean nothing in any photo, so re-using them would segment the
        // wrong thing while looking like it worked.
        let (dir, mut session) = a_session("sam-unpinned");
        let mut near = a_view();
        near.t = [0.0, 0.0, 0.0];
        let mut far = a_view();
        far.index = 1;
        far.name = "IMG_0001.jpg".to_owned();
        // `c = -R^T t`, so this camera's CENTRE is (50, 0, 0).
        far.t = [-50.0, 0.0, 0.0];
        let views = [near.clone(), far];

        session.request_sam(SamGesture::Point((10.0, 20.0)));
        session.resolve_sam(
            &near.camera(),
            &near,
            false,
            &views,
            glam::Vec3::new(40.0, 0.0, 0.0),
        );
        assert!(session.sam.prompt.is_none(), "the gesture is discarded, not reused");
        assert_eq!(session.take_sam_snap(), Some(1), "snapped to the nearer camera");
        assert!(session.take_sam_snap().is_none(), "the request is taken once");
        assert!(session.sam.note.contains("IMG_0001.jpg"), "{}", session.sam.note);
        assert!(session.sam.note.contains("snapped"), "{}", session.sam.note);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_run_without_a_prompt_or_a_scene_root_refuses_with_a_reason() {
        let (dir, mut session) = a_session("sam-refuse");
        let message = session.start_sam(&dir).unwrap_err();
        assert!(message.contains("drag a box"), "{message}");

        session.set_sam_prompt("IMG_0000.jpg", SamPrompt::Point((1.0, 2.0)));
        let message = session.start_sam(&dir).unwrap_err();
        assert!(message.contains("scene_root"), "{message}");
        assert!(!session.sam_running());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_imported_region_is_one_undo_step_and_lights_up() {
        // The import itself, without a child: what `poll_sam` does once the
        // process has landed. The child half is tested in `sam_child`.
        let (dir, mut session) = a_session("sam-import");
        let child_out = dir.join("child-edits.json");
        let mut child_doc = EditDocument::new("trippy-bundle-1".to_owned());
        child_doc
            .add_region(
                &Region::new(
                    "r-child".to_owned(),
                    "sam: box".to_owned(),
                    Params::Pointset {
                        point_ids: vec![2, 3, 5, 8],
                    },
                    0.0,
                    Op::Fade,
                ),
                None,
            )
            .unwrap();
        child_doc.save(&child_out).unwrap();

        session.import_sam_region(&child_out, 1.5).expect("imported");
        assert_eq!(session.sam_selection(), [2, 3, 5, 8]);
        assert_eq!(session.preview_ids(), Some(vec![2, 3, 5, 8]), "the tint is on");
        assert_eq!(session.doc.regions().len(), 1);
        assert!(session.dirty, "an imported region is an unsaved change");
        assert!(session.needs_apply, "the tint has to reach the render");

        // One undo step, exactly like every other edit.
        session.undo_once();
        assert!(session.doc.regions().is_empty(), "Cmd-Z takes it back out");
        assert!(session.doc.can_redo());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn importing_something_that_is_not_a_pointset_is_refused() {
        let (dir, mut session) = a_session("sam-badkind");
        let child_out = dir.join("child-edits.json");
        let mut child_doc = EditDocument::new("trippy-bundle-1".to_owned());
        child_doc
            .add_region(
                &Region::new(
                    "r-child".to_owned(),
                    "a sphere".to_owned(),
                    Params::Sphere {
                        center: [0.0, 0.0, 0.0],
                        radius: 1.0,
                    },
                    0.0,
                    Op::Fade,
                ),
                None,
            )
            .unwrap();
        child_doc.save(&child_out).unwrap();

        let message = session.import_sam_region(&child_out, 0.1).unwrap_err();
        assert!(message.contains("not a pointset"), "{message}");
        assert!(session.doc.regions().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn h_toggles_the_sam_tint_when_the_sam_tool_has_focus() {
        let (dir, mut session) = a_session("sam-h");
        session.sam.selection = vec![7];
        session.tool = Tool::Sam;
        assert_eq!(session.preview_ids(), Some(vec![7]));
        session.set_sam_preview(false);
        assert_eq!(session.preview_ids(), None);
        session.set_sam_preview(true);
        assert_eq!(session.preview_ids(), Some(vec![7]));
        std::fs::remove_dir_all(&dir).ok();
    }

    // --- the brush tool, the gizmos and Named Objects -----------------------

    /// The camera `edit::gizmo`'s own tests use: at the origin, looking down
    /// +Z, 100 px focal length, principal point (100, 100).
    fn a_camera() -> ClickCamera {
        ClickCamera {
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            t: [0.0, 0.0, 0.0],
            fx: 100.0,
            fy: 100.0,
            cx: 100.0,
            cy: 100.0,
        }
    }

    #[test]
    fn the_tool_cycle_reaches_the_brush_and_returns() {
        let mut tool = Tool::Sam;
        tool = tool.next();
        assert_eq!(tool, Tool::Brush);
        assert_eq!(tool.next(), Tool::Regions);
    }

    #[test]
    fn one_brush_stroke_is_one_region_one_source_and_one_undo_step() {
        let (dir, mut session) = a_session("brush-stroke");
        session.init_brush_defaults(4.0);
        assert!(session.brush_radius() > 0.0, "the radius comes from the scene");
        session.set_brush_settings(0.5, 0.75, Op::Delete, 0.0);
        assert_eq!(session.tool, Tool::Brush);

        // Two frames of one stroke. (`resolve_brush` needs a live Renderer for
        // its depth anchor, so the sphere is painted here and the SESSION's own
        // commit path — which is what carries the undo semantics — is exercised.)
        session.begin_brush_stroke(false);
        for centre in [[0.0, 0.0, 4.0], [0.3, 0.0, 4.0]] {
            let stroke = session.brush.stroke.as_mut().expect("stroking");
            brush::paint_sphere(
                &mut stroke.cells,
                stroke.origin,
                stroke.cell_size,
                centre,
                0.5,
                0.75,
            )
            .unwrap();
            session.commit_stroke();
        }
        session.end_brush_stroke();

        assert_eq!(session.doc.regions().len(), 1, "one stroke, one region");
        assert_eq!(session.doc.log.len(), 1, "one stroke, one undo entry");
        let region = &session.doc.regions()[0];
        assert_eq!(region.name, "brush-1", "the auto name Python would give it");
        assert_eq!(region.source_tool(), Some("brush"));
        assert_eq!(region.op, Op::Delete);
        let Params::Brush { cells, .. } = &region.params else {
            panic!("a brush stroke makes a brush region")
        };
        assert!(cells.len() > 1, "both dabs painted");
        assert!(
            cells.weights().iter().all(|w| (w - 0.75).abs() < 1e-12),
            "the weight slider reaches the cells"
        );

        // One Cmd-Z takes the whole stroke back.
        session.undo_once();
        assert!(session.doc.regions().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_second_stroke_paints_into_the_same_region_as_its_own_undo_step() {
        let (dir, mut session) = a_session("brush-second");
        session.set_brush_settings(0.5, 1.0, Op::Delete, 0.0);
        session.begin_brush_stroke(false);
        {
            let stroke = session.brush.stroke.as_mut().expect("stroking");
            brush::paint_sphere(&mut stroke.cells, stroke.origin, stroke.cell_size,
                                [0.0, 0.0, 4.0], 0.5, 1.0).unwrap();
        }
        session.commit_stroke();
        session.end_brush_stroke();
        let after_first = match &session.doc.regions()[0].params {
            Params::Brush { cells, .. } => cells.len(),
            _ => panic!("a brush"),
        };

        session.begin_brush_stroke(false);
        {
            let stroke = session.brush.stroke.as_mut().expect("stroking");
            assert!(!stroke.creating, "the second stroke reuses the tool's region");
            brush::paint_sphere(&mut stroke.cells, stroke.origin, stroke.cell_size,
                                [2.0, 0.0, 4.0], 0.5, 1.0).unwrap();
        }
        session.commit_stroke();
        session.end_brush_stroke();

        assert_eq!(session.doc.regions().len(), 1, "still one region");
        assert_eq!(session.doc.log.len(), 2, "two strokes, two undo steps");
        let after_second = match &session.doc.regions()[0].params {
            Params::Brush { cells, .. } => cells.len(),
            _ => panic!("a brush"),
        };
        assert!(after_second > after_first, "the second stroke added cells");

        // "start a new region" is what makes the next stroke its own object.
        session.new_brush_region();
        session.begin_brush_stroke(false);
        assert!(session.brush.stroke.as_ref().expect("stroking").creating);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_alt_stroke_erases_what_a_stroke_painted() {
        let (dir, mut session) = a_session("brush-erase");
        session.set_brush_settings(0.5, 1.0, Op::Delete, 0.0);
        session.begin_brush_stroke(false);
        {
            let stroke = session.brush.stroke.as_mut().expect("stroking");
            brush::paint_sphere(&mut stroke.cells, stroke.origin, stroke.cell_size,
                                [0.0, 0.0, 4.0], 0.5, 1.0).unwrap();
            assert!(!stroke.erasing);
        }
        session.commit_stroke();
        session.end_brush_stroke();

        session.begin_brush_stroke(true);
        {
            let stroke = session.brush.stroke.as_mut().expect("stroking");
            assert!(stroke.erasing, "Alt makes the stroke an erase");
            brush::erase(&mut stroke.cells, stroke.origin, stroke.cell_size,
                         [0.0, 0.0, 4.0], 0.5).unwrap();
            assert!(stroke.cells.is_empty(), "the erase cleared the dab");
        }
        session.commit_stroke();
        session.end_brush_stroke();

        let Params::Brush { cells, .. } = &session.doc.regions()[0].params else {
            panic!("a brush")
        };
        assert!(cells.is_empty(), "the region survives, its cells do not");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_brush_keys_change_the_radius_not_the_selected_region() {
        let (dir, mut session) = a_session("brush-keys");
        session.set_brush_settings(1.0, 1.0, Op::Delete, 0.0);
        // `[` / `]` are read in `handle_keys`, which needs an egui context; the
        // branch they take is the thing worth pinning, and it is this one.
        assert_eq!(session.tool, Tool::Brush);
        session.brush.radius *= BRUSH_RADIUS_STEP;
        assert!((session.brush_radius() - BRUSH_RADIUS_STEP).abs() < 1e-12);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_gizmo_drag_moves_the_region_and_is_one_undo_step() {
        let (dir, mut session) = a_session("gizmo-drag");
        session.active = true;
        session.add(Region::new(
            "r-g".to_owned(),
            "sphere".to_owned(),
            Params::Sphere {
                center: [0.0, 0.0, 10.0],
                radius: 1.0,
            },
            0.0,
            Op::Blend,
        ));
        let entries = session.doc.log.len();
        session.update_gizmo(&a_camera());
        let screen = session.gizmo_screen().expect("a sphere in front has handles");
        let tip = screen.arms[0].tip;

        // Grab the +X handle and drag it 16 px right: one arm length, 1.6 u.
        assert!(session.begin_gizmo_drag(tip, false, false), "the handle is grabbed");
        session.gizmo_drag_to((tip.0 + 8.0, tip.1));
        session.gizmo_drag_to((tip.0 + 16.0, tip.1));
        session.end_gizmo_drag();

        let Params::Sphere { center, radius } = session.doc.region("r-g").unwrap().params else {
            panic!("still a sphere")
        };
        assert!((center[0] - 1.6).abs() < 1e-6, "{center:?}");
        assert!((radius - 1.0).abs() < 1e-12, "a translate does not resize");
        assert_eq!(
            session.doc.log.len(),
            entries + 1,
            "however many frames the drag took, it is one entry"
        );
        session.undo_once();
        assert!(
            matches!(session.doc.region("r-g").unwrap().params,
                     Params::Sphere { center, .. } if center[0] == 0.0),
            "one Cmd-Z puts it back"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_drag_that_starts_off_a_handle_is_not_a_gizmo_drag() {
        // This is what keeps navigation: `app.rs` orbits with every drag
        // `begin_gizmo_drag` refuses.
        let (dir, mut session) = a_session("gizmo-miss");
        session.active = true;
        session.add(Region::new(
            "r-g".to_owned(),
            "sphere".to_owned(),
            Params::Sphere {
                center: [0.0, 0.0, 10.0],
                radius: 1.0,
            },
            0.0,
            Op::Blend,
        ));
        session.update_gizmo(&a_camera());
        assert!(!session.begin_gizmo_drag((400.0, 400.0), false, false));
        assert!(!session.gizmo_dragging());

        // Nor is one made while the Brush tool has the drag.
        session.tool = Tool::Brush;
        session.update_gizmo(&a_camera());
        assert!(session.gizmo_screen().is_none(), "no handles to steal the stroke");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn shift_and_ctrl_drags_resize_and_rotate() {
        let (dir, mut session) = a_session("gizmo-mods");
        session.active = true;
        session.add(Region::new(
            "r-b".to_owned(),
            "box".to_owned(),
            Params::Box {
                center: [0.0, 0.0, 10.0],
                half_extents: [1.0, 1.0, 1.0],
                quat: [1.0, 0.0, 0.0, 0.0],
            },
            0.0,
            Op::Blend,
        ));
        session.update_gizmo(&a_camera());
        let tip = session.gizmo_screen().expect("handles").arms[0].tip;

        assert!(session.begin_gizmo_drag(tip, true, false), "shift grabs a resize");
        session.gizmo_drag_to((tip.0 + gizmo::RESIZE_PX_PER_DOUBLING, tip.1));
        session.end_gizmo_drag();
        let Params::Box { half_extents, .. } = session.doc.region("r-b").unwrap().params else {
            panic!("still a box")
        };
        assert!((half_extents[0] - 2.0).abs() < 1e-6, "{half_extents:?}");

        session.update_gizmo(&a_camera());
        let centre = session.gizmo_screen().expect("handles").centre;
        let grab = (centre.0 + 40.0, centre.1);
        // The handle moved with the resize; grab it wherever it is now.
        let tip = session.gizmo_screen().expect("handles").arms[0].tip;
        assert!(session.begin_gizmo_drag(tip, false, true), "ctrl grabs a rotate");
        session.gizmo_drag_to((grab.0, grab.1 + 40.0));
        session.end_gizmo_drag();
        let Params::Box { quat, .. } = session.doc.region("r-b").unwrap().params else {
            panic!("still a box")
        };
        assert!(
            (quat[0] - 1.0).abs() > 1e-9,
            "the box really turned: {quat:?}"
        );

        // A sphere has no orientation, so ctrl refuses rather than pretending.
        session.add(Region::new(
            "r-s".to_owned(),
            "sphere".to_owned(),
            Params::Sphere {
                center: [0.0, 0.0, 10.0],
                radius: 1.0,
            },
            0.0,
            Op::Blend,
        ));
        session.update_gizmo(&a_camera());
        let tip = session.gizmo_screen().expect("handles").arms[0].tip;
        assert!(!session.begin_gizmo_drag(tip, false, true));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dragging_the_lids_normal_handle_tilts_up_as_one_undo_step_and_typing_still_works() {
        let (dir, mut session) = a_session("gizmo-lid-normal");
        session.active = true;
        session.add(Region::new(
            "r-lid".to_owned(),
            "pool lid".to_owned(),
            Params::Lid(KAREKARE_LID),
            1.0,
            Op::Delete,
        ));
        let entries = session.doc.log.len();
        session.update_gizmo(&a_camera());
        let screen = session.gizmo_screen().expect("a lid in front has handles");
        let normal_tip = screen.normal.expect("a lid has a 4th handle").tip;

        // Grab the normal handle (not an arm) and drag it -- one undo step
        // however many frames the drag took, exactly like a translate/resize.
        assert!(
            session.begin_gizmo_drag(normal_tip, false, false),
            "the normal handle is grabbed"
        );
        session.gizmo_drag_to((normal_tip.0 + 6.0, normal_tip.1));
        session.gizmo_drag_to((normal_tip.0 + 12.0, normal_tip.1));
        session.end_gizmo_drag();

        let Params::Lid(after_drag) = session.doc.region("r-lid").unwrap().params else {
            panic!("still a lid")
        };
        assert_ne!(after_drag.up, KAREKARE_LID.up, "the drag tilted it");
        let up_norm: f64 = after_drag.up.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((up_norm - 1.0).abs() < 1e-9, "tilting must keep up unit length: {up_norm}");
        assert_eq!(after_drag.height, KAREKARE_LID.height, "only up moved");
        assert_eq!(after_drag.center, KAREKARE_LID.center);
        assert_eq!(
            session.doc.log.len(),
            entries + 1,
            "however many frames the drag took, it is one entry"
        );

        // The Inspector's own path (typing a new `up`, `Params::Lid`'s
        // `vec3_row` branch in `ui()`) goes through `set_params`, not the
        // gizmo drag machinery -- it must still work after a tilt-drag.
        let typed = Params::Lid(LidParams {
            up: [0.0, 1.0, 0.0],
            ..after_drag
        });
        session.set_params("r-lid", &typed);
        let Params::Lid(after_typing) = session.doc.region("r-lid").unwrap().params else {
            panic!("still a lid")
        };
        assert_eq!(after_typing.up, [0.0, 1.0, 0.0]);

        // One Cmd-Z undoes the TYPED edit (a separate, later step from the
        // drag), landing back on the tilt the drag produced.
        session.undo_once();
        let Params::Lid(after_first_undo) = session.doc.region("r-lid").unwrap().params else {
            panic!("still a lid")
        };
        assert_eq!(after_first_undo.up, after_drag.up, "back to the drag's own result");

        // A second Cmd-Z undoes the drag itself, back to the untouched region.
        session.undo_once();
        let Params::Lid(after_second_undo) = session.doc.region("r-lid").unwrap().params else {
            panic!("still a lid")
        };
        assert_eq!(after_second_undo.up, KAREKARE_LID.up, "back to the original, untilted lid");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn named_objects_groups_by_the_tool_that_made_each_region() {
        let hand = Region::new(
            "r-1".to_owned(),
            "pool lid".to_owned(),
            Params::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 1.0,
            },
            0.0,
            Op::Delete,
        );
        let painted = Region::new(
            "r-2".to_owned(),
            "brush-1".to_owned(),
            Params::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 1.0,
            },
            0.0,
            Op::Delete,
        )
        .with_source("brush", json!({ "radius": 0.5 }));
        let clicked = Region::new(
            "r-3".to_owned(),
            "click-2".to_owned(),
            Params::Pointset { point_ids: vec![1] },
            0.0,
            Op::Fade,
        )
        .with_source("click", json!({}));
        let painted_again = Region {
            id: "r-4".to_owned(),
            ..painted.clone()
        };

        let groups = group_by_tool(&[hand, painted, clicked, painted_again]);
        let names: Vec<&str> = groups.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["hand-authored", "brush", "click"], "first-seen order");
        assert_eq!(groups[1].1.len(), 2, "both brush regions in one group");
        assert_eq!(groups[0].1[0].name, "pool lid");
    }

    #[test]
    fn solo_is_a_way_of_looking_and_not_an_edit() {
        let (dir, mut session) = a_session("solo");
        session.add(Region::new(
            "r-1".to_owned(),
            "a".to_owned(),
            Params::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 1.0,
            },
            0.0,
            Op::Delete,
        ));
        let entries = session.doc.log.len();
        session.needs_apply = false;

        session.set_solo(Some("r-1"));
        assert_eq!(session.solo(), Some("r-1"));
        assert!(session.needs_apply, "solo has to reach the render");
        assert_eq!(session.doc.log.len(), entries, "and nothing else");

        // Removing the soloed region takes solo off with it, or the panel would
        // show "SOLO is on" over a region that no longer exists.
        session.selected = Some("r-1".to_owned());
        session.delete_selected();
        session.set_solo(None);
        assert_eq!(session.solo(), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_region_can_be_selected_by_name_for_the_headless_flags() {
        let (dir, mut session) = a_session("select-by-name");
        session.add(Region::new(
            "r-xyz".to_owned(),
            "left blob".to_owned(),
            Params::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 1.0,
            },
            0.0,
            Op::Delete,
        ));
        assert!(session.select_region("left blob"));
        assert_eq!(session.selected_id(), Some("r-xyz"));
        assert!(session.select_region("r-xyz"));
        assert!(!session.select_region("nothing of the sort"));

        // `--move-region` goes through the same translate the gizmo does.
        assert!(session.move_selected_region([0.5, 0.0, 0.0]));
        assert!(matches!(
            session.doc.region("r-xyz").unwrap().params,
            Params::Sphere { center, .. } if (center[0] - 0.5).abs() < 1e-12
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_mix_slider_drag_is_one_undo_step_and_a_later_edit_ends_it() {
        let (dir, mut session) = a_session("mix-coalesce");
        session.add(Region::new(
            "r-m".to_owned(),
            "sphere".to_owned(),
            Params::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 1.0,
            },
            1.0,
            Op::Blend,
        ));
        let after_add = session.doc.log.len();
        for mix in [0.9, 0.6, 0.3, 0.1] {
            session.set_mix("r-m", mix);
        }
        assert_eq!(session.doc.log.len(), after_add + 1, "one drag, one entry");
        session.undo_once();
        assert!(
            (session.doc.region("r-m").unwrap().mix - 1.0).abs() < 1e-12,
            "Cmd-Z returns the mix to what it was before the fiddle"
        );
        session.redo();

        // Another edit ends the run, so the NEXT drag is its own step.
        session.selected = Some("r-m".to_owned());
        session.nudge_selected([0.1, 0.0, 0.0]);
        session.set_mix("r-m", 0.4);
        assert_eq!(session.doc.log.len(), after_add + 3);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_pointset_region_cannot_be_nudged_or_resized() {
        let dir = std::env::temp_dir().join(format!("trips-edit-pts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut session = EditSession::open_with_path(&dir, "trippy-bundle-1", None);
        session.add(Region::new(
            "r-p".to_owned(),
            "shade".to_owned(),
            Params::Pointset {
                point_ids: vec![1, 2, 3],
            },
            0.0,
            Op::Fade,
        ));
        let before = session.doc.log.len();
        session.nudge_selected([1.0, 1.0, 1.0]);
        session.resize_selected(2.0);
        assert_eq!(session.doc.log.len(), before, "no meaningless log entries");
        std::fs::remove_dir_all(&dir).ok();
    }
}
