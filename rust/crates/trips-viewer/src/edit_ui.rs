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
use trips_viewer::edit::cluster::{
    self, ClickCamera, ClickParams, ClickSelection, PointGrid, DEFAULT_MAX_POINTS,
};
use trips_viewer::edit::model::{new_region_id, LidParams, Op, Params, Region, KAREKARE_LID};
use trips_viewer::edit::shade::{self, ShadeSelection, ShadeViews, Thresholds};
use trips_viewer::edit::weights::{compose_gaussian_weights, compose_trips_weights, widen};
use trips_viewer::edit::{EditDocument, DEFAULT_REGION_SCENE_FRACTION, EDITS_FILENAME, NUDGE_SCENE_FRACTION, RESIZE_STEP};
use trips_viewer::renderer::Renderer;

/// The editor's key bindings, shown in the panel and in `docs/USER_GUIDE.md`.
///
/// Chosen to avoid every key `app.rs::ViewerApp::handle_input` already binds
/// (`V X B Tab - = F R N P W A S D Q E`), per `docs/EDITOR.md` §4.
pub const KEYS_HELP: &str = "\
M edit mode | T cycle tool | H preview highlight | SHIFT-CLICK the render to select\n\
arrows + PageUp/PageDown nudge the selected region | [ / ] shrink / grow\n\
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
}

impl Tool {
    /// Cycle order for the `T` key.
    const fn next(self) -> Self {
        match self {
            Self::Regions => Self::ShadeFinder,
            Self::ShadeFinder => Self::ClickCluster,
            Self::ClickCluster => Self::Regions,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Regions => "regions",
            Self::ShadeFinder => "shade-cloud finder",
            Self::ClickCluster => "click-to-cluster",
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
            self.last_apply_ms = started.elapsed().as_secs_f64() * 1e3;
            return Ok(());
        }

        // The borrow of `renderer` ends with this block; `edited` owns its data.
        let (edited, gaussian_needed) = {
            let base = renderer.base_points();
            let composed = compose_trips_weights(&self.doc, &widen(&base.xyz));
            let gaussian_needed = composed.delete_mask.iter().any(|d| *d);
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
                    let composed = compose_gaussian_weights(&self.doc, &widen(&means));
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
    fn nudge_selected(&mut self, delta: [f64; 3]) {
        let Some(region) = self
            .selected
            .as_ref()
            .and_then(|id| self.doc.region(id))
            .cloned()
        else {
            return;
        };
        let moved = match region.params.clone() {
            Params::Box {
                center,
                half_extents,
                quat,
            } => Params::Box {
                center: add3(center, delta),
                half_extents,
                quat,
            },
            Params::Sphere { center, radius } => Params::Sphere {
                center: add3(center, delta),
                radius,
            },
            Params::Lid(lid) => Params::Lid(LidParams {
                center: add3(lid.center, delta),
                ..lid
            }),
            // A pointset has no centre to move; nudging it would have to move
            // the points themselves, which is not an edit this tool makes.
            Params::Pointset { .. } => return,
        };
        self.set_params(&region.id, &moved);
    }

    /// Scale the selected region's size by `factor`.
    fn resize_selected(&mut self, factor: f64) {
        let Some(region) = self
            .selected
            .as_ref()
            .and_then(|id| self.doc.region(id))
            .cloned()
        else {
            return;
        };
        let resized = match region.params.clone() {
            Params::Box {
                center,
                half_extents,
                quat,
            } => Params::Box {
                center,
                half_extents: [
                    half_extents[0] * factor,
                    half_extents[1] * factor,
                    half_extents[2] * factor,
                ],
                quat,
            },
            Params::Sphere { center, radius } => Params::Sphere {
                center,
                radius: radius * factor,
            },
            Params::Lid(lid) => Params::Lid(LidParams {
                radius: lid.radius * factor,
                ..lid
            }),
            Params::Pointset { .. } => return,
        };
        self.set_params(&region.id, &resized);
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
            if self.tool == Tool::ClickCluster {
                self.click.preview = !self.click.preview;
            } else {
                self.shade.preview = !self.shade.preview;
            }
            self.needs_apply = true;
        }
        if remove {
            self.delete_selected();
        }
        if resize > 0.0 {
            self.resize_selected(resize);
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
                let result = self.doc.update_region(&region.id, json!({ "mix": mix }));
                self.record(result);
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
            for tool in [Tool::Regions, Tool::ShadeFinder, Tool::ClickCluster] {
                ui.selectable_value(&mut self.tool, tool, tool.label());
            }
        });
        if previous == Tool::ClickCluster && self.tool != previous {
            // Leaving the tool drops its index: a `f64` copy of the cloud is
            // not something to hold for a tool nobody is using (`ClickCache`).
            self.click.cache = None;
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

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
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

    #[test]
    fn the_tool_key_cycles_and_returns() {
        let mut tool = Tool::default();
        assert_eq!(tool, Tool::Regions);
        tool = tool.next();
        assert_eq!(tool, Tool::ShadeFinder);
        tool = tool.next();
        assert_eq!(tool, Tool::ClickCluster);
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
