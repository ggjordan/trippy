//! Loading a trippy asset bundle: points, weights and a scene manifest.
//!
//! Module: `trips_viewer::bundle`
//! Purpose: one directory that fully describes a renderable TRIPS scene, so
//!     the viewer takes a single path (from `argv` or a file dialog) and needs
//!     nothing else. Written by `trippy export-bundle`
//!     (`trippy/render/bundle.py`); this is the Rust half of that contract and
//!     the only place the on-disk schema is spelled out on this side.
//! Invariants:
//!     - `xyz` in `points.npz` is **world space**, not the camera-space,
//!       pre-distorted form `tools/export_unet_safetensors.py horse-e2e`
//!       writes for a single view. The lens distortion that export bakes into
//!       the points lives in each view's `distortion` array instead and is
//!       applied by [`brush_pyramid`]'s projection, so a free-flying camera
//!       and the reference view both come out right.
//!     - `params` deserialises straight into [`PyramidParams`], which has
//!       serde defaults for every v0.4.0 performance lever — so a bundle
//!       written before the levers existed loads as the *exact* pipeline.
//!     - Anything unrecognised in the manifest is an error, not a warning: a
//!       bundle whose `format` this build does not know is refused rather than
//!       rendered wrongly.
//! Units: `fx`/`fy`/`cx`/`cy` and `width`/`height` are pixels; `R`/`t` are
//!     world-to-camera with `R` row-major (`docs/GEOMETRY.md`); `distortion`
//!     is Saiga's 8-parameter order `k1 k2 k3 k4 k5 k6 p1 p2`.
//! Related docs: `docs/USER_GUIDE.md`; `docs/GEOMETRY.md`;
//!     `docs/decisions/ADR-0006-viewer-integration.md`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brush_pyramid::params::PyramidParams;
use brush_pyramid::scene::{Camera, PointSet};
use brush_unet::Weights;
use serde::Deserialize;

/// The only manifest version this build understands.
pub const BUNDLE_FORMAT: &str = "trippy-bundle-1";

/// The manifest file's name inside a bundle directory.
pub const MANIFEST_NAME: &str = "bundle.json";

/// One dataset view: a real camera the scene was captured from.
#[derive(Debug, Clone, Deserialize)]
pub struct BundleView {
    /// Dataset image index, and the tone mapper's frame index.
    pub index: usize,
    /// Source image file name, shown in the UI.
    #[serde(default)]
    pub name: String,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Focal length x, pixels.
    pub fx: f32,
    /// Focal length y, pixels.
    pub fy: f32,
    /// Principal point x, pixels.
    pub cx: f32,
    /// Principal point y, pixels.
    pub cy: f32,
    /// World-to-camera rotation, row-major 3x3.
    #[serde(rename = "R")]
    pub r: [f32; 9],
    /// World-to-camera translation.
    #[serde(rename = "t")]
    pub t: [f32; 3],
    /// Saiga 8-parameter lens distortion; all zeros for a plain pinhole.
    #[serde(default)]
    pub distortion: [f32; 8],
}

impl BundleView {
    /// This view exactly as the rasteriser's camera, at its own resolution.
    #[must_use]
    pub fn camera(&self) -> Camera {
        Camera {
            width: self.width,
            height: self.height,
            fx: self.fx,
            fy: self.fy,
            cx: self.cx,
            cy: self.cy,
            r: self.r,
            t: self.t,
            distortion: self.distortion,
        }
    }

    /// Camera centre in world coordinates, `-R^T t`.
    #[must_use]
    pub fn position(&self) -> glam::Vec3 {
        let r = &self.r;
        let t = &self.t;
        glam::Vec3::new(
            -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
            -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
            -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]),
        )
    }
}

/// `bundle.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    /// Must equal [`BUNDLE_FORMAT`].
    pub format: String,
    /// Human-readable scene name, shown in the window title.
    #[serde(default)]
    pub name: String,
    /// Point-set file name, relative to the bundle directory.
    pub points: String,
    /// Weight file name, relative to the bundle directory.
    pub weights: String,
    /// `C`, the feature width. Cross-checked against `points.npz`.
    pub num_channels: usize,
    /// `C` background feature values, composited as `out += t_final * bg`.
    #[serde(default)]
    pub background: Vec<f32>,
    /// Rasteriser settings; every performance lever defaults to exact.
    pub params: PyramidParams,
    /// Scene up vector in world coordinates, for the orbit controls.
    #[serde(default = "default_up")]
    pub up: [f32; 3],
    /// The blend gate + precomputed splat renders, when this bundle was
    /// exported from a run trained with one.
    ///
    /// `#[serde(default)]`, so every bundle written before the gate existed
    /// still parses -- which is why [`BUNDLE_FORMAT`] does not change.
    #[serde(default)]
    pub blend: Option<BlendManifest>,
    /// The scene directory the views were captured in, when the exporter
    /// recorded one (`trippy.render.bundle.bundle_document`'s `scene_root`).
    ///
    /// The viewer never opens a photograph (`AGENTS.md` §6) and nothing in the
    /// render path reads this. It exists for the SAM tool, which spawns
    /// `trippy edits sam` and lets the CHILD find the photographs from here
    /// rather than being told twice — which is why `--scene` is optional on
    /// that command and absent from `crate::sam_child`'s argument list.
    /// `#[serde(default)]`: a bundle exported before 2026-09-07 has no such key
    /// and the SAM panel says so instead of guessing.
    #[serde(default)]
    pub scene_root: Option<String>,
    /// The trippy checkout that wrote this bundle, when the exporter recorded
    /// one (`trippy.render.bundle.trippy_repo_root`).
    ///
    /// `crate::sam_child` resolves the interpreter it spawns as
    /// `<trippy_root>/.venv/bin/python`; `$TRIPPY_ROOT` and `$TRIPPY_PYTHON`
    /// override it, which is what makes a bundle copied off this machine
    /// usable. `#[serde(default)]` for the same reason as `scene_root`.
    #[serde(default)]
    pub trippy_root: Option<String>,
    /// Index **into `views`** (not a dataset image index) the viewer opens at.
    #[serde(default)]
    pub default_view: usize,
    /// Every capture view, in dataset order.
    pub views: Vec<BundleView>,
}

/// TRIPS/ADOP scenes are Y-down, so the default matches `dataset.ini`'s
/// `up_vector = 0 -1 0` rather than a graphics-conventional `+Y`.
const fn default_up() -> [f32; 3] {
    [0.0, -1.0, 0.0]
}

/// A loaded bundle: the manifest plus everything it points at, in memory.
pub struct Bundle {
    /// Where it was loaded from.
    pub dir: PathBuf,
    /// The parsed manifest.
    pub manifest: Manifest,
    /// The world-space point set.
    pub points: PointSet,
    /// The U-Net + tone-mapper weights.
    pub weights: Weights,
    /// Precomputed Gaussian-splat renders, keyed by **dataset image index**
    /// (the same number [`BundleView::index`] carries and the tone mapper uses
    /// as its frame index), so the renderer can look one up from the frame it
    /// is drawing without consulting the manifest.
    ///
    /// Empty on any bundle without a `blend` block, and also on one whose
    /// `splat.npz` could not be read -- the Blend panel then says so rather
    /// than the scene refusing to open.
    pub splats: HashMap<usize, SplatImage>,
}

/// The `blend` block of `bundle.json` (see `trippy.render.bundle.BlendInfo`).
#[derive(Debug, Clone, Deserialize)]
pub struct BlendManifest {
    /// Whether `weights.safetensors` carries the gate head.
    #[serde(default)]
    pub gate: bool,
    /// Output channel of the gate logit; always [`brush_unet::GATE_CHANNEL`].
    #[serde(default = "default_gate_channel")]
    pub gate_channel: usize,
    /// The run's own default gate scale; the Blend panel's slider starts here.
    #[serde(default = "default_gate_scale")]
    pub gate_scale: f32,
    /// Absolute path of the Gaussian PLY the splat half came from, or "".
    ///
    /// **Not yet read.** It is what the live splat path will open (Brush's
    /// `brush-render`, rendering the PLY at the viewer's own pose); until that
    /// lands the viewer shows the precomputed renders below and says so. See
    /// `docs/USER_GUIDE.md` "Blend panel".
    #[serde(default)]
    pub splat_ply: String,
    /// File name of the precomputed render archive, or "" when none.
    #[serde(default)]
    pub splat_renders: String,
    /// Array positions into `views` that have a precomputed render.
    #[serde(default)]
    pub splat_views: Vec<usize>,
    /// Fraction of each view's own size the stored render was scaled to.
    #[serde(default = "default_splat_scale")]
    pub splat_scale: f32,
    /// Whether the stored rgb is already alpha-masked.
    #[serde(default)]
    pub mask_by_alpha: bool,
    /// The Gaussian block's channel groups, in `HYBRID_A_CHANNEL_ORDER`
    /// (`rgb`, `alpha`, `depth`), exactly as the network was trained with.
    /// Their total width is `G` and the U-Net's `in_channels` is `C + G`.
    #[serde(default)]
    pub channels: Vec<String>,
    /// `"all_levels"` or `"concat_level0"`: how the block reaches the pyramid's
    /// levels. Pooling it to every level on a `concat_level0` run would feed the
    /// network something it was never trained on.
    #[serde(default = "default_block_mode")]
    pub mode: String,
    /// `C`, the TRIPS feature width, so `C + G` can be checked against the
    /// weight file's own `in_channels`.
    #[serde(default)]
    pub feature_channels: usize,
}

impl BlendManifest {
    /// `G`: the Gaussian block's total width in channels.
    #[must_use]
    pub fn block_channels(&self) -> usize {
        self.channels.iter().map(|g| group_width(g)).sum()
    }

    /// True when the block is pooled onto every pyramid level.
    #[must_use]
    pub fn all_levels(&self) -> bool {
        self.mode != "concat_level0"
    }
}

/// Channel width of one `HYBRID_A_CHANNEL_ORDER` group (0 for an unknown one).
fn group_width(group: &str) -> usize {
    match group {
        "rgb" => 3,
        "alpha" | "depth" => 1,
        _ => 0,
    }
}

fn default_block_mode() -> String {
    "all_levels".to_owned()
}

fn default_gate_channel() -> usize {
    brush_unet::GATE_CHANNEL
}

const fn default_gate_scale() -> f32 {
    1.0
}

const fn default_splat_scale() -> f32 {
    1.0
}

/// One view's precomputed Gaussian block, on the host.
///
/// Every plane is stored **planar CHW** because that is the layout the network's
/// own output and its pyramid inputs use, so nothing here needs a transpose.
/// This is both halves of the feature at once: `rgb` is the Blend panel's splat
/// image, and the whole block is what the design-A U-Net's extra input channels
/// are fed (see [`BlendManifest`] -- without it a hybrid bundle cannot render).
#[derive(Debug, Clone, Default)]
pub struct SplatImage {
    /// `3 * height * width` values in [0, 1], planar, ALREADY alpha-masked when
    /// the manifest says so. Empty when the run's block has no `rgb` group.
    pub rgb: Vec<f32>,
    /// `height * width` coverage values in [0, 1]; empty when not stored.
    pub alpha: Vec<f32>,
    /// `height * width` NORMALISED depths (already divided by the run's
    /// `depth_scale`); empty when not stored.
    pub depth: Vec<f32>,
    /// Rows.
    pub height: usize,
    /// Columns.
    pub width: usize,
}

impl SplatImage {
    /// The splat colour the blend mixes towards: `rgb`, alpha-applied.
    ///
    /// Mirrors `trippy.hybrid.gate.splat_rgb_from_block`: when the block's rgb
    /// is already masked this is `rgb` verbatim, otherwise alpha is applied
    /// here, so the gate always mixes in "the splat where the splat exists"
    /// regardless of the run's `mask_by_alpha` ablation.
    #[must_use]
    pub fn masked_rgb(&self, mask_by_alpha: bool) -> Vec<f32> {
        if mask_by_alpha || self.alpha.is_empty() || self.rgb.is_empty() {
            return self.rgb.clone();
        }
        let pixels = self.height * self.width;
        let mut out = self.rgb.clone();
        for c in 0..3 {
            for i in 0..pixels {
                out[c * pixels + i] *= self.alpha[i];
            }
        }
        out
    }

    /// The whole `(G, height, width)` block, planar, in `channels` order,
    /// resampled to `(h, w)`.
    ///
    /// The Rust twin of `trippy.hybrid.gaussian_input.block_from_arrays` +
    /// `resample_to`: same group order, same normalisation, so the network sees
    /// what it was trained on. A group with no stored plane contributes zeros --
    /// design A's own "no Gaussian information here" state, not a fabrication.
    #[must_use]
    pub fn block(&self, channels: &[String], mask_by_alpha: bool, h: usize, w: usize) -> Vec<f32> {
        let mut out = Vec::new();
        for group in channels {
            match group.as_str() {
                "rgb" => out.extend(self.resample_planes(&self.masked_rgb(mask_by_alpha), 3, h, w)),
                "alpha" => out.extend(self.resample_planes(&self.alpha, 1, h, w)),
                "depth" => out.extend(self.resample_planes(&self.depth, 1, h, w)),
                _ => {}
            }
        }
        out
    }

    /// Nearest-neighbour resample of `planes` (planar, `count` of them) to `(h, w)`.
    ///
    /// An empty `planes` gives zeros, which is exactly how the trainer represents
    /// "no Gaussian information at this pixel".
    fn resample_planes(&self, planes: &[f32], count: usize, h: usize, w: usize) -> Vec<f32> {
        if planes.is_empty() {
            return vec![0.0; count * h * w];
        }
        if h == self.height && w == self.width {
            return planes.to_vec();
        }
        let mut out = vec![0.0f32; count * h * w];
        for c in 0..count {
            let src = c * self.height * self.width;
            let dst = c * h * w;
            for y in 0..h {
                let sy = (y * self.height) / h;
                for x in 0..w {
                    let sx = (x * self.width) / w;
                    out[dst + y * w + x] = planes[src + sy * self.width + sx];
                }
            }
        }
        out
    }
    /// Nearest-neighbour resample to `(height, width)`, planar CHW throughout.
    ///
    /// Nearest, not bilinear, on purpose: the stored render is already a
    /// downscale of the run's own resolution ([`BlendManifest::splat_scale`]),
    /// and the panel's job is to show *where* the splat and TRIPS disagree.
    /// Inventing intermediate colours on the way back up would soften exactly
    /// the edges that answer that question.
    #[must_use]
    pub fn resampled(&self, height: usize, width: usize) -> Vec<f32> {
        self.resample_planes(&self.rgb, 3, height, width)
    }
}

/// Decode `splat.npz` into `{dataset index: SplatImage}`.
///
/// The archive holds, per array position `p` in [`BlendManifest::splat_views`]:
/// `view_<p>` (H, W, 3) uint8 rgb, `alpha_<p>` (H, W) uint8, and `depth_<p>`
/// (H, W) **float32** -- the last one f32 rather than u8 because the normalised
/// depth is unbounded and the network was trained on it directly. Only the
/// groups [`BlendManifest::channels`] lists are present.
///
/// A member that is missing or mis-shaped is skipped, not fatal: a partial
/// archive should grey out those views' Blend controls (and feed the network
/// zeros in those channels, which is design A's own honest "no Gaussian
/// information here" state), not refuse the scene.
///
/// # Errors
/// Returns `Err` only when the archive itself cannot be parsed.
pub fn decode_splats(
    bytes: &[u8],
    blend: &BlendManifest,
    views: &[BundleView],
    origin: &str,
) -> Result<HashMap<usize, SplatImage>, String> {
    let archive = brush_pyramid::npz::read_npz_bytes(bytes)
        .map_err(|e| format!("{origin}: {e}"))?;
    let mut out = HashMap::new();
    for &position in &blend.splat_views {
        let Some(view) = views.get(position) else {
            continue;
        };
        let mut image = SplatImage::default();

        // rgb: (H, W, 3) uint8 -> planar (3, H, W) in [0, 1]. It also fixes the
        // block's (H, W) for every other plane.
        if let Some(array) = archive.get(&format!("view_{position}")) {
            if array.shape.len() == 3 && array.shape[2] == 3 {
                let (height, width) = (array.shape[0], array.shape[1]);
                if let Ok(values) = array.to_f32() {
                    if values.len() == height * width * 3 {
                        let mut rgb = vec![0.0f32; values.len()];
                        for c in 0..3 {
                            let plane = c * height * width;
                            for i in 0..height * width {
                                rgb[plane + i] = values[i * 3 + c] / 255.0;
                            }
                        }
                        image.rgb = rgb;
                        image.height = height;
                        image.width = width;
                    }
                }
            }
        }
        if image.height == 0 || image.width == 0 {
            continue;
        }

        // alpha: uint8 -> [0, 1]. depth: already f32 and already normalised, so
        // it is taken verbatim -- scaling it here would be a second division by
        // `depth_scale`.
        let pixels = image.height * image.width;
        for (name, scale, target) in [
            (format!("alpha_{position}"), 1.0 / 255.0, 0usize),
            (format!("depth_{position}"), 1.0, 1usize),
        ] {
            let Some(array) = archive.get(&name) else { continue };
            if array.shape != vec![image.height, image.width] {
                continue;
            }
            let Ok(values) = array.to_f32() else { continue };
            if values.len() != pixels {
                continue;
            }
            let plane: Vec<f32> = values.iter().map(|v| v * scale).collect();
            if target == 0 {
                image.alpha = plane;
            } else {
                image.depth = plane;
            }
        }
        out.insert(view.index, image);
    }
    Ok(out)
}

impl Bundle {
    /// Load every file in `dir`.
    ///
    /// # Arguments
    /// - `dir`: a directory holding `bundle.json` and the two files it names.
    ///
    /// # Errors
    /// Returns `Err` if the manifest is missing, is not [`BUNDLE_FORMAT`], has
    /// no views, or disagrees with the point set about `C`.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let manifest_path = dir.join(MANIFEST_NAME);
        let origin = manifest_path.display().to_string();
        let text = std::fs::read_to_string(&manifest_path).map_err(|e| format!("{origin}: {e}"))?;
        // Read the manifest first so only the two files it actually names are
        // touched -- the bundle directory is not assumed to hold anything else.
        let manifest = Self::parse_manifest(&text, &origin)?;
        let points_bytes = std::fs::read(dir.join(&manifest.points))
            .map_err(|e| format!("{}: {e}", dir.join(&manifest.points).display()))?;
        let weight_bytes = std::fs::read(dir.join(&manifest.weights))
            .map_err(|e| format!("{}: {e}", dir.join(&manifest.weights).display()))?;
        // The splat archive is OPTIONAL in the strongest sense: an unreadable or
        // absent one greys out the Blend panel and leaves everything else working.
        let splat_bytes = manifest
            .blend
            .as_ref()
            .filter(|b| !b.splat_renders.is_empty())
            .and_then(|b| std::fs::read(dir.join(&b.splat_renders)).ok());
        Self::from_parts_with_splats(
            dir.to_path_buf(),
            manifest,
            &points_bytes,
            &weight_bytes,
            splat_bytes.as_deref(),
            &origin,
        )
    }

    /// Parse and validate `bundle.json`, without touching the files it names.
    ///
    /// Split out so the web viewer can read the manifest, learn the two file
    /// names, and only then spend 80 MB of `fetch` on them.
    ///
    /// # Arguments
    /// - `text`: the manifest's JSON.
    /// - `origin`: what to name in error messages (a path or a URL).
    ///
    /// # Errors
    /// Returns `Err` if the JSON does not parse, the format is not
    /// [`BUNDLE_FORMAT`], or there are no views.
    pub fn parse_manifest(text: &str, origin: &str) -> Result<Manifest, String> {
        let manifest: Manifest =
            serde_json::from_str(text).map_err(|e| format!("{origin}: {e}"))?;
        if manifest.format != BUNDLE_FORMAT {
            return Err(format!(
                "{origin}: format is {:?}, this build only reads {BUNDLE_FORMAT:?}",
                manifest.format
            ));
        }
        if manifest.views.is_empty() {
            return Err(format!("{origin}: no views"));
        }
        Ok(manifest)
    }

    /// Assemble a bundle from bytes that are already in memory.
    ///
    /// This is the **web** entry point: the browser has no filesystem, so
    /// `trips-web` fetches `points.npz` and `weights.safetensors` over
    /// loopback and calls this. Every cross-check [`Self::load`] performs is
    /// performed here — the two paths differ only in where the bytes came
    /// from, which is what makes the browser's frame comparable with the
    /// native screenshot.
    ///
    /// # Arguments
    /// - `dir`: what to report as the bundle's location (a URL, on the web).
    /// - `manifest`: from [`Self::parse_manifest`].
    /// - `points_bytes`: the whole `points.npz`.
    /// - `weight_bytes`: the whole `weights.safetensors`.
    /// - `origin`: what to name in error messages.
    ///
    /// # Errors
    /// Returns `Err` if either file fails to parse or disagrees with the
    /// manifest about `C`.
    pub fn from_parts(
        dir: PathBuf,
        manifest: Manifest,
        points_bytes: &[u8],
        weight_bytes: &[u8],
        origin: &str,
    ) -> Result<Self, String> {
        Self::from_parts_with_splats(dir, manifest, points_bytes, weight_bytes, None, origin)
    }

    /// [`Self::from_parts`], plus the optional precomputed splat archive.
    ///
    /// Separate entry point rather than a changed signature so `trips-web`
    /// (which fetches exactly the files the manifest names) keeps compiling
    /// unchanged and can opt in to the Blend panel by fetching one more.
    ///
    /// # Arguments
    /// - `splat_bytes`: the whole `splat.npz`, or `None`.
    ///
    /// # Errors
    /// As [`Self::from_parts`]. A `splat_bytes` that fails to decode is
    /// **not** an error: it leaves `splats` empty.
    pub fn from_parts_with_splats(
        dir: PathBuf,
        manifest: Manifest,
        points_bytes: &[u8],
        weight_bytes: &[u8],
        splat_bytes: Option<&[u8]>,
        origin: &str,
    ) -> Result<Self, String> {
        let points = PointSet::from_npz_bytes(points_bytes, &manifest.points)?;
        if points.num_channels != manifest.num_channels {
            return Err(format!(
                "{origin}: manifest says C = {}, {} has C = {}",
                manifest.num_channels, manifest.points, points.num_channels
            ));
        }
        if !manifest.background.is_empty() && manifest.background.len() != points.num_channels {
            return Err(format!(
                "{origin}: background has {} values, expected {}",
                manifest.background.len(),
                points.num_channels
            ));
        }
        let weights = Weights::from_bytes(weight_bytes)?;

        let splats = match (&manifest.blend, splat_bytes) {
            (Some(blend), Some(bytes)) => {
                decode_splats(bytes, blend, &manifest.views, origin).unwrap_or_default()
            }
            _ => HashMap::new(),
        };

        Ok(Self {
            dir,
            manifest,
            points,
            weights,
            splats,
        })
    }

    /// Array position of the view the viewer opens at, and the one the `R`
    /// key returns to.
    ///
    /// The manifest's own `default_view` when it is in range. When it is not,
    /// the view **nearest the centre of the camera box** rather than the last
    /// one a clamp would land on: a broken manifest should still open on
    /// something that looks at the scene.
    #[must_use]
    pub fn home_view_position(&self) -> usize {
        let views = &self.manifest.views;
        if self.manifest.default_view < views.len() {
            self.manifest.default_view
        } else {
            SceneScale::most_central_view(views).unwrap_or(0)
        }
    }

    /// The background feature vector, or `None` for a zero background.
    #[must_use]
    pub fn background(&self) -> Option<&[f32]> {
        (!self.manifest.background.is_empty()).then_some(self.manifest.background.as_slice())
    }

    /// The point cloud's axis-aligned world-space bounds, `(min, max)`.
    ///
    /// Computed once, on load, by one pass over `xyz`. Used for the fly speed
    /// and for [`Bounds::depth_span`].
    #[must_use]
    pub fn bounds(&self) -> Bounds {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for chunk in self.points.xyz.chunks_exact(3) {
            for axis in 0..3 {
                min[axis] = min[axis].min(chunk[axis]);
                max[axis] = max[axis].max(chunk[axis]);
            }
        }
        if !min[0].is_finite() {
            // An empty point set: any finite box will do.
            min = [0.0; 3];
            max = [1.0; 3];
        }
        Bounds { min, max }
    }
}

/// The scene's axis-aligned world-space bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    /// Minimum corner, world units.
    pub min: [f32; 3],
    /// Maximum corner, world units.
    pub max: [f32; 3],
}

/// Multiplied into the depth span so the far end is never exactly the last
/// bucket edge, where a rounding difference between host and device could
/// clamp two distinguishable depths together.
const DEPTH_SPAN_SLACK: f32 = 1.05;

/// Smallest `hi / lo` the span is allowed to collapse to. Only bites when the
/// camera is looking edge-on at a flat scene; a wider floor would waste depth
/// buckets on empty space in the ordinary case, which is the opposite of what
/// this function is for.
const MIN_SPAN_RATIO: f32 = 1.01;

impl Bounds {
    /// The box's centre, world units.
    #[must_use]
    pub fn centre(&self) -> glam::Vec3 {
        glam::Vec3::new(
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        )
    }

    /// `point` moved to the nearest position inside the box.
    #[must_use]
    pub fn clamp_point(&self, point: glam::Vec3) -> glam::Vec3 {
        let p = point.to_array();
        let mut out = [0.0f32; 3];
        for axis in 0..3 {
            out[axis] = p[axis].clamp(self.min[axis], self.max[axis]);
        }
        glam::Vec3::from(out)
    }

    /// Is `point` inside the box (inclusive)?
    #[must_use]
    pub fn contains(&self, point: glam::Vec3) -> bool {
        let p = point.to_array();
        (0..3).all(|axis| p[axis] >= self.min[axis] && p[axis] <= self.max[axis])
    }

    /// The box scaled by `factor` about its own centre, with every half-extent
    /// grown by at least `(factor - 1)` times the half-extent a cube of the
    /// same diagonal would have.
    ///
    /// Used for the "you have flown out of the scene" test: a camera outside
    /// `camera::LOST_BOX_FACTOR` times the capture box is somewhere no training view
    /// ever saw, so the viewer offers the reset key rather than pretending the
    /// black frame is a render.
    ///
    /// That floor is what makes the test usable on a real capture. A rig walked
    /// round a subject at roughly one height -- the horse scene's cameras fill
    /// 11.6 x 1.0 x 10.4 world units -- has an almost flat axis, and a pure
    /// scaling would call the camera lost after rising 1.5 units while allowing
    /// 17 sideways. For a cube the floor never bites, so `expanded` is then
    /// exactly a scaling; for `factor <= 1` (shrinking) it is disabled.
    #[must_use]
    pub fn expanded(&self, factor: f32) -> Self {
        let centre = self.centre().to_array();
        let floor = 0.5 * (factor - 1.0).max(0.0) * self.diameter() / 3.0f32.sqrt();
        let mut min = [0.0f32; 3];
        let mut max = [0.0f32; 3];
        for axis in 0..3 {
            let half = (0.5 * (self.max[axis] - self.min[axis]) * factor).max(floor);
            min[axis] = centre[axis] - half;
            max[axis] = centre[axis] + half;
        }
        Self { min, max }
    }

    /// The box's diagonal length, world units.
    #[must_use]
    pub fn diameter(&self) -> f32 {
        ((self.max[0] - self.min[0]).powi(2)
            + (self.max[1] - self.min[1]).powi(2)
            + (self.max[2] - self.min[2]).powi(2))
        .sqrt()
        .max(f32::MIN_POSITIVE)
    }

    /// The camera-space depth range this box occupies, for
    /// [`brush_pyramid::params::DepthRange`].
    ///
    /// Transforming eight corners on the host is a rounding error's worth of
    /// work and gives a range tight around what the camera can actually see —
    /// which matters, because the packed sort key has only ten bits of depth
    /// and spends them evenly across whatever range it is given. Measuring the
    /// frame's *real* depths would need a device reduction and a readback per
    /// frame, i.e. exactly the host sync the lever exists to remove.
    ///
    /// # Arguments
    /// - `camera`: this frame's camera.
    /// - `znear`: the render's near plane, world units; the returned `lo` is
    ///   never below it, because nothing nearer is drawn.
    ///
    /// # Returns
    /// `(lo, hi)` with `0 < lo < hi`.
    #[must_use]
    pub fn depth_span(&self, camera: &Camera, znear: f32) -> (f32, f32) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for corner in 0..8 {
            let pick = |axis: usize| {
                if corner & (1 << axis) == 0 {
                    self.min[axis]
                } else {
                    self.max[axis]
                }
            };
            let (_, _, z) = camera.world_to_cam(pick(0), pick(1), pick(2));
            lo = lo.min(z);
            hi = hi.max(z);
        }
        // The camera may sit inside or behind the box, in which case the near
        // end is the near plane and only the far end is informative.
        let near = lo.max(znear).max(f32::MIN_POSITIVE);
        let far = (hi * DEPTH_SPAN_SLACK).max(near * MIN_SPAN_RATIO);
        (near, far)
    }
}

/// How the viewer's units are derived from a bundle: the box the capture
/// cameras occupy and how far apart consecutive ones are.
///
/// This exists because the **point cloud** is not a scene scale. A TRIPS export
/// carries an environment sphere of far-field points — on the horse bundle that
/// sphere is 7500 units across each axis (a 12 990-unit box diagonal), while
/// every capture camera fits in a box 15.6 units across — so a fly speed
/// derived from the point bounding box is out by three orders of magnitude and
/// one key tap leaves the scene. That is exactly the "fly 1948.53 u/s" Jordan
/// was given on 2026-09-06. The cameras are the honest ruler: they are where a
/// person could actually stand.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneScale {
    /// Axis-aligned box containing every capture camera centre, world units.
    pub bounds: Bounds,
    /// Median distance between *consecutive* views in dataset order, world
    /// units. Median rather than mean because capture sequences routinely
    /// contain one huge jump between passes, which a mean would follow.
    pub median_spacing: f32,
}

/// Default fly speed, as a multiple of [`SceneScale::median_spacing`] per
/// second: one second of held `W` crosses two capture positions.
///
/// Was 0.5 (half a gap per second) until 2026-09-06. That is a step rather
/// than a teleport, which was the right instinct and the wrong number: on
/// kk-coherent the median gap between consecutive photographs is ~0.2 world
/// units, so half a gap per second crossed the 15-unit capture area in about
/// two and a half minutes of held `W` and Jordan reported he "moves so slow he
/// can't explore the areas he wants". At 2.0 the same traverse is ~40 s, and
/// the scroll wheel reaches [`crate::camera::MAX_SPEED_SCALE`] x that.
pub const BASE_SPEED_FRACTION: f32 = 2.0;

/// Fallback spacing for a bundle whose views are all in one place: this
/// fraction of the camera box's diagonal.
const SPACING_FROM_BOX: f32 = 1.0 / 32.0;

/// Last-resort spacing, world units, when there is neither a gap nor a box
/// (a single view). Arbitrary, and only reachable from a one-view bundle.
const SPACING_FALLBACK: f32 = 1.0;

/// Below this a distance counts as zero, world units.
const TINY: f32 = 1e-6;

impl SceneScale {
    /// Measure `views`.
    ///
    /// # Arguments
    /// - `views`: every capture view, in dataset order (the order matters:
    ///   the spacing is between *consecutive* entries).
    #[must_use]
    pub fn from_views(views: &[BundleView]) -> Self {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        let mut previous: Option<glam::Vec3> = None;
        let mut steps = Vec::with_capacity(views.len());
        for view in views {
            let position = view.position();
            let p = position.to_array();
            for axis in 0..3 {
                min[axis] = min[axis].min(p[axis]);
                max[axis] = max[axis].max(p[axis]);
            }
            if let Some(before) = previous {
                let step = (position - before).length();
                if step.is_finite() {
                    steps.push(step);
                }
            }
            previous = Some(position);
        }
        if !min[0].is_finite() {
            min = [0.0; 3];
            max = [0.0; 3];
        }
        steps.sort_by(f32::total_cmp);
        let median_spacing = if steps.is_empty() {
            0.0
        } else {
            steps[steps.len() / 2]
        };
        Self {
            bounds: Bounds { min, max },
            median_spacing,
        }
    }

    /// Default fly speed for this scene, world units per second.
    #[must_use]
    pub fn base_speed(&self) -> f32 {
        let spacing = if self.median_spacing > TINY {
            self.median_spacing
        } else {
            let diagonal = self.bounds.diameter();
            if diagonal > TINY {
                diagonal * SPACING_FROM_BOX
            } else {
                SPACING_FALLBACK
            }
        };
        BASE_SPEED_FRACTION * spacing
    }

    /// The camera box's diagonal: the number the HUD reports speed against and
    /// the orbit distance is clamped by.
    ///
    /// A one-view bundle has a zero-size camera box, so this falls back to the
    /// same ruler [`Self::base_speed`] used. Without that, "scene per second"
    /// and the orbit clamps would divide by ~0.
    #[must_use]
    pub fn diameter(&self) -> f32 {
        let diagonal = self.bounds.diameter();
        if diagonal > TINY {
            diagonal
        } else {
            self.base_speed() / BASE_SPEED_FRACTION
        }
    }

    /// Array position of the view nearest the camera box's centre.
    ///
    /// Returns `None` for an empty slice.
    #[must_use]
    pub fn most_central_view(views: &[BundleView]) -> Option<usize> {
        let centre = Self::from_views(views).bounds.centre();
        views
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (a.position() - centre)
                    .length_squared()
                    .total_cmp(&(b.position() - centre).length_squared())
            })
            .map(|(position, _)| position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal manifest, as `trippy export-bundle` writes it.
    fn manifest_json(format: &str) -> String {
        format!(
            r#"{{
              "format": "{format}",
              "name": "horse",
              "points": "points.npz",
              "weights": "weights.safetensors",
              "num_points": 3,
              "num_channels": 4,
              "background": [0.34, 0.04, 0.12, 0.37],
              "params": {{
                "mode": "trips", "num_layers": 8, "pixel_center": "integer",
                "halving": "ceil", "max_frags": 16, "t_cutoff": 0.001,
                "alpha_min": 0.0, "znear": 1e-06
              }},
              "up": [0.0, -1.0, 0.0],
              "default_view": 1,
              "views": [
                {{"index": 0, "name": "00001.jpg", "width": 1920, "height": 1080,
                  "fx": 1164.46, "fy": 1164.46, "cx": 960.0, "cy": 540.0,
                  "R": [1,0,0,0,1,0,0,0,1], "t": [0,0,0],
                  "distortion": [-0.064, 0.044, 0, 0, 0, 0, 0, 0]}},
                {{"index": 8, "name": "00009.jpg", "width": 1920, "height": 1080,
                  "fx": 1164.46, "fy": 1164.46, "cx": 960.0, "cy": 540.0,
                  "R": [1,0,0,0,1,0,0,0,1], "t": [1.0,2.0,3.0],
                  "distortion": [0,0,0,0,0,0,0,0]}}
              ]
            }}"#
        )
    }

    /// `manifest_json` plus a `blend` block, as a gate run exports it.
    fn manifest_json_with_blend() -> String {
        let base = manifest_json(BUNDLE_FORMAT);
        let blend = r#""blend": {
              "gate": true, "gate_channel": 3, "gate_scale": 1.5,
              "splat_ply": "/tmp/kklid.ply", "splat_renders": "splat.npz",
              "splat_views": [0, 1], "splat_scale": 0.5, "mask_by_alpha": true
            }, "up":"#;
        base.replacen(r#""up":"#, blend, 1)
    }

    /// Load a REAL bundle directory named by `TRIPPY_TEST_BUNDLE`, if one is set.
    ///
    /// Skipped (and passing) when the variable is absent, because no bundle may
    /// be committed to this public repo -- but `scripts/test.sh` can be pointed
    /// at a freshly exported synthetic one, and the blend-panel GPU check does
    /// exactly that. This is the only test that reads a whole bundle off disk,
    /// so it is the one that would catch a Python-writer / Rust-reader
    /// disagreement about `splat.npz` before a GPU job pays for it.
    #[test]
    fn a_real_bundle_on_disk_loads_with_its_splats() {
        let Some(dir) = std::env::var("TRIPPY_TEST_BUNDLE").ok().filter(|d| !d.is_empty())
        else {
            return;
        };
        let bundle = Bundle::load(Path::new(&dir)).unwrap_or_else(|e| panic!("{e}"));
        let Some(blend) = bundle.manifest.blend.as_ref() else {
            assert!(bundle.splats.is_empty(), "no blend block, so no splats");
            return;
        };
        assert_eq!(blend.gate_channel, brush_unet::GATE_CHANNEL);
        assert_eq!(
            blend.gate,
            bundle.weights.unet.has_gate(),
            "the manifest and the weights disagree about the gate head"
        );
        // The bug this whole block exists to fix: a design-A network takes
        // `C + G` input channels, so a bundle without a wide enough block cannot
        // run its own U-Net at all.
        assert_eq!(
            bundle.manifest.num_channels + blend.block_channels(),
            bundle.weights.unet.in_channels,
            "C + G must equal the weights' in_channels"
        );
        assert_eq!(blend.feature_channels, bundle.manifest.num_channels);
        assert!(!blend.channels.is_empty());
        assert_eq!(
            bundle.splats.len(),
            blend.splat_views.len(),
            "every view the manifest lists must decode out of splat.npz"
        );
        for &position in &blend.splat_views {
            let view = &bundle.manifest.views[position];
            let image = bundle
                .splats
                .get(&view.index)
                .unwrap_or_else(|| panic!("no splat for view {}", view.index));
            assert_eq!(image.rgb.len(), 3 * image.height * image.width);
            assert!(image.rgb.iter().all(|v| (0.0..=1.0).contains(v)));
            // The reassembled block is exactly as wide as the manifest promises,
            // at whatever size a pyramid level asks for.
            let block = image.block(&blend.channels, blend.mask_by_alpha, 7, 5);
            assert_eq!(block.len(), blend.block_channels() * 7 * 5);
            assert!(block.iter().all(|v| v.is_finite()));
            // The stored render is the view's own size times `splat_scale`.
            let expected_w = (view.width as f32 * blend.splat_scale).round() as usize;
            assert!(
                image.width.abs_diff(expected_w) <= 1,
                "view {} stored at {} px, expected ~{expected_w}",
                view.index,
                image.width
            );
        }
    }

    #[test]
    fn a_bundle_without_a_blend_block_parses_exactly_as_before() {
        // The whole backwards-compatibility claim: `blend` is optional, and the
        // format tag does not change, so every bundle already on disk still opens.
        let m: Manifest = serde_json::from_str(&manifest_json(BUNDLE_FORMAT)).expect("parse");
        assert!(m.blend.is_none());
    }

    #[test]
    fn the_blend_block_parses_every_field() {
        let m: Manifest = serde_json::from_str(&manifest_json_with_blend()).expect("parse");
        let blend = m.blend.expect("blend block");
        assert!(blend.gate);
        assert_eq!(blend.gate_channel, brush_unet::GATE_CHANNEL);
        assert_eq!(blend.gate_scale, 1.5);
        assert_eq!(blend.splat_ply, "/tmp/kklid.ply");
        assert_eq!(blend.splat_renders, "splat.npz");
        assert_eq!(blend.splat_views, vec![0, 1]);
        assert_eq!(blend.splat_scale, 0.5);
        assert!(blend.mask_by_alpha);
    }

    #[test]
    fn a_blend_block_missing_optional_fields_falls_back_to_the_documented_defaults() {
        let base = manifest_json(BUNDLE_FORMAT);
        let text = base.replacen(r#""up":"#, r#""blend": {"gate": true}, "up":"#, 1);
        let m: Manifest = serde_json::from_str(&text).expect("parse");
        let blend = m.blend.expect("blend block");
        assert_eq!(blend.gate_channel, brush_unet::GATE_CHANNEL);
        assert_eq!(blend.gate_scale, 1.0);
        assert_eq!(blend.splat_scale, 1.0);
        assert!(blend.splat_views.is_empty());
        assert!(blend.splat_renders.is_empty());
    }

    #[test]
    fn a_splat_image_resamples_planar_and_nearest() {
        // 2x2 red/green/blue planes, upscaled to 4x4: every source pixel becomes a
        // 2x2 block, and no new colour appears anywhere.
        let mut rgb = Vec::new();
        for c in 0..3 {
            for i in 0..4 {
                rgb.push((c * 4 + i) as f32 / 12.0);
            }
        }
        let image = SplatImage {
            rgb,
            height: 2,
            width: 2,
            ..SplatImage::default()
        };
        let out = image.resampled(4, 4);
        assert_eq!(out.len(), 3 * 16);
        for c in 0..3 {
            for y in 0..4 {
                for x in 0..4 {
                    let expected = image.rgb[c * 4 + (y / 2) * 2 + (x / 2)];
                    assert_eq!(out[c * 16 + y * 4 + x], expected, "c{c} y{y} x{x}");
                }
            }
        }
    }

    #[test]
    fn resampling_to_the_same_size_is_the_identity() {
        let image = SplatImage {
            rgb: (0..12).map(|i| i as f32).collect(),
            height: 2,
            width: 2,
            ..SplatImage::default()
        };
        assert_eq!(image.resampled(2, 2), image.rgb);
    }

    #[test]
    fn the_manifest_parses_and_default_view_is_an_array_position() {
        let m: Manifest = serde_json::from_str(&manifest_json(BUNDLE_FORMAT)).expect("parse");
        assert_eq!(m.views.len(), 2);
        // `default_view` is 1, i.e. the SECOND entry, whose dataset index is 8.
        assert_eq!(m.views[m.default_view].index, 8);
        assert_eq!(m.num_channels, 4);
        assert_eq!(m.background.len(), 4);
        assert_eq!(m.up, [0.0, -1.0, 0.0]);
    }

    #[test]
    fn params_default_every_performance_lever_to_exact() {
        let m: Manifest = serde_json::from_str(&manifest_json(BUNDLE_FORMAT)).expect("parse");
        assert!(
            m.params.is_exact(),
            "a bundle written before the levers existed must load as the exact pipeline"
        );
        assert_eq!(m.params.num_layers, 8);
    }

    #[test]
    fn a_view_becomes_a_camera_including_its_distortion() {
        let m: Manifest = serde_json::from_str(&manifest_json(BUNDLE_FORMAT)).expect("parse");
        let camera = m.views[0].camera();
        assert_eq!(camera.width, 1920);
        assert!((camera.distortion[0] + 0.064).abs() < 1e-6);
        // The second view omits nothing but is a plain pinhole.
        assert_eq!(m.views[1].camera().distortion, [0.0; 8]);
    }

    #[test]
    fn camera_position_inverts_the_pose() {
        let m: Manifest = serde_json::from_str(&manifest_json(BUNDLE_FORMAT)).expect("parse");
        // R = I, t = (1, 2, 3)  =>  centre = -t.
        let p = m.views[1].position();
        assert!((p.x + 1.0).abs() < 1e-6 && (p.y + 2.0).abs() < 1e-6 && (p.z + 3.0).abs() < 1e-6);
    }

    #[test]
    fn an_unknown_format_is_refused() {
        let dir = std::env::temp_dir().join(format!("trips-bundle-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmpdir");
        std::fs::write(dir.join(MANIFEST_NAME), manifest_json("trippy-bundle-999"))
            .expect("write");
        let error = match Bundle::load(&dir) {
            Err(e) => e,
            Ok(_) => panic!("an unknown bundle format must be refused"),
        };
        assert!(error.contains("trippy-bundle-999"), "{error}");
        assert!(error.contains(BUNDLE_FORMAT), "{error}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_out_of_range_default_view_falls_back_to_the_most_central_one() {
        let mut m: Manifest = serde_json::from_str(&manifest_json(BUNDLE_FORMAT)).expect("parse");
        assert_eq!(m.default_view, 1);
        // Both views are pinhole-identity poses at (0,0,0) and (-1,-2,-3);
        // whichever is nearest the box centre, it must be a real position.
        m.default_view = 99;
        let views = m.views.clone();
        let fallback = SceneScale::most_central_view(&views).expect("a position");
        assert!(fallback < views.len());
    }

    #[test]
    fn a_missing_manifest_is_an_error_not_a_panic() {
        let dir = std::env::temp_dir().join("trips-bundle-test-does-not-exist");
        assert!(Bundle::load(&dir).is_err());
    }
}

#[cfg(test)]
mod bounds_tests {
    use super::*;

    fn unit_box() -> Bounds {
        Bounds {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        }
    }

    fn camera_at(z: f32) -> Camera {
        Camera {
            width: 64,
            height: 48,
            fx: 50.0,
            fy: 50.0,
            cx: 32.0,
            cy: 24.0,
            // Identity rotation, camera centre at (0, 0, -z) => t = (0, 0, z).
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            t: [0.0, 0.0, z],
            distortion: [0.0; 8],
        }
    }

    #[test]
    fn the_span_brackets_the_box_in_front_of_the_camera() {
        let (lo, hi) = unit_box().depth_span(&camera_at(5.0), 1e-6);
        // Corners run from z = 4 to z = 6.
        assert!((lo - 4.0).abs() < 1e-5, "{lo}");
        assert!(hi >= 6.0 && hi <= 6.0 * DEPTH_SPAN_SLACK + 1e-3, "{hi}");
        assert!(lo < hi);
    }

    #[test]
    fn a_camera_inside_the_box_clamps_to_the_near_plane() {
        // Camera centre at the origin: half the corners are behind it.
        let (lo, hi) = unit_box().depth_span(&camera_at(0.0), 0.05);
        assert!(lo >= 0.05, "{lo}");
        assert!(hi > lo, "{lo} {hi}");
        assert!(lo.is_finite() && hi.is_finite());
    }

    #[test]
    fn a_degenerate_box_still_yields_an_ordered_span() {
        let flat = Bounds {
            min: [0.0; 3],
            max: [0.0; 3],
        };
        let (lo, hi) = flat.depth_span(&camera_at(3.0), 1e-3);
        assert!(lo > 0.0 && hi > lo, "{lo} {hi}");
        assert!(flat.diameter() > 0.0);
    }

    #[test]
    fn diameter_is_the_diagonal() {
        assert!((unit_box().diameter() - (12.0f32).sqrt()).abs() < 1e-5);
    }

    #[test]
    fn a_box_clamps_contains_and_expands_about_its_centre() {
        let b = Bounds {
            min: [0.0, -2.0, 10.0],
            max: [4.0, 2.0, 14.0],
        };
        assert!((b.centre() - glam::Vec3::new(2.0, 0.0, 12.0)).length() < 1e-6);
        assert!(b.contains(glam::Vec3::new(2.0, 0.0, 12.0)));
        assert!(!b.contains(glam::Vec3::new(2.0, 0.0, 20.0)));
        let clamped = b.clamp_point(glam::Vec3::new(-5.0, 7.0, 12.5));
        assert!((clamped - glam::Vec3::new(0.0, 2.0, 12.5)).length() < 1e-6);
        // Expanding keeps the centre and triples the half-extents.
        let wide = b.expanded(3.0);
        assert!((wide.centre() - b.centre()).length() < 1e-5);
        assert!((wide.min[0] + 4.0).abs() < 1e-5 && (wide.max[0] - 8.0).abs() < 1e-5);
        assert!(wide.contains(glam::Vec3::new(2.0, 0.0, 17.0)));
        assert!(!wide.contains(glam::Vec3::new(2.0, 0.0, 19.0)));
    }
}

#[cfg(test)]
mod scene_scale_tests {
    use super::*;

    /// `count` views spaced `step` apart along +X, each looking down +Z.
    fn line(count: usize, step: f32) -> Vec<BundleView> {
        (0..count)
            .map(|i| BundleView {
                index: i,
                name: String::new(),
                width: 64,
                height: 48,
                fx: 50.0,
                fy: 50.0,
                cx: 32.0,
                cy: 24.0,
                r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                // R = I, so the centre is -t.
                t: [-(i as f32) * step, 0.0, 0.0],
                ..zeroed()
            })
            .collect()
    }

    /// The distortion field, spelled once.
    fn zeroed() -> BundleView {
        BundleView {
            index: 0,
            name: String::new(),
            width: 1,
            height: 1,
            fx: 1.0,
            fy: 1.0,
            cx: 0.0,
            cy: 0.0,
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            t: [0.0; 3],
            distortion: [0.0; 8],
        }
    }

    #[test]
    fn the_camera_box_and_spacing_come_from_the_views() {
        let views = line(5, 2.0);
        let scene = SceneScale::from_views(&views);
        assert!((scene.bounds.min[0] - 0.0).abs() < 1e-6);
        assert!((scene.bounds.max[0] - 8.0).abs() < 1e-6);
        assert!((scene.median_spacing - 2.0).abs() < 1e-6);
        // BASE_SPEED_FRACTION x the median spacing per second.
        assert!((scene.base_speed() - BASE_SPEED_FRACTION * 2.0).abs() < 1e-6);
    }

    #[test]
    fn one_outlier_jump_does_not_move_the_median() {
        // Four views 1 unit apart, then one 1000 units away: a mean would call
        // this scene 250x bigger than it is, which is the bug this guards.
        let mut views = line(5, 1.0);
        views[4].t[0] = -1000.0;
        let scene = SceneScale::from_views(&views);
        assert!((scene.median_spacing - 1.0).abs() < 1e-6, "{}", scene.median_spacing);
        assert!((scene.base_speed() - BASE_SPEED_FRACTION).abs() < 1e-6);
    }

    #[test]
    fn expanding_a_flat_box_grows_its_flat_axis_too() {
        // The horse capture's shape: wide, deep, and one unit tall.
        let flat = Bounds {
            min: [-5.8, -0.5, -5.2],
            max: [5.8, 0.5, 5.2],
        };
        let wide = flat.expanded(3.0);
        // The wide axes are simply tripled...
        assert!((wide.max[0] - 3.0 * 5.8).abs() < 1e-3, "{}", wide.max[0]);
        // ... and the flat one is grown to the floor, not to 1.5 units.
        assert!(wide.max[1] > 5.0, "flat axis only grew to {}", wide.max[1]);
        assert!(wide.contains(glam::Vec3::new(0.0, 5.0, 0.0)));
        assert!(!wide.contains(glam::Vec3::new(0.0, 50.0, 0.0)));
        // A cube is unaffected by the floor: expansion is exactly a scaling.
        let cube = Bounds {
            min: [-1.0; 3],
            max: [1.0; 3],
        };
        assert!((cube.expanded(3.0).max[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn a_single_view_still_yields_a_usable_speed() {
        let scene = SceneScale::from_views(&line(1, 1.0));
        assert_eq!(scene.median_spacing, 0.0);
        assert!(scene.base_speed() > 0.0 && scene.base_speed().is_finite());
        // The zero-size camera box must not become a zero divisor: the HUD
        // divides the speed by this, and so does the orbit clamp.
        assert!(scene.diameter() > TINY && scene.diameter().is_finite());
        assert!((scene.base_speed() / scene.diameter()).is_finite());
    }

    #[test]
    fn views_all_in_one_place_fall_back_to_the_box() {
        let views = line(4, 0.0);
        let scene = SceneScale::from_views(&views);
        assert_eq!(scene.median_spacing, 0.0);
        assert!(scene.base_speed() > 0.0 && scene.base_speed().is_finite());
    }

    #[test]
    fn the_most_central_view_is_the_middle_of_the_line() {
        let views = line(5, 2.0);
        assert_eq!(SceneScale::most_central_view(&views), Some(2));
        assert_eq!(SceneScale::most_central_view(&[]), None);
    }
}
