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
        Self::from_parts(dir.to_path_buf(), manifest, &points_bytes, &weight_bytes, &origin)
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

        Ok(Self {
            dir,
            manifest,
            points,
            weights,
        })
    }

    /// The view the viewer opens at, clamped into range.
    #[must_use]
    pub fn default_view(&self) -> &BundleView {
        let index = self
            .manifest
            .default_view
            .min(self.manifest.views.len() - 1);
        &self.manifest.views[index]
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
}
