//! Loader for the synthetic parity fixtures.
//!
//! Module: `brush_pyramid::fixture`
//! Purpose: read one `tests/fixtures/synthetic/raster_fixture_*/` directory —
//!     the point set, camera, render parameters and the per-layer images the
//!     Python reference produced for them — so the CPU and GPU parity tests
//!     share exactly one definition of the on-disk contract.
//! Invariants:
//!     - The directory layout is written by `tools/dump_raster_fixture.py` and
//!       nothing else. Changing one side without the other must fail loudly,
//!       so every field is required and every shape is checked.
//!     - `points.npz` is stored and `expected.npz` deflated on purpose; both
//!       branches of [`crate::npz`] are therefore exercised by loading a
//!       fixture at all.
//!     - Fixtures are SYNTHETIC ONLY (`AGENTS.md` section 6).
//! Units: as [`crate::scene`] and [`crate::output`].
//! Related docs: `tools/dump_raster_fixture.py`'s module docstring.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::npz::read_npz;
use crate::output::{LayerImage, PyramidImages};
use crate::params::{Mode, PixelCenter, PyramidHalving, PyramidParams};
use crate::scene::{Camera, PointSet};

/// The `params.json` document.
#[derive(Debug, Clone, Deserialize)]
struct FixtureParamsJson {
    mode: Mode,
    pixel_center: PixelCenter,
    pyramid_halving: PyramidHalving,
    num_layers: usize,
    num_channels: usize,
    max_frags: u32,
    t_cutoff: f32,
    alpha_min: f32,
    znear: f32,
    background: Vec<f32>,
}

/// The `meta.json` document.
#[derive(Debug, Clone, Deserialize)]
struct FixtureMetaJson {
    num_fragments: u32,
    fragments_per_layer: Vec<u32>,
    layer_shapes: Vec<[usize; 2]>,
}

/// A loaded fixture: inputs, render options, and the Python reference output.
#[derive(Debug, Clone)]
pub struct Fixture {
    /// Directory this was loaded from, for error messages.
    pub path: PathBuf,
    /// The synthetic point set.
    pub points: PointSet,
    /// The camera it was rendered from.
    pub camera: Camera,
    /// Render options, already mapped onto [`PyramidParams`].
    pub params: PyramidParams,
    /// The `C` background values.
    pub background: Vec<f32>,
    /// What `trippy.raster.render_pyramid` produced for these inputs on CPU.
    pub expected: PyramidImages,
}

/// Directory holding every fixture, relative to the repository root.
pub const FIXTURE_ROOT: &str = "tests/fixtures/synthetic";

/// Repository root, derived from this crate's manifest directory.
///
/// `CARGO_MANIFEST_DIR` is `<repo>/rust/crates/brush-pyramid`, so the root is
/// three levels up. Used by the tests and by the example's default paths.
#[must_use]
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("crate is nested at <repo>/rust/crates/brush-pyramid")
        .to_path_buf()
}

/// Every fixture directory, sorted by name.
///
/// # Errors
/// Returns `Err` if [`FIXTURE_ROOT`] cannot be listed.
pub fn fixture_dirs() -> Result<Vec<PathBuf>, String> {
    let root = repo_root().join(FIXTURE_ROOT);
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .map_err(|e| format!("{}: {e}", root.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("raster_fixture_"))
        })
        .collect();
    dirs.sort();
    Ok(dirs)
}

impl Fixture {
    /// Load the fixture in `dir`.
    ///
    /// # Errors
    /// Returns `Err` if any file is missing or malformed, or if the expected
    /// images do not match the layer shapes recorded in `meta.json`.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let points = PointSet::from_npz(&dir.join("points.npz"))?;
        let camera = Camera::from_json(&dir.join("camera.json"))?;

        let params_json: FixtureParamsJson = read_json(&dir.join("params.json"))?;
        let meta: FixtureMetaJson = read_json(&dir.join("meta.json"))?;

        if params_json.num_channels != points.num_channels {
            return Err(format!(
                "{}: params.json says C = {}, points.npz has {}",
                dir.display(),
                params_json.num_channels,
                points.num_channels
            ));
        }
        if params_json.background.len() != points.num_channels {
            return Err(format!(
                "{}: background has {} values, expected {}",
                dir.display(),
                params_json.background.len(),
                points.num_channels
            ));
        }

        let params = PyramidParams {
            mode: params_json.mode,
            num_layers: params_json.num_layers,
            pixel_center: params_json.pixel_center,
            halving: params_json.pyramid_halving,
            max_frags: params_json.max_frags,
            t_cutoff: params_json.t_cutoff,
            alpha_min: params_json.alpha_min,
            znear: params_json.znear,
            // Fixtures pin the *exact* pipeline; the v0.4.0 performance levers
            // are viewer settings and never come from a fixture.
            ..PyramidParams::default()
        };

        let arrays = read_npz(&dir.join("expected.npz"))?;
        let channels = points.num_channels;
        let mut layers = Vec::with_capacity(params.num_layers);
        for layer in 0..params.num_layers {
            let [h_l, w_l] = *meta
                .layer_shapes
                .get(layer)
                .ok_or_else(|| format!("{}: meta.json has no shape for layer {layer}", dir.display()))?;
            let get = |key: String| {
                arrays
                    .get(&key)
                    .ok_or_else(|| format!("{}: expected.npz has no {key}", dir.display()))
            };
            let feature_arr = get(format!("layer_{layer}"))?;
            feature_arr.expect_shape("layer", &[channels, h_l, w_l])?;
            let t_final_arr = get(format!("t_final_{layer}"))?;
            t_final_arr.expect_shape("t_final", &[h_l, w_l])?;
            let n_used_arr = get(format!("n_used_{layer}"))?;
            n_used_arr.expect_shape("n_used", &[h_l, w_l])?;

            layers.push(LayerImage {
                height: h_l,
                width: w_l,
                channels,
                feature: feature_arr.to_f32()?,
                t_final: t_final_arr.to_f32()?,
                n_used: n_used_arr.to_i32()?.into_iter().map(|v| v as u32).collect(),
            });
        }

        Ok(Self {
            path: dir.to_path_buf(),
            points,
            camera,
            params,
            background: params_json.background,
            expected: PyramidImages {
                layers,
                num_fragments: meta.num_fragments,
                fragments_per_layer: meta.fragments_per_layer,
            },
        })
    }

    /// The fixture's directory name, e.g. `raster_fixture_trips_half`.
    #[must_use]
    pub fn name(&self) -> String {
        self.path
            .file_name()
            .map_or_else(|| self.path.display().to_string(), |n| n.to_string_lossy().into_owned())
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}
