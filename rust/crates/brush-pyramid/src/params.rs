//! Render parameters: the Rust mirror of `trippy.constants`' `RASTER_*` block.
//!
//! Module: `brush_pyramid::params`
//! Purpose: one place holding every knob the pyramid rasteriser takes, with
//!     the same names, defaults and units as the Python side, so a parity
//!     failure is never "the two sides were configured differently".
//! Invariants:
//!     - Every default here must equal the correspondingly named constant in
//!       `trippy/constants.py`. `tests/parity_cpu.rs` reads each fixture's
//!       `params.json` (written by `tools/dump_raster_fixture.py` straight
//!       from those constants) and asserts the match, so drift is caught.
//!     - `Mode`, `PixelCenter` and `PyramidHalving` serialise to exactly the
//!       lowercase strings `trippy.constants.RASTER_MODES` /
//!       `RASTER_PIXEL_CENTERS` / `RASTER_PYRAMID_HALVINGS` use.
//! Units: `t_cutoff` and `alpha_min` are dimensionless; `znear` is in world
//!     units; `max_frags` is a count.
//! Related docs: `docs/GEOMETRY.md` ("Pyramid level selection", "Pixel-centre
//!     convention", "Image pyramid"); `docs/TRIPS_REFERENCE.md` sections 3,
//!     3a, 3b.

use serde::{Deserialize, Serialize};

/// Number of pyramid layers (`trippy.constants.RASTER_NUM_LAYERS`).
pub const DEFAULT_NUM_LAYERS: usize = 5;
/// Near-plane cull, world units (`RASTER_ZNEAR`).
pub const DEFAULT_ZNEAR: f32 = 1e-3;
/// Emission-time alpha floor (`RASTER_ALPHA_MIN`).
pub const DEFAULT_ALPHA_MIN: f32 = 1e-5;
/// Transmittance below which compositing stops (`RASTER_T_CUTOFF`, TRIPS's
/// `ALPHA_DEST_CUTOFF`).
pub const DEFAULT_T_CUTOFF: f32 = 1e-3;
/// Fragments composited per layer-pixel (`RASTER_MAX_FRAGS`, TRIPS: 16).
pub const DEFAULT_MAX_FRAGS: u32 = 16;
/// Floor on the sub-pixel blend factor (`RASTER_SMALL_POINT_CUTOFF`, TRIPS
/// `PointBlending.h:106`).
pub const SMALL_POINT_CUTOFF: f32 = 0.25;
/// Slack, in coarsest-layer pixels, added to the visibility cull box
/// (`RASTER_CULL_MARGIN_COARSE_PX`). Must stay >= 1.5 — see [`crate::cpu`].
pub const CULL_MARGIN_COARSE_PX: f32 = 2.0;
/// serde default for [`PyramidParams::frustum_cull`].
const fn yes() -> bool {
    true
}

/// Feature widths the GPU blend kernel is specialised for
/// (`RASTER_SUPPORTED_CHANNELS`).
pub const SUPPORTED_CHANNELS: [usize; 3] = [3, 4, 8];

/// Which pyramid layers a point is written into, and with what weight.
///
/// All three are ports of real TRIPS code paths; see `docs/GEOMETRY.md`
/// "Pyramid level selection and the layer factor" for the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Layers `[lower, upper]` only, weighted by [`crate::factor::layer_factor`].
    /// TRIPS's `CollectTiled2Pointsize` (`RenderForward.cu:2296-2360`).
    Trilinear,
    /// Every layer, weight 1. TRIPS's `use_layer_point_size = false`.
    Broadcast,
    /// Layers `0 ..= layer_higher`, weighted by the layer factor, with TRIPS's
    /// all-four-corners footprint gate and its `break`. What every published
    /// checkpoint renders with (`Settings.cpp:39`).
    Trips,
}

/// Where the centre of pixel `i` sits in continuous layer coordinates.
///
/// The two conventions cannot be reconciled by shifting `cx`/`cy`: the offset
/// is applied *after* the per-layer halving, so it is worth `2^(l-1)` layer-0
/// pixels at layer `l` (`docs/GEOMETRY.md`, "Pixel-centre convention").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PixelCenter {
    /// Centre of pixel `i` at `i + 0.5`. trippy's own convention (COLMAP,
    /// `grid_sample(align_corners=False)`), and the default.
    Half,
    /// Centre of pixel `i` at `i`. TRIPS's `ip`
    /// (`PointBlending.h:216-240`); needed only for checkpoint parity.
    Integer,
}

impl PixelCenter {
    /// The constant subtracted from a layer coordinate before `floor` to find
    /// the 2x2 footprint's base pixel (`trippy.raster.emit._CENTRE_SHIFT`).
    #[must_use]
    pub const fn shift(self) -> f32 {
        match self {
            Self::Half => 0.5,
            Self::Integer => 0.0,
        }
    }
}

/// How layer `l`'s resolution is derived from layer 0's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PyramidHalving {
    /// `h_l = ceil(H / 2^l)`. What TRIPS does for every published checkpoint
    /// (`PointRenderer.cu:385-391`), and the default.
    Ceil,
    /// `h_l = H / 2^l` (integer division). TRIPS's `MultiScaleUnet2d` branch,
    /// which silently drops the last row/column of an odd-sized level.
    Floor,
}

/// The finest layer mode [`Mode::Trips`] emits into — the v0.4.0 "fragment
/// cap" performance lever.
///
/// TRIPS's `compute_point_size_fac` returns **1.0 for every layer below
/// `layer_lower`** (`PointBlending.h:92-96`), so a point large enough to sit
/// at layer 4 also paints layers 0..3 at full alpha. That is exact, and it is
/// also where most of the fragments come from on a scene with big points.
///
/// Related docs: `docs/LIMITATIONS.md` ("viewer performance levers").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerFloor {
    /// Layer 0. **Exact**, and the default.
    #[default]
    Zero,
    /// `layer_lower.saturating_sub(1)`: emit only the two layers that carry a
    /// non-trivial factor plus one finer for safety. An approximation —
    /// measure PSNR against [`Self::Zero`] before shipping it.
    NearLower,
}

/// How the fragment list is ordered before compositing.
///
/// Both produce `(layer, pixel, depth)` order; they differ in how much sorting
/// work that costs and in how ties are broken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortMode {
    /// Two stable LSB radix passes — 32 bits of float depth, then
    /// `ceil(log2 P)` bits of layer-pixel key. **Exact**, and the default.
    /// Costs `8 + ceil(bits/4)` four-bit passes (14 at 1080p, L = 8).
    #[default]
    DepthThenKey,
    /// One radix pass over a single packed `u32`:
    /// `layer_pixel << depth_bits | quantise(depth)`, with
    /// `depth_bits = 32 - ceil(log2 P)` (10 at 1080p, L = 8). Costs 8 passes
    /// instead of 14 and skips a gather, at the price of quantised depth
    /// ordering — fragments landing in one depth bucket keep emission order
    /// (ascending point id) instead of true depth order.
    PackedKey,
}

/// Element type the per-point feature buffer is uploaded and gathered as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FeatureStore {
    /// float32, exactly as the point set is loaded. The default.
    #[default]
    F32,
    /// float16, halving the bandwidth of the blend kernel's random gathers.
    /// Rounds every feature to ~3 decimal digits before compositing.
    F16,
}

/// Depth quantisation range for [`SortMode::PackedKey`], world units.
///
/// The mapping is linear in `log2(depth)` between these bounds, which spends
/// the available buckets evenly in *relative* depth — the right choice when
/// what matters is resolving two surfaces that are close together compared to
/// their distance from the camera. Depths outside the range clamp to the end
/// buckets, which is safe (they simply tie with their neighbours) but wasteful,
/// so a viewer should keep them tight around the visible scene.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DepthRange {
    /// Nearest depth resolved, world units. Must be > 0.
    pub lo: f32,
    /// Farthest depth resolved, world units. Must be > `lo`.
    pub hi: f32,
}

impl Default for DepthRange {
    /// A deliberately wide default (1 cm to 1 km) so a caller that forgets to
    /// set it gets a coarse-but-correct ordering rather than a broken one.
    fn default() -> Self {
        Self { lo: 1e-2, hi: 1e3 }
    }
}

/// Everything the forward pass needs besides the points and the camera.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PyramidParams {
    /// Layer-selection rule (see [`Mode`]).
    pub mode: Mode,
    /// `L`, the number of pyramid layers (>= 1).
    pub num_layers: usize,
    /// Pixel-centre convention (see [`PixelCenter`]).
    pub pixel_center: PixelCenter,
    /// Per-layer resolution rule (see [`PyramidHalving`]).
    pub halving: PyramidHalving,
    /// Fragments composited per layer-pixel before the loop stops.
    pub max_frags: u32,
    /// Transmittance at which compositing stops for a pixel.
    pub t_cutoff: f32,
    /// Fragments with `alpha` below this are never emitted. TRIPS has no such
    /// floor; pass `0.0` for a bit-exact TRIPS port.
    pub alpha_min: f32,
    /// Camera-space depths at or below this are behind the near plane.
    pub znear: f32,
    /// Frustum-cull lever: when false, only the near plane culls and every
    /// point is offered to the layer-selection loop. **Exists to measure what
    /// the cull is worth**; there is no reason to turn it off in a viewer.
    #[serde(default = "yes")]
    pub frustum_cull: bool,
    /// Fragment-cap lever: the finest layer `Trips` emits into.
    #[serde(default)]
    pub layer_floor: LayerFloor,
    /// Sort strategy lever (GPU path only).
    #[serde(default)]
    pub sort: SortMode,
    /// Feature-storage lever (GPU path only).
    #[serde(default)]
    pub feature_store: FeatureStore,
    /// Depth range used by [`SortMode::PackedKey`]; ignored otherwise.
    #[serde(default)]
    pub depth_range: DepthRange,
}

impl Default for PyramidParams {
    /// trippy's training defaults: mode `trips`, 5 layers, `half` pixel
    /// centres, `ceil` halving (all from `trippy.constants`).
    fn default() -> Self {
        Self {
            mode: Mode::Trips,
            num_layers: DEFAULT_NUM_LAYERS,
            pixel_center: PixelCenter::Half,
            halving: PyramidHalving::Ceil,
            max_frags: DEFAULT_MAX_FRAGS,
            t_cutoff: DEFAULT_T_CUTOFF,
            alpha_min: DEFAULT_ALPHA_MIN,
            znear: DEFAULT_ZNEAR,
            frustum_cull: true,
            layer_floor: LayerFloor::Zero,
            sort: SortMode::DepthThenKey,
            feature_store: FeatureStore::F32,
            depth_range: DepthRange::default(),
        }
    }
}

impl PyramidParams {
    /// True when every performance lever is at its exact setting, i.e. the
    /// render is bit-comparable with the CPU reference and the Python engine.
    #[must_use]
    pub fn is_exact(&self) -> bool {
        self.frustum_cull
            && self.layer_floor == LayerFloor::Zero
            && self.sort == SortMode::DepthThenKey
            && self.feature_store == FeatureStore::F32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_exact_pipeline() {
        assert!(PyramidParams::default().is_exact());
    }

    #[test]
    fn a_params_json_without_the_levers_still_loads_exact() {
        let json = r#"{"mode":"trips","num_layers":8,"pixel_center":"integer",
                       "halving":"ceil","max_frags":16,"t_cutoff":0.001,
                       "alpha_min":0.0,"znear":1e-6}"#;
        let params: PyramidParams = serde_json::from_str(json).expect("deserialise");
        assert!(params.is_exact());
        assert_eq!(params.depth_range, DepthRange::default());
    }

    #[test]
    fn lever_names_round_trip_through_their_json_spellings() {
        let params = PyramidParams {
            layer_floor: LayerFloor::NearLower,
            sort: SortMode::PackedKey,
            feature_store: FeatureStore::F16,
            ..PyramidParams::default()
        };
        let json = serde_json::to_string(&params).expect("serialise");
        assert!(json.contains("\"near_lower\""), "{json}");
        assert!(json.contains("\"packed_key\""), "{json}");
        assert!(json.contains("\"f16\""), "{json}");
        assert!(!params.is_exact());
        let back: PyramidParams = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(params, back);
    }
}
