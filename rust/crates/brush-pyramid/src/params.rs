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
        }
    }
}
