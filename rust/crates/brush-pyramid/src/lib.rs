//! brush-pyramid: TRIPS image-pyramid rasteriser (skeleton).
//!
//! Module: brush_pyramid
//! Purpose: v0.4.0 placeholder for the Rust/CubeCL port of trippy's pyramid
//!     rasteriser (`trippy/raster/emit.py`). Only `layer_bounds`/`layer_factor`
//!     are implemented so far, as a faithful scalar port validated against the
//!     same hand-computed values as `tests/test_raster_layer_factor.py`. The
//!     real workload -- `emit_fragments`, two radix argsorts, `prefix_sum`, and
//!     the `blend_fwd`/`blend_bwd` CubeCL kernels -- is future work tracked in
//!     docs/SPEC.md's v0.4.0 row.
//! Invariants:
//!     - `layer_factor`/`layer_bounds` must stay numerically identical to the
//!       Python reference (`trippy.raster.emit.layer_bounds`/`layer_factor`,
//!       an exact port of TRIPS's `compute_point_size_fac`,
//!       third_party/TRIPS/src/lib/rendering/PointBlending.h:81-149). Do not
//!       "clean up" the quirks documented below; they match the C++ source.
//!     - `NUM_LAYERS` is a fixed constant here (unlike the Python version,
//!       which takes it as a parameter) because this is a placeholder; the
//!       real port should take it as a parameter once wired into brush-app.
//! Related docs: docs/GEOMETRY.md "Pyramid level selection"; docs/SPEC.md
//!     v0.4.0 row; docs/decisions/ADR-0005-brush-fork-layout.md.

/// Number of pyramid layers. Matches `tests/test_raster_layer_factor.py`'s
/// `NUM_LAYERS = 5` so the hand-computed values below carry over directly.
/// A future, fully-wired version of this crate should take this as a
/// parameter (as the Python `layer_bounds`/`layer_factor` do) rather than a
/// constant.
pub const NUM_LAYERS: u32 = 5;

/// Floor on the sub-pixel blend factor (TRIPS `PointBlending.h:106`). A point
/// smaller than one pixel still splats, with the factor never dropping below
/// this cutoff.
const SMALL_POINT_CUTOFF: f32 = 0.25;

/// Lower/upper pyramid layer a point of pixel size `size_px` writes into.
///
/// Direct port of `compute_point_size_fac`'s first block
/// (`PointBlending.h:86-93`): both layers are 0 when `size_px <= 1`,
/// otherwise floor/ceil of `log2(size_px)` clamped to `[0, NUM_LAYERS - 1]`.
fn layer_bounds(size_px: f32) -> (u32, u32) {
    if size_px <= 1.0 {
        return (0, 0);
    }
    let max_layer = (NUM_LAYERS - 1) as f32;
    let log_ps = size_px.log2();
    let lower = log_ps.floor().clamp(0.0, max_layer) as u32;
    let upper = log_ps.ceil().clamp(0.0, max_layer) as u32;
    (lower, upper)
}

/// TRIPS's per-layer blend factor for a point of pixel size `size_px`.
///
/// Exact port of `compute_point_size_fac`
/// (`third_party/TRIPS/src/lib/rendering/PointBlending.h:81-149`), including
/// its quirks (see `tests/test_raster_layer_factor.py` for the Python-side
/// documentation of each one):
///
/// - `layer < lower` returns **1.0** in the C++, not 0.0. The branch is
///   unreachable from TRIPS's own point-size emission kernel, so the value
///   is inert either way; this matches the source, not a "cleaner" reading.
/// - `upper == 0` (a sub-pixel point, `size_px <= 1`) gives the exponential
///   floor `(1 - c) * exp(size_px - 1) + c` with `c = SMALL_POINT_CUTOFF`.
/// - `lower == upper` (an exact power of two, or clamped to the top layer)
///   gives 1.0.
/// - Otherwise the interpolation is linear **in point-size units** between
///   `2**lower` and `2**upper`, not linear in the log2 fraction.
///
/// # Arguments
/// - `s`: projected point size in layer-0 pixels (world-unit radius already
///   converted via `fx * size_world / z`; see docs/GEOMETRY.md).
/// - `l`: which pyramid layer the factor is wanted for, in `[0, NUM_LAYERS)`.
pub fn layer_factor(s: f32, l: u32) -> f32 {
    let (lower, upper) = layer_bounds(s);

    if l < lower {
        return 1.0;
    }
    if upper == 0 {
        return (1.0 - SMALL_POINT_CUTOFF) * (s.min(1.0) - 1.0).exp() + SMALL_POINT_CUTOFF;
    }
    if lower == upper {
        return 1.0;
    }

    let lo_pow = 2f32.powi(lower as i32);
    let hi_pow = 2f32.powi(upper as i32);
    let f = (s - lo_pow) / (hi_pow - lo_pow);
    if l == lower { 1.0 - f } else { f }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same hand-computed values as `tests/test_raster_layer_factor.py`
    /// (which cites `PointBlending.h:81-149` line numbers directly), so the
    /// two ports can be eyeballed against each other.
    const TOL: f32 = 1e-6;

    #[test]
    fn layer_bounds_hand_values() {
        assert_eq!(layer_bounds(0.25), (0, 0));
        assert_eq!(layer_bounds(0.5), (0, 0));
        assert_eq!(layer_bounds(1.0), (0, 0));
        assert_eq!(layer_bounds(1.5), (0, 1));
        assert_eq!(layer_bounds(2.0), (1, 1));
        assert_eq!(layer_bounds(2.0f32.powf(1.3)), (1, 2));
        assert_eq!(layer_bounds(4.0), (2, 2));
        assert_eq!(layer_bounds(12.0), (3, 4));
        assert_eq!(layer_bounds(16.0), (4, 4));
        assert_eq!(layer_bounds(100.0), (4, 4));
    }

    #[test]
    fn sub_pixel_points_get_the_exponential_floor() {
        let expected_half = (1.0 - SMALL_POINT_CUTOFF) * (0.5f32 - 1.0).exp() + SMALL_POINT_CUTOFF;
        assert!((expected_half - 0.704_897_99).abs() < TOL);
        assert!((layer_factor(0.5, 0) - expected_half).abs() < TOL);

        let expected_tiny = 0.75 * (-0.99f32).exp() + 0.25;
        assert!((layer_factor(0.01, 0) - expected_tiny).abs() < TOL);
        assert!(layer_factor(0.01, 0) > SMALL_POINT_CUTOFF);

        assert!((layer_factor(1.0, 0) - 1.0).abs() < TOL);
    }

    #[test]
    fn exact_power_of_two_gives_factor_one() {
        assert!((layer_factor(2.0, 1) - 1.0).abs() < TOL);
        assert!((layer_factor(4.0, 2) - 1.0).abs() < TOL);
        assert!((layer_factor(8.0, 3) - 1.0).abs() < TOL);
    }

    #[test]
    fn interpolation_is_linear_in_point_size_not_in_log2() {
        // s = 2**1.3 = 2.46 -> 0.23 toward the upper layer, not 0.3.
        let size_px = 2.0f32.powf(1.3);
        let upper = layer_factor(size_px, 2);
        let lower = layer_factor(size_px, 1);
        let expected_upper = (size_px - 2.0) / (4.0 - 2.0);
        assert!((upper - expected_upper).abs() < TOL);
        assert!((upper - 0.23).abs() < 2e-3);
        assert!((upper - 0.3).abs() > 1e-2);
        assert!((lower - (1.0 - upper)).abs() < TOL);
    }

    #[test]
    fn interpolation_weights_sum_to_one() {
        for (size_px, lower_layer) in [(1.5, 0), (3.0, 1), (6.0, 2), (12.0, 3)] {
            let total = layer_factor(size_px, lower_layer) + layer_factor(size_px, lower_layer + 1);
            assert!((total - 1.0).abs() < TOL);
        }
    }

    #[test]
    fn sizes_beyond_the_top_layer_clamp_to_one() {
        for size_px in [16.0, 20.0, 100.0, 1e6] {
            assert!((layer_factor(size_px, NUM_LAYERS - 1) - 1.0).abs() < TOL);
        }
    }

    #[test]
    fn layer_below_lower_returns_one_matching_the_cpp_source() {
        assert!((layer_factor(4.0, 0) - 1.0).abs() < TOL);
        assert!((layer_factor(100.0, 3) - 1.0).abs() < TOL);
    }
}
