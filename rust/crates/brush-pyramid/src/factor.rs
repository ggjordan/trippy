//! TRIPS's per-layer point-size blend factor.
//!
//! Module: `brush_pyramid::factor`
//! Purpose: scalar port of `compute_point_size_fac`
//!     (`third_party/TRIPS/src/lib/rendering/PointBlending.h:81-149`) and of
//!     its first block, `layer_bounds`. Mirrors
//!     `trippy.raster.emit.layer_bounds` / `layer_factor` and
//!     `trippy.raster.ref_numpy.layer_bounds_scalar` /
//!     `layer_factor_scalar`.
//! Invariants:
//!     - Numerically identical to the Python reference. **Do not "clean up"
//!       the quirks below** — each one matches the C++ source and is pinned by
//!       `tests/test_raster_layer_factor.py` on the Python side and by the
//!       unit tests at the bottom of this file on ours.
//!     - Pure `f32` scalar arithmetic and no allocation, so the same
//!       expressions can be transcribed into the `#[cube]` kernels
//!       (`crate::gpu::kernels`) and give bit-comparable results.
//! Units: `size_px` is the projected point size in **layer-0 pixels**,
//!     `fx * size_world / z` (TRIPS uses `fx` only, `RenderForward.cu:1489`).
//! Related docs: `docs/GEOMETRY.md` "Pyramid level selection and the layer
//!     factor"; `docs/TRIPS_REFERENCE.md` sections 3 and 10.2.

use crate::params::SMALL_POINT_CUTOFF;

/// Lower/upper pyramid layer a point of size `size_px` straddles.
///
/// Port of `PointBlending.h:86-93`: both layers are 0 when `size_px <= 1`
/// (the `point_size_opt > 1` test is strict), otherwise `floor`/`ceil` of
/// `log2(size_px)` clamped to `[0, num_layers - 1]`.
///
/// **`log2` is computed from the IEEE-754 exponent, not by calling
/// `f32::log2`.** CubeCL exposes `ln` but not `log2`, so the kernel has to
/// read the exponent field; this function does the same thing so the CPU twin
/// and the kernel agree *by construction* rather than by luck. The exponent
/// of a positive normal float is exactly `floor(log2 x)`, and `ceil` differs
/// from it exactly when the mantissa is non-zero — both are exact integers,
/// with no rounding step to disagree about.
///
/// The cost is a known, bounded divergence from the Python reference, which
/// does call `torch.log2`: see [`layer_bounds_via_log2`] and
/// `docs/LIMITATIONS.md`.
///
/// # Arguments
/// - `size_px`: projected point size in layer-0 pixels.
/// - `num_layers`: `L`, >= 1.
///
/// # Returns
/// `(lower, upper)` with `0 <= lower <= upper <= L - 1`.
#[must_use]
pub fn layer_bounds(size_px: f32, num_layers: usize) -> (u32, u32) {
    // Also catches NaN, which must never reach the exponent arithmetic.
    if !(size_px > 1.0) {
        return (0, 0);
    }
    let bits = size_px.to_bits();
    let floor_log = ((bits >> 23) & 0xFF) - 127;
    let ceil_log = floor_log + u32::from(bits & 0x007F_FFFF != 0);
    let max_layer = (num_layers - 1) as u32;
    (floor_log.min(max_layer), ceil_log.min(max_layer))
}

/// [`layer_bounds`] as the Python reference computes it: via `f32::log2`.
///
/// Kept, and tested against [`layer_bounds`], to document exactly where the
/// two differ. `f32::log2` is correctly rounded, and *that rounding* is the
/// problem: for a size one ulp below a power of two, say `7.9999995`, the
/// true `log2` is `2.99999991...` but the nearest `f32` to it is exactly
/// `3.0`, so `floor` returns 3 where the exact answer is 2. The exponent-based
/// [`layer_bounds`] has no such step.
///
/// The divergence band is one or two float values per octave — about 1 in
/// 6 million sizes — and is described in `docs/LIMITATIONS.md`.
///
/// # Arguments
/// - `size_px`: projected point size in layer-0 pixels.
/// - `num_layers`: `L`, >= 1.
#[must_use]
pub fn layer_bounds_via_log2(size_px: f32, num_layers: usize) -> (u32, u32) {
    if !(size_px > 1.0) {
        return (0, 0);
    }
    let max_layer = (num_layers - 1) as f32;
    let log_ps = size_px.log2();
    let lower = log_ps.floor().clamp(0.0, max_layer) as u32;
    let upper = log_ps.ceil().clamp(0.0, max_layer) as u32;
    (lower, upper)
}

/// TRIPS's `layer_higher`: the coarsest layer [`crate::params::Mode::Trips`]
/// writes into.
///
/// `RenderForward.cu:334-338`. Identical to [`layer_bounds`]'s upper bound;
/// kept as its own name because that is what the TRIPS source calls it and
/// what the emission loop bounds itself by.
#[must_use]
pub fn layer_higher(size_px: f32, num_layers: usize) -> u32 {
    layer_bounds(size_px, num_layers).1
}

/// The per-layer blend factor for a point of size `size_px`.
///
/// Exact port of `compute_point_size_fac`, quirks included:
///
/// - `layer < lower` returns **1.0** in the C++ (`PointBlending.h:92-96`),
///   not 0.0 as `docs/TRIPS_REFERENCE.md` section 3 claims. Inert for
///   `Trilinear`/`Broadcast`, which never ask below `lower`, but in `Trips`
///   it is the whole reason a big point paints every finer layer at full
///   alpha — worth 0.8 dB in `experiments/EXP-0002-horse-parity`.
/// - `upper == 0` (a sub-pixel point) gives the exponential *floor*
///   `(1 - c) * exp(size_px - 1) + c` with `c = SMALL_POINT_CUTOFF`
///   (`PointBlending.h:106`). There is no sub-pixel size clamp; a 0.01 px
///   point still splats, at factor ~0.529.
/// - `lower == upper` (an exact power of two, or clamped to the top layer)
///   gives 1.0.
/// - Otherwise the interpolation is linear **in point-size units** between
///   `2^lower` and `2^upper` (`PointBlending.h:126-130`), *not* linear in the
///   log2 fraction: `size = 2^1.3 = 2.46` gives `(2.46 - 2) / (4 - 2) = 0.23`,
///   not 0.3.
///
/// # Arguments
/// - `size_px`: projected point size in layer-0 pixels.
/// - `layer`: which pyramid layer the factor is wanted for, in `[0, L)`.
/// - `num_layers`: `L`.
///
/// # Returns
/// The blend factor, in `[0, 1]`.
#[must_use]
pub fn layer_factor(size_px: f32, layer: u32, num_layers: usize) -> f32 {
    let (lower, upper) = layer_bounds(size_px, num_layers);

    if layer < lower {
        return 1.0;
    }
    if upper == 0 {
        // Mirrors the Python's `torch.clamp(size_px, max=1.0)` before `exp`,
        // which stops the unused lanes overflowing to inf. Written as an `if`
        // rather than `f32::min`, because `min` returns the non-NaN operand
        // while `torch.clamp` propagates NaN — and a NaN `size_px` must stay
        // NaN so the resulting alpha fails the `alpha >= alpha_min` test and
        // the fragment is dropped, exactly as it is in Python.
        let clamped = if size_px > 1.0 { 1.0 } else { size_px };
        return (1.0 - SMALL_POINT_CUTOFF) * (clamped - 1.0).exp() + SMALL_POINT_CUTOFF;
    }
    if lower == upper {
        return 1.0;
    }
    // Dead in practice — `upper` is clamped to `L - 1`, so `lower == L - 1`
    // implies `lower == upper` and the branch above already returned. Kept
    // for fidelity with `PointBlending.h:138-143` and with the Python port.
    if lower as usize == num_layers - 1 {
        return 1.0;
    }

    let lo_pow = (1u32 << lower) as f32;
    let hi_pow = (1u32 << upper) as f32;
    let f = (size_px - lo_pow) / (hi_pow - lo_pow);
    if layer == lower {
        1.0 - f
    } else {
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same hand-computed values as `tests/test_raster_layer_factor.py`
    /// (which cites `PointBlending.h:81-149` line numbers directly), so the
    /// two ports can be eyeballed against each other.
    const TOL: f32 = 1e-6;
    const L: usize = 5;

    #[test]
    fn layer_bounds_hand_values() {
        assert_eq!(layer_bounds(0.25, L), (0, 0));
        assert_eq!(layer_bounds(0.5, L), (0, 0));
        assert_eq!(layer_bounds(1.0, L), (0, 0));
        assert_eq!(layer_bounds(1.5, L), (0, 1));
        assert_eq!(layer_bounds(2.0, L), (1, 1));
        assert_eq!(layer_bounds(2.0f32.powf(1.3), L), (1, 2));
        assert_eq!(layer_bounds(4.0, L), (2, 2));
        assert_eq!(layer_bounds(12.0, L), (3, 4));
        assert_eq!(layer_bounds(16.0, L), (4, 4));
        assert_eq!(layer_bounds(100.0, L), (4, 4));
    }

    #[test]
    fn layer_higher_is_the_upper_bound() {
        for s in [0.1f32, 1.0, 1.5, 3.0, 12.0, 1e6] {
            assert_eq!(layer_higher(s, L), layer_bounds(s, L).1);
        }
    }

    #[test]
    fn sub_pixel_points_get_the_exponential_floor() {
        let expected_half = (1.0 - SMALL_POINT_CUTOFF) * (0.5f32 - 1.0).exp() + SMALL_POINT_CUTOFF;
        assert!((expected_half - 0.704_897_99).abs() < TOL);
        assert!((layer_factor(0.5, 0, L) - expected_half).abs() < TOL);

        let expected_tiny = 0.75 * (-0.99f32).exp() + 0.25;
        assert!((layer_factor(0.01, 0, L) - expected_tiny).abs() < TOL);
        assert!(layer_factor(0.01, 0, L) > SMALL_POINT_CUTOFF);

        assert!((layer_factor(1.0, 0, L) - 1.0).abs() < TOL);
    }

    #[test]
    fn exact_power_of_two_gives_factor_one() {
        assert!((layer_factor(2.0, 1, L) - 1.0).abs() < TOL);
        assert!((layer_factor(4.0, 2, L) - 1.0).abs() < TOL);
        assert!((layer_factor(8.0, 3, L) - 1.0).abs() < TOL);
    }

    #[test]
    fn interpolation_is_linear_in_point_size_not_in_log2() {
        // s = 2**1.3 = 2.46 -> 0.23 toward the upper layer, not 0.3.
        let size_px = 2.0f32.powf(1.3);
        let upper = layer_factor(size_px, 2, L);
        let lower = layer_factor(size_px, 1, L);
        let expected_upper = (size_px - 2.0) / (4.0 - 2.0);
        assert!((upper - expected_upper).abs() < TOL);
        assert!((upper - 0.23).abs() < 2e-3);
        assert!((upper - 0.3).abs() > 1e-2);
        assert!((lower - (1.0 - upper)).abs() < TOL);
    }

    #[test]
    fn interpolation_weights_sum_to_one() {
        for (size_px, lower_layer) in [(1.5, 0), (3.0, 1), (6.0, 2), (12.0, 3)] {
            let total = layer_factor(size_px, lower_layer, L) + layer_factor(size_px, lower_layer + 1, L);
            assert!((total - 1.0).abs() < TOL);
        }
    }

    #[test]
    fn sizes_beyond_the_top_layer_clamp_to_one() {
        for size_px in [16.0, 20.0, 100.0, 1e6] {
            assert!((layer_factor(size_px, L as u32 - 1, L) - 1.0).abs() < TOL);
        }
    }

    #[test]
    fn layer_below_lower_returns_one_matching_the_cpp_source() {
        assert!((layer_factor(4.0, 0, L) - 1.0).abs() < TOL);
        assert!((layer_factor(100.0, 3, L) - 1.0).abs() < TOL);
    }

    #[test]
    fn a_nan_size_never_reaches_log2() {
        assert_eq!(layer_bounds(f32::NAN, L), (0, 0));
        // Falls into the `upper == 0` branch, where the NaN must survive the
        // clamp: a NaN alpha then fails `alpha >= alpha_min` and the fragment
        // is dropped, which is what the Python reference does. `f32::min`
        // would have returned 1.0 here and silently emitted the fragment.
        assert!(layer_factor(f32::NAN, 0, L).is_nan());
    }

    /// Relative distance from `x` to the nearer of the two powers of two
    /// bracketing it: `min_k |x - 2^k| / 2^k`.
    ///
    /// Relative rather than in ulps because the `log2` rounding band is a
    /// fixed *relative* width: `f32::log2` returns exactly `k` while
    /// `|log2 x - k|` stays under half an ulp of `k`, and that half-ulp is
    /// `2^(floor(log2 k) - 24)`, so the band in `x` is
    /// `ln(2) * 2^(floor(log2 k) - 24)` wide relative to `2^k` — a couple of
    /// ulps at `k = 5`, a few more at `k = 8`, but never more than about
    /// `1e-6` for any pyramid we render.
    fn relative_distance_to_a_power_of_two(x: f32) -> f32 {
        let exponent = ((x.to_bits() >> 23) & 0xFF) as i32 - 127;
        let below = 2.0f32.powi(exponent);
        let above = 2.0f32.powi(exponent + 1);
        (((x - below) / below).abs()).min(((x - above) / above).abs())
    }

    /// Widest relative distance from a power of two at which `f32::log2`
    /// still rounds to the integer. Quoted in `docs/LIMITATIONS.md`.
    const LOG2_ROUNDING_BAND: f32 = 1e-6;

    #[test]
    fn exponent_and_log2_bounds_differ_only_next_to_a_power_of_two() {
        // The kernel cannot call `log2`, so `layer_bounds` reads the IEEE
        // exponent. Sweep hard, collect every disagreement with the
        // Python-style `log2` formula, and assert each one is the documented
        // rounding case: `f32::log2` is correctly rounded, so within one ulp
        // either side of 2^k it returns exactly `k` and collapses a genuine
        // straddle `(k-1, k)` or `(k, k+1)` into `(k, k)`.
        let mut sizes: Vec<f32> = Vec::new();
        for exponent in -4i32..=8 {
            let power = 2.0f32.powi(exponent);
            for delta in -8i32..=8 {
                sizes.push(f32::from_bits(
                    power.to_bits().wrapping_add(delta as u32),
                ));
            }
        }
        for step in 0..4000 {
            sizes.push(0.01 * 1.005f32.powi(step));
        }

        let mut divergences = 0usize;
        for num_layers in [1usize, 2, 3, 5, 8, 16] {
            for &size_px in &sizes {
                let exact = layer_bounds(size_px, num_layers);
                let via_log2 = layer_bounds_via_log2(size_px, num_layers);
                if exact == via_log2 {
                    continue;
                }
                divergences += 1;
                assert!(
                    relative_distance_to_a_power_of_two(size_px) <= LOG2_ROUNDING_BAND,
                    "divergence at size_px = {size_px} (bits {:#x}), L = {num_layers}: \
                     {exact:?} vs {via_log2:?}, but it is {:e} away from a power of two \
                     (relative), outside the documented {LOG2_ROUNDING_BAND:e} band",
                    size_px.to_bits(),
                    relative_distance_to_a_power_of_two(size_px)
                );
                assert_eq!(
                    via_log2.0, via_log2.1,
                    "log2 should have collapsed the straddle at size_px = {size_px}"
                );
                assert_eq!(
                    exact.1,
                    exact.0 + 1,
                    "the exact bounds should straddle at size_px = {size_px}"
                );
            }
        }
        // The sweep deliberately includes the immediate neighbours of every
        // power of two, so the case must actually have fired -- otherwise
        // this test would pass while proving nothing.
        assert!(divergences > 0, "the ulp sweep never hit the rounding case");
    }

    #[test]
    fn away_from_powers_of_two_the_two_formulas_are_identical() {
        // The complement of the test above: everywhere else, the exponent
        // trick and `log2` agree exactly, which is what makes the GPU port
        // faithful to Python for every realistic point size.
        for step in 0..20_000 {
            let size_px = 0.05 * 1.001f32.powi(step);
            if relative_distance_to_a_power_of_two(size_px) <= LOG2_ROUNDING_BAND {
                continue;
            }
            assert_eq!(
                layer_bounds(size_px, 8),
                layer_bounds_via_log2(size_px, 8),
                "size_px = {size_px}"
            );
        }
    }

    #[test]
    fn bounds_are_exact_at_powers_of_two() {
        // The specific failure mode the exponent trick exists to avoid: a
        // power-of-two size must give lower == upper, so its layer factor is
        // 1.0 and it lands on exactly one layer.
        for exponent in 0u32..5 {
            let size_px = (1u32 << exponent) as f32;
            let (lower, upper) = layer_bounds(size_px, 8);
            assert_eq!((lower, upper), (exponent, exponent), "size_px = {size_px}");
        }
    }

    #[test]
    fn a_size_on_a_power_of_two_is_a_knife_edge_and_this_pins_which_side_is_which() {
        // `layer_bounds` is floor/ceil of log2, so an exact power of two is a
        // discontinuity: `lower == upper` there, and one ulp *below* it the
        // point straddles two layers with almost all the weight on the upper
        // one. This is not a defect to fix -- it is TRIPS's rule -- but it
        // does mean a point whose projected size lands within an ulp of a
        // power of two can be assigned differently by two implementations
        // that compute `fx * size / z` with different associativity. The
        // parity fixtures deliberately avoid the edge; this test keeps the
        // behaviour on both sides of it specified. See docs/LIMITATIONS.md.
        const L: usize = 3;
        let exact = 2.0f32;
        let below = f32::from_bits(exact.to_bits() - 1);
        let above = f32::from_bits(exact.to_bits() + 1);

        assert_eq!(layer_bounds(exact, L), (1, 1));
        assert_eq!(layer_bounds(below, L), (0, 1));
        assert_eq!(layer_bounds(above, L), (1, 2));

        // On the edge, both layers a `Trips` point writes get full weight.
        assert!((layer_factor(exact, 0, L) - 1.0).abs() < TOL, "layer 0 at 2.0");
        assert!((layer_factor(exact, 1, L) - 1.0).abs() < TOL, "layer 1 at 2.0");

        // One ulp below, layer 0 is `lower` and keeps only `1 - f` with
        // f ~= 1, i.e. a weight far under RASTER_ALPHA_MIN (1e-5) -- so its
        // four fragments are dropped at emission and the point contributes
        // half as many. That is the whole 4-fragment step.
        let lost = layer_factor(below, 0, L);
        assert!(lost < 1e-6, "layer 0 one ulp below 2.0 has factor {lost}");
        assert!((layer_factor(below, 1, L) - 1.0).abs() < 1e-6);

        // One ulp above, the *upper* layer is the one that gets ~nothing.
        assert!((layer_factor(above, 1, L) - 1.0).abs() < 1e-6);
        assert!(layer_factor(above, 2, L) < 1e-6);
    }

    #[test]
    fn num_layers_is_respected_not_hardcoded() {
        assert_eq!(layer_bounds(100.0, 3), (2, 2));
        assert_eq!(layer_bounds(100.0, 8), (6, 7));
        assert!((layer_factor(100.0, 2, 3) - 1.0).abs() < TOL);
    }
}
