//! CubeCL kernels for the pyramid forward pass.
//!
//! Module: `brush_pyramid::gpu::kernels`
//! Purpose: the six device-side stages of the forward pass. Every expression
//!     here is a transcription of the corresponding one in [`crate::cpu`],
//!     which is itself checked against the Python `.npy` fixtures — so a GPU
//!     parity failure localises to this file.
//! Invariants:
//!     - **No `log2`.** CubeCL exposes `ln` but not `log2`, and computing
//!       `ln(x) * 1/ln(2)` in f32 lands on the wrong side of an integer for
//!       exact powers of two, which would move a point between pyramid
//!       layers. `exp2_floor` reads the IEEE exponent field instead, which
//!       is exactly `floor(log2 x)` for every positive normal float.
//!     - **Counting and emission cannot disagree.** `project_and_count`
//!       writes a *slot budget* — four slots per selected layer — and
//!       `emit_fragments` derives the layers it writes from that same budget
//!       (`budget / 4`) rather than re-running the selection. Per-corner
//!       bounds and `alpha_min` tests happen only at emission, and a rejected
//!       corner writes the sentinel key `total_layer_pixels`, which sorts
//!       after every real key and lies outside the segment table. No slot is
//!       ever left uninitialised and no atomic is needed anywhere.
//!     - Division is written as `a / b`, never `a * (1 / b)`: the two differ
//!       in the last ulp and the CPU twin uses division.
//!     - Both compositing stop rules are tested *before* consuming a
//!       fragment, so `n_used` and `t_final` describe the composited prefix.
//! Units: as [`crate::cpu`] — `size_px` in layer-0 pixels, depth in world
//!     units, alpha dimensionless.
//! Related docs: `docs/GEOMETRY.md`; `docs/TRIPS_REFERENCE.md` sections 3/3a;
//!     `trippy/raster/metal_src/blend_fwd.metal` (the Metal twin of
//!     `blend_fwd_kernel`).

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::frontend::CompilationArg;
use burn_cubecl::cubecl::frontend::IndexMutExpand;
use burn_cubecl::cubecl::prelude::*;

use crate::params::SMALL_POINT_CUTOFF;

/// Workgroup size for the per-point and per-pixel kernels.
pub const WG_SIZE: u32 = 256;
/// Fragment slots per selected layer: the 2x2 bilinear footprint.
pub const CORNERS_PER_LAYER: u32 = 4;
/// `u32` lanes per layer in the `layer_geom` lookup table.
pub const LAYER_GEOM_LANES: u32 = 4;
/// Fragments each `segment_bounds_kernel` thread inspects.
pub const SEGMENT_CHECKS_PER_ITER: u32 = 8;

/// [`crate::params::Mode::Trilinear`] as a comptime discriminant.
pub const MODE_TRILINEAR: u32 = 0;
/// [`crate::params::Mode::Broadcast`] as a comptime discriminant.
pub const MODE_BROADCAST: u32 = 1;
/// [`crate::params::Mode::Trips`] as a comptime discriminant.
pub const MODE_TRIPS: u32 = 2;

/// [`crate::params::LayerFloor::Zero`] as a comptime discriminant.
pub const LAYER_FLOOR_ZERO: u32 = 0;
/// [`crate::params::LayerFloor::NearLower`] as a comptime discriminant.
pub const LAYER_FLOOR_NEAR_LOWER: u32 = 1;

/// Mirror of [`crate::cpu::MIN_SORT_DEPTH`], inlined for the kernels.
pub const MIN_SORT_DEPTH: f32 = 1e-38;

/// Scalars shared by the per-point kernels.
///
/// The rotation is the **row-major** world-to-camera 3x3 of
/// [`crate::scene::Camera`], flattened; `rNM` is row `N`, column `M`.
#[derive(CubeLaunch, CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct PyramidUniforms {
    /// Row 0 of `R`.
    pub r00: f32,
    /// Row 0 of `R`.
    pub r01: f32,
    /// Row 0 of `R`.
    pub r02: f32,
    /// Row 1 of `R`.
    pub r10: f32,
    /// Row 1 of `R`.
    pub r11: f32,
    /// Row 1 of `R`.
    pub r12: f32,
    /// Row 2 of `R`.
    pub r20: f32,
    /// Row 2 of `R`.
    pub r21: f32,
    /// Row 2 of `R`.
    pub r22: f32,
    /// World-to-camera translation, x.
    pub t0: f32,
    /// World-to-camera translation, y.
    pub t1: f32,
    /// World-to-camera translation, z.
    pub t2: f32,
    /// Focal length x, pixels. Also the one used for `size_px`.
    pub fx: f32,
    /// Focal length y, pixels.
    pub fy: f32,
    /// Principal point x, pixels.
    pub cx: f32,
    /// Principal point y, pixels.
    pub cy: f32,
    /// Near-plane cull, world units.
    pub znear: f32,
    /// Emission-time alpha floor.
    pub alpha_min: f32,
    /// 0.5 or 0.0; subtracted after the per-layer halving.
    pub centre_shift: f32,
    /// Right edge of the conservative cull box, layer-0 pixels.
    pub cull_padded_w: f32,
    /// Bottom edge of the conservative cull box, layer-0 pixels.
    pub cull_padded_h: f32,
    /// Slack added to the cull box, layer-0 pixels.
    pub cull_slack: f32,
    /// `N`.
    pub num_points: u32,
    /// `L`.
    pub num_layers: u32,
    /// `P`; doubles as the sentinel key for a rejected fragment slot.
    pub total_layer_pixels: u32,
    /// Saiga distortion `k1`. All eight are ignored unless the `distort`
    /// comptime flag is set, which the host only does for a non-identity set.
    pub d0: f32,
    /// Saiga distortion `k2`.
    pub d1: f32,
    /// Saiga distortion `k3`.
    pub d2: f32,
    /// Saiga distortion `k4`.
    pub d3: f32,
    /// Saiga distortion `k5`.
    pub d4: f32,
    /// Saiga distortion `k6`.
    pub d5: f32,
    /// Saiga distortion `p1`.
    pub d6: f32,
    /// Saiga distortion `p2`.
    pub d7: f32,
    /// `log2(depth_range.lo)`, for [`crate::params::SortMode::PackedKey`].
    pub depth_log_lo: f32,
    /// `(2^depth_bits - 1) / (log2(hi) - log2(lo))`, the quantisation gain.
    pub depth_log_gain: f32,
    /// Largest depth bucket, `2^depth_bits - 1`.
    pub depth_max_bucket: f32,
    /// Bits the layer-pixel index is shifted left by in a packed key; 0 for
    /// [`crate::params::SortMode::DepthThenKey`].
    pub key_shift: u32,
}

/// `x^2 + y^2` past which [`distort`] returns the cull sentinel; mirror of
/// [`crate::scene::DIST_CUTOFF`] squared.
pub const DIST_CUTOFF_SQ: f32 = 400.0;
/// Mirror of [`crate::scene::DISTORTION_SENTINEL`].
pub const DISTORTION_SENTINEL: f32 = 100_000.0;

/// Saiga's 8-parameter lens distortion; scalar twin of
/// [`crate::scene::distort_normalized`], which carries the full citation.
/// Returns the distorted `(x, y)`, or the cull sentinel past the cutoff.
#[cube]
fn distort(x: f32, y: f32, uniforms: PyramidUniforms) -> (f32, f32) {
    let x2 = x * x;
    let y2 = y * y;
    let r2 = x2 + y2;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let two_xy = 2.0f32 * x * y;
    let radial = (1.0f32 + uniforms.d0 * r2 + uniforms.d1 * r4 + uniforms.d2 * r6)
        / (1.0f32 + uniforms.d3 * r2 + uniforms.d4 * r4 + uniforms.d5 * r6);
    let xd = x * radial + (uniforms.d6 * two_xy + uniforms.d7 * (r2 + 2.0f32 * x2));
    let yd = y * radial + (uniforms.d6 * (r2 + 2.0f32 * y2) + uniforms.d7 * two_xy);
    (
        select(r2 > DIST_CUTOFF_SQ, DISTORTION_SENTINEL, xd),
        select(r2 > DIST_CUTOFF_SQ, DISTORTION_SENTINEL, yd),
    )
}

/// `1 / 2^23`, the mantissa step of an f32 exponent field.
pub const INV_MANTISSA_SCALE: f32 = 1.192_092_9e-7;

/// A monotone, transcendental-free stand-in for `log2` on positive floats.
///
/// Reading the whole IEEE-754 bit pattern as an integer and scaling gives
/// `floor(log2 x) + mantissa_fraction`, i.e. `log2` linearised inside each
/// octave (exact at powers of two, at most 0.086 off in between). Two
/// properties matter here and both are exact: it is **strictly increasing**
/// in `x`, and the host computes the identical expression
/// ([`crate::gpu::fast_log2`]) for the range endpoints, so a depth and the
/// bucket edges are always compared on the same curve. CubeCL has no `log2`
/// anyway (see the module invariants), and `ln` would cost a transcendental
/// per fragment for an answer that is only used to pick a bucket.
#[cube]
fn fast_log2(x: f32) -> f32 {
    (u32::reinterpret(x) as f32) * INV_MANTISSA_SCALE - 127.0f32
}

/// Depth to its bucket index, linear in `log2(depth)`.
///
/// See [`crate::params::DepthRange`] for why the mapping is logarithmic, and
/// [`crate::params::SortMode::PackedKey`] for what the bucket is used for.
#[cube]
fn quantise_depth(depth: f32, uniforms: PyramidUniforms) -> u32 {
    let safe = max(depth, MIN_SORT_DEPTH);
    let bucket = (fast_log2(safe) - uniforms.depth_log_lo) * uniforms.depth_log_gain;
    let clamped = min(max(bucket, 0.0f32), uniforms.depth_max_bucket);
    clamped as u32
}

/// Height of `layer`, from the device-side geometry table.
#[cube]
fn layer_height(layer_geom: &Tensor<u32>, layer: u32) -> u32 {
    layer_geom[(layer * LAYER_GEOM_LANES) as usize]
}

/// Width of `layer`.
#[cube]
fn layer_width(layer_geom: &Tensor<u32>, layer: u32) -> u32 {
    layer_geom[(layer * LAYER_GEOM_LANES + 1u32) as usize]
}

/// First flat layer-pixel index of `layer`.
#[cube]
fn layer_offset(layer_geom: &Tensor<u32>, layer: u32) -> u32 {
    layer_geom[(layer * LAYER_GEOM_LANES + 2u32) as usize]
}

/// `floor(log2(s))` for a positive **normal** float, read straight off the
/// IEEE-754 exponent field. Exact by construction, unlike `ln(s) / ln(2)`.
/// Callers guarantee `s > 1`, so the biased exponent is at least 127.
#[cube]
fn exp2_floor(s: f32) -> u32 {
    ((u32::reinterpret(s) >> 23u32) & 0xFFu32) - 127u32
}

/// True when `s` is an exact power of two (all mantissa bits clear).
#[cube]
fn is_exact_power_of_two(s: f32) -> bool {
    (u32::reinterpret(s) & 0x007F_FFFFu32) == 0u32
}

/// `layer_bounds`' lower bound. See [`crate::factor::layer_bounds`].
#[cube]
fn layer_lower(size_px: f32, num_layers: u32) -> u32 {
    let mut out = 0u32;
    if size_px > 1.0f32 {
        out = min(exp2_floor(size_px), num_layers - 1u32);
    }
    out
}

/// `layer_bounds`' upper bound, i.e. TRIPS's `layer_higher`.
#[cube]
fn layer_upper(size_px: f32, num_layers: u32) -> u32 {
    let mut out = 0u32;
    if size_px > 1.0f32 {
        let floor_log = exp2_floor(size_px);
        // `ceil(log2 s)` is `floor(log2 s)` exactly when s is a power of two.
        let raw = floor_log + select(is_exact_power_of_two(size_px), 0u32, 1u32);
        out = min(raw, num_layers - 1u32);
    }
    out
}

/// TRIPS's `compute_point_size_fac`. See [`crate::factor::layer_factor`] for
/// the annotated version of every branch — including why `layer < lower`
/// returns 1.0 and why the interpolation is linear in point-size units.
// Every arm of the chain below assigns `out`, so its initialiser is dead --
// but `let mut` needs one, and CubeCL has no `return` from a branch.
#[allow(unused_assignments)]
#[cube]
fn layer_factor(size_px: f32, layer: u32, num_layers: u32) -> f32 {
    let lower = layer_lower(size_px, num_layers);
    let upper = layer_upper(size_px, num_layers);
    let mut out = 1.0f32;
    if layer < lower {
        out = 1.0f32;
    } else if upper == 0u32 {
        // `select` rather than `min`, so a NaN size stays NaN and its
        // fragment is dropped by the `alpha >= alpha_min` test.
        let clamped = select(size_px > 1.0f32, 1.0f32, size_px);
        out = (1.0f32 - SMALL_POINT_CUTOFF) * f32::exp(clamped - 1.0f32) + SMALL_POINT_CUTOFF;
    } else if lower == upper {
        out = 1.0f32;
    } else if lower == num_layers - 1u32 {
        // Unreachable (upper is clamped to L-1, so lower == L-1 implies
        // lower == upper); kept to mirror `PointBlending.h:138-143`.
        out = 1.0f32;
    } else {
        let lo_pow = (1u32 << lower) as f32;
        let hi_pow = (1u32 << upper) as f32;
        let f = (size_px - lo_pow) / (hi_pow - lo_pow);
        out = select(layer == lower, 1.0f32 - f, f);
    }
    out
}

/// The finest layer mode `Trips` starts emitting at; twin of
/// [`crate::cpu::trips_start_layer`]. `layer_floor` is comptime, so only one
/// of the two arms is ever compiled into a given pipeline.
#[cube]
fn trips_start_layer(size_px: f32, num_layers: u32, #[comptime] layer_floor: u32) -> u32 {
    let mut out = 0u32;
    if comptime![layer_floor == LAYER_FLOOR_NEAR_LOWER] {
        let lower = layer_lower(size_px, num_layers);
        out = select(lower == 0u32, 0u32, lower - 1u32);
    }
    out
}

/// TRIPS's `valid_point`: all four footprint corners inside `layer`.
#[cube]
fn footprint_fits(
    u_px: f32,
    v_px: f32,
    layer: u32,
    layer_geom: &Tensor<u32>,
    centre_shift: f32,
) -> bool {
    let scale = 1.0f32 / ((1u32 << layer) as f32);
    let base_x = f32::floor(u_px * scale - centre_shift);
    let base_y = f32::floor(v_px * scale - centre_shift);
    let w = layer_width(layer_geom, layer) as f32;
    let h = layer_height(layer_geom, layer) as f32;
    base_x >= 0.0f32 && base_x <= w - 2.0f32 && base_y >= 0.0f32 && base_y <= h - 2.0f32
}

/// How many pyramid layers a visible point writes into.
///
/// Mirrors [`crate::cpu::selected_layers`]. For `Trips` this walks the layers
/// applying TRIPS's gate and stopping at the first failure — the `break` that
/// also suppresses every coarser layer.
#[cube]
fn selected_layer_count(
    u_px: f32,
    v_px: f32,
    size_px: f32,
    layer_geom: &Tensor<u32>,
    num_layers: u32,
    centre_shift: f32,
    #[comptime] mode: u32,
    #[comptime] layer_floor: u32,
) -> u32 {
    let mut count = 0u32;
    if comptime![mode == MODE_BROADCAST] {
        count = num_layers;
    } else if comptime![mode == MODE_TRILINEAR] {
        count = layer_upper(size_px, num_layers) - layer_lower(size_px, num_layers) + 1u32;
    } else {
        let upper = layer_upper(size_px, num_layers);
        let start = trips_start_layer(size_px, num_layers, layer_floor);
        // A bounded `for` over every layer with an `alive` latch, rather than
        // a `while` with a compound condition and an early exit: same
        // semantics (once the gate fails, no coarser layer is counted), but
        // the loop trip count is a small comptime-ish bound and the control
        // flow is uniform, which is friendlier to every shader backend.
        // Layers below `start` are skipped WITHOUT clearing the latch: with
        // the default `LAYER_FLOOR_ZERO`, `start` is 0 and this is a no-op.
        let mut alive = true;
        for layer in 0u32..num_layers {
            if layer < start {
                // Not selected, and not a gate failure either.
            } else if alive
                && layer <= upper
                && footprint_fits(u_px, v_px, layer, layer_geom, centre_shift)
            {
                count += 1u32;
            } else {
                alive = false;
            }
        }
    }
    count
}

/// Stage 1: project every point and reserve its fragment slots.
///
/// Writes `proj` as `(u, v, depth, size_px)` per point and `counts` as the
/// slot budget (0 for a culled point).
#[cube(launch)]
pub fn project_and_count_kernel(
    xyz: &Tensor<f32>,
    size: &Tensor<f32>,
    layer_geom: &Tensor<u32>,
    proj: &mut Tensor<f32>,
    counts: &mut Tensor<u32>,
    uniforms: PyramidUniforms,
    #[comptime] mode: u32,
    #[comptime] layer_floor: u32,
    #[comptime] distorted: bool,
    #[comptime] frustum_cull: bool,
) {
    let gid = ABSOLUTE_POS as u32;
    if gid >= uniforms.num_points {
        terminate!();
    }
    let base = (gid * 3u32) as usize;
    let x = xyz[base];
    let y = xyz[base + 1];
    let z = xyz[base + 2];

    let xc = uniforms.r00 * x + uniforms.r01 * y + uniforms.r02 * z + uniforms.t0;
    let yc = uniforms.r10 * x + uniforms.r11 * y + uniforms.r12 * z + uniforms.t1;
    let zc = uniforms.r20 * x + uniforms.r21 * y + uniforms.r22 * z + uniforms.t2;

    // Written as a division, matching the CPU twin exactly.
    let mut nx = xc / zc;
    let mut ny = yc / zc;
    if comptime![distorted] {
        let (dx, dy) = distort(nx, ny, uniforms);
        nx = dx;
        ny = dy;
    }
    let u_px = uniforms.fx * nx + uniforms.cx;
    let v_px = uniforms.fy * ny + uniforms.cy;
    // TRIPS uses fx only, never fy (`RenderForward.cu:1489`).
    let size_px = uniforms.fx * size[gid as usize] / max(zc, uniforms.znear);

    // The near plane always culls; the box test is the measurable lever, and
    // `frustum_cull` is comptime so turning it off removes the comparisons
    // rather than branching over them.
    let mut visible = zc > uniforms.znear;
    if comptime![frustum_cull] {
        let radius = 0.5f32 * size_px + uniforms.cull_slack;
        visible = visible
            && u_px + radius > 0.0f32
            && u_px - radius < uniforms.cull_padded_w
            && v_px + radius > 0.0f32
            && v_px - radius < uniforms.cull_padded_h;
    }

    let out_base = (gid * 4u32) as usize;
    proj[out_base] = u_px;
    proj[out_base + 1] = v_px;
    proj[out_base + 2] = zc;
    proj[out_base + 3] = size_px;

    let mut budget = 0u32;
    if visible {
        budget = CORNERS_PER_LAYER
            * selected_layer_count(
                u_px,
                v_px,
                size_px,
                layer_geom,
                uniforms.num_layers,
                uniforms.centre_shift,
                mode,
                layer_floor,
            );
    }
    counts[gid as usize] = budget;
}

/// Stage 3: write one fragment per reserved slot.
///
/// `cum_counts` is the **inclusive** prefix sum of `counts`, so a point's
/// first slot is `cum_counts[gid - 1]` (0 for `gid == 0`). A slot whose corner
/// falls outside its layer, or whose alpha is below `alpha_min`, is filled
/// with the sentinel key `total_layer_pixels` instead of being skipped, which
/// is what lets counting and emission stay in lockstep.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn emit_fragments_kernel(
    proj: &Tensor<f32>,
    conf: &Tensor<f32>,
    counts: &Tensor<u32>,
    cum_counts: &Tensor<u32>,
    layer_geom: &Tensor<u32>,
    keys: &mut Tensor<u32>,
    depth_keys: &mut Tensor<u32>,
    alphas: &mut Tensor<f32>,
    point_ids: &mut Tensor<u32>,
    uniforms: PyramidUniforms,
    #[comptime] mode: u32,
    #[comptime] layer_floor: u32,
    #[comptime] packed: bool,
) {
    let gid = ABSOLUTE_POS as u32;
    if gid >= uniforms.num_points {
        terminate!();
    }
    let budget = counts[gid as usize];
    if budget == 0u32 {
        terminate!();
    }
    // Index with `max(gid, 1) - 1` so the read is always in bounds.
    let prev = max(gid, 1u32) - 1u32;
    let slot_base = select(gid == 0u32, 0u32, cum_counts[prev as usize]);
    let num_selected = budget / CORNERS_PER_LAYER;

    let read_base = (gid * 4u32) as usize;
    let u_px = proj[read_base];
    let v_px = proj[read_base + 1];
    let depth = proj[read_base + 2];
    let size_px = proj[read_base + 3];
    let confidence = conf[gid as usize];

    // `Trilinear` starts at `lower`; `Broadcast` at 0; `Trips` at whatever
    // its layer floor says (0 for the exact rule).
    let mut start_layer = 0u32;
    if comptime![mode == MODE_TRILINEAR] {
        start_layer = layer_lower(size_px, uniforms.num_layers);
    } else if comptime![mode == MODE_TRIPS] {
        start_layer = trips_start_layer(size_px, uniforms.num_layers, layer_floor);
    }

    // Positive-float bit patterns order like the values themselves, so the
    // radix sort can order by depth directly.
    let depth_key = u32::reinterpret(max(depth, MIN_SORT_DEPTH));
    // ... and in packed mode the same ordering, coarsened to `key_shift` bits
    // so it can ride in the low half of the one and only sort key.
    let mut depth_bucket = 0u32;
    if comptime![packed] {
        depth_bucket = quantise_depth(depth, uniforms);
    }

    for selected in 0u32..num_selected {
        let layer = start_layer + selected;
        let scale = 1.0f32 / ((1u32 << layer) as f32);
        // The layer coordinate halves exactly, then the pixel-centre shift is
        // applied (`RenderForward.cu:1610`; docs/GEOMETRY.md).
        let cu = u_px * scale - uniforms.centre_shift;
        let cv = v_px * scale - uniforms.centre_shift;
        let base_x = f32::floor(cu);
        let base_y = f32::floor(cv);
        let frac_x = cu - base_x;
        let frac_y = cv - base_y;

        let mut factor = 1.0f32;
        if comptime![mode != MODE_BROADCAST] {
            factor = layer_factor(size_px, layer, uniforms.num_layers);
        }

        let w_l = layer_width(layer_geom, layer);
        let h_l = layer_height(layer_geom, layer);
        let offset = layer_offset(layer_geom, layer);

        // Corner order is TRIPS's `blend_vec` order: index = 2 * dy + dx.
        #[unroll]
        for corner in 0u32..CORNERS_PER_LAYER {
            let dx = corner & 1u32;
            let dy = corner >> 1u32;
            let weight_x = select(dx == 1u32, frac_x, 1.0f32 - frac_x);
            let weight_y = select(dy == 1u32, frac_y, 1.0f32 - frac_y);
            let alpha = weight_x * weight_y * confidence * factor;

            let px = base_x + dx as f32;
            let py = base_y + dy as f32;
            let inside =
                px >= 0.0f32 && py >= 0.0f32 && px < w_l as f32 && py < h_l as f32;
            // Drop, never clamp (docs/GEOMETRY.md bug class 3).
            let keep = inside && alpha >= uniforms.alpha_min;

            // Clamp before the cast: converting a negative float to u32 is
            // implementation-defined, and this value is discarded anyway
            // whenever `keep` is false.
            let safe_x = max(px, 0.0f32) as u32;
            let safe_y = max(py, 0.0f32) as u32;
            let flat = offset + safe_y * w_l + safe_x;

            let slot = (slot_base + selected * CORNERS_PER_LAYER + corner) as usize;
            // `key_shift` is 0 in the exact two-pass mode, so this is the
            // plain layer-pixel index; in packed mode the pixel moves into
            // the high bits and the depth bucket fills the low ones. The
            // sentinel `P << key_shift` still sorts after every real key,
            // because a real key is at most `(P - 1) << key_shift | mask`.
            let mut key = flat;
            if comptime![packed] {
                key = (flat << uniforms.key_shift) | depth_bucket;
            }
            keys[slot] = select(
                keep,
                key,
                uniforms.total_layer_pixels << uniforms.key_shift,
            );
            // The separate depth key only exists for the two-pass sort; in
            // packed mode the depth already rides in `key`, and writing a
            // second 10M-element array per frame would be pure bandwidth.
            if comptime![!packed] {
                depth_keys[slot] = select(keep, depth_key, 0xFFFF_FFFFu32);
            }
            alphas[slot] = select(keep, alpha, 0.0f32);
            point_ids[slot] = gid;
        }
    }
}

/// Stage 5: turn the sorted key array into per-layer-pixel `[start, end)`
/// runs.
///
/// `segments` is `(P, 2)` and must be **zero-initialised**: a layer-pixel with
/// no fragments then reads back as `(0, 0)`, an empty range.
///
/// Unlike Brush's `get_tile_offsets`, the writes are gated individually
/// rather than the whole body being gated on `key < num_pixels`. That matters
/// here because the array ends in sentinel keys: the transition from the last
/// real key to the first sentinel is exactly where that key's `end` has to be
/// written, and a whole-body gate would skip it.
#[cube(launch)]
pub fn segment_bounds_kernel(
    num_fragments: u32,
    num_pixels: u32,
    key_shift: u32,
    sorted_keys: &Tensor<u32>,
    segments: &mut Tensor<u32>,
) {
    let base_id = (ABSOLUTE_POS as u32) * SEGMENT_CHECKS_PER_ITER;

    #[unroll]
    for i in 0u32..SEGMENT_CHECKS_PER_ITER {
        let index = base_id + i;
        if index < num_fragments {
            // The run boundary is a change of layer-PIXEL, not of key: in
            // `SortMode::PackedKey` a key's low `key_shift` bits are the depth
            // bucket, and two depths inside one pixel must not start a new
            // segment. `key_shift` is 0 in the exact mode, where the shift is
            // the identity and this is the original code.
            let pixel = sorted_keys[index as usize] >> key_shift;

            if index == num_fragments - 1u32 && pixel < num_pixels {
                segments[(pixel * 2u32 + 1u32) as usize] = index + 1u32;
            }

            if index == 0u32 {
                if pixel < num_pixels {
                    segments[(pixel * 2u32) as usize] = 0u32;
                }
            } else {
                let prev_pixel = sorted_keys[(index - 1u32) as usize] >> key_shift;
                if pixel != prev_pixel {
                    if prev_pixel < num_pixels {
                        segments[(prev_pixel * 2u32 + 1u32) as usize] = index;
                    }
                    if pixel < num_pixels {
                        segments[(pixel * 2u32) as usize] = index;
                    }
                }
            }
        }
    }
}

/// Stage 6: front-to-back alpha compositing, one thread per layer-pixel.
///
/// The direct twin of `trippy/raster/metal_src/blend_fwd.metal`. Every write
/// goes to an address owned by exactly one thread, so there are no atomics.
/// `num_channels` is comptime, so the accumulator lives in registers and both
/// inner loops unroll. The feature buffer's element type `F` is generic so the
/// same source serves [`crate::params::FeatureStore::F32`] and `F16`; the
/// accumulator is always f32, so only the *stored* features are rounded.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn blend_fwd_kernel<F: Float>(
    segments: &Tensor<u32>,
    permutation: &Tensor<u32>,
    alphas: &Tensor<f32>,
    point_ids: &Tensor<u32>,
    feat: &Tensor<F>,
    out: &mut Tensor<f32>,
    t_final: &mut Tensor<f32>,
    n_used: &mut Tensor<u32>,
    num_pixels: u32,
    max_frags: u32,
    t_cutoff: f32,
    #[comptime] num_channels: u32,
) {
    let gid = ABSOLUTE_POS as u32;
    if gid >= num_pixels {
        terminate!();
    }
    let start = segments[(gid * 2u32) as usize];
    let end = segments[(gid * 2u32 + 1u32) as usize];

    let mut acc = Array::<f32>::new(comptime![num_channels as usize]);
    #[unroll]
    for c in 0u32..num_channels {
        acc[c as usize] = 0.0f32;
    }

    let mut transmittance = 1.0f32;
    let mut used = 0u32;
    let mut index = start;
    let mut running = true;

    while running && index < end {
        // Both stopping rules are checked BEFORE consuming the fragment, so
        // `used` and `t_final` describe exactly the composited prefix.
        if used >= max_frags || transmittance < t_cutoff {
            running = false;
        } else {
            let slot = permutation[index as usize];
            let alpha = alphas[slot as usize];
            let point = point_ids[slot as usize];
            let weight = transmittance * alpha;
            let feat_base = point * num_channels;
            #[unroll]
            for c in 0u32..num_channels {
                acc[c as usize] += weight * f32::cast_from(feat[(feat_base + c) as usize]);
            }
            transmittance *= 1.0f32 - alpha;
            used += 1u32;
            index += 1u32;
        }
    }

    let out_base = gid * num_channels;
    #[unroll]
    for c in 0u32..num_channels {
        out[(out_base + c) as usize] = acc[c as usize];
    }
    t_final[gid as usize] = transmittance;
    n_used[gid as usize] = used;
}

/// Fill `out` with `0, 1, 2, ...`: the identity permutation the first radix
/// pass sorts. Done on device so a large render never round-trips an index
/// array through host memory.
#[cube(launch)]
pub fn iota_kernel(out: &mut Tensor<u32>, count: u32) {
    let gid = ABSOLUTE_POS as u32;
    if gid >= count {
        terminate!();
    }
    out[gid as usize] = gid;
}

/// Stage 7: `out += t_final * bg`, one thread per layer-pixel.
///
/// A separate pass rather than a tail on `blend_fwd_kernel`, to keep the
/// background exactly where TRIPS puts it — outside the compositing loop
/// (`RenderForward.cu:3610-3620`) — and to keep the blend kernel's register
/// budget free of the background vector.
#[cube(launch)]
pub fn add_background_kernel(
    background: &Tensor<f32>,
    t_final: &Tensor<f32>,
    out: &mut Tensor<f32>,
    num_pixels: u32,
    #[comptime] num_channels: u32,
) {
    let gid = ABSOLUTE_POS as u32;
    if gid >= num_pixels {
        terminate!();
    }
    let transmittance = t_final[gid as usize];
    let base = gid * num_channels;
    #[unroll]
    for c in 0u32..num_channels {
        out[(base + c) as usize] += transmittance * background[c as usize];
    }
}
