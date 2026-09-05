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
) -> u32 {
    let mut count = 0u32;
    if comptime![mode == MODE_BROADCAST] {
        count = num_layers;
    } else if comptime![mode == MODE_TRILINEAR] {
        count = layer_upper(size_px, num_layers) - layer_lower(size_px, num_layers) + 1u32;
    } else {
        let upper = layer_upper(size_px, num_layers);
        // A bounded `for` over every layer with an `alive` latch, rather than
        // a `while` with a compound condition and an early exit: same
        // semantics (once the gate fails, no coarser layer is counted), but
        // the loop trip count is a small comptime-ish bound and the control
        // flow is uniform, which is friendlier to every shader backend.
        let mut alive = true;
        for layer in 0u32..num_layers {
            if alive && layer <= upper && footprint_fits(u_px, v_px, layer, layer_geom, centre_shift)
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
    let u_px = uniforms.fx * xc / zc + uniforms.cx;
    let v_px = uniforms.fy * yc / zc + uniforms.cy;
    // TRIPS uses fx only, never fy (`RenderForward.cu:1489`).
    let size_px = uniforms.fx * size[gid as usize] / max(zc, uniforms.znear);

    let radius = 0.5f32 * size_px + uniforms.cull_slack;
    let visible = zc > uniforms.znear
        && u_px + radius > 0.0f32
        && u_px - radius < uniforms.cull_padded_w
        && v_px + radius > 0.0f32
        && v_px - radius < uniforms.cull_padded_h;

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

    // `Trilinear` starts at `lower`; the other two start at layer 0.
    let mut start_layer = 0u32;
    if comptime![mode == MODE_TRILINEAR] {
        start_layer = layer_lower(size_px, uniforms.num_layers);
    }

    // Positive-float bit patterns order like the values themselves, so the
    // radix sort can order by depth directly.
    let depth_key = u32::reinterpret(max(depth, MIN_SORT_DEPTH));

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
            keys[slot] = select(keep, flat, uniforms.total_layer_pixels);
            depth_keys[slot] = select(keep, depth_key, 0xFFFF_FFFFu32);
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
    sorted_keys: &Tensor<u32>,
    segments: &mut Tensor<u32>,
) {
    let base_id = (ABSOLUTE_POS as u32) * SEGMENT_CHECKS_PER_ITER;

    #[unroll]
    for i in 0u32..SEGMENT_CHECKS_PER_ITER {
        let index = base_id + i;
        if index < num_fragments {
            let key = sorted_keys[index as usize];

            if index == num_fragments - 1u32 && key < num_pixels {
                segments[(key * 2u32 + 1u32) as usize] = index + 1u32;
            }

            if index == 0u32 {
                if key < num_pixels {
                    segments[(key * 2u32) as usize] = 0u32;
                }
            } else {
                let prev_key = sorted_keys[(index - 1u32) as usize];
                if key != prev_key {
                    if prev_key < num_pixels {
                        segments[(prev_key * 2u32 + 1u32) as usize] = index;
                    }
                    if key < num_pixels {
                        segments[(key * 2u32) as usize] = index;
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
/// inner loops unroll.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn blend_fwd_kernel(
    segments: &Tensor<u32>,
    permutation: &Tensor<u32>,
    alphas: &Tensor<f32>,
    point_ids: &Tensor<u32>,
    feat: &Tensor<f32>,
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
                acc[c as usize] += weight * feat[(feat_base + c) as usize];
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
