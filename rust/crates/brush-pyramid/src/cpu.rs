//! CPU reference forward pass: project, emit, sort, segment, blend.
//!
//! Module: `brush_pyramid::cpu`
//! Purpose: the twin of the CubeCL pipeline in `brush_pyramid::gpu`, written the
//!     obvious way with explicit loops. It is what the GPU kernels are
//!     measured against on machines with no usable GPU, and — because it is
//!     itself checked against Python `.npy` fixtures by
//!     `tests/parity_cpu.rs` — it makes a GPU parity failure immediately
//!     attributable to either the kernels or the shared geometry.
//!     Mirrors `trippy.raster.ref_numpy.render_pyramid_numpy` and the
//!     emission half of `trippy.raster.emit`.
//! Invariants:
//!     - **Out-of-bounds fragments are dropped, never clamped**
//!       (`docs/GEOMETRY.md` historical bug class 3). Mode `Trips` instead
//!       uses TRIPS's stricter all-four-corners gate, and that gate is a
//!       `break`: failing at layer `l` suppresses every *coarser* layer too.
//!     - The sort is by `(layer_pixel, float32 depth bits)` and **stable**, so
//!       equal depths keep emission order, which is ascending point index.
//!       This reproduces `trippy.raster.sort.sort_fragments`' composite key
//!       exactly (a point can never contribute two fragments to one
//!       layer-pixel, so "emission order" and "ascending point id" agree).
//!     - Both compositing stop rules are checked *before* consuming a
//!       fragment, so `n_used` and `t_final` describe exactly the composited
//!       prefix (`trippy/raster/metal_src/blend_fwd.metal`).
//!     - float32 throughout, matching the GPU. The Python fixtures are
//!       rendered in float32 too.
//! Units / frames: see [`crate::scene`]; `depth` is camera-space z in world
//!     units, `size_px` is layer-0 pixels.
//! Related docs: `docs/GEOMETRY.md`; `docs/TRIPS_REFERENCE.md` sections 3,
//!     3a, 10; `docs/ARCHITECTURE.md`.

use crate::factor::{layer_bounds, layer_factor};
use crate::grid::LayerGrid;
use crate::output::{LayerImage, PyramidImages};
use crate::params::{Mode, PyramidParams, CULL_MARGIN_COARSE_PX};
use crate::scene::{Camera, PointSet};

/// Smallest positive float32. Depth is clamped to it before its bit pattern
/// is used as a sort key, because "IEEE bits sort like the value" only holds
/// for non-negative floats (`trippy.raster.sort._MIN_SORT_DEPTH`).
pub const MIN_SORT_DEPTH: f32 = 1e-38;

/// Offsets of the four bilinear footprint corners, in TRIPS's `blend_vec`
/// order (`PointBlending.h:216-240`): index = `2 * dy + dx`.
pub const CORNER_DX: [i64; 4] = [0, 1, 0, 1];
/// Companion to [`CORNER_DX`].
pub const CORNER_DY: [i64; 4] = [0, 0, 1, 1];

/// One emitted fragment, before sorting.
#[derive(Debug, Clone, Copy)]
struct Fragment {
    /// Flat layer-major pyramid index (see [`LayerGrid`]).
    layer_pixel: u32,
    /// float32 depth bits — the sort key's low half.
    depth_bits: u32,
    /// Row index into the point arrays.
    point_id: u32,
    /// `bilinear_weight * conf * layer_factor`.
    alpha: f32,
}

/// A projected point: everything emission needs, computed once.
#[derive(Debug, Clone, Copy)]
pub struct Projected {
    /// Continuous layer-0 pixel coordinate, `u` (right).
    pub u: f32,
    /// Continuous layer-0 pixel coordinate, `v` (down).
    pub v: f32,
    /// Camera-space z, world units.
    pub depth: f32,
    /// Projected size in layer-0 pixels, `fx * size / max(z, znear)`.
    pub size_px: f32,
    /// Passed the near-plane and visibility cull.
    pub visible: bool,
}

/// Project one point and apply the conservative visibility cull.
///
/// The cull must never remove a point that the exact per-fragment bounds test
/// would have kept, or a border point silently loses its coarse-layer
/// contribution. Two effects widen the box beyond the image: the coarsest
/// layer covers up to `2^(L-1) - 1` layer-0 columns past `W`, and a fragment
/// at layer `l` survives for `uv / 2^l` anywhere in `(-1.5, w_l + 0.5)`.
/// Port of `trippy.raster.emit.project_points` + `cull_points`.
///
/// # Arguments
/// - `camera`: intrinsics and world-to-camera pose.
/// - `x`, `y`, `z`: the point's world position.
/// - `size`: its world-unit radius.
/// - `grid`: the pyramid being rendered into.
/// - `znear`: near-plane cull, world units.
#[must_use]
pub fn project_point(
    camera: &Camera,
    x: f32,
    y: f32,
    z: f32,
    size: f32,
    grid: &LayerGrid,
    znear: f32,
) -> Projected {
    let (xc, yc, zc) = camera.world_to_cam(x, y, z);
    let u = camera.fx * xc / zc + camera.cx;
    let v = camera.fy * yc / zc + camera.cy;
    // TRIPS uses fx only, not fy (`RenderForward.cu:1489`).
    let size_px = camera.fx * size / zc.max(znear);

    let num_layers = grid.num_layers();
    let coarse = (1usize << (num_layers - 1)) as f32;
    let (h_coarse, w_coarse) = grid.shapes()[num_layers - 1];
    let padded_h = h_coarse as f32 * coarse;
    let padded_w = w_coarse as f32 * coarse;
    let radius = 0.5 * size_px + CULL_MARGIN_COARSE_PX * coarse;

    let visible = zc > znear
        && u + radius > 0.0
        && u - radius < padded_w
        && v + radius > 0.0
        && v - radius < padded_h;

    Projected {
        u,
        v,
        depth: zc,
        size_px,
        visible,
    }
}

/// Number of `(layer, corner)` slots a point reserves, and its layer factors.
///
/// This is the *budget*, not the final fragment count: it applies the layer
/// selection rule (including mode `Trips`'s gate and its `break`) but not the
/// per-corner bounds test or the `alpha_min` floor, both of which are applied
/// at emission. Splitting the two is what lets the GPU's counting kernel and
/// its emission kernel agree without having to reproduce a float comparison
/// bit-for-bit across two separate shader compilations.
///
/// # Arguments
/// - `p`: the projected point.
/// - `grid`, `params`: pyramid geometry and render options.
/// - `layers_out`: filled with the layers the point writes into, in
///   increasing order. Cleared first.
///
/// # Returns
/// `4 * layers_out.len()`, the slot budget.
pub fn selected_layers(
    p: &Projected,
    grid: &LayerGrid,
    params: &PyramidParams,
    layers_out: &mut Vec<u32>,
) -> u32 {
    layers_out.clear();
    if !p.visible {
        return 0;
    }
    let num_layers = grid.num_layers();
    let shift = params.pixel_center.shift();
    let (lower, upper) = layer_bounds(p.size_px, num_layers);

    match params.mode {
        Mode::Broadcast => layers_out.extend(0..num_layers as u32),
        Mode::Trilinear => layers_out.extend(lower..=upper),
        Mode::Trips => {
            // `for (layer = 0; layer <= layer_higher; ++layer, ip *= 0.5f)`
            // with `if (!valid_point(floor(ip), z, layer)) break;`
            // (`RenderForward.cu:340-352`). The gate requires all four
            // footprint corners in bounds, and it is a BREAK: failing at
            // layer l suppresses every coarser layer as well.
            for layer in 0..=upper {
                let (h_l, w_l) = grid.shapes()[layer as usize];
                let scale = 1.0 / (1u32 << layer) as f32;
                let base_x = (p.u * scale - shift).floor();
                let base_y = (p.v * scale - shift).floor();
                if !(base_x >= 0.0
                    && base_x <= (w_l as f32) - 2.0
                    && base_y >= 0.0
                    && base_y <= (h_l as f32) - 2.0)
                {
                    break;
                }
                layers_out.push(layer);
            }
        }
    }
    layers_out.len() as u32 * 4
}

/// The four corner fragments a point contributes at one layer.
///
/// Returns, per corner in [`CORNER_DX`]/[`CORNER_DY`] order, either the
/// `(layer_pixel, alpha)` pair or `None` when the corner falls outside the
/// layer or below `alpha_min`. Dropping — never clamping — is bug class 3 in
/// `docs/GEOMETRY.md`.
///
/// # Arguments
/// - `p`: the projected point.
/// - `layer`: which pyramid layer.
/// - `conf`: the point's effective confidence.
/// - `grid`, `params`: pyramid geometry and render options.
#[must_use]
pub fn corner_fragments(
    p: &Projected,
    layer: u32,
    conf: f32,
    grid: &LayerGrid,
    params: &PyramidParams,
) -> [Option<(u32, f32)>; 4] {
    let (h_l, w_l) = grid.shapes()[layer as usize];
    let shift = params.pixel_center.shift();
    // The layer coordinate halves exactly, with no additive term
    // (`RenderForward.cu:1610`); `shift` re-anchors the 2x2 footprint on
    // pixel centres *after* the halving.
    let scale = 1.0 / (1u32 << layer) as f32;
    let cu = p.u * scale - shift;
    let cv = p.v * scale - shift;
    let base_x = cu.floor();
    let base_y = cv.floor();
    let frac_x = cu - base_x;
    let frac_y = cv - base_y;

    let factor = match params.mode {
        Mode::Broadcast => 1.0,
        Mode::Trilinear | Mode::Trips => layer_factor(p.size_px, layer, grid.num_layers()),
    };

    let mut out = [None; 4];
    for corner in 0..4usize {
        let dx = CORNER_DX[corner];
        let dy = CORNER_DY[corner];
        let weight_x = if dx == 1 { frac_x } else { 1.0 - frac_x };
        let weight_y = if dy == 1 { frac_y } else { 1.0 - frac_y };
        let alpha = weight_x * weight_y * conf * factor;
        let px = base_x as i64 + dx;
        let py = base_y as i64 + dy;
        if px < 0 || py < 0 || px >= w_l as i64 || py >= h_l as i64 {
            continue;
        }
        if !(alpha >= params.alpha_min) {
            continue;
        }
        out[corner] = Some((
            grid.flat_index(layer as usize, py as usize, px as usize) as u32,
            alpha,
        ));
    }
    out
}

/// float32 depth to a `u32` whose unsigned ordering matches the depth
/// ordering. Port of `trippy.raster.sort._depth_key_bits`.
#[must_use]
pub fn depth_sort_key(depth: f32) -> u32 {
    depth.max(MIN_SORT_DEPTH).to_bits()
}

/// Per-point fragment **slot budget**, the quantity the GPU's counting kernel
/// writes and its prefix sum turns into write offsets.
///
/// Exposed so `tests/parity_gpu.rs` can diff the GPU's `counts` buffer against
/// the CPU's, which localises a layer-selection disagreement to a specific
/// point instead of leaving only a whole-image fragment-count mismatch.
///
/// # Arguments
/// - `points`, `camera`, `params`: as [`render_pyramid_cpu`].
///
/// # Errors
/// Returns `Err` if the pyramid geometry is invalid.
pub fn slot_budgets(
    points: &PointSet,
    camera: &Camera,
    params: &PyramidParams,
) -> Result<(Vec<u32>, Vec<Projected>), String> {
    let grid = LayerGrid::new(camera.height, camera.width, params.num_layers, params.halving)?;
    let mut layers_buf = Vec::with_capacity(params.num_layers);
    let mut budgets = Vec::with_capacity(points.len());
    let mut projected = Vec::with_capacity(points.len());
    for i in 0..points.len() {
        let p = project_point(
            camera,
            points.xyz[i * 3],
            points.xyz[i * 3 + 1],
            points.xyz[i * 3 + 2],
            points.size[i],
            &grid,
            params.znear,
        );
        budgets.push(selected_layers(&p, &grid, params, &mut layers_buf));
        projected.push(p);
    }
    Ok((budgets, projected))
}

/// Fragments emitted per layer-pixel, `P` values in flat layer-major order.
///
/// The CPU twin of the GPU's segment table. Comparing the two localises a
/// disagreement to a single pyramid pixel, which `num_fragments` alone
/// cannot.
///
/// # Arguments
/// - `points`, `camera`, `params`: as [`render_pyramid_cpu`].
///
/// # Errors
/// Returns `Err` if the pyramid geometry is invalid.
pub fn fragments_per_pixel(
    points: &PointSet,
    camera: &Camera,
    params: &PyramidParams,
) -> Result<(Vec<u32>, LayerGrid), String> {
    let grid = LayerGrid::new(camera.height, camera.width, params.num_layers, params.halving)?;
    let mut counts = vec![0u32; grid.total()];
    let mut layers_buf = Vec::with_capacity(params.num_layers);
    for i in 0..points.len() {
        let p = project_point(
            camera,
            points.xyz[i * 3],
            points.xyz[i * 3 + 1],
            points.xyz[i * 3 + 2],
            points.size[i],
            &grid,
            params.znear,
        );
        selected_layers(&p, &grid, params, &mut layers_buf);
        for &layer in &layers_buf {
            for slot in corner_fragments(&p, layer, points.conf[i], &grid, params) {
                if let Some((layer_pixel, _)) = slot {
                    counts[layer_pixel as usize] += 1;
                }
            }
        }
    }
    Ok((counts, grid))
}

/// Render one image as an `L`-layer alpha-composited pyramid, on the CPU.
///
/// # Arguments
/// - `points`: the point set; `points.num_channels` sets `C`.
/// - `camera`: intrinsics and world-to-camera pose; `camera.width/height` set
///   the layer-0 image size.
/// - `params`: layer-selection mode, pyramid depth, stop rules (see
///   [`PyramidParams`]).
/// - `background`: `C` values composited as `out += t_final * bg`, exactly as
///   TRIPS does after the kernel (`RenderForward.cu:3610-3620`). `None` means
///   a zero background.
///
/// # Errors
/// Returns `Err` if the pyramid geometry is invalid or `background` is the
/// wrong length.
pub fn render_pyramid_cpu(
    points: &PointSet,
    camera: &Camera,
    params: &PyramidParams,
    background: Option<&[f32]>,
) -> Result<PyramidImages, String> {
    let channels = points.num_channels;
    if let Some(bg) = background {
        if bg.len() != channels {
            return Err(format!(
                "background has {} values, expected {channels}",
                bg.len()
            ));
        }
    }
    let grid = LayerGrid::new(camera.height, camera.width, params.num_layers, params.halving)?;

    // --- project + cull ---------------------------------------------------
    let projected: Vec<Projected> = (0..points.len())
        .map(|i| {
            project_point(
                camera,
                points.xyz[i * 3],
                points.xyz[i * 3 + 1],
                points.xyz[i * 3 + 2],
                points.size[i],
                &grid,
                params.znear,
            )
        })
        .collect();

    // --- emit -------------------------------------------------------------
    let mut fragments: Vec<Fragment> = Vec::new();
    let mut layers_buf: Vec<u32> = Vec::with_capacity(params.num_layers);
    for (i, p) in projected.iter().enumerate() {
        selected_layers(p, &grid, params, &mut layers_buf);
        let depth_bits = depth_sort_key(p.depth);
        for &layer in &layers_buf {
            for slot in corner_fragments(p, layer, points.conf[i], &grid, params) {
                if let Some((layer_pixel, alpha)) = slot {
                    fragments.push(Fragment {
                        layer_pixel,
                        depth_bits,
                        point_id: i as u32,
                        alpha,
                    });
                }
            }
        }
    }

    // --- sort: (layer, pixel) then depth, stable -------------------------
    // `sort_by_key` is stable, so equal (layer_pixel, depth) keep emission
    // order. Emission is point-major here and layer-major in the Python
    // reference, but a point contributes at most one fragment to any single
    // layer-pixel, so within a segment both orders are "ascending point id".
    fragments.sort_by_key(|f| (f.layer_pixel, f.depth_bits));

    // --- segment offsets --------------------------------------------------
    let total = grid.total();
    let mut segment_offsets = vec![0u32; total + 1];
    for f in &fragments {
        segment_offsets[f.layer_pixel as usize + 1] += 1;
    }
    for p in 0..total {
        segment_offsets[p + 1] += segment_offsets[p];
    }

    // --- blend ------------------------------------------------------------
    let mut out = vec![0f32; total * channels];
    let mut t_final = vec![1f32; total];
    let mut n_used = vec![0u32; total];
    for pixel in 0..total {
        let start = segment_offsets[pixel] as usize;
        let end = segment_offsets[pixel + 1] as usize;
        let mut transmittance = 1f32;
        let mut used = 0u32;
        for f in &fragments[start..end] {
            // Both stopping rules are checked BEFORE consuming the fragment.
            if used >= params.max_frags || transmittance < params.t_cutoff {
                break;
            }
            let weight = transmittance * f.alpha;
            let feat = &points.feat[f.point_id as usize * channels..][..channels];
            for c in 0..channels {
                out[pixel * channels + c] += weight * feat[c];
            }
            transmittance *= 1.0 - f.alpha;
            used += 1;
        }
        t_final[pixel] = transmittance;
        n_used[pixel] = used;
    }

    // --- background + split into layers -----------------------------------
    let mut layers = Vec::with_capacity(grid.num_layers());
    for (layer, &(h_l, w_l)) in grid.shapes().iter().enumerate() {
        let lo = grid.offsets()[layer];
        let mut feature = vec![0f32; channels * h_l * w_l];
        for y in 0..h_l {
            for x in 0..w_l {
                let flat = lo + y * w_l + x;
                for c in 0..channels {
                    let mut value = out[flat * channels + c];
                    if let Some(bg) = background {
                        value += t_final[flat] * bg[c];
                    }
                    feature[(c * h_l + y) * w_l + x] = value;
                }
            }
        }
        layers.push(LayerImage {
            height: h_l,
            width: w_l,
            channels,
            feature,
            t_final: t_final[lo..lo + h_l * w_l].to_vec(),
            n_used: n_used[lo..lo + h_l * w_l].to_vec(),
        });
    }

    Ok(PyramidImages {
        fragments_per_layer: grid.fragments_per_layer(&segment_offsets),
        num_fragments: fragments.len() as u32,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{PixelCenter, PyramidHalving};

    fn camera(width: usize, height: usize) -> Camera {
        Camera {
            width,
            height,
            fx: 10.0,
            fy: 10.0,
            cx: width as f32 / 2.0,
            cy: height as f32 / 2.0,
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            t: [0.0, 0.0, 0.0],
        }
    }

    fn params(mode: Mode, num_layers: usize) -> PyramidParams {
        PyramidParams {
            mode,
            num_layers,
            pixel_center: PixelCenter::Half,
            halving: PyramidHalving::Ceil,
            ..PyramidParams::default()
        }
    }

    /// One point on the optical axis, its footprint centred on a pixel
    /// corner so all four bilinear weights are 0.25.
    fn single_point(size: f32, conf: f32, channels: usize) -> PointSet {
        PointSet::new(
            vec![0.0, 0.0, 1.0],
            vec![size],
            vec![1.0; channels],
            vec![conf],
            channels,
        )
        .expect("point set")
    }

    #[test]
    fn depth_sort_key_is_monotonic_in_depth() {
        let mut previous = 0u32;
        for depth in [1e-30f32, 1e-3, 0.5, 1.0, 2.0, 100.0, 1e30] {
            let key = depth_sort_key(depth);
            assert!(key > previous, "depth {depth} key {key} <= {previous}");
            previous = key;
        }
    }

    #[test]
    fn a_single_centred_point_splits_its_alpha_over_four_pixels() {
        // 16x16 image, fx = 10, point at (0, 0, 1) -> u = v = 8.0 exactly.
        // With pixel_center "half", `8.0 - 0.5 = 7.5`, so base = 7 and both
        // fracs are 0.5: four corners at weight 0.25 each.
        let points = single_point(0.01, 0.8, 3);
        let p = params(Mode::Broadcast, 1);
        let img = render_pyramid_cpu(&points, &camera(16, 16), &p, None).expect("render");
        assert_eq!(img.num_fragments, 4);
        let layer = &img.layers[0];
        for (y, x) in [(7, 7), (7, 8), (8, 7), (8, 8)] {
            // Broadcast mode's factor is 1, so alpha = 0.25 * conf.
            let alpha = 0.25 * 0.8;
            assert!((layer.at(0, y, x) - alpha).abs() < 1e-6, "{}", layer.at(0, y, x));
            assert!((layer.t_final[y * 16 + x] - (1.0 - alpha)).abs() < 1e-6);
            assert_eq!(layer.n_used[y * 16 + x], 1);
        }
        // Everything else is untouched: nothing drawn means t_final == 1.
        assert!((layer.at(0, 0, 0) - 0.0).abs() < 1e-9);
        assert!((layer.t_final[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn out_of_bounds_corners_are_dropped_never_clamped() {
        // Put the point just off the left edge so only its right-hand corners
        // land inside. u = fx * x / z + cx; choose u = 0.4 -> base = -1,
        // so corners at x = -1 (dropped) and x = 0 (kept).
        let mut points = single_point(0.01, 0.8, 3);
        points.xyz[0] = (0.4 - 8.0) / 10.0;
        let p = params(Mode::Broadcast, 1);
        let img = render_pyramid_cpu(&points, &camera(16, 16), &p, None).expect("render");
        assert_eq!(img.num_fragments, 2, "only the x = 0 column survives");
        let layer = &img.layers[0];
        // Column 0 got weight frac_x = 0.4 - (-1) - ... : just assert it is
        // non-zero and that nothing wrapped around to the far edge.
        assert!(layer.at(0, 7, 0) > 0.0);
        assert!((layer.at(0, 7, 15) - 0.0).abs() < 1e-9, "wrapped to the right edge");
    }

    #[test]
    fn trips_mode_gate_suppresses_coarser_layers_after_a_failure() {
        // A big point near the edge: it fits layer 0 but its layer-1
        // footprint hangs off, so `Trips` must write layer 0 only, whereas
        // `Broadcast` writes both.
        let mut points = single_point(0.5, 0.8, 3);
        // u = 1.2 at layer 0 -> layer 1 coordinate 0.6, minus 0.5 = 0.1,
        // base 0, and w_1 - 2 = 6, so layer 1 would pass... push further out.
        points.xyz[0] = (0.6 - 8.0) / 10.0;
        points.xyz[1] = 0.0;
        let grid_params = params(Mode::Trips, 2);
        let img = render_pyramid_cpu(&points, &camera(16, 16), &grid_params, None).expect("render");
        assert_eq!(
            img.fragments_per_layer[1], 0,
            "layer 1 must be suppressed by the break"
        );
        assert!(img.fragments_per_layer[0] > 0);
    }

    #[test]
    fn max_frags_caps_the_composited_prefix() {
        // 40 identical points stacked on one line of sight at increasing
        // depth, low alpha so transmittance never reaches the cutoff.
        let n = 40usize;
        let mut xyz = Vec::new();
        for i in 0..n {
            xyz.extend_from_slice(&[0.0, 0.0, 1.0 + i as f32 * 0.1]);
        }
        let points = PointSet::new(
            xyz,
            vec![0.001; n],
            vec![1.0; n * 3],
            vec![0.02; n],
            3,
        )
        .expect("points");
        let mut p = params(Mode::Broadcast, 1);
        p.max_frags = 16;
        let img = render_pyramid_cpu(&points, &camera(16, 16), &p, None).expect("render");
        assert_eq!(img.layers[0].n_used[7 * 16 + 7], 16);
        assert!(img.layers[0].t_final[7 * 16 + 7] > p.t_cutoff);
    }

    #[test]
    fn t_cutoff_stops_before_max_frags_when_alpha_is_high() {
        let n = 40usize;
        let mut xyz = Vec::new();
        for i in 0..n {
            xyz.extend_from_slice(&[0.0, 0.0, 1.0 + i as f32 * 0.1]);
        }
        let points =
            PointSet::new(xyz, vec![0.001; n], vec![1.0; n * 3], vec![0.99; n], 3).expect("points");
        let mut p = params(Mode::Broadcast, 1);
        p.max_frags = 1024;
        let img = render_pyramid_cpu(&points, &camera(16, 16), &p, None).expect("render");
        let used = img.layers[0].n_used[7 * 16 + 7];
        assert!(used < 1024, "cutoff should have stopped the loop, used {used}");
        assert!(img.layers[0].t_final[7 * 16 + 7] < p.t_cutoff);
    }

    #[test]
    fn background_is_weighted_by_the_remaining_transmittance() {
        let points = single_point(0.01, 0.8, 3);
        let p = params(Mode::Broadcast, 1);
        let bg = [0.1f32, 0.2, 0.3];
        let plain = render_pyramid_cpu(&points, &camera(16, 16), &p, None).expect("render");
        let with_bg = render_pyramid_cpu(&points, &camera(16, 16), &p, Some(&bg)).expect("render");
        for c in 0..3 {
            // An untouched pixel gets the full background.
            assert!((with_bg.layers[0].at(c, 0, 0) - bg[c]).abs() < 1e-6);
            // A covered pixel gets t_final * bg on top of the composited value.
            let expected = plain.layers[0].at(c, 7, 7) + plain.layers[0].t_final[7 * 16 + 7] * bg[c];
            assert!((with_bg.layers[0].at(c, 7, 7) - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn a_background_of_the_wrong_length_is_rejected() {
        let points = single_point(0.01, 0.8, 3);
        let p = params(Mode::Broadcast, 1);
        assert!(render_pyramid_cpu(&points, &camera(16, 16), &p, Some(&[0.0, 0.0])).is_err());
    }

    #[test]
    fn points_behind_the_near_plane_are_culled() {
        let mut points = single_point(0.01, 0.8, 3);
        points.xyz[2] = -1.0;
        let p = params(Mode::Broadcast, 1);
        let img = render_pyramid_cpu(&points, &camera(16, 16), &p, None).expect("render");
        assert_eq!(img.num_fragments, 0);
    }

    /// The kernel cannot `break` out of a loop as freely as the CPU can, so
    /// `gpu::kernels::selected_layer_count` walks every layer with an
    /// `alive` latch instead. This is that latch form, written out so the
    /// restructuring can be checked against the `break` form on the CPU.
    fn selected_layers_latch_form(
        p: &Projected,
        grid: &LayerGrid,
        params: &PyramidParams,
    ) -> u32 {
        if !p.visible {
            return 0;
        }
        let num_layers = grid.num_layers();
        let shift = params.pixel_center.shift();
        let (lower, upper) = layer_bounds(p.size_px, num_layers);
        match params.mode {
            Mode::Broadcast => num_layers as u32 * 4,
            Mode::Trilinear => (upper - lower + 1) * 4,
            Mode::Trips => {
                let mut count = 0u32;
                let mut alive = true;
                for layer in 0..num_layers as u32 {
                    let (h_l, w_l) = grid.shapes()[layer as usize];
                    let scale = 1.0 / (1u32 << layer) as f32;
                    let base_x = (p.u * scale - shift).floor();
                    let base_y = (p.v * scale - shift).floor();
                    let fits = base_x >= 0.0
                        && base_x <= (w_l as f32) - 2.0
                        && base_y >= 0.0
                        && base_y <= (h_l as f32) - 2.0;
                    if alive && layer <= upper && fits {
                        count += 1;
                    } else {
                        alive = false;
                    }
                }
                count * 4
            }
        }
    }

    #[test]
    fn the_kernels_latch_loop_matches_the_cpus_break_loop() {
        // Sweep points across the whole frame, including well outside it, so
        // the gate fails at every layer from 0 upward and the `break` /
        // latch difference would show up if the transcription were wrong.
        let camera = camera(64, 48);
        let grid = LayerGrid::new(48, 64, 3, PyramidHalving::Ceil).expect("grid");
        let mut layers_buf = Vec::new();
        let mut checked = 0usize;
        let mut nonzero = 0usize;
        for mode in [Mode::Trips, Mode::Trilinear, Mode::Broadcast] {
            for pixel_center in [PixelCenter::Half, PixelCenter::Integer] {
                let p = PyramidParams {
                    mode,
                    num_layers: 3,
                    pixel_center,
                    halving: PyramidHalving::Ceil,
                    ..PyramidParams::default()
                };
                for ui in -20..100 {
                    for vi in -20..80 {
                        for &size_px in &[0.3f32, 1.5, 3.0, 6.0, 40.0] {
                            let proj = Projected {
                                u: ui as f32 * 0.97,
                                v: vi as f32 * 0.97,
                                depth: 2.0,
                                size_px,
                                visible: true,
                            };
                            let want = selected_layers(&proj, &grid, &p, &mut layers_buf);
                            let got = selected_layers_latch_form(&proj, &grid, &p);
                            assert_eq!(
                                want, got,
                                "mode {mode:?} {pixel_center:?} at u={} v={} size_px={size_px}",
                                proj.u, proj.v
                            );
                            checked += 1;
                            if want > 0 {
                                nonzero += 1;
                            }
                        }
                    }
                }
            }
        }
        let _ = camera;
        assert!(checked > 100_000, "swept only {checked} cases");
        assert!(nonzero > 1000, "only {nonzero} cases selected any layer");
    }

    #[test]
    fn deeper_fragments_composite_after_nearer_ones() {
        // Two points on one line of sight, far one listed FIRST in the input,
        // each fully opaque-ish. The near one must dominate the output.
        let points = PointSet::new(
            vec![0.0, 0.0, 5.0, 0.0, 0.0, 1.0],
            vec![0.001, 0.001],
            // far point is channel 0, near point is channel 1.
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.99, 0.99],
            2,
        )
        .expect("points");
        let p = params(Mode::Broadcast, 1);
        let img = render_pyramid_cpu(&points, &camera(16, 16), &p, None).expect("render");
        let near = img.layers[0].at(1, 7, 7);
        let far = img.layers[0].at(0, 7, 7);
        assert!(near > far, "near {near} should outweigh far {far}");
    }
}
