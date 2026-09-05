//! GPU forward pass: Burn/wgpu host code driving the CubeCL kernels.
//!
//! Module: `brush_pyramid::gpu`
//! Purpose: the production path. Runs on Metal today through wgpu and on
//!     WebGPU later from the same source, and hands back Burn tensors so the
//!     U-Net decoder and the viewer's backbuffer can consume them without a
//!     round trip through host memory.
//! Invariants:
//!     - Same six stages, same arithmetic and same stop rules as
//!       [`crate::cpu`]; `tests/parity_gpu.rs` asserts both reproduce the
//!       Python fixtures.
//!     - The sort is **two stable radix passes, depth then key**, mirroring
//!       Brush's own depth-then-tile pattern in
//!       `brush-render/src/render.rs`. Stability is what makes the second
//!       pass preserve the depth order inside each layer-pixel.
//!     - Rejected fragment slots carry the sentinel key `P`, which sorts
//!       after every real key and is excluded from the segment table, so they
//!       are never composited.
//!     - Background is added **after** the blend kernel, as
//!       `out += t_final * bg`, exactly as TRIPS does
//!       (`RenderForward.cu:3610-3620`).
//!     - The only host synchronisation is one readback of the total fragment
//!       count, which is genuinely needed to size the fragment buffers.
//! Units / frames: see [`crate::scene`].
//! Related docs: `docs/ARCHITECTURE.md`; `rust/README.md`;
//!     `docs/decisions/ADR-0005-brush-fork-layout.md`.

pub mod burn_bridge;
pub mod kernels;

use brush_cube::{
    calc_cube_count_1d, create_tensor, create_tensor_from_slice, MainBackendBase, Runtime,
};
use brush_prefix_sum::prefix_sum;
use brush_sort::radix_argsort;
use burn::backend::ops::{FloatTensorOps, IntTensorOps};
use burn::tensor::{DType, IntDType};
use burn_cubecl::cubecl::CubeDim;
use burn_wgpu::{CubeTensor, WgpuRuntime};

pub use burn_wgpu::WgpuDevice;

/// The system default wgpu device — Metal on macOS, WebGPU in the browser.
#[must_use]
pub fn default_device() -> WgpuDevice {
    WgpuDevice::DefaultDevice
}

/// Drive a future to completion on the calling thread.
///
/// [`render_pyramid`] and [`PyramidRender::to_host`] are `async` only because
/// they each contain one device readback. Tests, examples and CLI tools have no
/// executor of their own, and pulling in a full async runtime just to `await`
/// twice would put it on `scripts/test.sh`'s critical path — so this parks the
/// thread and lets the waker unpark it. A real application (the viewer) already
/// has a runtime and should `await` normally instead.
///
/// # Arguments
/// - `future`: the future to run to completion.
pub fn block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    struct Unparker(std::thread::Thread);
    impl Wake for Unparker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }

    let waker = Waker::from(Arc::new(Unparker(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

use crate::grid::LayerGrid;
use crate::output::{LayerImage, PyramidImages};
use crate::params::{
    FeatureStore, LayerFloor, Mode, PyramidParams, SortMode, CULL_MARGIN_COARSE_PX,
    SUPPORTED_CHANNELS,
};
use crate::scene::{is_identity_distortion, Camera, PointSet};
use kernels::{PyramidUniformsLaunch, CORNERS_PER_LAYER, LAYER_GEOM_LANES, WG_SIZE};

/// The backend these kernels run on: Burn's CubeCL/wgpu backend.
pub type Backend = MainBackendBase;

/// The half-precision element type [`FeatureStore::F16`] stores features as.
pub type HalfF32 = burn::tensor::f16;

/// Fewest depth buckets [`SortMode::PackedKey`] is worth using: with under
/// this many bits left under the pixel index, every fragment in a pixel would
/// land in the same bucket and the "sort" would be emission order.
pub const MIN_PACKED_DEPTH_BITS: u32 = 4;

/// A rendered pyramid, still on the device.
///
/// The buffers stay flat and device-resident: [`PyramidRender::feature_buffer`]
/// hands out the composited `(P, C)` buffer for a viewer to bind, and
/// [`PyramidRender::to_host`] reads everything back for tests and offline
/// tools.
pub struct PyramidRender {
    grid: LayerGrid,
    channels: usize,
    num_fragment_slots: u32,
    /// `(P, C)` f32, background already composited in.
    out: CubeTensor<WgpuRuntime>,
    /// `(P,)` f32.
    t_final: CubeTensor<WgpuRuntime>,
    /// `(P,)` u32.
    n_used: CubeTensor<WgpuRuntime>,
    /// `(P, 2)` u32 `[start, end)` runs.
    segments: CubeTensor<WgpuRuntime>,
    /// `(N,)` u32 per-point slot budgets, kept for diagnostics.
    counts: CubeTensor<WgpuRuntime>,
}

impl PyramidRender {
    /// The pyramid geometry this was rendered into.
    #[must_use]
    pub fn grid(&self) -> &LayerGrid {
        &self.grid
    }

    /// Feature width `C`.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Fragment **slots** allocated (the reserved budget, which is `>=` the
    /// number of fragments that survived emission). Use
    /// [`PyramidImages::num_fragments`] from [`Self::to_host`] for the real
    /// count.
    #[must_use]
    pub fn num_fragment_slots(&self) -> u32 {
        self.num_fragment_slots
    }

    /// The composited feature buffer, `(P, C)` f32, layer-major and with the
    /// background already added.
    ///
    /// Device-resident and zero-copy: this is the buffer the viewer binds
    /// (compare `brush-render`'s `resolve_to_cube_float`, which exists to hand
    /// exactly this type to a wgpu pipeline). Layer `l` occupies rows
    /// `grid().offsets()[l] .. + h_l * w_l`.
    ///
    /// It is deliberately **not** a `burn::Tensor<4>`. In the Burn revision
    /// this fork pins, `Tensor<const D>` is backend-erased over the *fusion*
    /// backend, and the only public route from a raw `CubeTensor` back into
    /// one is a registered fusion `Operation` — the ~90 lines of `BindOp`
    /// boilerplate in `brush-render/src/burn_glue.rs`, normally generated by
    /// `#[backend_extension]` + `#[derive(ExtensionType)]`. That wrapper
    /// belongs with the U-Net/viewer hookup that will consume it; see
    /// `docs/LIMITATIONS.md`.
    #[must_use]
    pub fn feature_buffer(&self) -> &CubeTensor<WgpuRuntime> {
        &self.out
    }

    /// Pyramid layer `l` as a `burn::Tensor<4>` in NCHW, `[1, C, h_l, w_l]`.
    ///
    /// This is the U-Net's input. The composited buffer is one flat
    /// `(P, C)` allocation, pixel-major and channel-last; this slices layer
    /// `l`'s rows out of it, reshapes to `[1, h_l, w_l, C]` and permutes to
    /// NCHW. Everything happens **on the device** — the only host work is the
    /// one fusion `Operation` registration in
    /// [`burn_bridge::float_tensor`].
    ///
    /// # Arguments
    /// - `layer`: pyramid level, 0 = finest.
    ///
    /// # Panics
    /// Panics if `layer >= grid().num_layers()`.
    #[must_use]
    pub fn layer_tensor(&self, layer: usize) -> burn::tensor::Tensor<4> {
        let (rows, h_l, w_l) = self.layer_span(layer);
        let flat: burn::tensor::Tensor<2> = burn_bridge::float_tensor(self.out.clone());
        flat.slice([rows])
            .reshape([1, h_l, w_l, self.channels])
            .permute([0, 3, 1, 2])
    }

    /// Every pyramid layer as a `Tensor<4>`, finest first — exactly the list
    /// `brush_unet`'s decoder consumes.
    #[must_use]
    pub fn layer_tensors(&self) -> Vec<burn::tensor::Tensor<4>> {
        (0..self.grid.num_layers())
            .map(|layer| self.layer_tensor(layer))
            .collect()
    }

    /// Remaining transmittance per layer-pixel, `(P,)` f32. 1.0 means nothing
    /// was drawn — the coverage/honesty map.
    #[must_use]
    pub fn t_final_buffer(&self) -> &CubeTensor<WgpuRuntime> {
        &self.t_final
    }

    /// Fragments composited per layer-pixel, `(P,)` u32.
    #[must_use]
    pub fn n_used_buffer(&self) -> &CubeTensor<WgpuRuntime> {
        &self.n_used
    }

    /// Row range of layer `l` inside the flat `(P, C)` buffers, plus its
    /// `(height, width)`.
    ///
    /// # Panics
    /// Panics if `layer >= grid().num_layers()`.
    #[must_use]
    pub fn layer_span(&self, layer: usize) -> (std::ops::Range<usize>, usize, usize) {
        let (h_l, w_l) = self.grid.shapes()[layer];
        let lo = self.grid.offsets()[layer];
        (lo..lo + h_l * w_l, h_l, w_l)
    }

    /// The per-point fragment **slot budget** the counting kernel produced.
    ///
    /// Diagnostic only: comparing this with [`crate::cpu::slot_budgets`]
    /// localises a layer-selection disagreement to a specific point, which a
    /// whole-image fragment count cannot.
    ///
    /// # Errors
    /// Returns `Err` if the readback fails.
    pub async fn slot_counts(&self) -> Result<Vec<u32>, String> {
        read_u32(self.counts.clone()).await
    }

    /// Fragments per layer-pixel, read off the segment table: the GPU twin of
    /// [`crate::cpu::fragments_per_pixel`].
    ///
    /// # Errors
    /// Returns `Err` if the readback fails.
    pub async fn fragments_per_pixel(&self) -> Result<Vec<u32>, String> {
        let segments = read_u32(self.segments.clone()).await?;
        Ok((0..self.grid.total())
            .map(|p| segments[p * 2 + 1].saturating_sub(segments[p * 2]))
            .collect())
    }

    /// Read every buffer back to host memory as a [`PyramidImages`].
    ///
    /// This synchronises with the device and is meant for tests and offline
    /// tools, not for a per-frame viewer path.
    ///
    /// # Errors
    /// Returns `Err` if any readback fails.
    pub async fn to_host(&self) -> Result<PyramidImages, String> {
        let out = read_f32(self.out.clone()).await?;
        let t_final = read_f32(self.t_final.clone()).await?;
        let n_used = read_u32(self.n_used.clone()).await?;
        let segments = read_u32(self.segments.clone()).await?;

        let total = self.grid.total();
        // Rebuild the `(P + 1,)` offset form from the `(P, 2)` runs so
        // `fragments_per_layer` can be read off it exactly as on the CPU.
        let mut segment_offsets = vec![0u32; total + 1];
        for pixel in 0..total {
            let count = segments[pixel * 2 + 1].saturating_sub(segments[pixel * 2]);
            segment_offsets[pixel + 1] = segment_offsets[pixel] + count;
        }

        let mut layers = Vec::with_capacity(self.grid.num_layers());
        for (layer, &(h_l, w_l)) in self.grid.shapes().iter().enumerate() {
            let lo = self.grid.offsets()[layer];
            let mut feature = vec![0f32; self.channels * h_l * w_l];
            for y in 0..h_l {
                for x in 0..w_l {
                    let flat = lo + y * w_l + x;
                    for c in 0..self.channels {
                        feature[(c * h_l + y) * w_l + x] = out[flat * self.channels + c];
                    }
                }
            }
            layers.push(LayerImage {
                height: h_l,
                width: w_l,
                channels: self.channels,
                feature,
                t_final: t_final[lo..lo + h_l * w_l].to_vec(),
                n_used: n_used[lo..lo + h_l * w_l].to_vec(),
            });
        }

        Ok(PyramidImages {
            fragments_per_layer: self.grid.fragments_per_layer(&segment_offsets),
            num_fragments: segment_offsets[total],
            layers,
        })
    }
}

/// Wall-clock cost of each forward-pass stage, milliseconds.
///
/// Only produced by [`render_pyramid_timed`], which inserts a real device
/// `sync` at every stage boundary. Those syncs cost a few hundred microseconds
/// each and destroy the pipelining a viewer relies on, so the numbers are a
/// *profile*, not a frame time: read [`Self::total_ms`] as an upper bound and
/// use [`render_pyramid`] plus one barrier for the honest whole-frame figure.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StageTimings {
    /// Stage 1, project + cull + count slots.
    pub project_count_ms: f64,
    /// Stage 2, prefix sum, including the one host readback of the total.
    pub prefix_ms: f64,
    /// Stage 3, emit fragments.
    pub emit_ms: f64,
    /// Stage 4, the radix sort(s).
    pub sort_ms: f64,
    /// Stage 5, segment bounds.
    pub segment_ms: f64,
    /// Stages 6 and 7, blend and background.
    pub blend_ms: f64,
    /// Sum of the above.
    pub total_ms: f64,
    /// Fragment slots reserved, i.e. how much the sort actually moved.
    pub fragment_slots: u32,
    /// Radix passes performed (four bits each).
    pub radix_passes: u32,
}

/// A monotone stand-in for `log2` on positive floats, identical to the
/// kernels' [`kernels::fast_log2`] down to the last bit.
///
/// Used to place [`crate::params::DepthRange`]'s endpoints on the same curve
/// the depth quantisation uses, so a depth exactly at `lo` lands in bucket 0
/// and one exactly at `hi` in the last bucket.
///
/// # Arguments
/// - `x`: a positive float.
#[must_use]
pub fn fast_log2(x: f32) -> f32 {
    (x.to_bits() as f32) * kernels::INV_MANTISSA_SCALE - 127.0
}

/// Wait for every kernel queued on `device` to finish.
///
/// The honest end-of-frame barrier for a benchmark: unlike reading a tensor
/// back, it moves no data, so the measured time is the render's and not the
/// readback's.
///
/// # Arguments
/// - `device`: the device to drain.
///
/// # Errors
/// Returns `Err` if the device reports a failure while draining.
pub async fn sync(device: &WgpuDevice) -> Result<(), String> {
    WgpuRuntime::<burn_wgpu::AutoCompiler>::client(device)
        .sync()
        .await
        .map_err(|e| format!("device sync failed: {e:?}"))
}

/// Bits the layer-pixel index needs, i.e. the width of the exact sort key.
fn key_bits_for(total_pixels: usize) -> u32 {
    (u32::BITS - (total_pixels as u32).leading_zeros()).max(1)
}

/// Render one image as an `L`-layer alpha-composited pyramid, on the GPU.
///
/// # Arguments
/// - `device`: the wgpu device to run on (`WgpuDevice::DefaultDevice` picks
///   the system default — Metal on macOS).
/// - `points`: the point set; `points.num_channels` sets `C` and must be one
///   of [`SUPPORTED_CHANNELS`], because the blend kernel specialises on it.
/// - `camera`: intrinsics, lens distortion and world-to-camera pose.
/// - `params`: layer-selection mode, pyramid depth, stop rules and the
///   performance levers.
/// - `background`: `C` values composited as `out += t_final * bg`; `None`
///   means a zero background.
///
/// # Errors
/// Returns `Err` on invalid geometry, an unsupported `C`, or a failed
/// readback of the fragment count.
pub async fn render_pyramid(
    device: &WgpuDevice,
    points: &PointSet,
    camera: &Camera,
    params: &PyramidParams,
    background: Option<&[f32]>,
) -> Result<PyramidRender, String> {
    let (render, _) = render_inner(device, points, camera, params, background, false).await?;
    Ok(render)
}

/// [`render_pyramid`] with a device `sync` at every stage boundary, returning
/// the per-stage wall times as well.
///
/// # Errors
/// As [`render_pyramid`], plus any device synchronisation failure.
pub async fn render_pyramid_timed(
    device: &WgpuDevice,
    points: &PointSet,
    camera: &Camera,
    params: &PyramidParams,
    background: Option<&[f32]>,
) -> Result<(PyramidRender, StageTimings), String> {
    let (render, timings) = render_inner(device, points, camera, params, background, true).await?;
    Ok((render, timings.unwrap_or_default()))
}

// `clock` is reassigned by the last `mark!` and then dropped; that final
// assignment is dead by construction and rewriting the macro to avoid it would
// make every other stage's timing harder to read.
#[allow(clippy::too_many_lines, unused_assignments)]
async fn render_inner(
    device: &WgpuDevice,
    points: &PointSet,
    camera: &Camera,
    params: &PyramidParams,
    background: Option<&[f32]>,
    timed: bool,
) -> Result<(PyramidRender, Option<StageTimings>), String> {
    let channels = points.num_channels;
    if !SUPPORTED_CHANNELS.contains(&channels) {
        return Err(format!(
            "C = {channels} is not supported on the GPU path; the blend kernel is \
             specialised for {SUPPORTED_CHANNELS:?}"
        ));
    }
    if let Some(bg) = background {
        if bg.len() != channels {
            return Err(format!(
                "background has {} values, expected {channels}",
                bg.len()
            ));
        }
    }
    let grid = LayerGrid::new(camera.height, camera.width, params.num_layers, params.halving)?;
    let total_pixels = grid.total();
    let num_points = points.len();
    let client = WgpuRuntime::client(device);

    let key_bits = key_bits_for(total_pixels);
    // `PackedKey` needs room for at least a couple of depth buckets under the
    // pixel index. If the pyramid is so large that it does not have it, fall
    // back to the exact two-pass sort rather than silently losing depth
    // ordering altogether.
    let depth_bits = 32 - key_bits;
    let packed = params.sort == SortMode::PackedKey && depth_bits >= MIN_PACKED_DEPTH_BITS;
    let key_shift = if packed { depth_bits } else { 0 };

    let mut timings = timed.then(StageTimings::default);
    let mut clock = std::time::Instant::now();
    // A real device sync, not a one-element readback: these are eager CubeCL
    // launches, so `sync` genuinely waits for the queue to drain.
    macro_rules! mark {
        ($field:ident) => {
            if let Some(t) = timings.as_mut() {
                client
                    .sync()
                    .await
                    .map_err(|e| format!("device sync failed: {e:?}"))?;
                t.$field = clock.elapsed().as_secs_f64() * 1e3;
                clock = std::time::Instant::now();
            }
        };
    }

    // --- upload -----------------------------------------------------------
    // `.max(1)`: a zero-length buffer is not a valid binding.
    let xyz = create_tensor_from_slice(pad1(&points.xyz), device, DType::F32);
    let size = create_tensor_from_slice(pad1(&points.size), device, DType::F32);
    let conf = create_tensor_from_slice(pad1(&points.conf), device, DType::F32);
    let feat = match params.feature_store {
        FeatureStore::F32 => create_tensor_from_slice(pad1(&points.feat), device, DType::F32),
        FeatureStore::F16 => {
            let half: Vec<HalfF32> = pad1(&points.feat).iter().copied().map(HalfF32::from_f32).collect();
            create_tensor_from_slice(&half, device, DType::F16)
        }
    };

    // (L, 4) u32: height, width, flat offset, unused padding lane.
    let mut geom = Vec::with_capacity(grid.num_layers() * LAYER_GEOM_LANES as usize);
    for (layer, &(h_l, w_l)) in grid.shapes().iter().enumerate() {
        geom.extend_from_slice(&[h_l as u32, w_l as u32, grid.offsets()[layer] as u32, 0]);
    }
    let layer_geom = create_tensor_from_slice(&geom, device, DType::U32);

    // `PyramidUniformsLaunch` is not `Clone`, so each launch gets its own.
    let mode_code = match params.mode {
        Mode::Trilinear => kernels::MODE_TRILINEAR,
        Mode::Broadcast => kernels::MODE_BROADCAST,
        Mode::Trips => kernels::MODE_TRIPS,
    };
    let floor_code = match params.layer_floor {
        LayerFloor::Zero => kernels::LAYER_FLOOR_ZERO,
        LayerFloor::NearLower => kernels::LAYER_FLOOR_NEAR_LOWER,
    };
    let distorted = !is_identity_distortion(&camera.distortion);
    let cube_dim = CubeDim::new_1d(WG_SIZE);
    let points_dispatch = calc_cube_count_1d(num_points.max(1) as u32, WG_SIZE);
    let uniforms = |grid: &LayerGrid| build_uniforms(camera, params, grid, num_points, key_shift);

    // --- stage 1: project & count ----------------------------------------
    let proj = create_tensor([num_points.max(1) * 4], device, DType::F32);
    let counts = Backend::int_zeros([num_points.max(1)].into(), device, IntDType::U32);
    kernels::project_and_count_kernel::launch::<WgpuRuntime>(
        &client,
        points_dispatch.clone(),
        cube_dim,
        xyz.into_tensor_arg(),
        size.into_tensor_arg(),
        layer_geom.clone().into_tensor_arg(),
        proj.clone().into_tensor_arg(),
        counts.clone().into_tensor_arg(),
        uniforms(&grid),
        mode_code,
        floor_code,
        distorted,
        params.frustum_cull,
    );
    mark!(project_count_ms);

    // --- stage 2: prefix sum ---------------------------------------------
    let cum_counts = prefix_sum(counts.clone());
    let counts_for_readback = counts.clone();

    // --- the one host sync: how many fragment slots were reserved ---------
    let num_slots = if num_points == 0 {
        0
    } else {
        let last = Backend::int_slice(cum_counts.clone(), &[(num_points - 1..num_points).into()]);
        read_u32(last).await?[0]
    };
    let buffer_len = (num_slots as usize).max(1);
    mark!(prefix_ms);

    // --- stage 3: emit ----------------------------------------------------
    let keys = create_tensor([buffer_len], device, DType::U32);
    // Only the two-pass sort has a separate depth key. `PackedKey` still needs
    // a bindable buffer for the kernel argument, but one element of it.
    let depth_keys = create_tensor(
        [if packed { 1 } else { buffer_len }],
        device,
        DType::U32,
    );
    let alphas = create_tensor([buffer_len], device, DType::F32);
    let point_ids = create_tensor([buffer_len], device, DType::U32);
    if num_slots > 0 {
        kernels::emit_fragments_kernel::launch::<WgpuRuntime>(
            &client,
            points_dispatch,
            cube_dim,
            proj.into_tensor_arg(),
            conf.into_tensor_arg(),
            counts.into_tensor_arg(),
            cum_counts.into_tensor_arg(),
            layer_geom.into_tensor_arg(),
            keys.clone().into_tensor_arg(),
            depth_keys.clone().into_tensor_arg(),
            alphas.clone().into_tensor_arg(),
            point_ids.clone().into_tensor_arg(),
            uniforms(&grid),
            mode_code,
            floor_code,
            packed,
        );
    }
    mark!(emit_ms);

    // --- stage 4: order the fragments ------------------------------------
    let identity = create_tensor([buffer_len], device, DType::U32);
    kernels::iota_kernel::launch::<WgpuRuntime>(
        &client,
        calc_cube_count_1d(buffer_len as u32, WG_SIZE),
        cube_dim,
        identity.clone().into_tensor_arg(),
        buffer_len as u32,
    );
    let (sorted_keys, permutation, radix_passes) = if packed {
        // One pass: the key already carries the (quantised) depth in its low
        // bits, so a single ascending sort puts every layer-pixel run in
        // depth order.
        let (k, v) = radix_argsort(keys, identity, 32);
        (k, v, 32u32.div_ceil(4))
    } else {
        // Pass 1: order by depth. Positive-float bit patterns sort like the
        // values, and the sentinel slots carry 0xFFFFFFFF so they trail.
        let (_, by_depth) = radix_argsort(depth_keys, identity, 32);
        // Pass 2: order by key. LSB radix is stable, so the depth order
        // survives inside every layer-pixel run — which is exactly Brush's
        // own depth-then-tile pattern.
        let keys_by_depth = Backend::int_gather(0, keys, by_depth.clone());
        let (k, v) = radix_argsort(keys_by_depth, by_depth, key_bits);
        (k, v, 32u32.div_ceil(4) + key_bits.div_ceil(4))
    };
    mark!(sort_ms);

    // --- stage 5: segment offsets ----------------------------------------
    let segments = Backend::int_zeros([total_pixels, 2].into(), device, IntDType::U32);
    if num_slots > 0 {
        kernels::segment_bounds_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(num_slots, WG_SIZE * kernels::SEGMENT_CHECKS_PER_ITER),
            cube_dim,
            num_slots,
            total_pixels as u32,
            key_shift,
            sorted_keys.into_tensor_arg(),
            segments.clone().into_tensor_arg(),
        );
    }
    mark!(segment_ms);

    // --- stage 6: blend ---------------------------------------------------
    let out = create_tensor([total_pixels, channels], device, DType::F32);
    let t_final = create_tensor([total_pixels], device, DType::F32);
    let n_used = create_tensor([total_pixels], device, DType::U32);
    let blend_dispatch = calc_cube_count_1d(total_pixels as u32, WG_SIZE);
    match params.feature_store {
        FeatureStore::F32 => kernels::blend_fwd_kernel::launch::<f32, WgpuRuntime>(
            &client,
            blend_dispatch.clone(),
            cube_dim,
            segments.clone().into_tensor_arg(),
            permutation.into_tensor_arg(),
            alphas.into_tensor_arg(),
            point_ids.into_tensor_arg(),
            feat.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            t_final.clone().into_tensor_arg(),
            n_used.clone().into_tensor_arg(),
            total_pixels as u32,
            params.max_frags,
            params.t_cutoff,
            channels as u32,
        ),
        FeatureStore::F16 => kernels::blend_fwd_kernel::launch::<HalfF32, WgpuRuntime>(
            &client,
            blend_dispatch.clone(),
            cube_dim,
            segments.clone().into_tensor_arg(),
            permutation.into_tensor_arg(),
            alphas.into_tensor_arg(),
            point_ids.into_tensor_arg(),
            feat.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            t_final.clone().into_tensor_arg(),
            n_used.clone().into_tensor_arg(),
            total_pixels as u32,
            params.max_frags,
            params.t_cutoff,
            channels as u32,
        ),
    }

    // --- stage 7: background, added after the blend ----------------------
    if let Some(bg) = background {
        let bg_tensor = create_tensor_from_slice(bg, device, DType::F32);
        kernels::add_background_kernel::launch::<WgpuRuntime>(
            &client,
            blend_dispatch,
            cube_dim,
            bg_tensor.into_tensor_arg(),
            t_final.clone().into_tensor_arg(),
            out.clone().into_tensor_arg(),
            total_pixels as u32,
            channels as u32,
        );
    }
    mark!(blend_ms);

    if let Some(t) = timings.as_mut() {
        t.fragment_slots = num_slots;
        t.radix_passes = radix_passes;
        t.total_ms = t.project_count_ms
            + t.prefix_ms
            + t.emit_ms
            + t.sort_ms
            + t.segment_ms
            + t.blend_ms;
    }

    Ok((
        PyramidRender {
            grid,
            channels,
            num_fragment_slots: num_slots,
            out,
            t_final,
            n_used,
            segments,
            counts: counts_for_readback,
        },
        timings,
    ))
}

/// A zero-length slice cannot be bound as a buffer; substitute one element.
fn pad1(values: &[f32]) -> &[f32] {
    if values.is_empty() {
        &[0.0]
    } else {
        values
    }
}

fn build_uniforms(
    camera: &Camera,
    params: &PyramidParams,
    grid: &LayerGrid,
    num_points: usize,
    key_shift: u32,
) -> PyramidUniformsLaunch<WgpuRuntime> {
    let num_layers = grid.num_layers();
    let coarse = (1usize << (num_layers - 1)) as f32;
    let (h_coarse, w_coarse) = grid.shapes()[num_layers - 1];
    let r = &camera.r;
    // Depth quantisation for `SortMode::PackedKey`. `key_shift == 0` means the
    // exact two-pass sort, where none of these three lanes is read.
    let max_bucket = ((1u64 << key_shift) - 1) as f32;
    let depth_log_lo = fast_log2(params.depth_range.lo.max(f32::MIN_POSITIVE));
    // No widening here: the caller chose the range, and silently stretching it
    // would spend depth buckets on empty space. A degenerate `hi <= lo` falls
    // through to gain 0, i.e. one bucket, which is the documented worst case
    // (emission order inside a pixel) rather than a divide by zero.
    let depth_log_hi = fast_log2(params.depth_range.hi.max(params.depth_range.lo));
    let depth_log_gain = if depth_log_hi > depth_log_lo {
        max_bucket / (depth_log_hi - depth_log_lo)
    } else {
        0.0
    };
    PyramidUniformsLaunch::new(
        r[0],
        r[1],
        r[2],
        r[3],
        r[4],
        r[5],
        r[6],
        r[7],
        r[8],
        camera.t[0],
        camera.t[1],
        camera.t[2],
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        params.znear,
        params.alpha_min,
        params.pixel_center.shift(),
        w_coarse as f32 * coarse,
        h_coarse as f32 * coarse,
        CULL_MARGIN_COARSE_PX * coarse,
        num_points as u32,
        num_layers as u32,
        grid.total() as u32,
        camera.distortion[0],
        camera.distortion[1],
        camera.distortion[2],
        camera.distortion[3],
        camera.distortion[4],
        camera.distortion[5],
        camera.distortion[6],
        camera.distortion[7],
        depth_log_lo,
        depth_log_gain,
        max_bucket,
        key_shift,
    )
}

async fn read_f32(tensor: CubeTensor<WgpuRuntime>) -> Result<Vec<f32>, String> {
    let data = Backend::float_into_data(tensor)
        .await
        .map_err(|e| format!("readback failed: {e:?}"))?;
    data.into_vec::<f32>()
        .map_err(|e| format!("expected f32 data: {e:?}"))
}

async fn read_u32(tensor: CubeTensor<WgpuRuntime>) -> Result<Vec<u32>, String> {
    let data = Backend::int_into_data(tensor)
        .await
        .map_err(|e| format!("readback failed: {e:?}"))?;
    data.into_vec::<u32>()
        .map_err(|e| format!("expected u32 data: {e:?}"))
}

/// Fragment slots reserved per selected pyramid layer (the 2x2 footprint).
/// Re-exported so tests can reason about buffer sizes.
pub const CORNERS: u32 = CORNERS_PER_LAYER;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_log2_is_exact_at_powers_of_two_and_monotone() {
        for exponent in -20i32..=20 {
            let x = 2.0f32.powi(exponent);
            assert!(
                (fast_log2(x) - exponent as f32).abs() < 1e-4,
                "2^{exponent}: {}",
                fast_log2(x)
            );
        }
        let mut previous = f32::NEG_INFINITY;
        let mut x = 1e-6f32;
        while x < 1e6 {
            let value = fast_log2(x);
            assert!(value > previous, "not monotone at {x}");
            previous = value;
            x *= 1.031; // an irrational-ish step, so it lands mid-octave too
        }
    }

    #[test]
    fn the_packed_key_leaves_room_for_depth_at_every_plausible_pyramid_size() {
        // 1920x1080 with 8 levels and `ceil` halving.
        let total_1080p = 2_073_600 + 518_400 + 129_600 + 32_400 + 8_160 + 2_040 + 510 + 135;
        let bits = key_bits_for(total_1080p);
        assert_eq!(bits, 22, "1080p pyramid needs 22 key bits, got {bits}");
        assert!(32 - bits >= MIN_PACKED_DEPTH_BITS);
        // The sentinel must still fit in a u32 and sort past every real key.
        let shift = 32 - bits;
        let sentinel = (total_1080p as u64) << shift;
        let biggest_real = (((total_1080p - 1) as u64) << shift) | ((1u64 << shift) - 1);
        assert!(sentinel <= u64::from(u32::MAX), "sentinel overflows u32");
        assert!(biggest_real < sentinel);
    }

    #[test]
    fn a_4k_pyramid_still_has_depth_bits() {
        // 3840x2160, 8 levels: four times the pixels, one more key bit.
        let total_4k = 8_294_400 + 2_073_600 + 518_400 + 129_600 + 32_400 + 8_160 + 2_040 + 510;
        assert!(32 - key_bits_for(total_4k) >= MIN_PACKED_DEPTH_BITS);
    }
}
