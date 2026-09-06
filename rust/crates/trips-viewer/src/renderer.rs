//! One TRIPS frame: pyramid -> U-Net -> tone map, or a diagnostic view.
//!
//! Module: `trips_viewer::renderer`
//! Purpose: the whole per-frame pipeline, in one place, producing a *device*
//!     buffer the paint callback binds. This is `brush-unet`'s
//!     `examples/render_frame_full.rs` turned into something a viewer can call
//!     sixty times a second: the weights are loaded once, nothing is read back
//!     except the fragment count the rasteriser genuinely needs, and the
//!     result never touches host memory on its way to the screen.
//! Invariants:
//!     - The three view modes render the SAME pyramid; they differ only in
//!       which buffer is handed to the blit. So switching views cannot change
//!       the geometry being shown, which is the point of an honesty view.
//!       They are NOT the same cost, though: `RawLevel0` and `Coverage` stop
//!       before the network, which at 1080p is ~89% of the frame, so they run
//!       about ten times faster (`docs/LIMITATIONS.md`).
//!     - `RawLevel0` and `Coverage` bind the rasteriser's own buffers with
//!       **no copy**: level 0 occupies rows `0 .. h*w` of the flat, channel-
//!       last `(P, C)` allocation, so its data is already exactly the layout
//!       the blit shader wants, starting at offset 0.
//!     - Performance levers live in [`Settings`] and default to the exact
//!       pipeline; [`Settings::params`] is the only place they are applied.
//! Units: milliseconds for timings, pixels for sizes.
//! Related docs: `rust/README.md`; `docs/LIMITATIONS.md`;
//!     `docs/decisions/ADR-0006-viewer-integration.md`.

use brush_pyramid::gpu::burn_bridge;
use brush_pyramid::gpu::{
    render_pyramid_uploaded, render_pyramid_uploaded_timed, StageTimings, UploadedPoints,
    WgpuDevice,
};
use brush_pyramid::params::{DepthRange, FeatureStore, LayerFloor, PyramidParams, SortMode};
use brush_pyramid::scene::{Camera, PointSet};
use brush_unet::camera::NeuralCamera;
use brush_unet::net::Unet;
use burn::tensor::Tensor;
use burn_wgpu::{CubeTensor, WgpuRuntime};

use crate::blend::{Blend, BlendMode};
use crate::bundle::{Bounds, Bundle, SplatImage};

/// `cat([layer_i, block_i], dim=1)` for every level.
///
/// Split out so the cached path and the freshly-uploaded path build the inputs
/// through the same three lines.
fn concat_blocks(layers: &[Tensor<4>], blocks: &[Tensor<4>]) -> Vec<Tensor<4>> {
    layers
        .iter()
        .zip(blocks.iter())
        .map(|(layer, block)| Tensor::cat(vec![layer.clone(), block.clone()], 1))
        .collect()
}

/// Get the network's frame into a buffer the blit can bind, **without a
/// readback**, on native and in the browser alike.
///
/// `burn_bridge::resolve_to_cube_float` drains the tensor's fusion stream and
/// hands back the allocation the last kernel wrote to, so the frame never
/// leaves the GPU.
///
/// # Why this is no longer `cfg`-split
///
/// v0.5.0 shipped a wasm twin of this function that read the frame back with
/// `into_data_async` and re-uploaded it, because
/// `resolve_to_cube_float` was believed to end in CubeCL's `read_sync`, which
/// cannot block on `wasm32-unknown-unknown`. Reading the pinned revisions
/// settled it: it does not.
///
/// - `FusionClient::resolve_tensor_float` (`burn-fusion/src/client.rs:362`)
///   calls `DeviceHandle::submit_blocking`, then `drain_stream` +
///   `get_float_tensor`.
/// - `submit_blocking` is only a channel round trip when cubecl's
///   `multi_threading` cfg is on, and `cubecl-common/build.rs:11` defines that
///   as `all(feature = "std", not(target_family = "wasm"))`. On wasm the
///   handle is `ReentrantMutexDeviceHandle`, whose `submit_blocking`
///   (`cubecl-common/src/device/handle/reentrant.rs:51`) runs the closure
///   inline under a reentrant mutex — no thread parks, no future is polled.
/// - `drain_stream` only launches kernels; there is no `read_sync` anywhere in
///   `burn-fusion`, `burn-cubecl-fusion` or `burn-ir`.
///
/// The `read_sync` trap the browser really hit is CubeCL's autotune roofline
/// probe, three layers below this; see `docs/WEB_VIEWER.md` blocker 4 and
/// `brush_pyramid::gpu::disable_autotune_roofline_bounds`.
///
/// # Panics
/// Panics if `rgb` is not on a wgpu backend, which is a programming error at
/// the call site rather than a runtime condition.
fn resolve_network_output(rgb: burn::tensor::Tensor<4>) -> CubeTensor<WgpuRuntime> {
    burn_bridge::resolve_to_cube_float(rgb)
}

/// A monotonic clock that is a no-op on the web.
///
/// `std::time::Instant::now()` panics on `wasm32-unknown-unknown` ("time not
/// implemented on this platform"), and the browser front end measures frames
/// with `requestAnimationFrame` timestamps anyway, so on wasm this reports
/// 0 ms and [`FrameStats::frame_ms`] is simply not populated there. The native
/// viewer is unchanged. See `docs/WEB_VIEWER.md`.
#[derive(Debug, Clone, Copy)]
struct SubmitClock {
    #[cfg(not(target_family = "wasm"))]
    start: std::time::Instant,
}

impl SubmitClock {
    fn start() -> Self {
        Self {
            #[cfg(not(target_family = "wasm"))]
            start: std::time::Instant::now(),
        }
    }

    fn elapsed_ms(self) -> f64 {
        #[cfg(not(target_family = "wasm"))]
        {
            self.start.elapsed().as_secs_f64() * 1e3
        }
        #[cfg(target_family = "wasm")]
        {
            0.0
        }
    }
}

/// What the viewer draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewMode {
    /// The displayed frame: pyramid -> U-Net -> tone mapper.
    #[default]
    Network,
    /// The rasteriser's finest pyramid level, straight out of the blend, with
    /// no network in between. Photographed-ish evidence, before invention.
    RawLevel0,
    /// `1 - t_final` at level 0: which pixels the rasteriser actually covered.
    /// Everything dark here was invented by the U-Net.
    Coverage,
}

impl ViewMode {
    /// Cycle order for the `V` key.
    #[must_use]
    pub fn next(self) -> Self {
        match self {
            Self::Network => Self::RawLevel0,
            Self::RawLevel0 => Self::Coverage,
            Self::Coverage => Self::Network,
        }
    }

    /// Short label for the on-screen readout.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Network => "network",
            Self::RawLevel0 => "raw level-0",
            Self::Coverage => "coverage",
        }
    }

    /// The `mode` lane of the blit shader's uniform block.
    #[must_use]
    pub const fn shader_code(self) -> u32 {
        match self {
            Self::Network => 0,
            Self::RawLevel0 => 1,
            Self::Coverage => 2,
        }
    }
}

/// Which exposure the tone mapper applies to the frame.
///
/// The tone mapper's exposure is **per image**: it is the one term of the
/// whole camera model that differs between two views of the same scene (white
/// balance and vignette are frozen in every TRIPS config we run). A viewer
/// that renders arbitrary poses therefore has to answer a question a training
/// run never asks — *whose* exposure does a pose that is not a photograph
/// get? See `docs/LIMITATIONS.md` "Per-image exposure".
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ExposureMode {
    /// The reference view's own exposure while the camera is sitting exactly
    /// on it, the scene median once you have moved off it. The default,
    /// because both halves are the honest answer: on a capture pose the
    /// frame is comparable with that photograph, and off it there is no
    /// photograph to be comparable with, so the scene's own grade is better
    /// than whichever view you last pressed `N` past.
    #[default]
    Auto,
    /// Always the reference view's own exposure, moved or not.
    View,
    /// Always the scene median, moved or not — one grade for the whole scene.
    Median,
    /// A hand-set EV, applied as a gain of `2 ** -EV`.
    Manual(f32),
}

/// Range of the manual EV slider, in stops either side of 0.
///
/// Wider than any capture's own spread (kk-coherent's learned EVs span 2.3
/// stops) so it can also be used to inspect a frame that is nearly black or
/// nearly clipped, and narrow enough that the slider stays usable.
pub const MANUAL_EXPOSURE_LIMIT: f32 = 8.0;

impl ExposureMode {
    /// The EV to apply, or `None` for "image `frame`'s own learned value".
    ///
    /// Pure, so the policy is testable without a device.
    ///
    /// # Arguments
    /// - `pinned`: is the camera exactly on the reference view?
    /// - `median`: the scene's median EV, or `None` when the tone mapper has
    ///   no exposure table at all (in which case there is nothing to override
    ///   and every mode falls back to `None`).
    #[must_use]
    pub fn resolve(self, pinned: bool, median: Option<f32>) -> Option<f32> {
        match self {
            Self::Auto => {
                if pinned {
                    None
                } else {
                    median
                }
            }
            Self::View => None,
            Self::Median => median,
            Self::Manual(ev) => Some(ev),
        }
    }

    /// Short label for the on-screen readout.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::View => "view",
            Self::Median => "median",
            Self::Manual(_) => "manual",
        }
    }
}

/// The runtime-settable performance levers, all defaulting to exact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// Fraction of the window the pyramid is rasterised at; the blit
    /// upsamples. 1.0 renders at native size.
    pub render_scale: f32,
    /// Turn the frustum box cull OFF. Only exists so the perf table can say
    /// what the cull is worth; a viewer should never set it.
    pub no_cull: bool,
    /// Fragment cap: start `Trips`' emission at `layer_lower - 1`.
    pub cap_fragments: bool,
    /// One packed 32-bit sort key instead of two exact radix passes.
    pub packed_sort: bool,
    /// Store point features as f16.
    pub half_features: bool,
    /// Run the U-Net in f16. **The lever that actually matters at 1080p** —
    /// see `research/trips-metal.md`; every rasteriser-side lever measured
    /// within noise because the frame is network-bound, not sort-bound.
    pub half_net: bool,
    /// Collect per-stage timings. Costs a device sync per stage, so it is off
    /// unless the profiler panel is open.
    pub profile: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            render_scale: 1.0,
            no_cull: false,
            cap_fragments: false,
            packed_sort: false,
            half_features: false,
            half_net: false,
            profile: false,
        }
    }
}

impl Settings {
    /// True when nothing has been traded away and the frame is bit-comparable
    /// with `render_frame_full`'s.
    #[must_use]
    pub fn is_exact(&self) -> bool {
        self.render_scale >= 1.0
            && !self.no_cull
            && !self.cap_fragments
            && !self.packed_sort
            && !self.half_features
            && !self.half_net
    }

    /// Apply the levers to the bundle's base render parameters.
    ///
    /// # Arguments
    /// - `base`: the bundle manifest's `params`.
    /// - `depth_range`: the scene depth span the packed key quantises over.
    #[must_use]
    pub fn params(&self, base: PyramidParams, depth_range: (f32, f32)) -> PyramidParams {
        PyramidParams {
            frustum_cull: !self.no_cull,
            layer_floor: if self.cap_fragments {
                LayerFloor::NearLower
            } else {
                LayerFloor::Zero
            },
            sort: if self.packed_sort {
                SortMode::PackedKey
            } else {
                SortMode::DepthThenKey
            },
            feature_store: if self.half_features {
                FeatureStore::F16
            } else {
                FeatureStore::F32
            },
            depth_range: DepthRange {
                lo: depth_range.0,
                hi: depth_range.1,
            },
            ..base
        }
    }
}

/// What one frame cost, and what it contained.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameStats {
    /// Whole frame, wall clock, from the first kernel launch to the last
    /// result being ready.
    pub frame_ms: f64,
    /// Fragment slots the sort moved.
    pub fragment_slots: u32,
    /// Rendered width, pixels (window width times the render scale).
    pub width: usize,
    /// Rendered height, pixels.
    pub height: usize,
    /// Per-stage profile, only populated when [`Settings::profile`] is set.
    pub stages: Option<StageTimings>,
    /// What the Blend panel actually drew this frame, and why it may not be
    /// what was asked for. `None` when no blend was requested.
    ///
    /// The viewer must never silently substitute one image for another: if the
    /// requested mode needed a splat this view has no precomputed render for,
    /// this says so and the frame is the TRIPS one.
    pub blend: Option<BlendStatus>,
}

/// The outcome of one frame's blend request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendStatus {
    /// What was asked for.
    pub requested: BlendMode,
    /// What was drawn. Equal to `requested` unless something was missing.
    pub applied: BlendMode,
    /// Whether a precomputed splat render existed for this view.
    pub splat_available: bool,
    /// Whether the network carries the gate head.
    pub gate_available: bool,
}

impl BlendStatus {
    /// True when the frame is not what the panel asked for.
    #[must_use]
    pub const fn degraded(&self) -> bool {
        !matches!(
            (self.requested, self.applied),
            (BlendMode::Trips, BlendMode::Trips)
        ) && !self.requested_equals_applied()
    }

    const fn requested_equals_applied(&self) -> bool {
        matches!(
            (self.requested, self.applied),
            (BlendMode::Trips, BlendMode::Trips)
                | (BlendMode::Splat, BlendMode::Splat)
                | (BlendMode::Gated, BlendMode::Gated)
                | (BlendMode::Mix, BlendMode::Mix)
                | (BlendMode::Split, BlendMode::Split)
        )
    }

    /// Why the frame is not what was asked for, or `None` when it is.
    #[must_use]
    pub fn note(&self) -> Option<String> {
        if self.requested_equals_applied() {
            return None;
        }
        if self.requested.needs_gate() && !self.gate_available {
            return Some(
                "this bundle's weights have no gate head (trained without hybrid.gate) \
                 -- showing TRIPS"
                    .to_owned(),
            );
        }
        Some(
            "no splat for THIS pose: the bundle carries renders of the capture views only, \
             and reusing one from a view you have flown off would be a picture taken from \
             somewhere else. Press R or N/P to sit on a capture view -- showing TRIPS"
                .to_owned(),
        )
    }
}

/// A frame that is finished and still on the device.
pub struct RenderedFrame {
    /// Flat, channel-last `height * width * channels` f32 buffer.
    pub buffer: CubeTensor<WgpuRuntime>,
    /// Values per pixel in `buffer`.
    pub channels: u32,
    /// Buffer width, pixels.
    pub width: u32,
    /// Buffer height, pixels.
    pub height: u32,
    /// Which view mode produced it.
    pub mode: ViewMode,
    /// Cost and contents.
    pub stats: FrameStats,
}

/// Everything needed to render, loaded once.
pub struct Renderer {
    device: WgpuDevice,
    /// The host copy, kept **only** so the f16-features lever can re-upload.
    /// Nothing in a frame reads it.
    points: PointSet,
    /// The points as device buffers, uploaded once per bundle.
    ///
    /// This is the fix for the viewer's single biggest fixed cost:
    /// `render_pyramid` used to push all ~80 MB of the horse bundle's four
    /// point arrays across the bus **every frame**, worth a flat 12.2 ms —
    /// which is 55 % of a `raw level-0` frame at 1080p and took the shipped
    /// `--half-net --scale 0.75` view from 21.7 to 29.5 fps. See
    /// `research/trips-metal.md`.
    ///
    /// A `RefCell` because the f16-features lever is a runtime
    /// toggle and the feature buffer's element type is fixed at upload time,
    /// so flipping `half_features` has to re-upload. The borrow is never held
    /// across an `await`: [`UploadedPoints`] is cloned out first, which costs
    /// a handle clone, not the bytes.
    uploaded: std::cell::RefCell<UploadedPoints>,
    base_params: PyramidParams,
    background: Option<Vec<f32>>,
    /// The **point cloud's** world-space box, used only for
    /// [`Bounds::depth_span`]. Deliberately not exposed: it is not a scene
    /// scale, because a TRIPS export's environment sphere makes it thousands
    /// of units across. Anything the camera controller needs comes from
    /// [`crate::bundle::SceneScale`], which measures the capture cameras.
    bounds: Bounds,
    /// The float32 network.
    net: Unet,
    /// The same weights in f16. Built at load time alongside the f32 copy —
    /// the file is 411 KiB, so keeping two copies costs nothing and makes the
    /// precision a *runtime* toggle rather than a restart.
    ///
    /// `None` when this device cannot hold f16 tensors at all. That never
    /// happens on Metal, but a WebGPU adapter without the `shader-f16`
    /// feature is a real possibility (`docs/WEB_VIEWER.md`), and refusing to
    /// open the scene would be a worse answer than rendering it in f32 and
    /// saying so — see [`Self::half_net_error`].
    net_half: Option<Unet>,
    /// Why [`Self::net_half`] is `None`, verbatim from Burn.
    half_net_error: Option<String>,
    tone: NeuralCamera,
    /// Which per-image exposure the tone mapper applies, and whether the
    /// camera is currently sitting on its reference view. Set once per frame
    /// by the UI (see [`Self::set_exposure`]); resolved to an EV, or to
    /// "use the frame's own", by [`ExposureMode::resolve`].
    exposure: ExposureMode,
    /// Whether the camera is pinned to its reference view — the one input
    /// [`ExposureMode::Auto`] needs that the renderer cannot see for itself.
    pinned: bool,
    /// Precomputed Gaussian-splat renders by dataset image index, from the
    /// bundle's `splat.npz`. Empty on any bundle without a `blend` block.
    splats: std::collections::HashMap<usize, SplatImage>,
    /// The most recently uploaded splat, as `(dataset index, height, width)`
    /// and its device tensor.
    ///
    /// One entry, not a map: a frame draws one view, and the camera stays on a
    /// view for many frames, so a single slot turns a per-frame host resample +
    /// upload into a per-*view-change* one. A `RefCell` for exactly the reason
    /// [`Self::uploaded`] is one — the borrow is never held across an `await`.
    splat_cache: std::cell::RefCell<Option<((usize, usize, usize), Tensor<4>)>>,
    /// The bundle's `blend` block, or `None` on a non-hybrid bundle.
    blend_manifest: Option<crate::bundle::BlendManifest>,
    /// The Gaussian block, pooled and uploaded per pyramid level, cached on
    /// `(dataset index, level-0 height, level-0 width)`.
    ///
    /// Separate from [`Self::splat_cache`] because the two are wanted at
    /// different sizes: this one is the network's *input* at every level, that
    /// one is the blend's *operand* at the output size.
    block_cache: std::cell::RefCell<Option<((usize, usize, usize), Vec<Tensor<4>>)>>,
}

impl Renderer {
    /// Upload the bundle's weights **and its points**, once.
    ///
    /// # Arguments
    /// - `bundle`: the loaded scene.
    /// - `device`: the wgpu device Burn and the UI share.
    ///
    /// # Errors
    /// Returns `Err` if the weights do not match the schema this build reads.
    pub fn new(bundle: Bundle, device: WgpuDevice) -> Result<Self, String> {
        let burn_device = device.clone().into();
        let net = Unet::load(&bundle.weights, &burn_device)?;
        // An f16 load failure degrades to f32 rather than refusing the scene;
        // see the `net_half` field.
        let (net_half, half_net_error) = match Unet::load_with_precision(
            &bundle.weights,
            &burn_device,
            brush_unet::net::Precision::F16,
        ) {
            Ok(half) => (Some(half), None),
            Err(e) => (None, Some(e)),
        };
        let tone = NeuralCamera::load(&bundle.weights, &burn_device)?;
        let splats = bundle.splats.clone();
        let blend_manifest = bundle.manifest.blend.clone();
        // A design-A network takes `C + G` input channels; the bundle must carry a
        // block wide enough to fill them or nothing can be rendered. Refusing here,
        // with both numbers, beats failing inside the first `forward` with a shape
        // mismatch a user cannot act on.
        if let Some(blend) = blend_manifest.as_ref() {
            let expected = bundle.manifest.num_channels + blend.block_channels();
            if expected != net.config().in_channels {
                return Err(format!(
                    "bundle.json's blend block describes {} feature + {} Gaussian channels \
                     ({expected} total) but the weights want in_channels = {}",
                    bundle.manifest.num_channels,
                    blend.block_channels(),
                    net.config().in_channels
                ));
            }
        } else if bundle.manifest.num_channels != net.config().in_channels {
            return Err(format!(
                "points.npz has C = {} but the weights want in_channels = {}; a hybrid \
                 (design A) checkpoint needs a bundle with a `blend` block -- re-export it",
                bundle.manifest.num_channels,
                net.config().in_channels
            ));
        }
        let bounds = bundle.bounds();
        let background = bundle.background().map(<[f32]>::to_vec);
        // The one upload. `FeatureStore::F32` is `Settings::default()`'s
        // precision; turning `--fp16` on re-uploads once, in `render`.
        let uploaded = UploadedPoints::new(
            &bundle.points,
            brush_pyramid::params::FeatureStore::F32,
            &device,
        )?;
        Ok(Self {
            device,
            points: bundle.points,
            uploaded: std::cell::RefCell::new(uploaded),
            base_params: bundle.manifest.params,
            background,
            bounds,
            net,
            net_half,
            half_net_error,
            tone,
            exposure: ExposureMode::default(),
            pinned: true,
            splats,
            splat_cache: std::cell::RefCell::new(None),
            blend_manifest,
            block_cache: std::cell::RefCell::new(None),
        })
    }

    /// True when this bundle's weights carry the blend gate's output channel.
    #[must_use]
    pub fn has_gate(&self) -> bool {
        self.net.config().has_gate()
    }

    /// Dataset image indices that have a precomputed splat render, sorted.
    #[must_use]
    pub fn splat_views(&self) -> Vec<usize> {
        let mut out: Vec<usize> = self.splats.keys().copied().collect();
        out.sort_unstable();
        out
    }

    /// True when `frame_index` has a precomputed splat render **that is valid
    /// for the camera right now** -- i.e. it exists AND the camera is sitting on
    /// that view.
    ///
    /// The `pinned` half is not a detail. A stored block is a render of the
    /// Gaussians *from that view's camera*; once you have flown off it, feeding
    /// it to the network (or blending it into the frame) would be showing a
    /// picture taken from somewhere else and calling it this pose. Training
    /// refuses the same substitution for the same reason -- see
    /// `trippy.hybrid.gsrender_live.gaussian_provider_for`, which renders the PLY
    /// live at the pose rather than reusing the anchor image's render. The live
    /// splat path is what will make this true everywhere; until then, off a
    /// capture view the answer is honestly "no".
    #[must_use]
    pub fn has_splat(&self, frame_index: usize) -> bool {
        self.pinned && self.splats.contains_key(&frame_index)
    }

    /// The stored block for `frame_index`, or `None` when it must not be used.
    fn usable_splat(&self, frame_index: usize) -> Option<&SplatImage> {
        if !self.pinned {
            return None;
        }
        self.splats.get(&frame_index)
    }

    /// The splat render for `frame_index`, resampled to `height x width` and
    /// resident on the device, or `None` when this view has none.
    ///
    /// Cached in [`Self::splat_cache`] on `(frame, height, width)`, so panning
    /// the split-screen slider or holding a view costs one upload, not sixty a
    /// second.
    fn splat_tensor(&self, frame_index: usize, height: usize, width: usize) -> Option<Tensor<4>> {
        let key = (
            if self.pinned { frame_index } else { usize::MAX },
            height,
            width,
        );
        if let Some((cached_key, tensor)) = self.splat_cache.borrow().as_ref() {
            if *cached_key == key {
                return Some(tensor.clone());
            }
        }
        let image = self.usable_splat(frame_index)?;
        let host = brush_unet::HostTensor {
            shape: vec![1, 3, height, width],
            data: image.resampled(height, width),
        };
        let tensor = brush_unet::net::upload::<4>(&host, &self.device.clone().into()).ok()?;
        *self.splat_cache.borrow_mut() = Some((key, tensor.clone()));
        Some(tensor)
    }

    /// The pyramid levels the U-Net actually consumes.
    ///
    /// On a plain bundle this is `render.layer_tensors()` unchanged. On a hybrid
    /// (design A) bundle the Gaussian block is concatenated onto **every** level,
    /// which is what `trippy.hybrid.gaussian_input.GaussianInputs.attach` does in
    /// training -- pooled to each level's own size in `all_levels` mode, and real
    /// on level 0 with zeros above it in `concat_level0` mode. Getting that split
    /// wrong would feed the network something it was never trained on.
    ///
    /// A view with no stored block gets zeros in those channels: design A's own
    /// "no Gaussian information here" state, which `dropout_gaussian_p` trained
    /// the network to render through. It is never another view's block.
    fn network_inputs(
        &self,
        render: &brush_pyramid::gpu::PyramidRender,
        frame_index: usize,
    ) -> Result<Vec<Tensor<4>>, String> {
        let layers = render.layer_tensors();
        let Some(blend) = self.blend_manifest.as_ref() else {
            return Ok(layers);
        };
        let width = blend.block_channels();
        if width == 0 {
            return Ok(layers);
        }
        let [_, _, level0_h, level0_w] = layers[0].dims();
        // `pinned` is part of the key: flying off a view must re-upload zeros, not
        // keep serving the block that was cached while the camera sat on it.
        let key = (
            if self.pinned { frame_index } else { usize::MAX },
            level0_h,
            level0_w,
        );
        if let Some((cached_key, blocks)) = self.block_cache.borrow().as_ref() {
            if *cached_key == key && blocks.len() == layers.len() {
                return Ok(concat_blocks(&layers, blocks));
            }
        }

        let device = self.device.clone().into();
        // `usable_splat`, not `splats.get`: off a capture view there is no block for
        // this pose, and the network gets the honest zeros it was trained to render
        // through rather than another camera's picture. See `has_splat`.
        let image = self.usable_splat(frame_index);
        let mut blocks = Vec::with_capacity(layers.len());
        for (level, layer) in layers.iter().enumerate() {
            let [_, _, h, w] = layer.dims();
            let zeros = image.is_none() || (level > 0 && !blend.all_levels());
            let data = if zeros {
                vec![0.0f32; width * h * w]
            } else {
                image
                    .expect("checked above")
                    .block(&blend.channels, blend.mask_by_alpha, h, w)
            };
            let host = brush_unet::HostTensor {
                shape: vec![1, width, h, w],
                data,
            };
            blocks.push(brush_unet::net::upload::<4>(&host, &device)?);
        }
        let inputs = concat_blocks(&layers, &blocks);
        *self.block_cache.borrow_mut() = Some((key, blocks));
        Ok(inputs)
    }

    /// Compose the displayed image from the TRIPS frame, the splat and the gate.
    ///
    /// The single place the Blend panel's arithmetic happens, for both the live
    /// window and `--screenshot` -- so the picture a screenshot diff measures is
    /// the picture on screen.
    ///
    /// Returns `(image, status)`; `status` records what was actually drawn (see
    /// [`BlendStatus`]), which is never silently different from what was asked.
    fn compose(
        &self,
        trips: Tensor<4>,
        gate: Option<Tensor<4>>,
        frame_index: usize,
        blend: Blend,
    ) -> Result<(Tensor<4>, BlendStatus), String> {
        let [_, _, height, width] = trips.dims();
        let gate_available = gate.is_some();
        let splat = if blend.mode.needs_splat() {
            self.splat_tensor(frame_index, height, width)
        } else {
            None
        };
        let mut status = BlendStatus {
            requested: blend.mode,
            applied: blend.mode,
            splat_available: splat.is_some(),
            gate_available,
        };

        // Anything the frame genuinely cannot draw falls back to TRIPS and says so.
        let unavailable = (blend.mode.needs_splat() && splat.is_none())
            || (blend.mode.needs_gate() && !gate_available);
        if unavailable {
            status.applied = BlendMode::Trips;
            return Ok((trips, status));
        }

        let image = match blend.mode {
            BlendMode::Trips => trips,
            BlendMode::Splat => splat.expect("checked above"),
            BlendMode::Gated => brush_unet::blend_gate(
                splat.expect("checked above"),
                trips,
                gate.expect("checked above"),
                blend.clamped_gate_scale(),
            )?,
            BlendMode::Mix => {
                // A uniform weight, so no gate tensor is needed at all: this is
                // the mode that works on a bundle trained WITHOUT the gate.
                let g = blend.uniform_gate();
                splat
                    .expect("checked above")
                    .mul_scalar(g)
                    .add(trips.mul_scalar(1.0 - g))
            }
            BlendMode::Split => {
                // Slice and concatenate rather than mask-and-blend: it is exact
                // (each side is the untouched image, not a lerp that happens to
                // land on 0 or 1) and costs no per-pixel upload.
                let column = blend.split_column(width);
                let splat = splat.expect("checked above");
                if column == 0 {
                    trips
                } else if column >= width {
                    splat
                } else {
                    Tensor::cat(
                        vec![
                            splat.slice_dim(3, 0..column),
                            trips.slice_dim(3, column..width),
                        ],
                        3,
                    )
                }
            }
        };
        Ok((image, status))
    }

    /// Choose the exposure policy for the frames that follow.
    ///
    /// # Arguments
    /// - `mode`: the policy.
    /// - `pinned`: whether the camera is sitting exactly on its reference
    ///   view. Only [`ExposureMode::Auto`] reads it.
    pub fn set_exposure(&mut self, mode: ExposureMode, pinned: bool) {
        self.exposure = mode;
        self.pinned = pinned;
    }

    /// The current exposure policy.
    #[must_use]
    pub fn exposure(&self) -> ExposureMode {
        self.exposure
    }

    /// The scene's median learned EV, or `None` when there is no exposure
    /// table. What the HUD shows next to the "median" choice.
    #[must_use]
    pub fn median_exposure(&self) -> Option<f32> {
        self.tone.median_exposure()
    }

    /// The learned EV of one image, or `None` when there is no exposure table.
    #[must_use]
    pub fn view_exposure(&self, frame: usize) -> Option<f32> {
        self.tone.exposure_of(frame)
    }

    /// The EV this frame will actually be tone mapped with, or `None` for
    /// "image `frame`'s own learned value".
    #[must_use]
    fn exposure_override(&self) -> Option<f32> {
        self.exposure.resolve(self.pinned, self.tone.median_exposure())
    }

    /// `N`, the number of points.
    #[must_use]
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// The decoder to run this frame, at the requested precision.
    ///
    /// Falls back to f32 when the f16 copy could not be built at all, so
    /// `half_net` is a request, not a promise. [`Self::half_net_error`] is
    /// what makes the difference visible instead of silent.
    fn net(&self, settings: &Settings) -> &Unet {
        match (settings.half_net, self.net_half.as_ref()) {
            (true, Some(half)) => half,
            _ => &self.net,
        }
    }

    /// The device-resident points, at the feature precision `params` asks
    /// for, re-uploading only if the `--fp16` lever just changed.
    ///
    /// Returns a clone so no `RefCell` borrow is alive across the `await` in
    /// [`Self::render`]; the clone is four `CubeTensor` handles.
    ///
    /// # Errors
    /// Returns `Err` if a re-upload fails.
    fn resident_points(&self, params: &PyramidParams) -> Result<UploadedPoints, String> {
        {
            let current = self.uploaded.borrow();
            if current.feature_store() == params.feature_store {
                return Ok(current.clone());
            }
        }
        let fresh = UploadedPoints::new(&self.points, params.feature_store, &self.device)?;
        *self.uploaded.borrow_mut() = fresh.clone();
        Ok(fresh)
    }

    /// Bytes of device memory the point set occupies — the per-frame upload
    /// this viewer no longer pays for.
    #[must_use]
    pub fn resident_point_bytes(&self) -> usize {
        self.uploaded.borrow().device_bytes()
    }

    /// Why the f16 network is unavailable on this device, if it is.
    ///
    /// `None` on every device that can hold f16 tensors, which is every Metal
    /// GPU this project has run on. A WebGPU adapter that reports no
    /// `shader-f16` support is the case this exists for.
    #[must_use]
    pub fn half_net_error(&self) -> Option<&str> {
        self.half_net_error.as_deref()
    }

    /// Render one frame.
    ///
    /// # Arguments
    /// - `camera`: this frame's camera, already at the render resolution.
    /// - `frame_index`: the tone mapper's per-image exposure/white-balance
    ///   slot. Ignored by `RawLevel0` and `Coverage`.
    /// - `mode`: which buffer to hand back.
    /// - `settings`: the performance levers.
    /// - `blend`: the Blend panel's state. Only read by [`ViewMode::Network`]
    ///   (the honesty views show the rasteriser's own buffers and must not be
    ///   mixed with anything). [`crate::blend::BlendMode::Trips`], the default,
    ///   is a hard no-op.
    ///
    /// # Errors
    /// Returns `Err` on any rasteriser or network failure; the caller shows
    /// the message rather than panicking mid-frame.
    pub async fn render(
        &self,
        camera: &Camera,
        frame_index: usize,
        mode: ViewMode,
        settings: &Settings,
        blend: Blend,
    ) -> Result<RenderedFrame, String> {
        let params = settings.params(
            self.base_params,
            self.bounds.depth_span(camera, self.base_params.znear),
        );
        let background = self.background.as_deref();
        let points = self.resident_points(&params)?;
        let clock = SubmitClock::start();

        let (render, stages) = if settings.profile {
            let (r, t) =
                render_pyramid_uploaded_timed(&points, camera, &params, background).await?;
            (r, Some(t))
        } else {
            let r = render_pyramid_uploaded(&points, camera, &params, background).await?;
            (r, None)
        };

        let (width, height) = {
            let (h, w) = render.grid().shapes()[0];
            (w as u32, h as u32)
        };
        let channels = render.channels() as u32;

        let mut blend_status = None;
        let (buffer, out_channels) = match mode {
            ViewMode::RawLevel0 => {
                // Level 0 is rows 0..h*w of the flat (P, C) buffer, so its
                // bytes are already the blit's layout at offset 0. Nothing
                // to slice, nothing to copy.
                (render.feature_buffer().clone(), channels)
            }
            ViewMode::Coverage => (render.t_final_buffer().clone(), 1),
            ViewMode::Network => {
                // The network may emit a fourth channel (the blend gate). Split it
                // off FIRST: the tone mapper is rgb-only, and the gate must not be
                // pushed through an exposure curve.
                let net = self.net(settings);
                let (colour, gate) = net.split_gate(net.forward(&self.network_inputs(&render, frame_index)?)?);
                let trips = self.tone.forward_with_exposure(
                    colour,
                    frame_index,
                    self.exposure_override(),
                )?;
                let (rgb, status) = self.compose(trips, gate, frame_index, blend)?;
                blend_status = Some(status);
                let [_, out_c, h, w] = rgb.dims();
                if h != height as usize || w != width as usize {
                    return Err(format!(
                        "network returned {h}x{w}, expected {height}x{width}"
                    ));
                }
                if out_c < 3 {
                    return Err(format!("network returned {out_c} channels, need >= 3"));
                }
                // Handed to the blit **as it is**, in planar NCHW; `blit.wgsl`
                // indexes `MODE_NETWORK` as `c * H * W + y * W + x`.
                //
                // The obvious alternative -- `permute([0, 2, 3, 1])` then
                // `reshape` into channel-last -- was tried and rejected: a
                // `CubeTensor` carries its own strides, so whether the resolved
                // buffer is really re-laid-out or merely re-described is a
                // property of the backend, and getting it wrong shows up as a
                // scrambled window that no test and no agent may look at. Below
                // the layout is asserted instead of assumed.
                let buffer = resolve_network_output(rgb);
                if !buffer.is_contiguous() {
                    return Err(
                        "the network's output buffer is not contiguous; the blit                          shader's planar indexing would be wrong"
                            .to_owned(),
                    );
                }
                (buffer, out_c as u32)
            }
        };

        // NOTE: `frame_ms` below is measured without a trailing device sync.
        // `render_pyramid` already synchronises once, for the fragment count,
        // so the number covers everything up to the *launch* of the blend and
        // the network -- it is a submission time, not a completion time. The
        // honest completion time is the frame interval the UI reports, which
        // is what the on-screen readout shows; this field is only used to
        // separate "the render errored" from "the render was slow".
        Ok(RenderedFrame {
            buffer,
            channels: out_channels,
            width,
            height,
            mode,
            stats: FrameStats {
                frame_ms: clock.elapsed_ms(),
                fragment_slots: render.num_fragment_slots(),
                width: width as usize,
                height: height as usize,
                stages,
                blend: blend_status,
            },
        })
    }

    /// Render one frame and read the displayed RGB back to host memory.
    ///
    /// Used by `--screenshot`, which is the viewer's own correctness check:
    /// the PNG it writes is compared with `render_frame_full`'s.
    ///
    /// # Arguments
    /// - `camera`, `frame_index`, `settings`: as [`Self::render`].
    ///
    /// # Returns
    /// `(rgb, channels, height, width)` with `rgb` in **planar CHW** order,
    /// which is what `brush_pyramid::png::feature_to_rgb8` expects.
    ///
    /// # Errors
    /// Returns `Err` on a render failure or a failed readback.
    pub async fn render_to_host(
        &self,
        camera: &Camera,
        frame_index: usize,
        settings: &Settings,
        blend: Blend,
    ) -> Result<(Vec<f32>, usize, usize, usize), String> {
        let params = settings.params(
            self.base_params,
            self.bounds.depth_span(camera, self.base_params.znear),
        );
        let points = self.resident_points(&params)?;
        let render =
            render_pyramid_uploaded(&points, camera, &params, self.background.as_deref()).await?;
        // Exactly `render`'s Network branch, so `--screenshot` measures the
        // picture the window draws -- including the blend.
        let net = self.net(settings);
        let (colour, gate) = net.split_gate(net.forward(&self.network_inputs(&render, frame_index)?)?);
        let trips =
            self.tone
                .forward_with_exposure(colour, frame_index, self.exposure_override())?;
        let (rgb, _status) = self.compose(trips, gate, frame_index, blend)?;
        let [_, channels, height, width] = rgb.dims();
        let data = rgb
            .into_data_async()
            .await
            .map_err(|e| format!("readback: {e:?}"))?
            .into_vec::<f32>()
            .map_err(|e| format!("expected f32: {e:?}"))?;
        Ok((data, channels, height, width))
    }
}

#[cfg(test)]
mod exposure_tests {
    use super::*;

    const MEDIAN: Option<f32> = Some(0.25);

    #[test]
    fn auto_uses_the_view_while_pinned_and_the_median_once_moved() {
        // Pinned: `None` means "image `frame`'s own learned EV", which is what
        // makes a frame taken from a capture pose comparable with that photo.
        assert_eq!(ExposureMode::Auto.resolve(true, MEDIAN), None);
        // Moved off: the scene's grade, not whichever view was last touched.
        assert_eq!(ExposureMode::Auto.resolve(false, MEDIAN), MEDIAN);
    }

    #[test]
    fn the_explicit_modes_ignore_whether_the_camera_moved() {
        for pinned in [true, false] {
            assert_eq!(ExposureMode::View.resolve(pinned, MEDIAN), None);
            assert_eq!(ExposureMode::Median.resolve(pinned, MEDIAN), MEDIAN);
            assert_eq!(
                ExposureMode::Manual(-1.5).resolve(pinned, MEDIAN),
                Some(-1.5)
            );
        }
    }

    #[test]
    fn a_scene_with_no_exposure_table_falls_back_to_the_frames_own() {
        // `Manual` is the exception: an EV the user typed is still an EV, and
        // the tone mapper simply has nothing to apply it to.
        assert_eq!(ExposureMode::Auto.resolve(false, None), None);
        assert_eq!(ExposureMode::Median.resolve(false, None), None);
        assert_eq!(ExposureMode::Manual(2.0).resolve(false, None), Some(2.0));
    }

    #[test]
    fn every_mode_has_a_label() {
        assert_eq!(ExposureMode::Auto.label(), "auto");
        assert_eq!(ExposureMode::View.label(), "view");
        assert_eq!(ExposureMode::Median.label(), "median");
        assert_eq!(ExposureMode::Manual(0.0).label(), "manual");
    }

    #[test]
    fn the_manual_slider_spans_more_than_any_capture_needs() {
        // kk-coherent's learned EVs span 2.3 stops after the exposure fix; the
        // slider has to cover that and leave room to inspect a clipped frame.
        assert!(MANUAL_EXPOSURE_LIMIT >= 4.0);
    }
}
