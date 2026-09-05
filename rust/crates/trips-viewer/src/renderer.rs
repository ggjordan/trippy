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

use brush_pyramid::gpu::{
    burn_bridge, render_pyramid, render_pyramid_timed, StageTimings, WgpuDevice,
};
use brush_pyramid::params::{DepthRange, FeatureStore, LayerFloor, PyramidParams, SortMode};
use brush_pyramid::scene::{Camera, PointSet};
use brush_unet::camera::NeuralCamera;
use brush_unet::net::Unet;
use burn_wgpu::{CubeTensor, WgpuRuntime};

use crate::bundle::{Bounds, Bundle};

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
    points: PointSet,
    base_params: PyramidParams,
    background: Option<Vec<f32>>,
    bounds: Bounds,
    /// The float32 network.
    net: Unet,
    /// The same weights in f16. Both are built at load time — the file is
    /// 411 KiB, so keeping two copies costs nothing and makes the precision a
    /// *runtime* toggle rather than a restart.
    net_half: Unet,
    tone: NeuralCamera,
}

impl Renderer {
    /// Upload the bundle's weights and keep its points for per-frame upload.
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
        let net_half = Unet::load_with_precision(
            &bundle.weights,
            &burn_device,
            brush_unet::net::Precision::F16,
        )?;
        let tone = NeuralCamera::load(&bundle.weights, &burn_device)?;
        let bounds = bundle.bounds();
        let background = bundle.background().map(<[f32]>::to_vec);
        Ok(Self {
            device,
            points: bundle.points,
            base_params: bundle.manifest.params,
            background,
            bounds,
            net,
            net_half,
            tone,
        })
    }

    /// `N`, the number of points.
    #[must_use]
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// The decoder to run this frame, at the requested precision.
    fn net(&self, settings: &Settings) -> &Unet {
        if settings.half_net {
            &self.net_half
        } else {
            &self.net
        }
    }

    /// The scene's world-space bounding box.
    #[must_use]
    pub fn bounds(&self) -> Bounds {
        self.bounds
    }

    /// Render one frame.
    ///
    /// # Arguments
    /// - `camera`: this frame's camera, already at the render resolution.
    /// - `frame_index`: the tone mapper's per-image exposure/white-balance
    ///   slot. Ignored by `RawLevel0` and `Coverage`.
    /// - `mode`: which buffer to hand back.
    /// - `settings`: the performance levers.
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
    ) -> Result<RenderedFrame, String> {
        let params = settings.params(
            self.base_params,
            self.bounds.depth_span(camera, self.base_params.znear),
        );
        let background = self.background.as_deref();
        let clock = std::time::Instant::now();

        let (render, stages) = if settings.profile {
            let (r, t) = render_pyramid_timed(
                &self.device,
                &self.points,
                camera,
                &params,
                background,
            )
            .await?;
            (r, Some(t))
        } else {
            let r =
                render_pyramid(&self.device, &self.points, camera, &params, background).await?;
            (r, None)
        };

        let (width, height) = {
            let (h, w) = render.grid().shapes()[0];
            (w as u32, h as u32)
        };
        let channels = render.channels() as u32;

        let (buffer, out_channels) = match mode {
            ViewMode::RawLevel0 => {
                // Level 0 is rows 0..h*w of the flat (P, C) buffer, so its
                // bytes are already the blit's layout at offset 0. Nothing
                // to slice, nothing to copy.
                (render.feature_buffer().clone(), channels)
            }
            ViewMode::Coverage => (render.t_final_buffer().clone(), 1),
            ViewMode::Network => {
                let rgb = self
                    .tone
                    .forward(self.net(settings).forward(&render.layer_tensors())?, frame_index)?;
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
                let buffer = burn_bridge::resolve_to_cube_float(rgb);
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
                frame_ms: clock.elapsed().as_secs_f64() * 1e3,
                fragment_slots: render.num_fragment_slots(),
                width: width as usize,
                height: height as usize,
                stages,
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
    ) -> Result<(Vec<f32>, usize, usize, usize), String> {
        let params = settings.params(
            self.base_params,
            self.bounds.depth_span(camera, self.base_params.znear),
        );
        let render = render_pyramid(
            &self.device,
            &self.points,
            camera,
            &params,
            self.background.as_deref(),
        )
        .await?;
        let rgb = self
            .tone
            .forward(self.net(settings).forward(&render.layer_tensors())?, frame_index)?;
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
