//! Acquiring a WebGPU device from a `<canvas>`, and sharing it with Burn.
//!
//! Module: `trips_web::gpu`
//! Purpose: the browser half of what `eframe` does for the native viewer —
//!     create a `wgpu::Instance`, get an adapter, open a device, configure the
//!     canvas surface, and hand the very same device to Burn so the
//!     rasteriser's output buffer can be bound into a render pass with no
//!     copy.
//! Invariants:
//!     - **WebGPU only.** `Backends::BROWSER_WEBGPU`, never `GL`. A WebGL2
//!       fallback would be dishonest here: every stage of
//!       `brush_pyramid::gpu` is a compute shader and WebGL2 has none, so the
//!       fallback could not render the scene, only an empty canvas.
//!     - Burn is initialised on **this** device (`burn_wgpu::init_device`),
//!       exactly as `trips-viewer`'s `main.rs` does with eframe's. Binding a
//!       buffer from one device into another device's pipeline is undefined
//!       and wgpu refuses it.
//!     - The device is requested with `adapter.features()` minus
//!       [`WEB_UNSUPPORTED`] and with `adapter.limits()` — the native viewer's
//!       `egui_options()` descriptor plus one web-specific mask. The limits
//!       matter here more than on Metal: the horse's pyramid needs a storage
//!       binding far above WebGPU's 128 MiB *default*
//!       `maxStorageBufferBindingSize`, and `adapter.limits()` is what raises
//!       it to whatever the browser actually supports.
//!     - The surface format is a **non-sRGB** `*_unorm`, so the values
//!       `blit.wgsl` writes reach the canvas unchanged and a `toBlob()`
//!       screenshot is directly comparable with the native `--screenshot`
//!       PNG (which is `png::feature_to_rgb8`, i.e. `round(clamp(v) * 255)`).
//!       Picking an `*_unorm_srgb` format instead would silently apply a
//!       transfer function to every pixel and turn the PSNR check into
//!       nonsense.
//! Related docs: `docs/WEB_VIEWER.md`;
//!     `docs/decisions/ADR-0006-viewer-integration.md`.

use burn_wgpu::WgpuDevice;
use wgpu::{Adapter, Device, ExperimentalFeatures, Features, Instance, Queue, Surface};

/// A WebGPU device, its canvas surface, and Burn's handle on the same device.
pub struct Gpu {
    /// The wgpu device every pipeline and every Burn tensor lives on.
    pub device: Device,
    /// Its queue.
    pub queue: Queue,
    /// The canvas surface frames are presented to.
    pub surface: Surface<'static>,
    /// The surface's current configuration; `width`/`height` track the canvas.
    pub config: wgpu::SurfaceConfiguration,
    /// Burn's handle on `device`.
    pub burn: WgpuDevice,
    /// What the adapter says it is, for the status readout.
    pub adapter_info: wgpu::AdapterInfo,
    /// True when the adapter advertises `SHADER_F16`, i.e. when `--half-net`'s
    /// web equivalent can actually run. Reported rather than assumed.
    pub has_f16: bool,
    /// True when the adapter advertises `SUBGROUP`. `brush-sort`'s radix
    /// passes cannot run without it on this stack (see [`WEB_UNSUPPORTED`]),
    /// so it is reported, not assumed.
    pub has_subgroups: bool,
}

/// Adapter features this viewer refuses even when the browser offers them.
///
/// - `MAPPABLE_PRIMARY_BUFFERS`: excluded on every platform, exactly as the
///   native viewer's device descriptor does — CubeCL's kernels will not run on
///   a device that has it.
/// `SUBGROUP` is deliberately **kept**, even though it is the source of the
/// v0.5.0 web blocker. `brush-sort`'s radix kernels call `plane_sum` /
/// `plane_inclusive_sum` unconditionally in their `#[cube]` source, and
/// CubeCL's WGSL backend translates those to `subgroupAdd` /
/// `subgroupInclusiveAdd` whether or not the device advertises the feature —
/// masking it off was tried and changes nothing except that the shader then
/// asks for a builtin the device has not even enabled. What the generated
/// WGSL is missing is the `enable subgroups;` directive; `web/trips.js`'s
/// `installWebGpuWorkaround` adds it, which is why the feature must stay
/// requested here. See `docs/WEB_VIEWER.md` for the whole diagnosis.
const WEB_UNSUPPORTED: Features = Features::MAPPABLE_PRIMARY_BUFFERS;

/// wgpu runtime options for Burn.
///
/// Copied from `trips-viewer`'s `main.rs::burn_options`, which took them from
/// the Brush fork's `brush_process::burn_options`, so the browser's memory
/// behaviour matches the one the kernels were tuned against.
fn burn_options() -> burn_wgpu::RuntimeOptions {
    burn_wgpu::RuntimeOptions {
        tasks_max: 64,
        memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
    }
}

impl Gpu {
    /// Open a WebGPU device on `canvas`.
    ///
    /// # Arguments
    /// - `canvas`: the `<canvas>` element to present to. Its `width` and
    ///   `height` attributes (not its CSS size) are the render target size.
    ///
    /// # Errors
    /// Returns `Err` with a message suitable for showing to a person if the
    /// browser has no `navigator.gpu`, if no adapter is available, or if the
    /// device request is refused.
    pub async fn create(canvas: web_sys::HtmlCanvasElement) -> Result<Self, String> {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            // WebGPU only -- see the module invariants.
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| format!("could not create a WebGPU surface on the canvas: {e}"))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                // Leave `apply_limit_buckets` at its default (off): bucketed
                // limits are an anti-fingerprinting measure for browsers
                // hosting untrusted content, and they would cap
                // `maxStorageBufferBindingSize` well below what the horse's
                // pyramid needs.
                ..Default::default()
            })
            .await
            .map_err(|e| {
                format!(
                    "no WebGPU adapter: {e}. This viewer needs WebGPU; \
                     check that navigator.gpu exists in this browser."
                )
            })?;

        let has_f16 = adapter.features().contains(Features::SHADER_F16);
        let has_subgroups = adapter.features().contains(Features::SUBGROUP);
        let adapter_info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("trips-web"),
                required_features: adapter.features().difference(WEB_UNSUPPORTED),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
                // SAFETY: passthrough shaders are what CubeCL emits.
                experimental_features: unsafe { ExperimentalFeatures::enabled() },
            })
            .await
            .map_err(|e| format!("WebGPU device request refused: {e}"))?;

        let caps = surface.get_capabilities(&adapter);
        let format = pick_format(&caps.formats)?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            // `Auto` on the browser WebGPU backend means the canvas
            // defaults: sRGB primaries, standard tone mapping, no HDR. With a
            // non-sRGB *storage* format (see `pick_format`) that means the
            // bytes the shader wrote are the bytes `toBlob()` reads.
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: canvas.width().max(1),
            height: canvas.height().max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let burn = init_burn(instance, adapter, device.clone(), queue.clone());

        Ok(Self {
            device,
            queue,
            surface,
            config,
            burn,
            adapter_info,
            has_f16,
            has_subgroups,
        })
    }

    /// Reconfigure the surface after the canvas was resized.
    ///
    /// # Arguments
    /// - `width`, `height`: the canvas' backing-store size in pixels.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if self.config.width == width && self.config.height == height {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }
}

/// Choose a **non-sRGB** 8-bit surface format; see the module invariants.
///
/// # Arguments
/// - `formats`: what the surface reports as supported, in preference order.
///
/// # Errors
/// Returns `Err` if the canvas offers no plain `unorm` format at all, which
/// would mean the screenshot comparison could not be trusted — better to say
/// so than to silently gamma-shift every frame.
fn pick_format(formats: &[wgpu::TextureFormat]) -> Result<wgpu::TextureFormat, String> {
    const PREFERRED: [wgpu::TextureFormat; 2] = [
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Rgba8Unorm,
    ];
    PREFERRED
        .into_iter()
        .find(|f| formats.contains(f))
        .ok_or_else(|| {
            format!(
                "this canvas offers no non-sRGB 8-bit surface format (got {formats:?}); \
                 rendering would be gamma-shifted against the reference render"
            )
        })
}

/// Register an existing wgpu device with Burn and return its handle.
///
/// The web twin of `trips-viewer`'s `main.rs::init_burn_on`. The `instance` is
/// moved in because `WgpuSetup` requires one; `init_device` does not use it.
fn init_burn(instance: Instance, adapter: Adapter, device: Device, queue: Queue) -> WgpuDevice {
    use burn_wgpu::graphics::{AutoGraphicsApi, GraphicsApi};
    let setup = burn_wgpu::WgpuSetup {
        instance,
        adapter,
        device,
        queue,
        backend: AutoGraphicsApi::backend(),
    };
    burn_wgpu::init_device(setup, burn_options())
}
