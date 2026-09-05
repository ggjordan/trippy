//! `trips-viewer` — the native Mac TRIPS viewer.
//!
//! Module: `trips_viewer` (binary crate)
//! Purpose: open a trippy asset bundle (points + weights + scene manifest) and
//!     render it live through the real forward pass — `brush-pyramid`'s
//!     rasteriser, `brush-unet`'s U-Net and tone mapper — in a window, at the
//!     window's own size, with a toggle between the network's frame, the raw
//!     level-0 composite and the coverage map.
//! Invariants:
//!     - Burn is initialised on **eframe's** wgpu device, never its own; that
//!       is what lets the rasteriser's output buffer be bound straight into
//!       egui's render pass (`crate::blit`).
//!     - `--screenshot` runs the identical render path headlessly and writes a
//!       PNG, so the viewer's correctness can be checked against
//!       `brush-unet`'s `render_frame_full` without anyone having to look at a
//!       window.
//!     - This binary is **separate from Brush's own `brush` binary**, which is
//!       untouched — see `docs/decisions/ADR-0006-viewer-integration.md`. Its
//!       `.ply` viewing therefore cannot regress.
//! Units: `--scale` is a fraction of the window; timings are milliseconds.
//! Related docs: `docs/USER_GUIDE.md`; `rust/README.md`;
//!     `docs/decisions/ADR-0006-viewer-integration.md`.

mod app;
mod blit;

// The platform-neutral half lives in this package's library target (`src/lib.rs`)
// so `rust/crates/trips-web` can compile the identical bundle loader, camera and
// render pipeline for wasm32. These imports are what keeps every `crate::bundle`
// / `crate::camera` / `crate::renderer` path below (and in `app.rs`/`blit.rs`)
// resolving unchanged.
use trips_viewer::{bundle, camera, renderer};

use std::path::PathBuf;

use brush_pyramid::gpu::{block_on, WgpuDevice};
use brush_pyramid::png;

use crate::bundle::Bundle;
use crate::renderer::{Renderer, Settings, ViewMode};

const USAGE: &str = "\
trips-viewer — a TRIPS scene, live, on Metal

  trips-viewer [BUNDLE_DIR] [options]

With no BUNDLE_DIR a folder picker opens.

Options:
  --scale <f>          render at this fraction of the window (default 1.0)
  --packed-sort        one packed 32-bit sort key instead of two exact passes
  --cap-fragments      start TRIPS emission at layer_lower-1, not layer 0
  --no-cull            disable the frustum box cull (measurement only)
  --fp16               store point features as f16
  --half-net           run the U-Net in f16 (the big one at 1080p)
  --view <n>           open at this dataset view index (default: the bundle's)
  --mode <m>           network | raw | coverage (default network)
  --free               open in free-fly mode instead of orbit

Headless (no window; used by the acceptance check and the perf table):
  --screenshot <o.png> render one frame to a PNG and exit
  --camera-yaw-deg <d> yaw the camera off the chosen view by <d> degrees first;
                       a scripted camera change, so two --screenshot runs can
                       prove that moving the camera reaches the renderer
  --frames <n>         warm-up frames before the screenshot / benchmark (default 2)
  --bench <n>          time <n> frames and print ms + fps, then exit

Keys:
  left-drag  orbit (or look, in free mode)   right/middle-drag  pan
  W A S D    move        Q / E   down / up along the scene up axis
  scroll     zoom in orbit mode, fly speed in free mode
  F          orbit <-> free      R  back to the view it opened at
  N / P      next / previous capture view
  V          cycle network / raw level-0 / coverage
  - / =      render scale        TAB  hide the panel
";

/// wgpu runtime options for Burn. Copied from the Brush fork's
/// `brush_process::burn_options` so this viewer's memory behaviour matches the
/// one the rasteriser's kernels were tuned against — in particular
/// `ExclusivePages`, which keeps every tensor its own buffer and is what makes
/// binding one into a wgpu pipeline sound.
fn burn_options() -> burn_wgpu::RuntimeOptions {
    burn_wgpu::RuntimeOptions {
        tasks_max: 64,
        memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
    }
}

/// Register an existing wgpu device with Burn and return its handle.
///
/// # Arguments
/// - `adapter`, `device`, `queue`: eframe's, cloned out of its render state.
pub fn init_burn_on(
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
) -> WgpuDevice {
    use burn_wgpu::graphics::{AutoGraphicsApi, GraphicsApi};
    let setup = burn_wgpu::WgpuSetup {
        // Unused by `init_device`, but the struct requires one.
        instance: wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle()),
        adapter,
        device,
        queue,
        backend: AutoGraphicsApi::backend(),
    };
    burn_wgpu::init_device(setup, burn_options())
}

struct Args {
    bundle: Option<PathBuf>,
    settings: Settings,
    view: Option<usize>,
    mode: ViewMode,
    screenshot: Option<PathBuf>,
    warmup: usize,
    bench: Option<usize>,
    /// Degrees to yaw the camera off the chosen view before rendering.
    camera_yaw_deg: Option<f32>,
    /// Open in free-fly rather than the default orbit mode.
    free: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        bundle: None,
        settings: Settings::default(),
        view: None,
        mode: ViewMode::Network,
        screenshot: None,
        warmup: 2,
        bench: None,
        camera_yaw_deg: None,
        free: false,
    };
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let mut value = || {
            argv.next()
                .ok_or_else(|| format!("{flag} needs a value\n\n{USAGE}"))
        };
        match flag.as_str() {
            "--scale" => {
                args.settings.render_scale =
                    value()?.parse().map_err(|e| format!("--scale: {e}"))?;
            }
            "--no-cull" => args.settings.no_cull = true,
            "--packed-sort" => args.settings.packed_sort = true,
            "--cap-fragments" => args.settings.cap_fragments = true,
            "--fp16" => args.settings.half_features = true,
            "--half-net" => args.settings.half_net = true,
            "--profile" => args.settings.profile = true,
            "--view" => args.view = Some(value()?.parse().map_err(|e| format!("--view: {e}"))?),
            "--mode" => {
                args.mode = match value()?.as_str() {
                    "network" => ViewMode::Network,
                    "raw" => ViewMode::RawLevel0,
                    "coverage" => ViewMode::Coverage,
                    other => return Err(format!("--mode {other:?}\n\n{USAGE}")),
                };
            }
            "--screenshot" => args.screenshot = Some(PathBuf::from(value()?)),
            "--camera-yaw-deg" => {
                args.camera_yaw_deg =
                    Some(value()?.parse().map_err(|e| format!("--camera-yaw-deg: {e}"))?);
            }
            "--free" => args.free = true,
            "--frames" => args.warmup = value()?.parse().map_err(|e| format!("--frames: {e}"))?,
            "--bench" => args.bench = Some(value()?.parse().map_err(|e| format!("--bench: {e}"))?),
            "-h" | "--help" => return Err(USAGE.to_owned()),
            other if other.starts_with('-') => {
                return Err(format!("unknown flag {other:?}\n\n{USAGE}"))
            }
            other => args.bundle = Some(PathBuf::from(other)),
        }
    }
    Ok(args)
}

/// Locate the bundle: `argv`, else a native folder picker.
fn resolve_bundle(explicit: Option<PathBuf>, headless: bool) -> Result<PathBuf, String> {
    if let Some(path) = explicit {
        return Ok(path);
    }
    if headless {
        return Err(format!("a bundle directory is required headlessly\n\n{USAGE}"));
    }
    rfd::FileDialog::new()
        .set_title("Open a TRIPS bundle (the folder holding bundle.json)")
        .pick_folder()
        .ok_or_else(|| "no bundle chosen".to_owned())
}

/// The headless paths: `--screenshot` and `--bench`.
///
/// Both create their **own** Burn device (there is no window to borrow one
/// from) and run exactly the render path the window does.
fn run_headless(args: &Args, bundle: Bundle) -> Result<(), String> {
    let home = pick_view_position(&bundle, args.view)?;
    let views = bundle.manifest.views.clone();
    let up = bundle.manifest.up;
    let view = views[home].clone();
    let device = block_on(async {
        burn_wgpu::init_setup_async::<burn_wgpu::graphics::AutoGraphicsApi>(
            &WgpuDevice::DefaultDevice,
            burn_options(),
        )
        .await;
        WgpuDevice::DefaultDevice
    });
    let frame_index = view.index;
    let camera_view = view.clone();
    let renderer = Renderer::new(bundle, device)?;
    let scale = args.settings.render_scale.clamp(0.1, 1.0);
    let width = ((camera_view.width as f32 * scale).round() as usize).max(16);
    let height = ((camera_view.height as f32 * scale).round() as usize).max(16);
    let mut controller = crate::camera::Controller::new(&views, home, up);
    if args.free {
        controller.set_mode(crate::camera::Mode::Free);
    }
    // A scripted camera change: `yaw` unpins even at zero degrees, so two runs
    // with different angles differ only in the rotation, never in which code
    // path built the camera.
    if let Some(degrees) = args.camera_yaw_deg {
        controller.yaw(degrees.to_radians());
        eprintln!("camera yawed {degrees} deg off view {}", camera_view.index);
    }
    let camera = controller.render_camera(width, height, &camera_view);

    eprintln!(
        "{} points, {width}x{height}, view {frame_index} ({}), levers: {}",
        renderer.num_points(),
        camera_view.name,
        if args.settings.is_exact() {
            "exact".to_owned()
        } else {
            format!("{:?}", args.settings)
        }
    );

    // Warm-up frames pay for shader compilation and buffer-pool growth, which
    // is not a viewer's steady state.
    for _ in 0..args.warmup.max(1) {
        block_on(renderer.render(&camera, frame_index, args.mode, &args.settings))?;
    }

    if args.settings.profile {
        let frame = block_on(renderer.render(
            &camera,
            frame_index,
            args.mode,
            &args.settings,
        ))?;
        if let Some(s) = frame.stats.stages {
            println!(
                "PROFILE upload {:.1} | project {:.1} | prefix {:.1} | emit {:.1} | sort \
                 {:.1} ({} radix passes over {} slots) | segment {:.1} | blend {:.1} | sum \
                 {:.1} ms",
                s.upload_ms,
                s.project_count_ms,
                s.prefix_ms,
                s.emit_ms,
                s.sort_ms,
                s.radix_passes,
                s.fragment_slots,
                s.segment_ms,
                s.blend_ms,
                s.total_ms
            );
        }
    }

    if let Some(count) = args.bench {
        let mut samples = Vec::with_capacity(count);
        let mut profile = args.settings;
        // Never profile inside the timed loop: its per-stage syncs are the one
        // thing that would make the number wrong.
        profile.profile = false;
        for _ in 0..count.max(1) {
            let start = std::time::Instant::now();
            let frame = block_on(renderer.render(&camera, frame_index, args.mode, &profile))?;
            // Draining the queue is the honest end-of-frame barrier, and moves
            // no data -- unlike a readback, which would charge the frame for
            // 24 MB of transfer the window never pays.
            block_on(brush_pyramid::gpu::sync(frame_device(&frame)))?;
            samples.push(start.elapsed().as_secs_f64() * 1e3);
        }
        samples.sort_by(f64::total_cmp);
        let median = samples[samples.len() / 2];
        println!(
            "BENCH {width}x{height} median over {} frames: {median:.2} ms ({:.2} fps){}",
            samples.len(),
            1000.0 / median,
            if args.settings.is_exact() {
                "  [exact]"
            } else {
                "  [approximate]"
            }
        );
    }

    if let Some(out) = &args.screenshot {
        let (data, channels, height, width) =
            block_on(renderer.render_to_host(&camera, frame_index, &args.settings))?;
        let pixels = png::feature_to_rgb8(&data, channels, height, width, 1.0)?;
        png::write_rgb8(out, &pixels, width, height)?;
        eprintln!("wrote {}", out.display());
    }
    Ok(())
}

/// The device a finished frame's buffer lives on.
fn frame_device(frame: &crate::renderer::RenderedFrame) -> &WgpuDevice {
    &frame.buffer.device
}

/// Resolve `--view <dataset index>` to an ARRAY POSITION in `manifest.views`,
/// defaulting to the bundle's own `default_view` (which is already a position;
/// see `crate::bundle::Manifest`).
fn pick_view_position(bundle: &Bundle, index: Option<usize>) -> Result<usize, String> {
    let views = &bundle.manifest.views;
    match index {
        None => Ok(bundle.home_view_position()),
        Some(wanted) => views
            .iter()
            .position(|v| v.index == wanted)
            .ok_or_else(|| format!("no view with index {wanted} in this bundle")),
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    // `--profile` prints per-stage numbers and must not open a window: a
    // headless queue job has no display, and an earlier version hung there.
    let headless = args.screenshot.is_some() || args.bench.is_some() || args.settings.profile;
    let dir = resolve_bundle(args.bundle.clone(), headless)?;
    let bundle = Bundle::load(&dir)?;
    eprintln!(
        "bundle {:?}: {} points, C = {}, {} views",
        bundle.manifest.name,
        bundle.points.len(),
        bundle.manifest.num_channels,
        bundle.manifest.views.len()
    );

    if headless {
        return run_headless(&args, bundle);
    }

    let mut settings = args.settings;
    let bundle = if let Some(wanted) = args.view {
        let position = bundle
            .manifest
            .views
            .iter()
            .position(|v| v.index == wanted)
            .ok_or_else(|| format!("no view with index {wanted} in this bundle"))?;
        let mut bundle = bundle;
        bundle.manifest.default_view = position;
        bundle
    } else {
        bundle
    };
    settings.render_scale = settings.render_scale.clamp(0.1, 1.0);
    let title = format!("TRIPS — {}", bundle.manifest.name);
    let mode = args.mode;
    let navigation = if args.free {
        crate::camera::Mode::Free
    } else {
        crate::camera::Mode::Orbit
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(egui::Vec2::new(1600.0, 950.0))
            .with_active(true),
        wgpu_options: egui_options(),
        ..Default::default()
    };
    eframe::run_native(
        &title,
        options,
        Box::new(move |cc| {
            let mut app = app::ViewerApp::new(cc, bundle, settings)
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
            app.set_mode(mode);
            app.set_navigation(navigation);
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| format!("eframe: {e}"))
}

/// wgpu setup for egui.
///
/// Copied from the Brush fork's `apps/brush-app/src/ui/mod.rs`
/// `create_egui_options` (Apache-2.0, ArthurBrussee) and kept byte-compatible
/// with it: `MAPPABLE_PRIMARY_BUFFERS` must be excluded and the experimental
/// passthrough shaders must be enabled, or CubeCL's kernels will not run on
/// the device egui created.
fn egui_options() -> eframe::egui_wgpu::WgpuConfiguration {
    use std::sync::Arc;
    use wgpu::{Adapter, ExperimentalFeatures, Features};

    eframe::egui_wgpu::WgpuConfiguration {
        wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(
            eframe::egui_wgpu::WgpuSetupCreateNew {
                instance_descriptor: wgpu::InstanceDescriptor::new_without_display_handle(),
                display_handle: None,
                native_adapter_selector: None,
                power_preference: wgpu::PowerPreference::HighPerformance,
                device_descriptor: Arc::new(|adapter: &Adapter| wgpu::DeviceDescriptor {
                    label: Some("egui+burn (trips-viewer)"),
                    required_features: adapter
                        .features()
                        .difference(Features::MAPPABLE_PRIMARY_BUFFERS),
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                    // SAFETY: passthrough shaders are what CubeCL emits.
                    experimental_features: unsafe { ExperimentalFeatures::enabled() },
                }),
            },
        ),
        ..Default::default()
    }
}

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .init();
    if let Err(message) = run() {
        eprintln!("{message}");
        std::process::exit(1);
    }
}
