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
use trips_viewer::{blend, bundle, camera, renderer};

use std::path::PathBuf;

use brush_pyramid::gpu::{block_on, WgpuDevice};
use brush_pyramid::png;

use crate::bundle::Bundle;
use crate::blend::{Blend, BlendMode};
use crate::renderer::{ExposureMode, Renderer, Settings, ViewMode};

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
  --exposure <e>       auto | view | median | <EV>  (default auto: the view's
                       own exposure while pinned to it, the scene median once
                       you have moved off it)
  --free               open in free-fly mode instead of orbit

Blend panel (any bundle with a `blend` block -- a hybrid run, or any run seeded
from a Gaussian .ply; see docs/USER_GUIDE.md):
  --blend-mode <m>     trips | splat | gated | mix | split (default trips)
  --gate-scale <f>     0..2 multiplier on the trained gate, for `gated`
                       (0 = TRIPS, 1 = as trained, 2 = pushed to the splat)
  --mix <f>            0 = splat, 1 = TRIPS, for `mix`
  --split <f>          fraction of the width where `split` changes over
  --splat-ply <p>      render THIS .ply live instead of the one bundle.json
                       names (`blend.splat_ply`)
  --no-live-splat      do not open the ply at all: fall back to the bundle's
                       precomputed capture-view renders, as before v0.6.0
  --splat-subsample <n> keep every n-th Gaussian while parsing (a stride, so it
                       also shrinks the parser's allocation -- the only lever
                       that helps peak memory on a multi-GB ply)

Headless (no window; used by the acceptance check and the perf table):
  --screenshot <o.png> render one frame to a PNG and exit
  --camera-yaw-deg <d> yaw the camera off the chosen view by <d> degrees first;
                       a scripted camera change, so two --screenshot runs can
                       prove that moving the camera reaches the renderer
  --frames <n>         warm-up frames before the screenshot / benchmark (default 2)
  --bench <n>          time <n> frames and print ms + fps, then exit
  --render-size <WxH>  render at exactly this size instead of the view's own
                       size times --scale (so any bundle can be measured at
                       1080p, whatever its capture resolution)
  --splat-bench <n>    time <n> LIVE SPLAT renders on their own -- no pyramid,
                       no U-Net -- and print the median ms, then exit

Keys:
  left-drag  orbit (or look, in free mode)   right/middle-drag  pan
  W A S D    move        Q / E   down / up along the scene up axis
  scroll     zoom in orbit mode, fly speed in free mode
  F          orbit <-> free      R  back to the view it opened at
  N / P      next / previous capture view
  B          cycle the blend mode (hybrid bundles)
  V          cycle network / raw level-0 / coverage
  X          cycle the exposure the tone mapper applies
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
    exposure: ExposureMode,
    screenshot: Option<PathBuf>,
    warmup: usize,
    bench: Option<usize>,
    /// Degrees to yaw the camera off the chosen view before rendering.
    camera_yaw_deg: Option<f32>,
    /// Open in free-fly rather than the default orbit mode.
    free: bool,
    /// The Blend panel's starting state.
    blend: Blend,
    /// How to get the live Gaussian splat, if at all.
    splat: SplatArgs,
    /// `--render-size WxH`: an explicit headless render resolution.
    render_size: Option<(usize, usize)>,
    /// `--splat-bench <n>`: time the splat render alone.
    splat_bench: Option<usize>,
}

/// Parse `1920x1080`.
fn parse_size(text: &str) -> Result<(usize, usize), String> {
    let (w, h) = text
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("--render-size {text:?}: expected WxH, e.g. 1920x1080"))?;
    let width: usize = w.trim().parse().map_err(|e| format!("--render-size width: {e}"))?;
    let height: usize = h.trim().parse().map_err(|e| format!("--render-size height: {e}"))?;
    if width == 0 || height == 0 {
        return Err(format!("--render-size {text:?}: both sides must be > 0"));
    }
    Ok((width, height))
}

/// The live-splat flags. Kept together so `run_headless` and the window take
/// the same three decisions through the same code.
#[derive(Clone, Debug, Default)]
struct SplatArgs {
    /// `--splat-ply`, which overrides `bundle.json`'s `blend.splat_ply`.
    ply: Option<PathBuf>,
    /// `--no-live-splat`: keep the pre-v0.6.0 precomputed-only behaviour.
    disabled: bool,
    /// `--splat-subsample <n>`.
    subsample: Option<u32>,
}

/// Open the bundle's Gaussian `.ply` and hand it to `renderer`, printing what
/// happened.
///
/// Never fatal. A bundle with no `blend.splat_ply`, a `--no-live-splat` run, or
/// a ply that will not open all leave the renderer on the precomputed path,
/// which is exactly the behaviour every bundle had before this existed. The
/// only difference is a line on stderr saying which of those it was.
///
/// # Arguments
/// - `renderer`: already built on the device the window (or the headless run)
///   uses.
/// - `dir`: the bundle directory, so a relative `splat_ply` resolves against it.
/// - `blend`: the bundle's `blend` block, or `None`.
/// - `args`: the three flags.
fn attach_live_splat(
    renderer: &mut Renderer,
    dir: &std::path::Path,
    blend: Option<&crate::bundle::BlendManifest>,
    args: &SplatArgs,
) {
    if args.disabled {
        eprintln!("--no-live-splat: the Blend panel uses the precomputed capture-view renders");
        return;
    }
    let named = args.ply.clone().or_else(|| {
        blend
            .map(|b| b.splat_ply.trim())
            .filter(|p| !p.is_empty())
            .map(PathBuf::from)
    });
    let Some(named) = named else {
        return;
    };
    // A bundle written on this machine records an absolute path; a bundle
    // someone moved may not, so a relative one is read next to `bundle.json`.
    let path = if named.is_absolute() {
        named
    } else {
        dir.join(named)
    };
    eprintln!("loading splat {} ...", path.display());
    let started = std::time::Instant::now();
    match renderer.load_live_splat(&path, args.subsample) {
        Ok(splat) => eprintln!(
            "splat loaded: {} Gaussians, SH degree {}, {:.0} ms",
            splat.num_splats(),
            splat.sh_degree(),
            splat.load_ms()
        ),
        Err(e) => eprintln!(
            "splat NOT loaded after {:.0} ms ({e}); the Blend panel falls back to the \
             bundle's precomputed capture-view renders",
            started.elapsed().as_secs_f64() * 1e3
        ),
    }
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        bundle: None,
        settings: Settings::default(),
        view: None,
        mode: ViewMode::Network,
        exposure: ExposureMode::default(),
        screenshot: None,
        warmup: 2,
        bench: None,
        camera_yaw_deg: None,
        free: false,
        blend: Blend::default(),
        splat: SplatArgs::default(),
        render_size: None,
        splat_bench: None,
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
            "--exposure" => {
                let raw = value()?;
                args.exposure = match raw.as_str() {
                    "auto" => ExposureMode::Auto,
                    "view" => ExposureMode::View,
                    "median" => ExposureMode::Median,
                    other => ExposureMode::Manual(
                        other
                            .parse()
                            .map_err(|e| format!("--exposure {other:?}: {e}\n\n{USAGE}"))?,
                    ),
                };
            }
            "--screenshot" => args.screenshot = Some(PathBuf::from(value()?)),
            "--camera-yaw-deg" => {
                args.camera_yaw_deg =
                    Some(value()?.parse().map_err(|e| format!("--camera-yaw-deg: {e}"))?);
            }
            "--blend-mode" => args.blend.mode = BlendMode::parse(&value()?)?,
            "--gate-scale" => {
                args.blend.gate_scale =
                    value()?.parse().map_err(|e| format!("--gate-scale: {e}"))?;
            }
            "--mix" => args.blend.mix = value()?.parse().map_err(|e| format!("--mix: {e}"))?,
            "--split" => args.blend.split = value()?.parse().map_err(|e| format!("--split: {e}"))?,
            "--render-size" => args.render_size = Some(parse_size(&value()?)?),
            "--splat-bench" => {
                args.splat_bench =
                    Some(value()?.parse().map_err(|e| format!("--splat-bench: {e}"))?);
            }
            "--splat-ply" => args.splat.ply = Some(PathBuf::from(value()?)),
            "--no-live-splat" => args.splat.disabled = true,
            "--splat-subsample" => {
                args.splat.subsample = Some(
                    value()?
                        .parse()
                        .map_err(|e| format!("--splat-subsample: {e}"))?,
                );
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
    let bundle_dir = bundle.dir.clone();
    let blend_manifest = bundle.manifest.blend.clone();
    let mut renderer = Renderer::new(bundle, device.clone())?;
    attach_live_splat(&mut renderer, &bundle_dir, blend_manifest.as_ref(), &args.splat);
    let scale = args.settings.render_scale.clamp(0.1, 1.0);
    // `--render-size` wins over `--scale`: measuring a 1080p frame must not
    // depend on the capture happening to be 1080p.
    let (width, height) = args.render_size.unwrap_or((
        ((camera_view.width as f32 * scale).round() as usize).max(16),
        ((camera_view.height as f32 * scale).round() as usize).max(16),
    ));
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
    // `--camera-yaw-deg` unpins the controller, so `ExposureMode::Auto` here
    // means exactly what it means in the window: the view's own exposure for a
    // frame taken from that view, the scene median for one taken from
    // somewhere else.
    renderer.set_exposure(args.exposure, controller.is_pinned());

    eprintln!(
        "{} points, {width}x{height}, view {frame_index} ({}), exposure {} (EV {}), levers: {}",
        renderer.num_points(),
        camera_view.name,
        args.exposure.label(),
        match renderer
            .exposure()
            .resolve(controller.is_pinned(), renderer.median_exposure())
            .or_else(|| renderer.view_exposure(frame_index))
        {
            Some(ev) => format!("{ev:+.4}"),
            None => "none".to_owned(),
        },
        if args.settings.is_exact() {
            "exact".to_owned()
        } else {
            format!("{:?}", args.settings)
        }
    );

    // Warm-up frames pay for shader compilation and buffer-pool growth, which
    // is not a viewer's steady state.
    for _ in 0..args.warmup.max(1) {
        block_on(renderer.render(&camera, frame_index, args.mode, &args.settings, args.blend))?;
    }

    if args.settings.profile {
        let frame = block_on(renderer.render(
            &camera,
            frame_index,
            args.mode,
            &args.settings,
            args.blend,
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
            let frame = block_on(renderer.render(&camera, frame_index, args.mode, &profile, args.blend))?;
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

    if let Some(count) = args.splat_bench {
        // The splat render ON ITS OWN: no pyramid, no U-Net, no compositing --
        // the number the perf table wants, and the one a whole-frame `--bench`
        // cannot separate out.
        let Some(splat) = renderer.live_splat() else {
            return Err(
                "--splat-bench needs a live splat; this bundle names no `blend.splat_ply` \
                 (or --no-live-splat was passed)"
                    .to_owned(),
            );
        };
        // One untimed render pays for shader compilation and pool growth, and
        // reports how much of the splat this camera can actually see -- see
        // `LiveSplat::render_counting`.
        let (_warm, (visible, intersections)) = block_on(splat.render_counting(&camera))?;
        block_on(brush_pyramid::gpu::sync(&device))?;
        if visible == 0 {
            eprintln!(
                "WARNING: 0 of {} Gaussians are visible from this camera. The ply and the \
                 bundle are probably from different reconstructions, and the timing below \
                 measures the frustum cull, not the rasteriser.",
                splat.num_splats()
            );
        }
        let mut samples = Vec::with_capacity(count.max(1));
        for _ in 0..count.max(1) {
            let start = std::time::Instant::now();
            let _image = block_on(splat.render(&camera))?;
            // Drain the queue: without it every sample would time a submission,
            // not a render, and the median would be meaningless.
            block_on(brush_pyramid::gpu::sync(&device))?;
            samples.push(start.elapsed().as_secs_f64() * 1e3);
        }
        samples.sort_by(f64::total_cmp);
        let median = samples[samples.len() / 2];
        println!(
            "SPLAT-BENCH {width}x{height} {} Gaussians, SH degree {}: median over {} renders \
             {median:.2} ms ({:.1} fps); {visible} visible, {intersections} tile \
             intersections; load {:.0} ms",
            splat.num_splats(),
            splat.sh_degree(),
            samples.len(),
            1000.0 / median,
            splat.load_ms(),
        );
    }

    if let Some(out) = &args.screenshot {
        let (data, channels, height, width) =
            block_on(renderer.render_to_host(&camera, frame_index, &args.settings, args.blend))?;
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
    let headless = args.screenshot.is_some()
        || args.bench.is_some()
        || args.splat_bench.is_some()
        || args.settings.profile;
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
    let blend_state = args.blend;
    let splat_args = args.splat.clone();
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
            let mut app = app::ViewerApp::new(cc, bundle, settings, &splat_args)
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
            app.set_mode(mode);
            app.set_navigation(navigation);
            if blend_state != Blend::default() {
                // Only override the bundle's own starting state when a flag asked
                // for something else, so a plain launch still opens on the frame
                // the run's metrics describe.
                app.set_blend(blend_state);
            }
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
