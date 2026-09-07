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
mod edit_ui;

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

Editing (docs/EDITOR.md; press M in the window for the panels):
  --edit               open with the Regions/Inspector/Tools panels already up
                       (the same thing M toggles)
  --edits <p>          read this edits.json instead of <BUNDLE_DIR>/edits.json
  --dump-weights <o>   compose the per-point weights from edits.json, write them
                       to <o> as JSON and exit. No GPU, no window: this is the
                       half of the Python/Rust parity check that runs against a
                       REAL bundle (`trippy apply-edits` must agree to 1e-6)
  --dump-shade <o>     run the shade-cloud finder over <BUNDLE_DIR>/shade_views.json
                       at the thresholds below and write the selected point ids
                       to <o> as JSON (the headless twin of the Tools panel);
                       must select the same ids as `trippy edits shade-find`
  --shade-lum <f>      dark-luminance cutoff (default 0.25)
  --shade-conf <f>     effective-confidence cutoff (default 0.50)
  --shade-znear <f>    near plane / each frame's median depth (default: the
                       sidecar's own, else 0.05)
  --shade-zfar <f>     far plane / median depth (default: the sidecar's, else 0.50)
  --click <U> <V>      click-to-cluster at pixel (U, V) of the chosen view, as
                       SHIFT-CLICK does in the window. With --screenshot the
                       selection is tinted into the frame; with --dump-click the
                       selected ids are written out. Both together are the E4
                       screenshot proof: tint on, then a run without --click
                       for the clear-restores-it half
  --dump-click <o>     write the click's selected point ids to <o> as JSON and
                       exit. No GPU, no window; must select the same ids as
                       `trippy edits click` at the same view/pixel/parameters
  --click-radius-px <f>   click catchment radius, pixels (default 12)
  --click-colour-tol <f>  colour gate, distance in [0,1]^3 (default 0.15)
  --click-max-radius <f>  growth cap from the seed centroid, world units
                       (default: the bundle's median camera spacing)
  --click-max-points <n>  hard cap on the selection (default 200000)

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
  M          edit mode: Regions / Inspector / Tools (docs/EDITOR.md)
  shift-click  in edit mode: select the object under the pointer (E4)
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
    /// `--edit`: open with the editor's panels shown.
    edit: bool,
    /// `--edits <p>`: an `edits.json` somewhere other than next to `bundle.json`.
    edits: Option<PathBuf>,
    /// `--dump-weights <o>`: write the composed per-point weights and exit.
    dump_weights: Option<PathBuf>,
    /// `--dump-shade <o>`: write the shade finder's selection and exit.
    dump_shade: Option<PathBuf>,
    /// `--shade-lum` / `--shade-conf` / `--shade-znear` / `--shade-zfar`, each
    /// `None` for "the sidecar's own value, else the shipped default".
    shade_lum: Option<f64>,
    /// See [`Self::shade_lum`].
    shade_conf: Option<f64>,
    /// See [`Self::shade_lum`].
    shade_znear: Option<f64>,
    /// See [`Self::shade_lum`].
    shade_zfar: Option<f64>,
    /// `--click U V`: the pixel to cluster from, in the chosen view's own
    /// pixel coordinates (the headless twin of a Shift-click).
    click: Option<(f64, f64)>,
    /// `--dump-click <o>`: write the click's selection and exit.
    dump_click: Option<PathBuf>,
    /// `--click-radius-px`, each `None` for "the shipped default".
    click_radius_px: Option<f64>,
    /// See [`Self::click_radius_px`].
    click_colour_tol: Option<f64>,
    /// See [`Self::click_radius_px`]; `None` means the bundle's own median
    /// camera spacing (`trippy.edit.cluster.default_max_radius_from_bundle`).
    click_max_radius: Option<f64>,
    /// See [`Self::click_radius_px`].
    click_max_points: Option<usize>,
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
        edit: false,
        edits: None,
        dump_weights: None,
        dump_shade: None,
        shade_lum: None,
        shade_conf: None,
        shade_znear: None,
        shade_zfar: None,
        click: None,
        dump_click: None,
        click_radius_px: None,
        click_colour_tol: None,
        click_max_radius: None,
        click_max_points: None,
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
            "--edit" => args.edit = true,
            "--edits" => args.edits = Some(PathBuf::from(value()?)),
            "--dump-weights" => args.dump_weights = Some(PathBuf::from(value()?)),
            "--dump-shade" => args.dump_shade = Some(PathBuf::from(value()?)),
            "--shade-lum" => {
                args.shade_lum = Some(value()?.parse().map_err(|e| format!("--shade-lum: {e}"))?);
            }
            "--shade-conf" => {
                args.shade_conf = Some(value()?.parse().map_err(|e| format!("--shade-conf: {e}"))?);
            }
            "--shade-znear" => {
                args.shade_znear =
                    Some(value()?.parse().map_err(|e| format!("--shade-znear: {e}"))?);
            }
            "--shade-zfar" => {
                args.shade_zfar = Some(value()?.parse().map_err(|e| format!("--shade-zfar: {e}"))?);
            }
            "--click" => {
                let u: f64 = value()?.parse().map_err(|e| format!("--click U: {e}"))?;
                let v: f64 = value()?.parse().map_err(|e| format!("--click V: {e}"))?;
                args.click = Some((u, v));
            }
            "--dump-click" => args.dump_click = Some(PathBuf::from(value()?)),
            "--click-radius-px" => {
                args.click_radius_px =
                    Some(value()?.parse().map_err(|e| format!("--click-radius-px: {e}"))?);
            }
            "--click-colour-tol" => {
                args.click_colour_tol =
                    Some(value()?.parse().map_err(|e| format!("--click-colour-tol: {e}"))?);
            }
            "--click-max-radius" => {
                args.click_max_radius =
                    Some(value()?.parse().map_err(|e| format!("--click-max-radius: {e}"))?);
            }
            "--click-max-points" => {
                args.click_max_points =
                    Some(value()?.parse().map_err(|e| format!("--click-max-points: {e}"))?);
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

/// `"format"` of the file `--dump-weights` writes.
const DUMP_WEIGHTS_FORMAT: &str = "trippy-edit-weights-1";

/// Compose the per-point weights from `edits.json` and write them as JSON.
///
/// **No GPU and no window.** This is the half of the Python/Rust parity check
/// that runs against a real bundle: `trippy apply-edits` writes the same
/// `weight` array into `blend_weights.npy`, and the two must agree to 1e-6
/// (`docs/EDITOR.md` §6, `trippy/edit/golden.py`). The committed fixture in
/// `tests/fixtures/synthetic/edit_golden/` is the automated version of the same
/// comparison; this one is what you point at a scene.
///
/// # Arguments
/// - `bundle`: the loaded scene (only its `points.npz` `xyz` is read).
/// - `edits`: the `edits.json` to compose; a missing file composes to the
///   identity, which is a meaningful answer rather than an error.
/// - `out`: destination JSON.
///
/// # Errors
/// Returns `Err` when `edits` exists but does not parse or does not validate
/// against this bundle's format, or when `out` cannot be written.
fn dump_weights(bundle: &Bundle, edits: &std::path::Path, out: &std::path::Path) -> Result<(), String> {
    use trips_viewer::edit::weights::{compose_trips_weights, widen};
    use trips_viewer::edit::EditDocument;

    let doc = if edits.exists() {
        EditDocument::load(edits)?
    } else {
        eprintln!("{} does not exist: composing the identity", edits.display());
        EditDocument::new(bundle.manifest.format.clone())
    };
    doc.validate(Some(&bundle.manifest.format))?;
    let composed = compose_trips_weights(&doc, &widen(&bundle.points.xyz));
    let document = serde_json::json!({
        "format": DUMP_WEIGHTS_FORMAT,
        "bundle": bundle.dir.display().to_string(),
        "edits": edits.display().to_string(),
        "n": composed.weight.len(),
        "num_regions": doc.regions().len(),
        "num_deleted": composed.num_deleted(),
        "num_touched": composed.num_touched(),
        "weight": composed.weight,
        "delete_mask": composed.delete_mask,
    });
    let text =
        serde_json::to_string(&document).map_err(|e| format!("serialising weights: {e}"))?;
    std::fs::write(out, text).map_err(|e| format!("{}: {e}", out.display()))?;
    eprintln!(
        "wrote {} ({} points, {} deleted, {} region-mixed, {} regions)",
        out.display(),
        composed.weight.len(),
        composed.num_deleted(),
        composed.num_touched(),
        doc.regions().len()
    );
    Ok(())
}

/// `"format"` of the file `--dump-shade` writes.
const DUMP_SHADE_FORMAT: &str = "trippy-edit-shade-1";

/// Run the shade-cloud finder headlessly and write its selection.
///
/// The Tools panel's sliders, without a window: the same
/// `trips_viewer::edit::shade::find` the panel calls, over the same
/// `shade_views.json`, so "the viewer selects what
/// `trippy.edit.shade_finder.find_shade_pointset` selects" is checkable without
/// anyone opening a GUI (`docs/EDITOR.md` §6, E2's acceptance).
///
/// Base colour is `clip(feat[:, :3], 0, 1)`, exactly the slice
/// `find_shade_pointset_in_bundle` reads and the exporter writes.
///
/// # Arguments
/// - `bundle`: the loaded scene.
/// - `args`: the four `--shade-*` overrides.
/// - `out`: destination JSON.
///
/// # Errors
/// Returns `Err` when the bundle carries no `shade_views.json`, when it does
/// not parse, or when `out` cannot be written.
fn dump_shade(bundle: &Bundle, args: &Args, out: &std::path::Path) -> Result<(), String> {
    use trips_viewer::edit::shade;

    let sidecar = shade::ShadeViews::load(&bundle.dir)?.ok_or_else(|| {
        format!(
            "{} has no {}; write one with trippy.edit.golden.write_shade_views",
            bundle.dir.display(),
            shade::SHADE_VIEWS_FILENAME
        )
    })?;
    let defaults = shade::Thresholds::default();
    let thresholds = shade::Thresholds {
        znear_frac: args.shade_znear.unwrap_or(sidecar.znear_frac),
        zfar_frac: args.shade_zfar.unwrap_or(sidecar.zfar_frac),
        lum_threshold: args.shade_lum.unwrap_or(defaults.lum_threshold),
        conf_threshold: args.shade_conf.unwrap_or(defaults.conf_threshold),
    };

    let points = &bundle.points;
    let channels = points.num_channels;
    let mut rgb = Vec::with_capacity(points.len() * 3);
    for row in 0..points.len() {
        for c in 0..3 {
            rgb.push(f64::from(points.feat[row * channels + c].clamp(0.0, 1.0)));
        }
    }
    let conf: Vec<f64> = points.conf.iter().map(|c| f64::from(*c)).collect();
    let found = shade::find(
        &sidecar.views,
        &trips_viewer::edit::weights::widen(&points.xyz),
        &rgb,
        &conf,
        thresholds,
    );

    let document = serde_json::json!({
        "format": DUMP_SHADE_FORMAT,
        "bundle": bundle.dir.display().to_string(),
        "frames": sidecar.views.iter().map(|v| v.name.clone()).collect::<Vec<_>>(),
        "thresholds": {
            "znear_frac": thresholds.znear_frac,
            "zfar_frac": thresholds.zfar_frac,
            "lum_threshold": thresholds.lum_threshold,
            "conf_threshold": thresholds.conf_threshold,
        },
        "n": points.len(),
        "n_in_region": found.n_in_region,
        "mass_in_region": found.mass_in_region,
        "dark_mass_fraction": found.dark_mass_fraction,
        "point_ids": found.point_ids,
    });
    let text = serde_json::to_string(&document).map_err(|e| format!("serialising shade: {e}"))?;
    std::fs::write(out, text).map_err(|e| format!("{}: {e}", out.display()))?;
    eprintln!(
        "wrote {} ({} of {} points selected, {} in region, dark mass fraction {:.6})",
        out.display(),
        found.point_ids.len(),
        points.len(),
        found.n_in_region,
        found.dark_mass_fraction
    );
    Ok(())
}

/// `"format"` of the file `--dump-click` writes.
const DUMP_CLICK_FORMAT: &str = "trippy-edit-click-dump-1";

/// The click tool's parameters for this run: the flags, over the bundle's own defaults.
///
/// `max_radius` defaults to the bundle's median nearest-CAMERA spacing, exactly
/// as `trippy.edit.cluster.default_max_radius_from_bundle` computes it, so a
/// `--click` with no `--click-max-radius` and a `trippy edits click` with no
/// `--max-radius` are the same click.
fn click_params_for(bundle: &Bundle, args: &Args) -> trips_viewer::edit::ClickParams {
    use trips_viewer::edit::cluster;

    let defaults = cluster::ClickParams::default();
    let max_radius = args.click_max_radius.unwrap_or_else(|| {
        let mut centres = Vec::with_capacity(bundle.manifest.views.len() * 3);
        for view in &bundle.manifest.views {
            let c = view.position();
            centres.extend_from_slice(&[f64::from(c.x), f64::from(c.y), f64::from(c.z)]);
        }
        if centres.len() / 3 >= 2 {
            cluster::default_max_radius(&centres, &[])
        } else {
            cluster::default_max_radius(
                &centres,
                &trips_viewer::edit::weights::widen(&bundle.points.xyz),
            )
        }
    });
    cluster::ClickParams {
        radius_px: args.click_radius_px.unwrap_or(defaults.radius_px),
        colour_tol: args.click_colour_tol.unwrap_or(defaults.colour_tol),
        max_radius,
        max_points: args.click_max_points.unwrap_or(defaults.max_points),
        ..defaults
    }
}

/// Run click-to-cluster headlessly and write its selection.
///
/// The Selection panel's Shift-click, without a window: the same
/// `trips_viewer::edit::cluster::click_to_cluster` the panel calls, projected
/// with the chosen view's OWN camera at its own resolution (never a
/// window-shaped one), so "the viewer selects what `trippy edits click`
/// selects" is checkable without anyone opening a GUI
/// (`tests/test_edit_viewer_parity.py`, `docs/EDITOR.md` §6's E4 row).
///
/// Base colour is `clip(feat[:, :3], 0, 1)`, the slice
/// `trippy.edit.cluster.click_to_cluster` reads.
///
/// # Arguments
/// - `bundle`: the loaded scene.
/// - `args`: `--click`, `--view` and the four `--click-*` overrides.
/// - `out`: destination JSON.
///
/// # Errors
/// Returns `Err` when `--click` was not given, when `--view` names no view, or
/// when `out` cannot be written.
fn dump_click(bundle: &Bundle, args: &Args, out: &std::path::Path) -> Result<(), String> {
    use trips_viewer::edit::cluster;

    let px = args
        .click
        .ok_or("--dump-click needs --click <U> <V> (the pixel to cluster from)")?;
    let position = pick_view_position(bundle, args.view)?;
    let view = &bundle.manifest.views[position];
    let camera = cluster::ClickCamera::from_render_camera(&view.camera());
    let params = click_params_for(bundle, args);

    let points = &bundle.points;
    let channels = points.num_channels;
    let mut rgb = Vec::with_capacity(points.len() * 3);
    for row in 0..points.len() {
        for c in 0..3 {
            rgb.push(f64::from(points.feat[row * channels + c].clamp(0.0, 1.0)));
        }
    }
    let xyz = trips_viewer::edit::weights::widen(&points.xyz);
    let started = std::time::Instant::now();
    let grid = cluster::PointGrid::build(&xyz);
    let index_ms = started.elapsed().as_secs_f64() * 1e3;
    let started = std::time::Instant::now();
    let found = cluster::click_to_cluster(&grid, &xyz, &rgb, &camera, px, &params);
    let cluster_ms = started.elapsed().as_secs_f64() * 1e3;

    let document = serde_json::json!({
        "format": DUMP_CLICK_FORMAT,
        "bundle": bundle.dir.display().to_string(),
        "view": view.name,
        "view_index": view.index,
        "px": [px.0, px.1],
        "params": {
            "radius_px": params.radius_px,
            "colour_tol": params.colour_tol,
            "max_radius": params.max_radius,
            "max_points": params.max_points,
            "knn_k": params.knn_k,
            "depth_gap_factor": params.depth_gap_factor,
        },
        "n": points.len(),
        "n_candidates": found.n_candidates,
        "n_seed": found.n_seed,
        "n_selected": found.point_ids.len(),
        "hit_max_points": found.hit_max_points,
        "seed_depth_mean": found.seed_depth_mean,
        "warning": found.warning,
        "index_ms": index_ms,
        "cluster_ms": cluster_ms,
        "point_ids": found.point_ids,
    });
    let text = serde_json::to_string(&document).map_err(|e| format!("serialising click: {e}"))?;
    std::fs::write(out, text).map_err(|e| format!("{}: {e}", out.display()))?;
    eprintln!(
        "wrote {} ({} of {} points selected from ({}, {}) in view {}; {} candidates, {} seeded; \
         index {index_ms:.0} ms, cluster {cluster_ms:.0} ms)",
        out.display(),
        found.point_ids.len(),
        points.len(),
        px.0,
        px.1,
        view.name,
        found.n_candidates,
        found.n_seed,
    );
    Ok(())
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
    let bundle_format = bundle.manifest.format.clone();
    let blend_manifest = bundle.manifest.blend.clone();
    // Resolved before the bundle moves into the renderer: the default
    // `max_radius` needs every view's camera centre.
    let click_params = args.click.map(|_| click_params_for(&bundle, args));
    let mut renderer = Renderer::new(bundle, device.clone())?;
    attach_live_splat(&mut renderer, &bundle_dir, blend_manifest.as_ref(), &args.splat);
    // `--screenshot` must draw the picture the WINDOW draws, edits included --
    // that is the whole point of the screenshot check (`main.rs`'s own
    // invariant). A bundle with no `edits.json` applies nothing and costs
    // nothing.
    let mut edits = crate::edit_ui::EditSession::open_with_path(
        &bundle_dir,
        &bundle_format,
        args.edits.as_deref(),
    );
    edits.apply(&mut renderer)?;
    if let Some((deleted, touched)) = renderer.edit_summary() {
        eprintln!(
            "edits applied: {} regions, {deleted} points deleted, {touched} region-mixed",
            edits.doc.regions().len()
        );
    }
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

    // `--click U V`: the headless twin of a Shift-click. The selection is
    // tinted into the frame the screenshot writes -- that tint changing, and
    // going away again on a run without `--click`, is E4's screenshot proof
    // (`docs/EDITOR.md` §6). The pixel is in the RENDER's coordinates, which at
    // `--scale 1.0` on a pinned view are the capture image's own.
    if let Some(px) = args.click {
        edits.set_click_params(click_params.expect("built alongside args.click"));
        edits.run_click(
            &trips_viewer::edit::cluster::ClickCamera::from_render_camera(&camera),
            px,
            &renderer,
        );
        edits.set_click_preview(true);
        edits.apply(&mut renderer)?;
        let found = edits.click_selection();
        eprintln!(
            "click ({}, {}): {} of {} points selected ({} candidates, {} seeded){}",
            px.0,
            px.1,
            found.point_ids.len(),
            renderer.num_points(),
            found.n_candidates,
            found.n_seed,
            found
                .warning
                .as_ref()
                .map_or_else(String::new, |w| format!(" -- {w}"))
        );
    }

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
        || args.dump_weights.is_some()
        || args.dump_shade.is_some()
        || args.dump_click.is_some()
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

    if let Some(out) = &args.dump_weights {
        let edits = args
            .edits
            .clone()
            .unwrap_or_else(|| dir.join(trips_viewer::edit::EDITS_FILENAME));
        return dump_weights(&bundle, &edits, out);
    }

    if let Some(out) = &args.dump_shade {
        return dump_shade(&bundle, &args, out);
    }

    if let Some(out) = &args.dump_click {
        return dump_click(&bundle, &args, out);
    }

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
    let edits = args.edits.clone();
    let edit_mode = args.edit;
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
            let mut app = app::ViewerApp::new(cc, bundle, settings, &splat_args, edits.as_deref())
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
            app.set_mode(mode);
            app.set_navigation(navigation);
            app.set_edit_mode(edit_mode);
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
