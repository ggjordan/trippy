//! `trips-web` — the TRIPS viewer in a browser, on WebGPU, from wasm32.
//!
//! Module: `trips_web` (cdylib)
//! Purpose: run trippy's real forward pass — `brush-pyramid`'s CubeCL
//!     rasteriser and `brush-unet`'s Burn decoder, the identical crates the
//!     native `trips-viewer` binary uses — in a page, and present the result
//!     on a `<canvas>` through WebGPU. The JavaScript in `web/` owns the
//!     event loop, the fps readout and the screenshot beacon; everything
//!     below the canvas is this module.
//! Invariants:
//!     - **One viewer per page.** The state lives in a `thread_local`, not in
//!       a `#[wasm_bindgen]` struct, because every entry point that touches
//!       the GPU is `async` and a borrow may not be held across an `.await`.
//!       Each function borrows, copies what it needs, drops the borrow, and
//!       only then awaits — see [`with_state`].
//!     - **WebGPU only**, no WebGL fallback: see [`crate::gpu`].
//!     - The camera, bundle format, render modes and performance levers are
//!       `trips_viewer`'s, not re-implemented. A frame rendered here differs
//!       from `trips-viewer --screenshot` only in the device it ran on and in
//!       the defaults the launcher picks (`--half-net --scale 0.75`), which
//!       is what makes the screenshot PSNR check meaningful.
//!     - `std::time::Instant` is never used (it panics on wasm32) and
//!       `brush_pyramid::gpu::block_on` is not even compiled for wasm. Frame
//!       timing comes from `requestAnimationFrame` on the JS side.
//! Units: `scale` is a fraction of the canvas; sizes are device pixels; the
//!     returned `frame_ms` fields are milliseconds.
//! Related docs: `docs/WEB_VIEWER.md`; `docs/USER_GUIDE.md`;
//!     `docs/decisions/ADR-0006-viewer-integration.md`; `web/index.html`.

// Everything here is browser code: `web-sys`, `js-sys`, `wasm-bindgen` and
// `console_error_panic_hook` are `cfg(target_family = "wasm")` dependencies, so
// on the host this crate compiles to nothing at all. That is deliberate --
// `scripts/build.sh` and `scripts/test.sh` stay a seconds-long native check and
// never try to resolve a browser API surface they cannot exercise.
#![cfg(target_family = "wasm")]

mod blit;
mod gpu;

use std::cell::RefCell;
use std::rc::Rc;

use brush_pyramid::png;
use brush_pyramid::scene::Camera;
use trips_viewer::bundle::{Bundle, BundleView, Manifest};
use trips_viewer::camera::{Controller, Mode};
use trips_viewer::renderer::{Renderer, Settings, ViewMode};
use wasm_bindgen::prelude::*;

use crate::blit::Blit;
use crate::gpu::Gpu;

/// Render-scale presets the `-`/`=` keys step between, as in the native app.
const SCALE_STEPS: [f32; 4] = [0.5, 0.75, 0.9, 1.0];

/// The default the launcher ships, matching `scripts/open_mac_viewer.sh`'s
/// `--half-net --scale 0.75`. See `docs/LIMITATIONS.md` for what it costs.
const DEFAULT_SCALE: f32 = 0.75;

thread_local! {
    /// The page's one viewer. See the module invariants for why it is not a
    /// `#[wasm_bindgen]` struct.
    static VIEWER: RefCell<Option<Viewer>> = const { RefCell::new(None) };
}

/// Everything the page needs between frames.
struct Viewer {
    /// Behind an `Rc` so a frame can hold it across the render `.await`
    /// without keeping the `RefCell` borrowed.
    renderer: Rc<Renderer>,
    controller: Controller,
    views: Vec<BundleView>,
    /// Position in `views` of the view the camera is referenced to.
    view_pos: usize,
    settings: Settings,
    mode: ViewMode,
    gpu: Gpu,
    blit: Blit,
    scene_name: String,
    num_points: usize,
    /// Set once, the first time an f16 render fails and the viewer falls back
    /// to f32. `None` means f16 is working (or was never asked for).
    half_net_fallback: Option<String>,
    frames: u64,
}

impl Viewer {
    /// The dataset view the camera is referenced to.
    fn reference(&self) -> &BundleView {
        &self.views[self.view_pos]
    }

    /// The render target size for the current canvas and render scale.
    fn render_size(&self) -> (usize, usize) {
        let scale = self.settings.render_scale.clamp(0.1, 1.0);
        let width = ((self.gpu.config.width as f32 * scale).round() as usize).max(16);
        let height = ((self.gpu.config.height as f32 * scale).round() as usize).max(16);
        (width, height)
    }

    /// This frame's camera.
    fn camera(&self) -> Camera {
        let (width, height) = self.render_size();
        self.controller
            .render_camera(width, height, self.reference())
    }
}

/// Borrow the viewer for a **synchronous** step.
///
/// # Errors
/// Returns `Err` if [`start`] has not run or has failed.
fn with_state<T>(f: impl FnOnce(&mut Viewer) -> Result<T, String>) -> Result<T, String> {
    VIEWER.with_borrow_mut(|slot| {
        let viewer = slot
            .as_mut()
            .ok_or_else(|| "the viewer has not been started".to_owned())?;
        f(viewer)
    })
}

/// `String` errors become JS exceptions with the message intact.
fn js(e: String) -> JsValue {
    JsValue::from_str(&e)
}

/// What [`start`] accepts, as a JSON object from the page's query string.
#[derive(Debug, Clone, Copy)]
struct Options {
    scale: f32,
    half_net: bool,
    mode: ViewMode,
    /// A **dataset** view index (`views[i].index`), not a position.
    view: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            scale: DEFAULT_SCALE,
            half_net: true,
            mode: ViewMode::Network,
            view: None,
        }
    }
}

impl Options {
    /// Parse the options object the page passes in.
    ///
    /// Unknown keys are ignored; an unknown `mode` is an error, because
    /// silently rendering the wrong view mode is exactly the kind of thing
    /// that makes a screenshot check meaningless.
    fn parse(json: &str) -> Result<Self, String> {
        let mut out = Self::default();
        if json.trim().is_empty() {
            return Ok(out);
        }
        let value: serde_json::Value =
            serde_json::from_str(json).map_err(|e| format!("options: {e}"))?;
        if let Some(scale) = value.get("scale").and_then(serde_json::Value::as_f64) {
            out.scale = (scale as f32).clamp(0.1, 1.0);
        }
        if let Some(half) = value.get("halfNet").and_then(serde_json::Value::as_bool) {
            out.half_net = half;
        }
        if let Some(view) = value.get("view").and_then(serde_json::Value::as_u64) {
            out.view = Some(view as usize);
        }
        if let Some(mode) = value.get("mode").and_then(serde_json::Value::as_str) {
            out.mode = parse_mode(mode)?;
        }
        Ok(out)
    }
}

/// `network` / `raw` / `coverage`, the same three the native viewer's `--mode`
/// and its `V` key cycle through.
fn parse_mode(name: &str) -> Result<ViewMode, String> {
    match name {
        "network" => Ok(ViewMode::Network),
        "raw" => Ok(ViewMode::RawLevel0),
        "coverage" => Ok(ViewMode::Coverage),
        other => Err(format!(
            "unknown mode {other:?}; expected network, raw or coverage"
        )),
    }
}

/// Fetch a URL and return the whole body.
///
/// Uses `web_sys::window().fetch_with_str`, i.e. the browser's own loader —
/// no HTTP client is linked into the wasm. Everything is same-origin on
/// 127.0.0.1.
///
/// # Arguments
/// - `url`: absolute or page-relative.
///
/// # Errors
/// Returns `Err` on a network failure or any non-2xx status.
async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
    let window = web_sys::window().ok_or_else(|| "no window".to_owned())?;
    let response = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|e| format!("fetch {url}: {e:?}"))?;
    let response: web_sys::Response = response
        .dyn_into()
        .map_err(|_| format!("fetch {url}: not a Response"))?;
    if !response.ok() {
        return Err(format!("fetch {url}: HTTP {}", response.status()));
    }
    let buffer = wasm_bindgen_futures::JsFuture::from(
        response
            .array_buffer()
            .map_err(|e| format!("fetch {url}: {e:?}"))?,
    )
    .await
    .map_err(|e| format!("fetch {url}: {e:?}"))?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

/// Fetch a URL as text.
async fn fetch_text(url: &str) -> Result<String, String> {
    let bytes = fetch_bytes(url).await?;
    String::from_utf8(bytes).map_err(|e| format!("fetch {url}: not UTF-8: {e}"))
}

/// Install the panic hook. Call once, before anything else.
///
/// Without it a Rust panic surfaces in the console as `unreachable executed`
/// with no message, which would make every blocker in `docs/WEB_VIEWER.md`
/// twice as expensive to find.
#[wasm_bindgen]
pub fn install_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Open a WebGPU device on `canvas`, fetch the bundle at `bundle_url`, and
/// get ready to render. Returns a JSON status string.
///
/// # Arguments
/// - `canvas`: the `<canvas>` to present to. Its `width`/`height` attributes
///   are the render size; CSS size is irrelevant.
/// - `bundle_url`: a directory URL holding `bundle.json` (trailing slash
///   optional). The two files the manifest names are fetched from beside it.
/// - `options_json`: `{"scale":0.75,"halfNet":true,"mode":"network",
///   "view":8}`; every key optional.
///
/// # Errors
/// Returns a JS exception carrying the exact failure — no WebGPU adapter, a
/// bundle that would not load, a device that refuses the pipeline — rather
/// than leaving a blank canvas.
#[wasm_bindgen]
pub async fn start(
    canvas: web_sys::HtmlCanvasElement,
    bundle_url: String,
    options_json: String,
) -> Result<String, JsValue> {
    let options = Options::parse(&options_json).map_err(js)?;
    let base = if bundle_url.ends_with('/') {
        bundle_url.clone()
    } else {
        format!("{bundle_url}/")
    };

    let manifest_url = format!("{base}bundle.json");
    let manifest_text = fetch_text(&manifest_url).await.map_err(js)?;
    let manifest: Manifest = Bundle::parse_manifest(&manifest_text, &manifest_url).map_err(js)?;

    // Only the two files the manifest names, and only after it parsed --
    // `points.npz` is ~80 MB and there is no point spending it on a bundle
    // this build cannot read anyway.
    let points_url = format!("{base}{}", manifest.points);
    let weights_url = format!("{base}{}", manifest.weights);
    let points_bytes = fetch_bytes(&points_url).await.map_err(js)?;
    let weight_bytes = fetch_bytes(&weights_url).await.map_err(js)?;

    let gpu = Gpu::create(canvas).await.map_err(js)?;

    let bundle = Bundle::from_parts(
        std::path::PathBuf::from(&base),
        manifest,
        &points_bytes,
        &weight_bytes,
        &base,
    )
    .map_err(js)?;
    // The decompressed copies are inside `bundle` now; 160 MB of wasm heap is
    // worth reclaiming before any GPU allocation happens.
    drop(points_bytes);
    drop(weight_bytes);

    let views = bundle.manifest.views.clone();
    let view_pos = match options.view {
        None => bundle
            .manifest
            .default_view
            .min(views.len().saturating_sub(1)),
        Some(wanted) => views
            .iter()
            .position(|v| v.index == wanted)
            .ok_or_else(|| js(format!("no view with index {wanted} in this bundle")))?,
    };
    let up = bundle.manifest.up;
    let scene_name = bundle.manifest.name.clone();
    let num_points = bundle.points.len();

    let renderer = Renderer::new(bundle, gpu.burn.clone()).map_err(js)?;
    // Fly speed comes from `bundle::SceneScale` -- the box the capture cameras
    // occupy -- and never from the point cloud, whose far-field environment
    // sphere is thousands of units across (see the 2026-09-06 entry in
    // research/trips-metal.md). `Mode::Free` because this page's controls are
    // the free-fly verbs: `look`, `fly`, `adjust_speed`.
    let mut controller = Controller::new(&views, view_pos, up);
    controller.set_mode(Mode::Free);

    let blit = Blit::new(&gpu.device, gpu.config.format);

    let settings = Settings {
        render_scale: options.scale,
        half_net: options.half_net,
        ..Settings::default()
    };
    // An adapter without SHADER_F16 cannot run the f16 decoder at all; say so
    // up front instead of failing on the first frame.
    let half_net_fallback = match (options.half_net, gpu.has_f16, renderer.half_net_error()) {
        (true, false, _) => Some("this WebGPU adapter does not report SHADER_F16".to_owned()),
        (true, _, Some(e)) => Some(format!("the f16 network could not be built: {e}")),
        _ => None,
    };

    // No substitution any more: `network` is the default and it renders.
    // v0.5.0 swapped in `raw level-0` here because the U-Net view trapped on
    // wasm; the trap was CubeCL's autotune roofline probe, and
    // `Gpu::create` now turns that probe off. See `docs/WEB_VIEWER.md`.
    let mode = options.mode;

    let viewer = Viewer {
        renderer: Rc::new(renderer),
        controller,
        views,
        view_pos,
        settings: Settings {
            half_net: settings.half_net && half_net_fallback.is_none(),
            ..settings
        },
        mode,
        gpu,
        blit,
        scene_name,
        num_points,
        half_net_fallback,
        frames: 0,
    };
    let status = viewer_status(&viewer);
    VIEWER.set(Some(viewer));
    Ok(status)
}

/// Render one frame and present it. Returns a small JSON object.
///
/// The page calls this from `requestAnimationFrame` and awaits it, so the
/// frame interval it measures is an honest end-to-end number: nothing is left
/// queued behind the `await` except the presentation itself.
///
/// # Errors
/// Returns a JS exception on any render or present failure.
#[wasm_bindgen]
pub async fn frame() -> Result<String, JsValue> {
    let (renderer, camera, frame_index, mode, settings) = with_state(|v| {
        Ok((
            v.renderer.clone(),
            v.camera(),
            v.reference().index,
            v.mode,
            v.settings,
        ))
    })
    .map_err(js)?;

    let rendered = match renderer.render(&camera, frame_index, mode, &settings).await {
        Ok(frame) => frame,
        Err(first) if settings.half_net => {
            // The documented web fallback: an adapter can advertise SHADER_F16
            // and still fail to compile an f16 convolution. Drop to f32 once,
            // remember why, and never try again this session.
            let retry = Settings {
                half_net: false,
                ..settings
            };
            let frame = renderer
                .render(&camera, frame_index, mode, &retry)
                .await
                .map_err(|second| {
                    js(format!(
                        "render failed in f16 ({first}) and again in f32 ({second})"
                    ))
                })?;
            with_state(|v| {
                v.settings.half_net = false;
                v.half_net_fallback = Some(format!("the f16 network failed at render time: {first}"));
                Ok(())
            })
            .map_err(js)?;
            frame
        }
        Err(e) => return Err(js(e)),
    };

    with_state(|v| {
        // `CurrentSurfaceTexture` is not `Debug` (a `SurfaceTexture` is not),
        // so each non-drawable status is named explicitly.
        let texture = match v.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            // A canvas resized between frames, or a tab that is not visible:
            // report it and skip, rather than tearing the page down.
            wgpu::CurrentSurfaceTexture::Timeout => {
                return Err("surface not ready: timeout".to_owned())
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                return Err("surface not ready: occluded".to_owned())
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                return Err("surface not ready: outdated (canvas resized?)".to_owned())
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                return Err("surface not ready: lost".to_owned())
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err("surface not ready: validation error".to_owned())
            }
        };
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        v.blit
            .draw(&v.gpu.device, &v.gpu.queue, &view, &rendered)?;
        v.gpu.queue.present(texture);
        v.frames += 1;
        Ok(format!(
            r#"{{"frames":{},"width":{},"height":{},"fragments":{},"mode":"{}","halfNet":{}}}"#,
            v.frames,
            rendered.stats.width,
            rendered.stats.height,
            rendered.stats.fragment_slots,
            rendered.mode.label(),
            v.settings.half_net
        ))
    })
    .map_err(js)
}

/// Render one frame **to host memory** and encode it as a PNG.
///
/// This is the web twin of `trips-viewer --screenshot`, and exists because a
/// `canvas.toBlob()` on a WebGPU canvas is not guaranteed to capture anything
/// once the frame has been presented. Having both means the pixel check
/// survives a browser where `toBlob` comes back blank — and when both work,
/// they cross-check each other.
///
/// # Errors
/// Returns a JS exception on a render, readback or encode failure.
#[wasm_bindgen]
pub async fn screenshot_png() -> Result<Vec<u8>, JsValue> {
    let (renderer, camera, frame_index, settings) = with_state(|v| {
        Ok((
            v.renderer.clone(),
            v.camera(),
            v.reference().index,
            v.settings,
        ))
    })
    .map_err(js)?;

    let (data, channels, height, width) = renderer
        .render_to_host(&camera, frame_index, &settings)
        .await
        .map_err(js)?;
    // `scale = 1.0`: the network's output is already display-referred, and
    // `feature_to_rgb8` is byte for byte what the native `--screenshot` path
    // runs, which is the whole point of comparing the two PNGs.
    let pixels = png::feature_to_rgb8(&data, channels, height, width, 1.0).map_err(js)?;
    png::encode_rgb8(&pixels, width, height).map_err(js)
}

/// Rotate the camera by a mouse drag, in CSS pixels.
#[wasm_bindgen]
pub fn look(dx: f32, dy: f32) -> Result<(), JsValue> {
    with_state(|v| {
        v.controller.look(dx, dy);
        Ok(())
    })
    .map_err(js)
}

/// Translate along the camera basis; each axis in `[-1, 1]`, `dt` in seconds.
#[wasm_bindgen]
pub fn fly(forward: f32, right: f32, up: f32, dt: f32) -> Result<(), JsValue> {
    with_state(|v| {
        v.controller.fly(forward, right, up, dt);
        Ok(())
    })
    .map_err(js)
}

/// Scale the fly speed by `notches` scroll steps.
#[wasm_bindgen]
pub fn adjust_speed(notches: f32) -> Result<(), JsValue> {
    with_state(|v| {
        v.controller.adjust_speed(notches);
        Ok(())
    })
    .map_err(js)
}

/// Snap the camera back to its dataset view, pinning it again.
#[wasm_bindgen]
pub fn snap_to_view() -> Result<(), JsValue> {
    with_state(|v| {
        v.controller.snap_to_position(&v.views, v.view_pos);
        Ok(())
    })
    .map_err(js)
}

/// Cycle network -> raw level-0 -> coverage, as the native viewer's `V` key
/// does. Returns the new mode's label.
#[wasm_bindgen]
pub fn cycle_mode() -> Result<String, JsValue> {
    with_state(|v| {
        v.mode = v.mode.next();
        Ok(v.mode.label().to_owned())
    })
    .map_err(js)
}

/// Set the view mode by name (`network`, `raw`, `coverage`).
#[wasm_bindgen]
pub fn set_mode(name: &str) -> Result<(), JsValue> {
    let mode = parse_mode(name).map_err(js)?;
    with_state(|v| {
        v.mode = mode;
        Ok(())
    })
    .map_err(js)
}

/// Step the render scale through [`SCALE_STEPS`]; returns the new value.
///
/// # Arguments
/// - `direction`: `+1` to go up a step, `-1` to go down.
#[wasm_bindgen]
pub fn step_scale(direction: i32) -> Result<f32, JsValue> {
    with_state(|v| {
        let current = v.settings.render_scale;
        let index = SCALE_STEPS
            .iter()
            .position(|s| (s - current).abs() < 1e-3)
            .unwrap_or(1);
        let next = (index as i32 + direction.signum())
            .clamp(0, SCALE_STEPS.len() as i32 - 1) as usize;
        v.settings.render_scale = SCALE_STEPS[next];
        Ok(v.settings.render_scale)
    })
    .map_err(js)
}

/// Ask for the f16 decoder (or not). Returns whether it is actually in use —
/// which is `false` if this adapter could not build it, never a silent yes.
#[wasm_bindgen]
pub fn set_half_net(on: bool) -> Result<bool, JsValue> {
    with_state(|v| {
        v.settings.half_net = on && v.half_net_fallback.is_none();
        Ok(v.settings.half_net)
    })
    .map_err(js)
}

/// Tell the viewer the canvas' backing store changed size.
#[wasm_bindgen]
pub fn resize(width: u32, height: u32) -> Result<(), JsValue> {
    with_state(|v| {
        v.gpu.resize(width, height);
        Ok(())
    })
    .map_err(js)
}

/// A JSON status blob: adapter, scene, current settings, and any fallback
/// that had to be taken. This is what the page's beacon posts.
#[wasm_bindgen]
pub fn status() -> Result<String, JsValue> {
    with_state(|v| Ok(viewer_status(v))).map_err(js)
}

fn viewer_status(v: &Viewer) -> String {
    let (width, height) = v.render_size();
    let info = &v.gpu.adapter_info;
    let fallback = match &v.half_net_fallback {
        None => "null".to_owned(),
        Some(reason) => serde_json::Value::String(reason.clone()).to_string(),
    };
    format!(
        r#"{{"scene":{},"points":{},"adapter":{{"name":{},"vendor":{},"backend":"{:?}","device_type":"{:?}"}},"canvas":[{},{}],"render":[{},{}],"scale":{},"halfNet":{},"shaderF16":{},"subgroups":{},"halfNetFallback":{},"mode":"{}","view":{},"pinned":{},"frames":{}}}"#,
        serde_json::Value::String(v.scene_name.clone()),
        v.num_points,
        serde_json::Value::String(info.name.clone()),
        info.vendor,
        info.backend,
        info.device_type,
        v.gpu.config.width,
        v.gpu.config.height,
        width,
        height,
        v.settings.render_scale,
        v.settings.half_net,
        v.gpu.has_f16,
        v.gpu.has_subgroups,
        fallback,
        v.mode.label(),
        v.reference().index,
        v.controller.is_pinned(),
        v.frames,
    )
}
