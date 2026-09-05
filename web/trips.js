// trips.js — the web viewer's whole JavaScript half.
//
// Module: web/ (loaded as an ES module by index.html)
// Purpose: start `trips-web`'s wasm module on a canvas, run the frame loop,
//     translate mouse/keyboard into camera calls, show the fps readout, and
//     implement the `?screenshot=1` verification mode.
// Invariants:
//   - The frame loop `await`s `frame()` before asking for the next
//     requestAnimationFrame. The fps number is therefore an honest end-to-end
//     rate (render + present), not the display's refresh rate: nothing is left
//     queued behind the await except presentation itself.
//   - Nothing leaves the machine. Every fetch is same-origin on 127.0.0.1;
//     the two beacon POSTs go to this same origin and simply fail (harmlessly,
//     caught) when the plain `python3 -m http.server` behind
//     OPEN_TRIPS_WEB_*.command answers 501 to POST.
//   - The frame is honest about which sort ran. The packed 32-bit key is a
//     real approximation (36.9 dB against the native reference, versus
//     104.5 dB for the exact two-pass sort) and it is OFF by default, exactly
//     as it is natively, even though it is worth 11-45 % more frames. `P`
//     toggles it and the HUD always names the sort in use.
//   - `?screenshot=1` sizes the canvas to exactly `SHOT_SIZE` and pins the
//     camera to the reference view, so the PNG it posts is directly
//     comparable with `trips-viewer --screenshot --half-net --scale 0.75`.
//     Changing that size silently would invalidate the PSNR check, so it is a
//     named constant with this comment on it.
// Query parameters:
//   ?bundle=<url>     bundle directory (default ./bundle)
//   ?scale=<f>        render scale, 0.1..1 (default 0.75, the shipped setting)
//   ?half=0|1         f16 decoder (default 1)
//   ?packed=0|1       packed 32-bit sort key (default 0; `P` toggles it live)
//   ?mode=network|raw|coverage
//   ?view=<n>         dataset view index
//   ?screenshot=1     run the verification sequence, then stop
//   ?seconds=<f>      how long the fps window is in screenshot mode (default 5)
// Related docs: docs/WEB_VIEWER.md; rust/crates/trips-web/src/lib.rs.

import init, * as trips from "./pkg/trips_web.js";

const params = new URLSearchParams(window.location.search);
const canvas = document.getElementById("canvas");
const hud = document.getElementById("hud");
const errorBox = document.getElementById("error");

/** Where a `?screenshot=1` run posts its JSON status. */
const BEACON_URL = "/__trips_beacon";
/**
 * Progress trace. A plain GET, because a plain `python3 -m http.server` logs
 * every request line even when it 404s -- so the access log of ANY server
 * behind the page becomes a stage-by-stage trace of how far it got. This is
 * the only way to see inside a Safari tab from a non-interactive session:
 * there is no headless mode and safaridriver needs an interactive sudo.
 */
const STAGE_URL = "/__trips_stage";

/** Tracing is on for `?screenshot=1` and `?trace=1`, silent otherwise. */
const tracing =
  params.get("trace") === "1" || params.get("screenshot") === "1";

function mark(stage, detail = "") {
  if (!tracing) return;
  // 400 chars: an http.server log line is not a transcript, and a wgpu
  // validation message's first paragraph is the part that identifies it.
  const query = `?s=${encodeURIComponent(stage)}&d=${encodeURIComponent(String(detail).slice(0, 400))}`;
  // Fire and forget; a 404 is fine, the log line is the point.
  fetch(STAGE_URL + query).catch(() => {});
}

/**
 * Work around a wgpu-on-WebGPU bug that is fatal on the first kernel launch.
 *
 * `wgpu`'s WebGPU backend reads `GPUDevice.popErrorScope()` into a
 * `js_sys::JsOption<GpuError>`, whose `into_option()` returns `None` only for
 * `undefined`. But the WebGPU specification says a CLEAN pop resolves with
 * `null`, and Chrome and Safari both do exactly that. So `null` is taken for a
 * real error, handed to `wgpu::backend::webgpu::Error::from_js`, matches
 * neither `GPUValidationError` nor `GPUOutOfMemoryError`, and reaches its
 * `panic!("Unexpected error")` -- taking the page down. (`JsNullable<T>`, the
 * null-aware sibling in the same wasm-bindgen module, is what that code wants.)
 *
 * CubeCL wraps every kernel launch and pipeline creation in an error scope
 * (`cubecl-wgpu` `backend/base.rs`, `compute/stream.rs`), so the very first
 * TRIPS frame trips it, in every browser, every time.
 *
 * This shim rejects the pop instead of resolving it: wgpu maps a rejected
 * promise to "no error" and never calls `from_js`. Errors are NOT hidden --
 * every one is logged to the console and to the trace first. The cost is that
 * wgpu itself stops seeing validation errors, which for a viewer means a bad
 * frame shows as bad pixels rather than as an exception.
 *
 * Delete this the day the pinned `ArthurBrussee/wgpu` fork carries the
 * `JsOption` -> `JsNullable` fix. `docs/WEB_VIEWER.md` records the diagnosis.
 */
/** How many shader modules needed the missing `enable subgroups;` directive. */
let subgroupShaderPatches = 0;
/**
 * Whether this adapter advertises WebGPU `subgroups`.
 *
 * Set by [`subgroupSupportError`] before the wasm starts. The `enable subgroups;`
 * injection below is only legal on an adapter that has the feature; on one
 * that does not (Safari 26.6.2) the directive turns a precise error --
 * "cannot call built-in function 'subgroupAdd' without extension 'subgroups'"
 * -- into a misleading one, "Expected 'f16'", which cost this project a whole
 * wrong diagnosis. Measured, see docs/WEB_VIEWER.md.
 */
let adapterHasSubgroups = false;
/**
 * Every WebGPU error seen this session.
 *
 * This list is shown ON SCREEN, in red, and it is not optional. Neutralising
 * wgpu's fatal error path (below) means a kernel that fails to compile no
 * longer throws — it just produces a wrong picture. That happened for real:
 * Safari 26.6.2 cannot compile any of `brush-sort`'s four radix kernels (it has
 * no WebGPU subgroups) and then draws confident-looking stripes instead of the
 * horse. `subgroupSupportError` now catches that case before the first frame,
 * but it can only catch the one it knows about. A viewer that shows an invented
 * image without saying so is the one thing this project must not do.
 */
const gpuErrors = [];

function installWebGpuWorkaround() {
  if (!window.GPUDevice || !window.GPUAdapter) return;

  const pop = GPUDevice.prototype.popErrorScope;
  GPUDevice.prototype.popErrorScope = async function popErrorScopeShim() {
    const error = await pop.call(this);
    if (error) {
      const text = `${error.constructor.name}: ${error.message}`;
      console.error("[trips-web] WebGPU error scope:", text);
      mark("gpu-scope", text);
      if (gpuErrors.length < 8) gpuErrors.push(text);
    }
    throw new Error("trips-web: error scope neutralised (see docs/WEB_VIEWER.md)");
  };

  // Uncaptured errors do not go through an error scope; report those too, and
  // report which optional features the browser actually granted -- `subgroups`
  // and `shader-f16` are the two the TRIPS pipeline's fate turns on.
  // Second workaround, and the one the TRIPS pipeline actually needs.
  //
  // `brush-sort`'s radix passes call CubeCL's `plane_sum` /
  // `plane_inclusive_sum`, which its WGSL backend translates to `subgroupAdd`
  // / `subgroupInclusiveAdd`. WGSL requires an `enable subgroups;` directive
  // before either can be called; CubeCL never emits one (its MSL and SPIR-V
  // backends need no directive, which is how this went unnoticed on the
  // fork's Metal-first path). So every one of `sort_reduce_kernel`,
  // `sort_scan_kernel`, `sort_scan_add_kernel` and `sort_scatter_kernel`
  // fails to compile with "cannot call built-in function 'subgroupAdd'
  // without extension 'subgroups'", the pipelines come out invalid, and no
  // frame is ever produced. Masking `Features::SUBGROUP` off the device was
  // tried and does NOT help -- CubeCL emits the calls regardless.
  //
  // Prepending the directive is exactly the missing line, and it is only
  // legal because `crate::gpu` requests `Features::SUBGROUP`. Browsers whose
  // WGSL front end does not implement subgroups at all (Safari 26.6.2) are
  // not rescued by it; see docs/WEB_VIEWER.md.
  const createShaderModule = GPUDevice.prototype.createShaderModule;
  GPUDevice.prototype.createShaderModule = function createShaderModuleShim(descriptor) {
    const code = descriptor?.code ?? "";
    const usesSubgroups = /\bsubgroup[A-Z][A-Za-z]*\s*\(|@builtin\(subgroup_/.test(code);
    if (!usesSubgroups || !adapterHasSubgroups || /^\s*enable\s+subgroups\s*;/m.test(code)) {
      return createShaderModule.call(this, descriptor);
    }
    subgroupShaderPatches += 1;
    return createShaderModule.call(this, { ...descriptor, code: `enable subgroups;\n${code}` });
  };

  const requestDevice = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function requestDeviceShim(...args) {
    const device = await requestDevice.apply(this, args);
    mark("device", `granted=[${[...device.features].join(" ")}]`);
    device.addEventListener("uncapturederror", (event) => {
      const text = `${event.error?.constructor?.name}: ${event.error?.message}`;
      console.error("[trips-web] WebGPU uncaptured error:", text);
      mark("gpu-uncaptured", text);
      if (gpuErrors.length < 8) gpuErrors.push(text);
    });
    device.lost?.then((info) => mark("gpu-lost", `${info.reason}: ${info.message}`));
    return device;
  };
}

// Mirror console.error/warn into the trace. wgpu's default uncaptured-error
// handler is fatal -- it panics -- and `console_error_panic_hook` prints the
// message to console.error, which a non-interactive session cannot read. This
// is what turns "RuntimeError: unreachable" into an actual diagnosis.
// Deliberately additive: the real console still gets everything.
for (const level of ["error", "warn"]) {
  const original = console[level].bind(console);
  console[level] = (...args) => {
    original(...args);
    mark(`console.${level}`, args.map(String).join(" "));
  };
}
/** Where a `?screenshot=1` run posts its PNGs. */
const SHOT_URL = "/__trips_shot";

// The reference render `output/brush/viewer/halfnet_s75.png` is 1440x810: the
// horse's 1920x1080 view 8 at --scale 0.75. Rendering the canvas at exactly
// that size with scale 1.0 reproduces the identical camera (render_camera
// scales fx by width/reference.width, which is what --scale does natively)
// and makes the blit a 1:1 copy rather than an upsample -- so the posted PNG
// can be compared pixel for pixel.
const SHOT_SIZE = { width: 1440, height: 810 };

/** Seconds of steady-state rendering the screenshot run measures fps over. */
const DEFAULT_FPS_WINDOW_S = 5;

/** Frames rendered before the fps window opens, to pay for shader compilation. */
const WARMUP_FRAMES = 3;

const screenshotMode = params.get("screenshot") === "1";
const fpsWindowSeconds = Number(params.get("seconds") ?? DEFAULT_FPS_WINDOW_S);

function fail(message) {
  mark("fail", String(message).slice(0, 200));
  errorBox.textContent = message;
  errorBox.style.display = "block";
  hud.style.display = "none";
  // Best effort: tell the local endpoint too, so a headless run records the
  // failure instead of timing out with no explanation.
  post(BEACON_URL, JSON.stringify({ ok: false, error: String(message) })).catch(() => {});
}

async function post(url, body, type = "application/json") {
  return fetch(url, { method: "POST", headers: { "Content-Type": type }, body });
}

/** Size the canvas backing store to its CSS box at the device pixel ratio. */
function fitCanvas() {
  if (screenshotMode) return false;
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(16, Math.round(canvas.clientWidth * dpr));
  const height = Math.max(16, Math.round(canvas.clientHeight * dpr));
  if (canvas.width === width && canvas.height === height) return false;
  canvas.width = width;
  canvas.height = height;
  return true;
}

// --- input ------------------------------------------------------------------
// Deliberately the native viewer's bindings (docs/USER_GUIDE.md): WASD to
// move, Q/E down/up, drag to look, scroll for fly speed, V to cycle modes,
// -/= for render scale, R to snap back to the dataset view.
const held = new Set();
let dragging = false;

function installInput() {
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointerup", (e) => {
    dragging = false;
    canvas.releasePointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (dragging) trips.look(e.movementX, e.movementY);
  });
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      trips.adjust_speed(-Math.sign(e.deltaY));
    },
    { passive: false },
  );
  window.addEventListener("keydown", (e) => {
    const key = e.key.toLowerCase();
    if (key === "v") {
      trips.cycle_mode();
    } else if (key === "-" || key === "_") {
      trips.step_scale(-1);
    } else if (key === "=" || key === "+") {
      trips.step_scale(1);
    } else if (key === "r") {
      trips.snap_to_view();
    } else if (key === "h") {
      trips.set_half_net(!lastStatus.halfNet);
    } else if (key === "p") {
      trips.set_packed_sort(!lastStatus.packedSort);
    } else {
      held.add(key);
      return;
    }
    e.preventDefault();
  });
  window.addEventListener("keyup", (e) => held.delete(e.key.toLowerCase()));
  window.addEventListener("blur", () => held.clear());
}

function applyMovement(dt) {
  const axis = (a, b) => (held.has(a) ? 1 : 0) - (held.has(b) ? 1 : 0);
  const forward = axis("w", "s");
  const right = axis("d", "a");
  const up = axis("e", "q");
  if (forward || right || up) trips.fly(forward, right, up, dt);
}

// --- frame loop -------------------------------------------------------------
let lastStatus = {};
let lastTimestamp = 0;
const intervals = [];
/** How many recent frame intervals the readout averages, as in the native app. */
const FPS_WINDOW = 30;

function recordInterval(ms) {
  intervals.push(ms);
  if (intervals.length > FPS_WINDOW) intervals.shift();
}

function meanFps() {
  if (intervals.length === 0) return null;
  const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
  return mean > 0 ? 1000 / mean : null;
}

function drawHud(frameInfo) {
  const fps = meanFps();
  const parts = [];
  parts.push(
    fps === null
      ? "measuring…"
      : `<b>${fps.toFixed(1)} fps</b>  (${(1000 / fps).toFixed(1)} ms)`,
  );
  parts.push(
    `${lastStatus.scene ?? "?"} — ${(lastStatus.points ?? 0).toLocaleString()} points`,
  );
  parts.push(
    `${frameInfo.width}x${frameInfo.height} @ scale ${lastStatus.scale ?? "?"}` +
      `  ·  ${frameInfo.mode}  ·  net ${frameInfo.halfNet ? "f16" : "f32"}` +
      `  ·  sort ${frameInfo.packedSort ? "packed (approx depth)" : "exact"}`,
  );
  parts.push(`WebGPU: ${lastStatus.adapter?.name ?? "?"} (${lastStatus.adapter?.backend ?? "?"})`);
  if (lastStatus.halfNetFallback) parts.push(`f32 fallback: ${lastStatus.halfNetFallback}`);
  if (gpuErrors.length > 0) {
    parts.push(
      `<span style="color:#ff8a8a">${gpuErrors.length} WebGPU error(s) — THIS IMAGE IS NOT TRUSTWORTHY:
${gpuErrors[0].slice(0, 300)}</span>`,
    );
  }
  hud.innerHTML =
    parts.join("\n") +
    `<div id="keys">WASD move · Q/E down/up · drag look · scroll speed
V modes · -/= scale · H f16 · P sort · R back to view</div>`;
}

async function renderOneFrame(timestamp) {
  if (fitCanvas()) trips.resize(canvas.width, canvas.height);
  const dt = lastTimestamp ? (timestamp - lastTimestamp) / 1000 : 0;
  if (dt > 0) applyMovement(Math.min(dt, 0.1));
  lastTimestamp = timestamp;
  const info = JSON.parse(await trips.frame());
  lastStatus = JSON.parse(trips.status());
  return info;
}

function interactiveLoop() {
  let previous = 0;
  const step = async (timestamp) => {
    try {
      const info = await renderOneFrame(timestamp);
      if (previous) recordInterval(timestamp - previous);
      previous = timestamp;
      drawHud(info);
      requestAnimationFrame(step);
    } catch (e) {
      fail(`render failed: ${e}`);
    }
  };
  requestAnimationFrame(step);
}

/**
 * The `?screenshot=1` verification sequence.
 *
 * Warm up, measure fps over a fixed window, capture the canvas with
 * `toBlob()` AND (as a fallback that cannot come back blank) a GPU readback
 * encoded to PNG inside the wasm, then post everything to the local endpoint.
 * Deliberately bounded in time: a browser rendering for a few seconds on
 * loopback is a functional check, not a benchmark (AGENTS.md's GPU rule).
 */
async function screenshotRun() {
  const started = performance.now();
  let frames = 0;
  let info = null;

  mark("warmup-start");
  for (let i = 0; i < WARMUP_FRAMES; i += 1) {
    info = await renderOneFrame(performance.now());
    mark("warmup", `${i + 1}/${WARMUP_FRAMES} ${Math.round(performance.now() - started)}ms`);
  }

  const windowStart = performance.now();
  let previous = windowStart;
  while (performance.now() - windowStart < fpsWindowSeconds * 1000) {
    const now = performance.now();
    info = await renderOneFrame(now);
    recordInterval(performance.now() - previous);
    previous = performance.now();
    frames += 1;
  }
  const elapsedMs = performance.now() - windowStart;
  const fps = frames / (elapsedMs / 1000);
  drawHud(info);
  mark("fps", `${fps.toFixed(2)} over ${frames} frames`);

  // 1. The canvas as the compositor has it. This is the real pixel check --
  //    it goes through presentation, so it proves what is on screen.
  const canvasPng = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  mark("toBlob", String(canvasPng ? canvasPng.size : 0));
  if (canvasPng && canvasPng.size > 0) {
    await post(`${SHOT_URL}?src=canvas`, canvasPng, "image/png").catch(() => {});
  }

  // 2. The same frame read back off the GPU and encoded by the same
  //    `png::feature_to_rgb8` the native --screenshot uses. Belt and braces:
  //    toBlob on a WebGPU canvas is not guaranteed to capture a presented
  //    frame, and a blank capture would look like a rendering failure.
  //    It runs the U-Net, which is what makes it directly comparable with the
  //    native `trips-viewer --screenshot --half-net --scale 0.75` reference.
  let readback = null;
  try {
    readback = await trips.screenshot_png();
    mark("readback", String(readback.length));
    await post(`${SHOT_URL}?src=readback`, new Blob([readback], { type: "image/png" }), "image/png")
      .catch(() => {});
  } catch (e) {
    mark("readback-skipped", String(e));
  }

  const status = JSON.parse(trips.status());
  await post(
    BEACON_URL,
    JSON.stringify({
      ok: true,
      userAgent: navigator.userAgent,
      hasWebGpu: Boolean(navigator.gpu),
      status,
      frames,
      elapsedMs,
      fps,
      fpsWindowSeconds,
      canvasPngBytes: canvasPng ? canvasPng.size : 0,
      readbackPngBytes: readback ? readback.length : 0,
      canvas: { width: canvas.width, height: canvas.height },
      packedSort: status.packedSort,
      adapterHasSubgroups,
      subgroupShaderPatches,
      gpuErrors,
    }),
  ).catch(() => {});
  mark("done");
}

// Anything that escapes the promise chain -- a wgpu validation error surfaced
// as an unhandled rejection, say -- must reach the trace, or a stall looks
// identical to a hang.
window.addEventListener("unhandledrejection", (e) =>
  mark("unhandled", String(e.reason).slice(0, 200)),
);
window.addEventListener("error", (e) => mark("jserror", String(e.message).slice(0, 200)));

/**
 * Refuse to render on a WebGPU implementation that cannot compile the sort.
 *
 * This replaces a user-agent sniff, and it replaces a WRONG diagnosis. v0.5.0
 * said Safari failed on f16 because the error it reports is
 *
 *     GPUValidationError: 1 error generated while compiling the shader:
 *     1:0: Expected 'f16'
 *
 * It is not f16. Safari 26.6.2 grants `shader-f16`, and the only two shaders
 * in the pipeline that mention f16 at all (`cast_element_i_f32_o_f16_n_*`,
 * which carry their own `enable f16;`) compile there without complaint. The
 * message is Safari's WGSL parser saying that `f16` is the only extension name
 * its `enable` directive accepts: `enable subgroups;`, `enable subgroups_basic;`
 * and even `enable f16, subgroups;` all fail at 1:0 with that same text.
 * Position 1:0 is exactly where `installWebGpuWorkaround` prepends the
 * directive. Measured 2026-09-06 with a shader-compile-only probe in both
 * browsers; docs/WEB_VIEWER.md has the table.
 *
 * The real gate is subgroups, in all three of its forms: Safari lists no
 * `subgroups` adapter feature, rejects the `enable` directive, and rejects
 * `@builtin(subgroup_invocation_id)` ("9:66: Unknown builtin value"). And
 * `brush-sort`'s four radix kernels call CubeCL's `plane_sum` /
 * `plane_inclusive_sum` unconditionally, which its WGSL backend lowers to
 * `subgroupAdd` / `subgroupInclusiveAdd`. So `sort_reduce_kernel`,
 * `sort_scan_kernel`, `sort_scan_add_kernel` and `sort_scatter_kernel` cannot
 * compile, the fragments are never ordered, and the frame is stripe noise.
 * There is no f32 path to fall back to: this is a missing browser feature,
 * not a precision choice.
 *
 * Checked on the adapter, not the user agent, so a future Safari that ships
 * subgroups starts working with no change here.
 *
 * @returns {Promise<string|null>} the refusal text, or `null` to proceed.
 */
async function subgroupSupportError() {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  adapterHasSubgroups = Boolean(adapter && adapter.features.has("subgroups"));
  mark("adapter-features", adapter ? [...adapter.features].join(" ") : "none");
  if (adapterHasSubgroups || params.get("anyway") === "1") return null;
  return (
    "This browser's WebGPU has no `subgroups` extension, so the TRIPS\n" +
    "rasteriser cannot run here. It would draw stripe noise, not the scene.\n\n" +
    "Exactly what fails: brush-sort's four radix kernels (sort_reduce_kernel,\n" +
    "sort_scan_kernel, sort_scan_add_kernel, sort_scatter_kernel) call\n" +
    "CubeCL's plane_sum / plane_inclusive_sum, which its WGSL backend emits as\n" +
    "subgroupAdd / subgroupInclusiveAdd with @builtin(subgroup_invocation_id).\n" +
    "This adapter advertises:\n\n    " +
    (adapter ? [...adapter.features].join(", ") : "(no adapter)") +
    "\n\nwhich does not include `subgroups`. Safari 26.6.2 additionally rejects\n" +
    "the `enable subgroups;` directive with \"1:0: Expected 'f16'\" -- that\n" +
    "message is about the extension NAME, not about f16; Safari does grant\n" +
    "shader-f16 and compiles this pipeline's f16 kernels fine.\n\n" +
    "Open this same address in Google Chrome or Edge instead:\n\n    " +
    window.location.href +
    "\n\nTo look at the broken output anyway, add &anyway=1 to the address.\n" +
    "docs/WEB_VIEWER.md has the full diagnosis."
  );
}

async function main() {
  mark("start");
  if (!navigator.gpu) {
    fail(
      "This browser has no WebGPU (navigator.gpu is undefined).\n\n" +
        "The TRIPS viewer is WebGPU-only on purpose: every stage of the\n" +
        "rasteriser is a compute shader, and WebGL2 has none, so a WebGL\n" +
        "fallback could only show you an empty canvas.\n\n" +
        "Safari 26+ and Chrome 134+ on macOS have WebGPU.",
    );
    return;
  }

  // Reads the adapter's feature list, which is what `installWebGpuWorkaround`
  // needs before it decides whether prepending `enable subgroups;` is legal.
  // `?screenshot=1` is the verification path and must keep measuring a broken
  // browser honestly, so only ordinary viewing is refused.
  const refusal = await subgroupSupportError();
  if (refusal && !screenshotMode) {
    fail(refusal);
    return;
  }
  installWebGpuWorkaround();
  const wasm = await init();

  // Run the module's static constructors, exactly once.
  //
  // `rust/crates/trips-web/build.rs` links with `--export=__wasm_call_ctors`,
  // which stops `wasm-ld` wrapping every export in a `.command_export` shim
  // that re-runs the whole `.init_array` on entry. That matters here far more
  // than it usually would: `cubecl-ir` -> `pliron` registers thousands of
  // dialect entries through `inventory`, one `__wasm_call_ctors` costs ~110 us,
  // and `wasm-bindgen` resolves `__externref_table_alloc` by export name — so
  // every `JsValue` `wgpu` created while building a bind group was paying for
  // it. Measured: `raw level-0` went from 297 ms to ~20 ms a frame.
  //
  // Suppressing the wrappers means nothing calls the constructors any more,
  // so this line does. It must run before any exported function; `init()` has
  // only run `__wbindgen_start`, and this crate registers no
  // `#[wasm_bindgen(start)]` function.
  if (typeof wasm?.__wasm_call_ctors !== "function") {
    fail(
      "this build did not export `__wasm_call_ctors`.\n\n" +
        "rust/crates/trips-web/build.rs is supposed to pass\n" +
        "`--export=__wasm_call_ctors` to the wasm linker. Without it the page\n" +
        "would still render, but ~15x slower, because wasm-ld wraps every\n" +
        "export in a shim that re-runs pliron's whole inventory registration.\n" +
        "Refusing rather than shipping that silently — docs/WEB_VIEWER.md.",
    );
    return;
  }
  wasm.__wasm_call_ctors();

  trips.install_panic_hook();
  mark("wasm-ready");

  if (screenshotMode) {
    canvas.width = SHOT_SIZE.width;
    canvas.height = SHOT_SIZE.height;
    canvas.style.width = `${SHOT_SIZE.width}px`;
    canvas.style.height = `${SHOT_SIZE.height}px`;
  } else {
    fitCanvas();
  }

  const options = {
    // In screenshot mode the canvas IS the render target size, so scale 1.0
    // keeps the blit a 1:1 copy; see SHOT_SIZE.
    scale: screenshotMode ? 1.0 : Number(params.get("scale") ?? 0.75),
    halfNet: (params.get("half") ?? "1") !== "0",
    // Off by default, as natively -- see the module header.
    packedSort: params.get("packed") === "1",
    mode: params.get("mode") ?? "network",
  };
  if (params.has("view")) options.view = Number(params.get("view"));

  const bundleUrl = params.get("bundle") ?? "./bundle";
  hud.textContent = `fetching ${bundleUrl} …`;

  mark("fetching");
  try {
    lastStatus = JSON.parse(await trips.start(canvas, bundleUrl, JSON.stringify(options)));
    mark("started", lastStatus.adapter?.name ?? "?");
    // The first network frame runs CubeCL's convolution autotune at
    // `AutotuneLevel::Full` -- the level that registers no roofline bounds
    // generator, which is what keeps the U-Net view off wasm's `read_sync`
    // trap (docs/WEB_VIEWER.md). Full means every candidate is benchmarked,
    // so the first frame is tens of seconds and every later one is not. A
    // blank canvas for that long looks like a hang unless the page says so.
    hud.textContent =
      "first frame: autotuning the convolutions (once per shape, ~20 s) …";
  } catch (e) {
    fail(`could not start the viewer:\n\n${e}`);
    return;
  }

  if (screenshotMode) {
    try {
      await screenshotRun();
    } catch (e) {
      fail(`screenshot run failed: ${e}`);
    }
    return;
  }

  installInput();
  interactiveLoop();
}

main().catch((e) => fail(String(e && e.stack ? e.stack : e)));
