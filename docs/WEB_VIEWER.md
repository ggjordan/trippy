# Web viewer

Two things live in this document:

1. **The TRIPS web viewer (v0.5.0)** — `rust/crates/trips-web` + `web/`: trippy's
   *own* forward pass (`brush-pyramid`'s CubeCL rasteriser, `brush-unet`'s Burn
   decoder — the same crates the native `trips-viewer` binary runs) compiled to
   `wasm32-unknown-unknown` and driving WebGPU on a `<canvas>`. This is the part
   that matters; it is first, below.
2. **The v0.5.0 groundwork (2026-09-06, earlier the same day)** — proving the
   toolchain with the *stock* Brush web app. Kept because its build notes
   (`npm ci` vs `npm install`, the base-path trap) still apply, and because its
   Quest research is still the only Quest assessment that exists.

---

# 1. The TRIPS web viewer (`trips-web`)

## What it is

```
web/index.html + web/trips.js           <- no framework, no bundler
        │  <script type="module">
        ▼
pkg/trips_web.js  (wasm-pack --target web)
        │
        ▼
rust/crates/trips-web  (cdylib)
        ├── gpu.rs    wgpu Instance -> Adapter -> Device on the canvas surface,
        │             then burn_wgpu::init_device on THAT device
        ├── blit.rs   trips_viewer::BLIT_WGSL, as a plain wgpu render pipeline
        └── lib.rs    wasm-bindgen entry points; thread_local viewer state
                │
                ▼
rust/crates/trips-viewer  (LIBRARY target, new in v0.5.0)
        bundle.rs · camera.rs · renderer.rs   <- shared with the native binary
                │
                ▼
brush-pyramid (gpu) · brush-unet (gpu) · brush-cube/-sort/-prefix-sum
```

`trips-viewer` was split into a library plus a binary so the browser runs the
**same** bundle loader, the **same** fly camera and the **same** per-frame
pipeline as the Mac app. There is no second implementation of anything, and
`shaders/blit.wgsl` is one `&'static str` shared by both front ends
(`trips_viewer::BLIT_WGSL`), so a frame cannot be laid out two different ways.

What is deliberately *not* shared: the window toolkit. `eframe`/`egui`/`rfd`
moved to `[target.'cfg(not(target_family = "wasm"))'.dependencies]` of
`trips-viewer`; the web front end is `wgpu` on a canvas plus ~380 lines of
plain JavaScript. New runtime dependencies are exactly `wasm-bindgen`,
`wasm-bindgen-futures`, `web-sys`, `js-sys`, `console_error_panic_hook` and
`getrandom`'s `wasm_js` backend (the last is a feature selection for a crate
already in the graph via `ahash` -> `burn-core`, done in the root crate of the
wasm build exactly as `apps/brush-app/Cargo.toml` does it).

**WebGPU only, no WebGL fallback.** Every stage of `brush_pyramid::gpu` is a
compute shader and WebGL2 has none, so a WebGL path could not render the scene
— only an empty canvas that looked like a viewer.

## Build

```bash
# Same one-time setup as the groundwork build below (wasm-pack, wasm32 target).
bash scripts/web_build.sh --check --trips        # verify the toolchain, build nothing
bash scripts/cpu_heavy.sh trips-web -- bash -c \
  'TRIPPY_OUTPUT=$PWD/output bash scripts/web_build.sh --trips'

# When a wasm PANIC has to be read: --profiling keeps release codegen and the
# wasm name section, which --release's wasm-opt strips. Add wasm-pack's
# --no-opt (run it directly) if even -O is losing frames, and set
# `Error.stackTraceLimit = 300` in the page, or the trace stops at panic_fmt.
bash scripts/cpu_heavy.sh trips-web -- bash -c \
  'TRIPPY_OUTPUT=$PWD/output bash scripts/web_build.sh --trips --profiling'
```

`scripts/web_build.sh --trips`:

1. Guards: submodule present (`brush-cube`/`-sort`/`-prefix-sum` are path deps),
   `npm`, `wasm-pack`, the `wasm32-unknown-unknown` target, and that
   `--bundle` names a directory containing `bundle.json`.
2. `wasm-pack build rust/crates/trips-web --release --target web --out-dir
   $TRIPPY_OUTPUT/web/trips-dist/pkg`. **`--target web`, not `bundler`** — that
   emits a plain ES module, so `index.html` can load it with
   `<script type="module">` and there is no vite, no `npm install`, and nothing
   to bundle. (The stock-Brush build below still needs vite; this one does not.)
3. Copies `web/index.html` and `web/trips.js` next to `pkg/`.
4. Copies the bundle's three files (`bundle.json`, and whatever the manifest
   names for `points`/`weights`) into `dist/bundle/`. Default bundle:
   `$TRIPPY_OUTPUT/brush/horse_bundle` — the **public** horse scene.

Output: `$TRIPPY_OUTPUT/web/trips-dist/`, ~102 MB, of which 80 MB is
`points.npz` and 26.9 MB is `trips_web_bg.wasm`.

### Build timings on this Mac (M3 Ultra, warm cargo registry, shared target dir)

| step | time |
|---|---|
| `cargo build --release` for `wasm32-unknown-unknown` (brush-pyramid, brush-unet, trips-viewer, trips-web + Burn/CubeCL/wgpu) — **cold** | 1 m 30 s |
| the same, incremental after a one-crate edit | 9–23 s |
| `wasm-bindgen` + `wasm-opt -Oz --converge` (75 MB → **26.9 MB** `trips_web_bg.wasm`) | ~40 s |
| **total `--trips --release`, cold** | **2 m 14 s wall** (20 m 16 s user) |
| `--profiling` variant (wasm-opt `-O`; 27 MB). Note `-O` **still strips the name section** — add `wasm-pack --no-opt` (75 MB, loads fine over loopback) when a stack trace has to be readable | 23–31 s |

## Running it

`scripts/deliver.sh output/web/trips-dist <name> "<why>"` writes
`OPEN_TRIPS_WEB_<NAME>.command`: `python3 -m http.server` bound to
**127.0.0.1 only**, then `open`s the page. Nothing leaves the machine; the
80 MB point set is fetched over loopback, which costs nothing worth measuring.

Controls are the native viewer's (`docs/USER_GUIDE.md`). Query parameters:

| parameter | default | meaning |
|---|---|---|
| `?bundle=<url>` | `./bundle` | bundle directory URL |
| `?scale=<f>` | `0.75` | render scale — the shipped default, same as `scripts/open_mac_viewer.sh` |
| `?half=0\|1` | `1` | f16 decoder — the shipped default, same as the native launcher |
| `?packed=0\|1` | `0` | packed 32-bit sort key (approximate depth). Off, as natively; the `P` key toggles it and the HUD names the sort in use |
| `?anyway=1` | off | render even on an adapter without `subgroups`, which draws stripe noise on purpose |
| `?mode=` | `network` | `network` / `raw` / `coverage`, the same three honesty views. All three render; `network` costs ~20 s of autotune on its first frame |
| `?view=<n>` | bundle's | dataset view index |
| `?screenshot=1` | off | run the verification sequence (below) and stop |
| `?trace=1` | off | POST a stage-by-stage progress trace |

## Verifying it without looking at it

Safari has no headless mode on this Mac and `safaridriver --enable` needs an
interactive sudo, so the page reports on itself. `?screenshot=1`:

1. sizes the canvas to exactly **1440x810** and renders at scale 1.0 — which
   reproduces the identical camera to `trips-viewer --screenshot --half-net
   --scale 0.75` on a 1920x1080 view (`render_camera` scales `fx` by
   `width / reference.width`, which is what `--scale` does natively) and makes
   the blit a 1:1 copy rather than an upsample;
2. renders 3 warm-up frames, then measures fps over a fixed 5 s window;
3. posts the canvas via `canvas.toBlob()` **and** a second PNG produced by
   `screenshot_png()` — a GPU readback encoded by the very same
   `png::feature_to_rgb8` the native `--screenshot` uses, as a capture that
   cannot come back blank;
4. posts a JSON beacon: adapter, granted WebGPU features, scene, point count,
   render size, mode, whether f16 is really in use, fps, frame count.

The receiving end is a throwaway harness under `$TRIPPY_OUTPUT/web/`
(`beacon_server.py`, `verify.sh`, `psnr.py` — not in the repo): a static file
server that also accepts `POST /__trips_beacon` and `POST /__trips_shot`, and
logs every `GET /__trips_stage?s=...` line, which is what turns a browser tab
into a readable trace. The delivered `.command` launcher uses a plain
`http.server`, which answers 501 to those POSTs; the page catches that and
carries on.

`beacon_server.py` also takes `POST /__trips_log?s=<stage>` and appends the
**untruncated** body to `trace.log`. A `__trips_stage` GET is capped at 400
characters, which is fine for a stage name and useless for a wasm panic's
stack trace — that is what the POST is for, and it is how blocker 4 below was
finally read. `web/trips.js` does not send it: the diagnostic is two lines
added to the *copy* of `trips.js` inside a scratch dist, next to
`Error.stackTraceLimit = 300`, so the shipped page stays free of it.

## Blockers hit, and what each one turned out to be

All five were found in this order. None of them is in TRIPS code: three are in
the pinned dependency stack, one is a real browser gap, and the fifth — the
27x — is in the wasm *link*. **Blockers 4 and 5 were both misdiagnosed first**
and their entries below are the corrected ones, kept in full because the wrong
readings are instructive.

### 1. `std::time::Instant` and `block_on` panic on wasm32 (ours, fixed)

`Instant::now()` panics with "time not implemented on this platform" on
`wasm32-unknown-unknown`, and `brush_pyramid::gpu::block_on` parks the only
thread the page has. Fixed at the definition sites: `block_on` is now
`#[cfg(not(target_family = "wasm"))]` so misuse cannot compile, and
`render_inner`'s clock is constructed only when per-stage timings were asked
for (`timed.then(Instant::now)`). `trips_viewer::renderer` got a `SubmitClock`
that reports 0 on the web, where the frame interval comes from
`requestAnimationFrame` anyway.

### 2. wgpu panics on every **clean** error-scope pop (fatal; JS shim)

```
panicked at wgpu/src/backend/webgpu.rs:85:13: Unexpected error
```

`future_pop_error_scope` reads `GPUDevice.popErrorScope()` into a
`js_sys::JsOption<GpuError>`, whose `into_option()` returns `None` **only for
`undefined`**. The WebGPU specification says a clean pop resolves with
**`null`**, and Chrome and Safari both do. So `null` is taken for an error,
handed to `Error::from_js`, matches neither `GPUValidationError` nor
`GPUOutOfMemoryError`, and reaches `panic!("Unexpected error")`.
`JsNullable<T>` — the null-aware sibling in the same wasm-bindgen module — is
what that code wants. CubeCL wraps every kernel launch and pipeline creation in
an error scope (`cubecl-wgpu` `backend/base.rs:138`, `compute/stream.rs:401`),
so the **first** TRIPS frame kills the page, in every browser, every time.

Cannot be fixed in trippy's Rust (it is inside the pinned
`ArthurBrussee/wgpu#js-interop-30`, reached through wasm-bindgen 0.2.127's
newer js-sys API). `web/trips.js`'s `installWebGpuWorkaround()` rejects the pop
instead: wgpu maps a rejected promise to "no error" and never calls `from_js`.
Every error is still logged to the console and to the trace first, so nothing
is hidden — the cost is that wgpu itself stops seeing validation errors, which
for a viewer means a bad frame shows as bad pixels rather than as an exception.
Delete the shim when the fork carries the `JsOption` -> `JsNullable` fix.

### 3. CubeCL emits subgroup builtins without `enable subgroups;` (JS shim)

```
Error while parsing WGSL: :81:11 error: cannot call built-in function
'subgroupAdd' without extension 'subgroups'
  let v75 = subgroupAdd(v74);
  - While calling [Device "trips-web"].CreateShaderModule(["sort_reduce_kernel"])
```

`brush-sort`'s radix passes call CubeCL's `plane_sum` / `plane_inclusive_sum`
**unconditionally** in their `#[cube]` source (`crates/brush-sort/src/kernels.rs`),
and CubeCL's WGSL backend translates those to `subgroupAdd` /
`subgroupInclusiveAdd` — but never emits the `enable subgroups;` directive WGSL
requires before either can be called. Its MSL and SPIR-V backends need no such
directive, which is how this survived on the fork's Metal-first path. All four
of `sort_reduce_kernel`, `sort_scan_kernel`, `sort_scan_add_kernel` and
`sort_scatter_kernel` fail to compile, their pipelines come out invalid, and no
frame is ever produced.

**Masking `Features::SUBGROUP` off the device does not help** — measured, not
assumed: with `subgroups` absent from the granted feature list, CubeCL emitted
the same calls and Chrome reported the same four errors. There is no
non-subgroup lowering to fall back to.

What does work is supplying the missing line: `installWebGpuWorkaround()`
prepends `enable subgroups;` to any shader source that calls a subgroup builtin
and does not already declare it — 4 shaders per session, reported in the beacon
as `subgroupShaderPatches`. Chrome then compiles all four and renders. The mask
was reverted so `crate::gpu` keeps requesting `Features::SUBGROUP`, which is
what makes the directive legal there.

**The injection is now conditional on the adapter actually having the feature**
(`adapterHasSubgroups`), and the reason is in "Safari: the exact construct"
below: on an adapter without subgroups the directive turns a precise error into
a misleading one, and that misled this project for a whole session.

### 4. CubeCL's autotune roofline probe traps on wasm — the U-Net view (FIXED)

```
panicked at cubecl-environment/src/future/reader.rs:9:22:
Failed to read tensor data synchronously. This can happen on platforms that
don't support blocking futures like WASM.
```

**v0.5.0 read this wrongly and shipped `networkBlocked: true` because of it.**
The message names `read_sync`, the two obvious suspects were both readbacks of
the U-Net's output tensor, and the entry this replaces named them: the trap was
said to be in `burn_bridge::resolve_to_cube_float` (through burn-fusion's
`submit_blocking`) and in `Tensor::into_data_async` (through burn-dispatch's
`float_into_data`). **Neither is true on the pinned revisions.** Two things
settled it:

1. Reading the sources. `FusionClient::resolve_tensor_float`
   (`burn-fusion/src/client.rs:362`) calls `DeviceHandle::submit_blocking`,
   which is only a channel round trip when cubecl's `multi_threading` cfg is
   on — and `cubecl-common/build.rs:11` defines that cfg as
   `all(feature = "std", not(target_family = "wasm"))`. On wasm the handle is
   `ReentrantMutexDeviceHandle`, whose `submit_blocking`
   (`cubecl-common/src/device/handle/reentrant.rs:51`) runs the closure inline
   under a reentrant mutex: no thread parks, no future is polled. There is no
   `read_sync` in `burn-fusion`, `burn-cubecl-fusion` or `burn-ir` at all. And
   `burn-dispatch`'s `read_sync` (`ops/tensor.rs:54`) is in `float_to_device`,
   the cross-backend move, not in `float_into_data` (`ops/tensor.rs:29`, a
   plain `.await`).
2. Reading the **stack**, which is how it was actually found. A wasm panic's
   stack is a list of `wasm-function[N]` unless the name section survives, and
   `wasm-opt` strips it. `scripts/web_build.sh --profiling` plus
   `wasm-pack --no-opt` keeps it, and `Error.stackTraceLimit = 300` makes the
   trace long enough to reach past `panic_fmt`.

The real path, bottom up:

```
trips_viewer::renderer::Renderer::render
  brush_unet::camera::NeuralCamera::forward
    brush_unet::camera::linspace_centered
      Tensor::<1>::from_data  ->  Fusion::float_from_data  ->  client.register(NoOp)
        MultiStream::register -> Processor::process -> execute_block_optimization
          OrderedExecution::execute_operations -> Conv2dOps::execute      <-- the queued U-Net conv
            burn_cubecl::kernel::conv::forward::tune::conv_autotune
              LocalTuner::execute -> BoundsGenerator::generate
                burn_cubecl::kernel::autotune_bounds::with_bounds
                  cubecl_std::throughput::measure_peak_throughput          <-- "Native only, panics on WASM"
                    ComputeClient::measure_throughput
                      ThroughputBenchmarker::measure
                        compute_direct::build_kernel
                          cubecl_environment::future::base::block_on       <-- read_sync -> trap
```

It is **not the output tensor at all**. `Tensor::from_data` inside the tone
mapper's `linspace_centered` is merely what drains the fusion stream; the
queued operation that runs is the U-Net's first `conv2d`, and CubeCL autotunes
it. The tuner's *roofline bounds generator* asks the device for its peak
throughput first, and `cubecl-std/src/throughput/base.rs` says of that function,
in its own doc comment, **"Native only, panics on WASM"**. That is why
`raw level-0` and `coverage` were fine: they run no convolution, so nothing is
ever autotuned.

**The fix, and it needs no fork and no `[patch]`.**
`burn-cubecl/src/kernel/autotune_bounds.rs::with_bounds` registers **no**
bounds generator when the autotune level is `AutotuneLevel::Full`:

```rust
if configured_thresholds().is_none() {   // None <=> AutotuneLevel::Full
    return set;
}
```

So `brush_pyramid::gpu::disable_autotune_roofline_bounds()` sets
`CubeClRuntimeConfig { autotune.level = Full, .. }` through the supported
`RuntimeConfig::try_set`, and `trips_web::gpu::Gpu::create` calls it **before
the first CubeCL device exists** (`try_set` refuses to override a config that
has already been read; the return value is checked and a failure is warned
about on the console). Autotune itself still runs — its wasm path
(`tune_fixed_samples`, resolved through `wasm_bindgen_futures::spawn_local`) is
asynchronous throughout.

**What `Full` costs.** "Full" means "benchmark every candidate, no roofline
short circuit", so the **first** frame that meets a new convolution shape pays
for the whole candidate set: ~20 s on this Mac, once, per shape. Every frame
after it is unaffected. `web/trips.js` says so on the canvas while it happens,
because 20 s of blank canvas is indistinguishable from a hang. There is no
cheaper level available: `Minimal`, `Balanced` and `Extensive` all install the
generator that traps.

**Consequence for `trips_viewer::renderer`.** `resolve_network_output` is no
longer `cfg`-split: both front ends now call
`burn_bridge::resolve_to_cube_float`, so the browser's frame goes from the
U-Net to the blit with **no readback and no re-upload**, exactly as the Mac
app's does. `brush_pyramid::gpu::upload_f32` — added in v0.5.0 for the
re-upload half of the old web path — is kept because it is a reasonable
primitive, but nothing in the viewer calls it any more.

**Measured result.** Chrome 152 renders the network view of the public horse
scene at 1440x810, and the GPU-readback PNG it produces (the same
`png::feature_to_rgb8` the native `--screenshot` runs) matches
`output/brush/viewer/halfnet_s75.png`, the native
`trips-viewer --screenshot --half-net --scale 0.75` reference, at
**PSNR 104.54 dB** (`canvas.toBlob()` capture: 87.01 dB).

That number was **62.04 dB** when this entry was first written, and the
difference is the autotune, not the picture: a `network` session that has
rendered ~50 frames before the screenshot has let CubeCL converge on the same
convolution kernel the native build uses, while the v0.5.0 run managed 6 frames
at 1.09 fps and had not. A `raw level-0` session still reads 62.04 dB today,
because the only convolution it ever runs is the screenshot's own. The v0.5.0
reading of that residual as "f16 rounding" was wrong.

### 5. `wasm-ld` links the module as a *command*, so every call re-ran every static constructor (FIXED — this was the 27x)

This is the answer to the question the v0.5.0 entry left open, and it is worth
reading before the numbers because every other explanation was wrong.

**The symptom.** `raw level-0` at 1440x810 ran at 3.32 fps in Chrome and
102 fps natively on the same Mac. The per-frame point upload had already been
removed and bought only 2.90 → 3.32.

**What it is not.** Each of these was tested and rejected with a number
(method below):

| hypothesis | verdict | the number that settled it |
|---|---|---|
| the error-scope shim's promise per kernel launch serialises the queue | **rejected** | **0** `popErrorScope` calls per frame. CubeCL scopes shader *compilation* and `sync()`, not launches (`cubecl-wgpu` `backend/base.rs:138`, `compute/stream.rs:401` — the latter is `sync`, which the v0.5.0 note misread as a per-launch scope) |
| the subgroup shim recompiles shaders every frame | **rejected** | **0** `createShaderModule` per frame after warm-up (21 for the whole session) |
| validation overhead of many small dispatches | **rejected** | 85 dispatches, 86 bind groups, 8 submits per frame, and **3.9 ms** of JS-side WebGPU API time in a 315 ms frame |
| CubeCL maps a buffer per launch on web | **rejected** | **0** `createBuffer` and **1** `mapAsync` per frame |
| a per-frame readback | **rejected as the cost** | it does exist — `render_inner`'s one 4-byte fragment-count read — and it costs **1.2 ms** |
| WGSL-vs-MSL codegen of the radix sort makes the GPU slow | **rejected** | the frame is **291 ms at scale 1.0, 291 ms at 0.5 and 291 ms at 0.35** — 8x less pixel work, identical time. The GPU was never the bottleneck |
| `wasm-opt -Oz` optimises for size and costs speed | **rejected** | no wasm-opt at all: 476 ms; `-Oz --converge`: 297 ms; `-O3`: 300 ms |
| V8 keeps the 27 MB module in the Liftoff baseline tier | **rejected** | `--js-flags=--no-liftoff`: 323 ms vs 297 ms |
| **the JS↔wasm boundary itself is pathologically slow** | **CONFIRMED** | `trips.look(0, 0)` — an exported function that adds two floats to a camera — cost **113 microseconds** a call. `trips.status()`, 210 us |

**What it is.** `wasm32-unknown-unknown` has no libc, so `wasm-ld` synthesises
`__wasm_call_ctors` (the `.init_array` chain) itself, **unguarded**, and — for a
module it does not treat as a reactor — wraps *every* export in a
`<name>.command_export` shim that calls it on entry and `__wasm_call_dtors` on
exit. That is the WASI *command* ABI, where each call is a fresh program run.
It is normally free, because Rust code has no static constructors.

This graph does. `cubecl-ir` pulls in **`pliron`**, whose dialect and
trait-cast registrations are thousands of `inventory::submit` calls, all of them
in `.init_array`. One `__wasm_call_ctors` run measures ~110 us here.

And the wrappers are not only on the functions the page calls. Dumping the
export table shows all 21 of them wrapped, including
`__externref_table_alloc`, `__externref_table_dealloc`, `__wbindgen_malloc` and
`__wbindgen_free` — and `wasm-bindgen` resolves those **by export name**, so its
own generated shims called the wrappers too. Every `js_sys::Object::new()` that
`wgpu`'s WebGPU backend performs while building one bind group therefore
re-registered the whole of `pliron`. At ~850 `Object::new` plus ~850 drops plus
mallocs — about **2,500 constructor runs per frame** — that is the entire frame:
2,500 x 110 us = 275 ms of the 297 ms.

That is also why the CPU profile looked absurd: its hottest leaves really were
`__wasm_call_ctors`, `inventory::Registry::submit` and
`pliron::utils::trait_cast::TraitCasterInfo`, sitting under
`wgpu::backend::webgpu::WebDevice::create_bind_group`. It was read as symbol
misattribution at first. It was literal.

**The fix** is one linker flag and one line of JavaScript, and they only work
together:

- `rust/crates/trips-web/build.rs` emits
  `cargo::rustc-link-arg=--export=__wasm_call_ctors` for `wasm32` targets.
  Exporting the symbol tells `wasm-ld` the caller will run the constructors, so
  it emits **no** command-export wrappers (verified: 42 occurrences of
  `command_export` in the module before, 0 after). `cargo::rustc-link-arg`
  applies to this crate's own cdylib only — no dependency is recompiled and the
  native `trips-viewer` link never sees it. (`RUSTFLAGS` would have worked too
  and would have rebuilt all ~500 dependency crates.)
- `web/trips.js` then calls `wasm.__wasm_call_ctors()` exactly once, right after
  `init()`, because nothing else does any more. It **refuses to start** if the
  export is missing, so a toolchain change cannot silently restore the slow
  build; `tests/test_web_build_script.py` asserts both halves.

**The result**, same machine, same headless Chrome, same view, exact sort:

| | before | after |
|---|---|---|
| `trips.look(0, 0)`, one exported no-op | 113 us | **0.065 us** |
| `raw level-0`, 1440x810 | 3.32 fps | **75.9 fps** |
| `network` (U-Net), 1440x810 | 1.09 fps | **17.7 fps** |
| PSNR of the readback PNG vs the native `--half-net --scale 0.75` frame | 62.04 dB | **104.54 dB** |

The PSNR improved because it was never really about f16 rounding: with the
frame no longer taking a third of a second, CubeCL's autotune settles on the
same convolution kernel the native build picks.

### Where a browser frame goes now, and how each line was measured

Chrome 152 headless, 1440x810, view 8, `raw level-0`, exact sort, release
build, with a Splats training on the same GPU. Both columns are the same
instrumented page on the same machine; the per-frame **API counts are identical**
(85 dispatches, 86 bind groups, 8 submits, 5 `writeBuffer`, 1 `mapAsync`) — the
work did not change, only what each boundary crossing cost.

| | before, ms/frame | after, ms/frame | how it was measured |
|---|---|---|---|
| `wasm-ld` command-export wrappers re-running `.init_array` | **~275** | **0** | ~2,500 boundary crossings/frame x 110 us; the crossing cost timed directly with 20,000 `trips.look()` calls (113 us -> 0.065 us) |
| **waiting for the GPU at the one fragment-count readback** | 1.2 | **10.9** | `GPUBuffer.mapAsync` wrapped, call to resolution |
| JS-side WebGPU API calls (86 `createBindGroup`, 85 `setPipeline`/`setBindGroup`/`dispatchWorkgroups`, 8 `submit`) | 3.9 | **0.30** | every method on every `GPU*` prototype wrapped with a timer |
| everything else in the wasm — CubeCL's scheduler and `wgpu`'s descriptor marshalling for 85 launches | ~16 | **~0.3** | frame total minus the rows above |
| shader compilation, buffer creation, error scopes, point upload | 0 | 0 | 0 calls per frame after warm-up |
| **whole frame** | **297** | **11.5** | median of 20 frames |

**Read the readback row carefully.** It has not got slower; the frame has got
faster around it. `render_inner` reads the total fragment count back after the
prefix sum, because the fragment buffers cannot be sized without it, and that
read waits for the GPU to finish stages 1-2. Before, the host was so slow that
the GPU had long since finished and the wait was 1.2 ms of nothing. Now it is
**95 % of the frame**, which is the correct shape — the native viewer pays the
same sync, and its `raw` profile is likewise dominated by real GPU work
(`sort 7.4 ms` of 10.6 ms at 1080p). The browser frame is GPU-bound again.

That also names the next thing worth attacking, if anything is: removing the
readback entirely (indirect dispatch off a device-side count, or a fixed
fragment-slot budget) would let the whole frame be submitted without a stall.
It is a `brush-pyramid` change that would help both front ends, and it is not
attempted here.

**Method, so it is repeatable.** Everything above came from a throwaway harness
under `$TRIPPY_OUTPUT/web/` (not in the repo): `perf-dist/`, a copy of the page
that wraps every `GPU*` prototype method, every `__wbg_*` import and
`GPUBuffer.mapAsync` with counters and a per-frame event timeline; `perf_run.sh`,
which serves it to `--headless=new` Chrome on 127.0.0.1 and waits for the beacon
(**8 seconds per run**, which is what made this affordable while a training held
the GPU); and `cdp_profile.js`, which drives the DevTools protocol to take a CPU
sampling profile of exactly the measured frames, gated on a `PERF:READY` console
message so module load is not in it. Symbol names need a build with the wasm
name section — `wasm-pack build --release --no-opt` keeps it, and unlike
`--profiling` it keeps release codegen too.

Headless Chrome reproduced the windowed number exactly (3.33 vs 3.32 fps), which
is what made it usable as the measurement environment.

### Safari: the exact construct, and it is not f16

**v0.5.0's reading was wrong.** Safari 26.6.2 reports

```
GPUValidationError: 1 error generated while compiling the shader:
1:0: Expected 'f16'
```

and that was taken for an f16 problem. It is not. Safari grants `shader-f16`,
and the only two shaders in the whole pipeline that mention f16 —
`cast_element_i_f32_o_f16_n_1` and `..._n_4`, which carry their own
`enable f16;` — compile there without complaint.

`1:0` is the position where `installWebGpuWorkaround()` prepends
`enable subgroups;`. The message is Safari's WGSL parser saying that **`f16` is
the only extension name its `enable` directive accepts**.

Measured with a shader-compile-only probe (no rendering, ~2 s, results POSTed
to the beacon server; the probe page is in `$TRIPPY_OUTPUT/web/safari-probe/`):

| case | Chrome 152 | Safari 26.6.2 |
|---|---|---|
| `subgroups` in `adapter.features` | **yes** | **no** |
| `shader-f16` in `adapter.features` | yes | **yes** |
| bare trivial compute shader | ok | ok |
| `enable f16;` + trivial | ok | **ok** |
| `enable subgroups;` + trivial | ok | `1:0: Expected 'f16'` |
| `enable f16, subgroups;` + trivial | ok | `1:0: Expected 'f16'` |
| `enable subgroups_basic;` + trivial | `1:8 expected extension` | `1:0: Expected 'f16'` |
| real `iota_kernel` (no directive, no subgroups) | ok | **ok** |
| real `cast_element_i_f32_o_f16_n_1` (has `enable f16;`) | ok | **ok** |
| real `sort_reduce_kernel`, **no** directive | `cannot call built-in function 'subgroupAdd' without extension 'subgroups'` | `9:66: Unknown builtin value` (i.e. `@builtin(subgroup_invocation_id)`) |
| real `sort_reduce_kernel` **with** `enable subgroups;` | ok | `1:0: Expected 'f16'` |

**Verdict: Safari 26.6.2 has no WebGPU subgroups support in any of its three
forms** — no `subgroups` adapter feature, no `enable subgroups;` directive, no
`subgroup_*` builtin. `brush-sort`'s four radix kernels
(`sort_reduce_kernel`, `sort_scan_kernel`, `sort_scan_add_kernel`,
`sort_scatter_kernel`) call CubeCL's `plane_sum` / `plane_inclusive_sum`
unconditionally, so they cannot compile, the fragments are never ordered, and
the frame is stripe noise. There is no f32 path to offer: this is a missing
browser feature, not a precision choice. That also matches the old Safari
beacon exactly — it logged **four** "Expected 'f16'" errors and
`subgroupShaderPatches: 4`, one per patched shader; the earlier note that "one
shader fails" was reading only the first.

What the page does about it, as of this session:

- `web/trips.js` asks the adapter for its features **before** starting the wasm.
  No `subgroups` means it refuses to render and prints the whole diagnosis —
  the four kernel names, the builtins involved, this adapter's actual feature
  list, and the fact that the `f16` message is about the extension *name*.
  `?anyway=1` still proceeds, because "measured wrong on 2026-09-06" should stay
  checkable rather than become folklore.
- The check is on **capability, not user agent**, so a Safari that ships
  subgroups starts working with no change here. (It replaces a UA sniff.)
- The `enable subgroups;` injection is now conditional on the adapter having the
  feature. Injecting it into a browser that has no subgroups only replaces an
  accurate error with a confusing one.

### Browser support matrix (this Mac, 2026-09-06, view 8, 1440x810, exact sort)

| | Chrome 152.0.7977.83 | Safari 26.6.2 |
|---|---|---|
| installed | `brew install --cask google-chrome` (dev tooling; nothing was sent anywhere) | system |
| `navigator.gpu`, adapter | yes (unnamed, `BrowserWebGpu`) | yes (`apple`) |
| `shader-f16` granted | yes | **yes** |
| `subgroups` granted | **yes** | **no** — and no `enable` directive and no builtin either |
| shaders needing the injected `enable subgroups;` | 4-5 | not injected any more (it cannot help) |
| **`raw level-0`** | **75.9 fps** (was 3.32) | refused, with the exact reason |
| **`network` (the U-Net view)** | **17.7 fps** (was 1.09) | refused |
| `canvas.toBlob()` PNG | 2,023,351 bytes | — |
| GPU-readback PNG (`screenshot_png`) | 2,546,343 bytes | — |
| **PSNR vs the native `--half-net --scale 0.75` frame** | **104.54 dB** readback, 87.01 dB `toBlob` | — |
| **is the picture right?** | **YES — the horse** | it never gets to draw one |

### The packed sort key: offered, not shipped

`?packed=1` (and the `P` key) switch the rasteriser to one 32-bit radix sort
instead of two, which is 8 radix passes instead of 14 and 54 kernel launches
instead of 85. While launches *were* the frame it was worth 1.45x and it was
briefly the web default. It is not any more:

| sort | `raw level-0` | `network` | PSNR vs the native reference |
|---|---|---|---|
| exact two-pass (default, as native) | 79.1 fps | 17.4 fps | **104.54 dB** |
| packed 32-bit key | 114.6 fps | 19.4 fps | 36.85 dB |

Those four fps are a pairwise A/B on one binary, so they are comparable with
each other; the shipped `wasm-opt`'d dist measures 75.9 / 17.7 with the exact
sort, which is the same within the run-to-run spread of a browser sharing the
GPU with a training.

11 % more frames in the shipped view for 68 dB is not a trade this viewer makes
on anyone's behalf. The HUD always names the sort in use.

### Still not measured

- **fps was measured with a Splats training running on the same GPU**, so every
  browser number is a lower bound. Runs are ~8 s each: AGENTS.md's GPU rule caps
  an unqueued browser check at ~10-15 s.
- **Interactive fps at window size** — every number here is the fixed 1440x810
  screenshot canvas at render scale 1.0. The shipped launcher renders at 0.75 of
  the window, which is cheaper.
- **Whether the remaining ~7 ms of host launch path is worth attacking.** At
  17.7 fps for the network view the frame is now GPU- and U-Net-bound again
  (260 of its 314 kernel launches are the decoder), which is the same shape as
  the native profile. `docs/LIMITATIONS.md` has the native breakdown.
- **Other browsers.** Edge shares Chrome's Dawn stack and is untested here;
  Firefox's WebGPU was not checked.

---

# 2. Groundwork: the stock Brush web app

Status as of 2026-09-06 (earlier the same day): the toolchain was proven end to
end with the **stock** Brush fork's web app before any TRIPS code was compiled
for wasm. Everything in this half is still accurate for that build target
(`scripts/web_build.sh` with no `--trips`), and its `npm ci` and base-path notes
are the reason that build works at all.

## Build (stock Brush app)

`scripts/web_build.sh` wraps the fork's own build (`rust/brush-trips/apps/brush-app/web/package.json`):

```bash
# One-time setup (this Mac had neither installed):
npm i -g wasm-pack                        # 0.15.0, prebuilt binary via npm, ~3 s
rustup target add wasm32-unknown-unknown  # already had aarch64-apple-darwin only

# Check the toolchain without building anything:
bash scripts/web_build.sh --check

# Real build -- always through cpu_heavy.sh (compiles Burn/CubeCL/wgpu/egui to
# wasm32, plus a cold wasm-bindgen-cli cargo-install on first run):
bash scripts/cpu_heavy.sh web-build -- bash -c 'time bash scripts/web_build.sh'
```

`scripts/web_build.sh` does, in order:
1. Guard clauses: `rust/brush-trips` submodule present, `npm`/`wasm-pack` on
   `PATH`, `wasm32-unknown-unknown` in `rustup target list --installed`. Each
   missing prerequisite exits 2 with the exact fix command (never silently skips
   or fakes the step — see `AGENTS.md` "faking unsupported APIs").
2. `npm ci` (not `npm install` — see below) in `apps/brush-app/web`.
3. `npm run build:wasm-release` → `wasm-pack build .. --release --target bundler
   --out-dir web/pkg` (cargo build for `wasm32-unknown-unknown` + `wasm-bindgen`
   + `wasm-opt -Oz --converge`).
4. `BRUSH_BASE_PATH=/ npx vite build` — **not** the package.json `npm run build`
   target, which hardcodes `BRUSH_BASE_PATH=/brush-demo` for GitHub Pages. A
   `/brush-demo`-rooted bundle 404s every JS/wasm asset when served from a plain
   `http.server` document root, which is how `scripts/deliver.sh`'s
   `OPEN_*.command` serves things. Base path `/` is what makes local serving work.
5. Copies `apps/brush-app/web/dist/` to `$TRIPPY_OUTPUT/web/brush-dist/`
   (gitignored, not left inside the submodule) so `scripts/deliver.sh`'s
   artifact-location check (`$TRIPPY_OUTPUT` or `$SPLATS_ROOT/output`) passes.

**Why `npm ci`, not `npm install` (found the hard way this session):**
`rust/brush-trips` is itself an npm workspace root (`package.json`
`"workspaces": ["apps/brush-app/web", "apps/brush-js/web"]`), with the
lockfile at `rust/brush-trips/package-lock.json` — not inside
`apps/brush-app/web/`. Running `npm install` from the workspace-member
directory still rewrites that root lockfile: on this Mac it silently dropped
an `"extraneous": true` `brush_nextjs` workspace entry the checked-in lockfile
carries but this checkout doesn't have on disk (an older workspace layout
upstream, not something trippy touched). That is a write to a tracked file
*inside the submodule*, which is against `rust/README.md`'s submodule model
(Splats' patches as ordinary commits on a fork branch — not a place for
build-tool side effects). `npm ci` installs from the existing lockfile
without writing to it and was confirmed to leave `git status` inside
`rust/brush-trips` clean.

### First-build timings (this Mac, cold `wasm-bindgen-cli` install, warm cargo
registry cache from the v0.4.0 Brush-fork build)

| Step | Time |
|---|---|
| `npm ci` | a few seconds |
| `cargo install wasm-bindgen-cli` (wasm-pack's own dependency, first run only) | ~24 s |
| `cargo build --release` for `wasm32-unknown-unknown` (brush-app + all deps: Burn, CubeCL, wgpu, egui) | 1 m 30 s |
| `wasm-bindgen` + `wasm-opt -Oz --converge` (53 MB unoptimized `brush_app.wasm` → 21.7 MB `brush_app_bg.wasm`) | remainder of wasm-pack's 3 m 27 s |
| `vite build` (bundles the React/TS shell, embeds the wasm) | 1.48 s |
| **Total (`time bash scripts/web_build.sh`)** | **3 m 36 s wall** (23 m 33 s user, 0 m 54 s sys — the user/wall gap is `wasm-opt`'s multi-threaded `-Oz --converge` pass) |

A rebuild with a warm `wasm32-unknown-unknown` target directory is much faster
(incremental cargo build); the numbers above are a cold first build on this
machine, which is the number that matters for "does the toolchain work at all."

## Serving on 127.0.0.1

`scripts/deliver.sh <dist-dir> <name> "<why>"` generates
`output/deliver/<name>/OPEN_<NAME>.command`: a double-click launcher that starts
`python3 -m http.server` bound to `127.0.0.1` only (never `0.0.0.0`) and opens the
page in the default browser. This is the same pattern already used for
`OPEN_PANORAMA.command` in Jordan's review folder — nothing new was needed in
`deliver.sh` itself; the existing "directory containing `index.html`" branch
handles a vite `dist/` output correctly as-is (vite's default `index.html` +
relative asset paths work with a plain `http.server` root at base path `/`).

## Loading a splat

Brush's web viewer accepts `?url=<ply-url>` (see
`rust/brush-trips/apps/brush-app/web/README.md`), read by `App.tsx` from
`window.location.search` and passed to `app.load_url(...)` in
`BrushViewer.tsx`. A `.ply` placed anywhere the same `http.server` can reach it
(same origin is simplest) works, e.g.
`http://127.0.0.1:<port>/index.html?url=http://127.0.0.1:<port>/assets/synthetic/synthetic_2000.ply`.

For the one real delivery in this session (`web-brush-stock`), the copy of
`index.html` under `output/web/brush-dist/` (not the submodule's, not
`scripts/web_build.sh`'s general output — see below) has a small inline
redirect added *after* delivery: if no `?url=` is present, it redirects to
`?url=assets/synthetic/synthetic_2000.ply` so double-clicking
`OPEN_WEB-BRUSH-STOCK.command` shows the splat immediately with no manual URL
entry. `scripts/web_build.sh` itself does **not** do this — it produces the
stock, unmodified `dist/`; the redirect is a proof-of-concept convenience
patched into that one delivered copy only, documented here so it isn't
mistaken for something the build script does automatically.

**Never** load anything from `~/Splats` this way — this whole exercise used a
2,000-point synthetic Gaussian cloud generated by
`trippy.train.export.write_gaussian_ply` (random positions, colours, opacities,
isotropic sizes; see `output/web/assets/synthetic_2000.ply`, not committed,
regenerable from the snippet in `research/trips-metal.md`).

## Browser support matrix (this Mac, checked 2026-09-06)

| Browser | Installed? | WebGPU | Notes |
|---|---|---|---|
| Chrome | **No** — not installed on this machine | N/A | Fork's README and its own hosted demo (`arthurbrussee.github.io/brush-demo`) both say "Only works on Chrome and Edge." Not verified here for lack of a Chrome binary; see below for what was actually run. |
| Safari | Yes, 26.6.2 (macOS 26.6.2) | **Confirmed working**: `navigator.gpu.requestAdapter()` resolves (vendor/architecture/device all report `"apple"`), and the stock Brush wasm app loads a synthetic `.ply` and renders to a correctly-sized canvas with zero JS errors. Not the fork's officially claimed-supported browser ("hopefully supported soon" per its README) — but it works here regardless. No fps measurement (see below). |
| Edge | Not installed | — | Same Chromium/Dawn WebGPU stack as Chrome; not tested. |

`docs/SPEC.md`'s v0.5.0 acceptance line is "≥15 fps 1080p **in Chrome** on the
Mac" — Chrome is not installed on this machine as of this session. That is a real
gap for the fps acceptance criterion specifically (see "Open questions").

### What was actually run

No headless-screenshot verification was possible on this machine, honestly reported:
- **Chrome**: not installed (`/Applications/Google Chrome.app` absent, `mdfind` found
  nothing). The `--headless=new --enable-unsafe-webgpu --use-angle=metal
  --screenshot=...` check the task brief specified could not be attempted.
- **Safari `safaridriver`**: present (`/usr/bin/safaridriver`, "Included with
  Safari 26.6.2"), but `safaridriver --enable` requires an interactive sudo
  password this session does not have — automation via Safari's WebDriver was
  not available.
- **Safari AppleScript `do JavaScript`**: attempted (`osascript -e 'tell
  application "Safari" to do JavaScript ... in document 1'`); failed with
  `Safari got an error: AppleEvent timed out. (-1712)` — Automation/Apple-Events
  permission for this shell was not grantable non-interactively either. No
  pixel screenshot of the rendered splat was obtained by any method.

What **was** verified, without needing screen access or elevated permissions —
a small local-only diagnostic page (`fetch(...).then(POST result to
127.0.0.1)`), opened with the plain `open` command (no special permission
needed) and read back from disk:

1. **WebGPU is live in Safari 26.6.2 on this Mac.** The diagnostic page ran
   `typeof navigator.gpu` and `await navigator.gpu.requestAdapter()` and posted
   the result back:
   ```json
   {"ua":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 ... Version/26.6.2 Safari/605.1.15",
    "hasGpu":true,"adapter":"obtained",
    "adapterInfo":{"vendor":"apple","architecture":"apple","device":"apple","description":"apple"},
    "error":null}
   ```
2. **The full asset chain loads.** `python3 -m http.server`'s access log for a
   request to `index.html?url=.../synthetic_2000.ply` in Safari:
   `index.html` 200 → `index-*.js` 200 → `BrushViewer-*.js` 200 →
   `brush_app_bg-*.wasm` 200 (21.7 MB, served with `Content-Type:
   application/wasm`) → `synthetic_2000.ply` 200. Every asset the wasm app
   needs was fetched and returned successfully.
3. **The wasm app initialised without error and is driving a canvas sized to
   the window.** A second diagnostic (same beacon trick, injected into a
   throwaway copy of the dist directory, never the delivered one) waited 4 s
   after page load and reported:
   ```json
   {"rootHTMLlen":472,"canvasCount":1,"canvasInfo":[{"w":1285,"h":1230}],"errors":[]}
   ```
   One `<canvas>` element, sized to the actual Safari window (1285×1230) —
   `EmbeddedApp.start(canvas)` in `BrushViewer.tsx` only reaches the point of
   creating and sizing that canvas if `wasm-bindgen`'s init and Brush's own
   wgpu/WebGPU device setup succeeded. `window.onerror` and
   `unhandledrejection` listeners captured **zero** errors.
4. **The delivered `.command` launcher was smoke-tested directly** (not just
   its logic re-derived): `bash output/deliver/web-brush-stock/OPEN_WEB-BRUSH-STOCK.command`
   started the server on `127.0.0.1:8944` and `curl` confirmed 200 on
   `index.html`, the `.wasm`, and the `.ply`.

**Honest summary**: strong functional evidence (WebGPU adapter obtained, full
asset chain 200, wasm init succeeded, canvas created, zero JS errors) that the
stock Brush web viewer is rendering the synthetic splat in Safari on this
machine — but **no visual/pixel confirmation** exists, because no headless
Chrome and no permitted Safari screenshot/automation path was available in
this non-interactive session. `docs/SPEC.md`'s v0.5.0 fps acceptance
(`≥15 fps 1080p in Chrome`) is **not evaluated at all** here: Chrome is not
installed on this machine, so neither functional nor fps verification in
Chrome specifically was possible this session (see "Open questions").

## Quest, honestly (updated 2026-09-06 after the TRIPS viewer ran)

Nothing was measured on a Quest — none is here. What changed today is that the
*desktop* result now bounds the Quest question much harder than the paper
argument below did, and in the wrong direction.

The web viewer now renders trippy's **complete** frame — rasteriser and U-Net —
in Chrome on an M3 Ultra at **17.7 fps at 1440x810**, and the rasteriser alone
at 75.9 fps, measured with a Splats training on the same GPU (so lower bounds).

**That is a revision, and it cuts the other way from everything else here.**
When this section was written the numbers were 1.09 and 3.32 fps, and the
argument was that a 27x gap to the Mac app plus a mobile GPU settles Quest by
arithmetic. The 27x turned out to be the wasm *link* re-running every static
constructor on every call (blocker 5), and it is gone: the browser is now within
~1.7x of the Mac app. So the honest position is no longer "arithmetic says no";
it is "the desktop browser is fine, and the Quest question is back to being
about the Quest".

What the arithmetic still says: a Quest 3's GPU is roughly one to two orders of
magnitude slower than an M3 Ultra, so 17.7 fps here does not become an
interactive frame there — but it is now a *measurement* question rather than a
foregone conclusion, and it would need a device. Nothing here has been run on
one.

Two of the three walls below still stand, and each is enough on its own:

1. ~~**The U-Net cannot run in a browser at all**~~ — fixed (blocker 4 was
   CubeCL's autotune roofline probe, not the tensor read). It costs ~20 s of
   autotune on the first frame, once per convolution shape per page load, and
   then runs. Together with blocker 5 it turned "no finished frame at all" into
   17.7 fps.
2. **Two dependency bugs already need JavaScript shims** to get any frame out of
   a desktop browser (blockers 2 and 3), and one of the two browsers here cannot
   run the rasteriser at all (no WebGPU subgroups). Quest Browser is a third WebGPU implementation with its
   own gaps; on today's evidence the base rate for "this stack works
   unmodified in a new browser" is poor.
3. **Meta's own release notes still scope Quest's WebGPU to WebXR sessions**
   (146.0, 149.1, 150.1 — all "experimental", all tied to depth projection /
   space-warp / foveation). `trips-web` is a flat 2D canvas app that never opens
   an XR session, which is exactly the case those notes do not document. That
   research is unchanged and is kept below.

So: **still do not promise interactive Quest performance — but the reason has
changed, and it is now weaker.** Steps (a) "get the U-Net running in a desktop
browser", (b) "cache the per-frame point upload" and (c) "find what the browser
was actually spending its frame on" are all done; a desktop browser frame is
56 ms, not the ~900 ms that made this paragraph read as arithmetic. What is
left against Quest is a mobile GPU one to two orders of magnitude slower than
this one, plus walls 2 and 3, and neither of those has been measured on a
device. The shipped Quest answer therefore stays the fallback that already
exists and does not depend on WebGPU at all — distilled Gaussians via
`~/Splats/tools/publish/publish_splat.sh`, or `tools/flythrough.py` MP4s — but
"try it on a headset" is now a reasonable experiment rather than a waste of a
session.

The on-device checklist below is still the right checklist for the day (c)
arrives, and the "does it even get a WebGPU device on a flat page" question is
still the first thing to answer.

## Quest assessment (paper only — nothing measured on-device)

`docs/SPEC.md`'s Quest note and `docs/LIMITATIONS.md`'s "Quest rendering" section
both require an honest, on-paper assessment before any Quest hardware is
involved, plus a plan for what must actually be measured on a headset. No Quest
device was used in this session (none is available here); everything below is
research, not measurement.

**The fork's own claim** (`rust/brush-trips/README.md`): *"WebGPU is still an
upcoming standard, and as such, only Chrome 134+ on Windows and macOS is
currently supported."* The hosted demo repeats: *"NOTE: Only works on Chrome and
Edge. Firefox and Safari are hopefully supported soon."* Quest Browser is not
mentioned at all — the fork was never targeting Quest's browser as a supported
platform; it does list an Android *native* build path (`cargo ndk`, a separate
APK), which is a different thing from "open the web demo URL in Quest Browser."

**Meta Quest Browser WebGPU status** (web search against Meta's own developer
release notes, 2026-09-06 — see Sources; not from training-data memory, which
predates all of this):
- Meta Horizon OS Quest Browser **146.0 (2026-04-21)**: "Experimental WebGPU and
  WebXR depth projection support."
- **149.1 (2026-07-27)**: WebGPU support added for space-warp layers.
- **150.1 (2026-08-28)**: "Experimental WebGPU foveation support."
- All three release-note entries tie WebGPU to **WebXR-specific** features
  (depth projection, space-warp, foveation) — i.e., WebGPU as a compute/render
  backend *inside a WebXR session*, not confirmed as general-purpose 2D-canvas
  WebGPU for an ordinary flat webpage.
- Every mention is qualified "experimental"; the release notes give no
  indication the flag is on by default. The one community-forum thread found
  discussing "WebGPU compute into WebXR on Quest" was not accessible (403) to
  read developer first-hand reports.

**What this means for Brush specifically**: Brush's web viewer is a flat 2D
canvas app (egui rendered via wgpu → WebGPU), not a WebXR app — it does not open
an XR session, request a headset pose, or render stereo. It is exactly the kind
of "ordinary flat webpage calling `navigator.gpu`" that the Quest release notes
do **not** document support for. The documented WebGPU support is scoped to
WebXR sessions specifically. This is a material, previously-undocumented risk:
**the Quest note in `docs/SPEC.md` assumed the same web viewer that runs in
desktop Chrome would also just run in Quest Browser; that may not be true even
before asking about frame rate** — it may fail to acquire a WebGPU device at
all outside a WebXR session, independent of the ~120 GFLOP/eye/frame budget
argument already in `docs/LIMITATIONS.md`.

**What must be measured on-device (cannot be determined on paper)**:
1. Does `navigator.gpu.requestAdapter()` resolve at all in Quest Browser on a
   flat (non-WebXR) page, with and without the "webXR experimental features"
   flag in `chrome://flags` enabled? If it requires that flag, that is itself
   worth reporting plainly (a shipped Quest fallback cannot depend on a flag
   Jordan has to remember to flip).
2. If it does resolve: does the stock Brush demo (or trippy's own build) load a
   `.ply` and render anything at all (functional check before any fps number)?
3. If it renders: actual fps at whatever resolution the Quest Browser tab runs
   at (not headset-native res unless in an XR session) for (a) the stock demo's
   own sample splat and (b) a splat of comparable point count to what trippy's
   pipeline would ship.
4. Whether the CPU-side wasm (Burn/CubeCL host code, not just the GPU shader
   work) is a bottleneck on Quest's mobile SoC independent of GPU throughput.
5. Confirm the fallback path stays viable regardless: distilled Gaussians via
   `~/Splats/tools/publish/publish_splat.sh`, or `tools/flythrough.py` MP4s
   (both already exist and do not depend on Quest's WebGPU status at all).

**Recommendation**: do not promise interactive Quest performance (per
`docs/SPEC.md`'s existing instruction), and go into the on-device measurement
session expecting to test "does it load and show anything" before "how many
fps" — the WebXR-scoping of Quest's WebGPU support is a plausible reason for a
hard failure, not just a slow one.

## Next: wiring TRIPS in

Everything above proves the pipe works with the **stock** Brush renderer. Wiring
trippy's own forward pass in means, at minimum:
- `brush-pyramid`'s CubeCL kernels and `brush-unet`'s Burn graph both need to
  compile for `wasm32-unknown-unknown` too (they currently only build for the
  native target via `cargo check -p brush-pyramid -p brush-unet` and the `gpu`
  feature's desktop wgpu backend — wasm32 has not been attempted for these
  crates at all).
- The viewer hook-in point is `apps/brush-app/src/ui/splat_backbuffer.rs` (see
  `rust/README.md`'s "Status" section) — that is a native-and-wasm shared path
  in Brush's own architecture, so the hook, once written, should reach the web
  build automatically once the crates above compile for wasm32.
- `rust/brush-trips/Cargo.toml`'s wasm-specific `burn` feature list
  (`default-features = false` dropping `rl`, see the comment above the `burn`
  dependency) and the `[patch]` table pinning `cubecl`/`wgpu` forks for wasm
  compatibility (`js-interop-30`, the `msl-trial` branch) both apply to any new
  crate added to the wasm build graph — a new crate pulling in an unpatched
  transitive dependency (e.g. anything requiring `gix-tempfile`/threads/signals)
  will break the wasm build the same way `burn-rl` did.

## Open questions

- **Chrome is not installed on this Mac.** `docs/SPEC.md`'s v0.5.0 acceptance
  line names Chrome specifically (`≥15 fps 1080p in Chrome`); this session
  proved the pipeline in Safari instead, honestly, but the actual acceptance
  criterion has not been checked at all yet. Installing Chrome and re-running
  the same functional check (plus an actual fps measurement, e.g. via
  `chrome://gpu` + a frame-time overlay or a headless
  `--enable-unsafe-webgpu --use-angle=metal --screenshot=` smoke test) is
  the natural next step and does not require any new tooling beyond the
  Chrome binary itself.
- **No fps number exists yet, in any browser.** Everything measured this
  session is load/init correctness, not frame rate — deliberately, per the
  brief's "keep it brief, do not run benchmarks." A real fps number (needed
  for the v0.5.0 gate and to decide the Quest fallback) is separate follow-up
  work, and should use the real TRIPS-driven splat once that's wired in, not
  the tiny 2,000-point synthetic cloud used here (too small to load the GPU
  the way a real scene would).
- **wasm32 build coverage for `brush-pyramid`/`brush-unet` is unknown.** Only
  the stock `brush-app` was proven to compile and run for wasm32 in this
  session. Whether trippy's own CubeCL kernels and Burn U-Net graph compile
  for `wasm32-unknown-unknown` at all — separately from whether they'd be fast
  enough — is untested and is exactly the risk flagged in "Next: wiring TRIPS
  in" above.
- **Quest measurement is unstarted** (no device available this session) and,
  per the research above, may hit a hard functional wall (WebGPU scoped to
  WebXR sessions in Quest Browser's current experimental support) before frame
  rate is even a question — see "Quest assessment" above for exactly what to
  check first on-device.

Sources:
- [Release notes | Meta Horizon OS Developers](https://developers.meta.com/horizon/release-notes/web/)
- [WebXR Browser Support in 2026: What Works, What Breaks](https://www.testmuai.com/learning-hub/webxr-compatible-browsers/)
