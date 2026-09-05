# Web viewer (v0.5.0 groundwork)

Status as of 2026-09-06: the **toolchain is proven end to end** on this Mac — the
stock Brush fork's web app builds from the `rust/brush-trips` submodule to a wasm
bundle, serves from a `.command`-launched local web server on 127.0.0.1, and loads
a real (synthetic) `.ply` splat in a browser with WebGPU active. Dropping trippy's
own TRIPS render output into this pipeline instead of the stock Brush renderer is a
**separate, later task** (see "Next: wiring TRIPS in" below); nothing here changes
`brush-pyramid`/`brush-unet` or `splat_backbuffer.rs`.

This is groundwork for `docs/SPEC.md`'s v0.5.0 row: *"Web viewer via
`apps/brush-app/web` (wasm-pack + vite), `OPEN_TRIPS_WEB.command` on 127.0.0.1"*.

## Build

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
