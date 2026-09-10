# Quest assessment (Stage 3, item 3) — status 2026-09-06

The brief asked for an honest answer, measured before promised. Nothing has been measured on a
headset yet; everything below is what the desktop numbers and Meta's own release notes bound.

**Update 2026-09-09**: `docs/decisions/ADR-0008-supersplat.md` Stage 1 (self-hosted SuperSplat)
is done; next up is that ADR's Stage 3 — SOG export + the self-hosted `supersplat-viewer`
package, which is the concrete answer to "what ships to the Quest instead" below.

**Update 2026-09-10 — Stage 3 tooling built, real conversion running in the GPU queue.**
See "Stage 3: SOG export + self-hosted WebXR viewer" below for the actual path Jordan uses.
Short version: `scripts/sog_export.sh` compresses a splat PLY to `.sog` (+ a `.compressed.ply`
second artefact) with `@playcanvas/splat-transform`, `scripts/quest_viewer_bootstrap.sh` turns
the `.sog` into a self-hosted static viewer bundle (the same npm package @playcanvas/supersplat
itself uses), and `scripts/open_quest_viewer.sh` writes two double-click launchers per splat —
a Mac preview and a Quest one (LAN + self-signed HTTPS, required for WebXR off `localhost`).
Two real conversions are queued as of 2026-09-10 (see that section for job names and why the
first attempt was killed); this page will get real sizes/counts once they land.

## Stage 3: SOG export + self-hosted WebXR viewer

### What ships
- **`scripts/sog_export.sh <input.ply>`** — compresses a Gaussian-splat PLY to `.sog` (the
  compact format the browser viewer streams) and, as a second artefact, a `.compressed.ply`,
  using `@playcanvas/splat-transform` pinned to **3.3.3** (the exact version SuperSplat 3.0.0
  itself devDependency-pins, confirmed by reading its `package.json` after Stage 1's checkout).
  Installed locally into `$TRIPPY_OUTPUT/tools/splat-transform` — gitignored, never global, never
  vendored. Verifies the round-trip: converts the `.sog` back to a scratch PLY and compares
  `numGaussians` across input / `.sog` / `.compressed.ply` / round-trip before calling it done.
- **`scripts/quest_viewer_bootstrap.sh <input.sog-or-ply> <name>`** — builds the actual viewer
  page. Rather than separately cloning and building `@playcanvas/supersplat-viewer`, this calls
  splat-transform's own `.html --unbundled` output, which is generated (at splat-transform's
  build time) by that exact npm package's `renderViewerHtml` — confirmed by reading
  splat-transform 3.3.3's `package.json` (`"@playcanvas/supersplat-viewer": "1.30.2"` as a
  devDependency) and finding `renderViewerHtml`/`supersplat-viewer` strings inside its bundled
  `dist/cli.mjs`. One pinned tool produces both the `.sog` and the WebXR-capable viewer, with no
  second build pipeline to keep in sync — both packages are MIT (ADR-0008 Sec 3's licence table).
  Output is a flat 5-file static directory: `index.html`, `index.js`, `index.css`,
  `settings.json`, `index.sog`.
- **`scripts/open_quest_viewer.sh <bundle-dir> <name>`** — writes two double-click launchers per
  bundle:
  - `OPEN_<NAME>.command` — binds `127.0.0.1` only, plain http, opens the Mac's own browser.
    Good for a quick look, but **cannot** show the VR entry button (see below).
  - `OPEN_<NAME>_ON_QUEST.command` — binds `0.0.0.0` (this Wi-Fi network only) over **HTTPS with
    a self-signed certificate generated on the fly**, prints `https://<mac-lan-ip>:<port>/?webgl`
    to open on the Quest browser, and runs in the foreground so it's obvious when it's serving
    (Ctrl-C, or close the window, stops it).
  - Both launchers can be generated **before** the bundle exists (wired to the path a queued
    conversion job will populate) — each one checks for `index.html` + (`index.sog` or
    `index.compressed.ply`) at run time and refuses with a plain "Not ready yet" message instead
    of opening a broken page.

### Why HTTPS is required for the Quest but not for the Mac preview
The WebXR Device API (`navigator.xr`) is only exposed in a **secure context** per the W3C spec,
and Chromium (which the Quest/Horizon OS browser is built on) enforces this at the engine level
regardless of what the page does. `127.0.0.1`/`localhost` is exempted from that rule; a LAN IP
(what the Quest sees this Mac as) is not. Plain http to the LAN IP would load the viewer but hide
the VR entry button entirely — a silent dead end, not an error. This is documented browser/spec
behaviour, established by reading the spec and the built viewer's own code
(`url.searchParams.has('webgl') ? 'webgl' : 'webgpu'` — the viewer defaults to WebGPU, and its own
VR entry button only appears under the WebGL renderer, which is why the printed URL always
includes `?webgl`), **not something that needed a headset to discover.**

The self-signed certificate is cached under `$TRIPPY_OUTPUT/tools/quest-viewer/certs/` (shared
across every bundle) and regenerated automatically only when the Mac's current LAN IP changes
(e.g. a different Wi-Fi network). **Jordan will see a "this connection is not private" warning on
the Quest browser the first time** (and again any time the cert regenerates) — that is expected
for a self-signed cert made on this Mac, not a sign of a problem. Accept it once per network
("Advanced" → "Proceed to `<ip>` (unsafe)" or the Quest browser's equivalent wording) and the
viewer loads normally after that.

### How to use it (once a bundle is built)
1. Double-click the `..._ON_QUEST.command` launcher in `~/Splats/output/Jordan-Review/4-other/`.
2. It prints something like `https://192.168.1.97:8968/?webgl` and a reminder that it's only
   reachable on your home network while the window stays open.
3. Put on the Quest, open its browser, type that exact URL (the `?webgl` matters).
4. Accept the one-time certificate warning.
5. Enter VR from the viewer's own UI.
6. When done, close the Terminal window on the Mac (or Ctrl-C) to stop serving.
7. Report back: does the frame rate feel comfortable, does the splat look right, and did the VR
   button actually appear (confirms the `?webgl` + HTTPS + secure-context reasoning above held up
   on real hardware, not just on the spec).

### The two real conversions (2026-09-10)
Two splats were queued for conversion: `kklid-tripsclean-shade-keep.ply` (the TRIPS-cleaned splat,
2,016,542,793 bytes, 8,544,666 Gaussians) and the untouched `kklid_20000.ply` (2,102,851,769
bytes) for comparison. **First attempt (CPU-only, `-g cpu`) ran 14 hours with the SOG's scratch
`.tmp` file static at 78 MB and made no further progress — killed** (pid 78171, `.tmp` file
removed). Splat-transform's own `--gpu <n>` device flag uses its native WebGPU (Dawn) backend
instead, confirmed working locally (`--list-gpus` enumerates the M3 Ultra without needing to
approve the tool's blocked npm postinstall script, since that script only fetches a binary this
repo never needs when running CPU-only). Both conversions were resubmitted through
`scripts/gpu_submit.sh --prio 40` (short-checks band; Splats' own queue had drained) using `--gpu
0`, as `bash -c` jobs chaining `sog_export.sh` then `quest_viewer_bootstrap.sh`:
- `trippy-quest-sog-shade-keep` — log: `~/Splats/tools/gpu_queue/logs/trippy-quest-sog-shade-keep.log`
- `trippy-quest-sog-kklid20000` — log: `~/Splats/tools/gpu_queue/logs/trippy-quest-sog-kklid20000.log`

**Done 2026-09-10 12:35** (both jobs rc=0, run on the GPU in minutes after a CPU attempt made no progress in 14 h):

| splat | Gaussians in | `.sog` | `.compressed.ply` | round trip |
|---|---:|---:|---:|---:|
| kklid_20000 (untouched) | 8,910,382 | 119 MB | 546 MB | 8,910,382 |
| kklid-tripsclean-shade-keep | 8,544,666 | 114 MB | ~520 MB | 8,544,666 |

Both `quest-viewer-*-quest` launchers in Jordan-Review/4-other are live now.

## What we know
- The full TRIPS frame (pyramid + U-Net + camera) costs ~34 ms at 1440x810 on the M3 Ultra in the
  native viewer (29 fps) and ~55 ms in Chrome on the same machine (18 fps) after the wasm fixes.
  The U-Net is ~89% of a native frame (docs/ARCHITECTURE.md, research/trips-metal.md).
- A Quest 3 GPU delivers well under one tenth of this machine's compute. Two eyes at headset
  resolution multiply the work again. A complete TRIPS frame on the headset is therefore in the
  hundreds of milliseconds by arithmetic alone, before any browser overhead.
- Meta's Horizon OS release notes (2026) scope WebGPU on the Quest browser to WebXR sessions and mark
  it experimental. trips-web is a flat 2D-canvas WebGPU app; that path is not documented as
  supported. Safari on the Mac already cannot run our radix sort (no WebGPU subgroups); a third
  WebGPU implementation is a poor bet without measuring.
- The rasteriser alone (raw level-0, no network) runs at ~100 fps natively and ~76 fps in Chrome, so
  a point-only view might be reachable on a headset, but that view has holes by design.

## Verdict
Do not promise interactive TRIPS on the Quest. Sequence: (1) get a Karekare candidate that passes the
shade verdict; (2) only then spend a headset session measuring the raw view in the Quest browser
inside a WebXR session; (3) treat the network view as out of reach for this hardware generation.

## What ships to the Quest instead (both paths exist today)
- Design B distilled Gaussians: `trippy distill` (EXP-0008 proved the pipeline end to end) produces a
  plain 3DGS ply that goes through Splats' existing publish path (`~/Splats/tools/publish/`) exactly
  like any other splat. Quality cannot exceed the TRIPS checkpoint it came from.
- Fly-through videos of the TRIPS render along any camera path (`trippy candidate-report` dolly
  machinery; Splats' `tools/flythrough.py` for Gaussians).
