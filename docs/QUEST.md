# Quest assessment (Stage 3, item 3) — status 2026-09-06

The brief asked for an honest answer, measured before promised. Nothing has been measured on a
headset yet; everything below is what the desktop numbers and Meta's own release notes bound.

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
