# Results digest for Jordan (kept current by the Orchestrator; numbers in research/trips-metal.md)

Last updated 2026-09-06 18:10. Open things in `~/Splats/output/Jordan-Review/` (`4-other/`, `2-open-in-brush/`).

## What to click first
1. `4-other/full2-broadcast-viewer-v2.command` — the 300-epoch Karekare TRIPS model, free navigation.
   R = home view, N/P = step through the capture, V = network / raw points / coverage, X = exposure mode,
   scroll = faster. Walk toward the shade under the trees.
2. `4-other/trips-leaderboard.png` — one table of every run vs the plain Gaussians.
3. `4-other/trips-mac-viewer-horse-v3.command` — the public TRIPS scene, as a reference for what a
   fully trained TRIPS looks like at 22-30 fps.
4. `4-other/full-trips-2-bc-viewer.command` — Hunua (second scene), 120 epochs.

## Where the numbers stand (Karekare kk-coherent, 219 registered photos, 33 held out incl. the 6 shade frames)
| | all held-out PSNR | shade frames PSNR | dark mass in the shade volume |
|---|---|---|---|
| Plain Gaussians (kkc_15000; trained on 5 of the 6 shade frames) | 15.53 | 14.94 | 19.9% |
| TRIPS from Gaussian centres, 300 epochs, broadcast | **17.12** | **15.27** | 36.9% |

- PSNR uses exposure borrowed from neighbouring training frames (TRIPS's own method; no held-out photo is used).
  The earlier 8.49 dB shade number was an exposure bug (10 photos without EXIF got a 58x gain).
- The dark-mass audit is the metric that tracks your complaint directly, and it still favours the Gaussians.
  Whether the shade reads as shading is your viewer verdict.

## What is queued (each self-delivers a viewer launcher + audit table when done)
trips-mode 300-epoch Karekare (resumed), alternating shade hold-out (broadcast + trips), hybrid A
(Gaussian render fused into the network; broadcast + trips), union point set (Gaussians + monocular depth;
broadcast + trips). About 4-5 hours each, in that order, behind Splats' own jobs.

## Negative results so far
- Design C (network only refines Gaussian renders): shade got worse (-2 dB). Not the fix.
- Distillation back to plain Gaussians works as a pipeline but cannot beat the checkpoint it came from.
- Quest: not interactive by any measure (docs/QUEST.md); distilled Gaussians / videos remain the Quest path.

## Engineering state
v0.4.0 released. Native Mac viewer 29.5 fps at 1080p; browser viewer 18 fps in Chrome (Safari unsupported:
no WebGPU subgroups). Rust and Python renderers agree to 115 dB on the public scene.
