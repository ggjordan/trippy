# STATE — externalized progress (update at end of every session)

Last updated: 2026-09-06 (web-perf session, Orchestrator: Claude Fable 5.1)

## Done
- **feat/web-perf (2026-09-06): the browser viewer was 27x slower because of the LINKER, not the renderer.** `wasm-ld` wrapped every export in a `.command_export` shim that re-runs the whole `.init_array`; `cubecl-ir` -> `pliron` makes one such run cost ~110 us, and `wasm-bindgen` resolves `__externref_table_alloc` by export name, so every `JsValue` `wgpu` built for a bind group paid it -- ~2,500 constructor runs a frame, 275 ms of a 297 ms frame. `trips-web/build.rs` now links with `--export=__wasm_call_ctors`; `web/trips.js` runs them once and refuses without the export. **Chrome 1440x810: `raw level-0` 3.32 -> 75.9 fps, `network` 1.09 -> 17.7 fps, readback PSNR 62.04 -> 104.54 dB.** Safari re-diagnosed: "Expected 'f16'" was never about f16 -- Safari 26.6.2 has no WebGPU subgroups in any form, so `brush-sort`'s four radix kernels cannot compile; the page now refuses with the exact kernels and builtins, checked on `adapter.features`, not the user agent. Launcher `trips-web-viewer-horse` re-delivered.
- Spec + plan grilled and approved 2026-09-05.
- Phase 1 skeleton reviewed and pushed as `build-0001` (public repo github.com/ggjordan/trippy, 27 CPU tests green).
- feat/points merged (build-0004): PointSet, GaussianPlySource (5.74M pts on kk-coherent, median nn 0.080), density CLI; 50 tests green.
- Web viewer fixed (PR #28): 76 fps raw / 18 fps network in Chrome at 1440x810 (was 3.3 / 1.1); cause was the wasm linker re-running static constructors on every export call. 104.5 dB vs native. Safari: no WebGPU subgroups at all -> page refuses. NOTE: that agent bypassed cpu_heavy.sh's 28 GB memory guard twice under its own watchdog while a training held memory; logged, not repeated.
- Leaderboard merged (PR #27): `trips-leaderboard.png` in Jordan-Review, regenerated after every run.
- v0.3.0 RELEASED (web viewer milestone): full pipeline incl. U-Net in Chrome (62 dB vs native, ~1 fps; 27x gap under investigation in feat/web-perf); Mac viewer 29.5 fps after the point-upload cache. EXP-0008 distillation proven end to end on the weak checkpoint (distilled ply in 2-open-in-brush; dark mass 37% vs 36% TRIPS export vs 20% Gaussians; pipeline proof only).
- Viewer v2 merged (PR #22): drag/orbit/pan/scroll, scene-scaled speed, R/N/P/F keys; launchers `trips-mac-viewer-horse-v2.command` and `trips-mac-viewer-karekare-full1.command` in Jordan-Review.
- v0.5.0 (merged, PR #25): the full TRIPS pipeline incl. the U-Net renders in Chrome via WebGPU (62 dB vs native; ~1.1 fps network / 3.3 fps raw at 1440x810 measured while a Splats training held the GPU; ~20 s first-frame autotune). Safari still draws wrong output and is blocked by the page. Mac viewer after the point-upload cache: shipped preset 29.5 fps, raw 102 fps at 1080p. Chrome was installed as dev tooling. Launcher `trips-web-viewer-horse` delivered. Quest: not interactive by any measure; distilled Gaussians / videos remain the Quest path.
- v0.2.0 RELEASED: native Mac viewer at 22 fps (horse scene), 82.7 dB screenshot parity; launcher `trips-mac-viewer-horse.command` in Jordan-Review. Finding: the U-Net is 89% of frame time, the rasteriser 11%.
- Merged brush-unet (build-0032): Rust pyramid + U-Net + camera on wgpu match the PyTorch path at 115 dB on the horse scene; whole frame 193 ms at 1080p (5 fps, sort-dominated) -> perf work in feat/mac-viewer.
- Merged: self-reporting trainings (build-0028), union point set (EXP-0006: monodepth 3.79M, union 5.89M), web toolchain. Baseline shade audit on kkc_15000: dark(lum<0.25) 19.9% of region mass.
- EXP-0003 full1-broadcast (40 ep, 11 min): held-out 14.42 dB / SSIM 0.39 / LPIPS 0.51, still rising -> long runs queued. Hybrid C (EXP-0005) negative for shade (-1.96 dB). brush-pyramid CubeCL port merged (GPU parity 2e-6). Raster NaN guard merged.
- Trainer fixed (smoke 12.26 dB), native trips mode merged, candidate-report merged. Finding: with Gaussian-scale sizes the U-Net invents ~90% of every frame (t_final 0.93); full runs use kNN sizes.
- v0.1.0 RELEASED on GitHub. Merged since: Brush fork submodule (ggjordan/brush trippy-fork) + crate skeletons, candidate-report (dolly/off-path/audits), se3_exp fix.
- v0.1.0 GATE PASSED 2026-09-06: horse parity 22.27 dB vs GT (authors 22.34), 36.99 dB vs authors' render (EXP-0002). Merged: backward (grads <4e-6), trainer, monodepth, render CLI (EXP-0001, EXP-0004 sheets delivered).
- Merged: raster forward (Metal = reference to 2e-6; 41.6 ms @1008x756/200k pts), net port (34/34 tensors match the public horse checkpoint with num_layers=8), scene loader, PLY export/sheets/video. build-0009, 187 tests.
- docs/TRIPS_REFERENCE.md written from source (key finding: TRIPS's shipped default broadcasts every point into all 5 layers; the 2-layer trilinear path exists but is unreachable from configs).
- Jordan set the goal (2026-09-05 ~22:50): finish all stages autonomously; anything needing Jordan goes in the review queue below.

## In flight
- GPU queue: full2-broadcast (EXP-0003, epoch ~240/300, plateau ~15.1 dB) -> self-report + leaderboard; then web-perf-parity (prio 12), shade-split-eval-1 (prio 15, backfills full1-broadcast's shade split), then prio-70: full-trips (Hunua clip5923), full2-trips, hybrid-a-all-levels, union-broadcast, union-trips.
- No subagents running. Next merges happen when runs report.

## Next (in order)
1. Merge scene-io + points; launch feat/raster (large/high: numpy reference + Metal blend_fwd + pyramid forward) and feat/net (mid/high: U-Net + tone mapper ports) once TRIPS_REFERENCE.md lands.
2. v0.1.0: colmap_io, xform_a/b, dataset, GaussianPlySource, ref_numpy, blend_fwd pyramid forward, U-Net tone mapper.
3. v0.2.0: blend_bwd gradcheck, trainer, MonoDepthSource, eval/export/dolly, source experiments.
4. v0.3.0: Hybrid designs C then A1, comparison harness.
5. v0.4.0: Brush fork viewer (Mac), v0.5.0 web viewer, Quest measurement.

## Blocked
- None.

## Open questions for Jordan (review queue; nothing blocks on these)
- REVIEW: open `trips-mac-viewer-horse.command` (4-other) to step into the public horse scene rendered live by TRIPS on this Mac; V toggles network/raw/coverage.
- Disk cleanup 2026-09-06 12:15 (Jordan's request): 49 -> 71 GB free. Removed re-downloadable Zenodo zips, the fork's rebuildable target dir, smoke runs, distillation intermediates, and old epoch checkpoints (kept latest/best-so-far/ep0000). Trainer retention policy in progress (fix/ckpt-retention) so 300-epoch runs stop writing ~24 GB each.
- BUG (Jordan 14:00): `full2-broadcast-viewer.command` shows a near-white frame with speckles for Karekare while the Python path renders it at 15 dB -> export/viewer camera-model or f16 path wrong for this scene (fix/viewer-kk, numbers-only parity). Also: default fly speed too slow -> 4x base, 50x scroll range.
- STOP-OR-GO SIGNAL (2026-09-06 13:50): plain TRIPS from Gaussian centres does NOT fix the Karekare shade. full2-broadcast (300 ep): held-out all 15.02 dB vs Gaussians 15.53; held-out SHADE frames 8.49 dB vs Gaussians 14.94; dark mass in the shade volume 36.9% vs 19.9%. The network invents the shade (it never sees a shade photo: all 6 are held out) and does it badly. Remaining hopes in the queue: full2-trips, union (more geometry in the shade), hybrid A (Gaussian render as input). Also testing whether held-out exposure calibration and an alternating shade hold-out change the picture (feat/eval-calib).
- REVIEW (auto-delivered): EXP-0003 full2-broadcast (300 epochs): `full2-broadcast-viewer.command` (walk it yourself), -dolly.mp4, -honesty.png. Held-out 15.02 dB (plain Gaussians 15.53). Its audit/leaderboard step failed on a mid-run code update (fixed: eager imports); a re-report job is queued and will refresh the leaderboard with shade dark-mass + shade PSNR.
- Jordan 2026-09-06 07:20: EXP-0003 first candidate is "a good starting point"; the dolly direction is not one he cares about (hard to compare) -> deliver Karekare bundles for the free-navigation viewer from now on; his main interest is Splats combined with TRIPS (hybrid). No queue jump wanted; keep prio 70.
- LOST ARTEFACT: EXP-0005's Gaussian renders + design-C checkpoint were inside a removed worktree (before output routing was fixed); the dangling review links (EXP-0005 sheet, stock web viewer) were removed; their README rows remain as history. Numbers survive in the README/research log; renders are being regenerated for EXP-0009. Guard added: scripts/worktree_rm.sh.
- PRIVACY INCIDENT (2026-09-05 ~23:40): the render-kk subagent opened `output/runs/EXP-0001/.../sheet.png` (a CPU dry-run contact sheet whose first panel is a kk-coherent photo) with its Read tool to sanity-check layout. Image pixels went to the model API. My task brief caused it ("look at the coverage image with the Read tool"). Fixed: AGENTS.md now forbids viewing any scene-derived imagery; all running agents were told. Nothing else left the machine. Please decide whether you want any further action.
