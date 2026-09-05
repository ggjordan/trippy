# STATE — externalized progress (update at end of every session)

Last updated: 2026-09-05 (session 1, Orchestrator: Claude Fable 5.1)

## Done
- Spec + plan grilled and approved 2026-09-05.
- Phase 1 skeleton reviewed and pushed as `build-0001` (public repo github.com/ggjordan/trippy, 27 CPU tests green).
- feat/points merged (build-0004): PointSet, GaussianPlySource (5.74M pts on kk-coherent, median nn 0.080), density CLI; 50 tests green.
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
- GPU queue (prio 70, self-reporting, in order): full2-broadcast, full2-trips (EXP-0003, 300 ep), union-broadcast, union-trips (EXP-0006), full-trips (EXP-0007 Hunua clip4982). Each delivers dolly/honesty/ply + audit table to Jordan-Review on completion.
- feat/mac-viewer (large/high): TRIPS in the Brush app (pyramid -> U-Net -> screen), perf levers to reach >= 20 fps at 1080p, OPEN_TRIPS_MAC launcher, export-bundle CLI.
- feat/distill (mid/high): design B fallback -- distil a TRIPS checkpoint into plain Gaussians with the Brush fork for Quest/publish.

## Next (in order)
1. Merge scene-io + points; launch feat/raster (large/high: numpy reference + Metal blend_fwd + pyramid forward) and feat/net (mid/high: U-Net + tone mapper ports) once TRIPS_REFERENCE.md lands.
2. v0.1.0: colmap_io, xform_a/b, dataset, GaussianPlySource, ref_numpy, blend_fwd pyramid forward, U-Net tone mapper.
3. v0.2.0: blend_bwd gradcheck, trainer, MonoDepthSource, eval/export/dolly, source experiments.
4. v0.3.0: Hybrid designs C then A1, comparison harness.
5. v0.4.0: Brush fork viewer (Mac), v0.5.0 web viewer, Quest measurement.

## Blocked
- None.

## Open questions for Jordan (review queue; nothing blocks on these)
- REVIEW: first TRIPS candidate in Jordan-Review (EXP-0003-full1-broadcast-dolly.mp4, -honesty.png, -points.ply). Numbers say: 14.4 dB, and the point cloud has MORE dark mass in the shade volume than the Gaussian baseline (36% vs 20% of region mass at lum<0.25); the network output may still look better, which is exactly the hallucination question. Your viewer verdict decides whether this direction is worth the long runs already queued.
- PRIVACY INCIDENT (2026-09-05 ~23:40): the render-kk subagent opened `output/runs/EXP-0001/.../sheet.png` (a CPU dry-run contact sheet whose first panel is a kk-coherent photo) with its Read tool to sanity-check layout. Image pixels went to the model API. My task brief caused it ("look at the coverage image with the Read tool"). Fixed: AGENTS.md now forbids viewing any scene-derived imagery; all running agents were told. Nothing else left the machine. Please decide whether you want any further action.
