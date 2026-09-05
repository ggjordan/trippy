# STATE — externalized progress (update at end of every session)

Last updated: 2026-09-05 (session 1, Orchestrator: Claude Fable 5.1)

## Done
- Spec + plan grilled and approved 2026-09-05.
- Phase 1 skeleton reviewed and pushed as `build-0001` (public repo github.com/ggjordan/trippy, 27 CPU tests green).
- feat/points merged (build-0004): PointSet, GaussianPlySource (5.74M pts on kk-coherent, median nn 0.080), density CLI; 50 tests green.
- v0.1.0 RELEASED on GitHub. Merged since: Brush fork submodule (ggjordan/brush trippy-fork) + crate skeletons, candidate-report (dolly/off-path/audits), se3_exp fix.
- v0.1.0 GATE PASSED 2026-09-06: horse parity 22.27 dB vs GT (authors 22.34), 36.99 dB vs authors' render (EXP-0002). Merged: backward (grads <4e-6), trainer, monodepth, render CLI (EXP-0001, EXP-0004 sheets delivered).
- Merged: raster forward (Metal = reference to 2e-6; 41.6 ms @1008x756/200k pts), net port (34/34 tensors match the public horse checkpoint with num_layers=8), scene loader, PLY export/sheets/video. build-0009, 187 tests.
- docs/TRIPS_REFERENCE.md written from source (key finding: TRIPS's shipped default broadcasts every point into all 5 layers; the 2-layer trilinear path exists but is unreachable from configs).
- Jordan set the goal (2026-09-05 ~22:50): finish all stages autonomously; anything needing Jordan goes in the review queue below.

## In flight
- GPU queue: trippy-train-smoke-2 (EXP-0003 smoke, prio 16; smoke-1 failed on an MPS float64 crop bug, fixed) and trippy-hybrid-c-render-1 (prio 17).
- fix/se3-exp-grad (mid): rotation gradient at phi=0.
- feat/hybrid-c (mid/high): design C render->photo refinement on gsrender outputs (EXP-0005).
- feat/trips-mode (large/high): native TRIPS layer rule (layers 0..ceil(log2 s)), integer pixel centres, ceil halving; parity re-run.

## Next (in order)
1. Merge scene-io + points; launch feat/raster (large/high: numpy reference + Metal blend_fwd + pyramid forward) and feat/net (mid/high: U-Net + tone mapper ports) once TRIPS_REFERENCE.md lands.
2. v0.1.0: colmap_io, xform_a/b, dataset, GaussianPlySource, ref_numpy, blend_fwd pyramid forward, U-Net tone mapper.
3. v0.2.0: blend_bwd gradcheck, trainer, MonoDepthSource, eval/export/dolly, source experiments.
4. v0.3.0: Hybrid designs C then A1, comparison harness.
5. v0.4.0: Brush fork viewer (Mac), v0.5.0 web viewer, Quest measurement.

## Blocked
- None.

## Open questions for Jordan (review queue; nothing blocks on these)
- PRIVACY INCIDENT (2026-09-05 ~23:40): the render-kk subagent opened `output/runs/EXP-0001/.../sheet.png` (a CPU dry-run contact sheet whose first panel is a kk-coherent photo) with its Read tool to sanity-check layout. Image pixels went to the model API. My task brief caused it ("look at the coverage image with the Read tool"). Fixed: AGENTS.md now forbids viewing any scene-derived imagery; all running agents were told. Nothing else left the machine. Please decide whether you want any further action.
