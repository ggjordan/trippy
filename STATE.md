# STATE — externalized progress (update at end of every session)

Last updated: 2026-09-05 (session 1, Orchestrator: Claude Fable 5.1)

## Done
- Spec + plan grilled and approved 2026-09-05.
- Phase 1 skeleton reviewed and pushed as `build-0001` (public repo github.com/ggjordan/trippy, 27 CPU tests green).
- feat/points merged (build-0004): PointSet, GaussianPlySource (5.74M pts on kk-coherent, median nn 0.080), density CLI; 50 tests green.
- docs/TRIPS_REFERENCE.md written from source (key finding: TRIPS's shipped default broadcasts every point into all 5 layers; the 2-layer trilinear path exists but is unreachable from configs).
- Jordan set the goal (2026-09-05 ~22:50): finish all stages autonomously; anything needing Jordan goes in the review queue below.

## In flight
- GPU-queue smoke job `trippy-smoke` (prio 15) queued behind a Hunua training; result -> ~/Splats/tools/gpu_queue/done/trippy-smoke.rc.
- feat/scene-io (mid/normal): COLMAP bin+txt loader, undistort+cache dataset, splits.
- feat/raster (large/high): layer-factor formula, fragment emission, sort/segments, Metal blend_fwd, numpy + torch references, GPU tests via queue.
- feat/net (mid/high): U-Net, gated conv, neural camera, losses, checkpoint loader.
- Background downloads: third_party/zenodo/tt_checkpoints.zip (2.7 GB), tt_scenes.zip (3.2 GB).

## Next (in order)
1. Merge scene-io + points; launch feat/raster (large/high: numpy reference + Metal blend_fwd + pyramid forward) and feat/net (mid/high: U-Net + tone mapper ports) once TRIPS_REFERENCE.md lands.
2. v0.1.0: colmap_io, xform_a/b, dataset, GaussianPlySource, ref_numpy, blend_fwd pyramid forward, U-Net tone mapper.
3. v0.2.0: blend_bwd gradcheck, trainer, MonoDepthSource, eval/export/dolly, source experiments.
4. v0.3.0: Hybrid designs C then A1, comparison harness.
5. v0.4.0: Brush fork viewer (Mac), v0.5.0 web viewer, Quest measurement.

## Blocked
- None.

## Open questions for Jordan (review queue; nothing blocks on these)
- None yet.
