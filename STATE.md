# STATE — externalized progress (update at end of every session)

Last updated: 2026-09-05 (session 1, Orchestrator: Claude Fable 5.1)

## Done
- Spec + plan grilled and approved 2026-09-05.
- Phase 1 skeleton reviewed and pushed as `build-0001` (public repo github.com/ggjordan/trippy, 27 CPU tests green).
- Jordan set the goal (2026-09-05 ~22:50): finish all stages autonomously; anything needing Jordan goes in the review queue below.

## In flight
- GPU-queue smoke job `trippy-smoke` (prio 15) queued behind a Hunua training; result -> ~/Splats/tools/gpu_queue/done/trippy-smoke.rc.
- feat/scene-io (mid/normal): COLMAP bin+txt loader, undistort+cache dataset, splits.
- feat/points (mid/normal): PointSource ABC, GaussianPlySource, kNN sizes, union, `trippy density`.
- docs/TRIPS_REFERENCE.md (mid/normal): porting-grade extraction from third_party/TRIPS @ a59a65b.
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
