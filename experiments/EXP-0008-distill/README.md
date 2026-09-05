# EXP-0008: Design-B distillation pipeline proof

## Question

docs/SPEC.md D2: "A plain splat that incorporates TRIPS learning (Design B) is a valid
fallback path, not the primary deliverable." Does the full pipeline -- render TRIPS network
output at training + near-path interpolated cameras, write a Brush-trainable COLMAP image
set, train an ordinary 3DGS model with Brush, audit the result -- run end to end without a
human stitching steps together by hand?

**This is a pipeline proof, not a quality result.** It is exercised once against the
existing weak `EXP-0003-kk-trips-train/full1-broadcast` checkpoint (40 epochs, mode
`broadcast`, held-out PSNR 14.42 dB -- docs/EXPERIMENTS.md's own review-queue note: "the
point cloud has MORE dark mass in the shade volume than the Gaussian baseline"). Any
distilled PLY this run produces inherits that checkpoint's own weakness by construction
(docs/LIMITATIONS.md "Distillation (design B)": "a distilled splat can only be as good as
the checkpoint it was distilled from") -- it proves the pipeline moves data through every
stage correctly, not that Design B is ready to ship on a good checkpoint.

## Point source

Not a `trippy.points.PointSource` -- the "points" here are the TRIPS network's own rendered
output at a dense camera set, fed to Brush as ordinary photographs. `points3D.txt`'s initial
splat cloud (Brush's own init input) is the checkpoint's trained point cloud
(`Trainer.export_ply`'s output), capped at `DISTILL_DEFAULT_MAX_INIT_POINTS` (300,000) rows.

## Configuration

- **Checkpoint**: `output/runs/EXP-0003-kk-trips-train/full1-broadcast/checkpoints/checkpoint_latest.pt`
  (scene `karekare/kk-coherent`, width 1008, mode `broadcast`, epoch 39/40, point source
  `kkc_15000.ply` min_opacity=0.05 size_mode=knn).
- **`trippy distill --stage render`**: `--interp-k 1` (one interpolated camera per
  consecutive anchor pair, rather than the CLI default of 2) -- kept lean for a pipeline
  proof run on an already-queued, contended GPU; `--max-jump-multiplier` and
  `--max-init-points` at their defaults (4x, 300,000).
- **`trippy distill --stage brush-cmd`**: `--brush-iters` at the CLI default
  (`DISTILL_DEFAULT_BRUSH_ITERS`, 6000) -- within the task's own "5k-8k steps" budget for a
  queue job that already has several long trainings ahead of it (see "GPU queue state"
  below). `--sh-degree 0`, `--max-resolution 1008` (the render width), `--eval-split-every 8`.
- **Brush binary**: `rust/brush-trips/target/release/brush-cli`, built via `scripts/
  cpu_heavy.sh` (see "Planned commands").

## Planned commands

```bash
# 1. Build the Brush CLI binary (CPU-heavy queue, not the GPU queue).
bash scripts/cpu_heavy.sh brush-cli-build -- bash -c \
  'cd rust/brush-trips && cargo build --release -p brush-cli'

# 2. Render the TRIPS network output at the training + near-path interpolated cameras,
#    and write the Brush-trainable COLMAP image set (GPU work, prio 15).
TRIPPY_OUTPUT=/Users/nzbirdranch/trippy/output scripts/gpu_submit.sh --prio 15 \
  distill-render-full1-broadcast -- trippy distill \
  --checkpoint /Users/nzbirdranch/trippy/output/runs/EXP-0003-kk-trips-train/full1-broadcast/checkpoints/checkpoint_latest.pt \
  --out /Users/nzbirdranch/trippy/output/runs/EXP-0008-distill/full1-broadcast \
  --stage render --device mps --interp-k 1

# 3. Print the Brush training command + write the queue-ready job script (CPU-only,
#    no GPU work -- just string/file building).
trippy distill \
  --checkpoint /Users/nzbirdranch/trippy/output/runs/EXP-0003-kk-trips-train/full1-broadcast/checkpoints/checkpoint_latest.pt \
  --out /Users/nzbirdranch/trippy/output/runs/EXP-0008-distill/full1-broadcast \
  --stage brush-cmd --brush-iters 6000

# 4. Queue the Brush training itself (GPU work, prio 70 -- behind Splats' own jobs and
#    every trippy training already queued, D9). Printed by step 3; pasted here once run.

# 5. Audit: baseline (kkc_15000.ply) vs TRIPS export vs distilled (once step 4 returns).
trippy distill \
  --checkpoint /Users/nzbirdranch/trippy/output/runs/EXP-0003-kk-trips-train/full1-broadcast/checkpoints/checkpoint_latest.pt \
  --out /Users/nzbirdranch/trippy/output/runs/EXP-0008-distill/full1-broadcast \
  --stage compare \
  --baseline-ply /Users/nzbirdranch/Splats/output/Training-Data/karekare/kk-coherent/kkc_15000.ply \
  --distilled-ply /Users/nzbirdranch/trippy/output/runs/EXP-0008-distill/full1-broadcast/brush_out/distilled_6000.ply
```

## GPU queue state at submission (2026-09-06, ~05:57 local)

The queue already had a `60-hunua-clip5127-train` job running and five prio-70 trippy
trainings queued (`full-trips`, `full2-broadcast`, `full2-trips`, `union-broadcast`,
`union-trips`) plus two prio-30 sfm jobs, ahead of anything newly submitted at prio 70. The
render job (prio 15) queues ahead of all of those except the one already-running job. This
section is updated with actual rc/timing once each job returns; see `research/trips-metal.md`
for the running log.

## Gate

Not a stage gate (Design B is an explicit fallback, D2) -- the deliverable is a working
pipeline plus an honest read of the audit numbers on this one weak checkpoint.

## Results

**Camera plan (CPU-only, no render, sanity check before submitting the GPU job)**:
`build_distill_camera_plan("~/Splats/scenes/karekare/kk-coherent", width=1008, k=1)` on the
real scene -- 219 anchors (matches docs/EXPERIMENTS.md's "6 OPENCV cameras, 219 registered
images"), 203 interpolated poses, 15 of 218 consecutive pairs skipped (all 15 for
"different camera_id" -- kk-coherent's calibration splits the single physical lens into
several COLMAP camera groups across the shoot; the honesty-guard distance check never
triggered on this scene: median consecutive distance 0.298 m, jump threshold 1.191 m, and
every same-camera consecutive pair falls under that threshold). 422 total poses to render.

**Pipeline build**: `trippy-brush-cli-build` (`scripts/cpu_heavy.sh`, not the GPU queue)
rc=0, 2m44s -- `rust/brush-trips/target/release/brush-cli` built. 35 new CPU tests (5
colmap_io writer round-trip tests + 30 across `trippy/distill/*`) plus the existing suite:
670 CPU tests green, `ruff check` clean, `scripts/build.sh`/`scripts/test.sh` both green.

**GPU jobs**: filled in below as each returns.
