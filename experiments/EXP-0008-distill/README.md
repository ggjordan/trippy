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
670 CPU tests green at merge time (679 after rebasing onto `origin/main` post-merge),
`ruff check` clean, `scripts/build.sh`/`scripts/test.sh` both green.

**Job 1 -- render** (`trippy-distill-render-full1-broadcast`, prio 15, `--device mps
--interp-k 1`): **rc=0**, ~06:44:07-06:47:17 (**~3m10s** for 422 frames at 1008 wide through
the pyramid + U-Net + tone mapper on MPS -- queue wait beforehand was longer than the render
itself, behind the already-running hunua job plus one higher-priority `brush-unet` job that
arrived after submission). Log: `rendered 219 anchor + 203 interpolated cameras (15 pair(s)
skipped by the honesty guard)`. `distill_report.json`:

| Field | Value |
|---|---|
| n_anchor_images | 219 |
| n_interpolated_images | 203 |
| n_skipped_pairs | 15 (all "different camera_id" -- kk-coherent's 6 COLMAP camera groups; the jump-distance guard never triggered: median 0.298 m, threshold 1.191 m) |
| n_cameras (COLMAP cameras written) | 6 |
| n_points_source (TRIPS export) | 5,736,619 |
| n_points_written (points3D.txt, capped) | 300,000 |
| mean_coverage_full (raw, level-0, T_final-derived) | 0.2916 |

Artifacts: `output/runs/EXP-0008-distill/full1-broadcast/{trips_export.ply (372 MB),
images/ (422 PNGs), sparse_txt/{cameras,images,points3D}.txt, renders/, distill_report.json}`.

**Job 2 -- Brush training** (`trippy-distill-full1-broadcast`, prio 70, `--total-train-iters
6000 --sh-degree 0 --max-resolution 1008 --eval-split-every 8 --eval-every 1000
--export-every 6000 --export-path .../brush_out/ --export-name distilled_{iter}.ply --seed
0`): submitted (submit.sh rc=0), queued behind Splats' own jobs and two hunua/sfm jobs
already ahead of it (it landed *ahead* of the five pre-existing prio-70 trippy trainings on
an alphabetical tie-break within the priority band) -- **rc=0**. Brush's own log: training
loop `857s` (14m17s) for 6000 iterations, growing from the ~5.7M-point TRIPS-export init to
**5,995,586** final splats; Brush's own held-out eval (its own `--eval-split-every 8` split
of the *rendered* image set, i.e. "how well did Brush reproduce the TRIPS network's own
renders", not a real-photo fidelity number) climbed from PSNR 23.51/SSIM 0.799 at iter 1000
to PSNR 24.69/SSIM 0.866 at iter 6000 -- Brush fit the rendered image set well; this says
nothing about the underlying checkpoint's own quality. Output:
`output/runs/EXP-0008-distill/full1-broadcast/brush_out/distilled_6000.ply` (336 MB).

**Audit comparison** (`trippy distill --stage compare`, baseline = `kkc_15000.ply`, TRIPS
export = this run's own `trips_export.ply`, distilled = `distilled_6000.ply`):

| Metric | baseline | TRIPS export | distilled |
|---|---|---|---|
| Point count | 7,364,913 | 5,736,619 | 5,995,586 |
| Shade dark-mass fraction (lum<0.25) | 19.9% | 36.2% | 37.0% |
| Extent radius p99 | 52.21 | 40.02 | 39.65 |
| Extent radius max | 133.35 | 124.48 | 161.36 |

**Honest read**: this matches the pre-existing EXP-0003 review-queue finding (STATE.md,
2026-09-05) exactly -- the TRIPS export already has *more* dark mass in the shade region
than the plain-Gaussian baseline (36.2% vs 19.9%), i.e. by this metric the checkpoint being
distilled here had **not** fixed the shade cloud before distillation even started. The
distilled PLY carries that same defect through almost unchanged (37.0%, +0.8 points on the
TRIPS export it was distilled from) -- Brush neither fixed nor meaningfully worsened the
shade dark-mass fraction, which is exactly what "distillation, not compression" predicts
(docs/EXPERIMENTS.md "Distillation (design B)": a distilled splat can only be as good as the
checkpoint it came from). Point count grew modestly (5.736M init -> 5.996M, +4.5%) from
Brush's own densification. Extent p99 held roughly steady (40.02 -> 39.65); extent **max**
grew past *both* the TRIPS export and the original Gaussian baseline (124.48 -> 161.36,
above baseline's own 133.35) -- Brush's refine/growth loop has no equivalent of trippy's own
soft extent-penalty toward the initial bbox (`trippy.train.trainer._extent_penalty`), so a
handful of splats grew out past the union of every other cloud's own extent; worth watching
if this pipeline runs again on a scene where sprawl matters more.

**Because the input checkpoint already scored worse than its own baseline on the one metric
this project gates on, the distilled column cannot be read as evidence Design B fixes the
shade cloud** -- garbage in, garbage out is exactly the expected and observed outcome here.
The point of this run was solely to prove steps 1-3 of the pipeline move data through
correctly end to end, which they did: render (rc=0, 3m10s) -> Brush training (rc=0, 14m17s)
-> audit comparison, no manual stitching in between beyond the two `scripts/gpu_submit.sh`
calls the pipeline itself prints.

**This is a pipeline proof only, run against a known-weak checkpoint** (EXP-0003
full1-broadcast, 40 epochs, 14.42 dB held-out, already flagged in the review queue as *not*
having removed the shade cloud). The delivered `distilled_6000.ply` is **not a candidate for
Jordan's review as a scene fix** -- it demonstrates the pipeline, nothing about shade
quality. A real Design-B candidate needs a checkpoint that has actually passed the v0.2.0
stop-or-go gate first. Delivered via `scripts/deliver.sh` (linked into
`~/Splats/output/Jordan-Review/2-open-in-brush/EXP-0008-distill-full1-broadcast.ply`) with
that same "pipeline proof, not a candidate" line as its "why".

## How this would reach Splats' publish path (documented, not run)

`~/Splats/tools/publish/publish_splat.sh <in.ply> <name> [images.txt]` was read (read-only,
never executed per this task's constraints). The delivered `distilled_6000.ply` hands to it
directly as `<in.ply>` -- it is already an ordinary 3DGS-compatible PLY, Brush's own export
format, exactly what that script expects (no SH-stripping needed since `--sh-degree 0` never
wrote higher-order coefficients to begin with). This run's own
`sparse_txt/images.txt` (COLMAP text, world-to-camera quat+t, written by
`trippy.distill.colmap_writer`) is exactly the optional `[images.txt]` argument
`publish_splat.sh` uses to derive a start camera at the camera-centroid looking along the
mean forward direction -- so the full invocation, if this were a real (non-proof) candidate,
would be:

```bash
~/Splats/tools/publish/publish_splat.sh \
  output/runs/EXP-0008-distill/full1-broadcast/brush_out/distilled_6000.ply \
  karekare-kk-distill-<name> \
  output/runs/EXP-0008-distill/full1-broadcast/sparse_txt/images.txt
```

Not run here: this is a pipeline proof from a known-weak checkpoint, not a publishable
candidate (see "Honest read" above).
