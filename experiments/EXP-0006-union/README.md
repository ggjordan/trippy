# EXP-0006: TRIPS training on kk-coherent (point source 3, Union(Gaussian, MonoDepth))

## Question

Trained end to end on kk-coherent starting from the **union** of point source 1 (trained
3DGS Gaussian centres) and point source 2 (Apple DepthPro monocular depth, all 219
registered images, not just the 12-image sample from EXP-0004), voxel-deduped against
each other -- does adding dense monocular-depth geometry on top of the Gaussians change
held-out PSNR and, more importantly, does it make the shade region under the trees read
as *shading* rather than a cloud? This is the third of the three point-source experiments
(1 = Gaussians, EXP-0003; 2 = monocular depth alone, EXP-0004; 3 = union, this experiment)
the v0.2.0 stop-or-go gate compares (docs/SPEC.md "Stop-or-go point: v0.2.0").

## Point source

3 = `UnionSource` (`trippy.points.union.UnionSource`, reached via
`trippy.train.config.PointSourceConfig(type="union")`) of:

- Point source 1: `GaussianPlySource` on
  `$SPLATS_ROOT/output/Training-Data/karekare/kk-coherent/kkc_15000.ply`, `min_opacity=0.05`,
  `size_mode="knn"` (matching EXP-0003's full2 configs -- Gaussian-scale sizes leave
  `t_final~0.93` at 1008 wide, so the trained scale is not used as the initial splat size).
- Point source 2: `MonoDepthSource` over **all 219** registered kk-coherent images (not
  the 12-image sample from EXP-0004), `width=1008`, `stride=6`, `voxel=0.03`,
  `conf0=0.35`, `scale_mode="median_ratio"` (all defaults).
- Union voxel: `0.03` (same cell size MonoDepthSource already dedupes its own points at,
  applied a second time across the two sources so a Gaussian centre and a MonoDepth point
  landing in the same cell keep only the higher-`conf0` one -- Gaussian `conf0` values
  from `sigmoid(opacity)` are typically well above MonoDepth's fixed `0.35`, so Gaussian
  points win essentially every collision; see the provenance histogram below).

Built offline (not at train start) via `trippy points-build` against
`output/points/union_source_config.yaml`, because `size_mode="knn"` on the full 5.74M-row
Gaussian PLY runs a k-d-tree query for every point -- CPU-heavy enough that AGENTS.md's
"one heavy CPU job at a time" rule applies (`scripts/cpu_heavy.sh`), and re-running it at
the start of every training job would be wasteful when the result is deterministic.
Training configs below load the saved result with `point_source: {type: npz, path:
output/points/kk-coherent-union-full.npz}` -- a sub-second load instead of a multi-minute
rebuild.

## DepthPro run: all 219 registered images

Extends EXP-0004 (12 images: 6 shade + 6 spread) to the full sequence.

- Job: `depthpro-kk-coherent`, prio 11 (`scripts/gpu_submit.sh --prio 11 --wait`),
  `apple/DepthPro-hf`, MPS, fp16.
- Inputs: `trippy depth-points --scene ~/Splats/scenes/karekare/kk-coherent --images
  <all 219 names> --width 1008 --depth-dir output/depth/kk-coherent-all --cache-dir
  output/cache --run-depth` (undistorts every registered image not already cached at
  `w1008`, writes `output/depth/kk-coherent-all/manifest.json`, prints the GPU command).
- **rc=0**; **219/219 images** processed at 1008x756, `valid_fraction=1.0` for every
  frame; wall clock **293.6 s** (~1.34 s/image), consistent with EXP-0004's 1.3-1.7
  s/image range. No image was skipped for having too few sparse-COLMAP scale matches
  (`MONODEPTH_MIN_SCALE_MATCHES`).
- Scale-alignment quality (median-ratio `s`, MAD, sparse-match count `n`, averaged over
  the 6 `SHADE_FRAMES_KK` frames vs. the other 213): shade mean scale=1.304, MAD=0.188,
  n_matches=1,914; non-shade mean scale=1.712, MAD=0.221, n_matches=3,978. Shade frames
  again get ~2x fewer sparse matches to anchor scale against (darker region, fewer
  SIFT/COLMAP keypoints -- same finding as EXP-0004's 12-image sample) but a *lower*
  MAD, i.e. the few matches they get agree with each other unusually well (same caveat
  as EXP-0004: this is not yet evidence the scale itself is *correct*, only that it's
  internally consistent).

## Density numbers

Built via:

```bash
# 1. Full-219 MonoDepth PointSet (all registered images, not just the EXP-0004 sample).
PYTHONPATH=. TRIPPY_OUTPUT=/Users/nzbirdranch/trippy/output \
  .venv/bin/python -m trippy.cli depth-points \
  --scene /Users/nzbirdranch/Splats/scenes/karekare/kk-coherent \
  --images <all 219 names> --width 1008 \
  --depth-dir /Users/nzbirdranch/trippy/output/depth/kk-coherent-all \
  --cache-dir /Users/nzbirdranch/trippy/output/cache \
  --out /Users/nzbirdranch/trippy/output/points/kk-coherent-monodepth-219.npz

# 2. Union(Gaussian size_mode=knn, MonoDepth-219), voxel=0.03 -- CPU-heavy (kNN over 5.74M
#    Gaussian points), run via the single-heavy-job lock.
scripts/cpu_heavy.sh union-build -- bash -c \
  '.venv/bin/python -m trippy.cli points-build \
     --config output/points/union_source_config.yaml \
     --out output/points/kk-coherent-union-full.npz'
```

| Source | Count | bbox (world units) | median nn-distance |
|---|---|---|---|
| gaussian (`min_opacity=0.05`, full) | 5,736,619 | [-79.3,-85.6,-61.4] to [83.1,68.3,94.1] | 0.0795 |
| monodepth (12-image sample, EXP-0004, for comparison) | 234,712 | [-29.7,-38.2,-58.5] to [30.9,5.5,34.6] | 0.166 |
| **monodepth (full 219 images)** | **3,786,345** | [-39.6,-138.8,-203.4] to [145.1,7.2,60.5] | **0.2806** |
| **union (Gaussian ∩ MonoDepth-219, voxel=0.03)** | **5,887,647** | [-79.3,-138.8,-203.4] to [145.1,68.3,94.1] | **0.2984** |

Raw (pre-union-dedupe) point counts: gaussian 5,736,619 + monodepth-219 3,786,345 =
9,522,964. The union's global voxel dedupe (cell edge 0.03, applied once over the
concatenation of both sources -- not just at cross-source collisions) collapses this to
5,887,647, i.e. **38.2% of all raw points share a voxel cell with a higher-`conf0`
survivor**. Per-provenance histogram of the survivors: `{"gaussian": 2,205,602,
"monodepth": 3,682,045}`. Two findings worth flagging plainly:

1. **Most of the collapse is Gaussian-vs-Gaussian, not Gaussian-vs-MonoDepth.** Gaussian
   `conf0` (`sigmoid(opacity)`, typically well above MonoDepth's fixed `conf0=0.35`) wins
   essentially every direct collision with a MonoDepth point, so MonoDepth only loses
   3,786,345 - 3,682,045 = 104,300 points (2.8%) to Gaussian competition. But the
   Gaussian source itself drops from 5,736,619 to 2,205,602 survivors (61.5% collapsed)
   -- almost entirely Gaussian points colliding with *other* Gaussian points in the same
   0.03 cell. The Gaussian cloud's median nn-distance is 0.0795 (see EXP-0004's table),
   more than 2.5x the 0.03 voxel edge, but that is a *median* over a heavy-tailed,
   highly non-uniform distribution: 3DGS training densifies far more aggressively in
   some regions (near-camera surfaces, high-gradient detail) than others, and those
   dense regions are exactly where a 0.03 voxel collapses many points into one.
2. **`voxel=0.03` was chosen to match `MonoDepthSource`'s own internal dedupe cell
   size** (`MONODEPTH_DEFAULT_VOXEL`, also used unchanged in EXP-0004), not tuned for
   the Gaussian cloud's density. Given finding 1, a coarser union voxel would erase
   even more Gaussian detail for no monodepth-side benefit, while a finer one would
   preserve more Gaussian points at the cost of a larger union set (more training-time
   memory/compute per crop). This experiment ships with `0.03` as the same "already
   documented" value rather than re-tuning it for the union step specifically --
   flagged here per AGENTS.md ("never cull an idea for effort"; this is a parked
   voxel-size sensitivity study, not a rejected one) rather than silently accepted.

### Shade-frame coverage (MonoDepth-219 source, same 8px-radius point-presence check as EXP-0004)

Numeric-only (no image opened/viewed to produce these numbers -- AGENTS.md "Never send
scene imagery to a model"): projects the full 3,786,345-point MonoDepth-219 set into
each shade camera, marks the pixel each point lands on, then dilates by an 8px-radius
disk and reports the covered fraction.

| Shade frame | n points visible | coverage (full image) | coverage (central 50% box) |
|---|---|---|---|
| IMG_3828 | 1,175,233 | 100.00% | 100.00% |
| IMG_3829 | 1,148,813 | 100.00% | 100.00% |
| IMG_3830 | 1,022,870 | 100.00% | 100.00% |
| IMG_3831 | 930,171 | 100.00% | 100.00% |
| IMG_3832 | 772,928 | 100.00% | 100.00% |
| IMG_3833 | 773,535 | 100.00% | 100.00% |

Same read as EXP-0004, now even more saturated: with ~16x the raw points feeding each
shade camera (every one of the 219 images contributes points, not just the 12-image
sample), the 8px-radius coverage metric is pinned at 100% everywhere -- it confirms
"some geometry lands near every pixel" (true almost by construction at this density),
not "the geometry is metrically correct there." That question still needs the shade
audit / Jordan's viewer verdict once this source feeds the training runs below.

## Configuration

Two config files, both under this directory (see `docs/EXPERIMENTS.md` "Training runs"
for the config file format):

- **`config_broadcast.yaml`** -- `width=1008`, `crop=384`, `mode=broadcast`, `layers=5`,
  `epochs=300`, `train_factor=1.0`, `point_source={type: npz, path:
  output/points/kk-coherent-union-full.npz}`.
- **`config_trips.yaml`** -- identical except `mode=trips` (TRIPS's own published
  layer-selection rule; see `docs/GEOMETRY.md`).

Both set `forced_heldout: SHADE_FRAMES_KK` (`IMG_3828.jpg`..`IMG_3833.jpg`) so every eval
reports a shade-region number, and `device: mps`, run only via `scripts/gpu_submit.sh
--train` (AGENTS.md -- never invoked directly).

## Planned commands

```bash
# Actually submitted (this experiment was built and run from a git worktree, so
# TRIPPY_OUTPUT and --run-dir are pinned to the main checkout's output/ tree --
# docs/EXPERIMENTS.md "Override the output directory ... useful when the job runs from
# a git worktree but its artefacts should land in the main checkout's output/"):
TRIPPY_OUTPUT=/Users/nzbirdranch/trippy/output scripts/gpu_submit.sh --train train-union-broadcast -- \
  trippy train --config experiments/EXP-0006-union/config_broadcast.yaml --max-minutes 330 \
  --run-dir /Users/nzbirdranch/trippy/output/runs/EXP-0006-union/broadcast

TRIPPY_OUTPUT=/Users/nzbirdranch/trippy/output scripts/gpu_submit.sh --train train-union-trips -- \
  trippy train --config experiments/EXP-0006-union/config_trips.yaml --max-minutes 330 \
  --run-dir /Users/nzbirdranch/trippy/output/runs/EXP-0006-union/trips

# Both accepted (prio 70, queued behind Splats' own jobs and EXP-0003's two full2
# trainings already in the queue -- scripts/gpu_submit.sh appends the submission to
# research/trips-metal.md automatically):
#   queued as 70-trippy-train-union-broadcast.sh; submit.sh rc=0
#   queued as 70-trippy-train-union-trips.sh; submit.sh rc=0

# Resume if a job is interrupted (queue timeout, machine restart):
scripts/gpu_submit.sh --train train-union-broadcast-resume -- \
  trippy train --config experiments/EXP-0006-union/config_broadcast.yaml \
  --resume /Users/nzbirdranch/trippy/output/runs/EXP-0006-union/broadcast/checkpoints/checkpoint_latest.pt

# Candidate report once a checkpoint exists (export PLY, shade/extent audits, dolly,
# off-path honesty sheet):
trippy candidate-report --checkpoint /Users/nzbirdranch/trippy/output/runs/EXP-0006-union/broadcast/checkpoints/checkpoint_latest.pt \
  --out /Users/nzbirdranch/trippy/output/runs/EXP-0006-union/broadcast/candidate
```

## Gate

**v0.2.0 acceptance** (docs/SPEC.md): held-out PSNR within 1.5 dB of the best plain
Gaussian on non-shade frames; shade dolly shows shading, not a cloud; shade audit number
drops vs. `kkc_15000.ply`'s own audit and vs. EXP-0003's Gaussian-only run; extent
(p99/p99.9/max radius) not inflated beyond +20% of the initial COLMAP/Gaussian bbox;
honesty sheet reviewed. Cross-reference: EXP-0003 (point source 1) and EXP-0004 (point
source 2, 12-image sample) results are the baselines this experiment is compared against
for the v0.2.0 stop-or-go decision.

## Verdict

(Filled in once the queued training jobs complete -- both are prio 70, behind
`train-full1`/`train-full2-broadcast`/`train-full2-trips` (EXP-0003) already in the
queue; `--max-minutes 330` caps each at ~5.5 h wall clock.)

| Metric | broadcast | trips |
|---|---|---|
| Held-out PSNR (non-shade frames) | TBD | TBD |
| Held-out PSNR (shade frames, `SHADE_FRAMES_KK`) | TBD | TBD |
| Held-out SSIM | TBD | TBD |
| Held-out LPIPS | TBD | TBD |
| Shade audit (opacity mass in shade region) | TBD | TBD |
| Extent (p99 / p99.9 / max radius) | TBD | TBD |
| Wall-clock | TBD | TBD |
| Jordan's viewer verdict | TBD | TBD |

Artifact paths (once run): `output/runs/EXP-0006-union/{broadcast,trips}/` (`export.ply`,
`eval_ep*/sheet.png`, `metrics.jsonl`, `checkpoints/`). Dolly video and delivery follow
via `scripts/deliver.sh` once the runs complete.

## Artifacts

- `output/points/kk-coherent-monodepth-219.npz` (+ `.summary.json`)
- `output/points/kk-coherent-union-full.npz` (+ `.summary.json`)
- `output/points/union_source_config.yaml` (the `points-build` input config)
- this worktree's `output/jobs/trippy-depthpro-kk-coherent.sh` (submitted before
  `TRIPPY_OUTPUT` was pinned to the main checkout; harmless since the manifest path was
  already absolute) and the main checkout's
  `output/jobs/trippy-train-union-{broadcast,trips}.sh` (submitted with
  `TRIPPY_OUTPUT=/Users/nzbirdranch/trippy/output`)
- `$SPLATS_ROOT/tools/gpu_queue/logs/trippy-depthpro-kk-coherent.log` (the GPU queue's
  own log path)
- `output/logs/union-build.log` (`scripts/cpu_heavy.sh` job log)
- `output/runs/EXP-0006-union/coverage_stats.json` (shade-frame coverage numbers)
- `output/runs/EXP-0006-union/{broadcast,trips}/`
