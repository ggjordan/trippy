# EXP-0004: MonoDepthSource (DepthPro) on kk-coherent

## Question

Does D4 point source 2 (`trippy.points.monodepth.MonoDepthSource`) produce
scale-aligned, world-frame points from Apple DepthPro's monocular metric
depth, and does the shade region (IMG_3828-3833) end up with monodepth
geometry the same way the rest of the sequence does?

## Point source

Source 2 (monocular depth): `trippy.points.monodepth.MonoDepthSource`,
scale-aligned via `median_ratio` against reprojected COLMAP sparse points,
`provenance=MONODEPTH`.

## Config

- Scene: `~/Splats/scenes/karekare/kk-coherent` (219 registered images).
- Images (12): the 6 shade frames `SHADE_FRAMES_KK` (IMG_3828-3833) + 6
  non-shade frames spread evenly across the registered sequence by index
  (`np.linspace(0, 218, 6)`, nudged off any shade index): IMG_3703 (idx 0),
  IMG_3753 (idx 44), IMG_3796 (idx 87), IMG_3840 (idx 131), IMG_3896
  (idx 174), IMG_3940 (idx 218, last).
- `width=1008`, `stride=6`, `voxel=0.03`, `conf0=0.35`,
  `scale_mode="median_ratio"` (all defaults).
- Undistortion cache: `output/cache/kk-coherent/w1008/` (reused an
  existing partial cache from earlier work; `trippy.points.depth_io`'s
  cache is byte-compatible with `trippy.scene.dataset.SceneDataset`'s).
- Depth outputs: `output/depth/kk-coherent/` (Apple DepthPro via
  `~/Splats/tools/ldi/depth_batch.py`).

## Planned commands

```bash
# 1. Prepare inputs + print the exact GPU job (exits 3, depth outputs missing):
PYTHONPATH=. TRIPPY_OUTPUT=/Users/nzbirdranch/trippy/output \
  .venv/bin/python -m trippy.cli depth-points \
  --scene /Users/nzbirdranch/Splats/scenes/karekare/kk-coherent \
  --images IMG_3828.jpg,IMG_3829.jpg,IMG_3830.jpg,IMG_3831.jpg,IMG_3832.jpg,IMG_3833.jpg,IMG_3703.jpg,IMG_3753.jpg,IMG_3796.jpg,IMG_3840.jpg,IMG_3896.jpg,IMG_3940.jpg \
  --width 1008 --depth-dir /Users/nzbirdranch/trippy/output/depth/kk-coherent \
  --cache-dir /Users/nzbirdranch/trippy/output/cache --run-depth

# 2. Run the printed job (GPU queue, apple/DepthPro-hf via transformers, MPS):
scripts/gpu_submit.sh --prio 11 --wait depthpro-kk-1 -- bash -c \
  '/Users/nzbirdranch/Splats/tools/vggt/.venv/bin/python3 /Users/nzbirdranch/Splats/tools/ldi/depth_batch.py /Users/nzbirdranch/trippy/output/depth/kk-coherent/manifest.json'

# 3. Build the PointSet:
PYTHONPATH=. TRIPPY_OUTPUT=/Users/nzbirdranch/trippy/output \
  .venv/bin/python -m trippy.cli depth-points \
  --scene /Users/nzbirdranch/Splats/scenes/karekare/kk-coherent \
  --images IMG_3828.jpg,...,IMG_3940.jpg \
  --width 1008 --depth-dir /Users/nzbirdranch/trippy/output/depth/kk-coherent \
  --cache-dir /Users/nzbirdranch/trippy/output/cache \
  --out /Users/nzbirdranch/trippy/output/points/kk-coherent-monodepth-12.npz
```

## Gate

v0.2.0 milestone: MonoDepthSource implemented and run on kk-coherent (see
docs/SPEC.md's milestone table and "Stop-or-go point: v0.2.0"). This
experiment is a build/inspection step, not itself the v0.2.0 shade-audit
gate (that needs a trained model, not just raw points).

## Results

DepthPro (apple/DepthPro-hf, MPS, fp16) job `trippy-depthpro-kk-1`: rc=0,
12/12 images at 1008x756, 1.3-1.7s/image, 18.0s total, `valid_fraction=1.0`
for every frame.

Per-image median-ratio scale `s` (COLMAP depth / DepthPro depth), MAD, and
sparse-match count `n`:

| Image | shade? | scale s | MAD | n matches | points contributed |
|---|---|---|---|---|---|
| IMG_3828 | yes | 1.065 | 0.230 | 2436 | 21,168 |
| IMG_3829 | yes | 1.420 | 0.267 | 2450 | 21,168 |
| IMG_3830 | yes | 1.154 | 0.184 | 1882 | 21,168 |
| IMG_3831 | yes | 1.227 | 0.152 | 1768 | 21,168 |
| IMG_3832 | yes | 1.485 | 0.153 | 1366 | 21,168 |
| IMG_3833 | yes | 1.471 | 0.142 | 1580 | 21,168 |
| IMG_3703 | no | 2.781 | 0.429 | 2937 | 21,168 |
| IMG_3753 | no | 1.353 | 0.229 | 5361 | 21,168 |
| IMG_3796 | no | 2.015 | 0.177 | 4127 | 21,168 |
| IMG_3840 | no | 1.595 | 0.091 | 2729 | 21,168 |
| IMG_3896 | no | 1.744 | 0.107 | 5629 | 21,168 |
| IMG_3940 | no | 3.358 | 0.462 | 5456 | 21,168 |

Shade mean: scale=1.304, MAD=0.188, n_matches=1,914.
Non-shade mean: scale=2.141, MAD=0.249, n_matches=4,373.

Union PointSet: 254,016 raw points (21,168/frame x 12, uniform because
`valid_fraction=1.0` everywhere) -> **234,712** after voxel dedupe
(voxel=0.03, 7.6% collapsed). Shade and non-shade frames each contributed
exactly 127,008 raw points (uniform density regardless of scene content).
`median_nn_distance=0.166` world units. bbox
`[-29.7,-38.2,-58.5]` to `[30.9,5.5,34.6]`.

Numeric shade coverage (8px-radius point-presence, full 234,712-point union
projected into each shade camera; **no image was opened/viewed** to produce
these numbers, only arrays -- see AGENTS.md "Never send scene imagery to a
model"):

| Shade frame | n points visible | coverage (full image) | coverage (central 50% box) |
|---|---|---|---|
| IMG_3828 | 109,746 | 100.00% | 100.00% |
| IMG_3829 | 117,744 | 100.00% | 100.00% |
| IMG_3830 | 119,957 | 100.00% | 100.00% |
| IMG_3831 | 112,897 | 99.998% | 100.00% |
| IMG_3832 | 95,689 | 99.9996% | 100.00% |
| IMG_3833 | 86,272 | 100.00% | 100.00% |

## Verdict

**Inconclusive on the coverage metric as specified, but a real signal on
scale-alignment quality.** The 8px-radius point-presence coverage is
saturated at ~100% for every shade frame, full image and central box alike.
Read plainly: this metric mostly measures "did DepthPro return a valid
depth value here" (yes, everywhere -- `valid_fraction=1.0`) combined with
stride=6 backprojection density (~6px native spacing, well under the 8px
radius), not "is the depth geometrically correct here." So it does not by
itself answer "do we now have *correct* geometry inside the shade region" --
it answers "do we have *some* geometry," and the answer to that weaker
question is yes, trivially, by construction of this source. The more
informative number is that shade frames have 44% fewer usable sparse-COLMAP
matches on average (1,914 vs 4,373) to anchor their scale against -- exactly
what you'd expect from a darker region producing fewer SIFT/COLMAP
keypoints -- yet their MAD is *lower* on average (0.188 vs 0.249), meaning
the few matches they do get agree with each other unusually well; whether
that agreement reflects a genuinely well-calibrated scale or a small,
possibly-biased sample is not distinguishable from this experiment alone.
Establishing whether the resulting points are *correct* (not just present)
requires the shade audit and/or Jordan's viewer verdict once this source
feeds an actual training run (docs/SPEC.md v0.2.0) -- this experiment only
confirms the pipeline runs end to end and produces a plausible, densely
covering point set.

## Artifacts

- `output/points/kk-coherent-monodepth-12.npz` (+ `.summary.json`)
- `output/runs/EXP-0004/coverage_stats.json`
- `output/runs/EXP-0004/{IMG_3828..3833}_coverage.png`, `sheet.png`
  (delivered as `EXP-0004-monodepth-shade-coverage`, not opened/viewed by
  the agent that produced them)
- `output/jobs/trippy-depthpro-kk-1.sh`, `output/logs/trippy-depthpro-kk-1.log`
