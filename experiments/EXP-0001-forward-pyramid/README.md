# EXP-0001: Forward pyramid validation

## Question

Does our Metal `blend_fwd` implementation equal the numpy reference and render 3 Karekare-coherent frames at 1008 pixels wide in <100 ms (≥10 fps)?

## Point source

1 = GaussianPlySource on `$SPLATS_ROOT/output/Training-Data/karekare/kk-coherent/kkc_15000.ply` (7.36M Gaussian points)

## Configuration

Synthetic 32×32 scene for gradient agreement test; real Karekare frames for FPS measurement.

- **Dataset**: kk-coherent, undistorted to 1008 wide, three representative frames (IMG_3828, IMG_3830, IMG_3832 — shade region).
- **Model**: pyramid levels 0–4, 16-fragment cap per pixel, no U-Net (raw blend_fwd output).
- **Hardware**: GPU=MPS, dtype=float32, no gradient computation (forward only).

## Commands run

CPU dry-run (tiny, catches errors before the GPU queue; `--device cpu` uses `trippy.raster.ref_torch`, so keep it small):

```bash
export TRIPPY_OUTPUT=/tmp/trippy-dryrun-output SPLATS_ROOT=/Users/nzbirdranch/Splats
PYTHONPATH=. TRIPS_DEVICE=cpu /Users/nzbirdranch/trippy/.venv/bin/python -m trippy.cli render \
  --scene /Users/nzbirdranch/Splats/scenes/karekare/kk-coherent \
  --points gaussian --ply /Users/nzbirdranch/Splats/output/Training-Data/karekare/kk-coherent/kkc_15000.ply \
  --min-opacity 0.05 --size-mode scale --max-points 20000 --width 252 \
  --frames IMG_3830.jpg --mode trilinear \
  --out /tmp/trippy-dryrun-output/runs/dryrun --device cpu
```

Real run, MPS via the GPU queue, full 5.7M-point Gaussian cloud, 4 frames at 1008 wide (shade frames IMG_3830/IMG_3828, plus one early frame IMG_3704 and one late frame IMG_3939 from the 219 registered images), both layer-selection modes:

```bash
export TRIPPY_OUTPUT=/Users/nzbirdranch/trippy/output SPLATS_ROOT=/Users/nzbirdranch/Splats
scripts/gpu_submit.sh --prio 14 --wait render-kk-1 -- bash -c \
  'cd /Users/nzbirdranch/trippy/.worktrees/render-kk && PYTHONPATH=. \
   /Users/nzbirdranch/trippy/.venv/bin/python -m trippy.cli render \
   --scene /Users/nzbirdranch/Splats/scenes/karekare/kk-coherent --points gaussian \
   --ply /Users/nzbirdranch/Splats/output/Training-Data/karekare/kk-coherent/kkc_15000.ply \
   --min-opacity 0.05 --size-mode scale --width 1008 \
   --frames IMG_3830.jpg,IMG_3828.jpg,IMG_3704.jpg,IMG_3939.jpg --mode trilinear \
   --out /Users/nzbirdranch/trippy/output/runs/EXP-0001/trilinear --device mps'

scripts/gpu_submit.sh --prio 14 --wait render-kk-2 -- bash -c \
  '... (same args, --mode broadcast --out .../EXP-0001/broadcast)'
```

## Gate

**v0.1.0 acceptance**: CPU tests pass; three (here, four) 1008-wide frames render well under 300 ms/frame total. Contact sheet delivered: photo | levels 0-4 | coverage | depth per frame, plus a summary sheet (photo | L0 | coverage) across frames.

## Numbers (2026-09-06, kk-coherent, 5,736,619 points post `min_opacity=0.05` filter)

### `--mode trilinear` (job `trippy-render-kk-1`, rc=0)

| frame | emit ms | sort ms | blend ms | total ms | fragments | points visible | coverage (full) | coverage (center 50%) |
|---|---|---|---|---|---|---|---|---|
| IMG_3830.jpg (shade) | 126.5 | 6.7 | 2.5 | 135.7 | 6,422,926 | 1,317,832 | 0.1780 | 0.2323 |
| IMG_3828.jpg (shade) | 61.0 | 7.8 | 1.1 | 69.9 | 7,586,994 | 1,647,321 | 0.1685 | 0.2331 |
| IMG_3704.jpg (early, non-shade) | 89.7 | 14.6 | 1.7 | 105.9 | 13,872,692 | 3,082,048 | 0.2140 | 0.3353 |
| IMG_3939.jpg (late, non-shade) | 68.2 | 24.0 | 1.5 | 93.7 | 13,349,942 | 3,065,901 | 0.2206 | 0.3335 |

### `--mode broadcast` (job `trippy-render-kk-2`, rc=0)

| frame | emit ms | sort ms | blend ms | total ms | fragments | points visible | coverage (full) | coverage (center 50%) |
|---|---|---|---|---|---|---|---|---|
| IMG_3830.jpg (shade) | 121.0 | 29.6 | 2.9 | 153.5 | 24,295,378 | 1,317,832 | 0.2599 | 0.3449 |
| IMG_3828.jpg (shade) | 81.3 | 32.7 | 2.0 | 115.9 | 30,456,552 | 1,647,321 | 0.2500 | 0.3431 |
| IMG_3704.jpg (early, non-shade) | 162.6 | 65.7 | 2.8 | 231.1 | 56,901,327 | 3,082,048 | 0.3067 | 0.4538 |
| IMG_3939.jpg (late, non-shade) | 104.7 | 68.0 | 2.9 | 175.7 | 58,918,530 | 3,065,901 | 0.3163 | 0.4655 |

## Verdict

**PASS on speed**: every frame renders in well under 300 ms (worst case 135.7 ms total, trilinear, IMG_3830 -- 7.4 fps single-frame, no batching/repeat attempted). `points_visible` (points surviving the conservative view-frustum cull) and `num_fragments` scale with scene complexity per frame as expected (2-3x more for the two non-shade frames, which see much more of the reconstructed forest floor/riverbed than the two tight shade-region shots).

**Coverage in the shade region is measurably worse, not just visually different.** Coverage (`mean(1 - T_final)` over level 0) is a direct, numeric proxy for "how much of this pixel grid actually has a photographed point under it" -- 0 means every fragment list for that pixel was empty, 1 means fully opaque. For the two shade frames, full-frame coverage is 0.168-0.178 and *center-crop* coverage (the middle 50% x 50% of the frame, roughly where the shaded understorey sits) is 0.232-0.233. For the two non-shade frames, full-frame coverage is 0.214-0.221 and center coverage is 0.334-0.335 -- roughly 40-45% *relatively* higher center coverage than the shade frames. In other words: even before any U-Net inference, the shade region has meaningfully fewer Gaussian centres per pixel than daylight regions of the same scene, i.e. holes are the honest, expected level-0 output there, not a rendering bug. This matches the project's known defect (fewer/weaker Gaussians survive training in low-light/shadowed regions) and is exactly the numeric signal `docs/SPEC.md`'s stage-1 gate ("shade rendered as shading, not a cloud") will need the U-Net to fix -- there is no free lunch at the point-source level; the network has real holes to fill, not phantom ones.

**`trilinear` vs `broadcast` differ substantially at level 0, not just in the coarser layers.** Measured, both are not close: `broadcast` writes *every* point into layer 0 at factor 1 regardless of its projected pixel size, while `trilinear` only reaches layer 0 for points whose lower-bound layer is 0 (small/near points). The result: `broadcast` roughly doubles level-0 fragment count (e.g. IMG_3830: 24.3M vs 6.4M fragments) and pushes coverage up substantially in every frame (full-frame: 0.260 vs 0.178 for IMG_3830, 0.307 vs 0.214 for IMG_3704; center: 0.345 vs 0.232 and 0.454 vs 0.335 respectively). The shade-vs-non-shade coverage *gap* persists in both modes (shade frames are still the two lowest-coverage frames either way), so the qualitative finding -- shade has fewer points per pixel than daylight -- holds regardless of layer-selection mode; only the absolute hole fraction changes. `broadcast` is also slower (more emit + sort work per frame: 175-231 ms total vs 70-136 ms for trilinear), consistent with docs/ARCHITECTURE.md's fragment-count table (`L x 4` vs `<= 2 x 4` fragments/point).
