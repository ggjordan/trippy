# EXP-0005: Hybrid design C -- render->photo U-Net refinement (cheap side-experiment)

## Question

Design C (docs/PLAN-2026-09-05.md "Hybrid (v0.3): (C) render->photo U-Net refinement on
gsrender.py outputs first (cheap, validates net/losses)"): render the existing best Gaussian
splat (`kkc_15000.ply`) for every registered kk-coherent view with Splats' `gsrender.py`
(RGB + depth + alpha), then train trippy's U-Net (with the neural camera) to map
render -> photo. Does a *learned renderer* change anything in the shade region specifically,
or does it only sharpen/relight already-well-covered pixels? Evaluated on held-out views
(modulo-8 split) including the six forced shade frames (`SHADE_FRAMES_KK`,
`IMG_3828.jpg`..`IMG_3833.jpg`).

This is the cheapest of the three planned hybrid designs (C, then A1, then B) -- it needs no
new rasteriser, no point-cloud training, no pose refinement, just a fixed image pair per
training step -- and its purpose is partly to validate the U-Net/loss/NeuralCamera machinery
end to end on real photos before the more expensive A1 (Gaussians as TRIPS points with
learned feature vectors, joint training) design.

## Point source

Not a `trippy.points.PointSource` at all -- Design C's "points" are already baked into the
Gaussian-splat render itself. Splats' `gsrender.py` renders `kkc_15000.ply` (the same 7.36M
trained-Gaussian PLY used as point source 1 in EXP-0003) at `max_hw=400` (never the gsrender
default of 32, which corrupts near-camera Gaussian footprints -- Splats' PROJECT.md).

## Configuration

- **`trippy/hybrid/render_splat_views.py`** -- renders every registered kk-coherent view
  (219 images) against `kkc_15000.ply` at `width=1008` (`SceneDataset`'s own undistorted grid,
  so every render shares its photo's exact `(H, W, K)`), writing `<stem>.png` (rgb, uint8),
  `<stem>.depth.npy` (float16), `<stem>.alpha.npy` (float16) under
  `output/hybrid-c/renders/w1008/`. Run in two shards (see "Planned commands").
- **`config.yaml`** (this directory) -- the training run: `renders_dir:
  output/hybrid-c/renders/w1008`, `width: 1008`, `crop: 384`, `channels: 4` (rgb + alpha; no
  depth channel this run), `layers: 5` (same `NetworkConfig` TRIPS itself ships), `epochs:
  2000` with a `--max-minutes 40` wall-clock cap (a 60k-param U-Net on 384x384 crops finishes
  many thousand steps well inside that budget), `loss_l1/loss_ssim/loss_lpips: 1.0` (L1 + SSIM
  + LPIPS(alex), no vgg term -- see `trippy/hybrid/config_c.py`), `forced_heldout:
  SHADE_FRAMES_KK`.

Loss mask: the render's own `alpha > 0` is OR-ed with an all-ones mask, which always
evaluates to all-ones -- a deliberate, documented design decision (see
`trippy/hybrid/train_c.py`'s `train_step` docstring/comment), not a bug. Design C's loss runs
over the *whole* crop, including pixels the render left uncovered (alpha == 0, e.g. holes in
the Gaussian cloud), so the U-Net is pushed to hallucinate sensible content there too, not
just refine already-covered pixels. This is one candidate explanation if PSNR/LPIPS improve
overall but the shade region specifically does not: a network trained to fix holes everywhere
has no special incentive to fix the shade cloud's holes *correctly* (vs. plausibly).

Everything else keeps `HybridCConfig`'s scaled-for-trippy defaults (see
`trippy/constants.py` "hybrid/" section and "train/" section for values reused verbatim from
`TrainConfig`, e.g. `lr_network`, `lr_exposure`, `lr_response`, `heldout_k`).

## Planned commands

```bash
# 1+2. Render every registered view against kkc_15000.ply, in two shards (queue-friendly).
scripts/gpu_submit.sh --prio 17 --wait hybrid-c-render-1 -- bash -c \
  'cd /Users/nzbirdranch/trippy/.worktrees/hybrid-c && PYTHONPATH=. \
   /Users/nzbirdranch/Splats/tools/ml-sharp/.venv/bin/python -m trippy.hybrid.render_splat_views \
   --scene /Users/nzbirdranch/Splats/scenes/karekare/kk-coherent \
   --ply /Users/nzbirdranch/Splats/output/Training-Data/karekare/kk-coherent/kkc_15000.ply \
   --out /Users/nzbirdranch/trippy/.worktrees/hybrid-c/output/hybrid-c/renders/w1008 \
   --width 1008 --device mps --start-index 0 --end-index 110'

scripts/gpu_submit.sh --prio 17 --wait hybrid-c-render-2 -- bash -c \
  'cd /Users/nzbirdranch/trippy/.worktrees/hybrid-c && PYTHONPATH=. \
   /Users/nzbirdranch/Splats/tools/ml-sharp/.venv/bin/python -m trippy.hybrid.render_splat_views \
   --scene /Users/nzbirdranch/Splats/scenes/karekare/kk-coherent \
   --ply /Users/nzbirdranch/Splats/output/Training-Data/karekare/kk-coherent/kkc_15000.ply \
   --out /Users/nzbirdranch/trippy/.worktrees/hybrid-c/output/hybrid-c/renders/w1008 \
   --width 1008 --device mps --start-index 110 --end-index 219'

# 3. Train the U-Net + neural camera, 40-minute wall-clock budget.
#    NOTE: `gpu_submit.sh`'s leading-`trippy`-token rewrite resolves to
#    "$(worktree_root)/.venv/bin/python" -- git worktrees do not get their own `.venv`
#    (uv sync / the venv is gitignored and lives only in the primary checkout), so that
#    plain form fails with "No such file or directory" (hit on the first submission of this
#    job, see the trips-metal.md entry timestamped just before this one). Use the same
#    `bash -c` + explicit-python pattern as the render jobs instead, pointing at the primary
#    checkout's venv with PYTHONPATH=. so this worktree's `trippy` package (not the primary
#    checkout's editable install) is what actually gets imported:
scripts/gpu_submit.sh --prio 18 --wait hybrid-c-train-1 -- bash -c \
  'cd /Users/nzbirdranch/trippy/.worktrees/hybrid-c && PYTHONPATH=. \
   /Users/nzbirdranch/trippy/.venv/bin/python -m trippy.cli hybrid-c train \
   --config experiments/EXP-0005-hybrid-c/config.yaml --max-minutes 40'

# 4. Standalone re-evaluation of the final checkpoint (no re-training) -- only if needed;
#    `fit()` already writes eval_ep*/metrics.json + sheet.png + shade_frames/*.png per
#    `eval_every` and at the final epoch/time-budget cutoff.
trippy hybrid-c eval \
  --checkpoint output/runs/EXP-0005-hybrid-c/EXP-0005-hybrid-c_1/checkpoints/checkpoint_latest.pt

# 5. Deliver the summary sheet.
scripts/deliver.sh \
  output/runs/EXP-0005-hybrid-c/EXP-0005-hybrid-c_1/eval_ep<N>/sheet.png \
  EXP-0005-hybrid-c-refine \
  "Design C: U-Net refines Gaussian renders of kk-coherent toward photos. Sheet: photo | Gaussian render | refined | diff on held-out frames incl. shade. Numbers in README."
```

## Gate

Not a stage gate (that binding gate was v0.2.0, already passed/failed on its own terms) --
this is the "cheap side-experiment" comparison harness step of v0.3.0 (docs/SPEC.md
"Milestones" v0.3.0 row: "hybrid beats best plain Gaussian on shade audit AND extent gate;
LPIPS not worse"). Design C has no points/extent to audit (it never touches the Gaussian
cloud), so its own bar is narrower: does the learned renderer improve held-out PSNR/LPIPS at
all, and -- the actual question this experiment exists to answer -- does the *shade* bucket
improve by a comparable amount to the *non-shade* bucket, or does it lag (evidence the hole/
hallucination problem is shade-specific, motivating A1 next)?

## Verdict

Render jobs: `hybrid-c-render-1` (frames 0-110, rc=0, 1495.1s) and `hybrid-c-render-2`
(frames 110-219, rc=0, 1675.8s) -- 219/219 kk-coherent registered views rendered against
`kkc_15000.ply` at `max_hw=400`, width 1008. Training job: `hybrid-c-train-1` (rc=0),
40.0-minute wall-clock budget, reached epoch 1125 (~27,000 crop steps, 384x384, MPS) before
the budget cut it off. Held-out split: 33 frames (27 non-shade + 6 forced shade,
`SHADE_FRAMES_KK`). Numbers below are the final eval (`eval_ep1125/metrics.json`); the
"baseline" (raw render, no U-Net) numbers are identical at every epoch by construction.

| Metric | Baseline (raw render vs photo) | Refined (U-Net vs photo) | delta |
|---|---|---|---|
| PSNR, all held-out (n=33) | 15.53 dB | 15.54 dB | +0.01 dB |
| PSNR, non-shade (n=27) | 15.66 dB | 16.11 dB | **+0.45 dB** |
| PSNR, shade (n=6, `SHADE_FRAMES_KK`) | 14.94 dB | 12.97 dB | **-1.96 dB** |
| SSIM, all held-out | 0.431 | 0.476 | +0.045 |
| SSIM, non-shade | 0.432 | 0.483 | +0.051 |
| SSIM, shade | 0.427 | 0.442 | +0.015 |
| LPIPS, all held-out (lower better) | 0.477 | 0.461 | -0.015 |
| LPIPS, non-shade | 0.465 | 0.448 | -0.018 |
| LPIPS, shade | 0.526 | 0.519 | -0.007 |

The shade-region PSNR regression is not a transient early-training blip: it is already
present by epoch 50 (refined shade PSNR 12.15 dB vs baseline 14.94 dB) and stays in the
same -1.4 to -2.0 dB band through epoch 200, 500, 800, and the final 1125 (12.97 dB) --
checked directly against `eval_ep{0050,0200,0500,0800,1125}/metrics.json`. Non-shade PSNR,
by contrast, rises from baseline by epoch 200 and holds a stable +0.4 to +0.5 dB gain from
there on. The aggregate "all" PSNR is essentially flat (15.53 -> 15.54 dB) because the
shade regression and the non-shade gain partly cancel in a 27-vs-6-frame average.

**Read** (numbers only): the learned renderer measurably helps the non-shade region on
every metric (PSNR, SSIM, LPIPS all improve) but measurably hurts the shade region's PSNR
by about 2 dB, a stable regression from early training onward, while giving the shade
region only small SSIM/LPIPS gains. So: yes, a learned renderer changes the shade region --
but on the primary photometric metric it makes it worse, not better, while it improves
everything else. That is consistent with (not proof of) the shade region's Gaussian render
having more low-coverage holes than non-shade (EXP-0001's T_final finding) and this design's
deliberately full-frame loss mask giving the U-Net no extra incentive to fill those holes
*correctly* rather than merely plausibly (SSIM/LPIPS reward structural/perceptual
plausibility more than exact per-pixel brightness). Design C's cheap side-experiment
therefore does not remove or improve the shade defect on PSNR; it trades shade-region
photometric accuracy for non-shade gains and a wash on the aggregate. This is a data point
against iterating further on C for the shade problem specifically, and toward trying A1
(Gaussians as TRIPS points with learned feature vectors, joint training) next, per
docs/SPEC.md's v0.3.0 plan.

Artifact path: `output/runs/EXP-0005-hybrid-c/EXP-0005-hybrid-c_1/`
(`eval_ep*/metrics.json`, `eval_ep*/sheet.png`, `eval_ep*/shade_frames/*.png`,
`metrics.jsonl`, `log.txt`, `checkpoints/`). Delivered summary sheet (photo | render |
refined | |diff| on the 6 held-out sheet rows, shade frames first) via `scripts/deliver.sh`
as `EXP-0005-hybrid-c-refine` (`~/Splats/output/Jordan-Review/4-other/EXP-0005-hybrid-c-refine.png`).
