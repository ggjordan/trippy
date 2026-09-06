# EXP-0011 — TRIPS on the full Karekare outing (`karekare-v2`)

**The question.** *Does TRIPS render the shade under the big tree as shading, when it has
seen the tree?*

Every Karekare run so far (EXP-0003, 0005, 0009, 0010) trained on `kk-coherent`, a
238-image subset. The big tree Jordan walks under on the way to the waterfall pool is not
in that subset at all — which is the simplest available explanation for why those runs
"break" when he heads there: they were asked to synthesise a place they had never been
shown. This experiment moves to the scene Jordan actually cares about, seeds it from the
splat he actually likes, and splits the shade frames so half of them are in training.

If the answer is still no *here*, the defect is TRIPS's shading, not the data. That is a
different and much more interesting result than the ones so far, and it is the first time
the experiment has been able to say so.

---

## The scene

`/Users/nzbirdranch/Splats/scenes/karekare/karekare-v2`

| | |
|---|---|
| Registered images (`sparse/0`) | **756** |
| Camera models | **202**, all `OPENCV` (stills, Brielle's phone, 4K video frames) |
| Sparse points | 208,570 |
| `images/` | 1,376 JPEGs (620 unregistered) |
| `masks/` | 1,376 person-mask PNGs |
| Frame shapes at width 1008 | 1008×756, 1008×1344, 1008×1792, 1008×567 |
| Frames with **no EXIF at all** | **189 / 756 (25%)** — every 4K video frame |

Point source: `/Users/nzbirdranch/Splats/output/Training-Data/karekare/karekare-lid/kklid_20000.ply`
(2.1 GB, 8,910,382 Gaussians), trained on exactly this COLMAP model.

**The 25% no-EXIF figure is a standing hazard on this scene, not a footnote.** A frame with
no EXIF has a per-image exposure that is never trained if it is held out, which is the bug
that cost EXP-0003 full2-broadcast six frames at 6.2–6.9 dB (see
`experiments/EXP-0003-kk-trips-train/README.md`, "The exposure artefact"). Two fixes are
already in and both are on in every config here: `Trainer._initial_exposure` starts an
EXIF-less frame at the scene mean (gain 1), and `eval_exposure_mode: neighbours`
interpolates a held-out frame's exposure from its training neighbours. On `kk-coherent`
this affected 10 of 219 frames. Here it affects 189 of 756, so the "all"/"psnr_mean"
columns are only readable *because* of those fixes — read `psnr_mean_eval`, not `psnr_mean`.

---

## Finding the shade frames — by MEASURING

`~/Splats/PROJECT.md`: *"Find the shade frames by MEASURING, not by eye. Every shade
experiment in this project picked its test frames by assumption, and one of them
misdiagnosed the defect as a result."* The prescribed method is per-image mean luminance
over the registered photos, looking for a contiguous dark RUN, corroborated by EXIF
exposure. That is what was done, plus one addition the multi-location `karekare-v2`
needed.

### Method

1. **Luminance.** Rec.709 mean luminance of every one of the 756 **registered** photos
   (JPEG decoded at 1/8 scale via PIL's DCT-scaled `draft`, which is a scaling of the same
   pixels, not a different measurement). Scene mean **121.31**, sd **14.72**.
2. **Dark runs.** Contiguous runs (in sorted-name order) of ≥ 3 frames below
   `mean − 1 sd = 106.59`. Ten such runs.
3. **Location.** `karekare-v2` covers a whole outing, so "dark" alone can mean "a different
   place" or "a different time of day". Each run's camera centroid was computed
   (`C = −Rᵀt`) and compared to the median run centroid. **Nine of the ten runs sit within
   1.7 world units of one spot**; the tenth (`IMG_4204`–`IMG_4206`, and the shallowest of
   the ten at luminance 105.4–106.6) is **9.28 units away** and was dropped. That the dark
   runs cluster at one location at all is itself the finding: it is a place, not a time.
4. **EXIF corroboration.** Auto-exposure on the kept runs: **ISO median 400 (p10 200,
   p90 800) at a median 1/60 s**. Everywhere else in the scene: **ISO median 80 at 1/99 s**.
   The camera is compensating by roughly 2½ stops on exactly the frames the luminance test
   picked, and it is doing it *despite* that compensation still leaving them 22 luminance
   points dark. Two unrelated signals, same frames.

### Result — the big-tree shade run: 93 frames

Mean luminance **99.50** vs the scene's **121.31** (−1.48 sd); darkest frame 73.9.

| sub-run | n | mean luminance |
|---|---|---|
| `IMG_4032`–`IMG_4041` | 9 | 98.9 |
| `IMG_4057`–`IMG_4062` | 6 | 96.5 |
| `IMG_4083`–`IMG_4085` | 3 | 105.8 |
| `IMG_4116`–`IMG_4118` | 3 | 105.5 |
| `IMG_4260`–`IMG_4284` | 17 | 100.5 |
| `IMG_4292`–`IMG_4321` | 22 | 100.7 |
| `IMG_5540`–`IMG_5561` | 18 | 95.2 |
| `IMG_5646`–`IMG_5660` | 15 | 100.8 |

The exact 93 names are the `forced_heldout:` block of `config.yaml` (minus the six below),
and the machine-readable list is `$TRIPPY_OUTPUT/scratch/shade_frames.json`. Pass them to
`depthprior_shade_audit.py --frames` — that script's own default is `SHADE_FRAMES_KK`,
which as the next section shows is **not this place**.

### The kk-coherent shade frames are a *different* shady place

`IMG_3828`–`IMG_3833` are registered in `karekare-v2` and they are genuinely dark
(mean luminance **113.17** vs 121.31), so the same measurement finds them. But their camera
centroid is **5.79 world units** from the big-tree cluster. They are not the big tree. Two
further measurements say the same thing:

- Within 0.5 units of the `IMG_3828`–`IMG_3833` spot there are 74 registered frames whose
  mean luminance is 121.7 — i.e. **the location is not dark; the direction is**. The
  correlation between "angle away from the shade direction" and mean luminance there is
  **+0.592**, and the only 7 frames pointing within 40° of the shade are
  `IMG_3827`–`IMG_3833` (mean luminance 114.1).
- Their EXIF signature is the opposite one: ISO drops to the sensor floor of 32 across
  `IMG_3827`–`IMG_3836`, i.e. metering pulled *down* by something very bright in frame,
  not up by shade.

They are kept in `forced_heldout` anyway, so EXP-0011 still reports a number comparable
with every EXP-0003/0009/0010 run. **When reading results, split the shade group**: the
per-image dict in each `eval_*/metrics.json` carries every name, so `shade_bigtree` (93)
and `shade_kkc` (6) are recoverable without re-running anything. Do not average them and
call it "shade".

### The split: `forced_heldout_mode: alternate`

99 forced frames → **50 held out, 49 pushed into training** (alternating by sorted name,
so every sub-run above is half-observed). With `heldout_k: 16` on the remaining 657 frames
the run is **664 train / 92 held out (50 shade + 42 other)**.

This is the whole point of the experiment. `mode: all` asks "can it synthesise a shade
region it has never photographed?" (EXP-0003 full2: 8.49 dB). `alternate` asks "can it
render shade it *has* seen as shade?" — and it is also the protocol the Gaussian baseline
is implicitly measured under, since `kklid_20000` saw nearly all of these frames.

---

## Masks

`masks/` holds 1,376 PNGs, one per image, named by stem (`IMG_3683.jpg` → `IMG_3683.png`).

**Polarity: BLACK (0) = person, ignore. WHITE (255) = keep.** Established two ways:

1. *Documented.* `~/Splats/tools/make_masks.py:3`, `make_masks2.py:9`, `make_masks3.py:11`
   all state "Output convention: BLACK = ignore (person), WHITE = keep. Matches both COLMAP
   `--ImageReader.mask_path` and Brush's default masks folder."
2. *Measured.* Every sampled mask is mode `L`, strictly binary `{0, 255}`, at exactly the
   photo's own resolution. The white (keep) fraction over all 756 registered frames is
   **95.83% mean, 97.47% median, p90 90.0%, minimum 43.5%**, and **161 frames have nobody
   masked at all**. A few percent of a family photo being people is the right order of
   magnitude; an inverted read would have shown up here as ~4% keep, not 96%.

That polarity maps straight onto trippy's existing validity-mask convention (1 = the loss
may use this pixel), so a mask is simply multiplied into the mask `dataset.crop` already
returns. One mask, two reasons to be zero: crop overshoot, or a person. Every existing
consumer — L1, SSIM, LPIPS (by zeroing), PSNR, the exposure diagnostics — honours it
unchanged. Masks are resampled through the **same undistortion grid as the photo** but with
**nearest** interpolation, because a mask is a decision, not a signal.

`evaluate()` applies it too. That is deliberate: `kklid_20000` was itself trained with
these masks, so scoring TRIPS over pixels the splat was never asked to reconstruct would
compare two different quantities.

**Masks are not a rule here.** Jordan's kids are family, not noise, and he is curious what
unmasked looks like; the masks exist for comparability with the splat. So EXP-0011 queues
**both** arms — `kkv2-full-masked` and `kkv2-full-unmasked` — differing in exactly one
boolean. When comparing their PSNRs, check `masks.frac_masked_mean` in each
`eval_*/metrics.json` first: the two arms do not score the same pixels.

---

## Scale (measured, CPU, before any GPU time was committed)

| | |
|---|---|
| Dataset build, 756 images @ w1008 + masks | **132.6 s** (0.18 s/image) |
| Undistortion cache on disk | **2.91 GB** (pixels + masks; prebuilt, so no run pays it) |
| `GaussianPlySource(min_opacity=0.05)` | 8,910,382 → **7,542,137 points** (84.6% kept) |
| ...`size_mode: scale` | **2.7 s**, median size 0.002673 |
| ...`size_mode: knn` (4-NN, chunked, full cloud) | **31.1 s**, median size **0.005975** |
| knn / scale median ratio | **2.235** |
| Peak RSS for the whole point build | **4.12 GB** |
| Confidence (= source opacity) median | 0.1923; 70.0% below 0.3, 87.8% below 0.5 |
| Point bbox extent (world units) | 49.4 × 39.0 × 30.7 |

**The kNN worry was unfounded.** The brief allowed for a >10-minute kNN on 7–9M points and
a calibrated fallback. `scipy.spatial.cKDTree` does the whole 7.54M-point 4-NN pass in
**31 s**, so `size_mode: knn` is used directly, on the full cloud, with no subsampling and
no calibration factor. Nothing was approximated.

**Training memory.** 7.54M points × (3 xyz + 1 size + 1 conf + 8 feat) floats, ×3 for Adam's
two moments, ≈ **1.2 GB** of parameters + optimiser state. Fine.

**Eval full-frame pass — it fits.** TRIPS emits 4 bilinear fragments per (point, layer) and
`broadcast` writes all 5 layers, so a full frame costs `20 × (points passing the frustum
gate)` fragments. Measured on five representative views (two of them big-tree shade frames,
one a 1008×1344 portrait frame):

| view | shape | points visible | fragments | fragment arrays |
|---|---|---|---|---|
| `IMG_3703` (worst case seen) | 1008×756 | 1,892,232 (25.1%) | **37.8 M** | 1.82 GB |
| `IMG_4202` | 1008×756 | 960,898 (12.7%) | 19.2 M | 0.92 GB |
| `IMG_4032` (shade) | 1008×1344 | 958,818 (12.7%) | 19.2 M | 0.92 GB |
| `IMG_4300` (shade) | 1008×1344 | 581,256 (7.7%) | 11.6 M | 0.56 GB |
| `IMG_5660` (shade) | 1008×756 | 171,460 (2.3%) | 3.4 M | 0.16 GB |

3.4–37.8 M fragments, i.e. **0.16–1.82 GB** of fragment arrays for a full frame at 1008 —
inside the brief's 25–60 M expectation and comfortably inside memory. Full data in
`$TRIPPY_OUTPUT/scratch/frag_estimate.json`. Note the median projected point size is only
1.2–4.0 px even with kNN sizes, which is the same sub-pixel regime EXP-0003 measured
(`t_final ≈ 0.93`, the U-Net inventing ~90% of every frame) — expect that to persist here.

**Confidence, and why the removal arm is `relative`.** 70% of this cloud's points are
already below TRIPS's 0.3 cutoff at epoch 0 and 88% below its shipped 0.5 — because trippy
seeds confidence from the source PLY's opacity, not from TRIPS's uniform
`sigmoid(10·0.5) = 0.9933`. An absolute cutoff would therefore mostly measure where a point
*started*. `config_removal.yaml` uses EXP-0010 arm A' (`mode: relative`, `rel_factor: 0.3`,
`conf_threshold: 0.1` as a floor), which is the faithful analogue. `min_points` is scaled
from EXP-0010's 1M to **4M**, since this cloud is ~4× `kkc_15000`.

---

## Configs

| file | run_dir basename | what differs |
|---|---|---|
| `config_smoke.yaml` | `kkv2-0-smoke` | w504, 300k points, 2 epochs — proves rc 0 before ~7 h of GPU |
| `config.yaml` | `kkv2-1-full-masked` | the run: w1008, knn sizes, 300 epochs, `train_factor 1.0`, masks on |
| `config_unmasked.yaml` | `kkv2-2-full-unmasked` | `use_masks: false`. Otherwise byte-identical |
| `config_removal.yaml` | `kkv2-3-removal` | + EXP-0010 arm A' point removal (`mode: relative`) |
| `config_hybrid.yaml` | `kkv2-5-hybrid` | + design A: the `kklid_20000` render fed to the U-Net |

All five carry `eval_exposure_mode: neighbours`, `forced_heldout_mode: alternate`, the same
99-frame shade list, `heldout_k: 16`, and **absolute** `run_dir`s under
`/Users/nzbirdranch/trippy/output/runs/EXP-0011-karekare-v2/` (these are queued from the
`.worktrees/karekare-v2` worktree; a relative `run_dir` would resolve inside it and be lost
when the worktree is removed — the way EXP-0005's renders were).

### Hybrid renders

`config_hybrid.yaml` needs 756 rgb/depth/alpha triples from `kklid_20000` at this scene's
own undistorted 1008-wide grid, in
`/Users/nzbirdranch/trippy/output/hybrid-v2/renders/w1008/`. They are produced by three
sharded queue jobs (`kkv2-4-render-1/2/3`, 252 views each) that run *before* the hybrid
training is queued. `trippy.hybrid.render_splat_views` reuses `SceneDataset`, so each render
shares its photo's exact `(H, W, K)` by construction, and `gsrender` takes `K` per view from
that cached meta — the scene's 202 camera models cost it nothing. `max_hw` is passed
explicitly (never gsrender's own default of 32, which corrupts near-camera footprints).
`missing: zeros` means a dead shard still trains rather than crashing; check the shard
manifests before trusting the numbers.

---

## CPU verification (run before the queued MPS smoke, so the queue is not the long pole)

The queued smoke sits behind ~7 other prio-70 jobs, so the identical config was run on
**CPU** first — same 756-image dataset, same masks, same 2.1 GB PLY, same split, real
`train_step`s and a real `evaluate()`. It is the same code the MPS smoke will run; only the
device differs, so the *timings* below are CPU timings and nothing else is.

| | |
|---|---|
| `Trainer.__init__` (w504 cache build for 756 images + 300k-point source) | 78.0 s |
| Points | 300,000 (`max_points`, subsampled before kNN) |
| Split | **664 train / 92 held out**, of which **50 shade held out, 49 shade in training** |
| Masks | ON, `frac_masked_mean` **4.19%**, max **56.5%**, 161 frames fully visible |
| CPU step time | 0.11 s |
| `mask_excluded_frac` over 10 crops | 0.00, 0.31, 0.19, 0.00, 0.00, 0.00, 0.00, 0.09, 0.00, 0.49 |
| Loss, step 1 → step 10 | 0.7687 → 0.4954 |
| Non-finite gradients | **0** |
| CPU eval | 0.5 s/frame |
| Held-out PSNR after **10 optimiser steps** | 11.32 dB (`psnr_mean_eval` 11.10, `exposure_mode: neighbours`), SSIM 0.193 |
| ...shade / other | 10.97 dB / 11.67 dB |

The per-crop `mask_excluded_frac` column is the load-bearing one: it is 0 on crops with
nobody in them and up to 0.49 on crops with a child in them, which is the mask actually
reaching individual crops rather than being computed and dropped. Per-frame eval exclusions
ranged 0.0–18.3% across the eight frames scored.

Artefacts: `$TRIPPY_OUTPUT/runs/EXP-0011-karekare-v2/kkv2-cpu-verify/`,
`$TRIPPY_OUTPUT/scratch/cpu_smoke.json`.

---

## Queue

Every job at **prio 70** (behind Splats' own 60). No queue jumping: these sit behind the
six prio-70 jobs already queued. `scripts/queue_training.sh` submits
`trippy train --config … --report`, so each run self-reports with a viewer launcher and
needs no follow-up step.

The runner picks the lowest-sorted filename within a priority, so the **order digit in each
run_dir basename is what sequences them** — smoke before the 7-hour trainings, render shards
before the hybrid:

| # | job | submit line |
|---|---|---|
| 1 | `trippy-kkv2-0-smoke` | `scripts/queue_training.sh experiments/EXP-0011-karekare-v2/config_smoke.yaml --max-minutes 40` |
| 2 | `trippy-kkv2-1-full-masked` | `scripts/queue_training.sh experiments/EXP-0011-karekare-v2/config.yaml --max-minutes 420` |
| 3 | `trippy-kkv2-2-full-unmasked` | `scripts/queue_training.sh experiments/EXP-0011-karekare-v2/config_unmasked.yaml --max-minutes 420` |
| 4 | `trippy-kkv2-3-removal` | `scripts/queue_training.sh experiments/EXP-0011-karekare-v2/config_removal.yaml --max-minutes 420` |
| 5–7 | `trippy-kkv2-4-render-{1,2,3}` | `scripts/gpu_submit.sh --train kkv2-4-render-N -- python -m trippy.hybrid.render_splat_views --scene …/karekare-v2 --ply …/kklid_20000.ply --out $TRIPPY_OUTPUT/hybrid-v2/renders/w1008 --width 1008 --device mps --start-index S --end-index E` (shards 0–252, 252–504, 504–756) |
| 8 | `trippy-kkv2-5-hybrid` | `scripts/queue_training.sh experiments/EXP-0011-karekare-v2/config_hybrid.yaml --max-minutes 420` |

All eight returned `submit.sh rc=0`; the exact lines are also in `research/trips-metal.md`.

**`--max-minutes 420` will not reach epoch 300.** At ~664 steps/epoch the full run is
~15 h; the budget stops it cleanly at roughly epoch 150 with a checkpoint and an eval
written, and `--resume` picks it up. That is deliberate — 300 epochs is the schedule the
LR/lock/VGG *fractions* are computed against, not a promise to run them all.

> ⚠️ **`.worktrees/karekare-v2` must survive until all eight jobs finish.** The generated
> job files `cd` into the worktree and put it on `PYTHONPATH`. The `run_dir`s are absolute,
> so the *artefacts* are safe either way, but removing the worktree early kills the jobs —
> the same trap that lost EXP-0005's renders. Remove it with `scripts/worktree_rm.sh
> karekare-v2` afterwards.

---

## Results

_Placeholders — filled in as each run reports._

| run | epochs | held-out PSNR (`psnr_mean_eval`) | shade (big tree) | shade (kkc) | other | dark mass | verdict |
|---|---|---|---|---|---|---|---|
| `kkv2-0-smoke` | 2 | | | | | | |
| `kkv2-1-full-masked` | | | | | | | |
| `kkv2-2-full-unmasked` | | | | | | | |
| `kkv2-3-removal` | | | | | | | |
| `kkv2-5-hybrid` | | | | | | | |

Baselines to beat: EXP-0003 `full2-broadcast` on `kk-coherent` scored all 17.12 dB / shade
15.27 dB under neighbour-exposure eval, against plain Gaussians at 15.53 / 14.94. Those are
a *different scene and a different shade region*, so they are context, not a target. The
target here is the Gaussian baseline on this scene, and ultimately Jordan's viewer verdict —
which has overruled the metrics before.

**Stage gate (docs/SPEC.md):** "shade rendered as shading, not a cloud". Measured with
`depthprior_shade_audit.py`, and **the `--frames` argument must be the measured 93-frame
big-tree list above**, not the script's `SHADE_FRAMES_KK` default.
