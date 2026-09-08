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
| `config_hybrid_gate.yaml` | `kkv2-7-hybrid-gate` | `config_hybrid.yaml` + the **blend gate** (see below) |
| `config_fullres_smoke.yaml` | `kkv2-9-fullres-smoke` | w2016/crop512 (unshrunk), 140 images, 2 epochs, resumes the real kkv2-1 checkpoint — proves rc 0 before the 12 h full-res run |
| `config_fullres.yaml` | `kkv2-9-fullres` | w2016/crop512, 300 epochs, **seeded from `kkv2-1-full-masked`'s checkpoint_latest.pt (epoch 122)** via `--resume` — see "Full-resolution variant" below |

All eight carry `eval_exposure_mode: neighbours`, `forced_heldout_mode: alternate`, the same
99-frame shade list, `heldout_k: 16`, and **absolute** `run_dir`s under
`/Users/nzbirdranch/trippy/output/runs/EXP-0011-karekare-v2/` — a relative `run_dir` would
resolve inside whatever working copy submitted the job and be lost when that copy is removed,
the way EXP-0005's renders were. The first six were queued from the `.worktrees/karekare-v2`
worktree (still must survive until their jobs finish, see the warning below). The two
full-resolution jobs (`kkv2-9-fullres-smoke`, `kkv2-9-fullres`) were queued from
`.worktrees/kkv2-fullres` but their generated job scripts `cd` into the **main** checkout
(`/Users/nzbirdranch/trippy`) rather than that worktree, precisely so removing
`.worktrees/kkv2-fullres` after review cannot kill them the way an early worktree removal has
before — see "Full-resolution variant" below for the one thing that dependency requires.

### The blend-gate arm (`config_hybrid_gate.yaml`, `kkv2-7-hybrid-gate`)

Byte-identical to `config_hybrid.yaml` except the `run_dir` and four keys at the bottom of the
`hybrid:` block, so "gate vs no gate on karekare-v2" is a one-variable comparison against
`kkv2-5-hybrid`:

```yaml
  gate:
    enabled: true          # one extra U-Net output channel: g in [0, 1]
  gate_prior:
    target: 0.5
    weight: 0.0            # OFF -- see below
  gate_scale: 1.0          # the default mix eval/report/viewer open at
```

**What it buys.** `kkv2-5-hybrid` can tell us *whether* feeding the splat to the U-Net helps.
It cannot tell us *where*, or *how much*, because the mix is implicit in the weights. This arm
makes it a tensor: the displayed image is `g * splat_rgb + (1 - g) * trips_rgb`, and every eval
writes the gate map out as a heatmap (`eval_ep*/gate/*.gate.png`, from-scratch colour ramp, no
photographed pixels) plus mean/percentiles in `metrics.json` and `report.json`. See
docs/EXPERIMENTS.md "The blend gate".

**Read the map, not just the PSNR.** Three outcomes are interesting, in order:

1. The gate is a **structured map** rather than a constant — the two renderers are genuinely
   complementary, which is design A's whole thesis.
2. The gate is near 0 in the shade — TRIPS is carrying the big tree and the splat is not.
3. The gate is near 1 in the shade — the splat is, and TRIPS is not.

A constant map either way says the mix is not where the win is.

**Why the prior is off.** The first question is "what mix does this scene choose when nothing
pushes it?"; a prior would answer a different one. If the unpushed answer is degenerate, re-run
with `weight: 0.1` and `target: 0.2` / `0.8` and see what the scene gives up.

**Nothing has to be retrained to change the mix.** `trippy eval --gate-scale 0|1|2` re-scores
the same checkpoint at pure TRIPS / as trained / pushed-to-splat, `trippy candidate-report
--gate-scale` re-renders it, and the viewer's Blend panel moves it live.

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

### Full-resolution variant (`kkv2-9-fullres`, `kkv2-9-fullres-smoke`)

**The question.** Jordan's 2026-09-08 19:10 verdict on `kkv2-1-full-masked` /
`kkv2-2-full-unmasked`: the shade fix passed ("no longer a cloud"), but the cost was
"literally everything else looks worse: fuzzy and pixelated across the scene". Two live,
non-exclusive explanations: the run stopped at epoch 122 of 300 (under-trained), or width
1008 / crop 384 is intrinsically too coarse for this scene (README "Scale": median
projected Gaussian footprint is only 1.2–4.0 px even with kNN sizes, so the U-Net is
inventing ~90% of every frame from a sub-pixel signal). This variant isolates the second
explanation: same scene, same masks, same split, same point source, same 300-epoch
schedule, at **width 2016 / crop 512**, so a sharper or unchanged result at equal epochs
reads directly on "is this a resolution limit or a training-time limit".

**2016 is not native resolution — it is 2x the trained width.** `karekare-v2`'s 202 camera
models range 2160–5712 px wide natively (measured via `colmap_io.load_colmap_model`,
2026-09-08), and `SceneDataset` takes one destination width for every camera, deriving each
one's height from its own aspect ratio — there is no single width that is "the" native
resolution across a scene this heterogeneous. 2016 is the largest round 2x step that stays
strictly below the smallest native camera width (2160), so undistorting at 2016 never
upsamples any of the 756 registered photos. A literal per-camera native-resolution run
would need a variable-width dataset, which `SceneDataset` does not support and this task
did not add.

**Seeded, not trained from scratch.** `--resume` (a CLI flag, not a config field) is passed
against `kkv2-1-full-masked/checkpoints/checkpoint_latest.pt` — **epoch 122**, the
most-trained checkpoint available when this was queued (2026-09-08), not `checkpoint_best.pt`
(only epoch 40, the best-held-out-PSNR-so-far checkpoint from early in a run that kept
improving — resuming from it would throw away 82 epochs of training for a worse starting
point). The comparison this buys is "more resolution from the same point in training", not
"more resolution from scratch".

**Does `--resume` tolerate a different width/crop? Checked two ways, not assumed.**
1. *By code*: `Trainer.resume`/`load_state` (`trippy/train/trainer.py`) load
   `point_params`, `pose_params`, `net`, `camera`, `background`, `optimizer` and `scheduler`
   state; none of it is read from, or checked against, `cfg.width`/`cfg.crop`. The U-Net
   (`trippy/net/unet.py::MultiScaleUnet2dDecOnlySmallFixed`) is fully convolutional — no
   `Linear`/`AdaptiveAvgPool` layer whose shape depends on input resolution — and
   `PoseParams`/camera state are keyed by image **index**, not pixel geometry, so they only
   line up correctly if the resumed config keeps the exact same `scene_root`,
   `forced_heldout`, `heldout_k` and `forced_heldout_mode` as the checkpoint's own run (this
   config does, byte-for-byte, so the sorted 756-name index order is identical).
2. *Empirically*, on the synthetic fixtures (`tests/test_train_helpers.py`, no scene
   imagery): a `Trainer` built at width 48 / crop 24, trained two steps and checkpointed,
   was resumed by a **second** `Trainer` built at width 96 / crop 32 (both width *and* crop
   changed). It loaded with no error, matched the first trainer's `point_params.xyz` and
   `net.state_dict()["final.0.weight"]` exactly, and then ran a further real `train_step()`
   and `evaluate()` at its own new width/crop with no error. `RESUME ACROSS WIDTH/CROP:
   PASS` (ad hoc CPU check, 2026-09-08, not checked in as a test file — the assertions
   above are what it proved).

Point count is a separate story: `load_state` resizes `point_params` to the **checkpoint's**
point count before loading it (`_resize_point_params`), so `kkv2-9-fullres`'s and
`kkv2-9-fullres-smoke`'s own `point_source:` blocks are moot once `--resume` runs — they
exist only so a future `--resume`-less arm of either file is still well-formed on its own.
The real run inherits kkv2-1's exact 7,542,137 points.

**Memory/time estimate**, from the measured 1008 numbers (`kkv2-1-full-masked-resume`:
~3.9 min/epoch at width 1008 / crop 384 / 664 steps per epoch):
- Steps/epoch is unchanged (`train_factor: 1.0` × 664 train images = 664 steps): the split
  is byte-identical to `config.yaml`.
- Per-step cost is dominated by crop area for training (points are frustum-culled to what
  is visible in the crop; the render itself emits 4 bilinear fragments per point per layer
  **regardless of a point's pixel footprint**, so fragment count does not scale with width
  on its own — see the "Eval full-frame pass" table below). Crop area ratio
  (512/384)² ≈ **1.78x**. Estimated training time: 3.9 min × 1.78 ≈ **~6.9 min/epoch**.
- Full-frame **eval** passes (`eval_every: 20`, `eval_max_images: 8`) do scale with total
  pixel count — the compositing buffer and the U-Net's per-pixel decode are both O(pixels)
  — so eval frames cost roughly **4x** their 1008 figures (README "Scale" table: 0.16–1.82
  GB fragment-array peak per frame at 1008 → an estimated 0.6–7 GB at 2016), but eval runs
  on only 8 of 756 images once every 20 epochs, so it does not dominate the per-epoch
  average.
- Dataset cache: **~4x** the 1008 numbers (756 images × ~4x the pixels/image). Measured
  1008 build was 132.6 s / 2.91 GB; estimated 2016 build **≈530 s (~9 min) one-time, ≈11.6
  GB on disk**. This happens automatically inside the job (CPU-only `grid_sample`
  undistortion, the same code path `SceneDataset._load_or_build_cache` always runs) —
  no separate submission needed, and it is not "direct GPU/MPS work" (device is irrelevant
  to that step; it always runs on CPU).
- Point-cloud build/kNN cost is width-independent (world-space; unaffected by resolution) —
  unchanged from the ~31 s / 4.12 GB peak RSS already measured, and moot anyway once
  `--resume` overwrites it.
- At ~6.9 min/epoch, the queued `--max-minutes 720` (12 h) budget advances roughly
  **~100 epochs** from wherever the run resumes — like every other 300-epoch arm here, one
  12 h job will not reach epoch 300 in a single shot; it is expected to need at least one
  further `--resume` continuation (the same pattern `kkv2-1-full-masked` → `-resume` →
  `kkv2-8-full-masked-cont` already went through).

**Acceptance.** Sharper held-out PSNR/LPIPS than `kkv2-1-full-masked` **at equal epoch
count** (compare `psnr_mean_eval`/`lpips_mean` at the same `epoch` in each run's
`metrics.jsonl`, not at "run finished" — the two runs will not finish at the same epoch on
the same wall-clock budget), AND shade stays shading (Jordan's viewer verdict, not the dark-
mass audit — the 2026-09-08 19:10 verdict already retired dark-mass as a shade pass/fail
signal on this scene). If PSNR/LPIPS improves but the viewer still calls it fuzzy, or if
resolution alone does not move either number, the defect is more likely epoch budget or a
genuine limit of TRIPS + these images, not pixel count — in which case the next lever is
finishing `kkv2-8-full-masked-cont` to real epoch 300 at the *current* resolution, so the
two variables (epochs, resolution) are not still confounded.

**Queue (2026-09-07 queue policy, not the prio-70 table above).** Both jobs were submitted
via `scripts/gpu_submit.sh` directly, not `scripts/queue_training.sh` (that script hardcodes
`--train`, i.e. prio 70; these need the target-scene band, prio 40, and a short prio-15
smoke):

| job | prio | submit line |
|---|---|---|
| `trippy-kkv2-9-fullres-smoke` | 15 | `scripts/gpu_submit.sh --prio 15 kkv2-9-fullres-smoke -- trippy train --config experiments/EXP-0011-karekare-v2/config_fullres_smoke.yaml --resume <kkv2-1 checkpoints>/checkpoint_latest.pt --device mps --max-minutes 30 --report` |
| `trippy-kkv2-9-fullres` | 40 | `scripts/gpu_submit.sh --prio 40 kkv2-9-fullres -- bash -c '<rc guard, see below>; python -m trippy.cli train --config experiments/EXP-0011-karekare-v2/config_fullres.yaml --resume <kkv2-1 checkpoints>/checkpoint_latest.pt --device mps --max-minutes 720 --report'` |

The full job's guard refuses to start unless the smoke already succeeded:
`RC=.../gpu_queue/done/trippy-kkv2-9-fullres-smoke.rc; [ -f "$RC" ] && [ "$(cat "$RC")" = "0" ]`,
else it exits 97 without touching the GPU. `kkv2-9-fullres` sorts after every existing
`kkv2-*` prio-40 job by filename (`8` < `9`), so it does not jump the queue.

> ⚠️ **Both job scripts `cd` into the MAIN checkout (`/Users/nzbirdranch/trippy`), not this
> worktree — which means `experiments/EXP-0011-karekare-v2/config_fullres{,_smoke}.yaml`
> must exist in main before either job's turn comes up.** The smoke sits behind only two
> short prio-15 jobs (`edit-sam-5-cpu`, `edit-sam-5`) when this was queued, so it could start
> within minutes to an hour. **This branch (`exp/kkv2-fullres`) must be reviewed and merged
> — or the two config files otherwise placed in main — before that happens**, or the smoke
> fails immediately with a missing-config error and the guarded full job never starts either.
> This is a deliberate change from the `.worktrees/karekare-v2` pattern above (cd into the
> submitting worktree, keep it alive until the jobs finish): it trades "the worktree must
> survive" for "main must have the configs first", because `.worktrees/kkv2-fullres` is a
> short-lived review branch, not a long-queue holding pen.

Both submissions also wrote their own `research/trips-metal.md` one-liner (`submitted job
...`) directly into the **main** checkout's live file, a side effect of `gpu_submit.sh`
always logging relative to whatever `REPO_ROOT` a job's `cd` target uses — since these jobs
correctly `cd` into main, that line landed in main's working tree, not in this branch's
diff. The Orchestrator should commit those two lines in main (or fold them in when merging
this branch) rather than expect them to arrive via this PR.

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
| 9 | `trippy-kkv2-7-hybrid-gate` | `scripts/gpu_submit.sh --prio 45 kkv2-7-hybrid-gate -- trippy train --config <abs>/experiments/EXP-0011-karekare-v2/config_hybrid_gate.yaml --report --max-minutes 420` |

Job 9 is at **prio 45** (the hybrid band under the 2026-09-07 queue policy) and is named
`kkv2-7`, not `-6`, so that once the earlier kkv2 jobs are rebanded to prio 40 the filename
order keeps it behind `kkv2-6-shade-prune`. **Until that rebanding happens it sorts ahead of
them** (they are still at 70) and ahead of Splats' own `60-hunua-run01` — reband the kkv2 jobs
before this one's turn comes, or hold it.

The first eight returned `submit.sh rc=0`; the exact lines are also in `research/trips-metal.md`.

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
| `kkv2-7-hybrid-gate` | | | | | | | |
| `kkv2-9-fullres-smoke` | 2 | | | | | | |
| `kkv2-9-fullres` | | | | | | | |

For `kkv2-7-hybrid-gate`, also record the gate: mean and p5/p50/p95 from
`report.json`'s `gate.held_out` block, whether the map is structured or constant, and the
held-out PSNR re-scored at `--gate-scale 0` (pure TRIPS) and `2` (pushed to the splat).

Baselines to beat: EXP-0003 `full2-broadcast` on `kk-coherent` scored all 17.12 dB / shade
15.27 dB under neighbour-exposure eval, against plain Gaussians at 15.53 / 14.94. Those are
a *different scene and a different shade region*, so they are context, not a target. The
target here is the Gaussian baseline on this scene, and ultimately Jordan's viewer verdict —
which has overruled the metrics before.

**Stage gate (docs/SPEC.md):** "shade rendered as shading, not a cloud". Measured with
`depthprior_shade_audit.py`, and **the `--frames` argument must be the measured 93-frame
big-tree list above**, not the script's `SHADE_FRAMES_KK` default.
