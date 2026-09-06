# EXP-0003: TRIPS training on kk-coherent (point source 1, trained Gaussian centres)

## Question

Trained end to end on kk-coherent starting from trained 3DGS Gaussian centres (point
source 1, docs/SPEC.md D4), does the TRIPS-style pyramid + U-Net pipeline (a) reach
held-out PSNR within 1.5 dB of the best plain Gaussian on non-shade frames, and (b) show
the shade region under the trees rendered as *shading*, not a cloud, in a viewer? This is
the first of the three point-source experiments (1 = Gaussians, 2 = monocular depth,
3 = union) the v0.2.0 stop-or-go gate compares (docs/SPEC.md "Stop-or-go point: v0.2.0").

## Point source

1 = `GaussianPlySource` on
`$SPLATS_ROOT/output/Training-Data/karekare/kk-coherent/kkc_15000.ply` (7.36M trained
Gaussian centres before the opacity filter; `min_opacity=0.05` in the configs below drops
near-transparent points, per `docs/EXPERIMENTS.md`'s point-sources table).

## Configuration

Two config files, both under this directory (see `docs/EXPERIMENTS.md` "Training runs"
for the config file format -- every field not listed keeps `TrainConfig`'s
scaled-for-trippy default, cited in `trippy/constants.py`):

- **`config_smoke.yaml`** -- first queue-job rehearsal: `width=504`, `crop=256`,
  `max_points=200000`, `epochs=2`, `eval_every=1`. Small enough to finish in minutes and
  prove the queue round-trip (dataset cache build, point source load, one train step, one
  eval, one checkpoint, one export) before committing compute to the full run.
- **`config.yaml`** -- the real run: `width=1008`, `crop=384`, `mode=trilinear`,
  `layers=5`, `epochs=150`, full (unfiltered by `max_points`) Gaussian point cloud.
  `forced_heldout` is `trippy.constants.SHADE_FRAMES_KK` (`IMG_3828.jpg`..`IMG_3833.jpg`)
  so every eval reports a shade-region number, not just an average over easy frames.

Both target `device: mps` and must only ever run via `scripts/gpu_submit.sh` (AGENTS.md
"GPU and compute" -- never invoked directly). The trainer itself is CPU-testable end to
end on a synthetic scene (`tests/test_train_*.py`); this experiment is the first time it
runs against a real scene and real compute.

## Planned commands

```bash
# 1. Smoke run: prove the queue round-trip on a small config.
bash scripts/gpu_submit.sh --prio 15 kk-trips-train-smoke -- \
  trippy train --config experiments/EXP-0003-kk-trips-train/config_smoke.yaml

# 2. Full training run (overnight, prio 70 behind Splats' own jobs).
bash scripts/gpu_submit.sh --train kk-trips-train -- \
  trippy train --config experiments/EXP-0003-kk-trips-train/config.yaml

# 3. Resume if the job is interrupted (queue timeout, machine restart).
bash scripts/gpu_submit.sh --train kk-trips-train-resume -- \
  trippy train --config experiments/EXP-0003-kk-trips-train/config.yaml \
    --resume output/runs/EXP-0003-kk-trips-train/EXP-0003-kk-trips-train_1/checkpoints/checkpoint_latest.pt

# 4. Standalone re-evaluation of the final checkpoint (no re-training).
trippy eval --checkpoint output/runs/EXP-0003-kk-trips-train/EXP-0003-kk-trips-train_1/checkpoints/checkpoint_latest.pt

# 5. Shade audit + extent gate against the exported PLY (docs/EXPERIMENTS.md).
python ~/Splats/tools/depthprior_shade_audit.py \
  --scene ~/Splats/scenes/karekare/kk-coherent/sparse_txt \
  output/runs/EXP-0003-kk-trips-train/EXP-0003-kk-trips-train_1/export.ply
python ~/Splats/tools/tmp/extent-audit/extent_gate.py \
  output/runs/EXP-0003-kk-trips-train/EXP-0003-kk-trips-train_1/export.ply
```

## Gate

**v0.2.0 acceptance** (docs/SPEC.md): held-out PSNR within 1.5 dB of the best plain
Gaussian on non-shade frames; shade dolly shows shading, not a cloud; shade audit number
drops vs. `kkc_15000.ply`'s own audit; extent (p99/p99.9/max radius) not inflated beyond
+20% of the initial COLMAP/Gaussian bbox; honesty sheet reviewed. This experiment alone
does not decide the v0.2.0 stop-or-go point -- that requires point sources 2 (monocular
depth) and 3 (union) to also run and be compared (docs/SPEC.md "Stop-or-go point:
v0.2.0"); this README will be updated with cross-references once those experiments exist.

## Verdict

(Filled in after run -- placeholders below.)

| Metric | Value |
|---|---|
| Held-out PSNR (non-shade frames) | TBD |
| Held-out PSNR (shade frames, `SHADE_FRAMES_KK`) | TBD |
| Held-out SSIM | TBD |
| Held-out LPIPS | TBD |
| Shade audit (opacity mass in shade region) | TBD, vs. `kkc_15000.ply` baseline TBD |
| Extent (p99 / p99.9 / max radius) | TBD, vs. initial bbox +TBD% |
| Wall-clock (150 epochs on M3 Ultra) | TBD |
| Jordan's viewer verdict | TBD |

Artifact path (once run): `output/runs/EXP-0003-kk-trips-train/EXP-0003-kk-trips-train_1/`
(`export.ply`, `eval_ep*/sheet.png`, `metrics.jsonl`, `checkpoints/`). Dolly video and
delivery follow via `scripts/deliver.sh` once the run completes.

## full1-broadcast candidate numbers

First real candidate report against this experiment's `full1-broadcast` run (`mode: broadcast`,
epoch 39; `output/runs/EXP-0003-kk-trips-train/full1-broadcast/candidate/report.json` +
`README.md`, baseline audit at `output/runs/EXP-0003-kk-trips-train/baseline_shade_audit.json`
against the source `kkc_15000.ply`). These are real numbers, distinct from the still-TBD
`config.yaml` (150-epoch) run's own verdict table above; docs/EXPERIMENTS.md "Self-reporting
training runs" describes the `trippy train --report`/`comparison_table_markdown` machinery that
will produce this table automatically for every future candidate.

| Metric | Baseline (`kkc_15000.ply`) | Candidate (`full1-broadcast`, epoch 39) |
|---|---|---|
| Held-out PSNR (dB) | n/a | 14.42 |
| Held-out SSIM | n/a | 0.3900 |
| Held-out LPIPS | n/a | 0.5131 |
| Shade dark-mass fraction (lum<0.25) | 19.9% (67,068.8 / 336,873.5) | 36.2% (124,120.0 / 342,813.4) |
| Extent radius p99 | n/a (not audited yet) | 40.02 |
| Extent radius max | n/a (not audited yet) | 124.48 |
| Dolly mean centre coverage (kept path: frames 0..28 of 47, `dolly_stop_index`=28 at threshold 0.05) | n/a | 0.26 (0.46 at `t=-0.35` falling to 0.08 at `t=+0.51`; unstopped mean over all 48 frames is 0.16 -- see docs/EXPERIMENTS.md "Dolly camera paths" stop rule) |

The dark-mass fraction moved the *wrong* direction (candidate darker than baseline, not
lighter) -- this run does not clear the shade-audit gate yet; PSNR (14.42 dB) is also well
below the v0.2.0 target. Recorded here as a real data point, not a pass. Jordan's viewer
verdict on the dolly video/honesty sheet is still the actual call (docs/EXPERIMENTS.md
"Jordan's viewer verdict is final").


## full2-broadcast (300 epochs): the shade verdict, and how much of it is a measurement artefact

Headline numbers as reported (`output/runs/EXP-0003-kk-trips-train/full2-broadcast/`,
`metrics.jsonl`'s last eval row, epoch 299): held-out all **15.02 dB** (Gaussians 15.53),
held-out shade **8.49 dB** (Gaussians 14.94), other held-out **16.47 dB**.

### The exposure artefact (found 2026-09-06, CPU-side, no re-render)

Four of the six shade frames have **no EXIF ExposureTime/ISO** in the scene cache. `Trainer.
_initial_exposure` used to fall back to an absolute `EV = 0` and then subtract the scene mean
(5.87 EV), giving those frames a *relative* EV of -5.87 -- i.e. `NeuralCamera` multiplied their
prediction by `2 ** 5.87 = 58.5x` before the response LUT. A held-out frame's exposure is never
trained (only sampled training frames get gradients), so that 58x stood for all 300 epochs.

| frame | EXIF ExposureTime/ISO | init EV (rel) | gain `2**-EV` | PSNR (dB) | SSIM |
|---|---|---|---|---|---|
| IMG_3828.jpg | 0.01010 / 32 | -0.88 | 1.85x | 12.45 | 0.353 |
| IMG_3829.jpg | **missing** | -5.87 | 58.50x | 6.28 | 0.206 |
| IMG_3830.jpg | 0.01010 / 32 | -0.88 | 1.85x | 12.29 | 0.356 |
| IMG_3831.jpg | **missing** | -5.87 | 58.50x | 6.59 | 0.258 |
| IMG_3832.jpg | **missing** | -5.87 | 58.50x | 6.50 | 0.322 |
| IMG_3833.jpg | **missing** | -5.87 | 58.50x | 6.82 | 0.316 |

Ten of the 219 registered kk-coherent images have no usable EXIF; six of them landed in the
held-out split -- four shade frames plus `IMG_3703.jpg` and `IMG_3896.jpg`, which are *not*
shade frames and scored 6.19 dB and 6.92 dB. **The ~6.5 dB cluster is the signature of the 58x
gain, not of the shade.** Splitting the held-out set by EXIF presence instead of by region:

| held-out group | n | mean PSNR (dB) |
|---|---|---|
| all held-out (reported) | 33 | 15.02 |
| shade (reported) | 6 | 8.49 |
| other (reported) | 27 | 16.47 |
| all held-out, EXIF present | 27 | 16.90 |
| shade, EXIF present | 2 | 12.37 |
| shade, EXIF missing (58x gain) | 4 | 6.55 |
| other, EXIF present | 25 | 17.26 |
| other, EXIF missing (58x gain) | 2 | 6.56 |

Read that table plainly:

- The six broken frames average **6.55 dB whether or not they are shade frames** (6.55 shade,
  6.56 other). That number measures exposure, nothing else.
- On the frames whose exposure was initialised sanely, the shade deficit is **12.37 dB vs
  17.26 dB = -4.89 dB**, not the -7.98 dB the reported split implies.
- Correcting only the exposure artefact would move the reported shade number from 8.49 dB to
  roughly 12.4 dB and the all-held-out number from 15.02 dB to about 16.9 dB.
- **A real shade deficit remains** (~4.9 dB below non-shade frames on the same run). Item 1
  explains roughly half the reported gap; it does not explain it away.

Fixed in `Trainer._initial_exposure`: an image with no usable EXIF now starts at the scene mean
(relative EV 0, gain 1.0), and the trainer logs how many images that applies to. This does not
retro-fix `full2-broadcast`'s checkpoint -- its held-out exposures are stored as trained -- which
is what `trippy eval --calibrate` is for.

### Test-time camera calibration: `trippy eval --checkpoint ... --calibrate`

Fits **only** each held-out image's own exposure (optionally its red/blue white balance) against
its own photo, everything else frozen, and reports the calibrated numbers next to the strict ones
(docs/EXPERIMENTS.md "Test-time camera calibration"). Precedent: TRIPS itself ships
`optimize_eval_camera`, a per-epoch gradient pass over the *test* crops that steps the camera and
pose optimisers with texture/network frozen (`third_party/TRIPS/src/apps/train.cpp:591-596,
693-697`), and `interpolate_eval_settings`, which copies a test frame's exposure from its
neighbouring train frames (`NeuralCamera.cpp:481-520`). Both default to false there, which is why
trippy's default is off too and why the calibrated number is always quoted as a diagnostic.

GPU job `eval-calib-1` (prio 15) reports before/after per frame -- including mean predicted vs
photographed brightness, the closed-form best global gain and the PSNR it buys. Those per-frame
brightness numbers cannot be computed from the existing eval outputs: `metrics.jsonl`'s
`per_image` rows carry PSNR/SSIM/LPIPS only, and no predicted image is written to disk (the eval
sheet is a lossy JPEG of photo panels, which agents must not open). They come from the job.

**Job status (2026-09-06 14:35):** `trippy-eval-calib-1` is queued at prio 15 behind the running
`full2-trips` training (`--max-minutes 330`, started 13:44, ~4 min/epoch), so it starts in the
evening, not during the session that wrote this. Its numbers land in
`output/runs/EXP-0003-kk-trips-train/full2-broadcast/eval_manual_<ts>/metrics.json`, in a fresh
`{"eval": true, ..., "calibrated": true}` row of that run's `metrics.jsonl`, and -- after the next
`trippy leaderboard` -- in the leaderboard's "Held-out shade PSNR (calibrated)" column. Paste the
before/after table here when it lands.

<!-- eval-calib-1 results: filled in below when the job completes -->

### Two hold-out protocols, and an unfair baseline

`full2-broadcast` holds out **all six** consecutive shade frames, so the training set contains no
photograph of the shade region at all: the U-Net has to invent it. That is a legitimate and hard
question (novel view of an unobserved region) -- but it is *not* the question the Gaussian
baseline answers.

**`kkc_15000` never held the shade out.** It was trained with Brush's `--eval-split-every 10`
(`~/Splats/research/kk-coherent.md:61-67`), holding out 22 of the 219 registered views. Of the six
shade frames, only `IMG_3829.jpg` was held out; `IMG_3828/3830/3831/3832/3833` were **training
views**, including `IMG_3830.jpg`, the dolly anchor. Of the 33 frames in trippy's held-out split,
27 were Gaussian training views.

So "TRIPS 8.49 dB vs Gaussians 14.94 dB in the shade" compares a genuine novel-view score against
what is, for 5 of 6 frames, a training-set reconstruction score. The comparison is unfair to TRIPS,
and this README says so plainly rather than leaving the 6.5 dB gap standing as a like-for-like
result. (Nothing here says TRIPS would win a fair comparison -- only that this one is not it.)

`config_full3_alt.yaml` runs the other protocol: `forced_heldout_mode: alternate` trains on
`IMG_3829/3831/3833` and holds out `IMG_3828/3830/3832`, i.e. interpolation inside an *observed*
shade region -- the protocol the Gaussian baseline is implicitly measured under. Queued at prio 70
with `--max-minutes 240`. Two caveats to record before its numbers land:

1. **Confound:** it also inherits the missing-EXIF fix, so it differs from `full2-trips` in two
   ways, not one. The clean protocol comparison is full3-alt vs a `--calibrate` re-eval of
   full2-trips.
2. **The held-out sets are different sizes** (verified against the real name list): `all` gives
   186 train / 33 held out, `alternate` gives 189 train / 30 held out. So full3-alt's
   *all-held-out* PSNR is not directly comparable with full2's either -- compare the shade group
   (3 frames vs 6) and say which protocol produced it.
