# Experiments: structure, metrics, and verdicts

## Experiment folder layout

Each experiment lives under `experiments/EXP-NNNN-<slug>/`:

```
experiments/
└── EXP-0001-forward-pyramid/
    ├── README.md          (question, point source, config, job name(s), gate, verdict)
    ├── config.toml        (training parameters, optional, cited in README)
    └── (no artifacts here; see "Run location" below)
```

The README contains:
- **Question**: What does this experiment answer? E.g., "Does our Metal forward equal the numpy reference and render 3 kk-coherent frames at 1008 wide in <100 ms?"
- **Point source**: Which source is being tested? (1 = Gaussians, 2 = monocular depth, 3 = union, etc.)
- **Config**: Reference to config.toml or inline parameters.
- **Planned commands**: shell commands to run the experiment (placeholder until executed).
- **Gate**: What passes this experiment? E.g., "v0.1.0 acceptance".
- **Verdict**: Outcome after run (filled in post-run).

## Naming convention

- **EXP-NNNN**: zero-padded 4-digit experiment counter.
- **slug**: hyphenated lowercase description, ≤40 chars. E.g., `forward-pyramid`, `gaussian-density-on-hunua`.

## Run location

Experiments do not write artifacts to `experiments/EXP-NNNN-*/`. Instead:

```
output/
├── runs/
│   └── <exp>/
│       ├── EXP-0001-forward-pyramid_1/
│       │   ├── config.yaml       (actual run parameters)
│       │   ├── log.txt           (stdout/stderr)
│       │   ├── checkpoints/
│       │   ├── export.ply        (final model as 3DGS PLY)
│       │   ├── shade_audit.txt   (metrics)
│       │   └── honesty_sheet.png (raw | network | coverage/provenance)
│       └── EXP-0001-forward-pyramid_2/   (if re-run)
├── Training-Data/
│   └── karekare/kk-coherent/
│       └── candidates/
│           ├── EXP-0001-gaussian-source/
│           ├── EXP-0001-depth-source/
│           └── EXP-0001-union-source/
└── deliver/
    └── (artifacts for Jordan review, via scripts/deliver.sh)
```

Run directories are **gitignored**. Only the `experiments/` README lives in version control.

## Ranking candidates: metrics and gates

### Shade audit

```bash
python ~/Splats/tools/depthprior_shade_audit.py \
  --scene ~/Splats/scenes/karekare/kk-coherent/sparse_txt \
  output/Training-Data/karekare/kk-coherent/candidates/EXP-NNNN-<source>/export.ply
```

Output: opacity mass in shade region (lower is better; 0 = shade is transparent).

The same statistic is also computed **in-process, from the live parameters**, at every
`Trainer.evaluate` (`trippy.train.prune.dark_mass_stats`), and written into the eval's
`metrics.json` / `metrics.jsonl` under a `points` key:

```json
"points": {
  "n_points": 5738000, "n_removed_total": 0,
  "shade_region": {"n_in_region": 1633974, "mass_in_region": 336873.5,
                   "dark_mass_lum0.25": 67068.8, "dark_n_lum0.25": 286884,
                   "dark_mass_fraction": 0.1991}
}
```

so a removal run's effect on Jordan's complaint is visible per epoch rather than only after
an export plus a Splats-tool run. It is a port, not an approximation: the region code copies
`depthprior_shade_audit.py`'s `build_region`/`in_region` field for field, and the
colour/opacity mapping is exact through `trippy.train.export` (the exporter writes
`logit(conf)` and `(clip(feat[:,:3],0,1) - 0.5)/SH_C0`; the audit inverts both). Verified on
`kkc_15000`: identical `n_in_region` (1,633,974), `mass_in_region` (336873.52631),
`dark_mass_lum0.25` (67068.80576) and all six views' `d`/`nobs`/`znear`/`zfar` as the tool's
own cached JSON, in under a second for 7.36M points. Turn it off per run with
`shade_prune.log_dark_mass: false`; it degrades to an `{"error": ...}` field (never an
exception) on any scene whose shade frames are not registered.

### Held-out PSNR and LPIPS

After exporting the model to PLY, render with:

```bash
python ~/Splats/tools/gsrender.py <export.ply> --outdir output/renders
```

Use a modulo-8 split (hold out every 8th training view as test). Report:
- **PSNR** on non-shade frames (frames outside the shade region).
- **LPIPS** on the same split.

`Trainer.evaluate`'s PSNR is `-10*log10(masked_mse)` where the masked MSE averages over every
*(channel, pixel)* element the mask keeps (`trippy.net.losses.mse_loss`). Dividing a 3-channel error sum
by a 1-channel mask sum instead costs exactly `10*log10(3) = 4.771 dB`; that bug shipped in the first
EXP-0003 smoke run (`docs/LIMITATIONS.md`).

**Sanity floor.** Before believing any held-out number, compare it with the PSNR of a *constant* image at
the target's own mean. Anything below that floor is not a bad render, it is a broken pipeline (black,
inverted, or scaled into the wrong range). `tests/test_train_regression.py` asserts the floor on the
synthetic scene so the CPU suite catches it in seconds.

Target: v0.2.0 acceptance requires PSNR within 1.5 dB of the best plain Gaussian on non-shade frames.

### Forced hold-out protocols: `forced_heldout_mode: all | alternate`

The six shade frames (`IMG_3828`-`IMG_3833`, `trippy.constants.SHADE_FRAMES_KK`) are consecutive, so
*how* they are held out decides which question the shade PSNR answers. `TrainConfig.forced_heldout_mode`
picks the protocol (`trippy.scene.splits.partition_forced`):

| mode | shade frames in training | the question it answers |
|---|---|---|
| `all` (default) | none | **Novel view of an unobserved region.** The network has never seen a photograph of the shade; whatever it draws there is invented. |
| `alternate` | `IMG_3829/3831/3833` (offset 0 holds out `IMG_3828/3830/3832`) | **Interpolation inside an observed region.** The shade is photographed three times; the score is for three unseen viewpoints between them. |

Neither is "the right one" -- they are different experiments, and a run should say which it used.
`all` is the harder, more honest test of the project's actual goal (walking into a region the camera
never stood in). `alternate` is the protocol that makes the Gaussian baseline comparable, because:

**The Gaussian baseline never held the shade out.** `kkc_15000.ply` was trained with Brush's
`--eval-split-every 10` (`~/Splats/research/kk-coherent.md:61-67`), which holds out every 10th name of the
219 registered images -- 22 views. Of the six shade frames only `IMG_3829.jpg` was held out; the other
five, including the dolly anchor `IMG_3830.jpg`, were **training views**. Of the 33 frames in trippy's own
held-out split, 27 were `kkc_15000` training views. So the "Gaussians 15.53 dB all / 14.94 dB shade"
baseline rows are largely *training-set reconstruction* numbers being compared against genuine held-out
numbers. In the shade specifically the comparison is 5-of-6 unfair to TRIPS, and any README or leaderboard
row quoting it must say so. (Recomputing the baseline on a fair split would mean retraining the Gaussians
with trippy's split -- not done; the numbers stand as measured, with this caveat attached.)

### Test-time camera calibration (`eval_calibrate_camera` / `trippy eval --calibrate`)

A held-out image's row of `NeuralCamera.exposures_values` is **never trained** -- only sampled training
frames get gradients -- so a held-out frame renders through whatever EXIF initialisation it happened to
get (`Trainer._initial_exposure`). Exposure is a property of the camera that took the photo, not of the
reconstruction, so a wrong one costs many dB that have nothing to do with geometry.

TRIPS has the same problem and ships two opt-in answers, both **off** in `configs/train_normalnet.ini:48-49`
and in the released horse checkpoint (`third_party/zenodo/tt_checkpoints/checkpoint_horse/params.ini:53-54`):

- `optimize_eval_camera` -- a per-epoch "EvalRefine" gradient pass over the **test** crops that steps the
  camera and pose optimisers with texture/network frozen (`src/apps/train.cpp:591-596, 693-697`;
  `NeuralScene.cpp:1473-1503`, which skips `texture_optimizer` but always steps
  `camera_adam_optimizer`, i.e. exposure, white balance, response and vignette).
- `interpolate_eval_settings` -- copy each test frame's exposure/WB from its two neighbouring train frames
  (`NeuralCamera.cpp:481-520`, called from `TestEpoch`, `train.cpp:1604-1611`). The two are mutually
  exclusive (`SAIGA_ASSERT` at `train.cpp:1606`).

trippy's `eval_calibrate_camera` is `optimize_eval_camera` cut down to the photometric scalars:
per held-out image, `Trainer.calibrate_frame` fits **only** that image's exposure (and, with
`eval_calibrate_white_balance` / `--calibrate-wb`, its red/blue white balance -- green stays pinned as in
training) with `eval_calibrate_steps` (200) Adam steps at `eval_calibrate_lr` (0.05) on masked L1 against
that image's own photo. The points, poses, U-Net, vignette and response LUT are frozen; the fitted scalar
lives in a local tensor and is never written back into the module or the checkpoint
(`NeuralCamera.forward_with`). Adam is warm-started at `-log2(best_global_gain)` because it moves an
exposure by only ~`lr` per step and a broken frame can start 5.87 EV away.

Measured once, on EXP-0003 full2-broadcast (job `eval-calib-1`, rc 0): held-out shade
**8.49 -> 15.32 dB** calibrated, all held-out 15.02 -> 17.66, other 16.47 -> 18.18; the closed-form
global-gain number (nothing fitted) is 14.59 dB in the shade. The Gaussian baseline it is compared with
is 14.94 dB. See `experiments/EXP-0003-kk-trips-train/README.md` for the per-frame table and the caveats.

Rules for reporting it:

- **Default off** (`eval_calibrate_camera: false`, the `--calibrate` side column below). Every
  training-time eval computes the strict protocol; `trippy eval --checkpoint ... --calibrate` turns this
  side column on for a diagnostic re-run, which reports both numbers side by side (`psnr_mean` and
  `psnr_mean_calibrated`, `shade` and `shade_calibrated`).
- A calibrated PSNR **uses the held-out photo** and is therefore not a novel-view number. It answers one
  question: how much of the gap is photometric. Quote it as "calibrated", never as the run's PSNR.
- The fit minimises L1, not MSE, so on a frame whose exposure was already right the calibrated PSNR can
  come out a few tenths of a dB *below* the strict one. That is expected, not a bug.
- **The Gaussian baseline cannot be calibrated**: raw Gaussians have no per-image exposure model at all.
  Its shade PSNR is what it is. The leaderboard's calibrated column is blank for it, on purpose.

### `interpolate_eval_settings` ported: `eval_exposure_mode` (default `neighbours`)

TRIPS's second opt-in answer above -- `interpolate_eval_settings` -- is now ported too
(`trippy.net.camera_model.interpolate_from_train_neighbours`, feat/eval-interp, 2026-09-06). It fixes the
same held-out-exposure artefact **without ever reading the held-out photo**, which is why it is the
default rather than an opt-in diagnostic: the wrong 8.49 dB EXP-0003 shade number (STATE.md's
2026-09-05 correction) was mostly this artefact, not a reconstruction failure.

**The exact TRIPS rule, and what trippy ports instead of it.** `NeuralCameraImpl::InterpolateFromNeighbors`
(`NeuralCamera.cpp:481-520`) does not do a single distance-weighted lerp between the nearest train frame on
each side. It runs **10 Jacobi passes** of `exp[i] = 0.5*(exp[i-1] + exp[i+1])` over every non-training
index `i` (`TestEpoch`'s `not_training_indices`, `train.cpp:1604-1611`), with `i-1`/`i+1` wrapping
circularly at the dataset's ends (`index_before = (i-1)>=0 ? i-1 : n-1`); training rows are never
themselves overwritten, so they act as fixed boundary values. Two non-training indices sitting next to
each other therefore feed off each other's *previous-pass* value across all 10 passes, not just their nearest
train neighbours in one step. This is the standard Jacobi iteration for the 1-D discrete Laplace equation on
a path graph with those boundaries, and its exact fixed point -- the equation the iteration is converging
toward -- is the **linear interpolation, by index distance, between the two nearest fixed (training) values
surrounding any run of non-training indices** (a linear function already equals the average of its own
neighbours everywhere, so it solves the difference equation exactly, boundaries included).

`interpolate_from_train_neighbours` computes that fixed point directly instead of iterating 10 times:
for an isolated run of held-out frames (kk-coherent's 6-consecutive shade frames, or any single held-out
frame) this is mathematically identical to what TRIPS's own 10 passes converge toward, and it is exact
regardless of how long that run is, whereas a fixed 10-pass budget only fully converges for short runs.
Given `is_train` (per-frame-index, True for a TRAINING frame, in capture order) and a held-out `index`, it
walks backward and forward circularly (same wrap-around as TRIPS) to the nearest training index each way,
then weights each by the *other* direction's distance (the nearer training frame dominates); when both
directions resolve to the same single training frame, that frame's value is used unweighted; when no
training frame exists anywhere (impossible via `Trainer`, whose `__init__` already requires a non-empty
training split, but exercised directly in `tests/test_train_eval_interp.py`) it falls back to
`values.mean(dim=0)`, a plain scene-mean.

**Config and CLI.** `TrainConfig.eval_exposure_mode` (`trippy.constants.EVAL_EXPOSURE_MODES`):

| mode | what a held-out frame's headline number uses |
|---|---|
| `own` | its own never-trained exposure/WB row, unmodified -- the only behaviour before this feature |
| `neighbours` (**default**) | `interpolate_from_train_neighbours`'s exposure/WB, per the rule above |
| `calibrate` | the `eval_calibrate_camera` per-image Adam fit above, promoted to be the headline number instead of only a side column |

`Trainer.evaluate(exposure_mode=...)` resolves per row: a name in `self.train_names` is **always** "own"
regardless of the requested mode (TRIPS's own `InterpolateFromNeighbors` is likewise only ever called on
`not_training_indices`). `trippy eval --checkpoint ... --exposure-mode {own,neighbours,calibrate}` threads
this through `trippy.train.eval.evaluate_checkpoint`; omitting the flag keeps the checkpoint's own
`cfg.eval_exposure_mode` (`"neighbours"` unless the run that produced it set something else).

**Where the numbers land.** `Trainer.evaluate`'s per-image dict gains `"exposure_mode"` (which mode
actually produced *that* row -- `"own"` for every training-set name) and
`"psnr_eval"`/`"ssim_eval"`/`"lpips_eval"` (the headline number under the resolved mode); the top level
gains `"exposure_mode"`, `"psnr_mean_eval"`, `"shade_eval"` and `"other_eval"`. These are **new, parallel
fields** -- the existing `"psnr"`/`"ssim"`/`"lpips"`/`"psnr_mean"`/`"shade"`/`"other"` fields are
deliberately left meaning exactly what they meant before this feature: each frame's own raw exposure,
unconditionally. `tests/test_train_eval_calibrate.py`'s `test_calibration_recovers_a_broken_exposure` and
`test_calibrated_psnr_barely_depends_on_the_broken_starting_exposure` (which deliberately break a held-out
frame's own exposure by 5.87 EV, the kk-coherent no-EXIF artefact) pin that meaning down explicitly: they
assert the strict `psnr` field still shows the break, **and** that the headline `psnr_eval` field does
NOT -- it is computed entirely from the frame's TRAINING neighbours and never reads that frame's own
(broken) row, so it stays bit-identical no matter how badly that row is broken.

`trippy eval`'s printout, `trippy train --report`'s summary line/README table/`report.json`, and
`trippy.render.leaderboard` are now all repointed at the `_eval` fields as the headline numbers (this
task's own follow-up wiring) -- "neighbours" is visibly the default everywhere, not just in `trippy eval`'s
own output. The strict, own-exposure numbers stay visible everywhere too: as a secondary row in the
`--report` comparison table and summary line, and as the compact "Strict own-exposure PSNR (all/shade)"
leaderboard column (see "Leaderboard" below) -- never confused with the headline, never silently dropped.
A report.json/metrics.jsonl row written before this wiring landed has no `"_eval"` fields at all; every
consumer falls back to the strict fields for its headline column in that case and marks the cell
`" (own)"` (or, in `--report`'s table, labels the row `"(own -- pre eval-fields run)"`) so an old run's
README/leaderboard entry still renders a real number, honestly labelled.

### Per-image exposure diagnostics

`Trainer.evaluate` now records, for every evaluated image and at no extra render cost, the numbers that
separate an exposure artefact from a structural failure:

| key | meaning |
|---|---|
| `exposure_ev` / `exposure_gain` | the per-image EV this render actually used, and `2**-EV`, the gain it applied |
| `pred_mean` / `target_mean` | mean brightness of the prediction and of the photo |
| `brightness_ratio` | `target_mean / pred_mean` -- a pure gain error shows up here directly |
| `gain_best` | the closed-form least-squares global gain, `sum(p*t)/sum(p*p)` (`trippy.train.trainer.best_global_gain`) |
| `psnr_gain` | PSNR after applying `gain_best` -- the best any single global brightness factor can do, no training involved |
| `exposure_mode` / `psnr_eval` / `ssim_eval` / `lpips_eval` | which `eval_exposure_mode` produced this row's headline number (see above), and that number -- `"own"` for every training-set row regardless of the requested mode |

`trippy eval` prints them as a markdown table. A frame where `psnr_gain` is many dB above `psnr` is being
scored on its exposure; a frame where they are equal has a structural problem.

### Extent gate

Prevents scene sprawl. Compute the bounding box of the trained point set:

```bash
python ~/Splats/tools/tmp/extent-audit/extent_gate.py output/Training-Data/karekare/kk-coherent/candidates/EXP-NNNN-<source>/export.ply
```

Output: radius p99, p99.9, max. These should not exceed the extent of the original sparse COLMAP points by >20%. Scene sprawl makes rendering unusable.

## Point sources

`trippy/points/` (D4) turns a scene into a `PointSet` (xyz, size0, rgb0, conf0, provenance):

- `GaussianPlySource`: trained 3DGS Gaussian centres from a binary PLY. `min_opacity` filters by `sigmoid(opacity)`; `size_mode="scale"` uses the trained `exp(log_scale)` extent, `size_mode="knn"` ignores it and uses local point spacing instead. Reads the ~7M-row author PLYs in ~1-2 s (structured-dtype `np.fromfile`, no plyfile, no per-row loop).
- `ColmapSparseSource`: the sparse triangulated points from `points3D.txt`, fixed `conf0=0.5`, size from kNN spacing.
- `UnionSource`: concatenates any sources; with `voxel` set, dedupes colliding points keeping the highest-`conf0` survivor per cell.
- `MonoDepthSource`: implemented (v0.2.0); see "Monocular depth points" below. `LidarSource`: not implemented yet ("later"); constructor documents planned inputs.

Inspect any source without training via the CLI:

```bash
trippy density --source gaussian --path <ply> --min-opacity 0.05 --size-mode scale --max-points 200000
trippy density --source colmap --path <sparse_txt_dir>
```

`density` prints `PointSet.summary()` (count, bbox, median nearest-neighbour distance on a subsample, provenance histogram) as both a human table and a `JSON:`-prefixed line, and can also write it to `--out summary.json`. `size_mode="knn"` on the full multi-million-point Gaussian PLY is expensive by design (kNN over the whole cloud) -- pass `--max-points` first when exploring interactively.

One-shot numbers on `kkc_15000.ply` (7.36M Gaussians) and its `sparse_txt` COLMAP model, `min_opacity=0.05`, `size_mode=scale`:

| Source | Count (post-filter) | bbox (world units) | median nn-distance |
|---|---|---|---|
| gaussian (full) | 5,736,619 | [-79.3,-85.6,-61.4] to [83.1,68.3,94.1] | 0.0795 |
| gaussian (`--max-points 200000`) | 200,000 | [-32.5,-83.8,-55.5] to [81.2,24.1,74.0] | 0.0803 |
| colmap | 153,515 | [-130.6,-117.2,-207.3] to [213.5,168.7,188.1] | 0.0865 |

The COLMAP sparse cloud's bbox is much larger than the Gaussians' -- expected, since sparse triangulated points include noisy far-field/sky points that training prunes away.

## Mandatory honesty sheet

Every candidate must include a three-panel image:

1. **Raw composite** (level-0 blend_fwd output, no U-Net).
2. **Network output** (after U-Net tone mapping).
3. **Coverage/provenance map**: colourised by coverage count and point source (Gaussians = red, depth = blue, union = purple). Pixels with coverage <0.3 are outlined in white.

The honesty sheet makes clear which image regions are photographed (covered by point sources) and which are inferred (U-Net hallucination). Jordan reviews these alongside the dolly video.

## Dolly camera paths

The dolly renders a fixed path through the shade region:

```bash
python ~/Splats/tools/depthprior_shade_dolly.py <export.ply> --outdir output/renders
```

**Pose source**: use the same pose as `IMG_3830.jpg` (the centre of the shade region), then translate the camera along the `+X` axis (horizontal, perpendicular to viewing direction) from `-0.35` m to `+1.20` m. This walks the observer through the shade volume and shows whether TRIPS renders it as a lighting effect or as a cloud of points.

Output: MP4 video (typically 2–5 seconds at 24 fps).

**Stop rule (the camera must not exit the geometry).** Splats' own `depthprior_shade_dolly.py`
uses `t` from `-0.35` to `+1.20` of local depth because it dollies through *Gaussians*, which
have volume everywhere along that range. TRIPS point sources do not: past the shade volume's
far surface there is nothing left to render. On EXP-0003's full1-broadcast candidate the raw
(level-0, no U-Net) centre coverage collapses across the path -- `0.46` at `t=-0.35`, `0.08` at
`t=+0.51`, `0.0001` at `t=+1.20` (`output/runs/EXP-0003-kk-trips-train/full1-broadcast/candidate/
report.json`) -- so a video that plays the whole default path drifts through several seconds of
visibly empty space at the end. `trippy.render.dolly.dolly_stop_index(coverage_center, threshold=
DOLLY_COVERAGE_STOP_THRESHOLD)` finds the last frame whose centre coverage is still at or above
`DOLLY_COVERAGE_STOP_THRESHOLD` (`0.05`); `trippy.render.candidate.render_candidate`'s
`stop_at_low_coverage=True` (on by default for the dolly render in both `trippy candidate-report`
and `trippy train --report`) truncates `dolly.mp4`/`dolly_raw.mp4` there, while still rendering
and recording per-frame metrics for every pose in the full path. `metrics.json` then carries
`dolly_stop_index`, `dolly_stop_threshold`, and `dolly_stopped_early`. Unit-tested against
synthetic coverage profiles (monotonic-decreasing, all-above, all-below, non-monotonic) in
`tests/test_render_report.py`, since the rule itself needs no rendering to test.

## Jordan's viewer verdict is final

All metrics are rankings. **Jordan's visual assessment in the viewer overrides any metric.** If PSNR is high but the shade looks wrong, the metric is wrong. If LPIPS is high but the scene looks good, fine. The only verdict that matters: can you step into the scene and see shading, not a cloud?

## Experiment tracking: research/trips-metal.md

Each completed experiment adds an entry to the running log `research/trips-metal.md`. Entries are appended chronologically (never rewritten) and record:

- **Date** (YYYY-MM-DD HH:MM).
- **Question**: What was tested?
- **Job name**: reference to `output/jobs/trippy-<name>.sh`.
- **Numbers**: PSNR, shade audit, extent radius, FPS (if applicable).
- **Verdict**: Pass/fail/inconclusive.
- **Artifact path**: where to find the export PLY, video, honesty sheet.

Example:

```
## 2026-09-06 10:30 — EXP-0001 forward pass validation
Question: does our Metal forward equal the numpy reference?
Job: trippy-forward-check
Numbers: agreement <1e-5, 3 frames at 1008 wide rendered in 87 ms (11.5 fps)
Verdict: PASS
Artifact: output/runs/EXP-0001-forward-pyramid_1/
```

This log serves as the experiment audit trail.

## Exporting TRIPS point sets as PLY

`trippy.train.export` (`write_gaussian_ply` / `export_pointset_ply`) writes
any `PointSet` (or raw `xyz`/`rgb`/`conf`/`size` arrays) as a
3DGS-compatible binary PLY, so Splats' existing audit tools (extent gate,
`ply_extract.py`, `depthprior_shade_audit.py`) and the Brush viewer can
open a TRIPS point set unchanged -- exactly the mapping documented in
`docs/GEOMETRY.md` "3DGS PLY export mapping":

```
f_dc_{0,1,2}  = (rgb - 0.5) / SH_C0
opacity       = logit(clamp(conf, 1e-4, 1 - 1e-4))
scale_{0,1,2} = log(size)                    (isotropic)
rot_{0,1,2,3} = (1, 0, 0, 0)                 (wxyz identity; TRIPS has no rotation)
nx, ny, nz    = 0
```

The writer is the exact mathematical inverse of `GaussianPlySource`'s read
side (`trippy/points/gaussian_ply.py`): a 200-point synthetic round trip
(`write_gaussian_ply` -> `GaussianPlySource`) reproduces `xyz`/`rgb0`/
`conf0`/`size0` to within float32 precision, and the point count matches
exactly (`tests/test_export_ply.py`). Higher-order SH is zero (`sh_degree`
defaults to 0, no `f_rest_*` properties); passing `sh_degree=3` also
writes 45 zero-filled `f_rest_0..44` properties for viewers that expect a
full SH basis. A per-point `provenance` array, if supplied, is written
alongside the `.ply` as a `<path>.provenance.npy` sidecar
(`write_provenance_sidecar`) for post hoc per-source diagnostics -- not
read by any 3DGS tool.

Verified against Splats' own (unmodified) `extent_gate.py` on a synthetic
200-point PLY:

```
$ ~/Splats/tools/ml-sharp/.venv/bin/python \
    ~/Splats/tools/tmp/extent-audit/extent_gate.py synthetic.ply

synthetic.ply  (200 gaussians)
  median centre           [ 0.1767 -0.3056  0.2197]
  radius p50/p99/p99.9/max  4.85 / 7.30 / 7.70 / 7.77
  scene diagonal (min/max box)  17.11
  non-finite means         0
  non-finite scales        0
```
exit code 0 -- the gate accepts the synthetic PLY with no changes to
Splats' code. `tests/test_export_ply.py::test_splats_extent_gate_accepts_synthetic_ply`
runs this same check via subprocess and skips cleanly on a machine
without the Splats `ml-sharp` venv.

## `trippy render` output layout

`trippy render` (`trippy/render/pyramid_render.py`, `render_frames`) is the
CLI entry point for the no-U-Net forward pass in this document's "Mandatory
honesty sheet" spirit, applied per pyramid level. Given `--out <dir>` and
`--frames a.jpg,b.jpg`, it writes:

```
<dir>/
├── a/
│   ├── photo.png        (undistorted source image)
│   ├── level_0.png .. level_{L-1}.png   (native per-level resolution)
│   ├── coverage.png     (1 - T_final at level 0, colorized)
│   ├── depth.png        (expected depth at level 0, colorized; uncovered
│   │                      pixels are exactly black, never a fabricated value)
│   └── sheet.png         (photo | L0 .. L{L-1} | coverage | depth, one row;
│                           levels are nearest-upsampled to level-0 size only
│                           for this sheet, so coarse blockiness stays visible)
├── b/ (same layout)
├── summary_sheet.png     (all frames, one row each: photo | L0 | coverage)
├── metrics.json          (per frame: image_hw, timing_ms {emit, sort, blend,
│                           total}, num_fragments, points_visible)
└── README.md             (the command that produced the run + the timing table)
```

`points_visible` is the count of points that survived the conservative
view-frustum cull (`trippy.raster.cull_points`) -- candidates handed to
fragment emission, not the (more expensive) count of points that actually
survived the per-pixel fragment cap/transmittance cutoff inside compositing.
## Monocular depth points

`trippy.points.monodepth.MonoDepthSource` is D4 point source 2: per-image
Apple DepthPro metric depth (via Splats' `tools/ldi/depth_batch.py`, run
only through `scripts/gpu_submit.sh`) -> median-ratio scale alignment
against reprojected COLMAP sparse depth -> unprojection to a world-frame
`PointSet`, voxel-deduped, `provenance=MONODEPTH`. `trippy.points.depth_io`
builds depth_batch.py's manifest and parses its `<id>_depth.npy` /
`_mask.npy` / `_meta.json` outputs; it also reimplements
`trippy.scene.dataset.SceneDataset`'s undistortion+cache step for an
arbitrary curated image subset (SceneDataset's own constructor only
supports a `limit`-first-N-sorted-images slice, which would force
undistorting most of a scene just to reach frames near the end of it).

Resolution/EXIF choice: DepthPro is always run on the same undistorted
pinhole image `SceneDataset`'s cache would produce (never the original
distorted capture), so DepthPro's pixel grid and the scale-alignment /
unprojection math share one `K` -- no separate "undistort these keypoints"
step is needed. EXIF orientation is left untouched, matching both
`SceneDataset` and `depth_batch.py`'s own documented convention for these
photo folders.

```bash
# Print the exact GPU job to run (writes the manifest + undistorted PNG
# inputs as a side effect; exits 3 while depth outputs are missing):
trippy depth-points --scene ~/Splats/scenes/karekare/kk-coherent \
  --images IMG_3828.jpg,IMG_3829.jpg,... --width 1008 \
  --depth-dir output/depth/kk-coherent --run-depth

# Then, after the printed scripts/gpu_submit.sh command has completed:
trippy depth-points --scene ~/Splats/scenes/karekare/kk-coherent \
  --images IMG_3828.jpg,IMG_3829.jpg,... --width 1008 \
  --depth-dir output/depth/kk-coherent --out output/points/kk-coherent-monodepth-12.npz
```

One-shot numbers on kk-coherent, 12 frames (6 shade + 6 spread across the
219-image sequence), `width=1008`, `stride=6`, `voxel=0.03` -- see
`research/trips-metal.md`'s 2026-09-05 13:20 entry and
`experiments/EXP-0004-monodepth-points/README.md` for the full breakdown:
234,712 points after dedupe (from 254,016 raw), median nn-distance 0.166.
Shade frames (IMG_3828-3833) average 1,914 usable sparse-COLMAP matches for
scale alignment vs 4,373 for the spread frames -- fewer keypoints in the
darker region, as expected -- but a *lower* mean MAD (0.188 vs 0.249),
i.e. the few matches shade frames get agree well with each other. An
8px-radius point-presence coverage check (projecting the full 12-frame
union into each shade camera) comes back ~100% for every shade frame both
over the whole image and a central 50% box -- this mostly reflects
DepthPro's `valid_fraction=1.0` and stride-6 density (a point lands near
almost every pixel by construction) rather than confirming the depth
values themselves are *correct* there; that needs the shade audit /
Jordan's viewer verdict once this source feeds a training run.

### Full-scene MonoDepth build and the Union point source (EXP-0006)

The 12-image sample above was extended to **all 219 registered kk-coherent images**
(same `width=1008`, `stride=6`, `voxel=0.03`, `conf0=0.35`, `scale_mode="median_ratio"`):
DepthPro job `depthpro-kk-coherent` (prio 11), rc=0, 219/219 images, `valid_fraction=1.0`
throughout, 293.6 s total (~1.34 s/image). The resulting `MonoDepthSource` PointSet is
**3,786,345** points (from 5,063,856 raw, 25.2% collapsed by its own internal voxel
dedupe -- a much higher collapse rate than the 12-image sample's 7.6%, expected since a
dense sequential walk has far more frame-to-frame overlap than 12 frames spread across
the whole scene), median nn-distance 0.2806. See `experiments/EXP-0006-union/README.md`
for the full per-image scale/MAD table and shade-frame coverage numbers.

`trippy.points.union.UnionSource` (point source 3, docs/SPEC.md D4) is reached from a
config file the same way any other source is -- `trippy.train.config.PointSourceConfig`
already implements `type: "union"` (nested `sources:` list + a `voxel`) and `type:
"npz"` (loads a `PointSet` written by `PointSet.save_npz` verbatim, e.g. a MonoDepth set
built once via `depth-points --out` and reused across multiple later builds/trainings
without recomputation). `trippy points-build --config <cfg.yaml> --out <path.npz>` is
the generic CLI entry point that builds *any* `PointSourceConfig`-described source
(gaussian/colmap/union/npz) and writes it to `.npz` + a summary JSON, exactly like
`density`/`depth-points --out` already do for their own single source -- this is how
EXP-0006's `Union(Gaussian, MonoDepth-219)` point set was built offline once (CPU-heavy:
`size_mode="knn"` on the full 5.74M-row Gaussian PLY runs a k-d-tree query per point, so
this ran via `scripts/cpu_heavy.sh`) and then loaded into training configs with
`point_source: {type: npz, path: <the union .npz>}` -- a sub-second load instead of
re-running the kNN + voxel dedupe at the start of every training job.

`trippy.render.sheets` (`contact_sheet`, `side_by_side`, `colorize`,
`save_png`) and `trippy.render.video` (`write_video`, `frames_from_dir`)
give every export an inspectable artifact: a labelled contact sheet
(PIL only, no matplotlib) and an ffmpeg-piped MP4 (`h264_videotoolbox`
hardware encoder when ffmpeg reports it available, `libx264` otherwise;
raises a clear `RuntimeError` if ffmpeg isn't on `PATH` rather than
failing silently). `colorize` maps depth/coverage scalars to RGB with a
hand-picked 5-stop viridis-like ramp implemented in numpy, so
`docs/SPEC.md`'s honesty sheet (raw composite | network output |
coverage/provenance map) needs no new plotting dependency.

## Training runs

`trippy train --config <cfg.yaml>` (`trippy/train/{config,params,trainer,eval,checkpoint_io}.py`) runs
the crop/render/tone-map/loss loop described in `docs/ARCHITECTURE.md`'s "train/" section. This section
covers the config file format and the on-disk run layout.

### Config file format

A config file is any subset of `TrainConfig`'s fields (see `trippy/train/config.py` and `trippy/
constants.py`'s "train/" section for every default and its `train_normalnet.ini` citation); everything
else keeps its scaled-for-trippy default:

```yaml
scene_root: /Users/nzbirdranch/Splats/scenes/karekare/kk-coherent
run_dir: output/runs/EXP-0003-kk-trips-train/EXP-0003-kk-trips-train_1
width: 1008
crop: 384
mode: trips        # trips | trilinear | broadcast, docs/GEOMETRY.md; "trips" is the default
layers: 5
epochs: 150
device: mps
forced_heldout:
  - IMG_3828.jpg   # ... SHADE_FRAMES_KK
point_source:
  type: gaussian
  path: /Users/nzbirdranch/Splats/output/Training-Data/karekare/kk-coherent/kkc_15000.ply
  min_opacity: 0.05
  size_mode: scale
```

`point_source.type` is `gaussian` (`GaussianPlySource`), `colmap` (`ColmapSparseSource`), `union`
(`UnionSource` of nested `sources:`), or `npz` (a `PointSet` previously dumped via `PointSet.save_npz`).
`forced_heldout` should be `trippy.constants.SHADE_FRAMES_KK` for any karekare scene, so every eval reports
a shade-region number rather than one that got lucky and landed in train.

Resume a run: `trippy train --config <cfg.yaml> --resume <run_dir>/checkpoints/checkpoint_latest.pt`.
Override the wall-clock budget from the CLI without editing the file: `--max-minutes 90`.
Override the output directory the same way: `--run-dir <abs path>` (useful when the job runs from a
git worktree but its artefacts should land in the main checkout's `output/`).

**How long is an epoch?** `steps_per_epoch = ceil(train_factor * n_train)` and each step is **one** crop
(`crops_per_step` is in `TrainConfig` but the trainer does not batch yet). With the default
`train_factor = 0.125` and kk-coherent's 186 training images that is 24 crops per epoch -- two orders of
magnitude fewer crops than a TRIPS epoch (`batch_size=4 x inner_batch_size=4` crops per step, one step
per image). Read "epoch" in a trippy run as "1/8 of a pass over the training images", and size smoke runs
accordingly: the EXP-0003 2-epoch smoke run does 48 optimiser steps in total.

### Point removal: `point_removal:` (TRIPS's rule) and `shade_prune:` (trippy's)

Both blocks default to `enabled: false`, a hard no-op: with neither set, a run's point count
is fixed from construction to export, exactly as before this existed.

```yaml
point_removal:            # TRIPS's own rule, ported
  enabled: true
  start_epoch: 100        # TRIPS: start_removing_points_epoch (Settings.h:403, default 200)
  every_epochs: 25        # TRIPS: point_removal_epoch_interval (Settings.h:406, default 50)
  mode: absolute          # "absolute" (TRIPS's rule) or "relative" (trippy's analogue, see below)
  conf_threshold: 0.1     # TRIPS: removal_confidence_cutoff (Settings.h:427 = 0.3; ini:134 = 0.5)
  rel_factor: 0.3         # only read in mode: relative -- see below
  min_points: 1000000     # trippy addition; TRIPS has no floor

shade_prune:              # trippy's audit-aligned heuristic -- NOT a TRIPS rule
  enabled: true
  log_dark_mass: true     # measurement only, on by default even when enabled is false
  frames: [IMG_3828.jpg, ...]   # defaults to SHADE_FRAMES_KK
  znear_frac: 0.05
  zfar_frac: 0.5
  lum_threshold: 0.25
  mode: absolute          # same mode/rel_factor choice as point_removal, independent field
  conf_threshold: 0.5
  rel_factor: 0.3
  start_epoch: 100
  every_epochs: 25
  min_points: 1000000
  scene_txt: ""           # "" = the run's own scene model (resolve_sparse_dir)
```

**TRIPS's rule, verbatim.** Drop every point whose *effective* confidence
`sigmoid(10 * raw_conf)` is below `conf_threshold`
(`third_party/TRIPS/src/apps/train.cpp:846-851`, with the confidence defined at
`src/lib/models/NeuralTexture.h:42` -- docs/TRIPS_REFERENCE.md Sec. 2), on epochs
`start_epoch + i*every_epochs` (`train.cpp:533-538`, `Settings.h:403-406`), evaluated once
per epoch before that epoch's training steps (`train.cpp:670-674`). No gradient, opacity-mass,
visibility or error term enters it. TRIPS's shipped configs disable it outright
(`configs/train_normalnet.ini:130-133`: first removal at epoch 2000 of a 600-epoch run), which
is why trippy defaults it off too.

**Threshold caveat, and it matters.** TRIPS initialises every confidence at
`sigmoid(10*0.5) = 0.9933`, so its 0.3/0.5 cutoffs mean "training pushed this point down".
trippy initialises confidence from the source PLY's opacity, so on `kkc_15000` (min_opacity
0.05) **74% of points are already below 0.3 and 90% below 0.5 at epoch 0** -- TRIPS's own
number would delete the scene on the first pass. Pick the cutoff against the source's opacity
distribution, not against TRIPS's ini. See experiments/EXP-0010-point-removal/README.md.

**`mode: relative` -- the faithful analogue, for trippy's own confidence init.** TRIPS's
absolute cutoff (above) is only "faithful" because TRIPS's confidence starts identical for
every point (`sigmoid(10*0.5) = 0.9933`), so "below cutoff" already means "training moved
this point". Under trippy's own init, an absolute cutoff mostly measures where a point
started. `mode: relative` fixes that by comparing each point's *current* confidence to its
OWN value at construction: it is dropped once `sigmoid(10*raw_conf) < rel_factor * init_conf`
(`trippy.train.params.PointParams.init_conf`, a buffer snapshotted once at `PointParams.
__init__` and carried, index-selected, through every later removal pass and checkpoint
round trip -- never touched by the optimiser). `conf_threshold` doubles as an optional
absolute floor in this mode (an independent OR trigger; set it to `0` to disable and use the
relative test alone) -- see `trippy.train.prune.confidence_drop_mask` and
`trippy.constants.POINT_REMOVAL_MODE_ABSOLUTE` for the exact rule and full rationale. Same
`mode`/`rel_factor` fields exist on `shade_prune`, independently, for its own confidence leg.
`config_removal_rel.yaml` in EXP-0010 is `config_removal.yaml` with only `mode: relative` and
`rel_factor: 0.3` changed, so the two runs isolate exactly this one choice.

**Point adding is NOT ported.** TRIPS's default adder shells out to an external NeAT CT
reconstruction binary; its in-tree grid-loss fallback (`NeuralScene.cpp:1330-1373`) is dead
code -- it scales the number of added points by `t_cell_value`, which no shipped renderer
kernel ever writes (`SetValueForCell`/`GetPointerForValueForCell`,
`NeuralPointCloudCuda.h:201-203`, have zero callers), so it always adds zero. The full finding
is in experiments/EXP-0010-point-removal/README.md; a trippy-native depth-driven adder is
parked there as a future experiment rather than culled.

**`shade_prune` is not TRIPS and is deliberately loaded.** It removes points that are inside
the shade audit's own region AND dark by its own Rec.709 test AND below a confidence cutoff --
i.e. it removes the thing the audit measures. Any run using it must report its dark-mass
fraction *next to* its held-out shade PSNR: if that PSNR drops, the removed points were
carrying real signal and the metric win is an artefact. It is a probe, not a fix.

**Mechanics.** Removing points rebuilds every per-point parameter *and* index-selects its Adam
moments onto the survivors, so moment row `i` still belongs to point `i`
(`Trainer._apply_keep_mask`) -- trippy's equivalent of `NeuralScene::RemovePoints` +
`ShrinkTextureOptimizer` + `MyAdam::shrinkInternalState`
(`NeuralScene.cpp:1375-1470,362-370`, `MyAdam.cu:346-374`). A checkpoint written after a
removal holds fewer points than the config's point source rebuilds, so `Trainer.load_state`
resizes before loading; resume, `trippy eval --checkpoint` and `--report` all work unchanged.
The extent penalty's bounding box is deliberately **not** recomputed after a prune (it is
anchored to the initial cloud).

### Output layout

Matches the "Run location" table above -- a training run writes everything under its own `run_dir`
(gitignored, never committed):

```
output/runs/<exp>/<run>/
├── log.txt                       (one line per checkpoint/eval/prune event -- see below)
├── metrics.jsonl                 (one JSON object per train_step AND per evaluate() call)
├── checkpoints/
│   ├── checkpoint_ep0000.pt      (kept: epoch 0 is always a checkpoint_keep_every multiple)
│   ├── checkpoint_ep0100.pt      (kept: checkpoint_keep_every multiple, default 100)
│   ├── checkpoint_latest.pt      (always the most recent, for --resume)
│   ├── checkpoint_best.pt        (the epoch with the best held-out PSNR so far)
│   └── best.json                 ({"epoch", "psnr"} for checkpoint_best.pt)
├── eval_ep0000/
│   ├── metrics.json               ({"epoch", "n_images", "psnr_mean", "ssim_mean", "lpips_mean", "names",
│   │                                "per_image": {name: {"psnr", "ssim", "lpips"}}, "shade", "other"})
│   └── sheet.jpg                  (honesty sheet: photo | render | raw L0 | coverage, up to
│                                    cfg.eval_max_images rows, default 6; JPEG q85, not PNG --
│                                    a quick progress check, unlike candidate-report's PNGs below)
├── eval_ep0010/
│   └── ...
├── eval_manual_<timestamp>/      (from a standalone `trippy eval --checkpoint`, see below)
│   └── ...
├── export.ply                    (final trained point cloud, 3DGS-compatible)
└── export.ply.provenance.npy     (per-point provenance sidecar)
```

**Checkpoint retention** (`trippy.train.retention`, added 2026-09-06 after disk hit 94%):
`checkpoint_ep<NNNN>.pt` files not covered by any of the rules above (not the best epoch, not a
`checkpoint_keep_every` multiple, not among the `checkpoint_keep_last` most recent epochs) are deleted
immediately after each `save_checkpoint` call -- so a 300-epoch run at `checkpoint_every=10` keeps a
handful of epoch files (~ep0000/ep0100/ep0200/ep0300 + whichever is best + the latest) instead of all
~30. `checkpoint_latest.pt`/`checkpoint_best.pt` are separate files, never subject to this deletion.
`trippy prune-run <run_dir> [--dry-run]` applies the identical policy to a run directory after the fact
(a run trained before this policy existed, or one still in progress): it reads `checkpoints/best.json`
for the best-epoch protection if present, never deletes `checkpoint_latest.pt` or the single newest
epoch file, and skips anything modified within `--protect-seconds` (default 120s) so it cannot race a
still-writing job.

`metrics.jsonl` is append-only and safe to `tail -f` during a run: per-step records have keys `step`,
`epoch`, `image`, `zoom`, `loss`, `image_loss`, `extent_penalty`, `camera_reg`, `nonfinite_grads`
(gradient entries zeroed before the optimizer step -- normally 0, see `docs/LIMITATIONS.md`); per-eval records have
`eval: true` plus the same fields as that eval's `metrics.json` (minus `names`, to keep each line short) --
including `per_image`, `shade`, and `other` (`{"n", "psnr", "ssim", "lpips"}` each): "shade" is
`cfg.forced_heldout` when non-empty, else `SHADE_FRAMES_KK` (the kk-coherent scene's shade region),
intersected with the images actually evaluated; "other" is every other evaluated image. This is the "held-out
shade" leaderboard column's own data source -- see "Leaderboard" below.
`log.txt` gets one human-readable line per checkpoint save, per eval, per pruned checkpoint
(`"pruned <path> (retention policy)"`), and per new best (`"new best held-out psnr=... -> checkpoint_best.pt"`),
plus a line when a `--max-minutes` budget cuts a run short.

### Standalone evaluation

`trippy eval --checkpoint <run_dir>/checkpoints/checkpoint_latest.pt [--images IMG_3830.jpg ...]`
(`trippy.train.eval.evaluate_checkpoint`) re-evaluates a checkpoint without re-training, rebuilding the
exact dataset/point-source/split the checkpoint was trained with from its own saved config. It writes to a
fresh `<run_dir>/eval_manual_<timestamp>/` directory (never collides with a mid-training/`--report` eval's own
`eval_ep<NNNN>/`) and appends an `{"eval": true, ...}` row to the run's own `metrics.jsonl` -- including the
`shade`/`other` split above -- so `trippy leaderboard` picks up a real "held-out shade" number for a
checkpoint that finished training before that split existed, with no retraining required. Omitting
`--images` evaluates the checkpoint's own held-out split. `--calibrate` (plus optional `--calibrate-wb`)
adds the test-time photometric calibration described above -- the strict numbers are still computed and
printed, with the calibrated ones next to them, and a per-image diagnostics table underneath.
`--exposure-mode {own,neighbours,calibrate}` selects `eval_exposure_mode` ("`interpolate_eval_settings`
ported: `eval_exposure_mode`" above) for that run; omitting it keeps the checkpoint's own config
(`"neighbours"` by default). `trippy.train.eval.render_offpath` renders
honesty triplets (no ground truth) at arbitrary poses from a JSON file; the dolly/off-path camera-path
generators below use the richer `trippy.render.candidate.render_candidate` pipeline instead (per-frame
raw/net/coverage/honesty PNGs, videos, a summary honesty sheet, and `metrics.json`), not this JSON-file API.

## Candidate report

`trippy candidate-report --checkpoint <ckpt.pt> --out <dir> [--dolly-pose IMG_3830.jpg] [--offpath
IMG_3828.jpg,...] [--device cpu|mps]` runs the full per-checkpoint evaluation pipeline docs/SPEC.md D10
requires -- export PLY, both Splats audits, the shade dolly video, and the off-path honesty sheet -- for
one checkpoint, in one command. It never opens an image (AGENTS.md privacy rule): every number below comes
from a metrics dict or a subprocess's own stdout/JSON, never from reading a rendered PNG.

**Shade dolly camera path** (`trippy.render.dolly.shade_dolly_poses`): the same construction as
`~/Splats/tools/depthprior_shade_dolly.py` -- `pose_name`'s own COLMAP orientation is frozen, and the
camera centre slides along that pose's forward ray from `t_range[0]` to `t_range[1]` times the scene's
local depth at that point (median distance to sparse points in front of the camera, 5th-95th percentile
trimmed). Reimplemented in numpy against `trippy.scene.colmap_io` / `trippy.geom.xform_a`, not imported
from `~/Splats`.

**Off-path honesty poses** (`trippy.render.offpath.offpath_poses`): the same construction as
`~/Splats/research/visual/render_offpath.py` -- for each requested image, a `lateral` pose (step sideways,
perpendicular to both the forward direction and the scene's up vector) and an `oblique` pose (rise above
the capture height and look down at the scene centroid), neither ever photographed. The scene's up vector
is the mean of every registered image's own world-frame up axis.

**Renderer** (`trippy.render.candidate.render_candidate`): loads the checkpoint via
`trippy.train.eval.build_trainer_from_checkpoint`, renders every pose through the pyramid + U-Net + tone
mapper (using that pose's own image's trained exposure/white-balance when its name matches a registered
image, else the mean across every trained frame), and writes, per pose, `raw_level0.png` (level-0
composite, no U-Net), `net.png` (network output), `coverage.png` (a from-scratch `1 - T_final` heatmap,
no photo pixels), and `honesty.png` (raw | net | coverage, with pixels below `coverage_threshold`
outlined in white on the *network* panel -- so a reviewer sees exactly which part of the pretty render is
inferred). When the poses form one camera path (the dolly case), also assembles `dolly.mp4` (network) and
`dolly_raw.mp4` (raw); either way writes a `honesty_sheet.png` contact sheet (up to
`CANDIDATE_HONESTY_MAX_SHEET_FRAMES` poses) and a `metrics.json` (per-frame mean coverage, full frame and
centre crop).

**Audit wrappers** (`trippy.eval.audits`): `run_shade_audit` and `run_extent_gate` run Splats' own
`depthprior_shade_audit.py` and `extent_gate.py`, unmodified, via subprocess through Splats' `ml-sharp`
venv interpreter, and parse their output into plain dicts -- the shade audit via its own `--json-out` flag
(a stable structured artifact), the extent gate via a regex parse of its stdout table (that script has no
JSON output). `audit_report` runs both and catches failures independently, so a missing Splats installation
or an audit that can't run against a particular scene never stops `candidate-report` from finishing; it
records `{"error": "..."}` for whichever audit failed instead.

`candidate-report` writes, under `--out`:

```
<out>/
├── export.ply                 (trainer.export_ply())
├── dolly/                     (render_candidate output for the shade dolly path)
│   ├── frames/<pose>/{raw_level0,net,coverage,honesty}.png
│   ├── dolly.mp4, dolly_raw.mp4, honesty_sheet.png, metrics.json
├── offpath/                   (render_candidate output for the off-path honesty poses,
│                                no video -- poses may not share one image size)
│   ├── frames/<pose>/{raw_level0,net,coverage,honesty}.png
│   └── honesty_sheet.png, metrics.json
├── report.json                (checkpoint, device, scene_root, export_ply, dolly, offpath, audits)
└── README.md                  (human summary: numbers + artifact paths only, no pixel content)
```

Tested on CPU with a synthetic scene and a randomly initialised (untrained) checkpoint
(`tests/test_render_dolly.py`, `tests/test_render_offpath.py`, `tests/test_eval_audits.py`,
`tests/test_cli_candidate_report.py`); the shade audit needs a real COLMAP text scene with observed
points to report anything but an error (the minimal synthetic scene used elsewhere in this repo's tests
has neither), so `tests/test_eval_audits.py`'s real-audit test runs against the (read-only) Karekare
`sparse_txt` geometry with a synthetic PLY placed inside its bounding box, skipping cleanly when
`~/Splats`/the ml-sharp venv aren't present. The Orchestrator runs `candidate-report` against real trained
checkpoints (GPU-trained, via `scripts/gpu_submit.sh`) once one exists.
honesty triplets (no ground truth) at arbitrary poses from a JSON file -- the stable API the dolly-camera-
path generator (see "Dolly camera paths" above) will plug into once it exists.

## Self-reporting training runs

`trippy train --config <cfg.yaml> --report` (`trippy.cli._cmd_train`, `trippy.render.report.
run_train_report`) makes a training run report on itself: no human or orchestrator step is
needed between "training finished" and "Jordan has something to open". After `Trainer.fit()`
returns, it:

1. Runs the same per-checkpoint pipeline `candidate-report` runs (dolly + off-path poses,
   `render_candidate`, both with the dolly's `stop_at_low_coverage=True`, see "Dolly camera
   paths" above) against the run's own final checkpoint and `export.ply` (both already written
   by `fit()` -- no re-export). The dolly/off-path pose names default to `cfg.forced_heldout`
   (e.g. `SHADE_FRAMES_KK` for a karekare config) or, if that's empty, the dataset's first
   registered image name, so this works on any scene without CLI flags.
2. Runs Splats' shade audit + extent gate (`trippy.eval.audits.audit_report`) on the candidate
   `export.ply`, and a **cached baseline audit** of the training run's own source PLY
   (`cfg.point_source.path`, only when `point_source.type == "gaussian"`) via
   `trippy.eval.audits.cached_baseline_audit` -- keyed on the PLY's path + mtime + size under
   `$TRIPPY_OUTPUT/audits/`, so repeated `--report` runs against the same unchanged source PLY
   (e.g. re-running the same experiment) don't re-pay Splats' full points3D.txt + multi-GB PLY
   audit cost.
3. Builds a baseline-vs-candidate comparison table (`trippy.render.report.
   comparison_table_markdown`) -- held-out PSNR/SSIM/LPIPS (candidate only; a baseline PLY has
   no held-out concept), shade dark-mass fraction (`dark_mass_lum0.25 / mass_in_region`) baseline
   vs candidate, extent radius p99/max baseline vs candidate, and dolly mean centre coverage over
   the kept (post-stop-rule) path -- and appends it, with a one-line summary
   (`trippy.render.report.summary_line`), to the run's own `<run_dir>/README.md` (created if it
   doesn't exist yet). Every cell that can't be computed (a failed/missing audit) reads `"n/a"`
   rather than being silently omitted or fabricated, so the table always renders even when an
   audit legitimately fails (e.g. the synthetic CPU test scene's empty `points3D.txt`).
4. Exports a **free-navigation viewer bundle** (`trippy.render.bundle.export_bundle`, the same
   code `trippy export-bundle` runs) from this same final checkpoint into `<run_dir>/bundle/`,
   and generates a Mac double-click launcher via `scripts/open_mac_viewer.sh`
   (`OPEN_TRIPS_MAC_<run_name>.command` under `$TRIPPY_OUTPUT/deliver/<run_name>/`). Jordan: "fixed
   dolly paths are hard to judge, I want to navigate freely" -- the dolly video is still delivered
   (it is cheap and still useful), but the viewer bundle is the artifact that lets him fly through
   the scene instead of judging it off one baked camera path. If the viewer binary
   (`rust/target/release/trips-viewer`) hasn't been built yet, `trippy.render.report.
   build_mac_viewer_launcher` never fails the run: it records the failure in `<run_dir>/report/
   VIEWER_LAUNCHER_FAILED.txt` and the rest of the report still completes (`report.json`'s
   `bundle.viewer` field carries the same status/note).
5. Delivers the viewer launcher, `dolly.mp4`, `honesty_sheet.png`, and `export.ply` via
   `scripts/deliver.sh`, **launcher first** (requirement: Jordan wants free navigation front and
   centre, not buried under the fixed-path dolly video). Every delivery shares the same honest
   one-line summary as the "why" -- e.g. `"trippy train report full1-broadcast: epoch 39, held-out
   PSNR 14.42 dB, shade dark-mass 36.2% vs baseline 19.9%"` -- and the viewer launcher's delivery
   appends `"; open in the free-navigation viewer; N/P step capture views"`
   (`trippy.render.report.viewer_delivery_why`). No verdict language ("looks good", etc.) -- just
   the numbers, per AGENTS.md's honesty rule and "Jordan's viewer verdict is final" above.
   `TRIPPY_DELIVER_DRY_RUN=1` skips the `deliver.sh` subprocess entirely (recorded as
   `"skipped: TRIPPY_DELIVER_DRY_RUN=1"` in `report.json`'s `deliveries` list, and prints the
   `deliver.sh` command that would have run) -- set by the CPU test suite so it never touches
   Splats' review queue or `research/trips-metal.md`.

Output layout: `<run_dir>/report/` mirrors `candidate-report`'s own `<out>/` layout (`export.ply`
is the run's own, not re-exported; `dolly/`, `offpath/`, `report.json`), `<run_dir>/bundle/` holds
the free-navigation viewer bundle (`bundle.json`, `points.npz`, `weights.safetensors`), and the
comparison table plus a "### Deliveries" list (viewer launcher first) are appended to
`<run_dir>/README.md`. `report.json` additionally carries `epoch`, `held_out`, `bundle`
(`{"bundle_dir", "viewer"}`), `summary_line`, and `deliveries` (a 4-entry list, launcher first).

**Never crashes the run** (requirement, not just a hope): `trippy.cli._run_train_report_safely`
wraps the whole report step in a `try`/`except`. If reporting itself throws -- a broken audit
tool, a missing scene sparse dir, a bundle export failure, `deliver.sh` refusing an artifact --
the exception is logged to stderr and written to `<run_dir>/REPORT_FAILED.txt`; `trippy train`'s
exit code still reflects `fit()` alone (0 on a successful training run, regardless of `--report`'s
outcome). A missing/stale viewer *binary* specifically never reaches that path -- it is caught one
level down, inside `build_mac_viewer_launcher`, per point 4 above.

Tested end to end on CPU with the synthetic scene (`tests/test_cli_train_report.py`): `--report`
writes `report.json`, the exported `bundle/` directory, and the README table (with its deliveries
list, viewer launcher first) even though the synthetic scene's empty `points3D.txt` makes the
shade audit degrade to `{"error": ...}` for both candidate and baseline (requirement 6); a
separate unit test drives `_run_train_report_safely` with a monkeypatched `run_train_report` that
raises, asserting `REPORT_FAILED.txt` is written and nothing propagates. The pure comparison-table/
summary-line/dolly-stop-index/viewer-launcher functions are unit-tested directly in
`tests/test_render_report.py` against synthetic dicts, coverage profiles, and a fake/missing
viewer binary, so those invariants don't depend on a full training run.

### `trippy bundle-launcher`: the same viewer bundle + launcher, for any checkpoint

`trippy bundle-launcher --checkpoint <ckpt> --name <name> [--scene <adop scene>] [--epoch <ep>]
[--out <dir>]` (`trippy.cli._cmd_bundle_launcher`, `trippy.render.report.
export_bundle_and_viewer_launcher`) runs steps 4-5 above -- bundle export, Mac launcher, delivery
-- against **any** checkpoint, not just a run's own just-finished final one: an older run that
predates this feature, a checkpoint someone wants a fresh launcher for without re-training, or a
TRIPS/ADOP checkpoint (`--scene` required, same auto-detection as `export-bundle`). `--out`
defaults to `<run_dir>/bundle` when the checkpoint is `<run_dir>/checkpoints/checkpoint_*.pt`, else
a `bundle/` alongside the checkpoint itself (`trippy.render.report.default_bundle_out_dir`).
Exit code is always 0 with the bundle written even if the viewer-launcher step fails (same
never-fails contract as `--report`'s own bundle step) -- CPU-only, same as `export-bundle`.

Tested end to end on CPU with a synthetic trippy-native checkpoint
(`tests/test_cli_bundle_launcher.py`, `--out` explicit and defaulted); run for real on a trained
checkpoint the same way `train --report` runs it internally.

### `scripts/queue_training.sh`: submit a self-reporting run in one command

`scripts/queue_training.sh <config.yaml> [--max-minutes M] [--dry-run]` is the one-liner Jordan
runs instead of hand-assembling a `scripts/gpu_submit.sh --train ... -- trippy train --config ...
--report` call. It validates `<config.yaml>` exists and parses as YAML with a top-level `run_dir:`
key (exit 2 otherwise), names the queue job after that `run_dir`'s own basename (so
`output/jobs/trippy-<name>.sh` and `output/runs/.../<name>` line up without cross-referencing
anything), and always submits `trippy train --config <config.yaml> --report` at
`gpu_submit.sh --train` priority (70, behind Splats' own jobs at 60) -- `--report` is never
optional here, matching this section's "no orchestrator step needed" goal. `--dry-run` forwards
to `gpu_submit.sh` (prints/writes the job file only). Tested in `tests/test_queue_training_script.py`
(missing config, missing/invalid `run_dir:`, invalid job-name characters, `--dry-run` output,
`--max-minutes` forwarding) -- never calls the real GPU queue.

## Leaderboard

`trippy leaderboard [--out <dir>] [--deliver]` (`trippy.cli._cmd_leaderboard`, `trippy.render.
leaderboard`) scans every run directory under `$TRIPPY_OUTPUT/runs/**/` that has finished at
least one self-report -- `report/report.json` (`trippy train --report`) or `candidate/report.json`
(`trippy candidate-report`) -- plus its own `metrics.jsonl`, and writes ONE markdown + PNG
comparison table so Jordan can see every TRIPS run so far against the fixed baselines without
opening each run's own README. `trippy.render.report.run_train_report` calls this automatically
(`regenerate_and_deliver_safely`) at the end of every `--report` run, so the sheet never goes
stale; `trippy leaderboard` on its own is for rebuilding it by hand (e.g. after deleting a run).

Columns: run name (suffixed `(smoke)` when the run name contains "smoke" --
`trippy.render.leaderboard.is_smoke_run`), experiment (the run's parent directory name under
`runs/`), mode and point-source type (matched back to the `experiments/<experiment>/*.yaml` whose
own `run_dir` field names this run -- `trippy.render.leaderboard.match_run_config`; matching is on
just the last two path components, since `run_dir` is written either relative to the repo root or
as an absolute, machine-specific path -- see EXP-0009's configs), epochs reached/planned and total
training steps (from `metrics.jsonl`), **Held-out all (neighbour-exposure) PSNR/SSIM/LPIPS** and
**Held-out shade (neighbour-exposure) PSNR/SSIM/LPIPS** (the headline numbers, below), **Strict
own-exposure PSNR (all/shade)** (a compact secondary column, below), shade dark-mass fraction and
extent p99/max (`trippy.render.report.dark_mass_fraction`/`extent_p99_max`, reused directly -- the
same numbers `comparison_table_markdown` computes for a single run's own README), dolly mean-centre
coverage (`trippy.render.report.dolly_mean_center_coverage`), an approximate wall time
(`metrics.jsonl`'s own creation time to report.json's mtime -- no per-step timestamp is recorded, so
this is a filesystem-timestamp estimate, not instrumented), and the delivered Mac viewer launcher's
filename (`n/a` for `candidate-report` runs, which never export a viewer bundle).

**"Held-out all"/"Held-out shade" headline the neighbour-exposure number.** `Trainer.evaluate()`
records a per-image `shade`/`other` split (see "Standalone evaluation" above) under both the strict
own-exposure keys and the parallel `"_eval"`-suffixed headline keys (default
`exposure_mode="neighbours"`, "interpolate_eval_settings ported" above) --
`trippy.render.leaderboard.build_run_row` reads the headline split from the run's own `report.json`
(`held_out`/`heldout_split.shade_eval`) first, else the last `metrics.jsonl` eval row's own
`psnr_mean_eval`/`shade_eval` fields. **A row whose report.json/metrics.jsonl predate the "_eval"
fields (PR #32) falls back to the strict split for these two columns and marks the cell `" (own)"`**
-- there is nothing to fall back FROM in that case, so the headline number there simply IS the
strict, own-exposure number, honestly labelled rather than hidden. A row still reads `n/a` when
NEITHER split exists yet for a metric: either the run predates any held-out split at all, or the
scene has neither `cfg.forced_heldout` nor any `SHADE_FRAMES_KK` frame in its held-out set. Run
`trippy eval --checkpoint <run_dir>/checkpoints/checkpoint_latest.pt` against an existing checkpoint
to backfill it without retraining (it appends a fresh eval row `trippy leaderboard` then picks up).
The two fixed baseline rows below always carry a real shade-only PSNR/SSIM/LPIPS split from their
own source experiment's per-bucket eval -- both raw Gaussians and Design C have a fixed, already-
measured number with no per-image exposure model to fix, so their headline columns are permanently
`" (own)"` too (see the row descriptions below).

**"Strict own-exposure PSNR (all/shade)"** is a compact secondary column: the same, unmodified
own-exposure numbers the leaderboard used before this feature, `psnr_all/psnr_shade`, kept visible
next to the headline so the two numbers are never confused. It is not used for sorting or ranking --
only the headline "Held-out all" column's PSNR feeds `_sort_key` (falling back to the strict number
for a pre-"_eval" row, i.e. whatever "Held-out all" actually displays).

**Held-out shade PSNR (calibrated)** is an *optional* column: it appears only once some run has a
calibrated eval (`trippy eval --checkpoint ... --calibrate`, whose `shade_calibrated` split lands in
`report.json`'s `heldout_split` or in the last `metrics.jsonl` eval row). With no such run the table is
byte-identical to the one before this column existed (`trippy.render.leaderboard.leaderboard_headers` /
`row_cells`). It is a diagnostic, not a ranking number -- see "Test-time camera calibration" above -- and
the Gaussian baseline's cell is permanently `n/a` because raw Gaussians have no per-image exposure model
to calibrate. Do not sort on it.

**The Gaussian baseline row is not a like-for-like comparison in the shade.** `kkc_15000` was trained on
five of the six shade frames (see "Forced hold-out protocols" above); its 14.94 dB shade number is mostly
a training-set reconstruction score, while every trippy run's shade number is a genuine held-out score.

Two fixed baseline rows are always included (not scanned -- neither is a trippy-native training
run with its own `metrics.jsonl`), sourced verbatim from their own experiment READMEs:
- **Gaussians kkc_15000 (baseline PLY)**: the plain Gaussian point source rendered directly
  (Splats' `gsrender.py`, no TRIPS/U-Net) -- PSNR/SSIM/LPIPS from EXP-0005's own "Baseline (raw
  render vs photo)" row, dark-mass fraction (19.9%) and extent p99/max (52.2/133.4) from Splats'
  shade audit + extent gate on the same PLY (experiments/EXP-0003-kk-trips-train/README.md).
- **Design C: render->photo U-Net (EXP-0005)**: the same PLY's render refined by a trained U-Net
  (experiments/EXP-0005-hybrid-c/README.md's own "Refined" row, final eval epoch 1125). No point
  cloud/extent of its own to audit.

**Baseline rows are unaffected by the neighbour-exposure headline change.** Raw Gaussians
(`gsrender.py`) have no per-image exposure model at all -- there is no held-out-exposure artefact
for `interpolate_from_train_neighbours` to fix, so its "Held-out all"/"Held-out shade" cells are the
same as-measured numbers the leaderboard always showed, now suffixed `" (own)"` for honesty. Design
C does have a `NeuralCamera` tone mapper, but its row is a fixed number quoted verbatim from
EXP-0005's own README (predating this feature, and never re-evaluated), so it gets the same
`" (own)"` treatment rather than a fabricated re-computation.

Sorted by shade dark-mass fraction ascending (closer to/below the 19.9% Gaussian baseline first),
then held-out PSNR descending -- a row missing either number (a failed audit, or a baseline with
no held-out/points concept) sorts to the end of that axis rather than crashing the sort.

Output: `<out>/leaderboard.md` and `<out>/leaderboard.png` (PIL-rendered table, no matplotlib --
same "no new dependencies" rule as `trippy.render.sheets`); `--out` defaults to
`$TRIPPY_OUTPUT/leaderboard`. `--deliver` (or the automatic end-of-`--report` hook) hands the PNG
to `scripts/deliver.sh` under the fixed name **`trips-leaderboard`** -- `review_add.sh`'s
`ln -sfn` replaces the same symlink every time rather than accumulating one leaderboard per
training run, so Jordan always has exactly one up-to-date sheet to open (it does append one row to
Splats' review-queue README per delivery, which is expected/acceptable). `TRIPPY_DELIVER_DRY_RUN=1`
skips the `deliver.sh` subprocess exactly like every other delivery in this codebase. The
end-of-run hook never fails an otherwise-successful `--report` run: `regenerate_and_deliver_safely`
catches any exception from rebuilding the leaderboard (e.g. a corrupt run directory somewhere else
under `runs/`) and only prints/records it.

Tested on CPU against synthetic `runs/`/`experiments/` trees written directly under `tmp_path`
(`tests/test_render_leaderboard.py`: both report.json layouts, missing/malformed fields, the sort
order, the fixed baselines, markdown + PNG rendering, and the held-out shade column's own
report-first/eval-row-fallback/`n/a` precedence; `tests/test_cli_leaderboard.py`: the
`trippy leaderboard` subprocess end to end, `--deliver`'s dry-run safety, the `--out` default) --
never a real scene, checkpoint, or PLY.

## Hybrid design C: render->photo U-Net refinement

Design C (docs/PLAN-2026-09-05.md "Hybrid (v0.3): (C) render->photo U-Net refinement on
gsrender.py outputs first (cheap, validates net/losses)") is deliberately independent of
`trippy.train.trainer.Trainer`: there is no point cloud, no rasteriser, no pose refinement.
Input is a *fixed* Gaussian-splat render (rgb + depth + alpha, from Splats' `gsrender.py`),
target is the photo, and the only trainable state is the U-Net and the per-image `NeuralCamera`
tone mapper -- both reused unmodified from `trippy.net`.

### Rendering the splat views

`trippy.hybrid.render_splat_views` (`python -m trippy.hybrid.render_splat_views`, run only via
`scripts/gpu_submit.sh` since it calls Splats' MPS `gsrender.render`) renders every registered
image of a scene against a binary 3DGS PLY, at `SceneDataset`'s own undistorted `(H, W, K)` grid
so every render lines up pixel-for-pixel with its photo. `max_hw` always passes
`HYBRID_C_GSRENDER_MAX_HW` (400) -- gsrender's own kwarg default (32) corrupts near-camera
Gaussian footprints. Output layout, per frame (`stem = Path(name).stem`):

```
<out_dir>/
├── <stem>.png              (rgb, uint8, gsrender's composited RGB)
├── <stem>.depth.npy        (float16, alpha-weighted expected camera-space z; 0 where alpha~0)
├── <stem>.alpha.npy        (float16, accumulated opacity in [0, 1])
└── manifest_<start>_<end>.json   (per-shard: scene_root, ply_path, width, device, max_hw,
                                    min_opacity, num_requested/rendered/skipped, elapsed_s,
                                    per-frame timing)
```

Idempotent per frame (a frame whose three files already exist is skipped unless `--force`) and
shardable via `--start-index`/`--end-index`, so a long scene renders across multiple queue jobs
without re-deciding which frames go where.

### Pairing and pyramids

`trippy.hybrid.dataset_c` pairs a render triple with its photo by filename stem
(`paired_names`); a photo with no matching render is silently excluded, never an error (useful
mid-shard). `render_to_tensor` stacks `[r, g, b, alpha, (depth)]` into the U-Net's input
tensor -- `channels=4` (rgb+alpha, the default, same `NetworkConfig` TRIPS itself ships) or
`channels=5` (+ a coarsely-normalised depth channel, `HYBRID_C_DEPTH_NORM_SCALE`).
`build_pyramid` is plain repeated `avg_pool2d(kernel=2, stride=2)`, finest level first, matching
`trippy.net.unet`'s own convention (and its odd-size `combine_bridge` generalisation handles the
resulting floor-halved chain on full-frame eval, where the source image need not be divisible by
`2 ** (layers - 1)`; training crops are chosen divisible by that quantity so every level is exact).

### Config and training

`trippy hybrid-c train --config <cfg.yaml>` (`trippy.hybrid.config_c.HybridCConfig`,
`trippy.hybrid.train_c.HybridCTrainer`) runs the crop/pyramid/U-Net/tone-map/loss loop. Config
format mirrors `TrainConfig`'s "state only what differs from the default" YAML convention:

```yaml
scene_root: /Users/nzbirdranch/Splats/scenes/karekare/kk-coherent
renders_dir: output/hybrid-c/renders/w1008
run_dir: output/runs/EXP-0005-hybrid-c/EXP-0005-hybrid-c_1
width: 1008
crop: 384          # must be divisible by 2**(layers-1) -- HybridCConfig validates this
channels: 4        # rgb + alpha; 5 adds depth
layers: 5
loss_l1: 1.0
loss_ssim: 1.0
loss_lpips: 1.0    # L1 + SSIM + LPIPS(alex); no vgg term (unlike TrainConfig's TRIPS parity)
forced_heldout:
  - IMG_3828.jpg   # ... SHADE_FRAMES_KK
```

Loss mask is deliberately full-frame: `(render alpha > 0) | ones_like(...)` always evaluates to
all-ones (an OR with True is always True) -- documented in `train_step`'s own comment, not a
bug. The point is a network that can also repair the render's own holes toward the photo, not
just refine already-covered pixels.

`trippy hybrid-c train --config <cfg.yaml> --max-minutes 40` caps wall clock the same way
`trippy train` does; `--resume <checkpoint.pt>` continues a run. Output layout matches the
point-based trainer's (`checkpoints/`, `log.txt`, `metrics.jsonl`, `eval_ep<NNNN>/`) with one
addition: `evaluate()` reports **two** numbers per bucket, `baseline` (raw render vs photo, no
U-Net at all) and `refined` (U-Net + tone-mapper output vs photo), each split into `all`,
`shade` (`cfg.forced_heldout`), and `nonshade` -- so a shade-region verdict is never averaged
away by the easy frames:

```
eval_ep<NNNN>/
├── metrics.json          ({"epoch", "n_images",
│                            "baseline": {"all"|"shade"|"nonshade": {"n","psnr_mean",
│                                                                     "ssim_mean","lpips_mean"}},
│                            "refined": {...same shape...}, "names"})
├── sheet.png              (photo | render | refined | |diff|, up to HYBRID_C_EVAL_MAX_SHEET_IMAGES
│                            rows, shade frames first)
└── shade_frames/
    └── <stem>_refined.png  (standalone refined PNG per cfg.forced_heldout name -- the
                              delivery artifact for Jordan)
```

`trippy hybrid-c eval --checkpoint <run_dir>/checkpoints/checkpoint_latest.pt [--images ...]`
(`trippy.hybrid.train_c.evaluate_checkpoint`) re-evaluates a checkpoint without re-training,
mirroring `trippy eval`.

## Hybrid design A: the Gaussian render fed to the U-Net *alongside* the TRIPS pyramid

Design C (above) replaces the point pyramid with a Gaussian render. Design A1
(`docs/PLAN-2026-09-05.md`) replaces the Gaussian render with points. **Design A keeps both**:
the same `trippy.train.trainer.Trainer` that trains a TRIPS point cloud also receives the
scene's Gaussian-splat render (rgb + alpha + depth) as extra U-Net input channels, and the
whole thing -- points, sizes, features, poses, tone mapper, network -- trains end to end
against the photos. This is the hybrid Jordan asked for on 2026-09-06 ("Splats combined with
TRIPS"): the network can keep the Gaussians where they are good and use the TRIPS points
where they fail, without anyone having to choose globally. Experiment: EXP-0009.

### Config

`hybrid:` is a block inside a normal `trippy train` config (`trippy/hybrid/config_a.py`).
`enabled: false` is the default and a hard no-op -- a non-hybrid run's network width,
checkpoint contents and numerics are exactly what they were before design A existed.

```yaml
hybrid:
  enabled: true
  renders_dir: /Users/nzbirdranch/trippy/output/hybrid-c/renders/w1008
  channels: [rgb, alpha, depth]   # canonical order; any non-empty subset
  mode: all_levels                # all_levels (default) | concat_level0
  dropout_gaussian_p: 0.2         # ablation 1: fraction of crops with the block zeroed
  mask_by_alpha: true             # ablation 2: rgb *= alpha before concatenation
  depth_scale: null               # null = measure and record; else world units
  missing: zeros                  # zeros (train through a half-rendered set) | error
  ply_path: .../kkc_15000.ply     # for LIVE rendering at unphotographed poses
```

`renders_dir` is exactly design C's output layout (`<stem>.png` + `<stem>.depth.npy` +
`<stem>.alpha.npy`, written by `trippy.hybrid.render_splat_views`); the two designs share the
renderer and the on-disk contract. A render set produced at a different width is area-averaged
onto the run's own grid (`resample_to`) -- same camera, proportional intrinsics.

### Channels and the two modes

Each U-Net input level becomes `[TRIPS features (C) | gaussian rgb (3) | alpha (1) |
normalised depth (1)]`. `TrainConfig.net_input_channels` (= `feature_channels + G`) drives
`NetworkConfig.num_input_channels`; the point cloud, background and rasteriser stay at
`feature_channels`, so only the network gets wider. With the shipped defaults that is
4 + 5 = 9 input channels against `filters = 32`, and every block's channel arithmetic still
closes (`filters - 2C` / `filters - C`, see `trippy/net/unet.py`).

- `all_levels` (**default**) area-averages the Gaussian block down to every level's own
  `(h, w)`. It is the default because TRIPS's `CombineBridge` re-concatenates each level's
  *raw* input twice per level, so a level-0-only signal never reaches the coarse blocks that
  decide large-scale structure -- exactly where "trust the Gaussians here, not there" has to
  be decided.
- `concat_level0` puts the real block on level 0 and zeros in those channels on the coarser
  levels (the U-Net requires a uniform channel count across levels).

A missing render and a dropped-out crop are the same thing to the network: an all-zero block
of the right width. The level count and channel count never vary within a run.

The pooling itself is done **host-side**: `GaussianInputs` keeps every block on the CPU and
`attach` copies only the finished per-level tensor onto the network's device. The pool costs a
few milliseconds either way, and doing it on the CPU means design A needs no additional Metal
kernel -- which matters because queue jobs run with `PYTORCH_ENABLE_MPS_FALLBACK=0` (an
unimplemented op is a hard failure) and there is no way to test the MPS path outside the queue.

### Crops: identical to the photo's, by construction

`GaussianInputs.crop_frame` calls `trippy.scene.dataset.crop` with the **same**
`(size, zoom, center)` and the **same** `K` the photo crop used, on the render block laid out
channels-last. The K-adjust and the overshoot/validity mask are therefore identical because
they are literally the same code on the same arguments, not a parallel implementation.
`tests/test_hybrid_a_crop.py` asserts bit-identical `K` and mask against the photo path, and
pixel agreement against an independently hand-written gather, over five (size, zoom, centre)
cases including one that overshoots the frame.

Depth is a metric camera-space z: a crop or zoom changes the intrinsics, not the distance to
the surface, so depth values are resampled with the same gather and never rescaled by the
crop. Their only rescale is the scene-global `depth_scale` -- with `depth_scale: null` that is
the median camera-to-Gaussian depth over `alpha >= 0.5` pixels of 12 evenly-spaced rendered
frames, measured at `Trainer` construction and **written back into the config**, so the
checkpoint records the exact normaliser its weights were trained with and eval/report
normalise identically.

### Eval, and poses that were never photographed

Renders exist for every registered image (held-out included), so held-out eval reads them from
disk by image name and never renders anything live. The candidate report's dolly and off-path
cameras are a different matter: no photo, therefore no precomputed render.
`trippy/hybrid/gsrender_live.py` renders `hybrid.ply_path` on the fly through Splats'
`gsrender` (imported by path, never copied; MPS, therefore only ever inside a queue job) and
caches the 1.7 GB PLY once **per process**, so a report that calls `render_candidate` twice
pays for it once.

`trippy.train.eval.build_trainer_from_checkpoint` installs that provider lazily on any hybrid
checkpoint -- lazily meaning nothing is loaded and MPS is never touched unless an
unphotographed pose is actually rendered. `render_candidate(..., gaussian_provider=...)`
overrides it with any `(name, K, R, t, image_hw) -> (G, H, W) tensor | None` callback; the CPU
tests inject a fake so the suite never touches a PLY or MPS.

**The precomputed renders are deliberately unreachable from that callback.** A
`CameraPose.image_name` means "anchored to that image", and every dolly and off-path pose is
*displaced* from the photographed one, so that image's render belongs to a different camera;
substituting it would feed the network a Gaussian image from the wrong viewpoint. The cache is
used only where the pose genuinely is the image's own, i.e. `Trainer.evaluate`. When no live
renderer is configured the provider returns `None` -- an all-zero block, i.e. the TRIPS-only
state `dropout_gaussian_p` trained the network to survive. That is the honest failure mode,
not a fabricated render.

### Ablations

- `dropout_gaussian_p` (default 0.2): a fifth of training crops see zeros in the Gaussian
  channels. Without it the network is free to become a thin residual on top of the Gaussian
  render everywhere -- precisely the failure design C already measured (EXP-0005: +0.45 dB
  non-shade, **-1.96 dB shade**). It is recorded per step in `metrics.jsonl` as
  `gaussian_dropped` / `gaussian_present`.
- `mask_by_alpha` (default true): an uncovered pixel then reads as "nothing here" rather than
  as gsrender's composited background colour, which carries no scene information but does look
  like content to a conv net.

### Baselines any design-A run is judged against

| Baseline | Held-out PSNR, all | Held-out PSNR, shade | Source |
|---|---|---|---|
| Plain Gaussians (raw render vs photo) | 15.53 dB | 14.94 dB | EXP-0005 |
| Plain TRIPS (`full1-broadcast`, 40 ep) | 14.42 dB | n/a | EXP-0003 |
| Design C (U-Net on the render only) | 15.54 dB | 12.97 dB | EXP-0005 |

Recorded as `HYBRID_A_BASELINE_*` in `trippy/constants.py`. The plain-TRIPS row is the 40-epoch
number; the fair comparison is EXP-0009 against EXP-0003's 300-epoch `full2-trips`, whose
config EXP-0009 copies verbatim outside its `hybrid:` block.

## Distillation (design B)

docs/SPEC.md D2: "A plain splat that incorporates TRIPS learning (Design B) is a valid
fallback path, not the primary deliverable" -- and the Quest honesty note: "ship fallback:
distilled Gaussians via the existing `~/Splats/tools/publish/` path". `trippy distill`
(`trippy/distill/{cameras,colmap_writer,render_set,brush_runner,compare}.py`) turns any
trained TRIPS checkpoint into a plain 3DGS PLY that opens unchanged in every existing
viewer (Brush, Splats' `tools/publish/publish_splat.sh`, Quest), by rendering the TRIPS
network's own output into an ordinary photo set and training an ordinary Gaussian splat
model (Brush) on it.

### Why this works: distillation, not compression

The checkpoint's point cloud + U-Net + tone mapper together define a function from
(camera pose) to (image). Design B samples that function at a dense set of poses close to
the real capture path and hands the resulting (pose, image) pairs to an off-the-shelf 3DGS
trainer exactly as if they were photographs -- the trained Gaussians end up approximating
whatever the TRIPS network learned to render (including, if it worked, the shade-as-lighting
effect the whole project is chasing), while paying none of TRIPS's own runtime cost in the
viewer. This is strictly a fallback (D2): a distilled splat re-quantises TRIPS's continuous,
network-refined appearance back into per-point Gaussian primitives, so it can only be as
good as (never better than) the TRIPS checkpoint it was distilled from, and it re-introduces
whatever splat artefacts Gaussians have (Jordan's stated worry, docs/SPEC.md D6).

### Step 1: render the image set (`trippy.distill.render_set`, `trippy distill --stage render`)

For each registered training camera ("anchor") plus a small number of cameras interpolated
between *consecutive* anchor pairs ("near-path", `trippy.distill.cameras`), render the
checkpoint's network output (`trippy.render.candidate.render_candidate`'s own `net.png` --
after the U-Net and tone mapper, never the raw pre-U-Net composite) at the checkpoint's own
training width. Anchors use each image's real COLMAP pose (no pose-refinement delta, same
convention `trippy.render.dolly`/`trippy.render.offpath` already use for arbitrary poses);
interpolated poses slerp rotation and lerp the camera centre between two anchors, at
`k` intermediates per pair (`--interp-k`, default `DISTILL_DEFAULT_INTERP_K`).

**Honesty guard** (AGENTS.md's honesty rule extended to camera poses, not just pixels): a
consecutive pair is only bridged with interpolated poses if (a) both images share one
physical camera (`camera_id`) and (b) their centre-to-centre distance is at most
`--max-jump-multiplier` (default `DISTILL_MAX_JUMP_MULTIPLIER`, 4x) times the scene's own
median consecutive-pair distance. A pair failing either check is not "consecutive along one
continuous walk" (a lens change, a registration gap, two separate sweeps of the same scene)
and is skipped rather than interpolated through -- recorded in
`DistillCameraPlan.skipped_pairs` and `distill_report.json`'s `skipped_pairs`/
`n_skipped_pairs`. Every interpolated pose is a linear blend between two real, photographed
anchors, so by construction it can never be farther from the nearest anchor than the anchor-
to-anchor distance itself -- "no far off-path invention".

Output layout, under `--out`:

```
<out>/
├── trips_export.ply           (the checkpoint's own trained point cloud, Trainer.export_ply)
├── renders/                   (render_candidate's full per-pose tree: raw/net/coverage/
│                                honesty PNGs + metrics.json -- for inspection, never opened)
├── images/<name>.png          (one copy of each pose's net.png, flat layout Brush expects)
├── sparse_txt/{cameras,images,points3D}.txt   (COLMAP text model, trippy.distill.colmap_writer
│                                                on trippy.scene.colmap_io's writers -- points3D
│                                                is the TRIPS export's own point cloud, seeded
│                                                deterministically down to --max-init-points rows,
│                                                default DISTILL_DEFAULT_MAX_INIT_POINTS)
└── distill_report.json        (every count above, plus skipped_pairs, mean_coverage_full)
```

`images/` + `sparse_txt/` together are `dataset_dir` for step 2 -- exactly the COLMAP layout
Brush's own dataset loader (`rust/brush-trips/crates/brush-dataset/src/formats/colmap.rs`)
auto-detects: `cameras.txt`/`images.txt` for every view, `points3D.txt` as Brush's own
initial-splat point cloud (positions + colours; Brush has no `--init-ply` flag and does not
read a size/opacity/rotation column from points3D.txt -- it seeds means + SH-DC from there
and initialises everything else itself). "Init from the TRIPS export ply" is therefore this
points3D.txt, not a separate Brush CLI flag.

`--device mps` only runs inside a `scripts/gpu_submit.sh` job (same rule as `trippy train`/
`trippy render`); the render step is GPU work, prio 15 (short job, not a training job).

### Step 2: train Gaussians on the image set (`trippy.distill.brush_runner`, `--stage brush-cmd`)

Brush's trainer must never run outside the GPU queue (AGENTS.md), so this module only builds
the exact command line and a self-contained job script -- it never executes Brush or
`scripts/gpu_submit.sh` itself, the same "print the GPU command, don't run it" convention
`trippy depth-points --run-depth` already uses. `resolve_brush_binary` prefers the lean
headless `brush-cli` binary over the full `brush`/`brush-app` GUI binary (both share the same
`Cli`/`TrainStreamConfig` flags, `apps/brush-cli/src/lib.rs`); neither is built by
`scripts/build.sh`/`scripts/test.sh` (rust/README.md), so build first:

```bash
bash scripts/cpu_heavy.sh brush-cli-build -- bash -c \
  'cd rust/brush-trips && cargo build --release -p brush-cli'
```

`brush_train_command` builds the argv: `<binary> <dataset_dir> --total-train-iters
<--brush-iters, default DISTILL_DEFAULT_BRUSH_ITERS=6000> --sh-degree 0 --max-resolution
<the render width> --eval-split-every 8 --eval-every 1000 --export-every <total-train-iters>
--export-path <out>/brush_out/ --export-name distilled_{iter}.ply --seed 0`. `--sh-degree 0`
(view-independent colour only): a single-checkpoint distillation gives Brush no multi-view
specular signal to recover with higher SH orders, so degree 0 is the honest choice, not
merely the cheap one. `--export-every` always equals `--total-train-iters` -- this pipeline
wants the final distilled PLY, not intermediate checkpoints. `write_brush_job_script` writes
that argv as `<out>/brush_train_job.sh` (mirrors `scripts/gpu_submit.sh`'s own generated job-
file shape: `set -eu` then one `exec` line); `brush_gpu_submit_command` prints the
copy-pasteable `scripts/gpu_submit.sh --train <name> -- bash <script>` line -- training is a
prio-70 job, behind Splats' own jobs and any other trippy trainings already queued (D9); the
task's own iteration budget ("5k-8k steps") is sized for a queue that already has several
long trainings ahead of it.

### Step 3: audit and compare (`trippy.distill.compare`, `--stage compare`)

Runs Splats' shade audit + extent gate (`trippy.eval.audits.audit_report`, unmodified,
subprocess) on up to three PLYs -- the training run's own baseline source PLY (`--baseline-
ply`, e.g. `kkc_15000.ply`), the checkpoint's own TRIPS export (`trips_export.ply` from step
1), and the Brush-trained distilled PLY (`--distilled-ply`, once step 2's queued job has
finished) -- and prints a markdown table (point count, shade dark-mass fraction, extent
p99/max) with one column per PLY given. A column whose PLY was not given (most commonly
`distilled`, before that training job returns) renders "pending", never a fabricated number
(AGENTS.md's honesty rule); a PLY that was given but whose audit itself failed (e.g. no
`~/Splats` installation on this machine) renders "n/a" per cell. `trippy distill --stage all`
(the default) runs render then compare in one command, using render's own `trips_export.ply`
and `scene_root` automatically -- `compare` alone (after an earlier `render`) reuses the same
`distill_report.json` rather than requiring every flag again.

### `trippy distill` end to end

```bash
trippy distill --checkpoint <run_dir>/checkpoints/checkpoint_latest.pt --out <out_dir> \
  --stage all --device mps --interp-k 2 --baseline-ply <path/to/kkc_15000.ply>
```

runs steps 1 and 3 in one command (step 2's actual training always needs the separate queued
`scripts/gpu_submit.sh --train` call `--stage brush-cmd` prints); `--stage render`,
`--stage brush-cmd`, `--stage compare` also run individually against the same `--out`
directory. See `experiments/EXP-0008-distill/README.md` for the worked pipeline-proof run
against a weak (40-epoch, 14.4 dB) EXP-0003 checkpoint, including the audit comparison table
and an honest read of the numbers.
