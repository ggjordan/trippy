# EXP-0009: Hybrid design A — Splats **combined with** TRIPS (joint, end to end)

## Question

Jordan, 2026-09-06: *"his main interest is Splats combined with TRIPS (hybrid)"* (STATE.md,
review queue). Designs C and A1 in `docs/PLAN-2026-09-05.md` each pick a side — C throws the
point cloud away and refines the Gaussian render; A1 throws the Gaussian render away and
turns Gaussians into TRIPS points. **Design A is the one that keeps both.**

The Gaussian splat render of the scene (rgb + alpha + depth, from Splats' `gsrender.py`
against `kkc_15000.ply`) is concatenated onto every level of the TRIPS point pyramid before
the U-Net sees it, and the whole thing — points, sizes, features, poses, tone mapper, network
— is trained end to end against the photos. The network is therefore free to *keep the
Gaussians where they are good and use the TRIPS points where they fail*, and nothing forces
it to choose globally.

Concretely the question is:

> Does a network that can see both a Gaussian render and a TRIPS point pyramid beat **both**
> of them alone on held-out PSNR — and, the number that actually matters here, does it beat
> them in the **shade** region specifically?

## The two baselines to beat

| Baseline | Held-out PSNR, all | Held-out PSNR, shade (6 × `SHADE_FRAMES_KK`) | Source |
|---|---|---|---|
| Plain Gaussians (raw `kkc_15000.ply` render vs photo) | **15.53 dB** | **14.94 dB** | EXP-0005 README "Verdict" table |
| Plain TRIPS (`full1-broadcast`, 40 epochs, still rising) | **14.42 dB** | not split out | STATE.md / EXP-0003 |
| Design C (U-Net refining the Gaussian render only) | 15.54 dB | 12.97 dB | EXP-0005 |

Both numbers are recorded as named constants in `trippy/constants.py`
(`HYBRID_A_BASELINE_GAUSSIAN_PSNR_ALL/_SHADE`, `HYBRID_A_BASELINE_TRIPS_PSNR_ALL`) so a later
report can cite them without re-deriving them.

Read the design-C row as the warning: a learned renderer on top of Gaussians *gained*
0.45 dB on non-shade and *lost* 1.96 dB on shade. Design A's whole bet is that adding the
TRIPS point branch back — plus `dropout_gaussian_p`, which forbids the network from becoming a
thin residual on the Gaussians everywhere — is what recovers the shade region.

**Honest caveat on the plain-TRIPS baseline.** EXP-0003's 300-epoch `full2-trips` run (the
config this experiment copies verbatim outside its `hybrid:` block) is still in the prio-70
queue. 14.42 dB is the 40-epoch `full1-broadcast` number and will be beaten by `full2-trips`
alone. The comparison that settles this experiment is EXP-0009 vs EXP-0003 `full2-trips`,
same epochs, same point source, same everything-but-the-hybrid-block — fill it in when both
land.

## Design as built

- **Trainer option** `hybrid:` in `TrainConfig` (`trippy/hybrid/config_a.py`,
  `trippy/hybrid/gaussian_input.py`). `enabled: false` is the default and a hard no-op.
- **Channels.** Each U-Net input level becomes `[TRIPS features (C=4) | gaussian rgb (3) |
  alpha (1) | normalised depth (1)]` = 9 channels. `NetworkConfig.num_input_channels` follows
  `TrainConfig.net_input_channels`; the point cloud, background and rasteriser stay at 4.
- **`mode: all_levels`** (the default) area-averages the Gaussian block down to each pyramid
  level's own `(h, w)`. This is the default because TRIPS's `CombineBridge` re-concatenates
  each level's *raw* input twice per level, so a level-0-only signal is structurally
  unavailable to the coarse blocks that decide large-scale structure — exactly where
  "trust the Gaussians here / don't there" has to be decided. `mode: concat_level0` puts the
  real block on level 0 only and zeros on the coarser levels.
- **Crops.** The render is cropped by handing `trippy.scene.dataset.crop` the *same*
  `(size, zoom, center)` and the *same* `K` the photo crop used, so the K-adjust and the
  overshoot mask are identical by construction rather than by parallel implementation
  (`tests/test_hybrid_a_crop.py` proves it against the photo path itself, and against an
  independent hand-written gather). Depth is a metric camera-space z: a crop/zoom changes the
  intrinsics, not the distance to the surface, so depth is resampled and never rescaled by
  the crop. Its only rescale is the scene-global `depth_scale`.
- **Depth normalisation.** `depth_scale: null` measures the scene's median camera-to-Gaussian
  depth (median over `alpha >= 0.5` pixels of 12 evenly-spaced rendered frames) at Trainer
  construction and writes the number back into the config, so the checkpoint records the exact
  normaliser its weights were trained with. The depth channel reaches the network unitless.
- **Held-out eval** reads renders from disk by image name — renders exist for all 219
  registered images, held-out included, so no live rendering happens during eval.
- **Arbitrary poses** (candidate report / dolly / off-path) have no precomputed render.
  `trippy/hybrid/gsrender_live.py` renders `hybrid.ply_path` live through Splats' `gsrender`
  (imported by path, never copied), caching the 1.7 GB PLY once per process.
  `trippy.train.eval.build_trainer_from_checkpoint` installs that provider lazily on any
  hybrid checkpoint; `render_candidate(..., gaussian_provider=...)` overrides it (tests inject
  a fake, so the CPU suite never touches MPS or the PLY). A dolly/off-path pose never borrows
  its *anchor* image's precomputed render -- that pose is displaced from the photographed one,
  so the cached render is from a different camera. It is rendered live, or it is zeros.
- **Ablations in config.** `hybrid.dropout_gaussian_p` (default 0.2) zeroes the Gaussian
  channels of that fraction of training crops; `hybrid.mask_by_alpha` (default true)
  multiplies the Gaussian rgb by its own alpha before concatenation.

### Deviations from the brief

1. **The EXP-0005 renders had to be re-created.** They were written inside the since-removed
   `.worktrees/hybrid-c/output/` and went with it. Re-rendered by
   `trippy-hybrid-a-render-1` / `-2` (prio 17) into the absolute path
   `/Users/nzbirdranch/trippy/output/hybrid-c/renders/w1008`, using
   `trippy.hybrid.render_splat_views` unchanged.
2. **`report.py` was not modified.** The brief allowed a "gaussian_provider hook only" there;
   it turned out none was needed — `build_trainer_from_checkpoint` installs the provider, so
   `run_train_report`'s two `render_candidate` calls get it for free. The two baselines are
   therefore recorded here and in `trippy/constants.py` rather than injected into
   `comparison_table_markdown` (which has no access to the run's config).
3. **Both configs use an ABSOLUTE `run_dir`.** The job runs from a git worktree, so a
   relative `run_dir` would write into the worktree's own `output/` -- exactly how EXP-0005's
   renders were lost -- and `scripts/deliver.sh` refuses any artifact that is not under
   `$TRIPPY_OUTPUT`. (`trippy train --run-dir <abs>` is the other way to do this.)
4. **The smoke runs at width 504 against the w1008 renders.** Rather than spend a second
   ~50-minute GPU render pass, `resample_to` area-averages the render onto the narrower grid
   (same camera, proportional K). This also exercises the resolution-mismatch path, which the
   long run does not.

## Commands

```bash
# 0. Re-create the Gaussian renders (two shards, ~25 min each, prio 17).
scripts/gpu_submit.sh --prio 17 hybrid-a-render-1 -- bash -c \
  'cd /Users/nzbirdranch/trippy/.worktrees/hybrid-a && PYTHONPATH=. \
   /Users/nzbirdranch/Splats/tools/ml-sharp/.venv/bin/python -m trippy.hybrid.render_splat_views \
   --scene /Users/nzbirdranch/Splats/scenes/karekare/kk-coherent \
   --ply /Users/nzbirdranch/Splats/output/Training-Data/karekare/kk-coherent/kkc_15000.ply \
   --out /Users/nzbirdranch/trippy/output/hybrid-c/renders/w1008 \
   --width 1008 --device mps --start-index 0 --end-index 110'
# ... and --start-index 110 --end-index 219 as hybrid-a-render-2.

# 1. Smoke (prio 16): proves the MPS training path AND the live-gsrender candidate report.
#    Submitted AFTER the render shards -- lower prio number runs first, so a prio-16 job
#    submitted earlier would jump ahead of the prio-17 renders it depends on.
export TRIPPY_OUTPUT=/Users/nzbirdranch/trippy/output   # both configs' run_dir is absolute anyway
scripts/gpu_submit.sh --prio 16 --wait hybrid-a-smoke -- \
  trippy train --config experiments/EXP-0009-hybrid-a/config_smoke.yaml --report --max-minutes 40

# 2. The run (prio 70, self-reporting, delivers on completion).
scripts/queue_training.sh experiments/EXP-0009-hybrid-a/config.yaml --max-minutes 330
```

## Gate

Not a stage gate (v0.2.0's binding gate has already passed). This is v0.3.0's own bar
(`docs/SPEC.md` Milestones): *"hybrid beats best plain Gaussian on shade audit AND extent
gate; LPIPS not worse"*, plus this experiment's own question above: held-out PSNR must beat
**both** 15.53 dB (plain Gaussians, all) and EXP-0003's plain-TRIPS number at equal epochs,
and the shade bucket must not regress the way design C's did.

Per `docs/SPEC.md` D10 the verdict is Jordan's eyes in the viewer; the numbers rank
candidates, they do not decide.

## Jobs

| Job | prio | rc | numbers |
|---|---|---|---|
| `trippy-hybrid-a-render-1` | 17 | 0 | 110/110 frames, 1329.7 s |
| `trippy-hybrid-a-render-2` | 17 | 0 | 109/109 frames, 1452.5 s |
| `trippy-hybrid-a-smoke` | 16 | 0 | see below |
| `trippy-hybrid-a-all-levels` | 70 | queued | the run |

### Smoke (`trippy-hybrid-a-smoke`, rc 0, 18.5 min including `--report`)

2 epochs / 48 crops, width 504, 200k points — a plumbing check, not a result.

- Hybrid active: `mode=all_levels channels=['rgb', 'alpha', 'depth'] (+5 net input channels)`,
  measured `depth_scale` 3.898 world units, renders found for **219/219** images.
- Held-out (n=33, incl. the 6 shade frames): PSNR **7.40 → 8.88 dB**, SSIM 0.098 → 0.162,
  LPIPS 0.860. All 48 train steps had the Gaussian block present (`dropout_gaussian_p: 0.0`
  in the smoke); loss 1.691 → 1.614.
- Candidate report: **48 dolly + 12 off-path frames rendered through live `gsrender` on MPS**,
  no `REPORT_FAILED.txt`, all three deliveries succeeded. `--report` failures are swallowed by
  `_run_train_report_safely`, so rc 0 alone would not prove the live path — the rendered frame
  counts and the missing `REPORT_FAILED.txt` do.
- Dolly mean coverage 0.031 (stop index 13 of 48) is the 200k-point subset showing through,
  not a hybrid effect.

## Verdict

*(pending — `trippy-hybrid-a-all-levels` is queued at prio 70 behind the existing training
queue; it self-reports and delivers on completion. See
`output/runs/EXP-0009-hybrid-a/hybrid-a-all-levels/README.md` and `report/report.json`.)*
