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
