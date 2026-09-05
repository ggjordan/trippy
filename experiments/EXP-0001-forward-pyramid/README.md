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

## Planned commands

```bash
# Validation test: numpy vs Metal agreement
python -m trippy test_blend_fwd

# Smoke test: load one frame
python -m trippy render --scene kk-coherent --frames IMG_3828 --outdir output/runs/EXP-0001/ --device mps

# FPS measurement: render 3 frames, 10 passes each
bash scripts/gpu_submit.sh --prio 15 forward-pyramid -- \
  python -m trippy render --scene kk-coherent \
    --frames IMG_3828 IMG_3830 IMG_3832 \
    --repeat 10 \
    --width 1008 \
    --outdir output/runs/EXP-0001-forward-pyramid_1/ \
    --device mps
```

## Gate

**v0.1.0 acceptance**: CPU tests pass; Metal forward <1e-5 error vs. numpy on 32×32; three 1008-wide frames render in <300 ms total (≥10 fps). Contact sheet delivered: [original photo | numpy reference | Metal output | 5 pyramid levels separately].

## Verdict

(Filled in after run.)
