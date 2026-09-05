# EXP-0002 — horse parity: does trippy's forward render match TRIPS's own?

**Stage gate:** v0.1.0 — "forward renders match a reference".
**Date:** 2026-09-06 · **Branch:** `feat/adop-parity` · **GPU job:** `trippy-adop-parity-1` (prio 13, rc 0)
**Verdict: PASS.** Mean PSNR **22.27 dB** against the ground-truth photographs, versus **22.34 dB** for the
authors' own rendered images of the same three frames — a **0.07 dB** gap, far inside the 1.5 dB bar.
Against the authors' renders directly, **37.0 dB**.

## What was run

The published Zenodo (record 10687419, CC-BY) Tanks & Temples `tt_horse` scene and `checkpoint_horse`
`ep0600`, rendered entirely through trippy: `trippy.scene.adop_io` reads the scene,
`trippy.net.checkpoint` loads the 2,218,471-point cloud / neural texture / confidences / sizes / poses /
intrinsics / tone-mapper, `trippy.raster.pyramid` rasterises 8 pyramid layers on MPS,
`trippy.net.unet` (34/34 checkpoint tensors, 101,291 parameters) fuses them and
`trippy.net.camera_model` applies that frame's exposure, white balance, vignette and response LUT.

```
scripts/gpu_submit.sh --prio 13 --wait adop-parity-1 -- bash -c 'cd .worktrees/adop-parity && \
  PYTHONPATH=. .venv/bin/python -m trippy.cli parity \
    --scene third_party/zenodo/scenes/tnt_scenes/tt_horse \
    --checkpoint third_party/zenodo/tt_checkpoints/checkpoint_horse \
    --epoch ep0600 --indices 8,120,144 --render-scale 1 \
    --modes trips,broadcast,trilinear --device mps --out output/EXP-0002-horse-parity'
```

Frames 8, 120 and 144 are all in the checkpoint's own held-out test split
(`test_indices_tt_horse.txt` = 0, 8, ..., 144); none is a training view. Frame 144 replaces the brief's
`00200.jpg`, which does not exist — the scene has 151 images.

## Results

All metrics crop 16 px off every side. TRIPS blacks out exactly that border
(`train_mask_border = 16`) in the test images it writes, so an uncropped comparison scores the authors'
*own* render at 15.7 dB against its own ground truth. Cropped, it scores 25.2 dB. This one detail is
worth ~10 dB and is the single easiest way to publish a wrong parity number.

| frame | mode | PSNR vs GT | SSIM vs GT | LPIPS vs GT | authors' PSNR vs GT | PSNR vs authors | s |
|---|---|---:|---:|---:|---:|---:|---:|
| 00009.jpg (idx 8) | **trips** | **25.099** | 0.8072 | 0.1031 | 25.186 | 36.285 | 1.2 |
| | broadcast | 14.399 | 0.6502 | 0.4138 | | 14.576 | 0.3 |
| | trilinear | 23.821 | 0.7984 | 0.1390 | | 27.835 | 0.2 |
| 00121.jpg (idx 120) | **trips** | **21.979** | 0.8121 | 0.1319 | 22.043 | 37.751 | 0.3 |
| | broadcast | 16.630 | 0.7358 | 0.2720 | | 17.080 | 0.2 |
| | trilinear | 21.411 | 0.8062 | 0.1630 | | 28.278 | 0.1 |
| 00145.jpg (idx 144) | **trips** | **19.716** | 0.7812 | 0.1450 | 19.775 | 36.930 | 0.2 |
| | broadcast | 14.392 | 0.6700 | 0.3376 | | 14.999 | 0.2 |
| | trilinear | 19.191 | 0.7740 | 0.1824 | | 25.552 | 0.1 |

| mode | mean PSNR vs GT | mean SSIM | mean LPIPS | mean PSNR vs authors |
|---|---:|---:|---:|---:|
| **trips** (the path the checkpoint was trained with) | **22.265** | 0.8002 | 0.1266 | **36.989** |
| broadcast (all layers, factor 1) | 15.141 | 0.6853 | 0.3411 | 15.552 |
| trilinear (two straddling layers only) | 21.474 | 0.7929 | 0.1615 | 27.222 |
| *authors' own renders* | *22.335* | *0.8171* | *0.1382* | — |

trippy's LPIPS is **better** than the authors' own render (0.1266 vs 0.1382) while its SSIM is slightly
worse (0.8002 vs 0.8171) — consistent with a marginally different high-frequency noise floor from a
different blend order, not with a systematic error.

Timing: 0.2-1.2 s per 1920x1080 frame on MPS (first frame includes shader compilation), ~10.4M fragments
per frame in `trips` mode.

## What had to be fixed to get there

Three findings, in the order they mattered. Full source citations in `docs/TRIPS_REFERENCE.md`
(new sections 2a, 2b, 3a, 3b, 6a, 8a, 8b, 8c, 9b).

1. **The neural texture is used raw, not `abs()`-ed. Worth +16.6 dB.** `docs/TRIPS_REFERENCE.md` Sec. 2
   said `PrepareTexture` is called with `!non_subzero_texture`. It is called with `non_subzero_texture`
   itself (`Pipeline.cpp:257`), which is `false`, so no `abs()` is taken. Taking it tripled the composited
   feature magnitude, pushed the U-Net output past the response LUT's `[0, 1]` domain and produced a
   washed-out cyan image at **8.46 dB**. Removing it: **25.10 dB** on the same frame. Everything else —
   pose conversion, distortion, layer coordinates — was already correct at 8.46 dB; the render was
   geometrically perfect and photometrically wrong.
2. **`use_layer_point_size` is `true` for every published checkpoint.** It has no `SAIGA_PARAM`, which is
   why the reference doc called it unreachable, but `CombinedParams::Check` derives it:
   `use_layer_point_size = !fix_point_size` (`Settings.cpp:39`), and the horse `params.ini` sets
   `fix_point_size = false`. That selects a different forward kernel entirely (`RenderFast16` /
   `CountAndCollectTiled`, `PointRenderer.cu:730`) whose layer rule is "write layers
   `0 .. ceil(log2(size_px))`, weighted by `compute_point_size_fac`, which returns 1.0 below
   `floor(log2(size_px))`". That is trippy's new `trips` mode. The brief's `broadcast` reading of the
   reference doc costs **7.1 dB**; the `trilinear` reading costs **0.8 dB**.
3. **TRIPS's pixel centres are on integers, trippy's on half-integers** (`PointBlending.h:216-240` vs
   `docs/GEOMETRY.md`), and the pyramid halves with `ceil`, not integer division
   (`PointRenderer.cu:385-391` — the `h/=2` branch only runs for `network_version == "MultiScaleUnet2d"`).
   `parity.py` renders each layer with its own `num_layers=1` call at `K/2**l` with `cx, cy` shifted by
   +0.5, which reproduces TRIPS's `ip *= 0.5f` exactly. A single multi-layer call cannot: the two
   conventions differ by a layer-dependent offset.

Also required, and correct on the first attempt (so they cost nothing but are load-bearing): xyzw
camera-to-world -> wxyz world-to-camera pose conversion (verified against the checkpoint's own
`poses_se3` buffer to 1e-7), Saiga's 8-parameter distortion with its `k1 k2 k3 k4 k5 k6 p1 p2` ordering
(~24 px at the corner if skipped), `sigmoid(10 * confidence_raw)`, `softplus(point_size_raw)`, and the
fact that the horse "environment map" is 400,000 extra sphere points inside the cloud rather than a
texture — which is why `use_environment_map` is `false` and the learned background colour *is*
composited.

## Artifacts

- `output/EXP-0002-horse-parity/summary_sheet.png` — delivered via `scripts/deliver.sh` to
  `~/Splats/output/Jordan-Review/4-other/EXP-0002-horse-parity.png`.
- `contact_000NN.png` per frame: GT | authors' render | ours | abs-diff heatmap | **raw level-0
  composite** (the honesty panel: photographed points before the U-Net infers anything) — for each of the
  three modes.
- `metrics.json`, `README.md`, and per-mode `_ours` / `_absdiff_gt` / `_level0` PNGs.
- Job log: `output/logs/trippy-adop-parity-1.log`.

Nothing under `output/` or `third_party/` is committed.

## Honest reading of the number

22.27 dB is not a good render in absolute terms — the horse scene is hard and the checkpoint itself only
reaches 22.34 dB. The claim here is narrow and it is the claim the gate asks for: **trippy's forward path
reproduces TRIPS's forward path.** The raw level-0 panels show why the remaining error is where it is —
layer 0 covers only ~60% of pixels and the U-Net invents the rest, which is exactly the
photographed-vs-inferred boundary this project exists to keep visible.

## Open questions

- Does the same code hold on the other seven published scenes? Nothing scene-specific was hard-coded, but
  only `tt_horse` has been run.
- The remaining 37 dB gap to the authors' render is float32 blend-order noise as far as anything measured
  here can tell. Proving that would need a CUDA machine to dump their fragment lists; not worth it.
- `render_scale != 1` parity is unverified (see `docs/LIMITATIONS.md`).
