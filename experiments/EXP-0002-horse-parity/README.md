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

---

## Follow-up (feat/trips-mode, 2026-09-06): the `trips` rule moved into the rasteriser

**GPU jobs:** `trippy-trips-mode-gpu-1` (prio 12, **rc 0**), `trippy-trips-mode-gpu-2` (prio 12, **rc 0**).

The numbers above were produced by a harness that lived entirely in `trippy/render/parity.py`: it did
TRIPS's layer selection, its `compute_point_size_fac` weights and its `valid_point` break by hand, then
called `render_pyramid(num_layers=1, mode="broadcast")` once per pyramid level with
`K_l = K / 2**l` and `cx, cy + 0.5`. Nothing the *trainer* ran shared that code, so the 22.27 dB was a
statement about the parity script, not about trippy's rasteriser.

That rule is now `trippy.raster.emit`'s `mode="trips"`, together with two convention options
(`pixel_center="half"|"integer"`, `pyramid_halving="ceil"|"floor"`). One
`render_pyramid(mode="trips", pixel_center="integer", pyramid_halving="ceil", num_layers=8)` call
reproduces the whole pyramid — which `docs/TRIPS_REFERENCE.md` §6a said was impossible, on the grounds
that a fixed `cx + 0.5` shift is layer-dependent after halving. It is; applying the shift *after* the
halving, where it belongs, is not.

`trippy parity --engine native|perlayer` selects the implementation; `--compare-engines` renders both and
diffs the raw pyramid levels.

### native vs per-layer, same checkpoint, same three held-out frames

| frame | PSNR vs GT, per-layer | PSNR vs GT, native | Δ | PSNR vs authors, per-layer | native | Δ | fragments (both) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00009.jpg (idx 8) | 25.0985549739 | 25.0985550361 | 6.2e-08 | 36.2850564390 | 36.2850565255 | 8.7e-08 | 10,351,708 |
| 00121.jpg (idx 120) | 21.9790784988 | 21.9790785367 | 3.8e-08 | 37.7512073890 | 37.7512073830 | 6.0e-09 | 7,039,440 |
| 00145.jpg (idx 144) | 19.7161949726 | 19.7161949118 | 6.1e-08 | 36.9296691108 | 36.9296692467 | 1.4e-07 | 6,711,744 |
| **mean** | **22.264609482** | **22.264609495** | **1.3e-08** | **36.988644313** | **36.988644385** | **7.2e-08** | — |

SSIM agrees to ten decimal places on every frame (0.8071932197 / 0.8121092319 / 0.7811747789). The
acceptance bar was 0.05 dB; the measured gap is **1.3e-8 dB**, about 4 million times inside it.

The stronger statement is the discrete one: the two engines select **exactly the same fragments**. Total
fragment counts match to the unit, and so does the per-layer active-point vector on every frame — e.g.
frame 8: `[1091740, 863924, 428288, 150457, 46140, 6217, 1001, 160]` from both. That is not a tolerance;
`layer_higher`, the `compute_point_size_fac` branch selection, the all-four-corners `valid_point` gate and
its `break` to coarser layers either agree or they do not, and they agree on 2.2 M points x 8 layers x 3
frames. The residual in the images is one float32 ulp of the layer coordinate: `perlayer` computes it as
`fl(ip·2^-l + fl(cx·2^-l + 0.5)) - 0.5` and `native` as `fl(ip)·2^-l`, and adding then subtracting 0.5 at
a coordinate of order 10^3 loses about 1.2e-4 px.

The `broadcast` (15.141 dB) and `trilinear` (21.474 dB) ablation columns reproduced their original values
exactly; they are still rendered by the per-layer engine.

### Level-image agreement (rasteriser output, before the U-Net)

`--compare-engines` diffs the raw pyramid levels. Read `max abs diff` against the level's own feature
magnitude — these are neural-texture channels in roughly [-100, 100], not pixels in [0, 1].

| frame | level | shape | max abs diff | level magnitude | max rel diff | mean abs diff | pixels > 1e-3 |
|---|---|---|---:|---:|---:|---:|---:|
| 00009.jpg | 0 | 1080x1920 | 1.941e-03 | 6.056e+01 | 3.205e-05 | 8.895e-08 | 4 / 2 073 600 |
| | 1 | 540x960 | 3.574e-04 | 4.331e+01 | 8.254e-06 | 1.337e-07 | 0 / 518 400 |
| | 2 | 270x480 | 3.247e-04 | 3.045e+01 | 1.067e-05 | 1.379e-07 | 0 / 129 600 |
| | 3 | 135x240 | 8.297e-05 | 2.785e+01 | 2.979e-06 | 1.474e-07 | 0 / 32 400 |
| | 4 | 68x120 | 5.627e-05 | 1.148e+01 | 4.903e-06 | 1.092e-07 | 0 / 8 160 |
| | 5 | 34x60 | 6.437e-06 | 1.366e+01 | 4.711e-07 | 6.334e-08 | 0 / 2 040 |
| | 6 | 17x30 | 4.232e-06 | 1.094e+01 | 3.868e-07 | 7.037e-08 | 0 / 510 |
| | 7 | 9x15 | 1.669e-06 | 6.242e+00 | 2.674e-07 | 4.421e-08 | 0 / 135 |
| 00121.jpg | 0 | 1080x1920 | 2.954e-03 | 5.365e+01 | 5.507e-05 | 8.026e-08 | 4 / 2 073 600 |
| | 1..7 | | <= 6.06e-04 | | <= 1.35e-05 | <= 1.30e-07 | 0 |
| 00145.jpg | 0 | 1080x1920 | 1.550e-03 | 5.537e+01 | 2.799e-05 | 7.762e-08 | 10 / 2 073 600 |
| | 1..7 | | <= 4.15e-04 | | <= 8.75e-06 | <= 1.23e-07 | 0 |

Worst relative disagreement anywhere in the pyramid: **5.5e-05**, on layer 0, on four pixels out of two
million. Mean absolute disagreement: **~1e-07** on every level of every frame. Levels 1-7 have **zero**
pixels differing by more than 1e-3. The handful of layer-0 outliers are `floor()` flips: a coordinate that
lands within one float32 ulp of an integer rounds to a different base pixel in the two engines, moving one
fragment by one pixel. Full table in `output/EXP-0002-horse-parity-trips-mode/native2/README.md`.

### What this buys

`trippy train` now defaults to `mode: trips` — the layer rule the published checkpoints were trained with,
worth **+0.79 dB** over the `trilinear` reading and **+7.12 dB** over the `broadcast` one — and the
trainer, `trippy render` and the parity harness all go through one implementation, which is validated
against a real TRIPS checkpoint. Training keeps `pixel_center: half`, trippy's own convention: `integer`
exists to reproduce a TRIPS checkpoint, and using it against trippy's own undistorted images would put
every render half a pixel off its ground truth.

One caveat, and it is TRIPS's own behaviour: `mode="trips"` evaluates `valid_point` against the image
being rendered, so a training *crop*'s edge is a real image edge. Crop/full-frame equivalence therefore
holds only in the crop's interior (exact one pixel in, a `2**l`-wide band at layer l on the rim). See
`docs/LIMITATIONS.md`.

### A trap found on the way

`x.to("cpu", torch.float64)` on an MPS tensor does not raise and does not fall back — it casts on MPS,
which has no float64, and returns reinterpreted bytes (job `trippy-trips-mode-gpu-1` printed 1.5e10
maxima, NaNs and float64 denormals for feature layers whose real range is about [-100, 100]). The render
was correct; only the diagnostic reading it was wrong, which is the dangerous shape of the bug. Always
`.cpu()` first, then `.to(torch.float64)`. Logged in `docs/LIMITATIONS.md`.
