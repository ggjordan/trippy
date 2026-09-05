# trips-metal — running log

This is the chronological experiment and decision log for the trippy project. Each entry is **appended** (never rewritten) and records a decision or experiment outcome with supporting numbers and artifacts.

Entries follow this format:

```
## YYYY-MM-DD HH:MM — [Experiment / Decision / Milestone]

[One sentence summary]

**Question**: What was tested?
**Job name**: reference to `output/jobs/trippy-<name>.sh` (if applicable)
**Numbers**: key metrics (FPS, PSNR, shade audit, etc.)
**Verdict**: PASS / FAIL / INCONCLUSIVE / DECISION_MADE
**Artifact**: path to rendered output, PLY, logs, or decision doc
```

Entries describe observed facts, not intentions. Once written, an entry is not edited.

---

## 2026-09-05 22:30 — Plan approved and skeleton initialized

Decisions D1–D12 locked. Repository skeleton created: AGENTS.md, CLAUDE.md, README, STATE, VERSION, scripts, and docs/decisions/ with four ADRs. All phase 1 infrastructure in place.

**Question**: Is the repo skeleton complete and ready for research work?
**Verdict**: PASS
**Artifact**: this commit; `git ls-files` confirms no images/plys/checkpoints.
- 2026-09-05T10:49:54Z submitted job trippy-smoke prio 15: trippy smoke --device mps
- 2026-09-05T11:11:03Z smoke job trippy-smoke rc=0: torch 2.14.0 on MPS inside the Splats GPU queue, inline Metal kernel ran (add_one -> 1.0 x8). Queue round-trip proven. Log: output/logs/trippy-smoke.log
- 2026-09-05T11:24:04Z submitted job trippy-raster-gpu-tests prio 12: bash -c cd /Users/nzbirdranch/trippy/.worktrees/raster && PYTHONPATH=. /Users/nzbirdranch/trippy/.venv/bin/python -m pytest -q -m gpu -s tests/test_raster_metal.py
- 2026-09-05T11:35:26Z raster GPU tests rc=0: Metal blend_fwd vs float64 refs max|out| 1.5e-6 (trilinear+broadcast, C=3/4); 1008x756, 200k pts, L=5: 41.6 ms/forward, 1.52M fragments. int64 argsort/searchsorted/bincount all OK on MPS (no fallback needed). Log: output/logs/trippy-raster-gpu-tests.log
- 2026-09-05T11:51:31Z submitted job trippy-render-kk-1 prio 14: bash -c cd /Users/nzbirdranch/trippy/.worktrees/render-kk && PYTHONPATH=. /Users/nzbirdranch/trippy/.venv/bin/python -m trippy.cli render --scene /Users/nzbirdranch/Splats/scenes/karekare/kk-coherent --points gaussian --ply /Users/nzbirdranch/Splats/output/Training-Data/karekare/kk-coherent/kkc_15000.ply --min-opacity 0.05 --size-mode scale --width 1008 --frames IMG_3830.jpg,IMG_3828.jpg,IMG_3704.jpg,IMG_3939.jpg --mode trilinear --out /Users/nzbirdranch/trippy/output/runs/EXP-0001/trilinear --device mps
- 2026-09-05T13:13:43Z submitted job trippy-render-kk-2 prio 14: bash -c cd /Users/nzbirdranch/trippy/.worktrees/render-kk && PYTHONPATH=. /Users/nzbirdranch/trippy/.venv/bin/python -m trippy.cli render --scene /Users/nzbirdranch/Splats/scenes/karekare/kk-coherent --points gaussian --ply /Users/nzbirdranch/Splats/output/Training-Data/karekare/kk-coherent/kkc_15000.ply --min-opacity 0.05 --size-mode scale --width 1008 --frames IMG_3830.jpg,IMG_3828.jpg,IMG_3704.jpg,IMG_3939.jpg --mode broadcast --out /Users/nzbirdranch/trippy/output/runs/EXP-0001/broadcast --device mps

## 2026-09-06 01:15 — EXP-0001 `trippy render` wired: TRIPS pyramid forward on kk-coherent (no U-Net yet)

`trippy render` (new: `trippy/render/pyramid_render.py`) now loads a COLMAP scene restricted to named frames, builds a GaussianPlySource point set, rasterises the TRIPS pyramid (RGB, no network), and writes per-frame contact sheets + a summary sheet + `metrics.json`. Both layer-selection modes were run on kk-coherent's full 5,736,619-point Gaussian cloud (`min_opacity=0.05`) at 1008 wide, on MPS via the GPU queue, on 4 frames: the two shade frames IMG_3830.jpg/IMG_3828.jpg and two non-shade frames spanning the capture (IMG_3704.jpg, early; IMG_3939.jpg, late, of 219 registered images).

**Question**: does the wired-together forward pass (scene + Gaussian points + pyramid raster, RGB only) run well under budget on MPS, and does the shade region show measurably less point coverage than daylight regions?

**Job names**: `trippy-render-kk-1` (trilinear), `trippy-render-kk-2` (broadcast); both rc=0. Logs: `output/logs/trippy-render-kk-1.log`, `output/logs/trippy-render-kk-2.log`.

**Numbers** (timing_ms are emit/sort/blend/total; coverage is `mean(1 - T_final)` at level 0, computed directly from the T_final tensor, full frame and a central 50%x50% crop):

| mode | frame | total ms | fragments | points visible | coverage (full) | coverage (center) |
|---|---|---|---|---|---|---|
| trilinear | IMG_3830 (shade) | 135.7 | 6,422,926 | 1,317,832 | 0.1780 | 0.2323 |
| trilinear | IMG_3828 (shade) | 69.9 | 7,586,994 | 1,647,321 | 0.1685 | 0.2331 |
| trilinear | IMG_3704 (non-shade) | 105.9 | 13,872,692 | 3,082,048 | 0.2140 | 0.3353 |
| trilinear | IMG_3939 (non-shade) | 93.7 | 13,349,942 | 3,065,901 | 0.2206 | 0.3335 |
| broadcast | IMG_3830 (shade) | 153.5 | 24,295,378 | 1,317,832 | 0.2599 | 0.3449 |
| broadcast | IMG_3828 (shade) | 115.9 | 30,456,552 | 1,647,321 | 0.2500 | 0.3431 |
| broadcast | IMG_3704 (non-shade) | 231.1 | 56,901,327 | 3,082,048 | 0.3067 | 0.4538 |
| broadcast | IMG_3939 (non-shade) | 175.7 | 58,918,530 | 3,065,901 | 0.3163 | 0.4655 |

**Verdict**: PASS on speed (worst case 231 ms/frame, both under the 300 ms/frame budget; trilinear is 1.4-2.4x faster than broadcast because it emits far fewer layer-0 fragments). Shade-region coverage is measurably lower than non-shade in *both* modes -- center coverage is shade=0.232-0.234 vs non-shade=0.334-0.335 (trilinear) and shade=0.343-0.345 vs non-shade=0.454-0.466 (broadcast), a consistent ~30-35% relative shortfall. This is a numeric fact about the trained Gaussian cloud (fewer/weaker points survive in low light), not a rendering artifact -- holes in the shade region at level 0 are the honest, expected state before any U-Net inference, matching docs/SPEC.md's stage-1 gate framing ("shade rendered as shading, not a cloud" is the network's job, not the point source's). `broadcast` mode roughly doubles coverage at every frame relative to `trilinear` (it writes every point into layer 0 regardless of projected size) but does not close the shade/non-shade gap.

**Artifact**: `output/runs/EXP-0001/trilinear/` and `output/runs/EXP-0001/broadcast/` (summary_sheet.png, per-frame sheet.png/photo.png/level_*.png/coverage.png/depth.png, metrics.json, README.md). Delivered via `scripts/deliver.sh` as `EXP-0001-trips-pyramid-kk-trilinear` and `EXP-0001-trips-pyramid-kk-broadcast` (see delivery log below).

**Privacy note**: during CPU dry-run sanity-checking, a `sheet.png` containing the source photo panel was opened with the Read tool (family photograph, kk-coherent). This is a violation of AGENTS.md's "family photographs never leave this machine" rule (Read sends image bytes to the model API). No further photo/sheet/summary-sheet images derived from Jordan's scenes were opened afterward; all shade-coverage numbers above were computed directly from the T_final tensor, not by viewing any image. AGENTS.md was updated (see "Never send scene imagery to a model" section) to make this explicit for future sessions.
- 2026-09-05T13:16:57Z delivered EXP-0001-trips-pyramid-kk-trilinear: TRIPS pyramid forward (no network yet) on kk-coherent from 5.7M Gaussian centres: photo | level-0 splat | coverage for 4 frames incl. shade frame IMG_3830. Holes are expected before the U-Net; look at whether the shade region has point coverage. (/Users/nzbirdranch/trippy/output/runs/EXP-0001/trilinear/summary_sheet.png)
- 2026-09-05T13:17:04Z delivered EXP-0001-trips-pyramid-kk-broadcast: TRIPS pyramid forward (no network yet, broadcast layer mode) on kk-coherent from 5.7M Gaussian centres: photo | level-0 splat | coverage for 4 frames incl. shade frame IMG_3830. Holes are expected before the U-Net; look at whether the shade region has point coverage. (/Users/nzbirdranch/trippy/output/runs/EXP-0001/broadcast/summary_sheet.png)
- 2026-09-05T13:13:55Z submitted job trippy-depthpro-kk-1 prio 11: bash -c /Users/nzbirdranch/Splats/tools/vggt/.venv/bin/python3 /Users/nzbirdranch/Splats/tools/ldi/depth_batch.py /Users/nzbirdranch/trippy/output/depth/kk-coherent/manifest.json
- 2026-09-05T13:17:54Z delivered EXP-0004-monodepth-shade-coverage: DepthPro-derived points (source 2) over the 6 Karekare shade frames: do we now have geometry inside the shade region? (/Users/nzbirdranch/trippy/output/runs/EXP-0004/sheet.png)

## 2026-09-05 13:20 — EXP-0004 MonoDepthSource (DepthPro) on kk-coherent, 12 frames

Implemented `trippy.points.monodepth.MonoDepthSource` (D4 point source 2):
per-image Apple DepthPro metric depth (via Splats' `tools/ldi/depth_batch.py`,
run only through the GPU queue) -> median-ratio scale alignment to
reprojected COLMAP sparse depth -> unprojection into a world-frame
`PointSet`, voxel-deduped (reusing `UnionSource`'s dedupe helper),
provenance=MONODEPTH. New `trippy depth-points` CLI (`--run-depth` prints
the exact GPU job and exits 3 when depth outputs are missing).

**Question**: Run for real on the 6 kk-coherent shade frames (IMG_3828-3833)
+ 6 non-shade frames spread across the 219-image sequence (IMG_3703,
IMG_3753, IMG_3796, IMG_3840, IMG_3896, IMG_3940); does the shade region now
have monocular-depth geometry, and how does per-frame scale-alignment
quality compare shade vs non-shade?
**Job name**: `trippy-depthpro-kk-1` (`output/jobs/trippy-depthpro-kk-1.sh`)
**Numbers**:
- DepthPro (apple/DepthPro-hf, MPS, fp16): 12/12 images, 1008x756 each,
  1.3-1.7s/image, 18.0s total, valid_fraction=1.0 for every frame (no
  invalid/NaN depth pixels).
- Per-image median-ratio scale s (z_colmap/d_pred) and MAD (n_matches =
  sparse COLMAP points landing on valid depth pixels):
  shade -- 3828: s=1.065 mad=0.230 n=2436; 3829: s=1.420 mad=0.267 n=2450;
  3830: s=1.154 mad=0.184 n=1882; 3831: s=1.227 mad=0.152 n=1768;
  3832: s=1.485 mad=0.153 n=1366; 3833: s=1.471 mad=0.142 n=1580.
  non-shade -- 3703: s=2.781 mad=0.429 n=2937; 3753: s=1.353 mad=0.229 n=5361;
  3796: s=2.015 mad=0.177 n=4127; 3840: s=1.595 mad=0.091 n=2729;
  3896: s=1.744 mad=0.106 n=5629; 3940: s=3.358 mad=0.462 n=5456.
  Shade mean: scale=1.304, mad=0.188, n_matches=1914.
  Non-shade mean: scale=2.141, mad=0.249, n_matches=4373.
- Points contributed pre-dedupe: 21,168/image (stride=6 on 1008x756,
  valid_fraction=1.0 everywhere) x 12 = 254,016; after voxel dedupe
  (voxel=0.03): 234,712 total (7.6% collapsed). median_nn_distance=0.166
  world units. bbox [-29.7,-38.2,-58.5] to [30.9,5.5,34.6].
- Shade-frame numeric coverage (8px radius, projecting the FULL 234,712-pt
  union into each shade camera; no image was viewed by the agent, only
  computed from arrays): IMG_3828 n_visible=109,746 full=100.00%
  central-50%-box=100.00%; 3829 n_visible=117,744 full=100.00% central=100.00%;
  3830 n_visible=119,957 full=100.00% central=100.00%; 3831
  n_visible=112,897 full=99.998% central=100.00%; 3832 n_visible=95,689
  full=99.9996% central=100.00%; 3833 n_visible=86,272 full=100.00%
  central=100.00%.
**Verdict**: INCONCLUSIVE (on the coverage metric as specified) / signal
found (on scale-alignment quality). The 8px-radius point-presence coverage
metric saturates near 100% for all 6 shade frames -- this is close to a
foregone conclusion given the union of 12 dense (stride=6, i.e. ~6px native
spacing) frames and DepthPro's valid_fraction=1.0 everywhere: MonoDepthSource
assigns *some* depth to nearly every pixel including inside the shade band,
so "is there a point near this pixel" cannot distinguish shade from
non-shade. The genuinely informative number is scale-alignment confidence:
shade frames have ~44% fewer usable sparse COLMAP matches (1914 vs 4373
mean) than the spread frames, consistent with COLMAP/SIFT finding fewer
keypoints in the darker shade region, though shade-frame MAD is actually
*lower* on average (0.188 vs 0.249) -- the few matches shade frames do get
happen to agree well with each other. Whether the resulting points are
*correct* geometry (not just present) is not established by this
experiment; that needs the shade audit / Jordan's viewer verdict once this
source feeds a training run (docs/SPEC.md v0.2.0).
**Artifact**: `output/points/kk-coherent-monodepth-12.npz` (+ `.summary.json`),
`output/runs/EXP-0004/coverage_stats.json`,
`output/runs/EXP-0004/{IMG_38*_coverage.png,sheet.png}` (sheet delivered,
see delivery line above), `experiments/EXP-0004-monodepth-points/README.md`.
- 2026-09-05T13:18:13Z submitted job trippy-raster-bwd-gpu-1 prio 12: bash -c cd /Users/nzbirdranch/trippy/.worktrees/raster-bwd && PYTHONPATH=. /Users/nzbirdranch/trippy/.venv/bin/python -m pytest -q -m gpu -s tests/test_raster_bwd_metal.py

## 2026-09-06 — Milestone: the MPS pyramid rasteriser is differentiable (blend_bwd)

`blend_bwd.metal` + `blend_autograd.py` land; `render_pyramid` on MPS now returns tensors connected to
autograd for positions, sizes, confidences, features, background and an optional SE(3) pose delta.

**Question**: Do the float32 Metal backward gradients match the float64 CPU reference, and what does
forward+backward cost at a realistic frame size?

**Job name**: `output/jobs/trippy-raster-bwd-gpu-1.sh` (prio 12), rc=0, 8/8 gpu tests passed.
Log: `~/Splats/tools/gpu_queue/logs/trippy-raster-bwd-gpu-1.log`

**Numbers** — worst relative gradient error (max|metal - ref| / max|ref|, float32 MPS vs float64 CPU),
budget was 1e-3:

| input | trilinear C=3 | broadcast C=3 | trilinear C=4 | broadcast C=4 | cap+cutoff scene |
|---|---|---|---|---|---|
| xyz        | 1.170e-06 | 1.120e-06 | 1.596e-06 | 6.108e-07 | 1.268e-06 |
| size       | 1.677e-07 | no grad*  | 5.508e-07 | no grad*  | 1.437e-06 |
| conf       | 1.530e-07 | 1.765e-07 | 2.119e-07 | 2.596e-07 | 5.174e-07 |
| feat       | 1.159e-07 | 1.054e-07 | 1.466e-07 | 1.098e-07 | 1.102e-06 |
| pose_delta | 3.831e-06 | 2.601e-06 | 1.535e-06 | 2.122e-06 | n/a |
| forward    | 1.791e-07 | 2.079e-07 | 1.791e-07 | 2.150e-07 | — |

\* `mode="broadcast"` gives the layer factor 1 everywhere, so per-point size feeds nothing and has no
gradient on *either* device — TRIPS's shipped default trains no point sizes at all (§10.1).

**Worst error anywhere: 3.83e-06**, i.e. 260x inside the 1e-3 budget and roughly float32 round-off.

Fragment cap and transmittance cutoff both fire in one scene (24 fragments stacked on one layer-0 pixel,
stacked confidence raised to 0.95): `n_used` max 16 on both devices, `t_final` min 5.517e-04 vs the 1e-3
cutoff, and `n_used` is *bit-identical* between float32 Metal and float64 CPU — so the gradient comparison
is not masking a discrete disagreement.

Timing, 256x192, 50k points, C=4, L=5, 345,934 fragments: **forward 20.0 ms, forward+backward 26.4 ms,
backward 6.4 ms** (backward is 32% of forward — the whole backward is one kernel plus one `index_add_`).

Feature-only SGD, 20 steps, lr 0.2 on a 32x32 scene: loss falls monotonically
2.26881 -> 0.44316 (5.1x). Features enter the composite linearly, so the objective is exactly quadratic and
any increase would have been a wrong gradient.

CPU: 217 tests green (`-m "not gpu"`), including float64 `torch.autograd.gradcheck` on all five learnable
inputs individually and jointly (atol 1e-6, rtol 1e-4) on a hand-built fixture that sits >= 0.05 from every
discrete switch in the rasteriser.

**Design note**: the kernel uses two *division-free* suffix recurrences
(`U_{i-1} = a_i f_i + (1-a_i) U_i`, `Q_{i-1} = (1-a_i) Q_i`) instead of TRIPS's
`colour_behind / (1 - alpha + 1e-9)` (`RenderBackward.cu:290`). Algebraically identical, but exact at
`alpha == 1` with no epsilon to tune. `tests/test_raster_bwd_src.py` asserts the kernel body contains no
`/` at all.

**Verdict**: PASS
**Artifact**: `~/Splats/tools/gpu_queue/logs/trippy-raster-bwd-gpu-1.log`;
`docs/ARCHITECTURE.md` "Backward pass data flow" (formulas and the reason for the design).

**Open finding (not a blocker, but a trap for the trainer)**: `trippy.geom.xform_b.se3_exp` returns an
exactly zero gradient for the *rotation* half of a twist at `phi == 0`, because it builds the rotation as
`a * |phi| * skew(phi / max(|phi|, 1e-8))`, which is second order in `phi` at the origin (true derivative:
the SO(3) generator, magnitude 1). A pose delta initialised at exactly zero would therefore learn
translation but never rotation. Pinned by
`tests/test_raster_bwd_ref.py::test_pose_delta_rotation_gradient_vanishes_at_zero`; the fix belongs in
`xform_b.se3_exp` (and needs the xform_a/xform_b agreement test re-run), not in the rasteriser.
- 2026-09-05T13:24:52Z submitted job trippy-adop-parity-1 prio 13: bash -c cd /Users/nzbirdranch/trippy/.worktrees/adop-parity && PYTHONPATH=. /Users/nzbirdranch/trippy/.venv/bin/python -m trippy.cli parity --scene /Users/nzbirdranch/trippy/third_party/zenodo/scenes/tnt_scenes/tt_horse --checkpoint /Users/nzbirdranch/trippy/third_party/zenodo/tt_checkpoints/checkpoint_horse --epoch ep0600 --indices 8,120,144 --render-scale 1 --modes trips,broadcast,trilinear --device mps --out /Users/nzbirdranch/trippy/output/EXP-0002-horse-parity
- 2026-09-05T13:29:37Z submitted job trippy-train-smoke-1 prio 16: bash -c PYTHONPATH=. /Users/nzbirdranch/trippy/.venv/bin/python -m trippy.cli train --config experiments/EXP-0003-kk-trips-train/config_smoke.yaml --device mps --max-minutes 25
- 2026-09-05T13:26:40Z delivered EXP-0002-horse-parity: Authors' TRIPS horse checkpoint rendered through trippy's Metal rasteriser + U-Net vs ground truth (PSNR in the sheet). This is the public Tanks&Temples scene, not family data. (/Users/nzbirdranch/trippy/.worktrees/adop-parity/output/EXP-0002-horse-parity/summary_sheet.png)

## 2026-09-06 01:25 — EXP-0002: does trippy's forward render match TRIPS's own checkpoint?

**Question**: Rendered through trippy's ADOP reader + Metal pyramid rasteriser + ported U-Net +
NeuralCamera, does the authors' public `checkpoint_horse` @ ep0600 reproduce their own render of the
public Tanks & Temples `tt_horse` scene?

**Job**: `trippy-adop-parity-1` (prio 13, MPS, rc 0). Log: `output/logs/trippy-adop-parity-1.log`.
3 held-out frames (indices 8, 120, 144 from the checkpoint's own test split), 1920x1080, render_scale 1,
2,218,471 points, 8 pyramid layers, 3 layer-selection modes. 0.2-1.2 s/frame.

**Numbers** (all cropped by TRIPS's own `train_mask_border = 16`; uncropped costs ~10 dB because the
authors' saved test JPGs are blacked out that far in):

| mode | PSNR vs GT | SSIM | LPIPS | PSNR vs authors' render |
|---|---:|---:|---:|---:|
| trips (the checkpoint's real path) | **22.265** | 0.8002 | 0.1266 | **36.989** |
| broadcast (all layers, factor 1) | 15.141 | 0.6853 | 0.3411 | 15.552 |
| trilinear (two straddling layers) | 21.474 | 0.7929 | 0.1615 | 27.222 |
| *the authors' own renders* | *22.335* | *0.8171* | *0.1382* | — |

Per frame (trips): 25.099 / 21.979 / 19.716 dB against the authors' 25.186 / 22.043 / 19.775 dB.

**Verdict**: **PASS** — 0.07 dB behind the authors' own render, against a 1.5 dB bar. The v0.1.0 gate
"forward renders match a reference" is met.

**Three source-level corrections were needed** (now in `docs/TRIPS_REFERENCE.md` 2a/2b/3a/3b/6a/8a-c/9b):
1. The neural texture is used **raw**, not `abs()`-ed — `Pipeline.cpp:257` passes `non_subzero_texture`
   un-negated, contradicting the reference doc. Worth **+16.6 dB** (8.46 -> 25.10 on one frame).
2. `use_layer_point_size` is **true** for every published checkpoint — it is derived from
   `!fix_point_size` (`Settings.cpp:39`), not read from an ini, so the "always false" claim was wrong.
   It selects a different forward kernel (`RenderFast16`) whose layer rule is neither of the two the docs
   describe. Getting it wrong costs 7.1 dB (broadcast) or 0.8 dB (trilinear).
3. TRIPS's pixel centres sit on integers and its pyramid halves with `ceil`, not integer division
   (`PointRenderer.cu:385-391`, `PointBlending.h:216-240`).

**Artifact**: `output/EXP-0002-horse-parity/` (summary sheet delivered to Jordan-Review; per-frame contact
sheets carry GT | authors' render | ours | abs-diff | raw level-0 honesty panel).
- 2026-09-05T13:29:37Z submitted job trippy-train-smoke-1 prio 16: bash -c PYTHONPATH=. /Users/nzbirdranch/trippy/.venv/bin/python -m trippy.cli train --config experiments/EXP-0003-kk-trips-train/config_smoke.yaml --device mps --max-minutes 25
- 2026-09-05T13:34:01Z delivered EXP-0002-horse-parity: Authors' public TRIPS horse checkpoint rendered through trippy (Metal rasteriser + U-Net) vs ground truth: 22.27 dB, authors' own render 22.34 dB. Public Tanks&Temples data, not family photos. (/Users/nzbirdranch/trippy/output/EXP-0002-horse-parity/summary_sheet.png)
- 2026-09-05T13:53:15Z submitted job trippy-train-smoke-2 prio 16: bash -c PYTHONPATH=. /Users/nzbirdranch/trippy/.venv/bin/python -m trippy.cli train --config experiments/EXP-0003-kk-trips-train/config_smoke.yaml --device mps --max-minutes 25
- 2026-09-05T14:25:59Z submitted job trippy-trips-mode-gpu-2 prio 12: bash /tmp/trips_mode_gpu2.sh

---

## 2026-09-06 — feat/trips-mode: TRIPS's real layer rule as a native rasteriser mode

**Question**: `trippy/render/parity.py` reproduced the published TRIPS horse render (22.27 dB) with a
hand-written harness — TRIPS's layer selection, `compute_point_size_fac` weights and `valid_point` break
all lived in that script, driving one `render_pyramid(num_layers=1, mode="broadcast")` call per pyramid
level. Nothing the trainer ran shared that code. Can the same rule live in `trippy.raster.emit` as a
first-class mode, so a single multi-layer `render_pyramid` call reproduces it — and does it still score
22.27 dB?

**Jobs**: `trippy-trips-mode-gpu-1` (prio 12, **rc 0**) — GPU raster tests + both engines on tt_horse;
`trippy-trips-mode-gpu-2` (prio 12, **rc 0**) — re-run of the per-level engine diff after fixing an
MPS float64 bug in the diagnostic itself.

**What was added**
- `mode="trips"` in `emit.py` / `ref_numpy.py` / `ref_torch.py`: layers `0 .. layer_higher` with
  `layer_higher = clamp(ceil(log2 s), 0, L-1)` (`RenderForward.cu:334-338`), weight
  `compute_point_size_fac(s, layer, L)` (`PointBlending.h:81-149` — **1.0** for every layer below
  `layer_lower`, then the two interpolation weights), plus TRIPS's `valid_point` gate (all four footprint
  corners in bounds) and its `break` to coarser layers (`:340-352`).
- `pixel_center="half"|"integer"` — where the centre of pixel `i` sits (`i + 0.5` = trippy, `i` = TRIPS).
  Applied *after* the per-layer halving, which is the whole trick: `docs/TRIPS_REFERENCE.md` §6a claimed a
  single multi-layer call could not reproduce TRIPS because a fixed `cx + 0.5` shift becomes
  layer-dependent once you halve. It does — so don't apply it to `cx`.
- `pyramid_halving="ceil"|"floor"` — `ceil` is TRIPS's own branch for every published checkpoint
  (`PointRenderer.cu:385-391`), so this stops being "a deviation" and becomes an option.
- `trippy parity --engine native|perlayer --compare-engines`; trainer default `mode: trips`.

**Numbers** (tt_horse `checkpoint_horse` @ ep0600, frames 8/120/144, 16 px border crop):

| engine | mean PSNR vs GT | mean PSNR vs authors' render | mean SSIM |
|---|---:|---:|---:|
| perlayer (the original harness) | 22.264609482 | 36.988644313 | 0.8001590768 |
| **native** (one `render_pyramid(mode="trips")` call) | **22.264609495** | **36.988644385** | 0.8001590768 |
| Δ | **1.3e-08 dB** | 7.2e-08 dB | 0 |

Per frame the gap is at most 1.4e-07 dB. The acceptance bar was 0.05 dB. Ablations reproduced exactly:
`trilinear` 21.474 dB, `broadcast` 15.141 dB, authors' own render 22.335 dB.

The discrete check is the stronger one: the two engines select **identical fragments**. Total counts match
to the unit (10,351,708 / 7,039,440 / 6,711,744) and so does the per-layer active-point vector on every
frame (frame 8: `[1091740, 863924, 428288, 150457, 46140, 6217, 1001, 160]` from both). `layer_higher`,
the factor branch, the four-corner gate and the `break` either agree or they do not, over 2.2 M points x 8
layers x 3 frames. The image residual is one float32 ulp of the layer coordinate (~1.2e-4 px at layer 0):
`perlayer` computes `fl(ip·2^-l + fl(cx·2^-l + 0.5)) - 0.5`, `native` computes `fl(ip)·2^-l`.

Level images (the U-Net's input, before any network runs): worst relative disagreement anywhere in the
8-level pyramid is **5.5e-05**, on layer 0, on 4 pixels out of 2 073 600; mean absolute disagreement
**~1e-07** per level; levels 1-7 have zero pixels differing by more than 1e-3 (feature channels range
about [-100, 100]). The layer-0 outliers are `floor()` flips — a coordinate within one float32 ulp of an
integer picks a different base pixel — not a rule difference.

**Verdict: PASS.** The trainer, `trippy render` and the parity harness now share one rasteriser, and that
rasteriser is validated against a real TRIPS checkpoint. `mode: trips` is worth **+0.79 dB** over trippy's
old `trilinear` default and **+7.12 dB** over `broadcast`.

**Two things worth remembering**
1. `x.to("cpu", torch.float64)` on an MPS tensor does **not** raise and does **not** fall back — it casts
   on MPS, which has no float64, and returns reinterpreted bytes. Job 1's engine-diff table printed 1.5e10
   maxima, NaNs and float64 denormals for feature layers whose real range is about [-100, 100]. The render
   was correct; only the diagnostic reading it was wrong. Always `.cpu()` first, then `.to(torch.float64)`.
2. `mode="trips"` evaluates `valid_point` against the image being rendered, so a training **crop**'s edge
   is a real image edge and crop/full-frame equivalence holds only in the crop's interior (exact one pixel
   in; a `2**l`-wide band at layer l on the rim). This is TRIPS's own behaviour — it trains on crops with
   the same rule — but it is new for trippy and it is in `docs/LIMITATIONS.md`.

**Artifacts**: `output/EXP-0002-horse-parity-trips-mode/{perlayer,native,native2}/` (metrics.json, README
with the per-level engine-agreement table, per-frame contact sheets). Job logs:
`$SPLATS_ROOT/tools/gpu_queue/logs/trippy-trips-mode-gpu-{1,2}.log`. Nothing under `output/` is committed.
- 2026-09-05T14:01:19Z submitted job trippy-trips-mode-gpu-1 prio 12: bash /tmp/trips_mode_gpu.sh
- 2026-09-05T14:52:56Z submitted job trippy-train-smoke-3 prio 16: bash -c cd /Users/nzbirdranch/trippy/.worktrees/train-debug && PYTHONPATH=. /Users/nzbirdranch/trippy/.venv/bin/python -m trippy.cli train --config experiments/EXP-0003-kk-trips-train/config_smoke.yaml --run-dir /Users/nzbirdranch/trippy/output/runs/EXP-0003-kk-trips-train/EXP-0003-kk-trips-train_smoke3 --device mps --max-minutes 25

## 2026-09-06 02:53 — EXP-0003: why did the first real training run render black (1.61 dB)?

**Question**: `trippy-train-smoke-2` (kk-coherent, 504 px, 2 epochs, 48 steps, MPS) finished rc 0 but
reported held-out **PSNR 1.61 dB / SSIM 0.054 / LPIPS 0.824** with the training loss flat at 1.2-1.8.
A PSNR that low means MSE ~0.7 on [0,1] images — the prediction is not merely bad, the pipeline is broken.
Which stage?

**Method**: a CPU diagnostic (`output/diag/train_stage_stats.py`) rebuilds a `Trainer` from the run's own
`checkpoint_ep0000.pt` with `trippy.train.eval.build_trainer_from_checkpoint(device="cpu")`, renders one
held-out 252 px crop through the trainer's own `_render` / `_tone_map`, and prints min/max/mean/std at
every stage. Runs in 3 s. No image was opened; numbers only.

**Stage trace** (held-out `IMG_3703.jpg`, 200k points, `mode=trilinear`, 5 layers):

| stage | min | max | mean |
|---|---:|---:|---:|
| pyramid layer 0 (4 ch) | +0.0028 | +0.812 | +0.276 |
| U-Net output (3 ch) | -0.177 | +0.364 | +0.113 |
| after exposure (`x * 2**-EV`, EV = 6.585) | -0.0019 | +0.0038 | +0.0012 |
| after white balance / vignette | unchanged (init is identity) | | |
| after response LUT | +0.0022 | +0.0235 | **+0.0116** |
| target photo | 0.0 | 1.0 | **+0.457** |

**Two independent root causes**:

1. **Exposure was initialised with the absolute EXIF EV instead of the EV relative to the scene mean.**
   TRIPS: `colmap2adop.cpp:105` stores `scene_exposure_value = mean(EV)`, `NeuralScene.cpp:38` initialises
   the per-frame exposure as `f.exposure_value - scene_exposure_value`. `Trainer._initial_exposure` used
   the raw EV. kk-coherent's EVs are 4.99-7.31, mean **6.14**, so every prediction was multiplied by
   `2**-6.14 = 1/70` before the response LUT. `lr_exposure = 5e-4` moves it 4e-3 per epoch — unrecoverable.
2. **The eval PSNR was 4.771 dB too low.** `((pred - target)**2 * mask).sum() / mask.sum()` with a
   3-channel error and a 1-channel mask is exactly `3x` the MSE. Measured ratio on the checkpoint:
   `3.0000`. The "1.61 dB" was really 6.38 dB.

Three more found while measuring, all fixed: `cfg.background` was ignored (the constant was used);
training crop centres were sampled over the whole frame so roughly half of every crop was masked-out
padding that still cost a rasterisation (TRIPS's `RandomImageCrop`, `Dataset.cpp:264`, keeps the crop
inside the image); and the trainer never seeded the *global* torch RNG, so `cfg.seed` did not reproduce
the U-Net init (6.7 dB vs 8.4 dB at init across two runs of one config).

**Jobs**: `trippy-train-smoke-3` (prio 16, MPS, **rc 0**) — same config as smoke-2 with only `run_dir`
changed, so the comparison is the fixes and nothing else. `trippy-train-smoke-4` (prio 16, MPS, **rc 0**)
— the same again after rebasing onto main's native `mode: trips` (#11), to check the fixes are
mode-independent. Both: 2 epochs, 24 steps/epoch, 33 held-out images.

| run | epoch | PSNR | SSIM | LPIPS | loss (first / last / mean) |
|---|---:|---:|---:|---:|---|
| smoke-2 (before) | 0 | 1.183 | 0.0253 | 0.7845 | 1.184 / 1.047 / 1.511 |
| smoke-2 (before) | 1 | **1.609** | 0.0537 | 0.8245 | 1.490 / 1.507 / 1.525 |
| smoke-3 (after, `trilinear`) | 0 | 12.116 | 0.1975 | 0.7921 | 1.639 / 1.375 / 1.473 |
| smoke-3 (after, `trilinear`) | 1 | **12.250** | 0.1995 | 0.7773 | 1.324 / 1.397 / 1.376 |
| smoke-4 (after, `trips`) | 0 | 12.130 | 0.1972 | 0.7879 | 1.639 / 1.375 / 1.467 |
| smoke-4 (after, `trips`) | 1 | **12.258** | 0.1990 | 0.7738 | 1.323 / 1.393 / 1.373 |

**Verdict**: **PASS** — +10.65 dB on the identical config and step count, loss now decreasing rather than
flat, and above the >12 dB bar the brief set for 2 epochs. Neither root cause interacts with the pyramid
layer-selection mode (the exposure gain is applied after the U-Net; the PSNR bug is in the metric), and
smoke-4 confirms it: `trips` and `trilinear` land within 0.01 dB of each other. The remaining number is
still low because a trippy "epoch" is 24 crops (`train_factor = 0.125`, one crop per step): a CPU
rehearsal at 186 steps/epoch reaches **13.09 dB after epoch 0** and **13.47 dB after epoch 5** on the
same scene. `nonfinite_grads = 0` on every step of both runs.

**A ceiling worth knowing about**: on kk-coherent the pyramid is nearly empty above level 0 in *every*
mode — mean `t_final` per level (finest to coarsest) is 0.93/0.98/0.97/0.94/0.95 for `trilinear`,
0.93/0.98/0.97/0.94/0.95 for `trips`, and only `broadcast` fills the coarse levels
(0.90/0.76/0.59/0.54/0.66). `trips` collapses onto `trilinear` here because
`layer_higher = clamp(ceil(log2(size_px)))` is 0 for every sub-pixel footprint and the 3DGS-derived point
sizes mostly are. The U-Net is inventing 90%+ of every frame; more points, larger `size0`, or
`mode: broadcast` are the levers. Table in `docs/LIMITATIONS.md`.

**Also found, not fixed (out of file scope)**: a NaN gradient escapes `trippy.raster`'s backward for a
degenerate fragment. Reproduced on CPU (`train_factor = 1.0`, 6 epochs) — one point's `xyz`/`raw_size`
and 1-2 frames' pose deltas become NaN in epoch 4. The image loss stays finite (a NaN position fails
every bounds test, so the point is culled), but `_extent_penalty` reduces over all points, so the
*reported* loss is NaN for the rest of the run while held-out PSNR keeps climbing (13.39 -> 13.47 dB).
Contained by `Trainer._sanitise_gradients` (zeroes non-finite grads, logs the count as `nonfinite_grads`
in `metrics.jsonl`; smoke-3 recorded 0). Root cause belongs in `trippy/raster/` — see
`docs/LIMITATIONS.md`, "NaN gradient out of the rasteriser backward".

**Regression cover**: `tests/test_train_regression.py` (8 tests, 2 s on CPU). The synthetic fixture's
photos now carry real EXIF (EV ~8.2), so the exposure bug is reachable from the CPU suite at all; with
both bugs reintroduced, 4 of the 8 tests fail. The headline assertion is a sanity floor: held-out PSNR
after 40 steps must beat the PSNR of a constant image at the target's own mean (20.03 dB vs 18.26 dB).

**Incident (self-reported)**: the first CPU diagnostics ran from the worktree without `TRIPPY_OUTPUT`
set, so `trippy.config.load_settings` defaulted to `<repo>/output` and `SceneDataset` wrote a 131 MB
undistorted cache of kk-coherent into `.worktrees/train-debug/output/cache/` — scene imagery inside the
repo, against AGENTS.md Sec. 6. It was `.gitignore`d (`output/`) and never staged; deleted with `rm -rf`
and re-run with `TRIPPY_OUTPUT=/Users/nzbirdranch/trippy/output`. Any script run with cwd inside a
worktree needs that variable set explicitly (worktrees carry no `.env`).

**Artifacts**: `output/runs/EXP-0003-kk-trips-train/EXP-0003-kk-trips-train_smoke{3,4}/` (log.txt,
metrics.jsonl, eval_ep0000/, eval_ep0001/ incl. the honesty sheets, checkpoints/, export.ply);
diagnostics under `output/diag/`.
- 2026-09-05T15:02:32Z submitted job trippy-train-smoke-4 prio 16: bash -c cd /Users/nzbirdranch/trippy/.worktrees/train-debug && PYTHONPATH=. /Users/nzbirdranch/trippy/.venv/bin/python -m trippy.cli train --config experiments/EXP-0003-kk-trips-train/config_smoke.yaml --run-dir /Users/nzbirdranch/trippy/output/runs/EXP-0003-kk-trips-train/EXP-0003-kk-trips-train_smoke4 --device mps --max-minutes 25
- 2026-09-05T15:09:55Z submitted job trippy-train-full1 prio 70: bash -c PYTHONPATH=. /Users/nzbirdranch/trippy/.venv/bin/python -m trippy.cli train --config experiments/EXP-0003-kk-trips-train/config.yaml --device mps --max-minutes 360
- 2026-09-05T15:09:55Z submitted job trippy-train-full1-broadcast prio 70: bash -c PYTHONPATH=. /Users/nzbirdranch/trippy/.venv/bin/python -m trippy.cli train --config experiments/EXP-0003-kk-trips-train/config_broadcast.yaml --device mps --max-minutes 360
- 2026-09-05T13:47:54Z submitted job trippy-hybrid-c-render-1 prio 17: bash -c cd /Users/nzbirdranch/trippy/.worktrees/hybrid-c && PYTHONPATH=. /Users/nzbirdranch/Splats/tools/ml-sharp/.venv/bin/python -m trippy.hybrid.render_splat_views --scene /Users/nzbirdranch/Splats/scenes/karekare/kk-coherent --ply /Users/nzbirdranch/Splats/output/Training-Data/karekare/kk-coherent/kkc_15000.ply --out /Users/nzbirdranch/trippy/.worktrees/hybrid-c/output/hybrid-c/renders/w1008 --width 1008 --device mps --start-index 0 --end-index 110
- 2026-09-05T13:53:29Z submitted job trippy-hybrid-c-render-2 prio 17: bash -c cd /Users/nzbirdranch/trippy/.worktrees/hybrid-c && PYTHONPATH=. /Users/nzbirdranch/Splats/tools/ml-sharp/.venv/bin/python -m trippy.hybrid.render_splat_views --scene /Users/nzbirdranch/Splats/scenes/karekare/kk-coherent --ply /Users/nzbirdranch/Splats/output/Training-Data/karekare/kk-coherent/kkc_15000.ply --out /Users/nzbirdranch/trippy/.worktrees/hybrid-c/output/hybrid-c/renders/w1008 --width 1008 --device mps --start-index 110 --end-index 219
- 2026-09-05T14:52:23Z submitted job trippy-hybrid-c-train-1 prio 18: trippy hybrid-c train --config experiments/EXP-0005-hybrid-c/config.yaml --max-minutes 40
- 2026-09-05T14:54:22Z submitted job trippy-hybrid-c-train-1 prio 18: bash -c cd /Users/nzbirdranch/trippy/.worktrees/hybrid-c && PYTHONPATH=. /Users/nzbirdranch/trippy/.venv/bin/python -m trippy.cli hybrid-c train --config experiments/EXP-0005-hybrid-c/config.yaml --max-minutes 40
- 2026-09-05T15:45:57Z delivered EXP-0005-hybrid-c-refine: Design C: U-Net refines Gaussian renders of kk-coherent toward photos. Sheet: photo | Gaussian render | refined | diff on held-out frames incl. shade. Numbers in README. (/Users/nzbirdranch/trippy/.worktrees/hybrid-c/output/runs/EXP-0005-hybrid-c/EXP-0005-hybrid-c_1/eval_ep1125/sheet.png)

## 2026-09-06 -- EXP-0005 Hybrid design C: render->photo U-Net refinement

**Question**: render `kkc_15000.ply` for every registered kk-coherent view with Splats'
`gsrender.py` (rgb + depth + alpha, `max_hw=400`), then train trippy's U-Net + neural camera
to map render -> photo -- does a learned renderer change anything in the shade region
specifically, or only sharpen already-well-covered pixels? (docs/PLAN-2026-09-05.md's cheap
side-experiment, meant to validate net/losses before the more expensive A1 design.)

**Job names**: `hybrid-c-render-1` (prio 17, rc=0, 1495.1 s, frames 0-110), `hybrid-c-render-2`
(prio 17, rc=0, 1675.8 s, frames 110-219) -- 219/219 registered views rendered, 0 skipped,
0 errors. `hybrid-c-train-1` (prio 18, rc=0; first submission failed rc=1, `.venv` missing
in this git worktree -- fixed by invoking the main repo's `.venv/bin/python` directly with
`PYTHONPATH=.` from the worktree dir, same `bash -c` pattern as the render jobs, then
resubmitted under the same job name). Training: 40.0-minute wall-clock budget, reached
epoch 1125 (~27,000 crop steps at 384x384, `mode` n/a -- Design C has no rasteriser --
on MPS) before the budget stopped it.

**Numbers** (held-out split: 33 frames, 27 non-shade + 6 forced shade `SHADE_FRAMES_KK`;
final eval `eval_ep1125/metrics.json`; baseline = raw Gaussian render vs photo, no U-Net at
all, identical at every epoch by construction):

| Metric | Baseline | Refined | delta |
|---|---|---|---|
| PSNR, all (n=33) | 15.53 dB | 15.54 dB | +0.01 dB |
| PSNR, non-shade (n=27) | 15.66 dB | 16.11 dB | +0.45 dB |
| PSNR, shade (n=6) | 14.94 dB | 12.97 dB | -1.96 dB |
| SSIM, all | 0.431 | 0.476 | +0.045 |
| SSIM, non-shade | 0.432 | 0.483 | +0.051 |
| SSIM, shade | 0.427 | 0.442 | +0.015 |
| LPIPS, all (lower better) | 0.477 | 0.461 | -0.015 |
| LPIPS, non-shade | 0.465 | 0.448 | -0.018 |
| LPIPS, shade | 0.526 | 0.519 | -0.007 |

Checked the shade PSNR regression against 5 intermediate checkpoints (`eval_ep{0050,0200,
0500,0800,1125}/metrics.json`): refined shade PSNR is 12.15, 14.86, 13.54, 13.48, 12.97 dB
against a flat 14.94 dB baseline -- a stable -1 to -2 dB deficit from early training onward
(epoch 200 briefly nearly matches baseline, then the regression widens and holds), not a
transient early-training artifact. Non-shade PSNR rises above baseline by epoch 200 and
holds a stable +0.4 to +0.5 dB gain from there on. Aggregate "all" PSNR stays flat because
the two effects partly cancel across a 27-vs-6-frame average.

**Verdict**: FAIL on the question this experiment asked. A learned renderer does change the
shade region, but it makes shade PSNR *worse* (~2 dB), not better, while giving only small
SSIM/LPIPS gains there; it clearly helps the non-shade region on every metric. Consistent
with (not proof of): the shade region's Gaussian render carries more low-coverage/alpha
holes than non-shade (EXP-0001's T_final finding), and this design's deliberately
full-frame loss mask (`alpha>0` OR-ed with an all-ones mask, always all-ones) gives the
U-Net no extra incentive to fill those holes *correctly* rather than merely plausibly --
SSIM/LPIPS reward structural/perceptual plausibility more than exact per-pixel brightness,
which is exactly the asymmetry the numbers show. Does not remove or improve the shade
defect on the primary photometric metric; trades shade accuracy for non-shade gains. Not a
reason to iterate further on Design C for the shade problem -- supports moving to A1
(Gaussians as TRIPS points with learned feature vectors, joint training) next per
docs/SPEC.md's v0.3.0 plan.

**Artifact**: `output/runs/EXP-0005-hybrid-c/EXP-0005-hybrid-c_1/` (`eval_ep*/metrics.json`,
`eval_ep*/sheet.png`, `eval_ep*/shade_frames/*.png`, `metrics.jsonl`, `log.txt`,
`checkpoints/`); delivered sheet `EXP-0005-hybrid-c-refine`
(`~/Splats/output/Jordan-Review/4-other/EXP-0005-hybrid-c-refine.png`); full writeup
`experiments/EXP-0005-hybrid-c/README.md`.
- 2026-09-05T15:34:04Z submitted job trippy-brush-pyramid-gpu-1 prio 12: bash -c cd /Users/nzbirdranch/trippy/.worktrees/brush-pyramid/rust && cargo test -p brush-pyramid --features gpu --offline --test parity_gpu -- --nocapture --test-threads=1
- 2026-09-05T15:50:39Z submitted job trippy-brush-pyramid-gpu-2 prio 12: bash -c cd /Users/nzbirdranch/trippy/.worktrees/brush-pyramid/rust && cargo test -p brush-pyramid --features gpu --offline --test parity_gpu -- --nocapture --test-threads=1

## 2026-09-06 — v0.4.0: TRIPS pyramid rasteriser forward pass ported to Rust/CubeCL

**Question**: can the Python/Metal pyramid forward be ported to Rust + Burn + CubeCL on
wgpu (Metal now, WebGPU later) and reproduce `trippy.raster.render_pyramid` exactly?

**Layout finding (no GPU needed)**: trippy's crates did **not** have to move into the
`ggjordan/brush` fork. `rust/crates/brush-pyramid` path-depends into
`rust/brush-trips/crates/{brush-cube,brush-sort,brush-prefix-sum}` from the separate
thin workspace, given three things: `exclude = ["brush-trips"]` (Cargo otherwise
auto-adopts a path dependency under the workspace root as a *member*, and
`brush-cube`'s `log.workspace = true` then resolves against the wrong workspace); a
verbatim copy of both `[patch]` tables (Cargo reads `[patch]` only from the workspace
root being built, so without it we would link unpatched wgpu/cubecl, which cannot
compile Brush's kernels to MSL); and `rust/Cargo.lock` seeded from the submodule's
lock (burn is pinned only by `branch = "main"`). ADR-0005 stands; nothing was pushed
to the fork. Build times: 1m25s for the library cold, 55s more for the test binaries;
`scripts/test.sh` unaffected at 37s because the `gpu` feature is off by default.

**CubeCL/Metal finding**: CubeCL exposes `ln` but **not** `log2`, and
`ln(x) * (1/ln 2)` in f32 lands on the wrong side of an integer at exact powers of
two — which moves a point into the wrong pyramid layer. Both Rust paths now read the
IEEE-754 exponent field instead (`floor(log2 x)` exactly, for any positive normal
float). Writing the CPU twin the same way turned a GPU-only risk into a CPU test:
`factor::tests::exponent_and_log2_bounds_differ_only_next_to_a_power_of_two` sweeps
every power of two and its neighbours and pins the divergence from `torch.log2` to
within ~1e-6 relative of a power of two. `docs/LIMITATIONS.md` records it.

**CPU parity (job: none, runs anywhere)**: `cargo test -p brush-pyramid` — Rust CPU
reference vs the Python `.npy` fixtures, all six (mode x pixel_center) combinations:
max |feature| diff **1.2e-7 to 2.1e-7**, max |t_final| diff 2.4e-7 to 4.2e-7,
`n_used` and per-layer fragment counts **exactly equal**. Fixtures:
`tests/fixtures/synthetic/raster_fixture_*/`, 292 KiB total, synthetic, generated by
`tools/dump_raster_fixture.py` (64x48, 3 layers, 500 points, C=4, with deliberate
clusters that force the `max_frags` cap, the `t_cutoff` early-out, and exact depth
ties so the sort's tie-break is exercised).

**GPU parity, run 1 (job `trippy-brush-pyramid-gpu-1`, prio 12, rc 101)**:
`broadcast` and `trilinear` matched Python **exactly** on fragment counts and to
**1.4e-6 - 2.2e-6** on the images, in both pixel-centre conventions. Mode `trips`
came out **4 fragments short** (1704 vs 1708) — exactly one point losing one pyramid
layer, i.e. the only trips-specific code, the `valid_point` footprint gate and its
`break`. Ruled out by measurement rather than guesswork: projection reassociation and
FMA variants (`(fx*x)/z + c`, `fx*(x/z) + c`, single-rounded, fused `+c`) and both
`size_px` associations all give the same 1708 in a float32 Python model of the
emitter, and `broadcast` matching to 2e-6 with exact counts proves `u`, `v`, the
footprint bases, the sort, the blend and the background are all already right.

**Root cause of the mode-`trips` shortfall (found on CPU, no GPU needed)**: not a
logic bug — a fixture parked on a floating-point knife edge. `layer_bounds` is
floor/ceil of `log2(size_px)`, so at exactly `size_px = 2^k` a point has
`lower == upper` and mode `trips` writes both layers at factor 1.0 (8 fragments);
one ulp *below* `2^k` it straddles, the lower layer's factor collapses to ~1.2e-7,
and those four fragments fall under `alpha_min` (4 fragments). `size_px` is
`fx * size / z`, which a shader compiler may reassociate or lower to a fast
reciprocal, so CPU and GPU can land on opposite sides. The cutoff cluster in
`tools/dump_raster_fixture.py` had 40 points at exactly `size_px = 2.0`; one of them
flipping accounts for exactly the observed -4. Ruled the alternatives out first by
enumerating 32 combinations of projection association, alpha association, the
`layer < lower` branch, gate strictness and log2-vs-exponent bounds in a float32
Python model — none reproduced 1704, which is what pointed at the reciprocal rather
than at any of them. Fix: the cluster now uses `size_px = 6.0`, which clamps to
`lower == upper == 2` at 3 layers and is stable under any perturbation (it also
raised layer-2 coverage from 68 to 228 fragments);
`factor::tests::a_size_on_a_power_of_two_is_a_knife_edge_...` pins the semantics on
both sides of the edge, and `docs/LIMITATIONS.md` records that the residual
~1e-7-probability disagreement is inherent to TRIPS's discontinuous rule.

**GPU parity, run 2 (job `trippy-brush-pyramid-gpu-2`, prio 12, rc 0)** — all five
tests green, 0.82 s:

| fixture | fragments | slots | per layer | max abs diff vs Python | max abs `t_final` |
|---|---|---|---|---|---|
| `broadcast_half`    | 4130 | 6000 | 1303 / 1411 / 1416 | 2.056e-6 | 3.040e-6 |
| `broadcast_integer` | 4263 | 6000 | 1412 / 1422 / 1429 | 2.205e-6 | 3.636e-6 |
| `trilinear_half`    | 1520 | 2380 | 1057 / 220 / 243   | 1.431e-6 | 1.907e-6 |
| `trilinear_integer` | 1634 | 2380 | 1170 / 219 / 245   | 1.580e-6 | 2.921e-6 |
| `trips_half`        | 1868 | 1988 | 1252 / 388 / 228   | 1.431e-6 | 1.907e-6 |
| `trips_integer`     | 2008 | 2008 | 1372 / 400 / 236   | 1.580e-6 | 2.921e-6 |

Tolerance is 1e-4, so the worst case has ~45x headroom. `n_used`, `num_fragments`
and the per-layer split are **exactly** equal to Python's on all six. Two stronger
checks also pass: per-**layer-pixel** fragment counts agree across all 4032 pyramid
pixels, and the per-point slot budgets agree for all 500 points — so the counting
kernel, the emission kernel and the segment table are in lockstep, not merely
agreeing in aggregate. GPU vs the Rust CPU twin: 1.4e-6 to 2.2e-6, which matters
because the two emit in different orders (CPU layer-major, GPU point-major) and so
this is what proves the two-pass radix sort's tie-breaking really is equivalent to
Python's composite key. An empty point set renders a pure background with
`t_final == 1` everywhere.

**Verdict**: the TRIPS pyramid forward pass reproduces trippy's Python forward on
wgpu/Metal, for all three layer-selection modes and both pixel-centre conventions.
**Artifacts**: `output/logs/trippy-brush-pyramid-gpu-{1,2}.log`; fixtures at
`tests/fixtures/synthetic/raster_fixture_*/` (388 KiB, committed).
- 2026-09-05T16:08:33Z submitted job trippy-brush-pyramid-gpu-3 prio 12: bash -c cd /Users/nzbirdranch/trippy/.worktrees/brush-pyramid/rust && cargo run --release --example render_frame --features gpu --offline -- --points ../tests/fixtures/synthetic/raster_fixture_trips_half/points.npz --camera ../tests/fixtures/synthetic/raster_fixture_trips_half/camera.json --mode trips --layers 3 --background 0.1,0.2,0.3,0.4 --out /Users/nzbirdranch/trippy/.worktrees/brush-pyramid/output/render_frame_gpu.png

(The `trippy-brush-pyramid-gpu-3` line logged above was cancelled before it ran: it
would have done a full `--release` build of the Burn/CubeCL/wgpu tree *inside* the
GPU lock, which is CPU work that has no business holding a shared GPU queue slot.
Rebuilt in debug through `scripts/cpu_heavy.sh` and resubmitted instead.)
- 2026-09-05T16:11:11Z submitted job trippy-brush-pyramid-gpu-4 prio 12: bash -c set -e; cd /Users/nzbirdranch/trippy/.worktrees/brush-pyramid/rust && cargo test -p brush-pyramid --features gpu --offline --test parity_gpu -- --nocapture --test-threads=1 && cargo run --example render_frame --features gpu --offline -- --points ../tests/fixtures/synthetic/raster_fixture_trips_half/points.npz --camera ../tests/fixtures/synthetic/raster_fixture_trips_half/camera.json --mode trips --layers 3 --background 0.1,0.2,0.3,0.4 --out /Users/nzbirdranch/trippy/.worktrees/brush-pyramid/output/render_frame_gpu.png
- 2026-09-05T15:38:47Z submitted job trippy-raster-nan-gpu-1 prio 12: bash -c cd /Users/nzbirdranch/trippy/.worktrees/raster-nan && PYTHONPATH=. TRIPPY_OUTPUT=/Users/nzbirdranch/trippy/output /Users/nzbirdranch/trippy/.venv/bin/python -m pytest -q -m gpu tests/test_raster_nan_metal.py

## 2026-09-06 — fix/raster-nan: the NaN gradient out of the rasteriser backward, found and fixed

**Question**: where does the NaN gradient reported in `docs/LIMITATIONS.md` ("Not fixed: a NaN gradient
out of the rasteriser backward") actually come from, and can the reference *and* the Metal path be made
finite on every degenerate fragment?

**Repro** (CPU, no GPU): `experiments/EXP-0003-kk-trips-train/config_smoke.yaml`, `device=cpu`,
`mode=trilinear`, `max_points=200000`, `train_factor=1.0`, 6 epochs, `Trainer._sanitise_gradients`
replaced by a finiteness probe that runs after `backward()` and before `optimizer.step()`.
(`output/diag/find_nan3.py` and `cpu_short_train.py`, which produced the original report, both survive.)

**Root cause**: `trippy.raster.emit.project_points` divided by the raw camera-space z. The failing input
is a point at camera-space z **exactly 0.0** — point 964/200000, world `(-1.3989772, 0.60192317,
5.9451413)`, camera-space `(-1.2281361, 3.7178385, 0.0)`, frame `IMG_3811.jpg`. The point is culled
(`cull_points` needs `depth > znear`), so its incoming `uv` gradient is exactly zero — but torch
differentiates `n / z` w.r.t. the denominator as `-grad * (n/z/z)` and evaluates it for every row,
culled ones included: `-0 * inf = NaN`. The NaN goes into that point's `xyz` gradient and, via
`world_to_cam`, into all six components of that frame's pose delta; Adam converts it to a NaN parameter.
On the *next* step the NaN `xyz` gave a NaN depth, `clamp(nan, min=znear)` kept it NaN, and
`size_px = fx*size/nan` handed `raw_size` a NaN gradient — which is exactly the reported
"`xyz` (3) + `raw_size` (1) + one pose delta" signature, one step apart. `z == 0` is not exotic in
float32: z is the third component of `xyz @ R.T + t`, so any point on a camera's principal plane rounds
to it.

**Fix**: `trippy.raster.emit.safe_depth(depth, znear) = where(depth > znear, depth, znear)` is now the
divisor of both projection divisions. Bit-identical for every point that survives the cull, so the
forward is untouched; `where` not `clamp` so a NaN depth is replaced rather than propagated. Second,
smaller fix: `ref_torch.composite_sorted`'s alpha clamp now uses
`max(RASTER_ALPHA_MAX_EPS, finfo(dtype).eps)` — the 1e-12 constant rounds back to 1.0 in float32, so
`alpha == 1` gave `log1p(-1) = -inf` and then `-inf - -inf = NaN` (docs/LIMITATIONS.md EXP-0002 entry).

**Numbers (controlled A/B on the failing run, only the patch differs)**: bit-identical losses for the
first 693 steps (3 epochs + 135 steps). At step 693 the unpatched run emits `{xyz: 3, pose: 6}` non-finite
gradient entries; at step 694 `{xyz: 3, raw_size: 1, pose: 6}` and the loss is NaN from there on. The
patched run completes 5 epochs / 930 steps with **zero** non-finite gradients across every parameter
(points, sizes, confidences, features, background, poses, U-Net, exposure, response) and no NaN loss;
the full 6-epoch run is clean too. The synthetic-scene trainer repro (`tests/test_train_helpers.py`
scene, `train_factor=1.0`, 6 epochs, sanitiser disabled) is likewise clean.

**Metal**: no kernel change needed. Emission/projection is the same torch code on both devices, so
`safe_depth` covers MPS; `blend_bwd.metal` is division free by construction (suffix recurrences `U`/`Q`,
never TRIPS's `colour_behind / (1 - alpha)`), so it never had the alpha hazard the torch twin did.

**Verdict**: fixed, not contained. `Trainer._sanitise_gradients` stays as a backstop.

**Tests**: `tests/test_raster_nan_ref.py` (117 CPU cases: zero depth, depths at/inside/behind the near
plane, depths that overflow `n/z/z` in float32, fragment on an exact pixel boundary, `size_px` an exact
power of two / 1 / 0, alpha exactly 0 and 1, plus the reduced single-fragment case and forward-neutrality
checks) — 25 of them fail with either bug reintroduced. `tests/test_raster_nan_metal.py` (51 GPU cases,
each diffed against the float64 CPU reference at 1e-3 relative). Full CPU suite 530 passed.

**Job**: `trippy-raster-nan-gpu-1` (prio 12) — 51 passed in 1.60s, **rc 0**.
