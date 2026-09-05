# Limitations and constraints

## Hardware and software

- **No 64-bit atomics on Metal via `torch.mps.compile_shader`**: This rules out TRIPS's reference depth/id packing trick. Our design avoids atomics entirely by pre-sorting fragments and using segment offsets instead.

- **No float64 on MPS**: gradient checking (gradcheck) must run on CPU. We validate gradients at float32 precision on GPU and float64 on CPU before any MPS job, but cannot run float64 gradcheck directly on the GPU.

- **int64 argsort at 50M elements unverified**: sorting 7.36M points × up to 8 fragments per point can produce ≤50M fragments. PyTorch's int64 argsort is not guaranteed stable at this scale on MPS. We have a fallback: two stable 32-bit sorts (Brush's pattern, e.g., radix sort by layer first, then by pixel+depth).

- **Perceptual loss differs from TRIPS's caffe VGG**: we use `torchvision.models.vgg16` + `lpips` for perceptual loss. TRIPS's original uses caffe VGG. Results may differ slightly. We do not attempt to port the caffe model; the difference is noted and accepted.

## Data and geometry

- **Distortion refinement out of scope**: COLMAP models include lens distortion parameters, but we apply them once during dataset loading (undistort) and do not refine them during training. If distortion changes during training (e.g., due to pose drift), it will not be corrected.

- **iPhone LiDAR 5 m range, incomplete coverage**: LiDAR on iPhone 13 Pro Max has a ~5 m practical range and often fails on glass/reflective surfaces. Scenes exceeding 5 m or with reflective geometry require depth backfill from monocular depth (DepthPro/MoGe). This is handled via `UnionSource` but cannot be automated.

## Quest rendering

- **~120 GFLOP per eye per frame**: A CNN per eye per frame at mobile headset resolution (≈1024×1024) requires approximately 120 billion floating-point operations. This exceeds the typical throughput budget of a mobile GPU (~30 GFLOP/s for sustained load), meaning interactive rendering on Quest is not expected.

- **Measurement deferred until v0.5.0**: We will measure the web viewer on a Quest browser once the web viewer ships. If fps is unacceptable (<15 fps), we ship a fallback: distilled Gaussians via the existing `~/Splats/tools/publish/publish_splat.sh` path, or fly-through MP4 videos.

## Memory and disk

- **96 GB unified memory on this Mac**: sufficient for training at half-resolution (1024 wide) with crop sizes 384–512 and batch size 1. Full-resolution training would require careful scheduling or gradient checkpointing.

- **Disk was 93% full on 2026-09-05**: scenes and PLYs are read in place from `SPLATS_ROOT` (an environment variable, never a copy); run outputs live under `output/` (gitignored) and `scripts/deliver.sh` symlinks them into the review folder rather than copying.

## Reproduction

- **Public Zenodo scene only**: this repo is public. We use a public TRIPS scene from Zenodo (record 10687419, CC-BY) as the reproducible example for outsiders. Jordan's Karekare and Hunua scenes are private and must never be committed.

- **No checkpoint files committed**: trained models (`.pt`, `.pth`, `.safetensors`) are delivered via `scripts/deliver.sh` to `~/Splats/output/Jordan-Review/`, outside the repo. The repo contains only training code and public data.

## Pyramid rasteriser forward pass (v0.1.0)

- **~~The MPS forward is not differentiable.~~ Resolved (v0.2.0, `blend_bwd`).** `render_pyramid` on `device="mps"` now returns tensors connected to autograd; `xyz`, `size`, `conf`, `feat`, `bg` and an optional SE(3) `pose_delta` all receive gradients. See the "Pyramid rasteriser backward pass" section below for what the backward still does *not* cover.

- **Fragments below `RASTER_ALPHA_MIN` (1e-5) are dropped at emission.** TRIPS keeps every in-bounds bilinear corner. We drop the negligible ones because each costs a slot in the 16-deep per-pixel list; the composite changes by at most 1e-5 × the feature magnitude. All three implementations (numpy, torch, Metal) apply the identical rule, so the references stay comparable, but a bit-exact TRIPS comparison would need it set to 0.

- **The background is always added as `T_final · bg`.** TRIPS only adds it when `alpha_dest >= ALPHA_DEST_CUTOFF` (`RenderForward.cu:3610`), i.e. it hard-zeroes a residual of up to 0.001. Ours is the continuous version — the difference is ≤0.001 × bg and it avoids a discontinuity in the loss.

- **The out-of-bounds test is per fragment in `trilinear`/`broadcast`, per point in `trips`.** TRIPS rejects a point's *whole* 2×2 footprint at a layer if `floor(ip)` is out of `[0, w_l-2] × [0, h_l-2]`, and then abandons the remaining (coarser) layers entirely (`RenderForward.cu:340-352`). `mode="trips"` implements exactly that rule, including the `break`. Modes `"trilinear"` and `"broadcast"` keep trippy's older per-corner rule instead, so a point straddling the border still contributes its in-bounds corners and a point off the edge of layer 0 can still draw in a coarse layer. That is strictly more correct at borders and it means those two modes' fragment counts near the frame edge do not match TRIPS's. Two rules now coexist on purpose; do not "unify" them.

- **Pyramid halving is a `pyramid_halving` option, default `"ceil"`.** `"ceil"` is what TRIPS does for every published checkpoint (`network_version != "MultiScaleUnet2d"`, `PointRenderer.cu:385-391`); `"floor"` reproduces the other branch. This entry previously called `ceil` a deviation from TRIPS — it is not (docs/TRIPS_REFERENCE.md §3b). `layer_grid` raises rather than build an empty layer under `"floor"`.

- **The pixel-centre convention is a `pixel_center` option, default `"half"`.** TRIPS's `ip` puts pixel centres on integers; trippy puts them at `i + 0.5`, which is what its own COLMAP intrinsics and undistortion cache assume. Training therefore always runs on `"half"`; `"integer"` exists only for TRIPS-checkpoint parity. Rendering a trippy-trained model with `"integer"` (or a TRIPS checkpoint with `"half"`) shifts the image by half a layer-`l` pixel on layer `l`, which is a real, visible error at coarse layers — not a rounding detail.

- **`use_layer_point_size` mode parity is testable, and `"trips"` is the mode that matters.** This entry previously said the point-size path could not be reached from any shipped `.ini`. It can: `use_layer_point_size = !fix_point_size` (`Settings.cpp:39`) and `fix_point_size = false` in every published `params.ini` (docs/TRIPS_REFERENCE.md §2b/3a). `mode="trips"` is that path and is validated against a real checkpoint. `mode="trilinear"` ports `CollectTiled2Pointsize`, reachable only with `combine_lists = true`, and has *not* been checked against a released checkpoint; `mode="broadcast"` corresponds to `fix_point_size = true`, which no released Tanks & Temples checkpoint uses.

- **`mode="trips"` breaks crop/full-frame equivalence on the crop's rim** — and this is TRIPS's own behaviour, not a porting bug. TRIPS's `valid_point` gate is evaluated against *the image being rendered*, so a training crop's edge is a real image edge: a point one pixel outside the crop is dropped, where in the full frame it was interior and drew normally. The affected band is `2**l` layer-0 pixels wide at layer l (1 px at layer 0, 16 px at layer 4), so at the trainer's 384 px crop with L=5 the outer ~4% of the crop has thinner coarse-layer coverage than a full-frame render would give it. Modes `"trilinear"` and `"broadcast"` keep exact crop equivalence, because their per-corner rule only ever loses the corners that genuinely fall outside. Measured and pinned by `tests/test_train_crop_equivalence.py::test_crop_equals_cropped_full_render_trips_mode_in_the_interior`: exact one pixel in, ~0.45 feature units on the outermost ring. If this ever shows up as a visible crop-boundary artefact in training, the fix is to render a crop with a `2**(L-1)` px margin and discard it, not to weaken the gate.

- **Fragment memory scales with the mode.** In `trips` and `broadcast` modes a point emits up to `4 × L = 20` fragments, not 8. At 7.36M points and L=5 that is up to 147M fragments before the bounds cull, well past the 50M figure assumed above for int64 argsort; large scenes will need tiling or crop-based rendering (the trainer already renders 384–512 px crops). `trips` is the trainer default, so this is now the *normal* case, not a corner: in practice most points are sub-pixel (`layer_higher = 0`) and only the near-field ones reach `4 · L`.

- **Composite sort key caps the pyramid at 2^31 layer-pixels.** The key packs `layer_pixel << 32 | float32_bits(depth)` into an int64. Beyond ~2.1 G layer-pixels (far above any realistic frame) `sort_fragments` raises and the `two_pass` fallback must be used; it has no such limit.

- **Depth is compared at float32 resolution in both sort paths**, because the composite key has only 32 bits for it. Two fragments whose depths differ only below float32 resolution are ordered by point index, not by true depth.

## Pyramid rasteriser backward pass (v0.2.0)

- **`aux["depth_sum"]` carries no gradient on MPS.** `blend_bwd.metal` differentiates the composited colour and the final transmittance only; the depth moment is returned detached (`ctx.mark_non_differentiable`), so backpropagating through it raises instead of silently producing zeros. The CPU reference *is* differentiable in `depth_sum`, so a depth-supervision loss written and validated on CPU will fail on MPS until the kernel gains a `d_depth` output. `aux["n_used"]` is an integer count and carries no gradient anywhere.

- **Fragment ordering carries no gradient**, exactly as in TRIPS. The sort permutation and the segment offsets are discrete functions of depth; they are applied with `index_select` on integer indices. The rendered image is therefore piecewise smooth, not smooth: at a depth crossing, at an image border, at `alpha == RASTER_ALPHA_MIN`, at `size_px == 2**k` (where `layer_bounds`' `floor/ceil` switches), at the 16-fragment cap and at the transmittance cutoff, the gradient is a one-sided derivative. This is why `tests/test_raster_bwd_scenes.py` is a hand-built fixture rather than a random scene: gradcheck's finite differences straddle those switches on a random scene and report failures that are not gradient bugs.

- **Fragments outside the composited prefix get exactly zero gradient.** The backward is handed the forward's `n_used`, so anything the forward skipped (past the 16-fragment cap, or after transmittance fell below `RASTER_T_CUTOFF`) contributes nothing. This is the correct derivative of the function actually computed, but note that the function actually computed is not the true volume-rendering integral — a point hidden behind a saturated pixel receives no gradient and cannot learn its way to the front.

- **No `1 / (1 - alpha)` guard, by design.** TRIPS divides by `1 - alpha + 1e-9` when back-propagating the transmittance dependency (`RenderBackward.cu:290`). We instead carry two division-free suffix recurrences (see docs/ARCHITECTURE.md "Backward pass"), which are exact for every `alpha` in `[0, 1]` including `alpha == 1`. There is therefore no epsilon to tune and no accuracy cliff for large alpha; `tests/test_raster_bwd_src.py::test_kernel_never_divides_by_one_minus_alpha` pins the absence of the division.

- **No camera-intrinsics or distortion gradients.** TRIPS computes `g_k` (5 params) and `g_dis` (8 params) alongside the pose gradient (`RenderBackward.cu:384-465`). trippy computes neither: `K` is treated as a constant. Pose refinement is supported through `pose_delta`; intrinsics refinement is not, and adding it means extending `trippy.geom.xform_b.project_pinhole`'s graph, not the kernel.

- **Double backward is not supported on MPS.** `BlendFunction.backward` is wrapped in `once_differentiable`, so a second-order method (or any loss whose graph differentiates a gradient) raises rather than returning wrong numbers. The CPU reference has no such restriction.

- **The per-point feature gradient reduction is a torch `index_add_`, not an atomic.** On MPS this is a separate kernel launch over F fragments after `blend_bwd`, and its summation order is torch's, not ours; results are deterministic in practice on this backend but the order is not contractually guaranteed by torch. TRIPS instead accumulates with `atomicAdd` per fragment, which is *less* deterministic.

- **`mode="broadcast"` gives per-point sizes no gradient at all** — which is why it is not the trainer default. With `use_layer_point_size=false` the layer factor is 1 everywhere and each point is written to every layer, so `size` feeds nothing that affects the image; the render is not even connected to `size` in the autograd graph. This is not a bug in the backward; it is what `fix_point_size = true` does. Modes `trips` and `trilinear` both give `size` a real gradient, on the two layers the projected footprint straddles.
## net/ (feat/net, 2026-09-05): U-Net, neural camera, losses port

- **Gated conv block: VERIFIED, not a guess.** `External/saiga/` is empty in the vendored TRIPS
  checkout, but Saiga is public MIT source; it was fetched directly from
  `https://github.com/darglein/saiga` @ `ee7a4e6b65832433e2ca521353b7b7431c8e17a0`
  (`src/saiga/vision/torch/PartialConvUnet2d.h:108-152`). The formula is exactly `norm(act(conv_a(x)) *
  sigmoid(conv_b(x)))` with two independent 3x3 convs reading the same input, `norm="id"` a no-op by
  default. See `docs/TRIPS_REFERENCE.md` Sec. 5a for the full quote and a real-checkpoint verification
  (34/34 tensors shape-matched against `third_party/zenodo/tt_checkpoints/checkpoint_horse/ep0600/
  render_net.pth`, see the checkpoint-load report below).

- **SSIM: VERIFIED, and NOT the generic 11x11 Wang et al. window this task's brief assumed as a
  fallback.** Also fetched from the same public Saiga repo (commit
  `5fb87057f09f518b1ecf7de1a486420681455892`,
  `src/saiga/vision/torch/ImageSimilarity.h:73-126`, class `SSIMImpl`). TRIPS instantiates it with its own
  class defaults (`SSIM loss_ssim = SSIM();`, `Pipeline.h:238`): `radius=2` (a **5x5** Gaussian window, not
  11x11), `sigma=1.5`, `max_value=1` (`C1=1e-4, C2=9e-4`), depthwise (`groups=channels`) convolution. Ported
  exactly in `trippy/net/losses.py::ssim_map`. Note for future readers of this task's own brief: it
  authorized falling back to the generic 11x11 window "if you could not fetch it" -- that fallback was not
  needed and was NOT used; do not swap in an 11x11 window later under the mistaken impression that was the
  final state.

- **VGG perceptual loss: NOT bit-exact, by design, per the task brief.** TRIPS's actual `loss_vgg` is
  `Saiga::PretrainedVGG19Loss`, a custom Caffe-derived VGG19 loaded from a pre-traced TorchScript file
  (`loss/traced_caffe_vgg_optim.pt`) that is not in this checkout and not human-readable even if it were.
  `trippy.net.losses.TripsLoss` substitutes `lpips.LPIPS(net='vgg')` (torchvision VGG16 backbone +
  lpips.LPIPS's own linear calibration layers) for the `vgg` weight, as directed. Loss values will differ
  numerically from a real TRIPS run; do not compare absolute loss magnitudes across the two
  implementations, only trends within trippy's own training runs.

- **LPIPS (`loss_lpips`, weight 0 by default): VERIFIED net choice.** Saiga's own `LPIPS` class docstring
  (`ImageSimilarity.h:192-198`, same fetch as above) gives the exact tracing recipe used to produce
  `loss/traced_lpips.pt`: `lpips.LPIPS(net='alex')`. `trippy.net.losses.TripsLoss` uses `net='alex'` for
  the `lpips` weight (distinct from the `vgg` weight's `net='vgg'`) for this reason.

- **Rolling shutter: not ported.** `NeuralCameraImpl` supports a learned per-image 2-channel flow-field
  grid-sample warp (`enable_rolling_shutter`, `NeuralCamera.cpp:189-212`), off by default in every shipped
  config. `trippy/net/camera_model.py::NeuralCamera` does not implement it; out of scope for this task.

- **`configs/train_normalnet.ini`'s `num_layers=5` does NOT match the publicly released Tanks & Temples
  checkpoints, which all use `num_layers=8`.** Discovered by extracting and inspecting a real checkpoint's
  own `params.ini` (see the load report immediately below) -- not something visible from reading the
  shipped config alone. `trippy.net.unet.NetworkConfig` defaults to `num_layers=5` per this task's brief
  (which named `train_normalnet.ini` as the config to read), but every field needed to reconstruct the
  released checkpoints is otherwise identical (see `docs/TRIPS_REFERENCE.md` Sec. 5a) -- a caller wanting
  checkpoint-compatible weights must pass `NetworkConfig(num_layers=8)`.

- **`CombineBridge` odd-size generalization is a deliberate, documented deviation from TRIPS's literal
  C++.** TRIPS's own `CombineBridge(below, skip)` (`Networks.h:766-773,1060-1067`) assumes
  `skip.size() >= below.size()` and only crops `skip`. Given TRIPS's own floor-halving pyramid construction
  (`h/=2; w/=2` per level, `PointRenderer.cu:378`), an odd intermediate pyramid dimension makes the raw
  finer-level input (`below`) one pixel *larger* than 2x the coarser level (`skip`) -- the opposite of what
  TRIPS's crop direction assumes. A literal port, fed such a pyramid, would fail (or silently mis-size) in
  `torch.cat`. `trippy.net.unet.combine_bridge` generalizes to crop whichever operand is larger down to the
  shared minimum size on each axis; this is bit-identical to TRIPS in the well-behaved (even) case and only
  differs, safely, in the case TRIPS's own code cannot handle. Consequence: the network's output spatial
  size for a base resolution not divisible by `2**(num_layers-1)` is `floor(size / 2**(num_layers-1)) *
  2**(num_layers-1)`, not the base size. Worked numerically in `trippy/net/unet.py`'s module docstring and
  tested in `tests/test_net_unet.py::test_forward_odd_size_centre_crops_to_multiple_of_16`.

- **`torch.jit.load` DOES read `render_net.pth`**, correcting `docs/TRIPS_REFERENCE.md` Sec. 9/11's
  original "not TorchScript, cannot `torch.jit.load`" claim -- see the new Sec. 9a there for the full
  explanation (it reads as a named-tensor bag, not a runnable traced graph, but that is exactly what a
  shape-matched weight loader needs). `trippy.net.checkpoint.try_load_trips_network` still tries
  `torch.load(weights_only=False)` as a documented fallback in case a different TRIPS build's archive
  format does not parse via `torch.jit.load`.

### Checkpoint load attempt report (task item 5)

`third_party/zenodo/tt_checkpoints.zip` (2,654,478,381 bytes) finished downloading during this task and
`unzip -t` reported no errors across the whole archive. Per the task's "extract selectively, never the
whole zip" instruction, only the smallest scene's *network*-related files were extracted (total scene
sizes compared via `unzip -l`: horse=142,299,316 bytes was smallest of the 8 scenes; family, francis,
lighthouse, m60, panther, playground, train are all larger) into
`third_party/zenodo/tt_checkpoints/checkpoint_horse/` (512 KB total): `ep0600/render_net.pth`,
`ep0600/scene_tt_horse_{response,vignette,wb,ex}.{pth,txt,csv}`, and `params.ini`. The multi-hundred-MB
`scene_tt_horse_texture.pth` and `scene_tt_horse_points.pth` were deliberately NOT extracted (out of scope
for the net/ module; texture and point-cloud state belong to a future points/scene task).

`try_load_trips_network("render_net.pth", target=MultiScaleUnet2dDecOnlySmallFixed(NetworkConfig
(num_layers=5)))` (trippy's shipped default) reports `ok=False`: 34 tensors found via the `jit` reader, but
only 16/22 of the `num_layers=5` target's tensors shape-match (the checkpoint's `up7..up4` blocks have no
counterpart in a 5-layer network, and the 5-layer target's `up[0]`/`final` land on the wrong checkpoint
tensors once the layer counts diverge). Rebuilding the target with `NetworkConfig(num_layers=8)` (matching
`checkpoint_horse/params.ini`, see above) gives `ok=True`: **34/34 tensors assigned, every shape matching
exactly**, in registration order. This was run manually against the extracted file (not committed as an
automated pytest test, since the checkpoint file lives outside the git-tracked worktree and cannot be
required by CI); the exact commands are reproducible from `docs/TRIPS_REFERENCE.md` Sec. 9a.

### `x.to("cpu", torch.float64)` on an MPS tensor returns garbage, silently

MPS has no float64. `Tensor.to(device, dtype)` performs the dtype cast on the *source* device, so this
one-liner casts float32 -> float64 on MPS: it does **not** raise, even with
`PYTORCH_ENABLE_MPS_FALLBACK=0`, and it does not fall back — it returns reinterpreted bytes. Observed in
job `trippy-trips-mode-gpu-1`: a 1920x1080 feature layer whose real range is about [-100, 100] came back
with a maximum of 1.5e10, NaNs on three of eight levels, and float64 denormals (4.9e-324) on others. The
render itself was correct; only the diagnostic that read it was wrong, which is the dangerous shape of
this bug. **Always `.cpu()` first, then `.to(torch.float64)`.** `.to("cpu", torch.float32)` is safe and is
what the rest of `trippy.render.parity` uses.

### Native vs per-layer `trips` engine (feat/trips-mode)

`trippy parity --engine native` renders the whole pyramid with a single
`render_pyramid(mode="trips", pixel_center="integer", pyramid_halving="ceil")` call;
`--engine perlayer` is the original loop of `num_layers=1` calls with `cx, cy + 0.5`. They select
*identical* fragments (same `points_active` per layer, same total fragment count — a discrete quantity, so
it matches exactly or not at all) but they are not bit-identical in float32: `perlayer` computes the layer
coordinate as `fl(ip·2^-l + fl(cx·2^-l + 0.5)) - 0.5` and `native` as `fl(ip)·2^-l`. Adding and then
subtracting 0.5 at a coordinate of order 10³ costs about one float32 ulp (~1.2e-4 px at layer 0), which
propagates into the bilinear weights. `--compare-engines` measures it per level. `native` is the more
accurate of the two and is the default; `perlayer` is kept as the independent check.

Measured on tt_horse at 1920x1080 x 8 levels (job `trippy-trips-mode-gpu-2`, rc 0): identical fragment
counts and identical per-layer active-point vectors; worst relative level-image disagreement 5.5e-05 on
layer 0 (4 pixels out of 2 073 600), mean ~1e-07 per level, zero pixels over 1e-3 on levels 1-7; PSNR gap
1.3e-08 dB. The layer-0 outliers are `floor()` flips at coordinates within one ulp of an integer.

## EXP-0002 (feat/adop-parity, 2026-09-06): reproducing the published TRIPS horse render

`trippy parity` now renders the authors' `checkpoint_horse` @ `ep0600` end to end and lands within
0.07 dB of their own saved render (22.27 dB vs 22.34 dB mean over three held-out frames; see
`experiments/EXP-0002-horse-parity/README.md`). What is *not* bit-exact, and what is still unverified:

- **Not bit-exact, and cannot be from Python.** TRIPS blends in a CUDA warp with `__shfl` reductions over
  float32; trippy sorts the whole fragment list and blends in a Metal kernel. Ordering of equal-depth
  fragments, float32 accumulation order, and the `abs(ip - g) > 1` defensive guard
  (`RenderForward.cu:3507`, unreachable for an exact 2x2 footprint) all differ at the ULP level. Measured
  agreement against the authors' own render is 37.0 dB PSNR, not infinity.
- **Border-pixel rule: now implemented in the rasteriser itself.** TRIPS drops a point from layer `l` when
  `floor(ip_l)` is outside `[0, w_l-2] x [0, h_l-2]` and then `break`s out of all coarser layers.
  `trippy.raster.emit.emit_fragments` implements both rules in `mode="trips"` (they used to live only in
  `trippy.render.parity`'s per-layer loop). Its per-corner drop still runs afterwards but is a no-op there,
  since the point-level test already guarantees all four corners are inside.
- **`render_scale != 1` is untested for parity.** `size_px = fx * softplus(size) / z` scales with `fx`, so
  a downscaled render moves points into different pyramid layers than the checkpoint was trained for. A
  1/8-scale smoke render of frame 8 scores 10.4 dB. Parity numbers are only claimed at `render_scale = 1`.
- **The `trilinear` and `broadcast` ablation columns are still rendered by the per-layer engine.**
  `broadcast` reproduces TRIPS's per-layer gate exactly; `trilinear` renders one multi-layer
  `render_pyramid` call with `cx, cy + 0.5`, so its layer 0 is aligned and its coarser layers are up to
  half a layer-pixel off. Both exist to reproduce *wrong* readings of the reference doc, so neither is a
  parity claim. Mode `trips` has two engines that must agree (below).
- **`alpha == 1.0` exactly makes the float32 CPU reference compositor produce NaN.**
  `trippy.raster.ref_torch.composite_sorted` clamps alpha to `1 - RASTER_ALPHA_MAX_EPS` with
  `RASTER_ALPHA_MAX_EPS = 1e-12`, which is a no-op in float32 (`1 - 1e-12 == 1.0f`), so `log1p(-1) = -inf`
  and the per-segment rebase `exclusive - exclusive[start]` becomes `-inf - -inf`. Unreachable in a real
  render (confidence is a sigmoid output, strictly < 1) and unreachable on the Metal path (which loops
  sequentially), but it bit `tests/test_parity_render.py` when a test used `conf = 1.0`. Not fixed here:
  `trippy/raster/ref_torch.py` is outside this task's file list.
- **Masks are untested.** `tt_horse`'s `masks.txt` is 151 blank lines and its `params.ini` has
  `use_image_masks = false`, so the mask path in `trippy.scene.adop_io` (which parses the file and exposes
  `AdopView.mask_path`) has never been exercised against a scene that actually has masks.
- **Only one scene, one epoch, three frames.** The other seven published Tanks & Temples scenes, and every
  frame outside `{8, 120, 144}`, are unrendered. The three chosen frames are all in the authors' own test
  split (`checkpoint_horse/test_indices_tt_horse.txt`), so none of them is a training view.
- **The brief asked for `00200.jpg`; the scene has 151 images (`00001.jpg`..`00151.jpg`).** Frame index
  144 was substituted — the last index in the checkpoint's own test split, and the one its
  `img_worst_144_output.jpg` names as the run's worst test frame.
- **Rolling shutter and motion blur are still not ported** (both `false` in the horse `params.ini`, so
  they do not affect this result).

## train/ (fix/train-debug, 2026-09-06): why the first EXP-0003 smoke run rendered black

The first real training run (`trippy-train-smoke-2`, kk-coherent, 2 epochs, 48 steps) finished rc 0 and
reported **1.61 dB / SSIM 0.054 / LPIPS 0.824** on the held-out split. Two independent root causes (an
exposure-initialisation bug that rendered the frame black, and a metric bug that understated the number
by 4.771 dB), plus three smaller ones found while measuring -- all now fixed. The same config, same 48
steps, re-run as `trippy-train-smoke-3` reports **12.25 dB / SSIM 0.199 / LPIPS 0.777**.
None of these interact with the pyramid layer-selection `mode`: the exposure gain is applied after the
U-Net and the PSNR bug is in the metric, so `trilinear`, `trips` and `broadcast` were all equally black.

**Fixed — exposure was initialised with the absolute EXIF EV, not the EV relative to the scene mean.**
`NeuralCamera` applies exposure as a gain, `x = x * 2 ** -EV` (`NeuralCamera.cpp:307-309`). TRIPS
initialises the per-frame value as `EV - scene_exposure_value` where `scene_exposure_value` is the
*mean* EV over the dataset (`colmap2adop.cpp:105` writes it, `NeuralScene.cpp:38` subtracts it), so the
initialisation only encodes relative brightness differences and the average frame starts at gain 1.
`Trainer._initial_exposure` used the absolute EV. kk-coherent's EVs run 4.99-7.31 (mean 6.14), i.e. every
prediction was divided by ~70 before the response LUT. Measured on `checkpoint_ep0000`: U-Net output mean
+0.113, after exposure +0.0012, after the LUT +0.0116 against a target mean of 0.457 -- a black frame.
`lr_exposure = 5e-4` moves the value by 4e-3 in a whole epoch, so training cannot climb out of it.

**Fixed — the eval PSNR was 4.771 dB too low.** `Trainer.evaluate` computed
`((pred - target)**2 * mask).sum() / mask.sum()` with `pred` (1, 3, H, W) and `mask` (1, 1, H, W): a
3-channel error sum over a 1-channel mask sum, i.e. exactly `3x` the MSE and `10*log10(3) = 4.771 dB` off
every reported number. Now `trippy.net.losses.mse_loss`, which averages over the elements the mask keeps.
The reported 1.61 dB was really 6.38 dB.

**Fixed — `cfg.background` was ignored**; the trainer always used the `TRAIN_DEFAULT_BACKGROUND` constant.

**Fixed — training crops were sampled overlapping the frame border.** `_sample_crop_center` drew the
centre uniformly over the *whole* frame, so a `crop/zoom` window routinely hung half outside the image;
those pixels are masked out of the loss but still cost a full rasterisation. TRIPS's `RandomImageCrop`
(`Dataset.cpp:264`) keeps the crop inside the image and only *biases* where it lands
(`crop_prefere_border`). trippy now samples uniformly inside, and only falls back to the frame centre on
an axis where the window is genuinely larger than the image.

**Fixed — the trainer never seeded the global torch RNG.** `cfg.seed` only fed `Trainer._rng` (image,
zoom and crop sampling); the U-Net's and NeuralCamera's weight init went through torch's default
generator, so two runs of one config started from different networks (observed: 6.7 dB vs 8.4 dB held-out
at init on the same config). `Trainer.__init__` now calls `torch.manual_seed(cfg.seed)`.

### Not fixed: a NaN gradient out of the rasteriser backward (contained, not cured)

Reproduced on CPU with the fixed trainer on kk-coherent (`config_smoke.yaml`, `train_factor = 1.0`,
6 epochs): somewhere in epoch 4 a *single* point's `xyz` (3 entries) and `raw_size` (1 entry) become NaN,
along with the pose delta of the 1-2 frames that point was last rendered in. The image loss stays finite
-- a NaN position fails every bounds comparison, so the point is culled from every subsequent render --
but `_extent_penalty` reduces over *all* points, so the reported total loss is NaN from then on while the
held-out PSNR keeps improving (13.39 -> 13.47 dB across the NaN). The root cause is a NaN gradient
produced inside `trippy.raster`'s backward for a degenerate fragment; that package is outside this task's
file list, so it is reported rather than fixed. Related and already known:
`trippy.raster.ref_torch.composite_sorted` produces NaN for `alpha == 1.0` exactly (see the EXP-0002
section above), and `xform_b.se3_exp` has a zero -- not NaN -- rotation gradient at `phi == 0`.

Containment: `Trainer._sanitise_gradients` zeroes non-finite gradient entries before `optimizer.step()`
and records the count as `nonfinite_grads` in `metrics.jsonl`, so one degenerate fragment can no longer
write NaN into a checkpoint, and the event is visible instead of silent. `trippy-train-smoke-3` recorded
`nonfinite_grads = 0` for all 48 steps.

### Known gaps in the trainer (not bugs, but they bound what a smoke run can show)

- **`crops_per_step` is not implemented.** It exists in `TrainConfig` but the trainer does one crop per
  optimiser step; TRIPS does `batch_size=4 x inner_batch_size=4 = 16`.
- **An "epoch" is 24 crops on kk-coherent** (`train_factor = 0.125` x 186 training images), so a 2-epoch
  smoke run is 48 optimiser steps. 12.25 dB is what 48 steps buys; a CPU rehearsal at 186 steps/epoch
  reaches 13.09 dB after one epoch and 13.47 dB after six (both `mode: trilinear`; `mode: trips`
  gives 12.27 / 12.53 dB at 24 steps/epoch, within noise of `trilinear`'s 12.12 / 12.25 -- see the
  coverage table below for why).
- **The pyramid is nearly empty above level 0 on this scene, in every layer-selection mode.** Measured
  at init on a held-out kk-coherent view (200k points from `kkc_15000.ply`, 504 px), mean `t_final` per
  level, finest to coarsest -- 1.0 means "nothing was drawn here":

  | mode | L0 | L1 | L2 | L3 | L4 | fragments |
  |---|---:|---:|---:|---:|---:|---:|
  | `trilinear` | 0.934 | 0.985 | 0.975 | 0.944 | 0.948 | 416,721 |
  | `trips` (TRIPS's own rule, now native since #11) | 0.931 | 0.978 | 0.965 | 0.943 | 0.949 | 430,376 |
  | `broadcast` | 0.901 | 0.761 | 0.592 | 0.544 | 0.661 | 1,951,937 |

  `trips` behaves almost identically to `trilinear` here because `layer_higher = clamp(ceil(log2(size_px)))`
  is 0 for every point whose projected footprint is under a pixel, and the 3DGS-derived point sizes on
  kk-coherent mostly are. The U-Net therefore sees 90%+ background at *every* level, so it has to invent
  most of the frame. This bounds achievable PSNR far more than any remaining trainer bug; the levers are
  more points (`point_source.max_points`), larger `size0`, or `mode: broadcast`. Not a bug -- recorded so
  the next person does not read a 12 dB smoke run as a broken trainer.
