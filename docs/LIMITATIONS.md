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

- **The out-of-bounds test is per fragment, not per point.** TRIPS rejects a point's *whole* 2×2 footprint at a layer if `floor(ip)` is out of range, and then abandons the remaining (coarser) layers entirely: the broadcast kernel `break`s out of the layer loop (`RenderForward.cu:1610-1620`) and the point-size kernel nests the `layer_higher` write inside the `layer_lower` bounds check (`RenderForward.cu:2296-2360`), so a point on the border loses its coarse-layer contribution too. We test each of the four corners independently and each layer independently, so a point straddling the border still contributes its in-bounds corners, and a point off the edge of layer 0 can still draw in a coarse layer. This is strictly more correct at borders; it also means our fragment counts near the frame edge do not match TRIPS's.

- **Pyramid layers use `ceil` halving, TRIPS uses integer division.** See docs/GEOMETRY.md. Odd-sized images differ in the last row/column of every layer.

- **`use_layer_point_size` mode parity is untestable against a TRIPS checkpoint.** `mode="trilinear"` ports a code path that no shipped TRIPS `.ini` can enable (docs/TRIPS_REFERENCE.md §11), so it can never be validated against a released TRIPS checkpoint — only against our own references. `mode="broadcast"` is the mode a TRIPS checkpoint would have been trained in.

- **Fragment memory scales with the mode.** In `broadcast` mode a point emits up to `4 × L = 20` fragments, not 8. At 7.36M points and L=5 that is up to 147M fragments before the bounds cull, well past the 50M figure assumed above for int64 argsort; large scenes will need tiling or crop-based rendering (the trainer already renders 384–512 px crops).

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

- **`mode="broadcast"` gives per-point sizes no gradient at all.** With `use_layer_point_size=false` the layer factor is 1 everywhere and each point is written to every layer, so `size` feeds nothing that affects the image and `size.grad` is `None`. This is not a bug in the backward; it is what TRIPS's shipped default does, and it is why every released TRIPS checkpoint's point sizes are untrained (docs/TRIPS_REFERENCE.md §10.1).
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

## EXP-0002 (feat/adop-parity, 2026-09-06): reproducing the published TRIPS horse render

`trippy parity` now renders the authors' `checkpoint_horse` @ `ep0600` end to end and lands within
0.07 dB of their own saved render (22.27 dB vs 22.34 dB mean over three held-out frames; see
`experiments/EXP-0002-horse-parity/README.md`). What is *not* bit-exact, and what is still unverified:

- **Not bit-exact, and cannot be from Python.** TRIPS blends in a CUDA warp with `__shfl` reductions over
  float32; trippy sorts the whole fragment list and blends in a Metal kernel. Ordering of equal-depth
  fragments, float32 accumulation order, and the `abs(ip - g) > 1` defensive guard
  (`RenderForward.cu:3507`, unreachable for an exact 2x2 footprint) all differ at the ULP level. Measured
  agreement against the authors' own render is 37.0 dB PSNR, not infinity.
- **Border-pixel rule differs by one pixel.** TRIPS drops a point from layer `l` when
  `floor(ip_l)` is outside `[0, w_l-2] x [0, h_l-2]` and then `break`s out of all coarser layers.
  `trippy.render.parity` reproduces both rules, but `trippy.raster.emit.emit_fragments` additionally drops
  individual out-of-bounds *corners*, which TRIPS never reaches because its point-level test already
  guarantees all four are inside. No observable difference; noted so a future refactor does not "fix" one
  of the two.
- **`render_scale != 1` is untested for parity.** `size_px = fx * softplus(size) / z` scales with `fx`, so
  a downscaled render moves points into different pyramid layers than the checkpoint was trained for. A
  1/8-scale smoke render of frame 8 scores 10.4 dB. Parity numbers are only claimed at `render_scale = 1`.
- **The `trilinear` ablation renders one multi-layer `render_pyramid` call**, so its coarse layers use
  trippy's corner-origin halving rather than TRIPS's `ip *= 0.5f`. Layer 0 is aligned (the `cx, cy` +0.5
  shift is applied); coarser layers are up to half a layer-pixel off. That is fine for an ablation column
  and wrong for a parity claim, which is why the `trips` mode renders each layer separately.
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
