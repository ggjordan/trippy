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

- **Quest Browser's WebGPU support may not even apply to a flat (non-XR) page** (found 2026-09-06, researched on paper, not yet measured on a device — see `docs/WEB_VIEWER.md` "Quest assessment"). Meta's own Horizon OS release notes show WebGPU landing only as an **experimental, WebXR-session-scoped** feature: v146.0 (2026-04-21) "Experimental WebGPU and WebXR depth projection support," v149.1 (2026-07-27) WebGPU for space-warp layers, v150.1 (2026-08-28) "Experimental WebGPU foveation support." All three are XR-session features (depth projection, space-warp, foveation), not general 2D-canvas WebGPU. Brush's web viewer is a flat egui/wgpu canvas app — it never opens a WebXR session — so it is exactly the kind of page the release notes do not document support for. This means the Quest question may be "does it load at all" before it is "how many fps," independent of the ~120 GFLOP/eye/frame budget argument above. Confirming this requires an actual Quest device (`navigator.gpu.requestAdapter()` on a plain page, with and without the "webXR experimental features" `chrome://flags` toggle) — not yet done.

- **The fork's own supported-browser claim does not mention Quest at all.** `rust/brush-trips/README.md`: "WebGPU is still an upcoming standard, and as such, only Chrome 134+ on Windows and macOS is currently supported." The hosted demo repeats "Only works on Chrome and Edge. Firefox and Safari are hopefully supported soon." Quest Browser (Chromium-based, but a distinct browser from desktop Chrome/Edge) is not a claimed target. Empirically, on this Mac (2026-09-06), it does render in **Safari** despite not being an officially claimed-supported browser (see `docs/WEB_VIEWER.md` for the verification): WebGPU adapter obtained, wasm app initialised, canvas created, zero JS errors — so "not officially claimed" is not the same as "does not work," and the same could turn out true or false on Quest. Only an on-device check settles it.

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
- **`alpha == 1.0` exactly used to make the float32 CPU reference compositor produce NaN. Fixed
  (fix/raster-nan, 2026-09-06).** `trippy.raster.ref_torch.composite_sorted` clamped alpha to
  `1 - RASTER_ALPHA_MAX_EPS` with `RASTER_ALPHA_MAX_EPS = 1e-12`, which is a no-op in float32
  (`1 - 1e-12 == 1.0f`), so `log1p(-1) = -inf` and the per-segment rebase `exclusive - exclusive[start]`
  became `-inf - -inf`. It bit `tests/test_parity_render.py` when a test used `conf = 1.0`. The clamp is
  now `max(RASTER_ALPHA_MAX_EPS, finfo(dtype).eps)`, i.e. unchanged in float64 and `1 - 2**-23` in
  float32. The Metal path never needed a guard (`blend_fwd` loops `T *= 1 - a`, `blend_bwd` uses
  division-free suffix recurrences), so this only brings the torch twin up to the kernel's semantics.
  `alpha == 1.0` is reachable from a real render, not just from a test: it needs a fragment on an exact
  pixel boundary (bilinear weight 1), `conf == 1`, and `size_px` an exact power of two (layer factor 1) --
  see `tests/test_raster_nan_ref.py::test_alpha_exactly_one_is_reachable_from_a_real_render`.
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

### Fixed (fix/raster-nan, 2026-09-06): the NaN gradient out of the rasteriser backward

**Symptom.** Reproduced on CPU on kk-coherent (`config_smoke.yaml`, `mode = trilinear`,
`max_points = 200000`, `train_factor = 1.0`, 6 epochs, `Trainer._sanitise_gradients` disabled): a
*single* point's `xyz` (3 entries) and `raw_size` (1 entry) become NaN, along with the pose delta of the
frame that point was last rendered in. The image loss stays finite -- a NaN position fails every bounds
comparison, so the point is culled from every subsequent render -- but `_extent_penalty` reduces over
*all* points, so the reported total loss is NaN from then on while the held-out PSNR keeps improving
(13.39 -> 13.47 dB across the NaN).

**Root cause: `project_points` divided by the raw camera-space z.** The reproducing input is a point
whose camera-space z is **exactly 0.0** -- point 964 of 200000, world
`(-1.3989772, 0.60192317, 5.9451413)`, camera-space `(-1.2281361, 3.7178385, 0.0)`, in `IMG_3811.jpg`.
That is not a measure-zero curiosity in float32: z is the third component of `xyz @ R.T + t`, so any
point that drifts onto a camera's principal plane rounds to exactly 0.0, and there are 200k points x 186
views x 6 epochs of chances.

The point is *culled* (`cull_points` requires `depth > znear`), so its incoming `uv` gradient is exactly
zero -- but torch differentiates `n / z` w.r.t. the denominator as `-grad * (n / z / z)` and evaluates
that product for **every** row, culled ones included. At `z == 0` that is `-0 * inf = NaN`; the
numerator's own derivative `grad / z` is `0 / 0`, also NaN. (Even a non-zero z is unsafe once
`n / z / z` overflows float32, i.e. below `|z| ~ 1e-19`.) The NaN lands on that point's `xyz` gradient
and, through `world_to_cam`'s matmul, on all six components of that frame's pose delta. Adam turns a
NaN gradient into a permanently NaN parameter. On the *next* step the now-NaN `xyz` gave a NaN `depth`,
`torch.clamp(nan, min=znear)` kept it NaN, and `size_px = fx * size / nan` handed `raw_size` a NaN
gradient too -- which is why `raw_size` always went NaN exactly one step after `xyz` did.

**Fix.** `trippy.raster.emit.safe_depth(depth, znear) = where(depth > znear, depth, znear)` is now the
divisor of *both* projection divisions (`fx * x / z` and `fx * size / z`). It is bit-identical for every
point that survives the near-plane cull, so the forward is unchanged; `where` rather than `clamp` so a
NaN depth is replaced instead of propagated. Verified as an A/B on the failing run: with and without the
patch the trainer is bit-identical for 693 steps (loss equal to the last float32 bit), at which point
the unpatched run emits `{xyz: 3, pose: 6}` NaN gradients and its loss is NaN from the next step onward,
while the patched run completes 5 epochs / 930 steps with **zero** non-finite gradients in any parameter
(points, sizes, confidences, features, background, poses, U-Net, exposure, response) and no NaN loss.

**The Metal path needed no change.** Everything before compositing is the same vectorised torch code on
both devices (`build_sorted_fragments`), so `safe_depth` fixes MPS too; and `blend_bwd.metal` is
division free by construction (suffix recurrences `U`/`Q`, never TRIPS's `colour_behind / (1 - alpha)`),
so it never had the alpha-side hazard either. `tests/test_raster_nan_metal.py` pins all of it on MPS
(51 tests, GPU job `trippy-raster-nan-gpu-1`, rc 0).

Related and still true: `xform_b.se3_exp` has a zero -- not NaN -- rotation gradient at `phi == 0`.

**Containment stays.** `Trainer._sanitise_gradients` still zeroes non-finite gradient entries before
`optimizer.step()` and records the count as `nonfinite_grads` in `metrics.jsonl`. It is now a backstop
rather than the only defence.

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
## Rust pyramid rasteriser forward pass (v0.4.0, `rust/crates/brush-pyramid`)

- **`layer_bounds` uses the IEEE exponent, not `log2`, so it differs from the Python
  reference within ~1e-6 relative of a power of two.** CubeCL has no `log2` (only
  `ln`), and `ln(x)/ln(2)` in float32 lands on the wrong side of an integer at exact
  powers of two, which would move a point into the wrong pyramid layer. Both Rust
  paths therefore read the exponent field, which is exactly `floor(log2 x)` for a
  positive normal float. `torch.log2` is correctly rounded, and *that rounding* is
  the difference: for `size_px = 7.9999995` the true `log2` is `2.99999991...` but
  the nearest float32 is exactly `3.0`, so Python reports `lower = upper = 3` where
  the exact answer is `(2, 3)`; symmetrically at `16.000002` Python reports `(4, 4)`
  where the exact answer is `(4, 5)`. The Rust answer is the correct one. The band is
  a couple of ulps wide and bounded by `1e-6` relative for every pyramid we render
  (pinned by `factor::tests::exponent_and_log2_bounds_differ_only_next_to_a_power_of_two`),
  so roughly one point in 6 million could land in it; in mode `trips` such a point
  would gain or lose one pyramid layer's worth of fragments. None of the six parity
  fixtures hits it. Note this cannot separate the Rust CPU path from the Rust GPU
  path — they evaluate the same exponent formula — but they can still land on
  opposite sides of the same discontinuity via a differing *input*; see next.

- **A projected size landing exactly on a power of two is a knife edge, and the GPU
  and CPU can fall on opposite sides of it.** `layer_bounds` is floor/ceil of
  `log2(size_px)`, so at `size_px = 2^k` we get `lower == upper` and a `trips` point
  writes both layers at factor 1.0; one ulp *below*, the point straddles, and the
  lower layer's factor collapses to ~1e-7 — under `alpha_min`, so its four fragments
  are dropped. `size_px` is computed as `fx * size / z`, and a shader compiler is
  free to reassociate that or to lower the division to a fast reciprocal, so the two
  paths can legitimately disagree by an ulp there. Measured: an early parity run had
  mode `trips` 4 fragments short of Python for exactly this reason, because the
  fixture parked 40 points on a target of exactly `size_px = 2.0`. The fixture now uses 6.0
  (which clamps to `lower == upper == 2` at 3 layers and is stable under any
  perturbation), and `factor::tests::a_size_on_a_power_of_two_is_a_knife_edge_...`
  pins the behaviour on both sides. In a real scene the probability of a point
  landing within an ulp of a power of two is ~1e-7, and the visible consequence is
  one splat gaining or losing one pyramid layer. Not fixable in general: the rule
  itself is discontinuous, and it is TRIPS's rule.

- ~~**The forward pass returns device buffers, not `burn::Tensor<4>`.**~~ **Fixed
  (feat/brush-unet).** `brush_pyramid::gpu::burn_bridge` now registers the
  zero-input `BindOp` this entry predicted, and `PyramidRender::layer_tensor(l)` /
  `layer_tensors()` hand back NCHW `Tensor<4>`s zero-copy. The raw
  `CubeTensor<WgpuRuntime>` accessors stay, because that *is* what a viewer binds
  (compare `brush-render`'s `resolve_to_cube_float`). Confirmed on the pinned
  revision: `Tensor::from_primitive(CubeTensor)` does not exist and there is no
  readback-free alternative to the custom-operation route — a fusion tensor is a
  handle in a lazily recorded op stream, not a buffer.

- **No backward pass.** Only the forward is ported. Training still runs in
  Python/PyTorch; `blend_bwd` has no Rust twin yet.

- **`depth_sum` is not computed.** The Metal forward also accumulates
  `sum(T * alpha * depth)`, which divided by `1 - t_final` gives an expected-depth
  map. The Rust port writes the feature images, `t_final` and `n_used` only. Adding
  it is one accumulator in `blend_fwd_kernel`.

- **`C` is specialised at compile time to 3, 4 or 8 channels**
  (`RASTER_SUPPORTED_CHANNELS`). The blend kernel takes `num_channels` as a CubeCL
  comptime parameter so the accumulator stays in registers; any other `C` is a clear
  error rather than a slow path.

- **One host synchronisation per render.** The total fragment count has to be read
  back before the fragment buffers can be sized, exactly as `brush-render` reads back
  `num_visible`/`num_intersections`. Everything else stays on device.

- **The npz reader rejects ZIP64 members.** `brush_pyramid::npz` supports stored and
  deflate only, and errors clearly past 4 GiB per array. numpy only emits ZIP64 above
  that, which no trippy point set reaches (5.74M points is ~69 MB of positions).

- **`scripts/test.sh` never builds or runs the GPU path.** The `gpu` feature is off
  by default, so the CubeCL kernels, the GPU parity test and the whole
  Burn/CubeCL/wgpu tree are excluded from every push. They are built through
  `scripts/cpu_heavy.sh` and run through `scripts/gpu_submit.sh`.

## Rust U-Net + tone mapper (v0.4.0, `rust/crates/brush-unet`)

- **Only TRIPS's shipped configuration is implemented.** `activation = elu`,
  `norm = id`, `upsample_mode = bilinear`, `last_act = id`. The `nearest` and
  `deconv` upsample modes and the `bn` norm that `trippy.net` supports for config
  flexibility have no Burn twin. `brush_unet::weights` reads those four keys out of
  the file's `__metadata__` and **refuses to load** anything else, so an unsupported
  config is a loud error at load time rather than a silently different render.

- **No `partial_multi` conv block, and masks are not plumbed.** TRIPS asserts the
  decoder-only network never uses `partial_multi` (Networks.h:1028) and its gated
  blocks pass the validity mask through untouched, so the Rust `GatedBlock` simply
  has no mask argument. Nothing is lost; the API is just narrower than the Python
  one's.

- **The tone mapper is eval-mode only.** `CameraResponseNetImpl`'s "leaky"
  linear / `1/sqrt` extrapolation outside `[0, 1]` is training-only in TRIPS and in
  `trippy.net.camera_model` (`self.training`), so it is not ported. Neither is
  rolling shutter (already off by default and not ported on the Python side either),
  nor `ApplyConstraints` / `ParamLoss`, which only matter while optimising.

- **The response LUT is evaluated by an explicit gather, not `grid_sample`.** With
  `align_corners = true` and `padding_mode = border`, PyTorch clips the *sample
  coordinate* before the two taps are read, which makes the whole operation
  identical to `clamp(x, 0, 1)` followed by a plain lerp between control points
  `floor(s)` and `min(floor(s)+1, P-1)`. That equivalence is exercised by the
  fixture, whose `camera_probe` deliberately runs from below 0 to above 1. If a
  future TRIPS config ever used a different `padding_mode`, this would silently be
  the wrong function.

- **No autodiff, no training.** The Burn modules are inference-only. `Unet` and
  `GatedBlock` are `#[derive(Module)]` types so a record *could* be saved, but the
  loader binds `Param::from_tensor` directly from safetensors and never goes through
  Burn's record machinery — deliberately, so the schema is one table in
  `rust/README.md` rather than a serialisation format.

- **Batch size is fixed at 1.** `NeuralCamera::forward` takes a single `frame`
  index, so a batch would need per-image exposure/white-balance gathers. A viewer
  renders one frame at a time; a batched trainer would not use this crate.

- **f32 only.** The exporter writes float32 and the reader rejects anything else
  (including f16). The `half` crate is therefore not a dependency. A quantised or
  half-precision weight file would need both sides changed together.

- **`num_frames` must cover the frame index.** The tone mapper is indexed by the
  *dataset* image index, exactly as `trippy.render.parity` does it. Rendering a
  novel view (a dolly path) has no exposure/white-balance row of its own; the caller
  has to pick one, and the crate cannot tell that this happened.

- **The horse end-to-end test exercises only part of the tone mapper.** The public
  `checkpoint_horse` learned a real response LUT but left the vignette at exactly
  zero, every white-balance gain at 1.0, and frame 8's exposure at 0.0 EV (the whole
  scene's |EV| maxes out at 0.066). Exposure, white balance and the vignette
  polynomial are therefore only covered by the synthetic fixture, which sets all
  three to deliberately non-trivial values.

- **Per-stage timings are differences of cumulative prefixes, not directly measured
  stages.** Putting a barrier *between* stages inside one timed run does not work on
  this backend: both a one-element readback and a full `sum()` readback reported a
  U-Net cost of 1.3-2.3 ms at 1920x1080, which is impossible — that network is ~82
  GFLOP at this resolution and the machine's GPU peaks near 21.5 TFLOPS, so ~4 ms is
  a hard floor. The work was evidently still landing outside the window being
  measured. `render_frame_full` therefore times three *cumulative prefixes*
  (pyramid; pyramid+U-Net; the whole frame), each from scratch with a single barrier
  at its own end, interleaved round-robin so GPU clock ramp cannot bias one against
  another, and reports the differences. The whole-frame number is measured directly
  and is the one to quote; the per-stage split inherits the noise of two
  measurements and each prefix charges its own barrier's readback.
## Web viewer (`rust/crates/trips-web` + `web/`)

The whole diagnosis, with the exact error strings, is in `docs/WEB_VIEWER.md`.
This is the short list of what the browser build costs and cannot do. As of
this entry it renders all three views, including the U-Net's.

### The U-Net view runs in a browser, but the first frame autotunes for ~20 s

**Fixed since v0.5.0, and the v0.5.0 diagnosis was wrong.** That release
recorded the browser's `network` view as blocked by CubeCL's `read_sync` on the
route from the U-Net's `burn::Tensor<4>` to a bindable buffer, and substituted
`raw level-0`. Neither `burn_bridge::resolve_to_cube_float` nor
`Tensor::into_data_async` calls `read_sync` on these pinned revisions. The trap
was CubeCL's **convolution autotuner**: its roofline bounds generator calls
`cubecl_std::throughput::measure_peak_throughput`, whose own doc comment says
"Native only, panics on WASM". `docs/WEB_VIEWER.md` blocker 4 has the stack and
how it was read.

What is left is the cost of the way round it. `with_bounds` registers no bounds
generator only at `AutotuneLevel::Full`, so that is the level
`brush_pyramid::gpu::disable_autotune_roofline_bounds()` sets, and `Full` means
"benchmark every candidate, no roofline short circuit". **The first frame of a
new convolution shape therefore takes about 20 seconds in the browser**, once
per shape per session; every frame after it is unaffected. `web/trips.js` says
so on the canvas while it happens. There is no cheaper level: `Minimal`,
`Balanced` and `Extensive` all install the generator that traps.

Two smaller consequences:

- The autotune result is not cached across page loads (the persistent cache is
  a filesystem cache), so every reload pays the ~20 s again.
- The kernel autotune picks in the browser is not necessarily the one it picks
  natively, which is part of why the browser's frame matches the native
  reference at 62 dB rather than exactly.

### Two shims in `web/trips.js` compensate for dependency bugs

Neither is a preference; without them the page dies on its first frame in every
browser.

1. **`popErrorScope` is neutralised.** wgpu's WebGPU backend reads a clean pop
   (`null` per spec) into a `js_sys::JsOption`, which only treats `undefined`
   as "none", so it panics `"Unexpected error"` on *every* clean pop — and
   CubeCL wraps every kernel launch in an error scope. Errors are still logged
   to the console and the trace; what is lost is wgpu's own view of validation
   errors, so a bad frame shows as bad pixels rather than as an exception.
2. **`enable subgroups;` is prepended to shaders that need it.** CubeCL's WGSL
   backend emits `subgroupAdd`/`subgroupInclusiveAdd` for `brush-sort`'s radix
   passes without the directive WGSL requires. Masking `Features::SUBGROUP` off
   does *not* avoid it — measured; there is no non-subgroup lowering.

### Safari draws a WRONG image — use Chrome

Safari 26.6.2 gets all the way to producing frames (3.25 fps, a
plausible-looking 1 MB PNG) and **the picture is horizontal stripe noise, not
the horse**. One CubeCL compute shader fails to compile there —
`GPUValidationError: 1 error generated while compiling the shader: 1:0:
Expected 'f16'`, then `createComputePipeline failed` — and because the shim
above removes wgpu's fatal error path, the missing stage silently contributes
nothing and the rest of the pipeline draws garbage.

Chrome 152 renders the same scene correctly (verified against the pixels; the
horse is the public Zenodo scene).

This is why `web/trips.js` prints every WebGPU error **on screen in red** with
"THIS IMAGE IS NOT TRUSTWORTHY" and puts them in the beacon. Neutralising a
fatal error handler is only defensible if the errors stay visible: a viewer
built around telling photographed pixels from invented ones must not quietly
show an invented picture.

Note the contrast with the v0.5.0 groundwork, where Safari ran the *stock*
Brush splat viewer fine — Brush's Gaussian rasteriser and trippy's TRIPS
rasteriser share no kernels.

### The browser is much slower than the Mac app

Measured in Chrome 152 on this M3 Ultra, 1440x810, view 8, release build,
**while a Splats training was running on the same GPU** (so both browser
numbers are lower bounds):

| view | Chrome | native `trips-viewer` |
|---|---|---|
| `network`, `--half-net` equivalent | **1.09 fps** (6 frames / 5.49 s) | 29.46 fps |
| `raw level-0` | **3.32 fps** (17 frames / 5.12 s) | 116.21 fps |

The v0.5.0 entry here blamed the per-frame point upload for most of the gap.
That upload is now gone (`brush_pyramid::gpu::UploadedPoints`, below), and the
browser's `raw level-0` went 2.90 → 3.32 fps while the native number more than
doubled — so on this stack the upload was **not** the browser's main cost. What
is left is unexplained and unmeasured from inside the page: a wasm build has no
`Instant`, `?trace=1` only times whole stages, and Chrome's own profiler is an
interactive tool. Candidates, in no measured order: WGSL compiled by the
browser rather than MSL, the error-scope shim adding a promise round trip per
kernel launch, and WebGPU's own per-dispatch validation.

## Distillation (design B)

- **A distilled splat can only be as good as the checkpoint it was distilled from.**
  `trippy distill` samples the TRIPS network's own output and trains an ordinary 3DGS
  model to reproduce it; it cannot exceed the checkpoint's own quality, and any defect
  the checkpoint still has (including the shade cloud, if the checkpoint hasn't fixed
  it yet) gets baked into the distilled PLY as an ordinary Gaussian, indistinguishable
  from a directly-trained one. Design B is a fallback for viewer compatibility (D2),
  never a quality upgrade over the checkpoint it came from.

- **Distillation re-introduces Gaussian splat artefacts.** This is Jordan's own stated
  worry (docs/SPEC.md D6) about porting a winning design into a Brush fork rather than
  distilling: popping, view-dependent floaters, and the shade-cloud defect itself if
  Gaussians alone cannot represent it well, all become possible again once the design-B
  output is an ordinary Gaussian splat trained by an off-the-shelf 3DGS trainer. The
  audit numbers (`trippy.distill.compare`) rank candidates; only Jordan's viewer verdict
  on the distilled PLY (not the TRIPS checkpoint it came from) settles whether this
  fallback is actually acceptable for a given scene.

- **No pose refinement on interpolated (or anchor) cameras.** `trippy.distill.render_set`
  renders every pose -- anchor and interpolated alike -- at each image's raw COLMAP pose,
  never the checkpoint's own trained per-image pose-refinement delta
  (`trippy.train.params.PoseParams`). This matches the existing convention
  `trippy.render.dolly`/`trippy.render.offpath` already use for arbitrary poses, but it
  means the rendered image set is very slightly mis-registered relative to what the
  checkpoint was actually optimised against, by however much pose refinement moved that
  frame. Not measured to matter in practice (pose-refinement deltas are typically far
  sub-pixel), but not proven negligible either.

- **The honesty guard is a heuristic, not a geometric proof.** `build_distill_camera_plan`
  skips a consecutive anchor pair when its distance exceeds `--max-jump-multiplier` times
  the scene's own median consecutive-pair distance, or when the two images use different
  `camera_id`s. This catches the common failure modes (a registration gap, two separate
  sweeps of the same scene, a lens change) but cannot detect every way two "consecutive"
  registered images might not actually be adjacent along the walked path -- e.g. a capture
  that revisits the same physical location twice at a similar spacing to the rest of the
  walk. `skipped_pairs` in `distill_report.json` is the audit trail; nothing beyond the
  two checks above is applied.

- **Brush has no `--init-ply` flag.** "Init from the TRIPS export ply" (the task's own
  phrasing) is implemented by writing the TRIPS export's point cloud into points3D.txt,
  which Brush's COLMAP dataset loader reads as its initial splat means + SH-DC colours
  (positions + colour only -- no size/opacity/rotation column exists in points3D.txt, and
  Brush initialises those itself). This was verified by reading
  `rust/brush-trips/crates/brush-dataset/src/formats/colmap.rs` (`init.actor.run(...)`
  block), not by a `--help` flag, since neither `brush-cli --help` nor `apps/brush-cli/src/
  lib.rs`'s `Cli`/`TrainStreamConfig` struct expose any point-cloud-initialisation flag at
  all.

- **The distilled PLY's colour is view-independent by choice, not by TRIPS's own limit.**
  `trippy.distill.brush_runner`'s default `--sh-degree 0` means Brush trains flat RGB per
  Gaussian, no higher-order spherical harmonics. A single TRIPS checkpoint distilled at a
  fixed set of poses gives Brush no multi-view specular/view-dependent signal to recover
  with a higher SH degree (every rendered "photo" of a given surface point already carries
  whatever appearance the TRIPS network baked in for that pose), so degree 0 was chosen as
  the honest default; a higher `--sh-degree` is a free CLI override if a future scene's
  distillation shows visible view-dependent effects worth capturing.

## Native Mac viewer (v0.4.0, `rust/crates/trips-viewer`)

### The frame is rendered on the UI thread

`ViewerApp::ui` calls `render_pyramid` and blocks on its one device readback (the
fragment count) before it can even launch the sort. Brush's own splat backbuffer
instead runs the render on an `AsyncMap` actor and paints whatever the last finished
image was, so its UI stays responsive while a frame is in flight. The consequence
here is that the displayed fps *is* the render's fps — there is no queue hiding
work, which makes the readout honest — but a slow frame also freezes the mouse. If
the viewer ever needs to stay smooth on a heavier scene, the fix is Brush's: move
the render to an actor and accept a frame of latency.

### The frame is network-bound, not sort-bound — measured, 2026-09-06

This is the most important thing in this section, because the whole v0.4.0 brief
(and `research/trips-metal.md`'s first Mac timing) assumed the opposite.

At 1920x1080 on the horse bundle, on this M3 Ultra:

- whole frame, exact: **190 ms** (204 ms before `UploadedPoints`)
- the same frame in `raw level-0` or `coverage` view, i.e. **the identical
  rasteriser with the U-Net removed**: **9.8 ms (102 fps)** (21.5 ms / 46.6 fps
  before `UploadedPoints`)

So the pyramid rasteriser — projection, emission, both radix sorts over 10.4 M
fragments, the segment scan and the blend — is **~5 %** of the frame, and the
U-Net plus tone mapper are the other **~95 %**. (It was 11 % / 89 % before the
point upload was taken out of the rasteriser's side of the split; the change
moved the ratio, not the conclusion.) Every rasteriser-side lever below
therefore measured *within run-to-run noise* of the baseline, and the two levers
that move the number are the two that reduce the network's work: fewer pixels
(`render_scale`) and cheaper arithmetic (`half_net`).

~82 GFLOP of 3x3 convolutions in 180 ms is about 450 GFLOP/s on a GPU that peaks
near 21.5 TFLOPS — 2 % of peak. That is the real finding, and it is a
`cubek-convolution`/CubeCL question, not a TRIPS one.

### The performance levers, and what each costs

Everything except `render_scale` and `half_net` is a
`brush_pyramid::params::PyramidParams` field with a serde default equal to the
**exact** pipeline, so an old `params.json` or bundle is unaffected and the parity
tests never see them.

| lever | field | what it changes | exactness |
|---|---|---|---|
| **f16 network** | `Settings::half_net` (viewer) | runs the U-Net's convolutions in f16; the weights are uploaded twice at load, so it is a runtime toggle | **approximate**, and the only arithmetic lever that targets the actual bottleneck |
| **render scale** | viewer only | rasterise *and convolve* at a fraction of the window, upsample in the blit | **approximate**, and a resolution choice rather than an arithmetic one — but it is the lever with the largest effect, because it scales the network quadratically |
| frustum + znear cull | `frustum_cull` | skips the visibility box test | on by default; **exact either way** in `trips` mode, because a point outside the image fails `footprint_fits` at layer 0 anyway. Confirmed: identical PNG. Kept only so the perf table can say what it is worth (nothing). |
| fragment cap | `layer_floor` | `Trips` starts emission at `layer_lower - 1` instead of layer 0 | **approximate**, 41 dB. TRIPS's `compute_point_size_fac` returns 1.0 below `layer_lower`, so this drops full-alpha copies of big points from the finest layers. Buys nothing measurable. |
| f16 features | `feature_store` | uploads and gathers point features as `f16` | **approximate**, 76 dB (essentially free). Buys nothing measurable. |
| packed sort key | `sort` | one 32-bit radix sort over `layer_pixel << d \| quantise(depth)` instead of two (32-bit float depth, then the key) | **approximate**, 34 dB — the worst quality/benefit ratio of the five, because 14 four-bit radix passes become 8 in a stage that is already only a fraction of 11 % of the frame. Depth ties inside a bucket fall back to emission order. |

The three rasteriser levers are kept, documented and default-off rather than
deleted: they are correct, they are the right levers for a scene with far more
fragments than this one, and a measurement that says "this is not where the time
goes" is worth keeping the apparatus that proved it. (The packed sort key *is*
worth 3.8 ms of the rasteriser's then-21.6 ms — 17 % — it is simply invisible
under the network. The rasteriser is now 9.8 ms, so the same 3.8 ms would be a
much larger share of it; that has not been re-measured, and the frame is still
network-bound either way.)

### The point set is uploaded once per bundle (was: every frame)

**Fixed.** `brush_pyramid::gpu::render_pyramid` used to take a host-side
`PointSet` and call `create_tensor_from_slice` on `xyz`, `size`, `conf` and
`feat` on **every frame** — 80 MB of host-to-device traffic per frame, on the
horse, for data that never changes. `gpu::UploadedPoints` is that upload as a
handle: `trips_viewer::Renderer` builds one when the bundle loads and binds it
every frame through `gpu::render_pyramid_uploaded`. The `PointSet` entry points
are unchanged and still work; they upload and delegate.

Worth **12.2 ms of every frame** at any resolution — a fixed cost, so it
matters most where the frame is cheapest (job `trippy-web-unet-gpu-3`, public
horse, 30-frame medians):

| view | before | after |
|---|---|---|
| 1080p `network` exact | 4.93 fps | 5.28 fps |
| 1080p `network --half-net` | 12.45 fps | 14.64 fps |
| 1440x810 `network --half-net` (shipped) | 21.72 fps | **29.46 fps** |
| 1080p `raw level-0` | 45.40 fps | **102.31 fps** |
| 1440x810 `raw level-0` | 46.17 fps | **116.21 fps** |

The screenshot the new binary writes is byte-identical to the old one.

What remains: **the f16-features lever re-uploads.** `FeatureStore::F16` is a
different buffer, not a different binding, so toggling `--fp16` at runtime
rebuilds the handle (one 40 MB upload, once per toggle) and the host copy of
the point set is kept alive for exactly that reason — the native viewer's
panel has an "f16 features" checkbox, so this is a real path, not a
hypothetical one. The browser front end does not expose the lever, so on wasm
that host copy is 80 MB of dead weight in the heap.

**And the old "stage 1 reads 178 ms" number in this file was an artefact**, not
a measurement of the upload: `--profile` was run in `network` mode, where the
first stage's forced device sync drains the previous warm-up frame's still
queued U-Net. Profiled in `raw` mode the same stage read 12.1 ms before the
change and 0.4 ms after it. `StageTimings` now carries a separate `upload_ms`
so the cost cannot hide inside stage 1 again.

### Shipped configuration

`scripts/open_mac_viewer.sh` writes `--half-net --scale 0.75` into the launcher:
**34.0 ms, 29.5 fps** in a 1080p window (45.3 ms / 22.1 fps before
`UploadedPoints`). `--half-net` is free (59.8 dB); the 31.5 dB
against the 1080p reference is entirely the 0.75 resolution, and `-`/`=` change it
live. `--packed-sort` and `--cap-fragments` are deliberately **not** shipped: 1.7 ms
for 0.75 dB, and nothing at all for 42 dB, respectively.

Two things about the packed key are worth knowing before trusting it:

- The depth quantisation is linear in a **fast `log2`** — the float's bit pattern read
  as an integer and scaled — which is monotone by construction and computed by the
  identical expression on host and device (`gpu::fast_log2` /
  `kernels::fast_log2`). It is not a call to `log2`, which CubeCL does not have.
- The range it quantises over comes from the **point cloud's bounding box**, not from
  the frame's actual depths, because measuring the latter would need a device
  reduction and a readback per frame — the very thing the lever exists to remove. A
  camera much further out than the scene's diameter therefore loses depth resolution.

### The CPU reference does not implement two of them

`render_pyramid_cpu` **returns an error** for `SortMode::PackedKey` and
`FeatureStore::F16` rather than silently rendering something else. They are GPU
storage and ordering strategies with no meaning for a `sort_by_key` on a `Vec`.
`layer_floor` *is* implemented on both paths, because it changes which fragments
exist and the reference has to be able to model that.

### `trips-viewer` is not on the push path

`scripts/build.sh` and `scripts/test.sh` still only compile `brush-pyramid` and
`brush-unet` without the `gpu` feature (ADR-0005's whole point). `trips-viewer` pulls
Burn, CubeCL, wgpu, egui and eframe, so its 13 unit tests run only on demand:

```bash
bash scripts/cpu_heavy.sh trips-viewer-test -- bash -c \
  'cd rust && cargo test -p trips-viewer --release'
```

They are pure-logic tests (manifest parsing, the camera controller's orthonormality,
the shader/enum agreement) and need no GPU — but they do need the whole tree built.

### What `--screenshot` does *not* cover

The offscreen PNG is produced by `Renderer::render_to_host`, which reads the tone
mapper's tensor back with `into_data_async` — a path that understands strides. The
**window** instead binds that tensor's raw buffer into a wgpu render pass and lets
`blit.wgsl` index it. So a PSNR of 115 dB against `render_frame_full` proves the
whole forward pass and says nothing at all about whether the blit's indexing is
right; a transposed or channel-swapped display would score exactly the same.

Three things stand in for the check nobody can run:

1. The shader reads the network's output in its **native planar NCHW layout**
   rather than a permuted one, so there is no re-layout to get wrong. An earlier
   version did `permute` + `reshape` into channel-last, which is correct only if
   the backend materialises the copy rather than re-describing the strides —
   unverifiable from here, and it was removed for that reason.
2. `Renderer::render` **errors** rather than displaying anything if the resolved
   buffer is not contiguous.
3. `blit::tests::shader_codes_match_the_wgsl_constants` asserts the Rust `ViewMode`
   discriminants equal the WGSL `MODE_*` constants, so adding a view mode cannot
   silently render as an older one.

That is honest but not complete: the remaining gap is a render-to-texture readback
of the actual egui pass, which would need a headless eframe harness.

### Still missing

- **No backward pass**, so nothing can be trained from the viewer. Same gap as the
  rest of v0.4.0.
- **Only `trips` mode is exercised.** The viewer renders whatever `mode` the bundle's
  manifest names, but every bundle written so far says `trips`.
- **One frame index for the tone mapper.** The per-image exposure and white balance
  come from the *reference view* the camera was last pinned or snapped to, so flying
  far from that view keeps its exposure. There is no principled alternative — a
  free-flown camera has no image of its own — but it means the colour grade is a
  choice, not a measurement, and it changes when you jump views.
- **No `.ply` or splat rendering.** That is Brush's binary, which this does not touch.
