# Architecture: trippy rendering and training pipeline

## Module overview

```
trippy/
├── scene/       COLMAP i/o, dataset loading, train/val/test splits
├── geom/        transforms (xform_a numpy, xform_b torch), camera models
├── points/      Gaussian PLY source, monocular depth source, union, kNN size estimation
├── raster/      pyramid rasteriser: emit.py (project + layer select + 2x2 splat),
│                sort.py (order by layer/pixel/depth, segment offsets),
│                metal_src/blend_fwd.metal + metal_lib.py (Metal compositing),
│                pyramid.py (device dispatch), ref_numpy.py / ref_torch.py
├── net/         U-Net decoder, gated ELU convolutions, perceptual loss
├── train/       trainer loop, config, eval harness, export to 3DGS PLY
├── render/      dolly camera paths, off-path rendering, video export, honesty sheets
└── cli.py       command-line interface, smoke tests
```

## Forward pass data flow

```
COLMAP poses + intrinsics → colmap_io.load()
                          ↓
          GaussianPlySource / MonoDepthSource / UnionSource
                          ↓
    PyTorch tensor: positions [N, 3], sizes [N] (world units, post-softplus),
                    features [N, C], confidence [N] (post-sigmoid),
                    camera pose SE(3) delta
                          ↓
        raster.emit.project_points()   (xform_b: pose + K → uv [N,2], depth [N],
                                        size_px = fx · size / z)
                          ↓
        raster.emit.cull_points()      (z ≤ znear, or footprint fully off the
                                        padded coarsest layer)
                          ↓
        raster.emit.emit_fragments()   (per selected layer: uv_l = uv / 2^l,
                                        2×2 bilinear footprint anchored on pixel
                                        centres, alpha = β · conf · layer_factor;
                                        drop out-of-bounds and alpha < ALPHA_MIN,
                                        never clamp)
                          ↓
        raster.sort.sort_fragments()   (one int64 argsort on
                                        layer_pixel · 2^32 + float32_bits(depth);
                                        fallback: two stable sorts)
                          ↓
        raster.sort.segment_offsets()  (searchsorted or bincount+cumsum, into a
                                        flat layer-pixel space of length
                                        Σ_l h_l·w_l, offsets array length P+1)
                          ↓
              blend_fwd Metal kernel (metal_src/blend_fwd.metal)
            (one thread per layer-pixel, front-to-back, cap 16 fragments,
             stop at T < 0.001; writes out [P,C], T_final [P], n_used [P],
             depth_sum [P]; CPU dispatches to raster.ref_torch instead)
                          ↓
              background in torch: out += T_final · bg
                          ↓
            U-Net decoder (PyTorch, torch.nn.Conv2d)
                          ↓
              Tone mapper (per-image exposure + response LUT)
                          ↓
         Loss (L1 + SSIM + LPIPS/VGG16 perceptual)
```

## Backward pass data flow

```
Loss gradients (autograd)
         ↓
    U-Net backward (autograd)
         ↓
Gradients ready for blend_fwd inputs
         ↓
   blend_bwd Metal kernel
(writes d_alpha per fragment, d_feature per fragment)
         ↓
  index_add_ to points (reduce per-fragment grads to point indices)
         ↓
Autograd backprop to positions, sizes, colours, confidence, pose
(PyTorch handles these for free once per-point gradients arrive)
```

## Fragment emission and the two layer-selection modes

`render_pyramid(..., mode=...)` picks how a point is spread over the pyramid. Both modes are implemented and tested; neither is "the" TRIPS behaviour on its own.

| mode | layers written | layer factor | fragments per point | corresponds to |
|---|---|---|---|---|
| `"trilinear"` | `[lower, upper]` from `floor/ceil(log2(size_px))`, clamped to `[0, L-1]` | TRIPS's `compute_point_size_fac` (see docs/GEOMETRY.md) | ≤ 2 × 4 = **8** | TRIPS's `use_layer_point_size=true` path, whose emission kernel `CollectTiled2Pointsize` writes exactly those two layers (`RenderForward.cu:2296-2360`) |
| `"broadcast"` | **every** layer | 1 everywhere | L × 4 = **20** at L=5 | TRIPS's shipped default (`use_layer_point_size=false`) |

The distinction matters and was previously misdocumented here (docs/TRIPS_REFERENCE.md §10.1). `use_layer_point_size` has no `SAIGA_PARAM` entry, so **no shipped TRIPS `.ini` can turn the trilinear path on**: every TRIPS checkpoint in the wild was trained in broadcast mode, with each point splatted into all five layers at full alpha and the layers fused only inside the U-Net. `"trilinear"` is the mode the *paper* describes and the one that makes the per-point size parameter do any work, so trippy implements both and defaults to `"trilinear"`.

Sizing consequence: fragment buffers must be sized for `4 · L` fragments per point in broadcast mode, not `4 · 2` (docs/TRIPS_REFERENCE.md §11).

## Core principle: No atomics anywhere — a deliberate redesign, not a port

**TRIPS uses `atomicAdd` extensively** — for per-pixel list counting and slot allocation in `CountTiled`/`CollectTiled2`, and for every gradient reduction in `RenderBackward.cu`. We do **not**, by design: 64-bit atomics do not compile in Metal via `torch.mps.compile_shader`, so the atomic list-building step is replaced by a global sort. Nothing below describes TRIPS's own algorithm; TRIPS's fragment counts and list caps do not carry over to an atomic-free formulation without re-deriving them (docs/TRIPS_REFERENCE.md §10.3).

1. **Sorting by (layer, pixel, depth)** happens once, on CPU or GPU, before any parallel writes.
2. **Parallel read-only access** in `blend_fwd`: each thread reads a contiguous segment of sorted fragments for its (layer, pixel).
3. **Segment offsets** (computed via prefix sum before the kernel) tell each thread where to read and how many fragments to process.
4. **Per-fragment gradients** in `blend_bwd` are written without conflicts: fragment i is owned by exactly one thread.
5. **Reduction to points** happens in PyTorch via `index_add_`, not in Metal.

This design avoids any need for atomic operations and runs efficiently on both Metal and future GPU targets.

## Memory notes for 96 GB unified

- **Unified memory**: all tensors live in one pool. No copy overhead between GPU and CPU.
- **Peak consumption**: forward pass ≈ point tensors + intermediate fragments + U-Net activations + target image.
  - 7.36M points × ~32 bytes (pos, size, colour, conf, pose delta) ≈ 235 MB
  - Fragments: ≤8 per point, ≤400² crop size, 16-cap per pixel ≈ 200 MB per batch
  - U-Net: ~130k params, 5 levels, largest activation ~512×512×32 ≈ 8 MB
  - Target: 2048×2048×3 ≈ 48 MB (two floats: RGB + α)
  - **Total ≈ 500 MB per batch:** well within 96 GB.
- **GPU queue enforces 28 GB guard** to prevent OOM on other jobs.

## Later: Brush fork mapping (v0.4.0 onward)

When TRIPS training is complete and we port the design to Rust/Burn/CubeCL:

```
rust/brush-trips/
├── crates/brush-pyramid/
│   └── src/
│       ├── lib.rs          emit_fragments, radix argsort ×2, prefix_sum (Burn)
│       └── kernels/        blend_fwd, blend_bwd (CubeCL, wgpu)
├── crates/brush-unet/
│   └── src/
│       ├── lib.rs          U-Net conv2d inference (Burn)
│       └── weights.rs      safetensors loader (like lpips-convert)
└── apps/brush-app/
    └── src/
        └── ui/
            └── splat_backbuffer.rs   viewer hook-in (pyramid rasteriser per frame)
```

The viewer loads a trained `.ply` and runs the full forward pass (emit → sort → blend_fwd → U-Net) every frame without leaving the browser or the native app.

## Validation strategy

CPU pytest (before any GPU job):

1. **Transform agreement**: `xform_a` (numpy) vs `xform_b` (torch) produce identical projects; reprojection of COLMAP `points3D` matches stored keypoints (~1 px, depth positive).
2. **Geometry** (`tests/test_raster_bounds.py`): no emitted fragment lands outside its own pyramid layer; a footprint straddling the border keeps only its in-bounds corners; a point a few pixels off screen still draws in the coarse layers (so the cull is conservative); odd-sized images keep their last row/column.
3. **Reference pair** (`tests/test_raster_ref.py`): `ref_numpy` (numpy, xform_a, explicit per-point/per-layer/per-corner loops) vs `ref_torch` (torch float64, xform_b, vectorised segment prefix sums) agree to <1e-6 on a 32×32 scene containing a pixel stacked past the 16-fragment cap and points on every border, in both modes.
4. **Sort equivalence** (`tests/test_raster_sort.py`): the composite int64 key and the two-stable-sort fallback produce identical permutations, including on depth ties; both segment-offset methods agree.
5. **Metal kernel vs torch reference** (`tests/test_raster_metal.py`, marked `gpu`): 32×32 synthetic scene, float32 Metal vs float64 CPU, max abs diff <1e-4.
6. **Gradcheck** (v0.2.0): float64 on CPU via `ref_torch`, then Metal grads vs reference <1e-3 (gradient magnitudes must match). The Metal forward path is not differentiable until `blend_bwd` lands — see docs/LIMITATIONS.md.
7. **U-Net shape**: odd-size crops, verify autograd.

If any test fails, training is not submitted.

## net/ -- U-Net, neural camera, losses (feat/net, 2026-09-05)

`trippy/net/` ports TRIPS's default render/tone-mapping network, verified against
`third_party/TRIPS/src/lib/models/{Networks,NeuralCamera}.{h,cpp}` and, for the one piece not vendored in
this checkout (Saiga's gated conv block and SSIM), against the public Saiga source fetched over the
network. See `docs/TRIPS_REFERENCE.md` Sec. 5-7, 9 and `docs/LIMITATIONS.md`'s "net/" section for full
citations; this is a summary for the architecture overview.

- **`trippy/net/gated.py`**: `GatedConvBlock` -- two independent 3x3 convs (`feature_transform` ->
  activation, `mask_transform` -> sigmoid) multiplied together then normed; exact Saiga formula, not a
  guess (see LIMITATIONS.md).
- **`trippy/net/unet.py`**: `MultiScaleUnet2dDecOnlySmallFixed` -- a decoder-only U-Net that consumes a
  finest-first image pyramid (`inputs[0]` = full resolution) with no encoder/downsample path at all (the
  rasteriser itself produces every pyramid level directly). At the *shipped default* config
  (`train_normalnet.ini`: `filters=32, num_input_channels=4, num_layers=5`), the exact, hand-verified
  parameter count is **59,675** (corrects an earlier unverified "~130k" estimate in this file's memory
  section below). The publicly released Tanks & Temples checkpoints were actually trained with
  `num_layers=8` (101,291 params) -- see `docs/LIMITATIONS.md` for how that was discovered (a real
  checkpoint's shapes matched trippy's from-scratch port tensor-for-tensor once `num_layers` was corrected).
  Odd input resolutions are centre-cropped to a multiple of `2**(num_layers-1)`, a documented, safe
  generalization of TRIPS's own `CombineBridge` (see LIMITATIONS.md).
- **`trippy/net/camera_model.py`**: `NeuralCamera` -- per-image exposure (`x * 2**-ev`), per-image white
  balance (green fixed to 1), a radial vignette (zero-init, so a no-op until trained), and a 25-point
  learned response-curve LUT (init to a gamma-1/2.2 curve, applied via `grid_sample`). Rolling shutter is
  not ported (off by default in TRIPS).
- **`trippy/net/losses.py`**: `TripsLoss` combines L1 + SSIM (Saiga's real 5x5-window formula, fetched and
  verified, not the generic 11x11 window) + a `lpips.LPIPS(net='vgg')` stand-in for TRIPS's un-portable
  Caffe VGG19 perceptual loss + an optional `lpips.LPIPS(net='alex')` term (Saiga's own verified choice for
  its separate, off-by-default `loss_lpips`), with a validity mask honoured by all four terms.
- **`trippy/net/checkpoint.py`**: `try_load_trips_network` -- best-effort loader for TRIPS's
  `render_net.pth` files. Tries `torch.jit.load` (which, in practice, does succeed at reading the named
  tensors -- see `docs/TRIPS_REFERENCE.md` Sec. 9a for the correction to this doc's earlier assumption)
  then `torch.load(weights_only=False)` as a fallback, and assigns into a target module by shape-matched
  registration order.

The memory-notes section above ("U-Net: ~130k params, 5 levels") predates this verification and is now
known to undercount the "params" figure's precision (true default is 59,675, not "~130k"); the "5 levels"
figure is correct for the shipped `train_normalnet.ini` default (though not for the released checkpoints,
which use 8).
