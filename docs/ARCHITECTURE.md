# Architecture: trippy rendering and training pipeline

## Module overview

```
trippy/
├── scene/       COLMAP i/o, dataset loading, train/val/test splits
├── geom/        transforms (xform_a numpy, xform_b torch), camera models
├── points/      Gaussian PLY source, monocular depth source, union, kNN size estimation
├── raster/      Metal rasteriser (blend_fwd/blend_bwd kernels), pyramid.py autograd.Function
│                numpy and torch reference implementations, tests
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
    PyTorch tensor: positions [N, 3], sizes [N, 3], colours [N, 3],
                    confidence [N, 1], camera pose SE(3) delta
                          ↓
                    xform_b.project()     (torch: poses + intrinsics → 2D locations)
                          ↓
                emit_fragments()          (PyTorch: emit ≤8 fragments per point,
                                            2 pyramid levels × 2×2 bilinear weights,
                                            alpha = bilinear weight × sigmoid(conf)
                                            × layer_factor; drop out-of-bounds,
                                            never clamp)
                          ↓
              int64 argsort by (layer, pixel, depth)
            (fallback: two stable 32-bit sorts)
                          ↓
            segment offsets per (layer, pixel)
                          ↓
                blend_fwd Metal kernel
            (one thread per layer-pixel, front-to-back,
             cap 16 fragments, accumulate over, writes features + T_final)
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

## Core principle: No atomics anywhere

**64-bit atomics do not compile in Metal via `torch.mps.compile_shader`.** TRIPS's reference implementation uses atomic depth/id packing. We avoid atomics entirely:

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
2. **Geometry**: padding test (no fragment outside crop); pyramid level selection agrees between reference and Metal.
3. **Metal kernel vs numpy reference**: 32×32 synthetic scene, compare `blend_fwd` output <1e-5.
4. **Gradcheck**: float64 on CPU via `ref_torch`, then Metal grads vs reference <1e-3 (gradient magnitudes must match).
5. **U-Net shape**: odd-size crops, verify autograd.

If any test fails, training is not submitted.
