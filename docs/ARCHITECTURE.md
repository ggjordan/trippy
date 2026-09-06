# Architecture: trippy rendering and training pipeline

## Module overview

```
trippy/
├── scene/       COLMAP i/o, dataset loading, train/val/test splits
├── geom/        transforms (xform_a numpy, xform_b torch), camera models
├── points/      Gaussian PLY source, monocular depth source, union, kNN size estimation
├── raster/      pyramid rasteriser: emit.py (project + layer select + 2x2 splat),
│                sort.py (order by layer/pixel/depth, segment offsets),
│                metal_src/blend_{fwd,bwd}.metal + metal_lib.py (Metal compositing
│                and its gradient), blend_autograd.py (torch.autograd.Function +
│                index_add_ reduction), pyramid.py (device dispatch),
│                ref_numpy.py / ref_torch.py
├── net/         U-Net decoder, gated ELU convolutions, perceptual loss
├── hybrid/      Gaussian-splat renders as network input: render_splat_views (batch
│                renderer via Splats' gsrender), dataset_c/train_c/config_c (design C,
│                render->photo only), config_a/gaussian_input/gsrender_live (design A,
│                render + TRIPS pyramid together inside train/'s own Trainer)
├── train/       trainer loop, config, eval harness, export to 3DGS PLY
├── render/      dolly camera paths, off-path rendering, video export, honesty sheets,
│                bundle.py (the `trippy-bundle-1` writer) and bundle_render.py
│                (the Python twin of the Rust viewer, for numeric parity)
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
grad_out [P, C]  and  grad_T_final [P]   (the latter is non-zero exactly when
         ↓                                a background was composited)
   blend_bwd Metal kernel (metal_src/blend_bwd.metal)
   one thread per layer-pixel, same segment layout as blend_fwd,
   writes d_alpha [F] and d_feat [F, C] — per FRAGMENT, no atomics
         ↓
  grad_feat = zeros(N, C).index_add_(0, point_id_sorted, d_feat)   (torch)
         ↓
  d_alpha flows back through emission in ordinary autograd:
  alpha = bilinear_weight(uv) · conf · layer_factor(size_px)
         ↓                  ↓            ↓
        uv               conf         size_px = fx · size / z
         ↓                             ↓
   xyz, pose_delta                 size, xyz, pose_delta
```

`trippy/raster/blend_autograd.py` is the only glue: `BlendFunction` (a
`torch.autograd.Function`) calls `blend_fwd` in forward, saves
`(alpha, feat, offsets, point_id, n_used)`, and calls `blend_bwd` plus the one
`index_add_` in backward. `blend_fragments()` dispatches on device — MPS to the
kernels, everything else to `ref_torch.composite_sorted`, which is plain
differentiable torch — so `pyramid.py` has a single code path.

### The compositing gradient, and why it needs no epsilon

Forward, per layer-pixel, over the composited prefix `i = 0 … n-1`
(`n = n_used`, set by the 16-fragment cap and the transmittance cutoff):

```
T_0 = 1        T_{i+1} = T_i · (1 - a_i)        out = Σ_i T_i · a_i · f_i
T_final = T_n                     (the coverage/honesty map; background uses it)
```

The three derivatives the kernel writes:

```
d out     / d f_i[c] = T_i · a_i                       →  d_feat[i][c]
d out     / d a_i    = T_i · (f_i - U_i)               ⎫
d T_final / d a_i    = -T_i · Q_i                      ⎭→  d_alpha[i]

U_i = Σ_{j>i} ( Π_{i<k<j} (1 - a_k) ) · a_j · f_j     "colour behind i,
                                                       relative to T_{i+1}"
Q_i = Π_{i<k<n} (1 - a_k)                             "transmittance behind i"
```

`U` and `Q` are computed by a single back-to-front sweep with two recurrences,
seeded at `U_{n-1} = 0`, `Q_{n-1} = 1`:

```
U_{i-1} = a_i · f_i + (1 - a_i) · U_i
Q_{i-1} = (1 - a_i) · Q_i
```

**This is the design decision worth stating plainly.** The textbook form of the
same gradient — the one TRIPS uses — is `d out / d a_i = T_i · f_i - B_i / (1 - a_i)`
with `B_i` the absolute accumulated colour behind fragment `i`
(`RenderBackward.cu:284-301`, `dem = 1 / (1 - alpha_val[j] + 1e-9)`). Both forms
are algebraically identical, because `B_i = T_i · (1 - a_i) · U_i`. But the
divided form is `0 / 0` in the limit `a_i → 1` and needs an epsilon; the
epsilon then makes the gradient *wrong* (it returns ~0 where the true limit is
`T_i · U_i`, which is finite and generally non-zero), and it loses precision for
any large alpha. The suffix recurrences carry the `(1 - a_i)` factor forward
instead of dividing it out, so they are exact for every `a_i ∈ [0, 1]`, need no
tuning constant, and cost one fused multiply-add per channel per fragment.
`tests/test_raster_bwd_src.py` asserts the kernel body contains no `/` at all;
`tests/test_raster_bwd_ref.py` checks the `a_i = 1` case against the closed form.

The kernel replays `T_i` in a first pass and stashes it in a thread-private
`float[MAX_FRAGS]` array (16 floats), rather than saving a `[F]` transmittance
buffer from the forward. Recomputation is cheaper than the memory traffic here,
and it uses the identical `T *= (1 - a)` recurrence in the identical float32, so
forward and backward agree bit-for-bit.

### Which prefix the backward differentiates

The backward is handed the forward's `n_used`; it never re-derives the stopping
point from `T_CUTOFF`. Forward and backward therefore agree by construction
rather than by two implementations of the same rule agreeing numerically.
Fragments the forward skipped keep the zero the host allocated — they did not
influence the output, so their derivative is zero.

### What carries a gradient, and what does not

| quantity | gradient | why |
|---|---|---|
| `feat` (N, C) | yes | `index_add_` of `d_feat` onto point ids |
| `conf` (N,) | yes | linear factor of `alpha` |
| `xyz` (N, 3) | yes | `uv` (bilinear weight) and `z` (`size_px`) |
| `size` (N,) | yes in `trips` and `trilinear` mode; **`None` in `broadcast`** | the layer factor is 1 everywhere in broadcast, so size feeds nothing (docs/TRIPS_REFERENCE.md §10.1) |
| `bg` (C,) | yes | `out += T_final · bg`, ordinary torch |
| `pose_delta` (6,) | yes | `se3_exp(delta) @ [R\|t]`, left-multiplicative (`trippy.geom.xform_b.compose`); see docs/LIMITATIONS.md for the zero-initialisation trap |
| `K` (intrinsics) | no | out of scope; TRIPS computes it, we treat `K` as constant |
| fragment ordering | no | discrete, piecewise constant — as in TRIPS |
| `aux["depth_sum"]` | CPU only | the kernel has no `d_depth` output (docs/LIMITATIONS.md) |
| `aux["n_used"]` | no | integer count |

Consequence: the render is **piecewise** smooth. Culling, the per-corner bounds
test, `floor()` on the footprint base pixel, the `alpha ≥ ALPHA_MIN` drop,
`floor/ceil(log2(size_px))`, the fragment cap and the transmittance cutoff are
all switches. `tests/test_raster_bwd_scenes.py` is therefore a hand-built
fixture that sits ≥ 0.05 away from every one of those switches (≈ 5·10⁴
gradcheck steps), not a random scene.

### The two guards that keep the backward finite

The backward has exactly two places where a *degenerate* input — not a wrong
one — used to produce NaN. Both are now guarded, and both guards are
forward-neutral: they change no pixel of any render.

**1. Never divide by the raw camera-space z** (`trippy.raster.emit.safe_depth`).
`project_points` divides twice by depth: `uv = fx·x/z + cx` and
`size_px = fx·size/z`. torch differentiates `n / z` w.r.t. the denominator as
`-grad · (n / z / z)` and evaluates that product for **every** row of the batch,
including rows whose upstream gradient is exactly zero because the point was
culled. A point whose camera-space z is exactly `0.0` therefore contributes
`-0 · inf = NaN` to its own `xyz` gradient and, through `world_to_cam`, to the
frame's pose delta — from a point that draws nothing. `z == 0` is ordinary in
float32 (it is the third component of `xyz @ R.T + t`; any point on a camera's
principal plane rounds to it), and it cost a real training run one point and one
pose (docs/LIMITATIONS.md). Both divisions now use
`safe_depth(z, znear) = where(z > znear, z, znear)`, which is bit-identical for
every point `cull_points` keeps (`z > znear`) and finite for every point it
drops. `where` and not `clamp`, because `clamp` propagates NaN: that is how a
NaN `xyz` used to poison `raw_size` on the following step.

**2. The alpha clamp must be resolvable in the compute dtype**
(`trippy.raster.ref_torch.composite_sorted`). The vectorised torch compositor
takes `log1p(-alpha)` and rebases per segment, so `alpha == 1` gives
`-inf - (-inf) = NaN`. The clamp is `1 - max(RASTER_ALPHA_MAX_EPS,
finfo(dtype).eps)`: the constant 1e-12 alone rounds back to 1.0 in float32 and
guarded nothing there. The **Metal path has never needed this**: `blend_fwd`
loops `T *= (1 - a)` sequentially and `blend_bwd`'s suffix recurrences are
division free (above), so `alpha == 1` is an ordinary value on the GPU. The
guard exists only to give the torch twin the kernel's semantics.

`tests/test_raster_nan_ref.py` pins both on CPU in float32 and float64 (zero
depth, depths at/inside/behind the near plane, a fragment on an exact pixel
boundary, `size_px` an exact power of two, `size_px` 0, alpha 0 and 1);
`tests/test_raster_nan_metal.py` pins the same set on MPS and diffs it against
the float64 reference.

## Fragment emission: three layer-selection modes and two conventions

`render_pyramid(..., mode=..., pixel_center=..., pyramid_halving=...)` decides how a point is spread over the pyramid. All three modes are ports of real TRIPS code paths (`trippy.constants.RASTER_MODES`); the full derivation, with the exact `compute_point_size_fac` formula and its `path:line` citations, is in **docs/GEOMETRY.md** "Pyramid level selection and the layer factor".

| mode | layers written | layer factor | fragments per point | corresponds to |
|---|---|---|---|---|
| **`"trips"` (default)** | `0 … layer_higher` inclusive, `layer_higher = clamp(ceil(log2 size_px), 0, L-1)` | TRIPS's `compute_point_size_fac`: **1.0** on every layer below `layer_lower`, then the two interpolation weights | ≤ L × 4 = **20** at L=5 | `CountAndCollectTiled` / `RenderFast16` (`RenderForward.cu:168-368`, `:3511-3517`), selected by `use_layer_point_size = !fix_point_size = true` (`Settings.cpp:39`) |
| `"trilinear"` | `[lower, upper]` from `floor/ceil(log2(size_px))`, clamped to `[0, L-1]` | same factor | ≤ 2 × 4 = **8** | `CollectTiled2Pointsize` (`RenderForward.cu:2296-2360`), the `combine_lists = true` branch |
| `"broadcast"` | **every** layer | 1 everywhere | L × 4 = **20** at L=5 | `use_layer_point_size = false` |

**`"trips"` is what every published TRIPS checkpoint actually renders with**, and it is trippy's training default (`trippy.constants.TRAIN_DEFAULT_MODE`). This corrects two earlier readings of the source recorded here and in docs/TRIPS_REFERENCE.md §10.1: `use_layer_point_size` has no `SAIGA_PARAM` entry, but it is *derived* from one (`!optimizer_params.fix_point_size`), and `fix_point_size = false` in every published `params.ini`. The three modes score **22.27 / 21.47 / 15.14 dB** on the same three held-out `tt_horse` frames from the authors' own checkpoint (`experiments/EXP-0002-horse-parity/README.md`), so the choice is worth 7 dB, not a detail.

`"trips"` also carries TRIPS's own footprint gate — all four corners must be in bounds, and failing it at layer `l` `break`s out of every coarser layer — which the other two modes do not (docs/GEOMETRY.md).

Two orthogonal convention options, both defaulting to trippy's own behaviour:

| option | values | default | when to change it |
|---|---|---|---|
| `pixel_center` | `"half"` (centre of pixel `i` at `i + 0.5`) / `"integer"` (at `i`) | `"half"` | `"integer"` only to reproduce a TRIPS checkpoint bit-for-bit. trippy's scenes are undistorted in the `"half"` convention, so training on `"integer"` would render half a pixel off its own ground truth. |
| `pyramid_halving` | `"ceil"` / `"floor"` | `"ceil"` | `"ceil"` is TRIPS's own branch for every published network; `"floor"` is its `MultiScaleUnet2d` branch. |

Sizing consequence: fragment buffers must be sized for `4 · L` fragments per point in `"trips"` and `"broadcast"` modes, not `4 · 2` (docs/TRIPS_REFERENCE.md §11).

### Emission cost: mode `trips` vs `broadcast` on MPS (perf/trips-mode, 2026-09-06)

`emit_fragments` has two implementations, selected by `impl=` and proven
bit-identical (same tensors, same order) by `tests/test_raster_emit_impl.py`:

- **`"loop"`** — the original: a Python `for layer in range(L)` that selects the
  points writing into that layer, gathers them, and appends per-layer fragment
  lists that are `torch.cat`ed at the end. It is the readable statement of the
  rule and is kept as the A/B baseline for `tools/profile_raster.py`.
- **`"vectorised"`** (the default) — compact the culled points once, then do all
  L layers as one `(L, M, …)` block, layer-major so the flattened order is
  identical, and compact once more per pass.

**The measured cause is not arithmetic — it is the number of distinct tensor
shapes per render.** Three facts settle it (`tools/profile_raster.py`, 5.74M
kk-coherent points, 384-px crop, L=5, jobs `trippy-trips-perf-1` and `-3`):

1. Mode `trips` emits **fewer** fragments than `broadcast`, not more: 9.02M vs
   24.61M from the same 1.43M culled points (6.3 vs 17.2 fragments per point).
   A fragment-count explosion was the obvious hypothesis and it is wrong.
2. On CPU, where a new shape costs nothing, `trips` is *cheaper* than
   `broadcast` end to end (1.29 s vs 3.62 s for one crop's forward+backward at
   the same sizes), and the two implementations are within 8% of each other in
   every mode. The effect is MPS-only.
3. MPS charges **6-9x** for an elementwise kernel on a tensor shape the process
   has not used before — 1.90 ms vs 0.32 ms for `floor(x * 0.5)` on 1.43M rows
   (`--shape-probe`; 3.19 vs 0.36 ms on the earlier, more contended run). And
   the loop implementation produces a *different number of shapes per mode*: in
   `"broadcast"` all five layers select the same rows, so one shape serves the
   whole render, while in `"trips"`/`"trilinear"` each layer selects a different
   count, and every one of those counts moves each step as the crop moves.

That shows up directly as the gap between a frozen camera and a moving one.
Rasteriser forward+backward on MPS, milliseconds, medians of 5
(`trippy-trips-perf-3`):

| mode | impl | fragments | emit | sort | blend fwd | backward | fixed camera | **moving camera** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| broadcast | loop | 24.61M | 35.8 | 20.9 | 1.04 | 42.2 | 118.5 | 119.6 |
| broadcast | **vectorised** | 24.61M | 34.9 | 20.8 | 0.96 | 27.5 | 102.7 | **108.4** |
| trips | loop | 9.02M | 28.3 | 7.4 | 0.79 | 31.0 | 81.8 | 127.3 |
| trips | **vectorised** | 9.02M | 21.8 | 7.4 | 0.76 | 21.5 | 65.2 | **79.6** |
| trilinear | loop | 7.87M | 22.9 | 6.1 | 0.57 | 27.4 | 71.9 | 126.7 |
| trilinear | **vectorised** | 7.87M | 19.0 | 6.1 | 0.63 | 20.7 | 60.2 | **65.6** |

`"broadcast"`'s loop pays nothing for the camera moving (+1%); `"trips"`'s pays
+56% and `"trilinear"`'s +76%, and the vectorised implementation removes almost
all of it. With it, the three modes finally rank by the work they do:
`trilinear` (65.6 ms, 7.87M fragments) < `trips` (79.6, 9.02M) < `broadcast`
(108.4, 24.61M).

**Whole training steps** on the same machine and the real configs
(`trippy-trips-perf-2`, `EXP-0003` `config_full2_*.yaml`, 20 timed steps,
median seconds), bisected into rasteriser / +U-Net / everything:

| config | impl | raster | +U-Net | **whole step** |
|---|---|---:|---:|---:|
| `config_full2_trips.yaml` | loop | 0.135 | 0.143 | **0.164** |
| `config_full2_trips.yaml` | **vectorised** | 0.081 | 0.085 | **0.100** |
| `config_full2_broadcast.yaml` | loop | 0.096 | 0.116 | **0.109** |
| `config_full2_broadcast.yaml` | **vectorised** | 0.081 | 0.100 | **0.090** |

A `trips` step goes from 164 ms to 100 ms and from 1.50x a `broadcast` step to
1.11x. Note that neither number is the ~1.4 s/step the EXP-0003 `full2-trips`
run logged: that run shared the machine with several heavy CPU jobs and with
swap in use, so most of its observed 7x over `full2-broadcast` (which had the
machine largely to itself overnight) was contention, not mode. The mode's own
contribution, measured back to back, was 1.5x and is now 1.1x.

Two secondary effects come with the same change and are worth naming because
they are cheap to reintroduce by accident:

- **Per-layer work over all N points.** The loop evaluates mode `"trips"`'s
  four-corner gate (and `layer_factor`, and the `layer_bounds` inside it) over
  the full point array at every layer, because it reads the gate off the
  full-length `uv`. `alive` starts at `valid`, so none of that can matter on a
  row the cull already dropped.
- **Data-dependent readbacks.** Per layer the loop issues one `torch.nonzero`,
  six boolean-mask gathers (each an implicit `nonzero`) and one
  `bool(keep.any())` — 40 queue drains per render at L=5, against 3.

What is deliberately **kept** from the loop: bilinear weights and alpha are
computed per emitted fragment, not for all `L · M · 4` candidates, which is why
the vectorised path runs geometry, then alpha, then the alpha floor as three
passes. Computing alpha for every candidate was measurably worse on CPU and
bought nothing on MPS.

Two more MPS numbers from the same job, both load-bearing for the defaults:

- `segment_offsets(method="bincount")` costs **81.6 ms** against
  **0.28 ms** for `"searchsorted"` at 196k layer-pixels. `"searchsorted"` is
  the default and must stay it on MPS.
- `sort_method="composite"` (one stable int64 argsort) is **19.3 ms** at 24.6M
  fragments; the `"two_pass"` fallback is **36.5 ms**. The int64 key is not the
  problem it was assumed to be (docs/LIMITATIONS.md).

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

### Actual state as of v0.4.0 (see ADR-0005, `rust/README.md`)

The diagram above is the eventual, fully-wired layout. What exists today:

```
rust/
├── Cargo.toml                      trippy's own thin workspace (version 0.1.0)
│                                    + exclude = ["brush-trips"] and a verbatim copy
│                                    of the submodule's two [patch] tables
├── Cargo.lock                      seeded from the submodule's, pinning the same
│                                    burn / cubecl / wgpu revisions
├── crates/
│   ├── brush-pyramid/              the pyramid rasteriser FORWARD pass, ported
│   │   ├── src/params.rs            PyramidParams, mirroring trippy.constants
│   │   ├── src/grid.rs              LayerGrid: shapes, layer-major flat index
│   │   ├── src/factor.rs            layer_bounds + TRIPS's compute_point_size_fac
│   │   ├── src/scene.rs             PointSet, Camera (npz + JSON loaders)
│   │   ├── src/npz.rs               minimal numpy .npz reader (stored + deflate)
│   │   ├── src/cpu.rs               CPU reference forward (the twin)
│   │   ├── src/gpu/kernels.rs       six #[cube(launch)] kernels    | `gpu` feature
│   │   ├── src/gpu/mod.rs           Burn/wgpu host pipeline        | `gpu` feature
│   │   ├── src/gpu/burn_bridge.rs   CubeTensor -> burn::Tensor<D> | `gpu` feature
│   │   ├── src/png.rs               tiny PNG writer for the example
│   │   ├── examples/render_frame.rs points + camera JSON -> PNG
│   │   └── tests/parity_{cpu,gpu}.rs
│   └── brush-unet/                  the U-Net decoder + tone mapper, ported
│       ├── src/config.rs            UnetConfig/CameraConfig + safetensors key schema
│       ├── src/weights.rs           safetensors reader (host-side, no Burn)
│       ├── src/net.rs               GatedBlock/UpBlock/Unet (Burn)  | `gpu` feature
│       ├── src/camera.rs            NeuralCamera tone mapper (Burn) | `gpu` feature
│       ├── examples/render_frame_full.rs  whole frame -> PNG + per-stage ms
│       └── tests/{schema_cpu,parity_gpu}.rs
│   └── trips-viewer/                the native Mac viewer (v0.4.0, ADR-0006)
│       ├── src/bundle.rs            the `trippy-bundle-1` reader
│       ├── src/camera.rs            fly camera -> brush_pyramid::Camera per frame
│       ├── src/renderer.rs          one frame; the performance levers
│       ├── src/app.rs               egui shell, ms/fps readout, view toggle
│       ├── src/blit.rs + shaders/   bind a Burn buffer into egui's render pass
│       └── src/main.rs              window, --screenshot, --bench, --profile
└── brush-trips/                    git submodule -> github.com/ggjordan/brush,
                                     branch trippy-fork (upstream 8b7f5c6c + Splats'
                                     robust/appearance/surface patches, merged)
```

`brush-pyramid` and `brush-unet` stayed in trippy's own workspace rather than moving
into the fork: the `gpu` feature reaches into `rust/brush-trips/crates/{brush-cube,
brush-sort,brush-prefix-sum}` by path, which works across a workspace boundary once
the submodule is `exclude`d and the `[patch]` tables are duplicated (`rust/README.md`
explains why all three ingredients are load-bearing). Without that feature the crate
has no dependency heavier than `serde` and `flate2`, so `scripts/build.sh` /
`scripts/test.sh` stay in their seconds-long budget on every push.

#### Forward data flow in Rust

Identical on both paths, and atomic-free for the same reason as the Python/Metal
version — every write goes to an address owned by exactly one thread:

```
points + camera
   -> project & count      per point: uv, depth, size_px, cull, slot budget
   -> prefix sum           brush-prefix-sum, giving each point its write offset
   -> emit fragments       one (layer,pixel) key + depth key + alpha + point id per slot
   -> radix argsort x2     by depth, then by key (both stable) = (layer, pixel, depth)
   -> segment bounds       per-layer-pixel [start, end)
   -> blend_fwd            one thread per layer-pixel, front-to-back, cap + cutoff
   -> add background       out += t_final * bg
   -> L feature images, finest first, plus t_final and n_used
```

and then, in `brush-unet`, still on the device:

```
L x CubeTensor (P, C)   the flat composited buffer
   -> burn_bridge       one zero-input fusion `Operation` binds the buffer into
                        the fusion stream; slice + reshape + permute give
                        `Tensor<4>` [1, C, h_l, w_l] per layer  (zero-copy)
   -> Unet              start gated block on the coarsest level, then L-1
                        upsample blocks (bilinear x2, CombineBridge centre-crop,
                        gated conv), then a 1x1 conv to RGB
   -> NeuralCamera      x * 2**-ev  ->  wb * x  ->  vignette(uv) * x  ->  LUT(x)
   -> [1, 3, H, W] display RGB
```

The bridge is the piece `docs/LIMITATIONS.md` used to list as missing: in the pinned
Burn revision `Tensor<D>` is backend-erased over the *fusion* backend, so a raw
`CubeTensor` cannot simply be wrapped — it has to be bound as the output of a
registered custom operation. `rust/README.md` has the details.

Two decisions are worth carrying forward:

- **Counting and emission cannot disagree.** The counting kernel reserves four slots
  per selected layer; the emission kernel derives its layer loop from
  `budget / 4` rather than re-deriving the selection. Per-corner bounds and
  `alpha_min` tests happen only at emission, and a rejected corner writes a sentinel
  key that sorts past every real key and sits outside the segment table. This removes
  the whole class of bug `brush-render` guards against with its cap-and-pad in
  `map_gaussians_to_intersect`.
- **No `log2` in the kernels.** CubeCL exposes `ln` but not `log2`, and
  `ln(x)/ln(2)` in f32 straddles the wrong side of an integer at exact powers of two,
  which would move a point into the wrong pyramid layer. Both paths read the IEEE-754
  exponent instead, which is exactly `floor(log2 x)` for a positive normal float.
  That is *more* accurate than the Python reference and differs from it only within
  ~1e-6 relative of a power of two (`docs/LIMITATIONS.md`).

#### The viewer, and where it attaches (v0.4.0, ADR-0006)

`crates/trips-viewer` closes the loop. Per frame:

```
bundle load (once)      -> gpu::UploadedPoints   xyz/size/conf/feat, on the device
                                                 and kept there for the session

fly camera (WASD/drag)  -> brush_pyramid::Camera  (R row-major, t, fx/fy/cx/cy,
                                                   8-param Saiga distortion)
   -> render_pyramid_uploaded              the six kernels above, binding the
                                           resident buffers -- no upload
   -> Unet + NeuralCamera                  (view mode "network" only)
   -> resolve_to_cube_float                back to one bindable buffer
   -> egui paint callback + blit.wgsl      fullscreen triangle samples it
```

The **`UploadedPoints` step is the whole reason the frame rate is what it is.**
`render_pyramid` takes a host-side `PointSet` and uploads all four arrays on
every call: 80 MB per frame on the horse, for data that never changes, worth a
flat 12.2 ms. Splitting the upload out took the shipped `--half-net --scale
0.75` view from 21.7 to 29.5 fps and the `raw level-0` view from 46.2 to 116.2
fps (`research/trips-metal.md`, job `trippy-web-unet-gpu-3`). The `PointSet`
entry points still exist and still work — they upload and delegate — so every
test, example and CLI tool is unchanged.

and the view toggle picks *which* buffer is blitted, from the **same** render:

| view | buffer | meaning |
|---|---|---|
| network | the tone mapper's `[1, 3, H, W]`, permuted to channel-last | the displayed frame |
| raw level-0 | rows `0 .. h*w` of the rasteriser's `(P, C)` output — no copy, level 0 is already at offset 0 | the evidence, before the network |
| coverage | `t_final`, shown as `1 - t_final` | which pixels the network invented |

The viewer does **not** live in `apps/brush-app`, and the crates did **not** move into
the fork. `docs/decisions/ADR-0006-viewer-integration.md` records that decision and its
costs; the short version is that a separate binary needed no fork push, keeps
`scripts/build.sh` fast, and makes "Brush's own `.ply` viewing still works" true by
construction rather than by test.

Two things changed *outside* the viewer to make it possible, and both are visible to
the rest of the pipeline:

- **`Camera` gained lens distortion.** A bundle stores world-space points and each
  view's Saiga coefficients, where the older per-view export baked one view's pose and
  distortion into the point positions. All-zeros is the identity, so every fixture and
  parity test is unaffected.
- **`PyramidParams` gained four performance levers** (`frustum_cull`, `layer_floor`,
  `sort`, `feature_store`, plus `depth_range` which configures `sort`), each defaulting
  to the exact pipeline, each with a serde default so older JSON still loads. The CPU
  reference implements `layer_floor` and *refuses* the two GPU-only ones rather than
  ignoring them.
- **`Unet::load_with_precision`** lets the decoder run in f16. This is the lever that
  matters: measuring the viewer's `raw level-0` view — the identical rasteriser with the
  network removed — gives **9.8 ms against a 190 ms frame**, so the rasteriser is ~5 %
  and the network ~95 %, and the f16 network alone is 2.58x for 59.8 dB (visually free).
  Every rasteriser-side lever measures within run-to-run noise. `docs/LIMITATIONS.md`
  and `research/trips-metal.md` carry the table; this overturns the "sort-dominated"
  reading of the first Mac timing.

Still open: the backward pass (`blend_bwd`) in Rust, and why a browser frame is
~27x a native one now that the point upload — the obvious suspect — has been
removed and did not explain it (`docs/LIMITATIONS.md`). The point-upload API and
the web build are both done (`gpu::UploadedPoints`; `docs/WEB_VIEWER.md`).

## Validation strategy

CPU pytest (before any GPU job):

1. **Transform agreement**: `xform_a` (numpy) vs `xform_b` (torch) produce identical projects; reprojection of COLMAP `points3D` matches stored keypoints (~1 px, depth positive).
2. **Geometry** (`tests/test_raster_bounds.py`): no emitted fragment lands outside its own pyramid layer; a footprint straddling the border keeps only its in-bounds corners; a point a few pixels off screen still draws in the coarse layers (so the cull is conservative); odd-sized images keep their last row/column.
3. **Reference pair** (`tests/test_raster_ref.py`, `tests/test_raster_trips_mode.py`): `ref_numpy` (numpy, xform_a, explicit per-point/per-layer/per-corner loops) vs `ref_torch` (torch float64, xform_b, vectorised segment prefix sums) agree to <1e-6 on a 32×32 scene containing a pixel stacked past the 16-fragment cap and points on every border, in all three modes and in both pixel-centre and both pyramid-halving conventions.
4. **Sort equivalence** (`tests/test_raster_sort.py`): the composite int64 key and the two-stable-sort fallback produce identical permutations, including on depth ties; both segment-offset methods agree.
5. **Metal kernel vs torch reference** (`tests/test_raster_metal.py`, marked `gpu`): 32×32 synthetic scene, float32 Metal vs float64 CPU, max abs diff <1e-4.
6. **Gradcheck** (`tests/test_raster_bwd_ref.py`, CPU): `torch.autograd.gradcheck` in float64 on `ref_torch`, w.r.t. all five learnable inputs (`xyz`, `size`, `conf`, `feat`, `pose_delta`), individually and jointly, on the smooth fixture. Plus a line-for-line python transcription of `blend_bwd.metal` checked against autograd on the very function it differentiates — so the *formulas* are pinned without a GPU, and the GPU job only has to validate the *Metal translation*.
7. **Metal backward vs reference** (`tests/test_raster_bwd_metal.py`, marked `gpu`): float32 MPS gradients vs float64 CPU gradients, relative error < 1e-3, in all three layer modes at C=3 and C=4 (and in mode `"trips"` under both pixel-centre conventions); a case where both forward stop rules fire (16-fragment cap *and* transmittance cutoff) with an exact `n_used` agreement check; a 256×192 / 50k-point / C=4 / L=5 timing case; and 20 feature-only SGD steps whose loss must fall monotonically (features enter linearly, so the objective is exactly quadratic and any increase is a wrong gradient).
8. **U-Net shape**: odd-size crops, verify autograd.

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

## train/ -- config, trainable params, trainer loop, eval, checkpointing (feat/train, 2026-09-06)

`trippy/train/` is the CPU-testable trainer that drives `raster/` + `net/` end to end: sample a crop,
render the pyramid, decode + tone-map, compute loss, backprop, step. Every render call goes through
`trippy.raster.pyramid.render_pyramid` unmodified -- on `device="cpu"` that dispatches to the fully
differentiable `ref_torch` path, so the whole loop (including gradients into points/pose/net/camera) is
testable today; the Metal backward pass (`blend_bwd`) is developed concurrently and plugs in with no API
change here, since nothing in `trainer.py` inspects which backward path is active.

- **`trippy/train/config.py`**: `TrainConfig` -- every default is `configs/train_normalnet.ini`
  (TRIPS @ a59a65b) scaled for trippy's smaller compute budget, or an explicit trippy addition; each is
  cited in `trippy.constants`' "train/" section. Loaded via `TrainConfig.load_yaml(path)` =
  `TrainConfig(**yaml.safe_load(...))` -- a config file only states what differs from the defaults.
  Epoch-fraction fields (`lock_cameras_frac`, `lock_structure_frac`, `vgg_start_frac`) are stored as
  fractions of `epochs`, not absolute counts, so the *proportion* of a run spent locked/pre-VGG matches
  TRIPS (`lock_camera_params_epochs=100`, `lock_structure_params_epochs=10`,
  `only_start_vgg_after_epochs=100`, all out of `num_epochs=600`) regardless of how many epochs a given
  trippy run actually does. `PointSourceConfig` describes any `trippy.points.PointSource` (gaussian /
  colmap / union / npz) from a config file.
- **`trippy/train/params.py`**: `PointParams` -- the trainable point cloud. `xyz`, `raw_size`, `raw_conf`,
  `feat` are `nn.Parameter`s; `size()` = `softplus(raw_size)` and `conf()` = `sigmoid(10 * raw_conf)` are
  the *effective* (post-activation) values used at render time, exactly mirroring TRIPS's own raw/effective
  split (docs/TRIPS_REFERENCE.md Sec. 2) -- including the `x10` confidence scale, which trippy keeps rather
  than dropping (that section's Sec. 10.4 flags dropping it as a fidelity gap). `feat`'s first 3 channels
  are seeded from the point source's `rgb0` (so the untrained U-Net immediately sees real colour); any
  remaining channels get small Gaussian noise, not TRIPS's own full-range `Uniform(0,1)` texture init -- a
  deliberate trippy simplification. `provenance` is a buffer, never a `Parameter`. `PoseParams` holds one
  learnable SE(3) twist per training image, applied via `trippy.geom.xform_b.compose` at render time so
  gradients reach the delta through the projection, never the frozen COLMAP pose itself.
- **`trippy/train/trainer.py`**: `Trainer` -- owns the dataset, point source, `PointParams`/`PoseParams`,
  `NeuralCamera`, the U-Net, and one `torch.optim.Adam` with one param group per learning rate in
  `TrainConfig` (points, size, confidence, texture, background, poses, network, exposure, response --
  vignette and white balance are frozen, matching TRIPS's `fix_vignette`/`fix_wb` defaults). Structure/
  camera "locking" is implemented by toggling `requires_grad_` on the frozen parameters for the locked
  epoch range (not by zeroing an optimizer-group lr), so it composes cleanly with the shared
  `ReduceLROnPlateau` schedule (`mode="max"` on held-out PSNR, `factor`/`patience` from the ini).
  **Crop strategy ("K-adjust", not "render-full-then-crop"):** `train_step` samples a crop centre/zoom,
  calls `trippy.scene.dataset.crop()` for the crop-adjusted intrinsics and validity mask, and hands that
  adjusted `K` straight to `render_pyramid` with `image_hw = (crop, crop)` -- only the crop's own fragments
  are ever rasterised. `tests/test_train_crop_equivalence.py` proves this equals cropping a full render of
  the same points/pose to <1e-5 (float64 compute), for every pyramid mode. A soft AABB extent penalty (not
  in TRIPS) keeps `xyz` near the point source's initial bounding box, guarding docs/SPEC.md's "extent
  inflation" risk. `evaluate()` renders full frames for the held-out split, reports PSNR/SSIM (always) and
  LPIPS (only if `cfg.eval_lpips`, gated so CPU tests never require a network-reachable VGG backbone unless
  asked), and writes a 4-column honesty contact sheet (photo | render | raw level-0 | coverage) for up to
  `TRAIN_EVAL_MAX_SHEET_IMAGES` images, forced-held-out shade frames first. `save_checkpoint`/`resume`
  round-trip through `trippy.train.checkpoint_io` (atomic write, one `torch.save`d dict of state_dicts).
  `export_ply` writes the trained point cloud via `trippy.train.export.write_gaussian_ply`, using the
  trained feature vector's first 3 channels (clamped to [0,1]) as an approximate colour -- the true
  rendered appearance needs the trained U-Net decoding every pyramid level together, which no 3DGS viewer
  can run. `fit()` runs the full epoch loop (locks, vgg start, eval/checkpoint cadence) with an optional
  `max_minutes` wall-clock budget so a queue job ends cleanly (checkpoint + export always written before
  returning).
  **Point removal is the one thing that changes the shape of the trained state mid-run.**
  `maybe_prune_points(epoch)` runs at the top of each epoch, before that epoch's steps -- the same
  place TRIPS calls `AddAndRemovePoints` (`src/apps/train.cpp:670-674`) -- and hands
  `_apply_keep_mask` a boolean mask from `trippy.train.prune`. That method rebuilds each per-point
  `nn.Parameter` by `index_select` and **index-selects that parameter's Adam moments with it**, then
  swaps the new object into both the module and its optimiser param group, so moment row `i` still
  belongs to point `i`; it is trippy's equivalent of `NeuralScene::RemovePoints` +
  `ShrinkTextureOptimizer` + `MyAdam::shrinkInternalState` (`NeuralScene.cpp:1375-1470,362-370`,
  `MyAdam.cu:346-374`). `load_state` resizes the point parameters to the checkpoint's own count before
  loading, so a post-removal checkpoint resumes and re-evaluates normally. `point_stats()` adds the
  point count plus the shade audit's in-region dark-mass fraction to every eval record.
- **`trippy/train/prune_config.py` + `trippy/train/prune.py`**: point removal, and the shade-audit
  statistic computed in-process. `prune_config.py` holds two dataclasses (stdlib only, so
  `train/config.py` stays cheap to import -- the same split `hybrid/config_a.py` uses):
  `PointRemovalConfig` is **TRIPS's own rule**, ported -- drop every point whose effective
  confidence `sigmoid(10*raw_conf)` is below `conf_threshold`, on epochs
  `start_epoch + i*every_epochs` (`src/apps/train.cpp:846-851` and `:533-538`,
  `src/lib/models/NeuralTexture.h:42`, `src/lib/data/Settings.h:403-406,427`) -- and
  `ShadePruneConfig` is **trippy's own, deliberately audit-aligned heuristic**, which drops points
  that are inside the shade audit's region AND dark AND low-confidence. Both default off.
  `prune.py` is pure numpy and side-effect free: `build_shade_region`/`in_region`/`dark_mass_stats`
  are a field-for-field port of `~/Splats/tools/depthprior_shade_audit.py` (verified to reproduce its
  cached numbers on `kkc_15000` exactly), and the two `*_keep_mask` functions return boolean masks.
  Nothing here touches the optimiser; the Trainer does. TRIPS's point *adding* is not ported and the
  reason is written down in `prune.py`'s docstring and experiments/EXP-0010-point-removal/README.md
  (the default path shells out to an external NeAT binary; the in-tree fallback is dead code).
- **`trippy/train/eval.py`**: standalone, checkpoint-only evaluation -- `evaluate_checkpoint` rebuilds a
  `Trainer` from the checkpoint's own saved `cfg` (so dataset/point-source/split reconstruct identically)
  and loads the trained state, never re-training. `render_offpath` renders honesty triplets (raw | network
  | coverage, no ground truth) at arbitrary poses from a JSON file -- the stable API a later dolly-camera-
  path generator (docs/EXPERIMENTS.md "Dolly camera paths") plugs into.
- **`trippy/train/checkpoint_io.py`**: a checkpoint is one `torch.save`d plain dict (`epoch`, `cfg`,
  `point_params`, `pose_params`, `net`, `camera`, `background`, `optimizer`, `scheduler`), written to a
  temp file and renamed into place so a killed job never leaves a half-written checkpoint.

CLI: `trippy train --config cfg.yaml [--resume ckpt] [--max-minutes M] [--device cpu|mps]` and
`trippy eval --checkpoint ckpt [--images ...] [--device cpu|mps]` (`trippy/cli.py`).

## hybrid/ -- Gaussian splat renders as U-Net input (v0.3.0, designs C and A)

Two designs share one on-disk contract and one renderer, and nothing else.

- **`trippy/hybrid/render_splat_views.py`** renders every registered view of a scene against a
  binary 3DGS PLY through Splats' `gsrender.render` (imported by path, never copied), on
  `SceneDataset`'s own undistorted `(H, W, K)` grid so every render lines up pixel-for-pixel
  with its photo. Writes `<stem>.png` (uint8 rgb) + `<stem>.depth.npy` + `<stem>.alpha.npy`
  (float16) + a per-shard manifest. Idempotent per frame and shardable; MPS, so only ever run
  inside a GPU-queue job.
- **Design C** (`config_c.py`, `dataset_c.py`, `train_c.py`; `trippy hybrid-c train`) is a
  standalone image->image trainer: render in, photo out, no point cloud and no rasteriser.
  See docs/EXPERIMENTS.md "Hybrid design C".
- **Design A** (`config_a.py`, `gaussian_input.py`, `gsrender_live.py`) is not a trainer at
  all -- it is an *option on the existing point-based `Trainer`*. `HybridConfig` is a block
  inside `TrainConfig`; `GaussianInputs` loads/crops/pools the render and concatenates it onto
  every level of the TRIPS pyramid before the U-Net; `gsrender_live` supplies the same block
  at poses no photo exists for. See docs/EXPERIMENTS.md "Hybrid design A".

Design A's three load-bearing decisions:

1. **Only the network widens.** `TrainConfig.net_input_channels = feature_channels + G` feeds
   `NetworkConfig.num_input_channels`. Point features, background and the rasteriser stay at
   `feature_channels`, so `render_pyramid` is untouched and `layers[0]` is still the pure
   TRIPS composite every honesty artifact ("raw L0", coverage) is defined against.
2. **The render is cropped by the photo's own crop function**, with the same
   `(size, zoom, center)` and the same `K`, so the K-adjust cannot drift. Depth is metric
   camera-space z and is resampled, never rescaled, by the crop; its only scale is the
   scene-global `depth_scale` (measured once, recorded in the checkpoint).
3. **Missing Gaussian information is an all-zero block, never a fabrication.** A frame with no
   render, a crop chosen by `dropout_gaussian_p`, and a pose with no live renderer all produce
   the same thing. The channel count and level count never vary within a run, and the network
   is trained (by the dropout ablation) to be a working TRIPS renderer in that state. In
   particular a dolly/off-path pose never borrows its anchor image's precomputed render: that
   render is from a different camera, so it is rendered live or it is zeros.

`enabled: false` is the default and a hard no-op: `Trainer.hybrid is None`, the network keeps
its old width, and every code path is the pre-design-A one.
