# Rust: Brush fork for production viewers

This directory holds trippy's own thin Rust workspace (`rust/Cargo.toml`) and the
Brush fork (`rust/brush-trips`, a git submodule), brought in at v0.4.0 once the
Python training phase locked the winning design and it's time to port the forward
pass to production Rust/Burn/CubeCL code.

## Why Rust and why later

The Python phase (v0.1.0–v0.3.0) is where we iterate rapidly on algorithms and loss functions using PyTorch's mature autograd. Once the design is locked (v0.3.0), we port the forward pass to the Brush fork for production viewers:

- **Mac viewer**: `apps/brush-app` (egui, Metal via wgpu).
- **Web viewer**: `apps/brush-app/web` (wasm-pack, WebGPU).

## Two Cargo workspaces, on purpose

```
rust/
├── Cargo.toml                 trippy's OWN thin workspace (version 0.1.0, tracks VERSION)
├── Cargo.lock                 seeded from brush-trips' lock -> same burn/cubecl/wgpu revisions
├── crates/
│   ├── brush-pyramid/          TRIPS pyramid rasteriser: CPU reference + CubeCL forward pass
│   │   ├── src/{params,grid,factor,scene,npz,cpu,output,fixture,png}.rs   no heavy deps
│   │   ├── src/gpu/{mod,kernels}.rs                                       `gpu` feature only
│   │   ├── examples/render_frame.rs
│   │   └── tests/{parity_cpu,parity_gpu}.rs
│   └── brush-unet/             skeleton: UnetConfig placeholder + unit tests
└── brush-trips/                git submodule -> https://github.com/ggjordan/brush (branch trippy-fork)
    ├── Cargo.toml               Brush's OWN workspace (version 1.0.0, upstream's own scheme)
    ├── crates/                  brush-cube, brush-sort, brush-prefix-sum, brush-render, ...
    └── apps/brush-app/          the actual viewer app
```

`rust/brush-trips` is a **submodule**, not a subtree — see
`docs/decisions/ADR-0005-brush-fork-layout.md` for the reasoning (short version: `gh`
was already authenticated with fork access, so a public fork + submodule was less
work and easier to maintain than squashing Brush's history into trippy's own).

The two `Cargo.toml`s are deliberately **separate workspaces**: a crate can't belong
to two workspaces at once, and nesting `brush-pyramid`/`brush-unet` under
`rust/brush-trips/crates/` (the eventual real-integration layout sketched in
`docs/ARCHITECTURE.md`) would force that conflict today for no benefit. Keeping them
apart also means `scripts/build.sh`/`scripts/test.sh` never need to touch the much
larger Brush workspace on every push (see below).

### How the thin workspace reaches into the submodule (v0.4.0)

`brush-pyramid`'s `gpu` feature depends on three crates *inside* the submodule
(`brush-cube`, `brush-sort`, `brush-prefix-sum`) by **path**, plus Burn, CubeCL and
wgpu. That works from a separate workspace, but it took three things — all of which
are load-bearing, so don't "tidy" them away:

1. **`exclude = ["brush-trips"]` in `rust/Cargo.toml`.** The submodule physically
   lives inside the `rust/` workspace directory, and Cargo auto-adopts any path
   dependency under the workspace root as a *member*. Without the exclude,
   `brush-cube` is pulled into trippy's workspace and its `log.workspace = true`
   is looked up in trippy's `[workspace.dependencies]` instead of the submodule's,
   failing with ``` `dependency.log` was not found in `workspace.dependencies` ```.
   With it, each crate keeps inheriting from the workspace it really belongs to.
2. **The two `[patch]` tables copied verbatim** from `rust/brush-trips/Cargo.toml`
   into `rust/Cargo.toml`. Cargo only reads `[patch]` from the workspace root it is
   *building*, so without the copy we would silently link unpatched upstream
   wgpu/cubecl — which cannot compile Brush's kernels to MSL (the
   `workgroup_uniform_load` barrier fix, tracel-ai/cubecl#1525, lives on the fork).
   Keep the two tables byte-identical; a mismatch is a silently different GPU stack.
3. **`rust/Cargo.lock` seeded from `rust/brush-trips/Cargo.lock`.** Burn is pinned
   only by `branch = "main"`, so a fresh resolve would pick whatever `main` is
   today and drift away from the revision Brush's own code compiles against.
   Seeding the lock pins both workspaces to burn `b6e27bdc`, cubecl `0e0a3116`
   (ArthurBrussee/cubecl `msl-trial`) and wgpu `28d01c4f`
   (ArthurBrussee/wgpu `js-interop-30`). The dependency *specs* in
   `[workspace.dependencies]` must stay byte-identical to the submodule's for this
   to hold: two different git specs for one repository are two different Cargo
   sources and cannot be linked together.

So **the crates did not have to move into the fork**, and ADR-0005's
two-workspace decision stands unchanged. `cargo check -p brush-pyramid` (no
features) still has no dependency heavier than `serde` + `flate2`, so
`scripts/build.sh` and `scripts/test.sh` are unaffected.

## Brush fork: commit and patches

- Fork: **https://github.com/ggjordan/brush** (public, Apache-2.0).
- Base commit: `8b7f5c6c0638892204b540d9aced219f62fc2192` — same as Splats' working
  fork's starting point and, as of 2026-09-06, `ArthurBrussee/brush`'s `main` HEAD.
- trippy's pinned commit (the `trippy-fork` branch tip):
  `b2f2c3ea27e39c28509fc470b528cfee4cf6f6f6`.

Splats' patch set (`~/Splats/tools/patches/*.patch`) was reapplied as three separate
branches off the base commit (`patch-robust`, `patch-appearance`, `patch-surface`,
each one commit), then combined with two `git merge` steps into `trippy-fork` rather
than applying all three `.patch` files in sequence — sequential `git apply` failed
past the first patch because all three touch the same files
(`crates/brush-train/src/{config,train,lib}.rs`,
`crates/brush-dataset/src/{scene,scene_loader}.rs`) at overlapping line offsets. A
real three-way merge (common ancestor = the unpatched base) resolves the
non-overlapping additions automatically and only asks for a decision where two
patches genuinely touch the same lines.

| Patch | Applied? | What it adds | Merge conflicts resolved by hand |
|---|---|---|---|
| `brush-robust.patch` | Yes | Robust photometric loss: per-pixel down-weighting of view-inconsistent transients (people, moving foliage), with EMA residual stats, spatial-coherence gating, and optional rejection-mask viz dumps. New `crates/brush-train/src/robust.rs`. | — (first patch applied, nothing to merge against yet) |
| `brush-appearance.patch` | Yes | Per-image appearance embedding (NeRF-W / WildGaussians style): `--appearance-dim`, an embedding-to-affine-colour MLP, its own Adam optimizer state and warmup/regularisation. New `crates/brush-train/src/appearance.rs`; touches every dataset loader (`colmap.rs`, `nerfstudio.rs`, `realitycapture.rs`) to carry an `appearance_id` per view. | `scene.rs`/`scene_loader.rs` (`SceneBatch` gained both `view_name` from robust and `appearance_id` from appearance — kept both fields); `config.rs` (two independent option blocks — concatenated); `train.rs` (struct fields, `Self` initializer, and the loss computation — merged the robust-weighted match arms with appearance's `do_alpha_match` branch, since robust's version is a strict superset). |
| `brush-surface.patch` | Yes | Surface-lid penalty (world-space half-space penalty suppressing gaussians above a fitted surface, e.g. a water plane) and a per-ray depth-distortion penalty (2DGS/Mip-NeRF360-style, squared-difference form). New `crates/brush-train/src/surface.rs`; adds `--lid-*` and `--distortion-*` CLI options. | `config.rs` (another independent option block — concatenated); `lib.rs` (`pub mod robust;` + `pub mod surface;` — both kept); `train.rs` (struct fields for `lid`/`distortion` added alongside the robust+appearance fields; `Self` initializer likewise). |

None of the three patches are UI-only — all touch `brush-train`/`brush-dataset`/
`brush-render`, i.e. rendering/training correctness — so none were skipped. Nothing
in the brief's description ("distortion weight, bounds multiplier, surface lid, 2DGS
terms") turned out to reference a fourth, undocumented patch; `bounds_multiplier`
does not appear anywhere in the three `.patch` files or the merged result — only
`distortion_weight`, `lid_weight`, and the existing `BOUND_PERCENTILE` constant
(unrelated, pre-existing upstream code for `get_splat_bounds`).

All five branches (`upstream-base`, `patch-robust`, `patch-appearance`,
`patch-surface`, `trippy-fork`) are pushed to the fork for traceability.

## Building and testing

**trippy's own crates** (`brush-pyramid`, `brush-unet`) — fast, runs on every push.
No `gpu` feature, so no Burn/CubeCL/wgpu: a cold build is seconds.

```bash
cd rust
cargo check -p brush-pyramid -p brush-unet   # scripts/build.sh does this
cargo test  -p brush-pyramid -p brush-unet   # scripts/test.sh does this
```

**The GPU forward pass** — needs the whole Burn/CubeCL/wgpu tree, so build it
through the CPU-heavy queue (measured: 1m25s for the library, 55s more for the
test binaries, on a warm registry):

```bash
bash scripts/cpu_heavy.sh brush-pyramid-build -- bash -c \
  'cd rust && cargo test -p brush-pyramid --features gpu --no-run'
```

Running the tests **is GPU work** and must go through the GPU queue:

```bash
bash scripts/gpu_submit.sh --prio 12 --wait brush-pyramid-gpu-1 -- bash -c \
  'cd rust && cargo test -p brush-pyramid --features gpu --test parity_gpu -- --nocapture'
```

Render a frame to a PNG (CPU reference without `--features gpu`, wgpu with it):

```bash
cd rust && cargo run --example render_frame --features gpu -- \
  --points ../tests/fixtures/synthetic/raster_fixture_trips_half/points.npz \
  --camera ../tests/fixtures/synthetic/raster_fixture_trips_half/camera.json \
  --mode trips --layers 3 --background 0.1,0.2,0.3,0.4 --out /tmp/frame.png
```

**The Brush fork itself** — slow (Rust + Metal shader compilation across the whole
vendored tree), never run on every push. Use `scripts/cpu_heavy.sh` so only one
heavy CPU job runs at a time:

```bash
# Local build (fast, incremental, once warm)
cd rust/brush-trips
cargo build --release -p brush-app

# Heavy build via the CPU-heavy job queue (checks free memory, single global lock)
bash scripts/cpu_heavy.sh brush-build -- bash -c \
  'cd rust/brush-trips && cargo build --release -p brush-app'
```

The resulting binary is `rust/brush-trips/target/release/brush` (the `brush-app`
package's binary target is named `brush`, per its own `Cargo.toml`; `cargo build
--release -p brush-app` still builds it). Confirmed working:

```
$ rust/brush-trips/target/release/brush --version
brush-cli 1.0.0
$ rust/brush-trips/target/release/brush --help
Brush - universal splats
...
Robust loss options: --robust-loss, --robust-start-iter, ...
Appearance options: --appearance-dim, --appearance-hidden, --lid-weight, --distortion-weight, ...
```

All three patches' CLI options showed up in the built binary, confirming the
merge (see the patch table above) is not just syntactically clean but produces the
combined feature set.

## `brush-unet`: the U-Net + tone mapper, and the weight schema

`brush-unet` is the second half of the forward pass: it takes the `L` feature
images `brush-pyramid` composites and produces the displayed RGB frame.

```
crates/brush-unet/
├── src/config.rs     UnetConfig / CameraConfig + the safetensors key schema.
│                     NO dependencies -> compiled and tested on every push.
├── src/weights.rs    the safetensors reader (host-side `Vec<f32>` only).
├── src/net.rs        GatedBlock / UpBlock / Unet, Burn      | `gpu` feature
├── src/camera.rs     NeuralCamera (exposure/WB/vignette/LUT)| `gpu` feature
├── examples/render_frame_full.rs   points+camera+weights -> PNG, with timings
└── tests/{schema_cpu,parity_gpu}.rs
```

### CubeTensor -> `burn::Tensor<4>` (the bridge that was missing)

`brush_pyramid::gpu::PyramidRender` composites into one flat
`CubeTensor<WgpuRuntime>` of shape `(P, C)` — pixel-major, channel-last,
layer-major over the pyramid. Burn's `Conv2d` wants a `Tensor<4>` in NCHW.
In the Burn revision this workspace pins (`b6e27bdc`), `Tensor<const D>` is
backend-erased over the **fusion** backend, and a fusion tensor is a handle in
a lazily recorded op stream, not a buffer — so there is no
`Tensor::from_primitive(CubeTensor)`. The supported way in is a custom
operation with **zero inputs** whose single output the op binds to an
already-computed concrete tensor:

- `brush-pyramid/src/gpu/burn_bridge.rs` — ~90 lines: a `BindOp` implementing
  `burn_fusion::stream::Operation`, registered with
  `client.register(StreamId::current(), OperationIr::Custom(desc), op)`. This is
  the one-output, no-input case of the seven-output `BindOp` in the fork's
  `brush-render/src/burn_glue.rs`. Zero-copy: no readback, one stream
  registration.
- `PyramidRender::layer_tensor(l)` then slices layer `l`'s rows out of the
  `(P, C)` tensor, reshapes to `[1, h_l, w_l, C]` and permutes to
  `[1, C, h_l, w_l]` — all on device.

This needed two extra dependencies, `burn-fusion` and `burn-ir`, whose specs
are copied verbatim from the submodule's `[workspace.dependencies]` for the
same reason every other burn spec is (see above).

### Weight schema (`trippy.net.export_safetensors`, format `trippy-unet-1`)

`tools/export_unet_safetensors.py` writes one `.safetensors` file holding the
U-Net and the tone mapper. Every tensor is **float32, C-contiguous**, written
in the order below so the data segment is contiguous. `brush_unet::weights`
is the 1:1 reader and refuses a file that deviates.

`__metadata__` (all values are strings):

| key | meaning |
|---|---|
| `format` | `trippy-unet-1` |
| `num_layers` `filters` `in_channels` `out_channels` | `L`, `F`, `C`, `O` |
| `activation` `norm` `upsample_mode` `last_act` | must be `elu` / `id` / `bilinear` / `id`; anything else is rejected rather than silently approximated |
| `has_camera` | `1` when the tone mapper is included |
| `num_frames` `response_params` | `M`, `P` |
| `image_height` `image_width` | the resolution the vignette's aspect correction was fitted at |
| `enable_exposure` `enable_white_balance` `enable_vignette` `enable_response` | `0`/`1` |

Tensors:

| key | shape | source (PyTorch) |
|---|---|---|
| `unet.start.feature.{weight,bias}` | `(F-2C, C, 3, 3)`, `(F-2C,)` | `start.conv.feature_conv` |
| `unet.start.gate.{weight,bias}` | same | `start.conv.gate_conv` |
| `unet.up.{k}.feature.{weight,bias}` | `(out_k, F, 3, 3)`, `(out_k,)` | `up[k].conv.feature_conv` |
| `unet.up.{k}.gate.{weight,bias}` | same | `up[k].conv.gate_conv` |
| `unet.final.{weight,bias}` | `(O, F, 1, 1)`, `(O,)` | `final[0]` |
| `camera.exposure` | `(M,)` | `exposures_values`, squeezed |
| `camera.white_balance` | `(M, 3)` | `white_balance_values`, squeezed |
| `camera.vignette_params` | `(3,)` | `vignette_net.vignette_params` |
| `camera.vignette_center` | `(2,)` | `vignette_net.vignette_center`, squeezed |
| `camera.response` | `(O, P)` | `camera_response.response`, squeezed |

with `out_k = F - C` for the last block (`k == L-2`) and `F - 2C` otherwise —
TRIPS's "-2C" trick (Networks.h:1033-1034), which is what makes the final
bridge concat land on exactly `F` channels.

**`up.{k}` is indexed in application order, not by pyramid level.** `k = 0` is
the block that consumes `inputs[L-2]` (the coarsest input the start block did
not take) and `k = L-2` is the `last = true` block that consumes `inputs[0]`.
`UnetConfig::up_level(k) == L - 2 - k`. This matches the Python `up`
ModuleList index; getting it backwards produces a network that runs, has the
right parameter count, and renders nonsense.

### Building and testing `brush-unet`

Default (no Burn — `scripts/build.sh` / `scripts/test.sh`, seconds):

```bash
cd rust && cargo test -p brush-unet          # config + schema_cpu
```

GPU parity (build through the CPU-heavy queue, run through the GPU queue):

```bash
bash scripts/cpu_heavy.sh brush-unet-build -- bash -c \
  'cd rust && cargo test -p brush-unet --features gpu --release --no-run'

bash scripts/gpu_submit.sh --prio 12 --wait brush-unet-gpu-1 -- bash -c \
  'cd rust && cargo test -p brush-unet --features gpu --release --offline \
   --test parity_gpu -- --nocapture --test-threads=1'
```

The three fixture tests are self-contained (committed random weights,
`tests/fixtures/synthetic/unet_fixture_small/`, ~290 KiB). The fourth,
`horse_frame_matches_the_python_parity_engine`, **skips** unless the public
Zenodo horse exports exist under `$TRIPPY_OUTPUT/brush/horse/`; generate them
with

```bash
PYTHONPATH=. TRIPS_DEVICE=cpu python tools/export_unet_safetensors.py horse-e2e --index 8
```

Render a whole frame and print per-stage timings:

```bash
cargo run --release --example render_frame_full --features gpu -- \
  --points  $TRIPPY_OUTPUT/brush/horse/view_00008_points.npz \
  --camera  $TRIPPY_OUTPUT/brush/horse/view_00008_camera.json \
  --params  $TRIPPY_OUTPUT/brush/horse/view_00008_params.json \
  --weights $TRIPPY_OUTPUT/brush/horse/horse_unet.safetensors \
  --out /tmp/frame.png --iters 10
```

## `trips-viewer`: the native Mac viewer

`rust/crates/trips-viewer` is the third crate in the thin workspace and the thing
the other two exist for: a window that renders a TRIPS scene live, at the window's
own size, with the real forward pass in the middle.

```
crates/trips-viewer/
├── src/main.rs        argv, the eframe launch, and the headless --screenshot/--bench paths
├── src/bundle.rs      the `trippy-bundle-1` reader (bundle.json + points.npz + weights)
├── src/camera.rs      the fly camera; how a drag becomes (R, t)
├── src/renderer.rs    one frame: pyramid -> U-Net -> tone map, or a diagnostic buffer
├── src/app.rs         the egui shell: input, the ms/fps readout, the lever checkboxes
├── src/blit.rs        the egui paint callback that binds a Burn buffer, no copy
└── src/shaders/blit.wgsl
```

**It is a separate binary, not a panel inside `apps/brush-app`.** That was the open
question ADR-0005 left; `docs/decisions/ADR-0006-viewer-integration.md` decides it and
says why. The practical consequence: `rust/brush-trips` is untouched at its pinned
commit, so Brush's own `brush` binary and its `.ply` viewing cannot have regressed.

### How the buffer gets on screen

eframe creates the `wgpu::Device`; `burn_wgpu::init_device` hands that same device to
Burn (exactly as `brush_process::burn_init_device` does for `brush-app`); the
rasteriser's `(P, C)` output buffer — or, in network mode,
`burn_bridge::resolve_to_cube_float` of the tone mapper's output — is bound as a
read-only storage buffer in an egui paint callback, and a fullscreen triangle samples
it. Nothing round-trips through host memory.

`resolve_to_cube_float` (the reverse of the existing `float_tensor`) and `gpu::sync`
were added to `brush-pyramid` for this, so the viewer needs no dependency on
`brush-render`.

### Building and running

```bash
bash scripts/cpu_heavy.sh trips-viewer-build -- bash -c \
  'cd rust && cargo build --release -p trips-viewer'

rust/target/release/trips-viewer $TRIPPY_OUTPUT/brush/horse_bundle
```

Running the window **is GPU work**: keep it to a few seconds for a functional check,
and take every number through the queue (below). Unit tests (13, pure logic, no GPU)
are not on the push path — see `docs/LIMITATIONS.md` — and run with:

```bash
bash scripts/cpu_heavy.sh trips-viewer-test -- bash -c \
  'cd rust && cargo test -p trips-viewer --release'
```

### Headless, for correctness and timing

The same binary renders without a window, which is how the viewer is verified given
that no agent may look at a render:

```bash
bash scripts/gpu_submit.sh --prio 12 --wait mac-viewer-gpu-N -- bash -c \
  'rust/target/release/trips-viewer $TRIPPY_OUTPUT/brush/horse_bundle \
     --view 8 --frames 3 --bench 7 --screenshot /tmp/frame.png'
```

- `--screenshot out.png` writes one frame through the identical render path. The
  acceptance check is PSNR between that PNG and `render_frame_full`'s.
- `--bench N` times `N` frames, each ended by a real device sync (`gpu::sync`) rather
  than a readback, so the number is the render's and not a 24 MB transfer's.
- `--profile` prints per-stage milliseconds (a device sync per stage; a profile, not a
  frame time).
- `--half-net`, `--scale F`, `--no-cull`, `--cap-fragments`, `--fp16`,
  `--packed-sort` are the performance levers, all off / 1.0 by default.
  `docs/LIMITATIONS.md` says what each costs and `research/trips-metal.md` has
  the measured table.

### Where the frame time actually goes (measured 2026-09-06, M3 Ultra)

The viewer's `raw level-0` view runs the **identical** rasteriser and stops
before the network, which makes the split free to measure:

| what | 1920x1080, horse bundle |
|---|---|
| whole frame (`network` view, exact) | 204 ms — 4.9 fps |
| pyramid rasteriser alone (`raw` / `coverage` view) | **21.5 ms — 46.6 fps** |

So the rasteriser, 10.4 M fragments and two 32-bit radix sorts included, is about
**11 %** of the frame; the U-Net and tone mapper are the other **89 %**. This
contradicts the "sort-dominated" reading of the first Mac timing recorded in
`research/trips-metal.md` (which timed cumulative prefixes and could not separate
the last stage from the whole). Every rasteriser-side lever consequently measures
within noise, and the levers that move the number are the two that reduce the
*network's* work: `--scale` and `--half-net`.

### The bundle format

The viewer knows nothing about checkpoints or ADOP scenes. It reads a directory:

```
<bundle>/bundle.json          format "trippy-bundle-1"; params + every camera
<bundle>/points.npz           xyz (N,3) WORLD space, size (N,), feat (N,C), conf (N,)
<bundle>/weights.safetensors  format "trippy-unet-1", unchanged
```

written by `trippy export-bundle --checkpoint <ckpt> --out <dir>`. World space, not
the camera-space pre-distorted points `tools/export_unet_safetensors.py horse-e2e`
writes, because a free camera cannot use points with one view's pose baked in. Lens
distortion therefore had to move into the renderer: `brush_pyramid::scene::Camera`
gained `distortion: [f32; 8]` (Saiga order `k1 k2 k3 k4 k5 k6 p1 p2`, all-zeros =
identity = the old behaviour) and both the CPU reference and the CubeCL kernel apply
it. On the horse (`k1 = -0.064`, `k2 = 0.044`) ignoring it would move a corner pixel
by ~21 px.

## Parity testing

The Rust forward pass is checked against trippy's Python forward on identical
synthetic inputs. `tools/dump_raster_fixture.py` renders six tiny fixtures on CPU
(three layer-selection modes x both pixel-centre conventions, 64x48, 3 layers, 500
points, C=4) into `tests/fixtures/synthetic/raster_fixture_*/`, ~294 KiB in total:

```
points.npz     xyz/size/feat/conf          np.savez            (ZIP_STORED)
camera.json    width/height/fx/fy/cx/cy/R/t
params.json    mode, pixel_center, halving, max_frags, t_cutoff, alpha_min, znear, bg
expected.npz   layer_l, t_final_l, n_used_l  np.savez_compressed (ZIP_DEFLATE)
meta.json      num_fragments, fragments_per_layer, layer_shapes
```

The two archives use different compression on purpose: between them they exercise
both branches of `brush_pyramid::npz`, which is a ~200-line reader rather than a ZIP
dependency.

- `tests/parity_cpu.rs` — CPU reference vs the Python `.npy`. Runs on any machine in
  milliseconds; part of `scripts/test.sh`.
- `tests/parity_gpu.rs` — CubeCL/wgpu vs the same `.npy`, **and** GPU vs CPU. Behind
  the `gpu` feature, so `scripts/test.sh` never builds or runs it.
- `tests/test_dump_raster_fixture.py` — re-renders from the pinned seed and compares,
  so a semantic change in `trippy.raster` fails on the Python side instead of
  silently invalidating fixtures the Rust tests trust.

Float images and `t_final` are compared with a 1e-4 absolute tolerance; `n_used` and
the fragment counts are integers and must match **exactly**.

`brush-unet` is checked the same way, with `tools/export_unet_safetensors.py fixture`
in the role of `dump_raster_fixture.py`. It writes
`tests/fixtures/synthetic/unet_fixture_small/` (~290 KiB, pinned seed, random
weights, `num_layers=5`, `C=4`, `F=32`, a 32x24 pyramid):

```
weights.safetensors   the U-Net + tone mapper, schema `trippy-unet-1`
io.safetensors        input.0 .. input.4, unet_out, rgb_out, camera_probe,
                      camera_probe_out         (PyTorch's own answers)
meta.json             seed, shapes, parameter_count, output magnitudes
```

- `crates/brush-unet/tests/schema_cpu.rs` — the key schema, the metadata, and that a
  truncated or unsupported file is an *error*, not a panic. Part of `scripts/test.sh`.
- `crates/brush-unet/tests/parity_gpu.rs` — Burn vs PyTorch at 1e-4 on the fixture
  (U-Net alone, camera alone, and the two chained), plus the horse end-to-end.
- `tests/test_net_export_safetensors.py` — the container round-trips, the header is
  8-byte aligned with a contiguous data segment (the Rust reader validates both), the
  key schema covers every parameter exactly once, and regenerating from the pinned
  seed reproduces the committed bytes.

The 32x24 base is not arbitrary: with `ceil` halving it gives level shapes
`(24,32) (12,16) (6,8) (3,4) (2,2)`, so the coarsest upsample produces a 4x4 that
must be centre-cropped down to the 3x4 raw input — i.e. the fixture exercises the
odd-size `CombineBridge` branch that TRIPS's own code cannot handle.

## What `brush-pyramid` implements

The same six stages on both paths, atomic-free by design:

| # | Stage | CPU (`src/cpu.rs`) | GPU (`src/gpu/kernels.rs`) |
|---|---|---|---|
| 1 | project, cull, count slots | `project_point`, `selected_layers` | `project_and_count_kernel` |
| 2 | prefix sum over counts | `Vec` running total | `brush_prefix_sum::prefix_sum` |
| 3 | emit fragments | `corner_fragments` | `emit_fragments_kernel` |
| 4 | sort by depth, then key | one stable `sort_by_key` | two `brush_sort::radix_argsort` passes |
| 5 | segment offsets | counting scan | `segment_bounds_kernel` |
| 6 | blend front-to-back | inline loop | `blend_fwd_kernel` |
| 7 | background | inline | `add_background_kernel` |

Stage 4 mirrors `brush-render`'s own depth-then-tile pattern: LSB radix sort is
stable, so ordering by depth first and by `(layer, pixel)` second leaves each
layer-pixel run in depth order, with ties broken by point id — the same order the
Python composite key produces.

Counting (stage 1) and emission (stage 3) cannot disagree, because stage 1 writes a
*slot budget* of four slots per selected layer and stage 3 derives its layer loop
from `budget / 4` rather than re-running the selection. Corners that fall outside a
layer, or below `alpha_min`, write a sentinel key equal to `P` (the layer-pixel
count), which sorts after every real key and lies outside the segment table.

## Web viewer (v0.5.0 groundwork)

`apps/brush-app/web` (wasm-pack + vite + React) is the fork's web demo. Full
build steps, timings, the browser support matrix on this Mac, and the on-paper
Quest assessment live in `docs/WEB_VIEWER.md`; the short version:

- `scripts/web_build.sh` builds it reproducibly (guard clauses for
  `npm`/`wasm-pack`/`wasm32-unknown-unknown`, then `wasm-pack build --release`
  + `vite build` at base path `/`) into `$TRIPPY_OUTPUT/web/brush-dist/`.
  First cold build on this Mac: **3 m 36 s** wall (`npm ci` seconds; cargo
  build for `wasm32-unknown-unknown` 1 m 30 s; `wasm-bindgen` + `wasm-opt -Oz
  --converge` shrinking `brush_app.wasm` 53 MB → 21.7 MB for the rest;
  `vite build` 1.48 s).
- `scripts/deliver.sh <dist-dir> <name> "<why>"` generates the
  `OPEN_<NAME>.command` launcher (127.0.0.1-only `http.server`) — no change to
  `deliver.sh` was needed; its existing "directory with `index.html`" branch
  already serves a vite build correctly.
- Proven end to end on this Mac (2026-09-06) with the **stock** Brush
  renderer and a synthetic 2,000-point Gaussian `.ply`
  (`trippy.train.export.write_gaussian_ply`, never anything from `~/Splats`):
  WebGPU adapter obtained, full asset chain loads, wasm app initialises a
  correctly-sized canvas with zero JS errors, in **Safari** (Chrome is not
  installed on this machine, so the actual `docs/SPEC.md` v0.5.0 acceptance
  criterion — fps in Chrome — is not yet checked).
- **Not yet done, and not in scope of this groundwork**: compiling
  `brush-pyramid`/`brush-unet` for `wasm32-unknown-unknown` at all, and hooking
  their output into this web build the way `splat_backbuffer.rs` will hook
  them into the native viewer. See `docs/WEB_VIEWER.md` "Next: wiring TRIPS
  in" for the specific risks (untested wasm32 compilation, the wasm-only `burn`
  feature set and `[patch]` table any new crate must respect).

## Licensing

The Brush fork retains Apache-2.0 license (inherited from the upstream Brush project by ArthurBrussee). trippy's own `rust/` workspace crates (`brush-pyramid`, `brush-unet` skeletons) are MIT, per ADR-0004. When distributing trippy's Rust code, include the `NOTICE` file with attribution.

## Status

- **v0.3.0**: Python training complete; design locked.
- **v0.4.0, in progress**: Brush fork forked and pinned (`rust/brush-trips`), Splats'
  patches reapplied, and the fork builds `brush-app` in release mode.
  `brush-pyramid` now holds the **real forward pass** — the CPU reference, the six
  CubeCL kernels, the npz/camera loaders, the `render_frame` example and both parity
  tests. `brush-unet` now holds the **Burn U-Net + tone mapper**, the safetensors
  loader, the `CubeTensor -> Tensor<4>` bridge, and the `render_frame_full` example,
  so `points -> pyramid -> U-Net -> tone map -> PNG` runs entirely on wgpu and
  matches trippy's Python parity engine on the public horse scene at 115 dB.
  **Not yet done**: the backward pass (`blend_bwd`) and the viewer hook-in at
  `apps/brush-app/src/ui/splat_backbuffer.rs`.
- **v0.5.0, in progress**: web-viewer *toolchain* proven end to end with the
  stock Brush renderer (build script, `.command` launcher, WebGPU render of a
  synthetic splat in Safari — see "Web viewer" section above and
  `docs/WEB_VIEWER.md`). **Not yet done**: wiring trippy's own TRIPS forward
  pass into the web build, a Chrome fps measurement, and the Quest on-device
  check.

## Submodule vs. subtree decision

See `docs/decisions/ADR-0005-brush-fork-layout.md`: submodule to a public fork
(`ggjordan/brush`), not a subtree, because `gh repo fork` was already available and
simple, and keeping Splats' patches as ordinary commits on a fork branch is easier
to maintain than replaying `.patch` files against a moving upstream.
