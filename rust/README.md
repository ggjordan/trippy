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
  tests. **Not yet done**: the backward pass (`blend_bwd`), the U-Net Burn graph +
  safetensors loader in `brush-unet`, wrapping the output as `burn::Tensor<4>` (see
  `docs/LIMITATIONS.md`), and the viewer hook-in at
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
