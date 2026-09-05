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
├── crates/
│   ├── brush-pyramid/          skeleton: layer_bounds/layer_factor port + unit tests
│   └── brush-unet/             skeleton: UnetConfig placeholder + unit tests
└── brush-trips/                git submodule -> https://github.com/ggjordan/brush (branch trippy-fork)
    ├── Cargo.toml               Brush's OWN workspace (version 1.0.0, upstream's own scheme)
    ├── crates/                  brush-dataset, brush-render, brush-train, ... (upstream + patches)
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

**trippy's own crates** (`brush-pyramid`, `brush-unet`) — fast, runs on every push:

```bash
cd rust
cargo check -p brush-pyramid -p brush-unet   # scripts/build.sh does this
cargo test  -p brush-pyramid -p brush-unet   # scripts/test.sh does this
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

Before shipping, the Brush forward pass is validated against the Python version:

```bash
bash scripts/gpu_submit.sh --prio 15 parity-check -- \
  cargo test --release --test parity_vs_pytorch -- --nocapture
```

This test loads a trained `.ply`, runs it through both PyTorch and Rust pipelines, and asserts output agreement <1e-3.

## Licensing

The Brush fork retains Apache-2.0 license (inherited from the upstream Brush project by ArthurBrussee). trippy's own `rust/` workspace crates (`brush-pyramid`, `brush-unet` skeletons) are MIT, per ADR-0004. When distributing trippy's Rust code, include the `NOTICE` file with attribution.

## Status

- **v0.3.0**: Python training complete; design locked.
- **v0.4.0, in progress**: Brush fork forked and pinned (`rust/brush-trips`), Splats'
  patches reapplied, `brush-pyramid`/`brush-unet` crate skeletons added (with
  placeholder logic + passing unit tests) in trippy's own thin workspace, and the
  fork builds `brush-app` in release mode. **Not yet done**: the real
  `emit_fragments`/CubeCL kernels, the real U-Net Burn graph, moving/wiring
  `brush-pyramid`/`brush-unet` into `rust/brush-trips` proper, the viewer hook-in at
  `apps/brush-app/src/ui/splat_backbuffer.rs`, and the parity test.
- **v0.5.0**: Web viewer complete.

## Submodule vs. subtree decision

See `docs/decisions/ADR-0005-brush-fork-layout.md`: submodule to a public fork
(`ggjordan/brush`), not a subtree, because `gh repo fork` was already available and
simple, and keeping Splats' patches as ordinary commits on a fork branch is easier
to maintain than replaying `.patch` files against a moving upstream.
