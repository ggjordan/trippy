# Rust: Brush fork for production viewers

This directory will hold the Brush fork (`rust/brush-trips`) at v0.4.0, when the Python training phase is complete and the winning design is ported to production Rust/Burn/CubeCL code.

## Why Rust and why later

The Python phase (v0.1.0–v0.3.0) is where we iterate rapidly on algorithms and loss functions using PyTorch's mature autograd. Once the design is locked (v0.3.0), we port the forward pass to the Brush fork for production viewers:

- **Mac viewer**: `apps/brush-app` (egui, Metal via wgpu).
- **Web viewer**: `apps/brush-app/web` (wasm-pack, WebGPU).

## Structure (v0.4.0 onward)

```
rust/brush-trips/
├── crates/
│   ├── brush-pyramid/
│   │   ├── src/lib.rs        emit_fragments, radix argsort, prefix_sum
│   │   └── src/kernels/      blend_fwd, blend_bwd (CubeCL)
│   ├── brush-unet/
│   │   └── src/lib.rs        U-Net conv2d inference via Burn
│   └── [existing brush crates...]
├── apps/
│   └── brush-app/
│       ├── src/ui/splat_backbuffer.rs   TRIPS rasteriser hook-in
│       ├── src/                         viewer app
│       └── web/                         WebGPU web viewer
├── build.rs                  version detection
├── Cargo.toml                workspace config
└── [other Brush files...]
```

## Building and testing

Builds are resource-intensive (Rust compilation + Metal shader compilation):

```bash
# Local build (fast, incremental)
cd rust/brush-trips
cargo build --release

# Heavy build (clean, slow, used in CI)
bash scripts/cpu_heavy.sh brush-build -- cargo clean && cargo build --release
```

The `cpu_heavy.sh` script ensures only one heavy compilation job runs at a time (global lock) and checks free memory beforehand.

## Parity testing

Before shipping, the Brush forward pass is validated against the Python version:

```bash
bash scripts/gpu_submit.sh --prio 15 parity-check -- \
  cargo test --release --test parity_vs_pytorch -- --nocapture
```

This test loads a trained `.ply`, runs it through both PyTorch and Rust pipelines, and asserts output agreement <1e-3.

## Licensing

The Brush fork retains Apache-2.0 license (inherited from the upstream Brush project by ArthurBrussee). When distributing trippy's Rust code, include the `NOTICE` file with attribution.

## Status

- **v0.3.0**: Python training complete; design locked.
- **v0.4.0**: Brush fork cloned, patches applied, `brush-pyramid` and `brush-unet` crates added, viewer hook-in wired, parity test passing.
- **v0.5.0**: Web viewer complete.

## Future: submodule vs. subtree decision

Whether the Brush fork lives as a git submodule or subtree in the trippy repo will be decided by a future ADR. For now, both are feasible given the locked design.
