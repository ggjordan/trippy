# ADR-0001: PyTorch on MPS first, then Brush fork for production

Date: 2026-09-05 · Status: Accepted

## Context

We need to port TRIPS (Trilinear Point Splatting) to Apple Silicon for interactive rendering and training on Jordan's family scenes. Two viable paths exist:

1. **Option 1 (MLX)**: Use Apple's MLX framework for the entire pipeline. MLX compiles to Metal directly and has fewer external dependencies. MLX also supports inline Metal kernels (`mx.fast.metal_kernel`) and a local MLX port of gsplat exists (`~/Splats/tools/gsplat-mlx`). Rejected because the TRIPS network, losses and checkpoints are PyTorch, and the perceptual-loss and depth tooling we reuse live in the PyTorch ecosystem.

2. **Option 2 (Brush-first)**: Start directly in the Brush fork (Rust/Burn/CubeCL). Brush already has a portable differentiable splatting trainer and Metal-via-wgpu support. However, Rust/Burn debugging is slower, and Jordan's team has more PyTorch expertise than Burn expertise. Fast iteration on geometry and loss functions is critical in the research phase.

3. **Option 3 (PyTorch MPS + compile_shader)**: Prototype and train in PyTorch on MPS, using `torch.mps.compile_shader` to write custom Metal kernels for the sorted rasterisation step. PyTorch's autograd is mature; gradient debugging is straightforward. Once the design is proven, port the winning approach to the Brush fork (v0.4.0) for production viewers.

## Decision

Adopt **Option 3**: PyTorch 2.13 on MPS + inline Metal via `torch.mps.compile_shader` for research training (v0.1.0–v0.3.0), then port to a fresh Brush fork (v0.4.0+) for Mac/web viewers.

## Consequences

### For research (Python, v0.1.0–v0.3.0)

- **Fast iteration**: PyTorch autograd, standard debugging tools (gradcheck, hooks, probing), rapid hyperparameter sweeps.
- **Proven platform**: We trust PyTorch's correctness on autograd and numerical stability.
- **Custom Metal kernels**: `torch.mps.compile_shader` allows us to write two sorting/blending kernels in Metal and call them from Python.
- **No 64-bit atomics**: compile_shader does not support 64-bit atomic operations. Our design avoids atomics (pre-sort, segment offsets, per-fragment grads), making this a non-issue.

### For production (Rust/Burn, v0.4.0+)

- **Portable rendering**: CubeCL kernels run on Metal (wgpu), WebGPU, and future targets.
- **Shipping format**: Brush viewers on Mac and web without Python dependencies.
- **Parity testing**: PyTorch forward pass validates Rust/Burn correctness to <1e-3 tolerance.

### Transition

- v0.3.0 accepts the best hybrid design (Gaussian + TRIPS).
- v0.4.0 ports `blend_fwd`, `blend_bwd`, and U-Net to Rust/Burn.
- v0.4.0 also ports `emit_fragments` and sorting to Burn/CubeCL (proven to work in Python).
- Brush fork is a **fresh copy** of `~/Splats/tools/brush-final`, not a merge. Patches are applied, and TRIPS-specific crates are added.

## Alternatives considered

### Why not MLX?
- **Ecosystem, not capability**: MLX can run inline Metal kernels too. But TRIPS's U-Net, tone mapper, losses and released checkpoints are PyTorch; porting them to MLX is extra work with no payoff.
- **Reuse**: `lpips`, `torchvision` VGG and the DepthPro/MoGe depth tools already run in PyTorch on this machine.
- **Proven here**: `torch.mps.compile_shader` was verified on this Mac (torch 2.13.0) before the decision.
- **Speed does not matter at this scale**: MLX's per-op advantage on Apple Silicon is not the bottleneck; the rasteriser and U-Net are.

### Why not Brush-first?

- **Slower iteration**: Rust/Burn compilation is slow (minutes). Python reloads in seconds. Early-stage research needs fast iteration.
- **Gradient debugging**: PyTorch has mature tools (hooks, gradcheck, numerical gradient comparison). Burn's debugging story is weaker.
- **Learning curve**: The team is fluent in PyTorch; Burn would require ramp-up time.
- **Offline first**: Commit to Brush once the design is locked (v0.3.0), not before. This avoids porting a wrong design.

## Commitment

- **v0.3.0 freeze**: By end of v0.3.0, the winning hybrid design (Gaussian + TRIPS) is locked. No more algorithm changes.
- **v0.4.0 parity**: Brush implementation passes parity test (forward pass agrees to <1e-3 vs Python).
- **Maintenance**: After v0.4.0, Python code is read-only (bug fixes only); Brush is the source of truth for viewers.
