# Changelog
All notable changes to trippy. Format: Keep a Changelog. Versions: semver tags `vX.Y.Z`. Every push also gets a `build-NNNN` tag.

## [Unreleased]

## [v0.1.0] - 2026-09-06

## [0.1.0] - 2026-09-06
### Added
- Repo governance (AGENTS.md, scripts, hooks, GPU-queue and delivery wrappers), public MIT repo.
- Python package: two independent geometry implementations, COLMAP bin/txt loader, undistort+cache dataset, splits.
- Point sources: GaussianPlySource (5.7M pts on kk-coherent), ColmapSparseSource, MonoDepthSource (DepthPro via queue), UnionSource, density CLI.
- TRIPS pyramid rasteriser: torch emission/sort + Metal blend_fwd/blend_bwd via torch.mps.compile_shader, no atomics; gradients match float64 to <4e-6.
- Network port (gated U-Net, neural camera, L1+SSIM+LPIPS), checkpoint loader (34/34 tensors match the public horse checkpoint).
- Trainer (crops by K-adjust, schedule, eval, export), `trippy render/train/eval/density/depth-points/parity`.
- EXP-0001 Karekare pyramid sheets, EXP-0002 horse parity: 22.27 dB vs GT (authors 22.34), 36.99 dB vs authors' render. v0.1.0 gate passed.
### Known issues
- Rasteriser needs a native "trips" layer mode (layers 0..ceil(log2 s)); parity used per-layer calls.
- se3_exp rotation gradient at phi=0 (fix in flight).

### Added
- Phase 1 skeleton: agent rules (AGENTS.md), review-gate git hooks, build/test/push/release scripts.
- pyproject.toml with PyTorch, Metal compilation, testing, linting.
- trippy module skeleton: geometry transforms (xform_a, xform_b), agreement test.
- Initial GPU queue smoke test round-trip.
