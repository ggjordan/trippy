# Changelog
All notable changes to trippy. Format: Keep a Changelog. Versions: semver tags `vX.Y.Z`. Every push also gets a `build-NNNN` tag.

## [Unreleased]
### Added
- Native "trips" rasteriser mode (TRIPS's real layer rule), pixel_center/pyramid_halving options; native engine == per-layer parity to 1e-8 dB.
- Candidate report: shade dolly, off-path poses, Splats audit wrappers, honesty sheet.
- Brush fork as submodule (ggjordan/brush trippy-fork) + brush-pyramid/brush-unet crate skeletons (ADR-0005).
- MonoDepthSource (DepthPro via GPU queue), EXP-0004 sheet.
### Fixed
- Trainer: exposure init relative to scene-mean EV, masked MSE normalisation (PSNR was 4.77 dB low), background, crop sampling inside image, seeding, non-finite gradient guard. Smoke run 1.6 -> 12.26 dB.
- se3_exp rotation gradient at phi=0; dataset.crop float64 on MPS; MPS->float64 casts.

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
