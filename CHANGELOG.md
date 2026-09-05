# Changelog
All notable changes to trippy. Format: Keep a Changelog. Versions: semver tags `vX.Y.Z`. Every push also gets a `build-NNNN` tag.

## [Unreleased]

## [0.2.0] - 2026-09-06
Milestone numbering note: the plan tied v0.2.0 to the Karekare shade verdict and v0.4.0 to the Mac viewer. The viewer landed first (the long Karekare trainings are still queued behind Splats' jobs), so this release carries the viewer and the complete trainer; the shade verdict will be recorded in the release where it lands.
### Added
- Native Mac TRIPS viewer `trips-viewer` (pyramid -> U-Net -> camera -> screen) on wgpu: horse scene at 1920x1080 exact 204 ms; f16 U-Net + 0.75 render scale 45 ms = 22 fps; screenshot parity 82.7 dB vs the reference path. Launcher delivered (ADR-0006).
- Rust pipeline: brush-pyramid (CubeCL, parity 2e-6 vs Python) + brush-unet (Burn, 115 dB end-to-end parity), safetensors export, `trippy export-bundle`.
- Differentiable MPS rasteriser (blend_bwd), trainer with self-reporting runs, candidate report (dolly, honesty sheet, audits), design-B distillation pipeline, union point sources, web toolchain (`scripts/web_build.sh`).
### Findings
- Design C (U-Net refinement of Gaussian renders) does not fix the shade: shade PSNR -1.96 dB.
- First TRIPS training from Gaussian centres (40 ep): 14.4 dB held-out; the point cloud carries more dark mass in the shade volume (36%) than the Gaussian baseline (20%). Long runs queued.
### Added
- **Native Mac TRIPS viewer** (`rust/crates/trips-viewer`, ADR-0006): opens a trippy asset bundle from argv or a folder picker and renders it live -- pyramid rasteriser -> U-Net -> tone mapper -> screen -- at the window's size, with WASD/mouse flight, a `V` toggle between network / raw level-0 / coverage views, an on-screen ms+fps readout, and headless `--screenshot` / `--bench` / `--profile` paths for verification. A separate binary from Brush's own `brush`, which is untouched.
- `trippy export-bundle --checkpoint ... --out <dir>`: writes a `trippy-bundle-1` directory (`bundle.json` + `points.npz` + `weights.safetensors`) from either a published TRIPS checkpoint or a trippy-native one, so any scene opens in the viewer with no Rust change.
- `brush_pyramid::scene::Camera` gained an 8-parameter Saiga lens distortion (all-zeros = identity), applied on both the CPU reference and the CubeCL kernel, so bundles can store world-space points instead of one view's pre-distorted ones.
- `Unet::load_with_precision`: the decoder can run in f16. **2.58x on the whole frame (204 -> 79 ms at 1080p) for 59.8 dB against the exact pipeline, i.e. visually free** -- the lever that actually matters, because the network is ~89% of the frame.
- Six measured performance levers, every one defaulting to the exact pipeline: `frustum_cull`, `layer_floor` (fragment cap), `sort` (one packed 32-bit key instead of two radix passes), `feature_store` (f16 features), viewer-side render scaling, and the f16 network above.
- `scripts/open_mac_viewer.sh`: generates the double-click `OPEN_TRIPS_MAC_<name>.command` launcher.
- Native "trips" rasteriser mode (TRIPS's real layer rule), pixel_center/pyramid_halving options; native engine == per-layer parity to 1e-8 dB.
- Candidate report: shade dolly, off-path poses, Splats audit wrappers, honesty sheet.
- Brush fork as submodule (ggjordan/brush trippy-fork) + brush-pyramid/brush-unet crate skeletons (ADR-0005).
- MonoDepthSource (DepthPro via GPU queue), EXP-0004 sheet.
### Changed
- **Corrected a wrong performance conclusion.** The first Mac timing read as "sort-dominated over 10.4M fragments"; the viewer's `raw level-0` view runs the identical rasteriser with the network removed and measures **21.6 ms (46 fps)** against a 204 ms frame, so the rasteriser is ~11% and the U-Net ~89%. Every rasteriser-side lever measures within noise. See `research/trips-metal.md`.
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
