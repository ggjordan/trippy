# Upstream: TRIPS, Brush, and reference materials

## TRIPS: original paper and code

- **Repository**: https://github.com/lfranke/TRIPS (MIT licence)
- **Commit**: (to be filled when cloned via `tools/fetch_upstream.sh`)
- **Paper**: arXiv 2401.06003 (Franke et al. 2024)
- **Author project page**: https://linus-franke.com/trips/

The reference implementation is in CUDA/PyTorch and includes:
- Rasteriser (trilinear splatting to image pyramid)
- U-Net decoder (5 levels, 32 filters, ~130k params)
- Training loop with pose refinement
- Example scenes and pre-trained checkpoints

We port the rasteriser and U-Net to Apple Silicon (Metal + PyTorch MPS) and train on our own data.

## TRIPS data: Zenodo record 10687419

Zenodo record 10687419 (CC-BY 4.0 license) contains:
- **tt_scenes.zip** (3.2 GB): example scenes in COLMAP format
- **tt_checkpoints.zip** (2.7 GB): pre-trained model checkpoints for each scene
- **boat_scene_and_checkpoint.zip** (6.6 GB): single large scene with checkpoint
- **mipnerf360_our_resolutions.zip** (5.1 GB): MipNeRF-360 scenes at various resolutions

These files are **linked, never redistributed** in this repo. To use them:

```bash
bash tools/fetch_upstream.sh --download tt_scenes.zip
```

**Attribution**: when referencing Zenodo data in papers or documentation, use:

> Trilinear Point Splatting for Real-time Radiance Field Rendering. Linus Franke. Zenodo. https://zenodo.org/record/10687419

## Brush fork

- **Original repository**: https://github.com/ArthurBrussee/brush (Apache-2.0 licence)
- **Public fork**: https://github.com/ggjordan/brush (Apache-2.0 licence, forked via `gh repo fork`)
- **In trippy**: `rust/brush-trips`, a git submodule pointing at the public fork's
  `trippy-fork` branch (see `.gitmodules` and ADR-0005).
- **Upstream base commit**: `8b7f5c6c0638892204b540d9aced219f62fc2192` (2026-08-17) —
  the same commit Splats' working fork (`~/Splats/tools/brush-final`, patches in
  `~/Splats/tools/patches/`) started from. This was also `origin/main`'s HEAD on
  `ArthurBrussee/brush` at the time the fork was created (2026-09-06), so the fork's
  `main` and `upstream-base` branches match it too.
- **trippy's pinned commit**: the `trippy-fork` branch tip on `ggjordan/brush` —
  `8b7f5c6c06...` plus three merge commits, one per applied Splats patch. See
  `rust/README.md` for the exact SHA and per-patch notes (applied/skipped, and any
  hand-resolved merge conflicts between patches touching the same files).

The Brush fork extends upstream with:
- Splats' `brush-robust` patch: robust photometric loss (transient-distractor rejection).
- Splats' `brush-appearance` patch: per-image appearance embedding (NeRF-W / WildGaussians style).
- Splats' `brush-surface` patch: surface-lid penalty + depth-distortion (2DGS-style) regularisation.

Not yet added (v0.4.0 work, tracked in `docs/SPEC.md`):
- `crates/brush-pyramid`: trilinear splatting and pyramid rasterisation (CubeCL). A
  placeholder skeleton with only `layer_bounds`/`layer_factor` exists today, but in
  trippy's own thin `rust/Cargo.toml` workspace, not yet inside `rust/brush-trips/crates/`
  (see ADR-0005 for why the two are kept apart until the real integration).
- `crates/brush-unet`: U-Net inference via Burn (conv2d + safetensors loading). Same
  placeholder-only status as above.
- `apps/brush-app/src/ui/splat_backbuffer.rs`: viewer integration.

The fork retains Apache-2.0 licence. When distributing the Brush fork as part of trippy (v0.4.0 onward), include `NOTICE` files attributing ArthurBrussee's original work.

## Related reading

- **VR-Splatting** (arXiv 2410.17932): neural points in fovea, Gaussians in periphery. Existence proof for hybrid Gaussian + neural point rendering. Recommended reading for understanding the design rationale.

## Reference implementations and tools

This project reuses:
- `~/Splats/tools/depthprior_shade_audit.py`: measure opacity mass in shade region
- `~/Splats/tools/depthprior_shade_dolly.py`: render dolly video through shade
- `~/Splats/tools/tmp/extent-audit/extent_gate.py`: check scene extent inflation
- `~/Splats/tools/gsrender.py`: render 3DGS PLY files
- `~/Splats/tools/ldi/depth_batch.py`: Apple DepthPro depth inference
- `~/Splats/tools/va_depth/`: MoGe monocular depth (backup)
- `~/Splats/tools/publish/publish_splat.sh`: distill Gaussians for Quest fallback

These tools are read-only and shared across projects. trippy never modifies them.
