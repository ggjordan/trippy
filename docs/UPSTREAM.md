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
- **Local fork**: `~/Splats/tools/brush-final` (patches in `~/Splats/tools/patches/`)
- **Commit**: (to be recorded in `rust/README.md` and `docs/UPSTREAM.md` when integrated)

The fork extends Brush with:
- `crates/brush-pyramid`: trilinear splatting and pyramid rasterisation (CubeCL)
- `crates/brush-unet`: U-Net inference via Burn (conv2d + safetensors loading)
- `apps/brush-app/src/ui/splat_backbuffer.rs`: viewer integration

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
