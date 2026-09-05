"""Pairs Splats-rendered (rgb, depth, alpha) frames with their photos, and builds pyramids.

Module: trippy.hybrid.dataset_c
Invariants:
    - Pairing is by filename stem only (`Path(name).stem`): a photo `IMG_3830.jpg` pairs with
      renders `IMG_3830.png` / `IMG_3830.depth.npy` / `IMG_3830.alpha.npy` written by
      `trippy.hybrid.render_splat_views`. A photo with no matching render triple (e.g. a
      still-pending shard) is silently excluded from `paired_names`, never a hard error --
      training/eval only ever iterate frames that actually have both halves of the pair.
    - `render_to_tensor` always produces channels ordered `[r, g, b, alpha, (depth)]`; `alpha`
      is included unconditionally (`HYBRID_C_DEFAULT_CHANNELS` = 4 = rgb+alpha) so the same
      `NetworkConfig` trippy.net.unet already validates against TRIPS applies unchanged
      (task brief). `channels=5` appends a normalised depth channel.
    - `build_pyramid` is *plain average pooling*, never re-run per pyramid level from the
      original photo/render -- the design brief's "5-level pyramid built by average-pooling
      the 4-channel render" is exactly `F.avg_pool2d(x, 2, 2)` applied `num_layers - 1` times,
      finest level first, matching trippy.net.unet's own `inputs[0]` = finest convention.
    - `crop_pair` returns *whole-pixel* windows, no zoom/resample, and only ever samples
      windows fully inside both tensors (no overshoot/padding mask is needed here, unlike
      `trippy.scene.dataset.crop`, because Design C's render and photo are guaranteed the same
      (H, W) by construction -- both come from the same `SceneDataset` grid).
Related docs: docs/PLAN-2026-09-05.md "Hybrid (v0.3): (C) render->photo U-Net refinement";
    trippy.hybrid.render_splat_views (produces the render triple this module consumes);
    trippy.hybrid.train_c (the trainer that calls crop_pair/build_pyramid per step).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from trippy.constants import (
    HYBRID_C_CHANNELS_WITH_DEPTH,
    HYBRID_C_DEFAULT_CHANNELS,
    HYBRID_C_DEPTH_NORM_SCALE,
)


def output_stems(renders_dir: str | Path) -> set[str]:
    """Stems in `renders_dir` that have all three render outputs (rgb png, depth/alpha npy)."""
    renders_dir = Path(renders_dir)
    stems: set[str] = set()
    for png in renders_dir.glob("*.png"):
        stem = png.stem
        if (renders_dir / f"{stem}.depth.npy").exists() and (renders_dir / f"{stem}.alpha.npy").exists():
            stems.add(stem)
    return stems


def paired_names(renders_dir: str | Path, photo_names: list[str]) -> list[str]:
    """Photo names (from `photo_names`) that have a complete render triple in `renders_dir`.

    Pairing key is `Path(name).stem` (see module docstring). Returns a sorted list.
    """
    stems = output_stems(renders_dir)
    return sorted(name for name in photo_names if Path(name).stem in stems)


def load_render_arrays(renders_dir: str | Path, stem: str) -> dict[str, np.ndarray]:
    """Load one frame's render triple as float32 numpy arrays.

    Returns:
        {"rgb": (H, W, 3) float32 in [0, 1], "alpha": (H, W) float32, "depth": (H, W) float32}.
    """
    renders_dir = Path(renders_dir)
    with Image.open(renders_dir / f"{stem}.png") as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    alpha = np.load(renders_dir / f"{stem}.alpha.npy").astype(np.float32)
    depth = np.load(renders_dir / f"{stem}.depth.npy").astype(np.float32)
    return {"rgb": rgb, "alpha": alpha, "depth": depth}


def render_to_tensor(arrays: dict[str, np.ndarray], channels: int = HYBRID_C_DEFAULT_CHANNELS) -> torch.Tensor:
    """Stack a loaded render triple into the U-Net's `(channels, H, W)` input tensor.

    Args:
        arrays: as returned by `load_render_arrays`.
        channels: `HYBRID_C_DEFAULT_CHANNELS` (4, rgb+alpha) or
            `HYBRID_C_CHANNELS_WITH_DEPTH` (5, + a normalised depth channel).

    Returns:
        float32 `(channels, H, W)` tensor, channel order `[r, g, b, alpha, (depth)]`.
    """
    if channels not in (HYBRID_C_DEFAULT_CHANNELS, HYBRID_C_CHANNELS_WITH_DEPTH):
        raise ValueError(f"channels must be {HYBRID_C_DEFAULT_CHANNELS} or {HYBRID_C_CHANNELS_WITH_DEPTH}, got {channels}")
    rgb = torch.from_numpy(arrays["rgb"]).permute(2, 0, 1)  # (3, H, W)
    alpha = torch.from_numpy(arrays["alpha"]).unsqueeze(0)  # (1, H, W)
    parts = [rgb, alpha]
    if channels == HYBRID_C_CHANNELS_WITH_DEPTH:
        depth_norm = torch.from_numpy(arrays["depth"]).unsqueeze(0) / HYBRID_C_DEPTH_NORM_SCALE
        parts.append(depth_norm)
    return torch.cat(parts, dim=0).to(torch.float32)


def photo_to_tensor(rgb_uint8: torch.Tensor | np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 photo -> (3, H, W) float32 in [0, 1]."""
    if isinstance(rgb_uint8, np.ndarray):
        rgb_uint8 = torch.from_numpy(rgb_uint8)
    return rgb_uint8.to(torch.float32).permute(2, 0, 1) / 255.0


def build_pyramid(x: torch.Tensor, num_layers: int) -> list[torch.Tensor]:
    """Average-pool `x` (C, H, W) into `num_layers` levels, finest first.

    Level 0 is `x` itself; level `i` is `F.avg_pool2d(level_{i-1}, kernel_size=2, stride=2)`
    (floor division on odd sizes, exactly `torch.nn.functional.avg_pool2d`'s own behaviour --
    trippy.net.unet's `combine_bridge` already generalizes to the resulting odd-size chain,
    see that module's docstring "CombineBridge / odd-size handling").

    Args:
        x: (C, H, W) float tensor (no batch dim).
        num_layers: number of pyramid levels to produce (>= 1).

    Returns:
        List of `num_layers` (C, h_i, w_i) tensors, `h_0, w_0 == x`'s own (H, W).

    Raises:
        ValueError: `num_layers < 1`, or a level would need to pool a spatial dim already < 2.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if x.dim() != 3:
        raise ValueError(f"x must be (C, H, W), got shape {tuple(x.shape)}")

    levels = [x]
    for i in range(1, num_layers):
        prev = levels[-1]
        if prev.shape[-2] < 2 or prev.shape[-1] < 2:
            raise ValueError(
                f"cannot build level {i}/{num_layers - 1}: level {i - 1} is already "
                f"{tuple(prev.shape[-2:])}, too small to halve"
            )
        levels.append(F.avg_pool2d(prev.unsqueeze(0), kernel_size=2, stride=2).squeeze(0))
    return levels


def crop_pair(
    render: torch.Tensor,
    photo: torch.Tensor,
    size: int,
    y0: int,
    x0: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice matching `size` x `size` windows from `render` (C, H, W) and `photo` (3, H, W).

    Both tensors must already share the same (H, W) -- true by construction for a
    render/photo pair built from the same `SceneDataset` grid (see module docstring). No
    padding/overshoot handling: the caller is responsible for choosing `y0, x0` such that the
    window lies fully inside both tensors (see `sample_crop_origin`).
    """
    render_crop = render[:, y0 : y0 + size, x0 : x0 + size]
    photo_crop = photo[:, y0 : y0 + size, x0 : x0 + size]
    if render_crop.shape[-2:] != (size, size) or photo_crop.shape[-2:] != (size, size):
        raise ValueError(
            f"crop window (y0={y0}, x0={x0}, size={size}) does not fit inside "
            f"render {tuple(render.shape[-2:])} / photo {tuple(photo.shape[-2:])}"
        )
    return render_crop, photo_crop


def sample_crop_origin(height: int, width: int, size: int, generator: torch.Generator) -> tuple[int, int]:
    """Uniform-random top-left (y0, x0) for a `size` x `size` window fully inside (height, width)."""
    if height < size or width < size:
        raise ValueError(f"frame ({height}, {width}) is smaller than crop size {size}")
    y0 = int(torch.randint(0, height - size + 1, (), generator=generator).item())
    x0 = int(torch.randint(0, width - size + 1, (), generator=generator).item())
    return y0, x0
