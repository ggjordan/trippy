"""GaussianInputs: turn Gaussian-splat renders into extra U-Net input channels (design A).

Module: trippy.hybrid.gaussian_input
Purpose: the whole of design A's data path. Given a `HybridConfig` and a
    directory of `trippy.hybrid.render_splat_views` triples, this module
    (a) loads/normalises a frame's render into one `(G, H, W)` block,
    (b) crops that block **through the exact same function and arguments the
    photo crop uses** so the K-adjust can never drift, and (c) concatenates the
    block onto every level of the TRIPS pyramid before the U-Net sees it.
Invariants:
    - Crop equivalence: `crop_frame` calls `trippy.scene.dataset.crop` with the
      identical `(size, zoom, center)` the trainer passed for the photo, on a
      dict carrying the *same* `K`. That function's K-adjust and validity mask
      are therefore identical by construction, not by parallel implementation.
      Depth is a camera-space z in world units: a crop/zoom changes the
      intrinsics, not the metric distance to the surface, so depth values are
      resampled (nearest, with the photo's own gather) and never rescaled.
      Their only rescale is the scene-global `depth_scale` divisor.
    - Channel order inside the block is always `HYBRID_A_CHANNEL_ORDER`
      filtered by `cfg.channels` (`HybridConfig` canonicalises it), and the
      block is always appended *after* the TRIPS feature channels, so a level
      is `[trips features (C) | gaussian block (G)]`.
    - A frame with no render triple, and a crop selected for Gaussian dropout,
      are the same thing to the network: an all-zero Gaussian block of the
      right width. The level count and channel count never vary within a run.
    - Nothing here touches MPS or `~/Splats`: it only reads png/npy files that
      `render_splat_views` already wrote. Live rendering at unphotographed
      poses lives in `trippy.hybrid.gsrender_live`.
    - Blocks stay on the CPU until the last possible moment: `attach` pools to
      each pyramid level's size on the block's own (CPU) device and copies only
      the finished level across. The pooling is a few milliseconds either way,
      and doing it host-side means design A depends on no additional Metal
      kernel -- which matters because `PYTORCH_ENABLE_MPS_FALLBACK=0` is
      enforced in queue jobs, so an unimplemented op is a hard failure, and
      there is no way to test the MPS path outside the queue (AGENTS.md).
Units: rgb and alpha are unitless in [0, 1]; the depth channel is
    `gsrender depth (world units) / cfg.depth_scale (world units)`, unitless.
Related docs: docs/EXPERIMENTS.md "Hybrid design A"; docs/ARCHITECTURE.md
    "hybrid/"; trippy.hybrid.dataset_c (the design-C loader whose on-disk
    layout this module reuses verbatim).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from trippy.constants import (
    HYBRID_A_DEPTH_SCALE_ALPHA_MIN,
    HYBRID_A_DEPTH_SCALE_FALLBACK,
    HYBRID_A_DEPTH_SCALE_FRAMES,
    HYBRID_A_RENDER_CACHE_FRAMES,
)
from trippy.hybrid.config_a import HybridConfig
from trippy.hybrid.dataset_c import load_render_arrays, output_stems
from trippy.scene.dataset import crop as dataset_crop


def stem_of(name: str) -> str:
    """Pairing key between a photo name and its render triple (`IMG_3830.jpg` -> `IMG_3830`)."""
    return Path(name).stem


def block_from_arrays(
    arrays: dict[str, np.ndarray],
    channels: list[str],
    depth_scale: float,
    mask_by_alpha: bool,
) -> torch.Tensor:
    """Stack one loaded render triple into the `(G, H, W)` Gaussian block.

    Args:
        arrays: `{"rgb": (H, W, 3), "alpha": (H, W), "depth": (H, W)}` float
            arrays (as `trippy.hybrid.dataset_c.load_render_arrays` returns);
            torch tensors are accepted too so a live renderer can hand its
            output straight over.
        channels: which groups to include, already canonicalised by
            `HybridConfig`.
        depth_scale: world-unit divisor for the depth channel (must be > 0).
        mask_by_alpha: multiply rgb by alpha before stacking.

    Returns:
        float32 `(G, H, W)` tensor, `G == gaussian_channel_count(channels)`.
    """
    if depth_scale <= 0.0:
        raise ValueError(f"depth_scale must be positive, got {depth_scale}")

    def _as_tensor(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.detach().to(torch.float32)
        return torch.as_tensor(np.asarray(x), dtype=torch.float32)

    rgb = _as_tensor(arrays["rgb"])
    if rgb.shape[-1] == 3 and rgb.dim() == 3:
        rgb = rgb.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
    alpha = _as_tensor(arrays["alpha"]).reshape(1, *rgb.shape[-2:])

    parts: list[torch.Tensor] = []
    for group in channels:
        if group == "rgb":
            parts.append(rgb * alpha if mask_by_alpha else rgb)
        elif group == "alpha":
            parts.append(alpha)
        else:  # "depth"
            depth = _as_tensor(arrays["depth"]).reshape(1, *rgb.shape[-2:])
            parts.append(depth / depth_scale)
    return torch.cat(parts, dim=0).to(torch.float32)


def measure_depth_scale(
    renders_dir: str | Path,
    names: list[str],
    alpha_min: float = HYBRID_A_DEPTH_SCALE_ALPHA_MIN,
    max_frames: int = HYBRID_A_DEPTH_SCALE_FRAMES,
    fallback: float = HYBRID_A_DEPTH_SCALE_FALLBACK,
) -> float:
    """Median camera-to-Gaussian depth over a sample of rendered frames, in world units.

    This is the scene scale the depth channel is divided by so it reaches the
    network unitless. Only pixels the Gaussian cloud actually covered
    (`alpha >= alpha_min`) are counted: gsrender writes depth 0 where nothing
    was hit, so an unmasked median would mostly measure holes.

    Args:
        renders_dir: directory of render triples.
        names: photo names (or stems) to sample from, in dataset order;
            `max_frames` are taken evenly spaced across the list.
        alpha_min: coverage threshold.
        max_frames: cap on frames read.
        fallback: returned when no sampled frame has a single covered pixel.

    Returns:
        A positive float in world units.
    """
    renders_dir = Path(renders_dir)
    available = output_stems(renders_dir)
    stems = [stem_of(n) for n in names if stem_of(n) in available]
    if not stems:
        return fallback
    step = max(1, len(stems) // max(1, max_frames))
    sampled = stems[::step][:max_frames]

    values: list[np.ndarray] = []
    for stem in sampled:
        arrays = load_render_arrays(renders_dir, stem)
        covered = arrays["alpha"] >= alpha_min
        if bool(covered.any()):
            values.append(arrays["depth"][covered])
    if not values:
        return fallback
    median = float(np.median(np.concatenate(values)))
    return median if median > 0.0 else fallback


def resample_to(x: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    """Resample `(G, H, W)` (or `(B, G, H, W)`) to `image_hw`, area-averaging when shrinking.

    Design A's renders may have been produced at a different width than the
    training run's `SceneDataset` grid (e.g. the w1008 render set reused by a
    504-wide smoke run). Both grids are the same undistorted pinhole camera
    with proportional intrinsics, so a plain resample lands pixel-for-pixel;
    "area" is used when both axes shrink because that is exactly the average
    pooling the pyramid itself uses, and bilinear otherwise.
    """
    batched = x.dim() == 4
    src = x if batched else x.unsqueeze(0)
    height, width = int(image_hw[0]), int(image_hw[1])
    if src.shape[-2] == height and src.shape[-1] == width:
        return x
    if src.shape[-2] >= height and src.shape[-1] >= width:
        out = F.interpolate(src, size=(height, width), mode="area")
    else:
        out = F.interpolate(src, size=(height, width), mode="bilinear", align_corners=False)
    return out if batched else out.squeeze(0)


class GaussianInputs:
    """Loads, crops, pools and concatenates the Gaussian block for one training run.

    Construct via `GaussianInputs.build` (which also resolves `depth_scale`).
    Instances hold a small bounded LRU of decoded frames
    (`HYBRID_A_RENDER_CACHE_FRAMES`) so a long run's memory stays flat.
    Everything here returns CPU tensors; `attach` is what copies each pooled
    level onto the network's device (see module docstring).
    """

    def __init__(self, cfg: HybridConfig) -> None:
        if cfg.depth_scale is None and cfg.wants_depth:
            raise ValueError("GaussianInputs needs a resolved cfg.depth_scale; use GaussianInputs.build")
        self.cfg = cfg
        self.renders_dir = Path(cfg.renders_dir)
        self._available = output_stems(self.renders_dir) if self.renders_dir.exists() else set()
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    # --- construction ---

    @classmethod
    def build(cls, cfg: HybridConfig, names: list[str]) -> GaussianInputs:
        """Resolve `cfg.depth_scale` (in place, so the checkpoint records it) and build.

        There is no `device` argument on purpose: every block this class
        produces stays on the CPU, and `attach` is the one place that copies a
        pooled level onto the network's device (see the module docstring).

        Args:
            cfg: the run's `hybrid:` block. Mutated only to fill in a `None`
                `depth_scale` with the measured value.
            names: the dataset's image names, used to sample frames for the
                depth measurement.
        """
        if cfg.wants_depth and cfg.depth_scale is None:
            cfg.depth_scale = measure_depth_scale(cfg.renders_dir, names)
        return cls(cfg)

    # --- properties ---

    @property
    def num_channels(self) -> int:
        """Width of the Gaussian block (see `HybridConfig.num_channels`)."""
        return self.cfg.num_channels

    def has(self, name: str) -> bool:
        """True when `name`'s render triple exists on disk."""
        return stem_of(name) in self._available

    def available_names(self, names: list[str]) -> list[str]:
        """Subset of `names` that have a render triple (for logging a coverage number)."""
        return [n for n in names if self.has(n)]

    # --- frame access ---

    def _cached_block(self, name: str) -> torch.Tensor | None:
        """Decoded `(G, H, W)` block for `name` at the render's own resolution, or None."""
        stem = stem_of(name)
        if stem not in self._available:
            if self.cfg.missing == "error":
                raise FileNotFoundError(
                    f"no render triple for {name!r} (stem {stem!r}) under {self.renders_dir} "
                    "-- set hybrid.missing: zeros to train through it"
                )
            return None
        cached = self._cache.get(stem)
        if cached is not None:
            self._cache.move_to_end(stem)
            return cached
        block = block_from_arrays(
            load_render_arrays(self.renders_dir, stem),
            self.cfg.channels,
            float(self.cfg.depth_scale or HYBRID_A_DEPTH_SCALE_FALLBACK),
            self.cfg.mask_by_alpha,
        )
        self._cache[stem] = block
        while len(self._cache) > HYBRID_A_RENDER_CACHE_FRAMES:
            self._cache.popitem(last=False)
        return block

    def frame(self, name: str, image_hw: tuple[int, int]) -> torch.Tensor | None:
        """Full-frame `(G, H, W)` block for `name`, resampled to `image_hw` (on the CPU).

        Returns None when no render exists and `cfg.missing == "zeros"` (the
        caller then gets an all-zero block from `attach`).
        """
        block = self._cached_block(name)
        if block is None:
            return None
        return resample_to(block, image_hw)

    def crop_frame(
        self,
        name: str,
        item: dict[str, Any],
        size: int,
        zoom: float,
        center: tuple[float, float],
    ) -> torch.Tensor | None:
        """The Gaussian block cropped **exactly like the photo** (see module docstring).

        Args:
            name: photo name (pairing key).
            item: the `SceneDataset` item the photo crop was taken from --
                supplies the source `(H, W)` grid and the `K` whose adjust must
                match. Never read for pixels.
            size, zoom, center: the identical arguments handed to
                `trippy.scene.dataset.crop` for the photo.

        Returns:
            float32 `(G, size, size)` on the CPU, or None when the frame has no
            render.
        """
        photo_hw = (int(item["rgb"].shape[0]), int(item["rgb"].shape[1]))
        block = self._cached_block(name)
        if block is None:
            return None
        block = resample_to(block, photo_hw)
        # `dataset_crop` gathers on the last two dims of a channels-last image and
        # multiplies by the validity mask in the image's own dtype; feeding it the
        # (H, W, G) float32 block reuses that code verbatim -- same window, same mask,
        # same K-adjust as the photo.
        cropped = dataset_crop(
            {"rgb": block.permute(1, 2, 0).contiguous(), "K": item["K"]},
            size=size,
            zoom=zoom,
            center=center,
        )
        return cropped["rgb"].permute(2, 0, 1).contiguous()

    # --- ablation 1: dropout ---

    def should_drop(self, generator: torch.Generator | None = None) -> bool:
        """True with probability `cfg.dropout_gaussian_p` (drawn from `generator`)."""
        probability = self.cfg.dropout_gaussian_p
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        return bool(torch.rand((), generator=generator).item() < probability)

    # --- the network hook ---

    def attach(
        self, inputs: list[torch.Tensor], gaussian: torch.Tensor | None
    ) -> list[torch.Tensor]:
        """Concatenate the Gaussian block onto every U-Net input level.

        Args:
            inputs: the TRIPS pyramid, finest first, each `(B, C, h_i, w_i)`
                (exactly what `trippy.train.trainer.Trainer._render` builds).
            gaussian: `(G, H, W)` or `(B, G, H, W)` block at level-0
                resolution, on any device (CPU is the normal case -- see the
                module docstring on why the pooling is host-side), or None for
                "no Gaussian information here" (a missing render, or a
                dropped-out crop).

        Returns:
            A new list of `(B, C + G, h_i, w_i)` tensors. `cfg.mode ==
            "concat_level0"` puts the real block on level 0 only and zeros on
            the coarser levels; `"all_levels"` area-averages it down to every
            level's own size.
        """
        if not inputs:
            return inputs
        block = gaussian
        if block is not None and block.dim() == 3:
            block = block.unsqueeze(0)
        if block is not None:
            block = block.to(dtype=inputs[0].dtype)  # device deliberately unchanged

        out: list[torch.Tensor] = []
        for level, x in enumerate(inputs):
            hw = (x.shape[-2], x.shape[-1])
            if block is None or (level > 0 and self.cfg.mode == "concat_level0"):
                extra = x.new_zeros((x.shape[0], self.num_channels, hw[0], hw[1]))
            else:
                pooled = resample_to(block, hw).to(x.device)
                extra = pooled.expand(x.shape[0], -1, -1, -1)
            out.append(torch.cat([x, extra], dim=1))
        return out
