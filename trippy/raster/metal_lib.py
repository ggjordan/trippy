"""Host side of the Metal compositing kernel: compile, cache, dispatch.

Module: trippy.raster.metal_lib
Purpose: load trippy/raster/metal_src/blend_fwd.metal at import time, template
    it per (num_channels, max_frags, t_cutoff), compile it once per template
    with torch.mps.compile_shader, and dispatch it with fully checked buffers.
Invariants:
    - The .metal source is a real text file (not an embedded Python string) so
      it can be syntax-checked with `xcrun -sdk macosx metal -c` without a GPU
      and reviewed as Metal (AGENTS.md review checklist: "Metal kernel source
      files (*.metal) in trippy/raster/metal_src/").
    - Every buffer handed to the kernel is asserted contiguous and of the exact
      expected dtype: torch.mps.compile_shader binds raw storage, so a
      non-contiguous or mistyped tensor is read as garbage with no error.
    - Compiled libraries are cached per template; compiling is ~100 ms and
      would otherwise dominate a per-frame render.
    - This module never runs on CPU-only machines: importing it is safe, but
      blend_fwd() raises unless the tensors are on an MPS device.
Units / frames: see metal_src/blend_fwd.metal's header.
Related docs: docs/ARCHITECTURE.md; docs/LIMITATIONS.md (no 64-bit atomics on
    Metal); docs/TRIPS_REFERENCE.md section 3.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch
from torch import Tensor

from trippy.constants import RASTER_MAX_FRAGS, RASTER_SUPPORTED_CHANNELS, RASTER_T_CUTOFF

_METAL_SRC_PATH = Path(__file__).parent / "metal_src" / "blend_fwd.metal"
# Read once at import time; the file is part of the installed package.
BLEND_FWD_SOURCE = _METAL_SRC_PATH.read_text(encoding="utf-8")


def render_source(num_channels: int, max_frags: int, t_cutoff: float) -> str:
    """Substitute the compile-time constants into the .metal template.

    Args:
        num_channels: C, feature channels per point; must be one of
            trippy.constants.RASTER_SUPPORTED_CHANNELS.
        max_frags: per-layer-pixel fragment cap (TRIPS uses 16).
        t_cutoff: transmittance below which compositing stops.

    Returns:
        Metal source with TRIPPY_* tokens replaced.
    """
    if num_channels not in RASTER_SUPPORTED_CHANNELS:
        raise ValueError(
            f"num_channels must be one of {RASTER_SUPPORTED_CHANNELS}, got {num_channels}"
        )
    if max_frags < 1:
        raise ValueError(f"max_frags must be >= 1, got {max_frags}")
    return (
        BLEND_FWD_SOURCE.replace("TRIPPY_NUM_CHANNELS", str(int(num_channels)))
        .replace("TRIPPY_MAX_FRAGS", str(int(max_frags)))
        .replace("TRIPPY_T_CUTOFF", f"{float(t_cutoff):.9g}f")
    )


@functools.cache
def _compiled_library(num_channels: int, max_frags: int, t_cutoff: float):
    """Compile (and memoise) the blend_fwd library for one template."""
    if not hasattr(torch.mps, "compile_shader"):  # pragma: no cover (old torch)
        raise RuntimeError("torch.mps.compile_shader is unavailable; need torch >= 2.13 on macOS")
    return torch.mps.compile_shader(render_source(num_channels, max_frags, t_cutoff))


def clear_cache() -> None:
    """Drop cached compiled libraries (used by tests that vary the template)."""
    _compiled_library.cache_clear()


def _check(name: str, tensor: Tensor, dtype: torch.dtype, device: torch.device) -> None:
    """Assert one kernel buffer's dtype, contiguity and device."""
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def blend_fwd(
    offsets: Tensor,
    frag_point_id: Tensor,
    frag_alpha: Tensor,
    frag_depth: Tensor,
    feat: Tensor,
    max_frags: int = RASTER_MAX_FRAGS,
    t_cutoff: float = RASTER_T_CUTOFF,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Composite depth-sorted fragments into one value per layer-pixel.

    One Metal thread per layer-pixel; no atomics (see module docstring).

    Args:
        offsets: (P + 1,) int32 on MPS, contiguous. Segment starts from
            trippy.raster.sort.segment_offsets, cast to int32.
        frag_point_id: (F,) int32 on MPS, index into `feat`'s rows.
        frag_alpha: (F,) float32 on MPS, in (0, 1), depth-sorted.
        frag_depth: (F,) float32 on MPS, camera-space z in world units.
        feat: (N, C) float32 on MPS, contiguous point features.
        max_frags: per-pixel fragment cap, baked into the kernel.
        t_cutoff: transmittance stop threshold, baked into the kernel.

    Returns:
        out: (P, C) float32, sum of T * alpha * feature (background NOT added;
            the caller adds `t_final * bg` in torch).
        t_final: (P,) float32, transmittance left after the composited prefix.
        n_used: (P,) int32, number of fragments actually composited.
        depth_sum: (P,) float32, sum of T * alpha * depth -- divide by
            (1 - t_final) for an expected depth, or use directly as a
            coverage/honesty map numerator.
    """
    if feat.ndim != 2:
        raise ValueError(f"feat must be (N, C), got {tuple(feat.shape)}")
    device = feat.device
    if device.type != "mps":
        raise ValueError(f"blend_fwd needs MPS tensors, got device {device}")
    num_channels = int(feat.shape[1])
    num_pixels = int(offsets.shape[0]) - 1
    if num_pixels < 1:
        raise ValueError(f"offsets must have length P + 1 >= 2, got {offsets.shape[0]}")

    _check("offsets", offsets, torch.int32, device)
    _check("frag_point_id", frag_point_id, torch.int32, device)
    _check("frag_alpha", frag_alpha, torch.float32, device)
    _check("frag_depth", frag_depth, torch.float32, device)
    _check("feat", feat, torch.float32, device)
    if frag_point_id.shape != frag_alpha.shape or frag_alpha.shape != frag_depth.shape:
        raise ValueError("frag_point_id / frag_alpha / frag_depth must share shape (F,)")

    out = torch.zeros((num_pixels, num_channels), dtype=torch.float32, device=device)
    t_final = torch.ones(num_pixels, dtype=torch.float32, device=device)
    n_used = torch.zeros(num_pixels, dtype=torch.int32, device=device)
    depth_sum = torch.zeros(num_pixels, dtype=torch.float32, device=device)

    lib = _compiled_library(num_channels, int(max_frags), float(t_cutoff))
    lib.blend_fwd(
        out,
        t_final,
        n_used,
        depth_sum,
        offsets,
        frag_point_id,
        frag_alpha,
        frag_depth,
        feat,
        num_pixels,
        threads=[num_pixels],
    )
    return out, t_final, n_used, depth_sum
