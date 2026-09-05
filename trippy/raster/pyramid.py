"""Pyramid rasteriser forward pass: device dispatch and the public entry point.

Module: trippy.raster.pyramid
Purpose: `render_pyramid()` -- project a point set into an L-layer image
    pyramid and alpha-composite it, running the compositing step in Metal on
    MPS and in torch on CPU. Everything before compositing (projection,
    layer selection, footprint emission, sorting, segmentation) is the same
    vectorised torch code on both devices, so the only device-dependent
    numerics are float32-vs-float64 and the compositing kernel itself.
Invariants:
    - CPU dispatches to trippy.raster.ref_torch (no Metal, no MPS), so CPU
      tests and gradcheck run on any machine.
    - MPS dispatches to trippy.raster.metal_lib.blend_fwd. That kernel is
      forward-only: the MPS path returns tensors detached from autograd until
      blend_bwd lands (docs/LIMITATIONS.md, v0.2.0 milestone).
    - Background is added in torch, never in the kernel:
      `out += t_final * bg` (TRIPS: RenderForward.cu:3610-3620).
    - No silent fallback: an unsupported device raises.
Units / frames: `xyz` COLMAP world frame in world units; `size` world units;
    `K` layer-0 pixels; depth is camera-space z (positive in front).
Related docs: docs/ARCHITECTURE.md (forward data flow, no-atomics redesign);
    docs/GEOMETRY.md (pixel-centre convention, pyramid, layer selection);
    docs/TRIPS_REFERENCE.md sections 3, 10, 11.
"""

from __future__ import annotations

import torch
from torch import Tensor

from trippy.constants import (
    RASTER_ALPHA_MIN,
    RASTER_MAX_FRAGS,
    RASTER_NUM_LAYERS,
    RASTER_T_CUTOFF,
    RASTER_ZNEAR,
)
from trippy.raster import metal_lib
from trippy.raster.emit import build_sorted_fragments, layer_grid
from trippy.raster.ref_torch import render_pyramid_ref, split_layers


def render_pyramid(
    xyz: Tensor,
    size: Tensor,
    feat: Tensor,
    conf: Tensor,
    K: Tensor,
    R: Tensor,
    t: Tensor,
    image_hw: tuple[int, int],
    num_layers: int = RASTER_NUM_LAYERS,
    mode: str = "trilinear",
    bg: Tensor | None = None,
    max_frags: int = RASTER_MAX_FRAGS,
    t_cutoff: float = RASTER_T_CUTOFF,
    alpha_min: float = RASTER_ALPHA_MIN,
    znear: float = RASTER_ZNEAR,
    sort_method: str = "composite",
    segment_method: str = "searchsorted",
    compute_dtype: torch.dtype | None = None,
) -> tuple[list[Tensor], dict]:
    """Render one image as an L-layer alpha-composited pyramid.

    Args:
        xyz: (N, 3) float, COLMAP world-frame positions, world units.
        size: (N,) float, effective (post-softplus) point radius, world units.
            The trainer owns the raw parameter.
        feat: (N, C) float, per-point features. C must be in
            trippy.constants.RASTER_SUPPORTED_CHANNELS on MPS.
        conf: (N,) float in (0, 1), effective (post-sigmoid) confidence.
        K: (3, 3) float, layer-0 pinhole intrinsics in pixels.
        R: (3, 3) float, world->camera rotation (x_cam = R @ x_world + t).
        t: (3,) float, world->camera translation, world units.
        image_hw: (H, W) layer-0 image size in pixels.
        num_layers: L, pyramid layers (layer l is ceil(H / 2**l) tall).
        mode: "trilinear" -- each point writes into the (at most two) layers
            its projected size straddles, weighted by TRIPS's layer factor;
            or "broadcast" -- every point writes into every layer with factor
            1, TRIPS's shipped default (docs/TRIPS_REFERENCE.md section 10.1).
        bg: (C,) float background feature, composited as `t_final * bg`.
            None means a black/zero background.
        max_frags: fragments composited per layer-pixel (TRIPS: 16).
        t_cutoff: stop compositing a pixel once transmittance drops below
            this (TRIPS ALPHA_DEST_CUTOFF = 0.001).
        alpha_min: drop fragments below this alpha at emission time.
        znear: cull points with camera-space z <= this, world units.
        sort_method: "composite" or "two_pass" (trippy.raster.sort).
        segment_method: "searchsorted" or "bincount".
        compute_dtype: CPU only -- dtype the reference path computes in;
            None keeps the input dtype, torch.float64 gives the reference
            precision used by tests.

    Returns:
        layers: list of L tensors; layer l is (C, h_l, w_l), channel-first.
        aux: {"t_final": list of L (h_l, w_l) float -- 1 means "nothing was
            drawn here", the coverage/honesty map;
            "n_used": list of L (h_l, w_l) int -- fragments composited, equal
            to max_frags where the 16-deep list overflowed;
            "depth_sum": list of L (h_l, w_l) float -- sum of T * alpha *
            depth, divide by (1 - t_final) for an expected depth in world
            units; "num_fragments": int; "grid": LayerGrid}.

    Raises:
        ValueError: on an unsupported device or malformed inputs.
    """
    device = feat.device
    if device.type == "cpu":
        return render_pyramid_ref(
            xyz,
            size,
            feat,
            conf,
            K,
            R,
            t,
            image_hw,
            num_layers=num_layers,
            mode=mode,
            bg=bg,
            max_frags=max_frags,
            t_cutoff=t_cutoff,
            alpha_min=alpha_min,
            znear=znear,
            sort_method=sort_method,
            segment_method=segment_method,
            compute_dtype=compute_dtype,
        )
    if device.type != "mps":
        raise ValueError(f"render_pyramid supports device 'cpu' and 'mps', got {device.type!r}")
    if compute_dtype not in (None, torch.float32):
        raise ValueError(f"MPS renders in float32; compute_dtype={compute_dtype} is unsupported")
    return _render_pyramid_mps(
        xyz,
        size,
        feat,
        conf,
        K,
        R,
        t,
        image_hw,
        num_layers=num_layers,
        mode=mode,
        bg=bg,
        max_frags=max_frags,
        t_cutoff=t_cutoff,
        alpha_min=alpha_min,
        znear=znear,
        sort_method=sort_method,
        segment_method=segment_method,
    )


def _render_pyramid_mps(
    xyz: Tensor,
    size: Tensor,
    feat: Tensor,
    conf: Tensor,
    K: Tensor,
    R: Tensor,
    t: Tensor,
    image_hw: tuple[int, int],
    num_layers: int,
    mode: str,
    bg: Tensor | None,
    max_frags: int,
    t_cutoff: float,
    alpha_min: float,
    znear: float,
    sort_method: str,
    segment_method: str,
) -> tuple[list[Tensor], dict]:
    """MPS path: torch emission/sort + the Metal blend_fwd kernel.

    Forward only; the returned tensors are detached (see module docstring).
    Buffers are made contiguous and cast to the kernel's exact dtypes here,
    because torch.mps.compile_shader binds raw storage without checking.
    """
    height, width = int(image_hw[0]), int(image_hw[1])
    grid = layer_grid(height, width, num_layers)
    frags = build_sorted_fragments(
        xyz.to(torch.float32),
        size.to(torch.float32),
        conf.to(torch.float32),
        K.to(torch.float32),
        R.to(torch.float32),
        t.to(torch.float32),
        grid,
        mode=mode,
        alpha_min=alpha_min,
        znear=znear,
        sort_method=sort_method,
        segment_method=segment_method,
    )
    out, t_final, n_used, depth_sum = metal_lib.blend_fwd(
        frags.offsets.to(torch.int32).contiguous(),
        frags.point_id.to(torch.int32).contiguous(),
        frags.alpha.detach().to(torch.float32).contiguous(),
        frags.depth.detach().to(torch.float32).contiguous(),
        feat.detach().to(torch.float32).contiguous(),
        max_frags=max_frags,
        t_cutoff=t_cutoff,
    )
    if bg is not None:
        out = out + t_final.reshape(-1, 1) * bg.detach().to(torch.float32).reshape(1, -1)

    layers, aux = split_layers(out, t_final, n_used, depth_sum, grid)
    aux["num_fragments"] = len(frags)
    return layers, aux
