"""Pure-torch reference rasteriser: float64, CPU, differentiable.

Module: trippy.raster.ref_torch
Purpose: the twin of the Metal kernel. It runs the identical algorithm --
    same layer selection, same footprint, same drop rules, same 16-fragment
    cap, same transmittance cutoff -- in plain torch, so (a) CPU tests and
    gradcheck need no GPU and (b) `blend_fwd`'s output can be diffed against
    something readable. AGENTS.md review checklist: "Metal kernels: CPU twin
    exists + comparison test passes".
Invariants:
    - Correctness over speed. The compositing is vectorised (segment-wise
      prefix products via cumulative log1p) rather than looped, but the
      result is the sequential front-to-back recurrence exactly, because the
      per-pixel "stop" rules (fragment cap, transmittance cutoff) both cut a
      *prefix* of an already depth-sorted segment.
    - float64 by default: the whole point is to be more accurate than the
      float32 GPU path it validates.
    - Differentiable in alpha (hence in uv, size, conf), in feat/bg and
      in the optional SE(3) pose delta.
      Fragment *ordering* is a discrete function of the inputs and carries no
      gradient, exactly as in TRIPS.
Units / frames: see trippy.raster.emit (world-frame xyz, world-unit size,
    layer-0 pixel uv, camera-space depth).
Related docs: docs/ARCHITECTURE.md; docs/TRIPS_REFERENCE.md section 3.
"""

from __future__ import annotations

import torch
from torch import Tensor

from trippy.constants import (
    RASTER_ALPHA_MAX_EPS,
    RASTER_ALPHA_MIN,
    RASTER_MAX_FRAGS,
    RASTER_NUM_LAYERS,
    RASTER_T_CUTOFF,
    RASTER_ZNEAR,
)
from trippy.raster.emit import LayerGrid, build_sorted_fragments, layer_grid
from trippy.raster.sort import fragment_rank


def composite_sorted(
    layer_pixel: Tensor,
    depth: Tensor,
    point_id: Tensor,
    alpha: Tensor,
    offsets: Tensor,
    feat: Tensor,
    max_frags: int = RASTER_MAX_FRAGS,
    t_cutoff: float = RASTER_T_CUTOFF,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Front-to-back alpha compositing of depth-sorted fragments, in torch.

    The torch twin of metal_src/blend_fwd.metal. For each layer-pixel p, walk
    its segment `[offsets[p], offsets[p+1])` in order:
    `out += T * alpha * feat[pid]; T *= (1 - alpha)`, stopping before the
    fragment at which either `T < t_cutoff` or `max_frags` fragments have
    already been used.

    Args:
        layer_pixel: (F,) int64, non-decreasing flat layer-pixel index.
        depth: (F,) float, camera-space z, world units (sorted within segment).
        point_id: (F,) int64, row index into `feat`.
        alpha: (F,) float in (0, 1), differentiable.
        offsets: (P + 1,) int64 segment starts.
        feat: (N, C) float point features.
        max_frags: per-pixel fragment cap.
        t_cutoff: transmittance stop threshold.

    Returns:
        out: (P, C), sum of T * alpha * feat (no background).
        t_final: (P,), transmittance left after the composited prefix.
        n_used: (P,) int64, fragments composited.
        depth_sum: (P,), sum of T * alpha * depth.
    """
    num_pixels = int(offsets.shape[0]) - 1
    num_channels = int(feat.shape[1])
    dtype = feat.dtype
    device = feat.device

    out = torch.zeros((num_pixels, num_channels), dtype=dtype, device=device)
    depth_sum = torch.zeros(num_pixels, dtype=dtype, device=device)
    n_used = torch.zeros(num_pixels, dtype=torch.int64, device=device)
    log_kept = torch.zeros(num_pixels, dtype=dtype, device=device)

    if layer_pixel.numel() > 0:
        # log(1 - alpha) is clamped away from log(0); alpha == 1 would make the
        # rest of the segment invisible anyway.
        alpha_c = torch.clamp(alpha, max=1.0 - RASTER_ALPHA_MAX_EPS)
        log1m = torch.log1p(-alpha_c)
        # Exclusive prefix sum over the flat list, then rebased per segment.
        # The running sum spans every fragment in the image, so it is taken in
        # float64 (where available) even when the render dtype is float32:
        # subtracting two large partial sums in float32 would destroy the
        # per-segment transmittance.
        acc_dtype = torch.float32 if device.type == "mps" else torch.float64
        log1m_acc = log1m.to(acc_dtype)
        inclusive = torch.cumsum(log1m_acc, dim=0)
        exclusive = inclusive - log1m_acc
        segment_start = offsets.index_select(0, layer_pixel)
        transmittance = torch.exp(exclusive - exclusive.index_select(0, segment_start)).to(dtype)

        rank = fragment_rank(layer_pixel, offsets)
        # Both stop rules cut a prefix of the segment, so a per-fragment mask
        # reproduces the sequential `break` exactly.
        keep = (rank < max_frags) & (transmittance >= t_cutoff)
        keep_f = keep.to(dtype)

        weight = transmittance * alpha * keep_f
        out = out.index_add(0, layer_pixel, weight.reshape(-1, 1) * feat.index_select(0, point_id))
        depth_sum = depth_sum.index_add(0, layer_pixel, weight * depth)
        n_used = n_used.index_add(0, layer_pixel, keep.to(torch.int64))
        log_kept = log_kept.index_add(0, layer_pixel, log1m * keep_f)

    return out, torch.exp(log_kept), n_used, depth_sum


def render_pyramid_ref(
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
    compute_dtype: torch.dtype | None = torch.float64,
    pose_delta: Tensor | None = None,
) -> tuple[list[Tensor], dict]:
    """Render the whole pyramid in torch (the float64 CPU reference path).

    Args:
        xyz: (N, 3) float, COLMAP world-frame positions, world units.
        size: (N,) float, effective (post-softplus) point radius, world units.
        feat: (N, C) float, per-point features (linear RGB or learned).
        conf: (N,) float in (0, 1), effective (post-sigmoid) confidence.
        K: (3, 3) float, layer-0 pinhole intrinsics, pixels.
        R: (3, 3) float world->camera rotation; t: (3,) translation.
        image_hw: (H, W) layer-0 image size in pixels.
        num_layers: L pyramid layers.
        mode: "trilinear" (footprint-weighted, <= 2 layers) or "broadcast"
            (every layer, factor 1 -- TRIPS's shipped default).
        bg: (C,) float background feature, added as `t_final * bg`. None = 0.
        max_frags, t_cutoff, alpha_min, znear: see trippy.constants.
        sort_method, segment_method: see trippy.raster.sort.
        compute_dtype: dtype the whole render runs in; None keeps the input
            dtype. Default float64 (this is the reference).
        pose_delta: optional (6,) SE(3) twist refining `(R, t)`
            left-multiplicatively (trippy.raster.emit.apply_pose_delta).
            Differentiable, so `pose_delta.grad` is the camera-pose gradient.

    Returns:
        layers: list of L tensors, layer l is (C, h_l, w_l) with
            h_l = ceil(H / 2**l).
        aux: {"t_final": list of L (h_l, w_l), "n_used": list of L (h_l, w_l)
            int64, "depth_sum": list of L (h_l, w_l), "num_fragments": int,
            "grid": LayerGrid}.
    """
    height, width = int(image_hw[0]), int(image_hw[1])
    grid = layer_grid(height, width, num_layers)
    dtype = feat.dtype if compute_dtype is None else compute_dtype

    xyz_d, size_d, feat_d, conf_d = (a.to(dtype) for a in (xyz, size, feat, conf))
    K_d, R_d, t_d = (a.to(dtype) for a in (K, R, t))
    delta_d = None if pose_delta is None else pose_delta.to(dtype)

    frags = build_sorted_fragments(
        xyz_d,
        size_d,
        conf_d,
        K_d,
        R_d,
        t_d,
        grid,
        mode=mode,
        alpha_min=alpha_min,
        znear=znear,
        sort_method=sort_method,
        segment_method=segment_method,
        pose_delta=delta_d,
    )
    out, t_final, n_used, depth_sum = composite_sorted(
        frags.layer_pixel,
        frags.depth,
        frags.point_id,
        frags.alpha,
        frags.offsets,
        feat_d,
        max_frags=max_frags,
        t_cutoff=t_cutoff,
    )
    if bg is not None:
        out = out + t_final.reshape(-1, 1) * bg.to(dtype).reshape(1, -1)

    layers, aux = split_layers(out, t_final, n_used, depth_sum, grid)
    aux["num_fragments"] = len(frags)
    return layers, aux


def split_layers(
    out: Tensor,
    t_final: Tensor,
    n_used: Tensor,
    depth_sum: Tensor,
    grid: LayerGrid,
) -> tuple[list[Tensor], dict]:
    """Reshape flat (P, ...) compositing results into per-layer image tensors.

    Args:
        out: (P, C); t_final, depth_sum: (P,); n_used: (P,) integer.
        grid: LayerGrid describing the flat index space.

    Returns:
        (layers, aux) as documented on render_pyramid_ref. `layers[l]` is
        (C, h_l, w_l) -- channel-first, matching torch.nn.Conv2d's NCHW.
    """
    layers: list[Tensor] = []
    aux: dict = {"t_final": [], "n_used": [], "depth_sum": [], "grid": grid}
    for layer, (h_l, w_l) in enumerate(grid.shapes):
        lo = grid.offsets[layer]
        hi = lo + h_l * w_l
        layers.append(out[lo:hi].reshape(h_l, w_l, -1).permute(2, 0, 1).contiguous())
        aux["t_final"].append(t_final[lo:hi].reshape(h_l, w_l))
        aux["n_used"].append(n_used[lo:hi].reshape(h_l, w_l))
        aux["depth_sum"].append(depth_sum[lo:hi].reshape(h_l, w_l))
    return layers, aux
