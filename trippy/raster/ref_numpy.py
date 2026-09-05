"""Numpy reference rasteriser: explicit loops, no torch, no cleverness.

Module: trippy.raster.ref_numpy
Purpose: the independent second implementation of the forward rasteriser,
    written the obvious way -- per point, per layer, per 2x2 corner, then a
    per-pixel sorted list and a sequential blend loop. It exists to disagree
    with trippy.raster.ref_torch until both are right (AGENTS.md section 7:
    "implement transforms twice independently"), and it is the ground truth
    the vectorised torch path and the Metal kernel are measured against.
Invariants:
    - NO torch import (mirrors trippy.geom.xform_a), so this module can be
      run from a plain numpy script and cannot share a torch-specific bug
      with ref_torch.
    - Projection uses trippy.geom.xform_a (the numpy transform), not
      xform_b, so a projection bug has to be made twice to go unnoticed.
    - The layer-factor and bilinear formulas are re-derived here in scalar
      Python from the TRIPS source rather than imported from
      trippy.raster.emit, for the same reason.
    - float64 throughout; speed is irrelevant (this is 32x32-test code).
Units / frames: `xyz` COLMAP world frame, world units; `size` world units;
    `K` layer-0 pixels; depth is camera-space z, positive in front of the
    camera; `uv` is corner-origin continuous pixel coordinates (the centre of
    pixel i is at i + 0.5, docs/GEOMETRY.md).
Related docs: docs/TRIPS_REFERENCE.md sections 3, 3a and 10;
    third_party/TRIPS/src/lib/rendering/PointBlending.h:81-149 and
    RenderForward.cu:1610 (per-layer halving), 334-352 (mode "trips":
    layers 0..layer_higher, the all-four-corners gate and its `break`),
    2296-2360 (mode "trilinear": which two layers the CollectTiled2Pointsize
    path writes) and 3529-3559 (the blend recurrence).
"""

from __future__ import annotations

import math

import numpy as np

from trippy.constants import (
    RASTER_ALPHA_MIN,
    RASTER_MAX_FRAGS,
    RASTER_NUM_LAYERS,
    RASTER_SMALL_POINT_CUTOFF,
    RASTER_T_CUTOFF,
    RASTER_ZNEAR,
)
from trippy.geom.xform_a import project_pinhole, world_to_cam


def layer_bounds_scalar(size_px: float, num_layers: int) -> tuple[int, int]:
    """Lower/upper pyramid layer for one point (PointBlending.h:86-93).

    Args:
        size_px: projected point size in layer-0 pixels (> 0).
        num_layers: L.

    Returns:
        (lower, upper), both in [0, L - 1], both 0 when size_px <= 1.
    """
    if size_px <= 1.0:
        return 0, 0
    log_ps = math.log2(size_px)
    lower = min(max(0, math.floor(log_ps)), num_layers - 1)
    upper = min(max(0, math.ceil(log_ps)), num_layers - 1)
    return lower, upper


def layer_factor_scalar(size_px: float, layer: int, num_layers: int) -> float:
    """TRIPS's per-layer blend factor for one point (PointBlending.h:81-149).

    Args:
        size_px: projected point size in layer-0 pixels (> 0).
        layer: pyramid layer the factor is wanted for.
        num_layers: L.

    Returns:
        The blend factor in [0, 1]. See trippy.raster.emit.layer_factor for
        the full annotated description of the piecewise cases (including the
        `layer < lower` branch returning 1.0 in the C++ source).
    """
    lower, upper = layer_bounds_scalar(size_px, num_layers)
    if layer < lower:
        return 1.0
    if upper == 0:
        return (1.0 - RASTER_SMALL_POINT_CUTOFF) * math.exp(size_px - 1.0) + RASTER_SMALL_POINT_CUTOFF
    if lower == upper:
        return 1.0
    lo_pow = float(1 << lower)
    hi_pow = float(1 << upper)
    factor = (size_px - lo_pow) / (hi_pow - lo_pow)
    if layer == lower:
        factor = 1.0 - factor
    if lower == num_layers - 1:
        factor = 1.0
    return factor


def layer_higher_scalar(size_px: float, num_layers: int) -> int:
    """TRIPS's `layer_higher`: the coarsest layer mode "trips" writes into.

    Re-derived here from `RenderForward.cu:334-338`
    (`CountAndCollectTiled`), not imported from trippy.raster.emit:

        int layer_higher = 0;
        if (point_size_opt > 1) layer_higher = min(int(ceil(log2f(point_size_opt))), num_layers - 1);

    Args:
        size_px: projected point size in layer-0 pixels.
        num_layers: L.

    Returns:
        An int in [0, L - 1]; 0 for any sub-pixel point.
    """
    if size_px <= 1.0:
        return 0
    # `math.ceil` already returns an int, so the C++ `int(...)` cast is implicit.
    return min(math.ceil(math.log2(size_px)), num_layers - 1)


def layer_shapes(height: int, width: int, num_layers: int, pyramid_halving: str = "ceil") -> list[tuple[int, int]]:
    """Pyramid layer sizes for layer l.

    Args:
        height, width: layer-0 size in pixels.
        num_layers: L.
        pyramid_halving: "ceil" -> (ceil(H / 2**l), ceil(W / 2**l)), what
            TRIPS does for every published checkpoint
            (PointRenderer.cu:385-391); "floor" -> (H // 2**l, W // 2**l),
            its `MultiScaleUnet2d` branch.

    Returns:
        List of L (h_l, w_l) pairs.
    """
    if pyramid_halving == "ceil":
        return [(-(-height // (1 << layer)), -(-width // (1 << layer))) for layer in range(num_layers)]
    if pyramid_halving == "floor":
        return [(height // (1 << layer), width // (1 << layer)) for layer in range(num_layers)]
    raise ValueError(f"pyramid_halving must be 'ceil' or 'floor', got {pyramid_halving!r}")


def render_pyramid_numpy(
    xyz: np.ndarray,
    size: np.ndarray,
    feat: np.ndarray,
    conf: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_hw: tuple[int, int],
    num_layers: int = RASTER_NUM_LAYERS,
    mode: str = "trilinear",
    bg: np.ndarray | None = None,
    max_frags: int = RASTER_MAX_FRAGS,
    t_cutoff: float = RASTER_T_CUTOFF,
    alpha_min: float = RASTER_ALPHA_MIN,
    znear: float = RASTER_ZNEAR,
    pixel_center: str = "half",
    pyramid_halving: str = "ceil",
) -> tuple[list[np.ndarray], dict]:
    """Render an L-layer pyramid the slow, obvious way.

    Args:
        xyz: (N, 3) float, COLMAP world frame, world units.
        size: (N,) float, effective point radius, world units.
        feat: (N, C) float, per-point features.
        conf: (N,) float in (0, 1), effective confidence.
        K: (3, 3) float, layer-0 intrinsics in pixels.
        R: (3, 3) float world->camera rotation; t: (3,) translation.
        image_hw: (H, W) layer-0 image size in pixels.
        num_layers: L.
        mode: "trips" (layers 0..layer_higher, weighted, with TRIPS's
            all-four-corners footprint gate and its `break`), "trilinear"
            (layers [lower, upper], weighted) or "broadcast" (all layers,
            factor 1). See trippy.raster.emit.emit_fragments.
        bg: (C,) float background, added as `t_final * bg`; None means zero.
        max_frags: per-pixel fragment cap.
        t_cutoff: transmittance stop threshold.
        alpha_min: emission-time alpha floor.
        znear: near-plane cull, world units.
        pixel_center: "half" (pixel i centred at i + 0.5, trippy) or
            "integer" (centred at i, TRIPS).
        pyramid_halving: "ceil" or "floor" (see layer_shapes).

    Returns:
        layers: list of L float64 arrays, layer l is (C, h_l, w_l).
        aux: {"t_final": list of L (h_l, w_l), "n_used": list of L
            (h_l, w_l) int64, "depth_sum": list of L (h_l, w_l),
            "num_fragments": int, "fragments_per_layer": list of L int}.
    """
    if mode not in ("trilinear", "broadcast", "trips"):
        raise ValueError(f"mode must be 'trips', 'trilinear' or 'broadcast', got {mode!r}")
    if pixel_center not in ("half", "integer"):
        raise ValueError(f"pixel_center must be 'half' or 'integer', got {pixel_center!r}")
    # 0.5 puts the centre of pixel i at i + 0.5 (docs/GEOMETRY.md), 0.0 puts
    # it at i (TRIPS's `ip`, PointBlending.h:216-240).
    centre_shift = 0.5 if pixel_center == "half" else 0.0
    height, width = int(image_hw[0]), int(image_hw[1])
    shapes = layer_shapes(height, width, num_layers, pyramid_halving)
    num_channels = int(feat.shape[1])

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    xyz_c = world_to_cam(np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64), xyz)
    uv, depth = project_pinhole(xyz_c, fx, fy, cx, cy)

    # buckets[layer][(y, x)] = list of (depth_key_float32, point_id, alpha, depth)
    buckets: list[dict[tuple[int, int], list[tuple[np.float32, int, float, float]]]] = [
        {} for _ in range(num_layers)
    ]
    num_fragments = 0

    for i in range(xyz.shape[0]):
        z = float(depth[i])
        if z <= znear:
            continue
        size_px = fx * float(size[i]) / z
        if mode == "broadcast":
            selected = range(num_layers)
        elif mode == "trips":
            selected = range(layer_higher_scalar(size_px, num_layers) + 1)
        else:
            lower, upper = layer_bounds_scalar(size_px, num_layers)
            selected = range(lower, upper + 1)
        for layer in selected:
            h_l, w_l = shapes[layer]
            scale = 1.0 / float(1 << layer)
            # The layer coordinate halves exactly (RenderForward.cu:1610);
            # `centre_shift` re-anchors the 2x2 footprint on pixel centres.
            u = float(uv[i, 0]) * scale - centre_shift
            v = float(uv[i, 1]) * scale - centre_shift
            base_x = math.floor(u)
            base_y = math.floor(v)
            # `if (!valid_point(p_rd, z, layer, ...)) break;`
            # (RenderForward.cu:344-345) -- all four corners must be in
            # bounds, and failing here also skips every coarser layer.
            if mode == "trips" and not (0 <= base_x < w_l - 1 and 0 <= base_y < h_l - 1):
                break
            factor = 1.0 if mode == "broadcast" else layer_factor_scalar(size_px, layer, num_layers)
            frac_x = u - base_x
            frac_y = v - base_y
            for corner_y in (0, 1):
                for corner_x in (0, 1):
                    weight_x = frac_x if corner_x else 1.0 - frac_x
                    weight_y = frac_y if corner_y else 1.0 - frac_y
                    alpha = weight_x * weight_y * float(conf[i]) * factor
                    pixel_x = base_x + corner_x
                    pixel_y = base_y + corner_y
                    # Drop, never clamp (docs/GEOMETRY.md bug class 3).
                    if not (0 <= pixel_x < w_l and 0 <= pixel_y < h_l):
                        continue
                    if alpha < alpha_min:
                        continue
                    buckets[layer].setdefault((pixel_y, pixel_x), []).append(
                        (np.float32(z), i, alpha, z)
                    )
                    num_fragments += 1

    layers: list[np.ndarray] = []
    aux: dict = {
        "t_final": [],
        "n_used": [],
        "depth_sum": [],
        "num_fragments": num_fragments,
        "fragments_per_layer": [sum(len(v) for v in buckets[layer].values()) for layer in range(num_layers)],
    }
    for layer in range(num_layers):
        h_l, w_l = shapes[layer]
        image = np.zeros((num_channels, h_l, w_l), dtype=np.float64)
        t_final = np.ones((h_l, w_l), dtype=np.float64)
        n_used = np.zeros((h_l, w_l), dtype=np.int64)
        depth_sum = np.zeros((h_l, w_l), dtype=np.float64)

        for (pixel_y, pixel_x), fragments in buckets[layer].items():
            # Sort key matches trippy.raster.sort: float32 depth, then point
            # id as the stable tie-break.
            fragments.sort(key=lambda frag: (frag[0], frag[1]))
            transmittance = 1.0
            used = 0
            accum = np.zeros(num_channels, dtype=np.float64)
            d_sum = 0.0
            for _, point_id, alpha, z in fragments:
                if used >= max_frags:
                    break
                if transmittance < t_cutoff:
                    break
                weight = transmittance * alpha
                accum += weight * feat[point_id].astype(np.float64)
                d_sum += weight * z
                transmittance *= 1.0 - alpha
                used += 1
            image[:, pixel_y, pixel_x] = accum
            t_final[pixel_y, pixel_x] = transmittance
            n_used[pixel_y, pixel_x] = used
            depth_sum[pixel_y, pixel_x] = d_sum

        if bg is not None:
            image = image + t_final[None, :, :] * np.asarray(bg, dtype=np.float64).reshape(-1, 1, 1)
        layers.append(image)
        aux["t_final"].append(t_final)
        aux["n_used"].append(n_used)
        aux["depth_sum"].append(depth_sum)

    return layers, aux
