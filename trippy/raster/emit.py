"""Fragment emission: project points, pick pyramid layers, splat 2x2 footprints.

Module: trippy.raster.emit
Purpose: turn a point cloud plus a camera into a flat, unsorted list of
    *fragments* -- one (layer, pixel, depth, point id, alpha) record per
    bilinear footprint corner that lands inside a pyramid layer. Everything
    here is vectorised torch and dtype/device agnostic, so the same code
    drives the CPU float64 reference (trippy.raster.ref_torch) and the MPS
    float32 path (trippy.raster.pyramid).
Invariants:
    - Coordinate frames: `xyz` is COLMAP world frame; `(R, t)` is
      world->camera (x_cam = R @ x_world + t); `uv` is continuous layer-0
      pixel coordinates in the *corner-origin* convention of
      docs/GEOMETRY.md (the centre of pixel index i sits at i + 0.5).
    - Layer-l coordinates are `uv / 2**l` exactly, with no half-pixel
      offsets: in corner-origin coordinates a 2x downsample is exactly a
      halving. This mirrors TRIPS's `ip *= 0.5f` per layer
      (third_party/TRIPS/src/lib/rendering/RenderForward.cu:1610).
    - Out-of-bounds fragments are DROPPED, never clamped (docs/GEOMETRY.md
      historical bug class 3: clamped/padded pixels unproject as scene).
    - `alpha` stays connected to autograd; the integer key parts do not.
Units: `size` is a world-unit radius; `size_px = fx * size / z` is the same
    quantity in layer-0 pixels; `depth` is camera-space z in world units.
Related docs: docs/TRIPS_REFERENCE.md sections 3 and 10; docs/GEOMETRY.md
    "Image pyramid" / "Pyramid level selection"; docs/ARCHITECTURE.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from trippy.constants import (
    RASTER_ALPHA_MIN,
    RASTER_CULL_MARGIN_COARSE_PX,
    RASTER_NUM_LAYERS,
    RASTER_SMALL_POINT_CUTOFF,
    RASTER_ZNEAR,
)
from trippy.geom.xform_b import project_pinhole, world_to_cam
from trippy.raster.sort import segment_offsets, sort_fragments

EMIT_MODES = ("trilinear", "broadcast")

# Offsets of the four bilinear footprint corners, in TRIPS's blend_vec order
# (PointBlending.h:216-240): index = 2 * dy + dx.
_CORNER_DX = (0, 1, 0, 1)
_CORNER_DY = (0, 0, 1, 1)


@dataclass(frozen=True)
class LayerGrid:
    """Geometry of the image pyramid and of the flat layer-pixel index space.

    Attributes:
        shapes: list of L (h_l, w_l) int pairs, h_l = ceil(H / 2**l).
        offsets: list of L ints; `offsets[l]` is the first flat layer-pixel
            index belonging to layer l.
        total: int, number of layer-pixels across the whole pyramid
            (= sum of h_l * w_l). Flat index of layer l pixel (y, x) is
            `offsets[l] + y * w_l + x`.
    """

    shapes: list[tuple[int, int]]
    offsets: list[int]
    total: int


def layer_grid(height: int, width: int, num_layers: int = RASTER_NUM_LAYERS) -> LayerGrid:
    """Build the pyramid geometry for a layer-0 image of size (height, width).

    Args:
        height, width: layer-0 image size in pixels (ints, > 0).
        num_layers: L, number of pyramid layers (>= 1).

    Returns:
        LayerGrid with `ceil` halving per docs/GEOMETRY.md. NOTE this is a
        deliberate deviation from TRIPS, which halves with integer division
        (`h /= 2`, PointRenderer.cu:378) and therefore loses the last row and
        column of odd-sized layers.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"image size must be positive, got {(height, width)}")
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    shapes: list[tuple[int, int]] = []
    offsets: list[int] = []
    total = 0
    for layer in range(num_layers):
        step = 1 << layer
        h_l = -(-height // step)
        w_l = -(-width // step)
        shapes.append((h_l, w_l))
        offsets.append(total)
        total += h_l * w_l
    return LayerGrid(shapes=shapes, offsets=offsets, total=total)


def layer_bounds(size_px: Tensor, num_layers: int) -> tuple[Tensor, Tensor]:
    """Lower/upper pyramid layer a point of pixel size `size_px` writes into.

    Direct port of `compute_point_size_fac`'s first block
    (third_party/TRIPS/src/lib/rendering/PointBlending.h:86-93):
    both layers are 0 when `size_px <= 1`, otherwise floor/ceil of
    log2(size_px) clamped to [0, num_layers - 1].

    Args:
        size_px: (N,) float tensor, projected point size in layer-0 pixels.
        num_layers: L.

    Returns:
        (lower, upper): two (N,) int64 tensors with 0 <= lower <= upper <= L-1.
    """
    log_ps = torch.log2(torch.clamp(size_px, min=torch.finfo(size_px.dtype).tiny))
    big = size_px > 1.0
    lower = torch.clamp(torch.floor(log_ps), 0.0, float(num_layers - 1)).to(torch.int64)
    upper = torch.clamp(torch.ceil(log_ps), 0.0, float(num_layers - 1)).to(torch.int64)
    zero = torch.zeros_like(lower)
    return torch.where(big, lower, zero), torch.where(big, upper, zero)


def layer_factor(size_px: Tensor, layer: int, num_layers: int) -> Tensor:
    """TRIPS's per-layer blend factor for a point of pixel size `size_px`.

    Exact port of `compute_point_size_fac`
    (third_party/TRIPS/src/lib/rendering/PointBlending.h:81-149), including
    its quirks:

    - `layer < lower` returns **1.0** in the C++ (`return 1.f;`,
      PointBlending.h:96-100), not 0.0 as docs/TRIPS_REFERENCE.md section 3
      states. The branch is unreachable from TRIPS's own point-size emission
      kernel `CollectTiled2Pointsize`, which writes only `layer_lower` and
      `layer_higher` (RenderForward.cu:2296-2360), and from ours, so the
      value is inert; we match the source, not the doc.
    - `upper == 0` (i.e. `size_px <= 1`, a sub-pixel point) gives the
      exponential floor `(1 - c) * exp(size_px - 1) + c` with
      c = RASTER_SMALL_POINT_CUTOFF = 0.25 (PointBlending.h:106). This is a
      floor on the *factor*, not a clamp on the size.
    - `lower == upper` (size an exact power of two, or clamped to the top
      layer) gives 1.0.
    - otherwise the interpolation is linear **in point-size units** between
      2**lower and 2**upper (PointBlending.h:126-130), not linear in the
      log2 fraction: size 2**1.3 = 2.46 gives (2.46-2)/(4-2) = 0.23, not 0.3.

    Args:
        size_px: (N,) float tensor, projected point size in layer-0 pixels.
        layer: int, which pyramid layer the factor is wanted for.
        num_layers: L.

    Returns:
        (N,) float tensor in [0, 1], same dtype/device as `size_px`.
    """
    lower, upper = layer_bounds(size_px, num_layers)
    one = torch.ones_like(size_px)

    # Sub-pixel branch. exp() is evaluated on a clamped size so the unused
    # lanes (size_px >> 1) cannot overflow to inf before torch.where picks.
    small = (1.0 - RASTER_SMALL_POINT_CUTOFF) * torch.exp(
        torch.clamp(size_px, max=1.0) - 1.0
    ) + RASTER_SMALL_POINT_CUTOFF

    lo_pow = torch.pow(torch.full_like(size_px, 2.0), lower.to(size_px.dtype))
    hi_pow = torch.pow(torch.full_like(size_px, 2.0), upper.to(size_px.dtype))
    denom = hi_pow - lo_pow
    denom_safe = torch.where(denom == 0, torch.ones_like(denom), denom)
    interp = (size_px - lo_pow) / denom_safe
    interp = torch.where(lower == layer, 1.0 - interp, interp)
    # Dead in practice (upper is clamped to num_layers-1, so lower ==
    # num_layers-1 implies lower == upper), kept for source fidelity.
    interp = torch.where(lower == num_layers - 1, one, interp)

    factor = torch.where(lower == upper, one, interp)
    factor = torch.where(upper == 0, small, factor)
    return torch.where(lower > layer, one, factor)


def project_points(
    xyz: Tensor,
    size: Tensor,
    K: Tensor,
    R: Tensor,
    t: Tensor,
    znear: float = RASTER_ZNEAR,
) -> tuple[Tensor, Tensor, Tensor]:
    """Project world points to layer-0 pixels and pixel-space sizes.

    Uses trippy.geom.xform_b (autograd-friendly torch projection).

    Args:
        xyz: (N, 3) float, COLMAP world-frame positions, world units.
        size: (N,) float, effective (post-softplus) point radius in world
            units. The trainer owns the raw parameter.
        K: (3, 3) float, layer-0 pinhole intrinsics in pixels.
        R: (3, 3) float, world->camera rotation.
        t: (3,) float, world->camera translation, world units.
        znear: float, depths at or below this are treated as behind the
            camera (see trippy.constants.RASTER_ZNEAR).

    Returns:
        uv: (N, 2) float, continuous layer-0 pixel coordinates (corner
            origin; pixel i spans [i, i+1)).
        depth: (N,) float, camera-space z in world units (may be <= znear;
            callers cull on it).
        size_px: (N,) float, `fx * size / max(z, znear)`, the projected point
            diameter in layer-0 pixels. TRIPS uses fx only, not fy
            (RenderForward.cu:1489).
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    xyz_c = world_to_cam(R, t, xyz)
    uv, depth = project_pinhole(xyz_c, fx, fy, cx, cy)
    depth_safe = torch.clamp(depth, min=znear)
    size_px = fx * size / depth_safe
    return uv, depth, size_px


def cull_points(
    uv: Tensor,
    depth: Tensor,
    size_px: Tensor,
    grid: LayerGrid,
    znear: float = RASTER_ZNEAR,
    margin_coarse_px: float = RASTER_CULL_MARGIN_COARSE_PX,
) -> Tensor:
    """Coarse visibility mask: in front of the camera and footprint on screen.

    Deliberately conservative -- this cull must never remove a point that the
    exact per-fragment bounds test in emit_fragments would have kept, or a
    border point would silently lose its coarse-layer contribution. Two
    effects widen the box: the coarsest layer covers
    `ceil(W / 2**(L-1)) * 2**(L-1)` layer-0 columns (up to 2**(L-1) - 1 more
    than W), and a fragment at layer l survives for `uv / 2**l` anywhere in
    (-1.5, w_l + 0.5), i.e. 1.5 coarsest-layer pixels outside that box.

    Args:
        uv, depth, size_px: as returned by project_points.
        grid: LayerGrid for the image being rendered.
        znear: near-plane cull, world units.
        margin_coarse_px: slack in coarsest-layer pixels (>= 1.5).

    Returns:
        (N,) bool tensor, True for points worth emitting.
    """
    num_layers = len(grid.shapes)
    coarse = 1 << (num_layers - 1)
    padded_h = grid.shapes[-1][0] * coarse
    padded_w = grid.shapes[-1][1] * coarse
    slack = margin_coarse_px * coarse
    radius = 0.5 * size_px + slack
    u = uv[:, 0]
    v = uv[:, 1]
    return (
        (depth > znear)
        & (u + radius > 0.0)
        & (u - radius < float(padded_w))
        & (v + radius > 0.0)
        & (v - radius < float(padded_h))
    )


@dataclass
class Fragments:
    """A flat, unsorted fragment list for one image.

    All tensors share axis 0 (F fragments) and live on one device.

    Attributes:
        layer_pixel: (F,) int64, flat index into the pyramid's layer-pixel
            space (see LayerGrid.total). Sorting by this then by depth is
            exactly "sort by (layer, pixel, depth)".
        layer: (F,) int64, pyramid layer index in [0, L).
        pixel: (F,) int64, row-major pixel index within that layer
            (y * w_l + x).
        depth: (F,) float, camera-space z of the source point, world units.
        point_id: (F,) int64, index into the input point arrays.
        alpha: (F,) float, `bilinear_weight * conf * layer_factor`, in
            (0, 1). Connected to autograd.
        grid: the LayerGrid the indices refer to.
    """

    layer_pixel: Tensor
    layer: Tensor
    pixel: Tensor
    depth: Tensor
    point_id: Tensor
    alpha: Tensor
    grid: LayerGrid

    def __len__(self) -> int:
        return int(self.layer_pixel.shape[0])


def emit_fragments(
    uv: Tensor,
    depth: Tensor,
    size_px: Tensor,
    conf: Tensor,
    grid: LayerGrid,
    mode: str = "trilinear",
    valid: Tensor | None = None,
    alpha_min: float = RASTER_ALPHA_MIN,
) -> Fragments:
    """Emit bilinear fragments for every (point, layer) pair the point covers.

    Layer selection:
        - mode "trilinear": layers [lower, upper] from `layer_bounds`, each
          weighted by `layer_factor` -- TRIPS's `use_layer_point_size=true`
          path, whose emission kernel `CollectTiled2Pointsize` writes exactly
          those two layers (RenderForward.cu:2296-2360). That path is
          unreachable from any shipped TRIPS .ini (docs/TRIPS_REFERENCE.md
          section 11). At most 2 layers x 4 corners = 8 fragments per point.
        - mode "broadcast": every layer, factor 1 -- TRIPS's actual shipped
          default (`use_layer_point_size=false`, docs/TRIPS_REFERENCE.md
          section 10.1). Up to L x 4 = 20 fragments per point.

    Footprint: layer coordinate `uv_l = uv / 2**l`; the 2x2 footprint is
    anchored at pixel *centres*, so the base pixel is
    `floor(uv_l - 0.5)` and the fractional weights come from
    `uv_l - 0.5 - base` (docs/GEOMETRY.md pixel-centre convention).

    Args:
        uv: (N, 2) float, layer-0 continuous pixel coordinates.
        depth: (N,) float, camera-space z, world units.
        size_px: (N,) float, projected size in layer-0 pixels.
        conf: (N,) float in (0, 1), effective (post-sigmoid) confidence. The
            trainer owns the raw parameter.
        grid: LayerGrid describing the pyramid.
        mode: "trilinear" or "broadcast".
        valid: optional (N,) bool mask from cull_points; None means all.
        alpha_min: fragments with alpha below this are dropped.

    Returns:
        Fragments (unsorted; ordering is by layer then by input point index).
    """
    if mode not in EMIT_MODES:
        raise ValueError(f"mode must be one of {EMIT_MODES}, got {mode!r}")
    num_layers = len(grid.shapes)
    device = uv.device
    dtype = uv.dtype
    if valid is None:
        valid = torch.ones(uv.shape[0], dtype=torch.bool, device=device)

    dx = torch.tensor(_CORNER_DX, dtype=torch.int64, device=device)
    dy = torch.tensor(_CORNER_DY, dtype=torch.int64, device=device)

    lower, upper = layer_bounds(size_px, num_layers)

    out_lp: list[Tensor] = []
    out_layer: list[Tensor] = []
    out_pix: list[Tensor] = []
    out_depth: list[Tensor] = []
    out_pid: list[Tensor] = []
    out_alpha: list[Tensor] = []

    for layer in range(num_layers):
        h_l, w_l = grid.shapes[layer]
        if mode == "broadcast":
            selected = valid
            factor_all = torch.ones_like(size_px)
        else:
            selected = valid & (lower <= layer) & (layer <= upper)
            factor_all = layer_factor(size_px, layer, num_layers)
        idx = torch.nonzero(selected, as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            continue

        scale = 1.0 / float(1 << layer)
        centred = uv.index_select(0, idx) * scale - 0.5
        base = torch.floor(centred)
        frac = centred - base
        base_i = base.to(torch.int64)

        wx = torch.cat([1.0 - frac[:, 0:1], frac[:, 0:1]], dim=1)
        wy = torch.cat([1.0 - frac[:, 1:2], frac[:, 1:2]], dim=1)
        # (M, 4) in _CORNER_DX/_CORNER_DY order: index = 2 * dy + dx.
        beta = (wy.unsqueeze(2) * wx.unsqueeze(1)).reshape(-1, 4)

        px = base_i[:, 0:1] + dx.reshape(1, 4)
        py = base_i[:, 1:2] + dy.reshape(1, 4)

        alpha = (
            beta
            * conf.index_select(0, idx).reshape(-1, 1)
            * factor_all.index_select(0, idx).reshape(-1, 1).to(dtype)
        )

        in_bounds = (px >= 0) & (px < w_l) & (py >= 0) & (py < h_l)
        keep = (in_bounds & (alpha >= alpha_min)).reshape(-1)
        if not bool(keep.any()):
            continue

        pixel_flat = (py * w_l + px).reshape(-1)[keep]
        out_pix.append(pixel_flat)
        out_lp.append(pixel_flat + grid.offsets[layer])
        out_layer.append(torch.full_like(pixel_flat, layer))
        out_alpha.append(alpha.reshape(-1)[keep])
        out_depth.append(depth.index_select(0, idx).reshape(-1, 1).expand(-1, 4).reshape(-1)[keep])
        out_pid.append(idx.reshape(-1, 1).expand(-1, 4).reshape(-1)[keep])

    if not out_lp:
        empty_i = torch.zeros(0, dtype=torch.int64, device=device)
        empty_f = torch.zeros(0, dtype=dtype, device=device)
        return Fragments(empty_i, empty_i.clone(), empty_i.clone(), empty_f, empty_i.clone(), empty_f.clone(), grid)

    return Fragments(
        layer_pixel=torch.cat(out_lp),
        layer=torch.cat(out_layer),
        pixel=torch.cat(out_pix),
        depth=torch.cat(out_depth),
        point_id=torch.cat(out_pid),
        alpha=torch.cat(out_alpha),
        grid=grid,
    )


@dataclass
class SortedFragments:
    """Fragments ordered by (layer, pixel, depth) plus their segment offsets.

    This is the exact input contract of the compositing step, whether that
    step runs in Metal (trippy.raster.metal_lib) or in torch
    (trippy.raster.ref_torch.composite_sorted).

    Attributes:
        layer_pixel: (F,) int64, non-decreasing flat layer-pixel index.
        layer: (F,) int64, pyramid layer of each fragment.
        pixel: (F,) int64, row-major pixel index within its layer.
        depth: (F,) float, camera-space z, world units, non-decreasing within
            each layer_pixel run.
        point_id: (F,) int64, row index into the point arrays.
        alpha: (F,) float in (0, 1), differentiable.
        offsets: (P + 1,) int64, segment starts; P == grid.total.
        grid: LayerGrid the indices refer to.
    """

    layer_pixel: Tensor
    layer: Tensor
    pixel: Tensor
    depth: Tensor
    point_id: Tensor
    alpha: Tensor
    offsets: Tensor
    grid: LayerGrid

    def __len__(self) -> int:
        return int(self.layer_pixel.shape[0])


def build_sorted_fragments(
    xyz: Tensor,
    size: Tensor,
    conf: Tensor,
    K: Tensor,
    R: Tensor,
    t: Tensor,
    grid: LayerGrid,
    mode: str = "trilinear",
    alpha_min: float = RASTER_ALPHA_MIN,
    znear: float = RASTER_ZNEAR,
    sort_method: str = "composite",
    segment_method: str = "searchsorted",
    sort_stable: bool = True,
) -> SortedFragments:
    """Project -> cull -> emit -> sort -> segment, the whole pre-compositing pipeline.

    Shared by the CPU reference (trippy.raster.ref_torch) and the MPS path
    (trippy.raster.pyramid) so both composite exactly the same fragment list.

    Args:
        xyz: (N, 3) float, COLMAP world frame, world units.
        size: (N,) float, effective point radius, world units.
        conf: (N,) float in (0, 1), effective confidence.
        K: (3, 3) float, layer-0 intrinsics in pixels.
        R: (3, 3) float, world->camera rotation. t: (3,) float, translation.
        grid: LayerGrid for the target image.
        mode: "trilinear" or "broadcast" (see emit_fragments).
        alpha_min: emission-time alpha floor.
        znear: near-plane cull, world units.
        sort_method: "composite" or "two_pass" (see trippy.raster.sort).
        segment_method: "searchsorted" or "bincount".
        sort_stable: stable sort flag passed to sort_fragments.

    Returns:
        SortedFragments on the same device/dtype as the inputs.
    """
    uv, depth, size_px = project_points(xyz, size, K, R, t, znear=znear)
    valid = cull_points(uv, depth, size_px, grid, znear=znear)
    frags = emit_fragments(uv, depth, size_px, conf, grid, mode=mode, valid=valid, alpha_min=alpha_min)
    perm = sort_fragments(frags.layer_pixel, frags.depth, method=sort_method, stable=sort_stable)
    layer_pixel = frags.layer_pixel.index_select(0, perm)
    offsets = segment_offsets(layer_pixel, grid.total, method=segment_method)
    return SortedFragments(
        layer_pixel=layer_pixel,
        layer=frags.layer.index_select(0, perm),
        pixel=frags.pixel.index_select(0, perm),
        depth=frags.depth.index_select(0, perm),
        point_id=frags.point_id.index_select(0, perm),
        alpha=frags.alpha.index_select(0, perm),
        offsets=offsets,
        grid=grid,
    )
