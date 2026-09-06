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
      pixel coordinates. Whether the centre of pixel i sits at `i + 0.5`
      (docs/GEOMETRY.md, `pixel_center="half"`, the default) or at `i`
      (TRIPS's `ip`, `pixel_center="integer"`) is a *rendering option*, not a
      property of `uv`: it only changes where the 2x2 footprint is anchored.
    - Layer-l coordinates are `uv / 2**l` exactly, with no additive term, in
      both conventions. This mirrors TRIPS's `ip *= 0.5f` per layer
      (third_party/TRIPS/src/lib/rendering/RenderForward.cu:1610). The
      footprint anchor is `floor(uv/2**l - c)`, c = 0.5 or 0.0, so the two
      conventions differ by half a *layer-l* pixel, i.e. by a
      layer-dependent amount in layer-0 units -- see emit_fragments.
    - Three layer-selection modes, all ports of real TRIPS code paths:
      "trips" (layers 0..layer_higher, TRIPS's shipped-checkpoint rule),
      "trilinear" (the two straddling layers) and "broadcast" (all layers,
      factor 1). See trippy.constants.RASTER_MODES and emit_fragments.
    - Out-of-bounds fragments are DROPPED, never clamped (docs/GEOMETRY.md
      historical bug class 3: clamped/padded pixels unproject as scene).
    - Nothing in this module ever divides by the raw camera-space z. Both
      `fx * x / z` and `fx * size / z` divide by `safe_depth(z, znear)`
      instead, which is bit-identical for every point that survives the
      near-plane cull and finite for every point that does not -- because
      torch's division backward evaluates `-grad * (n / z / z)` on *all*
      rows, culled ones included, and turns a `z == 0` row into NaN even
      though its upstream gradient is exactly zero. See safe_depth.
    - `alpha` stays connected to autograd; the integer key parts do not.
    - An optional (6,) `pose_delta` refines `(R, t)` *left-multiplicatively*
      (`se3_exp(delta) @ [R | t]`, trippy.geom.xform_b.compose's convention)
      and is differentiable, so camera-pose refinement trains through the
      same graph as everything else (see apply_pose_delta).
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
    RASTER_MODES,
    RASTER_NUM_LAYERS,
    RASTER_PIXEL_CENTERS,
    RASTER_PYRAMID_HALVINGS,
    RASTER_SMALL_POINT_CUTOFF,
    RASTER_ZNEAR,
)
from trippy.geom.xform_b import compose, project_pinhole, world_to_cam
from trippy.raster.sort import segment_offsets, sort_fragments

EMIT_MODES = RASTER_MODES
PIXEL_CENTERS = RASTER_PIXEL_CENTERS
PYRAMID_HALVINGS = RASTER_PYRAMID_HALVINGS

# Emission implementations. Both compute the same Fragments bit-for-bit
# (tests/test_raster_emit_impl.py); "vectorised" builds one (L, M, 4) block
# over the culled points, "loop" is the original per-layer Python loop kept as
# the readable reference and the A/B baseline for tools/profile_raster.py.
EMIT_IMPLS = ("vectorised", "loop")

# Continuous-coordinate offset subtracted before `floor` to find the 2x2
# footprint's base pixel, per pixel-centre convention (RASTER_PIXEL_CENTERS).
_CENTRE_SHIFT = {"half": 0.5, "integer": 0.0}

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


def layer_grid(
    height: int,
    width: int,
    num_layers: int = RASTER_NUM_LAYERS,
    pyramid_halving: str = "ceil",
) -> LayerGrid:
    """Build the pyramid geometry for a layer-0 image of size (height, width).

    Args:
        height, width: layer-0 image size in pixels (ints, > 0).
        num_layers: L, number of pyramid layers (>= 1).
        pyramid_halving: "ceil" (default) -- `h_l = ceil(H / 2**l)`; or
            "floor" -- `h_l = H // 2**l` (see RASTER_PYRAMID_HALVINGS).
            "ceil" is what TRIPS does for every `network_version` other than
            the literal `"MultiScaleUnet2d"`, i.e. for every published
            checkpoint (`PointRenderer.cu:385-391`, docs/TRIPS_REFERENCE.md
            Sec. 3b); "floor" is TRIPS's other branch, which loses the last
            row and column of an odd-sized layer.

    Returns:
        LayerGrid.

    Raises:
        ValueError: on a bad size, a bad `num_layers`, an unknown
            `pyramid_halving`, or a pyramid so deep that "floor" halving
            would produce an empty layer.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"image size must be positive, got {(height, width)}")
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if pyramid_halving not in PYRAMID_HALVINGS:
        raise ValueError(f"pyramid_halving must be one of {PYRAMID_HALVINGS}, got {pyramid_halving!r}")
    shapes: list[tuple[int, int]] = []
    offsets: list[int] = []
    total = 0
    for layer in range(num_layers):
        step = 1 << layer
        if pyramid_halving == "ceil":
            h_l = -(-height // step)
            w_l = -(-width // step)
        else:
            # Repeated `h /= 2` is exactly one `h // 2**l` (floor composes).
            h_l = height // step
            w_l = width // step
        if h_l < 1 or w_l < 1:
            raise ValueError(
                f"pyramid_halving={pyramid_halving!r} gives an empty layer {layer} "
                f"({h_l}x{w_l}) for a {height}x{width} image; reduce num_layers"
            )
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


def apply_pose_delta(R: Tensor, t: Tensor, delta: Tensor) -> tuple[Tensor, Tensor]:
    """Refine a world->camera pose with a learnable SE(3) twist.

    Left-multiplicative (global-frame) convention, the one
    trippy.geom.xform_b.compose implements:
    `[R' | t'] = se3_exp(delta) @ [R | t]`. `delta` is the Sophus/g2o twist
    xi = (rho, phi): `delta[:3]` translation generator (world units),
    `delta[3:]` rotation vector (axis * angle, radians). A zero delta is
    exactly the identity, so `pose_delta=torch.zeros(6)` renders identically
    to passing no delta at all.

    This is the only hook the trainer needs for pose refinement: the returned
    (R', t') flow through projection into every fragment's alpha and depth,
    so `delta.grad` is populated by an ordinary backward.

    Args:
        R: (3, 3) float, world->camera rotation.
        t: (3,) float, world->camera translation, world units.
        delta: (6,) float twist; cast to `R`'s dtype/device.

    Returns:
        (R_refined (3, 3), t_refined (3,)) with the same dtype/device as `R`.

    Raises:
        ValueError: if `delta` is not shape (6,).
    """
    if tuple(delta.shape) != (6,):
        raise ValueError(f"pose_delta must have shape (6,), got {tuple(delta.shape)}")
    delta_c = delta.to(dtype=R.dtype, device=R.device)
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=R.dtype, device=R.device)
    pose = torch.cat([torch.cat([R, t.reshape(3, 1)], dim=1), bottom], dim=0)
    refined = compose(pose, delta_c)
    return refined[:3, :3], refined[:3, 3]


def safe_depth(depth: Tensor, znear: float = RASTER_ZNEAR) -> Tensor:
    """`depth` with every non-renderable value replaced by `znear`.

    This is the divisor of *both* projection divisions -- `fx * x / z` in
    `project_points` and `fx * size / z` for `size_px`. It exists for the
    backward pass, not the forward one.

    The forward result is unchanged for every point that can produce a
    fragment. `cull_points` keeps a point only when `depth > znear`, and for
    those `safe_depth(depth) == depth` bit-for-bit; the substituted value is
    only ever seen by points that are dropped before emission.

    The backward is where the raw `z` is unusable. torch differentiates
    `n / z` w.r.t. the denominator as `-grad * (n / z / z)`, and it evaluates
    that product for *every* row, including rows whose upstream gradient is
    exactly zero because the point was culled. At `z == 0` that is
    `-0 * inf = NaN`; the numerator's own derivative `grad / z` is `0 / 0`,
    also NaN. `z == 0` is not a measure-zero curiosity in float32: `z` is the
    third component of `xyz @ R.T + t`, so any point sitting on a camera's
    principal plane rounds to exactly 0.0. One such point in one training
    view is enough -- the NaN lands on that point's `xyz` gradient and, via
    `world_to_cam`, on all six components of that frame's pose delta, and Adam
    turns a NaN gradient into a permanently NaN parameter (observed on
    kk-coherent: 200k points, epoch 4, one point, docs/LIMITATIONS.md).
    Floors below `znear` are dangerous for the same reason even when non-zero:
    `n / z / z` overflows float32 once `|z|` drops under ~1e-19.

    `torch.where` rather than `torch.clamp`: `clamp` propagates NaN (a NaN
    depth stays NaN and poisons the numerator's `grad / z` in turn, which is
    how `raw_size` went NaN one step after `xyz` did), whereas `NaN > znear`
    is False, so `where` substitutes the finite `znear` and the lane's
    gradient is a clean zero.

    Args:
        depth: (N,) float, camera-space z in world units (any sign).
        znear: float > 0, the same near plane `cull_points` culls on.

    Returns:
        (N,) float, `depth` where `depth > znear` and `znear` everywhere else.
        Same dtype/device as `depth`, and differentiable (zero gradient on
        the substituted lanes).
    """
    return torch.where(depth > znear, depth, torch.full_like(depth, znear))


def project_points(
    xyz: Tensor,
    size: Tensor,
    K: Tensor,
    R: Tensor,
    t: Tensor,
    znear: float = RASTER_ZNEAR,
    pose_delta: Tensor | None = None,
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
        znear: float > 0, depths at or below this are treated as behind the
            camera (see trippy.constants.RASTER_ZNEAR). Both divisions here
            use `safe_depth(depth, znear)` as their divisor, so `znear` is
            also what keeps the *backward* finite -- see safe_depth.
        pose_delta: optional (6,) float SE(3) twist applied to `(R, t)` via
            `apply_pose_delta` before projecting. Carries gradient, so this
            is how camera-pose refinement is trained. None = no refinement.

    Returns:
        uv: (N, 2) float, continuous layer-0 pixel coordinates (corner
            origin; pixel i spans [i, i+1)). Computed as
            `fx * x / safe_depth(z) + cx`, hence exact for every point
            `cull_points` keeps (`z > znear`) and meaningless -- but always
            finite -- for the ones it drops.
        depth: (N,) float, the *true* camera-space z in world units, never
            floored (may be <= znear or negative; callers cull on it).
        size_px: (N,) float, `fx * size / safe_depth(z)`, the projected point
            diameter in layer-0 pixels. TRIPS uses fx only, not fy
            (RenderForward.cu:1489).
    """
    if pose_delta is not None:
        R, t = apply_pose_delta(R, t, pose_delta)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    xyz_c = world_to_cam(R, t, xyz)
    depth = xyz_c[:, 2]
    # Both divisions below run through `safe_depth`, never through the raw z:
    # a point with z == 0 (or |z| small enough that fx * x / z / z overflows)
    # makes torch's division backward emit NaN *even when its incoming
    # gradient is exactly zero*, and every such point is culled anyway. See
    # safe_depth for the full argument; this is the fix for docs/LIMITATIONS.md
    # "NaN gradient out of the rasteriser backward".
    depth_safe = safe_depth(depth, znear)
    xyz_c_safe = torch.cat([xyz_c[:, :2], depth_safe.reshape(-1, 1)], dim=1)
    uv, _ = project_pinhole(xyz_c_safe, fx, fy, cx, cy)
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
            [0, 1). Exactly 0 is reachable when `alpha_min=0` (TRIPS's own
            setting: it writes all four corners even at weight 0).
            Connected to autograd.
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
    pixel_center: str = "half",
    impl: str = "vectorised",
) -> Fragments:
    """Emit bilinear fragments for every (point, layer) pair the point covers.

    Layer selection (all three are real TRIPS code paths, see
    docs/GEOMETRY.md "Pyramid level selection" and trippy.constants.RASTER_MODES):

    - mode "trips": layers `0 .. layer_higher` inclusive, each weighted by
      `layer_factor`. This is what the published checkpoints render with:
      `use_layer_point_size = !fix_point_size = true` (`Settings.cpp:39`)
      selects `RenderFast16` / `CountAndCollectTiled`, whose emission loop is

          layer_higher = (s > 1) ? min(ceil(log2 s), L-1) : 0   # :334-338
          for (layer = 0; layer <= layer_higher; ++layer, ip *= 0.5f)
              if (!valid_point(floor(ip), z, layer)) break;     # break, NOT continue
              atomicAdd 4x into per_pixel_list_lengths[layer]   # :340-352

      and whose per-fragment alpha is `bilinear * confidence *
      compute_point_size_fac(s, layer, L)` (`RenderForward.cu:3511-3517`).
      `compute_point_size_fac` returns exactly **1.0** for every
      `layer < layer_lower` (`PointBlending.h:92-96`), so a 5-px point writes
      layers 0..3 with factors 1, 1, 0.75, 0.25. Up to L x 4 fragments per
      point. Two TRIPS-only rules come with this mode:
        * the footprint gate is `valid_point`, which requires **all four**
          corners in bounds (`0 <= floor(ip) < size - 1`), not the
          per-corner drop the other modes use; and
        * it is a `break`, so failing that gate at layer l also suppresses
          every *coarser* layer, even ones the point would have passed.
      Layer 0's gate is TRIPS's pre-loop `valid_point(ip, z, 0, ...)`
      (`RenderForward.cu:296`) -- identical, because `floor(x) < w - 1` and
      `x < w - 1` are the same statement for integers.
    - mode "trilinear": layers [lower, upper] from `layer_bounds`, each
      weighted by `layer_factor` -- TRIPS's `CollectTiled2Pointsize`
      emission (RenderForward.cu:2296-2360), reachable only through the
      `combine_lists = true` path. At most 2 layers x 4 corners = 8
      fragments per point.
    - mode "broadcast": every layer, factor 1 --
      `use_layer_point_size = false` (docs/TRIPS_REFERENCE.md section 10.1).
      Up to L x 4 fragments per point.

    Footprint: layer coordinate `uv_l = uv / 2**l` (an exact halving in both
    conventions, mirroring TRIPS's `ip *= 0.5f`), then

        base = floor(uv_l - c),  frac = uv_l - c - base

    with `c = 0.5` for `pixel_center="half"` (trippy's convention: the centre
    of pixel i is at i + 0.5, docs/GEOMETRY.md) and `c = 0.0` for
    `pixel_center="integer"` (TRIPS's `ip`, whose pixel centres are on
    integers -- `compute_blending_fac` splats at `floor(ip)`,
    `PointBlending.h:216-240`).

    Why `c` is subtracted *after* the halving and not before: TRIPS halves
    its own coordinate, `ip_l = ip / 2**l`, and anchors the footprint at
    `floor(ip_l)`. In the "half" convention the same operation is
    `floor(uv/2**l - 0.5)`. The two therefore differ by a *layer-dependent*
    half pixel, `2**(l-1)` layer-0 pixels, which is why a TRIPS-parity render
    needs `pixel_center="integer"` rather than a fixed `+0.5` on `cx, cy`
    (docs/TRIPS_REFERENCE.md Sec. 6a).

    Args:
        uv: (N, 2) float, layer-0 continuous pixel coordinates.
        depth: (N,) float, camera-space z, world units.
        size_px: (N,) float, projected size in layer-0 pixels.
        conf: (N,) float in (0, 1), effective (post-sigmoid) confidence. The
            trainer owns the raw parameter.
        grid: LayerGrid describing the pyramid.
        mode: one of EMIT_MODES.
        valid: optional (N,) bool mask from cull_points; None means all.
        alpha_min: fragments with alpha below this are dropped. TRIPS uses no
            such floor; pass 0.0 for an exact port (every in-bounds corner
            then takes a slot in the 16-deep list, even at weight 0).
        pixel_center: "half" or "integer" (see PIXEL_CENTERS).
        impl: which implementation runs. "vectorised" (default) handles
            every layer in one (L, M, ...) block over the M culled points;
            "loop" is the original per-layer Python loop. The two return
            bit-identical Fragments -- same order, same values -- and are
            checked against each other in tests/test_raster_emit_impl.py;
            only their cost differs, and only really on MPS, where a tensor
            shape the backend has not seen before costs several times a
            repeat (docs/ARCHITECTURE.md "Emission cost"). "loop" is kept as
            the readable statement of the rule and as the A/B baseline for
            tools/profile_raster.py.

    Returns:
        Fragments (unsorted; ordering is by layer, then by input point index,
        then by footprint corner).
    """
    if mode not in EMIT_MODES:
        raise ValueError(f"mode must be one of {EMIT_MODES}, got {mode!r}")
    if pixel_center not in PIXEL_CENTERS:
        raise ValueError(f"pixel_center must be one of {PIXEL_CENTERS}, got {pixel_center!r}")
    if impl not in EMIT_IMPLS:
        raise ValueError(f"impl must be one of {EMIT_IMPLS}, got {impl!r}")
    args = (uv, depth, size_px, conf, grid, mode, valid, alpha_min, pixel_center)
    if impl == "loop":
        return _emit_fragments_loop(*args)
    return _emit_fragments_vectorised(*args)


def _emit_fragments_loop(
    uv: Tensor,
    depth: Tensor,
    size_px: Tensor,
    conf: Tensor,
    grid: LayerGrid,
    mode: str,
    valid: Tensor | None,
    alpha_min: float,
    pixel_center: str,
) -> Fragments:
    """Original per-layer loop emission. See emit_fragments for the contract.

    Kept because it is the readable statement of the layer rule and the
    reference the vectorised path is checked against
    (tests/test_raster_emit_impl.py). It is no longer the default: on MPS it
    costs one `torch.nonzero`, six boolean-mask gathers and one
    `bool(keep.any())` readback per layer, and in mode "trips" it evaluates
    the four-corner gate -- and `layer_factor`, and `layer_bounds` inside it
    -- over all N points at every layer instead of over the M points the cull
    kept.
    """
    if mode not in EMIT_MODES:
        raise ValueError(f"mode must be one of {EMIT_MODES}, got {mode!r}")
    if pixel_center not in PIXEL_CENTERS:
        raise ValueError(f"pixel_center must be one of {PIXEL_CENTERS}, got {pixel_center!r}")
    num_layers = len(grid.shapes)
    device = uv.device
    dtype = uv.dtype
    centre_shift = _CENTRE_SHIFT[pixel_center]
    if valid is None:
        valid = torch.ones(uv.shape[0], dtype=torch.bool, device=device)

    dx = torch.tensor(_CORNER_DX, dtype=torch.int64, device=device)
    dy = torch.tensor(_CORNER_DY, dtype=torch.int64, device=device)

    lower, upper = layer_bounds(size_px, num_layers)
    # TRIPS's `layer_higher` is `layer_bounds`' upper bound: both are
    # `min(ceil(log2 s), L-1)` for s > 1 and 0 otherwise.
    alive = valid

    out_lp: list[Tensor] = []
    out_layer: list[Tensor] = []
    out_pix: list[Tensor] = []
    out_depth: list[Tensor] = []
    out_pid: list[Tensor] = []
    out_alpha: list[Tensor] = []

    for layer in range(num_layers):
        h_l, w_l = grid.shapes[layer]
        scale = 1.0 / float(1 << layer)

        if mode == "trips":
            # The gate has to be evaluated for every surviving point, not just
            # the selected ones, because it *propagates* to coarser layers.
            centred_all = uv * scale - centre_shift
            base_all = torch.floor(centred_all)
            fits = (
                (base_all[:, 0] >= 0.0)
                & (base_all[:, 0] <= float(w_l - 2))
                & (base_all[:, 1] >= 0.0)
                & (base_all[:, 1] <= float(h_l - 2))
            )
            alive = alive & fits
            selected = alive & (upper >= layer)
        elif mode == "broadcast":
            selected = valid
        else:  # "trilinear"
            selected = valid & (lower <= layer) & (layer <= upper)

        idx = torch.nonzero(selected, as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            continue

        if mode == "trips":
            centred = centred_all.index_select(0, idx)
            base = base_all.index_select(0, idx)
        else:
            centred = uv.index_select(0, idx) * scale - centre_shift
            base = torch.floor(centred)
        frac = centred - base
        base_i = base.to(torch.int64)

        size_sel = size_px.index_select(0, idx)
        # layer_factor is elementwise, so evaluating it on the selected rows
        # only is numerically identical and avoids L passes over all N points.
        factor = (
            torch.ones_like(size_sel)
            if mode == "broadcast"
            else layer_factor(size_sel, layer, num_layers)
        )

        wx = torch.cat([1.0 - frac[:, 0:1], frac[:, 0:1]], dim=1)
        wy = torch.cat([1.0 - frac[:, 1:2], frac[:, 1:2]], dim=1)
        # (M, 4) in _CORNER_DX/_CORNER_DY order: index = 2 * dy + dx.
        beta = (wy.unsqueeze(2) * wx.unsqueeze(1)).reshape(-1, 4)

        px = base_i[:, 0:1] + dx.reshape(1, 4)
        py = base_i[:, 1:2] + dy.reshape(1, 4)

        alpha = beta * conf.index_select(0, idx).reshape(-1, 1) * factor.reshape(-1, 1).to(dtype)

        # A no-op in mode "trips" (its `fits` gate already guarantees all four
        # corners are inside), the drop rule everywhere else.
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


def _empty_fragments(dtype: torch.dtype, device: torch.device, grid: LayerGrid) -> Fragments:
    """A zero-length Fragments on the given dtype/device (shared exit path)."""
    empty_i = torch.zeros(0, dtype=torch.int64, device=device)
    empty_f = torch.zeros(0, dtype=dtype, device=device)
    return Fragments(
        layer_pixel=empty_i,
        layer=empty_i.clone(),
        pixel=empty_i.clone(),
        depth=empty_f,
        point_id=empty_i.clone(),
        alpha=empty_f.clone(),
        grid=grid,
    )


def _layer_factor_grid(
    size_px: Tensor,
    lower: Tensor,
    upper: Tensor,
    layer: Tensor,
    num_layers: int,
) -> Tensor:
    """`layer_factor` evaluated for every (point, layer) pair in one shot.

    Bit-identical to `layer_factor(size_px, l, num_layers)` called once per
    layer: the arithmetic on each element is the same expression applied to
    the same values; only the broadcast shape differs. The size-dependent
    terms are computed once on (M,) and broadcast over the layer axis, which
    is the whole point -- the loop version recomputed `exp`, two `pow`s and a
    divide L times over the same sizes.

    Args:
        size_px: (M,) float, projected point size in layer-0 pixels.
        lower, upper: (M,) int64 from `layer_bounds(size_px, num_layers)`.
        layer: (L, 1) int64, the layer indices to evaluate at.
        num_layers: L.

    Returns:
        (L, M) float, same dtype/device as `size_px`.
    """
    m = int(size_px.shape[0])
    one = torch.ones_like(size_px)

    # Sub-pixel branch; exp() on a clamped size so unused lanes cannot overflow.
    small = (1.0 - RASTER_SMALL_POINT_CUTOFF) * torch.exp(
        torch.clamp(size_px, max=1.0) - 1.0
    ) + RASTER_SMALL_POINT_CUTOFF

    lo_pow = torch.pow(torch.full_like(size_px, 2.0), lower.to(size_px.dtype))
    hi_pow = torch.pow(torch.full_like(size_px, 2.0), upper.to(size_px.dtype))
    denom = hi_pow - lo_pow
    denom_safe = torch.where(denom == 0, torch.ones_like(denom), denom)

    lower_r = lower.reshape(1, m)
    upper_r = upper.reshape(1, m)
    one_r = one.reshape(1, m)
    interp = ((size_px - lo_pow) / denom_safe).reshape(1, m)
    interp = torch.where(lower_r == layer, 1.0 - interp, interp)
    # Dead in practice (see layer_factor), kept for source fidelity.
    interp = torch.where(lower_r == num_layers - 1, one_r, interp)
    factor = torch.where(lower_r == upper_r, one_r, interp)
    factor = torch.where(upper_r == 0, small.reshape(1, m), factor)
    return torch.where(lower_r > layer, one_r, factor)


def _emit_fragments_vectorised(
    uv: Tensor,
    depth: Tensor,
    size_px: Tensor,
    conf: Tensor,
    grid: LayerGrid,
    mode: str,
    valid: Tensor | None,
    alpha_min: float,
    pixel_center: str,
) -> Fragments:
    """Layer-vectorised emission. Same contract and same output as the loop.

    Three structural differences from `_emit_fragments_loop`, all of them
    pure cost, not semantics (tests/test_raster_emit_impl.py asserts the two
    return identical tensors):

    1. **The number of distinct tensor shapes per render drops from ~5 to 1.**
       This is the one that matters, and it is why mode "trips" cost 1.5x a
       mode "broadcast" training step on MPS despite emitting 2.7x FEWER
       fragments. The loop version's per-layer tensors are sized by how
       many points that layer selected, and in modes "trips"/"trilinear" that
       count differs per layer *and* moves every training step as the crop
       moves; in mode "broadcast" all five layers select the same rows, so
       one shape serves them all. MPS charges 6-9x for an elementwise kernel
       on a shape it has not seen before (measured, tools/profile_raster.py
       --shape-probe), so the two modes were paying very different amounts of
       that tax: with the camera moving, mode "trips"'s rasteriser cost rose
       56% over its own frozen-camera cost and "trilinear"'s 76%, while
       "broadcast"'s rose 1%. Here there is one M for the whole render, and
       the three modes finally rank by their fragment counts. See
       docs/ARCHITECTURE.md "Emission cost".
    2. The point set is compacted to the `valid` rows **once**, before
       anything per-layer runs. The loop version evaluates mode "trips"'s
       four-corner gate -- and `layer_factor`, and the `layer_bounds` inside
       it -- over all N points at every layer, because it reads the gate off
       the full-length `uv`; but `alive` starts at `valid`, so the gate can
       only ever matter on rows the cull already kept.
    3. Per layer the loop version issues one `torch.nonzero`, six
       boolean-mask gathers (each an implicit `nonzero`) and one
       `bool(keep.any())` readback -- 40 data-dependent operations for L=5,
       every one of which drains the MPS command queue. Here there are 3,
       and the fragment count is only read back once.

    What is deliberately kept from the loop: the bilinear weights and the
    alpha are evaluated **per emitted fragment**, not for all (L, M, 4)
    candidates, so mode "trips" does not pay for the coarse layers a point
    never writes. That is why emission runs in three passes -- geometry,
    then alpha, then the alpha floor -- rather than one.

    Layout note: every intermediate is (L, M) or (L, M, 4) with the layer
    axis FIRST, so `torch.nonzero` enumerates fragments in exactly the loop's
    order (layer, then input point index, then footprint corner). The
    fragment order is load-bearing: the depth sort is stable, so ties would
    otherwise composite in a different order.
    """
    num_layers = len(grid.shapes)
    device = uv.device
    dtype = uv.dtype
    centre_shift = _CENTRE_SHIFT[pixel_center]

    if valid is None:
        point_id = torch.arange(uv.shape[0], dtype=torch.int64, device=device)
    else:
        point_id = torch.nonzero(valid, as_tuple=False).reshape(-1)
    m = int(point_id.shape[0])
    if m == 0:
        return _empty_fragments(dtype, device, grid)

    uv_v = uv.index_select(0, point_id)
    size_v = size_px.index_select(0, point_id)
    conf_v = conf.index_select(0, point_id)
    depth_v = depth.index_select(0, point_id)

    heights = [s[0] for s in grid.shapes]
    widths = [s[1] for s in grid.shapes]
    scale = torch.tensor(
        [1.0 / float(1 << layer) for layer in range(num_layers)], dtype=dtype, device=device
    ).reshape(num_layers, 1)
    w_i = torch.tensor(widths, dtype=torch.int64, device=device)
    h_i = torch.tensor(heights, dtype=torch.int64, device=device)
    off_i = torch.tensor(grid.offsets, dtype=torch.int64, device=device)
    layer_ids = torch.arange(num_layers, dtype=torch.int64, device=device).reshape(num_layers, 1)

    # u and v are kept as separate (L, M) tensors rather than one (L, M, 2):
    # every later use is per-axis, and a stride-2 slice of a big MPS tensor is
    # materialised on every read.
    centred_u = uv_v[:, 0].reshape(1, m) * scale - centre_shift
    centred_v = uv_v[:, 1].reshape(1, m) * scale - centre_shift
    base_u = torch.floor(centred_u)
    base_v = torch.floor(centred_v)
    frac_u = centred_u - base_u
    frac_v = centred_v - base_v
    base_ui = base_u.to(torch.int64)
    base_vi = base_v.to(torch.int64)

    selected: Tensor | None
    factor: Tensor | None
    if mode == "broadcast":
        # Every layer, factor 1: neither `layer_bounds` nor `layer_factor` is
        # reachable, so neither is computed.
        selected = None
        factor = None
    else:
        lower, upper = layer_bounds(size_v, num_layers)
        if mode == "trips":
            w_gate = torch.tensor(
                [float(w - 2) for w in widths], dtype=dtype, device=device
            ).reshape(num_layers, 1)
            h_gate = torch.tensor(
                [float(h - 2) for h in heights], dtype=dtype, device=device
            ).reshape(num_layers, 1)
            fits = (base_u >= 0.0) & (base_u <= w_gate) & (base_v >= 0.0) & (base_v <= h_gate)
            # TRIPS `break`s out of its layer loop, so a failed gate suppresses
            # every coarser layer too: alive[l] = fits[0] & ... & fits[l].
            alive_rows = [fits[0]]
            for layer in range(1, num_layers):
                alive_rows.append(alive_rows[-1] & fits[layer])
            selected = torch.stack(alive_rows, dim=0) & (upper.reshape(1, m) >= layer_ids)
        else:  # "trilinear"
            selected = (lower.reshape(1, m) <= layer_ids) & (layer_ids <= upper.reshape(1, m))
        factor = _layer_factor_grid(size_v, lower, upper, layer_ids, num_layers)

    # --- pass 1: geometry, which needs no bilinear weight ---
    #
    # `base + d < size` and `base + d >= 0` are, for integers, `base < size - d`
    # and `base >= -d`; testing it that way keeps this in the (L, M) domain
    # instead of materialising two (L, M, 2) int64 corner tensors.
    w_l = w_i.reshape(num_layers, 1)
    h_l = h_i.reshape(num_layers, 1)
    ok_x = torch.stack(
        [(base_ui >= 0) & (base_ui < w_l), (base_ui >= -1) & (base_ui < w_l - 1)], dim=-1
    )
    ok_y = torch.stack(
        [(base_vi >= 0) & (base_vi < h_l), (base_vi >= -1) & (base_vi < h_l - 1)], dim=-1
    )
    # A no-op in mode "trips" (its `fits` gate already guarantees all four
    # corners are inside), the drop rule everywhere else.
    keep = (ok_y.unsqueeze(-1) & ok_x.unsqueeze(-2)).reshape(num_layers, m, 4)
    if selected is not None:
        keep = keep & selected.unsqueeze(-1)

    # `as_tuple=False` on the 3-D mask hands back the (layer, point-row, corner)
    # triple directly, so every offset below is multiply-add -- no integer
    # division. `nonzero` enumerates row-major, i.e. layer, then point, then
    # corner: exactly the loop implementation's fragment order.
    nz = torch.nonzero(keep, as_tuple=False)
    if int(nz.shape[0]) == 0:
        return _empty_fragments(dtype, device, grid)
    # Contiguous copies, not stride-3 views: each column is read several times
    # below, and a strided read of a multi-million-element MPS tensor is
    # materialised every time.
    frag_layer = nz[:, 0].contiguous()
    frag_row = nz[:, 1].contiguous()
    corner = nz[:, 2].contiguous()

    # --- pass 2: alpha, on the survivors only ---
    #
    # This is the loop implementation's one real economy and it is kept: the
    # 2x2 weights are evaluated per *fragment*, never for the (L, M, 4) pairs
    # the geometry already threw away. In mode "trips" that is ~3x fewer
    # multiplies, because a point only writes layers 0..layer_higher.
    dx = torch.tensor(_CORNER_DX, dtype=torch.int64, device=device)
    dy = torch.tensor(_CORNER_DY, dtype=torch.int64, device=device)
    corner_dx = dx.index_select(0, corner)
    corner_dy = dy.index_select(0, corner)
    lm = frag_layer * m + frag_row

    frag_frac_u = frac_u.reshape(-1).index_select(0, lm)
    frag_frac_v = frac_v.reshape(-1).index_select(0, lm)
    # `1 - frac` for the near corner, `frac` for the far one -- the two columns
    # of the loop version's `wx` / `wy`, picked per fragment.
    weight_x = torch.where(corner_dx == 0, 1.0 - frag_frac_u, frag_frac_u)
    weight_y = torch.where(corner_dy == 0, 1.0 - frag_frac_v, frag_frac_v)
    # Multiplication order is `(beta * conf) * factor` exactly as in the loop:
    # `beta` is `wy * wx`, not `wx * wy`, and float multiply is not associative.
    alpha = weight_y * weight_x * conf_v.index_select(0, frag_row)
    if factor is not None:
        alpha = alpha * factor.reshape(-1).index_select(0, lm).to(dtype)
    # (mode "broadcast"'s factor is exactly 1.0, and `x * 1.0 == x` for every
    # float, so skipping that multiply changes no bit of the result.)

    # --- pass 3: the alpha floor ---
    surviving = torch.nonzero(alpha >= alpha_min, as_tuple=False).reshape(-1)
    if int(surviving.shape[0]) == 0:
        return _empty_fragments(dtype, device, grid)
    if int(surviving.shape[0]) != int(alpha.shape[0]):
        frag_layer = frag_layer.index_select(0, surviving)
        frag_row = frag_row.index_select(0, surviving)
        corner_dx = corner_dx.index_select(0, surviving)
        corner_dy = corner_dy.index_select(0, surviving)
        lm = lm.index_select(0, surviving)
        alpha = alpha.index_select(0, surviving)

    px = base_ui.reshape(-1).index_select(0, lm) + corner_dx
    py = base_vi.reshape(-1).index_select(0, lm) + corner_dy
    pixel = py * w_i.index_select(0, frag_layer) + px

    return Fragments(
        layer_pixel=pixel + off_i.index_select(0, frag_layer),
        layer=frag_layer,
        pixel=pixel,
        depth=depth_v.index_select(0, frag_row),
        point_id=point_id.index_select(0, frag_row),
        alpha=alpha,
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
    pose_delta: Tensor | None = None,
    pixel_center: str = "half",
    emit_impl: str = "vectorised",
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
        mode: one of EMIT_MODES (see emit_fragments).
        alpha_min: emission-time alpha floor.
        znear: near-plane cull, world units.
        sort_method: "composite" or "two_pass" (see trippy.raster.sort).
        segment_method: "searchsorted" or "bincount".
        sort_stable: stable sort flag passed to sort_fragments.
        pose_delta: optional (6,) SE(3) twist refining `(R, t)`; see
            apply_pose_delta. Differentiable.
        pixel_center: "half" or "integer" (see emit_fragments).
        emit_impl: one of EMIT_IMPLS; "loop" selects the original per-layer
            emission, which is bit-identical and slower (see emit_fragments).

    Returns:
        SortedFragments on the same device/dtype as the inputs.
    """
    uv, depth, size_px = project_points(xyz, size, K, R, t, znear=znear, pose_delta=pose_delta)
    valid = cull_points(uv, depth, size_px, grid, znear=znear)
    frags = emit_fragments(
        uv,
        depth,
        size_px,
        conf,
        grid,
        mode=mode,
        valid=valid,
        alpha_min=alpha_min,
        pixel_center=pixel_center,
        impl=emit_impl,
    )
    # `grid.total - 1` is emission's own hard bound on layer_pixel, so handing it
    # to the sort removes the composite key's `.max().item()` synchronisation.
    perm = sort_fragments(
        frags.layer_pixel,
        frags.depth,
        method=sort_method,
        stable=sort_stable,
        max_layer_pixel=grid.total - 1,
    )
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
