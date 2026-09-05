"""Degenerate-fragment gradient guards for the CPU rasteriser backward.

Module: tests.test_raster_nan_ref
Invariants under test: `render_pyramid` on CPU must return a finite forward
    *and* finite gradients for every learnable input, for inputs that sit
    exactly on one of the rasteriser's internal discontinuities. Each case
    below is a real hazard that was reachable in a training run:

    1. A point whose camera-space z is exactly 0.0 -- the reproducing input
       for docs/LIMITATIONS.md "NaN gradient out of the rasteriser backward".
       The point is culled, so its upstream gradient is exactly zero, but
       torch differentiates `n / z` w.r.t. the denominator as
       `-grad * (n / z / z)` on *every* row, and `-0 * inf` is NaN.
       `trippy.raster.emit.safe_depth` is the fix.
    2. Depths at, just inside, and behind the near plane, plus depths small
       enough that `n / z / z` would overflow float32.
    3. A fragment sitting exactly on a pixel boundary (bilinear weight
       exactly 0 on two corners, exactly 1 on one).
    4. `size_px` exactly a power of two (`layer_bounds` lower == upper, so
       `layer_factor`'s interpolation denominator is zero), exactly 1.0 (the
       sub-pixel branch's switch), and exactly 0.
    5. Fragment alpha exactly 0 and exactly 1. `alpha == 1` makes
       `log1p(-alpha) = -inf` in `composite_sorted`; the epsilon that used to
       guard it was 1e-12, which rounds away in float32.

    Every case is checked in float32 (the trainer's compute dtype, where all
    of these first bit) as well as float64 (the reference dtype).
Related docs: docs/LIMITATIONS.md, docs/ARCHITECTURE.md ("Backward pass data
    flow"), trippy.raster.emit.safe_depth.
"""

from __future__ import annotations

import pytest
import torch

from trippy.constants import RASTER_ALPHA_MIN, RASTER_MODES, RASTER_ZNEAR
from trippy.raster import render_pyramid
from trippy.raster.emit import build_sorted_fragments, layer_grid, project_points, safe_depth
from trippy.raster.ref_torch import composite_sorted

# A tiny pyramid whose layer-0 pixel grid is small enough that every
# coordinate below is exactly representable in float32.
IMAGE_HW = (8, 8)
NUM_LAYERS = 3
FX = 8.0
CX = CY = 4.0

# A point that always renders, so the graph exists no matter what the probe
# point does. Camera is the identity pose, so this is camera-space too.
ANCHOR_XYZ = (0.0, 0.0, 5.0)
ANCHOR_SIZE = 0.5
ANCHOR_CONF = 0.75

# Probe placement that lands on the exact pixel boundary uv = (4.5, 4.5):
# u = FX * 0.25 / 4.0 + CX. At depth 4 with size 1.0, size_px = FX * 1 / 4 = 2,
# an exact power of two, so layer_factor's lower == upper branch fires too.
BOUNDARY_XY = 0.25
BOUNDARY_DEPTH = 4.0
BOUNDARY_SIZE = 1.0

# The same trick one layer up: uv = 5.0 makes `uv / 2 - 0.5` the integer 2,
# so the layer-1 footprint (the only layer mode "trilinear" selects at
# size_px == 2) has three bilinear weights of exactly 0.
LAYER1_BOUNDARY_XY = 0.5
LAYER1_BOUNDARY_UV = 5.0

# Depths that are degenerate for the projection divisions. 0.0 is the value
# observed in the kk-coherent repro (200k points, epoch 4, one point sitting
# on the camera's principal plane); the rest bracket the near plane and the
# float32 overflow threshold of `n / z / z`.
DEGENERATE_DEPTHS = (0.0, -0.0, RASTER_ZNEAR, 0.5 * RASTER_ZNEAR, -1.5, 1e-25, 1e-38)

# size_px values that hit every branch of layer_bounds / layer_factor:
# exact powers of two (zero interpolation denominator), the sub-pixel switch
# at 1.0, a size below it, and exactly zero (log2 of the dtype's tiny).
DEGENERATE_SIZE_PX = (0.0, 1e-30, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

DTYPES = (torch.float32, torch.float64)


def _render_grads(
    probe_xyz: tuple[float, float, float],
    probe_size: float,
    probe_conf: float,
    dtype: torch.dtype,
    mode: str,
    alpha_min: float = 0.0,
    anchor_xyz: tuple[float, float, float] = ANCHOR_XYZ,
    anchor_size: float = ANCHOR_SIZE,
    anchor_conf: float = ANCHOR_CONF,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Render an anchor + one probe point and backprop a non-symmetric loss.

    Args:
        probe_xyz, probe_size, probe_conf: the degenerate point.
        dtype, mode: compute dtype and trippy.raster.emit.EMIT_MODES entry.
        alpha_min: emission alpha floor (0.0 keeps zero-weight corners, which
            is what exercises the alpha == 0 case).
        anchor_xyz, anchor_size, anchor_conf: the point that keeps the graph
            alive; the defaults always render.

    Returns:
        (flat_render, {name: gradient}) -- gradients only for the inputs that
        received one (`size` gets none in mode "broadcast", where the layer
        factor is a constant 1).
    """
    K = torch.tensor([[FX, 0.0, CX], [0.0, FX, CY], [0.0, 0.0, 1.0]], dtype=dtype)
    R = torch.eye(3, dtype=dtype)
    t = torch.zeros(3, dtype=dtype)
    variables = {
        "xyz": torch.tensor([anchor_xyz, probe_xyz], dtype=dtype),
        "size": torch.tensor([anchor_size, probe_size], dtype=dtype),
        "conf": torch.tensor([anchor_conf, probe_conf], dtype=dtype),
        "feat": torch.tensor([[0.2, 0.4, 0.6], [0.7, 0.1, 0.3]], dtype=dtype),
        "pose_delta": torch.zeros(6, dtype=dtype),
    }
    for tensor in variables.values():
        tensor.requires_grad_(True)

    layers, _aux = render_pyramid(
        variables["xyz"],
        variables["size"],
        variables["feat"],
        variables["conf"],
        K,
        R,
        t,
        IMAGE_HW,
        num_layers=NUM_LAYERS,
        mode=mode,
        alpha_min=alpha_min,
        pose_delta=variables["pose_delta"],
    )
    flat = torch.cat([layer.reshape(-1) for layer in layers])
    # Non-uniform weights so no gradient can vanish by symmetry.
    weights = torch.linspace(0.3, 1.7, flat.numel(), dtype=dtype)
    (flat * weights).sum().backward()
    grads = {name: var.grad for name, var in variables.items() if var.grad is not None}
    return flat.detach(), grads


def _assert_all_finite(flat: torch.Tensor, grads: dict[str, torch.Tensor], label: str) -> None:
    """Fail with a readable message naming the first non-finite tensor."""
    assert torch.isfinite(flat).all(), f"{label}: non-finite forward"
    assert grads, f"{label}: nothing received a gradient, the case tests nothing"
    for name, grad in grads.items():
        bad = int((~torch.isfinite(grad)).sum())
        assert bad == 0, f"{label}: {bad} non-finite entries in d/d {name}: {grad.reshape(-1).tolist()}"


# --- 1 + 2: the projection divisions ---------------------------------------


@pytest.mark.parametrize("depth", DEGENERATE_DEPTHS)
@pytest.mark.parametrize("mode", RASTER_MODES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_degenerate_depth_keeps_every_gradient_finite(
    depth: float, mode: str, dtype: torch.dtype
) -> None:
    """A culled point at a degenerate depth must not poison anyone's gradient.

    `depth == 0.0` is the reproducing input from the kk-coherent run: the
    point is culled, its upstream gradient is exactly zero, and torch's
    division backward still evaluated `-0 * (fx * x / 0 / 0)` = NaN.
    """
    flat, grads = _render_grads((1.0, 0.5, depth), 0.1, 0.9, dtype, mode)
    _assert_all_finite(flat, grads, f"depth={depth!r} mode={mode} dtype={dtype}")


@pytest.mark.parametrize("mode", RASTER_MODES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_depth_point_is_culled_and_leaves_the_image_untouched(
    mode: str, dtype: torch.dtype
) -> None:
    """The guard is a backward-only change: it must not add pixels.

    A point at z == 0 and a point far behind the camera are both culled, so
    the two renders have to be bit-identical -- if `safe_depth` had leaked
    into the forward, the z == 0 point would splat at `fx * x / znear`.
    """
    at_zero, _ = _render_grads((1.0, 0.5, 0.0), 0.1, 0.9, dtype, mode)
    behind, _ = _render_grads((1.0, 0.5, -7.0), 0.1, 0.9, dtype, mode)
    torch.testing.assert_close(at_zero, behind, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_safe_depth_is_the_identity_on_every_renderable_depth(dtype: torch.dtype) -> None:
    """`safe_depth` may only ever change depths the near-plane cull drops."""
    depth = torch.tensor(
        [5.0, 1.0, 0.5, 2.0 * RASTER_ZNEAR, RASTER_ZNEAR, 0.0, -0.0, -3.0, 1e-25], dtype=dtype
    )
    guarded = safe_depth(depth, RASTER_ZNEAR)
    renderable = depth > RASTER_ZNEAR
    torch.testing.assert_close(guarded[renderable], depth[renderable], rtol=0.0, atol=0.0)
    assert torch.all(guarded[~renderable] == RASTER_ZNEAR)
    assert torch.isfinite(guarded).all()


def test_safe_depth_substitutes_rather_than_clamps_so_nan_cannot_propagate() -> None:
    """`torch.clamp` would keep the NaN; `torch.where` replaces it.

    This is the second half of the observed failure: once one point's `xyz`
    was NaN, `clamp(nan, min=znear)` stayed NaN, and `size_px = fx * size /
    nan` then handed `raw_size` a NaN gradient (`0 / nan`) on the next step.
    """
    depth = torch.tensor([float("nan"), 5.0], dtype=torch.float32, requires_grad=True)
    guarded = safe_depth(depth, RASTER_ZNEAR)
    assert torch.isfinite(guarded).all()
    assert float(guarded[0].detach()) == pytest.approx(RASTER_ZNEAR)
    (guarded * torch.tensor([1.0, 2.0])).sum().backward()
    assert torch.isfinite(depth.grad).all()
    assert float(depth.grad[0]) == 0.0


@pytest.mark.parametrize("dtype", DTYPES)
def test_projection_still_divides_by_the_true_depth_where_it_matters(dtype: torch.dtype) -> None:
    """uv and size_px are unchanged for every point in front of the near plane."""
    K = torch.tensor([[FX, 0.0, CX], [0.0, FX, CY], [0.0, 0.0, 1.0]], dtype=dtype)
    xyz = torch.tensor([[0.25, 0.25, 4.0], [-1.0, 2.0, 0.5], [1.0, 0.5, 0.0]], dtype=dtype)
    size = torch.tensor([1.0, 0.25, 0.1], dtype=dtype)
    uv, depth, size_px = project_points(xyz, size, K, torch.eye(3, dtype=dtype), torch.zeros(3, dtype=dtype))
    keep = depth > RASTER_ZNEAR
    expect_u = FX * xyz[keep, 0] / xyz[keep, 2] + CX
    expect_v = FX * xyz[keep, 1] / xyz[keep, 2] + CY
    torch.testing.assert_close(uv[keep, 0], expect_u, rtol=0.0, atol=0.0)
    torch.testing.assert_close(uv[keep, 1], expect_v, rtol=0.0, atol=0.0)
    torch.testing.assert_close(size_px[keep], FX * size[keep] / xyz[keep, 2], rtol=0.0, atol=0.0)
    # The true depth is returned untouched -- culling and fragment sorting
    # both depend on its sign.
    torch.testing.assert_close(depth, xyz[:, 2], rtol=0.0, atol=0.0)
    assert torch.isfinite(uv).all()


# --- 3: bilinear weights on a pixel boundary -------------------------------


@pytest.mark.parametrize("mode", RASTER_MODES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fragment_exactly_on_a_pixel_boundary(mode: str, dtype: torch.dtype) -> None:
    """frac == 0 exactly: two corners get bilinear weight 0, one gets 1."""
    probe = (BOUNDARY_XY, BOUNDARY_XY, BOUNDARY_DEPTH)
    K = torch.tensor([[FX, 0.0, CX], [0.0, FX, CY], [0.0, 0.0, 1.0]], dtype=dtype)
    uv, _depth, _size_px = project_points(
        torch.tensor([probe], dtype=dtype),
        torch.tensor([BOUNDARY_SIZE], dtype=dtype),
        K,
        torch.eye(3, dtype=dtype),
        torch.zeros(3, dtype=dtype),
    )
    # Assert the fixture really is degenerate rather than merely close.
    assert uv.reshape(-1).tolist() == [4.5, 4.5], "fixture no longer lands on a pixel boundary"

    flat, grads = _render_grads(probe, BOUNDARY_SIZE, 0.9, dtype, mode, alpha_min=0.0)
    _assert_all_finite(flat, grads, f"boundary mode={mode} dtype={dtype}")


# --- 4: layer selection at exact powers of two -----------------------------


@pytest.mark.parametrize("size_px", DEGENERATE_SIZE_PX)
@pytest.mark.parametrize("mode", RASTER_MODES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_degenerate_size_px_keeps_every_gradient_finite(
    size_px: float, mode: str, dtype: torch.dtype
) -> None:
    """size_px == 2**k, size_px == 1 and size_px == 0 are all branch switches."""
    # size_px = FX * size / depth, so this is the world-unit size that lands
    # exactly on the requested pixel size at BOUNDARY_DEPTH.
    world_size = size_px * BOUNDARY_DEPTH / FX
    flat, grads = _render_grads(
        (0.1, -0.2, BOUNDARY_DEPTH), world_size, 0.9, dtype, mode, alpha_min=0.0
    )
    _assert_all_finite(flat, grads, f"size_px={size_px} mode={mode} dtype={dtype}")


# --- 5: alpha exactly 0 and exactly 1 --------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_alpha_exactly_one_is_reachable_from_a_real_render(dtype: torch.dtype) -> None:
    """Pin the fixture: boundary uv + conf 1 + size_px 2 really does give alpha 1."""
    K = torch.tensor([[FX, 0.0, CX], [0.0, FX, CY], [0.0, 0.0, 1.0]], dtype=dtype)
    frags = build_sorted_fragments(
        torch.tensor([[BOUNDARY_XY, BOUNDARY_XY, BOUNDARY_DEPTH]], dtype=dtype),
        torch.tensor([BOUNDARY_SIZE], dtype=dtype),
        torch.tensor([1.0], dtype=dtype),
        K,
        torch.eye(3, dtype=dtype),
        torch.zeros(3, dtype=dtype),
        layer_grid(*IMAGE_HW, NUM_LAYERS),
        mode="trips",
        alpha_min=0.0,
    )
    assert int((frags.alpha == 1.0).sum()) == 1
    assert int((frags.alpha == 0.0).sum()) > 0


@pytest.mark.parametrize("mode", ("trips", "broadcast"))
@pytest.mark.parametrize("dtype", DTYPES)
def test_alpha_exactly_one_renders_finite(mode: str, dtype: torch.dtype) -> None:
    """`log1p(-alpha)` at alpha == 1 used to give -inf, then -inf - -inf = NaN.

    The 1e-12 epsilon that guarded it rounds back to 1.0 in float32, so this
    only ever failed in the trainer's dtype -- exactly where it matters.
    """
    flat, grads = _render_grads(
        (BOUNDARY_XY, BOUNDARY_XY, BOUNDARY_DEPTH),
        BOUNDARY_SIZE,
        1.0,
        dtype,
        mode,
        alpha_min=0.0,
    )
    _assert_all_finite(flat, grads, f"alpha==1 mode={mode} dtype={dtype}")


@pytest.mark.parametrize("dtype", DTYPES)
def test_composite_sorted_survives_alpha_exactly_zero_and_one(dtype: torch.dtype) -> None:
    """The compositor itself, on a hand-built segment holding both extremes."""
    alpha = torch.tensor([0.0, 0.5, 1.0, 0.25], dtype=dtype, requires_grad=True)
    feat = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 0.0, 0.5]], dtype=dtype)
    feat.requires_grad_(True)
    layer_pixel = torch.zeros(4, dtype=torch.int64)
    point_id = torch.arange(4, dtype=torch.int64)
    depth = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    offsets = torch.tensor([0, 4], dtype=torch.int64)

    out, t_final, _n_used, depth_sum = composite_sorted(
        layer_pixel, depth, point_id, alpha, offsets, feat
    )
    for name, tensor in (("out", out), ("t_final", t_final), ("depth_sum", depth_sum)):
        assert torch.isfinite(tensor).all(), f"non-finite {name}: {tensor}"
    # An alpha == 1 fragment must swallow everything behind it.
    assert float(t_final[0].detach()) == pytest.approx(0.0, abs=1e-6)
    (out.sum() + t_final.sum() + depth_sum.sum()).backward()
    assert torch.isfinite(alpha.grad).all(), f"non-finite d/d alpha: {alpha.grad}"
    assert torch.isfinite(feat.grad).all()


# --- the reduced single-fragment case --------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_single_fragment_on_a_boundary_at_an_exact_power_of_two(dtype: torch.dtype) -> None:
    """The whole hazard set reduced to one surviving fragment.

    Boundary uv at layer 1 (three of the four bilinear weights are exactly
    0, so the default alpha floor drops them), `size_px == 2` exactly
    (lower == upper, zero interpolation denominator, and the only layer
    mode "trilinear" selects), and a companion point at z == 0 whose only
    role is to be culled.
    """
    K = torch.tensor([[FX, 0.0, CX], [0.0, FX, CY], [0.0, 0.0, 1.0]], dtype=dtype)
    xyz = torch.tensor(
        [[LAYER1_BOUNDARY_XY, LAYER1_BOUNDARY_XY, BOUNDARY_DEPTH], [1.0, 0.5, 0.0]], dtype=dtype
    )
    size = torch.tensor([BOUNDARY_SIZE, 0.1], dtype=dtype)
    conf = torch.tensor([0.9, 0.9], dtype=dtype)
    frags = build_sorted_fragments(
        xyz,
        size,
        conf,
        K,
        torch.eye(3, dtype=dtype),
        torch.zeros(3, dtype=dtype),
        layer_grid(*IMAGE_HW, NUM_LAYERS),
        mode="trilinear",
        alpha_min=RASTER_ALPHA_MIN,
    )
    assert len(frags) == 1, f"expected exactly one fragment, got {len(frags)}"
    assert int(frags.layer[0]) == 1
    uv, _depth, size_px = project_points(
        xyz, size, K, torch.eye(3, dtype=dtype), torch.zeros(3, dtype=dtype)
    )
    assert float(uv[0, 0]) == LAYER1_BOUNDARY_UV
    assert float(size_px[0]) == 2.0

    flat, grads = _render_grads(
        (1.0, 0.5, 0.0),
        0.1,
        0.9,
        dtype,
        "trilinear",
        alpha_min=RASTER_ALPHA_MIN,
        anchor_xyz=(LAYER1_BOUNDARY_XY, LAYER1_BOUNDARY_XY, BOUNDARY_DEPTH),
        anchor_size=BOUNDARY_SIZE,
        anchor_conf=0.9,
    )
    _assert_all_finite(flat, grads, f"single fragment dtype={dtype}")
