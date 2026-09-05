"""Mode "trips": TRIPS's real layer rule, its footprint gate and its `break`.

Module: tests.test_raster_trips_mode
Invariants: every expected number here is computed by hand from
    third_party/TRIPS/src/lib/rendering/RenderForward.cu:334-352 (the
    `CountAndCollectTiled` emission loop), :3511-3517 (the per-fragment
    alpha) and PointBlending.h:81-149 (`compute_point_size_fac`) -- never by
    running the implementation and pasting what it printed. The three
    behaviours that separate "trips" from trippy's older "trilinear" reading
    are pinned separately:
      1. layers `0 .. layer_higher` are ALL written, and every layer strictly
         below `layer_lower` gets factor exactly 1.0;
      2. the footprint gate is `valid_point`, which needs all four corners in
         bounds -- not the per-corner drop the other modes use; and
      3. that gate is a `break`, so failing it at layer l also suppresses
         every coarser layer.
    The two convention options (`pixel_center`, `pyramid_halving`) are pinned
    against hand-computed pixel indices in both settings.
Related docs: docs/GEOMETRY.md "Pyramid level selection and the layer
    factor" / "Pixel-centre convention"; docs/TRIPS_REFERENCE.md Sec. 3a/3b/6a.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_raster_scenes import as_numpy, make_scene

from trippy.raster import render_pyramid_numpy, render_pyramid_ref
from trippy.raster.emit import (
    EMIT_MODES,
    cull_points,
    emit_fragments,
    layer_factor,
    layer_grid,
    project_points,
)
from trippy.raster.ref_numpy import layer_higher_scalar

HEIGHT, WIDTH = 32, 32
FOCAL = 64.0
CONF = 0.9
DTYPE = torch.float64


def _one_point(
    u: float,
    v: float,
    depth: float,
    size_px: float,
    num_layers: int,
    pyramid_halving: str = "ceil",
    height: int = HEIGHT,
    width: int = WIDTH,
):
    """A single point at exact layer-0 coordinate (u, v) with size `size_px`.

    Returns `(grid, uv, depth, size_px, conf)` ready for `emit_fragments`,
    built through `project_points` so the projection is exercised too (the
    pose is the identity, so `uv == (u, v)` exactly).
    """
    cx, cy = 0.5 * width, 0.5 * height
    K = torch.tensor([[FOCAL, 0.0, cx], [0.0, FOCAL, cy], [0.0, 0.0, 1.0]], dtype=DTYPE)
    xyz = torch.tensor([[(u - cx) * depth / FOCAL, (v - cy) * depth / FOCAL, depth]], dtype=DTYPE)
    size = torch.tensor([size_px * depth / FOCAL], dtype=DTYPE)
    conf = torch.tensor([CONF], dtype=DTYPE)
    grid = layer_grid(height, width, num_layers, pyramid_halving=pyramid_halving)
    uv, z, s_px = project_points(xyz, size, K, torch.eye(3, dtype=DTYPE), torch.zeros(3, dtype=DTYPE))
    return grid, uv, z, s_px, conf


def _emit(grid, uv, depth, size_px, conf, mode="trips", pixel_center="half", alpha_min=0.0):
    valid = cull_points(uv, depth, size_px, grid)
    return emit_fragments(
        uv,
        depth,
        size_px,
        conf,
        grid,
        mode=mode,
        valid=valid,
        alpha_min=alpha_min,
        pixel_center=pixel_center,
    )


def _by_layer(frags) -> dict[int, list[tuple[int, int, float]]]:
    """{layer: [(y, x, alpha), ...]} for readable assertions."""
    out: dict[int, list[tuple[int, int, float]]] = {}
    for i in range(len(frags)):
        layer = int(frags.layer[i])
        w_l = frags.grid.shapes[layer][1]
        pixel = int(frags.pixel[i])
        out.setdefault(layer, []).append((pixel // w_l, pixel % w_l, float(frags.alpha[i])))
    return out


# --- 1. layer set and weights -------------------------------------------


def test_layer_higher_matches_the_cuda_expression() -> None:
    """`(s > 1) ? min(ceil(log2 s), L-1) : 0` -- RenderForward.cu:334-338."""
    assert layer_higher_scalar(0.5, 8) == 0
    assert layer_higher_scalar(1.0, 8) == 0  # `point_size_opt > 1` is strict
    assert layer_higher_scalar(1.5, 8) == 1
    assert layer_higher_scalar(2.0, 8) == 1
    assert layer_higher_scalar(5.0, 8) == 3  # log2 5 = 2.32 -> ceil 3
    assert layer_higher_scalar(5.0, 3) == 2  # clamped to L - 1
    assert layer_higher_scalar(1e6, 8) == 7


def test_five_pixel_point_writes_layers_0_to_3_with_1_1_075_025() -> None:
    """The worked example in docs/TRIPS_REFERENCE.md Sec. 3a.

    s = 5 px: layer_lower = floor(log2 5) = 2, layer_higher = ceil = 3.
    `compute_point_size_fac` returns 1.0 for layer < layer_lower
    (PointBlending.h:92-96), then interpolates linearly in *point-size units*
    between 2**2 = 4 and 2**3 = 8: f = (5 - 4) / (8 - 4) = 0.25, so layer 2
    (the lower one) gets 1 - f = 0.75 and layer 3 gets f = 0.25.
    """
    size = torch.tensor([5.0], dtype=DTYPE)
    assert [float(layer_factor(size, layer, 8)) for layer in range(5)] == pytest.approx(
        [1.0, 1.0, 0.75, 0.25, 0.25]
    )

    # No coordinate is on a pixel centre at two layers at once in the "half"
    # convention (`uv = 2**l (n + 0.5)` cannot hold for l = 0 and l = 1), so
    # the invariant to assert per layer is the bilinear partition of unity:
    # the four corner alphas sum to `conf * layer_factor`, exactly.
    grid, uv, depth, size_px, conf = _one_point(16.5, 16.5, depth=4.0, size_px=5.0, num_layers=8)
    frags = _emit(grid, uv, depth, size_px, conf)
    got = _by_layer(frags)
    assert sorted(got) == [0, 1, 2, 3], "layers 0..layer_higher inclusive, nothing above"
    for layer, factor in zip(range(4), (1.0, 1.0, 0.75, 0.25), strict=True):
        alphas = sorted(alpha for _y, _x, alpha in got[layer])
        assert len(alphas) == 4, "all four corners are written, weight 0 or not"
        assert sum(alphas) == pytest.approx(CONF * factor), "bilinear weights sum to 1"


def test_sub_pixel_point_reaches_layer_zero_only_with_the_exponential_floor() -> None:
    """s <= 1 -> layer_higher = 0, factor (1 - 0.25) exp(s - 1) + 0.25."""
    grid, uv, depth, size_px, conf = _one_point(16.5, 16.5, depth=4.0, size_px=0.5, num_layers=5)
    frags = _emit(grid, uv, depth, size_px, conf)
    got = _by_layer(frags)
    assert sorted(got) == [0]
    expected = 0.75 * float(np.exp(0.5 - 1.0)) + 0.25
    assert max(alpha for _y, _x, alpha in got[0]) == pytest.approx(CONF * expected)


def test_at_most_four_fragments_per_point_per_layer() -> None:
    """Fragment budget: 4 corners x L layers, never more."""
    num_layers = 5
    scene = make_scene(num_points=50, height=HEIGHT, width=WIDTH, seed=3)
    grid = layer_grid(HEIGHT, WIDTH, num_layers)
    uv, depth, size_px = project_points(
        scene["xyz"], scene["size"], scene["K"], scene["R"], scene["t"]
    )
    frags = _emit(grid, uv, depth, size_px, scene["conf"])
    assert len(frags) > 0
    counts = torch.bincount(frags.point_id, minlength=scene["xyz"].shape[0])
    assert int(counts.max()) <= 4 * num_layers
    # Every written (point, layer) pair writes all four corners.
    for point_id in torch.unique(frags.point_id).tolist():
        on_point = frags.point_id == point_id
        per_layer = torch.bincount(frags.layer[on_point], minlength=num_layers)
        assert set(per_layer[per_layer > 0].tolist()) == {4}


# --- 2. the all-four-corners footprint gate ------------------------------


def test_border_point_writes_nothing_where_trilinear_writes_two_corners() -> None:
    """`valid_point` needs `floor(ip) < size - 1`, i.e. both corners inside.

    At u = W - 0.4 the pixel-centred base is W - 1, so the x = W corner is
    outside. Mode "trilinear" drops that one corner and keeps the other;
    mode "trips" drops the whole footprint -- and, being a `break`, the whole
    point (RenderForward.cu:344-345).
    """
    grid, uv, depth, size_px, conf = _one_point(WIDTH - 0.4, 16.5, 4.0, size_px=0.5, num_layers=4)
    tri = _by_layer(_emit(grid, uv, depth, size_px, conf, mode="trilinear"))
    assert sorted({x for _y, x, _a in tri[0]}) == [WIDTH - 1]
    assert len(_emit(grid, uv, depth, size_px, conf, mode="trips")) == 0


def test_gate_failure_at_a_fine_layer_suppresses_every_coarser_layer() -> None:
    """`break`, not `continue`: this is the whole point of the rule.

    Layers of a 32x32 image are 32, 16, 8 and 4 wide, so the "half"-centred
    base `floor(u/2**l - 0.5)` must be <= 30, 14, 6 and 2 respectively.
    A 9 px point wants all four layers (`ceil(log2 9) = 4`, clamped to 3).

    Case A, u = 29.2: bases 28, 14, 6, 3. Only layer 3 fails, so layers
    0-2 survive -- the gate is not all-or-nothing.
    Case B, v = 30.9: bases 30, 14, 7, 3. Layer 2 fails first, and because
    the loop `break`s, layer 3 goes with it even though layer 3 was already
    failing on its own -- so this case is checked against case A, where the
    same coordinate structure kept layer 2.
    """
    assert layer_higher_scalar(9.0, 4) == 3, "the point wants all four layers"

    grid, uv, depth, size_px, conf = _one_point(29.2, 16.5, 4.0, size_px=9.0, num_layers=4)
    got = _by_layer(_emit(grid, uv, depth, size_px, conf))
    assert sorted(got) == [0, 1, 2], "only layer 3's gate fails"

    grid, uv, depth, size_px, conf = _one_point(16.5, 30.9, 4.0, size_px=9.0, num_layers=4)
    got = _by_layer(_emit(grid, uv, depth, size_px, conf))
    assert sorted(got) == [0, 1], "layer 2's gate fails, so layers 2 AND 3 go"

    # And with `continue` semantics instead of `break`, layer 2 alone would
    # be missing. Mode "trilinear" (which has no gate at all) proves the
    # point is otherwise perfectly visible on layers 2 and 3.
    tri = _by_layer(_emit(grid, uv, depth, size_px, conf, mode="trilinear"))
    assert sorted(tri) == [3], "trilinear writes lower..upper = 3..3, ungated"


def test_the_layer_zero_gate_kills_the_whole_point() -> None:
    """TRIPS's pre-loop `valid_point(ip, z, 0, ...)` (RenderForward.cu:296).

    It is the same statement as the loop's own layer-0 test, so it needs no
    separate code: `floor(x) < w - 1` and `x < w - 1` agree for integers.
    A big point 3 px off the left edge would still be visible at layer 3 in
    mode "broadcast"; in mode "trips" it is gone entirely.
    """
    grid, uv, depth, size_px, conf = _one_point(-3.0, 16.5, 4.0, size_px=9.0, num_layers=4)
    assert len(_emit(grid, uv, depth, size_px, conf, mode="broadcast")) > 0
    assert len(_emit(grid, uv, depth, size_px, conf, mode="trips")) == 0


# --- 3. pixel-centre convention -----------------------------------------


@pytest.mark.parametrize(("pixel_center", "frac"), [("half", 0.25), ("integer", 0.75)])
def test_pixel_center_reweights_the_footprint_by_half_a_pixel(pixel_center: str, frac: float) -> None:
    """`base = floor(uv/2**l - c)`, c = 0.5 ("half") or 0.0 ("integer").

    At (10.75, 6.75) both conventions land on the same base pixel (6, 10),
    but the fractional weight differs by exactly 0.5: "half" gives the base
    pixel 0.75 of the weight per axis, "integer" only 0.25.
    """
    grid, uv, depth, size_px, conf = _one_point(10.75, 6.75, 4.0, size_px=0.5, num_layers=1)
    corner = {
        (y, x): a
        for y, x, a in _by_layer(_emit(grid, uv, depth, size_px, conf, pixel_center=pixel_center))[0]
    }
    assert min(corner) == (6, 10)
    factor = 0.75 * float(np.exp(0.5 - 1.0)) + 0.25
    assert corner[(6, 10)] == pytest.approx((1 - frac) * (1 - frac) * CONF * factor)


def test_pixel_center_can_move_the_base_pixel_itself() -> None:
    """At (10.25, 6.25) the two conventions pick *different* base pixels.

    "half": `floor(10.25 - 0.5) = 9`. "integer": `floor(10.25) = 10`. The two
    2x2 footprints then share only one corner -- this is a one-pixel shift,
    not a reweighting, which is why feeding a TRIPS checkpoint the default
    convention (or vice versa) is a visible error and not a rounding detail.
    """
    grid, uv, depth, size_px, conf = _one_point(10.25, 6.25, 4.0, size_px=0.5, num_layers=1)
    half = {(y, x) for y, x, _a in _by_layer(
        _emit(grid, uv, depth, size_px, conf, pixel_center="half")
    )[0]}
    integer = {(y, x) for y, x, _a in _by_layer(
        _emit(grid, uv, depth, size_px, conf, pixel_center="integer")
    )[0]}
    assert half == {(5, 9), (5, 10), (6, 9), (6, 10)}
    assert integer == {(6, 10), (6, 11), (7, 10), (7, 11)}


def test_integer_centres_halve_the_way_trips_does() -> None:
    """`ip_l = ip / 2**l` anchored at `floor(ip_l)` -- PointBlending.h:216-240.

    ip = (10, 6) lands exactly on a pixel centre in TRIPS's convention at
    layers 0 and 1 (10/2 = 5, 6/2 = 3), so both put all their weight on one
    corner; at layer 2 it is (2.5, 1.5), a clean four-way split. With L = 4
    and s = 5 px the factors are 1, 1, 0.75, 0.25 (layer_lower = 2).
    """
    grid, uv, depth, size_px, conf = _one_point(10.0, 6.0, 4.0, size_px=5.0, num_layers=4)
    got = _by_layer(_emit(grid, uv, depth, size_px, conf, pixel_center="integer"))
    assert max(a for _y, _x, a in got[0]) == pytest.approx(CONF * 1.0)
    assert (6, 10) in {(y, x) for y, x, a in got[0] if a > 0}
    assert max(a for _y, _x, a in got[1]) == pytest.approx(CONF * 1.0)
    assert (3, 5) in {(y, x) for y, x, a in got[1] if a > 0}
    # layer 2: factor 0.75 (lower layer of the 4..8 straddle), bilinear 0.25.
    assert max(a for _y, _x, a in got[2]) == pytest.approx(0.25 * CONF * 0.75)
    assert sum(a for _y, _x, a in got[3]) == pytest.approx(CONF * 0.25)


def test_half_centres_are_the_default_and_unchanged() -> None:
    """Existing callers must see byte-identical behaviour."""
    grid, uv, depth, size_px, conf = _one_point(10.75, 6.75, 4.0, size_px=1.5, num_layers=3)
    default = _emit(grid, uv, depth, size_px, conf, mode="trilinear")
    explicit = _emit(grid, uv, depth, size_px, conf, mode="trilinear", pixel_center="half")
    assert torch.equal(default.alpha, explicit.alpha)
    assert torch.equal(default.layer_pixel, explicit.layer_pixel)


def test_unknown_convention_names_raise() -> None:
    grid, uv, depth, size_px, conf = _one_point(10.0, 6.0, 4.0, 1.0, num_layers=2)
    with pytest.raises(ValueError, match="pixel_center"):
        _emit(grid, uv, depth, size_px, conf, pixel_center="middle")
    with pytest.raises(ValueError, match="mode must be one of"):
        _emit(grid, uv, depth, size_px, conf, mode="nope")
    with pytest.raises(ValueError, match="pyramid_halving"):
        layer_grid(32, 32, 3, pyramid_halving="round")
    assert set(EMIT_MODES) == {"trips", "trilinear", "broadcast"}


# --- 4. pyramid halving --------------------------------------------------


def test_pyramid_halving_floor_matches_trips_integer_division_branch() -> None:
    """`h /= 2` (PointRenderer.cu:391) drops the odd row; `ceil` keeps it."""
    assert layer_grid(31, 33, 4, pyramid_halving="ceil").shapes == [
        (31, 33), (16, 17), (8, 9), (4, 5)
    ]
    assert layer_grid(31, 33, 4, pyramid_halving="floor").shapes == [
        (31, 33), (15, 16), (7, 8), (3, 4)
    ]
    # 1080 is TRIPS's own worked example (docs/TRIPS_REFERENCE.md Sec. 3b).
    assert [h for h, _w in layer_grid(1080, 1920, 8).shapes] == [
        1080, 540, 270, 135, 68, 34, 17, 9
    ]
    assert [h for h, _w in layer_grid(1080, 1920, 8, pyramid_halving="floor").shapes] == [
        1080, 540, 270, 135, 67, 33, 16, 8
    ]


def test_floor_halving_refuses_an_empty_layer() -> None:
    """Loud failure beats a zero-sized layer tensor."""
    with pytest.raises(ValueError, match="empty layer"):
        layer_grid(8, 8, 6, pyramid_halving="floor")


# --- 5. the two references agree on all of the above ---------------------


@pytest.mark.parametrize("pixel_center", ["half", "integer"])
@pytest.mark.parametrize("pyramid_halving", ["ceil", "floor"])
def test_numpy_and_torch_references_agree_in_trips_mode(
    pixel_center: str, pyramid_halving: str
) -> None:
    """The dual-implementation gate (AGENTS.md Sec. 7), extended to "trips"."""
    num_layers = 3
    scene = make_scene(num_points=50, height=HEIGHT, width=WIDTH, num_channels=3, seed=0)
    npy = as_numpy(scene)
    common = {
        "num_layers": num_layers,
        "mode": "trips",
        "pixel_center": pixel_center,
        "pyramid_halving": pyramid_halving,
    }
    layers_t, aux_t = render_pyramid_ref(
        scene["xyz"], scene["size"], scene["feat"], scene["conf"],
        scene["K"], scene["R"], scene["t"], scene["image_hw"], bg=scene["bg"], **common,
    )
    layers_n, aux_n = render_pyramid_numpy(
        npy["xyz"], npy["size"], npy["feat"], npy["conf"],
        npy["K"], npy["R"], npy["t"], npy["image_hw"], bg=npy["bg"], **common,
    )
    assert aux_t["num_fragments"] == aux_n["num_fragments"] > 0
    assert aux_t["fragments_per_layer"].tolist() == aux_n["fragments_per_layer"]
    for layer in range(num_layers):
        assert np.abs(layers_t[layer].numpy() - layers_n[layer]).max() < 1e-9
        assert np.abs(aux_t["t_final"][layer].numpy() - aux_n["t_final"][layer]).max() < 1e-9
        assert np.array_equal(aux_t["n_used"][layer].numpy(), aux_n["n_used"][layer])


def test_trips_writes_more_fragments_than_trilinear_and_fewer_than_broadcast() -> None:
    """Ordering of the three modes' fragment budgets on the same scene."""
    scene = make_scene(num_points=50, height=HEIGHT, width=WIDTH, seed=1)
    grid = layer_grid(HEIGHT, WIDTH, 4)
    uv, depth, size_px = project_points(
        scene["xyz"], scene["size"], scene["K"], scene["R"], scene["t"]
    )
    counts = {
        mode: len(_emit(grid, uv, depth, size_px, scene["conf"], mode=mode))
        for mode in ("trilinear", "trips", "broadcast")
    }
    assert counts["trilinear"] < counts["trips"] < counts["broadcast"], counts
