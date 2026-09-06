"""The two emission implementations must be indistinguishable, not merely close.

Module: tests.test_raster_emit_impl
Purpose: `trippy.raster.emit.emit_fragments` has two implementations --
    "loop" (the original per-layer Python loop, kept as the readable
    statement of the layer rule) and "vectorised" (the default: one
    compaction of the culled points, one (L, M, 4) block, one
    `torch.nonzero`). The vectorised one exists only to make mode "trips"
    affordable on MPS (docs/ARCHITECTURE.md "Emission cost"), so the bar is
    *bit-identical output in the same order*, not approximate agreement.
Invariants:
    - Fragment ORDER is load-bearing: the depth sort is stable, so two
      implementations that emitted the same set in a different order would
      composite depth ties differently. Every assertion here is on the raw
      tensors, elementwise, with no sorting applied first.
    - Every mode x pixel_center x pyramid_halving combination is covered, plus
      the two degenerate cases (nothing culled in, nothing emitted).
Related docs: docs/ARCHITECTURE.md; docs/GEOMETRY.md "Pyramid level selection".
"""

from __future__ import annotations

import pytest
import torch
from test_raster_scenes import make_scene

from trippy.constants import RASTER_PIXEL_CENTERS, RASTER_PYRAMID_HALVINGS
from trippy.raster.emit import (
    EMIT_IMPLS,
    EMIT_MODES,
    build_sorted_fragments,
    cull_points,
    emit_fragments,
    layer_grid,
    project_points,
)

NUM_LAYERS = 5


def _emit_both(scene: dict, mode: str, pixel_center: str, pyramid_halving: str, alpha_min: float):
    """Project + cull once, then emit with each implementation."""
    grid = layer_grid(*scene["image_hw"], NUM_LAYERS, pyramid_halving=pyramid_halving)
    uv, depth, size_px = project_points(
        scene["xyz"], scene["size"], scene["K"], scene["R"], scene["t"]
    )
    valid = cull_points(uv, depth, size_px, grid)
    out = []
    for impl in EMIT_IMPLS:
        out.append(
            emit_fragments(
                uv,
                depth,
                size_px,
                scene["conf"],
                grid,
                mode=mode,
                valid=valid,
                alpha_min=alpha_min,
                pixel_center=pixel_center,
                impl=impl,
            )
        )
    return out


def _assert_identical(a, b) -> None:
    for field in ("layer_pixel", "layer", "pixel", "depth", "point_id", "alpha"):
        left = getattr(a, field)
        right = getattr(b, field)
        assert left.shape == right.shape, f"{field}: {left.shape} vs {right.shape}"
        assert left.dtype == right.dtype, f"{field}: {left.dtype} vs {right.dtype}"
        assert torch.equal(left, right), f"{field} differs"


@pytest.mark.parametrize("mode", EMIT_MODES)
@pytest.mark.parametrize("pixel_center", RASTER_PIXEL_CENTERS)
def test_implementations_are_bit_identical(mode: str, pixel_center: str) -> None:
    scene = make_scene(num_points=400, height=37, width=53, seed=3)
    loop, vec = _emit_both(scene, mode, pixel_center, "ceil", 1e-5)[::-1]
    _assert_identical(loop, vec)
    assert len(loop) > 0


@pytest.mark.parametrize("mode", EMIT_MODES)
@pytest.mark.parametrize("pyramid_halving", RASTER_PYRAMID_HALVINGS)
def test_implementations_agree_for_both_halvings(mode: str, pyramid_halving: str) -> None:
    scene = make_scene(num_points=300, height=40, width=40, seed=11)
    vec, loop = _emit_both(scene, mode, "half", pyramid_halving, 1e-5)
    _assert_identical(loop, vec)


@pytest.mark.parametrize("mode", EMIT_MODES)
def test_implementations_agree_with_no_alpha_floor(mode: str) -> None:
    """alpha_min=0 is TRIPS's own setting: zero-weight corners still take a slot."""
    scene = make_scene(num_points=200, height=32, width=32, seed=5)
    vec, loop = _emit_both(scene, mode, "half", "ceil", 0.0)
    _assert_identical(loop, vec)
    assert len(vec) > 0


@pytest.mark.parametrize("mode", EMIT_MODES)
def test_implementations_agree_when_valid_is_none(mode: str) -> None:
    scene = make_scene(num_points=120, height=32, width=32, seed=7)
    grid = layer_grid(*scene["image_hw"], NUM_LAYERS)
    uv, depth, size_px = project_points(
        scene["xyz"], scene["size"], scene["K"], scene["R"], scene["t"]
    )
    emitted = [
        emit_fragments(uv, depth, size_px, scene["conf"], grid, mode=mode, valid=None, impl=impl)
        for impl in ("vectorised", "loop")
    ]
    _assert_identical(emitted[1], emitted[0])


@pytest.mark.parametrize("mode", EMIT_MODES)
def test_nothing_culled_in_gives_the_same_empty_fragments(mode: str) -> None:
    """Every point behind the camera: both implementations return an empty list."""
    scene = make_scene(num_points=60, height=32, width=32, seed=2)
    grid = layer_grid(*scene["image_hw"], NUM_LAYERS)
    uv, depth, size_px = project_points(
        scene["xyz"], scene["size"], scene["K"], scene["R"], scene["t"]
    )
    none_valid = torch.zeros(uv.shape[0], dtype=torch.bool)
    emitted = [
        emit_fragments(
            uv, depth, size_px, scene["conf"], grid, mode=mode, valid=none_valid, impl=impl
        )
        for impl in ("vectorised", "loop")
    ]
    assert len(emitted[0]) == 0
    _assert_identical(emitted[1], emitted[0])


@pytest.mark.parametrize("mode", EMIT_MODES)
def test_an_impossible_alpha_floor_empties_both(mode: str) -> None:
    """alpha_min > 1 drops every fragment; the empty exit path must still match."""
    scene = make_scene(num_points=80, height=32, width=32, seed=4)
    vec, loop = _emit_both(scene, mode, "half", "ceil", 2.0)
    assert len(vec) == 0
    _assert_identical(loop, vec)


@pytest.mark.parametrize("mode", EMIT_MODES)
def test_gradients_match_between_implementations(mode: str) -> None:
    """The alpha path carries gradient; the two implementations must agree there too.

    Gradients are compared to float64 round-off rather than bit-for-bit,
    unlike the forward. The two paths scatter the same per-fragment
    derivatives back onto the same points, but through differently shaped
    reductions (L separate `index_select` backwards vs one), so the summation
    ORDER differs and the last ulp with it. Measured on this scene the worst
    disagreement is ~3e-16 relative -- round-off, not a different rule.
    """
    scene = make_scene(num_points=150, height=32, width=32, seed=9)
    grads = []
    for impl in EMIT_IMPLS:
        xyz = scene["xyz"].clone().requires_grad_(True)
        size = scene["size"].clone().requires_grad_(True)
        conf = scene["conf"].clone().requires_grad_(True)
        grid = layer_grid(*scene["image_hw"], NUM_LAYERS)
        uv, depth, size_px = project_points(xyz, size, scene["K"], scene["R"], scene["t"])
        valid = cull_points(uv, depth, size_px, grid)
        frags = emit_fragments(
            uv, depth, size_px, conf, grid, mode=mode, valid=valid, impl=impl
        )
        # A weighted sum, so the gradient depends on *which* fragment is which,
        # not only on their total.
        weights = torch.linspace(0.5, 1.5, len(frags), dtype=frags.alpha.dtype)
        (frags.alpha * weights).sum().backward()
        grads.append((xyz.grad, size.grad, conf.grad))
    for left, right in zip(grads[0], grads[1], strict=True):
        if left is None:  # e.g. `size` in mode "broadcast": factor is a constant 1
            assert right is None
            continue
        assert torch.allclose(left, right, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", EMIT_MODES)
def test_build_sorted_fragments_matches_across_implementations(mode: str) -> None:
    """The whole project -> cull -> emit -> sort -> segment pipeline, both ways.

    Also covers `sort_fragments`' `max_layer_pixel` shortcut, which
    `build_sorted_fragments` always supplies.
    """
    scene = make_scene(num_points=250, height=36, width=44, seed=13)
    grid = layer_grid(*scene["image_hw"], NUM_LAYERS)
    built = [
        build_sorted_fragments(
            scene["xyz"],
            scene["size"],
            scene["conf"],
            scene["K"],
            scene["R"],
            scene["t"],
            grid,
            mode=mode,
            emit_impl=impl,
        )
        for impl in EMIT_IMPLS
    ]
    _assert_identical(built[0], built[1])
    assert torch.equal(built[0].offsets, built[1].offsets)


def test_unknown_impl_names_raise() -> None:
    scene = make_scene(num_points=40, height=32, width=32, seed=1)
    grid = layer_grid(*scene["image_hw"], NUM_LAYERS)
    uv, depth, size_px = project_points(
        scene["xyz"], scene["size"], scene["K"], scene["R"], scene["t"]
    )
    with pytest.raises(ValueError, match="impl must be one of"):
        emit_fragments(uv, depth, size_px, scene["conf"], grid, impl="fast")


@pytest.mark.parametrize("mode", EMIT_MODES)
def test_a_single_layer_pyramid_still_agrees(mode: str) -> None:
    """L = 1 degenerates the layer loop; the two implementations must still match."""
    scene = make_scene(num_points=100, height=32, width=32, seed=17)
    grid = layer_grid(*scene["image_hw"], 1)
    uv, depth, size_px = project_points(
        scene["xyz"], scene["size"], scene["K"], scene["R"], scene["t"]
    )
    valid = cull_points(uv, depth, size_px, grid)
    emitted = [
        emit_fragments(
            uv, depth, size_px, scene["conf"], grid, mode=mode, valid=valid, impl=impl
        )
        for impl in ("vectorised", "loop")
    ]
    _assert_identical(emitted[1], emitted[0])


@pytest.mark.parametrize("mode", EMIT_MODES)
def test_an_enormous_footprint_agrees_and_stays_in_bounds(mode: str) -> None:
    """A point whose splat is far wider than the image.

    `size_px` is only bounded by `cull_points`' generous box, so `floor(uv/2^l)`
    can be a large integer. Both implementations index pixels in int64 and must
    still agree -- and no fragment may land outside its own layer, which is the
    failure mode a narrower integer type would produce silently.
    """
    scene = make_scene(num_points=60, height=32, width=32, seed=19)
    grid = layer_grid(*scene["image_hw"], NUM_LAYERS)
    uv, depth, size_px = project_points(
        scene["xyz"], scene["size"], scene["K"], scene["R"], scene["t"]
    )
    size_px = size_px.clone()
    size_px[0] = 5.0e5
    uv = uv.clone()
    uv[0] = torch.tensor([1.0e5, -1.0e5], dtype=uv.dtype)
    valid = cull_points(uv, depth, size_px, grid)
    assert bool(valid[0]), "the huge-footprint point must survive the cull for this test to bite"
    emitted = [
        emit_fragments(
            uv, depth, size_px, scene["conf"], grid, mode=mode, valid=valid, impl=impl
        )
        for impl in ("vectorised", "loop")
    ]
    _assert_identical(emitted[1], emitted[0])
    frags = emitted[0]
    for layer, (height, width) in enumerate(grid.shapes):
        on_layer = frags.layer == layer
        assert bool((frags.pixel[on_layer] >= 0).all())
        assert bool((frags.pixel[on_layer] < height * width).all())
