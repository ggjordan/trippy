"""Fragment bounds: drop out-of-range pixels, never clamp them.

Module: tests.test_raster_bounds
Invariants: docs/GEOMETRY.md historical bug class 3 -- "padded pixels
    unprojected as scene". A fragment whose pixel index falls outside its own
    pyramid layer must vanish, not be clamped onto the border (which would
    smear every off-screen point into a bright rim and corrupt gradients).
    These tests also pin the coarse cull in trippy.raster.emit.cull_points as
    conservative: it must never remove a point that a coarse layer would
    still have drawn.
Related docs: docs/GEOMETRY.md; docs/ARCHITECTURE.md ("Geometry: padding
    test (no fragment outside crop)").
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_raster_scenes import as_numpy, make_scene

from trippy.raster import layer_grid, render_pyramid_numpy, render_pyramid_ref
from trippy.raster.emit import cull_points, emit_fragments, project_points

NUM_LAYERS = 4
HEIGHT, WIDTH = 32, 32
FOCAL = 64.0


def _one_point_scene(u: float, v: float, depth: float, size_px: float, num_layers: int = NUM_LAYERS):
    """A single point placed at an exact layer-0 pixel coordinate.

    Args:
        u, v: continuous layer-0 pixel coordinates (corner origin).
        depth: camera-space z, world units.
        size_px: wanted projected size in layer-0 pixels.
        num_layers: pyramid depth.

    Returns:
        (grid, xyz, size, conf, K, R, t) all float64 torch tensors, with an
        identity pose so `uv` is exactly (u, v).
    """
    cx, cy = 0.5 * WIDTH, 0.5 * HEIGHT
    K = torch.tensor(
        [[FOCAL, 0.0, cx], [0.0, FOCAL, cy], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    xyz = torch.tensor(
        [[(u - cx) * depth / FOCAL, (v - cy) * depth / FOCAL, depth]], dtype=torch.float64
    )
    size = torch.tensor([size_px * depth / FOCAL], dtype=torch.float64)
    conf = torch.tensor([0.9], dtype=torch.float64)
    grid = layer_grid(HEIGHT, WIDTH, num_layers)
    return grid, xyz, size, conf, K, torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)


def _emit(grid, xyz, size, conf, K, R, t, mode: str = "broadcast"):
    uv, depth, size_px = project_points(xyz, size, K, R, t)
    valid = cull_points(uv, depth, size_px, grid)
    return uv, emit_fragments(uv, depth, size_px, conf, grid, mode=mode, valid=valid)


@pytest.mark.parametrize("mode", ["trilinear", "broadcast"])
def test_no_fragment_lands_outside_its_layer(mode: str) -> None:
    """Every emitted pixel index decodes to a valid (y, x) in its own layer."""
    scene = make_scene(seed=0)
    grid = layer_grid(*scene["image_hw"], NUM_LAYERS)
    uv, depth, size_px = project_points(
        scene["xyz"], scene["size"], scene["K"], scene["R"], scene["t"]
    )
    valid = cull_points(uv, depth, size_px, grid)
    frags = emit_fragments(uv, depth, size_px, scene["conf"], grid, mode=mode, valid=valid)
    assert len(frags) > 0
    for layer, (h_l, w_l) in enumerate(grid.shapes):
        on_layer = frags.layer == layer
        pixels = frags.pixel[on_layer]
        assert bool((pixels >= 0).all())
        assert bool((pixels < h_l * w_l).all())
        assert bool(((pixels // w_l) < h_l).all())
        # The flat pyramid index must land in this layer's slice.
        flat = frags.layer_pixel[on_layer]
        assert bool((flat >= grid.offsets[layer]).all())
        assert bool((flat < grid.offsets[layer] + h_l * w_l).all())


@pytest.mark.parametrize(
    ("u", "kept_x"),
    [
        (-0.3, [0]),  # base = floor(-0.8) = -1, so only the x = 0 corner survives
        (0.4, [0]),  # base = floor(-0.1) = -1
        (0.6, [0, 1]),  # base = 0, both corners inside
        (WIDTH - 0.4, [WIDTH - 1]),  # base = W - 1, the x = W corner is dropped
        (WIDTH + 0.2, [WIDTH - 1]),
    ],
)
def test_border_straddling_point_contributes_only_in_bounds(u: float, kept_x: list[int]) -> None:
    """A footprint hanging off the edge loses exactly its outside corners."""
    grid, *args = _one_point_scene(u, 16.5, depth=4.0, size_px=0.5)
    _, frags = _emit(grid, *args)
    layer0 = frags.layer == 0
    xs = sorted({int(pixel) % WIDTH for pixel in frags.pixel[layer0]})
    assert xs == kept_x


def test_out_of_bounds_pixels_are_dropped_not_clamped() -> None:
    """Clamping would pile every off-screen point onto column 0 / W-1."""
    # Far enough out that even the coarsest layer sees nothing.
    grid, *args = _one_point_scene(-200.0, 16.5, depth=4.0, size_px=0.5)
    _, frags = _emit(grid, *args)
    assert len(frags) == 0


def test_coarse_cull_keeps_points_only_a_coarse_layer_can_see() -> None:
    """A point 3 px off the edge is invisible at layer 0 but drawn at layer 3.

    This is the case a naive "1 pixel of margin" cull silently drops: at
    layer 3 the coordinate is u / 8 = -0.375, whose pixel-centred base is
    -1, so the x = 0 corner of the footprint is still inside that layer.
    """
    grid, *args = _one_point_scene(-3.0, 16.5, depth=4.0, size_px=0.5)
    _, frags = _emit(grid, *args, mode="broadcast")
    layers_hit = sorted({int(layer) for layer in frags.layer})
    assert 0 not in layers_hit
    assert NUM_LAYERS - 1 in layers_hit


def test_torch_and_numpy_agree_on_border_heavy_scenes() -> None:
    """The reference pair must agree exactly where the drop rule bites."""
    scene = make_scene(seed=12)
    npy = as_numpy(scene)
    layers_t, aux_t = render_pyramid_ref(
        scene["xyz"],
        scene["size"],
        scene["feat"],
        scene["conf"],
        scene["K"],
        scene["R"],
        scene["t"],
        scene["image_hw"],
        num_layers=NUM_LAYERS,
        mode="broadcast",
        bg=scene["bg"],
    )
    layers_n, aux_n = render_pyramid_numpy(
        npy["xyz"],
        npy["size"],
        npy["feat"],
        npy["conf"],
        npy["K"],
        npy["R"],
        npy["t"],
        npy["image_hw"],
        num_layers=NUM_LAYERS,
        mode="broadcast",
        bg=npy["bg"],
    )
    assert aux_t["num_fragments"] == aux_n["num_fragments"]
    for layer in range(NUM_LAYERS):
        assert np.abs(layers_t[layer].numpy() - layers_n[layer]).max() < 1e-6


def test_points_behind_the_camera_emit_nothing() -> None:
    """Depth sign bug class: z <= znear must be culled before projection."""
    grid, xyz, size, conf, K, R, t = _one_point_scene(16.5, 16.5, depth=4.0, size_px=2.0)
    behind = xyz.clone()
    behind[:, 2] = -4.0
    _, frags = _emit(grid, behind, size, conf, K, R, t)
    assert len(frags) == 0


def test_odd_sized_images_keep_their_last_row_and_column() -> None:
    """ceil halving (docs/GEOMETRY.md), not TRIPS's integer division."""
    grid = layer_grid(31, 33, 4)
    assert grid.shapes == [(31, 33), (16, 17), (8, 9), (4, 5)]
    assert grid.total == sum(h * w for h, w in grid.shapes)
    assert grid.offsets == [0, 31 * 33, 31 * 33 + 16 * 17, 31 * 33 + 16 * 17 + 8 * 9]
