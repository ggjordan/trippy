"""Fragment ordering and segment offsets: both sort paths must be identical.

Module: tests.test_raster_sort
Invariants: the composite int64 key and the two-stable-sort fallback are
    interchangeable. docs/LIMITATIONS.md flags int64 argsort at ~50M elements
    as unverified, so the fallback has to be provably equivalent before it is
    ever switched on in a training run.
Related docs: docs/ARCHITECTURE.md ("int64 argsort by (layer, pixel, depth)",
    "fallback: two stable 32-bit sorts"); docs/LIMITATIONS.md.
"""

from __future__ import annotations

import pytest
import torch
from test_raster_scenes import make_scene

from trippy.constants import RASTER_SORT_MAX_LAYER_PIXELS
from trippy.raster import build_sorted_fragments, layer_grid, render_pyramid_ref
from trippy.raster.sort import fragment_rank, segment_offsets, sort_fragments


def _random_fragments(num_fragments: int, num_pixels: int, seed: int):
    """Random (layer_pixel, depth) fragment keys, with deliberate depth ties."""
    generator = torch.Generator().manual_seed(seed)
    layer_pixel = torch.randint(0, num_pixels, (num_fragments,), generator=generator)
    # Quantised depths so exact ties happen often and the tie-break is tested.
    depth = (torch.randint(1, 40, (num_fragments,), generator=generator).double() / 8.0).float()
    return layer_pixel, depth


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_composite_and_two_pass_sorts_are_identical(seed: int) -> None:
    """Same permutation, element for element, including on depth ties."""
    layer_pixel, depth = _random_fragments(4000, 97, seed)
    perm_composite = sort_fragments(layer_pixel, depth, method="composite")
    perm_two_pass = sort_fragments(layer_pixel, depth, method="two_pass")
    assert torch.equal(perm_composite, perm_two_pass)


def test_sorted_order_is_layer_pixel_then_depth() -> None:
    """The contract the Metal kernel relies on: contiguous, depth-ascending."""
    layer_pixel, depth = _random_fragments(2000, 53, seed=5)
    perm = sort_fragments(layer_pixel, depth)
    lp_sorted = layer_pixel[perm]
    depth_sorted = depth[perm]
    assert bool((lp_sorted[1:] >= lp_sorted[:-1]).all())
    same_segment = lp_sorted[1:] == lp_sorted[:-1]
    ascending = depth_sorted[1:] >= depth_sorted[:-1]
    assert bool((ascending | ~same_segment).all())


def test_empty_fragment_list_is_handled() -> None:
    """A frame where every point is culled must not crash the pipeline."""
    empty_lp = torch.zeros(0, dtype=torch.int64)
    empty_depth = torch.zeros(0, dtype=torch.float32)
    for method in ("composite", "two_pass"):
        perm = sort_fragments(empty_lp, empty_depth, method=method)
        assert perm.numel() == 0
    offsets = segment_offsets(empty_lp, 9)
    assert offsets.numel() == 10
    assert int(offsets.max()) == 0


@pytest.mark.parametrize("method", ["searchsorted", "bincount"])
def test_segment_offsets_match_a_manual_count(method: str) -> None:
    """offsets[p] is the first slot of layer-pixel p; offsets[-1] == F."""
    num_pixels = 41
    layer_pixel, depth = _random_fragments(900, num_pixels, seed=8)
    lp_sorted = layer_pixel[sort_fragments(layer_pixel, depth)]
    offsets = segment_offsets(lp_sorted, num_pixels, method=method)

    assert offsets.shape == (num_pixels + 1,)
    assert int(offsets[0]) == 0
    assert int(offsets[-1]) == lp_sorted.numel()
    expected = [0]
    for pixel in range(num_pixels):
        expected.append(expected[-1] + int((lp_sorted == pixel).sum()))
    assert offsets.tolist() == expected


def test_segment_offsets_methods_agree() -> None:
    """searchsorted and bincount must produce the same array."""
    layer_pixel, depth = _random_fragments(1500, 37, seed=9)
    lp_sorted = layer_pixel[sort_fragments(layer_pixel, depth)]
    a = segment_offsets(lp_sorted, 37, method="searchsorted")
    b = segment_offsets(lp_sorted, 37, method="bincount")
    assert torch.equal(a, b)


def test_fragment_rank_counts_from_the_nearest_fragment() -> None:
    """rank 0 is the front-most fragment of each layer-pixel."""
    lp_sorted = torch.tensor([0, 0, 0, 2, 2, 5], dtype=torch.int64)
    offsets = segment_offsets(lp_sorted, 6)
    assert fragment_rank(lp_sorted, offsets).tolist() == [0, 1, 2, 0, 1, 0]


def test_composite_key_overflow_is_refused_not_silently_wrong() -> None:
    """An image too big for the 32-bit depth packing must raise, not wrap."""
    layer_pixel = torch.tensor([0, RASTER_SORT_MAX_LAYER_PIXELS], dtype=torch.int64)
    depth = torch.tensor([1.0, 2.0], dtype=torch.float32)
    with pytest.raises(ValueError, match="composite sort key"):
        sort_fragments(layer_pixel, depth, method="composite")
    # The fallback has no such limit.
    assert sort_fragments(layer_pixel, depth, method="two_pass").numel() == 2


def test_unknown_methods_are_rejected() -> None:
    """No silent fallback to a default (AGENTS.md review checklist)."""
    layer_pixel, depth = _random_fragments(8, 4, seed=0)
    with pytest.raises(ValueError, match="method must be"):
        sort_fragments(layer_pixel, depth, method="radix")
    with pytest.raises(ValueError, match="method must be"):
        segment_offsets(layer_pixel, 4, method="prefix")


def test_real_fragment_lists_sort_the_same_both_ways() -> None:
    """End to end on a synthetic scene, not just random integers."""
    scene = make_scene(seed=6)
    grid = layer_grid(*scene["image_hw"], 3)
    common = (scene["xyz"], scene["size"], scene["conf"], scene["K"], scene["R"], scene["t"], grid)
    a = build_sorted_fragments(*common, sort_method="composite")
    b = build_sorted_fragments(*common, sort_method="two_pass", segment_method="bincount")
    assert torch.equal(a.layer_pixel, b.layer_pixel)
    assert torch.equal(a.point_id, b.point_id)
    assert torch.equal(a.alpha, b.alpha)
    assert torch.equal(a.offsets, b.offsets)


def test_renders_are_identical_across_sort_methods() -> None:
    """Switching the fallback on must not change a single pixel."""
    scene = make_scene(seed=10)
    args = (
        scene["xyz"],
        scene["size"],
        scene["feat"],
        scene["conf"],
        scene["K"],
        scene["R"],
        scene["t"],
        scene["image_hw"],
    )
    layers_a, _ = render_pyramid_ref(*args, num_layers=3, bg=scene["bg"])
    layers_b, _ = render_pyramid_ref(
        *args, num_layers=3, bg=scene["bg"], sort_method="two_pass", segment_method="bincount"
    )
    for a, b in zip(layers_a, layers_b, strict=True):
        assert torch.equal(a, b)
