"""Tests for trippy.edit.weights: composition order, blend/fade/delete semantics.

Module: tests.test_edit_weights
Invariants under test:
  - The gate default (1.0 = pure TRIPS) is the starting weight when no
    region touches a point.
  - `op="blend"` sets the weight to `mix` where a hard region applies
    (last-write-wins across overlapping enabled regions, in `edits.order`).
  - `op="fade"` multiplies the running weight by `mix` where it applies
    (stacking fades multiply further).
  - `op="delete"` zeroes the weight and is permanent: a later `blend`/`fade`
    region covering the same points must not un-delete them.
  - Disabled regions are skipped entirely.
  - `compose_gaussian_weights` ignores `pointset` regions (different row
    order than a Gaussian PLY -- see module docstring), but still applies
    box/sphere/lid regions.
All fixtures are synthetic numpy arrays.
"""

from __future__ import annotations

import numpy as np

from trippy.edit.model import EditDocument, Region
from trippy.edit.weights import (
    compose_gaussian_opacity_scale,
    compose_gaussian_weights,
    compose_trips_weights,
)


def _box(rid: str, mix: float, op: str = "blend", enabled: bool = True, half=1.0) -> Region:
    return Region(
        id=rid,
        name=rid,
        kind="box",
        params={"center": [0.0, 0.0, 0.0], "half_extents": [half, half, half]},
        mix=mix,
        op=op,
        enabled=enabled,
    )


def test_default_weight_is_the_gate_default_when_untouched() -> None:
    edits = EditDocument.new()
    xyz = np.array([[10.0, 10.0, 10.0]])  # outside every region below (there are none)
    result = compose_trips_weights(edits, xyz, default=1.0)
    assert result.weight.tolist() == [1.0]
    assert result.delete_mask.tolist() == [False]


def test_blend_sets_mix_inside_hard_region_last_write_wins() -> None:
    edits = EditDocument.new()
    edits.add_region(_box("a", mix=0.2, half=2.0))  # covers everything below
    edits.add_region(_box("b", mix=0.8, half=1.0))  # smaller, overlapping subset
    xyz = np.array(
        [
            [1.5, 0.0, 0.0],  # only inside "a" (half_extents 2.0, outside 1.0)
            [0.5, 0.0, 0.0],  # inside both -> "b" (later in order) wins outright
            [5.0, 0.0, 0.0],  # inside neither -> gate default
        ]
    )
    result = compose_trips_weights(edits, xyz, default=1.0)
    np.testing.assert_allclose(result.weight, [0.2, 0.8, 1.0])
    assert not result.delete_mask.any()


def test_fade_multiplies_and_stacks() -> None:
    edits = EditDocument.new()
    edits.add_region(_box("a", mix=0.5, op="fade"))
    edits.add_region(_box("b", mix=0.5, op="fade"))
    xyz = np.array([[0.0, 0.0, 0.0]])
    result = compose_trips_weights(edits, xyz, default=1.0)
    # 1.0 -> *0.5 (fade a) -> *0.5 (fade b) = 0.25
    np.testing.assert_allclose(result.weight, [0.25])


def test_delete_is_permanent_against_a_later_blend() -> None:
    edits = EditDocument.new()
    edits.add_region(_box("a", mix=1.0, op="delete"))
    edits.add_region(_box("b", mix=1.0, op="blend"))  # overlaps "a", drawn later
    xyz = np.array([[0.0, 0.0, 0.0]])
    result = compose_trips_weights(edits, xyz, default=1.0)
    assert result.delete_mask.tolist() == [True]
    assert result.weight.tolist() == [0.0]  # "b"'s blend must not un-delete it


def test_disabled_region_is_skipped() -> None:
    edits = EditDocument.new()
    edits.add_region(_box("a", mix=0.0, enabled=False))
    xyz = np.array([[0.0, 0.0, 0.0]])
    result = compose_trips_weights(edits, xyz, default=1.0)
    assert result.weight.tolist() == [1.0]
    assert result.delete_mask.tolist() == [False]


def test_compose_order_matters() -> None:
    # Same two regions, opposite order -> opposite winner where they overlap.
    xyz = np.array([[0.0, 0.0, 0.0]])

    forward = EditDocument.new()
    forward.add_region(_box("a", mix=0.1))
    forward.add_region(_box("b", mix=0.9))
    assert compose_trips_weights(forward, xyz).weight.tolist() == [0.9]

    backward = EditDocument.new()
    backward.add_region(_box("b", mix=0.9))
    backward.add_region(_box("a", mix=0.1))
    assert compose_trips_weights(backward, xyz).weight.tolist() == [0.1]


def test_compose_gaussian_weights_ignores_pointset_regions() -> None:
    edits = EditDocument.new()
    edits.add_region(Region(id="p", name="p", kind="pointset", params={"point_ids": [0]}, op="delete"))
    edits.add_region(_box("b", mix=0.3))  # still applies to Gaussians
    xyz = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])

    gaussian = compose_gaussian_weights(edits, xyz, default=1.0)
    assert gaussian.delete_mask.tolist() == [False, False]  # pointset ignored entirely
    np.testing.assert_allclose(gaussian.weight, [0.3, 1.0])  # box region still applies

    trips = compose_trips_weights(edits, xyz, default=1.0)
    assert trips.delete_mask.tolist() == [True, False]  # pointset applies to TRIPS points


# --- compose_gaussian_opacity_scale: the distilled-ply reapply path (docs/EDITOR.md Sec 5) ---


def test_opacity_scale_deletes_and_fades_but_ignores_blend() -> None:
    edits = EditDocument.new()
    edits.add_region(_box("del", mix=1.0, op="delete", half=1.0))
    edits.add_region(_box("fade", mix=0.25, op="fade", half=1.0))
    edits.add_region(_box("blend", mix=0.0, op="blend", half=1.0))  # ignored: no meaning for a plain PLY
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],  # inside every region: delete wins (it fires first, hard removal)
            [10.0, 0.0, 0.0],  # outside every region: untouched, opacity_scale == 1.0
        ]
    )
    result = compose_gaussian_opacity_scale(edits, xyz)
    assert result.delete_mask.tolist() == [True, False]
    np.testing.assert_allclose(result.weight, [0.0, 1.0])  # delete zeroes; untouched point stays 1.0


def test_opacity_scale_fade_only_region_ramps_down() -> None:
    edits = EditDocument.new()
    edits.add_region(_box("fade", mix=0.25, op="fade", half=1.0))
    xyz = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    result = compose_gaussian_opacity_scale(edits, xyz)
    assert not result.delete_mask.any()
    # Inside the fade region: 1.0 * (1 - 1.0*(1-0.25)) = 0.25. Outside: untouched at 1.0.
    np.testing.assert_allclose(result.weight, [0.25, 1.0])


def test_compose_point_weights_ops_filter_skips_regions_outside_the_set() -> None:
    from trippy.edit.weights import compose_point_weights

    edits = EditDocument.new()
    edits.add_region(_box("blend", mix=0.0, op="blend", half=1.0))
    xyz = np.array([[0.0, 0.0, 0.0]])
    result = compose_point_weights(edits, xyz, default=1.0, ops=("delete", "fade"))
    assert result.weight.tolist() == [1.0]  # the blend region is invisible to this call
