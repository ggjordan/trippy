"""Tests for trippy.edit.model: Region validation, world-space membership, undo/save-load.

Module: tests.test_edit_model
Invariants under test:
  - `box_membership` handles both axis-aligned and rotated boxes correctly
    (a rotated-box test is the acceptance criterion this task's brief adds
    beyond docs/EDITOR.md's own axis-aligned-only schema).
  - `sphere_membership` is a straightforward radius test with an inclusive
    boundary.
  - `lid_membership`'s `inside` is the literal hard "below the plane,
    inside the radius" clip, and its `weight` ramps by `band` (vertical)
    and `falloff` (horizontal) exactly per the formulas in the module
    docstring.
  - `Region.__post_init__` rejects bad `kind`/`op`/`mix`/`params`.
  - `EditDocument`'s undo cursor: add/undo/redo, and the "edit after undo
    discards the redone future" rule.
  - `EditDocument.save`/`.load` round-trips exactly (regions, order, and
    undo history), matching "closing and reopening the bundle preserves
    undo history" (docs/EDITOR.md Sec 1).
All fixtures are synthetic numpy arrays; no scene or bundle is touched.
"""

from __future__ import annotations

import numpy as np
import pytest

from trippy.edit.model import (
    EditDocument,
    Region,
    box_membership,
    lid_membership,
    pointset_membership,
    region_contains,
    region_weight,
    sphere_membership,
)

# --- box_membership, incl. rotated ---------------------------------------------------


def test_box_membership_axis_aligned() -> None:
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],  # centre, inside
            [0.9, 0.9, 0.9],  # inside, near corner
            [1.0, 0.0, 0.0],  # exactly on the boundary -> inside (inclusive)
            [1.1, 0.0, 0.0],  # just outside on x
            [0.0, 0.0, 1.1],  # just outside on z
        ]
    )
    inside = box_membership(xyz, center=(0.0, 0.0, 0.0), half_extents=(1.0, 1.0, 1.0))
    assert inside.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_box_membership_rotated_differs_from_axis_aligned() -> None:
    # A 2x1x1 (half-extents) box, and a point that is outside the y-half-extent (1.0)
    # of the UNROTATED box but inside the box once it is rotated 90deg about Z (which
    # swaps the roles of the local x/y half-extents in world space).
    center = (0.0, 0.0, 0.0)
    half_extents = (2.0, 1.0, 1.0)
    point = np.array([[0.0, 1.5, 0.0]])

    identity_quat = (1.0, 0.0, 0.0, 0.0)
    rot90z_quat = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))  # (w, x, y, z)

    unrotated = box_membership(point, center, half_extents, identity_quat)
    rotated = box_membership(point, center, half_extents, rot90z_quat)

    assert unrotated.tolist() == [0.0]  # |1.5| > half_extents.y (1.0)
    assert rotated.tolist() == [1.0]  # local x becomes 1.5, within half_extents.x (2.0)


def test_box_membership_quat_need_not_be_pre_normalised() -> None:
    # Scaling the identity quaternion must not change the result (it is normalised
    # internally).
    xyz = np.array([[0.5, 0.5, 0.5], [2.0, 0.0, 0.0]])
    a = box_membership(xyz, (0, 0, 0), (1, 1, 1), (1.0, 0.0, 0.0, 0.0))
    b = box_membership(xyz, (0, 0, 0), (1, 1, 1), (3.0, 0.0, 0.0, 0.0))
    assert a.tolist() == b.tolist() == [1.0, 0.0]


# --- sphere_membership -----------------------------------------------------------------


def test_sphere_membership_inclusive_boundary() -> None:
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0001, 0.0, 0.0], [0.0, 2.0, 0.0]])
    inside = sphere_membership(xyz, center=(0.0, 0.0, 0.0), radius=1.0)
    assert inside.tolist() == [1.0, 1.0, 0.0, 0.0]


# --- lid_membership: hard clip + falloff/band ramp --------------------------------------


# An idealised lid, chosen so `center` sits EXACTLY on the plane (dot(up, center) ==
# height with no floating-point residue) -- the Karekare pool's own fitted numbers
# (docs/EDITOR.md Sec 1 / ~/Splats/tools/SURFACE_LID.md Sec 3) are close to but not
# exactly on-plane after their own rounding, which would make an exact "weight == 0.0
# at the plane" assertion flaky. `radius`/`falloff`/`band` are the real fitted values
# (scale matters; orientation does not, for what this test checks).
_LID_UP = (0.0, 1.0, 0.0)
_LID_HEIGHT = 2.0
_LID_CENTER = (0.0, 2.0, 0.0)
_LID_RADIUS = 2.5
_LID_FALLOFF = 1.0
_LID_BAND = 0.05


def _lid_point(along_up: float, rho: float) -> np.ndarray:
    """A point at `center + along_up * up + rho * perp`, `perp` orthogonal to `up`."""
    up = np.asarray(_LID_UP, dtype=np.float64)
    up = up / np.linalg.norm(up)
    # Any vector not parallel to `up`, Gram-Schmidt'd to orthogonal.
    seed = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = seed - np.dot(seed, up) * up
    perp = perp / np.linalg.norm(perp)
    center = np.asarray(_LID_CENTER, dtype=np.float64)
    return center + along_up * up + rho * perp


def _lid(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return lid_membership(xyz, _LID_UP, _LID_HEIGHT, _LID_CENTER, _LID_RADIUS, _LID_FALLOFF, _LID_BAND)


def test_lid_hard_clip_is_below_plane_inside_radius() -> None:
    points = np.stack(
        [
            _lid_point(along_up=-1.0, rho=0.0),  # deep below, centred -> inside
            _lid_point(along_up=+1.0, rho=0.0),  # above the plane -> not inside, regardless of rho
            _lid_point(along_up=-1.0, rho=_LID_RADIUS + 10.0),  # below, but way outside radius
            _lid_point(along_up=0.0, rho=0.0),  # exactly on the plane -> inside (h <= 0 inclusive)
        ]
    )
    inside, _weight = _lid(points)
    assert inside.tolist() == [True, False, False, True]


def test_lid_falloff_band_ramp_matches_the_formula() -> None:
    # Vertical ramp: 0 at the plane, 0.5 halfway through `band`, 1 once `band` below.
    vertical_points = np.stack(
        [
            _lid_point(along_up=0.0, rho=0.0),
            _lid_point(along_up=-_LID_BAND / 2.0, rho=0.0),
            _lid_point(along_up=-_LID_BAND, rho=0.0),
            _lid_point(along_up=-10 * _LID_BAND, rho=0.0),  # saturates at 1, does not overshoot
        ]
    )
    _inside, weight = _lid(vertical_points)
    np.testing.assert_allclose(weight, [0.0, 0.5, 1.0, 1.0], atol=1e-9)

    # Horizontal ramp (deep below the plane so vertical=1): 1 inside radius, 0.5 halfway
    # through falloff beyond it, 0 once fully past radius+falloff.
    horizontal_points = np.stack(
        [
            _lid_point(along_up=-1.0, rho=0.0),
            _lid_point(along_up=-1.0, rho=_LID_RADIUS),
            _lid_point(along_up=-1.0, rho=_LID_RADIUS + _LID_FALLOFF / 2.0),
            _lid_point(along_up=-1.0, rho=_LID_RADIUS + _LID_FALLOFF),
            _lid_point(along_up=-1.0, rho=_LID_RADIUS + 10 * _LID_FALLOFF),
        ]
    )
    _inside, weight = _lid(horizontal_points)
    np.testing.assert_allclose(weight, [1.0, 1.0, 0.5, 0.0, 0.0], atol=1e-9)


def test_lid_zero_band_and_falloff_are_hard_steps() -> None:
    points = np.stack(
        [
            _lid_point(along_up=-1e-6, rho=0.0),  # just below the plane
            _lid_point(along_up=+1e-6, rho=0.0),  # just above the plane
            _lid_point(along_up=-1.0, rho=_LID_RADIUS - 1e-6),
            _lid_point(along_up=-1.0, rho=_LID_RADIUS + 1e-6),
        ]
    )
    _inside, weight = lid_membership(points, _LID_UP, _LID_HEIGHT, _LID_CENTER, _LID_RADIUS, falloff=0.0, band=0.0)
    assert weight.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_region_contains_lid_uses_hard_clip_not_falloff_weight() -> None:
    region = Region(
        id="r-lid",
        name="lid",
        kind="lid",
        params={
            "up": list(_LID_UP),
            "height": _LID_HEIGHT,
            "center": list(_LID_CENTER),
            "radius": _LID_RADIUS,
            "falloff": _LID_FALLOFF,
            "band": _LID_BAND,
        },
        op="delete",
    )
    # A point in the falloff halo (beyond the strict radius) has weight > 0 but is
    # NOT in the hard "inside the radius" clip.
    point = _lid_point(along_up=-1.0, rho=_LID_RADIUS + _LID_FALLOFF / 2.0)[None, :]
    assert region_weight(region, point)[0] > 0.0
    assert region_contains(region, point)[0] == np.False_


# --- pointset_membership ---------------------------------------------------------------


def test_pointset_membership_and_out_of_range_ids_ignored() -> None:
    weight = pointset_membership(5, [1, 3, 3, 99, -1])
    assert weight.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]


def test_pointset_membership_empty_ids() -> None:
    assert pointset_membership(4, []).tolist() == [0.0, 0.0, 0.0, 0.0]


# --- Region validation -------------------------------------------------------------------


def test_region_rejects_bad_kind_op_mix() -> None:
    with pytest.raises(ValueError, match="kind"):
        Region(id="r1", name="x", kind="bogus", params={})
    with pytest.raises(ValueError, match="op"):
        Region(id="r1", name="x", kind="sphere", params={"center": [0, 0, 0], "radius": 1.0}, op="bogus")
    with pytest.raises(ValueError, match="mix"):
        Region(id="r1", name="x", kind="sphere", params={"center": [0, 0, 0], "radius": 1.0}, mix=1.5)


def test_region_rejects_bad_params_per_kind() -> None:
    with pytest.raises(ValueError, match="half_extents"):
        Region(id="r1", name="x", kind="box", params={"center": [0, 0, 0], "half_extents": [1, -1, 1]})
    with pytest.raises(ValueError, match="radius"):
        Region(id="r1", name="x", kind="sphere", params={"center": [0, 0, 0], "radius": 0.0})
    with pytest.raises(ValueError, match="point_ids"):
        Region(id="r1", name="x", kind="pointset", params={"point_ids": "not-a-list"})


def test_region_json_round_trip() -> None:
    region = Region(id="r1", name="x", kind="sphere", params={"center": [1, 2, 3], "radius": 4.0}, mix=0.25)
    assert Region.from_json(region.to_json()).to_json() == region.to_json()


# --- EditDocument: undo cursor -----------------------------------------------------------


def _box_region(rid: str, mix: float = 1.0) -> Region:
    return Region(id=rid, name=rid, kind="box", params={"center": [0, 0, 0], "half_extents": [1, 1, 1]}, mix=mix)


def test_undo_cursor_add_undo_redo() -> None:
    edits = EditDocument.new()
    edits.add_region(_box_region("a"))
    edits.add_region(_box_region("b"))
    assert [r.id for r in edits.regions] == ["a", "b"]
    assert edits.cursor == 2 and len(edits.log) == 2

    assert edits.undo() is True
    assert [r.id for r in edits.regions] == ["a"]
    assert edits.cursor == 1

    assert edits.undo() is True
    assert edits.regions == [] and edits.cursor == 0
    assert edits.undo() is False  # nothing left to undo

    assert edits.redo() is True
    assert [r.id for r in edits.regions] == ["a"]
    assert edits.redo() is True
    assert [r.id for r in edits.regions] == ["a", "b"]
    assert edits.redo() is False  # nothing left to redo


def test_edit_after_undo_discards_the_redone_future() -> None:
    edits = EditDocument.new()
    edits.add_region(_box_region("a"))
    edits.add_region(_box_region("b"))
    edits.undo()  # back to just "a"; "b"'s add_region entry is still in the log past cursor
    edits.add_region(_box_region("c"))  # new edit must truncate "b" out of the log

    assert [r.id for r in edits.regions] == ["a", "c"]
    assert len(edits.log) == 2  # add a, add c -- "b" is gone
    assert edits.redo() is False  # nothing to redo; "b" was discarded, not merely hidden


def test_remove_and_update_region_are_undoable() -> None:
    edits = EditDocument.new()
    edits.add_region(_box_region("a", mix=1.0))
    edits.update_region("a", mix=0.5)
    assert edits.regions_by_id["a"].mix == 0.5
    edits.undo()
    assert edits.regions_by_id["a"].mix == 1.0

    edits.remove_region("a")
    assert edits.regions == []
    edits.undo()
    assert [r.id for r in edits.regions] == ["a"]


def test_reorder_changes_paint_order_without_changing_membership() -> None:
    edits = EditDocument.new()
    edits.add_region(_box_region("a"))
    edits.add_region(_box_region("b"))
    edits.reorder(["b", "a"])
    assert edits.order == ["b", "a"]
    with pytest.raises(ValueError, match="permute"):
        edits.reorder(["a"])  # not a permutation of the current ids


def test_bad_cursor_raises() -> None:
    with pytest.raises(ValueError, match="cursor"):
        EditDocument(cursor=5, log=[])


# --- save/load round trip, incl. undo history survives close+reopen ----------------------


def test_save_load_round_trip_preserves_regions_order_and_undo_history(tmp_path) -> None:
    edits = EditDocument.new(bundle_format="trippy-bundle-1")
    edits.add_region(_box_region("a"))
    edits.add_region(_box_region("b"))
    edits.undo()  # cursor now 1, "b" still in the log past cursor

    path = tmp_path / "edits.json"
    edits.save(path)
    reloaded = EditDocument.load(path)

    assert reloaded.to_json() == edits.to_json()
    assert reloaded.cursor == 1
    assert [r.id for r in reloaded.regions] == ["a"]
    assert reloaded.redo() is True  # the undone-but-not-discarded "b" survives reopening
    assert [r.id for r in reloaded.regions] == ["a", "b"]


def test_load_rejects_wrong_format(tmp_path) -> None:
    path = tmp_path / "edits.json"
    path.write_text('{"format": "not-the-right-format"}')
    with pytest.raises(ValueError, match="format"):
        EditDocument.load(path)


def test_validate_checks_bundle_format(tmp_path) -> None:
    edits = EditDocument.new(bundle_format="trippy-bundle-1")
    edits.validate(expected_bundle_format="trippy-bundle-1")  # ok
    with pytest.raises(ValueError, match="bundle_format"):
        edits.validate(expected_bundle_format="something-else")
