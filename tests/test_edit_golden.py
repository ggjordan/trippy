"""Tests for trippy.edit.golden: the committed Python/Rust parity fixture.

Module: tests.test_edit_golden
Invariants under test:
  - `tests/fixtures/synthetic/edit_golden/` is byte-identical to what
    `trippy.edit.golden.build_golden_fixture()` produces today. This is the
    Python half of the parity check; the Rust half
    (`trips_viewer::edit::golden`) reads the SAME committed files and
    compares its own composition to 1e-6. If `trippy.edit.weights` changes
    behaviour, this test fails BEFORE the fixture can be re-blessed and the
    Rust side silently follows.
  - The fixture's `edits.json` really is a document the ordinary loader
    accepts, with a live undo history whose cursor sits before its last
    entry (so the replay, not just the materialised `regions`, is exercised).
  - The fixture's expected weights are what `compose_trips_weights` /
    `compose_gaussian_weights` return for its own clouds, and its expected
    shade selection is what `trippy.train.prune`'s rule returns for its own
    frames -- i.e. the file is not merely self-consistent, it is derived.
  - `write_shade_views` writes a sidecar `trips_viewer::edit::shade` can read
    (format string and per-view field names pinned here, because the Rust
    reader is not importable from Python).
  - The brush fixture's region is what replaying its own `"strokes"` list
    through `paint_sphere`/`paint_along`/`erase` produces, so the Rust twin
    (`trips_viewer::edit::brush`) has to reproduce the AUTHORING helpers and
    not merely the membership lookup.
  - `names.json`'s expected names are what `trippy.edit.model.auto_region_name`
    returns for its own cases, and the cases between them cover every branch of
    the "-<digits> suffix" rule (the Rust twin numbers a viewer-made region the
    same way, or a session's Named Objects list stops reading as one list).
  - The click fixture's expected selections are what
    `trippy.edit.cluster.click_to_cluster` returns for its own scene, camera,
    pixels and parameters -- and each of its four cases exercises a different
    branch (depth-mode seeding, the colour gate, the `max_points` cap, a
    miss), so the Rust twin cannot pass by reproducing only the easy one.
All fixtures are synthetic (seeded RNG); nothing here reads a scene.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trippy.edit import golden
from trippy.edit.cluster import CameraView, click_to_cluster
from trippy.edit.model import (
    EditDocument,
    Region,
    auto_region_name,
    region_contains,
    region_weight,
)
from trippy.edit.weights import compose_gaussian_weights, compose_trips_weights
from trippy.train import prune

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "synthetic" / "edit_golden"

#: Every file the Rust golden test reads. Named here so deleting one fails
#: loudly on the Python side too, rather than only in a `cargo test` nobody ran.
EXPECTED_FILES = (
    "points.json",
    "edits.json",
    "expected_weights.json",
    "shade_views.json",
    "expected_shade.json",
    "click.json",
    "expected_click.json",
    "brush.json",
    "names.json",
)


def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


def test_every_fixture_file_is_present() -> None:
    for name in EXPECTED_FILES:
        assert (FIXTURE_DIR / name).is_file(), f"{name} missing from {FIXTURE_DIR}"


def test_the_committed_fixture_is_what_the_generator_produces_today(tmp_path: Path) -> None:
    golden.write_golden_fixture(tmp_path)
    for name in EXPECTED_FILES:
        fresh = (tmp_path / name).read_text()
        committed = (FIXTURE_DIR / name).read_text()
        assert fresh == committed, (
            f"{name} is stale. If trippy.edit.weights changed on purpose, "
            f"re-run `python -m trippy.edit.golden` AND re-run the Rust golden "
            f"test (cargo test -p trips-viewer)."
        )


def test_the_fixtures_edits_json_loads_with_a_live_undo_history() -> None:
    edits = EditDocument.from_json(_load("edits.json"))
    edits.validate("trippy-bundle-1")
    assert len(edits.regions) == 6, "every kind and op is exercised"
    assert {r.kind for r in edits.regions} == {"box", "sphere", "lid", "pointset"}
    assert {r.op for r in edits.regions} == {"blend", "fade", "delete"}
    assert any(not r.enabled for r in edits.regions), "a disabled region is present"
    # The cursor sits before the last log entry: the fixture was saved after an
    # undo, so a reader that ignored the cursor would compose different weights.
    assert edits.cursor == len(edits.log) - 1
    assert edits.redo() is True
    assert edits.undo() is True


def test_the_expected_weights_are_derived_not_transcribed() -> None:
    points = _load("points.json")
    expected = _load("expected_weights.json")
    edits = EditDocument.from_json(_load("edits.json"))

    xyz = np.asarray(points["xyz"], dtype=np.float64).reshape(-1, 3)
    trips = compose_trips_weights(edits, xyz)
    np.testing.assert_allclose(trips.weight, expected["weight"], rtol=0, atol=0)
    assert trips.delete_mask.tolist() == expected["delete_mask"]
    assert trips.delete_mask.any(), "the delete region must actually remove points"
    assert (trips.weight < 1.0).any(), "the blend/fade regions must actually lower weights"

    gaussian_xyz = np.asarray(points["gaussian_xyz"], dtype=np.float64).reshape(-1, 3)
    gaussian = compose_gaussian_weights(edits, gaussian_xyz)
    np.testing.assert_allclose(gaussian.weight, expected["gaussian_weight"], rtol=0, atol=0)
    assert gaussian.delete_mask.tolist() == expected["gaussian_delete_mask"]


def test_the_pointset_region_only_moves_the_trips_half() -> None:
    """The one asymmetry the Rust twin must reproduce (different row orders)."""
    points = _load("points.json")
    edits = EditDocument.from_json(_load("edits.json"))
    pointset = next(r for r in edits.regions if r.kind == "pointset")
    ids = np.asarray(pointset.params["point_ids"], dtype=np.int64)
    assert ids.size > 0

    xyz = np.asarray(points["xyz"], dtype=np.float64).reshape(-1, 3)
    with_ids = compose_trips_weights(edits, xyz).weight
    edits_without = EditDocument.from_json(_load("edits.json"))
    edits_without.update_region(pointset.id, enabled=False)
    without_ids = compose_trips_weights(edits_without, xyz).weight
    assert not np.allclose(with_ids, without_ids), "the pointset region changes the TRIPS weights"


def test_the_expected_shade_selection_is_the_audits_own_rule() -> None:
    points = _load("points.json")
    expected = _load("expected_shade.json")
    sidecar = _load("shade_views.json")
    thresholds = expected["thresholds"]

    views = [
        prune.ShadeView(
            name=v["name"],
            R=np.asarray(v["r"], dtype=np.float64).reshape(3, 3),
            C=np.asarray(v["c"], dtype=np.float64),
            d=v["d"],
            fx=v["fx"],
            fy=v["fy"],
            cx=v["cx"],
            cy=v["cy"],
            width=int(v["width"]),
            height=int(v["height"]),
            nobs=int(v["nobs"]),
            znear=thresholds["znear_frac"] * v["d"],
            zfar=thresholds["zfar_frac"] * v["d"],
        )
        for v in sidecar["views"]
    ]
    xyz = np.asarray(points["xyz"], dtype=np.float64).reshape(-1, 3)
    rgb = np.asarray(points["rgb"], dtype=np.float64).reshape(-1, 3)
    conf = np.asarray(points["conf"], dtype=np.float64)

    inside, _zfrac = prune.in_region(views, xyz)
    inside = inside & np.isfinite(xyz).all(axis=1)
    drop = prune.confidence_drop_mask(conf, thresholds["conf_threshold"], mode="absolute")
    selected = inside & (prune.luminance(rgb) < thresholds["lum_threshold"]) & drop
    assert np.flatnonzero(selected).tolist() == expected["point_ids"]
    assert len(expected["point_ids"]) > 0, "the finder must find something to be a test"

    stats = prune.dark_mass_stats(views, xyz, rgb, conf, thresholds["lum_threshold"])
    assert stats["n_in_region"] == expected["n_in_region"]
    assert stats["dark_mass_fraction"] == expected["dark_mass_fraction"]


def test_widening_zfar_selects_a_superset(tmp_path: Path) -> None:
    """What the viewer's znear/zfar sliders do, checked on the Python side.

    The Rust finder re-derives `znear = znear_frac * d` per slider move rather
    than reading the baked `znear`/`zfar`; this pins the monotonicity that makes
    such a slider meaningful.
    """
    points = _load("points.json")
    sidecar = _load("shade_views.json")
    xyz = np.asarray(points["xyz"], dtype=np.float64).reshape(-1, 3)

    def region(zfar_frac: float) -> np.ndarray:
        views = [
            prune.ShadeView(
                name=v["name"],
                R=np.asarray(v["r"], dtype=np.float64).reshape(3, 3),
                C=np.asarray(v["c"], dtype=np.float64),
                d=v["d"],
                fx=v["fx"],
                fy=v["fy"],
                cx=v["cx"],
                cy=v["cy"],
                width=int(v["width"]),
                height=int(v["height"]),
                nobs=int(v["nobs"]),
                znear=sidecar["znear_frac"] * v["d"],
                zfar=zfar_frac * v["d"],
            )
            for v in sidecar["views"]
        ]
        return prune.in_region(views, xyz)[0]

    narrow = region(sidecar["zfar_frac"])
    wide = region(sidecar["zfar_frac"] * 2.0)
    assert wide.sum() > narrow.sum()
    assert bool((wide | narrow == wide).all()), "widening the slab only adds points"


def test_write_shade_views_writes_the_schema_the_viewer_reads(tmp_path: Path) -> None:
    views = [
        prune.ShadeView(
            name="SYN_0000.jpg",
            R=np.eye(3),
            C=np.zeros(3),
            d=4.0,
            fx=320.0,
            fy=320.0,
            cx=160.0,
            cy=120.0,
            width=320,
            height=240,
            nobs=8,
            znear=0.2,
            zfar=2.0,
        )
    ]
    path = golden.write_shade_views(tmp_path, views)
    doc = json.loads(path.read_text())
    assert path.name == "shade_views.json"
    assert doc["format"] == "trippy-shade-views-1"
    # `trips_viewer::edit::shade::ShadeView::from_json` requires exactly these.
    entry = doc["views"][0]
    for key in ("name", "r", "c", "d", "fx", "fy", "cx", "cy", "width", "height"):
        assert key in entry, key
    assert len(entry["r"]) == 9
    assert len(entry["c"]) == 3


# --- the click-to-cluster case (E4) ---------------------------------------------------


def _click_camera() -> CameraView:
    """The fixture's camera, rebuilt as a `CameraView` from `click.json`."""
    doc = _load("click.json")["camera"]
    return CameraView(
        R=np.asarray(doc["r"], dtype=np.float64).reshape(3, 3),
        t=np.asarray(doc["t"], dtype=np.float64),
        fx=doc["fx"],
        fy=doc["fy"],
        cx=doc["cx"],
        cy=doc["cy"],
        width=int(doc["width"]),
        height=int(doc["height"]),
        name=doc["name"],
    )


def test_the_expected_click_selection_is_derived_not_transcribed() -> None:
    scene = _load("click.json")
    expected = _load("expected_click.json")
    assert scene["format"] == "trippy-edit-click-1"
    assert expected["format"] == scene["format"]

    xyz = np.asarray(scene["xyz"], dtype=np.float64).reshape(-1, 3)
    rgb = np.asarray(scene["rgb"], dtype=np.float64).reshape(-1, 3)
    camera = _click_camera()
    assert len(scene["cases"]) == len(expected["cases"])

    for case, want in zip(scene["cases"], expected["cases"], strict=True):
        assert case["name"] == want["name"]
        region, summary = click_to_cluster(
            xyz,
            rgb,
            camera,
            (case["px"][0], case["px"][1]),
            radius_px=case["radius_px"],
            colour_tol=case["colour_tol"],
            max_radius=case["max_radius"],
            max_points=case["max_points"],
            knn_k=case["knn_k"],
            depth_gap_factor=case["depth_gap_factor"],
        )
        assert region.params["point_ids"] == want["point_ids"], case["name"]
        assert summary["n_candidates"] == want["n_candidates"], case["name"]
        assert summary["n_seed"] == want["n_seed"], case["name"]
        assert summary["hit_max_points"] == want["hit_max_points"], case["name"]


def test_each_click_case_pins_a_different_branch() -> None:
    """A parity fixture is only worth its bytes if its cases disagree."""
    expected = {c["name"]: c for c in _load("expected_click.json")["cases"]}
    assert set(expected) == {"default", "capped", "loose_colour", "miss"}

    default = expected["default"]
    # The depth-mode seed really does discard candidates: the blob behind the
    # clicked one projects into the same pixels and is dropped before growth.
    assert 0 < default["n_seed"] < default["n_candidates"]
    assert default["n_selected"] > default["n_seed"], "growth adds points"

    # The colour gate is load-bearing: opening it swallows the neighbouring blob.
    loose = set(expected["loose_colour"]["point_ids"])
    assert set(default["point_ids"]) < loose
    assert len(loose) > 1.5 * len(default["point_ids"])

    # The cap stops the fill part-way, which is the order-dependent branch.
    capped = expected["capped"]
    assert capped["hit_max_points"] is True
    assert capped["n_selected"] == 94
    assert set(capped["point_ids"]) < set(default["point_ids"])

    # A miss is an empty selection with a warning, never an exception.
    miss = expected["miss"]
    assert miss["point_ids"] == []
    assert miss["n_candidates"] == 0
    assert miss["warning"]


# --- the brush region (Python-side only; recorded for a future Rust twin) ------------


def test_the_brush_fixture_is_derived_not_transcribed() -> None:
    doc = _load("brush.json")
    assert doc["format"] == "trippy-edit-brush-1"
    region = Region.from_json(doc["region"])
    assert region.kind == "brush"

    xyz = np.asarray(doc["xyz"], dtype=np.float64).reshape(-1, 3)
    weight = region_weight(region, xyz)
    contains = region_contains(region, xyz)
    np.testing.assert_allclose(weight, doc["expected_weight"], rtol=0, atol=0)
    assert contains.tolist() == doc["expected_contains"]


def test_the_brush_fixture_exercises_paint_and_erase_and_a_graded_weight() -> None:
    doc = _load("brush.json")
    weight = doc["expected_weight"]
    assert any(w == 0.0 for w in weight), "some query points must land outside the brush"
    assert any(w == 1.0 for w in weight), "some query points must land in a full-weight cell"
    assert any(0.0 < w < 1.0 for w in weight), "the erase must leave some graded (< 1.0) cells behind"
    params = doc["region"]["params"]
    assert params["weights"] is not None and len(params["weights"]) == len(params["cells"])


def test_the_brush_region_is_what_replaying_its_own_strokes_produces() -> None:
    """The fixture's `"strokes"` really do build its `"region"`, so the Rust twin can replay them."""
    doc = _load("brush.json")
    initial = Region.from_json(doc["initial"])
    assert initial.params["cells"] == [], "the strokes start from an empty brush"

    replayed = golden.replay_brush_strokes(initial, doc["strokes"])
    assert replayed.to_json() == doc["region"], "the committed region is the strokes' own result"
    # Order included: both sides keep cells in first-painted order, which is what
    # lets the Rust comparison be exact rather than set-wise.
    assert replayed.params["cells"] == doc["region"]["params"]["cells"]

    ops = [s["op"] for s in doc["strokes"]]
    assert ops == ["paint_sphere", "paint_along", "erase"], "all three helpers are exercised"


def test_the_brush_strokes_paint_cells_a_centre_test_would_miss() -> None:
    """`_sphere_touched_cells` is a box-sphere OVERLAP test, not a cell-centre test.

    Pinned here because it is the one place a plausible Rust port would differ
    silently: voxelising by "is the cell's centre inside the sphere" gives a
    strictly smaller cell set and every membership lookup would still work.
    """
    doc = _load("brush.json")
    params = doc["region"]["params"]
    origin = np.asarray(params["origin"], dtype=np.float64)
    cell = float(params["cell_size"])
    cells = np.asarray(params["cells"], dtype=np.int64)
    centres = origin + (cells + 0.5) * cell

    first = doc["strokes"][0]
    d = np.linalg.norm(centres - np.asarray(first["center"], dtype=np.float64), axis=1)
    assert (d > first["radius"]).any(), (
        "some painted cell's CENTRE lies outside the sphere that painted it"
    )


# --- the auto-name rule (the Named Objects panel, both sides) -------------------------


def test_the_name_fixture_is_derived_not_transcribed() -> None:
    doc = _load("names.json")
    assert doc["format"] == "trippy-edit-names-1"
    assert doc["cases"], "an empty case list would pass vacuously"
    for case in doc["cases"]:
        got = auto_region_name(case["existing_names"], case["tool"], case["detail"])
        assert got == case["expected"], case["name"]


def test_each_name_case_pins_a_different_branch() -> None:
    """A parity fixture is only worth its bytes if its cases disagree."""
    cases = {c["name"]: c for c in _load("names.json")["cases"]}
    assert cases["first"]["expected"] == "click-1", "an empty document starts at 1"
    # The counter is shared across tools, which is the whole point of the rule.
    assert cases["across_tools"]["expected"].endswith("-2")
    assert cases["with_detail"]["expected"] == "sam-box-IMG_3703-3"
    # Names with no counter, an interior counter, or digits with no hyphen are ignored.
    for name in ("no_counters", "interior_digits", "no_hyphen"):
        assert cases[name]["expected"].endswith("-1"), name
    assert cases["highest_not_last"]["expected"].endswith("-11"), "the HIGHEST suffix wins"
    assert cases["bare_counter"]["expected"].endswith("-6")


def test_the_rule_self_heals_after_a_removal() -> None:
    """Stateless numbering: dropping the highest-numbered region lowers the next name."""
    names = ["click-1", "brush-2", "sam-box-3"]
    assert auto_region_name(names, "brush") == "brush-4"
    names.remove("sam-box-3")
    assert auto_region_name(names, "brush") == "brush-3"
