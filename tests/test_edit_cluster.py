"""Tests for trippy.edit.cluster: click-to-cluster, the E4 milestone.

Module: tests.test_edit_cluster
Invariants under test:
  - Clicking an isolated coloured blob selects it and not a same-coloured
    blob sitting behind it along the same ray (depth-mode seed selection
    picks the nearest-camera group among the click's candidates, docs/
    EDITOR.md Sec 3 "1. Click-to-cluster (E4)").
  - The colour-distance gate separates two touching, differently-coloured
    clusters at the same depth (growth stops at the colour boundary even
    though the two clusters are geometrically contiguous).
  - `max_radius` and `max_points` are hard caps: growth never selects a
    point farther than `max_radius` from the seed centroid, and never
    selects more than `max_points` points.
  - `default_max_radius_from_bundle` returns the median nearest-CAMERA
    spacing (not point spacing) when a bundle has >= 2 views.
  - `render_click_preview` writes a real PNG file whose reported
    `n_selected` matches the region's own point count.
  - `trippy edits click` round-trips through the CLI: it appends a valid
    `pointset` region that `EditDocument.load` accepts, with the requested
    `op`/`mix`.
All fixtures are synthetic numpy arrays / hand-built bundle directories; no
real scene or photograph is ever touched (AGENTS.md Sec 6).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trippy import cli
from trippy.edit.cluster import (
    CameraView,
    click_to_cluster,
    click_to_cluster_in_bundle,
    default_max_radius_from_bundle,
    render_click_preview,
)
from trippy.edit.model import EditDocument

# A simple frontal camera: identity rotation, camera at the world origin looking
# down +Z (COLMAP convention, docs/GEOMETRY.md) -- world coordinates and camera
# coordinates coincide, which makes the synthetic fixtures easy to reason about.
_CAMERA = CameraView(
    R=np.eye(3), t=np.zeros(3), fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480, name="v"
)


def _grid_blob(center: np.ndarray, half_extent: float, n_per_axis: int, rgb: tuple[float, float, float]):
    """An `n_per_axis^2` grid of points in a plane at `center`'s own z, one colour."""
    xs = np.linspace(-half_extent, half_extent, n_per_axis) + center[0]
    ys = np.linspace(-half_extent, half_extent, n_per_axis) + center[1]
    xx, yy = np.meshgrid(xs, ys)
    xyz = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, center[2])], axis=1)
    feat = np.zeros((xyz.shape[0], 4), dtype=np.float64)
    feat[:, :3] = rgb
    return xyz, feat


def _write_bundle(bundle_dir: Path, xyz: np.ndarray, feat: np.ndarray, views: list[dict]) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    n = xyz.shape[0]
    np.savez(
        bundle_dir / "points.npz",
        xyz=xyz.astype(np.float32),
        size=np.full(n, 0.05, dtype=np.float32),
        feat=feat.astype(np.float32),
        conf=np.full(n, 0.9, dtype=np.float32),
    )
    (bundle_dir / "bundle.json").write_text(
        json.dumps({"format": "trippy-bundle-1", "points": "points.npz", "num_points": n, "views": views})
    )


def _identity_view(name: str, tx: float = 0.0) -> dict:
    return {
        "index": 0,
        "name": name,
        "width": 640,
        "height": 480,
        "fx": 500.0,
        "fy": 500.0,
        "cx": 320.0,
        "cy": 240.0,
        "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "t": [tx, 0.0, 0.0],
        "distortion": [0.0] * 8,
    }


# --- isolated blob vs. same-colour blob behind it ------------------------------------


def test_click_selects_foreground_blob_not_background_through_gap() -> None:
    rng = np.random.default_rng(0)
    n_front = 150
    n_back = 150
    front = rng.normal(scale=0.03, size=(n_front, 3)) + np.array([0.0, 0.0, 5.0])
    back = rng.normal(scale=0.03, size=(n_back, 3)) + np.array([0.0, 0.0, 20.0])
    xyz = np.vstack([front, back])
    feat = np.zeros((xyz.shape[0], 4))
    feat[:, :3] = [1.0, 0.0, 0.0]  # both blobs the SAME colour, on purpose

    region, summary = click_to_cluster(xyz, feat, _CAMERA, (320.0, 240.0), max_radius=1.0)
    ids = region.params["point_ids"]

    assert summary["n_candidates"] > 0
    assert len(ids) > 0
    assert all(i < n_front for i in ids), "background blob must not be selected through the gap"
    assert summary["seed_depth_mean"] == pytest.approx(5.0, abs=0.2)


def test_click_misses_when_nothing_projects_near_the_pixel() -> None:
    xyz, feat = _grid_blob(np.array([0.0, 0.0, 5.0]), 0.1, 5, (1.0, 0.0, 0.0))
    region, summary = click_to_cluster(xyz, feat, _CAMERA, (5.0, 5.0), radius_px=1.0, max_radius=1.0)
    assert summary["n_candidates"] == 0
    assert region.params["point_ids"] == []
    assert "warning" in summary


# --- colour gate separates touching, differently-coloured clusters ------------------


def test_colour_gate_separates_touching_clusters() -> None:
    xs = np.linspace(-1.0, 1.0, 40)
    ys = np.linspace(-1.0, 1.0, 40)
    xx, yy = np.meshgrid(xs, ys)
    xyz = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, 5.0)], axis=1)
    feat = np.zeros((xyz.shape[0], 4))
    is_red = xyz[:, 0] < 0.0
    feat[is_red, :3] = [1.0, 0.0, 0.0]
    feat[~is_red, :3] = [0.0, 0.0, 1.0]

    # click well inside the red half, in pixel space (x=-0.7 world -> u; y=0 -> v=cy).
    u = 500.0 * (-0.7) / 5.0 + 320.0
    v = 240.0
    region, _summary = click_to_cluster(xyz, feat, _CAMERA, (u, v), max_radius=5.0, colour_tol=0.15)
    ids = region.params["point_ids"]

    assert len(ids) > 0
    selected_x = xyz[ids][:, 0]
    assert selected_x.max() < 0.0, "growth must not cross into the blue half"
    assert not np.any(feat[ids, 2] > 0.5), "no blue point should be selected"


def test_loose_colour_tol_crosses_the_boundary() -> None:
    """Sanity check that the gate is actually doing the separating work above."""
    xs = np.linspace(-1.0, 1.0, 40)
    ys = np.linspace(-1.0, 1.0, 40)
    xx, yy = np.meshgrid(xs, ys)
    xyz = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, 5.0)], axis=1)
    feat = np.zeros((xyz.shape[0], 4))
    is_red = xyz[:, 0] < 0.0
    feat[is_red, :3] = [1.0, 0.0, 0.0]
    feat[~is_red, :3] = [0.0, 0.0, 1.0]

    u = 500.0 * (-0.7) / 5.0 + 320.0
    region, _summary = click_to_cluster(
        xyz, feat, _CAMERA, (u, 240.0), max_radius=5.0, colour_tol=2.0  # wide enough to admit blue too
    )
    ids = region.params["point_ids"]
    assert np.any(feat[ids, 2] > 0.5), "a loose colour_tol should cross into the blue half"


# --- max_radius / max_points caps -----------------------------------------------------


def test_max_radius_caps_selection_extent() -> None:
    xyz, feat = _grid_blob(np.array([0.0, 0.0, 5.0]), 5.0, 80, (1.0, 0.0, 0.0))
    region, summary = click_to_cluster(xyz, feat, _CAMERA, (320.0, 240.0), max_radius=1.0, colour_tol=0.15)
    ids = region.params["point_ids"]
    centroid = xyz[ids].mean(axis=0)
    dist = np.linalg.norm(xyz[ids] - centroid, axis=1)
    assert dist.max() <= 1.0 + 1e-9
    assert not summary["hit_max_points"]
    # Fewer than the whole grid (400^2 would be selected with no radius cap).
    assert len(ids) < xyz.shape[0]


def test_max_points_caps_selection_size() -> None:
    xyz, feat = _grid_blob(np.array([0.0, 0.0, 5.0]), 5.0, 80, (1.0, 0.0, 0.0))
    region, summary = click_to_cluster(
        xyz, feat, _CAMERA, (320.0, 240.0), max_radius=100.0, max_points=10, colour_tol=0.15
    )
    assert len(region.params["point_ids"]) <= 10
    assert summary["hit_max_points"]


# --- default max_radius from median camera spacing ------------------------------------


def test_default_max_radius_from_bundle_uses_camera_spacing(tmp_path: Path) -> None:
    xyz, _feat = _grid_blob(np.array([0.0, 0.0, 5.0]), 0.2, 5, (1.0, 0.0, 0.0))
    views = [_identity_view("a.jpg", tx=0.0), _identity_view("b.jpg", tx=2.0), _identity_view("c.jpg", tx=4.0)]
    bundle_doc = {"format": "trippy-bundle-1", "points": "points.npz", "views": views}
    default = default_max_radius_from_bundle(bundle_doc, xyz)
    # Camera centers at x = 0, 2, 4 -> nearest-neighbour spacing is 2.0 everywhere;
    # CLICK_DEFAULT_MAX_RADIUS_CAMERA_FACTOR is 1.0, so the default should be ~2.0,
    # not the (much smaller) point-cloud spacing.
    assert default == pytest.approx(2.0)


def test_default_max_radius_falls_back_with_one_view(tmp_path: Path) -> None:
    xyz, _feat = _grid_blob(np.array([0.0, 0.0, 5.0]), 0.2, 5, (1.0, 0.0, 0.0))
    bundle_doc = {"format": "trippy-bundle-1", "points": "points.npz", "views": [_identity_view("a.jpg")]}
    default = default_max_radius_from_bundle(bundle_doc, xyz)
    assert default > 0.0


# --- render_click_preview: a from-scratch heatmap, no photo content ------------------


def test_render_click_preview_writes_png(tmp_path: Path) -> None:
    xyz, _feat = _grid_blob(np.array([0.0, 0.0, 5.0]), 0.2, 6, (1.0, 0.0, 0.0))
    point_ids = list(range(xyz.shape[0]))
    out_path = tmp_path / "preview.png"
    info = render_click_preview(xyz, _CAMERA, point_ids, out_path, click_px=(320.0, 240.0))
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert info["n_selected"] == len(point_ids)
    assert info["path"] == str(out_path)


def test_render_click_preview_empty_selection(tmp_path: Path) -> None:
    xyz, _feat = _grid_blob(np.array([0.0, 0.0, 5.0]), 0.2, 4, (1.0, 0.0, 0.0))
    out_path = tmp_path / "empty.png"
    info = render_click_preview(xyz, _CAMERA, [], out_path)
    assert out_path.exists()
    assert info["n_selected"] == 0


# --- click_to_cluster_in_bundle ---------------------------------------------------------


def test_click_to_cluster_in_bundle_selects_blob(tmp_path: Path) -> None:
    xyz, feat = _grid_blob(np.array([0.0, 0.0, 5.0]), 0.2, 8, (1.0, 0.0, 0.0))
    bundle_dir = tmp_path / "bundle"
    views = [_identity_view("IMG_0.jpg", tx=0.0), _identity_view("IMG_1.jpg", tx=0.3)]
    _write_bundle(bundle_dir, xyz, feat, views)

    region, summary = click_to_cluster_in_bundle(bundle_dir, "IMG_0.jpg", (320.0, 240.0))
    assert region.kind == "pointset"
    assert summary["view"] == "IMG_0.jpg"
    assert summary["n_selected"] > 0


def test_click_to_cluster_in_bundle_unknown_view_raises(tmp_path: Path) -> None:
    xyz, feat = _grid_blob(np.array([0.0, 0.0, 5.0]), 0.2, 4, (1.0, 0.0, 0.0))
    bundle_dir = tmp_path / "bundle"
    _write_bundle(bundle_dir, xyz, feat, [_identity_view("IMG_0.jpg")])
    with pytest.raises(ValueError, match="no view named"):
        click_to_cluster_in_bundle(bundle_dir, "nope.jpg", (320.0, 240.0))


def test_click_to_cluster_in_bundle_writes_preview(tmp_path: Path) -> None:
    xyz, feat = _grid_blob(np.array([0.0, 0.0, 5.0]), 0.2, 6, (1.0, 0.0, 0.0))
    bundle_dir = tmp_path / "bundle"
    _write_bundle(bundle_dir, xyz, feat, [_identity_view("IMG_0.jpg"), _identity_view("IMG_1.jpg", tx=0.4)])
    preview_path = tmp_path / "preview.png"

    region, summary = click_to_cluster_in_bundle(
        bundle_dir, "IMG_0.jpg", (320.0, 240.0), preview_path=preview_path
    )
    assert preview_path.exists()
    assert summary["preview"]["n_selected"] == len(region.params["point_ids"])


# --- CLI round trip -------------------------------------------------------------------


def test_cli_edits_click_appends_valid_region(tmp_path: Path) -> None:
    xyz, feat = _grid_blob(np.array([0.0, 0.0, 5.0]), 0.2, 8, (1.0, 0.0, 0.0))
    bundle_dir = tmp_path / "bundle"
    _write_bundle(bundle_dir, xyz, feat, [_identity_view("IMG_0.jpg"), _identity_view("IMG_1.jpg", tx=0.4)])
    edits_path = tmp_path / "edits.json"
    preview_path = tmp_path / "preview.png"

    rc = cli.main(
        [
            "edits", "click",
            "--bundle", str(bundle_dir),
            "--view", "IMG_0.jpg",
            "--px", "320", "240",
            "--out", str(edits_path),
            "--op", "fade",
            "--mix", "0.5",
            "--preview", str(preview_path),
        ]
    )  # fmt: skip
    assert rc == 0

    edits = EditDocument.load(edits_path)  # raises on a malformed document
    assert len(edits.regions) == 1
    region = edits.regions[0]
    assert region.kind == "pointset"
    assert region.op == "fade"
    assert region.mix == pytest.approx(0.5)
    assert len(region.params["point_ids"]) > 0
    assert preview_path.exists()


def test_cli_edits_click_appends_to_existing_edits_json(tmp_path: Path) -> None:
    xyz, feat = _grid_blob(np.array([0.0, 0.0, 5.0]), 0.2, 6, (1.0, 0.0, 0.0))
    bundle_dir = tmp_path / "bundle"
    _write_bundle(bundle_dir, xyz, feat, [_identity_view("IMG_0.jpg")])
    edits_path = bundle_dir / "edits.json"

    assert cli.main(["edits", "add-box", "--edits", str(edits_path), "--center", "0", "0", "0",
                      "--half-extents", "1", "1", "1"]) == 0  # fmt: skip
    rc = cli.main(
        ["edits", "click", "--bundle", str(bundle_dir), "--view", "IMG_0.jpg",
         "--px", "320", "240", "--out", str(edits_path)]
    )  # fmt: skip
    assert rc == 0
    doc = json.loads(edits_path.read_text())
    assert [r["kind"] for r in doc["regions"]] == ["box", "pointset"]


def test_cli_edits_click_unknown_view_exits_2(tmp_path: Path, capsys) -> None:
    xyz, feat = _grid_blob(np.array([0.0, 0.0, 5.0]), 0.2, 4, (1.0, 0.0, 0.0))
    bundle_dir = tmp_path / "bundle"
    _write_bundle(bundle_dir, xyz, feat, [_identity_view("IMG_0.jpg")])
    rc = cli.main(
        ["edits", "click", "--bundle", str(bundle_dir), "--view", "nope.jpg",
         "--px", "320", "240", "--out", str(tmp_path / "edits.json")]
    )  # fmt: skip
    assert rc == 2
    assert "no view named" in capsys.readouterr().err


def test_cli_edits_click_missing_required_flag_is_argparse_exit_2() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["edits", "click", "--bundle", "/tmp/nope", "--view", "a.jpg"])  # no --px/--out
    assert exc_info.value.code == 2
