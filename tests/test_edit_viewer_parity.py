"""End-to-end Python/Rust parity on a real bundle, via the viewer's headless dumps.

Module: tests.test_edit_viewer_parity
Invariants under test:
  - `trips-viewer --dump-weights` composes the SAME per-point weights (to
    1e-6) and the SAME delete mask that `trippy.edit.weights` /
    `trippy apply-edits` compose from the same `edits.json` -- the E1
    acceptance in docs/EDITOR.md Sec 6.
  - `trips-viewer --dump-shade` selects the SAME point ids that
    `trippy.edit.shade_finder.find_shade_pointset` selects at the same
    thresholds, and reports the same `dark_mass_fraction` -- the E2
    acceptance.
  - `trips-viewer --click U V --dump-click` selects the SAME point ids that
    `trippy.edit.cluster.click_to_cluster` selects at the same view, pixel
    and parameters -- the E4 acceptance.
  - `--dump-weights` composes the same weights for a `brush` region as
    `trippy.edit.model.brush_membership` does, over a REAL points.npz -- the
    viewer-side brush tool's own parity check (the committed
    `edit_golden/brush.json` pins the arithmetic; this pins it at scale).
  - A `--brush` stroke with `op = delete` changes the rendered frame and
    `--brush-undo` restores it byte for byte; `--move-region` (the translate
    gizmo's headless twin) and `--solo` each change it too -- docs/EDITOR.md
    Sec 6's screenshot proofs, run as tests.
  - The committed fixture pair (tests/test_edit_golden.py plus the Rust
    `edit::golden` tests) is the always-on version of the same check; this
    module is the one that runs against a REAL bundle directory, points.npz
    and all.

SKIPPED unless both halves exist:
  - a built viewer at `rust/target/{release,debug}/trips-viewer`;
  - a synthetic bundle at `$TRIPPY_EDIT_TEST_BUNDLE`, else
    `$TRIPPY_OUTPUT/fixtures/editor-ui-synth/bundle`.
Build both with:
    export TRIPPY_OUTPUT=/Users/nzbirdranch/trippy/output
    PYTHONPATH=. TRIPS_DEVICE=cpu python tools/make_synthetic_splat_bundle.py \\
        --out "$TRIPPY_OUTPUT/fixtures/editor-ui-synth"
    (cd rust && cargo build --release -p trips-viewer)
Nothing here reads a photograph or a scene of Jordan's: the bundle above is
generated from a seeded RNG (AGENTS.md section 6).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from trippy.constants import EDIT_BRUSH_NPZ_CELL_THRESHOLD
from trippy.edit.cluster import CameraView, click_to_cluster
from trippy.edit.model import EditDocument, Region, erase, paint_along, paint_sphere
from trippy.edit.shade_finder import find_shade_pointset
from trippy.edit.weights import compose_trips_weights

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Thresholds chosen for THIS synthetic bundle, not the shipped defaults: its
#: points all carry `conf = 0.9` (the PLY's own opacity, unchanged by a
#: two-epoch run) and their luminance clusters around 0.43, so the shipped
#: `conf < 0.5` / `lum < 0.25` would select nothing and the test would pass
#: vacuously. These select roughly a fifth of the cloud.
SHADE_LUM = 0.30
SHADE_CONF = 0.95

#: Click parameters for THIS bundle, again not the shipped defaults: its capture
#: views are 48x36 px, so the default 12 px catchment covers most of the frame
#: and every point becomes a candidate. Four pixels is a click on an object.
#: `max_radius` is passed explicitly on both sides so the comparison measures
#: the clustering rather than two implementations of a default.
CLICK_PX = (24.0, 18.0)
CLICK_RADIUS_PX = 4.0
CLICK_COLOUR_TOL = 0.15
CLICK_MAX_RADIUS = 0.8


def _viewer_binary() -> Path | None:
    for profile in ("release", "debug"):
        candidate = REPO_ROOT / "rust" / "target" / profile / "trips-viewer"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _bundle_dir() -> Path | None:
    explicit = os.environ.get("TRIPPY_EDIT_TEST_BUNDLE")
    if explicit:
        path = Path(explicit)
        return path if (path / "bundle.json").is_file() else None
    output = os.environ.get("TRIPPY_OUTPUT")
    if not output:
        return None
    path = Path(output) / "fixtures" / "editor-ui-synth" / "bundle"
    return path if (path / "bundle.json").is_file() else None


@pytest.fixture(scope="module")
def viewer() -> Path:
    binary = _viewer_binary()
    if binary is None:
        pytest.skip("trips-viewer is not built (cd rust && cargo build --release -p trips-viewer)")
    return binary


@pytest.fixture(scope="module")
def bundle() -> Path:
    path = _bundle_dir()
    if path is None:
        pytest.skip("no synthetic bundle; see this module's docstring")
    return path


def _run(viewer: Path, args: list[str]) -> None:
    result = subprocess.run(
        [str(viewer), *args], capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, f"{args}\nstdout: {result.stdout}\nstderr: {result.stderr}"


def test_dump_weights_matches_the_python_composition(
    viewer: Path, bundle: Path, tmp_path: Path
) -> None:
    out = tmp_path / "weights.json"
    _run(viewer, [str(bundle), "--dump-weights", str(out)])
    dumped = json.loads(out.read_text())
    assert dumped["format"] == "trippy-edit-weights-1"

    edits = EditDocument.load(bundle / "edits.json")
    with np.load(bundle / "points.npz") as data:
        xyz = np.asarray(data["xyz"], dtype=np.float64)
    composed = compose_trips_weights(edits, xyz)

    assert dumped["n"] == xyz.shape[0]
    assert dumped["num_regions"] == len(edits.regions)
    np.testing.assert_allclose(dumped["weight"], composed.weight, rtol=0, atol=1e-6)
    assert dumped["delete_mask"] == composed.delete_mask.tolist()
    assert dumped["num_deleted"] == int(composed.delete_mask.sum())
    # The synthetic edits.json is only a test if it does something.
    assert dumped["num_deleted"] > 0
    assert dumped["num_touched"] > 0


def test_dump_weights_on_an_absent_edits_file_is_the_identity(
    viewer: Path, bundle: Path, tmp_path: Path
) -> None:
    out = tmp_path / "identity.json"
    _run(
        viewer,
        [str(bundle), "--edits", str(tmp_path / "nope.json"), "--dump-weights", str(out)],
    )
    dumped = json.loads(out.read_text())
    assert dumped["num_regions"] == 0
    assert dumped["num_deleted"] == 0
    assert all(w == 1.0 for w in dumped["weight"])


def test_dump_shade_selects_what_the_python_finder_selects(
    viewer: Path, bundle: Path, tmp_path: Path
) -> None:
    sidecar = json.loads((bundle / "shade_views.json").read_text())
    frames = [v["name"] for v in sidecar["views"]]
    scene_root = bundle.parent / "scene"
    sparse_dir = scene_root / "sparse_txt"
    if not sparse_dir.is_dir():
        pytest.skip(f"{sparse_dir} is missing; the bundle was built somewhere else")

    out = tmp_path / "shade.json"
    _run(
        viewer,
        [
            str(bundle),
            "--dump-shade",
            str(out),
            "--shade-lum",
            str(SHADE_LUM),
            "--shade-conf",
            str(SHADE_CONF),
        ],
    )
    dumped = json.loads(out.read_text())
    assert dumped["format"] == "trippy-edit-shade-1"
    assert dumped["frames"] == frames

    with np.load(bundle / "points.npz") as data:
        xyz = np.asarray(data["xyz"], dtype=np.float64)
        feat = np.asarray(data["feat"], dtype=np.float64)
        conf = np.asarray(data["conf"], dtype=np.float64)
    region, summary = find_shade_pointset(
        sparse_dir,
        frames,
        xyz,
        np.clip(feat[:, :3], 0.0, 1.0),
        conf,
        znear_frac=sidecar["znear_frac"],
        zfar_frac=sidecar["zfar_frac"],
        lum_threshold=SHADE_LUM,
        conf_threshold=SHADE_CONF,
        mode="absolute",
    )
    assert dumped["point_ids"] == region.params["point_ids"]
    # Non-vacuous: the finder has to select some of the cloud and not all of it.
    assert 0 < len(dumped["point_ids"]) < xyz.shape[0]
    assert dumped["dark_mass_fraction"] == pytest.approx(summary["mass_fraction"], abs=1e-9)
    assert dumped["n_in_region"] == summary["dark_mass_stats"]["n_in_region"]


def test_widening_the_luminance_slider_only_adds_points(
    viewer: Path, bundle: Path, tmp_path: Path
) -> None:
    """The monotonicity a slider needs, measured through the real binary."""
    ids = {}
    for lum in (SHADE_LUM, SHADE_LUM * 1.5):
        out = tmp_path / f"shade-{lum}.json"
        _run(
            viewer,
            [
                str(bundle),
                "--dump-shade",
                str(out),
                "--shade-lum",
                str(lum),
                "--shade-conf",
                str(SHADE_CONF),
            ],
        )
        ids[lum] = set(json.loads(out.read_text())["point_ids"])
    assert ids[SHADE_LUM] < ids[SHADE_LUM * 1.5]


def _f32(value):
    """Round to float32 and back, as the Rust viewer's own parse does.

    `crate::bundle::BundleView` stores every camera field as `f32`; the Python
    side reads `bundle.json`'s decimals straight into `f64`. Rounding here makes
    both halves project with the SAME numbers, so a disagreement can only come
    from the clustering, which is the thing under test.
    """
    return np.asarray(value, dtype=np.float32).astype(np.float64)


def _click_camera(bundle: Path) -> tuple[CameraView, dict]:
    """The bundle's default view as a `CameraView`, float32-rounded."""
    doc = json.loads((bundle / "bundle.json").read_text())
    view = doc["views"][doc.get("default_view", 0)]
    camera = CameraView(
        R=_f32(view["R"]).reshape(3, 3),
        t=_f32(view["t"]).reshape(3),
        fx=float(_f32(view["fx"])),
        fy=float(_f32(view["fy"])),
        cx=float(_f32(view["cx"])),
        cy=float(_f32(view["cy"])),
        width=int(view["width"]),
        height=int(view["height"]),
        name=view["name"],
    )
    return camera, view


def _dump_click(viewer: Path, bundle: Path, out: Path, px, radius_px=CLICK_RADIUS_PX) -> dict:
    _run(
        viewer,
        [
            str(bundle),
            "--click",
            str(px[0]),
            str(px[1]),
            "--click-radius-px",
            str(radius_px),
            "--click-colour-tol",
            str(CLICK_COLOUR_TOL),
            "--click-max-radius",
            str(CLICK_MAX_RADIUS),
            "--dump-click",
            str(out),
        ],
    )
    return json.loads(out.read_text())


def test_dump_click_selects_what_the_python_clusterer_selects(
    viewer: Path, bundle: Path, tmp_path: Path
) -> None:
    dumped = _dump_click(viewer, bundle, tmp_path / "click.json", CLICK_PX)
    assert dumped["format"] == "trippy-edit-click-dump-1"

    camera, view = _click_camera(bundle)
    assert dumped["view"] == view["name"]
    with np.load(bundle / "points.npz") as data:
        xyz = np.asarray(data["xyz"], dtype=np.float64)
        feat = np.asarray(data["feat"], dtype=np.float64)
    region, summary = click_to_cluster(
        xyz,
        feat,
        camera,
        CLICK_PX,
        radius_px=CLICK_RADIUS_PX,
        colour_tol=CLICK_COLOUR_TOL,
        max_radius=CLICK_MAX_RADIUS,
    )

    assert dumped["n"] == xyz.shape[0]
    assert dumped["point_ids"] == region.params["point_ids"]
    assert dumped["n_candidates"] == summary["n_candidates"]
    assert dumped["n_seed"] == summary["n_seed"]
    assert dumped["hit_max_points"] == summary["hit_max_points"]
    assert dumped["seed_depth_mean"] == pytest.approx(summary["seed_depth_mean"], abs=1e-9)
    # Non-vacuous: a click has to select some of the cloud and not all of it.
    assert 0 < len(dumped["point_ids"]) < xyz.shape[0]


def test_dump_click_on_empty_space_selects_nothing_and_says_so(
    viewer: Path, bundle: Path, tmp_path: Path
) -> None:
    """A miss is a normal outcome of a click tool, not an error."""
    camera, _view = _click_camera(bundle)
    # Far outside a 48x36 image: nothing can project there.
    dumped = _dump_click(viewer, bundle, tmp_path / "miss.json", (-500.0, -500.0), radius_px=1.0)
    assert dumped["n_candidates"] == 0
    assert dumped["point_ids"] == []
    assert dumped["warning"]

    with np.load(bundle / "points.npz") as data:
        xyz = np.asarray(data["xyz"], dtype=np.float64)
        feat = np.asarray(data["feat"], dtype=np.float64)
    region, summary = click_to_cluster(
        xyz, feat, camera, (-500.0, -500.0), radius_px=1.0, max_radius=CLICK_MAX_RADIUS
    )
    assert region.params["point_ids"] == []
    assert summary["n_candidates"] == 0


# --- the brush region (the viewer's own paint tool, docs/EDITOR.md Sec 1) ------------


def _brush_edits(bundle: Path, out: Path) -> EditDocument:
    """An `edits.json` with one painted `brush` region, written by the Python side.

    Deliberately built with the SAME helpers the viewer's brush tool calls
    (`paint_sphere`/`paint_along`/`erase` via `trippy edits add-brush`'s own
    code path), so the file this test hands the viewer is one a stroke could
    have produced.
    """
    doc = EditDocument.new(json.loads((bundle / "bundle.json").read_text())["format"])
    with np.load(bundle / "points.npz") as data:
        xyz = np.asarray(data["xyz"], dtype=np.float64)
    centre = np.median(xyz, axis=0)
    region = Region(
        id="r-brushparity",
        name="brush-1",
        kind="brush",
        params={"origin": [0.0, 0.0, 0.0], "cell_size": 0.25, "cells": []},
        mix=0.0,
        op="fade",
        source={"tool": "brush", "radius": 0.9},
    )
    region = paint_sphere(region, centre, radius=0.9, weight=0.4)
    region = paint_along(
        region,
        [centre, centre + np.array([0.6, 0.0, 0.0])],
        radius=0.5,
        weight=1.0,
    )
    region = erase(region, centre + np.array([0.6, 0.0, 0.0]), radius=0.2)
    doc.add_region(region)
    doc.save(out)
    return doc


def test_dump_weights_matches_python_for_a_brush_region(
    viewer: Path, bundle: Path, tmp_path: Path
) -> None:
    """The viewer's `brush` membership against a REAL bundle's points.

    `tests/fixtures/synthetic/edit_golden/brush.json` pins the arithmetic on a
    60-point grid; this runs the same membership over 4 000 points of an actual
    `points.npz`, through the same `--dump-weights` path the other kinds use.
    """
    edits_path = tmp_path / "brush-edits.json"
    doc = _brush_edits(bundle, edits_path)
    cells = doc.regions[0].params["cells"]
    assert len(cells) > 0, "the strokes must paint something"

    out = tmp_path / "brush-weights.json"
    _run(viewer, [str(bundle), "--edits", str(edits_path), "--dump-weights", str(out)])
    dumped = json.loads(out.read_text())

    with np.load(bundle / "points.npz") as data:
        xyz = np.asarray(data["xyz"], dtype=np.float64)
    composed = compose_trips_weights(doc, xyz)

    assert dumped["num_regions"] == 1
    np.testing.assert_allclose(dumped["weight"], composed.weight, rtol=0, atol=1e-9)
    assert dumped["delete_mask"] == composed.delete_mask.tolist()
    # A brush region that claims nothing would pass every assertion above.
    assert dumped["num_touched"] > 0
    assert (composed.weight < 1.0).sum() == dumped["num_touched"]


def test_a_painted_stroke_changes_the_frame_and_undo_restores_it(
    viewer: Path, bundle: Path, tmp_path: Path
) -> None:
    """The brush's screenshot proof, as a test (docs/EDITOR.md Sec 6).

    A `--brush` stroke with `op = delete` must change the rendered frame, and
    `--brush-undo` must give back the unedited frame BYTE for byte -- the same
    shape as E5's own `--sam-undo` proof.
    """
    png = pytest.importorskip("PIL.Image", reason="Pillow is needed to compare frames")
    absent = tmp_path / "no-edits.json"
    size = ["--render-size", "480x360"]

    base = tmp_path / "base.png"
    _run(viewer, [str(bundle), "--edits", str(absent), *size, "--screenshot", str(base)])

    stroke = tmp_path / "stroke.png"
    brush_args = [
        "--brush", "240", "180",
        "--brush-radius", "0.5",
        "--brush-op", "delete",
    ]
    _run(
        viewer,
        [str(bundle), "--edits", str(absent), *size, *brush_args, "--screenshot", str(stroke)],
    )

    undone = tmp_path / "undone.png"
    _run(
        viewer,
        [
            str(bundle), "--edits", str(absent), *size, *brush_args, "--brush-undo",
            "--screenshot", str(undone),
        ],
    )

    def pixels(path: Path) -> np.ndarray:
        return np.asarray(png.open(path).convert("RGB"), dtype=np.int16)

    a, b, c = pixels(base), pixels(stroke), pixels(undone)
    changed = (np.abs(b - a).max(axis=2) > 0).mean()
    assert changed > 0.01, f"the stroke changed only {changed:.4%} of the frame"
    assert np.array_equal(a, c), "undo must restore the unedited frame byte for byte"


def test_moving_a_region_and_soloing_one_change_the_frame(
    viewer: Path, bundle: Path, tmp_path: Path
) -> None:
    """The gizmo's and the Named Objects panel's own screenshot proofs.

    `--move-region` is the translate gizmo's headless twin (it goes through the
    same `edit::gizmo::translated`), and `--solo` is the panel's solo button.
    Both must visibly change the frame the same edits render without them.
    """
    png = pytest.importorskip("PIL.Image", reason="Pillow is needed to compare frames")
    edits_path = tmp_path / "regions.json"
    fmt = json.loads((bundle / "bundle.json").read_text())["format"]
    doc = EditDocument.new(fmt)
    doc.add_region(
        Region(
            id="r-a", name="left blob", kind="sphere",
            params={"center": [0.0, 0.0, 5.2], "radius": 0.7}, mix=0.0, op="delete",
        )
    )
    doc.add_region(
        Region(
            id="r-b", name="right blob", kind="sphere",
            params={"center": [1.2, 0.6, 6.4], "radius": 0.7}, mix=0.0, op="delete",
        )
    )
    doc.save(edits_path)

    size = ["--render-size", "480x360"]
    base = tmp_path / "all.png"
    moved = tmp_path / "moved.png"
    solo = tmp_path / "solo.png"
    _run(viewer, [str(bundle), "--edits", str(edits_path), *size, "--screenshot", str(base)])
    _run(
        viewer,
        [
            str(bundle), "--edits", str(edits_path), *size,
            "--move-region", "left blob", "0.9", "0", "0", "--screenshot", str(moved),
        ],
    )
    _run(
        viewer,
        [
            str(bundle), "--edits", str(edits_path), *size,
            "--solo", "left blob", "--screenshot", str(solo),
        ],
    )

    def pixels(path: Path) -> np.ndarray:
        return np.asarray(png.open(path).convert("RGB"), dtype=np.int16)

    a = pixels(base)
    assert (np.abs(pixels(moved) - a).max(axis=2) > 0).mean() > 0.01, "the gizmo moved nothing"
    assert (np.abs(pixels(solo) - a).max(axis=2) > 0).mean() > 0.01, "solo isolated nothing"


def test_brush_npz_sidecar_the_viewer_writes_loads_with_plain_numpy(
    viewer: Path, tmp_path: Path
) -> None:
    """The Rust brush-sidecar writer's own file, read with nothing but `numpy.load`.

    `--brush-npz-selftest` needs no bundle: it writes the SAME synthetic
    over-threshold stroke `tests/test_edit_model.py`'s
    `test_brush_npz_sidecar_written_above_threshold` pins on the Python side
    (`EDIT_BRUSH_NPZ_CELL_THRESHOLD + 10` cells at `[[i, 0, 0], ...]`,
    `cell_size = 1.0`) and exits. This is the cross-language half of
    docs/EDITOR.md Sec 1's "brush" sidecar round trip: `edit::model`'s own
    Rust unit test already proves `brush_pyramid::npz` (the Rust reader) can
    read the file back; this proves plain `numpy.load` -- what an external
    reader, or a human with a Python shell, would actually use -- can too.
    """
    out_dir = tmp_path / "selftest"
    out_dir.mkdir()
    _run(viewer, ["--brush-npz-selftest", str(out_dir)])

    doc = json.loads((out_dir / "edits.json").read_text())
    params = doc["regions"][0]["params"]
    n = EDIT_BRUSH_NPZ_CELL_THRESHOLD + 10
    assert "cells" not in params, "cells must be externalised, not left inline"
    assert "weights" not in params
    assert params["cells_npz"] == "edits_brush_r-selftest.npz"
    assert params["n_cells"] == n

    with np.load(out_dir / params["cells_npz"]) as sidecar:
        assert sidecar["cells"].dtype == np.int32
        assert sidecar["cells"].shape == (n, 3)
        expected = np.zeros((n, 3), dtype=np.int32)
        expected[:, 0] = np.arange(n)
        np.testing.assert_array_equal(sidecar["cells"], expected)
        assert "weights" not in sidecar.files, "an all-1.0 brush writes no weights array"
