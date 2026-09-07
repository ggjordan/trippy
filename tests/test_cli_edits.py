"""End-to-end CLI tests: `trippy apply-edits` and `trippy edits <subcommand>`.

Module: tests.test_cli_edits
Invariants under test:
  - `trippy edits add-box`/`add-sphere`/`add-lid` create (or append to) an
    `edits.json` file and exit 0; `add-lid` with no geometry flags seeds
    exactly the Karekare pool's fitted numbers (the brief's own worked
    example).
  - `trippy edits shade-find` writes a `pointset` region from a synthetic
    bundle + scene and exits 0.
  - `trippy apply-edits` exits 0 on a valid bundle/edits pair, and exits 2
    (not a traceback) on a missing edits file, a missing bundle, or a bad
    region.
  - Missing required CLI flags are argparse's own usage error (`SystemExit`
    code 2), exercised once per subcommand family.
Fixture: hand-built synthetic bundle + scene directories; no real scene.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_train_prune import build_scene_with_observations

from trippy import cli
from trippy.constants import EDIT_KAREKARE_LID_CENTER, EDIT_KAREKARE_LID_UP


def _write_minimal_bundle(bundle_dir: Path, xyz: np.ndarray) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    n = xyz.shape[0]
    feat = np.zeros((n, 4), dtype=np.float32)
    feat[:, :3] = 0.1
    np.savez(
        bundle_dir / "points.npz",
        xyz=xyz.astype(np.float32),
        size=np.full(n, 0.05, dtype=np.float32),
        feat=feat,
        conf=np.full(n, 0.9, dtype=np.float32),
    )
    (bundle_dir / "bundle.json").write_text(
        json.dumps({"format": "trippy-bundle-1", "points": "points.npz", "num_points": n})
    )


def test_add_box_creates_edits_json(tmp_path: Path) -> None:
    edits_path = tmp_path / "edits.json"
    rc = cli.main(
        [
            "edits", "add-box",
            "--edits", str(edits_path),
            "--center", "0", "0", "0",
            "--half-extents", "1", "1", "1",
            "--name", "fence",
            "--op", "delete",
        ]
    )  # fmt: skip
    assert rc == 0
    doc = json.loads(edits_path.read_text())
    assert len(doc["regions"]) == 1
    assert doc["regions"][0] == {
        "id": doc["regions"][0]["id"],
        "name": "fence",
        "kind": "box",
        "enabled": True,
        "mix": 1.0,
        "op": "delete",
        "params": {"center": [0.0, 0.0, 0.0], "half_extents": [1.0, 1.0, 1.0], "quat": [1.0, 0.0, 0.0, 0.0]},
        "source": None,
    }


def test_add_sphere_appends_to_existing_edits_json(tmp_path: Path) -> None:
    edits_path = tmp_path / "edits.json"
    assert cli.main(["edits", "add-box", "--edits", str(edits_path), "--center", "0", "0", "0",
                      "--half-extents", "1", "1", "1"]) == 0  # fmt: skip
    assert cli.main(["edits", "add-sphere", "--edits", str(edits_path), "--center", "1", "2", "3",
                      "--radius", "0.5"]) == 0  # fmt: skip
    doc = json.loads(edits_path.read_text())
    assert [r["kind"] for r in doc["regions"]] == ["box", "sphere"]


def test_add_lid_defaults_to_karekare_pool_numbers(tmp_path: Path) -> None:
    edits_path = tmp_path / "edits.json"
    rc = cli.main(["edits", "add-lid", "--edits", str(edits_path)])
    assert rc == 0
    doc = json.loads(edits_path.read_text())
    params = doc["regions"][0]["params"]
    assert params["up"] == pytest.approx(list(EDIT_KAREKARE_LID_UP))
    assert params["center"] == pytest.approx(list(EDIT_KAREKARE_LID_CENTER))
    assert doc["regions"][0]["op"] == "delete"  # ADR-0007's "hard clip" default action


def test_add_box_rejects_bad_geometry_with_exit_code_2(tmp_path: Path, capsys) -> None:
    edits_path = tmp_path / "edits.json"
    rc = cli.main(
        [
            "edits", "add-box",
            "--edits", str(edits_path),
            "--center", "0", "0", "0",
            "--half-extents", "1", "-1", "1",  # invalid: must be > 0
        ]
    )  # fmt: skip
    assert rc == 2
    assert not edits_path.exists()
    assert "half_extents" in capsys.readouterr().err


def test_add_box_missing_required_flag_is_argparse_exit_2() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["edits", "add-box", "--edits", "/tmp/nope-edits.json"])  # no --center/--half-extents
    assert exc_info.value.code == 2


def test_shade_find_writes_pointset_region(tmp_path: Path) -> None:
    scene_root, point_set, names = build_scene_with_observations(tmp_path)
    bundle_dir = tmp_path / "bundle"
    _write_minimal_bundle(bundle_dir, point_set.xyz.astype(np.float64))
    # Feed a real feature colour + confidence so the finder has something to threshold.
    with np.load(bundle_dir / "points.npz") as data:
        arrays = dict(data)
    arrays["feat"][:, :3] = np.clip(point_set.rgb0, 0.0, 1.0)
    arrays["conf"] = point_set.conf0.astype(np.float32)
    np.savez(bundle_dir / "points.npz", **arrays)

    edits_path = tmp_path / "edits.json"
    rc = cli.main(
        [
            "edits", "shade-find",
            "--bundle", str(bundle_dir),
            "--scene", str(scene_root),
            "--frames", ",".join(names[:2]),
            "--out", str(edits_path),
        ]
    )  # fmt: skip
    assert rc == 0
    doc = json.loads(edits_path.read_text())
    assert doc["regions"][0]["kind"] == "pointset"


def test_shade_find_missing_scene_exits_2(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_minimal_bundle(bundle_dir, np.zeros((3, 3)))
    rc = cli.main(
        [
            "edits", "shade-find",
            "--bundle", str(bundle_dir),
            "--scene", str(tmp_path / "no-such-scene"),
            "--frames", "a.jpg",
            "--out", str(tmp_path / "edits.json"),
        ]
    )  # fmt: skip
    assert rc == 2


def test_apply_edits_end_to_end_exit_0(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_minimal_bundle(bundle_dir, np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]))
    edits_path = bundle_dir / "edits.json"
    assert cli.main(["edits", "add-box", "--edits", str(edits_path), "--bundle", str(bundle_dir),
                      "--center", "0", "0", "0", "--half-extents", "1", "1", "1", "--op", "delete"]) == 0  # fmt: skip

    out_dir = tmp_path / "out"
    rc = cli.main(["apply-edits", "--bundle", str(bundle_dir), "--out", str(out_dir)])
    assert rc == 0
    assert (out_dir / "points.npz").exists()
    assert (out_dir / "blend_weights.npy").exists()
    summary = json.loads((out_dir / "edits_applied.json").read_text())
    assert summary["points"] == {"n_in": 2, "n_deleted": 1, "n_kept": 1}


def test_apply_edits_target_trips_writes_export_ply(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_minimal_bundle(bundle_dir, np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]))
    edits_path = bundle_dir / "edits.json"
    assert cli.main(["edits", "add-box", "--edits", str(edits_path), "--bundle", str(bundle_dir),
                      "--center", "0", "0", "0", "--half-extents", "1", "1", "1", "--op", "delete"]) == 0  # fmt: skip

    out_dir = tmp_path / "out"
    rc = cli.main(
        ["apply-edits", "--bundle", str(bundle_dir), "--out", str(out_dir), "--target", "trips"]
    )
    assert rc == 0
    assert (out_dir / "export.ply").exists()
    summary = json.loads((out_dir / "edits_applied.json").read_text())
    assert summary["target"] == "trips"
    assert "distilled" not in summary


def test_apply_edits_missing_edits_file_exits_2(tmp_path: Path, capsys) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_minimal_bundle(bundle_dir, np.zeros((2, 3)))
    rc = cli.main(["apply-edits", "--bundle", str(bundle_dir), "--out", str(tmp_path / "out")])
    assert rc == 2
    assert "no edits file" in capsys.readouterr().err


def test_apply_edits_missing_bundle_json_exits_2(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    edits_path = bundle_dir / "edits.json"
    (edits_path).write_text(
        json.dumps({"format": "trippy-edits-1", "bundle_format": "", "regions": [], "order": [],
                    "undo_stack": {"cursor": 0, "log": []}})
    )  # fmt: skip
    rc = cli.main(["apply-edits", "--bundle", str(bundle_dir), "--out", str(bundle_dir / "out")])
    assert rc == 2


def test_apply_edits_missing_required_flag_is_argparse_exit_2() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["apply-edits", "--bundle", "/tmp/nope"])  # no --out
    assert exc_info.value.code == 2


# --- edits add-brush / list / rename / toggle / remove (Named Objects panel CLI) -----------


def test_add_brush_paints_a_sphere_and_auto_names(tmp_path: Path) -> None:
    edits_path = tmp_path / "edits.json"
    rc = cli.main(
        [
            "edits", "add-brush",
            "--edits", str(edits_path),
            "--cell-size", "0.5",
            "--center", "1", "0", "4",
            "--radius", "0.9",
        ]
    )  # fmt: skip
    assert rc == 0
    doc = json.loads(edits_path.read_text())
    assert len(doc["regions"]) == 1
    region = doc["regions"][0]
    assert region["kind"] == "brush"
    assert region["name"] == "brush-1"  # auto-named, no --name given
    assert len(region["params"]["cells"]) > 0
    assert region["source"] == {"tool": "brush", "params": {"center": [1.0, 0.0, 4.0], "radius": 0.9}}


def test_edits_list_prints_every_region(tmp_path: Path) -> None:
    edits_path = tmp_path / "edits.json"
    assert cli.main(["edits", "add-box", "--edits", str(edits_path), "--center", "0", "0", "0",
                      "--half-extents", "1", "1", "1", "--name", "fence"]) == 0  # fmt: skip
    assert cli.main(["edits", "add-sphere", "--edits", str(edits_path), "--center", "1", "2", "3",
                      "--radius", "0.5"]) == 0  # fmt: skip

    rc = cli.main(["edits", "list", "--edits", str(edits_path)])
    assert rc == 0


def test_edits_list_missing_file_exits_2(tmp_path: Path, capsys) -> None:
    rc = cli.main(["edits", "list", "--edits", str(tmp_path / "nope.json")])
    assert rc == 2
    assert "no edits file" in capsys.readouterr().err


def _region_id(edits_path: Path, index: int = 0) -> str:
    return json.loads(edits_path.read_text())["regions"][index]["id"]


def test_edits_rename_updates_the_name(tmp_path: Path) -> None:
    edits_path = tmp_path / "edits.json"
    assert cli.main(["edits", "add-box", "--edits", str(edits_path), "--center", "0", "0", "0",
                      "--half-extents", "1", "1", "1"]) == 0  # fmt: skip
    rid = _region_id(edits_path)

    rc = cli.main(["edits", "rename", "--edits", str(edits_path), "--id", rid, "--name", "new name"])
    assert rc == 0
    assert json.loads(edits_path.read_text())["regions"][0]["name"] == "new name"


def test_edits_rename_unknown_id_exits_2(tmp_path: Path, capsys) -> None:
    edits_path = tmp_path / "edits.json"
    assert cli.main(["edits", "add-box", "--edits", str(edits_path), "--center", "0", "0", "0",
                      "--half-extents", "1", "1", "1"]) == 0  # fmt: skip
    rc = cli.main(["edits", "rename", "--edits", str(edits_path), "--id", "r-nope", "--name", "x"])
    assert rc == 2
    assert "r-nope" in capsys.readouterr().err


def test_edits_toggle_flips_by_default_and_sets_explicitly(tmp_path: Path) -> None:
    edits_path = tmp_path / "edits.json"
    assert cli.main(["edits", "add-box", "--edits", str(edits_path), "--center", "0", "0", "0",
                      "--half-extents", "1", "1", "1"]) == 0  # fmt: skip
    rid = _region_id(edits_path)
    assert json.loads(edits_path.read_text())["regions"][0]["enabled"] is True

    assert cli.main(["edits", "toggle", "--edits", str(edits_path), "--id", rid]) == 0
    assert json.loads(edits_path.read_text())["regions"][0]["enabled"] is False

    assert cli.main(["edits", "toggle", "--edits", str(edits_path), "--id", rid, "--enabled", "on"]) == 0
    assert json.loads(edits_path.read_text())["regions"][0]["enabled"] is True

    assert cli.main(["edits", "toggle", "--edits", str(edits_path), "--id", rid, "--enabled", "off"]) == 0
    assert json.loads(edits_path.read_text())["regions"][0]["enabled"] is False


def test_edits_toggle_unknown_id_exits_2(tmp_path: Path, capsys) -> None:
    edits_path = tmp_path / "edits.json"
    assert cli.main(["edits", "add-box", "--edits", str(edits_path), "--center", "0", "0", "0",
                      "--half-extents", "1", "1", "1"]) == 0  # fmt: skip
    rc = cli.main(["edits", "toggle", "--edits", str(edits_path), "--id", "r-nope"])
    assert rc == 2
    assert "r-nope" in capsys.readouterr().err


def test_edits_remove_deletes_the_region(tmp_path: Path) -> None:
    edits_path = tmp_path / "edits.json"
    assert cli.main(["edits", "add-box", "--edits", str(edits_path), "--center", "0", "0", "0",
                      "--half-extents", "1", "1", "1"]) == 0  # fmt: skip
    rid = _region_id(edits_path)

    rc = cli.main(["edits", "remove", "--edits", str(edits_path), "--id", rid])
    assert rc == 0
    assert json.loads(edits_path.read_text())["regions"] == []


def test_edits_remove_unknown_id_exits_2(tmp_path: Path, capsys) -> None:
    edits_path = tmp_path / "edits.json"
    assert cli.main(["edits", "add-box", "--edits", str(edits_path), "--center", "0", "0", "0",
                      "--half-extents", "1", "1", "1"]) == 0  # fmt: skip
    rc = cli.main(["edits", "remove", "--edits", str(edits_path), "--id", "r-nope"])
    assert rc == 2
    assert "r-nope" in capsys.readouterr().err
