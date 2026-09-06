"""Tests for trippy.edit.apply: `trippy apply-edits` on a synthetic gate bundle.

Module: tests.test_edit_apply
Invariants under test:
  - `apply_edits` deletes exactly the right TRIPS points (a box-region
    delete tested against `points.npz`'s own `xyz`), writes a per-point
    `blend_weights.npy` reflecting a `fade` region's effect on the
    survivors, and writes `edits_applied.json` matching the returned
    summary.
  - The output `bundle.json` is updated (`num_points`, `blend.splat_ply`)
    but `weights.safetensors`-sized artefacts are never touched (module
    docstring's "not a full copy" invariant).
  - When `bundle.json` names a `blend.splat_ply`, a filtered copy is
    written that (a) removes exactly the Gaussians a box region's geometry
    test marks, (b) round-trips through `GaussianPlySource`, and (c)
    preserves an unrelated extra vertex property (`f_rest_0`, standing in
    for 3DGS SH-rest coefficients) byte-for-byte in the surviving rows --
    the "header-preserving structured copy" requirement.
  - `filter_gaussian_ply` reads the PLY body exactly once (`np.fromfile`
    called a single time), verified here by counting calls via monkeypatch.
Fixture: hand-built synthetic bundle directory + PLY (plyfile), no real
    scene or Splats PLY.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from trippy.edit import apply as edit_apply
from trippy.edit.model import EditDocument, Region
from trippy.points.gaussian_ply import GaussianPlySource


def _write_bundle(bundle_dir: Path, xyz: np.ndarray, splat_ply_name: str | None) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    n = xyz.shape[0]
    size = np.full(n, 0.05, dtype=np.float32)
    feat = np.zeros((n, 4), dtype=np.float32)
    feat[:, :3] = 0.5  # base colour channel, per trippy.train.params.PointParams convention
    conf = np.full(n, 0.9, dtype=np.float32)
    np.savez(bundle_dir / "points.npz", xyz=xyz.astype(np.float32), size=size, feat=feat, conf=conf)

    doc = {
        "format": "trippy-bundle-1",
        "name": "synthetic",
        "points": "points.npz",
        "weights": "weights.safetensors",
        "num_points": n,
    }
    if splat_ply_name is not None:
        doc["blend"] = {
            "gate": True,
            "gate_channel": 3,
            "gate_scale": 1.0,
            "splat_ply": splat_ply_name,
        }
    (bundle_dir / "bundle.json").write_text(json.dumps(doc, indent=2))


def _write_gaussian_ply_with_extra_field(path: Path, xyz: np.ndarray) -> np.ndarray:
    """A synthetic 3DGS-shaped PLY with an extra `f_rest_0` field carrying unique per-row
    values, so filtering can be checked for exact field preservation (not just xyz)."""
    n = xyz.shape[0]
    f_rest_0 = np.arange(n, dtype=np.float32) * 1.5 + 100.0  # unique, easy to spot post-filter
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ("f_rest_0", "f4"),
    ]  # fmt: skip
    verts = np.zeros(n, dtype=dtype)
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["opacity"] = 5.0  # sigmoid(5) ~= 0.993, well above any min_opacity filter
    verts["scale_0"] = verts["scale_1"] = verts["scale_2"] = np.log(0.02)
    verts["rot_0"] = 1.0
    verts["f_rest_0"] = f_rest_0
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(path))
    return f_rest_0


def test_apply_edits_deletes_points_writes_weights_and_filtered_ply(tmp_path: Path) -> None:
    # 6 TRIPS points; a box region (half-extent 1 at the origin) covers indices 0, 1.
    trips_xyz = np.array(
        [
            [0.0, 0.0, 0.0],   # inside the box -> deleted
            [0.5, 0.5, 0.5],   # inside the box -> deleted
            [5.0, 0.0, 0.0],   # outside the box -> kept, and inside the fade sphere below
            [8.0, 0.0, 0.0],   # outside both regions -> kept, untouched weight
            [-5.0, 0.0, 0.0],
            [-5.1, 0.0, 0.0],
        ]
    )
    bundle_dir = tmp_path / "bundle"
    splat_name = "splat.ply"
    _write_bundle(bundle_dir, trips_xyz, splat_ply_name=splat_name)

    # 4 Gaussians; the SAME box region deletes index 0 only (Gaussian cloud has its own layout).
    gaussian_xyz = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [-5.0, 0.0, 0.0], [-5.1, 0.0, 0.0]])
    f_rest_0 = _write_gaussian_ply_with_extra_field(bundle_dir / splat_name, gaussian_xyz)

    edits = EditDocument.new(bundle_format="trippy-bundle-1")
    edits.add_region(
        Region(
            id="r-box",
            name="delete box",
            kind="box",
            params={"center": [0.0, 0.0, 0.0], "half_extents": [1.0, 1.0, 1.0]},
            op="delete",
        )
    )
    edits.add_region(
        Region(
            id="r-fade",
            name="fade sphere",
            kind="sphere",
            params={"center": [5.0, 0.0, 0.0], "radius": 0.5},
            mix=0.25,
            op="fade",
        )
    )
    edits_path = bundle_dir / "edits.json"
    edits.save(edits_path)

    out_dir = tmp_path / "out"
    summary = edit_apply.apply_edits(bundle_dir, edits_path, out_dir)

    # --- TRIPS points ---
    assert summary["points"] == {"n_in": 6, "n_deleted": 2, "n_kept": 4}
    with np.load(out_dir / "points.npz") as data:
        out_xyz = data["xyz"]
    expected_xyz = trips_xyz[2:]  # indices 0, 1 deleted; order preserved
    np.testing.assert_allclose(out_xyz, expected_xyz, atol=1e-6)

    weights = np.load(out_dir / "blend_weights.npy")
    assert weights.shape == (4,)
    # Surviving order: [5.0,0,0] (in the fade sphere -> 1.0*0.25=0.25), [5.1,0,0] (untouched
    # -> 1.0), [-5.0,0,0] (untouched -> 1.0), [-5.1,0,0] (untouched -> 1.0).
    np.testing.assert_allclose(weights, [0.25, 1.0, 1.0, 1.0], atol=1e-6)

    applied_doc = json.loads((out_dir / "edits_applied.json").read_text())
    assert applied_doc == summary
    region_ids = {r["id"] for r in summary["regions"]}
    assert region_ids == {"r-box", "r-fade"}

    out_bundle_doc = json.loads((out_dir / "bundle.json").read_text())
    assert out_bundle_doc["num_points"] == 4
    assert not (out_dir / "weights.safetensors").exists()  # never copied (module docstring)

    # --- filtered splat PLY ---
    assert summary["splat_ply"] == {
        "in": str((bundle_dir / splat_name).resolve()),
        "out": str((out_dir / splat_name).resolve()),
        "n_in": 4,
        "n_deleted": 1,
        "n_kept": 3,
    }
    assert out_bundle_doc["blend"]["splat_ply"] == str((out_dir / splat_name).resolve())

    filtered = GaussianPlySource(out_dir / splat_name, min_opacity=0.0).build()
    assert len(filtered) == 3
    np.testing.assert_allclose(sorted(filtered.xyz[:, 0].tolist()), sorted(gaussian_xyz[1:, 0].tolist()), atol=1e-4)

    # The extra f_rest_0 field (not read by GaussianPlySource at all) must have survived
    # untouched in the kept rows, in original order -- the "header-preserving" requirement.
    raw = PlyData.read(str(out_dir / splat_name))["vertex"]
    np.testing.assert_allclose(np.asarray(raw["f_rest_0"]), f_rest_0[1:], atol=1e-4)


def test_apply_edits_without_splat_ply_skips_ply_filtering(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    xyz = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    _write_bundle(bundle_dir, xyz, splat_ply_name=None)

    edits = EditDocument.new(bundle_format="trippy-bundle-1")
    edits_path = bundle_dir / "edits.json"
    edits.save(edits_path)

    out_dir = tmp_path / "out"
    summary = edit_apply.apply_edits(bundle_dir, edits_path, out_dir)
    assert "splat_ply" not in summary
    assert summary["points"] == {"n_in": 2, "n_deleted": 0, "n_kept": 2}


def test_apply_edits_rejects_bundle_format_mismatch(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _write_bundle(bundle_dir, np.zeros((2, 3)), splat_ply_name=None)
    edits = EditDocument.new(bundle_format="some-other-format")
    edits_path = bundle_dir / "edits.json"
    edits.save(edits_path)

    with pytest.raises(ValueError, match="bundle_format"):
        edit_apply.apply_edits(bundle_dir, edits_path, tmp_path / "out")


def test_filter_gaussian_ply_reads_the_body_exactly_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    in_path = tmp_path / "in.ply"
    _write_gaussian_ply_with_extra_field(in_path, xyz)

    calls = []
    real_fromfile = np.fromfile

    def _counting_fromfile(*args, **kwargs):
        calls.append((args, kwargs))
        return real_fromfile(*args, **kwargs)

    monkeypatch.setattr(np, "fromfile", _counting_fromfile)
    result = edit_apply.filter_gaussian_ply(in_path, tmp_path / "out.ply", lambda a: a[:, 0] > 0.5)
    assert result == {"n_in": 3, "n_deleted": 2, "n_kept": 1}
    assert len(calls) == 1  # the whole vertex body is loaded exactly once
