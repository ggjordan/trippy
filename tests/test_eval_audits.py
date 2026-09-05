"""Tests for trippy.eval.audits: Splats' shade audit + extent gate, via subprocess.

Module: tests.test_eval_audits
Invariants under test: `run_extent_gate` parses Splats' own (unmodified)
    `extent_gate.py` stdout table into the documented fields, in input
    order; `run_shade_audit` parses that script's own `--json-out` payload
    unchanged; both raise `FileNotFoundError` immediately (no subprocess
    attempted) when the Splats ml-sharp venv/scripts aren't present, and
    `audit_report` catches that (and any subprocess failure) per-audit so
    one missing/broken tool never blocks the other's numbers.
Fixtures: synthetic PLYs only (`trippy.train.export.write_gaussian_ply`,
    never a real Splats PLY); the real-scene shade-audit test uses
    `tests/conftest.py`'s `splats_scene` fixture (read-only COLMAP
    geometry -- camera poses and 3D point coordinates, not photographs --
    which AGENTS.md's privacy rule does not restrict) with a synthetic
    500-point PLY placed inside that scene's own point-cloud bounding box,
    per this task's brief; it skips cleanly (like tests/test_export_ply.py's
    equivalent extent-gate test) when ~/Splats or the ml-sharp venv aren't
    present on this machine.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from trippy.config import Settings
from trippy.eval import audits
from trippy.eval.audits import audit_report, run_extent_gate, run_shade_audit
from trippy.scene import colmap_io
from trippy.train.export import write_gaussian_ply

_SPLATS_ML_SHARP_PYTHON = Path("/Users/nzbirdranch/Splats/tools/ml-sharp/.venv/bin/python")
_SPLATS_EXTENT_GATE = Path("/Users/nzbirdranch/Splats/tools/tmp/extent-audit/extent_gate.py")
_SPLATS_SHADE_AUDIT = Path("/Users/nzbirdranch/Splats/tools/depthprior_shade_audit.py")


def _extent_gate_available() -> bool:
    return _SPLATS_ML_SHARP_PYTHON.exists() and _SPLATS_EXTENT_GATE.exists()


def _shade_audit_available() -> bool:
    return _SPLATS_ML_SHARP_PYTHON.exists() and _SPLATS_SHADE_AUDIT.exists()


def _random_pointset(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-5.0, 5.0, size=(n, 3)).astype(np.float32)
    rgb = rng.uniform(0.05, 0.95, size=(n, 3)).astype(np.float32)
    conf = rng.uniform(0.1, 0.9, size=(n,)).astype(np.float32)
    size = rng.uniform(0.01, 2.0, size=(n,)).astype(np.float32)
    return xyz, rgb, conf, size


def test_run_extent_gate_parses_real_tool_output(tmp_path: Path) -> None:
    if not _extent_gate_available():
        pytest.skip("Splats ml-sharp venv or extent_gate.py not available on this machine")

    xyz, rgb, conf, size = _random_pointset(n=200, seed=42)
    path = tmp_path / "synthetic.ply"
    write_gaussian_ply(path, xyz, rgb, conf, size)

    result = run_extent_gate([path])
    assert len(result["plys"]) == 1
    rec = result["plys"][0]
    assert rec["ply_path"] == str(path)
    assert rec["n"] == 200
    assert rec["radius_p50"] > 0
    assert rec["radius_p99"] >= rec["radius_p50"]
    assert rec["radius_max"] >= rec["radius_p999"] >= rec["radius_p99"]
    assert rec["scene_diagonal"] > 0
    assert rec["non_finite_means"] == 0
    assert rec["non_finite_scales"] == 0
    assert len(rec["median_centre"]) == 3


def test_run_extent_gate_multiple_plys_preserves_order(tmp_path: Path) -> None:
    if not _extent_gate_available():
        pytest.skip("Splats ml-sharp venv or extent_gate.py not available on this machine")

    xyz1, rgb1, conf1, size1 = _random_pointset(n=50, seed=1)
    xyz2, rgb2, conf2, size2 = _random_pointset(n=90, seed=2)
    path1, path2 = tmp_path / "a.ply", tmp_path / "b.ply"
    write_gaussian_ply(path1, xyz1, rgb1, conf1, size1)
    write_gaussian_ply(path2, xyz2, rgb2, conf2, size2)

    result = run_extent_gate([path1, path2])
    assert [r["ply_path"] for r in result["plys"]] == [str(path1), str(path2)]
    assert [r["n"] for r in result["plys"]] == [50, 90]


def test_run_extent_gate_missing_tool_raises_file_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        audits, "load_settings", lambda: Settings(splats_root=tmp_path / "no-splats", trippy_output=tmp_path / "out")
    )
    xyz, rgb, conf, size = _random_pointset(n=5)
    path = tmp_path / "a.ply"
    write_gaussian_ply(path, xyz, rgb, conf, size)
    with pytest.raises(FileNotFoundError):
        run_extent_gate([path])


def test_run_shade_audit_missing_tool_raises_file_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        audits, "load_settings", lambda: Settings(splats_root=tmp_path / "no-splats", trippy_output=tmp_path / "out")
    )
    xyz, rgb, conf, size = _random_pointset(n=5)
    path = tmp_path / "a.ply"
    write_gaussian_ply(path, xyz, rgb, conf, size)
    with pytest.raises(FileNotFoundError):
        run_shade_audit([path], tmp_path / "sparse_txt")


def test_audit_report_catches_missing_tool_and_returns_error_dicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        audits, "load_settings", lambda: Settings(splats_root=tmp_path / "no-splats", trippy_output=tmp_path / "out")
    )
    xyz, rgb, conf, size = _random_pointset(n=5)
    path = tmp_path / "a.ply"
    write_gaussian_ply(path, xyz, rgb, conf, size)

    report = audit_report([path], tmp_path / "sparse_txt")
    assert "error" in report["shade_audit"]
    assert "error" in report["extent_gate"]


def test_run_shade_audit_against_real_karekare_scene_with_synthetic_ply(splats_scene: Path) -> None:
    if not _shade_audit_available():
        pytest.skip("Splats ml-sharp venv or depthprior_shade_audit.py not available on this machine")

    scene = colmap_io.load_colmap_model(splats_scene)
    xyz_all = np.array([p.xyz for p in scene.points3D.values()], dtype=np.float64)
    bbox_min, bbox_max = xyz_all.min(axis=0), xyz_all.max(axis=0)

    rng = np.random.default_rng(0)
    n = 500
    xyz = rng.uniform(bbox_min, bbox_max, size=(n, 3)).astype(np.float32)
    rgb = rng.uniform(0.02, 0.15, size=(n, 3)).astype(np.float32)  # dark, so the audit's dark-mass branch runs too
    conf = np.full(n, 0.8, dtype=np.float32)
    size = np.full(n, 0.2, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ply_path = Path(tmp_dir) / "synthetic_shade.ply"
        write_gaussian_ply(ply_path, xyz, rgb, conf, size)

        payload = run_shade_audit([ply_path], splats_scene)

    assert payload["scene"] == str(splats_scene)
    assert len(payload["results"]) == 1
    res = payload["results"][0]
    assert res["n"] == n
    assert res["n_in_region"] >= 0
    assert res["mass_in_region"] >= 0.0
