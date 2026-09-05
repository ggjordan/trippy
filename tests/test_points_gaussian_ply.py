"""Tests for trippy.points.gaussian_ply.GaussianPlySource.

Module: tests.test_points_gaussian_ply
Invariants under test: rgb0/conf0/size0 math match the documented formulas
    exactly (docs/GEOMETRY.md "3DGS PLY export mapping"), the min_opacity
    filter drops exactly the points it should, and size_mode="knn" agrees
    with trippy.points.knn_size.knn_mean_distance directly.
Fixture: a tiny (200-vertex) synthetic binary_little_endian 3DGS PLY
    written with plyfile (never a real Splats scene, per AGENTS.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from trippy.constants import PROVENANCE_GAUSSIAN, SH_C0
from trippy.points.gaussian_ply import GaussianPlySource
from trippy.points.knn_size import knn_mean_distance

N = 200
LOW_OPACITY_LOGIT = -5.0  # sigmoid(-5) ~= 0.0067, well below default min_opacity=0.05
HIGH_OPACITY_LOGIT = 5.0  # sigmoid(5) ~= 0.9933, well above default min_opacity=0.05
KNOWN_SCALE = 0.02  # world units; exp(log(KNOWN_SCALE)) round-trips exactly in float32


def _write_synthetic_gaussian_ply(path: Path, n: int = N, seed: int = 0) -> dict:
    """Write a tiny synthetic 3DGS PLY and return the arrays used to build it."""
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-5.0, 5.0, size=(n, 3)).astype(np.float32)
    f_dc = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    # First half low-opacity (filtered out), second half high-opacity (kept).
    half = n // 2
    opacity = np.concatenate(
        [np.full(half, LOW_OPACITY_LOGIT, dtype=np.float32), np.full(n - half, HIGH_OPACITY_LOGIT, dtype=np.float32)]
    )
    scale = np.full((n, 3), np.log(KNOWN_SCALE), dtype=np.float32)
    rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]  # fmt: skip
    verts = np.zeros(n, dtype=dtype)
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["f_dc_0"], verts["f_dc_1"], verts["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    verts["opacity"] = opacity
    verts["scale_0"], verts["scale_1"], verts["scale_2"] = scale[:, 0], scale[:, 1], scale[:, 2]
    verts["rot_0"], verts["rot_1"], verts["rot_2"], verts["rot_3"] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]

    el = PlyElement.describe(verts, "vertex")
    PlyData([el], text=False).write(str(path))
    return {"xyz": xyz, "f_dc": f_dc, "opacity": opacity, "half": half}


def test_min_opacity_filter_count(tmp_path: Path) -> None:
    path = tmp_path / "gaussians.ply"
    fixture = _write_synthetic_gaussian_ply(path)
    source = GaussianPlySource(path, min_opacity=0.05)
    ps = source.build()
    assert len(ps) == N - fixture["half"]


def test_rgb0_and_conf0_exact(tmp_path: Path) -> None:
    path = tmp_path / "gaussians.ply"
    fixture = _write_synthetic_gaussian_ply(path)
    half = fixture["half"]
    source = GaussianPlySource(path, min_opacity=0.05)
    ps = source.build()

    # min_opacity keeps the high-opacity half in original order (boolean
    # mask preserves row order), so ps rows line up with xyz[half:].
    np.testing.assert_allclose(ps.xyz, fixture["xyz"][half:], atol=1e-6)

    expected_rgb0 = np.clip(0.5 + SH_C0 * fixture["f_dc"][half:], 0.0, 1.0)
    np.testing.assert_allclose(ps.rgb0, expected_rgb0, atol=1e-6)

    expected_conf0 = 1.0 / (1.0 + np.exp(-HIGH_OPACITY_LOGIT))
    np.testing.assert_allclose(ps.conf0, expected_conf0, atol=1e-6)

    assert np.all(ps.provenance == PROVENANCE_GAUSSIAN)


def test_size_mode_scale_exact(tmp_path: Path) -> None:
    path = tmp_path / "gaussians.ply"
    _write_synthetic_gaussian_ply(path)
    source = GaussianPlySource(path, min_opacity=0.05, size_mode="scale")
    ps = source.build()
    np.testing.assert_allclose(ps.size0, KNOWN_SCALE, atol=1e-6)


def test_size_mode_knn_matches_helper(tmp_path: Path) -> None:
    path = tmp_path / "gaussians.ply"
    _write_synthetic_gaussian_ply(path)
    source = GaussianPlySource(path, min_opacity=0.05, size_mode="knn")
    ps = source.build()
    expected = knn_mean_distance(ps.xyz)
    np.testing.assert_allclose(ps.size0, expected, atol=1e-6)


def test_max_points_subsample(tmp_path: Path) -> None:
    path = tmp_path / "gaussians.ply"
    fixture = _write_synthetic_gaussian_ply(path)
    max_points = 10
    source = GaussianPlySource(path, min_opacity=0.05, max_points=max_points, seed=42)
    ps = source.build()
    assert len(ps) == max_points
    assert len(ps) < N - fixture["half"]


def test_describe_reports_config(tmp_path: Path) -> None:
    path = tmp_path / "gaussians.ply"
    _write_synthetic_gaussian_ply(path)
    source = GaussianPlySource(path, min_opacity=0.1, size_mode="knn", max_points=5, seed=7)
    d = source.describe()
    assert d["type"] == "GaussianPlySource"
    assert d["min_opacity"] == 0.1
    assert d["size_mode"] == "knn"
    assert d["max_points"] == 5
    assert d["seed"] == 7
