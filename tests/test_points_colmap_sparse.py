"""Tests for trippy.points.colmap_sparse.ColmapSparseSource.

Module: tests.test_points_colmap_sparse
Invariants under test: reads a synthetic points3D.txt (COLMAP text sparse
    model), maps rgb uint8 -> [0,1] float32 exactly, sets a fixed conf0,
    stamps PROVENANCE_COLMAP, and delegates size0 to knn_mean_distance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trippy.constants import COLMAP_DEFAULT_CONF0, PROVENANCE_COLMAP
from trippy.points.colmap_sparse import ColmapSparseSource
from trippy.points.knn_size import knn_mean_distance

N = 30


def _write_points3d_txt(path: Path, n: int = N, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-10.0, 10.0, size=(n, 3))
    rgb = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    lines = ["# 3D point list with one line of data per point:\n"]
    for i in range(n):
        x, y, z = xyz[i]
        r, g, b = rgb[i]
        lines.append(f"{i} {x} {y} {z} {r} {g} {b} 0.5\n")
    path.write_text("".join(lines))
    return xyz, rgb


def test_colmap_sparse_source_fields(tmp_path: Path) -> None:
    xyz, rgb = _write_points3d_txt(tmp_path / "points3D.txt")
    source = ColmapSparseSource(tmp_path)
    ps = source.build()

    assert len(ps) == N
    np.testing.assert_allclose(np.sort(ps.xyz, axis=0), np.sort(xyz.astype(np.float32), axis=0), atol=1e-3)
    np.testing.assert_allclose(ps.conf0, COLMAP_DEFAULT_CONF0)
    assert np.all(ps.provenance == PROVENANCE_COLMAP)

    expected_rgb0 = rgb.astype(np.float32) / 255.0
    # rgb0 rows are in points3D.txt insertion order (dict preserves it);
    # compare directly since we wrote ids 0..N-1 in order.
    np.testing.assert_allclose(ps.rgb0, expected_rgb0, atol=1e-6)

    expected_size0 = knn_mean_distance(ps.xyz)
    np.testing.assert_allclose(ps.size0, expected_size0, atol=1e-5)


def test_colmap_sparse_source_empty(tmp_path: Path) -> None:
    (tmp_path / "points3D.txt").write_text("# empty\n")
    source = ColmapSparseSource(tmp_path)
    ps = source.build()
    assert len(ps) == 0


def test_colmap_sparse_source_describe(tmp_path: Path) -> None:
    source = ColmapSparseSource(tmp_path)
    d = source.describe()
    assert d["type"] == "ColmapSparseSource"
    assert d["sparse_txt_dir"] == str(tmp_path)
