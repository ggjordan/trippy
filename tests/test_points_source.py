"""Tests for trippy.points.source.PointSet: summary(), save_npz/load_npz.

Module: tests.test_points_source
Invariants under test: summary() reports exact bbox/count/provenance mix
    on a hand-built PointSet with a known nearest-neighbour structure (a
    1D chain, spacing 1), and save_npz/load_npz round-trips every field
    bit-for-bit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trippy.constants import PROVENANCE_COLMAP, PROVENANCE_GAUSSIAN
from trippy.points.source import PointSet

N = 10


def _make_chain_point_set() -> PointSet:
    # Points at (0,0,0), (1,0,0), ..., (9,0,0): every point's nearest
    # neighbour (including endpoints) is exactly 1 unit away.
    xyz = np.stack([np.arange(N, dtype=np.float32), np.zeros(N, np.float32), np.zeros(N, np.float32)], axis=1)
    size0 = np.full(N, 0.1, dtype=np.float32)
    rgb0 = np.zeros((N, 3), dtype=np.float32)
    conf0 = np.full(N, 0.5, dtype=np.float32)
    provenance = np.array([PROVENANCE_GAUSSIAN] * 5 + [PROVENANCE_COLMAP] * 5, dtype=np.uint8)
    return PointSet(xyz=xyz, size0=size0, rgb0=rgb0, conf0=conf0, provenance=provenance)


def test_summary_counts_bbox_nn_distance_provenance() -> None:
    ps = _make_chain_point_set()
    summary = ps.summary()

    assert summary["count"] == N
    np.testing.assert_allclose(summary["bbox_min"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(summary["bbox_max"], [9.0, 0.0, 0.0])
    assert summary["median_nn_distance"] == 1.0
    assert summary["provenance_histogram"] == {"gaussian": 5, "colmap": 5}


def test_summary_empty_point_set() -> None:
    ps = PointSet(
        xyz=np.zeros((0, 3), np.float32),
        size0=np.zeros(0, np.float32),
        rgb0=np.zeros((0, 3), np.float32),
        conf0=np.zeros(0, np.float32),
        provenance=np.zeros(0, np.uint8),
    )
    summary = ps.summary()
    assert summary["count"] == 0
    assert summary["provenance_histogram"] == {}


def test_npz_round_trip(tmp_path: Path) -> None:
    ps = _make_chain_point_set()
    path = tmp_path / "points.npz"
    ps.save_npz(path)
    assert path.exists()

    loaded = PointSet.load_npz(path)
    np.testing.assert_array_equal(loaded.xyz, ps.xyz)
    np.testing.assert_array_equal(loaded.size0, ps.size0)
    np.testing.assert_array_equal(loaded.rgb0, ps.rgb0)
    np.testing.assert_array_equal(loaded.conf0, ps.conf0)
    np.testing.assert_array_equal(loaded.provenance, ps.provenance)
    assert loaded.xyz.dtype == ps.xyz.dtype
    assert loaded.provenance.dtype == ps.provenance.dtype


def test_mismatched_shape_raises() -> None:
    with pytest.raises(ValueError):
        PointSet(
            xyz=np.zeros((3, 3), np.float32),
            size0=np.zeros(2, np.float32),  # wrong length
            rgb0=np.zeros((3, 3), np.float32),
            conf0=np.zeros(3, np.float32),
            provenance=np.zeros(3, np.uint8),
        )
