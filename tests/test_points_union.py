"""Tests for trippy.points.union.UnionSource (concat + voxel dedupe).

Module: tests.test_points_union
Invariants under test: without a voxel size, all points from all sources
    are concatenated (provenance preserved); with a voxel size, colliding
    points keep only the highest-conf0 survivor per cell, and points in
    distinct cells are all kept untouched.
"""

from __future__ import annotations

import numpy as np

from trippy.constants import PROVENANCE_COLMAP, PROVENANCE_GAUSSIAN
from trippy.points.source import PointSet, PointSource
from trippy.points.union import UnionSource


class _FixedSource(PointSource):
    """A PointSource that just returns a pre-built PointSet (test double)."""

    def __init__(self, point_set: PointSet) -> None:
        self._point_set = point_set

    def build(self) -> PointSet:
        return self._point_set

    def describe(self) -> dict:
        return {"type": "_FixedSource", "count": len(self._point_set)}


def _ps(xyz, conf0, provenance) -> PointSet:
    xyz = np.asarray(xyz, dtype=np.float32)
    n = xyz.shape[0]
    return PointSet(
        xyz=xyz,
        size0=np.full(n, 0.1, dtype=np.float32),
        rgb0=np.zeros((n, 3), dtype=np.float32),
        conf0=np.asarray(conf0, dtype=np.float32),
        provenance=np.full(n, provenance, dtype=np.uint8),
    )


def test_union_no_voxel_concatenates_all() -> None:
    a = _ps([[0, 0, 0], [10, 10, 10]], [0.9, 0.5], PROVENANCE_GAUSSIAN)
    b = _ps([[0.01, 0.01, 0.01], [20, 20, 20]], [0.3, 0.7], PROVENANCE_COLMAP)
    union = UnionSource([_FixedSource(a), _FixedSource(b)], voxel=None)
    ps = union.build()
    assert len(ps) == 4
    assert list(ps.provenance) == [PROVENANCE_GAUSSIAN, PROVENANCE_GAUSSIAN, PROVENANCE_COLMAP, PROVENANCE_COLMAP]


def test_union_voxel_dedupe_keeps_highest_conf() -> None:
    # a[0] and b[0] collide in the same 1.0 voxel cell; a wins (conf 0.9 > 0.3).
    a = _ps([[0, 0, 0], [10, 10, 10]], [0.9, 0.5], PROVENANCE_GAUSSIAN)
    b = _ps([[0.01, 0.01, 0.01], [20, 20, 20]], [0.3, 0.7], PROVENANCE_COLMAP)
    union = UnionSource([_FixedSource(a), _FixedSource(b)], voxel=1.0)
    ps = union.build()

    assert len(ps) == 3  # one collision resolved, two untouched singletons

    order = np.argsort(ps.xyz[:, 0])
    xyz_sorted = ps.xyz[order]
    conf_sorted = ps.conf0[order]
    prov_sorted = ps.provenance[order]

    np.testing.assert_allclose(xyz_sorted[0], [0.0, 0.0, 0.0], atol=1e-5)
    assert conf_sorted[0] == np.float32(0.9)
    assert prov_sorted[0] == PROVENANCE_GAUSSIAN

    np.testing.assert_allclose(xyz_sorted[1], [10.0, 10.0, 10.0])
    np.testing.assert_allclose(xyz_sorted[2], [20.0, 20.0, 20.0])


def test_union_empty_sources() -> None:
    union = UnionSource([], voxel=None)
    ps = union.build()
    assert len(ps) == 0

    union_voxel = UnionSource([], voxel=1.0)
    assert len(union_voxel.build()) == 0


def test_union_describe_includes_children() -> None:
    a = _ps([[0, 0, 0]], [0.9], PROVENANCE_GAUSSIAN)
    union = UnionSource([_FixedSource(a)], voxel=2.0)
    d = union.describe()
    assert d["type"] == "UnionSource"
    assert d["voxel"] == 2.0
    assert len(d["sources"]) == 1
