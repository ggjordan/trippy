"""Tests for `PointSourceConfig.to_source()`: config -> real `PointSource`.

Module: tests.test_train_config_sources
Invariants under test: this complements tests/test_train_config.py (which
    only checks the YAML round-trip stays lossless) by checking `to_source()`
    actually builds a working `trippy.points.PointSource` for the "npz" and
    "union" types EXP-0006 depends on -- "npz" loads a `PointSet` written by
    `PointSet.save_npz` verbatim, and "union" of nested "npz" sources
    concatenates + voxel-dedupes exactly like `trippy.points.union.UnionSource`
    does directly (tests/test_points_union.py), reached this time through the
    config layer a training run or `points-build` actually uses. CPU-only,
    synthetic data throughout -- no real Splats scene or MPS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trippy.constants import PROVENANCE_COLMAP, PROVENANCE_GAUSSIAN, PROVENANCE_MONODEPTH
from trippy.points.source import PointSet
from trippy.train.config import PointSourceConfig


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


def test_npz_type_loads_pointset_verbatim(tmp_path: Path) -> None:
    ps = _ps([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [0.9, 0.4], PROVENANCE_GAUSSIAN)
    npz_path = tmp_path / "points.npz"
    ps.save_npz(npz_path)

    cfg = PointSourceConfig(type="npz", path=str(npz_path))
    source = cfg.to_source()
    assert source.describe() == {"type": "npz", "path": str(npz_path)}

    loaded = source.build()
    assert len(loaded) == 2
    np.testing.assert_allclose(loaded.xyz, ps.xyz)
    np.testing.assert_allclose(loaded.conf0, ps.conf0)
    assert list(loaded.provenance) == list(ps.provenance)


def test_union_of_npz_sources_concatenates_without_voxel(tmp_path: Path) -> None:
    a_path = tmp_path / "a.npz"
    b_path = tmp_path / "b.npz"
    _ps([[0.0, 0.0, 0.0]], [0.9], PROVENANCE_GAUSSIAN).save_npz(a_path)
    _ps([[5.0, 5.0, 5.0]], [0.35], PROVENANCE_MONODEPTH).save_npz(b_path)

    cfg = PointSourceConfig(
        type="union",
        sources=[
            PointSourceConfig(type="npz", path=str(a_path)),
            PointSourceConfig(type="npz", path=str(b_path)),
        ],
    )
    source = cfg.to_source()
    ps = source.build()

    assert len(ps) == 2
    assert set(ps.provenance.tolist()) == {PROVENANCE_GAUSSIAN, PROVENANCE_MONODEPTH}


def test_union_of_npz_sources_voxel_dedupe_keeps_highest_conf(tmp_path: Path) -> None:
    a_path = tmp_path / "a.npz"
    b_path = tmp_path / "b.npz"
    # a and b each have one point colliding in the same voxel cell (a wins, higher conf0)
    # plus one point each that stays distinct.
    _ps([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], [0.9, 0.5], PROVENANCE_GAUSSIAN).save_npz(a_path)
    _ps([[0.01, 0.01, 0.01], [20.0, 20.0, 20.0]], [0.35, 0.35], PROVENANCE_MONODEPTH).save_npz(b_path)

    cfg = PointSourceConfig(
        type="union",
        voxel=1.0,
        sources=[
            PointSourceConfig(type="npz", path=str(a_path)),
            PointSourceConfig(type="npz", path=str(b_path)),
        ],
    )
    ps = cfg.to_source().build()
    summary = ps.summary()

    assert summary["count"] == 3  # one collision resolved, two untouched singletons
    assert summary["provenance_histogram"] == {"gaussian": 2, "monodepth": 1}

    order = np.argsort(ps.xyz[:, 0])
    np.testing.assert_allclose(ps.xyz[order][0], [0.0, 0.0, 0.0], atol=1e-5)
    assert ps.provenance[order][0] == PROVENANCE_GAUSSIAN  # the higher-conf survivor


def test_union_config_describe_matches_children() -> None:
    cfg = PointSourceConfig(
        type="union",
        voxel=0.03,
        sources=[PointSourceConfig(type="colmap", path="sparse_txt")],
    )
    d = cfg.to_source().describe()
    assert d["type"] == "UnionSource"
    assert d["voxel"] == 0.03
    assert d["sources"] == [{"type": "ColmapSparseSource", "sparse_txt_dir": "sparse_txt"}]


def test_unknown_type_raises() -> None:
    cfg = PointSourceConfig(type="lidar")
    with pytest.raises(ValueError, match="unknown point source type"):
        cfg.to_source()


def test_provenance_marker_survives_ordinary_colmap_type() -> None:
    """Sanity check that "union"/"npz" additions didn't disturb the pre-existing types."""
    cfg = PointSourceConfig(type="colmap", path="sparse_txt")
    source = cfg.to_source()
    assert source.describe()["type"] == "ColmapSparseSource"
    assert PROVENANCE_COLMAP == 4
