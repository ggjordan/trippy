"""PointSource that reads a COLMAP sparse reconstruction (points3D.txt).

Module: trippy.points.colmap_sparse
Invariants: text format only. Binary points3D.bin reading belongs to
    trippy.scene (COLMAP dataset loading), not here -- this module only
    turns an already-parsed sparse model into a PointSet.
Related docs: docs/SPEC.md D4 (point sources); docs/GEOMETRY.md (COLMAP
    world frame); trippy.geom.xform_a.read_points3d_txt (the parser used).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trippy.constants import COLMAP_DEFAULT_CONF0, KNN_CHUNK, KNN_SIZE_K, PROVENANCE_COLMAP
from trippy.geom.xform_a import read_points3d_txt
from trippy.points.knn_size import knn_mean_distance
from trippy.points.source import PointSet, PointSource

_POINTS3D_FILENAME = "points3D.txt"


class ColmapSparseSource(PointSource):
    """COLMAP sparse triangulated points as splat points (D4 point source).

    Args:
        sparse_txt_dir: directory containing a text-format COLMAP sparse
            model (must contain points3D.txt; cameras.txt/images.txt are
            not read here).
    """

    def __init__(self, sparse_txt_dir: str | Path) -> None:
        self.sparse_txt_dir = Path(sparse_txt_dir)

    def describe(self) -> dict:
        return {
            "type": "ColmapSparseSource",
            "sparse_txt_dir": str(self.sparse_txt_dir),
        }

    def build(self) -> PointSet:
        points3d_path = self.sparse_txt_dir / _POINTS3D_FILENAME
        points = read_points3d_txt(str(points3d_path))

        n = len(points)
        if n == 0:
            return PointSet(
                xyz=np.zeros((0, 3), dtype=np.float32),
                size0=np.zeros(0, dtype=np.float32),
                rgb0=np.zeros((0, 3), dtype=np.float32),
                conf0=np.zeros(0, dtype=np.float32),
                provenance=np.zeros(0, dtype=np.uint8),
            )

        xyz = np.stack([p["xyz"] for p in points.values()], axis=0).astype(np.float32)
        rgb_uint8 = np.stack([p["rgb"] for p in points.values()], axis=0)
        rgb0 = (rgb_uint8.astype(np.float32)) / 255.0
        conf0 = np.full(n, COLMAP_DEFAULT_CONF0, dtype=np.float32)
        size0 = knn_mean_distance(xyz, k=KNN_SIZE_K, chunk=KNN_CHUNK)
        provenance = np.full(n, PROVENANCE_COLMAP, dtype=np.uint8)

        return PointSet(xyz=xyz, size0=size0, rgb0=rgb0, conf0=conf0, provenance=provenance)
