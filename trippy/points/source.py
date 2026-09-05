"""PointSet data model and the PointSource abstract base class.

Module: trippy.points.source
Invariants: numpy only, no torch import (point sources run on CPU during
    data prep; conversion to torch tensors happens in trippy.train). All
    PointSet arrays are COLMAP world-frame, float32, and row-aligned by
    point index (xyz[i], size0[i], rgb0[i], conf0[i], provenance[i] all
    describe the same point).
Related docs: docs/SPEC.md D4 (pluggable point sources); docs/ARCHITECTURE.md
    (points/ module); docs/GEOMETRY.md (COLMAP world frame conventions).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trippy.constants import PROVENANCE_NAMES
from trippy.points.knn_size import median_nn_distance


@dataclass
class PointSet:
    """A set of splattable points, independent of which source produced them.

    All arrays share axis 0 (N points) and are row-aligned.

    Attributes:
        xyz: (N, 3) float32, COLMAP world-frame position, world units.
        size0: (N,) float32, world-unit radius-like initial splat size
            (semantics depend on the source: 3DGS scale, or a kNN spacing
            estimate; always positive).
        rgb0: (N, 3) float32, initial colour in linear [0, 1] per channel.
        conf0: (N,) float32, initial confidence/opacity in the open
            interval (0, 1) (never exactly 0 or 1 -- callers may take
            logit(conf0)).
        provenance: (N,) uint8, one of the PROVENANCE_* constants in
            trippy.constants, recording which PointSource produced the row.
    """

    xyz: np.ndarray
    size0: np.ndarray
    rgb0: np.ndarray
    conf0: np.ndarray
    provenance: np.ndarray

    def __post_init__(self) -> None:
        n = self.xyz.shape[0]
        if self.xyz.shape != (n, 3):
            raise ValueError(f"xyz must be (N, 3), got {self.xyz.shape}")
        if self.rgb0.shape != (n, 3):
            raise ValueError(f"rgb0 must be (N, 3), got {self.rgb0.shape}")
        for name, arr in (("size0", self.size0), ("conf0", self.conf0), ("provenance", self.provenance)):
            if arr.shape != (n,):
                raise ValueError(f"{name} must be ({n},), got {arr.shape}")

    def __len__(self) -> int:
        return self.xyz.shape[0]

    def summary(self) -> dict:
        """Diagnostic summary: count, bbox, density estimate, provenance mix.

        Returns:
            {
              "count": int,
              "bbox_min": [x, y, z], "bbox_max": [x, y, z] (world units),
              "median_nn_distance": float (world units, 0.0 if N < 2;
                  see trippy.points.knn_size.median_nn_distance),
              "provenance_histogram": {name: count, ...} for provenance
                  values present in the set (unknown codes fall back to
                  their integer as a string key),
            }
        """
        n = len(self)
        if n == 0:
            return {
                "count": 0,
                "bbox_min": [0.0, 0.0, 0.0],
                "bbox_max": [0.0, 0.0, 0.0],
                "median_nn_distance": 0.0,
                "provenance_histogram": {},
            }

        bbox_min = self.xyz.min(axis=0).astype(np.float64).tolist()
        bbox_max = self.xyz.max(axis=0).astype(np.float64).tolist()
        nn_dist = median_nn_distance(self.xyz)

        values, counts = np.unique(self.provenance, return_counts=True)
        histogram = {
            PROVENANCE_NAMES.get(int(v), str(int(v))): int(c) for v, c in zip(values, counts, strict=True)
        }

        return {
            "count": n,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "median_nn_distance": nn_dist,
            "provenance_histogram": histogram,
        }

    def save_npz(self, path: str | Path) -> None:
        """Serialise to a compressed .npz (one array per field)."""
        np.savez_compressed(
            path,
            xyz=self.xyz,
            size0=self.size0,
            rgb0=self.rgb0,
            conf0=self.conf0,
            provenance=self.provenance,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> PointSet:
        """Load a PointSet previously written by save_npz()."""
        with np.load(path) as data:
            return cls(
                xyz=data["xyz"],
                size0=data["size0"],
                rgb0=data["rgb0"],
                conf0=data["conf0"],
                provenance=data["provenance"],
            )


class PointSource(ABC):
    """Common interface for anything that can produce a PointSet.

    Concrete sources: GaussianPlySource, ColmapSparseSource, UnionSource,
    and the MonoDepthSource/LidarSource stubs (see docs/SPEC.md D4).
    """

    @abstractmethod
    def build(self) -> PointSet:
        """Produce the PointSet. May be expensive (I/O, kNN); not cached."""
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict:
        """Return a small JSON-able config dict (path, thresholds, ...).

        For logging/reproducibility -- not the resulting points themselves.
        """
        raise NotImplementedError
