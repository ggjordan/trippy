"""PointSource that concatenates other sources, with optional voxel dedupe.

Module: trippy.points.union
Invariants: vectorised numpy throughout (no per-point Python loop), even
    for the voxel dedupe -- union sets can be large (Gaussian + monodepth
    combined). Provenance is preserved through concatenation and dedupe so
    downstream code can still tell which source each surviving point came
    from.
Related docs: docs/SPEC.md D4 point source 3 ("union" of Gaussians +
    monocular depth).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trippy.points.source import PointSet, PointSource


class UnionSource(PointSource):
    """Concatenate PointSets from multiple sources.

    Args:
        sources: child PointSources, built and concatenated in order.
        voxel: if set, world-unit voxel edge length; points are grouped
            into a regular grid of this cell size and, within each
            occupied cell, only the highest-conf0 point survives. If
            None, all points from all sources are kept (no dedupe).
    """

    def __init__(self, sources: Sequence[PointSource], voxel: float | None = None) -> None:
        if voxel is not None and voxel <= 0:
            raise ValueError(f"voxel must be positive, got {voxel}")
        self.sources = list(sources)
        self.voxel = voxel

    def describe(self) -> dict:
        return {
            "type": "UnionSource",
            "voxel": self.voxel,
            "sources": [s.describe() for s in self.sources],
        }

    def build(self) -> PointSet:
        point_sets = [s.build() for s in self.sources]
        if not point_sets:
            return PointSet(
                xyz=np.zeros((0, 3), dtype=np.float32),
                size0=np.zeros(0, dtype=np.float32),
                rgb0=np.zeros((0, 3), dtype=np.float32),
                conf0=np.zeros(0, dtype=np.float32),
                provenance=np.zeros(0, dtype=np.uint8),
            )

        xyz = np.concatenate([p.xyz for p in point_sets], axis=0)
        size0 = np.concatenate([p.size0 for p in point_sets], axis=0)
        rgb0 = np.concatenate([p.rgb0 for p in point_sets], axis=0)
        conf0 = np.concatenate([p.conf0 for p in point_sets], axis=0)
        provenance = np.concatenate([p.provenance for p in point_sets], axis=0)

        if self.voxel is None or xyz.shape[0] == 0:
            return PointSet(xyz=xyz, size0=size0, rgb0=rgb0, conf0=conf0, provenance=provenance)

        keep = _voxel_dedupe_keep_highest_conf(xyz, conf0, self.voxel)
        return PointSet(
            xyz=xyz[keep],
            size0=size0[keep],
            rgb0=rgb0[keep],
            conf0=conf0[keep],
            provenance=provenance[keep],
        )


def _voxel_dedupe_keep_highest_conf(xyz: np.ndarray, conf0: np.ndarray, voxel: float) -> np.ndarray:
    """Indices (into xyz/conf0) of the highest-conf0 point per voxel cell.

    Fully vectorised: groups points by integer voxel coordinate via
    np.unique, then a single lexsort (by group, then by descending conf0)
    picks the first row of each group as the survivor.

    Args:
        xyz: (N, 3) float, world-frame positions.
        conf0: (N,) float, confidence used as the tie-break/selection key.
        voxel: world-unit cell edge length (> 0).

    Returns:
        (M,) int array of surviving row indices, one per occupied voxel
        cell, in arbitrary (group-id) order.
    """
    voxel_idx = np.floor(xyz.astype(np.float64) / voxel).astype(np.int64)
    _, inverse = np.unique(voxel_idx, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)

    # Sort by group id ascending, then by conf0 descending within a group,
    # so the first row of each contiguous group run is the highest-conf0
    # member of that voxel cell.
    order = np.lexsort((-conf0, inverse))
    sorted_groups = inverse[order]
    is_first_in_group = np.empty(order.shape[0], dtype=bool)
    is_first_in_group[0] = True
    is_first_in_group[1:] = sorted_groups[1:] != sorted_groups[:-1]

    return order[is_first_in_group]
