"""k-nearest-neighbour distance helpers used for point-size initialisation.

Module: trippy.points.knn_size
Invariants: numpy only, no torch import; all functions are vectorised
    (no per-point Python loops) so they stay usable on multi-million-point
    Gaussian PLYs. Distances/sizes are in world units (COLMAP world frame,
    same scale as the input xyz).
Related docs: docs/SPEC.md D4 (point sources); docs/ARCHITECTURE.md
    (points/ module: "kNN size estimation").
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from trippy.constants import KNN_CHUNK, KNN_SIZE_K, SUMMARY_NN_SAMPLE


def knn_mean_distance(
    xyz: np.ndarray,
    k: int = KNN_SIZE_K,
    chunk: int = KNN_CHUNK,
) -> np.ndarray:
    """Mean distance from each point to its k nearest neighbours.

    Used as a size proxy: a point's local spacing is a reasonable radius
    for its splat when no learned/estimated scale is available.

    Args:
        xyz: (N, 3) float, world-frame positions.
        k: number of neighbours to average (excludes the point itself).
        chunk: rows per cKDTree.query call, bounding peak memory for large
            N (the (rows, k+1) distance array is the dominant allocation).

    Returns:
        (N,) float32 array, mean neighbour distance per point, same units
        as `xyz`. For N < 2, returns zeros (no neighbours exist).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = xyz.shape[0]
    out = np.zeros(n, dtype=np.float32)
    if n < 2:
        return out

    k_eff = min(k, n - 1)
    tree = cKDTree(xyz)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        # k_eff + 1 because column 0 is always the point itself (distance 0).
        dist, _ = tree.query(xyz[start:end], k=k_eff + 1)
        neighbour_dist = dist[:, 1 : k_eff + 1]
        out[start:end] = neighbour_dist.mean(axis=1).astype(np.float32)
    return out


def median_nn_distance(
    xyz: np.ndarray,
    sample: int = SUMMARY_NN_SAMPLE,
    seed: int = 0,
) -> float:
    """Median nearest-neighbour distance on a random subsample.

    Builds the kd-tree on the subsample itself (not the full cloud), so
    this stays fast for multi-million-point clouds; it is a density
    estimate for diagnostics (PointSet.summary()), not an exact per-point
    statistic over the full set.

    Args:
        xyz: (N, 3) float, world-frame positions.
        sample: max number of points to draw without replacement.
        seed: RNG seed for the subsample draw (reproducible summaries).

    Returns:
        Median distance (world units) from each sampled point to its
        nearest neighbour within the subsample. 0.0 if fewer than 2 points.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = xyz.shape[0]
    if n < 2:
        return 0.0

    rng = np.random.default_rng(seed)
    m = min(sample, n)
    idx = rng.choice(n, size=m, replace=False)
    sub = xyz[idx]

    tree = cKDTree(sub)
    dist, _ = tree.query(sub, k=2)
    return float(np.median(dist[:, 1]))
