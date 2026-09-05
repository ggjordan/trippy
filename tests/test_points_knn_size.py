"""Tests for trippy.points.knn_size (kNN mean distance, median nn distance).

Module: tests.test_points_knn_size
Invariants under test: on a regular grid of known spacing, both the kNN
    mean distance (k=4, interior points) and the median nearest-neighbour
    distance (k=1, whole grid) equal the grid spacing exactly -- a regular
    grid is the one shape where "the answer" is unambiguous.
"""

from __future__ import annotations

import numpy as np

from trippy.points.knn_size import knn_mean_distance, median_nn_distance

SPACING = 0.5
GRID_SIDE = 5  # 5x5 grid: interior is the 3x3 block not touching any edge


def _make_grid(spacing: float = SPACING, side: int = GRID_SIDE) -> tuple[np.ndarray, np.ndarray]:
    """A side x side grid in the z=0 plane; returns (xyz, interior_mask)."""
    coords = np.arange(side) * spacing
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    xyz = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)

    ii, jj = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    interior = (ii.ravel() > 0) & (ii.ravel() < side - 1) & (jj.ravel() > 0) & (jj.ravel() < side - 1)
    return xyz, interior


def test_knn_mean_distance_interior_equals_spacing() -> None:
    xyz, interior = _make_grid()
    dist = knn_mean_distance(xyz, k=4)
    # Every interior point's 4 nearest neighbours are its 4 orthogonal
    # neighbours, each exactly `spacing` away -> mean == spacing exactly.
    np.testing.assert_allclose(dist[interior], SPACING, atol=1e-6)
    assert interior.sum() == (GRID_SIDE - 2) ** 2


def test_knn_mean_distance_too_few_points_returns_zeros() -> None:
    xyz = np.zeros((1, 3), dtype=np.float32)
    dist = knn_mean_distance(xyz, k=4)
    np.testing.assert_array_equal(dist, np.zeros(1, dtype=np.float32))


def test_median_nn_distance_whole_grid_equals_spacing() -> None:
    xyz, _ = _make_grid()
    # Every point on a full rectangular grid -- interior, edge, or corner --
    # has an axis-aligned neighbour at exactly `spacing`, and no closer
    # neighbour exists, so the nearest-neighbour distance is exactly
    # `spacing` for every point, hence the median is too.
    result = median_nn_distance(xyz, sample=xyz.shape[0])
    assert result == SPACING


def test_median_nn_distance_single_point_is_zero() -> None:
    xyz = np.zeros((1, 3), dtype=np.float32)
    assert median_nn_distance(xyz) == 0.0
