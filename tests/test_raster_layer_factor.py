"""TRIPS's piecewise layer-factor formula, against hand-computed values.

Module: tests.test_raster_layer_factor
Invariants: the expected numbers below are computed by hand from
    third_party/TRIPS/src/lib/rendering/PointBlending.h:81-149, not by
    running the implementation. They pin the three behaviours
    docs/GEOMETRY.md previously got wrong (docs/TRIPS_REFERENCE.md section
    10.2): the exponential sub-pixel floor on the *factor*, interpolation
    linear in point-size units rather than in the log2 fraction, and the
    clamp to the top layer.
Related docs: docs/GEOMETRY.md "Pyramid level selection";
    docs/TRIPS_REFERENCE.md sections 3 and 10.
"""

from __future__ import annotations

import math

import pytest
import torch

from trippy.constants import RASTER_SMALL_POINT_CUTOFF
from trippy.raster.emit import layer_bounds, layer_factor
from trippy.raster.ref_numpy import layer_bounds_scalar, layer_factor_scalar

NUM_LAYERS = 5
TOL = 1e-12


def _factor(size_px: float, layer: int, num_layers: int = NUM_LAYERS) -> float:
    """Evaluate the torch implementation on a single float64 size."""
    size = torch.tensor([size_px], dtype=torch.float64)
    return float(layer_factor(size, layer, num_layers)[0])


@pytest.mark.parametrize(
    ("size_px", "expected"),
    [
        (0.25, (0, 0)),
        (0.5, (0, 0)),
        (1.0, (0, 0)),  # `point_size_opt > 1` is strict: exactly 1 stays on layer 0
        (1.5, (0, 1)),
        (2.0, (1, 1)),  # exact power of two: floor == ceil
        (2.46, (1, 2)),
        (4.0, (2, 2)),
        (12.0, (3, 4)),
        (16.0, (4, 4)),
        (100.0, (4, 4)),  # log2 = 6.64, both clamped to the top layer
    ],
)
def test_layer_bounds_hand_values(size_px: float, expected: tuple[int, int]) -> None:
    """floor/ceil of log2(size), clamped -- PointBlending.h:86-93."""
    sizes = torch.tensor([size_px], dtype=torch.float64)
    lower, upper = layer_bounds(sizes, NUM_LAYERS)
    assert (int(lower[0]), int(upper[0])) == expected
    assert layer_bounds_scalar(size_px, NUM_LAYERS) == expected


def test_sub_pixel_points_get_the_exponential_floor() -> None:
    """size <= 1 -> (1 - 0.25) * exp(size - 1) + 0.25 (PointBlending.h:106)."""
    expected_half = (1.0 - RASTER_SMALL_POINT_CUTOFF) * math.exp(0.5 - 1.0) + RASTER_SMALL_POINT_CUTOFF
    assert expected_half == pytest.approx(0.70489799478447507, abs=TOL)
    assert _factor(0.5, 0) == pytest.approx(expected_half, abs=TOL)
    # It is a floor on the factor, NOT a clamp on the size: a 0.01 px point
    # still contributes, at just above the 0.25 cutoff.
    assert _factor(0.01, 0) == pytest.approx(0.75 * math.exp(-0.99) + 0.25, abs=TOL)
    assert _factor(0.01, 0) > RASTER_SMALL_POINT_CUTOFF
    # ... and the branch is continuous with 1.0 at size == 1.
    assert _factor(1.0, 0) == pytest.approx(1.0, abs=TOL)


def test_exact_power_of_two_gives_factor_one() -> None:
    """floor(log2 s) == ceil(log2 s) -> the point lands wholly on one layer."""
    assert _factor(2.0, 1) == pytest.approx(1.0, abs=TOL)
    assert _factor(4.0, 2) == pytest.approx(1.0, abs=TOL)
    assert _factor(8.0, 3) == pytest.approx(1.0, abs=TOL)


def test_interpolation_is_linear_in_point_size_not_in_log2() -> None:
    """s = 2**1.3 = 2.46 -> 0.23 toward the upper layer, not 0.3.

    This is the worked example in docs/TRIPS_REFERENCE.md section 10.2.
    """
    size_px = 2.0**1.3
    upper = _factor(size_px, 2)
    lower = _factor(size_px, 1)
    assert upper == pytest.approx((size_px - 2.0) / (4.0 - 2.0), abs=TOL)
    assert upper == pytest.approx(0.23, abs=2e-3)
    assert upper != pytest.approx(0.3, abs=1e-2)
    assert lower == pytest.approx(1.0 - upper, abs=TOL)


@pytest.mark.parametrize(("size_px", "lower_layer"), [(1.5, 0), (3.0, 1), (6.0, 2), (12.0, 3)])
def test_interpolation_weights_sum_to_one(size_px: float, lower_layer: int) -> None:
    """The two layers a point straddles always share a unit of alpha."""
    total = _factor(size_px, lower_layer) + _factor(size_px, lower_layer + 1)
    assert total == pytest.approx(1.0, abs=TOL)


def test_sizes_beyond_the_top_layer_clamp_to_one() -> None:
    """A point bigger than the coarsest layer writes into it with factor 1."""
    for size_px in (16.0, 20.0, 100.0, 1e6):
        assert _factor(size_px, NUM_LAYERS - 1) == pytest.approx(1.0, abs=TOL)


def test_layer_below_lower_returns_one_matching_the_cpp_source() -> None:
    """`if (layer < layer_lower) return 1.f;` -- PointBlending.h:96-100.

    docs/TRIPS_REFERENCE.md section 3 says this branch returns 0; the C++
    returns 1. It is inert either way: TRIPS's point-size emission kernel
    `CollectTiled2Pointsize` writes only `layer_lower` and `layer_higher`
    (RenderForward.cu:2296-2360), and so does trippy's "trilinear" mode --
    neither ever asks for a layer below `lower`. The port matches the
    source, not the doc.
    """
    assert _factor(4.0, 0) == pytest.approx(1.0, abs=TOL)
    assert _factor(100.0, 3) == pytest.approx(1.0, abs=TOL)


def test_torch_and_numpy_implementations_agree() -> None:
    """The vectorised torch formula matches the scalar numpy reference."""
    sizes = torch.logspace(-2, 2.5, 200, dtype=torch.float64)
    for layer in range(NUM_LAYERS):
        got = layer_factor(sizes, layer, NUM_LAYERS)
        for index in range(sizes.numel()):
            want = layer_factor_scalar(float(sizes[index]), layer, NUM_LAYERS)
            assert float(got[index]) == pytest.approx(want, abs=1e-12)


def test_factor_is_finite_for_extreme_sizes() -> None:
    """exp(size - 1) must not overflow for huge sizes (clamped input)."""
    sizes = torch.tensor([1e-30, 1e30], dtype=torch.float64)
    for layer in range(NUM_LAYERS):
        assert torch.isfinite(layer_factor(sizes, layer, NUM_LAYERS)).all()


def test_single_layer_pyramid_is_degenerate_but_valid() -> None:
    """L = 1 clamps everything to layer 0 with factor 1 (no interpolation)."""
    assert _factor(10.0, 0, num_layers=1) == pytest.approx(1.0, abs=TOL)
    sizes = torch.tensor([10.0], dtype=torch.float64)
    lower, upper = layer_bounds(sizes, 1)
    assert int(lower[0]) == 0 and int(upper[0]) == 0
