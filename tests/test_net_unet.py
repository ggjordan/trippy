"""Tests for trippy.net.unet (MultiScaleUnet2dDecOnlySmallFixed).

Module: tests.test_net_unet
Invariants under test: forward() on a finest-first pyramid produces
    (B, num_output_channels, H, W); the default config's parameter count
    matches a hand-computed total from the architecture table below
    (independently derived from third_party/TRIPS/src/lib/models/
    Networks.h, not from reading unet.py's implementation); odd base
    sizes are centre-cropped to a multiple of 2**(num_layers-1) rather
    than crashing (see trippy/net/unet.py module docstring "CombineBridge
    / odd-size handling").

-- Architecture table (default config: filters=32, num_input_channels=4,
   num_layers=5, num_output_channels=3), verified against
   third_party/TRIPS/src/lib/models/Networks.h -- see docs/TRIPS_REFERENCE.md
   Sec. 5 for the full per-level table with exact line numbers:

   | Stage      | Conv in -> out | GatedConvBlock params = 2*(9*in*out + out) |
   |------------|----------------|---------------------------------------------|
   | start      | 4  -> 24       | 2*(9*4*24 + 24)   = 1776                    |
   | up[3]      | 32 -> 24       | 2*(9*32*24 + 24)  = 13872                   |
   | up[2]      | 32 -> 24       | 2*(9*32*24 + 24)  = 13872                   |
   | up[1]      | 32 -> 24       | 2*(9*32*24 + 24)  = 13872                   |
   | up[0] last | 32 -> 28       | 2*(9*32*28 + 28)  = 16184                   |
   | final      | Conv2d(32->3, kernel=1): 3*32*1*1 + 3 = 99                  |

   Each GatedConvBlock has two independent 3x3 convs (feature_transform,
   mask_transform) of identical (in, out) shape and no norm params
   ("id" is parameter-free) -- hence the `2*(9*in*out + out)` formula
   (9 = 3x3 kernel, "+out" is each conv's bias). Upsample (bilinear) has
   no parameters.

   Total = 1776 + 13872*3 + 16184 + 99 = 59675.
"""

from __future__ import annotations

import torch

from trippy.net.unet import MultiScaleUnet2dDecOnlySmallFixed, NetworkConfig

HAND_COMPUTED_DEFAULT_PARAM_COUNT = 1776 + 13872 * 3 + 16184 + 99


def _make_pyramid(base_h: int, base_w: int, num_layers: int, channels: int, batch: int) -> list[torch.Tensor]:
    """Finest-first pyramid: level i has floor(base/2**i) spatial size (matches
    docs/TRIPS_REFERENCE.md Sec. 3: "Layer 0 = full render resolution; each subsequent layer
    is h/=2; w/=2 (integer division)")."""
    pyramid = []
    h, w = base_h, base_w
    for _ in range(num_layers):
        pyramid.append(torch.randn(batch, channels, h, w))
        h, w = h // 2, w // 2
    return pyramid


def test_hand_computed_param_count_matches_worked_arithmetic() -> None:
    # Sanity-check the docstring's arithmetic itself before comparing to the model.
    assert HAND_COMPUTED_DEFAULT_PARAM_COUNT == 59675


def test_default_config_parameter_count_matches_hand_derivation() -> None:
    net = MultiScaleUnet2dDecOnlySmallFixed(NetworkConfig())
    assert net.parameter_count() == HAND_COMPUTED_DEFAULT_PARAM_COUNT


def test_forward_even_size() -> None:
    net = MultiScaleUnet2dDecOnlySmallFixed(NetworkConfig())
    pyramid = _make_pyramid(base_h=64, base_w=48, num_layers=5, channels=4, batch=2)
    out = net(pyramid)
    assert out.shape == (2, 3, 64, 48)
    assert torch.isfinite(out).all()


def test_forward_odd_size_centre_crops_to_multiple_of_16() -> None:
    """Base 63x47 is not divisible by 2**(5-1)=16; TRIPS's own floor-halving pyramid
    construction (docs/TRIPS_REFERENCE.md Sec. 3) then produces a raw finer-level input one
    pixel larger than 2x the coarser level at some intermediate stage (worked out by hand in
    trippy/net/unet.py's module docstring and docs/LIMITATIONS.md). trippy's generalized
    combine_bridge() centre-crops rather than crashing; the well-defined result is the
    largest multiple of 16 that fits inside the base size: floor(63/16)*16=48,
    floor(47/16)*16=32."""
    net = MultiScaleUnet2dDecOnlySmallFixed(NetworkConfig())
    pyramid = _make_pyramid(base_h=63, base_w=47, num_layers=5, channels=4, batch=1)
    out = net(pyramid)
    assert out.shape == (1, 3, 48, 32)
    assert torch.isfinite(out).all()


def test_wrong_number_of_pyramid_levels_raises() -> None:
    net = MultiScaleUnet2dDecOnlySmallFixed(NetworkConfig())
    pyramid = _make_pyramid(base_h=32, base_w=32, num_layers=4, channels=4, batch=1)
    try:
        net(pyramid)
    except ValueError as exc:
        assert "num_layers" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched pyramid length")
