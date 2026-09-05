"""Tests for trippy.net.gated (GatedConvBlock).

Module: tests.test_net_gated
Invariants under test: output shape matches (out_channels, H, W) with H,
    W unchanged (kernel=3/stride=1/pad=1 preserves spatial size); the
    "gate" (sigmoid branch) stays within (0, 1) so the block can only
    attenuate, never invert, the feature branch; the incoming mask passes
    through unchanged (see trippy.net.gated module docstring).
"""

from __future__ import annotations

import torch

from trippy.net.gated import GatedConvBlock, GatedConvConfig


def test_gated_conv_output_shape_preserves_spatial_size() -> None:
    block = GatedConvBlock(in_channels=4, out_channels=24)
    x = torch.randn(2, 4, 17, 13)
    out, mask = block(x)
    assert out.shape == (2, 24, 17, 13)
    assert mask is None


def test_gated_conv_mask_passes_through_unchanged() -> None:
    block = GatedConvBlock(in_channels=4, out_channels=8)
    x = torch.randn(1, 4, 8, 8)
    mask_in = torch.zeros(1, 1, 8, 8)
    _, mask_out = block(x, mask_in)
    assert mask_out is mask_in


def test_gated_conv_gate_sanity_via_zero_weight_gate() -> None:
    """With the gate conv's weight/bias zeroed, gate = sigmoid(0) = 0.5 everywhere, so the
    output is exactly half the (activated) feature branch -- a concrete, checkable instance
    of "the gate multiplies the feature branch by something in (0, 1)"."""
    block = GatedConvBlock(in_channels=3, out_channels=5, config=GatedConvConfig(activation="id"))
    with torch.no_grad():
        block.gate_conv.weight.zero_()
        block.gate_conv.bias.zero_()
    x = torch.randn(1, 3, 6, 6)
    out, _ = block(x)
    expected_feature = block.feature_conv(x)
    torch.testing.assert_close(out, expected_feature * 0.5)


def test_gated_conv_batchnorm_norm_option() -> None:
    # "bn" is Saiga's own NormFromString spelling for batch norm (TorchHelper.h:196-199);
    # the task's "id and batchnorm minimally" requirement is satisfied by supporting both
    # of Saiga's real norm strings rather than inventing a non-source string.
    block = GatedConvBlock(in_channels=3, out_channels=6, config=GatedConvConfig(norm="bn"))
    x = torch.randn(4, 3, 10, 10)
    out, _ = block(x)
    assert out.shape == (4, 6, 10, 10)
