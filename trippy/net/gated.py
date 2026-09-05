"""Gated convolution block, ported from Saiga::GatedBlockImpl.

Module: trippy.net.gated
Invariants: kernel_size is always 3 (matches Saiga's own
    `SAIGA_ASSERT(kernel_size == 3)`); this is the only conv block type
    `MultiScaleUnet2dDecOnlySmallFixed` ever instantiates
    (`conv_block_up=gated`, configs/train_normalnet.ini:215).
Related docs: docs/TRIPS_REFERENCE.md Sec. 5 (network) for the exact
    formula + fetch provenance; docs/LIMITATIONS.md for what remains
    unverified.

Source (fetched over the network; MIT, no private data -- External/saiga/
is an empty dir in the vendored TRIPS checkout so this could not be read
locally):
    https://github.com/darglein/saiga
    commit ee7a4e6b65832433e2ca521353b7b7431c8e17a0
    src/saiga/vision/torch/PartialConvUnet2d.h:108-150 (GatedBlockImpl)
    src/saiga/vision/torch/TorchHelper.h:194-246 (NormFromString / ActivationFromString)

Exact C++ (GatedBlockImpl::forward, PartialConvUnet2d.h:139-145):
    auto x_t = feature_transform->forward(x);   // Conv2d(in,out,k,s,d,pad) -> Activation
    auto m_t = mask_transform->forward(x);      // Conv2d(in,out,k,s,d,pad) -> Sigmoid
    auto res = norm.forward(x_t * m_t);
    return {res, mask};                          // the incoming validity `mask` passes through
                                                  // UNCHANGED -- gated blocks never touch it
                                                  // (only conv_block="partial_multi" does real
                                                  # partial-conv masking, which this network
                                                  # never uses: Networks.h:1028 asserts
                                                  # conv_block != "partial_multi").
Both convs read from the SAME input `x` (not from each other's output and not from `mask`);
they have independent weights but identical (in_channels, out_channels, kernel, stride,
dilation, padding).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from trippy.constants import GATED_CONV_KERNEL_SIZE


def activation_from_string(name: str) -> nn.Module:
    """Port of Saiga::ActivationFromString (TorchHelper.h:220-246), subset actually used.

    TRIPS's shipped configs only ever pass "elu" (gated-block feature activation) or "id"
    (the network's `last_act`), so only the branches needed to be bit-exact for those are
    implemented plus a few obvious extras for config flexibility (sigmoid/tanh/relu/silu use
    torch's defaults, which match libtorch's -- both are the same upstream library).
    """
    key = name.lower()
    if key in ("id", "none", ""):
        return nn.Identity()
    if key == "sigmoid":
        return nn.Sigmoid()
    if key == "tanh":
        return nn.Tanh()
    if key == "elu":
        return nn.ELU()
    if key == "relu":
        return nn.ReLU()
    if key == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation {name!r}")


def norm_from_string(name: str, channels: int) -> nn.Module:
    """Port of Saiga::NormFromString (TorchHelper.h:194-218), "id" and "bn" (minimum required).

    "bn" uses momentum=0.01, matching Saiga's
    `torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channels).momentum(0.01))` exactly
    (TorchHelper.h:198) -- PyTorch's own default momentum is 0.1, so this must be set
    explicitly to match.
    """
    key = name.lower()
    if key == "id":
        return nn.Identity()
    if key == "bn":
        return nn.BatchNorm2d(channels, momentum=0.01)
    raise ValueError(f"Unknown norm {name!r}")


@dataclass
class GatedConvConfig:
    """Config for GatedConvBlock.

    Attributes:
        activation: name for the feature-branch activation (Saiga
            ActivationFromString). TRIPS always uses "elu"
            (train_normalnet.ini:217 `activation = elu`).
        norm: name for the post-gate norm (Saiga NormFromString). TRIPS
            always uses "id" (train_normalnet.ini:211-212
            `norm_layer_down`/`norm_layer_up = id`).
    """

    activation: str = "elu"
    norm: str = "id"


class GatedConvBlock(nn.Module):
    """Port of Saiga::GatedBlockImpl -- see module docstring for the source formula.

    Shapes:
        input x: (B, in_channels, H, W)
        input mask: any shape or None -- passed through unchanged, never read.
        output: ((B, out_channels, H, W), mask) -- H, W unchanged (kernel=3,
            stride=1, padding=1 always preserves spatial size).
    """

    def __init__(self, in_channels: int, out_channels: int, config: GatedConvConfig | None = None) -> None:
        super().__init__()
        config = config or GatedConvConfig()
        kernel_size = GATED_CONV_KERNEL_SIZE
        dilation = 1
        stride = 1
        # PartialConvUnet2d.h:114 `n_pad_pxl = int(dilation * (kernel_size - 1) / 2)`.
        padding = dilation * (kernel_size - 1) // 2

        self.feature_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding
        )
        self.activation = activation_from_string(config.activation)
        self.gate_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding
        )
        self.norm = norm_from_string(config.norm, out_channels)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        feature = self.activation(self.feature_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        out = self.norm(feature * gate)
        return out, mask
