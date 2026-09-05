"""MultiScaleUnet2dDecOnlySmallFixed: TRIPS's default render/tone network.

Module: trippy.net.unet
Invariants: pyramid inputs are always ordered finest-first, i.e. `inputs[0]`
    is full render resolution and `inputs[len(inputs) - 1]` is the
    coarsest level -- this matches TRIPS's own convention (see
    docs/TRIPS_REFERENCE.md Sec. 3: "Layer 0 = full render resolution
    (h,w) from the dataset; each subsequent layer is h/=2; w/=2"). All
    pyramid tensors have `num_input_channels` channels and are float32.
Related docs: docs/TRIPS_REFERENCE.md Sec. 5 (network architecture table,
    verified against third_party/TRIPS/src/lib/models/Networks.h);
    docs/LIMITATIONS.md (odd-size CombineBridge generalization).

Source: third_party/TRIPS/src/lib/models/Networks.h (TRIPS @
commit a59a65b6d9a8b1c14c73bc004cc9a8956f054c24):
    SmallDecStartBlockImpl                    Networks.h:751-787
    UpsampleDecOnlySmallBlockFixedImpl        Networks.h:999-1097
    MultiScaleUnet2dDecOnlySmallFixedImpl     Networks.h:1100-1208
Config defaults: configs/train_normalnet.ini:202-219 (`[NetParams]`, name
inferred from context -- section header itself not re-verified here since
Sec. 5 of TRIPS_REFERENCE.md already extracted every field individually).

-- Architecture (verified against source; see docs/TRIPS_REFERENCE.md Sec. 5
   for the full per-level table) --
Let C = num_input_channels (4), F = filters (32, constant across all used
levels per the shipped ini -- see trippy.constants.NET_DEFAULT_FILTERS),
L = num_layers (5). Coarsest level is L-1, finest is 0.

    start (level L-1):  gated(C -> F-2C) on inputs[L-1]
                         bridge = cat(inputs[L-1], conv_out)             -> F-C channels
    up[i] (i = L-2..1):  upsample(bridge, x2) -> cat(inputs[i], .) = F   (in-channels)
                         gated(F -> F-2C)
                         bridge = cat(inputs[i], conv_out)               -> F-C channels
    up[0] (last=True):   upsample(bridge, x2) -> cat(inputs[0], .) = F
                         gated(F -> F-C)                                 (NOT F-2C: Networks.h:1034
                                                                           `(!last) ? out-2C : out-C`)
                         bridge = cat(inputs[0], conv_out)               -> F channels
    final:               Conv2d(F -> num_output_channels, kernel=1) + Activation(last_act)

-- CombineBridge / odd-size handling (a deliberate, documented generalization) --
TRIPS's `CombineBridge(below, skip)` (Networks.h:766-773, 1060-1067) centre-crops `skip` down
to `below`'s spatial size when they differ, using C++ `int(...)/2` (truncating) diffs. This
assumes `skip.size() >= below.size()`, which holds whenever every intermediate pyramid level's
H and W happen to be even. It is *not* always true: TRIPS's own pyramid halves each level with
plain integer division (`h/=2; w/=2`, PointRenderer.cu:378, see docs/TRIPS_REFERENCE.md Sec. 3),
so an odd intermediate dimension H_i produces a *finer* raw skip input (H_i) one pixel *larger*
than 2x the coarser level's floor-halved size (2*floor(H_i/2) = H_i-1 when H_i is odd) -- i.e.
`below` ends up larger than `skip`, the opposite of what TRIPS's literal crop-`skip`-to-`below`
assumes. A faithful byte-for-byte port of that exact code, fed such a pyramid, would attempt to
"crop" a smaller tensor `skip` up to a larger target and then fail (or silently mis-size) in
`torch.cat`. See docs/LIMITATIONS.md for the worked numeric example (base 1008x756 with
num_layers=5).

trippy's `combine_bridge()` therefore generalizes: crop *whichever* of `below`/`skip` is larger,
down to the shared minimum H and W, symmetric centre-crop on each axis independently. When
`skip >= below` (TRIPS's designed case -- true whenever the base resolution divides evenly by
`2 ** (num_layers - 1)`), this is bit-identical to TRIPS's own CombineBridge (crops only
`skip`, `below` untouched). It only differs, deliberately and safely, in the odd-size case
TRIPS's own code cannot handle. One consequence: for a base resolution not divisible by
2**(num_layers-1), the whole network's *output* spatial size becomes
`floor(H / 2**(L-1)) * 2**(L-1)` (by W as well) -- a centre-cropped multiple of `2**(L-1)`, not
the original H, W. This is tested in tests/test_net_unet.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from trippy.constants import (
    NET_DEFAULT_ACTIVATION,
    NET_DEFAULT_FILTERS,
    NET_DEFAULT_LAST_ACT,
    NET_DEFAULT_NORM,
    NET_DEFAULT_NUM_INPUT_CHANNELS,
    NET_DEFAULT_NUM_LAYERS,
    NET_DEFAULT_NUM_OUTPUT_CHANNELS,
    NET_DEFAULT_UPSAMPLE_MODE,
    NET_UPSAMPLE_SCALE_FACTOR,
)
from trippy.net.gated import GatedConvBlock, GatedConvConfig, activation_from_string


def _center_crop_to(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Centre-crop x's spatial dims down to (target_h, target_w). Requires x >= target."""
    h, w = x.shape[2], x.shape[3]
    assert h >= target_h and w >= target_w, (
        f"_center_crop_to requires x >= target, got x=({h},{w}) target=({target_h},{target_w})"
    )
    diff_h = (h - target_h) // 2
    diff_w = (w - target_w) // 2
    return x[:, :, diff_h : diff_h + target_h, diff_w : diff_w + target_w]


def combine_bridge(below: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """Concatenate `below` and `skip` on the channel dim, centre-cropping to match.

    Generalization of Networks.h's `CombineBridge(below, skip)` -- see module docstring
    "CombineBridge / odd-size handling" for the exact relationship to the C++ source and why
    this crops symmetrically instead of assuming `skip >= below`.
    """
    if below.shape[2] == skip.shape[2] and below.shape[3] == skip.shape[3]:
        return torch.cat([below, skip], dim=1)
    target_h = min(below.shape[2], skip.shape[2])
    target_w = min(below.shape[3], skip.shape[3])
    below = _center_crop_to(below, target_h, target_w)
    skip = _center_crop_to(skip, target_h, target_w)
    return torch.cat([below, skip], dim=1)


@dataclass
class NetworkConfig:
    """Config for MultiScaleUnet2dDecOnlySmallFixed.

    Every default cites configs/train_normalnet.ini (TRIPS @
    a59a65b6d9a8b1c14c73bc004cc9a8956f054c24); see trippy.constants for the
    named constant + exact ini line each maps to.

    Attributes:
        num_input_channels: channels per raw pyramid input tensor
            (ini:203 num_input_channels).
        num_output_channels: channels of the final displayed image
            (ini:204 num_output_channels).
        filters: constant channel budget used at every pyramid level
            (ini:219 filters_network -- see trippy.constants for why
            TRIPS's per-level list simplifies to one scalar here).
        num_layers: number of pyramid levels consumed, coarsest to
            finest (ini:206 num_layers; must equal len(inputs) at
            forward() time).
        activation: gated-block feature activation (ini:217 activation).
        norm: gated-block post-gate norm (ini:211-212 norm_layer_up).
        upsample_mode: "bilinear" (ini:210), "nearest", or "deconv" --
            Networks.h:1012-1026 supports all three; ini ships "bilinear".
        last_act: activation after the final 1x1 conv (ini:213 last_act).
    """

    num_input_channels: int = NET_DEFAULT_NUM_INPUT_CHANNELS
    num_output_channels: int = NET_DEFAULT_NUM_OUTPUT_CHANNELS
    filters: int = NET_DEFAULT_FILTERS
    num_layers: int = NET_DEFAULT_NUM_LAYERS
    activation: str = NET_DEFAULT_ACTIVATION
    norm: str = NET_DEFAULT_NORM
    upsample_mode: str = NET_DEFAULT_UPSAMPLE_MODE
    last_act: str = NET_DEFAULT_LAST_ACT
    gated: GatedConvConfig = field(default_factory=GatedConvConfig)

    def __post_init__(self) -> None:
        self.gated = GatedConvConfig(activation=self.activation, norm=self.norm)
        if self.filters <= 2 * self.num_input_channels:
            raise ValueError(
                "filters must be > 2*num_input_channels (the start block's conv output "
                f"channel count filters-2*num_input_channels must stay positive); got "
                f"filters={self.filters}, num_input_channels={self.num_input_channels}"
            )


def _build_upsample(mode: str, channels: int) -> nn.Module:
    """Port of the `up` Sequential built in UpsampleDecOnlySmallBlockFixedImpl (Networks.h:1009-1026)."""
    scale = NET_UPSAMPLE_SCALE_FACTOR
    if mode == "bilinear":
        return nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)
    if mode == "nearest":
        return nn.Upsample(scale_factor=scale, mode="nearest")
    if mode == "deconv":
        # Networks.h:1014-1015: ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1).
        # in_channels == out_channels == `channels` here because filters is constant across
        # levels in every shipped config (see NetworkConfig.filters docstring).
        return nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
    raise ValueError(f"Unknown upsample_mode {mode!r}")


class SmallDecStartBlock(nn.Module):
    """Port of SmallDecStartBlockImpl (Networks.h:751-787): the coarsest-level entry block."""

    def __init__(self, in_channels: int, out_channels: int, gated_config: GatedConvConfig) -> None:
        super().__init__()
        self.conv = GatedConvBlock(in_channels, out_channels, gated_config)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        conv_out, mask = self.conv(x, mask)
        # Networks.h:780: `output.first = CombineBridge(x, output.first)`. The gated conv
        # always preserves H, W (kernel=3, stride=1, padding=1), so this is always the
        # equal-size branch of combine_bridge (plain concat, no crop).
        return combine_bridge(x, conv_out), mask


class UpsampleDecOnlySmallBlockFixed(nn.Module):
    """Port of UpsampleDecOnlySmallBlockFixedImpl (Networks.h:999-1097)."""

    def __init__(
        self,
        filters: int,
        num_input_channels: int,
        last: bool,
        gated_config: GatedConvConfig,
        upsample_mode: str,
    ) -> None:
        super().__init__()
        self.upsample = _build_upsample(upsample_mode, filters)
        # Networks.h:1033-1034: last blocks output `filters - num_input_channels`, others
        # `filters - 2*num_input_channels` (the "-2C" trick referenced in the task brief).
        conv_out_channels = filters - num_input_channels if last else filters - 2 * num_input_channels
        self.conv = GatedConvBlock(filters, conv_out_channels, gated_config)

    def forward(
        self,
        layer_below: tuple[torch.Tensor, torch.Tensor | None],
        raw_input: tuple[torch.Tensor, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        below_x, _below_mask = layer_below
        raw_x, raw_mask = raw_input

        upsampled = self.upsample(below_x)
        # Networks.h:1082: `combined_tensors.first = CombineBridge(features_input.first,
        # upsample_input.first)` -- below=raw pyramid input, skip=upsampled path.
        combined = combine_bridge(raw_x, upsampled)
        conv_out, _ = self.conv(combined, raw_mask)
        # Networks.h:1088: `output.first = CombineBridge(features_input.first, output.first)`.
        out = combine_bridge(raw_x, conv_out)
        return out, raw_mask


class MultiScaleUnet2dDecOnlySmallFixed(nn.Module):
    """Port of MultiScaleUnet2dDecOnlySmallFixedImpl (Networks.h:1100-1208).

    Call as `net(inputs)` with `inputs` a list of `num_layers` tensors, each
    `(B, num_input_channels, h_i, w_i)`, **finest first**: `inputs[0]` is
    the full-resolution rasteriser output, `inputs[-1]` is the coarsest
    pyramid level (see module docstring).
    """

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        cfg = self.config
        f, c = cfg.filters, cfg.num_input_channels

        out_first_conv = f - 2 * c
        self.start = SmallDecStartBlock(c, out_first_conv, cfg.gated)

        # up[i] for i in [num_layers-2 .. 0], matching Networks.h's registration order
        # ("up1".."up(num_layers-1)"); index 0 is `last=True`.
        self.up = nn.ModuleList(
            [
                UpsampleDecOnlySmallBlockFixed(
                    f, c, last=(i == 0), gated_config=cfg.gated, upsample_mode=cfg.upsample_mode
                )
                for i in range(cfg.num_layers - 2, -1, -1)
            ]
        )
        # Index up[i] the same way the C++ array is indexed (up[0]..up[num_layers-2]) even
        # though it was built coarse-to-fine; store the mapping explicitly for forward().
        self._up_indices = list(range(cfg.num_layers - 2, -1, -1))

        self.final = nn.Sequential(
            nn.Conv2d(f, cfg.num_output_channels, kernel_size=1),
            activation_from_string(cfg.last_act),
        )

    def forward(
        self, inputs: list[torch.Tensor], masks: list[torch.Tensor | None] | None = None
    ) -> torch.Tensor:
        """Run the decoder-only U-Net.

        Args:
            inputs: `num_layers` tensors, finest first, each
                `(B, num_input_channels, h_i, w_i)`.
            masks: optional per-level validity masks, same order as
                `inputs`; TRIPS asserts `masks[i].requires_grad() ==
                False` (Networks.h:1168) but the `gated` conv block never
                reads them (see trippy.net.gated docstring), so they are
                only plumbed through for API parity. Defaults to `None`
                per level.

        Returns:
            `(B, num_output_channels, H_out, W_out)`. `H_out, W_out ==
            inputs[0]`'s spatial size when that size is divisible by
            `2 ** (num_layers - 1)`; otherwise centre-cropped, see module
            docstring "CombineBridge / odd-size handling".
        """
        n = self.config.num_layers
        if len(inputs) != n:
            raise ValueError(f"expected {n} pyramid levels (num_layers), got {len(inputs)}")
        masks = masks or [None] * n
        for level, x in enumerate(inputs):
            if x.shape[1] != self.config.num_input_channels:
                raise ValueError(
                    f"inputs[{level}] has {x.shape[1]} channels, expected "
                    f"num_input_channels={self.config.num_input_channels}"
                )

        state = self.start(inputs[n - 1], masks[n - 1])
        for block, i in zip(self.up, self._up_indices, strict=True):
            state = block(state, (inputs[i], masks[i]))

        x, _mask = state
        return self.final(x)

    def parameter_count(self) -> int:
        """Total learnable parameter count (sum of numel() over all parameters)."""
        return sum(p.numel() for p in self.parameters())
