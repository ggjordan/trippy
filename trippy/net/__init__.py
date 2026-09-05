"""Decoder-only U-Net, gated ELU convs, camera/tone-mapper model, losses.

Module: trippy.net
Invariants: every submodule here is a faithful (or explicitly documented,
    non-guessed) port of third_party/TRIPS/src/lib/models/{Networks,
    NeuralCamera}.{h,cpp} -- see docs/TRIPS_REFERENCE.md Sec. 5-7 and
    docs/LIMITATIONS.md for exactly which formulas are verified from
    source vs. deliberately substituted/generalized.
Related docs: docs/SPEC.md "Verified facts" (TRIPS default net:
    decoder-only U-Net, 5 levels, 32 filters, gated ELU convs -- the
    default config's exact parameter count is 59,675, hand-derived and
    tested in tests/test_net_unet.py; an earlier "~130k" estimate in this
    docstring was an unverified guess, corrected here) and "Technical
    design" (L1 + SSIM + LPIPS/VGG perceptual losses).
"""

from __future__ import annotations

from trippy.net.camera_model import NeuralCamera, NeuralCameraConfig
from trippy.net.checkpoint import CheckpointLoadResult, try_load_trips_network
from trippy.net.gated import GatedConvBlock, GatedConvConfig
from trippy.net.losses import LossWeights, TripsLoss, l1_loss, ssim
from trippy.net.unet import MultiScaleUnet2dDecOnlySmallFixed, NetworkConfig

__all__ = [
    "CheckpointLoadResult",
    "GatedConvBlock",
    "GatedConvConfig",
    "LossWeights",
    "MultiScaleUnet2dDecOnlySmallFixed",
    "NetworkConfig",
    "NeuralCamera",
    "NeuralCameraConfig",
    "TripsLoss",
    "l1_loss",
    "ssim",
    "try_load_trips_network",
]
