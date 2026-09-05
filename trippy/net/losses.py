"""L1, SSIM, VGG/LPIPS perceptual losses, and the combined TripsLoss.

Module: trippy.net.losses
Invariants: every loss function accepts (B, 3, H, W) float tensors in
    [0, 1] and an optional validity mask broadcastable to that shape
    (1 = valid, 0 = excluded); no loss mutates its inputs in place.
Related docs: docs/TRIPS_REFERENCE.md Sec. 7 (losses/schedule);
    docs/LIMITATIONS.md (VGG-vs-lpips substitution, mask handling is a
    trippy addition not present in TRIPS's own loss code).

Loss combination formula, verified at src/lib/models/Pipeline.cpp:700-780 (TRIPS @
a59a65b6d9a8b1c14c73bc004cc9a8956f054c24):
    loss = w_vgg * VGG(x, target)
         + w_l1 * L1(x, target)
         + w_mse * MSE(x, target)
         + w_ssim * (1 - SSIM(x, target)) / 2
         + w_lpips * LPIPS(x, target).sum()
Default weights, configs/train_normalnet.ini:40-42,62-63 (`[TrainParams]`):
    loss_vgg=1, loss_l1=1, loss_mse=0, loss_ssim=1, loss_lpips=0.
`w_mse` is accepted for parity but MSE is not otherwise used by TRIPS's default config
(weight 0); trippy's TripsLoss still computes plain `torch.nn.functional.mse_loss` for it
so the weight is honoured if a caller changes it.

SSIM: exact port of Saiga::SSIMImpl (see trippy.constants for the fetch provenance:
https://github.com/darglein/saiga @ 5fb87057f09f518b1ecf7de1a486420681455892,
src/saiga/vision/torch/ImageSimilarity.h:73-126), instantiated by TRIPS with its own
defaults (`SSIM loss_ssim = SSIM();`, Pipeline.h:238) -- radius=2 (5x5 Gaussian window,
*not* the generic 11x11 Wang et al. window), sigma=1.5, max_value=1. A depthwise
(`groups=channels`) 2D convolution computes the local mean/variance/covariance per channel;
the per-pixel-per-channel SSIM map is averaged over every element to produce the scalar.

VGG / perceptual loss: TRIPS's actual `loss_vgg` is `Saiga::PretrainedVGG19Loss`, a custom
Caffe-derived VGG19 loaded from a pre-traced TorchScript file (`loss/traced_caffe_vgg_optim.pt`)
that is not present in this checkout and not human-readable even if it were (a binary traced
graph). Per the task brief, trippy substitutes `lpips.LPIPS(net='vgg')` (ImageNet-pretrained
torchvision VGG16 + LPIPS's own linear calibration layers) as a stand-in perceptual loss for
the `vgg` weight -- **not a bit-exact port**, see docs/LIMITATIONS.md. TRIPS's separate,
always-off-by-default `loss_lpips` term (`loss/traced_lpips.pt`) is, by contrast, verifiably
`lpips.LPIPS(net='alex')` -- Saiga's own `LPIPS` class docstring (ImageSimilarity.h:192-198)
gives the exact Python tracing recipe: `lpips.LPIPS(net='alex')`. trippy uses that network for
the `lpips` weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from trippy.constants import (
    LOSS_DEFAULT_WEIGHT_L1,
    LOSS_DEFAULT_WEIGHT_LPIPS,
    LOSS_DEFAULT_WEIGHT_MSE,
    LOSS_DEFAULT_WEIGHT_SSIM,
    LOSS_DEFAULT_WEIGHT_VGG,
    SSIM_C1_COEFF,
    SSIM_C2_COEFF,
    SSIM_GAUSSIAN_RADIUS,
    SSIM_GAUSSIAN_SIGMA,
    SSIM_MAX_VALUE,
)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean of x over valid positions only (mask broadcastable to x's shape, 1=valid)."""
    if mask is None:
        return x.mean()
    mask = mask.to(dtype=x.dtype, device=x.device).expand_as(x)
    total = mask.sum()
    if total <= 0:
        return x.new_zeros(())
    return (x * mask).sum() / total


def l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean absolute error, optionally restricted to `mask` (1=valid)."""
    return _masked_mean((pred - target).abs(), mask)


def _gaussian_kernel_2d(radius: int, sigma: float) -> torch.Tensor:
    """Port of Saiga's gaussianBlurKernel2d_tinyeigen (ImageSimilarity.h:13-30).

    kernel(y, x) = exp(-(x^2 + y^2) / (2*sigma^2)), normalized to sum to 1.
    Returns (1, 1, 2*radius+1, 2*radius+1).
    """
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    inv_var2 = 1.0 / (2.0 * sigma * sigma)
    kernel = torch.exp(-(xx * xx + yy * yy) * inv_var2)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel.shape[0], kernel.shape[1])


def ssim_map(
    img1: torch.Tensor,
    img2: torch.Tensor,
    radius: int = SSIM_GAUSSIAN_RADIUS,
    sigma: float = SSIM_GAUSSIAN_SIGMA,
    max_value: float = SSIM_MAX_VALUE,
) -> torch.Tensor:
    """Per-pixel, per-channel SSIM map. Port of Saiga::SSIMImpl::get_ssim_map (ImageSimilarity.h:97-118)."""
    assert img1.dim() == 4 and img2.dim() == 4
    channels = img1.shape[1]
    kernel = _gaussian_kernel_2d(radius, sigma).to(device=img1.device, dtype=img1.dtype)
    kernel = kernel.repeat(channels, 1, 1, 1)  # depthwise: one copy of the same kernel per channel

    def blur(x: torch.Tensor) -> torch.Tensor:
        return functional.conv2d(x, kernel, padding=radius, groups=channels)

    mu1, mu2 = blur(img1), blur(img2)
    mu11, mu22, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma11 = blur(img1 * img1) - mu11
    sigma22 = blur(img2 * img2) - mu22
    sigma12 = blur(img1 * img2) - mu12

    c1 = (SSIM_C1_COEFF * max_value) ** 2
    c2 = (SSIM_C2_COEFF * max_value) ** 2
    return ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu11 + mu22 + c1) * (sigma11 + sigma22 + c2))


def ssim(img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Scalar SSIM (1 = identical). `(1 - ssim(...)) / 2` matches TRIPS's loss term.

    `mask` restricts the mean to valid positions -- a trippy addition; Saiga's own `SSIM`
    always averages over the whole map (TRIPS instead excludes borders via a training-time
    crop, `train_mask_border=16`, configs/train_normalnet.ini, not via a per-pixel mask).
    Caveat: the mask is applied *after* the Gaussian blur, so it does not stop the blur's
    radius-`SSIM_GAUSSIAN_RADIUS` receptive field from mixing invalid-region pixel values
    into the ssim_map value at a nearby *valid* pixel. A validity mask that should fully
    isolate SSIM from a corrupted region must grow that region by `SSIM_GAUSSIAN_RADIUS`
    pixels on every side (see tests/test_net_losses.py for a worked example).
    """
    return _masked_mean(ssim_map(img1, img2), mask)


class _LazyLPIPS(nn.Module):
    """Lazily-constructed lpips.LPIPS network, moved to the input's device on first use.

    Constructing `lpips.LPIPS` downloads/loads a torchvision backbone; deferring that to
    first `forward()` call keeps `import trippy.net.losses` (and TripsLoss construction with
    a zero weight for this term) cheap and offline-safe.
    """

    def __init__(self, net: str) -> None:
        super().__init__()
        self.net = net
        self._model: nn.Module | None = None

    def _model_on(self, device: torch.device) -> nn.Module:
        if self._model is None:
            import lpips  # deferred: heavy import + first-use backbone load.

            model = lpips.LPIPS(net=self.net)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self._model = model
        self._model = self._model.to(device)
        return self._model

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.to(dtype=pred.dtype, device=pred.device).expand_as(pred)
            pred = pred * mask
            target = target * mask
        model = self._model_on(pred.device)
        # lpips expects inputs in [-1, 1] (ImageSimilarity.h:220-221 does the same rescale).
        return model(pred * 2 - 1, target * 2 - 1).mean()


@dataclass
class LossWeights:
    """Loss term weights. Defaults from configs/train_normalnet.ini:40-42,62-63."""

    vgg: float = LOSS_DEFAULT_WEIGHT_VGG
    l1: float = LOSS_DEFAULT_WEIGHT_L1
    mse: float = LOSS_DEFAULT_WEIGHT_MSE
    ssim: float = LOSS_DEFAULT_WEIGHT_SSIM
    lpips: float = LOSS_DEFAULT_WEIGHT_LPIPS


class TripsLoss(nn.Module):
    """Combined training loss: w_vgg*VGG + w_l1*L1 + w_mse*MSE + w_ssim*(1-SSIM)/2 + w_lpips*LPIPS.

    See module docstring for the exact Pipeline.cpp formula and the VGG/lpips substitution.
    """

    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        # net='vgg' approximates TRIPS's PretrainedVGG19Loss (see module docstring);
        # net='alex' is Saiga's own verified choice for its separate LPIPS loss term.
        self._vgg = _LazyLPIPS(net="vgg")
        self._lpips = _LazyLPIPS(net="alex")

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert pred.shape == target.shape and pred.shape[1] == 3
        w = self.weights
        total = pred.new_zeros(())

        if w.l1 != 0:
            total = total + w.l1 * l1_loss(pred, target, mask)
        if w.mse != 0:
            diff2 = (pred - target) ** 2
            total = total + w.mse * _masked_mean(diff2, mask)
        if w.ssim != 0:
            total = total + w.ssim * (1.0 - ssim(pred, target, mask)) / 2.0
        if w.vgg != 0:
            total = total + w.vgg * self._vgg(pred, target, mask)
        if w.lpips != 0:
            total = total + w.lpips * self._lpips(pred, target, mask)
        return total
