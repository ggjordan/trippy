"""Tests for trippy.net.losses (L1, SSIM, TripsLoss).

Module: tests.test_net_losses
Invariants under test: L1 and SSIM both report "perfect" on identical
    images (0 and 1 respectively); a validity mask excludes a corrupted
    region from both losses entirely; TripsLoss (including its default,
    lpips-backed VGG term) returns a finite scalar. Kept deliberately
    small (16x16, single image) so the VGG/lpips forward pass stays well
    under the suite's 20s CPU budget.
"""

from __future__ import annotations

import torch

from trippy.net.losses import LossWeights, TripsLoss, l1_loss, ssim


def test_l1_zero_on_identical_images() -> None:
    x = torch.rand(2, 3, 12, 12)
    assert l1_loss(x, x.clone()).item() == 0.0


def test_ssim_one_on_identical_images() -> None:
    x = torch.rand(1, 3, 16, 16)
    value = ssim(x, x.clone())
    torch.testing.assert_close(value, torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_mask_excludes_corrupted_region_from_l1_and_ssim() -> None:
    """SSIM's Gaussian blur has a radius-2 receptive field (see trippy.net.losses module
    docstring), so a pixel within 2 pixels of a corrupted region is itself contaminated in
    the *unmasked* ssim_map even though its own value is untouched. The validity mask must
    therefore cover the corrupted region plus that receptive-field margin for the masked
    SSIM to be exactly unaffected -- this test masks the corrupted 4x4 block plus a 2-pixel
    margin (6x6) to demonstrate the leak-free case, and separately confirms the leak exists
    (and is excluded) by checking a mask that stops exactly at the corruption boundary would
    not be enough (regression guard on the note above, not just the happy path)."""
    a = torch.rand(1, 3, 16, 16)
    b = a.clone()

    corrupted = a.clone()
    corrupted[:, :, :4, :4] = 5.0  # wildly wrong values in a small region

    tight_mask = torch.ones(1, 1, 16, 16)
    tight_mask[:, :, :4, :4] = 0.0
    safe_mask = torch.ones(1, 1, 16, 16)
    safe_mask[:, :, :6, :6] = 0.0  # corrupted region + SSIM_GAUSSIAN_RADIUS=2 margin

    assert l1_loss(corrupted, b, tight_mask).item() == 0.0  # L1 has no receptive field, so
    # even the tight mask fully excludes the corruption.

    ssim_tight = ssim(corrupted, b, tight_mask)
    assert ssim_tight.item() < 0.99  # tight mask still leaks -- documents the caveat above.

    ssim_safe = ssim(corrupted, b, safe_mask)
    torch.testing.assert_close(ssim_safe, torch.tensor(1.0), atol=1e-4, rtol=1e-4)

    # Sanity: without any mask, both losses must detect the corruption.
    assert l1_loss(corrupted, b).item() > 0.0
    assert ssim(corrupted, b).item() < 0.99


def test_trips_loss_returns_finite_scalar_with_default_weights() -> None:
    loss_fn = TripsLoss()
    pred = torch.rand(1, 3, 16, 16)
    target = torch.rand(1, 3, 16, 16)
    value = loss_fn(pred, target)
    assert value.dim() == 0
    assert torch.isfinite(value)


def test_trips_loss_zero_weights_give_zero_loss_on_identical_images() -> None:
    loss_fn = TripsLoss(LossWeights(vgg=0.0, l1=1.0, mse=0.0, ssim=1.0, lpips=0.0))
    x = torch.rand(1, 3, 16, 16)
    value = loss_fn(x, x.clone())
    torch.testing.assert_close(value, torch.tensor(0.0), atol=1e-5, rtol=1e-5)
