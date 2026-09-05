"""Tests for trippy.net.camera_model (NeuralCamera).

Module: tests.test_net_camera
Invariants under test: at init (exposure=0, white_balance=1, vignette
    params=0), the only non-identity operation is the response-curve
    gamma mapping, so `forward(x)` on a constant image equals
    `x ** (1/2.2)` per-pixel; vignette/exposure/white-balance are all
    verified as pure identities at their init values.
"""

from __future__ import annotations

import torch

from trippy.net.camera_model import (
    CameraResponseNet,
    NeuralCamera,
    NeuralCameraConfig,
    VignetteNet,
    default_uv_grid,
)


def test_vignette_is_identity_factor_at_init() -> None:
    net = VignetteNet(image_height=8, image_width=16)
    uv = default_uv_grid(height=8, width=16, device=torch.device("cpu"), dtype=torch.float32)
    factor = net(uv)
    torch.testing.assert_close(factor, torch.ones_like(factor))


def test_response_curve_maps_half_to_gamma_curve() -> None:
    response = CameraResponseNet(num_params=25, num_channels=3, initial_gamma=1.0 / 2.2, leak_factor=0.01)
    response.eval()
    x = torch.full((1, 3, 2, 2), 0.5)
    out = response(x)
    expected = 0.5 ** (1.0 / 2.2)
    torch.testing.assert_close(out, torch.full_like(out, expected), atol=1e-4, rtol=1e-4)


def test_response_curve_endpoints_are_exact() -> None:
    """MakeGamma forces irradiance[0]=0 and irradiance[-1]=1 exactly (HDR.h:95-96)."""
    response = CameraResponseNet(num_params=25, num_channels=3, initial_gamma=1.0 / 2.2, leak_factor=0.01)
    response.eval()
    zeros = torch.zeros(1, 3, 1, 1)
    ones = torch.ones(1, 3, 1, 1)
    torch.testing.assert_close(response(zeros), torch.zeros_like(zeros), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(response(ones), torch.ones_like(ones), atol=1e-5, rtol=1e-5)


def test_neural_camera_identity_except_gamma() -> None:
    """At init (exposure=0, wb=1, vignette params=0), forward == the response-curve gamma
    mapping applied per pixel -- exposure/wb/vignette are all no-ops at their init values."""
    cam = NeuralCamera(image_height=8, image_width=8, num_frames=3)
    cam.eval()
    x = torch.full((2, 3, 8, 8), 0.5)
    frame_index = torch.tensor([0, 2])
    out = cam(x, frame_index)
    expected = 0.5 ** (1.0 / 2.2)
    torch.testing.assert_close(out, torch.full_like(out, expected), atol=1e-4, rtol=1e-4)


def test_neural_camera_response_disabled_falls_back_to_clamp() -> None:
    cfg = NeuralCameraConfig(enable_response=False, enable_vignette=False)
    cam = NeuralCamera(image_height=4, image_width=4, num_frames=1, config=cfg)
    cam.eval()
    x = torch.tensor([[-0.5, 0.0, 0.5, 1.5]]).view(1, 1, 1, 4).expand(1, 3, 1, 4).clone()
    out = cam(x, torch.tensor([0]))
    torch.testing.assert_close(out, torch.clamp(x, 0.0, 1.0))


def test_regularizer_is_finite_and_nonnegative() -> None:
    cam = NeuralCamera(image_height=4, image_width=4, num_frames=1)
    reg = cam.regularizer()
    assert torch.isfinite(reg)
    assert reg.item() >= 0.0


def test_apply_constraints_pins_green_channel_and_first_image() -> None:
    cam = NeuralCamera(image_height=4, image_width=4, num_frames=2)
    with torch.no_grad():
        cam.white_balance_values.add_(0.3)  # perturb every image's wb, including green
    cam.apply_constraints()
    wb = cam.white_balance_values.detach()
    ref = cam.white_balance_reference
    torch.testing.assert_close(wb[0:1], ref[0:1])
    torch.testing.assert_close(wb[:, 1:2], ref[:, 1:2])
    # Non-green channel of image 1 (not pinned) should still show the perturbation.
    assert not torch.allclose(wb[1:2, 0:1], ref[1:2, 0:1])
