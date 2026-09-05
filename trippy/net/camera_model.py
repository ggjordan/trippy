"""NeuralCamera: per-image exposure/white-balance/vignette/response tone mapper.

Module: trippy.net.camera_model
Invariants: forward() consumes and returns (B, 3, H, W) tensors; the uv
    grid convention (image centre = (0, 0), corners at +-1) matches
    TRIPS's own `InitialUVImage` exactly (see module docstring below) so a
    loaded checkpoint's vignette_center is directly comparable.
Related docs: docs/TRIPS_REFERENCE.md Sec. 6 (neural camera / tone
    mapping, verified against source); docs/LIMITATIONS.md (rolling
    shutter not ported -- off by default in TRIPS and out of scope here).

Source: third_party/TRIPS/src/lib/models/NeuralCamera.{h,cpp} (TRIPS @
commit a59a65b6d9a8b1c14c73bc004cc9a8956f054c24):
    VignetteNetImpl                NeuralCamera.h:19-49,  .cpp:14-42
    CameraResponseNetImpl          NeuralCamera.h:98-118, .cpp:64-176
    NeuralCameraImpl                NeuralCamera.h:120-161, .cpp:214-426
    NeuralCameraParams defaults    src/lib/data/Settings.h:110-141

Order of operations in NeuralCameraImpl::forward (NeuralCamera.cpp:258-390), all defaults
per configs/train_normalnet.ini:190-198 (see trippy.constants for exact citations):
    1. exposure:       x = x * 2 ** -exposure[frame]                  (per-image scalar)
    2. white balance:   x = wb[frame] * x                             (per-image 3-vector, green fixed to 1)
    3. vignette:       x = (1 + p0*r2 + p1*r4 + p2*r6) * x            (r about a learnable centre)
    4. response curve: x = LUT(x) via grid_sample                     (or clamp(x,0,1) if disabled)
    5. rolling shutter: NOT PORTED (off by default, `enable_rolling_shutter=false`; a
       differentiable per-image 2-channel flow-field grid_sample warp -- out of scope for this
       task, see docs/LIMITATIONS.md).

`log_render` (NeuralCamera.cpp:263, a hardcoded local `false`) makes the `x = x - exposure`
branch permanently dead code in TRIPS itself; only the `x = x * 2**-exposure` branch is ported.

uv coordinate convention (`InitialUVImage`, third_party/TRIPS/src/lib/data/Dataset.cpp:11-30):
    texel = (pixel_xy) / (size_xy - 1)          # in [0, 1], corner-aligned
    uv = (texel - 0.5) * 2                      # in [-1, 1], image centre = (0, 0)
channel 0 = x (u, horizontal), channel 1 = y (v, vertical) -- VignetteNetImpl applies the
aspect-ratio correction only to channel 0 (NeuralCamera.cpp:33
`transformed_uv.slice(1, 0, 1) = transformed_uv.slice(1, 0, 1) * aspect`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from trippy.constants import (
    CAMERA_DEFAULT_ENABLE_EXPOSURE,
    CAMERA_DEFAULT_ENABLE_RESPONSE,
    CAMERA_DEFAULT_ENABLE_VIGNETTE,
    CAMERA_DEFAULT_ENABLE_WHITE_BALANCE,
    CAMERA_DEFAULT_RESPONSE_GAMMA,
    CAMERA_DEFAULT_RESPONSE_LEAK_FACTOR,
    CAMERA_DEFAULT_RESPONSE_PARAMS,
    CAMERA_RESPONSE_LEAK_SQRT_EPS,
    CAMERA_RESPONSE_SMOOTHNESS_INTERNAL_FACTOR,
    CAMERA_RESPONSE_SMOOTHNESS_OUTER_WEIGHT,
)


def default_uv_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build the (1, 2, H, W) uv grid TRIPS's InitialUVImage produces (see module docstring).

    Corner pixels map exactly to -1/+1 (align_corners=True-style spacing), matching
    Dataset.cpp:11-30's `texel = pixel/(size-1); uv = (texel-0.5)*2` construction.
    """
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)


class VignetteNet(nn.Module):
    """Port of VignetteNetImpl (NeuralCamera.h:19-49, .cpp:14-42)."""

    def __init__(self, image_height: int, image_width: int) -> None:
        super().__init__()
        # NeuralCamera.cpp:27 `float aspect = image_size.x() / image_size.y()` -- x is width.
        self.aspect = image_width / image_height
        self.vignette_params = nn.Parameter(torch.zeros(3))
        self.vignette_center = nn.Parameter(torch.zeros(1, 2, 1, 1))

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        """uv: (B, 2, H, W) -> factor: (B, 1, H, W), applied multiplicatively to rgb."""
        assert uv.dim() == 4
        transformed = uv - self.vignette_center
        # Only the x/u channel gets the aspect correction (NeuralCamera.cpp:33).
        transformed = torch.cat([transformed[:, :1] * self.aspect, transformed[:, 1:]], dim=1)
        transformed = transformed * transformed
        r2 = transformed.sum(dim=1, keepdim=True)
        r4 = r2 * r2
        r6 = r4 * r2
        p0, p1, p2 = self.vignette_params[0], self.vignette_params[1], self.vignette_params[2]
        return 1.0 + p0 * r2 + p1 * r4 + p2 * r6


def _make_gamma_response(num_params: int, gamma: float) -> torch.Tensor:
    """Port of Saiga::DiscreteResponseFunction::MakeGamma + normalize(1) (HDR.h:81-103).

    irradiance[0] = 0, irradiance[-1] = 1 (set exactly, not through the pow formula),
    irradiance[i] = (i / (n-1)) ** gamma for interior i. `normalize(1)` divides every entry
    by `irradiance.back()`, which MakeGamma already forced to exactly 1 -- a no-op here, kept
    only as a comment for fidelity (Saiga always calls both in sequence,
    NeuralCamera.cpp:66-69: `crf.MakeGamma(initial_gamma); crf.normalize(1);`).

    Returns: (1, 1, 1, num_params) float32 tensor, shaped to be repeated across channels by
    the caller (matches `response.repeat({1, num_channels, 1, 1})`, NeuralCamera.cpp:78).
    """
    assert num_params > 1
    alpha = torch.linspace(0.0, 1.0, num_params, dtype=torch.float32)
    irradiance = alpha.pow(gamma)
    irradiance[0] = 0.0
    irradiance[-1] = 1.0
    return irradiance.view(1, 1, 1, num_params)


class CameraResponseNet(nn.Module):
    """Port of CameraResponseNetImpl (NeuralCamera.h:98-118, .cpp:64-176).

    A per-channel 1D LUT (`response_params` control points), applied via bilinear
    grid_sample with align_corners=True + border padding, plus an optional "leaky"
    linear/1-over-sqrt extrapolation outside [0, 1] active only in training mode.
    """

    def __init__(self, num_params: int, num_channels: int, initial_gamma: float, leak_factor: float) -> None:
        super().__init__()
        response = _make_gamma_response(num_params, initial_gamma).repeat(1, num_channels, 1, 1)
        self.response = nn.Parameter(response)
        self.leak_factor = leak_factor

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        assert image.dim() == 4
        assert image.shape[1] == self.response.shape[1]
        num_batches, num_channels = image.shape[0], image.shape[1]

        leak_add = None
        if self.training and self.leak_factor > 0:
            clamp_low = image < 0
            clamp_high = image > 1
            # NeuralCamera.cpp:101 (below-0 leak: linear).
            leak_add = (image * self.leak_factor) * clamp_low
            # NeuralCamera.cpp:104 (above-1 leak: 1/sqrt taper back to `leak_factor`).
            leak_add = (
                leak_add
                + (
                    -self.leak_factor / torch.sqrt(image.abs() + CAMERA_RESPONSE_LEAK_SQRT_EPS)
                    + self.leak_factor
                )
                * clamp_high
            )

        batched_response = self.response.repeat(num_batches, 1, 1, 1)
        # Grid-sample uv space is [-1, 1] (NeuralCamera.cpp:114).
        scaled = image * 2.0 - 1.0
        y_offset = torch.zeros_like(scaled)
        # (B, C, H, W, 2): sample coordinate (x, y=0) per Saiga's 1D-as-2D grid_sample trick.
        grid = torch.stack([scaled, y_offset], dim=-1)

        result = torch.ones_like(image)
        for c in range(num_channels):
            channel_grid = grid[:, c]  # (B, H, W, 2)
            channel_response = batched_response[:, c : c + 1]  # (B, 1, 1, n)
            result[:, c : c + 1] = functional.grid_sample(
                channel_response, channel_grid, mode="bilinear", padding_mode="border", align_corners=True
            )

        if leak_add is not None:
            result = result + leak_add
        return result

    def param_loss(self) -> torch.Tensor:
        """Port of CameraResponseNetImpl::ParamLoss (NeuralCamera.cpp:137-157).

        A smoothness regularizer: pulls each interior control point toward the mean of its
        two neighbours, and the first point toward 0 (the last point is left unconstrained --
        the C++ source has a commented-out `target.slice(..., n-1, n).fill_(1)`, i.e. that term
        was deliberately dropped, not merely unimplemented; ported as dropped here too).
        """
        n = self.response.shape[-1]
        factor = n * math.sqrt(CAMERA_RESPONSE_SMOOTHNESS_INTERNAL_FACTOR)
        low = self.response[..., : n - 2]
        high = self.response[..., 2:]
        target = self.response.clone()
        target[..., 0:1] = 0.0
        target[..., 1 : n - 1] = (low + high) * 0.5
        diff = self.response * factor - target * factor
        return (diff * diff).sum()


@dataclass
class NeuralCameraConfig:
    """Config for NeuralCamera. Every default cites configs/train_normalnet.ini (TRIPS @
    a59a65b6d9a8b1c14c73bc004cc9a8956f054c24); see trippy.constants for exact line numbers.

    Attributes:
        enable_exposure: per-image learned exposure (ini:191).
        enable_white_balance: per-image learned white balance, green
            channel held at its reference value (ini:193).
        enable_vignette: radial vignette factor; module is always present
            when True but its parameters init to 0, so the effect is a
            no-op until trained (ini:190).
        enable_response: learned per-channel response-curve LUT; falls
            back to `clamp(x, 0, 1)` when False (ini:192).
        response_params: number of LUT control points (ini:196).
        response_gamma: initial gamma for the LUT (ini:197).
        response_leak_factor: leaky-extrapolation slope outside [0, 1],
            training-mode only (ini:198).
    """

    enable_exposure: bool = CAMERA_DEFAULT_ENABLE_EXPOSURE
    enable_white_balance: bool = CAMERA_DEFAULT_ENABLE_WHITE_BALANCE
    enable_vignette: bool = CAMERA_DEFAULT_ENABLE_VIGNETTE
    enable_response: bool = CAMERA_DEFAULT_ENABLE_RESPONSE
    response_params: int = CAMERA_DEFAULT_RESPONSE_PARAMS
    response_gamma: float = CAMERA_DEFAULT_RESPONSE_GAMMA
    response_leak_factor: float = CAMERA_DEFAULT_RESPONSE_LEAK_FACTOR


class NeuralCamera(nn.Module):
    """Port of NeuralCameraImpl (NeuralCamera.h:120-161, .cpp:214-426).

    Rolling shutter is intentionally not ported (off by default in TRIPS, out of scope --
    see docs/LIMITATIONS.md).
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        num_frames: int,
        config: NeuralCameraConfig | None = None,
        initial_exposure: torch.Tensor | None = None,
        initial_white_balance: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config = config or NeuralCameraConfig()
        self.image_height = image_height
        self.image_width = image_width

        if config.enable_response:
            self.camera_response = CameraResponseNet(
                config.response_params, 3, config.response_gamma, config.response_leak_factor
            )
        else:
            self.camera_response = None

        if config.enable_vignette:
            self.vignette_net = VignetteNet(image_height, image_width)
        else:
            self.vignette_net = None

        if config.enable_exposure:
            init = initial_exposure if initial_exposure is not None else torch.zeros(num_frames)
            assert init.shape == (num_frames,)
            self.exposures_values = nn.Parameter(init.clone().view(num_frames, 1, 1, 1).float())
        else:
            self.exposures_values = None

        if config.enable_white_balance:
            init_wb = (
                initial_white_balance if initial_white_balance is not None else torch.ones(num_frames, 3)
            )
            assert init_wb.shape == (num_frames, 3)
            assert torch.allclose(init_wb[:, 1], torch.ones(num_frames)), "green channel WB must init to 1"
            self.white_balance_values = nn.Parameter(init_wb.clone().view(num_frames, 3, 1, 1).float())
            self.register_buffer("white_balance_reference", init_wb.clone().view(num_frames, 3, 1, 1).float())
        else:
            self.white_balance_values = None

    def forward(
        self, x: torch.Tensor, frame_index: torch.Tensor, uv: torch.Tensor | None = None
    ) -> torch.Tensor:
        """x: (B, 3, H, W) linear-ish rgb -> (B, 3, H, W) display rgb.

        frame_index: (B,) long tensor indexing into the per-image exposure/white-balance
        parameters. uv: (B, 2, H, W) or None (built from `default_uv_grid`, broadcast over
        the batch); only read when `enable_vignette`.
        """
        assert x.shape[1] == 3
        if self.exposures_values is not None:
            exposure = self.exposures_values[frame_index]
            x = x * torch.exp2(-exposure)

        if self.white_balance_values is not None:
            wb = self.white_balance_values[frame_index]
            x = wb * x

        if self.vignette_net is not None:
            if uv is None:
                uv = default_uv_grid(x.shape[2], x.shape[3], x.device, x.dtype).expand(x.shape[0], -1, -1, -1)
            x = self.vignette_net(uv) * x

        if self.camera_response is not None:
            x = self.camera_response(x)
        else:
            x = torch.clamp(x, 0.0, 1.0)

        return x

    def apply_constraints(self) -> None:
        """Port of NeuralCameraImpl::ApplyConstraints (NeuralCamera.cpp:407-426).

        Only white balance is constrained: image 0's WB is pinned to its reference value, and
        the green channel is pinned to its reference for every image. (The equivalent
        "pin exposure[0]" line, NeuralCamera.cpp:416, is commented out in TRIPS itself -- dead
        code, not ported.) Called after each optimizer step during training, outside autograd.
        """
        if self.white_balance_values is None:
            return
        with torch.no_grad():
            self.white_balance_values[0:1] = self.white_balance_reference[0:1]
            self.white_balance_values[:, 1:2] = self.white_balance_reference[:, 1:2]

    def regularizer(self) -> torch.Tensor:
        """Smoothness regularizer on the response LUT.

        Port of `params->optimizer_params.response_smoothness * camera->ParamLoss()`
        (Pipeline.cpp:804) folded into a single call: trippy's outer weight constant
        (CAMERA_RESPONSE_SMOOTHNESS_OUTER_WEIGHT, = ini's `response_smoothness = 1`) times
        CameraResponseNetImpl::ParamLoss(). Zero when response curve learning is disabled.
        """
        if self.camera_response is None:
            device = (
                self.exposures_values.device if self.exposures_values is not None else torch.device("cpu")
            )
            return torch.zeros((), device=device)
        return CAMERA_RESPONSE_SMOOTHNESS_OUTER_WEIGHT * self.camera_response.param_loss()
