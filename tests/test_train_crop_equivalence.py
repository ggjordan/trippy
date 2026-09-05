"""Proves the trainer's crop strategy: render-with-adjusted-K == crop-of-full-render.

Module: tests.test_train_crop_equivalence
Invariants under test: for the same points and pose, rendering a
    `crop x crop` image with `trippy.scene.dataset.crop`'s K-adjustment
    equals rendering the full frame and cropping the result with the same
    `crop()` call, to within 1e-5 (float64 compute) -- this is the "K-adjust"
    strategy `Trainer.train_step` uses so a training crop only ever
    rasterises the crop's own fragments, never the full frame's.

    EXCEPTION, and it is TRIPS's own behaviour, not a bug: in `mode="trips"`
    the equivalence holds only in the crop's *interior*. That mode carries
    TRIPS's `valid_point` gate, which drops a point from layer l unless all
    four footprint corners are inside layer l *of the image being rendered*
    (`RenderForward.cu:340-352`). A crop is a smaller image, so its own edge
    is a real edge: a point one pixel outside it is gone, where in the full
    frame it was interior and drew normally. The affected band is
    `2**l` layer-0 pixels wide at layer l. See docs/LIMITATIONS.md.
"""

from __future__ import annotations

import torch
from test_train_helpers import CX, CY, FX, FY, IMG_HEIGHT, IMG_WIDTH, camera_pose, synthetic_point_set

from trippy.raster.pyramid import render_pyramid
from trippy.scene.dataset import crop as dataset_crop


def _render_full(mode: str, num_layers: int = 3):
    ps = synthetic_point_set()
    xyz = torch.from_numpy(ps.xyz.astype("float64"))
    feat = torch.from_numpy(ps.rgb0.astype("float64"))
    size = torch.from_numpy(ps.size0.astype("float64"))
    conf = torch.from_numpy(ps.conf0.astype("float64"))
    K = torch.tensor([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=torch.float64)
    R, t = camera_pose(0)
    layers, _aux = render_pyramid(
        xyz, size, feat, conf, K, R, t, (IMG_HEIGHT, IMG_WIDTH),
        num_layers=num_layers, mode=mode, compute_dtype=torch.float64,
    )  # fmt: skip
    return xyz, size, feat, conf, K, R, t, layers


def _check_crop_matches(mode: str, center: tuple[float, float], crop_size: int = 16, num_layers: int = 3) -> None:
    xyz, size, feat, conf, K, R, t, full_layers = _render_full(mode, num_layers)
    full_img = full_layers[0]  # (C, H, W), finest layer

    item = {"rgb": full_img.permute(1, 2, 0), "K": K}
    cropped = dataset_crop(item, size=crop_size, zoom=1.0, center=center)
    assert cropped["mask"].min().item() == 1.0, "test centre must keep the crop fully in-bounds"
    cropped_from_full = cropped["rgb"].permute(2, 0, 1)

    crop_layers, _crop_aux = render_pyramid(
        xyz, size, feat, conf, cropped["K"], R, t, (crop_size, crop_size),
        num_layers=num_layers, mode=mode, compute_dtype=torch.float64,
    )  # fmt: skip
    direct_crop = crop_layers[0]

    max_diff = (cropped_from_full - direct_crop).abs().max().item()
    assert max_diff < 1e-5, f"crop-of-full vs direct-crop max diff {max_diff}"


def test_crop_equals_cropped_full_render_broadcast_mode() -> None:
    _check_crop_matches("broadcast", center=(20.0, 15.0))


def test_crop_equals_cropped_full_render_trilinear_mode() -> None:
    _check_crop_matches("trilinear", center=(20.0, 15.0))


def test_crop_equals_cropped_full_render_off_center() -> None:
    _check_crop_matches("broadcast", center=(8.0, 8.0))


def test_crop_equals_cropped_full_render_near_border_but_in_bounds() -> None:
    # crop_size=16 centred at (10, 10) spans [2, 18) x [2, 18) -- fully inside 48x36.
    _check_crop_matches("broadcast", center=(10.0, 10.0), crop_size=16)


def test_crop_equals_cropped_full_render_trips_mode_in_the_interior() -> None:
    """Mode "trips": exact inside, deliberately different on the crop's rim.

    The rim difference is TRIPS's `valid_point` gate seeing the crop's edge
    as an image edge (see the module docstring). Measured on this fixture it
    is ~0.45 in feature units on the outermost ring and exactly zero one
    pixel in -- i.e. it is a band, not a smeared error, which is what makes
    it safe to train on (TRIPS itself trains on crops with this rule).
    """
    xyz, size, feat, conf, K, R, t, full_layers = _render_full("trips", 3)
    item = {"rgb": full_layers[0].permute(1, 2, 0), "K": K}
    cropped = dataset_crop(item, size=16, zoom=1.0, center=(20.0, 15.0))
    cropped_from_full = cropped["rgb"].permute(2, 0, 1)
    crop_layers, _aux = render_pyramid(
        xyz, size, feat, conf, cropped["K"], R, t, (16, 16),
        num_layers=3, mode="trips", compute_dtype=torch.float64,
    )  # fmt: skip
    diff = (cropped_from_full - crop_layers[0]).abs()
    assert diff[:, 1:-1, 1:-1].max().item() < 1e-5, "the crop's interior must be exact"
    assert diff.max().item() > 1e-3, "the rim must differ -- if not, the gate is missing"
