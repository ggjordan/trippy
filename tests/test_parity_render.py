"""Unit tests for trippy.render.parity's geometry, layer selection and metrics.

Everything here is synthetic (AGENTS.md Sec. 6) and CPU-only. The point of
these tests is the *conventions*: TRIPS's distortion polynomial, its
pixel-centre-at-integer footprint, its ceil-halved pyramid and its
`layers 0..layer_higher` emission rule. Each one is checked against a hand
computation, not against another implementation of the same idea.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from trippy.constants import ADOP_DIST_CUTOFF, ADOP_DISTORTION_SENTINEL, PARITY_EVAL_BORDER_PX
from trippy.render import parity
from trippy.scene.adop_io import AdopView

# --- distortion ----------------------------------------------------------


def test_distort_normalized_is_identity_with_zero_coefficients():
    xy = torch.tensor([[0.0, 0.0], [0.3, -0.4], [1.0, 1.0]])
    out = parity.distort_normalized(xy, torch.zeros(8))
    assert torch.allclose(out, xy, atol=1e-7)


def test_distort_normalized_hand_computed_radial():
    """`xd = x * (1 + k1 r2 + k2 r4) / (1 + k4 r2 + ...)`, Distortion.h:148-164."""
    k1, k2 = -0.06404954, 0.04441944  # tt_horse camera0.ini
    dist = torch.tensor([k1, k2, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
    x, y = 0.6, -0.45
    r2 = x * x + y * y
    expected_radial = 1.0 + k1 * r2 + k2 * r2 * r2
    out = parity.distort_normalized(torch.tensor([[x, y]], dtype=torch.float64), dist)
    assert out[0, 0].item() == pytest.approx(x * expected_radial, rel=1e-12)
    assert out[0, 1].item() == pytest.approx(y * expected_radial, rel=1e-12)


def test_distort_normalized_tangential_terms():
    p1, p2 = 0.003, -0.002
    dist = torch.tensor([0, 0, 0, 0, 0, 0, p1, p2], dtype=torch.float64)
    x, y = 0.2, 0.1
    r2 = x * x + y * y
    out = parity.distort_normalized(torch.tensor([[x, y]], dtype=torch.float64), dist)
    assert out[0, 0].item() == pytest.approx(x + p1 * 2 * x * y + p2 * (r2 + 2 * x * x), rel=1e-12)
    assert out[0, 1].item() == pytest.approx(y + p1 * (r2 + 2 * y * y) + p2 * 2 * x * y, rel=1e-12)


def test_distort_normalized_sentinel_beyond_dist_cutoff():
    xy = torch.tensor([[ADOP_DIST_CUTOFF + 1.0, 0.0], [0.1, 0.1]])
    out = parity.distort_normalized(xy, torch.zeros(8))
    assert out[0, 0].item() == ADOP_DISTORTION_SENTINEL
    assert out[1, 0].item() == pytest.approx(0.1)


# --- projection ----------------------------------------------------------


def _K(fx=100.0, fy=120.0, cx=32.0, cy=16.0):
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float64)


def test_project_adop_optical_axis_and_offsets():
    K = _K()
    R = torch.eye(3, dtype=torch.float64)
    t = torch.zeros(3, dtype=torch.float64)
    xyz = torch.tensor([[0.0, 0.0, 5.0], [0.5, 0.25, 5.0], [0.0, 0.0, -2.0]], dtype=torch.float64)
    _ndc, ip, z = parity.project_adop(xyz, R, t, K, torch.zeros(8, dtype=torch.float64))
    assert torch.allclose(ip[0], torch.tensor([32.0, 16.0], dtype=torch.float64))
    # x/z = 0.1, y/z = 0.05 -> ip = (100*0.1 + 32, 120*0.05 + 16)
    assert torch.allclose(ip[1], torch.tensor([42.0, 22.0], dtype=torch.float64))
    assert z[2].item() == -2.0


def test_project_adop_respects_world_to_camera_pose():
    K = _K(fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    # 90-degree yaw: world (0, 0, 1) -> camera (0, 0, 1) after translation.
    R = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    t = torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64)
    xyz = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    _ndc, ip, z = parity.project_adop(xyz, R, t, K, torch.zeros(8, dtype=torch.float64))
    # camera-space = R @ (1,0,0) + t = (0, 1, 3)
    assert z[0].item() == pytest.approx(3.0)
    assert ip[0, 0].item() == pytest.approx(0.0)
    assert ip[0, 1].item() == pytest.approx(1.0 / 3.0)


# --- pyramid geometry ----------------------------------------------------


def test_trips_layer_shapes_ceil_halving():
    """PointRenderer.cu:385-391 uses std::ceil for every non-"MultiScaleUnet2d" net."""
    assert parity.trips_layer_shapes(1080, 1920, 8) == [
        (1080, 1920),
        (540, 960),
        (270, 480),
        (135, 240),
        (68, 120),
        (34, 60),
        (17, 30),
        (9, 15),
    ]


def test_trips_layer_shapes_matches_raster_layer_grid():
    from trippy.raster.emit import layer_grid

    assert parity.trips_layer_shapes(1080, 1920, 8) == layer_grid(1080, 1920, 8).shapes


# --- fragment emission / hand-checked composite --------------------------


def _view(height=16, width=24, fx=100.0, fy=100.0, cx=12.0, cy=8.0) -> AdopView:
    return AdopView(
        index=0,
        image_name="synthetic.jpg",
        image_path=None,  # type: ignore[arg-type]
        mask_path=None,
        camera_index=0,
        K=np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64),
        R=np.eye(3),
        t=np.zeros(3),
        height=height,
        width=width,
        distortion=np.zeros(8),
        exposure=0.0,
        white_balance=np.ones(3),
    )


def _points_at(ip_x: float, ip_y: float, depth: float, view: AdopView, feat, conf, size, bg):
    """One point whose projection lands exactly on `(ip_x, ip_y)`."""
    fx, fy = view.K[0, 0], view.K[1, 1]
    cx, cy = view.K[0, 2], view.K[1, 2]
    xyz = torch.tensor(
        [[(ip_x - cx) / fx * depth, (ip_y - cy) / fy * depth, depth]], dtype=torch.float32
    )
    return parity.ScenePoints(
        xyz=xyz,
        size=torch.tensor([size], dtype=torch.float32),
        feat=torch.tensor([feat], dtype=torch.float32),
        conf=torch.tensor([conf], dtype=torch.float32),
        bg=torch.tensor(bg, dtype=torch.float32),
    )


def test_level0_composite_bilinear_weights_hand_checked():
    """Bisect step (a) of the brief: one point, one pixel, computed by hand.

    TRIPS's 2x2 footprint sits at `floor(ip)` and `floor(ip)+1` with weights
    `(1-fx)(1-fy), fx(1-fy), (1-fx)fy, fx fy` where `f = ip - floor(ip)`
    (`PointBlending.h:216-240`). With a single point of confidence `c` and a
    single feature channel `v`, pixel `(gy, gx)` composites to
    `w * c * v + (1 - w * c) * bg`.
    """
    view = _view()
    ip_x, ip_y = 10.25, 6.75
    conf, value, bg = 0.5, 3.0, 0.125
    # size 0 => size_px 0 => layer_higher 0 => only layer 0 is written, and
    # compute_point_size_fac takes the sub-pixel branch: 0.75*exp(0-1)+0.25.
    points = _points_at(ip_x, ip_y, 4.0, view, [value], conf, 0.0, [bg])
    layers, _aux = parity.render_trips_layers(points, view, num_layers=1, mode="trips")

    layer_fac = 0.75 * math.exp(0.0 - 1.0) + 0.25
    fx_frac, fy_frac = ip_x - 10.0, ip_y - 6.0
    expected_weights = {
        (6, 10): (1 - fx_frac) * (1 - fy_frac),
        (6, 11): fx_frac * (1 - fy_frac),
        (7, 10): (1 - fx_frac) * fy_frac,
        (7, 11): fx_frac * fy_frac,
    }
    out = layers[0][0]
    for (gy, gx), w in expected_weights.items():
        alpha = w * conf * layer_fac
        assert out[gy, gx].item() == pytest.approx(alpha * value + (1 - alpha) * bg, rel=1e-5), (gy, gx)
    # Everywhere else is pure background.
    assert out[0, 0].item() == pytest.approx(bg, rel=1e-6)
    assert out[6, 12].item() == pytest.approx(bg, rel=1e-6)


def test_broadcast_mode_writes_every_layer_with_factor_one():
    # conf is deliberately < 1: an alpha of exactly 1.0 makes the float32 CPU
    # reference compositor's log1p(-alpha) overflow to -inf (its 1e-12 clamp
    # only survives in float64). Real confidences are sigmoid outputs, so
    # alpha == 1 is unreachable in a real render -- see docs/LIMITATIONS.md.
    conf = 0.9
    view = _view()
    points = _points_at(10.0, 6.0, 4.0, view, [1.0], conf, 0.0, [0.0])
    layers, aux = parity.render_trips_layers(points, view, num_layers=3, mode="broadcast")
    # ip halves per layer: (10, 6) -> (5, 3) -> (2.5, 1.5); each lands exactly
    # on an integer or half-integer, so every layer gets a fragment.
    assert aux["points_active"] == [1, 1, 1]
    assert layers[0][0, 6, 10].item() == pytest.approx(conf)
    assert layers[1][0, 3, 5].item() == pytest.approx(conf)
    assert layers[2][0, 1, 2].item() == pytest.approx(0.25 * conf)  # bilinear at (2.5, 1.5)


def test_trips_mode_stops_at_layer_higher():
    """A sub-pixel point reaches layer 0 only; a 5px point reaches layers 0..3."""
    view = _view()
    small = _points_at(10.0, 6.0, 4.0, view, [1.0], 1.0, 0.0, [0.0])
    _l, aux_small = parity.render_trips_layers(small, view, num_layers=4, mode="trips")
    assert aux_small["points_active"] == [1, 0, 0, 0]

    # size_px = fx * size / z = 100 * 0.2 / 4 = 5 -> ceil(log2 5) = 3.
    big = _points_at(10.0, 6.0, 4.0, view, [1.0], 1.0, 0.2, [0.0])
    _l2, aux_big = parity.render_trips_layers(big, view, num_layers=4, mode="trips")
    assert aux_big["points_active"] == [1, 1, 1, 1]


def test_trips_mode_layer_factors_match_compute_point_size_fac():
    """Layers below layer_lower get factor 1.0 (PointBlending.h:92-96)."""
    from trippy.raster.emit import layer_factor

    view = _view()
    size_px = torch.tensor([5.0])
    # lower = 2, upper = 3, interp = (5 - 4) / (8 - 4) = 0.25
    assert layer_factor(size_px, 0, 4).item() == pytest.approx(1.0)
    assert layer_factor(size_px, 1, 4).item() == pytest.approx(1.0)
    assert layer_factor(size_px, 2, 4).item() == pytest.approx(0.75)
    assert layer_factor(size_px, 3, 4).item() == pytest.approx(0.25)

    conf = 0.9  # see test_broadcast_mode_writes_every_layer_with_factor_one
    points = _points_at(10.0, 6.0, 4.0, view, [1.0], conf, 0.2, [0.0])
    layers, _ = parity.render_trips_layers(points, view, num_layers=4, mode="trips")
    assert layers[0][0, 6, 10].item() == pytest.approx(conf, rel=1e-5)
    # layer 2: bilinear weight 0.25 at (2.5, 1.5) x layer factor 0.75.
    assert layers[2][0, 1, 2].item() == pytest.approx(0.75 * 0.25 * conf, rel=1e-5)


def test_points_behind_or_outside_the_layer0_frame_are_dropped():
    view = _view()
    behind = _points_at(10.0, 6.0, 4.0, view, [1.0], 1.0, 0.0, [0.0])
    behind.xyz[:, 2] = -4.0
    _l, aux = parity.render_trips_layers(behind, view, num_layers=2, mode="trips")
    assert aux["points_active"] == [0, 0]

    outside = _points_at(100.0, 6.0, 4.0, view, [1.0], 1.0, 0.0, [0.0])
    _l2, aux2 = parity.render_trips_layers(outside, view, num_layers=2, mode="trips")
    assert aux2["points_active"] == [0, 0]


def test_empty_layer_is_pure_background():
    view = _view()
    points = _points_at(10.0, 6.0, 4.0, view, [0.5, 0.25], 1.0, 0.0, [0.75, 0.125])
    layers, _ = parity.render_trips_layers(points, view, num_layers=3, mode="trips")
    assert layers[2].shape == (2, 4, 6)
    assert torch.allclose(layers[2][0], torch.full((4, 6), 0.75))
    assert torch.allclose(layers[2][1], torch.full((4, 6), 0.125))


def test_layer_shapes_match_the_pyramid():
    view = _view(height=17, width=23)
    points = _points_at(10.0, 6.0, 4.0, view, [1.0, 2.0, 3.0, 4.0], 0.9, 0.05, [0.0, 0.0, 0.0, 0.0])
    layers, _ = parity.render_trips_layers(points, view, num_layers=4, mode="trips")
    assert [tuple(x.shape) for x in layers] == [(4, 17, 23), (4, 9, 12), (4, 5, 6), (4, 3, 3)]


def test_render_modes_are_all_reachable():
    view = _view()
    points = _points_at(10.0, 6.0, 4.0, view, [1.0, 0.0, 0.0, 0.0], 0.9, 0.05, [0.0] * 4)
    for mode in parity.RENDER_MODES:
        layers, _ = parity.render_trips_layers(points, view, num_layers=3, mode=mode)
        assert len(layers) == 3
    with pytest.raises(ValueError, match="mode must be one of"):
        parity.render_trips_layers(points, view, num_layers=3, mode="nope")


# --- metrics -------------------------------------------------------------


def test_psnr_matches_the_definition():
    a = torch.zeros(1, 3, 8, 8)
    b = torch.full((1, 3, 8, 8), 0.25)
    assert parity.psnr(a, b) == pytest.approx(10.0 * math.log10(1.0 / 0.0625))
    assert parity.psnr(a, a.clone()) == pytest.approx(99.0)


def test_crop_border_drops_every_side():
    x = torch.arange(1 * 3 * 40 * 60, dtype=torch.float32).reshape(1, 3, 40, 60)
    cropped = parity.crop_border(x, PARITY_EVAL_BORDER_PX)
    assert cropped.shape == (1, 3, 40 - 32, 60 - 32)
    assert torch.equal(parity.crop_border(x, 0), x)


def test_compare_border_excludes_a_corrupted_frame():
    torch.manual_seed(0)
    target = torch.rand(1, 3, 64, 64)
    pred = target.clone()
    pred[:, :, :PARITY_EVAL_BORDER_PX, :] = 0.0
    pred[:, :, -PARITY_EVAL_BORDER_PX:, :] = 0.0
    pred[:, :, :, :PARITY_EVAL_BORDER_PX] = 0.0
    pred[:, :, :, -PARITY_EVAL_BORDER_PX:] = 0.0
    assert parity.compare(pred, target).psnr_db < 15.0
    assert parity.compare(pred, target, PARITY_EVAL_BORDER_PX).psnr_db == pytest.approx(99.0)


def test_to_hwc_and_abs_diff_shapes():
    img = torch.rand(1, 3, 12, 20)
    arr = parity.to_hwc(img)
    assert arr.shape == (12, 20, 3) and arr.dtype == np.uint8
    heat = parity.abs_diff_heatmap(img, torch.zeros_like(img))
    assert heat.shape[:2] == (12, 20)
