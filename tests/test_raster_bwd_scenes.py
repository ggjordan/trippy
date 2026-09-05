"""A synthetic scene built to be *smooth*: the shared fixture for gradcheck.

Module: tests.test_raster_bwd_scenes
Purpose: `make_smooth_scene()` returns a tiny point set whose render is a
    differentiable function of every input in a neighbourhood far larger
    than gradcheck's finite-difference step. The rasteriser is only
    piecewise smooth -- culling, the in-bounds test, `floor()` on the
    footprint's base pixel, the `alpha >= alpha_min` drop, `floor/ceil` on
    log2(size_px), the 16-fragment cap, the transmittance cutoff and the
    depth sort are all discrete -- so a *random* scene makes gradcheck fail
    for reasons that have nothing to do with the gradient formulas. Every
    number below is chosen to sit far from one of those switching points,
    and `assert_smooth_margins()` re-checks the important ones at run time
    so the fixture cannot silently drift.
Invariants: synthetic only (AGENTS.md section 6). float64 by default.
Related docs: docs/ARCHITECTURE.md ("Backward pass data flow"),
    docs/GEOMETRY.md (pixel-centre convention).
"""

from __future__ import annotations

import math

import torch

from trippy.constants import RASTER_ALPHA_MIN, RASTER_MAX_FRAGS
from trippy.geom.xform_b import qvec2R
from trippy.raster.emit import apply_pose_delta

# Layer-0 image and pyramid used by every gradcheck case: small enough that a
# full jacobian is cheap, deep enough that the transmittance chain matters.
GRAD_HW = (8, 8)
GRAD_NUM_LAYERS = 2
GRAD_NUM_CHANNELS = 3
GRAD_FOCAL = 16.0

# Target layer-0 pixel coordinates (u, v). Each coordinate keeps the
# footprint's fractional weight at least MIN_FRAC_MARGIN away from 0 and 1 on
# *both* layers, so `floor(uv / 2**l - 0.5)` cannot flip under perturbation.
GRAD_UV = (
    (3.2, 2.2),  # \
    (3.2, 2.2),  #  } three points stacked on one pixel: the transmittance
    (3.2, 2.2),  # /  recurrence (and hence the alpha gradient) is exercised
    (5.8, 4.2),
    (5.8, 4.2),
    (2.2, 5.8),
    (4.2, 1.7),
    (1.7, 3.6),
    (6.2, 6.2),
    (2.6, 4.6),
)
# Camera-space depths, world units. Pairwise gaps are >= 0.3, i.e. ~10^6
# times gradcheck's step, so the depth sort order is locally constant.
GRAD_DEPTH = (2.6, 3.1, 3.7, 4.3, 5.0, 3.4, 4.9, 5.5, 2.9, 6.1)
# Projected sizes in layer-0 pixels. Values in (1, 2) straddle layers 0 and 1
# (the interpolating branch of layer_factor); values below 1 stay on layer 0
# (the exp() sub-pixel branch). None is near 1.0 or 2.0, where layer_bounds
# switches.
GRAD_SIZE_PX = (1.45, 1.45, 1.45, 0.55, 0.55, 1.65, 1.35, 0.70, 1.55, 1.25)
# Effective (post-sigmoid) confidences: high enough that no fragment lands
# near RASTER_ALPHA_MIN, low enough that transmittance never approaches the
# cutoff over a three-deep stack.
GRAD_CONF = (0.55, 0.45, 0.65, 0.75, 0.35, 0.85, 0.60, 0.50, 0.70, 0.40)

# World->camera pose (unit quaternion wxyz + translation) and the SE(3) twist
# the scene is built around. The delta is deliberately non-zero: at theta == 0
# xform_b's se3_exp takes its Taylor branch, and the *other* branch's
# (1 - cos t)/t^2 term is ill-conditioned for t ~ 1e-6, which is exactly the
# size of gradcheck's perturbation. theta here is ~0.037 rad, comfortably
# inside the well-conditioned regime.
# (normalised in make_smooth_scene: back-projection uses R^T as R^-1, which
# only holds for a true rotation)
GRAD_QVEC = (0.995004, 0.06, -0.05, 0.07)
GRAD_TVEC = (0.05, -0.08, 0.12)
GRAD_POSE_DELTA = (0.010, -0.020, 0.015, 0.020, -0.010, 0.030)

# Background feature, so `out += t_final * bg` is part of the output and the
# kernel's dL/d t_final path is exercised rather than fed zeros. Long enough
# for every channel count in RASTER_SUPPORTED_CHANNELS; the scene slices it.
GRAD_BG = (0.2, 0.4, 0.6, 0.3, 0.5, 0.7, 0.25, 0.45)

# Minimum distance a bilinear fractional weight keeps from 0 and 1, on every
# layer. gradcheck's default step is 1e-6, so this is a ~10^5 safety factor.
MIN_FRAC_MARGIN = 0.08
# Minimum distance log2(size_px) keeps from an integer (where layer_bounds'
# floor/ceil switches) and size_px keeps from 1.0.
MIN_SIZE_MARGIN = 0.05


def make_smooth_scene(
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    num_channels: int = GRAD_NUM_CHANNELS,
) -> dict:
    """Build the gradcheck scene.

    Points are placed by choosing their *image-space* target (uv, depth,
    size_px) and back-projecting through the pose already refined by
    `GRAD_POSE_DELTA`, so the listed uv values are the ones actually hit when
    the scene is rendered with `pose_delta=scene["pose_delta"]`.

    Args:
        dtype: torch float dtype for every tensor (float64 for gradcheck).
        device: torch device string.
        num_channels: C feature channels.

    Returns:
        dict with xyz (N, 3), size (N,), feat (N, C), conf (N,), K (3, 3),
        R (3, 3), t (3,), pose_delta (6,), bg (C,), image_hw, num_layers.
        `size` and `conf` are effective values (post-softplus / post-sigmoid).
    """
    kw = {"dtype": dtype, "device": device}
    height, width = GRAD_HW
    focal = GRAD_FOCAL
    cx, cy = 0.5 * width, 0.5 * height

    uv = torch.tensor(GRAD_UV, **kw)
    depth = torch.tensor(GRAD_DEPTH, **kw)
    size_px = torch.tensor(GRAD_SIZE_PX, **kw)
    conf = torch.tensor(GRAD_CONF, **kw)
    num_points = uv.shape[0]

    xyz_cam = torch.stack(
        [
            (uv[:, 0] - cx) * depth / focal,
            (uv[:, 1] - cy) * depth / focal,
            depth,
        ],
        dim=1,
    )
    qvec = torch.tensor(GRAD_QVEC, **kw)
    R = qvec2R(qvec / torch.linalg.norm(qvec))
    t = torch.tensor(GRAD_TVEC, **kw)
    pose_delta = torch.tensor(GRAD_POSE_DELTA, **kw)
    # Back-project through the *refined* pose: x_cam = R' x_world + t'.
    R_ref, t_ref = apply_pose_delta(R, t, pose_delta)
    xyz_world = (xyz_cam - t_ref.reshape(1, 3)) @ R_ref

    # Deterministic, spread-out features; no RNG, so the fixture is readable.
    idx = torch.arange(num_points, **kw).reshape(-1, 1)
    chan = torch.arange(num_channels, **kw).reshape(1, -1)
    feat = 0.15 + 0.7 * torch.frac(0.37 * idx + 0.19 * chan + 0.11)

    return {
        "xyz": xyz_world,
        "size": size_px * depth / focal,
        "feat": feat,
        "conf": conf,
        "K": torch.tensor([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], **kw),
        "R": R,
        "t": t,
        "pose_delta": pose_delta,
        "bg": torch.tensor(GRAD_BG[:num_channels], **kw),
        "image_hw": GRAD_HW,
        "num_layers": GRAD_NUM_LAYERS,
    }


def assert_smooth_margins() -> None:
    """Re-derive the fixture's smoothness margins; raises if one is violated.

    Checks the three discrete decisions the hand-picked numbers exist to keep
    away from: the footprint's base pixel (`floor`), layer selection
    (`floor/ceil` of log2 size), and the alpha floor.
    """
    for u, v in GRAD_UV:
        for coord in (u, v):
            for layer in range(GRAD_NUM_LAYERS):
                centred = coord / (2**layer) - 0.5
                frac = centred - math.floor(centred)
                margin = min(frac, 1.0 - frac)
                assert margin >= MIN_FRAC_MARGIN, f"uv {coord} layer {layer}: margin {margin}"
    for size_px in GRAD_SIZE_PX:
        assert abs(size_px - 1.0) >= MIN_SIZE_MARGIN, f"size_px {size_px} sits on the s == 1 switch"
        log_ps = math.log2(size_px)
        assert abs(log_ps - round(log_ps)) >= MIN_SIZE_MARGIN, f"size_px {size_px} near a power of 2"
    # Worst-case alpha: smallest bilinear corner weight x smallest confidence
    # x the smallest layer factor the sub-pixel branch can return (0.25).
    worst_beta = MIN_FRAC_MARGIN**2
    assert worst_beta * min(GRAD_CONF) * 0.25 > 10.0 * RASTER_ALPHA_MIN


def test_smooth_scene_is_actually_smooth() -> None:
    """The fixture's margins hold, and no pixel reaches the fragment cap."""
    from trippy.raster import render_pyramid_ref

    assert_smooth_margins()
    scene = make_smooth_scene()
    _, aux = render_pyramid_ref(
        scene["xyz"],
        scene["size"],
        scene["feat"],
        scene["conf"],
        scene["K"],
        scene["R"],
        scene["t"],
        scene["image_hw"],
        num_layers=scene["num_layers"],
        bg=scene["bg"],
        pose_delta=scene["pose_delta"],
    )
    used = max(int(n.max()) for n in aux["n_used"])
    assert 0 < used < RASTER_MAX_FRAGS, f"n_used {used} must be non-trivial but under the cap"
    # The stacked triple really does composite together on one layer-0 pixel.
    assert used >= 3
    assert aux["num_fragments"] > 20


def test_scene_lands_on_the_intended_pixels() -> None:
    """Back-projection through the refined pose reproduces GRAD_UV exactly."""
    from trippy.raster.emit import project_points

    scene = make_smooth_scene()
    uv, depth, size_px = project_points(
        scene["xyz"],
        scene["size"],
        scene["K"],
        scene["R"],
        scene["t"],
        pose_delta=scene["pose_delta"],
    )
    assert torch.allclose(uv, torch.tensor(GRAD_UV, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(depth, torch.tensor(GRAD_DEPTH, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(size_px, torch.tensor(GRAD_SIZE_PX, dtype=torch.float64), atol=1e-9)
