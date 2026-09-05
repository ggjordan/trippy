"""Shared synthetic scenes for the rasteriser tests.

Module: tests.test_raster_scenes
Invariants: synthetic only, never a real capture (AGENTS.md section 6:
    "test fixtures must be synthetic only"). Scenes are built by choosing the
    *image-space* target (uv, depth) for every point and back-projecting
    through a non-trivial world->camera pose, so a test can guarantee awkward
    cases -- a pixel with more than the 16-fragment cap, points straddling
    every image border, sub-pixel and multi-pixel footprints -- without
    relying on luck.
Related docs: docs/GEOMETRY.md (COLMAP conventions); docs/TRIPS_REFERENCE.md
    section 3.
"""

from __future__ import annotations

import numpy as np
import torch

from trippy.geom.xform_a import qvec2R

# A pixel deliberately stacked with more fragments than RASTER_MAX_FRAGS, so
# the per-pixel cap and the transmittance cutoff are both exercised.
STACK_PIXEL_UV = (10.7, 12.3)
STACK_COUNT = 24

# The stacked points are deliberately sub-pixel (so "trilinear" mode puts all
# of them on layer 0) and low-confidence (so 16 of them still composite before
# transmittance falls under RASTER_T_CUTOFF). Without both, the fragment cap
# would never be the binding constraint.
STACK_SIZE_PX = 0.9
STACK_CONF = 0.15

# World->camera pose used by every scene: a small rotation (unit quaternion,
# wxyz) plus a translation, so the tests exercise world_to_cam rather than
# an identity shortcut.
SCENE_QVEC = (0.987688, 0.09, -0.12, 0.05)
SCENE_TVEC = (0.10, -0.20, 0.30)


def make_scene(
    num_points: int = 50,
    height: int = 32,
    width: int = 32,
    num_channels: int = 3,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> dict:
    """Build a synthetic point set + camera with known awkward cases.

    Args:
        num_points: N; must be at least STACK_COUNT + 6 border points.
        height, width: layer-0 image size in pixels.
        num_channels: C feature channels.
        seed: RNG seed (deterministic).
        dtype: torch dtype for every float tensor.
        device: torch device string.

    Returns:
        dict with keys xyz (N, 3) world-frame, size (N,) world units,
        feat (N, C), conf (N,) in (0, 1), K (3, 3), R (3, 3), t (3,),
        image_hw (H, W), bg (C,). Confidence and size are already
        "effective" values (post-sigmoid / post-softplus).
    """
    rng = np.random.default_rng(seed)
    focal = 2.0 * float(max(height, width))
    cx = 0.5 * width
    cy = 0.5 * height

    uv_list: list[tuple[float, float]] = []
    depth_list: list[float] = []

    # 1. A single pixel stacked far beyond the 16-fragment cap.
    for k in range(STACK_COUNT):
        uv_list.append(STACK_PIXEL_UV)
        depth_list.append(3.0 + 0.13 * k)

    # 2. Points straddling every border, inside and just outside.
    border = [
        (-0.4, 5.5),
        (0.3, 20.5),
        (width - 0.2, 8.5),
        (width + 0.4, 25.5),
        (12.5, -0.3),
        (18.5, height + 0.35),
    ]
    for k, uv in enumerate(border):
        uv_list.append(uv)
        depth_list.append(2.5 + 0.2 * k)

    # 3. The rest: uniform over a slightly enlarged image, uniform in depth.
    remaining = num_points - len(uv_list)
    if remaining < 0:
        raise ValueError(f"num_points must be >= {len(uv_list)}, got {num_points}")
    for _ in range(remaining):
        uv_list.append(
            (float(rng.uniform(-2.0, width + 2.0)), float(rng.uniform(-2.0, height + 2.0)))
        )
        depth_list.append(float(rng.uniform(2.0, 8.0)))

    uv = np.asarray(uv_list, dtype=np.float64)
    depth = np.asarray(depth_list, dtype=np.float64)

    # Projected sizes spanning sub-pixel (layer 0 only) to ~12 px (layers 3/4).
    size_px = rng.uniform(0.1, 12.0, size=num_points)
    conf = rng.uniform(0.2, 0.98, size=num_points)
    size_px[:STACK_COUNT] = STACK_SIZE_PX
    conf[:STACK_COUNT] = STACK_CONF
    size_world = size_px * depth / focal

    xyz_cam = np.stack(
        [(uv[:, 0] - cx) * depth / focal, (uv[:, 1] - cy) * depth / focal, depth], axis=1
    )
    R = qvec2R(np.asarray(SCENE_QVEC, dtype=np.float64))
    t = np.asarray(SCENE_TVEC, dtype=np.float64)
    # x_cam = R @ x_world + t  =>  x_world = R^T (x_cam - t).
    xyz_world = (xyz_cam - t.reshape(1, 3)) @ R

    K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    feat = rng.uniform(0.0, 1.0, size=(num_points, num_channels))
    bg = rng.uniform(0.0, 1.0, size=num_channels)

    def to_torch(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype, device=device)

    return {
        "xyz": to_torch(xyz_world),
        "size": to_torch(size_world),
        "feat": to_torch(feat),
        "conf": to_torch(conf),
        "K": to_torch(K),
        "R": to_torch(R),
        "t": to_torch(t),
        "image_hw": (height, width),
        "bg": to_torch(bg),
    }


def as_numpy(scene: dict) -> dict:
    """Same scene as float64 numpy arrays, for trippy.raster.ref_numpy."""
    return {
        key: (value.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(value) else value)
        for key, value in scene.items()
    }


def test_scene_is_deterministic_and_awkward() -> None:
    """The shared fixture really does contain the cases the tests rely on."""
    a = make_scene(seed=3)
    b = make_scene(seed=3)
    assert torch.equal(a["xyz"], b["xyz"])
    assert a["xyz"].shape == (50, 3)
    # The stacked pixel gives one layer-0 pixel far more than the 16 cap.
    assert STACK_COUNT > 16
    # Border points exist on all four sides.
    scene = as_numpy(make_scene(seed=3))
    assert scene["xyz"].dtype == np.float64
