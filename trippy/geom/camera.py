"""Camera intrinsics: pinhole + OpenCV-style lens distortion.

Module: trippy.geom.camera
Invariants: numpy only; distort() and undistort_maps() below are used by
    trippy.scene.dataset for the one-time undistortion + multi-resolution
    cache -- it is not yet wired into xform_a/xform_b's project_pinhole,
    which stays distortion-free by design.
Coordinate frame: same COLMAP convention as trippy.geom.xform_a/xform_b.
    "Normalized" coordinates below means camera-frame x/z, y/z (i.e. pixel
    coordinates with fx=fy=1, cx=cy=0) -- not yet scaled by focal length.
Related docs: docs/GEOMETRY.md "Image coordinates" (pixel-centre
    convention, used by undistort_maps' grid_sample coordinate mapping);
    tests/test_colmap_reprojection.py (uses intrinsics_from_colmap_params
    to get fx,fy,cx,cy from any of PINHOLE/SIMPLE_PINHOLE/OPENCV/
    SIMPLE_RADIAL, ignoring distortion); trippy.scene.dataset (undistort
    cache, calls undistort_maps).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# COLMAP camera models this module knows how to reduce to plain fx,fy,cx,cy
# (distortion params, if any, are ignored by intrinsics_from_colmap_params).
_SIMPLE_F_MODELS = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}
_SEPARATE_FXFY_MODELS = {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "RADIAL"}


@dataclass(frozen=True)
class Pinhole:
    """Distortion-free pinhole intrinsics, in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class OpenCVDistortion:
    """COLMAP/OpenCV radial-tangential distortion coefficients (no k3)."""

    k1: float
    k2: float
    p1: float
    p2: float

    def distort(self, uv_normalized: np.ndarray) -> np.ndarray:
        """Apply radial+tangential distortion to normalized coordinates.

        Args:
            uv_normalized: shape (N, 2), float, (x/z, y/z) camera-frame
                ratios (i.e. pixel coords with fx=fy=1, cx=cy=0).

        Returns:
            shape (N, 2), distorted normalized coordinates. Multiply by
            (fx, fy) and add (cx, cy) to get distorted pixel coordinates.
        """
        uv = np.asarray(uv_normalized, dtype=np.float64).reshape(-1, 2)
        x, y = uv[:, 0], uv[:, 1]
        r2 = x * x + y * y
        radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        x_d = x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
        y_d = y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
        return np.stack([x_d, y_d], axis=1)


def undistort_maps(
    fx_src: float,
    fy_src: float,
    cx_src: float,
    cy_src: float,
    width_src: int,
    height_src: int,
    fx_dst: float,
    fy_dst: float,
    cx_dst: float,
    cy_dst: float,
    width_dst: int,
    height_dst: int,
    distortion: OpenCVDistortion | None,
) -> np.ndarray:
    """Sampling grid to undistort+resize a source image to a pinhole dst image.

    For each destination pixel (a pinhole camera with no distortion), casts
    the ray implied by (fx_dst, fy_dst, cx_dst, cy_dst), applies the source
    camera's forward OpenCV distortion to find where that ray lands in the
    as-captured (distorted) source image, and expresses that location as a
    normalized grid_sample coordinate.

    Pixel-centre convention (docs/GEOMETRY.md): pixel (row i, col j) spans
    continuous coordinates [j, j+1) x [i, i+1) with centre (j+0.5, i+0.5).
    Combined with `align_corners=False`, `torch.nn.functional.grid_sample`
    maps its normalized coordinate g in [-1, 1] to continuous pixel
    coordinate `x = (g + 1) * size / 2`, i.e. `g = 2*x/size - 1` -- exactly
    the pixel-centre convention above, no extra +/-0.5 shift needed.

    Args:
        fx_src, fy_src, cx_src, cy_src: source camera pinhole intrinsics
            (before distortion), pixels.
        width_src, height_src: source image size, pixels.
        fx_dst, fy_dst, cx_dst, cy_dst: destination (pinhole) intrinsics,
            pixels.
        width_dst, height_dst: destination image size, pixels.
        distortion: source camera's OpenCVDistortion, or None/all-zero for
            an already-pinhole source (then this degenerates to a plain
            resize grid).

    Returns:
        grid: shape (height_dst, width_dst, 2), float32, (x, y) normalized
            coordinates in [-1, 1] (may fall outside that range at image
            corners under strong distortion -- callers should grid_sample
            with `padding_mode="zeros"`), ready for
            `torch.nn.functional.grid_sample(..., align_corners=False)`
            after adding a batch dimension.
    """
    rows = np.arange(height_dst, dtype=np.float64)
    cols = np.arange(width_dst, dtype=np.float64)
    v_dst = rows[:, None] + 0.5  # pixel-centre row coordinate, shape (H, 1)
    u_dst = cols[None, :] + 0.5  # pixel-centre col coordinate, shape (1, W)

    x = (u_dst - cx_dst) / fx_dst  # broadcasts to (H, W)
    y = (v_dst - cy_dst) / fy_dst
    x = np.broadcast_to(x, (height_dst, width_dst))
    y = np.broadcast_to(y, (height_dst, width_dst))

    if distortion is not None and any((distortion.k1, distortion.k2, distortion.p1, distortion.p2)):
        uv = np.stack([x.ravel(), y.ravel()], axis=1)
        uv_d = distortion.distort(uv)
        x_d = uv_d[:, 0].reshape(height_dst, width_dst)
        y_d = uv_d[:, 1].reshape(height_dst, width_dst)
    else:
        x_d, y_d = x, y

    u_src = fx_src * x_d + cx_src
    v_src = fy_src * y_d + cy_src

    grid_x = 2.0 * u_src / width_src - 1.0
    grid_y = 2.0 * v_src / height_src - 1.0
    return np.stack([grid_x, grid_y], axis=-1).astype(np.float32)


def intrinsics_from_colmap_params(model: str, params: list[float]) -> tuple[float, float, float, float]:
    """Extract (fx, fy, cx, cy) from a COLMAP camera's model + params, ignoring distortion.

    Supports PINHOLE, SIMPLE_PINHOLE, OPENCV, SIMPLE_RADIAL (and the other
    common models with either a shared or separate fx/fy in COLMAP's fixed
    param ordering). Distortion terms (k1, k2, p1, p2, ...), if present, are
    dropped -- callers needing them should read cameras.txt params directly
    and use OpenCVDistortion.

    Args:
        model: COLMAP camera model name, e.g. "OPENCV".
        params: the camera's raw params list, in COLMAP's per-model order.

    Returns:
        (fx, fy, cx, cy) as plain floats.
    """
    if model in _SIMPLE_F_MODELS:
        f, cx, cy = params[0], params[1], params[2]
        return float(f), float(f), float(cx), float(cy)
    if model in _SEPARATE_FXFY_MODELS:
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return float(fx), float(fy), float(cx), float(cy)
    raise ValueError(f"unsupported COLMAP camera model: {model!r}")
