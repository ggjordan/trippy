"""Camera intrinsics: pinhole + OpenCV-style lens distortion.

Module: trippy.geom.camera
Invariants: numpy only; distort() is used only for undistortion later (per
    plan v0.1.0 "dataset (undistort+cache)") -- it is not yet wired into
    xform_a/xform_b's project_pinhole, which is distortion-free by design.
Coordinate frame: same COLMAP convention as trippy.geom.xform_a/xform_b.
    "Normalized" coordinates below means camera-frame x/z, y/z (i.e. pixel
    coordinates with fx=fy=1, cx=cy=0) -- not yet scaled by focal length.
Related docs: docs/SPEC.md "Technical design" (dataset undistort);
    tests/test_colmap_reprojection.py (uses intrinsics_from_colmap_params
    to get fx,fy,cx,cy from any of PINHOLE/SIMPLE_PINHOLE/OPENCV/
    SIMPLE_RADIAL, ignoring distortion).
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
