"""Camera geometry, implementation A: numpy, COLMAP quaternion convention.

Module: trippy.geom.xform_a
Invariants: NO torch import (verified by tests/test_xform_agreement.py via
    sys.modules) so this module can be used from plain numpy scripts (it
    also drives the offline dolly renderers per the plan, so drift between
    xform_a and xform_b shows up as misaligned sheets).
Coordinate frame: COLMAP world/camera convention throughout. World points
    are row vectors in an arbitrary right-handed world frame (whatever frame
    COLMAP's sparse reconstruction used). Camera space is x-right, y-down,
    z-forward (the point the camera is looking at has z > 0). Quaternions
    are unit, stored (qw, qx, qy, qz), and rotate world->camera:
        x_cam = R(q) @ x_world + t
    This matches COLMAP's images.txt convention and
    ~/Splats/research/visual/render_offpath.py's qvec2R/viewmat.
Related docs: docs/SPEC.md "Reference files" (render_offpath.py);
    trippy.geom.xform_b (independent torch implementation).
"""

from __future__ import annotations

import numpy as np


def qvec2R(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (COLMAP wxyz order) to a 3x3 rotation matrix.

    Args:
        q: shape (4,), float, (qw, qx, qy, qz), unit norm.

    Returns:
        R: shape (3, 3), float64, world->camera rotation such that
           x_cam = R @ x_world (before translation).
    """
    qw, qx, qy, qz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def world_to_cam(R: np.ndarray, t: np.ndarray, xyz_w: np.ndarray) -> np.ndarray:
    """Transform world-frame points into camera frame: x_cam = R @ x_world + t.

    Args:
        R: shape (3, 3), world->camera rotation (see qvec2R).
        t: shape (3,), world->camera translation, same units as xyz_w.
        xyz_w: shape (N, 3), world-frame points, row vectors.

    Returns:
        xyz_c: shape (N, 3), camera-frame points (x right, y down, z forward).
    """
    xyz_w = np.asarray(xyz_w, dtype=np.float64).reshape(-1, 3)
    return xyz_w @ R.T + t.reshape(1, 3)


def project_pinhole(
    xyz_c: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole-project camera-frame points to pixel coordinates.

    Args:
        xyz_c: shape (N, 3), camera-frame points (see world_to_cam). z is
            depth along the optical axis; z > 0 is in front of the camera.
        fx, fy, cx, cy: pinhole intrinsics in pixels.

    Returns:
        uv: shape (N, 2), float64, pixel coordinates (u right, v down; no
            distortion applied -- callers needing OPENCV/RADIAL distortion
            should use trippy.geom.camera.distort first).
        depth: shape (N,), float64, camera-space z (positive = in front).
    """
    xyz_c = np.asarray(xyz_c, dtype=np.float64).reshape(-1, 3)
    depth = xyz_c[:, 2].copy()
    u = fx * xyz_c[:, 0] / depth + cx
    v = fy * xyz_c[:, 1] / depth + cy
    uv = np.stack([u, v], axis=1)
    return uv, depth


def read_cameras_txt(path: str) -> dict[int, dict]:
    """Parse a COLMAP cameras.txt.

    Returns:
        dict camera_id -> {"model": str, "width": int, "height": int,
        "params": list[float]} (params meaning depends on model, e.g.
        PINHOLE=[fx,fy,cx,cy], OPENCV=[fx,fy,cx,cy,k1,k2,p1,p2]).
    """
    cameras: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            camera_id = int(parts[0])
            cameras[camera_id] = {
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(x) for x in parts[4:]],
            }
    return cameras


def read_images_txt(path: str) -> dict[str, dict]:
    """Parse a COLMAP images.txt (the two-line-per-image format).

    Uses a structural check to identify pose lines (never naive stride-2
    pairing on pre-filtered lines): a zero-observation image still writes a
    genuine, blank POINTS2D line, and filtering blanks before pairing would
    shift every later image's pairing. We instead scan line-by-line, keep
    blank lines (only comments are dropped), and consume exactly the line
    following each recognised pose line as that image's POINTS2D line.

    Returns:
        dict image_name -> {
            "image_id": int, "qvec": (4,) ndarray (qw,qx,qy,qz),
            "tvec": (3,) ndarray, "camera_id": int, "name": str,
            "points2d": list[(x: float, y: float, point3d_id: int)],
        }
        point3d_id == -1 means the 2D keypoint has no triangulated 3D point.
    """

    def _is_int(tok: str) -> bool:
        try:
            int(tok)
            return True
        except ValueError:
            return False

    def _is_num(tok: str) -> bool:
        try:
            float(tok)
            return True
        except ValueError:
            return False

    with open(path) as f:
        lines = [line for line in f if not line.startswith("#")]

    images: dict[str, dict] = {}
    i = 0
    n = len(lines)
    while i < n:
        parts = lines[i].split()
        is_pose_line = (
            len(parts) >= 10 and _is_int(parts[0]) and _is_int(parts[8]) and not _is_num(parts[9])
        )
        if not is_pose_line:
            i += 1
            continue

        image_id = int(parts[0])
        qvec = np.array([float(x) for x in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
        camera_id = int(parts[8])
        name = parts[9]

        points2d: list[tuple[float, float, int]] = []
        if i + 1 < n:
            p2 = lines[i + 1].split()
            for k in range(0, len(p2) - 2, 3):
                points2d.append((float(p2[k]), float(p2[k + 1]), int(p2[k + 2])))

        images[name] = {
            "image_id": image_id,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": camera_id,
            "name": name,
            "points2d": points2d,
        }
        i += 2

    return images


def read_points3d_txt(path: str) -> dict[int, dict]:
    """Parse a COLMAP points3D.txt.

    Returns:
        dict point3d_id -> {"xyz": (3,) ndarray float64,
        "rgb": (3,) ndarray uint8, "error": float}.
    """
    points: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            point3d_id = int(parts[0])
            points[point3d_id] = {
                "xyz": np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64),
                "rgb": np.array([int(parts[4]), int(parts[5]), int(parts[6])], dtype=np.uint8),
                "error": float(parts[7]),
            }
    return points
