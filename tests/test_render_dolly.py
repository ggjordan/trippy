"""Tests for trippy.render.dolly: the shade dolly camera path generator.

Module: tests.test_render_dolly
Invariants under test: orientation is identical across every frame of a
    dolly path; the camera centre slides along the pose's own forward ray
    by `t * local_depth`; `local_depth_estimate` matches the documented
    Splats construction (median of in-front, percentile-trimmed distances,
    with the documented fallbacks); intrinsics scale correctly with
    `width`; unknown pose names raise.
Fixture: a minimal synthetic COLMAP text scene (cameras.txt/images.txt/
    points3D.txt only -- no `images/` directory or photos are needed, since
    `shade_dolly_poses` never touches pixels), never a real Splats scene
    (AGENTS.md: test fixtures must be synthetic only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trippy.constants import DOLLY_FALLBACK_DEPTH
from trippy.render.dolly import camera_center, local_depth_estimate, shade_dolly_poses

IDENTITY_QVEC = (1.0, 0.0, 0.0, 0.0)
CAM_WIDTH, CAM_HEIGHT = 64, 48
FX = FY = 64.0
CX, CY = 32.0, 24.0


def _write_synthetic_colmap_scene(
    scene_root: Path,
    poses: list[tuple[str, tuple[float, float, float, float], tuple[float, float, float]]],
    points_xyz: np.ndarray,
    width: int = CAM_WIDTH,
    height: int = CAM_HEIGHT,
    fx: float = FX,
    fy: float = FY,
    cx: float = CX,
    cy: float = CY,
) -> Path:
    """Write a minimal 1-camera PINHOLE COLMAP text scene: cameras/images/points3D.txt only."""
    sparse_dir = scene_root / "sparse_txt"
    sparse_dir.mkdir(parents=True)
    (sparse_dir / "cameras.txt").write_text(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    lines = []
    for i, (name, qvec, tvec) in enumerate(poses, start=1):
        q = " ".join(f"{v:.10f}" for v in qvec)
        t = " ".join(f"{v:.10f}" for v in tvec)
        lines.append(f"{i} {q} {t} 1 {name}")
        lines.append("")  # zero observations -- dolly/offpath never read per-image points2D.
    (sparse_dir / "images.txt").write_text("\n".join(lines) + "\n")

    point_lines = [f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 128 128 128 0.5" for i, p in enumerate(points_xyz, start=1)]
    (sparse_dir / "points3D.txt").write_text("\n".join(point_lines) + ("\n" if point_lines else ""))
    return sparse_dir


def _synthetic_points_in_front(n: int = 200, seed: int = 0) -> np.ndarray:
    """Points scattered in front of an identity-rotation, origin-centred camera (z in [5, 15])."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-2.0, 2.0, size=(n, 2))
    z = rng.uniform(5.0, 15.0, size=(n, 1))
    return np.concatenate([xy, z], axis=1)


def test_shade_dolly_poses_fixed_orientation_and_slides_along_forward_ray(tmp_path: Path) -> None:
    points = _synthetic_points_in_front()
    _write_synthetic_colmap_scene(tmp_path, [("IMG_A.jpg", IDENTITY_QVEC, (0.0, 0.0, 0.0))], points)

    t_range = (-0.35, 1.2)
    n = 5
    poses = shade_dolly_poses(tmp_path, pose_name="IMG_A.jpg", t_range=t_range, n=n, width=CAM_WIDTH)
    assert len(poses) == n

    expected_depth = local_depth_estimate(points, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    assert expected_depth > 0

    ts = np.linspace(t_range[0], t_range[1], n)
    for pose, t in zip(poses, ts, strict=True):
        np.testing.assert_allclose(pose.R, np.eye(3), atol=1e-8)  # orientation frozen every frame
        center = camera_center(pose.R, pose.t)
        np.testing.assert_allclose(center, [0.0, 0.0, expected_depth * t], atol=1e-6)
        assert pose.image_name == "IMG_A.jpg"
        assert f"{t:+.2f}" in pose.name

    # Native camera width requested -> intrinsics unscaled.
    np.testing.assert_allclose(poses[0].K, [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
    assert poses[0].image_hw == (CAM_HEIGHT, CAM_WIDTH)


def test_shade_dolly_poses_scales_intrinsics_and_height_with_width(tmp_path: Path) -> None:
    points = _synthetic_points_in_front()
    _write_synthetic_colmap_scene(tmp_path, [("IMG_A.jpg", IDENTITY_QVEC, (0.0, 0.0, 0.0))], points)

    poses = shade_dolly_poses(tmp_path, pose_name="IMG_A.jpg", n=2, width=CAM_WIDTH // 2)
    height, width = poses[0].image_hw
    assert width == CAM_WIDTH // 2
    assert height == CAM_HEIGHT // 2
    np.testing.assert_allclose(poses[0].K[0, 0], FX * 0.5)
    np.testing.assert_allclose(poses[0].K[1, 1], FY * 0.5)


def test_shade_dolly_poses_unknown_pose_name_raises(tmp_path: Path) -> None:
    _write_synthetic_colmap_scene(tmp_path, [("IMG_A.jpg", IDENTITY_QVEC, (0.0, 0.0, 0.0))], _synthetic_points_in_front())
    with pytest.raises(KeyError):
        shade_dolly_poses(tmp_path, pose_name="IMG_NOPE.jpg")


def test_shade_dolly_poses_falls_back_to_default_depth_with_no_points(tmp_path: Path) -> None:
    _write_synthetic_colmap_scene(
        tmp_path, [("IMG_A.jpg", IDENTITY_QVEC, (0.0, 0.0, 0.0))], np.zeros((0, 3), dtype=np.float64)
    )
    poses = shade_dolly_poses(tmp_path, pose_name="IMG_A.jpg", t_range=(0.0, 1.0), n=2)
    center_t1 = camera_center(poses[1].R, poses[1].t)
    np.testing.assert_allclose(center_t1, [0.0, 0.0, DOLLY_FALLBACK_DEPTH], atol=1e-6)


def test_local_depth_estimate_matches_manual_median_of_infront_trimmed_points() -> None:
    rng = np.random.default_rng(1)
    infront = rng.uniform(5.0, 15.0, size=(120,))
    behind = rng.uniform(-10.0, -1.0, size=(10,))
    points = np.stack([np.zeros(130), np.zeros(130), np.concatenate([infront, behind])], axis=1)

    depth = local_depth_estimate(points, np.zeros(3), np.array([0.0, 0.0, 1.0]))

    lo, hi = np.percentile(infront, [5.0, 95.0])
    trimmed = infront[(infront >= lo) & (infront <= hi)]
    assert depth == pytest.approx(float(np.median(trimmed)))
