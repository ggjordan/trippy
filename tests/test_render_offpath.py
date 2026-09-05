"""Tests for trippy.render.offpath: lateral + oblique off-path honesty poses.

Module: tests.test_render_offpath
Invariants under test: `up_vector` averages every registered image's
    world-frame up axis (R.T @ [0, -1, 0]) and renormalises;
    `offpath_poses` returns a (lateral, oblique) pair per requested image,
    positioned/oriented per the documented Splats construction (see
    trippy.render.offpath's module docstring for exact source lines); an
    unregistered image name raises.
Fixture: a minimal synthetic COLMAP text scene (cameras.txt/images.txt/
    points3D.txt only -- no `images/` directory or photos needed, since
    `offpath_poses` never touches pixels), never a real Splats scene
    (AGENTS.md: test fixtures must be synthetic only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trippy.constants import (
    OFFPATH_DEFAULT_ELEVATE_FRAC,
    OFFPATH_DEFAULT_LATERAL_FRAC,
    OFFPATH_OBLIQUE_BACK_FRAC,
)
from trippy.geom import xform_a
from trippy.render.dolly import camera_center, local_depth_estimate
from trippy.render.offpath import offpath_poses, up_vector
from trippy.scene import colmap_io
from trippy.scene.dataset import resolve_sparse_dir

IDENTITY_QVEC = (1.0, 0.0, 0.0, 0.0)
# 90-degree rotation about the world x-axis: qw=qx=cos/sin(45deg), qy=qz=0.
ROT90_X_QVEC = (0.70710678, 0.70710678, 0.0, 0.0)
CAM_WIDTH, CAM_HEIGHT = 64, 48
FX = FY = 64.0
CX, CY = 32.0, 24.0


def _write_synthetic_colmap_scene(
    scene_root: Path,
    poses: list[tuple[str, tuple[float, float, float, float], tuple[float, float, float]]],
    points_xyz: np.ndarray,
    width: int = CAM_WIDTH,
    height: int = CAM_HEIGHT,
) -> Path:
    """Write a minimal 1-camera PINHOLE COLMAP text scene: cameras/images/points3D.txt only."""
    sparse_dir = scene_root / "sparse_txt"
    sparse_dir.mkdir(parents=True)
    (sparse_dir / "cameras.txt").write_text(f"1 PINHOLE {width} {height} {FX} {FY} {CX} {CY}\n")

    lines = []
    for i, (name, qvec, tvec) in enumerate(poses, start=1):
        q = " ".join(f"{v:.10f}" for v in qvec)
        t = " ".join(f"{v:.10f}" for v in tvec)
        lines.append(f"{i} {q} {t} 1 {name}")
        lines.append("")
    (sparse_dir / "images.txt").write_text("\n".join(lines) + "\n")

    point_lines = [f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 128 128 128 0.5" for i, p in enumerate(points_xyz, start=1)]
    (sparse_dir / "points3D.txt").write_text("\n".join(point_lines) + ("\n" if point_lines else ""))
    return sparse_dir


def _synthetic_points_in_front(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-2.0, 2.0, size=(n, 2))
    z = rng.uniform(5.0, 15.0, size=(n, 1))
    return np.concatenate([xy, z], axis=1)


def test_up_vector_averages_and_renormalises_every_registered_image(tmp_path: Path) -> None:
    poses = [
        ("IMG_A.jpg", IDENTITY_QVEC, (0.0, 0.0, 0.0)),
        ("IMG_B.jpg", ROT90_X_QVEC, (1.0, 0.0, 0.0)),
    ]
    _write_synthetic_colmap_scene(tmp_path, poses, _synthetic_points_in_front())

    sparse_dir = resolve_sparse_dir(tmp_path)
    scene = colmap_io.load_colmap_model(sparse_dir)

    expected = np.mean(
        [xform_a.qvec2R(np.array(qvec)).T @ np.array([0.0, -1.0, 0.0]) for _, qvec, _ in poses], axis=0
    )
    expected = expected / np.linalg.norm(expected)

    np.testing.assert_allclose(up_vector(scene), expected, atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(up_vector(scene)), 1.0, atol=1e-8)


def test_offpath_poses_returns_lateral_then_oblique_pair_per_image(tmp_path: Path) -> None:
    points = _synthetic_points_in_front()
    _write_synthetic_colmap_scene(tmp_path, [("IMG_A.jpg", IDENTITY_QVEC, (0.0, 0.0, 0.0))], points)

    poses = offpath_poses(tmp_path, ["IMG_A.jpg"], width=CAM_WIDTH)
    assert [p.name for p in poses] == ["IMG_A_lateral", "IMG_A_oblique"]
    assert all(p.image_name == "IMG_A.jpg" for p in poses)
    assert all(p.image_hw == (CAM_HEIGHT, CAM_WIDTH) for p in poses)


def test_offpath_lateral_pose_position_and_look_at_target(tmp_path: Path) -> None:
    points = _synthetic_points_in_front()
    _write_synthetic_colmap_scene(tmp_path, [("IMG_A.jpg", IDENTITY_QVEC, (0.0, 0.0, 0.0))], points)

    lateral_frac = OFFPATH_DEFAULT_LATERAL_FRAC
    poses = offpath_poses(tmp_path, ["IMG_A.jpg"], lateral_frac=lateral_frac, width=CAM_WIDTH)
    lateral = poses[0]

    depth = local_depth_estimate(points, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    up = np.array([0.0, -1.0, 0.0])  # identity rotation -> R.T @ [0, -1, 0] = [0, -1, 0]
    fwd0 = np.array([0.0, 0.0, 1.0])
    side = np.cross(up, fwd0)
    side = side / np.linalg.norm(side)
    expected_center = side * lateral_frac * depth
    expected_target = fwd0 * depth

    center = camera_center(lateral.R, lateral.t)
    np.testing.assert_allclose(center, expected_center, atol=1e-6)

    # Row 2 of R is the camera's forward axis (see look_at_pose); it must point at the target.
    expected_forward = expected_target - expected_center
    expected_forward = expected_forward / np.linalg.norm(expected_forward)
    np.testing.assert_allclose(lateral.R[2], expected_forward, atol=1e-6)


def test_offpath_oblique_pose_rises_and_looks_at_centroid(tmp_path: Path) -> None:
    points = _synthetic_points_in_front()
    _write_synthetic_colmap_scene(tmp_path, [("IMG_A.jpg", IDENTITY_QVEC, (0.0, 0.0, 0.0))], points)

    elevate_frac = OFFPATH_DEFAULT_ELEVATE_FRAC
    poses = offpath_poses(tmp_path, ["IMG_A.jpg"], elevate_frac=elevate_frac, width=CAM_WIDTH)
    oblique = poses[1]

    depth = local_depth_estimate(points, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    up = np.array([0.0, -1.0, 0.0])
    fwd0 = np.array([0.0, 0.0, 1.0])
    centroid = np.zeros(3)  # single camera at the origin -> centroid is the origin
    expected_center = up * elevate_frac * depth - fwd0 * depth * OFFPATH_OBLIQUE_BACK_FRAC

    center = camera_center(oblique.R, oblique.t)
    np.testing.assert_allclose(center, expected_center, atol=1e-6)

    expected_forward = centroid - expected_center
    expected_forward = expected_forward / np.linalg.norm(expected_forward)
    np.testing.assert_allclose(oblique.R[2], expected_forward, atol=1e-6)


def test_offpath_poses_unregistered_name_raises(tmp_path: Path) -> None:
    _write_synthetic_colmap_scene(tmp_path, [("IMG_A.jpg", IDENTITY_QVEC, (0.0, 0.0, 0.0))], _synthetic_points_in_front())
    with pytest.raises(KeyError):
        offpath_poses(tmp_path, ["IMG_NOPE.jpg"])
