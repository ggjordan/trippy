"""Tests for trippy.distill.cameras: near-path interpolation + the honesty guard.

Module: tests.test_distill_cameras
Invariants under test: `rotmat_to_qvec` inverts `trippy.geom.xform_a.qvec2R`
    (up to the R->R identity, since q and -q are the same rotation);
    `slerp` reproduces the expected intermediate rotation at t=0/0.5/1;
    `build_distill_camera_plan` produces `k` interpolated cameras per
    consecutive pair at the expected linearly-interpolated centre, skips a
    pair whose distance exceeds `max_jump_multiplier` x the median
    consecutive distance (recording it in `skipped_pairs`), and skips a
    pair whose two images use different `camera_id`s; `image_filename`
    always returns an extension-free stem + ".png".
Fixture: a minimal synthetic COLMAP text scene (cameras.txt/images.txt/
    points3D.txt only, no images/ directory -- camera interpolation never
    touches pixels), never a real Splats scene (AGENTS.md: test fixtures
    must be synthetic only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trippy.distill.cameras import (
    build_distill_camera_plan,
    image_filename,
    rotmat_to_qvec,
    slerp,
)
from trippy.geom import xform_a

IDENTITY_QVEC = (1.0, 0.0, 0.0, 0.0)
CAM_WIDTH, CAM_HEIGHT = 64, 48
FX = FY = 64.0
CX, CY = 32.0, 24.0


def _write_scene(
    scene_root: Path,
    poses: list[tuple[str, tuple[float, float, float, float], tuple[float, float, float], int]],
    cameras: dict[int, tuple[int, int, float, float, float, float]] | None = None,
) -> Path:
    """Write a minimal COLMAP text scene: cameras/images/points3D.txt only.

    `poses` is (name, qvec, tvec, camera_id); `cameras` defaults to one
    PINHOLE camera (id 1) at the module-level CAM_WIDTH/HEIGHT/FX/FY/CX/CY.
    """
    sparse_dir = scene_root / "sparse_txt"
    sparse_dir.mkdir(parents=True)

    if cameras is None:
        cameras = {1: (CAM_WIDTH, CAM_HEIGHT, FX, FY, CX, CY)}
    cam_lines = [
        f"{cid} PINHOLE {w} {h} {fx} {fy} {cx} {cy}" for cid, (w, h, fx, fy, cx, cy) in cameras.items()
    ]
    (sparse_dir / "cameras.txt").write_text("\n".join(cam_lines) + "\n")

    lines = []
    for i, (name, qvec, tvec, camera_id) in enumerate(poses, start=1):
        q = " ".join(f"{v:.10f}" for v in qvec)
        t = " ".join(f"{v:.10f}" for v in tvec)
        lines.append(f"{i} {q} {t} {camera_id} {name}")
        lines.append("")
    (sparse_dir / "images.txt").write_text("\n".join(lines) + "\n")
    (sparse_dir / "points3D.txt").write_text("")
    return scene_root


def _identity_pose_at_x(x: float) -> tuple[tuple[float, float, float, float], tuple[float, float, float]]:
    """Identity-rotation pose whose camera centre is (x, 0, 0): tvec = -R@C = -C."""
    return IDENTITY_QVEC, (-x, 0.0, 0.0)


# --- rotmat_to_qvec / slerp ---


def test_rotmat_to_qvec_inverts_qvec2r() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = xform_a.qvec2R(q)
        q_back = rotmat_to_qvec(R)
        assert q_back[0] >= 0.0  # canonical sign
        np.testing.assert_allclose(np.linalg.norm(q_back), 1.0, atol=1e-10)
        R_back = xform_a.qvec2R(q_back)
        np.testing.assert_allclose(R_back, R, atol=1e-8)


def test_rotmat_to_qvec_identity() -> None:
    q = rotmat_to_qvec(np.eye(3))
    np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-10)


def test_slerp_endpoints_and_midpoint_90_degrees() -> None:
    q0 = np.array([1.0, 0.0, 0.0, 0.0])  # identity
    q1 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # 90 deg about z

    np.testing.assert_allclose(slerp(q0, q1, 0.0), q0, atol=1e-10)
    np.testing.assert_allclose(slerp(q0, q1, 1.0), q1, atol=1e-10)

    q_half = slerp(q0, q1, 0.5)
    expected = np.array([np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)])
    np.testing.assert_allclose(q_half, expected, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(q_half), 1.0, atol=1e-10)


def test_slerp_takes_short_way_round() -> None:
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = -np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # same rotation, opposite sign
    q_half = slerp(q0, q1, 0.5)
    # Must match the short-way interpolation (as if q1 were not negated), not fly the long way round.
    expected = np.array([np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)])
    np.testing.assert_allclose(np.abs(np.dot(q_half, expected)), 1.0, atol=1e-10)


# --- image_filename ---


def test_image_filename_strips_extension_and_appends_png() -> None:
    assert image_filename("IMG_3830.jpg") == "IMG_3830.png"
    assert image_filename("INTERP_IMG_0_IMG_1_01") == "INTERP_IMG_0_IMG_1_01.png"


# --- build_distill_camera_plan ---


def test_build_distill_camera_plan_interpolates_between_consecutive_anchors(tmp_path: Path) -> None:
    xs = [0.0, 1.0, 2.0, 3.0]
    poses = [(f"IMG_{i}.jpg", *_identity_pose_at_x(x), 1) for i, x in enumerate(xs)]
    scene_root = _write_scene(tmp_path, poses)

    plan = build_distill_camera_plan(scene_root, width=CAM_WIDTH, k=2)

    assert len(plan.anchors) == 4
    assert len(plan.interpolated) == 2 * 3  # k=2 per each of 3 consecutive pairs
    assert plan.skipped_pairs == []
    assert plan.median_consecutive_distance == pytest.approx(1.0)
    assert len(plan.all_poses) == 4 + 6

    # Anchor centres/orientation match the written poses exactly.
    for pose, x in zip(plan.anchors, xs, strict=True):
        np.testing.assert_allclose(pose.R, np.eye(3), atol=1e-8)
        center = -pose.R.T @ pose.t
        np.testing.assert_allclose(center, [x, 0.0, 0.0], atol=1e-8)

    # Interpolated centres for the first pair (x=0 -> x=1) sit at frac=1/3, 2/3.
    first_pair = [p for p in plan.interpolated if p.name.startswith("INTERP_IMG_0_IMG_1_")]
    assert len(first_pair) == 2
    fracs = sorted(-p.t[0] for p in first_pair)  # identity rotation: centre_x == -t[0]
    np.testing.assert_allclose(fracs, [1.0 / 3.0, 2.0 / 3.0], atol=1e-8)
    for pose in first_pair:
        assert pose.image_name is None  # no natural source photo


def test_build_distill_camera_plan_honesty_guard_skips_large_jump(tmp_path: Path) -> None:
    xs = [0.0, 1.0, 2.0, 100.0]  # last consecutive pair jumps 98 units
    poses = [(f"IMG_{i}.jpg", *_identity_pose_at_x(x), 1) for i, x in enumerate(xs)]
    scene_root = _write_scene(tmp_path, poses)

    plan = build_distill_camera_plan(scene_root, width=CAM_WIDTH, k=2, max_jump_multiplier=4.0)

    assert plan.median_consecutive_distance == pytest.approx(1.0)  # distances [1, 1, 98] -> median 1
    assert len(plan.skipped_pairs) == 1
    skipped = plan.skipped_pairs[0]
    assert (skipped.name_a, skipped.name_b) == ("IMG_2.jpg", "IMG_3.jpg")
    assert skipped.distance == pytest.approx(98.0)
    assert "honesty guard" in skipped.reason
    # Only the two normal pairs get interpolated (k=2 each); the jump pair gets none.
    assert len(plan.interpolated) == 2 * 2
    assert not any(p.name.startswith("INTERP_IMG_2_IMG_3_") for p in plan.interpolated)


def test_build_distill_camera_plan_skips_pairs_with_different_camera_id(tmp_path: Path) -> None:
    cameras = {1: (CAM_WIDTH, CAM_HEIGHT, FX, FY, CX, CY), 2: (CAM_WIDTH, CAM_HEIGHT, FX, FY, CX, CY)}
    poses = [
        ("IMG_0.jpg", *_identity_pose_at_x(0.0), 1),
        ("IMG_1.jpg", *_identity_pose_at_x(1.0), 2),  # different camera_id
    ]
    scene_root = _write_scene(tmp_path, poses, cameras=cameras)

    plan = build_distill_camera_plan(scene_root, width=CAM_WIDTH, k=2)

    assert len(plan.interpolated) == 0
    assert len(plan.skipped_pairs) == 1
    assert plan.skipped_pairs[0].reason == "different camera_id"


def test_build_distill_camera_plan_k_zero_produces_no_interpolation(tmp_path: Path) -> None:
    xs = [0.0, 1.0, 2.0]
    poses = [(f"IMG_{i}.jpg", *_identity_pose_at_x(x), 1) for i, x in enumerate(xs)]
    scene_root = _write_scene(tmp_path, poses)

    plan = build_distill_camera_plan(scene_root, width=CAM_WIDTH, k=0)

    assert len(plan.anchors) == 3
    assert plan.interpolated == []
    assert plan.skipped_pairs == []


def test_build_distill_camera_plan_unknown_name_raises(tmp_path: Path) -> None:
    poses = [("IMG_0.jpg", *_identity_pose_at_x(0.0), 1)]
    scene_root = _write_scene(tmp_path, poses)
    with pytest.raises(KeyError):
        build_distill_camera_plan(scene_root, width=CAM_WIDTH, names=["IMG_0.jpg", "IMG_NOPE.jpg"])


def test_build_distill_camera_plan_single_anchor_has_no_pairs(tmp_path: Path) -> None:
    poses = [("IMG_0.jpg", *_identity_pose_at_x(0.0), 1)]
    scene_root = _write_scene(tmp_path, poses)

    plan = build_distill_camera_plan(scene_root, width=CAM_WIDTH, k=2)

    assert len(plan.anchors) == 1
    assert plan.interpolated == []
    assert plan.skipped_pairs == []
    assert plan.median_consecutive_distance == 0.0
    assert plan.jump_threshold == float("inf")
