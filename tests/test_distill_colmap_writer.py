"""Tests for trippy.distill.colmap_writer: camera grouping + points3D subsampling.

Module: tests.test_distill_colmap_writer
Invariants under test: `write_distill_colmap_model` groups CameraPoses into
    COLMAP cameras by (fx, fy, cx, cy, height, width) -- one row in
    cameras.txt per distinct intrinsics/size, however many poses share it;
    every pose becomes one row in images.txt named via
    `trippy.distill.cameras.image_filename`; the point cloud is subsampled
    (seeded, deterministic) down to `max_init_points` rows, or written in
    full when `max_init_points is None` or the cloud is already small; the
    written model round-trips through `trippy.scene.colmap_io.
    load_colmap_model` (positions/colours/camera intrinsics all match).
Fixture: a synthetic DistillCameraPlan built directly from CameraPose
    objects (no COLMAP scene needed -- this module never reads one) and a
    synthetic random point cloud (AGENTS.md: test fixtures must be
    synthetic only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trippy.distill.cameras import DistillCameraPlan, image_filename
from trippy.distill.colmap_writer import write_distill_colmap_model
from trippy.render.dolly import CameraPose
from trippy.scene import colmap_io


def _pose(name: str, x: float, fx: float = 64.0, width: int = 64, height: int = 48, image_name: str | None = None) -> CameraPose:
    K = np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]])
    return CameraPose(name=name, R=np.eye(3), t=np.array([-x, 0.0, 0.0]), K=K, image_hw=(height, width), image_name=image_name)


def _random_points(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-5.0, 5.0, (n, 3))
    rgb = rng.uniform(0.0, 1.0, (n, 3))
    return xyz, rgb


def test_write_distill_colmap_model_groups_cameras_by_intrinsics(tmp_path: Path) -> None:
    anchors = [
        _pose("IMG_0.jpg", 0.0, image_name="IMG_0.jpg"),
        _pose("IMG_1.jpg", 1.0, image_name="IMG_1.jpg"),
    ]
    interpolated = [_pose("INTERP_IMG_0_IMG_1_01", 0.5)]  # same intrinsics as the two anchors above
    plan = DistillCameraPlan(anchors=anchors, interpolated=interpolated)
    xyz, rgb = _random_points(50)

    sparse_dir = tmp_path / "sparse_txt"
    summary = write_distill_colmap_model(sparse_dir, plan, xyz, rgb, max_init_points=None)

    assert summary.n_cameras == 1  # all three poses share one (fx, fy, cx, cy, h, w)
    assert summary.n_images == 3
    assert summary.n_anchor_images == 2
    assert summary.n_interpolated_images == 1
    assert summary.n_points_source == 50
    assert summary.n_points_written == 50

    scene = colmap_io.load_colmap_model(sparse_dir)
    assert len(scene.cameras) == 1
    assert len(scene.images) == 3
    names = {im.name for im in scene.images.values()}
    assert names == {image_filename("IMG_0.jpg"), image_filename("IMG_1.jpg"), image_filename("INTERP_IMG_0_IMG_1_01")}


def test_write_distill_colmap_model_distinct_intrinsics_get_separate_cameras(tmp_path: Path) -> None:
    anchors = [
        _pose("IMG_0.jpg", 0.0, fx=64.0, width=64, height=48, image_name="IMG_0.jpg"),
        _pose("IMG_1.jpg", 1.0, fx=32.0, width=32, height=24, image_name="IMG_1.jpg"),  # different camera
    ]
    plan = DistillCameraPlan(anchors=anchors)
    xyz, rgb = _random_points(10)

    summary = write_distill_colmap_model(tmp_path / "sparse_txt", plan, xyz, rgb, max_init_points=None)
    assert summary.n_cameras == 2


def test_write_distill_colmap_model_subsamples_points_deterministically(tmp_path: Path) -> None:
    plan = DistillCameraPlan(anchors=[_pose("IMG_0.jpg", 0.0, image_name="IMG_0.jpg")])
    xyz, rgb = _random_points(1000)

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    summary_a = write_distill_colmap_model(dir_a, plan, xyz, rgb, max_init_points=100, seed=7)
    write_distill_colmap_model(dir_b, plan, xyz, rgb, max_init_points=100, seed=7)

    assert summary_a.n_points_written == 100
    assert summary_a.n_points_source == 1000

    scene_a = colmap_io.load_colmap_model(dir_a)
    scene_b = colmap_io.load_colmap_model(dir_b)
    xyz_a = np.array([p.xyz for p in scene_a.points3D.values()])
    xyz_b = np.array([p.xyz for p in scene_b.points3D.values()])
    np.testing.assert_allclose(sorted(xyz_a.tolist()), sorted(xyz_b.tolist()))  # same seed -> same rows


def test_write_distill_colmap_model_no_subsample_when_under_cap(tmp_path: Path) -> None:
    plan = DistillCameraPlan(anchors=[_pose("IMG_0.jpg", 0.0, image_name="IMG_0.jpg")])
    xyz, rgb = _random_points(10)
    summary = write_distill_colmap_model(tmp_path / "sparse_txt", plan, xyz, rgb, max_init_points=1000)
    assert summary.n_points_written == 10


def test_write_distill_colmap_model_rgb_round_trips_to_uint8(tmp_path: Path) -> None:
    plan = DistillCameraPlan(anchors=[_pose("IMG_0.jpg", 0.0, image_name="IMG_0.jpg")])
    xyz = np.array([[1.0, 2.0, 3.0]])
    rgb = np.array([[1.0, 0.5, 0.0]])  # -> uint8 (255, 128 or 127, 0)
    sparse_dir = tmp_path / "sparse_txt"
    write_distill_colmap_model(sparse_dir, plan, xyz, rgb, max_init_points=None)

    scene = colmap_io.load_colmap_model(sparse_dir)
    (point,) = scene.points3D.values()
    np.testing.assert_allclose(point.xyz, [1.0, 2.0, 3.0])
    assert point.rgb[0] == 255
    assert point.rgb[2] == 0


def test_write_distill_colmap_model_raises_on_empty_plan(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_distill_colmap_model(tmp_path / "sparse_txt", DistillCameraPlan(), np.zeros((0, 3)), np.zeros((0, 3)))
