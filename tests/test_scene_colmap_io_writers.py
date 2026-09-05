"""Tests for trippy.scene.colmap_io's text writers: round-trip against the readers.

Module: tests.test_scene_colmap_io_writers
Invariants under test: write_cameras_txt/write_images_txt/write_points3d_txt/
    save_colmap_model_txt are the exact textual inverse of
    load_colmap_model's text-format path -- writing a ColmapScene then
    reading it back reproduces every field the text format itself carries
    (cameras, image poses/names/camera_ids, point xyz/rgb/error); a
    zero-observation image still gets a genuine blank POINTS2D line (never
    omitted); the track field round-trips through the writer (unlike the
    reader, which drops it -- see read_points3d_txt's own docstring) but is
    not required to be non-empty for a point to write/read correctly.
Fixture: synthetic ColmapScene objects only (AGENTS.md: test fixtures must
    be synthetic, never real Splats scenes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trippy.scene import colmap_io
from trippy.scene.colmap_io import Camera, ColmapScene, Image, Point3D


def _synthetic_scene() -> ColmapScene:
    cameras = {
        1: Camera(model="PINHOLE", width=100, height=80, params=[80.0, 80.0, 50.0, 40.0]),
        2: Camera(model="OPENCV", width=64, height=48, params=[60.0, 60.0, 32.0, 24.0, 0.01, -0.002, 0.0005, -0.0003]),
    }
    images = {
        1: Image(
            name="a.jpg",
            camera_id=1,
            qvec=np.array([1.0, 0.0, 0.0, 0.0]),
            tvec=np.array([0.0, 0.0, 0.0]),
            xys=np.array([[10.5, 20.5], [30.2, 10.1]]),
            point3D_ids=np.array([1, -1]),
        ),
        2: Image(
            name="b.jpg",
            camera_id=2,
            qvec=np.array([0.9938, 0.1108, 0.0, 0.0]),
            tvec=np.array([0.1, -0.2, 5.0]),
            xys=np.zeros((0, 2)),
            point3D_ids=np.zeros((0,), dtype=np.int64),
        ),
    }
    points3D = {
        1: Point3D(xyz=np.array([1.0, 2.0, 3.0]), rgb=np.array([10, 20, 30], dtype=np.uint8), error=0.5, track=[(1, 0)]),
        2: Point3D(xyz=np.array([-1.0, -2.0, -3.0]), rgb=np.array([1, 2, 3], dtype=np.uint8), error=0.0, track=[]),
    }
    return ColmapScene(cameras=cameras, images=images, points3D=points3D)


def test_write_then_load_round_trips_every_text_field(tmp_path: Path) -> None:
    scene = _synthetic_scene()
    sparse_dir = tmp_path / "sparse_txt"
    result = colmap_io.save_colmap_model_txt(sparse_dir, scene)
    assert result == sparse_dir
    assert (sparse_dir / "cameras.txt").exists()
    assert (sparse_dir / "images.txt").exists()
    assert (sparse_dir / "points3D.txt").exists()

    loaded = colmap_io.load_colmap_model(sparse_dir)

    assert set(loaded.cameras.keys()) == {1, 2}
    for cid, cam in scene.cameras.items():
        got = loaded.cameras[cid]
        assert got.model == cam.model
        assert got.width == cam.width
        assert got.height == cam.height
        np.testing.assert_allclose(got.params, cam.params)

    assert set(loaded.images.keys()) == {1, 2}
    for iid, im in scene.images.items():
        got = loaded.images[iid]
        assert got.name == im.name
        assert got.camera_id == im.camera_id
        np.testing.assert_allclose(got.qvec, im.qvec, atol=1e-9)
        np.testing.assert_allclose(got.tvec, im.tvec, atol=1e-9)
        np.testing.assert_allclose(got.xys, im.xys, atol=1e-9)
        np.testing.assert_array_equal(got.point3D_ids, im.point3D_ids)

    assert set(loaded.points3D.keys()) == {1, 2}
    for pid, p in scene.points3D.items():
        got = loaded.points3D[pid]
        np.testing.assert_allclose(got.xyz, p.xyz, atol=1e-9)
        np.testing.assert_array_equal(got.rgb, p.rgb)
        assert got.error == p.error
        # The text reader drops the track (see read_points3d_txt's docstring);
        # the writer still emits it (test_write_points3d_txt_writes_track below).
        assert got.track == []


def test_write_images_txt_writes_blank_points2d_line_for_zero_observations(tmp_path: Path) -> None:
    scene = _synthetic_scene()
    path = colmap_io.write_images_txt(tmp_path / "images.txt", scene.images)
    lines = path.read_text().splitlines()
    # Find image "b.jpg"'s pose line (camera_id 2, zero observations) and assert
    # the very next line exists and is blank (not omitted).
    pose_line_idx = next(i for i, line in enumerate(lines) if line.endswith(" 2 b.jpg"))
    assert lines[pose_line_idx + 1] == ""


def test_write_points3d_txt_writes_track(tmp_path: Path) -> None:
    scene = _synthetic_scene()
    path = colmap_io.write_points3d_txt(tmp_path / "points3D.txt", scene.points3D)
    text = path.read_text()
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    point1_line = next(line for line in lines if line.startswith("1 "))
    assert "1 0" in point1_line  # track (image_id=1, point2d_idx=0) appended after error


def test_write_cameras_txt_sorted_and_creates_parent_dirs(tmp_path: Path) -> None:
    scene = _synthetic_scene()
    nested = tmp_path / "a" / "b" / "cameras.txt"
    path = colmap_io.write_cameras_txt(nested, scene.cameras)
    assert path == nested
    lines = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    ids = [int(line.split()[0]) for line in lines]
    assert ids == sorted(ids)
