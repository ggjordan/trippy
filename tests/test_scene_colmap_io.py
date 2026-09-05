"""Tests for trippy.scene.colmap_io: binary vs text reader agreement.

Module: tests.test_scene_colmap_io
Invariants under test: `load_colmap_model` parses a COLMAP binary model
    (cameras.bin/images.bin/points3D.bin, written here directly with
    `struct` against the documented layout -- independent of
    trippy.scene.colmap_io's own writer-less reader) and a COLMAP text
    model describing the *same* scene into identical `ColmapScene`
    contents (up to the text format's known limitation: it carries no
    point3D track). All synthetic fixtures are written to `tmp_path`; no
    real scene data or committed fixtures are used.
Related docs: docs/ARCHITECTURE.md "Module overview" (trippy/scene);
    tests/conftest.py (splats_scene fixture, skips cleanly without
    ~/Splats).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from trippy.scene import colmap_io

# COLMAP camera model ids used by this test's binary writer, hardcoded
# independently of trippy.constants.COLMAP_CAMERA_MODEL_TABLE (both are the
# same fixed COLMAP spec; keeping this literal here means a bug in that
# table would not silently make the writer and reader agree with each
# other for the wrong reason).
_MODEL_ID_PINHOLE = 1
_MODEL_ID_OPENCV = 4

_CAMERAS = [
    # (camera_id, model_name, model_id, width, height, params)
    (1, "PINHOLE", _MODEL_ID_PINHOLE, 100, 80, [80.0, 80.0, 50.0, 40.0]),
    (2, "OPENCV", _MODEL_ID_OPENCV, 64, 48, [60.0, 60.0, 32.0, 24.0, 0.01, -0.002, 0.0005, -0.0003]),
]

_IMAGES = [
    # (image_id, qvec, tvec, camera_id, name, points2d[(x, y, point3d_id)])
    (1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, "a.jpg", [(10.5, 20.5, 1), (30.2, 10.1, -1)]),
    (2, (0.9938, 0.1108, 0.0, 0.0), (0.1, -0.2, 5.0), 2, "b.jpg", []),
    (3, (0.70710678, 0.0, 0.70710678, 0.0), (1.0, 2.0, 3.0), 1, "c.jpg", [(5.5, 5.5, 2), (6.5, 6.5, 3), (7.5, 7.5, -1)]),
]

_POINTS3D = [
    # (point3d_id, xyz, rgb, error, track[(image_id, point2d_idx)])
    (1, (1.0, 2.0, 3.0), (10, 20, 30), 0.5, [(1, 0)]),
    (2, (4.0, 5.0, 6.0), (40, 50, 60), 0.6, [(3, 0)]),
    (3, (7.0, 8.0, 9.0), (70, 80, 90), 0.7, [(3, 1)]),
    (4, (-1.0, -2.0, -3.0), (1, 2, 3), 0.1, []),
    (5, (0.0, 0.0, 0.0), (255, 255, 255), 0.0, []),
]


def _write_synthetic_bin(sparse_dir: Path) -> None:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    with open(sparse_dir / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(_CAMERAS)))
        for camera_id, _name, model_id, width, height, params in _CAMERAS:
            f.write(struct.pack("<iiQQ", camera_id, model_id, width, height))
            f.write(struct.pack("<" + "d" * len(params), *params))

    with open(sparse_dir / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(_IMAGES)))
        for image_id, qvec, tvec, camera_id, name, points2d in _IMAGES:
            f.write(struct.pack("<i", image_id))
            f.write(struct.pack("<dddd", *qvec))
            f.write(struct.pack("<ddd", *tvec))
            f.write(struct.pack("<i", camera_id))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", len(points2d)))
            f.writelines(struct.pack("<ddq", x, y, pid) for x, y, pid in points2d)

    with open(sparse_dir / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(_POINTS3D)))
        for point3d_id, xyz, rgb, error, track in _POINTS3D:
            f.write(struct.pack("<Q", point3d_id))
            f.write(struct.pack("<ddd", *xyz))
            f.write(struct.pack("<BBB", *rgb))
            f.write(struct.pack("<d", error))
            f.write(struct.pack("<Q", len(track)))
            for image_id, p2d_idx in track:
                f.write(struct.pack("<ii", image_id, p2d_idx))


def _write_synthetic_txt(sparse_dir: Path) -> None:
    sparse_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# Camera list with one line of data per camera:"]
    for camera_id, model_name, _model_id, width, height, params in _CAMERAS:
        params_str = " ".join(str(p) for p in params)
        lines.append(f"{camera_id} {model_name} {width} {height} {params_str}")
    (sparse_dir / "cameras.txt").write_text("\n".join(lines) + "\n")

    lines = ["# Image list with two lines of data per image:"]
    for image_id, qvec, tvec, camera_id, name, points2d in _IMAGES:
        qvec_str = " ".join(str(v) for v in qvec)
        tvec_str = " ".join(str(v) for v in tvec)
        lines.append(f"{image_id} {qvec_str} {tvec_str} {camera_id} {name}")
        # Zero-observation images still get a genuine (blank) POINTS2D line
        # -- see trippy.geom.xform_a.read_images_txt's docstring.
        p2d_str = " ".join(f"{x} {y} {pid}" for x, y, pid in points2d)
        lines.append(p2d_str)
    (sparse_dir / "images.txt").write_text("\n".join(lines) + "\n")

    lines = ["# 3D point list"]
    for point3d_id, xyz, rgb, error, track in _POINTS3D:
        xyz_str = " ".join(str(v) for v in xyz)
        rgb_str = " ".join(str(v) for v in rgb)
        # Real COLMAP text also appends the track; xform_a.read_points3d_txt
        # deliberately ignores it (see its docstring), so writing it here
        # (unused by the reader) keeps this fixture format-realistic.
        track_str = " ".join(f"{iid} {pidx}" for iid, pidx in track)
        line = f"{point3d_id} {xyz_str} {rgb_str} {error}"
        if track_str:
            line += " " + track_str
        lines.append(line)
    (sparse_dir / "points3D.txt").write_text("\n".join(lines) + "\n")


def test_binary_and_text_readers_agree(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    txt_dir = tmp_path / "txt"
    _write_synthetic_bin(bin_dir)
    _write_synthetic_txt(txt_dir)

    scene_bin = colmap_io.load_colmap_model(bin_dir)
    scene_txt = colmap_io.load_colmap_model(txt_dir)

    assert set(scene_bin.cameras.keys()) == set(scene_txt.cameras.keys()) == {1, 2}
    for cid in scene_bin.cameras:
        cb, ct = scene_bin.cameras[cid], scene_txt.cameras[cid]
        assert cb.model == ct.model
        assert cb.width == ct.width
        assert cb.height == ct.height
        np.testing.assert_allclose(cb.params, ct.params)

    assert set(scene_bin.images.keys()) == set(scene_txt.images.keys()) == {1, 2, 3}
    for iid in scene_bin.images:
        ib, it = scene_bin.images[iid], scene_txt.images[iid]
        assert ib.name == it.name
        assert ib.camera_id == it.camera_id
        np.testing.assert_allclose(ib.qvec, it.qvec)
        np.testing.assert_allclose(ib.tvec, it.tvec)
        np.testing.assert_allclose(ib.xys, it.xys)
        np.testing.assert_array_equal(ib.point3D_ids, it.point3D_ids)

    # points3D: xyz/rgb/error must agree; track is a binary-only field (the
    # text reader deliberately drops it, see xform_a.read_points3d_txt).
    assert set(scene_bin.points3D.keys()) == set(scene_txt.points3D.keys()) == {1, 2, 3, 4, 5}
    for pid in scene_bin.points3D:
        pb, pt = scene_bin.points3D[pid], scene_txt.points3D[pid]
        np.testing.assert_allclose(pb.xyz, pt.xyz)
        np.testing.assert_array_equal(pb.rgb, pt.rgb)
        assert pb.error == pytest.approx(pt.error)
        assert pt.track == []
    assert scene_bin.points3D[1].track == [(1, 0)]
    assert scene_bin.points3D[2].track == [(3, 0)]
    assert scene_bin.points3D[3].track == [(3, 1)]


def test_intrinsics_and_distortion() -> None:
    cam_pinhole = colmap_io.Camera(model="PINHOLE", width=100, height=80, params=[80.0, 80.0, 50.0, 40.0])
    assert colmap_io.intrinsics(cam_pinhole) == (80.0, 80.0, 50.0, 40.0)
    assert colmap_io.distortion(cam_pinhole) == (0.0, 0.0, 0.0, 0.0)

    cam_opencv = colmap_io.Camera(
        model="OPENCV", width=64, height=48, params=[60.0, 60.0, 32.0, 24.0, 0.01, -0.002, 0.0005, -0.0003]
    )
    assert colmap_io.intrinsics(cam_opencv) == (60.0, 60.0, 32.0, 24.0)
    assert colmap_io.distortion(cam_opencv) == (0.01, -0.002, 0.0005, -0.0003)

    with pytest.raises(ValueError):
        colmap_io.distortion(colmap_io.Camera(model="FOV", width=1, height=1, params=[1, 1, 1, 1, 1]))


def test_load_colmap_model_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        colmap_io.load_colmap_model(tmp_path / "does_not_exist")


# --- real-scene integration test (skips cleanly without ~/Splats) ---


def test_colmap_io_real_scene_kk_coherent(splats_scene: Path) -> None:
    """Cross-check bin vs txt readers on the real kk-coherent scene.

    `splats_scene` (see tests/conftest.py) points at
    ~/Splats/scenes/karekare/kk-coherent/sparse_txt; its parent is the
    scene root, which also has a sparse/0 binary export.
    """
    scene_root = splats_scene.parent
    bin_dir = scene_root / "sparse" / "0"
    if not bin_dir.exists():
        pytest.skip(f"no sparse/0 binary export at {bin_dir}")

    scene_txt = colmap_io.load_colmap_model(splats_scene)
    scene_bin = colmap_io.load_colmap_model(bin_dir)

    assert len(scene_bin.cameras) == 6
    assert len(scene_txt.cameras) == 6

    # docs/PLAN-2026-09-05.md says "kk-coherent: 238 images", but that
    # counts raw captures under images/ -- COLMAP only registers a subset
    # during SfM. Check both numbers explicitly rather than assuming they
    # are the same.
    images_dir = scene_root / "images"
    raw_image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")]
    assert len(raw_image_files) == 238, "expected 238 raw capture files under images/"
    assert len(scene_bin.images) == len(scene_txt.images), "bin/txt registered-image counts disagree"
    assert len(scene_bin.images) > 0

    names_bin = scene_bin.images_by_name()
    names_txt = scene_txt.images_by_name()
    assert set(names_bin.keys()) == set(names_txt.keys())

    # Spot-check one image's pose agrees between formats within COLMAP's
    # text-export rounding.
    sample_name = min(names_bin.keys())
    im_bin, im_txt = names_bin[sample_name], names_txt[sample_name]
    np.testing.assert_allclose(im_bin.qvec, im_txt.qvec, atol=1e-6)
    np.testing.assert_allclose(im_bin.tvec, im_txt.tvec, atol=1e-6)
