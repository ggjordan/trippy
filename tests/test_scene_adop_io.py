"""Tests for trippy.scene.adop_io (ADOP scene directory reader).

Fixtures are synthetic only (AGENTS.md Sec. 6): the point-cloud test writes a
file in Saiga's own compressed UnifiedMesh format with
`adop_io.write_point_cloud_bin` and reads it back, so the byte layout is
exercised end to end without shipping any real scene data.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest
import torch

from trippy.constants import ADOP_POINT_CLOUD_MAGIC
from trippy.scene import adop_io

# --- ini parsing ---------------------------------------------------------


def test_read_ini_sections_keys_and_comments(tmp_path):
    path = tmp_path / "dataset.ini"
    lines = [
        "[SceneDatasetParams]",
        "# fx fy cx cy s",
        "file_model = ",
        "camera_files = camera0.ini camera1.ini",
        "render_scale = 0.5",
        "",
        "[Other]",
        "znear = 0.1000000015",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    ini = adop_io.read_ini(path)
    assert ini["SceneDatasetParams"]["camera_files"] == "camera0.ini camera1.ini"
    assert ini["SceneDatasetParams"]["file_model"] == ""
    assert ini["SceneDatasetParams"]["render_scale"] == "0.5"
    assert ini["Other"]["znear"] == "0.1000000015"


# --- quaternions / poses -------------------------------------------------


def test_quat_order_round_trip():
    q_xyzw = np.array([0.1, 0.2, 0.3, 0.9])
    assert np.allclose(adop_io.quat_wxyz_to_xyzw(adop_io.quat_xyzw_to_wxyz(q_xyzw)), q_xyzw)


def test_qvec2R_agrees_with_xform_b():
    """AGENTS.md Sec. 7: geometry implemented twice, agreement tested."""
    from trippy.geom import xform_b

    rng = np.random.default_rng(20260906)
    for _ in range(32):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        ours = adop_io.qvec2R(q)
        theirs = xform_b.qvec2R(torch.tensor(q, dtype=torch.float64)).numpy()
        assert np.allclose(ours, theirs, atol=1e-10), (q, ours, theirs)


def test_qvec2R_is_a_rotation():
    rng = np.random.default_rng(7)
    for _ in range(16):
        q = rng.normal(size=4)
        R = adop_io.qvec2R(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_pose_conversion_hand_checked_example():
    """A 90-degree yaw, worked by hand.

    ADOP stores camera-to-world (`SceneData.cpp:458-469`, xyzw). Take
    `R_c2w = Rz(+90)` (quaternion xyzw = (0, 0, sin45, cos45)) and camera
    centre `C = (1, 2, 3)`. Then `R_w2c = Rz(-90)`, which maps
    `(x, y, z) -> (y, -x, z)`, and `t_w2c = -R_w2c @ C = -(2, -1, 3)`.
    """
    root2 = np.sqrt(0.5)
    q_xyzw = np.array([0.0, 0.0, root2, root2])
    t_c2w = np.array([1.0, 2.0, 3.0])

    q_w2c, t_w2c = adop_io.pose_c2w_xyzw_to_w2c_wxyz(q_xyzw, t_c2w)
    R_w2c = adop_io.qvec2R(q_w2c)

    assert np.allclose(R_w2c @ np.array([1.0, 0.0, 0.0]), [0.0, -1.0, 0.0], atol=1e-12)
    assert np.allclose(R_w2c @ np.array([0.0, 1.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(t_w2c, [-2.0, 1.0, -3.0], atol=1e-12)
    # The camera centre must map to the camera origin.
    assert np.allclose(R_w2c @ t_c2w + t_w2c, np.zeros(3), atol=1e-12)


def test_pose_conversion_round_trips_both_ways():
    rng = np.random.default_rng(11)
    for _ in range(32):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        if q[3] < 0:  # fix the sign so the xyzw round trip is bit-comparable
            q = -q
        t = rng.normal(size=3) * 5.0
        q_w2c, t_w2c = adop_io.pose_c2w_xyzw_to_w2c_wxyz(q, t)
        q_back, t_back = adop_io.pose_w2c_wxyz_to_c2w_xyzw(q_w2c, t_w2c)
        assert np.allclose(q_back, q, atol=1e-12)
        assert np.allclose(t_back, t, atol=1e-12)


def test_read_poses_matches_manual_conversion(tmp_path):
    path = tmp_path / "poses.txt"
    # Row 2 is line 1 of the public tt_horse poses.txt.
    horse_row = (
        "1.248415078221213e-02 1.515351343304283e-01 2.093144314751329e-02 "
        "9.881513668110107e-01 -5.572968439234115e-01 4.159452156592154e-01 -3.309433841025000e+00"
    )
    rows = ["0.0 0.0 0.0 1.0 1.0 2.0 3.0", horse_row]
    path.write_text("\n".join(rows), encoding="utf-8")
    q, t = adop_io.read_poses(path)
    assert q.shape == (2, 4) and t.shape == (2, 3)
    # Identity rotation: t_w2c = -C.
    assert np.allclose(q[0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(t[0], [-1.0, -2.0, -3.0])
    # Second row: this is line 1 of the public tt_horse poses.txt; TRIPS's own
    # PoseModule buffer for that frame is (xyzw) (-0.0125, -0.1515, -0.0209,
    # 0.9882), t = (-0.4769, -0.3338, 3.3312) -- see docs/TRIPS_REFERENCE.md 9b.
    assert np.allclose(
        adop_io.quat_wxyz_to_xyzw(q[1]),
        [-0.01248415, -0.15153513, -0.02093144, 0.98815137],
        atol=1e-7,
    )
    assert np.allclose(t[1], [-0.47694324, -0.33375021, 3.33122777], atol=1e-7)


# --- point_cloud.bin -----------------------------------------------------


def _synthetic_cloud(n: int, seed: int = 3) -> adop_io.AdopPointCloud:
    rng = np.random.default_rng(seed)
    return adop_io.AdopPointCloud(
        position=rng.normal(size=(n, 3)).astype(np.float32),
        normal=rng.normal(size=(n, 3)).astype(np.float32),
        color=rng.random((n, 4)).astype(np.float32),
        data=rng.random((n, 4)).astype(np.float32),
    )


def test_point_cloud_bin_round_trip(tmp_path):
    cloud = _synthetic_cloud(1234)
    path = adop_io.write_point_cloud_bin(tmp_path / "point_cloud.bin", cloud)
    back = adop_io.read_point_cloud_bin(path)
    assert len(back) == 1234
    for name in ("position", "normal", "color", "data"):
        assert np.array_equal(getattr(back, name), getattr(cloud, name)), name


def test_point_cloud_bin_header_is_saiga_compress(tmp_path):
    path = adop_io.write_point_cloud_bin(tmp_path / "pc.bin", _synthetic_cloud(7))
    raw = path.read_bytes()
    magic, _csize, dsize = struct.unpack_from("<QQQ", raw, 0)
    assert magic == ADOP_POINT_CLOUD_MAGIC
    assert len(zlib.decompress(raw[24:])) == dsize
    # Field order: position count first, immediately after the header.
    blob = zlib.decompress(raw[24:])
    assert struct.unpack_from("<Q", blob, 0)[0] == 7


def test_point_cloud_bin_rejects_bad_magic(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(struct.pack("<QQQ", 0xDEADBEEF, 1, 1) + b"\x00" * 8)
    with pytest.raises(ValueError, match="magic"):
        adop_io.read_point_cloud_bin(path)


# --- whole-scene reader --------------------------------------------------


def _write_scene(root, num_images: int = 3, width: int = 64, height: int = 32) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    dataset_lines = [
        "[SceneDatasetParams]",
        "image_dir = scenes/synthetic/images/",
        "camera_files = camera0.ini",
        "file_point_cloud_compressed = point_cloud.bin",
        "render_scale = 0.5",
        "znear = 0.25",
        "zfar = 500",
        "scene_exposure_value = 0",
    ]
    (root / "dataset.ini").write_text("\n".join(dataset_lines), encoding="utf-8")
    (root / "camera0.ini").write_text(
        "\n".join(
            [
                "[SceneCameraParams]",
                f"w = {width}",
                f"h = {height}",
                "# fx fy cx cy s",
                f"K = 100 110 {width / 2} {height / 2} 0",
                "# 8 paramter distortion model. see distortion.h",
                "distortion = -0.1 0.02 0 0 0 0 0.001 -0.002",
            ]
        ),
        encoding="utf-8",
    )
    names = [f"{i:05d}.jpg" for i in range(1, num_images + 1)]
    (root / "images.txt").write_text("\n".join(names), encoding="utf-8")
    (root / "camera_indices.txt").write_text("\n".join(["0"] * num_images), encoding="utf-8")
    (root / "poses.txt").write_text(
        "\n".join(f"0 0 0 1 {i} {i + 1} {i + 2}" for i in range(num_images)), encoding="utf-8"
    )
    (root / "exposure.txt").write_text("\n".join(str(0.25 * i) for i in range(num_images)), encoding="utf-8")
    (root / "white_balance.txt").write_text("\n".join("1 1 1" for _ in range(num_images)), encoding="utf-8")
    (root / "masks.txt").write_text("\n".join("" for _ in range(num_images)), encoding="utf-8")


def test_load_adop_scene_and_view(tmp_path):
    root = tmp_path / "scene"
    _write_scene(root)
    scene = adop_io.load_adop_scene(root)

    assert len(scene) == 3
    assert scene.render_scale == 0.5
    assert scene.znear == 0.25
    assert scene.image_names[0] == "00001.jpg"
    assert scene.index_of("00002.jpg") == 1
    assert scene.cameras[0].width == 64 and scene.cameras[0].height == 32
    assert scene.cameras[0].distortion[6] == pytest.approx(0.001)
    assert np.allclose(scene.white_balance, 1.0)
    assert scene.exposure[1] == pytest.approx(0.25)
    assert scene.mask_names == ["", "", ""]

    view = scene.view(1)
    # render_scale 0.5 halves both the intrinsics and the buffer size.
    assert (view.height, view.width) == (16, 32)
    assert view.K[0, 0] == pytest.approx(50.0)
    assert view.K[0, 2] == pytest.approx(16.0)
    assert np.allclose(view.R, np.eye(3))
    assert np.allclose(view.t, [-1.0, -2.0, -3.0])
    assert view.image_path.name == "00002.jpg"
    assert view.mask_path is None

    # An explicit render scale overrides the scene's.
    full = scene.view(1, render_scale=1.0)
    assert (full.height, full.width) == (32, 64)
    assert full.K[0, 0] == pytest.approx(100.0)


def test_load_adop_scene_rejects_length_mismatch(tmp_path):
    root = tmp_path / "scene"
    _write_scene(root)
    (root / "poses.txt").write_text("0 0 0 1 0 0 0", encoding="utf-8")
    with pytest.raises(ValueError, match="poses.txt"):
        adop_io.load_adop_scene(root)


def test_load_adop_scene_defaults_missing_optional_files(tmp_path):
    root = tmp_path / "scene"
    _write_scene(root)
    for name in ("exposure.txt", "white_balance.txt", "masks.txt"):
        (root / name).unlink()
    scene = adop_io.load_adop_scene(root)
    assert np.allclose(scene.exposure, 0.0)
    assert np.allclose(scene.white_balance, 1.0)
    assert scene.mask_names == ["", "", ""]
