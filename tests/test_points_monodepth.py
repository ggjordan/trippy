"""Tests for trippy.points.monodepth.MonoDepthSource and trippy.points.depth_io.

Module: tests.test_points_monodepth
Invariants under test:
    1. Median-ratio scale alignment recovers a known synthetic scale error
       to within 1% on a fronto-parallel plane, with backprojected points
       landing back on that plane.
    2. The camera-to-world unprojection formula (`x_w = R^T (x_c - t)`)
       agrees to <1e-5 whether built from `trippy.geom.xform_a` (numpy) or
       `trippy.geom.xform_b` (torch), and round-trips back onto the
       originating pixel/depth through xform_a's forward transform --
       xform_a/xform_b themselves are out of scope for this task's file
       list, so both independent routes live here rather than in
       trippy/points/monodepth.py (see AGENTS.md "implement transforms
       twice").
    3. Voxel dedupe (MonoDepthSource reuses
       trippy.points.union's private helper) collapses exact duplicate
       points from processing the same image twice.
    4. trippy.points.depth_io parses a synthetic depth_batch.py-style
       output triple, and detects missing/malformed ones.
    5. `trippy depth-points` CLI: --run-depth prints the GPU command and
       exits 3 when depth outputs are missing; without it, builds and
       writes a PointSet .npz and exits 0.
All fixtures are synthetic (generated in-test into tmp_path); no photos
are committed (AGENTS.md test-fixture rule).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from trippy.constants import MONODEPTH_MIN_SCALE_MATCHES, PROVENANCE_MONODEPTH
from trippy.geom import xform_a, xform_b
from trippy.points import depth_io
from trippy.points.monodepth import MonoDepthSource
from trippy.points.source import PointSet

# --- synthetic scene: single PINHOLE camera at the world origin (R=I, t=0)
# looking at a fronto-parallel plane at world z = plane_depth. Sparse COLMAP
# points are placed exactly at chosen pixel centres on that plane so the
# expected median-ratio scale and backprojected world position are known
# in closed form. ---

_W, _H = 64, 48
_FX = _FY = float(_W)
_CX, _CY = _W / 2.0, _H / 2.0


def _write_pinhole_scene(
    scene_root: Path,
    name: str,
    plane_depth: float,
    margin: int = 4,
    step: int = 8,
    seed: int = 0,
) -> list[tuple[float, float, int]]:
    """Write a synthetic sparse_txt scene; returns [(u, v, point3d_id), ...]."""
    images_dir = scene_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(_H, _W, 3), dtype=np.uint8)
    PILImage.fromarray(rgb, mode="RGB").save(images_dir / name)

    sparse_dir = scene_root / "sparse_txt"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    (sparse_dir / "cameras.txt").write_text(f"1 PINHOLE {_W} {_H} {_FX} {_FY} {_CX} {_CY}\n")

    obs = []
    lines3d = []
    pid = 1
    for row in range(margin, _H - margin, step):
        for col in range(margin, _W - margin, step):
            u, v = col + 0.5, row + 0.5
            x = (u - _CX) / _FX * plane_depth
            y = (v - _CY) / _FY * plane_depth
            lines3d.append(f"{pid} {x} {y} {plane_depth} 128 128 128 0.1")
            obs.append((u, v, pid))
            pid += 1
    (sparse_dir / "points3D.txt").write_text("\n".join(lines3d) + "\n")

    pose_line = f"1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 {name}"
    tokens = [str(tok) for triple in obs for tok in triple]
    (sparse_dir / "images.txt").write_text(pose_line + "\n" + " ".join(tokens) + "\n")

    assert len(obs) >= MONODEPTH_MIN_SCALE_MATCHES
    return obs


def _write_depth_output(depth_dir: Path, stem: str, depth_value: float, valid: bool = True) -> None:
    depth_dir.mkdir(parents=True, exist_ok=True)
    depth = np.full((_H, _W), depth_value, dtype=np.float32)
    mask = np.full((_H, _W), valid, dtype=bool)
    meta = {
        "width": _W,
        "height": _H,
        "orig_width": _W,
        "orig_height": _H,
        "focal_length_px": _FX,
        "has_person_mask": False,
    }
    np.save(depth_dir / f"{stem}_depth.npy", depth)
    np.save(depth_dir / f"{stem}_mask.npy", mask)
    (depth_dir / f"{stem}_meta.json").write_text(json.dumps(meta))


# --- 1. median-ratio scale recovery + unprojection lands back on the plane ---


def test_monodepth_scale_recovery_within_one_percent(tmp_path: Path) -> None:
    plane_depth = 10.0
    scale_error = 1.7  # DepthPro under-predicts by this factor -> s should recover ~1.7

    scene_root = tmp_path / "scene"
    _write_pinhole_scene(scene_root, "IMG_0001.jpg", plane_depth)

    depth_dir = tmp_path / "depth"
    _write_depth_output(depth_dir, "IMG_0001", plane_depth / scale_error)

    source = MonoDepthSource(
        scene_root,
        ["IMG_0001.jpg"],
        _W,
        depth_dir,
        tmp_path / "cache",
        stride=4,
        voxel=None,
        conf0=0.35,
    )
    point_set = source.build()

    stats = source.describe()["per_image"][0]
    assert stats["scale"] == pytest.approx(scale_error, rel=0.01)
    assert stats["mad"] < 1e-6
    assert stats["valid_fraction"] == pytest.approx(1.0)
    assert stats["points_contributed"] == len(point_set) > 0

    # Fronto-parallel plane at R=I, t=0: world z == camera z == corrected depth.
    np.testing.assert_allclose(point_set.xyz[:, 2], plane_depth, atol=1e-2)
    assert np.all(point_set.conf0 == np.float32(0.35))
    assert np.all(point_set.provenance == PROVENANCE_MONODEPTH)


def test_monodepth_skips_frame_with_too_few_sparse_matches(tmp_path: Path) -> None:
    plane_depth = 10.0
    scene_root = tmp_path / "scene"
    obs = _write_pinhole_scene(scene_root, "IMG_0001.jpg", plane_depth)
    assert len(obs) >= MONODEPTH_MIN_SCALE_MATCHES

    depth_dir = tmp_path / "depth"
    # Mark everything invalid except (row=margin,col=margin)'s neighbourhood is irrelevant --
    # zero valid pixels means zero matches regardless of sparse point count.
    _write_depth_output(depth_dir, "IMG_0001", plane_depth, valid=False)

    source = MonoDepthSource(scene_root, ["IMG_0001.jpg"], _W, depth_dir, tmp_path / "cache", voxel=None)
    point_set = source.build()

    assert len(point_set) == 0
    stats = source.describe()["per_image"][0]
    assert stats["scale"] is None
    assert stats["n_matches"] == 0
    assert "skipped_reason" in stats


# --- 2. unprojection formula agrees whether built from xform_a or xform_b,
# and round-trips onto the same pixel/depth (bug-class insurance: xform_a.py
# and xform_b.py are out of scope for this task, so both independent
# unprojection routes are written here, not in trippy/points/monodepth.py). ---


def _random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


@pytest.mark.parametrize("seed", range(8))
def test_unprojection_agrees_via_xform_a_and_xform_b(seed: int) -> None:
    rng = np.random.default_rng(seed)
    q = _random_unit_quat(rng)
    t = rng.normal(size=3) * 2.0
    fx, fy, cx, cy = 800.0, 820.0, 400.0, 300.0
    col, row = 123.4, 55.2  # arbitrary continuous (already pixel-centre) coords
    z = 5.0 + rng.random()

    x_c = np.array([(col - cx) / fx * z, (row - cy) / fy * z, z])

    # Route A: numpy, xform_a's qvec2R.
    R_a = xform_a.qvec2R(q)
    x_w_a = (x_c - t) @ R_a  # x_w = R^T (x_c - t), row-vector form (see world_to_cam's x_c = x_w@R.T+t)

    # Route B: torch, xform_b's independently-derived qvec2R (axis-angle + Rodrigues).
    R_b = xform_b.qvec2R(torch.tensor(q, dtype=torch.float64)).numpy()
    x_w_b = (x_c - t) @ R_b

    np.testing.assert_allclose(x_w_a, x_w_b, atol=1e-5)

    # Round trip: xform_a's own forward transform recovers the source pixel/depth.
    xyz_c_check = xform_a.world_to_cam(R_a, t, x_w_a.reshape(1, 3))
    uv_check, depth_check = xform_a.project_pinhole(xyz_c_check, fx, fy, cx, cy)
    np.testing.assert_allclose(uv_check[0], [col, row], atol=1e-6)
    assert depth_check[0] == pytest.approx(z, abs=1e-6)


# --- 3. voxel dedupe (reuses UnionSource's helper) ---


def test_monodepth_voxel_dedupe_collapses_exact_duplicates(tmp_path: Path) -> None:
    plane_depth = 10.0
    scene_root = tmp_path / "scene"
    _write_pinhole_scene(scene_root, "IMG_0001.jpg", plane_depth)
    depth_dir = tmp_path / "depth"
    _write_depth_output(depth_dir, "IMG_0001", plane_depth)

    names = ["IMG_0001.jpg", "IMG_0001.jpg"]  # same image processed twice -> bit-identical duplicates
    no_dedupe = MonoDepthSource(scene_root, names, _W, depth_dir, tmp_path / "cache", voxel=None)
    ps_no_dedupe = no_dedupe.build()

    deduped = MonoDepthSource(scene_root, names, _W, depth_dir, tmp_path / "cache", voxel=1e-4)
    ps_deduped = deduped.build()

    assert len(ps_no_dedupe) % 2 == 0
    assert len(ps_deduped) == len(ps_no_dedupe) // 2 > 0


# --- 4. depth_io parsing ---


def test_read_depth_output_parses_synthetic_triple(tmp_path: Path) -> None:
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    depth = np.random.default_rng(0).random((10, 12)).astype(np.float32)
    mask = np.ones((10, 12), dtype=bool)
    mask[0, 0] = False
    meta = {
        "width": 12,
        "height": 10,
        "orig_width": 12,
        "orig_height": 10,
        "focal_length_px": 500.0,
        "has_person_mask": False,
    }
    np.save(depth_dir / "abc_depth.npy", depth)
    np.save(depth_dir / "abc_mask.npy", mask)
    (depth_dir / "abc_meta.json").write_text(json.dumps(meta))

    out = depth_io.read_depth_output(depth_dir, "abc")
    np.testing.assert_array_equal(out.depth, depth)
    np.testing.assert_array_equal(out.mask, mask)
    assert out.meta == meta

    assert depth_io.depth_output_missing(depth_dir, ["abc", "missing"]) == ["missing"]


def test_read_depth_output_shape_mismatch_raises(tmp_path: Path) -> None:
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    np.save(depth_dir / "x_depth.npy", np.zeros((5, 5), dtype=np.float32))
    np.save(depth_dir / "x_mask.npy", np.ones((5, 5), dtype=bool))
    (depth_dir / "x_meta.json").write_text(json.dumps({"width": 9, "height": 9}))
    with pytest.raises(ValueError):
        depth_io.read_depth_output(depth_dir, "x")


def test_read_depth_output_missing_file_raises(tmp_path: Path) -> None:
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        depth_io.read_depth_output(depth_dir, "nope")


# --- 5. CLI exit codes ---


def test_cli_depth_points_run_depth_missing_exits_3(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    _write_pinhole_scene(scene_root, "IMG_0001.jpg", 10.0)
    depth_dir = tmp_path / "depth"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "trippy.cli",
            "depth-points",
            "--scene",
            str(scene_root),
            "--images",
            "IMG_0001.jpg",
            "--width",
            str(_W),
            "--depth-dir",
            str(depth_dir),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--run-depth",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert "gpu_submit.sh" in result.stdout
    assert (depth_dir / "manifest.json").exists()


def test_cli_depth_points_run_depth_present_exits_0(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    _write_pinhole_scene(scene_root, "IMG_0001.jpg", 10.0)
    depth_dir = tmp_path / "depth"
    _write_depth_output(depth_dir, "IMG_0001", 10.0)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "trippy.cli",
            "depth-points",
            "--scene",
            str(scene_root),
            "--images",
            "IMG_0001.jpg",
            "--width",
            str(_W),
            "--depth-dir",
            str(depth_dir),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--run-depth",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_depth_points_builds_pointset_and_writes_npz(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    _write_pinhole_scene(scene_root, "IMG_0001.jpg", 10.0)
    depth_dir = tmp_path / "depth"
    _write_depth_output(depth_dir, "IMG_0001", 10.0)
    out_path = tmp_path / "points.npz"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "trippy.cli",
            "depth-points",
            "--scene",
            str(scene_root),
            "--images",
            "IMG_0001.jpg",
            "--width",
            str(_W),
            "--depth-dir",
            str(depth_dir),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert out_path.exists()
    assert out_path.with_suffix(".summary.json").exists()

    point_set = PointSet.load_npz(out_path)
    assert len(point_set) > 0
    assert np.all(point_set.provenance == PROVENANCE_MONODEPTH)
