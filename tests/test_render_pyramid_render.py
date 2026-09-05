"""Tests for trippy.render.pyramid_render: the `trippy render` orchestration.

Module: tests.test_render_pyramid_render
Invariants under test: render_frames() on a tiny synthetic 2-camera,
300-point scene (never a real capture -- AGENTS.md section 6 "test fixtures
must be synthetic only") produces, on CPU:
    - per-frame photo/level/coverage/depth PNGs at the expected shapes,
    - a per-frame sheet and one summary sheet,
    - metrics.json with a non-negative emit/sort/blend/total timing
      breakdown and plausible fragment/point counts,
    - a README.md documenting the command.
Related docs: docs/ARCHITECTURE.md (forward pass data flow);
    experiments/EXP-0001-forward-pyramid/README.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image as PILImage
from plyfile import PlyData, PlyElement
from test_scene_dataset import _synthetic_gradient, _write_txt_scene

from trippy.render import pyramid_render

WIDTH = 64
NUM_LAYERS = 3
NUM_POINTS = 300


def _write_synthetic_gaussian_ply(path: Path, n: int = NUM_POINTS, seed: int = 0) -> None:
    """A small binary 3DGS PLY whose points project inside both test cameras.

    World xy in [-1, 1], world z in [2, 3]; combined with the synthetic
    scene's identity-rotation, tvec=(0, 0, image_index) cameras (see
    tests.test_scene_dataset._write_txt_scene), camera-space depth is
    [3, 4] for the first image and [4, 5] for the second -- both positive
    and well inside the fx=fy=WIDTH, cx=cy=WIDTH/2 frustum.
    """
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)
    z = rng.uniform(2.0, 3.0, size=(n, 1)).astype(np.float32)
    xyz = np.concatenate([xy, z], axis=1)
    f_dc = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    opacity = np.full(n, 2.0, dtype=np.float32)  # sigmoid(2) ~= 0.88
    scale = np.full((n, 3), np.log(0.15), dtype=np.float32)  # size_px ~= 2-3 at these depths

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
    ]  # fmt: skip
    verts = np.zeros(n, dtype=dtype)
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["f_dc_0"], verts["f_dc_1"], verts["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    verts["opacity"] = opacity
    verts["scale_0"], verts["scale_1"], verts["scale_2"] = scale[:, 0], scale[:, 1], scale[:, 2]

    el = PlyElement.describe(verts, "vertex")
    PlyData([el], text=False).write(str(path))


def _build_scene(scene_root: Path) -> None:
    # Order matters: _write_txt_scene assigns tvec=(0, 0, i) by list position
    # (1-indexed), not by name -- cam_a.jpg gets tvec_z=1, cam_b.jpg tvec_z=2.
    images = [
        ("cam_a.jpg", _synthetic_gradient(WIDTH, WIDTH, seed=10)),
        ("cam_b.jpg", _synthetic_gradient(WIDTH, WIDTH, seed=11)),
    ]
    _write_txt_scene(scene_root, images)


def _render(tmp_path: Path) -> tuple[Path, dict]:
    scene_root = tmp_path / "scene"
    _build_scene(scene_root)
    ply_path = tmp_path / "points.ply"
    _write_synthetic_gaussian_ply(ply_path)

    out_dir = tmp_path / "out"
    cache_root = tmp_path / "cache"

    metrics = pyramid_render.render_frames(
        scene_root=scene_root,
        ply_path=ply_path,
        frame_names=["cam_a.jpg", "cam_b.jpg"],
        width=WIDTH,
        out_dir=out_dir,
        device=torch.device("cpu"),
        mode="trilinear",
        num_layers=NUM_LAYERS,
        cache_root=cache_root,
        command="trippy render --scene scene --ply points.ply --frames cam_a.jpg,cam_b.jpg",
    )
    return out_dir, metrics


def test_render_frames_metrics_shape_and_values(tmp_path: Path) -> None:
    _out_dir, metrics = _render(tmp_path)

    assert metrics["num_layers"] == NUM_LAYERS
    assert metrics["width"] == WIDTH
    assert metrics["num_points_total"] == NUM_POINTS
    assert len(metrics["frames"]) == 2
    assert [f["name"] for f in metrics["frames"]] == ["cam_a.jpg", "cam_b.jpg"]

    for frame in metrics["frames"]:
        tm = frame["timing_ms"]
        for key in ("emit", "sort", "blend", "total"):
            assert tm[key] >= 0.0
        assert tm["total"] >= tm["emit"]
        assert frame["num_fragments"] >= 0
        assert 0 <= frame["points_visible"] <= NUM_POINTS
        assert frame["image_hw"] == [WIDTH, WIDTH]
        cov = frame["coverage"]
        assert 0.0 <= cov["mean_full"] <= 1.0
        assert 0.0 <= cov["mean_center"] <= 1.0
        assert cov["center_frac"] == pytest.approx(0.5)

    # At these depths/positions essentially every point should be visible.
    assert metrics["frames"][0]["points_visible"] > 0
    assert metrics["frames"][0]["num_fragments"] > 0


def test_render_frames_writes_expected_files_and_shapes(tmp_path: Path) -> None:
    out_dir, metrics = _render(tmp_path)

    assert (out_dir / "metrics.json").exists()
    on_disk = json.loads((out_dir / "metrics.json").read_text())
    assert on_disk == metrics

    readme = (out_dir / "README.md").read_text()
    assert "trippy render" in readme
    assert "cam_a.jpg" in readme

    summary = np.array(PILImage.open(out_dir / "summary_sheet.png"))
    assert summary.ndim == 3 and summary.shape[2] == 3

    layer_sizes = {0: (WIDTH, WIDTH), 1: (32, 32), 2: (16, 16)}
    for name in ("cam_a", "cam_b"):
        frame_dir = out_dir / name
        photo = np.array(PILImage.open(frame_dir / "photo.png"))
        assert photo.shape == (WIDTH, WIDTH, 3)

        for level, (h, w) in layer_sizes.items():
            level_img = np.array(PILImage.open(frame_dir / f"level_{level}.png"))
            assert level_img.shape == (h, w, 3)

        coverage = np.array(PILImage.open(frame_dir / "coverage.png"))
        depth = np.array(PILImage.open(frame_dir / "depth.png"))
        assert coverage.shape == (WIDTH, WIDTH, 3)
        assert depth.shape == (WIDTH, WIDTH, 3)

        sheet = np.array(PILImage.open(frame_dir / "sheet.png"))
        assert sheet.ndim == 3 and sheet.shape[2] == 3


def test_render_frames_rejects_unknown_frame_name(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    _build_scene(scene_root)
    ply_path = tmp_path / "points.ply"
    _write_synthetic_gaussian_ply(ply_path)

    with pytest.raises(KeyError):
        pyramid_render.render_frames(
            scene_root=scene_root,
            ply_path=ply_path,
            frame_names=["does_not_exist.jpg"],
            width=WIDTH,
            out_dir=tmp_path / "out",
            device=torch.device("cpu"),
            num_layers=NUM_LAYERS,
            cache_root=tmp_path / "cache",
        )
