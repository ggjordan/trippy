"""End-to-end subprocess test for `python -m trippy.cli render`.

Module: tests.test_cli_render
Invariants under test: the render subcommand runs against a synthetic
2-camera COLMAP scene and a synthetic 3DGS PLY (never a real Splats scene),
exits 0 on --device cpu, and writes the expected output tree. TRIPPY_OUTPUT
is overridden to a tmp directory for the subprocess so the SceneDataset
cache never touches the real (shared, gitignored) output/ tree.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from test_scene_dataset import _synthetic_gradient, _write_txt_scene

WIDTH = 48


def _write_synthetic_gaussian_ply(path: Path, n: int = 150, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)
    z = rng.uniform(2.0, 3.0, size=(n, 1)).astype(np.float32)
    xyz = np.concatenate([xy, z], axis=1)
    f_dc = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    opacity = np.full(n, 2.0, dtype=np.float32)
    scale = np.full((n, 3), np.log(0.15), dtype=np.float32)

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


def test_cli_render_synthetic_scene_cpu(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    images = [
        ("cam_a.jpg", _synthetic_gradient(WIDTH, WIDTH, seed=20)),
        ("cam_b.jpg", _synthetic_gradient(WIDTH, WIDTH, seed=21)),
    ]
    _write_txt_scene(scene_root, images)

    ply_path = tmp_path / "points.ply"
    _write_synthetic_gaussian_ply(ply_path)

    out_dir = tmp_path / "out"
    trippy_output = tmp_path / "trippy_output"

    env = dict(os.environ)
    env["TRIPPY_OUTPUT"] = str(trippy_output)
    env["TRIPS_DEVICE"] = "cpu"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "trippy.cli",
            "render",
            "--scene",
            str(scene_root),
            "--points",
            "gaussian",
            "--ply",
            str(ply_path),
            "--width",
            str(WIDTH),
            "--frames",
            "cam_a.jpg,cam_b.jpg",
            "--mode",
            "trilinear",
            "--layers",
            "3",
            "--out",
            str(out_dir),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "JSON:" in result.stdout
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "summary_sheet.png").exists()
    assert (out_dir / "README.md").exists()

    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert len(metrics["frames"]) == 2

    # Cache went to the overridden TRIPPY_OUTPUT, never the shared/real output/.
    assert (trippy_output / "cache" / "scene" / f"w{WIDTH}").exists()
