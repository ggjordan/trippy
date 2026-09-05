"""End-to-end subprocess test for `python -m trippy.cli density`.

Module: tests.test_cli_density
Invariants under test: the density subcommand runs against a synthetic
    3DGS PLY (never a real Splats scene), exits 0, and prints a `JSON:`
    -prefixed line whose payload is the PointSet.summary() dict.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def _write_synthetic_gaussian_ply(path: Path, n: int = 40, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-5.0, 5.0, size=(n, 3)).astype(np.float32)
    f_dc = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    opacity = np.full(n, 5.0, dtype=np.float32)  # sigmoid(5) ~= 0.993, all kept at default min_opacity
    scale = np.full((n, 3), np.log(0.02), dtype=np.float32)

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


def _extract_json_line(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("JSON:"):
            return json.loads(line[len("JSON:") :])
    raise AssertionError(f"no JSON: line found in stdout:\n{stdout}")


def test_cli_density_gaussian_synthetic_ply(tmp_path: Path) -> None:
    ply_path = tmp_path / "gaussians.ply"
    n = 40
    _write_synthetic_gaussian_ply(ply_path, n=n)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "trippy.cli",
            "density",
            "--source",
            "gaussian",
            "--path",
            str(ply_path),
            "--min-opacity",
            "0.05",
            "--size-mode",
            "scale",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    payload = _extract_json_line(result.stdout)
    assert payload["count"] == n
    assert len(payload["bbox_min"]) == 3
    assert len(payload["bbox_max"]) == 3
    assert payload["provenance_histogram"] == {"gaussian": n}


def test_cli_density_writes_out_file(tmp_path: Path) -> None:
    ply_path = tmp_path / "gaussians.ply"
    _write_synthetic_gaussian_ply(ply_path, n=20)
    out_path = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "trippy.cli",
            "density",
            "--source",
            "gaussian",
            "--path",
            str(ply_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["count"] == 20
