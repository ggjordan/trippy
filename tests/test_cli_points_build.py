"""End-to-end subprocess test for `python -m trippy.cli points-build`.

Module: tests.test_cli_points_build
Invariants under test: `points-build` reads a YAML file whose root is a
    `trippy.train.config.PointSourceConfig` (the same schema as a
    TrainConfig's `point_source:` block), builds the described source, and
    writes both a `.npz` (loadable via `PointSet.load_npz`) and a
    `<name>.summary.json` sidecar -- exercised here with a "union" of a
    synthetic "gaussian" PLY leaf and a synthetic "npz" leaf (standing in
    for a MonoDepth point set built by an earlier command), with a voxel
    dedupe collapsing one deliberately-colliding pair of points. Never
    touches a real Splats scene or MPS.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from plyfile import PlyData, PlyElement

from trippy.points.source import PointSet


def _write_synthetic_gaussian_ply(path: Path, n: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-5.0, 5.0, size=(n, 3)).astype(np.float32)
    f_dc = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    opacity = np.full(n, 5.0, dtype=np.float32)  # sigmoid(5) ~= 0.993
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
    return xyz


def _write_synthetic_monodepth_npz(path: Path, n: int, colliding_xyz: np.ndarray) -> None:
    """Writes an npz whose first `len(colliding_xyz)` points sit in the same
    voxel cell as the gaussian PLY's first points, so the union's voxel
    dedupe has something to collapse."""
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-5.0, 5.0, size=(n, 3)).astype(np.float32)
    k = colliding_xyz.shape[0]
    xyz[:k] = colliding_xyz + 1e-4  # same voxel cell, distinct point

    ps = PointSet(
        xyz=xyz,
        size0=np.full(n, 0.05, dtype=np.float32),
        rgb0=rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32),
        conf0=np.full(n, 0.35, dtype=np.float32),  # lower than the gaussian leaf's ~0.99
        provenance=np.full(n, 2, dtype=np.uint8),  # PROVENANCE_MONODEPTH
    )
    ps.save_npz(path)


def _extract_json_line(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("JSON:"):
            return json.loads(line[len("JSON:") :])
    raise AssertionError(f"no JSON: line found in stdout:\n{stdout}")


def test_points_build_union_of_gaussian_and_npz(tmp_path: Path) -> None:
    n_gauss, n_mono, n_collide = 30, 25, 5
    ply_path = tmp_path / "gaussians.ply"
    xyz = _write_synthetic_gaussian_ply(ply_path, n=n_gauss)

    npz_path = tmp_path / "monodepth.npz"
    _write_synthetic_monodepth_npz(npz_path, n=n_mono, colliding_xyz=xyz[:n_collide])

    config_path = tmp_path / "union.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "type": "union",
                "voxel": 1.0,
                "sources": [
                    {"type": "gaussian", "path": str(ply_path), "min_opacity": 0.05, "size_mode": "scale"},
                    {"type": "npz", "path": str(npz_path)},
                ],
            }
        )
    )

    out_path = tmp_path / "union_out.npz"
    result = subprocess.run(
        [sys.executable, "-m", "trippy.cli", "points-build", "--config", str(config_path), "--out", str(out_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    payload = _extract_json_line(result.stdout)
    summary = payload["summary"]
    # n_collide gaussian/monodepth pairs share a voxel cell; the gaussian point (conf~0.99)
    # wins each, so the union is exactly n_gauss + n_mono - n_collide points.
    assert summary["count"] == n_gauss + n_mono - n_collide
    assert summary["provenance_histogram"]["gaussian"] == n_gauss
    assert summary["provenance_histogram"]["monodepth"] == n_mono - n_collide

    assert out_path.exists()
    summary_path = out_path.with_suffix(".summary.json")
    assert summary_path.exists()
    on_disk = json.loads(summary_path.read_text())
    assert on_disk["summary"]["count"] == summary["count"]

    loaded = PointSet.load_npz(out_path)
    assert len(loaded) == summary["count"]


def test_points_build_plain_npz_passthrough(tmp_path: Path) -> None:
    """type: npz with no union wrapper loads the PointSet verbatim (no dedupe)."""
    n = 12
    npz_path = tmp_path / "source.npz"
    _write_synthetic_monodepth_npz(npz_path, n=n, colliding_xyz=np.zeros((0, 3), dtype=np.float32))

    config_path = tmp_path / "npz.yaml"
    config_path.write_text(yaml.safe_dump({"type": "npz", "path": str(npz_path)}))

    out_path = tmp_path / "out.npz"
    result = subprocess.run(
        [sys.executable, "-m", "trippy.cli", "points-build", "--config", str(config_path), "--out", str(out_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    payload = _extract_json_line(result.stdout)
    assert payload["summary"]["count"] == n
