"""Tests for trippy.train.export: 3DGS PLY writer.

Module: tests.test_export_ply
Invariants under test: write_gaussian_ply's field mapping is the exact
    inverse of trippy.points.gaussian_ply.GaussianPlySource's read side
    (round-trip equality within float32 precision), the written header's
    property list matches the documented order exactly (with and without
    sh_degree=3's f_rest_* fields), and -- when available on this
    machine -- Splats' own extent_gate.py accepts a synthetic PLY written
    by this module without modification.
Fixture: synthetic random points only (AGENTS.md: test fixtures must be
    synthetic, never real Splats scenes).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from trippy.constants import PROVENANCE_GAUSSIAN, SH_DEGREE_THREE, SH_DEGREE_THREE_NUM_REST_COEFFS
from trippy.points.gaussian_ply import GaussianPlySource
from trippy.points.source import PointSet
from trippy.train.export import (
    export_pointset_ply,
    read_back_check,
    write_gaussian_ply,
    write_provenance_sidecar,
)

N = 200

_EXPECTED_BASE_HEADER = [
    "x", "y", "z",
    "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]  # fmt: skip

_SPLATS_EXTENT_GATE = Path("/Users/nzbirdranch/Splats/tools/tmp/extent-audit/extent_gate.py")
_SPLATS_ML_SHARP_PYTHON = Path("/Users/nzbirdranch/Splats/tools/ml-sharp/.venv/bin/python")


def _random_pointset(n: int = N, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """xyz/rgb/conf/size arrays kept well away from clamp/clip boundaries,
    so the export/import round trip is exact to float32 precision."""
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-5.0, 5.0, size=(n, 3)).astype(np.float32)
    rgb = rng.uniform(0.05, 0.95, size=(n, 3)).astype(np.float32)
    conf = rng.uniform(0.1, 0.9, size=(n,)).astype(np.float32)
    size = rng.uniform(0.01, 2.0, size=(n,)).astype(np.float32)
    return xyz, rgb, conf, size


def _read_header_props(path: Path) -> list[str]:
    props = []
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        assert f.readline().strip() == b"format binary_little_endian 1.0"
        line = f.readline()
        assert line.strip().startswith(b"element vertex")
        while True:
            line = f.readline().strip()
            if line == b"end_header":
                break
            tokens = line.split()
            assert tokens[0] == b"property"
            props.append(tokens[2].decode())
    return props


def test_round_trip_xyz_rgb_conf_size(tmp_path: Path) -> None:
    xyz, rgb, conf, size = _random_pointset()
    path = tmp_path / "export.ply"
    write_gaussian_ply(path, xyz, rgb, conf, size)

    source = GaussianPlySource(path, min_opacity=0.0, size_mode="scale")
    ps = source.build()

    assert len(ps) == N
    np.testing.assert_allclose(ps.xyz, xyz, atol=1e-5)
    np.testing.assert_allclose(ps.rgb0, rgb, atol=1e-5)
    np.testing.assert_allclose(ps.conf0, conf, atol=1e-5)
    np.testing.assert_allclose(ps.size0, size, rtol=1e-5, atol=1e-5)


def test_read_back_check_count_matches(tmp_path: Path) -> None:
    xyz, rgb, conf, size = _random_pointset()
    path = tmp_path / "export.ply"
    write_gaussian_ply(path, xyz, rgb, conf, size)
    assert read_back_check(path) == N


def test_header_field_list_default_sh_degree(tmp_path: Path) -> None:
    xyz, rgb, conf, size = _random_pointset(n=5)
    path = tmp_path / "export.ply"
    write_gaussian_ply(path, xyz, rgb, conf, size)
    assert _read_header_props(path) == _EXPECTED_BASE_HEADER


def test_header_field_list_sh_degree_three(tmp_path: Path) -> None:
    xyz, rgb, conf, size = _random_pointset(n=5)
    path = tmp_path / "export.ply"
    write_gaussian_ply(path, xyz, rgb, conf, size, sh_degree=SH_DEGREE_THREE)
    expected = _EXPECTED_BASE_HEADER + [f"f_rest_{i}" for i in range(SH_DEGREE_THREE_NUM_REST_COEFFS)]
    assert _read_header_props(path) == expected

    # f_rest_* must be present and all-zero; reading back must still work
    # (GaussianPlySource selects fields by name, ignoring the extras).
    assert read_back_check(path) == 5


def test_invalid_sh_degree_raises(tmp_path: Path) -> None:
    xyz, rgb, conf, size = _random_pointset(n=3)
    path = tmp_path / "export.ply"
    with pytest.raises(ValueError, match="sh_degree"):
        write_gaussian_ply(path, xyz, rgb, conf, size, sh_degree=2)


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    xyz, rgb, conf, size = _random_pointset(n=10)
    path = tmp_path / "export.ply"
    with pytest.raises(ValueError, match="rgb"):
        write_gaussian_ply(path, xyz, rgb[:-1], conf, size)


def test_non_positive_size_raises(tmp_path: Path) -> None:
    xyz, rgb, conf, size = _random_pointset(n=4)
    size[0] = 0.0
    path = tmp_path / "export.ply"
    with pytest.raises(ValueError, match="positive"):
        write_gaussian_ply(path, xyz, rgb, conf, size)


def test_export_pointset_ply_and_provenance_sidecar(tmp_path: Path) -> None:
    xyz, rgb, conf, size = _random_pointset(n=30)
    provenance = np.full(30, PROVENANCE_GAUSSIAN, dtype=np.uint8)
    ps = PointSet(xyz=xyz, size0=size, rgb0=rgb, conf0=conf, provenance=provenance)

    path = tmp_path / "export.ply"
    export_pointset_ply(path, ps)

    assert read_back_check(path) == 30
    sidecar = tmp_path / "export.ply.provenance.npy"
    assert sidecar.exists()
    np.testing.assert_array_equal(np.load(sidecar), provenance)


def test_write_provenance_sidecar_standalone(tmp_path: Path) -> None:
    path = tmp_path / "foo.ply"
    provenance = np.array([1, 1, 2, 3], dtype=np.uint8)
    sidecar = write_provenance_sidecar(path, provenance)
    assert sidecar == tmp_path / "foo.ply.provenance.npy"
    np.testing.assert_array_equal(np.load(sidecar), provenance)


def test_no_sidecar_written_when_provenance_omitted(tmp_path: Path) -> None:
    xyz, rgb, conf, size = _random_pointset(n=5)
    path = tmp_path / "export.ply"
    write_gaussian_ply(path, xyz, rgb, conf, size)
    assert not (tmp_path / "export.ply.provenance.npy").exists()


def test_splats_extent_gate_accepts_synthetic_ply(tmp_path: Path) -> None:
    """Run Splats' own extent_gate.py (unmodified) against a synthetic PLY.

    Skips cleanly if the Splats ml-sharp venv or the extent-audit script
    aren't present on this machine (per tests/conftest.py's rule: this
    repo must stay green on a machine without ~/Splats).
    """
    if not _SPLATS_ML_SHARP_PYTHON.exists() or not _SPLATS_EXTENT_GATE.exists():
        pytest.skip("Splats ml-sharp venv or extent_gate.py not available on this machine")

    xyz, rgb, conf, size = _random_pointset(n=N, seed=42)
    path = tmp_path / "synthetic_extent.ply"
    write_gaussian_ply(path, xyz, rgb, conf, size)

    result = subprocess.run(
        [str(_SPLATS_ML_SHARP_PYTHON), str(_SPLATS_EXTENT_GATE), str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"extent_gate.py failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert path.name in result.stdout
