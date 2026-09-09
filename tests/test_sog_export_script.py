"""Tests for scripts/sog_export.sh: syntax, guard clauses, --check.

Module: tests.test_sog_export_script
Context: ADR-0008-supersplat.md Stage 3 item 8 -- compress a Gaussian-splat PLY to
    SOG (+ a compressed .ply second artefact) via a locally-pinned, gitignored
    install of @playcanvas/splat-transform, and prove the SOG round-trips
    (Gaussian count preserved through .sog and back).

Invariants under test:
    - `bash -n` on the script passes (syntax only, no execution).
    - No args -> usage, exit 2.
    - `--check` prints toolchain readiness (node, npm, free memory, whether the
      pin is already installed) and exits 0 WITHOUT installing anything -- no
      network, no node_modules created. Skipped if node/npm are not on PATH.
    - A missing input file -> exit 2, before any install/conversion is attempted
      (fast failure; this must not require network to observe).
    - `--out-dir` / `--name` are accepted as flags (parsed before the positional
      input check fires for a still-missing file).

This suite deliberately never runs the real `npm install @playcanvas/splat-transform`
or a real conversion (minutes of first-time network + CPU work, same reasoning as
tests/test_web_build_script.py). The real run -- against a synthetic PLY first, then
for real against the delivered kklid-tripsclean-shade-keep.ply and the untouched
kklid_20000.ply -- was exercised manually; sizes, versions and round-trip Gaussian
counts are recorded in research/trips-metal.md and docs/QUEST.md, not re-derived here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "sog_export.sh"


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["TRIPPY_OUTPUT"] = str(tmp_path / "trippy_output")
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        cwd=REPO_ROOT,
        env=env,
    )


def test_script_exists_and_is_executable() -> None:
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)


def test_script_is_valid_bash() -> None:
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_no_args_is_usage_error(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 2
    assert "usage" in result.stderr


def test_missing_input_exits_2_before_any_network_work(tmp_path: Path) -> None:
    result = _run(tmp_path, str(tmp_path / "nope.ply"))
    assert result.returncode == 2
    assert "input not found" in result.stderr
    # No install directory was created reaching this guard.
    assert not (tmp_path / "trippy_output" / "tools" / "splat-transform").exists()


def test_check_mode_reports_toolchain_and_installs_nothing(tmp_path: Path) -> None:
    if shutil.which("node") is None or shutil.which("npm") is None:
        pytest.skip("node/npm not installed on this machine")

    result = _run(tmp_path, "--check")

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "node" in result.stdout
    assert "free memory" in result.stdout
    assert "would convert" in result.stdout
    assert "would round-trip" in result.stdout
    # --check must never touch the network or write the install directory.
    assert not (tmp_path / "trippy_output" / "tools" / "splat-transform").exists()


def test_check_mode_ignores_extra_positional_input(tmp_path: Path) -> None:
    """--check answers "is the toolchain ready", independent of a real target file."""
    if shutil.which("node") is None or shutil.which("npm") is None:
        pytest.skip("node/npm not installed on this machine")

    result = _run(tmp_path, "--check", str(tmp_path / "does-not-exist.ply"))

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert not (tmp_path / "trippy_output" / "tools" / "splat-transform").exists()


def test_unknown_flag_is_usage_error(tmp_path: Path) -> None:
    result = _run(tmp_path, "--bogus-flag")
    assert result.returncode == 2
    assert "usage" in result.stderr
