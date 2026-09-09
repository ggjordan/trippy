"""Tests for scripts/quest_viewer_bootstrap.sh: syntax, guard clauses, --check.

Module: tests.test_quest_viewer_bootstrap_script
Context: ADR-0008-supersplat.md Stage 3 item 9 -- build a self-hosted, static WebXR
    splat-viewer bundle from a .sog (or .ply) using @playcanvas/splat-transform's
    `.html --unbundled` output. That output is generated, at splat-transform's own
    build time, by calling @playcanvas/supersplat-viewer's own `renderViewerHtml`
    (verified: splat-transform 3.3.3 devDependency-pins
    "@playcanvas/supersplat-viewer": "1.30.2"), so this script shares its pinned,
    local, gitignored install with scripts/sog_export.sh rather than cloning and
    building the viewer repo a second time.

Invariants under test:
    - `bash -n` on the script passes.
    - No args -> usage, exit 2.
    - `--check` prints toolchain readiness and exits 0 WITHOUT installing or
      building anything. Skipped if node/npm are not on PATH.
    - A missing input file -> exit 2 before any install/build is attempted.

This suite never runs the real install/build (network + CPU work) for the same
reason tests/test_sog_export_script.py and tests/test_web_build_script.py don't.
The real build -- against a synthetic .sog first, then the two real bundles for
Jordan's shade-cleaned and untouched Karekare splats -- was run manually; sizes and
file lists are recorded in research/trips-metal.md and docs/QUEST.md.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "quest_viewer_bootstrap.sh"


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


def test_missing_name_is_usage_error(tmp_path: Path) -> None:
    fake_sog = tmp_path / "fake.sog"
    fake_sog.write_bytes(b"")
    result = _run(tmp_path, str(fake_sog))
    assert result.returncode == 2
    assert "usage" in result.stderr


def test_missing_input_exits_2_before_any_network_work(tmp_path: Path) -> None:
    result = _run(tmp_path, str(tmp_path / "nope.sog"), "some-name")
    assert result.returncode == 2
    assert "input not found" in result.stderr
    assert not (tmp_path / "trippy_output" / "tools" / "splat-transform").exists()


def test_check_mode_reports_toolchain_and_installs_nothing(tmp_path: Path) -> None:
    if shutil.which("node") is None or shutil.which("npm") is None:
        pytest.skip("node/npm not installed on this machine")

    result = _run(tmp_path, "--check")

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "node" in result.stdout
    assert "would build" in result.stdout
    assert not (tmp_path / "trippy_output" / "tools" / "splat-transform").exists()


def test_unknown_flag_is_usage_error(tmp_path: Path) -> None:
    result = _run(tmp_path, "--bogus-flag")
    assert result.returncode == 2
    assert "usage" in result.stderr
