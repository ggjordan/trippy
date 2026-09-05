"""scripts/web_build.sh: syntax, dry-run (--check), and guard-clause behaviour.

Module: tests.test_web_build_script
Invariants under test: the script is valid bash; `--check` verifies the
    toolchain (npm, wasm-pack, wasm32-unknown-unknown) and exits 0 without
    building anything; each missing-prerequisite guard clause exits non-zero
    with an actionable message instead of silently doing the wrong thing
    (AGENTS.md: "faking unsupported APIs" is forbidden).
Related docs: docs/WEB_VIEWER.md "Build"; rust/README.md "web" section;
    scripts/web_build.sh itself; scripts/deliver.sh (the intended next step
    after a build, not exercised here -- it has its own review-required
    delivery path and is not something a test should invoke for real).

These tests never run the real wasm-pack/vite build (that's minutes of heavy
CPU work gated behind scripts/cpu_heavy.sh, exercised manually / in
research/trips-metal.md, not on every `pytest` run).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "web_build.sh"


def test_script_exists_and_is_executable() -> None:
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)


def test_script_is_valid_bash() -> None:
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def _minimal_path(*extra_dirs: str) -> str:
    """A PATH with only coreutils/bash (no npm, wasm-pack, or rustup) plus extras."""
    return ":".join([*extra_dirs, "/usr/bin", "/bin", "/usr/sbin", "/sbin"])


def test_check_mode_passes_when_toolchain_present() -> None:
    """--check on the real dev machine: npm/wasm-pack/wasm32 target are installed."""
    npm = shutil.which("npm")
    wasm_pack = shutil.which("wasm-pack")
    if npm is None or wasm_pack is None:
        pytest.skip("npm/wasm-pack not installed on this machine")

    result = subprocess.run(
        ["bash", str(SCRIPT), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "toolchain OK" in result.stdout
    assert "would run: npm ci" in result.stdout
    assert "would copy" in result.stdout


def test_check_mode_dev_flag_selects_dev_wasm_target() -> None:
    npm = shutil.which("npm")
    wasm_pack = shutil.which("wasm-pack")
    if npm is None or wasm_pack is None:
        pytest.skip("npm/wasm-pack not installed on this machine")

    result = subprocess.run(
        ["bash", str(SCRIPT), "--check", "--dev"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "build:wasm-dev" in result.stdout


def test_rejects_unknown_argument() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--bogus"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unknown argument" in result.stderr


def test_missing_npm_exits_nonzero_with_message(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PATH"] = _minimal_path()
    result = subprocess.run(
        ["bash", str(SCRIPT), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 2
    assert "npm not found" in result.stderr


def test_missing_wasm_pack_exits_nonzero_with_message(tmp_path: Path) -> None:
    npm = shutil.which("npm")
    if npm is None:
        pytest.skip("npm not installed on this machine")
    # Symlink only npm into an otherwise-empty bin dir so wasm-pack stays absent.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "npm").symlink_to(npm)

    env = dict(os.environ)
    env["PATH"] = _minimal_path(str(fake_bin))
    result = subprocess.run(
        ["bash", str(SCRIPT), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 2
    assert "wasm-pack not found" in result.stderr


def test_missing_wasm32_target_exits_nonzero_with_message(tmp_path: Path) -> None:
    npm = shutil.which("npm")
    wasm_pack = shutil.which("wasm-pack")
    if npm is None or wasm_pack is None:
        pytest.skip("npm/wasm-pack not installed on this machine")

    # npm and wasm-pack present, but a fake `rustup` reports no wasm32 target,
    # so the guard clause (not the real toolchain) is what's under test.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "npm").symlink_to(npm)
    (fake_bin / "wasm-pack").symlink_to(wasm_pack)
    fake_rustup = fake_bin / "rustup"
    fake_rustup.write_text("#!/bin/bash\necho aarch64-apple-darwin\n")
    fake_rustup.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = _minimal_path(str(fake_bin))
    result = subprocess.run(
        ["bash", str(SCRIPT), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 2
    assert "wasm32-unknown-unknown target missing" in result.stderr


def test_missing_submodule_exits_nonzero_with_message(tmp_path: Path) -> None:
    """Run a copy of the script from a fake repo root with no rust/brush-trips checkout."""
    fake_repo = tmp_path / "repo"
    (fake_repo / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, fake_repo / "scripts" / "web_build.sh")
    (fake_repo / "scripts" / "web_build.sh").chmod(0o755)

    result = subprocess.run(
        ["bash", "scripts/web_build.sh", "--check"],
        cwd=fake_repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "brush-trips submodule" in result.stderr
