"""scripts/web_build.sh: syntax, dry-run (--check), and guard-clause behaviour.

Module: tests.test_web_build_script
Invariants under test: the script is valid bash; `--check` verifies the
    toolchain (npm, wasm-pack, wasm32-unknown-unknown) and exits 0 without
    building anything; each missing-prerequisite guard clause exits non-zero
    with an actionable message instead of silently doing the wrong thing
    (AGENTS.md: "faking unsupported APIs" is forbidden). Both build targets are
    covered: the default (stock Brush demo, wasm-pack + vite) and `--trips`
    (trippy's own trips-web viewer, wasm-pack --target web, no bundler).
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


# --- `--trips`: trippy's own web viewer ------------------------------------


def _toolchain_or_skip() -> None:
    if shutil.which("npm") is None or shutil.which("wasm-pack") is None:
        pytest.skip("npm/wasm-pack not installed on this machine")


def _fake_bundle(tmp_path: Path) -> Path:
    """A directory that satisfies the --trips bundle guard clause.

    Synthetic, not a real scene: the guard only reads bundle.json, and a test
    must never depend on an exported bundle being present (AGENTS.md's
    "synthetic fixtures only").
    """
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bundle.json").write_text(
        '{"format": "trippy-bundle-1", "points": "points.npz", '
        '"weights": "weights.safetensors", "num_channels": 4, '
        '"params": {}, "views": []}'
    )
    return bundle


def test_trips_check_mode_prints_the_trips_plan(tmp_path: Path) -> None:
    _toolchain_or_skip()
    bundle = _fake_bundle(tmp_path)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--check", "--trips", "--bundle", str(bundle)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "toolchain OK" in result.stdout
    # The trips target is bundler-free on purpose: --target web, no vite.
    assert "--target web" in result.stdout
    assert "wasm-pack build rust/crates/trips-web" in result.stdout
    assert "web/index.html" in result.stdout
    assert str(bundle) in result.stdout
    # It must NOT run the stock Brush demo's npm/vite steps.
    assert "npm ci" not in result.stdout
    assert "vite build" not in result.stdout


def test_trips_profiling_flag_selects_the_profiling_build(tmp_path: Path) -> None:
    """`--profiling` is what makes a wasm panic's stack trace readable.

    wasm-opt strips the name section in a `--release` build, so a browser
    stack is a list of `wasm-function[N]`. `--profiling` keeps release
    codegen and the names, which is how the CubeCL autotune roofline trap was
    located (docs/WEB_VIEWER.md blocker 4). The flag has to keep existing for
    the next person who has to read one.
    """
    _toolchain_or_skip()
    bundle = _fake_bundle(tmp_path)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--check", "--trips", "--profiling", "--bundle", str(bundle)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--profiling" in result.stdout


def test_trips_out_flag_overrides_the_output_directory(tmp_path: Path) -> None:
    _toolchain_or_skip()
    bundle = _fake_bundle(tmp_path)
    out = tmp_path / "elsewhere"
    result = subprocess.run(
        [
            "bash", str(SCRIPT), "--check", "--trips",
            "--bundle", str(bundle), "--out", str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert str(out) in result.stdout


def test_trips_missing_bundle_directory_exits_nonzero(tmp_path: Path) -> None:
    _toolchain_or_skip()
    result = subprocess.run(
        ["bash", str(SCRIPT), "--check", "--trips", "--bundle", str(tmp_path / "nope")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "bundle directory not found" in result.stderr


def test_trips_directory_without_a_manifest_exits_nonzero(tmp_path: Path) -> None:
    _toolchain_or_skip()
    empty = tmp_path / "empty"
    empty.mkdir()
    result = subprocess.run(
        ["bash", str(SCRIPT), "--check", "--trips", "--bundle", str(empty)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "no bundle.json" in result.stderr


def test_flag_needing_a_value_says_so() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--trips", "--bundle"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--bundle needs a directory" in result.stderr


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


# --- the wasm module must not be linked as a "command" -----------------------
#
# These two guard the single largest performance fact about the browser viewer.
# `wasm-ld` wraps every export of a non-reactor wasm module in a
# `<name>.command_export` shim that re-runs the whole `.init_array` on entry.
# Rust normally has no static constructors so that is free; this graph pulls in
# `pliron` through `cubecl-ir`, whose `inventory` registrations make one
# `__wasm_call_ctors` cost ~110 us -- and `wasm-bindgen` resolves
# `__externref_table_alloc` by export name, so every `JsValue` `wgpu` built for
# a bind group paid it. Measured: `raw level-0` 3.3 fps with the wrappers,
# 114 fps without (docs/WEB_VIEWER.md).
#
# The fix is two halves that only work together, in two different languages, so
# each half is asserted here: `build.rs` suppresses the wrappers, and
# `web/trips.js` then has to run the constructors itself.

TRIPS_WEB_BUILD_RS = REPO_ROOT / "rust" / "crates" / "trips-web" / "build.rs"
TRIPS_JS = REPO_ROOT / "web" / "trips.js"


def test_trips_web_build_script_exports_wasm_call_ctors() -> None:
    """`build.rs` must pass `--export=__wasm_call_ctors` to the wasm linker."""
    assert TRIPS_WEB_BUILD_RS.is_file(), "rust/crates/trips-web/build.rs is missing"
    source = TRIPS_WEB_BUILD_RS.read_text(encoding="utf-8")
    assert "rustc-link-arg=--export=__wasm_call_ctors" in source, (
        "trips-web/build.rs no longer exports __wasm_call_ctors; wasm-ld will go "
        "back to wrapping every export in a command_export shim that re-runs "
        "pliron's inventory registration (~15x slower). See docs/WEB_VIEWER.md."
    )
    # wasm only: the native trips-viewer link must not see this flag.
    assert "CARGO_CFG_TARGET_FAMILY" in source and '"wasm"' in source, (
        "the link argument must be gated on the wasm target family"
    )


def test_page_runs_the_static_constructors_once() -> None:
    """`web/trips.js` must call `__wasm_call_ctors` and refuse if it is absent."""
    source = TRIPS_JS.read_text(encoding="utf-8")
    assert "wasm.__wasm_call_ctors();" in source, (
        "web/trips.js must run the module's static constructors itself: "
        "build.rs suppresses wasm-ld's command_export wrappers, which were the "
        "only thing calling them."
    )
    assert 'typeof wasm?.__wasm_call_ctors !== "function"' in source, (
        "web/trips.js must refuse to start when the export is missing, rather "
        "than silently rendering ~15x slower"
    )
