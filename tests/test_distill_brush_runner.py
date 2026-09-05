"""Tests for trippy.distill.brush_runner: command/job-script building, never execution.

Module: tests.test_distill_brush_runner
Invariants under test: `resolve_brush_binary` prefers brush-cli over brush
    and returns None when neither exists; `brush_train_command` builds the
    exact argv (source positional, --total-train-iters/--sh-degree/
    --export-every/--export-path/--export-name/--seed always present,
    --max-resolution/--eval-split-every only when given); `write_brush_job_
    script` writes an executable script whose body is exactly `exec
    <quoted argv>`; `brush_gpu_submit_command` never invokes anything, only
    builds the string. No test in this module runs Brush, gpu_submit.sh, or
    any subprocess.
"""

from __future__ import annotations

import stat
from pathlib import Path

from trippy.constants import (
    DISTILL_BRUSH_APP_BINARY_REL,
    DISTILL_BRUSH_CLI_BINARY_REL,
    DISTILL_BRUSH_EVAL_EVERY,
    DISTILL_BRUSH_SH_DEGREE,
)
from trippy.distill.brush_runner import (
    brush_gpu_submit_command,
    brush_train_command,
    resolve_brush_binary,
    write_brush_job_script,
)


def test_resolve_brush_binary_returns_none_when_neither_built(tmp_path: Path) -> None:
    assert resolve_brush_binary(rust_dir=tmp_path) is None


def test_resolve_brush_binary_prefers_cli_over_app(tmp_path: Path) -> None:
    app_path = tmp_path / DISTILL_BRUSH_APP_BINARY_REL
    app_path.parent.mkdir(parents=True)
    app_path.write_text("#!/bin/sh\n")
    assert resolve_brush_binary(rust_dir=tmp_path) == app_path

    cli_path = tmp_path / DISTILL_BRUSH_CLI_BINARY_REL
    cli_path.write_text("#!/bin/sh\n")
    assert resolve_brush_binary(rust_dir=tmp_path) == cli_path


def test_brush_train_command_includes_required_flags() -> None:
    argv = brush_train_command("/path/to/brush-cli", "/data/set", "/data/out", total_train_iters=1234)

    assert argv[0] == "/path/to/brush-cli"
    assert argv[1] == "/data/set"
    assert "--total-train-iters" in argv and argv[argv.index("--total-train-iters") + 1] == "1234"
    assert "--sh-degree" in argv and argv[argv.index("--sh-degree") + 1] == str(DISTILL_BRUSH_SH_DEGREE)
    assert "--eval-every" in argv and argv[argv.index("--eval-every") + 1] == str(DISTILL_BRUSH_EVAL_EVERY)
    assert "--export-every" in argv and argv[argv.index("--export-every") + 1] == "1234"
    assert "--export-path" in argv and argv[argv.index("--export-path") + 1] == "/data/out/"
    assert "--seed" in argv
    assert "--with-viewer" not in argv  # a source positional already disables the viewer (Cli's own default)


def test_brush_train_command_omits_optional_flags_when_none() -> None:
    argv = brush_train_command(
        "brush-cli", "dataset", "out", max_resolution=None, eval_split_every=None
    )
    assert "--max-resolution" not in argv
    assert "--eval-split-every" not in argv


def test_brush_train_command_includes_optional_flags_when_given() -> None:
    argv = brush_train_command("brush-cli", "dataset", "out", max_resolution=1008, eval_split_every=8)
    assert argv[argv.index("--max-resolution") + 1] == "1008"
    assert argv[argv.index("--eval-split-every") + 1] == "8"


def test_write_brush_job_script_is_executable_and_execs_argv(tmp_path: Path) -> None:
    argv = ["/bin/echo", "hello world", "--flag"]
    script = write_brush_job_script(tmp_path / "job.sh", argv)

    assert script.exists()
    mode = script.stat().st_mode
    assert mode & stat.S_IXUSR
    text = script.read_text()
    assert text.startswith("#!/bin/bash\nset -eu\n")
    assert "exec /bin/echo 'hello world' --flag" in text


def test_write_brush_job_script_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "job.sh"
    script = write_brush_job_script(nested, ["/bin/true"])
    assert script == nested
    assert nested.exists()


def test_brush_gpu_submit_command_shape() -> None:
    cmd = brush_gpu_submit_command("distill-foo", Path("/out/job.sh"))
    assert cmd == "scripts/gpu_submit.sh --train distill-foo -- bash /out/job.sh"


def test_brush_gpu_submit_command_quotes_special_characters() -> None:
    cmd = brush_gpu_submit_command("distill foo", Path("/out/has space/job.sh"))
    assert "'distill foo'" in cmd
    assert "'/out/has space/job.sh'" in cmd
