"""Builds the Brush CLI command line and job script for design-B distillation training.

Module: trippy.distill.brush_runner
Purpose: the task brief's pipeline step 2 -- "train Gaussians on that image
    set with the Brush fork's CLI (rust/brush-trips, binary brush-cli or the
    app's CLI mode) ... the training itself is GPU work -> scripts/
    gpu_submit.sh --train". This module never runs Brush itself (AGENTS.md:
    "Brush's trainer must NOT be run outside the queue"; the forbidden list
    also bars calling gpu_submit.sh programmatically from library code) --
    it only resolves the built binary, builds the exact argv, writes a
    self-contained job script (mirroring `scripts/gpu_submit.sh`'s own
    generated job-file shape), and returns the `scripts/gpu_submit.sh
    --train` command line for a human (or the Orchestrator) to run, the
    same "print the command, don't run it" convention `trippy depth-points
    --run-depth` already uses for its own GPU step.
Invariants: `brush_train_command`/`write_brush_job_script`/
    `brush_gpu_submit_command` are pure string/file builders -- none of them
    executes `binary`, `scripts/gpu_submit.sh`, or any subprocess.
Related docs: docs/EXPERIMENTS.md "Distillation (design B)"; rust/README.md
    "Building and testing" (how the binary itself gets built, via
    scripts/cpu_heavy.sh); apps/brush-cli/src/lib.rs (the `Cli`/
    `TrainStreamConfig` flags this module's argv must match).
"""

from __future__ import annotations

import shlex
import stat
from pathlib import Path

from trippy.constants import (
    DISTILL_BRUSH_APP_BINARY_REL,
    DISTILL_BRUSH_CLI_BINARY_REL,
    DISTILL_BRUSH_EVAL_EVERY,
    DISTILL_BRUSH_EVAL_SPLIT_EVERY,
    DISTILL_BRUSH_EXPORT_NAME,
    DISTILL_BRUSH_SH_DEGREE,
    DISTILL_DEFAULT_BRUSH_ITERS,
)

_RUST_DIR = Path(__file__).resolve().parents[2] / "rust"


def resolve_brush_binary(rust_dir: Path | None = None) -> Path | None:
    """Find a built Brush binary, preferring the lean headless brush-cli.

    Args:
        rust_dir: override the repo's `rust/` directory (tests only);
            defaults to this checkout's own `rust/`.

    Returns:
        The first of `DISTILL_BRUSH_CLI_BINARY_REL` /
        `DISTILL_BRUSH_APP_BINARY_REL` that exists on disk, or `None` if
        neither has been built yet (see rust/README.md "Building and
        testing" -- build via `scripts/cpu_heavy.sh`, never on a plain push).
    """
    base = rust_dir if rust_dir is not None else _RUST_DIR
    for rel in (DISTILL_BRUSH_CLI_BINARY_REL, DISTILL_BRUSH_APP_BINARY_REL):
        candidate = base / rel
        if candidate.exists():
            return candidate
    return None


def brush_train_command(
    binary: str | Path,
    dataset_dir: str | Path,
    export_dir: str | Path,
    total_train_iters: int = DISTILL_DEFAULT_BRUSH_ITERS,
    sh_degree: int = DISTILL_BRUSH_SH_DEGREE,
    max_resolution: int | None = None,
    eval_split_every: int | None = DISTILL_BRUSH_EVAL_SPLIT_EVERY,
    eval_every: int = DISTILL_BRUSH_EVAL_EVERY,
    export_name: str = DISTILL_BRUSH_EXPORT_NAME,
    seed: int = 0,
) -> list[str]:
    """The exact argv brush-cli/brush needs to train headlessly on `dataset_dir`.

    `dataset_dir` is passed as the positional `source`; a source argument
    alone already flips `--with-viewer` off by `Cli`'s own
    `default_value_if` (apps/brush-cli/src/lib.rs), so this never passes
    `--with-viewer` explicitly. `--export-every` is always set to
    `total_train_iters` (export exactly once, at the end -- this pipeline
    only wants the final distilled PLY, not intermediate checkpoints).

    Args:
        binary: path to the built `brush-cli`/`brush` executable (not
            invoked here -- see module docstring).
        dataset_dir: the COLMAP-text image set `trippy.distill.render_set`
            wrote (contains `images/` and `sparse_txt/`).
        export_dir: directory Brush writes its exported PLY(s) into
            (`--export-path`; a trailing slash is added if missing, per
            that flag's own "Path is relative to the dataset's parent
            directory" doc -- an absolute path with a trailing slash is
            unambiguous either way).
        total_train_iters, sh_degree, max_resolution, eval_split_every,
            eval_every, export_name, seed: forwarded to the matching
            brush-cli flags (`--total-train-iters`, `--sh-degree`,
            `--max-resolution`, `--eval-split-every`, `--eval-every`,
            `--export-name`, `--seed`). `max_resolution=None` and
            `eval_split_every=None` omit those flags (brush-cli's own
            defaults apply); every other argument is always passed
            explicitly so the printed command is self-documenting.

    Returns:
        The full argv list, `binary` first.
    """
    export_path = str(export_dir)
    if not export_path.endswith("/"):
        export_path += "/"

    argv = [
        str(binary),
        str(dataset_dir),
        "--total-train-iters",
        str(total_train_iters),
        "--sh-degree",
        str(sh_degree),
        "--eval-every",
        str(eval_every),
        "--export-every",
        str(total_train_iters),
        "--export-path",
        export_path,
        "--export-name",
        export_name,
        "--seed",
        str(seed),
    ]
    if max_resolution is not None:
        argv += ["--max-resolution", str(max_resolution)]
    if eval_split_every is not None:
        argv += ["--eval-split-every", str(eval_split_every)]
    return argv


def write_brush_job_script(path: str | Path, argv: list[str]) -> Path:
    """Write a self-contained, executable job script that `exec`s `argv`.

    Mirrors `scripts/gpu_submit.sh`'s own generated job-file shape (`set
    -eu` then one `exec` line) so the printed
    `brush_gpu_submit_command(...)` line is a normal `scripts/
    gpu_submit.sh --train <name> -- bash <path>` invocation -- never
    executed by this function itself.

    Args:
        path: output `.sh` path; parent directories are created if missing.
        argv: as returned by `brush_train_command`.

    Returns:
        `path`, made executable (mode +x).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exec_line = "exec " + " ".join(shlex.quote(tok) for tok in argv)
    path.write_text(f"#!/bin/bash\nset -eu\n{exec_line}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def brush_gpu_submit_command(job_name: str, script_path: str | Path) -> str:
    """The `scripts/gpu_submit.sh --train` line that queues `script_path`.

    Never executed here -- printed/returned for a human (or the
    Orchestrator) to run, the same convention `trippy depth-points
    --run-depth` uses for its own printed GPU command.

    Args:
        job_name: passed to `gpu_submit.sh`'s own `<name>` positional (it
            prefixes this with `trippy-` itself).
        script_path: as written by `write_brush_job_script`.

    Returns:
        A copy-pasteable shell command line.
    """
    return f"scripts/gpu_submit.sh --train {shlex.quote(job_name)} -- bash {shlex.quote(str(script_path))}"
