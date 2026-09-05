"""Tests for scripts/queue_training.sh: config validation, job naming, --dry-run.

Module: tests.test_queue_training_script
Invariants under test:
    - Missing config path -> exit 2, no gpu_submit.sh call.
    - Config with no top-level `run_dir:` key -> exit 2.
    - Config with a `run_dir:` whose basename has characters gpu_submit.sh's
      job-name rule rejects -> exit 2, caught here (before ever reaching
      gpu_submit.sh).
    - A valid config + `--dry-run` submits (dry-run) at `--train` priority
      (70) via scripts/gpu_submit.sh, named after the config's own
      `run_dir:` basename, running `trippy train --config <cfg> --report`
      (never just `trippy train` -- this task's brief: queue_training.sh
      always makes a run self-reporting).
    - `--max-minutes` is forwarded into the generated job's command line.
Fixture: temp YAML config files only; --dry-run never touches the real GPU
    queue, ~/Splats, or research/trips-metal.md (scripts/gpu_submit.sh's own
    documented invariant).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "queue_training.sh"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_missing_config_exits_2(tmp_path: Path) -> None:
    result = _run(str(tmp_path / "nope.yaml"), "--dry-run")
    assert result.returncode == 2
    assert "config not found" in result.stderr


def test_config_missing_run_dir_key_exits_2(tmp_path: Path) -> None:
    config = tmp_path / "cfg.yaml"
    config.write_text("scene_root: /tmp/nowhere\nwidth: 64\n")
    result = _run(str(config), "--dry-run")
    assert result.returncode == 2
    assert "run_dir" in result.stderr


def test_config_run_dir_with_invalid_chars_exits_2(tmp_path: Path) -> None:
    config = tmp_path / "cfg.yaml"
    config.write_text("run_dir: output/runs/EXP-TEST/bad name with spaces\n")
    result = _run(str(config), "--dry-run")
    assert result.returncode == 2
    assert "must match" in result.stderr


def test_valid_config_dry_run_submits_train_prio_with_report_flag(tmp_path: Path) -> None:
    config = tmp_path / "cfg.yaml"
    config.write_text("scene_root: /tmp/nowhere\nrun_dir: output/runs/EXP-TEST/queue_test_run\nwidth: 64\n")

    result = _run(str(config), "--dry-run")
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "prio=70" in result.stdout
    assert "name=trippy-queue_test_run" in result.stdout
    assert "trippy.cli train" in result.stdout
    assert "--report" in result.stdout
    assert str(config) in result.stdout


def test_max_minutes_forwarded_into_job_command(tmp_path: Path) -> None:
    config = tmp_path / "cfg.yaml"
    config.write_text("run_dir: output/runs/EXP-TEST/queue_test_run2\n")

    result = _run(str(config), "--max-minutes", "45", "--dry-run")
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "--max-minutes 45" in result.stdout


def test_usage_error_when_no_config_given() -> None:
    result = _run("--dry-run")
    assert result.returncode == 2
    assert "usage" in result.stderr
