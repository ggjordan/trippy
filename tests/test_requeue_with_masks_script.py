"""Tests for scripts/requeue_with_masks.sh: validation, sibling-config generation, --dry-run.

Module: tests.test_requeue_with_masks_script
Context: kk-coherent trained without person-exclusion masks (experiments/MASKS.md). Jordan
    wants BOTH the existing unmasked results and masked results (2026-09-06 correction), so
    this script must never touch, edit, or dequeue an original config or its GPU-queue entry
    -- it only ever writes a "_masked" SIBLING config and submits that.

Invariants under test:
    - `bash -n` on the script passes (syntax only, no execution).
    - No args -> usage, exit 2.
    - Missing config path -> exit 2, no sibling written, queue_training.sh never reached.
    - Config with a `scene_root:` that does not mention "kk-coherent" -> exit 2 (masks were
      only generated for kk-coherent; refuse rather than write a meaningless masks_dir).
    - Config with no top-level `run_dir:` key -> exit 2.
    - A valid kk-coherent config + `--dry-run`:
        - exits 0 and reports the sibling's `masks_dir:` and `-masked` run_dir.
        - forwards --dry-run all the way to gpu_submit.sh (prio=70, job name
          `trippy-<run_dir-basename>-masked`, `trippy.cli train ... --report`), matching
          queue_training.sh's own --train + --report contract.
        - writes its scratch sibling under $TRIPPY_OUTPUT/tmp/, NEVER next to the original
          config, and deletes it again before exiting (real-mode writes go next to the
          original as `<name>_masked.yaml`, but this test suite never exercises real mode --
          that would submit a real GPU-queue job, which is out of scope for a unit test and
          forbidden by this task's brief).
    - `--masks-dir` overrides the default masks_dir value written into the sibling.
    - The original config file's bytes are never modified by any of the above.
Fixture: temp YAML config files + a temp TRIPPY_OUTPUT only; --dry-run never touches
    ~/Splats, the real GPU queue, or research/trips-metal.md (gpu_submit.sh's own documented
    --dry-run invariant, inherited via queue_training.sh --dry-run).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "requeue_with_masks.sh"

_KK_SCENE_ROOT = "/Users/nzbirdranch/Splats/scenes/karekare/kk-coherent"
_DEFAULT_MASKS_DIR = "/Users/nzbirdranch/trippy/output/masks/kk-coherent"


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["TRIPPY_OUTPUT"] = str(tmp_path / "trippy_output")
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        cwd=_REPO_ROOT,
        env=env,
    )


def _write_config(
    path: Path, *, scene_root: str = _KK_SCENE_ROOT, run_dir: str | None = "output/runs/EXP-TEST/full1"
) -> None:
    lines = [f"scene_root: {scene_root}\n"]
    if run_dir is not None:
        lines.append(f"run_dir: {run_dir}\n")
    lines.append("\nwidth: 1008\ndevice: mps\n")
    path.write_text("".join(lines))


def test_script_is_syntactically_valid_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(_SCRIPT)], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr


def test_no_args_is_usage_error(tmp_path: Path) -> None:
    result = _run(tmp_path, "--dry-run")
    assert result.returncode == 2
    assert "usage" in result.stderr


def test_missing_config_exits_2(tmp_path: Path) -> None:
    result = _run(tmp_path, "--dry-run", str(tmp_path / "nope.yaml"))
    assert result.returncode == 2
    assert "config not found" in result.stderr


def test_non_kk_coherent_scene_root_refused(tmp_path: Path) -> None:
    config = tmp_path / "cfg.yaml"
    _write_config(config, scene_root="/Users/nzbirdranch/Splats/scenes/hunua/run00")
    result = _run(tmp_path, "--dry-run", str(config))
    assert result.returncode == 2
    assert "kk-coherent" in result.stderr


def test_missing_run_dir_key_exits_2(tmp_path: Path) -> None:
    config = tmp_path / "cfg.yaml"
    _write_config(config, run_dir=None)
    result = _run(tmp_path, "--dry-run", str(config))
    assert result.returncode == 2
    assert "run_dir" in result.stderr


def test_valid_config_dry_run_reports_masks_dir_and_masked_run_dir(tmp_path: Path) -> None:
    config = tmp_path / "cfg.yaml"
    _write_config(config, run_dir="output/runs/EXP-TEST/full1")
    original_bytes = config.read_bytes()

    result = _run(tmp_path, "--dry-run", str(config))

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert f"masks_dir: {_DEFAULT_MASKS_DIR}" in result.stdout
    assert "run_dir:   output/runs/EXP-TEST/full1-masked" in result.stdout
    # Forwarded through queue_training.sh --dry-run to gpu_submit.sh --dry-run.
    assert "prio=70" in result.stdout
    assert "name=trippy-full1-masked" in result.stdout
    assert "trippy.cli train" in result.stdout
    assert "--report" in result.stdout
    # Original config is never modified.
    assert config.read_bytes() == original_bytes


def test_dry_run_never_writes_next_to_original_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "expdir"
    config_dir.mkdir()
    config = config_dir / "cfg.yaml"
    _write_config(config)

    result = _run(tmp_path, "--dry-run", str(config))

    assert result.returncode == 0, result.stderr
    # No sibling (e.g. cfg_masked.yaml) was left behind next to the original.
    assert sorted(p.name for p in config_dir.iterdir()) == ["cfg.yaml"]


def test_dry_run_scratch_file_is_cleaned_up(tmp_path: Path) -> None:
    config = tmp_path / "cfg.yaml"
    _write_config(config)
    trippy_output = tmp_path / "trippy_output"

    result = _run(tmp_path, "--dry-run", str(config))

    assert result.returncode == 0, result.stderr
    scratch_dir = trippy_output / "tmp"
    leftover = list(scratch_dir.glob("requeue_masked.*")) if scratch_dir.exists() else []
    assert leftover == []


def test_masks_dir_override_is_forwarded(tmp_path: Path) -> None:
    config = tmp_path / "cfg.yaml"
    _write_config(config)

    result = _run(tmp_path, "--dry-run", "--masks-dir", "/tmp/custom-masks", str(config))

    assert result.returncode == 0, result.stderr
    assert "masks_dir: /tmp/custom-masks" in result.stdout


def test_multiple_configs_each_get_their_own_sibling(tmp_path: Path) -> None:
    config_a = tmp_path / "a.yaml"
    config_b = tmp_path / "b.yaml"
    _write_config(config_a, run_dir="output/runs/EXP-TEST/run-a")
    _write_config(config_b, run_dir="output/runs/EXP-TEST/run-b")

    result = _run(tmp_path, "--dry-run", str(config_a), str(config_b))

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "name=trippy-run-a-masked" in result.stdout
    assert "name=trippy-run-b-masked" in result.stdout
