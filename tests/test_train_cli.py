"""End-to-end subprocess test for `trippy train` / `trippy eval` on the synthetic scene.

Module: tests.test_train_cli
Invariants under test: `python -m trippy.cli train --config <yaml> --device cpu`
    exits 0 within a small time budget on the tiny synthetic scene (the
    `config_smoke.yaml`-style first queue-run rehearsal), producing a
    checkpoint and an export; `trippy eval --checkpoint <ckpt>` then exits 0
    and prints held-out metrics.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config


def test_cli_train_then_eval_on_synthetic_scene(tmp_path: Path) -> None:
    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    run_dir = tmp_path / "run"
    cfg = tiny_train_config(scene_root, ply_path, run_dir, tmp_path / "cache")
    config_path = cfg.save_yaml(tmp_path / "config.yaml")

    train_result = subprocess.run(
        [sys.executable, "-m", "trippy.cli", "train", "--config", str(config_path), "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert train_result.returncode == 0, f"stdout={train_result.stdout}\nstderr={train_result.stderr}"
    assert "trippy train" in train_result.stdout

    checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint_ep*.pt"))
    assert checkpoints, "expected at least one checkpoint to have been written"
    assert (run_dir / "export.ply").exists()

    eval_result = subprocess.run(
        [sys.executable, "-m", "trippy.cli", "eval", "--checkpoint", str(checkpoints[-1]), "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert eval_result.returncode == 0, f"stdout={eval_result.stdout}\nstderr={eval_result.stderr}"
    assert "psnr_mean" in eval_result.stdout


def test_cli_train_respects_max_minutes_flag(tmp_path: Path) -> None:
    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    run_dir = tmp_path / "run"
    cfg = tiny_train_config(scene_root, ply_path, run_dir, tmp_path / "cache", epochs=1000)
    config_path = cfg.save_yaml(tmp_path / "config.yaml")

    argv = [
        sys.executable, "-m", "trippy.cli", "train",
        "--config", str(config_path), "--device", "cpu", "--max-minutes", "0.02",
    ]  # fmt: skip
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert (run_dir / "export.ply").exists()
