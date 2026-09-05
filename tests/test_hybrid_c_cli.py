"""End-to-end subprocess test for `trippy hybrid-c train` / `trippy hybrid-c eval`.

Module: tests.test_hybrid_c_cli
Invariants under test: `python -m trippy.cli hybrid-c train --config <yaml> --device cpu`
    exits 0 within a small time budget on the tiny synthetic scene/render pair, producing a
    checkpoint; `trippy hybrid-c eval --checkpoint <ckpt>` then exits 0 and prints its
    baseline/refined metrics as JSON.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from test_hybrid_c_helpers import build_synthetic_renders, build_synthetic_scene, tiny_hybrid_c_config


def test_cli_hybrid_c_train_then_eval_on_synthetic_scene(tmp_path: Path) -> None:
    scene_root, names = build_synthetic_scene(tmp_path)
    renders_dir = build_synthetic_renders(scene_root, tmp_path / "renders", names)
    run_dir = tmp_path / "run"
    cfg = tiny_hybrid_c_config(scene_root, renders_dir, run_dir, tmp_path / "cache")
    config_path = cfg.save_yaml(tmp_path / "config.yaml")

    train_result = subprocess.run(
        [sys.executable, "-m", "trippy.cli", "hybrid-c", "train", "--config", str(config_path), "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert train_result.returncode == 0, f"stdout={train_result.stdout}\nstderr={train_result.stderr}"
    assert "trippy hybrid-c train" in train_result.stdout

    checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint_ep*.pt"))
    assert checkpoints, "expected at least one checkpoint to have been written"

    eval_result = subprocess.run(
        [sys.executable, "-m", "trippy.cli", "hybrid-c", "eval", "--checkpoint", str(checkpoints[-1]), "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )  # fmt: skip
    assert eval_result.returncode == 0, f"stdout={eval_result.stdout}\nstderr={eval_result.stderr}"
    line = next(line for line in eval_result.stdout.splitlines() if line.startswith("JSON:"))
    payload = json.loads(line[len("JSON:") :])
    assert "baseline" in payload and "refined" in payload
