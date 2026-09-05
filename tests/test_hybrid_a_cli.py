"""End-to-end `trippy train --report` on a hybrid design-A config (the smoke job's own path).

Module: tests.test_hybrid_a_cli
Invariants under test: the exact command the EXP-0009 smoke job runs
    (`trippy train --config <hybrid yaml> --report`) completes with rc 0 on
    the synthetic scene, writes a report whose checkpoint carries the resolved
    `hybrid:` block, and reaches the candidate-report renderer with the
    Gaussian channels attached. `hybrid.ply_path` is deliberately empty here
    so no live `gsrender` import (and therefore no `~/Splats`, no MPS) is ever
    attempted: poses anchored to a registered image use that image's cached
    render, and any other pose gets an honest all-zero block.
    `TRIPPY_DELIVER_DRY_RUN=1` keeps deliver.sh out of the CPU suite.
Fixture: the synthetic scene + fake renders from tests/test_hybrid_a_helpers.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from test_hybrid_a_helpers import hybrid_train_config

from trippy.constants import TRAIN_REPORT_FAILED_FILENAME


def test_cli_train_report_end_to_end_with_hybrid_enabled(tmp_path: Path) -> None:
    cfg, _names = hybrid_train_config(tmp_path, ply_path="")
    run_dir = Path(cfg.run_dir)
    config_path = cfg.save_yaml(tmp_path / "config.yaml")

    env = dict(os.environ)
    env["TRIPPY_DELIVER_DRY_RUN"] = "1"
    env["TRIPPY_OUTPUT"] = str(tmp_path / "trippy_output")

    argv = [
        sys.executable, "-m", "trippy.cli", "train",
        "--config", str(config_path), "--device", "cpu", "--report",
    ]  # fmt: skip
    result = subprocess.run(argv, capture_output=True, text=True, timeout=600, env=env, check=False)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert not (run_dir / TRAIN_REPORT_FAILED_FILENAME).exists(), result.stdout + result.stderr

    report = json.loads((run_dir / "report" / "report.json").read_text())
    assert report["dolly"]["n_frames"] > 0
    assert report["offpath"]["n_frames"] > 0
    assert "## Report: epoch" in (run_dir / "README.md").read_text()

    # The checkpoint the report reloaded carries the resolved hybrid block.
    payload = torch.load(
        run_dir / "checkpoints" / "checkpoint_latest.pt", map_location="cpu", weights_only=False
    )
    hybrid = payload["cfg"]["hybrid"]
    assert hybrid["enabled"] is True
    assert hybrid["depth_scale"] is not None
    # 4 TRIPS feature channels + rgb/alpha/depth = 9 per level; the coarsest block's
    # gated conv reads exactly that many.
    assert payload["net"]["start.conv.feature_conv.weight"].shape[1] == 9
    assert payload["net"]["start.conv.gate_conv.weight"].shape[1] == 9
