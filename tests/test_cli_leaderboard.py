"""End-to-end tests for `trippy leaderboard`.

Module: tests.test_cli_leaderboard
Invariants under test:
    - `trippy leaderboard --out <dir>` (subprocess, matching the style of
      tests/test_cli_train_report.py) exits 0 against a synthetic
      `$TRIPPY_OUTPUT/runs/` tree with one real-shaped `train --report` run,
      writes `leaderboard.md` and `leaderboard.png` under `--out`, and never
      shells out to `scripts/deliver.sh` (no `--deliver` flag passed).
    - `--deliver` under `TRIPPY_DELIVER_DRY_RUN=1` still exits 0 and reports
      a dry-run delivery status rather than calling `scripts/deliver.sh` for
      real -- this repo's forbidden list (no writing into Splats' review
      queue from a CPU test) and AGENTS.md's dry-run-safe rule both require
      this.
    - `--out` defaults to `$TRIPPY_OUTPUT/leaderboard` when omitted.
Fixture: a synthetic `runs/` tree written directly to tmp_path (a
    `report/report.json` + `metrics.jsonl` pair) -- never a real scene,
    checkpoint, or PLY (AGENTS.md: test fixtures must be synthetic only).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from trippy.constants import LEADERBOARD_MARKDOWN_FILENAME, LEADERBOARD_PNG_FILENAME


def _write_synthetic_run(trippy_output: Path) -> None:
    run_dir = trippy_output / "runs" / "EXP-TEST" / "run-1"
    (run_dir / "report").mkdir(parents=True)
    report = {
        "epoch": 5,
        "held_out": {"epoch": 5, "psnr_mean": 20.0, "ssim_mean": 0.5, "lpips_mean": 0.3},
        "dolly": {"frames": [{"coverage_mean_center": 0.3}]},
        "audits": {
            "candidate": {
                "shade_audit": {"results": [{"mass_in_region": 100.0, "dark_mass_lum0.25": 15.0}]},
                "extent_gate": {"plys": [{"radius_p99": 10.0, "radius_max": 20.0}]},
            },
            "baseline": {"shade_audit": {"error": "no ml-sharp venv"}, "extent_gate": {"error": "no ml-sharp venv"}},
        },
    }
    (run_dir / "report" / "report.json").write_text(json.dumps(report))
    metrics_rows = [
        {"step": 1, "epoch": 0},
        {"step": 50, "epoch": 5},
        {"eval": True, "epoch": 5, "psnr_mean": 20.0, "ssim_mean": 0.5, "lpips_mean": 0.3},
    ]
    (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in metrics_rows) + "\n")


def test_cli_leaderboard_writes_markdown_and_png(tmp_path: Path) -> None:
    trippy_output = tmp_path / "trippy_output"
    _write_synthetic_run(trippy_output)
    out_dir = tmp_path / "leaderboard_out"

    env = dict(os.environ)
    env["TRIPPY_OUTPUT"] = str(trippy_output)
    env["TRIPPY_DELIVER_DRY_RUN"] = "1"

    argv = [sys.executable, "-m", "trippy.cli", "leaderboard", "--out", str(out_dir)]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=120, env=env, check=False)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    md_path = out_dir / LEADERBOARD_MARKDOWN_FILENAME
    png_path = out_dir / LEADERBOARD_PNG_FILENAME
    assert md_path.exists()
    assert png_path.exists()

    markdown = md_path.read_text()
    assert "run-1" in markdown
    assert "Gaussians kkc_15000" in markdown
    assert "Design C" in markdown
    assert "20.00/0.500/0.300" in markdown
    assert str(md_path) in result.stdout
    assert str(png_path) in result.stdout


def test_cli_leaderboard_deliver_flag_is_dry_run_safe(tmp_path: Path) -> None:
    trippy_output = tmp_path / "trippy_output"
    _write_synthetic_run(trippy_output)
    out_dir = tmp_path / "leaderboard_out"

    env = dict(os.environ)
    env["TRIPPY_OUTPUT"] = str(trippy_output)
    env["TRIPPY_DELIVER_DRY_RUN"] = "1"

    argv = [sys.executable, "-m", "trippy.cli", "leaderboard", "--out", str(out_dir), "--deliver"]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=120, env=env, check=False)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "delivery skipped: TRIPPY_DELIVER_DRY_RUN=1" in result.stdout
    assert (out_dir / LEADERBOARD_PNG_FILENAME).exists()


def test_cli_leaderboard_out_defaults_to_trippy_output_leaderboard(tmp_path: Path) -> None:
    trippy_output = tmp_path / "trippy_output"
    _write_synthetic_run(trippy_output)

    env = dict(os.environ)
    env["TRIPPY_OUTPUT"] = str(trippy_output)
    env["TRIPPY_DELIVER_DRY_RUN"] = "1"

    argv = [sys.executable, "-m", "trippy.cli", "leaderboard"]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=120, env=env, check=False)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert (trippy_output / "leaderboard" / LEADERBOARD_MARKDOWN_FILENAME).exists()
    assert (trippy_output / "leaderboard" / LEADERBOARD_PNG_FILENAME).exists()
