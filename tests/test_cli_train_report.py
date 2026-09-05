"""End-to-end + failure-path tests for `trippy train --report`.

Module: tests.test_cli_train_report
Invariants under test:
    - `python -m trippy.cli train --config <yaml> --device cpu --report`
      exits 0 on the synthetic scene, writes `<run_dir>/report/report.json`,
      and appends a "## Report: epoch N" section (with the baseline-vs-
      candidate comparison table) to `<run_dir>/README.md` -- this task's
      brief requirement 6. The synthetic scene's `points3D.txt` is empty
      (see tests/test_train_helpers.py), so the shade audit legitimately
      degrades to `{"error": ...}`; the table must still render (no
      exception, no missing section).
    - `TRIPPY_DELIVER_DRY_RUN=1` (set for this whole test module) means
      every delivery is recorded as skipped rather than shelling out to
      scripts/deliver.sh -- this repo's forbidden list ("no GPU jobs";
      AGENTS.md "never write into ~/Splats/output/Jordan-Review") and this
      task's brief both require CPU tests to never call deliver.sh for real.
    - `trippy.cli._run_train_report_safely` (requirement 1: "--report never
      crashes the run") catches an exception from `run_train_report` and
      writes `<run_dir>/REPORT_FAILED.txt` instead of propagating it, and
      never touches sys.exit itself (the caller, `_cmd_train`, always
      returns 0 once `fit()` has succeeded).
Fixture: the shared synthetic scene/config builders from
    tests/test_train_helpers.py (never a real Splats scene or checkpoint).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config

from trippy.constants import TRAIN_REPORT_FAILED_FILENAME


def test_cli_train_report_end_to_end_on_synthetic_scene(tmp_path: Path) -> None:
    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    run_dir = tmp_path / "run"
    cfg = tiny_train_config(scene_root, ply_path, run_dir, tmp_path / "cache")
    config_path = cfg.save_yaml(tmp_path / "config.yaml")

    env = dict(os.environ)
    env["TRIPPY_DELIVER_DRY_RUN"] = "1"
    env["TRIPPY_OUTPUT"] = str(tmp_path / "trippy_output")

    argv = [
        sys.executable, "-m", "trippy.cli", "train",
        "--config", str(config_path), "--device", "cpu", "--report",
    ]  # fmt: skip
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "trippy train" in result.stdout
    assert not (run_dir / TRAIN_REPORT_FAILED_FILENAME).exists(), result.stdout + result.stderr
    assert "report ->" in result.stdout

    report_dir = run_dir / "report"
    report_json_path = report_dir / "report.json"
    assert report_json_path.exists()
    report = json.loads(report_json_path.read_text())
    for key in ("checkpoint", "device", "scene_root", "export_ply", "epoch", "held_out", "dolly", "offpath", "audits"):
        assert key in report

    # Empty points3D.txt (test_train_helpers' synthetic scene) -- the shade
    # audit legitimately cannot compute anything; it must degrade to an
    # error, not crash the whole report.
    assert "error" in report["audits"]["candidate"]["shade_audit"]
    assert "error" in report["audits"]["baseline"]["shade_audit"]

    assert "summary_line" in report
    assert f"epoch {report['epoch']}" in report["summary_line"]
    assert "good" not in report["summary_line"].lower()

    assert len(report["deliveries"]) == 3
    for delivery in report["deliveries"]:
        assert delivery["status"].startswith("skipped")  # TRIPPY_DELIVER_DRY_RUN=1

    readme_text = (run_dir / "README.md").read_text()
    assert f"## Report: epoch {report['epoch']}" in readme_text
    assert "| Metric | Baseline | Candidate |" in readme_text
    assert report["summary_line"] in readme_text


def test_cli_train_report_failure_writes_report_failed_but_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trippy.render.report as report_mod
    from trippy import cli

    def _boom(trainer, metrics):
        raise RuntimeError("audit tool exploded")

    monkeypatch.setattr(report_mod, "run_train_report", _boom)

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    class _FakeTrainer:
        pass

    fake_trainer = _FakeTrainer()
    fake_trainer.run_dir = run_dir

    cli._run_train_report_safely(fake_trainer, {"epoch": 3})  # must not raise

    failed_path = run_dir / TRAIN_REPORT_FAILED_FILENAME
    assert failed_path.exists()
    assert "audit tool exploded" in failed_path.read_text()
