"""End-to-end + failure-path tests for `trippy train --report`.

Module: tests.test_cli_train_report
Invariants under test:
    - `python -m trippy.cli train --config <yaml> --device cpu --report`
      exits 0 on the synthetic scene, writes `<run_dir>/report/report.json`,
      and appends a "## Report: epoch N" section (with a deliveries list and
      the baseline-vs-candidate comparison table) to `<run_dir>/README.md`
      -- this task's brief requirement 6. The synthetic scene's
      `points3D.txt` is empty (see tests/test_train_helpers.py), so the
      shade audit legitimately degrades to `{"error": ...}`; the table must
      still render (no exception, no missing section).
    - Jordan wants free navigation, not just a fixed dolly path: `--report`
      also exports a `bundle/` directory (`trippy.render.bundle.export_bundle`'s
      output, i.e. `bundle.json` at minimum) under the run dir and generates
      a Mac double-click launcher (`OPEN_TRIPS_MAC_<run_name>.command` under
      `$TRIPPY_OUTPUT/deliver/<run_name>/`), and lists that launcher FIRST
      among the 4 deliveries (viewer, dolly, honesty, export) both in
      `report.json["deliveries"]` and in the README's "### Deliveries" list.
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

    # Requirements 1-3: a free-navigation bundle + Mac launcher, exported from this
    # run's own final checkpoint, alongside the existing dolly/honesty artifacts.
    assert "bundle" in report
    bundle_dir = Path(report["bundle"]["bundle_dir"])
    assert bundle_dir == run_dir / "bundle"
    assert (bundle_dir / "bundle.json").exists()
    assert (bundle_dir / "points.npz").exists()
    assert (bundle_dir / "weights.safetensors").exists()

    viewer = report["bundle"]["viewer"]
    if viewer["status"] == "ok":
        # The common case on the dev machine (rust/target/release/trips-viewer built).
        command_path = Path(viewer["command_path"])
        assert command_path.name == f"OPEN_TRIPS_MAC_{run_dir.name}.command"
        assert command_path == Path(env["TRIPPY_OUTPUT"]) / "deliver" / run_dir.name / command_path.name
        assert command_path.exists()
        assert os.access(command_path, os.X_OK)
        assert not (report_dir / "VIEWER_LAUNCHER_FAILED.txt").exists()
    else:
        # Viewer binary not built on whatever machine runs this suite -- requirement 2:
        # this must never fail the run, just leave a visible note.
        assert "note" in viewer
        assert (report_dir / "VIEWER_LAUNCHER_FAILED.txt").read_text() == viewer["note"] + "\n"

    # Requirement 4: viewer launcher listed FIRST, dolly/honesty/export kept (cheap).
    assert len(report["deliveries"]) == 4
    assert report["deliveries"][0]["name"] == f"{run_dir.name}-viewer"
    for delivery in report["deliveries"]:
        assert delivery["status"].startswith("skipped") or delivery["status"].startswith("failed")

    readme_text = (run_dir / "README.md").read_text()
    assert f"## Report: epoch {report['epoch']}" in readme_text
    assert "### Deliveries" in readme_text
    assert "| Metric | Baseline | Candidate |" in readme_text
    assert report["summary_line"] in readme_text
    # The viewer launcher bullet must precede the dolly one in the README text.
    deliveries_section = readme_text.split("### Deliveries", 1)[1]
    assert deliveries_section.index(f"{run_dir.name}-viewer") < deliveries_section.index(f"{run_dir.name}-dolly")


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
