"""End-to-end tests for `trippy distill` (subprocess, matching tests/test_cli_candidate_report.py's style).

Module: tests.test_cli_distill
Invariants under test: `--stage render` produces the full dataset
    (images/, sparse_txt/, trips_export.ply, distill_report.json);
    `--stage brush-cmd` (run afterwards, in the same `--out`) picks up that
    report's own scene width, writes a job script, and never fails even
    when no brush binary is built yet (`--brush-binary` overrides
    `resolve_brush_binary` so this test never depends on rust/ having been
    built); `--stage compare` writes compare.md and never crashes even
    though this synthetic scene has no real Splats installation to audit
    against; `--stage all` runs render then compare in one command.
Fixture: the shared synthetic scene/ply/config builders from
    tests/test_train_helpers.py and a checkpoint saved straight after
    `Trainer.__init__` (a randomly initialised model, per this task's
    brief -- never a real Splats scene or checkpoint).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config

from trippy.train.trainer import Trainer


def _build_untrained_checkpoint(tmp_path: Path) -> Path:
    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(scene_root, ply_path, tmp_path / "run", tmp_path / "cache")
    trainer = Trainer(cfg)
    return trainer.save_checkpoint()


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "trippy.cli", *args],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def test_cli_distill_render_stage(tmp_path: Path) -> None:
    checkpoint = _build_untrained_checkpoint(tmp_path)
    out_dir = tmp_path / "distill_out"

    result = _run(
        ["distill", "--checkpoint", str(checkpoint), "--out", str(out_dir), "--stage", "render", "--device", "cpu"]
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert (out_dir / "trips_export.ply").exists()
    assert (out_dir / "images").is_dir()
    assert (out_dir / "sparse_txt" / "cameras.txt").exists()
    report = json.loads((out_dir / "distill_report.json").read_text())
    assert report["n_anchor_images"] > 0


def test_cli_distill_brush_cmd_stage_reuses_render_report(tmp_path: Path) -> None:
    checkpoint = _build_untrained_checkpoint(tmp_path)
    out_dir = tmp_path / "distill_out"
    render_result = _run(
        ["distill", "--checkpoint", str(checkpoint), "--out", str(out_dir), "--stage", "render", "--device", "cpu"]
    )
    assert render_result.returncode == 0

    result = _run(
        [
            "distill",
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(out_dir),
            "--stage",
            "brush-cmd",
            "--brush-binary",
            "/fake/brush-cli",
            "--brush-iters",
            "500",
        ]
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "/fake/brush-cli" in result.stdout
    assert "--total-train-iters 500" in result.stdout
    job_script = out_dir / "brush_train_job.sh"
    assert job_script.exists()
    assert "/fake/brush-cli" in job_script.read_text()
    assert "scripts/gpu_submit.sh --train" in result.stdout


def test_cli_distill_brush_cmd_stage_never_crashes_without_explicit_binary(tmp_path: Path) -> None:
    """`--brush-binary` is optional -- whether or not rust/brush-trips has been built on

    this machine, `--stage brush-cmd` must exit 0 and always write the job script (this
    test deliberately does not assert on `resolve_brush_binary`'s own found/not-found
    branch, which depends on this checkout's own build state -- see
    tests/test_distill_brush_runner.py for that logic in isolation)."""
    checkpoint = _build_untrained_checkpoint(tmp_path)
    out_dir = tmp_path / "distill_out"
    render_result = _run(
        ["distill", "--checkpoint", str(checkpoint), "--out", str(out_dir), "--stage", "render", "--device", "cpu"]
    )
    assert render_result.returncode == 0

    result = _run(["distill", "--checkpoint", str(checkpoint), "--out", str(out_dir), "--stage", "brush-cmd"])
    assert result.returncode == 0
    assert "brush training command" in result.stdout
    assert (out_dir / "brush_train_job.sh").exists()


def test_cli_distill_compare_stage_pending_columns(tmp_path: Path) -> None:
    checkpoint = _build_untrained_checkpoint(tmp_path)
    out_dir = tmp_path / "distill_out"
    render_result = _run(
        ["distill", "--checkpoint", str(checkpoint), "--out", str(out_dir), "--stage", "render", "--device", "cpu"]
    )
    assert render_result.returncode == 0

    result = _run(["distill", "--checkpoint", str(checkpoint), "--out", str(out_dir), "--stage", "compare"])
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    compare_path = out_dir / "compare.md"
    assert compare_path.exists()
    table = compare_path.read_text()
    assert "| Metric | baseline | TRIPS export | distilled |" in table
    assert "pending" in table  # baseline/distilled not given on this synthetic scene


def test_cli_distill_render_stage_requires_checkpoint(tmp_path: Path) -> None:
    result = _run(["distill", "--out", str(tmp_path / "out"), "--stage", "render", "--device", "cpu"])
    assert result.returncode == 2
    assert "--checkpoint is required" in result.stderr


def test_cli_distill_all_stage_runs_render_then_compare(tmp_path: Path) -> None:
    checkpoint = _build_untrained_checkpoint(tmp_path)
    out_dir = tmp_path / "distill_out"

    result = _run(
        ["distill", "--checkpoint", str(checkpoint), "--out", str(out_dir), "--stage", "all", "--device", "cpu"]
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert (out_dir / "trips_export.ply").exists()
    assert (out_dir / "compare.md").exists()
    assert (out_dir / "brush_train_job.sh").exists()
