"""Tests for trippy.render.candidate.render_candidate and `trippy candidate-report`.

Module: tests.test_cli_candidate_report
Invariants under test:
    - `render_candidate` writes raw/net/coverage/honesty PNGs per pose, a
      metrics.json matching its own return value, and (when
      `write_video_files=True` and poses share one image size) `dolly.mp4`
      + `dolly_raw.mp4`; `write_video_files=False` skips both videos.
    - `trippy candidate-report` (end-to-end subprocess, matching the style
      of tests/test_cli_render.py and tests/test_train_cli.py) exits 0 on a
      synthetic scene and a randomly-initialised (never trained) checkpoint,
      writing export.ply, dolly/ and offpath/ render trees, report.json and
      README.md; the shade audit numbers may fail gracefully (this
      synthetic scene has no points3D/observations for the audit script's
      shade frames) but must never crash the command -- `audits.shade_audit`
      always reports an "error" instead.
Fixture: the shared synthetic scene/ply/config builders from
    tests/test_train_helpers.py, and a checkpoint saved straight after
    `Trainer.__init__` (no `fit()`/`train_step()` call) -- i.e. a randomly
    initialised model, per this task's brief. Never a real Splats scene or
    checkpoint.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config

from trippy.render.candidate import render_candidate
from trippy.render.dolly import shade_dolly_poses
from trippy.render.offpath import offpath_poses
from trippy.train.trainer import Trainer


def _build_untrained_checkpoint(tmp_path: Path) -> Path:
    """A checkpoint saved immediately after construction: a randomly initialised model."""
    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(scene_root, ply_path, tmp_path / "run", tmp_path / "cache")
    trainer = Trainer(cfg)
    return trainer.save_checkpoint()


def test_render_candidate_writes_expected_artifacts(tmp_path: Path) -> None:
    checkpoint = _build_untrained_checkpoint(tmp_path)
    scene_root = tmp_path / "scene"
    poses = shade_dolly_poses(scene_root, pose_name="IMG_0.jpg", n=3, width=48)

    out_dir = tmp_path / "render_out"
    metrics = render_candidate(checkpoint, poses, out_dir, device="cpu", write_video_files=True)

    assert metrics["n_frames"] == 3
    assert len(metrics["frames"]) == 3
    assert 0.0 <= metrics["mean_coverage_full"] <= 1.0
    assert (out_dir / "metrics.json").exists()
    assert json.loads((out_dir / "metrics.json").read_text()) == metrics

    for pose in poses:
        frame_dir = out_dir / "frames" / pose.name
        assert (frame_dir / "raw_level0.png").exists()
        assert (frame_dir / "net.png").exists()
        assert (frame_dir / "coverage.png").exists()
        assert (frame_dir / "honesty.png").exists()

    assert (out_dir / "dolly.mp4").exists()
    assert (out_dir / "dolly_raw.mp4").exists()
    assert metrics["videos"]["net"] == str(out_dir / "dolly.mp4")
    assert metrics["videos"]["raw"] == str(out_dir / "dolly_raw.mp4")
    assert (out_dir / "honesty_sheet.png").exists()
    assert metrics["honesty_sheet"] == str(out_dir / "honesty_sheet.png")


def test_render_candidate_skips_video_when_disabled(tmp_path: Path) -> None:
    checkpoint = _build_untrained_checkpoint(tmp_path)
    scene_root = tmp_path / "scene"
    poses = offpath_poses(scene_root, ["IMG_0.jpg", "IMG_1.jpg"], width=48)

    out_dir = tmp_path / "render_out"
    metrics = render_candidate(checkpoint, poses, out_dir, device="cpu", write_video_files=False)

    assert metrics["n_frames"] == 4  # lateral + oblique per image
    assert "videos" not in metrics
    assert not (out_dir / "dolly.mp4").exists()
    assert not (out_dir / "dolly_raw.mp4").exists()
    assert (out_dir / "honesty_sheet.png").exists()


def test_cli_candidate_report_end_to_end(tmp_path: Path) -> None:
    checkpoint = _build_untrained_checkpoint(tmp_path)
    out_dir = tmp_path / "report_out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "trippy.cli",
            "candidate-report",
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(out_dir),
            "--dolly-pose",
            "IMG_0.jpg",
            "--offpath",
            "IMG_0.jpg,IMG_1.jpg",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "JSON:" in result.stdout

    assert (out_dir / "export.ply").exists()
    assert (out_dir / "report.json").exists()
    assert (out_dir / "README.md").exists()

    report = json.loads((out_dir / "report.json").read_text())
    for key in ("checkpoint", "device", "scene_root", "export_ply", "dolly", "offpath", "audits"):
        assert key in report

    # This synthetic scene has no points3D/observations for the shade audit's frames, so
    # the audit must degrade to a recorded error rather than crashing the whole command.
    assert "error" in report["audits"]["shade_audit"]
    assert "extent_gate" in report["audits"]  # may succeed (real numbers) or fail gracefully

    dolly_dir = out_dir / "dolly"
    assert (dolly_dir / "dolly.mp4").exists()
    assert (dolly_dir / "dolly_raw.mp4").exists()
    assert (dolly_dir / "honesty_sheet.png").exists()
    assert (dolly_dir / "metrics.json").exists()

    offpath_dir = out_dir / "offpath"
    assert not (offpath_dir / "dolly.mp4").exists()
    assert (offpath_dir / "honesty_sheet.png").exists()
    assert (offpath_dir / "metrics.json").exists()

    readme = (out_dir / "README.md").read_text()
    assert "trippy candidate report" in readme
