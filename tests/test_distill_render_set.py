"""Tests for trippy.distill.render_set: end-to-end render + COLMAP-model write on CPU.

Module: tests.test_distill_render_set
Invariants under test: `render_distill_set` builds the camera plan from the
    checkpoint's own scene/width, renders every anchor + interpolated pose
    (via `render_candidate`, never re-derived here), writes one PNG per
    pose under `images/` (named via `trippy.distill.cameras.image_filename`),
    writes a loadable COLMAP text model under `sparse_txt/`, exports the
    checkpoint's own trained point cloud, and records every count in both
    its return value and `distill_report.json`.
Fixture: the shared synthetic scene/ply/config builders from
    tests/test_train_helpers.py and a checkpoint saved straight after
    `Trainer.__init__` (mirrors tests/test_cli_candidate_report.py's
    `_build_untrained_checkpoint` -- a randomly initialised model is enough
    to exercise the render/write pipeline; no training needed). Never a
    real Splats scene or checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_train_helpers import N_IMAGES, build_synthetic_ply, build_synthetic_scene, tiny_train_config

from trippy.distill.cameras import image_filename
from trippy.distill.render_set import render_distill_set
from trippy.scene import colmap_io
from trippy.train.trainer import Trainer


def _build_untrained_checkpoint(tmp_path: Path) -> Path:
    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(scene_root, ply_path, tmp_path / "run", tmp_path / "cache")
    trainer = Trainer(cfg)
    return trainer.save_checkpoint()


def test_render_distill_set_writes_full_dataset(tmp_path: Path) -> None:
    checkpoint = _build_untrained_checkpoint(tmp_path)
    out_dir = tmp_path / "distill_out"

    report = render_distill_set(checkpoint, out_dir, device="cpu", interp_k=2)

    n_pairs = N_IMAGES - 1
    assert report["n_anchor_images"] == N_IMAGES
    assert report["n_interpolated_images"] == 2 * n_pairs
    assert report["n_skipped_pairs"] == 0
    assert report["n_cameras"] == 1  # one physical camera in the synthetic scene

    assert Path(report["trips_export_ply"]).exists()
    assert Path(report["trips_export_ply"]) == out_dir / "trips_export.ply"

    images_dir = Path(report["images_dir"])
    png_files = sorted(p.name for p in images_dir.glob("*.png"))
    assert len(png_files) == N_IMAGES + 2 * n_pairs
    assert image_filename("IMG_0.jpg") in png_files

    sparse_dir = Path(report["sparse_dir"])
    scene = colmap_io.load_colmap_model(sparse_dir)
    assert len(scene.images) == N_IMAGES + 2 * n_pairs
    assert len(scene.cameras) == 1
    assert len(scene.points3D) == report["n_points_written"] > 0

    report_path = out_dir / "distill_report.json"
    assert report_path.exists()
    assert json.loads(report_path.read_text()) == report

    # renders/ keeps render_candidate's own full per-pose tree for inspection.
    renders_dir = out_dir / "renders"
    assert (renders_dir / "metrics.json").exists()


def test_render_distill_set_caps_init_points(tmp_path: Path) -> None:
    checkpoint = _build_untrained_checkpoint(tmp_path)
    out_dir = tmp_path / "distill_out"

    report = render_distill_set(checkpoint, out_dir, device="cpu", interp_k=0, max_init_points=5)

    assert report["n_interpolated_images"] == 0
    assert report["n_points_written"] == 5
    assert report["n_points_source"] > 5


def test_render_distill_set_raises_on_missing_scene(tmp_path: Path) -> None:
    # A checkpoint whose scene_root no longer exists should fail loudly, not
    # silently produce an empty dataset.
    checkpoint = _build_untrained_checkpoint(tmp_path)
    import shutil

    shutil.rmtree(tmp_path / "scene")
    with pytest.raises(Exception):  # noqa: B017 -- exact type depends on which cache-miss path fires first
        render_distill_set(checkpoint, tmp_path / "distill_out", device="cpu")
