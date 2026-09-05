"""Tests for trippy.train.trainer.Trainer: train_step, evaluate, checkpoint, export.

Module: tests.test_train_trainer
Invariants under test: `train_step` (with a pinned crop) decreases loss
    over 5 steps on the synthetic scene; `evaluate` writes `metrics.json`
    + `sheet.png` under `<run_dir>/eval_ep<N>/`; `save_checkpoint` +
    `resume` on a freshly constructed Trainer reproduces identical
    parameters; `export_ply` round-trips through `GaussianPlySource`.
All fixtures are the synthetic scene from `tests/test_train_helpers.py`
(never a real Splats scene).
"""

from __future__ import annotations

from pathlib import Path

import torch
from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config

from trippy.points.gaussian_ply import GaussianPlySource
from trippy.train.trainer import Trainer


def _build_trainer(tmp_path: Path, **overrides) -> Trainer:
    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(scene_root, ply_path, tmp_path / "run", tmp_path / "cache", **overrides)
    return Trainer(cfg)


def test_train_step_decreases_loss_over_five_steps(tmp_path: Path) -> None:
    trainer = _build_trainer(
        tmp_path,
        lr_texture=1e-2,
        lr_network=1e-3,
        lr_points=0.0,
        lr_size=0.0,
        lr_confidence=0.0,
        lr_poses=0.0,
        lr_background=0.0,
        lr_exposure=0.0,
        lr_response=0.0,
        extent_penalty_weight=0.0,
    )
    name = trainer.train_names[0]
    losses = [trainer.train_step(name=name, zoom=1.0, center=(24.0, 18.0))["loss"] for _ in range(5)]
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_train_step_returns_expected_metric_keys(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    record = trainer.train_step()
    for key in ("step", "epoch", "image", "loss", "image_loss", "extent_penalty", "camera_reg"):
        assert key in record
    assert trainer.metrics_path.exists()
    assert trainer.log_path.parent.exists()


def test_evaluate_writes_metrics_and_sheet(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    metrics = trainer.evaluate()

    assert metrics["n_images"] == len(trainer.heldout_names)
    assert metrics["psnr_mean"] > 0.0
    assert -1.0 <= metrics["ssim_mean"] <= 1.0  # SSIM's range, not clipped to [0, 1] (untrained net output)
    assert metrics["lpips_mean"] is None  # eval_lpips=False in tiny_train_config

    eval_dir = trainer.run_dir / "eval_ep0000"
    assert (eval_dir / "metrics.json").exists()
    assert (eval_dir / "sheet.png").exists()


def test_evaluate_prioritises_forced_heldout_in_sheet(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, forced_heldout=["IMG_1.jpg"], heldout_k=8)
    assert "IMG_1.jpg" in trainer.heldout_names
    metrics = trainer.evaluate()
    assert "IMG_1.jpg" in metrics["names"]


def test_checkpoint_save_and_resume_reproduces_state(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    trainer.train_step()
    trainer.train_step()
    ckpt_path = trainer.save_checkpoint(epoch=3)
    assert ckpt_path.exists()

    scene_root, point_set = build_synthetic_scene(tmp_path / "scene2")
    ply_path = build_synthetic_ply(tmp_path / "scene2", point_set)
    cfg2 = tiny_train_config(scene_root, ply_path, tmp_path / "run2", tmp_path / "cache2")
    resumed = Trainer(cfg2)
    resumed.resume(ckpt_path)

    assert resumed.epoch == 3
    torch.testing.assert_close(resumed.point_params.xyz, trainer.point_params.xyz)
    torch.testing.assert_close(resumed.point_params.feat, trainer.point_params.feat)
    torch.testing.assert_close(resumed.net.state_dict()["final.0.weight"], trainer.net.state_dict()["final.0.weight"])
    torch.testing.assert_close(resumed.background, trainer.background)


def test_export_ply_round_trips_via_gaussian_ply_source(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    out_path = trainer.export_ply()
    assert out_path.exists()

    read_back = GaussianPlySource(out_path, min_opacity=0.0).build()
    assert len(read_back) == len(trainer.point_params)
    import numpy as np

    np.testing.assert_allclose(read_back.xyz, trainer.point_params.xyz.detach().numpy(), atol=1e-4)
    np.testing.assert_allclose(
        read_back.size0, trainer.point_params.size().detach().numpy(), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(read_back.conf0, trainer.point_params.conf().detach().numpy(), atol=1e-3)


def test_fit_runs_full_schedule_and_produces_export(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    metrics = trainer.fit()
    assert trainer.epoch == trainer.cfg.epochs
    assert metrics  # at least one eval ran
    assert (trainer.run_dir / "export.ply").exists()
    checkpoints = list((trainer.run_dir / "checkpoints").glob("checkpoint_ep*.pt"))
    assert len(checkpoints) >= 1


def test_fit_respects_max_minutes_budget(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, epochs=1000)
    metrics = trainer.fit(max_minutes=1e-6)  # effectively "stop immediately"
    assert trainer.epoch < trainer.cfg.epochs
    assert (trainer.run_dir / "export.ply").exists()
    assert metrics
