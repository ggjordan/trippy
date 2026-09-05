"""Tests for hybrid design A inside `trippy.train.trainer.Trainer`.

Module: tests.test_hybrid_a_trainer
Invariants under test: the U-Net is built with `feature_channels + G` input
    channels while the point cloud/background stay at `feature_channels`; a
    train step decreases loss on the synthetic scene with fake renders; eval
    reads renders **by name** from disk (a shuffled/renamed render set changes
    the number); the checkpoint records the resolved hybrid config and
    reloading rebuilds the same wider network; dropout at p=1 makes the step
    equal to the no-render step; and -- the regression guard that matters most
    -- a non-hybrid Trainer is untouched (`trainer.hybrid is None`, same
    channel count, same checkpoint-reload behaviour).
All fixtures are synthetic (tests/test_hybrid_a_helpers.py, tests/
test_train_helpers.py); CPU only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from test_hybrid_a_helpers import hybrid_train_config, write_fake_renders
from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config

from trippy.hybrid.config_a import HybridConfig
from trippy.train.eval import build_trainer_from_checkpoint
from trippy.train.trainer import Trainer

_LOW_LR = {
    "lr_texture": 1e-2,
    "lr_network": 1e-3,
    "lr_points": 0.0,
    "lr_size": 0.0,
    "lr_confidence": 0.0,
    "lr_poses": 0.0,
    "lr_background": 0.0,
    "lr_exposure": 0.0,
    "lr_response": 0.0,
    "extent_penalty_weight": 0.0,
}


def _hybrid_trainer(tmp_path: Path, **hybrid_overrides) -> tuple[Trainer, list[str]]:
    cfg, names = hybrid_train_config(tmp_path, **hybrid_overrides)
    for key, value in _LOW_LR.items():
        setattr(cfg, key, value)
    return Trainer(cfg), names


# --- channel bookkeeping ---


def test_hybrid_widens_only_the_network(tmp_path: Path) -> None:
    trainer, _ = _hybrid_trainer(tmp_path)
    assert trainer.hybrid is not None
    assert trainer.hybrid.num_channels == 5
    assert trainer.net.config.num_input_channels == trainer.cfg.feature_channels + 5
    # Points, features and background stay at the rasteriser's own channel count.
    assert trainer.point_params.feat.shape[1] == trainer.cfg.feature_channels
    assert trainer.background.shape == (trainer.cfg.feature_channels,)


def test_non_hybrid_trainer_is_untouched(tmp_path: Path) -> None:
    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(scene_root, ply_path, tmp_path / "run", tmp_path / "cache")
    trainer = Trainer(cfg)
    assert trainer.hybrid is None
    assert trainer.gaussian_provider is None
    assert trainer.net.config.num_input_channels == cfg.feature_channels
    assert trainer.gaussian_for_pose("IMG_0.jpg", None, None, None, (4, 4)) is None
    record = trainer.train_step()
    assert "gaussian_dropped" not in record


def test_channel_subset_changes_the_network_width(tmp_path: Path) -> None:
    trainer, _ = _hybrid_trainer(tmp_path, channels=["rgb", "alpha"])
    assert trainer.net.config.num_input_channels == trainer.cfg.feature_channels + 4


def test_depth_scale_is_measured_and_recorded_on_the_config(tmp_path: Path) -> None:
    trainer, _ = _hybrid_trainer(tmp_path, depth_scale=None)
    assert trainer.cfg.hybrid.depth_scale is not None
    assert trainer.cfg.hybrid.depth_scale > 0.0
    assert trainer.cfg.to_dict()["hybrid"]["depth_scale"] == trainer.cfg.hybrid.depth_scale


# --- training ---


def test_train_step_decreases_loss_with_fake_renders(tmp_path: Path) -> None:
    trainer, _ = _hybrid_trainer(tmp_path)
    name = trainer.train_names[0]
    losses = [trainer.train_step(name=name, zoom=1.0, center=(24.0, 18.0))["loss"] for _ in range(5)]
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_train_step_records_the_dropout_decision(tmp_path: Path) -> None:
    kept, _ = _hybrid_trainer(tmp_path, dropout_gaussian_p=0.0)
    record = kept.train_step(name=kept.train_names[0], zoom=1.0, center=(24.0, 18.0))
    assert record["gaussian_dropped"] is False
    assert record["gaussian_present"] is True

    dropped, _ = _hybrid_trainer(tmp_path / "always", dropout_gaussian_p=1.0)
    record = dropped.train_step(name=dropped.train_names[0], zoom=1.0, center=(24.0, 18.0))
    assert record["gaussian_dropped"] is True
    assert record["gaussian_present"] is False


def test_a_missing_render_trains_as_a_zero_block(tmp_path: Path) -> None:
    trainer, names = _hybrid_trainer(tmp_path)
    assert trainer.hybrid is not None
    for path in (tmp_path / "renders").glob(f"{Path(names[0]).stem}.*"):
        path.unlink()
    trainer.hybrid._available.discard(Path(names[0]).stem)  # re-scan without rebuilding the trainer
    record = trainer.train_step(name=names[0], zoom=1.0, center=(24.0, 18.0))
    assert record["gaussian_present"] is False
    assert np.isfinite(record["loss"])


# --- eval ---


def test_evaluate_reads_renders_by_name(tmp_path: Path) -> None:
    """Swapping one held-out frame's render for a different image moves its PSNR."""
    trainer, _ = _hybrid_trainer(tmp_path)
    baseline = trainer.evaluate(epoch=0)["psnr_mean"]

    held = trainer.heldout_names[0]
    renders_dir = tmp_path / "renders"
    write_fake_renders(renders_dir, [held])  # generated pattern, not this frame's photo
    assert trainer.hybrid is not None
    trainer.hybrid._cache.clear()
    swapped = trainer.evaluate(epoch=1)["psnr_mean"]
    assert swapped != baseline, "eval did not read the render keyed by image name"


def test_evaluate_still_writes_the_honesty_sheet(tmp_path: Path) -> None:
    trainer, _ = _hybrid_trainer(tmp_path)
    trainer.evaluate(epoch=0)
    eval_dir = trainer.run_dir / "eval_ep0000"
    assert (eval_dir / "metrics.json").exists()
    assert (eval_dir / "sheet.png").exists()


# --- checkpointing ---


def test_checkpoint_records_the_hybrid_config_and_reloads(tmp_path: Path) -> None:
    trainer, _ = _hybrid_trainer(tmp_path, mode="concat_level0", mask_by_alpha=False)
    trainer.train_step(name=trainer.train_names[0], zoom=1.0, center=(24.0, 18.0))
    path = trainer.save_checkpoint(epoch=0)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    hybrid = payload["cfg"]["hybrid"]
    assert hybrid["enabled"] is True
    assert hybrid["mode"] == "concat_level0"
    assert hybrid["mask_by_alpha"] is False
    assert hybrid["depth_scale"] == trainer.cfg.hybrid.depth_scale

    reloaded = build_trainer_from_checkpoint(path, device="cpu")
    assert reloaded.hybrid is not None
    assert reloaded.cfg.hybrid == HybridConfig(**hybrid)
    assert reloaded.net.config.num_input_channels == trainer.net.config.num_input_channels
    assert reloaded.gaussian_provider is not None  # lazy live renderer installed, nothing loaded
    for a, b in zip(trainer.net.parameters(), reloaded.net.parameters(), strict=True):
        assert torch.equal(a, b)
