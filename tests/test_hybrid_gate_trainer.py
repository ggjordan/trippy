"""End-to-end tests for the blend gate inside `trippy.train.trainer.Trainer`.

Module: tests.test_hybrid_gate_trainer
Invariants under test (the task brief's own acceptance list):
    - `gate_scale = 0` makes the rendered image EQUAL the run's own TRIPS path,
      exactly -- the gate is switched out, not faded out.
    - `gate_scale = 2` on a saturated gate makes it EQUAL the splat wherever the
      render's alpha is 1, exactly.
    - training with a `gate_prior` moves the mean gate towards the target.
    - the gate widens only the network's OUTPUT (3 -> 4 channels); the tone
      mapper, the point cloud and the input width are untouched.
    - every eval writes a gate heatmap per frame and gate mean/percentiles into
      metrics.json.
    - a checkpoint round-trips the gate config and rebuilds the 4-channel head.
    - a frame with no render (or a dropped-out crop) is NOT blended: there is no
      splat evidence there, and blending its all-zero stand-in would paint black.
    - gate OFF is untouched: 3 output channels, no "gate" key anywhere.
All fixtures are synthetic (tests/test_hybrid_a_helpers.py); CPU only, no MPS,
no `~/Splats`, no photographs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from test_hybrid_a_helpers import hybrid_train_config

from trippy.constants import TRAIN_EVAL_GATE_DIRNAME
from trippy.hybrid import gate as gate_mod
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


def _gate_trainer(
    tmp_path: Path, render_alpha: float | None = None, **hybrid_overrides
) -> tuple[Trainer, list[str]]:
    """A tiny CPU hybrid trainer with the blend gate on."""
    overrides = {"gate": {"enabled": True}, "dropout_gaussian_p": 0.0}
    overrides.update(hybrid_overrides)
    cfg, names = hybrid_train_config(tmp_path, render_alpha=render_alpha, **overrides)
    for key, value in _LOW_LR.items():
        setattr(cfg, key, value)
    return Trainer(cfg), names


def _saturate_gate(trainer: Trainer, logit: float = 40.0) -> None:
    """Force the gate channel to a constant logit, so `sigmoid` is exactly 1 (or 0).

    Zeroing the gate row of the final 1x1 conv and setting its bias makes the
    gate independent of the image, which is what turns "the extremes are exact"
    from a statement about this particular synthetic render into a statement
    about the arithmetic.
    """
    with torch.no_grad():
        final = trainer.net.final[0]
        final.weight[3].zero_()
        final.bias[3] = logit


def _render_heldout(trainer: Trainer, name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render `name` at its own pose. Returns `(pred, net_out, gaussian_block)`."""
    frame_index = trainer._name_to_index[name]
    item = trainer.dataset[frame_index]
    height, width = int(item["rgb"].shape[0]), int(item["rgb"].shape[1])
    gaussian = trainer.hybrid.frame(name, (height, width))
    R, t = trainer._pose_for(item, frame_index)
    with torch.no_grad():
        net_out, _layers, _aux = trainer._render(item["K"], R, t, (height, width), gaussian=gaussian)
        pred, _gate = trainer._decode(net_out, frame_index, gaussian)
    return pred, net_out, gaussian


# --- channel bookkeeping ---


def test_the_gate_widens_only_the_network_output(tmp_path: Path) -> None:
    trainer, _ = _gate_trainer(tmp_path)
    assert trainer.gate_enabled
    assert trainer.net.config.num_output_channels == 4
    # Input width is the hybrid width, unchanged by the gate.
    assert trainer.net.config.num_input_channels == trainer.cfg.feature_channels + 5
    # The tone mapper still works on three channels only.
    assert trainer.camera.camera_response.response.shape[1] == 3
    assert trainer.point_params.feat.shape[1] == trainer.cfg.feature_channels


def test_gate_off_keeps_three_output_channels(tmp_path: Path) -> None:
    cfg, _ = hybrid_train_config(tmp_path, dropout_gaussian_p=0.0)
    trainer = Trainer(cfg)
    assert not trainer.gate_enabled
    assert trainer.net.config.num_output_channels == 3
    record = trainer.train_step()
    assert "gate_mean" not in record


# --- the two exact extremes (the brief's CPU acceptance test) ---


def test_gate_scale_zero_is_exactly_the_trips_path(tmp_path: Path) -> None:
    trainer, names = _gate_trainer(tmp_path)
    _saturate_gate(trainer, logit=40.0)  # gate == 1 everywhere: the worst case for scale 0
    name = names[0]
    frame_index = trainer._name_to_index[name]

    trainer.gate_scale = 0.0
    pred_zero, net_out, _gaussian = _render_heldout(trainer, name)

    # The TRIPS path is "tone-map the three colour channels and stop".
    with torch.no_grad():
        trips_only = trainer._tone_map(net_out[:, :3], frame_index)
    assert torch.equal(pred_zero, trips_only)


def test_a_saturated_gate_at_scale_two_is_exactly_the_splat(tmp_path: Path) -> None:
    # alpha = 1 everywhere: every pixel is fully covered by the Gaussian render, so
    # "the splat where alpha == 1" is the whole frame.
    trainer, names = _gate_trainer(tmp_path, render_alpha=1.0)
    _saturate_gate(trainer, logit=40.0)
    trainer.gate_scale = 2.0
    name = names[0]

    pred, _net_out, gaussian = _render_heldout(trainer, name)
    splat_rgb = gate_mod.splat_rgb_from_block(gaussian, trainer.cfg.hybrid)
    assert splat_rgb is not None
    splat_rgb = splat_rgb[..., : pred.shape[-2], : pred.shape[-1]]
    assert torch.equal(pred, splat_rgb)


def test_the_gate_is_exactly_one_when_saturated(tmp_path: Path) -> None:
    trainer, names = _gate_trainer(tmp_path, render_alpha=1.0)
    _saturate_gate(trainer, logit=40.0)
    _pred, net_out, _g = _render_heldout(trainer, names[0])
    _rgb, gate = trainer.split_net_output(net_out)
    assert gate is not None
    assert float(gate.min()) == 1.0


def test_scale_one_sits_between_the_two_extremes(tmp_path: Path) -> None:
    trainer, names = _gate_trainer(tmp_path, render_alpha=1.0)
    name = names[0]
    trainer.gate_scale = 0.0
    trips, _n, _g = _render_heldout(trainer, name)
    trainer.gate_scale = 2.0
    splat, _n, _g = _render_heldout(trainer, name)
    trainer.gate_scale = 1.0
    mid, _n, _g = _render_heldout(trainer, name)
    # A random, untrained gate is genuinely mixed, so the middle render differs from both.
    assert not torch.equal(mid, trips)
    assert float((mid - trips).abs().max()) > 0.0
    lo, hi = torch.minimum(trips, splat), torch.maximum(trips, splat)
    assert bool(((mid >= lo - 1e-5) & (mid <= hi + 1e-5)).all())


# --- no splat means no blend ---


def test_a_frame_with_no_render_is_not_blended(tmp_path: Path) -> None:
    trainer, _names = _gate_trainer(tmp_path)
    _saturate_gate(trainer, logit=40.0)
    trainer.gate_scale = 2.0
    pred = torch.rand(1, 3, 4, 4)
    gate = torch.ones(1, 1, 4, 4)
    # `gaussian=None` is a missing render / dropped crop: there is no splat evidence,
    # so the prediction must come back untouched rather than blended towards black.
    out, returned = trainer.apply_gate(pred, gate, None)
    assert torch.equal(out, pred)
    assert returned is not None


def test_a_dropped_out_crop_trains_without_a_blend(tmp_path: Path) -> None:
    trainer, names = _gate_trainer(tmp_path, dropout_gaussian_p=1.0)
    record = trainer.train_step(name=names[0], zoom=1.0, center=(8.0, 6.0))
    assert record["gaussian_dropped"] is True
    assert record["gaussian_present"] is False
    # The gate is still produced and logged; it is simply not applied.
    assert "gate_mean" in record
    assert np.isfinite(record["loss"])


# --- the prior ---


def test_the_gate_prior_moves_the_mean_gate_towards_its_target(tmp_path: Path) -> None:
    def mean_gate_after(target: float, weight: float, steps: int = 30) -> tuple[float, float]:
        trainer, names = _gate_trainer(
            tmp_path / f"prior-{target}-{weight}",
            gate_prior={"target": target, "weight": weight},
        )
        trainer.cfg.lr_network = 5e-2
        for group in trainer.optimizer.param_groups:
            group["lr"] = 5e-2
        before = float(trainer.train_step(name=names[0], zoom=1.0, center=(8.0, 6.0))["gate_mean"])
        after = before
        for _ in range(steps):
            after = float(trainer.train_step(name=names[0], zoom=1.0, center=(8.0, 6.0))["gate_mean"])
        return before, after

    # Towards 1 (all splat).
    before_up, after_up = mean_gate_after(target=1.0, weight=50.0)
    assert after_up > before_up
    # Towards 0 (all TRIPS), from the same starting point.
    before_down, after_down = mean_gate_after(target=0.0, weight=50.0)
    assert after_down < before_down


def test_the_prior_is_recorded_and_is_zero_when_off(tmp_path: Path) -> None:
    trainer, names = _gate_trainer(tmp_path)
    record = trainer.train_step(name=names[0], zoom=1.0, center=(8.0, 6.0))
    assert record["gate_prior"] == 0.0
    assert 0.0 <= record["gate_mean"] <= 1.0


# --- eval artifacts ---


def test_every_eval_writes_a_gate_heatmap_and_logs_the_stats(tmp_path: Path) -> None:
    trainer, _ = _gate_trainer(tmp_path)
    metrics = trainer.evaluate(epoch=0)

    assert "gate" in metrics
    stats = metrics["gate"]
    assert 0.0 <= stats["mean"] <= 1.0
    assert stats["gate_scale"] == 1.0
    assert stats["n_frames"] == metrics["n_images"]
    for key in ("p5", "p50", "p95"):
        assert 0.0 <= stats["percentiles"][key] <= 1.0

    gate_dir = trainer.run_dir / "eval_ep0000" / TRAIN_EVAL_GATE_DIRNAME
    written = sorted(p.name for p in gate_dir.glob("*.gate.png"))
    assert len(written) == metrics["n_images"]

    # Per image, both the raw and the effective (post-gate_scale) summary.
    for row in metrics["per_image"].values():
        assert 0.0 <= row["gate"]["mean"] <= 1.0
        assert row["gate_effective"]["mean"] == row["gate"]["mean"]  # scale 1 -> identical
        assert row["gate_splat_present"] is True


def test_the_effective_gate_stats_follow_gate_scale(tmp_path: Path) -> None:
    trainer, _ = _gate_trainer(tmp_path)
    trainer.gate_scale = 0.0
    metrics = trainer.evaluate(epoch=1)
    assert metrics["gate"]["gate_scale"] == 0.0
    for row in metrics["per_image"].values():
        assert row["gate_effective"]["mean"] == 0.0
        assert row["gate"]["mean"] > 0.0


def test_gate_off_writes_no_gate_key_or_heatmaps(tmp_path: Path) -> None:
    cfg, _ = hybrid_train_config(tmp_path, dropout_gaussian_p=0.0)
    trainer = Trainer(cfg)
    metrics = trainer.evaluate(epoch=0)
    assert "gate" not in metrics
    assert not (trainer.run_dir / "eval_ep0000" / TRAIN_EVAL_GATE_DIRNAME).exists()
    for row in metrics["per_image"].values():
        assert "gate" not in row


# --- checkpoints ---


def test_a_checkpoint_round_trips_the_gate(tmp_path: Path) -> None:
    trainer, _ = _gate_trainer(tmp_path, gate_prior={"target": 0.25, "weight": 3.0})
    trainer.cfg.hybrid.gate_scale = 1.5
    trainer.train_step()
    path = trainer.save_checkpoint()

    reloaded = build_trainer_from_checkpoint(path, device="cpu")
    assert reloaded.gate_enabled
    assert reloaded.net.config.num_output_channels == 4
    assert reloaded.cfg.hybrid.gate.enabled is True
    assert reloaded.cfg.hybrid.gate_prior.target == 0.25
    assert reloaded.cfg.hybrid.gate_prior.weight == 3.0
    # The run's own default mix is what a fresh Trainer starts at.
    assert reloaded.gate_scale == 1.5
    for a, b in zip(trainer.net.parameters(), reloaded.net.parameters(), strict=True):
        assert torch.equal(a, b)


# --- render_at_pose carries the gate ---


def test_render_at_pose_returns_the_gate_in_aux(tmp_path: Path) -> None:
    trainer, names = _gate_trainer(tmp_path)
    name = names[0]
    frame_index = trainer._name_to_index[name]
    item = trainer.dataset[frame_index]
    height, width = int(item["rgb"].shape[0]), int(item["rgb"].shape[1])
    R, t = trainer._pose_for(item, frame_index)
    with torch.no_grad():
        pred, _layers, aux = trainer.render_at_pose(
            item["K"], R, t, (height, width), frame_index=frame_index, image_name=name
        )
    assert "gate" in aux
    assert aux["gate"].shape[-2:] == pred.shape[-2:]
    assert bool(((aux["gate"] >= 0.0) & (aux["gate"] <= 1.0)).all())
