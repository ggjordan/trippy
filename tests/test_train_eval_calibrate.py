"""Tests for test-time photometric calibration and the per-image exposure diagnostics.

Module: tests.test_train_eval_calibrate
Invariants under test:
    - `best_global_gain` is the closed-form least-squares gain (exact on a
      synthetic pair built with a known gain), and the "brightness ratio"
      diagnostic recovers a known gain offset.
    - `Trainer.calibrate_frame` recovers a *deliberately broken* per-image
      exposure: PSNR after calibration is far above PSNR before, and is
      (nearly) independent of how wrong the starting exposure was.
    - Calibration touches photometry only: point xyz/size/features, the
      U-Net weights, the pose deltas, the response LUT and even the
      camera's own `exposures_values` are bit-identical afterwards (the
      fitted scalar lives in a local tensor, never in the module).
    - `Trainer.evaluate(calibrate=True)` reports the strict numbers *and*
      the calibrated ones, in separate keys.
All fixtures are the synthetic scene from `tests/test_train_helpers.py`
(never a real Splats scene, never a photo).
"""

from __future__ import annotations

from pathlib import Path

import torch
from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config

from trippy.train.trainer import Trainer, best_global_gain


def _build_trainer(tmp_path: Path, **overrides) -> Trainer:
    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(scene_root, ply_path, tmp_path / "run", tmp_path / "cache", **overrides)
    return Trainer(cfg)


def _parameter_snapshot(trainer: Trainer) -> dict[str, torch.Tensor]:
    """Every trainable tensor that calibration must NOT touch, cloned."""
    snapshot = {
        "xyz": trainer.point_params.xyz,
        "raw_size": trainer.point_params.raw_size,
        "raw_conf": trainer.point_params.raw_conf,
        "feat": trainer.point_params.feat,
        "background": trainer.background,
        "pose_delta": trainer.pose_params.delta,
        "exposures": trainer.camera.exposures_values,
    }
    if trainer.camera.camera_response is not None:
        snapshot["response"] = trainer.camera.camera_response.response
    for name, param in trainer.net.named_parameters():
        snapshot[f"net.{name}"] = param
    return {k: v.detach().clone() for k, v in snapshot.items() if v is not None}


# --- the closed-form gain / brightness-ratio diagnostics (pure math, no scene) ---


def test_best_global_gain_recovers_a_known_gain_exactly() -> None:
    pred = torch.rand(1, 3, 8, 9) + 0.1
    target = pred * 2.5
    assert abs(best_global_gain(pred, target) - 2.5) < 1e-5


def test_best_global_gain_is_least_squares_not_a_mean_ratio() -> None:
    # A gain that is not a constant multiple: least squares weights bright pixels more, so the
    # answer must be sum(p*t)/sum(p*p), not mean(t)/mean(p).
    pred = torch.tensor([[[[1.0, 2.0]]]])
    target = torch.tensor([[[[1.0, 6.0]]]])
    expected = (1.0 * 1.0 + 2.0 * 6.0) / (1.0 * 1.0 + 2.0 * 2.0)
    assert abs(best_global_gain(pred, target) - expected) < 1e-6
    mean_ratio = float(target.mean() / pred.mean())
    assert abs(mean_ratio - expected) > 0.1  # the two really do differ


def test_best_global_gain_respects_the_mask() -> None:
    pred = torch.ones(1, 1, 1, 2)
    target = torch.tensor([[[[3.0, 100.0]]]])
    mask = torch.tensor([[[[1.0, 0.0]]]])
    assert abs(best_global_gain(pred, target, mask) - 3.0) < 1e-6


def test_best_global_gain_of_an_all_zero_prediction_is_one() -> None:
    assert best_global_gain(torch.zeros(1, 3, 4, 4), torch.ones(1, 3, 4, 4)) == 1.0


# --- calibration on the synthetic scene ---


def test_calibration_recovers_a_broken_exposure(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    name = trainer.heldout_names[0]
    index = trainer._name_to_index[name]

    # 5.87 EV is exactly what a kk-coherent frame with no EXIF used to be initialised at
    # (a 58x gain, see Trainer._initial_exposure); it is never trained away for a held-out
    # frame, which is the artefact this whole feature exists to measure.
    with torch.no_grad():
        trainer.camera.exposures_values[index] = -5.87

    metrics = trainer.evaluate(names=[name], calibrate=True)
    row = metrics["per_image"][name]

    assert row["psnr_calibrated"] > row["psnr"] + 5.0, row
    assert row["exposure_gain"] > 50.0  # the render really was 58x too bright
    # The brightness-ratio diagnostic sees the same thing without any fitting at all.
    assert row["brightness_ratio"] < 0.5
    assert row["gain_best"] < 0.5
    assert row["psnr_gain"] > row["psnr"] + 5.0


def test_calibrated_psnr_barely_depends_on_the_broken_starting_exposure(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    name = trainer.heldout_names[0]
    index = trainer._name_to_index[name]

    calibrated = []
    for start_ev in (-5.87, -2.0, 0.0, 2.0):
        with torch.no_grad():
            trainer.camera.exposures_values[index] = start_ev
        row = trainer.evaluate(names=[name], calibrate=True)["per_image"][name]
        calibrated.append(row["psnr_calibrated"])

    assert max(calibrated) - min(calibrated) < 1.0, calibrated


def test_calibration_leaves_geometry_network_and_camera_parameters_untouched(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    name = trainer.heldout_names[0]
    with torch.no_grad():
        trainer.camera.exposures_values[trainer._name_to_index[name]] = -3.0

    before = _parameter_snapshot(trainer)
    trainer.evaluate(names=[name], calibrate=True)
    after = _parameter_snapshot(trainer)

    assert set(before) == set(after)
    for key, value in before.items():
        assert torch.equal(value, after[key]), f"{key} changed during calibration"


def test_calibrate_frame_reports_the_fitted_exposure_without_writing_it_back(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    name = trainer.heldout_names[0]
    index = trainer._name_to_index[name]
    with torch.no_grad():
        trainer.camera.exposures_values[index] = -4.0

    item = trainer.dataset[index]
    target = item["rgb"].to(torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    R, t = trainer._pose_for(item, index)
    with torch.no_grad():
        net_out, _layers, _aux = trainer._render(item["K"], R, t, target.shape[-2:])
    pred, info = trainer.calibrate_frame(net_out, target, index)

    assert pred.shape == net_out.shape
    assert info["exposure_before"] == -4.0
    assert info["exposure_after"] != info["exposure_before"]
    assert info["l1_after"] <= info["l1_before"]
    assert info["white_balance"] is None  # cfg.eval_calibrate_white_balance defaults to False
    assert float(trainer.camera.exposures_values[index].detach().reshape(-1)[0]) == -4.0


def test_calibrate_white_balance_keeps_green_pinned(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, eval_calibrate_white_balance=True)
    name = trainer.heldout_names[0]
    index = trainer._name_to_index[name]
    item = trainer.dataset[index]
    target = item["rgb"].to(torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    R, t = trainer._pose_for(item, index)
    with torch.no_grad():
        net_out, _layers, _aux = trainer._render(item["K"], R, t, target.shape[-2:])

    _pred, info = trainer.calibrate_frame(net_out, target, index, steps=10)
    assert info["white_balance"] is not None
    assert abs(info["white_balance"][1] - 1.0) < 1e-6, "green channel must stay at its reference"


# --- evaluate() plumbing ---


def test_evaluate_without_calibration_is_unchanged_and_flagged(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    metrics = trainer.evaluate()

    assert metrics["calibrated"] is False
    assert "psnr_mean_calibrated" not in metrics
    assert "shade_calibrated" not in metrics
    for row in metrics["per_image"].values():
        assert "psnr_calibrated" not in row
        # ... but the free diagnostics are always there.
        for key in ("pred_mean", "target_mean", "brightness_ratio", "gain_best", "psnr_gain"):
            assert key in row


def test_evaluate_with_calibration_keeps_both_numbers(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    strict = trainer.evaluate()
    calibrated = trainer.evaluate(calibrate=True)

    assert calibrated["calibrated"] is True
    # The strict number is computed identically whether or not calibration also runs.
    assert abs(calibrated["psnr_mean"] - strict["psnr_mean"]) < 1e-6
    assert "psnr_mean_calibrated" in calibrated
    assert calibrated["other_calibrated"]["n"] == calibrated["other"]["n"]


def test_eval_calibrate_camera_config_flag_turns_it_on(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, eval_calibrate_camera=True, eval_calibrate_steps=5)
    metrics = trainer.evaluate()
    assert metrics["calibrated"] is True
    assert all("psnr_calibrated" in row for row in metrics["per_image"].values())
