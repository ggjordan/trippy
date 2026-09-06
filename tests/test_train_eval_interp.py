"""Tests for `exposure_mode="neighbours"` -- TRIPS's `interpolate_eval_settings` ported.

Module: tests.test_train_eval_interp
Invariants under test:
    - `trippy.net.camera_model.nearest_train_neighbour_distances`/
      `interpolate_from_train_neighbours` reproduce, in closed form, the fixed point of
      `NeuralCameraImpl::InterpolateFromNeighbors` (third_party/TRIPS/src/lib/models/
      NeuralCamera.cpp:481-520): a held-out index's replacement value is the linear
      interpolation, by dataset-index distance, between its nearest TRAINING neighbours,
      walking backward/forward *circularly* (wrap-around at both ends, matching the C++
      `(i-1)>=0 ? i-1 : n-1` / `(i+1)<n ? i+1 : 0`) -- covering a held-out frame at the
      first index, the last index, a middle index, and the "alternate" forced-hold-out
      protocol's every-other pattern (docs/EXPERIMENTS.md "Forced hold-out protocols").
    - The helper falls back to `values.mean(dim=0)` (a scene-mean) when there is no
      training frame anywhere to interpolate from.
    - `Trainer.evaluate`'s "_eval"-suffixed fields (the headline number under this
      feature -- see trainer.py's own module docstring / `evaluate`'s docstring) are
      identical to the plain "own" fields for a TRAINING-set name regardless of the
      requested `exposure_mode` (TRIPS's own `InterpolateFromNeighbors` is likewise only
      ever called on `not_training_indices`), and record which mode produced each row via
      the per-image `"exposure_mode"` key.
    - `exposure_mode="neighbours"` recovers a deliberately broken held-out exposure without
      ever touching that frame's own photo (the recovery comes entirely from its TRAINING
      neighbours' rows).
    - The plain "own" `psnr`/`ssim`/`lpips`/`psnr_mean`/`shade`/`other` fields are
      byte-for-byte unaffected by `exposure_mode` (this is what
      tests/test_train_regression.py and tests/test_train_eval_calibrate.py assume, and
      this feature must not disturb them).
All fixtures are the synthetic scene from `tests/test_train_helpers.py` (never a real
Splats scene, never a photo).
"""

from __future__ import annotations

from pathlib import Path

import torch
from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config

from trippy.net.camera_model import (
    interpolate_from_train_neighbours,
    nearest_train_neighbour_distances,
)
from trippy.train.config import TrainConfig
from trippy.train.trainer import Trainer

# --- pure helper: nearest_train_neighbour_distances / interpolate_from_train_neighbours ---


def test_default_eval_exposure_mode_is_neighbours() -> None:
    assert TrainConfig().eval_exposure_mode == "neighbours"


def test_neighbours_first_index_held_out_wraps_to_the_last_training_frame() -> None:
    # index 0 held out, 1/2/3 training: previous wraps around to index 3 (distance 1), next
    # is index 1 (distance 1) -- both immediate neighbours in capture order.
    is_train = [False, True, True, True]
    prev_idx, next_idx, d_prev, d_next = nearest_train_neighbour_distances(is_train, 0)
    assert (prev_idx, next_idx, d_prev, d_next) == (3, 1, 1, 1)

    values = torch.tensor([0.0, 10.0, 20.0, 30.0])
    interpolated = interpolate_from_train_neighbours(values, is_train, 0)
    assert torch.allclose(interpolated, torch.tensor(20.0))  # avg(values[3]=30, values[1]=10)


def test_neighbours_last_index_held_out_wraps_to_the_first_training_frame() -> None:
    is_train = [True, True, True, False]
    prev_idx, next_idx, d_prev, d_next = nearest_train_neighbour_distances(is_train, 3)
    assert (prev_idx, next_idx, d_prev, d_next) == (2, 0, 1, 1)

    values = torch.tensor([0.0, 10.0, 20.0, 30.0])
    interpolated = interpolate_from_train_neighbours(values, is_train, 3)
    assert torch.allclose(interpolated, torch.tensor(10.0))  # avg(values[2]=20, values[0]=0)


def test_neighbours_middle_index_held_out_averages_its_immediate_neighbours() -> None:
    is_train = [True, False, True]
    prev_idx, next_idx, d_prev, d_next = nearest_train_neighbour_distances(is_train, 1)
    assert (prev_idx, next_idx, d_prev, d_next) == (0, 2, 1, 1)

    values = torch.tensor([2.0, 999.0, 8.0])
    interpolated = interpolate_from_train_neighbours(values, is_train, 1)
    assert torch.allclose(interpolated, torch.tensor(5.0))


def test_neighbours_alternate_protocol_every_held_out_frame_averages_its_two_train_neighbours() -> None:
    # trippy.scene.splits.partition_forced's "alternate" mode holds out every other frame --
    # exactly this pattern (docs/EXPERIMENTS.md "Forced hold-out protocols").
    is_train = [True, False, True, False, True, False]
    values = torch.tensor([0.0, 100.0, 10.0, 100.0, 20.0, 100.0])
    for held_out_index, expected in ((1, 5.0), (3, 15.0)):
        prev_idx, next_idx, d_prev, d_next = nearest_train_neighbour_distances(is_train, held_out_index)
        assert d_prev == 1 and d_next == 1
        interpolated = interpolate_from_train_neighbours(values, is_train, held_out_index)
        assert torch.allclose(interpolated, torch.tensor(expected)), (held_out_index, interpolated)
    # Index 5 (also held out) wraps forward to index 0 and backward to index 4.
    prev_idx, next_idx, d_prev, d_next = nearest_train_neighbour_distances(is_train, 5)
    assert (prev_idx, next_idx, d_prev, d_next) == (4, 0, 1, 1)


def test_neighbours_asymmetric_gap_weights_the_closer_training_frame_more() -> None:
    # index 1 is 1 step behind training index 0 but 2 steps ahead of training index 3 (index 2
    # is held out too) -- the interpolated value must lean toward the CLOSER frame (index 0).
    is_train = [True, False, False, True, False]
    prev_idx, next_idx, d_prev, d_next = nearest_train_neighbour_distances(is_train, 1)
    assert (prev_idx, next_idx, d_prev, d_next) == (0, 3, 1, 2)

    values = torch.tensor([0.0, 1.0, 1.0, 90.0, 1.0])
    interpolated = interpolate_from_train_neighbours(values, is_train, 1)
    # weight toward index 0 is d_next/(d_prev+d_next) = 2/3, toward index 3 is 1/3:
    # (2/3)*0 + (1/3)*90 = 30 -- closer to the near (index 0) value than a plain average.
    assert torch.allclose(interpolated, torch.tensor(30.0))


def test_neighbours_only_one_training_frame_in_the_dataset_uses_it_unweighted() -> None:
    is_train = [True, False, False, False, False]
    prev_idx, next_idx, _d_prev, _d_next = nearest_train_neighbour_distances(is_train, 1)
    assert prev_idx == next_idx == 0
    values = torch.tensor([7.0, 1.0, 1.0, 1.0, 1.0])
    interpolated = interpolate_from_train_neighbours(values, is_train, 1)
    assert torch.allclose(interpolated, torch.tensor(7.0))  # the single training value, unweighted


def test_neighbours_falls_back_to_scene_mean_when_no_training_frame_exists() -> None:
    is_train = [False, False, False, False]
    prev_idx, next_idx, d_prev, d_next = nearest_train_neighbour_distances(is_train, 0)
    assert prev_idx is None and next_idx is None
    assert d_prev == d_next == len(is_train)

    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    interpolated = interpolate_from_train_neighbours(values, is_train, 0)
    assert torch.allclose(interpolated, values.mean(dim=0))


def test_neighbours_handles_trailing_dims_like_white_balance_rows() -> None:
    # white_balance_values is (n, 3, 1, 1) -- the helper must broadcast over trailing dims,
    # not just scalars.
    is_train = [True, False, True]
    values = torch.zeros(3, 3, 1, 1)
    values[0] = torch.tensor([1.0, 0.5, 0.5]).view(3, 1, 1)
    values[2] = torch.tensor([1.0, 1.5, 1.5]).view(3, 1, 1)
    interpolated = interpolate_from_train_neighbours(values, is_train, 1)
    expected = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)
    assert torch.allclose(interpolated, expected)


# --- Trainer.evaluate(exposure_mode=...) ---


def _build_trainer(tmp_path: Path, **overrides) -> Trainer:
    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(scene_root, ply_path, tmp_path / "run", tmp_path / "cache", **overrides)
    return Trainer(cfg)


def test_evaluate_records_exposure_mode_per_row_and_own_for_training_rows(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    heldout_name = trainer.heldout_names[0]
    train_name = trainer.train_names[0]

    metrics = trainer.evaluate(names=[heldout_name, train_name], exposure_mode="neighbours")

    assert metrics["exposure_mode"] == "neighbours"
    assert metrics["per_image"][heldout_name]["exposure_mode"] == "neighbours"
    # A training-set name always reports "own", regardless of the requested mode -- TRIPS's
    # InterpolateFromNeighbors is likewise only ever called on not_training_indices.
    assert metrics["per_image"][train_name]["exposure_mode"] == "own"


def test_evaluate_mode_own_is_bit_identical_to_the_plain_fields_for_a_train_frame_eval(
    tmp_path: Path,
) -> None:
    """Acceptance: training-path behaviour is unchanged -- mode 'own' reproduces the strict number."""
    trainer = _build_trainer(tmp_path)
    train_name = trainer.train_names[0]

    metrics = trainer.evaluate(names=[train_name], exposure_mode="own")
    row = metrics["per_image"][train_name]

    assert row["psnr_eval"] == row["psnr"]
    assert row["ssim_eval"] == row["ssim"]
    assert row["lpips_eval"] == row["lpips"]
    assert metrics["psnr_mean_eval"] == metrics["psnr_mean"]


def test_evaluate_mode_own_is_bit_identical_for_a_held_out_frame_too(tmp_path: Path) -> None:
    """mode='own' must reproduce the plain fields even for a held-out row (no interpolation)."""
    trainer = _build_trainer(tmp_path)
    name = trainer.heldout_names[0]

    metrics = trainer.evaluate(names=[name], exposure_mode="own")
    row = metrics["per_image"][name]

    assert row["exposure_mode"] == "own"
    assert row["psnr_eval"] == row["psnr"]
    assert row["ssim_eval"] == row["ssim"]


def test_evaluate_neighbours_recovers_a_broken_held_out_exposure_without_reading_its_photo(
    tmp_path: Path,
) -> None:
    """The actual bug fix: mode='neighbours' must undo a broken held-out exposure.

    Mirrors tests/test_train_eval_calibrate.py's calibration test, but the fix here comes
    entirely from the frame's TRAINING neighbours' own exposure rows -- never from fitting
    against this frame's own photo.
    """
    trainer = _build_trainer(tmp_path)
    name = trainer.heldout_names[0]
    index = trainer._name_to_index[name]

    with torch.no_grad():
        trainer.camera.exposures_values[index] = -5.87  # the kk-coherent no-EXIF artefact

    metrics = trainer.evaluate(names=[name], exposure_mode="neighbours")
    row = metrics["per_image"][name]

    # The plain "own" field still shows the broken render (unaffected by exposure_mode).
    assert row["exposure_gain"] > 50.0
    # The headline "_eval" number, computed from the TRAINING neighbours instead, is far better.
    assert row["psnr_eval"] > row["psnr"] + 5.0, row


def test_evaluate_calibrate_mode_matches_the_legacy_calibrated_side_column(tmp_path: Path) -> None:
    """exposure_mode='calibrate' promotes calibrate_frame's own fit to be the headline number."""
    trainer = _build_trainer(tmp_path)
    name = trainer.heldout_names[0]
    index = trainer._name_to_index[name]
    with torch.no_grad():
        trainer.camera.exposures_values[index] = -3.0

    metrics = trainer.evaluate(names=[name], calibrate=True, exposure_mode="calibrate")
    row = metrics["per_image"][name]

    assert row["exposure_mode"] == "calibrate"
    assert row["psnr_eval"] == row["psnr_calibrated"]


def test_evaluate_neighbours_never_mutates_camera_parameters(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    name = trainer.heldout_names[0]
    before_exposures = trainer.camera.exposures_values.detach().clone()
    before_wb = trainer.camera.white_balance_values.detach().clone()

    trainer.evaluate(names=[name], exposure_mode="neighbours")

    assert torch.equal(trainer.camera.exposures_values.detach(), before_exposures)
    assert torch.equal(trainer.camera.white_balance_values.detach(), before_wb)


def test_evaluate_default_exposure_mode_is_neighbours_and_plain_fields_are_unaffected(
    tmp_path: Path,
) -> None:
    """The default changed (config `eval_exposure_mode`), but the plain fields never move."""
    trainer = _build_trainer(tmp_path)
    name = trainer.heldout_names[0]
    index = trainer._name_to_index[name]
    with torch.no_grad():
        trainer.camera.exposures_values[index] = -4.0

    default_call = trainer.evaluate(names=[name])
    own_call = trainer.evaluate(names=[name], exposure_mode="own")

    assert default_call["exposure_mode"] == "neighbours"
    # The plain "own" psnr is exactly the same whether or not exposure_mode is requested --
    # this is the invariant tests/test_train_regression.py and test_train_eval_calibrate.py
    # rely on.
    assert default_call["per_image"][name]["psnr"] == own_call["per_image"][name]["psnr"]
    assert default_call["psnr_mean"] == own_call["psnr_mean"]


def test_evaluate_invalid_exposure_mode_raises(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    try:
        trainer.evaluate(exposure_mode="bogus")
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("expected a ValueError for an invalid exposure_mode")
