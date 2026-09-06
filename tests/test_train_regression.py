"""Regression tests for the bugs that made the EXP-0003 smoke run render black.

Module: tests.test_train_regression
Invariants under test (each one would have failed, on CPU, in seconds,
    before the 2026-09-06 fix -- see research/trips-metal.md):
    1. `Trainer._initial_exposure` centres the per-image EXIF EV on the
       scene mean, so the tone mapper's `x * 2 ** -EV` gain starts at ~1.
       Absolute EVs (kk-coherent mean 6.14, the synthetic fixture 8.2)
       otherwise divide every prediction by 70--300x and no learning rate
       can recover it inside a smoke run.
    2. A masked MSE/PSNR averages over every (channel, pixel) element the
       mask keeps. Dividing a 3-channel squared-error sum by a 1-channel
       mask sum inflates the MSE by exactly 3 and understates PSNR by
       10*log10(3) = 4.771 dB.
    3. Every intermediate tensor (pyramid levels, U-Net input/output,
       tone-mapped prediction, target) stays in a sane range at init.
    4. After a short training run the held-out PSNR beats the PSNR of a
       constant prediction -- the sanity floor a black or inverted render
       cannot clear.
    5. A non-finite gradient is zeroed before the optimizer step, so one
       degenerate fragment cannot permanently NaN a parameter.
    6. An image with no usable EXIF initialises at the scene mean (gain
       1.0), not at absolute EV 0 (which on kk-coherent meant a 58.5x
       gain that a held-out frame never trains away).
All fixtures are the synthetic scene from `tests/test_train_helpers.py`
(never a real Splats scene); its photos now carry real EXIF so bug 1 is
reachable from CPU tests at all.
Related docs: docs/TRIPS_REFERENCE.md Sec. 6 (exposure init),
    docs/LIMITATIONS.md.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from test_train_helpers import (
    EXIF_EXPOSURE_TIMES,
    EXIF_ISO,
    build_synthetic_ply,
    build_synthetic_scene,
    tiny_train_config,
)

from trippy.constants import SCENE_CACHE_META_FILENAME
from trippy.net.losses import mse_loss
from trippy.scene.dataset import crop as dataset_crop
from trippy.train.trainer import Trainer, _center_crop_like

# Steps of training before the held-out PSNR is required to clear the constant-prediction
# floor. Measured on this fixture: 17.9 dB at init, 20.0 dB at 40 steps, floor 18.3 dB.
FLOOR_TEST_STEPS = 40


def _build_trainer(tmp_path: Path, **overrides) -> Trainer:
    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(scene_root, ply_path, tmp_path / "run", tmp_path / "cache", **overrides)
    return Trainer(cfg)


def _heldout_prediction(trainer: Trainer, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(pred, target) for one full-frame held-out image, both (1, 3, H', W') in [0, 1]-ish."""
    frame_index = trainer._name_to_index[name]
    item = trainer.dataset[frame_index]
    height, width = int(item["rgb"].shape[0]), int(item["rgb"].shape[1])
    target = item["rgb"].to(torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    trainer.net.eval()
    trainer.camera.eval()
    with torch.no_grad():
        R, t = trainer._pose_for(item, frame_index)
        net_out, _layers, _aux = trainer._render(item["K"], R, t, (height, width))
        pred = trainer._tone_map(net_out, frame_index)
    return pred, _center_crop_like(target, pred.shape[-2], pred.shape[-1])


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(-10.0 * torch.log10(mse_loss(pred, target) + 1e-10))


# --- 1. exposure initialisation -----------------------------------------


def test_initial_exposure_is_centred_on_the_scene_mean(tmp_path: Path) -> None:
    """TRIPS inits per-frame exposure as `EV - mean(EV)` (NeuralScene.cpp:38)."""
    trainer = _build_trainer(tmp_path)
    exposures = trainer.camera.exposures_values.detach().reshape(-1)

    raw_ev = [math.log2(1.0 / t) + math.log2(EXIF_ISO / 100.0) for t in EXIF_EXPOSURE_TIMES]
    assert min(raw_ev) > 4.0, "fixture must have a large absolute EV or this test proves nothing"

    assert abs(float(exposures.mean())) < 1e-4, (
        f"exposure init is not zero-mean ({float(exposures.mean()):+.4f}); the tone mapper's "
        f"gain 2**-EV would be {2 ** -float(exposures.mean()):.5f} instead of ~1"
    )
    gain = torch.exp2(-exposures)
    assert float(gain.min()) > 0.5 and float(gain.max()) < 2.0, f"exposure gains out of range: {gain.tolist()}"

    # Relative EVs must survive: the spread is the only thing the init is allowed to encode.
    spread_expected = max(raw_ev) - min(raw_ev)
    spread_actual = float(exposures.max() - exposures.min())
    assert abs(spread_actual - spread_expected) < 1e-3


def test_initial_exposure_of_an_image_without_exif_is_the_scene_mean(tmp_path: Path) -> None:
    """A missing-EXIF image must start at gain 1.0, not `2 ** mean(EV)` (58.5x on kk-coherent).

    The old fallback was an absolute `EV = 0`, which after subtracting the scene mean left the
    frame at `-mean(EV)` -- and a held-out frame's exposure is never trained, so it stayed there
    for the whole run (docs/LIMITATIONS.md "train/", experiments/EXP-0003-kk-trips-train/README.md
    "The exposure artefact").
    """
    trainer = _build_trainer(tmp_path)
    meta_path = trainer.dataset.cache_dir / SCENE_CACHE_META_FILENAME
    meta = json.loads(meta_path.read_text())
    stripped = trainer.dataset.names[0]
    meta["images"][stripped].pop("exposure_time", None)
    meta["images"][stripped].pop("iso", None)
    meta_path.write_text(json.dumps(meta))

    exposures = trainer._initial_exposure()
    index = trainer.dataset.names.index(stripped)

    known_ev = [math.log2(1.0 / t) + math.log2(EXIF_ISO / 100.0) for t in EXIF_EXPOSURE_TIMES]
    assert min(known_ev) > 4.0, "fixture must have a large absolute EV or this test proves nothing"
    # Exactly the scene mean -> relative 0 -> gain 1.0.
    assert abs(float(exposures[index])) < 1e-5, (
        f"missing-EXIF image initialised at {float(exposures[index]):+.3f} EV, i.e. a gain of "
        f"{2 ** -float(exposures[index]):.1f}x"
    )
    # The remaining images keep their relative spread, now centred on the known-EV mean only.
    others = [float(v) for i, v in enumerate(exposures) if i != index]
    assert abs(sum(others) / len(others)) < 1e-4


# --- 2. masked MSE / PSNR ------------------------------------------------


def test_masked_mse_averages_over_channels_not_pixels() -> None:
    """A (1, 1, H, W) mask on a (1, 3, H, W) error must not divide by the pixel count."""
    pred = torch.zeros(1, 3, 4, 5)
    target = torch.full((1, 3, 4, 5), 0.5)
    mask = torch.ones(1, 1, 4, 5)

    assert abs(float(mse_loss(pred, target, mask)) - 0.25) < 1e-6
    assert abs(float(mse_loss(pred, target, None)) - 0.25) < 1e-6

    # Half the pixels masked out -> same value, not half and not 3x.
    mask[..., :2] = 0.0
    assert abs(float(mse_loss(pred, target, mask)) - 0.25) < 1e-6


def test_evaluate_psnr_matches_an_independent_masked_psnr(tmp_path: Path) -> None:
    """`Trainer.evaluate`'s psnr_mean must equal a PSNR recomputed from its own render."""
    trainer = _build_trainer(tmp_path)
    trainer.train_step()

    metrics = trainer.evaluate()
    independent = [_psnr(*_heldout_prediction(trainer, name)) for name in metrics["names"]]
    expected = sum(independent) / len(independent)

    assert abs(metrics["psnr_mean"] - expected) < 0.01, (
        f"evaluate() reported {metrics['psnr_mean']:.3f} dB but an independent masked PSNR of the "
        f"same render is {expected:.3f} dB (a 4.771 dB gap means the MSE was divided by the "
        f"pixel count instead of the element count)"
    )


# --- 3. intermediate ranges ---------------------------------------------


def test_render_stages_stay_in_sane_ranges(tmp_path: Path) -> None:
    """Pyramid -> U-Net -> tone mapper: nothing goes non-finite or wildly out of range."""
    trainer = _build_trainer(tmp_path)
    name = trainer.heldout_names[0]
    frame_index = trainer._name_to_index[name]
    item = trainer.dataset[frame_index]
    cropped = dataset_crop(item, size=trainer.cfg.crop, zoom=1.0)

    target = cropped["rgb"].to(torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    assert float(target.min()) >= 0.0 and float(target.max()) <= 1.0

    trainer.net.eval()
    trainer.camera.eval()
    with torch.no_grad():
        R, t = trainer._pose_for(item, frame_index)
        net_out, layers, aux = trainer._render(cropped["K"], R, t, (trainer.cfg.crop, trainer.cfg.crop))
        pred = trainer._tone_map(net_out, frame_index)

    # The composite is a convex combination of point features and the background, both of
    # which start inside [0, 1] (PointParams seeds channels 0:3 from rgb0).
    for level, layer in enumerate(layers):
        assert torch.isfinite(layer).all(), f"pyramid level {level} has non-finite values"
        assert -0.2 <= float(layer.min()) and float(layer.max()) <= 1.2, (
            f"pyramid level {level} out of range: [{float(layer.min()):.3f}, {float(layer.max()):.3f}]"
        )
        t_final = aux["t_final"][level]
        assert -1e-5 <= float(t_final.min()) and float(t_final.max()) <= 1.0 + 1e-5

    assert torch.isfinite(net_out).all(), "U-Net output has non-finite values"
    assert float(net_out.abs().max()) < 50.0, f"U-Net output exploded: max |x| = {float(net_out.abs().max())}"

    assert torch.isfinite(pred).all(), "tone-mapped prediction has non-finite values"
    assert -0.01 <= float(pred.min()) and float(pred.max()) <= 1.01, (
        f"tone-mapped prediction out of display range: [{float(pred.min()):.4f}, {float(pred.max()):.4f}]"
    )
    # A black render is the exact failure mode of the un-centred exposure init.
    assert float(pred.mean()) > 0.05, f"prediction is essentially black (mean {float(pred.mean()):.5f})"


def test_background_parameter_comes_from_the_config(tmp_path: Path) -> None:
    """`cfg.background` must actually initialise the background feature."""
    trainer = _build_trainer(tmp_path, background=0.4)
    assert torch.allclose(trainer.background.detach(), torch.full_like(trainer.background.detach(), 0.4))


# --- 4. the sanity floor -------------------------------------------------


def test_heldout_psnr_beats_the_constant_prediction_floor(tmp_path: Path) -> None:
    """After a short run the held-out PSNR must beat the best constant-colour prediction.

    The floor is deliberately weak: any render that is black, inverted or
    scaled into the wrong range scores *below* a flat grey image, so this
    catches the whole "nonsense render" failure class without asserting a
    particular quality level.
    """
    trainer = _build_trainer(tmp_path)
    for _ in range(FLOOR_TEST_STEPS):
        trainer.train_step()

    metrics = trainer.evaluate()
    for name in metrics["names"]:
        pred, target = _heldout_prediction(trainer, name)
        grey_half = _psnr(torch.full_like(target, 0.5), target)
        best_constant = _psnr(torch.full_like(target, float(target.mean())), target)
        actual = _psnr(pred, target)
        assert actual > grey_half, f"{name}: PSNR {actual:.3f} dB is below flat-grey {grey_half:.3f} dB"
        assert actual > best_constant, (
            f"{name}: PSNR {actual:.3f} dB does not beat the best constant prediction "
            f"{best_constant:.3f} dB after {FLOOR_TEST_STEPS} steps"
        )

    assert metrics["psnr_mean"] > 0.0


# --- 5. non-finite gradient containment ---------------------------------


def test_nonfinite_gradients_are_zeroed_before_the_optimizer_step(tmp_path: Path) -> None:
    """One NaN gradient must not turn a parameter -- and the whole run -- into NaN.

    The rasteriser backward can emit a NaN gradient for a single
    degenerate point (docs/LIMITATIONS.md); Adam would make that
    parameter permanently NaN and, via `_extent_penalty`'s reduction over
    every point, make the reported loss NaN for the rest of the run.
    """
    trainer = _build_trainer(tmp_path)
    trainer.train_step()  # populate optimizer state

    before = trainer.point_params.xyz.detach().clone()
    trainer.optimizer.zero_grad(set_to_none=True)
    grad = torch.zeros_like(trainer.point_params.xyz)
    grad[0, 0] = float("nan")
    grad[1, 1] = float("inf")
    grad[2, 2] = 0.5
    trainer.point_params.xyz.grad = grad

    count = trainer._sanitise_gradients()
    assert int(count.item()) == 2
    assert torch.isfinite(trainer.point_params.xyz.grad).all()
    assert float(trainer.point_params.xyz.grad[2, 2]) == 0.5

    trainer.optimizer.step()
    after = trainer.point_params.xyz.detach()
    assert torch.isfinite(after).all(), "a NaN gradient reached the parameters"
    assert not torch.equal(after[2], before[2]), "the finite gradient entry was dropped too"


def test_train_step_reports_the_nonfinite_gradient_count(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    record = trainer.train_step()
    assert record["nonfinite_grads"] == 0
