"""Tests for trippy.train.trainer.Trainer: train_step, evaluate, checkpoint, export.

Module: tests.test_train_trainer
Invariants under test: `train_step` (with a pinned crop) decreases loss
    over 5 steps on the synthetic scene; `evaluate` writes `metrics.json`
    + `sheet.jpg` (JPEG, not PNG -- see trainer.py's `_save_eval_sheet_jpeg`)
    under `<run_dir>/eval_ep<N>/` (or `eval_dirname` when given), capped at
    `cfg.eval_max_images` rows; records a `per_image` dict plus a
    `shade`/`other` held-out split (shade = `cfg.forced_heldout`
    intersected with the evaluated names, else `SHADE_FRAMES_KK`), and
    appends that same split (minus `names`) to `metrics.jsonl`;
    `save_checkpoint` + `resume` on a freshly constructed Trainer reproduces
    identical parameters; `export_ply` round-trips through
    `GaussianPlySource`; the checkpoint retention policy
    (`trippy.train.retention`, wired into `save_checkpoint`) keeps
    `checkpoint_latest.pt`/`checkpoint_best.pt`, `checkpoint_keep_every`
    multiples, and the last `checkpoint_keep_last` epochs, deleting the
    rest -- and a run can still `resume` from `checkpoint_latest.pt` after
    pruning.
All fixtures are the synthetic scene from `tests/test_train_helpers.py`
(never a real Splats scene).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image as PILImage
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
    assert (eval_dir / "sheet.jpg").exists()  # JPEG, not PNG -- see trainer._save_eval_sheet_jpeg


def test_evaluate_prioritises_forced_heldout_in_sheet(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, forced_heldout=["IMG_1.jpg"], heldout_k=8)
    assert "IMG_1.jpg" in trainer.heldout_names
    metrics = trainer.evaluate()
    assert "IMG_1.jpg" in metrics["names"]


def test_evaluate_eval_max_images_caps_sheet_rows(tmp_path: Path) -> None:
    """`cfg.eval_max_images` caps sheet ROWS, not the PSNR/SSIM population.

    Three forced-held-out images guarantee more eval images exist than a
    small `eval_max_images` cap allows into the sheet; the sheet is 4
    columns (photo | render | raw L0 | coverage) so its height scales
    1:1 with the number of image rows actually drawn -- a smaller cap must
    produce a strictly shorter sheet.
    """
    forced = ["IMG_1.jpg", "IMG_2.jpg", "IMG_3.jpg"]

    def _build(run_tag: str, **overrides) -> Trainer:
        # 6 images (not the module default of 4): 3 forced held out leaves 3 remaining for
        # the modulo split to divide between train/heldout, so train never goes empty.
        scene_root, point_set = build_synthetic_scene(tmp_path / run_tag, n_images=6)
        ply_path = build_synthetic_ply(tmp_path / run_tag, point_set)
        cfg = tiny_train_config(
            scene_root, ply_path, tmp_path / run_tag / "run", tmp_path / run_tag / "cache",
            forced_heldout=forced, **overrides,
        )  # fmt: skip
        return Trainer(cfg)

    small = _build("small", eval_max_images=1)
    big = _build("big", eval_max_images=3)

    small_metrics = small.evaluate()
    big_metrics = big.evaluate()
    # The cap only affects the sheet, not which images are scored.
    assert small_metrics["n_images"] == big_metrics["n_images"] == len(small.heldout_names)

    small_sheet = np.array(PILImage.open(small.run_dir / "eval_ep0000" / "sheet.jpg"))
    big_sheet = np.array(PILImage.open(big.run_dir / "eval_ep0000" / "sheet.jpg"))
    assert small_sheet.shape[0] < big_sheet.shape[0]


def test_evaluate_records_per_image_and_shade_split(tmp_path: Path) -> None:
    # forced_heldout is the shade set here (docs/EXPERIMENTS.md "Leaderboard": shade frames are
    # cfg.forced_heldout when non-empty, else SHADE_FRAMES_KK) -- IMG_1.jpg is forced into
    # held-out and is therefore the one and only "shade" frame; everything else held out lands
    # in "other".
    trainer = _build_trainer(tmp_path, forced_heldout=["IMG_1.jpg"], heldout_k=8)
    metrics = trainer.evaluate()

    assert set(metrics["per_image"].keys()) == set(metrics["names"])
    for entry in metrics["per_image"].values():
        assert entry["psnr"] == pytest.approx(entry["psnr"])  # a real float, not NaN/None
        assert entry["ssim"] is not None
        assert entry["lpips"] is None  # eval_lpips=False in tiny_train_config

    assert metrics["shade"]["n"] == 1
    assert metrics["shade"]["psnr"] == pytest.approx(metrics["per_image"]["IMG_1.jpg"]["psnr"])
    assert metrics["shade"]["lpips"] is None
    assert metrics["other"]["n"] == metrics["n_images"] - 1

    # Also written to eval_ep0000/metrics.json, and appended (minus "names") to metrics.jsonl.
    eval_dir = trainer.run_dir / "eval_ep0000"
    written = json.loads((eval_dir / "metrics.json").read_text())
    assert written["shade"]["n"] == 1
    assert set(written["per_image"].keys()) == set(metrics["names"])

    rows = [json.loads(line) for line in trainer.metrics_path.read_text().splitlines()]
    eval_rows = [r for r in rows if r.get("eval")]
    assert eval_rows[-1]["shade"]["n"] == 1
    assert "per_image" in eval_rows[-1]
    assert "names" not in eval_rows[-1]


def test_evaluate_shade_split_empty_when_no_forced_heldout_matches(tmp_path: Path) -> None:
    # No forced_heldout, and none of the synthetic scene's IMG_0..3.jpg names are in
    # SHADE_FRAMES_KK -- "shade" degrades to an empty (not fabricated) group.
    trainer = _build_trainer(tmp_path)
    metrics = trainer.evaluate()
    assert metrics["shade"] == {"n": 0, "psnr": None, "ssim": None, "lpips": None}
    assert metrics["other"]["n"] == metrics["n_images"]


def test_evaluate_eval_dirname_override(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    trainer.evaluate(eval_dirname="eval_manual_20260906-000000")
    assert (trainer.run_dir / "eval_manual_20260906-000000" / "metrics.json").exists()
    assert not (trainer.run_dir / "eval_ep0000").exists()


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


def test_save_checkpoint_prunes_by_keep_every_and_keep_last(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, checkpoint_keep_every=3, checkpoint_keep_last=1)
    for epoch in range(6):
        trainer.save_checkpoint(epoch=epoch)

    remaining = {int(p.stem.split("checkpoint_ep")[1]) for p in trainer.checkpoint_dir.glob("checkpoint_ep*.pt")}
    # kept: multiples of 3 (0, 3) + the single most recent epoch (5); everything else pruned.
    assert remaining == {0, 3, 5}
    assert (trainer.checkpoint_dir / "checkpoint_latest.pt").exists()


def test_save_checkpoint_writes_checkpoint_best_when_epoch_is_the_best_so_far(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    trainer._best_epoch = 5
    trainer._best_psnr = 12.34
    trainer.save_checkpoint(epoch=5)

    best_path = trainer.checkpoint_dir / "checkpoint_best.pt"
    assert best_path.exists()
    best_json = json.loads((trainer.checkpoint_dir / "best.json").read_text())
    assert best_json == {"epoch": 5, "psnr": 12.34}


def test_save_checkpoint_skips_checkpoint_best_when_epoch_is_not_the_best(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    trainer._best_epoch = 5
    trainer._best_psnr = 12.34
    trainer.save_checkpoint(epoch=6)  # evaluate() never scored epoch 6 as the best

    assert not (trainer.checkpoint_dir / "checkpoint_best.pt").exists()
    assert not (trainer.checkpoint_dir / "best.json").exists()


def test_evaluate_updates_best_epoch_and_save_checkpoint_promotes_it(tmp_path: Path) -> None:
    """The natural `fit()`-style flow: evaluate() finds a best epoch, then save_checkpoint(same epoch) promotes it."""
    trainer = _build_trainer(tmp_path)
    metrics = trainer.evaluate(epoch=0)
    assert trainer._best_epoch == 0
    trainer.save_checkpoint(epoch=0)

    best_json = json.loads((trainer.checkpoint_dir / "best.json").read_text())
    assert best_json["epoch"] == 0
    assert best_json["psnr"] == metrics["psnr_mean"]
    assert (trainer.checkpoint_dir / "checkpoint_best.pt").exists()


def test_resume_from_latest_works_after_retention_pruning(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, checkpoint_keep_every=3, checkpoint_keep_last=1)
    for epoch in range(6):
        trainer.train_step()
        trainer.save_checkpoint(epoch=epoch)

    remaining = {int(p.stem.split("checkpoint_ep")[1]) for p in trainer.checkpoint_dir.glob("checkpoint_ep*.pt")}
    assert remaining == {0, 3, 5}  # epoch 4's own file was pruned -- resume must still work

    scene_root, point_set = build_synthetic_scene(tmp_path / "scene2")
    ply_path = build_synthetic_ply(tmp_path / "scene2", point_set)
    cfg2 = tiny_train_config(scene_root, ply_path, tmp_path / "run2", tmp_path / "cache2")
    resumed = Trainer(cfg2)
    resumed.resume(trainer.checkpoint_dir / "checkpoint_latest.pt")

    assert resumed.epoch == 5
    torch.testing.assert_close(resumed.point_params.xyz, trainer.point_params.xyz)
    torch.testing.assert_close(resumed.net.state_dict()["final.0.weight"], trainer.net.state_dict()["final.0.weight"])


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
