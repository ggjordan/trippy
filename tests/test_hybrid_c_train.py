"""Tests for trippy.hybrid.train_c.HybridCTrainer: train_step, evaluate, checkpointing.

Module: tests.test_hybrid_c_train
Invariants under test: `train_step` (with a pinned crop) decreases loss over several steps on
    the synthetic render/photo pair; `evaluate` returns the documented
    baseline/refined x all/shade/nonshade metrics schema and writes `metrics.json` + `sheet.png`
    + a standalone `shade_frames/<stem>_refined.png` per forced-held-out name; checkpoint
    save/resume reproduces trained state.
All fixtures are the synthetic scene/render pair from `tests/test_hybrid_c_helpers.py` (never
a real Splats scene).
"""

from __future__ import annotations

from pathlib import Path

from test_hybrid_c_helpers import build_synthetic_renders, build_synthetic_scene, tiny_hybrid_c_config

from trippy.hybrid.train_c import HybridCTrainer, build_trainer_from_checkpoint


def _build_trainer(tmp_path: Path, **overrides) -> HybridCTrainer:
    scene_root, names = build_synthetic_scene(tmp_path)
    renders_dir = build_synthetic_renders(scene_root, tmp_path / "renders", names)
    cfg = tiny_hybrid_c_config(scene_root, renders_dir, tmp_path / "run", tmp_path / "cache", **overrides)
    return HybridCTrainer(cfg)


def test_train_step_decreases_loss_over_steps(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path, lr_network=5e-3, lr_exposure=0.0, lr_response=0.0, loss_lpips=0.0)
    name = trainer.train_names[0]
    losses = [trainer.train_step(name=name, y0=4, x0=4)["loss"] for _ in range(10)]
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_train_step_returns_expected_metric_keys(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    record = trainer.train_step()
    for key in ("step", "epoch", "image", "loss", "image_loss", "camera_reg"):
        assert key in record
    assert trainer.metrics_path.exists()


def test_evaluate_metrics_schema_and_shade_bucket(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    metrics = trainer.evaluate()

    assert metrics["n_images"] == len(trainer.heldout_names)
    assert metrics["names"] == trainer.heldout_names
    for top_key in ("baseline", "refined"):
        assert top_key in metrics
        for bucket in ("all", "shade", "nonshade"):
            assert bucket in metrics[top_key]
            entry = metrics[top_key][bucket]
            assert set(entry) == {"n", "psnr_mean", "ssim_mean", "lpips_mean"}
            assert entry["lpips_mean"] is None  # eval_lpips=False in tiny_hybrid_c_config

    # forced_heldout=["IMG_1.jpg"] -> exactly one shade-bucket frame.
    assert metrics["baseline"]["shade"]["n"] == 1
    assert metrics["refined"]["shade"]["n"] == 1
    assert metrics["baseline"]["all"]["n"] == metrics["n_images"]

    eval_dir = trainer.run_dir / "eval_ep0000"
    assert (eval_dir / "metrics.json").exists()
    assert (eval_dir / "sheet.png").exists()
    assert (eval_dir / "shade_frames" / "IMG_1_refined.png").exists()


def test_checkpoint_save_and_resume_reproduces_state(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    trainer.train_step()
    trainer.train_step()
    ckpt_path = trainer.save_checkpoint(epoch=3)
    assert ckpt_path.exists()

    resumed = build_trainer_from_checkpoint(ckpt_path, device="cpu")
    assert resumed.epoch == 3
    for (name_a, p_a), (name_b, p_b) in zip(
        trainer.net.state_dict().items(), resumed.net.state_dict().items(), strict=True
    ):
        assert name_a == name_b
        assert (p_a == p_b).all()


def test_fit_runs_short_budget_and_writes_export_free_run(tmp_path: Path) -> None:
    """`fit()` under a tight time budget still produces at least one eval + checkpoint."""
    trainer = _build_trainer(tmp_path, epochs=5, eval_every=1, checkpoint_every=1, train_factor=1.0)
    metrics = trainer.fit(max_minutes=1.0)
    assert metrics  # at least one evaluate() ran
    assert (trainer.checkpoint_dir / "checkpoint_latest.pt").exists()
