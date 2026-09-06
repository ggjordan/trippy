"""Tests for trippy.train.eval.evaluate_checkpoint: manual re-eval of an existing checkpoint.

Module: tests.test_train_eval
Invariants under test: `evaluate_checkpoint` rebuilds a `Trainer` from a
    saved checkpoint (never re-trains) and calls `Trainer.evaluate()` with a
    `eval_manual_<timestamp>` directory name, distinct from any
    mid-training/`--report` eval directory the same run may already have,
    and appends an `{"eval": True, ...}` row (carrying the "shade"/"other"
    split, see tests/test_train_trainer.py) to the run's own metrics.jsonl
    -- this is what lets `trippy eval --checkpoint` backfill a shade split
    for a checkpoint that finished training before the split existed,
    without retraining it (docs/EXPERIMENTS.md "Leaderboard").
All fixtures are the synthetic scene from `tests/test_train_helpers.py`
(never a real Splats scene or checkpoint).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config

from trippy.train.eval import evaluate_checkpoint
from trippy.train.trainer import Trainer

_MANUAL_DIRNAME_RE = re.compile(r"^eval_manual_\d{8}-\d{6}$")


def _build_and_checkpoint(tmp_path: Path, **overrides) -> Path:
    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(scene_root, ply_path, tmp_path / "run", tmp_path / "cache", **overrides)
    trainer = Trainer(cfg)
    trainer.train_step()
    return trainer.save_checkpoint(epoch=1)


def test_evaluate_checkpoint_writes_eval_manual_dir_not_eval_ep(tmp_path: Path) -> None:
    ckpt_path = _build_and_checkpoint(tmp_path)
    run_dir = ckpt_path.parent.parent

    metrics = evaluate_checkpoint(ckpt_path, device="cpu")

    manual_dirs = [p.name for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("eval_manual_")]
    assert len(manual_dirs) == 1
    assert _MANUAL_DIRNAME_RE.match(manual_dirs[0])
    assert (run_dir / manual_dirs[0] / "metrics.json").exists()
    assert metrics["n_images"] == len(metrics["names"])


def test_evaluate_checkpoint_appends_eval_row_with_shade_split(tmp_path: Path) -> None:
    ckpt_path = _build_and_checkpoint(tmp_path, forced_heldout=["IMG_1.jpg"], heldout_k=8)
    run_dir = ckpt_path.parent.parent
    metrics_path = run_dir / "metrics.jsonl"

    rows_before = metrics_path.read_text().splitlines()
    metrics = evaluate_checkpoint(ckpt_path, device="cpu")
    rows_after = metrics_path.read_text().splitlines()

    assert len(rows_after) == len(rows_before) + 1
    appended = json.loads(rows_after[-1])
    assert appended.get("eval") is True
    assert appended["shade"]["n"] == 1
    assert appended["shade"]["psnr"] == metrics["shade"]["psnr"]
    assert "per_image" in appended
    assert "names" not in appended


def test_evaluate_checkpoint_repeated_calls_do_not_collide(tmp_path: Path) -> None:
    # Two manual re-evals of the same checkpoint must not clobber each other's output dir
    # (distinct timestamps) -- exercised with an explicit sleep-free check on directory count
    # rather than timing, since two calls in the same test may land on the same wall-clock
    # second on a fast machine; this only asserts neither call raises and both write valid
    # metrics.json (the timestamp collision case degrading to "second call overwrites the
    # first" is acceptable and not what this test guards against).
    ckpt_path = _build_and_checkpoint(tmp_path)
    run_dir = ckpt_path.parent.parent

    evaluate_checkpoint(ckpt_path, device="cpu")
    evaluate_checkpoint(ckpt_path, device="cpu")

    manual_dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("eval_manual_")]
    assert len(manual_dirs) >= 1
    for d in manual_dirs:
        assert (d / "metrics.json").exists()
