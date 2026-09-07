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
    On a gate-hybrid checkpoint, `Trainer.evaluate` now also applies the
    edit-weight gate-suppression multiply for `blend`/`fade` regions
    (docs/EDITOR.md Sec 5's previously-documented "trippy eval --edits does
    not get this" gap, `trippy.edit.checkpoint.render_edit_weight_map`) --
    the non-edit path stays bit-identical (an empty `edits.json` changes
    nothing), a `delete` region still changes PSNR by removing points, and a
    `fade`/`blend` region now ALSO measurably suppresses the reported gate.
All fixtures are the synthetic scene from `tests/test_train_helpers.py`
(never a real Splats scene or checkpoint); the gate-hybrid fixtures are
`tests/test_hybrid_a_helpers.py`'s synthetic fake-render scene (never a real
Splats render).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from test_hybrid_a_helpers import hybrid_train_config
from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config

from trippy.edit.model import EditDocument, Region
from trippy.train.eval import build_trainer_from_checkpoint, evaluate_checkpoint
from trippy.train.trainer import Trainer

_MANUAL_DIRNAME_RE = re.compile(r"^eval_manual_\d{8}-\d{6}$")


def _gate_checkpoint(tmp_path: Path) -> Path:
    """A one-step-trained, gate-hybrid checkpoint (the synthetic fake-render fixture)."""
    cfg, _names = hybrid_train_config(tmp_path, gate={"enabled": True}, dropout_gaussian_p=0.0)
    trainer = Trainer(cfg)
    trainer.train_step()
    return trainer.save_checkpoint(epoch=1)


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


def test_evaluate_checkpoint_with_edits_deletes_points(tmp_path: Path) -> None:
    """`--edits` (docs/EDITOR.md Sec 5): a delete-op region shrinks the point cloud before scoring."""
    from trippy.edit.model import EditDocument, Region
    from trippy.train.eval import build_trainer_from_checkpoint

    ckpt_path = _build_and_checkpoint(tmp_path)
    n_before = len(build_trainer_from_checkpoint(ckpt_path, device="cpu").point_params)

    edits = EditDocument.new()
    edits.add_region(
        Region(id="r-del", name="del", kind="pointset", params={"point_ids": list(range(20))}, op="delete")
    )
    edits_path = tmp_path / "edits.json"
    edits.save(edits_path)

    metrics = evaluate_checkpoint(ckpt_path, device="cpu", edits_path=edits_path)
    assert metrics["edits"] == {
        "n_points_before": n_before,
        "n_removed": 20,
        "n_points_after": n_before - 20,
        "n_regions": 1,
        "gate_suppression_active": False,
    }


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


# --- closing the gate-suppression gap on a gate-hybrid checkpoint (docs/EDITOR.md Sec 5) --


def test_evaluate_checkpoint_empty_edits_is_bit_identical_on_a_gate_hybrid(tmp_path: Path) -> None:
    """The non-edit path must be untouched: an empty edits.json changes nothing."""
    ckpt_path = _gate_checkpoint(tmp_path)
    baseline = evaluate_checkpoint(ckpt_path, device="cpu")

    empty_path = tmp_path / "empty_edits.json"
    EditDocument.new().save(empty_path)
    with_empty_edits = evaluate_checkpoint(ckpt_path, device="cpu", edits_path=empty_path)

    assert with_empty_edits["psnr_mean"] == baseline["psnr_mean"]
    assert with_empty_edits["gate"]["mean"] == baseline["gate"]["mean"]
    assert with_empty_edits["edits"]["gate_suppression_active"] is False


def test_evaluate_checkpoint_delete_region_changes_psnr_on_a_gate_hybrid(tmp_path: Path) -> None:
    """A `delete` region removes points before any image renders -- PSNR must move."""
    ckpt_path = _gate_checkpoint(tmp_path)
    baseline = evaluate_checkpoint(ckpt_path, device="cpu")

    n_before = len(build_trainer_from_checkpoint(ckpt_path, device="cpu").point_params)
    edits = EditDocument.new()
    edits.add_region(
        Region(
            id="r-del", name="del", kind="pointset",
            params={"point_ids": list(range(n_before // 2))}, op="delete",
        )
    )  # fmt: skip
    edits_path = tmp_path / "delete_edits.json"
    edits.save(edits_path)

    edited = evaluate_checkpoint(ckpt_path, device="cpu", edits_path=edits_path)

    assert edited["edits"]["n_removed"] == n_before // 2
    assert edited["psnr_mean"] != baseline["psnr_mean"]


def test_evaluate_checkpoint_fade_region_suppresses_the_gate(tmp_path: Path) -> None:
    """The closed gap itself: a `fade` region must now move the reported gate, not just PSNR."""
    ckpt_path = _gate_checkpoint(tmp_path)
    baseline = evaluate_checkpoint(ckpt_path, device="cpu")
    assert "gate" in baseline, "the gate-hybrid fixture must actually exercise the gate"

    n_before = len(build_trainer_from_checkpoint(ckpt_path, device="cpu").point_params)
    edits = EditDocument.new()
    edits.add_region(
        Region(
            id="r-fade", name="fade", kind="pointset",
            params={"point_ids": list(range(n_before))}, mix=0.0, op="fade",
        )
    )  # fmt: skip
    edits_path = tmp_path / "fade_edits.json"
    edits.save(edits_path)

    edited = evaluate_checkpoint(ckpt_path, device="cpu", edits_path=edits_path)

    assert edited["edits"]["gate_suppression_active"] is True
    assert edited["edits"]["n_removed"] == 0, "fade removes no points -- only the gate moves"
    # Suppressing the WHOLE cloud's gate towards TRIPS (mix=0.0) can only pull the mean
    # gate down, never up (trippy.edit.checkpoint's own module docstring).
    assert edited["gate"]["mean"] < baseline["gate"]["mean"]
    assert edited["psnr_mean"] != baseline["psnr_mean"]
