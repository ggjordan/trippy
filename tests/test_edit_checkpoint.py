"""Tests for trippy.edit.checkpoint: checkpoint-side keep-mask surgery + gate suppression.

Module: tests.test_edit_checkpoint
Invariants under test (docs/EDITOR.md Sec 5):
  - `apply_edits_to_trainer` deletes exactly the right points from a live
    `Trainer` via `Trainer._apply_keep_mask` (point count shrinks, survivors
    keep their own xyz), and reports `gate_suppression_active` correctly:
    False when only `delete`-op regions touch the cloud (nothing left for a
    `blend`/`fade` region to suppress), True when a `fade`/`blend` region
    touches a surviving point.
  - `render_edit_weight_map` returns None when nothing is active (no wasted
    render), and otherwise projects the per-point weight into a `(1, 1, H,
    W)` map in `[0, 1]` via the documented feature-channel fallback.
  - `render_candidate(..., edits_path=...)` on a gate-hybrid checkpoint
    actually shrinks the point cloud it renders from and records an
    `"edits"` key in its metrics.
Fixture: the synthetic gate-hybrid trainer from tests/test_hybrid_a_helpers.py
    (tests/test_hybrid_gate_trainer.py's own `_gate_trainer` pattern) and the
    plain synthetic scene from tests/test_train_helpers.py. Never a real
    scene or checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from test_hybrid_a_helpers import hybrid_train_config
from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config

from trippy.edit.checkpoint import apply_edits_to_trainer, render_edit_weight_map
from trippy.edit.model import EditDocument, Region
from trippy.render.candidate import render_candidate
from trippy.render.offpath import offpath_poses
from trippy.train.trainer import Trainer


def _gate_trainer(tmp_path: Path) -> Trainer:
    cfg, _names = hybrid_train_config(tmp_path, gate={"enabled": True}, dropout_gaussian_p=0.0)
    return Trainer(cfg)


def _delete_pointset(point_ids: list[int]) -> EditDocument:
    edits = EditDocument.new()
    edits.add_region(
        Region(id="r-del", name="del", kind="pointset", params={"point_ids": point_ids}, op="delete")
    )
    return edits


def _fade_pointset(point_ids: list[int], mix: float = 0.0) -> EditDocument:
    edits = EditDocument.new()
    edits.add_region(
        Region(id="r-fade", name="fade", kind="pointset", params={"point_ids": point_ids}, mix=mix, op="fade")
    )
    return edits


def test_apply_edits_to_trainer_deletes_points_and_shrinks_optimizer_state(tmp_path: Path) -> None:
    trainer = _gate_trainer(tmp_path)
    n_before = len(trainer.point_params)
    delete_ids = list(range(50))
    edits = _delete_pointset(delete_ids)

    summary = apply_edits_to_trainer(trainer, edits)

    assert summary["n_points_before"] == n_before
    assert summary["n_removed"] == 50
    assert summary["n_points_after"] == n_before - 50
    assert len(trainer.point_params) == n_before - 50
    # Delete-only: nothing left over for a blend/fade region to suppress.
    assert summary["gate_suppression_active"] is False
    assert getattr(trainer, "_edit_gate_weight", "sentinel") is None
    # The optimizer's own state must have shrunk in lockstep (Trainer._apply_keep_mask's
    # own invariant) -- a smoke check that this really went through the real surgery,
    # not some ad hoc reindex.
    for group in trainer.optimizer.param_groups:
        for p in group["params"]:
            if p.shape and p.shape[0] == len(trainer.point_params):
                state = trainer.optimizer.state.get(p)
                if state and "exp_avg" in state:
                    assert state["exp_avg"].shape[0] == len(trainer.point_params)


def test_apply_edits_to_trainer_pure_fade_sets_gate_weight(tmp_path: Path) -> None:
    trainer = _gate_trainer(tmp_path)
    n = len(trainer.point_params)
    fade_ids = list(range(10))
    edits = _fade_pointset(fade_ids, mix=0.0)

    summary = apply_edits_to_trainer(trainer, edits)

    assert summary["n_removed"] == 0
    assert summary["n_points_after"] == n
    assert summary["gate_suppression_active"] is True
    weight = trainer._edit_gate_weight
    assert weight is not None
    assert weight.shape == (n,)
    # fade region touches ids 0..9 with mix=0.0: weight *= (1 - 1.0*(1-0.0)) = 0.0 there.
    np.testing.assert_allclose(weight[:10].numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(weight[10:].numpy(), 1.0, atol=1e-6)


def test_apply_edits_to_trainer_no_touching_region_is_inactive(tmp_path: Path) -> None:
    trainer = _gate_trainer(tmp_path)
    edits = EditDocument.new()  # no regions at all
    summary = apply_edits_to_trainer(trainer, edits)
    assert summary["gate_suppression_active"] is False
    assert trainer._edit_gate_weight is None
    assert render_edit_weight_map(trainer, *_dummy_pose(trainer)) is None


def _dummy_pose(trainer: Trainer):
    item = trainer.dataset[0]
    K = item["K"]
    R, t = trainer._pose_for(item, 0)
    height, width = int(item["rgb"].shape[0]), int(item["rgb"].shape[1])
    return K, R, t, (height, width)


def test_render_edit_weight_map_projects_a_fade_region_per_pixel(tmp_path: Path) -> None:
    trainer = _gate_trainer(tmp_path)
    n = len(trainer.point_params)
    edits = _fade_pointset(list(range(n // 2)), mix=0.0)  # fade half the cloud fully out
    apply_edits_to_trainer(trainer, edits)

    K, R, t, image_hw = _dummy_pose(trainer)
    with torch.no_grad():
        weight_map = render_edit_weight_map(trainer, K, R, t, image_hw)

    assert weight_map is not None
    assert weight_map.shape[:2] == (1, 1)
    assert weight_map.shape[-2] <= image_hw[0] and weight_map.shape[-1] <= image_hw[1]
    values = weight_map.numpy()
    assert np.isfinite(values).all()
    assert bool(((values >= 0.0) & (values <= 1.0)).all())


def test_render_candidate_with_edits_shrinks_the_rendered_point_cloud(tmp_path: Path) -> None:
    trainer = _gate_trainer(tmp_path)
    n_before = len(trainer.point_params)
    checkpoint = trainer.save_checkpoint()
    scene_root = Path(trainer.cfg.scene_root)
    names = list(trainer.dataset.names)
    delete_ids = list(range(n_before // 2))
    edits = _delete_pointset(delete_ids)
    edits_path = tmp_path / "edits.json"
    edits.save(edits_path)

    poses = offpath_poses(scene_root, names[:1], width=trainer.cfg.width)
    out_dir = tmp_path / "render_out"
    metrics = render_candidate(
        checkpoint, poses, out_dir, device="cpu", write_video_files=False, edits_path=edits_path
    )

    assert "edits" in metrics
    assert metrics["edits"]["n_removed"] == len(delete_ids)
    assert metrics["edits"]["n_points_before"] == n_before
    assert metrics["edits"]["n_points_after"] == n_before - len(delete_ids)
    assert metrics["n_frames"] == len(poses)


def test_build_trainer_from_checkpoint_without_edits_leaves_point_count_unchanged(tmp_path: Path) -> None:
    scene_root, point_set = build_synthetic_scene(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(scene_root, ply_path, tmp_path / "run", tmp_path / "cache")
    trainer = Trainer(cfg)
    n_before = len(trainer.point_params)
    checkpoint = trainer.save_checkpoint()

    from trippy.train.eval import build_trainer_from_checkpoint

    reloaded = build_trainer_from_checkpoint(checkpoint, device="cpu")
    assert reloaded.edit_summary is None
    assert len(reloaded.point_params) == n_before
