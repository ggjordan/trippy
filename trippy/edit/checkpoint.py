"""Apply an `EditDocument` to a live `Trainer`: checkpoint-side keep-mask surgery + gate suppression.

Module: trippy.edit.checkpoint
Purpose: the checkpoint-side half of docs/EDITOR.md Sec 5 "Publish" -- let
    `trippy candidate-report --edits`/`trippy eval --edits`/`trippy distill
    --edits` run the SAME `edits.json` the bundle-publish path
    (`trippy.edit.apply`) uses directly against a trained checkpoint, so a
    candidate report, held-out eval, or distillation render pass reflects an
    edit without first re-exporting a bundle.
Invariants:
    - Point deletion reuses `Trainer._apply_keep_mask` verbatim (the exact
      index-select-and-rebuild-optimiser-state surgery `maybe_prune_points`
      already performs at epoch boundaries) -- the edited `Trainer`'s point
      count shrinks exactly once, permanently, on the in-memory object this
      call was given; nothing here writes back to the `.pt` file.
    - `blend`/`fade` (non-delete) regions have no live per-point SPLAT
      channel to move on a checkpoint-only path: docs/EDITOR.md Sec 7 notes
      the splat-side weight-compositing half of the region model "has not
      been built or verified" at all, so a region that wants "more splat
      here" cannot be honoured by editing a Gaussian PLY this path never
      touches. The one thing this path CAN safely do is stop an un-edited,
      live-rendered splat (`Trainer.gaussian_provider`, wired independently
      of `edits.json`) from leaking back into a region the edit asked to
      hide: the trained gate (`trippy.hybrid.gate`, 0 = TRIPS, 1 = splat) is
      multiplied, per pixel, by the SAME per-point weight
      `trippy.edit.weights.compose_trips_weights` already produces for the
      bundle-publish path (1 = pure TRIPS/no override, ramping down inside a
      blend/fade/lid region) -- suppressing towards TRIPS wherever an edit
      wants less of the ORIGINAL content shown, never amplifying a splat
      presence the trained gate did not already choose on its own. A
      `delete` region needs no multiply at all: its points are already gone
      from the pyramid render by the time `render_edit_weight_map` runs, so
      the multiplier there is 1.0 -- a true no-op -- and the deletion alone
      is what keeps that geometry out of the image.
    - The per-pixel projection reuses `render_pyramid`'s own alpha
      compositing (docs/EDITOR.md Sec 2 "why per-point, not per-pixel depth
      lookup"), not a second per-pixel geometry test: the per-point weight
      rides as feature channel `EDIT_GATE_WEIGHT_CHANNEL_INDEX` of an
      `EDIT_GATE_WEIGHT_NUM_CHANNELS`-wide `render_pyramid` call and is read
      back from level 0 after the same front-to-back composite the colour
      channels use -- the documented fallback for "per-pixel weight
      rendering is not yet available in the Python renderer"
      (docs/EDITOR.md Sec 5).
    - `evaluate_checkpoint`'s own render path (`Trainer.evaluate`, in
      `trippy.train.trainer`) is NOT wired to the gate-suppression multiply:
      `Trainer.evaluate` is out of this module's reach without editing
      `trippy/train/trainer.py`. `trippy eval --edits` therefore applies the
      keep-mask deletion (real, and reflected in every held-out render) but
      not the gate multiply -- a documented gap, not a silent one. The
      `render_candidate` path (`trippy candidate-report`, `trippy distill
      --stage render`) gets both.
Related docs: docs/EDITOR.md Sec 5 "Publish"; trippy.edit.apply (the
    bundle-side sibling of this module); trippy.edit.weights
    (`compose_trips_weights`, the same composition this reuses);
    trippy.train.trainer.Trainer._apply_keep_mask.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from trippy.constants import EDIT_GATE_WEIGHT_CHANNEL_INDEX, EDIT_GATE_WEIGHT_NUM_CHANNELS
from trippy.edit.model import EditDocument
from trippy.edit.weights import compose_trips_weights
from trippy.raster.pyramid import render_pyramid

__all__ = ["apply_edits_to_trainer", "render_edit_weight_map"]


def apply_edits_to_trainer(trainer: Any, edits: EditDocument) -> dict[str, Any]:
    """Delete edited-out points from `trainer` in place; stash the survivors' gate-suppression weight.

    Args:
        trainer: a live `trippy.train.trainer.Trainer` (e.g. from
            `trippy.train.eval.build_trainer_from_checkpoint`) -- mutated in
            place via `Trainer._apply_keep_mask`.
        edits: the loaded `EditDocument`.

    Returns:
        `{"n_points_before", "n_removed", "n_points_after", "n_regions",
        "gate_suppression_active"}`. `gate_suppression_active` is True iff
        some surviving point's composed weight differs from
        `trippy.constants.EDIT_GATE_DEFAULT_WEIGHT` (i.e. an enabled
        `blend`/`fade` region touches a surviving point) -- when False,
        `render_edit_weight_map` always returns None for this trainer (the
        gate is left exactly as trained) and the per-pose auxiliary render
        it would otherwise cost is skipped entirely.
    """
    n_before = len(trainer.point_params)
    xyz = trainer.point_params.xyz.detach().cpu().numpy().astype(np.float64)
    composed = compose_trips_weights(edits, xyz)
    keep = ~composed.delete_mask
    n_removed = trainer._apply_keep_mask(keep, "edit_delete")

    remaining_weight = composed.weight[keep]
    active = bool(remaining_weight.size) and not np.allclose(remaining_weight, 1.0)
    trainer._edit_gate_weight = (
        torch.from_numpy(remaining_weight.astype(np.float32)).to(trainer.point_params.xyz.device)
        if active
        else None
    )
    return {
        "n_points_before": n_before,
        "n_removed": n_removed,
        "n_points_after": len(trainer.point_params),
        "n_regions": len(edits.regions),
        "gate_suppression_active": active,
    }


def render_edit_weight_map(
    trainer: Any, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, image_hw: tuple[int, int]
) -> torch.Tensor | None:
    """Per-pixel projection of `trainer._edit_gate_weight`, or None when there is nothing to suppress.

    See module docstring: splats the per-point weight through the ordinary
    `render_pyramid` alpha composite as one auxiliary feature channel and
    reads level 0's own copy of that channel back -- no depth buffer, no
    second geometry test, and no change to the network's own render.

    Args:
        trainer: a `Trainer` `apply_edits_to_trainer` has already run on
            (reads `trainer._edit_gate_weight`, `.point_params`, `.cfg`).
        K, R, t, image_hw: the same pose/size this frame's real render uses.

    Returns:
        `(1, 1, H, W)` float tensor in `[0, 1]`, or None when
        `apply_edits_to_trainer` found nothing to suppress (never called,
        or every surviving point's weight was exactly the default).
    """
    weight = getattr(trainer, "_edit_gate_weight", None)
    if weight is None:
        return None
    n = weight.shape[0]
    feat_aux = torch.zeros(
        (n, EDIT_GATE_WEIGHT_NUM_CHANNELS), dtype=trainer.point_params.feat.dtype, device=weight.device
    )
    feat_aux[:, EDIT_GATE_WEIGHT_CHANNEL_INDEX] = weight
    layers, _aux = render_pyramid(
        trainer.point_params.xyz,
        trainer.point_params.size(),
        feat_aux,
        trainer.point_params.conf(),
        K,
        R,
        t,
        image_hw,
        num_layers=trainer.cfg.layers,
        mode=trainer.cfg.mode,
        bg=None,
        pixel_center=trainer.cfg.pixel_center,
        pyramid_halving=trainer.cfg.pyramid_halving,
    )
    channel = layers[0][EDIT_GATE_WEIGHT_CHANNEL_INDEX : EDIT_GATE_WEIGHT_CHANNEL_INDEX + 1]
    return channel.unsqueeze(0).clamp(0.0, 1.0)
