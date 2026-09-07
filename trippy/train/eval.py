"""Standalone evaluation from a checkpoint: held-out metrics + off-path honesty renders.

Module: trippy.train.eval
Invariants: a hybrid (design A) checkpoint gets a lazy live-gsrender
    `gaussian_provider` installed by `build_trainer_from_checkpoint`, so
    off-path/dolly poses -- which have no precomputed Gaussian render -- still
    reach the network with their Gaussian channels filled in.
    `evaluate_checkpoint` rebuilds a full `Trainer` from the
    checkpoint's own saved `cfg` (so the dataset/point-source/split are
    reconstructed identically to how the checkpoint was trained) and then
    loads the trained state into it -- it never re-runs training. This
    module does no optimisation and never writes back to the checkpoint.
    `edits_path` (docs/EDITOR.md Sec 5) runs `trippy.edit.checkpoint.
    apply_edits_to_trainer` right after the trained state loads: every
    `delete`-op region's points are removed from the freshly built
    `Trainer` (the same index-select surgery `Trainer._apply_keep_mask`
    performs at a training epoch boundary), permanently for the life of
    this in-memory `Trainer` object -- never written back to the `.pt`
    file. `evaluate_checkpoint`'s own render path (`Trainer.evaluate`)
    reflects that deletion in full (fewer points render) but NOT the
    gate-suppression multiply `trippy.render.candidate.render_candidate`
    applies for `blend`/`fade` regions -- see `trippy.edit.checkpoint`'s
    own module docstring for why (`Trainer.evaluate` lives outside this
    module's editable surface).
Related docs: docs/EXPERIMENTS.md "Training runs", "Held-out PSNR and
    LPIPS"; docs/EXPERIMENTS.md "Dolly camera paths" (the off-path renderer
    here is the API the later dolly-path generator plugs into -- it does
    not itself generate a camera path); docs/EDITOR.md Sec 5 "Publish".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from trippy.constants import TRAIN_EVAL_MANUAL_DIRNAME_FMT
from trippy.hybrid import gate as gate_mod
from trippy.hybrid.gsrender_live import gaussian_provider_for
from trippy.render.sheets import colorize, save_png, side_by_side
from trippy.train import checkpoint_io
from trippy.train.config import TrainConfig
from trippy.train.trainer import Trainer


def build_trainer_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | None = None,
    gate_scale: float | None = None,
    edits_path: str | Path | None = None,
) -> Trainer:
    """Rebuild a Trainer (dataset, point source, net, camera) from a checkpoint's own config.

    Args:
        checkpoint_path: a `.pt` file written by `Trainer.save_checkpoint`.
        device: override the checkpoint's own `cfg.device` (e.g. force
            "cpu" to inspect an `mps`-trained checkpoint on a laptop);
            None keeps the checkpoint's original device.
        gate_scale: override the blend gate's post-training scale
            (`trippy.hybrid.gate`): 0 = pure TRIPS, 1 = the mix training
            chose, 2 = pushed all the way to the splat. None keeps the
            checkpoint's own `hybrid.gate_scale`; ignored (with no error) on
            a checkpoint trained without the gate, so a caller may always
            pass it.
        edits_path: an `edits.json` path (docs/EDITOR.md Sec 5). When given,
            `trippy.edit.checkpoint.apply_edits_to_trainer` runs against the
            freshly loaded `Trainer` before it is returned: `delete`-op
            regions permanently remove their points from this in-memory
            Trainer (never written back to `checkpoint_path`), and any
            `blend`/`fade` region's per-point weight is stashed for
            `trippy.render.candidate.render_candidate`'s gate-suppression
            step to pick up. The summary dict `apply_edits_to_trainer`
            returns is attached as `trainer.edit_summary` (None when
            `edits_path` is None). None (default) leaves the checkpoint's
            point cloud untouched, exactly as before this parameter existed.

    Returns:
        A `Trainer` with the checkpoint's trained state loaded (weights,
        optimizer, epoch counter) -- ready for `evaluate()` or
        `render_at_pose`-style inspection.
    """
    payload = checkpoint_io.load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = TrainConfig.from_dict(payload["cfg"])
    if device is not None:
        cfg.device = device
    trainer = Trainer(cfg)
    trainer.load_state(payload)
    if trainer.hybrid is not None:
        # Design A: poses with no precomputed render (dolly/off-path) need the PLY
        # rendered live. The renderer is lazy -- nothing is loaded, and MPS is never
        # touched, unless such a pose is actually rendered (a held-out eval never is).
        trainer.gaussian_provider = gaussian_provider_for(
            cfg.hybrid, trainer.hybrid, device=str(trainer.device)
        )
    if gate_scale is not None and trainer.gate_enabled:
        trainer.gate_scale = gate_mod.clamp_scale(gate_scale)
    trainer.edit_summary = None
    if edits_path is not None:
        from trippy.edit.checkpoint import apply_edits_to_trainer
        from trippy.edit.model import EditDocument

        edits = EditDocument.load(edits_path)
        trainer.edit_summary = apply_edits_to_trainer(trainer, edits)
    return trainer


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    images: list[str] | None = None,
    device: str | None = None,
    calibrate: bool | None = None,
    calibrate_white_balance: bool | None = None,
    exposure_mode: str | None = None,
    gate_scale: float | None = None,
    edits_path: str | Path | None = None,
) -> dict:
    """Evaluate a checkpoint on `images` (default: its own held-out split).

    Args:
        checkpoint_path: a `.pt` file written by `Trainer.save_checkpoint`.
        images: image names to evaluate (must exist in the checkpoint's
            scene); None uses the checkpoint's own held-out split.
        device: forwarded to `build_trainer_from_checkpoint`.
        calibrate: fit each held-out image's own exposure (test-time
            photometric calibration, `Trainer.calibrate_frame`) and report
            the calibrated metrics *alongside* the strict ones. None keeps
            the checkpoint's own `cfg.eval_calibrate_camera` (False for
            every run so far). This still does no training and never
            writes back to the checkpoint -- the fitted scalars are local
            to the call.
        calibrate_white_balance: fit red/blue white balance too (green is
            pinned, as in training). None keeps the checkpoint's config.
        exposure_mode: which exposure/WB a held-out image's headline
            "_eval" numbers use -- "own"/"neighbours"/"calibrate"
            (`trippy.constants.EVAL_EXPOSURE_MODES`). None keeps the
            checkpoint's own `cfg.eval_exposure_mode` ("neighbours" for any
            checkpoint saved by a config that didn't set it explicitly --
            see `trippy.train.config.TrainConfig.eval_exposure_mode`).
        gate_scale: forwarded to `build_trainer_from_checkpoint` -- re-score
            the same checkpoint at a different splat-vs-TRIPS mix without
            retraining. The gate map itself is unchanged (it is what the
            network learned); only how much of it is applied changes.
        edits_path: forwarded to `build_trainer_from_checkpoint`
            (docs/EDITOR.md Sec 5): `delete`-op regions permanently remove
            their points from this evaluation's `Trainer` before any image
            is rendered. See `build_trainer_from_checkpoint`'s own
            docstring for what this does NOT do (the gate-suppression
            multiply, out of `Trainer.evaluate`'s reach from this module).

    Returns:
        The `Trainer.evaluate()` metrics dict -- including the "per_image"
        exposure diagnostics, the "shade"/"other" held-out split, the
        `exposure_mode`-resolved "shade_eval"/"other_eval" headline numbers,
        and (with `calibrate`) "shade_calibrated"/"other_calibrated" (see
        `Trainer.evaluate`) -- also
        written to `<run_dir>/eval_manual_<timestamp>/metrics.json` +
        `sheet.jpg` (a distinct directory from any mid-training/`--report`
        eval, named by wall-clock time rather than epoch, so repeated manual
        re-evals never collide with each other or with the checkpoint's own
        epoch directory) and appended as an `{"eval": True, ...}` row to the
        run's own `metrics.jsonl`, so `trippy leaderboard` picks up the
        shade split for a checkpoint that finished training before this
        split existed, without retraining it. Includes an "edits" key
        (`trainer.edit_summary`) when `edits_path` was given.
    """
    trainer = build_trainer_from_checkpoint(
        checkpoint_path, device=device, gate_scale=gate_scale, edits_path=edits_path
    )
    if calibrate_white_balance is not None:
        trainer.cfg.eval_calibrate_white_balance = bool(calibrate_white_balance)
    eval_dirname = TRAIN_EVAL_MANUAL_DIRNAME_FMT.format(ts=time.strftime("%Y%m%d-%H%M%S"))
    metrics = trainer.evaluate(
        names=images, eval_dirname=eval_dirname, calibrate=calibrate, exposure_mode=exposure_mode
    )
    if trainer.edit_summary is not None:
        metrics["edits"] = trainer.edit_summary
    return metrics


def render_offpath(
    checkpoint_path: str | Path,
    poses_path: str | Path,
    out_dir: str | Path,
    device: str | None = None,
) -> list[Path]:
    """Render honesty triplets (raw | network | coverage) at arbitrary, non-photographed poses.

    No ground-truth photo exists for an off-path pose, so no PSNR/SSIM/LPIPS
    is computed here -- only the three-panel honesty sheet (docs/
    EXPERIMENTS.md "Mandatory honesty sheet"). This function is the stable
    API the later dolly-camera-path generator (docs/EXPERIMENTS.md "Dolly
    camera paths") plugs into; it does not itself generate a path.

    Args:
        checkpoint_path: a `.pt` file written by `Trainer.save_checkpoint`.
        poses_path: JSON file containing a list of
            `{"name": str, "R": [[3x3]], "t": [3], "K": [[3x3]], "image_hw": [h, w]}`
            objects, world->camera COLMAP-frame poses.
        out_dir: directory `offpath_<name>.png` sheets are written to.
        device: forwarded to `build_trainer_from_checkpoint`.

    Returns:
        Paths to the written sheet PNGs, in `poses_path`'s order.
    """
    trainer = build_trainer_from_checkpoint(checkpoint_path, device=device)
    poses = json.loads(Path(poses_path).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer.net.eval()
    trainer.camera.eval()
    written: list[Path] = []
    with torch.no_grad():
        for i, pose in enumerate(poses):
            name = pose.get("name", f"offpath_{i:03d}")
            R = torch.tensor(pose["R"], dtype=torch.float32, device=trainer.device)
            t = torch.tensor(pose["t"], dtype=torch.float32, device=trainer.device)
            K = torch.tensor(pose["K"], dtype=torch.float32, device=trainer.device)
            height, width = int(pose["image_hw"][0]), int(pose["image_hw"][1])

            # frame_index=0: off-path poses have no associated training image, so there is
            # no natural per-image exposure/response to look up (see render_at_pose docstring).
            pred, layers, aux = trainer.render_at_pose(K, R, t, (height, width), frame_index=0)

            raw = layers[0][:3].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
            pred_np = pred[0].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
            coverage = colorize((1.0 - aux["t_final"][0]).clamp(0.0, 1.0).cpu().numpy(), 0.0, 1.0)

            sheet = side_by_side([raw, pred_np, coverage], ["raw L0", "network", "coverage"])
            path = out_dir / f"offpath_{name}.png"
            save_png(path, sheet)
            written.append(path)
    return written
