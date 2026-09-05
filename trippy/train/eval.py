"""Standalone evaluation from a checkpoint: held-out metrics + off-path honesty renders.

Module: trippy.train.eval
Invariants: `evaluate_checkpoint` rebuilds a full `Trainer` from the
    checkpoint's own saved `cfg` (so the dataset/point-source/split are
    reconstructed identically to how the checkpoint was trained) and then
    loads the trained state into it -- it never re-runs training. This
    module does no optimisation and never writes back to the checkpoint.
Related docs: docs/EXPERIMENTS.md "Training runs", "Held-out PSNR and
    LPIPS"; docs/EXPERIMENTS.md "Dolly camera paths" (the off-path renderer
    here is the API the later dolly-path generator plugs into -- it does
    not itself generate a camera path).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from trippy.render.sheets import colorize, save_png, side_by_side
from trippy.train import checkpoint_io
from trippy.train.config import TrainConfig
from trippy.train.trainer import Trainer


def build_trainer_from_checkpoint(checkpoint_path: str | Path, device: str | None = None) -> Trainer:
    """Rebuild a Trainer (dataset, point source, net, camera) from a checkpoint's own config.

    Args:
        checkpoint_path: a `.pt` file written by `Trainer.save_checkpoint`.
        device: override the checkpoint's own `cfg.device` (e.g. force
            "cpu" to inspect an `mps`-trained checkpoint on a laptop);
            None keeps the checkpoint's original device.

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
    return trainer


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    images: list[str] | None = None,
    device: str | None = None,
) -> dict:
    """Evaluate a checkpoint on `images` (default: its own held-out split).

    Args:
        checkpoint_path: a `.pt` file written by `Trainer.save_checkpoint`.
        images: image names to evaluate (must exist in the checkpoint's
            scene); None uses the checkpoint's own held-out split.
        device: forwarded to `build_trainer_from_checkpoint`.

    Returns:
        The `Trainer.evaluate()` metrics dict (also written to
        `<run_dir>/eval_ep<NNNN>/metrics.json` + `sheet.png`, same as a
        mid-training eval).
    """
    trainer = build_trainer_from_checkpoint(checkpoint_path, device=device)
    return trainer.evaluate(names=images)


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
