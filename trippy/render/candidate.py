"""Render a trained checkpoint at a list of poses: raw / network / coverage + honesty artifacts.

Module: trippy.render.candidate
Purpose: the per-checkpoint renderer every candidate needs (docs/SPEC.md
    D10: "every candidate ships viewer artifact + audit number + dolly
    video + off-path honesty sheet"). Given a checkpoint and a list of
    `trippy.render.dolly.CameraPose`s (from `shade_dolly_poses`,
    `trippy.render.offpath.offpath_poses`, or hand-built), renders each pose
    through the full pipeline (pyramid -> U-Net -> tone mapper) and writes,
    per frame: the raw level-0 composite (no U-Net -- AGENTS.md's honesty
    rule), the network output, a from-scratch coverage heatmap (no photo
    pixels, safe to open per AGENTS.md), and a three-panel honesty triptych
    with low-coverage pixels outlined on the network panel. When `poses`
    form a sequence (the usual dolly case), also assembles the network and
    raw sequences into MP4s.
Invariants:
    - Never opens/reads a photo: every artifact here is either purely
      synthetic (from a random/off-path pose that no photo exists for) or
      derived only from T_final/coverage (never blended with photographed
      pixels) -- inspecting any output of this module never violates
      AGENTS.md's "never send scene imagery to a model" rule for the agent
      writing this code, though the images themselves are still photo-
      derived renders once the checkpoint was trained on a real scene, and
      Jordan's own review of them stays the actual verdict either way.
    - Pyramid rendering here duplicates the two-line body of
      `trippy.train.trainer.Trainer._render` (private) rather than calling
      it, and tone-mapping duplicates `NeuralCamera.forward`'s per-frame
      steps (rather than `Trainer._tone_map`), because this module needs a
      "no natural frame" fallback (mean exposure/white-balance across every
      trained frame) that `Trainer`'s public `render_at_pose` does not
      expose (see `_tone_map_for_pose`) -- both duplications are built only
      from `Trainer`'s already-public attributes (`point_params`, `net`,
      `cfg`, `background`, `camera`, `dataset.names`).
Related docs: docs/EXPERIMENTS.md "Mandatory honesty sheet", "Dolly camera
    paths"; docs/SPEC.md D10.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from trippy.constants import (
    CANDIDATE_COVERAGE_FILENAME,
    CANDIDATE_FRAMES_DIRNAME,
    CANDIDATE_HONESTY_FRAME_FILENAME,
    CANDIDATE_HONESTY_MAX_SHEET_FRAMES,
    CANDIDATE_HONESTY_SHEET_FILENAME,
    CANDIDATE_LOW_COVERAGE_THRESHOLD,
    CANDIDATE_METRICS_FILENAME,
    CANDIDATE_NET_FILENAME,
    CANDIDATE_NET_VIDEO_FILENAME,
    CANDIDATE_OUTLINE_COLOR,
    CANDIDATE_RAW_FILENAME,
    CANDIDATE_RAW_VIDEO_FILENAME,
    DOLLY_COVERAGE_STOP_THRESHOLD,
    VIDEO_DEFAULT_FPS,
)
from trippy.net.camera_model import default_uv_grid
from trippy.raster.pyramid import render_pyramid
from trippy.render.dolly import CameraPose, dolly_stop_index
from trippy.render.pyramid_render import coverage_stats, coverage_tensor
from trippy.render.sheets import colorize, contact_sheet, save_png, side_by_side
from trippy.render.video import write_video
from trippy.train.eval import build_trainer_from_checkpoint
from trippy.train.trainer import Trainer


def _render_layers(
    trainer: Trainer,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    image_hw: tuple[int, int],
    image_name: str | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], dict]:
    """Pyramid render + U-Net, before tone-mapping (see module docstring).

    Mirrors `Trainer._render` (trippy/train/trainer.py) exactly, built only
    from `Trainer`'s public attributes -- including hybrid design A's
    `gaussian_for_pose`/`hybrid.attach` pair, which is skipped entirely on a
    non-hybrid checkpoint (`trainer.hybrid is None`).
    """
    layers, aux = render_pyramid(
        trainer.point_params.xyz,
        trainer.point_params.size(),
        trainer.point_params.feat,
        trainer.point_params.conf(),
        K,
        R,
        t,
        image_hw,
        num_layers=trainer.cfg.layers,
        mode=trainer.cfg.mode,
        bg=trainer.background,
    )
    inputs = [layer.unsqueeze(0) for layer in layers]
    if trainer.hybrid is not None:
        gaussian = trainer.gaussian_for_pose(image_name, K, R, t, image_hw)
        inputs = trainer.hybrid.attach(inputs, gaussian)
    net_out = trainer.net(inputs)
    return net_out, layers, aux


def _fallback_tone_map(trainer: Trainer, net_out: torch.Tensor) -> torch.Tensor:
    """Tone-map with no natural per-image frame: mean exposure/white-balance, same LUT/vignette.

    `NeuralCamera.forward` (trippy.net.camera_model) always indexes
    `exposures_values`/`white_balance_values` by one `frame_index`; there is
    no built-in "mean" mode. This replays `forward`'s five steps by hand for
    the two per-frame ones (mean over every trained frame instead of a
    lookup); the vignette module and response-curve LUT are already frame-
    independent, so they are called exactly as `forward` calls them.
    """
    camera = trainer.camera
    x = net_out
    if camera.exposures_values is not None:
        exposure = camera.exposures_values.mean(dim=0, keepdim=True)
        x = x * torch.exp2(-exposure)
    if camera.white_balance_values is not None:
        wb = camera.white_balance_values.mean(dim=0, keepdim=True)
        x = wb * x
    if camera.vignette_net is not None:
        uv = default_uv_grid(x.shape[2], x.shape[3], x.device, x.dtype)
        x = camera.vignette_net(uv) * x
    if camera.camera_response is not None:
        x = camera.camera_response(x)
    else:
        x = torch.clamp(x, 0.0, 1.0)
    return x


def _tone_map_for_pose(trainer: Trainer, net_out: torch.Tensor, image_name: str | None) -> torch.Tensor:
    """Use `image_name`'s own trained exposure/white-balance if it is a registered image, else the mean."""
    names = trainer.dataset.names
    if image_name is not None and image_name in names:
        frame_index = names.index(image_name)
        idx = torch.tensor([frame_index], device=trainer.device, dtype=torch.long)
        return trainer.camera(net_out, idx)
    return _fallback_tone_map(trainer, net_out)


def _to_uint8(arr01: np.ndarray) -> np.ndarray:
    return np.round(np.clip(arr01, 0.0, 1.0) * 255.0).astype(np.uint8)


def _outline_low_coverage(rgb_u8: np.ndarray, coverage: np.ndarray, threshold: float) -> np.ndarray:
    """Draw a 1px `CANDIDATE_OUTLINE_COLOR` border around pixels where `coverage < threshold`.

    A pixel is on the boundary if it is below `threshold` and at least one
    of its 4-connected neighbours is not (`np.roll` wrap-around at the
    image edge is harmless here: it only ever compares a border pixel
    against the opposite border, and honesty sheets are for visual review,
    not precise boundary geometry).
    """
    mask = coverage < threshold
    boundary = (
        (mask & ~np.roll(mask, 1, axis=0))
        | (mask & ~np.roll(mask, -1, axis=0))
        | (mask & ~np.roll(mask, 1, axis=1))
        | (mask & ~np.roll(mask, -1, axis=1))
    )
    out = rgb_u8.copy()
    out[boundary] = CANDIDATE_OUTLINE_COLOR
    return out


def render_candidate(
    checkpoint_path: str | Path,
    poses: Sequence[CameraPose],
    out_dir: str | Path,
    device: str | None = None,
    fps: int = VIDEO_DEFAULT_FPS,
    write_video_files: bool = True,
    coverage_threshold: float = CANDIDATE_LOW_COVERAGE_THRESHOLD,
    max_sheet_frames: int = CANDIDATE_HONESTY_MAX_SHEET_FRAMES,
    stop_at_low_coverage: bool = False,
    dolly_stop_threshold: float = DOLLY_COVERAGE_STOP_THRESHOLD,
    gaussian_provider: Callable[..., torch.Tensor | None] | None = None,
) -> dict:
    """Render `poses` through a checkpoint and write every honesty artifact.

    Args:
        checkpoint_path: a `.pt` file written by `Trainer.save_checkpoint`.
        poses: camera poses to render, in the order videos/sheets use them
            (e.g. `trippy.render.dolly.shade_dolly_poses`'s output).
        out_dir: output directory (created if missing). Writes:
            `<out_dir>/frames/<pose.name>/{raw_level0,net,coverage,honesty}.png`,
            `<out_dir>/dolly.mp4` + `dolly_raw.mp4` (if `write_video_files`
            and `poses` is non-empty), `<out_dir>/honesty_sheet.png` (up to
            `max_sheet_frames` poses), `<out_dir>/metrics.json`.
        device: forwarded to `build_trainer_from_checkpoint`.
        fps: output video frame rate.
        write_video_files: assemble `dolly.mp4`/`dolly_raw.mp4`. False for
            pose sets that are not a single coherent camera path (e.g.
            off-path honesty poses, which may not even share one image size).
        coverage_threshold: `_outline_low_coverage`'s threshold (docs/
            EXPERIMENTS.md "Mandatory honesty sheet": pixels with raw
            coverage below this are likely U-Net hallucination).
        max_sheet_frames: cap on how many poses' triptychs go into
            `honesty_sheet.png` (every pose still gets its own per-frame
            `honesty.png`).
        stop_at_low_coverage: for a dolly path (`poses` ordered along a
            camera trajectory), truncate the assembled videos to
            `trippy.render.dolly.dolly_stop_index`'s cutoff instead of the
            full path, so the clip stops before the camera visibly exits
            the point cloud (docs/EXPERIMENTS.md "Dolly camera paths").
            Every pose is still rendered and gets its own per-frame PNGs
            and an entry in `metrics["frames"]`; only the video/mean-
            coverage summary is affected. False for off-path honesty poses
            (not a single ordered path, so "stop" is meaningless there).
        dolly_stop_threshold: forwarded to `dolly_stop_index`.
        gaussian_provider: hybrid design A only -- a
            `(name, K, R, t, image_hw) -> (G, H, W) tensor | None` callback
            supplying the Gaussian block for **every** pose here. None keeps
            the lazy live-gsrender provider
            `trippy.train.eval.build_trainer_from_checkpoint` already
            installed; tests pass a fake so no PLY or MPS is ever touched.
            No pose here is a photographed one (dolly/off-path cameras are
            displaced from their anchor image), so a precomputed render is
            never a valid substitute -- see
            `trippy.hybrid.gsrender_live.gaussian_provider_for`. Ignored
            entirely on a non-hybrid checkpoint.

    Returns:
        The metrics dict also written to `<out_dir>/metrics.json`:
        `{"checkpoint", "device", "n_frames", "mean_coverage_full",
        "frames": [{"name", "image_hw", "coverage_mean_full",
        "coverage_mean_center"}, ...], "videos": {"net", "raw"} (only if
        written), "honesty_sheet" (only if written)}`. When
        `stop_at_low_coverage` is True, also includes `"dolly_stop_index"`
        (the last kept frame's index into `"frames"`), `"dolly_stop_threshold"`,
        and `"dolly_stopped_early"` (whether any frames were cut from the video).
    """
    trainer = build_trainer_from_checkpoint(checkpoint_path, device=device)
    trainer.net.eval()
    trainer.camera.eval()
    if gaussian_provider is not None:
        trainer.gaussian_provider = gaussian_provider

    out_dir = Path(out_dir)
    frames_dir = out_dir / CANDIDATE_FRAMES_DIRNAME
    frames_dir.mkdir(parents=True, exist_ok=True)

    net_frames: list[np.ndarray] = []
    raw_frames: list[np.ndarray] = []
    sheet_images: list[np.ndarray] = []
    sheet_labels: list[str] = []
    frame_metrics: list[dict] = []

    with torch.no_grad():
        for pose in poses:
            K = torch.tensor(np.asarray(pose.K, dtype=np.float32), device=trainer.device)
            R = torch.tensor(np.asarray(pose.R, dtype=np.float32), device=trainer.device)
            t = torch.tensor(np.asarray(pose.t, dtype=np.float32), device=trainer.device)
            image_hw = (int(pose.image_hw[0]), int(pose.image_hw[1]))

            net_out, layers, aux = _render_layers(trainer, K, R, t, image_hw, pose.image_name)
            pred = _tone_map_for_pose(trainer, net_out, pose.image_name)

            raw01 = layers[0][:3].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
            net01 = pred[0].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
            coverage_t = coverage_tensor(aux)
            cov_stats = coverage_stats(coverage_t)
            coverage_np = coverage_t.cpu().numpy()
            coverage_color = colorize(coverage_np, 0.0, 1.0)

            raw_u8 = _to_uint8(raw01)
            net_u8 = _to_uint8(net01)
            net_outlined_u8 = _outline_low_coverage(net_u8, coverage_np, coverage_threshold)

            frame_dir = frames_dir / pose.name
            save_png(frame_dir / CANDIDATE_RAW_FILENAME, raw_u8)
            save_png(frame_dir / CANDIDATE_NET_FILENAME, net_u8)
            save_png(frame_dir / CANDIDATE_COVERAGE_FILENAME, coverage_color)
            honesty = side_by_side(
                [raw_u8, net_outlined_u8, coverage_color],
                ["raw L0", f"network (outline: coverage<{coverage_threshold:g})", "coverage"],
            )
            save_png(frame_dir / CANDIDATE_HONESTY_FRAME_FILENAME, honesty)

            net_frames.append(net_u8)
            raw_frames.append(raw_u8)
            if len(frame_metrics) < max_sheet_frames:
                sheet_images += [raw_u8, net_outlined_u8, coverage_color]
                sheet_labels += [f"{pose.name}:raw", f"{pose.name}:net", f"{pose.name}:coverage"]

            frame_metrics.append(
                {
                    "name": pose.name,
                    "image_hw": list(image_hw),
                    "coverage_mean_full": cov_stats["mean_full"],
                    "coverage_mean_center": cov_stats["mean_center"],
                }
            )

    metrics: dict = {
        "checkpoint": str(checkpoint_path),
        "device": str(trainer.device),
        "n_frames": len(frame_metrics),
        "mean_coverage_full": (
            float(np.mean([f["coverage_mean_full"] for f in frame_metrics])) if frame_metrics else 0.0
        ),
        "frames": frame_metrics,
    }

    if stop_at_low_coverage and frame_metrics:
        stop_index = dolly_stop_index(
            [f["coverage_mean_center"] for f in frame_metrics], dolly_stop_threshold
        )
        metrics["dolly_stop_index"] = stop_index
        metrics["dolly_stop_threshold"] = dolly_stop_threshold
        metrics["dolly_stopped_early"] = stop_index < len(frame_metrics) - 1
        net_frames = net_frames[: stop_index + 1]
        raw_frames = raw_frames[: stop_index + 1]

    if write_video_files and net_frames:
        videos = {
            "net": str(write_video(out_dir / CANDIDATE_NET_VIDEO_FILENAME, net_frames, fps=fps)),
            "raw": str(write_video(out_dir / CANDIDATE_RAW_VIDEO_FILENAME, raw_frames, fps=fps)),
        }
        metrics["videos"] = videos

    if sheet_images:
        sheet = contact_sheet(sheet_images, sheet_labels, cols=3)
        sheet_path = save_png(out_dir / CANDIDATE_HONESTY_SHEET_FILENAME, sheet)
        metrics["honesty_sheet"] = str(sheet_path)

    (out_dir / CANDIDATE_METRICS_FILENAME).write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
