"""Render one TRIPS checkpoint into a Brush-trainable image set (design-B step 1).

Module: trippy.distill.render_set
Purpose: the task brief's pipeline step 1 -- "render the TRIPS network
    output for the training cameras PLUS near-path interpolated cameras
    ... at the training width, writing an image set + a COLMAP-style model
    ... that Brush can train from". Ties together
    `trippy.distill.cameras.build_distill_camera_plan` (which poses),
    `trippy.render.candidate.render_candidate` (the actual forward render
    -- reused unchanged, not duplicated, per its own module docstring's
    "never opens a photo" invariant), and
    `trippy.distill.colmap_writer.write_distill_colmap_model` (the on-disk
    COLMAP text model Brush's own loader reads).
Invariants:
    - Never applies pose-refinement deltas: both anchor and interpolated
      poses use each image's raw COLMAP pose, exactly the convention
      `trippy.render.dolly`/`trippy.render.offpath` already use for
      arbitrary poses (see their own module docstrings) -- consistent
      across every pose in the set, anchor or interpolated.
    - `render_candidate`'s own `net.png` (the network output, after the
      U-Net + tone mapper) is what becomes each "photo" in the distilled
      image set -- never `raw_level0.png` (the pre-U-Net composite, which
      still has real holes) or a photographed pixel.
    - This function only touches MPS when the caller passes `device="mps"`
      explicitly (the same rule as every other MPS-capable entry point in
      this repo, AGENTS.md); it must only ever be invoked with `mps` from
      inside a `scripts/gpu_submit.sh` job.
Related docs: docs/EXPERIMENTS.md "Distillation (design B)"; docs/SPEC.md D2.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from trippy.constants import (
    CANDIDATE_NET_FILENAME,
    DISTILL_DEFAULT_INTERP_K,
    DISTILL_DEFAULT_MAX_INIT_POINTS,
    DISTILL_IMAGES_DIRNAME,
    DISTILL_MAX_JUMP_MULTIPLIER,
    DISTILL_RENDERS_DIRNAME,
    DISTILL_REPORT_FILENAME,
    DISTILL_SPARSE_DIRNAME,
    DISTILL_TRIPS_EXPORT_FILENAME,
)
from trippy.distill.cameras import build_distill_camera_plan, image_filename
from trippy.distill.colmap_writer import write_distill_colmap_model
from trippy.render.candidate import render_candidate
from trippy.train.eval import build_trainer_from_checkpoint


def render_distill_set(
    checkpoint_path: str | Path,
    out_dir: str | Path,
    device: str | None = None,
    interp_k: int = DISTILL_DEFAULT_INTERP_K,
    max_jump_multiplier: float = DISTILL_MAX_JUMP_MULTIPLIER,
    max_init_points: int | None = DISTILL_DEFAULT_MAX_INIT_POINTS,
) -> dict:
    """Render `checkpoint_path`'s network output into a Brush-ready COLMAP image set.

    Args:
        checkpoint_path: a `.pt` file written by `Trainer.save_checkpoint`.
        out_dir: output directory (created if missing). Writes:
            `<out_dir>/{DISTILL_TRIPS_EXPORT_FILENAME}` (the checkpoint's own
            trained point cloud, `Trainer.export_ply`'s exact output -- also
            the source for points3D.txt below);
            `<out_dir>/{DISTILL_RENDERS_DIRNAME}/` (`render_candidate`'s full
            per-pose output tree: raw/net/coverage/honesty PNGs + metrics.json,
            kept for inspection/audit, never opened by this function itself);
            `<out_dir>/{DISTILL_IMAGES_DIRNAME}/<name>.png` (one copy of each
            pose's `net.png`, renamed to the flat layout Brush expects);
            `<out_dir>/{DISTILL_SPARSE_DIRNAME}/{cameras,images,points3D}.txt`
            (the COLMAP text model, `trippy.distill.colmap_writer`);
            `<out_dir>/{DISTILL_REPORT_FILENAME}` (this function's return
            value).
        device: forwarded to `build_trainer_from_checkpoint`/`render_candidate`.
        interp_k, max_jump_multiplier: forwarded to `build_distill_camera_plan`.
        max_init_points: forwarded to `write_distill_colmap_model`.

    Returns:
        A dict: `{"checkpoint", "device", "scene_root", "width", "out_dir",
        "trips_export_ply", "images_dir", "sparse_dir",
        "n_points_source", "n_points_written", "n_cameras",
        "n_anchor_images", "n_interpolated_images", "n_skipped_pairs",
        "skipped_pairs", "median_consecutive_distance", "jump_threshold",
        "mean_coverage_full"}`.

    Raises:
        ValueError: the checkpoint's scene has no registered images.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mirrors trippy.cli._cmd_candidate_report's own pattern: build one Trainer to
    # export the checkpoint's trained point cloud, then let render_candidate below
    # rebuild its own Trainer from the checkpoint for the actual per-pose renders.
    trainer = build_trainer_from_checkpoint(checkpoint_path, device=device)
    scene_root = Path(trainer.cfg.scene_root)
    width = trainer.cfg.width
    export_path = trainer.export_ply(out_dir / DISTILL_TRIPS_EXPORT_FILENAME)
    xyz = trainer.point_params.xyz.detach().cpu().numpy().astype(np.float64)
    rgb = np.clip(trainer.point_params.feat[:, :3].detach().cpu().numpy(), 0.0, 1.0).astype(np.float64)
    n_points_source = int(xyz.shape[0])
    resolved_device = str(trainer.device)
    del trainer

    camera_plan = build_distill_camera_plan(
        scene_root, width, k=interp_k, max_jump_multiplier=max_jump_multiplier
    )
    poses = camera_plan.all_poses
    if not poses:
        raise ValueError(f"no registered cameras under {scene_root}")

    renders_dir = out_dir / DISTILL_RENDERS_DIRNAME
    render_metrics = render_candidate(
        checkpoint_path,
        poses,
        renders_dir,
        device=resolved_device,
        write_video_files=False,
        stop_at_low_coverage=False,
    )

    images_dir = out_dir / DISTILL_IMAGES_DIRNAME
    images_dir.mkdir(parents=True, exist_ok=True)
    for pose in poses:
        src = renders_dir / "frames" / pose.name / CANDIDATE_NET_FILENAME
        shutil.copyfile(src, images_dir / image_filename(pose.name))

    sparse_dir = out_dir / DISTILL_SPARSE_DIRNAME
    write_summary = write_distill_colmap_model(
        sparse_dir, camera_plan, xyz, rgb, max_init_points=max_init_points
    )

    report = {
        "checkpoint": str(checkpoint_path),
        "device": resolved_device,
        "scene_root": str(scene_root),
        "width": width,
        "out_dir": str(out_dir),
        "trips_export_ply": str(export_path),
        "images_dir": str(images_dir),
        "sparse_dir": str(sparse_dir),
        "n_points_source": n_points_source,
        "n_points_written": write_summary.n_points_written,
        "n_cameras": write_summary.n_cameras,
        "n_anchor_images": write_summary.n_anchor_images,
        "n_interpolated_images": write_summary.n_interpolated_images,
        "n_skipped_pairs": len(camera_plan.skipped_pairs),
        "skipped_pairs": [asdict(p) for p in camera_plan.skipped_pairs],
        "median_consecutive_distance": camera_plan.median_consecutive_distance,
        "jump_threshold": camera_plan.jump_threshold,
        "mean_coverage_full": render_metrics["mean_coverage_full"],
    }
    (out_dir / DISTILL_REPORT_FILENAME).write_text(json.dumps(report, indent=2) + "\n")
    return report
