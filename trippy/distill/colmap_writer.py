"""Builds the synthetic COLMAP text model for one distillation image set.

Module: trippy.distill.colmap_writer
Purpose: turn a `trippy.distill.cameras.DistillCameraPlan` (anchor +
    interpolated CameraPoses) and a point cloud (the TRIPS export) into a
    `trippy.scene.colmap_io.ColmapScene` and write it with
    `trippy.scene.colmap_io.save_colmap_model_txt` -- the "COLMAP-style
    model (cameras/images txt with the interpolated poses, points3D from
    the TRIPS export)" the task brief asks for, which Brush's own COLMAP
    dataset loader (rust/brush-trips/crates/brush-dataset/src/formats/
    colmap.rs) reads unchanged: cameras.txt/images.txt for every view, and
    points3D.txt as its initial-splat point cloud (positions + colours
    only -- Brush's colmap loader does not read a size/opacity/rotation
    column from points3D.txt; it seeds means+SH-DC and initialises the
    rest itself).
Invariants: every CameraPose is written as a PINHOLE camera (the renders
    this model describes are always already-undistorted pinhole images,
    see trippy.distill.cameras' module docstring) grouped by its own
    (fx, fy, cx, cy, height, width), rounded to `DISTILL_CAMERA_KEY_
    DECIMALS` places so repeated K-matrix scaling roundoff never splits one
    logical camera into two COLMAP camera ids. Point cloud subsampling
    (`_subsample_points`) is seeded, so a re-run against the same source
    points reproduces the exact same points3D.txt.
Related docs: docs/EXPERIMENTS.md "Distillation (design B)"; trippy.scene.
    colmap_io (the generic text writers this module calls).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trippy.constants import (
    DISTILL_CAMERA_KEY_DECIMALS,
    DISTILL_CAMERA_MODEL,
    DISTILL_DEFAULT_MAX_INIT_POINTS,
    DISTILL_POINTS3D_DUMMY_ERROR,
)
from trippy.distill.cameras import DistillCameraPlan, image_filename, rotmat_to_qvec
from trippy.render.dolly import CameraPose
from trippy.scene import colmap_io
from trippy.scene.colmap_io import Camera, ColmapScene, Image, Point3D


def _camera_key(K: np.ndarray, image_hw: tuple[int, int]) -> tuple[float, float, float, float, int, int]:
    """Grouping key for `_assign_cameras`: rounded (fx, fy, cx, cy, height, width)."""
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    return (
        round(fx, DISTILL_CAMERA_KEY_DECIMALS),
        round(fy, DISTILL_CAMERA_KEY_DECIMALS),
        round(cx, DISTILL_CAMERA_KEY_DECIMALS),
        round(cy, DISTILL_CAMERA_KEY_DECIMALS),
        int(image_hw[0]),
        int(image_hw[1]),
    )


def _assign_cameras(poses: list[CameraPose]) -> tuple[dict[str, int], dict[int, Camera]]:
    """Group `poses` into COLMAP PINHOLE cameras by (fx, fy, cx, cy, height, width).

    Returns:
        (pose_name -> camera_id, camera_id -> Camera), camera_id assigned
        in first-seen order starting at 1.
    """
    key_to_id: dict[tuple, int] = {}
    cameras: dict[int, Camera] = {}
    pose_to_camera_id: dict[str, int] = {}

    for pose in poses:
        key = _camera_key(pose.K, pose.image_hw)
        camera_id = key_to_id.get(key)
        if camera_id is None:
            camera_id = len(key_to_id) + 1
            key_to_id[key] = camera_id
            fx, fy, cx, cy, height, width = key
            cameras[camera_id] = Camera(model=DISTILL_CAMERA_MODEL, width=width, height=height, params=[fx, fy, cx, cy])
        pose_to_camera_id[pose.name] = camera_id

    return pose_to_camera_id, cameras


def _subsample_points(xyz: np.ndarray, rgb: np.ndarray, max_points: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random (seeded) subsample of `xyz`/`rgb` down to `max_points` rows, sorted by index.

    `None` or an already-small cloud is returned unchanged. Sorting the
    sampled indices makes the output (and therefore point3D_id assignment)
    deterministic given a fixed seed, independent of `np.random.Generator`
    implementation details beyond which indices it picks.
    """
    n = xyz.shape[0]
    if max_points is None or n <= max_points:
        return xyz, rgb
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_points, replace=False))
    return xyz[idx], rgb[idx]


@dataclass(frozen=True)
class ColmapWriteSummary:
    """What `write_distill_colmap_model` wrote, for the caller's own report/README.

    Attributes:
        n_cameras: distinct COLMAP cameras written.
        n_images: total images (anchors + interpolated) written.
        n_anchor_images, n_interpolated_images: the same split as
            `DistillCameraPlan.anchors`/`.interpolated`.
        n_points_source: point count before subsampling.
        n_points_written: point count actually written to points3D.txt.
    """

    n_cameras: int
    n_images: int
    n_anchor_images: int
    n_interpolated_images: int
    n_points_source: int
    n_points_written: int


def write_distill_colmap_model(
    sparse_dir: str | Path,
    camera_plan: DistillCameraPlan,
    xyz: np.ndarray,
    rgb: np.ndarray,
    max_init_points: int | None = DISTILL_DEFAULT_MAX_INIT_POINTS,
    seed: int = 0,
) -> ColmapWriteSummary:
    """Write `camera_plan`'s cameras/images and a (possibly subsampled) point cloud as COLMAP text.

    Args:
        sparse_dir: output directory (created if missing); receives
            cameras.txt/images.txt/points3D.txt.
        camera_plan: as built by `trippy.distill.cameras.build_distill_camera_plan`.
        xyz: (N, 3) float, world-frame point positions (the TRIPS export's
            own point cloud -- trippy.distill.render_set passes the
            checkpoint's own trained `point_params.xyz`).
        rgb: (N, 3) float, linear colour in [0, 1] per channel.
        max_init_points: cap on points3D.txt row count (see
            `DISTILL_DEFAULT_MAX_INIT_POINTS`'s docstring); None writes
            every point.
        seed: subsample RNG seed (reproducible re-runs against the same
            source cloud).

    Returns:
        A `ColmapWriteSummary`.

    Raises:
        ValueError: `camera_plan` has no poses at all.
    """
    poses = camera_plan.all_poses
    if not poses:
        raise ValueError("camera_plan has no poses (need at least one registered image)")

    pose_to_camera_id, cameras = _assign_cameras(poses)

    images: dict[int, Image] = {}
    for image_id, pose in enumerate(poses, start=1):
        images[image_id] = Image(
            name=image_filename(pose.name),
            camera_id=pose_to_camera_id[pose.name],
            qvec=rotmat_to_qvec(pose.R),
            tvec=np.asarray(pose.t, dtype=np.float64),
            xys=np.zeros((0, 2), dtype=np.float64),
            point3D_ids=np.zeros((0,), dtype=np.int64),
        )

    xyz = np.asarray(xyz, dtype=np.float64)
    rgb_u8 = np.clip(np.asarray(rgb, dtype=np.float64) * 255.0, 0.0, 255.0).round().astype(np.uint8)
    n_points_source = xyz.shape[0]
    xyz_sub, rgb_sub = _subsample_points(xyz, rgb_u8, max_init_points, seed)

    points3D: dict[int, Point3D] = {
        point3d_id: Point3D(xyz=xyz_sub[i], rgb=rgb_sub[i], error=DISTILL_POINTS3D_DUMMY_ERROR, track=[])
        for i, point3d_id in enumerate(range(1, xyz_sub.shape[0] + 1))
    }

    colmap_io.save_colmap_model_txt(sparse_dir, ColmapScene(cameras=cameras, images=images, points3D=points3D))

    return ColmapWriteSummary(
        n_cameras=len(cameras),
        n_images=len(images),
        n_anchor_images=len(camera_plan.anchors),
        n_interpolated_images=len(camera_plan.interpolated),
        n_points_source=n_points_source,
        n_points_written=xyz_sub.shape[0],
    )
