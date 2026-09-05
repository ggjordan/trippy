"""Shade dolly camera path: fixed orientation, camera centre slides along its own forward ray.

Module: trippy.render.dolly
Purpose: build the camera path docs/EXPERIMENTS.md's "Dolly camera paths"
    section describes -- a `pose_name` image's real COLMAP orientation is
    frozen, and the camera centre slides along that pose's own forward ray
    from `t_range[0]` to `t_range[1]` times the scene's local depth at that
    point. Stepping a camera through the shade region this way is the test
    for whether a trained model renders the shade as a lighting effect
    (shading that changes gracefully as the camera moves through it) or as
    an object (a cloud of points the camera has to pass through).
Mirrors (read-only, never imported): ~/Splats/tools/depthprior_shade_dolly.py
    -- lines 44-51 (pose/camera lookup), 52-54 (forward vector), 56-63
    (local-depth estimate: median distance to sparse points in front of the
    camera, 5th-95th percentile trimmed, `DOLLY_DEPTH_INFRONT_MIN_COUNT`/
    `DOLLY_FALLBACK_DEPTH` fallbacks), 71 (t linspace), 78-81 (camera centre
    slide + view-matrix construction: `t = -R @ C`). Reimplemented here in
    numpy against trippy's own `trippy.scene.colmap_io` / `trippy.geom.
    xform_a` (identical COLMAP convention, see xform_a's module docstring)
    instead of Splats' `render_offpath.read_cameras/read_images/read_points/
    qvec2R/cam_center/intrinsics_for` helpers.
Related docs: docs/EXPERIMENTS.md "Dolly camera paths"; docs/SPEC.md D10.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trippy.constants import (
    DOLLY_DEFAULT_N_FRAMES,
    DOLLY_DEFAULT_POSE_NAME,
    DOLLY_DEFAULT_T_END,
    DOLLY_DEFAULT_T_START,
    DOLLY_DEFAULT_WIDTH,
    DOLLY_DEPTH_INFRONT_MIN_COUNT,
    DOLLY_DEPTH_PERCENTILE_HIGH,
    DOLLY_DEPTH_PERCENTILE_LOW,
    DOLLY_FALLBACK_DEPTH,
)
from trippy.geom import xform_a
from trippy.scene import colmap_io
from trippy.scene.dataset import resolve_sparse_dir


@dataclass(frozen=True)
class CameraPose:
    """One synthetic render pose: a world->camera pose plus intrinsics and image size.

    Shares the field names of the pose-dict shape `trippy.train.eval.
    render_offpath` already consumes (`{"name", "R", "t", "K", "image_hw"}`),
    plus `image_name` so a renderer can look up that source image's own
    trained exposure/white-balance (see `trippy.render.candidate`).

    Attributes:
        name: pose label, e.g. "IMG_3830_dolly_t+0.35" or "IMG_3828_lateral".
        R: (3, 3) float64, world->camera rotation (x_cam = R @ x_world + t).
        t: (3,) float64, world->camera translation.
        K: (3, 3) float64, pinhole intrinsics in pixels.
        image_hw: (H, W) pixels to render this pose at.
        image_name: the registered COLMAP image this pose is anchored to
            (its orientation, for a dolly frame; its own on-path view, for
            an off-path pair), or None if the pose has no natural source
            image.
    """

    name: str
    R: np.ndarray
    t: np.ndarray
    K: np.ndarray
    image_hw: tuple[int, int]
    image_name: str | None = None


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World-frame camera centre from a world->camera pose (x_cam = R @ x_world + t)."""
    return -R.T @ t


def scaled_intrinsics(cam: colmap_io.Camera, width: int) -> tuple[np.ndarray, int, int]:
    """Distortion-free pinhole K + (height, width) for `cam`, scaled to `width`.

    Mirrors ~/Splats/research/visual/render_offpath.py:113-119
    (`intrinsics_for`), reimplemented against `trippy.scene.colmap_io`'s own
    distortion-free intrinsics reader instead of re-parsing `cameras.txt`.

    Args:
        cam: the source camera (native resolution, from a loaded ColmapScene).
        width: destination pinhole image width in pixels.

    Returns:
        (K, height, width): K is (3, 3) float64; height keeps `cam`'s
        aspect ratio (rounded, forced even -- matching `intrinsics_for`'s
        `H -= H % 2`, since most video/H.264 pipelines want even dimensions).
    """
    fx, fy, cx, cy = colmap_io.intrinsics(cam)
    scale = width / cam.width
    height = round(cam.height * scale)
    height -= height % 2
    K = np.array(
        [[fx * scale, 0.0, cx * scale], [0.0, fy * scale, cy * scale], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K, height, width


def local_depth_estimate(points_xyz: np.ndarray, center: np.ndarray, forward: np.ndarray) -> float:
    """Median distance from `center` to `points_xyz`, restricted to points in front and trimmed.

    Mirrors ~/Splats/tools/depthprior_shade_dolly.py:56-63 exactly: falls
    back to the full (not-in-front-filtered) distance set when fewer than
    `DOLLY_DEPTH_INFRONT_MIN_COUNT` points are in front of the camera, trims
    to the [`DOLLY_DEPTH_PERCENTILE_LOW`, `DOLLY_DEPTH_PERCENTILE_HIGH`]
    percentile range, and falls back further to `DOLLY_FALLBACK_DEPTH` if no
    points are available at all (the Splats script would raise on an empty
    points3D.txt; this is a deliberately more defensive port so the
    generator stays usable on a minimal synthetic test scene).

    Args:
        points_xyz: (N, 3) float64, world-frame sparse points (every point
            in the scene's `points3D.txt`, not filtered to any one image's
            observations -- see module docstring).
        center: (3,) float64, the camera centre the distances are measured from.
        forward: (3,) float64, unit forward vector (camera-space +z, in world frame).

    Returns:
        Local scene depth, world units.
    """
    if points_xyz.shape[0] == 0:
        return DOLLY_FALLBACK_DEPTH
    rel = points_xyz - center
    dist = np.linalg.norm(rel, axis=1)
    infront = (rel @ forward) > 0
    dsel = dist[infront] if infront.sum() > DOLLY_DEPTH_INFRONT_MIN_COUNT else dist
    if dsel.size == 0:
        return DOLLY_FALLBACK_DEPTH
    lo, hi = np.percentile(dsel, [DOLLY_DEPTH_PERCENTILE_LOW, DOLLY_DEPTH_PERCENTILE_HIGH])
    trimmed = dsel[(dsel >= lo) & (dsel <= hi)]
    return float(np.median(trimmed)) if trimmed.size else float(np.median(dsel))


def shade_dolly_poses(
    scene_root: str | Path,
    pose_name: str = DOLLY_DEFAULT_POSE_NAME,
    t_range: tuple[float, float] = (DOLLY_DEFAULT_T_START, DOLLY_DEFAULT_T_END),
    n: int = DOLLY_DEFAULT_N_FRAMES,
    width: int = DOLLY_DEFAULT_WIDTH,
) -> list[CameraPose]:
    """Build the shade dolly camera path.

    `pose_name`'s own COLMAP rotation is frozen for every frame; the camera
    centre slides along that pose's forward ray (`C0 + fwd0 * depth * t`)
    for `n` values of `t` linearly spaced over `t_range`, where `depth` is
    the scene's local-depth estimate at that pose (`local_depth_estimate`).
    See the module docstring for the exact Splats source lines mirrored.

    Args:
        scene_root: COLMAP scene root (`images/` + `sparse/0` or `sparse_txt`).
        pose_name: registered image name whose pose/orientation is frozen
            (default matches docs/EXPERIMENTS.md "Dolly camera paths").
        t_range: (t_start, t_end), fraction of local depth to slide the
            camera centre by.
        n: number of frames, linearly spaced over `t_range` (inclusive).
        width: destination pinhole image width in pixels (see
            `scaled_intrinsics`); height keeps the source camera's aspect ratio.

    Returns:
        `n` `CameraPose`s, orientation identical across all of them, named
        `f"{Path(pose_name).stem}_dolly_t{t:+.2f}"` and each carrying
        `image_name=pose_name`.

    Raises:
        KeyError: `pose_name` is not a registered image under `scene_root`.
    """
    sparse_dir = resolve_sparse_dir(Path(scene_root))
    colmap_scene = colmap_io.load_colmap_model(sparse_dir)
    images_by_name = colmap_scene.images_by_name()
    if pose_name not in images_by_name:
        raise KeyError(f"pose_name {pose_name!r} is not a registered image under {scene_root}")

    im0 = images_by_name[pose_name]
    cam = colmap_scene.cameras[im0.camera_id]
    K, height, out_width = scaled_intrinsics(cam, width)

    R0 = xform_a.qvec2R(im0.qvec)
    C0 = camera_center(R0, im0.tvec)
    fwd0 = R0.T @ np.array([0.0, 0.0, 1.0])
    fwd0 = fwd0 / np.linalg.norm(fwd0)

    all_points = np.array([p.xyz for p in colmap_scene.points3D.values()], dtype=np.float64).reshape(-1, 3)
    depth = local_depth_estimate(all_points, C0, fwd0)

    t_start, t_end = t_range
    ts = np.linspace(t_start, t_end, n) if n > 1 else np.array([t_start], dtype=np.float64)
    stem = Path(pose_name).stem

    poses: list[CameraPose] = []
    for t in ts:
        c_pos = C0 + fwd0 * depth * float(t)
        t_cam = -R0 @ c_pos
        poses.append(
            CameraPose(
                name=f"{stem}_dolly_t{float(t):+.2f}",
                R=R0.copy(),
                t=t_cam,
                K=K.copy(),
                image_hw=(height, out_width),
                image_name=pose_name,
            )
        )
    return poses
