"""Off-path honesty poses: lateral + oblique views the training photos never captured.

Module: trippy.render.offpath
Purpose: for each of a set of registered images, build two camera poses that
    were never photographed -- a sideways step (`lateral`) and an elevated,
    downward-looking view (`oblique`) -- so a candidate can be judged on
    whether it hallucinates plausibly convincing (or obviously broken)
    content away from the training camera path (AGENTS.md "Honesty rule";
    docs/EXPERIMENTS.md "Mandatory honesty sheet").
Mirrors (read-only, never imported): ~/Splats/research/visual/
    render_offpath.py -- lines 107-110 (`up_vector`: mean of every
    registered image's world-frame "up" axis), 183-199 (per-image local
    depth, same construction as `trippy.render.dolly.local_depth_estimate`),
    202-205 (`side = normalize(cross(up, fwd))`), 228-230 (lateral pose),
    233-238 (oblique pose, looking at the scene centroid), 259-263
    (centroid = mean of every registered camera centre); and
    ~/Splats/tools/gsrender.py:133-137 (`look_at`). Reimplemented here in
    numpy against `trippy.scene.colmap_io` / `trippy.geom.xform_a` (same
    COLMAP convention, see xform_a's module docstring) instead of Splats'
    own COLMAP-txt parsers.
Related docs: docs/EXPERIMENTS.md "Mandatory honesty sheet"; docs/SPEC.md D10.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trippy.constants import (
    OFFPATH_DEFAULT_ELEVATE_FRAC,
    OFFPATH_DEFAULT_LATERAL_FRAC,
    OFFPATH_DEFAULT_WIDTH,
    OFFPATH_OBLIQUE_BACK_FRAC,
)
from trippy.geom import xform_a
from trippy.render.dolly import CameraPose, camera_center, local_depth_estimate, scaled_intrinsics
from trippy.scene import colmap_io
from trippy.scene.colmap_io import ColmapScene
from trippy.scene.dataset import resolve_sparse_dir


def up_vector(colmap_scene: ColmapScene) -> np.ndarray:
    """Mean world-frame "up" axis over every registered image in the scene.

    Mirrors render_offpath.py:107-110 exactly: each image's camera-space
    "up" is -y (COLMAP convention is y-down, see `trippy.geom.xform_a`'s
    module docstring), transformed to world with `R.T` and averaged, then
    renormalised.
    """
    ups = [xform_a.qvec2R(im.qvec).T @ np.array([0.0, -1.0, 0.0]) for im in colmap_scene.images.values()]
    mean_up = np.mean(ups, axis=0)
    return mean_up / np.linalg.norm(mean_up)


def look_at_pose(cam_pos: np.ndarray, target: np.ndarray, world_down: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World->camera (R, t) looking from `cam_pos` at `target`, `world_down` as the down reference.

    Mirrors ~/Splats/tools/gsrender.py:133-137 (`look_at`) exactly, returning
    trippy's (R, t) pair (x_cam = R @ x_world + t, see `trippy.geom.xform_a`)
    instead of a 4x4 matrix.

    Args:
        cam_pos: (3,) float64, world-frame camera centre.
        target: (3,) float64, world-frame point the camera looks at.
        world_down: (3,) float64, a world "down" reference (not necessarily
            orthogonal to the view direction -- only used to construct the
            camera's right axis via a cross product, exactly as gsrender.py
            does).

    Returns:
        (R, t): (3, 3) rotation, (3,) translation.
    """
    forward = target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    right = np.cross(world_down, forward)
    right = right / (np.linalg.norm(right) + 1e-9)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward], axis=0)
    t = -R @ cam_pos
    return R, t


def offpath_poses(
    scene_root: str | Path,
    image_names: list[str],
    lateral_frac: float = OFFPATH_DEFAULT_LATERAL_FRAC,
    elevate_frac: float = OFFPATH_DEFAULT_ELEVATE_FRAC,
    width: int = OFFPATH_DEFAULT_WIDTH,
) -> list[CameraPose]:
    """Build lateral + oblique off-path honesty poses for each of `image_names`.

    For every name in `image_names`, two poses nothing in the capture ever
    produced:
      - `<stem>_lateral`: step `lateral_frac * depth` sideways (perpendicular
        to both that image's forward direction and the scene's up vector),
        looking at the same point the on-path camera was looking at
        (render_offpath.py:228-230).
      - `<stem>_oblique`: rise `elevate_frac * depth` along the scene's up
        vector and pull back `OFFPATH_OBLIQUE_BACK_FRAC * depth`, looking
        down at the scene centroid -- the mean of every registered camera
        centre (render_offpath.py:233-238).

    Args:
        scene_root: COLMAP scene root (`images/` + `sparse/0` or `sparse_txt`).
        image_names: registered image names to build off-path pairs for.
        lateral_frac: fraction of local depth for the lateral step.
        elevate_frac: fraction of local depth for the oblique rise.
        width: destination pinhole image width in pixels (per image's own
            camera, aspect-preserved -- see `trippy.render.dolly.scaled_intrinsics`).

    Returns:
        `2 * len(image_names)` `CameraPose`s, lateral then oblique per name,
        in `image_names`'s order; each carries `image_name=name`.

    Raises:
        KeyError: any name in `image_names` is not a registered image.
    """
    sparse_dir = resolve_sparse_dir(Path(scene_root))
    colmap_scene = colmap_io.load_colmap_model(sparse_dir)
    images_by_name = colmap_scene.images_by_name()
    missing = sorted(n for n in set(image_names) if n not in images_by_name)
    if missing:
        raise KeyError(f"image_names not registered under {scene_root}: {missing}")

    up = up_vector(colmap_scene)
    down = -up
    centers = np.array(
        [camera_center(xform_a.qvec2R(im.qvec), im.tvec) for im in colmap_scene.images.values()]
    )
    centroid = centers.mean(axis=0)
    all_points = np.array([p.xyz for p in colmap_scene.points3D.values()], dtype=np.float64).reshape(-1, 3)

    poses: list[CameraPose] = []
    for name in image_names:
        im0 = images_by_name[name]
        cam = colmap_scene.cameras[im0.camera_id]
        K, height, out_width = scaled_intrinsics(cam, width)

        R0 = xform_a.qvec2R(im0.qvec)
        C0 = camera_center(R0, im0.tvec)
        fwd0 = R0.T @ np.array([0.0, 0.0, 1.0])
        fwd0 = fwd0 / np.linalg.norm(fwd0)
        depth = local_depth_estimate(all_points, C0, fwd0)

        side = np.cross(up, fwd0)
        if np.linalg.norm(side) < 1e-6:
            side = np.array([1.0, 0.0, 0.0])
        side = side / np.linalg.norm(side)

        target = C0 + fwd0 * depth
        stem = Path(name).stem
        image_hw = (height, out_width)

        cam_lateral = C0 + side * lateral_frac * depth
        r_lat, t_lat = look_at_pose(cam_lateral, target, down)
        poses.append(CameraPose(name=f"{stem}_lateral", R=r_lat, t=t_lat, K=K.copy(), image_hw=image_hw, image_name=name))

        cam_oblique = C0 + up * elevate_frac * depth - fwd0 * depth * OFFPATH_OBLIQUE_BACK_FRAC
        r_obl, t_obl = look_at_pose(cam_oblique, centroid, down)
        poses.append(CameraPose(name=f"{stem}_oblique", R=r_obl, t=t_obl, K=K.copy(), image_hw=image_hw, image_name=name))

    return poses
