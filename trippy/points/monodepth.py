"""PointSource for monocular-depth-derived points (D4 point source 2).

Module: trippy.points.monodepth
Invariants: numpy only (see trippy.points.source's module invariant) --
    `build()` calls `trippy.points.depth_io.undistort_and_cache`, which
    does use torch internally for one CPU-only `grid_sample` undistortion
    pass, but every torch tensor is converted back to numpy before this
    module ever sees it (`np.load`/`np.array`), and nothing here ever
    touches MPS.
Data flow (see docs/SPEC.md D4 point source 2):
    1. Undistort each requested image once (trippy.points.depth_io,
       reusing trippy.scene.dataset's cache format) so DepthPro's depth
       map and this module's pinhole K describe the exact same camera --
       no separate "undistort the pixel coordinates" step is needed.
    2. Read DepthPro's per-image metric depth + valid mask
       (trippy.points.depth_io.read_depth_output; DepthPro run out of
       band, only ever via scripts/gpu_submit.sh -- never here).
    3. Per-image scale alignment: reproject that image's observed COLMAP
       sparse points (Image.point3D_ids) through the SAME pinhole camera
       used for DepthPro's input, sample the predicted depth at each
       landing pixel, and take s = median(z_colmap / d_pred) (a robust,
       single-scalar-per-frame correction; MAD reported alongside it as
       the spread). Frames with too few matches
       (< MONODEPTH_MIN_SCALE_MATCHES) are dropped, not silently trusted.
    4. Backproject every `stride`-th valid, scaled-depth pixel to world
       frame via `x_w = R^T (x_c - t)` -- the analytic inverse of
       `trippy.geom.xform_a.world_to_cam`/`trippy.geom.xform_b.world_to_cam`
       (both independently cross-checked against this formula in
       tests/test_points_monodepth.py, per AGENTS.md's "implement
       transforms twice" rule -- xform_a/xform_b themselves are frozen
       per this task's file list, so the two independent unprojection
       routes live in the test file, not here).
    5. Concatenate all images' points and voxel-dedupe (reusing
       trippy.points.union's private `_voxel_dedupe_keep_highest_conf`
       helper rather than re-implementing it).
Related docs: docs/SPEC.md D4 (pluggable point sources); docs/GEOMETRY.md
    (COLMAP world frame, pixel-centre convention); trippy.points.depth_io
    (DepthPro manifest/output I/O); trippy.points.union (voxel dedupe).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trippy.constants import (
    MONODEPTH_DEFAULT_CONF0,
    MONODEPTH_DEFAULT_STRIDE,
    MONODEPTH_DEFAULT_VOXEL,
    MONODEPTH_MIN_SCALE_MATCHES,
    MONODEPTH_SCALE_MODE_MEDIAN_RATIO,
    PROVENANCE_MONODEPTH,
)
from trippy.geom import xform_a
from trippy.points import depth_io
from trippy.points.source import PointSet, PointSource
from trippy.points.union import _voxel_dedupe_keep_highest_conf
from trippy.scene import colmap_io
from trippy.scene.dataset import resolve_sparse_dir


class MonoDepthSource(PointSource):
    """Monocular-depth (Apple DepthPro) points, scale-aligned to sparse COLMAP depth.

    Args:
        scene_root: scene directory with `images/` and `sparse/0` or
            `sparse_txt`.
        image_names: registered image filenames to backproject (order is
            preserved in `describe()`'s per-image stats).
        width: undistorted pinhole image width in pixels -- must match the
            width the DepthPro depth maps under `depth_dir` were computed
            at (see `trippy.points.depth_io`'s module docstring).
        depth_dir: directory of DepthPro outputs
            (`<id>_depth.npy`/`_mask.npy`/`_meta.json`, `id` = image stem),
            already computed via `scripts/gpu_submit.sh` (never run here).
        cache_root: root of the undistortion cache (see
            `trippy.points.depth_io.cache_dir_for`); built/reused on
            `build()` if not already present.
        stride: backproject every `stride`-th valid pixel in both row and
            column (density scales as ~1/stride^2).
        voxel: world-unit voxel edge length for this source's own dedupe
            pass over its backprojected points (see
            `trippy.points.union._voxel_dedupe_keep_highest_conf`); `None`
            disables dedupe.
        conf0: fixed per-point confidence assigned to every backprojected
            point (see MONODEPTH_DEFAULT_CONF0's constant comment).
        scale_mode: only MONODEPTH_SCALE_MODE_MEDIAN_RATIO ("median_ratio")
            is implemented.
    """

    def __init__(
        self,
        scene_root: str | Path,
        image_names: list[str],
        width: int,
        depth_dir: str | Path,
        cache_root: str | Path,
        stride: int = MONODEPTH_DEFAULT_STRIDE,
        voxel: float | None = MONODEPTH_DEFAULT_VOXEL,
        conf0: float = MONODEPTH_DEFAULT_CONF0,
        scale_mode: str = MONODEPTH_SCALE_MODE_MEDIAN_RATIO,
    ) -> None:
        if scale_mode != MONODEPTH_SCALE_MODE_MEDIAN_RATIO:
            raise ValueError(f"unsupported scale_mode {scale_mode!r}; only 'median_ratio' is implemented")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if voxel is not None and voxel <= 0:
            raise ValueError(f"voxel must be positive, got {voxel}")

        self.scene_root = Path(scene_root)
        self.image_names = list(image_names)
        self.width = int(width)
        self.depth_dir = Path(depth_dir)
        self.cache_root = Path(cache_root)
        self.stride = int(stride)
        self.voxel = voxel
        self.conf0 = float(conf0)
        self.scale_mode = scale_mode
        self._per_image_stats: list[dict] = []

    def describe(self) -> dict:
        return {
            "type": "MonoDepthSource",
            "scene_root": str(self.scene_root),
            "image_names": list(self.image_names),
            "width": self.width,
            "depth_dir": str(self.depth_dir),
            "cache_root": str(self.cache_root),
            "stride": self.stride,
            "voxel": self.voxel,
            "conf0": self.conf0,
            "scale_mode": self.scale_mode,
            "per_image": list(self._per_image_stats),
        }

    def build(self) -> PointSet:
        sparse_dir = resolve_sparse_dir(self.scene_root)
        scene = colmap_io.load_colmap_model(sparse_dir)
        images_by_name = scene.images_by_name()

        cache_dir = depth_io.undistort_and_cache(
            self.scene_root, self.image_names, self.width, self.cache_root
        )
        cache_meta = depth_io.load_cache_meta(cache_dir)

        xyz_parts: list[np.ndarray] = []
        size_parts: list[np.ndarray] = []
        rgb_parts: list[np.ndarray] = []
        conf_parts: list[np.ndarray] = []
        prov_parts: list[np.ndarray] = []
        stats: list[dict] = []

        for name in self.image_names:
            im = images_by_name.get(name)
            if im is None:
                raise KeyError(f"{name!r} is not a registered image under {sparse_dir}")

            image_meta = cache_meta["images"][name]
            K = np.asarray(image_meta["K"], dtype=np.float64)
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            width_i, height_i = int(image_meta["width"]), int(image_meta["height"])
            R = xform_a.qvec2R(np.asarray(image_meta["qvec"], dtype=np.float64))
            t = np.asarray(image_meta["tvec"], dtype=np.float64)

            depth_out = depth_io.read_depth_output(self.depth_dir, Path(name).stem)
            if (depth_out.meta["width"], depth_out.meta["height"]) != (width_i, height_i):
                raise ValueError(
                    f"{name}: DepthPro was run at {depth_out.meta['width']}x{depth_out.meta['height']} "
                    f"but the undistortion cache is {width_i}x{height_i} -- resolution mismatch "
                    "(see trippy.points.depth_io module docstring: DepthPro's own WORK_MAX_SIDE cap "
                    "may have silently downscaled the input)"
                )
            depth_pred = depth_out.depth
            valid_mask = depth_out.mask & (depth_pred > 0)

            s, mad, n_matches = _median_ratio_scale(scene, im, R, t, fx, fy, cx, cy, width_i, height_i, depth_pred, valid_mask)
            valid_fraction = float(valid_mask.mean())

            if n_matches < MONODEPTH_MIN_SCALE_MATCHES:
                stats.append(
                    {
                        "image": name,
                        "scale": None,
                        "mad": None,
                        "n_matches": n_matches,
                        "valid_fraction": valid_fraction,
                        "points_contributed": 0,
                        "skipped_reason": f"only {n_matches} sparse matches (< {MONODEPTH_MIN_SCALE_MATCHES})",
                    }
                )
                continue

            depth_scaled = depth_pred * s
            rgb = depth_io.load_cached_rgb(cache_dir, name)

            rows = np.arange(0, height_i, self.stride)
            cols = np.arange(0, width_i, self.stride)
            rr, cc = np.meshgrid(rows, cols, indexing="ij")
            rr = rr.ravel()
            cc = cc.ravel()
            keep = valid_mask[rr, cc] & (depth_scaled[rr, cc] > 0)
            rr, cc = rr[keep], cc[keep]
            z = depth_scaled[rr, cc].astype(np.float64)

            xyz_w = _unproject_numpy(rr, cc, z, R, t, fx, fy, cx, cy)
            colour = rgb[rr, cc].astype(np.float32) / 255.0
            size0 = (z * self.stride / fx).astype(np.float32)
            n_pts = xyz_w.shape[0]

            xyz_parts.append(xyz_w.astype(np.float32))
            size_parts.append(size0)
            rgb_parts.append(colour)
            conf_parts.append(np.full(n_pts, self.conf0, dtype=np.float32))
            prov_parts.append(np.full(n_pts, PROVENANCE_MONODEPTH, dtype=np.uint8))

            stats.append(
                {
                    "image": name,
                    "scale": s,
                    "mad": mad,
                    "n_matches": n_matches,
                    "valid_fraction": valid_fraction,
                    "points_contributed": int(n_pts),
                }
            )

        self._per_image_stats = stats

        if not xyz_parts:
            return PointSet(
                xyz=np.zeros((0, 3), dtype=np.float32),
                size0=np.zeros(0, dtype=np.float32),
                rgb0=np.zeros((0, 3), dtype=np.float32),
                conf0=np.zeros(0, dtype=np.float32),
                provenance=np.zeros(0, dtype=np.uint8),
            )

        xyz = np.concatenate(xyz_parts, axis=0)
        size0 = np.concatenate(size_parts, axis=0)
        rgb0 = np.concatenate(rgb_parts, axis=0)
        conf0 = np.concatenate(conf_parts, axis=0)
        provenance = np.concatenate(prov_parts, axis=0)

        if self.voxel is not None:
            keep_idx = _voxel_dedupe_keep_highest_conf(xyz, conf0, self.voxel)
            xyz, size0, rgb0, conf0, provenance = (
                xyz[keep_idx],
                size0[keep_idx],
                rgb0[keep_idx],
                conf0[keep_idx],
                provenance[keep_idx],
            )

        return PointSet(xyz=xyz, size0=size0, rgb0=rgb0, conf0=conf0, provenance=provenance)


def _median_ratio_scale(
    scene: colmap_io.ColmapScene,
    im: colmap_io.Image,
    R: np.ndarray,
    t: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width_i: int,
    height_i: int,
    depth_pred: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float, float, int]:
    """Median ratio z_colmap/d_pred (+ MAD, + match count) for one image.

    Reprojects `im`'s observed sparse COLMAP points through the SAME
    pinhole camera (R, t, fx, fy, cx, cy) DepthPro's input was undistorted
    to, rather than trusting `im.xys` (recorded in the original *distorted*
    image) -- this way no separate "undistort these 2D keypoints" step is
    needed.

    Returns:
        (scale, mad, n_matches). scale/mad are 1.0/0.0 (unused; check
        n_matches < MONODEPTH_MIN_SCALE_MATCHES before trusting them)
        when there are no matches at all.
    """
    obs_ids = np.unique(im.point3D_ids[im.point3D_ids >= 0])
    xyz_w_obs = np.array(
        [scene.points3D[int(pid)].xyz for pid in obs_ids if int(pid) in scene.points3D], dtype=np.float64
    )
    if xyz_w_obs.shape[0] == 0:
        return 1.0, 0.0, 0

    xyz_c_obs = xform_a.world_to_cam(R, t, xyz_w_obs)
    uv_obs, depth_obs = xform_a.project_pinhole(xyz_c_obs, fx, fy, cx, cy)
    col_obs = np.floor(uv_obs[:, 0]).astype(np.int64)
    row_obs = np.floor(uv_obs[:, 1]).astype(np.int64)

    in_bounds = (
        (depth_obs > 0) & (col_obs >= 0) & (col_obs < width_i) & (row_obs >= 0) & (row_obs < height_i)
    )
    col_ib, row_ib, depth_ib = col_obs[in_bounds], row_obs[in_bounds], depth_obs[in_bounds]
    if col_ib.shape[0] == 0:
        return 1.0, 0.0, 0

    is_valid = valid_mask[row_ib, col_ib]
    col_v, row_v, depth_v = col_ib[is_valid], row_ib[is_valid], depth_ib[is_valid]
    if col_v.shape[0] == 0:
        return 1.0, 0.0, 0

    d_pred_v = depth_pred[row_v, col_v]
    ok = d_pred_v > 0
    ratios = depth_v[ok] / d_pred_v[ok]
    n_matches = int(ratios.shape[0])
    if n_matches == 0:
        return 1.0, 0.0, 0

    s = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - s)))
    return s, mad, n_matches


def _unproject_numpy(
    row: np.ndarray,
    col: np.ndarray,
    z: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Backproject pixel-centre (col+0.5, row+0.5) at depth z to world frame.

    Camera-to-world: x_w = R^T (x_c - t); in this module's row-vector
    convention (matching `trippy.geom.xform_a.world_to_cam`'s
    `x_c = x_w @ R.T + t`), that is `x_w = (x_c - t) @ R`.

    Pixel-centre convention matches docs/GEOMETRY.md / `undistort_maps`:
    pixel (row, col) spans continuous [col, col+1) x [row, row+1), centre
    (col+0.5, row+0.5).
    """
    x_c = (col.astype(np.float64) + 0.5 - cx) / fx * z
    y_c = (row.astype(np.float64) + 0.5 - cy) / fy * z
    xyz_c = np.stack([x_c, y_c, z], axis=1)
    return (xyz_c - t.reshape(1, 3)) @ R
