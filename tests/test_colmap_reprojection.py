"""Reproject real COLMAP points3D observations and check xform_a/xform_b agree with reality.

Module: tests.test_colmap_reprojection
Invariants under test: both geometry implementations reproject a sparse
    point back near the pixel COLMAP originally observed it at, and depth
    is positive for every observation (a point can't be triangulated from
    behind the camera that saw it).
Skips cleanly when ~/Splats/scenes/karekare/kk-coherent/sparse_txt is
    absent (see tests.conftest.splats_scene). Must finish in well under the
    suite's 60s CPU budget -- points are subsampled per image to stay fast.
Related docs: docs/SPEC.md "Validation (CPU pytest, before any GPU
    job)"; trippy.geom.camera.intrinsics_from_colmap_params (distortion
    handling note).
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch

from trippy.geom import camera, xform_a, xform_b

MAX_IMAGES = 20
MAX_POINTS_PER_IMAGE = 300
MEDIAN_REPROJ_PX_TOL = 3.0
TEST_TIME_BUDGET_S = 20.0


def test_colmap_reprojection_both_implementations(splats_scene) -> None:
    t0 = time.time()
    cameras = xform_a.read_cameras_txt(str(splats_scene / "cameras.txt"))
    images = xform_a.read_images_txt(str(splats_scene / "images.txt"))
    points3d = xform_a.read_points3d_txt(str(splats_scene / "points3D.txt"))

    names = sorted(images.keys())[:MAX_IMAGES]
    assert names, "no images parsed from images.txt"

    errors_a: list[np.ndarray] = []
    errors_b: list[np.ndarray] = []
    rng = random.Random(0)

    for name in names:
        im = images[name]
        cam = cameras[im["camera_id"]]
        fx, fy, cx, cy = camera.intrinsics_from_colmap_params(cam["model"], cam["params"])

        obs = [(x, y, pid) for (x, y, pid) in im["points2d"] if pid != -1 and pid in points3d]
        if len(obs) > MAX_POINTS_PER_IMAGE:
            obs = rng.sample(obs, MAX_POINTS_PER_IMAGE)
        if not obs:
            continue

        uv_observed = np.array([[x, y] for x, y, _ in obs], dtype=np.float64)
        xyz_w = np.array([points3d[pid]["xyz"] for _, _, pid in obs], dtype=np.float64)

        R = xform_a.qvec2R(im["qvec"])
        t = im["tvec"]

        xyz_c_a = xform_a.world_to_cam(R, t, xyz_w)
        uv_a, depth_a = xform_a.project_pinhole(xyz_c_a, fx, fy, cx, cy)
        assert np.all(depth_a > 0), f"{name}: non-positive depth via xform_a"
        errors_a.append(np.linalg.norm(uv_a - uv_observed, axis=1))

        R_b = xform_b.qvec2R(torch.tensor(im["qvec"], dtype=torch.float64))
        xyz_c_b = xform_b.world_to_cam(
            R_b, torch.tensor(t, dtype=torch.float64), torch.tensor(xyz_w, dtype=torch.float64)
        )
        uv_b, depth_b = xform_b.project_pinhole(xyz_c_b, fx, fy, cx, cy)
        assert torch.all(depth_b > 0), f"{name}: non-positive depth via xform_b"
        errors_b.append(np.linalg.norm(uv_b.numpy() - uv_observed, axis=1))

    all_errors_a = np.concatenate(errors_a)
    all_errors_b = np.concatenate(errors_b)

    # OPENCV/SIMPLE_RADIAL cameras carry real lens distortion that we
    # deliberately ignore here (fx,fy,cx,cy only -- see
    # trippy.geom.camera.intrinsics_from_colmap_params), hence a 3px
    # tolerance on the *median* rather than a sub-pixel bound.
    assert np.median(all_errors_a) < MEDIAN_REPROJ_PX_TOL, f"xform_a median error {np.median(all_errors_a):.2f}px"
    assert np.median(all_errors_b) < MEDIAN_REPROJ_PX_TOL, f"xform_b median error {np.median(all_errors_b):.2f}px"

    elapsed = time.time() - t0
    assert elapsed < TEST_TIME_BUDGET_S, f"test took {elapsed:.1f}s, budget is {TEST_TIME_BUDGET_S}s"
