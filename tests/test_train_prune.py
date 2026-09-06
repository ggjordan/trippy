"""Tests for trippy.train.prune + the Trainer's point-removal surgery.

Module: tests.test_train_prune
Invariants under test:
  - `PointRemovalConfig.fires_at` reproduces TRIPS's schedule
    (`src/apps/train.cpp:533-538`: the start epoch, then every
    `every_epochs` after it) and is a hard no-op when disabled.
  - `removal_keep_mask` implements TRIPS's rule literally: drop exactly the
    points whose effective confidence is `< conf_threshold`
    (`src/apps/train.cpp:846-851`).
  - `min_points` is respected by both rules, deterministically (the
    highest-confidence points survive).
  - `build_shade_region` / `in_region` match a brute-force, per-point,
    per-view reimplementation of `~/Splats/tools/depthprior_shade_audit.py`'s
    region definition written independently in this file.
  - `dark_mass_stats` computed in-process from PointParams equals the same
    statistic computed from the *exported* PLY's own fields (the exporter's
    opacity/f_dc mapping inverted the way the audit inverts it).
  - `Trainer._apply_keep_mask` drops the right points AND keeps the Adam
    moments aligned: after a removal, another `train_step()` runs with no
    shape error, and the surviving moments equal the pre-removal moments
    row-for-row.
  - a checkpoint written after a removal can be loaded back into a freshly
    constructed Trainer (whose point source still has every point).
  - `Trainer.evaluate` records the `points` block (count + in-region
    dark-mass fraction) in metrics.json and metrics.jsonl.
All fixtures are synthetic (tests/test_train_helpers.py plus the tiny
COLMAP model with observations built here); no real scene is ever read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image as PILImage
from test_train_helpers import (
    CX,
    CY,
    FX,
    FY,
    IMG_HEIGHT,
    IMG_WIDTH,
    build_synthetic_ply,
    render_reference_image,
    synthetic_point_set,
    tiny_train_config,
)

from trippy.constants import SH_C0
from trippy.points.gaussian_ply import GaussianPlySource
from trippy.train import prune
from trippy.train.prune_config import PointRemovalConfig, ShadePruneConfig
from trippy.train.trainer import Trainer

# The synthetic scene's cameras all look down +z from near the origin (see
# test_train_helpers.camera_pose) and its cloud sits at z in [4, 8]. The audit's
# slab is [0.05*d, 0.5*d] where d is a frame's own median observed sparse depth,
# so the sparse points this fixture triangulates are placed at SPARSE_DEPTH_SCALE
# times the cloud's depth: d ~ 12, slab ~ [0.6, 6], which cuts the cloud in half
# instead of missing it entirely (d ~ 6 would put the whole far plane in front of
# every point) or swallowing it whole. Real shade frames see geometry both nearer
# and further than the floating mass being audited; this reproduces that shape.
N_SCENE_IMAGES = 4
OBSERVED_STRIDE = 3  # every third point is "observed" by each frame
SPARSE_DEPTH_SCALE = 2.0
#: The synthetic scene's first two frames, used as stand-in "shade frames".
FIRST_TWO_FRAMES = ["IMG_0.jpg", "IMG_1.jpg"]


def build_scene_with_observations(tmp_path: Path, n_images: int = N_SCENE_IMAGES, seed: int = 0):
    """A synthetic COLMAP scene whose images.txt carries real observations + points3D.

    `tests/test_train_helpers.build_synthetic_scene` writes zero
    observations and an empty points3D.txt (nothing that used it needed
    them), but the shade region is *defined* by each frame's own observed
    sparse points, so this builds the same scene with a triangulated track.

    Returns:
        (scene_root, point_set, names).
    """
    point_set = synthetic_point_set(seed=seed)
    # `synthetic_point_set` gives every point conf0 = 0.95, which no confidence
    # threshold below 0.95 can ever select; spread it over the range a real
    # Gaussian PLY's opacity covers so the removal rules are actually exercised.
    point_set.conf0[:] = np.random.default_rng(seed + 7).uniform(0.05, 0.99, len(point_set.conf0)).astype(np.float32)
    scene_root = tmp_path / "scene"
    images_dir = scene_root / "images"
    images_dir.mkdir(parents=True)
    names = [f"IMG_{i}.jpg" for i in range(n_images)]
    for i, name in enumerate(names):
        PILImage.fromarray(render_reference_image(point_set, i), mode="RGB").save(images_dir / name)

    sparse_dir = scene_root / "sparse_txt"
    sparse_dir.mkdir(parents=True)
    (sparse_dir / "cameras.txt").write_text(f"1 PINHOLE {IMG_WIDTH} {IMG_HEIGHT} {FX} {FY} {CX} {CY}\n")

    observed = np.arange(0, len(point_set.xyz), OBSERVED_STRIDE)
    image_lines = []
    for i, name in enumerate(names, start=1):
        tx = -(i - 1) * 0.3
        image_lines.append(f"{i} 1.0 0.0 0.0 0.0 {tx} 0.0 0.0 1 {name}")
        # points2D triples: (x, y, point3D_id); the pixel coords are never read by the
        # region code (only the ids are), so a constant placeholder is fine.
        image_lines.append(" ".join(f"1.0 1.0 {pid}" for pid in observed))
    (sparse_dir / "images.txt").write_text("\n".join(image_lines) + "\n")

    point_lines = []
    for pid in observed:
        x, y, z = point_set.xyz[pid] * SPARSE_DEPTH_SCALE
        point_lines.append(f"{pid} {x} {y} {z} 128 128 128 0.5 1 0")
    (sparse_dir / "points3D.txt").write_text("\n".join(point_lines) + "\n")
    return scene_root, point_set, names


def brute_force_region(sparse_dir: Path, frames: list[str], znear_frac: float, zfar_frac: float, xyz: np.ndarray):
    """Independent reimplementation of depthprior_shade_audit.py's region, from the .txt files.

    Deliberately written from the text files with its own quaternion and
    projection code (no `trippy.geom`, no `colmap_io`), point by point and
    view by view, so agreeing with `trippy.train.prune` is evidence about
    the port rather than about a shared helper.
    """
    cam_line = (sparse_dir / "cameras.txt").read_text().split("\n")[0].split()
    width, height = int(cam_line[2]), int(cam_line[3])
    fx, fy, cx, cy = (float(v) for v in cam_line[4:8])

    poses: dict[str, tuple[np.ndarray, np.ndarray, list[int]]] = {}
    lines = [line for line in (sparse_dir / "images.txt").read_text().split("\n") if line.strip()]
    for header, obs in zip(lines[0::2], lines[1::2], strict=True):
        parts = header.split()
        name = parts[9]
        qvec = np.array([float(v) for v in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
        ids = [int(v) for v in obs.split()[2::3]]
        poses[name] = (qvec, tvec, ids)

    points: dict[int, np.ndarray] = {}
    for line in (sparse_dir / "points3D.txt").read_text().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        points[int(parts[0])] = np.array([float(v) for v in parts[1:4]], dtype=np.float64)

    inside = np.zeros(xyz.shape[0], dtype=bool)
    per_view_d = {}
    for name in frames:
        qvec, tvec, ids = poses[name]
        w, x, y, z = qvec
        R = np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
            ],
            dtype=np.float64,
        )
        C = -R.T @ tvec
        depths = [float((points[i] - C) @ R[2]) for i in ids if i in points]
        depths = [d for d in depths if d > 0]
        d = float(np.median(depths)) if depths else 1.0
        per_view_d[name] = d
        znear, zfar = znear_frac * d, zfar_frac * d
        for j, p in enumerate(xyz):
            rel = p - C
            zc = float(rel @ R[2])
            if not (znear < zc < zfar):
                continue
            u = fx * float(rel @ R[0]) / zc + cx
            v = fy * float(rel @ R[1]) / zc + cy
            if 0 <= u < width and 0 <= v < height:
                inside[j] = True
    return inside, per_view_d


# --- TRIPS's rule: schedule + threshold ---------------------------------------------


def test_fires_at_matches_trips_schedule() -> None:
    cfg = PointRemovalConfig(enabled=True, start_epoch=200, every_epochs=50)
    assert [e for e in range(401) if cfg.fires_at(e)] == [200, 250, 300, 350, 400]
    assert not any(PointRemovalConfig(enabled=False, start_epoch=0, every_epochs=1).fires_at(e) for e in range(10))


def test_shade_prune_fires_at_is_independent() -> None:
    cfg = ShadePruneConfig(enabled=True, start_epoch=3, every_epochs=2)
    assert [e for e in range(10) if cfg.fires_at(e)] == [3, 5, 7, 9]


def test_removal_keep_mask_is_the_trips_threshold() -> None:
    conf = np.array([0.01, 0.29999, 0.3, 0.7, 0.999])
    keep = prune.removal_keep_mask(conf, conf_threshold=0.3, min_points=0)
    assert keep.tolist() == [False, False, True, True, True]


def test_removal_keep_mask_respects_min_points_by_confidence() -> None:
    conf = np.array([0.9, 0.1, 0.5, 0.2, 0.8])
    # Threshold 0.95 would delete everything; min_points=3 keeps the top 3 by confidence.
    keep = prune.removal_keep_mask(conf, conf_threshold=0.95, min_points=3)
    assert keep.tolist() == [True, False, True, False, True]
    assert int(keep.sum()) == 3
    # min_points above the cloud size keeps everything.
    assert prune.removal_keep_mask(conf, 0.95, min_points=99).all()


def test_shade_prune_keep_mask_needs_all_three_conditions() -> None:
    inside = np.array([True, True, True, False])
    lum = np.array([0.1, 0.1, 0.9, 0.1])
    conf = np.array([0.2, 0.9, 0.2, 0.2])
    keep = prune.shade_prune_keep_mask(inside, lum, conf, 0.25, 0.5, min_points=0)
    # Only point 0 is in-region AND dark AND low-confidence.
    assert keep.tolist() == [False, True, True, True]


def test_shade_prune_keep_mask_respects_min_points() -> None:
    inside = np.ones(4, dtype=bool)
    lum = np.zeros(4)
    conf = np.array([0.4, 0.1, 0.3, 0.2])
    keep = prune.shade_prune_keep_mask(inside, lum, conf, 0.25, 0.5, min_points=2)
    assert keep.tolist() == [True, False, True, False]


# --- region definition ---------------------------------------------------------------


def test_region_matches_brute_force(tmp_path: Path) -> None:
    scene_root, point_set, names = build_scene_with_observations(tmp_path)
    sparse_dir = scene_root / "sparse_txt"
    frames = names[:2]
    xyz = point_set.xyz.astype(np.float64)

    views = prune.build_shade_region(sparse_dir, frames, 0.05, 0.5)
    inside, zfrac = prune.in_region(views, xyz)
    expected_inside, expected_d = brute_force_region(sparse_dir, frames, 0.05, 0.5, xyz)

    assert [v.name for v in views] == frames
    for v in views:
        assert v.d == pytest.approx(expected_d[v.name])
        assert v.znear == pytest.approx(0.05 * v.d)
        assert v.zfar == pytest.approx(0.5 * v.d)
    assert inside.tolist() == expected_inside.tolist()
    # The fixture is meaningful: the region is neither empty nor everything.
    assert 0 < int(inside.sum()) < len(xyz)
    assert np.isfinite(zfrac[inside]).all()
    assert not np.isfinite(zfrac[~inside]).any()


def test_region_rejects_unregistered_frame(tmp_path: Path) -> None:
    scene_root, _point_set, _names = build_scene_with_observations(tmp_path)
    with pytest.raises(ValueError, match="not registered"):
        prune.build_shade_region(scene_root / "sparse_txt", ["nope.jpg"], 0.05, 0.5)


def test_dark_mass_stats_matches_the_exported_ply(tmp_path: Path) -> None:
    """In-process statistic == the statistic the audit would read off the exported PLY."""
    scene_root, point_set, names = build_scene_with_observations(tmp_path)
    views = prune.build_shade_region(scene_root / "sparse_txt", names[:2], 0.05, 0.5)

    ply = build_synthetic_ply(tmp_path, point_set, color_noise_std=0.0)
    stats = prune.dark_mass_stats(views, point_set.xyz, point_set.rgb0, point_set.conf0, 0.25)

    # Re-read the PLY exactly the way depthprior_shade_audit.py does: opacity through a
    # sigmoid, colour through 0.5 + SH_C0 * f_dc.
    raw = GaussianPlySource(ply, min_opacity=0.0).build()
    header, body = Path(ply).read_bytes().split(b"end_header\n", 1)
    props = [line.split()[-1].decode() for line in header.split(b"\n") if line.startswith(b"property")]
    arr = np.frombuffer(body, dtype=np.dtype([(p, "<f4") for p in props]))
    op = 1.0 / (1.0 + np.exp(-arr["opacity"].astype(np.float64)))
    dc = np.stack([arr[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float64)
    rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    from_ply = prune.dark_mass_stats(views, raw.xyz, rgb, op, 0.25)

    assert stats["n_in_region"] == from_ply["n_in_region"]
    assert stats["mass_in_region"] == pytest.approx(from_ply["mass_in_region"], rel=1e-5)
    assert stats["dark_mass_lum0.25"] == pytest.approx(from_ply["dark_mass_lum0.25"], rel=1e-5)
    assert stats["dark_mass_fraction"] == pytest.approx(from_ply["dark_mass_fraction"], rel=1e-5)
    assert 0.0 <= stats["dark_mass_fraction"] <= 1.0


# --- Trainer surgery -----------------------------------------------------------------


def _trainer_with_observations(tmp_path: Path, **overrides) -> tuple[Trainer, list[str]]:
    scene_root, point_set, names = build_scene_with_observations(tmp_path)
    ply_path = build_synthetic_ply(tmp_path, point_set)
    cfg = tiny_train_config(scene_root, ply_path, tmp_path / "run", tmp_path / "cache", **overrides)
    return Trainer(cfg), names


def test_apply_keep_mask_drops_points_and_keeps_adam_moments_aligned(tmp_path: Path) -> None:
    trainer, _names = _trainer_with_observations(tmp_path)
    # Two steps so Adam actually holds first/second moments for every point group.
    trainer.train_step()
    trainer.train_step()

    n_before = len(trainer.point_params)
    xyz_before = trainer.point_params.xyz.detach().clone()
    moments_before = {
        attr: trainer.optimizer.state[getattr(trainer.point_params, attr)]["exp_avg"].detach().clone()
        for attr, _group in Trainer._POINT_PARAM_GROUPS
    }

    keep = np.ones(n_before, dtype=bool)
    keep[::4] = False  # drop every fourth point
    n_removed = trainer._apply_keep_mask(keep, "unit-test")

    index = torch.from_numpy(np.flatnonzero(keep).astype(np.int64))
    assert n_removed == n_before - int(keep.sum())
    assert len(trainer.point_params) == int(keep.sum())
    assert trainer.points_removed_total == n_removed
    assert torch.equal(trainer.point_params.xyz.detach(), xyz_before.index_select(0, index))
    assert trainer.point_params.provenance.shape[0] == int(keep.sum())

    for attr, group in Trainer._POINT_PARAM_GROUPS:
        param = getattr(trainer.point_params, attr)
        assert param.shape[0] == int(keep.sum())
        # The optimizer group holds the NEW parameter object, not a stale one.
        assert any(p is param for p in trainer.optimizer.param_groups[trainer._group_index[group]]["params"])
        state = trainer.optimizer.state[param]
        assert state["exp_avg"].shape == param.shape
        assert state["exp_avg_sq"].shape == param.shape
        assert torch.equal(state["exp_avg"], moments_before[attr].index_select(0, index))

    # And training still runs: no shape error anywhere in render -> loss -> step.
    trainer.train_step()
    for attr, _group in Trainer._POINT_PARAM_GROUPS:
        assert getattr(trainer.point_params, attr).shape[0] == int(keep.sum())


def test_apply_keep_mask_all_true_is_a_no_op(tmp_path: Path) -> None:
    trainer, _names = _trainer_with_observations(tmp_path)
    before = trainer.point_params.xyz
    assert trainer._apply_keep_mask(np.ones(len(trainer.point_params), dtype=bool), "noop") == 0
    assert trainer.point_params.xyz is before
    assert trainer.points_removed_total == 0


def test_maybe_prune_points_runs_the_trips_rule(tmp_path: Path) -> None:
    trainer, _names = _trainer_with_observations(
        tmp_path,
        point_removal={"enabled": True, "start_epoch": 0, "every_epochs": 1, "conf_threshold": 0.5, "min_points": 0},
    )
    conf = trainer.point_params.conf().detach().numpy()
    expected_removed = int((conf < 0.5).sum())
    assert expected_removed > 0, "fixture must have some points below the cutoff"

    result = trainer.maybe_prune_points(epoch=0)
    assert result["point_removal"] == expected_removed
    assert (trainer.point_params.conf().detach().numpy() >= 0.5).all()
    # Not a firing epoch for a start_epoch of 0 with every_epochs 1? It always is;
    # so check a disabled config instead.
    assert trainer.maybe_prune_points(epoch=1)["point_removal"] == 0


def test_maybe_prune_points_min_points_floor(tmp_path: Path) -> None:
    trainer, _names = _trainer_with_observations(
        tmp_path,
        point_removal={"enabled": True, "start_epoch": 0, "every_epochs": 1, "conf_threshold": 1.0, "min_points": 17},
    )
    trainer.maybe_prune_points(epoch=0)
    assert len(trainer.point_params) == 17


def test_shade_prune_only_touches_dark_in_region_points(tmp_path: Path) -> None:
    trainer, _names = _trainer_with_observations(
        tmp_path,
        shade_prune={
            "enabled": True,
            "frames": FIRST_TWO_FRAMES,
            "start_epoch": 0,
            "every_epochs": 1,
            "lum_threshold": 0.5,
            "conf_threshold": 1.0,
            "min_points": 0,
        },
    )
    xyz, rgb, conf = trainer._point_arrays()
    views = trainer.shade_views()
    assert views is not None
    inside, _z = prune.in_region(views, xyz)
    expected = int((inside & (prune.luminance(rgb) < 0.5) & (conf < 1.0)).sum())
    assert expected > 0, "fixture must have dark in-region points"

    result = trainer.maybe_prune_points(epoch=0)
    assert result["shade_prune"] == expected
    assert len(trainer.point_params) == len(xyz) - expected
    # Every survivor is now outside the region or not dark.
    xyz2, rgb2, _conf2 = trainer._point_arrays()
    inside2, _z2 = prune.in_region(views, xyz2)
    assert not (inside2 & (prune.luminance(rgb2) < 0.5)).any()


def test_shade_region_failure_is_logged_not_raised(tmp_path: Path) -> None:
    trainer, _names = _trainer_with_observations(
        tmp_path, shade_prune={"enabled": True, "frames": ["not_a_frame.jpg"], "start_epoch": 0, "every_epochs": 1}
    )
    assert trainer.shade_views() is None
    assert "not registered" in (trainer._shade_region_error or "")
    assert trainer.maybe_prune_points(epoch=0) == {}
    assert trainer.point_stats()["shade_region"]["error"]


# --- logging + checkpoints ------------------------------------------------------------


def test_evaluate_logs_point_count_and_dark_mass(tmp_path: Path) -> None:
    trainer, _names = _trainer_with_observations(tmp_path, shade_prune={"frames": FIRST_TWO_FRAMES})
    metrics = trainer.evaluate(epoch=0)
    points = metrics["points"]
    assert points["n_points"] == len(trainer.point_params)
    assert points["n_removed_total"] == 0
    region = points["shade_region"]
    assert set(region) >= {"n_in_region", "mass_in_region", "dark_mass_lum0.25", "dark_mass_fraction"}
    assert 0.0 <= region["dark_mass_fraction"] <= 1.0

    on_disk = json.loads((Path(trainer.cfg.run_dir) / "eval_ep0000" / "metrics.json").read_text())
    assert on_disk["points"]["n_points"] == points["n_points"]
    rows = [json.loads(line) for line in (Path(trainer.cfg.run_dir) / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["points"]["shade_region"]["dark_mass_fraction"] == pytest.approx(region["dark_mass_fraction"])


def test_log_dark_mass_off_omits_the_region_block(tmp_path: Path) -> None:
    trainer, _names = _trainer_with_observations(tmp_path, shade_prune={"log_dark_mass": False})
    stats = trainer.point_stats()
    assert "shade_region" not in stats
    assert stats["n_points"] == len(trainer.point_params)


def test_checkpoint_round_trips_after_a_removal(tmp_path: Path) -> None:
    trainer, _names = _trainer_with_observations(tmp_path)
    trainer.train_step()
    keep = np.ones(len(trainer.point_params), dtype=bool)
    keep[:50] = False
    trainer._apply_keep_mask(keep, "unit-test")
    trainer.train_step()
    path = trainer.save_checkpoint(epoch=0)

    # A fresh Trainer builds the FULL point source; loading must resize first.
    fresh, _names2 = _trainer_with_observations(tmp_path / "second")
    assert len(fresh.point_params) > len(trainer.point_params)
    from trippy.train import checkpoint_io

    fresh.load_state(checkpoint_io.load_checkpoint(path, map_location="cpu"))
    assert len(fresh.point_params) == len(trainer.point_params)
    assert torch.allclose(fresh.point_params.xyz, trainer.point_params.xyz)
    assert torch.allclose(fresh.point_params.feat, trainer.point_params.feat)
    assert fresh.points_removed_total == trainer.points_removed_total
    fresh.train_step()  # optimiser groups still line up with the resized params.


def test_fit_prunes_on_schedule(tmp_path: Path) -> None:
    trainer, _names = _trainer_with_observations(
        tmp_path,
        epochs=3,
        point_removal={"enabled": True, "start_epoch": 1, "every_epochs": 1, "conf_threshold": 0.5, "min_points": 0},
    )
    n_start = len(trainer.point_params)
    trainer.fit()
    assert len(trainer.point_params) < n_start
    assert trainer.points_removed_total == n_start - len(trainer.point_params)
    exported = GaussianPlySource(Path(trainer.cfg.run_dir) / "export.ply", min_opacity=0.0).build()
    assert len(exported) == len(trainer.point_params)
