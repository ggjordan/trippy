"""Tests for trippy.edit.shade_finder: the E2 "shade-cloud finder" pointset Region.

Module: tests.test_edit_shade_finder
Invariants under test:
  - `find_shade_pointset`'s selection equals a brute-force
    (in-region AND dark AND low-confidence) computed independently in this
    file (via `test_train_prune.brute_force_region`, a from-scratch
    reimplementation of the audit's own projection, and a plain numpy
    Rec.709/threshold test written here) -- the "shade finder on the
    synthetic scene equals brute force" acceptance criterion.
  - `summary["mass_fraction"]` equals `trippy.train.prune.dark_mass_stats`'s
    own `dark_mass_fraction` computed directly over the same views/points
    (E2's acceptance: "matches the audit's own number ... to float
    precision").
  - The returned Region is a valid, `op="fade", mix=0.0` `pointset` region
    whose `point_ids` are exactly the selection.
Fixture: `tests/test_train_prune.py`'s synthetic COLMAP scene builder
    (`build_scene_with_observations`) -- never a real Splats scene.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from test_train_prune import brute_force_region, build_scene_with_observations

from trippy.edit.shade_finder import find_shade_pointset
from trippy.train import prune


def test_shade_finder_selection_equals_brute_force(tmp_path: Path) -> None:
    scene_root, point_set, names = build_scene_with_observations(tmp_path)
    sparse_dir = scene_root / "sparse_txt"
    frames = names[:2]
    xyz = point_set.xyz.astype(np.float64)
    rgb = point_set.rgb0.astype(np.float64)
    conf = point_set.conf0.astype(np.float64)

    lum_threshold = 0.25
    conf_threshold = 0.5

    region, summary = find_shade_pointset(
        sparse_dir,
        frames,
        xyz,
        rgb,
        conf,
        znear_frac=0.05,
        zfar_frac=0.5,
        lum_threshold=lum_threshold,
        conf_threshold=conf_threshold,
        mode="absolute",
    )

    # Independent brute-force selection: this file's own projection (brute_force_region,
    # written from the .txt files with no shared helper -- see its own docstring) AND a
    # plain numpy luminance/confidence test, computed from scratch here.
    expected_inside, _expected_d = brute_force_region(sparse_dir, frames, 0.05, 0.5, xyz)
    expected_lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    expected_selected = expected_inside & (expected_lum < lum_threshold) & (conf < conf_threshold)

    assert region.kind == "pointset"
    assert region.op == "fade"
    assert region.mix == 0.0
    assert sorted(region.params["point_ids"]) == np.flatnonzero(expected_selected).tolist()
    assert summary["n_points"] == int(expected_selected.sum())
    # The fixture is meaningful: some points are selected, not all/none.
    assert 0 < summary["n_points"] < len(xyz)

    # mass_fraction matches the audit's own dark_mass_stats, computed independently here
    # (not read back from `summary` a second way -- a genuinely separate call).
    views = prune.build_shade_region(sparse_dir, frames, 0.05, 0.5)
    stats = prune.dark_mass_stats(views, xyz, rgb, conf, lum_threshold)
    assert summary["mass_fraction"] == stats["dark_mass_fraction"]


def test_shade_finder_thresholds_recorded_in_summary(tmp_path: Path) -> None:
    scene_root, point_set, names = build_scene_with_observations(tmp_path)
    sparse_dir = scene_root / "sparse_txt"
    frames = names[:2]

    _region, summary = find_shade_pointset(
        sparse_dir,
        frames,
        point_set.xyz.astype(np.float64),
        point_set.rgb0.astype(np.float64),
        point_set.conf0.astype(np.float64),
        znear_frac=0.1,
        zfar_frac=0.6,
        lum_threshold=0.3,
        conf_threshold=0.4,
    )
    thresholds = summary["thresholds"]
    assert thresholds["frames"] == frames
    assert thresholds["znear_frac"] == 0.1
    assert thresholds["zfar_frac"] == 0.6
    assert thresholds["lum_threshold"] == 0.3
    assert thresholds["conf_threshold"] == 0.4


def test_shade_finder_rejects_unregistered_frame(tmp_path: Path) -> None:
    scene_root, point_set, _names = build_scene_with_observations(tmp_path)
    import pytest

    with pytest.raises(ValueError, match="not registered"):
        find_shade_pointset(
            scene_root / "sparse_txt",
            ["nope.jpg"],
            point_set.xyz.astype(np.float64),
            point_set.rgb0.astype(np.float64),
            point_set.conf0.astype(np.float64),
        )
