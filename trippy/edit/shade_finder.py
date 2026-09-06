"""Shade-cloud finder: dark, low-confidence points in the audit region, as a `pointset` Region.

Module: trippy.edit.shade_finder
Purpose: the E2 milestone's "smart selection" for the shade problem
    (docs/EDITOR.md Sec 3 "2. Shade-cloud finder (E2)") -- a thin wrapper
    that calls `trippy.train.prune`'s exact, already-verified region /
    dark-mass functions (the same ones `depthprior_shade_audit.py`'s own
    numbers come from, docs/EXPERIMENTS.md "Shade audit") and turns the
    result into a `pointset` Region Jordan can delete/fade/blend/undo like
    any other, per ADR-0007's "reused here as a selector, deliberately ...
    the action it feeds is new" decision.
Invariants:
    - Uses `prune.build_shade_region` / `prune.in_region` / `prune.luminance`
      / `prune.confidence_drop_mask` / `prune.dark_mass_stats` verbatim --
      no re-derivation of the audit rule here.
    - The selection is `inside AND dark AND confidence_drop_mask` -- the
      same AND `prune.shade_prune_keep_mask` negates -- WITHOUT that
      function's `min_points` floor: a finder has no reason to guarantee a
      minimum selection size the way a training-time removal pass does.
    - The returned Region defaults to `op="fade", mix=0.0` -- docs/EDITOR.md
      Sec 1's own worked example for a shade-cloud region -- editable by
      the caller afterwards.
    - `summary["mass_fraction"]` is `prune.dark_mass_stats`'s
      `dark_mass_fraction`, computed over the WHOLE audit region (not just
      the selected points), so it is directly comparable to the audit's
      own headline number -- E2's acceptance: "matches the audit's own
      number for that scene to float precision".
Units: same as `trippy.train.prune` (COLMAP world units, dimensionless
    luminance/confidence).
Related docs: docs/EDITOR.md Sec 3 "2. Shade-cloud finder (E2)";
    docs/decisions/ADR-0007-viewer-editing.md Sec "3. Selection tools ship
    cheapest/most-certain first"; docs/EXPERIMENTS.md "Shade audit";
    trippy.train.prune (the functions this module is built on).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from trippy.constants import (
    POINT_REMOVAL_DEFAULT_MODE,
    POINT_REMOVAL_DEFAULT_REL_FACTOR,
    SHADE_PRUNE_DEFAULT_CONF_THRESHOLD,
    SHADE_PRUNE_DEFAULT_LUM_THRESHOLD,
    SHADE_PRUNE_DEFAULT_ZFAR_FRAC,
    SHADE_PRUNE_DEFAULT_ZNEAR_FRAC,
)
from trippy.edit.model import Region, new_region_id
from trippy.train import prune

__all__ = ["find_shade_pointset", "find_shade_pointset_in_bundle"]


def find_shade_pointset(
    sparse_dir: str | Path,
    frames: list[str],
    xyz: np.ndarray,
    rgb: np.ndarray,
    conf: np.ndarray,
    znear_frac: float = SHADE_PRUNE_DEFAULT_ZNEAR_FRAC,
    zfar_frac: float = SHADE_PRUNE_DEFAULT_ZFAR_FRAC,
    lum_threshold: float = SHADE_PRUNE_DEFAULT_LUM_THRESHOLD,
    conf_threshold: float = SHADE_PRUNE_DEFAULT_CONF_THRESHOLD,
    mode: str = POINT_REMOVAL_DEFAULT_MODE,
    rel_factor: float = POINT_REMOVAL_DEFAULT_REL_FACTOR,
    init_conf: np.ndarray | None = None,
    region_id: str | None = None,
    name: str = "shade cloud",
) -> tuple[Region, dict[str, Any]]:
    """Build a `pointset` Region from the shade audit's own dark-mass rule.

    Args:
        sparse_dir: COLMAP model directory (`sparse/0` or `sparse_txt`).
        frames: shade-frame image filenames (`prune.build_shade_region`);
            every one must be registered in `sparse_dir`.
        xyz: `(N, 3)` world positions -- the SAME array `point_ids` will
            index (the point cloud's own row order).
        rgb: `(N, 3)` base colour in `[0, 1]`.
        conf: `(N,)` effective confidence in `(0, 1)`.
        znear_frac, zfar_frac: the audit's depth slab, as fractions of each
            frame's own median observed depth.
        lum_threshold: "dark" Rec.709 luminance cutoff.
        conf_threshold, mode, rel_factor, init_conf: the confidence test,
            identical in meaning to `prune.confidence_drop_mask`.
        region_id: stable id for the new region (default: a fresh one via
            `trippy.edit.model.new_region_id`).
        name: shown in the Regions panel.

    Returns:
        `(region, summary)`. `summary` has `n_points` (selection size),
        `mass_fraction` (the audit's own `dark_mass_fraction`, over the
        whole region, not just the selection), the raw `dark_mass_stats`
        dict, and the `thresholds` used (so a saved region's provenance is
        reproducible without re-reading this function's defaults).

    Raises:
        ValueError: from `prune.build_shade_region` (empty/unregistered
            `frames`) or `prune.confidence_drop_mask` (bad `mode`, or
            `mode="relative"` with no `init_conf`).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)

    views = prune.build_shade_region(sparse_dir, frames, znear_frac, zfar_frac)
    inside, _zfrac = prune.in_region(views, xyz)
    inside = inside & np.isfinite(xyz).all(axis=1)
    lum = prune.luminance(rgb)
    dark = lum < lum_threshold
    conf_drop = prune.confidence_drop_mask(conf, conf_threshold, mode, rel_factor, init_conf)
    selected = inside & dark & conf_drop
    point_ids = np.flatnonzero(selected).tolist()

    stats = prune.dark_mass_stats(views, xyz, rgb, conf, lum_threshold)

    region = Region(
        id=region_id or new_region_id(),
        name=name,
        kind="pointset",
        params={"point_ids": point_ids},
        mix=0.0,
        op="fade",
        enabled=True,
    )
    summary = {
        "n_points": int(selected.sum()),
        "mass_fraction": stats["dark_mass_fraction"],
        "dark_mass_stats": stats,
        "thresholds": {
            "frames": list(frames),
            "znear_frac": float(znear_frac),
            "zfar_frac": float(zfar_frac),
            "lum_threshold": float(lum_threshold),
            "conf_threshold": float(conf_threshold),
            "mode": mode,
            "rel_factor": float(rel_factor),
        },
    }
    return region, summary


def find_shade_pointset_in_bundle(
    bundle_dir: str | Path,
    scene_root: str | Path,
    frames: list[str],
    **kwargs: Any,
) -> tuple[Region, dict[str, Any]]:
    """`find_shade_pointset`, sourcing `xyz`/`rgb`/`conf` from a bundle's `points.npz`.

    `rgb` is `clip(feat[:, :3], 0, 1)` -- the base-colour channel every
    trippy-native bundle's feature vector is seeded with
    (`trippy.train.params.PointParams.__init__`: `feat[:, :3] = rgb0`,
    enforced by `feature_channels >= 3`), the same slice
    `trippy.train.trainer.Trainer.evaluate`'s own dark-mass logging reads.
    A bundle built from a source whose feature layout diverges from that
    convention should call `find_shade_pointset` directly with its own
    `rgb` array instead.

    Args:
        bundle_dir: bundle directory (reads `bundle.json`'s `"points"`
            filename, default `points.npz`).
        scene_root: scene directory; resolved to `sparse/0` or `sparse_txt`
            via `trippy.scene.dataset.resolve_sparse_dir`.
        frames: shade-frame image filenames.
        **kwargs: forwarded to `find_shade_pointset` (thresholds, `mode`,
            `region_id`, `name`, ...).

    Returns:
        Same as `find_shade_pointset`.
    """
    import json

    from trippy.scene.dataset import resolve_sparse_dir

    bundle_dir = Path(bundle_dir)
    bundle_doc = json.loads((bundle_dir / "bundle.json").read_text())
    points_path = bundle_dir / bundle_doc.get("points", "points.npz")
    with np.load(points_path) as data:
        xyz = np.asarray(data["xyz"], dtype=np.float64)
        feat = np.asarray(data["feat"], dtype=np.float64)
        conf = np.asarray(data["conf"], dtype=np.float64)
    rgb = np.clip(feat[:, :3], 0.0, 1.0)

    sparse_dir = resolve_sparse_dir(scene_root)
    return find_shade_pointset(sparse_dir, frames, xyz, rgb, conf, **kwargs)
