"""Recover which Gaussian in the source PLY seeded which TRIPS point.

Module: trippy.clean.mapping
Purpose: Design B scores a Gaussian by its TRIPS twin, so it needs the
    index correspondence between the two clouds. That correspondence is
    NOT stored in a checkpoint -- it has to be reconstructed and proven.
The rule (read off `trippy.points.gaussian_ply.GaussianPlySource.build`):
    1. `keep = sigmoid(opacity) >= min_opacity` -- a boolean mask, so ROW
       ORDER IS PRESERVED and TRIPS point `i` is PLY row `flatnonzero(keep)[i]`.
    2. If `max_points` was set, a seeded RNG subsample is applied on top;
       the config that produced a checkpoint records it, and this module
       reproduces it with the same seed.
    3. `trippy.train.trainer.Trainer._apply_keep_mask` may then have dropped
       points DURING training (`points_removed_total` in the checkpoint).
       That breaks the 1:1 mapping, and there is no record of which points
       went, so the correspondence is recovered geometrically instead.
Invariants:
    - The mapping is always VERIFIED, never assumed. `PointParams.init_conf`
      is a frozen snapshot of `sigmoid(ply.opacity)` taken before the first
      optimiser step (see that class's docstring), so it is a per-point
      fingerprint of the source row: a correct mapping reproduces it
      exactly, and a wrong one does not. `recover_mapping` refuses to
      return a mapping whose fingerprint mismatch exceeds `tol`.
    - The geometric fallback matches on INITIAL positions (the PLY's own
      `x/y/z`), because a trained point has drifted; the drift is small
      (measured: p50 0.015, max 0.15 world units on kkv2-1-full-masked)
      compared with the median nearest-neighbour spacing, so a nearest
      neighbour on positions is well posed -- but it is still checked
      against `init_conf` afterwards.
Units: `min_opacity` and `init_conf` are dimensionless opacities in (0, 1);
    positions are COLMAP world units.
Related docs: docs/SPEC.md D4 (point source 1: trained Gaussian centres);
    trippy.points.gaussian_ply (the read side this inverts);
    trippy.train.params.PointParams (`init_conf`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trippy.clean.ply_filter import PlyLayout, read_columns
from trippy.constants import (
    CLEAN_MAPPING_FINGERPRINT_TOL,
    CLEAN_MAPPING_METHOD_NEAREST,
    CLEAN_MAPPING_METHOD_OPACITY,
    DEFAULT_MIN_OPACITY,
)
from trippy.points.gaussian_ply import _sigmoid


@dataclass(frozen=True)
class GaussianPointMapping:
    """TRIPS point index -> source PLY row index, plus the evidence for it.

    Attributes:
        ply_row: (n_points,) int64; `ply_row[i]` is the PLY row that seeded
            TRIPS point `i`.
        n_ply: total rows in the source PLY.
        method: `CLEAN_MAPPING_METHOD_OPACITY` (the deterministic
            reconstruction) or `CLEAN_MAPPING_METHOD_NEAREST` (the
            geometric fallback).
        evidence: numbers a reviewer can check the claim with -- see
            `recover_mapping`.
    """

    ply_row: np.ndarray
    n_ply: int
    method: str
    evidence: dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        """How many TRIPS points the mapping covers."""
        return int(self.ply_row.shape[0])


def recover_mapping(
    layout: PlyLayout,
    init_conf: np.ndarray,
    point_xyz: np.ndarray | None = None,
    min_opacity: float = DEFAULT_MIN_OPACITY,
    max_points: int | None = None,
    seed: int = 0,
    tol: float = CLEAN_MAPPING_FINGERPRINT_TOL,
) -> GaussianPointMapping:
    """Recover, and prove, the TRIPS point -> PLY row correspondence.

    Tries the deterministic reconstruction first (the opacity filter, then
    the seeded `max_points` subsample if one was configured). If the
    reconstructed count matches `len(init_conf)` and the `init_conf`
    fingerprint agrees within `tol`, that mapping is returned. Otherwise --
    which is what a training-time point removal looks like from here -- it
    falls back to a nearest neighbour, on positions, from each TRIPS point
    to the opacity-filtered PLY rows, and reports the same fingerprint
    check on the result rather than hiding it.

    Args:
        layout: source PLY layout (`trippy.clean.ply_filter.read_ply_layout`).
        init_conf: (n_points,) the checkpoint's `point_params.init_conf`.
        point_xyz: (n_points, 3) the checkpoint's trained positions.
            Required only for the fallback.
        min_opacity: the training config's `point_source.min_opacity`.
        max_points: the training config's `point_source.max_points`, if any.
        seed: the training config's `point_source.seed` (for `max_points`).
        tol: largest per-point `|init_conf - sigmoid(ply.opacity)|` the
            deterministic mapping may show and still be accepted. The
            snapshot is a copy of the same float32, so a correct mapping
            scores exactly 0; the tolerance only absorbs the
            `EXPORT_OPACITY_CLAMP_EPS` round trip a resumed pre-PR#37
            checkpoint would introduce.

    Returns:
        A `GaussianPointMapping` whose `evidence` holds `n_ply`,
        `n_points`, `n_opacity_keep`, `fingerprint_max_abs_err`,
        `fingerprint_exact_frac`, and (fallback only) `nn_distance_p50` /
        `nn_distance_max` / `nn_duplicate_targets`.

    Raises:
        ValueError: neither route produced a mapping within `tol`, or the
            fallback was needed but `point_xyz` was not supplied.
    """
    init_conf = np.asarray(init_conf, dtype=np.float64)
    n_points = init_conf.shape[0]
    opacity = read_columns(layout, ["opacity"])["opacity"]
    ply_conf = _sigmoid(np.asarray(opacity, dtype=np.float64))
    # `GaussianPlySource.build`'s filter, verbatim -- note its `>=`, not `>`.
    keep = ply_conf >= min_opacity
    keep_rows = np.flatnonzero(keep)

    candidate = keep_rows
    if max_points is not None and keep_rows.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        picked = rng.choice(keep_rows.shape[0], size=max_points, replace=False)
        candidate = keep_rows[picked]

    evidence = {
        "n_ply": int(layout.count),
        "n_points": int(n_points),
        "n_opacity_keep": int(keep_rows.shape[0]),
        "min_opacity": float(min_opacity),
        "max_points": max_points,
    }

    if candidate.shape[0] == n_points:
        err = np.abs(ply_conf[candidate].astype(np.float32).astype(np.float64) - init_conf)
        evidence["fingerprint_max_abs_err"] = float(err.max()) if err.size else 0.0
        evidence["fingerprint_exact_frac"] = float((err == 0).mean()) if err.size else 1.0
        if evidence["fingerprint_max_abs_err"] <= tol:
            return GaussianPointMapping(
                ply_row=candidate.astype(np.int64),
                n_ply=int(layout.count),
                method=CLEAN_MAPPING_METHOD_OPACITY,
                evidence=evidence,
            )

    if point_xyz is None:
        raise ValueError(
            "the deterministic mapping did not reproduce init_conf and no point_xyz was given "
            f"for the nearest-neighbour fallback (evidence: {evidence})"
        )

    # Deferred: scipy is a heavy import and only the fallback needs it.
    from scipy.spatial import cKDTree

    cols = read_columns(layout, ["x", "y", "z"])
    ply_xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float64)[keep_rows]
    tree = cKDTree(ply_xyz)
    dist, local = tree.query(np.asarray(point_xyz, dtype=np.float64), k=1)
    rows = keep_rows[local].astype(np.int64)

    err = np.abs(ply_conf[rows].astype(np.float32).astype(np.float64) - init_conf)
    evidence["fingerprint_max_abs_err"] = float(err.max()) if err.size else 0.0
    evidence["fingerprint_exact_frac"] = float((err == 0).mean()) if err.size else 1.0
    evidence["nn_distance_p50"] = float(np.percentile(dist, 50)) if dist.size else 0.0
    evidence["nn_distance_max"] = float(dist.max()) if dist.size else 0.0
    evidence["nn_duplicate_targets"] = int(rows.shape[0] - np.unique(rows).shape[0])
    if evidence["fingerprint_max_abs_err"] > tol:
        raise ValueError(
            "nearest-neighbour mapping does not reproduce the checkpoint's init_conf fingerprint "
            f"(max abs err {evidence['fingerprint_max_abs_err']:.3e} > tol {tol:.3e}); "
            f"evidence: {evidence}"
        )
    return GaussianPointMapping(
        ply_row=rows,
        n_ply=int(layout.count),
        method=CLEAN_MAPPING_METHOD_NEAREST,
        evidence=evidence,
    )
