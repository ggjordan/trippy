"""Per-point scores read out of a trained checkpoint, without building a Trainer.

Module: trippy.clean.score
Purpose: Design B needs four numbers per TRIPS point and nothing else --
    the learned confidence, the confidence it started at, how far the point
    drifted, and its base colour. Constructing a `Trainer` to get them
    would rebuild the point source (a kNN pass over 7.5M points) and load a
    U-Net; this reads the `point_params` sub-dict straight out of the
    checkpoint with `torch.load(..., mmap=True)` instead, so a 792 MB
    checkpoint costs ~180 MB of resident arrays.
Invariants:
    - `conf` is `sigmoid(CONF_SIGMOID_SCALE * raw_conf)` -- the same
      expression `trippy.train.params.PointParams.conf` evaluates, so the
      number here IS the trained model's confidence, not an approximation.
    - `rgb` is `clip(feat[:, :3], 0, 1)` -- what `trippy.train.export`
      writes into `f_dc_*` and what the Splats shade audit reads back
      (`trippy.train.prune`'s module docstring pins that chain).
    - Read-only: nothing here writes a checkpoint or moves a tensor to a
      device other than the CPU.
Units: `conf`/`init_conf`/`rgb` dimensionless; `drift` and `xyz` are COLMAP
    world units.
Related docs: trippy.train.params (the parametrisation);
    trippy.train.prune (the audit rule these feed).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trippy.constants import CONF_SIGMOID_SCALE


@dataclass(frozen=True)
class PointScores:
    """What a trained checkpoint knows about each of its points.

    Attributes:
        conf: (N,) learned confidence, `sigmoid(10 * raw_conf)`, in (0, 1).
        init_conf: (N,) the confidence the point was seeded with, i.e.
            `sigmoid(source_ply.opacity)`. Frozen at construction.
        xyz: (N, 3) trained world position. The SEED position lives in the
            PLY, not the checkpoint, so positional drift is computed by the
            caller once a mapping is in hand (`drift_stats`).
        rgb: (N, 3) base colour in [0, 1].
        epoch: the checkpoint's epoch.
        points_removed_total: how many points training dropped (0 means the
            seed correspondence is still 1:1 -- see `trippy.clean.mapping`).
        cfg: the checkpoint's serialised `TrainConfig` dict.
    """

    conf: np.ndarray
    init_conf: np.ndarray
    xyz: np.ndarray
    rgb: np.ndarray
    epoch: int
    points_removed_total: int
    cfg: dict

    @property
    def n(self) -> int:
        """Number of points."""
        return int(self.conf.shape[0])

    def point_source_cfg(self) -> dict:
        """The `point_source` block of the checkpoint's config, or `{}`."""
        source = self.cfg.get("point_source") if isinstance(self.cfg, dict) else None
        return source if isinstance(source, dict) else {}


def load_point_scores(checkpoint_path: str | Path) -> PointScores:
    """Read `point_params` (and the config) out of a checkpoint, CPU only.

    Args:
        checkpoint_path: a `.pt` written by `trippy.train.checkpoint_io`.

    Returns:
        A `PointScores`.

    Raises:
        KeyError: the checkpoint has no `point_params` block.
    """
    # Deferred: torch is a multi-second import and the CLI's --help must not pay it.
    import torch

    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False, mmap=True)
    if "point_params" not in payload:
        raise KeyError(f"{checkpoint_path}: no 'point_params' in checkpoint (keys: {sorted(payload)})")
    pp = payload["point_params"]
    raw_conf = pp["raw_conf"].numpy().astype(np.float64)
    conf = 1.0 / (1.0 + np.exp(-CONF_SIGMOID_SCALE * raw_conf))
    feat = pp["feat"].numpy()
    return PointScores(
        conf=conf,
        init_conf=pp["init_conf"].numpy().astype(np.float64),
        xyz=pp["xyz"].numpy().astype(np.float32),
        rgb=np.clip(feat[:, :3], 0.0, 1.0).astype(np.float64),
        epoch=int(payload.get("epoch", -1)),
        points_removed_total=int(payload.get("points_removed_total", 0)),
        cfg=payload.get("cfg", {}) or {},
    )


def confidence_summary(scores: PointScores, thresholds: list[float]) -> dict:
    """Distribution of `conf` and `conf / init_conf`, for the run's summary.json.

    Args:
        scores: from `load_point_scores`.
        thresholds: confidence cutoffs to count points below.

    Returns:
        `{"percentiles": {...}, "below": {"<t>": n, ...},
        "ratio_percentiles": {...}}`.
    """
    pcts = [1, 5, 10, 25, 50, 75, 90, 99]
    ratio = scores.conf / np.maximum(scores.init_conf, np.finfo(np.float64).tiny)
    return {
        "percentiles": {str(p): float(np.percentile(scores.conf, p)) for p in pcts},
        "below": {f"{t:g}": int((scores.conf < t).sum()) for t in thresholds},
        "ratio_percentiles": {str(p): float(np.percentile(ratio, p)) for p in pcts},
        "fell_below_half_of_init": int((ratio < 0.5).sum()),
    }


def drift_stats(point_xyz: np.ndarray, seed_xyz: np.ndarray) -> dict:
    """How far the trained points moved from the Gaussian centres that seeded them.

    A large drift means the TRIPS optimiser did not believe the seed
    position, which is a second (weaker) fog signal -- reported for the
    record, never used as a deletion criterion on its own, because a point
    that drifted onto a surface is exactly the case where the Gaussian's
    ORIGINAL position is the wrong thing to judge.

    Args:
        point_xyz: (N, 3) trained positions.
        seed_xyz: (N, 3) the PLY positions those points were seeded from,
            in the same order.

    Returns:
        `{"p50", "p90", "p99", "max", "mean"}` of the per-point distance,
        in COLMAP world units.
    """
    d = np.linalg.norm(np.asarray(point_xyz, dtype=np.float64) - np.asarray(seed_xyz, dtype=np.float64), axis=1)
    return {
        "p50": float(np.percentile(d, 50)),
        "p90": float(np.percentile(d, 90)),
        "p99": float(np.percentile(d, 99)),
        "max": float(d.max()) if d.size else 0.0,
        "mean": float(d.mean()) if d.size else 0.0,
    }
