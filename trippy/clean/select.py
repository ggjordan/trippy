"""The deletion rules: which Gaussians TRIPS says are fog.

Module: trippy.clean.select
Purpose: turn `trippy.clean.score.PointScores` (+ optionally the shade
    audit region) into a boolean "delete this Gaussian" mask, at three
    named aggressiveness levels so Jordan can pick with his own eyes
    rather than from a metric (AGENTS.md Sec 7: "the viewer verdict is the
    verdict").
The rule, and why it is only confidence:
    TRIPS's confidence is a per-point alpha trained against 756 photographs
    that all see the same world from different places. A Gaussian sitting
    in free space between the camera path and a surface is contradicted by
    every view that sees past it, so gradient descent drives its
    confidence down; a Gaussian ON a surface is confirmed by every view
    that sees it and its confidence rises. That is the whole signal, and it
    is the reason this works where the earlier Splats-side pruning failed:
    the Splats prune keyed on "dark and in the shade volume", which is also
    true of the GROUND under the tree, whereas confidence separates the two
    because TRIPS renders that ground.
    Colour is deliberately NOT part of the test (that is what removed the
    ground last time), and `init_conf` is not either: a Gaussian that
    started weak and stayed weak is still fog.
Invariants:
    - Every variant is a pure function of `conf` and (for `shade`) the
      audit's own `inside` mask -- no free parameter is introduced here
      that is not in `VARIANTS`.
    - A Gaussian with no TRIPS twin (dropped by the source's `min_opacity`
      filter before training) is NEVER deleted by this module: TRIPS was
      never shown it, so it has no opinion, and Design B's promise is that
      the splat is only ever subtracted from where TRIPS has evidence.
Units: confidences are dimensionless, in (0, 1).
Related docs: docs/SPEC.md D2 (Design B); docs/EDITOR.md Sec 3 (the shade
    finder, whose `inside AND dark AND low-conf` rule this deliberately
    narrows to `inside AND low-conf`); trippy.train.prune (the region).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trippy.constants import (
    CLEAN_VARIANT_ALL_005_THRESHOLD,
    CLEAN_VARIANT_ALL_015_THRESHOLD,
    CLEAN_VARIANT_SHADE_THRESHOLD,
)


@dataclass(frozen=True)
class CleanVariant:
    """One named aggressiveness level.

    Attributes:
        name: the `--variant` value and the `deliver.sh` name suffix.
        conf_threshold: delete a point whose learned confidence is strictly
            below this.
        shade_only: when True, only delete inside the shade audit region
            (so the rest of the scene is bit-identical to the original).
        why: one line for the summary and the delivery message.
    """

    name: str
    conf_threshold: float
    shade_only: bool
    why: str


VARIANTS: tuple[CleanVariant, ...] = (
    CleanVariant(
        name="shade",
        conf_threshold=CLEAN_VARIANT_SHADE_THRESHOLD,
        shade_only=True,
        why="lowest-confidence points, inside the measured shade volume only",
    ),
    CleanVariant(
        name="005",
        conf_threshold=CLEAN_VARIANT_ALL_005_THRESHOLD,
        shade_only=False,
        why="lowest-confidence points, everywhere in the scene",
    ),
    CleanVariant(
        name="015",
        conf_threshold=CLEAN_VARIANT_ALL_015_THRESHOLD,
        shade_only=False,
        why="low-confidence points, everywhere in the scene (most aggressive)",
    ),
)


def variant_by_name(name: str) -> CleanVariant:
    """Look up a `CleanVariant` by its `name`.

    Args:
        name: one of `[v.name for v in VARIANTS]`.

    Returns:
        The variant.

    Raises:
        KeyError: unknown name.
    """
    for variant in VARIANTS:
        if variant.name == name:
            return variant
    raise KeyError(f"unknown clean variant {name!r} (have {[v.name for v in VARIANTS]})")


def deletion_mask(
    conf: np.ndarray,
    conf_threshold: float,
    inside: np.ndarray | None = None,
) -> np.ndarray:
    """Which TRIPS points (and so which seed Gaussians) to delete.

    Args:
        conf: (N,) learned confidence per TRIPS point.
        conf_threshold: delete where `conf < conf_threshold`.
        inside: (N,) bool from `trippy.train.prune.in_region`, or None to
            apply the test scene-wide.

    Returns:
        (N,) bool, True where the point should be deleted.

    Raises:
        ValueError: `inside` is given with a different length to `conf`.
    """
    conf = np.asarray(conf, dtype=np.float64)
    drop = conf < conf_threshold
    if inside is None:
        return drop
    inside = np.asarray(inside, dtype=bool)
    if inside.shape != conf.shape:
        raise ValueError(f"inside has shape {inside.shape}, conf has {conf.shape}")
    return drop & inside


def ply_keep_mask(delete_points: np.ndarray, ply_row: np.ndarray, n_ply: int) -> np.ndarray:
    """Lift a per-TRIPS-point deletion onto a per-PLY-row keep mask.

    Rows no TRIPS point maps to are always kept (module invariant: TRIPS
    has no opinion about a Gaussian it never saw).

    Args:
        delete_points: (n_points,) bool, from `deletion_mask`.
        ply_row: (n_points,) int64, from `trippy.clean.mapping`.
        n_ply: total rows in the source PLY.

    Returns:
        (n_ply,) bool keep mask.
    """
    keep = np.ones(n_ply, dtype=bool)
    keep[np.asarray(ply_row)[np.asarray(delete_points, dtype=bool)]] = False
    return keep
