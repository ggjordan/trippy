"""Per-point blend-weight composition: gate default + ordered region overrides.

Module: trippy.edit.weights
Purpose: turn an `EditDocument`'s ordered, enabled regions into the scalar
    per-point weight `docs/EDITOR.md` Sec 2 describes riding through each
    renderer's existing alpha-compositing (`w_edit`), plus the delete mask
    that removes points/Gaussians from the input before rasterisation
    entirely (`docs/decisions/ADR-0007-viewer-editing.md` "Deletions").
Invariants:
    - Starts from `default` (the gate's own per-point opinion;
      `trippy.constants.EDIT_GATE_DEFAULT_WEIGHT` = 1.0 = pure TRIPS
      everywhere, when the bundle has no gate) and applies `edits.order` in
      sequence, skipping disabled regions.
    - `op="blend"` moves the running weight towards `region.mix`, graded by
      the region's own membership weight (`trippy.edit.model.region_weight`:
      hard 0/1 for box/sphere/pointset, so a hard region fully replaces the
      weight where it applies -- "last write wins",
      `docs/EDITOR.md` Sec 1 "Ordering and overlap"; the lid's falloff/band
      softens this at its edge).
    - `op="fade"` multiplies the running weight towards `weight * mix`
      (graded the same way) -- "fade multiplies" per the milestone brief;
      for a hard region this is exactly `w *= mix` where the region applies.
    - `op="delete"` zeroes the running weight and sets `delete_mask`, using
      the region's HARD membership (`trippy.edit.model.region_contains`,
      ignoring the lid's falloff/band -- ADR-0007's literal "nothing
      renders below the plane inside the radius").
    - A point once deleted stays deleted for the rest of composition, even
      if a later region's `blend`/`fade` would otherwise touch it --
      `docs/EDITOR.md` Sec 1: "a delete region must always be able to
      override an earlier blend, never the reverse, or a deleted object
      could be un-deleted by scrolling a slider on an unrelated region".
    - `pointset` regions only make sense against the SAME point cloud whose
      row order produced their `point_ids` (points.npz's own order, per
      `docs/EDITOR.md` Sec 1); `compose_gaussian_weights` therefore skips
      them entirely rather than guess an unrelated row correspondence
      (`docs/decisions/ADR-0007-viewer-editing.md` "pointset regions do not
      survive distillation, by construction").
Units: weights are dimensionless in `[0, 1]` (0 = pure splat, 1 = pure
    TRIPS, docs/EDITOR.md's own convention).
Related docs: docs/EDITOR.md Sec 2 "Render integration";
    docs/decisions/ADR-0007-viewer-editing.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trippy.constants import EDIT_GATE_DEFAULT_WEIGHT
from trippy.edit.model import EditDocument, region_contains, region_weight

__all__ = [
    "ComposedWeights",
    "compose_gaussian_opacity_scale",
    "compose_gaussian_weights",
    "compose_point_weights",
    "compose_trips_weights",
]


@dataclass(frozen=True)
class ComposedWeights:
    """The result of composing an `EditDocument` against one point cloud.

    Attributes:
        weight: `(N,)` float64 in `[0, 1]`, the per-point `w_edit`
            (meaningless -- always 0.0 -- at indices where `delete_mask` is
            True; a deleted point has no weight because it does not render
            at all).
        delete_mask: `(N,)` bool, True where a `delete`-op region removed
            the point.
    """

    weight: np.ndarray
    delete_mask: np.ndarray


def compose_point_weights(
    edits: EditDocument,
    xyz: np.ndarray,
    default: float | np.ndarray = EDIT_GATE_DEFAULT_WEIGHT,
    skip_pointset: bool = False,
    ops: tuple[str, ...] | None = None,
) -> ComposedWeights:
    """Compose `edits.order`'s enabled regions against `xyz`, starting from `default`.

    Args:
        edits: the edit document (regions applied in `edits.order`).
        xyz: `(N, 3)` world-frame positions. For a `pointset` region to
            mean anything, row `i` here must be the same point `point_ids`
            containing `i` refers to.
        default: the gate's own per-point weight before any region applies
            -- a scalar (broadcast) or an `(N,)` array (a real per-pixel
            gate, once `feat/blend-gate` lands).
        skip_pointset: when True, `pointset` regions are ignored entirely
            (see module docstring; used by `compose_gaussian_weights`).
        ops: when given, a region whose `op` is not in this tuple is
            skipped entirely (as if disabled) -- used by
            `compose_gaussian_opacity_scale` to honour `delete`/`fade`
            against an already-distilled Gaussian PLY while ignoring
            `blend` (a splat-only artifact has no TRIPS-vs-splat mix for
            `blend` to target, docs/EDITOR.md Sec 5). `None` (default)
            honours every op, matching `compose_trips_weights`/
            `compose_gaussian_weights`'s existing behaviour exactly.

    Returns:
        `ComposedWeights` over the same `N` rows as `xyz`.

    Raises:
        ValueError: an enabled region has an `op` outside
            `trippy.constants.EDIT_REGION_OPS` (unreachable via `Region`'s
            own validation, guarded here only as a defensive check).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = xyz.shape[0]
    if np.isscalar(default):
        weight = np.full(n, float(default), dtype=np.float64)
    else:
        weight = np.array(default, dtype=np.float64, copy=True).reshape(n)
    delete_mask = np.zeros(n, dtype=bool)

    for region_id in edits.order:
        region = edits.regions_by_id.get(region_id)
        if region is None or not region.enabled:
            continue
        if skip_pointset and region.kind == "pointset":
            continue
        if ops is not None and region.op not in ops:
            continue
        active = ~delete_mask

        if region.op == "delete":
            hit = region_contains(region, xyz) & active
            delete_mask |= hit
            weight[hit] = 0.0
            continue

        m = region_weight(region, xyz)
        m = np.where(active, m, 0.0)
        if region.op == "blend":
            weight = weight * (1.0 - m) + region.mix * m
        elif region.op == "fade":
            weight = weight * (1.0 - m * (1.0 - region.mix))
        else:  # pragma: no cover -- Region.__post_init__ already rejects this
            raise ValueError(f"unknown region op {region.op!r}")

    return ComposedWeights(weight=weight, delete_mask=delete_mask)


def compose_trips_weights(
    edits: EditDocument,
    xyz: np.ndarray,
    default: float | np.ndarray = EDIT_GATE_DEFAULT_WEIGHT,
) -> ComposedWeights:
    """TRIPS points: every region kind applies (`pointset.point_ids` index this array's own rows)."""
    return compose_point_weights(edits, xyz, default=default, skip_pointset=False)


def compose_gaussian_weights(
    edits: EditDocument,
    xyz: np.ndarray,
    default: float | np.ndarray = EDIT_GATE_DEFAULT_WEIGHT,
) -> ComposedWeights:
    """Gaussian splat centres: `pointset` regions skipped (different row order, see module docstring)."""
    return compose_point_weights(edits, xyz, default=default, skip_pointset=True)


def compose_gaussian_opacity_scale(edits: EditDocument, xyz: np.ndarray) -> ComposedWeights:
    """`delete`/`fade` regions re-applied directly to an already-distilled Gaussian PLY.

    docs/EDITOR.md Sec 5: a `box`/`sphere`/`lid` region is pure world-space
    geometry (never `pointset`, which does not survive distillation, see
    `compose_gaussian_weights`'s own skip), so it can be re-applied straight
    to a finished distilled PLY's own `xyz` -- no re-distillation needed.
    Only `delete` (hard removal, `.delete_mask`) and `fade` (opacity
    scaling, `.weight` in `[0, 1]`, 1 = unchanged) make sense against a
    plain Gaussian artifact with no TRIPS-vs-splat mix to `blend` towards;
    `blend`-op regions are skipped entirely here (`ops=("delete", "fade")`)
    rather than silently doing nothing useful with `region.mix`.

    Returns:
        `ComposedWeights`: `.delete_mask` as usual, `.weight` starting from
        `1.0` (full opacity) and multiplied down by every enabled `fade`
        region's own graded membership exactly as
        `compose_point_weights`'s `fade` branch already does -- the caller
        (`trippy.edit.apply.apply_gaussian_ply_edits`) reads `.weight` as a
        per-surviving-row opacity multiplier, not a TRIPS/splat mix.
    """
    return compose_point_weights(edits, xyz, default=1.0, skip_pointset=True, ops=("delete", "fade"))
