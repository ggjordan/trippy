"""`edits.json`'s data model: `Region`, `EditDocument`, and world-space membership tests.

Module: trippy.edit.model
Purpose: the E1 milestone's data model (docs/EDITOR.md Sec 1) -- a `Region`
    (box/sphere/lid/pointset, an op, a mix, enabled/disabled), an
    `EditDocument` holding an ordered region list plus an append-only undo
    log with a cursor, and the vectorised world-space "is this point inside
    this region" tests every renderer/publish path needs (docs/EDITOR.md
    Sec 2: "Region membership itself is evaluated once per edit change ...
    O(points x regions)").
Invariants:
    - `Region.__post_init__` validates `kind`/`op`/`mix` and the kind's own
      `params` shape immediately on construction -- a `Region` that exists
      in memory is always well-formed; `EditDocument` never has to
      re-validate one it already holds.
    - `EditDocument`'s CURRENT state (`.regions`, `.order`) is always a pure
      function of replaying `.log[:.cursor]` from empty
      (`EditDocument._replay`) -- undo/redo are cursor moves
      (docs/EDITOR.md Sec 1 "Undo"), never inverse-operation bookkeeping,
      so an undo and its forward edit cannot drift apart.
    - `to_json()`/`from_json()` round-trip byte-for-byte through
      `save()`/`load()`: the file carries BOTH the materialised
      `regions`/`order` (for a reader, e.g. the Rust viewer, that has no
      reason to replay the log) AND the full `undo_stack` (log + cursor),
      and `from_json` cross-checks the two agree -- "closing and reopening
      the bundle preserves undo history" (docs/EDITOR.md Sec 1) means both
      halves of the file must actually agree, not merely both be present.
    - `box`'s `params` are `center`/`half_extents`/`quat` (an oriented box),
      a superset of docs/EDITOR.md Sec 1's axis-aligned `min`/`max`
      (identity quat recovers an axis-aligned box exactly) -- the task
      brief this module implements against asks for a rotated-box
      membership test, which an axis-aligned-only schema cannot express;
      recorded here rather than silently deviating from docs/EDITOR.md
      undocumented.
    - `lid`'s hard membership (`region_contains`, used by `op="delete"`) is
      the LITERAL "below the plane, inside the radius" clip ADR-0007
      describes, ignoring `falloff`/`band`. Its graded membership
      (`region_weight`, used by `op="blend"`/`"fade"`) ramps by
      `falloff`/`band` at the edge, mirroring (mirrored to the opposite
      side of the plane) the exact ramp shapes
      `~/Splats/tools/SURFACE_LID.md`'s training-time penalty uses
      (`band_i`, `region_i`) -- see `lid_membership`'s own docstring for
      the formula and why it is mirrored, not copied verbatim.
Units: world units (COLMAP world frame, docs/GEOMETRY.md); `mix` is
    dimensionless in [0, 1] (0 = pure splat, 1 = pure TRIPS).
Related docs: docs/EDITOR.md Sec 1 "Data model"; docs/decisions/
    ADR-0007-viewer-editing.md; `~/Splats/tools/SURFACE_LID.md` (read-only,
    the lid's fitted geometry and training-time penalty formula).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from trippy.constants import (
    EDIT_FORMAT,
    EDIT_REGION_ID_HEX_LEN,
    EDIT_REGION_KINDS,
    EDIT_REGION_OPS,
)

__all__ = [
    "EditDocument",
    "Region",
    "box_membership",
    "lid_membership",
    "new_region_id",
    "pointset_membership",
    "region_contains",
    "region_weight",
    "sphere_membership",
]


def new_region_id() -> str:
    """A fresh, never-reused region id, `"r-<hex>"` (docs/EDITOR.md Sec 1 example ids)."""
    return f"r-{uuid.uuid4().hex[:EDIT_REGION_ID_HEX_LEN]}"


# --- Region ------------------------------------------------------------------------


def _as_vec(value: Any, n: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{name} must be {n} numbers, got {value!r}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return arr


def _positive(value: Any, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return v


def _non_negative(value: Any, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return v


def _validate_box_params(params: dict[str, Any]) -> None:
    _as_vec(params.get("center"), 3, "box.center")
    half_extents = _as_vec(params.get("half_extents"), 3, "box.half_extents")
    if np.any(half_extents <= 0.0):
        raise ValueError(f"box.half_extents must all be > 0, got {half_extents.tolist()}")
    quat = _as_vec(params.get("quat", (1.0, 0.0, 0.0, 0.0)), 4, "box.quat")
    if float(np.dot(quat, quat)) < 1e-18:
        raise ValueError(f"box.quat must not be ~zero, got {quat.tolist()}")


def _validate_sphere_params(params: dict[str, Any]) -> None:
    _as_vec(params.get("center"), 3, "sphere.center")
    _positive(params.get("radius"), "sphere.radius")


def _validate_lid_params(params: dict[str, Any]) -> None:
    up = _as_vec(params.get("up"), 3, "lid.up")
    if float(np.dot(up, up)) < 1e-18:
        raise ValueError(f"lid.up must not be ~zero, got {up.tolist()}")
    height = params.get("height")
    if height is None or not np.isfinite(float(height)):
        raise ValueError(f"lid.height must be finite, got {height!r}")
    _as_vec(params.get("center"), 3, "lid.center")
    _positive(params.get("radius"), "lid.radius")
    _non_negative(params.get("falloff"), "lid.falloff")
    _non_negative(params.get("band"), "lid.band")


def _validate_pointset_params(params: dict[str, Any]) -> None:
    ids = params.get("point_ids")
    if ids is None or not isinstance(ids, (list, tuple)):
        raise ValueError(f"pointset.point_ids must be a list, got {ids!r}")
    arr = np.asarray(ids, dtype=np.int64) if ids else np.zeros(0, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError("pointset.point_ids must be a flat list of integers")
    if arr.size and np.any(arr < 0):
        raise ValueError("pointset.point_ids must all be >= 0")


_PARAM_VALIDATORS = {
    "box": _validate_box_params,
    "sphere": _validate_sphere_params,
    "lid": _validate_lid_params,
    "pointset": _validate_pointset_params,
}


@dataclass
class Region:
    """One `edits.json` region -- docs/EDITOR.md Sec 1's `Region` table.

    Attributes:
        id: stable identity, never reused even after removal (the undo log
            keys off it).
        name: shown in the Regions panel.
        kind: one of `trippy.constants.EDIT_REGION_KINDS`
            ("box"/"sphere"/"lid"/"pointset"); selects which shape
            `params` must have (see the module docstring for `box`'s
            deliberate deviation from docs/EDITOR.md's axis-aligned-only
            schema).
        params: kind-specific geometry, validated in `__post_init__`:
            - box: `center` (3), `half_extents` (3, all > 0), `quat`
              (4, `(w, x, y, z)`, default identity -- an oriented box).
            - sphere: `center` (3), `radius` (> 0).
            - lid: `up` (3), `height` (float), `center` (3), `radius`
              (> 0), `falloff` (>= 0), `band` (>= 0) -- the same six
              numbers as `~/Splats/tools/SURFACE_LID.md`'s `--lid-*` flags.
            - pointset: `point_ids` (list of non-negative ints), indices
              into the SAME point cloud's own row order (points.npz's own
              order for TRIPS points; never a Gaussian PLY's rows -- see
              `trippy.edit.weights` module docstring).
        mix: `[0, 1]`; 0 = pure splat, 1 = pure TRIPS. Ignored by
            `op="delete"`.
        op: "blend" (mix wins where the region applies), "delete" (hard
            removal upstream of rendering), or "fade" (like blend, but
            multiplies the running weight rather than replacing it --
            `trippy.edit.weights`).
        enabled: soft "off" without deleting the region.

    Raises:
        ValueError: `kind`/`op` unrecognised, `mix` outside `[0, 1]`, `id`
            empty, or `params` fails the kind's own shape/range checks.
    """

    id: str
    name: str
    kind: str
    params: dict[str, Any]
    mix: float = 1.0
    op: str = "blend"
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Region.id must be non-empty")
        if self.kind not in EDIT_REGION_KINDS:
            raise ValueError(f"Region.kind must be one of {EDIT_REGION_KINDS}, got {self.kind!r}")
        if self.op not in EDIT_REGION_OPS:
            raise ValueError(f"Region.op must be one of {EDIT_REGION_OPS}, got {self.op!r}")
        if not (0.0 <= float(self.mix) <= 1.0):
            raise ValueError(f"Region.mix must be in [0, 1], got {self.mix!r}")
        self.mix = float(self.mix)
        self.enabled = bool(self.enabled)
        _PARAM_VALIDATORS[self.kind](self.params)

    def to_json(self) -> dict[str, Any]:
        """This region as the `regions[]` entry of `edits.json` (docs/EDITOR.md Sec 1)."""
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "enabled": bool(self.enabled),
            "mix": float(self.mix),
            "op": self.op,
            "params": dict(self.params),
        }

    @staticmethod
    def from_json(doc: dict[str, Any]) -> Region:
        """Inverse of `to_json`. Raises `ValueError`/`KeyError` on a malformed entry."""
        return Region(
            id=doc["id"],
            name=doc.get("name", ""),
            kind=doc["kind"],
            params=dict(doc.get("params", {})),
            mix=float(doc.get("mix", 1.0)),
            op=doc.get("op", "blend"),
            enabled=bool(doc.get("enabled", True)),
        )


# --- membership tests, vectorised (numpy) -------------------------------------------


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion `(w, x, y, z)` -> `(3, 3)` rotation matrix, float64.

    A generic object-space rotation for an oriented box -- NOT
    `trippy.geom.xform_a.qvec2R`'s COLMAP world->camera convention (that
    function answers a different question: a camera's orientation, not an
    arbitrary box's). `q` need not be pre-normalised; this normalises it.

    Raises:
        ValueError: `q`'s norm is ~zero (not a rotation).
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError(f"quaternion has ~zero norm: {q.tolist()}")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def box_membership(
    xyz: np.ndarray,
    center: Any,
    half_extents: Any,
    quat: Any = (1.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """Hard-edge oriented-box membership: `1.0` inside, `0.0` outside, per point.

    `local = R^T @ (p - center)`; a point is inside iff every local axis
    satisfies `|local_i| <= half_extents_i`. `quat = (1, 0, 0, 0)` (the
    default) is an axis-aligned box.

    Args:
        xyz: `(N, 3)` world-frame positions.
        center: `(3,)` world-frame box centre.
        half_extents: `(3,)`, all > 0, box half-size along its own local axes.
        quat: `(4,)` `(w, x, y, z)`, the box's own orientation (need not be
            pre-normalised).

    Returns:
        `(N,)` float64, `1.0` or `0.0`.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    half_extents = np.asarray(half_extents, dtype=np.float64).reshape(3)
    rot = _quat_to_rotmat(np.asarray(quat, dtype=np.float64))
    # (v @ R) is R^T @ v per row -- see _quat_to_rotmat's caller-facing contract above.
    local = (xyz - center) @ rot
    inside = np.all(np.abs(local) <= half_extents, axis=1)
    return inside.astype(np.float64)


def sphere_membership(xyz: np.ndarray, center: Any, radius: float) -> np.ndarray:
    """Hard-edge sphere membership: `1.0` inside (`||p - center|| <= radius`), else `0.0`."""
    xyz = np.asarray(xyz, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    dist2 = np.sum((xyz - center) ** 2, axis=1)
    return (dist2 <= float(radius) ** 2).astype(np.float64)


def lid_membership(
    xyz: np.ndarray,
    up: Any,
    height: float,
    center: Any,
    radius: float,
    falloff: float,
    band: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The Karekare-pool lid's world-space test: "below the plane, inside the radius".

    The plane is `dot(up, x) == height`; a point's signed height is
    `h(x) = dot(up, x) - height` (`> 0` above the plane, `< 0` below).
    `rho(x)` is the horizontal distance from `center`, measured
    perpendicular to `up`.

    Returns two views of the same geometry, matching the two ways
    `docs/decisions/ADR-0007-viewer-editing.md`'s "2." uses it:

    - `inside` (bool): the LITERAL hard clip, `h(x) <= 0 AND rho(x) <=
      radius`, ignoring `falloff`/`band` entirely -- what `op="delete"`
      uses, so "nothing renders below the plane inside the radius" is
      exactly true at every `falloff`/`band` setting.
    - `weight` (float64 in `[0, 1]`): a falloff-graded weight for
      `op="blend"`/`"fade"`'s soft edge, `vertical * horizontal`:
        `vertical  = clip(-h(x) / band, 0, 1)`      (0 at the plane, 1 once
                                                       `band` below it)
        `horizontal = clip((radius + falloff - rho(x)) / falloff, 0, 1)`
                                                     (1 inside `radius`, 0
                                                       once `falloff` past it)
      `band == 0` collapses `vertical` to a hard step at the plane;
      `falloff == 0` collapses `horizontal` to a hard step at `radius`.
      This is the exact ramp shape `~/Splats/tools/SURFACE_LID.md`'s
      training-time `band_i`/`region_i` terms use for its *opacity*
      penalty, MIRRORED across the plane (that penalty targets haze ABOVE
      the plane; the editor's hard-clip use case is the volume BELOW it,
      per ADR-0007's own "same geometry, different action, do not conflate
      them" decision) -- reusing the shape, not the sign, of an
      already-calibrated ramp.

    Args:
        xyz: `(N, 3)` world-frame positions.
        up: `(3,)` plane normal (need not be unit; normalised here).
        height: plane offset along `up`.
        center: `(3,)` centre used for the horizontal radius test.
        radius: horizontal radius, world units, > 0.
        falloff: horizontal ramp width beyond `radius`, world units, >= 0.
        band: vertical ramp depth below the plane, world units, >= 0.

    Returns:
        `(inside, weight)`, both `(N,)`.

    Raises:
        ValueError: `up` has ~zero norm.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64).reshape(3)
    up_norm = float(np.linalg.norm(up))
    if up_norm < 1e-12:
        raise ValueError(f"lid.up has ~zero norm: {up.tolist()}")
    up = up / up_norm
    center = np.asarray(center, dtype=np.float64).reshape(3)
    radius = float(radius)
    falloff = float(falloff)
    band = float(band)

    signed = xyz @ up - float(height)
    rel = xyz - center
    along = rel @ up
    perp = rel - np.outer(along, up)
    rho = np.linalg.norm(perp, axis=1)

    inside = (signed <= 0.0) & (rho <= radius)

    if band > 0.0:
        vertical = np.clip((-signed) / band, 0.0, 1.0)
    else:
        vertical = (signed <= 0.0).astype(np.float64)
    if falloff > 0.0:
        horizontal = np.clip((radius + falloff - rho) / falloff, 0.0, 1.0)
    else:
        horizontal = (rho <= radius).astype(np.float64)
    weight = vertical * horizontal
    return inside, weight


def pointset_membership(n_points: int, point_ids: Any) -> np.ndarray:
    """Hard-edge membership by index: `1.0` at every id in `point_ids`, else `0.0`.

    Ids `>= n_points` are silently ignored (a `pointset` region saved
    against a point cloud that has since shrunk should not raise -- the
    same "widen, never crash" posture `trippy.train.prune.apply_min_points`
    takes elsewhere in this codebase).
    """
    weight = np.zeros(int(n_points), dtype=np.float64)
    ids = np.asarray(point_ids, dtype=np.int64) if len(point_ids) else np.zeros(0, dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < n_points)]
    weight[ids] = 1.0
    return weight


def region_weight(region: Region, xyz: np.ndarray) -> np.ndarray:
    """Per-point membership weight in `[0, 1]`: hard 0/1 for box/sphere/pointset, graded for lid.

    This is the weight `op="blend"`/`"fade"` composite against (see
    `trippy.edit.weights`); for `op="delete"`'s hard clip use
    `region_contains` instead (identical to this for every kind except
    `lid`, where `lid_membership`'s docstring explains the difference).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    p = region.params
    if region.kind == "box":
        return box_membership(xyz, p["center"], p["half_extents"], p.get("quat", (1.0, 0.0, 0.0, 0.0)))
    if region.kind == "sphere":
        return sphere_membership(xyz, p["center"], p["radius"])
    if region.kind == "lid":
        _inside, weight = lid_membership(xyz, p["up"], p["height"], p["center"], p["radius"], p["falloff"], p["band"])
        return weight
    if region.kind == "pointset":
        return pointset_membership(xyz.shape[0], p["point_ids"])
    raise ValueError(f"unknown region kind {region.kind!r}")  # pragma: no cover -- guarded by Region.__post_init__


def region_contains(region: Region, xyz: np.ndarray) -> np.ndarray:
    """Hard boolean membership: the exact footprint `op="delete"` removes.

    Identical to `region_weight(region, xyz) > 0` for every kind except
    `lid`, where this is the literal "below the plane, inside the radius"
    clip (`lid_membership`'s `inside`), NOT the falloff-graded weight.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if region.kind == "lid":
        p = region.params
        inside, _weight = lid_membership(xyz, p["up"], p["height"], p["center"], p["radius"], p["falloff"], p["band"])
        return inside
    return region_weight(region, xyz) > 0.0


# --- EditDocument: ordered regions + append-only undo log ---------------------------


_ENTRY_TYPES = ("add_region", "remove_region", "update_region", "reorder")


@dataclass
class EditDocument:
    """`edits.json`'s full document: regions, paint order, and the undo log.

    The CURRENT `regions`/`order` are always derived by replaying
    `log[:cursor]` from empty (`_replay`, called after every mutation) --
    see the module docstring's "Invariants". Construct via `EditDocument.new()`
    (empty document) or `EditDocument.load(path)` (existing file); mutate via
    `add_region`/`remove_region`/`update_region`/`reorder`, not by touching
    `.log`/`.cursor` directly.

    Attributes:
        format: always `trippy.constants.EDIT_FORMAT`.
        bundle_format: cross-checked against the bundle's own `bundle.json`
            `"format"` field at `validate()` time (docs/EDITOR.md Sec 1);
            `""` means "not yet known" and skips the check.
        log: append-only list of diff dicts (`{"type": ..., ...}`), one of
            `_ENTRY_TYPES`. Truncated to `cursor` and appended to by every
            mutation (the standard "new edit after undo discards the
            redone future" rule).
        cursor: index into `log`; `undo`/`redo` move it by one.
        regions: derived, in paint order (`order`).
        order: derived, region ids in paint order (later wins on overlap).
    """

    format: str = EDIT_FORMAT
    bundle_format: str = ""
    log: list[dict[str, Any]] = field(default_factory=list)
    cursor: int = 0
    regions: list[Region] = field(default_factory=list, init=False, repr=False)
    order: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._replay()

    @property
    def regions_by_id(self) -> dict[str, Region]:
        return {r.id: r for r in self.regions}

    def _replay(self) -> None:
        if not (0 <= self.cursor <= len(self.log)):
            raise ValueError(f"undo cursor {self.cursor} out of range for a log of length {len(self.log)}")
        regions: dict[str, Region] = {}
        order: list[str] = []
        for entry in self.log[: self.cursor]:
            etype = entry.get("type")
            if etype == "add_region":
                region = Region.from_json(entry["region"])
                regions[region.id] = region
                index = entry.get("index")
                if index is None or index >= len(order):
                    order.append(region.id)
                else:
                    order.insert(max(0, index), region.id)
            elif etype == "remove_region":
                rid = entry["id"]
                regions.pop(rid, None)
                if rid in order:
                    order.remove(rid)
            elif etype == "update_region":
                rid = entry["id"]
                if rid in regions:
                    updated = regions[rid].to_json()
                    updated.update(entry.get("changes", {}))
                    regions[rid] = Region.from_json(updated)
            elif etype == "reorder":
                new_order = list(entry["order"])
                if set(new_order) != set(order):
                    raise ValueError("'reorder' entry must permute the current region ids exactly")
                order = new_order
            else:
                raise ValueError(f"unknown undo-log entry type {etype!r}; expected one of {_ENTRY_TYPES}")
        self.regions = [regions[rid] for rid in order]
        self.order = order

    def _push(self, entry: dict[str, Any]) -> None:
        self.log = [*self.log[: self.cursor], entry]
        self.cursor = len(self.log)
        self._replay()

    def add_region(self, region: Region, index: int | None = None) -> None:
        """Append (or insert at `index`) a new region and push it onto the undo log."""
        self._push({"type": "add_region", "region": region.to_json(), "index": index})

    def remove_region(self, region_id: str) -> None:
        """Remove a region by id (a no-op push if it is already absent -- undo still records it)."""
        self._push({"type": "remove_region", "id": region_id})

    def update_region(self, region_id: str, **changes: Any) -> None:
        """Edit one or more fields of an existing region (`mix`, `enabled`, `op`, `name`, `params`, ...)."""
        if region_id not in self.regions_by_id:
            raise KeyError(f"no region {region_id!r} to update")
        self._push({"type": "update_region", "id": region_id, "changes": dict(changes)})

    def reorder(self, new_order: list[str]) -> None:
        """Set the paint order to an exact permutation of the current region ids."""
        self._push({"type": "reorder", "order": list(new_order)})

    def undo(self) -> bool:
        """Move the cursor back one entry. Returns False (no-op) if already at the start."""
        if self.cursor == 0:
            return False
        self.cursor -= 1
        self._replay()
        return True

    def redo(self) -> bool:
        """Move the cursor forward one entry. Returns False (no-op) if already at the end."""
        if self.cursor >= len(self.log):
            return False
        self.cursor += 1
        self._replay()
        return True

    def validate(self, expected_bundle_format: str | None = None) -> None:
        """Cheap consistency checks beyond what construction already guarantees.

        Raises:
            ValueError: `format` is not `EDIT_FORMAT`; `bundle_format` is
                set and disagrees with `expected_bundle_format`; duplicate
                region ids; or `order` does not match the region id set
                (the latter two should be unreachable via the public
                mutation API, but a hand-edited `edits.json` can violate
                them).
        """
        if self.format != EDIT_FORMAT:
            raise ValueError(f"edits.json format {self.format!r} != expected {EDIT_FORMAT!r}")
        if expected_bundle_format and self.bundle_format and self.bundle_format != expected_bundle_format:
            raise ValueError(
                f"edits.json bundle_format {self.bundle_format!r} does not match "
                f"bundle.json format {expected_bundle_format!r}"
            )
        ids = [r.id for r in self.regions]
        if len(ids) != len(set(ids)):
            raise ValueError("edits.json has duplicate region ids")
        if set(self.order) != set(ids):
            raise ValueError("edits.json 'order' does not match its region ids")

    def to_json(self) -> dict[str, Any]:
        """The full `edits.json` document (docs/EDITOR.md Sec 1)."""
        return {
            "format": self.format,
            "bundle_format": self.bundle_format,
            "regions": [r.to_json() for r in self.regions],
            "order": list(self.order),
            "undo_stack": {"cursor": self.cursor, "log": self.log},
        }

    @staticmethod
    def from_json(doc: dict[str, Any]) -> EditDocument:
        """Inverse of `to_json`. Cross-checks the materialised `regions`/`order` against the log.

        Raises:
            ValueError: unsupported `format`, or the file's own
                `regions`/`order` disagree with replaying `undo_stack.log`
                up to `undo_stack.cursor` (a corrupted or hand-edited file).
        """
        fmt = doc.get("format", EDIT_FORMAT)
        if fmt != EDIT_FORMAT:
            raise ValueError(f"unsupported edits.json format {fmt!r}, expected {EDIT_FORMAT!r}")
        undo_stack = doc.get("undo_stack", {})
        log = list(undo_stack.get("log", []))
        cursor = int(undo_stack.get("cursor", len(log)))
        edits = EditDocument(format=fmt, bundle_format=doc.get("bundle_format", ""), log=log, cursor=cursor)
        saved_order = doc.get("order")
        if saved_order is not None and list(saved_order) != edits.order:
            raise ValueError("edits.json 'order' does not match replaying 'undo_stack.log' up to 'cursor'")
        return edits

    @staticmethod
    def new(bundle_format: str = "") -> EditDocument:
        """An empty document (no regions, no history)."""
        return EditDocument(bundle_format=bundle_format)

    @staticmethod
    def load(path: str | Path) -> EditDocument:
        """Read and validate an `edits.json` file."""
        return EditDocument.from_json(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> Path:
        """Write this document to `path` (parents created if missing)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")
        return path
