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
      ONE exception: `save()` externalises a large `brush` region's `cells`/
      `weights` into an `.npz` sidecar in the WRITTEN copy of `regions[]`
      only (`EditDocument._externalize_brush`, see `EDIT_BRUSH_NPZ_CELL_
      THRESHOLD`'s comment) -- `self.log`/`self.regions` in memory, and
      therefore `to_json()` itself, stay fully literal; only a caller that
      re-parses the WRITTEN FILE's `regions[]` (the Rust viewer; Python's own
      `load()` does not, see below) ever sees the reference form.
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
    - `brush`'s membership (`brush_membership`) is a plain voxel lookup: a
      point's own cell (`floor((p - origin) / cell_size)`) either is or is
      not in the region's `cells` set, and the returned weight is that
      cell's own `weights` entry (default `1.0` when the region carries no
      `weights` at all). `region_weight`/`region_contains` dispatch to it
      exactly like every other kind (`region_contains` falls back to
      `region_weight(...) > 0`, brush needs no lid-style hard/graded split)
      -- `trippy.edit.weights`/`trippy.edit.apply`/`trippy.edit.checkpoint`
      therefore need NO brush-specific code at all, per this task's brief
      ("it is just another membership"). `paint_sphere`/`paint_along`/
      `erase` are the authoring helpers a brush tool calls; each returns a
      NEW `Region` (this module's functions are pure, like every membership
      test above) with `cells`/`weights` updated by a box-sphere
      intersection test against every voxel in the sphere's bounding box
      (`_sphere_touched_cells`), not merely voxels whose CENTRE falls inside
      the sphere -- so a brush stroke paints every cell the sphere actually
      overlaps, matching what a person watching the stroke would expect.
    - `Region.source` (`{"tool": str, ...}` or `None`) records which tool
      produced a region and with what prompt/parameters, purely for display
      (a future Named Objects panel) and provenance -- it plays no part in
      any membership test or composition. `auto_region_name` is the sibling
      naming convention (docs/EDITOR.md Sec 1 "Named regions"): a tool that
      does not receive an explicit name from its caller gets
      `"<tool>[-<detail>]-<n>"`, `n` one more than the highest existing
      `-<digits>` suffix among the document's OWN region names -- so a
      session's tool-authored regions read as one continuously numbered
      list (`click-1`, `sam-box-IMG_3703-2`, `shade-clouds-3`, `brush-4`)
      regardless of which tool made each one, and the scheme self-heals
      after a region is removed or undone (it is derived from whatever
      names currently exist, never a counter stored anywhere).
Units: world units (COLMAP world frame, docs/GEOMETRY.md); `mix` is
    dimensionless in [0, 1] (0 = pure splat, 1 = pure TRIPS).
Related docs: docs/EDITOR.md Sec 1 "Data model"; docs/decisions/
    ADR-0007-viewer-editing.md; `~/Splats/tools/SURFACE_LID.md` (read-only,
    the lid's fitted geometry and training-time penalty formula).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from trippy.constants import (
    EDIT_AUTO_NAME_COUNTER_PATTERN,
    EDIT_BRUSH_CELL_INT32_ABS_MAX,
    EDIT_BRUSH_NPZ_CELL_THRESHOLD,
    EDIT_BRUSH_NPZ_FILENAME_FMT,
    EDIT_FORMAT,
    EDIT_REGION_ID_HEX_LEN,
    EDIT_REGION_KINDS,
    EDIT_REGION_OPS,
)

__all__ = [
    "EditDocument",
    "Region",
    "auto_region_name",
    "box_membership",
    "brush_membership",
    "erase",
    "lid_membership",
    "new_region_id",
    "paint_along",
    "paint_sphere",
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


def _validate_brush_params(params: dict[str, Any]) -> None:
    _as_vec(params.get("origin"), 3, "brush.origin")
    _positive(params.get("cell_size"), "brush.cell_size")
    cells = params.get("cells")
    if cells is None or not isinstance(cells, (list, tuple)):
        raise ValueError(f"brush.cells must be a list, got {cells!r}")
    cells_arr = _brush_cells_array(cells)
    if cells_arr.size and np.any(np.abs(cells_arr) > EDIT_BRUSH_CELL_INT32_ABS_MAX):
        raise ValueError(f"brush.cells must fit in a signed int32 (|i| <= {EDIT_BRUSH_CELL_INT32_ABS_MAX})")
    weights = params.get("weights")
    if weights is not None:
        if not isinstance(weights, (list, tuple)):
            raise ValueError(f"brush.weights must be a list, got {weights!r}")
        w = np.asarray(weights, dtype=np.float64) if weights else np.zeros(0, dtype=np.float64)
        if w.shape != (cells_arr.shape[0],):
            raise ValueError(f"brush.weights must have one entry per cell ({cells_arr.shape[0]}), got {len(weights)}")
        if w.size and (np.any(w < 0.0) or np.any(w > 1.0) or not np.all(np.isfinite(w))):
            raise ValueError("brush.weights must all be finite numbers in [0, 1]")


_PARAM_VALIDATORS = {
    "box": _validate_box_params,
    "sphere": _validate_sphere_params,
    "lid": _validate_lid_params,
    "pointset": _validate_pointset_params,
    "brush": _validate_brush_params,
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
        source: `{"tool": str, ...}` or `None` -- which tool produced this
            region (and, e.g., its prompt/parameters), for a future Named
            Objects panel; `None` for a hand-authored region (`trippy edits
            add-box`/`add-sphere`/`add-lid`/`add-brush`). Never read by any
            membership test or composition (module docstring).

    Raises:
        ValueError: `kind`/`op` unrecognised, `mix` outside `[0, 1]`, `id`
            empty, `source` neither a dict nor `None`, or `params` fails
            the kind's own shape/range checks.
    """

    id: str
    name: str
    kind: str
    params: dict[str, Any]
    mix: float = 1.0
    op: str = "blend"
    enabled: bool = True
    source: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Region.id must be non-empty")
        if self.kind not in EDIT_REGION_KINDS:
            raise ValueError(f"Region.kind must be one of {EDIT_REGION_KINDS}, got {self.kind!r}")
        if self.op not in EDIT_REGION_OPS:
            raise ValueError(f"Region.op must be one of {EDIT_REGION_OPS}, got {self.op!r}")
        if not (0.0 <= float(self.mix) <= 1.0):
            raise ValueError(f"Region.mix must be in [0, 1], got {self.mix!r}")
        if self.source is not None and not isinstance(self.source, dict):
            raise ValueError(f"Region.source must be a dict or None, got {self.source!r}")
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
            "source": dict(self.source) if self.source is not None else None,
        }

    @staticmethod
    def from_json(doc: dict[str, Any]) -> Region:
        """Inverse of `to_json`. Raises `ValueError`/`KeyError` on a malformed entry."""
        source = doc.get("source")
        return Region(
            id=doc["id"],
            name=doc.get("name", ""),
            kind=doc["kind"],
            params=dict(doc.get("params", {})),
            mix=float(doc.get("mix", 1.0)),
            op=doc.get("op", "blend"),
            enabled=bool(doc.get("enabled", True)),
            source=dict(source) if source is not None else None,
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


def _brush_cells_array(cells: Any) -> np.ndarray:
    """`cells` (a list of `[i, j, k]`) -> `(M, 3)` int64, `(0, 3)` if empty."""
    if not len(cells):
        return np.zeros((0, 3), dtype=np.int64)
    arr = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    return arr


#: Structured dtype three int64 fields are viewed as to sort/search a voxel-cell
#: set as ONE key. numpy compares a structured/void array FIELD BY FIELD (like a
#: Python tuple), not byte-by-byte, so this sorts and searches correctly even
#: though `i`/`j`/`k` may be negative -- verified directly (`np.sort`/
#: `np.searchsorted` on a signed structured array agree with sorting the plain
#: `(i, j, k)` tuples).
_BRUSH_CELL_KEY_DTYPE = np.dtype([("i", np.int64), ("j", np.int64), ("k", np.int64)])


def _brush_cell_key(idx: np.ndarray) -> np.ndarray:
    """`(N, 3)` int64 -> `(N,)` structured keys, one per row (see `_BRUSH_CELL_KEY_DTYPE`)."""
    return np.ascontiguousarray(idx, dtype=np.int64).view(_BRUSH_CELL_KEY_DTYPE).reshape(-1)


def _brush_voxel_index(xyz: np.ndarray, origin: Any, cell_size: float) -> np.ndarray:
    """World-space `(N, 3)` -> `(N, 3)` int64 voxel index, `floor((p - origin) / cell_size)`."""
    xyz = np.asarray(xyz, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    return np.floor((xyz - origin) / float(cell_size)).astype(np.int64)


def _brush_sorted_lookup(cells: Any, weights: Any) -> tuple[np.ndarray, np.ndarray]:
    """De-duplicated (last entry wins), sorted `(keys, weights)` for `cells`/`weights`.

    A region's `cells` list is a SET in spirit (docs/EDITOR.md Sec 1), but
    nothing stops a hand-edited file or an authoring helper from repeating a
    cell; the LAST occurrence wins, matching `EditDocument`'s own
    "later entries win" ordering convention (Sec 1 "Ordering and overlap").
    """
    cells_arr = _brush_cells_array(cells)
    if cells_arr.shape[0] == 0:
        return np.zeros(0, dtype=_BRUSH_CELL_KEY_DTYPE), np.zeros(0, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1) if weights is not None else np.ones(cells_arr.shape[0])
    dedup: dict[tuple[int, int, int], float] = {}
    for row, wi in zip(cells_arr.tolist(), w.tolist(), strict=True):
        dedup[(row[0], row[1], row[2])] = float(wi)
    keys = _brush_cell_key(np.asarray(list(dedup.keys()), dtype=np.int64).reshape(-1, 3))
    vals = np.asarray(list(dedup.values()), dtype=np.float64)
    order = np.argsort(keys)
    return keys[order], vals[order]


def brush_membership(xyz: np.ndarray, origin: Any, cell_size: float, cells: Any, weights: Any = None) -> np.ndarray:
    """Sparse-voxel membership: a point's own cell looked up in the region's occupied set.

    Args:
        xyz: `(N, 3)` world-frame positions.
        origin: `(3,)` world-frame corner the voxel grid is measured from.
        cell_size: voxel edge length, world units, > 0.
        cells: occupied cell indices, `[[i, j, k], ...]` (a set, not a dense
            array -- docs/EDITOR.md Sec 1).
        weights: optional per-cell weight, `[w0, w1, ...]` parallel to
            `cells`, each in `[0, 1]`; `None` (the common case -- a painted,
            not graded, brush) means every occupied cell has weight `1.0`.

    Returns:
        `(N,)` float64 in `[0, 1]`: the occupied cell's own weight at every
        point whose voxel is in `cells`, else `0.0`.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    result = np.zeros(xyz.shape[0], dtype=np.float64)
    sorted_keys, sorted_vals = _brush_sorted_lookup(cells, weights)
    if sorted_keys.size == 0 or xyz.shape[0] == 0:
        return result
    query = _brush_cell_key(_brush_voxel_index(xyz, origin, cell_size))
    pos = np.clip(np.searchsorted(sorted_keys, query), 0, sorted_keys.size - 1)
    match = sorted_keys[pos] == query
    result[match] = sorted_vals[pos[match]]
    return result


def _sphere_touched_cells(origin: Any, cell_size: float, center: Any, radius: float) -> np.ndarray:
    """Every voxel cell a world-space sphere actually OVERLAPS (box-sphere intersection).

    Not "voxels whose centre is inside the sphere": a cell is touched when
    the CLOSEST point of its own axis-aligned box (in world space) is within
    `radius` of `center`, so a brush stroke paints every cell the sphere
    visibly overlaps, matching what painting it would look like.

    Returns:
        `(M, 3)` int64 cell indices, `(0, 3)` if the sphere touches nothing
        (e.g. `radius <= 0`).
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    cell_size = float(cell_size)
    radius = float(radius)
    if radius <= 0.0:
        return np.zeros((0, 3), dtype=np.int64)

    lo = np.floor((center - radius - origin) / cell_size).astype(np.int64)
    hi = np.floor((center + radius - origin) / cell_size).astype(np.int64)
    ii, jj, kk = np.meshgrid(
        np.arange(lo[0], hi[0] + 1),
        np.arange(lo[1], hi[1] + 1),
        np.arange(lo[2], hi[2] + 1),
        indexing="ij",
    )
    idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
    if idx.shape[0] == 0:
        return idx
    cell_min = origin + idx.astype(np.float64) * cell_size
    cell_max = cell_min + cell_size
    closest = np.clip(center, cell_min, cell_max)
    dist2 = np.sum((closest - center) ** 2, axis=1)
    return idx[dist2 <= radius**2]


def _merge_brush_cells(
    existing_cells: Any, existing_weights: Any, new_cells: np.ndarray, weight: float
) -> tuple[list[list[int]], list[float] | None]:
    """`existing_cells`/`existing_weights` with `new_cells` added at `max(existing, weight)`.

    `max`, not "overwrite": repeated strokes over the same cell only ever
    strengthen it (an `erase` is the only way to reduce a cell's weight),
    which matches how a paint brush is expected to behave. Returns
    `weights=None` when every resulting weight is (approximately) `1.0`, so
    a plain (ungraded) brush never carries a redundant all-ones array.
    """
    weight = float(weight)
    if not (0.0 <= weight <= 1.0):
        raise ValueError(f"weight must be in [0, 1], got {weight!r}")
    table: dict[tuple[int, int, int], float] = {}
    existing_arr = _brush_cells_array(existing_cells)
    if existing_arr.shape[0]:
        ew = (
            np.asarray(existing_weights, dtype=np.float64).reshape(-1)
            if existing_weights is not None
            else np.ones(existing_arr.shape[0])
        )
        for row, wi in zip(existing_arr.tolist(), ew.tolist(), strict=True):
            table[(row[0], row[1], row[2])] = float(wi)
    for row in new_cells.tolist():
        key = (row[0], row[1], row[2])
        table[key] = max(table.get(key, 0.0), weight)
    cells_out = [list(key) for key in table]
    weights_out = list(table.values())
    if weights_out and all(abs(w - 1.0) < 1e-12 for w in weights_out):
        return cells_out, None
    return cells_out, (weights_out if weights_out else None)


def _remove_brush_cells(existing_cells: Any, existing_weights: Any, remove: np.ndarray) -> tuple[list[list[int]], list[float] | None]:
    """`existing_cells`/`existing_weights` with every cell in `remove` dropped."""
    drop = {(row[0], row[1], row[2]) for row in remove.tolist()}
    table: dict[tuple[int, int, int], float] = {}
    existing_arr = _brush_cells_array(existing_cells)
    if existing_arr.shape[0]:
        ew = (
            np.asarray(existing_weights, dtype=np.float64).reshape(-1)
            if existing_weights is not None
            else np.ones(existing_arr.shape[0])
        )
        for row, wi in zip(existing_arr.tolist(), ew.tolist(), strict=True):
            key = (row[0], row[1], row[2])
            if key not in drop:
                table[key] = float(wi)
    cells_out = [list(key) for key in table]
    weights_out = list(table.values())
    if weights_out and all(abs(w - 1.0) < 1e-12 for w in weights_out):
        return cells_out, None
    return cells_out, (weights_out if weights_out else None)


def _with_brush_params(region: Region, cells: list[list[int]], weights: list[float] | None) -> Region:
    if region.kind != "brush":
        raise ValueError(f"expected a brush region, got kind={region.kind!r}")
    params = dict(region.params)
    params["cells"] = cells
    if weights is not None:
        params["weights"] = weights
    else:
        params.pop("weights", None)
    return Region(
        id=region.id,
        name=region.name,
        kind=region.kind,
        params=params,
        mix=region.mix,
        op=region.op,
        enabled=region.enabled,
        source=region.source,
    )


def paint_sphere(region: Region, center: Any, radius: float, weight: float = 1.0) -> Region:
    """A new `brush` `Region` with every voxel cell a sphere touches added (or strengthened).

    Args:
        region: an existing `brush`-kind Region (its `origin`/`cell_size`
            define the voxel grid every cell is painted into).
        center: `(3,)` world-frame sphere centre.
        radius: sphere radius, world units.
        weight: the painted cell weight (`_merge_brush_cells`: existing
            cells only ever get stronger, `max(existing, weight)`).

    Returns:
        A NEW `Region` (this module's functions are pure); `region` itself
        is never mutated.

    Raises:
        ValueError: `region.kind != "brush"`, or `weight` outside `[0, 1]`.
    """
    if region.kind != "brush":
        raise ValueError(f"paint_sphere needs a brush region, got kind={region.kind!r}")
    p = region.params
    touched = _sphere_touched_cells(p["origin"], p["cell_size"], center, radius)
    cells, weights = _merge_brush_cells(p.get("cells", []), p.get("weights"), touched, weight)
    return _with_brush_params(region, cells, weights)


def paint_along(region: Region, points: Any, radius: float, weight: float = 1.0) -> Region:
    """A brush STROKE: the union of `paint_sphere` at every point of a path.

    Args:
        region: an existing `brush`-kind Region.
        points: `(K, 3)` world-frame path the brush was dragged through.
        radius: sphere radius at every point along the path, world units.
        weight: forwarded to the merge (see `paint_sphere`).

    Returns:
        A NEW `Region` with every cell any sphere along the path touches
        added, in ONE merge (not `K` sequential `Region` rebuilds).
    """
    if region.kind != "brush":
        raise ValueError(f"paint_along needs a brush region, got kind={region.kind!r}")
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    p = region.params
    touched = [_sphere_touched_cells(p["origin"], p["cell_size"], pt, radius) for pt in pts]
    all_touched = (
        np.concatenate(touched, axis=0) if touched else np.zeros((0, 3), dtype=np.int64)
    )
    cells, weights = _merge_brush_cells(p.get("cells", []), p.get("weights"), all_touched, weight)
    return _with_brush_params(region, cells, weights)


def erase(region: Region, center: Any, radius: float) -> Region:
    """A new `brush` `Region` with every voxel cell a sphere touches removed.

    The inverse of `paint_sphere`: any cell whose box the sphere overlaps
    (same `_sphere_touched_cells` test) is dropped entirely, regardless of
    its current weight.

    Returns:
        A NEW `Region`; `region` itself is never mutated.

    Raises:
        ValueError: `region.kind != "brush"`.
    """
    if region.kind != "brush":
        raise ValueError(f"erase needs a brush region, got kind={region.kind!r}")
    p = region.params
    touched = _sphere_touched_cells(p["origin"], p["cell_size"], center, radius)
    cells, weights = _remove_brush_cells(p.get("cells", []), p.get("weights"), touched)
    return _with_brush_params(region, cells, weights)


def auto_region_name(existing_names: Any, tool: str, detail: str | None = None) -> str:
    """A tool-authored region's auto name, `"<tool>[-<detail>]-<n>"` (docs/EDITOR.md Sec 1).

    `n` is one more than the highest `-<digits>` suffix among
    `existing_names` (from ANY tool, not just this one), so a session's
    tool-authored regions read as one continuously numbered list in a Named
    Objects panel (`click-1`, `sam-box-IMG_3703-2`, `shade-clouds-3`,
    `brush-4`, ...) no matter which tool created each one. Stateless by
    design: the counter is derived from whatever names exist right now, so
    it self-heals after a region is removed, renamed, or the edit is undone
    -- there is no counter stored anywhere to drift out of sync.

    Args:
        existing_names: the document's current region names (e.g.
            `[r.name for r in edits.regions]`).
        tool: the tool's own label, e.g. `"click"`, `"shade-clouds"`,
            `"brush"`, or `"sam-box"`/`"sam-point"`/`"sam-text"` (SAM's own
            prompt kind folded into the label, matching `"sam-box-
            IMG_3703-2"`).
        detail: an optional extra slug (e.g. a view's file stem), inserted
            between `tool` and the counter.

    Returns:
        `"<tool>-<n>"`, or `"<tool>-<detail>-<n>"` when `detail` is given.
    """
    pattern = re.compile(EDIT_AUTO_NAME_COUNTER_PATTERN)
    highest = 0
    for name in existing_names:
        match = pattern.search(str(name))
        if match:
            highest = max(highest, int(match.group(1)))
    label = tool if detail is None else f"{tool}-{detail}"
    return f"{label}-{highest + 1}"


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
    if region.kind == "brush":
        return brush_membership(xyz, p["origin"], p["cell_size"], p.get("cells", []), p.get("weights"))
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

    @staticmethod
    def _externalize_brush(region_json: dict[str, Any], out_dir: Path) -> dict[str, Any]:
        """A brush region's `regions[]` entry, cells/weights moved to an `.npz` sidecar if large.

        Only ever applied to the WRITTEN `regions[]` array (see `save`); the
        module docstring's "ONE exception" invariant explains why this never
        touches `self.log`/`self.regions`/`to_json()` itself. A region below
        `EDIT_BRUSH_NPZ_CELL_THRESHOLD` cells (or not a brush at all) is
        returned unchanged.
        """
        if region_json.get("kind") != "brush":
            return region_json
        params = region_json.get("params", {})
        cells = params.get("cells") or []
        if len(cells) <= EDIT_BRUSH_NPZ_CELL_THRESHOLD:
            return region_json
        cells_arr = _brush_cells_array(cells).astype(np.int32)
        weights = params.get("weights")
        npz_path = out_dir / EDIT_BRUSH_NPZ_FILENAME_FMT.format(region_id=region_json["id"])
        if weights is not None:
            np.savez(npz_path, cells=cells_arr, weights=np.asarray(weights, dtype=np.float32))
        else:
            np.savez(npz_path, cells=cells_arr)
        new_params = {k: v for k, v in params.items() if k not in ("cells", "weights")}
        new_params["cells_npz"] = npz_path.name
        new_params["n_cells"] = int(cells_arr.shape[0])
        return {**region_json, "params": new_params}

    def save(self, path: str | Path) -> Path:
        """Write this document to `path` (parents created if missing).

        A `brush` region above `EDIT_BRUSH_NPZ_CELL_THRESHOLD` cells is
        written with its `cells`/`weights` externalised into a sidecar
        `.npz` next to `path` (`_externalize_brush`) -- this affects only
        the bytes on disk, never `self`, which keeps holding the literal
        arrays exactly as before the call.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = self.to_json()
        doc["regions"] = [self._externalize_brush(r, path.parent) for r in doc["regions"]]
        path.write_text(json.dumps(doc, indent=2) + "\n")
        return path
