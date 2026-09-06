"""`trippy apply-edits`: publish region edits into a new bundle directory.

Module: trippy.edit.apply
Purpose: the E6 "publish" half of docs/EDITOR.md Sec 5 -- turn a bundle
    directory + `edits.json` into an output directory whose TRIPS points
    have deleted points removed, a per-point blend-weight array, and (when
    the source bundle names a Gaussian PLY) a filtered copy of that PLY
    with the same regions' deleted Gaussians removed.
Invariants:
    - Reads `bundle.json`/`points.npz` directly (not
      `trippy.render.bundle_render.load_bundle`, which also loads the
      U-Net weights this command never needs).
    - The output directory is NOT a full copy of the input bundle: it gets
      an updated `bundle.json` (same document, `num_points` and
      `blend.splat_ply` updated to match what was actually written),
      the filtered `points.npz`, `blend_weights.npy`, `edits_applied.json`,
      and (when applicable) the filtered splat PLY -- but NOT
      `weights.safetensors` or `splat.npz` (unaffected by edits; copying
      multi-hundred-MB files is the caller's job, e.g. `--out` the SAME
      directory as `--bundle` for an in-place publish).
    - The Gaussian PLY filter (`filter_gaussian_ply`) is a SINGLE
      `np.fromfile` read of the whole structured vertex array, matching
      `trippy.points.gaussian_ply`'s own "single np.fromfile, no per-row
      loop" invariant: the array's `x`/`y`/`z` columns double as the
      geometry-test input, so a ~2 GB production PLY is loaded exactly
      once, never twice, and every original property (including SH
      `f_rest_*`, whatever the exact property list is) survives untouched
      in the kept rows, in their original order.
    - `pointset` regions only ever apply to the TRIPS point cloud
      (`trippy.edit.weights` module docstring); the Gaussian filter tests
      only `box`/`sphere`/`lid` regions against the PLY's own `x, y, z`.
Related docs: docs/EDITOR.md Sec 5 "Publish"; docs/decisions/
    ADR-0007-viewer-editing.md Sec "4. Publish order is edit-TRIPS-first,
    then distil"; trippy.points.gaussian_ply (the read side this mirrors).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from trippy.constants import EDIT_APPLIED_JSON_FILENAME, EDIT_BLEND_WEIGHTS_FILENAME, EDIT_GATE_DEFAULT_WEIGHT
from trippy.edit.model import EditDocument
from trippy.edit.weights import compose_gaussian_weights, compose_trips_weights

__all__ = ["apply_edits", "filter_gaussian_ply"]

# --- generic binary PLY header parsing/writing (independent of trippy.points.gaussian_ply's
# private helpers -- see module docstring; this module preserves whatever property list the
# header names, sight unseen, rather than special-casing 3DGS's usual field names). ------------

_PLY_NP_DTYPE = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
    "uchar": "<u1",
    "uint8": "<u1",
    "char": "<i1",
    "int8": "<i1",
    "ushort": "<u2",
    "uint16": "<u2",
    "short": "<i2",
    "int16": "<i2",
    "uint": "<u4",
    "uint32": "<u4",
    "int": "<i4",
    "int32": "<i4",
}


def _read_ply_vertex_header(f: Any) -> tuple[int, list[tuple[str, str]]]:
    """Parse a binary_little_endian PLY header's `vertex` element.

    Returns:
        `(vertex_count, [(prop_name, prop_type), ...])` in header order.

    Raises:
        ValueError: not `binary_little_endian 1.0`, no `vertex` element, or
            a list property (unsupported -- 3DGS vertex properties are all
            scalar).
    """
    if f.readline().strip() != b"ply":
        raise ValueError("not a PLY file (missing magic 'ply' line)")
    fmt = f.readline().strip()
    if fmt != b"format binary_little_endian 1.0":
        raise ValueError(f"unsupported PLY format {fmt!r}; expected binary_little_endian 1.0")

    count: int | None = None
    props: list[tuple[str, str]] = []
    current_element: str | None = None
    while True:
        line = f.readline()
        if not line:
            raise ValueError("unexpected EOF while reading PLY header")
        tokens = line.strip().split()
        if not tokens or tokens[0] == b"comment":
            continue
        if tokens[0] == b"element":
            current_element = tokens[1].decode()
            if current_element == "vertex":
                count = int(tokens[2])
            continue
        if tokens[0] == b"property" and current_element == "vertex":
            if tokens[1] == b"list":
                raise ValueError(f"list property unsupported in vertex element: {line!r}")
            props.append((tokens[2].decode(), tokens[1].decode()))
            continue
        if tokens[0] == b"end_header":
            break
    if count is None:
        raise ValueError("PLY header has no 'vertex' element")
    return count, props


def _ply_dtype(props: list[tuple[str, str]]) -> np.dtype:
    return np.dtype([(name, _PLY_NP_DTYPE[ptype]) for name, ptype in props])


def _write_ply_header(f: Any, n: int, props: list[tuple[str, str]]) -> None:
    lines = [b"ply", b"format binary_little_endian 1.0", f"element vertex {n}".encode("ascii")]
    lines += [f"property {ptype} {name}".encode("ascii") for name, ptype in props]
    lines.append(b"end_header")
    f.write(b"\n".join(lines) + b"\n")


def filter_gaussian_ply(
    in_path: str | Path,
    out_path: str | Path,
    delete_test: Callable[[np.ndarray], np.ndarray],
) -> dict[str, int]:
    """Copy `in_path` to `out_path`, dropping rows `delete_test(xyz)` marks True.

    Every original vertex property survives in the kept rows, untouched and
    in its original order (see module docstring's PLY invariant).

    Args:
        in_path: source binary_little_endian 3DGS PLY.
        out_path: destination path (parent directories created if missing).
        delete_test: `xyz (N, 3) float64 -> delete_mask (N,) bool-like`.

    Returns:
        `{"n_in": N, "n_deleted": D, "n_kept": N - D}`.

    Raises:
        ValueError: `in_path` is not a supported binary PLY, has no `x`/
            `y`/`z` vertex properties, or `delete_test` returns the wrong
            length.
    """
    in_path, out_path = Path(in_path), Path(out_path)
    with open(in_path, "rb") as f:
        count, props = _read_ply_vertex_header(f)
        dtype = _ply_dtype(props)
        raw = np.fromfile(f, dtype=dtype, count=count)  # the one and only full-body read

    names = {name for name, _ in props}
    if not {"x", "y", "z"} <= names:
        raise ValueError(f"{in_path}: vertex element has no x/y/z properties (got {sorted(names)})")
    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float64)

    delete_mask = np.asarray(delete_test(xyz), dtype=bool)
    if delete_mask.shape != (count,):
        raise ValueError(f"delete_test returned shape {delete_mask.shape}, expected ({count},)")
    keep = ~delete_mask
    filtered = raw[keep]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        _write_ply_header(f, int(keep.sum()), props)
        filtered.tofile(f)

    return {"n_in": int(count), "n_deleted": int(delete_mask.sum()), "n_kept": int(keep.sum())}


# --- apply-edits: TRIPS points + (optional) splat PLY --------------------------------


def apply_edits(bundle_dir: str | Path, edits_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Apply `edits_path` to the bundle at `bundle_dir`, writing results into `out_dir`.

    Args:
        bundle_dir: bundle directory (holds `bundle.json` + `points.npz`).
        edits_path: an `edits.json` file (any path; not required to live
            inside `bundle_dir`).
        out_dir: output directory, created if missing. May be the same
            directory as `bundle_dir` for an in-place publish.

    Returns:
        A JSON-serialisable summary (also written to
        `<out_dir>/edits_applied.json`): point counts before/after, the
        applied region list, and (if the bundle names a splat PLY) the
        filtered-PLY counts.

    Raises:
        ValueError: `edits.json`'s `bundle_format` disagrees with the
            bundle's own `bundle.json` format, or a region/PLY is
            malformed (see `EditDocument.validate` / `filter_gaussian_ply`).
        FileNotFoundError: `bundle.json`, `points.npz`, or a named
            `splat_ply` does not exist.
    """
    from trippy.render.bundle import BUNDLE_JSON_FILENAME, BUNDLE_POINTS_FILENAME

    bundle_dir = Path(bundle_dir)
    out_dir = Path(out_dir)
    bundle_json_path = bundle_dir / BUNDLE_JSON_FILENAME
    bundle_doc = json.loads(bundle_json_path.read_text())
    bundle_format = bundle_doc.get("format", "")
    points_filename = bundle_doc.get("points", BUNDLE_POINTS_FILENAME)

    edits = EditDocument.load(edits_path)
    edits.validate(expected_bundle_format=bundle_format)

    points_path = bundle_dir / points_filename
    with np.load(points_path) as data:
        xyz = np.asarray(data["xyz"])
        size = np.asarray(data["size"])
        feat = np.asarray(data["feat"])
        conf = np.asarray(data["conf"])

    composed = compose_trips_weights(edits, xyz, default=EDIT_GATE_DEFAULT_WEIGHT)
    keep = ~composed.delete_mask
    n_in = int(xyz.shape[0])
    n_deleted = int(composed.delete_mask.sum())

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / points_filename,
        xyz=xyz[keep].astype(np.float32),
        size=size[keep].astype(np.float32),
        feat=feat[keep].astype(np.float32),
        conf=conf[keep].astype(np.float32),
    )
    np.save(out_dir / EDIT_BLEND_WEIGHTS_FILENAME, composed.weight[keep].astype(np.float32))

    out_bundle_doc = dict(bundle_doc)
    out_bundle_doc["num_points"] = int(keep.sum())

    summary: dict[str, Any] = {
        "bundle": str(bundle_dir.resolve()),
        "edits": str(Path(edits_path).resolve()),
        "out": str(out_dir.resolve()),
        "points": {"n_in": n_in, "n_deleted": n_deleted, "n_kept": n_in - n_deleted},
        "regions": [
            {"id": r.id, "name": r.name, "kind": r.kind, "op": r.op, "mix": r.mix, "enabled": r.enabled}
            for r in edits.regions
        ],
    }

    blend = bundle_doc.get("blend") or {}
    splat_ply = blend.get("splat_ply", "")
    if splat_ply:
        splat_in = Path(splat_ply)
        if not splat_in.is_absolute():
            splat_in = bundle_dir / splat_in
        if not splat_in.exists():
            raise FileNotFoundError(f"bundle.json names splat_ply {splat_in}, which does not exist")
        splat_out = out_dir / splat_in.name

        def _gaussian_delete_mask(gaussian_xyz: np.ndarray) -> np.ndarray:
            return compose_gaussian_weights(edits, gaussian_xyz, default=EDIT_GATE_DEFAULT_WEIGHT).delete_mask

        ply_summary = filter_gaussian_ply(splat_in, splat_out, _gaussian_delete_mask)
        summary["splat_ply"] = {"in": str(splat_in), "out": str(splat_out), **ply_summary}
        out_blend = dict(blend)
        out_blend["splat_ply"] = str(splat_out.resolve())
        out_bundle_doc["blend"] = out_blend

    (out_dir / BUNDLE_JSON_FILENAME).write_text(json.dumps(out_bundle_doc, indent=2) + "\n")
    (out_dir / EDIT_APPLIED_JSON_FILENAME).write_text(json.dumps(summary, indent=2) + "\n")
    return summary
