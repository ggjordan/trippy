"""`trippy apply-edits`: publish region edits into a new bundle directory.

Module: trippy.edit.apply
Purpose: the E6 "publish" half of docs/EDITOR.md Sec 5 -- turn a bundle
    directory + `edits.json` into an output directory whose TRIPS points
    have deleted points removed, a per-point blend-weight array, a filtered
    3DGS-style `export.ply` of those same TRIPS points, and (when the
    source bundle names a Gaussian PLY, or `--distilled-ply` names an
    already-distilled one) a filtered copy of that PLY with the same
    regions' deleted/faded Gaussians applied.
Invariants:
    - Reads `bundle.json`/`points.npz` directly (not
      `trippy.render.bundle_render.load_bundle`, which also loads the
      U-Net weights this command never needs).
    - The output directory is NOT a full copy of the input bundle: it gets
      an updated `bundle.json` (same document, `num_points` and
      `blend.splat_ply` updated to match what was actually written),
      the filtered `points.npz`, `blend_weights.npy`, `edits_applied.json`,
      `export.ply` (target `trips`/`both`), and (when applicable) the
      filtered splat PLY/distilled PLY -- but NOT `weights.safetensors` or
      `splat.npz` (unaffected by edits; copying multi-hundred-MB files is
      the caller's job, e.g. `--out` the SAME directory as `--bundle` for
      an in-place publish).
    - `target` (`trippy.constants.EDIT_APPLY_TARGETS`) selects which
      artefact(s) this call produces (docs/EDITOR.md Sec 5): `"trips"` is
      the bundle-side publish above; `"distilled"` re-applies `box`/
      `sphere`/`lid` regions directly to an already-distilled Gaussian PLY
      named by `distilled_ply` (no re-distillation -- `pointset` regions
      are skipped, they do not survive distillation by construction);
      `"both"` (the CLI default) always does the `trips` half and does the
      `distilled` half only when `distilled_ply` is given, recording a
      `summary["distilled"] = {"skipped": ...}` note otherwise so a missing
      `--distilled-ply` is visible in the summary rather than silently
      doing nothing (docs/EDITOR.md Sec 7's own "not discovered by Jordan
      after the fact" risk).
    - The Gaussian PLY filters (`filter_gaussian_ply`, `apply_gaussian_ply_edits`)
      are each a SINGLE `np.fromfile` read of the whole structured vertex
      array, matching `trippy.points.gaussian_ply`'s own "single
      np.fromfile, no per-row loop" invariant: the array's `x`/`y`/`z`
      columns double as the geometry-test input, so a ~2 GB production PLY
      is loaded exactly once, never twice, and every original property
      (including SH `f_rest_*`, whatever the exact property list is)
      survives untouched in the kept rows, in their original order (except
      `opacity`, which `apply_gaussian_ply_edits` may rescale for a `fade`
      region -- see there).
    - `pointset` regions only ever apply to the TRIPS point cloud
      (`trippy.edit.weights` module docstring); both Gaussian PLY filters
      test only `box`/`sphere`/`lid` regions against the PLY's own
      `x, y, z`.
Related docs: docs/EDITOR.md Sec 5 "Publish"; docs/decisions/
    ADR-0007-viewer-editing.md Sec "4. Publish order is edit-TRIPS-first,
    then distil"; trippy.points.gaussian_ply (the read side this mirrors);
    trippy.train.export.write_gaussian_ply (export.ply's writer).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from trippy.constants import (
    EDIT_APPLIED_JSON_FILENAME,
    EDIT_APPLY_DEFAULT_TARGET,
    EDIT_APPLY_TARGETS,
    EDIT_BLEND_WEIGHTS_FILENAME,
    EDIT_GATE_DEFAULT_WEIGHT,
    EXPORT_OPACITY_CLAMP_EPS,
    TRAIN_EXPORT_FILENAME,
)
from trippy.edit.model import EditDocument
from trippy.edit.weights import (
    compose_gaussian_opacity_scale,
    compose_gaussian_weights,
    compose_trips_weights,
)

__all__ = ["apply_edits", "apply_gaussian_ply_edits", "filter_gaussian_ply"]

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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def apply_gaussian_ply_edits(
    in_path: str | Path,
    out_path: str | Path,
    edits: EditDocument,
) -> dict[str, int]:
    """Re-apply `edits`' `box`/`sphere`/`lid` regions directly to an already-distilled Gaussian PLY.

    docs/EDITOR.md Sec 5: a pure-geometry region (never `pointset`, which
    does not survive distillation) can be re-applied straight to a finished
    distilled PLY's own `xyz` without re-distillation. `delete`-op regions
    remove rows exactly like `filter_gaussian_ply`; `fade`-op regions
    instead SCALE the row's own alpha (`sigmoid(opacity)`) by
    `trippy.edit.weights.compose_gaussian_opacity_scale`'s per-row weight
    (1.0 = unchanged, ramping down inside the region) and write the new
    alpha back as `opacity = logit(clamp(alpha, eps, 1-eps))` -- the
    "fade -> opacity scaling" the task brief asks for. `blend`-op regions
    are ignored (see `compose_gaussian_opacity_scale`'s own docstring: a
    plain Gaussian PLY has no TRIPS-vs-splat mix for `blend` to target).

    A single `np.fromfile` read of the whole vertex body (module docstring's
    PLY invariant); every property other than `opacity` survives untouched
    in the kept, possibly-faded rows, in their original order. A PLY with no
    `opacity` property (not 3DGS-shaped) still has its rows deleted
    correctly; fade is silently a no-op on it (nothing to scale) since there
    is no opacity channel to rescale.

    Args:
        in_path: source binary_little_endian Gaussian PLY (already distilled).
        out_path: destination path (parent directories created if missing).
        edits: the loaded `EditDocument`.

    Returns:
        `{"n_in": N, "n_deleted": D, "n_kept": N - D, "n_faded": F}` -- `F`
        is how many surviving rows had their opacity actually rescaled
        (`opacity_scale != 1.0`; 0 when the PLY has no `opacity` property).

    Raises:
        ValueError: `in_path` is not a supported binary PLY, or has no
            `x`/`y`/`z` vertex properties.
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

    composed = compose_gaussian_opacity_scale(edits, xyz)
    keep = ~composed.delete_mask
    filtered = raw[keep]
    scale = composed.weight[keep]
    n_faded = 0
    if "opacity" in names and filtered.size and not np.allclose(scale, 1.0):
        alpha = _sigmoid(filtered["opacity"].astype(np.float64))
        new_alpha = np.clip(alpha * scale, EXPORT_OPACITY_CLAMP_EPS, 1.0 - EXPORT_OPACITY_CLAMP_EPS)
        filtered = filtered.copy()
        filtered["opacity"] = _logit(new_alpha).astype(filtered["opacity"].dtype)
        n_faded = int(np.count_nonzero(~np.isclose(scale, 1.0)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        _write_ply_header(f, int(keep.sum()), props)
        filtered.tofile(f)

    return {
        "n_in": int(count),
        "n_deleted": int(composed.delete_mask.sum()),
        "n_kept": int(keep.sum()),
        "n_faded": n_faded,
    }


# --- apply-edits: TRIPS points + (optional) splat PLY --------------------------------


def apply_edits(
    bundle_dir: str | Path,
    edits_path: str | Path,
    out_dir: str | Path,
    target: str = EDIT_APPLY_DEFAULT_TARGET,
    distilled_ply: str | Path | None = None,
) -> dict[str, Any]:
    """Apply `edits_path` to the bundle at `bundle_dir`, writing results into `out_dir`.

    Args:
        bundle_dir: bundle directory (holds `bundle.json` + `points.npz`).
        edits_path: an `edits.json` file (any path; not required to live
            inside `bundle_dir`).
        out_dir: output directory, created if missing. May be the same
            directory as `bundle_dir` for an in-place publish.
        target: which artefact(s) to publish
            (`trippy.constants.EDIT_APPLY_TARGETS`: `"trips"`, `"distilled"`,
            or `"both"`, the default). `"trips"`/`"both"` write the filtered
            `points.npz`/`blend_weights.npy`/`export.ply` (and the filtered
            splat PLY, if `bundle.json` names one). `"distilled"`/`"both"`
            re-apply `box`/`sphere`/`lid` regions directly to `distilled_ply`
            (no re-distillation); see module docstring.
        distilled_ply: an already-distilled Gaussian PLY (e.g. from `trippy
            distill`'s Brush output) to re-apply geometry edits to. Required
            when `target == "distilled"`; optional for `target == "both"`
            (a missing one there is recorded as `summary["distilled"]
            = {"skipped": ...}` rather than an error, since the CLI's own
            default target is "both" and most callers have not distilled
            yet). Ignored when `target == "trips"`.

    Returns:
        A JSON-serialisable summary (also written to
        `<out_dir>/edits_applied.json`): `target`, the applied region list,
        and -- depending on `target` -- `points` (before/after counts),
        `export_ply`, `splat_ply` (if the bundle names one), and/or
        `distilled` (the re-applied distilled-PLY counts, or a `skipped`
        note).

    Raises:
        ValueError: `edits.json`'s `bundle_format` disagrees with the
            bundle's own `bundle.json` format, `target` is not one of
            `trippy.constants.EDIT_APPLY_TARGETS`, `target == "distilled"`
            with no `distilled_ply`, or a region/PLY is malformed (see
            `EditDocument.validate` / `filter_gaussian_ply`).
        FileNotFoundError: `bundle.json`, `points.npz`, a named `splat_ply`,
            or `distilled_ply` does not exist.
    """
    if target not in EDIT_APPLY_TARGETS:
        raise ValueError(f"target must be one of {EDIT_APPLY_TARGETS}, got {target!r}")
    if target == "distilled" and distilled_ply is None:
        raise ValueError("target='distilled' requires distilled_ply")

    from trippy.render.bundle import BUNDLE_JSON_FILENAME, BUNDLE_POINTS_FILENAME
    from trippy.train.export import write_gaussian_ply

    bundle_dir = Path(bundle_dir)
    out_dir = Path(out_dir)
    bundle_json_path = bundle_dir / BUNDLE_JSON_FILENAME
    bundle_doc = json.loads(bundle_json_path.read_text())
    bundle_format = bundle_doc.get("format", "")
    points_filename = bundle_doc.get("points", BUNDLE_POINTS_FILENAME)

    edits = EditDocument.load(edits_path)
    edits.validate(expected_bundle_format=bundle_format)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "bundle": str(bundle_dir.resolve()),
        "edits": str(Path(edits_path).resolve()),
        "out": str(out_dir.resolve()),
        "target": target,
        "regions": [
            {"id": r.id, "name": r.name, "kind": r.kind, "op": r.op, "mix": r.mix, "enabled": r.enabled}
            for r in edits.regions
        ],
    }

    if target in ("trips", "both"):
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

        kept_xyz = xyz[keep].astype(np.float32)
        kept_size = size[keep].astype(np.float32)
        kept_feat = feat[keep].astype(np.float32)
        kept_conf = conf[keep].astype(np.float32)

        np.savez(out_dir / points_filename, xyz=kept_xyz, size=kept_size, feat=kept_feat, conf=kept_conf)
        np.save(out_dir / EDIT_BLEND_WEIGHTS_FILENAME, composed.weight[keep].astype(np.float32))

        # The filtered 3DGS-style export.ply (docs/EDITOR.md Sec 5): the SAME kept TRIPS
        # points, so Splats' audits and Brush can open the edited TRIPS point set directly,
        # without a bundle loader. Reuses trippy.train.export.write_gaussian_ply verbatim --
        # the identical writer Trainer.export_ply calls at training time.
        export_path = out_dir / TRAIN_EXPORT_FILENAME
        write_gaussian_ply(
            export_path,
            xyz=kept_xyz.astype(np.float64),
            rgb=np.clip(kept_feat[:, :3].astype(np.float64), 0.0, 1.0),
            conf=kept_conf.astype(np.float64),
            size=kept_size.astype(np.float64),
        )
        summary["export_ply"] = str(export_path.resolve())

        out_bundle_doc = dict(bundle_doc)
        out_bundle_doc["num_points"] = int(keep.sum())
        summary["points"] = {"n_in": n_in, "n_deleted": n_deleted, "n_kept": n_in - n_deleted}

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

    if target in ("distilled", "both"):
        if distilled_ply is None:
            # target == "both" with no --distilled-ply: a visible note, not a silent no-op
            # (docs/EDITOR.md Sec 7 "not discovered by Jordan after the fact").
            summary["distilled"] = {
                "skipped": (
                    "no distilled_ply given -- run `trippy distill` first, then re-run "
                    "apply-edits with --distilled-ply to reapply box/sphere/lid edits to it"
                )
            }
        else:
            distilled_in = Path(distilled_ply)
            if not distilled_in.exists():
                raise FileNotFoundError(f"distilled_ply {distilled_in} does not exist")
            distilled_out = out_dir / distilled_in.name
            ply_summary = apply_gaussian_ply_edits(distilled_in, distilled_out, edits)
            summary["distilled"] = {"in": str(distilled_in), "out": str(distilled_out), **ply_summary}
            if any(r.enabled and r.kind == "pointset" for r in edits.regions):
                summary["distilled"]["warning"] = (
                    "pointset regions do not survive distillation (docs/EDITOR.md Sec 7) -- "
                    "only box/sphere/lid regions were re-applied to this distilled PLY"
                )

    (out_dir / EDIT_APPLIED_JSON_FILENAME).write_text(json.dumps(summary, indent=2) + "\n")
    return summary
