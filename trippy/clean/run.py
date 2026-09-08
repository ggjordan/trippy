"""Drive one Design B clean: score, select, filter, audit, draw, summarise.

Module: trippy.clean.run
Purpose: the one function `trippy splat-clean` calls, kept out of
    `trippy.cli` so it is testable without argparse and reusable from a
    queue job. It owns the ORDER of the steps and the contents of
    `summary.json`; every decision it makes is delegated to the module that
    owns it (`mapping`, `select`, `ply_filter`, `freespace`, `heatmap`,
    `trippy.train.prune`, `trippy.eval.audits`).
Invariants:
    - CPU only. Nothing here constructs a `Trainer`, a renderer, or a torch
      device; `torch` appears exactly once, inside
      `trippy.clean.score.load_point_scores`, under `map_location="cpu"`.
    - The mapping is recovered and its evidence is written into every
      `summary.json` before a single Gaussian is deleted.
    - Audits are optional and never fatal: Splats' `depthprior_shade_audit.py`
      and `extent_gate.py` live outside this repo (AGENTS.md Sec 6: read
      live, never copy), so a machine without them still produces the PLY
      and records why the audit is missing.
    - No image is opened. The heatmap is written, never read back.
Units: world units are COLMAP's; confidences are dimensionless.
Related docs: docs/SPEC.md D2 (Design B); docs/USER_GUIDE.md
    ("Cleaning a splat with TRIPS"); docs/EXPERIMENTS.md "Shade audit".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trippy.clean.heatmap import deleted_density_png
from trippy.clean.mapping import recover_mapping
from trippy.clean.ply_filter import filter_ply, read_columns, read_ply_layout
from trippy.clean.score import confidence_summary, drift_stats, load_point_scores
from trippy.clean.select import CleanVariant, deletion_mask, ply_keep_mask
from trippy.constants import (
    CLEAN_FREESPACE_DEPTH_SCALE,
    CLEAN_VARIANT_ALL_005_THRESHOLD,
    CLEAN_VARIANT_ALL_015_THRESHOLD,
    DEFAULT_MIN_OPACITY,
    SHADE_PRUNE_DEFAULT_ZFAR_FRAC,
    SHADE_PRUNE_DEFAULT_ZNEAR_FRAC,
)
from trippy.train import prune


def load_frames(frames: list[str] | None, frames_json: str | Path | None, key: str) -> list[str]:
    """Resolve the shade-frame list from an explicit list or a JSON file.

    Args:
        frames: explicit image names, or None.
        frames_json: a JSON file holding either a list of names or a dict
            of named lists (`$TRIPPY_OUTPUT/scratch/shade_frames.json` is
            the latter, with a `big_tree` key -- see
            experiments/EXP-0011-karekare-v2/README.md).
        key: which key to take when the JSON holds a dict.

    Returns:
        The frame names.

    Raises:
        ValueError: neither source was given, or the key is absent.
    """
    if frames:
        return list(frames)
    if frames_json is None:
        raise ValueError("need --frames or --frames-json to build the shade region")
    payload = json.loads(Path(frames_json).read_text())
    if isinstance(payload, list):
        return [str(f) for f in payload]
    if key not in payload:
        raise ValueError(f"{frames_json} has no key {key!r} (has {sorted(payload)})")
    return [str(f) for f in payload[key]]


def clean_splat(
    checkpoint: str | Path,
    ply: str | Path,
    out_dir: str | Path,
    variants: list[CleanVariant],
    sparse_dir: str | Path | None = None,
    frames: list[str] | None = None,
    znear_frac: float = SHADE_PRUNE_DEFAULT_ZNEAR_FRAC,
    zfar_frac: float = SHADE_PRUNE_DEFAULT_ZFAR_FRAC,
    min_opacity: float | None = None,
    freespace: bool = True,
    freespace_scale: int = CLEAN_FREESPACE_DEPTH_SCALE,
    heatmap: bool = True,
    audit: bool = True,
    audit_sparse_txt: str | Path | None = None,
    log=print,
) -> dict:
    """Produce one cleaned PLY per variant, plus a summary and an honesty map.

    Args:
        checkpoint: trained TRIPS checkpoint whose confidences do the judging.
        ply: the source 3DGS splat the checkpoint was seeded from.
        out_dir: directory for the PLYs, heatmaps and `summary.json`.
        variants: which `trippy.clean.select.CleanVariant`s to build.
        sparse_dir: COLMAP model for the shade region (needed by any
            `shade_only` variant, and by the free-space check).
        frames: shade-region frame names (see `load_frames`).
        znear_frac: near plane as a fraction of each frame's median
            observed depth (the shade audit's own default).
        zfar_frac: far plane, same units.
        min_opacity: the training config's `point_source.min_opacity`;
            None reads it out of the checkpoint's own config.
        freespace: run `trippy.clean.freespace` on each variant.
        freespace_scale: depth-buffer divisor for that check.
        heatmap: write the top-down deleted-density PNG per variant.
        audit: run Splats' shade audit + extent gate on the source PLY and
            on every variant.
        audit_sparse_txt: the TEXT COLMAP model those Splats tools need;
            defaults to `sparse_dir` when that already ends in `sparse_txt`.
        log: line printer.

    Returns:
        The summary dict (also written to `out_dir/summary.json`).

    Raises:
        ValueError: a `shade_only` variant was asked for without a
            `sparse_dir` + `frames` to build the region from.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    needs_region = any(v.shade_only for v in variants) or freespace

    log(f"reading checkpoint {checkpoint}")
    scores = load_point_scores(checkpoint)
    source_cfg = scores.point_source_cfg()
    if min_opacity is None:
        min_opacity = float(source_cfg.get("min_opacity", DEFAULT_MIN_OPACITY))
    log(f"  epoch {scores.epoch}, {scores.n:,} points, points_removed_total={scores.points_removed_total}")

    layout = read_ply_layout(ply)
    log(f"  source ply {layout.count:,} rows x {layout.itemsize} bytes, {len(layout.props)} properties")

    mapping = recover_mapping(
        layout,
        init_conf=scores.init_conf,
        point_xyz=scores.xyz,
        min_opacity=min_opacity,
        max_points=source_cfg.get("max_points"),
        seed=int(source_cfg.get("seed", 0)),
    )
    log(f"  mapping: {mapping.method}, {mapping.n_points:,} of {mapping.n_ply:,} rows, evidence {mapping.evidence}")

    cols = read_columns(layout, ["x", "y", "z"])
    ply_xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float32)
    del cols
    seed_xyz = ply_xyz[mapping.ply_row]

    inside = None
    views: list[prune.ShadeView] = []
    if needs_region:
        if sparse_dir is None or not frames:
            raise ValueError("a shade-region variant (or --freespace) needs --scene and --frames/--frames-json")
        log(f"building the shade region from {len(frames)} frames in {sparse_dir}")
        views = prune.build_shade_region(sparse_dir, frames, znear_frac, zfar_frac)
        inside, _zfrac = prune.in_region(views, scores.xyz)
        log(f"  {int(inside.sum()):,} of {scores.n:,} points are inside the region")

    summary: dict = {
        "checkpoint": str(checkpoint),
        "epoch": scores.epoch,
        "points_removed_total": scores.points_removed_total,
        "source_ply": str(layout.path),
        "n_ply": int(layout.count),
        "n_points": scores.n,
        "min_opacity": float(min_opacity),
        "mapping": {"method": mapping.method, **mapping.evidence},
        "drift": drift_stats(scores.xyz, seed_xyz),
        "confidence": confidence_summary(
            scores, [0.01, 0.02, CLEAN_VARIANT_ALL_005_THRESHOLD, 0.1, CLEAN_VARIANT_ALL_015_THRESHOLD, 0.3, 0.5]
        ),
        "region": {
            "frames": list(frames) if frames else [],
            "n_frames": len(views),
            "znear_frac": znear_frac,
            "zfar_frac": zfar_frac,
            "n_points_inside": int(inside.sum()) if inside is not None else None,
        },
        "variants": {},
    }

    for variant in variants:
        log(f"variant {variant.name}: conf < {variant.conf_threshold}{' inside region' if variant.shade_only else ''}")
        delete_points = deletion_mask(scores.conf, variant.conf_threshold, inside if variant.shade_only else None)
        keep_rows = ply_keep_mask(delete_points, mapping.ply_row, mapping.n_ply)
        n_deleted = int(mapping.n_ply - keep_rows.sum())
        dest = out_dir / f"kklid-tripsclean-{variant.name}.ply"
        log(f"  deleting {n_deleted:,} of {mapping.n_ply:,} Gaussians ({100 * n_deleted / mapping.n_ply:.2f}%)")
        n_written = filter_ply(layout, keep_rows, dest)
        log(f"  wrote {n_written:,} Gaussians to {dest} ({dest.stat().st_size / 1e9:.2f} GB)")

        entry: dict = {
            "name": variant.name,
            "conf_threshold": variant.conf_threshold,
            "shade_only": variant.shade_only,
            "why": variant.why,
            "ply": str(dest),
            "bytes": int(dest.stat().st_size),
            "n_deleted": n_deleted,
            "n_kept": n_written,
            "deleted_fraction": n_deleted / max(mapping.n_ply, 1),
            "deleted_mass": float(scores.init_conf[delete_points].sum()),
        }
        if inside is not None:
            entry["n_deleted_inside_region"] = int((delete_points & inside).sum())

        if freespace and views:
            log("  free-space check against the confident surface...")
            entry["freespace"] = classify = _freespace(views, scores, delete_points, freespace_scale)
            log(
                f"    front {classify['front']:,} / on {classify['on']:,} / behind {classify['behind']:,}"
                f" / no-surface {classify['no_surface']:,} (front {100 * classify['front_frac']:.1f}%)"
            )
            log(
                f"    holes: {classify['pixels_emptied']:,} of {classify['pixels_covered']:,} covered pixels"
                f" emptied ({100 * classify['pixels_emptied_frac']:.4f}%);"
                f" {classify['pixels_revealed']:,} pixels saw further,"
                f" {classify['pixels_revealed_confident']:,} of them onto a confident surface"
            )

        if heatmap:
            png = out_dir / f"kklid-tripsclean-{variant.name}-deleted-density.png"
            entry["heatmap"] = deleted_density_png(ply_xyz, ~keep_rows, png)
            log(f"  heatmap {png}")

        summary["variants"][variant.name] = entry
        del delete_points, keep_rows

    if audit:
        sparse_txt = audit_sparse_txt or sparse_dir
        summary["audits"] = _audits(
            [layout.path] + [Path(summary["variants"][v.name]["ply"]) for v in variants],
            sparse_txt,
            frames,
            log,
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"summary -> {out_dir / 'summary.json'}")
    return summary


def _freespace(views, scores, delete_points, scale) -> dict:
    """Wrap `trippy.clean.freespace.classify_against_surface` (kept out of the flow above)."""
    from trippy.clean.freespace import classify_against_surface

    return classify_against_surface(views, scores.xyz, scores.conf, delete_points, scale=scale)


def _audits(ply_paths: list[Path], sparse_txt, frames, log) -> dict:
    """Run Splats' shade audit + extent gate, recording a failure instead of raising.

    Both tools live under `$SPLATS_ROOT/tools` and are called through
    `trippy.eval.audits`, i.e. the exact same entry points every training
    run's report uses, so these numbers sit in the same column as the ones
    already in the run READMEs.
    """
    from trippy.eval import audits as audit_mod

    out: dict = {}
    paths = [str(p) for p in ply_paths]
    for label, fn, kwargs in (
        ("shade_audit", audit_mod.run_shade_audit, {"sparse_txt_dir": str(sparse_txt), "frames": frames}),
        ("extent_gate", audit_mod.run_extent_gate, {}),
    ):
        try:
            log(f"running {label} on {len(paths)} PLYs...")
            out[label] = fn(paths, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- an audit is evidence, never a gate
            log(f"  {label} FAILED: {type(exc).__name__}: {exc}")
            out[label] = {"error": f"{type(exc).__name__}: {exc}"}
    return out
