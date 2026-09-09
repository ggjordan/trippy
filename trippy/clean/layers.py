"""Two-layer splat export: the same fog decision as a pair of PLYs, for SuperSplat.

Module: trippy.clean.layers
Purpose: ADR-0008-supersplat.md Stage 1 task 3. `trippy.clean.run.clean_splat`
    writes only the survivors of each variant. SuperSplat's data model has no
    notion of "here is a deletion list, go inspect it" -- but it DOES have
    layers, each with independent show/hide/solo (ADR-0008 Sec 1a "Multiple
    splat files open as layers"). So this module writes the identical fog
    decision as a PARTITION of the source PLY into two files instead of one:
    `<name>-keep.ply` (everything the variant keeps) and `<name>-fog.ply`
    (everything it would delete). Opening both in SuperSplat gives Jordan an
    inspectable, toggleable fog mask for free, through the mechanism their
    editor actually has, with zero new file-format work on trippy's side.
Invariants:
    - Reuses trippy.clean's own scoring/selection/mapping machinery verbatim
      (score.load_point_scores, mapping.recover_mapping, select.deletion_mask,
      select.ply_keep_mask) -- there is exactly one definition of "what counts
      as fog" in this codebase, and it lives in trippy.clean.select.
    - Every source row lands in EXACTLY ONE of the two output files: a
      Gaussian TRIPS never saw (below the source's min_opacity) always goes to
      -keep.ply, same as trippy.clean.select's module invariant.
    - Both files are byte-for-byte row copies (trippy.clean.ply_filter.filter_ply
      gives this for free); no value is recomputed, rounded or re-encoded.
    - CPU only. Nothing here builds a Trainer, a renderer, or a torch device
      other than the CPU (score.load_point_scores loads with map_location="cpu").
Units: confidences dimensionless in (0, 1); positions COLMAP world units.
Related docs: docs/decisions/ADR-0008-supersplat.md Stage 1 task 3;
    trippy.clean.run (the single-file, keep-only sibling of this; also owns
    the freespace/heatmap/audit machinery this module deliberately omits --
    SuperSplat's own selection tools are the inspection step here, not those);
    docs/USER_GUIDE.md "Editing the cleaned splat in SuperSplat".
"""

from __future__ import annotations

from pathlib import Path

from trippy.clean.mapping import recover_mapping
from trippy.clean.ply_filter import filter_ply, read_ply_layout
from trippy.clean.score import load_point_scores
from trippy.clean.select import CleanVariant, deletion_mask, ply_keep_mask
from trippy.constants import (
    DEFAULT_MIN_OPACITY,
    SHADE_PRUNE_DEFAULT_ZFAR_FRAC,
    SHADE_PRUNE_DEFAULT_ZNEAR_FRAC,
)
from trippy.train import prune


def export_splat_layers(
    checkpoint: str | Path,
    ply: str | Path,
    out_dir: str | Path,
    variant: CleanVariant,
    name: str | None = None,
    sparse_dir: str | Path | None = None,
    frames: list[str] | None = None,
    znear_frac: float = SHADE_PRUNE_DEFAULT_ZNEAR_FRAC,
    zfar_frac: float = SHADE_PRUNE_DEFAULT_ZFAR_FRAC,
    min_opacity: float | None = None,
    log=print,
) -> dict:
    """Write `<name>-keep.ply` / `<name>-fog.ply` / `<name>-layers.txt` for one variant.

    Args:
        checkpoint: trained TRIPS checkpoint whose confidences do the judging.
        ply: the 3DGS PLY that checkpoint was seeded from.
        out_dir: directory for the two PLYs and the manifest.
        variant: which `trippy.clean.select.CleanVariant` to apply (the same
            object `trippy splat-clean` uses -- same threshold, same rule).
        name: file-name prefix; defaults to `Path(ply).stem`.
        sparse_dir: COLMAP model for the shade region; required if
            `variant.shade_only`.
        frames: shade-region frame names; required if `variant.shade_only`.
        znear_frac: shade region near plane, as a fraction of each frame's
            median observed depth.
        zfar_frac: shade region far plane, same units.
        min_opacity: the training config's `point_source.min_opacity`; None
            reads it out of the checkpoint's own config.
        log: line printer.

    Returns:
        A dict: `checkpoint`, `source_ply`, `n_ply`, `variant`,
        `conf_threshold`, `shade_only`, `mapping` (method + evidence),
        `keep_ply`, `fog_ply`, `layers_txt`, `n_kept`, `n_fog`.

    Raises:
        ValueError: `variant.shade_only` is set but `sparse_dir`/`frames`
            were not given, or the point<->row mapping cannot be recovered
            (see `trippy.clean.mapping.recover_mapping`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"reading checkpoint {checkpoint}")
    scores = load_point_scores(checkpoint)
    source_cfg = scores.point_source_cfg()
    if min_opacity is None:
        min_opacity = float(source_cfg.get("min_opacity", DEFAULT_MIN_OPACITY))
    log(f"  epoch {scores.epoch}, {scores.n:,} points")

    layout = read_ply_layout(ply)
    log(f"  source ply {layout.count:,} rows x {layout.itemsize} bytes")

    mapping = recover_mapping(
        layout,
        init_conf=scores.init_conf,
        point_xyz=scores.xyz,
        min_opacity=min_opacity,
        max_points=source_cfg.get("max_points"),
        seed=int(source_cfg.get("seed", 0)),
    )
    log(f"  mapping: {mapping.method}, {mapping.n_points:,} of {mapping.n_ply:,} rows, evidence {mapping.evidence}")

    inside = None
    if variant.shade_only:
        if sparse_dir is None or not frames:
            raise ValueError(
                f"variant {variant.name!r} is shade_only and needs --scene and --frames/--frames-json "
                "to build the shade region"
            )
        log(f"building the shade region from {len(frames)} frames in {sparse_dir}")
        views = prune.build_shade_region(sparse_dir, frames, znear_frac, zfar_frac)
        inside, _zfrac = prune.in_region(views, scores.xyz)
        log(f"  {int(inside.sum()):,} of {scores.n:,} points are inside the region")

    log(f"variant {variant.name}: conf < {variant.conf_threshold}{' inside region' if variant.shade_only else ''}")
    delete_points = deletion_mask(scores.conf, variant.conf_threshold, inside)
    keep_rows = ply_keep_mask(delete_points, mapping.ply_row, mapping.n_ply)
    fog_rows = ~keep_rows

    prefix = name or Path(ply).stem
    keep_dest = out_dir / f"{prefix}-{variant.name}-keep.ply"
    fog_dest = out_dir / f"{prefix}-{variant.name}-fog.ply"

    n_kept = filter_ply(layout, keep_rows, keep_dest)
    n_fog = filter_ply(layout, fog_rows, fog_dest)
    if n_kept + n_fog != mapping.n_ply:
        raise ValueError(  # pragma: no cover -- keep_rows/fog_rows are exact complements by construction
            f"partition invariant broken: {n_kept:,} + {n_fog:,} != {mapping.n_ply:,}"
        )
    log(f"  wrote {n_kept:,} rows -> {keep_dest}")
    log(f"  wrote {n_fog:,} rows -> {fog_dest}")

    layers_txt = out_dir / f"{prefix}-{variant.name}-layers.txt"
    layers_txt.write_text(
        "\n".join(
            [
                f"source ply: {layout.path}",
                f"checkpoint: {checkpoint}",
                f"variant: {variant.name}",
                f"threshold: conf < {variant.conf_threshold}"
                + (" (inside the shade region only)" if variant.shade_only else " (everywhere in the scene)"),
                f"why: {variant.why}",
                "",
                f"keep: {keep_dest.name}  ({n_kept:,} of {mapping.n_ply:,} rows, {100 * n_kept / mapping.n_ply:.2f}%)",
                f"fog:  {fog_dest.name}  ({n_fog:,} of {mapping.n_ply:,} rows, {100 * n_fog / mapping.n_ply:.2f}%)",
                "",
                "Open both in SuperSplat (scripts/open_supersplat.sh) -- they land as two",
                "layers in the Scene panel. Solo 'fog' to see exactly what TRIPS flagged;",
                "hide it once checked, then finish cleaning 'keep' with the selection tools",
                "and export when done (never click File > Publish).",
                "",
            ]
        )
    )
    log(f"  wrote {layers_txt}")

    return {
        "checkpoint": str(checkpoint),
        "source_ply": str(layout.path),
        "n_ply": int(mapping.n_ply),
        "variant": variant.name,
        "conf_threshold": variant.conf_threshold,
        "shade_only": variant.shade_only,
        "mapping": {"method": mapping.method, **mapping.evidence},
        "keep_ply": str(keep_dest),
        "fog_ply": str(fog_dest),
        "layers_txt": str(layers_txt),
        "n_kept": n_kept,
        "n_fog": n_fog,
    }
