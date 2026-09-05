"""Baseline / TRIPS-export / distilled audit comparison table (design-B step 3).

Module: trippy.distill.compare
Purpose: the task brief's pipeline step 3 -- "run the shade audit + extent
    gate on the distilled ply and compare to the baseline kkc_15000.ply and
    to the TRIPS export". Reuses `trippy.eval.audits.audit_report`
    unchanged (Splats' own shade audit + extent gate, run read-only via
    subprocess) and `trippy.render.report`'s existing pure extraction
    helpers (`dark_mass_fraction`, `extent_p99_max`) rather than
    reimplementing them, so this module's table reads exactly the same
    numbers `trippy train --report`'s own baseline-vs-candidate table does.
Invariants: every cell that can't be computed (a column whose PLY wasn't
    given, or an audit that failed/errored) renders "pending"/"n/a" rather
    than a fabricated number -- AGENTS.md's honesty rule and docs/
    EXPERIMENTS.md's "Jordan's viewer verdict is final" both forbid a table
    that silently hides a missing or failed audit behind a made-up value.
    `None` (column not given, e.g. the distilled ply doesn't exist yet
    because Brush training is still queued) is distinguished from `{"error":
    ...}` (the audit ran and failed) in `build_audit_columns`'s docstring,
    but both render as the honest "no number yet" cell in the table -- the
    distinction lives in `report.json`, not in the printed markdown.
Related docs: docs/EXPERIMENTS.md "Distillation (design B)", "Shade audit",
    "Extent gate"; trippy.eval.audits; trippy.render.report.
"""

from __future__ import annotations

from pathlib import Path

from trippy.eval.audits import audit_report
from trippy.render.report import dark_mass_fraction, extent_p99_max

_PENDING = "pending"
_NA = "n/a"


def build_audit_columns(
    sparse_txt_dir: str | Path,
    baseline_ply: str | Path | None = None,
    trips_export_ply: str | Path | None = None,
    distilled_ply: str | Path | None = None,
    frames: list[str] | None = None,
) -> dict[str, dict | None]:
    """Run `audit_report` for whichever of the three PLYs are given.

    Args:
        sparse_txt_dir: a COLMAP text-format sparse model directory
            (forwarded to `audit_report`'s shade audit).
        baseline_ply: the training run's own source PLY (e.g.
            `kkc_15000.ply`), or None if not available.
        trips_export_ply: the checkpoint's own TRIPS point export (`Trainer.
            export_ply`'s output), or None.
        distilled_ply: the Brush-trained output PLY, or None (the ordinary
            case before that queued training job has finished).
        frames: forwarded to `audit_report` (None uses its own default
            shade-region frames, `SHADE_FRAMES_KK` for a karekare scene).

    Returns:
        `{"baseline": <audit_report(...) or None>, "TRIPS export": ...,
        "distilled": ...}`, in that fixed column order. A `None` value
        means "no PLY given yet"; an `audit_report` dict may itself contain
        `{"error": ...}` per audit if the tool ran but failed -- both cases
        render identically (as "pending") in `audit_comparison_table`.
    """
    columns: dict[str, dict | None] = {}
    for label, ply in (("baseline", baseline_ply), ("TRIPS export", trips_export_ply), ("distilled", distilled_ply)):
        columns[label] = audit_report([str(ply)], sparse_txt_dir, frames=frames) if ply is not None else None
    return columns


def _fmt_pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else _NA


def _fmt(value: float | None, spec: str = ".2f") -> str:
    return format(value, spec) if value is not None else _NA


def _point_count(audits: dict | None) -> int | None:
    """Point count from a column's shade-audit result (`n`), or None if unavailable."""
    if audits is None:
        return None
    shade = audits.get("shade_audit")
    if not isinstance(shade, dict) or "error" in shade:
        return None
    results = shade.get("results") or []
    return int(results[0]["n"]) if results and "n" in results[0] else None


def audit_comparison_table(columns: dict[str, dict | None]) -> str:
    """Markdown table: point count, shade dark-mass fraction, extent p99/max per column.

    Args:
        columns: as returned by `build_audit_columns` (or hand-built with
            the same `{label: audit_report(...) | None}` shape).

    Returns:
        A markdown table with one row per metric and one column per label
        in `columns`' own (insertion) order. Never raises: a `None` column
        or a failed audit both render as "pending"/"n/a" (see module
        docstring), so this always produces a table even when Brush
        training hasn't finished yet.
    """
    labels = list(columns.keys())
    header = "| Metric | " + " | ".join(labels) + " |"
    sep = "|" + "---|" * (len(labels) + 1)

    def _row(name: str, fn) -> str:
        cells = [_PENDING if v is None else fn(v) for v in columns.values()]
        return f"| {name} | " + " | ".join(cells) + " |"

    rows = [
        _row("Point count", lambda v: _fmt(_point_count(v), ",.0f") if _point_count(v) is not None else _NA),
        _row("Shade dark-mass fraction (lum<0.25)", lambda v: _fmt_pct(dark_mass_fraction(v.get("shade_audit")))),
        _row(
            "Extent radius p99",
            lambda v: _fmt(ext[0]) if (ext := extent_p99_max(v.get("extent_gate"))) else _NA,
        ),
        _row(
            "Extent radius max",
            lambda v: _fmt(ext[1]) if (ext := extent_p99_max(v.get("extent_gate"))) else _NA,
        ),
    ]
    return "\n".join([header, sep] + rows)
