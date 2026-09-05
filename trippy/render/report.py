"""Self-reporting: `trippy train --report`'s comparison table, summary line, and delivery.

Module: trippy.render.report
Purpose: turn a finished `Trainer.fit()` run into something Jordan can open
    with no extra step (this task's brief: "no human or orchestrator step
    is needed between 'training finished' and 'Jordan has something to
    open'"). `run_train_report` runs the same per-checkpoint pipeline
    `trippy candidate-report` does (export already done by `Trainer.fit`,
    Splats' shade/extent audits, shade dolly video, off-path honesty
    sheet), adds a cached baseline audit of the training run's own source
    PLY, appends a baseline-vs-candidate comparison table to the run's own
    `README.md`, and delivers `dolly.mp4` + `honesty_sheet.png` +
    `export.ply` via `scripts/deliver.sh` with one honest summary line.
Invariants:
    - Every function here that reads a metrics/audit dict degrades to
      `None`/"n/a" on missing or `{"error": ...}` data rather than raising
      or fabricating a number -- `docs/EXPERIMENTS.md`'s "Jordan's viewer
      verdict is final" and AGENTS.md's honesty rule both forbid a report
      that silently hides a failed audit behind a made-up value.
    - `run_train_report` itself may raise (a checkpoint that fails to
      reload, a scene whose sparse dir is missing, ...) -- callers
      (`trippy.cli._cmd_train`) are responsible for catching that and
      writing `TRAIN_REPORT_FAILED_FILENAME` per this task's brief
      requirement 1 ("--report never crashes the run"); this module does
      not swallow its own top-level errors, only the per-audit ones that
      `trippy.eval.audits.audit_report`/`cached_baseline_audit` already
      catch.
    - `TRIPPY_DELIVER_DRY_RUN=1` skips the `scripts/deliver.sh` subprocess
      entirely (no artifact-under-TRIPPY_OUTPUT path check, no
      `research/trips-metal.md` write) -- set by tests so the CPU suite
      never touches Splats' review queue or the real research log.
Related docs: docs/EXPERIMENTS.md "Training runs", "Candidate report",
    "Dolly camera paths"; AGENTS.md section 6 ("Deliverables ... go only
    through scripts/deliver.sh").
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from trippy.constants import (
    CANDIDATE_HONESTY_SHEET_FILENAME,
    CANDIDATE_NET_VIDEO_FILENAME,
    CANDIDATE_REPORT_DOLLY_DIRNAME,
    CANDIDATE_REPORT_JSON_FILENAME,
    CANDIDATE_REPORT_OFFPATH_DIRNAME,
    DELIVER_SUBPROCESS_TIMEOUT_S,
    SHADE_AUDIT_DARK_MASS_LUM_KEY,
    TRAIN_CHECKPOINT_LATEST_FILENAME,
    TRAIN_EXPORT_FILENAME,
    TRAIN_REPORT_DIRNAME,
)
from trippy.eval.audits import audit_report, cached_baseline_audit
from trippy.render.candidate import render_candidate
from trippy.render.dolly import shade_dolly_poses
from trippy.render.offpath import offpath_poses

if TYPE_CHECKING:
    from trippy.train.trainer import Trainer

_DELIVER_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deliver.sh"
_RUN_README_FILENAME = "README.md"
_SPARSE_TXT_DIRNAME = "sparse_txt"  # matches _cmd_candidate_report's own literal (audits need COLMAP text)


# --- pure numeric extraction helpers: every one degrades to None on missing/error data ---


def dolly_mean_center_coverage(dolly_metrics: dict) -> float | None:
    """Mean `coverage_mean_center` over the dolly frames actually kept in the video.

    Uses `dolly_metrics["dolly_stop_index"]` when present (i.e.
    `render_candidate` was called with `stop_at_low_coverage=True`) so this
    number describes the path Jordan will actually see in `dolly.mp4`, not
    frames that were rendered but cut for drifting through empty space.
    """
    frames = dolly_metrics.get("frames") if isinstance(dolly_metrics, dict) else None
    if not frames:
        return None
    stop_index = dolly_metrics.get("dolly_stop_index")
    kept = frames[: stop_index + 1] if isinstance(stop_index, int) else frames
    values = [f["coverage_mean_center"] for f in kept if "coverage_mean_center" in f]
    return float(np.mean(values)) if values else None


def _first_shade_result(shade_audit: dict | None) -> dict | None:
    if not isinstance(shade_audit, dict) or "error" in shade_audit:
        return None
    results = shade_audit.get("results")
    return results[0] if results else None


def dark_mass_fraction(shade_audit: dict | None) -> float | None:
    """`dark_mass_lum0.25 / mass_in_region` from a `run_shade_audit` result, or None."""
    result = _first_shade_result(shade_audit)
    if result is None:
        return None
    mass = result.get("mass_in_region")
    dark = result.get(SHADE_AUDIT_DARK_MASS_LUM_KEY)
    if not mass or dark is None:
        return None
    return float(dark) / float(mass)


def _first_extent_record(extent_gate: dict | None) -> dict | None:
    if not isinstance(extent_gate, dict) or "error" in extent_gate:
        return None
    plys = extent_gate.get("plys")
    return plys[0] if plys else None


def extent_p99_max(extent_gate: dict | None) -> tuple[float, float] | None:
    """`(radius_p99, radius_max)` from a `run_extent_gate` result, or None."""
    record = _first_extent_record(extent_gate)
    if record is None or "radius_p99" not in record or "radius_max" not in record:
        return None
    return float(record["radius_p99"]), float(record["radius_max"])


def _fmt(value: float | None, spec: str = ".2f") -> str:
    return format(value, spec) if value is not None else "n/a"


def _fmt_pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "n/a"


# --- comparison table + summary line ---


def comparison_table_markdown(
    held_out_metrics: dict,
    candidate_audits: dict,
    baseline_audits: dict,
    dolly_metrics: dict,
) -> str:
    """Markdown table: held-out PSNR/SSIM/LPIPS, dark-mass fraction, extent p99/max, dolly coverage.

    Args:
        held_out_metrics: `Trainer.evaluate`'s return value (candidate only
            -- a baseline PLY has no trained model to hold images out from).
        candidate_audits: `audit_report` output for the trained export PLY.
        baseline_audits: `cached_baseline_audit` output for the training
            run's own source PLY.
        dolly_metrics: `render_candidate`'s dolly-path metrics dict.

    Returns:
        A markdown table (never raises on missing/error data -- every cell
        that can't be computed reads "n/a", so the table still renders per
        this task's brief requirement 6).
    """
    psnr = held_out_metrics.get("psnr_mean")
    ssim = held_out_metrics.get("ssim_mean")
    lpips = held_out_metrics.get("lpips_mean")

    candidate_dark = dark_mass_fraction(candidate_audits.get("shade_audit"))
    baseline_dark = dark_mass_fraction(baseline_audits.get("shade_audit"))

    candidate_extent = extent_p99_max(candidate_audits.get("extent_gate"))
    baseline_extent = extent_p99_max(baseline_audits.get("extent_gate"))

    dolly_coverage = dolly_mean_center_coverage(dolly_metrics)

    rows = [
        ("Held-out PSNR (dB)", "n/a", _fmt(psnr)),
        ("Held-out SSIM", "n/a", _fmt(ssim, ".4f")),
        ("Held-out LPIPS", "n/a", _fmt(lpips, ".4f") if lpips is not None else "n/a"),
        (
            "Shade dark-mass fraction (lum<0.25)",
            _fmt_pct(baseline_dark),
            _fmt_pct(candidate_dark),
        ),
        (
            "Extent radius p99",
            _fmt(baseline_extent[0]) if baseline_extent else "n/a",
            _fmt(candidate_extent[0]) if candidate_extent else "n/a",
        ),
        (
            "Extent radius max",
            _fmt(baseline_extent[1]) if baseline_extent else "n/a",
            _fmt(candidate_extent[1]) if candidate_extent else "n/a",
        ),
        ("Dolly mean centre coverage (kept path)", "n/a", _fmt(dolly_coverage, ".4f")),
    ]

    lines = ["| Metric | Baseline | Candidate |", "|---|---|---|"]
    lines += [f"| {name} | {base} | {cand} |" for name, base, cand in rows]

    notes: list[str] = []
    if "error" in candidate_audits.get("shade_audit", {}) or "error" in candidate_audits.get(
        "extent_gate", {}
    ):
        notes.append(
            "- Candidate audits: "
            + "; ".join(
                f"{k} FAILED -- {v['error']}"
                for k, v in candidate_audits.items()
                if isinstance(v, dict) and "error" in v
            )
        )
    if "error" in baseline_audits.get("shade_audit", {}) or "error" in baseline_audits.get(
        "extent_gate", {}
    ):
        notes.append(
            "- Baseline audits: "
            + "; ".join(
                f"{k} FAILED -- {v['error']}"
                for k, v in baseline_audits.items()
                if isinstance(v, dict) and "error" in v
            )
        )

    return "\n".join(lines + ([""] + notes if notes else []))


def summary_line(run_name: str, epoch: int, held_out_metrics: dict, candidate_audits: dict, baseline_audits: dict) -> str:
    """One honest line: epoch, held-out PSNR, dark-mass fraction vs baseline -- no verdict language.

    This is the exact text handed to `scripts/deliver.sh` as the delivery
    "why" (this task's brief: "an honest one-line summary containing the
    key numbers" -- PSNR, dark-mass fraction vs baseline, and epochs; no
    "looks good").
    """
    psnr = held_out_metrics.get("psnr_mean")
    candidate_dark = dark_mass_fraction(candidate_audits.get("shade_audit"))
    baseline_dark = dark_mass_fraction(baseline_audits.get("shade_audit"))
    return (
        f"trippy train report {run_name}: epoch {epoch}, held-out PSNR {_fmt(psnr)} dB, "
        f"shade dark-mass {_fmt_pct(candidate_dark)} vs baseline {_fmt_pct(baseline_dark)}"
    )


# --- delivery ---


def _deliver(artifact: Path, name: str, why: str) -> dict:
    """Hand one artifact to `scripts/deliver.sh`, or record a no-op under `TRIPPY_DELIVER_DRY_RUN=1`."""
    record: dict = {"artifact": str(artifact), "name": name}
    if not artifact.exists():
        record["status"] = "skipped: artifact not found"
        return record
    if os.environ.get("TRIPPY_DELIVER_DRY_RUN") == "1":
        record["status"] = "skipped: TRIPPY_DELIVER_DRY_RUN=1"
        return record
    result = subprocess.run(
        ["bash", str(_DELIVER_SCRIPT), str(artifact), name, why],
        capture_output=True,
        text=True,
        timeout=DELIVER_SUBPROCESS_TIMEOUT_S,
        check=False,
    )
    record["returncode"] = result.returncode
    record["status"] = "delivered" if result.returncode == 0 else "failed"
    if result.returncode != 0:
        record["stderr"] = result.stderr
    return record


# --- orchestration ---


def _baseline_ply_audits(cfg, sparse_txt_dir: Path) -> dict:
    """`cached_baseline_audit` on `cfg.point_source`'s own PLY, or a recorded error for other types."""
    point_source = cfg.point_source
    if point_source.type != "gaussian" or not point_source.path:
        error = {
            "error": (
                "baseline audit needs point_source.type == 'gaussian' with a path; "
                f"got type={point_source.type!r} path={point_source.path!r}"
            )
        }
        return {"shade_audit": error, "extent_gate": error}
    return cached_baseline_audit(point_source.path, sparse_txt_dir, frames=None)


def _ensure_run_readme(run_dir: Path) -> Path:
    readme_path = run_dir / _RUN_README_FILENAME
    if not readme_path.exists():
        readme_path.write_text(f"# {run_dir.name}\n\nSee `metrics.jsonl`/`log.txt` for the full run.\n")
    return readme_path


def run_train_report(trainer: Trainer, held_out_metrics: dict) -> dict:
    """Build and deliver the self-report for a finished `Trainer.fit()` run.

    Args:
        trainer: the `Trainer` after `fit()` has returned (its final
            checkpoint and `export.ply` must already exist -- `Trainer.fit`
            guarantees both).
        held_out_metrics: `fit()`'s return value (the most recent `evaluate()`
            call's metrics dict; may be `{}` if a `max_minutes` budget
            expired before the first eval).

    Returns:
        `{"checkpoint", "device", "scene_root", "export_ply", "epoch",
        "held_out", "dolly", "offpath", "audits": {"candidate", "baseline"},
        "summary_line", "deliveries"}` -- also written to
        `<run_dir>/report/report.json`, with the comparison table + summary
        line appended to `<run_dir>/README.md`.
    """
    run_dir = Path(trainer.run_dir)
    out_dir = run_dir / TRAIN_REPORT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = trainer.checkpoint_dir / TRAIN_CHECKPOINT_LATEST_FILENAME
    export_path = run_dir / TRAIN_EXPORT_FILENAME
    scene_root = Path(trainer.cfg.scene_root)
    device = str(trainer.device)
    width = trainer.cfg.width
    epoch = int(held_out_metrics.get("epoch", trainer.epoch))

    forced = list(trainer.cfg.forced_heldout)
    dolly_pose_name = forced[0] if forced else trainer.dataset.names[0]
    offpath_names = forced if forced else [trainer.dataset.names[0]]

    dolly_poses = shade_dolly_poses(scene_root, pose_name=dolly_pose_name, width=width)
    offpath_pose_list = offpath_poses(scene_root, offpath_names, width=width)

    dolly_metrics = render_candidate(
        checkpoint_path,
        dolly_poses,
        out_dir / CANDIDATE_REPORT_DOLLY_DIRNAME,
        device=device,
        write_video_files=True,
        stop_at_low_coverage=True,
    )
    offpath_metrics = render_candidate(
        checkpoint_path,
        offpath_pose_list,
        out_dir / CANDIDATE_REPORT_OFFPATH_DIRNAME,
        device=device,
        write_video_files=False,
    )

    sparse_txt_dir = scene_root / _SPARSE_TXT_DIRNAME
    candidate_audits = audit_report([str(export_path)], sparse_txt_dir, frames=None)
    baseline_audits = _baseline_ply_audits(trainer.cfg, sparse_txt_dir)

    run_name = run_dir.name
    line = summary_line(run_name, epoch, held_out_metrics, candidate_audits, baseline_audits)
    table = comparison_table_markdown(held_out_metrics, candidate_audits, baseline_audits, dolly_metrics)

    readme_path = _ensure_run_readme(run_dir)
    with open(readme_path, "a") as f:
        f.write(f"\n## Report: epoch {epoch}\n\n{line}\n\n{table}\n")

    dolly_mp4 = out_dir / CANDIDATE_REPORT_DOLLY_DIRNAME / CANDIDATE_NET_VIDEO_FILENAME
    honesty_sheet = out_dir / CANDIDATE_REPORT_DOLLY_DIRNAME / CANDIDATE_HONESTY_SHEET_FILENAME
    deliveries = [
        _deliver(dolly_mp4, f"{run_name}-dolly", line),
        _deliver(honesty_sheet, f"{run_name}-honesty", line),
        _deliver(export_path, f"{run_name}-export", line),
    ]

    report = {
        "checkpoint": str(checkpoint_path),
        "device": device,
        "scene_root": str(scene_root),
        "export_ply": str(export_path),
        "epoch": epoch,
        "held_out": held_out_metrics,
        "dolly": dolly_metrics,
        "offpath": offpath_metrics,
        "audits": {"candidate": candidate_audits, "baseline": baseline_audits},
        "summary_line": line,
        "deliveries": deliveries,
    }
    (out_dir / CANDIDATE_REPORT_JSON_FILENAME).write_text(json.dumps(report, indent=2) + "\n")
    return report
