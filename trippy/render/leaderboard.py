"""`trippy leaderboard`: one comparison table across every training run so far.

Module: trippy.render.leaderboard
Purpose: scan every run directory under `$TRIPPY_OUTPUT/runs/**/` that has
    finished at least one self-report (`report/report.json` from `trippy
    train --report`, or `candidate/report.json` from `trippy
    candidate-report`) plus a `metrics.jsonl`, and produce ONE markdown +
    PNG table Jordan can open to compare every run against the fixed
    baselines (the raw Gaussian point source, and Design C's render->photo
    U-Net) without hunting through individual run READMEs. `trippy.render.
    report.run_train_report` calls `regenerate_and_deliver` at the very end
    of every `--report` run so this sheet never goes stale.
Invariants:
    - Every field pulled from a run's report.json/metrics.jsonl degrades to
      "n/a" on missing/malformed data rather than raising or fabricating a
      number (AGENTS.md's honesty rule, same contract as
      `trippy.render.report`'s own extraction helpers -- several of which
      this module reuses directly rather than re-deriving the same numbers
      a second way).
    - `discover_run_dirs` never raises on a run directory that is only
      partially written (mid-training, or reporting failed) -- it simply
      excludes it: this module's whole job is "what's real reportable data
      right now", not "every directory that exists".
    - No per-image shade-only PSNR/SSIM/LPIPS is recorded by `Trainer.
      evaluate()` for a trippy-native run (only an aggregate over the whole
      held-out split, forced-shade frames included) -- so the "shade"
      held-out column for a *scanned* run is honestly "n/a", not a
      fabricated split; only the two fixed non-trippy baselines (the raw
      Gaussian render, Design C) carry a real shade-only PSNR breakdown,
      because those come from EXP-0005's own per-bucket eval.
    - `regenerate_and_deliver` never raises past `trippy.render.report.
      run_train_report`'s own end-of-run hook: a broken leaderboard
      rebuild (e.g. a corrupt run directory somewhere under `runs/`) must
      not turn an otherwise-successful training report into a
      `REPORT_FAILED.txt`. Its own failure is caught, printed, and recorded
      in the returned dict instead.
Related docs: docs/EXPERIMENTS.md "Leaderboard"; docs/USER_GUIDE.md "Where
    deliverables appear"; AGENTS.md section 6 ("Deliverables ... go only
    through scripts/deliver.sh").
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

from trippy.config import load_settings
from trippy.constants import (
    CANDIDATE_REPORT_JSON_FILENAME,
    LEADERBOARD_BASELINE_DESIGN_C_LPIPS_ALL,
    LEADERBOARD_BASELINE_DESIGN_C_LPIPS_SHADE,
    LEADERBOARD_BASELINE_DESIGN_C_NAME,
    LEADERBOARD_BASELINE_DESIGN_C_PSNR_ALL,
    LEADERBOARD_BASELINE_DESIGN_C_PSNR_SHADE,
    LEADERBOARD_BASELINE_DESIGN_C_SSIM_ALL,
    LEADERBOARD_BASELINE_DESIGN_C_SSIM_SHADE,
    LEADERBOARD_BASELINE_GAUSSIAN_DARK_MASS,
    LEADERBOARD_BASELINE_GAUSSIAN_EXTENT_MAX,
    LEADERBOARD_BASELINE_GAUSSIAN_EXTENT_P99,
    LEADERBOARD_BASELINE_GAUSSIAN_LPIPS_ALL,
    LEADERBOARD_BASELINE_GAUSSIAN_LPIPS_SHADE,
    LEADERBOARD_BASELINE_GAUSSIAN_NAME,
    LEADERBOARD_BASELINE_GAUSSIAN_PSNR_ALL,
    LEADERBOARD_BASELINE_GAUSSIAN_PSNR_SHADE,
    LEADERBOARD_BASELINE_GAUSSIAN_SSIM_ALL,
    LEADERBOARD_BASELINE_GAUSSIAN_SSIM_SHADE,
    LEADERBOARD_CANDIDATE_REPORT_DIRNAME,
    LEADERBOARD_DELIVER_NAME,
    LEADERBOARD_DELIVER_WHY,
    LEADERBOARD_MARKDOWN_FILENAME,
    LEADERBOARD_OUT_DIRNAME,
    LEADERBOARD_PNG_BASELINE_ROW_BG,
    LEADERBOARD_PNG_BG,
    LEADERBOARD_PNG_CELL_PAD_X,
    LEADERBOARD_PNG_CELL_PAD_Y,
    LEADERBOARD_PNG_FILENAME,
    LEADERBOARD_PNG_FONT_CANDIDATES,
    LEADERBOARD_PNG_FONT_SIZE,
    LEADERBOARD_PNG_HEADER_BG,
    LEADERBOARD_PNG_HEADER_FG,
    LEADERBOARD_PNG_MARGIN,
    LEADERBOARD_PNG_ROW_BG,
    LEADERBOARD_PNG_ROW_BG_ALT,
    LEADERBOARD_PNG_TEXT_COLOR,
    LEADERBOARD_PNG_TITLE_COLOR,
    LEADERBOARD_PNG_TITLE_FONT_SIZE,
    LEADERBOARD_SORT_MISSING_KEY,
    TRAIN_METRICS_FILENAME,
    TRAIN_REPORT_DIRNAME,
)
from trippy.render.report import (
    _deliver,  # reused deliberately: same "hand one artifact to deliver.sh, dry-run-safe" contract.
    dark_mass_fraction,
    dolly_mean_center_coverage,
    extent_p99_max,
)

_EXPERIMENTS_DIRNAME = "experiments"
_HEADERS = [
    "Run",
    "Experiment",
    "Mode",
    "Point source",
    "Epochs",
    "Steps",
    "Held-out all (PSNR/SSIM/LPIPS)",
    "Held-out shade (PSNR/SSIM/LPIPS)",
    "Shade dark-mass %",
    "Extent p99/max",
    "Dolly coverage",
    "Wall time",
    "Viewer launcher",
]
_FOOTNOTES = (
    (
        "Gaussian baseline dark-mass fraction: 19.9% (lum<0.25 opacity mass / mass in the shade "
        "region, kkc_15000.ply, Splats' depthprior_shade_audit.py). Rows sorted by shade dark-mass "
        "ascending (closer to/below 19.9% first), then held-out PSNR descending."
    ),
    (
        "\"Held-out shade\" is n/a for scanned trippy runs: `Trainer.evaluate()` records one "
        "aggregate PSNR/SSIM/LPIPS over the whole held-out split (forced-shade frames included), "
        "not a per-image breakdown -- only the two fixed non-trippy baselines below have a real "
        "shade-only number (EXP-0005's own per-bucket eval)."
    ),
    (
        "Wall time is approximate: run-directory metrics.jsonl creation time to the report.json's "
        "own mtime (no per-step timestamp is recorded in metrics.jsonl)."
    ),
)


def _repo_root() -> Path:
    """Root of this checkout (`.worktrees/<name>` or the main tree) -- this file's own grandparent."""
    return Path(__file__).resolve().parents[2]


def _default_experiments_root() -> Path:
    return _repo_root() / _EXPERIMENTS_DIRNAME


def _default_runs_root() -> Path:
    return load_settings().trippy_output / "runs"


# --- discovery ---


def _report_path(run_dir: Path) -> tuple[Path | None, str | None]:
    """`(path, layout)` for the first of `report/report.json`, `candidate/report.json` that exists."""
    train_report = run_dir / TRAIN_REPORT_DIRNAME / CANDIDATE_REPORT_JSON_FILENAME
    if train_report.exists():
        return train_report, "train_report"
    candidate_report = run_dir / LEADERBOARD_CANDIDATE_REPORT_DIRNAME / CANDIDATE_REPORT_JSON_FILENAME
    if candidate_report.exists():
        return candidate_report, "candidate_report"
    return None, None


def discover_run_dirs(runs_root: Path) -> list[Path]:
    """Every run directory under `runs_root` with both a `metrics.jsonl` and a report.json.

    Args:
        runs_root: `$TRIPPY_OUTPUT/runs` (or a synthetic tmp_path tree in tests).

    Returns:
        Sorted list of run directory paths (deepest-first traversal order,
        then alphabetical) -- directories missing either file (still
        training, or `--report` failed before writing report.json) are
        silently excluded, not an error.
    """
    if not runs_root.is_dir():
        return []
    found = []
    for metrics_path in runs_root.rglob(TRAIN_METRICS_FILENAME):
        run_dir = metrics_path.parent
        report_path, _ = _report_path(run_dir)
        if report_path is not None:
            found.append(run_dir)
    return sorted(set(found))


# --- metrics.jsonl parsing ---


def _read_jsonl_rows(path: Path) -> list[dict]:
    """Every parseable JSON object, one per line; a truncated/corrupt line is skipped, not fatal."""
    rows: list[dict] = []
    try:
        text = path.read_text()
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _metrics_summary(metrics_path: Path) -> dict[str, Any]:
    """`{"last_eval": dict|None, "max_step": int|None, "last_epoch": int|None}` from a run's metrics.jsonl."""
    rows = _read_jsonl_rows(metrics_path)
    eval_rows = [r for r in rows if r.get("eval")]
    train_rows = [r for r in rows if "step" in r]
    last_eval = eval_rows[-1] if eval_rows else None
    max_step = max((r["step"] for r in train_rows), default=None)
    if last_eval is not None:
        last_epoch = last_eval.get("epoch")
    elif train_rows:
        last_epoch = train_rows[-1].get("epoch")
    else:
        last_epoch = None
    return {"last_eval": last_eval, "max_step": max_step, "last_epoch": last_epoch}


# --- matching a run back to the experiments/ config that produced it ---


def _iter_experiment_configs(experiments_root: Path):
    if not experiments_root.is_dir():
        return
    for yaml_path in sorted(experiments_root.glob("*/*.yaml")):
        try:
            data = yaml.safe_load(yaml_path.read_text())
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(data, dict):
            yield data


def match_run_config(experiments_root: Path, experiment: str, run_name: str) -> dict | None:
    """The `experiments/<experiment>/*.yaml` whose own `run_dir` names this run, or None.

    A config's `run_dir` field is written either relative to the repo root
    (`output/runs/<experiment>/<run_name>`) or as an absolute path baked in
    for a specific machine/worktree (EXP-0009's configs, whose comment
    explains why: the job runs from a git worktree with no `.venv` of its
    own) -- neither form is safe to compare byte-for-byte against a
    `runs_root`-relative run directory, so this matches on just the last two
    path components (experiment dir name, run dir name), which both forms
    agree on by construction.

    Args:
        experiments_root: the repo's `experiments/` directory.
        experiment: the run's parent directory name under `runs/`.
        run_name: the run's own directory name.

    Returns:
        The matched config's parsed YAML dict (`TrainConfig`/`HybridCConfig`
        field names), or None if no `experiments/**/*.yaml` claims this run.
    """
    for data in _iter_experiment_configs(experiments_root):
        run_dir_field = data.get("run_dir")
        if not run_dir_field:
            continue
        parts = PurePosixPath(str(run_dir_field).replace("\\", "/")).parts
        if len(parts) >= 2 and parts[-1] == run_name and parts[-2] == experiment:
            return data
    return None


# --- formatting helpers (every one degrades to "n/a" rather than raising) ---


def _fmt_num(value: Any, spec: str = ".2f") -> str:
    return format(value, spec) if isinstance(value, (int, float)) else "n/a"


def _fmt_triplet(row: dict | None) -> str:
    """`"psnr/ssim/lpips"` from an eval-row-shaped dict (`psnr_mean`/`ssim_mean`/`lpips_mean`), or "n/a"."""
    if not row:
        return "n/a"
    return "/".join(
        [
            _fmt_num(row.get("psnr_mean"), ".2f"),
            _fmt_num(row.get("ssim_mean"), ".3f"),
            _fmt_num(row.get("lpips_mean"), ".3f"),
        ]
    )


def _fmt_pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "n/a"


def _fmt_extent(pair: tuple[float, float] | None) -> str:
    return f"{pair[0]:.1f}/{pair[1]:.1f}" if pair is not None else "n/a"


def _fmt_duration(seconds: float | None) -> str:
    """`"~N min"`/`"~N.Nh"` from a wall-clock delta, or "n/a" -- always prefixed `~` (approximate, see module docstring)."""
    if seconds is None or seconds <= 0:
        return "n/a"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"~{minutes:.0f} min"
    return f"~{minutes / 60.0:.1f} h"


def _wall_time_seconds(metrics_path: Path, report_path: Path) -> float | None:
    """metrics.jsonl's own creation time to report.json's mtime, or None if either stat() fails."""
    try:
        start_stat = metrics_path.stat()
        end_stat = report_path.stat()
    except OSError:
        return None
    start = getattr(start_stat, "st_birthtime", None) or start_stat.st_ctime
    delta = end_stat.st_mtime - start
    return delta if delta > 0 else None


def _viewer_launcher_name(report: dict) -> str:
    """The delivered Mac viewer launcher's filename, or "n/a" (candidate-report runs never have one)."""
    for delivery in report.get("deliveries") or []:
        if not isinstance(delivery, dict):
            continue
        if str(delivery.get("name", "")).endswith("-viewer") and delivery.get("artifact"):
            return Path(delivery["artifact"]).name
    bundle = report.get("bundle") or {}
    command_path = (bundle.get("viewer") or {}).get("command_path")
    return Path(command_path).name if command_path else "n/a"


def _candidate_audits(report: dict) -> dict:
    """The `{"shade_audit", "extent_gate"}` dict, whichever report.json layout produced it.

    `train --report`'s report.json nests it under `audits.candidate`;
    `candidate-report`'s own report.json (no baseline audit -- that command
    never audits a baseline PLY) has it directly under `audits`.
    """
    audits = report.get("audits") or {}
    if "shade_audit" in audits or "extent_gate" in audits:
        return audits
    return audits.get("candidate") or {}


# --- one row per run ---


def build_run_row(run_dir: Path, experiments_root: Path) -> dict[str, Any]:
    """Every leaderboard column for one scanned run directory, tolerant of missing fields throughout.

    Args:
        run_dir: a directory `discover_run_dirs` returned (has both
            `metrics.jsonl` and a report.json).
        experiments_root: forwarded to `match_run_config`.

    Returns:
        `{"cells": [str, ...]` (in `_HEADERS` order), `"dark_mass": float|None,
        "psnr_all": float|None, "is_baseline": False}` -- the last two are
        the sort key `rows_to_markdown`/`render_table_png`'s caller uses;
        `cells` is what's actually displayed.
    """
    metrics_path = run_dir / TRAIN_METRICS_FILENAME
    report_path, layout = _report_path(run_dir)
    try:
        report = json.loads(report_path.read_text()) if report_path else {}
    except (OSError, json.JSONDecodeError):
        report = {}

    run_name = run_dir.name
    experiment = run_dir.parent.name
    config = match_run_config(experiments_root, experiment, run_name) or {}

    mode = config.get("mode", "n/a")
    point_source_cfg = config.get("point_source")
    point_source = point_source_cfg.get("type", "n/a") if isinstance(point_source_cfg, dict) else "n/a"
    planned_epochs = config.get("epochs")

    summary = _metrics_summary(metrics_path)
    last_epoch = report.get("epoch") if layout == "train_report" else summary["last_epoch"]
    if last_epoch is not None and planned_epochs is not None:
        epochs_cell = f"{last_epoch}/{planned_epochs}"
    elif last_epoch is not None:
        epochs_cell = str(last_epoch)
    else:
        epochs_cell = "n/a"
    steps_cell = str(summary["max_step"]) if summary["max_step"] is not None else "n/a"

    held_out_all = summary["last_eval"] or report.get("held_out")
    held_out_shade = None  # see module docstring: no per-image split is recorded for scanned runs.

    candidate_audits = _candidate_audits(report)
    dark_mass = dark_mass_fraction(candidate_audits.get("shade_audit"))
    extent = extent_p99_max(candidate_audits.get("extent_gate"))
    dolly_coverage = dolly_mean_center_coverage(report.get("dolly") or {})

    wall_time = _wall_time_seconds(metrics_path, report_path) if report_path else None
    viewer_launcher = _viewer_launcher_name(report) if layout == "train_report" else "n/a"

    display_name = run_name if not is_smoke_run(run_name) else f"{run_name} (smoke)"

    cells = [
        display_name,
        experiment,
        str(mode),
        str(point_source),
        epochs_cell,
        steps_cell,
        _fmt_triplet(held_out_all),
        _fmt_triplet(held_out_shade),
        _fmt_pct(dark_mass),
        _fmt_extent(extent),
        _fmt_num(dolly_coverage, ".4f"),
        _fmt_duration(wall_time),
        viewer_launcher,
    ]
    psnr_all = held_out_all.get("psnr_mean") if held_out_all else None
    return {"cells": cells, "dark_mass": dark_mass, "psnr_all": psnr_all, "is_baseline": False}


def is_smoke_run(run_name: str) -> bool:
    """True if `run_name` looks like a queue-rehearsal smoke run, not a full training run."""
    return "smoke" in run_name.lower()


# --- fixed baseline rows (see trippy.constants "Fixed baseline rows") ---


def _baseline_rows() -> list[dict[str, Any]]:
    gaussian = {
        "cells": [
            LEADERBOARD_BASELINE_GAUSSIAN_NAME,
            "n/a (not a trippy run)",
            "n/a",
            "gaussian (raw, no TRIPS/U-Net)",
            "n/a",
            "n/a",
            "/".join(
                [
                    f"{LEADERBOARD_BASELINE_GAUSSIAN_PSNR_ALL:.2f}",
                    f"{LEADERBOARD_BASELINE_GAUSSIAN_SSIM_ALL:.3f}",
                    f"{LEADERBOARD_BASELINE_GAUSSIAN_LPIPS_ALL:.3f}",
                ]
            ),
            "/".join(
                [
                    f"{LEADERBOARD_BASELINE_GAUSSIAN_PSNR_SHADE:.2f}",
                    f"{LEADERBOARD_BASELINE_GAUSSIAN_SSIM_SHADE:.3f}",
                    f"{LEADERBOARD_BASELINE_GAUSSIAN_LPIPS_SHADE:.3f}",
                ]
            ),
            _fmt_pct(LEADERBOARD_BASELINE_GAUSSIAN_DARK_MASS),
            f"{LEADERBOARD_BASELINE_GAUSSIAN_EXTENT_P99:.1f}/{LEADERBOARD_BASELINE_GAUSSIAN_EXTENT_MAX:.1f}",
            "n/a",
            "n/a (fixed baseline)",
            "n/a",
        ],
        "dark_mass": LEADERBOARD_BASELINE_GAUSSIAN_DARK_MASS,
        "psnr_all": LEADERBOARD_BASELINE_GAUSSIAN_PSNR_ALL,
        "is_baseline": True,
    }
    design_c = {
        "cells": [
            LEADERBOARD_BASELINE_DESIGN_C_NAME,
            "EXP-0005-hybrid-c",
            "n/a (not a PointSource pipeline)",
            "n/a (render->photo U-Net, no PointSource)",
            "1125/2000 (40 min wall-clock cap)",
            "n/a",
            "/".join(
                [
                    f"{LEADERBOARD_BASELINE_DESIGN_C_PSNR_ALL:.2f}",
                    f"{LEADERBOARD_BASELINE_DESIGN_C_SSIM_ALL:.3f}",
                    f"{LEADERBOARD_BASELINE_DESIGN_C_LPIPS_ALL:.3f}",
                ]
            ),
            "/".join(
                [
                    f"{LEADERBOARD_BASELINE_DESIGN_C_PSNR_SHADE:.2f}",
                    f"{LEADERBOARD_BASELINE_DESIGN_C_SSIM_SHADE:.3f}",
                    f"{LEADERBOARD_BASELINE_DESIGN_C_LPIPS_SHADE:.3f}",
                ]
            ),
            "n/a (no points/extent to audit)",
            "n/a (no points/extent to audit)",
            "n/a",
            "n/a (fixed baseline)",
            "n/a",
        ],
        "dark_mass": None,
        "psnr_all": LEADERBOARD_BASELINE_DESIGN_C_PSNR_ALL,
        "is_baseline": True,
    }
    return [gaussian, design_c]


# --- assembling + sorting + rendering the table ---


def _sort_key(row: dict[str, Any]) -> tuple[float, float]:
    dark_mass = row["dark_mass"] if row["dark_mass"] is not None else LEADERBOARD_SORT_MISSING_KEY
    psnr = -row["psnr_all"] if row["psnr_all"] is not None else LEADERBOARD_SORT_MISSING_KEY
    return (dark_mass, psnr)


def build_leaderboard_rows(runs_root: Path | None = None, experiments_root: Path | None = None) -> list[dict[str, Any]]:
    """Every scanned run plus the fixed baselines, sorted per `_sort_key`.

    Args:
        runs_root: defaults to `$TRIPPY_OUTPUT/runs`.
        experiments_root: defaults to this checkout's `experiments/` dir.

    Returns:
        List of row dicts (`build_run_row`'s shape), sorted by shade
        dark-mass fraction ascending then held-out PSNR descending.
    """
    runs_root = runs_root if runs_root is not None else _default_runs_root()
    experiments_root = experiments_root if experiments_root is not None else _default_experiments_root()

    rows = [build_run_row(run_dir, experiments_root) for run_dir in discover_run_dirs(runs_root)]
    rows += _baseline_rows()
    rows.sort(key=_sort_key)
    return rows


def rows_to_markdown(rows: list[dict[str, Any]]) -> str:
    """The full markdown leaderboard: header, one row per run/baseline, footnotes."""
    lines = [
        "# TRIPS leaderboard",
        "",
        (
            "One comparison table across every training run with a self-report, plus the fixed "
            "non-trippy baselines. Sorted by shade dark-mass fraction ascending, then held-out "
            "PSNR descending. Regenerated by `trippy leaderboard` and, automatically, at the end "
            "of every `trippy train --report` run."
        ),
        "",
        "| " + " | ".join(_HEADERS) + " |",
        "|" + "---|" * len(_HEADERS),
    ]
    for row in rows:
        lines.append("| " + " | ".join(row["cells"]) + " |")
    lines.append("")
    for note in _FOOTNOTES:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


# --- PNG rendering (PIL only; see trippy.constants "PNG table rendering") ---


def _load_table_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in LEADERBOARD_PNG_FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_table_png(headers: list[str], rows: list[dict[str, Any]], out_path: Path, title: str) -> Path:
    """Render a headers+rows table as a PNG (PIL only), one row band per entry.

    Args:
        headers: column headers, top row.
        rows: `build_leaderboard_rows`' shape (`"cells"` displayed,
            `"is_baseline"` tints the row so fixed baselines are visually
            distinct from scanned runs).
        out_path: `.png` path; parent directories are created if missing.
        title: drawn above the table.

    Returns:
        `out_path`.
    """
    font = _load_table_font(LEADERBOARD_PNG_FONT_SIZE)
    title_font = _load_table_font(LEADERBOARD_PNG_TITLE_FONT_SIZE)

    measure_img = Image.new("RGB", (1, 1))
    measure_draw = ImageDraw.Draw(measure_img)

    def _text_w(text: str, f) -> int:
        bbox = measure_draw.textbbox((0, 0), text, font=f)
        return bbox[2] - bbox[0]

    all_rows = [headers] + [row["cells"] for row in rows]
    n_cols = len(headers)
    col_widths = [0] * n_cols
    for cells in all_rows:
        for i, cell in enumerate(cells):
            col_widths[i] = max(col_widths[i], _text_w(str(cell), font))
    col_widths = [w + LEADERBOARD_PNG_CELL_PAD_X * 2 for w in col_widths]

    row_h = LEADERBOARD_PNG_FONT_SIZE + LEADERBOARD_PNG_CELL_PAD_Y * 2
    title_h = LEADERBOARD_PNG_TITLE_FONT_SIZE + LEADERBOARD_PNG_CELL_PAD_Y * 2

    table_w = sum(col_widths)
    width = table_w + LEADERBOARD_PNG_MARGIN * 2
    height = title_h + row_h * len(all_rows) + LEADERBOARD_PNG_MARGIN * 2

    img = Image.new("RGB", (width, height), color=LEADERBOARD_PNG_BG)
    draw = ImageDraw.Draw(img)

    y = LEADERBOARD_PNG_MARGIN
    draw.text((LEADERBOARD_PNG_MARGIN, y), title, font=title_font, fill=LEADERBOARD_PNG_TITLE_COLOR)
    y += title_h

    for row_idx, cells in enumerate(all_rows):
        is_header = row_idx == 0
        is_baseline = (not is_header) and rows[row_idx - 1]["is_baseline"]
        if is_header:
            bg = LEADERBOARD_PNG_HEADER_BG
        elif is_baseline:
            bg = LEADERBOARD_PNG_BASELINE_ROW_BG
        elif row_idx % 2 == 0:
            bg = LEADERBOARD_PNG_ROW_BG_ALT
        else:
            bg = LEADERBOARD_PNG_ROW_BG
        draw.rectangle([LEADERBOARD_PNG_MARGIN, y, LEADERBOARD_PNG_MARGIN + table_w, y + row_h], fill=bg)

        x = LEADERBOARD_PNG_MARGIN
        fg = LEADERBOARD_PNG_HEADER_FG if is_header else LEADERBOARD_PNG_TEXT_COLOR
        for col_idx, cell in enumerate(cells):
            draw.text(
                (x + LEADERBOARD_PNG_CELL_PAD_X, y + LEADERBOARD_PNG_CELL_PAD_Y),
                str(cell),
                font=font,
                fill=fg,
            )
            x += col_widths[col_idx]
        y += row_h

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


# --- writing + delivering ---


def default_leaderboard_out_dir() -> Path:
    return load_settings().trippy_output / LEADERBOARD_OUT_DIRNAME


def write_leaderboard(
    out_dir: Path,
    runs_root: Path | None = None,
    experiments_root: Path | None = None,
) -> dict[str, Any]:
    """Build the rows and write both `leaderboard.md` and `leaderboard.png` under `out_dir`.

    Args:
        out_dir: output directory (created if missing).
        runs_root: forwarded to `build_leaderboard_rows`.
        experiments_root: forwarded to `build_leaderboard_rows`.

    Returns:
        `{"rows": list[dict], "markdown_path": Path, "png_path": Path}`.
    """
    out_dir = Path(out_dir)
    rows = build_leaderboard_rows(runs_root, experiments_root)

    markdown_path = out_dir / LEADERBOARD_MARKDOWN_FILENAME
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(rows_to_markdown(rows))

    png_path = render_table_png(_HEADERS, rows, out_dir / LEADERBOARD_PNG_FILENAME, title="TRIPS leaderboard")

    return {"rows": rows, "markdown_path": markdown_path, "png_path": png_path}


def regenerate_and_deliver(
    out_dir: Path | None = None,
    runs_root: Path | None = None,
    experiments_root: Path | None = None,
) -> dict[str, Any]:
    """`write_leaderboard` plus delivery of the PNG under the fixed `trips-leaderboard` name.

    Called at the end of every `trippy train --report` run (`trippy.render.
    report.run_train_report`) so Jordan always has one up-to-date sheet;
    `scripts/deliver.sh` (via `review_add.sh`'s `ln -sfn`) replaces the same
    symlink each time rather than accumulating one per run.
    `TRIPPY_DELIVER_DRY_RUN=1` skips the subprocess exactly like every other
    delivery in this codebase (`trippy.render.report._deliver`).

    Returns:
        `write_leaderboard`'s dict plus `"delivery"`.
    """
    out_dir = out_dir if out_dir is not None else default_leaderboard_out_dir()
    result = write_leaderboard(out_dir, runs_root, experiments_root)
    result["delivery"] = _deliver(result["png_path"], LEADERBOARD_DELIVER_NAME, LEADERBOARD_DELIVER_WHY)
    return result


def regenerate_and_deliver_safely(**kwargs: Any) -> dict[str, Any]:
    """`regenerate_and_deliver`, never raising -- see module docstring's hook invariant.

    Used by `trippy.render.report.run_train_report`'s end-of-run hook: a
    broken leaderboard rebuild must not cost Jordan an otherwise-successful
    training report.
    """
    try:
        return regenerate_and_deliver(**kwargs)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        print(f"trippy leaderboard: regenerate_and_deliver FAILED (report itself still succeeded): {exc}")
        return {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
