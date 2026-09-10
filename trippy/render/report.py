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
    `README.md`, exports a free-navigation viewer bundle from the same
    final checkpoint (Jordan: "fixed dolly paths are hard to judge, I want
    to navigate freely") with a Mac double-click launcher, and delivers the
    launcher + `dolly.mp4` + `honesty_sheet.png` + `export.ply` via
    `scripts/deliver.sh`, launcher first, all with one honest summary line.
    `export_bundle_and_viewer_launcher` (bundle export + launcher + delivery)
    is factored out so `trippy bundle-launcher` can run the same three steps
    against any existing checkpoint, not just a run's own final one.
Invariants:
    - Every function here that reads a metrics/audit dict degrades to
      `None`/"n/a" on missing or `{"error": ...}` data rather than raising
      or fabricating a number -- `docs/EXPERIMENTS.md`'s "Jordan's viewer
      verdict is final" and AGENTS.md's honesty rule both forbid a report
      that silently hides a failed audit behind a made-up value.
    - `run_train_report` itself may raise (a checkpoint that fails to
      reload, a scene whose sparse dir is missing, a bundle export that
      fails, ...) -- callers (`trippy.cli._cmd_train`) are responsible for
      catching that and writing `TRAIN_REPORT_FAILED_FILENAME` per this
      task's brief requirement 1 ("--report never crashes the run"); this
      module does not swallow its own top-level errors, only the per-audit
      ones that `trippy.eval.audits.audit_report`/`cached_baseline_audit`
      already catch, and the viewer-launcher step (`build_mac_viewer_launcher`),
      which never raises by its own contract -- a missing/stale viewer
      binary must not cost Jordan the rest of an otherwise-successful report.
    - `TRIPPY_DELIVER_DRY_RUN=1` skips the `scripts/deliver.sh` subprocess
      entirely (no artifact-under-TRIPPY_OUTPUT path check, no
      `research/trips-metal.md` write), printing the command that would
      have run instead -- set by tests so the CPU suite never touches
      Splats' review queue or the real research log.
Related docs: docs/EXPERIMENTS.md "Training runs", "Candidate report",
    "Dolly camera paths", "Self-reporting training runs"; docs/USER_GUIDE.md
    "How to open the native TRIPS viewer (Mac)", "How to make a scene
    openable in the native viewer"; AGENTS.md section 6 ("Deliverables ...
    go only through scripts/deliver.sh").
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from trippy.config import load_settings
from trippy.constants import (
    AUDIT_MODEL_CONVERTER_TIMEOUT_S,
    AUDIT_SPARSE_TXT_CACHE_SUBDIR,
    CANDIDATE_HONESTY_SHEET_FILENAME,
    CANDIDATE_NET_VIDEO_FILENAME,
    CANDIDATE_REPORT_DOLLY_DIRNAME,
    CANDIDATE_REPORT_JSON_FILENAME,
    CANDIDATE_REPORT_OFFPATH_DIRNAME,
    DELIVER_SUBPROCESS_TIMEOUT_S,
    SHADE_AUDIT_DARK_MASS_LUM_KEY,
    SHADE_FRAMES_KK,
    TRAIN_CHECKPOINT_DIRNAME,
    TRAIN_CHECKPOINT_LATEST_FILENAME,
    TRAIN_EXPORT_FILENAME,
    TRAIN_REPORT_BUNDLE_DIRNAME,
    TRAIN_REPORT_DIRNAME,
    VIEWER_DELIVERY_WHY_SUFFIX,
    VIEWER_LAUNCHER_FAILED_FILENAME,
)
from trippy.eval.audits import audit_report, cached_baseline_audit
from trippy.render.bundle import export_bundle
from trippy.render.candidate import render_candidate
from trippy.render.dolly import shade_dolly_poses
from trippy.render.offpath import offpath_poses

if TYPE_CHECKING:
    from trippy.train.trainer import Trainer

_DELIVER_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deliver.sh"
_OPEN_MAC_VIEWER_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "open_mac_viewer.sh"
_TRIPS_VIEWER_BINARY_REL = Path("rust") / "target" / "release" / "trips-viewer"
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


def _convert_sparse_bin_to_txt(bin_dir: Path, out_dir: Path) -> None:
    """`colmap model_converter --output_type TXT`, `bin_dir` -> `out_dir` (created if needed).

    Raises:
        FileNotFoundError: no `colmap` binary on PATH.
        RuntimeError: the subprocess exited non-zero.
    """
    colmap = shutil.which("colmap")
    if colmap is None:
        raise FileNotFoundError("colmap not found on PATH (needed to convert a binary sparse model to TEXT)")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap,
        "model_converter",
        "--input_path",
        str(bin_dir),
        "--output_path",
        str(out_dir),
        "--output_type",
        "TXT",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=AUDIT_MODEL_CONVERTER_TIMEOUT_S, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"colmap model_converter failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def resolve_sparse_txt_dir(scene_root: Path, override: str = "") -> Path:
    """Pick the COLMAP TEXT sparse dir the Splats shade/extent audits read.

    Bug this closes: `run_train_report`/`_cmd_candidate_report` used to hardcode
    `<scene_root>/sparse_txt` unconditionally. Every karekare-v2 (EXP-0011) audit
    silently fell back to `depthprior_shade_audit.py`'s OWN default scene
    (`~/Splats/scenes/karekare/kk-coherent/sparse_txt`) once that path raised
    `FileNotFoundError` deeper in `trippy.eval.audits._run` -- no, it did not even
    get that far in some paths; see `trippy-shade-audit-rerun2` job log -- so every
    reported dark-mass number was measured against the WRONG scene's sparse model
    and the wrong (kk-coherent, 6-frame) shade region. karekare-v2 has no
    `sparse_txt` at all, only 16 binary `sparse/<n>` models (COLMAP writes text
    only when asked); `trippy.scene.dataset.resolve_sparse_dir` already prefers
    `sparse/0` for training for this reason.

    Args:
        scene_root: the scene's root directory (`<scene_root>/sparse/0` and/or
            `<scene_root>/sparse_txt`).
        override: `TrainConfig.sparse_txt`. "" (default) auto-resolves per below;
            any other value is returned as-is (Path(override)), letting a config
            pin a specific converted directory or a non-default sparse sub-model.

    Returns:
        `Path(override)` when non-empty; else `<scene_root>/sparse_txt` if that
        already exists; else, when `<scene_root>/sparse/0` is a binary model,
        the path to an on-disk TEXT conversion of it under
        `$TRIPPY_OUTPUT/<AUDIT_SPARSE_TXT_CACHE_SUBDIR>/<scene_root.name>/sparse_txt`
        (converted once with `colmap model_converter`, re-used on every later
        call for the same scene name -- never written into `scene_root` itself,
        per AGENTS.md "never write into Splats' scene dirs"); else
        `<scene_root>/sparse_txt` unchanged (old behaviour: a missing path,
        left for the caller/subprocess to raise on).

    Raises:
        FileNotFoundError: `<scene_root>/sparse/0` is binary but `colmap` is not
            on PATH.
        RuntimeError: the `colmap model_converter` subprocess exited non-zero.
    """
    if override:
        return Path(override)

    scene_root = Path(scene_root)
    native_txt = scene_root / _SPARSE_TXT_DIRNAME
    if native_txt.exists():
        return native_txt

    bin_dir = scene_root / "sparse" / "0"
    if (bin_dir / "cameras.bin").exists() and (bin_dir / "images.bin").exists():
        cache_dir = load_settings().trippy_output / AUDIT_SPARSE_TXT_CACHE_SUBDIR / scene_root.name / _SPARSE_TXT_DIRNAME
        marker = cache_dir / "cameras.txt"
        if not marker.exists():
            _convert_sparse_bin_to_txt(bin_dir, cache_dir)
        return cache_dir

    return native_txt


def resolve_shade_frames(value: list[str] | str | None, base_dir: Path | None = None) -> list[str] | None:
    """Turn `TrainConfig.shade_frames` into a frame-name list for `run_shade_audit`'s `--frames`.

    This is the fix for the bug this task closes: `run_train_report` used to call
    `audit_report`/`cached_baseline_audit` with `frames=None` unconditionally, so every
    scene's shade dark-mass was measured on `depthprior_shade_audit.py`'s own default
    (`SHADE_FRAMES_KK`, kk-coherent's IMG_3828-3833) even on scenes -- karekare-v2 chief
    among them -- whose actual shade region is a different, measured set of frames
    (`experiments/EXP-0011-karekare-v2/README.md` "Finding the shade frames"). Every
    EXP-0011 config now sets `shade_frames` to that measured 93-frame big-tree list, and
    this function is what turns the config value into the list `run_shade_audit` needs.

    Args:
        value: `TrainConfig.shade_frames` -- `None` (old/unchanged behaviour: return
            `None`, so callers pass it straight through and the script's own default
            applies), a `list[str]` (returned verbatim), or a `str` path to either:
              - a `.json` file holding a bare list, or a dict with a "frames"/
                "big_tree"/"shade_frames" list key (matches
                `$TRIPPY_OUTPUT/scratch/shade_frames.json`'s own shape, so that exact
                file can be pointed to directly without reshaping it), or
              - a plain-text file, one frame name per line, blank lines and
                `#`-prefixed comments ignored.
        base_dir: directory a relative `value` path is resolved against (unused for an
            already-absolute path, a list, or `None`).

    Returns:
        `None` (script default applies) or the resolved list of frame names.

    Raises:
        FileNotFoundError: `value` is a path and it does not exist.
        ValueError: `value` is a `.json` path whose top-level shape has no usable list
            (not a bare list, and no "frames"/"big_tree"/"shade_frames" list key).
    """
    if value is None or isinstance(value, list):
        return value
    path = Path(value)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.exists():
        raise FileNotFoundError(f"shade_frames path not found: {path}")
    text = path.read_text()
    if path.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("frames", "big_tree", "shade_frames"):
                if isinstance(data.get(key), list):
                    return data[key]
        raise ValueError(
            f"{path}: expected a JSON list, or a dict with a 'frames'/'big_tree'/'shade_frames' list key"
        )
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def shade_frames_used_record(resolved: list[str] | None) -> dict:
    """`report.json["shade_frames"]`: which frames the shade audit actually used.

    Written unconditionally (not only when `TrainConfig.shade_frames` is set) so a report
    is never ambiguous about which shade region its dark-mass numbers describe -- the bug
    this task closes was exactly this ambiguity going unrecorded (a karekare-v2 report
    silently carrying kk-coherent's frame list). `resolved=None` means
    `depthprior_shade_audit.py`'s own default applied (`SHADE_FRAMES_KK`); that default is
    spelled out explicitly here rather than left as a bare `null`.
    """
    if resolved is None:
        return {
            "source": "script default (trippy.constants.SHADE_FRAMES_KK, kk-coherent)",
            "count": len(SHADE_FRAMES_KK),
            "frames": list(SHADE_FRAMES_KK),
        }
    return {"source": "config shade_frames", "count": len(resolved), "frames": resolved}


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


def gate_summary(held_out_metrics: dict, dolly_metrics: dict | None = None) -> dict | None:
    """The blend gate's numbers for `report.json`, or None on a gate-less run.

    Pulls the held-out eval's gate block (mean/percentiles of the per-pixel
    splat weight, plus the `gate_scale` it was rendered at) and, when the dolly
    render produced one, its own -- the two answer different questions: the eval
    gate is measured at photographed poses where a precomputed splat render
    exists, the dolly gate at poses where the splat had to be rendered live (or
    could not be, `n_frames_without_splat`).

    Returns None -- and the caller then writes no "gate" key at all -- when the
    run was trained without the gate, so an old report and a gate-off report are
    the same document they always were.
    """
    heldout = held_out_metrics.get("gate")
    dolly = (dolly_metrics or {}).get("gate")
    if not heldout and not dolly:
        return None
    out: dict = {}
    if heldout:
        out["held_out"] = heldout
    if dolly:
        out["dolly"] = dolly
    return out


def gate_markdown(gate: dict | None) -> str:
    """One markdown line per gate block, or "" when the run had no gate.

    Goes into the run README under the comparison table so the mix is visible
    next to the PSNR it produced -- a gate mean is not a quality number and must
    not be read as one, but "17.9 dB with 80% of the pixels coming from the
    splat" and "17.9 dB with 5%" are very different results.
    """
    if not gate:
        return ""
    lines = ["", "### Blend gate (0 = TRIPS, 1 = splat)", ""]
    for where, block in gate.items():
        pcts = block.get("percentiles", {})
        spread = " ".join(f"{k}={v:.3f}" for k, v in pcts.items())
        extra = ""
        if block.get("n_frames_without_splat"):
            extra = f", {block['n_frames_without_splat']} frame(s) had no splat to blend"
        lines.append(
            f"- **{where}** ({block.get('n_frames', 0)} frames, gate_scale "
            f"{block.get('gate_scale', 1.0):g}): mean {block['mean']:.3f}, {spread}{extra}"
        )
    return "\n".join(lines) + "\n"


def heldout_split(held_out_metrics: dict) -> dict:
    """`{"shade": {...}, "other": {...}}` from `Trainer.evaluate()`'s held-out split, or `{}` each.

    Plus `"shade_eval"`/`"other_eval"` (the neighbour-exposure headline split,
    `trippy.constants.EVAL_EXPOSURE_MODES`) when present, and
    `"shade_calibrated"`/`"other_calibrated"` when the eval ran calibrated.

    `held_out_metrics` is `Trainer.evaluate()`'s own return value (see there for the "shade"/
    "other" shape: `{"n", "psnr", "ssim", "lpips"}`, and the parallel "_eval"-suffixed shape
    under the resolved `exposure_mode`). Degrades to an empty dict per side -- not
    `None`, so `report.json`'s `heldout_split` key is always present and always a dict -- when
    `held_out_metrics` predates this split (an older report re-serialized, or `fit()` returning
    `{}` because a `max_minutes` budget expired before the first eval). `"shade_eval"`/
    `"other_eval"` are omitted the same way (present only when the source dict has them) --
    `trippy.render.leaderboard` falls back to the plain "shade"/"other" fields and marks the
    cell "(own)" when they are absent, so an old report.json still renders.
    """
    split = {
        "shade": held_out_metrics.get("shade") or {},
        "other": held_out_metrics.get("other") or {},
    }
    # The neighbour-exposure headline split (PR #32, `Trainer.evaluate`'s "_eval" fields) --
    # always present on a current run (default `exposure_mode="neighbours"`), absent on a
    # report.json written before this feature existed.
    for key in ("shade_eval", "other_eval"):
        if held_out_metrics.get(key):
            split[key] = held_out_metrics[key]
    # Only present when the eval that produced `held_out_metrics` ran with test-time
    # photometric calibration (`Trainer.evaluate(calibrate=True)`); the leaderboard's
    # calibrated column keys off exactly this absence (trippy.render.leaderboard).
    for key in ("shade_calibrated", "other_calibrated"):
        if held_out_metrics.get(key):
            split[key] = held_out_metrics[key]
    return split


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


def _headline_heldout(held_out_metrics: dict) -> tuple[str, float | None, float | None, float | None, bool]:
    """The neighbour-exposure headline PSNR/SSIM/LPIPS, a short label for it, and whether it fell back.

    `Trainer.evaluate` always resolves an `exposure_mode` (default "neighbours",
    `trippy.constants.EVAL_EXPOSURE_MODES`) and reports the headline number under
    `"psnr_mean_eval"`/`"ssim_mean_eval"`/`"lpips_mean_eval"` alongside the strict, own-exposure
    `"psnr_mean"`/`"ssim_mean"`/`"lpips_mean"` fields (docs/EXPERIMENTS.md "interpolate_eval_settings
    ported"). A `held_out_metrics` dict from before this feature existed has only the strict
    fields -- this falls back to those and says so, rather than reporting nothing.

    Returns:
        `(label_suffix, psnr, ssim, lpips, used_fallback)` -- `label_suffix` is
        `" ({exposure_mode}-exposure)"` when the headline fields are present, else
        `" (own -- pre eval-fields run)"` when falling back to the strict fields.
    """
    psnr_eval = held_out_metrics.get("psnr_mean_eval")
    ssim_eval = held_out_metrics.get("ssim_mean_eval")
    lpips_eval = held_out_metrics.get("lpips_mean_eval")
    if psnr_eval is not None or ssim_eval is not None:
        mode = held_out_metrics.get("exposure_mode") or "neighbours"
        return f" ({mode}-exposure)", psnr_eval, ssim_eval, lpips_eval, False
    return (
        " (own -- pre eval-fields run)",
        held_out_metrics.get("psnr_mean"),
        held_out_metrics.get("ssim_mean"),
        held_out_metrics.get("lpips_mean"),
        True,
    )


def comparison_table_markdown(
    held_out_metrics: dict,
    candidate_audits: dict,
    baseline_audits: dict,
    dolly_metrics: dict,
) -> str:
    """Markdown table: held-out PSNR/SSIM/LPIPS, dark-mass fraction, extent p99/max, dolly coverage.

    The held-out rows lead with the neighbour-exposure headline number (`_headline_heldout`,
    docs/EXPERIMENTS.md "interpolate_eval_settings ported: eval_exposure_mode (default
    neighbours)") -- the number the leaderboard also sorts and headlines on -- and keep the
    strict, own-exposure numbers visible immediately below them as a secondary row so the two
    are never confused. A `held_out_metrics` dict that predates the "_eval" fields (an older
    report re-serialized) falls back to the strict numbers for the headline row too and says
    so, rather than showing a duplicate/misleading pair.

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
    label_suffix, psnr, ssim, lpips, used_fallback = _headline_heldout(held_out_metrics)
    strict_psnr = held_out_metrics.get("psnr_mean")
    strict_ssim = held_out_metrics.get("ssim_mean")
    strict_lpips = held_out_metrics.get("lpips_mean")

    candidate_dark = dark_mass_fraction(candidate_audits.get("shade_audit"))
    baseline_dark = dark_mass_fraction(baseline_audits.get("shade_audit"))

    candidate_extent = extent_p99_max(candidate_audits.get("extent_gate"))
    baseline_extent = extent_p99_max(baseline_audits.get("extent_gate"))

    dolly_coverage = dolly_mean_center_coverage(dolly_metrics)

    rows = [
        (f"Held-out PSNR (dB){label_suffix}", "n/a", _fmt(psnr)),
        (f"Held-out SSIM{label_suffix}", "n/a", _fmt(ssim, ".4f")),
        (f"Held-out LPIPS{label_suffix}", "n/a", _fmt(lpips, ".4f") if lpips is not None else "n/a"),
    ]
    # Secondary rows: the strict, own-exposure numbers, kept visible next to the headline --
    # skipped when the headline already IS the strict number (an old report with no "_eval"
    # fields), so that case shows one row per metric, not a redundant duplicate pair.
    if not used_fallback:
        rows += [
            ("Held-out PSNR (dB) (strict, own exposure)", "n/a", _fmt(strict_psnr)),
            ("Held-out SSIM (strict, own exposure)", "n/a", _fmt(strict_ssim, ".4f")),
            (
                "Held-out LPIPS (strict, own exposure)",
                "n/a",
                _fmt(strict_lpips, ".4f") if strict_lpips is not None else "n/a",
            ),
        ]
    rows += [
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
    "looks good"). The headline PSNR is the neighbour-exposure number
    (`_headline_heldout`) with the strict, own-exposure number quoted alongside it -- a
    `held_out_metrics` dict from before the "_eval" fields existed falls back to the strict
    number for both and says so.
    """
    label_suffix, psnr, _ssim, _lpips, used_fallback = _headline_heldout(held_out_metrics)
    strict_psnr = held_out_metrics.get("psnr_mean")
    candidate_dark = dark_mass_fraction(candidate_audits.get("shade_audit"))
    baseline_dark = dark_mass_fraction(baseline_audits.get("shade_audit"))
    psnr_clause = f"held-out PSNR {_fmt(psnr)} dB{label_suffix}"
    if not used_fallback:
        psnr_clause += f" (strict, own exposure: {_fmt(strict_psnr)} dB)"
    return (
        f"trippy train report {run_name}: epoch {epoch}, {psnr_clause}, "
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
        print(
            "trippy: TRIPPY_DELIVER_DRY_RUN=1 -- would run: "
            f"bash {_DELIVER_SCRIPT} {str(artifact)!r} {name!r} {why!r}"
        )
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


# --- viewer bundle + Mac launcher (Jordan: "I want to navigate freely") ---


def _main_checkout_root() -> Path:
    """Root of the MAIN git checkout, not a `.worktrees/<name>` sandbox this code may run in.

    Matches `scripts/open_mac_viewer.sh`'s own `REPO_ROOT` computation
    (`git rev-parse --git-common-dir`'s parent): the generated `.command`
    launcher and the binary-exists check here must agree on where the
    viewer binary lives, and that place must still exist after any
    worktree this code happened to run in is removed (AGENTS.md section 5).
    Falls back to this file's own checkout root if `git` is unavailable for
    any reason (never raises -- this is a best-effort default, not a gate).
    """
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        timeout=5,
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )
    git_common_dir = result.stdout.strip()
    if result.returncode != 0 or not git_common_dir:
        return Path(__file__).resolve().parents[2]
    return Path(git_common_dir).parent


def viewer_delivery_why(base_line: str) -> str:
    """The Mac viewer launcher's delivery "why": `base_line` plus this task's free-navigation note."""
    return f"{base_line}; {VIEWER_DELIVERY_WHY_SUFFIX}"


def build_mac_viewer_launcher(bundle_dir: Path, name: str) -> dict:
    """Generate `OPEN_TRIPS_MAC_<name>.command` for `bundle_dir` via `scripts/open_mac_viewer.sh`.

    This task's brief, requirement 2. Never raises: a viewer binary that has
    not been built yet, or any other failure of the script, is recorded in
    the returned dict instead of propagating -- neither a training run's
    `--report` nor a standalone `trippy bundle-launcher` call should die
    because a Rust binary is stale.

    Args:
        bundle_dir: a `trippy-bundle-1` directory (`trippy.render.bundle.
            export_bundle`'s output) -- must already exist and contain
            `bundle.json`.
        name: the launcher's name; the script writes
            `$TRIPPY_OUTPUT/deliver/<name>/OPEN_TRIPS_MAC_<name>.command`.

    Returns:
        `{"command_path": str|None, "status": "ok"|"failed", "note": str}`
        (`"note"` only present when `"status" == "failed"`).
    """
    binary_path = _main_checkout_root() / _TRIPS_VIEWER_BINARY_REL
    if not binary_path.is_file():
        return {
            "command_path": None,
            "status": "failed",
            "note": (
                f"viewer binary not found at {binary_path} -- build it with "
                "`cd rust && cargo build --release -p trips-viewer` "
                "(scripts/cpu_heavy.sh recommended) before a launcher can be generated"
            ),
        }
    settings = load_settings()
    result = subprocess.run(
        ["bash", str(_OPEN_MAC_VIEWER_SCRIPT), str(bundle_dir), name],
        env={**os.environ, "TRIPPY_OUTPUT": str(settings.trippy_output)},
        capture_output=True,
        text=True,
        timeout=DELIVER_SUBPROCESS_TIMEOUT_S,
        check=False,
    )
    if result.returncode != 0:
        return {
            "command_path": None,
            "status": "failed",
            "note": f"scripts/open_mac_viewer.sh exited {result.returncode}: {result.stderr.strip()}",
        }
    command_path = settings.trippy_output / "deliver" / name / f"OPEN_TRIPS_MAC_{name}.command"
    return {"command_path": str(command_path), "status": "ok"}


def default_bundle_out_dir(checkpoint: Path) -> Path:
    """Default bundle output directory for a checkpoint, for `trippy bundle-launcher --out`.

    A trippy-native checkpoint is usually `<run_dir>/checkpoints/checkpoint_*.pt`
    (`trippy.render.bundle.native_checkpoint_path`); this mirrors `train --report`'s
    own `<run_dir>/bundle` layout in that case, and falls back to a `bundle/`
    alongside the checkpoint itself for anything else (a TRIPS checkpoint
    directory, or a `.pt` not inside a `checkpoints/` dir).
    """
    if checkpoint.is_file() and checkpoint.parent.name == TRAIN_CHECKPOINT_DIRNAME:
        return checkpoint.parent.parent / TRAIN_REPORT_BUNDLE_DIRNAME
    if checkpoint.is_dir():
        return checkpoint / TRAIN_REPORT_BUNDLE_DIRNAME
    return checkpoint.parent / TRAIN_REPORT_BUNDLE_DIRNAME


def export_bundle_and_viewer_launcher(
    checkpoint: str | Path,
    out: str | Path,
    name: str,
    why_base: str,
    scene: str | Path | None = None,
    epoch: str | None = None,
) -> dict:
    """Steps 1-3 of this task's brief: bundle export, Mac viewer launcher, delivery.

    Shared by `run_train_report` (called against the run's own final
    checkpoint, with `out=<run_dir>/bundle`) and the standalone `trippy
    bundle-launcher` command (any existing checkpoint) -- one place builds
    and delivers the free-navigation launcher either way.

    `export_bundle` itself is allowed to raise (a checkpoint that fails to
    load is a real failure, same as every other step of `run_train_report`);
    the viewer-launcher and delivery steps never raise, per
    `build_mac_viewer_launcher`'s own contract.

    Args:
        checkpoint: forwarded to `trippy.render.bundle.export_bundle`.
        out: bundle directory to write.
        name: bundle label, launcher name, and delivery name all at once
            (`bundle.json`'s `name` field and the delivered artifact's name
            agree by construction).
        why_base: the delivery "why" line before `viewer_delivery_why`
            appends the free-navigation note -- `run_train_report`'s own
            `summary_line`, or a simpler description for a standalone
            `bundle-launcher` call.
        scene, epoch: forwarded to `export_bundle` (TRIPS checkpoints only).

    Returns:
        `{"bundle_dir": str, "bundle_json": dict, "viewer": dict, "delivery": dict}`.
    """
    bundle_dir, bundle_json = export_bundle(checkpoint, out, scene=scene, epoch=epoch, name=name)
    viewer = build_mac_viewer_launcher(bundle_dir, name)
    why = viewer_delivery_why(why_base)
    if viewer["command_path"] is not None:
        delivery = _deliver(Path(viewer["command_path"]), f"{name}-viewer", why)
    else:
        delivery = {
            "artifact": None,
            "name": f"{name}-viewer",
            "status": f"failed: {viewer['note']}",
        }
    return {"bundle_dir": str(bundle_dir), "bundle_json": bundle_json, "viewer": viewer, "delivery": delivery}


def _deliveries_markdown(deliveries: list[dict]) -> str:
    """"### Deliveries" section, one bullet per artifact, in the given order.

    This task's brief, requirement 4: the viewer launcher is listed FIRST
    (Jordan wants free navigation front and centre, not buried under the
    fixed-path dolly video) simply by being first in `deliveries`.
    """
    lines = ["### Deliveries", ""]
    for record in deliveries:
        artifact = record.get("artifact") or "n/a"
        lines.append(f"- {record['name']}: `{artifact}` ({record['status']})")
    return "\n".join(lines)


# --- orchestration ---


def _baseline_ply_audits(cfg, sparse_txt_dir: Path, frames: list[str] | None = None) -> dict:
    """`cached_baseline_audit` on `cfg.point_source`'s own PLY, or a recorded error for other types.

    Args:
        frames: forwarded to `cached_baseline_audit`/`run_shade_audit` -- the resolved
            `TrainConfig.shade_frames` (`resolve_shade_frames`), or `None` for the script's
            own default. Must be the SAME list the candidate export was audited with, or
            the baseline-vs-candidate comparison table would be measuring two different
            regions (this task's brief).
    """
    point_source = cfg.point_source
    if point_source.type != "gaussian" or not point_source.path:
        error = {
            "error": (
                "baseline audit needs point_source.type == 'gaussian' with a path; "
                f"got type={point_source.type!r} path={point_source.path!r}"
            )
        }
        return {"shade_audit": error, "extent_gate": error}
    return cached_baseline_audit(point_source.path, sparse_txt_dir, frames=frames)


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
        "held_out", "heldout_split": {"shade", "other"}, "dolly", "offpath",
        "audits": {"candidate", "baseline"}, "shade_frames":
        {"source", "count", "frames"} (`shade_frames_used_record` -- which
        frames the shade audit above actually used: `trainer.cfg.shade_frames`
        resolved, or the script's own default spelled out explicitly),
        "bundle": {"bundle_dir",
        "viewer"}, "summary_line", "deliveries"}` (`deliveries[0]` is always
        the Mac viewer launcher), plus `"gate"` on a blend-gate run
        (`gate_summary`) -- also written
        to `<run_dir>/report/report.json`, with the comparison table,
        summary line, and deliveries list (launcher first) appended to
        `<run_dir>/README.md`, and the bundle itself written to
        `<run_dir>/bundle/`.
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

    # Bug this task closes: this used to be a hardcoded `frames=None` on both calls below,
    # so every scene's shade dark-mass was measured on depthprior_shade_audit.py's own
    # default (SHADE_FRAMES_KK, kk-coherent's IMG_3828-3833) even on karekare-v2, whose
    # measured shade region is a different, 93-frame group (`shade_frames:` in
    # experiments/EXP-0011-karekare-v2/*.yaml -- README "Finding the shade frames").
    # `resolve_shade_frames` turns the config value into the actual list (or leaves it
    # `None`, unchanged old behaviour, when the config has no `shade_frames` key).
    shade_frames = resolve_shade_frames(trainer.cfg.shade_frames)

    # `resolve_sparse_txt_dir`: another bug this task closes -- `sparse_txt_dir` used to be
    # a hardcoded `<scene_root>/sparse_txt` literal, which does not exist for karekare-v2
    # (only binary `sparse/<n>` models). That FileNotFoundError previously escaped from
    # deep inside `trippy.eval.audits._run`; now the binary `sparse/0` model (the one
    # karekare-v2's point source/trainings were built from, 756 registered images) is
    # auto-converted once into `$TRIPPY_OUTPUT/scenes/<name>/sparse_txt` and reused.
    sparse_txt_dir = resolve_sparse_txt_dir(scene_root, trainer.cfg.sparse_txt)
    candidate_audits = audit_report([str(export_path)], sparse_txt_dir, frames=shade_frames)
    baseline_audits = _baseline_ply_audits(trainer.cfg, sparse_txt_dir, frames=shade_frames)

    run_name = run_dir.name
    line = summary_line(run_name, epoch, held_out_metrics, candidate_audits, baseline_audits)
    table = comparison_table_markdown(held_out_metrics, candidate_audits, baseline_audits, dolly_metrics)

    # Jordan: "fixed dolly paths are hard to judge, I want to navigate freely" -- export
    # a free-navigation bundle + Mac viewer launcher from this same final checkpoint,
    # alongside the existing dolly/honesty artifacts (this task's brief, requirements
    # 1-3). Never raises past this point regardless of whether the viewer binary has
    # been built (`build_mac_viewer_launcher`'s own contract); `export_bundle` failing
    # for some other reason is a real failure and propagates like any other step here.
    bundle_result = export_bundle_and_viewer_launcher(
        checkpoint_path,
        run_dir / TRAIN_REPORT_BUNDLE_DIRNAME,
        run_name,
        why_base=line,
    )
    if bundle_result["viewer"]["status"] != "ok":
        (out_dir / VIEWER_LAUNCHER_FAILED_FILENAME).write_text(bundle_result["viewer"]["note"] + "\n")

    dolly_mp4 = out_dir / CANDIDATE_REPORT_DOLLY_DIRNAME / CANDIDATE_NET_VIDEO_FILENAME
    honesty_sheet = out_dir / CANDIDATE_REPORT_DOLLY_DIRNAME / CANDIDATE_HONESTY_SHEET_FILENAME
    # Viewer launcher goes first (requirement 4: Jordan wants free navigation front and
    # centre); the dolly/honesty/export deliveries stay too -- they are cheap.
    deliveries = [
        bundle_result["delivery"],
        _deliver(dolly_mp4, f"{run_name}-dolly", line),
        _deliver(honesty_sheet, f"{run_name}-honesty", line),
        _deliver(export_path, f"{run_name}-export", line),
    ]

    gate = gate_summary(held_out_metrics, dolly_metrics)
    readme_path = _ensure_run_readme(run_dir)
    with open(readme_path, "a") as f:
        f.write(
            f"\n## Report: epoch {epoch}\n\n{line}\n\n{_deliveries_markdown(deliveries)}\n\n"
            f"{table}\n{gate_markdown(gate)}"
        )

    report = {
        "checkpoint": str(checkpoint_path),
        "device": device,
        "scene_root": str(scene_root),
        "export_ply": str(export_path),
        "epoch": epoch,
        "held_out": held_out_metrics,
        "heldout_split": heldout_split(held_out_metrics),
        "dolly": dolly_metrics,
        "offpath": offpath_metrics,
        "audits": {"candidate": candidate_audits, "baseline": baseline_audits},
        "sparse_txt_dir": str(sparse_txt_dir),
        "shade_frames": shade_frames_used_record(shade_frames),
        "bundle": {"bundle_dir": bundle_result["bundle_dir"], "viewer": bundle_result["viewer"]},
        "summary_line": line,
        "deliveries": deliveries,
    }
    if gate is not None:
        report["gate"] = gate
    (out_dir / CANDIDATE_REPORT_JSON_FILENAME).write_text(json.dumps(report, indent=2) + "\n")

    # Jordan always has one up-to-date "trips-leaderboard" sheet: rebuild it from every
    # run's report.json/metrics.jsonl (this one now included) and re-deliver under the
    # same fixed name (scripts/deliver.sh's ln -sfn replaces the symlink each time, per
    # docs/EXPERIMENTS.md "Leaderboard"). Deferred import (same reason as cli.py's own
    # deferred `run_train_report` import: this pulls in PIL/yaml, no need for `trippy
    # train` runs without --report to pay that cost) and never allowed to turn an
    # otherwise-successful report into a REPORT_FAILED.txt (see
    # `regenerate_and_deliver_safely`'s own docstring).
    from trippy.render.leaderboard import regenerate_and_deliver_safely

    report["leaderboard"] = regenerate_and_deliver_safely()
    return report
