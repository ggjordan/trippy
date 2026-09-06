"""Tests for trippy.render.leaderboard: discovery, config matching, row building, sort, output.

Module: tests.test_render_leaderboard
Invariants under test:
    - `discover_run_dirs` finds a run directory only when it has BOTH
      `metrics.jsonl` and a report.json under either `report/report.json`
      (`train --report`) or `candidate/report.json` (`candidate-report`) --
      a run missing either (still training, or a bare metrics.jsonl with no
      report at all) is silently excluded, not an error.
    - `match_run_config` matches an `experiments/<experiment>/*.yaml`'s own
      `run_dir` field back to a scanned run by its last two path components,
      regardless of whether that field is relative-to-repo-root or an
      absolute, machine-specific path (EXP-0009's own configs are written
      this way -- see trippy.render.leaderboard's docstring).
    - `build_run_row` degrades every missing/malformed field to "n/a" (or
      `None` in the sort-key fields) rather than raising: a report.json with
      failed audits, a metrics.jsonl with a corrupt line, no matching
      experiments/ config, and a candidate-report layout with no `held_out`/
      `bundle`/`deliveries` keys at all are all exercised here.
    - `build_leaderboard_rows` sorts by shade dark-mass fraction ascending
      then held-out PSNR descending, rows with an unknown value on either
      axis sort to the end of it, and the two fixed baselines
      (`_baseline_rows`) are always present.
    - `rows_to_markdown`/`render_table_png` never raise on the sorted rows
      and produce non-trivial output (the real assertion that matters here
      -- pixel content of the PNG is not inspected, per AGENTS.md, but this
      table is synthetic text, not scene imagery, so reading its byte size
      is fine).
Fixtures: only synthetic JSON/YAML written directly to tmp_path -- no real
    scene, checkpoint, or PLY (AGENTS.md: test fixtures must be synthetic
    only).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trippy.render import leaderboard as lb

# --- fixtures: synthetic run trees ---


def _write_metrics_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _train_report_run(runs_root: Path, experiment: str, run_name: str, report: dict, metrics_rows: list[dict]) -> Path:
    run_dir = runs_root / experiment / run_name
    (run_dir / "report").mkdir(parents=True)
    (run_dir / "report" / "report.json").write_text(json.dumps(report))
    _write_metrics_jsonl(run_dir / "metrics.jsonl", metrics_rows)
    return run_dir


def _candidate_report_run(runs_root: Path, experiment: str, run_name: str, report: dict, metrics_rows: list[dict]) -> Path:
    run_dir = runs_root / experiment / run_name
    (run_dir / "candidate").mkdir(parents=True)
    (run_dir / "candidate" / "report.json").write_text(json.dumps(report))
    _write_metrics_jsonl(run_dir / "metrics.jsonl", metrics_rows)
    return run_dir


_REAL_SHAPED_TRAIN_REPORT = {
    "epoch": 39,
    "held_out": {"epoch": 39, "psnr_mean": 14.417, "ssim_mean": 0.390, "lpips_mean": 0.513},
    "dolly": {
        "frames": [{"coverage_mean_center": 0.46}, {"coverage_mean_center": 0.08}],
        "dolly_stop_index": 1,
    },
    "audits": {
        "candidate": {
            "shade_audit": {"results": [{"mass_in_region": 342813.4, "dark_mass_lum0.25": 124120.0}]},
            "extent_gate": {"plys": [{"radius_p99": 40.02, "radius_max": 124.48}]},
        },
        "baseline": {
            "shade_audit": {"results": [{"mass_in_region": 336873.5, "dark_mass_lum0.25": 67068.8}]},
            "extent_gate": {"plys": [{"radius_p99": 35.0, "radius_max": 100.0}]},
        },
    },
    "bundle": {"bundle_dir": "/x/bundle", "viewer": {"status": "ok", "command_path": "/x/OPEN_TRIPS_MAC_full1.command"}},
    "deliveries": [
        {"name": "full1-broadcast-viewer", "artifact": "/x/OPEN_TRIPS_MAC_full1.command", "status": "delivered"},
        {"name": "full1-broadcast-dolly", "artifact": "/x/dolly.mp4", "status": "delivered"},
    ],
}
_TRAIN_REPORT_METRICS = [
    {"step": 1, "epoch": 0},
    {"step": 3720, "epoch": 39},
    {"eval": True, "epoch": 39, "psnr_mean": 14.417, "ssim_mean": 0.390, "lpips_mean": 0.513},
]


# --- discover_run_dirs ---


def test_discover_run_dirs_finds_both_layouts(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _train_report_run(runs_root, "EXP-A", "run-1", _REAL_SHAPED_TRAIN_REPORT, _TRAIN_REPORT_METRICS)
    _candidate_report_run(runs_root, "EXP-A", "run-2", {"audits": {}}, [{"step": 1, "epoch": 0}])

    found = lb.discover_run_dirs(runs_root)
    names = {p.name for p in found}
    assert names == {"run-1", "run-2"}


def test_discover_run_dirs_excludes_runs_missing_either_file(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    # metrics.jsonl but no report.json anywhere (still training).
    _write_metrics_jsonl(runs_root / "EXP-A" / "still-training" / "metrics.jsonl", [{"step": 1, "epoch": 0}])
    # report.json but no metrics.jsonl (should not happen in practice, still must not crash).
    report_only = runs_root / "EXP-A" / "report-only"
    (report_only / "report").mkdir(parents=True)
    (report_only / "report" / "report.json").write_text("{}")

    assert lb.discover_run_dirs(runs_root) == []


def test_discover_run_dirs_missing_runs_root_returns_empty(tmp_path: Path) -> None:
    assert lb.discover_run_dirs(tmp_path / "does_not_exist") == []


# --- match_run_config ---


def test_match_run_config_relative_run_dir(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    exp_dir = experiments_root / "EXP-A"
    exp_dir.mkdir(parents=True)
    (exp_dir / "config.yaml").write_text(
        "run_dir: output/runs/EXP-A/run-1\nmode: broadcast\nepochs: 40\npoint_source:\n  type: gaussian\n"
    )
    config = lb.match_run_config(experiments_root, "EXP-A", "run-1")
    assert config is not None
    assert config["mode"] == "broadcast"
    assert config["epochs"] == 40
    assert config["point_source"]["type"] == "gaussian"


def test_match_run_config_absolute_machine_specific_run_dir(tmp_path: Path) -> None:
    # EXP-0009's own configs bake in an absolute, machine-specific run_dir (a git-worktree
    # job has no .venv of its own) -- matching must still work on just the last two parts.
    experiments_root = tmp_path / "experiments"
    exp_dir = experiments_root / "EXP-0009-hybrid-a"
    exp_dir.mkdir(parents=True)
    (exp_dir / "config_smoke.yaml").write_text(
        "run_dir: /Users/someone/trippy/output/runs/EXP-0009-hybrid-a/hybrid-a-smoke\nmode: trips\nepochs: 2\n"
    )
    config = lb.match_run_config(experiments_root, "EXP-0009-hybrid-a", "hybrid-a-smoke")
    assert config is not None
    assert config["mode"] == "trips"


def test_match_run_config_no_match_returns_none(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()
    assert lb.match_run_config(experiments_root, "EXP-A", "run-1") is None


def test_match_run_config_tolerates_malformed_yaml(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    exp_dir = experiments_root / "EXP-A"
    exp_dir.mkdir(parents=True)
    (exp_dir / "broken.yaml").write_text(":: not valid yaml :: [")
    assert lb.match_run_config(experiments_root, "EXP-A", "run-1") is None


# --- is_smoke_run ---


@pytest.mark.parametrize(
    "name,expected",
    [("hybrid-a-smoke", True), ("EXP-0003-kk-trips-train_smoke4", True), ("full1-broadcast", False), ("SMOKE-1", True)],
)
def test_is_smoke_run(name: str, expected: bool) -> None:
    assert lb.is_smoke_run(name) is expected


# --- build_run_row ---


def test_build_run_row_train_report_layout_matches_real_full1_broadcast_numbers(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    experiments_root = tmp_path / "experiments"
    exp_dir = experiments_root / "EXP-0003-kk-trips-train"
    exp_dir.mkdir(parents=True)
    (exp_dir / "config_broadcast.yaml").write_text(
        "run_dir: output/runs/EXP-0003-kk-trips-train/full1-broadcast\nmode: broadcast\nepochs: 40\n"
        "point_source:\n  type: gaussian\n"
    )
    run_dir = _train_report_run(
        runs_root, "EXP-0003-kk-trips-train", "full1-broadcast", _REAL_SHAPED_TRAIN_REPORT, _TRAIN_REPORT_METRICS
    )

    row = lb.build_run_row(run_dir, experiments_root)
    cells = row["cells"]
    assert cells[0] == "full1-broadcast"  # not flagged (smoke) -- real run
    assert cells[1] == "EXP-0003-kk-trips-train"
    assert cells[2] == "broadcast"
    assert cells[3] == "gaussian"
    assert cells[4] == "39/40"
    assert cells[5] == "3720"
    # `_REAL_SHAPED_TRAIN_REPORT`'s held_out has no "_eval" fields -- the headline column falls
    # back to the strict number and says so.
    assert cells[6] == "14.42/0.390/0.513 (own)"
    assert cells[8] == "14.42/n/a"  # strict own-exposure secondary column: all/shade, no shade here
    assert cells[9] == "36.2%"  # dark-mass fraction: 124120.0 / 342813.4
    assert cells[10] == "40.0/124.5"
    assert cells[13] == "OPEN_TRIPS_MAC_full1.command"  # viewer launcher, first deliveries entry
    assert row["dark_mass"] == pytest.approx(124120.0 / 342813.4)
    assert row["psnr_all"] == pytest.approx(14.417)
    assert row["is_baseline"] is False


def test_build_run_row_headlines_the_neighbour_exposure_eval_fields_when_present(tmp_path: Path) -> None:
    report = dict(_REAL_SHAPED_TRAIN_REPORT)
    report["held_out"] = {
        **_REAL_SHAPED_TRAIN_REPORT["held_out"],
        "exposure_mode": "neighbours",
        "psnr_mean_eval": 17.95,
        "ssim_mean_eval": 0.44,
        "lpips_mean_eval": 0.40,
    }
    report["heldout_split"] = {
        "shade": {"n": 6, "psnr": 8.49, "ssim": 0.30, "lpips": 0.69},
        "other": {"n": 27, "psnr": 16.47, "ssim": 0.45, "lpips": 0.42},
        "shade_eval": {"n": 6, "psnr": 15.10, "ssim": 0.41, "lpips": 0.35},
        "other_eval": {"n": 27, "psnr": 18.20, "ssim": 0.48, "lpips": 0.30},
    }
    metrics_rows = _TRAIN_REPORT_METRICS + [
        {
            "eval": True,
            "epoch": 39,
            "psnr_mean": 14.417,
            "ssim_mean": 0.390,
            "lpips_mean": 0.513,
            "psnr_mean_eval": 17.95,
            "ssim_mean_eval": 0.44,
            "lpips_mean_eval": 0.40,
            "shade": {"n": 6, "psnr": 8.49},
            "shade_eval": {"n": 6, "psnr": 15.10},
        }
    ]
    run_dir = _train_report_run(tmp_path / "runs", "EXP-A", "run-eval", report, metrics_rows)

    row = lb.build_run_row(run_dir, tmp_path / "no_experiments")
    cells = row["cells"]
    assert cells[6] == "17.95/0.440/0.400"  # neighbour-exposure headline, no "(own)" fallback tag
    assert cells[7] == "15.10/0.410/0.350"
    assert cells[8] == "14.42/8.49"  # strict own-exposure secondary column stays visible
    assert row["psnr_all"] == pytest.approx(17.95)  # sorts on the headline number, not the strict one


def test_build_run_row_held_out_shade_reads_report_heldout_split_first(tmp_path: Path) -> None:
    report = dict(_REAL_SHAPED_TRAIN_REPORT)
    report["heldout_split"] = {
        "shade": {"n": 6, "psnr": 12.5, "ssim": 0.30, "lpips": 0.60},
        "other": {"n": 27, "psnr": 15.0, "ssim": 0.45, "lpips": 0.40},
    }
    # A stale eval row shows a different shade number -- report.json must win.
    stale_metrics = _TRAIN_REPORT_METRICS + [
        {"eval": True, "epoch": 39, "psnr_mean": 14.417, "shade": {"n": 6, "psnr": 1.0, "ssim": 0.1, "lpips": 0.1}}
    ]
    run_dir = _train_report_run(tmp_path / "runs", "EXP-A", "run-shade", report, stale_metrics)

    row = lb.build_run_row(run_dir, tmp_path / "no_experiments")
    # No "shade_eval" in report.json -- falls back to the strict "shade" split, marked " (own)".
    assert row["cells"][7] == "12.50/0.300/0.600 (own)"


def test_build_run_row_held_out_shade_falls_back_to_last_eval_row(tmp_path: Path) -> None:
    # No heldout_split in report.json (an older report predating this feature) -- fall back to
    # the last metrics.jsonl eval row's own "shade" key (written by a `trippy eval --checkpoint`
    # re-run against the existing checkpoint).
    metrics_rows = _TRAIN_REPORT_METRICS + [
        {
            "eval": True,
            "epoch": 39,
            "psnr_mean": 14.417,
            "shade": {"n": 6, "psnr": 13.1, "ssim": 0.33, "lpips": 0.55},
            "other": {"n": 27, "psnr": 14.9, "ssim": 0.40, "lpips": 0.45},
        }
    ]
    run_dir = _train_report_run(
        tmp_path / "runs", "EXP-A", "run-shade-fallback", _REAL_SHAPED_TRAIN_REPORT, metrics_rows
    )

    row = lb.build_run_row(run_dir, tmp_path / "no_experiments")
    assert row["cells"][7] == "13.10/0.330/0.550 (own)"


def test_build_run_row_held_out_shade_is_na_when_neither_source_has_it(tmp_path: Path) -> None:
    run_dir = _train_report_run(
        tmp_path / "runs", "EXP-A", "run-no-shade", _REAL_SHAPED_TRAIN_REPORT, _TRAIN_REPORT_METRICS
    )
    row = lb.build_run_row(run_dir, tmp_path / "no_experiments")
    assert row["cells"][7] == "n/a"


def test_build_run_row_candidate_report_layout_no_baseline_or_bundle(tmp_path: Path) -> None:
    # candidate-report's own report.json has no `held_out`/`epoch`/`bundle`/`deliveries` keys,
    # and its `audits` dict is flat (shade_audit/extent_gate directly, not nested under
    # "candidate") -- both real shapes must be handled (see EXP-0003's real
    # candidate/report.json, which predates the train --report schema).
    runs_root = tmp_path / "runs"
    report = {
        "checkpoint": "/x/checkpoint_latest.pt",
        "audits": {
            "shade_audit": {"results": [{"mass_in_region": 100.0, "dark_mass_lum0.25": 20.0}]},
            "extent_gate": {"plys": [{"radius_p99": 5.0, "radius_max": 9.0}]},
        },
        "dolly": {"frames": [{"coverage_mean_center": 0.5}]},
    }
    metrics_rows = [
        {"step": 1, "epoch": 0},
        {"step": 48, "epoch": 1},
        {"eval": True, "epoch": 1, "psnr_mean": 8.88, "ssim_mean": 0.162, "lpips_mean": 0.860},
    ]
    run_dir = _candidate_report_run(runs_root, "EXP-A", "run-2", report, metrics_rows)

    row = lb.build_run_row(run_dir, tmp_path / "no_such_experiments_dir")
    cells = row["cells"]
    assert cells[0] == "run-2"
    assert cells[1] == "EXP-A"
    assert cells[2] == "n/a"  # no matching experiments/ config
    assert cells[3] == "n/a"
    assert cells[4] == "1"  # no `epochs:` in config, no `epoch` key in report -> last eval epoch only
    assert cells[5] == "48"
    assert cells[6] == "8.88/0.162/0.860 (own)"
    assert cells[8] == "8.88/n/a"  # strict own-exposure secondary column: no shade split here
    assert cells[9] == "20.0%"
    assert cells[13] == "n/a"  # candidate-report never exports a viewer bundle


def test_build_run_row_tolerates_missing_and_malformed_data(tmp_path: Path) -> None:
    # Failed audits (real shape when the ml-sharp venv/Splats scripts aren't present),
    # an empty dolly, and a corrupt trailing metrics.jsonl line -- nothing here may raise.
    runs_root = tmp_path / "runs"
    report = {
        "audits": {
            "candidate": {"shade_audit": {"error": "no ml-sharp venv"}, "extent_gate": {"error": "no ml-sharp venv"}},
            "baseline": {"shade_audit": {"error": "no ml-sharp venv"}, "extent_gate": {"error": "no ml-sharp venv"}},
        },
        "dolly": {},
    }
    run_dir = runs_root / "EXP-A" / "run-3"
    (run_dir / "report").mkdir(parents=True)
    (run_dir / "report" / "report.json").write_text(json.dumps(report))
    (run_dir / "metrics.jsonl").write_text('{"step": 1, "epoch": 0}\nnot valid json at all\n')

    row = lb.build_run_row(run_dir, tmp_path / "no_experiments")
    cells = row["cells"]
    assert cells[4] == "n/a"  # no eval row, no report["epoch"] -> no epoch at all
    assert cells[5] == "1"  # max step still found despite the corrupt trailing line
    assert cells[6] == "n/a"  # no eval row -> no held-out numbers
    assert cells[8] == "n/a"  # strict own-exposure secondary column
    assert cells[9] == "n/a"  # dark-mass
    assert cells[10] == "n/a"  # extent
    assert row["dark_mass"] is None
    assert row["psnr_all"] is None


def test_build_run_row_tolerates_unreadable_report_json(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "EXP-A" / "run-4"
    (run_dir / "report").mkdir(parents=True)
    (run_dir / "report" / "report.json").write_text("{not valid json")
    (run_dir / "metrics.jsonl").write_text('{"step": 1, "epoch": 0}\n')

    row = lb.build_run_row(run_dir, tmp_path / "no_experiments")
    assert row["cells"][0] == "run-4"
    assert row["dark_mass"] is None


# --- build_leaderboard_rows: sort + baselines ---


def test_build_leaderboard_rows_sorts_by_dark_mass_then_psnr_descending(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    experiments_root = tmp_path / "no_experiments"

    def _report(dark_mass_pct: float | None, psnr: float | None) -> dict:
        audits = (
            {}
            if dark_mass_pct is None
            else {"shade_audit": {"results": [{"mass_in_region": 100.0, "dark_mass_lum0.25": dark_mass_pct}]}}
        )
        rows = [{"step": 1, "epoch": 0}]
        if psnr is not None:
            rows.append({"eval": True, "epoch": 0, "psnr_mean": psnr, "ssim_mean": 0.1, "lpips_mean": 0.1})
        return {"audits": audits, "dolly": {}}, rows

    for name, dark_pct, psnr in [("low-dark", 10.0, 12.0), ("high-dark-hi-psnr", 90.0, 30.0), ("high-dark-lo-psnr", 90.0, 5.0)]:
        report, metrics_rows = _report(dark_pct, psnr)
        _candidate_report_run(runs_root, "EXP-A", name, report, metrics_rows)

    rows = lb.build_leaderboard_rows(runs_root, experiments_root)
    scanned_names = [r["cells"][0] for r in rows if not r["is_baseline"]]
    # low-dark (10%) first; between the two 90%-dark rows, higher PSNR (30) beats lower (5).
    assert scanned_names == ["low-dark", "high-dark-hi-psnr", "high-dark-lo-psnr"]


def test_build_leaderboard_rows_always_includes_fixed_baselines(tmp_path: Path) -> None:
    rows = lb.build_leaderboard_rows(tmp_path / "empty_runs", tmp_path / "empty_experiments")
    names = {r["cells"][0] for r in rows}
    assert any("Gaussians kkc_15000" in n for n in names)
    assert any("Design C" in n for n in names)
    assert len(rows) == 2  # no scanned runs, just the two baselines


# --- markdown + PNG rendering ---


def test_rows_to_markdown_has_header_row_and_footnotes() -> None:
    rows = lb.build_leaderboard_rows(Path("/nonexistent"), Path("/nonexistent"))
    md = lb.rows_to_markdown(rows)
    assert "| " + " | ".join(lb._HEADERS) + " |" in md
    assert "Gaussians kkc_15000" in md
    assert "19.9%" in md
    assert "Design C" in md
    for note in lb._FOOTNOTES:
        assert note in md


def test_render_table_png_writes_a_real_png(tmp_path: Path) -> None:
    rows = lb.build_leaderboard_rows(Path("/nonexistent"), Path("/nonexistent"))
    out_path = tmp_path / "out" / "leaderboard.png"
    result = lb.render_table_png(lb._HEADERS, rows, out_path, title="test leaderboard")
    assert result == out_path
    assert out_path.exists()
    assert out_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --- write_leaderboard / regenerate_and_deliver ---


def test_write_leaderboard_writes_both_files(tmp_path: Path) -> None:
    result = lb.write_leaderboard(tmp_path / "out", tmp_path / "empty_runs", tmp_path / "empty_experiments")
    assert result["markdown_path"].exists()
    assert result["png_path"].exists()
    assert len(result["rows"]) == 2  # just the fixed baselines


def test_regenerate_and_deliver_dry_run_never_shells_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIPPY_DELIVER_DRY_RUN", "1")

    def _boom(*args, **kwargs):  # pragma: no cover -- must never be called
        raise AssertionError("subprocess.run must not be called under TRIPPY_DELIVER_DRY_RUN=1")

    import trippy.render.report as report_mod

    monkeypatch.setattr(report_mod.subprocess, "run", _boom)

    result = lb.regenerate_and_deliver(tmp_path / "out", tmp_path / "empty_runs", tmp_path / "empty_experiments")
    assert result["delivery"]["status"] == "skipped: TRIPPY_DELIVER_DRY_RUN=1"
    assert result["markdown_path"].exists()
    assert result["png_path"].exists()


def test_regenerate_and_deliver_safely_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**kwargs):
        raise RuntimeError("leaderboard exploded")

    monkeypatch.setattr(lb, "regenerate_and_deliver", _boom)
    result = lb.regenerate_and_deliver_safely()
    assert result["status"] == "failed"
    assert "leaderboard exploded" in result["error"]


# --- optional "Held-out shade PSNR (calibrated)" column (trippy eval --calibrate) ---


def test_leaderboard_headers_omit_the_calibrated_column_when_no_run_has_one() -> None:
    rows = lb.build_leaderboard_rows(Path("/nonexistent"), Path("/nonexistent"))
    headers = lb.leaderboard_headers(rows)
    assert headers == lb._HEADERS
    assert lb._CALIBRATED_SHADE_HEADER not in headers
    for row in rows:
        assert len(lb.row_cells(row, headers)) == len(headers)


def test_build_run_row_reads_calibrated_shade_from_report_json(tmp_path: Path) -> None:
    report = dict(_REAL_SHAPED_TRAIN_REPORT)
    report["heldout_split"] = {
        "shade": {"n": 6, "psnr": 8.49, "ssim": 0.30, "lpips": 0.69},
        "other": {"n": 27, "psnr": 16.47, "ssim": 0.45, "lpips": 0.42},
        "shade_calibrated": {"n": 6, "psnr": 12.34, "ssim": 0.33, "lpips": 0.65},
    }
    run_dir = _train_report_run(tmp_path / "runs", "EXP-A", "run-cal", report, _TRAIN_REPORT_METRICS)

    row = lb.build_run_row(run_dir, tmp_path / "no_experiments")
    assert row["shade_calibrated"] == 12.34
    headers = lb.leaderboard_headers([row])
    assert headers[-1] == lb._CALIBRATED_SHADE_HEADER
    assert lb.row_cells(row, headers)[-1] == "12.34"


def test_build_run_row_falls_back_to_the_last_eval_row_for_the_calibrated_shade(tmp_path: Path) -> None:
    metrics_rows = _TRAIN_REPORT_METRICS + [
        {
            "eval": True,
            "epoch": 299,
            "psnr_mean": 15.02,
            "shade": {"n": 6, "psnr": 8.49},
            "shade_calibrated": {"n": 6, "psnr": 11.11},
        }
    ]
    run_dir = _train_report_run(
        tmp_path / "runs", "EXP-A", "run-cal-fallback", _REAL_SHAPED_TRAIN_REPORT, metrics_rows
    )
    assert lb.build_run_row(run_dir, tmp_path / "no_experiments")["shade_calibrated"] == 11.11


def test_build_run_row_calibrated_shade_is_none_without_a_calibrated_eval(tmp_path: Path) -> None:
    run_dir = _train_report_run(
        tmp_path / "runs", "EXP-A", "run-no-cal", _REAL_SHAPED_TRAIN_REPORT, _TRAIN_REPORT_METRICS
    )
    assert lb.build_run_row(run_dir, tmp_path / "no_experiments")["shade_calibrated"] is None


def test_markdown_and_png_gain_exactly_one_column_when_a_calibrated_run_exists(tmp_path: Path) -> None:
    report = dict(_REAL_SHAPED_TRAIN_REPORT)
    report["heldout_split"] = {
        "shade": {"n": 6, "psnr": 8.49},
        "other": {"n": 27, "psnr": 16.47},
        "shade_calibrated": {"n": 6, "psnr": 12.34},
    }
    _train_report_run(tmp_path / "runs", "EXP-A", "run-cal", report, _TRAIN_REPORT_METRICS)

    result = lb.write_leaderboard(tmp_path / "out", tmp_path / "runs", tmp_path / "no_experiments")
    markdown = result["markdown_path"].read_text()
    header_line = next(line for line in markdown.splitlines() if line.startswith("| Run |"))
    assert header_line.count("|") == len(lb._HEADERS) + 2
    assert lb._CALIBRATED_SHADE_HEADER in header_line
    # The Gaussian baseline has no exposure model, so its calibrated cell stays "n/a".
    baseline_line = next(line for line in markdown.splitlines() if "Gaussians kkc_15000" in line)
    assert baseline_line.rstrip().endswith("n/a |")
    assert result["png_path"].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
