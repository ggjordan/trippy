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
    assert cells[6] == "14.42/0.390/0.513"
    assert cells[8] == "36.2%"  # dark-mass fraction: 124120.0 / 342813.4
    assert cells[9] == "40.0/124.5"
    assert cells[12] == "OPEN_TRIPS_MAC_full1.command"  # viewer launcher, first deliveries entry
    assert row["dark_mass"] == pytest.approx(124120.0 / 342813.4)
    assert row["psnr_all"] == pytest.approx(14.417)
    assert row["is_baseline"] is False


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
    assert cells[6] == "8.88/0.162/0.860"
    assert cells[8] == "20.0%"
    assert cells[12] == "n/a"  # candidate-report never exports a viewer bundle


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
    assert cells[8] == "n/a"
    assert cells[9] == "n/a"
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
