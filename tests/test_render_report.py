"""Tests for trippy.render.report: comparison table, summary line, dolly stop rule, delivery.

Module: tests.test_render_report
Invariants under test:
    - `trippy.render.dolly.dolly_stop_index` finds the last index whose
      coverage is still >= threshold on a synthetic (non-rendered)
      coverage profile, including monotonic-decreasing, all-above,
      all-below, and non-monotonic profiles, per this task's brief
      requirement 4.
    - `dark_mass_fraction`, `extent_p99_max`, `dolly_mean_center_coverage`
      degrade to `None` on missing/`{"error": ...}` audit data rather than
      raising, and `comparison_table_markdown` still renders a full table
      (with "n/a" cells) when every audit failed -- requirement 6's "the
      table must still render" on a synthetic scene.
    - `summary_line` contains the epoch, held-out PSNR, and dark-mass
      fraction vs baseline, and never the word "good" (AGENTS.md/this
      task's honesty rule -- no verdict language).
    - `_deliver` never shells out to scripts/deliver.sh when
      `TRIPPY_DELIVER_DRY_RUN=1` is set, and reports a clean "skipped"
      status either way for a missing artifact.
Fixtures: only synthetic dicts (no real scene, checkpoint, or PLY --
    AGENTS.md test fixtures must be synthetic only).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trippy.constants import DOLLY_COVERAGE_STOP_THRESHOLD
from trippy.render import report as report_mod
from trippy.render.dolly import dolly_stop_index

# --- dolly_stop_index ---


def test_dolly_stop_index_monotonic_decreasing_profile() -> None:
    # Mirrors the shape of the real EXP-0003 full1-broadcast dolly path:
    # 0.46 -> ... -> 0.08 -> ... -> 0.0001 (docs/EXPERIMENTS.md dolly notes).
    coverage = [0.46, 0.40, 0.32, 0.24, 0.16, 0.10, 0.065, 0.044, 0.02, 0.0007, 0.0001]
    stop = dolly_stop_index(coverage, threshold=0.05)
    assert stop == 6  # last index with coverage >= 0.05 (0.065)
    assert coverage[stop] >= 0.05
    assert all(c < 0.05 for c in coverage[stop + 1 :])


def test_dolly_stop_index_default_threshold_matches_constant() -> None:
    coverage = [0.5, 0.2, 0.1, 0.04, 0.01]
    assert dolly_stop_index(coverage) == dolly_stop_index(coverage, DOLLY_COVERAGE_STOP_THRESHOLD)


def test_dolly_stop_index_all_frames_above_threshold_keeps_everything() -> None:
    coverage = [0.9, 0.8, 0.7, 0.6]
    assert dolly_stop_index(coverage, threshold=0.05) == len(coverage) - 1


def test_dolly_stop_index_all_frames_below_threshold_keeps_first_only() -> None:
    coverage = [0.03, 0.02, 0.01]
    assert dolly_stop_index(coverage, threshold=0.05) == 0


def test_dolly_stop_index_empty_profile_returns_zero() -> None:
    assert dolly_stop_index([], threshold=0.05) == 0


def test_dolly_stop_index_non_monotonic_profile_finds_true_last_qualifying_frame() -> None:
    # Coverage dips below threshold then recovers -- the rule is "last index
    # above threshold", not "first index below", so it must walk past the dip.
    coverage = [0.5, 0.5, 0.02, 0.5, 0.01, 0.01]
    assert dolly_stop_index(coverage, threshold=0.05) == 3


# --- pure numeric extraction helpers ---


def test_dolly_mean_center_coverage_respects_stop_index() -> None:
    dolly_metrics = {
        "frames": [
            {"coverage_mean_center": 0.5},
            {"coverage_mean_center": 0.3},
            {"coverage_mean_center": 0.01},  # cut by the stop rule
        ],
        "dolly_stop_index": 1,
    }
    assert report_mod.dolly_mean_center_coverage(dolly_metrics) == pytest.approx(0.4)


def test_dolly_mean_center_coverage_without_stop_index_averages_all_frames() -> None:
    dolly_metrics = {"frames": [{"coverage_mean_center": 0.2}, {"coverage_mean_center": 0.4}]}
    assert report_mod.dolly_mean_center_coverage(dolly_metrics) == pytest.approx(0.3)


def test_dolly_mean_center_coverage_no_frames_returns_none() -> None:
    assert report_mod.dolly_mean_center_coverage({"frames": []}) is None
    assert report_mod.dolly_mean_center_coverage({}) is None


def test_dark_mass_fraction_computes_ratio() -> None:
    shade_audit = {"results": [{"mass_in_region": 342813.4, "dark_mass_lum0.25": 124120.0}]}
    assert report_mod.dark_mass_fraction(shade_audit) == pytest.approx(124120.0 / 342813.4)


def test_dark_mass_fraction_none_on_error_or_missing() -> None:
    assert report_mod.dark_mass_fraction({"error": "no ml-sharp venv"}) is None
    assert report_mod.dark_mass_fraction({"results": []}) is None
    assert report_mod.dark_mass_fraction({}) is None
    assert report_mod.dark_mass_fraction(None) is None


def test_extent_p99_max_extracts_pair() -> None:
    extent_gate = {"plys": [{"radius_p99": 40.02, "radius_max": 124.48}]}
    assert report_mod.extent_p99_max(extent_gate) == (40.02, 124.48)


def test_extent_p99_max_none_on_error_or_missing() -> None:
    assert report_mod.extent_p99_max({"error": "boom"}) is None
    assert report_mod.extent_p99_max({"plys": []}) is None


# --- comparison table ---


_REAL_CANDIDATE_AUDITS = {
    "shade_audit": {"results": [{"mass_in_region": 342813.4, "dark_mass_lum0.25": 124120.0}]},
    "extent_gate": {"plys": [{"radius_p99": 40.02, "radius_max": 124.48}]},
}
_REAL_BASELINE_AUDITS = {
    "shade_audit": {"results": [{"mass_in_region": 336873.5, "dark_mass_lum0.25": 67068.8}]},
    "extent_gate": {"plys": [{"radius_p99": 35.0, "radius_max": 100.0}]},
}
_REAL_HELD_OUT = {"epoch": 39, "psnr_mean": 14.417, "ssim_mean": 0.39, "lpips_mean": 0.513}
_REAL_DOLLY = {"frames": [{"coverage_mean_center": 0.46}, {"coverage_mean_center": 0.08}], "dolly_stop_index": 1}


def test_comparison_table_renders_real_numbers() -> None:
    table = report_mod.comparison_table_markdown(_REAL_HELD_OUT, _REAL_CANDIDATE_AUDITS, _REAL_BASELINE_AUDITS, _REAL_DOLLY)
    assert "| Metric | Baseline | Candidate |" in table
    assert "14.42" in table  # held-out PSNR
    assert "36.2%" in table  # candidate dark-mass fraction
    assert "19.9%" in table  # baseline dark-mass fraction
    assert "40.02" in table and "124.48" in table  # candidate extent
    # Baseline has no held-out concept (it's an un-trained source PLY, not a
    # model) -- those three cells are legitimately "n/a", not fabricated.
    assert "| Held-out PSNR (dB) | n/a | 14.42 |" in table


def test_comparison_table_still_renders_when_every_audit_failed() -> None:
    # Requirement 6: on the synthetic CPU test scene, audits legitimately
    # return {"error": ...} -- the table must still render, with "n/a" cells,
    # not raise.
    failed = {"shade_audit": {"error": "no ml-sharp venv"}, "extent_gate": {"error": "no ml-sharp venv"}}
    table = report_mod.comparison_table_markdown({}, failed, failed, {"frames": []})
    assert "| Metric | Baseline | Candidate |" in table
    assert "n/a" in table
    assert "FAILED" in table


# --- summary line ---


def test_summary_line_contains_epoch_psnr_and_dark_mass_vs_baseline() -> None:
    line = report_mod.summary_line("full1-broadcast", 39, _REAL_HELD_OUT, _REAL_CANDIDATE_AUDITS, _REAL_BASELINE_AUDITS)
    assert "epoch 39" in line
    assert "14.42" in line
    assert "36.2%" in line
    assert "19.9%" in line
    assert "good" not in line.lower()  # AGENTS.md honesty rule: no verdict language


def test_summary_line_degrades_gracefully_with_no_audits() -> None:
    failed = {"shade_audit": {"error": "boom"}, "extent_gate": {"error": "boom"}}
    line = report_mod.summary_line("run", 0, {}, failed, failed)
    assert "epoch 0" in line
    assert "n/a" in line
    assert "good" not in line.lower()


# --- delivery ---


def test_deliver_dry_run_env_never_shells_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIPPY_DELIVER_DRY_RUN", "1")
    artifact = tmp_path / "dolly.mp4"
    artifact.write_bytes(b"not a real video")

    def _boom(*args, **kwargs):  # pragma: no cover -- must never be called
        raise AssertionError("subprocess.run must not be called under TRIPPY_DELIVER_DRY_RUN=1")

    monkeypatch.setattr(report_mod.subprocess, "run", _boom)
    record = report_mod._deliver(artifact, "test-dolly", "why")
    assert record["status"] == "skipped: TRIPPY_DELIVER_DRY_RUN=1"


def test_deliver_missing_artifact_is_skipped_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRIPPY_DELIVER_DRY_RUN", raising=False)
    record = report_mod._deliver(tmp_path / "does_not_exist.mp4", "test-dolly", "why")
    assert record["status"] == "skipped: artifact not found"
