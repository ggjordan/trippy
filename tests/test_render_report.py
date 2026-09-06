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
      `TRIPPY_DELIVER_DRY_RUN=1` is set, reports a clean "skipped" status
      either way for a missing artifact, and prints the deliver.sh command
      it would have run under the dry-run env var (so a dry-run
      `bundle-launcher` invocation has something to show for itself).
    - `build_mac_viewer_launcher` (Jordan: "I want to navigate freely")
      never raises: a missing viewer binary is caught before
      `scripts/open_mac_viewer.sh` even runs, and a script failure (e.g. a
      missing bundle directory) is reported the same way -- both return
      `{"status": "failed", "command_path": None, "note": ...}` rather than
      propagating. `viewer_delivery_why` appends the free-navigation note to
      an existing summary line. `default_bundle_out_dir` resolves a
      checkpoint's default bundle directory for `trippy bundle-launcher`.
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


def test_heldout_split_extracts_shade_and_other() -> None:
    held_out = {
        "epoch": 39,
        "psnr_mean": 14.417,
        "shade": {"n": 6, "psnr": 12.5, "ssim": 0.3, "lpips": 0.6},
        "other": {"n": 27, "psnr": 15.0, "ssim": 0.45, "lpips": 0.4},
    }
    split = report_mod.heldout_split(held_out)
    assert split == {
        "shade": {"n": 6, "psnr": 12.5, "ssim": 0.3, "lpips": 0.6},
        "other": {"n": 27, "psnr": 15.0, "ssim": 0.45, "lpips": 0.4},
    }


def test_heldout_split_degrades_to_empty_dicts_on_missing_data() -> None:
    assert report_mod.heldout_split({}) == {"shade": {}, "other": {}}
    assert report_mod.heldout_split({"epoch": 0, "psnr_mean": 0.0}) == {"shade": {}, "other": {}}


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


def test_deliver_dry_run_prints_the_would_run_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Acceptance: a dry-run bundle-launcher invocation must print the deliver.sh
    # command it would have run, not just silently record a skip.
    monkeypatch.setenv("TRIPPY_DELIVER_DRY_RUN", "1")
    artifact = tmp_path / "OPEN_TRIPS_MAC_test.command"
    artifact.write_text("#!/bin/bash\n")
    report_mod._deliver(artifact, "test-viewer", "some why")
    out = capsys.readouterr().out
    assert "would run" in out
    assert "deliver.sh" in out
    assert str(artifact) in out
    assert "some why" in out


# --- viewer bundle + Mac launcher ---


def test_viewer_delivery_why_appends_free_navigation_suffix() -> None:
    line = report_mod.viewer_delivery_why("trippy train report run: epoch 3, held-out PSNR n/a dB")
    assert line.startswith("trippy train report run: epoch 3, held-out PSNR n/a dB; ")
    assert "free-navigation viewer" in line
    assert "N/P" in line


def test_build_mac_viewer_launcher_missing_binary_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No rust/target/release/trips-viewer under this fake checkout root.
    monkeypatch.setattr(report_mod, "_main_checkout_root", lambda: tmp_path)

    def _boom(*args, **kwargs):  # pragma: no cover -- must never be called
        raise AssertionError("open_mac_viewer.sh must not run when the binary check fails first")

    monkeypatch.setattr(report_mod.subprocess, "run", _boom)

    result = report_mod.build_mac_viewer_launcher(tmp_path / "bundle", "test-run")
    assert result["command_path"] is None
    assert result["status"] == "failed"
    assert "trips-viewer" in result["note"]
    assert "cargo build" in result["note"]


def test_build_mac_viewer_launcher_script_failure_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "rust" / "target" / "release" / "trips-viewer"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setattr(report_mod, "_main_checkout_root", lambda: tmp_path)

    # open_mac_viewer.sh checks the bundle directory before the binary -- a bundle
    # directory that doesn't exist makes it fail deterministically, regardless of
    # whether the real repo's own trips-viewer binary happens to be built.
    missing_bundle_dir = tmp_path / "no_such_bundle"
    result = report_mod.build_mac_viewer_launcher(missing_bundle_dir, "test-run")
    assert result["command_path"] is None
    assert result["status"] == "failed"
    assert "open_mac_viewer.sh" in result["note"]


def test_default_bundle_out_dir_for_checkpoint_inside_checkpoints_dirname(tmp_path: Path) -> None:
    run_dir = tmp_path / "EXP-0006-union"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    checkpoint = checkpoints_dir / "checkpoint_latest.pt"
    checkpoint.write_bytes(b"")
    assert report_mod.default_bundle_out_dir(checkpoint) == run_dir / "bundle"


def test_default_bundle_out_dir_falls_back_to_alongside_the_checkpoint(tmp_path: Path) -> None:
    # A bare .pt not inside a checkpoints/ dir (e.g. handed a file directly).
    loose_checkpoint = tmp_path / "some_checkpoint.pt"
    loose_checkpoint.write_bytes(b"")
    assert report_mod.default_bundle_out_dir(loose_checkpoint) == tmp_path / "bundle"
