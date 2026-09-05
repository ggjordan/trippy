"""Tests for trippy.distill.compare: the 3-column audit comparison table.

Module: tests.test_distill_compare
Invariants under test: `build_audit_columns` leaves a column `None` when its
    PLY path is `None` (never calls `audit_report` for it); `audit_
    comparison_table` renders "pending" for a `None` column and "n/a" for a
    present-but-failed/missing metric, and reads the same numbers
    `trippy.render.report`'s pure helpers already extract (dark-mass
    fraction, extent p99/max) so the two tables never disagree by
    construction.
Fixture: hand-built `audit_report`-shaped dicts (this module never shells
    out to Splats' tools itself; `build_audit_columns`'s own use of
    `audit_report` is exercised end-to-end, network/subprocess-free, via a
    monkeypatch).
"""

from __future__ import annotations

from pathlib import Path

import trippy.distill.compare as compare_mod
from trippy.distill.compare import audit_comparison_table, build_audit_columns

_GOOD_AUDIT = {
    "shade_audit": {"results": [{"path": "x.ply", "n": 12345, "mass_in_region": 100.0, "dark_mass_lum0.25": 25.0}]},
    "extent_gate": {"plys": [{"radius_p99": 7.3, "radius_max": 7.7}]},
}
_FAILED_AUDIT = {"shade_audit": {"error": "boom"}, "extent_gate": {"error": "boom"}}


def test_build_audit_columns_none_for_missing_plys(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(compare_mod, "audit_report", lambda ply_paths, *a, **k: calls.append(ply_paths[0]) or _GOOD_AUDIT)

    columns = build_audit_columns(tmp_path, baseline_ply="baseline.ply", trips_export_ply=None, distilled_ply=None)

    assert list(columns.keys()) == ["baseline", "TRIPS export", "distilled"]
    assert columns["baseline"] == _GOOD_AUDIT
    assert columns["TRIPS export"] is None
    assert columns["distilled"] is None
    assert calls == ["baseline.ply"]  # audit_report only ever called for the given ply


def test_audit_comparison_table_pending_for_none_columns() -> None:
    table = audit_comparison_table({"baseline": _GOOD_AUDIT, "TRIPS export": None, "distilled": None})
    lines = table.splitlines()
    assert lines[0] == "| Metric | baseline | TRIPS export | distilled |"
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert cells[2] == "pending"
        assert cells[3] == "pending"


def test_audit_comparison_table_formats_good_audit_numbers() -> None:
    table = audit_comparison_table({"only": _GOOD_AUDIT})
    assert "12,345" in table
    assert "25.0%" in table  # dark_mass_lum0.25 / mass_in_region = 25/100
    assert "7.30" in table
    assert "7.70" in table


def test_audit_comparison_table_na_for_failed_audit() -> None:
    table = audit_comparison_table({"failed": _FAILED_AUDIT})
    lines = table.splitlines()
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert cells[1] == "n/a"


def test_audit_comparison_table_column_order_matches_input() -> None:
    table = audit_comparison_table({"z": None, "a": None})
    assert table.splitlines()[0] == "| Metric | z | a |"
