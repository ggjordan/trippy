"""The baseline shade-audit cache must key on the frame list, not only the PLY.

Module: tests.test_audit_cache_frames
Related: trippy/eval/audits.py (cached_baseline_audit), docs/RESULTS.md (2026-09-09 frames bug).
"""
from __future__ import annotations

from trippy.eval import audits


def test_cache_entries_differ_per_frame_list(tmp_path, monkeypatch):
    calls: list[list[str] | None] = []

    def fake_audit_report(plys, sparse_txt_dir, frames=None):
        calls.append(frames)
        return {"frames": frames, "n": len(calls)}

    monkeypatch.setattr(audits, "audit_report", fake_audit_report)
    ply = tmp_path / "a.ply"
    ply.write_bytes(b"ply")
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    r6 = audits.cached_baseline_audit(ply, sparse, frames=["IMG_1.jpg"], cache_root=tmp_path / "c")
    r93 = audits.cached_baseline_audit(ply, sparse, frames=["IMG_2.jpg", "IMG_3.jpg"], cache_root=tmp_path / "c")
    again = audits.cached_baseline_audit(ply, sparse, frames=["IMG_1.jpg"], cache_root=tmp_path / "c")
    assert r6 != r93, "two frame lists shared one cache entry"
    assert again == r6 and len(calls) == 2, "same frame list should hit the cache"
