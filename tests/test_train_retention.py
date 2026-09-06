"""Tests for trippy.train.retention: the pure checkpoint-retention selection function.

Module: tests.test_train_retention
Invariants under test: `select_checkpoints_to_delete` never returns a path
    whose epoch equals `best_epoch`, whose epoch is a `keep_every`
    multiple, whose epoch is among the `keep_last` largest present, or that
    was modified within `protect_newer_than_s` seconds of `now`; and it
    never returns a path it cannot parse an epoch out of (e.g.
    `checkpoint_latest.pt`/`checkpoint_best.pt`, even if a caller
    mistakenly includes them). Every candidate not covered by one of those
    rules IS returned. All fixtures are plain empty files under `tmp_path`
    -- no torch/Trainer involved, this module is pure path/mtime logic.
"""

from __future__ import annotations

import os
from pathlib import Path

from trippy.train.retention import epoch_of, select_checkpoints_to_delete


def _touch(path: Path, mtime: float | None = None) -> Path:
    path.write_bytes(b"x")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _epoch_paths(tmp_path: Path, epochs: list[int], mtime: float | None = None) -> list[Path]:
    return [_touch(tmp_path / f"checkpoint_ep{e:04d}.pt", mtime=mtime) for e in epochs]


def test_epoch_of_parses_the_standard_filename() -> None:
    assert epoch_of(Path("checkpoint_ep0042.pt")) == 42
    assert epoch_of(Path("/some/dir/checkpoint_ep0000.pt")) == 0


def test_epoch_of_rejects_non_epoch_filenames() -> None:
    assert epoch_of(Path("checkpoint_latest.pt")) is None
    assert epoch_of(Path("checkpoint_best.pt")) is None
    assert epoch_of(Path("best.json")) is None
    assert epoch_of(Path("checkpoint_ep0012.pt.bak")) is None  # not an exact match -- unrecognised


def test_best_epoch_is_never_deleted(tmp_path: Path) -> None:
    paths = _epoch_paths(tmp_path, [0, 1, 2, 3, 4], mtime=0.0)
    to_delete = select_checkpoints_to_delete(
        paths, best_epoch=2, keep_every=1000, keep_last=0, protect_newer_than_s=0.0, now=1e9
    )
    kept = {epoch_of(p) for p in paths} - {epoch_of(p) for p in to_delete}
    assert 2 in kept
    assert all(epoch_of(p) != 2 for p in to_delete)


def test_unrecognised_filenames_are_never_returned(tmp_path: Path) -> None:
    latest = _touch(tmp_path / "checkpoint_latest.pt", mtime=0.0)
    best = _touch(tmp_path / "checkpoint_best.pt", mtime=0.0)
    epoch_files = _epoch_paths(tmp_path, [0, 1, 2], mtime=0.0)
    to_delete = select_checkpoints_to_delete(
        [latest, best, *epoch_files], best_epoch=None, keep_every=1000, keep_last=0,
        protect_newer_than_s=0.0, now=1e9,
    )  # fmt: skip
    assert latest not in to_delete
    assert best not in to_delete


def test_keep_every_multiples_are_kept(tmp_path: Path) -> None:
    epochs = list(range(0, 250, 10))  # 0, 10, 20, ..., 240
    paths = _epoch_paths(tmp_path, epochs, mtime=0.0)
    to_delete = select_checkpoints_to_delete(
        paths, best_epoch=None, keep_every=100, keep_last=0, protect_newer_than_s=0.0, now=1e9
    )
    deleted_epochs = {epoch_of(p) for p in to_delete}
    kept_epochs = set(epochs) - deleted_epochs
    assert {0, 100, 200} <= kept_epochs
    assert deleted_epochs == set(epochs) - {0, 100, 200}


def test_epoch_zero_is_always_a_keep_every_multiple(tmp_path: Path) -> None:
    """0 % N == 0 for any N -- the very first checkpoint is always kept unless keep_every<=0."""
    paths = _epoch_paths(tmp_path, [0, 7], mtime=0.0)
    to_delete = select_checkpoints_to_delete(
        paths, best_epoch=None, keep_every=1000, keep_last=0, protect_newer_than_s=0.0, now=1e9
    )
    assert {epoch_of(p) for p in to_delete} == {7}


def test_keep_every_disabled_by_non_positive_value(tmp_path: Path) -> None:
    paths = _epoch_paths(tmp_path, [0, 100, 200], mtime=0.0)
    to_delete = select_checkpoints_to_delete(
        paths, best_epoch=None, keep_every=0, keep_last=0, protect_newer_than_s=0.0, now=1e9
    )
    assert {epoch_of(p) for p in to_delete} == {0, 100, 200}


def test_keep_last_keeps_the_n_most_recent_epochs(tmp_path: Path) -> None:
    # keep_every=0 (disabled) isolates keep_last from the "epoch 0 is always a multiple" rule.
    epochs = [0, 10, 20, 30, 40]
    paths = _epoch_paths(tmp_path, epochs, mtime=0.0)
    to_delete = select_checkpoints_to_delete(
        paths, best_epoch=None, keep_every=0, keep_last=2, protect_newer_than_s=0.0, now=1e9
    )
    kept_epochs = set(epochs) - {epoch_of(p) for p in to_delete}
    assert kept_epochs == {30, 40}


def test_keep_last_zero_keeps_nothing_extra(tmp_path: Path) -> None:
    epochs = [10, 20, 30]  # avoids epoch 0's "always a keep_every multiple" special case
    paths = _epoch_paths(tmp_path, epochs, mtime=0.0)
    to_delete = select_checkpoints_to_delete(
        paths, best_epoch=None, keep_every=1000, keep_last=0, protect_newer_than_s=0.0, now=1e9
    )
    assert {epoch_of(p) for p in to_delete} == set(epochs)


def test_recently_modified_files_are_protected(tmp_path: Path) -> None:
    now = 1_000_000.0
    old = _touch(tmp_path / "checkpoint_ep0005.pt", mtime=now - 500.0)
    recent = _touch(tmp_path / "checkpoint_ep0010.pt", mtime=now - 10.0)
    to_delete = select_checkpoints_to_delete(
        [old, recent], best_epoch=None, keep_every=1000, keep_last=0, protect_newer_than_s=120.0, now=now
    )
    assert old in to_delete
    assert recent not in to_delete


def test_combined_policy_deletes_only_what_no_rule_protects(tmp_path: Path) -> None:
    """A realistic mix: best + keep_every + keep_last + a freshly-written file all protected."""
    now = 1_000_000.0
    epochs = [0, 50, 100, 150, 190, 200]
    paths = {e: _touch(tmp_path / f"checkpoint_ep{e:04d}.pt", mtime=now - 10_000.0) for e in epochs}
    # ep0190 was "just written" (simulating a race with a still-running job).
    os.utime(paths[190], (now - 1.0, now - 1.0))

    to_delete = select_checkpoints_to_delete(
        list(paths.values()), best_epoch=50, keep_every=100, keep_last=1, protect_newer_than_s=120.0, now=now
    )
    deleted_epochs = {epoch_of(p) for p in to_delete}
    # kept: 0 and 100 (keep_every multiples), 50 (best), 200 (keep_last=1 -> newest), 190 (protected by mtime)
    assert deleted_epochs == {150}


def test_dry_run_selection_never_touches_the_filesystem(tmp_path: Path) -> None:
    """The function only ever reads mtimes -- calling it twice must be idempotent."""
    paths = _epoch_paths(tmp_path, [0, 1, 2, 3], mtime=0.0)
    first = select_checkpoints_to_delete(paths, best_epoch=None, keep_every=2, keep_last=1, now=1e9)
    for p in paths:
        assert p.exists()  # nothing was deleted by the selection call itself
    second = select_checkpoints_to_delete(paths, best_epoch=None, keep_every=2, keep_last=1, now=1e9)
    assert {epoch_of(p) for p in first} == {epoch_of(p) for p in second}
