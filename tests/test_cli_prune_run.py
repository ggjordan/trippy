"""End-to-end subprocess test for `trippy prune-run`.

Module: tests.test_cli_prune_run
Invariants under test: `trippy prune-run <run_dir>` applies the same policy
`Trainer.save_checkpoint` applies internally (`trippy.train.retention.
select_checkpoints_to_delete`) to an existing run directory's
`checkpoints/`; `--dry-run` prints what would be deleted and the bytes that
would be freed without deleting anything; a real (non-dry-run) run deletes
exactly the files the policy does not protect; `checkpoint_latest.pt` is
never touched (it never matches the `checkpoint_ep<NNNN>.pt` glob this
command operates on); the single newest epoch file is never deleted even
with `--keep-last 0`; `checkpoints/best.json`'s `epoch`, if present,
is protected the same as any other best epoch; and a file modified within
`--protect-seconds` is skipped regardless of the other rules.

No torch/Trainer involved: checkpoint files here are plain byte blobs (the
command only stats/unlinks paths by name, never loads their contents), so
this suite runs in milliseconds and needs no synthetic scene.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _write_checkpoints(
    run_dir: Path,
    epochs: list[int],
    size: int = 1000,
    latest_epoch: int | None = None,
    best_epoch: int | None = None,
    best_psnr: float = 10.0,
    old_mtime: float | None = None,
) -> Path:
    """Write a synthetic `<run_dir>/checkpoints/` directory: fixed-size byte blobs, not real .pt files."""
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in epochs:
        path = checkpoint_dir / f"checkpoint_ep{epoch:04d}.pt"
        path.write_bytes(b"x" * size)
        if old_mtime is not None:
            os.utime(path, (old_mtime, old_mtime))
    if latest_epoch is not None:
        (checkpoint_dir / "checkpoint_latest.pt").write_bytes(b"x" * size)
    if best_epoch is not None:
        (checkpoint_dir / "best.json").write_text(json.dumps({"epoch": best_epoch, "psnr": best_psnr}))
    return checkpoint_dir


def _run_prune(run_dir: Path, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "trippy.cli", "prune-run", str(run_dir), *extra_args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _remaining_epochs(checkpoint_dir: Path) -> set[int]:
    return {int(p.stem.split("checkpoint_ep")[1]) for p in checkpoint_dir.glob("checkpoint_ep*.pt")}


def _json_payload(stdout: str) -> dict:
    line = next(line for line in stdout.splitlines() if line.startswith("JSON:"))
    return json.loads(line[len("JSON:") :])


def test_dry_run_lists_deletions_without_touching_disk(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    epochs = list(range(0, 300, 10))  # 30 files: 0, 10, ..., 290
    checkpoint_dir = _write_checkpoints(run_dir, epochs, latest_epoch=290, old_mtime=0.0)

    result = _run_prune(run_dir, "--dry-run", "--keep-every", "100", "--keep-last", "1", "--protect-seconds", "0")
    assert result.returncode == 0, result.stderr
    assert "would delete" in result.stdout
    assert "would free" in result.stdout

    # Nothing was actually deleted.
    assert _remaining_epochs(checkpoint_dir) == set(epochs)
    assert (checkpoint_dir / "checkpoint_latest.pt").exists()

    payload = _json_payload(result.stdout)
    assert payload["dry_run"] is True
    assert payload["bytes_freed"] > 0
    assert set(payload["deleted"]) == {
        str(checkpoint_dir / f"checkpoint_ep{e:04d}.pt") for e in epochs if e not in (0, 100, 200, 290)
    }


def test_real_run_deletes_exactly_the_unprotected_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    epochs = list(range(0, 300, 10))
    checkpoint_dir = _write_checkpoints(run_dir, epochs, latest_epoch=290, old_mtime=0.0)

    result = _run_prune(run_dir, "--keep-every", "100", "--keep-last", "1", "--protect-seconds", "0")
    assert result.returncode == 0, result.stderr

    assert _remaining_epochs(checkpoint_dir) == {0, 100, 200, 290}
    assert (checkpoint_dir / "checkpoint_latest.pt").exists()  # never in the candidate glob


def test_newest_epoch_file_is_never_deleted_even_with_keep_last_zero(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    epochs = list(range(8))  # 0..7
    checkpoint_dir = _write_checkpoints(run_dir, epochs, latest_epoch=7, old_mtime=0.0)

    result = _run_prune(run_dir, "--keep-every", "1000", "--keep-last", "0", "--protect-seconds", "0")
    assert result.returncode == 0, result.stderr

    remaining = _remaining_epochs(checkpoint_dir)
    assert 7 in remaining  # the newest file survives the hard safety net regardless of --keep-last
    assert 0 in remaining  # epoch 0 is always a keep_every multiple (0 % N == 0)
    assert remaining == {0, 7}


def test_best_epoch_from_best_json_is_protected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    epochs = list(range(8))
    checkpoint_dir = _write_checkpoints(run_dir, epochs, latest_epoch=7, best_epoch=3, old_mtime=0.0)

    result = _run_prune(run_dir, "--keep-every", "1000", "--keep-last", "0", "--protect-seconds", "0")
    assert result.returncode == 0, result.stderr

    remaining = _remaining_epochs(checkpoint_dir)
    assert remaining == {0, 3, 7}  # 0 (keep_every), 3 (best.json), 7 (newest, hard safety net)


def test_recently_modified_files_are_protected_by_default(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    old_time = 0.0
    for epoch in (0, 1, 2, 3):
        path = checkpoint_dir / f"checkpoint_ep{epoch:04d}.pt"
        path.write_bytes(b"x" * 1000)
        os.utime(path, (old_time, old_time))
    # epoch 4 is "just written" -- current mtime, well within the default 120s protect window.
    (checkpoint_dir / "checkpoint_ep0004.pt").write_bytes(b"x" * 1000)
    (checkpoint_dir / "checkpoint_latest.pt").write_bytes(b"x" * 1000)

    result = _run_prune(run_dir, "--keep-every", "1000", "--keep-last", "0")  # default --protect-seconds (120s)
    assert result.returncode == 0, result.stderr

    remaining = _remaining_epochs(checkpoint_dir)
    # 0 kept (keep_every), 4 kept (newest AND recently modified), 1/2/3 deleted.
    assert remaining == {0, 4}


def test_missing_checkpoints_dir_exits_nonzero(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    result = _run_prune(run_dir)
    assert result.returncode == 2
    assert "checkpoints" in result.stderr
