"""Checkpoint retention policy: pure selection of which checkpoint_ep*.pt files to delete.

Module: trippy.train.retention
Invariants: every function here is pure with respect to the filesystem --
    `select_checkpoints_to_delete` only *reads* file metadata (mtime, for the
    `protect_newer_than_s` guard) and never deletes anything itself; callers
    (`trippy.train.trainer.Trainer.save_checkpoint`, `trippy prune-run`) do
    the actual `Path.unlink`. A path this module cannot parse an epoch out
    of (i.e. not `checkpoint_ep<NNNN>.pt`) is left alone rather than guessed
    at -- in particular `checkpoint_latest.pt` and `checkpoint_best.pt` must
    never be passed in by a caller, and if one slips through anyway it is
    silently kept, never deleted.
Related docs: docs/EXPERIMENTS.md "Training runs" (checkpoint retention
    policy); docs/LIMITATIONS.md (disk usage note).
"""

from __future__ import annotations

import re
import time
from pathlib import Path

_EPOCH_RE = re.compile(r"^checkpoint_ep(\d+)\.pt$")


def epoch_of(path: Path) -> int | None:
    """Parse the epoch number out of a `checkpoint_ep<NNNN>.pt` filename.

    Args:
        path: any path; only its `.name` is inspected.

    Returns:
        The epoch as an int, or None if `path.name` does not match
        `TRAIN_CHECKPOINT_FILENAME_FMT`'s pattern (e.g. `checkpoint_latest.pt`,
        `checkpoint_best.pt`, or an unrelated file).
    """
    m = _EPOCH_RE.match(path.name)
    return int(m.group(1)) if m else None


def select_checkpoints_to_delete(
    paths: list[Path],
    best_epoch: int | None,
    keep_every: int,
    keep_last: int,
    protect_newer_than_s: float = 0.0,
    *,
    now: float | None = None,
) -> list[Path]:
    """Return the subset of `paths` the retention policy says to delete.

    A path is KEPT (never appears in the return value) if any of the
    following hold:
      - its filename does not parse as `checkpoint_ep<NNNN>.pt` (unrecognised
        files are never touched, not even considered "extra");
      - its epoch equals `best_epoch` (the best held-out-PSNR checkpoint so
        far; None means "no best is known yet", so this rule never fires);
      - `keep_every > 0` and its epoch is a multiple of `keep_every`
        (`keep_every <= 0` disables this rule entirely);
      - its epoch is one of the `keep_last` largest epoch numbers present in
        `paths` (ties on epoch cannot occur -- `TRAIN_CHECKPOINT_FILENAME_FMT`
        is one file per epoch);
      - it was modified less than `protect_newer_than_s` seconds before `now`
        (defaults to `time.time()`) -- the safety net a concurrent
        still-running job needs (`trippy prune-run`'s default 2 minutes);
      - `path.stat()` fails (the file is already gone -- nothing to delete).

    Everything else is returned, to be deleted by the caller.

    Args:
        paths: candidate `checkpoint_ep<NNNN>.pt` files, any directory.
            Must NOT include `checkpoint_latest.pt` / `checkpoint_best.pt`
            (harmless if it does -- see module docstring -- but callers
            should filter those out themselves so this function's input is
            exactly "the epoch files under consideration").
        best_epoch: the epoch with the best held-out PSNR so far, or None.
        keep_every: keep every Nth epoch (e.g. 100 keeps ep0100, ep0200, ...).
        keep_last: keep this many of the most recent (highest-numbered)
            epoch files.
        protect_newer_than_s: never delete a file modified more recently
            than this many seconds ago.
        now: override for "the current time" (epoch seconds), for
            deterministic tests; defaults to `time.time()`.

    Returns:
        The list of `paths` entries to delete, in the order they appeared in
        `paths` (unordered with respect to epoch).
    """
    now = time.time() if now is None else now
    parsed = [(p, epoch_of(p)) for p in paths]

    recognised_epochs = sorted({e for _, e in parsed if e is not None}, reverse=True)
    keep_last_set = set(recognised_epochs[: max(0, keep_last)])

    to_delete: list[Path] = []
    for path, epoch in parsed:
        if epoch is None:
            continue  # unrecognised filename: never touch it
        if best_epoch is not None and epoch == best_epoch:
            continue
        if keep_every > 0 and epoch % keep_every == 0:
            continue
        if epoch in keep_last_set:
            continue
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue  # already gone -- nothing to delete
        if now - mtime < protect_newer_than_s:
            continue
        to_delete.append(path)
    return to_delete
