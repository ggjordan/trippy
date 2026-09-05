"""Checkpoint save/load: a plain dict of state_dicts + the resolved TrainConfig.

Module: trippy.train.checkpoint_io
Invariants: a checkpoint is one `torch.save`d dict with fixed top-level
    keys (`epoch`, `cfg`, `point_params`, `pose_params`, `net`, `camera`,
    `background`, `optimizer`) -- no custom pickle classes, so a checkpoint
    loads with `torch.load(..., weights_only=False)` on any machine with
    trippy imported (state_dicts are plain tensors, `cfg` is a plain dict).
    Writes go to a temporary file and are renamed into place, so a job
    killed mid-write (queue timeout, OOM) never leaves a half-written
    checkpoint at the final path (docs/EXPERIMENTS.md "Training runs":
    "so queue jobs end cleanly").
Related docs: docs/TRIPS_REFERENCE.md Sec. 9 (TRIPS's own per-epoch
    checkpoint layout -- trippy's is a single-file simplification, not a
    port of that directory structure); docs/EXPERIMENTS.md "Training runs".
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically write `payload` (a plain dict of state_dicts/scalars) to `path`.

    Args:
        path: destination `.pt` path; parent directories are created if
            missing.
        payload: dict to `torch.save`. By convention (see module docstring)
            this holds `epoch`, `cfg`, `point_params`, `pose_params`, `net`,
            `camera`, `background`, and `optimizer` keys, but this function
            itself does not inspect the dict's contents.

    Returns:
        The final path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint previously written by `save_checkpoint`.

    Args:
        path: `.pt` path.
        map_location: forwarded to `torch.load` (default "cpu", so a
            checkpoint written on `mps` still loads on a CPU-only machine).

    Returns:
        The payload dict, as passed to `save_checkpoint`.
    """
    return torch.load(Path(path), map_location=map_location, weights_only=False)
