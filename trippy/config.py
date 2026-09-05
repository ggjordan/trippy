"""Runtime settings and device selection for trippy.

Module: trippy.config
Invariants: load_settings() never raises (missing env vars fall back to
    documented defaults); pick_device() never silently falls back from mps
    to cpu -- requesting "mps" asserts torch.backends.mps.is_available()
    (AGENTS.md: "no silent MPS fallback").
Related docs: docs/SPEC.md "Repository layout" (SPLATS_ROOT,
    TRIPPY_OUTPUT), "D8" (own venv, reads Splats read-only).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from trippy.constants import DEFAULT_DEVICE, DEFAULT_SPLATS_ROOT


def _repo_root() -> Path:
    """Root of the trippy checkout, i.e. the parent of the trippy/ package dir."""
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Resolved runtime locations.

    Attributes:
        splats_root: Path to Jordan's Splats project (read-only: scenes,
            GPU queue, depth tools). Env override: SPLATS_ROOT.
        trippy_output: Path for trippy's own job logs/runs/exports. Env
            override: TRIPPY_OUTPUT. Gitignored; never scenes/plys.
    """

    splats_root: Path
    trippy_output: Path


def load_settings() -> Settings:
    """Build Settings from the environment, falling back to documented defaults."""
    splats_root = Path(os.environ.get("SPLATS_ROOT", DEFAULT_SPLATS_ROOT)).expanduser()
    default_output = _repo_root() / "output"
    trippy_output = Path(os.environ.get("TRIPPY_OUTPUT", str(default_output))).expanduser()
    return Settings(splats_root=splats_root, trippy_output=trippy_output)


def pick_device(requested: str | None = None) -> torch.device:
    """Resolve the torch device trippy should run on.

    Priority: explicit `requested` argument > TRIPS_DEVICE env var >
    DEFAULT_DEVICE ("cpu"). Only "cpu" and "mps" are supported.

    Requesting "mps" asserts torch.backends.mps.is_available() so a
    misconfigured machine fails loudly instead of silently running on CPU.

    Args:
        requested: "cpu", "mps", or None to defer to TRIPS_DEVICE/default.

    Returns:
        A torch.device for "cpu" or "mps".
    """
    choice = (requested or os.environ.get("TRIPS_DEVICE") or DEFAULT_DEVICE).lower()
    if choice not in ("cpu", "mps"):
        raise ValueError(f"unsupported device {choice!r}; expected 'cpu' or 'mps'")
    if choice == "mps":
        assert torch.backends.mps.is_available(), "MPS requested but not available on this machine"
    return torch.device(choice)
