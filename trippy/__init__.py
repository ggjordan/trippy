"""trippy: TRIPS (Trilinear Point Splatting) ported to PyTorch/MPS for Apple Silicon.

Module: trippy (package root)
Invariants: __version__ resolves without raising even before the VERSION
    file or an installed distribution metadata exist.
Related docs: docs/SPEC.md "Scripts and hooks" item "Version truth";
    tests/test_version_sync.py.
"""

from __future__ import annotations

from importlib import metadata as _metadata
from pathlib import Path as _Path


def _read_version() -> str:
    # In a repo checkout the VERSION file is the single source of truth (ADR-0003); the
    # editable install's metadata only refreshes on `uv sync`, so it is the fallback.
    version_file = _Path(__file__).resolve().parent.parent / "VERSION"
    try:
        text = version_file.read_text().strip()
        if text:
            return text
    except OSError:
        pass
    try:
        return _metadata.version("trippy")
    except _metadata.PackageNotFoundError:
        pass
    return "0.0.0+unknown"


__version__ = _read_version()

__all__ = ["__version__"]
