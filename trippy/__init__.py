"""trippy: TRIPS (Trilinear Point Splatting) ported to PyTorch/MPS for Apple Silicon.

Module: trippy (package root)
Invariants: __version__ resolves without raising even before the VERSION
    file or an installed distribution metadata exist.
Related docs: /tmp/trippy-plan.md "Scripts and hooks" item "Version truth";
    tests/test_version_sync.py.
"""

from __future__ import annotations

from importlib import metadata as _metadata
from pathlib import Path as _Path


def _read_version() -> str:
    try:
        return _metadata.version("trippy")
    except _metadata.PackageNotFoundError:
        pass
    version_file = _Path(__file__).resolve().parent.parent / "VERSION"
    try:
        return version_file.read_text().strip()
    except OSError:
        pass
    return "0.0.0+unknown"


__version__ = _read_version()

__all__ = ["__version__"]
