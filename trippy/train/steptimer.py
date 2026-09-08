"""Per-stage wall-clock timing for one training step, off by default.

Module: trippy.train.steptimer
Purpose: let `trippy profile-step` answer "where do the 0.35 seconds of a
    Karekare-v2 training step go?" without a second, drifting copy of
    `Trainer.train_step`. The real code path is instrumented with
    `with steptimer.stage("emit"):` markers; when no timer is active every
    marker is a single `is None` test and a shared no-op context manager, so
    a training run pays nothing measurable for them.
Invariants:
    - **Inactive is free and invisible.** `stage()` returns a module-level
      singleton whose `__enter__`/`__exit__` do nothing when no timer is
      installed. No allocation, no clock read, no device synchronisation.
    - **Stages are leaves and must not nest.** Every instrumented region is
      chosen so the regions are disjoint; a nested `stage()` inside an
      already-open one is charged to the inner name only and subtracted from
      the outer (see `_Region`), so an accidental nesting shows up as a
      smaller outer number rather than as double counting.
    - **A timed region ends device-synchronised.** On MPS the enclosed work
      is queued asynchronously, so the timer synchronises at both ends; that
      is what makes the numbers attributable, and also why the *sum* of the
      stages is slightly larger than an untimed step. `profile_step` reports
      both numbers for exactly that reason.
    - Timing is per *step*: `begin_step()` starts a fresh row, `end_step()`
      seals it. Rows are kept as plain dicts of seconds so the caller can
      take medians without importing anything from here.
Units: every recorded number is **seconds** of wall clock.
Layering note: `trippy.raster` imports this module, which normally would be
    backwards (raster is the lower layer). It is safe and deliberate: this
    module imports **nothing** from `trippy`, and `trippy/train/__init__.py`
    is a docstring with no imports, so there is no cycle. It lives under
    `train/` because a *training step* is the unit it exists to explain.
Related docs: docs/ARCHITECTURE.md ("Where a training step's time goes").
"""

from __future__ import annotations

import time
from typing import Any, Self

import torch

__all__ = ["StepTimer", "active", "note", "set_active", "stage"]


class _NullRegion:
    """The zero-cost stand-in returned by `stage()` when no timer is active."""

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return False


_NULL = _NullRegion()


class _Region:
    """One open timed region; charges its own elapsed time to `name` on exit."""

    __slots__ = ("child", "name", "start", "timer")

    def __init__(self, timer: StepTimer, name: str) -> None:
        self.timer = timer
        self.name = name
        self.start = 0.0
        # Time spent inside nested regions, subtracted so a stage is exclusive.
        self.child = 0.0

    def __enter__(self) -> Self:
        self.timer._sync()
        self.start = time.perf_counter()
        self.timer._stack.append(self)
        return self

    def __exit__(self, *exc: object) -> bool:
        self.timer._sync()
        elapsed = time.perf_counter() - self.start
        self.timer._stack.pop()
        exclusive = elapsed - self.child
        row = self.timer._row
        row[self.name] = row.get(self.name, 0.0) + exclusive
        if self.timer._stack:
            self.timer._stack[-1].child += elapsed
        return False


class StepTimer:
    """Accumulates per-stage seconds for a sequence of training steps.

    Attributes:
        device: the device to synchronise on (only `mps` actually syncs).
        rows: one dict per completed step, `{stage_name: seconds}` plus any
            scalars recorded with `note()`.
    """

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.rows: list[dict[str, Any]] = []
        self._row: dict[str, Any] = {}
        self._stack: list[_Region] = []

    # --- device ---

    def _sync(self) -> None:
        """Block until the device has finished everything queued so far."""
        if self.device.type == "mps":
            torch.mps.synchronize()

    def sync(self) -> None:
        """Public synchronise, for callers timing something by hand."""
        self._sync()

    # --- step lifecycle ---

    def begin_step(self) -> None:
        """Start a fresh row. Any half-open region from a failed step is dropped."""
        self._row = {}
        self._stack = []

    def row(self) -> dict[str, Any]:
        """The row being filled in right now (for callers that post-process it)."""
        return self._row

    def end_step(self) -> dict[str, Any]:
        """Seal the current row, append it to `rows`, and return it."""
        row = self._row
        self.rows.append(row)
        self._row = {}
        self._stack = []
        return row

    # --- recording ---

    def stage(self, name: str) -> _Region:
        """Context manager charging its body's exclusive wall time to `name`."""
        return _Region(self, name)

    def add(self, name: str, seconds: float) -> None:
        """Add `seconds` to `name` directly (for hook-derived measurements)."""
        self._row[name] = self._row.get(name, 0.0) + float(seconds)

    def note(self, name: str, value: Any) -> None:
        """Record a non-time scalar for this step (fragment counts, sizes...)."""
        self._row[name] = value

    def mark(self) -> float:
        """Synchronise and return a timestamp (for splitting the backward)."""
        self._sync()
        return time.perf_counter()


_ACTIVE: StepTimer | None = None


def set_active(timer: StepTimer | None) -> StepTimer | None:
    """Install (or clear) the process-wide timer; returns the previous one."""
    global _ACTIVE
    previous = _ACTIVE
    _ACTIVE = timer
    return previous


def active() -> StepTimer | None:
    """The installed timer, or None when profiling is off (the normal case)."""
    return _ACTIVE


def stage(name: str) -> Any:
    """`with steptimer.stage("emit"):` -- free when no timer is installed."""
    timer = _ACTIVE
    if timer is None:
        return _NULL
    return timer.stage(name)


def note(name: str, value: Any) -> None:
    """Record a diagnostic scalar on the active timer; no-op when inactive."""
    timer = _ACTIVE
    if timer is not None:
        timer.note(name, value)
