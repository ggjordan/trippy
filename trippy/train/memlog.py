"""Process + MPS memory instrumentation for the long, staged post-training wrap-up.

Module: trippy.train.memlog
Purpose: answer "which wrap-up stage was holding the memory when the kernel
    killed us?" with numbers instead of guesses. Two `trippy train --report`
    runs on the full karekare-v2 scene (kkv2-8-full-masked-cont, epoch 259,
    2026-09-11; kkv2-5c-hybrid-cont, epoch 244, 2026-09-13) finished their
    training budget, wrote `checkpoint_latest` and `export.ply`, and were then
    killed (`Killed: 9`, i.e. a macOS jetsam memory kill -- Splats' queue
    runner has no timeout, so nothing else kills a job) about an hour into the
    wrap-up. Shorter runs on the same scene finished theirs. This module makes
    every stage boundary record RSS, peak RSS and the two MPS allocator
    counters, so `docs/ARCHITECTURE.md`'s wrap-up memory table is measured
    rather than reasoned about.
Invariants:
    - **Never raises.** Every sample is best-effort; a missing `torch.mps`, a
      `ps` that is not on PATH, or an unwritable log path degrades to `None`
      values / a dropped line. A diagnostic must never be able to fail the
      job it is diagnosing.
    - No new dependency. RSS comes from `resource.getrusage` (peak) and a
      one-shot `ps -o rss=` (current); `psutil` is deliberately not used.
    - `release()` is the only thing here with a side effect on memory
      (`gc.collect()` + `torch.mps.empty_cache()`); it is called at every
      stage boundary by `MemoryLog.stage`, so a stage's tensors are actually
      returned before the next stage's "before" sample is taken.
    - All sizes in this module are **bytes**.
Related docs: docs/ARCHITECTURE.md "Wrap-up memory"; AGENTS.md Sec. 6
    ("GPU and compute" -- this only observes, it never submits work).
"""

from __future__ import annotations

import gc
import json
import os
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# macOS `getrusage` reports ru_maxrss in BYTES; Linux reports KILObytes. Detected once
# rather than hardcoded so the numbers stay right if this is ever run on a Linux box.
_RU_MAXRSS_TO_BYTES = 1 if sys.platform == "darwin" else 1024

# `ps -o rss=` reports kilobytes on both macOS and Linux.
_PS_RSS_TO_BYTES = 1024

# A `ps` call that hangs would stall the very stage it is measuring.
_PS_TIMEOUT_S = 5.0

BYTES_PER_GIB = 1024**3


def peak_rss_bytes() -> int | None:
    """Highest RSS this process has ever reached, in bytes (`getrusage`), or None."""
    try:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * _RU_MAXRSS_TO_BYTES
    except Exception:  # noqa: BLE001 -- a diagnostic never raises (module invariant)
        return None


def rss_bytes() -> int | None:
    """Current resident set size of this process, in bytes, or None if `ps` fails.

    `getrusage` only exposes the *peak*, which is exactly the wrong number for
    "did the previous stage actually give its memory back?" -- hence the `ps`
    shell-out. It runs a handful of times per wrap-up (stage boundaries only),
    so the subprocess cost is irrelevant.
    """
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=_PS_TIMEOUT_S,
            check=False,
        )
        text = out.stdout.strip()
        if not text:
            return None
        return int(text.split()[0]) * _PS_RSS_TO_BYTES
    except Exception:  # noqa: BLE001 -- a diagnostic never raises (module invariant)
        return None


def mps_memory() -> dict[str, int | None]:
    """`{"mps_allocated", "mps_driver"}` in bytes; both None when MPS is unavailable.

    `current_allocated_memory` is what torch's tensors hold right now;
    `driver_allocated_memory` is what the MPS allocator has taken from the
    system and may be *caching* rather than using. The gap between the two is
    the thing `release()` tries to close, and the thing a jetsam kill counts
    against the process.
    """
    empty: dict[str, int | None] = {"mps_allocated": None, "mps_driver": None}
    try:
        import torch

        if not (hasattr(torch, "mps") and torch.backends.mps.is_available()):
            return empty
        allocated = getattr(torch.mps, "current_allocated_memory", None)
        driver = getattr(torch.mps, "driver_allocated_memory", None)
        return {
            "mps_allocated": int(allocated()) if allocated is not None else None,
            "mps_driver": int(driver()) if driver is not None else None,
        }
    except Exception:  # noqa: BLE001 -- a diagnostic never raises (module invariant)
        return empty


def release() -> None:
    """Give memory back between stages: `gc.collect()` then `torch.mps.empty_cache()`.

    Order matters: `empty_cache` can only return blocks no live tensor
    references, so the collection has to happen first. Both are cheap next to
    a wrap-up stage that takes minutes.
    """
    try:
        gc.collect()
    except Exception:  # noqa: BLE001, S110 -- a diagnostic never raises (module invariant)
        pass
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001, S110 -- a diagnostic never raises (module invariant)
        pass


def sample(label: str) -> dict[str, Any]:
    """One memory reading: `{"label", "t", "rss", "peak_rss", "mps_allocated", "mps_driver"}`."""
    return {
        "label": label,
        "t": time.time(),
        "rss": rss_bytes(),
        "peak_rss": peak_rss_bytes(),
        **mps_memory(),
    }


def _gib(value: int | None) -> str:
    return "n/a" if value is None else f"{value / BYTES_PER_GIB:.2f}G"


def format_sample(record: dict[str, Any]) -> str:
    """One-line human form, used for the run log: `mem <label>: rss=.. peak=.. mps=../..`."""
    return (
        f"mem {record['label']}: rss={_gib(record.get('rss'))} "
        f"peak={_gib(record.get('peak_rss'))} "
        f"mps_alloc={_gib(record.get('mps_allocated'))} "
        f"mps_driver={_gib(record.get('mps_driver'))}"
    )


class MemoryLog:
    """Append-only JSONL memory log with a `stage()` context manager.

    Args:
        path: JSONL file to append to; None disables the file (samples are
            still returned and still go to `log`).
        log: optional one-line sink (e.g. `Trainer._log` or `print`). None
            means no text output.

    Every write is best-effort (module invariant): an unwritable path costs a
    dropped line, never an exception.
    """

    def __init__(self, path: str | Path | None = None, log: Callable[[str], None] | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._log = log
        self.records: list[dict[str, Any]] = []
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:  # noqa: BLE001 -- a diagnostic never raises (module invariant)
                self.path = None

    def sample(self, label: str, **extra: Any) -> dict[str, Any]:
        """Take, record, print and append one sample. Returns it."""
        record = sample(label)
        record.update(extra)
        self.records.append(record)
        if self._log is not None:
            try:
                self._log(format_sample(record))
            except Exception:  # noqa: BLE001, S110 -- a diagnostic never raises (module invariant)
                pass
        if self.path is not None:
            try:
                with open(self.path, "a") as handle:
                    handle.write(json.dumps(record) + "\n")
            except Exception:  # noqa: BLE001, S110 -- a diagnostic never raises (module invariant)
                pass
        return record

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Sample `<name>:begin`, run the body, `release()`, sample `<name>:end`.

        The release happens INSIDE the context manager's exit path, before the
        closing sample, and also on the exception path -- a stage that dies
        still hands its memory back before whatever runs next.
        """
        started = time.time()
        self.sample(f"{name}:begin")
        try:
            yield
        finally:
            release()
            self.sample(f"{name}:end", stage=name, seconds=round(time.time() - started, 1))

    def peak(self) -> int | None:
        """Max `rss` over the samples taken so far (None if none could be read)."""
        values = [r["rss"] for r in self.records if r.get("rss") is not None]
        return max(values) if values else None
