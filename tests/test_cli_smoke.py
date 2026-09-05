"""End-to-end subprocess test for `python -m trippy.cli smoke --device cpu`.

Module: tests.test_cli_smoke
Invariants under test: the CLI is invocable via `python -m trippy.cli`
    (not just as an installed console script), exits 0 on the CPU smoke
    path, and prints torch diagnostics.
Related docs: /tmp/trippy-plan.md "Verification (end-to-end)" item 1
    (`uv run trippy smoke` prints torch/MPS/compile_shader/build tag).
"""

from __future__ import annotations

import subprocess
import sys


def test_cli_smoke_cpu() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "trippy.cli", "smoke", "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "torch" in result.stdout
