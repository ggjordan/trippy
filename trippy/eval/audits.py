"""Wrappers around Splats' shade audit + extent gate: subprocess, parsed into plain dicts.

Module: trippy.eval.audits
Purpose: run two of Jordan's existing, unmodified Splats tools against a
    trippy-exported PLY and turn their output into JSON-serialisable dicts,
    so `trippy candidate-report` (trippy.cli) can rank a checkpoint the same
    way docs/EXPERIMENTS.md's "Ranking candidates: metrics and gates"
    section describes, without trippy re-implementing either audit:
      - `run_shade_audit`: ~/Splats/tools/depthprior_shade_audit.py
        (opacity mass sitting in the karekare shade's walkable volume --
        docs/SPEC.md's stage-1 gate). Uses that script's own `--json-out`
        flag (a stable, structured artifact) rather than parsing its
        human-readable stdout.
      - `run_extent_gate`: ~/Splats/tools/tmp/extent-audit/extent_gate.py
        (point-cloud radius percentiles / scene diagonal -- catches scene
        sprawl). This script has no `--json-out`, so its stdout table is
        parsed with regexes into the same fields it prints.
    Both run via Splats' own `ml-sharp` venv interpreter (not trippy's own
    venv -- these scripts import Splats-local helper modules, e.g.
    `ply_extract.py`, that trippy does not have and should not vendor).
Invariants: neither function ever raises on a *missing tool* silently --
    `FileNotFoundError` is raised immediately (before touching subprocess)
    if the venv interpreter or script path doesn't exist on this machine,
    matching AGENTS.md's "this repo must stay green on a machine without
    ~/Splats" via callers using `pytest.skip` (see tests/conftest.py's
    `splats_scene` fixture) -- `audit_report` instead *catches* both kinds
    of failure per-audit, so one broken/missing tool never blocks the
    other's numbers, and `trippy candidate-report` always finishes.
Related docs: docs/EXPERIMENTS.md "Shade audit", "Extent gate"; docs/SPEC.md D10.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

from trippy.config import load_settings
from trippy.constants import (
    AUDIT_EXTENT_SCRIPT_REL,
    AUDIT_SHADE_SCRIPT_REL,
    AUDIT_SUBPROCESS_TIMEOUT_S,
    AUDIT_VENV_PYTHON_REL,
)

_HEADER_RE = re.compile(r"^(?P<name>.+?)\s+\((?P<n>[\d,]+) gaussians\)\s*$")
_CENTRE_RE = re.compile(r"median centre\s+\[\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\]")
_RADIUS_RE = re.compile(
    r"radius p50/p99/p99\.9/max\s+([-\d.eE]+)\s*/\s*([-\d.eE]+)\s*/\s*([-\d.eE]+)\s*/\s*([-\d.eE]+)"
)
_DIAG_RE = re.compile(r"scene diagonal.*?box\)\s+([-\d.eE]+)")
_NF_MEANS_RE = re.compile(r"non-finite means\s+([\d,]+)")
_NF_SCALES_RE = re.compile(r"non-finite scales\s+([\d,]+)")


def _tool_paths(script_rel: str) -> tuple[Path, Path]:
    """(venv_python, script) absolute paths under `SPLATS_ROOT`, per trippy.config.load_settings."""
    settings = load_settings()
    return settings.splats_root / AUDIT_VENV_PYTHON_REL, settings.splats_root / script_rel


def _require_tool(python: Path, script: Path) -> None:
    if not python.exists():
        raise FileNotFoundError(f"Splats ml-sharp venv python not found: {python}")
    if not script.exists():
        raise FileNotFoundError(f"Splats audit script not found: {script}")


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{cmd[1]} failed (exit {result.returncode}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def run_shade_audit(
    ply_paths: list[str | Path],
    sparse_txt_dir: str | Path,
    frames: list[str] | None = None,
    timeout: float = AUDIT_SUBPROCESS_TIMEOUT_S,
) -> dict:
    """Run `depthprior_shade_audit.py` on `ply_paths`, parsed from its own `--json-out`.

    Args:
        ply_paths: one or more exported 3DGS-compatible PLYs to audit.
        sparse_txt_dir: a COLMAP *text*-format sparse model directory (the
            script parses `cameras.txt`/`images.txt`/`points3D.txt`
            directly -- a binary `sparse/0` model is not accepted).
        frames: image names defining the shade region; None uses the
            script's own default (karekare's `IMG_3828.jpg..IMG_3833.jpg`,
            matching `trippy.constants.SHADE_FRAMES_KK`).
        timeout: subprocess wall-clock timeout, seconds.

    Returns:
        The parsed `--json-out` payload: `{"scene", "frames", "znear_frac",
        "zfar_frac", "views": [...], "results": [{"path", "n",
        "n_in_region", "mass_in_region", "dark_mass_lum<T>", ...}, ...]}`.

    Raises:
        FileNotFoundError: the Splats ml-sharp venv or the script itself
            isn't present on this machine.
        RuntimeError: the subprocess exited non-zero, or exited zero but
            did not write the expected `--json-out` file.
    """
    python, script = _tool_paths(AUDIT_SHADE_SCRIPT_REL)
    _require_tool(python, script)

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_out = Path(tmp_dir) / "shade_audit.json"
        cmd = [str(python), str(script), "--scene", str(sparse_txt_dir)]
        if frames is not None:
            cmd += ["--frames", *frames]
        cmd += ["--json-out", str(json_out), *[str(p) for p in ply_paths]]
        _run(cmd, timeout)
        if not json_out.exists():
            raise RuntimeError(f"{script} exited 0 but did not write {json_out}")
        return json.loads(json_out.read_text())


def _parse_extent_gate_stdout(stdout: str) -> list[dict]:
    """Parse `extent_gate.py`'s per-PLY stdout table into a list of dicts, one per block."""
    records: list[dict] = []
    current: dict | None = None
    for line in stdout.splitlines():
        header = _HEADER_RE.match(line.strip())
        if header:
            if current is not None:
                records.append(current)
            current = {"ply_name": header.group("name"), "n": int(header.group("n").replace(",", ""))}
            continue
        if current is None:
            continue
        if m := _CENTRE_RE.search(line):
            current["median_centre"] = [float(x) for x in m.groups()]
        elif m := _RADIUS_RE.search(line):
            p50, p99, p999, mx = (float(x) for x in m.groups())
            current.update(radius_p50=p50, radius_p99=p99, radius_p999=p999, radius_max=mx)
        elif m := _DIAG_RE.search(line):
            current["scene_diagonal"] = float(m.group(1))
        elif m := _NF_MEANS_RE.search(line):
            current["non_finite_means"] = int(m.group(1).replace(",", ""))
        elif m := _NF_SCALES_RE.search(line):
            current["non_finite_scales"] = int(m.group(1).replace(",", ""))
    if current is not None:
        records.append(current)
    return records


def run_extent_gate(ply_paths: list[str | Path], timeout: float = AUDIT_SUBPROCESS_TIMEOUT_S) -> dict:
    """Run `extent_gate.py` on `ply_paths`, parsing its stdout table (no `--json-out` exists).

    Args:
        ply_paths: one or more 3DGS-compatible PLYs to audit.
        timeout: subprocess wall-clock timeout, seconds.

    Returns:
        `{"plys": [{"ply_path", "ply_name", "n", "median_centre",
        "radius_p50", "radius_p99", "radius_p999", "radius_max",
        "scene_diagonal", "non_finite_means", "non_finite_scales"}, ...],
        "raw_stdout": str}`, one record per input path, in order.

    Raises:
        FileNotFoundError: the Splats ml-sharp venv or the script itself
            isn't present on this machine.
        RuntimeError: the subprocess exited non-zero, or its stdout didn't
            parse into exactly `len(ply_paths)` records.
    """
    python, script = _tool_paths(AUDIT_EXTENT_SCRIPT_REL)
    _require_tool(python, script)

    cmd = [str(python), str(script), *[str(p) for p in ply_paths]]
    result = _run(cmd, timeout)
    records = _parse_extent_gate_stdout(result.stdout)
    if len(records) != len(ply_paths):
        raise RuntimeError(
            f"expected {len(ply_paths)} extent_gate.py records, parsed {len(records)} from stdout:\n{result.stdout}"
        )
    for record, path in zip(records, ply_paths, strict=True):
        record["ply_path"] = str(path)
    return {"plys": records, "raw_stdout": result.stdout}


def audit_report(
    ply_paths: list[str | Path],
    sparse_txt_dir: str | Path,
    frames: list[str] | None = None,
) -> dict:
    """Run both audits, catching failures independently (see module docstring).

    Args:
        ply_paths: one or more exported 3DGS-compatible PLYs.
        sparse_txt_dir: forwarded to `run_shade_audit`.
        frames: forwarded to `run_shade_audit`.

    Returns:
        `{"shade_audit": <run_shade_audit's dict> | {"error": str},
        "extent_gate": <run_extent_gate's dict> | {"error": str}}` -- a
        failure in one never suppresses the other's result.
    """
    report: dict = {}
    try:
        report["shade_audit"] = run_shade_audit(ply_paths, sparse_txt_dir, frames=frames)
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as exc:
        report["shade_audit"] = {"error": str(exc)}
    try:
        report["extent_gate"] = run_extent_gate(ply_paths)
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as exc:
        report["extent_gate"] = {"error": str(exc)}
    return report
