"""trippy command-line interface.

Module: trippy.cli
Invariants: `smoke` only touches MPS when --device mps is explicitly passed
    (never a silent default); the `render`/`train`/`eval` stubs do no work
    and always exit 2. `density` builds a PointSource and prints/saves its
    PointSet.summary(); it is CPU-only (point sources never touch MPS).
Related docs: docs/SPEC.md "Technical design", AGENTS.md forbidden
    list (no direct GPU/MPS work outside scripts/gpu_submit.sh -- `smoke
    --device mps` is only ever invoked by the GPU-queue job itself);
    docs/SPEC.md D4 (point sources).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from trippy import __version__
from trippy.config import load_settings, pick_device
from trippy.constants import (
    DEFAULT_DENSITY_COLMAP_SPARSE_DIR,
    DEFAULT_DENSITY_GAUSSIAN_PLY,
    DEFAULT_MIN_OPACITY,
    GIT_DESCRIBE_MATCH_PATTERN,
    SMOKE_MPS_TEST_TENSOR_LEN,
)
from trippy.points.colmap_sparse import ColmapSparseSource
from trippy.points.gaussian_ply import GaussianPlySource
from trippy.points.source import PointSource

_METAL_ADD_ONE_SRC = """
kernel void add_one(device float* x [[buffer(0)]],
                     uint id [[thread_position_in_grid]]) {
    x[id] = x[id] + 1.0f;
}
"""


def _git_build_tag() -> str:
    """Return `git describe --tags --match 'build-*' --always`, tolerating any failure."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--match", GIT_DESCRIBE_MATCH_PATTERN, "--always"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        tag = result.stdout.strip()
        return tag if tag else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _run_mps_smoke_kernel() -> list[float]:
    """Compile and run a trivial Metal kernel (add 1.0 to each element) via
    torch.mps.compile_shader, returning the result as a plain Python list.

    Only called when device.type == "mps"; never exercised by CPU tests.
    """
    lib = torch.mps.compile_shader(_METAL_ADD_ONE_SRC)
    t = torch.zeros(SMOKE_MPS_TEST_TENSOR_LEN, device="mps")
    lib.add_one(t)
    torch.mps.synchronize()
    return t.cpu().tolist()


def _cmd_smoke(args: argparse.Namespace) -> int:
    settings = load_settings()
    device = pick_device(args.device)

    print(f"trippy version: {__version__}")
    print(f"python version: {platform.python_version()}")
    print(f"torch version: {torch.__version__}")
    print(f"mps available: {torch.backends.mps.is_available()}")
    print(f"torch.mps.compile_shader available: {hasattr(torch.mps, 'compile_shader')}")
    print(f"git build tag: {_git_build_tag()}")
    print(f"SPLATS_ROOT: {settings.splats_root} (exists={settings.splats_root.exists()})")
    print(f"TRIPPY_OUTPUT: {settings.trippy_output}")
    print(f"selected device: {device}")

    if device.type == "mps":
        result = _run_mps_smoke_kernel()
        print(f"metal add_one([0]*{SMOKE_MPS_TEST_TENSOR_LEN}) -> {result}")

    return 0


def _cmd_not_implemented(name: str):
    def _run(_args: argparse.Namespace) -> int:
        print(f"{name}: not implemented yet (see docs/SPEC.md)")
        return 2

    return _run


def _build_density_source(args: argparse.Namespace) -> PointSource:
    """Construct the requested PointSource, filling in default paths.

    Default paths (relative to SPLATS_ROOT, see trippy.config.load_settings)
    point at the karekare kk-coherent scene per docs/SPEC.md's worked
    example; --path overrides either default explicitly.
    """
    settings = load_settings()
    if args.source == "gaussian":
        path = Path(args.path) if args.path else settings.splats_root / DEFAULT_DENSITY_GAUSSIAN_PLY
        return GaussianPlySource(
            path,
            min_opacity=args.min_opacity,
            size_mode=args.size_mode,
            max_points=args.max_points,
        )
    if args.source == "colmap":
        path = Path(args.path) if args.path else settings.splats_root / DEFAULT_DENSITY_COLMAP_SPARSE_DIR
        return ColmapSparseSource(path)
    raise ValueError(f"unsupported --source {args.source!r}")  # pragma: no cover (argparse choices guard this)


def _cmd_density(args: argparse.Namespace) -> int:
    source = _build_density_source(args)
    point_set = source.build()
    summary = point_set.summary()

    print(f"source: {source.describe()}")
    print(f"count: {summary['count']}")
    print(f"bbox_min: {summary['bbox_min']}")
    print(f"bbox_max: {summary['bbox_max']}")
    print(f"median_nn_distance: {summary['median_nn_distance']}")
    print(f"provenance_histogram: {summary['provenance_histogram']}")
    # Single-line JSON, marker-prefixed so scripts/tests can grep+parse it
    # without depending on the human-readable lines above staying stable.
    print(f"JSON:{json.dumps(summary)}")

    if args.out is not None:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {args.out}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trippy")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="print environment + device diagnostics")
    smoke.add_argument("--device", choices=["cpu", "mps"], default=None)
    smoke.set_defaults(func=_cmd_smoke)

    for name in ("render", "train", "eval"):
        stub = sub.add_parser(name, help=f"{name} (not implemented yet)")
        stub.set_defaults(func=_cmd_not_implemented(name))

    density = sub.add_parser("density", help="build a point source and print PointSet.summary()")
    density.add_argument("--source", choices=["gaussian", "colmap"], required=True)
    density.add_argument("--path", default=None, help="override the default source path")
    density.add_argument("--min-opacity", type=float, default=DEFAULT_MIN_OPACITY, help="gaussian source only")
    density.add_argument(
        "--size-mode", choices=["scale", "knn"], default="scale", help="gaussian source only"
    )
    density.add_argument("--max-points", type=int, default=None, help="gaussian source only")
    density.add_argument("--out", default=None, help="optional path to also write summary JSON")
    density.set_defaults(func=_cmd_density)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
