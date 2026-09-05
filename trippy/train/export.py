"""Export a trippy PointSet (or raw arrays) to a 3DGS-compatible binary PLY.

Module: trippy.train.export
Invariants: writes a single binary_little_endian PLY vertex block with one
    numpy structured-array `tofile()` call (no per-row Python loop), so
    exports scale to the ~7M-point production PLYs trippy.points.gaussian_ply
    already reads. The property list/order/dtypes written here are the
    reader's contract: trippy.points.gaussian_ply (and Splats' extent
    gate / ply_extract tools, and the Brush viewer) must be able to load
    this file back unchanged. This module is deliberately the exact
    mathematical inverse of trippy.points.gaussian_ply's read side.
Related docs: docs/GEOMETRY.md "3DGS PLY export mapping" (the four value
    mappings below); docs/SPEC.md D4 (point sources) and the v0.2.0
    milestone ("eval/export writing a 3DGS-compatible PLY so existing
    Splats tooling runs unchanged").

Value mapping (docs/GEOMETRY.md "3DGS PLY export mapping"):
    f_dc_{0,1,2} = (rgb - 0.5) / SH_C0          (inverse of gaussian_ply's
                                                  rgb = clip(0.5 + SH_C0*f_dc))
    opacity      = logit(clamp(conf, eps, 1-eps))  (inverse of sigmoid)
    scale_{0,1,2} = log(size)                    (isotropic; inverse of exp)
    rot_{0,1,2,3} = EXPORT_IDENTITY_ROT           (wxyz identity; TRIPS has
                                                    no learned rotation)
    nx, ny, nz    = 0                             (unused; present because
                                                    3DGS PLYs conventionally
                                                    carry a normal property)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trippy.constants import (
    EXPORT_IDENTITY_ROT,
    EXPORT_OPACITY_CLAMP_EPS,
    SH_C0,
    SH_DEGREE_THREE,
    SH_DEGREE_THREE_NUM_REST_COEFFS,
    SH_DEGREE_ZERO,
)
from trippy.points.gaussian_ply import GaussianPlySource
from trippy.points.source import PointSet

# Base property list, in the exact order written to the PLY header. When
# sh_degree=SH_DEGREE_THREE, f_rest_0..44 are appended after rot_3 (3DGS
# readers -- including trippy.points.gaussian_ply -- select properties by
# name from a structured dtype, so trailing placement is safe).
_BASE_PROPS = [
    "x", "y", "z",
    "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]  # fmt: skip


def _logit(p: np.ndarray) -> np.ndarray:
    """Inverse sigmoid, vectorised. Caller must clamp p away from {0, 1}."""
    return np.log(p / (1.0 - p))


def _prop_names(sh_degree: int) -> list[str]:
    """Full ordered property-name list for a given sh_degree flag.

    Args:
        sh_degree: SH_DEGREE_ZERO (default, DC colour only) or
            SH_DEGREE_THREE (also writes 45 zero-filled f_rest_* fields).

    Raises:
        ValueError: any other sh_degree value.
    """
    if sh_degree == SH_DEGREE_ZERO:
        return list(_BASE_PROPS)
    if sh_degree == SH_DEGREE_THREE:
        return list(_BASE_PROPS) + [f"f_rest_{i}" for i in range(SH_DEGREE_THREE_NUM_REST_COEFFS)]
    raise ValueError(f"sh_degree must be {SH_DEGREE_ZERO} or {SH_DEGREE_THREE}, got {sh_degree!r}")


def _ply_header(n: int, prop_names: list[str]) -> bytes:
    """Binary_little_endian PLY header for an all-float32 vertex element."""
    lines = [b"ply", b"format binary_little_endian 1.0", f"element vertex {n}".encode("ascii")]
    lines += [f"property float {name}".encode("ascii") for name in prop_names]
    lines.append(b"end_header")
    return b"\n".join(lines) + b"\n"


def write_gaussian_ply(
    path: str | Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    conf: np.ndarray,
    size: np.ndarray,
    provenance: np.ndarray | None = None,
    sh_degree: int = SH_DEGREE_ZERO,
) -> Path:
    """Write a 3DGS-compatible binary_little_endian PLY.

    Args:
        path: output .ply path; parent directories are created if missing.
        xyz: (N, 3) float, world-frame position, world units.
        rgb: (N, 3) float, linear colour in [0, 1] per channel.
        conf: (N,) float, confidence/opacity in the open interval (0, 1).
        size: (N,) float, world-unit splat radius; must be strictly
            positive (log(size) is undefined otherwise).
        provenance: optional (N,) uint8 PointSet provenance codes
            (trippy.constants.PROVENANCE_*). If given, also writes a
            `<path>.provenance.npy` sidecar via write_provenance_sidecar().
        sh_degree: SH_DEGREE_ZERO (default; no f_rest_* properties) or
            SH_DEGREE_THREE (also writes 45 zero-filled f_rest_0..44).

    Returns:
        The output path.

    Raises:
        ValueError: shape mismatch between xyz/rgb/conf/size, or size
            contains a non-positive value, or an unsupported sh_degree.
    """
    path = Path(path)
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)

    n = xyz.shape[0]
    if xyz.shape != (n, 3):
        raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")
    if rgb.shape != (n, 3):
        raise ValueError(f"rgb must be (N, 3), got {rgb.shape}")
    if conf.shape != (n,):
        raise ValueError(f"conf must be ({n},), got {conf.shape}")
    if size.shape != (n,):
        raise ValueError(f"size must be ({n},), got {size.shape}")
    if n > 0 and np.any(size <= 0):
        raise ValueError("size must be strictly positive (log(size) is undefined otherwise)")

    names = _prop_names(sh_degree)
    dtype = np.dtype([(name, "<f4") for name in names])
    verts = np.zeros(n, dtype=dtype)  # nx/ny/nz and any f_rest_* stay zero.

    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    f_dc = (rgb - 0.5) / SH_C0
    verts["f_dc_0"], verts["f_dc_1"], verts["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]

    conf_clamped = np.clip(conf, EXPORT_OPACITY_CLAMP_EPS, 1.0 - EXPORT_OPACITY_CLAMP_EPS)
    verts["opacity"] = _logit(conf_clamped)

    log_size = np.log(size)
    verts["scale_0"], verts["scale_1"], verts["scale_2"] = log_size, log_size, log_size

    verts["rot_0"], verts["rot_1"], verts["rot_2"], verts["rot_3"] = EXPORT_IDENTITY_ROT

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(_ply_header(n, names))
        verts.tofile(f)

    if provenance is not None:
        write_provenance_sidecar(path, provenance)

    return path


def export_pointset_ply(path: str | Path, pointset: PointSet, sh_degree: int = SH_DEGREE_ZERO) -> Path:
    """Convenience wrapper: export a PointSet directly (see write_gaussian_ply).

    Args:
        path: output .ply path.
        pointset: the PointSet to export (xyz/rgb0/conf0/size0/provenance).
        sh_degree: forwarded to write_gaussian_ply.

    Returns:
        The output path.
    """
    return write_gaussian_ply(
        path,
        xyz=pointset.xyz,
        rgb=pointset.rgb0,
        conf=pointset.conf0,
        size=pointset.size0,
        provenance=pointset.provenance,
        sh_degree=sh_degree,
    )


def write_provenance_sidecar(path: str | Path, provenance: np.ndarray) -> Path:
    """Write per-point provenance codes to `<path>.provenance.npy`.

    This sidecar is not read by any 3DGS tool (it is trippy-only metadata
    for post hoc per-source diagnostics); it lives next to the .ply purely
    for record-keeping, matching AGENTS.md's provenance-tracking rule.

    Args:
        path: the .ply path this sidecar accompanies (sidecar is
            `str(path) + ".provenance.npy"`, so it never collides with the
            .ply itself).
        provenance: (N,) provenance codes (trippy.constants.PROVENANCE_*).

    Returns:
        The sidecar path.
    """
    sidecar = Path(str(path) + ".provenance.npy")
    np.save(sidecar, np.asarray(provenance, dtype=np.uint8))
    return sidecar


def read_back_check(path: str | Path) -> int:
    """Round-trip a written PLY through GaussianPlySource; return its count.

    Uses min_opacity=0.0 so every row written by write_gaussian_ply
    survives the read-side opacity filter (sigmoid(opacity) is always
    strictly > 0 for a finite opacity, so nothing is dropped) -- this
    checks the export/import pipeline is lossless in *count*, not that
    every point would pass a real training run's default filter.

    Args:
        path: a .ply path previously written by write_gaussian_ply.

    Returns:
        Number of vertices read back.
    """
    return len(GaussianPlySource(path, min_opacity=0.0).build())
