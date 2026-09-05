#!/usr/bin/env python
"""Dump small synthetic rasteriser fixtures for the Rust parity test.

Module: tools.dump_raster_fixture
Purpose: write, for each (mode, pixel_center) combination, a self-contained
    directory holding a synthetic point set, a camera, the render parameters,
    and the per-layer images `trippy.raster.render_pyramid` produces for them
    on CPU. `rust/crates/brush-pyramid`'s parity tests load exactly these
    files and assert its own CPU and GPU forward passes reproduce them, so
    this script is the single definition of the Rust/Python contract.
Invariants:
    - SYNTHETIC ONLY. The point set comes from a seeded numpy Generator; no
      scene, photo, or checkpoint data is ever written here (AGENTS.md
      section 6, public-repo rules).
    - Fixtures are deliberately tiny (64x48, 3 layers, 500 points) so the
      whole `tests/fixtures/synthetic/` tree stays well under 1 MB and the
      Rust CPU parity test runs in milliseconds.
    - `points.npz` is written UNCOMPRESSED (`np.savez`, ZIP_STORED) and
      `expected.npz` COMPRESSED (`np.savez_compressed`, ZIP_DEFLATE) on
      purpose: between them they exercise both branches of the Rust npz
      reader (`brush_pyramid::npz`).
    - float32 in, float32 out. The Rust port is float32-only, so dumping a
      float64 reference would compare two different computations.
    - Regenerating must be deterministic: same seed, same bytes. The
      accompanying test (tests/test_dump_raster_fixture.py) re-renders and
      compares rather than re-writing, so a semantic change to
      `trippy.raster` fails CI instead of silently rewriting the fixture.
Units / frames: `xyz` COLMAP world frame in world units; `size` world units;
    `K` layer-0 pixels; depth is camera-space z (positive in front).
Related docs: docs/GEOMETRY.md (pixel-centre convention, pyramid layers,
    layer selection); docs/ARCHITECTURE.md (Rust section);
    rust/crates/brush-pyramid/README-less module docs in src/lib.rs.

Usage:
    PYTHONPATH=. TRIPS_DEVICE=cpu python tools/dump_raster_fixture.py
    PYTHONPATH=. TRIPS_DEVICE=cpu python tools/dump_raster_fixture.py --out /tmp/fx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from trippy.constants import (
    RASTER_ALPHA_MIN,
    RASTER_MAX_FRAGS,
    RASTER_T_CUTOFF,
    RASTER_ZNEAR,
)
from trippy.raster import render_pyramid

# Fixture geometry. Small on purpose -- see the module docstring's size
# invariant. 64x48 with 3 layers gives layer shapes (48, 64), (24, 32),
# (12, 16) under "ceil" halving, so every layer is non-degenerate.
FIXTURE_WIDTH = 64
FIXTURE_HEIGHT = 48
FIXTURE_NUM_LAYERS = 3
FIXTURE_NUM_POINTS = 500
# 4 channels: the rasteriser's default feature width (RGB + one learned
# channel), and a member of trippy.constants.RASTER_SUPPORTED_CHANNELS.
FIXTURE_NUM_CHANNELS = 4
FIXTURE_SEED = 20260906

# The two deliberate fragment pile-ups (see `build_scene`). The cap cluster must
# comfortably exceed RASTER_MAX_FRAGS (16) so `n_used` saturates; the cutoff
# cluster must drive transmittance below RASTER_T_CUTOFF before that. The
# quarter-pixel targets below are the ones verified to make the transmittance
# break fire in all six (mode, pixel_center) fixtures.
CLUSTER_CAP_POINTS = 40
CLUSTER_CAP_U = 44.5
CLUSTER_CAP_V = 12.5
CLUSTER_CAP_SIZE_PX = 0.36
CLUSTER_CAP_CONF = 0.04

CLUSTER_TIE_POINTS = 8
CLUSTER_TIE_U = 51.75
CLUSTER_TIE_V = 39.75
CLUSTER_TIE_SIZE_PX = 0.5
CLUSTER_TIE_DEPTH = 3.0

CLUSTER_CUTOFF_POINTS = 40
CLUSTER_CUTOFF_U = 20.75
CLUSTER_CUTOFF_V = 33.75
CLUSTER_CUTOFF_SIZE_PX = 6.0
CLUSTER_CUTOFF_CONF = 0.99

# A background that is neither zero nor uniform, so a Rust port that forgets
# `out += t_final * bg` (or applies it per channel wrongly) fails loudly.
FIXTURE_BG = (0.10, 0.20, 0.30, 0.40)

# Every combination the parity test covers: three layer-selection modes x
# both pixel-centre conventions (docs/GEOMETRY.md).
FIXTURE_MODES = ("trilinear", "broadcast", "trips")
FIXTURE_PIXEL_CENTERS = ("half", "integer")

DEFAULT_OUT = Path("tests/fixtures/synthetic")


def build_scene(rng: np.random.Generator) -> dict:
    """Synthesise a point cloud and a camera looking at it.

    The camera sits at the world origin looking down +Z (docs/GEOMETRY.md:
    `x_cam = R @ x_world + t`, +Z in front, +Y down). Points are scattered in
    a slab in front of it, with a deliberate spread of projected sizes so
    that all of `layer_bounds`' branches are exercised: sub-pixel points (the
    exponential floor), points straddling two layers, and points larger than
    the coarsest layer (which clamp).

    Args:
        rng: seeded numpy Generator; the only source of randomness.

    Returns:
        dict with numpy float32 arrays `xyz` (N, 3), `size` (N,),
        `feat` (N, C), `conf` (N,), plus `K` (3, 3), `R` (3, 3), `t` (3,).
    """
    n = FIXTURE_NUM_POINTS

    fx = 60.0
    fy = 60.0
    cx = FIXTURE_WIDTH / 2.0
    cy = FIXTURE_HEIGHT / 2.0

    # Depths from 1 to 9 world units. `size_px = fx * size / z`, so with
    # fx = 60 and the sizes below this spans roughly 0.01 .. 18 layer-0
    # pixels -- wide enough that every branch of `layer_bounds` fires and
    # that all three pyramid layers receive a workable number of fragments.
    depth = rng.uniform(1.0, 9.0, size=n)
    # Lateral spread wide enough that a good fraction of the footprints
    # straddle the image border -- that is the only way the "drop, never
    # clamp" rule and mode "trips"'s all-four-corners gate get tested.
    x = rng.uniform(-0.62, 0.62, size=n) * depth
    y = rng.uniform(-0.50, 0.50, size=n) * depth
    xyz = np.stack([x, y, depth], axis=1)

    # Log-uniform world sizes -> a wide, well-spread range of size_px.
    size = np.exp(rng.uniform(np.log(2e-3), np.log(3.0e-1), size=n))

    feat = rng.uniform(0.0, 1.0, size=(n, FIXTURE_NUM_CHANNELS))
    # Confidence strictly inside (0, 1), as PointSet.conf0 guarantees.
    conf = rng.uniform(0.05, 0.95, size=n)

    # Two deliberate pile-ups, because a random scatter never stacks more
    # than a handful of fragments on one pixel and would leave both of
    # blend_fwd's stopping rules untested (they are the two `break`s in
    # trippy/raster/metal_src/blend_fwd.metal).
    #
    # Cluster A -- CAP: many *low* alpha fragments on one line of sight.
    # Transmittance decays as (1 - alpha)^k, so with a sub-pixel size (whose
    # layer factor is the ~0.65 exponential floor) and conf 0.04 it is still
    # ~0.98 after 16 fragments: the `used >= max_frags` break is what stops
    # the loop, and n_used saturates at max_frags.
    #
    # Cluster B -- CUTOFF: fewer, much *higher* alpha fragments, so T falls
    # under RASTER_T_CUTOFF (1e-3) first and n_used stops below max_frags.
    # Reaching the cutoff before the cap needs alpha > ~0.35 per fragment,
    # and alpha = bilinear_weight * conf * layer_factor: hence conf 0.99, a
    # layer_factor of 1.0, and a target landing at a quarter-pixel offset.
    # A footprint centred exactly on a pixel *corner* would split its weight
    # 4 x 0.25 in both pixel-centre conventions, capping alpha at 0.25, and
    # the cutoff could then never fire before max_frags.
    #
    # The size is 6 layer-0 pixels, NOT a power of two, and that matters.
    # `layer_bounds` is floor/ceil of log2(size_px), so a size sitting exactly
    # on a power of two is a knife edge: one ulp either side moves the point
    # between `lower == upper` (layer factor 1.0) and a straddle whose lower
    # layer gets factor ~1e-7 -- below `alpha_min`, so four fragments appear
    # or vanish. `size_px` is computed as `fx * size / z`, and a shader
    # compiler is free to reassociate or use a fast reciprocal there, so the
    # GPU and the CPU can legitimately land on opposite sides. With 3 layers,
    # 6 px clamps to `lower == upper == 2` and stays there under any
    # perturbation, which keeps the fixture testing the compositing rules
    # rather than floating-point luck. See docs/LIMITATIONS.md.
    #
    # Both clusters share one line of sight (constant x/z and y/z) at
    # staggered depths, so their fragments land on the same pixel in a
    # well-defined front-to-back order -- which also makes the fixture
    # sensitive to a sort that is not stable in depth.
    for lo, hi, target_u, target_v, target_size_px, cluster_conf in (
        (
            n - CLUSTER_CAP_POINTS - CLUSTER_CUTOFF_POINTS - CLUSTER_TIE_POINTS,
            n - CLUSTER_CUTOFF_POINTS - CLUSTER_TIE_POINTS,
            CLUSTER_CAP_U,
            CLUSTER_CAP_V,
            CLUSTER_CAP_SIZE_PX,
            CLUSTER_CAP_CONF,
        ),
        (
            n - CLUSTER_CUTOFF_POINTS - CLUSTER_TIE_POINTS,
            n - CLUSTER_TIE_POINTS,
            CLUSTER_CUTOFF_U,
            CLUSTER_CUTOFF_V,
            CLUSTER_CUTOFF_SIZE_PX,
            CLUSTER_CUTOFF_CONF,
        ),
    ):
        count = hi - lo
        cluster_depth = np.linspace(2.0, 6.0, count)
        # Invert the projection: a constant image position means a constant
        # x/z and y/z, and a constant projected size means size ~ z.
        xyz[lo:hi, 0] = (target_u - cx) / fx * cluster_depth
        xyz[lo:hi, 1] = (target_v - cy) / fy * cluster_depth
        xyz[lo:hi, 2] = cluster_depth
        size[lo:hi] = target_size_px * cluster_depth / fx
        conf[lo:hi] = cluster_conf

    # Cluster C -- DEPTH TIES: several points at *exactly* the same depth on
    # one line of sight. Every other point in this scene has a distinct
    # depth, so without this the sort's tie-breaking rule would never be
    # tested -- and it matters: compositing is order-dependent, and both the
    # Python composite key and the Rust two-pass radix sort have to fall back
    # to ascending point index for equal depths. Confidences and features
    # stay at their random per-point values, so swapping any two of these
    # fragments changes the composited pixel.
    tie_lo = n - CLUSTER_TIE_POINTS
    xyz[tie_lo:, 0] = (CLUSTER_TIE_U - cx) / fx * CLUSTER_TIE_DEPTH
    xyz[tie_lo:, 1] = (CLUSTER_TIE_V - cy) / fy * CLUSTER_TIE_DEPTH
    xyz[tie_lo:, 2] = CLUSTER_TIE_DEPTH
    size[tie_lo:] = CLUSTER_TIE_SIZE_PX * CLUSTER_TIE_DEPTH / fx

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    # Identity world->camera: the synthetic points are already expressed in
    # the camera frame. A non-trivial R would only re-test xform_a/xform_b,
    # which have their own agreement test; here the point is the rasteriser.
    R = np.eye(3)
    t = np.zeros(3)

    return {
        "xyz": xyz.astype(np.float32),
        "size": size.astype(np.float32),
        "feat": feat.astype(np.float32),
        "conf": conf.astype(np.float32),
        "K": K.astype(np.float32),
        "R": R.astype(np.float32),
        "t": t.astype(np.float32),
    }


def render_reference(scene: dict, mode: str, pixel_center: str) -> tuple[list, dict]:
    """Run the CPU reference rasteriser for one (mode, pixel_center).

    Args:
        scene: the dict returned by `build_scene`.
        mode: one of FIXTURE_MODES.
        pixel_center: one of FIXTURE_PIXEL_CENTERS.

    Returns:
        `(layers, aux)` exactly as `trippy.raster.render_pyramid` returns
        them, with every tensor on CPU in float32.
    """
    as_t = {k: torch.from_numpy(v) for k, v in scene.items()}
    bg = torch.tensor(FIXTURE_BG, dtype=torch.float32)
    return render_pyramid(
        as_t["xyz"],
        as_t["size"],
        as_t["feat"],
        as_t["conf"],
        as_t["K"],
        as_t["R"],
        as_t["t"],
        (FIXTURE_HEIGHT, FIXTURE_WIDTH),
        num_layers=FIXTURE_NUM_LAYERS,
        mode=mode,
        bg=bg,
        max_frags=RASTER_MAX_FRAGS,
        t_cutoff=RASTER_T_CUTOFF,
        alpha_min=RASTER_ALPHA_MIN,
        znear=RASTER_ZNEAR,
        pixel_center=pixel_center,
        pyramid_halving="ceil",
        compute_dtype=None,
    )


def fixture_name(mode: str, pixel_center: str) -> str:
    """Directory name for one fixture (stable; referenced from Rust)."""
    return f"raster_fixture_{mode}_{pixel_center}"


def write_fixture(out_dir: Path, scene: dict, mode: str, pixel_center: str) -> dict:
    """Render one (mode, pixel_center) and write its fixture directory.

    Writes `points.npz` (uncompressed), `camera.json`, `params.json`,
    `expected.npz` (compressed) and `meta.json`. See the module docstring for
    why the two npz files use different compression.

    Args:
        out_dir: parent directory; the fixture goes in `out_dir/<name>/`.
        scene: the dict returned by `build_scene`.
        mode, pixel_center: which variant to render.

    Returns:
        The `meta.json` contents, for the caller's summary line.
    """
    layers, aux = render_reference(scene, mode, pixel_center)
    fixture_dir = out_dir / fixture_name(mode, pixel_center)
    fixture_dir.mkdir(parents=True, exist_ok=True)

    # Uncompressed (ZIP_STORED) -- exercises the Rust npz reader's stored path.
    np.savez(
        fixture_dir / "points.npz",
        xyz=scene["xyz"],
        size=scene["size"],
        feat=scene["feat"],
        conf=scene["conf"],
    )

    camera = {
        "width": FIXTURE_WIDTH,
        "height": FIXTURE_HEIGHT,
        "fx": float(scene["K"][0, 0]),
        "fy": float(scene["K"][1, 1]),
        "cx": float(scene["K"][0, 2]),
        "cy": float(scene["K"][1, 2]),
        # Row-major 3x3 world->camera rotation, then the translation.
        "R": [float(v) for v in scene["R"].reshape(-1)],
        "t": [float(v) for v in scene["t"].reshape(-1)],
    }
    (fixture_dir / "camera.json").write_text(json.dumps(camera, indent=2) + "\n")

    params = {
        "mode": mode,
        "pixel_center": pixel_center,
        "pyramid_halving": "ceil",
        "num_layers": FIXTURE_NUM_LAYERS,
        "num_channels": FIXTURE_NUM_CHANNELS,
        "max_frags": RASTER_MAX_FRAGS,
        "t_cutoff": RASTER_T_CUTOFF,
        "alpha_min": RASTER_ALPHA_MIN,
        "znear": RASTER_ZNEAR,
        "background": list(FIXTURE_BG),
    }
    (fixture_dir / "params.json").write_text(json.dumps(params, indent=2) + "\n")

    # Compressed (ZIP_DEFLATE) -- exercises the reader's inflate path, and
    # keeps the whole fixture tree small (these images are mostly background).
    expected: dict[str, np.ndarray] = {}
    for layer, image in enumerate(layers):
        expected[f"layer_{layer}"] = image.detach().cpu().numpy().astype(np.float32)
        expected[f"t_final_{layer}"] = aux["t_final"][layer].detach().cpu().numpy().astype(np.float32)
        expected[f"n_used_{layer}"] = aux["n_used"][layer].detach().cpu().numpy().astype(np.int32)
    np.savez_compressed(fixture_dir / "expected.npz", **expected)

    meta = {
        "num_fragments": int(aux["num_fragments"]),
        "fragments_per_layer": [int(v) for v in aux["fragments_per_layer"]],
        "layer_shapes": [[int(h), int(w)] for h, w in aux["grid"].shapes],
    }
    (fixture_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    """Write every fixture under `--out` and print a one-line summary each."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"parent directory for the fixture dirs (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(FIXTURE_SEED)
    scene = build_scene(rng)

    total_bytes = 0
    for mode in FIXTURE_MODES:
        for pixel_center in FIXTURE_PIXEL_CENTERS:
            meta = write_fixture(args.out, scene, mode, pixel_center)
            name = fixture_name(mode, pixel_center)
            size = sum(p.stat().st_size for p in (args.out / name).rglob("*") if p.is_file())
            total_bytes += size
            print(
                f"{name}: {meta['num_fragments']} fragments "
                f"{meta['fragments_per_layer']} per layer, {size / 1024:.0f} KiB"
            )
    print(f"total: {total_bytes / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
