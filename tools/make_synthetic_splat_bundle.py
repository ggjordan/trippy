#!/usr/bin/env python
"""Build a synthetic bundle whose `blend.splat_ply` the viewer can render live.

Module: tools.make_synthetic_splat_bundle
Purpose: the live splat path (`trips_viewer::splat`) needs a bundle that names a
    Gaussian PLY, and every real one is a private Karekare scene. This writes a
    throwaway, entirely generated one: a four-image COLMAP scene, a 3DGS PLY
    written by `trippy.train.export.write_gaussian_ply`, a two-epoch CPU
    training run over it, and `trippy export-bundle`'s own output. The result is
    what `scripts/viewer_splat_check.sh` renders twice to prove that mix 0 (pure
    splat) and mix 1 (pure TRIPS) differ at a pose no camera ever stood at.
Invariants:
    - SYNTHETIC ONLY. Every array comes from a seeded `numpy.random.Generator`;
      no photograph, checkpoint, render or PLY of Jordan's scenes is read or
      written (AGENTS.md section 6). The scene's "photos" are renders of the
      generated point cloud through trippy's own CPU rasteriser.
    - CPU ONLY. `TRIPS_DEVICE=cpu`; nothing here touches MPS, and it must stay
      that way so the fixture can be rebuilt outside the GPU queue.
    - The Gaussians are deliberately **larger** than the TRIPS points the same
      cloud trains as (`--splat-size`, default 0.25 world units against the
      point set's 0.05). A splat that projects to a fraction of a pixel renders
      to almost nothing, and "almost nothing vs the TRIPS frame" would pass a
      difference test for the wrong reason.
    - Written under `TRIPPY_OUTPUT`, never inside the repo: the bundle is a few
      MB of generated data, not a committed fixture.
Units: world units for positions and sizes; pixels for the scene's intrinsics.
Related docs: `docs/decisions/ADR-0006-viewer-integration.md`;
    `rust/crates/trips-viewer/src/splat.rs`; `docs/USER_GUIDE.md` "Blend panel".

Usage:
    PYTHONPATH=. TRIPS_DEVICE=cpu python tools/make_synthetic_splat_bundle.py \
        --out "$TRIPPY_OUTPUT/fixtures/synthetic-splat"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

#: `tests/` holds the synthetic COLMAP-scene builder this reuses rather than
#: forking. It is a dev tool, not part of the `trippy` package, so importing a
#: test helper is the honest dependency: the fixture and the tests must describe
#: the same scene or a screenshot diff would be measuring two different worlds.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests"))

from test_train_helpers import (
    build_synthetic_scene,
    tiny_train_config,
)

from trippy.edit.golden import write_shade_views
from trippy.edit.model import EditDocument, Region
from trippy.render.bundle import BUNDLE_JSON_FILENAME, export_bundle
from trippy.train import prune
from trippy.train.config import PointSourceConfig
from trippy.train.export import write_gaussian_ply
from trippy.train.trainer import Trainer

#: World-unit radius of every Gaussian in the written PLY. See the invariants.
DEFAULT_SPLAT_SIZE = 0.25

#: How many Gaussians the PLY holds. Enough that the splat render covers most of
#: the frame from an off-capture pose, few enough that loading is instant.
DEFAULT_NUM_SPLATS = 4000

#: The synthetic scene's own COLMAP images have ZERO observations
#: (`tests.test_train_helpers.build_synthetic_scene` writes an empty observation
#: line per image), so `prune.build_shade_region` falls back to its documented
#: `d = 1.0`. The default depth-slab fractions (0.05..0.50 of `d`) would then
#: describe a slab 0.05..0.5 world units deep, and this scene's points sit at
#: z in [4, 8] -- the finder would select nothing, for a reason that has nothing
#: to do with whether it works. These two fractions put the slab at 1..10 world
#: units instead, which is where the geometry actually is. Both sides read them
#: out of the sidecar, so the Python and Rust selections still agree exactly.
SHADE_ZNEAR_FRAC = 1.0
SHADE_ZFAR_FRAC = 10.0

#: A `delete` box covering roughly the +x half of the synthetic cloud
#: (`x in [-2, 2]`, `y in [-1.5, 1.5]`, `z in [4, 8]`). The screenshot check
#: renders with and without it; half the frame has to change.
EDIT_DELETE_BOX = {
    "center": [1.25, 0.0, 6.0],
    "half_extents": [1.5, 3.0, 3.0],
}

#: A `blend` sphere in the other half, at `mix = 0` (pure splat). Its job is to
#: exercise the probe pyramid pass and the per-pixel weight, which the delete
#: box does not touch.
EDIT_BLEND_SPHERE = {"center": [-1.25, 0.0, 6.0], "radius": 1.2}


def write_splat_ply(path: Path, num_splats: int, splat_size: float, seed: int) -> Path:
    """Write a 3DGS PLY filling the volume the synthetic cameras look into.

    The cloud spans the same box `tests.test_train_helpers.synthetic_point_set`
    uses (`x, y, z` in `[-2, 2] x [-1.5, 1.5] x [4, 8]`) so a camera yawed off a
    capture view still sees Gaussians, which is the whole point of the fixture.

    Args:
        path: destination `.ply`.
        num_splats: how many Gaussians.
        splat_size: world-unit radius of each (written as `log(size)` into
            `scale_0..2`, per `write_gaussian_ply`).
        seed: RNG seed; the same seed writes the same bytes.

    Returns:
        The path written.
    """
    rng = np.random.default_rng(seed)
    xyz = np.stack(
        [
            rng.uniform(-2.0, 2.0, num_splats),
            rng.uniform(-1.5, 1.5, num_splats),
            rng.uniform(4.0, 8.0, num_splats),
        ],
        axis=1,
    ).astype(np.float32)
    # A smooth colour field rather than noise: a splat render of noise and a
    # TRIPS render of noise differ everywhere for trivial reasons, whereas a
    # structured field makes an eyeball check (Jordan's, later) meaningful.
    rgb = np.stack(
        [
            0.5 + 0.4 * np.sin(xyz[:, 0] * 1.7),
            0.5 + 0.4 * np.cos(xyz[:, 1] * 2.3),
            0.5 + 0.4 * np.sin(xyz[:, 2] * 0.9),
        ],
        axis=1,
    ).astype(np.float32)
    conf = np.full(num_splats, 0.9, dtype=np.float32)
    size = np.full(num_splats, float(splat_size), dtype=np.float32)
    return write_gaussian_ply(path, xyz=xyz, rgb=rgb, conf=conf, size=size)


def write_edits(bundle_dir: Path, document: dict) -> Path:
    """Write a synthetic `edits.json` next to `bundle.json`.

    Two regions, chosen so one screenshot pair proves each half of
    docs/EDITOR.md Sec 2 separately:

    - a `delete` box over the +x half of the cloud, which removes TRIPS points
      AND (via `compose_gaussian_weights`) scales those Gaussians' opacity to
      zero, so the frame must change there and `undo` must restore it exactly;
    - a `blend` sphere at `mix = 0` in the -x half, which changes NO point's
      alpha at all and can only show up through the probe pass's per-pixel
      weight -- so if the frame changes inside it, that machinery works.

    The document is built through the ordinary mutation API, so it carries a
    real undo log the viewer (and `EditDocument.undo`) can walk.

    Args:
        bundle_dir: the bundle directory (holding `bundle.json`).
        document: the parsed `bundle.json`, for its `"format"`.

    Returns:
        The path written.
    """
    edits = EditDocument.new(document.get("format", ""))
    edits.add_region(
        Region(
            id="r-syndel1",
            name="delete: +x half",
            kind="box",
            params=dict(EDIT_DELETE_BOX),
            mix=0.0,
            op="delete",
        )
    )
    edits.add_region(
        Region(
            id="r-synmix2",
            name="blend to splat: -x sphere",
            kind="sphere",
            params=dict(EDIT_BLEND_SPHERE),
            mix=0.0,
            op="blend",
        )
    )
    return edits.save(bundle_dir / "edits.json")


def write_shade_sidecar(bundle_dir: Path, scene_root: Path) -> Path:
    """Write `shade_views.json` so the viewer's shade finder has frames to test.

    The cameras and each frame's median observed depth come from
    `trippy.train.prune.build_shade_region` -- the audit's own function, not a
    re-derivation -- so a selection the viewer makes from this file is the
    selection `trippy.edit.shade_finder.find_shade_pointset` makes from the same
    COLMAP model at the same thresholds. See `SHADE_ZNEAR_FRAC` for the one
    deliberate deviation from the shipped defaults, and why.

    Args:
        bundle_dir: the bundle directory.
        scene_root: the synthetic scene (holding `sparse_txt/`).

    Returns:
        The path written.
    """
    sparse_dir = scene_root / "sparse_txt"
    frames = sorted(p.name for p in (scene_root / "images").glob("*.jpg"))
    views = prune.build_shade_region(sparse_dir, frames, SHADE_ZNEAR_FRAC, SHADE_ZFAR_FRAC)
    return write_shade_views(bundle_dir, views, SHADE_ZNEAR_FRAC, SHADE_ZFAR_FRAC)


def build(out: Path, num_splats: int, splat_size: float, seed: int, epochs: int) -> Path:
    """Build the whole fixture under `out` and return the bundle directory.

    Args:
        out: workspace directory; wiped and recreated.
        num_splats: Gaussians in the PLY.
        splat_size: world-unit Gaussian radius.
        seed: RNG seed for the PLY.
        epochs: training epochs (2 is enough for a bundle that renders).

    Returns:
        The bundle directory (holding `bundle.json`).
    """
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print(f"▶ synthetic scene under {out}")
    scene_root, _point_set = build_synthetic_scene(out)

    ply_path = out / "splat.ply"
    started = time.monotonic()
    write_splat_ply(ply_path, num_splats, splat_size, seed)
    print(
        f"▶ wrote {ply_path} ({num_splats} Gaussians, size {splat_size}, "
        f"{ply_path.stat().st_size / 1024:.0f} KiB) in {time.monotonic() - started:.2f} s"
    )

    cfg = tiny_train_config(
        scene_root,
        ply_path,
        out / "run",
        out / "cache",
        epochs=epochs,
        # `size_mode="scale"` would give the TRIPS points the PLY's own large
        # radii; "knn" keeps them at the spacing the rest of trippy trains with,
        # so the splat half and the TRIPS half are visibly different renders of
        # the same geometry -- which is exactly what the Blend panel is for.
        point_source=PointSourceConfig(
            type="gaussian", path=str(ply_path), min_opacity=0.0, size_mode="knn"
        ),
    )
    print(f"▶ training {epochs} epoch(s) on CPU")
    trainer = Trainer(cfg)
    checkpoint = trainer.save_checkpoint()

    bundle_dir, _document = export_bundle(checkpoint, out / "bundle", name="synthetic-splat")
    document = json.loads((Path(bundle_dir) / BUNDLE_JSON_FILENAME).read_text())
    blend = document.get("blend")
    if not blend or not blend.get("splat_ply"):
        raise SystemExit(
            "export-bundle wrote no blend.splat_ply; the live splat path has nothing to open"
        )
    edits_path = write_edits(Path(bundle_dir), document)
    shade_path = write_shade_sidecar(Path(bundle_dir), Path(scene_root))
    print(f"✓ bundle {bundle_dir}")
    print(f"  blend.splat_ply = {blend['splat_ply']}")
    print(f"  views = {len(document['views'])}, default_view = {document['default_view']}")
    print(f"  edits = {edits_path} (2 regions: one delete box, one blend sphere)")
    print(f"  shade frames = {shade_path}")
    return Path(bundle_dir)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="workspace directory (wiped)")
    parser.add_argument("--splats", type=int, default=DEFAULT_NUM_SPLATS)
    parser.add_argument("--splat-size", type=float, default=DEFAULT_SPLAT_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    bundle = build(Path(args.out), args.splats, args.splat_size, args.seed, args.epochs)
    # Last line, machine-readable: the check script reads it.
    print(f"BUNDLE {bundle}")


if __name__ == "__main__":
    main()
