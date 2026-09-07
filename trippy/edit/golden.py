"""The Python/Rust golden fixture: one `edits.json`, two implementations, one answer.

Module: trippy.edit.golden
Purpose: the E1/E2 acceptance check that `trippy apply-edits` and the Rust
    viewer compute the SAME per-point weights from the same `edits.json`
    (docs/EDITOR.md Sec 2, Sec 6). This module owns the synthetic fixture both
    sides read: a seeded point cloud, a seeded Gaussian cloud, an
    `EditDocument` exercising every region kind and every op, the weights
    `trippy.edit.weights` composes from them, a shade-frame sidecar plus
    the selection `trippy.train.prune`'s own rule makes from it, and (E4)
    a structured click scene plus the `pointset` `trippy.edit.cluster.
    click_to_cluster` grows from three clicks in it.
Invariants:
    - SYNTHETIC ONLY. Every array comes from a seeded
      `numpy.random.Generator`; nothing here reads a photograph, a
      checkpoint, a PLY or a COLMAP model of Jordan's scenes (AGENTS.md
      section 6).
    - The fixture is written with plain Python floats, whose `json` repr is
      the shortest round-tripping decimal, so `serde_json` parses back the
      IDENTICAL f64. The 1e-6 tolerance the two tests hold to is therefore
      measuring the two IMPLEMENTATIONS, never the file format.
    - This module only WRITES the fixture and re-checks it; it is not part of
      any render or publish path. `tests/test_edit_golden.py` regenerates it
      in a temp directory and diffs against the committed copy, so a drift in
      `trippy.edit.weights` fails the Python suite before it can silently
      re-bless the Rust one.
    - The click fixture passes `max_radius` EXPLICITLY and never calls
      `trippy.edit.cluster.default_max_radius_from_bundle`: that default runs
      `median_nn_distance`, which draws a seeded numpy subsample when a cloud
      is larger than `SUMMARY_NN_SAMPLE` and so has no portable Rust twin.
      Pinning the number in the file keeps the parity test measuring the
      clustering, not two random-number generators.
    - `write_shade_views` is the "run once per bundle" precompute
      docs/EDITOR.md Sec 3 describes: it records each shade frame's camera and
      its median COLMAP-observed depth `d`, which is the one input the
      viewer's sliders cannot re-derive. Everything the sliders DO move
      (znear/zfar fractions, luminance, confidence) is recomputed live in Rust
      from these numbers.
Units: world units (COLMAP world frame, docs/GEOMETRY.md); pixels for
    intrinsics; weights, luminance and confidence are dimensionless in [0, 1].
Related docs: docs/EDITOR.md Sec 2 and Sec 6; rust/crates/trips-viewer/src/edit/;
    tests/test_edit_golden.py.

Usage:
    PYTHONPATH=. python -m trippy.edit.golden            # rewrite the committed fixture
    PYTHONPATH=. python -m trippy.edit.golden --out DIR  # write it somewhere else
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from trippy.constants import (
    REC709_LUMA_WEIGHTS,
    SHADE_PRUNE_DEFAULT_CONF_THRESHOLD,
    SHADE_PRUNE_DEFAULT_LUM_THRESHOLD,
    SHADE_PRUNE_DEFAULT_ZFAR_FRAC,
    SHADE_PRUNE_DEFAULT_ZNEAR_FRAC,
)
from trippy.edit.cluster import CameraView, click_to_cluster
from trippy.edit.model import EditDocument, Region
from trippy.edit.weights import compose_gaussian_weights, compose_trips_weights
from trippy.render.bundle import BUNDLE_FORMAT
from trippy.train import prune

__all__ = [
    "CLICK_FIXTURE_FORMAT",
    "GOLDEN_FIXTURE_DIR",
    "SHADE_VIEWS_FORMAT",
    "build_click_fixture",
    "build_golden_fixture",
    "write_golden_fixture",
    "write_shade_views",
]

#: The committed fixture, relative to the repository root.
GOLDEN_FIXTURE_DIR = Path("tests/fixtures/synthetic/edit_golden")

#: `"format"` of the shade-frame sidecar, matching
#: `trips_viewer::edit::shade::SHADE_VIEWS_FORMAT`.
SHADE_VIEWS_FORMAT = "trippy-shade-views-1"

#: Filename of that sidecar next to `bundle.json`, matching
#: `trips_viewer::edit::shade::SHADE_VIEWS_FILENAME`.
SHADE_VIEWS_FILENAME = "shade_views.json"

#: Points in the synthetic TRIPS cloud. Small enough to commit, large enough
#: that every region below selects a non-trivial, non-identical subset.
_NUM_POINTS = 400

#: Gaussians in the synthetic splat cloud (a DIFFERENT row order, which is the
#: whole point of `compose_gaussian_weights` skipping `pointset` regions).
_NUM_GAUSSIANS = 150

_SEED = 20260907

#: `"format"` of the click fixture, matching
#: `trips_viewer::edit::cluster::CLICK_FIXTURE_FORMAT`.
CLICK_FIXTURE_FORMAT = "trippy-edit-click-1"

#: Seed for the click scene. Deliberately NOT `_SEED`: the click fixture is a
#: structured scene (three blobs), not the uniform cloud the weight/shade
#: fixtures share, and reusing the seed would only invite the two to be
#: confused for one another.
_CLICK_SEED = 20260908

#: Points per blob in the click scene, and in the scatter around them.
_CLICK_BLOB_POINTS = 100
_CLICK_SCATTER_POINTS = 120


def _floats(array: np.ndarray) -> list[float]:
    """Flatten to a list of plain Python floats (exact JSON round-trip)."""
    return [float(v) for v in np.asarray(array, dtype=np.float64).reshape(-1)]


def _synthetic_clouds() -> dict[str, np.ndarray]:
    """The seeded point/Gaussian clouds both sides read.

    Returns:
        `{"xyz", "rgb", "conf", "gaussian_xyz"}`. `xyz` spans
        `[-2, 2] x [-1.5, 1.5] x [1, 8]`, the same box
        `tests.test_train_helpers.synthetic_point_set` uses, so the fixture and
        the synthetic bundle describe the same kind of world.
    """
    rng = np.random.default_rng(_SEED)
    xyz = np.stack(
        [
            rng.uniform(-2.0, 2.0, _NUM_POINTS),
            rng.uniform(-1.5, 1.5, _NUM_POINTS),
            rng.uniform(1.0, 8.0, _NUM_POINTS),
        ],
        axis=1,
    )
    # A third of the cloud is deliberately dark and low-confidence, so the
    # shade finder has something to find that the thresholds can move.
    rgb = rng.uniform(0.0, 1.0, (_NUM_POINTS, 3))
    dark = rng.random(_NUM_POINTS) < 0.35
    rgb[dark] *= 0.15
    conf = rng.uniform(0.05, 0.99, _NUM_POINTS)
    conf[dark] *= 0.4
    gaussian_xyz = np.stack(
        [
            rng.uniform(-2.0, 2.0, _NUM_GAUSSIANS),
            rng.uniform(-1.5, 1.5, _NUM_GAUSSIANS),
            rng.uniform(1.0, 8.0, _NUM_GAUSSIANS),
        ],
        axis=1,
    )
    # Round-trip through float32 exactly as a bundle's own `points.npz` would,
    # so the fixture holds values a real cloud could hold.
    return {
        "xyz": xyz.astype(np.float32).astype(np.float64),
        "rgb": rgb.astype(np.float32).astype(np.float64),
        "conf": conf.astype(np.float32).astype(np.float64),
        "gaussian_xyz": gaussian_xyz.astype(np.float32).astype(np.float64),
    }


def _golden_edits() -> EditDocument:
    """An `EditDocument` exercising every kind, every op, ordering and undo.

    Built through the public mutation API only, so the `undo_stack` it carries
    is a real history (including one undone entry after the cursor) rather than
    a hand-written log -- the Rust side replays exactly this.
    """
    edits = EditDocument.new(BUNDLE_FORMAT)
    # 1. A rotated box that blends towards the splat.
    edits.add_region(
        Region(
            id="r-box0001",
            name="rotated box",
            kind="box",
            params={
                "center": [0.0, 0.0, 4.0],
                "half_extents": [1.5, 0.8, 1.2],
                "quat": [0.9238795325112867, 0.0, 0.0, 0.3826834323650898],
            },
            mix=0.25,
            op="blend",
        )
    )
    # 2. A sphere that fades what the box already touched (fade multiplies).
    edits.add_region(
        Region(
            id="r-sph0002",
            name="fading sphere",
            kind="sphere",
            params={"center": [0.5, 0.2, 3.5], "radius": 1.1},
            mix=0.5,
            op="fade",
        )
    )
    # 3. A graded lid, so `falloff`/`band` are exercised on the blend path.
    edits.add_region(
        Region(
            id="r-lid0003",
            name="graded lid",
            kind="lid",
            params={
                "up": [0.0, -1.0, 0.0],
                "height": -0.5,
                "center": [0.0, 0.0, 5.0],
                "radius": 1.6,
                "falloff": 0.8,
                "band": 0.9,
            },
            mix=0.0,
            op="fade",
        )
    )
    # 4. A pointset the Gaussian half must ignore entirely.
    edits.add_region(
        Region(
            id="r-pts0004",
            name="shade cloud",
            kind="pointset",
            params={"point_ids": list(range(0, _NUM_POINTS, 17))},
            mix=0.0,
            op="fade",
        )
    )
    # 5. A hard delete LAST in paint order, so it overrides the blends above it
    #    (docs/EDITOR.md Sec 1 "Ordering and overlap").
    edits.add_region(
        Region(
            id="r-del0005",
            name="delete box",
            kind="box",
            params={"center": [-1.2, 0.0, 6.0], "half_extents": [0.9, 1.5, 1.4]},
            mix=0.0,
            op="delete",
        )
    )
    # 6. A disabled region: present in the file, invisible to composition.
    edits.add_region(
        Region(
            id="r-off0006",
            name="disabled sphere",
            kind="sphere",
            params={"center": [0.0, 0.0, 4.0], "radius": 10.0},
            mix=1.0,
            op="blend",
        )
    )
    edits.update_region("r-off0006", enabled=False)
    # 7. Reorder, then a trailing edit that is UNDONE -- the saved cursor sits
    #    before it, so both sides must reproduce "the state before the redo".
    edits.reorder(
        ["r-box0001", "r-sph0002", "r-lid0003", "r-pts0004", "r-off0006", "r-del0005"]
    )
    edits.update_region("r-box0001", mix=1.0)
    edits.undo()
    return edits


def _shade_views() -> list[prune.ShadeView]:
    """Three synthetic shade frames looking into the cloud from different places.

    Built directly rather than through `prune.build_shade_region` because that
    function needs a COLMAP model on disk; the six numbers it would derive
    (`R`, `C`, `d`, intrinsics, size) are set here explicitly so the fixture has
    no external dependency. Everything downstream -- `prune.in_region`,
    `prune.luminance`, `prune.dark_mass_stats` -- is the real thing.
    """

    def yaw(radians: float) -> np.ndarray:
        c, s = float(np.cos(radians)), float(np.sin(radians))
        # World->camera rotation about the scene's up axis (+Y is down here).
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)

    views: list[prune.ShadeView] = []
    for i, (angle, centre, d) in enumerate(
        [
            (0.0, (0.0, 0.0, 0.0), 4.0),
            (0.35, (-1.5, 0.2, 0.5), 5.0),
            (-0.35, (1.5, -0.2, 0.5), 3.5),
        ]
    ):
        views.append(
            prune.ShadeView(
                name=f"SYN_{i:04d}.jpg",
                R=yaw(angle),
                C=np.array(centre, dtype=np.float64),
                d=float(d),
                fx=320.0,
                fy=320.0,
                cx=160.0,
                cy=120.0,
                width=320,
                height=240,
                nobs=64,
                znear=SHADE_PRUNE_DEFAULT_ZNEAR_FRAC * d,
                zfar=SHADE_PRUNE_DEFAULT_ZFAR_FRAC * d,
            )
        )
    return views


def _view_to_json(view: prune.ShadeView) -> dict[str, Any]:
    """One `views[]` entry of `shade_views.json` (`trips_viewer::edit::shade::ShadeView`)."""
    return {
        "name": view.name,
        "r": _floats(view.R),
        "c": _floats(view.C),
        "d": float(view.d),
        "fx": float(view.fx),
        "fy": float(view.fy),
        "cx": float(view.cx),
        "cy": float(view.cy),
        "width": float(view.width),
        "height": float(view.height),
        "nobs": int(view.nobs),
    }


def write_shade_views(
    out_dir: str | Path,
    views: list[prune.ShadeView],
    znear_frac: float = SHADE_PRUNE_DEFAULT_ZNEAR_FRAC,
    zfar_frac: float = SHADE_PRUNE_DEFAULT_ZFAR_FRAC,
) -> Path:
    """Write the viewer's shade-frame sidecar next to a `bundle.json`.

    The one input the shade-finder sliders cannot re-derive is each frame's
    median COLMAP-observed depth `d`; this records it (with the camera) so the
    viewer can recompute `znear = znear_frac * d` live. The fractions are stored
    too, purely as the panel's starting slider positions.

    Args:
        out_dir: the bundle directory.
        views: from `trippy.train.prune.build_shade_region` (or built by hand,
            as this module's own fixture does).
        znear_frac, zfar_frac: the fractions the panel opens on.

    Returns:
        The path written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SHADE_VIEWS_FILENAME
    doc = {
        "format": SHADE_VIEWS_FORMAT,
        "znear_frac": float(znear_frac),
        "zfar_frac": float(zfar_frac),
        "views": [_view_to_json(v) for v in views],
    }
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


# --- the click-to-cluster fixture (E4) -------------------------------------------------


def _click_camera() -> CameraView:
    """The one camera the click cases are made in.

    Identity rotation with the centre at the world origin, so world and camera
    coordinates coincide and every number below can be read straight off the
    scene: `u = 500 * x / z + 320`. The same frontal camera
    `tests/test_edit_cluster.py` uses, for the same reason.
    """
    return CameraView(
        R=np.eye(3),
        t=np.zeros(3),
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
        name="SYN_CLICK.jpg",
    )


def _click_scene() -> tuple[np.ndarray, np.ndarray]:
    """A structured cloud that makes every gate in `click_to_cluster` matter.

    Four groups, all seeded:

    1. **front** -- a red blob at `z = 5` centred on the optical axis. This is
       what a click at the principal point is meant to select.
    2. **behind** -- the SAME red, at `z = 7`. It projects into the same pixels
       and passes the colour gate, so only the depth-mode seed (step 3 of
       `trippy.edit.cluster`'s docstring) keeps it out of the seed, and only
       the k-NN growth's own locality keeps it out of the region.
    3. **beside** -- green, at `z = 5`, offset `+0.22` in `x` so it is
       geometrically contiguous with **front** but 22 px away from the click.
       Only the colour gate separates the two.
    4. **scatter** -- uniform noise filling the volume, so the k-NN queries
       have a non-trivial tree to walk rather than four tight clusters.

    Returns:
        `(xyz, rgb)`, both `(N, 3)` float64 already round-tripped through
        float32 exactly as a bundle's `points.npz` would store them.
    """
    rng = np.random.default_rng(_CLICK_SEED)

    def blob(centre: tuple[float, float, float], colour: tuple[float, float, float]):
        xyz = rng.normal(scale=0.06, size=(_CLICK_BLOB_POINTS, 3)) + np.asarray(centre)
        rgb = np.clip(
            np.asarray(colour) + rng.normal(scale=0.01, size=(_CLICK_BLOB_POINTS, 3)),
            0.0,
            1.0,
        )
        return xyz, rgb

    front_xyz, front_rgb = blob((0.0, 0.0, 5.0), (0.85, 0.15, 0.15))
    behind_xyz, behind_rgb = blob((0.0, 0.0, 7.0), (0.85, 0.15, 0.15))
    beside_xyz, beside_rgb = blob((0.25, 0.0, 5.0), (0.15, 0.80, 0.20))
    scatter_xyz = np.stack(
        [
            rng.uniform(-2.0, 2.0, _CLICK_SCATTER_POINTS),
            rng.uniform(-1.5, 1.5, _CLICK_SCATTER_POINTS),
            rng.uniform(3.0, 12.0, _CLICK_SCATTER_POINTS),
        ],
        axis=1,
    )
    scatter_rgb = rng.uniform(0.0, 1.0, (_CLICK_SCATTER_POINTS, 3))

    xyz = np.vstack([front_xyz, behind_xyz, beside_xyz, scatter_xyz])
    rgb = np.vstack([front_rgb, behind_rgb, beside_rgb, scatter_rgb])
    return (
        xyz.astype(np.float32).astype(np.float64),
        rgb.astype(np.float32).astype(np.float64),
    )


def _click_cases() -> list[dict[str, Any]]:
    """The clicks both implementations replay, and the parameters for each.

    Three, each pinning a different branch:

    - `default` -- a hit on the front blob with every gate live. `max_radius`
      is passed EXPLICITLY (never `default_max_radius_from_bundle`, whose
      `median_nn_distance` draws a seeded numpy subsample that has no portable
      twin), and `depth_gap_factor` is small enough that the blob at `z = 7`
      is its own depth mode rather than part of the seed.
    - `capped` -- the same click with a small `max_points`, so the
      "stop mid-frontier" branch and its index order are pinned too.
    - `loose_colour` -- the same click with the colour gate opened, so the
      growth crosses into the neighbouring blob and both sides have to agree
      on a much longer flood fill.
    - `miss` -- a click where nothing projects, which must return an EMPTY
      selection and a warning rather than an error.
    """
    return [
        {
            "name": "default",
            "px": [320.0, 240.0],
            "radius_px": 12.0,
            "colour_tol": 0.15,
            "max_radius": 3.0,
            "max_points": 200000,
            "knn_k": 16,
            "depth_gap_factor": 0.3,
        },
        {
            "name": "capped",
            "px": [320.0, 240.0],
            "radius_px": 12.0,
            "colour_tol": 0.15,
            "max_radius": 3.0,
            # Above the seed's own size and below what the growth would reach,
            # so the cap really does stop the flood fill part-way through a
            # frontier -- the one branch whose answer depends on the order
            # neighbours are visited in.
            "max_points": 94,
            "knn_k": 16,
            "depth_gap_factor": 0.3,
        },
        {
            "name": "loose_colour",
            "px": [320.0, 240.0],
            "radius_px": 12.0,
            # Wide enough to admit any colour: the growth then crosses into the
            # green blob beside the red one, which the default case must not.
            "colour_tol": 2.0,
            "max_radius": 3.0,
            "max_points": 200000,
            "knn_k": 16,
            "depth_gap_factor": 0.3,
        },
        {
            "name": "miss",
            "px": [8.0, 8.0],
            "radius_px": 4.0,
            "colour_tol": 0.15,
            "max_radius": 3.0,
            "max_points": 200000,
            "knn_k": 16,
            "depth_gap_factor": 0.3,
        },
    ]


def build_click_fixture() -> dict[str, dict[str, Any]]:
    """The click-to-cluster half of the fixture: the scene, the clicks, the answers.

    `click.json` is the INPUT both sides read (scene, camera, clicks,
    parameters); `expected_click.json` is what `trippy.edit.cluster.
    click_to_cluster` returns for each of them, which
    `trips_viewer::edit::cluster` must reproduce exactly.

    Returns:
        `{"click.json": ..., "expected_click.json": ...}`.
    """
    xyz, rgb = _click_scene()
    camera = _click_camera()
    cases = _click_cases()

    results = []
    for case in cases:
        region, summary = click_to_cluster(
            xyz,
            rgb,
            camera,
            (case["px"][0], case["px"][1]),
            radius_px=case["radius_px"],
            colour_tol=case["colour_tol"],
            max_radius=case["max_radius"],
            max_points=case["max_points"],
            knn_k=case["knn_k"],
            depth_gap_factor=case["depth_gap_factor"],
            region_id=f"r-click{len(results):03d}",
            name=f"click {case['name']}",
        )
        results.append(
            {
                "name": case["name"],
                "point_ids": [int(i) for i in region.params["point_ids"]],
                "n_candidates": int(summary["n_candidates"]),
                "n_seed": int(summary["n_seed"]),
                "n_selected": int(summary["n_selected"]),
                "hit_max_points": bool(summary["hit_max_points"]),
                "seed_depth_mean": (
                    float(summary["seed_depth_mean"]) if "seed_depth_mean" in summary else None
                ),
                "warning": summary.get("warning"),
            }
        )

    return {
        "click.json": {
            "format": CLICK_FIXTURE_FORMAT,
            "n": int(xyz.shape[0]),
            "xyz": _floats(xyz),
            "rgb": _floats(rgb),
            "camera": {
                "name": camera.name,
                "r": _floats(camera.R),
                "t": _floats(camera.t),
                "fx": float(camera.fx),
                "fy": float(camera.fy),
                "cx": float(camera.cx),
                "cy": float(camera.cy),
                "width": float(camera.width),
                "height": float(camera.height),
            },
            "cases": cases,
        },
        "expected_click.json": {
            "format": CLICK_FIXTURE_FORMAT,
            "cases": results,
        },
    }


def build_golden_fixture() -> dict[str, dict[str, Any]]:
    """Compute every file of the fixture, without writing anything.

    Returns:
        `{filename: json-serialisable document}` for `points.json`,
        `edits.json`, `expected_weights.json`, `shade_views.json`,
        `expected_shade.json`, `click.json` and `expected_click.json`.
    """
    clouds = _synthetic_clouds()
    edits = _golden_edits()

    trips = compose_trips_weights(edits, clouds["xyz"])
    gaussian = compose_gaussian_weights(edits, clouds["gaussian_xyz"])

    views = _shade_views()
    inside, _zfrac = prune.in_region(views, clouds["xyz"])
    inside = inside & np.isfinite(clouds["xyz"]).all(axis=1)
    lum = prune.luminance(clouds["rgb"])
    conf_drop = prune.confidence_drop_mask(
        clouds["conf"], SHADE_PRUNE_DEFAULT_CONF_THRESHOLD, mode="absolute"
    )
    selected = inside & (lum < SHADE_PRUNE_DEFAULT_LUM_THRESHOLD) & conf_drop
    stats = prune.dark_mass_stats(
        views, clouds["xyz"], clouds["rgb"], clouds["conf"], SHADE_PRUNE_DEFAULT_LUM_THRESHOLD
    )

    return {
        "points.json": {
            "n": int(clouds["xyz"].shape[0]),
            "n_gaussians": int(clouds["gaussian_xyz"].shape[0]),
            "xyz": _floats(clouds["xyz"]),
            "rgb": _floats(clouds["rgb"]),
            "conf": _floats(clouds["conf"]),
            "gaussian_xyz": _floats(clouds["gaussian_xyz"]),
        },
        "edits.json": edits.to_json(),
        "expected_weights.json": {
            "weight": _floats(trips.weight),
            "delete_mask": [bool(v) for v in trips.delete_mask],
            "gaussian_weight": _floats(gaussian.weight),
            "gaussian_delete_mask": [bool(v) for v in gaussian.delete_mask],
        },
        "shade_views.json": {
            "format": SHADE_VIEWS_FORMAT,
            "znear_frac": float(SHADE_PRUNE_DEFAULT_ZNEAR_FRAC),
            "zfar_frac": float(SHADE_PRUNE_DEFAULT_ZFAR_FRAC),
            "views": [_view_to_json(v) for v in views],
        },
        "expected_shade.json": {
            "thresholds": {
                "znear_frac": float(SHADE_PRUNE_DEFAULT_ZNEAR_FRAC),
                "zfar_frac": float(SHADE_PRUNE_DEFAULT_ZFAR_FRAC),
                "lum_threshold": float(SHADE_PRUNE_DEFAULT_LUM_THRESHOLD),
                "conf_threshold": float(SHADE_PRUNE_DEFAULT_CONF_THRESHOLD),
            },
            "luma_weights": [float(w) for w in REC709_LUMA_WEIGHTS],
            "point_ids": [int(i) for i in np.flatnonzero(selected)],
            "n_in_region": int(stats["n_in_region"]),
            "mass_in_region": float(stats["mass_in_region"]),
            "dark_mass_fraction": float(stats["dark_mass_fraction"]),
        },
        **build_click_fixture(),
    }


def write_golden_fixture(out_dir: str | Path) -> list[Path]:
    """Write every file of `build_golden_fixture()` into `out_dir`.

    Args:
        out_dir: destination directory, created if missing.

    Returns:
        The paths written, in a stable order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, doc in build_golden_fixture().items():
        path = out_dir / name
        path.write_text(json.dumps(doc, indent=2) + "\n")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    """Rewrite the committed fixture (or one in `--out`)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"destination directory (default: {GOLDEN_FIXTURE_DIR})",
    )
    args = parser.parse_args(argv)
    out = args.out or (Path(__file__).resolve().parents[2] / GOLDEN_FIXTURE_DIR)
    for path in write_golden_fixture(out):
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover -- CLI entry point
    raise SystemExit(main())
