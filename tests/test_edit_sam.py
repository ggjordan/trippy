"""Tests for trippy.edit.sam_lift / sam_runner: the E5 SAM-3 mask lift, on CPU.

Module: tests.test_edit_sam
Invariants under test:
  - The depth gate rejects points BEHIND the segmented surface: a mask that
    covers both a near object and the wall visible past it lifts onto the
    object only (docs/EDITOR.md Sec 3 "4. SAM 3 lift (E5)").
  - Projection agrees with an independent, from-scratch pinhole projection
    written in this file (never `sam_lift`'s own helper), including the
    bundle-view -> photograph pixel scale.
  - Majority voting across views drops a distractor one view's mask
    happened to include, and keeps what every view agrees on.
  - The neighbour prompt really is the selection's centroid seen in that
    view: the fake segmenter asserts the point it is handed lands inside
    the object's own footprint there.
  - `trippy edits sam` writes a valid `pointset` region into `edits.json`
    and prints a summary; `--mask NAME=PATH` records `segmenter` as
    "mask-file", never claiming SAM 3 ran.
  - The `--preview` PNG is a from-scratch heatmap: every pixel lies on the
    ramp between the two configured colours, so it cannot contain
    photographic content (AGENTS.md Sec 6).
  - `sam3_command` names Splats' SAM venv python, the SAM 3 checkout and
    the local weights -- the exact command docs/EDITOR.md documents.
Fixture: a synthetic bundle + synthetic noise "photographs" built here.
    SAM 3 itself never runs in this suite (it is GPU work, and 50 s per
    view on CPU); a FAKE segmenter is injected everywhere, which is why
    every rule above is testable without the queue.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trippy.constants import (
    SAM_LIFT_PREVIEW_HIGH_RGB,
    SAM_LIFT_PREVIEW_LOW_RGB,
)
from trippy.edit.model import EditDocument
from trippy.edit.sam_lift import (
    LiftView,
    depth_mode_keep,
    lift_mask_in_view,
    neighbour_views,
    photo_scale,
    project_to_photo,
    sam_lift,
    write_selection_preview,
)
from trippy.edit.sam_runner import MaskFileSegmenter, SamPrompt, sam3_command

# --- the synthetic scene ---------------------------------------------------
# One 100x100 bundle view per camera, photographs at 200x200 (scale 2.0), a
# 0.8 x 0.8 object slab at z = 2 and a wide wall at z = 10 directly behind it.
VIEW_W, VIEW_H = 100, 100
PHOTO_W, PHOTO_H = 200, 200
FX = FY = 100.0
CX = CY = 50.0
OBJECT_Z = 2.0
WALL_Z = 10.0


def _view(name: str, camera_x: float, index: int) -> dict[str, Any]:
    """A `bundle.json` view looking down +Z from `(camera_x, 0, 0)`.

    Identity rotation, so `t = -C` and world-to-camera is a pure shift.
    """
    return {
        "index": index,
        "name": name,
        "width": VIEW_W,
        "height": VIEW_H,
        "fx": FX,
        "fy": FY,
        "cx": CX,
        "cy": CY,
        "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "t": [-camera_x, 0.0, 0.0],
        "distortion": [0.0] * 8,
    }


def build_fixture(tmp_path: Path, camera_xs: tuple[float, ...] = (0.0, 0.15, -0.15)) -> dict[str, Any]:
    """Write a synthetic bundle + synthetic photographs; return the pieces.

    Returns a dict with `bundle_dir`, `scene_root`, `names`, `xyz`,
    `object_ids` (rows of the near slab), `wall_ids` (rows of the wall),
    and `distractor_ids` (a second near slab off to the side).
    """
    rng = np.random.default_rng(11)

    grid = np.linspace(-0.4, 0.4, 31)
    ox, oy = np.meshgrid(grid, grid, indexing="xy")
    obj = np.stack([ox.ravel(), oy.ravel(), np.full(ox.size, OBJECT_Z)], axis=1)

    wall_grid = np.linspace(-3.0, 3.0, 41)
    wx, wy = np.meshgrid(wall_grid, wall_grid, indexing="xy")
    wall = np.stack([wx.ravel(), wy.ravel(), np.full(wx.size, WALL_Z)], axis=1)

    dist_grid = np.linspace(0.6, 0.9, 11)
    dx, dy = np.meshgrid(dist_grid, np.linspace(-0.4, 0.4, 11), indexing="xy")
    distractor = np.stack([dx.ravel(), dy.ravel(), np.full(dx.size, OBJECT_Z)], axis=1)

    xyz = np.concatenate([obj, wall, distractor], axis=0).astype(np.float64)
    n_obj, n_wall, n_dist = len(obj), len(wall), len(distractor)
    object_ids = np.arange(n_obj)
    wall_ids = np.arange(n_obj, n_obj + n_wall)
    distractor_ids = np.arange(n_obj + n_wall, n_obj + n_wall + n_dist)

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    scene_root = tmp_path / "scene"
    names = [f"IMG_{i:04d}.png" for i in range(len(camera_xs))]
    doc = {
        "format": "trippy-bundle-1",
        "name": "sam-lift-fixture",
        "points": "points.npz",
        "weights": "weights.safetensors",
        "num_points": int(xyz.shape[0]),
        "num_channels": 4,
        "background": [0.0, 0.0, 0.0, 0.0],
        "params": {"mode": "broadcast", "num_layers": 2},
        "up": [0.0, -1.0, 0.0],
        "default_view": 0,
        # `trippy.render.bundle.bundle_document` writes this so a tool that has
        # a bundle does not also have to be told where the photographs are.
        "scene_root": str(scene_root),
        "views": [_view(name, x, i) for i, (name, x) in enumerate(zip(names, camera_xs, strict=True))],
    }
    (bundle_dir / "bundle.json").write_text(json.dumps(doc))
    np.savez(
        bundle_dir / "points.npz",
        xyz=xyz.astype(np.float32),
        feat=rng.random((xyz.shape[0], 4), dtype=np.float32),
        conf=np.full(xyz.shape[0], 0.5, dtype=np.float32),
    )

    images = scene_root / "images"
    images.mkdir(parents=True)
    from PIL import Image

    for name in names:
        noise = rng.integers(0, 255, (PHOTO_H, PHOTO_W, 3), dtype=np.uint8)
        Image.fromarray(noise, mode="RGB").save(images / name)

    return {
        "bundle_dir": bundle_dir,
        "scene_root": scene_root,
        "names": names,
        "views": [LiftView.from_json(v, i) for i, v in enumerate(doc["views"])],
        "xyz": xyz,
        "object_ids": object_ids,
        "wall_ids": wall_ids,
        "distractor_ids": distractor_ids,
    }


def last_json_line(stdout: str) -> dict[str, Any]:
    """`trippy edits sam`'s output contract: the LAST stdout line is the summary.

    Everything before it is a `sam: ...` progress line, which is what
    `trips_viewer::sam_child` streams into the SAM panel while the child
    runs (docs/EDITOR.md Sec 3 "4. SAM 3 lift (E5)").
    """
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert lines, "no output at all"
    return json.loads(lines[-1])


def brute_force_project(view: LiftView, xyz: np.ndarray, scale: float) -> np.ndarray:
    """Pinhole projection written from scratch here (no sam_lift helper).

    `x_cam = R @ x_world + t`, `u = fx * x/z + cx`, then the uniform
    view->photo scale. Deliberately a plain Python loop: an independent
    second implementation, per AGENTS.md Sec 7's "implement transforms twice".
    """
    out = np.zeros((xyz.shape[0], 2), dtype=np.float64)
    for i, p in enumerate(xyz):
        cam = view.R @ p + view.t
        out[i, 0] = (view.fx * cam[0] / cam[2] + view.cx) * scale
        out[i, 1] = (view.fy * cam[1] / cam[2] + view.cy) * scale
    return out


@dataclass
class FakeSegmenter:
    """A stand-in for SAM 3: returns a rectangle mask per view name.

    Attributes:
        boxes: view file name -> `(x0, y0, x1, y1)` in photo pixels; the
            mask is True inside that rectangle.
        photo_wh: `(width, height)` every returned mask has.
        seen: every `(name, prompt)` this was called with, in order.
        require_prompt_inside: when True, a "point" prompt (the neighbour
            prompts, built from the selection's centroid) must land inside
            that view's own rectangle, or the test fails right here.
    """

    boxes: dict[str, tuple[float, float, float, float]]
    photo_wh: tuple[int, int] = (PHOTO_W, PHOTO_H)
    seen: list[tuple[str, SamPrompt]] = field(default_factory=list)
    require_prompt_inside: bool = True

    def __call__(self, image: Path, prompt: SamPrompt) -> tuple[np.ndarray, dict[str, Any]]:
        name = Path(image).name
        self.seen.append((name, prompt))
        x0, y0, x1, y1 = self.boxes[name]
        if prompt.kind == "point" and self.require_prompt_inside:
            u, v = prompt.point  # type: ignore[misc]
            assert x0 <= u <= x1 and y0 <= v <= y1, f"neighbour prompt {prompt.point} outside {name}"
        width, height = self.photo_wh
        mask = np.zeros((height, width), dtype=bool)
        mask[int(y0) : int(y1), int(x0) : int(x1)] = True
        return mask, {"segmenter": "fake", "mask_area_fraction": float(mask.mean())}


def object_box() -> tuple[float, float, float, float]:
    """The object slab's own footprint in photo pixels, in every view here.

    x in [-0.4, 0.4] at z = 2 with fx = 100, cx = 50 lands at view u in
    [30, 70], i.e. photo u in [60, 140] at scale 2 (the small camera
    offsets shift this by <= 15 view px, which the padding below covers).
    """
    return (55.0, 55.0, 145.0, 145.0)


# --- the depth gate --------------------------------------------------------


def test_depth_mode_keep_rejects_points_behind() -> None:
    # One cell, 30 points on an object at depth 2 and 60 on the wall behind
    # it at 10. The wall is the MAJORITY -- and must still be what gets
    # dropped: the gate takes the nearest SUPPORTED depth mode, not the
    # fullest one, because a mask over a near object nearly always contains
    # more background than object.
    u = np.zeros(90, dtype=np.int64)
    v = np.zeros(90, dtype=np.int64)
    depth = np.concatenate([np.full(30, 2.0), np.full(60, 10.0)])
    keep, surface = depth_mode_keep(u, v, depth, cell_px=16, tol=0.15)
    assert surface[0] == pytest.approx(2.0, rel=0.2)
    assert keep[:30].all()
    assert not keep[30:].any()


def test_depth_mode_keep_rejects_an_unsupported_floater() -> None:
    # A single stray point in front of a 40-point surface is not a surface:
    # 1 < 0.25 * 40, so the gate locks onto the real one behind it.
    u = np.zeros(41, dtype=np.int64)
    v = np.zeros(41, dtype=np.int64)
    depth = np.concatenate([[0.5], np.full(40, 4.0)])
    keep, surface = depth_mode_keep(u, v, depth, cell_px=16, tol=0.15)
    assert surface[0] == pytest.approx(4.0, rel=0.2)
    assert not keep[0]
    assert keep[1:].all()


def test_depth_mode_keep_is_per_cell() -> None:
    # Two cells, each with its own surface: 16 px cells means u = 0 and
    # u = 32 are different cells and get different modes.
    u = np.concatenate([np.zeros(20, dtype=np.int64), np.full(20, 32, dtype=np.int64)])
    v = np.zeros(40, dtype=np.int64)
    depth = np.concatenate([np.full(15, 3.0), np.full(5, 30.0), np.full(18, 12.0), np.full(2, 1.0)])
    keep, surface = depth_mode_keep(u, v, depth, cell_px=16, tol=0.15)
    assert surface[0] == pytest.approx(3.0, rel=0.2)
    assert surface[20] == pytest.approx(12.0, rel=0.2)
    assert keep[:15].all() and not keep[15:20].any()
    assert keep[20:38].all() and not keep[38:].any()


def test_depth_mode_keep_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="cell_px"):
        depth_mode_keep(np.zeros(1), np.zeros(1), np.ones(1), cell_px=0)
    with pytest.raises(ValueError, match="tol"):
        depth_mode_keep(np.zeros(1), np.zeros(1), np.ones(1), tol=0.0)
    with pytest.raises(ValueError, match="positive depths"):
        depth_mode_keep(np.zeros(1), np.zeros(1), np.zeros(1))


# --- projection ------------------------------------------------------------


def test_project_to_photo_matches_brute_force(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    view = fixture["views"][1]
    xyz = fixture["xyz"]
    scale = photo_scale(view, (PHOTO_W, PHOTO_H))
    assert scale == pytest.approx(2.0)

    uv, depth = project_to_photo(view, xyz, scale)
    expected = brute_force_project(view, xyz, scale)
    assert np.allclose(uv, expected, atol=1e-9)
    assert np.allclose(depth[fixture["object_ids"]], OBJECT_Z)
    assert np.allclose(depth[fixture["wall_ids"]], WALL_Z)


def test_photo_scale_rejects_mismatched_aspect(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    with pytest.raises(ValueError, match="aspect"):
        photo_scale(fixture["views"][0], (400, 200))


def test_lift_rejects_wrong_shaped_mask(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    with pytest.raises(ValueError, match="but the photo is"):
        lift_mask_in_view(
            fixture["views"][0], fixture["xyz"], np.ones((10, 10), dtype=bool), (PHOTO_W, PHOTO_H)
        )


# --- the lift ---------------------------------------------------------------


def test_lift_keeps_object_and_rejects_the_wall_behind_it(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    view = fixture["views"][0]
    mask = np.zeros((PHOTO_H, PHOTO_W), dtype=bool)
    x0, y0, x1, y1 = object_box()
    mask[int(y0) : int(y1), int(x0) : int(x1)] = True

    lifted = lift_mask_in_view(view, fixture["xyz"], mask, (PHOTO_W, PHOTO_H))
    selected = lifted["selected"]

    # The wall projects inside the same mask (it is directly behind), so the
    # ONLY thing separating the two is the depth gate.
    in_mask_wall = int(np.count_nonzero(lifted["visible"][fixture["wall_ids"]]))
    assert in_mask_wall > 0
    assert selected[fixture["object_ids"]].all()
    assert not selected[fixture["wall_ids"]].any()
    assert lifted["stats"]["n_in_mask"] > lifted["stats"]["n_selected"]


def test_sam_lift_single_view_selects_the_object(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    segmenter = FakeSegmenter(boxes={fixture["names"][0]: object_box()})
    region, summary = sam_lift(
        fixture["bundle_dir"],
        fixture["scene_root"],
        fixture["names"][0],
        SamPrompt(kind="box", box=object_box()),
        segmenter,
        views_around=0,
    )
    assert region.kind == "pointset"
    assert region.op == "fade"
    assert region.mix == 0.0
    assert sorted(region.params["point_ids"]) == fixture["object_ids"].tolist()
    assert summary["n_points"] == len(fixture["object_ids"])
    assert summary["views_used"] == [fixture["names"][0]]
    assert summary["per_view"][0]["segmenter"]["segmenter"] == "fake"
    assert len(segmenter.seen) == 1


def test_sam_lift_majority_vote_drops_a_one_view_distractor(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    names = fixture["names"]
    # View 0 and view 2 see the object only; view 1's mask is sloppy and also
    # covers the distractor slab beside it. 1 vote out of 3 must lose.
    wide = (55.0, 55.0, 200.0, 145.0)
    segmenter = FakeSegmenter(
        boxes={names[0]: object_box(), names[1]: wide, names[2]: object_box()}
    )
    region, summary = sam_lift(
        fixture["bundle_dir"],
        fixture["scene_root"],
        names[0],
        SamPrompt(kind="box", box=object_box()),
        segmenter,
        views_around=2,
    )
    ids = set(region.params["point_ids"])
    assert set(fixture["object_ids"].tolist()) <= ids
    assert not (set(fixture["distractor_ids"].tolist()) & ids)
    assert summary["vote"]["n_views"] == 3
    # The distractor DID get a vote; the vote is what discarded it.
    assert summary["vote"]["n_points_any_vote"] > summary["n_points"]
    # Neighbours are prompted with the centroid as a point prompt.
    assert [p.kind for _n, p in segmenter.seen] == ["box", "point", "point"]


def test_sam_lift_raises_when_nothing_is_selected(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    empty = FakeSegmenter(boxes={fixture["names"][0]: (1.0, 1.0, 2.0, 2.0)})
    with pytest.raises(ValueError, match="lifted onto 0 points"):
        sam_lift(
            fixture["bundle_dir"],
            fixture["scene_root"],
            fixture["names"][0],
            SamPrompt(kind="point", point=(1.5, 1.5)),
            empty,
            views_around=0,
        )


def test_sam_lift_rejects_an_unknown_view(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    with pytest.raises(ValueError, match="is not a view of"):
        sam_lift(
            fixture["bundle_dir"],
            fixture["scene_root"],
            "nope.png",
            SamPrompt(kind="text", text="a rock"),
            FakeSegmenter(boxes={}),
        )


def test_neighbour_views_skips_views_that_cannot_see_the_target(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    views = fixture["views"]
    # A point far off to the side is out of every frame -> no neighbour qualifies.
    assert neighbour_views(views, views[0], np.array([50.0, 0.0, OBJECT_Z]), 2) == []
    # The object's own centre is visible in both neighbours.
    chosen = neighbour_views(views, views[0], np.array([0.0, 0.0, OBJECT_Z]), 2)
    assert [v.name for v in chosen] == [fixture["names"][1], fixture["names"][2]]


# --- preview ---------------------------------------------------------------


def test_preview_is_a_from_scratch_heatmap(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    view = fixture["views"][0]
    uv, _depth = project_to_photo(view, fixture["xyz"], 2.0)
    selected = np.zeros(fixture["xyz"].shape[0], dtype=bool)
    selected[fixture["object_ids"]] = True

    from PIL import Image

    out = write_selection_preview(tmp_path / "preview.png", uv, selected, (PHOTO_W, PHOTO_H), cell_px=8)
    with Image.open(out) as handle:
        rgb = np.asarray(handle.convert("RGB"), dtype=np.float64)
    assert rgb.shape == (PHOTO_H // 8, PHOTO_W // 8, 3)

    # Every pixel is a point on the segment LOW -> HIGH: the image is a ramp
    # of counts and cannot carry photographic content.
    low = np.asarray(SAM_LIFT_PREVIEW_LOW_RGB, dtype=np.float64)
    high = np.asarray(SAM_LIFT_PREVIEW_HIGH_RGB, dtype=np.float64)
    alpha = (rgb[..., 1] - low[1]) / (high[1] - low[1])
    expected = low[None, None, :] + alpha[..., None] * (high - low)[None, None, :]
    assert np.abs(rgb - expected).max() <= 1.0
    assert alpha.max() > 0.0  # the selection is actually visible in it


# --- the runner's command, and the CLI --------------------------------------


def test_sam3_command_names_the_local_checkout_and_weights(tmp_path: Path) -> None:
    from trippy.constants import SAM3_REPO_DIR, SAM3_VENV_PYTHON, SAM3_WEIGHTS_PATH

    argv = sam3_command(
        tmp_path / "photo.jpg",
        SamPrompt(kind="box", box=(1.0, 2.0, 3.0, 4.0)),
        tmp_path / "mask.npy",
        tmp_path / "info.json",
        "cpu",
    )
    assert argv[0] == SAM3_VENV_PYTHON
    assert argv[1].endswith("trippy/edit/sam_runner.py")
    assert SAM3_REPO_DIR in argv
    assert SAM3_WEIGHTS_PATH in argv
    assert argv[argv.index("--kind") + 1] == "box"
    assert argv[argv.index("--device") + 1] == "cpu"


def test_sam_prompt_validation() -> None:
    with pytest.raises(ValueError, match="needs point"):
        SamPrompt(kind="point")
    with pytest.raises(ValueError, match="x1 > x0"):
        SamPrompt(kind="box", box=(5.0, 0.0, 1.0, 4.0))
    with pytest.raises(ValueError, match="non-empty text"):
        SamPrompt(kind="text", text="")
    with pytest.raises(ValueError, match="point\\|box\\|text"):
        SamPrompt(kind="blob")


def test_mask_file_segmenter_reports_itself(tmp_path: Path) -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:4] = True
    np.save(tmp_path / "m.npy", mask)
    seg = MaskFileSegmenter(masks={"a.png": tmp_path / "m.npy"})
    got, info = seg(tmp_path / "a.png", SamPrompt(kind="text", text="thing"))
    assert got.dtype == bool
    assert info["segmenter"] == "mask-file"
    assert info["mask_area_fraction"] == pytest.approx(6 / 20)
    with pytest.raises(KeyError, match="no mask file registered"):
        seg(tmp_path / "b.png", SamPrompt(kind="text", text="thing"))


def test_cli_edits_sam_with_mask_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from trippy.cli import main

    fixture = build_fixture(tmp_path)
    mask = np.zeros((PHOTO_H, PHOTO_W), dtype=bool)
    x0, y0, x1, y1 = object_box()
    mask[int(y0) : int(y1), int(x0) : int(x1)] = True
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, mask)

    edits_path = tmp_path / "edits.json"
    rc = main(
        [
            "edits",
            "sam",
            "--bundle",
            str(fixture["bundle_dir"]),
            "--scene",
            str(fixture["scene_root"]),
            "--view",
            fixture["names"][0],
            "--box",
            str(x0),
            str(y0),
            str(x1),
            str(y1),
            "--views-around",
            "0",
            "--mask",
            f"{fixture['names'][0]}={mask_path}",
            "--out",
            str(edits_path),
            "--op",
            "delete",
            "--mix",
            "0.0",
            "--name",
            "the object",
            "--preview",
            str(tmp_path / "preview.png"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert any(line.startswith("sam: segmenting ") for line in out.splitlines()), (
        "the viewer shows the child's progress lines while it waits"
    )
    summary = last_json_line(out)
    assert summary["n_points"] == len(fixture["object_ids"])
    assert summary["per_view"][0]["segmenter"]["segmenter"] == "mask-file"
    assert Path(summary["preview"]).exists()

    doc = EditDocument.load(edits_path)
    assert len(doc.regions) == 1
    region = doc.regions[0]
    assert region.kind == "pointset"
    assert region.op == "delete"
    assert region.name == "the object"
    assert sorted(region.params["point_ids"]) == fixture["object_ids"].tolist()


def test_cli_edits_sam_reports_a_bad_mask_flag(tmp_path: Path) -> None:
    from trippy.cli import main

    fixture = build_fixture(tmp_path)
    rc = main(
        [
            "edits",
            "sam",
            "--bundle",
            str(fixture["bundle_dir"]),
            "--scene",
            str(fixture["scene_root"]),
            "--view",
            fixture["names"][0],
            "--point",
            "100",
            "100",
            "--mask",
            "no-equals-sign",
            "--out",
            str(tmp_path / "edits.json"),
        ]
    )
    assert rc == 2


def test_lift_survives_points_almost_in_the_pinhole(tmp_path: Path) -> None:
    # A point at depth 1e-4 sends the distortion polynomial past int64 when
    # its pixel index is cast; a point behind the camera has no pixel at all.
    # Neither may warn, and neither may end up visible.
    import warnings

    fixture = build_fixture(tmp_path)
    xyz = np.vstack([fixture["xyz"], [[0.5, 0.5, 1e-4], [0.0, 0.0, -3.0]]])
    mask = np.ones((PHOTO_H, PHOTO_W), dtype=bool)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        lifted = lift_mask_in_view(
            fixture["views"][0],
            xyz,
            mask,
            (PHOTO_W, PHOTO_H),
            opencv_distortion=(0.057, -0.067, 8.0e-05, 0.0021),
        )
    assert not lifted["visible"][-2:].any()
    assert not lifted["selected"][-2:].any()
    # The ordinary points are unaffected by the two pathological ones. Not
    # quite ALL of them: with a whole-frame mask the object's outermost ring
    # of cells is mostly wall, and the gate erodes it (docs/LIMITATIONS.md
    # "The depth gate can drop an object's edge cells").
    assert lifted["selected"][fixture["object_ids"]].mean() > 0.99


# --- the fake segmenter: the viewer's child process, without SAM 3 ---------


def test_fake_segmenter_draws_the_prompt_and_never_claims_sam_ran(tmp_path: Path) -> None:
    from trippy.edit.sam_runner import FakeSegmenter

    fixture = build_fixture(tmp_path)
    photo = fixture["scene_root"] / "images" / fixture["names"][0]
    fake = FakeSegmenter()

    mask, info = fake(photo, SamPrompt(kind="box", box=(50.0, 60.0, 110.0, 130.0)))
    assert mask.shape == (PHOTO_H, PHOTO_W)
    assert mask.dtype == bool
    assert info["segmenter"] == "fake"
    assert mask[60:130, 50:110].all()
    assert not mask[0:59, :].any()
    assert fake.seen == [(fixture["names"][0], SamPrompt(kind="box", box=(50.0, 60.0, 110.0, 130.0)))]

    disc, _info = fake(photo, SamPrompt(kind="point", point=(100.0, 100.0)))
    assert disc[100, 100]
    assert disc[100, 100 + int(fake.point_radius_px) - 1]
    assert not disc[100, 100 + int(fake.point_radius_px) + 2]


def test_fake_requested_reads_the_env_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    from trippy.constants import SAM_FAKE_ENV
    from trippy.edit.sam_runner import fake_requested

    monkeypatch.delenv(SAM_FAKE_ENV, raising=False)
    assert not fake_requested()
    assert fake_requested(True)
    monkeypatch.setenv(SAM_FAKE_ENV, "1")
    assert fake_requested()
    monkeypatch.setenv(SAM_FAKE_ENV, "0")
    assert not fake_requested()


def test_cli_edits_sam_fake_takes_scene_root_from_the_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--fake` with NO `--scene`: exactly the command the viewer spawns."""
    from trippy.cli import main

    fixture = build_fixture(tmp_path)
    x0, y0, x1, y1 = object_box()
    edits_path = tmp_path / "edits.json"
    rc = main(
        [
            "edits", "sam",
            "--bundle", str(fixture["bundle_dir"]),
            "--view", fixture["names"][0],
            "--box", str(x0), str(y0), str(x1), str(y1),
            "--views-around", "0",
            "--out", str(edits_path),
            "--op", "fade",
            "--mix", "0.0",
            "--fake",
        ]
    )  # fmt: skip
    assert rc == 0
    summary = last_json_line(capsys.readouterr().out)
    assert summary["per_view"][0]["segmenter"]["segmenter"] == "fake"
    assert summary["scene_root"] == str(fixture["scene_root"])
    assert summary["settings"]["prompt_space"] == "photo"
    assert summary["n_points"] == len(fixture["object_ids"])

    doc = EditDocument.load(edits_path)
    assert len(doc.regions) == 1
    assert sorted(doc.regions[0].params["point_ids"]) == fixture["object_ids"].tolist()


def test_cli_edits_sam_env_switch_needs_no_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from trippy.cli import main
    from trippy.constants import SAM_FAKE_ENV

    monkeypatch.setenv(SAM_FAKE_ENV, "1")
    fixture = build_fixture(tmp_path)
    x0, y0, x1, y1 = object_box()
    rc = main(
        [
            "edits", "sam",
            "--bundle", str(fixture["bundle_dir"]),
            "--view", fixture["names"][0],
            "--box", str(x0), str(y0), str(x1), str(y1),
            "--views-around", "0",
            "--out", str(tmp_path / "edits.json"),
        ]
    )  # fmt: skip
    assert rc == 0
    assert last_json_line(capsys.readouterr().out)["per_view"][0]["segmenter"]["segmenter"] == "fake"


def test_cli_edits_sam_without_scene_or_scene_root_says_so(tmp_path: Path) -> None:
    from trippy.cli import main

    fixture = build_fixture(tmp_path)
    manifest = fixture["bundle_dir"] / "bundle.json"
    document = json.loads(manifest.read_text())
    del document["scene_root"]
    manifest.write_text(json.dumps(document))
    rc = main(
        [
            "edits", "sam",
            "--bundle", str(fixture["bundle_dir"]),
            "--view", fixture["names"][0],
            "--point", "100", "100",
            "--out", str(tmp_path / "edits.json"),
            "--fake",
        ]
    )  # fmt: skip
    assert rc == 2


# --- prompt spaces: the viewer measures VIEW pixels, SAM wants PHOTO pixels --


def test_scale_prompt_scales_points_and_boxes_but_not_text() -> None:
    from trippy.edit.sam_lift import scale_prompt

    point = scale_prompt(SamPrompt(kind="point", point=(10.0, 20.0)), 2.0)
    assert point.point == (20.0, 40.0)
    box = scale_prompt(SamPrompt(kind="box", box=(1.0, 2.0, 3.0, 4.0)), 2.5)
    assert box.box == (2.5, 5.0, 7.5, 10.0)
    text = SamPrompt(kind="text", text="a rock")
    assert scale_prompt(text, 3.0) is text


def test_sam_lift_view_space_prompt_matches_the_photo_space_one(tmp_path: Path) -> None:
    """A box in VIEW pixels selects what the same box in PHOTO pixels does.

    The photo is 2x the view here (`PHOTO_W / VIEW_W`), so the view-space box
    is the photo-space one halved -- which is exactly the conversion the
    viewer relies on `--prompt-space view` to make for it.
    """
    from trippy.edit.sam_runner import FakeSegmenter

    fixture = build_fixture(tmp_path)
    x0, y0, x1, y1 = object_box()
    photo_region, photo_summary = sam_lift(
        fixture["bundle_dir"], fixture["scene_root"], fixture["names"][0],
        SamPrompt(kind="box", box=(x0, y0, x1, y1)), FakeSegmenter(), views_around=0,
    )  # fmt: skip
    scale = PHOTO_W / VIEW_W
    view_region, view_summary = sam_lift(
        fixture["bundle_dir"], fixture["scene_root"], fixture["names"][0],
        SamPrompt(kind="box", box=(x0 / scale, y0 / scale, x1 / scale, y1 / scale)),
        FakeSegmenter(), views_around=0, prompt_space="view",
    )  # fmt: skip
    assert view_region.params["point_ids"] == photo_region.params["point_ids"]
    assert view_summary["settings"]["prompt_space"] == "view"
    assert view_summary["prompt_as_given"]["box"] == [x0 / scale, y0 / scale, x1 / scale, y1 / scale]
    assert view_summary["prompt"] == photo_summary["prompt"]


def test_sam_lift_rejects_an_unknown_prompt_space(tmp_path: Path) -> None:
    from trippy.edit.sam_runner import FakeSegmenter

    fixture = build_fixture(tmp_path)
    with pytest.raises(ValueError, match="prompt_space"):
        sam_lift(
            fixture["bundle_dir"], fixture["scene_root"], fixture["names"][0],
            SamPrompt(kind="point", point=(100.0, 100.0)), FakeSegmenter(),
            views_around=0, prompt_space="render",
        )  # fmt: skip


def test_sam_lift_reports_progress_for_every_view(tmp_path: Path) -> None:
    from trippy.edit.sam_runner import FakeSegmenter

    fixture = build_fixture(tmp_path)
    x0, y0, x1, y1 = object_box()
    lines: list[str] = []
    sam_lift(
        fixture["bundle_dir"], fixture["scene_root"], fixture["names"][0],
        SamPrompt(kind="box", box=(x0, y0, x1, y1)), FakeSegmenter(),
        views_around=2, progress=lines.append,
    )  # fmt: skip
    assert any(line.startswith(f"segmenting {fixture['names'][0]}") for line in lines)
    assert any("neighbour 1/2" in line for line in lines)
    assert lines[-1].startswith("vote over 3 view(s)")


# --- the SAM 3 shims (rebindings of SAM 3's own code, no SAM 3 needed) ------


def _stub_module(name: str, **attributes: Any) -> Any:
    """A throwaway module object standing in for one of SAM 3's."""
    import types

    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def test_install_fp32_addmm_rebinds_both_bindings_and_keeps_the_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`sam3.model.vitdet` imports `addmm_act` by value, so both must be patched.

    Patching only the definition site would leave the ONLY caller
    (`vitdet.Mlp.forward`) on the bfloat16 version -- the bug this whole shim
    exists to prevent (docs/LIMITATIONS.md "SAM-3 mask lift").
    """
    import torch

    from trippy.edit.sam_runner import _install_fp32_addmm

    def original(_activation: Any, _linear: Any, _mat1: Any) -> Any:
        raise AssertionError("the original addmm_act must not be reachable")

    fused = _stub_module("sam3.perflib.fused", addmm_act=original)
    vitdet = _stub_module("sam3.model.vitdet", addmm_act=original)
    monkeypatch.setitem(sys.modules, "sam3.perflib.fused", fused)
    monkeypatch.setitem(sys.modules, "sam3.model.vitdet", vitdet)

    assert _install_fp32_addmm()
    assert fused.addmm_act is vitdet.addmm_act
    assert fused.addmm_act is not original

    linear = torch.nn.Linear(3, 2)
    x = torch.ones(1, 3, dtype=torch.float32)
    with torch.no_grad():
        got = vitdet.addmm_act(torch.nn.GELU, linear, x)
        want = torch.nn.functional.gelu(linear(x))
    assert got.dtype == torch.float32, "no bfloat16 anywhere"
    assert torch.allclose(got, want, atol=1e-6)

    with torch.no_grad():
        relu = vitdet.addmm_act(torch.nn.functional.relu, linear, x)
        assert torch.allclose(relu, torch.nn.functional.relu(linear(x)), atol=1e-6)
        with pytest.raises(ValueError, match="Unexpected activation"):
            vitdet.addmm_act(torch.nn.Sigmoid, linear, x)


def test_install_fp32_addmm_reports_when_there_is_nothing_to_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trippy.edit.sam_runner import _install_fp32_addmm

    monkeypatch.delitem(sys.modules, "sam3.perflib.fused", raising=False)
    monkeypatch.delitem(sys.modules, "sam3.model.vitdet", raising=False)
    assert not _install_fp32_addmm()


def test_install_real_rope_forces_the_flag_the_builder_never_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MPS has no `torch.view_as_complex`; SAM 3's real-valued RoPE is exact."""
    from trippy.edit.sam_runner import _install_real_rope

    seen: list[dict[str, Any]] = []

    def create_vit_backbone(**kwargs: Any) -> str:
        seen.append(dict(kwargs))
        return "vit"

    builder = _stub_module("sam3.model_builder", _create_vit_backbone=create_vit_backbone)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", builder)

    assert _install_real_rope()
    assert builder._create_vit_backbone(compile_mode=None) == "vit"
    assert seen == [{"compile_mode": None, "use_rope_real": True}]

    monkeypatch.delitem(sys.modules, "sam3.model_builder")
    assert not _install_real_rope()
