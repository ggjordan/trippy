"""Tests for trippy.hybrid.render_splat_views: argument parsing + a fake-gsrender injection.

Module: tests.test_hybrid_c_render_splat_views
Invariants under test: `build_arg_parser` exposes every documented flag with the right
    default; `render_views`, given a fake `render_fn`/`load_ply_fn` pair (never importing the
    real gsrender.py, never touching MPS), writes the expected rgb/depth/alpha triple per
    frame, is idempotent (a frame whose outputs already exist is skipped unless `--force`),
    and shards correctly via `start_index`/`end_index`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_hybrid_c_helpers import IMG_HEIGHT, IMG_WIDTH, build_synthetic_scene

from trippy.hybrid import render_splat_views
from trippy.hybrid.render_splat_views import build_arg_parser, output_paths, render_views


def test_build_arg_parser_defaults() -> None:
    args = build_arg_parser().parse_args(["--scene", "/scene", "--ply", "/x.ply", "--out", "/out"])
    assert args.scene == "/scene"
    assert args.ply == "/x.ply"
    assert args.out == "/out"
    assert args.device == "mps"
    assert args.names is None
    assert args.start_index == 0
    assert args.end_index is None
    assert args.max_hw == 400
    assert args.min_opacity == pytest.approx(0.02)
    assert args.force is False


def test_build_arg_parser_overrides() -> None:
    args = build_arg_parser().parse_args(
        [
            "--scene", "/scene", "--ply", "/x.ply", "--out", "/out",
            "--width", "2016", "--device", "cpu", "--names", "a.jpg,b.jpg",
            "--start-index", "10", "--end-index", "20", "--max-hw", "128",
            "--min-opacity", "0.1", "--force",
        ]
    )  # fmt: skip
    assert args.width == 2016
    assert args.device == "cpu"
    assert args.names == "a.jpg,b.jpg"
    assert args.start_index == 10
    assert args.end_index == 20
    assert args.max_hw == 128
    assert args.min_opacity == pytest.approx(0.1)
    assert args.force is True


def _fake_load_ply(path: str):
    return {"path": path}, None, None


def _fake_render(gaussians, viewmat, k, width, height, dev, max_hw, min_opacity, return_depth):
    assert return_depth is True
    assert viewmat.shape == (4, 4)
    assert k.shape == (3, 3)
    rgb = np.full((height, width, 3), 0.5, dtype=np.float32)
    depth = np.full((height, width), 3.0, dtype=np.float32)
    alpha = np.full((height, width), 0.9, dtype=np.float32)
    return rgb, depth, alpha


def test_render_views_writes_expected_files(tmp_path: Path) -> None:
    scene_root, names = build_synthetic_scene(tmp_path, n_images=3)
    out_dir = tmp_path / "renders"

    manifest = render_views(
        scene_root=scene_root,
        ply_path="unused.ply",
        out_dir=out_dir,
        width=IMG_WIDTH,
        device="cpu",
        cache_root=tmp_path / "cache",
        render_fn=_fake_render,
        load_ply_fn=_fake_load_ply,
    )

    assert manifest["num_requested"] == 3
    assert manifest["num_rendered"] == 3
    assert manifest["num_skipped"] == 0

    for name in names:
        stem = Path(name).stem
        paths = output_paths(out_dir, stem)
        assert paths["rgb"].exists()
        assert paths["depth"].exists()
        assert paths["alpha"].exists()

        from PIL import Image

        with Image.open(paths["rgb"]) as img:
            arr = np.asarray(img.convert("RGB"))
        assert arr.shape == (IMG_HEIGHT, IMG_WIDTH, 3)
        assert arr.dtype == np.uint8
        assert np.all(arr == 128)  # round(0.5 * 255)

        depth = np.load(paths["depth"])
        alpha = np.load(paths["alpha"])
        assert depth.dtype == np.float16
        assert alpha.dtype == np.float16
        assert depth.shape == (IMG_HEIGHT, IMG_WIDTH)
        assert np.allclose(depth, 3.0)
        assert np.allclose(alpha, 0.9)

    manifest_path = out_dir / "manifest_0_3.json"
    assert manifest_path.exists()
    on_disk = json.loads(manifest_path.read_text())
    assert on_disk["num_rendered"] == 3


def test_render_views_skips_already_rendered_frames_unless_forced(tmp_path: Path) -> None:
    scene_root, _names = build_synthetic_scene(tmp_path, n_images=2)
    out_dir = tmp_path / "renders"

    calls = {"n": 0}

    def counting_render(*args, **kwargs):
        calls["n"] += 1
        return _fake_render(*args, **kwargs)

    common = {
        "scene_root": scene_root,
        "ply_path": "unused.ply",
        "out_dir": out_dir,
        "width": IMG_WIDTH,
        "device": "cpu",
        "cache_root": tmp_path / "cache",
        "render_fn": counting_render,
        "load_ply_fn": _fake_load_ply,
    }

    render_views(**common)
    assert calls["n"] == 2

    manifest = render_views(**common)
    assert calls["n"] == 2, "second run must skip already-rendered frames"
    assert manifest["num_skipped"] == 2
    assert manifest["num_rendered"] == 0

    manifest_forced = render_views(**{**common, "force": True})
    assert calls["n"] == 4
    assert manifest_forced["num_rendered"] == 2


def test_render_views_shards_by_start_end_index(tmp_path: Path) -> None:
    scene_root, _names = build_synthetic_scene(tmp_path, n_images=4)
    out_dir = tmp_path / "renders"

    manifest = render_views(
        scene_root=scene_root,
        ply_path="unused.ply",
        out_dir=out_dir,
        width=IMG_WIDTH,
        device="cpu",
        cache_root=tmp_path / "cache",
        start_index=0,
        end_index=2,
        render_fn=_fake_render,
        load_ply_fn=_fake_load_ply,
    )
    assert manifest["num_requested"] == 2
    rendered_stems = {f["stem"] for f in manifest["frames"]}
    assert rendered_stems == {"IMG_0", "IMG_1"}
    assert not (out_dir / "IMG_2.png").exists()


def test_render_views_rejects_unknown_names(tmp_path: Path) -> None:
    scene_root, _names = build_synthetic_scene(tmp_path, n_images=2)
    with pytest.raises(ValueError, match="not registered"):
        render_views(
            scene_root=scene_root,
            ply_path="unused.ply",
            out_dir=tmp_path / "renders",
            width=IMG_WIDTH,
            device="cpu",
            cache_root=tmp_path / "cache",
            names=["NOPE.jpg"],
            render_fn=_fake_render,
            load_ply_fn=_fake_load_ply,
        )


def test_import_gsrender_inserts_tools_dir_on_sys_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`_import_gsrender` never runs unless render_fn/load_ply_fn are both omitted (see
    `render_views`); exercise it against a fake `gsrender.py` module on a throwaway path so
    the real ~/Splats path (and its MPS-capable torch) is never touched."""
    tools_dir = tmp_path / "fake_tools"
    tools_dir.mkdir()
    (tools_dir / "gsrender.py").write_text(
        "def render(*a, **k):\n    return None\n\n\ndef load_ply(path):\n    return {}, None, None\n"
    )
    import sys

    monkeypatch.delitem(sys.modules, "gsrender", raising=False)
    module = render_splat_views._import_gsrender(tools_dir)
    assert hasattr(module, "render")
    assert hasattr(module, "load_ply")
    assert str(tools_dir) in sys.path
