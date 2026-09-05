"""Render every registered kk-coherent view with Splats' gsrender against a trained Gaussian PLY.

Module: trippy.hybrid.render_splat_views
Invariants:
    - Imports Splats' `tools/gsrender.py` BY PATH (`sys.path.insert`), never copies it into
      this repo (AGENTS.md forbidden list: "copying files from ~/Splats").
    - Dependency-light by design so this module also runs, unmodified, under Splats' own
      `tools/ml-sharp/.venv` (which has torch/numpy/PIL/plyfile but not trippy's scipy/lpips/
      pyyaml deps): only `numpy`, `torch`, `PIL`, and gsrender's own `plyfile` import are used
      directly, plus the handful of trippy modules that need none of those --
      `trippy.scene.dataset` (+ its own deps `trippy.geom.camera`, `trippy.scene.colmap_io`),
      `trippy.geom.xform_a`, `trippy.config`, `trippy.constants` -- verified scipy/lpips/yaml
      free by inspection of their own `import` lines (see docs comment in each file). Reusing
      `SceneDataset` (rather than re-deriving intrinsics/pose here) is what guarantees a render
      and its photo share pixel-for-pixel the same undistorted (H, W, K) grid.
    - Every call passes `max_hw=HYBRID_C_GSRENDER_MAX_HW` explicitly: gsrender's own kwarg
      default (32) corrupts near-camera Gaussian footprints (Splats' PROJECT.md note, also in
      this task's brief).
    - Only ever invoked with `--device mps` from inside a GPU-queue job
      (`scripts/gpu_submit.sh`); this module itself never touches MPS unless told to.
    - Idempotent per-frame: a frame whose three output files already exist is skipped (unless
      `--force`), and `--start-index`/`--end-index` shard a run into two-part-ready pieces
      (the outputs directory is the pairing key, not a manifest saying "did I run before").
Related docs: docs/PLAN-2026-09-05.md "Hybrid (v0.3): (C) render->photo U-Net refinement on
    gsrender.py outputs first"; trippy.hybrid.dataset_c (consumes this module's output layout).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from trippy.constants import HYBRID_C_GSRENDER_MAX_HW, HYBRID_C_GSRENDER_MIN_OPACITY, TRAIN_DEFAULT_WIDTH

# Splats' own tools/ directory holding gsrender.py -- read-only, never copied (see module
# docstring). Overridable via --gsrender-tools-dir / the `gsrender_tools_dir` argument, mainly
# so CPU tests never need this path to exist (they inject a fake render_fn/load_ply_fn).
DEFAULT_GSRENDER_TOOLS_DIR = Path("/Users/nzbirdranch/Splats/tools")

RenderFn = Callable[..., Any]
LoadPlyFn = Callable[[str], Any]


def _import_gsrender(tools_dir: Path):
    """Import Splats' gsrender.py by path (sys.path insert), never copied into this repo."""
    tools_dir_str = str(tools_dir)
    if tools_dir_str not in sys.path:
        sys.path.insert(0, tools_dir_str)
    import gsrender  # deferred: only real (non-injected) runs need this.

    return gsrender


def build_viewmat(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """World->camera 4x4 viewmat matching gsrender's convention.

    Identical construction to `~/Splats/research/visual/render_offpath.py`'s `viewmat()`:
    `V[:3, :3] = R(qvec)`, `V[:3, 3] = tvec` -- no inversion, since COLMAP's own qvec/tvec
    (and `SceneDataset`'s cached copy of them) are already the world->camera transform
    gsrender's `render()` expects (`mu = means @ Rcw.T + tr`).
    """
    from trippy.geom.xform_a import qvec2R  # numpy-only, deferred for clarity.

    v = np.eye(4, dtype=np.float32)
    v[:3, :3] = qvec2R(np.asarray(qvec, dtype=np.float64)).astype(np.float32)
    v[:3, 3] = np.asarray(tvec, dtype=np.float32)
    return v


def output_paths(out_dir: Path, stem: str) -> dict[str, Path]:
    """The three files one rendered frame writes: rgb png, depth npy, alpha npy."""
    return {
        "rgb": out_dir / f"{stem}.png",
        "depth": out_dir / f"{stem}.depth.npy",
        "alpha": out_dir / f"{stem}.alpha.npy",
    }


def already_rendered(out_dir: Path, stem: str) -> bool:
    """True if every one of `output_paths(out_dir, stem)` already exists on disk."""
    return all(p.exists() for p in output_paths(out_dir, stem).values())


def _to_numpy(x: Any) -> np.ndarray:
    """gsrender.render() returns numpy arrays by default (differentiable=False); tolerate
    a torch tensor too so an injected test double may return either."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def save_outputs(out_dir: Path, stem: str, rgb: np.ndarray, depth: np.ndarray, alpha: np.ndarray) -> dict[str, str]:
    """Write one frame's rgb (uint8 png), depth (float16 npy), alpha (float16 npy)."""
    from PIL import Image  # deferred: keeps a pure-arg-parsing import cheap.

    paths = output_paths(out_dir, stem)
    rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(rgb_u8, mode="RGB").save(paths["rgb"])
    np.save(paths["depth"], depth.astype(np.float16))
    np.save(paths["alpha"], alpha.astype(np.float16))
    return {k: str(v) for k, v in paths.items()}


def render_views(
    scene_root: str | Path,
    ply_path: str | Path,
    out_dir: str | Path,
    width: int = TRAIN_DEFAULT_WIDTH,
    device: str = "mps",
    names: list[str] | None = None,
    start_index: int = 0,
    end_index: int | None = None,
    force: bool = False,
    max_hw: int = HYBRID_C_GSRENDER_MAX_HW,
    min_opacity: float = HYBRID_C_GSRENDER_MIN_OPACITY,
    cache_root: str | Path | None = None,
    gsrender_tools_dir: str | Path = DEFAULT_GSRENDER_TOOLS_DIR,
    render_fn: RenderFn | None = None,
    load_ply_fn: LoadPlyFn | None = None,
) -> dict[str, Any]:
    """Render `names` (default: every registered image) of `scene_root` against `ply_path`.

    Builds (or reuses) `SceneDataset`'s undistortion cache at `width` so every render shares
    its photo's exact (H, W, K); for each frame calls `render_fn(gaussians, viewmat, K, W, H,
    dev=device, max_hw=max_hw, min_opacity=min_opacity, return_depth=True)` and writes the
    rgb/depth/alpha triple under `out_dir` (see `output_paths`).

    Args:
        scene_root: COLMAP scene root (images/ + sparse/0 or sparse_txt).
        ply_path: binary 3DGS PLY to render (e.g. kkc_15000.ply).
        out_dir: directory frame outputs are written to (created if missing).
        width: destination pinhole width in pixels -- forwarded to `SceneDataset` unchanged,
            so this must match whatever width `trippy.hybrid.dataset_c`/training later reads
            photos at.
        device: "cpu" or "mps", forwarded to `render_fn` as its `dev` kwarg. Real (non-test)
            runs always pass "mps" and only ever execute inside a GPU-queue job.
        names: image filenames to render; None renders every name in the scene (sorted).
        start_index, end_index: slice `names` (after sorting) to this half-open range, so a
            long scene can be rendered in multiple queue-job chunks without re-deciding which
            frames go where each time -- both jobs address the same sorted name list.
        force: re-render even if a frame's three output files already exist.
        max_hw, min_opacity: forwarded to `render_fn` (see trippy.constants for why the
            max_hw default must never be gsrender's own 32).
        cache_root: `SceneDataset`'s cache root; None uses `trippy.config.load_settings()`'s
            default (`TRIPPY_OUTPUT/cache`).
        gsrender_tools_dir: directory containing gsrender.py, only consulted when `render_fn`/
            `load_ply_fn` are not both supplied (see module docstring: never copied, imported
            by path).
        render_fn, load_ply_fn: injection points for tests (a fake render_fn/load_ply_fn pair
            avoids importing the real gsrender.py or touching MPS in CPU tests).

    Returns:
        A manifest dict (also written to `<out_dir>/manifest_<start>_<end>.json`):
        `{scene_root, ply_path, width, device, max_hw, min_opacity, num_requested,
        num_rendered, num_skipped, elapsed_s, frames: [...]}`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if render_fn is None or load_ply_fn is None:
        gsrender = _import_gsrender(Path(gsrender_tools_dir))
        render_fn = render_fn or gsrender.render
        load_ply_fn = load_ply_fn or gsrender.load_ply

    from trippy.config import load_settings  # deferred, see module docstring.
    from trippy.scene.dataset import SceneDataset  # deferred, see module docstring.

    settings = load_settings()
    resolved_cache_root = Path(cache_root) if cache_root is not None else settings.trippy_output / "cache"
    dataset = SceneDataset(scene_root, width, resolved_cache_root, device="cpu")
    name_to_index = {name: i for i, name in enumerate(dataset.names)}

    all_names = sorted(names) if names is not None else list(dataset.names)
    unknown = [n for n in all_names if n not in name_to_index]
    if unknown:
        raise ValueError(f"names not registered in {scene_root}: {unknown}")
    end = len(all_names) if end_index is None else end_index
    shard_names = all_names[start_index:end]

    gaussians, _ply_k, _ply_imsize = load_ply_fn(str(ply_path))

    frames: list[dict[str, Any]] = []
    t0 = time.monotonic()
    for name in shard_names:
        stem = Path(name).stem
        if not force and already_rendered(out_dir, stem):
            frames.append({"name": name, "stem": stem, "skipped": True})
            continue

        item = dataset[name_to_index[name]]
        height, width_img = int(item["rgb"].shape[0]), int(item["rgb"].shape[1])
        k = item["K"].numpy()
        qvec = item["qvec"].numpy()
        tvec = item["tvec"].numpy()
        viewmat = build_viewmat(qvec, tvec)

        t_frame = time.monotonic()
        rgb, depth, alpha = render_fn(
            gaussians,
            viewmat,
            k,
            width_img,
            height,
            dev=device,
            max_hw=max_hw,
            min_opacity=min_opacity,
            return_depth=True,
        )
        elapsed_ms = (time.monotonic() - t_frame) * 1000.0

        paths = save_outputs(out_dir, stem, _to_numpy(rgb), _to_numpy(depth), _to_numpy(alpha))
        frames.append({"name": name, "stem": stem, "skipped": False, "ms": elapsed_ms, "paths": paths})

    manifest = {
        "scene_root": str(scene_root),
        "ply_path": str(ply_path),
        "width": width,
        "device": device,
        "max_hw": max_hw,
        "min_opacity": min_opacity,
        "num_requested": len(shard_names),
        "num_rendered": sum(1 for f in frames if not f["skipped"]),
        "num_skipped": sum(1 for f in frames if f["skipped"]),
        "elapsed_s": time.monotonic() - t0,
        "frames": frames,
    }
    (out_dir / f"manifest_{start_index}_{end}.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, help="COLMAP scene root (images/ + sparse/0 or sparse_txt)")
    parser.add_argument("--ply", required=True, help="binary 3DGS PLY to render (e.g. kkc_15000.ply)")
    parser.add_argument("--out", required=True, help="output directory for rgb/depth/alpha triples")
    parser.add_argument("--width", type=int, default=TRAIN_DEFAULT_WIDTH, help="undistorted pinhole width")
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument(
        "--names", default=None, help="comma-separated subset of image filenames (default: all registered)"
    )
    parser.add_argument("--start-index", type=int, default=0, help="shard: first index (after sorting) to render")
    parser.add_argument("--end-index", type=int, default=None, help="shard: exclusive last index")
    parser.add_argument("--max-hw", type=int, default=HYBRID_C_GSRENDER_MAX_HW)
    parser.add_argument("--min-opacity", type=float, default=HYBRID_C_GSRENDER_MIN_OPACITY)
    parser.add_argument("--cache-root", default=None, help="SceneDataset cache root override")
    parser.add_argument("--force", action="store_true", help="re-render frames whose outputs already exist")
    parser.add_argument("--gsrender-tools-dir", default=str(DEFAULT_GSRENDER_TOOLS_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    names = [n.strip() for n in args.names.split(",") if n.strip()] if args.names else None
    manifest = render_views(
        scene_root=args.scene,
        ply_path=args.ply,
        out_dir=args.out,
        width=args.width,
        device=args.device,
        names=names,
        start_index=args.start_index,
        end_index=args.end_index,
        force=args.force,
        max_hw=args.max_hw,
        min_opacity=args.min_opacity,
        cache_root=args.cache_root,
        gsrender_tools_dir=Path(args.gsrender_tools_dir),
    )
    print(
        "JSON:"
        + json.dumps(
            {
                "num_requested": manifest["num_requested"],
                "num_rendered": manifest["num_rendered"],
                "num_skipped": manifest["num_skipped"],
                "elapsed_s": manifest["elapsed_s"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
