"""trippy command-line interface.

Module: trippy.cli
Invariants: `smoke` only touches MPS when --device mps is explicitly passed
    (never a silent default); the `train`/`eval` stubs do no work and always
    exit 2. `density` builds a PointSource and prints/saves its
    PointSet.summary(); it is CPU-only (point sources never touch MPS).
    `render` (trippy.render.pyramid_render.render_frames) only touches MPS
    when --device mps is explicitly passed, same rule as `smoke`; it never
    builds a SceneDataset over every registered image in a scene, only the
    frames named in --frames.
Related docs: docs/SPEC.md "Technical design", AGENTS.md forbidden
    list (no direct GPU/MPS work outside scripts/gpu_submit.sh -- `smoke
    --device mps` and `render --device mps` are only ever invoked by a
    GPU-queue job); docs/SPEC.md D4 (point sources).
    (never a silent default); the `render` stub does no work and always
    exits 2. `density` builds a PointSource and prints/saves its
    PointSet.summary(); it is CPU-only (point sources never touch MPS).
    `train`/`eval` only touch MPS when `--device mps` (or the config's own
    `device: mps`) is explicit -- same no-silent-fallback rule as `smoke`.
Related docs: docs/SPEC.md "Technical design", AGENTS.md forbidden
    list (no direct GPU/MPS work outside scripts/gpu_submit.sh -- `smoke
    --device mps` is only ever invoked by the GPU-queue job itself);
    docs/SPEC.md D4 (point sources); docs/EXPERIMENTS.md "Training runs".

`candidate-report` runs the full per-checkpoint evaluation pipeline docs/
SPEC.md D10 requires (export PLY -> Splats' shade/extent audits ->
dolly video -> off-path honesty sheet -> report.json + README.md); see
docs/EXPERIMENTS.md "Candidate report". It never opens an image itself
(AGENTS.md privacy rule) -- only metrics and file paths are printed/written.
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
    CANDIDATE_REPORT_DOLLY_DIRNAME,
    CANDIDATE_REPORT_JSON_FILENAME,
    CANDIDATE_REPORT_OFFPATH_DIRNAME,
    CANDIDATE_REPORT_README_FILENAME,
    DEFAULT_DENSITY_COLMAP_SPARSE_DIR,
    DEFAULT_DENSITY_GAUSSIAN_PLY,
    DEFAULT_MIN_OPACITY,
    DEPTH_POINTS_MISSING_DEPTH_EXIT_CODE,
    DOLLY_DEFAULT_POSE_NAME,
    GIT_DESCRIBE_MATCH_PATTERN,
    MONODEPTH_DEFAULT_CONF0,
    MONODEPTH_DEFAULT_STRIDE,
    MONODEPTH_DEFAULT_VOXEL,
    PARITY_DEFAULT_INDICES,
    PARITY_DEFAULT_NUM_LAYERS,
    RASTER_MODES,
    RASTER_NUM_LAYERS,
    RENDER_CACHE_SUBDIR,
    SHADE_FRAMES_KK,
    SMOKE_MPS_TEST_TENSOR_LEN,
    TRAIN_DEFAULT_MODE,
    TRAIN_EXPORT_FILENAME,
)
from trippy.eval.audits import audit_report
from trippy.points import depth_io
from trippy.points.colmap_sparse import ColmapSparseSource
from trippy.points.gaussian_ply import GaussianPlySource
from trippy.points.monodepth import MonoDepthSource
from trippy.points.source import PointSource
from trippy.render import pyramid_render
from trippy.render.candidate import render_candidate
from trippy.render.dolly import shade_dolly_poses
from trippy.render.offpath import offpath_poses
from trippy.train.config import TrainConfig
from trippy.train.eval import build_trainer_from_checkpoint, evaluate_checkpoint
from trippy.train.trainer import Trainer

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


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = TrainConfig.load_yaml(args.config)
    if args.device is not None:
        cfg.device = args.device
    if args.run_dir is not None:
        cfg.run_dir = args.run_dir
    trainer = Trainer(cfg)
    if args.resume is not None:
        trainer.resume(args.resume)
    metrics = trainer.fit(max_minutes=args.max_minutes)
    print(f"trippy train: run_dir={trainer.run_dir} final_epoch={trainer.epoch}")
    if metrics:
        print(f"trippy train: last eval psnr_mean={metrics.get('psnr_mean')} ssim_mean={metrics.get('ssim_mean')}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    metrics = evaluate_checkpoint(args.checkpoint, images=args.images, device=args.device)
    print(f"psnr_mean: {metrics['psnr_mean']}")
    print(f"ssim_mean: {metrics['ssim_mean']}")
    print(f"lpips_mean: {metrics['lpips_mean']}")
    print(f"n_images: {metrics['n_images']}")
    return 0


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


def _cmd_render(args: argparse.Namespace) -> int:
    device = pick_device(args.device)
    frame_names = [f.strip() for f in args.frames.split(",") if f.strip()]
    if not frame_names:
        raise ValueError("--frames must list at least one frame name")

    settings = load_settings()
    cache_root = settings.trippy_output / RENDER_CACHE_SUBDIR
    command = "trippy " + " ".join(sys.argv[1:])

    metrics = pyramid_render.render_frames(
        scene_root=args.scene,
        ply_path=args.ply,
        frame_names=frame_names,
        width=args.width,
        out_dir=args.out,
        device=device,
        mode=args.mode,
        num_layers=args.layers,
        min_opacity=args.min_opacity,
        size_mode=args.size_mode,
        max_points=args.max_points,
        cache_root=cache_root,
        command=command,
    )
    print(f"JSON:{json.dumps({'num_frames': len(metrics['frames']), 'out': str(args.out)})}")
def _cmd_depth_points(args: argparse.Namespace) -> int:
    settings = load_settings()
    scene_root = Path(args.scene)
    image_names = [s.strip() for s in args.images.split(",") if s.strip()]
    depth_dir = Path(args.depth_dir)
    cache_root = Path(args.cache_dir) if args.cache_dir else settings.trippy_output / "cache"

    if args.run_depth:
        cache_dir = depth_io.undistort_and_cache(scene_root, image_names, args.width, cache_root)
        manifest_path = depth_dir / "manifest.json"
        depth_io.write_depthpro_manifest(cache_dir, image_names, depth_dir, manifest_path)

        py_path, script_path = depth_io.resolve_depthpro_paths(settings.splats_root)
        depthpro_cmd = depth_io.format_depthpro_command(py_path, script_path, manifest_path)
        job_name = f"depthpro-{scene_root.name}"
        print("Run the following to compute DepthPro depth maps (GPU queue):")
        print(f"scripts/gpu_submit.sh --prio 11 --wait {job_name} -- bash -c {depthpro_cmd!r}")

        missing = depth_io.depth_output_missing(depth_dir, [Path(n).stem for n in image_names])
        if missing:
            print(f"missing depth outputs for: {missing}")
            return DEPTH_POINTS_MISSING_DEPTH_EXIT_CODE
        print("all depth outputs already present; re-run without --run-depth to build the PointSet.")
        return 0

    source = MonoDepthSource(
        scene_root,
        image_names,
        args.width,
        depth_dir,
        cache_root,
        stride=args.stride,
        voxel=args.voxel,
        conf0=args.conf0,
    )
    point_set = source.build()
    summary = point_set.summary()
    describe = source.describe()

    print(f"source: {describe}")
    print(f"count: {summary['count']}")
    print(f"bbox_min: {summary['bbox_min']}")
    print(f"bbox_max: {summary['bbox_max']}")
    print(f"median_nn_distance: {summary['median_nn_distance']}")
    print(f"provenance_histogram: {summary['provenance_histogram']}")
    print(f"JSON:{json.dumps({'summary': summary, 'describe': describe})}")

    if args.out is not None:
        out_path = Path(args.out)
        point_set.save_npz(out_path)
        summary_path = out_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps({"summary": summary, "describe": describe}, indent=2))
        print(f"wrote {out_path} and {summary_path}")

def _cmd_parity(args: argparse.Namespace) -> int:
    """Render published TRIPS checkpoint views through trippy and score them."""
    # Deferred import: pulls in torch/PIL/lpips and the whole raster stack, which
    # `trippy smoke` and `trippy density` have no need for.
    from trippy.render.parity import ParityConfig, run_parity

    indices: tuple[int, ...] = ()
    if args.indices:
        indices = tuple(int(tok) for tok in args.indices.replace(",", " ").split())
    images: tuple[str, ...] = ()
    if args.images:
        images = tuple(tok for tok in args.images.replace(",", " ").split() if tok)
    if not indices and not images:
        indices = tuple(PARITY_DEFAULT_INDICES)

    config = ParityConfig(
        scene_dir=args.scene,
        checkpoint_dir=args.checkpoint,
        epoch=args.epoch,
        scene_name=args.scene_name,
        out_dir=args.out,
        device=pick_device(args.device).type,
        indices=indices,
        images=images,
        num_layers=args.num_layers,
        render_scale=args.render_scale,
        modes=tuple(args.modes.replace(",", " ").split()),
        max_points=args.max_points,
        reference_dir=args.reference_dir,
        engine=args.engine,
        compare_engines=args.compare_engines,
    )
    report = run_parity(config)
    print(json.dumps(report["means"], indent=2))
    print(f"wrote {Path(config.out_dir) / 'summary_sheet.png'}")
    return 0


def _candidate_report_readme(report: dict) -> str:
    """Human-readable summary of a `candidate-report` run: numbers + artifact paths only.

    Never embeds or describes pixel content -- AGENTS.md's privacy rule
    means this function (and whoever wrote it) never opens the images it
    references; it only prints metrics and paths so Jordan can open them
    himself.
    """
    dolly = report["dolly"]
    offpath = report["offpath"]
    shade = report["audits"]["shade_audit"]
    extent = report["audits"]["extent_gate"]

    lines = [
        "# trippy candidate report",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Device: {report['device']}",
        f"- Scene: `{report['scene_root']}`",
        f"- Export PLY: `{report['export_ply']}`",
        "",
        "## Dolly (shade camera path)",
        f"- Frames: {dolly['n_frames']}",
        f"- Mean coverage (full frame, T_final-derived): {dolly['mean_coverage_full']:.4f}",
    ]
    if "videos" in dolly:
        lines.append(f"- Video (network output): `{dolly['videos']['net']}`")
        lines.append(f"- Video (raw level-0): `{dolly['videos']['raw']}`")
    if "honesty_sheet" in dolly:
        lines.append(f"- Honesty sheet: `{dolly['honesty_sheet']}`")

    lines += [
        "",
        "## Off-path honesty poses",
        f"- Frames: {offpath['n_frames']}",
        f"- Mean coverage (full frame, T_final-derived): {offpath['mean_coverage_full']:.4f}",
    ]
    if "honesty_sheet" in offpath:
        lines.append(f"- Honesty sheet: `{offpath['honesty_sheet']}`")

    lines += ["", "## Audits (Splats' own tools, run read-only via subprocess)"]
    if "error" in shade:
        lines.append(f"- Shade audit: FAILED -- {shade['error']}")
    else:
        for res in shade.get("results", []):
            lines.append(
                f"- Shade audit `{res.get('path')}`: mass_in_region={res.get('mass_in_region'):.1f} "
                f"(n_in_region={res.get('n_in_region')})"
            )
    if "error" in extent:
        lines.append(f"- Extent gate: FAILED -- {extent['error']}")
    else:
        for rec in extent.get("plys", []):
            lines.append(
                f"- Extent gate `{rec.get('ply_path')}`: radius p99={rec.get('radius_p99')} "
                f"max={rec.get('radius_max')} scene_diagonal={rec.get('scene_diagonal')}"
            )

    lines += [
        "",
        (
            "Jordan's viewer verdict is final (docs/EXPERIMENTS.md \"Jordan's viewer verdict is "
            "final\") -- these numbers rank candidates; they do not replace opening the dolly "
            "video and honesty sheet."
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def _cmd_candidate_report(args: argparse.Namespace) -> int:
    device = pick_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = build_trainer_from_checkpoint(args.checkpoint, device=str(device))
    export_path = trainer.export_ply(out_dir / TRAIN_EXPORT_FILENAME)
    scene_root = Path(trainer.cfg.scene_root)
    width = trainer.cfg.width
    del trainer  # render_candidate below rebuilds its own Trainer from the checkpoint.

    offpath_names = (
        [n.strip() for n in args.offpath.split(",") if n.strip()] if args.offpath else list(SHADE_FRAMES_KK)
    )
    dolly_poses = shade_dolly_poses(scene_root, pose_name=args.dolly_pose, width=width)
    offpath_pose_list = offpath_poses(scene_root, offpath_names, width=width)

    dolly_metrics = render_candidate(
        args.checkpoint,
        dolly_poses,
        out_dir / CANDIDATE_REPORT_DOLLY_DIRNAME,
        device=str(device),
        write_video_files=True,
    )
    offpath_metrics = render_candidate(
        args.checkpoint,
        offpath_pose_list,
        out_dir / CANDIDATE_REPORT_OFFPATH_DIRNAME,
        device=str(device),
        write_video_files=False,
    )

    audits = audit_report([str(export_path)], scene_root / "sparse_txt", frames=None)

    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "scene_root": str(scene_root),
        "export_ply": str(export_path),
        "dolly": dolly_metrics,
        "offpath": offpath_metrics,
        "audits": audits,
    }
    (out_dir / CANDIDATE_REPORT_JSON_FILENAME).write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / CANDIDATE_REPORT_README_FILENAME).write_text(_candidate_report_readme(report))

    print(
        "JSON:"
        + json.dumps(
            {
                "out": str(out_dir),
                "n_dolly_frames": dolly_metrics["n_frames"],
                "n_offpath_frames": offpath_metrics["n_frames"],
            }
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trippy")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="print environment + device diagnostics")
    smoke.add_argument("--device", choices=["cpu", "mps"], default=None)
    smoke.set_defaults(func=_cmd_smoke)

    render = sub.add_parser(
        "render", help="rasterise the TRIPS pyramid for chosen frames + a contact sheet (no U-Net yet)"
    )
    render.add_argument("--scene", required=True, help="COLMAP scene root (images/ + sparse/0 or sparse_txt)")
    render.add_argument("--points", choices=["gaussian"], default="gaussian", help="point source")
    render.add_argument("--ply", required=True, help="binary 3DGS PLY (--points gaussian)")
    render.add_argument("--min-opacity", type=float, default=DEFAULT_MIN_OPACITY, help="gaussian source only")
    render.add_argument("--size-mode", choices=["scale", "knn"], default="scale", help="gaussian source only")
    render.add_argument("--max-points", type=int, default=None, help="gaussian source only")
    render.add_argument("--width", type=int, required=True, help="layer-0 (undistorted) image width in pixels")
    render.add_argument(
        "--frames", required=True, help="comma-separated image filenames, e.g. IMG_3830.jpg,IMG_3700.jpg"
    )
    render.add_argument(
        "--mode",
        choices=list(RASTER_MODES),
        default=TRAIN_DEFAULT_MODE,
        help="pyramid layer-selection rule (see docs/GEOMETRY.md); 'trips' is TRIPS's own",
    )
    render.add_argument("--layers", type=int, default=RASTER_NUM_LAYERS)
    render.add_argument("--out", required=True, help="output directory")
    render.add_argument("--device", choices=["cpu", "mps"], default=None)
    render.set_defaults(func=_cmd_render)
    train = sub.add_parser("train", help="train a TRIPS-style model from a YAML config")
    train.add_argument("--config", required=True, help="path to a TrainConfig YAML file")
    train.add_argument("--resume", default=None, help="checkpoint .pt path to resume from")
    train.add_argument("--max-minutes", type=float, default=None, help="wall-clock budget override")
    train.add_argument("--device", choices=["cpu", "mps"], default=None, help="override the config's device")
    train.add_argument("--run-dir", default=None, help="override the config's run_dir (artefact output directory)")
    train.set_defaults(func=_cmd_train)

    ev = sub.add_parser("eval", help="evaluate a checkpoint's held-out (or given) images")
    ev.add_argument("--checkpoint", required=True, help="checkpoint .pt path")
    ev.add_argument("--images", nargs="*", default=None, help="image names to evaluate (default: held-out split)")
    ev.add_argument("--device", choices=["cpu", "mps"], default=None, help="override the checkpoint's device")
    ev.set_defaults(func=_cmd_eval)

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

    depth_points = sub.add_parser(
        "depth-points", help="MonoDepthSource: build DepthPro-derived points, or print the GPU job to run first"
    )
    depth_points.add_argument("--scene", required=True, help="scene root (images/ + sparse/0 or sparse_txt)")
    depth_points.add_argument("--images", required=True, help="comma-separated registered image filenames")
    depth_points.add_argument("--width", type=int, required=True, help="undistorted pinhole image width")
    depth_points.add_argument("--depth-dir", required=True, help="DepthPro output directory")
    depth_points.add_argument("--out", default=None, help="optional path to write the PointSet .npz")
    depth_points.add_argument("--stride", type=int, default=MONODEPTH_DEFAULT_STRIDE)
    depth_points.add_argument("--voxel", type=float, default=MONODEPTH_DEFAULT_VOXEL)
    depth_points.add_argument("--conf0", type=float, default=MONODEPTH_DEFAULT_CONF0)
    depth_points.add_argument(
        "--cache-dir", default=None, help="undistortion cache root (default: TRIPPY_OUTPUT/cache)"
    )
    depth_points.add_argument(
        "--run-depth",
        action="store_true",
        help="prepare inputs + print the gpu_submit.sh DepthPro command instead of building the PointSet",
    )
    depth_points.set_defaults(func=_cmd_depth_points)
    parity = sub.add_parser(
        "parity",
        help="render published TRIPS checkpoint views through trippy and score them vs the photos",
    )
    parity.add_argument("--scene", required=True, help="ADOP scene directory (dataset.ini, poses.txt, ...)")
    parity.add_argument("--checkpoint", required=True, help="checkpoint dir containing params.ini and ep<NNNN>/")
    parity.add_argument("--epoch", default="ep0600", help="epoch subdirectory name")
    parity.add_argument("--scene-name", default=None, help="override the scene_<name>_*.pth infix")
    parity.add_argument("--indices", default=None, help="comma-separated 0-based image indices")
    parity.add_argument("--images", default=None, help="comma-separated image filenames from images.txt")
    parity.add_argument("--out", required=True, help="output directory (under output/, gitignored)")
    parity.add_argument("--device", choices=["cpu", "mps"], default=None)
    parity.add_argument("--num-layers", type=int, default=PARITY_DEFAULT_NUM_LAYERS)
    parity.add_argument("--render-scale", type=float, default=None, help="override dataset.ini render_scale")
    parity.add_argument(
        "--modes",
        default="trips,broadcast,trilinear",
        help="comma-separated render modes: trips (the published path), broadcast, trilinear",
    )
    parity.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="random point subsample (CPU smoke runs only -- NOT a parity result)",
    )
    parity.add_argument(
        "--engine",
        choices=["native", "perlayer"],
        default="native",
        help=(
            "mode 'trips' implementation: 'native' = one render_pyramid(mode='trips') call, "
            "'perlayer' = a loop of num_layers=1 calls (the original harness)"
        ),
    )
    parity.add_argument(
        "--compare-engines",
        action="store_true",
        help="also render both trips engines and report the per-level difference",
    )
    parity.add_argument(
        "--reference-dir",
        default=None,
        help="directory of the authors' own renders (default: <checkpoint>/<epoch>/test)",
    )
    parity.set_defaults(func=_cmd_parity)

    candidate_report = sub.add_parser(
        "candidate-report",
        help="export a checkpoint's PLY, run Splats' shade/extent audits, render the dolly + honesty artifacts",
    )
    candidate_report.add_argument("--checkpoint", required=True, help="checkpoint .pt path")
    candidate_report.add_argument("--out", required=True, help="output directory")
    candidate_report.add_argument(
        "--dolly-pose", default=DOLLY_DEFAULT_POSE_NAME, help="frozen-orientation pose for the shade dolly"
    )
    candidate_report.add_argument(
        "--offpath",
        default=None,
        help="comma-separated registered image names for off-path honesty poses (default: SHADE_FRAMES_KK)",
    )
    candidate_report.add_argument("--device", choices=["cpu", "mps"], default=None, help="override the checkpoint's device")
    candidate_report.set_defaults(func=_cmd_candidate_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
