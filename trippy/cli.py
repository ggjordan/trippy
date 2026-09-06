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

`train --report` runs that same pipeline against the just-finished run's
final checkpoint (`trippy.render.report.run_train_report`), plus a cached
baseline audit of the run's own source PLY, a baseline-vs-candidate
comparison table appended to the run's README.md, a free-navigation viewer
bundle + Mac launcher exported from the same final checkpoint (Jordan:
"fixed dolly paths are hard to judge, I want to navigate freely"), and
delivery via `scripts/deliver.sh` (launcher first) -- see docs/EXPERIMENTS.md
"Self-reporting training runs". Reporting failures are caught here and
written to `<run_dir>/REPORT_FAILED.txt`; they never fail an
otherwise-successful training run (`trippy train`'s exit code reflects
`fit()` only). A missing/stale viewer binary alone never triggers that file --
`trippy.render.report.build_mac_viewer_launcher` records it in
`<run_dir>/report/VIEWER_LAUNCHER_FAILED.txt` instead and the rest of the
report still completes.
`points-build` builds any `trippy.train.config.PointSourceConfig`-described
source (gaussian/colmap/union/npz, the same schema as a TrainConfig YAML's
`point_source:` block, taken as the document root here) and writes it to
`.npz` + a summary JSON, exactly like `density`/`depth-points --out` do for
their own single source -- the generic entry point for building a
"union" of a Gaussian PLY and a MonoDepth `.npz` with a voxel dedupe
(EXP-0006). CPU-only; never touches MPS.

`distill` runs the design-B fallback pipeline (docs/SPEC.md D2,
docs/EXPERIMENTS.md "Distillation (design B)"): distils a trained TRIPS
checkpoint into a plain-Gaussian PLY every existing viewer (Brush, Splats'
publish path, Quest) can open. `--stage render` renders the checkpoint's
network output at the training cameras plus near-path interpolated cameras
and writes a Brush-trainable COLMAP image set (`trippy.distill.render_set`,
MPS-capable, same "only via --device mps inside a GPU-queue job" rule as
`train`/`render`); `--stage brush-cmd` resolves/prints the Brush CLI
command and writes a queue-ready job script without running it
(`trippy.distill.brush_runner` -- Brush's trainer must only run via
`scripts/gpu_submit.sh --train`, never from this process); `--stage
compare` runs Splats' shade/extent audits on the baseline, TRIPS-export,
and (once Brush training has finished) distilled PLYs and prints a 3-column
comparison table (`trippy.distill.compare`). `--stage all` (the default)
runs render then compare, printing the brush-cmd stage's output in between
so the exact GPU-training command to queue next is always in front of you.
`export-bundle` writes a `trippy-bundle-1` directory (bundle.json +
points.npz + weights.safetensors) that the native Rust viewer opens: the
points stay in WORLD space and every camera of the scene is listed, so the
viewer can fly freely rather than replay one baked view. It accepts either a
TRIPS/ADOP checkpoint (with `--scene`) or a trippy-native checkpoint, and
auto-detects which -- see `trippy.render.bundle`. CPU-only.

`bundle-launcher` runs the same three steps `train --report` now runs from
its own final checkpoint -- `export-bundle`, a Mac double-click launcher via
`scripts/open_mac_viewer.sh`, and delivery via `scripts/deliver.sh` -- against
any checkpoint passed on the command line (`trippy.render.report.
export_bundle_and_viewer_launcher`), so a free-navigation launcher can be
(re)built for an existing run without re-training. CPU-only; never fails on a
missing/stale viewer binary (prints the failure and still writes the bundle).

`leaderboard` scans every run directory under `$TRIPPY_OUTPUT/runs/**/` with a
finished self-report (`report/report.json` or `candidate/report.json`) plus a
`metrics.jsonl`, and writes one markdown + PNG comparison table across all of
them plus the fixed Gaussian/Design-C baselines
(`trippy.render.leaderboard.write_leaderboard`) -- see docs/EXPERIMENTS.md
"Leaderboard". CPU-only; `--deliver` also hands the PNG to `scripts/deliver.sh`
under the fixed `trips-leaderboard` name (the same thing `train --report`
already does automatically at the end of every run).

`prune-run` applies `trippy.train.retention`'s checkpoint-retention policy to
an existing run directory's `checkpoints/` (the same policy `Trainer.
save_checkpoint` now applies automatically after every save) -- for a run
that trained before this policy existed, or one that was never going to
finish its own retention because it is still running. It only ever
deletes `checkpoint_ep<NNNN>.pt` files matching that filename pattern:
`checkpoint_latest.pt`/`checkpoint_best.pt` are never in its candidate list,
the single newest epoch file is never deleted regardless of `--keep-last`,
and any file modified within `--protect-seconds` (default
`PRUNE_RUN_DEFAULT_PROTECT_SECONDS` = 120s) is skipped so a checkpoint a
still-running job just wrote is never raced. `--dry-run` prints exactly what
would be deleted (and the total bytes that would free) without deleting
anything.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml

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
    DISTILL_BRUSH_JOB_FILENAME,
    DISTILL_BRUSH_OUT_DIRNAME,
    DISTILL_COMPARE_FILENAME,
    DISTILL_DEFAULT_BRUSH_ITERS,
    DISTILL_DEFAULT_INTERP_K,
    DISTILL_DEFAULT_MAX_INIT_POINTS,
    DISTILL_MAX_JUMP_MULTIPLIER,
    DISTILL_SPARSE_DIRNAME,
    DOLLY_DEFAULT_POSE_NAME,
    EVAL_EXPOSURE_MODES,
    GIT_DESCRIBE_MATCH_PATTERN,
    MONODEPTH_DEFAULT_CONF0,
    MONODEPTH_DEFAULT_STRIDE,
    MONODEPTH_DEFAULT_VOXEL,
    PARITY_DEFAULT_INDICES,
    PARITY_DEFAULT_NUM_LAYERS,
    PRUNE_RUN_DEFAULT_PROTECT_SECONDS,
    RASTER_MODES,
    RASTER_NUM_LAYERS,
    RENDER_CACHE_SUBDIR,
    SHADE_FRAMES_KK,
    SMOKE_MPS_TEST_TENSOR_LEN,
    TRAIN_CHECKPOINT_BEST_JSON_FILENAME,
    TRAIN_CHECKPOINT_DIRNAME,
    TRAIN_DEFAULT_CHECKPOINT_KEEP_EVERY,
    TRAIN_DEFAULT_CHECKPOINT_KEEP_LAST,
    TRAIN_DEFAULT_MODE,
    TRAIN_EXPORT_FILENAME,
    TRAIN_REPORT_DIRNAME,
    TRAIN_REPORT_FAILED_FILENAME,
)
from trippy.eval.audits import audit_report
from trippy.hybrid.config_c import HybridCConfig
from trippy.hybrid.train_c import HybridCTrainer
from trippy.hybrid.train_c import evaluate_checkpoint as evaluate_hybrid_c_checkpoint
from trippy.points import depth_io
from trippy.points.colmap_sparse import ColmapSparseSource
from trippy.points.gaussian_ply import GaussianPlySource
from trippy.points.monodepth import MonoDepthSource
from trippy.points.source import PointSource
from trippy.render import pyramid_render
from trippy.render.bundle import TRIPS_DEFAULT_EPOCH
from trippy.render.bundle import export_bundle as write_export_bundle
from trippy.render.candidate import render_candidate
from trippy.render.dolly import shade_dolly_poses
from trippy.render.offpath import offpath_poses
from trippy.train import retention
from trippy.train.config import PointSourceConfig, TrainConfig
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


def _run_train_report_safely(trainer: Trainer, metrics: dict) -> None:
    """Run `trippy.render.report.run_train_report`, never letting it fail the training run.

    Requirement 1 of this task's brief: `--report` must not crash a run
    that trained successfully. Any exception here (a broken audit tool, a
    missing scene sparse dir, deliver.sh refusing an artifact, ...) is
    caught, logged to stderr, and recorded in `<run_dir>/REPORT_FAILED.txt`
    -- `trippy train`'s own exit code is unaffected either way.
    """
    # Deferred import: pulls in the render/audit stack `trippy train` without
    # `--report` has no need for.
    from trippy.render.report import run_train_report

    try:
        report = run_train_report(trainer, metrics)
        print(f"trippy train: report -> {trainer.run_dir / TRAIN_REPORT_DIRNAME}")
        print(f"trippy train: {report['summary_line']}")
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        failed_path = Path(trainer.run_dir) / TRAIN_REPORT_FAILED_FILENAME
        failed_path.write_text(
            "trippy train --report failed after a successful training run.\n"
            f"error: {exc!r}\n\n{traceback.format_exc()}"
        )
        print(f"trippy train: --report FAILED (see {failed_path}): {exc}", file=sys.stderr)


def _cmd_train(args: argparse.Namespace) -> int:
    # Eager import of everything --report needs, so a multi-hour run uses ONE consistent code
    # version even if main is updated underneath it (a lazy import mid-run hit an ImportError on
    # 2026-09-06 after constants.py changed on disk).
    if getattr(args, "report", False):
        from trippy.eval import audits as _au  # noqa: F401
        from trippy.render import candidate as _cd  # noqa: F401
        from trippy.render import leaderboard as _lb  # noqa: F401
        from trippy.render import report as _rp  # noqa: F401
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
    if args.report:
        _run_train_report_safely(trainer, metrics)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """Re-evaluate a checkpoint, including the shade-vs-other held-out split (see Trainer.evaluate).

    Writes `<run_dir>/eval_manual_<timestamp>/metrics.json` and appends an
    eval row to the run's own `metrics.jsonl` (`trippy.train.eval.
    evaluate_checkpoint`), so `trippy leaderboard` picks up the split for a
    checkpoint that finished training before the split existed -- no
    retraining needed.

    `--exposure-mode` selects what a held-out image's HEADLINE number
    ("_eval"-suffixed fields, printed as "shade (eval)"/"other (eval)"
    below) is computed with: "own" (the frame's own never-trained
    exposure -- the only behaviour before this feature), "neighbours"
    (default -- interpolated from the nearest TRAINING frames by capture
    order, never reading the held-out photo, `trippy.net.camera_model.
    interpolate_from_train_neighbours`), or "calibrate" (the `--calibrate`
    fit below, promoted to be the headline number). None keeps the
    checkpoint's own `cfg.eval_exposure_mode`.

    `--calibrate` additionally fits each held-out image's own exposure
    (`--calibrate-wb`: and its red/blue white balance) to its own photo
    before scoring it, with everything else frozen, and prints the
    calibrated numbers next to the strict ones -- TRIPS's own
    `optimize_eval_camera`, cut down to the photometric scalars (see
    `Trainer.calibrate_frame` for why that is legitimate and what it does
    NOT prove). Either way, a per-image diagnostics table is printed last.
    """
    metrics = evaluate_checkpoint(
        args.checkpoint,
        images=args.images,
        device=args.device,
        calibrate=True if args.calibrate else None,
        calibrate_white_balance=True if args.calibrate_wb else None,
        exposure_mode=args.exposure_mode,
    )
    shade, other = metrics.get("shade") or {}, metrics.get("other") or {}
    shade_eval, other_eval = metrics.get("shade_eval") or {}, metrics.get("other_eval") or {}
    print(f"psnr_mean: {metrics['psnr_mean']}")
    print(f"ssim_mean: {metrics['ssim_mean']}")
    print(f"lpips_mean: {metrics['lpips_mean']}")
    print(f"n_images: {metrics['n_images']}")
    print(f"shade: n={shade.get('n')} psnr={shade.get('psnr')} ssim={shade.get('ssim')} lpips={shade.get('lpips')}")
    print(f"other: n={other.get('n')} psnr={other.get('psnr')} ssim={other.get('ssim')} lpips={other.get('lpips')}")
    print(f"exposure_mode: {metrics.get('exposure_mode')}")
    print(f"psnr_mean (eval): {metrics.get('psnr_mean_eval')}")
    print(
        f"shade (eval): n={shade_eval.get('n')} psnr={shade_eval.get('psnr')} "
        f"ssim={shade_eval.get('ssim')} lpips={shade_eval.get('lpips')}"
    )
    print(
        f"other (eval): n={other_eval.get('n')} psnr={other_eval.get('psnr')} "
        f"ssim={other_eval.get('ssim')} lpips={other_eval.get('lpips')}"
    )
    if metrics.get("calibrated"):
        shade_c, other_c = metrics.get("shade_calibrated") or {}, metrics.get("other_calibrated") or {}
        print(f"psnr_mean_calibrated: {metrics.get('psnr_mean_calibrated')}")
        print(f"shade (calibrated): n={shade_c.get('n')} psnr={shade_c.get('psnr')} ssim={shade_c.get('ssim')}")
        print(f"other (calibrated): n={other_c.get('n')} psnr={other_c.get('psnr')} ssim={other_c.get('ssim')}")
    print(_per_image_diagnostics_table(metrics))
    return 0


def _per_image_diagnostics_table(metrics: dict) -> str:
    """Markdown per-image table: PSNR, brightness ratio, best-gain PSNR, calibrated/eval PSNR.

    Pure formatting over `Trainer.evaluate`'s "per_image" dict (docs/EXPERIMENTS.md
    "Per-image exposure diagnostics"); columns whose values are all missing print as "n/a",
    so an old metrics dict still renders. "mode" and "PSNR (eval)" are the headline number
    under `exposure_mode` (docs/EXPERIMENTS.md "Test-time camera calibration") -- "own" for
    every training-set row regardless of the requested mode, per `Trainer.evaluate`.
    """
    per_image = metrics.get("per_image") or {}
    shade = set((metrics.get("shade") or {}).get("names") or [])
    header = (
        "| image | group | PSNR | exposure gain | pred mean | photo mean | photo/pred | "
        "best gain | PSNR@best gain | mode | PSNR (eval) | PSNR calibrated |"
    )
    lines = [header, "|" + "---|" * 12]
    columns = [
        ("psnr", ".2f"),
        ("exposure_gain", ".3f"),
        ("pred_mean", ".4f"),
        ("target_mean", ".4f"),
        ("brightness_ratio", ".3f"),
        ("gain_best", ".3f"),
        ("psnr_gain", ".2f"),
    ]
    for name in sorted(per_image):
        row = per_image[name]
        cells = [name, "shade" if name in shade else "other"]
        cells += [_fmt_diagnostic(row.get(key), spec) for key, spec in columns]
        cells.append(str(row.get("exposure_mode", "n/a")))
        cells.append(_fmt_diagnostic(row.get("psnr_eval"), ".2f"))
        cells.append(_fmt_diagnostic(row.get("psnr_calibrated"), ".2f"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _fmt_diagnostic(value: object, spec: str) -> str:
    """One diagnostics-table cell: formatted number, or "n/a" for a missing/non-numeric value."""
    return format(value, spec) if isinstance(value, int | float) else "n/a"


def _cmd_hybrid_c_train(args: argparse.Namespace) -> int:
    cfg = HybridCConfig.load_yaml(args.config)
    if args.device is not None:
        cfg.device = args.device
    trainer = HybridCTrainer(cfg)
    if args.resume is not None:
        trainer.resume(args.resume)
    metrics = trainer.fit(max_minutes=args.max_minutes)
    print(f"trippy hybrid-c train: run_dir={trainer.run_dir} final_epoch={trainer.epoch}")
    if metrics:
        base = metrics["baseline"]["all"]["psnr_mean"]
        ref = metrics["refined"]["all"]["psnr_mean"]
        print(f"trippy hybrid-c train: last eval baseline_psnr={base} refined_psnr={ref}")
    return 0


def _cmd_hybrid_c_eval(args: argparse.Namespace) -> int:
    metrics = evaluate_hybrid_c_checkpoint(args.checkpoint, images=args.images, device=args.device)
    print(f"JSON:{json.dumps({k: v for k, v in metrics.items() if k != 'names'})}")
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

def _cmd_points_build(args: argparse.Namespace) -> int:
    """Build any `PointSourceConfig`-describable source and dump it to .npz + summary JSON.

    `--config` is a YAML file holding exactly the fields `PointSourceConfig`
    accepts (the same schema as a `TrainConfig` YAML's `point_source:` block,
    but as the document root) -- so a "union" type's nested `sources:` list
    works unchanged, including a "npz" leaf that loads a `PointSet` another
    tool (or this same command, run earlier) already wrote to disk. This is
    how EXP-0006's Union(Gaussian, MonoDepth) point set is built: one
    "gaussian" leaf, one "npz" leaf pointing at the MonoDepth `.npz`, under a
    "union" parent with a `voxel` dedupe.
    """
    data = yaml.safe_load(Path(args.config).read_text())
    source_config = PointSourceConfig(**(data or {}))
    source = source_config.to_source()

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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    point_set.save_npz(out_path)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"summary": summary, "describe": describe}, indent=2))
    print(f"wrote {out_path} and {summary_path}")
    return 0


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


def _cmd_export_bundle(args: argparse.Namespace) -> int:
    """Write a self-contained `trippy-bundle-1` directory for the native viewer."""
    bundle_dir, document = write_export_bundle(
        checkpoint=args.checkpoint,
        out=args.out,
        scene=args.scene,
        epoch=args.epoch,
        name=args.name,
    )
    summary = {k: v for k, v in document.items() if k != "views"}
    summary["num_views"] = len(document["views"])
    print(json.dumps(summary, indent=2))
    print(f"wrote {bundle_dir}")
    return 0


def _cmd_bundle_launcher(args: argparse.Namespace) -> int:
    """Standalone `bundle-launcher`: export-bundle + Mac viewer launcher + delivery, for any checkpoint.

    Runs the same three steps `trippy train --report` now runs from its own
    final checkpoint (`trippy.render.report.export_bundle_and_viewer_launcher`)
    against whatever checkpoint is passed here -- so a free-navigation bundle
    + launcher can be (re)built for an existing run without re-training, or
    for a checkpoint `--report` never got the chance to touch.
    """
    # Deferred import: pulls in the render/audit stack `trippy export-bundle`
    # alone has no need for.
    from trippy.render.report import default_bundle_out_dir, export_bundle_and_viewer_launcher

    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.out) if args.out else default_bundle_out_dir(checkpoint_path)
    why_base = f"trippy bundle-launcher {args.name}: checkpoint {checkpoint_path}"

    result = export_bundle_and_viewer_launcher(
        checkpoint_path,
        out_dir,
        args.name,
        why_base=why_base,
        scene=args.scene,
        epoch=args.epoch,
    )
    print(f"wrote bundle -> {result['bundle_dir']}")
    if result["viewer"]["status"] == "ok":
        print(f"wrote viewer launcher -> {result['viewer']['command_path']}")
    else:
        print(f"bundle-launcher: viewer launcher FAILED (bundle was still written): {result['viewer']['note']}", file=sys.stderr)
    print(f"delivery: {result['delivery']['status']}")
    print(
        "JSON:"
        + json.dumps(
            {"bundle_dir": result["bundle_dir"], "viewer": result["viewer"], "delivery": result["delivery"]}
        )
    )
    return 0


def _cmd_leaderboard(args: argparse.Namespace) -> int:
    """`trippy leaderboard --out <dir> [--deliver]`: one comparison table across every run."""
    # Deferred import: pulls in PIL/yaml, which `trippy smoke`/`density` have no need for.
    from trippy.render.leaderboard import (
        default_leaderboard_out_dir,
        regenerate_and_deliver,
        write_leaderboard,
    )

    out_dir = Path(args.out) if args.out else default_leaderboard_out_dir()
    if args.deliver:
        result = regenerate_and_deliver(out_dir)
    else:
        result = write_leaderboard(out_dir)

    print(f"trippy leaderboard: {len(result['rows'])} row(s) -> {result['markdown_path']}, {result['png_path']}")
    if "delivery" in result:
        print(f"trippy leaderboard: delivery {result['delivery']['status']}")
    return 0


def _cmd_prune_run(args: argparse.Namespace) -> int:
    """`trippy prune-run <run_dir> [--dry-run]`: apply the retention policy to an existing run.

    Reads `checkpoints/best.json` (if present -- older runs trained before
    this feature won't have one) for the best-epoch protection, globs
    `checkpoints/checkpoint_ep*.pt`, and defers the actual keep/delete
    decision to `trippy.train.retention.select_checkpoints_to_delete` --
    exactly the function `Trainer.save_checkpoint` itself calls, so this
    command reproduces the same policy for a run that already finished (or
    is still running) rather than re-implementing it.
    """
    run_dir = Path(args.run_dir)
    checkpoint_dir = run_dir / TRAIN_CHECKPOINT_DIRNAME
    if not checkpoint_dir.is_dir():
        print(f"trippy prune-run: no {TRAIN_CHECKPOINT_DIRNAME}/ directory under {run_dir}", file=sys.stderr)
        return 2

    best_epoch = None
    best_json_path = checkpoint_dir / TRAIN_CHECKPOINT_BEST_JSON_FILENAME
    if best_json_path.exists():
        try:
            best_epoch = json.loads(best_json_path.read_text()).get("epoch")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"trippy prune-run: could not read {best_json_path} ({exc}); continuing without it", file=sys.stderr)

    all_epoch_paths = sorted(checkpoint_dir.glob("checkpoint_ep*.pt"))
    epochs = [(p, retention.epoch_of(p)) for p in all_epoch_paths]
    recognised = [(p, e) for p, e in epochs if e is not None]
    if not recognised:
        print(f"trippy prune-run: no checkpoint_ep*.pt files under {checkpoint_dir}; nothing to do")
        return 0
    newest_epoch = max(e for _, e in recognised)

    to_delete = retention.select_checkpoints_to_delete(
        [p for p, _ in recognised],
        best_epoch=best_epoch,
        keep_every=args.keep_every,
        keep_last=args.keep_last,
        protect_newer_than_s=args.protect_seconds,
    )
    # Hard safety net independent of --keep-last (task brief: "must NEVER touch ... the
    # newest epoch file"): even a `--keep-last 0` misfire must not remove it.
    to_delete = [p for p in to_delete if retention.epoch_of(p) != newest_epoch]

    total_bytes = 0
    for victim in to_delete:
        try:
            size = victim.stat().st_size
        except FileNotFoundError:
            continue
        total_bytes += size
        if args.dry_run:
            print(f"would delete {victim} ({size} bytes)")
        else:
            victim.unlink(missing_ok=True)
            print(f"deleted {victim} ({size} bytes)")

    verb = "would free" if args.dry_run else "freed"
    prefix = "DRY RUN: " if args.dry_run else ""
    print(f"trippy prune-run: {prefix}{len(to_delete)} file(s), {verb} {total_bytes} bytes")
    print(
        "JSON:"
        + json.dumps(
            {
                "run_dir": str(run_dir),
                "dry_run": args.dry_run,
                "best_epoch": best_epoch,
                "newest_epoch": newest_epoch,
                "deleted": [str(p) for p in to_delete],
                "bytes_freed": total_bytes,
            }
        )
    )
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
    if "dolly_stop_index" in dolly:
        stopped = "yes" if dolly.get("dolly_stopped_early") else "no"
        lines.append(
            f"- Stopped before camera exits geometry: {stopped} "
            f"(kept frames 0..{dolly['dolly_stop_index']} of {dolly['n_frames'] - 1}, "
            f"centre coverage threshold {dolly['dolly_stop_threshold']:g})"
        )
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
        stop_at_low_coverage=True,
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


def _cmd_distill(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = ("render", "brush-cmd", "compare") if args.stage == "all" else (args.stage,)

    report: dict = {}
    report_path = out_dir / "distill_report.json"
    if "render" not in stages and report_path.exists():
        # A later-only stage (brush-cmd/compare run on its own) reuses an earlier
        # render stage's own report instead of requiring every flag again.
        report = json.loads(report_path.read_text())

    if "render" in stages:
        from trippy.distill.render_set import render_distill_set

        if args.checkpoint is None:
            print("trippy distill: --checkpoint is required for --stage render/all", file=sys.stderr)
            return 2
        device = pick_device(args.device)
        max_init_points = None if args.max_init_points < 0 else args.max_init_points
        report = render_distill_set(
            args.checkpoint,
            out_dir,
            device=str(device),
            interp_k=args.interp_k,
            max_jump_multiplier=args.max_jump_multiplier,
            max_init_points=max_init_points,
        )
        print(
            f"trippy distill: rendered {report['n_anchor_images']} anchor + "
            f"{report['n_interpolated_images']} interpolated cameras "
            f"({report['n_skipped_pairs']} pair(s) skipped by the honesty guard) -> {out_dir}"
        )
        print(f"trippy distill: TRIPS export -> {report['trips_export_ply']}")

    if "brush-cmd" in stages:
        from trippy.distill.brush_runner import (
            brush_gpu_submit_command,
            brush_train_command,
            resolve_brush_binary,
            write_brush_job_script,
        )

        binary = Path(args.brush_binary) if args.brush_binary else resolve_brush_binary()
        width = report.get("width") if report else args.max_resolution
        brush_out_dir = out_dir / DISTILL_BRUSH_OUT_DIRNAME
        argv = brush_train_command(
            binary if binary is not None else "<build rust/brush-trips first -- see rust/README.md>",
            out_dir,
            brush_out_dir,
            total_train_iters=args.brush_iters,
            max_resolution=width,
        )
        job_name = args.job_name or f"distill-{out_dir.name}"
        script_path = write_brush_job_script(out_dir / DISTILL_BRUSH_JOB_FILENAME, argv)
        print("trippy distill: brush training command")
        print("  " + " ".join(str(a) for a in argv))
        print(f"trippy distill: job script -> {script_path}")
        print(f"trippy distill: queue it with -> {brush_gpu_submit_command(job_name, script_path)}")
        if binary is None:
            print(
                "trippy distill: WARNING -- no brush binary found; build it first (rust/README.md, "
                "'Building and testing')",
                file=sys.stderr,
            )

    if "compare" in stages:
        from trippy.distill.compare import audit_comparison_table, build_audit_columns

        scene_root = Path(report["scene_root"]) if report.get("scene_root") else Path(args.scene_root)
        sparse_txt_dir = scene_root / DISTILL_SPARSE_DIRNAME
        trips_export_ply = report.get("trips_export_ply") or args.trips_export_ply
        columns = build_audit_columns(
            sparse_txt_dir,
            baseline_ply=args.baseline_ply,
            trips_export_ply=trips_export_ply,
            distilled_ply=args.distilled_ply,
        )
        table = audit_comparison_table(columns)
        (out_dir / DISTILL_COMPARE_FILENAME).write_text(table + "\n")
        print("trippy distill: audit comparison")
        print(table)

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
    train.add_argument(
        "--report",
        action="store_true",
        help=(
            "after fit(), run the candidate report on the final checkpoint, audit the baseline "
            "source PLY (cached), append a baseline-vs-candidate table to the run's README.md, "
            "export a free-navigation viewer bundle + Mac launcher, and deliver the launcher + "
            "dolly.mp4 + honesty_sheet.png + export.ply via scripts/deliver.sh -- never fails the "
            "run (see REPORT_FAILED.txt if reporting itself breaks)"
        ),
    )
    train.set_defaults(func=_cmd_train)

    ev = sub.add_parser("eval", help="evaluate a checkpoint's held-out (or given) images")
    ev.add_argument("--checkpoint", required=True, help="checkpoint .pt path")
    ev.add_argument("--images", nargs="*", default=None, help="image names to evaluate (default: held-out split)")
    ev.add_argument("--device", choices=["cpu", "mps"], default=None, help="override the checkpoint's device")
    ev.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "test-time photometric calibration: fit each held-out image's own exposure to its "
            "own photo (a few dozen Adam steps, everything else frozen -- TRIPS's own "
            "optimize_eval_camera, cut down to the photometric scalars) and report the "
            "calibrated metrics next to the strict ones. Never writes back to the checkpoint."
        ),
    )
    ev.add_argument(
        "--calibrate-wb",
        action="store_true",
        help="with --calibrate, also fit red/blue white balance (green stays pinned, as in training)",
    )
    ev.add_argument(
        "--exposure-mode",
        choices=EVAL_EXPOSURE_MODES,
        default=None,
        help=(
            "which exposure/WB a held-out image's headline '_eval' numbers use: 'own' (its own "
            "never-trained exposure, the only behaviour before this feature), 'neighbours' "
            "(interpolated from the nearest TRAINING frames by capture order, never reading the "
            "held-out photo -- TRIPS's interpolate_eval_settings, NeuralCamera.cpp:481-520), or "
            "'calibrate' (the --calibrate fit above, promoted to the headline number). Default: "
            "the checkpoint's own cfg.eval_exposure_mode ('neighbours' unless the run set it "
            "explicitly)."
        ),
    )
    ev.set_defaults(func=_cmd_eval)

    hybrid_c = sub.add_parser("hybrid-c", help="Design C: render->photo U-Net refinement")
    hybrid_c_sub = hybrid_c.add_subparsers(dest="hybrid_c_command", required=True)

    hc_train = hybrid_c_sub.add_parser("train", help="train a HybridCConfig YAML config")
    hc_train.add_argument("--config", required=True, help="path to a HybridCConfig YAML file")
    hc_train.add_argument("--resume", default=None, help="checkpoint .pt path to resume from")
    hc_train.add_argument("--max-minutes", type=float, default=None, help="wall-clock budget override")
    hc_train.add_argument("--device", choices=["cpu", "mps"], default=None, help="override the config's device")
    hc_train.set_defaults(func=_cmd_hybrid_c_train)

    hc_eval = hybrid_c_sub.add_parser("eval", help="evaluate a hybrid-c checkpoint's held-out (or given) images")
    hc_eval.add_argument("--checkpoint", required=True, help="checkpoint .pt path")
    hc_eval.add_argument("--images", nargs="*", default=None, help="image names to evaluate (default: held-out split)")
    hc_eval.add_argument("--device", choices=["cpu", "mps"], default=None, help="override the checkpoint's device")
    hc_eval.set_defaults(func=_cmd_hybrid_c_eval)

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

    points_build = sub.add_parser(
        "points-build",
        help="build any PointSourceConfig-described source (gaussian/colmap/union/npz) to .npz + summary JSON",
    )
    points_build.add_argument("--config", required=True, help="YAML file with PointSourceConfig fields at the root")
    points_build.add_argument("--out", required=True, help="output .npz path (summary JSON written alongside it)")
    points_build.set_defaults(func=_cmd_points_build)

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

    export_bundle = sub.add_parser(
        "export-bundle",
        help="write a self-contained bundle dir (bundle.json + points.npz + weights.safetensors)",
    )
    export_bundle.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "TRIPS/ADOP checkpoint dir (params.ini + ep<NNNN>/, needs --scene), or a "
            "trippy-native checkpoint .pt / run directory"
        ),
    )
    export_bundle.add_argument(
        "--scene", default=None, help="ADOP scene directory (TRIPS checkpoints only)"
    )
    export_bundle.add_argument(
        "--epoch",
        default=None,
        help=f"epoch subdirectory name (TRIPS only; default {TRIPS_DEFAULT_EPOCH}, else the newest ep*/)",
    )
    export_bundle.add_argument("--out", required=True, help="bundle directory to write")
    export_bundle.add_argument("--name", default=None, help="scene label written into bundle.json")
    export_bundle.set_defaults(func=_cmd_export_bundle)

    bundle_launcher = sub.add_parser(
        "bundle-launcher",
        help="export a bundle + generate/deliver a free-navigation Mac viewer launcher, for any checkpoint",
    )
    bundle_launcher.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "TRIPS/ADOP checkpoint dir (params.ini + ep<NNNN>/, needs --scene), or a "
            "trippy-native checkpoint .pt / run directory"
        ),
    )
    bundle_launcher.add_argument("--scene", default=None, help="ADOP scene directory (TRIPS checkpoints only)")
    bundle_launcher.add_argument(
        "--epoch",
        default=None,
        help=f"epoch subdirectory name (TRIPS only; default {TRIPS_DEFAULT_EPOCH}, else the newest ep*/)",
    )
    bundle_launcher.add_argument("--name", required=True, help="bundle label, launcher name, and delivery name")
    bundle_launcher.add_argument(
        "--out",
        default=None,
        help="bundle directory to write (default: <run_dir>/bundle when detectable, else alongside the checkpoint)",
    )
    bundle_launcher.set_defaults(func=_cmd_bundle_launcher)

    leaderboard = sub.add_parser(
        "leaderboard",
        help="scan every run with a self-report and write one comparison table (markdown + PNG)",
    )
    leaderboard.add_argument(
        "--out",
        default=None,
        help="output directory (default: $TRIPPY_OUTPUT/leaderboard)",
    )
    leaderboard.add_argument(
        "--deliver",
        action="store_true",
        help="also deliver the PNG via scripts/deliver.sh under the fixed 'trips-leaderboard' name",
    )
    leaderboard.set_defaults(func=_cmd_leaderboard)

    prune_run = sub.add_parser(
        "prune-run",
        help="apply the checkpoint retention policy to an existing run directory's checkpoints/",
    )
    prune_run.add_argument("run_dir", help="run directory (the one containing checkpoints/)")
    prune_run.add_argument(
        "--dry-run", action="store_true", help="print what would be deleted (and bytes freed) without deleting"
    )
    prune_run.add_argument(
        "--keep-every",
        type=int,
        default=TRAIN_DEFAULT_CHECKPOINT_KEEP_EVERY,
        help="keep every Nth epoch checkpoint (default matches TrainConfig.checkpoint_keep_every)",
    )
    prune_run.add_argument(
        "--keep-last",
        type=int,
        default=TRAIN_DEFAULT_CHECKPOINT_KEEP_LAST,
        help="keep this many of the most recent epoch checkpoints (default matches TrainConfig.checkpoint_keep_last)",
    )
    prune_run.add_argument(
        "--protect-seconds",
        type=float,
        default=PRUNE_RUN_DEFAULT_PROTECT_SECONDS,
        help="never delete a file modified more recently than this many seconds ago (default 120s)",
    )
    prune_run.set_defaults(func=_cmd_prune_run)

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

    distill = sub.add_parser(
        "distill",
        help="design-B pipeline: distil a TRIPS checkpoint into a plain-Gaussian PLY via Brush",
    )
    distill.add_argument("--checkpoint", default=None, help="checkpoint .pt path (required for --stage render/all)")
    distill.add_argument("--out", required=True, help="output directory")
    distill.add_argument(
        "--stage",
        choices=["render", "brush-cmd", "compare", "all"],
        default="all",
        help="which pipeline step(s) to run (see trippy.cli module docstring)",
    )
    distill.add_argument("--device", choices=["cpu", "mps"], default=None, help="--stage render/all only")
    distill.add_argument("--interp-k", type=int, default=DISTILL_DEFAULT_INTERP_K, help="near-path cameras per consecutive pair")
    distill.add_argument(
        "--max-jump-multiplier",
        type=float,
        default=DISTILL_MAX_JUMP_MULTIPLIER,
        help="honesty guard: skip a consecutive pair further apart than this x the median pair distance",
    )
    distill.add_argument(
        "--max-init-points",
        type=int,
        default=DISTILL_DEFAULT_MAX_INIT_POINTS,
        help="cap on points3D.txt rows written from the TRIPS export (None via -1 writes every point)",
    )
    distill.add_argument("--brush-binary", default=None, help="override the auto-detected brush-cli/brush binary path")
    distill.add_argument("--brush-iters", type=int, default=DISTILL_DEFAULT_BRUSH_ITERS, help="--total-train-iters for the printed brush command")
    distill.add_argument("--max-resolution", type=int, default=None, help="--stage brush-cmd only, when not preceded by --stage render in this invocation")
    distill.add_argument("--job-name", default=None, help="GPU-queue job name for --stage brush-cmd (default: distill-<out dirname>)")
    distill.add_argument("--baseline-ply", default=None, help="--stage compare: the training run's own source PLY (e.g. kkc_15000.ply)")
    distill.add_argument("--trips-export-ply", default=None, help="--stage compare only (without a preceding render stage): override the TRIPS export ply path")
    distill.add_argument("--distilled-ply", default=None, help="--stage compare: the Brush-trained output PLY, once training has finished")
    distill.add_argument("--scene-root", default=None, help="--stage compare only (without a preceding render stage): the scene root for sparse_txt/")
    distill.set_defaults(func=_cmd_distill)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
