#!/usr/bin/env python
"""Export trippy's U-Net + tone mapper (and their parity fixtures) for the Rust port.

Module: tools.export_unet_safetensors
Purpose: the single producer of every file `rust/crates/brush-unet` reads.
    Three subcommands:

      fixture     random, tiny, COMMITTED. Writes
                  tests/fixtures/synthetic/unet_fixture_small/ -- random
                  weights (num_layers=5, filters=32, C=4, O=3), a random
                  32x24 pyramid, and PyTorch's own U-Net and camera outputs
                  for it. This is the 1e-4 parity target for the Burn port.
      checkpoint  the PUBLIC Zenodo horse checkpoint's real weights
                  (num_layers=8) plus a JSON of the per-image exposure /
                  white-balance values. NOT committed (goes to $TRIPPY_OUTPUT).
      horse-e2e   one view's camera-space point set, camera JSON and the
                  parity engine's own final RGB frame, so the Rust end-to-end
                  test can compare pyramid -> U-Net -> tone map against
                  `trippy.render.parity`. NOT committed (large).

Invariants:
    - `fixture` is SYNTHETIC ONLY and deterministic from a pinned seed
      (AGENTS.md section 6); nothing derived from a scene or a checkpoint is
      ever written under tests/.
    - `checkpoint` / `horse-e2e` only ever touch the public Zenodo Tanks &
      Temples release, and only write under $TRIPPY_OUTPUT.
    - float32 everywhere: the Rust port is float32-only, so a float64
      reference would compare two different computations.
    - CPU only. No MPS/GPU work happens here (AGENTS.md section 6).
Related docs: rust/README.md ("brush-unet weight schema"),
    docs/TRIPS_REFERENCE.md Sec. 5/6.

Usage:
    PYTHONPATH=. TRIPS_DEVICE=cpu python tools/export_unet_safetensors.py fixture
    PYTHONPATH=. TRIPS_DEVICE=cpu python tools/export_unet_safetensors.py checkpoint
    PYTHONPATH=. TRIPS_DEVICE=cpu python tools/export_unet_safetensors.py horse-e2e --index 8
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from trippy.net.camera_model import NeuralCamera, NeuralCameraConfig
from trippy.net.export_safetensors import export, write_safetensors
from trippy.net.unet import MultiScaleUnet2dDecOnlySmallFixed, NetworkConfig

# --- fixture geometry (small on purpose; see tests/test_net_export_safetensors.py) ---
#: Seed for every random tensor in the committed fixture. Changing it rewrites
#: the fixture and invalidates the Rust parity test's recorded numbers.
FIXTURE_SEED = 20260906
FIXTURE_WIDTH = 32
FIXTURE_HEIGHT = 24
FIXTURE_LAYERS = 5
FIXTURE_CHANNELS = 4
FIXTURE_FILTERS = 32
FIXTURE_FRAMES = 3
#: Which frame index the camera fixture is evaluated at (exercises the
#: per-image exposure/white-balance lookup rather than row 0's identity).
FIXTURE_FRAME_INDEX = 1
#: Public Zenodo horse checkpoint: 8 pyramid levels (docs/TRIPS_REFERENCE.md Sec. 5).
HORSE_NUM_LAYERS = 8


def fixture_shapes(height: int, width: int, num_layers: int) -> list[tuple[int, int]]:
    """Pyramid layer shapes with TRIPS's `ceil` halving, finest first."""
    shapes = []
    h, w = int(height), int(width)
    for _ in range(num_layers):
        shapes.append((h, w))
        h = -(-h // 2)
        w = -(-w // 2)
    return shapes


def randomise(module: torch.nn.Module, generator: torch.Generator) -> None:
    """Fill every parameter with N(0, 0.15) so no branch is accidentally dead.

    The default PyTorch init already differs per layer; a single explicit
    distribution makes the fixture reproducible from `FIXTURE_SEED` alone and
    keeps the gated block's sigmoid away from its saturated tails (where a
    parity bug would be invisible).
    """
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.15)


def build_fixture(out_dir: Path) -> dict:
    """Write the committed random-weight fixture. Returns its meta dict."""
    generator = torch.Generator().manual_seed(FIXTURE_SEED)

    net = MultiScaleUnet2dDecOnlySmallFixed(
        NetworkConfig(
            num_layers=FIXTURE_LAYERS,
            num_input_channels=FIXTURE_CHANNELS,
            filters=FIXTURE_FILTERS,
        )
    )
    randomise(net, generator)
    net.eval()

    camera = NeuralCamera(
        image_height=FIXTURE_HEIGHT,
        image_width=FIXTURE_WIDTH,
        num_frames=FIXTURE_FRAMES,
        config=NeuralCameraConfig(),
    )
    # Deliberately non-trivial tone mapper: a real checkpoint's exposure is
    # O(0.1) EV, its WB gains are near 1 (green pinned to 1 by
    # ApplyConstraints) and its vignette is a small negative falloff.
    with torch.no_grad():
        camera.exposures_values.copy_(
            torch.tensor([0.0, 0.37, -0.22]).view(FIXTURE_FRAMES, 1, 1, 1)
        )
        wb = torch.tensor([[1.0, 1.0, 1.0], [1.08, 1.0, 0.93], [0.95, 1.0, 1.11]])
        camera.white_balance_values.copy_(wb.view(FIXTURE_FRAMES, 3, 1, 1))
        camera.vignette_net.vignette_params.copy_(torch.tensor([-0.21, 0.07, -0.013]))
        camera.vignette_net.vignette_center.copy_(torch.tensor([0.04, -0.03]).view(1, 2, 1, 1))
        # Perturb the gamma-initialised LUT so it is not a pure power curve.
        camera.camera_response.response.add_(
            torch.randn(camera.camera_response.response.shape, generator=generator) * 0.01
        )
    camera.eval()

    shapes = fixture_shapes(FIXTURE_HEIGHT, FIXTURE_WIDTH, FIXTURE_LAYERS)
    inputs = [
        torch.randn((1, FIXTURE_CHANNELS, h, w), generator=generator) * 0.5 for (h, w) in shapes
    ]
    with torch.no_grad():
        raw = net(inputs)
        frame_index = torch.tensor([FIXTURE_FRAME_INDEX], dtype=torch.long)
        rgb = camera(raw, frame_index)
        # Second camera input, independent of the U-Net, so the camera parity
        # test exercises values below 0 and above 1 (the LUT's clamped ends).
        camera_probe = torch.randn((1, 3, FIXTURE_HEIGHT, FIXTURE_WIDTH), generator=generator) * 0.6 + 0.4
        camera_probe_out = camera(camera_probe, frame_index)

    out_dir.mkdir(parents=True, exist_ok=True)
    export(
        net,
        camera,
        out_dir / "weights.safetensors",
        extra_metadata={"source": "tools/export_unet_safetensors.py fixture", "seed": str(FIXTURE_SEED)},
    )

    io_tensors: dict[str, np.ndarray] = {}
    for level, tensor in enumerate(inputs):
        io_tensors[f"input.{level}"] = tensor.numpy()
    io_tensors["unet_out"] = raw.numpy()
    io_tensors["rgb_out"] = rgb.numpy()
    io_tensors["camera_probe"] = camera_probe.numpy()
    io_tensors["camera_probe_out"] = camera_probe_out.numpy()
    write_safetensors(
        io_tensors,
        out_dir / "io.safetensors",
        {
            "format": "trippy-unet-io-1",
            "num_layers": str(FIXTURE_LAYERS),
            "frame_index": str(FIXTURE_FRAME_INDEX),
            "height": str(FIXTURE_HEIGHT),
            "width": str(FIXTURE_WIDTH),
        },
    )

    meta = {
        "seed": FIXTURE_SEED,
        "num_layers": FIXTURE_LAYERS,
        "filters": FIXTURE_FILTERS,
        "in_channels": FIXTURE_CHANNELS,
        "out_channels": 3,
        "num_frames": FIXTURE_FRAMES,
        "frame_index": FIXTURE_FRAME_INDEX,
        "height": FIXTURE_HEIGHT,
        "width": FIXTURE_WIDTH,
        "layer_shapes": [list(s) for s in shapes],
        "parameter_count": net.parameter_count(),
        "unet_out_absmax": float(raw.abs().max()),
        "rgb_out_absmax": float(rgb.abs().max()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


# --- real-checkpoint exports (public Zenodo horse) ---------------------------


def _load_horse(checkpoint: Path, scene: Path, epoch: str):
    """Load the horse checkpoint + scene through trippy's own parity code path."""
    from trippy.net.checkpoint import build_neural_camera, load_trips_scene_checkpoint
    from trippy.render.parity import (
        _scene_name_from_params,
        build_network,
        build_scene_points,
        resolve_intrinsics,
        resolve_pose,
    )
    from trippy.scene.adop_io import load_adop_scene

    adop = load_adop_scene(scene)
    scene_name = _scene_name_from_params(checkpoint, None)
    ckpt = load_trips_scene_checkpoint(checkpoint / epoch, scene_name)
    points = build_scene_points(ckpt, torch.device("cpu"))
    net, _ = build_network(checkpoint, epoch, HORSE_NUM_LAYERS, int(points.feat.shape[1]))
    return adop, ckpt, points, net, resolve_pose, resolve_intrinsics, build_neural_camera


#: Views EXP-0002 measured parity on: the checkpoint's own held-out test split
#: entries 8, 120 and 144 (00009.jpg / 00121.jpg / 00145.jpg).
PARITY_INDICES = (8, 120, 144)


def export_checkpoint(
    checkpoint: Path, scene: Path, epoch: str, out: Path, indices: tuple[int, ...] = PARITY_INDICES
) -> dict:
    """Export the horse checkpoint's U-Net + camera; also build the per-image info."""
    adop, ckpt, points, net, _pose, _intr, build_neural_camera = _load_horse(checkpoint, scene, epoch)
    first = adop.view(0)
    camera = build_neural_camera(ckpt.camera, first.height, first.width)
    out.parent.mkdir(parents=True, exist_ok=True)
    export(
        net,
        camera,
        out,
        extra_metadata={
            "source": str(checkpoint / epoch),
            "scene": str(scene),
            "num_points": str(len(points)),
        },
    )
    info = {
        "checkpoint": str(checkpoint / epoch),
        "num_layers": HORSE_NUM_LAYERS,
        "num_input_channels": int(points.feat.shape[1]),
        "parameter_count": net.parameter_count(),
        "num_frames": int(ckpt.camera.exposure.shape[0]),
        "image_height": first.height,
        "image_width": first.width,
        "background_color": [float(v) for v in ckpt.texture.background_color()],
        # The tone mapper's per-image state for the parity frames, in the
        # form the Rust `NeuralCamera` indexes it by (`frame` -> ev, gains).
        "exposure_white_balance": [
            {
                "index": int(i),
                "image_name": adop.view(int(i)).image_name,
                "exposure": float(ckpt.camera.exposure[int(i)]),
                "white_balance": [float(v) for v in ckpt.camera.white_balance[int(i)]],
            }
            for i in indices
        ],
        "per_image": [],
    }
    return info, adop, ckpt, points, net, camera


def export_horse_e2e(
    checkpoint: Path, scene: Path, epoch: str, out_dir: Path, indices: list[int]
) -> dict:
    """Write point set / camera / expected frame for each requested view index."""
    from trippy.render.parity import (
        project_adop,
        render_trips_layers,
        resolve_intrinsics,
        resolve_pose,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    info, adop, ckpt, points, net, camera = export_checkpoint(
        checkpoint, scene, epoch, out_dir / "horse_unet.safetensors"
    )

    for index in indices:
        view = adop.view(index)
        view = resolve_pose(adop, ckpt, view)
        view = resolve_intrinsics(ckpt, view, adop.render_scale)

        K = torch.as_tensor(view.K, dtype=torch.float32)
        R = torch.as_tensor(view.R, dtype=torch.float32)
        t = torch.as_tensor(view.t, dtype=torch.float32)
        distortion = torch.as_tensor(view.distortion, dtype=torch.float32)
        ndc, _ip, z = project_adop(points.xyz, R, t, K, distortion)
        # `_render_trips_native` feeds `render_pyramid` synthetic camera-space
        # points so the pinhole projection inside lands on TRIPS's distorted
        # `ip`. brush-pyramid's `Camera` is a plain pinhole, so the SAME
        # pre-distorted points are what it must be handed, with R = I, t = 0.
        xyz_cam = torch.stack([ndc[:, 0] * z, ndc[:, 1] * z, z], dim=1)

        name = f"view_{index:05d}"
        np.savez(
            out_dir / f"{name}_points.npz",
            xyz=xyz_cam.numpy().astype(np.float32),
            size=points.size.numpy().astype(np.float32),
            feat=points.feat.numpy().astype(np.float32),
            conf=points.conf.numpy().astype(np.float32),
        )
        camera_json = {
            "width": int(view.width),
            "height": int(view.height),
            "fx": float(view.K[0, 0]),
            "fy": float(view.K[1, 1]),
            "cx": float(view.K[0, 2]),
            "cy": float(view.K[1, 2]),
            "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "t": [0.0, 0.0, 0.0],
        }
        (out_dir / f"{name}_camera.json").write_text(json.dumps(camera_json, indent=2) + "\n")
        # Everything `_render_trips_native` passes to `render_pyramid` that is
        # not the camera: brush-pyramid's `PyramidParams` fields plus the
        # background feature vector, so the Rust side never has to hardcode a
        # TRIPS threshold.
        from trippy.constants import RASTER_MAX_FRAGS, RASTER_T_CUTOFF
        from trippy.render.parity import PARITY_MIN_DEPTH

        params_json = {
            "mode": "trips",
            "num_layers": HORSE_NUM_LAYERS,
            "pixel_center": "integer",
            "halving": "ceil",
            "max_frags": RASTER_MAX_FRAGS,
            "t_cutoff": RASTER_T_CUTOFF,
            "alpha_min": 0.0,
            "znear": PARITY_MIN_DEPTH,
            "frame_index": index,
            "num_channels": int(points.feat.shape[1]),
            "background": [float(v) for v in points.bg],
        }
        (out_dir / f"{name}_params.json").write_text(json.dumps(params_json, indent=2) + "\n")

        # The parity engine's own three stages, run once each (calling
        # `render_view` and then re-running the net would rasterise and
        # convolve 1920x1080 twice for nothing).
        with torch.no_grad():
            layers, aux = render_trips_layers(
                points, view, num_layers=HORSE_NUM_LAYERS, mode="trips", engine="native"
            )
            raw = net([layer.unsqueeze(0) for layer in layers])
            rgb = camera(raw, torch.tensor([index], dtype=torch.long))
        write_safetensors(
            {
                "rgb": rgb.numpy(),
                "unet_out": raw.numpy(),
            },
            out_dir / f"{name}_expected.safetensors",
            {
                "format": "trippy-unet-e2e-1",
                "frame_index": str(index),
                "image_name": view.image_name,
                "height": str(view.height),
                "width": str(view.width),
                "num_layers": str(HORSE_NUM_LAYERS),
            },
        )
        info["per_image"].append(
            {
                "index": index,
                "image_name": view.image_name,
                "exposure": float(ckpt.camera.exposure[index]),
                "white_balance": [float(v) for v in ckpt.camera.white_balance[index]],
                "num_fragments": int(aux["num_fragments"]),
                "points_active": [int(v) for v in aux["points_active"]],
                "rgb_mean": float(rgb.mean()),
            }
        )
        print(f"{name} ({view.image_name}): fragments={aux['num_fragments']} rgb_mean={float(rgb.mean()):.5f}")

    (out_dir / "horse_meta.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def default_output() -> Path:
    return Path(os.environ.get("TRIPPY_OUTPUT", "output"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    fixture = sub.add_parser("fixture", help="write the committed random-weight fixture")
    fixture.add_argument(
        "--out", default="tests/fixtures/synthetic/unet_fixture_small", help="fixture directory"
    )

    for name in ("checkpoint", "horse-e2e"):
        cmd = sub.add_parser(name, help=f"{name} export (public Zenodo horse)")
        cmd.add_argument(
            "--checkpoint", default="third_party/zenodo/tt_checkpoints/checkpoint_horse"
        )
        cmd.add_argument("--scene", default="third_party/zenodo/scenes/tnt_scenes/tt_horse")
        cmd.add_argument("--epoch", default="ep0600")
        cmd.add_argument("--out", default=None, help="output directory (default $TRIPPY_OUTPUT/brush)")
        if name == "horse-e2e":
            cmd.add_argument("--index", default="8", help="comma-separated view indices")

    args = parser.parse_args()

    if args.command == "fixture":
        meta = build_fixture(Path(args.out))
        print(json.dumps(meta, indent=2))
        return

    out_dir = Path(args.out) if args.out else default_output() / "brush"
    checkpoint, scene = Path(args.checkpoint), Path(args.scene)
    if args.command == "checkpoint":
        info, *_ = export_checkpoint(checkpoint, scene, args.epoch, out_dir / "horse_unet.safetensors")
        (out_dir / "horse_meta.json").write_text(json.dumps(info, indent=2) + "\n")
        print(json.dumps(info, indent=2))
        return

    indices = [int(v) for v in str(args.index).split(",") if v.strip()]
    info = export_horse_e2e(checkpoint, scene, args.epoch, out_dir / "horse", indices)
    print(json.dumps({k: v for k, v in info.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
