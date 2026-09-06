"""Render a `trippy-bundle-1` bundle in Python, exactly as `trips-viewer` does.

Module: trippy.render.bundle_render
Purpose: the bundle directory is the interface between Python and the Rust
    viewer (docs/decisions/ADR-0006). Until now the only thing that could
    render one was the viewer itself, so "is the viewer wrong, or is the
    checkpoint?" could not be answered without opening a window and looking --
    which is forbidden for Jordan's scenes (AGENTS.md Sec. 6). This module
    renders a bundle from its own three files with trippy's reference
    rasteriser + U-Net + tone mapper, so the two paths can be compared as
    NUMBERS: PSNR, per-channel brightness, saturated-pixel fraction.
Invariants:
    - Reads ONLY `bundle.json`, `points.npz` and `weights.safetensors`. It
      never opens a checkpoint, a scene or a photograph, so what it renders is
      exactly what a viewer shipped to someone else would render.
    - The camera it builds is byte-for-byte the one
      `trips_viewer::camera::Controller::render_camera` builds for a *pinned*
      view at `--scale 1.0`: the view's own `fx/fy/cx/cy/R/t`, unmodified.
      At other scales it reproduces the same rescaling rule (focal and cx by
      `width/ref_width`, cy by `height/ref_height`).
    - 8-bit conversion is `(clamp(v, 0, 1) * 255 + 0.5) as u8`, byte-identical
      to `brush_pyramid::png::feature_to_rgb8` with `scale = 1.0`, so a PSNR
      between the two PNGs measures the render and not the encoder.
    - Lens distortion is data the Rust projection kernel applies and
      `trippy.raster.pyramid.render_pyramid` does not, so a bundle with a
      non-zero distortion vector is REFUSED rather than silently rendered from
      a different camera model.
    - Nothing here displays an image. `stats()` returns numbers only.
Units: brightness values are display-referred in [0, 1]; PSNR is dB over
    8-bit RGB; `saturated` / `black` are fractions of all (pixel, channel)
    samples.
Related docs: docs/decisions/ADR-0006-viewer-integration.md ("the bundle
    format is the interface"), trippy/render/bundle.py (the writer),
    rust/crates/trips-viewer/src/{bundle,renderer,camera}.rs (the reader),
    docs/LIMITATIONS.md ("One frame index for the tone mapper").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trippy.net.camera_model import NeuralCamera, NeuralCameraConfig
from trippy.net.export_safetensors import read_safetensors
from trippy.net.unet import MultiScaleUnet2dDecOnlySmallFixed, NetworkConfig
from trippy.raster.pyramid import render_pyramid
from trippy.render.bundle import (
    BUNDLE_JSON_FILENAME,
    BUNDLE_POINTS_FILENAME,
    BUNDLE_WEIGHTS_FILENAME,
)

#: Largest absolute distortion coefficient a Python bundle render tolerates.
#: `render_pyramid` implements a plain pinhole; the Rust projection kernel
#: applies the Saiga polynomial. Anything above this would compare two
#: different camera models and call the difference a viewer bug.
MAX_DISTORTION = 1e-9

#: Percentiles reported per channel by `stats`. p50 is the honest "how bright
#: is this frame" number; p01/p99 say whether the tails are clipped.
STATS_PERCENTILES = (1.0, 50.0, 99.0)

#: A sample at or above this counts as saturated (255/255 after rounding needs
#: >= 254.5/255).
SATURATED_AT = 254.5 / 255.0

#: A sample at or below this counts as crushed black (0/255 after rounding).
BLACK_AT = 0.5 / 255.0


@dataclass(frozen=True)
class LoadedBundle:
    """A bundle directory, loaded and ready to render.

    Attributes:
        directory: where it came from.
        manifest: the parsed `bundle.json` document.
        xyz, size, feat, conf: the point arrays, `(N, 3)`, `(N,)`, `(N, C)`,
            `(N,)`, float32 torch tensors on the chosen device.
        background: `(C,)` background feature, or None for a zero background.
        net: the U-Net with the bundle's weights loaded, in eval mode.
        camera: the tone mapper with the bundle's parameters, in eval mode.
        metadata: the safetensors `__metadata__` block.
    """

    directory: Path
    manifest: dict[str, Any]
    xyz: torch.Tensor
    size: torch.Tensor
    feat: torch.Tensor
    conf: torch.Tensor
    background: torch.Tensor | None
    net: MultiScaleUnet2dDecOnlySmallFixed
    camera: NeuralCamera | None
    metadata: dict[str, str]

    def view_position(self, dataset_index: int | None) -> int:
        """Array position in `manifest["views"]` for a dataset image index.

        `None` gives the bundle's own `default_view`, which is already a
        position (see `trippy.render.bundle.default_view_position`).

        Raises:
            ValueError: if no view carries `dataset_index`.
        """
        views = self.manifest["views"]
        if dataset_index is None:
            return int(self.manifest.get("default_view", 0))
        for position, view in enumerate(views):
            if int(view["index"]) == dataset_index:
                return position
        raise ValueError(f"no view with dataset index {dataset_index} in {self.directory}")


def build_net(tensors: dict[str, np.ndarray], metadata: dict[str, str]) -> MultiScaleUnet2dDecOnlySmallFixed:
    """Rebuild the U-Net described by a `trippy-unet-1` file and load its weights.

    The inverse of `trippy.net.export_safetensors.unet_tensors`; `up.{k}` is
    read back in the same application order it was written in (see that
    module's KEY schema).

    Args:
        tensors: the file's tensors, by name.
        metadata: the file's `__metadata__` block.

    Returns:
        The network, in eval mode, on CPU.

    Raises:
        ValueError: on a missing tensor or a shape the config does not imply.
    """
    config = NetworkConfig(
        num_input_channels=int(metadata["in_channels"]),
        num_output_channels=int(metadata["out_channels"]),
        filters=int(metadata["filters"]),
        num_layers=int(metadata["num_layers"]),
        activation=metadata["activation"],
        norm=metadata["norm"],
        upsample_mode=metadata["upsample_mode"],
        last_act=metadata["last_act"],
    )
    net = MultiScaleUnet2dDecOnlySmallFixed(config)
    state: dict[str, torch.Tensor] = {}

    def take(key: str) -> torch.Tensor:
        if key not in tensors:
            raise ValueError(f"weight file is missing {key!r}")
        return torch.from_numpy(np.ascontiguousarray(tensors[key], dtype=np.float32))

    state["start.conv.feature_conv.weight"] = take("unet.start.feature.weight")
    state["start.conv.feature_conv.bias"] = take("unet.start.feature.bias")
    state["start.conv.gate_conv.weight"] = take("unet.start.gate.weight")
    state["start.conv.gate_conv.bias"] = take("unet.start.gate.bias")
    for k in range(config.num_layers - 1):
        state[f"up.{k}.conv.feature_conv.weight"] = take(f"unet.up.{k}.feature.weight")
        state[f"up.{k}.conv.feature_conv.bias"] = take(f"unet.up.{k}.feature.bias")
        state[f"up.{k}.conv.gate_conv.weight"] = take(f"unet.up.{k}.gate.weight")
        state[f"up.{k}.conv.gate_conv.bias"] = take(f"unet.up.{k}.gate.bias")
    state["final.0.weight"] = take("unet.final.weight")
    state["final.0.bias"] = take("unet.final.bias")

    missing, unexpected = net.load_state_dict(state, strict=False)
    # `strict=False` because the upsample module may hold non-learned buffers;
    # anything genuinely absent is a schema break and must be loud.
    if unexpected:
        raise ValueError(f"weight file has tensors the network does not want: {sorted(unexpected)}")
    real_missing = [k for k in missing if k.endswith(("weight", "bias"))]
    if real_missing:
        raise ValueError(f"network parameters not present in the weight file: {sorted(real_missing)}")
    return net.eval()


def build_camera(tensors: dict[str, np.ndarray], metadata: dict[str, str]) -> NeuralCamera | None:
    """Rebuild the tone mapper described by a `trippy-unet-1` file, or None.

    The inverse of `trippy.net.export_safetensors.camera_tensors`. Returns
    None when the file's `has_camera` is "0", which is what the U-Net-only
    fixtures write.
    """
    if metadata.get("has_camera", "0") != "1":
        return None
    config = NeuralCameraConfig(
        enable_exposure=metadata["enable_exposure"] == "1",
        enable_white_balance=metadata["enable_white_balance"] == "1",
        enable_vignette=metadata["enable_vignette"] == "1",
        enable_response=metadata["enable_response"] == "1",
        response_params=int(metadata["response_params"]),
    )
    num_frames = int(metadata["num_frames"])
    camera = NeuralCamera(
        image_height=int(metadata["image_height"]),
        image_width=int(metadata["image_width"]),
        num_frames=num_frames,
        config=config,
    )
    with torch.no_grad():
        if camera.exposures_values is not None:
            camera.exposures_values.copy_(
                torch.from_numpy(tensors["camera.exposure"]).view(num_frames, 1, 1, 1)
            )
        if camera.white_balance_values is not None:
            wb = torch.from_numpy(tensors["camera.white_balance"]).view(num_frames, 3, 1, 1)
            camera.white_balance_values.copy_(wb)
            camera.white_balance_reference.copy_(wb)
        if camera.vignette_net is not None:
            camera.vignette_net.vignette_params.copy_(
                torch.from_numpy(tensors["camera.vignette_params"]).view(3)
            )
            camera.vignette_net.vignette_center.copy_(
                torch.from_numpy(tensors["camera.vignette_center"]).view(1, 2, 1, 1)
            )
        if camera.camera_response is not None:
            response = torch.from_numpy(tensors["camera.response"])
            camera.camera_response.response.copy_(
                response.view(1, response.shape[0], 1, response.shape[1])
            )
    return camera.eval()


def load_bundle(directory: str | Path, device: torch.device | str = "cpu") -> LoadedBundle:
    """Load a bundle directory: manifest, points, network, tone mapper.

    Args:
        directory: the directory holding `bundle.json`.
        device: where the point tensors and modules go.

    Returns:
        A `LoadedBundle`.

    Raises:
        ValueError: if the manifest's `num_points`/`num_channels` disagree
            with `points.npz`, or a required file is absent.
    """
    directory = Path(directory)
    manifest = json.loads((directory / BUNDLE_JSON_FILENAME).read_text())
    device = torch.device(device)

    with np.load(directory / manifest.get("points", BUNDLE_POINTS_FILENAME)) as npz:
        arrays = {key: np.asarray(npz[key], dtype=np.float32) for key in ("xyz", "size", "feat", "conf")}
    n, channels = arrays["feat"].shape
    if n != int(manifest["num_points"]) or channels != int(manifest["num_channels"]):
        raise ValueError(
            f"{directory}: points.npz holds {n} x {channels} but bundle.json says "
            f"{manifest['num_points']} x {manifest['num_channels']}"
        )

    tensors, metadata = read_safetensors(directory / manifest.get("weights", BUNDLE_WEIGHTS_FILENAME))
    background = manifest.get("background") or None
    camera = build_camera(tensors, metadata)
    return LoadedBundle(
        directory=directory,
        manifest=manifest,
        xyz=torch.from_numpy(arrays["xyz"]).to(device),
        size=torch.from_numpy(arrays["size"]).to(device),
        feat=torch.from_numpy(arrays["feat"]).to(device),
        conf=torch.from_numpy(arrays["conf"]).to(device),
        background=(
            torch.tensor(background, dtype=torch.float32, device=device) if background else None
        ),
        net=build_net(tensors, metadata).to(device),
        camera=None if camera is None else camera.to(device),
        metadata=metadata,
    )


def view_camera(view: dict[str, Any], scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """`(K, R, t, width, height)` for a bundle view at a render scale.

    Reproduces `Controller::render_camera` for a *pinned* camera: at
    `scale == 1.0` the view's own intrinsics verbatim; otherwise focal length
    and `cx` scale with the width and `cy` with the height, which keeps the
    field of view the view's own.
    """
    width = max(16, round(view["width"] * scale))
    height = max(16, round(view["height"] * scale))
    if width == view["width"] and height == view["height"]:
        fx, fy, cx, cy = view["fx"], view["fy"], view["cx"], view["cy"]
    else:
        x_scale = width / view["width"]
        fx, fy = view["fx"] * x_scale, view["fy"] * x_scale
        cx, cy = view["cx"] * x_scale, view["cy"] * (height / view["height"])
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    R = torch.tensor(view["R"], dtype=torch.float32).reshape(3, 3)
    t = torch.tensor(view["t"], dtype=torch.float32).reshape(3)
    return K, R, t, width, height


def render_view(
    bundle: LoadedBundle, position: int, scale: float = 1.0
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Render one of the bundle's own views, tone mapper included.

    Args:
        bundle: from `load_bundle`.
        position: ARRAY POSITION in `manifest["views"]` (not the dataset
            index -- use `LoadedBundle.view_position` to convert).
        scale: render at this fraction of the view's own size, matching
            `trips-viewer --scale`.

    Returns:
        `(rgb, info)`: `rgb` is `(3, H, W)` float32 display-referred, exactly
        what the viewer would blit; `info` records the frame index, the
        exposure applied and the coverage (`1 - t_final`) mean.

    Raises:
        ValueError: if the view carries a lens distortion this path cannot
            apply (see `MAX_DISTORTION`).
    """
    view = bundle.manifest["views"][position]
    distortion = np.asarray(view.get("distortion", []), dtype=np.float64)
    if distortion.size and np.abs(distortion).max() > MAX_DISTORTION:
        raise ValueError(
            f"view {view['index']} ({view['name']}) has lens distortion "
            f"{distortion.tolist()}; the Python reference renders a plain pinhole and "
            "would not be comparing the same camera model. Undistort first."
        )

    params = bundle.manifest["params"]
    device = bundle.xyz.device
    K, R, t, width, height = view_camera(view, scale)
    with torch.no_grad():
        layers, aux = render_pyramid(
            bundle.xyz,
            bundle.size,
            bundle.feat,
            bundle.conf,
            K.to(device),
            R.to(device),
            t.to(device),
            (height, width),
            num_layers=int(params["num_layers"]),
            mode=str(params["mode"]),
            bg=bundle.background,
            max_frags=int(params["max_frags"]),
            t_cutoff=float(params["t_cutoff"]),
            alpha_min=float(params["alpha_min"]),
            znear=float(params["znear"]),
            pixel_center=str(params["pixel_center"]),
            pyramid_halving=str(params["halving"]),
        )
        net_out = bundle.net([layer.unsqueeze(0) for layer in layers])
        frame_index = int(view["index"])
        if bundle.camera is not None:
            index = torch.tensor([frame_index], device=device, dtype=torch.long)
            rgb = bundle.camera(net_out, index)
        else:
            rgb = net_out.clamp(0.0, 1.0)
    exposure = None
    if bundle.camera is not None and bundle.camera.exposures_values is not None:
        exposure = float(bundle.camera.exposures_values[frame_index].item())
    info = {
        "view_position": position,
        "frame_index": frame_index,
        "name": view["name"],
        "width": width,
        "height": height,
        "exposure_ev": exposure,
        "exposure_gain": None if exposure is None else float(2.0**-exposure),
        # `aux["t_final"]` is one tensor per pyramid level; level 0 is the one
        # the honesty sheets call "coverage".
        "coverage_mean": float((1.0 - aux["t_final"][0]).clamp(0.0, 1.0).mean().item()),
    }
    return rgb[0].detach().float().cpu(), info


def to_rgb8(rgb: torch.Tensor | np.ndarray) -> np.ndarray:
    """`(3, H, W)` float -> `(H, W, 3)` uint8, `brush_pyramid::png`'s rounding.

    `(clamp(v, 0, 1) * 255 + 0.5) as u8`, with NaN clamping to 0 the way
    Rust's `f32::clamp` chain does.
    """
    array = rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb)
    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    array = np.clip(array, 0.0, 1.0) * 255.0 + 0.5
    return np.transpose(array.astype(np.uint8), (1, 2, 0))


def write_png(path: str | Path, rgb: torch.Tensor | np.ndarray) -> Path:
    """Write `(3, H, W)` float rgb as an 8-bit PNG. Returns the path."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_rgb8(rgb), mode="RGB").save(path)
    return path


def read_png(path: str | Path) -> np.ndarray:
    """Read an 8-bit RGB PNG as `(H, W, 3)` uint8."""
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def stats(rgb8: np.ndarray) -> dict[str, Any]:
    """Numbers-only description of a frame -- never an image.

    Args:
        rgb8: `(H, W, 3)` uint8.

    Returns:
        Per-channel mean and `STATS_PERCENTILES`, plus the fraction of
        (pixel, channel) samples that are saturated white or crushed black.
        Everything is in [0, 1], so a "nearly white frame" is mean ~0.88 with
        a large `saturated`, and a correct frame is mean ~0.3-0.5.
    """
    values = rgb8.astype(np.float64) / 255.0
    out: dict[str, Any] = {
        "size": [int(rgb8.shape[1]), int(rgb8.shape[0])],
        "mean": [float(values[..., c].mean()) for c in range(3)],
        "saturated": float((values >= SATURATED_AT).mean()),
        "black": float((values <= BLACK_AT).mean()),
    }
    for percentile in STATS_PERCENTILES:
        out[f"p{int(percentile):02d}"] = [
            float(np.percentile(values[..., c], percentile)) for c in range(3)
        ]
    return out


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB between two `(H, W, 3)` uint8 images (inf when identical)."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mse = float(np.mean((a.astype(np.float64) / 255.0 - b.astype(np.float64) / 255.0) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(-10.0 * np.log10(mse))
