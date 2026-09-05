"""Export trippy's U-Net + NeuralCamera to a safetensors file the Rust port reads.

Module: trippy.net.export_safetensors
Purpose: the Python half of the PyTorch -> Burn weight bridge. `rust/crates/
    brush-unet` rebuilds the same architecture from Burn primitives and needs
    every learned tensor under a *stable, self-describing* name, plus enough
    metadata (num_layers, filters, channel counts, image size, enable flags)
    to construct the modules before any tensor is bound. This module writes
    exactly that; `brush_unet::weights` is its 1:1 reader.
Invariants:
    - Every tensor is written **float32, C-contiguous**, in the declaration
      order of `KEY schema` below, so the file's data segment is contiguous
      and `safetensors` (Rust, 0.7) validates it without a re-sort.
    - `up.{k}` is indexed in **application order**: `k = 0` is the block that
      consumes the coarsest pyramid input (`inputs[num_layers - 2]`) and
      `k = num_layers - 2` is the `last=True` block that consumes
      `inputs[0]`. This is the order `MultiScaleUnet2dDecOnlySmallFixed.up`
      iterates in `forward()`, NOT the pyramid level index -- see
      `trippy.net.unet` `_up_indices`.
    - No third-party writer: safetensors' container is an 8-byte little-endian
      header length, a JSON header and a raw data segment, so it is written
      here directly rather than adding a runtime dependency for ~40 lines.
Units: exposure is in EV (applied as `x * 2**-ev`); white balance and the
    vignette polynomial are dimensionless gains; the response LUT's control
    points are display-referred values in roughly [0, 1].
Related docs: rust/README.md ("brush-unet weight schema"),
    docs/TRIPS_REFERENCE.md Sec. 5/5a (network), Sec. 6/6a (camera).

-- KEY schema (format = "trippy-unet-1") ------------------------------------
metadata (all values are strings):
    format, num_layers, filters, in_channels, out_channels,
    activation, norm, upsample_mode, last_act,
    has_camera, num_frames, response_params, image_height, image_width,
    enable_exposure, enable_white_balance, enable_vignette, enable_response

tensors (C = in_channels, F = filters, O = out_channels, L = num_layers,
         M = num_frames, P = response_params):
    unet.start.feature.weight   (F-2C, C, 3, 3)
    unet.start.feature.bias     (F-2C,)
    unet.start.gate.weight      (F-2C, C, 3, 3)
    unet.start.gate.bias        (F-2C,)
    unet.up.{k}.feature.weight  (F-2C | F-C when k == L-2, F, 3, 3)
    unet.up.{k}.feature.bias
    unet.up.{k}.gate.weight
    unet.up.{k}.gate.bias       for k in 0 .. L-2
    unet.final.weight           (O, F, 1, 1)
    unet.final.bias             (O,)
    camera.exposure             (M,)          -- only if enable_exposure
    camera.white_balance        (M, 3)        -- only if enable_white_balance
    camera.vignette_params      (3,)          -- only if enable_vignette
    camera.vignette_center      (2,)          -- only if enable_vignette
    camera.response             (O, P)        -- only if enable_response
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from trippy.net.camera_model import NeuralCamera
from trippy.net.unet import MultiScaleUnet2dDecOnlySmallFixed

#: Value of the `format` metadata key. Bump when the schema changes shape.
EXPORT_FORMAT = "trippy-unet-1"

#: safetensors pads its JSON header with spaces up to this alignment.
_HEADER_ALIGN = 8

#: safetensors dtype string for the only dtype this exporter writes.
_DTYPE = "F32"


def write_safetensors(
    tensors: Mapping[str, np.ndarray], path: str | Path, metadata: Mapping[str, str] | None = None
) -> Path:
    """Write `tensors` (float32, any shape) to `path` in safetensors format.

    Args:
        tensors: insertion-ordered mapping of name -> array. Every array is
            cast to float32 and made C-contiguous.
        path: destination file; parent directories are created.
        metadata: optional string->string map stored under `__metadata__`.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    blobs: list[bytes] = []
    offset = 0
    for name, array in tensors.items():
        arr = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
        blob = arr.tobytes(order="C")
        header[name] = {
            "dtype": _DTYPE,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)
        blobs.append(blob)

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(header_json)) % _HEADER_ALIGN
    header_json += b" " * pad

    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_json)))
        fh.write(header_json)
        for blob in blobs:
            fh.write(blob)
    return path


def read_safetensors(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Inverse of `write_safetensors` (float32 only). Returns (tensors, metadata)."""
    raw = Path(path).read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    data = raw[8 + header_len :]
    metadata = {str(k): str(v) for k, v in header.pop("__metadata__", {}).items()}
    tensors: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if info["dtype"] != _DTYPE:
            raise ValueError(f"{name}: only {_DTYPE} is supported, got {info['dtype']}")
        begin, end = info["data_offsets"]
        tensors[name] = np.frombuffer(data[begin:end], dtype=np.float32).reshape(info["shape"]).copy()
    return tensors, metadata


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().to(torch.float32).contiguous().numpy()


def unet_tensors(unet: MultiScaleUnet2dDecOnlySmallFixed) -> dict[str, np.ndarray]:
    """Every learned tensor of the U-Net, under the `unet.*` key schema."""
    out: dict[str, np.ndarray] = {}
    start = unet.start.conv
    out["unet.start.feature.weight"] = _np(start.feature_conv.weight)
    out["unet.start.feature.bias"] = _np(start.feature_conv.bias)
    out["unet.start.gate.weight"] = _np(start.gate_conv.weight)
    out["unet.start.gate.bias"] = _np(start.gate_conv.bias)
    for k, block in enumerate(unet.up):
        conv = block.conv
        out[f"unet.up.{k}.feature.weight"] = _np(conv.feature_conv.weight)
        out[f"unet.up.{k}.feature.bias"] = _np(conv.feature_conv.bias)
        out[f"unet.up.{k}.gate.weight"] = _np(conv.gate_conv.weight)
        out[f"unet.up.{k}.gate.bias"] = _np(conv.gate_conv.bias)
    final = unet.final[0]
    out["unet.final.weight"] = _np(final.weight)
    out["unet.final.bias"] = _np(final.bias)
    return out


def camera_tensors(camera: NeuralCamera) -> dict[str, np.ndarray]:
    """Every learned tensor of the tone mapper, under the `camera.*` key schema.

    Shapes are squeezed to their meaningful rank (PyTorch keeps broadcast
    singleton dims the Rust side re-adds itself): exposure `(M,)`, white
    balance `(M, 3)`, vignette centre `(2,)`, response `(O, P)`.
    """
    out: dict[str, np.ndarray] = {}
    if camera.exposures_values is not None:
        out["camera.exposure"] = _np(camera.exposures_values).reshape(-1)
    if camera.white_balance_values is not None:
        out["camera.white_balance"] = _np(camera.white_balance_values).reshape(-1, 3)
    if camera.vignette_net is not None:
        out["camera.vignette_params"] = _np(camera.vignette_net.vignette_params).reshape(-1)
        out["camera.vignette_center"] = _np(camera.vignette_net.vignette_center).reshape(-1)
    if camera.camera_response is not None:
        response = _np(camera.camera_response.response)  # (1, O, 1, P)
        out["camera.response"] = response.reshape(response.shape[1], response.shape[3])
    return out


def build_metadata(
    unet: MultiScaleUnet2dDecOnlySmallFixed, camera: NeuralCamera | None
) -> dict[str, str]:
    """The `__metadata__` block; see the module docstring's KEY schema."""
    cfg = unet.config
    meta = {
        "format": EXPORT_FORMAT,
        "num_layers": str(cfg.num_layers),
        "filters": str(cfg.filters),
        "in_channels": str(cfg.num_input_channels),
        "out_channels": str(cfg.num_output_channels),
        "activation": cfg.activation,
        "norm": cfg.norm,
        "upsample_mode": cfg.upsample_mode,
        "last_act": cfg.last_act,
        "has_camera": "0",
    }
    if camera is None:
        return meta
    ccfg = camera.config
    num_frames = 0
    if camera.exposures_values is not None:
        num_frames = int(camera.exposures_values.shape[0])
    elif camera.white_balance_values is not None:
        num_frames = int(camera.white_balance_values.shape[0])
    response_params = (
        int(camera.camera_response.response.shape[-1]) if camera.camera_response is not None else 0
    )
    meta.update(
        {
            "has_camera": "1",
            "num_frames": str(num_frames),
            "response_params": str(response_params),
            "image_height": str(camera.image_height),
            "image_width": str(camera.image_width),
            "enable_exposure": "1" if ccfg.enable_exposure else "0",
            "enable_white_balance": "1" if ccfg.enable_white_balance else "0",
            "enable_vignette": "1" if ccfg.enable_vignette else "0",
            "enable_response": "1" if ccfg.enable_response else "0",
        }
    )
    return meta


def export(
    unet: MultiScaleUnet2dDecOnlySmallFixed,
    camera: NeuralCamera | None,
    path: str | Path,
    extra_metadata: Mapping[str, str] | None = None,
) -> Path:
    """Write `unet` (+ optional `camera`) to `path` as safetensors.

    Args:
        unet: a `MultiScaleUnet2dDecOnlySmallFixed`, trained or random.
        camera: the matching `NeuralCamera`, or None to export weights only.
        path: destination `.safetensors` file.
        extra_metadata: additional string->string entries merged into
            `__metadata__` (e.g. the source checkpoint path).

    Returns:
        The path written.
    """
    tensors = unet_tensors(unet)
    if camera is not None:
        tensors.update(camera_tensors(camera))
    metadata = build_metadata(unet, camera)
    if extra_metadata:
        metadata.update({str(k): str(v) for k, v in extra_metadata.items()})
    return write_safetensors(tensors, path, metadata)
