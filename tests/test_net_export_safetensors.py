"""The safetensors export the Rust U-Net port reads.

Module: tests.test_net_export_safetensors
Purpose: pin the wire format between `trippy.net` and
    `rust/crates/brush-unet` -- the container itself (a hand-written
    safetensors writer, so it needs its own round-trip test), the key schema,
    the metadata block, and the committed fixture's contents. A change on
    either side that breaks the Rust parity test fails here first, on CPU, in
    milliseconds.
Invariants: CPU only, no GPU, no network; the fixture is regenerated in a
    tmp dir and compared, never rewritten in place.
Related docs: rust/README.md ("brush-unet weight schema").
"""

from __future__ import annotations

import json
import struct
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.export_unet_safetensors import (
    FIXTURE_CHANNELS,
    FIXTURE_FILTERS,
    FIXTURE_FRAMES,
    FIXTURE_HEIGHT,
    FIXTURE_LAYERS,
    FIXTURE_WIDTH,
    build_fixture,
    fixture_shapes,
)
from trippy.net.camera_model import NeuralCamera
from trippy.net.export_safetensors import (
    EXPORT_FORMAT,
    camera_tensors,
    export,
    read_safetensors,
    unet_tensors,
    write_safetensors,
)
from trippy.net.unet import MultiScaleUnet2dDecOnlySmallFixed, NetworkConfig

FIXTURE_DIR = Path("tests/fixtures/synthetic/unet_fixture_small")


def _net_and_camera() -> tuple[MultiScaleUnet2dDecOnlySmallFixed, NeuralCamera]:
    net = MultiScaleUnet2dDecOnlySmallFixed(
        NetworkConfig(
            num_layers=FIXTURE_LAYERS, num_input_channels=FIXTURE_CHANNELS, filters=FIXTURE_FILTERS
        )
    )
    camera = NeuralCamera(
        image_height=FIXTURE_HEIGHT, image_width=FIXTURE_WIDTH, num_frames=FIXTURE_FRAMES
    )
    return net, camera


def test_the_container_round_trips(tmp_path: Path) -> None:
    """The hand-written writer is readable by the hand-written reader."""
    tensors = {
        "a": np.arange(6, dtype=np.float32).reshape(2, 3),
        "b": np.array([1.5], dtype=np.float32),
        "c": np.zeros((), dtype=np.float32),
    }
    path = write_safetensors(tensors, tmp_path / "x.safetensors", {"format": "test", "n": "3"})
    back, metadata = read_safetensors(path)
    assert metadata == {"format": "test", "n": "3"}
    assert set(back) == set(tensors)
    for name, array in tensors.items():
        np.testing.assert_array_equal(back[name], array)


def test_the_header_is_eight_byte_aligned_and_offsets_are_contiguous(tmp_path: Path) -> None:
    """safetensors' Rust reader validates both; a violation is silent corruption here."""
    net, camera = _net_and_camera()
    path = export(net, camera, tmp_path / "w.safetensors")
    raw = path.read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    assert header_len % 8 == 0, "header must be padded to an 8-byte boundary"
    header = json.loads(raw[8 : 8 + header_len])
    header.pop("__metadata__")
    spans = sorted(info["data_offsets"] for info in header.values())
    assert spans[0][0] == 0
    for (_, end), (begin, _) in pairwise(spans):
        assert end == begin, "data segment must be contiguous"
    assert spans[-1][1] == len(raw) - 8 - header_len, "no trailing bytes"


def test_key_schema_covers_every_parameter_exactly_once() -> None:
    """Nothing learned is dropped, nothing is exported twice."""
    net, camera = _net_and_camera()
    tensors = unet_tensors(net)
    assert len(tensors) == len(list(net.state_dict()))
    exported_scalars = sum(int(np.prod(t.shape)) for t in tensors.values())
    assert exported_scalars == net.parameter_count()

    camera_keys = set(camera_tensors(camera))
    assert camera_keys == {
        "camera.exposure",
        "camera.white_balance",
        "camera.vignette_params",
        "camera.vignette_center",
        "camera.response",
    }


def test_up_blocks_are_exported_in_application_order() -> None:
    """`up.{k}` must be the k-th block `forward()` runs, not pyramid level k.

    The Rust port derives the pyramid level it reads as
    `num_layers - 2 - k`; exporting the ModuleList in construction order (as
    the Python module stores it) is what makes that mapping correct.
    """
    net, _ = _net_and_camera()
    tensors = unet_tensors(net)
    # The last block is the `last=True` one: out = filters - C, not - 2C.
    last = tensors[f"unet.up.{FIXTURE_LAYERS - 2}.feature.weight"]
    assert last.shape[0] == FIXTURE_FILTERS - FIXTURE_CHANNELS
    for k in range(FIXTURE_LAYERS - 2):
        assert tensors[f"unet.up.{k}.feature.weight"].shape[0] == (
            FIXTURE_FILTERS - 2 * FIXTURE_CHANNELS
        )
    # ...and `net.up[k]` really is applied in that order.
    assert net._up_indices == [FIXTURE_LAYERS - 2 - k for k in range(FIXTURE_LAYERS - 1)]


def test_metadata_describes_the_architecture() -> None:
    net, camera = _net_and_camera()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = export(net, camera, Path(tmp) / "w.safetensors")
        _, metadata = read_safetensors(path)
    assert metadata["format"] == EXPORT_FORMAT
    assert metadata["num_layers"] == str(FIXTURE_LAYERS)
    assert metadata["filters"] == str(FIXTURE_FILTERS)
    assert metadata["in_channels"] == str(FIXTURE_CHANNELS)
    assert metadata["out_channels"] == "3"
    # The Rust port refuses anything but these; the exporter must state them.
    assert (metadata["activation"], metadata["norm"]) == ("elu", "id")
    assert (metadata["upsample_mode"], metadata["last_act"]) == ("bilinear", "id")
    assert metadata["has_camera"] == "1"
    assert metadata["num_frames"] == str(FIXTURE_FRAMES)
    assert (metadata["image_height"], metadata["image_width"]) == (
        str(FIXTURE_HEIGHT),
        str(FIXTURE_WIDTH),
    )


@pytest.mark.skipif(not FIXTURE_DIR.exists(), reason="fixture not generated")
def test_the_committed_fixture_is_reproducible(tmp_path: Path) -> None:
    """Regenerating from the pinned seed must give the same bytes.

    This is what stops a semantic change in `trippy.net` from silently
    invalidating the Rust parity target: it fails here instead.
    """
    meta = build_fixture(tmp_path)
    committed = json.loads((FIXTURE_DIR / "meta.json").read_text())
    assert meta == committed
    for name in ("weights.safetensors", "io.safetensors"):
        assert (tmp_path / name).read_bytes() == (FIXTURE_DIR / name).read_bytes(), (
            f"{name} drifted; re-run tools/export_unet_safetensors.py fixture"
        )


@pytest.mark.skipif(not FIXTURE_DIR.exists(), reason="fixture not generated")
def test_the_fixture_is_small_and_shaped_as_declared() -> None:
    total = sum(p.stat().st_size for p in FIXTURE_DIR.rglob("*") if p.is_file())
    assert total < 400_000, f"fixture grew to {total} bytes"

    tensors, metadata = read_safetensors(FIXTURE_DIR / "io.safetensors")
    shapes = fixture_shapes(FIXTURE_HEIGHT, FIXTURE_WIDTH, FIXTURE_LAYERS)
    assert metadata["num_layers"] == str(FIXTURE_LAYERS)
    for level, (h, w) in enumerate(shapes):
        assert tensors[f"input.{level}"].shape == (1, FIXTURE_CHANNELS, h, w)
    assert tensors["unet_out"].shape == (1, 3, FIXTURE_HEIGHT, FIXTURE_WIDTH)
    assert tensors["rgb_out"].shape == (1, 3, FIXTURE_HEIGHT, FIXTURE_WIDTH)
    # The camera probe must straddle the response LUT's [0, 1] domain, or the
    # parity test would never exercise the clamped ends.
    probe = tensors["camera_probe"]
    assert probe.min() < 0.0 and probe.max() > 1.0
    # ...and the tone mapper's output must be bounded by the LUT's own control
    # points (the fixture perturbs the gamma curve, so lut[0] is slightly
    # negative -- that is deliberate: it proves the clamp happens on the LUT
    # *coordinate*, not on the output).
    rgb_out = tensors["rgb_out"]
    lut = read_safetensors(FIXTURE_DIR / "weights.safetensors")[0]["camera.response"]
    assert lut.min() - 1e-6 <= rgb_out.min() and rgb_out.max() <= lut.max() + 1e-6
    assert lut.min() < 0.0, "the fixture LUT must dip below 0 to exercise that branch"


@pytest.mark.skipif(not FIXTURE_DIR.exists(), reason="fixture not generated")
def test_the_fixture_weights_reproduce_the_fixture_outputs() -> None:
    """Loading the exported weights back into PyTorch reproduces `unet_out`.

    Closes the loop the Rust side opens: if this passes and the Rust parity
    test passes, the two implementations agree *through the same file*.
    """
    weights, metadata = read_safetensors(FIXTURE_DIR / "weights.safetensors")
    io_tensors, _ = read_safetensors(FIXTURE_DIR / "io.safetensors")

    net = MultiScaleUnet2dDecOnlySmallFixed(
        NetworkConfig(
            num_layers=int(metadata["num_layers"]),
            num_input_channels=int(metadata["in_channels"]),
            filters=int(metadata["filters"]),
        )
    )
    with torch.no_grad():
        start = net.start.conv
        start.feature_conv.weight.copy_(torch.from_numpy(weights["unet.start.feature.weight"]))
        start.feature_conv.bias.copy_(torch.from_numpy(weights["unet.start.feature.bias"]))
        start.gate_conv.weight.copy_(torch.from_numpy(weights["unet.start.gate.weight"]))
        start.gate_conv.bias.copy_(torch.from_numpy(weights["unet.start.gate.bias"]))
        for k, block in enumerate(net.up):
            for branch, conv in (("feature", block.conv.feature_conv), ("gate", block.conv.gate_conv)):
                conv.weight.copy_(torch.from_numpy(weights[f"unet.up.{k}.{branch}.weight"]))
                conv.bias.copy_(torch.from_numpy(weights[f"unet.up.{k}.{branch}.bias"]))
        net.final[0].weight.copy_(torch.from_numpy(weights["unet.final.weight"]))
        net.final[0].bias.copy_(torch.from_numpy(weights["unet.final.bias"]))
    net.eval()

    inputs = [
        torch.from_numpy(io_tensors[f"input.{level}"]) for level in range(int(metadata["num_layers"]))
    ]
    with torch.no_grad():
        out = net(inputs).numpy()
    np.testing.assert_allclose(out, io_tensors["unet_out"], atol=1e-6)
