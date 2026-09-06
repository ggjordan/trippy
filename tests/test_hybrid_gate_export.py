"""The gate's export path: safetensors metadata/weights and the bundle's `blend` block.

Module: tests.test_hybrid_gate_export
Invariants under test:
    - A gate-less export is UNCHANGED: no `gate` metadata key at all, three
      output channels, and the committed Rust parity fixture stays byte-identical
      (that last one is `tests/test_net_export_safetensors.py`'s job; here we
      prove the metadata block itself does not grow).
    - A gate export carries the fourth row of `unet.final.weight`/`.bias` and
      nothing else -- **no new tensor name** -- which is what keeps
      `brush_unet::weights`' "no unknown tensor" schema check passing.
    - The Python -> safetensors -> Python round trip reproduces the gate head's
      weights exactly, and re-running the network from the round-tripped tensors
      reproduces the gate map. This is the CPU numeric proof the task brief asks
      for in place of a GPU parity job.
    - `bundle.json` gains a `blend` block and a `splat.npz` on any HYBRID run
      (gate or not -- the block is what makes a design-A bundle renderable at
      all, since its U-Net takes `C + G` input channels), and neither on a
      non-hybrid one, so `trippy-bundle-1` stays the same format.
Synthetic fixtures only; CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from test_hybrid_a_helpers import hybrid_train_config
from test_train_helpers import build_synthetic_ply, build_synthetic_scene, tiny_train_config

from trippy.constants import HYBRID_A_GATE_CHANNEL_INDEX
from trippy.net.export_safetensors import build_metadata, export, read_safetensors, unet_tensors
from trippy.net.unet import MultiScaleUnet2dDecOnlySmallFixed, NetworkConfig
from trippy.render.bundle import (
    BUNDLE_JSON_FILENAME,
    BUNDLE_SPLAT_FILENAME,
    BUNDLE_SPLAT_MAX_VIEWS,
    export_bundle,
    splat_view_positions,
)
from trippy.train.trainer import Trainer

_LOW_LR = {"lr_points": 0.0, "lr_size": 0.0, "lr_confidence": 0.0, "lr_poses": 0.0}


def _net(out_channels: int) -> MultiScaleUnet2dDecOnlySmallFixed:
    torch.manual_seed(0)
    return MultiScaleUnet2dDecOnlySmallFixed(NetworkConfig(num_output_channels=out_channels))


# --- metadata ---


def test_a_gateless_export_has_no_gate_metadata_at_all(tmp_path: Path) -> None:
    meta = build_metadata(_net(3), None)
    assert "gate" not in meta
    assert "gate_channel" not in meta
    assert "gate_scale" not in meta
    assert meta["out_channels"] == "3"


def test_a_gate_export_declares_the_channel_and_the_scale(tmp_path: Path) -> None:
    meta = build_metadata(_net(4), None, gate_scale=1.75)
    assert meta["gate"] == "1"
    assert meta["gate_channel"] == str(HYBRID_A_GATE_CHANNEL_INDEX)
    assert float(meta["gate_scale"]) == 1.75
    assert meta["out_channels"] == "4"


def test_the_gate_is_inferred_from_the_channel_count(tmp_path: Path) -> None:
    # A caller that knows nothing about the gate still exports it correctly.
    assert build_metadata(_net(4), None)["gate"] == "1"
    assert "gate" not in build_metadata(_net(3), None)


# --- tensors: the gate adds rows, not names ---


def test_the_gate_adds_no_new_tensor_name(tmp_path: Path) -> None:
    plain = set(unet_tensors(_net(3)))
    gated = set(unet_tensors(_net(4)))
    assert plain == gated, "the gate must ride in unet.final.*, not in a new key"


def test_the_gate_widens_only_the_final_conv(tmp_path: Path) -> None:
    plain = unet_tensors(_net(3))
    gated = unet_tensors(_net(4))
    assert gated["unet.final.weight"].shape == (4, plain["unet.final.weight"].shape[1], 1, 1)
    assert gated["unet.final.bias"].shape == (4,)
    for key in plain:
        if key.startswith("unet.final."):
            continue
        assert plain[key].shape == gated[key].shape


# --- the round trip (the CPU stand-in for a GPU parity job) ---


def test_the_gate_head_round_trips_through_safetensors_exactly(tmp_path: Path) -> None:
    net = _net(4)
    path = export(net, None, tmp_path / "w.safetensors", gate_scale=0.5)
    tensors, metadata = read_safetensors(path)

    assert metadata["gate"] == "1"
    gate_row = HYBRID_A_GATE_CHANNEL_INDEX
    np.testing.assert_array_equal(
        tensors["unet.final.weight"][gate_row],
        net.final[0].weight[gate_row].detach().numpy(),
    )
    np.testing.assert_array_equal(
        tensors["unet.final.bias"][gate_row], net.final[0].bias[gate_row].detach().numpy()
    )


def test_a_network_rebuilt_from_the_file_reproduces_the_gate_map(tmp_path: Path) -> None:
    """Python -> safetensors -> Python must give the same gate, to float32 exactness.

    The Rust loader reads the same names and shapes (rust/crates/brush-unet
    `weights.rs`), so what this proves about the file is what the Burn port
    inherits; the numeric agreement of the two *implementations* is the job of
    `rust/crates/brush-unet/tests/parity_gpu.rs`.
    """
    torch.manual_seed(3)
    net = _net(4)
    path = export(net, None, tmp_path / "w.safetensors")
    tensors, _ = read_safetensors(path)

    rebuilt = _net(4)
    with torch.no_grad():
        for key, value in tensors.items():
            parts = key.split(".")
            if parts[1] == "final":
                target = rebuilt.final[0].weight if parts[2] == "weight" else rebuilt.final[0].bias
            elif parts[1] == "start":
                conv = rebuilt.start.conv
                sub = conv.feature_conv if parts[2] == "feature" else conv.gate_conv
                target = sub.weight if parts[3] == "weight" else sub.bias
            else:  # "up"
                conv = rebuilt.up[int(parts[2])].conv
                sub = conv.feature_conv if parts[3] == "feature" else conv.gate_conv
                target = sub.weight if parts[4] == "weight" else sub.bias
            target.copy_(torch.from_numpy(value))

    layers = [torch.randn(1, 4, 32 >> i, 32 >> i) for i in range(5)]
    with torch.no_grad():
        a, b = net(layers), rebuilt(layers)
    assert torch.equal(a, b)
    assert torch.equal(torch.sigmoid(a[:, 3:4]), torch.sigmoid(b[:, 3:4]))


# --- view selection ---


def test_splat_view_positions_always_include_the_default_view() -> None:
    views = [object()] * 40  # only the length matters
    chosen = splat_view_positions(views, default_position=17, limit=BUNDLE_SPLAT_MAX_VIEWS)
    assert 17 in chosen
    assert len(chosen) <= BUNDLE_SPLAT_MAX_VIEWS
    assert chosen == sorted(set(chosen))


def test_splat_view_positions_spread_across_the_capture() -> None:
    chosen = splat_view_positions([object()] * 100, default_position=0, limit=5)
    assert len(chosen) == 5
    # Not all bunched at the front.
    assert max(chosen) > 50


def test_splat_view_positions_handle_tiny_and_empty_view_lists() -> None:
    assert splat_view_positions([], 0, 5) == []
    assert splat_view_positions([object()], 0, 5) == [0]
    assert splat_view_positions([object()] * 3, 9, 5) == [0, 1, 2]


# --- the bundle ---


def _bundle_from(tmp_path: Path, gate: bool) -> tuple[Path, dict]:
    overrides = {"dropout_gaussian_p": 0.0}
    if gate:
        overrides["gate"] = {"enabled": True}
    cfg, _names = hybrid_train_config(tmp_path / ("gate" if gate else "plain"), **overrides)
    for key, value in _LOW_LR.items():
        setattr(cfg, key, value)
    trainer = Trainer(cfg)
    checkpoint = trainer.save_checkpoint()
    out = tmp_path / ("bundle-gate" if gate else "bundle-plain")
    bundle_dir, _document = export_bundle(checkpoint, out, name="synthetic")
    return bundle_dir, json.loads((bundle_dir / BUNDLE_JSON_FILENAME).read_text())


def _plain_bundle(tmp_path: Path) -> tuple[Path, dict]:
    """A bundle from a NON-hybrid run: the pre-blend-gate bundle, unchanged."""
    scene_root, point_set = build_synthetic_scene(tmp_path / "plain")
    ply_path = build_synthetic_ply(tmp_path / "plain", point_set)
    cfg = tiny_train_config(scene_root, ply_path, tmp_path / "plain-run", tmp_path / "plain-cache")
    for key, value in _LOW_LR.items():
        setattr(cfg, key, value)
    checkpoint = Trainer(cfg).save_checkpoint()
    bundle_dir, _document = export_bundle(checkpoint, tmp_path / "bundle-plain", name="synthetic")
    return bundle_dir, json.loads((bundle_dir / BUNDLE_JSON_FILENAME).read_text())


def test_a_non_hybrid_bundle_is_exactly_what_it_always_was(tmp_path: Path) -> None:
    bundle_dir, document = _plain_bundle(tmp_path)
    assert "blend" not in document
    assert not (bundle_dir / BUNDLE_SPLAT_FILENAME).exists()
    assert sorted(p.name for p in bundle_dir.iterdir()) == [
        "bundle.json",
        "points.npz",
        "weights.safetensors",
    ]
    assert document["format"] == "trippy-bundle-1"


def test_a_hybrid_bundle_without_the_gate_still_carries_its_block(tmp_path: Path) -> None:
    """The bug fix: a design-A bundle needs the Gaussian block to be renderable at all.

    Its U-Net takes `feature_channels + G` input channels, so without the block
    there is nothing to feed channels C..C+G and the first forward fails. That
    is true whether or not the run has the blend gate, so the `blend` block is
    written for every hybrid run -- with `gate: false` when there is no gate.
    """
    bundle_dir, document = _bundle_from(tmp_path, gate=False)
    blend = document["blend"]
    assert blend["gate"] is False
    assert blend["channels"] == ["rgb", "alpha", "depth"]
    assert blend["mode"] == "all_levels"
    assert blend["feature_channels"] + 5 == int(
        read_safetensors(bundle_dir / "weights.safetensors")[1]["in_channels"]
    )
    assert (bundle_dir / BUNDLE_SPLAT_FILENAME).exists()


def test_a_gate_bundle_carries_the_blend_block(tmp_path: Path) -> None:
    _bundle_dir, document = _bundle_from(tmp_path, gate=True)
    # The format tag does NOT change: `blend` is an additive optional key.
    assert document["format"] == "trippy-bundle-1"
    blend = document["blend"]
    assert blend["gate"] is True
    assert blend["gate_channel"] == HYBRID_A_GATE_CHANNEL_INDEX
    assert blend["gate_scale"] == 1.0
    assert blend["mask_by_alpha"] is True
    assert blend["splat_renders"] == BUNDLE_SPLAT_FILENAME
    assert blend["channels"] == ["rgb", "alpha", "depth"]
    assert blend["mode"] == "all_levels"
    assert blend["feature_channels"] == 4
    assert blend["splat_views"], "a hybrid run has renders, so some views must carry one"
    assert document["default_view"] in blend["splat_views"]
    # Every listed position is a real view.
    assert all(0 <= p < len(document["views"]) for p in blend["splat_views"])


def test_the_splat_archive_holds_a_uint8_render_per_listed_view(tmp_path: Path) -> None:
    bundle_dir, document = _bundle_from(tmp_path, gate=True)
    blend = document["blend"]
    with np.load(bundle_dir / BUNDLE_SPLAT_FILENAME) as archive:
        for position in blend["splat_views"]:
            rgb = archive[f"view_{position}"]
            alpha = archive[f"alpha_{position}"]
            depth = archive[f"depth_{position}"]
            assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
            assert alpha.dtype == np.uint8 and alpha.shape == rgb.shape[:2]
            # float32, not uint8: the normalised depth is unbounded and the
            # network was trained on it directly (BUNDLE_SPLAT_DEPTH_DTYPE).
            assert depth.dtype == np.float32 and depth.shape == rgb.shape[:2]
            view = document["views"][position]
            # Downscaled by blend.splat_scale, within a pixel of rounding.
            assert abs(rgb.shape[1] - view["width"] * blend["splat_scale"]) <= 1
            assert abs(rgb.shape[0] - view["height"] * blend["splat_scale"]) <= 1


def test_the_bundle_weights_declare_the_gate(tmp_path: Path) -> None:
    bundle_dir, _document = _bundle_from(tmp_path, gate=True)
    _tensors, metadata = read_safetensors(bundle_dir / "weights.safetensors")
    assert metadata["gate"] == "1"
    assert metadata["out_channels"] == "4"
    assert metadata["gate_channel"] == str(HYBRID_A_GATE_CHANNEL_INDEX)
