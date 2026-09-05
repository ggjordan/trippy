"""Tests for trippy.net.checkpoint (best-effort TRIPS checkpoint loading).

Module: tests.test_net_checkpoint
Invariants under test: try_load_trips_network() never raises, whether the
    file is missing, is a foreign/garbage file, or is a real PyTorch
    checkpoint (a plain torch.save'd state_dict is used as a stand-in for
    "some torch.load-able file", since no real TRIPS `render_net.pth` is
    available in this sandbox -- see docs/LIMITATIONS.md for the status
    of an actual attempt against the public Zenodo checkpoints).
"""

from __future__ import annotations

from pathlib import Path

import torch

from trippy.net.checkpoint import try_load_trips_network
from trippy.net.unet import MultiScaleUnet2dDecOnlySmallFixed, NetworkConfig


def test_missing_file_returns_not_ok_with_reason() -> None:
    result = try_load_trips_network("/nonexistent/path/render_net.pth")
    assert result.ok is False
    assert "does not exist" in result.reason


def test_garbage_file_returns_not_ok_with_reason(tmp_path: Path) -> None:
    bad = tmp_path / "garbage.pth"
    bad.write_bytes(b"not a checkpoint at all")
    result = try_load_trips_network(bad)
    assert result.ok is False
    assert result.reader is None
    assert result.reason is not None


def test_shape_matched_load_into_matching_target(tmp_path: Path) -> None:
    """A plain torch.save'd state_dict from an identical architecture should load cleanly
    via the torch.load(weights_only=False) path with every tensor shape-matched."""
    source = MultiScaleUnet2dDecOnlySmallFixed(NetworkConfig())
    ckpt_path = tmp_path / "render_net.pth"
    torch.save(source.state_dict(), ckpt_path)

    target = MultiScaleUnet2dDecOnlySmallFixed(NetworkConfig())
    result = try_load_trips_network(ckpt_path, target=target)

    assert result.ok is True
    assert result.reader == "torch_load"
    assert result.num_tensors_assigned == result.num_tensors_found
    for p_target, p_source in zip(target.parameters(), source.parameters(), strict=True):
        torch.testing.assert_close(p_target, p_source)


def test_read_only_report_without_target(tmp_path: Path) -> None:
    source = MultiScaleUnet2dDecOnlySmallFixed(NetworkConfig())
    ckpt_path = tmp_path / "render_net.pth"
    torch.save(source.state_dict(), ckpt_path)

    result = try_load_trips_network(ckpt_path)
    assert result.ok is True
    assert result.num_tensors_assigned == 0
    assert result.num_tensors_found > 0
    assert len(result.tensor_shapes_found) == result.num_tensors_found
