"""Tests for trippy.net.checkpoint's TRIPS scene-checkpoint loader.

Fixtures are synthetic (AGENTS.md Sec. 6): a `_TensorBag` module is scripted
and `torch.jit.save`d under the exact parameter names TRIPS's C++ modules
register, which is the same container shape `torch.jit.load` sees when it
opens a real `torch::save(nn::Module)` archive (docs/TRIPS_REFERENCE.md
Sec. 9a). No real checkpoint bytes are needed or shipped.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from trippy.constants import TRIPS_CONFIDENCE_SIGMOID_SCALE
from trippy.net import checkpoint as ckpt_mod


class _TensorBag(nn.Module):
    """A module whose only content is a set of registered buffers, in order."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        super().__init__()
        for name, tensor in tensors.items():
            self.register_buffer(name, tensor)


def _save_bag(path, tensors: dict[str, torch.Tensor]):
    torch.jit.save(torch.jit.script(_TensorBag(tensors)), str(path))
    return path


# --- read_module_tensors -------------------------------------------------


def test_read_module_tensors_via_jit(tmp_path):
    path = _save_bag(
        tmp_path / "points.pth",
        {
            "t_position": torch.randn(5, 4),
            "t_point_size": torch.randn(5, 1),
            "t_index": torch.arange(5, dtype=torch.int32).reshape(5, 1),
        },
    )
    tensors = ckpt_mod.read_module_tensors(path)
    assert {k: tuple(v.shape) for k, v in tensors.items()} == {
        "t_position": (5, 4),
        "t_point_size": (5, 1),
        "t_index": (5, 1),
    }


def test_read_module_tensors_falls_back_to_torch_load(tmp_path):
    path = tmp_path / "plain.pth"
    torch.save({"texture": torch.zeros(4, 3)}, path)
    assert tuple(ckpt_mod.read_module_tensors(path)["texture"].shape) == (4, 3)


def test_read_module_tensors_raises_on_missing_and_garbage(tmp_path):
    with pytest.raises(FileNotFoundError):
        ckpt_mod.read_module_tensors(tmp_path / "nope.pth")
    garbage = tmp_path / "garbage.pth"
    garbage.write_bytes(b"not a torch archive at all")
    with pytest.raises(ValueError, match="could not read"):
        ckpt_mod.read_module_tensors(garbage)


# --- parametrisations ----------------------------------------------------


def test_trips_confidence_uses_the_times_ten_sigmoid():
    raw = torch.tensor([0.0, 0.5, -0.42187652])
    out = ckpt_mod.trips_confidence(raw)
    assert out[0].item() == pytest.approx(0.5)
    # sigmoid(10 * 0.5) = sigmoid(5) ~= 0.9933, TRIPS's init value.
    assert out[1].item() == pytest.approx(1.0 / (1.0 + math.exp(-5.0)), rel=1e-6)
    # A plain sigmoid would give 0.396 here; the x10 scale gives 0.0145.
    assert out[2].item() == pytest.approx(1.0 / (1.0 + math.exp(4.2187652)), rel=1e-5)
    assert TRIPS_CONFIDENCE_SIGMOID_SCALE == 10.0


def test_trips_confidence_narrowing_term():
    raw = torch.tensor([0.5])
    assert ckpt_mod.trips_confidence(raw, narrowing=2.0).item() == pytest.approx(
        1.0 / (1.0 + math.exp(-6.0)), rel=1e-6
    )


def test_trips_point_size_inverts_the_knn_initialisation():
    """`inverse_softplus(0.5 * knn_radius)` then `softplus` must round-trip."""
    knn_radius = torch.tensor([0.02, 0.5, 3.0])
    raw = torch.log(torch.exp(0.5 * knn_radius) - 1.0)  # NeuralPointCloudCuda.cpp:19-24
    assert torch.allclose(ckpt_mod.trips_point_size(raw), 0.5 * knn_radius, atol=1e-6)


def test_texture_is_used_raw_not_abs_by_default():
    """Pipeline.cpp:257 passes `non_subzero_texture` straight into PrepareTexture."""
    texture = ckpt_mod.TripsTexture(
        texture_raw=torch.tensor([[-2.0, 3.0], [1.0, -4.0]]),
        background_color_raw=torch.tensor([-0.5, 0.25]),
        confidence_raw=torch.tensor([0.5, 0.5]),
    )
    assert torch.equal(texture.texture(), torch.tensor([[-2.0, 1.0], [3.0, -4.0]]))
    assert torch.equal(texture.background_color(), torch.tensor([-0.5, 0.25]))

    abs_texture = ckpt_mod.TripsTexture(
        texture_raw=texture.texture_raw,
        background_color_raw=texture.background_color_raw,
        confidence_raw=texture.confidence_raw,
        non_subzero_texture=True,
    )
    assert torch.equal(abs_texture.texture(), torch.tensor([[2.0, 1.0], [3.0, 4.0]]))
    assert torch.equal(abs_texture.background_color(), torch.tensor([0.5, 0.25]))


# --- whole-checkpoint loader --------------------------------------------


def write_synthetic_checkpoint(
    epoch_dir, scene: str = "synth", num_points: int = 7, num_frames: int = 4, channels: int = 4
):
    """Write an ep<NNNN>/ directory with the same tensor names/shapes TRIPS writes."""
    epoch_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(4)

    def rand(*shape):
        return torch.rand(*shape, generator=generator)

    _save_bag(
        epoch_dir / f"scene_{scene}_points.pth",
        {
            "t_position": torch.cat([rand(num_points, 3) * 4 - 2, rand(num_points, 1)], dim=1),
            "t_point_size": rand(num_points, 1) - 3.0,
            "t_index": torch.arange(num_points, dtype=torch.int32).reshape(num_points, 1),
            "t_original_color": rand(num_points, 4),
        },
    )
    _save_bag(
        epoch_dir / f"scene_{scene}_texture.pth",
        {
            "texture": rand(channels, num_points) * 2 - 1,
            "background_color": rand(channels) - 0.5,
            "confidence_value_of_point": rand(1, num_points),
        },
    )
    poses = torch.zeros(num_frames, 8, dtype=torch.float64)
    poses[:, 3] = 1.0  # identity quaternion, xyzw
    poses[:, 4:7] = torch.linspace(0.0, 1.0, num_frames).reshape(-1, 1).double()
    _save_bag(
        epoch_dir / f"scene_{scene}_poses.pth",
        {"tangent_poses": torch.zeros(num_frames, 6, dtype=torch.float64), "poses_se3": poses},
    )
    intrinsics = torch.zeros(1, 13)
    intrinsics[0, :5] = torch.tensor([100.0, 100.0, 12.0, 8.0, 0.0])
    _save_bag(epoch_dir / f"scene_{scene}_intrinsics.pth", {"intrinsics": intrinsics})
    _save_bag(epoch_dir / f"scene_{scene}_ex.pth", {"0": torch.zeros(num_frames, 1, 1, 1)})
    _save_bag(epoch_dir / f"scene_{scene}_wb.pth", {"0": torch.ones(num_frames, 3, 1, 1)})
    _save_bag(
        epoch_dir / f"scene_{scene}_response.pth",
        {"response": torch.linspace(0.0, 1.0, 25).reshape(1, 1, 1, 25).repeat(1, 3, 1, 1)},
    )
    _save_bag(
        epoch_dir / f"scene_{scene}_vignette.pth",
        {"vignette_params": torch.zeros(3), "vignette_center": torch.zeros(1, 2, 1, 1)},
    )
    return epoch_dir


def test_load_trips_scene_checkpoint_shapes_and_derived_values(tmp_path):
    epoch_dir = write_synthetic_checkpoint(tmp_path / "ep0600", num_points=7, num_frames=4)
    ckpt = ckpt_mod.load_trips_scene_checkpoint(epoch_dir, "synth")

    assert len(ckpt.points) == 7
    assert ckpt.points.position.shape == (7, 3)
    assert ckpt.points.dropout_radius.shape == (7,)
    assert ckpt.points.size().shape == (7,)
    assert torch.all(ckpt.points.size() > 0)  # softplus is strictly positive
    assert torch.equal(ckpt.points.index, torch.arange(7))

    assert ckpt.texture.texture().shape == (7, 4)  # (N, C), transposed from (C, N)
    conf = ckpt.texture.confidence()
    assert conf.shape == (7,) and torch.all((conf > 0) & (conf < 1))
    assert ckpt.texture.background_color().shape == (4,)

    assert ckpt.num_frames() == 4
    assert ckpt.camera.exposure.shape == (4,)
    assert ckpt.camera.white_balance.shape == (4, 3)
    assert ckpt.camera.response is not None and ckpt.camera.response.shape == (1, 3, 1, 25)
    assert ckpt.poses_w2c is not None and ckpt.poses_w2c.shape == (4, 7)
    assert ckpt.intrinsics is not None and ckpt.intrinsics.shape == (1, 13)


def test_load_trips_scene_checkpoint_missing_file_raises(tmp_path):
    epoch_dir = write_synthetic_checkpoint(tmp_path / "ep0600")
    (epoch_dir / "scene_synth_texture.pth").unlink()
    with pytest.raises(FileNotFoundError):
        ckpt_mod.load_trips_scene_checkpoint(epoch_dir, "synth")


def test_build_neural_camera_copies_the_checkpoint_state(tmp_path):
    epoch_dir = write_synthetic_checkpoint(tmp_path / "ep0600", num_frames=5)
    ckpt = ckpt_mod.load_trips_scene_checkpoint(epoch_dir, "synth")
    camera = ckpt_mod.build_neural_camera(ckpt.camera, image_height=16, image_width=24)

    assert not camera.training  # eval() disables the response curve's training-only leak
    assert camera.exposures_values is not None
    assert camera.exposures_values.shape == (5, 1, 1, 1)
    assert camera.camera_response is not None
    assert torch.allclose(camera.camera_response.response, ckpt.camera.response)
    assert camera.vignette_net is not None
    assert torch.allclose(camera.vignette_net.vignette_params, torch.zeros(3))
    # aspect = width / height (NeuralCamera.cpp:27)
    assert camera.vignette_net.aspect == pytest.approx(24 / 16)


def test_build_neural_camera_zero_vignette_is_a_no_op(tmp_path):
    epoch_dir = write_synthetic_checkpoint(tmp_path / "ep0600", num_frames=2)
    ckpt = ckpt_mod.load_trips_scene_checkpoint(epoch_dir, "synth")
    camera = ckpt_mod.build_neural_camera(ckpt.camera, image_height=8, image_width=8)
    from trippy.net.camera_model import default_uv_grid

    uv = default_uv_grid(8, 8, torch.device("cpu"), torch.float32)
    assert torch.allclose(camera.vignette_net(uv), torch.ones(1, 1, 8, 8))
