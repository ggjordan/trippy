"""Tests for trippy.train.params: PointParams/PoseParams init and activation round trips.

Module: tests.test_train_params
Invariants under test: `softplus(raw_size) == size0` and
    `sigmoid(CONF_SIGMOID_SCALE * raw_conf) == conf0` (the exact inverse
    functions the task brief asks for); `feat[:, :3] == rgb0` at init;
    `PoseParams.compose_pose` with a zero delta is the identity (returns
    `R`, `t` unchanged); a non-zero delta actually moves the pose.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional

from trippy.constants import CONF_SIGMOID_SCALE
from trippy.points.source import PointSet
from trippy.train.params import PointParams, PoseParams, inverse_softplus, logit


def _random_point_set(n: int = 50, seed: int = 0) -> PointSet:
    rng = np.random.default_rng(seed)
    return PointSet(
        xyz=rng.uniform(-5, 5, (n, 3)).astype(np.float32),
        size0=rng.uniform(0.01, 2.0, n).astype(np.float32),
        rgb0=rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32),
        conf0=rng.uniform(0.05, 0.95, n).astype(np.float32),
        provenance=np.full(n, 3, dtype=np.uint8),
    )


def test_inverse_softplus_round_trips() -> None:
    x = torch.tensor([0.001, 0.05, 0.5, 1.0, 5.0, 30.0])
    raw = inverse_softplus(x)
    torch.testing.assert_close(functional.softplus(raw), x, atol=1e-4, rtol=1e-4)


def test_logit_round_trips_via_sigmoid() -> None:
    p = torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99])
    torch.testing.assert_close(torch.sigmoid(logit(p)), p, atol=1e-4, rtol=1e-4)


def test_point_params_size_matches_size0() -> None:
    ps = _random_point_set()
    params = PointParams(ps)
    np.testing.assert_allclose(params.size().detach().numpy(), ps.size0, atol=1e-4, rtol=1e-4)


def test_point_params_conf_matches_conf0_with_x10_scale() -> None:
    ps = _random_point_set()
    params = PointParams(ps)
    np.testing.assert_allclose(params.conf().detach().numpy(), ps.conf0, atol=1e-3, rtol=1e-3)
    # The x10 decision (docs/TRIPS_REFERENCE.md Sec. 2/10.4): raw_conf should be an order of
    # magnitude smaller than a plain logit(conf0), not equal to it.
    plain_logit = logit(torch.from_numpy(ps.conf0))
    torch.testing.assert_close(params.raw_conf * CONF_SIGMOID_SCALE, plain_logit, atol=1e-3, rtol=1e-3)


def test_point_params_feat_seeds_rgb_in_first_three_channels() -> None:
    ps = _random_point_set()
    params = PointParams(ps, feature_channels=6)
    np.testing.assert_allclose(params.feat[:, :3].detach().numpy(), ps.rgb0, atol=1e-6)
    assert params.feat.shape == (len(ps), 6)
    # Remaining channels are small noise, not zero and not full-range [0, 1].
    extra = params.feat[:, 3:].detach().numpy()
    assert np.all(np.abs(extra) < 0.2)


def test_point_params_rejects_too_few_feature_channels() -> None:
    ps = _random_point_set(n=5)
    try:
        PointParams(ps, feature_channels=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError for feature_channels < 3")


def test_point_params_provenance_is_buffer_not_parameter() -> None:
    ps = _random_point_set(n=5)
    params = PointParams(ps)
    param_names = {name for name, _ in params.named_parameters()}
    assert "provenance" not in param_names
    buffer_names = {name for name, _ in params.named_buffers()}
    assert "provenance" in buffer_names
    np.testing.assert_array_equal(params.provenance.numpy(), ps.provenance)


def test_pose_params_zero_delta_is_identity() -> None:
    pose = PoseParams(3)
    R = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = torch.tensor([1.0, 2.0, 3.0])
    R_out, t_out = pose.compose_pose(1, R, t)
    torch.testing.assert_close(R_out, R, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(t_out, t, atol=1e-6, rtol=1e-6)


def test_pose_params_nonzero_delta_moves_pose() -> None:
    pose = PoseParams(2)
    with torch.no_grad():
        pose.delta[0, :3] = torch.tensor([0.1, 0.0, 0.0])  # pure translation twist
    R = torch.eye(3)
    t = torch.zeros(3)
    R_out, t_out = pose.compose_pose(0, R, t)
    torch.testing.assert_close(R_out, R, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(t_out, t)


def test_pose_params_gradient_flows_to_delta() -> None:
    pose = PoseParams(1)
    R = torch.eye(3)
    t = torch.zeros(3)
    R_out, t_out = pose.compose_pose(0, R, t)
    loss = R_out.sum() + t_out.sum()
    loss.backward()
    assert pose.delta.grad is not None
    assert torch.any(pose.delta.grad != 0)
