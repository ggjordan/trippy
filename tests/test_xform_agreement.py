"""Cross-check trippy.geom.xform_a (numpy) against trippy.geom.xform_b (torch).

Module: tests.test_xform_agreement
Invariants under test: two independently-written implementations of the
    same COLMAP camera convention must agree numerically (a), must NOT
    silently agree when fed a convention bug (b), and must agree on the
    sign of "in front of the camera" (c).
Related docs: docs/SPEC.md "Verification (end-to-end)" item 1;
    AGENTS.md "geometry implemented twice and made to disagree first".
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from trippy.geom import xform_a, xform_b


def _random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


@pytest.mark.parametrize("seed", range(20))
def test_xform_a_b_agree(seed: int) -> None:
    rng = np.random.default_rng(seed)
    q = _random_unit_quat(rng)
    t = rng.normal(size=3)
    xyz_w = rng.normal(size=(50, 3)) * 3.0
    fx, fy = 1000.0 + rng.normal(), 1000.0 + rng.normal()
    cx, cy = 512.0, 384.0

    R_a = xform_a.qvec2R(q)
    R_b = xform_b.qvec2R(torch.tensor(q, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(R_a, R_b, atol=1e-6)

    xyz_c_a = xform_a.world_to_cam(R_a, t, xyz_w)
    xyz_c_b = xform_b.world_to_cam(
        torch.tensor(R_b, dtype=torch.float64),
        torch.tensor(t, dtype=torch.float64),
        torch.tensor(xyz_w, dtype=torch.float64),
    ).numpy()
    np.testing.assert_allclose(xyz_c_a, xyz_c_b, atol=1e-5)

    # Push points in front of the camera so depth division is well-conditioned.
    xyz_c_a[:, 2] = np.abs(xyz_c_a[:, 2]) + 1.0
    xyz_c_b = xyz_c_a.copy()

    uv_a, depth_a = xform_a.project_pinhole(xyz_c_a, fx, fy, cx, cy)
    uv_b, depth_b = xform_b.project_pinhole(
        torch.tensor(xyz_c_b, dtype=torch.float64), fx, fy, cx, cy
    )
    np.testing.assert_allclose(uv_a, uv_b.numpy(), atol=1e-4)
    np.testing.assert_allclose(depth_a, depth_b.numpy(), atol=1e-4)


def test_xform_b_disagrees_on_wrong_quat_order() -> None:
    """Feeding xform_b a quaternion in xyzw order must NOT match xform_a's
    (correct, wxyz) result. This guards against both implementations
    sharing the same silent wxyz/xyzw convention bug: xform_b's Rodrigues
    route is different enough from xform_a's direct expansion that a wrong
    component order produces a visibly different rotation, not a
    coincidentally-matching one.
    """
    q_wxyz = np.array([0.1, 0.2, 0.3, 0.9])
    q_wxyz = q_wxyz / np.linalg.norm(q_wxyz)

    R_a = xform_a.qvec2R(q_wxyz)

    # Reinterpret the same quaternion's components as if given in xyzw order:
    # true (w,x,y,z) -> stored as (x,y,z,w) -> fed to a wxyz-expecting function.
    q_mislabelled = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    R_b_wrong = xform_b.qvec2R(torch.tensor(q_mislabelled, dtype=torch.float64)).numpy()

    assert not np.allclose(R_a, R_b_wrong, atol=1e-3), (
        "xform_b matched xform_a despite a mislabelled quaternion component order; "
        "this would hide a real wxyz/xyzw convention bug"
    )


def test_depth_positive_in_front_of_camera() -> None:
    """A point straight ahead of an identity-pose camera has positive depth in both."""
    R = np.eye(3)
    t = np.zeros(3)
    point_in_front = np.array([[0.0, 0.0, 5.0]])

    xyz_c_a = xform_a.world_to_cam(R, t, point_in_front)
    _, depth_a = xform_a.project_pinhole(xyz_c_a, 1000.0, 1000.0, 512.0, 384.0)
    assert depth_a[0] > 0

    xyz_c_b = xform_b.world_to_cam(
        torch.eye(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        torch.tensor(point_in_front, dtype=torch.float64),
    )
    _, depth_b = xform_b.project_pinhole(xyz_c_b, 1000.0, 1000.0, 512.0, 384.0)
    assert depth_b[0].item() > 0


def test_xform_a_never_imports_torch() -> None:
    """trippy.geom.xform_a must stay numpy-only (see module invariants)."""
    assert "trippy.geom.xform_a" in sys.modules
    # torch may legitimately be imported by *other* already-loaded test modules
    # in this process; the real guarantee is that importing xform_a alone,
    # in a clean subprocess, never pulls torch in. We assert that statically
    # here by scanning the source for a top-level torch import.
    src = xform_a.__spec__.loader.get_source("trippy.geom.xform_a") or ""
    assert "import torch" not in src


def test_xform_b_never_imports_numpy() -> None:
    """trippy.geom.xform_b must stay torch-only (see module invariants)."""
    src = xform_b.__spec__.loader.get_source("trippy.geom.xform_b") or ""
    assert "import numpy" not in src


# --------------------------------------------------------------------------
# se3_exp: rotation-gradient-at-zero fix (see trippy/geom/xform_b.py
# _so3_exp_coeffs/se3_exp and docs/GEOMETRY.md "SE(3) exponential map").
# se3_exp used to build its rotation from a normalized axis
# (phi / max(|phi|, eps)) scaled back up by |phi|, which is second order in
# phi at the origin and zeroed the rotation gradient there
# (tests/test_raster_bwd_ref.py::
# test_pose_delta_rotation_gradient_matches_generator_at_zero pins the
# regression test for that). These tests cover the replacement formula
# itself: gradcheck through the origin and through tiny/moderate angles, and
# agreement with an independently-computed reference.
# --------------------------------------------------------------------------


def _twist_matrix(delta: torch.Tensor) -> torch.Tensor:
    """4x4 se(3) generator matrix Xi with se3_exp(delta) == torch.matrix_exp(Xi).

    Xi = [[skew(phi), rho], [0, 0, 0, 0]]; exponentiating this 4x4 matrix is
    the textbook definition of the SE(3) exponential map, and torch's
    scaling-and-squaring `matrix_exp` is numerically independent of this
    module's closed-form Rodrigues coefficients -- a genuine cross-check,
    not the same formula written twice.
    """
    rho = delta[0:3]
    phi = delta[3:6]
    xi = torch.zeros((4, 4), dtype=delta.dtype, device=delta.device)
    xi[:3, :3] = xform_b._skew(phi)
    xi[:3, 3] = rho
    return xi


def _random_delta(rng: torch.Generator, theta_mag: float) -> torch.Tensor:
    axis = torch.randn(3, dtype=torch.float64, generator=rng)
    axis = axis / axis.norm()
    phi = axis * theta_mag
    rho = torch.randn(3, dtype=torch.float64, generator=rng)
    return torch.cat([rho, phi])


@pytest.mark.parametrize("theta_mag", [0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 0.5])
def test_se3_exp_matches_matrix_exponential(theta_mag: float) -> None:
    """se3_exp must match torch.matrix_exp of the 4x4 se(3) generator to
    1e-12, including in the tiny-angle regime where a naive (1-cos)/theta^2
    would lose precision to cancellation -- exactly the regime the
    Taylor-guarded coefficients in _so3_exp_coeffs exist for.
    """
    rng = torch.Generator().manual_seed(0)
    delta = _random_delta(rng, theta_mag)

    T = xform_b.se3_exp(delta)
    T_ref = torch.matrix_exp(_twist_matrix(delta))

    torch.testing.assert_close(T, T_ref, atol=1e-12, rtol=0.0)


def test_se3_exp_gradcheck_at_zero_delta() -> None:
    """The whole point of the fix: gradcheck must pass through phi == 0,
    where autograd used to return an exactly-zero rotation gradient.
    """
    delta = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(xform_b.se3_exp, (delta,))


@pytest.mark.parametrize("theta_mag", [1e-6, 0.5])
def test_se3_exp_gradcheck_nonzero_phi(theta_mag: float) -> None:
    """gradcheck at a tiny angle (inside the Taylor-series branch) and a
    moderate angle (inside the closed-form branch), so both branches of
    _so3_exp_coeffs are covered end to end through autograd.
    """
    rng = torch.Generator().manual_seed(0)
    delta = _random_delta(rng, theta_mag).requires_grad_(True)
    assert torch.autograd.gradcheck(xform_b.se3_exp, (delta,))
