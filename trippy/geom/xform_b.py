"""Camera geometry, implementation B: torch, independent formula from xform_a.

Module: trippy.geom.xform_b
Invariants: NO numpy import (verified by tests/test_xform_agreement.py via
    sys.modules) so autograd flows end to end through project_pinhole for
    pose/point-position gradients (see plan "Technical design": "all
    per-point differentiable math in vectorised PyTorch"). qvec2R is
    deliberately implemented via a different route than xform_a (axis-angle
    extracted from the quaternion, then Rodrigues' rotation formula) rather
    than the direct wxyz-expansion xform_a uses, so the two modules cannot
    silently share the same convention bug (see
    tests/test_xform_agreement.py's "deliberate disagreement" case).
Coordinate frame: identical convention to xform_a -- COLMAP world/camera:
    camera space is x-right, y-down, z-forward (z > 0 in front of camera);
    quaternions are unit, (qw, qx, qy, qz), and x_cam = R(q) @ x_world + t.
se3_exp/compose use the twist convention xi = (rho, phi) with
    rho = xi[:3] (translation generator) and phi = xi[3:] (rotation vector,
    axis * angle in radians) -- the Sophus/g2o SE3 convention.
Related docs: trippy.geom.xform_a (independent numpy implementation);
    docs/SPEC.md "Technical design" (learnable SE(3) pose delta).
"""

from __future__ import annotations

import torch

from trippy.constants import EPS_QUAT_AXIS


def _skew(v: torch.Tensor) -> torch.Tensor:
    """3-vector to its 3x3 skew-symmetric (cross-product) matrix."""
    zero = torch.zeros((), dtype=v.dtype, device=v.device)
    return torch.stack(
        [
            torch.stack([zero, -v[2], v[1]]),
            torch.stack([v[2], zero, -v[0]]),
            torch.stack([-v[1], v[0], zero]),
        ]
    )


def _rodrigues(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrix for a unit `axis` rotated by `angle` radians (Rodrigues' formula)."""
    K = _skew(axis)
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return eye + torch.sin(angle) * K + (1.0 - torch.cos(angle)) * (K @ K)


def qvec2R(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (COLMAP wxyz order) to a 3x3 rotation matrix.

    Computed via axis-angle extraction + Rodrigues' formula -- a different
    code path from xform_a.qvec2R's direct wxyz expansion, by design.

    Args:
        q: shape (4,), float, (qw, qx, qy, qz), unit norm.

    Returns:
        R: shape (3, 3), world->camera rotation such that x_cam = R @ x_world.
    """
    w = torch.clamp(q[0], -1.0, 1.0)
    xyz = q[1:4]
    angle = 2.0 * torch.acos(w)
    s = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    if s.item() > EPS_QUAT_AXIS:
        axis = xyz / s
    else:
        # angle ~ 0: rotation is the identity regardless of axis choice.
        axis = torch.tensor([1.0, 0.0, 0.0], dtype=q.dtype, device=q.device)
    return _rodrigues(axis, angle)


def world_to_cam(R: torch.Tensor, t: torch.Tensor, xyz_w: torch.Tensor) -> torch.Tensor:
    """Transform world-frame points into camera frame: x_cam = R @ x_world + t.

    Args:
        R: shape (3, 3), world->camera rotation (see qvec2R).
        t: shape (3,), world->camera translation, same units as xyz_w.
        xyz_w: shape (N, 3), world-frame points, row vectors.

    Returns:
        xyz_c: shape (N, 3), camera-frame points (x right, y down, z forward).
    """
    xyz_w = xyz_w.reshape(-1, 3)
    return xyz_w @ R.T + t.reshape(1, 3)


def project_pinhole(
    xyz_c: torch.Tensor, fx: float, fy: float, cx: float, cy: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pinhole-project camera-frame points to pixel coordinates.

    Args:
        xyz_c: shape (N, 3), camera-frame points (see world_to_cam). z is
            depth along the optical axis; z > 0 is in front of the camera.
        fx, fy, cx, cy: pinhole intrinsics in pixels (python floats/tensors).

    Returns:
        uv: shape (N, 2), pixel coordinates (u right, v down; no distortion
            applied).
        depth: shape (N,), camera-space z (positive = in front).
    """
    xyz_c = xyz_c.reshape(-1, 3)
    depth = xyz_c[:, 2].clone()
    u = fx * xyz_c[:, 0] / depth + cx
    v = fy * xyz_c[:, 1] / depth + cy
    uv = torch.stack([u, v], dim=1)
    return uv, depth


# Below this rotation-vector magnitude (radians), _so3_exp_coeffs uses the
# Taylor series for A, B, C instead of the closed form. This is a numerical
# -stability threshold, not a differentiability one (see _so3_exp_coeffs):
# float64 already loses precision computing 1 - cos(theta) for theta below
# ~1e-4 rad, because cos(theta) rounds too close to 1.0 to recover the
# theta^2/2 term (theta^2/2 ~= 5e-9 at theta = 1e-4, near the edge of what
# float64's ~1e-16 relative precision can resolve against 1.0). Kept local to
# this module (not trippy/constants.py) because it is an implementation
# detail of this one series expansion, not a cross-module convention.
_SE3_TAYLOR_THETA = 1e-4


def _so3_exp_coeffs(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rodrigues/SE3 series coefficients A, B, C for rotation vector phi.

    A*K + B*K@K gives the SO(3) exponential (K = skew(phi)); the additional
    C term gives the SE(3) "V" matrix that maps the translation generator to
    the true translation:
        A = sin(t)/t, B = (1 - cos t)/t^2, C = (1 - A)/t^2,   t = |phi|.

    A, B, C are each even, smooth functions of t and can be written as smooth
    functions of t^2 = phi . phi alone (a polynomial in phi with no square
    root involved) -- so the branch below is taken on t^2 = phi @ phi rather
    than on t = |phi|, deliberately avoiding torch.linalg.norm(phi) (whose
    gradient phi/|phi| has a 0/0 singularity at phi == 0). This keeps every
    branch differentiable *through the coefficients* everywhere, including
    at phi == 0 (where dA/dphi = dB/dphi = dC/dphi = 0, correctly -- A, B, C
    are stationary at the origin because they are even in t). The rotation
    gradient at phi == 0 instead comes entirely from K = skew(phi), which is
    linear in phi and differentiates to the SO(3) generator there; see
    se3_exp and tests/test_raster_bwd_ref.py::
    test_pose_delta_rotation_gradient_matches_generator_at_zero.

    Uses the small-angle Taylor expansion below _SE3_TAYLOR_THETA to avoid
    the float64 cancellation in (1 - cos t) for tiny t (see that constant's
    comment); the exact 0/0 at t == 0 is avoided the same way, as a special
    case of the same small-angle guard.
    """
    theta2 = torch.dot(phi, phi)
    if theta2.item() < _SE3_TAYLOR_THETA * _SE3_TAYLOR_THETA:
        a = 1.0 - theta2 / 6.0
        b = 0.5 - theta2 / 24.0
        c = 1.0 / 6.0 - theta2 / 120.0
    else:
        theta = torch.sqrt(theta2)
        a = torch.sin(theta) / theta
        b = (1.0 - torch.cos(theta)) / theta2
        c = (1.0 - a) / theta2
    return a, b, c


def se3_exp(delta: torch.Tensor) -> torch.Tensor:
    """SE(3) exponential map for a pose-refinement twist.

    Builds R and V directly from K = skew(phi) (not from a normalized axis
    scaled back up by theta): axis = phi / max(|phi|, eps) is exactly
    second-order in phi at phi == 0 (an axis times a magnitude that is
    itself ~|phi|), which zeroed the rotation gradient at the identity
    delta -- see docs/LIMITATIONS.md history and
    tests/test_raster_bwd_ref.py::
    test_pose_delta_rotation_gradient_matches_generator_at_zero. Using
    K = skew(phi) directly is first-order in phi (linear), so its gradient
    at phi == 0 is the SO(3) generator, as it should be.

    Args:
        delta: shape (6,), twist xi = (rho, phi): rho = delta[:3] is the
            translation generator, phi = delta[3:] is the rotation vector
            (axis * angle, radians).

    Returns:
        T: shape (4, 4), homogeneous transform [[R, V @ rho], [0, 0, 0, 1]].
    """
    rho = delta[0:3]
    phi = delta[3:6]
    a, b, c = _so3_exp_coeffs(phi)

    K = _skew(phi)
    KK = K @ K
    eye = torch.eye(3, dtype=delta.dtype, device=delta.device)
    R = eye + a * K + b * KK
    V = eye + b * K + c * KK

    T = torch.eye(4, dtype=delta.dtype, device=delta.device)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def compose(pose: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Apply a pose-refinement delta to a COLMAP world->camera pose.

    Left-multiplicative (global-frame) update: new_pose = se3_exp(delta) @ pose.

    Args:
        pose: shape (4, 4), homogeneous world->camera transform.
        delta: shape (6,), twist passed to se3_exp.

    Returns:
        new_pose: shape (4, 4).
    """
    return se3_exp(delta) @ pose
