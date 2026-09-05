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

from trippy.constants import EPS_QUAT_AXIS, EPS_SE3_ANGLE


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


def _so3_exp_coeffs(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rodrigues/SE3 series coefficients A, B, C for rotation-vector magnitude theta.

    A*K + B*K@K gives the SO(3) exponential; the additional C term gives the
    SE(3) "V" matrix that maps the translation generator to the true
    translation. Uses the small-angle Taylor expansion below EPS_SE3_ANGLE
    to avoid a 0/0 division at theta == 0 (the common case of a zero pose
    delta before any refinement).
    """
    if theta.item() < EPS_SE3_ANGLE:
        theta2 = theta * theta
        a = 1.0 - theta2 / 6.0
        b = 0.5 - theta2 / 24.0
        c = 1.0 / 6.0 - theta2 / 120.0
    else:
        a = torch.sin(theta) / theta
        b = (1.0 - torch.cos(theta)) / (theta * theta)
        c = (theta - torch.sin(theta)) / (theta * theta * theta)
    return a, b, c


def se3_exp(delta: torch.Tensor) -> torch.Tensor:
    """SE(3) exponential map for a pose-refinement twist.

    Args:
        delta: shape (6,), twist xi = (rho, phi): rho = delta[:3] is the
            translation generator, phi = delta[3:] is the rotation vector
            (axis * angle, radians).

    Returns:
        T: shape (4, 4), homogeneous transform [[R, V @ rho], [0, 0, 0, 1]].
    """
    rho = delta[0:3]
    phi = delta[3:6]
    theta = torch.linalg.norm(phi)
    theta_safe = torch.clamp(theta, min=EPS_SE3_ANGLE)
    axis = phi / theta_safe
    a, b, c = _so3_exp_coeffs(theta)

    K = _skew(axis)
    eye = torch.eye(3, dtype=delta.dtype, device=delta.device)
    R = eye + a * theta * K + b * theta * theta * (K @ K)
    V = eye + b * theta * K + c * theta * theta * (K @ K)

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
