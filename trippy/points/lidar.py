"""Stub PointSource for iPhone LiDAR points (D4 point source 4, "later").

Module: trippy.points.lidar
Invariants: build() always raises NotImplementedError; this module exists
    to fix the constructor signature and document the planned data flow.
    Per docs/SPEC.md D4 ("Revisit iPhone LiDAR later. Nothing to add to
    Revisit."), this source consumes an export from the existing Revisit
    app rather than adding new capture code anywhere.

Planned inputs (not read by this stub):
    - `points_lidar.ply`: a LiDAR point cloud exported by the Revisit app
      (ARKit scene reconstruction mesh vertices or raw depth-camera point
      cloud, already in the Revisit session's local frame).
    - ARKit poses for the same capture session, used to transform the
      LiDAR points from the Revisit/ARKit world frame into the scene's
      COLMAP world frame (a similarity transform -- rotation, translation,
      and a scale factor, since ARKit's world scale is metric but COLMAP's
      is defined up to an unknown global scale unless the scene was
      registered against known metric distances).
    - Optional colour: ARKit's per-vertex colour if present, otherwise
      colour sampled from the nearest captured frame.

None of the above is implemented yet -- see docs/SPEC.md D4 for context.
"""

from __future__ import annotations

from pathlib import Path

from trippy.points.source import PointSet, PointSource


class LidarSource(PointSource):
    """Planned: Revisit-app LiDAR points, aligned to the COLMAP world frame.

    Args:
        lidar_ply_path: path to `points_lidar.ply` exported by Revisit.
        arkit_poses_path: path to the matching ARKit pose/session export
            used to compute the ARKit-to-COLMAP similarity transform.
    """

    def __init__(self, lidar_ply_path: str | Path, arkit_poses_path: str | Path) -> None:
        self.lidar_ply_path = Path(lidar_ply_path)
        self.arkit_poses_path = Path(arkit_poses_path)

    def describe(self) -> dict:
        return {
            "type": "LidarSource",
            "lidar_ply_path": str(self.lidar_ply_path),
            "arkit_poses_path": str(self.arkit_poses_path),
            "status": "not implemented (see docs/SPEC.md D4; consumes a Revisit app export)",
        }

    def build(self) -> PointSet:
        raise NotImplementedError(
            "LidarSource.build() is not implemented yet -- see docs/SPEC.md D4. "
            "This source is meant to consume an existing export from the Revisit "
            "app (points_lidar.ply + ARKit poses); nothing new is added to Revisit."
        )
