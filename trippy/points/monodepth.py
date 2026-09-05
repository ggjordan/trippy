"""Stub PointSource for monocular-depth-derived points (D4 point source 2).

Module: trippy.points.monodepth
Invariants: build() always raises NotImplementedError; this module exists
    to fix the constructor signature and document the planned data flow
    ahead of the v0.2.0 implementation (see docs/SPEC.md milestone table).

Planned inputs (not read by this stub):
    - DepthPro (or MoGe) per-frame depth maps as .npy arrays, produced by
      `~/Splats/tools/ldi/depth_batch.py` on the scene's training images.
    - The scene's COLMAP sparse points (trippy.points.colmap_sparse or
      trippy.scene) reprojected into each camera to get sparse ground-truth
      depth samples.
    - Per-frame scale/shift alignment: monocular depth is scale-ambiguous,
      so each frame's dense depth map is rescaled by the *median ratio*
      between its predicted depth and the reprojected sparse COLMAP depths
      at the same pixels (a robust single-scalar-per-frame correction).
    - Backprojection of the aligned dense (or masked/subsampled) depth map
      through each camera's intrinsics/pose into world-frame xyz, plus
      colour sampled from the source image and a confidence derived from
      local depth-gradient smoothness or a predicted uncertainty channel.

None of the above is implemented yet -- see docs/SPEC.md v0.2.0 milestone
("MonoDepthSource") for the acceptance criteria this will need to meet.
"""

from __future__ import annotations

from pathlib import Path

from trippy.points.source import PointSet, PointSource


class MonoDepthSource(PointSource):
    """Planned: monocular-depth points, aligned to sparse COLMAP depth.

    Args:
        colmap_dir: text-format COLMAP sparse model directory (poses,
            intrinsics, and points3D.txt for scale alignment).
        depth_dir: directory of per-frame DepthPro/MoGe .npy depth maps
            (one file per training image, same basename).
        images_dir: directory of the corresponding RGB training images
            (for colour sampling at backprojected points).
        max_points_per_frame: cap on backprojected points per frame, to
            bound total point count when combined via UnionSource.
    """

    def __init__(
        self,
        colmap_dir: str | Path,
        depth_dir: str | Path,
        images_dir: str | Path,
        max_points_per_frame: int | None = None,
    ) -> None:
        self.colmap_dir = Path(colmap_dir)
        self.depth_dir = Path(depth_dir)
        self.images_dir = Path(images_dir)
        self.max_points_per_frame = max_points_per_frame

    def describe(self) -> dict:
        return {
            "type": "MonoDepthSource",
            "colmap_dir": str(self.colmap_dir),
            "depth_dir": str(self.depth_dir),
            "images_dir": str(self.images_dir),
            "max_points_per_frame": self.max_points_per_frame,
            "status": "not implemented (docs/SPEC.md v0.2.0)",
        }

    def build(self) -> PointSet:
        raise NotImplementedError(
            "MonoDepthSource.build() is not implemented yet -- see docs/SPEC.md "
            "v0.2.0 milestone ('MonoDepthSource') for the planned depth-alignment "
            "and backprojection pipeline."
        )
