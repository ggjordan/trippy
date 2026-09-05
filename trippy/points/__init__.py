"""Pluggable TRIPS point sources: Gaussian centres, monocular depth, union, LiDAR.

Module: trippy.points
Invariants: every PointSource.build() returns a trippy.points.source.PointSet
    (numpy, COLMAP world frame, float32); per-point provenance is a uint8
    carried through training (see trippy.constants PROVENANCE_* values).
Related docs: docs/SPEC.md D4 and docs/ARCHITECTURE.md "points/" (
    GaussianPlySource, ColmapSparseSource, MonoDepthSource/LidarSource
    stubs, UnionSource).
"""
