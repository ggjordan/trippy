"""Training loop: config, trainer, eval, PLY export.

Module: trippy.train
Invariants: trippy.train.export (write_gaussian_ply / export_pointset_ply)
    writes 3DGS-compatible PLYs that round-trip byte-exact through
    trippy.points.gaussian_ply.GaussianPlySource; trainer/eval are not
    implemented yet.
Related docs: docs/SPEC.md v0.2.0 milestone (trainer with crops
    384-512 half-res, 150-250 epochs, pose refine after lock; eval/export
    writing a 3DGS-compatible PLY so existing Splats tooling runs unchanged);
    docs/GEOMETRY.md "3DGS PLY export mapping".
"""
