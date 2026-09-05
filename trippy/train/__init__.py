"""Training loop: config, trainable point/pose params, trainer, eval, PLY export.

Module: trippy.train
Invariants: trippy.train.export (write_gaussian_ply / export_pointset_ply)
    writes 3DGS-compatible PLYs that round-trip byte-exact through
    trippy.points.gaussian_ply.GaussianPlySource. trippy.train.config
    (TrainConfig), trippy.train.params (PointParams/PoseParams),
    trippy.train.trainer (Trainer), trippy.train.eval
    (evaluate_checkpoint/render_offpath), and trippy.train.checkpoint_io
    implement the full CPU-testable training loop (docs/ARCHITECTURE.md
    "train/" section); the Metal backward pass for `render_pyramid` is
    developed concurrently elsewhere and plugs in with no API change here.
Related docs: docs/SPEC.md v0.2.0 milestone (trainer with crops
    384-512 half-res, 150-250 epochs, pose refine after lock; eval/export
    writing a 3DGS-compatible PLY so existing Splats tooling runs unchanged);
    docs/GEOMETRY.md "3DGS PLY export mapping"; docs/EXPERIMENTS.md
    "Training runs".
"""
