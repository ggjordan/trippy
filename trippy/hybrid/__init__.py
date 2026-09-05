"""trippy.hybrid: Hybrid design C -- render->photo U-Net refinement.

Module: trippy.hybrid (package root)
Invariants: this package is deliberately independent of trippy.train.trainer.Trainer (which
    is point-based / differentiable-rasteriser-based). Design C instead maps a *fixed*,
    already-rendered Gaussian-splat image (rgb + depth + alpha, from Splats' gsrender.py) to
    the photo via a small U-Net -- see trippy.hybrid.train_c.HybridCTrainer.
Related docs: docs/PLAN-2026-09-05.md "Hybrid (v0.3): (C) render->photo U-Net refinement on
    gsrender.py outputs first (cheap, validates net/losses)"; docs/SPEC.md "Milestones" v0.3.0
    row; experiments/EXP-0005-hybrid-c/README.md.
"""

from __future__ import annotations
