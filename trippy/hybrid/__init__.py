"""trippy.hybrid: Gaussian-splat renders as network input (designs C and A).

Module: trippy.hybrid (package root)
Invariants: **design C** (`config_c`, `dataset_c`, `train_c`) is deliberately independent of
    trippy.train.trainer.Trainer (which is point-based / differentiable-rasteriser-based): it
    maps a *fixed*, already-rendered Gaussian-splat image (rgb + depth + alpha, from Splats'
    gsrender.py) to the photo via a small U-Net -- see trippy.hybrid.train_c.HybridCTrainer.
    **Design A** (`config_a`, `gaussian_input`, `gsrender_live`, `gate`) is the opposite: an
    *option on* that same Trainer, concatenating the render onto every level of the TRIPS
    pyramid. `gate` is design A's blend gate -- the extra U-Net output channel that makes the
    splat-vs-TRIPS mix explicit; it is off by default and a hard no-op then.
Related docs: docs/PLAN-2026-09-05.md "Hybrid (v0.3): (C) render->photo U-Net refinement on
    gsrender.py outputs first (cheap, validates net/losses)"; docs/SPEC.md "Milestones" v0.3.0
    row; experiments/EXP-0005-hybrid-c/README.md.
"""

from __future__ import annotations
