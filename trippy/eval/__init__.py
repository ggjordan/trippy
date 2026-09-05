"""trippy.eval: subprocess wrappers around Splats' own candidate-ranking audits.

Module: trippy.eval
Purpose: a small package (distinct from `trippy.train.eval`, which
    evaluates a checkpoint's own held-out metrics) holding wrappers that
    shell out to Splats' `depthprior_shade_audit.py` and `extent_gate.py`
    and parse their output into plain dicts -- see `trippy.eval.audits`.
Related docs: docs/EXPERIMENTS.md "Shade audit", "Extent gate"; docs/SPEC.md D10.
"""

from __future__ import annotations
