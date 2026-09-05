"""Camera geometry: two independent world<->camera<->pixel implementations.

Module: trippy.geom
Invariants: xform_a is numpy-only, xform_b is torch-only; they are written
    independently (different formulas) and cross-checked in
    tests/test_xform_agreement.py so a shared silent convention bug cannot
    hide between them.
Related docs: /tmp/trippy-plan.md "Verification (end-to-end)" item 1;
    AGENTS.md NEW section 8 "geometry implemented twice and made to
    disagree first".
"""
