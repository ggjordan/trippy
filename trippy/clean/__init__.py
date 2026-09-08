"""Design B: use a trained TRIPS model to delete fog Gaussians from its own seed splat.

Module: trippy.clean
Purpose: docs/SPEC.md D2's fallback path -- "a plain splat that
    incorporates TRIPS learning". Jordan's verdict of 2026-09-08 19:10 was
    that the full-scene TRIPS runs make the canopy shade cloud disappear
    but render "nothing like a photo" elsewhere, while the Gaussian splat
    "felt like being in the scene". So the splat stays the base and TRIPS
    is used only as a *classifier*: every Gaussian that seeded a TRIPS
    point inherits that point's learned confidence, and the low-confidence
    ones are deleted from the splat. Nothing else about a surviving
    Gaussian is altered -- its PLY row is copied through byte for byte.
Invariants:
    - The Gaussian <-> TRIPS point correspondence is recovered, never
      assumed (`trippy.clean.mapping`), and every run records the evidence
      it used in its `summary.json`.
    - A survivor's PLY row is copied verbatim (`trippy.clean.ply_filter`);
      this package never rewrites a colour, opacity, scale or rotation.
    - Nothing here opens, decodes or renders a photograph (AGENTS.md Sec 6).
      The only image it writes is a from-scratch top-down histogram of
      deleted-Gaussian density, which contains no photographic content.
Related docs: docs/SPEC.md (D2, Design B); docs/USER_GUIDE.md
    ("Cleaning a splat with TRIPS"); docs/EDITOR.md Sec 3 (the shade
    finder this reuses the audit rule from); trippy.train.prune (the audit
    region maths, shared verbatim).
"""

from trippy.clean.mapping import GaussianPointMapping, recover_mapping
from trippy.clean.ply_filter import PlyLayout, filter_ply, read_ply_layout
from trippy.clean.score import PointScores, load_point_scores
from trippy.clean.select import VARIANTS, CleanVariant, deletion_mask, variant_by_name

__all__ = [
    "VARIANTS",
    "CleanVariant",
    "GaussianPointMapping",
    "PlyLayout",
    "PointScores",
    "deletion_mask",
    "filter_ply",
    "load_point_scores",
    "read_ply_layout",
    "recover_mapping",
    "variant_by_name",
]
