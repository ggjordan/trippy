"""Design-B distillation pipeline: TRIPS checkpoint -> plain-Gaussian PLY.

Module: trippy.distill
Purpose: docs/SPEC.md D2 ("A plain splat that incorporates TRIPS learning
    (Design B) is a valid fallback path") and the Quest honesty note ("ship
    fallback: distilled Gaussians via the existing ~/Splats/tools/publish/
    path"). This package renders a trained TRIPS checkpoint's own network
    output at the training cameras plus a small number of near-path
    interpolated cameras (trippy.distill.cameras), writes a COLMAP-text
    image set from those renders (trippy.distill.colmap_writer, on top of
    trippy.scene.colmap_io's writers), builds the Brush CLI invocation that
    trains an ordinary 3DGS model on that image set
    (trippy.distill.brush_runner -- never runs it directly, see AGENTS.md
    "Brush's trainer must NOT be run outside the queue"), and compares the
    resulting distilled PLY against the training run's own baseline PLY and
    TRIPS export via Splats' shade/extent audits (trippy.distill.compare).
Related docs: docs/EXPERIMENTS.md "Distillation (design B)"; `trippy distill`
    (trippy.cli).
"""
