"""Dolly/off-path renderers, video assembly, honesty/contact sheets.

Module: trippy.render
Invariants: trippy.render.sheets (contact_sheet/side_by_side/colorize) and
    trippy.render.video (write_video, an ffmpeg subprocess pipe) are
    implemented; trippy.render.pyramid_render.render_frames is `trippy
    render`'s orchestration (scene + GaussianPlySource -> pyramid raster ->
    per-frame contact sheet + summary sheet + timing metrics.json, no U-Net
    yet); the dolly/off-path camera-path renderer itself is not yet.
Related docs: docs/SPEC.md "Technical design" (honesty sheet: raw
    level-0 composite | network output | coverage/provenance map) and
    milestone acceptance rows (contact sheets, dolly MP4s);
    experiments/EXP-0001-forward-pyramid/README.md.
"""
