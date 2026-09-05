"""Dolly/off-path renderers, video assembly, honesty/contact sheets.

Module: trippy.render
Invariants: trippy.render.sheets (contact_sheet/side_by_side/colorize) and
    trippy.render.video (write_video, an ffmpeg subprocess pipe) are
    implemented; the dolly/off-path renderer itself is not yet.
Related docs: docs/SPEC.md "Technical design" (honesty sheet: raw
    level-0 composite | network output | coverage/provenance map) and
    milestone acceptance rows (contact sheets, dolly MP4s).
"""
