"""Scene loading: COLMAP I/O, dataset/dataloader, train/eval splits.

Module: trippy.scene
Invariants: submodules never write outside a caller-supplied cache_root
    (dataset.py) and never read/copy scene files into the repo (AGENTS.md
    "Disk and delivery"). colmap_io.py duplicates no text parsing --
    trippy.geom.xform_a owns that; colmap_io.py adds the binary reader and
    the ColmapScene dataclass on top.
Related docs: docs/ARCHITECTURE.md "Module overview" (trippy/scene);
    docs/SPEC.md v0.1.0 milestone (dataset with undistort+cache at
    1008/2016 wide); docs/GEOMETRY.md "Undistortion and image cache".
"""
