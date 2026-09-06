"""The viewer editor's Python side: `edits.json`, weight composition, shade finder, publish.

Module: trippy.edit
Purpose: everything docs/EDITOR.md's milestones need on the Python side,
    ahead of the Rust viewer UI: the `edits.json` schema and undo log
    (`trippy.edit.model`), per-point blend-weight composition
    (`trippy.edit.weights`), the shade-cloud finder built on
    `trippy.train.prune` (`trippy.edit.shade_finder`), and the
    `trippy apply-edits` publish path (`trippy.edit.apply`). `trippy.cli`
    wires all four into `trippy apply-edits` / `trippy edits <subcommand>`.
Invariants: pure numpy (+ stdlib json/dataclasses); no torch, no MPS, no
    Rust. Every array here is world-space and CPU-resident, matching what
    the viewer already holds (docs/EDITOR.md Sec 0).
Related docs: docs/EDITOR.md; docs/decisions/ADR-0007-viewer-editing.md.
"""

from __future__ import annotations
