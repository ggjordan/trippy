# Changelog
All notable changes to trippy. Format: Keep a Changelog. Versions: semver tags `vX.Y.Z`. Every push also gets a `build-NNNN` tag.

## [Unreleased]
### Delivered
- **Combined TRIPS+splat bundle for the full Karekare-v2 scene** (`kkv2-1-combined-viewer.command`):
  `kkv2-1-full-masked`'s TRIPS checkpoint re-exported with its Gaussian block (`blend.splat_ply` ->
  `kklid_20000.ply`, via the `feat/live-splat` export path every Gaussian-seeded run already qualifies
  for, just not yet re-run for this bundle) plus an `edits.json` preset -- "Big tree shade (TRIPS)"
  (758,178 points from the shade-cloud finder on the 93 measured big-tree frames, `mix=1.0`) full TRIPS,
  everything else full splat (`--blend-mode mix --mix 0.0`) -- so the per-region mix is visible on open.
  Alignment (TRIPS points vs their nearest splat Gaussian, reprojected into 3 training cameras): median
  0.87 px. No code changed; see `research/trips-metal.md` 2026-09-08 for the full numbers.

### Added
- **Lid plane-normal gizmo handle** (`docs/EDITOR.md` Sec 1, Sec 6): a 4th
  handle on a `lid` region, projected along its own `up` instead of a world
  axis, drags the plane's tilt (rotating about the two in-plane axes) as one
  undo entry; the Inspector's typed `up` still works, unchanged, alongside it.
- **Brush npz sidecar writer, viewer side** (`docs/EDITOR.md` Sec 1 "brush"):
  `EditDocument::save` now externalises a brush region above
  `EDIT_BRUSH_NPZ_CELL_THRESHOLD` cells into an `edits_brush_<id>.npz`
  sidecar, matching the Python CLI's writer array-for-array (a from-scratch
  `.npz` writer, `edit::npz_write`, no new dependency), closing the one
  deviation the previous release's brush entry noted.
- New headless viewer flags: `--bench-brush-anchor <n>` (brute force vs the
  new screen-space grid, ms/sample) and `--brush-npz-selftest <dir>` (writes
  a synthetic over-threshold brush region's sidecar with no bundle needed,
  for the Python/Rust cross-language parity test) and `--save-edits <p>`
  (write the document after any headless mutation).

### Fixed
- **The brush's per-sample depth anchor was an unmeasured `O(points)` scan.**
  Measured at 47.4 ms/sample on the real Karekare-scale bundle (7.5M points)
  -- too slow for a stroke sampling several times a second. `edit::brush::
  ScreenGrid`, a screen-space bucket index built once per camera pose, drops
  this to 0.011 ms/sample after a 325.5 ms one-time build, with the same
  exact result as the brute-force scan (checked against it directly, not
  approximated).

## [v0.6.0] - 2026-09-08
### Added
- **The viewer's brush tool, Named Objects panel and 3D drag gizmos
  (`docs/EDITOR.md` Sec 1, Sec 4, Sec 6).** The last three viewer pieces that
  document was still a spec for.
  - **Brush**: with the `brush` tool focused (`T`), a drag on the render paints
    spheres of `brush_radius` into a `brush`-kind region, each dab anchored at
    the depth of the nearest point under the cursor (`edit::brush::depth_anchor`
    + `ClickCamera::unproject` -- the click tool's own projection, inverted, so
    no depth buffer is needed here either). **Alt-drag erases**, `[` / `]` change
    the radius, a weight slider grades the painted cells, and **one stroke is one
    undo entry** however many frames the drag took
    (`EditDocument::update_region_coalesced` / `add_region_coalesced`: the
    gesture's frames replace each other in the log rather than piling up; the
    file format is unchanged and Python replays them like any other entry).
    Later strokes go into the same region until "start a new region" is pressed.
    `rust/crates/trips-viewer/src/edit/brush.rs` is a line-for-line twin of
    `trippy.edit.model`'s `brush_membership`/`paint_sphere`/`paint_along`/
    `erase`, down to the box-sphere overlap test and the first-painted cell
    order.
  - **Named Objects panel**: every region with its name, source tool, kind, live
    point count, enable toggle, mix slider, rename, remove and **solo** (render
    only that region's effect), grouped by `Region.source.tool` with
    hand-authored regions in their own group. Regions made in the viewer get the
    same auto-numbered names the CLI gives (`click-1`, `brush-2`,
    `sam-box-IMG_3703-3`), from `trips_viewer::edit::model::auto_region_name`.
    Solo is a way of looking and is never written to `edits.json`.
  - **3D drag gizmos** (`src/edit/gizmo.rs`): three projected world-axis handles
    (red/green/blue) on the selected box, sphere or lid, drawn by the egui
    painter over the finished frame -- so `--screenshot` shows the edit and never
    the tool. Drag a handle to translate along that axis, **Shift-drag** to
    resize, **Ctrl-drag** to rotate a box. A drag that does not START on a handle
    still orbits, so navigation loses nothing; the arrow-key nudges and `[`/`]`
    resize stay and now share `gizmo::translated`/`gizmo::resized` with the drag.
  - **Parity, pinned by fixtures**: `edit_golden/brush.json` now records the
    three AUTHORING calls as well as their result, and the Rust twin must replay
    them to the same cells, in the same order, with the same weights (a port that
    voxelised by cell-centre would produce a smaller set that still passed every
    membership lookup); `edit_golden/names.json` is new and pins
    `auto_region_name` case for case. `trips-viewer --dump-weights` matches
    `trippy.edit.weights.compose_trips_weights` exactly for a brush region on a
    real `points.npz` (`tests/test_edit_viewer_parity.py`).
  - **Headless twins** for scripts and proofs: `--brush U V`, `--brush-to U V`,
    `--brush-radius/-weight/-op/-mix`, `--brush-erase`, `--brush-undo`,
    `--move-region <ID|name> DX DY DZ` (the translate gizmo) and
    `--solo <ID|name>`. Measured on the synthetic bundle at 480x360: a dab with
    `op = delete` removes 110 of 4 000 points and changes **4.21 %** of the
    frame, a stroke **8.51 %**, `--brush-undo` is **byte-identical** to the
    unedited frame, `--move-region` changes **8.22 %**, and `--solo` **3.77 %**.
- **Brush regions (Python side), named regions, and the `trippy eval --edits`
  gate-suppression gap is closed (`docs/EDITOR.md` Sec 1, Sec 5).**
  - **`brush`-kind regions**: a sparse voxel set (`origin`/`cell_size`/`cells`,
    optional per-cell `weights`) is now one of `Region.kind`'s options
    (`trippy.edit.model`). `brush_membership` looks a point's own voxel up in
    the occupied set; `paint_sphere`/`paint_along`/`erase` are the pure,
    functional authoring helpers a brush tool calls (box-sphere intersection,
    not "voxel centre inside the sphere"), and `trippy edits add-brush` is
    the CLI equivalent of one stroke. It composes through the SAME
    `region_weight`/`region_contains` dispatch every other kind does, so
    `trippy.edit.weights`/`apply`/`checkpoint` needed no brush-specific code
    at all. `EditDocument.save` externalises a large brush's `cells`/
    `weights` into an `.npz` sidecar (`EDIT_BRUSH_NPZ_CELL_THRESHOLD`) in the
    WRITTEN copy only; Python's own loader always replays the (never
    externalised) undo log. `tests/fixtures/synthetic/edit_golden/brush.json`
    (`trippy.edit.golden.build_brush_fixture`) is the parity fixture the Rust
    `brush` twin was then built against (see the entry above).
  - **Named regions**: `Region.source` (`{"tool": ..., ...}` or `None`)
    records which tool made a region and with what prompt/parameters;
    `trippy.edit.model.auto_region_name` gives tool-authored regions a
    stateless, self-healing numbered name (`click-1`, `sam-box-IMG_3703-2`,
    `shade-clouds-3`, `brush-4`, ...) when the caller does not name one
    explicitly. `shade_finder`/`cluster`/`sam_lift` all fill both fields now.
    `trippy edits list/rename/toggle/remove --edits edits.json` is the new
    CLI for a future Named Objects panel to script against.
  - **`trippy eval --edits` gate multiply**: `Trainer.evaluate` now calls
    `trippy.edit.checkpoint.render_edit_weight_map` itself, exactly where
    `trippy.render.candidate.render_candidate` already did, so a `blend`/
    `fade` region's gate suppression shows up in `trippy eval` too, not only
    in `candidate-report`/`distill --stage render` -- closing the gap
    `trippy.edit.checkpoint`'s own module docstring used to document. The
    non-edit path is unchanged: `render_edit_weight_map` returns `None`
    whenever there is nothing to suppress, so an unedited `trippy eval` is
    bit-identical to before.
- **The SAM 3 lift is in the viewer (`docs/EDITOR.md` Sec 3 "5.", Sec 6's E5
  row): drag a box on the render and `trippy edits sam` runs as a child
  process.** E5's viewer half. The viewer segments nothing itself -- SAM 3 is
  never imported into trippy's process and never into the viewer's -- it is a
  process supervisor around the command that already existed.
  - **A `SAM 3 lift` tool** in `src/edit_ui.rs` (`T` now cycles Regions ->
    shade-cloud finder -> click-to-cluster -> SAM 3 lift): a primary **drag**
    on the render draws a marquee and becomes `--box`, **Alt-click** becomes
    `--point`, with op/mix/views-around/device controls, a **run** button, a
    **Cancel** button that kills the child, and the child's own stdout
    streaming into a log pane while it works. Shift-drag still orbits and
    right/middle-drag still pan, so the gesture is borrowed for one tool
    rather than taken (`src/app.rs`).
  - **The camera must be pinned to a capture view**, because the lift needs the
    PHOTOGRAPH. Off a view the gesture is discarded and the camera snaps to the
    nearest capture view (`edit::sam::nearest_view`) with the panel saying which
    one -- re-using pixels measured on a free-flying frame would segment the
    wrong part of the image while looking like it worked.
  - **The render-pixel -> view-pixel mapping** (`src/edit/sam.rs`,
    unit-tested). `Controller::render_camera` scales `fy` by the WIDTH ratio
    and `cy` by the HEIGHT ratio, so a plain height ratio in `v` is wrong on a
    window-shaped render; the mapping goes through normalised image
    coordinates and is exact for every camera sharing the view's `R`/`t`. Its
    tests project world points through both cameras and compare, and one test
    exists purely to fail if someone "simplifies" it back to a ratio.
  - **The child-process state machine** (`src/sam_child.rs`, unit-tested
    against a `/bin/sh` fake child): one child at a time, killed on Cancel
    **and on drop**; two reader threads so a full stderr pipe cannot deadlock
    it; stderr shown prefixed `!` and never parsed as JSON; exit 0 with no
    summary line treated as a FAILURE rather than a silent success; the
    interpreter resolved as `<trippy_root>/.venv/bin/python` from
    `bundle.json`, overridable with `$TRIPPY_ROOT` / `$TRIPPY_PYTHON`.
  - **The import is one undo step.** The child writes into a throwaway
    `edits.json` under `TMPDIR`; the viewer reads the last region out of it,
    gives it a fresh id and adds it through `EditDocument::add_region`, so
    `Cmd-Z` removes it. The lifted points are tinted immediately (`H` toggles).
  - **Headless `--sam-box X0 Y0 X1 Y1`** (plus `--sam-point`,
    `--sam-views-around`, `--sam-op`, `--sam-mix`, `--sam-device`,
    `--sam-fake`, `--sam-undo`). E5's screenshot proof on the generated
    `synthetic-splat` bundle, box `12 9 36 27` on `IMG_0.jpg`, `--sam-fake`:
    530 of 4000 points lifted; with `--sam-op delete` 1719 of 1728 pixels
    differ from the baseline frame (mean 6.15/255); with `--sam-op fade` the
    tint alone moves 1717 pixels (mean 12.50/255); and `--sam-undo` gives a
    frame **byte-identical** to a run with no `--sam-box` at all.
- **`trippy edits sam --fake` / `TRIPPY_SAM_FAKE=1`**: synthesise the mask from
  the prompt (a box fills its rectangle, a point fills a disc) and run the
  entire rest of the shipped path -- projection, depth gate, majority vote,
  region, summary -- with no SAM 3, no 3.4 GB checkpoint and no GPU. This is
  what makes the viewer's child-process path testable in CI and screenshottable
  (`trippy/edit/sam_runner.py::FakeSegmenter`). The summary records
  `"segmenter": "fake"` and never claims SAM ran.
- **`trippy edits sam --scene` is now optional**, because `bundle.json` records
  `scene_root` (and `trippy_root`) since this release
  (`trippy.render.bundle.bundle_document`). The viewer omits `--scene`
  entirely; a bundle exported before this has no such key, and both the CLI and
  the SAM panel say so rather than guessing where the photographs are.
- **`trippy edits sam --prompt-space photo|view`**: the viewer measures its
  drag-box in the bundle VIEW's pixel grid (it never opens a photograph, so it
  cannot know the photo's size); `view` multiplies the prompt by that scene's
  own `photo_scale` before SAM sees it. Default `photo`, unchanged.
- **`trippy edits sam`'s output is now a contract**: progress goes to stdout
  one `sam: ` line at a time (so the viewer can show it live) and the LAST
  stdout line is the whole summary as one compact JSON object.

- **Click-to-cluster in the viewer (`docs/EDITOR.md` Sec 3 "1.", Sec 4, E4):
  SHIFT-CLICK the render to select an object.** The Rust half of E4, a port of
  `trippy/edit/cluster.py` into `rust/crates/trips-viewer/src/edit/cluster.rs`:
  every point is projected with the camera the frame was actually drawn with
  (`ClickCamera::from_render_camera`; pinhole, no distortion, exactly as the
  Python side), the points landing within `radius_px` of the click and in front
  of the camera are collected, the group NEAREST the camera among them seeds the
  selection (so a click does not reach through a gap onto a same-coloured
  surface behind), and the region grows from that seed by k-NN hops in 3D gated
  by colour distance, a hard `max_radius` from the seed centroid and a
  `max_points` cap.
  - **A Selection panel** in `src/edit_ui.rs` (`T` now cycles Regions ->
    shade-cloud finder -> click-to-cluster): four sliders (radius px, colour
    tol, max radius, max points) that re-run the SAME click when moved, the
    magenta preview tint (`H`, reusing the shade finder's tint path), the
    candidate/seed/selected counts and timings, an op (`blend`/`fade`/`delete`)
    + mix chooser, **add as region** (commits a `pointset` region through the
    ordinary undo log) and **clear** (drops the selection, the tint and the
    neighbour index).
  - **Shift-click** is read from the scene `Response` in `src/app.rs` and scoped
    to `clicked()`, so a Shift-DRAG still orbits and no navigation gesture was
    taken away. The clicked pixel is converted into the render's own
    coordinates, and the camera is now built before the editor's per-frame work
    so a click is projected with the frame it was made on.
  - **An exact k-nearest spatial hash** (`PointGrid`) instead of a k-d tree: no
    k-d tree crate is vendored by either workspace's lock, so nothing was added
    to `Cargo.toml`. It expands cell rings until the k-th distance found is
    provably inside the scanned region (and falls back to a linear scan in a
    void), and sizes its cells from the cloud's INTERQUARTILE extent so a TRIPS
    export's far-field environment sphere cannot collapse the scene into one
    cell. Proved equivalent to a brute-force search in its own unit tests.
  - **Headless `--click U V`**, plus `--dump-click <o>` (write the selected ids
    as JSON and exit, no GPU) and `--click-radius-px` / `--click-colour-tol` /
    `--click-max-radius` / `--click-max-points`. With `--screenshot` the
    selection is tinted into the frame, which is E4's own screenshot proof.
  - **Exact Python/Rust parity, pinned twice.**
    `tests/fixtures/synthetic/edit_golden/click.json` +
    `expected_click.json` (written by `trippy.edit.golden.build_click_fixture`)
    carry a structured synthetic scene -- a red blob, a same-coloured blob
    behind it, a green blob touching it, and scatter -- and four clicks that
    each pin a different branch: depth-mode seeding, the colour gate, the
    `max_points` cut-off part-way through a frontier, and a miss. Both languages
    must return the IDENTICAL id list, not a tolerance;
    `rust/crates/trips-viewer/src/edit/cluster.rs`'s "Tie-breaking" section
    names the three places they could legitimately differ (k-th-neighbour
    distance ties, depth-sort ties, summation order in the seed centroid) and
    `tests/test_edit_golden.py` pins the fixture's margins so none is reachable.
    `trips-viewer --click U V --dump-click` runs the same comparison against a
    real bundle (`tests/test_edit_viewer_parity.py`).
  - Measured on the synthetic bundle at 480x360 (four sub-10-second local
    `--screenshot` launches): a click at (240, 180) selects **261 of 4 000
    points**; the preview tint changes **71.55 %** of the frame's pixels (3.81 %
    turning magenta, the rest dimmed by the preview) with a max channel diff of
    **79/255**; and a run without `--click` reproduces the untinted frame **bit
    for bit** (0.0000 % of pixels differ, max channel diff **0**).
- **Click-to-cluster, Python side (`docs/EDITOR.md` Sec 3 "1.", E4): `trippy
  edits click`.** New `trippy/edit/cluster.py`: projects every point of a
  bundle's `points.npz` into a named view (`trippy.geom.xform_a`, numbers
  only, no distortion model), takes the points whose projection lands
  within `--radius-px` of the click, picks the nearest-camera depth MODE
  among them as the seed (so a click through a gap onto a same-coloured
  surface behind the clicked object does not select the background), and
  grows a `pointset` region from that seed by repeated k-NN queries
  (`scipy.spatial.cKDTree`, the same query shape `trippy.points.knn_size`
  uses) gated by a colour-distance threshold (`feat[:, :3]`, the same base
  colour `trippy.edit.shade_finder` reads) and a hard `max_radius` (world
  units, defaulting to the bundle's own median nearest-CAMERA spacing —
  not point density) and `max_points` cap.
  - `trippy edits click --bundle DIR --view IMG_xxxx.jpg --px U V
    [--radius-px 12] [--colour-tol 0.15] [--max-radius R] [--max-points
    200000] [--knn-k 16] [--depth-gap-factor 1.0] --op fade|delete|blend
    --mix 0.5 --out edits.json [--preview heatmap.png]` appends the region
    to `edits.json` (created if missing, same convention as `add-box`/
    `add-sphere`/`add-lid`/`shade-find`).
  - `--preview` writes a from-scratch heatmap PNG (a synthetic point-density
    scatter of the selected points' own projections, no photograph content —
    `AGENTS.md` Sec 6's allowed "abstract heatmap" case) plus the selection
    counts in the printed JSON summary.
  - 17 new CPU tests in `tests/test_edit_cluster.py`: an isolated coloured
    blob is selected and a same-coloured blob behind it through a depth gap
    is not; the colour gate stops growth at a touching, differently-coloured
    cluster's boundary; `max_radius`/`max_points` caps are respected; the CLI
    round-trips through `EditDocument.load`. Synthetic point clouds only.
  - The viewer's own click handler (casting a ray from `camera.rs::Controller`
    and calling this) is a later, Rust-side task — not built here.
- **SAM-3 mask lift (`docs/EDITOR.md` §3 "4. SAM 3 lift", E5): `trippy edits
  sam --bundle <dir> --scene <root> --view IMG.jpg (--point U V | --box X0 Y0
  X1 Y1 | --text "...") [--views-around N] --out edits.json`.** Segments ONE
  registered photograph with the local SAM 3 and lifts the mask onto the
  bundle's own TRIPS points as a `pointset` region (same shape the shade-cloud
  finder produces). New `trippy/edit/sam_runner.py` — both halves of a
  subprocess pipe: imported it builds/runs the command, executed by *Splats'*
  SAM venv python it imports `sam3` by `sys.path` from
  `~/Splats/tools/sam3/repo`, builds the image model from the local
  `sam3.pt` (`load_from_HF=False`, nothing is ever downloaded) and writes a
  mask `.npy` + `info.json`. Masks are arrays; no photo, mask or overlay is
  ever rendered as an image. New `trippy/edit/sam_lift.py` — projection,
  depth gate and cross-view vote, all pure numpy with the segmenter injected,
  so the whole pipeline is CPU-testable with a fake (`tests/test_edit_sam.py`,
  19 tests).
  - Depth gate: points are binned into 16 px cells and kept within a relative
    band of their cell's NEAREST SUPPORTED depth mode (log bins of width
    `--depth-tol`, a bin counting as a surface at 25% of the cell's fullest
    bin) — nearest-with-support, not the plain mode, because a mask over a
    near object usually contains more background than object.
  - Cross-view vote: `--views-around N` re-segments the N nearest capture
    views that can see the selection's centroid, prompting SAM there with that
    centroid's own projected pixel, and keeps a point when MORE than
    `--vote-fraction` of the views that can see it voted for it.
  - `--preview` writes a from-scratch heatmap of the projected selection
    (counts per cell, two flat colours — no photographic content);
    `--sam-work-dir` keeps each view's `mask.npy`/`info.json`; `--mask
    NAME=PATH` re-lifts an existing mask array instead of running SAM and
    records `segmenter: "mask-file"` rather than claiming SAM ran.
  - `--mask-threshold` exposes the per-pixel probability cutoff
    `Sam3Processor` hardcodes at 0.5, because on a low-contrast subject SAM 3
    can be right about where the object is while peaking below it and return
    only the outline (`docs/LIMITATIONS.md` has the measurement).
  - `--device mps` is GPU work and only ever runs inside a
    `scripts/gpu_submit.sh` job; the default is `cpu`.
- **Viewer editor publish path finished (`docs/EDITOR.md` Sec 5, E6): `trippy
  apply-edits --target trips|distilled|both`, `export.ply`, distilled-PLY
  reapply, and `--edits` on `trippy candidate-report`/`trippy eval`/`trippy
  distill --stage render`.** New `trippy/edit/checkpoint.py`
  (`apply_edits_to_trainer` — the same `Trainer._apply_keep_mask` index-select
  surgery training uses, applied to a live checkpoint's `Trainer`, plus
  `render_edit_weight_map` — the per-point edit weight splatted as an
  auxiliary `render_pyramid` feature channel and read back from level 0, the
  documented fallback for "per-pixel weight rendering is not yet available in
  the Python renderer"). On a gate-hybrid checkpoint, `trippy.render.candidate.
  render_candidate` multiplies the trained blend gate by that per-pixel
  projection wherever a `blend`/`fade` region touches a surviving point, so an
  un-edited live splat cannot leak back into a deleted/faded region; `trippy
  eval` applies the keep-mask deletion only (a documented gap, not a silent
  one — `Trainer.evaluate`'s own render path is out of reach without editing
  `trippy/train/trainer.py`).
  - `trippy apply-edits --target trips` (now also the default half of `both`)
    additionally writes a filtered 3DGS-style `export.ply` of the kept TRIPS
    points via `trippy.train.export.write_gaussian_ply`.
  - `trippy apply-edits --target distilled --distilled-ply <path>` re-applies
    `box`/`sphere`/`lid` regions directly to an already-distilled Gaussian PLY
    (`trippy.edit.apply.apply_gaussian_ply_edits`): `delete` removes rows,
    `fade` scales the surviving row's own alpha and rewrites `opacity`
    ("fade → opacity scaling", since a plain Gaussian PLY has no TRIPS-vs-splat
    mix channel); `pointset` regions are skipped (do not survive distillation)
    and trigger a `summary["distilled"]["warning"]` if enabled.
  - `trippy distill --stage render/all --edits edits.json` deletes the edited
    region's points from the checkpoint's cloud BEFORE any camera renders (the
    edit-then-distil ordering ADR-0007 requires), so deleted content never
    reaches the images Brush trains on.
  - `trippy.edit.weights.compose_point_weights` gained an `ops` filter and a
    new `compose_gaussian_opacity_scale` wrapper for the distilled-PLY reapply
    path (honours `delete`/`fade` only, ignores `blend`).
  - 19 new CPU tests across `tests/test_edit_checkpoint.py` (new),
    `tests/test_edit_apply.py`, `tests/test_edit_weights.py`,
    `tests/test_cli_edits.py`, `tests/test_cli_candidate_report.py`,
    `tests/test_cli_distill.py`, `tests/test_distill_render_set.py`,
    `tests/test_train_eval.py`, `tests/test_train_cli.py`; synthetic fixtures
    only. See `docs/EXPERIMENTS.md` "Edits" for the worked run and full list.
- **Viewer editor, Rust side: E1 + E2 of `docs/EDITOR.md` (press `M`).** The
  Regions / Inspector / Tools panels, the render integration, and a headless
  parity check against the Python half.
  - New `rust/crates/trips-viewer/src/edit/`: `model.rs` (a line-for-line twin of
    `trippy/edit/model.py` -- the same `Region`/`EditDocument`, the same
    append-only undo log replayed the same way, the same oriented-box / sphere /
    lid / pointset membership arithmetic in f64), `weights.rs` (the twin of
    `trippy/edit/weights.py`: `blend` sets, `fade` multiplies, `delete` zeroes and
    latches), `apply.rs` (composed weights -> the two point sets the rasteriser
    draws) and `shade.rs` (the shade-cloud finder's arithmetic, ported from
    `trippy/train/prune.py`, so four sliders re-threshold at interaction speed
    with no subprocess in the loop). `src/edit_ui.rs` is the egui half.
  - **Regions panel**: create box / sphere / lid at the camera's look-at point,
    sized from `SceneScale` (never the point cloud's bounds); per-region mix
    slider (0 = splat, 1 = TRIPS), op (blend / fade / delete), enable toggle,
    reorder, delete, undo / redo, and `Cmd-S` save. Reopening the bundle restores
    the region list, the mixes AND the undo history, because `edits.json` carries
    the log and the cursor verbatim.
  - **Render integration**, hot-applied when a region changes and never per frame:
    `delete` removes rows from `PointSet` before rasterisation and scales the
    matching Gaussians' opacity to zero (`LiveSplat::set_opacity_scale`, a logit
    round trip through `sigmoid`, with the ply's own raw values kept so an undo is
    exact); `blend`/`fade` ride a **second three-channel pyramid pass** carrying
    `[w_edit, touched, 1]` over the same rows with the same `conf`, so level 0's
    channels divide into the alpha-weighted average `docs/EDITOR.md` §2 specifies.
    `C = 3` is `SUPPORTED_CHANNELS`' existing pipeline: no kernel change, no new
    fixture, no risk to the rasteriser's parity tests. The extra pass is paid only
    while an enabled `blend`/`fade` region exists.
  - **Shade-cloud finder** (E2): live sliders for luminance, confidence and the
    znear/zfar depth-slab fractions, over a `shade_views.json` sidecar written by
    the new `trippy.edit.golden.write_shade_views`; a preview that tints the
    selection magenta and dims the rest; and one click to commit it as a
    `pointset` region with op `fade` or `delete`.
  - **Python/Rust parity, pinned two ways.** New `trippy/edit/golden.py` writes
    `tests/fixtures/synthetic/edit_golden/` (a synthetic cloud, an `edits.json`
    exercising every kind and op with a live undo cursor, and the weights + shade
    selection Python computes from them); `tests/test_edit_golden.py` proves the
    fixture is still what Python produces, and the Rust `edit::golden` tests prove
    the viewer reproduces it to **1e-6**. New headless `trips-viewer
    --dump-weights` (no GPU, no window) and `--dump-shade` run the same comparison
    against a real bundle (`tests/test_edit_viewer_parity.py`).
  - New viewer flags: `--edit` (open with the panels up), `--edits <p>`,
    `--dump-weights <o>`, `--dump-shade <o>`,
    `--shade-lum/-conf/-znear/-zfar`. `--screenshot` now applies `edits.json` too,
    so the headless picture is the window's picture.
  - `tools/make_synthetic_splat_bundle.py` also writes a synthetic `edits.json`
    (one delete box, one blend sphere) and `shade_views.json`.
  - Not built: the 3D drag gizmos (E1 ships Inspector fields plus arrow-key nudge
    and `[`/`]` resize instead), click-to-cluster (E4), the SAM-3 lift (E5).
  - Docs: `docs/USER_GUIDE.md` "The Editor", `docs/EDITOR.md` status + §2 + §6.
- **Viewer editor, Python side (`docs/EDITOR.md`, ADR-0007): `edits.json`, weight
  composition, the shade-cloud finder, and `trippy apply-edits`, ahead of the Rust
  viewer UI.** New `trippy/edit/` package: `model.py` (`Region` — box/sphere/lid/
  pointset, `mix`/`op`/`enabled` — and `EditDocument`, an ordered region list plus
  an append-only undo log with a cursor, plus vectorised numpy membership tests
  incl. an oriented box and the lid's falloff/band ramp); `weights.py` (per-point
  blend-weight composition: gate default, then `blend`/`delete`/`fade` regions in
  paint order, with `delete` permanent against a later `blend`/`fade`);
  `shade_finder.py` (the shade-cloud finder, built directly on `trippy.train.
  prune`'s exact audit functions, returning a `pointset` region + a summary whose
  `mass_fraction` matches the audit's own number to float precision); `apply.py`
  (`trippy apply-edits`'s implementation, incl. a single-pass, header-preserving
  Gaussian PLY filter that never loads a ~2 GB PLY twice).
  - `box`'s schema is `center`/`half_extents`/`quat` (an oriented box), a
    deliberate, documented deviation from `docs/EDITOR.md`'s original
    axis-aligned-only `min`/`max`.
  - New CLI: `trippy apply-edits --bundle <dir> [--edits edits.json] --out <dir>`
    (deletes points/Gaussians, writes `blend_weights.npy` + `edits_applied.json` +
    a filtered splat PLY when `bundle.json` names one) and `trippy edits
    shade-find/add-box/add-sphere/add-lid` (author `edits.json` regions from the
    command line; `add-lid` defaults to the Karekare pool's already-fitted
    numbers, `~/Splats/tools/SURFACE_LID.md` Sec 3).
  - 46 new CPU tests (`tests/test_edit_model.py`, `test_edit_weights.py`,
    `test_edit_shade_finder.py`, `test_edit_apply.py`, `test_cli_edits.py`),
    synthetic fixtures only. See `docs/EXPERIMENTS.md` "Edits" for the worked run
    and the full list of what is/isn't implemented vs. `docs/EDITOR.md`'s
    milestones (the Rust viewer render integration and selection-tool UI are not
    part of this change).
- **The blend gate: the splat-vs-TRIPS mix is now an explicit, measurable, adjustable tensor
  (`hybrid.gate`, off by default).** Hybrid design A feeds the Gaussian render into the U-Net as
  *input*, so how much of a finished pixel came from the splat and how much from the TRIPS
  points was buried in the weights -- unmeasurable, unshowable, unadjustable after training. The
  network now grows **one extra output channel** whose sigmoid is a per-pixel weight `g`, and the
  displayed image is `g * splat_rgb + (1 - g) * trips_rgb`. Trained end to end by the ordinary
  image losses; nothing supervises `g` directly. `trippy/hybrid/gate.py`.
  - The blend is applied **after** the tone mapper, because both operands are only commensurable
    in display-referred space (the splat is a finished render of a 3DGS already fitted to these
    photos) -- and because that makes both extremes exact: `gate_scale 0` returns the TRIPS path
    bit for bit and a saturated gate returns the splat bit for bit. Both are asserted as
    equalities in `tests/test_hybrid_gate_math.py` / `tests/test_hybrid_gate_trainer.py`.
  - **A missing splat is never blended.** A frame with no render, a crop `dropout_gaussian_p`
    dropped, a pose with no live renderer: the gate is computed and logged, and not applied.
    Blending its all-zero stand-in would paint black holes and claim splat support that is not
    there.
  - **`hybrid.gate_scale` (0..2, default 1)** re-weights the trained gate at eval/render time --
    0 pure TRIPS, 1 as trained, 2 pushed to the splat. Exposed as `trippy eval --gate-scale` and
    `trippy candidate-report --gate-scale`, recorded in the checkpoint, and a viewer slider.
    Training always blends at 1.0: the scale is a viewing knob, and training through it would
    make the learned gate a function of the knob.
  - **`hybrid.gate_prior` (`target`, `weight`; off by default)** adds
    `weight * (mean(g) - target)^2` to the step loss -- a *mean* prior, so it asks how much of a
    scene wants to be splat without flattening the per-pixel map the feature exists to expose.
  - **Measured everywhere a number already is**: `gate_mean`/`gate_prior` per training step;
    a `gate` block (mean/min/max/percentiles + `gate_scale`) per eval and per candidate report,
    with raw and post-scale summaries per image; a `gate` key in `report.json` and a section in
    the run README. Plus **a from-scratch gate heatmap PNG with every eval**
    (`eval_ep*/gate/<stem>.gate.png`) and per candidate frame (`frames/<pose>/gate.png`) --
    no photographed pixels, so they are in AGENTS.md's "allowed to view" list.
- **Viewer: a Blend panel** (`rust/crates/trips-viewer`, `B` to cycle). TRIPS only / splat only /
  gated blend / manual mix / split screen at the same pose, with a `gate_scale` slider (0..2) and
  a global mix slider (0 = splat, 1 = TRIPS). Composed as Burn tensor ops into the *same* planar
  buffer the network already produced, so `blit.wgsl`, its uniform block and `--screenshot` are
  all unchanged -- which means a screenshot measures the picture on screen. Headless flags
  `--blend-mode`, `--gate-scale`, `--mix`, `--split`.
  - The precomputed `splat.npz` of capture-view renders is now the **fallback**, not the only
    source — see "the splat is rendered live" below. It still behaves exactly as it did: off a
    capture view the panel greys out the splat modes and says so, rather than fading to black or
    reusing the view you just left. See docs/USER_GUIDE.md "The Blend panel".
- **Viewer: the Gaussian splat is rendered LIVE, at any pose** (`trips-viewer/src/splat.rs`).
  The viewer opens `bundle.json`'s `blend.splat_ply` once into Brush's own `Splats` (means,
  quats, log-scales, SH coefficients, raw opacity, device-resident) via `brush-serde`, and calls
  `brush_render::render_splats` at the frame's own camera and resolution every frame. So
  `splat` / `gated` / `mix` / `split` work **everywhere**, not just on the dozen capture views
  the bundle carried renders of, and the gate blend `g*splat + (1-g)*trips` is a real picture of
  this pose on both sides.
  - **Camera conversion**: trippy's COLMAP world-to-camera `(R row-major, t, fx, fy, cx, cy)`
    into Brush's camera-to-world `(position, rotation quat, fov_x, fov_y, center_uv)`. The two
    share a camera frame (`+X` right, `+Y` down, `+Z` forward), so there is **no axis flip** —
    only `rotation = R^T`, `position = -R^T t`, `fov = focal_to_fov(focal, pixels)`,
    `center_uv = (cx/W, cy/H)`. Pinned by a CPU unit test that projects five world points
    through both cameras using each library's own code and requires the same pixel to 1e-2 px.
  - **`blit.wgsl` is unchanged.** `TextureMode::Float` gives a real `[H, W, 4]` f32 image, which
    is sliced/permuted into the same planar `[1, 3, H, W]` buffer the tone mapper already
    produces, so the splat reaches the screen through the existing `MODE_NETWORK` path. (Brush's
    own app uses `TextureMode::Packed` RGBA8, which would have to be unpacked at 8 bits a
    channel before it could be blended at all.)
  - **Composed in display space**, deliberately and documented: the splat's `f_dc_*` colours
    were fitted against display-referred photographs, and the TRIPS frame reaches `compose`
    after the tone mapper, so both operands are already graded the same way and neither gets a
    transfer function. The render uses `background = 0`, so its RGB is premultiplied by coverage
    — the same convention the precomputed `mask_by_alpha` renders use.
  - **Never fatal.** A bundle with no `splat_ply`, a missing file, `--no-live-splat`, or a
    render that errors all fall back to the precomputed path and then to TRIPS, with a line
    saying which. New flags: `--splat-ply`, `--no-live-splat`, `--splat-subsample`,
    `--render-size WxH`, `--splat-bench N`.
  - `brush-render` + `brush-serde` are path dependencies into the submodule (not `brush-dataset`,
    which is the same loader plus `image`/`reqwest`/`async_zip`/`clap`), `cfg`-gated to non-wasm
    so `trips-web`'s graph is unchanged. **The submodule itself is untouched.**
  - **Cost, measured** (`trippy-live-splat-perf-1`, rc 0, M3 Ultra): `kklid_20000.ply`
    (8 910 382 Gaussians, SH degree 3, 2.1 GB) loads in **2.7 s** and renders in **27.1 ms at
    1920x1080** (37 fps, 4 798 874 visible) or 33.4 ms at 1008x756; `--splat-subsample 4` gives
    10.4 ms. Against a ~185 ms U-Net-bound frame that is about **+15%**, and on the synthetic
    4 000-Gaussian fixture the blend is free within noise (TRIPS 185.67 ms vs mix 184.83 ms).
    Full table in `docs/LIMITATIONS.md`.
  - Proof without a window: `scripts/viewer_splat_check.sh` renders a synthetic bundle three
    times at a pose yawed off a capture view — `mix 0` (live splat), `mix 1` (TRIPS) and
    `mix 0 --no-live-splat` (control) — and fails unless the first two differ *and* the control
    is byte-identical to the TRIPS frame. On the synthetic fixture: mean |a-b| **58.1/255, 100%
    of pixels changed**, control **0.000**. `tools/make_synthetic_splat_bundle.py` writes the
    fixture (a generated 3DGS `.ply` + a two-epoch CPU run + `export-bundle`).
- **Export: `bundle.json` records `blend.splat_ply` for any Gaussian-seeded run**, not just a
  hybrid one — every Karekare run is seeded from `kklid_20000.ply` and could not offer the Blend
  panel before. Such a run gets a *minimal* `blend` block: `splat_ply` and nothing that changes
  behaviour (`channels` empty, so the Gaussian block's width `G` is 0 and the network's
  `in_channels` check is unchanged; no gate; no `splat.npz`). A run with no Gaussian PLY still
  gets no `blend` key at all. `trippy.render.bundle.gaussian_ply_path` walks a `union` source to
  find its Gaussian member.
- **Export**: `weights.safetensors` gains `gate`/`gate_channel`/`gate_scale` metadata **only when
  the gate is on**, so a gate-less export stays byte-identical to the file it was (including the
  committed Rust parity fixture). The gate adds **no new tensor** -- it is extra rows of
  `unet.final.weight`/`.bias` -- so `brush_unet::weights`' "no unknown tensor" schema rule is
  untouched. `bundle.json` gains an optional `blend` block and `splat.npz`; `trippy-bundle-1`
  does **not** change version, because both are additive and every previous bundle still loads.
- **Rust `brush-unet` accepts the extra output channel**: `out_channels` of 3 or 4 and nothing
  else, the metadata flag cross-checked against the channel count, the gate refused on any
  channel but 3, and the `camera.response` LUT pinned to 3 rows (the gate is never tone-mapped).
  `Unet::split_gate` and `brush_unet::blend_gate` are the Rust twins of
  `trippy.hybrid.gate.split_output` / `.blend`. Nine new CPU schema tests
  (`tests/gate_schema_cpu.rs`) build the four-channel file byte by byte from the safetensors
  container rules, independently of the Python writer.
- `experiments/EXP-0011-karekare-v2/config_hybrid_gate.yaml`: `config_hybrid.yaml` with the gate
  on and nothing else changed, queued as `kkv2-7-hybrid-gate`.
### Fixed
- **`trippy edits sam --device-audit`: a switch that finds the NEXT device wall
  instead of waiting for it.** A `TorchFunctionMode` that records every
  device-less tensor factory call SAM 3 makes during the forward pass (a
  `torch.zeros(...)` with no `device=` lands on the CPU, and meeting a model
  tensor on MPS is what walls 3, 4 and 5 all were) and prints each site to
  stderr, which survives a crash. Off by default and never for a timing run:
  it sees every torch call and costs real time. Run on the box-prompt path it
  reports five sites, every one of them already followed by an explicit
  `.to(device)` in SAM 3's own code.
- **`trippy edits sam --device mps`: four walls down, all in SAM 3's own code
  and all fixed by rebinding rather than by an MPS fallback
  (`PYTORCH_ENABLE_MPS_FALLBACK` is untouched). Whether MPS becomes the
  viewer's default is decided by jobs `trippy-edit-sam-5` / `-5-cpu`, whose
  rule is in `docs/EDITOR.md` Sec 3; until they land the default stays `cpu`:
  - `sam3.perflib.fused.addmm_act` casts its inputs to bfloat16 unconditionally
    and hands the result to the next fp32 layer. MPS's autocast policy casts a
    convolution's INPUT but not its WEIGHT, so the pass died with `Input type
    (MPSBFloat16Type) and weight type (torch.FloatTensor) should be the same`.
  - `build_sam3_image_model` only calls `.to(device)` for CUDA, leaving the
    weights on the CPU while the image went to the GPU (`MPSFloatType` vs
    `torch.FloatTensor`).
  - the ViT's default rotary embedding uses `torch.view_as_complex`, which MPS
    does not implement; SAM 3 ships the real-valued twin
    (`ViT(use_rope_real=True)`) but the builder never passes the flag, and it
    has no learned parameters, so the checkpoint loads identically.
  - `nn.Module.to()` moves registered parameters and buffers and nothing else,
    and SAM 3 parks precomputed caches in plain attributes: a **dict** in
    `PositionEmbeddingSine.cache` (job `trippy-edit-sam-3`: `_get_img_feats`
    indexed CPU tensors with MPS indices) and a **tuple** in
    `TransformerDecoder.compilable_cord_cache` (job `trippy-edit-sam-4`:
    `decoder.py:380` "found at least two devices, mps:0 and cpu"). Rather than
    wait for a third container shape, `_move_stray_tensors` walks every
    module's `__dict__` and moves every tensor at any depth inside dicts,
    lists, tuples and sets, and `_stray_tensor_devices` re-scans afterwards so
    the child refuses to start -- naming the attribute -- if anything is still
    off-device. Checked against the real model: exactly 10 such tensors exist,
    and all 10 move.
- **The `torch.autocast(bfloat16)` region round SAM 3 inference is gone, and
  the CPU lift got 6x faster: 9.2 s against 58.8 s** for the same mask (129,712
  mask pixels, 74,007 of 104,218 in-mask points selected on
  `exp0010-shade-prune`). bfloat16 on CPU is emulated; taking it out at its
  source instead of chasing it with an autocast is both correct on MPS and
  much faster on CPU. fp32 is strictly more accurate than the bf16 path CUDA
  takes. `docs/EDITOR.md` Sec 3 and `docs/LIMITATIONS.md`'s SAM section, which
  both described the autocast as the fix, are corrected.
- **`rust/crates/trips-web` did not compile.** The Blend panel changed `Renderer::render` and
  `render_to_host` to take a `Blend`, and the web viewer's three call sites were not updated;
  `wasm32` is not on the push path (`scripts/build.sh` checks the native graph only) so nothing
  caught it. They now pass a `WEB_BLEND` constant fixed at `BlendMode::Trips`, which is a hard
  no-op in `compose`, so the browser draws exactly the frame v0.5.0 shipped. Checked with
  `cargo check -p trips-web --target wasm32-unknown-unknown`.
- **A gate-less hybrid bundle wrote `gate = "1"` into `weights.safetensors`.** `export_bundle`
  set the key whenever a `blend` block existed, but the block is written for *any* design-A run,
  gate or not — and `brush_unet::weights` refuses a file whose `gate` key disagrees with
  `out_channels` ("metadata says gate=true but out_channels=3"). The key is now written only
  when the run really has the head.
- **A hybrid (design A) checkpoint's bundle could never be rendered — by the viewer or by
  `trippy bundle-parity`.** A design-A network's `in_channels` is `feature_channels + G`
  (9 on the kkv2 hybrids), but a bundle carried only the `C = 4` point features, so the first
  `forward` failed with "inputs[0] has 4 channels, expected num_input_channels=9". Every hybrid
  bundle exported so far is affected. `bundle.json`'s new `blend` block and `splat.npz` carry the
  Gaussian block itself (rgb + alpha + normalised depth, per capture view), and both the Rust
  renderer and the Python reference now concatenate it onto every pyramid level exactly as
  `GaussianInputs.attach` does in training — honouring `mode: all_levels` vs `concat_level0`, and
  feeding honest zeros (design A's own "no Gaussian information here" state) at a view with no
  stored block. `Renderer::new` now refuses a channel-count mismatch up front, with both numbers,
  instead of failing mid-frame. **Re-export any hybrid bundle to open it.**
  - Caveat, stated because it is a real compromise: the stored block is downscaled to 512 px on
    the long edge, so a hybrid bundle rendered at 1080p upsamples it on the way into the network
    and the frame is *close to*, not identical with, the run's own eval frame. The live splat
    path removes this too.
- **The Burn tone mapper panicked on a four-channel network.** `camera.response` was sized
  `[out_channels, P]`, so a blend-gate bundle reshaped a `[3, P]` LUT to `[1, 4*P]` and aborted at
  load. The LUT is RGB-only — the gate channel is never tone-mapped — and is now read through
  `get_shaped`, so a future disagreement is a message rather than a panic. Found by the GPU
  acceptance job, pinned by three new CPU tests.
### Changed
- `scripts/gpu_submit.sh` accepts priorities **40-59** (trippy manages the queue from
  2026-09-07: 40 target-scene runs, 45 hybrids, 50 other), alongside the existing 10-19 and 70.

## [v0.5.0] - 2026-09-07
### Milestone note
- v0.5.0 = first Karekare candidate that beats the Gaussian baseline on held-out PSNR (17.75 all / 15.59 shade vs 15.53 / 14.94) with shade dark mass down to 24.1% (EXP-0010 arm B: TRIPS point removal + audit-aligned shade prune). Also: person masks, per-scene measured shade frames, viewer v3 exposure control, browser viewer 18 fps. The full-scene (karekare-v2) runs are queued.
### Added
- Person masks (`masks_dir`, auto-discovered) in training loss and eval; EXP-0011 trains TRIPS on the full karekare-v2 scene (756 images, seeded from kklid_20000) masked and unmasked, plus removal and hybrid arms; shade frames measured per scene (93 big-tree frames; kk-coherent's six are a different spot).
- Point removal (TRIPS rule + relative mode) and the audit-aligned shade-prune probe (EXP-0010).
### Changed
- Rasteriser emission vectorised (bit-identical): a trips-mode training step 164 -> 100 ms on MPS (1.11x broadcast). The earlier "10x slower" reading was mostly machine contention. searchsorted segments; no per-step device sync.
### Measured
- **EXP-0003 full2-broadcast re-evaluated with test-time exposure calibration (job `eval-calib-1`, rc 0): the shade verdict flips sign.** Held-out shade **8.49 -> 15.32 dB** calibrated (Gaussian baseline 14.94), all held-out **15.02 -> 17.66**, other **16.47 -> 18.18**; shade SSIM 0.302 -> 0.398, LPIPS 0.689 -> 0.502. The stricter structure-only number -- the best a single closed-form global gain can do, nothing fitted -- is **14.59 dB** in the shade, level with the baseline. All six shade frames converge to the same fitted exposure (gain 0.64-0.77) from starting points 6 EV apart, and the two with *valid* EXIF were wrong too (1.85x where ~0.74x was needed): the U-Net's output scale is tuned to the training frames' exposure and no held-out frame's is ever adjusted to match. Caveats kept attached: a calibrated PSNR uses the held-out photo (quote 14.59 dB if you want the conservative one), and the shade dark-mass fraction is unchanged at 36.9% vs the baseline's 19.9% -- this says the metric was measuring exposure, not that the shade now looks right.
- **The delivered Karekare viewer's white frame was that same 58.5x exposure, and the viewer was rendering it faithfully** (job `viewer-kk-1`, rc 0). `trips-viewer --screenshot` matches trippy's Python reference on the *unfixed* bundle at **85.78 dB** (max 8-bit diff 4; f16 network 59.98 dB), which rules out the response LUT, the f16 network, the background colour and the feature layout in one measurement. View 0 `IMG_3703` mean RGB 0.874/0.867/0.848 -- i.e. `LUT(1)`, a flat clip -- and **6.20 dB against its own photograph**; after the fix 0.456/0.452/0.402 and **14.92 dB**, in line with its neighbour's 15.49. The two views that were never affected are byte-identical before and after. Public horse parity unchanged at **82.68470 dB**. One more datum for the calibration story above: `IMG_3830` is a held-out shade frame whose EXIF was *valid* and which none of this touched, and it still goes **12.30 -> 15.46 dB** when rendered with the scene median instead of its own never-trained EV.

### Fixed
- **A held-out image with no EXIF was rendered through a 58.5x exposure gain for the whole run.** `Trainer._initial_exposure` fell back to an absolute `EV = 0` for images whose cached EXIF has no ExposureTime/ISO, then subtracted the scene mean (5.87 EV on kk-coherent), leaving those frames at a relative -5.87 EV. Held-out frames never get an exposure gradient, so it stuck. 10 of kk-coherent's 219 images have no EXIF and 6 of them are in the held-out split -- 4 of the 6 shade frames plus 2 non-shade frames -- and all six scored 6.19-6.92 dB in EXP-0003 full2-broadcast, against 17.26 dB for the non-shade frames whose EXIF was present. Restated on EXIF-valid frames only, that run's held-out numbers are **all 16.90 / shade 12.37 / other 17.26 dB**, not 15.02 / 8.49 / 16.47: about half of the reported shade gap was this bug, and a real ~4.9 dB shade deficit remains. Missing EXIF now initialises at the scene mean (gain 1.0) and the trainer logs how many images that applies to.
- **The bundle exporter no longer ships an exposure that is an initialisation rather than a colour grade.** `trippy.render.bundle.trusted_exposures` replaces any per-view EV further than `BUNDLE_EXPOSURE_TRUST_STOPS = 2.0` stops from the scene **median** with that median, and records the decision (`exposure_reference_ev`, `exposure_substituted_count`, `exposure_substituted_frames`) in the safetensors `__metadata__`. On full2-broadcast it substitutes exactly the ten EXIF-less views -- the nearest kept view is 1.14 stops out and the nearest replaced one 4.46, so the threshold sits in a wide gap -- and the per-view gain range goes from 0.37x-58.5x to 0.37x-1.85x. Re-exporting changes one tensor; `points.npz` and `bundle.json` are byte-identical. The public horse bundle (EV range +-0.066) is untouched.

### Added
- **Test-time photometric calibration at eval: `trippy eval --checkpoint ... --calibrate [--calibrate-wb]`, config `eval_calibrate_camera` (default off).** Per held-out image, fits only that image's exposure (and optionally its red/blue white balance, green pinned) to its own photo with 200 Adam steps on masked L1, everything else -- points, poses, U-Net, vignette, response LUT -- frozen; the fitted scalar lives in a local tensor and is never written back into the module or the checkpoint (`NeuralCamera.forward_with`). Strict and calibrated numbers are always reported side by side. Precedent: TRIPS's own `optimize_eval_camera` (a per-epoch gradient pass over the *test* crops stepping the camera/pose optimisers, `src/apps/train.cpp:591-596, 693-697`) and `interpolate_eval_settings` (`NeuralCamera.cpp:481-520`), both off by default there too.
- **Per-image exposure diagnostics in every eval, at no extra render cost**: `exposure_ev`/`exposure_gain`, `pred_mean`/`target_mean`/`brightness_ratio`, the closed-form least-squares global gain `gain_best` (`trippy.train.trainer.best_global_gain`) and `psnr_gain`, the PSNR that gain buys. `trippy eval` prints them as a markdown table: a frame whose `psnr_gain` is far above its `psnr` is being scored on its exposure, not its geometry.
- **`forced_heldout_mode: all | alternate`** (`trippy.scene.splits.partition_forced`). `all` (default, unchanged) holds every shade frame out -- novel view of an unobserved region. `alternate` holds out IMG_3828/3830/3832 and forces IMG_3829/3831/3833 into training -- interpolation inside an observed region, which is the protocol the Gaussian baseline is implicitly measured under. New config `experiments/EXP-0003-kk-trips-train/config_full3_alt.yaml` (trips mode, knn sizes, 300 epochs).
- Leaderboard: an optional **"Held-out shade PSNR (calibrated)"** column, present only when some run has a calibrated eval; the Gaussian baseline's cell is permanently `n/a` because raw Gaussians have no per-image exposure model to calibrate.
- **The viewer now chooses which exposure to apply, and says so.** A free-flown pose has no photograph, so it has no exposure of its own; the viewer used to carry the last-pinned view's, which made the scene's brightness depend on which view you last pressed `N` past. `ExposureMode::Auto` (the new default) uses the reference view's own EV while the camera is sitting on it and the **scene median** once you have moved off it. `X` cycles `auto -> view -> median -> manual (EV slider)`, the panel prints the EV and the gain in force, and `trips-viewer --exposure auto|view|median|<EV>` does the same headlessly. Underneath: `brush_unet::NeuralCamera::forward_with_exposure`; plain `forward` still means "this image's own", so nothing else changed behaviour.
- **A bundle no longer opens on a view whose exposure is untrustworthy.** `default_view_with_trusted_exposure` keeps the loader's preferred view when its own exposure is within the trust threshold (the public horse still opens on view 8, where every recorded parity number was measured) and otherwise opens on the trustworthy view nearest the scene median. full2-broadcast moves from view 0 (`IMG_3703`, EV -5.8705) to view 26 (`IMG_3735`, EV +0.25011, exactly the scene median).
- **`trippy bundle-parity`** and `trippy.render.bundle_render`: render a bundle from its own three files with trippy's Python reference and report *numbers only* -- per-channel mean/p01/p50/p99, saturated and crushed-black pixel fractions, and PSNR against a `trips-viewer --screenshot` PNG of the same view. This is what makes "is the viewer wrong, or is the checkpoint?" answerable on a private scene, where no one may open the render. `scripts/viewer_parity_check.sh` drives both halves for a list of views.

### Changed
- **Viewer navigation is 4x faster by default and reaches 50x on the wheel** (Jordan: "I move so slow I can't explore the areas I want"). `BASE_SPEED_FRACTION` 0.5 -> **2.0** median camera gaps per second, and `MAX_SPEED_SCALE` 10 -> **50**, so one scroll ladder (18 notches at 1.25x) crosses a whole capture in under a second. Orbit zoom was already proportional to the pivot distance and now has a test saying so. The HUD and the delivered launcher say `scroll = faster`.

### Documented
- **The Gaussian baseline never held the shade out.** `kkc_15000` was trained with Brush's `--eval-split-every 10`: of the six shade frames only IMG_3829 was held out, so five (including the dolly anchor IMG_3830) were training views, as were 27 of the 33 frames in trippy's held-out split. "TRIPS 8.49 dB vs Gaussians 14.94 dB in the shade" therefore compares a genuine novel-view score against a mostly training-set reconstruction score. Stated in `experiments/EXP-0003-kk-trips-train/README.md` and `docs/EXPERIMENTS.md`; the baseline numbers stand as measured, with the caveat attached.

## [v0.4.0] - 2026-09-06
### Milestone note
- v0.4.0 = the browser viewer is interactive (plan's v0.5.0 acceptance: >= 15 fps at 1080p-class resolution in Chrome now passes: 18 fps network / 76 fps raw at 1440x810), plus the Mac viewer at 29.5 fps and the cross-run leaderboard. The Karekare shade verdict is still pending in the GPU queue.

## [v0.3.0] - 2026-09-06
### Milestone note
- v0.3.0 = the web viewer milestone (plan's v0.5.0), released before the Karekare shade verdict because it landed first; the verdict will be recorded in the release where it lands.
### Fixed
- **The browser viewer was 27x slower than the Mac app because of the *linker*, not the renderer: `raw level-0` 3.32 -> 75.9 fps and the shipped `network` view 1.09 -> 17.7 fps in Chrome, at 1440x810.** `wasm-ld` wraps every export of a wasm module it does not treat as a reactor in a `<name>.command_export` shim that re-runs the whole `.init_array` on entry (the WASI "command" ABI). That is normally free -- Rust has no static constructors -- but `cubecl-ir` pulls in `pliron`, whose dialect and trait-cast registrations are thousands of `inventory::submit` calls, and one `__wasm_call_ctors` run measures **~110 microseconds** here. `wasm-bindgen` resolves `__externref_table_alloc`, `__externref_table_dealloc`, `__wbindgen_malloc` and `__wbindgen_free` **by export name**, so its own shims called the wrappers too: every `JsValue` `wgpu` built while assembling a bind group re-registered the whole of `pliron`. At ~2,500 constructor runs a frame that was **275 ms of a 297 ms frame**. `rust/crates/trips-web/build.rs` now emits `cargo::rustc-link-arg=--export=__wasm_call_ctors` for wasm targets, which makes `wasm-ld` emit none of the wrappers (42 `command_export` functions before, 0 after) and touches no other crate and no native build; `web/trips.js` runs the constructors once itself and **refuses to start if the export is missing**, so the fast build cannot silently regress. One exported no-op (`trips.look(0, 0)`) went from **113 us to 0.065 us**. The GPU-readback PNG's agreement with the native `--half-net --scale 0.75` reference went **62.04 -> 104.54 dB** as a side effect: the old number was an unconverged convolution autotune, not f16 rounding.
- **Safari's "Expected 'f16'" is not about f16, and the viewer now says what it really is.** Safari 26.6.2 grants `shader-f16` and compiles this pipeline's two f16 shaders fine; `1:0: Expected 'f16'` is its WGSL parser reporting that `f16` is the only extension name its `enable` directive accepts, at the exact offset where `web/trips.js` prepends `enable subgroups;`. Proved with a shader-compile-only probe in both browsers: Safari rejects `enable subgroups;`, `enable subgroups_basic;` and even `enable f16, subgroups;` identically, lists no `subgroups` adapter feature, and rejects `@builtin(subgroup_invocation_id)` as an unknown builtin. So all four of `brush-sort`'s radix kernels fail there (the old beacon logged exactly four such errors), the fragments are never ordered and the frame is stripe noise. `web/trips.js` now checks `adapter.features` **before** starting and refuses with the exact kernel names, builtins and feature list instead of drawing something; the check is on capability, not user agent, so a Safari that ships subgroups will simply work. `?anyway=1` still shows the broken output. The `enable subgroups;` injection is also gated on the feature now, since on an adapter without it the directive only replaces an accurate error with a confusing one.
### Added
- `trips-web` exposes the **packed 32-bit sort key** as `?packed=1` and the `P` key, and the HUD always names the sort in use. Off by default, exactly as natively: pairwise on one binary it is worth 79.1 -> 114.6 fps raw and 17.4 -> 19.4 fps network, for **36.85 dB instead of 104.54 dB** against the native reference. It was briefly the web default while kernel launches were the whole frame (it removes 31 of 85); once the linker fix landed it went back to being what it is natively.
- `tests/test_web_build_script.py` asserts both halves of the linker fix -- `build.rs` passing `--export=__wasm_call_ctors` for wasm only, and `web/trips.js` calling `__wasm_call_ctors` and refusing without it.
### Added
- **`brush_pyramid::gpu::UploadedPoints`: upload the point set once per bundle, not once per frame.** `render_pyramid` re-uploaded `xyz`/`size`/`conf`/`feat` on every call -- 80 MB per frame on the horse, for data that never changes. `UploadedPoints::new` + `render_pyramid_uploaded` split the upload out; the `PointSet` entry points still work and simply delegate. Worth a flat **12.2 ms of every frame**: at 1920x1080 on the public horse, `raw level-0` **45.4 -> 102.3 fps** and the shipped `--half-net --scale 0.75` view **21.7 -> 29.5 fps** (job `trippy-web-unet-gpu-3`, 30-frame medians). The `--screenshot` PNG is byte-identical before and after, and the GPU parity tests are unchanged (brush-pyramid 5/5, brush-unet 4/4, job `trippy-web-unet-gpu-2`).
- `StageTimings::upload_ms`, so the upload can never hide inside stage 1 again. **`--profile` should be read in `--mode raw`**: in `network` mode the first stage's forced sync drains the previous frame's queued U-Net, which is what made the old "stage 1 = 178 ms" reading an artefact (the same stage is 0.4 ms).
- `scripts/web_build.sh --profiling`: release codegen with the wasm **name section kept**, which is the only way a wasm panic's stack trace is readable (`--release`'s wasm-opt strips it). This is how the blocker below was found.
### Fixed
- **The U-Net view now renders in a browser** (`trips-web`), and v0.5.0's diagnosis of why it could not was wrong. It was never the `burn::Tensor<4>` -> `CubeTensor` resolve: `resolve_tensor_float` reaches `submit_blocking`, which on wasm32 is an inline reentrant-mutex call, not a thread park. The real trap was **CubeCL's convolution autotuner**, whose roofline bounds generator calls `cubecl_std::throughput::measure_peak_throughput` -- documented upstream as *"Native only, panics on WASM"* -- which ends in `block_on`/`read_sync`. `brush_pyramid::gpu::disable_autotune_roofline_bounds()` sets `AutotuneLevel::Full`, the one level at which `burn-cubecl` registers no bounds generator, and `trips_web::gpu::Gpu::create` calls it before the first CubeCL device exists. No fork, no `[patch]`. Chrome 152 renders the network view of the horse at 1440x810 and its GPU-readback PNG matches the native `--half-net --scale 0.75` reference at **PSNR 62.04 dB**. Cost: the first frame of each new convolution shape autotunes for ~20 s, once per page load, and the page says so while it happens.
- `trips_viewer::renderer::resolve_network_output` is no longer `cfg`-split: the browser's frame goes U-Net -> blit with **no readback and no re-upload**, exactly as the Mac app's does. `networkBlocked` and `?force-network=1` are gone from `trips-web` and `web/trips.js`; `network` is the default mode again in both front ends.
- **Viewer input model** (from Jordan's 2026-09-06 test). Two bugs, both in the viewer only:
  1. *Click-and-drag did nothing.* `app.rs` gated every drag on `Context::egui_wants_pointer_input()`, which is true whenever **any** widget is being interacted with -- including the render canvas itself, allocated with `Sense::click_and_drag()`. Pressing the button therefore disabled the camera for the whole drag. Drags now come from the scene `Response` (`dragged_by` / `drag_delta`), the same thing Brush's own camera controls use. The keyboard gate moved from `egui_wants_keyboard_input()` (true whenever any widget holds focus, so clicking a checkbox killed WASD) to `text_edit_focused()`.
  2. *Fly speed was 1948 u/s in a 15.6-unit scene.* The speed came from the **point cloud** bounding box, which a TRIPS export fills with a far-field environment sphere 12 990 units across. Speed now comes from `bundle::SceneScale` -- the box the **capture cameras** occupy and the median distance between consecutive ones -- at 0.5 x that spacing per second (0.26 u/s on the horse, 0.15 u/s on Karekare), scroll x1.25 a notch within [0.01, 10] x base. The HUD shows the speed as a fraction of the captured area per second as well as in world units.
### Added
- Viewer navigation: **orbit** mode (the default) around a pivot pinned inside the camera box -- you cannot fly out of the scene -- and **free** fly as the alternative, `F` toggles. Right- or middle-drag pans; scroll zooms in orbit and changes speed in free. `R` returns to the view the viewer opened at, `N`/`P` step to the next/previous capture camera, and the "jump to view" dropdown tracks them. Free flight past 3x the camera box shows a "press R to reset" hint.
- `trips-viewer --camera-yaw-deg <d>` and `--free`: a scripted camera change for headless verification.
- The same speed fix reaches the **web** viewer (`trips-web`): `Controller::new` now takes the view list and derives the speed from `SceneScale` itself, and `Renderer::bounds()` — the point-cloud box that caused this bug in both viewers — is no longer public. `trips-web` opens in `Mode::Free`, which is what its `look`/`fly`/`adjust_speed` controls are.
- `scripts/viewer_camera_check.sh`: renders one bundle twice through `--screenshot` with different camera yaws and fails if the two PNGs match, proving without a human that camera changes reach the renderer. Refuses any bundle that is not a known-public scene.

## [v0.2.0] - 2026-09-06
Milestone numbering note: the plan tied v0.2.0 to the Karekare shade verdict and v0.4.0 to the Mac viewer. The viewer landed first (the long Karekare trainings are still queued behind Splats' jobs), so this release carries the viewer and the complete trainer; the shade verdict will be recorded in the release where it lands.
### Added
- Native Mac TRIPS viewer `trips-viewer` (pyramid -> U-Net -> camera -> screen) on wgpu: horse scene at 1920x1080 exact 204 ms; f16 U-Net + 0.75 render scale 45 ms = 22 fps; screenshot parity 82.7 dB vs the reference path. Launcher delivered (ADR-0006).
- Rust pipeline: brush-pyramid (CubeCL, parity 2e-6 vs Python) + brush-unet (Burn, 115 dB end-to-end parity), safetensors export, `trippy export-bundle`.
- Differentiable MPS rasteriser (blend_bwd), trainer with self-reporting runs, candidate report (dolly, honesty sheet, audits), design-B distillation pipeline, union point sources, web toolchain (`scripts/web_build.sh`).
### Findings
- Design C (U-Net refinement of Gaussian renders) does not fix the shade: shade PSNR -1.96 dB.
- First TRIPS training from Gaussian centres (40 ep): 14.4 dB held-out; the point cloud carries more dark mass in the shade volume (36%) than the Gaussian baseline (20%). Long runs queued.
### Added
- **Desktop web TRIPS viewer** (`rust/crates/trips-web` + `web/`, docs/WEB_VIEWER.md): trippy's own forward pass -- `brush-pyramid`'s CubeCL rasteriser and `brush-unet`'s Burn decoder, the same crates the native viewer runs -- compiled to `wasm32-unknown-unknown` and driving **WebGPU** on a `<canvas>`. No framework, no bundler: `wasm-pack --target web` plus one HTML file and one JS file. `scripts/web_build.sh --trips` builds it (2 m 14 s cold, 24.4 MB wasm); `scripts/deliver.sh` ships the double-click `OPEN_TRIPS_WEB_*.command` on 127.0.0.1.
- **The horse renders in Chrome**: 2,218,471 points, 1440x810, dataset view 8, **2.90 fps** in Chrome 152 (raw level-0, measured while a Splats training held the GPU), verified from the page's own `canvas.toBlob()` capture. `?screenshot=1` posts fps, a status beacon and a PNG to a local endpoint, which is how a headless session checks a Safari/Chrome tab it cannot see.
- `trips-viewer` now has a **library target**: bundle loading, the fly camera and the per-frame pipeline are shared verbatim with the web build, and `shaders/blit.wgsl` is one string (`trips_viewer::BLIT_WGSL`) used by both front ends. `eframe`/`egui`/`wgpu`/`rfd` moved to non-wasm target dependencies.
- `PointSet::from_npz_bytes`, `Bundle::parse_manifest`/`Bundle::from_parts`, `brush_pyramid::gpu::upload_f32`: byte-based loading for a browser that has no filesystem.
- The f16 decoder degrades to f32 with a reported reason instead of refusing to open a scene (`Renderer::half_net_error`).
- **Native Mac TRIPS viewer** (`rust/crates/trips-viewer`, ADR-0006): opens a trippy asset bundle from argv or a folder picker and renders it live -- pyramid rasteriser -> U-Net -> tone mapper -> screen -- at the window's size, with WASD/mouse flight, a `V` toggle between network / raw level-0 / coverage views, an on-screen ms+fps readout, and headless `--screenshot` / `--bench` / `--profile` paths for verification. A separate binary from Brush's own `brush`, which is untouched.
- `trippy export-bundle --checkpoint ... --out <dir>`: writes a `trippy-bundle-1` directory (`bundle.json` + `points.npz` + `weights.safetensors`) from either a published TRIPS checkpoint or a trippy-native one, so any scene opens in the viewer with no Rust change.
- `brush_pyramid::scene::Camera` gained an 8-parameter Saiga lens distortion (all-zeros = identity), applied on both the CPU reference and the CubeCL kernel, so bundles can store world-space points instead of one view's pre-distorted ones.
- `Unet::load_with_precision`: the decoder can run in f16. **2.58x on the whole frame (204 -> 79 ms at 1080p) for 59.8 dB against the exact pipeline, i.e. visually free** -- the lever that actually matters, because the network is ~89% of the frame.
- Six measured performance levers, every one defaulting to the exact pipeline: `frustum_cull`, `layer_floor` (fragment cap), `sort` (one packed 32-bit key instead of two radix passes), `feature_store` (f16 features), viewer-side render scaling, and the f16 network above.
- `scripts/open_mac_viewer.sh`: generates the double-click `OPEN_TRIPS_MAC_<name>.command` launcher.
- Native "trips" rasteriser mode (TRIPS's real layer rule), pixel_center/pyramid_halving options; native engine == per-layer parity to 1e-8 dB.
- Candidate report: shade dolly, off-path poses, Splats audit wrappers, honesty sheet.
- Brush fork as submodule (ggjordan/brush trippy-fork) + brush-pyramid/brush-unet crate skeletons (ADR-0005).
- MonoDepthSource (DepthPro via GPU queue), EXP-0004 sheet.
### Known limitations (v0.5.0 web)
- **The U-Net view does not run in a browser.** Both routes from a Burn tensor to a bindable buffer end at CubeCL's `read_sync`, which cannot block on wasm32 -- an unrecoverable trap. `trips-web` substitutes `raw level-0`, says so on screen and in its status, and keeps `?force-network=1` to reproduce it. So there is no browser-side PSNR against the native reference.
- **Safari 26.6.2 draws a wrong image** (stripe noise): one CubeCL shader fails to compile there ("Expected 'f16'"). Use Chrome. Every WebGPU error is now printed on screen in red as "THIS IMAGE IS NOT TRUSTWORTHY".
- Two dependency bugs need JavaScript shims to get any frame at all: wgpu panics on every *clean* WebGPU error-scope pop (`JsOption` vs `JsNullable`), and CubeCL emits `subgroupAdd` without the `enable subgroups;` directive WGSL requires. Both are documented in docs/WEB_VIEWER.md with the exact error strings.
### Changed
- **Corrected a wrong performance conclusion.** The first Mac timing read as "sort-dominated over 10.4M fragments"; the viewer's `raw level-0` view runs the identical rasteriser with the network removed and measures **21.6 ms (46 fps)** against a 204 ms frame, so the rasteriser is ~11% and the U-Net ~89%. Every rasteriser-side lever measures within noise. See `research/trips-metal.md`.
### Fixed
- Trainer: exposure init relative to scene-mean EV, masked MSE normalisation (PSNR was 4.77 dB low), background, crop sampling inside image, seeding, non-finite gradient guard. Smoke run 1.6 -> 12.26 dB.
- se3_exp rotation gradient at phi=0; dataset.crop float64 on MPS; MPS->float64 casts.

## [v0.1.0] - 2026-09-06

## [0.1.0] - 2026-09-06
### Added
- Repo governance (AGENTS.md, scripts, hooks, GPU-queue and delivery wrappers), public MIT repo.
- Python package: two independent geometry implementations, COLMAP bin/txt loader, undistort+cache dataset, splits.
- Point sources: GaussianPlySource (5.7M pts on kk-coherent), ColmapSparseSource, MonoDepthSource (DepthPro via queue), UnionSource, density CLI.
- TRIPS pyramid rasteriser: torch emission/sort + Metal blend_fwd/blend_bwd via torch.mps.compile_shader, no atomics; gradients match float64 to <4e-6.
- Network port (gated U-Net, neural camera, L1+SSIM+LPIPS), checkpoint loader (34/34 tensors match the public horse checkpoint).
- Trainer (crops by K-adjust, schedule, eval, export), `trippy render/train/eval/density/depth-points/parity`.
- EXP-0001 Karekare pyramid sheets, EXP-0002 horse parity: 22.27 dB vs GT (authors 22.34), 36.99 dB vs authors' render. v0.1.0 gate passed.
### Known issues
- Rasteriser needs a native "trips" layer mode (layers 0..ceil(log2 s)); parity used per-layer calls.
- se3_exp rotation gradient at phi=0 (fix in flight).

### Added
- Phase 1 skeleton: agent rules (AGENTS.md), review-gate git hooks, build/test/push/release scripts.
- pyproject.toml with PyTorch, Metal compilation, testing, linting.
- trippy module skeleton: geometry transforms (xform_a, xform_b), agreement test.
- Initial GPU queue smoke test round-trip.
