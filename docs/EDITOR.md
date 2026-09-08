# Editing: turning `trips-viewer` into an edit tool

Status: **Python side implemented, including the publish path and
click-to-cluster** (`trippy/edit/`: `model.py`, `weights.py`,
`shade_finder.py`, `apply.py`, `checkpoint.py`, `cluster.py`; `trippy
apply-edits --target trips|distilled|both`, `trippy edits
shade-find/add-box/add-sphere/add-lid/click`, and `--edits` on `trippy
candidate-report`/`trippy eval`/`trippy distill --stage render` in
`trippy/cli.py`; see `docs/EXPERIMENTS.md` "Edits" for the worked run and
test list).

Status: **E1, E2 and E4 shipped on both sides** (2026-09-07).

- Python: `trippy/edit/` (`model.py`, `weights.py`, `shade_finder.py`,
  `cluster.py`, `apply.py`, and `golden.py`, the parity fixture), `trippy
  apply-edits` and `trippy edits shade-find/add-box/add-sphere/add-lid/click`.
- Rust viewer: `rust/crates/trips-viewer/src/edit/` (the platform-neutral
  twin: `model.rs`, `weights.rs`, `apply.rs`, `shade.rs`, `cluster.rs`) plus
  `src/edit_ui.rs` (the Regions/Inspector/Tools panels, `M`; Shift-click on
  the render selects), the input hook in `src/app.rs` and the render
  integration in `renderer.rs`/`splat.rs`.
- The two sides are pinned together by `tests/fixtures/synthetic/edit_golden/`
  — written by `trippy.edit.golden`, read by BOTH
  `tests/test_edit_golden.py` and the Rust `edit::golden` tests, which agree
  to 1e-6 on the weights and **exactly, id for id**, on the shade and click
  selections. `trips-viewer --dump-weights` / `--dump-shade` / `--dump-click`
  run the same comparison against a real bundle
  (`tests/test_edit_viewer_parity.py`).

Status: **E1, E2, E4 and E5 shipped on both sides** (2026-09-07). E5's viewer
half is `src/edit/sam.rs` (the render-pixel -> view-pixel mapping),
`src/sam_child.rs` (the `trippy edits sam` child process) and the `SAM 3
lift` tool in `src/edit_ui.rs`; see §3's "5. The SAM tool in the viewer".

Status: **brush regions, named regions, and the `trippy eval --edits` gate
gap are done on the Python side** (2026-09-07). `trippy/edit/model.py`
adds `brush`-kind regions (`brush_membership`, `paint_sphere`/
`paint_along`/`erase`, an `.npz` sidecar for a large brush -- §1's `brush`
entry) and `Region.source` + `auto_region_name` (§1's "Named regions");
`shade_finder.py`/`cluster.py`/`sam_lift.py` fill both on every region they
produce; `trippy edits add-brush` and `edits list/rename/toggle/remove`
are the new CLI. `trippy.train.trainer.Trainer.evaluate` now applies the
gate-suppression multiply too, closing §5's own documented gap.

Status: **the viewer's brush tool, its Named Objects panel and the 3D drag
gizmos are shipped** (2026-09-08). This closes the last three viewer pieces
this document was still a spec for:

- **Brush** (`src/edit/brush.rs`, the `brush` tool in `src/edit_ui.rs`, the
  drag in `src/app.rs`, `--brush*` in `src/main.rs`): drag on the render to
  paint spheres of `brush_radius` at the depth of the nearest point under the
  cursor (the same projection trick click-to-cluster uses, inverted --
  `ClickCamera::unproject`), Alt-drag to erase, `[`/`]` for the radius, a
  weight slider, and **one undo entry per stroke** however many frames the
  drag took (`EditDocument::update_region_coalesced` /
  `add_region_coalesced`). Parity with the Python is pinned by
  `tests/fixtures/synthetic/edit_golden/brush.json`, which now records the
  three AUTHORING calls as well as their result: the Rust side replays
  `paint_sphere`/`paint_along`/`erase` and must arrive at the same cells, in
  the same order, with the same weights.
- **Named Objects** (`edit_ui.rs`'s `named_objects_panel`): every region with
  its name, source tool, kind, live point count, enable toggle, mix slider,
  rename, remove and **solo**, grouped by `Region.source.tool`. Regions made
  in the viewer get the same auto names the CLI gives
  (`model::auto_region_name`, pinned by `edit_golden/names.json`).
- **Gizmos** (`src/edit/gizmo.rs`, painted by `app.rs`'s egui painter):
  three projected axis handles on the selected box/sphere/lid; drag to
  translate along that axis, Shift-drag to resize, Ctrl-drag to rotate a box.
  A drag that does not START on a handle still orbits, so navigation loses
  nothing; the keyboard nudges stay, and now share `gizmo::translated` /
  `gizmo::resized` with the drag so the two cannot disagree. A `lid` gets a
  **4th handle** on its own plane normal (2026-09-08, see below).

Status: **three follow-ups from the 2026-09-08 brush/gizmo session closed**
(2026-09-08, `fix/editor-followups`, no queue job -- all local/CPU):

- **The brush depth anchor's `O(points)` scan, measured and fixed.**
  `brush::depth_anchor` cost **47.4 ms/sample on the real `kkv2-1-full-masked`
  bundle** (7.5M points) -- far past a paintable frame budget for a stroke
  that samples several times a second. `edit::brush::ScreenGrid` is a
  screen-space bucket index (cell edge = `ANCHOR_RADIUS_PX`) built ONCE per
  camera pose (`edit_ui.rs::resolve_brush` rebuilds it only when
  `ClickCamera` actually changes) and reused for every sample of a stroke and
  every later stroke from the same viewpoint: **0.011 ms/sample after a
  325.5 ms one-time build**, ~4300x faster per sample. It is an acceleration
  structure, not an approximation -- a cell edge `>= radius_px` gives the same
  3x3-neighbourhood-visits-every-candidate guarantee a uniform spatial hash
  always has, checked against the brute-force scan on randomised inputs
  (`edit::brush::tests::the_grid_agrees_with_the_brute_force_scan_on_random_points_and_pixels`,
  exact equality, not a tolerance). New headless flag `--bench-brush-anchor N`
  reproduces the before/after numbers on any bundle with no GPU device
  touched. See `research/trips-metal.md`'s 2026-09-08 entry for the full
  numbers.
- **The npz sidecar deviation is closed.** §1's `brush` entry used to note
  "the viewer keeps a brush's cells INLINE when it saves ... adding an npz
  WRITER to the viewer would buy nothing a reader can see" -- true for either
  loader (both replay `undo_stack.log`, never the sidecar), but the brief
  asked for byte-for-byte parity with the CLI's own writer regardless, so it
  now exists: `edit::npz_write` (a from-scratch `.npz` writer -- CRC-32,
  ZIP "stored", `.npy` v1.0 headers, no new dependency; `brush_pyramid::npz`
  already had the matching reader), wired into `EditDocument::save` via
  `externalize_brush`, the exact twin of `trippy.edit.model.
  EditDocument._externalize_brush`. A region above
  `EDIT_BRUSH_NPZ_CELL_THRESHOLD` (4096) cells now gets the SAME
  `edits_brush_<id>.npz` (`cells` int32, `weights` float32 when graded) from
  either language, pinned by the SAME synthetic recipe on both sides
  (`tests/test_edit_model.py::test_brush_npz_sidecar_written_above_threshold`,
  `edit::model::tests::a_brush_above_the_npz_threshold_is_externalised_on_save`)
  and cross-checked with plain `numpy.load` against a Rust-written file via a
  new `--brush-npz-selftest <dir>` flag
  (`tests/test_edit_viewer_parity.py::test_brush_npz_sidecar_the_viewer_writes_loads_with_plain_numpy`).
- **The lid's plane normal now has its own gizmo handle.** A 4th handle,
  `edit::gizmo::NormalHandle`, projected along `up` the same way the three
  world-axis arms are projected along X/Y/Z; dragging it tilts `up` by
  decomposing the screen drag onto two in-plane basis directions captured at
  projection time (`orthonormal_basis`, no further camera calls needed once
  the drag starts, same contract every other handle keeps), then
  re-normalising. `Drag::Tilt` is the gesture (Shift/Ctrl are ignored on this
  ONE handle -- there is no axis to resize or rotate about), coalesced into
  one undo entry like every other drag. The Inspector's typed `up`
  (`vec3_row`, unchanged) and the handle share nothing but the same
  `Region.params.up` field, so either can follow the other. Tests:
  `edit::gizmo::tests::{a_lid_gets_a_fourth_handle_for_its_normal,
  picking_the_normal_handle_returns_normal_axis,
  dragging_the_normal_handle_tilts_up_and_stays_unit_length}` (maths) and
  `edit_ui::tests::dragging_the_lids_normal_handle_tilts_up_as_one_undo_step_and_typing_still_works`
  (a drag, then a typed edit, then two undos, session-level).

Four implementation notes, all explained where they matter below:

- **`box`'s schema is `center`/`half_extents`/`quat`** (an oriented box),
  not this section's axis-aligned `min`/`max` -- a rotated box is testable
  where an axis-aligned-only schema is not; identity `quat` recovers an
  axis-aligned box exactly. See §1's `box` entry.
- **The Python `lid` region kind is E3's hard-clip action already**
  (`op="delete"`/`"fade"`, `trippy.edit.model.lid_membership`). Its 3D gizmo
  is E1's general one (`src/edit/gizmo.rs`, shipped 2026-09-08): a lid is
  moved by dragging a handle and its `radius` scaled by Shift-dragging one,
  rather than by a ring of its own, and (2026-09-08) its plane normal `up`
  has its own 4th handle to drag-tilt. `height` is still typed in the
  Inspector, and `up` can be either dragged or typed. See §1's `lid` entry
  and §6's E3 row.
- **The checkpoint-side gate-suppression multiply (§5) is a documented
  simplification, not the full per-pixel override §2 describes** -- it
  multiplies the trained blend gate by the editor's own per-point weight
  (1 = unedited, ramping down inside a `blend`/`fade` region), which only
  ever pulls the gate towards TRIPS, never towards splat. See §5's own
  paragraph for why (there is no live-editable per-point splat channel on
  a checkpoint-only path) and `trippy.edit.checkpoint`'s module docstring
  for the exact reasoning.
- **`blend`/`fade` do NOT scale a point's alpha, only `delete` does.** §2
  originally read as "the weight multiplies the point's alpha"; taken
  literally that makes a `mix = 0` region invisible in the TRIPS pass, so
  the per-pixel blend has no coverage there to mix the splat *into* and the
  two mechanisms cancel. The shipped rule is: `delete` removes rows before
  rasterisation (a weight of 0 that really means "not rendered"), and
  `blend`/`fade` leave alpha alone so the probe pass's alpha-weighted
  average is an average of the *untouched* alphas. See §2 "The probe pass"
  and `rust/crates/trips-viewer/src/edit/apply.rs`.
- **Click-to-cluster's Python side (§3's "1.") projects with
  `trippy.geom.xform_a` and applies NO lens distortion** -- that module has
  no distortion model at all (Saiga's 8-coefficient form lives in
  `trippy.render.parity`, torch-only; `trippy.geom.camera`'s OpenCV form is
  a third, different convention again). Exact on a trippy-native bundle
  (whose views carry all-zero distortion by construction,
  `trippy.render.bundle.native_views`); a first-order approximation of
  "which points are near this click" on a TRIPS/ADOP bundle. The default
  `max_radius` (how far a grown region may reach from its seed) is the
  bundle's own median NEAREST-CAMERA spacing, not point-cloud density --
  `trippy.edit.cluster.default_max_radius_from_bundle`, reusing
  `trippy.points.knn_size.median_nn_distance`'s exact cKDTree query pattern
  against camera centres instead of `xyz`. See `trippy.edit.cluster`'s
  module docstring for the full algorithm and every constant's reasoning.

E5's SAM-3 lift is complete on both sides: `trippy edits sam` /
`trippy/edit/{sam_lift,sam_runner}.py` on the Python side (§3's "4. SAM 3
lift (E5)" has the SAM 3 API and the command) and the viewer's own SAM tool
(§3's "5. The SAM tool in the viewer (E5, viewer half)"). §6's E5 row lists
what is still open.

`docs/decisions/ADR-0007-viewer-editing.md` is still accurate to the
sections below; this document is the detailed spec the milestones in §6
implement, in order.

**Scope narrowed 2026-09-09 — see `docs/decisions/ADR-0008-supersplat.md`:**
plain-Gaussian editing UX (lasso/polygon/flood/eyedropper selection,
select-by-value histograms, colour grading, geometry transform gizmos, orient
and measure tools, compressed export formats, splat-only camera timelines and
web/VR viewing) is **dropped in favour of a self-hosted SuperSplat Editor 3.0**
on 127.0.0.1. This document keeps everything SuperSplat cannot do: the TRIPS
render, per-region splat-vs-TRIPS mix, TRIPS-guided fog deletion, the
shade-cloud finder, SAM 3 lift, click-to-cluster and the honesty views.

Jordan's request (2026-09-07 09:45, `STATE.md`): a per-region/per-object
splat-vs-TRIPS mix, smart object selection with masks, the Karekare pool
"lid" as an edit, a tool that finds and lets him deal with floating shade
clouds, undo, save/load with the bundle, and a way to get edits into a
published, distilled splat. Everything here builds on the viewer described
in `docs/decisions/ADR-0006-viewer-integration.md` and read directly from
`rust/crates/trips-viewer/src/{app.rs,renderer.rs,bundle.rs,camera.rs}`.

## 0. What the viewer already gives this for free

- **A loaded, resident point set.** `Renderer` holds `points: PointSet`
  (`xyz`/`size`/`feat`/`conf`, world space, `brush_pyramid::scene`) and a
  cached device upload (`uploaded: RefCell<UploadedPoints>`) that is only
  rebuilt when a precision lever changes (`Renderer::resident_points`). Any
  edit tool that needs "every point's world position" already has it, on the
  CPU (`self.points`) and the GPU (`self.uploaded`), with no new upload path.
- **Three honesty views sharing one render** (`renderer.rs::ViewMode`):
  `Network`, `RawLevel0`, `Coverage`. The shade-cloud finder and the region
  Inspector both want a "preview highlight" mode; this is the natural fourth
  view, added the same way the other three are (one more `match` arm in
  `Renderer::render`, one more `shader_code`).
- **A world-space camera with a known basis** (`camera.rs::Controller`,
  `docs/GEOMETRY.md`: `+Z` forward, `+X` right, `+Y` down, row-major `R`).
  Click-to-cluster was expected to need a ray from a clicked pixel; **as
  shipped it needs no ray and no new camera code at all**. Projecting the whole
  cloud through the camera `render_camera` already returns and keeping what
  lands near the click answers the same question ("which points did I click"),
  costs one pass over an array the renderer already holds, and is the SAME
  arithmetic `trippy/edit/cluster.py` runs — which is what makes the two sides
  comparable id for id. `edit::cluster::ClickCamera::from_render_camera`
  widens that camera's `f32` to the `f64` the projection runs in; `camera.rs`
  was not touched.
- **A live Gaussian splat at every pose** (added 2026-09-07, after this document
  was first written). `Renderer` now holds an optional `crate::splat::LiveSplat`:
  `bundle.json`'s `blend.splat_ply` loaded once into `brush_render::Splats`
  (device-resident `transforms [N,10]`, `sh_coeffs [N,C,3]`, `raw_opacities [N]`)
  and rasterised by `brush_render::render_splats` at the frame's own camera,
  composed with the TRIPS frame in `Renderer::compose`. Two consequences for this
  design:
  - §2's splat-side trick — "the region weight rides as a Gaussian's own scalar
    attribute, rendered as a second forward pass with the scalar in place of the
    SH DC term" — now has a renderer in-process to do it with. It is a second
    `render_splats` call on a `Splats` whose `sh_coeffs` DC row has been replaced
    by `w_edit`, on the same device, with no new kernel. That was hypothetical
    when §2 was written; it is now a dozen lines.
  - The per-point region test (§1, §2) has to be evaluated against **two** point
    clouds, not one: `points.npz`'s `xyz` for the TRIPS half and
    `Splats::means()` for the Gaussian half. They are different clouds with
    different indices — which is exactly the `pointset`-vs-`box/sphere/lid`
    distinction ADR-0007 §4 already draws, arriving one stage earlier than
    expected: a `pointset` region selected on the TRIPS cloud cannot be replayed
    against the *live splat* either, not just against a distilled one.
- **A camera conversion into Brush's frame**, `crate::splat::to_brush_camera`,
  unit-tested to project a world point to the same pixel in both renderers.
  Anything that needs to ask "where does this world point land in the splat
  render" now has it.
- **A sidecar-friendly bundle.** `Bundle::load` reads exactly the two files
  `bundle.json` names (`src/bundle.rs`'s own doc comment: "only the two files
  it actually names are touched"). Nothing in the loader enumerates the
  bundle directory, so `edits.json` living beside `bundle.json` is invisible
  to every existing code path until something goes looking for it.

## 1. Data model: `edits.json`

Written next to `bundle.json` in the same bundle directory. Absent = no
edits, identical in spirit to a pre-lever bundle loading as the exact
pipeline (`PyramidParams`'s serde defaults, `docs/decisions/
ADR-0006-viewer-integration.md` "Performance levers are render parameters").

```jsonc
{
  "format": "trippy-edits-1",
  "bundle_format": "trippy-bundle-1",   // cross-check against bundle.json
  "regions": [
    {
      "id": "r-3f9a",                    // stable, never reused (undo/redo keys off it)
      "name": "pool lid",
      "kind": "lid",
      "enabled": true,
      "mix": 1.0,                        // 0 = pure splat, 1 = pure TRIPS; ignored by op:delete
      "op": "delete",                    // "blend" | "delete" | "fade"
      "params": {
        "up": [0.01290213, -0.95271846, -0.30358041],
        "height": -0.49,
        "center": [-0.00683348, -0.74759695, 3.95994345],
        "radius": 2.5,
        "falloff": 1.0,
        "band": 0.05
      }
    },
    {
      "id": "r-8b21",
      "name": "shade cloud, big tree",
      "kind": "pointset",
      "enabled": true,
      "mix": 0.0,
      "op": "fade",
      "params": { "point_ids": [1881, 4410, "... run-length or plain array"] }
    }
  ],
  "order": ["r-3f9a", "r-8b21"],          // paint order; later entries win on overlap
  "undo_stack": {
    "cursor": 4,                          // index into "log"; undo/redo just move it
    "log": [ /* opaque, append-only list of the same region-shaped diffs the UI applies */ ]
  }
}
```

**`Region`** fields, matching the brief exactly:

| field | type | meaning |
|---|---|---|
| `id` | string | stable identity; never reused, even after delete (undo needs it) |
| `name` | string | shown in the Regions panel |
| `kind` | `box \| sphere \| brush \| lid \| pointset` | which `params` shape applies |
| `mix` | float `[0,1]` | `0` = pure splat, `1` = pure TRIPS, per the brief's convention |
| `op` | `blend \| delete \| fade` | see below |
| `enabled` | bool | soft "off" without deleting the region |
| `source` | `{"tool": str, ...}` or `null` | which tool made this region and with what prompt/parameters (`trippy.edit.model.Region.source`, "Named regions" below); `null` for a hand-authored region |

**Named regions.** A tool-authored region (the shade-cloud finder,
click-to-cluster, the SAM-3 lift, the brush) that receives no explicit
`name` from its caller is auto-named `"<tool>[-<detail>]-<n>"`
(`trippy.edit.model.auto_region_name`): `click-1`, `sam-box-IMG_3703-2`,
`shade-clouds-3`, `brush-4`. `n` is one more than the highest `-<digits>`
suffix among the document's OWN region names (from ANY tool, not just this
one), so a session's tool-authored regions read as one continuously
numbered list in a future Named Objects panel regardless of which tool
made each one — and the scheme is stateless (no counter stored anywhere),
so it self-heals after a region is renamed, removed, or the edit undone.
`source` is the companion field: it never affects membership or
composition, only display/provenance, and is filled by
`trippy.edit.shade_finder`/`trippy.edit.cluster`/`trippy.edit.sam_lift`
alongside the name. `trippy edits list/rename/toggle/remove --edits
edits.json` is the CLI a Named Objects panel can already script against
(`list` prints every region's `id`/`name`/`kind`/`op`/`mix`/`enabled`/
`source`; `rename`/`toggle`/`remove` mutate one by `id`, through the same
undo-logged `EditDocument` mutation methods the viewer will eventually
call directly).

`kind`-specific `params`:

- `box`: **implemented as `center: [x,y,z]`, `half_extents: [x,y,z]`
  (all > 0), `quat: [w,x,y,z]` (default identity)** — an ORIENTED box, not
  the axis-aligned `min`/`max` this line originally specified. Identity
  `quat` is exactly an axis-aligned box (`min = center - half_extents`,
  `max = center + half_extents`); a non-identity `quat` is what a rotated
  box needs and `min`/`max` cannot express (`trippy.edit.model.
  box_membership`, `docs/EXPERIMENTS.md` "Edits").
- `sphere`: `center: [x,y,z]`, `radius: f32`. Implemented,
  `trippy.edit.model.sphere_membership`.
- `brush`: a sparse voxel grid — `origin: [x,y,z]`, `cell_size: f32`,
  `cells: [[i,j,k], ...]` (occupied cell indices; a set, not a dense array,
  because a brushed region is typically a tiny fraction of the scene's
  bounding volume), and an optional `weights: [f32, ...]` parallel to
  `cells` (one weight per occupied cell, default `1.0` for every cell when
  omitted — a plain, ungraded brush is the common case). **Python side
  implemented** (`trippy.edit.model.brush_membership`: a point's own voxel,
  `floor((p - origin) / cell_size)`, looked up in the occupied set;
  `region_weight`/`region_contains` dispatch to it exactly like every other
  kind, so `trippy.edit.weights`/`apply`/`checkpoint` needed no
  brush-specific code at all — "it is just another membership"). Painted by
  `paint_sphere(region, center, radius, weight=1.0)` and
  `paint_along(region, points, radius, weight=1.0)` (a stroke: the union of
  `paint_sphere` at every point of a path) — both a box-sphere intersection
  test against every voxel in the sphere's bounding box, so a stroke paints
  every cell the sphere actually OVERLAPS, not merely voxels whose centre
  falls inside it; repeated painting only ever strengthens a cell
  (`max(existing, weight)`). `erase(region, center, radius)` is the inverse,
  dropping every touched cell outright. All three are pure functions
  (return a NEW `Region`, never mutate their input), matching every other
  membership test in `trippy.edit.model`. `EditDocument.save` externalises
  a brush region's `cells`/`weights` into an `.npz` sidecar
  (`edits_brush_<id>.npz`) once it exceeds
  `EDIT_BRUSH_NPZ_CELL_THRESHOLD` cells — in the WRITTEN copy of
  `regions[]` only; Python's own loader never reads the sidecar back (it
  always reconstructs a region by replaying `undo_stack.log`, which is
  never externalised), so this only matters to an external reader (the Rust
  viewer) loading the materialised region list directly. CLI:
  `trippy edits add-brush` (one region + one sphere stroke).
  `tests/fixtures/synthetic/edit_golden/brush.json`
  (`trippy.edit.golden.build_brush_fixture`) exercises `paint_sphere`,
  `paint_along` and `erase` in sequence and records **both** the three
  authoring calls (`"strokes"`) and the region + per-point weights they
  produce, which is what the Rust twin
  (`rust/crates/trips-viewer/src/edit/brush.rs`) replays: it must arrive at
  the same cells, in the same order, with the same weights, and then agree
  on every query point's membership. Recording the calls and not only their
  result is deliberate — a port that voxelised a sphere by "is the cell's
  CENTRE inside it" would reproduce a smaller cell set that still passed
  every membership lookup. **Shipped on the viewer side too** (2026-09-08):
  the `brush` tool paints on the render (§4), and one stroke is one undo
  entry. **The npz sidecar writer is also shipped on the viewer side**
  (2026-09-08, closing what used to be the one deviation from the Python
  here): `EditDocument::save` externalises a brush region's `cells`/
  `weights` into `edits_brush_<id>.npz` above `EDIT_BRUSH_NPZ_CELL_THRESHOLD`
  cells, the exact array shapes/dtypes `trippy.edit.model.
  EditDocument._externalize_brush` writes (`edit::npz_write`, a from-scratch
  writer — no ZIP crate, matching `brush_pyramid::npz`'s own from-scratch
  reader). Neither loader ever reads that sidecar back (both reconstruct
  every region by replaying `undo_stack.log`, which is never externalised),
  so this still buys nothing either loader can see — the parity is for an
  EXTERNAL reader, and for keeping the written `edits.json` small, both of
  which the brief asked for regardless. Pinned by the same synthetic
  over-threshold stroke on both sides (`tests/test_edit_model.py`,
  `edit::model.rs`'s own unit test) and cross-checked with plain
  `numpy.load` against a Rust-written sidecar
  (`tests/test_edit_viewer_parity.py`, via `--brush-npz-selftest`).
- `lid`: `up`, `height`, `center`, `radius`, `falloff`, `band` — **the same
  six numbers as `~/Splats/tools/SURFACE_LID.md`'s `--lid-*` flags**, so the
  Karekare pool's already-fitted values
  (`SURFACE_LID.md` §3: `up = (0.01290213, -0.95271846, -0.30358041)`,
  `height = -0.49`, `center = (-0.00683348, -0.74759695, 3.95994345)`,
  `radius = 2.5`, `falloff = 1.0`, `band = 0.05`) drop straight into a
  region with zero re-fitting. This is a **hard clip**, not the training
  regulariser — see §2's "the lid is not `surface.rs`'s lid" box.
  **Implemented** (`trippy.edit.model.lid_membership`; `op="delete"`'s hard
  membership is the literal "below the plane, inside the radius", ignoring
  `falloff`/`band`; `op="blend"`/`"fade"`'s graded weight ramps by them —
  see that function's docstring for the formula, mirrored across the plane
  from `SURFACE_LID.md`'s own `band_i`/`region_i` shape). `trippy edits
  add-lid` with no geometry flags seeds exactly the numbers above. The 3D
  gizmo is E1's general one (drag a handle to move the lid, Shift-drag to
  scale its radius; §6's E3 row).
- `pointset`: `point_ids: [u32, ...]` — indices into `points.npz`'s `xyz`
  array (the TRIPS point cloud's own row order, stable for the life of a
  bundle since nothing in the viewer re-sorts or re-indexes points after
  `Bundle::load`). This is what the shade-cloud finder, click-to-cluster and
  the SAM-3 lift all produce. **Implemented for the shade-cloud finder**
  (`trippy.edit.shade_finder.find_shade_pointset`, CLI: `trippy edits
  shade-find`) **and for click-to-cluster** (`trippy.edit.cluster.
  click_to_cluster`, CLI: `trippy edits click`); the SAM-3 lift is not built.

**`op` semantics:**

- `blend`: the region's per-pixel weight is `mix` (constant across the
  region, or ramped by `falloff`/`band` at its edge — see §2). Composites
  with the gate exactly as described there.
- `delete`: hard removal. For `box`/`sphere`/`lid`, a point/Gaussian inside
  the region is excluded from rasterisation entirely (not merely given
  weight 0 — see §2 on why this has to happen upstream of blending, not as
  a `mix=0` special case, for the lid's "nothing renders" requirement).
  For `pointset`, the listed point IDs are excluded.
- `fade`: like `blend`, but the weight ramps from the region's current
  visibility down to fully hidden over a UI-exposed duration/distance rather
  than jumping — the soft version of `delete`, useful for the shade-cloud
  finder where an instant pop is more jarring than a fade (this is a
  presentation detail on top of the same masking machinery, not a new
  render path).

**Ordering and overlap.** `order` is authoritative paint order; where two
enabled regions both claim a point/pixel, the later region in `order` wins
outright (last-write-wins, not averaged) — the simplest rule that matches
how a UI's layer list usually reads ("this region on top of that one"), and
the only one that composes predictably with `delete` (a `delete` region
must always be able to override an earlier `blend`, never the reverse, or a
deleted object could be un-deleted by scrolling a slider on an unrelated
region).

**Undo.** `undo_stack.log` is an append-only list of the same diff shape the
UI already applies (add region / remove region / edit one field), so undo
and redo are pure cursor moves — no inverse-operation bookkeping, no risk of
an undo and its forward edit drifting apart. Every UI action pushes onto
`log` at `cursor`, truncating anything after it (the standard "new edit after
undo discards the redone future" rule). `save` writes `edits.json` verbatim,
undo stack included, so **closing and reopening the bundle preserves undo
history** — a deliberate choice: Jordan's sessions are long and interrupted
by GPU queue waits, and losing undo on reopen would make "try it, back out if
wrong" more expensive than it needs to be.

## 2. Render integration

### The weight, precisely

Per point (TRIPS) or per Gaussian (splat), not per pixel — see the "why not
depth" box below — each renderer computes a scalar `w_edit` in `[0, 1]`
before compositing:

```
w_edit(point) = mix of the LAST enabled region (in `order`) that contains it,
                or "no override" if no enabled region contains it
```

untouched by any region, `w_edit` is `None` and the pixel is whatever
`feat/blend-gate`'s own `g * gate_scale` already decided. Where a region
does apply, `w_edit` **overrides** the gate at that point — this is the
"Jordan's hand-set exception to the network's opinion" semantic from
`docs/decisions/ADR-0007-viewer-editing.md` §"The gate and region edits must
compose, not race". `delete`-op regions do not produce a weight at all; they
remove the point from the input entirely (next section).

### Why per-point, not per-pixel depth lookup

Both renderers were checked (`docs/decisions/ADR-0007-viewer-editing.md`
"Consequences") and **neither exposes a composited per-pixel depth buffer
today** — re-confirmed 2026-09-07 while wiring the live splat, which calls
`render_splats` for real: it returns `(Tensor<3> [H, W, 4], RenderAux)`, the
fourth channel of the image is **coverage** (`1 - T`), and `RenderAux` carries
`num_visible` / `num_intersections` / `visible` / `max_radius` (per *splat*) /
`tile_offsets` and nothing composited per pixel: `brush_pyramid::output::LayerImage` has `feature`/`t_final`/
`n_used`, no depth; `brush-render`'s `RenderOutput`/`RenderAuxInner`
(`rust/brush-trips/crates/brush-render/src/render_aux.rs`) has `out_img` and
per-*splat* (not per-pixel) `max_radius`, nothing composited per pixel
either. Building a real depth-lookup path would mean adding a depth output to
two independent rasterisers and keeping their conventions in agreement
(itself flagged as a risk in §7) — expensive, and unnecessary, because the
question a region actually needs answered is "is this point/Gaussian inside
the region", which is answerable in world space with nothing rasterised yet.

The per-point scalar then needs to become a *per-pixel* effect, which is
where each renderer's own existing alpha-compositing does the work for
free, using the same trick `crates/brush-train/src/surface.rs`'s depth-
distortion term already validated: substitute the scalar for the colour (or
concatenate it as one more feature channel) and let the ordinary
front-to-back blend produce a correctly-weighted composite, with **no new
kernels, no new backward pass, no depth buffer**:

- **TRIPS side** (**implemented**, the "sibling buffer" option): a second
  `PointSet` over the same surviving rows, with the same `xyz`/`size`/`conf`
  and a **three-channel** feature vector `[w_edit, touched, 1]`, rasterised
  by a second `render_pyramid_uploaded` call at the same camera with the
  same `PyramidParams` and an all-zero background. Because the two clouds
  share `conf`, the two passes place identical fragments with identical
  alphas, so at level 0 the three channels are `sum(T a w_edit)`,
  `sum(T a touched)` and `sum(T a) = 1 - t_final`. Dividing the first two by
  the third is exactly the alpha-weighted average this paragraph asks for.
  Three channels and not four or five because
  `brush_pyramid::params::SUPPORTED_CHANNELS` is `[3, 4, 8]`: `C = 3` compiles
  the rasteriser's **existing** pipeline, so this needed no kernel change, no
  new fixture, no widening of the U-Net's input and no risk to the parity
  tests. See `rust/crates/trips-viewer/src/edit/apply.rs` and
  `Renderer::edit_pixels`.
  - `touched` is the one quantity with no Python twin: it is "did any enabled
    `blend`/`fade` region claim this point", and it is what makes a region an
    *override* of the gate only where it applies. Where nothing is touched it
    is 0, the division gives 0, and `w_pix` falls back to 1 — a bit-exact
    no-op, which is why an unedited bundle renders the frame it always did.
  - The cost is one extra pyramid pass, paid **only** while an enabled
    `blend`/`fade` region exists. A `delete`-only session, and every unedited
    session, pays nothing.
  - Known limit: the edit is applied to the TRIPS operand *before* the Blend
    panel's own mode, so a region can always pull a pixel towards the splat
    but a `mix = 1` region does not force pure TRIPS over a gate that wanted
    the splat. The direction that matters (Jordan hiding TRIPS content)
    works; the reverse is a corner case, recorded rather than hidden.
- **Splat side**: `delete` is implemented and `blend`/`fade` deliberately are
  not. A `delete`-op region scales the matching Gaussians' opacity to
  effectively zero before `brush_render::render_splats` sees them
  (`LiveSplat::set_opacity_scale`: `raw' = logit(clamp(scale * sigmoid(raw)))`,
  because `raw_opacity` is a logit and `logit(0)` is not finite; the ply's own
  raw values are kept so an undo restores them exactly). `blend`/`fade` do NOT
  touch a Gaussian's opacity: their mix is the per-pixel choice above, and
  dimming the Gaussians as well would apply it twice and in the wrong
  direction — a `mix = 0` region asks for MORE splat, not less. This is
  exactly what `trippy.edit.apply.apply_edits` does on the publish side, where
  the PLY is filtered by `compose_gaussian_weights(...).delete_mask` and
  nothing else. The "scalar in place of the SH DC term" second pass is
  therefore still unbuilt, and still the route if a per-pixel splat-side
  weight is ever wanted.

Region membership itself is evaluated **once per edit change**, not once
per frame: `O(points × regions)`, cached until `edits.json` changes, exactly
the same "recompute only when the input actually changed" discipline
`Renderer::resident_points` already uses for the precision-lever re-upload.
This is cheaper than the brief's own suggested SDF/voxel texture sampled per
pixel per frame, and it never needs the depth buffer that does not exist.

### Deletions

`delete`-op regions do not produce a blend weight; they remove
points/Gaussians from the renderer's input *before* rasterisation:

- **TRIPS points**: a boolean keep-mask over `points.npz`'s rows,
  `index_select`ed out of `PointSet` before `UploadedPoints::new` — the same
  operation `trippy.train.trainer.Trainer._apply_keep_mask` already performs
  for training-time point removal (`docs/ARCHITECTURE.md` "train/"), just
  applied at load/edit time on the CPU-resident array in Rust instead of at
  epoch boundaries in Python. Recomputed (and the device buffer
  re-uploaded) only when the delete-region set changes.
- **Splat Gaussians**: a boolean keep-mask over the Gaussian PLY's rows,
  applied wherever the splat is loaded for rendering (the live splat-vs-
  TRIPS preview, and the publish path in §5).

This upstream-removal design is what makes the **lid's** "nothing renders
below the plane inside the radius" literal: a `delete`-op `lid` region drops
the affected points/Gaussians from the renderer entirely, so there is no
pixel where they could appear at any `mix`, at any exposure, from any angle
— stronger than a `mix=0` blend, which would still be "compositing zero
weight of something" rather than "that something was never there".

> **The lid here is not `crates/brush-train/src/surface.rs`'s lid.** That is
> a *training-time* opacity penalty (`SURFACE_LID.md`, read-only) that fades
> haze *during* optimisation, with a detached-means, opacity-only gradient
> designed never to fight the geometry it protects. The editor's `lid`
> region kind reuses the same plane/radius/falloff geometry and the same
> already-fitted, A/B-verified numbers, but the *action* is a hard
> view/publish-time clip with no gradient at all. Do not wire the editor's
> lid into `surface.rs`'s loss; they solve adjacent but different problems
> at different lifecycle stages.

## 3. Selection tools, in delivery order

### 1. Click-to-cluster (E4)

Click a pixel → cast a ray from `camera.rs::Controller`'s basis through that
pixel → find the nearest point(s) in `PointSet::xyz` along the ray (a k-d
tree over the already-resident CPU array, built once per bundle load and
rebuilt only if `points.npz` itself changes, which it does not during a
session) → grow a `pointset` region by k-NN in `(world position, colour)`
space from the clicked point, with a size/tightness slider in the Inspector.
No rendered depth buffer needed (§2's "why not depth lookup" applies here
too) — the ray/point-cloud nearest-neighbour test is a pure CPU geometry
query against data the renderer already holds.

**Python side implemented** (`trippy.edit.cluster`, CLI: `trippy edits
click --bundle DIR --view IMG_xxxx.jpg --px U V [--radius-px 12]
[--colour-tol 0.15] [--max-radius R] [--max-points 200000] --op
fade|delete|blend --mix M --out edits.json [--preview heatmap.png]`),
ahead of the Rust viewer's ray cast: every point is projected into a named
*registered* view (`trippy.geom.xform_a`, no distortion — see the status
header's implementation note) rather than cast along an arbitrary free-fly
camera ray, since the Python side has no live `camera.rs::Controller` to
ask. The points within `radius_px` of the click are clustered by camera-
space depth; the mode nearest the camera seeds the region (this is what
keeps a click from selecting a same-coloured surface glimpsed through a
gap behind the clicked object — see the module docstring's step 3). Growth
from that seed is k-NN in 3D (`scipy.spatial.cKDTree`, the same query
pattern `trippy.points.knn_size` uses) gated by colour distance
(`feat[:, :3]`, `trippy.edit.shade_finder`'s own base-colour slice) and a
hard `max_radius` (world units, defaulting to the bundle's median nearest-
CAMERA spacing) and `max_points` cap. `--preview` writes a from-scratch
point-density heatmap PNG of the selection (no photo content).

**Rust side implemented too** (2026-09-07,
`rust/crates/trips-viewer/src/edit/cluster.rs` + the Selection panel in
`src/edit_ui.rs` + the Shift-click hook in `src/app.rs`). Three things about
that port are worth stating plainly:

- **It projects with the CURRENT camera, not a registered view.** The Python
  side had no live `camera.rs::Controller` to ask and so needed a named view;
  the viewer has one, so a Shift-click is projected with exactly the camera
  that drew the frame the click landed on
  (`ClickCamera::from_render_camera`, widening the renderer's `f32` to the
  `f64` the arithmetic runs in). Everything downstream of the projection —
  the depth-mode seed, the colour gate, the radius and point caps — is the
  same arithmetic, ported line for line.
- **The neighbour search is a spatial hash, not a k-d tree.** No k-d tree
  crate is vendored by either workspace's lock, and `PointGrid::nearest` is
  an exact k-nearest query (it expands cell rings until the k-th distance
  found is provably inside the scanned region), so it returns what
  `cKDTree.query` returns wherever that answer is unambiguous. It sizes its
  cells from the cloud's **interquartile** extent, not its bounding box,
  because a TRIPS export's far-field environment sphere would otherwise drop
  the whole scene into one cell. `edit/cluster.rs`'s own "Tie-breaking"
  section names the three places the two implementations may legitimately
  differ (k-th-neighbour distance ties, depth-sort ties, summation order in
  the seed centroid) and why none of them is reachable in the fixture.
- **The parity is pinned twice.** `tests/fixtures/synthetic/edit_golden/
  click.json` + `expected_click.json` carry a structured scene (a red blob,
  a same-coloured blob behind it, a green blob beside it, and scatter) and
  four clicks that each pin a different branch — depth-mode seeding, the
  colour gate, the `max_points` cut-off, and a miss; both sides must return
  the identical id list. `trips-viewer --click U V --dump-click` runs the
  same comparison against a real bundle
  (`tests/test_edit_viewer_parity.py`).

The one deliberate deviation: the viewer never calls
`default_max_radius_from_bundle`'s point-cloud FALLBACK branch in a way that
is parity-checked, because `median_nn_distance` draws a seeded *numpy*
subsample above 20 000 points and has no portable twin. The camera-spacing
branch (which is what every multi-view bundle takes) is ported exactly; the
fallback uses a deterministic stride and is only reachable on a single-view
bundle.

### 2. Shade-cloud finder (E2)

Implements the audit rule directly, as a *selector* rather than a training
heuristic: **dark, low-confidence points in front of the geometry the shade
frames actually see.** This is exactly
`trippy.train.prune`'s existing, verified functions
(`docs/EXPERIMENTS.md` "Shade audit", "Point removal"):

- `build_shade_region(sparse_dir, frames, znear_frac, zfar_frac)` → per-frame
  camera + depth-slab geometry (a field-for-field port of
  `~/Splats/tools/depthprior_shade_audit.py`'s `build_region`).
- `in_region(views, xyz)` → `(inside, zfrac)` per point — is this point in
  the union of the shade frames' near depth slabs, projected and visible.
- `luminance(rgb)` → Rec.709 luminance of the point's own colour.
- `dark_mass_stats` reads the same `(inside, lum, conf)` triple for the
  audit's headline numbers.

Rather than re-deriving COLMAP projection in Rust, a small precompute step
(`trippy edit-prep`, Python, run once per bundle/scene — not per session)
calls these exact functions and writes `(inside: bool, lum: f32, conf: f32)`
per point into an auxiliary array next to `points.npz`. The viewer's shade-
cloud panel then just **thresholds already-computed numbers live**: three
sliders — `lum_threshold`, `conf_threshold`, and an implicit `inside`
toggle — each recomputing a `pointset` region's membership in microseconds
(a vectorised comparison over a resident array, no COLMAP, no camera math at
interaction time). "Preview highlight" is the new `ViewMode` from §0,
rendering the current threshold's matched points at high saturation against
a dimmed scene. The Inspector's "select" button turns the live preview into
a committed `pointset` region the user can then delete/fade/blend/undo like
any other.

This is the same rule `shade_prune` uses to decide what to drop during
training (`trippy.train.prune.shade_prune_keep_mask`: in-region **AND**
dark **AND** below a confidence cutoff) — reused here as a Jordan-facing
*finder*, not a silent training-time removal. `docs/EXPERIMENTS.md`'s own
caveat about `shade_prune` — "it removes the thing the audit measures; report
it next to held-out shade PSNR" — is why this tool proposes a selection and
lets Jordan decide the action (delete/fade/blend), rather than deleting
automatically the way training-time `shade_prune` does.

### 3. Lid (E3)

Not really a "selection tool" — a fixed-shape region seeded from
`SURFACE_LID.md`'s numbers when the scene is Karekare. **Shipped**, with one
deviation from this paragraph's original "drag the plane, drag the radius
ring": the lid uses the SAME gizmo every other shape does
(`src/edit/gizmo.rs`), so a drag on a handle moves it and a Shift-drag scales
`radius`. One gizmo that behaves identically on three region kinds is worth
more than a bespoke ring, and the plane's normal — the one thing a ring would
add — is already correct on the only scene that has one. See §1/§2.

### 4. SAM 3 lift (E5)

`~/Splats/tools/sam3/repo` (weights already on disk, all local, per
`AGENTS.md` §6 "Public weights and code are fine" / "No hosted APIs with
images") segments **one training photograph** — Jordan picks a registered
view, clicks (or SAM 3's own prompt UI selects) the object, gets a 2D mask.
That mask is lifted to 3D by projecting `points.npz`'s `xyz` into that
view's camera (`docs/GEOMETRY.md`'s pinhole + Saiga-distortion projection,
already implemented on the Rust side in `brush_pyramid::scene::Camera` and
on the Python side in `trippy.scene.colmap_io`/`prune.in_region`'s own
projection code) and keeping the points whose projected pixel falls inside
the mask. Because a single view's mask is ambiguous behind occluders, the
same lift is repeated over **several** registered views of the same object
and combined by **majority vote** per point — the same "several independent
signals must agree" discipline `~/Splats/tools/make_masks3.py`'s v3 person-
mask pipeline already uses (per-instance mask primary, segmentation +
rectangle fallback for recall) and the same discipline `docs/EXPERIMENTS.md`
"Which frames are the shade" uses (luminance + EXIF + camera centroid, three
independent signals). Output is a `pointset` region, same shape as the
shade-cloud finder's and click-to-cluster's — the Inspector does not need to
know which tool produced a `pointset` region.

Everything in this path runs on Jordan's machine: SAM 3's weights are local,
the photograph never leaves it, and no step here calls a hosted API — the
same privacy posture `AGENTS.md` §6 already requires project-wide.

**Implemented** (`trippy/edit/sam_lift.py`, `trippy/edit/sam_runner.py`,
CLI `trippy edits sam`). What actually got built, and the three places it
differs from the paragraph above:

```
trippy edits sam --bundle <dir> --scene <root> --view IMG.jpg \
                 (--point U V | --box X0 Y0 X1 Y1 | --text "a phrase") \
                 [--views-around N] [--device cpu|mps] \
                 --out edits.json [--op fade|delete|blend] [--mix M] \
                 [--preview heat.png] [--summary-out summary.json] \
                 [--sam-work-dir <dir>] [--mask NAME=PATH] \
                 [--mask-threshold 0.5] [--no-scene-distortion] \
                 [--depth-cell-px 16] [--depth-tol 0.15] [--vote-fraction 0.5]
```

- **SAM 3 is never imported into trippy's process.** `trippy/edit/
  sam_runner.py` is both halves of a subprocess pipe: imported, it builds
  and runs the command; executed by *Splats'* SAM venv python, it imports
  `sam3` by `sys.path` from the Splats checkout and writes a mask `.npy`.
  trippy's own `.venv` has none of SAM 3's dependencies (einops / timm /
  iopath / ftfy / pycocotools) and cannot get them without changing this
  repo's dependency set. The exact command (all three paths overridable
  with `TRIPPY_SAM3_PYTHON` / `TRIPPY_SAM3_REPO` / `TRIPPY_SAM3_WEIGHTS`):

  ```
  /Users/nzbirdranch/Splats/tools/sam3/.venv/bin/python \
      <trippy>/trippy/edit/sam_runner.py \
      --repo    /Users/nzbirdranch/Splats/tools/sam3/repo \
      --weights /Users/nzbirdranch/Splats/tools/sam3-weights/sam3.pt \
      --image <photo> --device cpu|mps \
      --resolution 1008 --threshold 0.5 --mask-threshold 0.5 \
      --kind box --box X0 Y0 X1 Y1 \
      --out-mask <dir>/mask.npy --out-json <dir>/info.json
  ```

  The SAM 3 API used is the image processor's own documented one:
  `build_sam3_image_model(device=..., checkpoint_path=<local sam3.pt>,
  load_from_HF=False)` → `Sam3Processor(model, resolution=1008,
  device=..., confidence_threshold=0.5)` → `set_image(PIL image)` →
  `set_text_prompt(phrase, state)` for `--text`,
  `add_geometric_prompt([cx, cy, w, h] normalised, True, state)` for
  `--box`, and — for `--point`, which the processor does not expose —
  `state["geometric_prompt"].append_points(xy_normalised, labels)` followed
  by the same `_forward_grounding(state)` that `add_geometric_prompt` ends
  with (`sam3/model/geometry_encoders.py`'s `Prompt` carries points as
  first-class prompts; only the processor's convenience wrapper omits
  them). Two host-specific details, both trippy's own shim code and
  neither copied from SAM 3: a `torch._dynamo` stub (no `triton` wheel
  exists for macOS/arm64, and `torch._inductor` imports one
  unconditionally — `~/Splats/research/sam3-person.md`), and an **fp32
  rebinding of `sam3.perflib.fused.addmm_act`**, which casts its own inputs
  to bfloat16 unconditionally and hands them to the next fp32 layer.

  This document first said that second shim was a `torch.autocast(bfloat16)`
  region. **That was wrong and it has been removed** (2026-09-07). It made
  the CPU run work but 6x slower (58.8 s against 9.2 s for the same lift
  with `addmm_act` rebound and no autocast, for the same mask — 129,712
  mask pixels, 74,007 points), because bfloat16 on CPU is emulated; and on
  MPS it did not work at all, because the MPS autocast policy casts a
  convolution's input but not its weight. The shipped fix takes bfloat16 out
  of the graph at its source and runs with no autocast on either device;
  fp32 is strictly more accurate than the bf16 path CUDA takes.
  `--device mps` needs three more repairs to SAM 3's own code — the model is
  moved to the device (`build_sam3_image_model` only does that for CUDA), the
  ViT is forced onto its own real-valued rotary embedding (MPS has no
  `torch.view_as_complex`), and **every tensor `nn.Module.to()` leaves
  behind** is moved with it.

  That last one is one bug wearing two hats, and each hat cost a queued run.
  `.to()` moves registered parameters and registered buffers and nothing
  else, so a precomputed cache parked in a plain attribute stays on the CPU:
  `PositionEmbeddingSine.cache` is a **dict** (job `trippy-edit-sam-3`, died
  in `_get_img_feats` with "indices should be either on cpu or on the same
  device as the indexed tensor"), and `TransformerDecoder`'s
  `compilable_cord_cache` is a **tuple** of boxRPB coordinate vectors built
  in `__init__` (job `trippy-edit-sam-4`, died at `decoder.py:380` with
  "found at least two devices, mps:0 and cpu"). So `_move_stray_tensors`
  stopped chasing container shapes and now walks every module's `__dict__`
  and moves every tensor at any depth inside dicts, lists, tuples and sets —
  what `.to()` would have done had SAM 3 registered them as buffers, so it
  moves data and changes no arithmetic. `_stray_tensor_devices` then re-scans
  and the child **refuses to start** if anything is still off-device, naming
  the attribute, rather than letting it surface six layers into a forward
  pass.

  That leaves one place the module tree cannot see: a device-less
  `torch.zeros(...)` made *during* the forward. `--device-audit` (a
  `TorchFunctionMode`, slow, never for a timing run) records every one SAM 3
  makes and prints it to stderr. Run on the box-prompt path (synthetic image,
  random weights, CPU, 2026-09-08) it found five, and all five are followed
  immediately by an explicit `.to(device)` in SAM 3's own code:
  `sam3_image_processor.py:54` and `:209`, `geometry_encoders.py:650`,
  `tokenizer_ve.py:250` and `:255`. The forward path is clean.
  `docs/LIMITATIONS.md`'s "SAM-3 mask lift" section has the shims with the
  job names that found them.
- **The mask is depth-gated before it becomes a selection.** A mask is 2D:
  everything behind the object along the same ray is inside it too. Points
  are binned into 16 px cells and each cell's *nearest supported* depth
  mode (log bins of relative width `--depth-tol`, a bin counting as a
  surface at 25% of the cell's fullest bin) is the surface there;
  candidates survive within a relative band of it. Nearest-**with-support**
  rather than the plain mode is load-bearing: a mask over a near object
  usually contains more background than object (a distant wall projects
  many more points per pixel), so the plain mode would lock onto the
  background and select exactly the wrong thing. Measured on the
  `exp0010-shade-prune` bundle: 104,218 points project inside the mask,
  74,007 survive the gate (fp32, `--views-around 0`, ~9 s on CPU).
- **Neighbour views are prompted with the selection's centroid, and the
  vote is strict.** `--views-around N` takes the N nearest capture views
  (by camera centre) in which the primary selection's centroid actually
  projects in-frame, prompts SAM there with that projected pixel (a point
  prompt), and keeps a point when **more** than `--vote-fraction` of the
  views that can *see* it voted for it (`floor(f·n) + 1`). "At least half"
  would let one sloppy mask carry a point past a neighbour that rejected
  it, which is the failure the vote exists to prevent. The corollary is
  that `--views-around 1` is an *intersection*, not a consensus (two
  eligible views, both must agree): measured on kk-coherent (under the
  since-removed bf16 autocast; the fp32 path selects 74,007 from the
  prompted view), 72,455 points from the prompted view and 15,476 from one
  neighbour left 15,111. Use 0 or >= 2.

Two more things the design did not say and the implementation had to
decide. The photographs under `--scene` are as-captured, while a
trippy-native bundle's views are undistorted (`distortion` all zeros,
`trippy.scene.dataset` undistorts on ingest), so projecting into the photo
re-applies the COLMAP camera's own `(k1, k2, p1, p2)` (`--no-scene-
distortion` turns that off; ~19 px at the frame edge on kk-coherent's
`k1 = 0.057`, ~0 at the centre). And nothing in the lift ever opens the
photograph: only the SAM child decodes pixels, masks are arrays, and
`--preview` draws a from-scratch heatmap of *projected point counts* — no
photographic content, so it is safe to look at under `AGENTS.md` §6.

### 5. The SAM tool in the viewer (E5, viewer half)

**Implemented** (`rust/crates/trips-viewer/src/edit/sam.rs`, `src/sam_child.rs`,
the `Tool::Sam` panel in `src/edit_ui.rs`, the gestures in `src/app.rs`, the
`--sam-*` flags in `src/main.rs`).

The viewer does not segment anything. It is a **process supervisor** for the
command above: SAM 3 is never imported into trippy's process (§3) and it is
never imported into the viewer's either. What the tool adds is the gesture,
the pixel mapping, one child process, and the import.

- **The gesture.** With the SAM tool selected (`T` cycles to it), a primary
  **drag** on the render draws a marquee and becomes a `--box`; **Alt-click**
  becomes a `--point`. The camera is not lost: right-drag still looks and shift+left- or
  middle-drag still pan, and outside this one tool the primary drag is
  untouched. A drag shorter than `MIN_BOX_PX` (8 render px) is treated as a
  click, not a 2 px box.
- **The camera must be pinned to a capture view.** The lift needs the
  PHOTOGRAPH, and a free-flying frame corresponds to no photograph at all. If
  the camera has been moved off a view the gesture is **discarded** and the
  camera snaps to the nearest capture view by camera centre
  (`edit::sam::nearest_view`), with the panel saying which one and asking for
  the drag again. Re-using the pixels after the snap would segment the wrong
  part of the image while looking like it worked.
- **The pixel mapping is the one piece of arithmetic here, and it is
  unit-tested** (`edit::sam::view_pixel_from_render`). A render pixel is not a
  view pixel: `camera::Controller::render_camera` re-fits the view to the
  window and to the render-scale lever, and it scales `fy` by the WIDTH ratio
  while scaling `cy` by the HEIGHT ratio — so on a window-shaped render a
  plain `height` ratio in `v` is wrong. The mapping goes through normalised
  image coordinates instead, `(u - cx_r)/fx_r · fx_v + cx_v`, which is exact
  for every camera that shares the view's `R`, `t` and distortion (the tests
  check it by projecting world points through both cameras and comparing).
  Distortion cancels for the same reason and none is applied.
- **The viewer sends VIEW pixels, not photo pixels.** It never opens a
  photograph (`AGENTS.md` §6), so it cannot know the photo's size; the child
  applies that scene's own `photo_scale` when it is told
  `--prompt-space view`. That flag exists solely for this.
- **`--scene` is not passed at all.** `bundle.json` now records `scene_root`
  (`trippy.render.bundle.bundle_document`) and the child reads it from there.
  A bundle exported before 2026-09-07 has no such key, and the panel says so
  and disables the run button rather than guessing where the photographs are.
- **One child, only when Jordan asks.** This is his interactive use of his own
  GPU, which `AGENTS.md` §6 allows outside the queue; a tool that could fan
  out into several SAM runs would not be. `SamJob::spawn` is the only place a
  process is created, `EditSession` holds at most one, and it is killed on
  Cancel **and on drop** so closing the window cannot leave a run holding the
  GPU. The interpreter is `<trippy_root>/.venv/bin/python`, from
  `bundle.json`'s own `trippy_root`, overridable with `$TRIPPY_ROOT` (a
  checkout) or `$TRIPPY_PYTHON` (an interpreter).
- **Progress and Cancel.** Two reader threads drain the child's stdout and
  stderr into the panel (a child that fills its stderr pipe while the parent
  reads only stdout deadlocks). The `sam: ` lines appear as they arrive;
  stderr appears prefixed `! ` and is never parsed as JSON. Exit 0 with **no**
  JSON summary line is a failure, not a silent success — the region file might
  exist but the counts would be invented.
- **The import is one undo step.** The child writes its region into a
  throwaway `edits.json` under `TMPDIR` (never the session's own file); the
  viewer reads the last region out of it, gives it a fresh id, and adds it
  through `EditDocument::add_region` like every other edit. `Cmd-Z` removes
  it. The imported points are tinted immediately (the same magenta the other
  two selection tools use); `H` toggles that tint while the SAM tool has
  focus.
- **The default device is `mps` (flipped 2026-09-09), and the rule that
  decided it is written down.** The decision rule, agreed 2026-09-08 while
  jobs `trippy-edit-sam-5` (MPS) and `trippy-edit-sam-5-cpu` (CPU, same
  photo, same box, same code) sat in the queue: **the default becomes `mps`
  only if `edit-sam-5` returns rc=0 AND its `per_view[0].segmenter.seconds`
  beats `edit-sam-5-cpu`'s.** Both landed: `edit-sam-5` rc=0 at **8.05 s per
  view**, `edit-sam-5-cpu` at **9.65 s per view**
  (`output/edits/edit-sam-5{,-cpu}/summary.json`, `segmenter.seconds`) — MPS
  is rc=0 AND faster, so the rule is satisfied and the default is now `mps`.
  `cpu` remains a fully supported fallback, one click away in the viewer's
  device radio group (`rust/crates/trips-viewer/src/edit_ui.rs`'s `SamUi`)
  and the headless twin's `--sam-device cpu`
  (`rust/crates/trips-viewer/src/main.rs`) — an MPS default that is slower
  than CPU would have been a trap, not a feature, but it is not: it is 17%
  faster. `trippy.edit.sam_runner.Sam3Segmenter`'s own dataclass default
  (the Python API a script builds a segmenter from directly, without going
  through the viewer or the `trippy edits sam` CLI) is `mps` for the same
  reason. **Known gap, flagged for the Orchestrator**: `trippy edits sam`'s
  own terminal CLI default (`SAM3_DEFAULT_DEVICE` in `trippy/constants.py`,
  read by `trippy/cli.py`) is unchanged at `cpu` — those two files were
  outside this task's edit list, so a one-line follow-up
  (`SAM3_DEFAULT_DEVICE = "mps"`, plus its `--device` help text) is needed
  to make the terminal command agree with the viewer and the Python API.
- **`--fake` is how this is tested and screenshotted.** `trippy edits sam
  --fake` (or `TRIPPY_SAM_FAKE=1`) synthesises the mask from the prompt — a
  box fills its rectangle, a point fills a disc — and runs the entire rest of
  the shipped path: projection, depth gate, vote, region, summary. It loads no
  checkpoint and touches no GPU, so the viewer's child-process state machine
  and the region import are testable in CI and in `--screenshot`. The summary
  says `"segmenter": "fake"` and never claims SAM ran.

Headless twin, and the E5 screenshot proof:

```
trips-viewer <bundle> --sam-box X0 Y0 X1 Y1 --sam-fake [--sam-op delete]
             [--sam-point U V] [--sam-views-around N] [--sam-mix M]
             [--sam-device cpu|mps] [--sam-undo] --screenshot out.png
```

Measured on the generated `synthetic-splat` bundle (`tools/make_synthetic_
splat_bundle.py`; 4000 points, 48x36 views), box `12 9 36 27` on `IMG_0.jpg`,
`--sam-fake --sam-views-around 0`:

| run | result |
|---|---|
| no `--sam-box` | 4000 points rendered — the baseline frame |
| `--sam-op delete` | 530 of 4000 points lifted and removed; 1719 of 1728 pixels differ from the baseline, mean abs difference 6.15/255 |
| `--sam-op fade --sam-mix 0` | nothing deleted, the tint alone: 1717 pixels differ, mean 12.50/255 |
| `--sam-op delete --sam-undo` | **byte-identical to the baseline** (max abs difference 0) |

## 4. UI sketch

### Simple Mode (default since 2026-09-08)

Jordan, on the full Karekare scene: *"I don't get how to use the editor at all,
needs to be more simple."* The four panels below are still there, but they are
now behind a **Simple / Advanced** switch and Advanced is not what opens.
Simple Mode is one panel, four numbered steps, and no word Jordan has to look
up — no "op", "mix", "threshold", "pointset", "voxel" and, specifically, no
"gizmo" (*"Idk what a gizmo is"*): the handles are **move arrows** in every
string the UI shows. `edit_ui.rs`'s own unit test asserts that.

```
┌─ Edit (M) ───────────────── [Simple] Advanced ─┐
│ Do these in any order. Cmd-Z undoes anything.  │
│                                                 │
│ 1. Find shade clouds                            │
│    [Find them] [x] highlight them               │
│    18 231 of 7 512 044 points look like shade   │
│    cloud (0.24% of the scene)                   │
│    [Soften them] [Remove them]                  │
│                                                 │
│ 2. Select an object                             │
│    [on -- click the scene]                      │
│    608 of 5 000 000 points selected (0.012%)    │
│    [smaller] [bigger]  reach 0.29 world units   │
│    [Keep this as an object] [Start again]       │
│                                                 │
│ 3. Paint an area                                │
│    [on -- drag on the scene]                    │
│    brush size |--------| 0.31 world units [ / ] │
│    what happens where you paint:                │
│      [remove] [soften] [choose]                 │
│    soften: leave the points where they are, but │
│      fade this area towards the splat.          │
│                                                 │
│ 4. What to show here                            │
│    [x] brush-1  (312 points)   [forget it]      │
│        [remove] [soften] [choose]               │
│        Splat  <-->  TRIPS  |------------|       │
│                                                 │
│ Put a shape somewhere                           │
│    [+ box] [+ ball] [+ pool lid]                │
│    it goes where you click, sized to how far    │
│    away that is                                 │
│                                                 │
│ [undo] [redo] [save] [reload]                   │
└─────────────────────────────────────────────────┘
```

Three rules make it work, and each is a change to behaviour rather than to
wording:

1. **A plain click is claimed, in this order**: an armed placement first, then
   Simple Mode's "select an object" step. Everywhere else selection is still
   SHIFT-click, so no launcher, script or habit changed.
2. **Placement is arm-then-click.** `+ box` does not create anything; it arms
   [`Placement`], and the next plain click on the render creates the region ON
   the point under it (`EditSession::resolve_placement`), sized to
   `PLACEMENT_SIZE_DEPTH_FRACTION` of that point's camera-space depth. Pressing
   the button again disarms.
3. **A greyed control says why.** With no `blend` block in `bundle.json` there
   is no Gaussian half, so the Splat/TRIPS slider is disabled and carries the
   sentence *"This bundle has no splat to mix. Open a combined bundle."*
   (`EditSession::no_splat_reason`). It used to move and do nothing.

The `?` overlay `app.rs` paints lists six gestures, from `edit_ui::MOUSE_HELP`
— one list, quoted by `docs/USER_GUIDE.md`, asserted jargon-free by a test.

### Advanced Mode

Four egui panels (the sketch's three, plus **Named Objects**), added the same
way the existing HUD window is built (`app.rs::ViewerApp::overlay`, an
`egui::Window`) — an "Edit" window shown alongside it, toggled independently of
the existing Tab-toggled HUD so Jordan can hide edit chrome while still flying
around:

```
┌─ Regions ──────────────────────┐  ┌─ Inspector ──────────────────┐
│ [x] pool lid            lid    │  │ name:  pool lid               │
│ [x] shade: big tree     pts    │  │ kind:  lid                    │
│ [ ] box: fence           box   │  │ op:    (o) delete ( ) fade    │
│                                 │  │        ( ) blend              │
│ [+ box] [+ sphere] [+ lid]      │  │ mix:   |----------------| 100%│
│                                 │  │        (splat)      (TRIPS)   │
│  drag to reorder (paint order) │  │ height: [-0.49    ] up: [...] │
└─────────────────────────────────┘  │ radius: [2.5] falloff:[1.0]  │
                                      │ band:   [0.05]                │
┌─ Named Objects (2 groups) ──────┐  │ [preview] [select] [delete]  │
│ brush (2)                        │  └────────────────────────────────┘
│  [x] brush-1   [brush delete] 312 pts
│      mix ----   [solo][rename][remove]
│  [x] brush-3   [brush fade]   88 pts
│ hand-authored (1)                │
│  [x] pool lid  [lid delete]  1204 pts
└───────────────────────────────────┘

┌─ Tools ─────────────────────────┐
│ ( ) click-to-cluster            │
│ ( ) shade-cloud finder           │
│     lum <  [0.25]                │
│     conf < [0.50]                │
│     inside shade region  [x]     │
│ (x) SAM 3 lift: drag a box on    │
│     the render, or alt-click     │
│     op/mix, views around, device │
│     [run SAM lift] [cancel]      │
│ ( ) brush: drag to paint, alt to │
│     erase; radius / weight       │
│     [start a new region]         │
└───────────────────────────────────┘

The 3D gizmo is not a panel: it is three coloured handles drawn over the render
itself, on whichever region the Inspector has selected (box, sphere or lid).
```

**Keys** (chosen to avoid every key `app.rs::ViewerApp::handle_input`
already binds — `V X Tab - = F R N P W A S D Q E` and drag/scroll):

**As shipped** (E1/E2/E4/E5, plus the brush and the gizmos, 2026-09-08).

| key | action |
|---|---|
| `M` | toggle edit mode (shows Regions/Named Objects/Inspector/Tools). `--edit` opens straight into it. Click-drag on the canvas still orbits *unless it starts on a gizmo handle* — the drag is scoped to what it started on, so navigation is never taken away |
| **Shift-click** | E4's selection gesture: cluster the object under the pointer. Read from the scene `Response` like every drag, and scoped to `clicked()` rather than `dragged()`, so a Shift-DRAG still orbits and navigation loses nothing |
| **drag** | E5's box prompt **while the SAM tool has focus**, a brush stroke **while the Brush tool has focus**, and a move-arrow drag when it STARTS on a handle. Right-drag still looks around and shift+left- or middle-drag still pans in every case, so each gesture is borrowed rather than taken |
| **Alt-click / Alt-drag** | E5's point prompt; with the Brush tool, an ERASE stroke. Alt is bound to nothing else in this viewer, so this costs no existing gesture |
| **drag a gizmo handle** | translate the selected box/sphere/lid along that handle's world axis. **Shift-drag** a handle resizes; **Ctrl-drag** rotates a box about that axis (nothing else has an orientation to rotate). One drag = one undo entry, however many frames it took |
| `T` | cycle the active tool (Regions → shade-cloud finder → click-to-cluster → SAM 3 lift → brush) |
| arrows, `PageUp`/`PageDown` | nudge the selected region along world X/Z and Y by `NUDGE_SCENE_FRACTION` of the scene diameter — **kept alongside the gizmos**: an exact step needs neither a visible handle nor a mouse, and both go through the same `gizmo::translated` |
| `[` / `]` | shrink/grow the selected region by `RESIZE_STEP` — or the **brush radius** by `BRUSH_RADIUS_STEP`, while the Brush tool has focus |
| `Delete` / `Backspace` | remove the selected region (its `op` is a separate field in the Inspector) |
| `Cmd`/`Ctrl` + `Z` | undo |
| `Cmd`/`Ctrl` + `Shift` + `Z` | redo |
| `Cmd`/`Ctrl` + `S` | save `edits.json` |
| `H` | toggle the preview highlight of whichever tool has focus (a tinted point cloud, not a new `ViewMode` — see §6). Where several tools have a live preview, the union is tinted the one colour. Since 2026-09-08 the **brush has one too**: the points its region claims are recomputed once at button-up (`EditSession::refresh_brush_tint`, one `brush::membership` hash probe per point, never per sample) and tinted like everything else, because "what a stroke paints IS the region" was only visible for a `delete` op. An undo or a redo DROPS that tint rather than leave a stale one, which is what keeps `--brush-undo`'s frame byte-identical to a run that never painted |
| `Cmd`/`Ctrl` + `N` | *not built*, and not needed: "new region from the current selection" is the **add as region** button next to each tool's own selection (E2's shade finder, E4's click-to-cluster), where the op and mix for it are chosen |

Left-click behaviour while in edit mode: **unchanged from viewing**, except
where a tool or a handle has explicitly claimed the drag. A plain drag orbits; a
plain click does nothing. E4 added exactly one gesture on top — **Shift-click**,
which clusters — rather than taking the plain click away. The gizmos keep the
same rule and are the clearest case of it: a drag is scoped to what it started
on (`GizmoScreen::pick`, within `GRAB_PX` of a handle, decided once when the
button goes down), the same `dragged_by`-not-a-global-flag discipline `app.rs`'s
own module doc calls out as the fix for the `egui_wants_pointer_input` trap. Two
consequences worth stating plainly: the handles a drag hit-tests against were
projected at the end of the PREVIOUS frame (the camera is built after
`handle_input`), which is invisible in practice because a camera moving fast
enough to matter is a camera being orbited; and the handles and the brush ring
are drawn by the egui painter OVER the finished frame, so `--screenshot` shows
the edit and never the tool.

The clicked pixel is handed on in the **render's** coordinates (egui points
times `pixels_per_point` times the render-scale lever), not egui points, because
that is the pixel space the camera the frame was drawn with is defined in. The
camera is therefore built *before* the editor's per-frame work in
`ViewerApp::ui`, so a click made on frame N is projected with frame N's camera
rather than the next one's.

## 5. Publish

Two artefacts an edit needs to reach, and a CLI that drives both:

```
trippy apply-edits --bundle <dir> [--edits <dir>/edits.json] \
                    --out <dir> [--target trips|distilled|both] \
                    [--distilled-ply <path>]
```

**Implemented** (`trippy/edit/apply.py`, `trippy/edit/checkpoint.py`, E6).
`--target` defaults to `both`:

- **TRIPS `points.npz` + `export.ply`** (`target=trips`/`both`):
  `apply_edits` writes the filtered `points.npz` + `blend_weights.npy` (as
  before) and, in the SAME run, a filtered 3DGS-style `export.ply` of the
  kept TRIPS points via `trippy.train.export.write_gaussian_ply` — the
  identical writer `Trainer.export_ply` calls at training time, so Splats'
  audits and Brush can open the edited TRIPS point set directly, with no
  bundle loader involved. `bundle.json`'s own `blend.splat_ply`, if named,
  is filtered in the same run exactly as before (`filter_gaussian_ply`).
  `blend`/`fade` regions do not change what is exported to a 3DGS-shaped
  PLY (that format has no TRIPS-vs-splat mix concept); they only affect
  `blend_weights.npy` and the checkpoint-side gate suppression below.
- **The distilled splat** (`target=distilled`/`both`, design B,
  `trippy/distill/render_set.py`): per `docs/decisions/
  ADR-0007-viewer-editing.md` §"Publish order is edit-TRIPS-first, then
  distil", `--edits` on `trippy distill --stage render/all` applies
  `delete`-op regions to the checkpoint's own point cloud (via
  `trippy.edit.checkpoint.apply_edits_to_trainer`, the SAME
  `Trainer._apply_keep_mask` surgery training uses) *before* any camera is
  rendered — deleted content never appears in the image set Brush trains
  on, not merely in the finished PLY. Separately, `trippy apply-edits
  --target distilled --distilled-ply <path>` re-applies `box`/`sphere`/
  `lid` regions (pure world-space geometry, not point-ID-based) directly to
  an *already-distilled* PLY's own `xyz`, no re-distillation needed — safe
  because these regions never depended on the TRIPS point cloud's row
  order. `delete` removes rows (`trippy.edit.apply.apply_gaussian_ply_edits`,
  a single-pass header-preserving PLY filter like `filter_gaussian_ply`);
  `fade` instead SCALES the surviving row's own alpha
  (`sigmoid(opacity)`) by the region's graded weight and writes it back as
  `opacity = logit(...)` — the "fade → opacity scaling" this PLY-only path
  uses in place of a TRIPS-vs-splat mix it has no channel for. `pointset`
  regions are skipped here (they do not survive distillation by
  construction, §1) and produce a `summary["distilled"]["warning"]` note if
  an enabled one exists.
- `--target both` (the CLI default) always runs the `trips` half and runs
  the `distilled` half only when `--distilled-ply` is given, so a single
  save in the viewer produces a consistent TRIPS export and (once a
  distillation exists) a consistent distilled-splat publish without Jordan
  re-specifying the same edits twice; omitting `--distilled-ply` is
  recorded as `summary["distilled"] = {"skipped": ...}` rather than
  silently doing nothing (§7's own "not discovered by Jordan after the
  fact" risk).

**Checkpoint-side keep mask** (`--edits edits.json` on `trippy
candidate-report`/`trippy eval`, `trippy.edit.checkpoint.
apply_edits_to_trainer`): the same `edits.json` can run directly against a
trained checkpoint, without first publishing a bundle, so a candidate
report or held-out eval reflects an edit immediately. It loads the doc,
computes the keep mask + per-point weights (`trippy.edit.weights.
compose_trips_weights`), and applies the keep mask via the identical
`Trainer._apply_keep_mask` index-select surgery the trainer uses at a
training epoch boundary — permanently, on the in-memory `Trainer` only,
never written back to the `.pt` file. On a gate hybrid checkpoint
(`trippy.hybrid.gate`), the trained gate is additionally multiplied, per
pixel, by the per-point weight's projection wherever a `blend`/`fade`
region touches a surviving point — suppressing the gate towards TRIPS
there so an un-edited, live-rendered splat cannot leak back into a region
the edit asked to hide (there is no live per-point splat channel to move
on a checkpoint-only path, §7). Per-pixel weight rendering has no
first-class output in the Python renderer today, so the per-point weight
is splatted as an auxiliary feature channel through `render_pyramid` and
read back from level 0 — the exact same alpha-compositing trick §2
documents for colour, applied to one more channel instead of a new kernel.
This gate-suppression step runs inside `trippy.render.candidate.
render_candidate` (`candidate-report`, and by extension `trippy distill
--stage render`'s own per-pose renders) AND inside `Trainer.evaluate`
itself (`trippy eval --edits`, `trippy.train.eval.evaluate_checkpoint`) —
**previously a documented gap** (`trippy eval`'s render path applied the
keep-mask deletion but not the gate multiply), **closed 2026-09-07**:
`Trainer.evaluate` now calls `trippy.edit.checkpoint.render_edit_weight_map`
itself, right after splitting the gate off the network output and before
`apply_gate`, exactly where `render_candidate` already did. The non-edit
path is unchanged bit-for-bit — `render_edit_weight_map` returns `None`
whenever `apply_edits_to_trainer` was never called or found nothing to
suppress, so an unedited `trippy eval` takes the exact code path it always
did (`tests/test_train_eval.py`: an empty `edits.json` reproduces the
baseline metrics exactly; a `delete` region still changes PSNR by removing
points; a `fade`/`blend` region now ALSO measurably lowers the reported
gate mean).

`apply-edits` and `--edits` are additive to the pipeline order that already
exists (train → export/distill), not a new stage inserted into training —
nothing here changes `Trainer.fit`'s loop or `maybe_prune_points`'s
schedule.

## 6. Milestones

| # | Scope | Estimate | Acceptance | Status |
|---|---|---|---|---|
| **E1** | Region data model (`edits.json` read/write), `box`/`sphere` region kinds, per-point weight compositing (§2's non-depth design) on the TRIPS side only, undo/redo, save/load surviving a bundle close+reopen | 1 wk | Draw a box or sphere region in the viewer, set `mix`, see the TRIPS render change inside it and nothing outside it; undo removes the region; closing and reopening the bundle restores it exactly (region list, mix values, undo history) | **Done** (2026-09-07); the **3D drag gizmos landed 2026-09-08** (`src/edit/gizmo.rs`: three projected world-axis handles on the selected box/sphere/lid, drag to translate, Shift-drag to resize, Ctrl-drag to rotate a box, one undo entry per drag), alongside the Inspector fields and the arrow-key nudge / `[`-`]` resize E1 shipped first — the keyboard steps stayed and now share `gizmo::translated`/`gizmo::resized` with the drag. `rust/crates/trips-viewer/src/edit/*` + `src/edit_ui.rs`. Measured on the synthetic bundle: a `blend` sphere at `mix = 0` changes 4.74 % of the pixels in its own half of the frame and **0.00 %** of the other half; a `delete` box changes 63.97 % of its half; undoing it reproduces the unedited frame **bit for bit** (max channel diff 0). Gizmo proof (2026-09-08, same bundle at 480x360): `--move-region "left blob" 0.9 0 0`, the headless twin of a translate drag, changes **8.22 %** of the frame's pixels (max channel diff 57) against the same edits rendered without it. |
| **E2** | Shade-cloud finder: `trippy edit-prep` precompute + live threshold sliders + preview-highlight `ViewMode` + "select" → `pointset` region | 3 d | On a bundle whose scene has registered shade frames, the finder's default thresholds select a `pointset` region whose `dark_mass_fraction` (via `trippy.train.prune.dark_mass_stats` on the selected IDs) matches the audit's own number for that scene to float precision; deleting the region and re-running the audit shows the drop | **Done** (2026-09-07). Four live sliders (lum, conf, znear/zfar fractions) over `shade_views.json`, the precompute written by `trippy.edit.golden.write_shade_views`; "add as region (fade / delete)"; and a preview that **tints** the selection instead of adding a `ViewMode` (see the note under this table). `trips-viewer --dump-shade` selects the same ids and the same `dark_mass_fraction` as `trippy.edit.shade_finder.find_shade_pointset` on the synthetic bundle (`tests/test_edit_viewer_parity.py`). |
| **E3** | `lid` region kind + 3D gizmo (plane/radius drag) + hard-clip delete semantics | 2 d | Loading the Karekare pool bundle with a `lid` region seeded from `SURFACE_LID.md`'s numbers removes the haze from every angle at every `mix`/exposure; dragging the radius ring changes the affected point count live | **Done** (2026-09-08). Region kind + hard-clip semantics: `trippy.edit.model.lid_membership`, `trippy edits add-lid`. The 3D gizmo is E1's (`src/edit/gizmo.rs`), which treats a lid like every other shape: its handles are placed at `lid.center` and a Shift-drag scales `lid.radius`, so "drag the radius ring" is a Shift-drag on a handle rather than a ring of its own. Not built: a handle for `up`/`height` specifically — the plane's normal is still typed in the Inspector, where the Karekare numbers are already correct. |
| **E4** | Click-to-cluster: ray cast + k-d tree + k-NN growth in world+colour space, radius slider | 3 d | Clicking a point-cloud cluster selects a `pointset` region that visibly matches the clicked object's extent, without needing a depth buffer | **Done** (2026-09-07), on both sides. Python: `trippy.edit.cluster`, `trippy edits click` (`tests/test_edit_cluster.py`). Rust: `edit/cluster.rs` (the projection, the depth-mode seed, an exact k-NN spatial hash in place of the unavailable k-d tree, the colour/radius/point gates), the **Selection panel** with the four sliders + op/mix + "add as region" + "clear" in `edit_ui.rs`, and **Shift-click** on the render in `app.rs`. Parity is exact, not approximate: the committed `click.json`/`expected_click.json` fixture replays four clicks (depth-mode seeding, the colour gate, the `max_points` cut-off, a miss) and both languages return the identical id list; `--click U V --dump-click` reproduces it against a real bundle. Measured on the synthetic bundle at 480x360: a click at (240, 180) selects 261 of 4000 points, the tint changes **71.55 %** of the frame's pixels (3.81 % of them turning magenta, the rest dimmed by the preview) with a max channel diff of 79, and a run without `--click` reproduces the untinted frame **bit for bit** (0.0000 % of pixels differ, max channel diff 0). Not built: an Inspector-side gizmo for a committed `pointset` region (there is no shape to drag). |
| **E5** | SAM 3 lift: photo segmentation, multi-view projection + majority vote, `pointset` region output | 2–3 wk | Segmenting an object in 2–3 registered views of the same scene produces one `pointset` region that, previewed, highlights that object and not its neighbours; runs entirely local (no image leaves the machine) | **Shipped, both sides.** Python: `trippy/edit/sam_lift.py`, `trippy/edit/sam_runner.py`, `trippy edits sam` (§3's "Implemented" note has the command, the depth-gate and the vote rules). Viewer: the `SAM 3 lift` tool — drag a box or Alt-click on the render while pinned to a capture view, one `trippy edits sam` child with live progress and a Cancel button, and the region imported through the undo log and tinted (§3's "5. The SAM tool in the viewer"). SAM 3 runs locally in a subprocess under Splats' SAM venv, on CPU (~9 s/view) or MPS; the whole path is CPU-testable and screenshottable with `--fake` / `TRIPPY_SAM_FAKE=1`, which needs no checkpoint and no GPU (`tests/test_edit_sam.py`, `trips-viewer --sam-box`). Not built: batching several views into ONE child (each view still pays a model load, `docs/LIMITATIONS.md`), and any per-point clean-up of the returned selection. |
| **E6** | Publish path: `trippy apply-edits`, TRIPS `export.ply` mask wiring, distilled-splat publish order (edit-then-distil for pointset regions, geometry-reapply for box/sphere/lid) | 3 d | `trippy apply-edits --target both` on a bundle with a mix of region kinds produces a TRIPS PLY with the deleted points absent and, after a `trippy distill` run on the same edited bundle, a distilled PLY that also lacks them; a box/sphere/lid region re-applied directly to an already-distilled PLY (no re-distillation) also removes the matching geometry | **Done** (`trippy/edit/apply.py`, `trippy/edit/checkpoint.py`): `--target trips\|distilled\|both` (default `both`); `trips`/`both` write the filtered TRIPS `points.npz`/`blend_weights.npy`/`export.ply` and, if named, a filtered `blend.splat_ply`; `distilled`/`both` (with `--distilled-ply`) re-apply box/sphere/lid regions to an already-distilled PLY as delete/opacity-scale-fade; `--edits` wired into `trippy distill --stage render` (edit-then-distil ordering) and into `trippy candidate-report`/`trippy eval` (checkpoint-side keep mask + gate suppression on gate-hybrid checkpoints, BOTH commands as of 2026-09-07 — see §5's own paragraph). |

**Usability pass, 2026-09-08 (`feat/viewer-simple-mode`).** Jordan drove the
finished editor on the full Karekare scene (7.5M points) and reported five
concrete failures. Each is fixed as a behaviour change, and each has a test:

| what Jordan said | what was wrong | what changed |
|---|---|---|
| "make it left click and drag to orbit, right click and drag to POV free move the camera, like brush" | a left-drag orbited or looked depending on an invisible mode; the right button panned; the wheel changed speed in one mode and distance in the other | `camera.rs`: left = orbit, right = first-person look (WASD flying while held), middle / shift+left = pan, wheel = dolly, shift+wheel = fly speed, **double-click = put the focus point on what you clicked** (`Controller::set_focus`, depth from `brush::depth_anchor_f32`). [`Mode`] is now only a fence, not a control scheme |
| "I don't get how to use the editor at all, needs to be more simple" | four dense panels of sliders with units | **Simple Mode**, on by default; see §4 |
| "I couldn't put boxes or spheres where I wanted" | `+ box` dropped a region at the camera's look-at point, at one size for the whole scene | arm-then-click placement: the region is created ON the clicked point, sized to `PLACEMENT_SIZE_DEPTH_FRACTION` of its depth (`EditSession::resolve_placement`, headless `--place`) |
| "Clicking to select an object seemed to just select the whole scene" | `max_radius` defaulted to the median nearest-CAMERA spacing — metres on a walked capture — and nothing stopped k-NN growth in a connected cloud | `cluster::depth_capped_max_radius` (min of 15% of the click depth and 2% of the captured area, floored at the catchment disc, times the grow/shrink scale), `ClickParams::density_gate` (stop where the cloud thins out), and a seed clipped to the surface under the cursor instead of a cone through the cloud. All three are OFF on the `trippy edits click` parity path |
| "Mix sliders didn't seem to make any difference" | that bundle had no Gaussian block, and the slider moved anyway | `EditSession::no_splat_reason`: the slider is greyed and says "This bundle has no splat to mix. Open a combined bundle." |
| "Brushing seemed to blur the foreground and the background" | a scene-sized default radius, an anchor that took the nearest point in a fixed 12 px catchment whatever the brush size, no depth continuity along a stroke, and a default op of `delete` | radius from the view distance (`DEFAULT_BRUSH_VIEW_FRACTION`), anchor inside the brush's OWN ring (`brush::ring_radius_px`), `brush::clamp_stroke_depth` (a sample may move at most 1.5 radii in depth from the last), default op `fade`, and a magenta tint of what was painted |
| "Idk what a gizmo is" | the word was in the UI | "move arrows" in every user-visible string; a test asserts it |
| "Undo worked" | — | unchanged |

**Three deviations worth naming.** (1) The preview highlight is not a fourth
`ViewMode`: it is a copy of the point cloud with the selection's first three
feature channels set to magenta and everything else dimmed
(`edit::apply::tinted_points`), which rides the existing render with no new
shader and no new `match` arm, and is *most* honest in the existing
`RawLevel0` view where the tint is the rasteriser's own pixels rather than the
network's guess about them. (2) The shade finder's `d` (each frame's median
COLMAP-observed depth) is precomputed into `shade_views.json` rather than
re-derived in Rust — it is the one input no slider moves. Everything a slider
DOES move (`znear = znear_frac * d`, `zfar`, luminance, confidence) is ported
and recomputed live in `edit/shade.rs`; there is no subprocess in the
interaction loop. Without the sidecar the finder falls back to estimating `d`
from the bundle's own points and **says so in the panel**.

E1 already includes undo/save per the brief; E2–E5 add tool-specific
selection UI on top of the E1 data model and do not need to repeat undo/save
plumbing. `brush`-kind (voxel) regions and the Named Objects panel are in no
milestone above — they are additive to E1's data model in the same way E2–E5
are, and are tracked here rather than given their own "E" numbers. Both are
now **shipped on both sides** (2026-09-08):

- **Brush.** Python: `trippy.edit.model`'s `brush_membership`/`paint_sphere`/
  `paint_along`/`erase` + `trippy edits add-brush`. Viewer:
  `src/edit/brush.rs` and the `brush` tool (drag to paint, Alt-drag to erase,
  `[`/`]` for the radius, a weight slider, one undo entry per stroke). Parity
  is exact, not approximate: `edit_golden/brush.json` records the three
  authoring calls and both sides replay them to the same cells **in the same
  order** with the same weights, then agree on every query point's membership;
  `--dump-weights` re-checks the membership over a real `points.npz`
  (`tests/test_edit_viewer_parity.py`). Measured on the synthetic bundle at
  480x360: one dab at (240, 180) with `radius 0.5 --brush-op delete` removes
  110 of 4 000 points and changes **4.21 %** of the frame (max channel diff
  50); a two-pixel stroke to (300, 210) removes 299 and changes **8.51 %**;
  `--brush-undo` reproduces the unedited frame **byte for byte** (0.00 % of
  pixels differ, max channel diff 0). **2026-09-08:** the per-sample depth
  anchor (`brush::depth_anchor`) went from **47.4 ms/sample brute force to
  0.011 ms/sample** via `edit::ScreenGrid` on the real `kkv2-1-full-masked`
  bundle (7.5M points), and the sidecar writer described in §1's `brush`
  entry landed on the viewer side.
- **Named Objects.** Every region with its name, source tool, kind, live point
  count, enable toggle, mix slider, rename, remove and **solo**, grouped by
  `Region.source.tool` with hand-authored regions in their own group. Viewer
  regions are auto-named by `model::auto_region_name`, the port of
  `trippy.edit.model.auto_region_name`, pinned case for case by
  `edit_golden/names.json`. Solo is a way of looking and never saved:
  `weights::compose_point_weights_solo` skips every other region for one
  composition. Measured on the same bundle: soloing one of two `delete`
  spheres changes **3.77 %** of the frame against both being on.

Splat-side weight compositing (§2's "splat side" bullet) is also not in any
milestone above — see §7.

## 7. Risks

- **Two renderers' geometric conventions must agree, indefinitely.**
  `brush_pyramid`'s camera and `brush-render`'s camera are independently
  implemented (`docs/decisions/ADR-0006-viewer-integration.md` already notes
  the TRIPS side gained `distortion` explicitly because a free-flying camera
  cannot use the older per-view-baked convention). A region test done once
  in world space, against `xyz`, sidesteps most of this — but the splat-side
  weight compositing (§2's second bullet, not scheduled in any milestone
  above) requires `brush-render`'s rasteriser to accept a substituted-colour
  second pass, which has not been built or verified the way
  `surface.rs`'s depth-distortion term was. Treat that as unproven until an
  agreement test exists (mirroring `docs/ARCHITECTURE.md`'s own "implement
  transforms twice independently" discipline).
- **Region-test cost, measured.** The Regions panel reports the wall time of
  the last recomposition ("weights recomputed in N ms"), so the number
  `docs/EDITOR.md` used to ask for is now on screen during a session rather
  than estimated. On the 4 000-point synthetic bundle with two regions it is
  under a millisecond; the Karekare-scale figure is still to be taken from a
  real session, and the two `O(points x regions)` passes it covers (TRIPS
  points, and Gaussian centres only when a `delete` region is enabled) are the
  ones to watch. The Gaussian pass is skipped entirely unless a `delete`
  region exists, which is what keeps a mix-slider drag on a multi-million-
  Gaussian ply cheap.
- **Region-test cost still needs a real budget check.** §2's "once per edit
  change, not once per frame" design is much cheaper than a per-pixel SDF,
  but "once per edit change" on a multi-million-point Karekare scene with
  several active regions is not free either (`O(points × regions)`); E1's
  acceptance should include a timing number on the full scene, not only the
  synthetic/small-bundle case.
- **Segmentation quality (E5) is the least certain estimate.** SAM 3 lifting
  through majority vote across views is a reasonable default (it mirrors
  `make_masks3.py`'s own fallback-chain discipline), but occluded or
  thin/translucent objects (exactly the kind of thing a "smart selection"
  tool gets asked to handle) may need a manual clean-up step in the
  Inspector (add/remove points from a `pointset` region by hand) that is not
  separately budgeted above — if it is needed, it is a small addition to E1's
  Inspector (a "shift-click to add/remove from selection" mode over an
  existing `pointset` region), not a new milestone.
- **Privacy stays intact only if SAM 3 stays local.** `~/Splats/tools/sam3`
  is a local repo + weights today; nothing in this design calls a hosted
  segmentation API, and it must stay that way per `AGENTS.md` §6. The
  segmentation UI should make "this ran on your machine" visually obvious
  (e.g. no "processing..." spinner that could be mistaken for a network
  call) so a future change accidentally routing through a hosted API would
  be conspicuous, not silent.
- **`pointset` regions do not survive distillation, by construction.** §5
  documents the edit-then-distil ordering as the fix (`trippy distill
  --stage render --edits` applies `delete`-op regions, `pointset` included,
  before rendering), but a `pointset` edit made *after* a distilled publish
  already exists requires either accepting the box/sphere/lid-only re-apply
  path or re-running `trippy distill` from the edited TRIPS checkpoint.
  **Implemented**: `trippy apply-edits --target distilled/both` records
  `summary["distilled"]["warning"]` when an enabled `pointset` region
  exists alongside a `--distilled-ply` re-apply, so this is visible in the
  CLI's own JSON output rather than only discoverable by Jordan opening
  the result; a Publish-UI-level warning (surfacing the same condition
  before the command even runs) is still a viewer-side task, not built.
- **The checkpoint-side gate-suppression multiply is a simplification, not
  the full per-pixel override §2 describes** (see the new note after the
  status header, and §5's own paragraph) — it can only ever pull the gate
  towards TRIPS, never manufacture splat presence a `blend`/`fade` region
  asked for. Fixing this properly needs the splat-side weight compositing
  in the bullet above (`brush-render` accepting a substituted-colour second
  pass), still unbuilt/unverified.

## Related

- `docs/decisions/ADR-0007-viewer-editing.md` — the decision this document
  details, and the alternatives considered.
- `docs/decisions/ADR-0006-viewer-integration.md` — the viewer this builds
  on.
- `~/Splats/tools/SURFACE_LID.md` — read-only; the fitted pool-plane numbers
  and the training-time regulariser this is deliberately not reusing as an
  action.
- `docs/EXPERIMENTS.md` "Shade audit", "Point removal" — the audit rule and
  `shade_prune` heuristic behind the shade-cloud finder.
- `trippy/train/prune.py` — the exact functions the shade-cloud finder calls.
- `~/Splats/tools/sam3/repo`, `~/Splats/tools/make_masks3.py` — read-only;
  SAM 3 and the multi-signal mask-fallback discipline the SAM-3 lift reuses.
- `docs/GEOMETRY.md` — camera/world conventions both projection paths use.
- `docs/USER_GUIDE.md` — where a future "how to edit a scene" walkthrough
  belongs once E1 ships.
