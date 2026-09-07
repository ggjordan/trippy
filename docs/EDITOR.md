# Editing: turning `trips-viewer` into an edit tool

Status: **Python side implemented, including the publish path and
click-to-cluster** (`trippy/edit/`: `model.py`, `weights.py`,
`shade_finder.py`, `apply.py`, `checkpoint.py`, `cluster.py`; `trippy
apply-edits --target trips|distilled|both`, `trippy edits
shade-find/add-box/add-sphere/add-lid/click`, and `--edits` on `trippy
candidate-report`/`trippy eval`/`trippy distill --stage render` in
`trippy/cli.py`; see `docs/EXPERIMENTS.md` "Edits" for the worked run and
test list). The Rust viewer (§0/§2's render integration, §3's selection
tools, §4's UI) is **not implemented** -- this document remains the spec
for that work. Three implementation notes, all explained where they matter
below:
Status: **E1 and E2 shipped on both sides** (2026-09-07).

- Python: `trippy/edit/` (`model.py`, `weights.py`, `shade_finder.py`,
  `apply.py`, and `golden.py`, the parity fixture), `trippy apply-edits`
  and `trippy edits shade-find/add-box/add-sphere/add-lid`.
- Rust viewer: `rust/crates/trips-viewer/src/edit/` (the platform-neutral
  twin: `model.rs`, `weights.rs`, `apply.rs`, `shade.rs`) plus
  `src/edit_ui.rs` (the Regions/Inspector/Tools panels, `M`) and the render
  integration in `renderer.rs`/`splat.rs`.
- The two sides are pinned together by `tests/fixtures/synthetic/edit_golden/`
  — written by `trippy.edit.golden`, read by BOTH
  `tests/test_edit_golden.py` and the Rust `edit::golden` tests, which agree
  to 1e-6. `trips-viewer --dump-weights` / `--dump-shade` run the same
  comparison against a real bundle
  (`tests/test_edit_viewer_parity.py`).

Not built: the 3D drag gizmos (E1 shipped Inspector fields + keyboard
nudge/resize instead), click-to-cluster (E4), the SAM-3 lift (E5), and the
`brush`-kind voxel region.

Three implementation notes, all explained where they matter below:
tools' UI/ray-cast half, §4's UI) is **not implemented** -- this document
remains the spec for that work. Four implementation notes, all explained
where they matter below:

- **`box`'s schema is `center`/`half_extents`/`quat`** (an oriented box),
  not this section's axis-aligned `min`/`max` -- a rotated box is testable
  where an axis-aligned-only schema is not; identity `quat` recovers an
  axis-aligned box exactly. See §1's `box` entry.
- **The Python `lid` region kind is E3's hard-clip action already**
  (`op="delete"`/`"fade"`, `trippy.edit.model.lid_membership`); its 3D
  gizmo (drag the plane/radius in the viewer) is not built. See §1's `lid`
  entry and §6's E3 row.
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

`docs/decisions/ADR-0007-viewer-editing.md` is still accurate to the
sections below; this document is the detailed spec the milestones in §6
implement, in order.

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
  Click-to-cluster needs a ray from a clicked pixel; that ray is
  `Controller`'s own forward/right/down basis plus the pixel's NDC offset —
  no new camera code, just a new method next to `render_camera`.
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
  bounding volume). Painted incrementally by the brush tool (§3); not one of
  E1's two shipped kinds (box/sphere) — tracked for later.
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
  gizmo (drag the plane/radius in the viewer) is not built.
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
point-density heatmap PNG of the selection (no photo content). The
viewer's own ray cast against a live camera and the Inspector's
size/tightness slider are still Rust-side, unbuilt work — see E4's own row
in §6.

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

Not really a "selection tool" — a fixed-shape region with a plane/radius/
falloff gizmo in the 3D view (drag the plane, drag the radius ring), seeded
from `SURFACE_LID.md`'s numbers when the scene is Karekare. See §1/§2.

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

## 4. UI sketch

Three egui panels, added the same way the existing HUD window is built
(`app.rs::ViewerApp::overlay`, an `egui::Window`) — an "Edit" window shown
alongside it, toggled independently of the existing Tab-toggled HUD so
Jordan can hide edit chrome while still flying around:

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
┌─ Tools ─────────────────────────┐  │ [preview] [select] [delete]  │
│ ( ) click-to-cluster            │  └────────────────────────────────┘
│ ( ) shade-cloud finder           │
│     lum <  [0.25]                │
│     conf < [0.50]                │
│     inside shade region  [x]     │
│ ( ) SAM-3 lift (pick a view...)  │
│ ( ) lid gizmo (drag in 3D view)  │
└───────────────────────────────────┘
```

**Keys** (chosen to avoid every key `app.rs::ViewerApp::handle_input`
already binds — `V X Tab - = F R N P W A S D Q E` and drag/scroll):

**As shipped** (E1/E2). Rows marked *not built* wait on E3/E4.

| key | action |
|---|---|
| `M` | toggle edit mode (shows Regions/Inspector/Tools). `--edit` opens straight into it. Click-drag on the canvas still orbits: with no gizmos to hit, swapping the drag would only take navigation away |
| `T` | cycle the active tool (Regions ↔ shade-cloud finder) |
| arrows, `PageUp`/`PageDown` | nudge the selected region along world X/Z and Y by `NUDGE_SCENE_FRACTION` of the scene diameter — **E1's replacement for the 3D drag gizmo** |
| `[` / `]` | shrink/grow the selected region by `RESIZE_STEP` |
| `Delete` / `Backspace` | remove the selected region (its `op` is a separate field in the Inspector) |
| `Cmd`/`Ctrl` + `Z` | undo |
| `Cmd`/`Ctrl` + `Shift` + `Z` | redo |
| `Cmd`/`Ctrl` + `S` | save `edits.json` |
| `H` | toggle the preview highlight (a tinted point cloud, not a new `ViewMode` — see §6) |
| `Cmd`/`Ctrl` + `N` | *not built*: "new region from the current selection" belongs with click-to-cluster (E4). The shade finder's own "add as region" buttons are E2's version of it |

Left-click behaviour while in edit mode: **unchanged from viewing** in E1/E2 —
there is no gizmo to hit yet, so the canvas keeps orbiting and every region is
placed at the camera's look-at point and shaped from the Inspector or the
keyboard. When the gizmos land (E3/E4) the rule is: on empty space with
click-to-cluster active, seed a new cluster at the ray hit; on an existing
region's gizmo (lid plane/ring, box corner, sphere radius handle), drag that
handle instead — the same "response is scoped to what the drag started on"
discipline `app.rs`'s own module doc already calls out as the fix for the
`egui_wants_pointer_input` trap (`dragged_by`, not a global "is anything
active" check).

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
This gate-suppression step only runs inside `trippy.render.candidate.
render_candidate` (`candidate-report`, and by extension `trippy distill
--stage render`'s own per-pose renders); `trippy eval`'s render path
(`Trainer.evaluate`) applies the keep-mask deletion but not the gate
multiply — a documented gap (`trippy.edit.checkpoint`'s own module
docstring), not a silent one.

`apply-edits` and `--edits` are additive to the pipeline order that already
exists (train → export/distill), not a new stage inserted into training —
nothing here changes `Trainer.fit`'s loop or `maybe_prune_points`'s
schedule.

## 6. Milestones

| # | Scope | Estimate | Acceptance | Status |
|---|---|---|---|---|
| **E1** | Region data model (`edits.json` read/write), `box`/`sphere` region kinds, per-point weight compositing (§2's non-depth design) on the TRIPS side only, undo/redo, save/load surviving a bundle close+reopen | 1 wk | Draw a box or sphere region in the viewer, set `mix`, see the TRIPS render change inside it and nothing outside it; undo removes the region; closing and reopening the bundle restores it exactly (region list, mix values, undo history) | **Done** (2026-09-07), except the 3D drag gizmos, which E1 replaced with Inspector fields + arrow-key nudge / `[`-`]` resize. `rust/crates/trips-viewer/src/edit/*` + `src/edit_ui.rs`. Measured on the synthetic bundle: a `blend` sphere at `mix = 0` changes 4.74 % of the pixels in its own half of the frame and **0.00 %** of the other half; a `delete` box changes 63.97 % of its half; undoing it reproduces the unedited frame **bit for bit** (max channel diff 0). |
| **E2** | Shade-cloud finder: `trippy edit-prep` precompute + live threshold sliders + preview-highlight `ViewMode` + "select" → `pointset` region | 3 d | On a bundle whose scene has registered shade frames, the finder's default thresholds select a `pointset` region whose `dark_mass_fraction` (via `trippy.train.prune.dark_mass_stats` on the selected IDs) matches the audit's own number for that scene to float precision; deleting the region and re-running the audit shows the drop | **Done** (2026-09-07). Four live sliders (lum, conf, znear/zfar fractions) over `shade_views.json`, the precompute written by `trippy.edit.golden.write_shade_views`; "add as region (fade / delete)"; and a preview that **tints** the selection instead of adding a `ViewMode` (see the note under this table). `trips-viewer --dump-shade` selects the same ids and the same `dark_mass_fraction` as `trippy.edit.shade_finder.find_shade_pointset` on the synthetic bundle (`tests/test_edit_viewer_parity.py`). |
| **E3** | `lid` region kind + 3D gizmo (plane/radius drag) + hard-clip delete semantics | 2 d | Loading the Karekare pool bundle with a `lid` region seeded from `SURFACE_LID.md`'s numbers removes the haze from every angle at every `mix`/exposure; dragging the radius ring changes the affected point count live | **Region kind + hard-clip semantics done** (`trippy.edit.model.lid_membership`, `trippy edits add-lid`); the 3D gizmo is not built. |
| **E4** | Click-to-cluster: ray cast + k-d tree + k-NN growth in world+colour space, radius slider | 3 d | Clicking a point-cloud cluster selects a `pointset` region that visibly matches the clicked object's extent, without needing a depth buffer | **Python side done** (`trippy.edit.cluster`, `trippy edits click`; synthetic-scene tests: an isolated blob is selected and a same-coloured blob behind it through a depth gap is not, the colour gate separates touching differently-coloured clusters, `max_radius`/`max_points` caps are respected, `tests/test_edit_cluster.py`); the viewer's ray cast against a live camera, the Inspector's size/tightness slider, and the "click on empty space seeds a new cluster" UI wiring (§4) are not built. |
| **E5** | SAM 3 lift: photo segmentation, multi-view projection + majority vote, `pointset` region output | 2–3 wk | Segmenting an object in 2–3 registered views of the same scene produces one `pointset` region that, previewed, highlights that object and not its neighbours; runs entirely local (no image leaves the machine) | Not started. |
| **E6** | Publish path: `trippy apply-edits`, TRIPS `export.ply` mask wiring, distilled-splat publish order (edit-then-distil for pointset regions, geometry-reapply for box/sphere/lid) | 3 d | `trippy apply-edits --target both` on a bundle with a mix of region kinds produces a TRIPS PLY with the deleted points absent and, after a `trippy distill` run on the same edited bundle, a distilled PLY that also lacks them; a box/sphere/lid region re-applied directly to an already-distilled PLY (no re-distillation) also removes the matching geometry | **Done** (`trippy/edit/apply.py`, `trippy/edit/checkpoint.py`): `--target trips\|distilled\|both` (default `both`); `trips`/`both` write the filtered TRIPS `points.npz`/`blend_weights.npy`/`export.ply` and, if named, a filtered `blend.splat_ply`; `distilled`/`both` (with `--distilled-ply`) re-apply box/sphere/lid regions to an already-distilled PLY as delete/opacity-scale-fade; `--edits` wired into `trippy distill --stage render` (edit-then-distil ordering) and into `trippy candidate-report`/`trippy eval` (checkpoint-side keep mask + gate suppression on gate-hybrid checkpoints, `candidate-report` only — see §5's own paragraph for the `eval` gap). |

**Two deviations worth naming.** (1) The preview highlight is not a fourth
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
plumbing. `brush`-kind (voxel) regions and splat-side weight compositing
(§2's "splat side" bullet) are not in any milestone above — see §7.

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
