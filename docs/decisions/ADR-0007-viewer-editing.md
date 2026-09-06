# ADR-0007: Viewer editing — trips-viewer grows an edit layer, not a new tool

Date: 2026-09-07 · Status: Accepted (architecture); implementation not started

## Context

Jordan, 2026-09-07 09:45 (`STATE.md`): a full editing GUI — per-object/per-region
splat-vs-TRIPS mix, smart object selection with masking, the Karekare pool lid
as an edit, and a way to find and get rid of floating shade clouds. Also
2026-09-07 09:30: the shade problem only shows up on the full Karekare set;
the *other* floating clouds already look better under TRIPS than under plain
Gaussians, and he expects the best result to be a TRIPS+splat combination —
which is what motivated `feat/blend-gate` (an explicit per-pixel gate channel
`g` from the U-Net, `final = g*splat + (1-g)*trips`, `gate_scale` at render
time; STATE.md 2026-09-07 09:30, not yet coded — `git diff main..feat/blend-gate`
touches only `STATE.md`). Editing has to sit on top of that gate, not replace
it: the gate is the network's own opinion of where TRIPS is trustworthy, and
region edits are Jordan's *override* of that opinion in specific places
(the pool, a cloud he has spotted, an object he wants to hide).

What `trips-viewer` already is (ADR-0006): a separate binary in trippy's own
Cargo workspace (`rust/crates/trips-viewer`), not a mode inside Brush's
`apps/brush-app`, reading a self-contained bundle directory
(`bundle.json` + `points.npz` + `weights.safetensors`, format
`trippy-bundle-1`, `src/bundle.rs::Bundle::load`). Per frame
(`src/renderer.rs::Renderer::render`) it uploads the point set once
(`gpu::UploadedPoints`, re-uploaded only when a precision lever changes,
`Renderer::resident_points`), rasterises it with `brush_pyramid`, and hands
the pyramid to `brush_unet::net::Unet` + `NeuralCamera` for tone mapping.
Three view modes already exist and share one render
(`renderer.rs::ViewMode`): `Network` (the displayed frame), `RawLevel0`
(the pyramid's finest level, pre-network — "photographed-ish evidence,
before invention"), and `Coverage` (`t_final`, "which pixels the rasteriser
actually covered"). Exposure is a per-image, per-frame policy
(`ExposureMode` + `Renderer::set_exposure`/`exposure_override`), not baked
into the render. The camera (`src/camera.rs::Controller`) works in world
space, row-major `R`, `x_cam = R@x_world + t`, `+Z` forward (`docs/GEOMETRY.md`),
and never touches the point cloud's own bounds for scale (the environment
sphere problem) — always `crate::bundle::SceneScale`, derived from capture
cameras.

None of that is an editor. There is no undo, no selection, no second
geometry (a plain-Gaussian splat) in the frame at all, and no notion of a
*region* — every pixel gets exactly the pipeline above, with no per-place
override. Building the editor means answering, precisely, three questions
this ADR settles:

1. Where does an edit live, so it survives sessions, composes with the
   in-flight gate work, and never risks the bundle format itself?
2. How does a per-pixel blend weight actually get computed, given what the
   two renderers currently expose (and what they do not)?
3. In what order do the selection tools ship, given that the riskiest one
   (SAM 3) is also the one that answers "smart object selection" most
   directly?

The full data model, render-integration design, tool order, UI sketch,
publish path and milestones are in `docs/EDITOR.md`; this ADR records the
decision and the two places the existing code makes the brief's assumptions
harder than they look from outside it.

## Decision

**Editing is a layer on top of `trips-viewer`: a sidecar `edits.json` next to
`bundle.json`, applied at render time as an override on the gate, and applied
at publish time by re-running the same region tests against whichever point
cloud is being exported.** No new binary, no new bundle format version, no
change to `Manifest`'s required fields (`src/bundle.rs::Bundle::parse_manifest`
does not use `deny_unknown_fields` and `edits.json` never enters `bundle.json`
at all, so every existing bundle, launcher and test is unaffected whether or
not an edit session exists next to it).

Four sub-decisions follow from that:

### 1. Regions carry a world-space *test*, not a per-pixel one

`docs/EDITOR.md` §2 works through this in full; the summary is that the brief's
framing — "region weights via depth lookup of both paths" — assumes a
per-pixel composited depth buffer that **does not exist in either renderer
today** (see Consequences). The design instead tests region membership once
per point, in world space, against `points.npz`'s own `xyz` (already resident
on the device as `gpu::UploadedPoints`, per `renderer.rs`'s own comment on why
that upload is cached rather than redone per frame). The result is a per-point
scalar that rides through each renderer's *existing* alpha-compositing exactly
the way colour already does — no new per-pixel machinery, and (for the splat
side) exactly the trick `crates/brush-train/src/surface.rs`'s depth-distortion
term already uses to get a composited per-pixel statistic out of an
unmodified rasteriser: substitute the value for the SH DC coefficient, render
a second time, read back a real alpha-composited quantity with zero new
kernels. This is cheaper than the brief's own suggested SDF/voxel texture (an
`O(points × regions)` cost paid once per edit change, not `O(pixels)` every
frame) and it sidesteps the depth gap entirely.

### 2. The lid is a region *kind*, reusing fitted, verified numbers — but it is not the training regulariser

`~/Splats/tools/SURFACE_LID.md` (read-only) documents a **training-time**
opacity penalty on `crates/brush-train/src/surface.rs`'s Karekare pool plane —
it makes floating haze fade out *during training*. Jordan's edit request is a
**view/publish-time hard clip**: nothing renders below the plane inside the
radius, full stop, no gradient, no fading. These are different operations
that happen to share one geometry (`lid_up`, `lid_height`, `lid_center`,
`lid_radius`, `lid_falloff`, `lid_band`) — the editor's `lid` region reuses
the *already-fitted and A/B-verified* numbers from `SURFACE_LID.md` §3
(`--lid-up 0.01290213 -0.95271846 -0.30358041 --lid-height -0.49
--lid-center -0.00683348 -0.74759695 3.95994345 --lid-radius 2.5
--lid-falloff 1.0 --lid-band 0.05`) as the region's *default* parameters for
the Karekare pool, editable in the Inspector, but the op is `delete`
(hard, `op: delete`) or `fade` (soft, ramps over `band`), never the training
penalty's opacity-only gradient trick. `docs/EDITOR.md` §1 has the exact
region schema.

### 3. Selection tools ship cheapest/most-certain first, SAM 3 last

Click-to-cluster needs nothing new (a ray through `camera.rs`'s existing
basis, nearest neighbours over the already-resident `xyz`). The shade-cloud
finder is a Rust-side consumer of numbers `trippy.train.prune` already
computes exactly once per audit (`build_shade_region`, `in_region`,
`luminance`, `dark_mass_stats` — a verified field-for-field port of
`~/Splats/tools/depthprior_shade_audit.py`, per `docs/EXPERIMENTS.md`
"Shade audit"); it needs those numbers precomputed and cached per point, not
re-derived in Rust, so the sliders in the brief ("thresholds as sliders")
threshold an already-computed `(inside, lum, conf)` triple rather than
re-running COLMAP projection live. SAM 3 (`~/Splats/tools/sam3/repo`, weights
already on disk, read-only) is the only tool here that touches a neural
network and a training photograph, and it ships last, after the cheaper tools
have already given Jordan a way to solve most of the same problem by hand.

### 4. Publish order is edit-TRIPS-first, then distil

`trippy distill` (`trippy/cli.py`'s `distill` subcommand, `--stage
render|brush-cmd|compare|all`) trains an **entirely new** Gaussian point
cloud with Brush from images rendered off the TRIPS checkpoint
(`trippy/distill/brush_runner.py`). That new cloud shares no point IDs with
`points.npz`. A `pointset` region (shade-cloud finder, SAM-3 lift,
click-to-cluster — anything selected by point ID) therefore **cannot** be
replayed against the distilled cloud after the fact; a `box`/`sphere`/`lid`
region can, because those are pure world-space geometry independent of which
renderer owns the points. The decision: apply pointset-region edits to the
TRIPS point cloud and its renders *before* `trippy distill --stage render`
runs, so the distilled Gaussians simply never see the deleted content in
their own training images. Box/sphere/lid regions may additionally be
re-applied directly to the distilled PLY at publish time, since their test
needs only the distilled cloud's own `xyz`.

## Consequences

### The depth buffer the brief assumed does not exist yet, on either renderer

- `brush_pyramid::output::LayerImage` (the pyramid's per-layer result) holds
  `feature` (channel-first `C×h×w`), `t_final` (`h×w` transmittance), and
  `n_used` (fragment count) — **no depth channel.** `Coverage` mode
  (`renderer.rs::ViewMode::Coverage`) binds `t_final`, not a depth buffer;
  the module doc is explicit that `t_final == 1.0` means "nothing was drawn
  here," which is coverage, not distance.
- `brush-render`'s public `RenderOutput<B>` (`rust/brush-trips/crates/
  brush-render/src/render_aux.rs`) exposes `out_img` and a `RenderAuxInner`
  with `num_visible`, `num_intersections`, `visible`, `max_radius` (a
  **per-splat**, not per-pixel, screen radius) and `tile_offsets` — also no
  composited per-pixel depth or alpha.
- The one place a per-pixel splat depth *does* exist today is
  `trippy/hybrid/render_splat_views.py` / `gsrender_live.py` (Design A's
  Gaussian-input block), which calls Splats' `gsrender.render` to produce
  `<stem>.depth.npy` + `<stem>.alpha.npy` per view — but that is an offline,
  Python/MPS path for training-time conditioning, not something the live
  Rust viewer can call per frame.

  This is why the render-integration design in `docs/EDITOR.md` §2 does not
  do "a depth lookup of both paths" as literally stated in the brief; it
  does the per-point-scalar-through-existing-compositing trick from §1
  above, which needs no depth buffer on either side. Flagging this as the
  one place the brief's own phrasing does not match what the code can do
  today, so nobody spends E1's budget building a depth buffer neither
  renderer has ever needed before.

### `edits.json` cannot touch `bundle.json`'s schema without care, and does not need to

`Manifest` deserializes with serde's default behaviour (no
`deny_unknown_fields` anywhere in `bundle.rs`), so an extra key would in fact
be silently ignored rather than rejected — but the module's own invariant
("anything unrecognised in the manifest is an error, not a warning") is about
`format`/`views` validation, and the safest way to honour its spirit is to
never touch `bundle.json` for this feature at all. A missing `edits.json`
means "no edits," identically to how a bundle written before the performance
levers existed loads as the exact pipeline (`PyramidParams`'s serde
defaults) — same pattern, applied one file down.

### The gate and region edits must compose, not race

`feat/blend-gate`'s `g` is a *learned*, per-pixel opinion; a region edit is a
*hand-set* override in a specific place. The render-integration design in
`docs/EDITOR.md` §2 makes region weight an override that composes with
`g * gate_scale` (region present → its `mix` wins in `blend` op, or forces
`0`/`1` in `delete`/hard-clip; region absent → the gate alone decides), rather
than a second, competing gate. This has to be decided now, before
`feat/blend-gate` lands its own viewer panel (split-screen / mix slider,
STATE.md 2026-09-07 09:30), so the two panels do not fight over the same
per-pixel weight.

## Alternatives considered

**A separate editing application**, built independently of `trips-viewer`.
Rejected for the same four reasons ADR-0006 rejected moving the viewer into
`apps/brush-app`: it would re-pay the wgpu/egui/Burn scaffolding cost ADR-0006
already paid once, it would need its own bundle loader kept in sync with
`bundle.rs`, and "walk it in the viewer" (`AGENTS.md` §7, "Jordan's viewer
verdict is the verdict") is exactly the workflow the existing viewer already
serves — editing is a mode of looking at a scene, not a different scene.

**All editing offline against exported PLYs, no live preview.** Rejected:
Jordan needs to see the splat/TRIPS mix change *as he adjusts it*, the same
reason the shade audit is not trusted over "Jordan's own eyes" (`AGENTS.md`
§7). A slider with no live render answers a different, less useful question.

**Folding this into training-time point removal** (`trippy/train/prune.py`'s
`shade_prune`). Rejected: `shade_prune` already exists and is a *training*
heuristic that changes what the network is optimised against, evaluated at
fixed epochs, reported next to held-out shade PSNR
(`docs/EXPERIMENTS.md` "Point removal"). What Jordan asked for is post-hoc,
interactive, reversible editing of an already-trained bundle — undo, save,
publish — which is a different lifecycle stage and a different audience (one
person looking at one scene, not a training run's own schedule). The
shade-cloud *finder* tool reuses `prune.py`'s exact rule as a **selector**,
deliberately, but the action it feeds (delete/fade/blend, undoable) is new.

## Related

- `docs/EDITOR.md` — the full data model, render integration, selection tool
  order, UI sketch, publish path and milestones this ADR summarises.
- ADR-0006 — the viewer this builds on; the separate-binary decision and its
  costs, which this ADR inherits rather than re-litigates.
- `~/Splats/tools/SURFACE_LID.md` — the fitted, A/B-verified pool-plane
  numbers the `lid` region's defaults come from (read-only; not modified by
  this work).
- `docs/EXPERIMENTS.md` "Shade audit", "Point removal" — the audit rule and
  `shade_prune` heuristic the shade-cloud finder reuses as a selector.
- `trippy/train/prune.py` — `build_shade_region`/`in_region`/`luminance`/
  `dark_mass_stats`, the exact functions the shade-cloud finder is built on.
- `STATE.md` 2026-09-07 entries — Jordan's request and the `feat/blend-gate`
  context this design has to compose with.
