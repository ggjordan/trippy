# ADR-0008: SuperSplat Editor 3.0 — self-host it as the splat editor, keep trippy's editor for what only trippy can do

Date: 2026-09-09 · Status: Accepted (decision); implementation not started

## Context

Jordan's verdicts of 2026-09-08 changed what trippy's editor is *for*:

- 19:10 — "TRIPS looks nothing like a photo anywhere except that the clouds/fogs
  are gone; **splat is the base**." TRIPS was demoted from *the renderer* to
  *a classifier over Jordan's own splat*.
- 19:40 — new track: TRIPS-confidence-guided cleaning of `kklid_20000`, producing
  **plain PLYs for a splat editor** (`kklid-tripsclean-shade/-005/-015`, delivered
  2026-09-08 21:10, `STATE.md`).
- Earlier the same day, driving the finished editor on the full Karekare scene:
  "I don't get how to use the editor at all, **needs to be more simple**", plus
  "make it left click and drag to orbit, right click and drag to POV free move,
  **like Brush**".

So the thing Jordan actually does now is: open a cleaned PLY, look at it, delete
what is left of the fog, and want a result he can show. That is a **plain 3DGS
editing job**, and `docs/EDITOR.md`'s milestone table (E1–E6, all Done) built a
plain-3DGS editing UI as a side effect of building the TRIPS edit layer.

Meanwhile PlayCanvas shipped **SuperSplat Editor 3.0** (2026, rebuilt on WebGPU),
which is a mature, MIT-licensed, browser-based 3DGS editor with roughly ten tools
we do not have and will not out-build. This ADR decides what trippy adopts,
reuses, or **drops in its favour**.

The hard constraint is `AGENTS.md` §6: Jordan's scenes are family photographs and
must never leave this machine. A hosted web editor is acceptable **only** if it
provably processes files client-side, and self-hosting on `127.0.0.1` is the
preferred answer. §3 below is the evidence for that verdict.

### Sources read (2026-09-09)

- Blog: "New in SuperSplat Editor 3.0 — rebuilt on WebGPU"
  (`blog.playcanvas.com/new-in-supersplat-editor-3-0-rebuilt-on-webgpu`).
- Repo `github.com/playcanvas/supersplat` at `main` (version **3.0.0**): the file
  tree, `LICENSE`, `README.md`, `package.json`, `src/index.html`, `src/index.ts`,
  `src/main.ts`, `src/publish.ts`, `src/editor.ts`, `src/edit-ops.ts`, `src/doc.ts`,
  `src/splat-serialize.ts`, `src/data-processor/index.ts`,
  `src/tools/sphere-brush-selection.ts`, `src/ui/menu.ts`.
- Docs: `developer.playcanvas.com/user-manual/supersplat/` (index, viewer,
  viewer/self-hosting, editor/import-export, editor/timeline).
- `github.com/playcanvas/supersplat-viewer` (MIT, npm `@playcanvas/supersplat-viewer`).

No SuperSplat code was run and no local file was sent anywhere while writing this.

---

## 1. Feature by feature: SuperSplat 3.0 vs trippy

Legend: **Y** = has it and it is good; **y** = has it, weaker; **—** = does not have it.

### 1a. Things that operate on plain Gaussians (their home ground)

| Capability | SuperSplat 3.0 | trippy viewer / editor | Note |
|---|---|---|---|
| Rect / box selection | **Y** (`select.rect`, `select.byBox`, GPU) | y (`box` region, arm-then-click) | ours is a *region* (persistent, mixable), theirs is a *selection* (transient) |
| Sphere selection | **Y** (`select.bySphere`) | y (`sphere` region) | |
| Lasso / polygon / freehand mask | **Y** (`select.byMask` from a canvas mask; press-and-hold tools in 3.0) | — | we have no screen-space lasso at all |
| 3D sphere **brush** (paint in space) | **Y** (`Shift+B`; depth-picks each stroke sample to world space, capsule path, radius matches brush size at the painted surface, auto-splits across depth discontinuities) | **Y** (`edit::brush`, `ScreenGrid` anchor, `clamp_stroke_depth`, one undo per stroke) | **we independently arrived at the same design**; ours is 0.011 ms/sample on 7.5M points |
| Flood fill selection | **Y** (`flood-selection.ts`) | — | |
| Eyedropper / select-by-colour | **Y** (`select.colorMatch`: depth-pick a pixel, then GPU per-channel colour distance under a 0–1 threshold) | y (click-to-cluster has a colour gate, but seeded by cluster growth not by colour alone) | |
| Select by **opacity / scale / value range** (histogram + range drag) | **Y** (`data-processor/select-by-range.ts`, GPU histogram, Data panel) | — | this is the single most useful thing we lack for fog cleanup |
| Selection **depth** toggle (frontmost only vs through all layers) | **Y** (`selection.useDepth`, per-pixel ID pick, `N`) | y (brush anchors at surface depth; no global toggle) | |
| Selection **footprint** toggle (centres vs full projected ellipse) | **Y** (`selection.footprint`, `M`, `footprint-intersect-shader`) | — (we test centres only) | matters for big/blurry fog Gaussians |
| Set ops on selection (add/remove/set/intersect), lock/unlock, invert, all/none | **Y** (`SelectOp` with 4 modes; `HideSelectionOp`, `UnhideAllOp`) | y (regions compose by weight; no explicit intersect/lock) | |
| Delete / hide / duplicate / **separate into a new layer** | **Y** (`RemoveInstancesOp`, `Duplicate`, `Separate`) | y (`delete`/`fade` ops; no layer split) | |
| Multiple splat files open as **layers**, with show / hide / **solo** | **Y** (scene panel `splat-list`, `shown/hidden/solo` icons) | y (Named Objects panel has enable/solo *per region*, not per file) | key for the fog-mask workflow, §4 |
| Colour grading on a selection (tint, temperature, saturation, brightness, black/white point, transparency) with live preview + Apply/Reset | **Y** (3.0 moved Colors to Scene Manager; `SplatsColorOp`, palette-indexed) | — | |
| Transform gizmos: move / rotate / scale, per-entity and per-selection | **Y** (`move/rotate/scale-tool`, `shape-transform-gizmo`, `transform-palette`) | y ("move arrows": translate, Shift-resize, Ctrl-rotate a box, plus a lid-normal handle) | ours transform *regions*, not the geometry |
| **Orient** tool (pick points to set the ground plane / local frame) | **Y** (GPU point picking, ~8.5 s → <25 ms in 3.0; `SetLocalFrameOp`) | — | |
| **Measure** tool | **Y** | — | |
| Undo / redo | **Y** (`edit-history.ts`; ops store index ranges + removed rows; `MultiOp` batches) | **Y** (`EditDocument` undo log, `update_region_coalesced`, one entry per stroke/drag; **verified bit-for-bit reversible**) | comparable; ours is the more strongly tested of the two |
| Save / reopen a project | **Y** (`.ssproj`: a ZIP of `document.json` v1 + shared PLY resources + `instances_*.bin` row-index/flag blobs; deleted rows are permanent at save) | **Y** (`edits.json` sidecar; regions are re-testable, nothing is baked) | **ours is the more editable representation** — see §2 |
| Import formats | **Y** (`.ply`, `.compressed.ply`, `.sog`, `meta.json`, `lod-meta.json`, `.splat`, `.ksplat`, `.spz`, `.lcc/.lcc2`, COLMAP `images.txt`, INRIA camera JSON) | y (`trippy-bundle-1` + `blend.splat_ply`; PLY via `brush-serde`) | |
| Export formats | **Y** (PLY, compressed PLY, **SOG**, SPZ, `.splat`, self-contained `.html` viewer, `.zip` viewer package, `.ssproj`) — streamed chunk-by-chunk | y (PLY via `trippy apply-edits` / `trippy distill`) | |
| Camera **timeline / keyframes / spline flythrough** | **Y** (`timeline-panel`, `anim/spline`, `camera-poses`; COLMAP + INRIA pose import) | y (`candidate-report` dolly, offline) | ours is a batch renderer, not an editable path |
| **Video export** | **Y** (Render menu → video settings; `mediabunny` = WebCodecs, client-side; resolution, bitrate, vertical) | y (offline frames → ffmpeg) | |
| Image / equirect render export | **Y** (`png-writer`, `equirect-renderer`) | y | |
| Web viewer for a finished splat | **Y** (`@playcanvas/supersplat-viewer`, MIT, self-hostable; orbit/pan/zoom, annotations, camera animation, post FX, optional walk/collision) | y (`trips-web`, WebGPU, our own TRIPS forward pass) | different jobs, see §2 |
| **VR / WebXR** | **Y** (viewer supports WebXR on capable headsets) | — (`docs/QUEST.md`: TRIPS on a Quest is out of reach this hardware generation) | **this is the answer QUEST.md was missing** |
| Renderer | WebGPU compute: projector, frustum cull, compaction, **GPU radix sort**, indirect draw; stochastic-alpha OIT mode; Chrome/Edge/Firefox/**Safari 26+** | wgpu native + `trips-web` wasm/WebGPU; our radix sort needs subgroups, so **Safari cannot run it** | their engine ships a *portable 4-bit* radix sort alongside a OneSweep path — a concrete lead for our Safari gap |
| Memory on a 4.4M-splat scene | 105 MB JS heap idle (was 1,557 MB pre-3.0) | n/a (native) | |
| Mobile / touch | requires WebGPU; no touch story documented | — | do not rely on it |
| Localization (i18n) | **Y** (i18next, `?lng=`) | — | |
| Installable PWA + OS file handlers (`.ply .splat .sog .spz .ksplat .ssproj`) | **Y** | n/a | |

### 1b. Things that operate on TRIPS (our home ground)

| Capability | SuperSplat 3.0 | trippy |
|---|---|---|
| Render a TRIPS point cloud (pyramid + U-Net + neural camera) | — (it cannot; it only knows Gaussians) | **Y** |
| Per-region **splat-vs-TRIPS mix** slider, live | — | **Y** (`blend`/`fade` ops, gate composition, combined bundle at 0.87 px median alignment) |
| Honesty views: `Network` / `RawLevel0` / `Coverage` (photographed vs inferred) | — | **Y** (`renderer.rs::ViewMode`) — required by `AGENTS.md` §7's honesty rule |
| TRIPS-confidence-guided **fog/cloud deletion** (the trained twin's own disbelief used as a classifier over Jordan's splat) | — | **Y** (`feat/splat-clean`; 1:1 mapping proven, max \|diff\| 0.0 on 7,542,137 points) |
| **Shade-cloud finder** (audit-rule selector with live luminance/confidence/depth sliders) | — | **Y** (`edit/shade.rs`, parity-pinned to `trippy.train.prune`) |
| **SAM 3 object lift** (segment in 2–3 registered photos, majority-vote to a point set), fully local | — | **Y** (`trippy edits sam`, MPS ~8.0 s/view) |
| Click-to-cluster on the TRIPS cloud | — | **Y** (exact Python/Rust parity, id for id) |
| Region membership as a **re-testable world-space predicate** that survives re-export and re-distillation | — (selections bake into row lists at save) | **Y** (`edits.json`; box/sphere/lid replay against *any* cloud) |
| Edit-then-distil publish ordering, checkpoint-side gate suppression | — | **Y** (`trippy apply-edits --target trips\|distilled\|both`) |
| Two-language parity discipline (numpy/torch, Python/Rust golden fixtures) | — | **Y** (`AGENTS.md` §7) |

---

## 2. What they have solved that we are still building — and what they cannot do

### They have solved (stop building these)

1. **Selection UX on plain Gaussians.** Lasso, polygon, flood, eyedropper,
   select-by-value-range, footprint-vs-centre testing, depth-limited picking, and
   four selection set-ops. All GPU, all producing a per-splat byte mask
   (`DataProcessor.intersect` → `Uint8Array`). This is ~10 tools; we have 4, and
   Jordan already said ours is too complex. Building the missing 6 well is
   **10–15 agent-days** we would spend to arrive behind where they already are.
2. **Transform maturity.** Move/rotate/scale gizmos, an Orient tool that sets the
   ground plane from picked points in <25 ms on 2.4M Gaussians, a Measure tool.
3. **Publish/export formats.** SOG, SPZ, compressed PLY, `.splat`, streamed
   chunk-by-chunk so a big scene does not blow the heap. SOG is the format that
   makes a Karekare splat *deliverable*; we have no compressed format at all.
4. **A WebGPU splat web viewer that is not ours to maintain.** MIT, npm-pinnable,
   self-hostable, exports as one `.html` or a 5-file ZIP.
5. **Camera paths and video export** — an editable spline timeline with keyframes
   and COLMAP/INRIA pose import, rendering to video locally via WebCodecs. Our
   dolly machinery is a batch renderer with no interactive path editing.
6. **VR.** Their viewer does WebXR. `docs/QUEST.md` (2026-09-06) concluded TRIPS
   on a Quest is out of reach and listed "distilled/cleaned Gaussians through
   Splats' publish path" as what ships instead — SuperSplat's viewer is a
   *better, MIT, self-hostable* version of that answer, and it is already built.
7. **A subgroup-free GPU radix sort.** SuperSplat 3.0 runs on Safari 26+; the
   PlayCanvas engine ships both a portable 4-bit radix sort and a OneSweep path.
   `docs/QUEST.md` and `docs/WEB_VIEWER.md` both record that *our* sort needs
   WebGPU subgroups and therefore cannot run in Safari. Their portable sort is
   MIT-licensed prior art for that exact problem.

### They cannot do (this is our actual value, and none of it is at risk)

1. **Render TRIPS at all.** SuperSplat is a Gaussian editor end to end
   (`gaussian-instances.ts`, `projected-splat-renderer.ts`). Nothing in it can
   consume `points.npz` + `weights.safetensors`.
2. **Per-region splat-vs-TRIPS mix.** The entire point of the combined bundle —
   splat everywhere, TRIPS only under the big tree — has no analogue there.
   SuperSplat has layers with visibility; it has no notion of two *renderers*
   blended per region.
3. **TRIPS-guided fog deletion.** The classifier that produced
   `kklid-tripsclean-*.ply` is a trained TRIPS twin's per-point confidence. That
   is a trippy-only artefact. SuperSplat can *act* on the result; it can never
   *compute* it.
4. **SAM 3 object lift** multi-view-consistent to a point set, locally.
5. **The shade-cloud finder** — a selector built from the same audit rule the
   verdicts are measured with.
6. **The honesty views.** `RawLevel0` and `Coverage` keep photographed and
   inferred pixels distinguishable (`AGENTS.md` §7). SuperSplat has no such
   concept, because for a plain splat there is nothing to distinguish.
7. **A re-testable edit representation.** `edits.json` stores *predicates*;
   `.ssproj` stores *row lists* and makes deletions permanent at save. Ours
   survives re-export, re-training and re-distillation; theirs does not need to.

---

## 3. Privacy verdict

**Verdict: self-hosting SuperSplat on `127.0.0.1` is SAFE and is the only
sanctioned way to use it with Jordan's scenes. The hosted editor at
`superspl.at/editor` is NOT sanctioned** — not because it uploads, but because a
Publish button that uploads `scene.ply` sits one click away in the same File menu
as Export. `AGENTS.md` §6 is a bright line; we do not put a bright line one
misclick from a family scene.

### Evidence for "no telemetry, no phone-home"

| Question | Finding | Where |
|---|---|---|
| Analytics/telemetry libraries? | **None.** No Google Analytics, gtag, Plausible, Sentry, PostHog, or error-reporting package appears in `dependencies`/`devDependencies`. | `package.json` (v3.0.0) |
| Analytics in the page shell? | **None.** The only `<script>` is `<script type="module" src="./index.js">`; no CDN, no external fonts, no remote meta. The one inline script *unregisters* stale service workers. | `src/index.html` |
| Network calls at startup? | **None to any external host.** `src/index.ts` has no fetch and no external URL. `src/main.ts` has exactly two remote-ish paths, both same-origin or user-initiated: `WebPCodec.wasmUrl = new URL('static/lib/webp/webp.wasm', document.baseURI)` (same origin), and `?load=<url>` / PWA `launchQueue` file handling (only if *you* pass a URL or open a file). | `src/index.ts`, `src/main.ts` |
| Does it contact PlayCanvas servers? | **Only through the Publish feature, and only on demand.** `origin` is `location.origin` (i.e. `http://127.0.0.1:<port>` when self-hosted, *not* a hardcoded PlayCanvas domain); `user.apiServer` is not hardcoded either — it comes back from `${origin}/api/id`, which is fetched inside the `publish.userStatus` handler, **not at startup**. Self-hosted, that request 404s against your own static server and publishing is simply unavailable. | `src/publish.ts` |
| What would Publish upload? | `scene.ply`, in 10 MB chunks, via `start-upload` → `signed-urls` → `complete-upload` → `splats/publish`, plus title/description/listed/animation metadata. **This is the one thing that must never be clicked.** | `src/publish.ts` |
| Is editing/exporting client-side? | **Yes, entirely.** Export runs through `@playcanvas/splat-transform` in-page with `MemoryFileSystem`/`ZipFileSystem`; SOG compression uses the local WebGPU device. Video export uses `mediabunny` (WebCodecs) in-page. | `src/splat-serialize.ts`, `package.json` |
| Other outbound links? | Menu → Help contains plain `<a>` links (YouTube tutorials, Discord, forums, GitHub, docs). Links, not fetches; nothing is sent. | `src/ui/menu.ts` |
| Locale loading | `i18next-http-backend` fetches locale JSON — from the same origin (`static/locales`). | `package.json`, `README` (`?lng=<locale>`) |

### Honesty about the limits of this audit

This is an entry-point audit (page shell, both entry modules, publish module,
dependency list, menu), not a line-by-line review of ~150 source files. Two cheap
empirical checks close the gap, and both are in §5's task list:

- **The airplane-mode test.** Build `dist/`, serve on `127.0.0.1`, pull the
  network, then load a scene, edit it and export. If it completes, nothing in the
  loop needs the internet. *(2 minutes of Jordan's time, or an agent's.)*
- **The DevTools Network-tab test.** Same session with the Network tab open and
  "preserve log" on; the expected result is same-origin requests only.

Two operational guards go with it: build once **with** network (`npm ci` pulls
packages), then run offline; and pin a tag rather than tracking `main`, so a
future upstream change cannot silently add a beacon to a scene-loaded editor.

### Self-hosting is well-supported, not a hack

`README.md`: `git clone` → `npm install` → `npm run develop` → `http://localhost:3000`.
`package.json` has `"build": "rollup -c"` and `"serve": "serve dist -C"`. The
output is a static site. PlayCanvas additionally *document* self-hosting for the
viewer (`developer.playcanvas.com/user-manual/supersplat/viewer/self-hosting/`):
a single `.html` (SOG base64-embedded, works from `file://` under ~32 MB) or a
5-file ZIP (`index.html`, `index.js`, `index.css`, `settings.json`, `index.sog`)
served over HTTP. That is exactly the shape `scripts/deliver.sh` already handles —
it generates an `OPEN_<NAME>.command` that runs `python3 http.server` **bound to
127.0.0.1 only** for any delivered directory containing an `index.html`.

### Licence compatibility

| Component | Licence | Compatible with trippy (MIT)? |
|---|---|---|
| `playcanvas/supersplat` 3.0.0 | **MIT**, "Copyright (c) 2011-2026 PlayCanvas Ltd." | **Yes** |
| `playcanvas/supersplat-viewer` | **MIT** | **Yes** |
| PlayCanvas engine 2.22.0 | **MIT** | **Yes** |
| `@playcanvas/splat-transform` 3.3.3 | verify its own `LICENSE` at pin time (PlayCanvas ships MIT) | expected yes |
| our brush fork | **Apache-2.0** | unaffected — no SuperSplat code goes near it |

MIT → MIT is the easy direction. Two rules: (1) **do not vendor** SuperSplat into
this repo — clone at a pinned tag into a gitignored directory, so the public repo
stays ours and there is no second copy to keep in sync; (2) if we ever *port* a
file (e.g. the portable radix sort), copy the MIT notice with it and say so in the
file header, per `AGENTS.md` §4's header rule.

---

## 4. Decision

Three options were considered.

**Option A — Self-host SuperSplat as THE splat editor.** Pin and serve it locally;
trippy exports splat-only artefacts into it (cleaned PLY + a TRIPS fog mask as a
second layer); we stop developing plain-Gaussian editing UX entirely.
*Effort: ~2 agent-days. Risk: low.*

**Option B — Port specific UX pieces into our viewer** (lasso, select-by-range
histogram, footprint testing, colour grading, an editable camera timeline).
*Effort: ~12–18 agent-days for a subset that would still be behind theirs. Risk:
medium-high — it re-runs the "too complex" failure mode.*

**Option C — Both, in stages.** A now; a *narrow, evidence-led* slice of B later,
chosen from what Jordan actually reaches for after living with A.

### Pick: **Option C, staged, starting with A in full.**

Stage 1 (A) is where all the value is and it is two days. Stage 2 (B) is
deliberately deferred until Jordan has used A, so that the port list comes from
observed use rather than from a feature table. Stage 3 is the Quest/delivery
payoff (SOG + self-hosted viewer), which is A's export path, not new UI.

Rationale in one paragraph: after 2026-09-08, trippy's job is **to produce the
cleaned splat and the fog mask**, and Jordan's job is **to look at it and finish
the cleanup**. SuperSplat is a better tool for the second job than anything we
will build, it is MIT, it is provably local when self-hosted, and it removes the
"the editor is too complex" problem by making our editor *not be* the general
splat editor. What we keep — TRIPS render, per-region mix, TRIPS-guided fog
classification, SAM lift, shade finder, honesty views — is untouched by this
decision, because SuperSplat cannot do any of it.

### What we drop in SuperSplat's favour (explicitly, so nobody re-opens them)

- Lasso / polygon / flood / eyedropper selection in `trips-viewer`. **Dropped.**
- Select-by-value histogram UI in `trips-viewer`. **Dropped as UI** (we still
  *compute* confidence; see task 3 — we export it, we do not draw sliders for it).
- Colour grading of Gaussians. **Dropped.**
- Geometry transform gizmos on the splat itself (as opposed to on regions).
  **Dropped.**
- Ground-plane/orient and measure tools. **Dropped.**
- Compressed export formats (SOG/SPZ/compressed PLY) written by us. **Dropped** —
  use `@playcanvas/splat-transform` (their CLI/library) instead.
- An editable camera timeline in `trips-viewer` for *splat-only* flythroughs.
  **Dropped** (TRIPS flythroughs stay ours — SuperSplat cannot render them).
- A trippy web viewer for *splat-only* delivery, and any VR ambition for splats.
  **Dropped** in favour of `@playcanvas/supersplat-viewer`. `trips-web` stays,
  for TRIPS.

### What is explicitly NOT dropped

Everything in §1b. Plus `trips-viewer`'s Simple Mode, brush, click-to-cluster and
region model — those are the TRIPS edit layer, and SuperSplat has no equivalent.

---

## 5. Task list for the chosen option

### Stage 1 — self-host (target: 2 agent-days, no GPU queue, no network at run time)

1. **`scripts/supersplat_bootstrap.sh`** (0.5 d). Idempotent, Bash 3.2 safe,
   `set -u` clean. Clones `https://github.com/playcanvas/supersplat` into
   `$TRIPPY_OUTPUT/tools/supersplat` (**outside** the repo, gitignored — no
   vendoring, §3's licence rule), checks out a **pinned tag** recorded in the
   script as a named constant with a comment (`SUPERSPLAT_PIN`, start at the
   3.0.0 release tag), runs `npm ci` then `npm run build`, and prints the `dist/`
   path. Re-running with the same pin must be a no-op. Refuses to run if `node`
   is older than 20.19.
2. **`OPEN_SUPERSPLAT.command`** (0.25 d). Generated next to `dist/` by the
   bootstrap script, same shape as `scripts/deliver.sh`'s existing launcher:
   `python3 -m http.server --bind 127.0.0.1 <port>` over `dist/`, then open the
   browser at `http://127.0.0.1:<port>`. **Bind 127.0.0.1 explicitly** — never
   `0.0.0.0`. Deliver it via `scripts/deliver.sh` so it lands in Jordan-Review
   with the rest.
3. **`trippy export-splat-layers`** (0.75 d). Takes a cleaned run and writes a
   **two-file pair** into `$TRIPPY_OUTPUT`:
   `<name>-keep.ply` (the survivors) and `<name>-fog.ply` (the Gaussians TRIPS
   stopped believing in), byte-for-byte row copies out of the source PLY, exactly
   as `feat/splat-clean` already does for the keep half. Jordan opens **both** in
   SuperSplat; they arrive as two layers with show/hide/**solo** in the scene
   panel, so the fog mask is inspectable, toggleable and deletable with zero new
   format work. Also emits a `<name>-layers.txt` one-pager naming the two files
   and the threshold used. *(This is the "TRIPS fog mask export SuperSplat can
   select/delete with" — via layers, which is the mechanism their data model
   actually has.)*
4. **The privacy proof, run and recorded** (0.25 d). Airplane-mode test +
   DevTools Network-tab test from §3, on the built `dist/`, with a **synthetic**
   PLY (never a Karekare scene, per `AGENTS.md` §6 — the point is to test the
   app, and a synthetic file tests it identically). Record date, exact steps and
   the observed request list in `research/trips-metal.md`. If **any** external
   request appears, stop and escalate to Jordan before this is used on a real
   scene.
5. **Docs** (0.25 d). A "Cleaning a splat in SuperSplat" section in
   `docs/USER_GUIDE.md`: the two-layer workflow, the three keys worth knowing
   (`Shift+B` sphere brush, `N` selection depth, `M` selection footprint), the
   Data-panel opacity histogram range-select for what the mask missed, and a
   **red-box warning: never click File → Publish**.

### Stage 2 — deferred B (do not start until Jordan has used Stage 1)

6. **Watch and ask** (0.25 d). After Jordan's first two SuperSplat sessions, ask
   exactly one question: *which SuperSplat tool do you wish the TRIPS viewer had?*
   Port at most **two** of them. Current best guesses, unvalidated: (a) a
   confidence **histogram + range drag** over the TRIPS points, which is our
   fog threshold made visual instead of typed; (b) **footprint-vs-centre**
   selection testing, which matters for exactly the big blurry Gaussians fog is
   made of. Estimate if both are chosen: **4–6 agent-days**.
7. **`.ssproj` writer — evaluate, do not build yet** (0.5 d to spike, 2–3 d if
   built). `document.json` v1 + shared PLY resource + `instances_*.bin` row-index
   blobs would let us hand Jordan **one file** that opens with the fog already
   split into its own named layer. It is strictly nicer than task 3's two files,
   but it means tracking an undocumented internal format across upstream versions.
   **Only build this if task 3's two-file workflow actually annoys Jordan.**

### Stage 3 — delivery and Quest (target: 2 agent-days)

8. **SOG export without the browser** (1 d). Use `@playcanvas/splat-transform`
   as a local CLI (`npx @playcanvas/splat-transform in.ply out.sog`) from
   `scripts/`, pinned to the same version SuperSplat 3.0.0 uses (**3.3.3**),
   installed into a gitignored local `node_modules` — never a global install
   (`AGENTS.md` §6's "no installing outside ./.venv" has the same spirit here).
   Verify the SOG round-trips by loading it back in the self-hosted editor.
   Confirm its licence at pin time (§3).
9. **Self-hosted viewer package for the Quest** (1 d). Export the ZIP viewer
   package from the self-hosted editor (or build `@playcanvas/supersplat-viewer`
   directly), serve it via a `deliver.sh`-generated `OPEN_*.command` on
   127.0.0.1, and have Jordan open it on the Quest browser over the LAN. This is
   the concrete answer `docs/QUEST.md`'s "what ships to the Quest instead"
   section was pointing at; update that document with the measured result.
   **Privacy note:** LAN-only, our own machine serving; nothing leaves the house.

### Parked, never culled (`AGENTS.md` §7)

10. **The portable radix sort** (3–5 d, parked in `research/README.md`). PlayCanvas
    engine's MIT 4-bit portable radix sort is the documented fix for
    `trips-web`'s "no WebGPU subgroups in Safari" blocker
    (`docs/QUEST.md`, `docs/WEB_VIEWER.md`). Not urgent — Chrome works on
    Jordan's machine — but it is the one piece of SuperSplat's *engine* worth
    reading with intent to port.

---

## 6. Consequences

- `docs/EDITOR.md`'s milestone table stops growing on the plain-Gaussian axis.
  Anything a future brief asks for from §4's "dropped" list should be answered
  with "SuperSplat does that; see ADR-0008."
- We take on one external dependency we do not control. Mitigated by pinning a
  tag, by never vendoring, and by the fact that a static `dist/` we have already
  built keeps working regardless of what upstream does next.
- Jordan gains a second application to learn. This is a real cost, and it is
  accepted because the alternative is a second *worse* application that we also
  have to build and maintain.
- The Publish button is a standing hazard. Mitigated by self-hosting (where it
  cannot authenticate), by the docs warning in task 5, and by never using the
  hosted editor at `superspl.at/editor` with Jordan's scenes.
- No decision here touches training, the bundle format, `edits.json`, or any
  GPU path. Stage 1 is entirely scripts and docs.

## 7. Related

- `docs/EDITOR.md` — the trippy editor this ADR narrows the scope of.
- `docs/decisions/ADR-0007-viewer-editing.md` — the edit-layer architecture that
  stays exactly as it is.
- `docs/decisions/ADR-0006-viewer-integration.md` — why `trips-viewer` is its own
  binary; the same reasoning is why we do not fork SuperSplat.
- `docs/decisions/ADR-0004-public-repo-privacy-guardrails.md` — the guardrails
  §3's verdict is measured against.
- `docs/QUEST.md` — "what ships to the Quest instead"; Stage 3 is its answer.
- `docs/WEB_VIEWER.md` — `trips-web`, which stays, and the subgroups blocker
  task 10 addresses.
- `STATE.md` 2026-09-08 entries — the verdicts ("splat is the base", "too
  complex") that make this decision the right one now and not a week ago.
