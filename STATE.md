# STATE — externalized progress (update at end of every session)

Last updated: 2026-09-10 (feat/supersplat-quest session; previous: feat/supersplat-selfhost)

## Done
- 2026-09-12 17:17: union-broadcast (kk-coherent) done: 17.25 dB, strict 16.19; union source no gain. GPU: union-trips (55), the last queued trippy job.
- 2026-09-12 12:33: removal-rel (kk-coherent) done: 17.69 dB, strict 16.52; no differentiator. GPU: union-broadcast (55) now, then union-trips; queue then empty.
- 2026-09-12 07:44: kkv2-6-shade-prune done (15.06 dB, dark-mass 31.9%): not a candidate (deletes geometry). GPU: removal-rel (55) now, then union x2; queue then empty.
- 2026-09-12 00:35: full3-alt (kk-coherent, alternating hold-out) done: 17.89 dB, strict 17.06. GPU: kkv2-6-shade-prune (55) now, then removal-rel, union x2.
- 2026-09-11 19:25: hybrid-a-all-levels (kk-coherent, trips mode) done: 17.26 dB, strict 16.88 at ep 87. GPU now on the 55s: full3-alt, then kkv2-6-shade-prune, removal-rel, union x2.
- 2026-09-11 14:16: plain TRIPS full scene at 300 epochs: 14.78 dB (worse than 15.06 at ep 122) -> longer plain training does not help; hybrids are the path. kkv2-1-full-masked launcher now = epoch 299. Remaining GPU: hybrid-a-all-levels (50) then the 55s.
- 2026-09-11 12:22: hybrid-a-all-levels-bc (kk-coherent) done: 17.45 dB, strict 16.39. kkv2-8b-finish running (45).
- 2026-09-11 08:20: hybrid audits corrected: hybrid A 34.1%, gate 32.6% (TRIPS 37.3, Gaussians 26.7). kkv2-8 killed at ep 259 during wrap-up -> kkv2-8b-finish queued (45). hybrid-a-all-levels-bc running (50).
- 2026-09-10 20:07: kkv2-7-hybrid-gate done (ep 103): PSNR 16.94 dB, strict 16.62 dB (best of everything). REVIEW QUEUE: 4-other/kkv2-7-hybrid-gate-viewer.command. Then kkv2-8 continuation, then the 50/55 runs.
- 2026-09-10 12:35: Quest SOGs done (119 MB untouched, 114 MB cleaned; counts round-trip exactly). REVIEW QUEUE: 4-other/quest-viewer-shade-keep-quest.command (put on the headset, open the printed URL) and quest-viewer-kklid20000-quest.command to compare. Corrected audit runs next, then kkv2-7 hybrid-gate.
- 2026-09-10 12:26: kkv2-5-hybrid done (ep 139): PSNR 16.85 dB, best full-scene number so far (plain TRIPS 15.06). REVIEW QUEUE: 4-other/kkv2-5-hybrid-viewer.command. Next on GPU: two Quest SOG conversions, corrected audit, then kkv2-7 hybrid-gate.
Last updated: 2026-09-10 (fix/audit-sparse-v2 session; previous: feat/supersplat-selfhost)

## Done
- 2026-09-10 (fix/audit-sparse-v2, worktree `.worktrees/audit-sparse`): second bug in the
  karekare-v2 shade audit, on top of the 2026-09-09 frame-list fix. `trippy train --report`/
  `trippy-shade-audit-rerun2` hardcoded `<scene_root>/sparse_txt`, which does not exist for
  karekare-v2 (only 16 binary `sparse/<n>` sub-models) -- the rerun job failed outright with
  `FileNotFoundError`. Confirmed `sparse/0` is the 756-registered-image model (via `images.bin`
  header parse; matches config.yaml's own comment) and converted it to TEXT with
  `colmap model_converter` into `$TRIPPY_OUTPUT/scenes/karekare-v2/sparse_txt` (never written
  into `~/Splats/scenes/`). New `trippy.render.report.resolve_sparse_txt_dir` auto-converts a
  binary `sparse/0` the same way for any scene lacking `sparse_txt`, caching the conversion;
  new `TrainConfig.sparse_txt` override, set explicitly on every EXP-0011 config;
  `report.json` now records `sparse_txt_dir` used. 7 new unit tests with a fake `colmap` on
  PATH (`tests/test_render_report.py`); full Python suite green (1281 passed, 10 skipped, same
  pre-existing 10 `test_web_build_script.py` npm/wasm-pack failures as prior sessions,
  unrelated). Re-ran the corrected audit (`trippy-shade-audit-rerun3`, prio 40, `--wait`):
  **corrected dark-mass numbers, sparse/0 + the correct 93 big-tree frames**: kkv2-1 37.3%,
  kkv2-2 37.4%, kkv2-3 37.2%, kklid_20000 baseline 26.7% (previously wrong-scene numbers never
  landed for the candidates; the baseline's earlier wrong-scene/wrong-frame number was 17.3%).
  Extent gate re-checked on the same four PLYs: PASS, no sprawl, consistent across all four.
  `docs/RESULTS.md` and `research/trips-metal.md` updated with the corrected table and full
  root-cause writeup (both bugs). Files touched: `trippy/render/report.py`,
  `trippy/train/config.py`, `trippy/constants.py`, `experiments/EXP-0011-karekare-v2/*.yaml`,
  `tests/test_render_report.py`. Not touched (out of file scope for this task, still has the
  same hardcoded `sparse_txt` literal): `trippy/cli.py`'s `_cmd_candidate_report` -- a
  follow-up should route it through `resolve_sparse_txt_dir` too. Not committed (per this
  task's Forbidden list) -- worktree `.worktrees/audit-sparse`, branch `fix/audit-sparse-v2`,
  awaiting Orchestrator review.
- 2026-09-09 14:40: disk cleanup #3, ~14 GB freed (details in research log).
- 2026-09-09 13:50 (Jordan): editor direction = self-host SuperSplat now, fork later only if the two-app workflow annoys. Stage 3 (SOG + self-hosted WebXR viewer for the Quest) started. Fork estimate recorded in ADR-0008 (3-5 agent-weeks; MIT permits it).
- 2026-09-10 (feat/supersplat-quest, worktree `.worktrees/supersplat-quest`):
  **ADR-0008 Stage 3 tooling built and tested: SOG export + a self-hosted WebXR viewer for
  the Quest.** `scripts/sog_export.sh` (PLY -> `.sog` + `.compressed.ply`, round-trip
  Gaussian-count verified, `@playcanvas/splat-transform@3.3.3` pinned local install) and
  `scripts/quest_viewer_bootstrap.sh` (`.sog` -> a self-hosted static viewer bundle via
  splat-transform's own `.html --unbundled` output, which calls `@playcanvas/supersplat-
  viewer`'s `renderViewerHtml` directly -- confirmed by reading splat-transform 3.3.3's
  `package.json`, so no second clone-and-build pipeline was needed for the two literal
  options the ADR named). `scripts/open_quest_viewer.sh` writes a `127.0.0.1` Mac-preview
  launcher and an `_ON_QUEST` launcher (self-signed HTTPS on the LAN, required because
  WebXR needs a secure context a bare LAN IP doesn't get, and the URL always carries
  `?webgl` because the viewer's VR button only appears under the WebGL renderer). Both
  launchers can be generated wired to a not-yet-built bundle path and refuse to open
  ("Not ready yet") until the SOG is actually there -- verified for real against the two
  launchers below, right now, before their jobs land. 24 new tests (`tests/test_sog_export_
  script.py`, `tests/test_quest_viewer_bootstrap_script.py`,
  `tests/test_open_quest_viewer_script.py`), `ruff check` clean.
  **Found live and fixed**: a direct CPU-only conversion (`-g cpu`) on the real
  8.5M-point `kklid-tripsclean-shade-keep.ply` hung 14 hours with no progress (scratch
  `.tmp` static at 78 MB) -- killed, and both scripts gained a `--gpu n|cpu` passthrough
  (default `cpu`, safe for direct use; `--gpu 0` = native WebGPU/Dawn, only from inside a
  GPU-queue slot) so the conversions could move to `scripts/gpu_submit.sh --prio 40`
  instead. **REVIEW QUEUE for Jordan (works right now, refuses cleanly until the jobs
  below land): `4-other/quest-viewer-shade-keep-preview.command` /
  `-quest.command`, `4-other/quest-viewer-kklid20000-preview.command` / `-quest.command`.**
  **PENDING**: GPU-queue jobs `trippy-quest-sog-shade-keep` and `trippy-quest-sog-
  kklid20000` (prio 40) are queued behind a running training (`kkv2-5b-hybrid-resume`)
  with free memory near zero at submit time; once their `.rc` files show `0`, fill in
  `docs/QUEST.md`'s "the two real conversions" section with real sizes/round-trip counts.
  **This worktree (`.worktrees/supersplat-quest`) must stay until both jobs finish**
  (their job files `cd` into it). Full narrative in `research/trips-metal.md`'s
  2026-09-10 entry. `scripts/build.sh`'s Rust half and the full pytest suite were NOT
  run this session (near-zero free memory with a live training running -- only
  `compileall`/`ruff`/`import trippy` and the three new test files, all green, plus the
  pre-existing 24 tests specific to this change); a full `scripts/test.sh` should be run
  once memory frees up, before merge.
- 2026-09-09 13:20 (Jordan): Splats Hunua at prio 30; trippy stays in 40-60 (short 40, kkv2 45, hybrids 50, other 55). Queue files renamed; gpu_submit bands updated. kkv2-5-hybrid was killed at ep 60 (Killed: 9, 10:23) -> resubmitted as kkv2-5b-hybrid-resume (45). shade-audit-rerun failed (pointed at a removed worktree) -> rerun2 at 40 from main. combined parity rc=5 (out-dir inside repo) -> resubmitted with an outside dir. SuperSplat Stage 1 merged (build-0132): supersplat.command launcher + keep/fog layer PLYs in 2-open-in-brush.
- 2026-09-09 (feat/supersplat-selfhost, worktree `.worktrees/supersplat-selfhost`):
  **ADR-0008-supersplat.md Stage 1 implemented and delivered** -- SuperSplat 3.0 is now a
  real, self-hosted tool, not just a decision. `scripts/supersplat_bootstrap.sh` clones
  `playcanvas/supersplat` at pinned tag `v3.0.0` into `$TRIPPY_OUTPUT/tools/supersplat`
  (gitignored, never vendored), builds it (`npm ci && npm run build`, 10.3 s), idempotent
  on re-run (verified twice). `scripts/open_supersplat.sh` generates
  `OPEN_SUPERSPLAT.command` (127.0.0.1:8877 only, "never click Publish" warning) --
  **REVIEW QUEUE for Jordan: 4-other/supersplat.command**. New `trippy export-splat-layers`
  (`trippy/clean/layers.py`, 4 new tests, all passing) writes a `<name>-keep.ply` /
  `<name>-fog.ply` byte-for-byte partition for one `splat-clean` variant, reusing
  `trippy.clean`'s own scoring/mapping/selection verbatim. Run for real on
  `kklid_20000.ply`'s `shade` and `005` thresholds (CPU direct, ~12 GB free, fast: 3.6 s /
  15.4 s -- no GPU queue needed, well under the task's 10 GB caution line); deletion counts
  match the existing `splat-clean` variants exactly (365,716 and 972,630 of 8,910,382).
  **REVIEW QUEUE: 2-open-in-brush/kklid-tripsclean-{shade,005}-{keep,fog}.ply** -- open all
  four (or just one pair) via the SuperSplat launcher above; the fog layer is soloable.
  **Privacy proof run for real, not just read from source: PASS.** Headless Chrome +
  full network logging drove a SYNTHETIC ply (never Karekare) through load, an edit
  (select-all + delete) and an export; every request tied to the page/ply/export was
  127.0.0.1-only. A `about:blank` control run on the identical Chrome profile proved the
  only other hosts seen (Google Safe Browsing / Chrome Web Store / GCM / optimization
  guide) are Chrome's own background traffic, present with no page loaded at all -- not
  something SuperSplat's code does. Full request list: `research/trips-metal.md`
  2026-09-09 entry. Docs: `docs/USER_GUIDE.md` "Editing the cleaned splat in SuperSplat",
  `docs/QUEST.md` points at Stage 3 next. `scripts/test.sh`'s Python suite green (1275
  passed, 10 skipped, the same pre-existing 10 `test_web_build_script.py` failures as the
  prior session -- `rust/brush-trips` submodule still not initialised on this machine,
  confirmed via `git submodule status`; `cargo check`/`cargo test` not run for the same
  reason, unrelated to this task's Python/scripts-only changes). Stage 2 (deferred by the
  ADR itself -- wait for Jordan's first two sessions) and Stage 3 (SOG export + the
  self-hosted viewer package for the Quest) are next, not started.
- 2026-09-09 07:50: merged splat-clean, viewer Simple Mode + Brush camera, perf harness (build pending). REVIEW QUEUE for Jordan: (1) 2-open-in-brush/kklid-tripsclean-shade.ply then -005 and -015 (your splat minus TRIPS-identified fog; open in Brush); (2) 4-other/kkv2-1-combined-viewer.command (splat + TRIPS under the tree, mix slider works); (3) 4-other/kkv2-3-removal-viewer.command (removal arm, 15.04 dB). Viewer Simple Mode lands in every launcher once the binary rebuilds in the next GPU gap. SAM default -> mps (8.0 s vs 9.6 s CPU). Perf: no exact-parity 2x exists (see research log); queue hold was not executed (permission layer) and is moot.
- 2026-09-09 (fix/report-shade-frames): two fixes, worktree `.worktrees/report-frames`.
  **(1) Shade-frame bug**: `trippy train --report` always audited shade dark-mass on
  `depthprior_shade_audit.py`'s own default (kk-coherent's `IMG_3828-3833`, 6 frames)
  instead of karekare-v2's measured 93-frame big-tree group, on EVERY karekare-v2 report
  so far (kkv2-1/2/3). Fixed with a new `TrainConfig.shade_frames` config key
  (`trippy.render.report.resolve_shade_frames` threads it to both the candidate and
  baseline audits; `report.json["shade_frames"]` now always records which frames were
  used); every EXP-0011 config now sets it. Tested with a fake `depthprior_shade_audit.py`
  stand-in (`tests/test_cli_train_report.py`) plus pure unit tests
  (`tests/test_render_report.py`). Also found (not fixed, out of file scope):
  `trippy.eval.audits.cached_baseline_audit`'s cache key ignores `frames` entirely --
  the stale `kklid_20000` cache entry (wrong-frame result, matched the old 17.3% number)
  was deleted, but the underlying key bug needs a follow-up touching
  `trippy/eval/audits.py`. **Re-audit DONE (2026-09-10, fix/audit-sparse-v2)**: the first
  rerun attempt (`trippy-shade-audit-rerun`/`-rerun2`) hit a SECOND bug -- `sparse_txt` was
  hardcoded and karekare-v2 has no such directory -- see that session's entry above for the
  fix and the final corrected numbers (kkv2-1 37.3%, kkv2-2 37.4%, kkv2-3 37.2%, baseline
  26.7%), now in `docs/RESULTS.md` and `research/trips-metal.md`.
  **(2) SAM 3 default device flipped `cpu` -> `mps`** (`docs/EDITOR.md` §3's decision rule
  satisfied: `edit-sam-5` MPS rc=0 at 8.05 s/view beat `edit-sam-5-cpu`'s 9.65 s/view).
  Changed: viewer's `SamUi::default()` and headless `--sam-device` default
  (`rust/crates/trips-viewer/src/edit_ui.rs`, `src/main.rs`), and
  `trippy.edit.sam_runner.Sam3Segmenter`'s own Python default. `cpu` stays a one-click/
  one-flag fallback everywhere. **Known gap**: `trippy edits sam`'s terminal CLI default
  (`SAM3_DEFAULT_DEVICE`, `trippy/constants.py`, read by `trippy/cli.py`) is UNCHANGED --
  both files were outside this task's edit-file list; a one-line follow-up is needed so the
  terminal command agrees with the viewer.
  Python suite green (1223 passed, `tests/test_web_build_script.py`'s 10 failures are
  pre-existing/unrelated -- `rust/brush-trips` submodule not initialised on this machine,
  confirmed via `git submodule status`). Rust `cargo test` NOT run: this worktree has no
  `rust/target` yet (cold build), and free memory was 12-16 GB with a live training running
  -- deferred per this task's own brief ("say so and I will run it on main").
- 2026-09-08 20:50 (Jordan): the 3-day full-res run is PARKED (not worth the GPU time now; keep as an option). Job files moved to ~/Splats/tools/gpu_queue/parked-trippy/ (kkv2-9-fullres-smoke, kkv2-9-fullres, 9b, 9c); configs merged on main. kkv2-8 (half-res to 300 ep) still runs.
- 2026-09-08 20:30 (Jordan): still wants the HIGHEST-QUALITY plain TRIPS scene for comparison. Plan: turn kkv2-9-fullres into a chained multi-day run (2016 wide, crop 512, full 300 epochs, resumable 12 h segments) at prio 42 (after the kkv2 hybrid arms, before hybrid-a), launcher at render scale 1.0 without half-net. Set up when the fullres agent reports.
- 2026-09-08 19:40: Jordan: TRIPS looks nothing like a photo anywhere except that the clouds/fogs are gone; splat is the base. New track: TRIPS-confidence-guided cleaning of kklid_20000 (delete fog Gaussians) -> clean PLY for Brush. Full-res variant demoted below hybrids once queued.
Last updated: 2026-09-08 (feat/splat-clean session; previous: fix/editor-followups)

## Done
- 2026-09-08 21:10 (feat/splat-clean): **Design B is built and three cleaned splats are in the review queue.** TRIPS is now used only as a classifier over Jordan's own splat: the Gaussians its trained twin stopped believing in are deleted, every survivor's PLY row copied byte for byte. **The 1:1 mapping question is answered exactly** -- `points_removed_total = 0`, the reconstructed `sigmoid(opacity) >= 0.05` filter keeps exactly 7,542,137 of 8,910,382 rows, `init_conf` matches `sigmoid(ply.opacity)` with **max |diff| = 0.0 on all 7,542,137 points**, and the positional control (drift p50 0.015 vs 6.18 shift-by-one / 6.37 permuted) is a 200x separation. No nearest-neighbour fallback was needed (it exists and is tested anyway). **Deviation, recorded:** used `checkpoint_latest.pt` (ep 122, the checkpoint Jordan's 19:10 verdict was actually about) not `checkpoint_best.pt` (ep 40) -- ep 122 has 12.90% of points below conf 0.05 vs 5.22%, and 74.7% vs 58.8% of its revealed pixels land on a confident surface. **REVIEW QUEUE for Jordan: open `kklid-tripsclean-shade.ply` in Brush first** (365,716 deleted, 4.10%, shade volume only), then `-005` (972,630, 10.92%) then `-015` (2,085,635, 23.41%) if fog remains. Dark mass barely moves (26.71% -> 27.08/27.08/27.62% on the correct 93-frame big-tree region), which is expected after 19:10 retired that metric; extent unchanged (p99 12.87 -> 12.89/12.91/12.98, max 34.40). Honesty pack `kklid-tripsclean-honesty` in `4-other/`. **Correction found on the way: every kkv2 run report's dark-mass number (17.3% / 34.5%) was measured on the SIX `IMG_3828-3833` kk-coherent frames, not the measured 93-frame big-tree list** -- `trippy.render.report` never passes `--frames`. On the right region the untouched splat is 26.71%.
- 2026-09-08 19:10: JORDAN VERDICT: full-scene TRIPS (kkv2-1/2) SOLVES the big-tree shade; rest of scene fuzzy/pixelated. Shade-prune approach rejected (removes geometry) -> kkv2-6 demoted to prio 50. Requests: Brush-style mouse controls; a simpler editor; mix that actually works (needs Gaussian block in bundles). In flight: combined bundle (TRIPS+kklid splat, shade preset) and viewer simple mode.
- 2026-09-08 (feat/combined-bundle): delivered the combined TRIPS+splat bundle Jordan asked for from
  the start -- re-exported `kkv2-1-full-masked` (same checkpoint, deterministic, CPU-only, 36.7 s) and
  its bundle now carries `blend.splat_ply` (the `native_blend`/`feat/live-splat` machinery already did
  this for any Gaussian-seeded run; the delivered bundle just predated being re-run through it). Added
  an `edits.json` region "Big tree shade (TRIPS)" (758,178 points, `trippy edits shade-find` on the
  exact 93 measured big-tree shade frames, `op=blend mix=1.0`) and a launcher that opens at global
  `mix=0.0` (full splat) so the per-region override is visible immediately: splat everywhere, TRIPS only
  under the tree. Alignment measured (never viewed): nearest-3D-neighbour TRIPS point vs its matching
  splat Gaussian, reprojected into 3 training cameras, **combined median 0.87 px** (< the 2 px bar).
  `scripts/test.sh` green, no code changed (only shipped tools run against real data). **Gap for the
  Orchestrator:** no GPU-verified fps or viewer-vs-Python screenshot parity number exists yet for this
  exact bundle+edits combo -- `scripts/viewer_parity_check.sh` needs the GPU (a training holds it) and
  was not queued (brief said don't wait); `scripts/viewer_splat_check.sh`'s public-scene allow-list
  would refuse a Karekare bundle even if queued, so that check does not apply here at all. Queue
  `viewer_parity_check.sh` via `scripts/gpu_submit.sh --prio 15` when convenient for a real fps number.
  Full numbers: `research/trips-metal.md` 2026-09-08 "the combined TRIPS+splat bundle" entry. Delivered
  `kkv2-1-combined-viewer.command` (Jordan-Review 4-other) -- **REVIEW QUEUE**: open it, fly from the
  big-tree shade (TRIPS) into the open lawn (splat) and see if the mix boundary reads as natural or as
  a seam.
- 2026-09-08 18:40: SAM 3 on MPS attempt 5 merged (PR #55, build-0124): generalised stray-tensor mover + start-up assertion + --device-audit. Jobs edit-sam-5 (mps) and edit-sam-5-cpu (prio 15) run after kkv2-3; default device follows the measured seconds/view (docs/EDITOR.md §3). Keep .worktrees/editor-sam-ui until both jobs finish.
- 2026-09-08 17:44: kkv2-2-full-unmasked done (ep 105/300): PSNR 14.63 dB, shade dark-mass 34.5% (same as masked). Launcher kkv2-2-full-unmasked-viewer.command in review queue. kkv2-3-removal running.
- 2026-09-08 11:50: disk cleanup, 22 -> 84 GB free (details in research log). removal-rel job repointed to main repo.
- 2026-09-08 (fix/editor-followups): the three items the 04:30 entry below left open are closed. **(1) Brush depth anchor measured and fixed**: 47.4 ms/sample brute force on the real `kkv2-1-full-masked` bundle (7.5M points) -> 0.011 ms/sample via a new screen-space bucket index (`edit::brush::ScreenGrid`, built once per camera pose, exact vs the brute-force scan, checked on randomised inputs); new `--bench-brush-anchor N` flag reproduces the numbers. **(2) npz sidecar writer shipped on the viewer side**: `EditDocument::save` externalises a brush region above `EDIT_BRUSH_NPZ_CELL_THRESHOLD` (4096) cells into `edits_brush_<id>.npz`, array-for-array identical to the Python CLI's writer (`edit::npz_write`, hand-rolled ZIP/npy, no new dependency); same synthetic over-threshold stroke pinned on both sides, cross-checked with plain `numpy.load` against a Rust-written file (`--brush-npz-selftest`). **(3) Lid plane-normal gizmo handle**: a 4th handle drags `up` (tilt about the two in-plane axes), one undo entry, Inspector typing unaffected. `scripts/build.sh` + `scripts/test.sh` green (1232 python, 154+50 trips-viewer unit tests). Numbers and verdicts: `research/trips-metal.md` 2026-09-08 entries (three bullets). Docs: `docs/EDITOR.md`, `docs/USER_GUIDE.md`, `CHANGELOG.md` updated.
- 2026-09-08 10:26: kkv2-1-full-masked done (epoch 122/300, budget): PSNR 15.06 dB, shade dark-mass 34.5% vs 17.3% Gaussians. Delivered kkv2-1-full-masked-viewer.command + dolly + honesty + PLY. REVIEW QUEUE for Jordan: open that launcher and judge the big-tree shade. kkv2-2-full-unmasked running; kkv2-8 continuation queued after kkv2-7. sam-ui-proof4 passed on MPS.
- 2026-09-08 04:30: Editor complete in the viewer: brush painting (per-stroke undo, golden parity), Named Objects panel (enable/mix/solo/rename), drag gizmos (PR #53, build-0119). Open (closed 2026-09-08, see above): brush depth anchor is an O(points) scan (unmeasured on a Karekare-scale cloud); viewer keeps brush cells inline in edits.json (no npz sidecar writer); lid has no plane-normal handle.
- **feat/live-splat (2026-09-07): the Gaussian splat is rendered LIVE in the viewer, at any pose.**
  `bundle.json`'s `blend.splat_ply` is loaded once into `brush_render::Splats` on the viewer's own
  device (`trips-viewer/src/splat.rs`, `brush-serde` for the ply, `brush-render` for the raster,
  both path deps into the untouched submodule, `cfg`-gated off wasm) and rasterised at every
  frame's camera, so `splat`/`gated`/`mix`/`split` work everywhere instead of only on the dozen
  capture views `splat.npz` carried. **The camera-convention finding: there is no axis flip** --
  trippy and Brush share `+X` right / `+Y` down / `+Z` forward, so the conversion is
  `rotation = R^T`, `position = -R^T t`, `fov = focal_to_fov(focal, pixels)`,
  `center_uv = (cx/W, cy/H)`, and the `R^T` is free (row-major `R` read column-major by glam).
  Pinned by a CPU test projecting five points through both cameras with each library's own code.
  **`blit.wgsl` needed no change**: `TextureMode::Float` gives `[H,W,4]` f32 which slices into the
  planar `[1,3,H,W]` the tone mapper already produces. Proof without a window
  (`scripts/viewer_splat_check.sh`, synthetic bundle, camera yawed 20 deg off a capture view):
  mix 0 vs mix 1 = **58.126/255, 100% of pixels changed**, control with `--no-live-splat` = **0.000**.
  Horse regression: the pre-change binary and the new one write a **byte-identical** frame
  (max|a-b| 0.0). Also: `export-bundle` now records `splat_ply` for **any** Gaussian-seeded run
  (every Karekare run), not just hybrids; two pre-existing bugs fixed on the way (`trips-web` did
  not compile after the Blend panel changed `Renderer::render`'s signature; a gate-less hybrid
  export wrote `gate = "1"` next to `out_channels = 3`, which `brush_unet::weights` refuses).
  **Measured** (`trippy-live-splat-perf-1`, rc 0): `kklid_20000.ply` (8 910 382 Gaussians, SH 3,
  2.1 GB) loads in **2.7 s** and renders in **27 ms at 1080p** (37 fps; 4.99 M of 8.91 M visible,
  so the PLY and the kk-coherent bundle DO share a world frame -- that was an open doubt);
  `--splat-subsample 4` takes it to 10 ms. Whole frame at 1080p: TRIPS 185.67 ms vs live-splat
  mix 184.83 ms, i.e. the blend is free within noise on a U-Net-bound frame (~+15% on Karekare).
  Odd result recorded but not chased: 1080p renders FASTER than 1008x756 (27.1 vs 33.4 ms).
- **fix/viewer-kk (2026-09-06): the white Karekare frame was an untrained per-image exposure, not a viewer bug.** `Trainer._initial_exposure` encoded "no EXIF" as "EV 0" *before* subtracting the scene mean, so an EXIF-less photo got a relative EV of -5.870477 on kk-coherent = a **58.5x** tone-mapper gain, which clips the response LUT flat to `LUT(1) = (0.888, 0.875, 0.863)`. Ten of 219 images have no EXIF; six are held out (their exposure never gets a gradient) and one of those six was view 0, the view the bundle opened at. Proof: the run's own six worst held-out PSNRs (6.19-6.92 dB) are exactly those six views. Job `trippy-viewer-kk-1` (rc 0) settled it numerically -- **the viewer matches trippy's Python reference at 85.78 dB on the BROKEN bundle**, which rules out f16, the response LUT, the background and the feature layout in one measurement. **View 0 vs its own photograph: 6.20 -> 14.92 dB** (its neighbour is 15.49); the unaffected views are byte-identical before and after; **horse parity unchanged at 82.68470 dB**. Four fixes (the trainer half is `feat/eval-calib`'s implementation, kept on rebase -- same bug found independently from two directions): trainer (no EXIF -> gain 1), exporter (`trusted_exposures` substitutes the scene median beyond 2 stops, recorded in the bundle metadata), `default_view` moves off an untrustworthy view (full2: 0 -> 26 `IMG_3735`), and the viewer picks its own exposure off a capture pose (`X`, `--exposure`). Plus navigation 4x faster with a 50x scroll ceiling. Re-delivered: `full2-broadcast-viewer-v2`, `trips-kk-full1-viewer-v2`, `trips-mac-viewer-horse-v3`. New tool: `trippy bundle-parity` + `scripts/viewer_parity_check.sh` (numbers only, safe on private scenes).
- **feat/web-perf (2026-09-06): the browser viewer was 27x slower because of the LINKER, not the renderer.** `wasm-ld` wrapped every export in a `.command_export` shim that re-runs the whole `.init_array`; `cubecl-ir` -> `pliron` makes one such run cost ~110 us, and `wasm-bindgen` resolves `__externref_table_alloc` by export name, so every `JsValue` `wgpu` built for a bind group paid it -- ~2,500 constructor runs a frame, 275 ms of a 297 ms frame. `trips-web/build.rs` now links with `--export=__wasm_call_ctors`; `web/trips.js` runs them once and refuses without the export. **Chrome 1440x810: `raw level-0` 3.32 -> 75.9 fps, `network` 1.09 -> 17.7 fps, readback PSNR 62.04 -> 104.54 dB.** Safari re-diagnosed: "Expected 'f16'" was never about f16 -- Safari 26.6.2 has no WebGPU subgroups in any form, so `brush-sort`'s four radix kernels cannot compile; the page now refuses with the exact kernels and builtins, checked on `adapter.features`, not the user agent. Launcher `trips-web-viewer-horse` re-delivered.
- Spec + plan grilled and approved 2026-09-05.
- Phase 1 skeleton reviewed and pushed as `build-0001` (public repo github.com/ggjordan/trippy, 27 CPU tests green).
- feat/points merged (build-0004): PointSet, GaussianPlySource (5.74M pts on kk-coherent, median nn 0.080), density CLI; 50 tests green.
- Web viewer fixed (PR #28): 76 fps raw / 18 fps network in Chrome at 1440x810 (was 3.3 / 1.1); cause was the wasm linker re-running static constructors on every export call. 104.5 dB vs native. Safari: no WebGPU subgroups at all -> page refuses. NOTE: that agent bypassed cpu_heavy.sh's 28 GB memory guard twice under its own watchdog while a training held memory; logged, not repeated.
- Leaderboard merged (PR #27): `trips-leaderboard.png` in Jordan-Review, regenerated after every run.
- v0.3.0 RELEASED (web viewer milestone): full pipeline incl. U-Net in Chrome (62 dB vs native, ~1 fps; 27x gap under investigation in feat/web-perf); Mac viewer 29.5 fps after the point-upload cache. EXP-0008 distillation proven end to end on the weak checkpoint (distilled ply in 2-open-in-brush; dark mass 37% vs 36% TRIPS export vs 20% Gaussians; pipeline proof only).
- Viewer v2 merged (PR #22): drag/orbit/pan/scroll, scene-scaled speed, R/N/P/F keys; launchers `trips-mac-viewer-horse-v2.command` and `trips-mac-viewer-karekare-full1.command` in Jordan-Review.
- v0.5.0 (merged, PR #25): the full TRIPS pipeline incl. the U-Net renders in Chrome via WebGPU (62 dB vs native; ~1.1 fps network / 3.3 fps raw at 1440x810 measured while a Splats training held the GPU; ~20 s first-frame autotune). Safari still draws wrong output and is blocked by the page. Mac viewer after the point-upload cache: shipped preset 29.5 fps, raw 102 fps at 1080p. Chrome was installed as dev tooling. Launcher `trips-web-viewer-horse` delivered. Quest: not interactive by any measure; distilled Gaussians / videos remain the Quest path.
- v0.2.0 RELEASED: native Mac viewer at 22 fps (horse scene), 82.7 dB screenshot parity; launcher `trips-mac-viewer-horse.command` in Jordan-Review. Finding: the U-Net is 89% of frame time, the rasteriser 11%.
- Merged brush-unet (build-0032): Rust pyramid + U-Net + camera on wgpu match the PyTorch path at 115 dB on the horse scene; whole frame 193 ms at 1080p (5 fps, sort-dominated) -> perf work in feat/mac-viewer.
- Merged: self-reporting trainings (build-0028), union point set (EXP-0006: monodepth 3.79M, union 5.89M), web toolchain. Baseline shade audit on kkc_15000: dark(lum<0.25) 19.9% of region mass.
- EXP-0003 full1-broadcast (40 ep, 11 min): held-out 14.42 dB / SSIM 0.39 / LPIPS 0.51, still rising -> long runs queued. Hybrid C (EXP-0005) negative for shade (-1.96 dB). brush-pyramid CubeCL port merged (GPU parity 2e-6). Raster NaN guard merged.
- Trainer fixed (smoke 12.26 dB), native trips mode merged, candidate-report merged. Finding: with Gaussian-scale sizes the U-Net invents ~90% of every frame (t_final 0.93); full runs use kNN sizes.
- v0.1.0 RELEASED on GitHub. Merged since: Brush fork submodule (ggjordan/brush trippy-fork) + crate skeletons, candidate-report (dolly/off-path/audits), se3_exp fix.
- v0.1.0 GATE PASSED 2026-09-06: horse parity 22.27 dB vs GT (authors 22.34), 36.99 dB vs authors' render (EXP-0002). Merged: backward (grads <4e-6), trainer, monodepth, render CLI (EXP-0001, EXP-0004 sheets delivered).
- Merged: raster forward (Metal = reference to 2e-6; 41.6 ms @1008x756/200k pts), net port (34/34 tensors match the public horse checkpoint with num_layers=8), scene loader, PLY export/sheets/video. build-0009, 187 tests.
- docs/TRIPS_REFERENCE.md written from source (key finding: TRIPS's shipped default broadcasts every point into all 5 layers; the 2-layer trilinear path exists but is unreachable from configs).
- Jordan set the goal (2026-09-05 ~22:50): finish all stages autonomously; anything needing Jordan goes in the review queue below.

## In flight
- **ADR-0008 Stage 3 real conversion (2026-09-10, `.worktrees/supersplat-quest`)**: GPU-queue
  jobs `trippy-quest-sog-shade-keep` and `trippy-quest-sog-kklid20000` (prio 40) convert
  Jordan's two splats to SOG + build the Quest viewer bundle; queued behind
  `kkv2-5b-hybrid-resume` with free memory near zero at submit time. **This worktree must
  stay until both `~/Splats/tools/gpu_queue/done/trippy-quest-sog-{shade-keep,kklid20000}.rc`
  appear** (their job files `cd` into it). Once done, update `docs/QUEST.md`'s "the two real
  conversions" section with sizes/round-trip counts, then run the full `scripts/test.sh`
  (deferred this session for memory reasons) before merge. Four launchers are already
  delivered and safe to leave in the review queue meanwhile (they refuse to open until ready).
- **perf/train-step (2026-09-09, `.worktrees/train-perf`): a Karekare-v2 step is 129 ms, and
  2x at exact parity is not available on it.** MEASURED (jobs `trippy-train-perf-baseline`,
  `-gputests`, `-sweep`, all rc 0). Steady-state step **129.2 ms = 1.43 min/epoch**; the split is
  rasteriser 68 ms (52%), perceptual VGG loss 36 ms (27%), Adam 10.5 (8%), dataset crop 10.7 (8%),
  U-Net **6.6 (5%)**. 36% of the step is the objective plus the update rule and cannot shrink
  without changing what the run computes -- that is the ceiling, written up with the arithmetic in
  `docs/ARCHITECTURE.md` "The ceiling". Three of the brief's candidates are now **dropped with
  numbers**: crop-aware culling is already in place (the K-adjust crop culls against the 384-px
  grid), per-image projection caching is dead (`xyz`/`size` are trainable from epoch 5 of 300), and
  padding fragment buffers to bucketed sizes buys nothing (`FROZEN/UNTIMED` = 1.08/0.99/0.92).
  **Both numerics-touching flags measured NEGATIVE and stay off**: float16 is 143-151 ms vs 131 and
  moves the 30-step loss by 1.5e-01; MPS's fused Adam is 145 ms vs 131. The post-sort fragment cap
  is worth **1.05x**. Also found: **MPS's float `index_add_` is not run-to-run deterministic**, which
  is why the rasteriser's gradients cannot be held to byte-equality (the forward can, and is);
  the GPU test now measures that noise floor and requires the change to sit under it.
  **THE HEADLINE, and it is not a good one** (`trippy-train-perf-ab`, rc 0): measured against an
  unmodified `main` worktree, alternating, twice, in one queue slot, **this branch is 1.2-1.6%
  SLOWER** (main 116.1/117.0 ms, branch 117.5/118.9 ms). Six changes that each provably issue fewer
  or smaller kernels net to nothing. **Do not merge this as a speed-up.** Parity IS proven properly:
  `max|dloss|` between checkouts (2.3e-04, 6.2e-04) equals the same-code-twice noise floor
  (4.7e-04, 3.8e-04). `pytest -m gpu`: **78 passed**. Prime suspects for giving the win back, both
  now behind runtime knobs. `trippy-train-perf-isolate` (rc 0, 7 arms with main first and last)
  found **one suspicion right and one wrong**: turning the host-side crop OFF is faster in both
  the median and the min in two separate jobs, so **`Trainer.use_fast_crop` now defaults to
  False** (it is kept, not deleted -- it should tip the other way at width 2016, where the frame
  it avoids uploading is 4x larger). `main` itself drifted **117.1 -> 112.0 ms** across that job,
  so the fragment cap and the sanitiser readback are **unresolved below a ~5 ms noise floor**;
  their apparent +8/+11 ms rows are contradicted by their own `min` columns. `trippy-train-perf-knobs`
  (rc 0) then settled them with 8 arms inside ONE process from ONE snapshot: paired effects
  **use_fast_crop +5.3 ms** (real, third confirmation), **raster_cap -0.6 ms** (NOT established --
  the earlier "1.05x" was one two-arm comparison; kept on only because it is byte-identical and
  strictly less traffic), **sanitise_conditional +0.3 ms** (neutral). Their `max|dloss|` spread of
  3.7e-04..1.6e-03 over 30 steps IS the noise floor at that step count, which closes the last loose
  end. **Net: with `use_fast_crop` off the branch is ~2 ms (~1.8%) faster than main** -- at the edge
  of resolution, not a speed-up worth the name. All five GPU jobs are done (rc 0); nothing is
  pending. Both worktrees (`train-perf`, `train-perf-base`) can be removed once this is merged or
  dropped. The run's own 3.9 min/epoch is unusable as a baseline: its per-epoch
  time drifted 1.63 -> 5.78 min/epoch over the same recipe.
  Next exact lever, briefed but NOT taken: `raster_project_cull` spends 14 ms to keep 5.1% of its
  input; a division-free cull would move the divisions (and most of the geometry backward) onto the
  386k survivors, worth ~10% -- see `docs/ARCHITECTURE.md` "The ceiling" for why it is not more.
  **Two worktrees must stay until `trippy-train-perf-ab` finishes**: `.worktrees/train-perf` and
  `.worktrees/train-perf-base` (a detached `main` checkout created purely as the A/B's "before";
  remove it with `git worktree remove` once the job is done).
- **`trippy-shade-audit-rerun` (prio 15, queued 2026-09-09)**: re-runs the Splats shade
  audit on the CORRECT 93-frame big-tree list for `kkv2-1-full-masked`,
  `kkv2-2-full-unmasked`, `kkv2-3-removal` and the `kklid_20000` baseline (see "Done"
  above). Worktree `.worktrees/report-frames` must stay until it finishes (script `cd`s
  into main, reading `output/scratch/shade_audit_rerun/reconstruct_and_audit.py`, so that
  script -- or its logic -- should be preserved even if the worktree is removed first).
  Once `~/Splats/tools/gpu_queue/done/trippy-shade-audit-rerun.rc` appears, read
  `output/scratch/shade_audit_rerun/*.shade_audit.json`, compute the corrected dark-mass
  fractions, and fill them into `docs/RESULTS.md` and a follow-up `research/trips-metal.md`
  entry (both currently say PENDING).
- GPU queue (reordered 13:58; trippy manages it now): Splats' last Hunua training running; then prio-12/15 short jobs (live-splat-perf-1, edit-sam-1), then 40: kkv2-0-smoke, -1-full-masked, -2-full-unmasked, -3-removal, -4-render-1/2/3, -5-hybrid, -6-shade-prune, -7-hybrid-gate; 45: hybrid-a (bc, trips); 50: full2-trips-resume2, full3-alt, removal-rel, union-broadcast, union-trips. Each training self-delivers a viewer launcher + audit table.
- EXP-0011 full-resolution variant (2026-09-08, branch `exp/kkv2-fullres`, answers Jordan's 19:10 "fuzzy and pixelated" open question): `config_fullres.yaml`/`config_fullres_smoke.yaml` (width 2016, crop 512, seeded from `kkv2-1-full-masked/checkpoints/checkpoint_latest.pt` epoch 122 via `--resume`; resume-across-resolution verified empirically on synthetic fixtures). Queued: `trippy-kkv2-9-fullres-smoke` (prio 15) and `trippy-kkv2-9-fullres` (prio 40, guarded on the smoke's rc). **BLOCKER-IN-WAITING**: both jobs' scripts `cd` into the MAIN checkout, so this branch must be reviewed+merged (or the two config files placed in main) before the smoke reaches the front of its short prio-15 lane (minutes to an hour out), or it fails on a missing-config error. See `experiments/EXP-0011-karekare-v2/README.md` "Full-resolution variant" for the estimate/method.
- Worktrees that MUST stay until their queued jobs finish (job scripts cd into them): .worktrees/karekare-v2 (kkv2-0..6), .worktrees/blend-gate (kkv2-7-hybrid-gate, blend-gate-viewer2/3), .worktrees/live-splat (live-splat-perf-1), .worktrees/edit-sam (edit-sam-1 done: remove). point-removal's jobs are done: remove it. Remove with scripts/worktree_rm.sh.
- Editor Python done through brush regions + named regions (PR #52). Remaining (Rust, after the Hunua window): viewer brush tool, Named Objects panel, drag gizmos. Worktree .worktrees/editor-sam-ui must stay until edit-sam-3 / sam-ui-proof4 (prio 15/12) run. (box/point on the current view -> `trippy edits sam` subprocess -> region), progress + cancel; 3D gizmos still deferred.
- Worktrees that MUST stay until their queued jobs finish (job scripts cd into them): .worktrees/karekare-v2 (kkv2-0..6), .worktrees/blend-gate (kkv2-7-hybrid-gate, blend-gate-viewer2/3), .worktrees/live-splat (live-splat-perf-1). point-removal's jobs are done: remove it. Remove with scripts/worktree_rm.sh.
- feat/edit-sam (large/high): SAM 3 mask lift -> pointset regions (Python; inference via the queue).
- feat/editor-click (mid/high): viewer click-to-select (Rust port of the cluster maths) + 'new region from selection'.
- feat/editor-ui (large/high): Regions panel (box/sphere/lid), per-region mix/op, undo/redo, save/load edits.json, hot-apply weights in both renderers, shade-cloud finder tool (ADR-0007 E1+E2).
- GPU queue (prio 70, filename order): exp0010-removal (running), exp0010-shade-prune, full2-trips-resume, full3-alt-bc, full3-alt, hybrid-a-all-levels-bc, hybrid-a-all-levels, kkv2-0-smoke, kkv2-1-full-masked, kkv2-2-full-unmasked, kkv2-3-removal, kkv2-4-render-1/2/3, kkv2-5-hybrid, removal-rel, union-broadcast, union-trips. Roughly two days of GPU. Each training self-delivers a viewer launcher + audit table.
- Worktrees that MUST stay until their queued jobs finish (job scripts cd into them): .worktrees/karekare-v2 (kkv2-0..6 jobs), .worktrees/blend-gate (kkv2-7-hybrid-gate + blend-gate-viewer2/3), point-removal's jobs are done: remove it. Remove with scripts/worktree_rm.sh.
- feat/edits-publish (mid): apply edits to the TRIPS export + edit-then-distil path (`--target trips|distilled|both`).

## Next (in order)

1. Merge scene-io + points; launch feat/raster (large/high: numpy reference + Metal blend_fwd + pyramid forward) and feat/net (mid/high: U-Net + tone mapper ports) once TRIPS_REFERENCE.md lands.
2. v0.1.0: colmap_io, xform_a/b, dataset, GaussianPlySource, ref_numpy, blend_fwd pyramid forward, U-Net tone mapper.
3. v0.2.0: blend_bwd gradcheck, trainer, MonoDepthSource, eval/export/dolly, source experiments.
4. v0.3.0: Hybrid designs C then A1, comparison harness.
5. v0.4.0: Brush fork viewer (Mac), v0.5.0 web viewer, Quest measurement.
6. Viewer editing: `docs/decisions/ADR-0007-viewer-editing.md` + `docs/EDITOR.md` (architecture written 2026-09-07, docs only); start at E1 (region data model + box/sphere blend + undo/save) once `feat/blend-gate` lands.

## Blocked
- None.

## Open questions for Jordan (review queue; nothing blocks on these)
- **NOT READY YET (2026-09-10): Quest viewer launchers, `4-other/quest-viewer-*.command`.**
  Four launchers (shade-keep and kklid_20000, each preview + on-quest) are delivered and
  will refuse to open with a plain message until their GPU-queue conversion jobs finish
  (see "In flight" above). Nothing to do yet -- they'll work once the jobs land; this row
  can come off the list once `docs/QUEST.md` has real numbers and Jordan has tried the
  headset.
- **REVIEW FIRST (2026-09-08 21:10): the three TRIPS-cleaned splats.** `2-open-in-brush/kklid-tripsclean-shade.ply` (least aggressive), `-005`, `-015`. These are YOUR splat with fog subtracted, not a TRIPS render -- everything that survives is bit-identical to `kklid_20000`. Question for you: is the canopy shade cloud gone, and did anything you wanted disappear with it? The measurable risk is thin/background coverage: 1.07% / 4.49% / 10.74% of covered pixels in the 93 shade views lose their last point.
- **Decision needed: is `depthprior_shade_audit.py`'s `--frames` default wrong everywhere?** `trippy.render.report` never passes `--frames`, so every kkv2 run report measured the six kk-coherent frames instead of the measured 93-frame big-tree list, despite `experiments/EXP-0011-karekare-v2/README.md` requiring the latter for the SPEC stage gate. Fixing it re-bases the whole leaderboard column (untouched splat 17.3% -> 26.71%). Not fixed in `feat/splat-clean` -- out of that branch's file list.
- **A second, independent reading of the exposure effect (2026-09-06).** `IMG_3830` is a held-out shade frame whose EXIF was valid, so none of the exposure fixes touched it -- and its viewer render still goes **12.30 -> 15.46 dB** when tone mapped with the scene median instead of its own never-trained EV. That is the same conclusion `eval-calib-1` reached by fitting the exposure per image (shade 8.49 -> 15.32 dB), reached instead from a screenshot. Two methods, one answer: a held-out frame's exposure is never brought to the scale the U-Net learned on the training frames.
- REVIEW: open `trips-mac-viewer-horse.command` (4-other) to step into the public horse scene rendered live by TRIPS on this Mac; V toggles network/raw/coverage.
- Disk cleanup 2026-09-06 12:15 (Jordan's request): 49 -> 71 GB free. Removed re-downloadable Zenodo zips, the fork's rebuildable target dir, smoke runs, distillation intermediates, and old epoch checkpoints (kept latest/best-so-far/ep0000). Trainer retention policy in progress (fix/ckpt-retention) so 300-epoch runs stop writing ~24 GB each.
- PERF (PR #35): trips mode was never 10x slower; contention was. Vectorised emission makes trips 1.11x broadcast per step. NOTE: that agent ran the GPU-marked pytest suite outside the queue twice (~60 s) before correcting itself; logged.
- EXP-0007 Hunua clip5923_best (broadcast, 120 ep, 4 h): held-out 13.28 dB / SSIM 0.215 / LPIPS 0.555 (no Gaussian baseline number for this clip yet); `full-trips-2-bc-viewer.command` + dolly/honesty delivered.
- FIXED + DELIVERED (16:00): viewer v3 launchers (`full2-broadcast-viewer-v2`, `trips-kk-full1-viewer-v2`, `trips-mac-viewer-horse-v3`): trusted per-view exposure (median substituted for the 10 EXIF-less views), X-cycled exposure modes, 4x speed / 50x scroll, opens on a trusted view. Viewer matched the Python reference at 85.8 dB even on the broken bundle.
- BUG (Jordan 14:00): `full2-broadcast-viewer.command` shows a near-white frame with speckles for Karekare while the Python path renders it at 15 dB -> export/viewer camera-model or f16 path wrong for this scene (fix/viewer-kk, numbers-only parity). Also: default fly speed too slow -> 4x base, 50x scroll range.
- MEASURED (17:30, neighbour-exposure eval, no held-out photo used): full2-broadcast all 17.12 dB / shade 15.27 dB vs Gaussians 15.53 / 14.94 (which trained on 5 of the 6 shade frames). Leaderboard refreshed in Jordan-Review. Dark mass in the shade volume still 36.9% vs 19.9%.
- MEASURED (15:40, eval-calib-1): with held-out exposure calibrated, full2-broadcast scores shade 15.32 dB (Gaussians 14.94), all 17.66 (Gaussians 15.53); the conservative closed-form-gain number is shade 14.59 = level with Gaussians. Shade dark mass unchanged at 36.9% vs 19.9%. So the PSNR story is now neutral-to-positive; the geometry story (dark mass) still favours Gaussians; Jordan's viewer decides.
- STOP-OR-GO SIGNAL, CORRECTED (14:40): the 8.49 dB shade number was mostly an exposure BUG: 10/219 images have no EXIF exposure and were initialised at a 58.5x gain that held-out frames never correct (4 of the 6 shade frames). On EXIF-valid frames full2-broadcast scores 12.37 dB shade / 16.90 all (Gaussians 14.94 / 15.53), so a real ~2.5 dB shade deficit remains, and the Gaussian baseline itself TRAINED on 5 of the 6 shade frames (only IMG_3829 was held out in kkc_15000), so that comparison flatters the Gaussians. Dark mass in the shade volume 36.9% vs 19.9% stands. Fixed (EXIF fallback -> scene mean); `trippy eval --calibrate` and an alternating hold-out run (full3-alt) are queued to get clean numbers. Remaining hopes in the queue: full2-trips, union (more geometry in the shade), hybrid A (Gaussian render as input). Also testing whether held-out exposure calibration and an alternating shade hold-out change the picture (feat/eval-calib).
- REVIEW (auto-delivered): EXP-0003 full2-broadcast (300 epochs): `full2-broadcast-viewer.command` (walk it yourself), -dolly.mp4, -honesty.png. Held-out 15.02 dB (plain Gaussians 15.53). Its audit/leaderboard step failed on a mid-run code update (fixed: eager imports); a re-report job is queued and will refresh the leaderboard with shade dark-mass + shade PSNR.
- E4 viewer Shift-click selection merged (PR #47; exact parity with Python); E5 SAM 3 mask lift merged (PR #48; real run on kk-coherent: 72k points selected from a centre box in 59 s on CPU).
- Editor UI E1+E2 merged (PR #45): M opens the editor (box/sphere/lid regions, mix/op, undo/redo, Cmd-S, shade-cloud finder with live thresholds); delete box proof 64%/undo bit-identical; Python golden parity. E4 click-to-cluster Python merged (PR #46). Release viewer rebuilt in main.
- Publish path merged (PR #44): `trippy apply-edits --target trips|distilled|both`; `--edits` on eval, candidate-report and distill (edit-then-distil). Known gap: eval applies deletions but not the gate multiply.
- Live splat in the viewer merged (PR #43): the Blend panel's splat/gated/mix/split modes now work at ANY pose (brush-render inside trips-viewer; off-view proof 58/255 vs 0 control). Existing Karekare bundles need re-export to carry splat_ply. Limitation: the network's Gaussian input block is still precomputed-only off-view (no per-pixel splat depth from brush-render).
- Editor Python side merged (PR #42): edits.json (box/sphere/lid/pointset regions, undo log), per-point weight composition, shade-cloud finder on the audit rule, `trippy apply-edits` (filters TRIPS points + splat ply, writes blend_weights.npy), `trippy edits add-box/add-sphere/add-lid/shade-find`. Viewer UI for regions follows once live splat lands.
- Blend gate merged (PR #41): explicit per-pixel splat/TRIPS gate + gate_scale + viewer Blend panel (B key: TRIPS/splat/gated/mix/split). Hybrid bundles were never renderable before this (missing Gaussian block) -> kkv2-5-hybrid's bundle must be re-exported after it trains. kkv2-7-hybrid-gate queued (temporarily at 70 so the Splats item stays first; the reorder watcher moves it).
- EXP-0003 full3-alt-bc (alternating shade hold-out, broadcast, stopped at ep 245 on its 4 h budget): all 17.94 / shade (3 held-out frames) 16.79 / other 18.07 (neighbour exposure); dark mass 36.7% (no prune). Highest shade PSNR so far -> when TRIPS sees half the shade frames it interpolates the rest well. Launcher `full3-alt-bc-viewer.command`.
- v0.5.0 RELEASED (first candidate beating the Gaussian baseline on PSNR). kkv2-6-shade-prune queued (shade prune on the 93 measured big-tree frames) after the other karekare-v2 arms.
- REVIEW FIRST: EXP-0010 arm B (TRIPS rule + audit-aligned shade prune, 300 ep): all 17.75 / shade 15.59 dB (neighbour exposure; best so far) AND shade dark mass 24.1% (was 36.9%; Gaussians 19.9%). Held-out shade PSNR went UP while dark mass went down, so the pruned points were not carrying signal. `exp0010-shade-prune-viewer.command` is the candidate to walk (kk-coherent scene; the big-tree scene comes with EXP-0011). Caveat: shade_prune removes what the audit measures; the PSNR is the independent check.
- EXP-0010 arm A (TRIPS point-removal rule, 300 ep, 3 h): all 17.67 / shade 15.44 dB (neighbour exposure); dark mass 36.8% -> the TRIPS rule alone does not move the audit. `exp0010-removal-viewer.command` delivered. Arm B (shade-prune probe) running.
- EXP-0011 queued (PR #39): masks in loss+eval; measured shade = 93 frames at the big-tree spot (kk-coherent's six shade frames are a different place 5.8 units away, reported separately). Full masked/unmasked runs stop at ~epoch 150 on their 7 h budgets (resumable).
- SAM tool in the viewer merged (PR #50): box/point prompt on a capture view runs SAM 3 locally (CPU ~9 s/view) and imports the object as an undoable region; MPS fixes applied, timing pending (edit-sam-3 queued).
- 15:40: Splats' queue emptied, so the runner started one of OUR parked-at-70 jobs; killed it and moved ALL trippy queue files out of the queue dir into gpu_queue/parked-trippy/ for the window. The restore script moves them back with priorities.
- 03:20 2026-09-08: Hunua window over; queue restored (kkv2 40, hybrids 45, rest 50, SAM 15) and kkv2-1-full-masked resumed from epoch 20 (running, ~6.5 h): priorities back to 40/45/50, SAM jobs to 15, kkv2-1 resumed from checkpoint_latest.
- Jordan 15:25: Hunua splat queue has priority for 6 h. kkv2-1-full-masked stopped at epoch 20 (checkpoint kept, 15.14 dB); all trippy jobs parked at prio 70 until ~21:30, then restored (kkv2 40, hybrids 45, rest 50) and kkv2-1 resumed. Only CPU-light Python work meanwhile (cpu_heavy.sh refuses builds under 28 GB free anyway).
- EXP-0011 smoke passed (14:30); kkv2-1-full-masked (the run Jordan is waiting for) started 14:35, ~7 h; then unmasked, removal, renders, hybrid, shade-prune, hybrid-gate. Baseline dark mass on the measured big-tree region: 17.3% (kklid_20000).
- Jordan 2026-09-07 13:40: GO on all five editor items. Done: per-region mix (box/sphere/lid), shade finder, Shift-click selection, SAM lift (CLI), lid, undo/save/load/publish. Remaining, in order: in-viewer SAM prompt (feat/editor-sam-ui), brushed volumes + drag gizmos, named-objects panel (toggle SAM/click selections by name).
- Jordan 2026-09-07 09:45: asked for a full editing GUI: per-object/per-region splat-vs-TRIPS mix, smart object selection + masking, the Karekare pool LID as an edit, and find-and-obliterate shade clouds. Assessed as feasible in stages (ADR-0007 to follow): per-region 3D blend volumes on top of the gate (1 wk), shade-cloud selector from the audit rule (days), object selection via depth/colour clustering then SAM 3 mask lifting (2-3 wks), lid as a constrained region (days), full editor with undo/save/publish (ongoing). Starts after feat/blend-gate lands.
- Jordan 2026-09-07 09:30: the shade problem lives in the FULL Karekare set (kk-coherent cannot show it), but the other floating clouds already improved in TRIPS; expects the TRIPS+splat combination to be the best result and asked which levers set the splat/TRIPS mix -> feat/blend-gate adds an explicit per-pixel gate + gate_scale + viewer Blend panel (splat-only / TRIPS-only / gated / split-screen, mix slider); config_hybrid_gate.yaml queued after kkv2-6.
- Jordan 2026-09-07 08:20: after the last Splats item (60-hunua-run01) runs, trippy manages the GPU queue for the rest of the project. Watcher renames queued trippy jobs once that item starts: kkv2-* -> prio 40 (target scene first: smoke, masked, unmasked, removal, render shards, hybrid, shade-prune), hybrid-a -> 45, the remaining kk-coherent runs -> 50. New submissions may use these bands.
- Jordan 2026-09-06 18:40: viewer works now. The real target is the FULL Karekare scene (karekare-v2, 756 registered images, seeded from kklid_20000.ply): the big tree you walk under toward the pool, and its shade, are in that set and NOT in kk-coherent, so all TRIPS runs so far never saw it. EXP-0011 (feat/karekare-v2) trains on it with person masks and alternating shade hold-out; hybrid-A on the same scene follows. Jordan 18:50-19:00: do NOT jump the queue (prio 70, not 65). Masks are a COMPARISON VARIABLE, not a rule (training on the kids is fine): run masked AND unmasked variants; the splat was masked, so the masked variant is the fair comparison, and the unmasked one is what Jordan is curious about. kk-coherent person masks being generated (feat/kk-masks) for masked siblings of the existing unmasked runs. Jordan wants to see the two combined.
- Jordan 2026-09-06 07:20: EXP-0003 first candidate is "a good starting point"; the dolly direction is not one he cares about (hard to compare) -> deliver Karekare bundles for the free-navigation viewer from now on; his main interest is Splats combined with TRIPS (hybrid). No queue jump wanted; keep prio 70.
- LOST ARTEFACT: EXP-0005's Gaussian renders + design-C checkpoint were inside a removed worktree (before output routing was fixed); the dangling review links (EXP-0005 sheet, stock web viewer) were removed; their README rows remain as history. Numbers survive in the README/research log; renders are being regenerated for EXP-0009. Guard added: scripts/worktree_rm.sh.
- PRIVACY INCIDENT (2026-09-05 ~23:40): the render-kk subagent opened `output/runs/EXP-0001/.../sheet.png` (a CPU dry-run contact sheet whose first panel is a kk-coherent photo) with its Read tool to sanity-check layout. Image pixels went to the model API. My task brief caused it ("look at the coverage image with the Read tool"). Fixed: AGENTS.md now forbids viewing any scene-derived imagery; all running agents were told. Nothing else left the machine. Please decide whether you want any further action.
