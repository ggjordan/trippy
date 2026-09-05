# TRIPS on Apple Silicon: the plan of record

## Context

Jordan has Gaussian splats of outdoor family scenes (Karekare, Hunua, etc) with one persistent defect: under trees, the shade becomes a dark cloud—an object you have to move through—rather than a lighting effect. Five training-side fixes failed to remove it. On the TRIPS paper's own side-by-sides, 3DGS shows this exact defect and TRIPS does not.

trippy is a new project that:
1. Ports TRIPS (Trilinear Point Splatting, MIT) to this Mac Studio (M3 Ultra, 96 GB unified, Metal)
2. Trains it on Jordan's Karekare and Hunua scenes
3. Tries three point-source strategies: pure Gaussians, monocular depth, their union
4. Hybridises the winner with Gaussians (keeps what Gaussians do well; uses TRIPS where they fail)
5. Ships live viewers: Mac native (Brush), web (WebGPU), Quest assessed honestly

If all three point sources fail to remove the shade, we write the negative plainly and do not ship.

## Decisions locked 2026-09-05

| # | Decision |
|---|---|
| D1 | Targets per brief: Mac viewer, desktop web viewer, Quest measured honestly. No iPhone app. |
| D2 | Goal: kill the shade cloud. A plain splat that incorporates TRIPS learning (Design B) is a valid fallback path, not the primary deliverable. |
| D3 | No separate Stage 0 gate. Forward rasteriser is the first checkpoint of the training port. No author-scene fps report unless free. |
| D4 | Point sources are pluggable: (1) trained Gaussian centres, (2) monocular depth (DepthPro/MoGe), (3) union, (4) Revisit iPhone LiDAR later. Run 1/2/3 as experiments. Nothing to add to Revisit. |
| D5 | Governance copied from `~/revisit/AGENTS.md`: Architect+Reviewer+Orchestrator, lower-tier subagents, retry ladder effort↑ → model↑ → both↑ → stop and ask, subagents never commit/push, ADHD output shaping on every reply, STATE.md, ADRs. |
| D6 | Option 3: Python research trainer first, then port winning design into a fresh Brush fork (Rust/Burn/CubeCL) for live rendering. Jordan worries distilled splats re-introduce splat artefacts. |
| D7 | Python host = PyTorch 2.13 on MPS + inline Metal kernels via `torch.mps.compile_shader` (verified). Not MLX. |
| D8 | Separate repo `ggjordan/trippy` (public, see D12), own uv venv (Python 3.13), own `~/trippy/output/`. Reads `~/Splats` read-only via `SPLATS_ROOT`. GPU via Splats' queue; delivery via Splats' `review_add.sh`. Never copy scenes/plys. |
| D9 | GPU priority: trippy short jobs prio 10-19, trainings prio 70 (behind Splats' 60). Jordan can re-prioritise by asking. |
| D10 | Verdict = Jordan's eyes in a viewer. Metrics (shade audit, held-out LPIPS/PSNR, extent gate) rank candidates. Every candidate ships viewer artifact + audit number + dolly video + off-path honesty sheet. |
| D11 | `build-NNNN` tag every push; vX.Y.Z GitHub Release per milestone (below). |
| D12 | **Public repo from day one, MIT licence.** TRIPS is MIT, Brush fork stays Apache-2.0 with notices, Zenodo data (CC-BY) linked not redistributed. Guardrails from the first commit: pre-commit hook blocks images/video/ply/checkpoints/files >5 MB; AGENTS.md forbids any render of Jordan's scenes in the repo (deliverables live in Splats' review folder, outside the repo); test fixtures synthetic only; a public Zenodo TRIPS scene is the reproducible example for outsiders. Place names OK, faces never. |

## Milestones → releases

| Release | Content | Acceptance | Jordan opens | Est. |
|---|---|---|---|---|
| build-0001/2 | Phase 1 skeleton + first queue round-trip smoke job | tests green, repo on GitHub, `.rc`=0 | nothing required | 2-3 h |
| v0.1.0 | colmap_io, xform_a/b, dataset (undistort+cache 1008/2016 wide), GaussianPlySource, ref_numpy, `blend_fwd` + pyramid forward, U-Net/tone-mapper ports. Stretch (half day): load authors' checkpoint via `torch.jit.load` | CPU tests green; Metal = reference; 3 kk frames at 1008 wide <100 ms | contact sheet: photo / ref / Metal / 5 pyramid levels | 3-4 d |
| v0.2.0 | `blend_bwd` + gradcheck (large/high), trainer (crops 384-512 half-res, 150-250 epochs, pose refine after lock), MonoDepthSource, eval/export/dolly/offpath/video. Experiments: sources 1/2/3 on kk-coherent, source 1 on clip4982 | held-out PSNR within 1.5 dB of best Gaussian on non-shade frames; shade dolly shows shading; audit drops vs kkc_15000; extent not inflated; honesty sheet reviewed | fly-through MP4 through the shade + exported PLY in Brush ("raw points, no net") | 5-7 d + 2-3 overnight trainings |
| v0.3.0 | Hybrid C then A1; comparison harness; B design note | hybrid beats best plain Gaussian on shade audit AND extent gate; LPIPS not worse | quad dolly video (Gaussian / TRIPS / C / A1) + A1 PLY | 5-8 d |
| v0.4.0 | Brush fork `rust/brush-trips`: `brush-pyramid` (CubeCL emit, 2× radix_argsort, prefix_sum, blend_fwd), `brush-unet` (Burn conv2d; weights via safetensors, converter like `lpips-convert`), viewer hookup in `apps/brush-app/src/ui/splat_backbuffer.rs`; parity test vs PyTorch | loads kk scene; 1080p ≥20 fps (target 30); parity <1e-3 | `OPEN_TRIPS_MAC.command` launching the viewer | 6-10 d |
| v0.5.0 | Web viewer via `apps/brush-app/web` (wasm-pack + vite), `OPEN_TRIPS_WEB.command` on 127.0.0.1 | ≥15 fps 1080p in Chrome on the Mac; loopback only | double-click `.command` | 3-5 d |
| Quest note | measure web viewer on Quest browser once; ship fallback: distilled Gaussians via `~/Splats/tools/publish/publish_splat.sh` + `tools/flythrough.py` videos | honest fps number in release notes | distilled `.ply`/SOG + video | 1-2 d |
| v1.0.0 | Jordan says the clouds are gone | his verdict | — | — |

Total ≈ 23–36 agent-days; realistic calendar 6–9 weeks with queue contention.

## Stop-or-go point: v0.2.0

At the end of v0.2.0, three experiments will have run: point sources 1 (Gaussians), 2 (monocular depth), and 3 (union) trained on kk-coherent. The shade audit will show which—if any—removes the cloud. If all three leave the shade untouched, write the negative plainly, close the repo, and explore other approaches. **This gate is binding.** Do not push on to hybrid if the cloud persists.

## Quest honesty note

The web viewer targets ≥15 fps at 1080p in Chrome on the Mac. Quest browser is mobile GPU. At v0.5.0, measure the web viewer on a Quest once and report the frame rate honestly. A CNN per eye per frame at headset resolution is probably beyond mobile budget (~120 GFLOP/eye/frame). If it is, ship a fallback: distilled Gaussians via the existing `~/Splats/tools/publish/` path, or fly-through MP4 videos. Do not promise interactive Quest performance.
