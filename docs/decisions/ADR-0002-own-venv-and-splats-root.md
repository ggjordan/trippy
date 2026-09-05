# ADR-0002: Own uv venv and read-only SPLATS_ROOT path

Date: 2026-09-05 · Status: Accepted

## Context

The trippy project must coexist with the Splats project on the same Mac. Both use Python (PyTorch), share access to the GPU queue, and consume significant disk space. Three isolation strategies exist:

1. **Shared venv**: use `~/Splats/tools/ml-sharp/.venv` for both projects. Saves disk; risks dependency conflicts.
2. **Isolated venv in trippy**: create `~/trippy/.venv` with trippy's own dependencies. Disk cost; complete isolation.
3. **SPLATS_ROOT environment variable**: trippy gets its own venv; reads Splats data via a `SPLATS_ROOT` environment variable pointing to `~/Splats` (read-only).

## Decision

Adopt **Option 3**: trippy has its own `uv venv` (Python 3.13), stores it in `~/.venv`, and reads Splats data via `SPLATS_ROOT=~/Splats` (enforced read-only).

## Consequences

### Isolation

- **No dependency conflicts**: trippy can pin PyTorch 2.13, torchvision, lpips independently; Splats can upgrade without affecting trippy.
- **Clean teardown**: removing `~/.venv` uninstalls trippy; no effect on Splats.
- **Separate queue**: trippy jobs run under the same GPU queue as Splats (via `scripts/gpu_submit.sh`) but with independent priorities (trippy prio 10–19 / 70, Splats prio 60).

### Data sharing

- **Read-only access**: `SPLATS_ROOT` points to Splats' scenes (`kk-coherent`, `clip4982`), tools (`depthprior_shade_audit.py`, `gsrender.py`), and reference meshes. trippy never modifies Splats data.
- **Delivery isolation**: trippy exports results via `scripts/deliver.sh` → `~/Splats/output/Jordan-Review/`, not to `~/trippy/output/`. This keeps Jordan's deliverables in one place (the Splats review folder) and prevents the trippy repo from containing private scene renders.

### Disk management

- **Status quo**: as of 2026-09-05, the Mac is 93% full. Splats is stable (~70 GB), growing slowly. Adding trippy training outputs and MLX/Burn builds must not overflow.
- **Guardrails**: GPU queue enforces a 28 GB free-memory guard before accepting jobs. Long-term: `output/runs/` and `output/deliver/` are gitignored and cleaned by the orchestrator before release.

## Why not shared venv?

- **Dependency fragility**: Splats might upgrade to PyTorch 2.14 for a fix unrelated to trippy. trippy's Metal kernels might break.
- **Queue conflicts**: simultaneous Splats and trippy jobs under shared venv make debugging harder.
- **Blame assignment**: if a tool in Splats' venv breaks, is it Splats' bug or trippy's?

## Why not just use ~/Splats directly?

- **Splats is stable**: Splats' venv is tuned for production Gaussian training. trippy's research code (frequent refactors, new dependencies) will destabilise it.
- **Replay and audit**: trippy's independent venv means `uv.lock` fully specifies all dependencies at a given commit, making replay and audit easier.

## Related

- `SPLATS_ROOT` is checked by `scripts/bootstrap.sh`; if not set or `~/Splats/tools/submit.sh` is missing, bootstrap fails.
- Environment variables for GPU priority and output paths are documented in `.env.example`.
- `scripts/bootstrap.sh` checks disk free-space before allowing `uv sync`.
