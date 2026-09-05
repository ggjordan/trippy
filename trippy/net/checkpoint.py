"""Loading (best-effort) TRIPS's own libtorch checkpoints into trippy's ported modules.

Module: trippy.net.checkpoint
Invariants: never raises on a malformed/foreign checkpoint file -- always
    returns a `CheckpointLoadResult` describing what happened, so callers
    (and tests) can branch on `.ok` instead of catching exceptions.
Related docs: docs/TRIPS_REFERENCE.md Sec. 9 (checkpoint layout, verified
    against src/apps/train.cpp, src/lib/models/{Pipeline,NeuralCamera,
    NeuralScene}.{h,cpp}); docs/LIMITATIONS.md (whether this loader has
    ever actually been exercised against a real TRIPS checkpoint).

-- TRIPS checkpoint layout (docs/TRIPS_REFERENCE.md Sec. 9) --
Per-epoch directory `experiments/<experiment_name>/ep<NNNN>/` (renamed to
`experiments/_f_<experiment_name>/` on completion, train.cpp:800-802). Inside each
`ep<NNNN>/`, saved via plain `torch::save`/`torch::load` on live `torch::nn::Module`s
(**not** `torch.jit.ScriptModule`s -- there is no traced graph to load, only a named-tensor
archive that assumes the C++ module structure is reconstructed independently first):
    render_net.pth                         -- the MultiScaleUnet2dDecOnlySmallFixed instance
    dynamic_refinement_module.pth          -- only if enabled (not ported here)
    scene_<scene>_texture.pth              -- NeuralPointTexture (skipped if reduced_check_point)
    scene_<scene>_points.pth               -- NeuralPointCloudCuda (skipped if reduced_check_point)
    scene_<scene>_env.pth                  -- environment map (only if enabled; off by default)
    scene_<scene>_poses.pth / _intrinsics.pth   -- always saved
    scene_<scene>_poses.txt                -- human-readable pose re-export (xyzw quats)
    scene_<scene>_vignette.pth, _response.pth (+.png/.csv), _wb.pth (+.txt), _ex.pth (+.txt)
                                            -- from NeuralCameraImpl::SaveCheckpoint
Alongside per-epoch dirs: log.txt, error.txt, params.ini (full resolved config), git.txt,
tfevents.pb. `render_net.pth` is the only file this module's loader targets (the U-Net).

-- Why this loader can only be best-effort --
A libtorch `torch::save(nn_module, path)` archive is a `torch::serialize::OutputArchive` zip
container keyed by the C++ module's `named_parameters()`/`named_buffers()` strings (e.g.
"start.conv.feature_transform.0.weight"). This is a *different* container format from a
TorchScript `torch.jit.ScriptModule` export (which additionally embeds a traced/scripted
graph) and from a plain Python `torch.save(state_dict)` pickle. Python's `torch.jit.load`
will only succeed if the file happens to be a real ScriptModule export; a bare OutputArchive
of a C++ `nn::Module` is not guaranteed to be. Rather than assume which one applies, this
loader tries both readers in turn and reports exactly what it observed, per the task brief.
When a reader succeeds, parameters are mapped into the target module *by shape-matched
ordered position* (not by name -- the naming scheme intentionally is not assumed to match
trippy's own module names), since that is the only mapping possible without also porting
TRIPS's exact C++ module-registration order byte-for-byte (out of scope here; flagged in
docs/LIMITATIONS.md as unverified until tried against a real checkpoint).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn


@dataclass
class CheckpointLoadResult:
    """Outcome of `try_load_trips_network`.

    Attributes:
        ok: True if a checkpoint file was read AND every one of its
            tensors was assigned into the target module by matching
            shape, in order.
        reader: which reader succeeded reading the raw file ("jit",
            "torch_load", or None if neither worked).
        num_tensors_found: how many tensors were extracted from the file
            (0 if reading failed entirely).
        num_tensors_assigned: how many were copied into the target
            module's parameters (<= num_tensors_found and
            <= number of target parameters).
        report: human-readable, printable summary (also printed to
            stdout as it's built, per the task brief).
        reason: why `ok` is False, if it is.
    """

    ok: bool
    reader: str | None
    num_tensors_found: int
    num_tensors_assigned: int
    report: str
    reason: str | None = None
    tensor_shapes_found: list[tuple[str, torch.Size]] = field(default_factory=list)


def _flatten_state(obj: object, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    """Recursively flatten whatever `torch.jit.load`/`torch.load` handed back into (name, tensor) pairs."""
    out: list[tuple[str, torch.Tensor]] = []
    if isinstance(obj, torch.Tensor):
        out.append((prefix or "tensor", obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_flatten_state(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, nn.Module):
        for name, tensor in obj.state_dict().items():
            out.append((f"{prefix}.{name}" if prefix else name, tensor))
    elif hasattr(obj, "state_dict"):
        # torch.jit.ScriptModule and similar expose state_dict() without being nn.Module.
        for name, tensor in obj.state_dict().items():
            out.append((f"{prefix}.{name}" if prefix else name, tensor))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.extend(_flatten_state(v, f"{prefix}[{i}]"))
    return out


def _try_read(path: Path) -> tuple[str | None, list[tuple[str, torch.Tensor]], str | None]:
    """Try torch.jit.load then torch.load(weights_only=False). Returns (reader, tensors, error)."""
    try:
        module = torch.jit.load(str(path), map_location="cpu")
        tensors = _flatten_state(module)
        if tensors:
            return "jit", tensors, None
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any foreign-format failure is expected
        jit_error = f"torch.jit.load failed: {exc!r}"
    else:
        jit_error = "torch.jit.load succeeded but yielded no tensors"

    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        tensors = _flatten_state(obj)
        if tensors:
            return "torch_load", tensors, None
        load_error = "torch.load succeeded but yielded no tensors"
    except Exception as exc:  # noqa: BLE001 -- see above
        load_error = f"torch.load(weights_only=False) failed: {exc!r}"

    return None, [], f"{jit_error}; {load_error}"


def try_load_trips_network(path: str | Path, target: nn.Module | None = None) -> CheckpointLoadResult:
    """Best-effort load of a TRIPS `render_net.pth` (or similar) checkpoint file.

    Args:
        path: path to the checkpoint file (e.g.
            `experiments/<name>/ep0599/render_net.pth`).
        target: module to receive shape-matched parameters (e.g. a
            `MultiScaleUnet2dDecOnlySmallFixed`). If None, only reads and
            reports what is in the file without assigning anything.

    Returns:
        A CheckpointLoadResult; see its docstring. Never raises.
    """
    path = Path(path)
    lines: list[str] = [f"try_load_trips_network({path})"]

    if not path.exists():
        reason = f"file does not exist: {path}"
        lines.append(reason)
        report = "\n".join(lines)
        print(report)
        return CheckpointLoadResult(
            ok=False, reader=None, num_tensors_found=0, num_tensors_assigned=0, report=report, reason=reason
        )

    reader, tensors, error = _try_read(path)
    if reader is None:
        lines.append(f"FAILED to read as either format: {error}")
        report = "\n".join(lines)
        print(report)
        return CheckpointLoadResult(
            ok=False, reader=None, num_tensors_found=0, num_tensors_assigned=0, report=report, reason=error
        )

    lines.append(f"read OK via {reader}: {len(tensors)} tensors")
    for name, t in tensors[:20]:
        lines.append(f"  {name}: {tuple(t.shape)} {t.dtype}")
    if len(tensors) > 20:
        lines.append(f"  ... and {len(tensors) - 20} more")

    tensor_shapes = [(name, t.shape) for name, t in tensors]

    if target is None:
        lines.append("no target module given -- read-only report")
        report = "\n".join(lines)
        print(report)
        return CheckpointLoadResult(
            ok=True,
            reader=reader,
            num_tensors_found=len(tensors),
            num_tensors_assigned=0,
            report=report,
            tensor_shapes_found=tensor_shapes,
        )

    target_params = list(target.state_dict().items())
    lines.append(f"target module has {len(target_params)} tensors")

    # Shape-matched ordered assignment: walk both sequences in file order / registration
    # order and copy wherever the shape matches; skip (and report) mismatches rather than
    # guessing a reshape or transpose.
    assigned = 0
    target_state = target.state_dict()
    src_by_pos = list(tensors)
    tgt_names = [name for name, _ in target_params]
    for i, (tgt_name, tgt_tensor) in enumerate(target_params):
        if i >= len(src_by_pos):
            lines.append(f"  no more source tensors for target[{i}] {tgt_name} {tuple(tgt_tensor.shape)}")
            continue
        src_name, src_tensor = src_by_pos[i]
        if tuple(src_tensor.shape) != tuple(tgt_tensor.shape):
            lines.append(
                f"  SKIP target[{i}] {tgt_name} {tuple(tgt_tensor.shape)} <- {src_name} "
                f"{tuple(src_tensor.shape)} (shape mismatch)"
            )
            continue
        target_state[tgt_name] = src_tensor.to(dtype=tgt_tensor.dtype)
        assigned += 1

    target.load_state_dict(target_state)
    ok = assigned == len(target_params) and assigned > 0
    lines.append(
        f"assigned {assigned}/{len(target_params)} target tensors (target names: {tgt_names[:5]}...)"
    )
    report = "\n".join(lines)
    print(report)
    return CheckpointLoadResult(
        ok=ok,
        reader=reader,
        num_tensors_found=len(tensors),
        num_tensors_assigned=assigned,
        report=report,
        reason=None if ok else "not every target tensor found a shape match",
        tensor_shapes_found=tensor_shapes,
    )
