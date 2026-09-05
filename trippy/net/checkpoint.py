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

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from trippy.constants import (
    TRIPS_CKPT_SCENE_FILES,
    TRIPS_CONFIDENCE_SIGMOID_SCALE,
    TRIPS_INTRINSICS_ROW_LEN,
    TRIPS_SE3_DOUBLES_PER_POSE,
    TRIPS_SOFTPLUS_BETA,
    TRIPS_SOFTPLUS_THRESHOLD,
)

if TYPE_CHECKING:  # pragma: no cover -- import cycle guard only
    from trippy.net.camera_model import NeuralCamera


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


# ---------------------------------------------------------------------------
# Scene-state checkpoints (points / texture / poses / camera)
#
# Everything below was added by feat/adop-parity (EXP-0002) once the public
# Zenodo Tanks & Temples checkpoints were on disk, so every mapping here is
# verified against a real file, not inferred. See docs/TRIPS_REFERENCE.md
# Sec. 9b for the field-by-field table and docs/LIMITATIONS.md for what is
# still unverified.
#
# Observed contents of checkpoint_horse/ep0600 (2,218,471 points, 151 frames):
#   scene_tt_horse_points.pth   t_position [N,4]  t_point_size [N,1]
#                               t_index [N,1] i32 t_original_color [N,4]
#   scene_tt_horse_texture.pth  texture [C,N]     background_color [C]
#                               confidence_value_of_point [1,N]
#   scene_tt_horse_poses.pth    tangent_poses [M,6] f64  poses_se3 [M,8] f64
#   scene_tt_horse_intrinsics.pth  intrinsics [num_cameras,13]
#   scene_tt_horse_ex.pth       "0" [M,1,1,1]        (exposure)
#   scene_tt_horse_wb.pth       "0" [M,3,1,1]        (white balance)
#   scene_tt_horse_response.pth response [1,3,1,25]
#   scene_tt_horse_vignette.pth vignette_params [3] vignette_center [1,2,1,1]
#
# NAMING TRAP: NeuralPointTextureImpl registers its *raw* parameters under the
# names of the derived tensors (NeuralTexture.cpp:52-55):
#     register_parameter("texture", texture_raw)
#     register_parameter("background_color", background_color_raw)
#     register_parameter("confidence_value_of_point", confidence_raw)
# so the file's "confidence_value_of_point" is pre-sigmoid (the horse file
# ranges [-0.42, 1.17], impossible for a sigmoid output) and its "texture" is
# pre-abs (ranges [-107.6, 95.9]). Use `.confidence()` / `.texture()` below,
# never the raw fields, when feeding the rasteriser.
# ---------------------------------------------------------------------------


def read_module_tensors(path: str | Path) -> dict[str, torch.Tensor]:
    """Read every named tensor out of one libtorch module archive.

    Thin wrapper over the same `torch.jit.load` -> `torch.load` ladder
    `try_load_trips_network` uses (docs/TRIPS_REFERENCE.md Sec. 9a), but
    returning a plain name -> tensor dict instead of a load report.

    Args:
        path: path to a `torch::save(nn::Module)` archive.

    Returns:
        Mapping of the C++ module's registered parameter/buffer names to
        CPU tensors.

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError: if neither reader could extract any tensor.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    reader, tensors, error = _try_read(path)
    if reader is None:
        raise ValueError(f"could not read {path} as a torch module archive: {error}")
    return dict(tensors)


def trips_confidence(confidence_raw: torch.Tensor, narrowing: float = 0.0) -> torch.Tensor:
    """Raw confidence parameter -> the alpha multiplier the rasteriser uses.

    `sigmoid((10 + narrowing) * confidence_raw)`
    (NeuralTexture.h:38-42; docs/TRIPS_REFERENCE.md Sec. 2). `narrowing`
    is `sigmoid_narrowing_factor * epoch`, 0 in every shipped config.
    """
    return torch.sigmoid((TRIPS_CONFIDENCE_SIGMOID_SCALE + narrowing) * confidence_raw)


def trips_point_size(point_size_raw: torch.Tensor) -> torch.Tensor:
    """Raw point-size parameter -> world-unit size: `softplus(raw)`.

    `_softplus(t_point_size)` with beta 1 / threshold 20
    (RenderForward.cu:154, NeuralPointCloudCuda.cpp:19-24). It exactly
    inverts the `inverse_softplus(0.5 * knn_radius)` initialisation.
    """
    return nn.functional.softplus(point_size_raw, beta=TRIPS_SOFTPLUS_BETA, threshold=TRIPS_SOFTPLUS_THRESHOLD)


@dataclass
class TripsPoints:
    """`scene_<scene>_points.pth` = a `NeuralPointCloudCuda` module.

    Attributes:
        position: (N, 3) float32 world positions (`t_position[:, :3]`).
        dropout_radius: (N,) float32 (`t_position[:, 3]`); only read when
            `drop_out_points_by_radius` is on, which it is not in any
            published config.
        point_size_raw: (N,) float32 pre-softplus size; use
            `trips_point_size()`.
        index: (N,) int64 render-order-id -> texture-column map
            (`t_index`). Identity in the published horse checkpoint.
    """

    position: torch.Tensor
    dropout_radius: torch.Tensor
    point_size_raw: torch.Tensor
    index: torch.Tensor

    def __len__(self) -> int:
        return int(self.position.shape[0])

    def size(self) -> torch.Tensor:
        """(N,) world-unit point size, `softplus(point_size_raw)`."""
        return trips_point_size(self.point_size_raw)


@dataclass
class TripsTexture:
    """`scene_<scene>_texture.pth` = a `NeuralPointTexture` module.

    Attributes:
        texture_raw: (C, N) float32, pre-`abs()`.
        background_color_raw: (C,) float32, pre-`abs()`.
        confidence_raw: (N,) float32, pre-sigmoid.
        non_subzero_texture: mirrors `pipeline_params.non_subzero_texture`.

    CORRECTION to docs/TRIPS_REFERENCE.md Sec. 2 (see Sec. 2a): that section
    says "the `abs` flag is `!non_subzero_texture` in the pipeline (config
    `non_subzero_texture=false` -> texture is abs'd)". The call site passes
    the flag **straight through**, not negated::

        scene.texture->PrepareTexture(params->pipeline_params.non_subzero_texture);
            -- third_party/TRIPS/src/lib/models/Pipeline.cpp:257
               (same at data/NeuralScene.cpp:1292)

    and `PrepareTexture(bool abs)` only takes `abs()` when its argument is
    true (NeuralTexture.h:44-57). So with `non_subzero_texture = false` --
    the value in every shipped config *and* in the published Tanks & Temples
    checkpoints -- the texture and background colour are used **raw, with
    their negative values intact**. Taking `abs()` roughly triples the
    composited feature magnitude and blows the U-Net's output past the
    response LUT's [0, 1] domain (measured on tt_horse: PSNR 8.5 dB with
    `abs()`, ~26 dB without).
    """

    texture_raw: torch.Tensor
    background_color_raw: torch.Tensor
    confidence_raw: torch.Tensor
    non_subzero_texture: bool = False

    def texture(self) -> torch.Tensor:
        """(N, C) per-point features, ready for `render_pyramid(feat=...)`."""
        tex = torch.abs(self.texture_raw) if self.non_subzero_texture else self.texture_raw
        return tex.transpose(0, 1).contiguous()

    def background_color(self) -> torch.Tensor:
        """(C,) background feature, composited as `t_final * bg`."""
        bg = self.background_color_raw
        return torch.abs(bg) if self.non_subzero_texture else bg

    def confidence(self, narrowing: float = 0.0) -> torch.Tensor:
        """(N,) post-sigmoid confidence, ready for `render_pyramid(conf=...)`."""
        return trips_confidence(self.confidence_raw, narrowing)


@dataclass
class TripsCameraState:
    """The four `NeuralCameraImpl::SaveCheckpoint` files, already unpacked.

    Attributes:
        exposure: (M,) float32 per-frame EV; applied as `x * 2**-exposure`.
        white_balance: (M, 3) float32 per-frame gains.
        response: (1, 3, 1, P) float32 LUT control points, or None.
        vignette_params: (3,) float32 radial polynomial, or None.
        vignette_center: (1, 2, 1, 1) float32 uv centre, or None.
    """

    exposure: torch.Tensor
    white_balance: torch.Tensor
    response: torch.Tensor | None
    vignette_params: torch.Tensor | None
    vignette_center: torch.Tensor | None


@dataclass
class TripsSceneCheckpoint:
    """Everything an `ep<NNNN>/` directory holds about one scene.

    Attributes:
        epoch_dir: the directory it was read from.
        scene_name: the `scene_<name>_*.pth` infix.
        points, texture, camera: see the dataclasses above.
        poses_w2c: (M, 7) float64 `[qx, qy, qz, qw, tx, ty, tz]`
            **world-to-camera** (PoseModuleImpl stores
            `frame.pose.inverse()`, NeuralStructure.cpp:20-33). Note the
            xyzw quaternion order, matching the on-disk ADOP convention;
            `trippy.scene.adop_io.quat_xyzw_to_wxyz` converts.
        intrinsics: (num_cameras, 13) float32 `fx fy cx cy s` + 8
            distortion coefficients, or None.
    """

    epoch_dir: Path
    scene_name: str
    points: TripsPoints
    texture: TripsTexture
    camera: TripsCameraState
    poses_w2c: torch.Tensor | None
    intrinsics: torch.Tensor | None

    def num_frames(self) -> int:
        return int(self.camera.exposure.shape[0])


def _first_tensor(tensors: dict[str, torch.Tensor], *names: str) -> torch.Tensor | None:
    for name in names:
        if name in tensors:
            return tensors[name]
    return None


def load_trips_scene_checkpoint(
    epoch_dir: str | Path, scene_name: str, non_subzero_texture: bool = False
) -> TripsSceneCheckpoint:
    """Load the point cloud / texture / poses / camera state of one epoch dir.

    Args:
        epoch_dir: e.g. `checkpoint_horse/ep0600`.
        scene_name: the infix in `scene_<name>_points.pth` (e.g. `tt_horse`;
            it is `train_params.scene_names` in the checkpoint's params.ini).
        non_subzero_texture: mirrors `pipeline_params.non_subzero_texture`;
            see `TripsTexture` (the published checkpoints all use `false`).

    Returns:
        A `TripsSceneCheckpoint`.

    Raises:
        FileNotFoundError: if the points or texture file is missing (they
            are only absent when the run set `reduced_check_point=true`,
            which the published checkpoints do not).
    """
    epoch_dir = Path(epoch_dir)

    def _path(kind: str) -> Path:
        return epoch_dir / TRIPS_CKPT_SCENE_FILES[kind].format(scene=scene_name)

    pt = read_module_tensors(_path("points"))
    position4 = pt["t_position"]
    points = TripsPoints(
        position=position4[:, :3].contiguous().float(),
        dropout_radius=position4[:, 3].contiguous().float(),
        point_size_raw=pt["t_point_size"].reshape(-1).float(),
        index=pt["t_index"].reshape(-1).long(),
    )

    tx = read_module_tensors(_path("texture"))
    texture = TripsTexture(
        texture_raw=tx["texture"].float(),
        background_color_raw=tx["background_color"].reshape(-1).float(),
        confidence_raw=tx["confidence_value_of_point"].reshape(-1).float(),
        non_subzero_texture=non_subzero_texture,
    )

    exposure_t = read_module_tensors(_path("exposure"))
    exposure = _first_tensor(exposure_t, "0", "exposures_values")
    assert exposure is not None, f"no exposure tensor in {_path('exposure')}: {list(exposure_t)}"
    wb_t = read_module_tensors(_path("white_balance"))
    white_balance = _first_tensor(wb_t, "0", "white_balance_values")
    assert white_balance is not None, f"no white-balance tensor in {_path('white_balance')}: {list(wb_t)}"

    response = None
    if _path("response").exists():
        response = read_module_tensors(_path("response"))["response"].float()
    vignette_params = vignette_center = None
    if _path("vignette").exists():
        vig = read_module_tensors(_path("vignette"))
        vignette_params = vig["vignette_params"].reshape(-1).float()
        vignette_center = vig["vignette_center"].reshape(1, 2, 1, 1).float()

    camera = TripsCameraState(
        exposure=exposure.reshape(-1).float(),
        white_balance=white_balance.reshape(-1, 3).float(),
        response=response,
        vignette_params=vignette_params,
        vignette_center=vignette_center,
    )

    poses_w2c = None
    if _path("poses").exists():
        poses = read_module_tensors(_path("poses"))
        # [M, 8]: Sophus::SE3d = quaternion(x,y,z,w) + translation(x,y,z) + 1 pad double.
        se3 = poses["poses_se3"].double()
        assert se3.shape[1] == TRIPS_SE3_DOUBLES_PER_POSE, f"unexpected poses_se3 shape {tuple(se3.shape)}"
        tangent = poses.get("tangent_poses")
        if tangent is not None and float(tangent.abs().max()) != 0.0:
            # ApplyTangent() folds the tangent into poses_se3 after every optimizer
            # step (NeuralStructure.cpp:47-49), so a saved checkpoint should have a
            # zero tangent. Warn rather than silently render the wrong pose.
            print(
                f"WARNING: {_path('poses')} has a non-zero tangent_poses "
                f"(max |.| = {float(tangent.abs().max()):.3e}); poses_se3 may be stale."
            )
        poses_w2c = se3[:, :7].contiguous()

    intrinsics = None
    if _path("intrinsics").exists():
        intrinsics = read_module_tensors(_path("intrinsics"))["intrinsics"].float()
        assert intrinsics.shape[1] == TRIPS_INTRINSICS_ROW_LEN, (
            f"unexpected intrinsics shape {tuple(intrinsics.shape)}"
        )

    return TripsSceneCheckpoint(
        epoch_dir=epoch_dir,
        scene_name=scene_name,
        points=points,
        texture=texture,
        camera=camera,
        poses_w2c=poses_w2c,
        intrinsics=intrinsics,
    )


def build_neural_camera(
    state: TripsCameraState, image_height: int, image_width: int, config: object | None = None
) -> NeuralCamera:
    """Instantiate a `trippy.net.camera_model.NeuralCamera` holding `state`.

    The response LUT / vignette parameters are copied in directly (they are
    the same tensors, same shapes, in both implementations -- see
    docs/TRIPS_REFERENCE.md Sec. 6). Exposure and white balance are passed
    as the module's *initial* values, which is exactly what they are: the
    published checkpoints' trained values.

    Args:
        state: as returned by `load_trips_scene_checkpoint`.
        image_height, image_width: render resolution, needed for the
            vignette's aspect correction.
        config: optional `NeuralCameraConfig` override.

    Returns:
        A `NeuralCamera` in `eval()` mode (which disables the response
        curve's training-only "leak" extrapolation, matching TRIPS's own
        eval path).
    """
    from trippy.net.camera_model import NeuralCamera, NeuralCameraConfig

    cfg = config if config is not None else NeuralCameraConfig()
    assert isinstance(cfg, NeuralCameraConfig)
    if state.response is not None:
        cfg = replace(cfg, response_params=int(state.response.shape[-1]))

    camera = NeuralCamera(
        image_height=image_height,
        image_width=image_width,
        num_frames=int(state.exposure.shape[0]),
        config=cfg,
        initial_exposure=state.exposure.clone(),
        initial_white_balance=state.white_balance.clone(),
    )
    with torch.no_grad():
        if camera.camera_response is not None and state.response is not None:
            camera.camera_response.response.copy_(state.response)
        if camera.vignette_net is not None and state.vignette_params is not None:
            camera.vignette_net.vignette_params.copy_(state.vignette_params)
            if state.vignette_center is not None:
                camera.vignette_net.vignette_center.copy_(state.vignette_center)
    camera.eval()
    return camera
