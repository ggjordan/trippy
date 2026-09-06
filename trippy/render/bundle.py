"""The "trippy asset bundle": one directory a native (Rust) viewer can open.

Module: trippy.render.bundle
Purpose: `trippy export-bundle`'s implementation. A bundle is the smallest
    self-contained description of a trained scene that a free-flying viewer
    needs: **world-space** points, the U-Net + tone-mapper weights, and the
    scene's real cameras. It is deliberately NOT the same thing as
    `tools/export_unet_safetensors.py horse-e2e`, which bakes one view's
    pose *and* its lens distortion into the point positions so that
    `brush_pyramid`'s plain-pinhole `Camera` reproduces TRIPS's projection.
    That trick is correct for a single-frame parity test and useless for a
    viewer: you cannot orbit a point cloud that has already been projected.
    Here the points stay in the world frame and every view carries its own
    `(R, t, K, distortion)`, so the Rust side applies the distortion in its
    projection kernel instead.

Invariants:
    - `points.npz` `xyz` is **COLMAP world frame** (docs/GEOMETRY.md): the
      camera looks down `+Z`, `+X` right, `+Y` down, and `(R, t)` is
      world-to-camera, `x_cam = R @ x_world + t`, with `R` serialised
      **row-major** -- byte-for-byte the convention of
      `brush_pyramid::scene::Camera`.
    - `size` and `conf` are the **effective** (post-softplus / post-sigmoid)
      values, never the raw trainable parameters: the viewer does no
      activation of its own.
    - `distortion` is Saiga's 8-parameter order `k1 k2 k3 k4 k5 k6 p1 p2`,
      exactly as `trippy.render.parity.distort_normalized` consumes it. All
      zeros means "no distortion" (which is what a trippy-native checkpoint
      always writes -- `trippy.scene.dataset` undistorts on ingest).
    - `weights.safetensors` is whatever `trippy.net.export_safetensors.export`
      writes today, format `trippy-unet-1`, unchanged. This module never
      touches that schema. It does sanitise ONE tensor's values before
      handing them over: `camera.exposure`, see `trusted_exposures` -- a
      per-view EV more than `BUNDLE_EXPOSURE_TRUST_STOPS` from the scene
      median is an unconverged initialisation, not a colour grade, and the
      substitution is recorded in the file's `__metadata__`.
    - `params` must equal the render settings the checkpoint's *own* engine
      uses, or the Rust render is a different picture: TRIPS checkpoints get
      TRIPS's thresholds (`PARITY_*`), trippy-native checkpoints get their
      own `TrainConfig` fields plus `trippy.constants.RASTER_*`.
    - float32 everywhere; CPU only; nothing here opens an image.
Units: `xyz`, `size` and `t` are world units; `fx/fy/cx/cy` are layer-0
    pixels; `conf` and the distortion coefficients are dimensionless.
Related docs: docs/GEOMETRY.md (frames, up vector), docs/TRIPS_REFERENCE.md
    Sec. 2/3/5/6, rust/README.md ("brush-unet weight schema"),
    rust/crates/brush-pyramid/src/{params,scene}.rs.

-- bundle.json schema (format = "trippy-bundle-1") ---------------------------
    format          str, always BUNDLE_FORMAT.
    name            str, scene label (from --name).
    points          str, filename of the point set (BUNDLE_POINTS_FILENAME).
    weights         str, filename of the weights (BUNDLE_WEIGHTS_FILENAME).
    num_points      int, N.
    num_channels    int, C (the feature width, = the U-Net's in_channels).
    background      C floats, the pyramid's background feature vector.
    params          the `brush_pyramid::PyramidParams` fields, enums as
                    lowercase strings: mode, num_layers, pixel_center,
                    halving, max_frags, t_cutoff, alpha_min, znear.
    up              3 floats, the scene up vector for orbit controls.
    default_view    int, the **ARRAY POSITION** in `views` the viewer opens
                    at (not the dataset image index -- see
                    `default_view_position`).
    views           every view of the scene, each:
                    index (dataset image index, also the tone mapper's frame
                    index), name, width, height, fx, fy, cx, cy,
                    R (9 floats, row-major world-to-camera),
                    t (3 floats), distortion (8 floats, Saiga order).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from trippy.constants import (
    RASTER_ALPHA_MIN,
    RASTER_MAX_FRAGS,
    RASTER_T_CUTOFF,
    RASTER_ZNEAR,
    TRAIN_CHECKPOINT_DIRNAME,
    TRAIN_CHECKPOINT_LATEST_FILENAME,
)
from trippy.net.export_safetensors import export

if TYPE_CHECKING:  # pragma: no cover -- typing only, keeps import cost off the CLI
    from trippy.net.camera_model import NeuralCamera
    from trippy.net.unet import MultiScaleUnet2dDecOnlySmallFixed

#: Wire format tag written into `bundle.json`; bump on any breaking change.
BUNDLE_FORMAT = "trippy-bundle-1"

#: The three files a bundle directory contains. Nothing else is written.
BUNDLE_JSON_FILENAME = "bundle.json"
BUNDLE_POINTS_FILENAME = "points.npz"
BUNDLE_WEIGHTS_FILENAME = "weights.safetensors"

#: Scene up vector for the viewer's orbit controls. COLMAP/ADOP put +Y *down*
#: (docs/GEOMETRY.md "Camera conventions" and "ADOP format for COLMAP export",
#: which fixes the up vector at (0, -1, 0)), so "up" is -Y.
BUNDLE_UP = (0.0, -1.0, 0.0)

#: Number of Saiga lens-distortion coefficients (k1 k2 k3 k4 k5 k6 p1 p2).
BUNDLE_DISTORTION_COEFFS = 8

#: How far, in stops (EV), a view's per-image exposure may sit from the
#: scene's median before the bundle stops treating it as a colour grade.
#:
#: `NeuralCamera` applies exposure as a gain of `2 ** -EV`, so 2 stops is a
#: factor of 4 either way -- far more spread than any hand-held capture of one
#: place shows, and comfortably inside the gap this threshold has to land in.
#: Measured on EXP-0003 full2-broadcast (219 views, median EV 0.250): the
#: nearest kept view is 1.14 stops out and the nearest replaced view is 4.46
#: stops out, so anything from ~1.2 to ~4.4 selects exactly the same 10 views
#: -- and those 10 are exactly the 10 images whose EXIF carried no
#: ExposureTime/ISO. See `trusted_exposures`.
BUNDLE_EXPOSURE_TRUST_STOPS = 2.0

#: Largest intrinsic skew `s` (pixels) tolerated before refusing to export.
#: The bundle's camera is `fx/fy/cx/cy` only -- `brush_pyramid::scene::Camera`
#: has no skew term -- so a non-zero `s` would be silently dropped geometry.
BUNDLE_MAX_SKEW_PX = 1e-6

#: Public Zenodo TRIPS checkpoints (the horse) use 8 pyramid levels
#: (docs/TRIPS_REFERENCE.md Sec. 5).
TRIPS_CHECKPOINT_NUM_LAYERS = 8

#: Epoch subdirectory tried first for a TRIPS checkpoint; if it is absent the
#: highest-numbered `ep*` directory is used instead (see `resolve_epoch`).
TRIPS_DEFAULT_EPOCH = "ep0600"

#: Dataset image index a TRIPS bundle opens at: the view EXP-0002 measured
#: parity on (00009.jpg), so the viewer's first frame is the one every
#: recorded number in research/trips-metal.md refers to.
TRIPS_DEFAULT_VIEW_INDEX = 8

#: Dataset image index a trippy-native bundle opens at: its first image.
NATIVE_DEFAULT_VIEW_INDEX = 0


# --- the in-memory bundle ----------------------------------------------------


@dataclass(frozen=True)
class BundleView:
    """One real camera of the scene, in `brush_pyramid::scene::Camera` form.

    Attributes:
        index: dataset image index; also the tone mapper's frame index.
        name: image filename, for the viewer's UI only.
        width, height: layer-0 image size, pixels.
        fx, fy, cx, cy: pinhole intrinsics, layer-0 pixels.
        R: 9 floats, **row-major** world-to-camera rotation.
        t: 3 floats, world-to-camera translation, world units.
        distortion: 8 floats, Saiga order `k1 k2 k3 k4 k5 k6 p1 p2`;
            all-zero means the images are already undistorted.
    """

    index: int
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: tuple[float, ...]
    t: tuple[float, ...]
    distortion: tuple[float, ...]

    def to_json(self) -> dict[str, Any]:
        """This view as the `views[]` entry of `bundle.json`."""
        return {
            "index": int(self.index),
            "name": str(self.name),
            "width": int(self.width),
            "height": int(self.height),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "R": [float(v) for v in self.R],
            "t": [float(v) for v in self.t],
            "distortion": [float(v) for v in self.distortion],
        }


@dataclass(frozen=True)
class BundleParams:
    """`brush_pyramid::PyramidParams`, with the enums as lowercase strings.

    Attributes:
        mode: layer-selection rule, one of `trippy.constants.RASTER_MODES`.
        num_layers: pyramid levels, = the U-Net's `num_layers`.
        pixel_center: "integer" (TRIPS) or "half" (trippy's own).
        halving: "ceil" or "floor" pyramid halving.
        max_frags: per-pixel fragment list depth.
        t_cutoff: transmittance below which compositing stops.
        alpha_min: alpha floor; TRIPS applies none (0.0).
        znear: near-plane cull, world units.
    """

    mode: str
    num_layers: int
    pixel_center: str
    halving: str
    max_frags: int
    t_cutoff: float
    alpha_min: float
    znear: float

    def to_json(self) -> dict[str, Any]:
        """This parameter block as the `params` object of `bundle.json`."""
        return {
            "mode": str(self.mode),
            "num_layers": int(self.num_layers),
            "pixel_center": str(self.pixel_center),
            "halving": str(self.halving),
            "max_frags": int(self.max_frags),
            "t_cutoff": float(self.t_cutoff),
            "alpha_min": float(self.alpha_min),
            "znear": float(self.znear),
        }


@dataclass
class BundleSource:
    """Everything `write_bundle` needs, independent of which loader built it.

    Attributes:
        name: scene label written into `bundle.json`.
        xyz: (N, 3) float32 **world-frame** positions, world units.
        size: (N,) float32 effective (post-softplus) radii, world units.
        feat: (N, C) float32 per-point features.
        conf: (N,) float32 effective (post-sigmoid) confidences in (0, 1).
        background: (C,) float32 pyramid background feature.
        net: the U-Net to export.
        camera: the matching `NeuralCamera` (tone mapper).
        params: render settings the checkpoint's own engine uses.
        views: every view of the scene, in dataset order.
        default_view_index: dataset image index the viewer should open at
            (converted to an array position by `write_bundle`).
        metadata: extra string->string entries merged into the
            safetensors `__metadata__` block.
    """

    name: str
    xyz: np.ndarray
    size: np.ndarray
    feat: np.ndarray
    conf: np.ndarray
    background: np.ndarray
    net: MultiScaleUnet2dDecOnlySmallFixed
    camera: NeuralCamera | None
    params: BundleParams
    views: list[BundleView]
    default_view_index: int = NATIVE_DEFAULT_VIEW_INDEX
    metadata: dict[str, str] | None = None

    @property
    def num_points(self) -> int:
        """N, the number of points."""
        return int(self.xyz.shape[0])

    @property
    def num_channels(self) -> int:
        """C, the feature width."""
        return int(self.feat.shape[1])


def default_view_position(views: list[BundleView], dataset_index: int) -> int:
    """Array position in `views` of the view whose dataset `index` matches.

    `bundle.json`'s `default_view` is an **array position**, not a dataset
    image index, so a viewer can index `views[default_view]` without a
    search. The two coincide whenever the bundle holds every view in dataset
    order (which both loaders here produce), but the distinction is the
    contract.

    Args:
        views: the bundle's views, in the order they will be serialised.
        dataset_index: the wanted view's dataset image index.

    Returns:
        The array position, or 0 when `dataset_index` is not present (a
        scene may simply have fewer views than the preferred index).
    """
    for position, view in enumerate(views):
        if view.index == dataset_index:
            return position
    return 0


# --- exposure hygiene --------------------------------------------------------


def trusted_exposures(
    exposure: np.ndarray, max_stops: float = BUNDLE_EXPOSURE_TRUST_STOPS
) -> tuple[np.ndarray, list[int], float]:
    """Replace per-view EVs that are an initialisation, not a colour grade.

    A bundle is a *walkthrough*: it renders arbitrary poses, and the frame
    index only picks which image's learned exposure/white balance to apply.
    That is sound as long as every per-view exposure is a measurement. It is
    not, in two ways that both produce a white or a black frame:

    - a **held-out** image's exposure never receives a gradient, so it is
      still whatever the initialiser wrote;
    - an image whose EXIF carried no ExposureTime/ISO was initialised to a
      wrong value by the bug fixed in `Trainer._initial_exposure`
      (2026-09-06), and `lr_exposure = 5e-4` recovers only part of it even
      over 300 epochs.

    Both show up the same way: an EV far from the scene's median. This
    replaces those with the median -- the grade of the average photograph,
    which is the honest stand-in for "this frame has no exposure of its own"
    and is exactly what a correct initialisation would have given it. The
    median (not the mean) is the reference precisely because the values being
    rejected would drag a mean.

    Args:
        exposure: `(M,)` per-image EV, applied downstream as a `2 ** -EV`
            gain.
        max_stops: how far from the median an EV may sit and still be kept.

    Returns:
        `(values, replaced, reference_ev)`: the sanitised copy, the sorted
        frame indices that were replaced, and the median EV used.
    """
    values = np.array(exposure, dtype=np.float32).reshape(-1).copy()
    if values.size == 0:
        return values, [], 0.0
    reference = float(np.median(values))
    outlier = np.abs(values - reference) > float(max_stops)
    replaced = [int(i) for i in np.flatnonzero(outlier)]
    values[outlier] = np.float32(reference)
    return values, replaced, reference


def default_view_with_trusted_exposure(
    views: list[BundleView], exposure: np.ndarray, fallback: int, max_stops: float = BUNDLE_EXPOSURE_TRUST_STOPS
) -> int:
    """Dataset index of the view the bundle should open at, exposure-aware.

    A viewer opens on `default_view` before the user has touched anything, so
    that frame is the scene's first impression and has to be one the model can
    actually be judged by. Opening on a view whose exposure was never trained
    is how the delivered Karekare bundle came to show a white frame
    (docs/LIMITATIONS.md "Per-image exposure").

    So: prefer `fallback` when its own exposure is trustworthy (which keeps
    every well-behaved scene, the public horse included, opening exactly where
    it always did). Otherwise pick the trustworthy view whose exposure is
    **nearest the scene median** -- the most ordinary-looking photograph of the
    capture -- breaking ties towards the lowest dataset index so the choice is
    deterministic.

    Args:
        views: the bundle's views, in the order they will be serialised.
        exposure: `(M,)` per-image EV, indexed by *dataset* index.
        fallback: the dataset index the loader would otherwise have picked.
        max_stops: the `trusted_exposures` threshold.

    Returns:
        A dataset image index. `fallback` unchanged when there is no exposure
        table, when `fallback` is already trustworthy, or when no view is.
    """
    values = np.asarray(exposure, dtype=np.float32).reshape(-1)
    if values.size == 0 or not views:
        return fallback
    reference = float(np.median(values))

    def trusted(index: int) -> bool:
        return 0 <= index < values.size and abs(float(values[index]) - reference) <= float(max_stops)

    if trusted(fallback):
        return fallback
    candidates = [v.index for v in views if trusted(v.index)]
    if not candidates:
        return fallback
    return min(candidates, key=lambda i: (abs(float(values[i]) - reference), i))


def _sanitise_camera_exposure(
    camera: NeuralCamera | None, max_stops: float = BUNDLE_EXPOSURE_TRUST_STOPS
) -> tuple[NeuralCamera | None, dict[str, str]]:
    """`trusted_exposures` applied to a copy of `camera`; the metadata it earns.

    The copy matters: `write_bundle` is handed the *live* tone mapper of a
    Trainer that a caller may keep using (`trippy train --report` exports a
    bundle and then goes on to evaluate), so the exported table must not be
    written back into it.

    Returns:
        `(camera_or_copy, metadata)`. The metadata keys land in the
        safetensors `__metadata__` block, so a delivered bundle always says
        on its own whether any exposure was substituted and which.
    """
    if camera is None or camera.exposures_values is None:
        return camera, {}
    original = camera.exposures_values.detach().cpu().numpy().reshape(-1)
    values, replaced, reference = trusted_exposures(original, max_stops)
    metadata = {
        "exposure_reference_ev": f"{reference:.6f}",
        "exposure_trust_stops": f"{float(max_stops):.6f}",
        "exposure_substituted_count": str(len(replaced)),
        "exposure_substituted_frames": ",".join(str(i) for i in replaced),
    }
    if not replaced:
        return camera, metadata

    sanitised = copy.deepcopy(camera)
    with torch.no_grad():
        sanitised.exposures_values.copy_(
            torch.from_numpy(values).view_as(sanitised.exposures_values)
        )
    print(
        f"export-bundle: {len(replaced)} of {values.size} per-view exposures were more than "
        f"{max_stops} stops from the scene median EV {reference:.3f} and are not a colour "
        f"grade; substituted the median (frames {replaced})"
    )
    return sanitised, metadata


# --- writing -----------------------------------------------------------------


def _as_f32(array: torch.Tensor | np.ndarray) -> np.ndarray:
    """Detached, contiguous float32 numpy view of a torch tensor or array."""
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(array, dtype=np.float32))


def bundle_document(source: BundleSource) -> dict[str, Any]:
    """Build `bundle.json`'s document for `source` (no file is written).

    Split out from `write_bundle` so tests can validate the schema without
    touching the filesystem.

    `default_view` is the loader's preferred view unless that view's own
    exposure is untrustworthy, in which case the most ordinarily-exposed view
    is chosen instead -- see `default_view_with_trusted_exposure`.
    """
    default_index = source.default_view_index
    if source.camera is not None and source.camera.exposures_values is not None:
        default_index = default_view_with_trusted_exposure(
            source.views,
            source.camera.exposures_values.detach().cpu().numpy().reshape(-1),
            default_index,
        )
    return {
        "format": BUNDLE_FORMAT,
        "name": source.name,
        "points": BUNDLE_POINTS_FILENAME,
        "weights": BUNDLE_WEIGHTS_FILENAME,
        "num_points": source.num_points,
        "num_channels": source.num_channels,
        "background": [float(v) for v in np.asarray(source.background).reshape(-1)],
        "params": source.params.to_json(),
        "up": [float(v) for v in BUNDLE_UP],
        "default_view": default_view_position(source.views, default_index),
        "views": [view.to_json() for view in source.views],
    }


def write_bundle(source: BundleSource, out_dir: str | Path) -> Path:
    """Write `source` as a bundle directory: the three files, nothing else.

    Args:
        source: the loaded scene (see `load_trips_bundle` /
            `load_native_bundle`).
        out_dir: destination directory, created if absent. Existing files of
            the same three names are overwritten.

    Returns:
        The bundle directory path.

    Raises:
        ValueError: if the point arrays are not row-aligned by point index,
            if `background` is not C-long, or if `views` is empty.
    """
    out = Path(out_dir)
    xyz, size, feat, conf = (_as_f32(a) for a in (source.xyz, source.size, source.feat, source.conf))
    background = _as_f32(source.background).reshape(-1)

    n = int(xyz.shape[0])
    if xyz.shape != (n, 3):
        raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")
    if feat.ndim != 2 or feat.shape[0] != n:
        raise ValueError(f"feat must be (N, C) with N={n}, got {feat.shape}")
    for label, array in (("size", size), ("conf", conf)):
        if array.shape != (n,):
            raise ValueError(f"{label} must be ({n},), got {array.shape}")
    if background.shape != (feat.shape[1],):
        raise ValueError(f"background must be ({feat.shape[1]},), got {background.shape}")
    if not source.views:
        raise ValueError("a bundle needs at least one view")

    out.mkdir(parents=True, exist_ok=True)
    # Uncompressed on purpose: the Rust reader mmaps a plain .npz, and the
    # point set is already float32 noise that deflate barely shrinks.
    np.savez(out / BUNDLE_POINTS_FILENAME, xyz=xyz, size=size, feat=feat, conf=conf)

    camera, exposure_metadata = _sanitise_camera_exposure(source.camera)
    metadata = {
        "source": "trippy.render.bundle",
        "bundle_format": BUNDLE_FORMAT,
        "bundle_name": source.name,
        "num_points": str(n),
    }
    metadata.update(exposure_metadata)
    metadata.update(source.metadata or {})
    export(source.net, camera, out / BUNDLE_WEIGHTS_FILENAME, extra_metadata=metadata)

    document = bundle_document(source)
    (out / BUNDLE_JSON_FILENAME).write_text(json.dumps(document, indent=2) + "\n")
    return out


# --- source detection --------------------------------------------------------


def resolve_epoch(checkpoint: Path, epoch: str | None = None) -> str:
    """Epoch subdirectory name to read out of a TRIPS checkpoint directory.

    Args:
        checkpoint: the checkpoint directory (holds `params.ini` and
            `ep<NNNN>/`).
        epoch: explicit subdirectory name, or None to auto-pick.

    Returns:
        `epoch` when given, else `TRIPS_DEFAULT_EPOCH` when it exists, else
        the highest-sorting `ep*` subdirectory.

    Raises:
        ValueError: if no `ep*` subdirectory exists at all.
    """
    if epoch:
        return epoch
    if (checkpoint / TRIPS_DEFAULT_EPOCH).is_dir():
        return TRIPS_DEFAULT_EPOCH
    candidates = sorted(p.name for p in checkpoint.glob("ep*") if p.is_dir())
    if not candidates:
        raise ValueError(f"no ep*/ subdirectory under {checkpoint}")
    return candidates[-1]


def native_checkpoint_path(checkpoint: Path) -> Path | None:
    """The trippy-native `.pt` file `checkpoint` refers to, or None.

    Accepts the file itself, a run directory (`<run>/checkpoints/
    checkpoint_latest.pt`, else the highest-numbered `checkpoint_ep*.pt`),
    or a `checkpoints/` directory directly.
    """
    if checkpoint.is_file() and checkpoint.suffix == ".pt":
        return checkpoint
    for directory in (checkpoint / TRAIN_CHECKPOINT_DIRNAME, checkpoint):
        latest = directory / TRAIN_CHECKPOINT_LATEST_FILENAME
        if latest.is_file():
            return latest
        numbered = sorted(directory.glob("checkpoint_ep*.pt"))
        if numbered:
            return numbered[-1]
    return None


def detect_source_kind(checkpoint: Path, scene: Path | None) -> str:
    """Decide which loader `checkpoint` belongs to: "trips" or "native".

    A TRIPS/ADOP checkpoint is a directory of `ep<NNNN>/` epoch dirs (plus
    `params.ini`) and needs a separate `--scene`; a trippy-native checkpoint
    is a `.pt` written by `Trainer.save_checkpoint` (or a run directory
    containing one) and carries its own scene path inside its config.

    Raises:
        ValueError: if neither shape matches, with what was looked for.
    """
    has_epochs = checkpoint.is_dir() and any(p.is_dir() for p in checkpoint.glob("ep*"))
    if has_epochs and scene is not None:
        return "trips"
    if native_checkpoint_path(checkpoint) is not None:
        return "native"
    if has_epochs:
        raise ValueError(
            f"{checkpoint} looks like a TRIPS checkpoint (it has ep*/ subdirectories); "
            "pass --scene <adop scene dir> as well"
        )
    raise ValueError(
        f"{checkpoint} is neither a TRIPS checkpoint directory (ep*/ + --scene) nor a "
        "trippy-native checkpoint (.pt file, or a run dir with checkpoints/*.pt)"
    )


# --- loader: TRIPS / ADOP checkpoints (the public Zenodo layout) -------------


@dataclass
class TripsScene:
    """A loaded TRIPS checkpoint + its ADOP scene, ready to render or export.

    Attributes:
        adop: the parsed ADOP scene directory.
        ckpt: the loaded per-scene checkpoint (points, texture, camera,
            trained poses/intrinsics).
        points: `trippy.render.parity.ScenePoints` -- world-frame `xyz`,
            post-softplus `size`, features, post-sigmoid `conf`, background.
        net: trippy's U-Net with `render_net.pth` transplanted in.
        num_layers: pyramid levels the checkpoint was trained with.
    """

    adop: Any
    ckpt: Any
    points: Any
    net: MultiScaleUnet2dDecOnlySmallFixed
    num_layers: int


def load_trips_scene(
    checkpoint: Path,
    scene: Path,
    epoch: str,
    num_layers: int = TRIPS_CHECKPOINT_NUM_LAYERS,
    device: torch.device | None = None,
) -> TripsScene:
    """Load a TRIPS/ADOP checkpoint through trippy's own parity code path.

    This is the single loader shared by `tools/export_unet_safetensors.py`
    (whose `_load_horse` is a thin wrapper over it) and by
    `load_trips_bundle` below, so the bundle and the parity fixtures can
    never drift apart.

    Args:
        checkpoint: checkpoint directory (`params.ini` + `ep<NNNN>/`).
        scene: ADOP scene directory (`dataset.ini`, `poses.txt`, ...).
        epoch: epoch subdirectory name, e.g. "ep0600".
        num_layers: pyramid levels the checkpoint's U-Net expects.
        device: where to put the point tensors (default CPU).

    Returns:
        A `TripsScene`.
    """
    from trippy.net.checkpoint import load_trips_scene_checkpoint
    from trippy.render.parity import _scene_name_from_params, build_network, build_scene_points
    from trippy.scene.adop_io import load_adop_scene

    adop = load_adop_scene(scene)
    scene_name = _scene_name_from_params(checkpoint, None)
    ckpt = load_trips_scene_checkpoint(checkpoint / epoch, scene_name)
    points = build_scene_points(ckpt, device or torch.device("cpu"))
    net, _ = build_network(checkpoint, epoch, num_layers, int(points.feat.shape[1]))
    return TripsScene(adop=adop, ckpt=ckpt, points=points, net=net, num_layers=num_layers)


def trips_views(loaded: TripsScene) -> list[BundleView]:
    """Every view of a TRIPS scene, with the checkpoint's *trained* pose/intrinsics.

    `resolve_pose`/`resolve_intrinsics` are authoritative over the scene's
    own `poses.txt`/`camera0.ini`: the published runs train both
    (`fix_poses = false`), so the checkpoint's values are what the parity
    engine renders with.

    Raises:
        ValueError: if a view's intrinsics carry a skew term the bundle's
            camera cannot represent (see `BUNDLE_MAX_SKEW_PX`).
    """
    from trippy.render.parity import resolve_intrinsics, resolve_pose

    views: list[BundleView] = []
    for index in range(len(loaded.adop)):
        view = loaded.adop.view(index)
        view = resolve_pose(loaded.adop, loaded.ckpt, view)
        view = resolve_intrinsics(loaded.ckpt, view, loaded.adop.render_scale)
        K = np.asarray(view.K, dtype=np.float64)
        skew = float(K[0, 1])
        if abs(skew) > BUNDLE_MAX_SKEW_PX:
            raise ValueError(
                f"view {index} ({view.image_name}) has intrinsic skew s={skew}; the bundle "
                "camera is fx/fy/cx/cy only and would silently drop it"
            )
        views.append(
            BundleView(
                index=index,
                name=str(view.image_name),
                width=int(view.width),
                height=int(view.height),
                fx=float(K[0, 0]),
                fy=float(K[1, 1]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
                R=tuple(float(v) for v in np.asarray(view.R, dtype=np.float64).reshape(-1)),
                t=tuple(float(v) for v in np.asarray(view.t, dtype=np.float64).reshape(-1)),
                distortion=tuple(
                    float(v) for v in np.asarray(view.distortion, dtype=np.float64).reshape(-1)
                ),
            )
        )
    return views


def trips_params(num_layers: int) -> BundleParams:
    """The render settings TRIPS's own engine uses (`_render_trips_native`).

    These must equal what `tools/export_unet_safetensors.py export_horse_e2e`
    writes into `view_XXXXX_params.json`, or the Rust render stops matching
    the parity engine: pixel centres on integers, `ceil` halving, no alpha
    floor, and TRIPS's near plane rather than trippy's.
    """
    from trippy.render.parity import PARITY_MIN_DEPTH

    return BundleParams(
        mode="trips",
        num_layers=int(num_layers),
        pixel_center="integer",
        halving="ceil",
        max_frags=int(RASTER_MAX_FRAGS),
        t_cutoff=float(RASTER_T_CUTOFF),
        # TRIPS applies no alpha floor: every in-bounds bilinear corner takes
        # a slot in the fragment list even when its weight is 0.
        alpha_min=0.0,
        znear=float(PARITY_MIN_DEPTH),
    )


def load_trips_bundle(
    checkpoint: Path,
    scene: Path,
    epoch: str | None = None,
    name: str | None = None,
    num_layers: int = TRIPS_CHECKPOINT_NUM_LAYERS,
) -> BundleSource:
    """Build a `BundleSource` from a TRIPS/ADOP checkpoint + its scene.

    The points come out of the checkpoint in the **world** frame (the same
    `ScenePoints.xyz` `trippy.render.parity.project_adop` consumes), so the
    viewer -- not the exporter -- applies each view's pose and distortion.

    Args:
        checkpoint: checkpoint directory (`params.ini` + `ep<NNNN>/`).
        scene: ADOP scene directory.
        epoch: epoch subdirectory name; None auto-picks (`resolve_epoch`).
        name: bundle label; defaults to the scene directory's name.
        num_layers: pyramid levels the checkpoint's U-Net expects.

    Returns:
        A `BundleSource` whose `views` hold every image of the scene.
    """
    from trippy.net.checkpoint import build_neural_camera

    resolved_epoch = resolve_epoch(checkpoint, epoch)
    loaded = load_trips_scene(checkpoint, scene, resolved_epoch, num_layers)
    views = trips_views(loaded)
    first = loaded.adop.view(0)
    camera = build_neural_camera(loaded.ckpt.camera, first.height, first.width)
    points = loaded.points
    return BundleSource(
        name=name or scene.name,
        xyz=_as_f32(points.xyz),
        size=_as_f32(points.size),
        feat=_as_f32(points.feat),
        conf=_as_f32(points.conf),
        background=_as_f32(points.bg),
        net=loaded.net,
        camera=camera,
        params=trips_params(num_layers),
        views=views,
        default_view_index=TRIPS_DEFAULT_VIEW_INDEX,
        metadata={
            "checkpoint": str(checkpoint / resolved_epoch),
            "scene": str(scene),
            "kind": "trips",
        },
    )


# --- loader: trippy-native checkpoints --------------------------------------


def native_views(trainer: Any) -> list[BundleView]:
    """Every view of a trippy-native run, with its trained pose delta applied.

    `trippy.scene.dataset.SceneDataset` undistorts on ingest and caches a
    pinhole image, so the distortion vector is all zeros here by
    construction -- the bundle keeps the field so both branches share one
    schema.
    """
    views: list[BundleView] = []
    no_distortion = tuple(0.0 for _ in range(BUNDLE_DISTORTION_COEFFS))
    with torch.no_grad():
        for index in range(len(trainer.dataset)):
            item = trainer.dataset[index]
            R, t = trainer._pose_for(item, index)
            K = item["K"].detach().cpu().numpy().astype(np.float64)
            height, width = int(item["rgb"].shape[0]), int(item["rgb"].shape[1])
            views.append(
                BundleView(
                    index=index,
                    name=str(item["name"]),
                    width=width,
                    height=height,
                    fx=float(K[0, 0]),
                    fy=float(K[1, 1]),
                    cx=float(K[0, 2]),
                    cy=float(K[1, 2]),
                    R=tuple(float(v) for v in R.detach().cpu().numpy().reshape(-1)),
                    t=tuple(float(v) for v in t.detach().cpu().numpy().reshape(-1)),
                    distortion=no_distortion,
                )
            )
    return views


def native_params(cfg: Any) -> BundleParams:
    """The render settings a trippy-native checkpoint's own engine uses.

    Everything the trainer passes to `trippy.raster.pyramid.render_pyramid`
    that the Rust rasteriser also takes: the config's `mode`,
    `pixel_center`, `pyramid_halving` and `layers`, plus trippy's own
    `RASTER_*` thresholds (which the trainer leaves at their defaults).
    """
    return BundleParams(
        mode=str(cfg.mode),
        num_layers=int(cfg.layers),
        pixel_center=str(cfg.pixel_center),
        halving=str(cfg.pyramid_halving),
        max_frags=int(RASTER_MAX_FRAGS),
        t_cutoff=float(RASTER_T_CUTOFF),
        alpha_min=float(RASTER_ALPHA_MIN),
        znear=float(RASTER_ZNEAR),
    )


def load_native_bundle(checkpoint: Path, name: str | None = None) -> BundleSource:
    """Build a `BundleSource` from a trippy-native checkpoint (`trippy train`).

    Rebuilds the run's `Trainer` from the checkpoint's own saved config
    (`trippy.train.eval.build_trainer_from_checkpoint`) on **CPU**, then
    reads off the effective point state: `PointParams.xyz` is already world
    frame, `size()`/`conf()` are the post-activation values.

    Args:
        checkpoint: a `.pt` file, or a run directory containing
            `checkpoints/checkpoint_latest.pt`.
        name: bundle label; defaults to the run directory's name.

    Returns:
        A `BundleSource` whose `views` hold every image of the run's dataset.
    """
    from trippy.train.eval import build_trainer_from_checkpoint

    path = native_checkpoint_path(checkpoint)
    if path is None:
        raise ValueError(f"no trippy-native checkpoint (.pt) found at {checkpoint}")
    trainer = build_trainer_from_checkpoint(path, device="cpu")
    points = trainer.point_params
    with torch.no_grad():
        size = points.size()
        conf = points.conf()
    return BundleSource(
        name=name or Path(trainer.cfg.run_dir).name,
        xyz=_as_f32(points.xyz),
        size=_as_f32(size),
        feat=_as_f32(points.feat),
        conf=_as_f32(conf),
        background=_as_f32(trainer.background),
        net=trainer.net,
        camera=trainer.camera,
        params=native_params(trainer.cfg),
        views=native_views(trainer),
        default_view_index=NATIVE_DEFAULT_VIEW_INDEX,
        metadata={"checkpoint": str(path), "scene": str(trainer.cfg.scene_root), "kind": "native"},
    )


# --- one entry point ---------------------------------------------------------


def export_bundle(
    checkpoint: str | Path,
    out: str | Path,
    scene: str | Path | None = None,
    epoch: str | None = None,
    name: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Detect the checkpoint's kind, load it, and write the bundle directory.

    Args:
        checkpoint: TRIPS checkpoint directory, or a trippy-native `.pt` /
            run directory.
        out: bundle directory to write.
        scene: ADOP scene directory (required for, and only used by, the
            TRIPS branch).
        epoch: TRIPS epoch subdirectory name; None auto-picks.
        name: bundle label written into `bundle.json`.

    Returns:
        `(bundle_dir, bundle_json_document)`.
    """
    checkpoint_path = Path(checkpoint)
    scene_path = Path(scene) if scene is not None else None
    kind = detect_source_kind(checkpoint_path, scene_path)
    if kind == "trips":
        assert scene_path is not None  # detect_source_kind guarantees it
        print(f"export-bundle: TRIPS/ADOP checkpoint {checkpoint_path} + scene {scene_path}")
        source = load_trips_bundle(checkpoint_path, scene_path, epoch=epoch, name=name)
    else:
        print(f"export-bundle: trippy-native checkpoint {checkpoint_path}")
        source = load_native_bundle(checkpoint_path, name=name)
    bundle_dir = write_bundle(source, out)
    return bundle_dir, bundle_document(source)
