"""Reproduce a TRIPS render from the authors' own checkpoint, through trippy.

Module: trippy.render.parity
Purpose: the v0.1.0 gate ("forward renders match a reference"). Loads a
    published ADOP scene + TRIPS checkpoint, rasterises each requested view
    with `trippy.raster.pyramid.render_pyramid`, runs trippy's U-Net and
    NeuralCamera, and reports PSNR / SSIM / LPIPS against both the ground-
    truth photograph and (when present) the authors' own rendered output.
Invariants:
    - Nothing here trains or mutates the checkpoint; every tensor is
      `detach()`ed and the modules run in `eval()`.
    - Pixel convention: TRIPS's `ip` puts pixel *centres* at integer
      coordinates (`PointBlending.h:216-240` takes the 2x2 footprint from
      `floor(ip)`), trippy's rasteriser puts them at `i + 0.5`
      (docs/GEOMETRY.md). Every K handed to `render_pyramid` therefore has
      `cx, cy` shifted by `+PARITY_PIXEL_CENTRE_OFFSET`, and each pyramid
      layer is rendered by its own `num_layers=1` call with
      `K_l = (fx, fy, cx, cy) / 2**l` so that TRIPS's per-layer
      `ip *= 0.5f` is reproduced exactly rather than approximated.
    - Lens distortion is applied *before* `render_pyramid`, by replacing
      each point's camera-space position with
      `(xd * z, yd * z, z)` where `(xd, yd)` is its distorted normalised
      coordinate. That is an identity for depth and for the sort order, and
      makes the plain pinhole projection inside `render_pyramid` land on
      TRIPS's distorted `ip`.
Units: world units for positions/sizes/depth; pixels for K and `size_px`;
    images are float32 in [0, 1] with shape (3, H, W) or (H, W, 3).
Related docs: docs/TRIPS_REFERENCE.md Sec. 2/3/3a/5a/6/8/9b;
    experiments/EXP-0002-horse-parity/README.md; docs/LIMITATIONS.md.

-- Which render path the published checkpoints actually used --
docs/TRIPS_REFERENCE.md Sec. 2/3 says `use_layer_point_size` is unreachable
from any `.ini` and therefore always false. That is wrong for these
checkpoints. `CombinedParams::Check` derives it from the optimizer flags
(third_party/TRIPS/src/lib/data/Settings.cpp:39):

    render_params.use_layer_point_size = !optimizer_params.fix_point_size;

and `checkpoint_horse/params.ini` has `fix_point_size = false`, so it is
**true**. Together with `render_points_in_all_lower_resolutions = true`
(Settings.h:78, also in that params.ini) and `combine_lists = false`, that
selects the `RenderFast16` / `CountAndCollectTiled` code path
(PointRenderer.cu:726-750, RenderForward.cu:168-368,1065), *not* the
`CountTiled`/`CollectTiled2` path Sec. 3 documents. In that kernel each
point is written into layers `0 .. layer_higher` inclusive, where
`layer_higher = clamp(ceil(log2(size_px)), 0, L-1)` and `size_px = 0` when
`size_px <= 1` (RenderForward.cu:334-352), and the per-layer alpha is
`bilinear_fac * confidence * compute_point_size_fac(size_px, layer, L)`
(RenderForward.cu:3511-3517). `compute_point_size_fac` returns **1.0** for
every layer strictly below `layer_lower` (PointBlending.h:92-96), so this is
neither trippy's `"broadcast"` (all layers, factor 1) nor its `"trilinear"`
(only the two straddling layers). This module calls that third mode
`"trips"`, and renders `"broadcast"` and `"trilinear"` as ablations.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from trippy.constants import (
    ADOP_DIST_CUTOFF,
    ADOP_DISTORTION_SENTINEL,
    PARITY_DEFAULT_NUM_LAYERS,
    PARITY_EVAL_BORDER_PX,
    PARITY_PIXEL_CENTRE_OFFSET,
    PARITY_PSNR_CAP_DB,
    PARITY_PSNR_MAX,
    RASTER_MAX_FRAGS,
    RASTER_T_CUTOFF,
)
from trippy.net.camera_model import NeuralCamera
from trippy.net.checkpoint import (
    TripsSceneCheckpoint,
    build_neural_camera,
    load_trips_scene_checkpoint,
    try_load_trips_network,
)
from trippy.net.losses import ssim as ssim_metric
from trippy.net.unet import MultiScaleUnet2dDecOnlySmallFixed, NetworkConfig
from trippy.raster.emit import layer_factor
from trippy.raster.pyramid import render_pyramid
from trippy.render.sheets import colorize, contact_sheet, save_png
from trippy.scene.adop_io import AdopScene, AdopView, load_adop_scene, quat_xyzw_to_wxyz, qvec2R

RENDER_MODES = ("trips", "broadcast", "trilinear")


# --- geometry ------------------------------------------------------------


def distort_normalized(xy: Tensor, distortion: Tensor, max_r: float = ADOP_DIST_CUTOFF) -> Tensor:
    """Saiga's 8-parameter lens distortion, applied to normalised image points.

    Exact port of `distortNormalizedPoint`
    (https://github.com/darglein/saiga @ ee7a4e6b658,
    `src/saiga/vision/cameraModel/Distortion.h:130-171`), whose coefficient
    order is `k1 k2 k3 k4 k5 k6 p1 p2` -- *not* OpenCV's.

    Args:
        xy: (N, 2) float, `x/z, y/z` in normalised image coordinates.
        distortion: (8,) float coefficients in Saiga order.
        max_r: points with `r2 > max_r**2` are mapped to the sentinel
            `ADOP_DISTORTION_SENTINEL`, which the caller turns into a cull
            (`RenderParams::dist_cutoff`, Settings.h:61).

    Returns:
        (N, 2) float distorted normalised coordinates.
    """
    k1, k2, k3, k4, k5, k6, p1, p2 = (distortion[i] for i in range(8))
    x = xy[:, 0]
    y = xy[:, 1]
    x2 = x * x
    y2 = y * y
    r2 = x2 + y2
    two_xy = 2.0 * x * y
    r4 = r2 * r2
    r6 = r4 * r2

    radial_u = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    radial_v = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    radial = radial_u / radial_v

    xd = x * radial + (p1 * two_xy + p2 * (r2 + 2.0 * x2))
    yd = y * radial + (p1 * (r2 + 2.0 * y2) + p2 * two_xy)

    out = torch.stack([xd, yd], dim=1)
    over = (r2 > max_r * max_r).reshape(-1, 1)
    return torch.where(over, torch.full_like(out, ADOP_DISTORTION_SENTINEL), out)


def project_adop(
    xyz: Tensor, R: Tensor, t: Tensor, K: Tensor, distortion: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Project world points exactly as `ProjectPointPinholeWoNormal` does.

    (`third_party/TRIPS/src/lib/rendering/PointRendererHelper.h:232-253`,
    then `ip = crop_transform.normalizedToImage(image_p)` which is the
    identity at eval time -- `Dataset.cpp:276` sets `crop_transform` to
    `IntrinsicsPinholef()`, i.e. fx=fy=1, cx=cy=s=0.)

    Args:
        xyz: (N, 3) world positions.
        R: (3, 3) world-to-camera rotation; t: (3,) translation.
        K: (3, 3) pinhole intrinsics in *full-resolution* pixels.
        distortion: (8,) Saiga coefficients.

    Returns:
        `(ndc, ip, z)` where `ndc` is (N, 2) distorted normalised
        coordinates, `ip` is (N, 2) pixel coordinates in TRIPS's
        centre-at-integer convention, and `z` is (N,) camera-space depth.
    """
    xyz_c = xyz @ R.transpose(0, 1) + t.reshape(1, 3)
    z = xyz_c[:, 2]
    # TRIPS divides by the raw z and only clamps the *returned* depth
    # (PointRendererHelper.h:239-241); z <= 0 is culled by the caller, so the
    # sign of the quotient for those points is irrelevant.
    z_safe = torch.where(z.abs() < 1e-12, torch.full_like(z, 1e-12), z)
    ndc = distort_normalized(xyz_c[:, :2] / z_safe.reshape(-1, 1), distortion)
    ip = torch.stack([K[0, 0] * ndc[:, 0] + K[0, 2], K[1, 1] * ndc[:, 1] + K[1, 2]], dim=1)
    return ndc, ip, z


# --- rasterisation -------------------------------------------------------


def trips_layer_shapes(height: int, width: int, num_layers: int) -> list[tuple[int, int]]:
    """Pyramid layer sizes TRIPS allocates for this network.

    `PointRenderer.cu:385-391`: every `network_version` except the literal
    string `"MultiScaleUnet2d"` -- so including
    `MultiScaleUnet2dDecOnlySmallFixed`, the one the published checkpoints
    use -- halves with `std::ceil`, not integer division. This matches
    `trippy.raster.emit.layer_grid` exactly, and is what keeps the U-Net's
    output at full resolution (1080 -> 540 -> 270 -> 135 -> 68 -> 34 -> 17
    -> 9, whose upsamples crop back down to 1080 rather than to 1024).
    """
    shapes = []
    h, w = int(height), int(width)
    for _ in range(num_layers):
        shapes.append((h, w))
        h = -(-h // 2)
        w = -(-w // 2)
    return shapes


@dataclass
class ScenePoints:
    """Device-resident point arrays shared by every view."""

    xyz: Tensor
    size: Tensor
    feat: Tensor
    conf: Tensor
    bg: Tensor

    def __len__(self) -> int:
        return int(self.xyz.shape[0])


def render_trips_layers(
    points: ScenePoints,
    view: AdopView,
    num_layers: int = PARITY_DEFAULT_NUM_LAYERS,
    mode: str = "trips",
    max_frags: int = RASTER_MAX_FRAGS,
    t_cutoff: float = RASTER_T_CUTOFF,
) -> tuple[list[Tensor], dict]:
    """Rasterise one view into `num_layers` feature images.

    Args:
        points: the scene's device tensors (see `ScenePoints`).
        view: camera for this image (`trippy.scene.adop_io.AdopScene.view`).
        num_layers: L (8 for the published Tanks & Temples checkpoints).
        mode: `"trips"` (the published checkpoints' real path -- layers
            `0..layer_higher`, alpha weighted by `compute_point_size_fac`),
            `"broadcast"` (every layer, factor 1) or `"trilinear"` (only the
            two straddling layers). See the module docstring.
        max_frags: per-pixel composite depth (TRIPS: 16).
        t_cutoff: transmittance floor (TRIPS ALPHA_DEST_CUTOFF = 0.001).

    Returns:
        `(layers, aux)`: `layers[l]` is `(C, h_l, w_l)`; `aux` carries
        `t_final` / `n_used` per layer plus `num_fragments` and
        `points_active` diagnostics.
    """
    if mode not in RENDER_MODES:
        raise ValueError(f"mode must be one of {RENDER_MODES}, got {mode!r}")
    device = points.xyz.device
    dtype = points.xyz.dtype

    def _t(arr: np.ndarray) -> Tensor:
        return torch.as_tensor(arr, dtype=dtype, device=device)

    K = _t(view.K)
    R = _t(view.R)
    t = _t(view.t)
    distortion = _t(view.distortion)
    assert abs(float(view.K[0, 1])) < 1e-12, "non-zero skew is not supported (never used by colmap2adop)"

    shapes = trips_layer_shapes(view.height, view.width, num_layers)
    height, width = shapes[0]

    ndc, ip, z = project_adop(points.xyz, R, t, K, distortion)

    # Layer-0 gate: `valid_point(ip, z, 0, ...)` (RenderForward.cu:211-215,
    # 305). A point failing this is dropped from *every* layer.
    alive = (z > 0) & (ip[:, 0] >= 0) & (ip[:, 0] < width - 1) & (ip[:, 1] >= 0) & (ip[:, 1] < height - 1)

    # point_size_opt = K.fx * crop_transform.fx * softplus(raw) / z, RenderForward.cu:268.
    size_px = K[0, 0] * points.size / torch.clamp(z, min=1e-6)

    if mode == "trilinear":
        return _render_trilinear(points, K, ndc, z, alive, shapes, max_frags, t_cutoff)

    # layer_higher, RenderForward.cu:334-338.
    log2_size = torch.log2(torch.clamp(size_px, min=torch.finfo(dtype).tiny))
    layer_higher = torch.where(
        size_px > 1.0,
        torch.clamp(torch.ceil(log2_size), 0.0, float(num_layers - 1)).to(torch.int64),
        torch.zeros_like(size_px, dtype=torch.int64),
    )

    layers: list[Tensor] = []
    aux: dict = {"t_final": [], "n_used": [], "num_fragments": 0, "points_active": []}
    for layer, (h_l, w_l) in enumerate(shapes):
        scale = 1.0 / float(1 << layer)
        ip_l = ip * scale
        floor_l = torch.floor(ip_l)
        # `valid_point(p_rd, z, layer, ...)` then `break`: failing at layer l
        # also skips every coarser layer (RenderForward.cu:344-345).
        alive = (
            alive
            & (floor_l[:, 0] >= 0)
            & (floor_l[:, 0] < w_l - 1)
            & (floor_l[:, 1] >= 0)
            & (floor_l[:, 1] < h_l - 1)
        )
        if mode == "trips":
            active = alive & (layer <= layer_higher)
            fac = layer_factor(size_px, layer, num_layers)
        else:  # "broadcast": every in-bounds point, no layer attenuation
            active = alive
            fac = torch.ones_like(size_px)

        idx = torch.nonzero(active, as_tuple=False).reshape(-1)
        aux["points_active"].append(int(idx.numel()))

        # K_l reproduces TRIPS's `ip *= 0.5f` per layer, plus the half-pixel
        # shift between the two pixel-centre conventions (module docstring).
        K_l = torch.zeros((3, 3), dtype=dtype, device=device)
        K_l[0, 0] = K[0, 0] * scale
        K_l[1, 1] = K[1, 1] * scale
        K_l[0, 2] = K[0, 2] * scale + PARITY_PIXEL_CENTRE_OFFSET
        K_l[1, 2] = K[1, 2] * scale + PARITY_PIXEL_CENTRE_OFFSET
        K_l[2, 2] = 1.0

        layer_out, layer_aux = _render_one_layer(
            points=points,
            idx=idx,
            ndc=ndc,
            z=z,
            conf_scale=fac,
            K_l=K_l,
            hw=(h_l, w_l),
            max_frags=max_frags,
            t_cutoff=t_cutoff,
        )
        layers.append(layer_out)
        aux["t_final"].append(layer_aux["t_final"][0])
        aux["n_used"].append(layer_aux["n_used"][0])
        aux["num_fragments"] += int(layer_aux["num_fragments"])
    return layers, aux


def _render_one_layer(
    points: ScenePoints,
    idx: Tensor,
    ndc: Tensor,
    z: Tensor,
    conf_scale: Tensor,
    K_l: Tensor,
    hw: tuple[int, int],
    max_frags: int,
    t_cutoff: float,
) -> tuple[Tensor, dict]:
    """One `render_pyramid(num_layers=1)` call for a single pyramid layer."""
    device = points.xyz.device
    dtype = points.xyz.dtype
    eye = torch.eye(3, dtype=dtype, device=device)
    zero3 = torch.zeros(3, dtype=dtype, device=device)

    if idx.numel() == 0:
        # No point reaches this layer: the whole layer is pure background.
        h_l, w_l = hw
        out = points.bg.reshape(-1, 1, 1).expand(-1, h_l, w_l).contiguous()
        return out, {
            "t_final": [torch.ones((h_l, w_l), dtype=dtype, device=device)],
            "n_used": [torch.zeros((h_l, w_l), dtype=torch.int64, device=device)],
            "num_fragments": 0,
        }

    z_sel = z.index_select(0, idx)
    ndc_sel = ndc.index_select(0, idx)
    # Synthetic camera-space points whose plain pinhole projection is the
    # distorted `ip / 2**layer` TRIPS computed; depth (and therefore the sort
    # order) is untouched.
    xyz_cam = torch.stack([ndc_sel[:, 0] * z_sel, ndc_sel[:, 1] * z_sel, z_sel], dim=1)

    layers, aux = render_pyramid(
        xyz_cam,
        torch.zeros_like(z_sel),  # `size` only feeds the coarse cull in broadcast mode
        points.feat.index_select(0, idx),
        points.conf.index_select(0, idx) * conf_scale.index_select(0, idx),
        K_l,
        eye,
        zero3,
        hw,
        num_layers=1,
        mode="broadcast",
        bg=points.bg,
        max_frags=max_frags,
        t_cutoff=t_cutoff,
        # TRIPS applies no alpha floor: every in-bounds bilinear corner takes a
        # slot in the 16-deep list even when its weight is 0.
        alpha_min=0.0,
    )
    return layers[0], aux


def _render_trilinear(
    points: ScenePoints,
    K: Tensor,
    ndc: Tensor,
    z: Tensor,
    alive: Tensor,
    shapes: list[tuple[int, int]],
    max_frags: int,
    t_cutoff: float,
) -> tuple[list[Tensor], dict]:
    """Ablation column: one `render_pyramid(mode="trilinear")` call, L layers.

    This is trippy's own `use_layer_point_size=true`-style emission (only the
    two layers the projected footprint straddles). `cx, cy` still carry the
    half-pixel shift so layer 0 lines up with TRIPS; coarser layers pick up
    trippy's exact corner-origin halving instead of TRIPS's `ip *= 0.5f`,
    a sub-pixel difference documented in docs/LIMITATIONS.md.
    """
    device = points.xyz.device
    dtype = points.xyz.dtype
    idx = torch.nonzero(alive, as_tuple=False).reshape(-1)
    z_sel = z.index_select(0, idx)
    ndc_sel = ndc.index_select(0, idx)
    xyz_cam = torch.stack([ndc_sel[:, 0] * z_sel, ndc_sel[:, 1] * z_sel, z_sel], dim=1)

    K_shift = K.clone()
    K_shift[0, 2] = K[0, 2] + PARITY_PIXEL_CENTRE_OFFSET
    K_shift[1, 2] = K[1, 2] + PARITY_PIXEL_CENTRE_OFFSET

    layers, aux = render_pyramid(
        xyz_cam,
        points.size.index_select(0, idx),
        points.feat.index_select(0, idx),
        points.conf.index_select(0, idx),
        K_shift,
        torch.eye(3, dtype=dtype, device=device),
        torch.zeros(3, dtype=dtype, device=device),
        shapes[0],
        num_layers=len(shapes),
        mode="trilinear",
        bg=points.bg,
        max_frags=max_frags,
        t_cutoff=t_cutoff,
        alpha_min=0.0,
    )
    aux["points_active"] = [int(idx.numel())] * len(shapes)
    return layers, aux


# --- metrics -------------------------------------------------------------


def psnr(pred: Tensor, target: Tensor, max_value: float = PARITY_PSNR_MAX) -> float:
    """Peak signal-to-noise ratio in dB over the whole image."""
    mse = float(torch.mean((pred.double() - target.double()) ** 2))
    if mse <= 0.0:
        return PARITY_PSNR_CAP_DB
    return min(PARITY_PSNR_CAP_DB, 10.0 * math.log10((max_value**2) / mse))


_LPIPS_CACHE: dict[str, object] = {}


def lpips_distance(pred: Tensor, target: Tensor, net: str = "alex") -> float:
    """LPIPS distance between two `(1, 3, H, W)` images in [0, 1].

    Uses the same `lpips` package (and the same `[-1, 1]` rescale) as
    `trippy.net.losses`. Runs on CPU: the backbone is small next to the
    rasteriser and this keeps the metric bit-identical across devices.
    """
    if net not in _LPIPS_CACHE:
        import lpips  # deferred: heavy import + first-use backbone load

        model = lpips.LPIPS(net=net)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        _LPIPS_CACHE[net] = model
    model = _LPIPS_CACHE[net]
    with torch.no_grad():
        a = pred.detach().to("cpu", torch.float32) * 2 - 1
        b = target.detach().to("cpu", torch.float32) * 2 - 1
        return float(model(a, b).mean())  # type: ignore[operator]


@dataclass
class ViewMetrics:
    """PSNR / SSIM / LPIPS of one render against one reference image."""

    psnr_db: float
    ssim: float
    lpips: float


def crop_border(image: Tensor, border: int) -> Tensor:
    """Drop `border` pixels from every side of a `(1, 3, H, W)` image."""
    if border <= 0:
        return image
    return image[:, :, border:-border, border:-border]


def compare(pred: Tensor, target: Tensor, border: int = 0) -> ViewMetrics:
    """Metrics for two `(1, 3, H, W)` float tensors in [0, 1] (CPU).

    `border` drops that many pixels from every side of both images first.
    Use `PARITY_EVAL_BORDER_PX` whenever one of the two is one of TRIPS's
    own saved test JPGs, which are blacked out that far in from each edge
    (`train_mask_border = 16`); comparing them un-cropped costs ~10 dB.
    """
    pred = crop_border(pred.detach().to("cpu", torch.float32).clamp(0.0, 1.0), border)
    target = crop_border(target.detach().to("cpu", torch.float32).clamp(0.0, 1.0), border)
    return ViewMetrics(
        psnr_db=psnr(pred, target),
        ssim=float(ssim_metric(pred, target)),
        lpips=lpips_distance(pred, target),
    )


# --- image helpers -------------------------------------------------------


def load_image(path: str | Path, height: int, width: int) -> Tensor:
    """Load an 8-bit image as `(1, 3, H, W)` float32 in [0, 1], resized if needed."""
    img = Image.open(path).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def to_hwc(image: Tensor) -> np.ndarray:
    """`(1, 3, H, W)` or `(3, H, W)` float tensor -> `(H, W, 3)` uint8 array."""
    x = image.detach().to("cpu", torch.float32)
    if x.dim() == 4:
        x = x[0]
    x = x.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)


def abs_diff_heatmap(pred: Tensor, target: Tensor, vmax: float = 0.25) -> np.ndarray:
    """Mean-absolute-error heatmap between two `(1, 3, H, W)` images."""
    diff = (pred.detach().cpu() - target.detach().cpu()).abs().mean(dim=1)[0].numpy()
    return colorize(diff, 0.0, vmax)


# --- orchestration -------------------------------------------------------


@dataclass
class ParityConfig:
    """Everything `run_parity` needs; also serialised into `metrics.json`."""

    scene_dir: str
    checkpoint_dir: str
    epoch: str = "ep0600"
    scene_name: str | None = None
    out_dir: str = "output/parity"
    device: str = "cpu"
    indices: tuple[int, ...] = ()
    images: tuple[str, ...] = ()
    num_layers: int = PARITY_DEFAULT_NUM_LAYERS
    render_scale: float | None = None
    modes: tuple[str, ...] = ("trips", "broadcast", "trilinear")
    max_points: int | None = None
    reference_dir: str | None = None


def _scene_name_from_params(checkpoint_dir: Path, override: str | None) -> str:
    """`train_params.scene_names` out of the checkpoint's own `params.ini`."""
    if override:
        return override
    from trippy.scene.adop_io import read_ini

    params = checkpoint_dir / "params.ini"
    if params.exists():
        for section in read_ini(params).values():
            if "scene_names" in section:
                return section["scene_names"].split()[0]
    raise ValueError(f"could not determine scene name; pass --scene-name (looked in {params})")


def build_scene_points(
    ckpt: TripsSceneCheckpoint,
    device: torch.device,
    max_points: int | None = None,
    seed: int = 0,
) -> ScenePoints:
    """Move the checkpoint's point cloud onto `device` in render-ready form.

    Applies TRIPS's forward parametrisations: `softplus` on the point size
    and `sigmoid(10 * .)` on the confidence. The texture and background
    colour are used **raw** -- `PrepareTexture` is called with
    `non_subzero_texture` un-negated (Pipeline.cpp:257), which is `false` in
    every published checkpoint, so no `abs()` is taken. See
    `trippy.net.checkpoint.TripsTexture` for the full correction.

    Args:
        ckpt: loaded scene checkpoint.
        device: target device.
        max_points: optional random subsample size (CPU smoke runs only --
            a subsample changes the render, it is not a parity result).
        seed: RNG seed for that subsample.
    """
    texture = ckpt.texture.texture()
    conf = ckpt.texture.confidence()
    xyz = ckpt.points.position
    size = ckpt.points.size()
    index = ckpt.points.index
    # t_index maps render-order point id -> texture column (identity in the
    # published checkpoints, but honour it rather than assume).
    if not torch.equal(index, torch.arange(index.numel(), dtype=index.dtype)):
        texture = texture.index_select(0, index)
        conf = conf.index_select(0, index)

    if max_points is not None and max_points < xyz.shape[0]:
        generator = torch.Generator().manual_seed(seed)
        keep = torch.randperm(xyz.shape[0], generator=generator)[:max_points]
        xyz, size, texture, conf = (x.index_select(0, keep) for x in (xyz, size, texture, conf))

    return ScenePoints(
        xyz=xyz.to(device, torch.float32).contiguous(),
        size=size.to(device, torch.float32).contiguous(),
        feat=texture.to(device, torch.float32).contiguous(),
        conf=conf.to(device, torch.float32).contiguous(),
        bg=ckpt.texture.background_color().to(device, torch.float32).contiguous(),
    )


def build_network(checkpoint_dir: Path, epoch: str, num_layers: int, num_channels: int) -> tuple[
    MultiScaleUnet2dDecOnlySmallFixed, object
]:
    """Build trippy's U-Net and transplant `render_net.pth` into it."""
    net = MultiScaleUnet2dDecOnlySmallFixed(
        NetworkConfig(num_layers=num_layers, num_input_channels=num_channels)
    )
    result = try_load_trips_network(checkpoint_dir / epoch / "render_net.pth", net)
    if not result.ok:
        raise ValueError(f"render_net.pth did not load cleanly: {result.reason}")
    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    return net, result


def resolve_pose(scene: AdopScene, ckpt: TripsSceneCheckpoint, view: AdopView) -> AdopView:
    """Replace a view's `(R, t)` with the checkpoint's trained pose, if present.

    `PoseModuleImpl` stores `frame.pose.inverse()`, i.e. world-to-camera, as
    `[qx, qy, qz, qw, tx, ty, tz]` (`NeuralStructure.cpp:20-33`). Poses are
    trainable in the published runs (`fix_poses = false`), so the checkpoint
    is authoritative over `poses.txt`.
    """
    from dataclasses import replace as _replace

    if ckpt.poses_w2c is None:
        return view
    row = ckpt.poses_w2c[view.index].numpy()
    q_wxyz = quat_xyzw_to_wxyz(row[:4])
    return _replace(view, R=qvec2R(q_wxyz), t=np.asarray(row[4:], dtype=np.float64))


def resolve_intrinsics(ckpt: TripsSceneCheckpoint, view: AdopView, scale: float) -> AdopView:
    """Replace a view's K/distortion with the checkpoint's trained intrinsics."""
    from dataclasses import replace as _replace

    if ckpt.intrinsics is None:
        return view
    row = ckpt.intrinsics[view.camera_index].numpy().astype(np.float64)
    fx, fy, cx, cy, s = row[:5]
    K = np.array(
        [[fx * scale, s * scale, cx * scale], [0.0, fy * scale, cy * scale], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return _replace(view, K=K, distortion=row[5:13].copy())


@dataclass
class ViewResult:
    """One rendered view in one mode, plus its metrics and timing."""

    index: int
    image_name: str
    mode: str
    seconds: float
    num_fragments: int
    points_active: list[int] = field(default_factory=list)
    vs_gt: ViewMetrics | None = None
    vs_gt_masked: ViewMetrics | None = None
    vs_reference_masked: ViewMetrics | None = None


def render_view(
    points: ScenePoints,
    view: AdopView,
    net: MultiScaleUnet2dDecOnlySmallFixed,
    camera: NeuralCamera,
    mode: str,
    num_layers: int,
) -> tuple[Tensor, list[Tensor], dict]:
    """Rasterise + U-Net + NeuralCamera for one view. Returns `(rgb, layers, aux)`."""
    device = points.xyz.device
    with torch.no_grad():
        layers, aux = render_trips_layers(points, view, num_layers=num_layers, mode=mode)
        inputs = [layer.unsqueeze(0) for layer in layers]
        raw = net(inputs)
        frame_index = torch.tensor([view.index], dtype=torch.long, device=device)
        rgb = camera(raw, frame_index)
    return rgb, layers, aux


def run_parity(config: ParityConfig) -> dict:
    """Render every requested view in every requested mode and write the report.

    Writes into `config.out_dir`: one PNG per (view, mode) plus the ground
    truth, the abs-diff heatmap and the raw level-0 composite, a contact
    sheet per view, a `summary_sheet.png` across views, `metrics.json` and
    `README.md`.

    Returns:
        The same dict that is written to `metrics.json`.
    """
    scene_dir = Path(config.scene_dir)
    checkpoint_dir = Path(config.checkpoint_dir)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    scene = load_adop_scene(scene_dir)
    scene_name = _scene_name_from_params(checkpoint_dir, config.scene_name)
    ckpt = load_trips_scene_checkpoint(checkpoint_dir / config.epoch, scene_name)
    scale = config.render_scale if config.render_scale is not None else scene.render_scale

    indices = list(config.indices)
    for name in config.images:
        indices.append(scene.index_of(name))
    if not indices:
        raise ValueError("no views requested: pass --indices or --images")

    points = build_scene_points(ckpt, device, max_points=config.max_points)
    net, load_report = build_network(
        checkpoint_dir, config.epoch, config.num_layers, int(points.feat.shape[1])
    )
    net = net.to(device)

    first_view = resolve_intrinsics(ckpt, scene.view(indices[0], scale), scale)
    camera = build_neural_camera(ckpt.camera, first_view.height, first_view.width).to(device)

    reference_dir = Path(config.reference_dir) if config.reference_dir else checkpoint_dir / config.epoch / "test"

    results: list[ViewResult] = []
    authors_baseline: dict[int, ViewMetrics] = {}
    sheet_rows: list[np.ndarray] = []
    sheet_labels: list[str] = []

    for index in indices:
        view = resolve_pose(scene, ckpt, resolve_intrinsics(ckpt, scene.view(index, scale), scale))
        gt = load_image(view.image_path, view.height, view.width)
        reference_path = reference_dir / f"{scene_name}_{index:05d}.jpg"
        reference = load_image(reference_path, view.height, view.width) if reference_path.exists() else None

        # Baseline: how well the authors' own render scores against the same
        # photograph, on the same masked region. This is the bar to hit.
        authors_vs_gt = compare(reference, gt, PARITY_EVAL_BORDER_PX) if reference is not None else None
        if authors_vs_gt is not None:
            authors_baseline[index] = authors_vs_gt

        panels: list[np.ndarray] = [to_hwc(gt)]
        labels: list[str] = [f"GT {view.image_name}"]
        if reference is not None and authors_vs_gt is not None:
            panels.append(to_hwc(reference))
            labels.append(f"TRIPS authors' render, PSNR {authors_vs_gt.psnr_db:.2f} dB vs GT")

        for mode in config.modes:
            started = time.time()
            rgb, layers, aux = render_view(points, view, net, camera, mode, config.num_layers)
            elapsed = time.time() - started
            rgb_cpu = rgb.detach().to("cpu", torch.float32).clamp(0.0, 1.0)

            result = ViewResult(
                index=index,
                image_name=view.image_name,
                mode=mode,
                seconds=elapsed,
                num_fragments=int(aux.get("num_fragments", 0)),
                points_active=list(aux.get("points_active", [])),
                vs_gt=compare(rgb_cpu, gt),
                vs_gt_masked=compare(rgb_cpu, gt, PARITY_EVAL_BORDER_PX),
                vs_reference_masked=(
                    compare(rgb_cpu, reference, PARITY_EVAL_BORDER_PX) if reference is not None else None
                ),
            )
            results.append(result)

            stem = f"{index:05d}_{mode}"
            save_png(out_dir / f"{stem}_ours.png", to_hwc(rgb_cpu))
            save_png(out_dir / f"{stem}_absdiff_gt.png", abs_diff_heatmap(rgb_cpu, gt))
            # Raw level-0 composite: first 3 texture channels, unmodified by
            # the network -- the "photographed vs inferred" honesty panel.
            level0 = layers[0][:3].detach().to("cpu", torch.float32)
            save_png(out_dir / f"{stem}_level0.png", to_hwc(level0.clamp(0.0, 1.0)))

            panels.extend(
                [to_hwc(rgb_cpu), abs_diff_heatmap(rgb_cpu, gt), to_hwc(level0.clamp(0.0, 1.0))]
            )
            gt_psnr = result.vs_gt_masked.psnr_db if result.vs_gt_masked else float("nan")
            labels.extend(
                [
                    f"ours [{mode}] PSNR {gt_psnr:.2f} dB vs GT",
                    f"|ours-GT| [{mode}]",
                    f"raw level-0 [{mode}]",
                ]
            )
            print(
                f"[parity] idx {index:>4} {mode:<10} {elapsed:6.1f}s  "
                f"PSNR(GT) {gt_psnr:6.2f} dB"
                + (
                    f"  PSNR(authors) {result.vs_reference_masked.psnr_db:6.2f} dB"
                    if result.vs_reference_masked is not None
                    else ""
                ),
                flush=True,
            )

        save_png(out_dir / f"{index:05d}_gt.png", to_hwc(gt))
        sheet = contact_sheet(panels, labels, cols=3)
        save_png(out_dir / f"contact_{index:05d}.png", sheet)
        sheet_rows.append(sheet)
        sheet_labels.append(f"{view.image_name} (index {index})")

    summary = contact_sheet(sheet_rows, sheet_labels, cols=1, cell_max=1400)
    save_png(out_dir / "summary_sheet.png", summary)

    report = {
        "config": asdict(config),
        "scene": {
            "dir": str(scene_dir),
            "name": scene_name,
            "num_images": len(scene),
            "render_scale": scale,
            "image_size": [first_view.height, first_view.width],
        },
        "checkpoint": {
            "dir": str(checkpoint_dir),
            "epoch": config.epoch,
            "num_points": len(points),
            "num_texture_channels": int(points.feat.shape[1]),
            "background_color": points.bg.detach().cpu().tolist(),
            "network_tensors_assigned": load_report.num_tensors_assigned,  # type: ignore[attr-defined]
            "network_parameters": net.parameter_count(),
        },
        "views": [
            {
                "index": r.index,
                "image": r.image_name,
                "mode": r.mode,
                "seconds": round(r.seconds, 3),
                "num_fragments": r.num_fragments,
                "points_active": r.points_active,
                "vs_ground_truth_full_frame": asdict(r.vs_gt) if r.vs_gt else None,
                "vs_ground_truth": asdict(r.vs_gt_masked) if r.vs_gt_masked else None,
                "vs_authors_render": asdict(r.vs_reference_masked) if r.vs_reference_masked else None,
                "authors_render_vs_ground_truth": (
                    asdict(authors_baseline[r.index]) if r.index in authors_baseline else None
                ),
            }
            for r in results
        ],
        "means": _means(results, config.modes),
        "authors_render_vs_ground_truth_mean": (
            {
                "psnr_db": sum(m.psnr_db for m in authors_baseline.values()) / len(authors_baseline),
                "ssim": sum(m.ssim for m in authors_baseline.values()) / len(authors_baseline),
                "lpips": sum(m.lpips for m in authors_baseline.values()) / len(authors_baseline),
            }
            if authors_baseline
            else None
        ),
        "eval_border_px": PARITY_EVAL_BORDER_PX,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(_readme(report), encoding="utf-8")
    return report


def _means(results: list[ViewResult], modes: tuple[str, ...]) -> dict:
    out: dict = {}
    for mode in modes:
        rows = [r for r in results if r.mode == mode]
        if not rows:
            continue
        entry: dict = {}
        for key, getter in (
            ("vs_ground_truth_full_frame", lambda r: r.vs_gt),
            ("vs_ground_truth", lambda r: r.vs_gt_masked),
            ("vs_authors_render", lambda r: r.vs_reference_masked),
        ):
            vals = [getter(r) for r in rows if getter(r) is not None]
            if vals:
                entry[key] = {
                    "psnr_db": sum(v.psnr_db for v in vals) / len(vals),
                    "ssim": sum(v.ssim for v in vals) / len(vals),
                    "lpips": sum(v.lpips for v in vals) / len(vals),
                }
        out[mode] = entry
    return out


def _readme(report: dict) -> str:
    lines = [
        "# TRIPS horse parity render",
        "",
        (
            f"Scene: `{report['scene']['dir']}` ({report['scene']['num_images']} images, "
            f"{report['scene']['image_size'][1]}x{report['scene']['image_size'][0]}, "
            f"render_scale {report['scene']['render_scale']})"
        ),
        (
            f"Checkpoint: `{report['checkpoint']['dir']}` @ {report['config']['epoch']}, "
            f"{report['checkpoint']['num_points']:,} points, "
            f"{report['checkpoint']['num_texture_channels']} texture channels, "
            f"{report['checkpoint']['network_parameters']:,} network parameters"
        ),
        "",
        (
            f"All metrics below crop {report['eval_border_px']} px off every side "
            "(TRIPS's `train_mask_border`, which it blacks out in its own saved test images)."
        ),
        "",
        "| view | mode | PSNR vs GT | SSIM vs GT | LPIPS vs GT | authors' PSNR vs GT | PSNR vs authors | s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for view in report["views"]:
        gt = view["vs_ground_truth"] or {}
        ref = view["vs_authors_render"] or {}
        base = view["authors_render_vs_ground_truth"] or {}

        def fmt(d: dict, key: str) -> str:
            return f"{d[key]:.4f}" if key in d else "-"

        lines.append(
            f"| {view['image']} | {view['mode']} | {fmt(gt, 'psnr_db')} | {fmt(gt, 'ssim')} | "
            f"{fmt(gt, 'lpips')} | {fmt(base, 'psnr_db')} | {fmt(ref, 'psnr_db')} | "
            f"{view['seconds']:.1f} |"
        )
    baseline = report.get("authors_render_vs_ground_truth_mean")
    if baseline:
        lines += [
            "",
            (
                f"Baseline -- the **authors' own** rendered test images vs the same photographs, "
                f"same {report['eval_border_px']}px border crop: "
                f"PSNR {baseline['psnr_db']:.3f} dB, SSIM {baseline['ssim']:.4f}, "
                f"LPIPS {baseline['lpips']:.4f}."
            ),
        ]
    lines += ["", "## Means", "", "| mode | PSNR vs GT | SSIM vs GT | LPIPS vs GT | PSNR vs authors |", "|---|---|---|---|---|"]
    for mode, entry in report["means"].items():
        gt = entry.get("vs_ground_truth", {})
        ref = entry.get("vs_authors_render", {})
        lines.append(
            f"| {mode} | {gt.get('psnr_db', float('nan')):.3f} | {gt.get('ssim', float('nan')):.4f} | "
            f"{gt.get('lpips', float('nan')):.4f} | {ref.get('psnr_db', float('nan')):.3f} |"
        )
    lines += [
        "",
        (
            "`trips` is the render path the published checkpoint actually used "
            "(`use_layer_point_size=true` via `fix_point_size=false`, see trippy/render/parity.py); "
            "`broadcast` and `trilinear` are ablations."
        ),
        "",
    ]
    return "\n".join(lines)
