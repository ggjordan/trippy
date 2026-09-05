"""Render the Gaussian PLY on the fly, for poses that have no precomputed render (design A).

Module: trippy.hybrid.gsrender_live
Purpose: a hybrid-A checkpoint needs the Gaussian block at *every* pose it is
    asked to render. Registered images are covered by
    `trippy.hybrid.render_splat_views`' precomputed triples on disk; the
    candidate report's dolly and off-path cameras are not photographed at all,
    so their block has to be produced live. This module wraps Splats'
    `gsrender.render` for that, and `gaussian_provider_for` wraps it into the
    one callback `trippy.render.candidate` (via `Trainer.gaussian_for_pose`)
    needs. The precomputed renders are deliberately NOT reachable through that
    callback: a pose *anchored to* an image is not that image's pose -- see
    `gaussian_provider_for`.
Invariants:
    - Splats' `gsrender.py` is imported BY PATH, never copied into this repo
      (AGENTS.md forbidden list); the import is deferred so CPU tests that
      inject a fake renderer never touch it or `~/Splats`.
    - The PLY is 1.7 GB: `LiveGaussianRenderer` loads it lazily and
      `_PLY_CACHE` keeps one loaded copy per (ply path, device) *per process*,
      so a report that calls `render_candidate` several times pays for it once.
      `clear_ply_cache()` exists for tests.
    - Real runs pass `device="mps"` and only ever execute inside a GPU-queue
      job (AGENTS.md: no MPS outside the queue). Nothing in this module picks
      a device on its own.
    - `max_hw` is always passed explicitly (gsrender's own default of 32
      corrupts near-camera footprints), with the same value design C's
      precomputed renders used, so a live block and a cached block are the
      same function of the pose.
Units: `R`/`t` are the world->camera COLMAP-frame pose (`x_cam = R @ x_world +
    t`), `K` is in level-0 pixels, depth comes back in world units.
Related docs: docs/EXPERIMENTS.md "Hybrid design A"; trippy.hybrid.
    render_splat_views (the batch renderer whose call convention this mirrors);
    experiments/EXP-0005-hybrid-c/README.md.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trippy.constants import HYBRID_C_GSRENDER_MAX_HW, HYBRID_C_GSRENDER_MIN_OPACITY
from trippy.hybrid.config_a import HybridConfig
from trippy.hybrid.gaussian_input import GaussianInputs, block_from_arrays
from trippy.hybrid.render_splat_views import DEFAULT_GSRENDER_TOOLS_DIR, _import_gsrender, _to_numpy

#: Process-wide loaded-PLY cache, keyed by `(resolved ply path, device)`. See module docstring.
_PLY_CACHE: dict[tuple[str, str], Any] = {}

#: A callback `(name, K, R, t, (H, W)) -> (G, H, W) tensor | None` -- what the trainer wants.
GaussianProvider = Callable[..., "torch.Tensor | None"]


def clear_ply_cache() -> None:
    """Drop every loaded PLY from `_PLY_CACHE` (tests; also frees ~1.7 GB per entry)."""
    _PLY_CACHE.clear()


def viewmat_from_rt(R: Any, t: Any) -> np.ndarray:
    """World->camera 4x4 viewmat for gsrender, from a COLMAP-frame `(R, t)`.

    Identical construction to `trippy.hybrid.render_splat_views.build_viewmat`
    (`V[:3, :3] = R`, `V[:3, 3] = t`, no inversion) -- that function starts
    from a quaternion, this one from an already-built rotation matrix.
    """
    if isinstance(R, torch.Tensor):
        R = R.detach().cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = np.asarray(R, dtype=np.float32)
    viewmat[:3, 3] = np.asarray(t, dtype=np.float32).reshape(3)
    return viewmat


class LiveGaussianRenderer:
    """Renders `ply_path` at arbitrary poses through Splats' `gsrender.render`.

    Args:
        ply_path: binary 3DGS PLY (e.g. `kkc_15000.ply`).
        device: `"mps"` in queue jobs, `"cpu"` never in practice (gsrender is
            an MPS path); forwarded to `gsrender.render` as its `dev` kwarg.
        max_hw, min_opacity: forwarded to `gsrender.render` (see module
            docstring on `max_hw`).
        tools_dir: directory holding `gsrender.py`; None uses
            `render_splat_views.DEFAULT_GSRENDER_TOOLS_DIR`.
        render_fn, load_ply_fn: injection points for CPU tests -- when both
            are given, neither `gsrender.py` nor `~/Splats` is ever touched.
    """

    def __init__(
        self,
        ply_path: str | Path,
        device: str = "mps",
        max_hw: int = HYBRID_C_GSRENDER_MAX_HW,
        min_opacity: float = HYBRID_C_GSRENDER_MIN_OPACITY,
        tools_dir: str | Path | None = None,
        render_fn: Callable[..., Any] | None = None,
        load_ply_fn: Callable[[str], Any] | None = None,
    ) -> None:
        self.ply_path = str(ply_path)
        self.device = str(device)
        self.max_hw = int(max_hw)
        self.min_opacity = float(min_opacity)
        self.tools_dir = Path(tools_dir) if tools_dir else DEFAULT_GSRENDER_TOOLS_DIR
        self._render_fn = render_fn
        self._load_ply_fn = load_ply_fn
        self._gaussians: Any | None = None

    def _resolve_fns(self) -> None:
        if self._render_fn is not None and self._load_ply_fn is not None:
            return
        gsrender = _import_gsrender(self.tools_dir)
        self._render_fn = self._render_fn or gsrender.render
        self._load_ply_fn = self._load_ply_fn or gsrender.load_ply

    def gaussians(self) -> Any:
        """The loaded Gaussians, from the per-process cache (loaded on first use)."""
        if self._gaussians is not None:
            return self._gaussians
        key = (self.ply_path, self.device)
        cached = _PLY_CACHE.get(key)
        if cached is None:
            self._resolve_fns()
            assert self._load_ply_fn is not None
            cached, _ply_k, _ply_imsize = self._load_ply_fn(self.ply_path)
            _PLY_CACHE[key] = cached
        self._gaussians = cached
        return cached

    def render(
        self, K: Any, R: Any, t: Any, image_hw: tuple[int, int]
    ) -> dict[str, np.ndarray]:
        """Render one pose. Returns `{"rgb": (H, W, 3), "alpha": (H, W), "depth": (H, W)}`."""
        gaussians = self.gaussians()
        self._resolve_fns()
        assert self._render_fn is not None
        height, width = int(image_hw[0]), int(image_hw[1])
        k = K.detach().cpu().numpy() if isinstance(K, torch.Tensor) else np.asarray(K)
        rgb, depth, alpha = self._render_fn(
            gaussians,
            viewmat_from_rt(R, t),
            np.asarray(k, dtype=np.float32),
            width,
            height,
            dev=self.device,
            max_hw=self.max_hw,
            min_opacity=self.min_opacity,
            return_depth=True,
        )
        return {"rgb": _to_numpy(rgb), "depth": _to_numpy(depth), "alpha": _to_numpy(alpha)}


def gaussian_provider_for(
    cfg: HybridConfig,
    inputs: GaussianInputs,
    device: str = "mps",
    live_renderer: LiveGaussianRenderer | None = None,
) -> GaussianProvider:
    """Build the `(name, K, R, t, image_hw) -> (G, H, W) tensor | None` provider callback.

    The block is **always rendered live at the pose given**, never substituted
    from `name`'s precomputed render: a `CameraPose.image_name` only means
    "anchored to that image", and every dolly/off-path pose is displaced from
    the photographed one (`trippy.render.offpath.offpath_poses`,
    `trippy.render.dolly.shade_dolly_poses`), so that image's render belongs to
    a different camera. `name` is accepted (and ignored) so the callback
    signature matches `Trainer.gaussian_for_pose`'s.

    When `cfg.ply_path` is empty and no `live_renderer` is given, the provider
    returns None -- an all-zero Gaussian block. That is honest: the network
    then runs on its TRIPS branch alone, exactly the state
    `hybrid.dropout_gaussian_p` trained it to survive. It is never a
    substituted or fabricated render.

    Args:
        cfg: the checkpoint's own `hybrid:` block (supplies `ply_path`,
            `channels`, `depth_scale`, `mask_by_alpha`, gsrender kwargs).
        inputs: the run's `GaussianInputs`. Only checked for agreement with
            `cfg` (a mismatched pair would silently produce a block of the
            wrong width); the live block is built straight from `cfg` so it
            lands on exactly the normalisation the trained frames used.
        device: forwarded to a freshly built `LiveGaussianRenderer`.
        live_renderer: use this renderer instead of building one (tests inject
            a fake; a caller may share one across `render_candidate` calls).
    """
    if inputs.num_channels != cfg.num_channels:
        raise ValueError(
            f"hybrid config and GaussianInputs disagree on the block width: cfg says "
            f"{cfg.num_channels}, inputs says {inputs.num_channels}"
        )
    renderer = live_renderer
    if renderer is None and cfg.ply_path:
        renderer = LiveGaussianRenderer(
            cfg.ply_path,
            device=device,
            max_hw=cfg.gsrender_max_hw,
            min_opacity=cfg.gsrender_min_opacity,
            tools_dir=cfg.gsrender_tools_dir or None,
        )

    def provider(
        name: str | None,  # part of the callback contract, deliberately unused: see docstring
        K: Any,
        R: Any,
        t: Any,
        image_hw: tuple[int, int],
    ) -> torch.Tensor | None:
        if renderer is None:
            return None
        arrays = renderer.render(K, R, t, image_hw)
        # Left on the CPU, like every other block: `GaussianInputs.attach` pools to each
        # pyramid level host-side and copies only the finished level to the device.
        return block_from_arrays(
            arrays,
            cfg.channels,
            float(cfg.depth_scale) if cfg.depth_scale else 1.0,
            cfg.mask_by_alpha,
        )

    return provider
