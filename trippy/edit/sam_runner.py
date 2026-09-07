"""SAM 3 as a subprocess: prompted segmentation of ONE photo, returned as a mask array.

Module: trippy.edit.sam_runner
Purpose: the segmentation half of docs/EDITOR.md Sec 3 "4. SAM 3 lift (E5)".
    SAM 3 is not vendored into trippy and never will be -- it lives in
    Jordan's Splats checkout (`~/Splats/tools/sam3/repo`, weights at
    `~/Splats/tools/sam3-weights/sam3.pt`) and is reached the same way
    `~/Splats/tools/make_masks3.py` reaches Vision: a self-contained child
    process, given an image path and a prompt, writing an array back.
    Everything is local; no image is sent anywhere (AGENTS.md Sec 6).

    This file is BOTH halves of that pipe:
      - imported as `trippy.edit.sam_runner`, it is the parent: it builds
        the command, runs it, and loads the mask (`Sam3Segmenter`);
      - executed as a script by Splats' SAM venv python, it is the child:
        it imports `sam3` by `sys.path`, builds the image model from the
        local checkpoint, prompts it, and writes `mask.npy` + `info.json`.

    The exact command the parent runs (also in docs/EDITOR.md Sec 3):

        /Users/nzbirdranch/Splats/tools/sam3/.venv/bin/python \\
            <trippy>/trippy/edit/sam_runner.py \\
            --repo /Users/nzbirdranch/Splats/tools/sam3/repo \\
            --weights /Users/nzbirdranch/Splats/tools/sam3-weights/sam3.pt \\
            --image <photo> --device cpu|mps \\
            --resolution 1008 --threshold 0.5 --mask-threshold 0.5 \\
            --kind box --box X0 Y0 X1 Y1 \\
            --out-mask <dir>/mask.npy --out-json <dir>/info.json

    Splats' SAM venv (python 3.11, torch 2.14, sam3 installed editable) is
    used because trippy's own .venv has none of SAM 3's dependencies
    (einops / timm / iopath / ftfy / pycocotools). Overridable with
    `TRIPPY_SAM3_PYTHON` / `TRIPPY_SAM3_REPO` / `TRIPPY_SAM3_WEIGHTS`.

Invariants:
    - Module-level imports are stdlib + numpy ONLY. The child runs under
      Splats' SAM venv, which cannot import `trippy` (python 3.11, and
      trippy needs >= 3.13); every `trippy.constants` import is therefore
      deferred into a parent-side function.
    - A mask is an ARRAY. This module writes `.npy` and nothing else -- no
      PNG, no overlay, no "mask preview" image -- and never opens the photo
      in the parent process (only the child, which is not a model, decodes
      pixels). docs/EDITOR.md Sec 3's privacy paragraph, AGENTS.md Sec 6.
    - Mask polarity is fixed and asserted in `info.json`: `True` = the
      prompted object, `False` = everything else, shape `(H, W)` of the
      photo's OWN pixel size, `dtype=bool`.
    - `--device mps` is GPU work: only ever inside a `scripts/gpu_submit.sh`
      job. The default is `cpu` so an accidental direct call cannot touch
      the GPU. The child never enables an MPS fallback of its own; if an op
      is missing on MPS the job fails loudly.
    - The child NEVER downloads anything: `build_sam3_image_model` is called
      with `load_from_HF=False` and an explicit local `checkpoint_path`.
Units: prompt coordinates are pixels of the photo file itself (origin at
    the top-left corner of pixel (0, 0), +u right, +v down); `box` is
    `x0 y0 x1 y1` inclusive-exclusive in those pixels; SAM 3's own API
    wants them normalised to [0, 1] and in cxcywh order, which
    `_child_main` converts to at the last moment.
Related docs: docs/EDITOR.md Sec 3 "4. SAM 3 lift (E5)" and Sec 7 (risks);
    ~/Splats/research/sam3-person.md (the local SAM 3 build: the
    hardcoded-CUDA patches, the triton shim, and the `torch._dynamo` stub
    reproduced in `_install_dynamo_stub` below);
    ~/Splats/tools/sam3/repo/sam3/model/sam3_image_processor.py (the API
    used: `set_image` -> `set_text_prompt` / `add_geometric_prompt`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "FakeSegmenter",
    "MaskFileSegmenter",
    "Sam3Segmenter",
    "SamPrompt",
    "fake_requested",
    "run_sam3",
    "sam3_command",
]


# --------------------------------------------------------------------------
# Prompts (shared by parent and child; plain stdlib so the child can use it)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SamPrompt:
    """One prompt for one photo, in that photo's own pixel coordinates.

    Attributes:
        kind: "point", "box" or "text" (`trippy.constants.SAM3_PROMPT_KINDS`).
        point: `(u, v)` pixels, for `kind="point"`.
        box: `(x0, y0, x1, y1)` pixels, for `kind="box"`; `x0 < x1`, `y0 < y1`.
        text: an open-vocabulary phrase, for `kind="text"`.
    """

    kind: str
    point: tuple[float, float] | None = None
    box: tuple[float, float, float, float] | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "point":
            if self.point is None:
                raise ValueError("SamPrompt(kind='point') needs point=(u, v)")
        elif self.kind == "box":
            if self.box is None:
                raise ValueError("SamPrompt(kind='box') needs box=(x0, y0, x1, y1)")
            x0, y0, x1, y1 = (float(v) for v in self.box)
            if not (x1 > x0 and y1 > y0):
                raise ValueError(f"SamPrompt box must have x1 > x0 and y1 > y0, got {self.box}")
        elif self.kind == "text":
            if not self.text:
                raise ValueError("SamPrompt(kind='text') needs a non-empty text")
        else:
            raise ValueError(f"SamPrompt.kind must be point|box|text, got {self.kind!r}")

    def to_json(self) -> dict[str, Any]:
        """JSON-shaped record of this prompt, for a summary or a job log."""
        doc: dict[str, Any] = {"kind": self.kind}
        if self.point is not None:
            doc["point"] = [float(v) for v in self.point]
        if self.box is not None:
            doc["box"] = [float(v) for v in self.box]
        if self.text is not None:
            doc["text"] = self.text
        return doc

    def cli_args(self) -> list[str]:
        """This prompt as `sam_runner.py` child-process arguments."""
        args = ["--kind", self.kind]
        if self.kind == "point":
            assert self.point is not None
            args += ["--point", repr(float(self.point[0])), repr(float(self.point[1]))]
        elif self.kind == "box":
            assert self.box is not None
            args += ["--box"] + [repr(float(v)) for v in self.box]
        else:
            assert self.text is not None
            args += ["--text", self.text]
        return args


# --------------------------------------------------------------------------
# Parent side
# --------------------------------------------------------------------------


def _defaults() -> tuple[str, str, str]:
    """`(python, repo, weights)` from the environment, else `trippy.constants`.

    Deferred import: this module's child half runs under Splats' SAM venv,
    which cannot import `trippy` at all (see the module docstring).
    """
    from trippy.constants import (
        SAM3_ENV_PYTHON,
        SAM3_ENV_REPO_DIR,
        SAM3_ENV_WEIGHTS,
        SAM3_REPO_DIR,
        SAM3_VENV_PYTHON,
        SAM3_WEIGHTS_PATH,
    )

    return (
        os.environ.get(SAM3_ENV_PYTHON, SAM3_VENV_PYTHON),
        os.environ.get(SAM3_ENV_REPO_DIR, SAM3_REPO_DIR),
        os.environ.get(SAM3_ENV_WEIGHTS, SAM3_WEIGHTS_PATH),
    )


def sam3_command(
    image: str | Path,
    prompt: SamPrompt,
    out_mask: str | Path,
    out_json: str | Path,
    device: str,
    python: str | Path | None = None,
    repo: str | Path | None = None,
    weights: str | Path | None = None,
    resolution: int | None = None,
    threshold: float | None = None,
    mask_threshold: float | None = None,
) -> list[str]:
    """Build the exact argv the SAM 3 child process is run with.

    Split out from `run_sam3` so a caller (or a test, or a job log) can see
    and record the command without running it -- docs/EDITOR.md asks for
    "the exact command" to be documented, and a command built in one place
    cannot drift from the one that ran.

    Args:
        image: the photograph to segment.
        prompt: what to segment (see `SamPrompt`).
        out_mask: `.npy` the child writes the `(H, W)` bool mask to.
        out_json: `.json` the child writes its own summary to.
        device: "cpu" or "mps" (`trippy.constants.SAM3_DEVICES`).
        python: SAM venv interpreter; default `SAM3_VENV_PYTHON`/env.
        repo: SAM 3 checkout to put on the child's `sys.path`.
        weights: local `sam3.pt`.
        resolution, threshold: `Sam3Processor` settings; default
            `SAM3_RESOLUTION` / `SAM3_CONFIDENCE_THRESHOLD`.
        mask_threshold: per-pixel probability cutoff; default
            `SAM3_MASK_THRESHOLD` (the processor's own 0.5).

    Returns:
        argv, ready for `subprocess.run`.
    """
    from trippy.constants import (
        SAM3_CONFIDENCE_THRESHOLD,
        SAM3_DEVICES,
        SAM3_MASK_THRESHOLD,
        SAM3_RESOLUTION,
    )

    if device not in SAM3_DEVICES:
        raise ValueError(f"device must be one of {SAM3_DEVICES}, got {device!r}")
    env_python, env_repo, env_weights = _defaults()
    argv = [
        str(python or env_python),
        str(Path(__file__).resolve()),
        "--repo",
        str(repo or env_repo),
        "--weights",
        str(weights or env_weights),
        "--image",
        str(image),
        "--device",
        device,
        "--resolution",
        str(int(resolution if resolution is not None else SAM3_RESOLUTION)),
        "--threshold",
        repr(float(threshold if threshold is not None else SAM3_CONFIDENCE_THRESHOLD)),
        "--mask-threshold",
        repr(float(mask_threshold if mask_threshold is not None else SAM3_MASK_THRESHOLD)),
        "--out-mask",
        str(out_mask),
        "--out-json",
        str(out_json),
    ]
    return argv + prompt.cli_args()


def run_sam3(
    image: str | Path,
    prompt: SamPrompt,
    device: str,
    work_dir: str | Path | None = None,
    keep_mask: bool = False,
    timeout: float | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Segment one photo with the local SAM 3, in a child process.

    Args:
        image: path to the photograph.
        prompt: see `SamPrompt`.
        device: "cpu" or "mps"; "mps" is GPU work (queue only).
        work_dir: where `mask.npy`/`info.json` are written; a temporary
            directory is used and deleted when this is None.
        keep_mask: with `work_dir`, keep `mask.npy` on disk afterwards
            (an array, never an image -- see the module docstring).
        timeout: seconds; default `SAM3_SUBPROCESS_TIMEOUT_S`.
        **kwargs: forwarded to `sam3_command` (python/repo/weights/...).

    Returns:
        `(mask, info)` -- `mask` is `(H, W)` bool in the photo's own pixel
        size (True = the prompted object), `info` is the child's summary
        plus `command` (the argv that ran), `seconds` and `device`.

    Raises:
        RuntimeError: the child failed; the message carries its stderr tail.
        FileNotFoundError: the SAM venv python or the photo is missing.
    """
    from trippy.constants import SAM3_SUBPROCESS_TIMEOUT_S

    image = Path(image)
    if not image.exists():
        raise FileNotFoundError(f"no photo at {image}")

    tmp: tempfile.TemporaryDirectory[str] | None = None
    if work_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="trippy-sam3-")
        out_dir = Path(tmp.name)
    else:
        out_dir = Path(work_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    try:
        out_mask = out_dir / "mask.npy"
        out_json = out_dir / "info.json"
        argv = sam3_command(image, prompt, out_mask, out_json, device, **kwargs)
        if not Path(argv[0]).exists():
            raise FileNotFoundError(
                f"SAM venv python not found at {argv[0]} "
                "(set TRIPPY_SAM3_PYTHON, or see trippy.constants.SAM3_VENV_PYTHON)"
            )
        limit = timeout if timeout is not None else SAM3_SUBPROCESS_TIMEOUT_S
        started = time.time()
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=limit,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            # Re-raised as RuntimeError so a hung SAM reads like any other
            # segmenter failure to the CLI (which turns it into exit 2)
            # rather than a traceback out of the subprocess module.
            raise RuntimeError(
                f"SAM 3 subprocess timed out after {limit:.0f}s on {image.name}; "
                "command: " + " ".join(argv)
            ) from exc
        seconds = time.time() - started
        if proc.returncode != 0 or not out_mask.exists():
            tail = (proc.stderr or "").strip().splitlines()[-20:]
            raise RuntimeError(
                f"SAM 3 subprocess failed (rc={proc.returncode}) for {image.name}:\n"
                + "\n".join(tail)
                + "\ncommand: "
                + " ".join(argv)
            )
        mask = np.load(out_mask)
        info = json.loads(out_json.read_text())
        info["command"] = argv
        info["seconds"] = float(seconds)
        info["device"] = device
        if keep_mask and work_dir is not None:
            info["mask_npy"] = str(out_mask)
        return mask.astype(bool), info
    finally:
        if tmp is not None:
            tmp.cleanup()


@dataclass
class Sam3Segmenter:
    """The default segmenter: local SAM 3 through `run_sam3`.

    A segmenter is any callable `(image_path, prompt) -> (mask, info)`.
    `trippy.edit.sam_lift` takes one as an argument so the whole lift is
    testable on CPU with a fake (docs/EDITOR.md Sec 3; tests/test_edit_sam.py).

    Attributes:
        device: "cpu" or "mps" (`mps` = GPU work, queue only).
        work_dir: keep each view's `mask.npy`/`info.json` under this
            directory instead of a temporary one (arrays only, never images).
        options: extra `sam3_command` kwargs (python/repo/weights/...).
    """

    device: str = "cpu"
    work_dir: str | Path | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __call__(self, image: str | Path, prompt: SamPrompt) -> tuple[np.ndarray, dict[str, Any]]:
        """Segment `image` with `prompt`; see `run_sam3` for the contract."""
        work_dir = None
        if self.work_dir is not None:
            work_dir = Path(self.work_dir) / Path(image).stem
        mask, info = run_sam3(
            image,
            prompt,
            self.device,
            work_dir=work_dir,
            keep_mask=work_dir is not None,
            **self.options,
        )
        info.setdefault("segmenter", "sam3")
        return mask, info


@dataclass
class MaskFileSegmenter:
    """A segmenter that reads masks from `.npy` files instead of running SAM 3.

    Two real uses, neither of them a stand-in for SAM 3's quality: running
    the lift on a machine (or in a test) where the 3.4 GB checkpoint is not
    worth loading, and re-lifting a mask a previous run already produced
    (`Sam3Segmenter(work_dir=...)` keeps them). It is recorded as
    `segmenter="mask-file"` in the summary so a bundle's provenance never
    silently claims SAM 3 ran.

    Attributes:
        masks: view image name -> `.npy` path holding an `(H, W)` bool mask.
    """

    masks: dict[str, str | Path]

    def __call__(self, image: str | Path, prompt: SamPrompt) -> tuple[np.ndarray, dict[str, Any]]:
        """Load the mask registered for `image`'s file name."""
        name = Path(image).name
        if name not in self.masks:
            raise KeyError(f"no mask file registered for {name} (have: {sorted(self.masks)})")
        path = Path(self.masks[name])
        mask = np.load(path).astype(bool)
        return mask, {
            "segmenter": "mask-file",
            "mask_npy": str(path),
            "prompt": prompt.to_json(),
            "height": int(mask.shape[0]),
            "width": int(mask.shape[1]),
            "mask_area_fraction": float(mask.mean()) if mask.size else 0.0,
        }


def _fake_point_radius() -> float:
    """[`FakeSegmenter`]'s default disc radius, photo pixels.

    A function, not a module constant, because this module's imports are
    stdlib + numpy only: the child half runs under Splats' python 3.11 SAM
    venv, which cannot import `trippy` at all (see the module invariants).
    """
    from trippy.constants import SAM_FAKE_POINT_RADIUS_PX

    return float(SAM_FAKE_POINT_RADIUS_PX)


@dataclass
class FakeSegmenter:
    """A synthetic segmenter: the lift's whole path, with no SAM 3 anywhere.

    `trippy edits sam --fake` (or `TRIPPY_SAM_FAKE=1`) installs this instead
    of [`Sam3Segmenter`]. It exists because the viewer's SAM tool
    (docs/EDITOR.md Sec 3 "4. SAM 3 lift (E5)") drives `trippy edits sam` as a
    CHILD PROCESS, and the viewer's own tests -- and the screenshot proof --
    need that child to start, print progress, finish, and hand back a real
    `pointset` region, on CPU, in under a second, without the 3.4 GB
    checkpoint or Splats' SAM venv. Everything downstream of the mask
    (projection, the depth gate, the majority vote, the region, the summary)
    is the SHIPPED code, so what the fake proves is the plumbing, never SAM 3's
    segmentation quality.

    The mask is a pure function of the prompt, in the photo's own pixels:
    a `box` prompt fills its rectangle, a `point` prompt fills a disc of
    `SAM_FAKE_POINT_RADIUS_PX` around it, and a `text` prompt fills the middle
    ninth of the frame (there is nothing in a phrase to derive geometry from,
    and a mask covering everything would make the depth gate meaningless).

    Nothing here decodes the photograph: only its size is read, from the file
    header, exactly as `trippy.edit.sam_lift.photo_size` does.

    Attributes:
        point_radius_px: disc radius for a `point` prompt, photo pixels.
        seen: every `(photo name, prompt)` this was asked for, in order --
            the viewer's tests assert on the neighbour prompts.
    """

    point_radius_px: float = field(default_factory=lambda: _fake_point_radius())
    seen: list[tuple[str, SamPrompt]] = field(default_factory=list)

    def __call__(self, image: str | Path, prompt: SamPrompt) -> tuple[np.ndarray, dict[str, Any]]:
        """Draw the prompt's own footprint as a mask at the photo's size."""
        # Deferred: `sam_lift` imports this module, so a module-level import
        # back the other way would be a cycle.
        from trippy.edit.sam_lift import photo_size

        path = Path(image)
        self.seen.append((path.name, prompt))
        width, height = photo_size(path)
        mask = np.zeros((height, width), dtype=bool)
        if prompt.kind == "box":
            x0, y0, x1, y1 = (float(v) for v in prompt.box)  # type: ignore[misc]
            c0 = max(0, min(width, round(min(x0, x1))))
            c1 = max(0, min(width, round(max(x0, x1))))
            r0 = max(0, min(height, round(min(y0, y1))))
            r1 = max(0, min(height, round(max(y0, y1))))
            mask[r0:r1, c0:c1] = True
        elif prompt.kind == "point":
            u, v = (float(c) for c in prompt.point)  # type: ignore[misc]
            rows = np.arange(height, dtype=np.float64)[:, None]
            cols = np.arange(width, dtype=np.float64)[None, :]
            mask = ((cols - u) ** 2 + (rows - v) ** 2) <= self.point_radius_px**2
        else:
            mask[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = True
        return mask, {
            "segmenter": "fake",
            "prompt": prompt.to_json(),
            "height": int(height),
            "width": int(width),
            "mask_area_fraction": float(mask.mean()) if mask.size else 0.0,
        }


def fake_requested(flag: bool = False) -> bool:
    """Whether this run must use [`FakeSegmenter`] rather than SAM 3.

    Args:
        flag: the `--fake` command-line switch.

    Returns:
        True when `flag` is set or `TRIPPY_SAM_FAKE` is "1" in the
        environment -- the switch the viewer sets on the child it spawns.
    """
    from trippy.constants import SAM_FAKE_ENV

    return bool(flag) or os.environ.get(SAM_FAKE_ENV, "") == "1"


# --------------------------------------------------------------------------
# Child side: runs under Splats' SAM venv python, imports sam3 by sys.path
# --------------------------------------------------------------------------


def _install_dynamo_stub() -> None:
    """Pre-seed `torch._dynamo` so `torchvision.ops` imports without triton.

    No `triton` wheel exists for macOS/arm64 (~/Splats/research/sam3-person.md
    "The real wall"), and `torch._inductor.runtime.hints` imports
    `triton.backends.compiler` unconditionally, so importing the real
    `torch._dynamo` explodes on this machine. torchvision only wants
    `is_compile_supported` from it, and SAM 3 never compiles anything here,
    so a stub with the handful of symbols the import chain touches is
    enough. This is trippy's own shim, not SAM 3 code: nothing from the
    Splats checkout is copied into this repo (docs/EDITOR.md Sec 3).
    """
    import types

    import torch

    def identity_decorator(fn: Any = None, *_a: Any, **_k: Any) -> Any:
        if fn is None or not callable(fn):
            return lambda f: f
        return fn

    dynamo = types.ModuleType("torch._dynamo")
    utils = types.ModuleType("torch._dynamo.utils")
    utils.is_compile_supported = lambda device_type: False  # type: ignore[attr-defined]
    utils.warn_once = lambda *a, **k: None  # type: ignore[attr-defined]
    dynamo.utils = utils  # type: ignore[attr-defined]
    dynamo.disable = identity_decorator  # type: ignore[attr-defined]
    dynamo.allow_in_graph = identity_decorator  # type: ignore[attr-defined]
    dynamo.mark_static = lambda *a, **k: None  # type: ignore[attr-defined]
    dynamo.mark_dynamic = lambda *a, **k: None  # type: ignore[attr-defined]
    dynamo.is_compiling = lambda: False  # type: ignore[attr-defined]
    dynamo.reset = lambda *a, **k: None  # type: ignore[attr-defined]

    class _OffFlags:
        """`torch._dynamo.config`: every flag reads False.

        `torch.utils.checkpoint` (SAM 3's segmentation head uses activation
        checkpointing) reads flags such as `error_on_nested_fx_trace` off
        this object. They are all "is dynamo doing something special"
        switches, and dynamo is doing nothing at all here, so False is the
        honest answer for every one of them rather than a guess.
        """

        def __getattr__(self, _name: str) -> bool:
            return False

    dynamo.config = _OffFlags()  # type: ignore[attr-defined]
    sys.modules.setdefault("torch._dynamo", dynamo)
    sys.modules.setdefault("torch._dynamo.utils", utils)
    torch._dynamo = dynamo  # type: ignore[attr-defined]


def _install_fp32_addmm() -> bool:
    """Take bfloat16 out of SAM 3's graph at its source. Both devices.

    `sam3/perflib/fused.py::addmm_act` casts bias, input and weight to
    bfloat16 unconditionally and returns bfloat16. On CUDA an autocast region
    round the forward pass then casts every FOLLOWING op's weights to match,
    so the bf16 tensor is harmless. Off CUDA it is not:

    - on CPU an autocast makes it *work*, but slowly -- bf16 on CPU is
      emulated, and the same lift measured 58.8 s with the autocast against
      **9.2 s** with this patch and no autocast, for the same mask (129,712
      mask pixels, 74,007 points selected; 2026-09-07);
    - on MPS an autocast is not even enough. Its op policy casts a
      convolution's INPUT but not its WEIGHT, so the pass dies with
      "Input type (MPSBFloat16Type) and weight type (torch.FloatTensor)
      should be the same" (job `trippy-edit-sam-1`).

    So the fix is not more autocast, it is no bfloat16: this rebinds
    `addmm_act` to the mathematically identical UNFUSED form in whatever
    dtype the input already has. `aten::_addmm_activation` is `addmm` plus
    `relu`/`gelu(approximate="none")`, and `F.linear` is that same `addmm`,
    so the only differences are the fusion and the dtype -- and fp32 is
    strictly MORE accurate than the bf16 path CUDA takes.

    Both bindings are replaced: `sam3.model.vitdet` does
    `from sam3.perflib.fused import addmm_act` at import time, so patching
    only the definition site would leave the caller on the old function.

    Returns:
        True when the patch was installed, False when SAM 3's layout has
        changed and there was nothing to patch -- reported in `info.json`
        rather than silently assumed.
    """
    import torch
    import torch.nn.functional as F

    def addmm_act_fp32(activation: Any, linear: Any, mat1: Any) -> Any:
        """`addmm_act`'s contract, unfused and without the bfloat16 cast."""
        if torch.is_grad_enabled():
            raise ValueError("Expected grad to be disabled.")
        out = F.linear(mat1, linear.weight.detach(), linear.bias.detach())
        if activation in (F.relu, torch.nn.ReLU):
            return F.relu(out)
        if activation in (F.gelu, torch.nn.GELU):
            return F.gelu(out)
        raise ValueError(f"Unexpected activation {activation}")

    installed = False
    for module_name in ("sam3.perflib.fused", "sam3.model.vitdet"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "addmm_act"):
            module.addmm_act = addmm_act_fp32  # type: ignore[attr-defined]
            installed = True
    return installed


def _install_real_rope() -> bool:
    """Make SAM 3's ViT use its REAL-valued rotary embedding. **MPS only.**

    The ViT's default rotary encoding goes through `torch.view_as_complex`,
    which MPS does not implement. SAM 3 already ships the real-valued twin
    (`ViT(use_rope_real=True)` -> `freqs_cis_real`/`freqs_cis_imag` and
    `apply_rotary_enc_real`) and it is arithmetically the same rotation with
    no learned parameters of its own, so the checkpoint loads identically --
    the model builder simply never passes the flag
    (`sam3/model_builder.py::_create_vit_backbone`). This rebinds that
    factory to default it on.

    Returns:
        True when the patch was installed, False when the factory is not
        where it used to be.
    """
    import functools

    module = sys.modules.get("sam3.model_builder")
    original = getattr(module, "_create_vit_backbone", None)
    if module is None or original is None:
        return False

    @functools.wraps(original)
    def create_vit_backbone_real(*args: Any, **kwargs: Any) -> Any:
        kwargs["use_rope_real"] = True
        return original(*args, **kwargs)

    module._create_vit_backbone = create_vit_backbone_real  # type: ignore[attr-defined]
    return True


def _move_module_caches(model: Any, device: str) -> int:
    """Move every module's plain-`dict` tensor cache onto `device`. **MPS only.**

    `sam3.model.position_encoding.PositionEmbeddingSine` warms a `self.cache`
    dict at construction time (a `precompute_resolution` optimisation for
    `torch.compile`), and `forward` returns the cached tensor verbatim when the
    input's `(H, W)` is a hit. That cache is a plain attribute, not a
    `register_buffer`, so `nn.Module.to(device)` does not touch it: after the
    model moves to MPS the visual features are on the GPU while their position
    encodings are still on the CPU, and `sam3_image.py::_get_img_feats`'s
    `x[img_ids]` dies with

        RuntimeError: indices should be either on cpu or on the same device as
        the indexed tensor (cpu)

    (job `trippy-edit-sam-3`, 2026-09-07). This does exactly what `.to()` would
    have done if the cache had been a buffer -- it moves data and changes no
    arithmetic. Generic over "any module with a `cache` dict of tensors" rather
    than hardcoding `PositionEmbeddingSine`, because the trap is the pattern,
    not the class.

    Args:
        model: the built SAM 3 model, already `.to(device)`.
        device: the torch device string.

    Returns:
        How many cached tensors were moved -- reported in `info.json`, because
        zero on MPS means SAM 3 stopped precomputing and this shim is no longer
        doing anything.
    """
    import torch

    moved = 0
    for module in model.modules():
        cache = getattr(module, "cache", None)
        if not isinstance(cache, dict):
            continue
        for key, value in list(cache.items()):
            if isinstance(value, torch.Tensor) and str(value.device) != str(device):
                cache[key] = value.to(device)
                moved += 1
    return moved


def _child_parser() -> argparse.ArgumentParser:
    """The child process's own arguments (see `sam3_command`)."""
    p = argparse.ArgumentParser(description="segment one photo with the local SAM 3")
    p.add_argument("--repo", required=True, help="SAM 3 checkout to put on sys.path")
    p.add_argument("--weights", required=True, help="local sam3.pt checkpoint")
    p.add_argument("--image", required=True, help="photograph to segment")
    p.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    p.add_argument("--resolution", type=int, default=1008)
    p.add_argument("--threshold", type=float, default=0.5, help="instance confidence threshold")
    p.add_argument(
        "--mask-threshold", type=float, default=0.5,
        help="probability above which a pixel is inside the mask (Sam3Processor's own value is 0.5)",
    )
    p.add_argument("--out-mask", required=True, help=".npy the (H, W) bool mask is written to")
    p.add_argument("--out-json", required=True, help=".json this run's summary is written to")
    p.add_argument("--kind", required=True, choices=("point", "box", "text"))
    p.add_argument("--point", type=float, nargs=2, default=None, metavar=("U", "V"))
    p.add_argument("--box", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--text", default=None)
    return p


def _pick_instance(
    masks: np.ndarray, scores: np.ndarray, kind: str, point: Any, box: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reduce SAM 3's per-instance masks to ONE mask, by prompt kind.

    SAM 3 returns every instance above the confidence threshold, which is
    the point of its open-vocabulary head. A lift needs one object:

    - "point": the highest-scoring instance whose mask covers the clicked
      pixel; if none does, the highest-scoring instance overall, with
      `contains_prompt=False` recorded so the caller can see it.
    - "box": the highest-scoring instance whose mask has the largest
      overlap with the prompt box (ties broken by score) -- the box says
      "the thing in here", so overlap with it is the selector.
    - "text": the UNION of every instance above threshold, because "select
      the chairs" means all of them (SAM 3's exhaustive-instance design).

    Args:
        masks: `(K, H, W)` bool.
        scores: `(K,)` float.
        kind, point, box: the prompt, in photo pixels.

    Returns:
        `(mask, info)` with `info` recording how the choice was made.
    """
    info: dict[str, Any] = {"n_instances": int(masks.shape[0])}
    if masks.shape[0] == 0:
        h, w = masks.shape[1], masks.shape[2]
        info["chosen"] = None
        info["contains_prompt"] = False
        return np.zeros((h, w), dtype=bool), info

    if kind == "text":
        info["chosen"] = "union"
        info["scores"] = [float(s) for s in scores]
        return masks.any(axis=0), info

    if kind == "point":
        u, v = round(float(point[0])), round(float(point[1]))
        u = min(max(u, 0), masks.shape[2] - 1)
        v = min(max(v, 0), masks.shape[1] - 1)
        hits = masks[:, v, u]
        if hits.any():
            candidates = np.flatnonzero(hits)
            chosen = int(candidates[int(np.argmax(scores[candidates]))])
            info["contains_prompt"] = True
        else:
            chosen = int(np.argmax(scores))
            info["contains_prompt"] = False
    else:
        x0, y0, x1, y1 = (float(v) for v in box)
        c0, r0 = int(max(0, np.floor(x0))), int(max(0, np.floor(y0)))
        c1, r1 = int(min(masks.shape[2], np.ceil(x1))), int(min(masks.shape[1], np.ceil(y1)))
        inside = masks[:, r0:r1, c0:c1].reshape(masks.shape[0], -1).sum(axis=1).astype(np.float64)
        best = float(inside.max()) if inside.size else 0.0
        if best > 0:
            candidates = np.flatnonzero(inside >= best)
            chosen = int(candidates[int(np.argmax(scores[candidates]))])
            info["contains_prompt"] = True
        else:
            chosen = int(np.argmax(scores))
            info["contains_prompt"] = False
        info["prompt_box_overlap_px"] = float(inside[chosen])

    info["chosen"] = chosen
    info["score"] = float(scores[chosen])
    info["scores"] = [float(s) for s in scores]
    return masks[chosen], info


def _child_main(argv: list[str] | None = None) -> int:
    """Load SAM 3, prompt it with one image, write `mask.npy` + `info.json`.

    Runs under Splats' SAM venv python; imports nothing from trippy.
    """
    args = _child_parser().parse_args(argv)

    _install_dynamo_stub()
    sys.path.insert(0, args.repo)

    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    # Right after the sam3 imports: both bindings of `addmm_act` exist to be
    # replaced, and nothing has run yet. Unconditional -- bfloat16 is a CUDA
    # optimisation that costs 6x on CPU and does not work at all on MPS.
    fp32_addmm = _install_fp32_addmm()
    if not fp32_addmm:
        raise RuntimeError(
            "could not patch sam3's addmm_act; its unconditional bfloat16 cast makes the "
            "CPU run 6x slower and the MPS run fail outright (see _install_fp32_addmm)"
        )
    # MPS has no `torch.view_as_complex`, which the ViT's default rotary
    # embedding uses; SAM 3's own real-valued twin is exact (_install_real_rope).
    real_rope = args.device == "mps" and _install_real_rope()
    if args.device == "mps" and not real_rope:
        raise RuntimeError(
            "could not patch sam3's ViT factory to use_rope_real=True; MPS has no "
            "torch.view_as_complex (see _install_real_rope)"
        )

    t0 = time.time()
    model = build_sam3_image_model(
        device=args.device,
        checkpoint_path=args.weights,
        load_from_HF=False,
    )
    # `build_sam3_image_model` only moves the model when `device == "cuda"`;
    # everything else is left on the CPU while `Sam3Processor` puts the IMAGE
    # on `device`, which on MPS fails with "Input type (MPSFloatType) and
    # weight type (torch.FloatTensor) should be the same" (job
    # `trippy-edit-sam-2`, 2026-09-07). A no-op when device is already cpu.
    model = model.to(args.device)
    # `.to()` moves parameters and buffers; SAM 3's position encoder keeps its
    # precomputed encodings in a plain dict, which it does not (see
    # `_move_module_caches`).
    moved_caches = _move_module_caches(model, args.device) if args.device == "mps" else 0
    processor = Sam3Processor(
        model, resolution=args.resolution, device=args.device, confidence_threshold=args.threshold
    )
    t_load = time.time() - t0

    t1 = time.time()
    # No autocast, on either device: `_install_fp32_addmm` above already took
    # bfloat16 out of the graph at its source, and adding an autocast back
    # would only reintroduce the mixed dtypes it removed (and, on CPU, the 6x
    # emulation cost). Nothing here is an MPS *fallback* either:
    # PYTORCH_ENABLE_MPS_FALLBACK stays whatever the queue job set it to.
    with Image.open(args.image) as handle:
        image = handle.convert("RGB")
        width, height = image.size
        state = processor.set_image(image)

    if args.kind == "text":
        state = processor.set_text_prompt(args.text, state)
    elif args.kind == "box":
        x0, y0, x1, y1 = args.box
        cx, cy = (x0 + x1) / 2.0 / width, (y0 + y1) / 2.0 / height
        bw, bh = abs(x1 - x0) / width, abs(y1 - y0) / height
        state = processor.add_geometric_prompt([cx, cy, bw, bh], True, state)
    else:
        # `Sam3Processor` exposes boxes but not points; the underlying
        # Prompt does carry them (geometry_encoders.Prompt.append_points,
        # "points in normalized xy"), and `add_geometric_prompt`'s own
        # body is exactly this sequence with `append_boxes` instead.
        u, v = args.point
        if "language_features" not in state["backbone_out"]:
            state["backbone_out"].update(
                model.backbone.forward_text(["visual"], device=args.device)
            )
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = model._get_dummy_prompt()
        points = torch.tensor([u / width, v / height], device=args.device, dtype=torch.float32)
        state["geometric_prompt"].append_points(
            points.view(1, 1, 2),
            torch.ones(1, 1, device=args.device, dtype=torch.long),
            mask=torch.zeros(1, 1, device=args.device, dtype=torch.bool),
        )
        state = processor._forward_grounding(state)
    t_infer = time.time() - t1

    # `masks_logits` is the per-pixel PROBABILITY at the photo's own
    # resolution (the processor sigmoids and resizes it before storing);
    # `state["masks"]` is exactly this thresholded at 0.5. Thresholding here
    # instead makes that 0.5 a flag: on a low-contrast subject the model can
    # be confidently right about WHERE the object is while peaking at ~0.4,
    # which at 0.5 returns only the object's outline (measured on a synthetic
    # flat-colour disc, 2026-09-07).
    probs = state["masks_logits"]
    if probs.ndim == 4:
        probs = probs[:, 0]
    # `.float()` first: numpy has no bfloat16 and no MPS.
    probs_np = probs.detach().to("cpu").float().numpy()
    masks_np = probs_np > args.mask_threshold
    scores_np = state["scores"].detach().to("cpu").float().numpy().astype(np.float64)

    mask, info = _pick_instance(masks_np, scores_np, args.kind, args.point, args.box)
    mask = np.ascontiguousarray(mask.astype(bool))
    if mask.shape != (height, width):
        raise RuntimeError(f"SAM 3 returned a {mask.shape} mask for a {(height, width)} photo")

    out_mask = Path(args.out_mask)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_mask, mask)

    info.update(
        {
            "segmenter": "sam3",
            "image": str(args.image),
            "width": int(width),
            "height": int(height),
            "mask_area_fraction": float(mask.mean()),
            "mask_area_px": int(mask.sum()),
            "device": args.device,
            "fp32_addmm": bool(fp32_addmm),
            "real_rope": bool(real_rope),
            "moved_caches": int(moved_caches),
            "resolution": int(args.resolution),
            "threshold": float(args.threshold),
            "mask_threshold": float(args.mask_threshold),
            "mask_prob_max": float(probs_np.max()) if probs_np.size else 0.0,
            "weights": str(args.weights),
            "seconds_model_load": float(t_load),
            "seconds_inference": float(t_infer),
            "prompt": {
                "kind": args.kind,
                "point": list(args.point) if args.point else None,
                "box": list(args.box) if args.box else None,
                "text": args.text,
            },
        }
    )
    Path(args.out_json).write_text(json.dumps(info, indent=2))
    print(
        f"sam3: {mask.sum()} px ({mask.mean():.6f} of frame), "
        f"load {t_load:.1f}s, infer {t_infer:.1f}s, device {args.device}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the child half, run by SAM's venv
    raise SystemExit(_child_main())
