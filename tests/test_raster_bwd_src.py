"""The blend_bwd Metal source and its host wrapper, checked without a GPU.

Module: tests.test_raster_bwd_src
Invariants: these tests never touch MPS. They cover the two failure modes
    that would otherwise cost a GPU-queue round trip to discover: a Metal
    source that does not compile, and a host wrapper that binds a wrongly
    typed or non-contiguous buffer (torch.mps.compile_shader reinterprets raw
    storage, so either is silent garbage rather than an error). The kernel's
    numerics are covered by tests/test_raster_bwd_metal.py (marked `gpu`) and
    its formulas by tests/test_raster_bwd_ref.py (CPU).
Related docs: docs/ARCHITECTURE.md; AGENTS.md review checklist ("Metal kernel
    source files (*.metal) in trippy/raster/metal_src/").
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

from trippy.constants import RASTER_MAX_FRAGS, RASTER_SUPPORTED_CHANNELS
from trippy.raster import metal_lib


def test_kernel_source_lives_in_metal_src_as_a_real_file() -> None:
    """AGENTS.md requires .metal files on disk, not embedded strings."""
    path = metal_lib._METAL_BWD_SRC_PATH
    assert path.name == "blend_bwd.metal"
    assert path.parent.name == "metal_src"
    assert path.is_file()
    assert "kernel void blend_bwd" in metal_lib.BLEND_BWD_SOURCE


def _kernel_code() -> str:
    """blend_bwd.metal with every `//` comment (whole-line or trailing) removed.

    The header and the buffer table talk *about* atomics and about the
    division the kernel deliberately avoids, so the checks below have to look
    at code only.
    """
    return "\n".join(
        line.split("//", 1)[0] for line in metal_lib.BLEND_BWD_SOURCE.splitlines()
    )


def test_kernel_uses_no_atomics() -> None:
    """The backward is where TRIPS needs atomics most; ours must not."""
    assert "atomic" not in _kernel_code()


def test_kernel_never_divides_by_one_minus_alpha() -> None:
    """The suffix recurrences are division free; a `/` would mean a guard.

    TRIPS divides by `1 - alpha + 1e-9` (RenderBackward.cu:290). Our
    formulation carries the (1 - alpha) factor forward instead, so it is
    exact even at alpha == 1 (tests/test_raster_bwd_ref.py pins that). If a
    division ever appears in the kernel body, the epsilon question is back
    and this test should be revisited deliberately, not silently.
    """
    assert "/" not in _kernel_code()


@pytest.mark.parametrize("num_channels", RASTER_SUPPORTED_CHANNELS)
def test_render_bwd_source_substitutes_every_template_token(num_channels: int) -> None:
    """No TRIPPY_* placeholder may survive into the compiled source."""
    source = metal_lib.render_bwd_source(num_channels, RASTER_MAX_FRAGS)
    assert "TRIPPY_" not in source
    assert f"NUM_CHANNELS = {num_channels}" in source
    assert f"MAX_FRAGS = {RASTER_MAX_FRAGS}" in source


def test_render_bwd_source_rejects_unsupported_templates() -> None:
    """Channel counts the kernel is not templated for must raise."""
    with pytest.raises(ValueError, match="num_channels"):
        metal_lib.render_bwd_source(5, RASTER_MAX_FRAGS)
    with pytest.raises(ValueError, match="max_frags"):
        metal_lib.render_bwd_source(3, 0)


@pytest.mark.parametrize("num_channels", RASTER_SUPPORTED_CHANNELS)
def test_kernel_compiles_with_the_metal_frontend(num_channels: int) -> None:
    """`xcrun metal -c` is a CPU-only syntax/semantic check -- use it.

    This catches a broken kernel in milliseconds instead of a GPU-queue round
    trip. Skipped (not failed) where the Metal toolchain is absent.
    """
    if shutil.which("xcrun") is None:  # pragma: no cover (non-macOS)
        pytest.skip("xcrun not available")
    source = metal_lib.render_bwd_source(num_channels, RASTER_MAX_FRAGS)
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "blend_bwd.metal"
        src_path.write_text(source, encoding="utf-8")
        result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", str(src_path), "-o", str(Path(tmp) / "o.air")],
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode != 0 and "unable to find utility" in (result.stderr or ""):
        pytest.skip("Metal toolchain not installed")  # pragma: no cover (non-dev mac)
    assert result.returncode == 0, result.stderr


def test_blend_bwd_refuses_non_mps_tensors() -> None:
    """No silent CPU fallback; the error names the device."""
    with pytest.raises(ValueError, match="needs MPS tensors"):
        metal_lib.blend_bwd(
            torch.zeros(5, dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
            torch.zeros(4, dtype=torch.float32),
            torch.zeros((2, 3), dtype=torch.float32),
            torch.zeros(4, dtype=torch.int32),
            torch.zeros((4, 3), dtype=torch.float32),
            torch.zeros(4, dtype=torch.float32),
        )


def test_blend_bwd_validates_upstream_gradient_shapes() -> None:
    """A (P,) grad_out or a (P, C) grad_t_final would be read as garbage."""
    with pytest.raises(ValueError, match=r"feat must be \(N, C\)"):
        metal_lib.blend_bwd(
            torch.zeros(5, dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
            torch.zeros(4, dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(4, dtype=torch.int32),
            torch.zeros((4, 3), dtype=torch.float32),
            torch.zeros(4, dtype=torch.float32),
        )


def test_clear_cache_drops_both_libraries() -> None:
    """Tests that vary the template must be able to reset forward *and* back."""
    metal_lib.clear_cache()
    assert metal_lib._compiled_library.cache_info().currsize == 0
    assert metal_lib._compiled_bwd_library.cache_info().currsize == 0
