"""The Metal kernel source and its host wrapper, checked without a GPU.

Module: tests.test_raster_metal_src
Invariants: these tests never touch MPS -- they exercise the templating and
    the argument validation in trippy.raster.metal_lib, which is where a
    silent-garbage bug would come from (torch.mps.compile_shader binds raw
    storage, so a wrong dtype or a non-contiguous tensor is read as noise
    with no error). The kernel's numerical behaviour is covered by
    tests/test_raster_metal.py, which is marked `gpu`.
Related docs: docs/ARCHITECTURE.md; AGENTS.md review checklist ("Metal kernel
    source files (*.metal) in trippy/raster/metal_src/").
"""

from __future__ import annotations

import pytest
import torch

from trippy.constants import RASTER_MAX_FRAGS, RASTER_SUPPORTED_CHANNELS, RASTER_T_CUTOFF
from trippy.raster import metal_lib


def test_kernel_source_lives_in_metal_src_as_a_real_file() -> None:
    """AGENTS.md requires .metal files on disk, not embedded strings."""
    path = metal_lib._METAL_SRC_PATH
    assert path.name == "blend_fwd.metal"
    assert path.parent.name == "metal_src"
    assert path.is_file()
    assert "kernel void blend_fwd" in metal_lib.BLEND_FWD_SOURCE


def test_kernel_uses_no_atomics() -> None:
    """The whole design exists to avoid them (docs/ARCHITECTURE.md).

    Comment lines are stripped first -- the header comment talks *about*
    atomics at length; what must not appear is a call.
    """
    code = "\n".join(
        line for line in metal_lib.BLEND_FWD_SOURCE.splitlines() if not line.lstrip().startswith("//")
    )
    assert "atomic" not in code


@pytest.mark.parametrize("num_channels", RASTER_SUPPORTED_CHANNELS)
def test_render_source_substitutes_every_template_token(num_channels: int) -> None:
    """No TRIPPY_* placeholder may survive into the compiled source."""
    source = metal_lib.render_source(num_channels, RASTER_MAX_FRAGS, RASTER_T_CUTOFF)
    assert "TRIPPY_" not in source
    assert f"NUM_CHANNELS = {num_channels}" in source
    assert f"MAX_FRAGS = {RASTER_MAX_FRAGS}" in source
    assert "T_CUTOFF = 0.001f" in source


def test_render_source_rejects_unsupported_templates() -> None:
    """Channel counts the kernel is not templated for must raise."""
    with pytest.raises(ValueError, match="num_channels"):
        metal_lib.render_source(5, RASTER_MAX_FRAGS, RASTER_T_CUTOFF)
    with pytest.raises(ValueError, match="max_frags"):
        metal_lib.render_source(3, 0, RASTER_T_CUTOFF)


def test_blend_fwd_refuses_non_mps_tensors() -> None:
    """No silent CPU fallback; the error names the device."""
    offsets = torch.zeros(5, dtype=torch.int32)
    ids = torch.zeros(0, dtype=torch.int32)
    alpha = torch.zeros(0, dtype=torch.float32)
    depth = torch.zeros(0, dtype=torch.float32)
    feat = torch.zeros((2, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="needs MPS tensors"):
        metal_lib.blend_fwd(offsets, ids, alpha, depth, feat)


def test_blend_fwd_validates_feature_rank() -> None:
    """A (N,) feature tensor would be read as garbage by the kernel."""
    with pytest.raises(ValueError, match=r"feat must be \(N, C\)"):
        metal_lib.blend_fwd(
            torch.zeros(5, dtype=torch.int32),
            torch.zeros(0, dtype=torch.int32),
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
        )


def test_compiled_library_cache_can_be_cleared() -> None:
    """Compilation is memoised per template; tests must be able to reset it."""
    metal_lib.clear_cache()
    assert metal_lib._compiled_library.cache_info().currsize == 0
