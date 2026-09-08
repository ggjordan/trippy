"""Stream a 3DGS PLY through a keep mask, copying surviving rows verbatim.

Module: trippy.clean.ply_filter
Purpose: `trippy.train.export` builds a whole PLY in memory from trippy's
    own arrays. Design B needs the opposite: keep the ORIGINAL Gaussians
    bit-for-bit (every SH coefficient, the trained anisotropic scale and
    rotation `trippy.train.export` cannot represent) and only drop rows.
    So this is a row filter, not a writer: it re-emits the source header
    with a new vertex count and copies each surviving row's bytes.
Invariants:
    - A survivor's bytes in the output are byte-identical to its bytes in
      the input. No value is ever recomputed, rounded or re-encoded.
    - Header lines are re-emitted in source order with only the `element
      vertex <n>` count changed, so any property (f_rest_*, or a vendor
      extension trippy has never heard of) survives untouched.
    - Reading and writing are chunked (`CLEAN_PLY_CHUNK_ROWS` rows at a
      time): kklid_20000.ply is 8.9M rows x 236 bytes = 2.1 GB and this
      machine has OOM'd before (AGENTS.md Sec 6).
Units: bytes and row indices; no geometry is interpreted here.
Related docs: docs/GEOMETRY.md "3DGS PLY export mapping" (what the fields
    mean -- deliberately not used by this module); trippy.points.gaussian_ply
    (the reader whose header parse this shares).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trippy.constants import CLEAN_PLY_CHUNK_ROWS
from trippy.points.gaussian_ply import _NP_DTYPE, _read_ply_header


@dataclass(frozen=True)
class PlyLayout:
    """Everything needed to read or rewrite a binary_little_endian PLY's vertex block.

    Attributes:
        path: the file this was parsed from.
        count: vertex count declared in the header.
        props: `[(name, ply_type), ...]` in header order.
        header_bytes: the raw header, magic line through `end_header\\n`.
        data_offset: byte offset of the first vertex row.
        dtype: numpy structured dtype for one vertex row.
    """

    path: Path
    count: int
    props: list[tuple[str, str]]
    header_bytes: bytes
    data_offset: int
    dtype: np.dtype

    @property
    def itemsize(self) -> int:
        """Bytes per vertex row."""
        return int(self.dtype.itemsize)


def read_ply_layout(path: str | Path) -> PlyLayout:
    """Parse `path`'s header and record where/how its vertex rows are stored.

    Args:
        path: a binary_little_endian PLY.

    Returns:
        A `PlyLayout`.

    Raises:
        ValueError: not a binary_little_endian PLY, or its header declares
            trailing elements after `vertex` (this module rewrites the
            vertex count, so a following element's rows would be orphaned).
    """
    path = Path(path)
    with open(path, "rb") as f:
        count, props = _read_ply_header(f)
        data_offset = f.tell()
        f.seek(0)
        header_bytes = f.read(data_offset)
    dtype = np.dtype([(name, _NP_DTYPE[ptype]) for name, ptype in props])
    expected = data_offset + count * dtype.itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"{path}: body is {actual - data_offset} bytes but the vertex element needs "
            f"{count * dtype.itemsize} (a second element after `vertex` is unsupported)"
        )
    return PlyLayout(
        path=path,
        count=count,
        props=list(props),
        header_bytes=header_bytes,
        data_offset=data_offset,
        dtype=dtype,
    )


def read_columns(layout: PlyLayout, names: list[str], chunk_rows: int = CLEAN_PLY_CHUNK_ROWS) -> dict:
    """Read just `names` out of every vertex row, chunked.

    Reading the whole structured array costs `count * itemsize` bytes
    (2.1 GB on kklid_20000.ply); reading four columns costs ~140 MB.

    Args:
        layout: from `read_ply_layout`.
        names: property names to extract.
        chunk_rows: rows per `np.fromfile` call.

    Returns:
        `{name: (count,) array}`, each in the file's own dtype.

    Raises:
        KeyError: a requested property is not in the header.
    """
    have = {name for name, _ in layout.props}
    missing = [n for n in names if n not in have]
    if missing:
        raise KeyError(f"{layout.path}: no such vertex properties: {missing}")

    out = {n: np.empty(layout.count, dtype=layout.dtype[n]) for n in names}
    with open(layout.path, "rb") as f:
        f.seek(layout.data_offset)
        done = 0
        while done < layout.count:
            k = min(chunk_rows, layout.count - done)
            block = np.fromfile(f, dtype=layout.dtype, count=k)
            if block.shape[0] != k:
                raise ValueError(f"{layout.path}: short read at row {done} ({block.shape[0]} of {k})")
            for n in names:
                out[n][done : done + k] = block[n]
            done += k
    return out


def rewrite_header(header_bytes: bytes, new_count: int) -> bytes:
    """Return `header_bytes` with the `element vertex <n>` count replaced by `new_count`.

    Only that one line is touched; every comment, format and property line
    is re-emitted byte for byte (see module invariants).

    Args:
        header_bytes: the source header, magic line through `end_header\\n`.
        new_count: the survivor count.

    Returns:
        The rewritten header.

    Raises:
        ValueError: no `element vertex` line was found.
    """
    lines = header_bytes.split(b"\n")
    for i, line in enumerate(lines):
        tokens = line.strip().split()
        if len(tokens) == 3 and tokens[0] == b"element" and tokens[1] == b"vertex":
            lines[i] = b"element vertex %d" % new_count
            return b"\n".join(lines)
    raise ValueError("PLY header has no 'element vertex <n>' line")


def filter_ply(
    layout: PlyLayout,
    keep: np.ndarray,
    dest: str | Path,
    chunk_rows: int = CLEAN_PLY_CHUNK_ROWS,
) -> int:
    """Copy the rows `keep` selects from `layout.path` into `dest`.

    Args:
        layout: from `read_ply_layout`.
        keep: (count,) bool mask over source rows.
        dest: output path (parents created; written to a `.tmp` sibling and
            renamed, so a killed job never leaves a half-written PLY --
            the same rule `trippy.train.checkpoint_io` follows).
        chunk_rows: rows per read/write call.

    Returns:
        The number of rows written.

    Raises:
        ValueError: `keep` is not a (count,) boolean mask.
    """
    keep = np.asarray(keep)
    if keep.dtype != np.bool_ or keep.shape != (layout.count,):
        raise ValueError(f"keep must be a ({layout.count},) bool mask, got {keep.shape} {keep.dtype}")

    n_keep = int(keep.sum())
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with open(layout.path, "rb") as src, open(tmp, "wb") as out:
            out.write(rewrite_header(layout.header_bytes, n_keep))
            src.seek(layout.data_offset)
            done = 0
            while done < layout.count:
                k = min(chunk_rows, layout.count - done)
                block = np.fromfile(src, dtype=layout.dtype, count=k)
                if block.shape[0] != k:
                    raise ValueError(f"{layout.path}: short read at row {done}")
                survivors = block[keep[done : done + k]]
                if survivors.size:
                    survivors.tofile(out)
                done += k
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return n_keep
