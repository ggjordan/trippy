//! A minimal writer for numpy `.npz` archives, the twin of `brush_pyramid::npz`'s reader.
//!
//! Module: `trips_viewer::edit::npz_write`
//! Purpose: `EditDocument::save`'s brush sidecar (`docs/EDITOR.md` §1 "brush"):
//!     a large brush region's `cells`/`weights` are externalised into an
//!     `.npz` next to `edits.json`, the same file `trippy.edit.model.
//!     EditDocument._externalize_brush` writes on the Python side. Numpy's
//!     `.npz` is a plain ZIP of `.npy` members; `brush_pyramid::npz` already
//!     reads one without a ZIP crate dependency, and this is the write side
//!     of the same trade — the archives here are a handful of small arrays,
//!     never worth a dependency.
//! Invariants:
//!     - Always "stored" (ZIP method 0), never deflate: `np.savez` (not
//!       `np.savez_compressed`) is what the Python writer calls, and this
//!       matches it byte-for-byte in SHAPE even though the two are never
//!       compared as raw bytes (`docs/EDITOR.md` §1's brush entry: "the two
//!       files differ only in size" applies here too — timestamps in the
//!       ZIP headers are not something either loader reads).
//!     - Every member is `.npy` v1.0, little-endian, C-ordered — the only
//!       three things `brush_pyramid::npz::parse_npy` accepts, and the only
//!       three things numpy itself ever writes for a plain array.
//!     - No archive here ever needs ZIP64: a brush sidecar's `cells` is an
//!       `(N, 3)` `int32` array, bounded by [`super::brush::MAX_STROKE_CELLS`]
//!       cells per stroke — nowhere near the 4 GiB a ZIP64 member would need.
//! Units: none — this module is pure container/array encoding.
//! Related docs: `rust/crates/brush-pyramid/src/npz.rs` (the reader this is
//!     the write side of); `trippy.edit.model._externalize_brush` (the
//!     Python writer this must match array-for-array); the npy/zip format
//!     spec links in `brush_pyramid::npz`'s own module doc.

use std::path::Path;

/// CRC-32 (IEEE 802.3, polynomial `0xEDB88320`), computed directly rather than
/// through a lookup table: every archive this module writes is at most a few
/// tens of megabytes (a brush stroke's cell count is capped well below that),
/// so the table's setup cost is not worth the code.
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// A `.npy` v1.0 header for a little-endian, C-ordered array.
///
/// Padded with spaces (and a trailing `\n`, per the format) so the total
/// preamble (`magic + version + header-length field + header`) is a multiple
/// of 16 bytes, matching what numpy itself writes — not required for
/// [`super::super::npz::read_npz`] (or numpy's own reader) to parse it, but
/// cheap to match exactly.
fn npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let joined = shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        format!("({joined})")
    };
    let mut header =
        format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");
    const PREAMBLE: usize = 10; // magic(6) + version(2) + header-length field(2)
    let unpadded = header.len() + 1; // +1 for the trailing '\n'
    let pad = (16 - (PREAMBLE + unpadded) % 16) % 16;
    header.extend(std::iter::repeat_n(' ', pad));
    header.push('\n');

    let mut out = Vec::with_capacity(PREAMBLE + header.len());
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1); // major version
    out.push(0); // minor version
    #[allow(clippy::cast_possible_truncation)]
    let header_len = header.len() as u16;
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out
}

/// Encode an `(N, 3)` `int32` array (C order) as a complete `.npy` buffer.
fn npy_i32_2d(rows: &[[i32; 3]]) -> Vec<u8> {
    let mut out = npy_header("<i4", &[rows.len(), 3]);
    out.reserve(rows.len() * 12);
    for row in rows {
        for v in row {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Encode a `(N,)` `float32` array as a complete `.npy` buffer.
fn npy_f32_1d(values: &[f32]) -> Vec<u8> {
    let mut out = npy_header("<f4", &[values.len()]);
    out.reserve(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Append one ZIP "stored" (uncompressed) local file header + its data.
fn push_local_entry(buf: &mut Vec<u8>, name: &str, data: &[u8]) -> (u32, u32) {
    let crc = crc32(data);
    #[allow(clippy::cast_possible_truncation)]
    let offset = buf.len() as u32;
    buf.extend_from_slice(&0x0403_4b50_u32.to_le_bytes()); // local file header signature
    buf.extend_from_slice(&20_u16.to_le_bytes()); // version needed to extract (2.0)
    buf.extend_from_slice(&0_u16.to_le_bytes()); // general purpose flag
    buf.extend_from_slice(&0_u16.to_le_bytes()); // compression method: stored
    buf.extend_from_slice(&0_u16.to_le_bytes()); // mod file time
    buf.extend_from_slice(&0x0021_u16.to_le_bytes()); // mod file date: 1980-01-01
    buf.extend_from_slice(&crc.to_le_bytes());
    #[allow(clippy::cast_possible_truncation)]
    let size = data.len() as u32;
    buf.extend_from_slice(&size.to_le_bytes()); // compressed size == uncompressed (stored)
    buf.extend_from_slice(&size.to_le_bytes());
    #[allow(clippy::cast_possible_truncation)]
    let name_len = name.len() as u16;
    buf.extend_from_slice(&name_len.to_le_bytes());
    buf.extend_from_slice(&0_u16.to_le_bytes()); // extra field length
    buf.extend_from_slice(name.as_bytes());
    buf.extend_from_slice(data);
    (crc, offset)
}

/// Append one ZIP central directory record for an entry [`push_local_entry`] wrote.
fn push_central_entry(buf: &mut Vec<u8>, name: &str, crc: u32, size: u32, offset: u32) {
    buf.extend_from_slice(&0x0201_4b50_u32.to_le_bytes()); // central directory signature
    buf.extend_from_slice(&20_u16.to_le_bytes()); // version made by
    buf.extend_from_slice(&20_u16.to_le_bytes()); // version needed to extract
    buf.extend_from_slice(&0_u16.to_le_bytes()); // general purpose flag
    buf.extend_from_slice(&0_u16.to_le_bytes()); // compression method: stored
    buf.extend_from_slice(&0_u16.to_le_bytes()); // mod file time
    buf.extend_from_slice(&0x0021_u16.to_le_bytes()); // mod file date
    buf.extend_from_slice(&crc.to_le_bytes());
    buf.extend_from_slice(&size.to_le_bytes());
    buf.extend_from_slice(&size.to_le_bytes());
    #[allow(clippy::cast_possible_truncation)]
    let name_len = name.len() as u16;
    buf.extend_from_slice(&name_len.to_le_bytes());
    buf.extend_from_slice(&0_u16.to_le_bytes()); // extra field length
    buf.extend_from_slice(&0_u16.to_le_bytes()); // file comment length
    buf.extend_from_slice(&0_u16.to_le_bytes()); // disk number start
    buf.extend_from_slice(&0_u16.to_le_bytes()); // internal file attributes
    buf.extend_from_slice(&0_u32.to_le_bytes()); // external file attributes
    buf.extend_from_slice(&offset.to_le_bytes());
    buf.extend_from_slice(name.as_bytes());
}

/// Write a ZIP archive of `entries` (name, already-encoded `.npy` bytes), stored
/// (uncompressed) — an `.npz`, byte-valid for numpy's own `np.load` and for
/// [`super::super::npz::read_npz`].
fn write_zip_stored(path: &Path, entries: &[(&str, Vec<u8>)]) -> Result<(), String> {
    let mut buf = Vec::new();
    let mut central: Vec<(String, u32, u32, u32)> = Vec::with_capacity(entries.len());
    for (name, data) in entries {
        let (crc, offset) = push_local_entry(&mut buf, name, data);
        #[allow(clippy::cast_possible_truncation)]
        let size = data.len() as u32;
        central.push(((*name).to_owned(), crc, size, offset));
    }
    #[allow(clippy::cast_possible_truncation)]
    let central_offset = buf.len() as u32;
    for (name, crc, size, offset) in &central {
        push_central_entry(&mut buf, name, *crc, *size, *offset);
    }
    #[allow(clippy::cast_possible_truncation)]
    let central_size = buf.len() as u32 - central_offset;

    buf.extend_from_slice(&0x0605_4b50_u32.to_le_bytes()); // end-of-central-directory signature
    buf.extend_from_slice(&0_u16.to_le_bytes()); // disk number
    buf.extend_from_slice(&0_u16.to_le_bytes()); // disk with the start of the central directory
    #[allow(clippy::cast_possible_truncation)]
    let count = entries.len() as u16;
    buf.extend_from_slice(&count.to_le_bytes()); // entries on this disk
    buf.extend_from_slice(&count.to_le_bytes()); // entries total
    buf.extend_from_slice(&central_size.to_le_bytes());
    buf.extend_from_slice(&central_offset.to_le_bytes());
    buf.extend_from_slice(&0_u16.to_le_bytes()); // comment length

    std::fs::write(path, buf).map_err(|e| format!("{}: {e}", path.display()))
}

/// Write a brush region's sidecar: `cells` as `(N, 3)` `int32`, `weights`
/// (when given) as `(N,)` `float32` — the exact arrays and dtypes
/// `trippy.edit.model.EditDocument._externalize_brush` writes with
/// `cells_arr = _brush_cells_array(cells).astype(np.int32)` and
/// `np.asarray(weights, dtype=np.float32)`, in the same member order
/// (`cells` then `weights`, matching `np.savez(path, cells=..., weights=...)`
/// keyword order).
///
/// # Errors
/// Returns `Err` on any I/O failure.
pub(crate) fn write_brush_npz(
    path: &Path,
    cells: &[[i32; 3]],
    weights: Option<&[f32]>,
) -> Result<(), String> {
    let mut entries: Vec<(&str, Vec<u8>)> = vec![("cells.npy", npy_i32_2d(cells))];
    if let Some(w) = weights {
        entries.push(("weights.npy", npy_f32_1d(w)));
    }
    write_zip_stored(path, &entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_written_npz_reads_back_through_the_existing_reader() {
        let dir = std::env::temp_dir().join(format!("trips-npz-write-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("roundtrip.npz");
        let cells = vec![[0, 0, 0], [-1, 2, 3], [1_000_000, -1_000_000, 7]];
        write_brush_npz(&path, &cells, Some(&[0.25_f32, 1.0, 0.5])).unwrap();

        let members = brush_pyramid::npz::read_npz(&path).unwrap();
        let cells_arr = members.get("cells").expect("cells member");
        assert_eq!(cells_arr.shape, vec![3, 3]);
        assert_eq!(
            cells_arr.to_i32().unwrap(),
            vec![0, 0, 0, -1, 2, 3, 1_000_000, -1_000_000, 7]
        );

        let weights_arr = members.get("weights").expect("weights member");
        assert_eq!(weights_arr.shape, vec![3]);
        assert_eq!(weights_arr.to_f32().unwrap(), vec![0.25, 1.0, 0.5]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_written_npz_with_no_weights_has_one_member() {
        let dir = std::env::temp_dir().join(format!("trips-npz-write-nw-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cells-only.npz");
        write_brush_npz(&path, &[[4, 5, 6]], None).unwrap();

        let members = brush_pyramid::npz::read_npz(&path).unwrap();
        assert!(members.contains_key("cells"));
        assert!(!members.contains_key("weights"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_empty_cells_array_still_writes_a_valid_shape() {
        let dir = std::env::temp_dir().join(format!("trips-npz-write-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.npz");
        write_brush_npz(&path, &[], None).unwrap();
        let members = brush_pyramid::npz::read_npz(&path).unwrap();
        assert_eq!(members.get("cells").unwrap().shape, vec![0, 3]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
