//! Minimal reader for numpy `.npz` archives and the `.npy` members inside.
//!
//! Module: `brush_pyramid::npz`
//! Purpose: load point sets and expected images written by numpy
//!     (`trippy.points.PointSet.save_npz`, `tools/dump_raster_fixture.py`)
//!     without depending on a full ZIP crate. An `.npz` is a plain ZIP whose
//!     members are `.npy` files, and we only ever need to *read* them.
//! Invariants:
//!     - Both ZIP compression methods numpy emits are supported: **stored**
//!       (method 0, `np.savez`) and **deflate** (method 8,
//!       `np.savez_compressed`). Anything else is a clear error, never a
//!       silent misread.
//!     - Entries are located through the **central directory**, not by
//!       scanning local headers, so a member written with a streaming data
//!       descriptor still reports correct sizes.
//!     - ZIP64 members are rejected with an explicit error rather than
//!       silently truncated. numpy only emits them past 4 GiB per array,
//!       which no trippy point set reaches.
//!     - Only C-ordered, little-endian arrays are accepted. numpy writes
//!       `fortran_order: False` for everything trippy saves; a Fortran-ordered
//!       member would otherwise be read transposed.
//! Units: none — this module is pure container parsing.
//! Related docs: `rust/README.md`; the npy format spec is
//!     <https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html>.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

/// The element types trippy actually writes. Anything else is an error.
#[derive(Debug, Clone, PartialEq)]
pub enum NpyData {
    /// `<f4`
    F32(Vec<f32>),
    /// `<f8`
    F64(Vec<f64>),
    /// `<i4`
    I32(Vec<i32>),
    /// `<i8`
    I64(Vec<i64>),
    /// `|u1`
    U8(Vec<u8>),
}

/// One decoded `.npy` member: its shape plus its elements in C order.
#[derive(Debug, Clone, PartialEq)]
pub struct NpyArray {
    /// Dimensions, outermost first.
    pub shape: Vec<usize>,
    /// The elements, C-ordered (row-major).
    pub data: NpyData,
}

impl NpyArray {
    /// Total element count, `shape.iter().product()`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// True when the array holds no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Elements as `f32`, converting from any numeric dtype.
    ///
    /// # Errors
    /// Returns `Err` if the member is not a numeric array.
    pub fn to_f32(&self) -> Result<Vec<f32>, String> {
        Ok(match &self.data {
            NpyData::F32(v) => v.clone(),
            NpyData::F64(v) => v.iter().map(|&x| x as f32).collect(),
            NpyData::I32(v) => v.iter().map(|&x| x as f32).collect(),
            NpyData::I64(v) => v.iter().map(|&x| x as f32).collect(),
            NpyData::U8(v) => v.iter().map(|&x| f32::from(x)).collect(),
        })
    }

    /// Elements as `i32`, converting from any integer dtype.
    ///
    /// # Errors
    /// Returns `Err` for float members, where a silent truncation would hide
    /// a fixture-format mistake.
    pub fn to_i32(&self) -> Result<Vec<i32>, String> {
        Ok(match &self.data {
            NpyData::I32(v) => v.clone(),
            NpyData::I64(v) => v.iter().map(|&x| x as i32).collect(),
            NpyData::U8(v) => v.iter().map(|&x| i32::from(x)).collect(),
            NpyData::F32(_) | NpyData::F64(_) => {
                return Err("expected an integer array, got a float one".to_owned());
            }
        })
    }

    /// Check the shape, returning a clear error naming the array.
    ///
    /// # Errors
    /// Returns `Err` when `self.shape != expected`.
    pub fn expect_shape(&self, name: &str, expected: &[usize]) -> Result<(), String> {
        if self.shape == expected {
            Ok(())
        } else {
            Err(format!(
                "{name}: expected shape {expected:?}, got {:?}",
                self.shape
            ))
        }
    }
}

/// Read every member of an `.npz` archive, keyed by name without the `.npy`
/// suffix (the same key numpy's `np.load` uses).
///
/// # Errors
/// Returns `Err` on I/O failure, a malformed ZIP or npy header, an
/// unsupported compression method or dtype, or a ZIP64 member.
pub fn read_npz(path: &Path) -> Result<HashMap<String, NpyArray>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    read_npz_bytes(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// Read an `.npz` archive already in memory. See [`read_npz`].
///
/// # Errors
/// As [`read_npz`], minus the I/O cases.
pub fn read_npz_bytes(bytes: &[u8]) -> Result<HashMap<String, NpyArray>, String> {
    let mut out = HashMap::new();
    for (name, member) in zip_entries(bytes)? {
        let key = name.strip_suffix(".npy").unwrap_or(&name).to_owned();
        out.insert(key, parse_npy(&member).map_err(|e| format!("member {name}: {e}"))?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// ZIP container
// ---------------------------------------------------------------------------

const EOCD_SIGNATURE: u32 = 0x0605_4b50;
const CENTRAL_DIR_SIGNATURE: u32 = 0x0201_4b50;
const LOCAL_HEADER_SIGNATURE: u32 = 0x0403_4b50;
/// Sentinel a ZIP64 record puts in the 32-bit size/offset fields.
const ZIP64_SENTINEL: u32 = 0xFFFF_FFFF;
/// The end-of-central-directory record is 22 bytes plus a comment of at most
/// 64 KiB, so it always starts within this many bytes of the end.
const EOCD_MAX_SEARCH: usize = 22 + 0xFFFF;

fn u16_at(b: &[u8], off: usize) -> Result<u16, String> {
    b.get(off..off + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| format!("truncated at byte {off}"))
}

fn u32_at(b: &[u8], off: usize) -> Result<u32, String> {
    b.get(off..off + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| format!("truncated at byte {off}"))
}

/// Decode every member, in central-directory order, as `(name, bytes)`.
fn zip_entries(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>, String> {
    // Scan backwards for the end-of-central-directory signature. Going
    // backwards matters: a stored member could contain those four bytes.
    let search_from = bytes.len().saturating_sub(EOCD_MAX_SEARCH);
    let eocd = (search_from..bytes.len().saturating_sub(21))
        .rev()
        .find(|&i| u32_at(bytes, i) == Ok(EOCD_SIGNATURE))
        .ok_or("not a ZIP archive (no end-of-central-directory record)")?;

    let count = u16_at(bytes, eocd + 10)? as usize;
    let cd_offset = u32_at(bytes, eocd + 16)? as usize;
    if cd_offset == ZIP64_SENTINEL as usize {
        return Err("ZIP64 archives are not supported".to_owned());
    }

    let mut entries = Vec::with_capacity(count);
    let mut cursor = cd_offset;
    for _ in 0..count {
        if u32_at(bytes, cursor)? != CENTRAL_DIR_SIGNATURE {
            return Err(format!("bad central directory entry at byte {cursor}"));
        }
        let method = u16_at(bytes, cursor + 10)?;
        let compressed_size = u32_at(bytes, cursor + 20)?;
        let uncompressed_size = u32_at(bytes, cursor + 24)?;
        let name_len = u16_at(bytes, cursor + 28)? as usize;
        let extra_len = u16_at(bytes, cursor + 30)? as usize;
        let comment_len = u16_at(bytes, cursor + 32)? as usize;
        let local_offset = u32_at(bytes, cursor + 42)? as usize;

        if compressed_size == ZIP64_SENTINEL
            || uncompressed_size == ZIP64_SENTINEL
            || local_offset == ZIP64_SENTINEL as usize
        {
            return Err("ZIP64 members are not supported (array larger than 4 GiB)".to_owned());
        }

        let name = String::from_utf8_lossy(
            bytes
                .get(cursor + 46..cursor + 46 + name_len)
                .ok_or("truncated central directory file name")?,
        )
        .into_owned();

        // The local header's own name/extra lengths are what locate the data;
        // the central directory's `extra_len` may legitimately differ.
        if u32_at(bytes, local_offset)? != LOCAL_HEADER_SIGNATURE {
            return Err(format!("bad local header for {name}"));
        }
        let local_name_len = u16_at(bytes, local_offset + 26)? as usize;
        let local_extra_len = u16_at(bytes, local_offset + 28)? as usize;
        let data_start = local_offset + 30 + local_name_len + local_extra_len;
        let raw = bytes
            .get(data_start..data_start + compressed_size as usize)
            .ok_or_else(|| format!("truncated member data for {name}"))?;

        let data = match method {
            0 => raw.to_vec(),
            8 => inflate(raw, uncompressed_size as usize)
                .map_err(|e| format!("inflating {name}: {e}"))?,
            other => {
                return Err(format!(
                    "{name}: unsupported ZIP compression method {other} \
                     (expected 0 = stored or 8 = deflate)"
                ));
            }
        };
        if data.len() != uncompressed_size as usize {
            return Err(format!(
                "{name}: decompressed to {} bytes, header says {uncompressed_size}",
                data.len()
            ));
        }
        entries.push((name, data));

        cursor += 46 + name_len + extra_len + comment_len;
    }
    Ok(entries)
}

fn inflate(raw: &[u8], expected: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(expected);
    flate2::read::DeflateDecoder::new(raw)
        .read_to_end(&mut out)
        .map_err(|e| e.to_string())?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// npy member
// ---------------------------------------------------------------------------

const NPY_MAGIC: &[u8] = b"\x93NUMPY";

/// Parse one `.npy` buffer into shape + elements.
fn parse_npy(bytes: &[u8]) -> Result<NpyArray, String> {
    if bytes.len() < 10 || &bytes[..6] != NPY_MAGIC {
        return Err("not an npy file (bad magic)".to_owned());
    }
    let major = bytes[6];
    // v1 stores the header length as u16, v2 and v3 as u32.
    let (header_len, header_start) = if major == 1 {
        (u16_at(bytes, 8)? as usize, 10)
    } else {
        (u32_at(bytes, 8)? as usize, 12)
    };
    let header = std::str::from_utf8(
        bytes
            .get(header_start..header_start + header_len)
            .ok_or("truncated npy header")?,
    )
    .map_err(|_| "npy header is not UTF-8")?;

    let descr = header_value(header, "descr")?;
    if header_value(header, "fortran_order")?.contains("True") {
        return Err("Fortran-ordered arrays are not supported".to_owned());
    }
    let shape = parse_shape(header)?;
    let count: usize = shape.iter().product();

    let body = &bytes[header_start + header_len..];
    let data = match descr.trim_matches(['\'', '"'].as_ref()) {
        "<f4" | "=f4" | "f4" => NpyData::F32(read_scalars(body, count, 4, |c| {
            f32::from_le_bytes([c[0], c[1], c[2], c[3]])
        })?),
        "<f8" | "=f8" | "f8" => NpyData::F64(read_scalars(body, count, 8, |c| {
            f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
        })?),
        "<i4" | "=i4" | "i4" => NpyData::I32(read_scalars(body, count, 4, |c| {
            i32::from_le_bytes([c[0], c[1], c[2], c[3]])
        })?),
        "<i8" | "=i8" | "i8" => NpyData::I64(read_scalars(body, count, 8, |c| {
            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
        })?),
        "|u1" | "u1" | "|b1" => NpyData::U8(read_scalars(body, count, 1, |c| c[0])?),
        other => {
            return Err(format!(
                "unsupported dtype {other:?}; expected a little-endian f4/f8/i4/i8/u1"
            ));
        }
    };
    Ok(NpyArray { shape, data })
}

fn read_scalars<T>(
    body: &[u8],
    count: usize,
    width: usize,
    decode: impl Fn(&[u8]) -> T,
) -> Result<Vec<T>, String> {
    let need = count * width;
    if body.len() < need {
        return Err(format!(
            "npy body has {} bytes, need {need} for {count} elements",
            body.len()
        ));
    }
    Ok(body[..need].chunks_exact(width).map(decode).collect())
}

/// Pull `'<key>': <value>` out of an npy header dict, up to the next comma at
/// paren depth 0. The header is a Python literal, but the three keys we need
/// have simple values, so a full parser would be dead weight.
fn header_value(header: &str, key: &str) -> Result<String, String> {
    let needle = format!("'{key}':");
    let start = header
        .find(&needle)
        .ok_or_else(|| format!("npy header has no {key:?} key"))?
        + needle.len();
    let mut depth = 0i32;
    for (i, ch) in header[start..].char_indices() {
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => return Ok(header[start..start + i].trim().to_owned()),
            '}' if depth == 0 => return Ok(header[start..start + i].trim().to_owned()),
            _ => {}
        }
    }
    Ok(header[start..].trim().to_owned())
}

fn parse_shape(header: &str) -> Result<Vec<usize>, String> {
    let raw = header_value(header, "shape")?;
    let inner = raw
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim();
    if inner.is_empty() {
        // A 0-d array: shape (), one element.
        return Ok(vec![]);
    }
    inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| format!("bad shape component {s:?}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_value_stops_at_the_right_comma() {
        let h = "{'descr': '<f4', 'fortran_order': False, 'shape': (500, 3), }";
        assert_eq!(header_value(h, "descr").expect("descr"), "'<f4'");
        assert_eq!(
            header_value(h, "fortran_order").expect("order"),
            "False"
        );
        assert_eq!(header_value(h, "shape").expect("shape"), "(500, 3)");
    }

    #[test]
    fn shapes_parse_including_the_awkward_ones() {
        let one_d = "{'shape': (7,), }";
        assert_eq!(parse_shape(one_d).expect("1d"), vec![7]);
        let two_d = "{'shape': (500, 3), }";
        assert_eq!(parse_shape(two_d).expect("2d"), vec![500, 3]);
        let scalar = "{'shape': (), }";
        assert_eq!(parse_shape(scalar).expect("0d"), Vec::<usize>::new());
    }

    #[test]
    fn a_non_zip_buffer_is_a_clear_error_not_a_panic() {
        let err = read_npz_bytes(b"definitely not a zip file at all").expect_err("should fail");
        assert!(err.contains("not a ZIP archive"), "{err}");
    }

    #[test]
    fn a_truncated_npy_is_a_clear_error() {
        let err = parse_npy(b"\x93NUMPY\x01\x00").expect_err("should fail");
        assert!(err.contains("npy"), "{err}");
    }
}
