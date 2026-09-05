//! Minimal 8-bit RGB PNG writer.
//!
//! Module: `brush_pyramid::png`
//! Purpose: let the `render_frame` example write the frame it rendered
//!     without pulling in an image-codec dependency. Writing is all we ever
//!     need — nothing in trippy reads a PNG from Rust.
//! Invariants:
//!     - Emits only the one PNG flavour we need: colour type 2 (truecolour),
//!       bit depth 8, no interlacing, filter type 0 (None) on every scanline.
//!       That is the simplest conformant subset and every decoder handles it.
//!     - Deflate compression comes from `flate2` (already a dependency, for
//!       reading `np.savez_compressed` archives), so the CRC and the zlib
//!       stream are the only things written by hand here.
//!     - Float input is clamped to `[0, 1]` before scaling, so an out-of-range
//!       feature value saturates rather than wrapping to a wild byte.
//! Units: input samples are linear in `[0, 1]`; output bytes are 0..=255 with
//!     no gamma applied (the caller decides on tone mapping).
//! Related docs: PNG spec (RFC 2083); `examples/render_frame.rs`.

use std::io::Write;
use std::path::Path;

const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// Encode `rgb` (`height * width * 3` bytes, row-major) as a PNG byte stream.
///
/// # Errors
/// Returns `Err` if the buffer length disagrees with `width * height * 3`, or
/// if the deflate encoder fails.
pub fn encode_rgb8(rgb: &[u8], width: usize, height: usize) -> Result<Vec<u8>, String> {
    if rgb.len() != width * height * 3 {
        return Err(format!(
            "expected {} bytes for {width}x{height} RGB, got {}",
            width * height * 3,
            rgb.len()
        ));
    }
    if width == 0 || height == 0 {
        return Err("cannot encode a zero-sized image".to_owned());
    }

    // Each scanline is prefixed with its filter byte; 0 means "no filter".
    let mut raw = Vec::with_capacity(height * (1 + width * 3));
    for y in 0..height {
        raw.push(0u8);
        raw.extend_from_slice(&rgb[y * width * 3..(y + 1) * width * 3]);
    }
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&raw).map_err(|e| e.to_string())?;
    let idat = encoder.finish().map_err(|e| e.to_string())?;

    let mut out = Vec::with_capacity(idat.len() + 64);
    out.extend_from_slice(&PNG_SIGNATURE);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&u32::try_from(width).map_err(|_| "width too large")?.to_be_bytes());
    ihdr.extend_from_slice(&u32::try_from(height).map_err(|_| "height too large")?.to_be_bytes());
    ihdr.extend_from_slice(&[
        8, // bit depth
        2, // colour type 2 = truecolour RGB
        0, // compression method: deflate
        0, // filter method: adaptive
        0, // interlace: none
    ]);
    write_chunk(&mut out, b"IHDR", &ihdr);
    write_chunk(&mut out, b"IDAT", &idat);
    write_chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

/// Write `rgb` to `path` as a PNG. See [`encode_rgb8`].
///
/// # Errors
/// Returns `Err` on an encoding failure or an I/O failure.
pub fn write_rgb8(path: &Path, rgb: &[u8], width: usize, height: usize) -> Result<(), String> {
    let bytes = encode_rgb8(rgb, width, height)?;
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// Convert a channel-first `(C, h, w)` float image to interleaved RGB bytes.
///
/// Channels beyond the third are ignored; a 1- or 2-channel image repeats its
/// last channel so a single-feature debug render is still viewable.
///
/// # Arguments
/// - `feature`: `channels * height * width` samples, channel-first.
/// - `channels`, `height`, `width`: the image's dimensions.
/// - `scale`: multiplier applied before clamping, for exposure control. Use
///   `1.0` for features already in `[0, 1]`.
///
/// # Errors
/// Returns `Err` if `feature` is not `channels * height * width` long.
pub fn feature_to_rgb8(
    feature: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    scale: f32,
) -> Result<Vec<u8>, String> {
    if feature.len() != channels * height * width {
        return Err(format!(
            "expected {} samples, got {}",
            channels * height * width,
            feature.len()
        ));
    }
    if channels == 0 {
        return Err("cannot convert a zero-channel image".to_owned());
    }
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let src = c.min(channels - 1);
                let value = feature[(src * height + y) * width + x] * scale;
                // Clamp, then scale by 255: NaN clamps to 0.0 via `max`.
                let byte = (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                rgb[(y * width + x) * 3 + c] = byte;
            }
        }
    }
    Ok(rgb)
}

fn write_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc = Crc32::new();
    crc.update(kind);
    crc.update(data);
    out.extend_from_slice(&crc.finish().to_be_bytes());
}

/// The PNG/zlib CRC-32 (polynomial 0xEDB88320, reflected).
struct Crc32 {
    value: u32,
}

impl Crc32 {
    fn new() -> Self {
        Self { value: 0xFFFF_FFFF }
    }

    fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            let mut acc = (self.value ^ u32::from(byte)) & 0xFF;
            for _ in 0..8 {
                acc = if acc & 1 == 1 {
                    (acc >> 1) ^ 0xEDB8_8320
                } else {
                    acc >> 1
                };
            }
            self.value = acc ^ (self.value >> 8);
        }
    }

    fn finish(self) -> u32 {
        self.value ^ 0xFFFF_FFFF
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_known_check_value() {
        // The standard CRC-32 check value for the ASCII string "123456789".
        let mut crc = Crc32::new();
        crc.update(b"123456789");
        assert_eq!(crc.finish(), 0xCBF4_3926);
    }

    #[test]
    fn encoded_png_has_the_signature_and_the_three_chunks() {
        let rgb = vec![128u8; 4 * 3 * 3];
        let png = encode_rgb8(&rgb, 4, 3).expect("encode");
        assert_eq!(&png[..8], &PNG_SIGNATURE);
        let find = |needle: &[u8]| png.windows(4).position(|w| w == needle);
        let ihdr = find(b"IHDR").expect("IHDR");
        let idat = find(b"IDAT").expect("IDAT");
        let iend = find(b"IEND").expect("IEND");
        assert!(ihdr < idat && idat < iend, "chunks out of order");
        // IHDR payload starts right after the type: width then height.
        assert_eq!(&png[ihdr + 4..ihdr + 8], &4u32.to_be_bytes());
        assert_eq!(&png[ihdr + 8..ihdr + 12], &3u32.to_be_bytes());
    }

    #[test]
    fn a_wrong_sized_buffer_is_rejected() {
        assert!(encode_rgb8(&[0u8; 5], 4, 3).is_err());
        assert!(encode_rgb8(&[], 0, 0).is_err());
    }

    #[test]
    fn feature_to_rgb8_clamps_and_repeats_missing_channels() {
        // 1x1 image, one channel, value above 1 -> saturates to 255 in all
        // three output channels.
        let rgb = feature_to_rgb8(&[4.0], 1, 1, 1, 1.0).expect("convert");
        assert_eq!(rgb, vec![255, 255, 255]);
        // Negative saturates to 0.
        let rgb = feature_to_rgb8(&[-1.0], 1, 1, 1, 1.0).expect("convert");
        assert_eq!(rgb, vec![0, 0, 0]);
        // A 4-channel image drops the fourth.
        let rgb = feature_to_rgb8(&[0.0, 1.0, 0.0, 1.0], 4, 1, 1, 1.0).expect("convert");
        assert_eq!(rgb, vec![0, 255, 0]);
        // `scale` is applied before clamping.
        let rgb = feature_to_rgb8(&[0.5], 1, 1, 1, 2.0).expect("convert");
        assert_eq!(rgb, vec![255, 255, 255]);
    }
}
