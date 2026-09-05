//! CPU-only checks of the weight schema and the committed fixture.
//!
//! Module: `brush-unet` integration test `schema_cpu`
//! Purpose: the half of the PyTorch -> Burn bridge that can be verified with
//!     no GPU: that `tools/export_unet_safetensors.py fixture` really wrote
//!     the keys, shapes and metadata `brush_unet::config` declares, and that
//!     the reader rejects a file that drifts from them. This runs in
//!     `scripts/test.sh` on every push, so a schema change on either side
//!     fails immediately instead of at the next GPU job.
//! Invariants: no `gpu` feature, no Burn, no device — only `safetensors`.
//! Related docs: `rust/README.md` ("brush-unet weight schema").

use brush_unet::{fixture_dir, UnetConfig, Weights};

#[test]
fn the_committed_fixture_matches_the_declared_schema() {
    let path = fixture_dir().join("weights.safetensors");
    let weights = Weights::from_file(&path).unwrap_or_else(|e| panic!("{e}"));

    assert_eq!(weights.unet, UnetConfig::default(), "fixture is the 5-layer reference config");
    let camera = weights.camera.expect("fixture exports a camera");
    assert_eq!(camera.num_frames, 3);
    assert_eq!(camera.response_params, 25, "TRIPS's shipped response_params");
    assert_eq!((camera.image_height, camera.image_width), (24, 32));
    assert!(camera.enable_exposure && camera.enable_white_balance);
    assert!(camera.enable_vignette && camera.enable_response);

    // Every declared key is present with the declared shape (`Weights`
    // already enforces this on load; asserting again documents it).
    for (name, shape) in weights.unet.weight_shapes() {
        let tensor = weights.get_shaped(&name, &shape).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(tensor.len(), shape.iter().product::<usize>());
    }
    for name in camera.weight_keys() {
        weights.get(&name).unwrap_or_else(|e| panic!("{e}"));
    }

    // Total scalars in the file == the analytic parameter count + camera.
    let unet_scalars: usize = weights
        .unet
        .weight_keys()
        .iter()
        .map(|k| weights.get(k).expect("present").len())
        .sum();
    assert_eq!(unet_scalars, weights.unet.parameter_count());
    assert_eq!(unet_scalars, 59_675, "matches trippy's parameter_count()");
}

#[test]
fn the_fixture_metadata_records_the_generator() {
    let weights = Weights::from_file(&fixture_dir().join("weights.safetensors")).expect("load");
    assert_eq!(
        weights.metadata.get("source").map(String::as_str),
        Some("tools/export_unet_safetensors.py fixture")
    );
    assert_eq!(weights.metadata.get("seed").map(String::as_str), Some("20260906"));
}

#[test]
fn a_truncated_file_is_an_error_not_a_panic() {
    let path = fixture_dir().join("weights.safetensors");
    let bytes = std::fs::read(&path).expect("read fixture");
    let err = Weights::from_bytes(&bytes[..bytes.len() / 2]).expect_err("truncated file must fail");
    assert!(!err.is_empty());
}

#[test]
fn an_unimplemented_activation_is_rejected() {
    // The Burn port only implements TRIPS's shipped `elu`/`id`/`bilinear`
    // combination; a file asking for anything else must not load silently.
    let path = fixture_dir().join("weights.safetensors");
    let bytes = std::fs::read(&path).expect("read fixture");
    let patched = replace_metadata_value(&bytes, "\"activation\":\"elu\"", "\"activation\":\"rlu\"");
    let err = Weights::from_bytes(&patched).expect_err("unknown activation must fail");
    assert!(err.contains("activation"), "{err}");
}

/// Rewrite one metadata value in place. Both strings must be the same length
/// so the 8-byte header length and every data offset stay valid.
fn replace_metadata_value(bytes: &[u8], from: &str, to: &str) -> Vec<u8> {
    assert_eq!(from.len(), to.len(), "replacement must preserve the header length");
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + header_len]).expect("utf-8 header");
    let patched = header.replace(from, to);
    assert_ne!(patched, header, "pattern {from:?} not found in the header");
    let mut out = bytes[..8].to_vec();
    out.extend_from_slice(patched.as_bytes());
    out.extend_from_slice(&bytes[8 + header_len..]);
    out
}
