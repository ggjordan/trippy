//! CPU-only checks of the BLEND GATE half of the weight schema.
//!
//! Module: `brush-unet` integration test `gate_schema_cpu`
//! Purpose: prove the Burn-side reader accepts the four-channel weight file
//!     `trippy.net.export_safetensors` writes for a run trained with
//!     `hybrid.gate.enabled` -- and rejects every way that file could be
//!     inconsistent -- without a GPU, a device or a committed fixture.
//!     Together with `tests/test_hybrid_gate_export.py` (which proves the
//!     Python writer's side of the same contract, including the numeric round
//!     trip of the gate head) this is the schema half of the Python -> Burn
//!     bridge for the gate. The *numeric* agreement of the two implementations
//!     is `tests/parity_gpu.rs`'s job and needs a GPU.
//! Invariants:
//!     - The file is built HERE, byte by byte, from the safetensors container
//!       rules (8-byte LE header length, JSON header, raw f32 data) -- the same
//!       rules `trippy/net/export_safetensors.py` writes by hand. Two
//!       independent implementations of one container is the point: a drift in
//!       either is a failing test rather than a corrupt bundle.
//!     - No `gpu` feature, no Burn. Runs in `scripts/test.sh` on every push.
//! Related docs: `rust/README.md` ("brush-unet weight schema");
//!     `trippy/hybrid/gate.py`; `docs/EXPERIMENTS.md` "The blend gate".

use std::collections::BTreeMap;

use brush_unet::{GateConfig, UnetConfig, Weights, GATE_CHANNEL, RGB_CHANNELS};

/// Build a minimal, schema-complete weight file with `out_channels` channels.
///
/// `extra_meta` is merged last, so a test can corrupt exactly one key.
fn weight_file(out_channels: usize, extra_meta: &[(&str, &str)]) -> Vec<u8> {
    weight_file_with_camera(out_channels, extra_meta, None)
}

/// As [`weight_file`], plus a camera block with `response_params` LUT points.
fn weight_file_with_camera(
    out_channels: usize,
    extra_meta: &[(&str, &str)],
    camera: Option<usize>,
) -> Vec<u8> {
    let config = UnetConfig {
        out_channels,
        ..UnetConfig::default()
    };

    let mut tensors: Vec<(String, Vec<usize>)> = config.weight_shapes();
    if let Some(response_params) = camera {
        // The tone mapper's LUT is RGB-only -- three rows whatever the network
        // emits, because the gate channel is never tone-mapped.
        tensors.push(("camera.response".into(), vec![RGB_CHANNELS, response_params]));
        tensors.push(("camera.exposure".into(), vec![2]));
        tensors.push(("camera.white_balance".into(), vec![2, 3]));
        tensors.push(("camera.vignette_params".into(), vec![3]));
        tensors.push(("camera.vignette_center".into(), vec![2]));
    }
    tensors.sort_by(|a, b| a.0.cmp(&b.0));

    let mut header = serde_json::Map::new();
    let mut meta = serde_json::Map::new();
    for (key, value) in [
        ("format", "trippy-unet-1"),
        ("num_layers", &config.num_layers.to_string()[..]),
        ("filters", &config.filters.to_string()[..]),
        ("in_channels", &config.in_channels.to_string()[..]),
        ("out_channels", &config.out_channels.to_string()[..]),
        ("activation", "elu"),
        ("norm", "id"),
        ("upsample_mode", "bilinear"),
        ("last_act", "id"),
        ("has_camera", if camera.is_some() { "1" } else { "0" }),
    ] {
        meta.insert(key.into(), serde_json::Value::String(value.into()));
    }
    if let Some(response_params) = camera {
        for (key, value) in [
            ("num_frames", "2".to_owned()),
            ("response_params", response_params.to_string()),
            ("image_height", "24".to_owned()),
            ("image_width", "32".to_owned()),
            ("enable_exposure", "1".to_owned()),
            ("enable_white_balance", "1".to_owned()),
            ("enable_vignette", "1".to_owned()),
            ("enable_response", "1".to_owned()),
        ] {
            meta.insert(key.into(), serde_json::Value::String(value));
        }
    }
    if config.has_gate() {
        meta.insert("gate".into(), serde_json::Value::String("1".into()));
        meta.insert(
            "gate_channel".into(),
            serde_json::Value::String(GATE_CHANNEL.to_string()),
        );
        meta.insert("gate_scale".into(), serde_json::Value::String("1.0".into()));
    }
    for (key, value) in extra_meta {
        if value.is_empty() {
            meta.remove(*key);
        } else {
            meta.insert((*key).into(), serde_json::Value::String((*value).into()));
        }
    }
    header.insert("__metadata__".into(), serde_json::Value::Object(meta));

    let mut body: Vec<u8> = Vec::new();
    for (name, shape) in &tensors {
        let count: usize = shape.iter().product();
        let begin = body.len();
        for i in 0..count {
            // Deterministic, distinguishable values so a mis-sliced tensor is visible.
            body.extend_from_slice(&(i as f32).to_le_bytes());
        }
        let mut entry = serde_json::Map::new();
        entry.insert("dtype".into(), serde_json::Value::String("F32".into()));
        entry.insert(
            "shape".into(),
            serde_json::Value::Array(shape.iter().map(|d| (*d).into()).collect()),
        );
        entry.insert(
            "data_offsets".into(),
            serde_json::Value::Array(vec![begin.into(), body.len().into()]),
        );
        header.insert(name.clone(), serde_json::Value::Object(entry));
    }

    let mut header_json = serde_json::to_vec(&serde_json::Value::Object(header)).expect("header");
    while header_json.len() % 8 != 0 {
        header_json.push(b' ');
    }
    let mut out = (header_json.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header_json);
    out.extend_from_slice(&body);
    out
}

#[test]
fn a_four_channel_file_loads_and_declares_the_gate() {
    let weights = Weights::from_bytes(&weight_file(RGB_CHANNELS + 1, &[])).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(weights.unet.out_channels, RGB_CHANNELS + 1);
    assert!(weights.unet.has_gate());
    let gate = weights.gate.expect("a four-channel file declares a gate");
    assert_eq!(gate.channel, GATE_CHANNEL);
    assert_eq!(gate.scale, 1.0);
}

#[test]
fn the_gate_widens_only_the_final_conv() {
    let plain = UnetConfig::default();
    let gated = UnetConfig {
        out_channels: RGB_CHANNELS + 1,
        ..plain
    };
    // Exactly the same key set: the gate rides in `unet.final.*`, it does not add
    // a tensor -- which is what keeps `check_schema`'s "no unknown tensor" rule
    // satisfied by a Python export that knows nothing about this crate.
    assert_eq!(plain.weight_keys(), gated.weight_keys());

    let plain_shapes: BTreeMap<_, _> = plain.weight_shapes().into_iter().collect();
    let gated_shapes: BTreeMap<_, _> = gated.weight_shapes().into_iter().collect();
    for (name, shape) in &plain_shapes {
        if name.starts_with("unet.final.") {
            continue;
        }
        assert_eq!(gated_shapes.get(name), Some(shape), "{name} must not change");
    }
    assert_eq!(gated_shapes["unet.final.weight"], vec![4, plain.filters, 1, 1]);
    assert_eq!(gated_shapes["unet.final.bias"], vec![4]);
    // One extra output row of the 1x1 conv, plus its bias.
    assert_eq!(
        gated.parameter_count() - plain.parameter_count(),
        plain.filters + 1
    );
}

#[test]
fn a_three_channel_file_has_no_gate() {
    let weights = Weights::from_bytes(&weight_file(RGB_CHANNELS, &[])).unwrap_or_else(|e| panic!("{e}"));
    assert!(!weights.unet.has_gate());
    assert!(weights.gate.is_none());
    assert!(!weights.metadata.contains_key("gate"), "a gate-less export writes no gate key");
}

#[test]
fn a_gate_flag_without_the_channel_is_rejected() {
    // Metadata claims a gate, but `out_channels` is 3: a corrupt export, and
    // guessing which one is right would render the wrong picture silently.
    let err = Weights::from_bytes(&weight_file(RGB_CHANNELS, &[("gate", "1")]))
        .expect_err("mismatched gate metadata must fail");
    assert!(err.contains("gate"), "{err}");
}

#[test]
fn a_fourth_channel_without_the_gate_flag_is_rejected() {
    let err = Weights::from_bytes(&weight_file(RGB_CHANNELS + 1, &[("gate", "")]))
        .expect_err("an undeclared fourth channel must fail");
    assert!(err.contains("gate"), "{err}");
}

#[test]
fn a_gate_on_a_channel_other_than_three_is_rejected() {
    // Colour must stay first; a reader that quietly accepted rgb-after-gate would
    // tone-map the gate and blend the blue channel.
    let err = Weights::from_bytes(&weight_file(RGB_CHANNELS + 1, &[("gate_channel", "0")]))
        .expect_err("a relocated gate channel must fail");
    assert!(err.contains("gate_channel"), "{err}");
}

#[test]
fn a_fifth_output_channel_is_rejected() {
    let err = Weights::from_bytes(&weight_file(RGB_CHANNELS + 2, &[]))
        .expect_err("only 3 or 4 output channels are defined");
    assert!(err.contains("out_channels"), "{err}");
}

#[test]
fn an_unparseable_gate_scale_is_an_error_not_a_default() {
    let err = Weights::from_bytes(&weight_file(RGB_CHANNELS + 1, &[("gate_scale", "lots")]))
        .expect_err("a bad gate_scale must fail");
    assert!(err.contains("gate_scale"), "{err}");
}

#[test]
fn the_scale_rule_matches_the_python_one() {
    // `trippy.hybrid.gate.effective_gate`: clamp(g * s, 0, 1), and both extremes
    // are exact (tests/test_hybrid_gate_math.py asserts the same three rows).
    assert_eq!(GateConfig::effective(0.7, 0.0), 0.0);
    assert_eq!(GateConfig::effective(0.7, 1.0), 0.7);
    assert_eq!(GateConfig::effective(0.7, 2.0), 1.0);
    assert_eq!(GateConfig::effective(0.25, 2.0), 0.5);
    assert_eq!(GateConfig::effective(1.0, 2.0), 1.0);
}

#[test]
fn the_response_lut_stays_three_rows_on_a_gate_network() {
    // The regression this pins: `camera.response` was sized `[out_channels, P]`,
    // so a four-channel network reshaped a [3, P] tensor to [1, 4*P] and panicked
    // when the viewer loaded the bundle. The tone mapper is colour-only; the gate
    // channel is never tone-mapped, so the LUT keeps three rows.
    let weights = Weights::from_bytes(&weight_file_with_camera(RGB_CHANNELS + 1, &[], Some(25)))
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(weights.unet.has_gate());
    let camera = weights.camera.expect("camera block");
    assert_eq!(camera.response_params, 25);
    let response = weights
        .get_shaped("camera.response", &[RGB_CHANNELS, 25])
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(response.len(), RGB_CHANNELS * 25);
}

#[test]
fn a_four_row_response_lut_is_rejected() {
    // The mirror image: a file whose LUT really does have four rows disagrees with
    // this reader and must say so rather than being reshaped into something.
    let mut bytes = weight_file_with_camera(RGB_CHANNELS + 1, &[], Some(25));
    // Length-preserving header edit: [3, 25] -> [4, 25] keeps the JSON the same size
    // but the data segment then no longer matches, which is exactly the corruption
    // `check_schema` exists to catch.
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + header_len]).expect("utf-8");
    // serde_json's default map is sorted, so the entry's keys come out
    // `data_offsets, dtype, shape`; match on the shape alone, which is unique in
    // this file ([3, 25] belongs to no other tensor).
    let patched = header.replacen(r#""shape":[3,25]"#, r#""shape":[4,25]"#, 1);
    assert_ne!(patched, header, "the response entry was not found in the header");
    let tail = bytes.split_off(8 + header_len);
    bytes.truncate(8);
    bytes.extend_from_slice(patched.as_bytes());
    bytes.extend_from_slice(&tail);

    // Rejected by the container itself here (the declared shape no longer matches
    // the byte range) rather than by `check_schema`; either is fine, and which one
    // fires is safetensors' business. What must never happen is a silent reshape.
    let err = Weights::from_bytes(&bytes).expect_err("a four-row LUT must fail");
    assert!(err.contains("shape"), "{err}");
}

#[test]
fn a_gateless_network_with_a_camera_still_loads() {
    let weights = Weights::from_bytes(&weight_file_with_camera(RGB_CHANNELS, &[], Some(25)))
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(!weights.unet.has_gate());
    assert!(weights.gate.is_none());
    assert_eq!(weights.camera.expect("camera").response_params, 25);
}
