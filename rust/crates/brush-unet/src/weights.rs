//! Reading the safetensors file `trippy.net.export_safetensors` writes.
//!
//! Module: `brush_unet::weights`
//! Purpose: the Rust half of the PyTorch -> Burn weight bridge. Turns a
//!     `.safetensors` file into (a) a [`crate::UnetConfig`] +
//!     [`crate::CameraConfig`] recovered from `__metadata__`, and (b) a
//!     name -> `(shape, Vec<f32>)` map the Burn modules bind. It deliberately
//!     stops at host `Vec<f32>`: uploading is the `gpu` half's job, so the
//!     schema itself stays testable by `scripts/test.sh` with no GPU.
//! Invariants:
//!     - Only `F32` tensors are accepted. The exporter casts everything to
//!       float32 precisely so there is no dtype ambiguity here; an `F16`
//!       file is rejected loudly rather than silently upcast.
//!     - Every key named by the config must be present with exactly the
//!       expected shape; a missing or mis-shaped tensor is an error, never a
//!       zero-filled default.
//! Units: see `trippy/net/export_safetensors.py`'s module docstring.
//! Related docs: `rust/README.md` ("brush-unet weight schema").

use std::collections::HashMap;
use std::path::Path;

use safetensors::tensor::{Dtype, SafeTensors};

use crate::config::{CameraConfig, UnetConfig, EXPORT_FORMAT};

/// One tensor, host-side.
#[derive(Debug, Clone, PartialEq)]
pub struct HostTensor {
    /// Row-major shape, exactly as written by the exporter.
    pub shape: Vec<usize>,
    /// `shape.iter().product()` float32 values, C-contiguous.
    pub data: Vec<f32>,
}

impl HostTensor {
    /// Total number of scalars.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor holds no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// A parsed weight file: metadata plus every tensor.
#[derive(Debug, Clone)]
pub struct Weights {
    /// The `__metadata__` block, verbatim.
    pub metadata: HashMap<String, String>,
    /// Network shape recovered from the metadata.
    pub unet: UnetConfig,
    /// Tone-mapper shape, or `None` when `has_camera = 0`.
    pub camera: Option<CameraConfig>,
    tensors: HashMap<String, HostTensor>,
}

fn meta_get<'a>(meta: &'a HashMap<String, String>, key: &str) -> Result<&'a str, String> {
    meta.get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("metadata key {key:?} is missing"))
}

fn meta_usize(meta: &HashMap<String, String>, key: &str) -> Result<usize, String> {
    meta_get(meta, key)?
        .parse::<usize>()
        .map_err(|e| format!("metadata key {key:?}: {e}"))
}

fn meta_flag(meta: &HashMap<String, String>, key: &str) -> Result<bool, String> {
    Ok(meta_get(meta, key)? == "1")
}

impl Weights {
    /// Parse a safetensors buffer.
    ///
    /// # Errors
    /// Returns `Err` if the container is malformed, the `format` metadata is
    /// not [`EXPORT_FORMAT`], a tensor is not `F32`, or the recovered
    /// [`UnetConfig`] fails [`UnetConfig::validate`].
    pub fn from_bytes(buffer: &[u8]) -> Result<Self, String> {
        let file = SafeTensors::deserialize(buffer).map_err(|e| format!("safetensors: {e}"))?;
        let (_, raw_meta) =
            SafeTensors::read_metadata(buffer).map_err(|e| format!("safetensors header: {e}"))?;
        let metadata: HashMap<String, String> = raw_meta.metadata().clone().unwrap_or_default();

        let format = meta_get(&metadata, "format")?;
        if format != EXPORT_FORMAT {
            return Err(format!(
                "unsupported export format {format:?} (this build reads {EXPORT_FORMAT:?})"
            ));
        }

        let unet = UnetConfig {
            in_channels: meta_usize(&metadata, "in_channels")?,
            out_channels: meta_usize(&metadata, "out_channels")?,
            filters: meta_usize(&metadata, "filters")?,
            num_layers: meta_usize(&metadata, "num_layers")?,
        };
        unet.validate()?;

        // The Burn port only implements what TRIPS actually ships; anything
        // else would silently render something different.
        for (key, supported) in [
            ("activation", "elu"),
            ("norm", "id"),
            ("upsample_mode", "bilinear"),
            ("last_act", "id"),
        ] {
            let value = meta_get(&metadata, key)?;
            if value != supported {
                return Err(format!(
                    "{key} = {value:?} is not implemented in the Burn port (only {supported:?} is)"
                ));
            }
        }

        let camera = if meta_flag(&metadata, "has_camera")? {
            Some(CameraConfig {
                num_frames: meta_usize(&metadata, "num_frames")?,
                response_params: meta_usize(&metadata, "response_params")?,
                image_height: meta_usize(&metadata, "image_height")?,
                image_width: meta_usize(&metadata, "image_width")?,
                enable_exposure: meta_flag(&metadata, "enable_exposure")?,
                enable_white_balance: meta_flag(&metadata, "enable_white_balance")?,
                enable_vignette: meta_flag(&metadata, "enable_vignette")?,
                enable_response: meta_flag(&metadata, "enable_response")?,
            })
        } else {
            None
        };

        let mut tensors = HashMap::new();
        for (name, view) in file.tensors() {
            if view.dtype() != Dtype::F32 {
                return Err(format!(
                    "{name}: dtype {:?} is not supported; the exporter writes F32 only",
                    view.dtype()
                ));
            }
            let bytes = view.data();
            let mut data = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            tensors.insert(
                name,
                HostTensor {
                    shape: view.shape().to_vec(),
                    data,
                },
            );
        }

        let weights = Self {
            metadata,
            unet,
            camera,
            tensors,
        };
        weights.check_schema()?;
        Ok(weights)
    }

    /// Read and parse a safetensors file.
    ///
    /// # Errors
    /// As [`Self::from_bytes`], plus any I/O failure.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let buffer = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(&buffer).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Every key present in the file, sorted.
    #[must_use]
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.tensors.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// Look one tensor up.
    ///
    /// # Errors
    /// Returns `Err` if the key is absent.
    pub fn get(&self, name: &str) -> Result<&HostTensor, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("tensor {name:?} is missing (have: {:?})", self.keys()))
    }

    /// Look one tensor up and check its shape.
    ///
    /// # Errors
    /// Returns `Err` if the key is absent or its shape differs.
    pub fn get_shaped(&self, name: &str, shape: &[usize]) -> Result<&HostTensor, String> {
        let tensor = self.get(name)?;
        if tensor.shape != shape {
            return Err(format!(
                "tensor {name:?} has shape {:?}, expected {shape:?}",
                tensor.shape
            ));
        }
        Ok(tensor)
    }

    /// Assert every tensor the configs call for is present, with the right
    /// shape, and that no U-Net tensor is left over.
    fn check_schema(&self) -> Result<(), String> {
        for (name, shape) in self.unet.weight_shapes() {
            self.get_shaped(&name, &shape)?;
        }
        if let Some(camera) = self.camera {
            let expected: Vec<(String, Vec<usize>)> = vec![
                ("camera.exposure".into(), vec![camera.num_frames]),
                ("camera.white_balance".into(), vec![camera.num_frames, 3]),
                ("camera.vignette_params".into(), vec![3]),
                ("camera.vignette_center".into(), vec![2]),
                (
                    "camera.response".into(),
                    vec![self.unet.out_channels, camera.response_params],
                ),
            ];
            let required = camera.weight_keys();
            for (name, shape) in expected {
                if required.contains(&name) {
                    self.get_shaped(&name, &shape)?;
                }
            }
        }
        let known: Vec<String> = self
            .unet
            .weight_keys()
            .into_iter()
            .chain(self.camera.map(|c| c.weight_keys()).unwrap_or_default())
            .collect();
        for key in self.keys() {
            if !known.contains(&key) {
                return Err(format!(
                    "unexpected tensor {key:?} in the weight file; the schema is {known:?}"
                ));
            }
        }
        Ok(())
    }
}

/// A plain safetensors file with no trippy metadata — the fixture's
/// `io.safetensors` (inputs and expected outputs).
///
/// # Errors
/// Returns `Err` on a malformed container or a non-`F32` tensor.
pub fn read_plain(path: &Path) -> Result<HashMap<String, HostTensor>, String> {
    let buffer = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let file = SafeTensors::deserialize(&buffer).map_err(|e| format!("safetensors: {e}"))?;
    let mut out = HashMap::new();
    for (name, view) in file.tensors() {
        if view.dtype() != Dtype::F32 {
            return Err(format!("{name}: only F32 is supported, got {:?}", view.dtype()));
        }
        let data = view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        out.insert(
            name,
            HostTensor {
                shape: view.shape().to_vec(),
                data,
            },
        );
    }
    Ok(out)
}
