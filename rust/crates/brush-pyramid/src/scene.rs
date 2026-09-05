//! Point sets and cameras: what the rasteriser draws, and from where.
//!
//! Module: `brush_pyramid::scene`
//! Purpose: host-side containers plus loaders for the two files a render
//!     needs — a numpy `.npz` point set and a JSON camera. Both formats are
//!     written by trippy's Python side, so this module is the Rust half of
//!     that contract.
//! Invariants:
//!     - Frames follow `docs/GEOMETRY.md`: `xyz` is COLMAP world frame,
//!       `(R, t)` is world-to-camera (`x_cam = R @ x_world + t`), the camera
//!       looks down `+Z`, `+X` right and `+Y` down, and depth is positive in
//!       front. `R` is stored **row-major**.
//!     - Every per-point array is row-aligned by point index and float32.
//!     - `feat` is `(N, C)` row-major with `C = num_channels`; the loader
//!       accepts trippy's `PointSet.save_npz` key names (`size0`, `rgb0`,
//!       `conf0`) as well as the fixture's (`size`, `feat`, `conf`), so both
//!       an exported point set and a test fixture load through one path.
//! Units: positions and `size` are world units; `conf` is dimensionless in
//!     `(0, 1)`; intrinsics are layer-0 pixels.
//! Related docs: `docs/GEOMETRY.md`; `trippy/points/source.py` (`PointSet`).

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::npz::{read_npz, NpyArray};

/// A splattable point set, flattened for upload.
#[derive(Debug, Clone, PartialEq)]
pub struct PointSet {
    /// `N * 3` world-frame positions, row-major `(N, 3)`.
    pub xyz: Vec<f32>,
    /// `N` effective (post-softplus) radii, world units, positive.
    pub size: Vec<f32>,
    /// `N * C` features, row-major `(N, C)`.
    pub feat: Vec<f32>,
    /// `N` effective (post-sigmoid) confidences in `(0, 1)`.
    pub conf: Vec<f32>,
    /// `C`, the feature width.
    pub num_channels: usize,
}

impl PointSet {
    /// `N`, the number of points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.size.len()
    }

    /// True when the set holds no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size.is_empty()
    }

    /// Build from already-flattened arrays, validating the shape relations.
    ///
    /// # Errors
    /// Returns `Err` if the arrays' lengths are not consistent with a single
    /// `N` and `C`, or if `C` is zero.
    pub fn new(
        xyz: Vec<f32>,
        size: Vec<f32>,
        feat: Vec<f32>,
        conf: Vec<f32>,
        num_channels: usize,
    ) -> Result<Self, String> {
        let n = size.len();
        if num_channels == 0 {
            return Err("num_channels must be >= 1".to_owned());
        }
        if xyz.len() != n * 3 {
            return Err(format!("xyz has {} values, expected {}", xyz.len(), n * 3));
        }
        if conf.len() != n {
            return Err(format!("conf has {} values, expected {n}", conf.len()));
        }
        if feat.len() != n * num_channels {
            return Err(format!(
                "feat has {} values, expected {}",
                feat.len(),
                n * num_channels
            ));
        }
        Ok(Self {
            xyz,
            size,
            feat,
            conf,
            num_channels,
        })
    }

    /// Load from a numpy `.npz`.
    ///
    /// Accepts either naming convention:
    /// - the fixture's, written by `tools/dump_raster_fixture.py`: `xyz`,
    ///   `size`, `feat`, `conf`;
    /// - trippy's own `PointSet.save_npz`: `xyz`, `size0`, `rgb0`, `conf0`.
    ///
    /// # Errors
    /// Returns `Err` if the archive cannot be read, a required array is
    /// missing, or the shapes disagree.
    pub fn from_npz(path: &Path) -> Result<Self, String> {
        let arrays = read_npz(path)?;
        let pick = |names: &[&str]| -> Result<&NpyArray, String> {
            names
                .iter()
                .find_map(|n| arrays.get(*n))
                .ok_or_else(|| format!("{}: none of {names:?} present", path.display()))
        };

        let xyz_arr = pick(&["xyz"])?;
        if xyz_arr.shape.len() != 2 || xyz_arr.shape[1] != 3 {
            return Err(format!("xyz must be (N, 3), got {:?}", xyz_arr.shape));
        }
        let n = xyz_arr.shape[0];

        let size_arr = pick(&["size", "size0"])?;
        size_arr.expect_shape("size", &[n])?;
        let conf_arr = pick(&["conf", "conf0"])?;
        conf_arr.expect_shape("conf", &[n])?;

        let feat_arr = pick(&["feat", "rgb0", "rgb"])?;
        if feat_arr.shape.len() != 2 || feat_arr.shape[0] != n {
            return Err(format!("feat must be (N, C), got {:?}", feat_arr.shape));
        }
        let num_channels = feat_arr.shape[1];

        Self::new(
            xyz_arr.to_f32()?,
            size_arr.to_f32()?,
            feat_arr.to_f32()?,
            conf_arr.to_f32()?,
            num_channels,
        )
    }
}

/// A pinhole camera: layer-0 intrinsics plus a world-to-camera pose.
///
/// This is the on-disk JSON shape too — `tools/dump_raster_fixture.py` writes
/// exactly these fields.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    /// Layer-0 image width in pixels.
    pub width: usize,
    /// Layer-0 image height in pixels.
    pub height: usize,
    /// Focal length in pixels along `x`. **The only focal the point-size
    /// projection uses** (`RenderForward.cu:1489`).
    pub fx: f32,
    /// Focal length in pixels along `y`.
    pub fy: f32,
    /// Principal point `x`, pixels.
    pub cx: f32,
    /// Principal point `y`, pixels.
    pub cy: f32,
    /// World-to-camera rotation, **row-major** 3x3.
    #[serde(rename = "R")]
    pub r: [f32; 9],
    /// World-to-camera translation, world units.
    #[serde(rename = "t")]
    pub t: [f32; 3],
}

impl Camera {
    /// Load from the JSON written by `tools/dump_raster_fixture.py`.
    ///
    /// # Errors
    /// Returns `Err` on I/O failure or a malformed/incomplete document.
    pub fn from_json(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Transform one world point into the camera frame: `x_cam = R @ x + t`.
    #[must_use]
    pub fn world_to_cam(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let r = &self.r;
        (
            r[0] * x + r[1] * y + r[2] * z + self.t[0],
            r[3] * x + r[4] * y + r[5] * z + self.t[1],
            r[6] * x + r[7] * y + r[8] * z + self.t[2],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_camera() -> Camera {
        Camera {
            width: 64,
            height: 48,
            fx: 60.0,
            fy: 60.0,
            cx: 32.0,
            cy: 24.0,
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            t: [0.0, 0.0, 0.0],
        }
    }

    #[test]
    fn world_to_cam_is_row_major_r_then_plus_t() {
        // A 90-degree rotation about +Z, row-major: (x, y, z) -> (-y, x, z).
        let cam = Camera {
            r: [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            t: [1.0, 2.0, 3.0],
            ..identity_camera()
        };
        let (x, y, z) = cam.world_to_cam(1.0, 0.0, 0.0);
        assert!((x - 1.0).abs() < 1e-6, "{x}");
        assert!((y - 3.0).abs() < 1e-6, "{y}");
        assert!((z - 3.0).abs() < 1e-6, "{z}");
    }

    #[test]
    fn point_set_new_rejects_inconsistent_lengths() {
        assert!(PointSet::new(vec![0.0; 6], vec![1.0; 2], vec![0.0; 8], vec![0.5; 2], 4).is_ok());
        // xyz too short for N = 2.
        assert!(PointSet::new(vec![0.0; 3], vec![1.0; 2], vec![0.0; 8], vec![0.5; 2], 4).is_err());
        // feat not a multiple of C.
        assert!(PointSet::new(vec![0.0; 6], vec![1.0; 2], vec![0.0; 7], vec![0.5; 2], 4).is_err());
        // conf length mismatch.
        assert!(PointSet::new(vec![0.0; 6], vec![1.0; 2], vec![0.0; 8], vec![0.5; 3], 4).is_err());
        assert!(PointSet::new(vec![], vec![], vec![], vec![], 0).is_err());
    }

    #[test]
    fn camera_json_round_trips_through_the_python_field_names() {
        let cam = identity_camera();
        let json = serde_json::to_string(&cam).expect("serialise");
        assert!(json.contains("\"R\""), "{json}");
        assert!(json.contains("\"t\""), "{json}");
        let back: Camera = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(cam, back);
    }
}
