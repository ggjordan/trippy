//! The LIVE Gaussian splat: a `.ply` on the device, rendered at any pose.
//!
//! Module: `trips_viewer::splat`
//! Purpose: the Blend panel's splat half, for real. Until this existed the
//!     viewer could only show the *precomputed* renders `bundle.json`'s
//!     `blend.splat_renders` carries — one per capture view, valid only while
//!     the camera sits exactly on that view — so `splat` / `gated` / `mix` /
//!     `split` went blank the moment you flew anywhere. This module loads
//!     `blend.splat_ply` once into Brush's own `Splats` (means, log-scales,
//!     quats, SH coefficients, raw opacity, all device-resident) and renders it
//!     with `brush_render::render_splats` at the viewer's own camera and
//!     resolution, every frame.
//! Invariants:
//!     - **One device.** The `Splats` tensors are built on the same
//!       `burn::tensor::Device` the pyramid, the U-Net and (in the window)
//!       eframe already share, so the splat image is a Burn tensor on the same
//!       allocator as the TRIPS frame and the two compose with ordinary tensor
//!       arithmetic — no readback, no second device, no copy.
//!     - **Loaded once.** `kklid_20000.ply` is 2.1 GB / 8.9 M Gaussians;
//!       [`LiveSplat::load`] is called at bundle-open time, reports how long it
//!       took, and the result is kept for the life of the process. Nothing here
//!       is called per frame except [`LiveSplat::render`].
//!     - **Display space.** A 3DGS `f_dc_*` coefficient is fitted against the
//!       capture's *display-referred* pixels, so `render_splats`' output is
//!       already in display space, and so is the TRIPS frame by the time
//!       `Renderer::compose` sees it (`NeuralCamera` has applied exposure, white
//!       balance, vignette and the response LUT). The blend is therefore done in
//!       display space, with **no** transfer function applied to either side.
//!       See `docs/USER_GUIDE.md` "Blend panel".
//!     - **Premultiplied over black.** The render uses `background = 0`, so
//!       `out_img`'s RGB is already `alpha * colour` and a pixel no Gaussian
//!       covers is exactly 0 — which is the same convention the precomputed
//!       path's `SplatImage::masked_rgb(mask_by_alpha = true)` produces, so the
//!       live and precomputed operands are interchangeable.
//!     - **Pinhole only.** `brush_render::camera::Camera` carries its own
//!       distortion models (KB4, RadTan8, thin prism); a TRIPS bundle carries
//!       Saiga's 8-parameter set, which is a different parameterisation. The
//!       conversion therefore renders the splat through a plain pinhole and the
//!       TRIPS half through its own distortion, so on a scene with strong
//!       distortion the two halves disagree by up to a few pixels at the
//!       corners. Recorded in `docs/LIMITATIONS.md` rather than silently
//!       approximated.
//! Units: `fov_x`/`fov_y` are radians; `center_uv` is a fraction of the image;
//!     positions are world units in the bundle's own COLMAP frame.
//! Related docs: `docs/decisions/ADR-0006-viewer-integration.md`;
//!     `docs/EDITOR.md` §2; `rust/README.md` "`trips-viewer`".

use std::path::{Path, PathBuf};

use brush_pyramid::scene::Camera as TripsCamera;
use brush_render::camera::{focal_to_fov, Camera as BrushCamera};
use brush_render::gaussian_splats::{SplatRenderMode, Splats, TextureMode};
use burn::tensor::Tensor;

/// Convert trippy's COLMAP camera into Brush's.
///
/// # The two conventions, and why this is only six lines
///
/// They agree on the hard part and differ only in *which direction* the pose is
/// stored:
///
/// | | trippy (`brush_pyramid::scene::Camera`, `docs/GEOMETRY.md`) | Brush (`brush_render::camera::Camera`) |
/// |---|---|---|
/// | camera axes | `+X` right, `+Y` down, `+Z` forward | `+X` right, `+Y` down, `+Z` forward |
/// | pose stored | **world-to-camera** `x_cam = R x_world + t`, `R` row-major | **camera-to-world**: `position` is the camera centre, `rotation` maps camera into world |
/// | intrinsics | `fx`, `fy`, `cx`, `cy` in layer-0 pixels | `fov_x`, `fov_y` in radians, `center_uv` as a fraction |
///
/// Same handedness, same axis directions, same "depth positive in front" — the
/// axis flip that `brush-dataset`'s `opengl_c2w_to_pose` performs for
/// nerfstudio scenes is **not** needed here, because a COLMAP pose is already in
/// Brush's frame (`brush-dataset/src/formats/colmap.rs` converts one with
/// nothing but an `.inverse()`).
///
/// So the whole conversion is the inverse of the pose plus a focal-to-fov:
///
/// - `rotation = R^T`. `R` is stored **row-major** and `glam::Mat3::from_cols_array`
///   reads **column-major**, so feeding `r` straight to it *is* the transpose —
///   the one place this could go wrong is also the place where doing nothing is
///   correct, which is why it gets its own assertion in the tests below.
/// - `position = -R^T t`, the camera centre in world coordinates (identical to
///   [`crate::bundle::BundleView::position`], which derives it independently).
/// - `fov = focal_to_fov(focal, pixels)`, Brush's own function, so
///   `camera.focal(img_size)` gives back exactly `fx`/`fy`.
/// - `center_uv = (cx / width, cy / height)`, so `camera.center(img_size)` gives
///   back exactly `cx`/`cy`.
///
/// # Arguments
/// - `camera`: the frame's TRIPS camera, already at the render resolution.
///
/// # Returns
/// A pinhole `brush_render::camera::Camera` that projects every world point to
/// the same pixel (see `tests::a_known_point_projects_identically_in_both`).
#[must_use]
pub fn to_brush_camera(camera: &TripsCamera) -> BrushCamera {
    // `from_cols_array` reads column-major; `camera.r` is row-major. Reading a
    // row-major matrix column-major yields its transpose, and R^T is exactly the
    // camera-to-world rotation Brush wants.
    let cam_to_world = glam::Mat3::from_cols_array(&camera.r);
    let rotation = glam::Quat::from_mat3(&cam_to_world).normalize();
    let position = cam_to_world * -glam::Vec3::from(camera.t);
    let model = brush_render::kernels::camera_model::CameraModel::Pinhole;
    #[allow(clippy::cast_possible_truncation)]
    let width = camera.width as u32;
    #[allow(clippy::cast_possible_truncation)]
    let height = camera.height as u32;
    BrushCamera::new(
        position,
        rotation,
        focal_to_fov(f64::from(camera.fx), width, &model),
        focal_to_fov(f64::from(camera.fy), height, &model),
        glam::vec2(
            camera.cx / camera.width as f32,
            camera.cy / camera.height as f32,
        ),
        model,
    )
}

/// A `.ply` streamed off disk into `tokio::io::AsyncRead`, with no copy of the
/// whole file in memory.
///
/// `brush_serde::load_splat_from_ply` wants an `AsyncRead`. The obvious
/// `std::io::Cursor<Vec<u8>>` satisfies it, but on `kklid_20000.ply` that is a
/// 2.1 GB buffer held for the whole parse **on top of** the ~2.2 GB of parsed
/// `Vec<f32>`s. This adapter does blocking reads on the calling thread instead:
/// the loader is already being driven by a blocking executor
/// (`brush_pyramid::gpu::block_on`), so there is no runtime to starve.
struct BlockingPly(std::fs::File);

impl tokio::io::AsyncRead for BlockingPly {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        use std::io::Read;
        let me = self.get_mut();
        // `initialize_unfilled` is the safe half of `ReadBuf`; the zeroing it
        // costs is a memset over an 8 MB chunk against a disk read of the same
        // size, i.e. noise.
        let dst = buf.initialize_unfilled();
        match me.0.read(dst) {
            Ok(n) => {
                buf.advance(n);
                std::task::Poll::Ready(Ok(()))
            }
            Err(e) => std::task::Poll::Ready(Err(e)),
        }
    }
}

/// A Gaussian splat loaded once and kept on the device.
pub struct LiveSplat {
    /// Brush's own splat parameters, device-resident: `transforms` `[N, 10]`
    /// (means, quats, log-scales), `sh_coeffs` `[N, C, 3]`, `raw_opacities`
    /// `[N]`.
    splats: Splats,
    /// Where it came from, for the HUD and for error messages.
    path: PathBuf,
    /// `N`.
    num_splats: usize,
    /// SH degree the file carried.
    sh_degree: u32,
    /// Wall clock of [`Self::load`], milliseconds. Reported once, not per frame.
    load_ms: f64,
}

impl LiveSplat {
    /// Read `path` and upload it, once.
    ///
    /// # Arguments
    /// - `path`: an absolute `.ply` path, normally `bundle.json`'s
    ///   `blend.splat_ply`.
    /// - `device`: the device Burn, the pyramid and (in the window) eframe share.
    /// - `subsample`: keep every `n`-th Gaussian. `None` = all of them. This is
    ///   a **stride applied while parsing**, so it also shrinks the parser's
    ///   up-front allocation — the only lever that helps peak memory on a 2 GB
    ///   file.
    ///
    /// # Errors
    /// Returns `Err` when the file cannot be opened or is not a Gaussian `.ply`
    /// Brush can read. A failure is never fatal to the viewer: the caller falls
    /// back to the precomputed renders and says so.
    ///
    /// # Panics
    /// Does not panic; the parse is driven by `block_on` on the calling thread.
    pub fn load(
        path: &Path,
        device: &burn::tensor::Device,
        subsample: Option<u32>,
    ) -> Result<Self, String> {
        let start = std::time::Instant::now();
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let message = brush_pyramid::gpu::block_on(brush_serde::load_splat_from_ply(
            BlockingPly(file),
            subsample,
        ))
        .map_err(|e| format!("{}: {e}", path.display()))?;
        let mode = message.meta.render_mode.unwrap_or(SplatRenderMode::Default);
        let splats = message.data.into_splats(device, mode);
        let num_splats = splats.num_splats() as usize;
        let sh_degree = splats.sh_degree();
        Ok(Self {
            splats,
            path: path.to_path_buf(),
            num_splats,
            sh_degree,
            load_ms: start.elapsed().as_secs_f64() * 1e3,
        })
    }

    /// `N`, the number of Gaussians.
    #[must_use]
    pub const fn num_splats(&self) -> usize {
        self.num_splats
    }

    /// The SH degree the file carried (0 for a DC-only export).
    #[must_use]
    pub const fn sh_degree(&self) -> u32 {
        self.sh_degree
    }

    /// How long [`Self::load`] took, milliseconds.
    #[must_use]
    pub const fn load_ms(&self) -> f64 {
        self.load_ms
    }

    /// Where it was loaded from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Render the splat at `camera`, as `[1, 3, H, W]` planar RGB.
    ///
    /// The shape and layout are chosen to be **exactly** what
    /// `Renderer::compose` already receives from the tone mapper, so the live
    /// splat drops into the existing blend arithmetic (and therefore into
    /// `blit.wgsl`'s existing `MODE_NETWORK` planar path) with no new shader
    /// branch and no new compositing code.
    ///
    /// # Arguments
    /// - `camera`: the frame's TRIPS camera; its `width`/`height` are the render
    ///   size, so this follows the viewer's window size and `--scale` for free.
    ///
    /// # Errors
    /// Returns `Err` if the camera is degenerate (zero-sized, or non-finite
    /// after conversion), which `brush_render` would otherwise assert on.
    pub async fn render(&self, camera: &TripsCamera) -> Result<Tensor<4>, String> {
        self.render_counting(camera).await.map(|(image, _)| image)
    }

    /// [`Self::render`], and how much of the splat the rasteriser actually
    /// touched.
    ///
    /// Only `--splat-bench` wants the second half, and it wants it badly: a
    /// millisecond figure for a camera that happens to be looking away from the
    /// Gaussians (a ply from a *different* COLMAP reconstruction than the
    /// bundle's, say) is a measurement of the frustum cull, not of the
    /// rasteriser. `visible == 0` is the honest way to notice.
    ///
    /// # Returns
    /// `(image, (visible splats, tile intersections))`.
    ///
    /// # Errors
    /// As [`Self::render`].
    pub async fn render_counting(
        &self,
        camera: &TripsCamera,
    ) -> Result<(Tensor<4>, (u32, u32)), String> {
        let (height, width) = (camera.height, camera.width);
        if height == 0 || width == 0 {
            return Err(format!("splat render at {width}x{height}: empty image"));
        }
        let brush_camera = to_brush_camera(camera);
        if !brush_camera.is_valid() {
            return Err(format!(
                "splat render: the converted camera is not finite (fx {}, fy {}, t {:?})",
                camera.fx, camera.fy, camera.t
            ));
        }
        #[allow(clippy::cast_possible_truncation)]
        let img_size = glam::uvec2(width as u32, height as u32);
        // `TextureMode::Float` -- a real `[H, W, 4]` f32 image, not the packed
        // RGBA8 `TextureMode::Packed` that `apps/brush-app` displays. Packed
        // would need a second branch in `blit.wgsl` AND would still have to be
        // unpacked before it could be blended with the TRIPS frame, at 8 bits
        // per channel; Float composites directly and at full precision.
        //
        // `background = 0`: RGB comes back premultiplied by coverage, matching
        // the precomputed path's alpha-masked renders. See the module invariants.
        let (image, aux) = brush_render::render_splats(
            self.splats.clone(),
            &brush_camera,
            img_size,
            glam::Vec3::ZERO,
            None,
            TextureMode::Float,
        )
        .await;
        let dims = image.dims();
        if dims[0] != height || dims[1] != width || dims[2] < 3 {
            return Err(format!(
                "splat render returned {dims:?}, expected [{height}, {width}, >= 3]"
            ));
        }
        // `[H, W, 4]` -> `[1, 3, H, W]`. `reshape` after `permute` materialises a
        // contiguous buffer, which `Renderer::render` asserts on before handing
        // it to the blit; a merely re-described (strided) tensor would index
        // wrong in the shader.
        Ok((
            image
                .slice([0..height, 0..width, 0..3])
                .permute([2, 0, 1])
                .reshape([1, 3, height, width]),
            (aux.num_visible, aux.num_intersections),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A camera that is not axis-aligned, not centred and not square, so a
    /// transposed rotation or a swapped `fx`/`fy` cannot pass by symmetry.
    fn awkward_camera() -> TripsCamera {
        // Yaw 0.4 rad, pitch -0.25 rad, roll 0.1 rad as a world-to-camera
        // rotation, built with glam and written out ROW-major (which is what
        // `TripsCamera::r` stores).
        let w2c = glam::Mat3::from_euler(glam::EulerRot::YXZ, 0.4, -0.25, 0.1);
        let mut r = [0.0f32; 9];
        for row in 0..3 {
            for col in 0..3 {
                r[row * 3 + col] = w2c.col(col)[row];
            }
        }
        TripsCamera {
            width: 1920,
            height: 1080,
            fx: 1234.5,
            fy: 1180.25,
            cx: 951.5,
            cy: 531.25,
            r,
            t: [0.35, -1.25, 4.5],
            distortion: [0.0; 8],
        }
    }

    /// Project a world point the way trippy does: `u = K (R x + t)`.
    fn project_trips(camera: &TripsCamera, point: glam::Vec3) -> (glam::Vec2, f32) {
        let r = &camera.r;
        let x = r[0] * point.x + r[1] * point.y + r[2] * point.z + camera.t[0];
        let y = r[3] * point.x + r[4] * point.y + r[5] * point.z + camera.t[1];
        let z = r[6] * point.x + r[7] * point.y + r[8] * point.z + camera.t[2];
        (
            glam::vec2(camera.fx * x / z + camera.cx, camera.fy * y / z + camera.cy),
            z,
        )
    }

    /// Project a world point the way `brush-render`'s kernel does: transform by
    /// `camera.world_to_local()`, then apply the pinhole parameters
    /// `camera.build_pinhole_params(img_size)` hands out. Both come from Brush's
    /// own code, so this is not a re-derivation of the formula under test.
    fn project_brush(camera: &BrushCamera, img_size: glam::UVec2, point: glam::Vec3) -> (glam::Vec2, f32) {
        let local = camera.world_to_local().transform_point3(point);
        let p = camera.build_pinhole_params(img_size);
        (
            glam::vec2(p.fx * local.x / local.z + p.cx, p.fy * local.y / local.z + p.cy),
            local.z,
        )
    }

    #[test]
    fn a_known_point_projects_identically_in_both() {
        let trips = awkward_camera();
        let brush = to_brush_camera(&trips);
        #[allow(clippy::cast_possible_truncation)]
        let img_size = glam::uvec2(trips.width as u32, trips.height as u32);
        // A spread of points in front of the camera, including one far off-axis
        // so an `fx`/`fy` swap or a principal-point error cannot cancel.
        for point in [
            glam::vec3(0.0, 0.0, 0.0),
            glam::vec3(1.0, -0.5, 2.0),
            glam::vec3(-3.0, 2.5, -1.0),
            glam::vec3(0.25, 0.25, 0.25),
            glam::vec3(-8.0, 6.0, 11.0),
        ] {
            let (uv_trips, z_trips) = project_trips(&trips, point);
            let (uv_brush, z_brush) = project_brush(&brush, img_size, point);
            assert!(
                z_trips > 0.0,
                "the fixture must place {point:?} in front of the camera"
            );
            assert!(
                (z_trips - z_brush).abs() < 1e-3,
                "depth disagrees at {point:?}: trips {z_trips} vs brush {z_brush}"
            );
            let delta = (uv_trips - uv_brush).length();
            assert!(
                delta < 1e-2,
                "projection disagrees at {point:?} by {delta} px: \
                 trips {uv_trips:?} vs brush {uv_brush:?}"
            );
        }
    }

    #[test]
    fn the_camera_centre_matches_the_bundles_own_derivation() {
        let trips = awkward_camera();
        let brush = to_brush_camera(&trips);
        // `-R^T t`, spelled out here rather than reusing the helper, so the two
        // derivations stay independent.
        let r = &trips.r;
        let t = &trips.t;
        let expected = glam::vec3(
            -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
            -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
            -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]),
        );
        assert!(
            (brush.position - expected).length() < 1e-4,
            "camera centre {:?} != {expected:?}",
            brush.position
        );
    }

    #[test]
    fn the_intrinsics_round_trip_through_fov_and_centre_uv() {
        let trips = awkward_camera();
        let brush = to_brush_camera(&trips);
        #[allow(clippy::cast_possible_truncation)]
        let img_size = glam::uvec2(trips.width as u32, trips.height as u32);
        let focal = brush.focal(img_size);
        let centre = brush.center(img_size);
        assert!((focal.x - trips.fx).abs() < 1e-2, "fx {} != {}", focal.x, trips.fx);
        assert!((focal.y - trips.fy).abs() < 1e-2, "fy {} != {}", focal.y, trips.fy);
        assert!((centre.x - trips.cx).abs() < 1e-3, "cx {} != {}", centre.x, trips.cx);
        assert!((centre.y - trips.cy).abs() < 1e-3, "cy {} != {}", centre.y, trips.cy);
    }

    #[test]
    fn the_rotation_is_the_transpose_not_the_matrix() {
        // The bug this whole conversion is one `transpose()` away from: reading
        // the row-major world-to-camera R as if it were camera-to-world. On the
        // deliberately non-symmetric fixture the two differ, so a future edit
        // that "fixes" the transpose fails here rather than in a window.
        let trips = awkward_camera();
        let brush = to_brush_camera(&trips);
        let as_written = glam::Mat3::from_cols_array(&trips.r);
        let wrong = glam::Quat::from_mat3(&as_written.transpose()).normalize();
        assert!(
            brush.rotation.angle_between(wrong) > 0.1,
            "the fixture is too symmetric to catch a transposed rotation"
        );
        // And the right one really does map camera +Z onto the world-space
        // viewing direction, which is R^T's third ROW... i.e. R's third row read
        // as a world vector.
        let forward = brush.rotation * glam::Vec3::Z;
        let expected = glam::vec3(trips.r[6], trips.r[7], trips.r[8]);
        assert!(
            (forward - expected).length() < 1e-4,
            "camera +Z maps to {forward:?}, expected the third row of R {expected:?}"
        );
    }

    #[test]
    fn a_pinhole_camera_is_valid_after_conversion() {
        assert!(to_brush_camera(&awkward_camera()).is_valid());
    }
}
