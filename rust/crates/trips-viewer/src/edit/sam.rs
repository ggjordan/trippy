//! The SAM tool's geometry: a gesture on the render becomes a box in a photo.
//!
//! Module: `trips_viewer::edit::sam`
//! Purpose: `docs/EDITOR.md` §3's "4. SAM 3 lift (E5)", viewer half. Jordan
//!     drags a box (or Alt-clicks a point) on the render while pinned to a
//!     capture view; `trippy edits sam` wants that prompt in the PHOTOGRAPH's
//!     own pixel grid. The render is neither: it is the view's camera re-fitted
//!     to the window and multiplied by the render-scale lever
//!     (`camera::Controller::render_camera`). This module is the one place that
//!     conversion is written, and the only part of the SAM tool with no
//!     subprocess, no filesystem and no window in it — which is what makes it
//!     unit-testable (`tests` at the bottom, and `tests/test_edit_sam.py`'s
//!     `prompt_space` tests on the Python side).
//! Invariants:
//!     - The mapping is **exact for any camera that shares the reference
//!       view's `R`, `t` and `distortion`**, which is every camera the
//!       controller builds while pinned. It is the identity when the render is
//!       the view's own size at scale 1, and it is NOT a plain `width` ratio in
//!       `v`: `render_camera` scales `fy` by the WIDTH ratio while scaling `cy`
//!       by the HEIGHT ratio, so an anisotropic window moves the two apart.
//!       Going through normalised image coordinates (`(u - cx) / fx`) is the
//!       only form that survives that.
//!     - Nothing here converts to PHOTO pixels. The viewer never opens a
//!       photograph (AGENTS.md §6), so it cannot know the photo's size; it
//!       sends VIEW pixels and `trippy edits sam --prompt-space view` applies
//!       that scene's own `photo_scale`. See `trippy/edit/sam_lift.py`.
//!     - A box is normalised (`x0 <= x1`, `y0 <= y1`) and clamped to the view's
//!       raster, so a drag that started or ended off-canvas is still a legal
//!       prompt rather than a negative-width rectangle SAM would reject.
//! Units: pixels throughout — `render` for the frame the gesture was made on,
//!     `view` for `bundle.json`'s `width`/`height`. `points` are egui's
//!     logical units, which is what a pointer position arrives in.
//! Related docs: `docs/EDITOR.md` §3 "4. SAM 3 lift (E5)", §4;
//!     `docs/USER_GUIDE.md` "Editor"; `trippy/edit/sam_lift.py`.

use brush_pyramid::scene::Camera;

use crate::bundle::BundleView;

/// Smallest drag, in render pixels, that counts as a box rather than a click.
///
/// Below this a gesture is a stray wobble during a click, and a 2 px box would
/// segment nothing useful. Chosen to be larger than the few pixels a hand moves
/// while pressing a trackpad and far smaller than any object worth lifting.
pub const MIN_BOX_PX: f64 = 8.0;

/// Where a pointer position in egui points lands in the render's pixel grid.
///
/// The render target is `rect` scaled by the display's `pixels_per_point` and
/// by the render-scale lever, so a click at the top-left of the canvas is pixel
/// `(0, 0)` whatever those two are. This is the same arithmetic
/// `app.rs`'s Shift-click already does, factored out so both gestures and the
/// tests share one copy.
///
/// # Arguments
/// - `pos`: pointer position, egui points, in the window's own space.
/// - `rect_min`: the render canvas's top-left, egui points.
/// - `pixels_per_point`: the display scale factor.
/// - `render_scale`: the `-`/`=` performance lever, in `(0, 1]`.
#[must_use]
pub fn render_pixel(
    pos: (f32, f32),
    rect_min: (f32, f32),
    pixels_per_point: f32,
    render_scale: f32,
) -> (f64, f64) {
    let scale = f64::from(pixels_per_point) * f64::from(render_scale.clamp(0.1, 1.0));
    (
        f64::from(pos.0 - rect_min.0) * scale,
        f64::from(pos.1 - rect_min.1) * scale,
    )
}

/// One render pixel expressed in the reference view's own pixel grid.
///
/// Inverts the render camera's intrinsics into normalised image coordinates and
/// re-applies the view's: `xn = (u - cx_r) / fx_r`, `u_view = fx_v * xn + cx_v`.
/// Distortion cancels because both cameras carry the same coefficients (the
/// controller copies them from the reference view), so no distortion model is
/// needed here and none is applied — the same reason
/// `edit::cluster::ClickCamera` does its own projection undistorted.
///
/// # Arguments
/// - `camera`: the camera this frame was rendered with.
/// - `view`: the capture view the camera is pinned to.
/// - `px`: the pixel, in `camera`'s own coordinates.
///
/// # Returns
/// The same point in `view`'s pixels. Not clamped: a gesture outside the
/// render maps outside the view, and [`view_box_from_render`] is where that is
/// dealt with.
#[must_use]
pub fn view_pixel_from_render(camera: &Camera, view: &BundleView, px: (f64, f64)) -> (f64, f64) {
    let (fx_r, fy_r) = (f64::from(camera.fx), f64::from(camera.fy));
    let (cx_r, cy_r) = (f64::from(camera.cx), f64::from(camera.cy));
    // A degenerate focal length cannot happen on a loaded bundle, but dividing
    // by it would silently produce infinities that end up on a command line.
    if fx_r == 0.0 || fy_r == 0.0 {
        return px;
    }
    (
        (px.0 - cx_r) / fx_r * f64::from(view.fx) + f64::from(view.cx),
        (px.1 - cy_r) / fy_r * f64::from(view.fy) + f64::from(view.cy),
    )
}

/// A drag rectangle on the render as a `--box X0 Y0 X1 Y1` in view pixels.
///
/// Both corners go through [`view_pixel_from_render`], then the result is
/// normalised (`x0 <= x1`) and clamped to `[0, width] x [0, height]`.
///
/// # Arguments
/// - `camera`: the camera this frame was rendered with.
/// - `view`: the capture view the camera is pinned to.
/// - `a`, `b`: the drag's two corners, render pixels, in either order.
///
/// # Returns
/// `[x0, y0, x1, y1]` in view pixels.
#[must_use]
pub fn view_box_from_render(
    camera: &Camera,
    view: &BundleView,
    a: (f64, f64),
    b: (f64, f64),
) -> [f64; 4] {
    let p = view_pixel_from_render(camera, view, a);
    let q = view_pixel_from_render(camera, view, b);
    let (w, h) = (view.width as f64, view.height as f64);
    let clamp = |v: f64, hi: f64| v.clamp(0.0, hi);
    [
        clamp(p.0.min(q.0), w),
        clamp(p.1.min(q.1), h),
        clamp(p.0.max(q.0), w),
        clamp(p.1.max(q.1), h),
    ]
}

/// Whether a drag is big enough to be a box prompt (see [`MIN_BOX_PX`]).
#[must_use]
pub fn is_box_drag(a: (f64, f64), b: (f64, f64)) -> bool {
    (b.0 - a.0).abs() >= MIN_BOX_PX || (b.1 - a.1).abs() >= MIN_BOX_PX
}

/// Array position of the capture view whose centre is nearest `position`.
///
/// The SAM lift needs a PHOTOGRAPH, so it needs a capture view; when the camera
/// has been flown off one the tool snaps here and says so rather than guessing
/// which photo a free-flying frame corresponds to (`docs/EDITOR.md` §3).
///
/// # Arguments
/// - `views`: every capture view.
/// - `position`: the camera centre, world units.
///
/// # Returns
/// `None` only when `views` is empty, which [`crate::bundle::Bundle`] refuses
/// to load.
#[must_use]
pub fn nearest_view(views: &[BundleView], position: glam::Vec3) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (index, view) in views.iter().enumerate() {
        let d = (view.position() - position).length_squared();
        if best.is_none_or(|(_, b)| d < b) {
            best = Some((index, d));
        }
    }
    best.map(|(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A view with a deliberately off-centre principal point, so a mapping that
    /// only divided by a width ratio would be visibly wrong.
    fn a_view() -> BundleView {
        BundleView {
            index: 3,
            name: "IMG_0003.jpg".to_owned(),
            width: 1008,
            height: 756,
            fx: 757.36,
            fy: 757.36,
            cx: 494.0,
            cy: 371.0,
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            t: [0.0, 0.0, 0.0],
            distortion: [0.0; 8],
        }
    }

    /// The projection `brush_pyramid` performs, written here from scratch, so
    /// the round-trip below is not checked against the code it is testing.
    fn project(camera: &Camera, p: [f64; 3]) -> (f64, f64) {
        let r = camera.r.map(f64::from);
        let t = camera.t.map(f64::from);
        let x = r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0];
        let y = r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1];
        let z = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2];
        (
            f64::from(camera.fx) * x / z + f64::from(camera.cx),
            f64::from(camera.fy) * y / z + f64::from(camera.cy),
        )
    }

    #[test]
    fn a_render_at_the_views_own_size_maps_pixel_for_pixel() {
        let view = a_view();
        let camera = view.camera();
        for px in [(0.0, 0.0), (500.0, 300.0), (1007.0, 755.0)] {
            let mapped = view_pixel_from_render(&camera, &view, px);
            assert!((mapped.0 - px.0).abs() < 1e-9, "{mapped:?} vs {px:?}");
            assert!((mapped.1 - px.1).abs() < 1e-9, "{mapped:?} vs {px:?}");
        }
    }

    #[test]
    fn a_half_scale_render_maps_back_onto_the_full_view() {
        let view = a_view();
        let controller = crate::camera::Controller::new(&[view.clone()], 0, [0.0, -1.0, 0.0]);
        let camera = controller.render_camera(504, 378, &view);
        let mapped = view_pixel_from_render(&camera, &view, (247.0, 185.5));
        // 247 render px is 2 * 247 = 494 view px, which is exactly `cx`.
        assert!((mapped.0 - 494.0).abs() < 1e-6, "{mapped:?}");
        assert!((mapped.1 - 371.0).abs() < 1e-6, "{mapped:?}");
    }

    #[test]
    fn the_mapping_is_the_inverse_of_the_render_cameras_own_projection() {
        // The load-bearing case: a WINDOW-shaped render, where the aspect no
        // longer matches the capture and `fy`/`cy` scale by different ratios.
        // A world point must land at the same place in both cameras' pixels.
        let view = a_view();
        let controller = crate::camera::Controller::new(&[view.clone()], 0, [0.0, -1.0, 0.0]);
        let render = controller.render_camera(1600, 900, &view);
        let view_camera = view.camera();
        for p in [
            [0.10, 0.05, 3.0],
            [-0.40, 0.22, 7.5],
            [0.90, -0.60, 2.25],
            [0.0, 0.0, 1.0],
        ] {
            let render_px = project(&render, p);
            let wanted = project(&view_camera, p);
            let got = view_pixel_from_render(&render, &view, render_px);
            assert!(
                (got.0 - wanted.0).abs() < 1e-6 && (got.1 - wanted.1).abs() < 1e-6,
                "point {p:?}: render {render_px:?} mapped to {got:?}, want {wanted:?}"
            );
        }
    }

    #[test]
    fn a_window_shaped_render_is_not_a_plain_height_ratio() {
        // Guards the invariant in the module header: if someone "simplifies"
        // the mapping to `v * view.height / render.height`, this fails.
        let view = a_view();
        let controller = crate::camera::Controller::new(&[view.clone()], 0, [0.0, -1.0, 0.0]);
        let render = controller.render_camera(1600, 900, &view);
        let got = view_pixel_from_render(&render, &view, (800.0, 450.0));
        let naive = 450.0 * 756.0 / 900.0;
        assert!(
            (got.1 - naive).abs() > 1.0,
            "the naive ratio would have been {naive}, the exact answer is {}",
            got.1
        );
        // And the exact answer really is the view's own principal point, since
        // (800, 450) is the render's centre only in x; in y `cy` moved.
        let expected = (450.0 - f64::from(render.cy)) / f64::from(render.fy)
            * f64::from(view.fy)
            + f64::from(view.cy);
        assert!((got.1 - expected).abs() < 1e-9);
    }

    #[test]
    fn a_drag_box_is_normalised_and_clamped_to_the_view() {
        let view = a_view();
        let camera = view.camera();
        // Dragged bottom-right to top-left, starting off the left edge.
        let b = view_box_from_render(&camera, &view, (900.0, 700.0), (-50.0, -30.0));
        assert_eq!(b, [0.0, 0.0, 900.0, 700.0]);
        // And off the far edge the other way.
        let c = view_box_from_render(&camera, &view, (100.0, 100.0), (5000.0, 5000.0));
        assert_eq!(c, [100.0, 100.0, 1008.0, 756.0]);
    }

    #[test]
    fn a_tiny_drag_is_a_click_not_a_box() {
        assert!(!is_box_drag((10.0, 10.0), (12.0, 13.0)));
        assert!(is_box_drag((10.0, 10.0), (10.0, 40.0)));
        assert!(is_box_drag((10.0, 10.0), (-40.0, 10.0)));
    }

    #[test]
    fn render_pixel_accounts_for_the_display_scale_and_the_render_lever() {
        assert_eq!(render_pixel((12.0, 34.0), (12.0, 34.0), 2.0, 1.0), (0.0, 0.0));
        assert_eq!(render_pixel((112.0, 84.0), (12.0, 34.0), 2.0, 1.0), (200.0, 100.0));
        assert_eq!(render_pixel((112.0, 84.0), (12.0, 34.0), 2.0, 0.5), (100.0, 50.0));
    }

    #[test]
    fn the_nearest_view_is_the_nearest_camera_centre() {
        let mut a = a_view();
        a.t = [0.0, 0.0, 0.0]; // centre at the origin
        let mut b = a_view();
        b.index = 4;
        b.t = [-5.0, 0.0, 0.0]; // centre at (5, 0, 0) since R = I and c = -R^T t
        let views = [a, b];
        assert_eq!(nearest_view(&views, glam::Vec3::new(1.0, 0.0, 0.0)), Some(0));
        assert_eq!(nearest_view(&views, glam::Vec3::new(4.0, 0.0, 0.0)), Some(1));
        assert_eq!(nearest_view(&[], glam::Vec3::ZERO), None);
    }
}
