//! The viewer's fly camera, and how it becomes a render camera each frame.
//!
//! Module: `trips_viewer::camera`
//! Purpose: turn mouse drags and WASD into the `(R, t)` and intrinsics
//!     [`brush_pyramid`] projects with. Deliberately small and explicit,
//!     rather than reusing Brush's `apps/brush-app/src/ui/camera_controls.rs`
//!     (400 lines built around its own `brush_render::camera::Camera`,
//!     splat-specific clamping and dataset focus points).
//! Invariants:
//!     - The frame is `docs/GEOMETRY.md`'s: the camera looks down **+Z**,
//!       **+X** is right and **+Y** is down, `x_cam = R @ x_world + t`, `R`
//!       row-major. The rows of `R` are therefore exactly `(right, down,
//!       forward)` as world vectors, which is what [`Controller::render_camera`]
//!       writes out.
//!     - The basis is re-orthonormalised every time it is rotated, so it can
//!       never drift into a skewed or mirrored frame no matter how many drags
//!       accumulate.
//!     - The controller starts **pinned** to a dataset view and reproduces its
//!       `(R, t)` bit-for-bit until the user moves. That is what makes the
//!       offscreen screenshot comparable with the reference render.
//! Units: distances are world units; angles radians; `look_speed` is radians
//!     per screen pixel; `move_speed` is world units per second.
//! Related docs: `docs/GEOMETRY.md`; `docs/USER_GUIDE.md`.

use brush_pyramid::scene::Camera;
use glam::Vec3;

use crate::bundle::BundleView;

/// Radians of rotation per pixel of mouse drag.
pub const LOOK_SPEED: f32 = 0.005;
/// How much one scroll notch scales the movement speed.
pub const SPEED_STEP: f32 = 1.15;
/// Pitch is clamped this far short of straight up/down, so the basis never
/// degenerates against the world up axis.
pub const PITCH_LIMIT: f32 = 0.01;

/// A world-to-camera frame plus the state needed to fly it.
#[derive(Debug, Clone)]
pub struct Controller {
    /// Camera centre, world units.
    pub position: Vec3,
    /// Unit vector the camera looks along (the `+Z` axis of camera space).
    forward: Vec3,
    /// Unit vector to the right of the image (`+X` of camera space).
    right: Vec3,
    /// World up. Note TRIPS scenes are usually Y-**down**, so this is often
    /// `(0, -1, 0)`; the controller only uses it as the yaw axis.
    up_world: Vec3,
    /// World units per second of held key.
    pub move_speed: f32,
    /// True until the user first moves the camera; see the module invariants.
    pinned: bool,
    /// The view being reproduced while `pinned`.
    pinned_view: BundleView,
}

impl Controller {
    /// Start pinned to `view`.
    ///
    /// # Arguments
    /// - `view`: the dataset view to open at.
    /// - `up_world`: the scene's up vector; normalised here, and replaced with
    ///   `-Y` if it is degenerate.
    /// - `move_speed`: initial fly speed, world units per second.
    #[must_use]
    pub fn new(view: &BundleView, up_world: [f32; 3], move_speed: f32) -> Self {
        let r = &view.r;
        // Rows of a world-to-camera rotation are the camera axes in world
        // coordinates.
        let right = Vec3::new(r[0], r[1], r[2]);
        let forward = Vec3::new(r[6], r[7], r[8]);
        let up = Vec3::from(up_world);
        let up_world = if up.length_squared() > 0.0 {
            up.normalize()
        } else {
            Vec3::NEG_Y
        };
        Self {
            position: view.position(),
            forward: forward.normalize_or(Vec3::Z),
            right: right.normalize_or(Vec3::X),
            up_world,
            move_speed,
            pinned: true,
            pinned_view: view.clone(),
        }
    }

    /// True while the controller is still reproducing its dataset view exactly.
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        self.pinned
    }

    /// Jump to `view` and pin to it again.
    pub fn snap_to(&mut self, view: &BundleView) {
        let speed = self.move_speed;
        let up = self.up_world.to_array();
        *self = Self::new(view, up, speed);
    }

    /// Down axis of camera space, `forward x right` — the `+Y` row of `R`.
    fn down(&self) -> Vec3 {
        self.forward.cross(self.right).normalize_or(Vec3::Y)
    }

    /// Re-orthonormalise `right` against `forward` and the world up axis.
    ///
    /// Keeping the horizon level is the whole reason `up_world` is stored:
    /// without this the camera would slowly roll as yaw and pitch compose.
    fn reorthonormalise(&mut self) {
        self.forward = self.forward.normalize_or(Vec3::Z);
        let mut right = self.forward.cross(self.up_world);
        if right.length_squared() < 1e-12 {
            // Looking straight along the up axis: keep the previous right.
            right = self.right;
        }
        self.right = right.normalize_or(Vec3::X);
    }

    /// Rotate by a mouse drag, in screen pixels.
    ///
    /// # Arguments
    /// - `dx`: rightward drag, pixels; yaws about the world up axis.
    /// - `dy`: downward drag, pixels; pitches about the camera's right axis.
    pub fn look(&mut self, dx: f32, dy: f32) {
        if dx == 0.0 && dy == 0.0 {
            return;
        }
        self.pinned = false;
        let yaw = glam::Quat::from_axis_angle(self.up_world, -dx * LOOK_SPEED);
        self.forward = yaw * self.forward;
        self.right = yaw * self.right;
        self.reorthonormalise();

        // Pitch, clamped so `forward` never lines up with `up_world`.
        let pitch_angle = -dy * LOOK_SPEED;
        let candidate = glam::Quat::from_axis_angle(self.right, pitch_angle) * self.forward;
        if candidate.normalize_or(Vec3::Z).dot(self.up_world).abs() < 1.0 - PITCH_LIMIT {
            self.forward = candidate;
            self.reorthonormalise();
        }
    }

    /// Translate along the camera basis.
    ///
    /// # Arguments
    /// - `forward`, `right`, `up`: each in `[-1, 1]`, the held-key axes.
    /// - `dt`: seconds since the last frame.
    pub fn fly(&mut self, forward: f32, right: f32, up: f32, dt: f32) {
        if forward == 0.0 && right == 0.0 && up == 0.0 {
            return;
        }
        self.pinned = false;
        let step = self.move_speed * dt;
        self.position +=
            (self.forward * forward + self.right * right + self.up_world * up) * step;
    }

    /// Scale the fly speed by `notches` scroll steps.
    pub fn adjust_speed(&mut self, notches: f32) {
        self.move_speed = (self.move_speed * SPEED_STEP.powf(notches)).clamp(1e-3, 1e4);
    }

    /// The camera to render this frame with.
    ///
    /// While pinned this returns the dataset view's own `(R, t)` and
    /// intrinsics untouched, only re-fitting the principal point and image
    /// size to `(width, height)` when the window is a different shape. Once
    /// the user has moved it builds `R` from the current basis and
    /// `t = -R c`.
    ///
    /// # Arguments
    /// - `width`, `height`: the size to render at, pixels (already scaled by
    ///   the render-scale setting).
    /// - `reference`: the dataset view supplying the focal length and lens
    ///   distortion. Focal length is scaled by `width / reference.width` so
    ///   the field of view is the view's regardless of the window size.
    #[must_use]
    pub fn render_camera(&self, width: usize, height: usize, reference: &BundleView) -> Camera {
        if self.pinned && width == reference.width && height == reference.height {
            return reference.camera();
        }
        // Same horizontal field of view as the reference view.
        let scale = width as f32 / reference.width as f32;
        let fx = reference.fx * scale;
        let fy = reference.fy * scale;

        if self.pinned {
            let mut camera = reference.camera();
            camera.width = width;
            camera.height = height;
            camera.fx = fx;
            camera.fy = fy;
            camera.cx = reference.cx * scale;
            camera.cy = reference.cy * (height as f32 / reference.height as f32);
            return camera;
        }

        let down = self.down();
        let r = [
            self.right.x,
            self.right.y,
            self.right.z,
            down.x,
            down.y,
            down.z,
            self.forward.x,
            self.forward.y,
            self.forward.z,
        ];
        // t = -R c, so that R c + t = 0 puts the camera centre at the origin.
        let t = [
            -(r[0] * self.position.x + r[1] * self.position.y + r[2] * self.position.z),
            -(r[3] * self.position.x + r[4] * self.position.y + r[5] * self.position.z),
            -(r[6] * self.position.x + r[7] * self.position.y + r[8] * self.position.z),
        ];
        Camera {
            width,
            height,
            fx,
            fy,
            cx: width as f32 / 2.0,
            cy: height as f32 / 2.0,
            r,
            t,
            distortion: reference.distortion,
        }
    }

    /// The view the controller was last pinned or snapped to.
    #[must_use]
    pub fn reference(&self) -> &BundleView {
        &self.pinned_view
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> BundleView {
        BundleView {
            index: 8,
            name: "00009.jpg".to_owned(),
            width: 1920,
            height: 1080,
            fx: 1164.46,
            fy: 1164.46,
            cx: 960.0,
            cy: 540.0,
            // A 90-degree yaw about world +Y, as a row-major world-to-camera R.
            r: [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            t: [1.0, 2.0, 3.0],
            distortion: [-0.064, 0.044, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    }

    #[test]
    fn a_pinned_controller_reproduces_its_view_exactly() {
        let v = view();
        let c = Controller::new(&v, [0.0, -1.0, 0.0], 1.0);
        assert!(c.is_pinned());
        let cam = c.render_camera(v.width, v.height, &v);
        assert_eq!(cam.r, v.r);
        assert_eq!(cam.t, v.t);
        assert_eq!(cam.distortion, v.distortion);
    }

    #[test]
    fn position_is_the_camera_centre() {
        let v = view();
        let c = Controller::new(&v, [0.0, -1.0, 0.0], 1.0);
        // R @ c + t must be the origin.
        let (r, t, p) = (v.r, v.t, c.position);
        for row in 0..3 {
            let value =
                r[row * 3] * p.x + r[row * 3 + 1] * p.y + r[row * 3 + 2] * p.z + t[row];
            assert!(value.abs() < 1e-5, "row {row} = {value}");
        }
    }

    #[test]
    fn moving_unpins_and_the_rebuilt_r_is_orthonormal() {
        let v = view();
        let mut c = Controller::new(&v, [0.0, -1.0, 0.0], 1.0);
        c.look(37.0, -11.0);
        c.fly(1.0, 0.5, -0.25, 0.016);
        assert!(!c.is_pinned());
        let cam = c.render_camera(1920, 1080, &v);
        let r = cam.r;
        for a in 0..3 {
            let row = Vec3::new(r[a * 3], r[a * 3 + 1], r[a * 3 + 2]);
            assert!((row.length() - 1.0).abs() < 1e-4, "row {a} not unit");
            for b in (a + 1)..3 {
                let other = Vec3::new(r[b * 3], r[b * 3 + 1], r[b * 3 + 2]);
                assert!(row.dot(other).abs() < 1e-4, "rows {a},{b} not orthogonal");
            }
        }
        // Right-handed: right x down = forward.
        let right = Vec3::new(r[0], r[1], r[2]);
        let down = Vec3::new(r[3], r[4], r[5]);
        let forward = Vec3::new(r[6], r[7], r[8]);
        assert!((right.cross(down) - forward).length() < 1e-4);
    }

    #[test]
    fn pitch_cannot_flip_over_the_up_axis() {
        let v = view();
        let mut c = Controller::new(&v, [0.0, -1.0, 0.0], 1.0);
        for _ in 0..1000 {
            c.look(0.0, 100.0);
        }
        assert!(c.forward.dot(c.up_world).abs() < 1.0);
        assert!((c.right.length() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn a_wider_window_keeps_the_reference_field_of_view() {
        let v = view();
        let mut c = Controller::new(&v, [0.0, -1.0, 0.0], 1.0);
        c.look(1.0, 0.0);
        let half = c.render_camera(960, 540, &v);
        assert!((half.fx - v.fx / 2.0).abs() < 1e-2);
        assert!((half.cx - 480.0).abs() < 1e-3);
    }
}
