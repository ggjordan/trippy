//! The viewer's camera controller: orbit and fly, and how either becomes a
//! render camera each frame.
//!
//! Module: `trips_viewer::camera`
//! Purpose: turn mouse drags and WASD into the `(R, t)` and intrinsics
//!     [`brush_pyramid`] projects with. Deliberately small and explicit,
//!     rather than reusing Brush's `apps/brush-app/src/ui/camera_controls.rs`
//!     (400 lines built around its own `brush_render::camera::Camera`,
//!     splat-specific clamping and dataset focus points) — but the *input
//!     model* is deliberately Brush's, because that is the one every splat
//!     viewer uses: left-drag orbits or looks, right/middle-drag pans, scroll
//!     zooms or changes fly speed.
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
//!     - Every distance the controller invents comes from
//!       [`crate::bundle::SceneScale`], i.e. from where the capture cameras
//!       are — never from the point cloud, which includes a far-field
//!       environment sphere thousands of units across. Angles do not: look
//!       sensitivity is radians per pixel and is the same in every scene.
//!     - In [`Mode::Orbit`] the pivot is clamped into the camera box and the
//!       camera is only ever moved *with* the pivot, so orbiting and panning
//!       cannot leave the scene. [`Mode::Free`] has no such clamp and reports
//!       [`Controller::is_lost`] instead.
//! Units: distances are world units; angles radians; [`LOOK_SPEED`] is radians
//!     per screen point; speeds are world units per second.
//! Related docs: `docs/GEOMETRY.md`; `docs/USER_GUIDE.md`;
//!     `docs/decisions/ADR-0006-viewer-integration.md`.

use brush_pyramid::scene::Camera;
use glam::{Quat, Vec3};

use crate::bundle::{BundleView, SceneScale};

/// Radians of rotation per screen point of mouse drag. Scene-scale
/// independent by construction: an angle is an angle in any scene.
pub const LOOK_SPEED: f32 = 0.005;

/// How much one scroll notch scales the fly speed (or the orbit distance).
pub const SPEED_STEP: f32 = 1.25;

/// Fly speed is clamped to this multiple of the scene's base speed, at least.
pub const MIN_SPEED_SCALE: f32 = 0.01;

/// Fly speed is clamped to this multiple of the scene's base speed, at most.
///
/// Was 10.0 until 2026-09-06. With [`crate::bundle::BASE_SPEED_FRACTION`] at
/// 2.0 this puts the top of the scroll range at 100 median camera gaps per
/// second — one press-and-hold crosses any capture in well under a second —
/// while [`SPEED_STEP`] still needs 18 notches to get there from the default,
/// so the wheel stays usable for small corrections.
pub const MAX_SPEED_SCALE: f32 = 50.0;

/// Pitch is clamped this far short of straight up/down, so the basis never
/// degenerates against the world up axis.
pub const PITCH_LIMIT: f32 = 0.01;

/// Orbit distance is clamped to this fraction of the camera box's diagonal, at
/// least — otherwise a few scroll notches put the pivot behind the near plane.
pub const MIN_ORBIT_FRACTION: f32 = 0.01;

/// Orbit distance is clamped to this multiple of the camera box's diagonal.
pub const MAX_ORBIT_FRACTION: f32 = 5.0;

/// A free-flying camera outside this multiple of the camera box (about its
/// centre) is somewhere no training view ever stood, so the HUD offers `R`.
pub const LOST_BOX_FACTOR: f32 = 3.0;

/// Below this a distance counts as zero, world units.
const TINY: f32 = 1e-6;

/// What a left-drag does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// Turn about a pivot inside the scene. The default: reviewing a scene
    /// means walking round the thing in it, and it cannot lose the subject.
    #[default]
    Orbit,
    /// Look around from where the camera is, first-person.
    Free,
}

impl Mode {
    /// The other one — what the `F` key does.
    #[must_use]
    pub const fn toggled(self) -> Self {
        match self {
            Self::Orbit => Self::Free,
            Self::Free => Self::Orbit,
        }
    }

    /// Short label for the HUD.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Orbit => "orbit",
            Self::Free => "free",
        }
    }
}

/// A world-to-camera frame plus the state needed to orbit or fly it.
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
    /// What a left-drag does.
    mode: Mode,
    /// The orbit pivot, always inside `scene.bounds`.
    target: Vec3,
    /// Where the capture cameras are: the source of every distance below.
    scene: SceneScale,
    /// Fly speed as a multiple of `scene.base_speed()`; the scroll wheel.
    speed_scale: f32,
    /// True until the user first moves the camera; see the module invariants.
    pinned: bool,
    /// The view being reproduced while `pinned`.
    pinned_view: BundleView,
    /// Array position in the bundle's `views` of `pinned_view`; `N`/`P`.
    view_position: usize,
    /// Array position the `R` key returns to.
    home_position: usize,
}

impl Controller {
    /// Open at `views[home_position]`, pinned to it, in [`Mode::Orbit`].
    ///
    /// # Arguments
    /// - `views`: every capture view, in dataset order.
    /// - `home_position`: array position to open at and to reset to; clamped
    ///   into range.
    /// - `up_world`: the scene's up vector; normalised here, and replaced with
    ///   `-Y` if it is degenerate.
    ///
    /// # Panics
    /// Panics if `views` is empty. `Bundle::load` rejects such a bundle, so
    /// this is unreachable from a loaded scene.
    #[must_use]
    pub fn new(views: &[BundleView], home_position: usize, up_world: [f32; 3]) -> Self {
        assert!(!views.is_empty(), "a camera controller needs at least one view");
        let home_position = home_position.min(views.len() - 1);
        let up = Vec3::from(up_world);
        let scene = SceneScale::from_views(views);
        let mut controller = Self {
            position: Vec3::ZERO,
            forward: Vec3::Z,
            right: Vec3::X,
            up_world: if up.length_squared() > 0.0 {
                up.normalize()
            } else {
                Vec3::NEG_Y
            },
            mode: Mode::default(),
            target: scene.bounds.centre(),
            scene,
            speed_scale: 1.0,
            pinned: true,
            pinned_view: views[home_position].clone(),
            view_position: home_position,
            home_position,
        };
        controller.snap_to_position(views, home_position);
        controller
    }

    /// True while the controller is still reproducing its dataset view exactly.
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        self.pinned
    }

    /// What a left-drag does.
    #[must_use]
    pub const fn mode(&self) -> Mode {
        self.mode
    }

    /// Switch between orbiting and free-look, re-deriving the pivot.
    pub fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
        if mode == Mode::Orbit {
            self.recentre_target();
        }
    }

    /// The scene's measured scale — the camera box and its spacing.
    #[must_use]
    pub const fn scene(&self) -> SceneScale {
        self.scene
    }

    /// Fly speed, world units per second.
    #[must_use]
    pub fn move_speed(&self) -> f32 {
        self.scene.base_speed() * self.speed_scale
    }

    /// Fly speed as a fraction of the camera box's diagonal per second — the
    /// scene-independent number the HUD shows, so "fast" means the same thing
    /// in the horse scene and in Karekare.
    #[must_use]
    pub fn speed_in_scenes(&self) -> f32 {
        self.move_speed() / self.scene.diameter()
    }

    /// Distance from the camera to the orbit pivot, world units.
    #[must_use]
    pub fn orbit_distance(&self) -> f32 {
        (self.position - self.target).length()
    }

    /// The orbit pivot, world units.
    #[must_use]
    pub const fn target(&self) -> Vec3 {
        self.target
    }

    /// Has a free-flying camera left the region the capture cameras cover?
    ///
    /// Only ever true in [`Mode::Free`]: orbiting clamps the pivot into the
    /// camera box and moves the camera with it, so it cannot happen there.
    #[must_use]
    pub fn is_lost(&self) -> bool {
        self.mode == Mode::Free
            && !self
                .scene
                .bounds
                .expanded(LOST_BOX_FACTOR)
                .contains(self.position)
    }

    /// Array position in `views` of the view last pinned or snapped to.
    #[must_use]
    pub const fn view_position(&self) -> usize {
        self.view_position
    }

    /// Jump to `views[position]` and pin to it again.
    ///
    /// Out-of-range positions are ignored rather than clamped: a stale combo
    /// box selection should do nothing, not move the camera somewhere else.
    pub fn snap_to_position(&mut self, views: &[BundleView], position: usize) {
        let Some(view) = views.get(position) else {
            return;
        };
        let r = &view.r;
        // Rows of a world-to-camera rotation are the camera axes in world
        // coordinates.
        self.right = Vec3::new(r[0], r[1], r[2]).normalize_or(Vec3::X);
        self.forward = Vec3::new(r[6], r[7], r[8]).normalize_or(Vec3::Z);
        self.position = view.position();
        self.pinned_view = view.clone();
        self.pinned = true;
        self.view_position = position;
        self.recentre_target();
    }

    /// Back to the view the viewer opened at — the `R` key.
    pub fn reset(&mut self, views: &[BundleView]) {
        self.snap_to_position(views, self.home_position);
    }

    /// Step `delta` capture views along, wrapping — the `N` and `P` keys.
    pub fn step_view(&mut self, views: &[BundleView], delta: isize) {
        if views.is_empty() {
            return;
        }
        let count = views.len() as isize;
        let wrapped = (self.view_position as isize + delta).rem_euclid(count);
        self.snap_to_position(views, wrapped as usize);
    }

    /// Put the orbit pivot on the view ray, as near the scene centre as that
    /// ray gets, then clamp it into the camera box.
    ///
    /// Using the ray rather than the scene centre itself is what stops the
    /// first orbit drag from snapping the view: the camera already looks
    /// exactly at a pivot chosen this way, so orbiting starts from the frame
    /// on screen. For a capture rig pointed at its subject — which is every
    /// TRIPS dataset — the pivot lands on the subject anyway.
    fn recentre_target(&mut self) {
        let along = (self.scene.bounds.centre() - self.position).dot(self.forward);
        let diagonal = self.scene.diameter();
        let distance = along.clamp(
            MIN_ORBIT_FRACTION * diagonal,
            MAX_ORBIT_FRACTION * diagonal,
        );
        self.target = self
            .scene
            .bounds
            .clamp_point(self.position + self.forward * distance);
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

    /// A left-drag, in screen points: orbit or look, depending on the mode.
    ///
    /// # Arguments
    /// - `dx`: rightward drag, points.
    /// - `dy`: downward drag, points.
    pub fn drag(&mut self, dx: f32, dy: f32) {
        match self.mode {
            Mode::Orbit => self.orbit(dx, dy),
            Mode::Free => self.look(dx, dy),
        }
    }

    /// Rotate in place by a mouse drag, in screen points (first-person look).
    ///
    /// # Arguments
    /// - `dx`: rightward drag, points; yaws about the world up axis.
    /// - `dy`: downward drag, points; pitches about the camera's right axis.
    pub fn look(&mut self, dx: f32, dy: f32) {
        if dx == 0.0 && dy == 0.0 {
            return;
        }
        self.pinned = false;
        self.rotate_in_place(-dx * LOOK_SPEED, -dy * LOOK_SPEED);
    }

    /// Yaw in place by an exact angle, radians — the scripted camera change
    /// `--camera-yaw-deg` uses, so a headless run can prove that a camera
    /// change reaches the renderer.
    pub fn yaw(&mut self, radians: f32) {
        self.pinned = false;
        self.rotate_in_place(radians, 0.0);
    }

    /// Yaw about `up_world` then pitch about the camera's right axis, radians,
    /// clamping the pitch short of the up axis.
    fn rotate_in_place(&mut self, yaw: f32, pitch: f32) {
        let turn = Quat::from_axis_angle(self.up_world, yaw);
        self.forward = turn * self.forward;
        self.right = turn * self.right;
        self.reorthonormalise();

        let candidate = Quat::from_axis_angle(self.right, pitch) * self.forward;
        if candidate.normalize_or(Vec3::Z).dot(self.up_world).abs() < 1.0 - PITCH_LIMIT {
            self.forward = candidate;
            self.reorthonormalise();
        }
    }

    /// Orbit the pivot by a mouse drag, in screen points.
    ///
    /// The distance to the pivot is preserved exactly: the offset is rotated
    /// by unit quaternions and never rescaled.
    ///
    /// # Arguments
    /// - `dx`: rightward drag, points; yaws the camera round the pivot.
    /// - `dy`: downward drag, points; raises the camera over the pivot.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        if dx == 0.0 && dy == 0.0 {
            return;
        }
        let mut offset = self.position - self.target;
        if offset.length_squared() < TINY * TINY {
            // Sitting on the pivot: there is nothing to orbit, so look.
            self.look(dx, dy);
            return;
        }
        self.pinned = false;

        let turn = Quat::from_axis_angle(self.up_world, -dx * LOOK_SPEED);
        offset = turn * offset;
        self.right = (turn * self.right).normalize_or(Vec3::X);

        let tip = Quat::from_axis_angle(self.right, -dy * LOOK_SPEED);
        let candidate = tip * offset;
        let candidate_forward = (-candidate).normalize_or(Vec3::Z);
        if candidate_forward.dot(self.up_world).abs() < 1.0 - PITCH_LIMIT {
            offset = candidate;
        }

        self.position = self.target + offset;
        self.forward = (self.target - self.position).normalize_or(self.forward);
        self.reorthonormalise();
    }

    /// Slide the camera (and, when orbiting, the pivot) across the image — a
    /// right- or middle-drag. Dragging moves the *scene* with the pointer.
    ///
    /// # Arguments
    /// - `dx`, `dy`: the drag, screen points.
    /// - `viewport_height`: the render area's height in the same points, so a
    ///   full-height drag pans by one pivot distance regardless of window size.
    pub fn pan(&mut self, dx: f32, dy: f32, viewport_height: f32) {
        if dx == 0.0 && dy == 0.0 {
            return;
        }
        self.pinned = false;
        let reference = match self.mode {
            Mode::Orbit => self.orbit_distance(),
            // Nothing is being looked *at* in free mode, so half the camera
            // box stands in for the pivot distance.
            Mode::Free => 0.5 * self.scene.diameter(),
        };
        let per_point = reference.max(TINY) / viewport_height.max(1.0);
        self.translate((self.right * -dx + self.down() * -dy) * per_point);
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
        let step = self.move_speed() * dt;
        self.translate(
            (self.forward * forward + self.right * right + self.up_world * up) * step,
        );
    }

    /// Move the camera, taking the orbit pivot with it and refusing any part
    /// of the move that would take the pivot out of the camera box.
    ///
    /// That refusal is the "you cannot fly out of the scene" rule: because the
    /// camera only ever moves by the same delta the pivot did, a clamped pivot
    /// clamps the camera too. Free mode is deliberately unclamped.
    fn translate(&mut self, delta: Vec3) {
        match self.mode {
            Mode::Free => self.position += delta,
            Mode::Orbit => {
                let clamped = self.scene.bounds.clamp_point(self.target + delta);
                let applied = clamped - self.target;
                self.target = clamped;
                self.position += applied;
            }
        }
    }

    /// The scroll wheel: fly speed when flying, distance when orbiting.
    ///
    /// # Arguments
    /// - `notches`: scroll notches, positive away from the user.
    pub fn scroll(&mut self, notches: f32) {
        match self.mode {
            Mode::Orbit => self.zoom(notches),
            Mode::Free => self.adjust_speed(notches),
        }
    }

    /// Scale the fly speed by `notches` scroll steps, clamped to
    /// `[MIN_SPEED_SCALE, MAX_SPEED_SCALE]` times the scene's base speed.
    pub fn adjust_speed(&mut self, notches: f32) {
        self.speed_scale =
            (self.speed_scale * SPEED_STEP.powf(notches)).clamp(MIN_SPEED_SCALE, MAX_SPEED_SCALE);
    }

    /// Move towards (positive `notches`) or away from the orbit pivot.
    pub fn zoom(&mut self, notches: f32) {
        let offset = self.position - self.target;
        let distance = offset.length();
        if distance < TINY {
            return;
        }
        self.pinned = false;
        let diagonal = self.scene.diameter();
        let wanted = (distance / SPEED_STEP.powf(notches)).clamp(
            MIN_ORBIT_FRACTION * diagonal,
            MAX_ORBIT_FRACTION * diagonal,
        );
        self.position = self.target + offset * (wanted / distance);
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

    const UP: [f32; 3] = [0.0, -1.0, 0.0];

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

    /// A ring of `count` cameras of radius `radius` in the world XZ plane, each
    /// looking inwards at the origin — a capture rig, the shape every test
    /// below wants and the shape the horse bundle actually has.
    fn ring(count: usize, radius: f32) -> Vec<BundleView> {
        (0..count)
            .map(|i| {
                let angle = std::f32::consts::TAU * i as f32 / count as f32;
                let centre = Vec3::new(radius * angle.cos(), 0.0, radius * angle.sin());
                let forward = (-centre).normalize();
                // World up is -Y, so camera "down" is +Y.
                let down = Vec3::Y;
                let right = down.cross(forward).normalize();
                let r = [
                    right.x, right.y, right.z, down.x, down.y, down.z, forward.x, forward.y,
                    forward.z,
                ];
                let t = [
                    -(r[0] * centre.x + r[1] * centre.y + r[2] * centre.z),
                    -(r[3] * centre.x + r[4] * centre.y + r[5] * centre.z),
                    -(r[6] * centre.x + r[7] * centre.y + r[8] * centre.z),
                ];
                BundleView {
                    index: i,
                    name: format!("{i:05}.jpg"),
                    width: 640,
                    height: 480,
                    fx: 500.0,
                    fy: 500.0,
                    cx: 320.0,
                    cy: 240.0,
                    r,
                    t,
                    distortion: [0.0; 8],
                }
            })
            .collect()
    }

    #[test]
    fn a_pinned_controller_reproduces_its_view_exactly() {
        let views = vec![view()];
        let c = Controller::new(&views, 0, UP);
        assert!(c.is_pinned());
        let v = &views[0];
        let cam = c.render_camera(v.width, v.height, v);
        assert_eq!(cam.r, v.r);
        assert_eq!(cam.t, v.t);
        assert_eq!(cam.distortion, v.distortion);
    }

    #[test]
    fn position_is_the_camera_centre() {
        let views = vec![view()];
        let c = Controller::new(&views, 0, UP);
        // R @ c + t must be the origin.
        let (r, t, p) = (views[0].r, views[0].t, c.position);
        for row in 0..3 {
            let value = r[row * 3] * p.x + r[row * 3 + 1] * p.y + r[row * 3 + 2] * p.z + t[row];
            assert!(value.abs() < 1e-5, "row {row} = {value}");
        }
    }

    #[test]
    fn moving_unpins_and_the_rebuilt_r_is_orthonormal() {
        let views = ring(12, 4.0);
        let mut c = Controller::new(&views, 0, UP);
        c.set_mode(Mode::Free);
        c.look(37.0, -11.0);
        c.fly(1.0, 0.5, -0.25, 0.016);
        assert!(!c.is_pinned());
        let cam = c.render_camera(1920, 1080, &views[0]);
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
        let views = ring(12, 4.0);
        let mut c = Controller::new(&views, 0, UP);
        c.set_mode(Mode::Free);
        for _ in 0..1000 {
            c.look(0.0, 100.0);
        }
        assert!(c.forward.dot(c.up_world).abs() < 1.0);
        assert!((c.right.length() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn a_wider_window_keeps_the_reference_field_of_view() {
        let views = vec![view()];
        let mut c = Controller::new(&views, 0, UP);
        c.set_mode(Mode::Free);
        c.look(1.0, 0.0);
        let half = c.render_camera(960, 540, &views[0]);
        assert!((half.fx - views[0].fx / 2.0).abs() < 1e-2);
        assert!((half.cx - 480.0).abs() < 1e-3);
    }

    // --- the input model (added after Jordan's 2026-09-06 test) -------------

    #[test]
    fn a_horizontal_drag_yaws_by_look_speed_per_pixel() {
        let views = ring(12, 4.0);
        let mut c = Controller::new(&views, 0, UP);
        c.set_mode(Mode::Free);
        let before = c.forward;
        c.look(100.0, 0.0);
        let after = c.forward;
        // 100 points of drag is 100 * LOOK_SPEED radians about the up axis.
        let expected = 100.0 * LOOK_SPEED;
        let cos = before.dot(after).clamp(-1.0, 1.0);
        assert!(
            (cos.acos() - expected).abs() < 1e-3,
            "yawed {} rad, wanted {expected}",
            cos.acos()
        );
        // Pure yaw: the vertical component of the view direction is unchanged.
        assert!((before.dot(c.up_world) - after.dot(c.up_world)).abs() < 1e-5);
        // ... and it is scene-scale independent: the same drag in a scene a
        // thousand times bigger turns by exactly the same angle.
        let big = ring(12, 4000.0);
        let mut c2 = Controller::new(&big, 0, UP);
        c2.set_mode(Mode::Free);
        let b2 = c2.forward;
        c2.look(100.0, 0.0);
        assert!((b2.dot(c2.forward).clamp(-1.0, 1.0).acos() - expected).abs() < 1e-3);
    }

    #[test]
    fn a_vertical_drag_pitches_and_the_clamp_holds_in_both_directions() {
        let views = ring(12, 4.0);
        for direction in [-1.0f32, 1.0] {
            let mut c = Controller::new(&views, 0, UP);
            c.set_mode(Mode::Free);
            c.look(0.0, direction * 40.0);
            let pitched = c.forward.dot(c.up_world);
            assert!(pitched.abs() > 1e-3, "a vertical drag must pitch");
            for _ in 0..500 {
                c.look(0.0, direction * 100.0);
            }
            assert!(
                c.forward.dot(c.up_world).abs() <= 1.0 - PITCH_LIMIT + 1e-3,
                "pitch escaped the clamp: {}",
                c.forward.dot(c.up_world)
            );
        }
    }

    #[test]
    fn orbiting_keeps_the_distance_to_the_pivot() {
        let views = ring(24, 4.0);
        let mut c = Controller::new(&views, 0, UP);
        assert_eq!(c.mode(), Mode::Orbit, "orbit is the default mode");
        let distance = c.orbit_distance();
        assert!(distance > 1.0, "the pivot should be near the ring centre");
        let target = c.target();
        for _ in 0..200 {
            c.orbit(9.0, -3.0);
        }
        assert!(
            (c.orbit_distance() - distance).abs() < 1e-3 * distance,
            "{} vs {distance}",
            c.orbit_distance()
        );
        assert!((c.target() - target).length() < 1e-5, "the pivot moved");
        // Still looking at the pivot, and still level.
        let to_target = (c.target() - c.position).normalize();
        assert!((to_target - c.forward).length() < 1e-4);
        assert!(c.right.dot(c.up_world).abs() < 1e-5);
    }

    #[test]
    fn orbiting_and_panning_cannot_leave_the_camera_box() {
        let views = ring(24, 4.0);
        let mut c = Controller::new(&views, 0, UP);
        let box_ = c.scene().bounds.expanded(1.0 + 1e-3);
        for _ in 0..2000 {
            c.fly(1.0, 0.3, 0.2, 0.05);
            c.pan(40.0, -25.0, 800.0);
        }
        assert!(box_.contains(c.target()), "the pivot left the camera box");
        assert!(!c.is_lost(), "orbit mode can never report itself lost");
    }

    #[test]
    fn free_flight_reports_itself_lost_outside_three_boxes() {
        let views = ring(24, 4.0);
        let mut c = Controller::new(&views, 0, UP);
        c.set_mode(Mode::Free);
        assert!(!c.is_lost());
        // Fly straight out for a long time at the top speed the wheel allows.
        for _ in 0..40 {
            c.adjust_speed(1.0);
        }
        for _ in 0..2000 {
            c.fly(1.0, 0.0, 0.0, 0.05);
        }
        assert!(c.is_lost(), "flew {} units out and was not lost", c.position.length());
        c.reset(&views);
        assert!(!c.is_lost(), "reset must bring the camera home");
    }

    #[test]
    fn the_base_speed_is_two_median_camera_gaps_per_second() {
        // A ring of 24 cameras of radius 4: the chord between neighbours is
        // 2 * 4 * sin(pi / 24) = 1.0405...
        let views = ring(24, 4.0);
        let scene = SceneScale::from_views(&views);
        let chord = 2.0 * 4.0 * (std::f32::consts::PI / 24.0).sin();
        assert!(
            (scene.median_spacing - chord).abs() < 1e-3,
            "{} vs {chord}",
            scene.median_spacing
        );
        let c = Controller::new(&views, 0, UP);
        // The shipped default, named as a number and not just as the constant:
        // one second of held `W` crosses two capture positions (2026-09-06,
        // Jordan: "I move so slow I can't explore the areas I want").
        assert!((crate::bundle::BASE_SPEED_FRACTION - 2.0).abs() < 1e-6);
        assert!(
            (c.move_speed() - 2.0 * chord).abs() < 1e-3,
            "{} should be 2 x the {chord} chord",
            c.move_speed()
        );
        // Ten thousand times bigger scene, ten thousand times the speed: the
        // ratio to the scene is what stays fixed.
        let big = Controller::new(&ring(24, 40000.0), 0, UP);
        assert!(
            (big.speed_in_scenes() - c.speed_in_scenes()).abs() < 1e-4,
            "{} vs {}",
            big.speed_in_scenes(),
            c.speed_in_scenes()
        );
    }

    #[test]
    fn the_scroll_wheel_reaches_fifty_times_the_base_speed() {
        // The top of the range, stated as a number: 50x the default, which on
        // a scene whose views are one chord apart is 100 capture gaps per
        // second. Reaching it takes ceil(ln 50 / ln 1.25) = 18 notches, so the
        // wheel is still fine-grained near the default.
        assert!((MAX_SPEED_SCALE - 50.0).abs() < 1e-6);
        let views = ring(24, 4.0);
        let mut c = Controller::new(&views, 0, UP);
        c.set_mode(Mode::Free);
        let base = c.move_speed();
        let notches = (MAX_SPEED_SCALE.ln() / SPEED_STEP.ln()).ceil() as usize;
        assert_eq!(notches, 18);
        for _ in 0..notches {
            c.scroll(1.0);
        }
        assert!(
            (c.move_speed() - base * MAX_SPEED_SCALE).abs() < 1e-3 * base,
            "{} vs {}",
            c.move_speed(),
            base * MAX_SPEED_SCALE
        );
    }

    #[test]
    fn orbit_zoom_moves_a_fixed_fraction_of_the_pivot_distance() {
        // Zoom is multiplicative, so the world-space step one notch takes is
        // proportional to how far the camera is from what it is looking at:
        // close in it creeps, far out it strides. A subtractive zoom would
        // take the same step at both distances and either crawl or overshoot.
        let views = ring(24, 4.0);
        let mut c = Controller::new(&views, 0, UP);
        let far = c.orbit_distance();
        c.scroll(1.0);
        let far_step = far - c.orbit_distance();
        assert!((c.orbit_distance() - far / SPEED_STEP).abs() < 1e-4 * far);

        // Now start ten times closer and measure the same single notch.
        for _ in 0..11 {
            c.scroll(1.0);
        }
        let near = c.orbit_distance();
        assert!(near < far / 5.0, "expected to be much closer: {near} vs {far}");
        c.scroll(1.0);
        let near_step = near - c.orbit_distance();
        assert!(
            (near_step / near - far_step / far).abs() < 1e-3,
            "step is not proportional: {near_step}/{near} vs {far_step}/{far}"
        );
    }

    #[test]
    fn the_scroll_wheel_scales_the_speed_and_clamps_it() {
        let views = ring(24, 4.0);
        let mut c = Controller::new(&views, 0, UP);
        c.set_mode(Mode::Free);
        let base = c.move_speed();
        c.scroll(1.0);
        assert!((c.move_speed() - base * SPEED_STEP).abs() < 1e-4 * base);
        c.scroll(-1.0);
        assert!((c.move_speed() - base).abs() < 1e-4 * base);
        for _ in 0..200 {
            c.scroll(1.0);
        }
        assert!((c.move_speed() - base * MAX_SPEED_SCALE).abs() < 1e-3 * base);
        for _ in 0..400 {
            c.scroll(-1.0);
        }
        assert!((c.move_speed() - base * MIN_SPEED_SCALE).abs() < 1e-4 * base);
    }

    #[test]
    fn scrolling_in_orbit_mode_zooms_instead_of_changing_the_speed() {
        let views = ring(24, 4.0);
        let mut c = Controller::new(&views, 0, UP);
        let speed = c.move_speed();
        let distance = c.orbit_distance();
        c.scroll(1.0);
        assert!((c.move_speed() - speed).abs() < 1e-9, "orbit scroll changed the fly speed");
        assert!(c.orbit_distance() < distance, "scrolling up must move closer");
        // And it cannot be scrolled through the pivot or out of the scene.
        for _ in 0..500 {
            c.scroll(1.0);
        }
        assert!(c.orbit_distance() >= MIN_ORBIT_FRACTION * c.scene().diameter() - 1e-4);
        for _ in 0..1000 {
            c.scroll(-1.0);
        }
        assert!(c.orbit_distance() <= MAX_ORBIT_FRACTION * c.scene().diameter() + 1e-3);
    }

    #[test]
    fn reset_restores_the_home_view_exactly() {
        let views = ring(24, 4.0);
        let mut c = Controller::new(&views, 0, UP);
        let home = c.render_camera(views[0].width, views[0].height, &views[0]);
        c.set_mode(Mode::Free);
        c.look(120.0, 40.0);
        c.fly(1.0, 1.0, 1.0, 1.0);
        assert!(!c.is_pinned());
        c.reset(&views);
        assert!(c.is_pinned(), "reset must re-pin to the dataset view");
        assert_eq!(c.view_position(), 0);
        let back = c.render_camera(views[0].width, views[0].height, &views[0]);
        assert_eq!(back.r, home.r);
        assert_eq!(back.t, home.t);
    }

    #[test]
    fn reset_returns_to_the_view_the_viewer_opened_at_not_view_zero() {
        let views = ring(24, 4.0);
        let mut c = Controller::new(&views, 7, UP);
        assert_eq!(c.view_position(), 7);
        c.step_view(&views, 3);
        assert_eq!(c.view_position(), 10);
        c.reset(&views);
        assert_eq!(c.view_position(), 7);
        assert!((c.position - views[7].position()).length() < 1e-5);
    }

    #[test]
    fn next_and_previous_cycle_through_every_view() {
        let views = ring(5, 4.0);
        let mut c = Controller::new(&views, 0, UP);
        let mut seen = Vec::new();
        for _ in 0..5 {
            seen.push(c.view_position());
            c.step_view(&views, 1);
        }
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
        assert_eq!(c.view_position(), 0, "next must wrap round the end");
        c.step_view(&views, -1);
        assert_eq!(c.view_position(), 4, "previous must wrap round the start");
        // Every step lands on a real capture pose, exactly.
        assert!((c.position - views[4].position()).length() < 1e-5);
        assert!(c.is_pinned());
    }

    #[test]
    fn the_f_key_toggles_between_orbit_and_free() {
        assert_eq!(Mode::Orbit.toggled(), Mode::Free);
        assert_eq!(Mode::Free.toggled(), Mode::Orbit);
        let views = ring(24, 4.0);
        let mut c = Controller::new(&views, 0, UP);
        c.set_mode(Mode::Free);
        c.look(200.0, 0.0);
        // Switching back to orbit re-derives a pivot the camera already looks
        // at, so the frame on screen does not jump.
        c.set_mode(Mode::Orbit);
        let to_target = (c.target() - c.position).normalize();
        assert!((to_target - c.forward).length() < 1e-4, "the pivot is off-axis");
    }

    #[test]
    fn a_scripted_yaw_rotates_by_exactly_the_angle_asked_for() {
        let views = ring(24, 4.0);
        let mut c = Controller::new(&views, 0, UP);
        let before = c.forward;
        c.yaw(12_f32.to_radians());
        assert!(!c.is_pinned(), "a scripted yaw must leave the pinned view");
        let turned = before.dot(c.forward).clamp(-1.0, 1.0).acos();
        assert!((turned - 12_f32.to_radians()).abs() < 1e-4, "{turned}");
        // The render camera must actually differ, or the screenshot check
        // downstream would be comparing two identical frames.
        let a = Controller::new(&views, 0, UP).render_camera(320, 240, &views[0]);
        let b = c.render_camera(320, 240, &views[0]);
        assert!(a.r.iter().zip(b.r).any(|(x, y)| (x - y).abs() > 1e-3));
    }
}
