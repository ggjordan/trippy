//! The 3D drag gizmos' maths: screen-space handles, and what a drag does to a region.
//!
//! Module: `trips_viewer::edit::gizmo`
//! Purpose: `docs/EDITOR.md` §6's E3/E1 gizmo row — translate a box/sphere/lid
//!     along its dominant axis by dragging, resize it with Shift-drag, rotate a
//!     box with Ctrl-drag. Everything here is arithmetic over a camera and a
//!     [`Params`]; the egui painter that draws the handles and the drag that
//!     feeds them live in `src/app.rs` and `src/edit_ui.rs`, the same split
//!     [`super::cluster`] uses.
//! Invariants:
//!     - A handle is placed by PROJECTING a world point, never by inventing a
//!       screen-space frame: the three arms are `centre ± arm * e_i` for the
//!       world axes, projected with the same [`ClickCamera`] a click is, so a
//!       handle sits where the axis really points in this frame.
//!     - The dominant axis is chosen ONCE, when the drag starts, and held for
//!       the whole gesture. Re-choosing it per frame makes a diagonal drag
//!       jitter between two axes, which is the classic gizmo bug.
//!     - Every drag maps to world units through the handle's own projected
//!       length, so dragging the handle to where the pointer is moves the
//!       region by exactly what the screen shows, at any zoom and any depth.
//!     - A region whose centre is behind the camera has NO handles
//!       ([`GizmoScreen::project`] returns `None`) rather than handles mirrored
//!       through the focal point.
//!     - `pointset` and `brush` regions have no gizmo: there is no shape to
//!       drag (a pointset is a list of ids, a brush is a voxel set), and the
//!       keyboard nudges refuse them for the same reason.
//! Units: world units for every centre, radius and half extent; pixels for
//!     every screen coordinate (the RENDER's pixels, not egui points);
//!     radians for rotation.
//! Related docs: `docs/EDITOR.md` §4 "Keys", §6's E1/E3 rows.

use super::cluster::ClickCamera;
use super::model::{LidParams, Params};

/// How long a gizmo arm is, as a multiple of the region's own size.
///
/// Long enough to grab without covering the region, short enough that three of
/// them do not fill the frame.
pub const ARM_FACTOR: f64 = 1.6;

/// How close (render pixels) the pointer must be to a handle to grab it.
///
/// A generous target: the handle is drawn small so it does not hide the scene,
/// but a 3D gizmo that needs pixel-perfect aim is a 3D gizmo nobody uses.
pub const GRAB_PX: f64 = 18.0;

/// A gizmo drag shorter than this many render pixels does nothing.
///
/// The same guard the SAM tool's `MIN_BOX_PX` applies: a click that wobbles two
/// pixels is a click, and it must not silently nudge a region.
pub const MIN_DRAG_PX: f64 = 2.0;

/// Screen pixels of Shift-drag that double (or halve) a region.
///
/// A resize is unbounded above, so it cannot be a linear map from pixels; this
/// is the exponent's scale: `factor = 2^(along / RESIZE_PX_PER_DOUBLING)`.
pub const RESIZE_PX_PER_DOUBLING: f64 = 160.0;

/// The smallest a Shift-drag may leave a region, world units.
///
/// A half extent of zero is rejected by [`Params::from_json`], so a resize that
/// could reach it would make the region unloadable.
pub const MIN_SIZE: f64 = 1e-6;

/// Which drag a gesture on a handle performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Drag {
    /// Plain drag: move along the grabbed axis.
    Translate,
    /// Shift-drag: grow or shrink.
    Resize,
    /// Ctrl-drag: rotate about the grabbed axis (`box` only).
    Rotate,
}

/// One projected axis arm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arm {
    /// 0 = world X, 1 = world Y, 2 = world Z.
    pub axis: usize,
    /// The handle tip, render pixels.
    pub tip: (f64, f64),
    /// Tip minus centre, render pixels. Zero-length when the axis points
    /// straight at the camera, which is exactly when it cannot be dragged.
    pub delta: (f64, f64),
    /// World units this arm is long (`ARM_FACTOR * size`).
    pub world_len: f64,
    /// Whether the arm points AWAY from the camera (its tip is deeper than the
    /// centre). The rotation sign follows it: a screen twist means opposite
    /// rotations about an axis pointing towards you and one pointing away.
    pub away: bool,
}

impl Arm {
    /// The arm's projected length in pixels.
    #[must_use]
    pub fn pixel_len(&self) -> f64 {
        self.delta.0.hypot(self.delta.1)
    }
}

/// A region's gizmo, projected into this frame.
#[derive(Debug, Clone, PartialEq)]
pub struct GizmoScreen {
    /// The region's centre, render pixels.
    pub centre: (f64, f64),
    /// The three world-axis arms.
    pub arms: [Arm; 3],
}

/// The centre and size a gizmo is built around, for the kinds that have one.
///
/// `size` is the region's own scale in world units: a box's largest half
/// extent, a sphere's radius, a lid's radius. `None` for `pointset`/`brush`.
#[must_use]
pub fn anchor(params: &Params) -> Option<([f64; 3], f64)> {
    match params {
        Params::Box {
            center,
            half_extents,
            ..
        } => Some((
            *center,
            half_extents[0].max(half_extents[1]).max(half_extents[2]),
        )),
        Params::Sphere { center, radius } => Some((*center, *radius)),
        Params::Lid(lid) => Some((lid.center, lid.radius)),
        Params::Pointset { .. } | Params::Brush { .. } => None,
    }
}

impl GizmoScreen {
    /// Project a region's handles into the frame `camera` drew.
    ///
    /// Returns `None` when the region has no shape to drag, or when its centre
    /// (or every arm tip) is not in front of the camera.
    #[must_use]
    pub fn project(camera: &ClickCamera, params: &Params) -> Option<Self> {
        let (centre_world, size) = anchor(params)?;
        let arm_len = (size * ARM_FACTOR).max(MIN_SIZE);
        let (cu, cv, cz) = camera.project(centre_world);
        if !(cz > 0.0) || !cu.is_finite() || !cv.is_finite() {
            return None;
        }
        let mut arms = [Arm {
            axis: 0,
            tip: (cu, cv),
            delta: (0.0, 0.0),
            world_len: arm_len,
            away: false,
        }; 3];
        for (axis, arm) in arms.iter_mut().enumerate() {
            let mut tip_world = centre_world;
            tip_world[axis] += arm_len;
            let (u, v, z) = camera.project(tip_world);
            // An arm tip behind the camera projects to a mirrored, meaningless
            // pixel; report a zero-length arm instead, which `pick` skips.
            let ok = z > 0.0 && u.is_finite() && v.is_finite();
            *arm = Arm {
                axis,
                tip: if ok { (u, v) } else { (cu, cv) },
                delta: if ok { (u - cu, v - cv) } else { (0.0, 0.0) },
                world_len: arm_len,
                away: z >= cz,
            };
        }
        Some(Self {
            centre: (cu, cv),
            arms,
        })
    }

    /// The axis whose handle is within `GRAB_PX` of `px`, nearest first.
    ///
    /// This is what makes the gizmo cost no navigation: a drag that does not
    /// START on a handle is not a gizmo drag at all, so a plain drag anywhere
    /// else still orbits (`docs/EDITOR.md` §4's "a drag is scoped to what it
    /// started on").
    #[must_use]
    pub fn pick(&self, px: (f64, f64)) -> Option<usize> {
        let mut best: Option<(f64, usize)> = None;
        for arm in &self.arms {
            if arm.pixel_len() <= f64::EPSILON {
                continue;
            }
            let d = (arm.tip.0 - px.0).hypot(arm.tip.1 - px.1);
            if d <= GRAB_PX && best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, arm.axis));
            }
        }
        best.map(|(_, axis)| axis)
    }

    /// How far along `axis`, in world units, a screen drag of `drag` moves the region.
    ///
    /// The projection of the drag onto the arm, scaled by the arm's own
    /// world-per-pixel ratio: dragging the handle under the pointer moves the
    /// region by what the screen shows.
    #[must_use]
    pub fn translate_world(&self, axis: usize, drag: (f64, f64)) -> f64 {
        let arm = &self.arms[axis];
        let len2 = arm.delta.0.mul_add(arm.delta.0, arm.delta.1 * arm.delta.1);
        if len2 <= f64::EPSILON {
            return 0.0;
        }
        let along = arm.delta.0.mul_add(drag.0, arm.delta.1 * drag.1) / len2;
        along * arm.world_len
    }

    /// The scale factor a Shift-drag of `drag` applies.
    ///
    /// Exponential in the pixels dragged along the arm, so growing and
    /// shrinking are symmetric and neither can reach zero.
    #[must_use]
    pub fn resize_factor(&self, axis: usize, drag: (f64, f64)) -> f64 {
        let arm = &self.arms[axis];
        let len = arm.pixel_len();
        if len <= f64::EPSILON {
            return 1.0;
        }
        let along = arm.delta.0.mul_add(drag.0, arm.delta.1 * drag.1) / len;
        (along / RESIZE_PX_PER_DOUBLING).exp2()
    }

    /// The angle (radians) a Ctrl-drag from `from` to `to` twists about `axis`.
    ///
    /// The signed screen angle the pointer swept around the region's centre,
    /// flipped when the axis points towards the camera — otherwise twisting the
    /// same way would rotate a box one way when seen from the front and the
    /// other way from behind.
    #[must_use]
    pub fn rotate_angle(&self, axis: usize, from: (f64, f64), to: (f64, f64)) -> f64 {
        let a = (from.0 - self.centre.0, from.1 - self.centre.1);
        let b = (to.0 - self.centre.0, to.1 - self.centre.1);
        if a.0.hypot(a.1) < 1.0 || b.0.hypot(b.1) < 1.0 {
            // Within a pixel of the centre there is no meaningful angle.
            return 0.0;
        }
        let cross = a.0.mul_add(b.1, -(a.1 * b.0));
        let dot = a.0.mul_add(b.0, a.1 * b.1);
        let angle = cross.atan2(dot);
        if self.arms[axis].away {
            angle
        } else {
            -angle
        }
    }
}

// --- what a drag does to the region -------------------------------------------------

/// `params` with its centre moved by `delta` world units, or `None` for a
/// kind with no centre.
///
/// Shared by the gizmo and by the arrow-key nudges, so the two can never drift.
#[must_use]
pub fn translated(params: &Params, delta: [f64; 3]) -> Option<Params> {
    let add = |a: [f64; 3]| [a[0] + delta[0], a[1] + delta[1], a[2] + delta[2]];
    match params {
        Params::Box {
            center,
            half_extents,
            quat,
        } => Some(Params::Box {
            center: add(*center),
            half_extents: *half_extents,
            quat: *quat,
        }),
        Params::Sphere { center, radius } => Some(Params::Sphere {
            center: add(*center),
            radius: *radius,
        }),
        Params::Lid(lid) => Some(Params::Lid(LidParams {
            center: add(lid.center),
            ..*lid
        })),
        // A pointset has no centre to move, and moving a brush would move the
        // voxel grid out from under the cells painted into it.
        Params::Pointset { .. } | Params::Brush { .. } => None,
    }
}

/// `params` scaled by `factor` about its own centre, or `None` for a kind with
/// no size. Never smaller than [`MIN_SIZE`].
#[must_use]
pub fn resized(params: &Params, factor: f64) -> Option<Params> {
    if !factor.is_finite() || factor <= 0.0 {
        return None;
    }
    match params {
        Params::Box {
            center,
            half_extents,
            quat,
        } => Some(Params::Box {
            center: *center,
            half_extents: [
                (half_extents[0] * factor).max(MIN_SIZE),
                (half_extents[1] * factor).max(MIN_SIZE),
                (half_extents[2] * factor).max(MIN_SIZE),
            ],
            quat: *quat,
        }),
        Params::Sphere { center, radius } => Some(Params::Sphere {
            center: *center,
            radius: (radius * factor).max(MIN_SIZE),
        }),
        Params::Lid(lid) => Some(Params::Lid(LidParams {
            radius: (lid.radius * factor).max(MIN_SIZE),
            ..*lid
        })),
        Params::Pointset { .. } | Params::Brush { .. } => None,
    }
}

/// `params` rotated by `angle` radians about world axis `axis`, `box` only.
///
/// A box is the one kind with an orientation: a sphere is invariant under
/// rotation and the lid's own plane is given by `up`, which the Inspector
/// edits directly. Pre-multiplying is what makes this a WORLD-frame rotation
/// (`R' = R_axis R`, matching `box_membership`'s `local = R^T (p - centre)`).
#[must_use]
pub fn rotated(params: &Params, axis: usize, angle: f64) -> Option<Params> {
    let Params::Box {
        center,
        half_extents,
        quat,
    } = params
    else {
        return None;
    };
    if !angle.is_finite() || angle == 0.0 {
        return None;
    }
    let mut world_axis = [0.0_f64; 3];
    world_axis[axis.min(2)] = 1.0;
    Some(Params::Box {
        center: *center,
        half_extents: *half_extents,
        quat: quat_mul(quat_from_axis_angle(world_axis, angle), *quat),
    })
}

/// A unit quaternion `(w, x, y, z)` for a rotation of `angle` about `axis`.
#[must_use]
pub fn quat_from_axis_angle(axis: [f64; 3], angle: f64) -> [f64; 4] {
    let norm = axis
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    if norm < 1e-12 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let (s, c) = (angle * 0.5).sin_cos();
    [
        c,
        axis[0] / norm * s,
        axis[1] / norm * s,
        axis[2] / norm * s,
    ]
}

/// Hamilton product `a * b`, both `(w, x, y, z)`.
#[must_use]
pub fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edit::model::box_membership;

    /// A camera at the world origin looking down +Z, 100 px focal length,
    /// principal point (100, 100) — so a world point at `(0, 0, 10)` lands
    /// dead centre and one world unit of X at that depth is 10 px.
    fn camera() -> ClickCamera {
        ClickCamera {
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            t: [0.0, 0.0, 0.0],
            fx: 100.0,
            fy: 100.0,
            cx: 100.0,
            cy: 100.0,
        }
    }

    fn sphere(centre: [f64; 3], radius: f64) -> Params {
        Params::Sphere {
            center: centre,
            radius,
        }
    }

    #[test]
    fn the_handles_sit_where_the_axes_really_point() {
        let params = sphere([0.0, 0.0, 10.0], 1.0);
        let g = GizmoScreen::project(&camera(), &params).expect("in front of the camera");
        assert!((g.centre.0 - 100.0).abs() < 1e-9 && (g.centre.1 - 100.0).abs() < 1e-9);
        // +X is 1.6 world units long at depth 10 with fx 100: 16 px to the right.
        assert!((g.arms[0].delta.0 - 16.0).abs() < 1e-6, "{:?}", g.arms[0]);
        assert!(g.arms[0].delta.1.abs() < 1e-9);
        // +Z points straight away from this camera: it projects to (almost)
        // nothing on screen and must not be grabbable.
        assert!(g.arms[2].pixel_len() < 1e-6, "{:?}", g.arms[2]);
        assert!(g.arms[2].away, "+Z is the away direction here");
    }

    #[test]
    fn a_region_behind_the_camera_has_no_gizmo() {
        assert!(GizmoScreen::project(&camera(), &sphere([0.0, 0.0, -5.0], 1.0)).is_none());
        // ... and a pointset never has one, wherever it is.
        assert!(GizmoScreen::project(
            &camera(),
            &Params::Pointset {
                point_ids: vec![1, 2]
            }
        )
        .is_none());
    }

    #[test]
    fn a_drag_moves_the_region_by_what_the_screen_shows() {
        let params = sphere([0.0, 0.0, 10.0], 1.0);
        let g = GizmoScreen::project(&camera(), &params).expect("visible");
        let axis = g.pick((116.0, 100.0)).expect("the +X handle");
        assert_eq!(axis, 0);
        // Dragging the handle 16 px further right is one more arm length: 1.6 u.
        let delta = g.translate_world(axis, (16.0, 0.0));
        assert!((delta - 1.6).abs() < 1e-6, "{delta}");
        // A perpendicular drag on the same axis moves nothing.
        assert!(g.translate_world(axis, (0.0, 20.0)).abs() < 1e-9);
        // And the drag lands where it says: 10 px right is 0.1 world units at
        // this depth, which is 10 px worth of the 100 px focal length.
        let moved = translated(&params, [g.translate_world(axis, (10.0, 0.0)), 0.0, 0.0])
            .expect("a sphere moves");
        let Params::Sphere { center, .. } = moved else {
            panic!("still a sphere")
        };
        assert!((center[0] - 1.0).abs() < 1e-6, "{center:?}");
    }

    #[test]
    fn a_drag_that_misses_every_handle_grabs_nothing() {
        let g = GizmoScreen::project(&camera(), &sphere([0.0, 0.0, 10.0], 1.0)).expect("visible");
        assert!(g.pick((160.0, 160.0)).is_none(), "far from every handle");
        // The +Z arm projects to zero length and is never picked, even though
        // its "tip" is exactly at the centre.
        assert_ne!(g.pick((100.0, 100.0)), Some(2));
    }

    #[test]
    fn shift_drag_scales_symmetrically_and_never_to_zero() {
        let g = GizmoScreen::project(&camera(), &sphere([0.0, 0.0, 10.0], 1.0)).expect("visible");
        let grow = g.resize_factor(0, (RESIZE_PX_PER_DOUBLING, 0.0));
        let shrink = g.resize_factor(0, (-RESIZE_PX_PER_DOUBLING, 0.0));
        assert!((grow - 2.0).abs() < 1e-9, "{grow}");
        assert!((shrink - 0.5).abs() < 1e-9, "{shrink}");
        // Even an absurd shrink leaves a loadable region.
        let tiny = resized(&sphere([0.0, 0.0, 10.0], 1.0), 1e-30).expect("a sphere resizes");
        let Params::Sphere { radius, .. } = tiny else {
            panic!("still a sphere")
        };
        assert!(radius >= MIN_SIZE, "{radius}");
    }

    #[test]
    fn ctrl_drag_rotates_a_box_in_the_world_frame() {
        let params = Params::Box {
            center: [0.0, 0.0, 10.0],
            half_extents: [2.0, 0.2, 0.2],
            quat: [1.0, 0.0, 0.0, 0.0],
        };
        // A quarter turn about world +Z takes the box's long axis onto +Y.
        let rotated = rotated(&params, 2, std::f64::consts::FRAC_PI_2).expect("a box rotates");
        let Params::Box {
            center,
            half_extents,
            quat,
        } = rotated
        else {
            panic!("still a box")
        };
        // (0, 1.5, 10) is inside the rotated box and outside the original one.
        assert_eq!(
            box_membership([0.0, 1.5, 10.0], center, half_extents, quat),
            1.0
        );
        assert_eq!(
            box_membership([0.0, 1.5, 10.0], center, half_extents, [1.0, 0.0, 0.0, 0.0]),
            0.0
        );
        // Nothing but a box rotates.
        assert!(super::rotated(&sphere([0.0, 0.0, 1.0], 1.0), 0, 1.0).is_none());
    }

    #[test]
    fn the_twist_sign_follows_the_axis_direction() {
        let g = GizmoScreen::project(&camera(), &sphere([0.0, 0.0, 10.0], 1.0)).expect("visible");
        // +Z points away from this camera; +X points across it (its tip is at
        // the same depth, so `away` is true for it too under `z >= cz`), so use
        // the two the test can tell apart: a twist about +Z one way and the
        // same twist about -Z would differ, which `away` is what encodes.
        let clockwise = g.rotate_angle(2, (120.0, 100.0), (100.0, 120.0));
        assert!(clockwise > 0.0, "a screen-clockwise twist about an away axis is positive");
        let back = g.rotate_angle(2, (100.0, 120.0), (120.0, 100.0));
        assert!((clockwise + back).abs() < 1e-9, "twisting back undoes it");
        // A twist that starts on the centre has no angle.
        assert_eq!(g.rotate_angle(2, (100.0, 100.0), (120.0, 100.0)), 0.0);
    }

    #[test]
    fn a_quaternion_product_composes_two_rotations() {
        let half = std::f64::consts::FRAC_PI_2;
        let a = quat_from_axis_angle([0.0, 0.0, 1.0], half);
        let both = quat_mul(a, a);
        let full = quat_from_axis_angle([0.0, 0.0, 1.0], 2.0 * half);
        for (x, y) in both.iter().zip(&full) {
            assert!((x - y).abs() < 1e-12, "{both:?} vs {full:?}");
        }
    }
}
