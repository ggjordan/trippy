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
    /// Drag the lid's own normal handle: tilt `up` (`lid` only).
    Tilt,
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

/// The pseudo-axis [`GizmoScreen::pick`] returns for the lid's normal handle —
/// past the three world axes (`0`/`1`/`2`), never a valid index into
/// [`GizmoScreen::arms`]. [`Drag`]'s `Tilt` variant is the only one
/// `edit_ui.rs` builds when this is picked.
pub const NORMAL_AXIS: usize = 3;

/// A projected screen-space direction from a gizmo's centre, stripped of the
/// axis/rotation bookkeeping [`Arm`] carries.
///
/// Used only to turn a 2D screen drag back into a 3D world displacement —
/// the lid's normal handle has no world AXIS to move along (it drags a
/// *direction*, `up`), so its two "arms" are in-plane basis directions chosen
/// from `up` itself, not world X/Y/Z, and nothing outside [`GizmoScreen::tilted_up`]
/// needs to know they exist.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ScreenArm {
    /// Tip minus centre, render pixels.
    delta: (f64, f64),
    /// World units this arm is long.
    world_len: f64,
}

impl ScreenArm {
    /// Project `centre_world + dir * len` and record its screen offset from
    /// `centre_screen`. A zero-length result (behind the camera, or the
    /// direction points straight at/away from it) is not an error — it just
    /// contributes nothing to [`GizmoScreen::tilted_up`]'s reconstruction,
    /// exactly like a zero-length [`Arm`] is skipped by [`GizmoScreen::pick`].
    fn project(
        camera: &ClickCamera,
        centre_world: [f64; 3],
        centre_screen: (f64, f64),
        dir: [f64; 3],
        len: f64,
    ) -> Self {
        let tip_world = [
            dir[0].mul_add(len, centre_world[0]),
            dir[1].mul_add(len, centre_world[1]),
            dir[2].mul_add(len, centre_world[2]),
        ];
        let (u, v, z) = camera.project(tip_world);
        let ok = z > 0.0 && u.is_finite() && v.is_finite();
        Self {
            delta: if ok { (u - centre_screen.0, v - centre_screen.1) } else { (0.0, 0.0) },
            world_len: len,
        }
    }

    /// How far along this direction, in world units, a screen drag of `drag`
    /// moves the tip — the same projection-onto-the-arm trick
    /// [`GizmoScreen::translate_world`] uses.
    fn along_world(&self, drag: (f64, f64)) -> f64 {
        let len2 = self.delta.0.mul_add(self.delta.0, self.delta.1 * self.delta.1);
        if len2 <= f64::EPSILON {
            return 0.0;
        }
        let along = self.delta.0.mul_add(drag.0, self.delta.1 * drag.1) / len2;
        along * self.world_len
    }
}

/// The lid's own 4th handle: drag it to tilt `up` (rotate about the two
/// in-plane axes), rather than translate/resize/rotate a world axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalHandle {
    /// The handle tip, render pixels.
    pub tip: (f64, f64),
    /// How far from the centre the tip sits, world units — `up`'s own length
    /// in the reconstruction [`GizmoScreen::tilted_up`] does.
    len: f64,
    /// Two in-plane basis directions (world unit vectors orthogonal to `up`
    /// and to each other), each projected exactly like a world-axis [`Arm`].
    /// Captured once, at projection time, so a drag needs no further camera
    /// calls — the same contract [`Arm`] already keeps.
    basis: [ScreenArm; 2],
}

/// Two unit vectors orthogonal to `n` and to each other — an arbitrary but
/// DETERMINISTIC basis for the plane `n` is normal to (Duff et al.'s "branch
/// on the largest component" trick is not needed here: a lid's `up` is never
/// exactly axis-aligned in practice, and the simple helper-vector construction
/// is exact wherever it is defined, which is everywhere `n` is not ~zero).
///
/// # Panics
/// Never in practice: [`Params::from_json`] already refuses a ~zero `up`, and
/// every caller here holds a validated [`LidParams`].
fn orthonormal_basis(n: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let norm = dot3(n, n).sqrt().max(1e-12);
    let n = [n[0] / norm, n[1] / norm, n[2] / norm];
    // A helper vector not (nearly) parallel to `n`: world X unless `n` is
    // mostly X, in which case world Y is never nearly parallel to it either.
    let helper = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let b1_raw = cross(helper, n);
    let b1_norm = dot3(b1_raw, b1_raw).sqrt().max(1e-12);
    let b1 = [b1_raw[0] / b1_norm, b1_raw[1] / b1_norm, b1_raw[2] / b1_norm];
    // `n` and `b1` are already orthonormal, so `n x b1` is unit length too.
    let b2 = cross(n, b1);
    (b1, b2)
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalized(v: [f64; 3]) -> [f64; 3] {
    let norm = dot3(v, v).sqrt();
    if norm < 1e-12 {
        return v;
    }
    [v[0] / norm, v[1] / norm, v[2] / norm]
}

/// A region's gizmo, projected into this frame.
#[derive(Debug, Clone, PartialEq)]
pub struct GizmoScreen {
    /// The region's centre, render pixels.
    pub centre: (f64, f64),
    /// The three world-axis arms.
    pub arms: [Arm; 3],
    /// The lid's own plane-normal handle — `None` for every other kind (there
    /// is no `up` to tilt).
    pub normal: Option<NormalHandle>,
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
        // The lid's own 4th handle: `up`'s own tip, plus two in-plane basis
        // directions captured now so a later drag needs no more camera calls
        // (`ScreenArm::project`'s own doc comment).
        let normal = if let Params::Lid(lid) = params {
            let up = normalized(lid.up);
            let (b1, b2) = orthonormal_basis(up);
            let tip_world = [
                up[0].mul_add(arm_len, centre_world[0]),
                up[1].mul_add(arm_len, centre_world[1]),
                up[2].mul_add(arm_len, centre_world[2]),
            ];
            let (u, v, z) = camera.project(tip_world);
            (z > 0.0 && u.is_finite() && v.is_finite()).then(|| NormalHandle {
                tip: (u, v),
                len: arm_len,
                basis: [
                    ScreenArm::project(camera, centre_world, (cu, cv), b1, arm_len),
                    ScreenArm::project(camera, centre_world, (cu, cv), b2, arm_len),
                ],
            })
        } else {
            None
        };

        Some(Self {
            centre: (cu, cv),
            arms,
            normal,
        })
    }

    /// The axis whose handle is within `GRAB_PX` of `px`, nearest first — or
    /// [`NORMAL_AXIS`] when the lid's own normal handle is the nearest.
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
        if let Some(normal) = &self.normal {
            let d = (normal.tip.0 - px.0).hypot(normal.tip.1 - px.1);
            if d <= GRAB_PX && best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, NORMAL_AXIS));
            }
        }
        best.map(|(_, axis)| axis)
    }

    /// The new `up` a drag of `drag` screen pixels on the normal handle makes,
    /// or `None` when this gizmo has no normal handle (not a lid) or the drag
    /// cannot be decomposed (both basis arms are degenerate — practically
    /// unreachable, since `orthonormal_basis` only fails for a ~zero `up`,
    /// which [`Params::from_json`] already refuses).
    ///
    /// Reconstructs the tip's new world position from `up`'s own length plus
    /// the drag's components along the two in-plane basis directions captured
    /// at projection time, then re-normalises — no live camera needed, the
    /// same contract every other drag method here keeps.
    #[must_use]
    pub fn tilted_up(&self, up: [f64; 3], drag: (f64, f64)) -> Option<[f64; 3]> {
        let normal = self.normal.as_ref()?;
        let up = normalized(up);
        let (b1, b2) = orthonormal_basis(up);
        let d1 = normal.basis[0].along_world(drag);
        let d2 = normal.basis[1].along_world(drag);
        let new_dir = [
            b1[0].mul_add(d1, b2[0].mul_add(d2, up[0] * normal.len)),
            b1[1].mul_add(d1, b2[1].mul_add(d2, up[1] * normal.len)),
            b1[2].mul_add(d1, b2[2].mul_add(d2, up[2] * normal.len)),
        ];
        if dot3(new_dir, new_dir) < 1e-18 {
            return None;
        }
        Some(normalized(new_dir))
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

/// `params` with its `up` replaced by `new_up`, `lid` only.
///
/// The normal handle's own drag result — [`GizmoScreen::tilted_up`] computes
/// `new_up` from a screen drag; this is the last step, mirroring how
/// [`translated`]/[`resized`]/[`rotated`] each turn a drag's number into a new
/// [`Params`]. `new_up` is expected already-normalised (as `tilted_up`
/// returns it) but this re-validates rather than trusting the caller, the same
/// way `Params::from_json` refuses a ~zero `up`.
#[must_use]
pub fn tilted(params: &Params, new_up: [f64; 3]) -> Option<Params> {
    let Params::Lid(lid) = params else {
        return None;
    };
    if dot3(new_up, new_up) < 1e-18 {
        return None;
    }
    Some(Params::Lid(LidParams {
        up: new_up,
        ..*lid
    }))
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

    fn lid(center: [f64; 3], up: [f64; 3], radius: f64) -> Params {
        Params::Lid(LidParams {
            up,
            height: 0.0,
            center,
            radius,
            falloff: 0.0,
            band: 0.0,
        })
    }

    #[test]
    fn a_lid_gets_a_fourth_handle_for_its_normal() {
        let params = lid([0.0, 0.0, 10.0], [0.0, 1.0, 0.0], 1.0);
        let g = GizmoScreen::project(&camera(), &params).expect("visible");
        let normal = g.normal.expect("a lid has a normal handle");
        // `up` here IS world +Y, so the handle sits exactly where the +Y arm
        // does — the same "handles sit where the axes really point" check the
        // three world arms get, extended to the fourth.
        assert!((normal.tip.0 - g.arms[1].tip.0).abs() < 1e-9, "{normal:?} vs {:?}", g.arms[1]);
        assert!((normal.tip.1 - g.arms[1].tip.1).abs() < 1e-9);

        // No other kind has one.
        let sphere_g = GizmoScreen::project(&camera(), &sphere([0.0, 0.0, 10.0], 1.0)).unwrap();
        assert!(sphere_g.normal.is_none());
    }

    #[test]
    fn picking_the_normal_handle_returns_normal_axis() {
        // `up` NOT aligned with a world axis, so the handle is unambiguously
        // its own point on screen (no tie with an arm to break).
        let params = lid([0.0, 0.0, 10.0], [0.6, 0.8, 0.0], 1.0);
        let g = GizmoScreen::project(&camera(), &params).expect("visible");
        let tip = g.normal.expect("normal handle").tip;
        assert_eq!(g.pick(tip), Some(NORMAL_AXIS));
        // Far from every handle still grabs nothing.
        assert!(g.pick((400.0, 400.0)).is_none());
    }

    #[test]
    fn dragging_the_normal_handle_tilts_up_and_stays_unit_length() {
        let params = lid([0.0, 0.0, 10.0], [0.0, 1.0, 0.0], 1.0);
        let g = GizmoScreen::project(&camera(), &params).expect("visible");

        // No drag at all: `up` comes back exactly as it went in (normalised).
        let same = g.tilted_up([0.0, 1.0, 0.0], (0.0, 0.0)).expect("a lid tilts");
        assert!(same[0].abs() < 1e-9 && (same[1] - 1.0).abs() < 1e-9 && same[2].abs() < 1e-9);

        // Dragging the tip sideways tilts `up` away from +Y, staying unit length.
        let tilted = g.tilted_up([0.0, 1.0, 0.0], (20.0, 0.0)).expect("a lid tilts");
        assert!((dot3(tilted, tilted) - 1.0).abs() < 1e-9, "not unit length: {tilted:?}");
        assert!((tilted[1] - 1.0).abs() > 1e-6, "the drag did not move it: {tilted:?}");

        // `tilted()` applies the result to the region, every other field kept.
        let moved = super::tilted(&params, tilted).expect("still a lid");
        let Params::Lid(after) = moved else {
            panic!("still a lid")
        };
        assert_eq!(after.up, tilted);
        assert_eq!(after.height, 0.0);
        assert_eq!(after.center, [0.0, 0.0, 10.0]);
        assert_eq!(after.radius, 1.0);

        // Nothing but a lid tilts.
        assert!(super::tilted(&sphere([0.0, 0.0, 10.0], 1.0), [0.0, 1.0, 0.0]).is_none());
        // A gizmo with no normal handle (not a lid) refuses to tilt anything.
        assert!(GizmoScreen::project(&camera(), &sphere([0.0, 0.0, 10.0], 1.0))
            .unwrap()
            .tilted_up([0.0, 1.0, 0.0], (20.0, 0.0))
            .is_none());
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
