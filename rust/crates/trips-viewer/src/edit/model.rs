//! `edits.json`'s data model in Rust: [`Region`], [`EditDocument`], membership tests.
//!
//! Module: `trips_viewer::edit::model`
//! Purpose: the viewer's half of `docs/EDITOR.md` §1. A **line-for-line twin**
//!     of `trippy/edit/model.py` so the two sides read and write the same
//!     `edits.json`: the same region kinds, the same `params` shapes, the same
//!     append-only undo log replayed the same way, and the same world-space
//!     membership arithmetic (in `f64`, like the Python, so the golden test
//!     in `weights.rs` can hold to 1e-6).
//! Invariants:
//!     - A [`Region`] that exists is well formed: [`Region::from_json`]
//!       validates `kind`/`op`/`mix` and the kind's own `params` shape exactly
//!       where `trippy.edit.model.Region.__post_init__` does, and returns the
//!       same class of error.
//!     - [`EditDocument`]'s current state (`regions`/`order`) is always a pure
//!       function of replaying `log[..cursor]` from empty ([`EditDocument::replay`]);
//!       undo/redo are cursor moves, never inverse operations.
//!     - Every region keeps the **raw** `params` JSON it was loaded with and
//!       writes it back verbatim, so a document written by the Python side
//!       round-trips through the viewer byte-identically (Python omits `quat`
//!       when the caller did; the viewer must not invent it on the way out).
//!       Only a params edit made *in the viewer* re-canonicalises them
//!       ([`Params::to_json`]).
//!     - `lid`'s hard membership ([`region_contains`], used by `op = delete`)
//!       is the literal "below the plane, inside the radius" clip, ignoring
//!       `falloff`/`band`; its graded membership ([`region_weight`], used by
//!       `blend`/`fade`) ramps by them. Same split as the Python.
//! Units: world units (COLMAP world frame, `docs/GEOMETRY.md`); `mix` is
//!     dimensionless in `[0, 1]` (0 = pure splat, 1 = pure TRIPS).
//! Related docs: `docs/EDITOR.md` §1; `docs/decisions/ADR-0007-viewer-editing.md`;
//!     `trippy/edit/model.py` (the twin).

use serde_json::{json, Map, Value};

use super::brush::{self, BrushCells};

/// `edits.json`'s `"format"` field. Mirrors `trippy.constants.EDIT_FORMAT`.
pub const EDIT_FORMAT: &str = "trippy-edits-1";

/// The sidecar's filename, next to `bundle.json`.
/// Mirrors `trippy.constants.EDIT_JSON_FILENAME`.
pub const EDITS_FILENAME: &str = "edits.json";

/// The per-point weight before any region applies: 1.0 = pure TRIPS.
/// Mirrors `trippy.constants.EDIT_GATE_DEFAULT_WEIGHT`.
pub const GATE_DEFAULT_WEIGHT: f64 = 1.0;

/// Hex digits in a generated region id (`"r-3f9a1b2c"`).
/// Mirrors `trippy.constants.EDIT_REGION_ID_HEX_LEN`.
pub const REGION_ID_HEX_LEN: usize = 8;

/// The most cells a `brush` region's `regions[]` entry carries inline before
/// [`EditDocument::save`] externalises `cells`/`weights` into an `.npz`
/// sidecar. Mirrors `trippy.constants.EDIT_BRUSH_NPZ_CELL_THRESHOLD`.
pub const EDIT_BRUSH_NPZ_CELL_THRESHOLD: usize = 4096;

/// `"edits_brush_<region id>.npz"` — mirrors
/// `trippy.constants.EDIT_BRUSH_NPZ_FILENAME_FMT`.
#[must_use]
pub fn brush_npz_filename(region_id: &str) -> String {
    format!("edits_brush_{region_id}.npz")
}

/// `"format"` of `tests/fixtures/synthetic/edit_golden/names.json`, matching
/// `trippy.edit.golden.NAMES_FIXTURE_FORMAT`.
pub const NAMES_FIXTURE_FORMAT: &str = "trippy-edit-names-1";

/// The Karekare pool lid's already-fitted geometry (`~/Splats/tools/SURFACE_LID.md` §3,
/// mirrored by `trippy.constants.EDIT_KAREKARE_LID_*`). Seeded by "+ lid" so the
/// numbers never have to be retyped.
pub const KAREKARE_LID: LidParams = LidParams {
    up: [0.012_902_13, -0.952_718_46, -0.303_580_41],
    height: -0.49,
    center: [-0.006_833_48, -0.747_596_95, 3.959_943_45],
    radius: 2.5,
    falloff: 1.0,
    band: 0.05,
};

/// What a region does where it applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Move the running weight to `mix` (last write wins).
    Blend,
    /// Hard removal, upstream of rasterisation.
    Delete,
    /// Multiply the running weight towards `mix` — the soft `delete`.
    Fade,
}

impl Op {
    /// The spelling used in `edits.json`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Blend => "blend",
            Self::Delete => "delete",
            Self::Fade => "fade",
        }
    }

    /// Parse an `op` field.
    ///
    /// # Errors
    /// Returns `Err` listing the accepted spellings when `text` is not one.
    pub fn parse(text: &str) -> Result<Self, String> {
        match text {
            "blend" => Ok(Self::Blend),
            "delete" => Ok(Self::Delete),
            "fade" => Ok(Self::Fade),
            other => Err(format!(
                "Region.op must be one of ('blend', 'delete', 'fade'), got {other:?}"
            )),
        }
    }

    /// Every op, in the order the Inspector shows them.
    pub const ALL: [Self; 3] = [Self::Blend, Self::Fade, Self::Delete];
}

/// Which `params` shape a region carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// An oriented box: `center`, `half_extents`, `quat`.
    Box,
    /// `center`, `radius`.
    Sphere,
    /// The plane clip: `up`, `height`, `center`, `radius`, `falloff`, `band`.
    Lid,
    /// `point_ids` into `points.npz`'s own row order.
    Pointset,
    /// A sparse voxel set painted with the brush tool: `origin`, `cell_size`,
    /// `cells`, optional `weights` (`docs/EDITOR.md` §1 "brush").
    Brush,
}

impl Kind {
    /// The spelling used in `edits.json`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Box => "box",
            Self::Sphere => "sphere",
            Self::Lid => "lid",
            Self::Pointset => "pointset",
            Self::Brush => "brush",
        }
    }

    /// Parse a `kind` field.
    ///
    /// # Errors
    /// Returns `Err` listing the accepted spellings when `text` is not one.
    pub fn parse(text: &str) -> Result<Self, String> {
        match text {
            "box" => Ok(Self::Box),
            "sphere" => Ok(Self::Sphere),
            "lid" => Ok(Self::Lid),
            "pointset" => Ok(Self::Pointset),
            "brush" => Ok(Self::Brush),
            other => Err(format!(
                "Region.kind must be one of ('box', 'sphere', 'lid', 'pointset', 'brush'), \
                 got {other:?}"
            )),
        }
    }
}

/// A `lid` region's six numbers — the same six as `SURFACE_LID.md`'s `--lid-*` flags.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LidParams {
    /// Plane normal, world frame (need not be unit; normalised on use).
    pub up: [f64; 3],
    /// Plane offset along `up`, world units.
    pub height: f64,
    /// Centre of the horizontal radius test, world frame.
    pub center: [f64; 3],
    /// Horizontal radius, world units, > 0.
    pub radius: f64,
    /// Horizontal ramp width beyond `radius`, world units, >= 0.
    pub falloff: f64,
    /// Vertical ramp depth below the plane, world units, >= 0.
    pub band: f64,
}

/// A region's kind-specific geometry, parsed and validated.
#[derive(Debug, Clone, PartialEq)]
pub enum Params {
    /// An oriented box. `quat` is `(w, x, y, z)`; identity is axis aligned.
    Box {
        /// World-frame box centre.
        center: [f64; 3],
        /// Half-size along the box's own local axes, all > 0.
        half_extents: [f64; 3],
        /// Orientation, `(w, x, y, z)`, not necessarily normalised.
        quat: [f64; 4],
    },
    /// A sphere.
    Sphere {
        /// World-frame centre.
        center: [f64; 3],
        /// Radius, world units, > 0.
        radius: f64,
    },
    /// The plane clip.
    Lid(LidParams),
    /// Indices into the TRIPS point cloud's own row order.
    Pointset {
        /// Point indices; may be empty, never negative.
        point_ids: Vec<u32>,
    },
    /// A sparse voxel set. See [`super::brush`] for the painting helpers.
    Brush {
        /// World-frame corner the voxel grid is measured from.
        origin: [f64; 3],
        /// Voxel edge length, world units, > 0.
        cell_size: f64,
        /// The occupied cells and their weights.
        cells: BrushCells,
    },
}

impl Params {
    /// The `kind` this shape belongs to.
    #[must_use]
    pub const fn kind(&self) -> Kind {
        match self {
            Self::Box { .. } => Kind::Box,
            Self::Sphere { .. } => Kind::Sphere,
            Self::Lid(_) => Kind::Lid,
            Self::Pointset { .. } => Kind::Pointset,
            Self::Brush { .. } => Kind::Brush,
        }
    }

    /// The canonical `params` object, written when the viewer itself edits a region.
    ///
    /// Deliberately not used for a region read off disk — see the module's
    /// third invariant.
    #[must_use]
    pub fn to_json(&self) -> Value {
        match self {
            Self::Box {
                center,
                half_extents,
                quat,
            } => json!({
                "center": center.to_vec(),
                "half_extents": half_extents.to_vec(),
                "quat": quat.to_vec(),
            }),
            Self::Sphere { center, radius } => json!({
                "center": center.to_vec(),
                "radius": radius,
            }),
            Self::Lid(l) => json!({
                "up": l.up.to_vec(),
                "height": l.height,
                "center": l.center.to_vec(),
                "radius": l.radius,
                "falloff": l.falloff,
                "band": l.band,
            }),
            Self::Pointset { point_ids } => json!({ "point_ids": point_ids }),
            Self::Brush {
                origin,
                cell_size,
                cells,
            } => {
                let mut params = json!({
                    "origin": origin.to_vec(),
                    "cell_size": cell_size,
                    "cells": cells.cells_json(),
                });
                // An ungraded brush carries no `weights` array at all, exactly
                // as `trippy.edit.model._merge_brush_cells` canonicalises it.
                if let Some(weights) = cells.weights_json() {
                    params["weights"] = weights;
                }
                params
            }
        }
    }
}

// --- validation helpers, matching trippy/edit/model.py's `_as_vec`/`_positive` -------

fn as_vec<const N: usize>(value: Option<&Value>, name: &str) -> Result<[f64; N], String> {
    let array = value
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{name} must be {N} numbers, got {}", show(value)))?;
    if array.len() != N {
        return Err(format!("{name} must be {N} numbers, got {}", show(value)));
    }
    let mut out = [0.0_f64; N];
    for (slot, item) in out.iter_mut().zip(array) {
        let v = item
            .as_f64()
            .ok_or_else(|| format!("{name} must be {N} numbers, got {}", show(value)))?;
        if !v.is_finite() {
            return Err(format!("{name} must be finite, got {}", show(value)));
        }
        *slot = v;
    }
    Ok(out)
}

fn positive(value: Option<&Value>, name: &str) -> Result<f64, String> {
    let v = value
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{name} must be > 0, got {}", show(value)))?;
    if !v.is_finite() || v <= 0.0 {
        return Err(format!("{name} must be > 0, got {}", show(value)));
    }
    Ok(v)
}

fn non_negative(value: Option<&Value>, name: &str) -> Result<f64, String> {
    let v = value
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{name} must be >= 0, got {}", show(value)))?;
    if !v.is_finite() || v < 0.0 {
        return Err(format!("{name} must be >= 0, got {}", show(value)));
    }
    Ok(v)
}

fn show(value: Option<&Value>) -> String {
    value.map_or_else(|| "None".to_owned(), ToString::to_string)
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

impl Params {
    /// Parse and validate a `params` object for `kind`.
    ///
    /// # Errors
    /// Returns `Err` with the same message shape `trippy.edit.model`'s own
    /// validators raise, so a malformed file reads the same on both sides.
    pub fn from_json(kind: Kind, params: &Value) -> Result<Self, String> {
        let get = |key: &str| params.get(key);
        match kind {
            Kind::Box => {
                let center = as_vec::<3>(get("center"), "box.center")?;
                let half_extents = as_vec::<3>(get("half_extents"), "box.half_extents")?;
                if half_extents.iter().any(|h| *h <= 0.0) {
                    return Err(format!(
                        "box.half_extents must all be > 0, got {half_extents:?}"
                    ));
                }
                let quat = match get("quat") {
                    Some(v) => as_vec::<4>(Some(v), "box.quat")?,
                    None => [1.0, 0.0, 0.0, 0.0],
                };
                if quat.iter().map(|q| q * q).sum::<f64>() < 1e-18 {
                    return Err(format!("box.quat must not be ~zero, got {quat:?}"));
                }
                Ok(Self::Box {
                    center,
                    half_extents,
                    quat,
                })
            }
            Kind::Sphere => Ok(Self::Sphere {
                center: as_vec::<3>(get("center"), "sphere.center")?,
                radius: positive(get("radius"), "sphere.radius")?,
            }),
            Kind::Lid => {
                let up = as_vec::<3>(get("up"), "lid.up")?;
                if dot3(up, up) < 1e-18 {
                    return Err(format!("lid.up must not be ~zero, got {up:?}"));
                }
                let height = get("height")
                    .and_then(Value::as_f64)
                    .filter(|h| h.is_finite())
                    .ok_or_else(|| {
                        format!("lid.height must be finite, got {}", show(get("height")))
                    })?;
                Ok(Self::Lid(LidParams {
                    up,
                    height,
                    center: as_vec::<3>(get("center"), "lid.center")?,
                    radius: positive(get("radius"), "lid.radius")?,
                    falloff: non_negative(get("falloff"), "lid.falloff")?,
                    band: non_negative(get("band"), "lid.band")?,
                }))
            }
            Kind::Pointset => {
                let ids = get("point_ids")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        format!(
                            "pointset.point_ids must be a list, got {}",
                            show(get("point_ids"))
                        )
                    })?;
                let mut point_ids = Vec::with_capacity(ids.len());
                for item in ids {
                    let id = item.as_i64().ok_or_else(|| {
                        "pointset.point_ids must be a flat list of integers".to_owned()
                    })?;
                    if id < 0 {
                        return Err("pointset.point_ids must all be >= 0".to_owned());
                    }
                    point_ids.push(u32::try_from(id).map_err(|_| {
                        format!("pointset.point_ids entry {id} does not fit a point index")
                    })?);
                }
                Ok(Self::Pointset { point_ids })
            }
            Kind::Brush => Ok(Self::Brush {
                origin: as_vec::<3>(get("origin"), "brush.origin")?,
                cell_size: positive(get("cell_size"), "brush.cell_size")?,
                cells: BrushCells::from_json(get("cells"), get("weights"))?,
            }),
        }
    }
}

// --- Region -------------------------------------------------------------------------

/// One `edits.json` region — `docs/EDITOR.md` §1's `Region` table.
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    /// Stable identity, never reused even after removal (the undo log keys off it).
    pub id: String,
    /// Shown in the Regions panel.
    pub name: String,
    /// Which `params` shape applies.
    pub kind: Kind,
    /// The parsed geometry.
    pub params: Params,
    /// The `params` object exactly as it appeared in the file / undo log, so a
    /// document round-trips unchanged. Rewritten only by [`Region::set_params`].
    pub raw_params: Value,
    /// `[0, 1]`; 0 = pure splat, 1 = pure TRIPS. Ignored by `op = delete`.
    pub mix: f64,
    /// What this region does where it applies.
    pub op: Op,
    /// Soft "off" without deleting the region.
    pub enabled: bool,
    /// `{"tool": str, ...}`, or `Some(Value::Null)` for an explicit `null`, or
    /// `None` when the key was absent entirely.
    ///
    /// Provenance only: which tool made this region and with what prompt
    /// (`trippy.edit.model.Region.source`, `docs/EDITOR.md` §1 "Named
    /// regions"). It never affects membership or composition — the Named
    /// Objects panel groups by it and nothing else reads it.
    ///
    /// The three-state representation exists so a document round-trips
    /// byte-identically: the Python writes `"source": null` for a
    /// hand-authored region and omits nothing, while a region written before
    /// the field existed has no key at all, and the viewer must write back
    /// what it read (the module's third invariant, extended to this field).
    pub source: Option<Value>,
}

impl Region {
    /// Build a region from parsed geometry, canonicalising its `params` JSON.
    #[must_use]
    pub fn new(id: String, name: String, params: Params, mix: f64, op: Op) -> Self {
        Self {
            id,
            name,
            kind: params.kind(),
            raw_params: params.to_json(),
            params,
            mix,
            op,
            enabled: true,
            source: None,
        }
    }

    /// The same region with a `source` block recording which tool made it.
    ///
    /// `tool` is the label [`auto_region_name`] uses (`"brush"`, `"click"`,
    /// `"shade-clouds"`, `"sam-box"`, ...); `extra` is that tool's own
    /// parameters, e.g. the radius a stroke was painted at.
    #[must_use]
    pub fn with_source(mut self, tool: &str, extra: Value) -> Self {
        let mut source = json!({ "tool": tool });
        if let (Some(target), Some(items)) = (source.as_object_mut(), extra.as_object()) {
            for (key, value) in items {
                target.insert(key.clone(), value.clone());
            }
        }
        self.source = Some(source);
        self
    }

    /// The `source.tool` label, if this region records one.
    #[must_use]
    pub fn source_tool(&self) -> Option<&str> {
        self.source.as_ref()?.get("tool")?.as_str()
    }

    /// Replace the geometry, rewriting `raw_params` canonically.
    pub fn set_params(&mut self, params: Params) {
        self.kind = params.kind();
        self.raw_params = params.to_json();
        self.params = params;
    }

    /// This region as a `regions[]` entry of `edits.json`.
    #[must_use]
    pub fn to_json(&self) -> Value {
        let mut doc = json!({
            "id": self.id,
            "name": self.name,
            "kind": self.kind.as_str(),
            "enabled": self.enabled,
            "mix": self.mix,
            "op": self.op.as_str(),
            "params": self.raw_params,
        });
        // Written back only when the region carried one, so a file with no
        // `source` key round-trips without gaining one.
        if let (Some(target), Some(source)) = (doc.as_object_mut(), self.source.as_ref()) {
            target.insert("source".to_owned(), source.clone());
        }
        doc
    }

    /// Inverse of [`Region::to_json`], with `trippy.edit.model.Region`'s own validation.
    ///
    /// # Errors
    /// Returns `Err` on a missing `id`/`kind`, an unknown `kind`/`op`, a `mix`
    /// outside `[0, 1]`, or a `params` object the kind rejects.
    pub fn from_json(doc: &Value) -> Result<Self, String> {
        let id = doc
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| "Region.id must be non-empty".to_owned())?;
        if id.is_empty() {
            return Err("Region.id must be non-empty".to_owned());
        }
        let kind = Kind::parse(
            doc.get("kind")
                .and_then(Value::as_str)
                .ok_or_else(|| "Region.kind is missing".to_owned())?,
        )?;
        let op = Op::parse(doc.get("op").and_then(Value::as_str).unwrap_or("blend"))?;
        let mix = doc.get("mix").and_then(Value::as_f64).unwrap_or(1.0);
        if !(0.0..=1.0).contains(&mix) {
            return Err(format!("Region.mix must be in [0, 1], got {mix}"));
        }
        let raw_params = doc.get("params").cloned().unwrap_or_else(|| json!({}));
        let params = Params::from_json(kind, &raw_params)?;
        let source = doc.get("source").cloned();
        if let Some(value) = &source {
            if !value.is_object() && !value.is_null() {
                return Err(format!("Region.source must be a dict or None, got {value}"));
            }
        }
        Ok(Self {
            id: id.to_owned(),
            name: doc
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned(),
            kind,
            params,
            raw_params,
            mix,
            op,
            enabled: doc.get("enabled").and_then(Value::as_bool).unwrap_or(true),
            source,
        })
    }

    /// A short "box", "sphere", "lid", "pts" tag for the Regions list.
    #[must_use]
    pub const fn short_kind(&self) -> &'static str {
        match self.kind {
            Kind::Box => "box",
            Kind::Sphere => "sphere",
            Kind::Lid => "lid",
            Kind::Pointset => "pts",
            Kind::Brush => "brush",
        }
    }
}

// --- membership tests, the numpy twins ----------------------------------------------

/// Unit quaternion `(w, x, y, z)` -> row-major `3x3` rotation.
///
/// A generic object-space rotation for an oriented box — deliberately **not**
/// `trippy.geom.xform_a.qvec2R`'s COLMAP world->camera convention. `q` need not
/// be normalised; this normalises it. A ~zero-norm quaternion returns identity,
/// which cannot happen through [`Params::from_json`] (it rejects one first).
#[must_use]
pub fn quat_to_rotmat(q: [f64; 4]) -> [[f64; 3]; 3] {
    let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < 1e-12 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let (w, x, y, z) = (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// Hard-edge oriented-box membership for one point: `1.0` inside, `0.0` outside.
///
/// `local = R^T (p - center)`; inside iff `|local_i| <= half_extents_i` on every
/// axis. Mirrors `trippy.edit.model.box_membership` (whose `(xyz - c) @ R` is
/// the same `R^T v`).
#[must_use]
pub fn box_membership(p: [f64; 3], center: [f64; 3], half_extents: [f64; 3], quat: [f64; 4]) -> f64 {
    let rot = quat_to_rotmat(quat);
    let rel = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
    for axis in 0..3 {
        // Column `axis` of R dotted with `rel` == row `axis` of R^T dotted with `rel`.
        let local = rel[0].mul_add(
            rot[0][axis],
            rel[1].mul_add(rot[1][axis], rel[2] * rot[2][axis]),
        );
        if local.abs() > half_extents[axis] {
            return 0.0;
        }
    }
    1.0
}

/// Hard-edge sphere membership: `1.0` when `||p - center|| <= radius`, else `0.0`.
#[must_use]
pub fn sphere_membership(p: [f64; 3], center: [f64; 3], radius: f64) -> f64 {
    let d = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
    f64::from(dot3(d, d) <= radius * radius)
}

/// The lid's two views of one point: `(inside, weight)`.
///
/// `inside` is the LITERAL hard clip `h(x) <= 0 && rho(x) <= radius`, ignoring
/// `falloff`/`band` — what `op = delete` uses. `weight` is the falloff-graded
/// `vertical * horizontal` ramp `op = blend`/`fade` use. Mirrors
/// `trippy.edit.model.lid_membership` term for term; see that docstring for why
/// the ramp is mirrored across the plane rather than copied from
/// `SURFACE_LID.md` verbatim.
#[must_use]
pub fn lid_membership(p: [f64; 3], lid: &LidParams) -> (bool, f64) {
    let norm = dot3(lid.up, lid.up).sqrt();
    if norm < 1e-12 {
        // Unreachable through `Params::from_json`, which rejects a ~zero `up`.
        return (false, 0.0);
    }
    let up = [lid.up[0] / norm, lid.up[1] / norm, lid.up[2] / norm];
    let signed = dot3(p, up) - lid.height;
    let rel = [
        p[0] - lid.center[0],
        p[1] - lid.center[1],
        p[2] - lid.center[2],
    ];
    let along = dot3(rel, up);
    let perp = [
        along.mul_add(-up[0], rel[0]),
        along.mul_add(-up[1], rel[1]),
        along.mul_add(-up[2], rel[2]),
    ];
    let rho = dot3(perp, perp).sqrt();

    let inside = signed <= 0.0 && rho <= lid.radius;
    let vertical = if lid.band > 0.0 {
        (-signed / lid.band).clamp(0.0, 1.0)
    } else {
        f64::from(signed <= 0.0)
    };
    let horizontal = if lid.falloff > 0.0 {
        ((lid.radius + lid.falloff - rho) / lid.falloff).clamp(0.0, 1.0)
    } else {
        f64::from(rho <= lid.radius)
    };
    (inside, vertical * horizontal)
}

/// Per-point membership weight in `[0, 1]` for one region: hard 0/1 for
/// box/sphere/pointset, graded for lid.
///
/// `index` is the point's row in the cloud `point_ids` indexes (only `pointset`
/// reads it). Mirrors `trippy.edit.model.region_weight`.
#[must_use]
pub fn region_weight(region: &Region, index: usize, p: [f64; 3]) -> f64 {
    match &region.params {
        Params::Box {
            center,
            half_extents,
            quat,
        } => box_membership(p, *center, *half_extents, *quat),
        Params::Sphere { center, radius } => sphere_membership(p, *center, *radius),
        Params::Lid(lid) => lid_membership(p, lid).1,
        Params::Pointset { point_ids } => {
            // A linear scan would be O(N x |ids|); callers that need speed build
            // a lookup once — see `weights::PointsetIndex`.
            f64::from(
                u32::try_from(index)
                    .ok()
                    .is_some_and(|i| point_ids.contains(&i)),
            )
        }
        Params::Brush {
            origin,
            cell_size,
            cells,
        } => brush::membership(cells, *origin, *cell_size, p),
    }
}

/// Hard boolean membership: the exact footprint `op = delete` removes.
///
/// Identical to `region_weight(..) > 0` for every kind except `lid`, where this
/// is the literal clip. Mirrors `trippy.edit.model.region_contains`.
#[must_use]
pub fn region_contains(region: &Region, index: usize, p: [f64; 3]) -> bool {
    match &region.params {
        Params::Lid(lid) => lid_membership(p, lid).0,
        _ => region_weight(region, index, p) > 0.0,
    }
}

// --- EditDocument -------------------------------------------------------------------

/// One undo-log entry type, spelled exactly as `trippy.edit.model._ENTRY_TYPES`.
const ENTRY_TYPES: &str = "('add_region', 'remove_region', 'update_region', 'reorder')";

/// `edits.json`'s full document: regions, paint order, and the append-only undo log.
///
/// Mutate through [`Self::add_region`] / [`Self::remove_region`] /
/// [`Self::update_region`] / [`Self::reorder`]; `regions`/`order` are derived and
/// must never be written directly.
#[derive(Debug, Clone, PartialEq)]
pub struct EditDocument {
    /// Always [`EDIT_FORMAT`].
    pub format: String,
    /// Cross-checked against `bundle.json`'s `"format"`; `""` skips the check.
    pub bundle_format: String,
    /// Append-only diff list. Truncated to `cursor` and appended to by every mutation.
    pub log: Vec<Value>,
    /// Index into `log`; undo/redo move it by one.
    pub cursor: usize,
    /// Derived, in paint order.
    regions: Vec<Region>,
    /// Derived, region ids in paint order (later wins on overlap).
    order: Vec<String>,
}

impl Default for EditDocument {
    fn default() -> Self {
        Self::new(String::new())
    }
}

impl EditDocument {
    /// An empty document (no regions, no history).
    #[must_use]
    pub fn new(bundle_format: String) -> Self {
        Self {
            format: EDIT_FORMAT.to_owned(),
            bundle_format,
            log: Vec::new(),
            cursor: 0,
            regions: Vec::new(),
            order: Vec::new(),
        }
    }

    /// The regions in paint order; later entries win on overlap.
    #[must_use]
    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    /// The region ids in paint order.
    #[must_use]
    pub fn order(&self) -> &[String] {
        &self.order
    }

    /// One region by id, or `None`.
    #[must_use]
    pub fn region(&self, id: &str) -> Option<&Region> {
        self.regions.iter().find(|r| r.id == id)
    }

    /// Whether [`Self::undo`] would do anything.
    #[must_use]
    pub const fn can_undo(&self) -> bool {
        self.cursor > 0
    }

    /// Whether [`Self::redo`] would do anything.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        self.cursor < self.log.len()
    }

    /// Rebuild `regions`/`order` by replaying `log[..cursor]` from empty.
    ///
    /// # Errors
    /// Returns `Err` when `cursor` is out of range, an entry has an unknown
    /// `type`, a `reorder` entry is not an exact permutation, or a logged region
    /// fails [`Region::from_json`] — i.e. exactly where the Python raises.
    fn replay(&mut self) -> Result<(), String> {
        if self.cursor > self.log.len() {
            return Err(format!(
                "undo cursor {} out of range for a log of length {}",
                self.cursor,
                self.log.len()
            ));
        }
        let mut regions: Vec<Region> = Vec::new();
        let mut order: Vec<String> = Vec::new();
        for entry in &self.log[..self.cursor] {
            match entry.get("type").and_then(Value::as_str) {
                Some("add_region") => {
                    let region = Region::from_json(
                        entry
                            .get("region")
                            .ok_or_else(|| "'add_region' entry has no 'region'".to_owned())?,
                    )?;
                    let id = region.id.clone();
                    regions.retain(|r| r.id != id);
                    regions.push(region);
                    let index = entry.get("index").and_then(Value::as_i64);
                    match index {
                        Some(i) if (i.max(0) as usize) < order.len() => {
                            order.insert(i.max(0) as usize, id);
                        }
                        _ => order.push(id),
                    }
                }
                Some("remove_region") => {
                    let id = entry
                        .get("id")
                        .and_then(Value::as_str)
                        .ok_or_else(|| "'remove_region' entry has no 'id'".to_owned())?;
                    regions.retain(|r| r.id != id);
                    order.retain(|o| o != id);
                }
                Some("update_region") => {
                    let id = entry
                        .get("id")
                        .and_then(Value::as_str)
                        .ok_or_else(|| "'update_region' entry has no 'id'".to_owned())?;
                    if let Some(slot) = regions.iter().position(|r| r.id == id) {
                        let mut doc = regions[slot].to_json();
                        if let (Some(target), Some(changes)) = (
                            doc.as_object_mut(),
                            entry.get("changes").and_then(Value::as_object),
                        ) {
                            for (key, value) in changes {
                                target.insert(key.clone(), value.clone());
                            }
                        }
                        regions[slot] = Region::from_json(&doc)?;
                    }
                }
                Some("reorder") => {
                    let new_order: Vec<String> = entry
                        .get("order")
                        .and_then(Value::as_array)
                        .ok_or_else(|| "'reorder' entry has no 'order'".to_owned())?
                        .iter()
                        .filter_map(|v| v.as_str().map(ToOwned::to_owned))
                        .collect();
                    let mut a = new_order.clone();
                    let mut b = order.clone();
                    a.sort_unstable();
                    b.sort_unstable();
                    if a != b {
                        return Err(
                            "'reorder' entry must permute the current region ids exactly".to_owned()
                        );
                    }
                    order = new_order;
                }
                other => {
                    return Err(format!(
                        "unknown undo-log entry type {}; expected one of {ENTRY_TYPES}",
                        other.map_or_else(|| "None".to_owned(), |t| format!("{t:?}"))
                    ))
                }
            }
        }
        self.regions = order
            .iter()
            .filter_map(|id| regions.iter().find(|r| &r.id == id).cloned())
            .collect();
        self.order = order;
        Ok(())
    }

    /// Truncate the log at the cursor, append `entry`, and replay.
    ///
    /// # Errors
    /// Returns `Err` if the resulting log does not replay (the document is left
    /// with the entry removed, so a rejected edit cannot corrupt the session).
    fn push(&mut self, entry: Value) -> Result<(), String> {
        self.log.truncate(self.cursor);
        self.log.push(entry);
        self.cursor = self.log.len();
        if let Err(e) = self.replay() {
            self.log.pop();
            self.cursor = self.log.len();
            // Replaying a prefix that replayed a moment ago cannot fail.
            let _ = self.replay();
            return Err(e);
        }
        Ok(())
    }

    /// Append (or insert at `index`) a new region.
    ///
    /// # Errors
    /// Returns `Err` if the region does not survive a round trip through
    /// [`Region::from_json`] (which the log replay performs).
    pub fn add_region(&mut self, region: &Region, index: Option<usize>) -> Result<(), String> {
        self.push(json!({
            "type": "add_region",
            "region": region.to_json(),
            "index": index,
        }))
    }

    /// Remove a region by id. A no-op push when it is already absent, exactly as
    /// the Python does, so undo still records the click.
    ///
    /// # Errors
    /// Returns `Err` only if the log fails to replay.
    pub fn remove_region(&mut self, region_id: &str) -> Result<(), String> {
        self.push(json!({ "type": "remove_region", "id": region_id }))
    }

    /// Edit one or more fields of an existing region (`mix`, `enabled`, `op`,
    /// `name`, `params`, ...).
    ///
    /// # Errors
    /// Returns `Err` when no such region exists, or when the change makes the
    /// region invalid.
    pub fn update_region(&mut self, region_id: &str, changes: Value) -> Result<(), String> {
        if self.region(region_id).is_none() {
            return Err(format!("no region {region_id:?} to update"));
        }
        self.push(json!({
            "type": "update_region",
            "id": region_id,
            "changes": changes,
        }))
    }

    /// [`Self::update_region`], optionally REPLACING the previous log entry.
    ///
    /// One continuous gesture — a gizmo drag, a brush stroke — has to be one
    /// undo step, but it also has to show its result while it happens, which
    /// means writing to the document on every frame of the drag. `coalesce =
    /// true` truncates the immediately preceding `update_region` entry for the
    /// same region before appending this one, so a 200-frame drag leaves ONE
    /// entry in `log` and `Cmd-Z` takes the whole gesture back.
    ///
    /// The file format is untouched: the result is an ordinary `update_region`
    /// entry that `trippy.edit.model.EditDocument` replays like any other. The
    /// caller owns the gesture boundary (it passes `false` for the first frame
    /// of a drag and `true` after), because only the caller knows when the
    /// button went down.
    ///
    /// # Errors
    /// As [`Self::update_region`].
    pub fn update_region_coalesced(
        &mut self,
        region_id: &str,
        changes: Value,
        coalesce: bool,
    ) -> Result<(), String> {
        if coalesce {
            self.drop_trailing("update_region", region_id);
        }
        self.update_region(region_id, changes)
    }

    /// [`Self::add_region`], optionally REPLACING the previous log entry.
    ///
    /// The same gesture rule [`Self::update_region_coalesced`] implements, for
    /// the case where the gesture CREATED the region: the first sample of a
    /// brush stroke on a new region adds it, and every later sample of the same
    /// stroke replaces that entry with the fuller region, so the whole stroke
    /// is one undo step rather than one plus a hundred.
    ///
    /// # Errors
    /// As [`Self::add_region`].
    pub fn add_region_coalesced(
        &mut self,
        region: &Region,
        index: Option<usize>,
        coalesce: bool,
    ) -> Result<(), String> {
        if coalesce {
            self.drop_trailing("add_region", &region.id);
        }
        self.add_region(region, index)
    }

    /// Drop the entry at `cursor - 1` when it is `entry_type` for `region_id`.
    ///
    /// Only ever at the very end of the log (`cursor == log.len()`): coalescing
    /// into an entry the user has already undone past would rewrite history
    /// they are looking at.
    fn drop_trailing(&mut self, entry_type: &str, region_id: &str) {
        if self.cursor == 0 || self.cursor != self.log.len() {
            return;
        }
        let previous = &self.log[self.cursor - 1];
        if previous.get("type").and_then(Value::as_str) != Some(entry_type) {
            return;
        }
        let previous_id = match entry_type {
            "add_region" => previous.pointer("/region/id").and_then(Value::as_str),
            _ => previous.get("id").and_then(Value::as_str),
        };
        if previous_id != Some(region_id) {
            return;
        }
        self.log.truncate(self.cursor - 1);
        self.cursor = self.log.len();
        // Dropping a trailing entry cannot make the surviving prefix unreplayable.
        let _ = self.replay();
    }

    /// Set the paint order to an exact permutation of the current region ids.
    ///
    /// # Errors
    /// Returns `Err` when `new_order` is not a permutation of the current ids.
    pub fn reorder(&mut self, new_order: Vec<String>) -> Result<(), String> {
        self.push(json!({ "type": "reorder", "order": new_order }))
    }

    /// Move the cursor back one entry. `false` (a no-op) when already at the start.
    pub fn undo(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }
        self.cursor -= 1;
        // A prefix of a log that already replayed cannot fail to replay.
        let _ = self.replay();
        true
    }

    /// Move the cursor forward one entry. `false` when already at the end.
    pub fn redo(&mut self) -> bool {
        if self.cursor >= self.log.len() {
            return false;
        }
        self.cursor += 1;
        let _ = self.replay();
        true
    }

    /// Cheap consistency checks beyond what construction already guarantees.
    ///
    /// # Errors
    /// Returns `Err` when `format` is wrong, `bundle_format` disagrees with
    /// `expected_bundle_format`, ids are duplicated, or `order` and the region
    /// id set differ. Mirrors `trippy.edit.model.EditDocument.validate`.
    pub fn validate(&self, expected_bundle_format: Option<&str>) -> Result<(), String> {
        if self.format != EDIT_FORMAT {
            return Err(format!(
                "edits.json format {:?} != expected {EDIT_FORMAT:?}",
                self.format
            ));
        }
        if let Some(expected) = expected_bundle_format {
            if !expected.is_empty() && !self.bundle_format.is_empty() && self.bundle_format != expected {
                return Err(format!(
                    "edits.json bundle_format {:?} does not match bundle.json format {expected:?}",
                    self.bundle_format
                ));
            }
        }
        let mut ids: Vec<&str> = self.regions.iter().map(|r| r.id.as_str()).collect();
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() != count {
            return Err("edits.json has duplicate region ids".to_owned());
        }
        let mut order: Vec<&str> = self.order.iter().map(String::as_str).collect();
        order.sort_unstable();
        if order != ids {
            return Err("edits.json 'order' does not match its region ids".to_owned());
        }
        Ok(())
    }

    /// The full `edits.json` document.
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "format": self.format,
            "bundle_format": self.bundle_format,
            "regions": self.regions.iter().map(Region::to_json).collect::<Vec<_>>(),
            "order": self.order,
            "undo_stack": { "cursor": self.cursor, "log": self.log },
        })
    }

    /// Inverse of [`Self::to_json`], cross-checking `order` against the replayed log.
    ///
    /// # Errors
    /// Returns `Err` on an unsupported `format`, a log that will not replay, or
    /// a saved `order` that disagrees with the replay — exactly the three the
    /// Python raises on.
    pub fn from_json(doc: &Value) -> Result<Self, String> {
        let format = doc
            .get("format")
            .and_then(Value::as_str)
            .unwrap_or(EDIT_FORMAT);
        if format != EDIT_FORMAT {
            return Err(format!(
                "unsupported edits.json format {format:?}, expected {EDIT_FORMAT:?}"
            ));
        }
        let empty = Map::new();
        let undo_stack = doc
            .get("undo_stack")
            .and_then(Value::as_object)
            .unwrap_or(&empty);
        let log: Vec<Value> = undo_stack
            .get("log")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let cursor = undo_stack
            .get("cursor")
            .and_then(Value::as_u64)
            .map_or(log.len(), |c| usize::try_from(c).unwrap_or(log.len()));
        let mut edits = Self {
            format: format.to_owned(),
            bundle_format: doc
                .get("bundle_format")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned(),
            log,
            cursor,
            regions: Vec::new(),
            order: Vec::new(),
        };
        edits.replay()?;
        if let Some(saved) = doc.get("order").and_then(Value::as_array) {
            let saved: Vec<&str> = saved.iter().filter_map(Value::as_str).collect();
            if saved != edits.order.iter().map(String::as_str).collect::<Vec<_>>() {
                return Err(
                    "edits.json 'order' does not match replaying 'undo_stack.log' up to 'cursor'"
                        .to_owned(),
                );
            }
        }
        Ok(edits)
    }

    /// Read and validate an `edits.json` file.
    ///
    /// # Errors
    /// Returns `Err` when the file cannot be read or does not parse.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let doc: Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_json(&doc).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Write this document to `path`, creating parents.
    ///
    /// The trailing newline and two-space indent match `EditDocument.save` in
    /// Python, so a file the viewer writes and one `trippy edits` writes differ
    /// only where their content differs. A `brush` region above
    /// [`EDIT_BRUSH_NPZ_CELL_THRESHOLD`] cells has its `cells`/`weights`
    /// externalised into an `.npz` sidecar next to `path`
    /// ([`externalize_brush`]) in the WRITTEN copy only — `self.regions`/
    /// `self.log` (and this method's own `self.to_json()` before the
    /// externalisation runs) stay fully literal, exactly as `trippy.edit.
    /// model.EditDocument.save`'s own docstring describes. Neither loader
    /// reads the sidecar back (both replay `undo_stack.log`, never
    /// externalised), so this only matters to an external reader of the
    /// materialised `regions[]` array.
    ///
    /// # Errors
    /// Returns `Err` when the directory cannot be created, a sidecar cannot be
    /// written, or the write fails.
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        let mut doc = self.to_json();
        if let Some(regions) = doc.get_mut("regions").and_then(Value::as_array_mut) {
            for region in regions.iter_mut() {
                *region = externalize_brush(region, parent)?;
            }
        }
        let text =
            serde_json::to_string_pretty(&doc).map_err(|e| format!("serialising edits.json: {e}"))?;
        std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// A `brush` region's `regions[]` entry, `cells`/`weights` moved to an `.npz`
/// sidecar in `out_dir` if large — the Rust twin of `trippy.edit.model.
/// EditDocument._externalize_brush`. Only ever applied to the WRITTEN copy of
/// a region (see [`EditDocument::save`]); a region below
/// [`EDIT_BRUSH_NPZ_CELL_THRESHOLD`] cells (or not a brush at all) is returned
/// unchanged.
///
/// # Errors
/// Returns `Err` when the sidecar cannot be written, or when `cells`/
/// `weights` are shaped in a way [`BrushCells::from_json`] would also refuse
/// (this function is only ever handed a `Region` this crate itself produced,
/// so that should not happen in practice).
fn externalize_brush(region_json: &Value, out_dir: &std::path::Path) -> Result<Value, String> {
    if region_json.get("kind").and_then(Value::as_str) != Some("brush") {
        return Ok(region_json.clone());
    }
    let empty = Vec::new();
    let params = region_json.get("params").cloned().unwrap_or_else(|| json!({}));
    let cells = params
        .get("cells")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    if cells.len() <= EDIT_BRUSH_NPZ_CELL_THRESHOLD {
        return Ok(region_json.clone());
    }

    let mut cells_i32: Vec<[i32; 3]> = Vec::with_capacity(cells.len());
    for row in cells {
        let triplet = row
            .as_array()
            .filter(|t| t.len() == 3)
            .ok_or_else(|| "brush.cells must be a list of [i, j, k] triplets".to_owned())?;
        let mut cell = [0_i32; 3];
        for (slot, item) in cell.iter_mut().zip(triplet) {
            let v = item
                .as_i64()
                .ok_or_else(|| "brush.cells must be a list of [i, j, k] triplets".to_owned())?;
            *slot = i32::try_from(v)
                .map_err(|_| format!("brush.cells must fit in a signed int32, got {v}"))?;
        }
        cells_i32.push(cell);
    }
    let weights: Option<Vec<f32>> = match params.get("weights") {
        Some(Value::Array(array)) => {
            #[allow(clippy::cast_possible_truncation)]
            let out = array.iter().map(|v| v.as_f64().unwrap_or(1.0) as f32).collect();
            Some(out)
        }
        _ => None,
    };

    let id = region_json.get("id").and_then(Value::as_str).unwrap_or("");
    let filename = brush_npz_filename(id);
    super::npz_write::write_brush_npz(&out_dir.join(&filename), &cells_i32, weights.as_deref())?;

    let mut new_params = Map::new();
    if let Value::Object(map) = &params {
        for (key, value) in map {
            if key != "cells" && key != "weights" {
                new_params.insert(key.clone(), value.clone());
            }
        }
    }
    new_params.insert("cells_npz".to_owned(), json!(filename));
    new_params.insert("n_cells".to_owned(), json!(cells_i32.len()));

    let mut out = region_json.clone();
    if let Some(object) = out.as_object_mut() {
        object.insert("params".to_owned(), Value::Object(new_params));
    }
    Ok(out)
}

/// The counter a name ends with, if it ends with one: `"click-12"` -> `Some(12)`.
///
/// The Python is `re.search(r"-(\d+)$", name)`, whose leftmost match with an
/// end anchor is exactly "the trailing run of digits, if a hyphen precedes it".
/// One deliberate narrowing: Python's `\d` also matches non-ASCII digits, and
/// this only matches ASCII. Every name either side GENERATES is ASCII, and a
/// hand-typed Devanagari counter is not a case either side promises to number.
/// A counter too large for `u64` saturates rather than wrapping.
fn trailing_counter(name: &str) -> Option<u64> {
    let digits: String = name
        .chars()
        .rev()
        .take_while(char::is_ascii_digit)
        .collect::<Vec<char>>()
        .into_iter()
        .rev()
        .collect();
    if digits.is_empty() {
        return None;
    }
    let head = &name[..name.len() - digits.len()];
    if !head.ends_with('-') {
        return None;
    }
    Some(digits.parse::<u64>().unwrap_or(u64::MAX))
}

/// A tool-authored region's auto name, `"<tool>[-<detail>]-<n>"`.
///
/// The twin of `trippy.edit.model.auto_region_name` (`docs/EDITOR.md` §1 "Named
/// regions"), pinned by `tests/fixtures/synthetic/edit_golden/names.json`: `n`
/// is one more than the highest `-<digits>` suffix among `existing_names`
/// **from any tool**, so a session's regions read as one continuously numbered
/// list in the Named Objects panel no matter which tool made each one — and the
/// rule is stateless, so it self-heals after a rename, a removal or an undo.
///
/// # Arguments
/// - `existing_names`: the document's current region names.
/// - `tool`: the tool's label, e.g. `"brush"`, `"click"`, `"shade-clouds"`,
///   `"sam-box"`.
/// - `detail`: an optional slug between the tool and the counter (a view's file
///   stem, for SAM).
#[must_use]
pub fn auto_region_name<'a, I>(existing_names: I, tool: &str, detail: Option<&str>) -> String
where
    I: IntoIterator<Item = &'a str>,
{
    let highest = existing_names
        .into_iter()
        .filter_map(trailing_counter)
        .max()
        .unwrap_or(0);
    let label = detail.map_or_else(|| tool.to_owned(), |d| format!("{tool}-{d}"));
    format!("{label}-{}", highest.saturating_add(1))
}

/// [`auto_region_name`] over a document's own regions.
#[must_use]
pub fn auto_name_for(doc: &EditDocument, tool: &str, detail: Option<&str>) -> String {
    auto_region_name(doc.regions().iter().map(|r| r.name.as_str()), tool, detail)
}

/// A fresh, never-reused region id, `"r-<hex>"`.
///
/// The Python uses `uuid4().hex[:8]`; there is no uuid crate in this graph, so
/// this hashes a monotonically increasing counter mixed with the system clock.
/// Uniqueness is what matters (undo keys off the id), not the derivation.
#[must_use]
pub fn new_region_id() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut hasher = DefaultHasher::new();
    n.hash(&mut hasher);
    #[cfg(not(target_family = "wasm"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
            .hash(&mut hasher);
    }
    format!("r-{:0width$x}", hasher.finish(), width = REGION_ID_HEX_LEN)
        .chars()
        .take(2 + REGION_ID_HEX_LEN)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_region(id: &str, mix: f64, op: Op) -> Region {
        Region::new(
            id.to_owned(),
            format!("box {id}"),
            Params::Box {
                center: [0.0, 0.0, 0.0],
                half_extents: [1.0, 1.0, 1.0],
                quat: [1.0, 0.0, 0.0, 0.0],
            },
            mix,
            op,
        )
    }

    #[test]
    fn an_identity_quaternion_box_is_exactly_axis_aligned() {
        let c = [0.5, -1.0, 2.0];
        let h = [1.0, 2.0, 0.5];
        let q = [1.0, 0.0, 0.0, 0.0];
        // Corners of the axis-aligned box are inside; a hair outside is not.
        assert_eq!(box_membership([1.5, 1.0, 2.5], c, h, q), 1.0);
        assert_eq!(box_membership([1.5001, 1.0, 2.5], c, h, q), 0.0);
        assert_eq!(box_membership([0.5, -1.0, 2.0], c, h, q), 1.0);
    }

    #[test]
    fn a_rotated_box_accepts_points_an_axis_aligned_one_would_reject() {
        // 45 deg about +Z (half-angle 22.5): the box's own X axis runs diagonally.
        let half = std::f64::consts::FRAC_PI_8;
        let q = [half.cos(), 0.0, 0.0, half.sin()];
        let c = [0.0, 0.0, 0.0];
        let h = [2.0, 0.2, 0.2];
        // (1.4, 1.4, 0) is 1.98 along the rotated X axis: inside.
        assert_eq!(box_membership([1.4, 1.4, 0.0], c, h, q), 1.0);
        // The same point is outside the axis-aligned box (|y| = 1.4 > 0.2).
        assert_eq!(box_membership([1.4, 1.4, 0.0], c, h, [1.0, 0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn the_lids_hard_clip_ignores_falloff_and_band() {
        let lid = LidParams {
            up: [0.0, -1.0, 0.0],
            height: 0.0,
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
            falloff: 1.0,
            band: 0.5,
        };
        // up = -Y, so "below the plane" is +Y. A point at y = +0.1 inside r = 1.
        let (inside, weight) = lid_membership([0.0, 0.1, 0.0], &lid);
        assert!(inside, "below the plane and inside the radius");
        assert!((weight - 0.2).abs() < 1e-12, "0.1 / band 0.5 = 0.2, got {weight}");
        // Just outside the radius: the hard clip says no, the graded weight ramps.
        let (inside, weight) = lid_membership([1.5, 0.4, 0.0], &lid);
        assert!(!inside);
        assert!(weight > 0.0 && weight < 1.0, "in the falloff ring, got {weight}");
        // Above the plane: neither.
        let (inside, weight) = lid_membership([0.0, -0.1, 0.0], &lid);
        assert!(!inside);
        assert_eq!(weight, 0.0);
    }

    #[test]
    fn undo_and_redo_are_pure_cursor_moves() {
        let mut doc = EditDocument::new("trippy-bundle-1".to_owned());
        doc.add_region(&box_region("r-1", 1.0, Op::Blend), None).unwrap();
        doc.add_region(&box_region("r-2", 0.0, Op::Fade), None).unwrap();
        assert_eq!(doc.regions().len(), 2);
        assert_eq!(doc.cursor, 2);

        assert!(doc.undo());
        assert_eq!(doc.regions().len(), 1);
        assert_eq!(doc.log.len(), 2, "undo must not shrink the log");
        assert!(doc.redo());
        assert_eq!(doc.regions().len(), 2);
        assert!(!doc.redo(), "already at the end");

        doc.undo();
        doc.undo();
        assert_eq!(doc.regions().len(), 0);
        assert!(!doc.undo(), "already at the start");
    }

    #[test]
    fn a_new_edit_after_an_undo_discards_the_redone_future() {
        let mut doc = EditDocument::default();
        doc.add_region(&box_region("r-1", 1.0, Op::Blend), None).unwrap();
        doc.add_region(&box_region("r-2", 1.0, Op::Blend), None).unwrap();
        doc.undo();
        doc.add_region(&box_region("r-3", 1.0, Op::Blend), None).unwrap();
        assert_eq!(doc.log.len(), 2, "r-2's entry was truncated");
        assert!(!doc.can_redo());
        let ids: Vec<&str> = doc.regions().iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["r-1", "r-3"]);
    }

    #[test]
    fn update_and_reorder_replay_the_same_way_the_python_does() {
        let mut doc = EditDocument::default();
        doc.add_region(&box_region("r-1", 1.0, Op::Blend), None).unwrap();
        doc.add_region(&box_region("r-2", 1.0, Op::Blend), None).unwrap();
        doc.update_region("r-1", json!({ "mix": 0.25, "enabled": false }))
            .unwrap();
        let r1 = doc.region("r-1").unwrap();
        assert!((r1.mix - 0.25).abs() < 1e-12);
        assert!(!r1.enabled);

        doc.reorder(vec!["r-2".to_owned(), "r-1".to_owned()]).unwrap();
        assert_eq!(doc.order(), ["r-2", "r-1"]);
        assert_eq!(doc.regions()[0].id, "r-2");

        // Not a permutation.
        assert!(doc.reorder(vec!["r-2".to_owned()]).is_err());
        // The failed push left the document exactly as it was.
        assert_eq!(doc.order(), ["r-2", "r-1"]);
    }

    #[test]
    fn a_document_round_trips_through_json_with_its_history() {
        let mut doc = EditDocument::new("trippy-bundle-1".to_owned());
        doc.add_region(&box_region("r-1", 1.0, Op::Blend), None).unwrap();
        doc.add_region(&box_region("r-2", 0.5, Op::Fade), None).unwrap();
        doc.update_region("r-2", json!({ "mix": 0.75 })).unwrap();
        doc.undo();

        let text = serde_json::to_string(&doc.to_json()).unwrap();
        let back = EditDocument::from_json(&serde_json::from_str(&text).unwrap()).unwrap();
        assert_eq!(back.cursor, doc.cursor);
        assert_eq!(back.log.len(), doc.log.len());
        assert_eq!(back.order(), doc.order());
        assert_eq!(back.regions(), doc.regions());
        // The reopened document can still redo the edit that was undone.
        let mut back = back;
        assert!(back.can_redo());
        back.redo();
        assert!((back.region("r-2").unwrap().mix - 0.75).abs() < 1e-12);
    }

    #[test]
    fn an_order_that_disagrees_with_the_log_is_rejected() {
        let mut doc = EditDocument::default();
        doc.add_region(&box_region("r-1", 1.0, Op::Blend), None).unwrap();
        doc.add_region(&box_region("r-2", 1.0, Op::Blend), None).unwrap();
        let mut raw = doc.to_json();
        raw["order"] = json!(["r-2", "r-1"]);
        let err = EditDocument::from_json(&raw).unwrap_err();
        assert!(err.contains("does not match replaying"), "{err}");
    }

    #[test]
    fn params_written_by_the_python_survive_a_round_trip_unchanged() {
        // Python omits `quat` when the caller did; the viewer must write back
        // exactly what it read.
        let raw = json!({
            "id": "r-x", "name": "n", "kind": "box", "enabled": true, "mix": 1.0,
            "op": "blend",
            "params": { "center": [0.0, 0.0, 0.0], "half_extents": [1.0, 1.0, 1.0] },
        });
        let region = Region::from_json(&raw).unwrap();
        assert_eq!(region.to_json(), raw);
        assert!(matches!(region.params, Params::Box { quat, .. } if quat == [1.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn malformed_regions_are_rejected_where_the_python_rejects_them() {
        let bad_mix = json!({"id": "a", "kind": "sphere", "mix": 1.5,
                             "params": {"center": [0, 0, 0], "radius": 1.0}});
        assert!(Region::from_json(&bad_mix).unwrap_err().contains("mix"));

        let bad_extent = json!({"id": "a", "kind": "box",
                                "params": {"center": [0, 0, 0], "half_extents": [1.0, 0.0, 1.0]}});
        assert!(Region::from_json(&bad_extent)
            .unwrap_err()
            .contains("half_extents"));

        let bad_op = json!({"id": "a", "kind": "sphere", "op": "vanish",
                            "params": {"center": [0, 0, 0], "radius": 1.0}});
        assert!(Region::from_json(&bad_op).unwrap_err().contains("op"));

        let bad_ids = json!({"id": "a", "kind": "pointset", "params": {"point_ids": [1, -2]}});
        assert!(Region::from_json(&bad_ids).unwrap_err().contains(">= 0"));
    }

    #[test]
    fn a_brush_region_round_trips_and_its_membership_is_its_own_voxel() {
        let raw = json!({
            "id": "r-b", "name": "brush-1", "kind": "brush", "enabled": true,
            "mix": 0.0, "op": "delete",
            "params": {
                "origin": [0.0, 0.0, 0.0], "cell_size": 0.5,
                "cells": [[0, 0, 0], [1, 0, 0]], "weights": [1.0, 0.25],
            },
            "source": { "tool": "brush", "radius": 0.3 },
        });
        let region = Region::from_json(&raw).unwrap();
        assert_eq!(region.kind, Kind::Brush);
        assert_eq!(region.short_kind(), "brush");
        assert_eq!(region.source_tool(), Some("brush"));
        assert_eq!(region.to_json(), raw, "a brush round-trips verbatim");

        // The point's own voxel, `floor((p - origin) / cell_size)`.
        assert_eq!(region_weight(&region, 0, [0.1, 0.1, 0.1]), 1.0);
        assert_eq!(region_weight(&region, 0, [0.6, 0.1, 0.1]), 0.25);
        assert_eq!(region_weight(&region, 0, [1.6, 0.1, 0.1]), 0.0);
        // `delete`'s hard membership is "weight > 0" for every kind but `lid`.
        assert!(region_contains(&region, 0, [0.6, 0.1, 0.1]));
        assert!(!region_contains(&region, 0, [1.6, 0.1, 0.1]));
    }

    /// A brush region of `n` cells at `[[0,0,0], [1,0,0], ..., [n-1,0,0]]`,
    /// `cell_size = 1.0` — the exact recipe `tests/test_edit_model.py`'s
    /// `test_brush_npz_sidecar_written_above_threshold` uses on the Python
    /// side, so the two tests pin the SAME synthetic stroke.
    fn brush_region_of_n_cells(id: &str, n: usize) -> Region {
        let cells: Vec<[i64; 3]> = (0..n).map(|i| [i as i64, 0, 0]).collect();
        let mut brush_cells = BrushCells::new();
        brush_cells.paint_cells(&cells, 1.0);
        Region::new(
            id.to_owned(),
            "brush-1".to_owned(),
            Params::Brush {
                origin: [0.0, 0.0, 0.0],
                cell_size: 1.0,
                cells: brush_cells,
            },
            1.0,
            Op::Delete,
        )
    }

    #[test]
    fn a_brush_above_the_npz_threshold_is_externalised_on_save() {
        let n = EDIT_BRUSH_NPZ_CELL_THRESHOLD + 10;
        let region = brush_region_of_n_cells("r-big", n);
        let mut doc = EditDocument::new(String::new());
        doc.add_region(&region, None).unwrap();

        let dir = std::env::temp_dir().join(format!("trips-edit-npz-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("edits.json");
        doc.save(&path).unwrap();

        let written: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let params = &written["regions"][0]["params"];
        assert!(params.get("cells").is_none(), "cells must not stay inline");
        assert_eq!(
            params["cells_npz"].as_str().unwrap(),
            format!("edits_brush_{}.npz", region.id)
        );
        assert_eq!(params["n_cells"].as_u64().unwrap(), n as u64);

        // Readable through the SAME reader the live splat's `.npz` loading
        // uses (`brush_pyramid::npz`) -- exactly what an external reader (not
        // either loader, which both replay `undo_stack.log` instead) would do.
        let sidecar = dir.join(format!("edits_brush_{}.npz", region.id));
        let members = brush_pyramid::npz::read_npz(&sidecar).unwrap();
        let cells_arr = members.get("cells").expect("a cells member");
        assert_eq!(cells_arr.shape, vec![n, 3]);
        let flat = cells_arr.to_i32().unwrap();
        for i in 0..n {
            assert_eq!(&flat[3 * i..3 * i + 3], [i as i32, 0, 0]);
        }
        assert!(
            !members.contains_key("weights"),
            "an all-1.0 brush writes no weights array, same as the inline case"
        );

        // In-memory state is untouched by the externalisation: `self.regions`
        // still holds the literal cells, and reloading (which replays the
        // log, never the sidecar) reproduces every cell exactly.
        let Params::Brush { cells, .. } = &doc.region("r-big").unwrap().params else {
            panic!("still a brush")
        };
        assert_eq!(cells.len(), n);

        let reopened = EditDocument::load(&path).unwrap();
        let Params::Brush { cells, .. } = &reopened.region("r-big").unwrap().params else {
            panic!("still a brush")
        };
        assert_eq!(cells.len(), n, "reload replays the log, not the sidecar");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_brush_at_or_below_the_npz_threshold_stays_inline() {
        let region = brush_region_of_n_cells("r-small", 2);
        let mut doc = EditDocument::new(String::new());
        doc.add_region(&region, None).unwrap();

        let dir = std::env::temp_dir().join(format!("trips-edit-npz-inline-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("edits.json");
        doc.save(&path).unwrap();

        let written: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let params = &written["regions"][0]["params"];
        assert_eq!(params["cells"], json!([[0, 0, 0], [1, 0, 0]]));
        assert!(params.get("cells_npz").is_none());
        assert!(
            !dir.join("edits_brush_r-small.npz").exists(),
            "no sidecar below the threshold"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_region_without_a_source_key_does_not_grow_one() {
        // The Python writes `"source": null`; a file written before the field
        // existed has no key. Both must survive the viewer unchanged.
        let explicit = json!({"id": "a", "kind": "sphere", "name": "s", "enabled": true,
                              "mix": 1.0, "op": "blend", "source": null,
                              "params": {"center": [0, 0, 0], "radius": 1.0}});
        assert_eq!(Region::from_json(&explicit).unwrap().to_json(), explicit);

        let absent = json!({"id": "a", "kind": "sphere", "name": "s", "enabled": true,
                            "mix": 1.0, "op": "blend",
                            "params": {"center": [0, 0, 0], "radius": 1.0}});
        let region = Region::from_json(&absent).unwrap();
        assert_eq!(region.to_json(), absent);
        assert_eq!(region.source_tool(), None);

        // A non-object source is refused where the Python refuses it.
        let bad = json!({"id": "a", "kind": "sphere", "source": "brush",
                         "params": {"center": [0, 0, 0], "radius": 1.0}});
        assert!(Region::from_json(&bad).unwrap_err().contains("source"));
    }

    #[test]
    fn one_gesture_is_one_undo_step_however_many_frames_it_took() {
        let mut doc = EditDocument::default();
        doc.add_region(&box_region("r-1", 1.0, Op::Blend), None).unwrap();
        // A 3-frame drag: the first frame starts the gesture, the rest coalesce.
        doc.update_region_coalesced("r-1", json!({ "mix": 0.9 }), false).unwrap();
        doc.update_region_coalesced("r-1", json!({ "mix": 0.5 }), true).unwrap();
        doc.update_region_coalesced("r-1", json!({ "mix": 0.1 }), true).unwrap();
        assert_eq!(doc.log.len(), 2, "add + one gesture");
        assert!((doc.region("r-1").unwrap().mix - 0.1).abs() < 1e-12);

        doc.undo();
        assert!(
            (doc.region("r-1").unwrap().mix - 1.0).abs() < 1e-12,
            "one Cmd-Z takes the whole drag back"
        );

        // A coalescing add is the same story for a region the gesture created.
        let mut doc = EditDocument::default();
        doc.add_region_coalesced(&box_region("r-2", 1.0, Op::Blend), None, false).unwrap();
        doc.add_region_coalesced(&box_region("r-2", 0.5, Op::Blend), None, true).unwrap();
        assert_eq!(doc.log.len(), 1);
        assert_eq!(doc.order(), ["r-2"], "no duplicate in the paint order");
        assert!((doc.region("r-2").unwrap().mix - 0.5).abs() < 1e-12);
        doc.undo();
        assert!(doc.regions().is_empty(), "one Cmd-Z removes the whole stroke");
    }

    #[test]
    fn coalescing_never_rewrites_an_entry_the_user_undid_past() {
        let mut doc = EditDocument::default();
        doc.add_region(&box_region("r-1", 1.0, Op::Blend), None).unwrap();
        doc.update_region("r-1", json!({ "mix": 0.5 })).unwrap();
        doc.undo();
        // The cursor is before the update: coalescing must append (truncating
        // the redo future, the ordinary rule), not eat an entry behind it.
        doc.update_region_coalesced("r-1", json!({ "mix": 0.25 }), true).unwrap();
        assert_eq!(doc.log.len(), 2);
        assert!((doc.region("r-1").unwrap().mix - 0.25).abs() < 1e-12);
        doc.undo();
        assert!((doc.region("r-1").unwrap().mix - 1.0).abs() < 1e-12);
    }

    #[test]
    fn auto_names_number_across_tools_and_self_heal() {
        // The fixture (`names.json`) is the parity check; this is the rule
        // stated once in Rust's own terms.
        assert_eq!(auto_region_name(std::iter::empty(), "brush", None), "brush-1");
        assert_eq!(
            auto_region_name(["click-1", "shade-clouds-2"], "brush", None),
            "brush-3"
        );
        assert_eq!(
            auto_region_name(["click-1"], "sam-box", Some("IMG_3703")),
            "sam-box-IMG_3703-2"
        );
        // A document, not a bare list.
        let mut doc = EditDocument::default();
        let mut region = box_region("r-1", 1.0, Op::Blend);
        region.name = "brush-4".to_owned();
        doc.add_region(&region, None).unwrap();
        assert_eq!(auto_name_for(&doc, "brush", None), "brush-5");
    }

    #[test]
    fn generated_ids_are_unique_and_shaped_like_the_pythons() {
        let a = new_region_id();
        let b = new_region_id();
        assert_ne!(a, b);
        assert!(a.starts_with("r-"), "{a}");
        assert_eq!(a.len(), 2 + REGION_ID_HEX_LEN, "{a}");
    }
}
