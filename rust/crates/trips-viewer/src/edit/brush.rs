//! The `brush` region kind: a sparse voxel set, painted and erased with spheres.
//!
//! Module: `trips_viewer::edit::brush`
//! Purpose: the viewer's half of `docs/EDITOR.md` §1's `brush` entry — a **twin
//!     of `trippy/edit/model.py`**'s `brush_membership` / `paint_sphere` /
//!     `paint_along` / `erase`, so a region painted in the viewer and one
//!     painted by `trippy edits add-brush` are the same region, cell for cell.
//! Invariants:
//!     - A point's membership is its OWN voxel looked up in the occupied set:
//!       `floor((p - origin) / cell_size)`, nothing interpolated, nothing
//!       distance-weighted. That is why `weights.rs`/`apply.rs` needed no
//!       brush-specific code — it is just another membership.
//!     - Painting is a box-sphere OVERLAP test ([`sphere_touched_cells`]), not
//!       a "is the cell's centre inside the sphere" test, so a stroke paints
//!       every cell the sphere visibly covers. `tests/fixtures/synthetic/
//!       edit_golden/brush.json`'s own `test_the_brush_strokes_paint_cells_a_
//!       centre_test_would_miss` pins the difference, because a centre test
//!       would give a strictly smaller cell set and every lookup would still
//!       work.
//!     - Repeated painting only ever STRENGTHENS a cell (`max(existing,
//!       weight)`); [`BrushCells::erase_cells`] is the only way to lower one,
//!       and it drops the cell outright regardless of its weight.
//!     - Cells keep **first-painted order** — the same order the Python's
//!       insertion-ordered dict produces — so the golden fixture can compare
//!       the two sides' cell lists exactly rather than as sets.
//!     - An all-`1.0` weight array is dropped on the way out ([`BrushCells::
//!       weights_json`] returns `None`), exactly as the Python canonicalises
//!       it, so a plain ungraded brush never carries a redundant array.
//! Units: world units for `origin`/`cell_size`/every centre and radius; cell
//!     indices are dimensionless integers; weights are in `[0, 1]`.
//! Related docs: `docs/EDITOR.md` §1 "brush"; `trippy/edit/model.py` (the twin);
//!     `trippy/edit/golden.py`'s `build_brush_fixture`.

use std::collections::HashMap;

use serde_json::{json, Value};

use super::cluster::ClickCamera;

/// `"format"` of `tests/fixtures/synthetic/edit_golden/brush.json`, matching
/// `trippy.edit.golden.BRUSH_FIXTURE_FORMAT`.
pub const BRUSH_FIXTURE_FORMAT: &str = "trippy-edit-brush-1";

/// A brush cell index must fit in a signed int32, mirroring
/// `trippy.constants.EDIT_BRUSH_CELL_INT32_ABS_MAX` (the `.npz` sidecar writes
/// int32 triplets, so a wider index could not be written back).
pub const CELL_INT32_ABS_MAX: i64 = 2_147_483_647;

/// The most cells ONE stroke may sweep before it is refused.
///
/// A stroke's bounding box is `(2r / cell_size)^3` cells, so a radius mistyped
/// two orders of magnitude too large would otherwise try to allocate tens of
/// gigabytes. The Python has no such guard (numpy would simply fail to
/// allocate); this returns an error the panel can show instead. 16.7 M cells is
/// a 256^3 box, far past anything a hand-painted stroke needs.
pub const MAX_STROKE_CELLS: usize = 1 << 24;

/// A brush's occupied voxels: insertion-ordered cells, their weights, and an index.
///
/// The `HashMap` makes membership O(1) per point (the composition walks every
/// point of the cloud against every region); the `Vec`s keep the on-disk order
/// stable so a save is byte-reproducible and the golden comparison can be exact.
#[derive(Debug, Clone, Default)]
pub struct BrushCells {
    /// Occupied cell indices, in first-painted order.
    cells: Vec<[i64; 3]>,
    /// One weight per cell, parallel to `cells`, each in `[0, 1]`.
    weights: Vec<f64>,
    /// `cell -> row in cells/weights`.
    index: HashMap<[i64; 3], usize>,
}

impl PartialEq for BrushCells {
    /// Cells and weights, in order. The index is derived and never compared.
    fn eq(&self, other: &Self) -> bool {
        self.cells == other.cells && self.weights == other.weights
    }
}

impl BrushCells {
    /// An empty brush.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many cells are occupied.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Whether nothing is painted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// The occupied cells, in first-painted order.
    #[must_use]
    pub fn cells(&self) -> &[[i64; 3]] {
        &self.cells
    }

    /// The per-cell weights, parallel to [`Self::cells`].
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// This cell's weight, or `0.0` when it is not occupied.
    #[must_use]
    pub fn weight_at(&self, cell: [i64; 3]) -> f64 {
        self.index.get(&cell).map_or(0.0, |i| self.weights[*i])
    }

    /// Add `touched` at `max(existing, weight)` — a stroke only ever strengthens.
    ///
    /// Mirrors `trippy.edit.model._merge_brush_cells`.
    pub fn paint_cells(&mut self, touched: &[[i64; 3]], weight: f64) {
        for cell in touched {
            match self.index.get(cell) {
                Some(row) => self.weights[*row] = self.weights[*row].max(weight),
                None => {
                    self.index.insert(*cell, self.cells.len());
                    self.cells.push(*cell);
                    self.weights.push(weight);
                }
            }
        }
    }

    /// Drop every cell in `touched`, whatever its weight.
    ///
    /// Mirrors `trippy.edit.model._remove_brush_cells`: an erase is not a
    /// weight subtraction, it removes the cell.
    pub fn erase_cells(&mut self, touched: &[[i64; 3]]) {
        let drop: std::collections::HashSet<[i64; 3]> = touched.iter().copied().collect();
        if drop.is_empty() {
            return;
        }
        let mut cells = Vec::with_capacity(self.cells.len());
        let mut weights = Vec::with_capacity(self.weights.len());
        for (cell, weight) in self.cells.iter().zip(&self.weights) {
            if !drop.contains(cell) {
                cells.push(*cell);
                weights.push(*weight);
            }
        }
        self.cells = cells;
        self.weights = weights;
        self.reindex();
    }

    fn reindex(&mut self) {
        self.index = self
            .cells
            .iter()
            .enumerate()
            .map(|(row, cell)| (*cell, row))
            .collect();
    }

    /// The `params["cells"]` array.
    #[must_use]
    pub fn cells_json(&self) -> Value {
        Value::Array(
            self.cells
                .iter()
                .map(|c| json!([c[0], c[1], c[2]]))
                .collect(),
        )
    }

    /// The `params["weights"]` array, or `None` when every weight is `1.0`.
    ///
    /// The same canonicalisation `_merge_brush_cells` applies, so an ungraded
    /// brush written by either side carries no weights array at all.
    #[must_use]
    pub fn weights_json(&self) -> Option<Value> {
        if self.weights.is_empty() || self.weights.iter().all(|w| (w - 1.0).abs() < 1e-12) {
            return None;
        }
        Some(Value::Array(
            self.weights.iter().map(|w| json!(w)).collect(),
        ))
    }

    /// Parse `params["cells"]` / `params["weights"]`.
    ///
    /// A repeated cell keeps its LAST weight, exactly as the Python's
    /// `_brush_sorted_lookup` dict does; the row order of the first occurrence
    /// is kept, which is what a hand-edited file most nearly means.
    ///
    /// # Errors
    /// Returns `Err` with the message shape `trippy.edit.model.
    /// _validate_brush_params` raises: a non-list `cells`, a row that is not
    /// three integers, an index outside int32, a non-list `weights`, a length
    /// mismatch, or a weight outside `[0, 1]`.
    pub fn from_json(cells: Option<&Value>, weights: Option<&Value>) -> Result<Self, String> {
        let rows = cells.and_then(Value::as_array).ok_or_else(|| {
            format!(
                "brush.cells must be a list, got {}",
                cells.map_or_else(|| "None".to_owned(), ToString::to_string)
            )
        })?;
        let mut parsed = Vec::with_capacity(rows.len());
        for row in rows {
            let triplet = row
                .as_array()
                .filter(|t| t.len() == 3)
                .ok_or_else(|| "brush.cells must be a list of [i, j, k] triplets".to_owned())?;
            let mut cell = [0_i64; 3];
            for (slot, item) in cell.iter_mut().zip(triplet) {
                let v = item
                    .as_i64()
                    .ok_or_else(|| "brush.cells must be a list of [i, j, k] triplets".to_owned())?;
                if v.abs() > CELL_INT32_ABS_MAX {
                    return Err(format!(
                        "brush.cells must fit in a signed int32 (|i| <= {CELL_INT32_ABS_MAX})"
                    ));
                }
                *slot = v;
            }
            parsed.push(cell);
        }

        let parsed_weights: Option<Vec<f64>> = match weights {
            None | Some(Value::Null) => None,
            Some(value) => {
                let array = value.as_array().ok_or_else(|| {
                    format!("brush.weights must be a list, got {value}")
                })?;
                if array.len() != parsed.len() {
                    return Err(format!(
                        "brush.weights must have one entry per cell ({}), got {}",
                        parsed.len(),
                        array.len()
                    ));
                }
                let mut out = Vec::with_capacity(array.len());
                for item in array {
                    let w = item.as_f64().filter(|w| w.is_finite()).ok_or_else(|| {
                        "brush.weights must all be finite numbers in [0, 1]".to_owned()
                    })?;
                    if !(0.0..=1.0).contains(&w) {
                        return Err(
                            "brush.weights must all be finite numbers in [0, 1]".to_owned()
                        );
                    }
                    out.push(w);
                }
                Some(out)
            }
        };

        let mut brush = Self::new();
        for (row, cell) in parsed.iter().enumerate() {
            let weight = parsed_weights.as_ref().map_or(1.0, |w| w[row]);
            match brush.index.get(cell) {
                // A duplicate keeps its LAST weight (the Python's dict does the same).
                Some(slot) => brush.weights[*slot] = weight,
                None => {
                    brush.index.insert(*cell, brush.cells.len());
                    brush.cells.push(*cell);
                    brush.weights.push(weight);
                }
            }
        }
        Ok(brush)
    }
}

/// The voxel a world point falls in: `floor((p - origin) / cell_size)`.
///
/// Mirrors `trippy.edit.model._brush_voxel_index`. A non-finite coordinate (or
/// an index past int32) yields `None`: such a point is in no cell, which is the
/// same answer `brush_membership` gives it.
#[must_use]
pub fn voxel_index(p: [f64; 3], origin: [f64; 3], cell_size: f64) -> Option<[i64; 3]> {
    let mut out = [0_i64; 3];
    for axis in 0..3 {
        let v = ((p[axis] - origin[axis]) / cell_size).floor();
        if !v.is_finite() || v.abs() > CELL_INT32_ABS_MAX as f64 {
            return None;
        }
        #[allow(clippy::cast_possible_truncation)]
        {
            out[axis] = v as i64;
        }
    }
    Some(out)
}

/// One point's membership weight in a brush: its own cell's weight, else `0.0`.
///
/// Mirrors `trippy.edit.model.brush_membership` for a single point.
#[must_use]
pub fn membership(cells: &BrushCells, origin: [f64; 3], cell_size: f64, p: [f64; 3]) -> f64 {
    voxel_index(p, origin, cell_size).map_or(0.0, |cell| cells.weight_at(cell))
}

/// Every cell a world-space sphere OVERLAPS (box-sphere intersection).
///
/// Not "cells whose centre is inside the sphere": a cell is touched when the
/// closest point of its own axis-aligned box is within `radius` of `center`.
/// Mirrors `trippy.edit.model._sphere_touched_cells`, including its iteration
/// order (`i` outermost, `k` innermost — numpy's `meshgrid(..., indexing="ij")`
/// then `ravel()`), which is what keeps the two sides' cell lists in the same
/// order.
///
/// # Errors
/// Returns `Err` when the sphere's bounding box exceeds [`MAX_STROKE_CELLS`]
/// cells — a mistyped radius, not a stroke.
pub fn sphere_touched_cells(
    origin: [f64; 3],
    cell_size: f64,
    center: [f64; 3],
    radius: f64,
) -> Result<Vec<[i64; 3]>, String> {
    if !(radius > 0.0) || !cell_size.is_finite() || cell_size <= 0.0 {
        return Ok(Vec::new());
    }
    if !center.iter().all(|c| c.is_finite()) || !radius.is_finite() {
        return Ok(Vec::new());
    }
    let mut lo = [0_i64; 3];
    let mut hi = [0_i64; 3];
    let mut span = 1_u128;
    for axis in 0..3 {
        let l = ((center[axis] - radius - origin[axis]) / cell_size).floor();
        let h = ((center[axis] + radius - origin[axis]) / cell_size).floor();
        if !l.is_finite() || !h.is_finite() || l.abs() > CELL_INT32_ABS_MAX as f64
            || h.abs() > CELL_INT32_ABS_MAX as f64
        {
            return Err(format!(
                "a brush stroke of radius {radius} on a {cell_size}-unit grid runs past the \
                 int32 cell range"
            ));
        }
        #[allow(clippy::cast_possible_truncation)]
        {
            lo[axis] = l as i64;
            hi[axis] = h as i64;
        }
        span *= (hi[axis] - lo[axis] + 1).max(0) as u128;
    }
    if span > MAX_STROKE_CELLS as u128 {
        return Err(format!(
            "a brush stroke of radius {radius} on a {cell_size}-unit grid sweeps {span} cells \
             (limit {MAX_STROKE_CELLS}); raise the cell size or lower the radius"
        ));
    }

    let r2 = radius * radius;
    let mut out = Vec::new();
    for i in lo[0]..=hi[0] {
        for j in lo[1]..=hi[1] {
            for k in lo[2]..=hi[2] {
                let idx = [i, j, k];
                let mut dist2 = 0.0_f64;
                for axis in 0..3 {
                    #[allow(clippy::cast_precision_loss)]
                    let min = idx[axis] as f64;
                    let cell_min = origin[axis] + min * cell_size;
                    let cell_max = cell_min + cell_size;
                    // `np.clip(center, cell_min, cell_max)`: the closest point of the box.
                    let closest = center[axis].clamp(cell_min, cell_max);
                    let d = closest - center[axis];
                    dist2 += d * d;
                }
                if dist2 <= r2 {
                    out.push(idx);
                }
            }
        }
    }
    Ok(out)
}

/// Paint one sphere into `cells` (`trippy.edit.model.paint_sphere`).
///
/// # Errors
/// As [`sphere_touched_cells`]; also when `weight` is outside `[0, 1]`, which
/// is where the Python raises too.
pub fn paint_sphere(
    cells: &mut BrushCells,
    origin: [f64; 3],
    cell_size: f64,
    center: [f64; 3],
    radius: f64,
    weight: f64,
) -> Result<usize, String> {
    if !(0.0..=1.0).contains(&weight) {
        return Err(format!("weight must be in [0, 1], got {weight}"));
    }
    let touched = sphere_touched_cells(origin, cell_size, center, radius)?;
    cells.paint_cells(&touched, weight);
    Ok(touched.len())
}

/// Paint a STROKE: the union of [`paint_sphere`] at every point of a path.
///
/// Mirrors `trippy.edit.model.paint_along`, including the order the cells are
/// added in (path point by path point, each in `sphere_touched_cells` order).
///
/// # Errors
/// As [`paint_sphere`].
pub fn paint_along(
    cells: &mut BrushCells,
    origin: [f64; 3],
    cell_size: f64,
    points: &[[f64; 3]],
    radius: f64,
    weight: f64,
) -> Result<usize, String> {
    if !(0.0..=1.0).contains(&weight) {
        return Err(format!("weight must be in [0, 1], got {weight}"));
    }
    let mut touched: Vec<[i64; 3]> = Vec::new();
    for point in points {
        touched.extend(sphere_touched_cells(origin, cell_size, *point, radius)?);
    }
    cells.paint_cells(&touched, weight);
    Ok(touched.len())
}

/// Erase one sphere from `cells` (`trippy.edit.model.erase`).
///
/// # Errors
/// As [`sphere_touched_cells`].
pub fn erase(
    cells: &mut BrushCells,
    origin: [f64; 3],
    cell_size: f64,
    center: [f64; 3],
    radius: f64,
) -> Result<usize, String> {
    let touched = sphere_touched_cells(origin, cell_size, center, radius)?;
    let before = cells.len();
    cells.erase_cells(&touched);
    Ok(before - cells.len())
}

/// Replay a `brush.json` `"strokes"` entry into `cells`.
///
/// The golden fixture records the authoring CALLS, not only their result
/// (`trippy.edit.golden.replay_brush_strokes` is the Python twin), so this
/// module has to reproduce the voxelisation and not merely the lookup.
///
/// # Errors
/// Returns `Err` on an unknown `"op"`, a malformed argument, or anything
/// [`paint_sphere`] refuses.
pub fn apply_stroke_json(
    cells: &mut BrushCells,
    origin: [f64; 3],
    cell_size: f64,
    stroke: &Value,
) -> Result<(), String> {
    let vec3 = |value: Option<&Value>, what: &str| -> Result<[f64; 3], String> {
        let array = value
            .and_then(Value::as_array)
            .filter(|a| a.len() == 3)
            .ok_or_else(|| format!("brush stroke: {what} must be three numbers"))?;
        let mut out = [0.0_f64; 3];
        for (slot, item) in out.iter_mut().zip(array) {
            *slot = item
                .as_f64()
                .ok_or_else(|| format!("brush stroke: {what} must be three numbers"))?;
        }
        Ok(out)
    };
    let radius = stroke
        .get("radius")
        .and_then(Value::as_f64)
        .ok_or_else(|| "brush stroke: 'radius' must be a number".to_owned())?;
    let weight = stroke.get("weight").and_then(Value::as_f64).unwrap_or(1.0);
    match stroke.get("op").and_then(Value::as_str) {
        Some("paint_sphere") => {
            paint_sphere(
                cells,
                origin,
                cell_size,
                vec3(stroke.get("center"), "center")?,
                radius,
                weight,
            )?;
        }
        Some("paint_along") => {
            let rows = stroke
                .get("points")
                .and_then(Value::as_array)
                .ok_or_else(|| "brush stroke: 'points' must be a list".to_owned())?;
            let mut points = Vec::with_capacity(rows.len());
            for row in rows {
                points.push(vec3(Some(row), "points")?);
            }
            paint_along(cells, origin, cell_size, &points, radius, weight)?;
        }
        Some("erase") => {
            erase(
                cells,
                origin,
                cell_size,
                vec3(stroke.get("center"), "center")?,
                radius,
            )?;
        }
        other => {
            return Err(format!(
                "unknown brush stroke op {}",
                other.map_or_else(|| "None".to_owned(), |o| format!("{o:?}"))
            ))
        }
    }
    Ok(())
}

/// How far from the cursor (render pixels) a point may project and still count
/// as "the point under the cursor" for the depth anchor.
///
/// The same job `cluster::ClickParams::radius_px` does for a click, with its
/// own constant because a brush wants the NEAREST surface under a small
/// crosshair, not a catchment wide enough to seed a cluster.
pub const ANCHOR_RADIUS_PX: f64 = 12.0;

/// The depth of the nearest point under the cursor, world units in front of
/// the camera.
///
/// `docs/EDITOR.md` §4: the brush paints in 3D, so a pixel has to become a
/// world position, and the only depth the viewer has is the cloud's own (there
/// is no depth buffer — §2's "why not per-pixel depth"). This is the same
/// projection trick click-to-cluster uses, cut down to "which point is nearest
/// the camera among those that land near this pixel".
///
/// Returns `None` when nothing projects within [`ANCHOR_RADIUS_PX`] — the
/// caller then keeps the stroke's previous anchor rather than painting into
/// empty space at an invented depth.
///
/// # Arguments
/// - `camera`: the camera the frame was drawn with.
/// - `xyz`: flat `(N, 3)` world positions.
/// - `px`: the cursor, in that camera's own (render) pixels.
/// - `radius_px`: the catchment, usually [`ANCHOR_RADIUS_PX`].
#[must_use]
pub fn depth_anchor(
    camera: &ClickCamera,
    xyz: &[f64],
    px: (f64, f64),
    radius_px: f64,
) -> Option<f64> {
    let r2 = radius_px * radius_px;
    let mut best: Option<f64> = None;
    for i in 0..xyz.len() / 3 {
        let p = [xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]];
        let (u, v, z) = camera.project(p);
        if !(z > 0.0) || !u.is_finite() || !v.is_finite() {
            continue;
        }
        let du = u - px.0;
        let dv = v - px.1;
        if du.mul_add(du, dv * dv) > r2 {
            continue;
        }
        if best.is_none_or(|b| z < b) {
            best = Some(z);
        }
    }
    best
}

/// A screen-space bucket index over a static cloud's projections through ONE
/// camera pose, so [`depth_anchor`]'s "nearest point under the cursor" query
/// costs candidates-in-a-few-buckets instead of every point in the cloud.
///
/// `docs/EDITOR.md` §4 flagged `depth_anchor` as an unmeasured `O(points)`
/// scan on a Karekare-scale cloud (7.5M points, `trippy-brush-anchor-perf-1`);
/// a drag can sample many times a second (`edit_ui.rs`'s `BRUSH_SAMPLE_PX`
/// throttles it, not eliminates it), so that scan repeats every sample. This
/// index amortises the projection: built ONCE per camera pose (the caller
/// rebuilds it only when the camera changes — `ClickCamera` is `PartialEq`,
/// so `edit_ui.rs::resolve_brush` compares by value), every later sample is a
/// handful of bucket lookups.
///
/// This is an ACCELERATION structure, not an approximation: with a cell edge
/// `>= radius_px`, the 3x3 neighbourhood of the query pixel's own cell visits
/// every point a brute-force `du^2 + dv^2 <= radius_px^2` scan would keep (the
/// standard uniform-grid range-query guarantee — a point within `radius_px`
/// of a pixel can only ever land in the cell the pixel is in or one of its
/// eight neighbours when the cell is at least as wide as the search radius).
/// [`tests::the_grid_agrees_with_the_brute_force_scan`] checks this against
/// [`depth_anchor`] directly on randomised inputs, including the exact
/// [`ANCHOR_RADIUS_PX`] this module ships with.
pub struct ScreenGrid {
    /// Cell edge, render pixels. Must be `>=` any `radius_px` a query uses.
    cell_px: f64,
    /// `(bucket i, bucket j) -> (u, v, z)` of every point that projects in
    /// front of the camera with finite screen coordinates. Kept `f64`, the
    /// same width [`depth_anchor`]'s brute-force scan uses, so the grid's
    /// answer is bit-identical to it rather than merely close — this is an
    /// index, not a lower-precision approximation.
    buckets: HashMap<(i64, i64), Vec<(f64, f64, f64)>>,
}

impl ScreenGrid {
    /// Project every point of `xyz` through `camera` once and bucket it by
    /// its screen position, `cell_px` wide.
    ///
    /// `cell_px` should be the largest `radius_px` [`Self::nearest`] will ever
    /// be called with (see the module doc's exactness argument); passing a
    /// smaller one is still correct as long as every query also widens its own
    /// search to the matching number of rings, which [`Self::nearest`] does
    /// not do — it always searches exactly one ring.
    #[must_use]
    pub fn build(camera: &ClickCamera, xyz: &[f64], cell_px: f64) -> Self {
        let cell_px = if cell_px.is_finite() && cell_px > 0.0 {
            cell_px
        } else {
            1.0
        };
        let mut buckets: HashMap<(i64, i64), Vec<(f64, f64, f64)>> = HashMap::new();
        for i in 0..xyz.len() / 3 {
            let p = [xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]];
            let (u, v, z) = camera.project(p);
            if !(z > 0.0) || !u.is_finite() || !v.is_finite() {
                continue;
            }
            #[allow(clippy::cast_possible_truncation)]
            let key = ((u / cell_px).floor() as i64, (v / cell_px).floor() as i64);
            buckets.entry(key).or_default().push((u, v, z));
        }
        Self { cell_px, buckets }
    }

    /// How many buckets are occupied (a diagnostic, not used by the query).
    #[must_use]
    pub fn len(&self) -> usize {
        self.buckets.values().map(Vec::len).sum()
    }

    /// True when the grid holds no points at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    /// The depth of the nearest point under `px`, exactly as [`depth_anchor`]
    /// would find scanning every point — see the module doc for why the 3x3
    /// neighbourhood is enough as long as `radius_px <= cell_px`.
    #[must_use]
    pub fn nearest(&self, px: (f64, f64), radius_px: f64) -> Option<f64> {
        let r2 = radius_px * radius_px;
        #[allow(clippy::cast_possible_truncation)]
        let base = (
            (px.0 / self.cell_px).floor() as i64,
            (px.1 / self.cell_px).floor() as i64,
        );
        let mut best: Option<f64> = None;
        for di in -1..=1_i64 {
            for dj in -1..=1_i64 {
                let Some(candidates) = self.buckets.get(&(base.0 + di, base.1 + dj)) else {
                    continue;
                };
                for &(u, v, z) in candidates {
                    let du = u - px.0;
                    let dv = v - px.1;
                    if du.mul_add(du, dv * dv) > r2 {
                        continue;
                    }
                    if best.is_none_or(|b| z < b) {
                        best = Some(z);
                    }
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORIGIN: [f64; 3] = [0.0, 0.0, 0.0];

    #[test]
    fn a_points_membership_is_its_own_voxels_weight() {
        let mut cells = BrushCells::new();
        cells.paint_cells(&[[0, 0, 0], [1, 0, 0]], 0.5);
        assert_eq!(membership(&cells, ORIGIN, 1.0, [0.5, 0.5, 0.5]), 0.5);
        assert_eq!(membership(&cells, ORIGIN, 1.0, [1.5, 0.5, 0.5]), 0.5);
        assert_eq!(membership(&cells, ORIGIN, 1.0, [2.5, 0.5, 0.5]), 0.0);
        // Negative coordinates floor DOWN, like numpy's own floor.
        assert_eq!(voxel_index([-0.25, 0.0, 0.0], ORIGIN, 1.0), Some([-1, 0, 0]));
    }

    #[test]
    fn painting_overlaps_the_cell_box_rather_than_testing_its_centre() {
        // A sphere of radius 0.1 at a cell CORNER touches all eight cells that
        // meet there, none of whose centres it contains.
        let touched = sphere_touched_cells(ORIGIN, 1.0, [1.0, 1.0, 1.0], 0.1).unwrap();
        assert_eq!(touched.len(), 8, "{touched:?}");
        assert!(touched.contains(&[0, 0, 0]));
        assert!(touched.contains(&[1, 1, 1]));
    }

    #[test]
    fn repeated_paint_only_strengthens_and_erase_removes_outright() {
        let mut cells = BrushCells::new();
        paint_sphere(&mut cells, ORIGIN, 1.0, [0.5, 0.5, 0.5], 0.2, 0.6).unwrap();
        assert_eq!(cells.weight_at([0, 0, 0]), 0.6);
        // A weaker stroke over the same cell leaves it alone.
        paint_sphere(&mut cells, ORIGIN, 1.0, [0.5, 0.5, 0.5], 0.2, 0.3).unwrap();
        assert_eq!(cells.weight_at([0, 0, 0]), 0.6);
        // A stronger one raises it.
        paint_sphere(&mut cells, ORIGIN, 1.0, [0.5, 0.5, 0.5], 0.2, 0.9).unwrap();
        assert_eq!(cells.weight_at([0, 0, 0]), 0.9);
        // An erase drops it whatever its weight.
        let removed = erase(&mut cells, ORIGIN, 1.0, [0.5, 0.5, 0.5], 0.2).unwrap();
        assert_eq!(removed, 1);
        assert_eq!(cells.weight_at([0, 0, 0]), 0.0);
        assert!(cells.is_empty());
    }

    #[test]
    fn a_stroke_is_the_union_of_its_spheres_in_path_order() {
        let mut along = BrushCells::new();
        paint_along(
            &mut along,
            ORIGIN,
            1.0,
            &[[0.5, 0.5, 0.5], [3.5, 0.5, 0.5]],
            0.2,
            1.0,
        )
        .unwrap();
        assert_eq!(along.cells(), [[0, 0, 0], [3, 0, 0]]);

        let mut apart = BrushCells::new();
        paint_sphere(&mut apart, ORIGIN, 1.0, [0.5, 0.5, 0.5], 0.2, 1.0).unwrap();
        paint_sphere(&mut apart, ORIGIN, 1.0, [3.5, 0.5, 0.5], 0.2, 1.0).unwrap();
        assert_eq!(along, apart, "one stroke == the same spheres one at a time");
    }

    #[test]
    fn an_all_ones_brush_writes_no_weights_array() {
        let mut cells = BrushCells::new();
        cells.paint_cells(&[[0, 0, 0]], 1.0);
        assert!(cells.weights_json().is_none());
        cells.paint_cells(&[[1, 0, 0]], 0.25);
        assert!(cells.weights_json().is_some(), "a graded brush keeps its weights");
    }

    #[test]
    fn a_json_round_trip_keeps_cells_weights_and_order() {
        let mut cells = BrushCells::new();
        cells.paint_cells(&[[2, 0, 0], [0, 1, 0]], 0.5);
        let back = BrushCells::from_json(Some(&cells.cells_json()), cells.weights_json().as_ref())
            .unwrap();
        assert_eq!(back, cells);
        assert_eq!(back.cells(), [[2, 0, 0], [0, 1, 0]], "order survives");
    }

    #[test]
    fn malformed_cells_and_weights_are_refused() {
        assert!(BrushCells::from_json(None, None).unwrap_err().contains("must be a list"));
        let bad_row = json!([[1, 2]]);
        assert!(BrushCells::from_json(Some(&bad_row), None)
            .unwrap_err()
            .contains("triplets"));
        let huge = json!([[3_000_000_000_i64, 0, 0]]);
        assert!(BrushCells::from_json(Some(&huge), None)
            .unwrap_err()
            .contains("int32"));
        let cells = json!([[0, 0, 0]]);
        let short = json!([]);
        assert!(BrushCells::from_json(Some(&cells), Some(&short))
            .unwrap_err()
            .contains("one entry per cell"));
        let out_of_range = json!([1.5]);
        assert!(BrushCells::from_json(Some(&cells), Some(&out_of_range))
            .unwrap_err()
            .contains("[0, 1]"));
    }

    #[test]
    fn an_absurd_radius_is_refused_rather_than_allocated() {
        let err = sphere_touched_cells(ORIGIN, 0.001, [0.0, 0.0, 0.0], 1e4).unwrap_err();
        assert!(err.contains("sweeps"), "{err}");
        // A zero or negative radius paints nothing, exactly as the Python does.
        assert!(sphere_touched_cells(ORIGIN, 1.0, [0.0, 0.0, 0.0], 0.0).unwrap().is_empty());
    }

    // --- ScreenGrid: the accelerated depth anchor must agree with brute force ---------

    /// A camera at the world origin looking down `+Z`, matching `gizmo`'s and
    /// `edit_ui.rs`'s own test camera, so a failure here reads the same units.
    fn test_camera() -> ClickCamera {
        ClickCamera {
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            t: [0.0, 0.0, 0.0],
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
        }
    }

    /// A tiny xorshift generator: deterministic and dependency-free, which is
    /// all a repeatable property test over "enough" random points needs.
    struct Xorshift(u64);
    impl Xorshift {
        fn next_f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            // Top 53 bits -> [0, 1).
            (self.0 >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
        }
    }

    /// A cloud spread through the camera's own view frustum, so a meaningful
    /// fraction of query pixels actually land near a point (an empty result on
    /// every query would let a broken grid pass by vacuous agreement).
    fn random_cloud(rng: &mut Xorshift, n: usize) -> Vec<f64> {
        let mut xyz = Vec::with_capacity(n * 3);
        for _ in 0..n {
            xyz.push((rng.next_f64() - 0.5) * 6.0); // x in [-3, 3]
            xyz.push((rng.next_f64() - 0.5) * 4.0); // y in [-2, 2]
            xyz.push(2.0 + rng.next_f64() * 8.0); // z in [2, 10], in front
        }
        xyz
    }

    #[test]
    fn the_grid_agrees_with_the_brute_force_scan_on_random_points_and_pixels() {
        let mut rng = Xorshift(0xC0FF_EE12_3456_789A);
        let camera = test_camera();
        let xyz = random_cloud(&mut rng, 5_000);
        let grid = ScreenGrid::build(&camera, &xyz, ANCHOR_RADIUS_PX);

        let mut agreements = 0;
        let mut hits = 0;
        for _ in 0..500 {
            let px = (rng.next_f64() * 640.0, rng.next_f64() * 480.0);
            let brute = depth_anchor(&camera, &xyz, px, ANCHOR_RADIUS_PX);
            let indexed = grid.nearest(px, ANCHOR_RADIUS_PX);
            match (brute, indexed) {
                (Some(b), Some(i)) => {
                    assert_eq!(b, i, "grid must be exact, not approximate, at {px:?}");
                    hits += 1;
                    agreements += 1;
                }
                (None, None) => agreements += 1,
                (b, i) => panic!("brute {b:?} vs grid {i:?} disagree on presence at {px:?}"),
            }
        }
        assert_eq!(agreements, 500, "every query must agree with brute force");
        assert!(hits > 50, "the cloud must actually land under some queries: {hits}");
    }

    #[test]
    fn a_grid_built_on_an_empty_cloud_finds_nothing() {
        let grid = ScreenGrid::build(&test_camera(), &[], ANCHOR_RADIUS_PX);
        assert!(grid.is_empty());
        assert!(grid.nearest((320.0, 240.0), ANCHOR_RADIUS_PX).is_none());
    }

    #[test]
    fn the_grid_ignores_points_behind_the_camera_exactly_like_brute_force() {
        let camera = test_camera();
        // One point behind the camera, one in front, at the same pixel.
        let xyz = [0.0, 0.0, -5.0, 0.0, 0.0, 5.0];
        let grid = ScreenGrid::build(&camera, &xyz, ANCHOR_RADIUS_PX);
        let px = (camera.cx, camera.cy);
        assert_eq!(grid.nearest(px, ANCHOR_RADIUS_PX), Some(5.0));
        assert_eq!(depth_anchor(&camera, &xyz, px, ANCHOR_RADIUS_PX), Some(5.0));
    }
}
