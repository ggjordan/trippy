//! Click-to-cluster: a clicked pixel to a `pointset` selection, by 3D k-NN growth.
//!
//! Module: `trips_viewer::edit::cluster`
//! Purpose: `docs/EDITOR.md` §3 "1. Click-to-cluster (E4)", the viewer's half.
//!     A **port of the arithmetic** in `trippy/edit/cluster.py`, not a call into
//!     it: the four tool sliders (radius px, colour tolerance, max radius, max
//!     points) have to re-run at interaction speed with no subprocess in the
//!     loop, exactly as [`super::shade`] does for the shade finder. The
//!     algorithm, step for step with the Python module's own docstring:
//!
//! 1. Project every point into the CURRENT camera ([`ClickCamera`], a pinhole
//!    `world_to_cam` + `project_pinhole`; no distortion — see the invariants).
//! 2. `candidates` = points landing within `radius_px` of the click AND in
//!    front of the camera (`depth > 0`).
//! 3. Cluster the candidates' camera-space depth by gap; the group nearest the
//!    camera is the SEED depth mode. This is what stops a click from selecting
//!    through a gap onto a same-coloured surface behind.
//! 4. `seed` = the candidates in that nearest mode.
//! 5. Grow from the seed by repeated k-NN queries in 3D ([`PointGrid`]),
//!    admitting a neighbour only if its base colour is within `colour_tol` of
//!    the seed's own mean colour AND it is within `max_radius` of the seed
//!    centroid. Stop when the frontier is exhausted or `max_points` is reached.
//!
//! Invariants:
//!     - Nothing here references Burn, wgpu, egui or eframe (the [`super`]
//!       module's first invariant). The camera arrives as plain numbers.
//!     - **No lens distortion is applied**, exactly as the Python side applies
//!       none: `trippy.geom.xform_a` has no distortion model at all. Exact on a
//!       trippy-native bundle (all-zero coefficients by construction); a
//!       first-order approximation of "which points are near this click" on a
//!       TRIPS/ADOP bundle. Recorded rather than silently diverging from the
//!       twin this is pinned against.
//!     - **The neighbour search is a spatial hash, not a k-d tree.** No k-d
//!       tree crate is vendored by either workspace's lock
//!       (`rust/Cargo.lock`, `rust/brush-trips/Cargo.lock`), and adding a
//!       dependency to compare against `scipy.spatial.cKDTree` would be a
//!       larger change than writing the 120 lines below. [`PointGrid::nearest`]
//!       is an EXACT k-nearest query — it expands whole cell rings until the
//!       k-th distance found is provably inside the scanned region — so it
//!       returns the same neighbour set cKDTree does whenever that set is
//!       unambiguous. See "Tie-breaking" below for when it is not.
//!     - Base colour is `clip(feat[:, :3], 0, 1)`, the same slice
//!       `trippy.edit.shade_finder` and [`super::shade`] read.
//!     - The returned ids index the cloud's OWN row order (`points.npz`'s), the
//!       `pointset` contract in `docs/EDITOR.md` §1.
//!
//! # Tie-breaking, and where the two implementations may legitimately differ
//!
//! Three places have an arbitrary answer that the two languages need not pick
//! identically. All three are documented rather than papered over:
//!
//! 1. **k-th nearest neighbour ties.** When the k-th and (k+1)-th neighbours
//!    are at *exactly* the same distance, `cKDTree` returns whichever its own
//!    build order reached first; this module returns the lower point index
//!    (its candidate sort is by `(distance, index)`). A tie is a measure-zero
//!    event on float coordinates and cannot happen at all in the committed
//!    fixture, whose points come from a continuous distribution.
//! 2. **Depth sort ties.** `numpy.argsort`'s default is *not* stable; this
//!    module sorts by `(depth, index)`. The two agree unless two candidates
//!    share a depth bit for bit, and even then they only disagree when the tie
//!    straddles the mode's own cut.
//! 3. **Summation order.** The seed centroid and mean colour are sums over the
//!    seed; numpy sums pairwise, this module sums in index order, so the two
//!    can differ in the last ulp. That only changes an answer for a point
//!    sitting within ~1e-15 of the `max_radius` or `colour_tol` boundary.
//!
//! None of the three is reachable in `tests/fixtures/synthetic/edit_golden/`,
//! whose margins are checked by `tests/test_edit_golden.py`
//! (`test_each_click_case_pins_a_different_branch`); the golden test at the
//! bottom of [`super`] therefore asserts an EXACT id-for-id match.
//!
//! Units: `xyz` and `max_radius` are world units (COLMAP world frame,
//!     `docs/GEOMETRY.md`); `radius_px` is screen pixels; `colour_tol` is a
//!     Euclidean distance in `[0, 1]^3` colour space.
//! Related docs: `docs/EDITOR.md` §3 and §6 (E4); `trippy/edit/cluster.py`;
//!     `trippy/constants.py` (every default below).

use std::collections::HashSet;

/// `"format"` of the click fixture, matching `trippy.edit.golden.CLICK_FIXTURE_FORMAT`.
pub const CLICK_FIXTURE_FORMAT: &str = "trippy-edit-click-1";

/// `trippy.constants.CLICK_DEFAULT_RADIUS_PX`.
pub const DEFAULT_RADIUS_PX: f64 = 12.0;
/// `trippy.constants.CLICK_DEFAULT_COLOUR_TOL`.
pub const DEFAULT_COLOUR_TOL: f64 = 0.15;
/// `trippy.constants.CLICK_DEFAULT_MAX_POINTS`.
pub const DEFAULT_MAX_POINTS: usize = 200_000;
/// `trippy.constants.CLICK_GROW_KNN_K`.
pub const GROW_KNN_K: usize = 16;
/// `trippy.constants.CLICK_DEFAULT_DEPTH_GAP_FACTOR`.
pub const DEFAULT_DEPTH_GAP_FACTOR: f64 = 1.0;
/// `trippy.constants.CLICK_DEFAULT_MAX_RADIUS_CAMERA_FACTOR`.
pub const DEFAULT_MAX_RADIUS_CAMERA_FACTOR: f64 = 1.0;
/// `trippy.constants.CLICK_FALLBACK_MAX_RADIUS_FACTOR`.
pub const FALLBACK_MAX_RADIUS_FACTOR: f64 = 50.0;
/// `trippy.constants.CLICK_DEFAULT_MIX`.
pub const DEFAULT_MIX: f64 = 0.5;
/// `trippy.constants.SUMMARY_NN_SAMPLE` — the point count above which
/// `trippy.points.knn_size.median_nn_distance` starts drawing a random
/// subsample (and so stops having a portable twin).
pub const SUMMARY_NN_SAMPLE: usize = 20_000;

/// Points per grid cell the spatial hash aims for. One is the usual choice for
/// a uniform cloud: fewer means more ring expansions, more means more distance
/// tests per ring.
const GRID_POINTS_PER_CELL: f64 = 1.0;

/// Buckets per point in the spatial hash's table. Two keeps the average chain
/// short without a resize path; collisions cost extra distance tests and never
/// correctness, because a bucket is always scanned as a superset of the cell.
const GRID_BUCKETS_PER_POINT: usize = 2;

/// How many cell rings a k-nearest query expands before giving up and scanning
/// the whole cloud.
///
/// Ring `r` visits `(2r + 1)^3 - (2r - 1)^3` cells, so the cost of expanding is
/// cubic while the cost of a linear scan is not. Eight rings is 4913 cells,
/// far more than a cloud with ~1 point per cell ever needs for `k = 16`; past
/// that the query is in a void (an isolated far-field point) and a direct scan
/// is both cheaper and still exact.
const MAX_RING_EXPANSIONS: i64 = 8;

// --- camera ---------------------------------------------------------------------------

/// The pieces of a camera a click projection needs, widened to `f64`.
///
/// Built from the frame's own `brush_pyramid::scene::Camera` (which is `f32`)
/// by [`Self::from_render_camera`], or straight from JSON by the golden test.
/// Distortion is deliberately absent — see the module invariants.
#[derive(Debug, Clone, PartialEq)]
pub struct ClickCamera {
    /// Row-major `3x3` world-to-camera rotation (`x_cam = R x_world + t`).
    pub r: [f64; 9],
    /// World-to-camera translation, world units.
    pub t: [f64; 3],
    /// Focal length in pixels along `x`.
    pub fx: f64,
    /// Focal length in pixels along `y`.
    pub fy: f64,
    /// Principal point `x`, pixels.
    pub cx: f64,
    /// Principal point `y`, pixels.
    pub cy: f64,
}

impl ClickCamera {
    /// Widen the camera the frame was rendered with.
    ///
    /// The renderer's camera is `f32`; the Python twin is `f64` throughout.
    /// Widening here means the projection arithmetic matches, while the INPUTS
    /// still carry whatever precision the render camera had — which is exactly
    /// the honest statement: a click in the window is as accurate as the
    /// camera that drew the frame, and the golden fixture (whose camera is
    /// `f64` from JSON) is what pins the arithmetic itself.
    #[must_use]
    pub fn from_render_camera(camera: &brush_pyramid::scene::Camera) -> Self {
        let mut r = [0.0_f64; 9];
        for (dst, src) in r.iter_mut().zip(camera.r.iter()) {
            *dst = f64::from(*src);
        }
        Self {
            r,
            t: [
                f64::from(camera.t[0]),
                f64::from(camera.t[1]),
                f64::from(camera.t[2]),
            ],
            fx: f64::from(camera.fx),
            fy: f64::from(camera.fy),
            cx: f64::from(camera.cx),
            cy: f64::from(camera.cy),
        }
    }

    /// Parse the `"camera"` block of `click.json`.
    ///
    /// # Errors
    /// Returns `Err` naming the first field that is missing or the wrong shape.
    pub fn from_json(doc: &serde_json::Value) -> Result<Self, String> {
        let num = |key: &str| -> Result<f64, String> {
            doc.get(key)
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| format!("click camera: {key:?} must be a number"))
        };
        let vec = |key: &str, n: usize| -> Result<Vec<f64>, String> {
            let array = doc
                .get(key)
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| format!("click camera: {key:?} must be an array of {n} numbers"))?;
            if array.len() != n {
                return Err(format!("click camera: {key:?} must have {n} numbers"));
            }
            array
                .iter()
                .map(|v| {
                    v.as_f64()
                        .ok_or_else(|| format!("click camera: {key:?} must be numbers"))
                })
                .collect()
        };
        let r = vec("r", 9)?;
        let t = vec("t", 3)?;
        Ok(Self {
            r: [r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]],
            t: [t[0], t[1], t[2]],
            fx: num("fx")?,
            fy: num("fy")?,
            cx: num("cx")?,
            cy: num("cy")?,
        })
    }

    /// Project one world point: `(u, v, depth)`, depth positive in front.
    ///
    /// `trippy.geom.xform_a.world_to_cam` followed by `project_pinhole`, in the
    /// same operation order (`fx * x / z + cx`), so the two agree bit for bit
    /// on the same inputs.
    #[must_use]
    pub fn project(&self, p: [f64; 3]) -> (f64, f64, f64) {
        let cam = [
            self.r[0] * p[0] + self.r[1] * p[1] + self.r[2] * p[2] + self.t[0],
            self.r[3] * p[0] + self.r[4] * p[1] + self.r[5] * p[2] + self.t[1],
            self.r[6] * p[0] + self.r[7] * p[1] + self.r[8] * p[2] + self.t[2],
        ];
        let z = cam[2];
        (
            self.fx * cam[0] / z + self.cx,
            self.fy * cam[1] / z + self.cy,
            z,
        )
    }

    /// The world point at pixel `(u, v)` and camera-space depth `z`.
    ///
    /// The exact inverse of [`Self::project`] (`x_world = R^T (x_cam - t)`).
    /// Viewer-only — there is no Python twin, because the Python side never
    /// has a live camera to un-project through — and it exists for the brush
    /// tool, which turns "this pixel, at the depth of the nearest point under
    /// it" into the world centre of a stroke (`docs/EDITOR.md` §4, the brush).
    #[must_use]
    pub fn unproject(&self, px: (f64, f64), depth: f64) -> [f64; 3] {
        let cam = [
            (px.0 - self.cx) * depth / self.fx - self.t[0],
            (px.1 - self.cy) * depth / self.fy - self.t[1],
            depth - self.t[2],
        ];
        [
            self.r[0] * cam[0] + self.r[3] * cam[1] + self.r[6] * cam[2],
            self.r[1] * cam[0] + self.r[4] * cam[1] + self.r[7] * cam[2],
            self.r[2] * cam[0] + self.r[5] * cam[1] + self.r[8] * cam[2],
        ]
    }

    /// The camera centre in world coordinates, `C = -R^T t`
    /// (`trippy.edit.cluster.camera_center`).
    #[must_use]
    pub fn centre(&self) -> [f64; 3] {
        [
            -(self.r[0] * self.t[0] + self.r[3] * self.t[1] + self.r[6] * self.t[2]),
            -(self.r[1] * self.t[0] + self.r[4] * self.t[1] + self.r[7] * self.t[2]),
            -(self.r[2] * self.t[0] + self.r[5] * self.t[1] + self.r[8] * self.t[2]),
        ]
    }
}

// --- the neighbour search -------------------------------------------------------------

/// An exact k-nearest-neighbour index over a static point cloud: a spatial hash.
///
/// Built once per bundle (`points.npz` never changes during a session,
/// `docs/EDITOR.md` §3) and reused by every click. See the module invariants
/// for why this is a hash and not a k-d tree.
///
/// The cell size is derived from the cloud's **interquartile** extent rather
/// than its bounding box: a TRIPS export's far-field environment sphere makes
/// the bounding box thousands of units across (`renderer.rs`'s `bounds` field
/// says so), and sizing cells from it would drop the whole scene into one cell
/// and turn every query into a linear scan.
pub struct PointGrid {
    /// Cell edge length, world units.
    cell: f64,
    /// Bucket count of the hash table.
    buckets: usize,
    /// `starts[b]..starts[b + 1]` indexes [`Self::order`] for bucket `b`.
    starts: Vec<u32>,
    /// Point indices, grouped by bucket.
    order: Vec<u32>,
}

/// Multipliers for the cell hash: three large odd primes, the usual choice for
/// a 3D lattice hash (Teschner et al., "Optimized spatial hashing").
const HASH_PRIMES: [i64; 3] = [73_856_093, 19_349_663, 83_492_791];

impl PointGrid {
    /// Build the index over a flat `(N, 3)` world-position array.
    ///
    /// # Panics
    /// Panics if `xyz.len()` is not a multiple of 3.
    #[must_use]
    pub fn build(xyz: &[f64]) -> Self {
        assert!(xyz.len() % 3 == 0, "xyz must be a flat (N, 3) array");
        let n = xyz.len() / 3;
        let cell = Self::cell_size(xyz);
        let buckets = (n * GRID_BUCKETS_PER_POINT).max(1);

        let mut counts = vec![0_u32; buckets + 1];
        let mut bucket_of = vec![0_u32; n];
        for i in 0..n {
            let key = Self::cell_of(cell, [xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]]);
            let b = Self::bucket(key, buckets);
            #[allow(clippy::cast_possible_truncation)]
            {
                bucket_of[i] = b as u32;
            }
            counts[b + 1] += 1;
        }
        for b in 0..buckets {
            counts[b + 1] += counts[b];
        }
        let starts = counts.clone();
        let mut cursor = counts;
        let mut order = vec![0_u32; n];
        for i in 0..n {
            let b = bucket_of[i] as usize;
            order[cursor[b] as usize] = u32::try_from(i).unwrap_or(u32::MAX);
            cursor[b] += 1;
        }
        Self {
            cell,
            buckets,
            starts,
            order,
        }
    }

    /// How many points the index holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// True when the index holds no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Cell edge length, world units — exposed for the panel's diagnostics.
    #[must_use]
    pub const fn cell_size_used(&self) -> f64 {
        self.cell
    }

    /// Pick the cell edge from the cloud's interquartile extent (see the type doc).
    fn cell_size(xyz: &[f64]) -> f64 {
        let n = xyz.len() / 3;
        if n == 0 {
            return 1.0;
        }
        let mut span = 0.0_f64;
        for axis in 0..3 {
            let mut values: Vec<f64> = (0..n).map(|i| xyz[3 * i + axis]).collect();
            values.sort_by(f64::total_cmp);
            let lo = values[n / 4];
            let hi = values[(3 * n) / 4];
            span = span.max(hi - lo);
        }
        #[allow(clippy::cast_precision_loss)]
        let per_axis = (n as f64 / GRID_POINTS_PER_CELL).cbrt().max(1.0);
        let cell = span / per_axis;
        if cell.is_finite() && cell > 0.0 {
            cell
        } else {
            // A degenerate cloud (every point identical, or a single point):
            // any positive cell size is correct, because every query then
            // scans the one occupied cell.
            1.0
        }
    }

    /// The integer cell a world position falls in.
    #[allow(clippy::cast_possible_truncation)]
    fn cell_of(cell: f64, p: [f64; 3]) -> [i64; 3] {
        [
            (p[0] / cell).floor() as i64,
            (p[1] / cell).floor() as i64,
            (p[2] / cell).floor() as i64,
        ]
    }

    /// Hash a cell into a bucket.
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn bucket(key: [i64; 3], buckets: usize) -> usize {
        let mixed = (key[0].wrapping_mul(HASH_PRIMES[0]))
            ^ (key[1].wrapping_mul(HASH_PRIMES[1]))
            ^ (key[2].wrapping_mul(HASH_PRIMES[2]));
        (mixed.rem_euclid(buckets as i64)) as usize
    }

    /// The `k` nearest points to `query`, nearest first.
    ///
    /// Exact: cell rings are expanded outward until the k-th distance found is
    /// no greater than `ring * cell`, which is a lower bound on the distance to
    /// anything still unscanned (a point in a cell at Chebyshev ring `ring + 1`
    /// or beyond differs by at least `ring` whole cells along some axis).
    ///
    /// Ties are broken by ascending point index — see the module's
    /// "Tie-breaking" section.
    ///
    /// # Arguments
    /// - `xyz`: the same flat array [`Self::build`] indexed.
    /// - `query`: the world position to search around.
    /// - `k`: how many neighbours to return (clamped to the cloud's size).
    #[must_use]
    pub fn nearest(&self, xyz: &[f64], query: [f64; 3], k: usize) -> Vec<u32> {
        let k = k.min(self.len());
        if k == 0 {
            return Vec::new();
        }
        let home = Self::cell_of(self.cell, query);
        let mut found: Vec<(f64, u32)> = Vec::with_capacity(k * 4);
        let mut ring: i64 = 0;
        loop {
            self.scan_ring(xyz, query, home, ring, &mut found);
            if found.len() >= k {
                found.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                found.truncate(k);
                #[allow(clippy::cast_precision_loss)]
                let safe = ring as f64 * self.cell;
                if found[k - 1].0.sqrt() <= safe {
                    return found.into_iter().map(|(_, i)| i).collect();
                }
            }
            ring += 1;
            if ring > MAX_RING_EXPANSIONS {
                // The query sits in a void: an isolated far-field point, or a
                // cloud so clustered that its interquartile cell size means
                // nothing here. Expanding further would scan O(ring^3) empty
                // cells, so answer the question directly instead. Still exact.
                return Self::brute_force(xyz, query, k);
            }
        }
    }

    /// The k nearest by a linear scan — the fallback [`Self::nearest`] takes in a void.
    fn brute_force(xyz: &[f64], query: [f64; 3], k: usize) -> Vec<u32> {
        let n = xyz.len() / 3;
        let mut all: Vec<(f64, u32)> = (0..n)
            .map(|i| {
                let d = [
                    xyz[3 * i] - query[0],
                    xyz[3 * i + 1] - query[1],
                    xyz[3 * i + 2] - query[2],
                ];
                (
                    d[0] * d[0] + d[1] * d[1] + d[2] * d[2],
                    u32::try_from(i).unwrap_or(u32::MAX),
                )
            })
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        all.truncate(k);
        all.into_iter().map(|(_, i)| i).collect()
    }

    /// Push every point of the cells at Chebyshev distance exactly `ring`.
    fn scan_ring(
        &self,
        xyz: &[f64],
        query: [f64; 3],
        home: [i64; 3],
        ring: i64,
        out: &mut Vec<(f64, u32)>,
    ) {
        let mut visit = |key: [i64; 3]| {
            let b = Self::bucket(key, self.buckets);
            let (lo, hi) = (self.starts[b] as usize, self.starts[b + 1] as usize);
            for &i in &self.order[lo..hi] {
                let row = i as usize;
                // A bucket is a superset of one cell (hash collisions), so the
                // cell test keeps a point from being scanned twice, in two
                // different rings, and counted twice in `found`.
                if Self::cell_of(
                    self.cell,
                    [xyz[3 * row], xyz[3 * row + 1], xyz[3 * row + 2]],
                ) != key
                {
                    continue;
                }
                let d = [
                    xyz[3 * row] - query[0],
                    xyz[3 * row + 1] - query[1],
                    xyz[3 * row + 2] - query[2],
                ];
                out.push((d[0] * d[0] + d[1] * d[1] + d[2] * d[2], i));
            }
        };
        if ring == 0 {
            visit(home);
            return;
        }
        for dx in -ring..=ring {
            for dy in -ring..=ring {
                for dz in -ring..=ring {
                    if dx.abs() != ring && dy.abs() != ring && dz.abs() != ring {
                        continue;
                    }
                    visit([home[0] + dx, home[1] + dy, home[2] + dz]);
                }
            }
        }
    }
}

// --- the tool's own parameters ---------------------------------------------------------

/// The four sliders the Selection panel shows, plus the two constants behind them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClickParams {
    /// Click catchment radius, screen pixels.
    pub radius_px: f64,
    /// Max Euclidean colour distance in `[0, 1]^3` from the seed's mean colour.
    pub colour_tol: f64,
    /// Max distance (world units) from the seed centroid a grown point may have.
    pub max_radius: f64,
    /// Hard cap on the selection's size.
    pub max_points: usize,
    /// Neighbours queried per growth step.
    pub knn_k: usize,
    /// Multiplies `max_radius` to get the depth-mode gap threshold.
    pub depth_gap_factor: f64,
}

impl Default for ClickParams {
    fn default() -> Self {
        Self {
            radius_px: DEFAULT_RADIUS_PX,
            colour_tol: DEFAULT_COLOUR_TOL,
            // 0 is not a usable radius; the session replaces it with
            // `default_max_radius` as soon as it knows the bundle's cameras.
            max_radius: 0.0,
            max_points: DEFAULT_MAX_POINTS,
            knn_k: GROW_KNN_K,
            depth_gap_factor: DEFAULT_DEPTH_GAP_FACTOR,
        }
    }
}

/// What one click selected, and the numbers the panel reports.
#[derive(Debug, Clone, Default)]
pub struct ClickSelection {
    /// Indices into the cloud's own row order — a `pointset` region's `point_ids`.
    pub point_ids: Vec<u32>,
    /// Points that projected within `radius_px` and in front of the camera.
    pub n_candidates: usize,
    /// Candidates in the nearest depth mode (the seed).
    pub n_seed: usize,
    /// True when growth stopped because `max_points` was reached.
    pub hit_max_points: bool,
    /// Mean camera-space depth of the seed, world units; `None` on a miss.
    pub seed_depth_mean: Option<f64>,
    /// Set (and the selection empty) when nothing projected near the click.
    pub warning: Option<String>,
}

/// The boolean mask over `depth` selecting its nearest-camera contiguous run.
///
/// `trippy.edit.cluster._nearest_depth_mode_mask`: sort ascending, cut at the
/// first gap wider than `gap_threshold`, keep everything before the cut.
///
/// Sorting is by `(depth, index)` where numpy's is unstable — see the module's
/// "Tie-breaking" section.
#[must_use]
pub fn nearest_depth_mode_mask(depth: &[f64], gap_threshold: f64) -> Vec<bool> {
    let mut order: Vec<usize> = (0..depth.len()).collect();
    order.sort_by(|a, b| depth[*a].total_cmp(&depth[*b]).then_with(|| a.cmp(b)));
    let mut end = order.len();
    for i in 1..order.len() {
        if depth[order[i]] - depth[order[i - 1]] > gap_threshold {
            end = i;
            break;
        }
    }
    let mut mask = vec![false; depth.len()];
    for &i in &order[..end] {
        mask[i] = true;
    }
    mask
}

/// Click `px` in `camera`'s view; grow a selection from the nearest surface there.
///
/// The twin of `trippy.edit.cluster.click_to_cluster`, minus the `Region`
/// wrapper (the caller decides the op and the mix — `docs/EDITOR.md` §4's
/// "Add as region").
///
/// # Arguments
/// - `grid`: [`PointGrid::build`] over the same `xyz`.
/// - `xyz`: flat `(N, 3)` world positions; the returned ids index this.
/// - `rgb`: flat `(N, 3)` base colour, already clipped to `[0, 1]`.
/// - `camera`: the view the click was made in.
/// - `px`: `(u, v)` clicked pixel, in that camera's own pixel coordinates.
/// - `params`: the tool's sliders.
///
/// # Panics
/// Panics if `xyz` / `rgb` are not flat `(N, 3)` arrays of the same `N`, or if
/// `grid` was built over a different cloud.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn click_to_cluster(
    grid: &PointGrid,
    xyz: &[f64],
    rgb: &[f64],
    camera: &ClickCamera,
    px: (f64, f64),
    params: &ClickParams,
) -> ClickSelection {
    assert!(xyz.len() % 3 == 0, "xyz must be a flat (N, 3) array");
    assert!(rgb.len() == xyz.len(), "rgb must be (N, 3) alongside xyz");
    assert!(grid.len() == xyz.len() / 3, "the grid indexes another cloud");
    let n = xyz.len() / 3;

    let mut candidates: Vec<usize> = Vec::new();
    let mut candidate_depth: Vec<f64> = Vec::new();
    for i in 0..n {
        let (u, v, depth) = camera.project([xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]]);
        if depth > 0.0 && (u - px.0).hypot(v - px.1) <= params.radius_px {
            candidates.push(i);
            candidate_depth.push(depth);
        }
    }

    let mut out = ClickSelection {
        n_candidates: candidates.len(),
        ..ClickSelection::default()
    };
    if candidates.is_empty() {
        out.warning = Some("no points projected within radius_px of the click".to_owned());
        return out;
    }

    let seed_mask = nearest_depth_mode_mask(
        &candidate_depth,
        params.depth_gap_factor * params.max_radius,
    );
    let seed: Vec<usize> = candidates
        .iter()
        .zip(&seed_mask)
        .filter_map(|(i, keep)| keep.then_some(*i))
        .collect();
    out.n_seed = seed.len();

    let mut centroid = [0.0_f64; 3];
    let mut colour = [0.0_f64; 3];
    let mut depth_sum = 0.0_f64;
    for (idx, keep) in candidates.iter().zip(&seed_mask) {
        if !keep {
            continue;
        }
        for c in 0..3 {
            centroid[c] += xyz[3 * idx + c];
            colour[c] += rgb[3 * idx + c];
        }
    }
    for (d, keep) in candidate_depth.iter().zip(&seed_mask) {
        if *keep {
            depth_sum += *d;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let count = seed.len() as f64;
    for c in 0..3 {
        centroid[c] /= count;
        colour[c] /= count;
    }
    out.seed_depth_mean = Some(depth_sum / count);

    // The flood fill, frontier by frontier, exactly as the Python loop runs it:
    // every frontier point's k nearest are gathered, deduplicated, and visited
    // in ASCENDING INDEX ORDER, which is what makes the `max_points` cut-off
    // reproducible across the two implementations.
    let mut selected: HashSet<u32> = seed
        .iter()
        .map(|i| u32::try_from(*i).unwrap_or(u32::MAX))
        .collect();
    let mut frontier: Vec<usize> = seed.clone();
    frontier.sort_unstable();
    let k_eff = params.knn_k.min(n);

    'growth: while !frontier.is_empty() && selected.len() < params.max_points {
        let mut neighbours: Vec<u32> = Vec::new();
        for &row in &frontier {
            let q = [xyz[3 * row], xyz[3 * row + 1], xyz[3 * row + 2]];
            neighbours.extend(grid.nearest(xyz, q, k_eff));
        }
        neighbours.sort_unstable();
        neighbours.dedup();

        let mut next: Vec<usize> = Vec::new();
        for id in neighbours {
            if selected.contains(&id) {
                continue;
            }
            let row = id as usize;
            let dc = [
                rgb[3 * row] - colour[0],
                rgb[3 * row + 1] - colour[1],
                rgb[3 * row + 2] - colour[2],
            ];
            if (dc[0] * dc[0] + dc[1] * dc[1] + dc[2] * dc[2]).sqrt() > params.colour_tol {
                continue;
            }
            let dp = [
                xyz[3 * row] - centroid[0],
                xyz[3 * row + 1] - centroid[1],
                xyz[3 * row + 2] - centroid[2],
            ];
            if (dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2]).sqrt() > params.max_radius {
                continue;
            }
            selected.insert(id);
            next.push(row);
            if selected.len() >= params.max_points {
                out.hit_max_points = true;
                break 'growth;
            }
        }
        frontier = next;
    }

    let mut point_ids: Vec<u32> = selected.into_iter().collect();
    point_ids.sort_unstable();
    out.point_ids = point_ids;
    out
}

/// The click tool's default `max_radius`: the bundle's median nearest-CAMERA spacing.
///
/// `trippy.edit.cluster.default_max_radius_from_bundle`. Camera baseline, not
/// point-cloud density: see `trippy.constants.
/// CLICK_DEFAULT_MAX_RADIUS_CAMERA_FACTOR`'s own comment for why.
///
/// # Arguments
/// - `centres`: every view's world-frame camera centre, flat `(V, 3)`.
/// - `xyz`: the cloud, used only for the single-camera fallback.
///
/// # Returns
/// A positive world-unit radius, or `0.0` only when neither cameras nor points
/// give any scale at all (fewer than two of each) — the caller must then ask
/// the user for a number rather than pretend to have one.
///
/// The fallback branch is **not** parity-checked against Python: with more than
/// [`SUMMARY_NN_SAMPLE`] points `median_nn_distance` draws a seeded *numpy*
/// subsample, which has no portable twin. It is reachable only on a
/// single-view bundle, and the panel shows the number it used.
#[must_use]
pub fn default_max_radius(centres: &[f64], xyz: &[f64]) -> f64 {
    // `>= 2` camera centres, the same test the Python side makes on `views`.
    if centres.len() / 3 >= 2 {
        let spacing = median_nn_distance(centres);
        if spacing > 0.0 {
            return DEFAULT_MAX_RADIUS_CAMERA_FACTOR * spacing;
        }
    }
    FALLBACK_MAX_RADIUS_FACTOR * median_nn_distance_sampled(xyz)
}

/// Median nearest-neighbour distance over a small flat `(N, 3)` array, brute force.
///
/// `trippy.points.knn_size.median_nn_distance` with no subsampling — which is
/// exactly what that function does when `N <= SUMMARY_NN_SAMPLE`, because its
/// `rng.choice(n, size=n, replace=False)` is then a permutation and the median
/// does not care about order. Intended for camera centres (hundreds), never for
/// a point cloud.
#[must_use]
pub fn median_nn_distance(xyz: &[f64]) -> f64 {
    assert!(xyz.len() % 3 == 0, "xyz must be a flat (N, 3) array");
    let n = xyz.len() / 3;
    if n < 2 {
        return 0.0;
    }
    let mut nearest = Vec::with_capacity(n);
    for i in 0..n {
        let mut best = f64::INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = [
                xyz[3 * i] - xyz[3 * j],
                xyz[3 * i + 1] - xyz[3 * j + 1],
                xyz[3 * i + 2] - xyz[3 * j + 2],
            ];
            best = best.min(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        }
        nearest.push(best.sqrt());
    }
    median(&mut nearest)
}

/// [`median_nn_distance`] over a deterministic stride subsample of a big cloud.
///
/// A stride, not an RNG draw: the fallback has no Python twin to match anyway
/// (see [`default_max_radius`]), and a stride is at least reproducible.
fn median_nn_distance_sampled(xyz: &[f64]) -> f64 {
    let n = xyz.len() / 3;
    if n < 2 {
        return 0.0;
    }
    if n <= SUMMARY_NN_SAMPLE {
        return median_nn_distance(xyz);
    }
    let stride = n.div_ceil(SUMMARY_NN_SAMPLE);
    let mut sub = Vec::with_capacity(3 * SUMMARY_NN_SAMPLE);
    let mut i = 0;
    while i < n {
        sub.extend_from_slice(&xyz[3 * i..3 * i + 3]);
        i += stride;
    }
    // Brute force over 20 000 points is 4e8 distance tests; the grid answers the
    // same question in a fraction of that, and this runs once per bundle.
    let grid = PointGrid::build(&sub);
    let count = sub.len() / 3;
    let mut nearest = Vec::with_capacity(count);
    for row in 0..count {
        let q = [sub[3 * row], sub[3 * row + 1], sub[3 * row + 2]];
        let ids = grid.nearest(&sub, q, 2);
        if let Some(&other) = ids.iter().find(|id| **id as usize != row) {
            let o = other as usize;
            let d = [
                sub[3 * row] - sub[3 * o],
                sub[3 * row + 1] - sub[3 * o + 1],
                sub[3 * row + 2] - sub[3 * o + 2],
            ];
            nearest.push((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
        }
    }
    median(&mut nearest)
}

/// `numpy.median`: the middle value, or the mean of the two middle ones.
fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        f64::midpoint(values[mid - 1], values[mid])
    } else {
        values[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A camera at the world origin looking down `+Z`, 500 px focal, 640x480 —
    /// the same one `tests/test_edit_cluster.py` uses.
    fn camera() -> ClickCamera {
        ClickCamera {
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            t: [0.0, 0.0, 0.0],
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
        }
    }

    /// A deterministic little LCG, so the tests need no rand dependency.
    struct Lcg(u64);

    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            #[allow(clippy::cast_precision_loss)]
            {
                (self.0 >> 11) as f64 / (1_u64 << 53) as f64
            }
        }

        /// Roughly normal, by the sum of 6 uniforms minus 3.
        fn next_normal(&mut self) -> f64 {
            (0..6).map(|_| self.next_f64()).sum::<f64>() - 3.0
        }
    }

    fn blob(rng: &mut Lcg, centre: [f64; 3], scale: f64, count: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(count * 3);
        for _ in 0..count {
            for c in 0..3 {
                out.push(centre[c] + scale * rng.next_normal());
            }
        }
        out
    }

    #[test]
    fn the_projection_is_the_pinhole_python_uses() {
        let c = camera();
        let (u, v, z) = c.project([0.0, 0.0, 5.0]);
        assert!((u - 320.0).abs() < 1e-12 && (v - 240.0).abs() < 1e-12);
        assert!((z - 5.0).abs() < 1e-12);
        // x = 0.05 at z = 5 is 5 px right of the principal point.
        let (u, _, _) = c.project([0.05, 0.0, 5.0]);
        assert!((u - 325.0).abs() < 1e-12);
    }

    #[test]
    fn a_camera_centre_is_minus_r_transpose_t() {
        // A camera translated so its centre sits at (1, 2, 3).
        let c = ClickCamera {
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            t: [-1.0, -2.0, -3.0],
            ..camera()
        };
        let centre = c.centre();
        assert!((centre[0] - 1.0).abs() < 1e-12);
        assert!((centre[1] - 2.0).abs() < 1e-12);
        assert!((centre[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn the_grid_finds_exactly_the_k_nearest_a_brute_force_search_does() {
        let mut rng = Lcg(7);
        let mut xyz = blob(&mut rng, [0.0, 0.0, 5.0], 0.4, 200);
        xyz.extend(blob(&mut rng, [3.0, 1.0, 9.0], 0.2, 120));
        // A far-field outlier, the environment-sphere case the cell sizing
        // is designed to survive.
        xyz.extend_from_slice(&[900.0, -900.0, 900.0]);
        let grid = PointGrid::build(&xyz);
        let n = xyz.len() / 3;
        assert_eq!(grid.len(), n);

        for row in [0_usize, 5, 199, 200, 260, n - 1] {
            let q = [xyz[3 * row], xyz[3 * row + 1], xyz[3 * row + 2]];
            let got = grid.nearest(&xyz, q, 8);
            let mut want: Vec<(f64, u32)> = (0..n)
                .map(|i| {
                    let d = [
                        xyz[3 * i] - q[0],
                        xyz[3 * i + 1] - q[1],
                        xyz[3 * i + 2] - q[2],
                    ];
                    #[allow(clippy::cast_possible_truncation)]
                    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2], i as u32)
                })
                .collect();
            want.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            let want: Vec<u32> = want.into_iter().take(8).map(|(_, i)| i).collect();
            assert_eq!(got, want, "row {row}");
        }
    }

    #[test]
    fn the_grid_survives_a_cloud_with_one_position_repeated() {
        let xyz = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let grid = PointGrid::build(&xyz);
        assert_eq!(grid.nearest(&xyz, [1.0, 2.0, 3.0], 2), vec![0, 1]);
        assert_eq!(grid.nearest(&xyz, [1.0, 2.0, 3.0], 9).len(), 3);
    }

    #[test]
    fn the_depth_mode_cuts_at_the_first_gap_wider_than_the_threshold() {
        let depth = [5.0, 5.05, 20.0, 5.02, 20.1];
        let mask = nearest_depth_mode_mask(&depth, 1.0);
        assert_eq!(mask, vec![true, true, false, true, false]);
        // A threshold wider than every gap keeps everything.
        assert_eq!(nearest_depth_mode_mask(&depth, 100.0), vec![true; 5]);
        // A threshold of zero leaves only the single nearest point.
        let only = nearest_depth_mode_mask(&depth, 0.0);
        assert_eq!(only.iter().filter(|m| **m).count(), 1);
        assert!(only[0]);
    }

    #[test]
    fn a_click_selects_the_near_blob_and_not_the_same_coloured_one_behind_it() {
        let mut rng = Lcg(11);
        let front = blob(&mut rng, [0.0, 0.0, 5.0], 0.03, 150);
        let back = blob(&mut rng, [0.0, 0.0, 20.0], 0.03, 150);
        let mut xyz = front.clone();
        xyz.extend_from_slice(&back);
        // Both blobs the SAME colour, on purpose.
        let rgb: Vec<f64> = (0..xyz.len() / 3).flat_map(|_| [1.0, 0.0, 0.0]).collect();

        let grid = PointGrid::build(&xyz);
        let params = ClickParams {
            max_radius: 1.0,
            ..ClickParams::default()
        };
        let found = click_to_cluster(&grid, &xyz, &rgb, &camera(), (320.0, 240.0), &params);
        assert!(found.n_candidates > 0);
        assert!(!found.point_ids.is_empty());
        assert!(
            found.point_ids.iter().all(|i| (*i as usize) < 150),
            "the blob behind the gap must not be selected"
        );
        let depth = found.seed_depth_mean.expect("a hit reports its seed depth");
        assert!((depth - 5.0).abs() < 0.2, "seed depth {depth}");
    }

    #[test]
    fn the_colour_gate_separates_two_touching_clusters() {
        // Two rows of points on one line at z = 5, 0.02 apart: red for
        // x in [-0.30, -0.02], green for x in [0.00, 0.28]. Geometrically one
        // contiguous bar, so only colour can separate them.
        let mut xyz = Vec::new();
        let mut rgb = Vec::new();
        for i in 0..30 {
            #[allow(clippy::cast_precision_loss)]
            let x = -0.30 + i as f64 * 0.02;
            xyz.extend_from_slice(&[x, 0.0, 5.0]);
            if i < 15 {
                rgb.extend_from_slice(&[0.85, 0.15, 0.15]);
            } else {
                rgb.extend_from_slice(&[0.15, 0.80, 0.20]);
            }
        }
        let grid = PointGrid::build(&xyz);
        // Click on the middle of the red half: x = -0.16 lands on u = 304.
        let click = (304.0, 240.0);
        let tight = ClickParams {
            max_radius: 3.0,
            ..ClickParams::default()
        };
        let found = click_to_cluster(&grid, &xyz, &rgb, &camera(), click, &tight);
        assert_eq!(found.point_ids.len(), 15, "the red half, whole");
        assert!(
            found.point_ids.iter().all(|i| *i < 15),
            "the green half is a different object"
        );

        // Opening the gate swallows it: the two really are contiguous, so the
        // colour test is what separated them, not the geometry.
        let loose = ClickParams {
            colour_tol: 2.0,
            ..tight
        };
        let all = click_to_cluster(&grid, &xyz, &rgb, &camera(), click, &loose);
        assert_eq!(all.point_ids.len(), 30, "every point, once the gate is open");
    }

    #[test]
    fn max_points_caps_the_selection_and_says_so() {
        // A 300-point bar 0.02 apart at z = 5. At that depth 0.02 world is
        // 2 px, so a 2 px catchment seeds on a handful of points and the rest
        // has to be grown -- which is what makes the cap the thing under test.
        let mut xyz = Vec::new();
        for i in 0..300 {
            #[allow(clippy::cast_precision_loss)]
            let x = (i as f64 - 150.0) * 0.02;
            xyz.extend_from_slice(&[x, 0.0, 5.0]);
        }
        let rgb: Vec<f64> = (0..300).flat_map(|_| [0.5, 0.5, 0.5]).collect();
        let grid = PointGrid::build(&xyz);
        let params = ClickParams {
            radius_px: 2.0,
            max_radius: 10.0,
            max_points: 40,
            ..ClickParams::default()
        };
        let found = click_to_cluster(&grid, &xyz, &rgb, &camera(), (320.0, 240.0), &params);
        assert!(
            found.n_seed < 40,
            "the seed ({}) must leave room to grow",
            found.n_seed
        );
        assert_eq!(found.point_ids.len(), 40);
        assert!(found.hit_max_points);

        // Without the cap the same click takes the whole bar.
        let uncapped = ClickParams {
            max_points: DEFAULT_MAX_POINTS,
            ..params
        };
        let all = click_to_cluster(&grid, &xyz, &rgb, &camera(), (320.0, 240.0), &uncapped);
        assert_eq!(all.point_ids.len(), 300);
        assert!(!all.hit_max_points);
    }

    #[test]
    fn max_radius_is_a_hard_cap_measured_from_the_seed_centroid() {
        let mut rng = Lcg(19);
        // A long bar of points reaching away from the click.
        let mut xyz = Vec::new();
        for i in 0..200 {
            #[allow(clippy::cast_precision_loss)]
            let x = i as f64 * 0.02;
            xyz.extend_from_slice(&[x, 0.005 * rng.next_normal(), 5.0]);
        }
        let rgb: Vec<f64> = (0..200).flat_map(|_| [0.5, 0.5, 0.5]).collect();
        let grid = PointGrid::build(&xyz);
        let params = ClickParams {
            radius_px: 2.0,
            max_radius: 0.5,
            depth_gap_factor: 0.3,
            ..ClickParams::default()
        };
        let found = click_to_cluster(&grid, &xyz, &rgb, &camera(), (320.0, 240.0), &params);
        assert!(!found.point_ids.is_empty());
        for id in &found.point_ids {
            let row = *id as usize;
            assert!(
                xyz[3 * row] <= 0.5 + 1e-9,
                "point {row} at x = {} is past max_radius",
                xyz[3 * row]
            );
        }
    }

    #[test]
    fn a_click_on_nothing_is_an_empty_selection_with_a_warning() {
        let xyz = vec![0.0, 0.0, 5.0, 0.1, 0.0, 5.0];
        let rgb = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let grid = PointGrid::build(&xyz);
        let params = ClickParams {
            radius_px: 1.0,
            max_radius: 1.0,
            ..ClickParams::default()
        };
        let found = click_to_cluster(&grid, &xyz, &rgb, &camera(), (5.0, 5.0), &params);
        assert_eq!(found.n_candidates, 0);
        assert!(found.point_ids.is_empty());
        assert!(found.warning.is_some());
        assert!(found.seed_depth_mean.is_none());
    }

    #[test]
    fn a_point_behind_the_camera_is_never_a_candidate() {
        // Two points on the optical axis, one in front and one behind. The one
        // behind projects to the SAME pixel (both x and z negate), which is
        // exactly the trap the `depth > 0` test exists for.
        let xyz = vec![0.0, 0.0, 5.0, 0.0, 0.0, -5.0];
        let rgb = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let grid = PointGrid::build(&xyz);
        let params = ClickParams {
            max_radius: 1.0,
            ..ClickParams::default()
        };
        let found = click_to_cluster(&grid, &xyz, &rgb, &camera(), (320.0, 240.0), &params);
        assert_eq!(found.n_candidates, 1);
        assert_eq!(found.point_ids, vec![0]);
    }

    #[test]
    fn the_default_max_radius_is_the_median_camera_spacing() {
        // Four cameras on a line, 2 units apart: every nearest gap is 2.
        let centres = vec![
            0.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, //
            4.0, 0.0, 0.0, //
            6.0, 0.0, 0.0,
        ];
        let xyz = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.5];
        let radius = default_max_radius(&centres, &xyz);
        assert!((radius - 2.0).abs() < 1e-12, "radius {radius}");
    }

    #[test]
    fn a_single_camera_falls_back_to_the_point_clouds_own_spacing() {
        let centres = vec![0.0, 0.0, 0.0];
        // Points 0.1 apart: 50 * 0.1 = 5.
        let xyz = vec![0.0, 0.0, 1.0, 0.1, 0.0, 1.0, 0.2, 0.0, 1.0];
        let radius = default_max_radius(&centres, &xyz);
        assert!((radius - 5.0).abs() < 1e-9, "radius {radius}");
    }

    #[test]
    fn median_matches_numpys_even_length_rule() {
        assert!((median(&mut [1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-12);
        assert!((median(&mut [3.0, 1.0, 2.0]) - 2.0).abs() < 1e-12);
        assert!((median(&mut []) - 0.0).abs() < 1e-12);
    }
}
