//! Pyramid geometry and the flat layer-pixel index space.
//!
//! Module: `brush_pyramid::grid`
//! Purpose: port of `trippy.raster.emit.layer_grid` / `LayerGrid`. Turns an
//!     image size into the per-layer shapes and the layer-major flat index
//!     that the whole rasteriser sorts and segments on.
//! Invariants:
//!     - The flat index is **layer-major**: `offsets[l] + y * w_l + x`. That
//!       is what makes "sort by `layer_pixel`, then by depth" identical to
//!       "sort by (layer, pixel, depth)" with no composite key, and what lets
//!       [`LayerGrid::fragments_per_layer`] read per-layer counts straight off
//!       the segment table.
//!     - A configuration that would produce an empty layer is an error, not a
//!       silently dropped layer (matches the Python `ValueError`).
//! Units: all values are pixel counts.
//! Related docs: `docs/GEOMETRY.md` "Image pyramid (`pyramid_halving`)".

use crate::params::PyramidHalving;

/// Geometry of the image pyramid and of the flat layer-pixel index space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerGrid {
    /// `L` pairs of `(h_l, w_l)`, finest first.
    shapes: Vec<(usize, usize)>,
    /// `offsets[l]` is the first flat index belonging to layer `l`.
    offsets: Vec<usize>,
    /// Number of layer-pixels across the whole pyramid, `sum(h_l * w_l)`.
    total: usize,
}

impl LayerGrid {
    /// Build the pyramid geometry for a layer-0 image of `height x width`.
    ///
    /// # Arguments
    /// - `height`, `width`: layer-0 image size in pixels, both > 0.
    /// - `num_layers`: `L`, >= 1.
    /// - `halving`: see [`PyramidHalving`].
    ///
    /// # Errors
    /// Returns `Err` on a zero size, `num_layers == 0`, or a pyramid deep
    /// enough that a layer would come out empty.
    pub fn new(
        height: usize,
        width: usize,
        num_layers: usize,
        halving: PyramidHalving,
    ) -> Result<Self, String> {
        if height == 0 || width == 0 {
            return Err(format!("image size must be positive, got {height}x{width}"));
        }
        if num_layers == 0 {
            return Err("num_layers must be >= 1, got 0".to_owned());
        }
        let mut shapes = Vec::with_capacity(num_layers);
        let mut offsets = Vec::with_capacity(num_layers);
        let mut total = 0usize;
        for layer in 0..num_layers {
            let step = 1usize << layer;
            // `ceil` keeps a 1080-row image at 1080 rows through the U-Net;
            // `floor` is TRIPS's other branch. Repeated halving composes into
            // a single division either way.
            let (h_l, w_l) = match halving {
                PyramidHalving::Ceil => (height.div_ceil(step), width.div_ceil(step)),
                PyramidHalving::Floor => (height / step, width / step),
            };
            if h_l < 1 || w_l < 1 {
                return Err(format!(
                    "halving {halving:?} gives an empty layer {layer} ({h_l}x{w_l}) for a \
                     {height}x{width} image; reduce num_layers"
                ));
            }
            shapes.push((h_l, w_l));
            offsets.push(total);
            total += h_l * w_l;
        }
        Ok(Self {
            shapes,
            offsets,
            total,
        })
    }

    /// Per-layer `(h_l, w_l)` shapes, finest first.
    #[must_use]
    pub fn shapes(&self) -> &[(usize, usize)] {
        &self.shapes
    }

    /// First flat layer-pixel index of each layer.
    #[must_use]
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Total number of layer-pixels, `P`.
    #[must_use]
    pub fn total(&self) -> usize {
        self.total
    }

    /// `L`, the number of layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.shapes.len()
    }

    /// Flat index of pixel `(y, x)` in `layer`. No bounds checking — callers
    /// have already applied the drop rule.
    #[must_use]
    pub fn flat_index(&self, layer: usize, y: usize, x: usize) -> usize {
        self.offsets[layer] + y * self.shapes[layer].1 + x
    }

    /// Fragment count per layer, read off a `(P + 1,)` segment-offset table.
    ///
    /// The flat index is layer-major, so layer `l`'s fragments occupy the
    /// contiguous slice `[offsets[grid.offsets[l]], offsets[grid.offsets[l+1]])`
    /// of the sorted list — `L` lookups, no histogram over `F`. Port of
    /// `trippy.raster.ref_torch.fragments_per_layer`.
    #[must_use]
    pub fn fragments_per_layer(&self, segment_offsets: &[u32]) -> Vec<u32> {
        (0..self.num_layers())
            .map(|l| {
                let lo = self.offsets[l];
                let hi = self.offsets.get(l + 1).copied().unwrap_or(self.total);
                segment_offsets[hi] - segment_offsets[lo]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_halving_matches_the_python_layer_grid() {
        // 48x64 with 3 layers is the fixture geometry (tools/dump_raster_fixture.py).
        let g = LayerGrid::new(48, 64, 3, PyramidHalving::Ceil).expect("grid");
        assert_eq!(g.shapes(), &[(48, 64), (24, 32), (12, 16)]);
        assert_eq!(g.offsets(), &[0, 3072, 3840]);
        assert_eq!(g.total(), 3072 + 768 + 192);
    }

    #[test]
    fn ceil_halving_rounds_odd_sizes_up() {
        // 1080 -> 540 -> 270 -> 135 -> 68, the worked example in docs/GEOMETRY.md.
        let g = LayerGrid::new(1080, 1920, 5, PyramidHalving::Ceil).expect("grid");
        let heights: Vec<usize> = g.shapes().iter().map(|s| s.0).collect();
        assert_eq!(heights, vec![1080, 540, 270, 135, 68]);
    }

    #[test]
    fn floor_halving_drops_the_last_row() {
        let g = LayerGrid::new(1080, 1920, 5, PyramidHalving::Floor).expect("grid");
        let heights: Vec<usize> = g.shapes().iter().map(|s| s.0).collect();
        assert_eq!(heights, vec![1080, 540, 270, 135, 67]);
    }

    #[test]
    fn an_empty_layer_is_an_error_not_a_silent_drop() {
        assert!(LayerGrid::new(4, 4, 6, PyramidHalving::Floor).is_err());
        assert!(LayerGrid::new(0, 4, 1, PyramidHalving::Ceil).is_err());
        assert!(LayerGrid::new(4, 4, 0, PyramidHalving::Ceil).is_err());
    }

    #[test]
    fn flat_index_is_layer_major_and_row_major_within_a_layer() {
        let g = LayerGrid::new(48, 64, 3, PyramidHalving::Ceil).expect("grid");
        assert_eq!(g.flat_index(0, 0, 0), 0);
        assert_eq!(g.flat_index(0, 1, 2), 66);
        assert_eq!(g.flat_index(1, 0, 0), 3072);
        assert_eq!(g.flat_index(2, 11, 15), 3840 + 11 * 16 + 15);
        assert_eq!(g.flat_index(2, 11, 15), g.total() - 1);
    }

    #[test]
    fn fragments_per_layer_reads_off_the_segment_table() {
        let g = LayerGrid::new(2, 2, 2, PyramidHalving::Ceil).expect("grid");
        // Layers are (2,2) then (1,1): P = 5. One fragment in each of the
        // four fine pixels, three in the single coarse pixel.
        let seg = [0u32, 1, 2, 3, 4, 7];
        assert_eq!(g.fragments_per_layer(&seg), vec![4, 3]);
    }
}
