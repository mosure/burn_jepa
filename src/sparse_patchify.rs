use crate::positional::{coords_to_token_index, token_index_to_coords};
use crate::{SparseTokenMask, TokenGridShape, VJepaConfig};
use anyhow::{Result, ensure};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

#[derive(Clone, Debug)]
pub struct SparsePatchifyPlan<B: Backend> {
    pub mask: SparseTokenMask,
    pub grid: TokenGridShape,
    pub batch: usize,
    pub coords: Tensor<B, 2, Int>,
    pub coords_host: Vec<[u32; 4]>,
}

impl<B: Backend> SparsePatchifyPlan<B> {
    pub fn new(
        mask: SparseTokenMask,
        grid: TokenGridShape,
        batch: usize,
        device: &B::Device,
    ) -> Result<Self> {
        ensure!(batch > 0, "sparse patchify batch must be nonzero");
        ensure!(
            mask.dense_len() == grid.len(),
            "sparse patchify mask dense token count must match grid"
        );
        let mut coords_host = Vec::with_capacity(batch * mask.len());
        let mut coords_flat = Vec::with_capacity(batch * mask.len() * 4);
        for batch_index in 0..batch {
            for &index in mask.indices() {
                let (tubelet, row, col) = token_index_to_coords(index, grid);
                let coord = [batch_index as u32, tubelet as u32, row as u32, col as u32];
                coords_host.push(coord);
                coords_flat.extend(coord.into_iter().map(|value| value as i64));
            }
        }
        let coords = Tensor::<B, 2, Int>::from_data(
            TensorData::new(coords_flat, [batch * mask.len(), 4]),
            device,
        );
        Ok(Self {
            mask,
            grid,
            batch,
            coords,
            coords_host,
        })
    }

    pub fn from_indices(
        indices: Vec<usize>,
        grid: TokenGridShape,
        batch: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let mask = SparseTokenMask::new(indices, grid.len())?;
        Self::new(mask, grid, batch, device)
    }

    pub fn token_count(&self) -> usize {
        self.mask.len()
    }

    pub fn output_rows(&self) -> usize {
        self.batch * self.mask.len()
    }
}

pub fn video_token_grid(
    config: &VJepaConfig,
    frames: usize,
    height: usize,
    width: usize,
) -> Result<TokenGridShape> {
    ensure!(
        frames.is_multiple_of(config.tubelet_size.max(1)),
        "video frames must be divisible by V-JEPA tubelet size"
    );
    ensure!(
        height.is_multiple_of(config.patch_size.max(1)),
        "video height must be divisible by V-JEPA patch size"
    );
    ensure!(
        width.is_multiple_of(config.patch_size.max(1)),
        "video width must be divisible by V-JEPA patch size"
    );
    Ok(TokenGridShape::new(
        frames / config.tubelet_size.max(1),
        height / config.patch_size.max(1),
        width / config.patch_size.max(1),
    ))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparsePatchRect {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl SparsePatchRect {
    pub fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self { x0, y0, x1, y1 }
    }

    fn normalized(self) -> Self {
        Self {
            x0: self.x0.min(self.x1).clamp(0.0, 1.0),
            y0: self.y0.min(self.y1).clamp(0.0, 1.0),
            x1: self.x0.max(self.x1).clamp(0.0, 1.0),
            y1: self.y0.max(self.y1).clamp(0.0, 1.0),
        }
    }

    fn intersects_patch(self, row: usize, col: usize, grid: TokenGridShape) -> bool {
        let rect = self.normalized();
        if rect.x1 <= rect.x0 || rect.y1 <= rect.y0 {
            return false;
        }
        let px0 = col as f32 / grid.width.max(1) as f32;
        let py0 = row as f32 / grid.height.max(1) as f32;
        let px1 = (col + 1) as f32 / grid.width.max(1) as f32;
        let py1 = (row + 1) as f32 / grid.height.max(1) as f32;
        rect.x0 < px1 && rect.x1 > px0 && rect.y0 < py1 && rect.y1 > py0
    }
}

pub fn sparse_mask_from_frame_rects(
    grid: TokenGridShape,
    tubelet_size: usize,
    frame_rects: &[Vec<SparsePatchRect>],
    dilation: usize,
    min_keep_tokens: usize,
) -> Result<SparseTokenMask> {
    ensure!(!grid.is_empty(), "sparse patchify grid must be non-empty");
    ensure!(tubelet_size > 0, "tubelet size must be nonzero");
    let mut keep = vec![false; grid.len()];
    for tubelet in 0..grid.depth {
        let start = tubelet * tubelet_size;
        let end = ((tubelet + 1) * tubelet_size).min(frame_rects.len());
        for frame_rects in &frame_rects[start..end] {
            for row in 0..grid.height {
                for col in 0..grid.width {
                    if frame_rects
                        .iter()
                        .any(|rect| rect.intersects_patch(row, col, grid))
                    {
                        mark_dilated(&mut keep, grid, tubelet, row, col, dilation);
                    }
                }
            }
        }
    }
    ensure_min_keep(&mut keep, grid, min_keep_tokens.max(1));
    let indices = keep
        .into_iter()
        .enumerate()
        .filter_map(|(index, value)| value.then_some(index))
        .collect();
    SparseTokenMask::new(indices, grid.len())
}

fn mark_dilated(
    keep: &mut [bool],
    grid: TokenGridShape,
    frame: usize,
    row: usize,
    col: usize,
    dilation: usize,
) {
    let row_min = row.saturating_sub(dilation);
    let row_max = (row + dilation).min(grid.height.saturating_sub(1));
    let col_min = col.saturating_sub(dilation);
    let col_max = (col + dilation).min(grid.width.saturating_sub(1));
    for r in row_min..=row_max {
        for c in col_min..=col_max {
            keep[coords_to_token_index(frame, r, c, grid)] = true;
        }
    }
}

fn ensure_min_keep(keep: &mut [bool], grid: TokenGridShape, min_keep_tokens: usize) {
    let mut kept = keep.iter().filter(|&&value| value).count();
    if kept >= min_keep_tokens {
        return;
    }
    let center = [grid.depth / 2, grid.height / 2, grid.width / 2];
    let mut candidates = (0..grid.len()).collect::<Vec<_>>();
    candidates.sort_by_key(|&index| {
        let (frame, row, col) = token_index_to_coords(index, grid);
        frame.abs_diff(center[0]) + row.abs_diff(center[1]) + col.abs_diff(center[2])
    });
    for index in candidates {
        if !keep[index] {
            keep[index] = true;
            kept += 1;
            if kept >= min_keep_tokens {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn::backend::NdArray<f32>;

    #[test]
    fn sparse_patchify_plan_maps_mask_indices_to_batched_coords() {
        let device = Default::default();
        let grid = TokenGridShape::new(2, 3, 4);
        let mask = SparseTokenMask::new(vec![0, 7, 23], grid.len()).expect("mask");
        let plan = SparsePatchifyPlan::<B>::new(mask, grid, 2, &device).expect("plan");
        assert_eq!(
            plan.coords_host,
            vec![
                [0, 0, 0, 0],
                [0, 0, 1, 3],
                [0, 1, 2, 3],
                [1, 0, 0, 0],
                [1, 0, 1, 3],
                [1, 1, 2, 3],
            ]
        );
    }

    #[test]
    fn sparse_mask_from_frame_rects_keeps_touched_tubelet_patches() {
        let grid = TokenGridShape::new(2, 4, 4);
        let mut frame_rects = vec![Vec::new(); 4];
        frame_rects[1].push(SparsePatchRect::new(0.5, 0.0, 0.75, 0.25));
        let mask = sparse_mask_from_frame_rects(grid, 2, &frame_rects, 0, 1).expect("mask");
        assert_eq!(mask.indices(), &[2]);
    }

    #[test]
    fn sparse_mask_from_frame_rects_falls_back_to_center_token() {
        let grid = TokenGridShape::new(3, 3, 3);
        let frame_rects = vec![Vec::new(); 6];
        let mask = sparse_mask_from_frame_rects(grid, 2, &frame_rects, 0, 1).expect("mask");
        assert_eq!(mask.indices(), &[coords_to_token_index(1, 1, 1, grid)]);
    }
}
