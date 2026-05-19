use crate::positional::{coords_to_token_index, token_index_to_coords};
use crate::{SparseMaskBatch, SparseTokenMask, TokenGridShape, VJepaConfig};
use anyhow::{Result, ensure};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

#[cfg(feature = "sparse-patchify-cuda")]
type FusionCudaBackend = burn::backend::Cuda<f32, i32>;

#[cfg(feature = "sparse-patchify-cuda")]
type RawCudaBackend = burn_flex_gmm::cuda::DefaultCudaBackend;

#[cfg(all(
    feature = "sparse-patchify-wgpu",
    any(not(target_arch = "wasm32"), feature = "wasm-fusion")
))]
type FusionWgpuBackend = burn::backend::Wgpu<f32, i32>;

#[cfg(all(
    feature = "sparse-patchify-wgpu",
    any(not(target_arch = "wasm32"), feature = "wasm-fusion")
))]
type RawWgpuBackend = burn_flex_gmm::wgpu::DefaultWgpuBackend;

#[derive(Clone, Debug)]
pub struct SparsePatchifyPlan<B: Backend> {
    pub mask: SparseTokenMask,
    pub grid: TokenGridShape,
    pub batch: usize,
    pub coords: Tensor<B, 2, Int>,
    pub coords_host: Vec<[u32; 4]>,
}

#[derive(Clone, Debug)]
pub struct SparsePatchifyBatchPlan<B: Backend> {
    pub mask: SparseMaskBatch<B>,
    pub grid: TokenGridShape,
    pub batch: usize,
    pub coords: Tensor<B, 2, Int>,
    pub coords_host: Vec<[u32; 4]>,
}

#[cfg(all(
    feature = "sparse-patchify-wgpu",
    any(not(target_arch = "wasm32"), feature = "wasm-fusion")
))]
pub fn sparse_patchify3d_forward_wgpu_fusion(
    config: &burn_flex_gmm::SparsePatchify3dConfig,
    input: Tensor<FusionWgpuBackend, 5>,
    coords: Tensor<FusionWgpuBackend, 2, Int>,
    weight: Tensor<FusionWgpuBackend, 5>,
    bias: Tensor<FusionWgpuBackend, 1>,
) -> Tensor<FusionWgpuBackend, 2> {
    use burn::tensor::Tensor as BurnTensor;
    use burn_backend::{DType, Shape, TensorPrimitive};
    use burn_fusion::stream::{Operation, OperationStreams};
    use burn_ir::{CustomOpIr, OperationIr, OperationOutput, TensorIr, TensorStatus};

    #[derive(Debug)]
    struct SparsePatchifyWgpuFusionOp {
        config: burn_flex_gmm::SparsePatchify3dConfig,
        desc: CustomOpIr,
    }

    impl Operation<<RawWgpuBackend as burn_fusion::FusionBackend>::FusionRuntime>
        for SparsePatchifyWgpuFusionOp
    {
        fn execute(
            &self,
            handles: &mut burn_ir::HandleContainer<
                burn_fusion::FusionHandle<
                    <RawWgpuBackend as burn_fusion::FusionBackend>::FusionRuntime,
                >,
            >,
        ) {
            let (inputs, outputs) = self.desc.as_fixed::<4, 1>();
            let input = BurnTensor::<RawWgpuBackend, 5>::from_primitive(TensorPrimitive::Float(
                handles.get_float_tensor::<RawWgpuBackend>(&inputs[0]),
            ));
            let coords = BurnTensor::<RawWgpuBackend, 2, Int>::from_primitive(
                handles.get_int_tensor::<RawWgpuBackend>(&inputs[1]),
            );
            let weight = BurnTensor::<RawWgpuBackend, 5>::from_primitive(TensorPrimitive::Float(
                handles.get_float_tensor::<RawWgpuBackend>(&inputs[2]),
            ));
            let bias = BurnTensor::<RawWgpuBackend, 1>::from_primitive(TensorPrimitive::Float(
                handles.get_float_tensor::<RawWgpuBackend>(&inputs[3]),
            ));
            let output = burn_flex_gmm::wgpu::sparse_patchify3d_forward_wgpu(
                &self.config,
                input,
                coords,
                weight,
                bias,
            )
            .expect("sparse patchify WGPU fusion op failed");
            handles.register_float_tensor::<RawWgpuBackend>(
                &outputs[0].id,
                output.into_primitive().tensor(),
            );
        }
    }

    let rows = coords.shape().dims::<2>()[0];
    let input = input.into_primitive().tensor();
    let coords = coords.into_primitive();
    let weight = weight.into_primitive().tensor();
    let bias = bias.into_primitive().tensor();
    let client = input.client.clone();
    let streams = OperationStreams::with_inputs([&input, &coords, &weight, &bias]);
    let inputs = [
        input.into_ir(),
        coords.into_ir(),
        weight.into_ir(),
        bias.into_ir(),
    ];
    let output = TensorIr {
        status: TensorStatus::NotInit,
        shape: Shape::new([rows, config.out_channels]),
        id: client.create_empty_handle(),
        dtype: DType::F32,
    };
    let desc = CustomOpIr::new(
        "burn_jepa::sparse_patchify3d_forward_wgpu",
        &inputs,
        std::slice::from_ref(&output),
    );
    let output = client
        .register(
            streams,
            OperationIr::Custom(desc.clone()),
            SparsePatchifyWgpuFusionOp {
                config: *config,
                desc,
            },
        )
        .output();
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

#[cfg(feature = "sparse-patchify-cuda")]
pub fn sparse_patchify3d_forward_cuda_fusion(
    config: &burn_flex_gmm::SparsePatchify3dConfig,
    input: Tensor<FusionCudaBackend, 5>,
    coords: Tensor<FusionCudaBackend, 2, Int>,
    weight: Tensor<FusionCudaBackend, 5>,
    bias: Tensor<FusionCudaBackend, 1>,
) -> Tensor<FusionCudaBackend, 2> {
    use burn::tensor::Tensor as BurnTensor;
    use burn_backend::{DType, Shape, TensorPrimitive};
    use burn_fusion::stream::{Operation, OperationStreams};
    use burn_ir::{CustomOpIr, OperationIr, OperationOutput, TensorIr, TensorStatus};

    #[derive(Debug)]
    struct SparsePatchifyCudaFusionOp {
        config: burn_flex_gmm::SparsePatchify3dConfig,
        desc: CustomOpIr,
    }

    impl Operation<<RawCudaBackend as burn_fusion::FusionBackend>::FusionRuntime>
        for SparsePatchifyCudaFusionOp
    {
        fn execute(
            &self,
            handles: &mut burn_ir::HandleContainer<
                burn_fusion::FusionHandle<
                    <RawCudaBackend as burn_fusion::FusionBackend>::FusionRuntime,
                >,
            >,
        ) {
            let (inputs, outputs) = self.desc.as_fixed::<4, 1>();
            let input = BurnTensor::<RawCudaBackend, 5>::from_primitive(TensorPrimitive::Float(
                handles.get_float_tensor::<RawCudaBackend>(&inputs[0]),
            ));
            let coords = BurnTensor::<RawCudaBackend, 2, Int>::from_primitive(
                handles.get_int_tensor::<RawCudaBackend>(&inputs[1]),
            );
            let weight = BurnTensor::<RawCudaBackend, 5>::from_primitive(TensorPrimitive::Float(
                handles.get_float_tensor::<RawCudaBackend>(&inputs[2]),
            ));
            let bias = BurnTensor::<RawCudaBackend, 1>::from_primitive(TensorPrimitive::Float(
                handles.get_float_tensor::<RawCudaBackend>(&inputs[3]),
            ));
            let output = burn_flex_gmm::cuda::sparse_patchify3d_forward_cuda(
                &self.config,
                input,
                coords,
                weight,
                bias,
            )
            .expect("sparse patchify CUDA fusion op failed");
            handles.register_float_tensor::<RawCudaBackend>(
                &outputs[0].id,
                output.into_primitive().tensor(),
            );
        }
    }

    let rows = coords.shape().dims::<2>()[0];
    let input = input.into_primitive().tensor();
    let coords = coords.into_primitive();
    let weight = weight.into_primitive().tensor();
    let bias = bias.into_primitive().tensor();
    let client = input.client.clone();
    let streams = OperationStreams::with_inputs([&input, &coords, &weight, &bias]);
    let inputs = [
        input.into_ir(),
        coords.into_ir(),
        weight.into_ir(),
        bias.into_ir(),
    ];
    let output = TensorIr {
        status: TensorStatus::NotInit,
        shape: Shape::new([rows, config.out_channels]),
        id: client.create_empty_handle(),
        dtype: DType::F32,
    };
    let desc = CustomOpIr::new(
        "burn_jepa::sparse_patchify3d_forward_cuda",
        &inputs,
        std::slice::from_ref(&output),
    );
    let output = client
        .register(
            streams,
            OperationIr::Custom(desc.clone()),
            SparsePatchifyCudaFusionOp {
                config: *config,
                desc,
            },
        )
        .output();
    Tensor::from_primitive(TensorPrimitive::Float(output))
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

impl<B: Backend> SparsePatchifyBatchPlan<B> {
    pub fn new(mask: SparseMaskBatch<B>, grid: TokenGridShape, device: &B::Device) -> Result<Self> {
        ensure!(mask.batch() > 0, "sparse patchify batch must be nonzero");
        ensure!(
            !mask.is_ragged(),
            "sparse patchify batch plans require uniform or fixed-width masks; use ragged rollout grouping"
        );
        ensure!(
            mask.dense_len() == grid.len(),
            "sparse patchify batch mask dense token count must match grid"
        );
        let batch = mask.batch();
        let token_count = mask.len();
        let rows = mask.rows();
        let mut coords_host = Vec::with_capacity(batch * token_count);
        let mut coords_flat = Vec::with_capacity(batch * token_count * 4);
        for (batch_index, row) in rows.iter().enumerate() {
            for &index in row {
                let (tubelet, token_row, col) = token_index_to_coords(index, grid);
                let coord = [
                    batch_index as u32,
                    tubelet as u32,
                    token_row as u32,
                    col as u32,
                ];
                coords_host.push(coord);
                coords_flat.extend(coord.into_iter().map(|value| value as i64));
            }
        }
        let coords = Tensor::<B, 2, Int>::from_data(
            TensorData::new(coords_flat, [batch * token_count, 4]),
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseImageTokenGrid {
    pub height: usize,
    pub width: usize,
}

impl SparseImageTokenGrid {
    pub const fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }

    pub const fn len(&self) -> usize {
        self.height * self.width
    }

    pub const fn is_empty(&self) -> bool {
        self.height == 0 || self.width == 0
    }

    pub fn token_rect(&self, token: usize) -> Option<SparsePatchRect> {
        if self.is_empty() || token >= self.len() {
            return None;
        }
        let row = token / self.width;
        let col = token % self.width;
        Some(SparsePatchRect::new(
            col as f32 / self.width as f32,
            row as f32 / self.height as f32,
            (col + 1) as f32 / self.width as f32,
            (row + 1) as f32 / self.height as f32,
        ))
    }
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

pub fn sparse_mask_from_frame_token_indices(
    grid: TokenGridShape,
    tubelet_size: usize,
    image_grid: SparseImageTokenGrid,
    frame_tokens: &[Vec<usize>],
    dilation: usize,
    keep_tokens: usize,
) -> Result<SparseTokenMask> {
    let pairs = frame_tokens
        .iter()
        .enumerate()
        .flat_map(|(frame, tokens)| tokens.iter().copied().map(move |token| (frame, token)));
    sparse_mask_from_frame_token_pairs(grid, tubelet_size, image_grid, pairs, dilation, keep_tokens)
}

pub fn sparse_mask_from_frame_token_pairs(
    grid: TokenGridShape,
    tubelet_size: usize,
    image_grid: SparseImageTokenGrid,
    frame_tokens: impl IntoIterator<Item = (usize, usize)>,
    dilation: usize,
    keep_tokens: usize,
) -> Result<SparseTokenMask> {
    ensure!(!grid.is_empty(), "sparse patchify grid must be non-empty");
    ensure!(tubelet_size > 0, "tubelet size must be nonzero");
    ensure!(
        !image_grid.is_empty(),
        "sparse image token grid must be non-empty"
    );
    let target = keep_tokens.max(1).min(grid.len());
    let mut builder = SparseMaskSelectionBuilder::new(grid, target);

    for (frame, token) in frame_tokens {
        let tubelet = frame / tubelet_size;
        if tubelet >= grid.depth {
            continue;
        }
        builder.push_image_token(token, image_grid, tubelet, dilation);
        if builder.is_full() {
            return builder.finish();
        }
    }

    for index in SparseTokenMask::evenly_spaced(grid.len(), target)
        .indices()
        .iter()
        .copied()
    {
        builder.push_sparse_index(index);
        if builder.is_full() {
            return builder.finish();
        }
    }
    for index in 0..grid.len() {
        builder.push_sparse_index(index);
        if builder.is_full() {
            break;
        }
    }
    builder.finish()
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

fn image_token_patch_bounds(
    token: usize,
    image_grid: SparseImageTokenGrid,
    grid: TokenGridShape,
) -> Option<(usize, usize, usize, usize)> {
    if image_grid.is_empty() || token >= image_grid.len() || grid.height == 0 || grid.width == 0 {
        return None;
    }
    let image_row = token / image_grid.width;
    let image_col = token % image_grid.width;
    let row_start = image_row * grid.height / image_grid.height;
    let row_end = ((image_row + 1) * grid.height)
        .div_ceil(image_grid.height)
        .saturating_sub(1)
        .min(grid.height - 1);
    let col_start = image_col * grid.width / image_grid.width;
    let col_end = ((image_col + 1) * grid.width)
        .div_ceil(image_grid.width)
        .saturating_sub(1)
        .min(grid.width - 1);
    Some((row_start, row_end, col_start, col_end))
}

#[derive(Debug)]
struct SparseMaskSelectionBuilder {
    grid: TokenGridShape,
    target: usize,
    keep: Vec<bool>,
    selected: Vec<usize>,
}

impl SparseMaskSelectionBuilder {
    fn new(grid: TokenGridShape, target: usize) -> Self {
        Self {
            grid,
            target,
            keep: vec![false; grid.len()],
            selected: Vec::with_capacity(target),
        }
    }

    fn push_image_token(
        &mut self,
        token: usize,
        image_grid: SparseImageTokenGrid,
        tubelet: usize,
        dilation: usize,
    ) {
        let Some((row_start, row_end, col_start, col_end)) =
            image_token_patch_bounds(token, image_grid, self.grid)
        else {
            return;
        };
        self.push_patch_bounds(row_start, row_end, col_start, col_end, tubelet, dilation);
    }

    fn push_patch_bounds(
        &mut self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
        tubelet: usize,
        dilation: usize,
    ) {
        let row_start = row_start.saturating_sub(dilation);
        let row_end = (row_end + dilation).min(self.grid.height.saturating_sub(1));
        let col_start = col_start.saturating_sub(dilation);
        let col_end = (col_end + dilation).min(self.grid.width.saturating_sub(1));
        for row in row_start..=row_end {
            for col in col_start..=col_end {
                let index = coords_to_token_index(tubelet, row, col, self.grid);
                self.push_sparse_index(index);
                if self.is_full() {
                    return;
                }
            }
        }
    }

    fn push_sparse_index(&mut self, index: usize) {
        if self.is_full() || index >= self.keep.len() || self.keep[index] {
            return;
        }
        self.keep[index] = true;
        self.selected.push(index);
    }

    fn is_full(&self) -> bool {
        self.selected.len() >= self.target
    }

    fn finish(self) -> Result<SparseTokenMask> {
        SparseTokenMask::new(self.selected, self.grid.len())
    }
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

#[cfg(all(test, feature = "ndarray"))]
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
    fn sparse_mask_from_frame_token_indices_maps_image_tokens_to_tubelets() {
        let grid = TokenGridShape::new(2, 4, 4);
        let image_grid = SparseImageTokenGrid::new(2, 2);
        let frame_tokens = vec![vec![], vec![1], vec![2], vec![]];
        let mask = sparse_mask_from_frame_token_indices(grid, 2, image_grid, &frame_tokens, 0, 4)
            .expect("mask");

        assert_eq!(mask.indices(), &[2, 3, 6, 7]);
    }

    #[test]
    fn sparse_mask_from_frame_token_pairs_matches_grouped_tokens() {
        let grid = TokenGridShape::new(2, 4, 4);
        let image_grid = SparseImageTokenGrid::new(2, 2);
        let frame_tokens = vec![vec![], vec![1], vec![2], vec![]];
        let grouped =
            sparse_mask_from_frame_token_indices(grid, 2, image_grid, &frame_tokens, 0, 4)
                .expect("grouped mask");
        let pairs = sparse_mask_from_frame_token_pairs(grid, 2, image_grid, [(1, 1), (2, 2)], 0, 4)
            .expect("pair mask");

        assert_eq!(pairs, grouped);
    }

    #[test]
    fn sparse_mask_from_frame_token_indices_fills_to_keep_count() {
        let grid = TokenGridShape::new(1, 4, 4);
        let image_grid = SparseImageTokenGrid::new(2, 2);
        let frame_tokens = vec![vec![0]];
        let mask = sparse_mask_from_frame_token_indices(grid, 1, image_grid, &frame_tokens, 0, 6)
            .expect("mask");

        assert_eq!(mask.len(), 6);
        for index in [0, 1, 4, 5] {
            assert!(mask.indices().contains(&index));
        }
    }

    #[test]
    fn sparse_mask_from_frame_token_indices_allows_partial_stream_window() {
        let grid = TokenGridShape::new(3, 2, 2);
        let image_grid = SparseImageTokenGrid::new(1, 1);
        let frame_tokens = vec![vec![0]];
        let mask = sparse_mask_from_frame_token_indices(grid, 2, image_grid, &frame_tokens, 0, 5)
            .expect("mask");

        assert_eq!(mask.len(), 5);
        for index in [0, 1, 2, 3] {
            assert!(mask.indices().contains(&index));
        }
    }

    #[test]
    fn sparse_mask_from_frame_rects_falls_back_to_center_token() {
        let grid = TokenGridShape::new(3, 3, 3);
        let frame_rects = vec![Vec::new(); 6];
        let mask = sparse_mask_from_frame_rects(grid, 2, &frame_rects, 0, 1).expect("mask");
        assert_eq!(mask.indices(), &[coords_to_token_index(1, 1, 1, grid)]);
    }
}
