use crate::{
    DensePredictionOutput, SparseImageTokenGrid, SparseTokenMask, TokenGridShape, VJepaConfig,
    VJepaPipeline, VJepaRgbaVideoShape, VJepaVideoShape, coords_to_token_index,
    rgba_video_to_tensor, sparse_mask_from_frame_token_indices, target_mask_from_context,
    token_index_to_coords,
};
use anyhow::{Result, ensure};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct SparseJepaTensorPipelineConfig {
    pub sparsity_driver: SparseJepaSparsityDriverConfig,
}

impl Default for SparseJepaTensorPipelineConfig {
    fn default() -> Self {
        Self {
            sparsity_driver: SparseJepaSparsityDriverConfig::KeepRatio {
                context_keep_ratio: 0.5,
            },
        }
    }
}

impl SparseJepaTensorPipelineConfig {
    pub fn with_sparsity_driver(mut self, driver: SparseJepaSparsityDriverConfig) -> Self {
        self.sparsity_driver = driver;
        self
    }

    pub fn keep_ratio(context_keep_ratio: f32) -> Self {
        Self {
            sparsity_driver: SparseJepaSparsityDriverConfig::KeepRatio { context_keep_ratio },
        }
    }
}

#[derive(Clone, Debug)]
pub enum SparseJepaSparsityDriverConfig {
    FullFrame {
        target_tokens: usize,
    },
    KeepRatio {
        context_keep_ratio: f32,
    },
    AutogazeSparse(SparseJepaAutogazeSparsityConfig),
    PatchDiff(SparseJepaPatchDiffSparsityConfig),
    PrecomputedMasks {
        context_mask: SparseTokenMask,
        target_mask: SparseTokenMask,
    },
}

impl SparseJepaSparsityDriverConfig {
    pub fn full_frame(target_tokens: usize) -> Self {
        Self::FullFrame { target_tokens }
    }

    pub fn keep_ratio(context_keep_ratio: f32) -> Self {
        Self::KeepRatio { context_keep_ratio }
    }
}

#[derive(Clone, Debug)]
pub struct SparseJepaAutogazeSparsityConfig {
    pub image_grid: SparseImageTokenGrid,
    pub frame_tokens: Vec<Vec<usize>>,
    pub context_tokens: usize,
    pub target_tokens: usize,
    pub dilation: usize,
}

impl SparseJepaAutogazeSparsityConfig {
    pub fn new(
        image_grid: SparseImageTokenGrid,
        frame_tokens: Vec<Vec<usize>>,
        context_tokens: usize,
        target_tokens: usize,
    ) -> Self {
        Self {
            image_grid,
            frame_tokens,
            context_tokens,
            target_tokens,
            dilation: 0,
        }
    }

    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }
}

#[derive(Clone, Debug)]
pub struct SparseJepaPatchDiffSparsityConfig {
    pub threshold: f32,
    pub context_tokens: usize,
    pub target_tokens: usize,
    pub dilation: usize,
}

impl SparseJepaPatchDiffSparsityConfig {
    pub fn new(threshold: f32, context_tokens: usize, target_tokens: usize) -> Self {
        Self {
            threshold,
            context_tokens,
            target_tokens,
            dilation: 0,
        }
    }

    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }
}

#[derive(Debug)]
pub struct SparseJepaPacket<B: Backend> {
    pub video: Tensor<B, 5>,
    pub context_mask: SparseTokenMask,
    pub target_mask: SparseTokenMask,
    pub output: DensePredictionOutput<B>,
}

pub trait SparseJepaInputNode<B: Backend> {
    fn next_video(&mut self, device: &B::Device) -> Result<Option<Tensor<B, 5>>>;
}

pub trait SparseJepaOutputNode<B: Backend> {
    fn push(&mut self, packet: SparseJepaPacket<B>) -> Result<()>;
}

#[derive(Clone, Debug)]
pub struct TensorVideoInput<B: Backend> {
    video: Option<Tensor<B, 5>>,
}

impl<B: Backend> TensorVideoInput<B> {
    pub fn new(video: Tensor<B, 5>) -> Self {
        Self { video: Some(video) }
    }
}

impl<B: Backend> SparseJepaInputNode<B> for TensorVideoInput<B> {
    fn next_video(&mut self, _device: &B::Device) -> Result<Option<Tensor<B, 5>>> {
        Ok(self.video.take())
    }
}

#[derive(Clone, Debug)]
pub struct RgbaVideoInput {
    rgba: Option<Vec<u8>>,
    shape: VJepaRgbaVideoShape,
}

impl RgbaVideoInput {
    pub fn new(rgba: Vec<u8>, shape: VJepaRgbaVideoShape) -> Self {
        Self {
            rgba: Some(rgba),
            shape,
        }
    }
}

impl<B: Backend> SparseJepaInputNode<B> for RgbaVideoInput {
    fn next_video(&mut self, device: &B::Device) -> Result<Option<Tensor<B, 5>>> {
        self.rgba
            .take()
            .map(|rgba| rgba_video_to_tensor::<B>(&rgba, self.shape, device))
            .transpose()
    }
}

#[derive(Debug, Default)]
pub struct VecOutputNode<B: Backend> {
    pub packets: Vec<SparseJepaPacket<B>>,
}

impl<B: Backend> VecOutputNode<B> {
    pub fn new() -> Self {
        Self {
            packets: Vec::new(),
        }
    }
}

impl<B: Backend> SparseJepaOutputNode<B> for VecOutputNode<B> {
    fn push(&mut self, packet: SparseJepaPacket<B>) -> Result<()> {
        self.packets.push(packet);
        Ok(())
    }
}

pub struct FnOutputNode<F> {
    f: F,
}

impl<F> FnOutputNode<F> {
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<B, F> SparseJepaOutputNode<B> for FnOutputNode<F>
where
    B: Backend,
    F: FnMut(SparseJepaPacket<B>) -> Result<()>,
{
    fn push(&mut self, packet: SparseJepaPacket<B>) -> Result<()> {
        (self.f)(packet)
    }
}

pub struct SparseJepaTensorPipeline<B: Backend, I, O> {
    pipeline: VJepaPipeline<B>,
    input: I,
    output: O,
    config: SparseJepaTensorPipelineConfig,
}

impl<B, I, O> SparseJepaTensorPipeline<B, I, O>
where
    B: Backend,
    I: SparseJepaInputNode<B>,
    O: SparseJepaOutputNode<B>,
{
    pub fn new(pipeline: VJepaPipeline<B>, input: I, output: O) -> Self {
        Self {
            pipeline,
            input,
            output,
            config: SparseJepaTensorPipelineConfig::default(),
        }
    }

    pub fn with_config(mut self, config: SparseJepaTensorPipelineConfig) -> Self {
        self.config = config;
        self
    }

    pub fn run_next(&mut self, device: &B::Device) -> Result<bool> {
        let Some(video) = self.input.next_video(device)? else {
            return Ok(false);
        };
        let grid = {
            let [_batch, _channels, frames, height, width] = video.shape().dims::<5>();
            crate::TokenGridShape::new(
                frames / self.pipeline.config().tubelet_size.max(1),
                height / self.pipeline.config().patch_size.max(1),
                width / self.pipeline.config().patch_size.max(1),
            )
        };
        let (context_mask, target_mask) = resolve_sparsity_driver_masks(
            &self.config.sparsity_driver,
            &video,
            self.pipeline.config(),
            grid,
        )?;
        let output = self.pipeline.model().predict_dense_targets(
            video.clone(),
            &context_mask,
            &target_mask,
        )?;
        self.output.push(SparseJepaPacket {
            video,
            context_mask,
            target_mask,
            output,
        })?;
        Ok(true)
    }

    pub fn into_output(self) -> O {
        self.output
    }
}

pub fn resolve_sparsity_driver_masks<B: Backend>(
    driver: &SparseJepaSparsityDriverConfig,
    video: &Tensor<B, 5>,
    model_config: &VJepaConfig,
    grid: TokenGridShape,
) -> Result<(SparseTokenMask, SparseTokenMask)> {
    let masks = match driver {
        SparseJepaSparsityDriverConfig::FullFrame { target_tokens } => {
            full_frame_masks(grid, *target_tokens)?
        }
        SparseJepaSparsityDriverConfig::KeepRatio { context_keep_ratio } => {
            let (mut context, mut target) =
                crate::make_context_target_masks(grid, *context_keep_ratio);
            if target.is_empty() && grid.len() > 1 {
                target = SparseTokenMask::evenly_spaced(grid.len(), 1);
                context = target.complement();
            }
            (context, target)
        }
        SparseJepaSparsityDriverConfig::AutogazeSparse(config) => {
            let context = sparse_mask_from_frame_token_indices(
                grid,
                model_config.tubelet_size.max(1),
                config.image_grid,
                &config.frame_tokens,
                config.dilation,
                context_budget(grid, config.context_tokens)?,
            )?;
            let target = target_mask_from_context(&context, config.target_tokens)?;
            (context, target)
        }
        SparseJepaSparsityDriverConfig::PatchDiff(config) => {
            let context = patch_diff_context_mask(video, model_config, grid, config)?;
            let target = target_mask_from_context(&context, config.target_tokens)?;
            (context, target)
        }
        SparseJepaSparsityDriverConfig::PrecomputedMasks {
            context_mask,
            target_mask,
        } => (context_mask.clone(), target_mask.clone()),
    };
    validate_sparsity_masks(&masks.0, &masks.1, grid)?;
    Ok(masks)
}

fn full_frame_masks(
    grid: TokenGridShape,
    target_tokens: usize,
) -> Result<(SparseTokenMask, SparseTokenMask)> {
    ensure!(
        grid.len() > 1,
        "full-frame JEPA mode requires at least two dense tokens"
    );
    let target =
        SparseTokenMask::evenly_spaced(grid.len(), target_tokens.max(1).min(grid.len() - 1));
    let context = target.complement();
    Ok((context, target))
}

fn context_budget(grid: TokenGridShape, context_tokens: usize) -> Result<usize> {
    ensure!(
        grid.len() > 1,
        "sparse JEPA mode requires at least two dense tokens"
    );
    Ok(context_tokens.max(1).min(grid.len() - 1))
}

fn validate_sparsity_masks(
    context: &SparseTokenMask,
    target: &SparseTokenMask,
    grid: TokenGridShape,
) -> Result<()> {
    ensure!(
        context.dense_len() == grid.len() && target.dense_len() == grid.len(),
        "sparsity driver masks must match the video token grid"
    );
    ensure!(
        !context.is_empty() && !target.is_empty(),
        "sparsity driver must produce non-empty context and target masks"
    );
    let mut keep = vec![false; grid.len()];
    for &index in context.indices() {
        keep[index] = true;
    }
    ensure!(
        target.indices().iter().all(|&index| !keep[index]),
        "sparsity driver context and target masks must be disjoint"
    );
    Ok(())
}

fn patch_diff_context_mask<B: Backend>(
    video: &Tensor<B, 5>,
    model_config: &VJepaConfig,
    grid: TokenGridShape,
    config: &SparseJepaPatchDiffSparsityConfig,
) -> Result<SparseTokenMask> {
    ensure!(
        config.threshold.is_finite() && config.threshold >= 0.0,
        "patch-diff sparsity threshold must be finite and non-negative"
    );
    let [_batch, _channels, frames, height, width] = video.shape().dims::<5>();
    let expected_grid = TokenGridShape::new(
        frames / model_config.tubelet_size.max(1),
        height / model_config.patch_size.max(1),
        width / model_config.patch_size.max(1),
    );
    ensure!(
        frames.is_multiple_of(model_config.tubelet_size.max(1))
            && height.is_multiple_of(model_config.patch_size.max(1))
            && width.is_multiple_of(model_config.patch_size.max(1)),
        "patch-diff sparsity requires frames/height/width divisible by tubelet and patch sizes"
    );
    ensure!(
        expected_grid == grid,
        "patch-diff sparsity grid must match the video shape"
    );
    let context_tokens = context_budget(grid, config.context_tokens)?;
    let scores = patch_diff_scores(video, model_config, grid)?;
    if config.threshold <= 0.0 && config.dilation == 0 {
        return patch_diff_topk_context_mask(scores, grid, context_tokens);
    }

    let score_values = scores
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("failed to read patch-diff score tensor: {err}"))?;
    ensure!(
        score_values.len() == grid.len(),
        "patch-diff sparsity received unexpected score tensor length"
    );
    let mut scores = score_values
        .into_iter()
        .enumerate()
        .collect::<Vec<(usize, f32)>>();
    scores.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });

    let mut keep = vec![false; grid.len()];
    let mut selected = Vec::with_capacity(context_tokens);
    for &(index, score) in &scores {
        if score < config.threshold {
            break;
        }
        push_dilated_sparse_index(
            index,
            grid,
            config.dilation,
            &mut keep,
            &mut selected,
            context_tokens,
        );
        if selected.len() >= context_tokens {
            break;
        }
    }
    if selected.len() < context_tokens {
        for &(index, _) in &scores {
            push_dilated_sparse_index(
                index,
                grid,
                config.dilation,
                &mut keep,
                &mut selected,
                context_tokens,
            );
            if selected.len() >= context_tokens {
                break;
            }
        }
    }
    SparseTokenMask::new(selected, grid.len())
}

fn patch_diff_topk_context_mask<B: Backend>(
    scores: Tensor<B, 1>,
    grid: TokenGridShape,
    context_tokens: usize,
) -> Result<SparseTokenMask> {
    let (_, indices) = scores.topk_with_indices(context_tokens, 0);
    let indices = indices
        .into_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .map_err(|err| anyhow::anyhow!("failed to read patch-diff top-k indices: {err}"))?
        .into_iter()
        .map(|index| index as usize)
        .collect::<Vec<_>>();
    SparseTokenMask::new(indices, grid.len())
}

fn patch_diff_scores<B: Backend>(
    video: &Tensor<B, 5>,
    model_config: &VJepaConfig,
    grid: TokenGridShape,
) -> Result<Tensor<B, 1>> {
    let [batch, channels, frames, height, width] = video.shape().dims::<5>();
    let patch_size = model_config.patch_size.max(1);
    let tubelet_size = model_config.tubelet_size.max(1);
    let device = video.device();
    let frame_tokens = grid.tokens_per_frame();
    let mut tubelet_scores = Vec::with_capacity(grid.depth);
    for tubelet in 0..grid.depth {
        let start_frame = tubelet * tubelet_size;
        let end_frame = ((tubelet + 1) * tubelet_size).min(frames);
        let frame_pairs = patch_diff_frame_pairs(start_frame, end_frame);
        if frame_pairs.is_empty() {
            tubelet_scores.push(Tensor::<B, 1>::zeros([frame_tokens], &device));
            continue;
        }

        let mut diff_sum = None;
        for (prev_frame, next_frame) in frame_pairs.iter().copied() {
            let prev = video
                .clone()
                .slice_dim(2, prev_frame..prev_frame + 1)
                .reshape([batch, channels, height, width]);
            let next = video
                .clone()
                .slice_dim(2, next_frame..next_frame + 1)
                .reshape([batch, channels, height, width]);
            let diff = (next - prev).abs();
            diff_sum = Some(match diff_sum {
                Some(acc) => acc + diff,
                None => diff,
            });
        }

        let diff_sum: Tensor<B, 4> = diff_sum.expect("non-empty frame pair list");
        let denom = (frame_pairs.len() * batch * channels * patch_size * patch_size) as f64;
        let scores = diff_sum
            .reshape([
                batch,
                channels,
                grid.height,
                patch_size,
                grid.width,
                patch_size,
            ])
            .sum_dims_squeeze::<2, _>(&[0, 1, 3, 5])
            .mul_scalar(1.0 / denom)
            .reshape([frame_tokens]);
        tubelet_scores.push(scores);
    }
    Ok(Tensor::cat(tubelet_scores, 0))
}

fn patch_diff_frame_pairs(start_frame: usize, end_frame: usize) -> Vec<(usize, usize)> {
    if end_frame > start_frame + 1 {
        (start_frame + 1..end_frame)
            .map(|frame| (frame - 1, frame))
            .collect()
    } else if start_frame > 0 {
        vec![(start_frame - 1, start_frame)]
    } else {
        Vec::new()
    }
}

fn push_dilated_sparse_index(
    index: usize,
    grid: TokenGridShape,
    dilation: usize,
    keep: &mut [bool],
    selected: &mut Vec<usize>,
    target: usize,
) {
    if selected.len() >= target || index >= grid.len() {
        return;
    }
    let (tubelet, row, col) = token_index_to_coords(index, grid);
    let row_start = row.saturating_sub(dilation);
    let row_end = (row + dilation).min(grid.height.saturating_sub(1));
    let col_start = col.saturating_sub(dilation);
    let col_end = (col + dilation).min(grid.width.saturating_sub(1));
    for row in row_start..=row_end {
        for col in col_start..=col_end {
            let index = coords_to_token_index(tubelet, row, col, grid);
            if keep[index] {
                continue;
            }
            keep[index] = true;
            selected.push(index);
            if selected.len() >= target {
                return;
            }
        }
    }
}

pub fn empty_rgb_video_shape(frames: usize, height: usize, width: usize) -> VJepaVideoShape {
    VJepaVideoShape::new(1, 3, frames, height, width)
}
