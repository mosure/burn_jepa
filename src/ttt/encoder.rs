use super::config::{TttBackpropMode, TttEncoderConfig, TttTargetMode};
use super::layer::{VJepaTttLayer, VJepaTttLayerProbe};
use super::state::TttState;
use crate::{
    SparseEncoderBatchPlan, SparseEncoderPlan, SparseMaskBatch, SparseTokenMask, TokenGridShape,
    VJepaConfig, VJepaEncoder, VJepaEncoderOutput, apply_mask_batch, apply_token_mask,
};
#[cfg(any(feature = "sparse-patchify-wgpu", feature = "sparse-patchify-cuda"))]
use crate::{SparsePatchifyBatchPlan, SparsePatchifyPlan};
use anyhow::{Result, ensure};
use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Clone, Debug)]
pub struct VJepaTttLayerProbeRecord<B: Backend> {
    pub encoder_layer: usize,
    pub ttt_layer: usize,
    pub probe: VJepaTttLayerProbe<B>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TttStateResetMode {
    #[default]
    Persistent,
    EachFrame,
    EachTubelet,
}

#[derive(Module, Debug)]
pub struct VJepaTttEncoder<B: Backend> {
    pub base: VJepaEncoder<B>,
    pub ttt_layers: Vec<VJepaTttLayer<B>>,
    #[module(skip)]
    config: VJepaConfig,
    #[module(skip)]
    layer_indices: Vec<usize>,
    #[module(skip)]
    hierarchical_layers: Vec<usize>,
    #[module(skip)]
    rollout_blocks: usize,
    #[module(skip)]
    backprop_mode: TttBackpropMode,
    #[module(skip)]
    backprop_truncate_blocks: usize,
    #[module(skip)]
    target_mode: TttTargetMode,
}

impl<B: Backend> VJepaTttEncoder<B> {
    pub fn new(
        base: VJepaEncoder<B>,
        model_config: &VJepaConfig,
        ttt_config: TttEncoderConfig,
        device: &B::Device,
    ) -> Result<Self> {
        ttt_config.validate(model_config)?;
        let layer_indices = ttt_config.resolved_layers(model_config);
        let ttt_layers = layer_indices
            .iter()
            .map(|_| VJepaTttLayer::new(model_config.encoder.embed_dim, &ttt_config, device))
            .collect();
        Ok(Self {
            base,
            ttt_layers,
            config: model_config.clone(),
            layer_indices,
            hierarchical_layers: model_config.encoder.hierarchical_layers(),
            rollout_blocks: ttt_config.rollout_blocks,
            backprop_mode: ttt_config.backprop_mode,
            backprop_truncate_blocks: ttt_config.backprop_truncate_blocks,
            target_mode: ttt_config.target,
        })
    }

    pub fn fresh_state(&self) -> TttState<B> {
        TttState::new(self.ttt_layers.len())
    }

    fn should_detach_after_tubelet(&self, tubelet_index: usize, grid_depth: usize) -> bool {
        let blocks = match self.backprop_mode {
            TttBackpropMode::TruncatedFinal => self.backprop_truncate_blocks,
            TttBackpropMode::FinalFeature | TttBackpropMode::LayerLocal => self.rollout_blocks,
        };
        blocks > 0 && (tubelet_index + 1) % blocks == 0 && tubelet_index + 1 < grid_depth
    }

    fn should_early_exit_after_layer(&self, layer_index: usize) -> bool {
        self.backprop_mode == TttBackpropMode::LayerLocal
            && self.layer_indices.last().copied() == Some(layer_index)
    }

    pub fn target_mode(&self) -> TttTargetMode {
        self.target_mode
    }

    pub fn ttt_layer_indices(&self) -> &[usize] {
        &self.layer_indices
    }

    pub fn forward_video(
        &self,
        video: Tensor<B, 5>,
        mask: Option<&SparseTokenMask>,
    ) -> Result<VJepaEncoderOutput<B>> {
        let mut state = self.fresh_state();
        self.forward_video_with_state(video, mask, None, &mut state)
    }

    pub fn forward_video_with_state(
        &self,
        video: Tensor<B, 5>,
        mask: Option<&SparseTokenMask>,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        let [batch, _channels, frames, height, width] = video.shape().dims::<5>();
        let grid = TokenGridShape::new(
            frames / self.config.tubelet_size.max(1),
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        let tokens = self.base.patch_embed.forward(video);
        self.forward_tokens(tokens, batch, grid, mask, true, target_tokens, state)
    }

    pub fn forward_image(
        &self,
        image: Tensor<B, 4>,
        mask: Option<&SparseTokenMask>,
    ) -> Result<VJepaEncoderOutput<B>> {
        let mut state = self.fresh_state();
        self.forward_image_with_state(image, mask, None, &mut state)
    }

    pub fn forward_image_with_state(
        &self,
        image: Tensor<B, 4>,
        mask: Option<&SparseTokenMask>,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        self.forward_image_with_state_impl(image, mask, target_tokens, state, true, None)
    }

    fn forward_image_with_state_impl(
        &self,
        image: Tensor<B, 4>,
        mask: Option<&SparseTokenMask>,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
        update_fast_weight: bool,
        probes: Option<&mut Vec<VJepaTttLayerProbeRecord<B>>>,
    ) -> Result<VJepaEncoderOutput<B>> {
        let [batch, channels, height, width] = image.shape().dims::<4>();
        let grid = TokenGridShape::new(
            1,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        let tokens = self
            .base
            .image_patch_embed
            .forward(image.reshape([batch, channels, 1, height, width]));
        self.forward_tokens_with_options(
            tokens,
            batch,
            grid,
            mask,
            false,
            target_tokens,
            state,
            update_fast_weight,
            probes,
        )
    }

    pub fn forward_single_frame_rollout(
        &self,
        video: Tensor<B, 5>,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        self.forward_single_frame_rollout_impl(
            video,
            None,
            target_tokens,
            state,
            true,
            TttStateResetMode::Persistent,
            None,
        )
    }

    pub fn forward_single_frame_rollout_sparse(
        &self,
        video: Tensor<B, 5>,
        mask: &SparseTokenMask,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        self.forward_single_frame_rollout_impl(
            video,
            Some(mask),
            target_tokens,
            state,
            true,
            TttStateResetMode::Persistent,
            None,
        )
    }

    pub fn forward_single_frame_rollout_sparse_batch(
        &self,
        video: Tensor<B, 5>,
        mask: &SparseMaskBatch<B>,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        if let Some(mask) = mask.uniform_mask() {
            return self.forward_single_frame_rollout_sparse(video, mask, target_tokens, state);
        }
        self.forward_single_frame_rollout_batch_impl(
            video,
            mask,
            target_tokens,
            state,
            true,
            TttStateResetMode::Persistent,
            None,
        )
    }

    pub fn forward_single_frame_rollout_with_diagnostics(
        &self,
        video: Tensor<B, 5>,
        mask: Option<&SparseMaskBatch<B>>,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
        update_fast_weight: bool,
        reset_mode: TttStateResetMode,
        probes: Option<&mut Vec<VJepaTttLayerProbeRecord<B>>>,
    ) -> Result<VJepaEncoderOutput<B>> {
        match mask {
            Some(mask) => {
                if let Some(mask) = mask.uniform_mask() {
                    self.forward_single_frame_rollout_impl(
                        video,
                        Some(mask),
                        target_tokens,
                        state,
                        update_fast_weight,
                        reset_mode,
                        probes,
                    )
                } else {
                    self.forward_single_frame_rollout_batch_impl(
                        video,
                        mask,
                        target_tokens,
                        state,
                        update_fast_weight,
                        reset_mode,
                        probes,
                    )
                }
            }
            None => self.forward_single_frame_rollout_impl(
                video,
                None,
                target_tokens,
                state,
                update_fast_weight,
                reset_mode,
                probes,
            ),
        }
    }

    fn forward_single_frame_rollout_impl(
        &self,
        video: Tensor<B, 5>,
        mask: Option<&SparseTokenMask>,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
        update_fast_weight: bool,
        reset_mode: TttStateResetMode,
        mut probes: Option<&mut Vec<VJepaTttLayerProbeRecord<B>>>,
    ) -> Result<VJepaEncoderOutput<B>> {
        let [batch, channels, frames, height, width] = video.shape().dims::<5>();
        let tubelet = self.config.tubelet_size.max(1);
        ensure!(
            frames % tubelet == 0,
            "single-frame rollout requires frames divisible by tubelet_size"
        );
        let grid = TokenGridShape::new(
            frames / tubelet,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        if let Some(mask) = mask {
            ensure!(
                mask.dense_len() == grid.len(),
                "single-frame sparse rollout mask must match video token grid"
            );
            ensure!(
                !mask.is_empty(),
                "single-frame sparse rollout mask must not be empty"
            );
        }
        let frame_grid = TokenGridShape::new(1, grid.height, grid.width);
        let frame_tokens = frame_grid.len();
        let mut outputs = Vec::with_capacity(grid.depth);
        for frame in 0..frames {
            let tubelet_index = frame / tubelet;
            if reset_mode == TttStateResetMode::EachFrame
                || (reset_mode == TttStateResetMode::EachTubelet && frame % tubelet == 0)
            {
                *state = self.fresh_state();
            }
            let frame_mask = mask
                .map(|mask| sparse_rollout_frame_mask(mask, grid, tubelet_index))
                .transpose()?
                .flatten();
            if mask.is_some() && frame_mask.is_none() {
                if frame % tubelet == tubelet - 1
                    && self.should_detach_after_tubelet(tubelet_index, grid.depth)
                {
                    state.detach();
                }
                continue;
            }
            let image = video
                .clone()
                .slice_dim(2, frame..frame + 1)
                .reshape([batch, channels, height, width]);
            let target_frame = rollout_target_frame(
                target_tokens.as_ref(),
                tubelet_index,
                frame_tokens,
                frame_mask.as_ref(),
                batch,
                &video.device(),
            );
            let encoded = self.forward_image_with_state_impl(
                image,
                frame_mask.as_ref(),
                target_frame,
                state,
                update_fast_weight,
                probes.as_mut().map(|records| &mut **records),
            )?;
            if frame % tubelet == tubelet - 1 {
                outputs.push(encoded.tokens);
                if self.should_detach_after_tubelet(tubelet_index, grid.depth) {
                    state.detach();
                }
            }
        }
        ensure!(
            !outputs.is_empty(),
            "single-frame rollout produced no output tokens"
        );
        let tokens = Tensor::cat(outputs, 1);
        let device = tokens.device();
        let mask = mask
            .cloned()
            .unwrap_or_else(|| SparseTokenMask::all(grid.len()));
        let plan = SparseEncoderPlan::new(&self.config, mask, grid, batch, true, &device)?;
        Ok(VJepaEncoderOutput {
            tokens,
            hierarchical: Vec::new(),
            token_indices: plan.positions.indices,
            grid,
        })
    }

    fn forward_single_frame_rollout_batch_impl(
        &self,
        video: Tensor<B, 5>,
        mask: &SparseMaskBatch<B>,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
        update_fast_weight: bool,
        reset_mode: TttStateResetMode,
        mut probes: Option<&mut Vec<VJepaTttLayerProbeRecord<B>>>,
    ) -> Result<VJepaEncoderOutput<B>> {
        let [batch, channels, frames, height, width] = video.shape().dims::<5>();
        let tubelet = self.config.tubelet_size.max(1);
        ensure!(
            frames % tubelet == 0,
            "single-frame rollout requires frames divisible by tubelet_size"
        );
        ensure!(
            batch == mask.batch(),
            "single-frame sparse rollout batch mask must match video batch"
        );
        let grid = TokenGridShape::new(
            frames / tubelet,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            mask.dense_len() == grid.len() && !mask.is_empty(),
            "single-frame sparse rollout batch mask must match a non-empty video token grid"
        );
        let frame_grid = TokenGridShape::new(1, grid.height, grid.width);
        let frame_tokens = frame_grid.len();
        let device = video.device();
        let mut outputs = Vec::with_capacity(grid.depth);
        for frame in 0..frames {
            let tubelet_index = frame / tubelet;
            if reset_mode == TttStateResetMode::EachFrame
                || (reset_mode == TttStateResetMode::EachTubelet && frame % tubelet == 0)
            {
                *state = self.fresh_state();
            }
            let Some(frame_mask) =
                sparse_rollout_frame_mask_batch(mask, grid, tubelet_index, &device)?
            else {
                if frame % tubelet == tubelet - 1
                    && self.should_detach_after_tubelet(tubelet_index, grid.depth)
                {
                    state.detach();
                }
                continue;
            };
            let image = video
                .clone()
                .slice_dim(2, frame..frame + 1)
                .reshape([batch, channels, height, width]);
            let target_frame = rollout_target_frame_batch(
                target_tokens.as_ref(),
                tubelet_index,
                frame_tokens,
                &frame_mask,
            );
            let tokens = self
                .base
                .image_patch_embed
                .forward(image.reshape([batch, channels, 1, height, width]));
            let tokens = apply_mask_batch(tokens, &frame_mask);
            let encoder_plan =
                SparseEncoderBatchPlan::new(&self.config, frame_mask, frame_grid, false, &device)?;
            let encoded = self.forward_sparse_tokens_with_batch_plan_options(
                tokens,
                &encoder_plan,
                target_frame,
                state,
                update_fast_weight,
                probes.as_mut().map(|records| &mut **records),
            )?;
            if frame % tubelet == tubelet - 1 {
                outputs.push(encoded.tokens);
                if self.should_detach_after_tubelet(tubelet_index, grid.depth) {
                    state.detach();
                }
            }
        }
        ensure!(
            !outputs.is_empty(),
            "single-frame sparse rollout produced no output tokens"
        );
        let tokens = Tensor::cat(outputs, 1);
        let plan = SparseEncoderBatchPlan::new(&self.config, mask.clone(), grid, true, &device)?;
        Ok(VJepaEncoderOutput {
            tokens,
            hierarchical: Vec::new(),
            token_indices: plan.positions.indices,
            grid,
        })
    }

    pub fn forward_sparse_tokens_with_plan(
        &self,
        tokens: Tensor<B, 3>,
        plan: &SparseEncoderPlan<B>,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        self.forward_sparse_tokens_impl(tokens, plan, target_tokens, state)
    }

    fn forward_tokens(
        &self,
        tokens: Tensor<B, 3>,
        batch: usize,
        grid: TokenGridShape,
        mask: Option<&SparseTokenMask>,
        video: bool,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        self.forward_tokens_with_options(
            tokens,
            batch,
            grid,
            mask,
            video,
            target_tokens,
            state,
            true,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_tokens_with_options(
        &self,
        tokens: Tensor<B, 3>,
        batch: usize,
        grid: TokenGridShape,
        mask: Option<&SparseTokenMask>,
        video: bool,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
        update_fast_weight: bool,
        probes: Option<&mut Vec<VJepaTttLayerProbeRecord<B>>>,
    ) -> Result<VJepaEncoderOutput<B>> {
        let device = tokens.device();
        let mask = mask
            .cloned()
            .unwrap_or_else(|| SparseTokenMask::all(grid.len()));
        let plan = SparseEncoderPlan::new(&self.config, mask, grid, batch, video, &device)?;
        let tokens = if plan.mask.len() < grid.len() {
            apply_token_mask(tokens, plan.positions.indices.clone())
        } else {
            tokens
        };
        self.forward_sparse_tokens_impl_options(
            tokens,
            &plan,
            target_tokens,
            state,
            update_fast_weight,
            probes,
        )
    }

    fn forward_sparse_tokens_impl(
        &self,
        tokens: Tensor<B, 3>,
        plan: &SparseEncoderPlan<B>,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        self.forward_sparse_tokens_impl_options(tokens, plan, target_tokens, state, true, None)
    }

    fn forward_sparse_tokens_impl_options(
        &self,
        mut tokens: Tensor<B, 3>,
        plan: &SparseEncoderPlan<B>,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
        update_fast_weight: bool,
        mut probes: Option<&mut Vec<VJepaTttLayerProbeRecord<B>>>,
    ) -> Result<VJepaEncoderOutput<B>> {
        ensure!(
            state.layers.len() == self.ttt_layers.len(),
            "TTT state layer count does not match encoder"
        );
        let [batch, token_count, dim] = tokens.shape().dims::<3>();
        ensure!(
            batch == plan.batch,
            "encoder token batch does not match plan"
        );
        ensure!(
            token_count == plan.mask.len(),
            "encoder token count does not match plan"
        );
        ensure!(
            dim == self.config.encoder.embed_dim,
            "encoder token dimension does not match config"
        );
        if let Some(target) = &target_tokens {
            let target_dims = target.shape().dims::<3>();
            ensure!(
                target_dims == [batch, token_count, dim],
                "TTT target token shape must match encoder tokens"
            );
        }
        if let Some(position_embed) = &plan.position_embed {
            tokens = tokens + position_embed.clone();
        }
        if self.config.encoder.modality_embedding {
            let embed = if plan.video {
                self.base.video_mod_embed.val()
            } else {
                self.base.image_mod_embed.val()
            }
            .reshape([1, 1, dim])
            .repeat_dim(0, batch)
            .repeat_dim(1, token_count);
            tokens = tokens + embed;
        }

        let mut hierarchical = Vec::with_capacity(self.base.norms_block.len());
        let mut x = tokens;
        for (layer_index, block) in self.base.blocks.iter().enumerate() {
            x = block.forward(x, Some(&plan.positions));
            if let Ok(ttt_index) = self.layer_indices.binary_search(&layer_index) {
                let layer_target = match self.target_mode {
                    TttTargetMode::TeacherFinal => target_tokens.clone(),
                    TttTargetMode::SelfHidden => None,
                };
                x = if let Some(records) = probes.as_mut() {
                    let (next, probe) = self.ttt_layers[ttt_index].forward_with_probe(
                        x,
                        layer_target,
                        &mut state.layers[ttt_index],
                        update_fast_weight,
                    );
                    records.push(VJepaTttLayerProbeRecord {
                        encoder_layer: layer_index,
                        ttt_layer: ttt_index,
                        probe,
                    });
                    next
                } else {
                    self.ttt_layers[ttt_index].forward_with_options(
                        x,
                        layer_target,
                        &mut state.layers[ttt_index],
                        update_fast_weight,
                    )
                };
                if self.should_early_exit_after_layer(layer_index) {
                    break;
                }
            }
            if let Some(norm_index) = self
                .hierarchical_layers
                .iter()
                .position(|&index| index == layer_index)
            {
                hierarchical.push(self.base.norms_block[norm_index].forward(x.clone()));
            }
        }
        let tokens = if let Some(norm) = self.base.norms_block.last() {
            norm.forward(x)
        } else {
            x
        };
        Ok(VJepaEncoderOutput {
            tokens,
            hierarchical,
            token_indices: plan.positions.indices.clone(),
            grid: plan.grid,
        })
    }

    pub fn forward_sparse_tokens_with_batch_plan(
        &self,
        tokens: Tensor<B, 3>,
        plan: &SparseEncoderBatchPlan<B>,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        self.forward_sparse_tokens_with_batch_plan_options(
            tokens,
            plan,
            target_tokens,
            state,
            true,
            None,
        )
    }

    fn forward_sparse_tokens_with_batch_plan_options(
        &self,
        mut tokens: Tensor<B, 3>,
        plan: &SparseEncoderBatchPlan<B>,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
        update_fast_weight: bool,
        mut probes: Option<&mut Vec<VJepaTttLayerProbeRecord<B>>>,
    ) -> Result<VJepaEncoderOutput<B>> {
        ensure!(
            state.layers.len() == self.ttt_layers.len(),
            "TTT state layer count does not match encoder"
        );
        let [batch, token_count, dim] = tokens.shape().dims::<3>();
        ensure!(
            batch == plan.batch,
            "encoder token batch does not match batch plan"
        );
        ensure!(
            token_count == plan.mask.len(),
            "encoder token count does not match batch plan"
        );
        ensure!(
            dim == self.config.encoder.embed_dim,
            "encoder token dimension does not match config"
        );
        if let Some(target) = &target_tokens {
            let target_dims = target.shape().dims::<3>();
            ensure!(
                target_dims == [batch, token_count, dim],
                "TTT target token shape must match encoder tokens"
            );
        }
        if let Some(position_embed) = &plan.position_embed {
            tokens = tokens + position_embed.clone();
        }
        if self.config.encoder.modality_embedding {
            let embed = if plan.video {
                self.base.video_mod_embed.val()
            } else {
                self.base.image_mod_embed.val()
            }
            .reshape([1, 1, dim])
            .repeat_dim(0, batch)
            .repeat_dim(1, token_count);
            tokens = tokens + embed;
        }

        let mut hierarchical = Vec::with_capacity(self.base.norms_block.len());
        let mut x = tokens;
        for (layer_index, block) in self.base.blocks.iter().enumerate() {
            x = block.forward(x, Some(&plan.positions));
            if let Ok(ttt_index) = self.layer_indices.binary_search(&layer_index) {
                let layer_target = match self.target_mode {
                    TttTargetMode::TeacherFinal => target_tokens.clone(),
                    TttTargetMode::SelfHidden => None,
                };
                x = if let Some(records) = probes.as_mut() {
                    let (next, probe) = self.ttt_layers[ttt_index].forward_with_probe(
                        x,
                        layer_target,
                        &mut state.layers[ttt_index],
                        update_fast_weight,
                    );
                    records.push(VJepaTttLayerProbeRecord {
                        encoder_layer: layer_index,
                        ttt_layer: ttt_index,
                        probe,
                    });
                    next
                } else {
                    self.ttt_layers[ttt_index].forward_with_options(
                        x,
                        layer_target,
                        &mut state.layers[ttt_index],
                        update_fast_weight,
                    )
                };
                if self.should_early_exit_after_layer(layer_index) {
                    break;
                }
            }
            if let Some(norm_index) = self
                .hierarchical_layers
                .iter()
                .position(|&index| index == layer_index)
            {
                hierarchical.push(self.base.norms_block[norm_index].forward(x.clone()));
            }
        }
        let tokens = if let Some(norm) = self.base.norms_block.last() {
            norm.forward(x)
        } else {
            x
        };
        Ok(VJepaEncoderOutput {
            tokens,
            hierarchical,
            token_indices: plan.positions.indices.clone(),
            grid: plan.grid,
        })
    }
}

fn sparse_rollout_frame_mask(
    mask: &SparseTokenMask,
    grid: TokenGridShape,
    tubelet: usize,
) -> Result<Option<SparseTokenMask>> {
    let frame_tokens = grid.tokens_per_frame();
    let start = tubelet * frame_tokens;
    let end = start + frame_tokens;
    let indices = mask
        .indices()
        .iter()
        .copied()
        .filter_map(|index| (index >= start && index < end).then_some(index - start))
        .collect::<Vec<_>>();
    if indices.is_empty() {
        Ok(None)
    } else {
        SparseTokenMask::new(indices, frame_tokens).map(Some)
    }
}

fn sparse_rollout_frame_mask_batch<B: Backend>(
    mask: &SparseMaskBatch<B>,
    grid: TokenGridShape,
    tubelet: usize,
    device: &B::Device,
) -> Result<Option<SparseMaskBatch<B>>> {
    let frame_tokens = grid.tokens_per_frame();
    let start = tubelet * frame_tokens;
    let end = start + frame_tokens;
    let rows = mask
        .rows()
        .into_iter()
        .map(|row| {
            row.into_iter()
                .filter_map(|index| (index >= start && index < end).then_some(index - start))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if rows.iter().all(Vec::is_empty) {
        return Ok(None);
    }
    ensure!(
        rows.iter().all(|row| !row.is_empty()),
        "fixed-width sparse rollout does not yet support batches where only some samples have frame tokens"
    );
    Ok(Some(SparseMaskBatch::from_rows(
        rows,
        frame_tokens,
        device,
    )?))
}

fn rollout_target_frame<B: Backend>(
    target_tokens: Option<&Tensor<B, 3>>,
    tubelet: usize,
    frame_tokens: usize,
    frame_mask: Option<&SparseTokenMask>,
    batch: usize,
    device: &B::Device,
) -> Option<Tensor<B, 3>> {
    target_tokens.map(|target| {
        let start = tubelet * frame_tokens;
        let target = target.clone().slice_dim(1, start..start + frame_tokens);
        if let Some(mask) = frame_mask {
            apply_token_mask(target, mask.to_tensor::<B>(batch, device))
        } else {
            target
        }
    })
}

fn rollout_target_frame_batch<B: Backend>(
    target_tokens: Option<&Tensor<B, 3>>,
    tubelet: usize,
    frame_tokens: usize,
    frame_mask: &SparseMaskBatch<B>,
) -> Option<Tensor<B, 3>> {
    target_tokens.map(|target| {
        let start = tubelet * frame_tokens;
        let target = target.clone().slice_dim(1, start..start + frame_tokens);
        apply_mask_batch(target, frame_mask)
    })
}

#[cfg(feature = "sparse-patchify-wgpu")]
impl VJepaTttEncoder<burn_flex_gmm::wgpu::DefaultWgpuBackend> {
    pub fn sparse_patchify_image_wgpu(
        &self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        plan: &SparsePatchifyPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 3>> {
        let [batch, channels, height, width] = image.shape().dims::<4>();
        ensure!(
            batch == plan.batch,
            "image batch does not match sparse patchify plan"
        );
        ensure!(
            channels == self.config.in_channels,
            "image channel count does not match V-JEPA config"
        );
        let grid = TokenGridShape::new(
            1,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            grid == plan.grid,
            "image token grid does not match sparse patchify plan"
        );
        let device = image.device();
        let patchify_config = burn_flex_gmm::SparsePatchify3dConfig {
            in_channels: channels,
            out_channels: self.config.encoder.embed_dim,
            frames: 1,
            height,
            width,
            tubelet_size: 1,
            patch_h: self.config.patch_size,
            patch_w: self.config.patch_size,
        };
        let bias = self
            .base
            .image_patch_embed
            .proj
            .bias
            .as_ref()
            .map(|bias| bias.val())
            .unwrap_or_else(|| {
                Tensor::<burn_flex_gmm::wgpu::DefaultWgpuBackend, 1>::zeros(
                    [self.config.encoder.embed_dim],
                    &device,
                )
            });
        burn_flex_gmm::wgpu::sparse_patchify3d_forward_wgpu(
            &patchify_config,
            image.reshape([batch, channels, 1, height, width]),
            plan.coords.clone(),
            self.base.image_patch_embed.proj.weight.val(),
            bias,
        )
        .map_err(anyhow::Error::msg)
        .map(|tokens| tokens.reshape([batch, plan.token_count(), self.config.encoder.embed_dim]))
    }

    pub fn forward_single_frame_rollout_sparse_patchify_wgpu(
        &self,
        video: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 5>,
        mask: &SparseTokenMask,
        target_tokens: Option<Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 3>>,
        state: &mut TttState<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        self.forward_single_frame_rollout_sparse_patchify_wgpu_impl(
            video,
            mask,
            target_tokens,
            state,
        )
    }

    fn forward_single_frame_rollout_sparse_patchify_wgpu_impl(
        &self,
        video: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 5>,
        mask: &SparseTokenMask,
        target_tokens: Option<Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 3>>,
        state: &mut TttState<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        let [batch, channels, frames, height, width] = video.shape().dims::<5>();
        let tubelet = self.config.tubelet_size.max(1);
        ensure!(
            frames % tubelet == 0,
            "single-frame sparse rollout requires frames divisible by tubelet_size"
        );
        let grid = TokenGridShape::new(
            frames / tubelet,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            mask.dense_len() == grid.len() && !mask.is_empty(),
            "single-frame sparse rollout mask must match a non-empty video token grid"
        );
        let frame_grid = TokenGridShape::new(1, grid.height, grid.width);
        let frame_tokens = frame_grid.len();
        let device = video.device();
        let mut outputs = Vec::with_capacity(grid.depth);
        for frame in 0..frames {
            let tubelet_index = frame / tubelet;
            let Some(frame_mask) = sparse_rollout_frame_mask(mask, grid, tubelet_index)? else {
                if frame % tubelet == tubelet - 1
                    && self.should_detach_after_tubelet(tubelet_index, grid.depth)
                {
                    state.detach();
                }
                continue;
            };
            let image = video
                .clone()
                .slice_dim(2, frame..frame + 1)
                .reshape([batch, channels, height, width]);
            let target_frame = rollout_target_frame(
                target_tokens.as_ref(),
                tubelet_index,
                frame_tokens,
                Some(&frame_mask),
                batch,
                &device,
            );
            let patchify_plan =
                SparsePatchifyPlan::new(frame_mask.clone(), frame_grid, batch, &device)?;
            let encoder_plan = SparseEncoderPlan::new(
                &self.config,
                frame_mask,
                frame_grid,
                batch,
                false,
                &device,
            )?;
            let tokens = self.sparse_patchify_image_wgpu(image, &patchify_plan)?;
            let encoded =
                self.forward_sparse_tokens_with_plan(tokens, &encoder_plan, target_frame, state)?;
            if frame % tubelet == tubelet - 1 {
                outputs.push(encoded.tokens);
                if self.should_detach_after_tubelet(tubelet_index, grid.depth) {
                    state.detach();
                }
            }
        }
        ensure!(
            !outputs.is_empty(),
            "single-frame sparse patchify rollout produced no output tokens"
        );
        let tokens = Tensor::cat(outputs, 1);
        let plan = SparseEncoderPlan::new(&self.config, mask.clone(), grid, batch, true, &device)?;
        Ok(VJepaEncoderOutput {
            tokens,
            hierarchical: Vec::new(),
            token_indices: plan.positions.indices,
            grid,
        })
    }
}

#[cfg(feature = "sparse-patchify-cuda")]
impl VJepaTttEncoder<burn_flex_gmm::cuda::DefaultCudaBackend> {
    pub fn sparse_patchify_image_cuda(
        &self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        plan: &SparsePatchifyPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 3>> {
        let [batch, channels, height, width] = image.shape().dims::<4>();
        ensure!(
            batch == plan.batch,
            "image batch does not match sparse patchify plan"
        );
        ensure!(
            channels == self.config.in_channels,
            "image channel count does not match V-JEPA config"
        );
        let grid = TokenGridShape::new(
            1,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            grid == plan.grid,
            "image token grid does not match sparse patchify plan"
        );
        let device = image.device();
        let patchify_config = burn_flex_gmm::SparsePatchify3dConfig {
            in_channels: channels,
            out_channels: self.config.encoder.embed_dim,
            frames: 1,
            height,
            width,
            tubelet_size: 1,
            patch_h: self.config.patch_size,
            patch_w: self.config.patch_size,
        };
        let bias = self
            .base
            .image_patch_embed
            .proj
            .bias
            .as_ref()
            .map(|bias| bias.val())
            .unwrap_or_else(|| {
                Tensor::<burn_flex_gmm::cuda::DefaultCudaBackend, 1>::zeros(
                    [self.config.encoder.embed_dim],
                    &device,
                )
            });
        burn_flex_gmm::cuda::sparse_patchify3d_forward_cuda(
            &patchify_config,
            image.reshape([batch, channels, 1, height, width]),
            plan.coords.clone(),
            self.base.image_patch_embed.proj.weight.val(),
            bias,
        )
        .map_err(anyhow::Error::msg)
        .map(|tokens| tokens.reshape([batch, plan.token_count(), self.config.encoder.embed_dim]))
    }

    pub fn forward_single_frame_rollout_sparse_patchify_cuda(
        &self,
        video: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 5>,
        mask: &SparseTokenMask,
        target_tokens: Option<Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 3>>,
        state: &mut TttState<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        let [batch, channels, frames, height, width] = video.shape().dims::<5>();
        let tubelet = self.config.tubelet_size.max(1);
        ensure!(
            frames % tubelet == 0,
            "single-frame sparse rollout requires frames divisible by tubelet_size"
        );
        let grid = TokenGridShape::new(
            frames / tubelet,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            mask.dense_len() == grid.len() && !mask.is_empty(),
            "single-frame sparse rollout mask must match a non-empty video token grid"
        );
        let frame_grid = TokenGridShape::new(1, grid.height, grid.width);
        let frame_tokens = frame_grid.len();
        let device = video.device();
        let mut outputs = Vec::with_capacity(grid.depth);
        for frame in 0..frames {
            let tubelet_index = frame / tubelet;
            let Some(frame_mask) = sparse_rollout_frame_mask(mask, grid, tubelet_index)? else {
                if frame % tubelet == tubelet - 1
                    && self.should_detach_after_tubelet(tubelet_index, grid.depth)
                {
                    state.detach();
                }
                continue;
            };
            let image = video
                .clone()
                .slice_dim(2, frame..frame + 1)
                .reshape([batch, channels, height, width]);
            let target_frame = rollout_target_frame(
                target_tokens.as_ref(),
                tubelet_index,
                frame_tokens,
                Some(&frame_mask),
                batch,
                &device,
            );
            let patchify_plan =
                SparsePatchifyPlan::new(frame_mask.clone(), frame_grid, batch, &device)?;
            let encoder_plan = SparseEncoderPlan::new(
                &self.config,
                frame_mask,
                frame_grid,
                batch,
                false,
                &device,
            )?;
            let tokens = self.sparse_patchify_image_cuda(image, &patchify_plan)?;
            let encoded =
                self.forward_sparse_tokens_with_plan(tokens, &encoder_plan, target_frame, state)?;
            if frame % tubelet == tubelet - 1 {
                outputs.push(encoded.tokens);
                if self.should_detach_after_tubelet(tubelet_index, grid.depth) {
                    state.detach();
                }
            }
        }
        ensure!(
            !outputs.is_empty(),
            "single-frame sparse patchify rollout produced no output tokens"
        );
        let tokens = Tensor::cat(outputs, 1);
        let plan = SparseEncoderPlan::new(&self.config, mask.clone(), grid, batch, true, &device)?;
        Ok(VJepaEncoderOutput {
            tokens,
            hierarchical: Vec::new(),
            token_indices: plan.positions.indices,
            grid,
        })
    }
}

#[cfg(feature = "sparse-patchify-wgpu")]
impl VJepaTttEncoder<burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
    pub fn forward_single_frame_rollout_sparse_patchify_wgpu_frozen(
        &self,
        video: Tensor<burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>, 5>,
        mask: &SparseTokenMask,
        target_tokens: Option<
            Tensor<burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>, 3>,
        >,
        state: &mut TttState<burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>>,
    ) -> Result<VJepaEncoderOutput<burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>>>
    {
        let [batch, channels, frames, height, width] = video.shape().dims::<5>();
        let tubelet = self.config.tubelet_size.max(1);
        ensure!(
            frames % tubelet == 0,
            "single-frame sparse rollout requires frames divisible by tubelet_size"
        );
        let grid = TokenGridShape::new(
            frames / tubelet,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            mask.dense_len() == grid.len() && !mask.is_empty(),
            "single-frame sparse rollout mask must match a non-empty video token grid"
        );
        let frame_grid = TokenGridShape::new(1, grid.height, grid.width);
        let frame_tokens = frame_grid.len();
        let device = video.device();
        let mut outputs = Vec::with_capacity(grid.depth);
        for frame in 0..frames {
            let tubelet_index = frame / tubelet;
            let Some(frame_mask) = sparse_rollout_frame_mask(mask, grid, tubelet_index)? else {
                if frame % tubelet == tubelet - 1
                    && self.should_detach_after_tubelet(tubelet_index, grid.depth)
                {
                    state.detach();
                }
                continue;
            };
            let image = video
                .clone()
                .slice_dim(2, frame..frame + 1)
                .reshape([batch, channels, height, width]);
            let target_frame = rollout_target_frame(
                target_tokens.as_ref(),
                tubelet_index,
                frame_tokens,
                Some(&frame_mask),
                batch,
                &device,
            );
            let patchify_plan =
                SparsePatchifyPlan::new(frame_mask.clone(), frame_grid, batch, &device)?;
            let encoder_plan = SparseEncoderPlan::new(
                &self.config,
                frame_mask,
                frame_grid,
                batch,
                false,
                &device,
            )?;
            let tokens = self.sparse_patchify_image_wgpu_frozen(image, &patchify_plan)?;
            let encoded =
                self.forward_sparse_tokens_with_plan(tokens, &encoder_plan, target_frame, state)?;
            if frame % tubelet == tubelet - 1 {
                outputs.push(encoded.tokens);
                if self.should_detach_after_tubelet(tubelet_index, grid.depth) {
                    state.detach();
                }
            }
        }
        ensure!(
            !outputs.is_empty(),
            "single-frame sparse patchify rollout produced no output tokens"
        );
        let tokens = Tensor::cat(outputs, 1);
        let plan = SparseEncoderPlan::new(&self.config, mask.clone(), grid, batch, true, &device)?;
        Ok(VJepaEncoderOutput {
            tokens,
            hierarchical: Vec::new(),
            token_indices: plan.positions.indices,
            grid,
        })
    }

    fn sparse_patchify_image_wgpu_frozen(
        &self,
        image: Tensor<burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>, 4>,
        plan: &SparsePatchifyPlan<burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>>,
    ) -> Result<Tensor<burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>, 3>> {
        let [batch, channels, height, width] = image.shape().dims::<4>();
        ensure!(
            batch == plan.batch,
            "image batch does not match sparse patchify plan"
        );
        ensure!(
            channels == self.config.in_channels,
            "image channel count does not match V-JEPA config"
        );
        let grid = TokenGridShape::new(
            1,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            grid == plan.grid,
            "image token grid does not match sparse patchify plan"
        );
        let device = image.device();
        let patchify_config = burn_flex_gmm::SparsePatchify3dConfig {
            in_channels: channels,
            out_channels: self.config.encoder.embed_dim,
            frames: 1,
            height,
            width,
            tubelet_size: 1,
            patch_h: self.config.patch_size,
            patch_w: self.config.patch_size,
        };
        let bias = self
            .base
            .image_patch_embed
            .proj
            .bias
            .as_ref()
            .map(|bias| bias.val().inner())
            .unwrap_or_else(|| {
                Tensor::<burn_flex_gmm::wgpu::DefaultWgpuBackend, 1>::zeros(
                    [self.config.encoder.embed_dim],
                    &device,
                )
            });
        let tokens = burn_flex_gmm::wgpu::sparse_patchify3d_forward_wgpu(
            &patchify_config,
            image.inner().reshape([batch, channels, 1, height, width]),
            plan.coords.clone().inner(),
            self.base.image_patch_embed.proj.weight.val().inner(),
            bias,
        )
        .map_err(anyhow::Error::msg)?
        .reshape([batch, plan.token_count(), self.config.encoder.embed_dim]);
        Ok(Tensor::from_inner(tokens))
    }

    pub fn forward_single_frame_rollout_sparse_patchify_wgpu_frozen_batch(
        &self,
        video: Tensor<burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>, 5>,
        mask: &SparseMaskBatch<burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>>,
        target_tokens: Option<
            Tensor<burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>, 3>,
        >,
        state: &mut TttState<burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>>,
    ) -> Result<VJepaEncoderOutput<burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>>>
    {
        if let Some(mask) = mask.uniform_mask() {
            return self.forward_single_frame_rollout_sparse_patchify_wgpu_frozen(
                video,
                mask,
                target_tokens,
                state,
            );
        }
        let [batch, channels, frames, height, width] = video.shape().dims::<5>();
        let tubelet = self.config.tubelet_size.max(1);
        ensure!(
            frames % tubelet == 0,
            "single-frame sparse rollout requires frames divisible by tubelet_size"
        );
        ensure!(
            batch == mask.batch(),
            "single-frame sparse rollout batch mask must match video batch"
        );
        let grid = TokenGridShape::new(
            frames / tubelet,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            mask.dense_len() == grid.len() && !mask.is_empty(),
            "single-frame sparse rollout batch mask must match a non-empty video token grid"
        );
        let frame_grid = TokenGridShape::new(1, grid.height, grid.width);
        let frame_tokens = frame_grid.len();
        let device = video.device();
        let mut outputs = Vec::with_capacity(grid.depth);
        for frame in 0..frames {
            let tubelet_index = frame / tubelet;
            let Some(frame_mask) =
                sparse_rollout_frame_mask_batch(mask, grid, tubelet_index, &device)?
            else {
                if frame % tubelet == tubelet - 1
                    && self.should_detach_after_tubelet(tubelet_index, grid.depth)
                {
                    state.detach();
                }
                continue;
            };
            let image = video
                .clone()
                .slice_dim(2, frame..frame + 1)
                .reshape([batch, channels, height, width]);
            let target_frame = rollout_target_frame_batch(
                target_tokens.as_ref(),
                tubelet_index,
                frame_tokens,
                &frame_mask,
            );
            let patchify_plan =
                SparsePatchifyBatchPlan::new(frame_mask.clone(), frame_grid, &device)?;
            let encoder_plan =
                SparseEncoderBatchPlan::new(&self.config, frame_mask, frame_grid, false, &device)?;
            let tokens = self.sparse_patchify_image_wgpu_frozen_batch(image, &patchify_plan)?;
            let encoded = self.forward_sparse_tokens_with_batch_plan(
                tokens,
                &encoder_plan,
                target_frame,
                state,
            )?;
            if frame % tubelet == tubelet - 1 {
                outputs.push(encoded.tokens);
                if self.should_detach_after_tubelet(tubelet_index, grid.depth) {
                    state.detach();
                }
            }
        }
        ensure!(
            !outputs.is_empty(),
            "single-frame sparse patchify rollout produced no output tokens"
        );
        let tokens = Tensor::cat(outputs, 1);
        let plan = SparseEncoderBatchPlan::new(&self.config, mask.clone(), grid, true, &device)?;
        Ok(VJepaEncoderOutput {
            tokens,
            hierarchical: Vec::new(),
            token_indices: plan.positions.indices,
            grid,
        })
    }

    fn sparse_patchify_image_wgpu_frozen_batch(
        &self,
        image: Tensor<burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>, 4>,
        plan: &SparsePatchifyBatchPlan<
            burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        >,
    ) -> Result<Tensor<burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>, 3>> {
        let [batch, channels, height, width] = image.shape().dims::<4>();
        ensure!(
            batch == plan.batch,
            "image batch does not match sparse patchify batch plan"
        );
        ensure!(
            channels == self.config.in_channels,
            "image channel count does not match V-JEPA config"
        );
        let grid = TokenGridShape::new(
            1,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            grid == plan.grid,
            "image token grid does not match sparse patchify batch plan"
        );
        let device = image.device();
        let patchify_config = burn_flex_gmm::SparsePatchify3dConfig {
            in_channels: channels,
            out_channels: self.config.encoder.embed_dim,
            frames: 1,
            height,
            width,
            tubelet_size: 1,
            patch_h: self.config.patch_size,
            patch_w: self.config.patch_size,
        };
        let bias = self
            .base
            .image_patch_embed
            .proj
            .bias
            .as_ref()
            .map(|bias| bias.val().inner())
            .unwrap_or_else(|| {
                Tensor::<burn_flex_gmm::wgpu::DefaultWgpuBackend, 1>::zeros(
                    [self.config.encoder.embed_dim],
                    &device,
                )
            });
        let tokens = burn_flex_gmm::wgpu::sparse_patchify3d_forward_wgpu(
            &patchify_config,
            image.inner().reshape([batch, channels, 1, height, width]),
            plan.coords.clone().inner(),
            self.base.image_patch_embed.proj.weight.val().inner(),
            bias,
        )
        .map_err(anyhow::Error::msg)?
        .reshape([batch, plan.token_count(), self.config.encoder.embed_dim]);
        Ok(Tensor::from_inner(tokens))
    }
}

#[cfg(feature = "sparse-patchify-cuda")]
impl VJepaTttEncoder<burn::backend::Autodiff<burn_flex_gmm::cuda::DefaultCudaBackend>> {
    pub fn forward_single_frame_rollout_sparse_patchify_cuda_frozen(
        &self,
        video: Tensor<burn::backend::Autodiff<burn_flex_gmm::cuda::DefaultCudaBackend>, 5>,
        mask: &SparseTokenMask,
        target_tokens: Option<
            Tensor<burn::backend::Autodiff<burn_flex_gmm::cuda::DefaultCudaBackend>, 3>,
        >,
        state: &mut TttState<burn::backend::Autodiff<burn_flex_gmm::cuda::DefaultCudaBackend>>,
    ) -> Result<VJepaEncoderOutput<burn::backend::Autodiff<burn_flex_gmm::cuda::DefaultCudaBackend>>>
    {
        let [batch, channels, frames, height, width] = video.shape().dims::<5>();
        let tubelet = self.config.tubelet_size.max(1);
        ensure!(
            frames % tubelet == 0,
            "single-frame sparse rollout requires frames divisible by tubelet_size"
        );
        let grid = TokenGridShape::new(
            frames / tubelet,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            mask.dense_len() == grid.len() && !mask.is_empty(),
            "single-frame sparse rollout mask must match a non-empty video token grid"
        );
        let frame_grid = TokenGridShape::new(1, grid.height, grid.width);
        let frame_tokens = frame_grid.len();
        let device = video.device();
        let mut outputs = Vec::with_capacity(grid.depth);
        for frame in 0..frames {
            let tubelet_index = frame / tubelet;
            let Some(frame_mask) = sparse_rollout_frame_mask(mask, grid, tubelet_index)? else {
                if frame % tubelet == tubelet - 1
                    && self.should_detach_after_tubelet(tubelet_index, grid.depth)
                {
                    state.detach();
                }
                continue;
            };
            let image = video
                .clone()
                .slice_dim(2, frame..frame + 1)
                .reshape([batch, channels, height, width]);
            let target_frame = rollout_target_frame(
                target_tokens.as_ref(),
                tubelet_index,
                frame_tokens,
                Some(&frame_mask),
                batch,
                &device,
            );
            let patchify_plan =
                SparsePatchifyPlan::new(frame_mask.clone(), frame_grid, batch, &device)?;
            let encoder_plan = SparseEncoderPlan::new(
                &self.config,
                frame_mask,
                frame_grid,
                batch,
                false,
                &device,
            )?;
            let tokens = self.sparse_patchify_image_cuda_frozen(image, &patchify_plan)?;
            let encoded =
                self.forward_sparse_tokens_with_plan(tokens, &encoder_plan, target_frame, state)?;
            if frame % tubelet == tubelet - 1 {
                outputs.push(encoded.tokens);
                if self.should_detach_after_tubelet(tubelet_index, grid.depth) {
                    state.detach();
                }
            }
        }
        ensure!(
            !outputs.is_empty(),
            "single-frame sparse patchify rollout produced no output tokens"
        );
        let tokens = Tensor::cat(outputs, 1);
        let plan = SparseEncoderPlan::new(&self.config, mask.clone(), grid, batch, true, &device)?;
        Ok(VJepaEncoderOutput {
            tokens,
            hierarchical: Vec::new(),
            token_indices: plan.positions.indices,
            grid,
        })
    }

    fn sparse_patchify_image_cuda_frozen(
        &self,
        image: Tensor<burn::backend::Autodiff<burn_flex_gmm::cuda::DefaultCudaBackend>, 4>,
        plan: &SparsePatchifyPlan<burn::backend::Autodiff<burn_flex_gmm::cuda::DefaultCudaBackend>>,
    ) -> Result<Tensor<burn::backend::Autodiff<burn_flex_gmm::cuda::DefaultCudaBackend>, 3>> {
        let [batch, channels, height, width] = image.shape().dims::<4>();
        ensure!(
            batch == plan.batch,
            "image batch does not match sparse patchify plan"
        );
        ensure!(
            channels == self.config.in_channels,
            "image channel count does not match V-JEPA config"
        );
        let grid = TokenGridShape::new(
            1,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            grid == plan.grid,
            "image token grid does not match sparse patchify plan"
        );
        let device = image.device();
        let patchify_config = burn_flex_gmm::SparsePatchify3dConfig {
            in_channels: channels,
            out_channels: self.config.encoder.embed_dim,
            frames: 1,
            height,
            width,
            tubelet_size: 1,
            patch_h: self.config.patch_size,
            patch_w: self.config.patch_size,
        };
        let bias = self
            .base
            .image_patch_embed
            .proj
            .bias
            .as_ref()
            .map(|bias| bias.val().inner())
            .unwrap_or_else(|| {
                Tensor::<burn_flex_gmm::cuda::DefaultCudaBackend, 1>::zeros(
                    [self.config.encoder.embed_dim],
                    &device,
                )
            });
        let tokens = burn_flex_gmm::cuda::sparse_patchify3d_forward_cuda(
            &patchify_config,
            image.inner().reshape([batch, channels, 1, height, width]),
            plan.coords.clone().inner(),
            self.base.image_patch_embed.proj.weight.val().inner(),
            bias,
        )
        .map_err(anyhow::Error::msg)?
        .reshape([batch, plan.token_count(), self.config.encoder.embed_dim]);
        Ok(Tensor::from_inner(tokens))
    }
}

#[cfg(feature = "sparse-patchify-cuda")]
impl VJepaTttEncoder<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>> {
    pub fn forward_single_frame_rollout_sparse_patchify_cuda_fusion_frozen(
        &self,
        video: Tensor<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>, 5>,
        mask: &SparseTokenMask,
        target_tokens: Option<Tensor<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>, 3>>,
        state: &mut TttState<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>,
    ) -> Result<VJepaEncoderOutput<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>> {
        let [batch, channels, frames, height, width] = video.shape().dims::<5>();
        let tubelet = self.config.tubelet_size.max(1);
        ensure!(
            frames % tubelet == 0,
            "single-frame sparse rollout requires frames divisible by tubelet_size"
        );
        let grid = TokenGridShape::new(
            frames / tubelet,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            mask.dense_len() == grid.len() && !mask.is_empty(),
            "single-frame sparse rollout mask must match a non-empty video token grid"
        );
        let frame_grid = TokenGridShape::new(1, grid.height, grid.width);
        let frame_tokens = frame_grid.len();
        let device = video.device();
        let mut outputs = Vec::with_capacity(grid.depth);
        for frame in 0..frames {
            let tubelet_index = frame / tubelet;
            let Some(frame_mask) = sparse_rollout_frame_mask(mask, grid, tubelet_index)? else {
                if frame % tubelet == tubelet - 1
                    && self.should_detach_after_tubelet(tubelet_index, grid.depth)
                {
                    state.detach();
                }
                continue;
            };
            let image = video
                .clone()
                .slice_dim(2, frame..frame + 1)
                .reshape([batch, channels, height, width]);
            let target_frame = rollout_target_frame(
                target_tokens.as_ref(),
                tubelet_index,
                frame_tokens,
                Some(&frame_mask),
                batch,
                &device,
            );
            let patchify_plan =
                SparsePatchifyPlan::new(frame_mask.clone(), frame_grid, batch, &device)?;
            let encoder_plan = SparseEncoderPlan::new(
                &self.config,
                frame_mask,
                frame_grid,
                batch,
                false,
                &device,
            )?;
            let tokens = self.sparse_patchify_image_cuda_fusion_frozen(image, &patchify_plan)?;
            let encoded =
                self.forward_sparse_tokens_with_plan(tokens, &encoder_plan, target_frame, state)?;
            if frame % tubelet == tubelet - 1 {
                outputs.push(encoded.tokens);
                if self.should_detach_after_tubelet(tubelet_index, grid.depth) {
                    state.detach();
                }
            }
        }
        ensure!(
            !outputs.is_empty(),
            "single-frame sparse patchify rollout produced no output tokens"
        );
        let tokens = Tensor::cat(outputs, 1);
        let plan = SparseEncoderPlan::new(&self.config, mask.clone(), grid, batch, true, &device)?;
        Ok(VJepaEncoderOutput {
            tokens,
            hierarchical: Vec::new(),
            token_indices: plan.positions.indices,
            grid,
        })
    }

    fn sparse_patchify_image_cuda_fusion_frozen(
        &self,
        image: Tensor<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>, 4>,
        plan: &SparsePatchifyPlan<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>,
    ) -> Result<Tensor<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>, 3>> {
        let [batch, channels, height, width] = image.shape().dims::<4>();
        ensure!(
            batch == plan.batch,
            "image batch does not match sparse patchify plan"
        );
        ensure!(
            channels == self.config.in_channels,
            "image channel count does not match V-JEPA config"
        );
        let grid = TokenGridShape::new(
            1,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            grid == plan.grid,
            "image token grid does not match sparse patchify plan"
        );
        let device = image.device();
        let patchify_config = burn_flex_gmm::SparsePatchify3dConfig {
            in_channels: channels,
            out_channels: self.config.encoder.embed_dim,
            frames: 1,
            height,
            width,
            tubelet_size: 1,
            patch_h: self.config.patch_size,
            patch_w: self.config.patch_size,
        };
        let bias = self
            .base
            .image_patch_embed
            .proj
            .bias
            .as_ref()
            .map(|bias| bias.val().inner())
            .unwrap_or_else(|| {
                Tensor::<burn::backend::Cuda<f32, i32>, 1>::zeros(
                    [self.config.encoder.embed_dim],
                    &device,
                )
            });
        let tokens = crate::sparse_patchify::sparse_patchify3d_forward_cuda_fusion(
            &patchify_config,
            image.inner().reshape([batch, channels, 1, height, width]),
            plan.coords.clone().inner(),
            self.base.image_patch_embed.proj.weight.val().inner(),
            bias,
        )
        .reshape([batch, plan.token_count(), self.config.encoder.embed_dim]);
        Ok(Tensor::from_inner(tokens))
    }

    pub fn forward_single_frame_rollout_sparse_patchify_cuda_fusion_frozen_batch(
        &self,
        video: Tensor<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>, 5>,
        mask: &SparseMaskBatch<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>,
        target_tokens: Option<Tensor<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>, 3>>,
        state: &mut TttState<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>,
    ) -> Result<VJepaEncoderOutput<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>> {
        if let Some(mask) = mask.uniform_mask() {
            return self.forward_single_frame_rollout_sparse_patchify_cuda_fusion_frozen(
                video,
                mask,
                target_tokens,
                state,
            );
        }
        let [batch, channels, frames, height, width] = video.shape().dims::<5>();
        let tubelet = self.config.tubelet_size.max(1);
        ensure!(
            frames % tubelet == 0,
            "single-frame sparse rollout requires frames divisible by tubelet_size"
        );
        ensure!(
            batch == mask.batch(),
            "single-frame sparse rollout batch mask must match video batch"
        );
        let grid = TokenGridShape::new(
            frames / tubelet,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            mask.dense_len() == grid.len() && !mask.is_empty(),
            "single-frame sparse rollout batch mask must match a non-empty video token grid"
        );
        let frame_grid = TokenGridShape::new(1, grid.height, grid.width);
        let frame_tokens = frame_grid.len();
        let device = video.device();
        let mut outputs = Vec::with_capacity(grid.depth);
        for frame in 0..frames {
            let tubelet_index = frame / tubelet;
            let Some(frame_mask) =
                sparse_rollout_frame_mask_batch(mask, grid, tubelet_index, &device)?
            else {
                if frame % tubelet == tubelet - 1
                    && self.should_detach_after_tubelet(tubelet_index, grid.depth)
                {
                    state.detach();
                }
                continue;
            };
            let image = video
                .clone()
                .slice_dim(2, frame..frame + 1)
                .reshape([batch, channels, height, width]);
            let target_frame = rollout_target_frame_batch(
                target_tokens.as_ref(),
                tubelet_index,
                frame_tokens,
                &frame_mask,
            );
            let patchify_plan =
                SparsePatchifyBatchPlan::new(frame_mask.clone(), frame_grid, &device)?;
            let encoder_plan =
                SparseEncoderBatchPlan::new(&self.config, frame_mask, frame_grid, false, &device)?;
            let tokens =
                self.sparse_patchify_image_cuda_fusion_frozen_batch(image, &patchify_plan)?;
            let encoded = self.forward_sparse_tokens_with_batch_plan(
                tokens,
                &encoder_plan,
                target_frame,
                state,
            )?;
            if frame % tubelet == tubelet - 1 {
                outputs.push(encoded.tokens);
                if self.should_detach_after_tubelet(tubelet_index, grid.depth) {
                    state.detach();
                }
            }
        }
        ensure!(
            !outputs.is_empty(),
            "single-frame sparse patchify rollout produced no output tokens"
        );
        let tokens = Tensor::cat(outputs, 1);
        let plan = SparseEncoderBatchPlan::new(&self.config, mask.clone(), grid, true, &device)?;
        Ok(VJepaEncoderOutput {
            tokens,
            hierarchical: Vec::new(),
            token_indices: plan.positions.indices,
            grid,
        })
    }

    fn sparse_patchify_image_cuda_fusion_frozen_batch(
        &self,
        image: Tensor<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>, 4>,
        plan: &SparsePatchifyBatchPlan<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>,
    ) -> Result<Tensor<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>, 3>> {
        let [batch, channels, height, width] = image.shape().dims::<4>();
        ensure!(
            batch == plan.batch,
            "image batch does not match sparse patchify batch plan"
        );
        ensure!(
            channels == self.config.in_channels,
            "image channel count does not match V-JEPA config"
        );
        let grid = TokenGridShape::new(
            1,
            height / self.config.patch_size.max(1),
            width / self.config.patch_size.max(1),
        );
        ensure!(
            grid == plan.grid,
            "image token grid does not match sparse patchify batch plan"
        );
        let device = image.device();
        let patchify_config = burn_flex_gmm::SparsePatchify3dConfig {
            in_channels: channels,
            out_channels: self.config.encoder.embed_dim,
            frames: 1,
            height,
            width,
            tubelet_size: 1,
            patch_h: self.config.patch_size,
            patch_w: self.config.patch_size,
        };
        let bias = self
            .base
            .image_patch_embed
            .proj
            .bias
            .as_ref()
            .map(|bias| bias.val().inner())
            .unwrap_or_else(|| {
                Tensor::<burn::backend::Cuda<f32, i32>, 1>::zeros(
                    [self.config.encoder.embed_dim],
                    &device,
                )
            });
        let tokens = crate::sparse_patchify::sparse_patchify3d_forward_cuda_fusion(
            &patchify_config,
            image.inner().reshape([batch, channels, 1, height, width]),
            plan.coords.clone().inner(),
            self.base.image_patch_embed.proj.weight.val().inner(),
            bias,
        )
        .reshape([batch, plan.token_count(), self.config.encoder.embed_dim]);
        Ok(Tensor::from_inner(tokens))
    }
}
