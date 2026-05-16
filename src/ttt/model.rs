use super::config::{TttBackpropMode, TttEncoderConfig};
use super::encoder::VJepaTttEncoder;
use super::layer::VJepaTttLayer;
use super::state::TttState;
use crate::{
    SparseMaskBatch, SparsePredictorPlan, SparseTokenMask, SparseVJepaForwardOutput, VJepa2_1Model,
    VJepaConfig, VJepaEncoderOutput, VJepaPredictor, VJepaPredictorOutput, apply_token_mask,
};
use anyhow::Result;
use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct VJepaTttModel<B: Backend> {
    pub encoder: VJepaTttEncoder<B>,
    pub predictor: VJepaPredictor<B>,
    pub predictor_ttt_layers: Option<Vec<VJepaTttLayer<B>>>,
    #[module(skip)]
    config: VJepaConfig,
    #[module(skip)]
    predictor_layer_indices: Vec<usize>,
}

impl<B: Backend> VJepaTttModel<B> {
    pub fn from_model(
        model: VJepa2_1Model<B>,
        ttt_config: TttEncoderConfig,
        device: &B::Device,
    ) -> Result<Self> {
        let model_config = model.config().clone();
        let model = if ttt_config.freeze_pretrained {
            model.no_grad()
        } else {
            model
        };
        let predictor_layer_indices = ttt_config.resolved_predictor_layers(&model_config);
        let predictor_ttt_layers = predictor_layer_indices
            .iter()
            .map(|_| VJepaTttLayer::new(model_config.predictor.embed_dim, &ttt_config, device))
            .collect();
        let encoder = VJepaTttEncoder::new(model.encoder, &model_config, ttt_config, device)?;
        Ok(Self {
            encoder,
            predictor: model.predictor,
            predictor_ttt_layers: Some(predictor_ttt_layers),
            config: model_config,
            predictor_layer_indices,
        })
    }

    pub fn config(&self) -> &VJepaConfig {
        &self.config
    }

    pub fn fresh_state(&self) -> TttState<B> {
        self.encoder.fresh_state()
    }

    pub fn predictor_ttt_layer_indices(&self) -> &[usize] {
        &self.predictor_layer_indices
    }

    fn predictor_ttt_layers(&self) -> &[VJepaTttLayer<B>] {
        self.predictor_ttt_layers.as_deref().unwrap_or_default()
    }

    pub fn set_backprop_mode(&mut self, mode: TttBackpropMode) {
        self.encoder.set_backprop_mode(mode);
    }

    pub fn encode_video(
        &self,
        video: Tensor<B, 5>,
        mask: Option<&SparseTokenMask>,
    ) -> Result<VJepaEncoderOutput<B>> {
        self.encoder.forward_video(video, mask)
    }

    pub fn forward_single_frame_rollout(
        &self,
        video: Tensor<B, 5>,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        self.encoder
            .forward_single_frame_rollout(video, target_tokens, state)
    }

    pub fn forward_single_frame_rollout_sparse(
        &self,
        video: Tensor<B, 5>,
        mask: &SparseTokenMask,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        self.encoder
            .forward_single_frame_rollout_sparse(video, mask, target_tokens, state)
    }

    pub fn forward_single_frame_rollout_sparse_batch(
        &self,
        video: Tensor<B, 5>,
        mask: &SparseMaskBatch<B>,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        self.encoder
            .forward_single_frame_rollout_sparse_batch(video, mask, target_tokens, state)
    }

    pub fn encode_image_batch_with_state(
        &self,
        image: Tensor<B, 4>,
        mask: SparseMaskBatch<B>,
        target_tokens: Option<Tensor<B, 3>>,
        state: &mut TttState<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        self.encoder
            .forward_image_with_mask_batch_state(image, mask, target_tokens, state)
    }

    pub fn forward_sparse(
        &self,
        video: Tensor<B, 5>,
        context_mask: &SparseTokenMask,
        target_mask: &SparseTokenMask,
    ) -> Result<SparseVJepaForwardOutput<B>> {
        let context = self
            .encoder
            .forward_video(video.clone(), Some(context_mask))?;
        let target = self.encoder.forward_video(video, Some(target_mask))?;
        let plan = SparsePredictorPlan::new(
            &self.config,
            context_mask.clone(),
            target_mask.clone(),
            context.grid,
            context.tokens.shape().dims::<3>()[0],
            &context.tokens.device(),
        )?;
        let predictor =
            self.forward_predictor_sparse_with_plan(context.tokens.clone(), &plan, 0)?;
        Ok(SparseVJepaForwardOutput {
            context,
            target,
            predictor,
        })
    }

    pub fn forward_predictor_sparse(
        &self,
        context_tokens: Tensor<B, 3>,
        context_mask: &SparseTokenMask,
        target_mask: &SparseTokenMask,
        grid: crate::TokenGridShape,
        mask_index: usize,
    ) -> Result<VJepaPredictorOutput<B>> {
        if self.predictor_ttt_layers().is_empty() {
            return self.predictor.forward_sparse(
                context_tokens,
                context_mask,
                target_mask,
                grid,
                mask_index,
            );
        }
        let batch = context_tokens.shape().dims::<3>()[0];
        let device = context_tokens.device();
        let plan = SparsePredictorPlan::new(
            &self.config,
            context_mask.clone(),
            target_mask.clone(),
            grid,
            batch,
            &device,
        )?;
        self.forward_predictor_sparse_with_plan(context_tokens, &plan, mask_index)
    }

    pub fn forward_predictor_sparse_with_plan(
        &self,
        context_tokens: Tensor<B, 3>,
        plan: &SparsePredictorPlan<B>,
        mask_index: usize,
    ) -> Result<VJepaPredictorOutput<B>> {
        let predictor_ttt_layers = self.predictor_ttt_layers();
        if predictor_ttt_layers.is_empty() {
            return self
                .predictor
                .forward_sparse_with_plan(context_tokens, plan, mask_index);
        }
        let context_mask = &plan.context_mask;
        let target_mask = &plan.target_mask;
        let [batch, context_len, _encoder_dim] = context_tokens.shape().dims::<3>();
        anyhow::ensure!(
            batch == plan.batch,
            "context token batch does not match sparse predictor plan"
        );
        anyhow::ensure!(
            context_len == context_mask.len(),
            "context token shape does not match context mask"
        );
        let mut context = self.predictor.predictor_embed.forward(context_tokens);
        if let Some(position_embed) = &plan.context_position_embed {
            context = context + position_embed.clone();
        }

        let target_len = target_mask.len();
        let token = self.predictor.mask_tokens[mask_index % self.predictor.mask_tokens.len()]
            .val()
            .reshape([1, 1, self.config.predictor.embed_dim])
            .repeat_dim(0, batch)
            .repeat_dim(1, target_len);
        let target = if let Some(position_embed) = &plan.target_position_embed {
            token + position_embed.clone()
        } else {
            token
        };
        let mut sequence = Tensor::cat(vec![context, target], 1);
        sequence = apply_token_mask(sequence, plan.sort_order.clone());
        if self.config.predictor.modality_embedding {
            let token_count = context_len + target_len;
            let embed = self
                .predictor
                .video_mod_embed
                .val()
                .reshape([1, 1, self.config.predictor.embed_dim])
                .repeat_dim(0, batch)
                .repeat_dim(1, token_count);
            sequence = sequence + embed;
        }

        let mut state = TttState::new(predictor_ttt_layers.len());
        for (layer_index, block) in self.predictor.blocks.iter().enumerate() {
            sequence = block.forward(sequence, Some(&plan.positions));
            if let Ok(ttt_index) = self.predictor_layer_indices.binary_search(&layer_index) {
                sequence = predictor_ttt_layers[ttt_index].forward_with_options(
                    sequence,
                    None,
                    &mut state.layers[ttt_index],
                    true,
                );
            }
        }
        sequence = self.predictor.norm.forward(sequence);
        sequence = apply_token_mask(sequence, plan.reverse_order.clone());
        let context_predictions = self.config.predictor.return_all_tokens.then(|| {
            let context = sequence.clone().slice_dim(1, 0..context_len);
            self.predictor
                .context_proj
                .as_ref()
                .expect("context projection")
                .forward(context)
        });
        let target_predictions = self.predictor.target_proj.forward(
            sequence
                .clone()
                .slice_dim(1, context_len..context_len + target_len),
        );
        Ok(VJepaPredictorOutput {
            target_predictions,
            context_predictions,
            sequence_tokens: sequence,
            sequence_indices: plan.sequence_indices.clone(),
        })
    }
}

#[cfg(feature = "sparse-patchify-wgpu")]
impl VJepaTttModel<burn_flex_gmm::wgpu::DefaultWgpuBackend> {
    pub fn forward_single_frame_rollout_sparse_patchify_wgpu(
        &self,
        video: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 5>,
        mask: &SparseTokenMask,
        target_tokens: Option<Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 3>>,
        state: &mut TttState<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        self.encoder
            .forward_single_frame_rollout_sparse_patchify_wgpu(video, mask, target_tokens, state)
    }
}

#[cfg(feature = "sparse-patchify-wgpu")]
impl VJepaTttModel<burn::backend::Wgpu<f32, i32>> {
    pub fn forward_image_sparse_patchify_wgpu_fusion_batch_state(
        &self,
        image: Tensor<burn::backend::Wgpu<f32, i32>, 4>,
        plan: &crate::SparsePatchifyBatchPlan<burn::backend::Wgpu<f32, i32>>,
        target_tokens: Option<Tensor<burn::backend::Wgpu<f32, i32>, 3>>,
        state: &mut TttState<burn::backend::Wgpu<f32, i32>>,
    ) -> Result<VJepaEncoderOutput<burn::backend::Wgpu<f32, i32>>> {
        self.encoder
            .forward_image_sparse_patchify_wgpu_fusion_batch_state(
                image,
                plan,
                target_tokens,
                state,
            )
    }

    pub fn forward_single_frame_rollout_sparse_patchify_wgpu_fusion(
        &self,
        video: Tensor<burn::backend::Wgpu<f32, i32>, 5>,
        mask: &SparseTokenMask,
        target_tokens: Option<Tensor<burn::backend::Wgpu<f32, i32>, 3>>,
        state: &mut TttState<burn::backend::Wgpu<f32, i32>>,
    ) -> Result<VJepaEncoderOutput<burn::backend::Wgpu<f32, i32>>> {
        self.encoder
            .forward_single_frame_rollout_sparse_patchify_wgpu_fusion(
                video,
                mask,
                target_tokens,
                state,
            )
    }
}

#[cfg(feature = "sparse-patchify-cuda")]
impl VJepaTttModel<burn_flex_gmm::cuda::DefaultCudaBackend> {
    pub fn forward_single_frame_rollout_sparse_patchify_cuda(
        &self,
        video: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 5>,
        mask: &SparseTokenMask,
        target_tokens: Option<Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 3>>,
        state: &mut TttState<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<VJepaEncoderOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        self.encoder
            .forward_single_frame_rollout_sparse_patchify_cuda(video, mask, target_tokens, state)
    }
}

#[cfg(feature = "sparse-patchify-wgpu")]
impl VJepaTttModel<burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
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
        self.encoder
            .forward_single_frame_rollout_sparse_patchify_wgpu_frozen(
                video,
                mask,
                target_tokens,
                state,
            )
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
        self.encoder
            .forward_single_frame_rollout_sparse_patchify_wgpu_frozen_batch(
                video,
                mask,
                target_tokens,
                state,
            )
    }
}

#[cfg(feature = "sparse-patchify-cuda")]
impl VJepaTttModel<burn::backend::Autodiff<burn_flex_gmm::cuda::DefaultCudaBackend>> {
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
        self.encoder
            .forward_single_frame_rollout_sparse_patchify_cuda_frozen(
                video,
                mask,
                target_tokens,
                state,
            )
    }
}

#[cfg(feature = "sparse-patchify-cuda")]
impl VJepaTttModel<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>> {
    pub fn forward_single_frame_rollout_sparse_patchify_cuda_fusion_frozen(
        &self,
        video: Tensor<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>, 5>,
        mask: &SparseTokenMask,
        target_tokens: Option<Tensor<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>, 3>>,
        state: &mut TttState<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>,
    ) -> Result<VJepaEncoderOutput<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>> {
        self.encoder
            .forward_single_frame_rollout_sparse_patchify_cuda_fusion_frozen(
                video,
                mask,
                target_tokens,
                state,
            )
    }

    pub fn forward_single_frame_rollout_sparse_patchify_cuda_fusion_frozen_batch(
        &self,
        video: Tensor<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>, 5>,
        mask: &SparseMaskBatch<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>,
        target_tokens: Option<Tensor<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>, 3>>,
        state: &mut TttState<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>,
    ) -> Result<VJepaEncoderOutput<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>> {
        self.encoder
            .forward_single_frame_rollout_sparse_patchify_cuda_fusion_frozen_batch(
                video,
                mask,
                target_tokens,
                state,
            )
    }
}
