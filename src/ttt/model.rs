use super::config::TttEncoderConfig;
use super::encoder::VJepaTttEncoder;
use super::state::TttState;
use crate::{
    SparseMaskBatch, SparsePredictorPlan, SparseTokenMask, SparseVJepaForwardOutput, VJepa2_1Model,
    VJepaConfig, VJepaEncoderOutput, VJepaPredictor,
};
use anyhow::Result;
use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct VJepaTttModel<B: Backend> {
    pub encoder: VJepaTttEncoder<B>,
    pub predictor: VJepaPredictor<B>,
    #[module(skip)]
    config: VJepaConfig,
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
        let encoder = VJepaTttEncoder::new(model.encoder, &model_config, ttt_config, device)?;
        Ok(Self {
            encoder,
            predictor: model.predictor,
            config: model_config,
        })
    }

    pub fn config(&self) -> &VJepaConfig {
        &self.config
    }

    pub fn fresh_state(&self) -> TttState<B> {
        self.encoder.fresh_state()
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
            self.predictor
                .forward_sparse_with_plan(context.tokens.clone(), &plan, 0)?;
        Ok(SparseVJepaForwardOutput {
            context,
            target,
            predictor,
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
