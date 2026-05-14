use crate::{JepaSampleMetadata, SparseMaskBatch, SparseTokenMask, VJepa2_1Model, VJepaTttModel};
use anyhow::{Result, bail, ensure};
use burn::tensor::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};

use crate::training::config::{BurnJepaTrainConfig, TttSparsePatchifyTrainingMode};
use crate::training::mask::TrainingMaskConfig;

#[derive(Clone, Debug)]
pub(super) struct ResolvedTttMasks<B: Backend> {
    pub context: SparseMaskBatch<B>,
    pub target: SparseMaskBatch<B>,
}

impl<B: Backend> ResolvedTttMasks<B> {
    pub fn new(context: SparseMaskBatch<B>, target: SparseMaskBatch<B>) -> Self {
        Self { context, target }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TttRolloutKind {
    Dense,
    SparseContext,
    SparseTarget,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TttPatchifyKind {
    DensePatchEmbed,
    FrozenSparsePatchify,
}

#[derive(Debug)]
pub(super) struct StudentEvalRollout<B: Backend> {
    pub primary: crate::VJepaEncoderOutput<B>,
    pub full: Option<crate::VJepaEncoderOutput<B>>,
}

impl<B: Backend> StudentEvalRollout<B> {
    pub fn full_tokens(&self) -> Option<Tensor<B, 3>> {
        self.full.as_ref().map(|output| output.tokens.clone())
    }
}

pub(super) fn masks_required(config: &BurnJepaTrainConfig) -> bool {
    config.training.mask.is_some() || config.loss.predictor_loss_weight > 0.0
}

pub(super) fn resolve_masks<B: Backend>(
    config: &BurnJepaTrainConfig,
    video: &Tensor<B, 5>,
    model_config: &crate::VJepaConfig,
    metadata: &[JepaSampleMetadata],
) -> Result<Option<ResolvedTttMasks<B>>> {
    if masks_required(config) {
        let [_, _, frames, height, width] = video.shape().dims::<5>();
        let grid = crate::video_token_grid(model_config, frames, height, width)?;
        if matches!(
            config.training.mask_config(),
            TrainingMaskConfig::ManifestPrecomputedMasks
        ) {
            let (context_rows, target_rows) = manifest_mask_rows(metadata)?;
            let device = video.device();
            return Ok(Some(ResolvedTttMasks::new(
                SparseMaskBatch::from_rows(context_rows, grid.len(), &device)?,
                SparseMaskBatch::from_rows(target_rows, grid.len(), &device)?,
            )));
        }
        let (context, target) = config.training.resolve_masks_for_grid_with_metadata(
            video,
            model_config,
            grid,
            metadata,
        )?;
        let device = video.device();
        Ok(Some(ResolvedTttMasks::new(
            SparseMaskBatch::uniform(context, video.shape().dims::<5>()[0], &device)?,
            SparseMaskBatch::uniform(target, video.shape().dims::<5>()[0], &device)?,
        )))
    } else {
        Ok(None)
    }
}

fn manifest_mask_rows(
    metadata: &[JepaSampleMetadata],
) -> Result<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
    ensure!(
        !metadata.is_empty(),
        "training.mask.manifest_precomputed_masks requires batch metadata"
    );
    let context = metadata
        .iter()
        .map(|row| {
            row.precomputed_context_indices.clone().ok_or_else(|| {
                anyhow::anyhow!("manifest row is missing precomputed_context_indices")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let target = metadata
        .iter()
        .map(|row| {
            row.precomputed_target_indices.clone().ok_or_else(|| {
                anyhow::anyhow!("manifest row is missing precomputed_target_indices")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((context, target))
}

pub(super) fn rollout_kind(config: &BurnJepaTrainConfig) -> TttRolloutKind {
    if config
        .training
        .use_sparse_rollout(config.loss.predictor_loss_weight)
    {
        match config.training.sparse_rollout {
            crate::training::config::TttSparseRolloutMode::ContextMask => {
                TttRolloutKind::SparseContext
            }
            _ => TttRolloutKind::SparseTarget,
        }
    } else {
        TttRolloutKind::Dense
    }
}

pub(super) fn patchify_kind<B: TttSparsePatchifyTrainingBackend>(
    config: &BurnJepaTrainConfig,
    rollout: TttRolloutKind,
) -> Result<TttPatchifyKind> {
    let can_use_sparse_patchify = rollout.sparse_mask_kind().is_some()
        && config.ttt.freeze_pretrained
        && B::frozen_sparse_patchify_supported();
    match config.training.sparse_patchify_training {
        TttSparsePatchifyTrainingMode::DensePatchEmbed => Ok(TttPatchifyKind::DensePatchEmbed),
        TttSparsePatchifyTrainingMode::Auto if can_use_sparse_patchify => {
            Ok(TttPatchifyKind::FrozenSparsePatchify)
        }
        TttSparsePatchifyTrainingMode::Auto => Ok(TttPatchifyKind::DensePatchEmbed),
        TttSparsePatchifyTrainingMode::FrozenSparsePatchify => {
            ensure!(
                rollout.sparse_mask_kind().is_some(),
                "frozen sparse patchify requires sparse rollout"
            );
            ensure!(
                config.ttt.freeze_pretrained,
                "frozen sparse patchify requires frozen pretrained V-JEPA weights"
            );
            ensure!(
                B::frozen_sparse_patchify_supported(),
                "frozen sparse patchify is not available for the selected backend/features"
            );
            Ok(TttPatchifyKind::FrozenSparsePatchify)
        }
    }
}

pub(super) fn teacher_tokens<B: Backend>(
    teacher: &VJepa2_1Model<B>,
    video: Tensor<B, 5>,
) -> Tensor<B, 3> {
    teacher.encode_video(video, None).tokens.detach()
}

pub(super) fn student_rollout<B: Backend>(
    model: &VJepaTttModel<B>,
    video: Tensor<B, 5>,
    teacher_tokens: Tensor<B, 3>,
    masks: Option<&ResolvedTttMasks<B>>,
    rollout: TttRolloutKind,
) -> Result<crate::VJepaEncoderOutput<B>> {
    let mut state = model.fresh_state();
    match rollout {
        TttRolloutKind::Dense => {
            model.forward_single_frame_rollout(video, Some(teacher_tokens), &mut state)
        }
        TttRolloutKind::SparseContext | TttRolloutKind::SparseTarget => {
            let Some(masks) = masks else {
                bail!("sparse TTT rollout requires resolved masks")
            };
            let mask = rollout.select_mask(masks);
            ensure!(
                !mask.is_empty(),
                "sparse TTT rollout requires a non-empty target mask"
            );
            model.forward_single_frame_rollout_sparse_batch(
                video,
                mask,
                Some(teacher_tokens),
                &mut state,
            )
        }
    }
}

pub(super) fn student_training_rollout<B: TttSparsePatchifyTrainingBackend>(
    model: &VJepaTttModel<B>,
    video: Tensor<B, 5>,
    teacher_tokens: Tensor<B, 3>,
    masks: Option<&ResolvedTttMasks<B>>,
    rollout: TttRolloutKind,
    patchify: TttPatchifyKind,
) -> Result<crate::VJepaEncoderOutput<B>> {
    if rollout.sparse_mask_kind().is_some() && patchify == TttPatchifyKind::FrozenSparsePatchify {
        let Some(masks) = masks else {
            bail!("frozen sparse patchify rollout requires resolved masks")
        };
        let mask = rollout.select_mask(masks);
        ensure!(
            !mask.is_empty(),
            "frozen sparse patchify rollout requires a non-empty target mask"
        );
        if let Some(mask) = mask.uniform_mask() {
            let mut state = model.fresh_state();
            return B::student_frozen_sparse_patchify_rollout(
                model,
                video,
                mask,
                Some(teacher_tokens),
                &mut state,
            );
        } else if B::frozen_sparse_patchify_batch_supported() {
            let mut state = model.fresh_state();
            return B::student_frozen_sparse_patchify_rollout_batch(
                model,
                video,
                mask,
                Some(teacher_tokens),
                &mut state,
            );
        }
    }
    student_rollout(model, video, teacher_tokens, masks, rollout)
}

pub(super) fn student_eval_rollout<B: TttSparsePatchifyTrainingBackend>(
    model: &VJepaTttModel<B>,
    video: Tensor<B, 5>,
    teacher_tokens: Tensor<B, 3>,
    masks: Option<&ResolvedTttMasks<B>>,
    rollout: TttRolloutKind,
    patchify: TttPatchifyKind,
    eval_full_grid: bool,
) -> Result<StudentEvalRollout<B>> {
    let primary = student_training_rollout(
        model,
        video.clone(),
        teacher_tokens.clone(),
        masks,
        rollout,
        patchify,
    )?;
    let full = (eval_full_grid && rollout.sparse_mask_kind().is_some())
        .then(|| {
            let mut full_state = model.fresh_state();
            model.forward_single_frame_rollout(video, Some(teacher_tokens), &mut full_state)
        })
        .transpose()?;
    Ok(StudentEvalRollout { primary, full })
}

impl TttRolloutKind {
    pub(super) fn sparse_mask_kind(self) -> Option<SparseMaskKind> {
        match self {
            Self::Dense => None,
            Self::SparseContext => Some(SparseMaskKind::Context),
            Self::SparseTarget => Some(SparseMaskKind::Target),
        }
    }

    pub(super) fn select_mask<'a, B: Backend>(
        self,
        masks: &'a ResolvedTttMasks<B>,
    ) -> &'a SparseMaskBatch<B> {
        match self {
            Self::Dense | Self::SparseTarget => &masks.target,
            Self::SparseContext => &masks.context,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SparseMaskKind {
    Context,
    Target,
}

pub trait TttSparsePatchifyTrainingBackend: AutodiffBackend {
    fn frozen_sparse_patchify_supported() -> bool {
        false
    }

    fn frozen_sparse_patchify_batch_supported() -> bool {
        false
    }

    fn student_frozen_sparse_patchify_rollout(
        _model: &VJepaTttModel<Self>,
        _video: Tensor<Self, 5>,
        _mask: &SparseTokenMask,
        _target_tokens: Option<Tensor<Self, 3>>,
        _state: &mut crate::TttState<Self>,
    ) -> Result<crate::VJepaEncoderOutput<Self>> {
        bail!("frozen sparse patchify is not available for this backend")
    }

    fn student_frozen_sparse_patchify_rollout_batch(
        _model: &VJepaTttModel<Self>,
        _video: Tensor<Self, 5>,
        _mask: &SparseMaskBatch<Self>,
        _target_tokens: Option<Tensor<Self, 3>>,
        _state: &mut crate::TttState<Self>,
    ) -> Result<crate::VJepaEncoderOutput<Self>> {
        bail!("batched frozen sparse patchify is not available for this backend")
    }
}

#[cfg(feature = "ndarray")]
impl TttSparsePatchifyTrainingBackend for burn::backend::Autodiff<burn::backend::NdArray<f32>> {}

#[cfg(feature = "webgpu")]
impl TttSparsePatchifyTrainingBackend for burn::backend::Autodiff<burn::backend::WebGpu<f32, i32>> {}

#[cfg(feature = "wgpu")]
impl TttSparsePatchifyTrainingBackend for burn::backend::Autodiff<burn::backend::Wgpu<f32, i32>> {}

#[cfg(feature = "sparse-patchify-wgpu")]
impl TttSparsePatchifyTrainingBackend
    for burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>
{
    fn frozen_sparse_patchify_supported() -> bool {
        true
    }

    fn frozen_sparse_patchify_batch_supported() -> bool {
        true
    }

    fn student_frozen_sparse_patchify_rollout(
        model: &VJepaTttModel<Self>,
        video: Tensor<Self, 5>,
        mask: &SparseTokenMask,
        target_tokens: Option<Tensor<Self, 3>>,
        state: &mut crate::TttState<Self>,
    ) -> Result<crate::VJepaEncoderOutput<Self>> {
        model.forward_single_frame_rollout_sparse_patchify_wgpu_frozen(
            video,
            mask,
            target_tokens,
            state,
        )
    }

    fn student_frozen_sparse_patchify_rollout_batch(
        model: &VJepaTttModel<Self>,
        video: Tensor<Self, 5>,
        mask: &SparseMaskBatch<Self>,
        target_tokens: Option<Tensor<Self, 3>>,
        state: &mut crate::TttState<Self>,
    ) -> Result<crate::VJepaEncoderOutput<Self>> {
        model.forward_single_frame_rollout_sparse_patchify_wgpu_frozen_batch(
            video,
            mask,
            target_tokens,
            state,
        )
    }
}

#[cfg(all(feature = "cuda", not(feature = "sparse-patchify-cuda")))]
impl TttSparsePatchifyTrainingBackend for burn::backend::Autodiff<burn::backend::Cuda<f32, i32>> {}

#[cfg(feature = "sparse-patchify-cuda")]
impl TttSparsePatchifyTrainingBackend for burn::backend::Autodiff<burn::backend::Cuda<f32, i32>> {
    fn frozen_sparse_patchify_supported() -> bool {
        true
    }

    fn frozen_sparse_patchify_batch_supported() -> bool {
        true
    }

    fn student_frozen_sparse_patchify_rollout(
        model: &VJepaTttModel<Self>,
        video: Tensor<Self, 5>,
        mask: &SparseTokenMask,
        target_tokens: Option<Tensor<Self, 3>>,
        state: &mut crate::TttState<Self>,
    ) -> Result<crate::VJepaEncoderOutput<Self>> {
        model.forward_single_frame_rollout_sparse_patchify_cuda_fusion_frozen(
            video,
            mask,
            target_tokens,
            state,
        )
    }

    fn student_frozen_sparse_patchify_rollout_batch(
        model: &VJepaTttModel<Self>,
        video: Tensor<Self, 5>,
        mask: &SparseMaskBatch<Self>,
        target_tokens: Option<Tensor<Self, 3>>,
        state: &mut crate::TttState<Self>,
    ) -> Result<crate::VJepaEncoderOutput<Self>> {
        model.forward_single_frame_rollout_sparse_patchify_cuda_fusion_frozen_batch(
            video,
            mask,
            target_tokens,
            state,
        )
    }
}

#[cfg(feature = "sparse-patchify-cuda")]
impl TttSparsePatchifyTrainingBackend
    for burn::backend::Autodiff<burn_flex_gmm::cuda::DefaultCudaBackend>
{
    fn frozen_sparse_patchify_supported() -> bool {
        true
    }

    fn student_frozen_sparse_patchify_rollout(
        model: &VJepaTttModel<Self>,
        video: Tensor<Self, 5>,
        mask: &SparseTokenMask,
        target_tokens: Option<Tensor<Self, 3>>,
        state: &mut crate::TttState<Self>,
    ) -> Result<crate::VJepaEncoderOutput<Self>> {
        model.forward_single_frame_rollout_sparse_patchify_cuda_frozen(
            video,
            mask,
            target_tokens,
            state,
        )
    }
}
