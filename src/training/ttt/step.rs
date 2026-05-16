use crate::{
    JepaSampleMetadata, SparseMaskBatch, SparseTokenMask, VJepa2_1Model, VJepaTttModel,
    apply_mask_batch,
};
use anyhow::{Result, bail, ensure};
use burn::tensor::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};

use crate::training::config::{BurnJepaTrainConfig, TttSparsePatchifyTrainingMode};
use crate::training::mask::TrainingMaskConfig;
use crate::{TttStateResetMode, VJepaTttLayerProbeRecord};

#[derive(Debug)]
pub(crate) struct TeacherTokenTargets<B: Backend> {
    pub final_tokens: Tensor<B, 3>,
    pub layer_tokens: Vec<(usize, Tensor<B, 3>)>,
}

impl<B: Backend> Clone for TeacherTokenTargets<B> {
    fn clone(&self) -> Self {
        Self {
            final_tokens: self.final_tokens.clone(),
            layer_tokens: self
                .layer_tokens
                .iter()
                .map(|(layer, tokens)| (*layer, tokens.clone()))
                .collect(),
        }
    }
}

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
    pub free_run: crate::VJepaEncoderOutput<B>,
    pub teacher_forced: Option<crate::VJepaEncoderOutput<B>>,
    pub full_free_run: Option<crate::VJepaEncoderOutput<B>>,
}

impl<B: Backend> StudentEvalRollout<B> {
    pub fn full_tokens(&self) -> Option<Tensor<B, 3>> {
        self.full_free_run
            .as_ref()
            .map(|output| output.tokens.clone())
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

pub(super) fn teacher_targets<B: Backend>(
    teacher: &VJepa2_1Model<B>,
    video: Tensor<B, 5>,
    capture_layers: &[usize],
) -> TeacherTokenTargets<B> {
    let output = if capture_layers.is_empty() {
        teacher.encode_video(video, None)
    } else {
        teacher
            .encoder
            .forward_video_capture_layers(video, None, capture_layers)
    };
    let layer_tokens = output
        .captured_layers
        .into_iter()
        .zip(output.hierarchical)
        .map(|(layer, tokens)| (layer, tokens.detach()))
        .collect();
    TeacherTokenTargets {
        final_tokens: output.tokens.detach(),
        layer_tokens,
    }
}

pub(super) fn teacher_predictor_targets<B: Backend>(
    teacher: &VJepa2_1Model<B>,
    teacher_tokens: &TeacherTokenTargets<B>,
    masks: Option<&ResolvedTttMasks<B>>,
    grid: crate::TokenGridShape,
    predictor_loss_weight: f32,
) -> Result<Option<Tensor<B, 3>>> {
    if predictor_loss_weight <= 0.0 {
        return Ok(None);
    }
    let Some(masks) = masks else {
        return Ok(None);
    };
    let context_mask = masks.context.uniform_mask().ok_or_else(|| {
        anyhow::anyhow!("predictor loss currently requires a uniform context mask")
    })?;
    let target_mask = masks.target.uniform_mask().ok_or_else(|| {
        anyhow::anyhow!("predictor loss currently requires a uniform target mask")
    })?;
    let direct_target = apply_mask_batch(teacher_tokens.final_tokens.clone(), &masks.target);
    let predictor_output_dim = teacher
        .config()
        .predictor
        .output_dim
        .unwrap_or(teacher.config().encoder.embed_dim);
    if predictor_output_dim == teacher.config().encoder.embed_dim {
        return Ok(Some(direct_target));
    }
    let context_tokens = apply_mask_batch(teacher_tokens.final_tokens.clone(), &masks.context);
    let prediction = teacher
        .predictor
        .forward_sparse(context_tokens, context_mask, target_mask, grid, 0)?
        .target_predictions
        .detach();
    Ok(Some(prediction))
}

pub(super) fn student_training_rollout<B: TttSparsePatchifyTrainingBackend>(
    model: &VJepaTttModel<B>,
    video: Tensor<B, 5>,
    teacher_tokens: Tensor<B, 3>,
    masks: Option<&ResolvedTttMasks<B>>,
    rollout: TttRolloutKind,
    patchify: TttPatchifyKind,
) -> Result<crate::VJepaEncoderOutput<B>> {
    student_rollout_with_patchify(model, video, Some(teacher_tokens), masks, rollout, patchify)
}

pub(super) fn student_training_rollout_with_state<B: TttSparsePatchifyTrainingBackend>(
    model: &VJepaTttModel<B>,
    video: Tensor<B, 5>,
    teacher_tokens: Tensor<B, 3>,
    masks: Option<&ResolvedTttMasks<B>>,
    rollout: TttRolloutKind,
    patchify: TttPatchifyKind,
    state: &mut crate::TttState<B>,
) -> Result<crate::VJepaEncoderOutput<B>> {
    student_rollout_with_patchify_and_state(
        model,
        video,
        Some(teacher_tokens),
        masks,
        rollout,
        patchify,
        state,
    )
}

pub(super) fn student_free_run_rollout_with_state<B: TttSparsePatchifyTrainingBackend>(
    model: &VJepaTttModel<B>,
    video: Tensor<B, 5>,
    masks: Option<&ResolvedTttMasks<B>>,
    rollout: TttRolloutKind,
    patchify: TttPatchifyKind,
    state: &mut crate::TttState<B>,
) -> Result<crate::VJepaEncoderOutput<B>> {
    student_rollout_with_patchify_and_state(model, video, None, masks, rollout, patchify, state)
}

fn student_rollout_with_patchify<B: TttSparsePatchifyTrainingBackend>(
    model: &VJepaTttModel<B>,
    video: Tensor<B, 5>,
    adapter_target_tokens: Option<Tensor<B, 3>>,
    masks: Option<&ResolvedTttMasks<B>>,
    rollout: TttRolloutKind,
    patchify: TttPatchifyKind,
) -> Result<crate::VJepaEncoderOutput<B>> {
    let mut state = model.fresh_state();
    student_rollout_with_patchify_and_state(
        model,
        video,
        adapter_target_tokens,
        masks,
        rollout,
        patchify,
        &mut state,
    )
}

fn student_rollout_with_patchify_and_state<B: TttSparsePatchifyTrainingBackend>(
    model: &VJepaTttModel<B>,
    video: Tensor<B, 5>,
    adapter_target_tokens: Option<Tensor<B, 3>>,
    masks: Option<&ResolvedTttMasks<B>>,
    rollout: TttRolloutKind,
    patchify: TttPatchifyKind,
    state: &mut crate::TttState<B>,
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
            return B::student_frozen_sparse_patchify_rollout(
                model,
                video,
                mask,
                adapter_target_tokens,
                state,
            );
        } else if B::frozen_sparse_patchify_batch_supported() {
            return B::student_frozen_sparse_patchify_rollout_batch(
                model,
                video,
                mask,
                adapter_target_tokens,
                state,
            );
        }
    }
    match rollout {
        TttRolloutKind::Dense => {
            model.forward_single_frame_rollout(video, adapter_target_tokens, state)
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
                adapter_target_tokens,
                state,
            )
        }
    }
}

pub(super) fn student_eval_rollout<B: TttSparsePatchifyTrainingBackend>(
    model: &VJepaTttModel<B>,
    video: Tensor<B, 5>,
    teacher_tokens: Tensor<B, 3>,
    masks: Option<&ResolvedTttMasks<B>>,
    rollout: TttRolloutKind,
    patchify: TttPatchifyKind,
    eval_full_grid: bool,
    teacher_forced_eval: bool,
) -> Result<StudentEvalRollout<B>> {
    let free_run =
        student_rollout_with_patchify(model, video.clone(), None, masks, rollout, patchify)?;
    let teacher_forced = teacher_forced_eval
        .then(|| {
            student_rollout_with_patchify(
                model,
                video.clone(),
                Some(teacher_tokens.clone()),
                masks,
                rollout,
                patchify,
            )
        })
        .transpose()?;
    let full_free_run = (eval_full_grid && rollout.sparse_mask_kind().is_some())
        .then(|| {
            let mut full_state = model.fresh_state();
            model.forward_single_frame_rollout(video, None, &mut full_state)
        })
        .transpose()?;
    Ok(StudentEvalRollout {
        free_run,
        teacher_forced,
        full_free_run,
    })
}

pub(super) fn student_eval_rollout_with_state<B: TttSparsePatchifyTrainingBackend>(
    model: &VJepaTttModel<B>,
    video: Tensor<B, 5>,
    teacher_tokens: Tensor<B, 3>,
    masks: Option<&ResolvedTttMasks<B>>,
    rollout: TttRolloutKind,
    patchify: TttPatchifyKind,
    eval_full_grid: bool,
    teacher_forced_eval: bool,
    state: &mut crate::TttState<B>,
) -> Result<StudentEvalRollout<B>> {
    let free_run =
        student_free_run_rollout_with_state(model, video.clone(), masks, rollout, patchify, state)?;
    let teacher_forced = teacher_forced_eval
        .then(|| {
            student_rollout_with_patchify(
                model,
                video.clone(),
                Some(teacher_tokens.clone()),
                masks,
                rollout,
                patchify,
            )
        })
        .transpose()?;
    let full_free_run = (eval_full_grid && rollout.sparse_mask_kind().is_some())
        .then(|| {
            let mut full_state = model.fresh_state();
            model.forward_single_frame_rollout(video, None, &mut full_state)
        })
        .transpose()?;
    Ok(StudentEvalRollout {
        free_run,
        teacher_forced,
        full_free_run,
    })
}

pub(super) fn student_probe_rollout<B: Backend>(
    model: &VJepaTttModel<B>,
    video: Tensor<B, 5>,
    adapter_target_tokens: Option<Tensor<B, 3>>,
    masks: Option<&ResolvedTttMasks<B>>,
    rollout: TttRolloutKind,
    update_fast_weight: bool,
    reset_mode: TttStateResetMode,
    probes: &mut Vec<VJepaTttLayerProbeRecord<B>>,
) -> Result<crate::VJepaEncoderOutput<B>> {
    let mut state = model.fresh_state();
    let mask = rollout
        .sparse_mask_kind()
        .map(|_| {
            masks
                .map(|masks| rollout.select_mask(masks))
                .ok_or_else(|| anyhow::anyhow!("sparse TTT probe rollout requires resolved masks"))
        })
        .transpose()?;
    if let Some(mask) = mask {
        ensure!(
            !mask.is_empty(),
            "sparse TTT probe rollout requires a non-empty target mask"
        );
    }
    model.encoder.forward_single_frame_rollout_with_diagnostics(
        video,
        mask,
        adapter_target_tokens,
        &mut state,
        update_fast_weight,
        reset_mode,
        Some(probes),
    )
}

pub(super) fn student_eval_ablation_rollout<B: Backend>(
    model: &VJepaTttModel<B>,
    video: Tensor<B, 5>,
    masks: Option<&ResolvedTttMasks<B>>,
    rollout: TttRolloutKind,
    update_fast_weight: bool,
    reset_mode: TttStateResetMode,
) -> Result<crate::VJepaEncoderOutput<B>> {
    let mut probes = Vec::new();
    student_probe_rollout(
        model,
        video,
        None,
        masks,
        rollout,
        update_fast_weight,
        reset_mode,
        &mut probes,
    )
}

impl TttRolloutKind {
    pub(super) fn sparse_mask_kind(self) -> Option<SparseMaskKind> {
        match self {
            Self::Dense => None,
            Self::SparseContext => Some(SparseMaskKind::Context),
            Self::SparseTarget => Some(SparseMaskKind::Target),
        }
    }

    pub(super) fn select_mask<B: Backend>(
        self,
        masks: &ResolvedTttMasks<B>,
    ) -> &SparseMaskBatch<B> {
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

#[cfg(feature = "flex")]
impl TttSparsePatchifyTrainingBackend for burn::backend::Autodiff<burn::backend::Flex<f32, i32>> {}

#[cfg(feature = "dispatch")]
impl TttSparsePatchifyTrainingBackend for burn::Dispatch {}

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
