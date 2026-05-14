use crate::{
    SparseMaskBatch, VJepa2_1Model, VJepaTttModel, apply_mask_batch, apply_token_mask,
    training::config::BurnJepaTrainConfig,
};
use anyhow::{Context, Result};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use super::step::{ResolvedTttMasks, TttRolloutKind};

#[derive(Debug)]
pub struct TttDistillationLoss<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub student_tokens: Tensor<B, 3>,
    pub teacher_tokens: Tensor<B, 3>,
}

pub fn evaluate_ttt_distillation<B: Backend>(
    student: &VJepaTttModel<B>,
    teacher: &VJepa2_1Model<B>,
    video: Tensor<B, 5>,
) -> Result<TttDistillationLoss<B>> {
    let teacher_tokens = teacher.encode_video(video.clone(), None).tokens.detach();
    let mut state = student.fresh_state();
    let student_tokens =
        student.forward_single_frame_rollout(video, Some(teacher_tokens.clone()), &mut state)?;
    let loss = (student_tokens.tokens.clone() - teacher_tokens.clone())
        .powf_scalar(2.0)
        .mean();
    Ok(TttDistillationLoss {
        loss,
        student_tokens: student_tokens.tokens,
        teacher_tokens,
    })
}

pub(super) fn training_loss<B: Backend>(
    model: &VJepaTttModel<B>,
    config: &BurnJepaTrainConfig,
    student: &crate::VJepaEncoderOutput<B>,
    teacher_tokens: Tensor<B, 3>,
    masks: Option<&ResolvedTttMasks<B>>,
    rollout: TttRolloutKind,
    batch_size: usize,
    device: &B::Device,
) -> Result<Tensor<B, 1>> {
    let primary_mask = masks.map(|masks| match rollout {
        TttRolloutKind::Dense | TttRolloutKind::SparseTarget => &masks.target,
        TttRolloutKind::SparseContext => &masks.context,
    });
    let feature_loss = primary_feature_loss(
        config,
        student.tokens.clone(),
        teacher_tokens.clone(),
        primary_mask,
        rollout,
        batch_size,
        device,
    );
    if config.loss.predictor_loss_weight > 0.0
        && let Some(masks) = masks
    {
        let context_mask = masks
            .context
            .uniform_mask()
            .context("predictor loss currently requires a uniform context mask")?;
        let target_mask = masks
            .target
            .uniform_mask()
            .context("predictor loss currently requires a uniform target mask")?;
        let context_tokens = apply_mask_batch(student.tokens.clone(), &masks.context);
        let target_tokens = apply_mask_batch(teacher_tokens, &masks.target);
        let predictions = model.predictor.forward_sparse(
            context_tokens,
            context_mask,
            target_mask,
            student.grid,
            0,
        )?;
        Ok(feature_loss
            + (predictions.target_predictions - target_tokens)
                .powf_scalar(2.0)
                .mean()
                .mul_scalar(config.loss.predictor_loss_weight as f64))
    } else {
        Ok(feature_loss)
    }
}

pub(super) fn primary_feature_loss<B: Backend>(
    config: &BurnJepaTrainConfig,
    student_tokens: Tensor<B, 3>,
    teacher_tokens: Tensor<B, 3>,
    target_mask: Option<&SparseMaskBatch<B>>,
    rollout: TttRolloutKind,
    batch_size: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    let (student_tokens, teacher_tokens) = align_primary_tokens(
        student_tokens,
        teacher_tokens,
        target_mask,
        rollout,
        batch_size,
        device,
    );
    (student_tokens - teacher_tokens)
        .powf_scalar(2.0)
        .mean()
        .mul_scalar(config.loss.feature_loss_weight as f64)
}

pub(super) fn primary_cosine<B: Backend>(
    student_tokens: Tensor<B, 3>,
    teacher_tokens: Tensor<B, 3>,
    target_mask: Option<&SparseMaskBatch<B>>,
    rollout: TttRolloutKind,
    batch_size: usize,
    device: &B::Device,
) -> Result<f64> {
    let (student_tokens, teacher_tokens) = align_primary_tokens(
        student_tokens,
        teacher_tokens,
        target_mask,
        rollout,
        batch_size,
        device,
    );
    tensor_cosine(student_tokens, teacher_tokens)
}

fn align_primary_tokens<B: Backend>(
    student_tokens: Tensor<B, 3>,
    teacher_tokens: Tensor<B, 3>,
    target_mask: Option<&SparseMaskBatch<B>>,
    rollout: TttRolloutKind,
    _batch_size: usize,
    _device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    match (rollout, target_mask) {
        (TttRolloutKind::SparseContext | TttRolloutKind::SparseTarget, Some(mask)) => {
            (student_tokens, apply_mask_batch(teacher_tokens, mask))
        }
        (_, Some(mask)) => {
            let indices = mask.indices();
            (
                apply_token_mask(student_tokens, indices.clone()),
                apply_token_mask(teacher_tokens, indices),
            )
        }
        (_, None) => (student_tokens, teacher_tokens),
    }
}

pub(super) fn tensor_cosine<B: Backend, const D: usize>(
    left: Tensor<B, D>,
    right: Tensor<B, D>,
) -> Result<f64> {
    let left = left
        .into_data()
        .to_vec::<f32>()
        .context("read left eval tensor")?;
    let right = right
        .into_data()
        .to_vec::<f32>()
        .context("read right eval tensor")?;
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    for (left, right) in left.iter().zip(right.iter()) {
        let left = *left as f64;
        let right = *right as f64;
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        Ok(0.0)
    } else {
        Ok(dot / (left_norm.sqrt() * right_norm.sqrt()))
    }
}
