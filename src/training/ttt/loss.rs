use crate::{
    SparseMaskBatch, VJepa2_1Model, VJepaTttModel, apply_mask_batch, apply_token_mask,
    training::config::BurnJepaTrainConfig,
};
use anyhow::{Context, Result, ensure};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use super::step::{ResolvedTttMasks, TeacherTokenTargets, TttRolloutKind};

#[derive(Debug)]
pub struct TttDistillationLoss<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub student_tokens: Tensor<B, 3>,
    pub teacher_tokens: Tensor<B, 3>,
}

pub(super) struct TttLossBreakdown<B: Backend> {
    pub total: Tensor<B, 1>,
    pub feature: Tensor<B, 1>,
    pub predictor: Option<Tensor<B, 1>>,
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
    teacher: &TeacherTokenTargets<B>,
    masks: Option<&ResolvedTttMasks<B>>,
    rollout: TttRolloutKind,
    batch_size: usize,
    device: &B::Device,
    supervision: crate::TttSupervisionMode,
    predictor_target_tokens: Option<Tensor<B, 3>>,
) -> Result<Tensor<B, 1>> {
    Ok(training_loss_breakdown(
        model,
        config,
        student,
        teacher,
        masks,
        rollout,
        batch_size,
        device,
        supervision,
        predictor_target_tokens,
    )?
    .total)
}

pub(super) fn training_loss_breakdown<B: Backend>(
    model: &VJepaTttModel<B>,
    config: &BurnJepaTrainConfig,
    student: &crate::VJepaEncoderOutput<B>,
    teacher: &TeacherTokenTargets<B>,
    masks: Option<&ResolvedTttMasks<B>>,
    rollout: TttRolloutKind,
    batch_size: usize,
    device: &B::Device,
    supervision: crate::TttSupervisionMode,
    predictor_target_tokens: Option<Tensor<B, 3>>,
) -> Result<TttLossBreakdown<B>> {
    let primary_mask = masks.map(|masks| match rollout {
        TttRolloutKind::Dense | TttRolloutKind::SparseTarget => &masks.target,
        TttRolloutKind::SparseContext => &masks.context,
    });
    let feature_loss = match supervision {
        crate::TttSupervisionMode::FinalTeacher | crate::TttSupervisionMode::Hybrid => {
            primary_feature_loss(
                config,
                student.tokens.clone(),
                teacher.final_tokens.clone(),
                primary_mask,
                rollout,
                batch_size,
                device,
            )
        }
        crate::TttSupervisionMode::LayerLocalTeacher => layer_local_feature_loss(
            config,
            model.config(),
            student,
            teacher,
            primary_mask,
            rollout,
            batch_size,
            device,
        )?,
    };
    let predictor = if config.loss.predictor_loss_weight > 0.0
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
        let context_tokens = match rollout {
            TttRolloutKind::SparseContext => student.tokens.clone(),
            TttRolloutKind::Dense | TttRolloutKind::SparseTarget => {
                apply_mask_batch(student.tokens.clone(), &masks.context)
            }
        };
        let target_tokens = predictor_target_tokens
            .unwrap_or_else(|| apply_mask_batch(teacher.final_tokens.clone(), &masks.target));
        let predictions = model.forward_predictor_sparse(
            context_tokens,
            context_mask,
            target_mask,
            student.grid,
            0,
        )?;
        ensure!(
            predictions.target_predictions.shape().dims::<3>() == target_tokens.shape().dims::<3>(),
            "predictor loss target shape {:?} does not match predictor output {:?}",
            target_tokens.shape().dims::<3>(),
            predictions.target_predictions.shape().dims::<3>()
        );
        Some(
            (predictions.target_predictions - target_tokens)
                .powf_scalar(2.0)
                .mean()
                .mul_scalar(config.loss.predictor_loss_weight as f64),
        )
    } else {
        None
    };
    let total = match predictor.clone() {
        Some(predictor) => feature_loss.clone() + predictor,
        None => feature_loss.clone(),
    };
    Ok(TttLossBreakdown {
        total,
        feature: feature_loss,
        predictor,
    })
}

fn layer_local_feature_loss<B: Backend>(
    config: &BurnJepaTrainConfig,
    model_config: &crate::VJepaConfig,
    student: &crate::VJepaEncoderOutput<B>,
    teacher: &TeacherTokenTargets<B>,
    target_mask: Option<&SparseMaskBatch<B>>,
    rollout: TttRolloutKind,
    batch_size: usize,
    device: &B::Device,
) -> Result<Tensor<B, 1>> {
    let layers = config.ttt.resolved_layers(model_config);
    ensure!(
        !layers.is_empty(),
        "layer-local TTT supervision requires at least one TTT layer"
    );
    let mut total = None;
    let mut matched = 0usize;
    for layer in layers {
        let Some(student_tokens) = captured_layer_tokens(student, layer) else {
            continue;
        };
        let teacher_tokens = teacher
            .layer_tokens
            .iter()
            .find_map(|(teacher_layer, tokens)| (*teacher_layer == layer).then(|| tokens.clone()))
            .with_context(|| format!("teacher did not capture TTT layer {layer}"))?;
        let loss = primary_feature_loss(
            config,
            student_tokens,
            teacher_tokens,
            target_mask,
            rollout,
            batch_size,
            device,
        );
        total = Some(match total {
            Some(total) => total + loss,
            None => loss,
        });
        matched += 1;
    }
    ensure!(
        matched > 0,
        "student did not capture any TTT layer-local features"
    );
    Ok(total
        .expect("matched layer-local loss exists")
        .mul_scalar(1.0 / matched as f64))
}

fn captured_layer_tokens<B: Backend>(
    output: &crate::VJepaEncoderOutput<B>,
    layer: usize,
) -> Option<Tensor<B, 3>> {
    output
        .captured_layers
        .iter()
        .position(|&captured| captured == layer)
        .and_then(|index| output.hierarchical.get(index).cloned())
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
    let squared = (student_tokens - teacher_tokens).powf_scalar(2.0);
    let loss = if let Some(mask) = target_mask.and_then(|mask| mask.valid_token_mask(device)) {
        let [batch, tokens, dim] = squared.shape().dims::<3>();
        let valid_tokens = target_mask
            .map(SparseMaskBatch::valid_token_count)
            .unwrap_or(batch * tokens)
            .max(1);
        let mask = mask.unsqueeze_dim::<3>(2).repeat_dim(2, dim);
        (squared * mask)
            .mean()
            .mul_scalar((batch * tokens) as f64 / valid_tokens as f64)
    } else {
        squared.mean()
    };
    loss.mul_scalar(config.loss.feature_loss_weight as f64)
}

pub(super) fn primary_cosine<B: Backend>(
    student_tokens: Tensor<B, 3>,
    teacher_tokens: Tensor<B, 3>,
    target_mask: Option<&SparseMaskBatch<B>>,
    rollout: TttRolloutKind,
    batch_size: usize,
    device: &B::Device,
) -> Result<f64> {
    let valid_mask = target_mask.and_then(|mask| mask.valid_token_mask(device));
    let (student_tokens, teacher_tokens) = align_primary_tokens(
        student_tokens,
        teacher_tokens,
        target_mask,
        rollout,
        batch_size,
        device,
    );
    if let Some(valid_mask) = valid_mask {
        tensor_cosine_with_token_mask(student_tokens, teacher_tokens, valid_mask)
    } else {
        tensor_cosine(student_tokens, teacher_tokens)
    }
}

pub(super) fn align_primary_tokens<B: Backend>(
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

fn tensor_cosine_with_token_mask<B: Backend>(
    left: Tensor<B, 3>,
    right: Tensor<B, 3>,
    mask: Tensor<B, 2>,
) -> Result<f64> {
    let [_batch, _tokens, dim] = left.shape().dims::<3>();
    let left = left
        .into_data()
        .to_vec::<f32>()
        .context("read left eval tensor")?;
    let right = right
        .into_data()
        .to_vec::<f32>()
        .context("read right eval tensor")?;
    let mask = mask
        .into_data()
        .to_vec::<f32>()
        .context("read ragged eval mask tensor")?;
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    for (token_index, &valid) in mask.iter().enumerate() {
        if valid <= 0.0 {
            continue;
        }
        let start = token_index * dim;
        let end = start + dim;
        if end > left.len() || end > right.len() {
            break;
        }
        for (left, right) in left[start..end].iter().zip(right[start..end].iter()) {
            let left = *left as f64;
            let right = *right as f64;
            dot += left * right;
            left_norm += left * left;
            right_norm += right * right;
        }
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        Ok(0.0)
    } else {
        Ok(dot / (left_norm.sqrt() * right_norm.sqrt()))
    }
}
