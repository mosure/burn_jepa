use super::loss;
use super::step::{self, TttRolloutKind};
use crate::training::batch::load_training_batch_with_policy;
use crate::training::config::BurnJepaTrainConfig;
use crate::training::report::{
    TttDomainEvalMetric, TttStageMetrics, TttTemporalDiagnosticMetrics, TttUtilizationMetrics,
    tensor_scalar,
};
use crate::{TttMemoryUpdateSource, TttStateResetMode, TttSupervisionMode};
use crate::{VJepa2_1Model, VJepaTttModel, dataset_from_config};
use anyhow::Result;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use std::collections::BTreeMap;
use std::time::Instant;

pub(super) struct TttEvalSummary {
    pub samples: usize,
    pub loss: f64,
    pub cosine: f64,
    pub teacher_forced_loss: Option<f64>,
    pub teacher_forced_cosine: Option<f64>,
    pub teacher_forcing_loss_gap: Option<f64>,
    pub teacher_forcing_cosine_gap: Option<f64>,
    pub full_loss: Option<f64>,
    pub full_cosine: Option<f64>,
    pub stage: TttStageMetrics,
    pub domains: Vec<TttDomainEvalMetric>,
    pub utilization: Option<TttUtilizationMetrics>,
    pub temporal_diagnostics: Option<TttTemporalDiagnosticMetrics>,
}

#[derive(Default)]
struct DomainTotals {
    samples: usize,
    loss: f64,
    cosine: f64,
    teacher_forced_samples: usize,
    teacher_forced_loss: f64,
    teacher_forced_cosine: f64,
    full_samples: usize,
    full_loss: f64,
    full_cosine: f64,
}

pub(super) fn evaluate_ttt_dataset<B: step::TttSparsePatchifyTrainingBackend>(
    model: &VJepaTttModel<B>,
    teacher: &VJepa2_1Model<B>,
    config: &BurnJepaTrainConfig,
    device: &B::Device,
    steps: usize,
) -> Result<TttEvalSummary> {
    let dataset = dataset_from_config(&config.dataset, false)?;
    let rollout = step::rollout_kind(config);
    let patchify = step::patchify_kind::<B>(config, rollout)?;
    let eval_batch_size = config.training.effective_eval_batch_size();
    let eval_full_grid = config.training.eval_full_grid || rollout == TttRolloutKind::Dense;
    let mut total = 0.0;
    let mut total_cosine = 0.0;
    let mut total_teacher_forced_loss = 0.0;
    let mut total_teacher_forced_cosine = 0.0;
    let mut total_full_loss = 0.0;
    let mut total_full_cosine = 0.0;
    let mut samples = 0usize;
    let mut teacher_forced_samples = 0usize;
    let mut full_samples = 0usize;
    let mut stage = TttStageMetrics::default();
    let mut domains = BTreeMap::<String, DomainTotals>::new();
    let mut teacher_cache = BTreeMap::<String, step::TeacherTokenTargets<B>>::new();
    let mut utilization = None;
    let mut temporal_diagnostics = None;
    let teacher_forced_eval =
        config.ttt.memory_update == TttMemoryUpdateSource::TeacherForcedDiagnostic;
    let capture_layers = config.ttt.capture_layers(model.config());
    for step_index in 0..steps {
        let start_index = step_index * eval_batch_size;
        let batch_size = if start_index < dataset.len() {
            (dataset.len() - start_index).min(eval_batch_size).max(1)
        } else {
            eval_batch_size
        };
        let batch = load_training_batch_with_policy::<B>(
            dataset.as_ref(),
            &config.dataset,
            model.config(),
            device,
            start_index,
            batch_size,
            config.training.batching,
        )?;
        let teacher_tokens = super::teacher_tokens_for_batch(
            teacher,
            batch.teacher.clone(),
            &batch.metadata,
            start_index,
            &capture_layers,
            config.training.cache_teacher_tokens,
            &mut teacher_cache,
            &mut stage,
        )?;
        let batch_size = batch.student.shape().dims::<5>()[0];
        let masks = super::timed(&mut stage.mask_ms, || {
            step::resolve_masks(config, &batch.student, model.config(), &batch.metadata)
        })?;
        let student = super::timed(&mut stage.student_forward_ms, || {
            step::student_eval_rollout(
                model,
                batch.student.clone(),
                teacher_tokens.final_tokens.clone(),
                masks.as_ref(),
                rollout,
                patchify,
                eval_full_grid,
                teacher_forced_eval,
            )
        })?;
        if step_index == 0 && config.training.eval_utilization_diagnostics {
            let mut probes = Vec::new();
            let _ = step::student_probe_rollout(
                model,
                batch.student.clone(),
                None,
                masks.as_ref(),
                rollout,
                true,
                TttStateResetMode::Persistent,
                &mut probes,
            )?;
            utilization = Some(super::metrics::ttt_utilization_metrics(
                config, model, probes, batch_size,
            )?);
        }
        if step_index == 0 && config.training.eval_temporal_diagnostics {
            temporal_diagnostics = Some(eval_temporal_diagnostics(
                model,
                config,
                batch.student.clone(),
                teacher_tokens.clone(),
                masks.as_ref(),
                rollout,
                batch_size,
                device,
            )?);
        }
        let loss_start = Instant::now();
        let primary_loss = loss::training_loss(
            model,
            config,
            &student.free_run,
            &teacher_tokens,
            masks.as_ref(),
            rollout,
            batch_size,
            device,
            TttSupervisionMode::FinalTeacher,
        )?;
        let primary_cosine = loss::primary_cosine(
            student.free_run.tokens.clone(),
            teacher_tokens.final_tokens.clone(),
            masks.as_ref().map(|masks| match rollout {
                TttRolloutKind::Dense | TttRolloutKind::SparseTarget => &masks.target,
                TttRolloutKind::SparseContext => &masks.context,
            }),
            rollout,
            batch_size,
            device,
        )?;
        let primary_loss = tensor_scalar(primary_loss.detach())?;
        total += primary_loss * batch_size as f64;
        total_cosine += primary_cosine * batch_size as f64;
        samples += batch_size;
        let mut teacher_forced_loss = None;
        let mut teacher_forced_cosine = None;
        if let Some(teacher_forced) = &student.teacher_forced {
            let batch_teacher_forced_loss = tensor_scalar(
                loss::training_loss(
                    model,
                    config,
                    teacher_forced,
                    &teacher_tokens,
                    masks.as_ref(),
                    rollout,
                    batch_size,
                    device,
                    TttSupervisionMode::FinalTeacher,
                )?
                .detach(),
            )?;
            let batch_teacher_forced_cosine = loss::primary_cosine(
                teacher_forced.tokens.clone(),
                teacher_tokens.final_tokens.clone(),
                masks.as_ref().map(|masks| match rollout {
                    TttRolloutKind::Dense | TttRolloutKind::SparseTarget => &masks.target,
                    TttRolloutKind::SparseContext => &masks.context,
                }),
                rollout,
                batch_size,
                device,
            )?;
            total_teacher_forced_loss += batch_teacher_forced_loss * batch_size as f64;
            total_teacher_forced_cosine += batch_teacher_forced_cosine * batch_size as f64;
            teacher_forced_samples += batch_size;
            teacher_forced_loss = Some(batch_teacher_forced_loss);
            teacher_forced_cosine = Some(batch_teacher_forced_cosine);
        }
        let mut full_loss = None;
        let mut full_cosine = None;
        if let Some(full_student_tokens) = student.full_tokens() {
            let full_feature_loss = loss::primary_feature_loss(
                config,
                full_student_tokens.clone(),
                teacher_tokens.final_tokens.clone(),
                None,
                TttRolloutKind::Dense,
                batch_size,
                device,
            );
            let batch_full_loss = tensor_scalar(full_feature_loss.detach())?;
            let batch_full_cosine =
                loss::tensor_cosine(full_student_tokens, teacher_tokens.final_tokens.clone())?;
            total_full_loss += batch_full_loss * batch_size as f64;
            total_full_cosine += batch_full_cosine * batch_size as f64;
            full_samples += batch_size;
            full_loss = Some(batch_full_loss);
            full_cosine = Some(batch_full_cosine);
        }
        let domain = batch_domain(&batch.metadata);
        let totals = domains.entry(domain).or_default();
        totals.samples += batch_size;
        totals.loss += primary_loss * batch_size as f64;
        totals.cosine += primary_cosine * batch_size as f64;
        if let (Some(loss), Some(cosine)) = (teacher_forced_loss, teacher_forced_cosine) {
            totals.teacher_forced_samples += batch_size;
            totals.teacher_forced_loss += loss * batch_size as f64;
            totals.teacher_forced_cosine += cosine * batch_size as f64;
        }
        if let (Some(full_loss), Some(full_cosine)) = (full_loss, full_cosine) {
            totals.full_samples += batch_size;
            totals.full_loss += full_loss * batch_size as f64;
            totals.full_cosine += full_cosine * batch_size as f64;
        }
        stage.loss_ms += loss_start.elapsed().as_millis();
    }
    let domains = domains
        .into_iter()
        .map(|(domain, totals)| TttDomainEvalMetric {
            domain,
            samples: totals.samples,
            loss: totals.loss / totals.samples.max(1) as f64,
            cosine: totals.cosine / totals.samples.max(1) as f64,
            teacher_forced_loss: (totals.teacher_forced_samples > 0)
                .then_some(totals.teacher_forced_loss / totals.teacher_forced_samples as f64),
            teacher_forced_cosine: (totals.teacher_forced_samples > 0)
                .then_some(totals.teacher_forced_cosine / totals.teacher_forced_samples as f64),
            teacher_forcing_loss_gap: (totals.teacher_forced_samples > 0).then_some(
                totals.loss / totals.samples.max(1) as f64
                    - totals.teacher_forced_loss / totals.teacher_forced_samples as f64,
            ),
            teacher_forcing_cosine_gap: (totals.teacher_forced_samples > 0).then_some(
                totals.cosine / totals.samples.max(1) as f64
                    - totals.teacher_forced_cosine / totals.teacher_forced_samples as f64,
            ),
            full_loss: (totals.full_samples > 0)
                .then_some(totals.full_loss / totals.full_samples as f64),
            full_cosine: (totals.full_samples > 0)
                .then_some(totals.full_cosine / totals.full_samples as f64),
        })
        .collect();
    Ok(TttEvalSummary {
        samples,
        loss: total / samples.max(1) as f64,
        cosine: total_cosine / samples.max(1) as f64,
        teacher_forced_loss: (teacher_forced_samples > 0)
            .then_some(total_teacher_forced_loss / teacher_forced_samples as f64),
        teacher_forced_cosine: (teacher_forced_samples > 0)
            .then_some(total_teacher_forced_cosine / teacher_forced_samples as f64),
        teacher_forcing_loss_gap: (teacher_forced_samples > 0).then_some(
            total / samples.max(1) as f64
                - total_teacher_forced_loss / teacher_forced_samples as f64,
        ),
        teacher_forcing_cosine_gap: (teacher_forced_samples > 0).then_some(
            total_cosine / samples.max(1) as f64
                - total_teacher_forced_cosine / teacher_forced_samples as f64,
        ),
        full_loss: (full_samples > 0).then_some(total_full_loss / full_samples as f64),
        full_cosine: (full_samples > 0).then_some(total_full_cosine / full_samples as f64),
        stage,
        domains,
        utilization,
        temporal_diagnostics,
    })
}

#[allow(clippy::too_many_arguments)]
fn eval_temporal_diagnostics<B: step::TttSparsePatchifyTrainingBackend>(
    model: &VJepaTttModel<B>,
    config: &BurnJepaTrainConfig,
    video: burn::tensor::Tensor<B, 5>,
    teacher_tokens: step::TeacherTokenTargets<B>,
    masks: Option<&step::ResolvedTttMasks<B>>,
    rollout: TttRolloutKind,
    batch_size: usize,
    device: &B::Device,
) -> Result<TttTemporalDiagnosticMetrics> {
    let reset_each_frame = diagnostic_loss_cosine(
        model,
        config,
        step::student_eval_ablation_rollout(
            model,
            video.clone(),
            masks,
            rollout,
            true,
            TttStateResetMode::EachFrame,
        )?,
        teacher_tokens.clone(),
        masks,
        rollout,
        batch_size,
        device,
    )?;
    let reset_each_tubelet = diagnostic_loss_cosine(
        model,
        config,
        step::student_eval_ablation_rollout(
            model,
            video.clone(),
            masks,
            rollout,
            true,
            TttStateResetMode::EachTubelet,
        )?,
        teacher_tokens.clone(),
        masks,
        rollout,
        batch_size,
        device,
    )?;
    let reverse_order = diagnostic_loss_cosine(
        model,
        config,
        step::student_eval_ablation_rollout(
            model,
            reverse_video_frames(video.clone()),
            masks,
            rollout,
            true,
            TttStateResetMode::Persistent,
        )?,
        teacher_tokens.clone(),
        masks,
        rollout,
        batch_size,
        device,
    )?;
    let shuffle_order = diagnostic_loss_cosine(
        model,
        config,
        step::student_eval_ablation_rollout(
            model,
            deterministic_shuffle_video_frames(video.clone()),
            masks,
            rollout,
            true,
            TttStateResetMode::Persistent,
        )?,
        teacher_tokens.clone(),
        masks,
        rollout,
        batch_size,
        device,
    )?;
    let freeze_fast_update = diagnostic_loss_cosine(
        model,
        config,
        step::student_eval_ablation_rollout(
            model,
            video,
            masks,
            rollout,
            false,
            TttStateResetMode::Persistent,
        )?,
        teacher_tokens.clone(),
        masks,
        rollout,
        batch_size,
        device,
    )?;
    Ok(TttTemporalDiagnosticMetrics {
        reset_each_frame_loss: Some(reset_each_frame.0),
        reset_each_frame_cosine: Some(reset_each_frame.1),
        reset_each_tubelet_loss: Some(reset_each_tubelet.0),
        reset_each_tubelet_cosine: Some(reset_each_tubelet.1),
        reverse_order_loss: Some(reverse_order.0),
        reverse_order_cosine: Some(reverse_order.1),
        shuffle_order_loss: Some(shuffle_order.0),
        shuffle_order_cosine: Some(shuffle_order.1),
        freeze_fast_update_loss: Some(freeze_fast_update.0),
        freeze_fast_update_cosine: Some(freeze_fast_update.1),
    })
}

fn reverse_video_frames<B: Backend>(video: Tensor<B, 5>) -> Tensor<B, 5> {
    let frames = video.shape().dims::<5>()[2];
    reorder_video_frames(video, (0..frames).rev().collect())
}

fn deterministic_shuffle_video_frames<B: Backend>(video: Tensor<B, 5>) -> Tensor<B, 5> {
    let frames = video.shape().dims::<5>()[2];
    let mut order = (0..frames).collect::<Vec<_>>();
    for chunk in order.chunks_mut(2) {
        if chunk.len() == 2 {
            chunk.swap(0, 1);
        }
    }
    reorder_video_frames(video, order)
}

fn reorder_video_frames<B: Backend>(video: Tensor<B, 5>, order: Vec<usize>) -> Tensor<B, 5> {
    let slices = order
        .into_iter()
        .map(|frame| video.clone().slice_dim(2, frame..frame + 1))
        .collect::<Vec<_>>();
    Tensor::cat(slices, 2)
}

fn diagnostic_loss_cosine<B: step::TttSparsePatchifyTrainingBackend>(
    model: &VJepaTttModel<B>,
    config: &BurnJepaTrainConfig,
    student: crate::VJepaEncoderOutput<B>,
    teacher_tokens: step::TeacherTokenTargets<B>,
    masks: Option<&step::ResolvedTttMasks<B>>,
    rollout: TttRolloutKind,
    batch_size: usize,
    device: &B::Device,
) -> Result<(f64, f64)> {
    let loss = tensor_scalar(
        loss::training_loss(
            model,
            config,
            &student,
            &teacher_tokens,
            masks,
            rollout,
            batch_size,
            device,
            TttSupervisionMode::FinalTeacher,
        )?
        .detach(),
    )?;
    let cosine = loss::primary_cosine(
        student.tokens,
        teacher_tokens.final_tokens,
        masks.map(|masks| match rollout {
            TttRolloutKind::Dense | TttRolloutKind::SparseTarget => &masks.target,
            TttRolloutKind::SparseContext => &masks.context,
        }),
        rollout,
        batch_size,
        device,
    )?;
    Ok((loss, cosine))
}

fn batch_domain(metadata: &[crate::JepaSampleMetadata]) -> String {
    let first = metadata
        .first()
        .and_then(|row| row.domain.as_deref())
        .unwrap_or("unknown");
    if metadata
        .iter()
        .all(|row| row.domain.as_deref().unwrap_or("unknown") == first)
    {
        first.to_string()
    } else {
        "mixed".to_string()
    }
}
