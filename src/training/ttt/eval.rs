use super::loss;
use super::step::{self, TttRolloutKind};
use crate::training::batch::load_training_batch_with_policy;
use crate::training::config::BurnJepaTrainConfig;
use crate::training::report::{TttDomainEvalMetric, TttStageMetrics, tensor_scalar};
use crate::{VJepa2_1Model, VJepaTttModel, dataset_from_config};
use anyhow::Result;
use std::collections::BTreeMap;
use std::time::Instant;

pub(super) struct TttEvalSummary {
    pub samples: usize,
    pub loss: f64,
    pub cosine: f64,
    pub full_loss: Option<f64>,
    pub full_cosine: Option<f64>,
    pub stage: TttStageMetrics,
    pub domains: Vec<TttDomainEvalMetric>,
}

#[derive(Default)]
struct DomainTotals {
    samples: usize,
    loss: f64,
    cosine: f64,
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
    let mut total_full_loss = 0.0;
    let mut total_full_cosine = 0.0;
    let mut samples = 0usize;
    let mut full_samples = 0usize;
    let mut stage = TttStageMetrics::default();
    let mut domains = BTreeMap::<String, DomainTotals>::new();
    let mut teacher_cache = BTreeMap::<String, burn::tensor::Tensor<B, 3>>::new();
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
                batch.student,
                teacher_tokens.clone(),
                masks.as_ref(),
                rollout,
                patchify,
                eval_full_grid,
            )
        })?;
        let loss_start = Instant::now();
        let primary_loss = loss::training_loss(
            model,
            config,
            &student.primary,
            teacher_tokens.clone(),
            masks.as_ref(),
            rollout,
            batch_size,
            device,
        )?;
        let primary_cosine = loss::primary_cosine(
            student.primary.tokens.clone(),
            teacher_tokens.clone(),
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
        let mut full_loss = None;
        let mut full_cosine = None;
        if let Some(full_student_tokens) = student.full_tokens() {
            let full_feature_loss = loss::primary_feature_loss(
                config,
                full_student_tokens.clone(),
                teacher_tokens.clone(),
                None,
                TttRolloutKind::Dense,
                batch_size,
                device,
            );
            let batch_full_loss = tensor_scalar(full_feature_loss.detach())?;
            let batch_full_cosine = loss::tensor_cosine(full_student_tokens, teacher_tokens)?;
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
        full_loss: (full_samples > 0).then_some(total_full_loss / full_samples as f64),
        full_cosine: (full_samples > 0).then_some(total_full_cosine / full_samples as f64),
        stage,
        domains,
    })
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
