mod eval;
mod loss;
mod metrics;
mod step;

use super::batch::load_training_batch_with_policy;
use super::config::BurnJepaTrainConfig;
use super::model_io::{load_student_model, load_teacher_model};
use super::report::{
    TrainingLossSummary, TttBackpropMetrics, TttEvalReport, TttStageMetrics, TttStepMetric,
    TttTrainingReport, samples_per_second, save_training_report, save_ttt_training_report,
    tensor_scalar,
};
use crate::{VJepaTttModel, dataset_from_config, video_token_grid};
use anyhow::{Context, Result};
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

pub use loss::{TttDistillationLoss, evaluate_ttt_distillation};
pub use step::TttSparsePatchifyTrainingBackend;

pub fn train_ttt_distillation<B: step::TttSparsePatchifyTrainingBackend>(
    config: &BurnJepaTrainConfig,
    device: &B::Device,
) -> Result<TttTrainingReport> {
    config.validate_for_ttt()?;
    let start = Instant::now();
    fs::create_dir_all(&config.model.output_dir)
        .with_context(|| format!("create {}", config.model.output_dir.display()))?;

    let teacher = load_teacher_model::<B>(config, device)?;
    let base = load_student_model::<B>(config, device)?;
    let mut model = VJepaTttModel::from_model(base, config.ttt.clone(), device)?;
    if let Some(path) = &config.model.ttt_checkpoint_path {
        model = model
            .load_file(
                path.clone(),
                &NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
                device,
            )
            .with_context(|| format!("load TTT checkpoint {}", path.display()))?;
    }
    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.training.weight_decay)
        .init::<B, VJepaTttModel<B>>();
    let dataset = dataset_from_config(&config.dataset, true)?;
    let memory = metrics::ttt_memory_metrics(config, model.config());
    let pre_train_eval = if config.training.eval_steps > 0 {
        Some(eval::evaluate_ttt_dataset(
            &model,
            &teacher,
            config,
            device,
            config.training.eval_steps,
        )?)
    } else {
        None
    };
    let mut progress = LossProgress::new(config.training.max_steps);
    let mut mask_metrics = None;
    let mut train_stage = TttStageMetrics::default();
    let mut teacher_cache = BTreeMap::<String, burn::tensor::Tensor<B, 3>>::new();
    let mut observed_dense_tokens = None;
    let rollout = step::rollout_kind(config);
    let patchify = step::patchify_kind::<B>(config, rollout)?;

    for step_index in 0..config.training.max_steps {
        let batch = load_training_batch_with_policy::<B>(
            dataset.as_ref(),
            &config.dataset,
            model.config(),
            device,
            step_index * config.training.batch_size,
            config.training.batch_size,
            config.training.batching,
        )?;
        observed_dense_tokens.get_or_insert_with(|| {
            let [_, _, frames, height, width] = batch.student.shape().dims::<5>();
            video_token_grid(model.config(), frames, height, width)
                .map(|grid| grid.len())
                .unwrap_or_else(|_| model.config().num_patches())
        });
        let batch_size = batch.student.shape().dims::<5>()[0];
        let masks = timed(&mut train_stage.mask_ms, || {
            step::resolve_masks(config, &batch.student, model.config(), &batch.metadata)
        })?;
        if mask_metrics.is_none()
            && let Some(masks) = &masks
        {
            let context = masks.context.first_mask()?;
            let target = masks.target.first_mask()?;
            mask_metrics = Some(metrics::mask_metrics_from_masks(&context, &target));
        }
        let teacher_tokens = teacher_tokens_for_batch(
            &teacher,
            batch.teacher.clone(),
            &batch.metadata,
            step_index,
            config.training.cache_teacher_tokens,
            &mut teacher_cache,
            &mut train_stage,
        )?;
        let student = timed(&mut train_stage.student_forward_ms, || {
            step::student_training_rollout(
                &model,
                batch.student,
                teacher_tokens.clone(),
                masks.as_ref(),
                rollout,
                patchify,
            )
        })?;
        let loss = timed(&mut train_stage.loss_ms, || {
            loss::training_loss(
                &model,
                config,
                &student,
                teacher_tokens,
                masks.as_ref(),
                rollout,
                batch_size,
                device,
            )
        })?;
        let step_number = step_index + 1;
        let save_partial =
            config.training.save_steps > 0 && step_number % config.training.save_steps == 0;
        let read_loss = progress.should_read_step(step_number, config) || save_partial;
        if read_loss {
            let final_loss = tensor_scalar(loss.clone().detach())?;
            progress.record(step_number, final_loss, config.training.loss_trace_interval);
        }

        let backward_start = Instant::now();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let backward_ms = backward_start.elapsed().as_millis();
        train_stage.backward_ms += backward_ms;
        train_stage.backward_optim_ms += backward_ms;

        let optim_start = Instant::now();
        model = optim.step(config.training.learning_rate, model, grads);
        let optimizer_ms = optim_start.elapsed().as_millis();
        train_stage.optimizer_ms += optimizer_ms;
        train_stage.backward_optim_ms += optimizer_ms;

        if save_partial {
            save_training_report(
                &config.model.output_dir,
                "ttt-report.partial.json",
                step_number,
                step_number * config.training.batch_size.max(1),
                TrainingLossSummary::ttt(
                    progress.initial_loss,
                    progress.best_loss,
                    progress.final_loss,
                ),
                start.elapsed().as_millis(),
                None,
            )?;
        }
    }

    let train_elapsed_ms = start.elapsed().as_millis();
    let eval_start = Instant::now();
    let (
        eval_loss,
        eval_cosine,
        eval_full_loss,
        eval_full_cosine,
        eval_samples,
        eval_stage,
        eval_domains,
    ) = if config.training.eval_steps > 0 {
        let eval = eval::evaluate_ttt_dataset(
            &model,
            &teacher,
            config,
            device,
            config.training.eval_steps,
        )?;
        (
            Some(eval.loss),
            Some(eval.cosine),
            eval.full_loss,
            eval.full_cosine,
            eval.samples,
            eval.stage,
            eval.domains,
        )
    } else {
        (
            None,
            None,
            None,
            None,
            0,
            TttStageMetrics::default(),
            Vec::new(),
        )
    };
    let eval_elapsed_ms = eval_start.elapsed().as_millis();
    let model_path = save_model_if_enabled(config, &model)?;
    let samples = config.training.max_steps * config.training.batch_size;
    let elapsed_ms = start.elapsed().as_millis();
    let rollout_metrics = metrics::rollout_metrics(
        model.config(),
        rollout,
        mask_metrics.as_ref(),
        config.training.eval_steps > 0 && config.training.eval_full_grid,
        observed_dense_tokens,
        patchify,
    );
    let mut report = TttTrainingReport {
        steps: config.training.max_steps,
        samples,
        initial_loss: progress.initial_loss.unwrap_or(progress.final_loss),
        best_loss: progress.best_loss,
        final_loss: progress.final_loss,
        loss_trace: progress.loss_trace,
        memory,
        mask: mask_metrics,
        rollout: rollout_metrics,
        backprop: TttBackpropMetrics {
            mode: config.ttt.backprop_mode,
            truncate_blocks: config.ttt.backprop_truncate_blocks,
        },
        pre_train_eval_loss: pre_train_eval.as_ref().map(|eval| eval.loss),
        pre_train_eval_cosine: pre_train_eval.as_ref().map(|eval| eval.cosine),
        pre_train_full_eval_loss: pre_train_eval.as_ref().and_then(|eval| eval.full_loss),
        pre_train_full_eval_cosine: pre_train_eval.as_ref().and_then(|eval| eval.full_cosine),
        eval_loss,
        eval_cosine,
        eval_full_loss,
        eval_full_cosine,
        eval_samples,
        train_stage,
        eval_stage,
        eval_domains,
        train_elapsed_ms,
        eval_elapsed_ms,
        elapsed_ms,
        samples_per_second: samples_per_second(samples, train_elapsed_ms),
        model_path,
        report_path: config.model.output_dir.join("ttt-report.json"),
    };
    report.report_path =
        save_ttt_training_report(&config.model.output_dir, "ttt-report.json", &report)?;
    drop(model);
    B::sync(device).context("sync TTT training backend")?;
    B::memory_cleanup(device);
    Ok(report)
}

pub fn evaluate_ttt_model_file<B: step::TttSparsePatchifyTrainingBackend>(
    config: &BurnJepaTrainConfig,
    model_path: impl AsRef<Path>,
    device: &B::Device,
    steps: usize,
) -> Result<TttEvalReport> {
    config.validate_for_ttt()?;
    let steps = steps.max(1);
    let start = Instant::now();
    let model_path = model_path.as_ref().to_path_buf();
    fs::create_dir_all(&config.model.output_dir)
        .with_context(|| format!("create {}", config.model.output_dir.display()))?;

    let teacher = load_teacher_model::<B>(config, device)?;
    let base = load_student_model::<B>(config, device)?;
    let model = VJepaTttModel::from_model(base, config.ttt.clone(), device)?
        .load_file(
            model_path.clone(),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
            device,
        )
        .with_context(|| format!("load TTT model {}", model_path.display()))?;
    let memory = metrics::ttt_memory_metrics_for_batch_size(
        config,
        model.config(),
        config.training.effective_eval_batch_size(),
    );
    let rollout = step::rollout_kind(config);
    let patchify = step::patchify_kind::<B>(config, rollout)?;

    let dataset = dataset_from_config(&config.dataset, false)?;
    let first_batch = load_training_batch_with_policy::<B>(
        dataset.as_ref(),
        &config.dataset,
        model.config(),
        device,
        0,
        config.training.effective_eval_batch_size(),
        config.training.batching,
    )?;
    let [_, _, frames, height, width] = first_batch.student.shape().dims::<5>();
    let dense_tokens = video_token_grid(model.config(), frames, height, width)
        .map(|grid| grid.len())
        .unwrap_or_else(|_| model.config().num_patches());
    let masks = step::resolve_masks(
        config,
        &first_batch.student,
        model.config(),
        &first_batch.metadata,
    )?;
    let mask_metrics = masks
        .as_ref()
        .map(|masks| {
            let context = masks.context.first_mask()?;
            let target = masks.target.first_mask()?;
            Ok::<_, anyhow::Error>(metrics::mask_metrics_from_masks(&context, &target))
        })
        .transpose()?;

    let eval = eval::evaluate_ttt_dataset(&model, &teacher, config, device, steps)?;
    let eval_samples = eval.samples;
    let elapsed_ms = start.elapsed().as_millis();
    let rollout = metrics::rollout_metrics(
        model.config(),
        rollout,
        mask_metrics.as_ref(),
        config.training.eval_full_grid,
        Some(dense_tokens),
        patchify,
    );
    let report_path = config.model.output_dir.join("ttt-eval-report.json");
    let report = TttEvalReport {
        model_path,
        eval_steps: steps,
        eval_samples,
        loss: eval.loss,
        cosine: eval.cosine,
        full_loss: eval.full_loss,
        full_cosine: eval.full_cosine,
        memory,
        mask: mask_metrics,
        rollout,
        stage: eval.stage,
        domains: eval.domains,
        elapsed_ms,
        samples_per_second: samples_per_second(eval_samples, elapsed_ms),
        report_path,
    };
    fs::write(&report.report_path, serde_json::to_string_pretty(&report)?)
        .with_context(|| format!("write {}", report.report_path.display()))?;
    B::sync(device).context("sync TTT eval backend")?;
    B::memory_cleanup(device);
    Ok(report)
}

pub(super) fn timed<T>(metric_ms: &mut u128, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let start = Instant::now();
    let output = f();
    *metric_ms += start.elapsed().as_millis();
    output
}

pub(super) fn teacher_tokens_for_batch<B: step::TttSparsePatchifyTrainingBackend>(
    teacher: &crate::VJepa2_1Model<B>,
    video: burn::tensor::Tensor<B, 5>,
    metadata: &[crate::JepaSampleMetadata],
    fallback_index: usize,
    enabled: bool,
    cache: &mut BTreeMap<String, burn::tensor::Tensor<B, 3>>,
    stage: &mut TttStageMetrics,
) -> Result<burn::tensor::Tensor<B, 3>> {
    if !enabled {
        return timed(&mut stage.teacher_forward_ms, || {
            Ok(step::teacher_tokens(teacher, video))
        });
    }
    let key = teacher_cache_key(metadata, fallback_index);
    if let Some(tokens) = cache.get(&key) {
        stage.teacher_cache_hits += 1;
        return Ok(tokens.clone());
    }
    stage.teacher_cache_misses += 1;
    let tokens = timed(&mut stage.teacher_forward_ms, || {
        Ok(step::teacher_tokens(teacher, video))
    })?;
    cache.insert(key, tokens.clone());
    Ok(tokens)
}

fn teacher_cache_key(metadata: &[crate::JepaSampleMetadata], fallback_index: usize) -> String {
    let has_identity = metadata.iter().any(|row| {
        row.clip_id.is_some()
            || row.source.is_some()
            || row.start_frame.is_some()
            || row.precomputed_context_indices.is_some()
            || row.precomputed_target_indices.is_some()
    });
    if has_identity {
        serde_json::to_string(metadata).unwrap_or_else(|_| format!("fallback:{fallback_index}"))
    } else {
        format!("fallback:{fallback_index}")
    }
}

fn save_model_if_enabled<B: step::TttSparsePatchifyTrainingBackend>(
    config: &BurnJepaTrainConfig,
    model: &VJepaTttModel<B>,
) -> Result<Option<std::path::PathBuf>> {
    if !config.model.save_model {
        return Ok(None);
    }
    let path = config.model.output_dir.join("ttt-model");
    model
        .clone()
        .save_file(
            path.clone(),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
        )
        .context("save TTT model")?;
    Ok(Some(path.with_extension("mpk")))
}

struct LossProgress {
    initial_loss: Option<f64>,
    best_loss: f64,
    final_loss: f64,
    loss_trace: Vec<TttStepMetric>,
}

impl LossProgress {
    fn new(max_steps: usize) -> Self {
        Self {
            initial_loss: None,
            best_loss: f64::INFINITY,
            final_loss: 0.0,
            loss_trace: Vec::with_capacity(max_steps),
        }
    }

    fn should_read_step(&self, step: usize, config: &BurnJepaTrainConfig) -> bool {
        step == config.training.max_steps
            || (config.training.loss_trace_interval > 0
                && step % config.training.loss_trace_interval == 0)
    }

    fn record(&mut self, step: usize, loss: f64, trace_interval: usize) {
        self.initial_loss.get_or_insert(loss);
        self.best_loss = self.best_loss.min(loss);
        self.final_loss = loss;
        if trace_interval > 0 && step % trace_interval == 0 {
            self.loss_trace.push(TttStepMetric { step, loss });
        }
    }
}
