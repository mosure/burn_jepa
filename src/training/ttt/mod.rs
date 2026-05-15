mod eval;
mod loss;
mod metrics;
mod step;

use super::batch::TrainingBatchPlanner;
use super::config::BurnJepaTrainConfig;
use super::model_io::{load_student_model, load_teacher_model};
use super::report::{
    TrainingLossSummary, TttBackpropMetrics, TttEvalReport, TttStageMetrics, TttStepMetric,
    TttStreamStepKind, TttStreamTrainingMetrics, TttTrainingReport, samples_per_second,
    save_training_report, save_ttt_training_report, tensor_scalar,
};
use crate::{JepaSampleMetadata, TttState, VJepaTttModel, dataset_from_config, video_token_grid};
use anyhow::{Context, Result, ensure};
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::time::Instant;

pub use loss::{TttDistillationLoss, evaluate_ttt_distillation};
pub use step::TttSparsePatchifyTrainingBackend;

#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
struct StreamKey {
    clip_id: Option<String>,
    domain: Option<String>,
    source: Option<String>,
}

impl StreamKey {
    fn from_metadata(metadata: &JepaSampleMetadata) -> Self {
        Self {
            clip_id: metadata.clip_id.clone(),
            domain: metadata.domain.clone(),
            source: metadata.source.clone(),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct StreamSlot<B: Backend> {
    start_frame: Option<usize>,
    windows_in_stream: usize,
    state: Option<TttState<B>>,
}

#[derive(Clone, Debug, Default)]
struct StreamStateTracker<B: Backend> {
    streams: BTreeMap<StreamKey, StreamSlot<B>>,
    active_keys: Vec<StreamKey>,
    carried_steps: usize,
    reset_steps: usize,
    detached_steps: usize,
    decayed_steps: usize,
    packed_batches: usize,
    max_packed_batch_size: usize,
    max_active_streams: usize,
    reset_batches: usize,
    carried_batches: usize,
    mixed_batches: usize,
    current_kind: Option<TttStreamStepKind>,
    current_reset_interval: Option<usize>,
}

impl<B: Backend> StreamStateTracker<B> {
    fn begin_step(
        &mut self,
        model: &VJepaTttModel<B>,
        config: &BurnJepaTrainConfig,
        step_index: usize,
        batch_size: usize,
        metadata: &[JepaSampleMetadata],
    ) -> Result<TttState<B>> {
        if !config.training.stream.enabled {
            self.current_kind = None;
            self.current_reset_interval = None;
            return Ok(model.fresh_state());
        }
        let batch_size = batch_size.max(1);
        ensure!(
            metadata.len() >= batch_size || batch_size == 1,
            "training.stream.enabled requires one metadata row per packed batch sample"
        );
        let mut seen = BTreeSet::new();
        self.active_keys.clear();
        self.packed_batches += 1;
        self.max_packed_batch_size = self.max_packed_batch_size.max(batch_size);
        let reset_interval = config.training.stream.reset_interval_for_step(step_index);
        self.current_reset_interval = Some(reset_interval);
        let mut row_states = Vec::with_capacity(batch_size);
        let mut saw_reset = false;
        let mut saw_carried = false;
        for row in 0..batch_size {
            let row_metadata = metadata.get(row).cloned().unwrap_or_default();
            let key = StreamKey::from_metadata(&row_metadata);
            ensure!(
                seen.insert(key.clone()),
                "training.stream.enabled packed batches require at most one window per stream key; duplicate key {:?} appears in one batch",
                key
            );
            let slot = self.streams.entry(key.clone()).or_default();
            let non_monotonic_start = config.training.stream.reset_on_non_monotonic_start
                && slot
                    .start_frame
                    .zip(row_metadata.start_frame)
                    .is_some_and(|(previous, current)| current <= previous);
            let interval_reset = reset_interval > 0 && slot.windows_in_stream >= reset_interval;
            let reset = slot.state.is_none() || non_monotonic_start || interval_reset;
            let state = if reset {
                slot.windows_in_stream = 1;
                self.reset_steps += 1;
                saw_reset = true;
                model.fresh_state()
            } else {
                slot.windows_in_stream += 1;
                self.carried_steps += 1;
                saw_carried = true;
                slot.state.take().unwrap_or_else(|| model.fresh_state())
            };
            slot.start_frame = row_metadata.start_frame;
            self.active_keys.push(key);
            row_states.push(state);
        }
        self.max_active_streams = self.max_active_streams.max(self.streams.len());
        let kind = match (saw_reset, saw_carried) {
            (true, true) => TttStreamStepKind::Mixed,
            (true, false) => TttStreamStepKind::Reset,
            (false, true) => TttStreamStepKind::Carried,
            (false, false) => TttStreamStepKind::Reset,
        };
        match kind {
            TttStreamStepKind::Reset => self.reset_batches += 1,
            TttStreamStepKind::Carried => self.carried_batches += 1,
            TttStreamStepKind::Mixed => self.mixed_batches += 1,
        }
        self.current_kind = Some(kind);
        Ok(TttState::pack_rows(&row_states))
    }

    fn current_kind(&self) -> Option<TttStreamStepKind> {
        self.current_kind
    }

    fn current_reset_interval(&self) -> Option<usize> {
        self.current_reset_interval
    }

    fn finish_step(&mut self, state: TttState<B>, config: &BurnJepaTrainConfig) {
        if !config.training.stream.enabled {
            return;
        }
        let row_count = self.active_keys.len();
        let row_states = state.unpack_rows(row_count);
        for (key, mut state) in self.active_keys.drain(..).zip(row_states.into_iter()) {
            if config.training.stream.detach_between_steps {
                state.detach();
                self.detached_steps += 1;
            }
            if config.training.stream.state_decay < 1.0 {
                state.decay(config.training.stream.state_decay);
                self.decayed_steps += 1;
            }
            self.streams.entry(key).or_default().state = Some(state);
        }
    }

    fn metrics(
        &self,
        config: &BurnJepaTrainConfig,
        include_optimizer_steps: bool,
    ) -> TttStreamTrainingMetrics {
        TttStreamTrainingMetrics {
            enabled: config.training.stream.enabled,
            detach_between_steps: config.training.stream.detach_between_steps,
            reset_on_clip_change: config.training.stream.reset_on_clip_change,
            reset_on_non_monotonic_start: config.training.stream.reset_on_non_monotonic_start,
            reset_interval_steps: config.training.stream.reset_interval_steps,
            curriculum_enabled: config.training.stream.curriculum.enabled,
            curriculum_initial_reset_interval_steps: config
                .training
                .stream
                .curriculum
                .initial_reset_interval_steps,
            curriculum_final_reset_interval_steps: config
                .training
                .stream
                .curriculum
                .final_reset_interval_steps,
            curriculum_warmup_steps: config.training.stream.curriculum.warmup_steps,
            final_effective_reset_interval_steps: config
                .training
                .stream
                .reset_interval_for_step(config.training.max_steps.saturating_sub(1)),
            state_decay: config.training.stream.state_decay,
            state_l2_weight: config.training.stream.state_l2_weight,
            update_l2_weight: config.training.stream.update_l2_weight,
            active_streams: self.streams.len(),
            max_active_streams: self.max_active_streams.max(self.streams.len()),
            packed_batches: self.packed_batches,
            max_packed_batch_size: self.max_packed_batch_size,
            carried_steps: self.carried_steps,
            reset_steps: self.reset_steps,
            optimizer_steps: include_optimizer_steps.then_some(self.packed_batches),
            reset_optimizer_steps: include_optimizer_steps.then_some(self.reset_batches),
            carried_optimizer_steps: include_optimizer_steps.then_some(self.carried_batches),
            mixed_optimizer_steps: include_optimizer_steps.then_some(self.mixed_batches),
            detached_steps: self.detached_steps,
            decayed_steps: self.decayed_steps,
        }
    }
}

fn add_stream_regularization<B: Backend>(
    loss: Tensor<B, 1>,
    config: &BurnJepaTrainConfig,
    before: &TttState<B>,
    after: &TttState<B>,
) -> Tensor<B, 1> {
    if !config.training.stream.enabled {
        return loss;
    }
    let mut total = loss;
    if config.training.stream.state_l2_weight > 0.0
        && let Some(penalty) = state_l2_penalty(after)
    {
        total = total + penalty.mul_scalar(config.training.stream.state_l2_weight);
    }
    if config.training.stream.update_l2_weight > 0.0
        && let Some(penalty) = state_update_l2_penalty(before, after)
    {
        total = total + penalty.mul_scalar(config.training.stream.update_l2_weight);
    }
    total
}

fn state_l2_penalty<B: Backend>(state: &TttState<B>) -> Option<Tensor<B, 1>> {
    mean_penalty(
        state
            .layers
            .iter()
            .filter_map(|layer| layer.fast_weight.as_ref())
            .map(|weight| weight.clone().powf_scalar(2.0).mean()),
    )
}

fn state_update_l2_penalty<B: Backend>(
    before: &TttState<B>,
    after: &TttState<B>,
) -> Option<Tensor<B, 1>> {
    mean_penalty(
        before
            .layers
            .iter()
            .zip(after.layers.iter())
            .filter_map(|(before, after)| {
                let after = after.fast_weight.as_ref()?;
                let delta = match before.fast_weight.as_ref() {
                    Some(before) => after.clone() - before.clone(),
                    None => after.clone(),
                };
                Some(delta.powf_scalar(2.0).mean())
            }),
    )
}

fn mean_penalty<B: Backend>(penalties: impl Iterator<Item = Tensor<B, 1>>) -> Option<Tensor<B, 1>> {
    let mut count = 0usize;
    let mut total = None;
    for penalty in penalties {
        count += 1;
        total = Some(match total {
            Some(total) => total + penalty,
            None => penalty,
        });
    }
    total.map(|total| total.div_scalar(count.max(1) as f64))
}

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
    validate_stream_dataset(config, dataset.as_ref())?;
    let batch_planner = TrainingBatchPlanner::new(dataset.as_ref(), config.training.batching)?;
    let memory = metrics::ttt_memory_metrics(config, model.config());
    model.set_backprop_mode(crate::TttBackpropMode::FinalFeature);
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
    let mut teacher_cache = BTreeMap::<String, step::TeacherTokenTargets<B>>::new();
    let mut observed_dense_tokens = None;
    let mut final_grad_metrics = None;
    let rollout = step::rollout_kind(config);
    let patchify = step::patchify_kind::<B>(config, rollout)?;
    let capture_layers = config.ttt.capture_layers(model.config());
    let mut stream_state = StreamStateTracker::<B>::default();
    let mut train_samples = 0usize;

    for step_index in 0..config.training.max_steps {
        let supervision = config
            .ttt
            .train_supervision_for_step(step_index, config.training.max_steps);
        model.set_backprop_mode(match supervision {
            crate::TttSupervisionMode::LayerLocalTeacher => crate::TttBackpropMode::LayerLocal,
            crate::TttSupervisionMode::FinalTeacher | crate::TttSupervisionMode::Hybrid => {
                config.ttt.backprop_mode
            }
        });
        let batch = batch_planner.load_batch::<B>(
            dataset.as_ref(),
            &config.dataset,
            model.config(),
            device,
            step_index * config.training.batch_size,
            config.training.batch_size,
        )?;
        observed_dense_tokens.get_or_insert_with(|| {
            let [_, _, frames, height, width] = batch.student.shape().dims::<5>();
            video_token_grid(model.config(), frames, height, width)
                .map(|grid| grid.len())
                .unwrap_or_else(|_| model.config().num_patches())
        });
        let batch_size = batch.student.shape().dims::<5>()[0];
        train_samples += batch_size;
        let masks = timed(&mut train_stage.mask_ms, || {
            step::resolve_masks(config, &batch.student, model.config(), &batch.metadata)
        })?;
        if mask_metrics.is_none()
            && let Some(masks) = &masks
        {
            mask_metrics = Some(metrics::mask_metrics_from_batches(
                &masks.context,
                &masks.target,
            ));
        }
        let mut carried_state =
            stream_state.begin_step(&model, config, step_index, batch_size, &batch.metadata)?;
        let previous_state = carried_state.clone();
        let stream_step_kind = stream_state.current_kind();
        let stream_reset_interval = stream_state.current_reset_interval();
        let teacher_tokens = teacher_tokens_for_batch(
            &teacher,
            batch.teacher.clone(),
            &batch.metadata,
            step_index,
            &capture_layers,
            config.training.cache_teacher_tokens,
            &mut teacher_cache,
            &mut train_stage,
        )?;
        let student = timed(&mut train_stage.student_forward_ms, || {
            if config.training.stream.enabled {
                step::student_training_rollout_with_state(
                    &model,
                    batch.student,
                    teacher_tokens.final_tokens.clone(),
                    masks.as_ref(),
                    rollout,
                    patchify,
                    &mut carried_state,
                )
            } else {
                step::student_training_rollout(
                    &model,
                    batch.student,
                    teacher_tokens.final_tokens.clone(),
                    masks.as_ref(),
                    rollout,
                    patchify,
                )
            }
        })?;
        let loss = timed(&mut train_stage.loss_ms, || {
            let predictor_target = step::teacher_predictor_targets(
                &teacher,
                &teacher_tokens,
                masks.as_ref(),
                student.grid,
                config.loss.predictor_loss_weight,
            )?;
            loss::training_loss(
                &model,
                config,
                &student,
                &teacher_tokens,
                masks.as_ref(),
                rollout,
                batch_size,
                device,
                supervision,
                predictor_target,
            )
            .map(|loss| add_stream_regularization(loss, config, &previous_state, &carried_state))
        })?;
        let step_number = step_index + 1;
        let save_partial =
            config.training.save_steps > 0 && step_number % config.training.save_steps == 0;
        let read_loss = progress.should_read_step(step_number, config) || save_partial;
        if read_loss {
            let final_loss = tensor_scalar(loss.clone().detach())?;
            progress.record(
                step_number,
                final_loss,
                config.training.loss_trace_interval,
                stream_step_kind,
                stream_reset_interval,
            );
        }

        let backward_start = Instant::now();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        if step_number == config.training.max_steps && config.training.eval_utilization_diagnostics
        {
            final_grad_metrics = Some(metrics::ttt_gradient_metrics(config, &model, &grads)?);
        }
        let backward_ms = backward_start.elapsed().as_millis();
        train_stage.backward_ms += backward_ms;
        train_stage.backward_optim_ms += backward_ms;

        let optim_start = Instant::now();
        model = optim.step(
            config.training.learning_rate_for_step(step_index),
            model,
            grads,
        );
        stream_state.finish_step(carried_state, config);
        let optimizer_ms = optim_start.elapsed().as_millis();
        train_stage.optimizer_ms += optimizer_ms;
        train_stage.backward_optim_ms += optimizer_ms;

        if save_partial {
            save_training_report(
                &config.model.output_dir,
                "ttt-report.partial.json",
                step_number,
                train_samples,
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
    model.set_backprop_mode(crate::TttBackpropMode::FinalFeature);
    let (
        eval_loss,
        eval_feature_loss,
        eval_predictor_loss,
        eval_cosine,
        teacher_forced_eval_loss,
        teacher_forced_eval_cosine,
        teacher_forcing_loss_gap,
        teacher_forcing_cosine_gap,
        eval_full_loss,
        eval_full_cosine,
        eval_samples,
        eval_stage,
        eval_domains,
        mut utilization,
        temporal_diagnostics,
        temporal_segments,
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
            Some(eval.feature_loss),
            eval.predictor_loss,
            Some(eval.cosine),
            eval.teacher_forced_loss,
            eval.teacher_forced_cosine,
            eval.teacher_forcing_loss_gap,
            eval.teacher_forcing_cosine_gap,
            eval.full_loss,
            eval.full_cosine,
            eval.samples,
            eval.stage,
            eval.domains,
            eval.utilization,
            eval.temporal_diagnostics,
            eval.temporal_segments,
        )
    } else {
        (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            TttStageMetrics::default(),
            Vec::new(),
            None,
            None,
            None,
        )
    };
    if let (Some(utilization), Some(gradients)) =
        (utilization.as_mut(), final_grad_metrics.as_ref())
    {
        metrics::merge_gradient_metrics(utilization, gradients);
    }
    let eval_elapsed_ms = eval_start.elapsed().as_millis();
    let model_path = save_model_if_enabled(config, &model)?;
    let samples = train_samples;
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
        stream: stream_state.metrics(config, true),
        lr_schedule: config.training.lr_schedule.clone(),
        lr_stats: config.training.learning_rate_stats(),
        target_supervision: metrics::target_supervision_metrics(config),
        pre_train_eval_loss: pre_train_eval.as_ref().map(|eval| eval.loss),
        pre_train_eval_feature_loss: pre_train_eval.as_ref().map(|eval| eval.feature_loss),
        pre_train_eval_predictor_loss: pre_train_eval.as_ref().and_then(|eval| eval.predictor_loss),
        pre_train_eval_cosine: pre_train_eval.as_ref().map(|eval| eval.cosine),
        pre_train_teacher_forced_eval_loss: pre_train_eval
            .as_ref()
            .and_then(|eval| eval.teacher_forced_loss),
        pre_train_teacher_forced_eval_cosine: pre_train_eval
            .as_ref()
            .and_then(|eval| eval.teacher_forced_cosine),
        pre_train_teacher_forcing_loss_gap: pre_train_eval
            .as_ref()
            .and_then(|eval| eval.teacher_forcing_loss_gap),
        pre_train_teacher_forcing_cosine_gap: pre_train_eval
            .as_ref()
            .and_then(|eval| eval.teacher_forcing_cosine_gap),
        pre_train_full_eval_loss: pre_train_eval.as_ref().and_then(|eval| eval.full_loss),
        pre_train_full_eval_cosine: pre_train_eval.as_ref().and_then(|eval| eval.full_cosine),
        eval_loss,
        eval_feature_loss,
        eval_predictor_loss,
        eval_cosine,
        teacher_forced_eval_loss,
        teacher_forced_eval_cosine,
        teacher_forcing_loss_gap,
        teacher_forcing_cosine_gap,
        eval_full_loss,
        eval_full_cosine,
        eval_samples,
        train_stage,
        eval_stage,
        eval_domains,
        utilization,
        temporal_diagnostics,
        temporal_segments,
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
    let mut model = VJepaTttModel::from_model(base, config.ttt.clone(), device)?
        .load_file(
            model_path.clone(),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
            device,
        )
        .with_context(|| format!("load TTT model {}", model_path.display()))?;
    model.set_backprop_mode(crate::TttBackpropMode::FinalFeature);
    let memory = metrics::ttt_memory_metrics_for_batch_size(
        config,
        model.config(),
        config.training.effective_eval_batch_size(),
    );
    let rollout = step::rollout_kind(config);
    let patchify = step::patchify_kind::<B>(config, rollout)?;

    let dataset = dataset_from_config(&config.dataset, false)?;
    let batch_planner = TrainingBatchPlanner::new(dataset.as_ref(), config.training.batching)?;
    let first_batch = batch_planner.load_batch::<B>(
        dataset.as_ref(),
        &config.dataset,
        model.config(),
        device,
        0,
        config.training.effective_eval_batch_size(),
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
        .map(|masks| metrics::mask_metrics_from_batches(&masks.context, &masks.target));

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
        feature_loss: eval.feature_loss,
        predictor_loss: eval.predictor_loss,
        cosine: eval.cosine,
        teacher_forced_loss: eval.teacher_forced_loss,
        teacher_forced_cosine: eval.teacher_forced_cosine,
        teacher_forcing_loss_gap: eval.teacher_forcing_loss_gap,
        teacher_forcing_cosine_gap: eval.teacher_forcing_cosine_gap,
        full_loss: eval.full_loss,
        full_cosine: eval.full_cosine,
        memory,
        mask: mask_metrics,
        rollout,
        target_supervision: metrics::target_supervision_metrics(config),
        stage: eval.stage,
        domains: eval.domains,
        utilization: eval.utilization,
        temporal_diagnostics: eval.temporal_diagnostics,
        temporal_segments: eval.temporal_segments,
        stream: eval.stream,
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
    capture_layers: &[usize],
    enabled: bool,
    cache: &mut BTreeMap<String, step::TeacherTokenTargets<B>>,
    stage: &mut TttStageMetrics,
) -> Result<step::TeacherTokenTargets<B>> {
    if !enabled {
        return timed(&mut stage.teacher_forward_ms, || {
            Ok(step::teacher_targets(teacher, video, capture_layers))
        });
    }
    let key = teacher_cache_key(metadata, fallback_index, capture_layers);
    if let Some(tokens) = cache.get(&key) {
        stage.teacher_cache_hits += 1;
        return Ok(tokens.clone());
    }
    stage.teacher_cache_misses += 1;
    let tokens = timed(&mut stage.teacher_forward_ms, || {
        Ok(step::teacher_targets(teacher, video, capture_layers))
    })?;
    cache.insert(key, tokens.clone());
    Ok(tokens)
}

fn teacher_cache_key(
    metadata: &[crate::JepaSampleMetadata],
    fallback_index: usize,
    capture_layers: &[usize],
) -> String {
    let has_identity = metadata.iter().any(|row| {
        row.clip_id.is_some()
            || row.source.is_some()
            || row.start_frame.is_some()
            || row.precomputed_context_indices.is_some()
            || row.precomputed_target_indices.is_some()
    });
    if has_identity {
        format!(
            "layers={capture_layers:?}:{}",
            serde_json::to_string(metadata)
                .unwrap_or_else(|_| format!("fallback:{fallback_index}"))
        )
    } else {
        format!("layers={capture_layers:?}:fallback:{fallback_index}")
    }
}

fn validate_stream_dataset(
    config: &BurnJepaTrainConfig,
    dataset: &dyn crate::JepaDataset,
) -> Result<()> {
    if !config.training.stream.enabled {
        return Ok(());
    }
    let max_batch_size = config
        .training
        .batch_size
        .max(config.training.effective_eval_batch_size());
    ensure!(
        max_batch_size == 1 || config.dataset.kind == crate::JepaDatasetKind::Manifest,
        "training.stream.enabled with batch_size > 1 requires a manifest dataset so each packed row has a stable stream identity"
    );
    if config.dataset.kind != crate::JepaDatasetKind::Manifest {
        return Ok(());
    }
    let rows = dataset.len().min(16);
    for index in 0..rows {
        let sample = dataset.sample(index)?;
        let metadata = sample.metadata().cloned().unwrap_or_default();
        ensure!(
            metadata.clip_id.is_some() || metadata.source.is_some(),
            "training.stream.enabled with a manifest dataset requires clip_id or source metadata on row {}",
            index + 1
        );
        ensure!(
            metadata.start_frame.is_some(),
            "training.stream.enabled with a manifest dataset requires start_frame metadata on row {}",
            index + 1
        );
    }
    Ok(())
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

    fn record(
        &mut self,
        step: usize,
        loss: f64,
        trace_interval: usize,
        stream_step: Option<TttStreamStepKind>,
        effective_reset_interval_steps: Option<usize>,
    ) {
        self.initial_loss.get_or_insert(loss);
        self.best_loss = self.best_loss.min(loss);
        self.final_loss = loss;
        if trace_interval > 0 && step % trace_interval == 0 {
            self.loss_trace.push(TttStepMetric {
                step,
                loss,
                stream_step,
                effective_reset_interval_steps,
            });
        }
    }
}
