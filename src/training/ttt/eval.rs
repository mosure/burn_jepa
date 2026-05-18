use super::loss;
use super::step::{self, TttRolloutKind};
use crate::training::batch::TrainingBatchPlanner;
use crate::training::config::BurnJepaTrainConfig;
use crate::training::report::{
    TttDomainEvalMetric, TttLongRolloutMetrics, TttLongRolloutSegmentMetric,
    TttLongRolloutStreamMetric, TttStageMetrics, TttStreamTrainingMetrics,
    TttTemporalDiagnosticMetrics, TttTemporalSegmentMetric, TttTemporalSegmentMetrics,
    TttUtilizationMetrics, tensor_scalar,
};
use crate::{
    JepaSampleMetadata, SparseMaskBatch, TokenGridShape, VJepa2_1Model, VJepaTttModel,
    dataset_from_config,
};
use crate::{TttMemoryUpdateSource, TttStateResetMode, TttSupervisionMode};
use anyhow::{Context, Result};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

pub(super) struct TttEvalSummary {
    pub samples: usize,
    pub loss: f64,
    pub feature_loss: f64,
    pub predictor_loss: Option<f64>,
    pub regularizer_loss: Option<f64>,
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
    pub temporal_segments: Option<TttTemporalSegmentMetrics>,
    pub long_rollout: Option<TttLongRolloutMetrics>,
    pub stream: TttStreamTrainingMetrics,
}

#[derive(Default)]
struct DomainTotals {
    samples: usize,
    loss: f64,
    cosine: f64,
    regularizer_samples: usize,
    regularizer_loss: f64,
    teacher_forced_samples: usize,
    teacher_forced_loss: f64,
    teacher_forced_cosine: f64,
    full_samples: usize,
    full_loss: f64,
    full_cosine: f64,
}

#[derive(Clone, Debug, Default)]
struct LongRolloutTotal {
    samples: usize,
    loss: f64,
    cosine: f64,
}

#[derive(Clone, Debug, Default)]
struct LongRolloutStreamTotal {
    domain: String,
    samples: usize,
    first_start_frame: Option<usize>,
    last_start_frame: Option<usize>,
    current_consecutive_windows: usize,
    longest_consecutive_windows: usize,
    loss: f64,
    cosine: f64,
}

struct LongRolloutAccumulator {
    expected_windows: usize,
    samples: usize,
    segments: Vec<LongRolloutTotal>,
    streams: BTreeMap<String, LongRolloutStreamTotal>,
    previous_scene_key: Option<String>,
    windows_since_scene_switch: Option<usize>,
    scene_switches: usize,
    switch_window: LongRolloutTotal,
    recovery_window: LongRolloutTotal,
    steady_state: LongRolloutTotal,
}

impl LongRolloutAccumulator {
    fn new(expected_windows: usize, segment_count: usize) -> Self {
        let expected_windows = expected_windows.max(1);
        let segment_count = segment_count.clamp(1, expected_windows);
        Self {
            expected_windows,
            samples: 0,
            segments: vec![LongRolloutTotal::default(); segment_count],
            streams: BTreeMap::new(),
            previous_scene_key: None,
            windows_since_scene_switch: None,
            scene_switches: 0,
            switch_window: LongRolloutTotal::default(),
            recovery_window: LongRolloutTotal::default(),
            steady_state: LongRolloutTotal::default(),
        }
    }

    fn add_batch(
        &mut self,
        loss: f64,
        cosine: f64,
        batch_size: usize,
        metadata: &[JepaSampleMetadata],
    ) {
        for row in 0..batch_size {
            let sample_index = self.samples;
            let segment = ((sample_index * self.segments.len()) / self.expected_windows)
                .min(self.segments.len().saturating_sub(1));
            let total = &mut self.segments[segment];
            total.samples += 1;
            total.loss += loss;
            total.cosine += cosine;

            let metadata = metadata.get(row).cloned().unwrap_or_default();
            self.add_scene_switch_metrics(loss, cosine, &metadata);
            let stream_key = rollout_stream_key(&metadata);
            let domain = metadata
                .domain
                .clone()
                .unwrap_or_else(|| "unknown".to_string());
            let stream = self
                .streams
                .entry(stream_key)
                .or_insert_with(|| LongRolloutStreamTotal {
                    domain,
                    ..LongRolloutStreamTotal::default()
                });
            if stream.samples == 0 {
                stream.first_start_frame = metadata.start_frame;
                stream.current_consecutive_windows = 1;
            } else if is_consecutive_window(stream.last_start_frame, metadata.start_frame) {
                stream.current_consecutive_windows += 1;
            } else {
                stream.current_consecutive_windows = 1;
            }
            stream.longest_consecutive_windows = stream
                .longest_consecutive_windows
                .max(stream.current_consecutive_windows);
            stream.last_start_frame = metadata.start_frame;
            stream.samples += 1;
            stream.loss += loss;
            stream.cosine += cosine;
            self.samples += 1;
        }
    }

    fn finish(self) -> Option<TttLongRolloutMetrics> {
        if self.samples == 0 {
            return None;
        }
        let count = self.segments.len();
        let segments = self
            .segments
            .into_iter()
            .enumerate()
            .map(|(segment, total)| {
                let samples = total.samples.max(1);
                TttLongRolloutSegmentMetric {
                    segment,
                    start_window: (segment * self.expected_windows).div_ceil(count),
                    end_window: ((segment + 1) * self.expected_windows).div_ceil(count),
                    samples: total.samples,
                    loss: total.loss / samples as f64,
                    cosine: total.cosine / samples as f64,
                }
            })
            .collect::<Vec<_>>();
        let first = segments.first();
        let last = segments.last();
        let late_minus_early_loss = first.zip(last).and_then(|(first, last)| {
            (first.samples > 0 && last.samples > 0).then_some(last.loss - first.loss)
        });
        let late_minus_early_cosine = first.zip(last).and_then(|(first, last)| {
            (first.samples > 0 && last.samples > 0).then_some(last.cosine - first.cosine)
        });
        let mut longest_stream_windows = 0usize;
        let mut longest_consecutive_windows = 0usize;
        let stream_segments = self
            .streams
            .into_iter()
            .map(|(stream, total)| {
                longest_stream_windows = longest_stream_windows.max(total.samples);
                longest_consecutive_windows =
                    longest_consecutive_windows.max(total.longest_consecutive_windows);
                let samples = total.samples.max(1);
                TttLongRolloutStreamMetric {
                    stream,
                    domain: total.domain,
                    samples: total.samples,
                    first_start_frame: total.first_start_frame,
                    last_start_frame: total.last_start_frame,
                    longest_consecutive_windows: total.longest_consecutive_windows,
                    loss: total.loss / samples as f64,
                    cosine: total.cosine / samples as f64,
                }
            })
            .collect::<Vec<_>>();
        Some(TttLongRolloutMetrics {
            samples: self.samples,
            windows: self.samples,
            streams: stream_segments.len(),
            longest_stream_windows,
            longest_consecutive_windows,
            scene_switches: self.scene_switches,
            switch_window_samples: self.switch_window.samples,
            switch_window_loss: mean_loss(&self.switch_window),
            switch_window_cosine: mean_cosine(&self.switch_window),
            recovery_window_samples: self.recovery_window.samples,
            recovery_window_loss: mean_loss(&self.recovery_window),
            recovery_window_cosine: mean_cosine(&self.recovery_window),
            steady_state_samples: self.steady_state.samples,
            steady_state_loss: mean_loss(&self.steady_state),
            steady_state_cosine: mean_cosine(&self.steady_state),
            segments,
            late_minus_early_loss,
            late_minus_early_cosine,
            stream_segments,
        })
    }

    fn add_scene_switch_metrics(&mut self, loss: f64, cosine: f64, metadata: &JepaSampleMetadata) {
        let scene_key = rollout_scene_key(metadata);
        let switched = self
            .previous_scene_key
            .as_ref()
            .is_some_and(|previous| previous != &scene_key);
        self.previous_scene_key = Some(scene_key);
        if switched {
            self.scene_switches += 1;
            self.windows_since_scene_switch = Some(0);
        }
        match self.windows_since_scene_switch {
            Some(0) => add_total(&mut self.switch_window, loss, cosine, 1),
            Some(1..=4) => add_total(&mut self.recovery_window, loss, cosine, 1),
            _ => add_total(&mut self.steady_state, loss, cosine, 1),
        }
        if let Some(distance) = self.windows_since_scene_switch.as_mut() {
            *distance += 1;
        }
    }
}

fn add_total(total: &mut LongRolloutTotal, loss: f64, cosine: f64, samples: usize) {
    total.samples += samples;
    total.loss += loss * samples as f64;
    total.cosine += cosine * samples as f64;
}

fn mean_loss(total: &LongRolloutTotal) -> Option<f64> {
    (total.samples > 0).then_some(total.loss / total.samples as f64)
}

fn mean_cosine(total: &LongRolloutTotal) -> Option<f64> {
    (total.samples > 0).then_some(total.cosine / total.samples as f64)
}

pub(super) fn evaluate_ttt_dataset<B: step::TttSparsePatchifyBackend>(
    model: &VJepaTttModel<B>,
    teacher: &VJepa2_1Model<B>,
    config: &BurnJepaTrainConfig,
    device: &B::Device,
    steps: usize,
) -> Result<TttEvalSummary> {
    let dataset = dataset_from_config(&config.dataset, false)?;
    super::validate_stream_dataset(config, dataset.as_ref())?;
    let batch_planner = TrainingBatchPlanner::new(dataset.as_ref(), config.training.batching)?;
    let rollout = step::rollout_kind(config);
    let patchify = step::patchify_kind::<B>(config, rollout)?;
    let eval_batch_size = config.training.effective_eval_batch_size();
    let eval_full_grid = config.training.eval_full_grid || rollout == TttRolloutKind::Dense;
    let mut total = 0.0;
    let mut total_feature = 0.0;
    let mut total_predictor = 0.0;
    let mut total_regularizer = 0.0;
    let mut total_cosine = 0.0;
    let mut total_teacher_forced_loss = 0.0;
    let mut total_teacher_forced_cosine = 0.0;
    let mut total_full_loss = 0.0;
    let mut total_full_cosine = 0.0;
    let mut samples = 0usize;
    let mut predictor_samples = 0usize;
    let mut regularizer_samples = 0usize;
    let mut teacher_forced_samples = 0usize;
    let mut full_samples = 0usize;
    let mut stage = TttStageMetrics::default();
    let mut domains = BTreeMap::<String, DomainTotals>::new();
    let mut teacher_cache = BTreeMap::<String, step::TeacherTokenTargets<B>>::new();
    let mut teacher_cache_order = VecDeque::<String>::new();
    let mut utilization = None;
    let mut temporal_diagnostics = TemporalDiagnosticAccumulator::default();
    let mut temporal_segments = TemporalSegmentAccumulator::new(3);
    let expected_windows = steps.saturating_mul(eval_batch_size.max(1)).max(1);
    let mut long_rollout = LongRolloutAccumulator::new(expected_windows, 4);
    let mut stream_state = super::StreamStateTracker::<B>::default();
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
        let batch = batch_planner.load_batch::<B>(
            dataset.as_ref(),
            &config.dataset,
            model.config(),
            device,
            start_index,
            batch_size,
        )?;
        let teacher_tokens = super::teacher_tokens_for_batch(
            teacher,
            batch.teacher.clone(),
            &batch.metadata,
            start_index,
            &capture_layers,
            config.training.cache_teacher_tokens,
            config.training.teacher_cache_max_entries,
            &mut teacher_cache,
            &mut teacher_cache_order,
            &mut stage,
        )?;
        let batch_size = batch.student.shape().dims::<5>()[0];
        let masks = super::timed(&mut stage.mask_ms, || {
            step::resolve_masks(config, &batch.student, model.config(), &batch.metadata)
        })?;
        let mut eval_state = config
            .training
            .stream
            .enabled
            .then(|| {
                stream_state.begin_step(model, config, step_index, batch_size, &batch.metadata)
            })
            .transpose()?;
        let student = super::timed(&mut stage.student_forward_ms, || {
            if let Some(state) = eval_state.as_mut() {
                step::student_eval_rollout_with_state(
                    model,
                    batch.student.clone(),
                    teacher_tokens.final_tokens.clone(),
                    masks.as_ref(),
                    rollout,
                    patchify,
                    eval_full_grid,
                    teacher_forced_eval,
                    state,
                )
            } else {
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
            }
        })?;
        if let Some(state) = eval_state {
            stream_state.finish_step(state, config);
        }
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
        if config.training.eval_temporal_diagnostics {
            let diagnostics = eval_temporal_diagnostics(
                model,
                config,
                batch.student.clone(),
                teacher_tokens.clone(),
                masks.as_ref(),
                rollout,
                batch_size,
                device,
            )?;
            temporal_diagnostics.add(&diagnostics, batch_size);
        }
        let loss_start = Instant::now();
        let predictor_target = step::teacher_predictor_targets(
            teacher,
            &teacher_tokens,
            masks.as_ref(),
            student.free_run.grid,
            config.loss.predictor_loss_weight,
        )?;
        let loss_breakdown = loss::training_loss_breakdown(
            model,
            config,
            &student.free_run,
            &teacher_tokens,
            masks.as_ref(),
            rollout,
            batch_size,
            device,
            TttSupervisionMode::FinalTeacher,
            predictor_target.clone(),
        )?;
        let primary_loss = tensor_scalar(loss_breakdown.total.detach())?;
        let feature_loss = tensor_scalar(loss_breakdown.feature.detach())?;
        let predictor_loss = loss_breakdown
            .predictor
            .map(|predictor| tensor_scalar(predictor.detach()))
            .transpose()?;
        let regularizer_loss = loss_breakdown
            .regularizer
            .map(|regularizer| tensor_scalar(regularizer.detach()))
            .transpose()?;
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
        long_rollout.add_batch(primary_loss, primary_cosine, batch_size, &batch.metadata);
        if config.training.eval_temporal_diagnostics {
            temporal_segments.add_batch(
                student.free_run.tokens.clone(),
                teacher_tokens.final_tokens.clone(),
                masks.as_ref().map(|masks| match rollout {
                    TttRolloutKind::Dense | TttRolloutKind::SparseTarget => &masks.target,
                    TttRolloutKind::SparseContext => &masks.context,
                }),
                rollout,
                student.free_run.grid,
                batch_size,
                device,
            )?;
        }
        total += primary_loss * batch_size as f64;
        total_feature += feature_loss * batch_size as f64;
        if let Some(predictor_loss) = predictor_loss {
            total_predictor += predictor_loss * batch_size as f64;
            predictor_samples += batch_size;
        }
        let regularizer_loss_for_domain = regularizer_loss;
        if let Some(regularizer_loss) = regularizer_loss {
            total_regularizer += regularizer_loss * batch_size as f64;
            regularizer_samples += batch_size;
        }
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
                    predictor_target,
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
        if let Some(regularizer_loss) = regularizer_loss_for_domain {
            totals.regularizer_samples += batch_size;
            totals.regularizer_loss += regularizer_loss * batch_size as f64;
        }
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
            regularizer_loss: (totals.regularizer_samples > 0)
                .then_some(totals.regularizer_loss / totals.regularizer_samples as f64),
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
        feature_loss: total_feature / samples.max(1) as f64,
        predictor_loss: (predictor_samples > 0)
            .then_some(total_predictor / predictor_samples as f64),
        regularizer_loss: (regularizer_samples > 0)
            .then_some(total_regularizer / regularizer_samples as f64),
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
        temporal_diagnostics: temporal_diagnostics.finish(),
        temporal_segments: temporal_segments.finish(),
        long_rollout: long_rollout.finish(),
        stream: stream_state.metrics(config, false),
    })
}

#[allow(clippy::too_many_arguments)]
fn eval_temporal_diagnostics<B: step::TttSparsePatchifyBackend>(
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
        samples: batch_size,
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

#[derive(Default)]
struct TemporalDiagnosticAccumulator {
    samples: usize,
    reset_each_frame_loss: f64,
    reset_each_frame_cosine: f64,
    reset_each_tubelet_loss: f64,
    reset_each_tubelet_cosine: f64,
    reverse_order_loss: f64,
    reverse_order_cosine: f64,
    shuffle_order_loss: f64,
    shuffle_order_cosine: f64,
    freeze_fast_update_loss: f64,
    freeze_fast_update_cosine: f64,
}

impl TemporalDiagnosticAccumulator {
    fn add(&mut self, metrics: &TttTemporalDiagnosticMetrics, batch_size: usize) {
        let weight = batch_size as f64;
        self.samples += batch_size;
        self.reset_each_frame_loss += metrics.reset_each_frame_loss.unwrap_or(0.0) * weight;
        self.reset_each_frame_cosine += metrics.reset_each_frame_cosine.unwrap_or(0.0) * weight;
        self.reset_each_tubelet_loss += metrics.reset_each_tubelet_loss.unwrap_or(0.0) * weight;
        self.reset_each_tubelet_cosine += metrics.reset_each_tubelet_cosine.unwrap_or(0.0) * weight;
        self.reverse_order_loss += metrics.reverse_order_loss.unwrap_or(0.0) * weight;
        self.reverse_order_cosine += metrics.reverse_order_cosine.unwrap_or(0.0) * weight;
        self.shuffle_order_loss += metrics.shuffle_order_loss.unwrap_or(0.0) * weight;
        self.shuffle_order_cosine += metrics.shuffle_order_cosine.unwrap_or(0.0) * weight;
        self.freeze_fast_update_loss += metrics.freeze_fast_update_loss.unwrap_or(0.0) * weight;
        self.freeze_fast_update_cosine += metrics.freeze_fast_update_cosine.unwrap_or(0.0) * weight;
    }

    fn finish(self) -> Option<TttTemporalDiagnosticMetrics> {
        let samples = self.samples;
        (samples > 0).then(|| {
            let denom = samples as f64;
            TttTemporalDiagnosticMetrics {
                samples,
                reset_each_frame_loss: Some(self.reset_each_frame_loss / denom),
                reset_each_frame_cosine: Some(self.reset_each_frame_cosine / denom),
                reset_each_tubelet_loss: Some(self.reset_each_tubelet_loss / denom),
                reset_each_tubelet_cosine: Some(self.reset_each_tubelet_cosine / denom),
                reverse_order_loss: Some(self.reverse_order_loss / denom),
                reverse_order_cosine: Some(self.reverse_order_cosine / denom),
                shuffle_order_loss: Some(self.shuffle_order_loss / denom),
                shuffle_order_cosine: Some(self.shuffle_order_cosine / denom),
                freeze_fast_update_loss: Some(self.freeze_fast_update_loss / denom),
                freeze_fast_update_cosine: Some(self.freeze_fast_update_cosine / denom),
            }
        })
    }
}

#[derive(Clone, Debug, Default)]
struct SegmentTotal {
    tokens: usize,
    values: usize,
    squared_error: f64,
    dot: f64,
    left_norm: f64,
    right_norm: f64,
}

struct TemporalSegmentAccumulator {
    samples: usize,
    grid_depth: usize,
    segments: Vec<SegmentTotal>,
}

impl TemporalSegmentAccumulator {
    fn new(segments: usize) -> Self {
        Self {
            samples: 0,
            grid_depth: 0,
            segments: vec![SegmentTotal::default(); segments.max(1)],
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_batch<B: Backend>(
        &mut self,
        student_tokens: Tensor<B, 3>,
        teacher_tokens: Tensor<B, 3>,
        primary_mask: Option<&SparseMaskBatch<B>>,
        rollout: TttRolloutKind,
        grid: TokenGridShape,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<()> {
        let rows = primary_token_rows(primary_mask, grid, batch_size);
        let (student_tokens, teacher_tokens) = loss::align_primary_tokens(
            student_tokens,
            teacher_tokens,
            primary_mask,
            rollout,
            batch_size,
            device,
        );
        let [batch, tokens, dim] = student_tokens.shape().dims::<3>();
        let student = student_tokens
            .into_data()
            .to_vec::<f32>()
            .context("read student temporal segment tokens")?;
        let teacher = teacher_tokens
            .into_data()
            .to_vec::<f32>()
            .context("read teacher temporal segment tokens")?;
        self.samples += batch;
        self.grid_depth = self.grid_depth.max(grid.depth);
        let frame_tokens = grid.tokens_per_frame().max(1);
        for sample in 0..batch {
            let row = rows
                .get(sample)
                .with_context(|| format!("missing temporal segment row {sample}"))?;
            for token in 0..tokens {
                let Some(&token_index) = row.get(token) else {
                    continue;
                };
                let tubelet = (token_index / frame_tokens).min(grid.depth.saturating_sub(1));
                let segment = ((tubelet * self.segments.len()) / grid.depth.max(1))
                    .min(self.segments.len().saturating_sub(1));
                let base = (sample * tokens + token) * dim;
                let total = &mut self.segments[segment];
                total.tokens += 1;
                total.values += dim;
                for offset in 0..dim {
                    let left = student[base + offset] as f64;
                    let right = teacher[base + offset] as f64;
                    let diff = left - right;
                    total.squared_error += diff * diff;
                    total.dot += left * right;
                    total.left_norm += left * left;
                    total.right_norm += right * right;
                }
            }
        }
        Ok(())
    }

    fn finish(self) -> Option<TttTemporalSegmentMetrics> {
        if self.samples == 0 || self.segments.iter().all(|segment| segment.tokens == 0) {
            return None;
        }
        let count = self.segments.len();
        let grid_depth = self.grid_depth.max(count);
        let segments = self
            .segments
            .into_iter()
            .enumerate()
            .map(|(segment, total)| {
                let loss = if total.values > 0 {
                    total.squared_error / total.values as f64
                } else {
                    0.0
                };
                let cosine = if total.left_norm > 0.0 && total.right_norm > 0.0 {
                    total.dot / (total.left_norm.sqrt() * total.right_norm.sqrt())
                } else {
                    0.0
                };
                TttTemporalSegmentMetric {
                    segment,
                    start_tubelet: (segment * grid_depth).div_ceil(count),
                    end_tubelet: ((segment + 1) * grid_depth).div_ceil(count),
                    tokens: total.tokens,
                    loss,
                    cosine,
                }
            })
            .collect::<Vec<_>>();
        let first = segments.first();
        let last = segments.last();
        let late_minus_early_loss = first.zip(last).and_then(|(first, last)| {
            (first.tokens > 0 && last.tokens > 0).then_some(last.loss - first.loss)
        });
        let late_minus_early_cosine = first.zip(last).and_then(|(first, last)| {
            (first.tokens > 0 && last.tokens > 0).then_some(last.cosine - first.cosine)
        });
        Some(TttTemporalSegmentMetrics {
            samples: self.samples,
            late_minus_early_loss,
            late_minus_early_cosine,
            segments,
        })
    }
}

fn primary_token_rows<B: Backend>(
    primary_mask: Option<&SparseMaskBatch<B>>,
    grid: TokenGridShape,
    batch_size: usize,
) -> Vec<Vec<usize>> {
    primary_mask
        .map(SparseMaskBatch::rows)
        .unwrap_or_else(|| (0..batch_size).map(|_| (0..grid.len()).collect()).collect())
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

fn diagnostic_loss_cosine<B: step::TttSparsePatchifyBackend>(
    _model: &VJepaTttModel<B>,
    config: &BurnJepaTrainConfig,
    student: crate::VJepaEncoderOutput<B>,
    teacher_tokens: step::TeacherTokenTargets<B>,
    masks: Option<&step::ResolvedTttMasks<B>>,
    rollout: TttRolloutKind,
    batch_size: usize,
    device: &B::Device,
) -> Result<(f64, f64)> {
    let primary_mask = masks.map(|masks| match rollout {
        TttRolloutKind::Dense | TttRolloutKind::SparseTarget => &masks.target,
        TttRolloutKind::SparseContext => &masks.context,
    });
    let loss = tensor_scalar(
        loss::primary_feature_loss(
            config,
            student.tokens.clone(),
            teacher_tokens.final_tokens.clone(),
            primary_mask,
            rollout,
            batch_size,
            device,
        )
        .detach(),
    )?;
    let cosine = loss::primary_cosine(
        student.tokens,
        teacher_tokens.final_tokens,
        primary_mask,
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

fn rollout_stream_key(metadata: &JepaSampleMetadata) -> String {
    metadata
        .clip_id
        .as_ref()
        .or(metadata.source.as_ref())
        .or(metadata.domain.as_ref())
        .cloned()
        .unwrap_or_else(|| "unknown".to_string())
}

fn rollout_scene_key(metadata: &JepaSampleMetadata) -> String {
    metadata
        .original_stream
        .as_ref()
        .or(metadata.clip_id.as_ref())
        .or(metadata.source.as_ref())
        .or(metadata.domain.as_ref())
        .cloned()
        .unwrap_or_else(|| "unknown".to_string())
}

fn is_consecutive_window(previous: Option<usize>, current: Option<usize>) -> bool {
    match (previous, current) {
        (Some(previous), Some(current)) => current > previous,
        (None, None) | (Some(_), None) | (None, Some(_)) => true,
    }
}
