use crate::{
    LearningRateScheduleConfig, LearningRateScheduleStats, TttBackpropMode, TttMemoryUpdateSource,
    TttSupervisionMode, TttTargetMode,
};
use anyhow::{Context, Result};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TttStreamStepKind {
    Reset,
    Carried,
    Mixed,
}

#[derive(Clone, Debug, Serialize)]
pub struct TttStepMetric {
    pub step: usize,
    pub loss: f64,
    pub stream_step: Option<TttStreamStepKind>,
    pub effective_reset_interval_steps: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TttMemoryMetrics {
    pub layers: Vec<usize>,
    pub predictor_layers: Vec<usize>,
    pub embed_dim: usize,
    pub predictor_embed_dim: usize,
    pub batch_size: usize,
    pub chunk_tokens: usize,
    pub ttt_lr: f32,
    pub fast_weight_elements: usize,
    pub fast_weight_bytes_f32: usize,
    pub trainable_param_elements: usize,
    pub trainable_param_bytes_f32: usize,
    pub adam_state_bytes_f32: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct TttMaskMetrics {
    pub context_tokens: usize,
    pub target_tokens: usize,
    pub context_min_tokens: usize,
    pub context_max_tokens: usize,
    pub context_mean_tokens: f32,
    pub target_min_tokens: usize,
    pub target_max_tokens: usize,
    pub target_mean_tokens: f32,
    pub dense_tokens: usize,
    pub context_density: f32,
    pub target_density: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct TttBackpropMetrics {
    pub mode: TttBackpropMode,
    pub truncate_blocks: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TttStreamTrainingMetrics {
    pub enabled: bool,
    pub detach_between_steps: bool,
    pub reset_on_clip_change: bool,
    pub reset_on_non_monotonic_start: bool,
    pub reset_interval_steps: usize,
    pub curriculum_enabled: bool,
    pub curriculum_initial_reset_interval_steps: usize,
    pub curriculum_final_reset_interval_steps: usize,
    pub curriculum_warmup_steps: usize,
    pub final_effective_reset_interval_steps: usize,
    pub state_decay: f64,
    pub state_l2_weight: f64,
    pub update_l2_weight: f64,
    pub state_regularization_width: usize,
    pub active_streams: usize,
    pub max_active_streams: usize,
    pub packed_batches: usize,
    pub max_packed_batch_size: usize,
    pub carried_steps: usize,
    pub reset_steps: usize,
    pub optimizer_steps: Option<usize>,
    pub reset_optimizer_steps: Option<usize>,
    pub carried_optimizer_steps: Option<usize>,
    pub mixed_optimizer_steps: Option<usize>,
    pub detached_steps: usize,
    pub decayed_steps: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TttStageMetrics {
    pub data_ms: u128,
    pub prefetch_wait_ms: u128,
    pub host_to_device_ms: u128,
    pub mask_ms: u128,
    pub stream_state_ms: u128,
    pub teacher_forward_ms: u128,
    pub teacher_cache_key_ms: u128,
    pub teacher_cache_evictions: usize,
    pub student_forward_ms: u128,
    pub loss_ms: u128,
    pub loss_read_ms: u128,
    pub backward_ms: u128,
    pub optimizer_ms: u128,
    pub backward_optim_ms: u128,
    pub report_ms: u128,
    pub teacher_cache_hits: usize,
    pub teacher_cache_misses: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct TttTargetSupervisionMetrics {
    pub mode: TttTargetMode,
    pub memory_update: TttMemoryUpdateSource,
    pub supervision: TttSupervisionMode,
    pub hybrid_final_steps: usize,
    pub train_adapter_target: &'static str,
    pub deploy_adapter_target: &'static str,
    pub layer_alignment: &'static str,
    pub teacher_forced_eval: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct TttDomainEvalMetric {
    pub domain: String,
    pub samples: usize,
    pub loss: f64,
    pub cosine: f64,
    pub teacher_forced_loss: Option<f64>,
    pub teacher_forced_cosine: Option<f64>,
    pub teacher_forcing_loss_gap: Option<f64>,
    pub teacher_forcing_cosine_gap: Option<f64>,
    pub full_loss: Option<f64>,
    pub full_cosine: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TttLayerUtilizationMetric {
    pub encoder_layer: usize,
    pub ttt_layer: usize,
    pub samples: usize,
    pub hidden_rms: f64,
    pub memory_read_rms: f64,
    pub adapter_delta_rms: f64,
    pub adapter_delta_to_hidden: f64,
    pub fast_weight_rms: f64,
    pub fast_update_rms: f64,
    pub target_proj_param_rms: Option<f64>,
    pub temporal_conv_param_rms: f64,
    pub out_proj_param_rms: f64,
    pub target_proj_grad_rms: Option<f64>,
    pub temporal_conv_grad_rms: Option<f64>,
    pub out_proj_grad_rms: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TttUtilizationMetrics {
    pub samples: usize,
    pub layers: Vec<TttLayerUtilizationMetric>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TttTemporalDiagnosticMetrics {
    pub samples: usize,
    pub reset_each_frame_loss: Option<f64>,
    pub reset_each_frame_cosine: Option<f64>,
    pub reset_each_tubelet_loss: Option<f64>,
    pub reset_each_tubelet_cosine: Option<f64>,
    pub reverse_order_loss: Option<f64>,
    pub reverse_order_cosine: Option<f64>,
    pub shuffle_order_loss: Option<f64>,
    pub shuffle_order_cosine: Option<f64>,
    pub freeze_fast_update_loss: Option<f64>,
    pub freeze_fast_update_cosine: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TttTemporalSegmentMetric {
    pub segment: usize,
    pub start_tubelet: usize,
    pub end_tubelet: usize,
    pub tokens: usize,
    pub loss: f64,
    pub cosine: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct TttTemporalSegmentMetrics {
    pub samples: usize,
    pub segments: Vec<TttTemporalSegmentMetric>,
    pub late_minus_early_loss: Option<f64>,
    pub late_minus_early_cosine: Option<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TttRolloutReportMode {
    Dense,
    SparseContext,
    SparseTarget,
}

#[derive(Clone, Debug, Serialize)]
pub struct TttRolloutMetrics {
    pub mode: TttRolloutReportMode,
    pub dense_tokens: usize,
    pub student_tokens: usize,
    pub student_token_density: f32,
    pub full_grid_eval: bool,
    pub autodiff_sparse_patchify: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct TttDenseSampleMetrics {
    pub enabled: bool,
    pub warmup_steps: usize,
    pub interval_steps: usize,
    pub dense_steps: usize,
    pub sparse_steps: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct TttTrainingReport {
    pub steps: usize,
    pub samples: usize,
    pub initial_loss: f64,
    pub best_loss: f64,
    pub final_loss: f64,
    pub loss_trace: Vec<TttStepMetric>,
    pub memory: TttMemoryMetrics,
    pub mask: Option<TttMaskMetrics>,
    pub rollout: TttRolloutMetrics,
    pub dense_samples: TttDenseSampleMetrics,
    pub backprop: TttBackpropMetrics,
    pub stream: TttStreamTrainingMetrics,
    pub lr_schedule: LearningRateScheduleConfig,
    pub lr_stats: LearningRateScheduleStats,
    pub target_supervision: TttTargetSupervisionMetrics,
    pub pre_train_eval_loss: Option<f64>,
    pub pre_train_eval_feature_loss: Option<f64>,
    pub pre_train_eval_predictor_loss: Option<f64>,
    pub pre_train_eval_cosine: Option<f64>,
    pub pre_train_teacher_forced_eval_loss: Option<f64>,
    pub pre_train_teacher_forced_eval_cosine: Option<f64>,
    pub pre_train_teacher_forcing_loss_gap: Option<f64>,
    pub pre_train_teacher_forcing_cosine_gap: Option<f64>,
    pub pre_train_full_eval_loss: Option<f64>,
    pub pre_train_full_eval_cosine: Option<f64>,
    pub eval_loss: Option<f64>,
    pub eval_feature_loss: Option<f64>,
    pub eval_predictor_loss: Option<f64>,
    pub eval_cosine: Option<f64>,
    pub teacher_forced_eval_loss: Option<f64>,
    pub teacher_forced_eval_cosine: Option<f64>,
    pub teacher_forcing_loss_gap: Option<f64>,
    pub teacher_forcing_cosine_gap: Option<f64>,
    pub eval_full_loss: Option<f64>,
    pub eval_full_cosine: Option<f64>,
    pub eval_samples: usize,
    pub train_stage: TttStageMetrics,
    pub eval_stage: TttStageMetrics,
    pub eval_domains: Vec<TttDomainEvalMetric>,
    pub utilization: Option<TttUtilizationMetrics>,
    pub temporal_diagnostics: Option<TttTemporalDiagnosticMetrics>,
    pub temporal_segments: Option<TttTemporalSegmentMetrics>,
    pub train_elapsed_ms: u128,
    pub eval_elapsed_ms: u128,
    pub elapsed_ms: u128,
    pub samples_per_second: f64,
    pub model_path: Option<PathBuf>,
    pub report_path: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
pub struct TttEvalReport {
    pub model_path: PathBuf,
    pub eval_steps: usize,
    pub eval_samples: usize,
    pub loss: f64,
    pub feature_loss: f64,
    pub predictor_loss: Option<f64>,
    pub cosine: f64,
    pub teacher_forced_loss: Option<f64>,
    pub teacher_forced_cosine: Option<f64>,
    pub teacher_forcing_loss_gap: Option<f64>,
    pub teacher_forcing_cosine_gap: Option<f64>,
    pub full_loss: Option<f64>,
    pub full_cosine: Option<f64>,
    pub memory: TttMemoryMetrics,
    pub mask: Option<TttMaskMetrics>,
    pub rollout: TttRolloutMetrics,
    pub target_supervision: TttTargetSupervisionMetrics,
    pub stage: TttStageMetrics,
    pub domains: Vec<TttDomainEvalMetric>,
    pub utilization: Option<TttUtilizationMetrics>,
    pub temporal_diagnostics: Option<TttTemporalDiagnosticMetrics>,
    pub temporal_segments: Option<TttTemporalSegmentMetrics>,
    pub stream: TttStreamTrainingMetrics,
    pub elapsed_ms: u128,
    pub samples_per_second: f64,
    pub report_path: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
pub struct DenseJepaTrainingReport {
    pub steps: usize,
    pub samples: usize,
    pub final_loss: f64,
    pub elapsed_ms: u128,
    pub samples_per_second: f64,
    pub model_path: Option<PathBuf>,
    pub report_path: PathBuf,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TrainingLossSummary {
    pub initial: Option<f64>,
    pub best: Option<f64>,
    pub final_loss: f64,
}

impl TrainingLossSummary {
    pub(super) fn dense(final_loss: f64) -> Self {
        Self {
            initial: None,
            best: None,
            final_loss,
        }
    }

    pub(super) fn ttt(initial: Option<f64>, best: f64, final_loss: f64) -> Self {
        Self {
            initial,
            best: Some(best),
            final_loss,
        }
    }
}

pub(super) fn tensor_scalar<B: Backend>(tensor: Tensor<B, 1>) -> Result<f64> {
    let values = tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .context("read scalar tensor")?;
    Ok(values.first().copied().unwrap_or_default() as f64)
}

pub(super) fn samples_per_second(samples: usize, elapsed_ms: u128) -> f64 {
    if elapsed_ms == 0 {
        samples as f64
    } else {
        samples as f64 / (elapsed_ms as f64 / 1000.0)
    }
}

pub(super) fn save_training_report(
    output_dir: &Path,
    name: &str,
    steps: usize,
    samples: usize,
    loss: TrainingLossSummary,
    elapsed_ms: u128,
    model_path: Option<PathBuf>,
) -> Result<PathBuf> {
    #[derive(Serialize)]
    struct Report<'a> {
        steps: usize,
        samples: usize,
        initial_loss: Option<f64>,
        best_loss: Option<f64>,
        final_loss: f64,
        eval_loss: Option<f64>,
        eval_cosine: Option<f64>,
        eval_samples: usize,
        elapsed_ms: u128,
        samples_per_second: f64,
        model_path: Option<&'a Path>,
    }

    let report = Report {
        steps,
        samples,
        initial_loss: loss.initial,
        best_loss: loss.best,
        final_loss: loss.final_loss,
        eval_loss: None,
        eval_cosine: None,
        eval_samples: 0,
        elapsed_ms,
        samples_per_second: samples_per_second(samples, elapsed_ms),
        model_path: model_path.as_deref(),
    };
    let path = output_dir.join(name);
    fs::write(&path, serde_json::to_string_pretty(&report)?)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

pub(super) fn save_ttt_training_report(
    output_dir: &Path,
    name: &str,
    report: &TttTrainingReport,
) -> Result<PathBuf> {
    let path = output_dir.join(name);
    fs::write(&path, serde_json::to_string_pretty(report)?)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}
