use super::mask::TrainingMaskConfig;
use crate::{
    JepaDatasetConfig, JepaSampleMetadata, SparseTokenMask, TokenGridShape, TttEncoderConfig,
    VJepaConfig, VJepaLoadOptions, load_config_from_hf_dir, video_token_grid,
};
use anyhow::{Context, Result, ensure};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JepaTrainBackend {
    #[default]
    NdArray,
    Flex,
    Wgpu,
    WebGpu,
    Cuda,
    Dispatch,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JepaDispatchBackend {
    #[default]
    Auto,
    NdArray,
    Flex,
    Wgpu,
    WebGpu,
    Cuda,
}

fn is_default_dispatch_backend(backend: &JepaDispatchBackend) -> bool {
    *backend == JepaDispatchBackend::Auto
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct BurnJepaTrainConfig {
    pub model: TrainModelConfig,
    pub dataset: JepaDatasetConfig,
    pub ttt: TttEncoderConfig,
    pub training: TrainingLoopConfig,
    pub loss: TttDistillationConfig,
}

impl Default for BurnJepaTrainConfig {
    fn default() -> Self {
        Self {
            model: TrainModelConfig::default(),
            dataset: JepaDatasetConfig::default(),
            ttt: TttEncoderConfig {
                chunk_tokens: 2,
                ..TttEncoderConfig::default()
            },
            training: TrainingLoopConfig::default(),
            loss: TttDistillationConfig::default(),
        }
    }
}

impl BurnJepaTrainConfig {
    pub fn from_toml_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .with_context(|| format!("read train config {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parse train config {}", path.display()))
    }

    pub fn to_toml_string(&self) -> Result<String> {
        toml::to_string_pretty(self).context("serialize train config")
    }

    pub fn validate_for_ttt(&self) -> Result<()> {
        self.validate_common()?;
        let model_config = self.model_config_for_validation()?;
        let encoder_layers = self.ttt.resolved_layers(&model_config);
        let predictor_layers = self.ttt.resolved_predictor_layers(&model_config);
        ensure!(
            !encoder_layers.is_empty() || !predictor_layers.is_empty(),
            "train-ttt requires at least one TTT layer"
        );
        ensure!(
            predictor_layers.is_empty() || self.loss.predictor_loss_weight > 0.0,
            "ttt.predictor_layers require loss.predictor_loss_weight > 0"
        );
        Ok(())
    }

    pub fn validate_common(&self) -> Result<()> {
        let model_config = self.model_config_for_validation()?;
        ensure!(
            self.training.max_steps > 0,
            "training.max_steps must be nonzero"
        );
        ensure!(
            self.training.batch_size > 0,
            "training.batch_size must be nonzero"
        );
        ensure!(
            self.training.learning_rate.is_finite() && self.training.learning_rate >= 0.0,
            "training.learning_rate must be finite and non-negative"
        );
        self.training
            .lr_schedule
            .validate(self.training.max_steps, self.training.learning_rate)?;
        self.training.validate_mask_config()?;
        self.training.validate_stream_config()?;
        ensure!(
            !self.training.prefetch_batches
                || self.dataset.kind == crate::JepaDatasetKind::Manifest,
            "training.prefetch_batches currently requires a manifest dataset"
        );
        ensure!(
            self.loss.feature_loss_weight > 0.0 || self.loss.predictor_loss_weight > 0.0,
            "at least one loss weight must be positive"
        );
        self.training.sparse_rollout.validate(
            self.training.mask.is_some(),
            self.loss.predictor_loss_weight,
        )?;
        self.training.sparse_patchify_training.validate(
            self.training.sparse_rollout,
            self.training.mask.is_some(),
            self.loss.predictor_loss_weight,
            self.ttt.freeze_pretrained,
        )?;
        ensure!(
            !self.training.dense_samples.enabled || self.loss.predictor_loss_weight <= 0.0,
            "training.dense_samples currently requires loss.predictor_loss_weight=0"
        );
        self.ttt.validate(&model_config)?;
        Ok(())
    }

    fn model_config_for_validation(&self) -> Result<VJepaConfig> {
        if let Some(config_path) = &self.model.config_path {
            VJepaConfig::from_json_file(config_path)
        } else if let Some(checkpoint_dir) = &self.model.checkpoint_dir {
            load_config_from_hf_dir(checkpoint_dir, &VJepaLoadOptions::default().config_name)
        } else {
            Ok(VJepaConfig::tiny_for_tests())
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainModelConfig {
    pub checkpoint_dir: Option<PathBuf>,
    pub teacher_checkpoint_dir: Option<PathBuf>,
    pub ttt_checkpoint_path: Option<PathBuf>,
    pub config_path: Option<PathBuf>,
    pub weights_name: Option<String>,
    pub output_dir: PathBuf,
    pub save_model: bool,
}

impl Default for TrainModelConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: None,
            teacher_checkpoint_dir: None,
            ttt_checkpoint_path: None,
            config_path: None,
            weights_name: None,
            output_dir: PathBuf::from("target/burn-jepa-train"),
            save_model: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingLoopConfig {
    pub backend: JepaTrainBackend,
    #[serde(default, skip_serializing_if = "is_default_dispatch_backend")]
    pub dispatch_backend: JepaDispatchBackend,
    pub batch_size: usize,
    pub max_steps: usize,
    pub learning_rate: f64,
    pub lr_schedule: LearningRateScheduleConfig,
    pub weight_decay: f32,
    pub context_keep_ratio: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<TrainingMaskConfig>,
    pub sparse_rollout: TttSparseRolloutMode,
    pub sparse_patchify_training: TttSparsePatchifyTrainingMode,
    pub batching: TrainingBatchingMode,
    pub loss_trace_interval: usize,
    pub eval_steps: usize,
    pub eval_batch_size: Option<usize>,
    pub eval_full_grid: bool,
    pub eval_utilization_diagnostics: bool,
    pub eval_temporal_diagnostics: bool,
    pub cache_teacher_tokens: bool,
    pub teacher_cache_max_entries: usize,
    pub prefetch_batches: bool,
    pub dense_samples: TttDenseSampleTrainingConfig,
    pub stream: TttStreamTrainingConfig,
    pub save_steps: usize,
}

impl Default for TrainingLoopConfig {
    fn default() -> Self {
        Self {
            backend: JepaTrainBackend::NdArray,
            dispatch_backend: JepaDispatchBackend::Auto,
            batch_size: 1,
            max_steps: 1,
            learning_rate: 1.0e-3,
            lr_schedule: LearningRateScheduleConfig::default(),
            weight_decay: 0.0,
            context_keep_ratio: 0.75,
            mask: None,
            sparse_rollout: TttSparseRolloutMode::Auto,
            sparse_patchify_training: TttSparsePatchifyTrainingMode::Auto,
            batching: TrainingBatchingMode::Sequential,
            loss_trace_interval: 1,
            eval_steps: 0,
            eval_batch_size: None,
            eval_full_grid: true,
            eval_utilization_diagnostics: false,
            eval_temporal_diagnostics: false,
            cache_teacher_tokens: false,
            teacher_cache_max_entries: 32,
            prefetch_batches: false,
            dense_samples: TttDenseSampleTrainingConfig::default(),
            stream: TttStreamTrainingConfig::default(),
            save_steps: 0,
        }
    }
}

impl TrainingLoopConfig {
    pub fn effective_eval_batch_size(&self) -> usize {
        self.eval_batch_size.unwrap_or(self.batch_size).max(1)
    }

    pub fn mask_config(&self) -> TrainingMaskConfig {
        self.mask.clone().unwrap_or(TrainingMaskConfig::KeepRatio {
            context_keep_ratio: self.context_keep_ratio,
        })
    }

    pub fn validate_mask_config(&self) -> Result<()> {
        self.mask_config().validate()
    }

    pub fn validate_stream_config(&self) -> Result<()> {
        self.dense_samples.validate()?;
        self.stream.validate(
            self.batch_size,
            self.effective_eval_batch_size(),
            self.eval_steps,
            self.batching,
        )
    }

    pub fn resolve_masks<B: Backend>(
        &self,
        video: &Tensor<B, 5>,
        model_config: &VJepaConfig,
    ) -> Result<(SparseTokenMask, SparseTokenMask)> {
        let [_, _, frames, height, width] = video.shape().dims::<5>();
        let grid = video_token_grid(model_config, frames, height, width)?;
        self.resolve_masks_for_grid(video, model_config, grid)
    }

    pub fn resolve_masks_with_metadata<B: Backend>(
        &self,
        video: &Tensor<B, 5>,
        model_config: &VJepaConfig,
        metadata: &[JepaSampleMetadata],
    ) -> Result<(SparseTokenMask, SparseTokenMask)> {
        let [_, _, frames, height, width] = video.shape().dims::<5>();
        let grid = video_token_grid(model_config, frames, height, width)?;
        self.resolve_masks_for_grid_with_metadata(video, model_config, grid, metadata)
    }

    pub fn resolve_masks_for_grid<B: Backend>(
        &self,
        video: &Tensor<B, 5>,
        model_config: &VJepaConfig,
        grid: TokenGridShape,
    ) -> Result<(SparseTokenMask, SparseTokenMask)> {
        self.mask_config()
            .resolve_masks(video, model_config, grid)
            .context("resolve training mask config")
    }

    pub fn resolve_masks_for_grid_with_metadata<B: Backend>(
        &self,
        video: &Tensor<B, 5>,
        model_config: &VJepaConfig,
        grid: TokenGridShape,
        metadata: &[JepaSampleMetadata],
    ) -> Result<(SparseTokenMask, SparseTokenMask)> {
        self.mask_config()
            .resolve_masks_with_metadata(video, model_config, grid, metadata)
            .context("resolve training mask config")
    }

    pub fn use_sparse_rollout(&self, predictor_loss_weight: f32) -> bool {
        self.sparse_rollout
            .uses_sparse_mask(self.mask.is_some(), predictor_loss_weight)
    }

    pub fn learning_rate_for_step(&self, step_index: usize) -> f64 {
        self.lr_schedule
            .learning_rate(self.learning_rate, step_index, self.max_steps)
    }

    pub fn learning_rate_stats(&self) -> LearningRateScheduleStats {
        let mut first = None;
        let mut final_lr = 0.0;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for step in 0..self.max_steps.max(1) {
            let lr = self.learning_rate_for_step(step);
            first.get_or_insert(lr);
            final_lr = lr;
            min = min.min(lr);
            max = max.max(lr);
        }
        LearningRateScheduleStats {
            base_learning_rate: self.learning_rate,
            first_learning_rate: first.unwrap_or(self.learning_rate),
            final_learning_rate: final_lr,
            min_learning_rate: min,
            max_learning_rate: max,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TttDenseSampleTrainingConfig {
    pub enabled: bool,
    pub warmup_steps: usize,
    pub interval_steps: usize,
}

impl TttDenseSampleTrainingConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.enabled || self.warmup_steps > 0 || self.interval_steps > 0,
            "training.dense_samples requires warmup_steps or interval_steps when enabled"
        );
        Ok(())
    }

    pub fn uses_dense_step(&self, step_index: usize) -> bool {
        if !self.enabled {
            return false;
        }
        step_index < self.warmup_steps
            || (self.interval_steps > 0 && step_index.is_multiple_of(self.interval_steps))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TttStreamTrainingConfig {
    pub enabled: bool,
    pub detach_between_steps: bool,
    pub reset_on_clip_change: bool,
    pub reset_on_non_monotonic_start: bool,
    pub reset_interval_steps: usize,
    pub state_decay: f64,
    pub state_l2_weight: f64,
    pub update_l2_weight: f64,
    pub state_regularization_width: usize,
    pub curriculum: TttSequenceCurriculumConfig,
}

impl Default for TttStreamTrainingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            detach_between_steps: true,
            reset_on_clip_change: true,
            reset_on_non_monotonic_start: true,
            reset_interval_steps: 0,
            state_decay: 1.0,
            state_l2_weight: 0.0,
            update_l2_weight: 0.0,
            state_regularization_width: 0,
            curriculum: TttSequenceCurriculumConfig::default(),
        }
    }
}

impl TttStreamTrainingConfig {
    pub fn validate(
        &self,
        _batch_size: usize,
        _eval_batch_size: usize,
        _eval_steps: usize,
        _batching: TrainingBatchingMode,
    ) -> Result<()> {
        ensure!(
            self.state_decay.is_finite() && self.state_decay >= 0.0 && self.state_decay <= 1.0,
            "training.stream.state_decay must be finite and in [0, 1]"
        );
        ensure!(
            self.state_l2_weight.is_finite() && self.state_l2_weight >= 0.0,
            "training.stream.state_l2_weight must be finite and non-negative"
        );
        ensure!(
            self.update_l2_weight.is_finite() && self.update_l2_weight >= 0.0,
            "training.stream.update_l2_weight must be finite and non-negative"
        );
        self.curriculum.validate()?;
        Ok(())
    }

    pub fn reset_interval_for_step(&self, step_index: usize) -> usize {
        self.curriculum
            .reset_interval_for_step(step_index)
            .unwrap_or(self.reset_interval_steps)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TttSequenceCurriculumConfig {
    pub enabled: bool,
    pub initial_reset_interval_steps: usize,
    pub final_reset_interval_steps: usize,
    pub warmup_steps: usize,
}

impl Default for TttSequenceCurriculumConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            initial_reset_interval_steps: 1,
            final_reset_interval_steps: 1,
            warmup_steps: 1,
        }
    }
}

impl TttSequenceCurriculumConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        ensure!(
            self.initial_reset_interval_steps > 0,
            "training.stream.curriculum.initial_reset_interval_steps must be positive"
        );
        ensure!(
            self.final_reset_interval_steps >= self.initial_reset_interval_steps,
            "training.stream.curriculum.final_reset_interval_steps must be >= initial_reset_interval_steps"
        );
        ensure!(
            self.warmup_steps > 0,
            "training.stream.curriculum.warmup_steps must be positive"
        );
        Ok(())
    }

    pub fn reset_interval_for_step(&self, step_index: usize) -> Option<usize> {
        if !self.enabled {
            return None;
        }
        let progress = ((step_index + 1).min(self.warmup_steps) as f64) / self.warmup_steps as f64;
        let span = self.final_reset_interval_steps - self.initial_reset_interval_steps;
        Some(self.initial_reset_interval_steps + (span as f64 * progress).round() as usize)
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct LearningRateScheduleStats {
    pub base_learning_rate: f64,
    pub first_learning_rate: f64,
    pub final_learning_rate: f64,
    pub min_learning_rate: f64,
    pub max_learning_rate: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LearningRateScheduleConfig {
    #[default]
    Constant,
    LinearWarmupCosine {
        #[serde(default)]
        warmup_steps: usize,
        min_learning_rate: f64,
    },
    StepDecay {
        decay_steps: Vec<usize>,
        decay_factor: f64,
        #[serde(default)]
        min_learning_rate: f64,
    },
}

impl LearningRateScheduleConfig {
    pub fn validate(&self, max_steps: usize, base_learning_rate: f64) -> Result<()> {
        match self {
            Self::Constant => {}
            Self::LinearWarmupCosine {
                warmup_steps,
                min_learning_rate,
            } => {
                ensure!(
                    *warmup_steps <= max_steps,
                    "training.lr_schedule.linear_warmup_cosine warmup_steps must be <= max_steps"
                );
                ensure!(
                    min_learning_rate.is_finite()
                        && *min_learning_rate >= 0.0
                        && *min_learning_rate <= base_learning_rate,
                    "training.lr_schedule.linear_warmup_cosine min_learning_rate must be finite and in [0, learning_rate]"
                );
            }
            Self::StepDecay {
                decay_steps,
                decay_factor,
                min_learning_rate,
            } => {
                ensure!(
                    decay_factor.is_finite() && *decay_factor > 0.0 && *decay_factor < 1.0,
                    "training.lr_schedule.step_decay decay_factor must be finite and in (0, 1)"
                );
                ensure!(
                    min_learning_rate.is_finite()
                        && *min_learning_rate >= 0.0
                        && *min_learning_rate <= base_learning_rate,
                    "training.lr_schedule.step_decay min_learning_rate must be finite and in [0, learning_rate]"
                );
                ensure!(
                    decay_steps
                        .iter()
                        .all(|step| *step > 0 && *step <= max_steps),
                    "training.lr_schedule.step_decay decay_steps must be in 1..=max_steps"
                );
                ensure!(
                    decay_steps.windows(2).all(|window| window[0] < window[1]),
                    "training.lr_schedule.step_decay decay_steps must be strictly increasing"
                );
            }
        }
        Ok(())
    }

    pub fn learning_rate(
        &self,
        base_learning_rate: f64,
        step_index: usize,
        max_steps: usize,
    ) -> f64 {
        match self {
            Self::Constant => base_learning_rate,
            Self::LinearWarmupCosine {
                warmup_steps,
                min_learning_rate,
            } => linear_warmup_cosine_lr(
                base_learning_rate,
                *min_learning_rate,
                *warmup_steps,
                step_index,
                max_steps,
            ),
            Self::StepDecay {
                decay_steps,
                decay_factor,
                min_learning_rate,
            } => {
                let step_number = step_index + 1;
                let decays = decay_steps
                    .iter()
                    .filter(|&&decay_step| step_number > decay_step)
                    .count() as i32;
                (base_learning_rate * decay_factor.powi(decays)).max(*min_learning_rate)
            }
        }
    }

    pub fn clamped_to_max_steps(&self, max_steps: usize) -> Self {
        match self {
            Self::Constant => Self::Constant,
            Self::LinearWarmupCosine {
                warmup_steps,
                min_learning_rate,
            } => Self::LinearWarmupCosine {
                warmup_steps: (*warmup_steps).min(max_steps),
                min_learning_rate: *min_learning_rate,
            },
            Self::StepDecay {
                decay_steps,
                decay_factor,
                min_learning_rate,
            } => Self::StepDecay {
                decay_steps: decay_steps
                    .iter()
                    .copied()
                    .filter(|step| *step <= max_steps)
                    .collect(),
                decay_factor: *decay_factor,
                min_learning_rate: *min_learning_rate,
            },
        }
    }
}

fn linear_warmup_cosine_lr(
    base_learning_rate: f64,
    min_learning_rate: f64,
    warmup_steps: usize,
    step_index: usize,
    max_steps: usize,
) -> f64 {
    if warmup_steps > 0 && step_index < warmup_steps {
        return base_learning_rate * (step_index + 1) as f64 / warmup_steps as f64;
    }
    let decay_steps = max_steps.saturating_sub(warmup_steps).max(1);
    let decay_index = step_index.saturating_sub(warmup_steps).min(decay_steps);
    let progress = if decay_steps == 1 {
        1.0
    } else {
        decay_index as f64 / (decay_steps - 1) as f64
    };
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
    min_learning_rate + (base_learning_rate - min_learning_rate) * cosine
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingBatchingMode {
    #[default]
    Sequential,
    GroupUniformMasks,
    FixedWidthMasks,
    PackedStreams,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TttSparseRolloutMode {
    #[default]
    Auto,
    Dense,
    ContextMask,
    TargetMask,
}

impl TttSparseRolloutMode {
    pub fn validate(self, mask_configured: bool, predictor_loss_weight: f32) -> Result<()> {
        ensure!(
            self != Self::TargetMask || predictor_loss_weight <= 0.0,
            "training.sparse_rollout target_mask is incompatible with predictor_loss_weight > 0"
        );
        ensure!(
            !matches!(self, Self::ContextMask | Self::TargetMask) || mask_configured,
            "training.sparse_rollout context_mask/target_mask requires training.mask"
        );
        Ok(())
    }

    pub fn uses_sparse_mask(self, mask_configured: bool, predictor_loss_weight: f32) -> bool {
        match self {
            Self::Dense => false,
            Self::Auto => mask_configured && predictor_loss_weight <= 0.0,
            Self::ContextMask | Self::TargetMask => true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TttSparsePatchifyTrainingMode {
    #[default]
    Auto,
    DensePatchEmbed,
    FrozenSparsePatchify,
}

impl TttSparsePatchifyTrainingMode {
    pub fn validate(
        self,
        sparse_rollout: TttSparseRolloutMode,
        mask_configured: bool,
        predictor_loss_weight: f32,
        freeze_pretrained: bool,
    ) -> Result<()> {
        if self == Self::FrozenSparsePatchify {
            ensure!(
                sparse_rollout.uses_sparse_mask(mask_configured, predictor_loss_weight),
                "training.sparse_patchify_training=frozen_sparse_patchify requires sparse rollout"
            );
            ensure!(
                freeze_pretrained,
                "training.sparse_patchify_training=frozen_sparse_patchify requires ttt.freeze_pretrained=true"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TttDistillationConfig {
    pub feature_loss_weight: f32,
    pub predictor_loss_weight: f32,
}

impl Default for TttDistillationConfig {
    fn default() -> Self {
        Self {
            feature_loss_weight: 1.0,
            predictor_loss_weight: 0.0,
        }
    }
}
