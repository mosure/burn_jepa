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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JepaTrainBackend {
    NdArray,
    Wgpu,
    WebGpu,
    Cuda,
}

impl Default for JepaTrainBackend {
    fn default() -> Self {
        Self::NdArray
    }
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
        ensure!(
            !self.ttt.layers.is_empty(),
            "train-ttt requires at least one TTT layer"
        );
        Ok(())
    }

    pub fn validate_common(&self) -> Result<()> {
        ensure!(
            self.training.max_steps > 0,
            "training.max_steps must be nonzero"
        );
        ensure!(
            self.training.batch_size > 0,
            "training.batch_size must be nonzero"
        );
        self.training.validate_mask_config()?;
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
        self.ttt.validate(&self.model_config_for_validation()?)?;
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
    pub batch_size: usize,
    pub max_steps: usize,
    pub learning_rate: f64,
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
    pub cache_teacher_tokens: bool,
    pub save_steps: usize,
}

impl Default for TrainingLoopConfig {
    fn default() -> Self {
        Self {
            backend: JepaTrainBackend::NdArray,
            batch_size: 1,
            max_steps: 1,
            learning_rate: 1.0e-3,
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
            cache_teacher_tokens: false,
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
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingBatchingMode {
    #[default]
    Sequential,
    GroupUniformMasks,
    FixedWidthMasks,
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
            !matches!(self, Self::ContextMask | Self::TargetMask) || predictor_loss_weight <= 0.0,
            "training.sparse_rollout context_mask/target_mask is incompatible with predictor_loss_weight > 0"
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
