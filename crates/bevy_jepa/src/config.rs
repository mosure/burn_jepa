use std::{fmt, path::PathBuf, str::FromStr};

use bevy::prelude::Resource;
use burn_jepa::AnyUpAttentionMode;
use serde::{Deserialize, Serialize};

pub const MIN_PIPELINE_IMAGE_SIZE: usize = 256;
pub const PIPELINE_IMAGE_SIZE_MULTIPLE: usize = 16;
pub const DEFAULT_IMAGE_SIZE: usize = 256;
pub const DEFAULT_CONTEXT_DENSITY: f32 = 1.0;
pub const DEFAULT_PATCH_DIFF_QUALITY: f32 = 0.85;
pub const DEFAULT_MIN_CONTEXT_DENSITY: f32 = 0.0;
pub const DEFAULT_BOOTSTRAP_CONTEXT_DENSITY: f32 = 1.0;
pub const DEFAULT_PATCH_DIFF_THRESHOLD: f32 = 1.0 - DEFAULT_PATCH_DIFF_QUALITY;
pub const DEFAULT_ANYUP_CHUNK_SIZE: usize = 16;
pub const DEFAULT_PCA_UPDATE_EVERY: u64 = 4;
pub const DEFAULT_HIGH_RES_PCA_EVERY: u64 = 8;
pub const DEFAULT_CAMERA_WIDTH: u32 = 640;
pub const DEFAULT_CAMERA_HEIGHT: u32 = 360;
pub const DEFAULT_CAMERA_FPS: u32 = 30;
pub const DEFAULT_TTT_MODEL_PATH: &str =
    "target/burn-jepa-production-final/stage1-stream-tbptt/ttt-model.mpk";
pub const DEFAULT_VJEPA21_CHECKPOINT_DIR: &str =
    "/home/mosure/.cache/burn_jepa/vjepa2_1_vitb_dist_vitG_384";
pub const DEFAULT_VJEPA21_CONFIG_PATH: &str =
    "/home/mosure/.cache/burn_jepa/vjepa2_1_vitb_dist_vitG_384/config.json";
pub const DEFAULT_VJEPA21_WEIGHTS_NAME: &str = "model.pt";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BevyJepaEncoderSource {
    #[default]
    TrainedTtt,
    BaseCheckpoint,
    TinyTest,
}

impl BevyJepaEncoderSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TrainedTtt => "trained-ttt",
            Self::BaseCheckpoint => "base-checkpoint",
            Self::TinyTest => "tiny-test",
        }
    }

    pub const fn valid_values() -> &'static [&'static str] {
        &[
            "trained-ttt",
            "ttt",
            "base-checkpoint",
            "base",
            "checkpoint",
            "tiny-test",
            "tiny",
            "test",
        ]
    }
}

impl fmt::Display for BevyJepaEncoderSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for BevyJepaEncoderSource {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "trained-ttt" | "ttt" | "trained" | "production" => Ok(Self::TrainedTtt),
            "base-checkpoint" | "base" | "checkpoint" | "vjepa" | "vjepa2.1" => {
                Ok(Self::BaseCheckpoint)
            }
            "tiny-test" | "tiny" | "test" | "synthetic" => Ok(Self::TinyTest),
            other => Err(format!(
                "unsupported JEPA encoder source `{other}`; expected one of {}",
                Self::valid_values().join(", ")
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BevyJepaEncodePath {
    #[default]
    Auto,
    DensePatchEmbed,
    SparsePatchify,
}

impl BevyJepaEncodePath {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::DensePatchEmbed => "dense-patch",
            Self::SparsePatchify => "sparse-patchify",
        }
    }

    pub const fn valid_values() -> &'static [&'static str] {
        &[
            "auto",
            "dense-patch",
            "dense-patch-embed",
            "dense",
            "sparse-patchify",
            "sparse",
            "flex-gmm",
        ]
    }
}

impl fmt::Display for BevyJepaEncodePath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for BevyJepaEncodePath {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "dense-patch" | "dense-patch-embed" | "dense" | "dense-patchify" => {
                Ok(Self::DensePatchEmbed)
            }
            "sparse-patchify" | "sparse" | "flex-gmm" | "flex_gmm" => Ok(Self::SparsePatchify),
            other => Err(format!(
                "unsupported JEPA encode path `{other}`; expected one of {}",
                Self::valid_values().join(", ")
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BevyJepaFrameSource {
    #[default]
    Camera,
    StaticImage,
    SyntheticLocalMotion,
}

impl BevyJepaFrameSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Camera => "camera",
            Self::StaticImage => "static",
            Self::SyntheticLocalMotion => "synthetic-local-motion",
        }
    }

    pub const fn valid_values() -> &'static [&'static str] {
        &[
            "camera",
            "webcam",
            "static",
            "image",
            "synthetic",
            "synthetic-local-motion",
            "local-motion",
        ]
    }
}

impl fmt::Display for BevyJepaFrameSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for BevyJepaFrameSource {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "camera" | "webcam" | "cam" | "live" => Ok(Self::Camera),
            "static" | "image" | "image-path" | "file" | "still" => Ok(Self::StaticImage),
            "synthetic"
            | "synthetic-local-motion"
            | "local-motion"
            | "local"
            | "subtle-motion"
            | "generated" => Ok(Self::SyntheticLocalMotion),
            other => Err(format!(
                "unsupported JEPA frame source `{other}`; expected one of {}",
                Self::valid_values().join(", ")
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BevyJepaMaskSource {
    Autogaze,
    #[default]
    PatchDiff,
}

impl BevyJepaMaskSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Autogaze => "autogaze",
            Self::PatchDiff => "patch-diff",
        }
    }

    pub const fn valid_values() -> &'static [&'static str] {
        &["autogaze", "auto-gaze", "patch-diff", "patchdiff", "diff"]
    }

    pub const fn next(self) -> Self {
        match self {
            Self::Autogaze => Self::PatchDiff,
            Self::PatchDiff => Self::PatchDiff,
        }
    }
}

impl fmt::Display for BevyJepaMaskSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for BevyJepaMaskSource {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "autogaze" | "auto-gaze" | "gaze" | "model" => Ok(Self::Autogaze),
            "patch-diff" | "patchdiff" | "diff" | "frame-diff" | "frame-difference" => {
                Ok(Self::PatchDiff)
            }
            other => Err(format!(
                "unsupported JEPA mask source `{other}`; expected one of {}",
                Self::valid_values().join(", ")
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BevyJepaDisplayTransfer {
    #[default]
    Gpu,
    Cpu,
}

impl BevyJepaDisplayTransfer {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
        }
    }
}

impl fmt::Display for BevyJepaDisplayTransfer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for BevyJepaDisplayTransfer {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "gpu" | "device" => Ok(Self::Gpu),
            "cpu" | "host" => Ok(Self::Cpu),
            other => Err(format!("unsupported display transfer `{other}`")),
        }
    }
}

#[derive(Clone, Debug, Resource, Serialize, Deserialize)]
#[serde(default)]
pub struct BevyJepaConfig {
    pub encoder_source: BevyJepaEncoderSource,
    pub encode_path: BevyJepaEncodePath,
    pub ttt_model_path: Option<PathBuf>,
    pub jepa_checkpoint_dir: Option<PathBuf>,
    pub jepa_config_path: Option<PathBuf>,
    pub jepa_weights_name: String,
    pub source: BevyJepaFrameSource,
    pub mask_source: BevyJepaMaskSource,
    pub display_transfer: BevyJepaDisplayTransfer,
    pub image_path: Option<PathBuf>,
    pub anyup_weights: Option<PathBuf>,
    pub anyup_attention_mode: AnyUpAttentionMode,
    pub image_size: usize,
    pub context_density: f32,
    pub min_context_density: f32,
    pub bootstrap_context_density: f32,
    pub patch_diff_threshold: f32,
    pub anyup_q_chunk_size: usize,
    pub pca_update_every: u64,
    pub high_res_pca_every: u64,
    pub show_metrics: bool,
    pub measure_stages: bool,
    pub sync_measurements: bool,
    pub camera_width: u32,
    pub camera_height: u32,
    pub camera_fps: u32,
}

impl Default for BevyJepaConfig {
    fn default() -> Self {
        Self {
            encoder_source: BevyJepaEncoderSource::TrainedTtt,
            encode_path: BevyJepaEncodePath::Auto,
            ttt_model_path: Some(PathBuf::from(DEFAULT_TTT_MODEL_PATH)),
            jepa_checkpoint_dir: Some(PathBuf::from(DEFAULT_VJEPA21_CHECKPOINT_DIR)),
            jepa_config_path: Some(PathBuf::from(DEFAULT_VJEPA21_CONFIG_PATH)),
            jepa_weights_name: DEFAULT_VJEPA21_WEIGHTS_NAME.to_string(),
            source: BevyJepaFrameSource::Camera,
            mask_source: BevyJepaMaskSource::PatchDiff,
            display_transfer: BevyJepaDisplayTransfer::Gpu,
            image_path: None,
            anyup_weights: None,
            anyup_attention_mode: AnyUpAttentionMode::EfficientLocal,
            image_size: DEFAULT_IMAGE_SIZE,
            context_density: DEFAULT_CONTEXT_DENSITY,
            min_context_density: DEFAULT_MIN_CONTEXT_DENSITY,
            bootstrap_context_density: DEFAULT_BOOTSTRAP_CONTEXT_DENSITY,
            patch_diff_threshold: DEFAULT_PATCH_DIFF_THRESHOLD,
            anyup_q_chunk_size: DEFAULT_ANYUP_CHUNK_SIZE,
            pca_update_every: DEFAULT_PCA_UPDATE_EVERY,
            high_res_pca_every: DEFAULT_HIGH_RES_PCA_EVERY,
            show_metrics: true,
            measure_stages: true,
            sync_measurements: false,
            camera_width: DEFAULT_CAMERA_WIDTH,
            camera_height: DEFAULT_CAMERA_HEIGHT,
            camera_fps: DEFAULT_CAMERA_FPS,
        }
    }
}

impl BevyJepaConfig {
    pub fn pipeline_image_size(&self) -> usize {
        self.image_size
            .max(MIN_PIPELINE_IMAGE_SIZE)
            .div_ceil(PIPELINE_IMAGE_SIZE_MULTIPLE)
            * PIPELINE_IMAGE_SIZE_MULTIPLE
    }

    pub fn context_tokens(&self, dense_tokens: usize) -> usize {
        let density = self.context_density.clamp(0.01, 1.0);
        ((dense_tokens as f32 * density).round() as usize).clamp(1, dense_tokens.max(1))
    }

    pub fn min_context_tokens(&self, dense_tokens: usize) -> usize {
        let density = self.min_context_density.clamp(0.0, 1.0);
        ((dense_tokens as f32 * density).ceil() as usize).clamp(1, dense_tokens.max(1))
    }

    pub fn bootstrap_context_tokens(&self, dense_tokens: usize) -> usize {
        let density = self.bootstrap_context_density.clamp(0.0, 1.0);
        ((dense_tokens as f32 * density).ceil() as usize).clamp(1, dense_tokens.max(1))
    }

    pub fn patch_diff_quality(&self) -> f32 {
        (1.0 - self.patch_diff_threshold).clamp(0.0, 1.0)
    }
}
