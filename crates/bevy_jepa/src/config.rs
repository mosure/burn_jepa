use std::{
    fmt,
    ops::{Deref, DerefMut},
    path::PathBuf,
    str::FromStr,
};

use bevy::prelude::Resource;
use burn_jepa::{
    AnyUpAttentionMode, BurnAnyUpModelProfile, BurnJepaModelProfile, FeatureFrameEncodeRoute,
    FeatureFrameSparseEncodeMode,
};
use serde::{Deserialize, Serialize};

pub use burn_jepa::{
    DEFAULT_ANYUP_CHUNK_SIZE, DEFAULT_BOOTSTRAP_CONTEXT_DENSITY,
    DEFAULT_BURN_ANYUP_CHECKPOINT_PATH, DEFAULT_CONTEXT_DENSITY, DEFAULT_HIGH_RES_PCA_EVERY,
    DEFAULT_IMAGE_SIZE, DEFAULT_MIN_CONTEXT_DENSITY,
    DEFAULT_PATCH_DIFF_AGE_REFRESH_INTERVAL_FRAMES, DEFAULT_PATCH_DIFF_AGE_REFRESH_MAX_DENSITY,
    DEFAULT_PATCH_DIFF_BLUE_NOISE_REFRESH_DENSITY, DEFAULT_PATCH_DIFF_DENSE_FALLBACK_DENSITY,
    DEFAULT_PATCH_DIFF_QUALITY, DEFAULT_PATCH_DIFF_REFRESH_ENABLED,
    DEFAULT_PATCH_DIFF_REFRESH_MAX_DENSITY, DEFAULT_PATCH_DIFF_SUBTHRESHOLD_DECAY,
    DEFAULT_PATCH_DIFF_SUBTHRESHOLD_MAX_DENSITY, DEFAULT_PATCH_DIFF_SUBTHRESHOLD_TRIGGER,
    DEFAULT_PATCH_DIFF_THRESHOLD, DEFAULT_PCA_MIN_SAMPLE_FRAMES, DEFAULT_PCA_SAMPLE_WINDOW_FRAMES,
    DEFAULT_PCA_UPDATE_EVERY, DEFAULT_PCA_UPDATE_ITERATIONS, DEFAULT_PREWARM_SHAPE_BUCKETS,
    DEFAULT_SPARSE_MASK_BUCKET_TOKENS, FeatureFrameViewerConfig, MIN_PIPELINE_IMAGE_SIZE,
    PIPELINE_IMAGE_SIZE_MULTIPLE, PatchDiffRefreshConfig,
};
pub const DEFAULT_ANYUP_CHECKPOINT_PATH: &str = DEFAULT_BURN_ANYUP_CHECKPOINT_PATH;
pub const DEFAULT_CAMERA_WIDTH: u32 = 640;
pub const DEFAULT_CAMERA_HEIGHT: u32 = 360;
pub const DEFAULT_CAMERA_FPS: u32 = 30;
pub const DEFAULT_MODEL_PACKAGE_DIR: &str = "target/burn-jepa-web/model";
pub const DEFAULT_MODEL_MANIFEST_PATH: &str =
    "target/burn-jepa-web/model/vjepa2_1_ttt/manifest.json";
pub const DEFAULT_ANYUP_PACKAGE_DIR: &str = "target/burn_anyup";
pub const DEFAULT_ANYUP_MODEL_MANIFEST_PATH: &str =
    "target/burn_anyup/anyup_multi_backbone/manifest.json";
pub const DEFAULT_TTT_MODEL_PATH: &str =
    "target/burn-jepa-production-final/stage1-stream-tbptt/ttt-model.mpk";
pub const DEFAULT_VJEPA21_CHECKPOINT_DIR: &str = "~/.cache/burn_jepa/vjepa2_1_vitb_dist_vitG_384";
pub const DEFAULT_VJEPA21_CONFIG_PATH: &str =
    "~/.cache/burn_jepa/vjepa2_1_vitb_dist_vitG_384/config.json";
pub const DEFAULT_VJEPA21_WEIGHTS_NAME: &str = "model.pt";

pub type BevyJepaEncodePath = FeatureFrameEncodeRoute;
pub type BevyJepaAnyUpModelPackageProfile = BurnAnyUpModelProfile;
pub type BevyJepaModelPackageProfile = BurnJepaModelProfile;
pub type BevyJepaSparseEncodeMode = FeatureFrameSparseEncodeMode;

pub fn default_model_manifest_path_for_profile(profile: BevyJepaModelPackageProfile) -> PathBuf {
    PathBuf::from(DEFAULT_MODEL_PACKAGE_DIR)
        .join(profile.as_str())
        .join("manifest.json")
}

pub fn default_anyup_model_manifest_path_for_profile(
    profile: BevyJepaAnyUpModelPackageProfile,
) -> PathBuf {
    PathBuf::from(DEFAULT_ANYUP_PACKAGE_DIR)
        .join(profile.as_str())
        .join("manifest.json")
}

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
    pub model_manifest_path: Option<PathBuf>,
    pub model_cache_dir: Option<PathBuf>,
    pub model_profile: BevyJepaModelPackageProfile,
    pub model_base_url: String,
    pub model_auto_download: bool,
    pub ttt_model_path: Option<PathBuf>,
    pub jepa_checkpoint_dir: Option<PathBuf>,
    pub jepa_config_path: Option<PathBuf>,
    pub jepa_weights_name: String,
    pub source: BevyJepaFrameSource,
    pub mask_source: BevyJepaMaskSource,
    pub display_transfer: BevyJepaDisplayTransfer,
    pub image_path: Option<PathBuf>,
    pub anyup_weights: Option<PathBuf>,
    pub anyup_model_manifest_path: Option<PathBuf>,
    pub anyup_model_cache_dir: Option<PathBuf>,
    pub anyup_model_profile: BevyJepaAnyUpModelPackageProfile,
    pub anyup_model_base_url: String,
    pub anyup_model_auto_download: bool,
    pub anyup_attention_mode: AnyUpAttentionMode,
    #[serde(flatten)]
    pub pipeline: FeatureFrameViewerConfig,
    pub show_metrics: bool,
    pub camera_width: u32,
    pub camera_height: u32,
    pub camera_fps: u32,
}

impl Default for BevyJepaConfig {
    fn default() -> Self {
        Self {
            encoder_source: BevyJepaEncoderSource::TrainedTtt,
            model_manifest_path: None,
            model_cache_dir: None,
            model_profile: BevyJepaModelPackageProfile::default(),
            model_base_url: burn_jepa::burn_jepa_model_profile_base_url(
                BevyJepaModelPackageProfile::default(),
            ),
            model_auto_download: true,
            ttt_model_path: None,
            jepa_checkpoint_dir: Some(PathBuf::from(DEFAULT_VJEPA21_CHECKPOINT_DIR)),
            jepa_config_path: Some(PathBuf::from(DEFAULT_VJEPA21_CONFIG_PATH)),
            jepa_weights_name: DEFAULT_VJEPA21_WEIGHTS_NAME.to_string(),
            source: BevyJepaFrameSource::Camera,
            mask_source: BevyJepaMaskSource::PatchDiff,
            display_transfer: BevyJepaDisplayTransfer::Gpu,
            image_path: None,
            anyup_weights: None,
            anyup_model_manifest_path: None,
            anyup_model_cache_dir: None,
            anyup_model_profile: BevyJepaAnyUpModelPackageProfile::default(),
            anyup_model_base_url: burn_jepa::burn_anyup_model_profile_base_url(
                BevyJepaAnyUpModelPackageProfile::default(),
            ),
            anyup_model_auto_download: true,
            anyup_attention_mode: AnyUpAttentionMode::EfficientLocal,
            pipeline: FeatureFrameViewerConfig::default(),
            show_metrics: true,
            camera_width: DEFAULT_CAMERA_WIDTH,
            camera_height: DEFAULT_CAMERA_HEIGHT,
            camera_fps: DEFAULT_CAMERA_FPS,
        }
    }
}

impl BevyJepaConfig {
    pub fn pipeline_config(&self) -> &FeatureFrameViewerConfig {
        &self.pipeline
    }
}

impl Deref for BevyJepaConfig {
    type Target = FeatureFrameViewerConfig;

    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

impl DerefMut for BevyJepaConfig {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.pipeline
    }
}
