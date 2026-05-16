use std::{fmt, str::FromStr};

use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};

pub const DEFAULT_IMAGE_SIZE: usize = 64;
pub const DEFAULT_CONTEXT_DENSITY: f32 = 0.25;
pub const DEFAULT_PATCH_DIFF_THRESHOLD: f32 = 0.15;
pub const DEFAULT_ANYUP_CHUNK_SIZE: usize = 4;
pub const DEFAULT_PCA_UPDATE_EVERY: u64 = 16;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BevyJepaMaskSource {
    #[default]
    Autogaze,
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
            Self::PatchDiff => Self::Autogaze,
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
    pub mask_source: BevyJepaMaskSource,
    pub display_transfer: BevyJepaDisplayTransfer,
    pub image_size: usize,
    pub context_density: f32,
    pub patch_diff_threshold: f32,
    pub anyup_q_chunk_size: usize,
    pub pca_update_every: u64,
    pub high_res_pca_every: u64,
    pub show_metrics: bool,
    pub measure_stages: bool,
    pub sync_measurements: bool,
}

impl Default for BevyJepaConfig {
    fn default() -> Self {
        Self {
            mask_source: BevyJepaMaskSource::Autogaze,
            display_transfer: BevyJepaDisplayTransfer::Gpu,
            image_size: DEFAULT_IMAGE_SIZE,
            context_density: DEFAULT_CONTEXT_DENSITY,
            patch_diff_threshold: DEFAULT_PATCH_DIFF_THRESHOLD,
            anyup_q_chunk_size: DEFAULT_ANYUP_CHUNK_SIZE,
            pca_update_every: DEFAULT_PCA_UPDATE_EVERY,
            high_res_pca_every: 1,
            show_metrics: true,
            measure_stages: true,
            sync_measurements: false,
        }
    }
}

impl BevyJepaConfig {
    pub fn context_tokens(&self, dense_tokens: usize) -> usize {
        let density = self.context_density.clamp(0.01, 1.0);
        ((dense_tokens as f32 * density).round() as usize).clamp(1, dense_tokens.max(1))
    }
}
