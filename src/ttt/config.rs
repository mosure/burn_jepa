use crate::VJepaConfig;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TttTargetMode {
    TeacherFinal,
    SelfHidden,
}

impl Default for TttTargetMode {
    fn default() -> Self {
        Self::TeacherFinal
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TttBackpropMode {
    #[default]
    FinalFeature,
    TruncatedFinal,
    LayerLocal,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TttEncoderConfig {
    pub layers: Vec<usize>,
    pub chunk_tokens: usize,
    pub ttt_lr: f32,
    pub use_projection: bool,
    pub conv_kernel: usize,
    pub target: TttTargetMode,
    pub rollout_blocks: usize,
    pub backprop_mode: TttBackpropMode,
    pub backprop_truncate_blocks: usize,
    pub freeze_pretrained: bool,
}

impl Default for TttEncoderConfig {
    fn default() -> Self {
        Self {
            layers: vec![0],
            chunk_tokens: 128,
            ttt_lr: 0.05,
            use_projection: true,
            conv_kernel: 3,
            target: TttTargetMode::TeacherFinal,
            rollout_blocks: 1,
            backprop_mode: TttBackpropMode::FinalFeature,
            backprop_truncate_blocks: 1,
            freeze_pretrained: true,
        }
    }
}

impl TttEncoderConfig {
    pub fn validate(&self, config: &VJepaConfig) -> Result<()> {
        ensure!(self.chunk_tokens > 0, "ttt.chunk_tokens must be nonzero");
        ensure!(self.ttt_lr >= 0.0, "ttt.ttt_lr must be non-negative");
        ensure!(self.conv_kernel > 0, "ttt.conv_kernel must be nonzero");
        ensure!(
            self.backprop_truncate_blocks > 0,
            "ttt.backprop_truncate_blocks must be nonzero"
        );
        for &layer in &self.layers {
            ensure!(
                layer < config.encoder.depth.max(1),
                "ttt layer {layer} is outside encoder depth {}",
                config.encoder.depth.max(1)
            );
        }
        Ok(())
    }

    pub(crate) fn normalized_layers(&self) -> Vec<usize> {
        let mut layers = self.layers.clone();
        layers.sort_unstable();
        layers.dedup();
        layers
    }
}
