use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JepaReconstructionOutputActivation {
    #[default]
    Sigmoid,
    Tanh01,
    None,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct JepaReconstructionConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_channels: usize,
    pub patch_size: usize,
    pub upsample_blocks: usize,
    pub residual_blocks_per_scale: usize,
    pub norm_groups: usize,
    pub epsilon: f64,
    pub output_activation: JepaReconstructionOutputActivation,
}

impl Default for JepaReconstructionConfig {
    fn default() -> Self {
        Self {
            input_dim: 768,
            hidden_dim: 256,
            output_channels: 3,
            patch_size: 16,
            upsample_blocks: 4,
            residual_blocks_per_scale: 1,
            norm_groups: 8,
            epsilon: 1.0e-5,
            output_activation: JepaReconstructionOutputActivation::Sigmoid,
        }
    }
}

impl JepaReconstructionConfig {
    pub fn tiny_for_tests() -> Self {
        Self {
            input_dim: 8,
            hidden_dim: 16,
            output_channels: 3,
            patch_size: 4,
            upsample_blocks: 2,
            residual_blocks_per_scale: 1,
            norm_groups: 4,
            epsilon: 1.0e-5,
            output_activation: JepaReconstructionOutputActivation::Sigmoid,
        }
    }

    pub fn output_scale(&self) -> usize {
        1usize << self.upsample_blocks.min(usize::BITS as usize - 1)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.input_dim > 0,
            "reconstruction input_dim must be positive"
        );
        ensure!(
            self.hidden_dim > 0,
            "reconstruction hidden_dim must be positive"
        );
        ensure!(
            self.output_channels > 0,
            "reconstruction output_channels must be positive"
        );
        ensure!(
            self.patch_size > 0,
            "reconstruction patch_size must be positive"
        );
        ensure!(
            self.upsample_blocks <= 8,
            "reconstruction upsample_blocks is unexpectedly large"
        );
        ensure!(
            self.norm_groups > 0,
            "reconstruction norm_groups must be positive"
        );
        ensure!(
            self.epsilon > 0.0,
            "reconstruction epsilon must be positive"
        );
        Ok(())
    }
}
