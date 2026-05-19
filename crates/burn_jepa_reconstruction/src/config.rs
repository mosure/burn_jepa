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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JepaReconstructionArchitecture {
    #[default]
    ResidualUniform,
    PyramidConvnext,
    PatchLinear,
    PatchConv,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct JepaReconstructionConfig {
    pub architecture: JepaReconstructionArchitecture,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub min_channels: usize,
    pub output_channels: usize,
    pub patch_size: usize,
    pub upsample_blocks: usize,
    pub residual_blocks_per_scale: usize,
    pub norm_groups: usize,
    pub convnext_expansion: usize,
    pub residual_scale: f64,
    pub epsilon: f64,
    pub output_activation: JepaReconstructionOutputActivation,
}

impl Default for JepaReconstructionConfig {
    fn default() -> Self {
        Self {
            architecture: JepaReconstructionArchitecture::ResidualUniform,
            input_dim: 768,
            hidden_dim: 256,
            min_channels: 32,
            output_channels: 3,
            patch_size: 16,
            upsample_blocks: 4,
            residual_blocks_per_scale: 1,
            norm_groups: 8,
            convnext_expansion: 2,
            residual_scale: 0.5,
            epsilon: 1.0e-5,
            output_activation: JepaReconstructionOutputActivation::Sigmoid,
        }
    }
}

impl JepaReconstructionConfig {
    pub fn tiny_for_tests() -> Self {
        Self {
            architecture: JepaReconstructionArchitecture::ResidualUniform,
            input_dim: 8,
            hidden_dim: 16,
            min_channels: 8,
            output_channels: 3,
            patch_size: 4,
            upsample_blocks: 2,
            residual_blocks_per_scale: 1,
            norm_groups: 4,
            convnext_expansion: 2,
            residual_scale: 0.5,
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
            self.min_channels > 0,
            "reconstruction min_channels must be positive"
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
            self.hidden_dim.is_multiple_of(self.norm_groups),
            "reconstruction hidden_dim must be divisible by norm_groups"
        );
        ensure!(
            self.convnext_expansion > 0,
            "reconstruction convnext_expansion must be positive"
        );
        ensure!(
            self.residual_scale.is_finite(),
            "reconstruction residual_scale must be finite"
        );
        ensure!(
            self.epsilon > 0.0,
            "reconstruction epsilon must be positive"
        );
        Ok(())
    }
}
