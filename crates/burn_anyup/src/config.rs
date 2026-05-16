use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AnyUpAttentionMode {
    #[default]
    EfficientLocal,
    UpstreamMasked,
}

impl AnyUpAttentionMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EfficientLocal => "efficient-local",
            Self::UpstreamMasked => "upstream-masked",
        }
    }

    pub const fn valid_values() -> &'static [&'static str] {
        &[
            "efficient-local",
            "efficient",
            "local",
            "natten",
            "upstream-masked",
            "upstream",
            "paper",
            "masked",
        ]
    }
}

impl fmt::Display for AnyUpAttentionMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for AnyUpAttentionMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "efficient-local" | "efficient_local" | "efficient" | "local" | "natten" => {
                Ok(Self::EfficientLocal)
            }
            "upstream-masked" | "upstream_masked" | "upstream" | "paper" | "masked"
            | "masked-mha" | "masked_mha" => Ok(Self::UpstreamMasked),
            other => Err(format!(
                "unsupported AnyUp attention mode `{other}`; expected one of {}",
                Self::valid_values().join(", ")
            )),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct AnyUpConfig {
    pub input_dim: usize,
    pub qk_dim: usize,
    pub lfu_dim: Option<usize>,
    pub kernel_size: usize,
    pub kernel_size_lfu: usize,
    pub window_ratio: f32,
    pub num_heads: usize,
    pub group_norm_groups: usize,
    pub group_norm_eps: f64,
    pub rms_norm_eps: f64,
    pub attention_mode: AnyUpAttentionMode,
}

impl Default for AnyUpConfig {
    fn default() -> Self {
        Self {
            input_dim: 3,
            qk_dim: 128,
            lfu_dim: None,
            kernel_size: 1,
            kernel_size_lfu: 5,
            window_ratio: 0.1,
            num_heads: 4,
            group_norm_groups: 8,
            group_norm_eps: 1.0e-5,
            rms_norm_eps: f32::EPSILON as f64,
            attention_mode: AnyUpAttentionMode::EfficientLocal,
        }
    }
}

impl AnyUpConfig {
    pub fn lfu_dim(&self) -> usize {
        self.lfu_dim.unwrap_or(self.qk_dim).max(1)
    }

    pub fn tiny_for_tests() -> Self {
        Self {
            input_dim: 3,
            qk_dim: 8,
            lfu_dim: None,
            kernel_size: 1,
            kernel_size_lfu: 3,
            window_ratio: 0.4,
            num_heads: 2,
            group_norm_groups: 2,
            group_norm_eps: 1.0e-5,
            rms_norm_eps: f32::EPSILON as f64,
            attention_mode: AnyUpAttentionMode::EfficientLocal,
        }
    }

    pub fn upstream_masked() -> Self {
        Self {
            attention_mode: AnyUpAttentionMode::UpstreamMasked,
            ..Self::default()
        }
    }

    pub fn with_attention_mode(mut self, attention_mode: AnyUpAttentionMode) -> Self {
        self.attention_mode = attention_mode;
        self
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.input_dim > 0, "input_dim must be nonzero");
        anyhow::ensure!(self.qk_dim > 0, "qk_dim must be nonzero");
        anyhow::ensure!(
            self.qk_dim.is_multiple_of(self.num_heads.max(1)),
            "qk_dim must be divisible by num_heads"
        );
        anyhow::ensure!(self.num_heads > 0, "num_heads must be nonzero");
        anyhow::ensure!(self.kernel_size > 0, "kernel_size must be nonzero");
        anyhow::ensure!(self.kernel_size_lfu > 0, "kernel_size_lfu must be nonzero");
        anyhow::ensure!(
            self.group_norm_groups > 0,
            "group_norm_groups must be nonzero"
        );
        anyhow::ensure!(
            self.qk_dim.is_multiple_of(self.group_norm_groups),
            "qk_dim must be divisible by group_norm_groups"
        );
        Ok(())
    }
}
