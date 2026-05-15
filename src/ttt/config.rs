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
pub enum TttMemoryUpdateSource {
    #[default]
    SelfHidden,
    TeacherForcedDiagnostic,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TttSupervisionMode {
    #[default]
    FinalTeacher,
    LayerLocalTeacher,
    Hybrid,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TttBackpropMode {
    #[default]
    FinalFeature,
    TruncatedFinal,
    LayerLocal,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TttLayerPlacement {
    Explicit,
    First,
    Middle,
    Last,
    #[default]
    FirstLast,
    Thirds,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TttEncoderConfig {
    pub layer_placement: TttLayerPlacement,
    pub layers: Vec<usize>,
    pub predictor_layers: Vec<usize>,
    pub chunk_tokens: usize,
    pub ttt_lr: f32,
    pub use_projection: bool,
    pub conv_kernel: usize,
    pub memory_update: TttMemoryUpdateSource,
    pub supervision: TttSupervisionMode,
    pub hybrid_final_steps: usize,
    #[serde(default, skip_serializing_if = "is_default_target_mode")]
    pub target: TttTargetMode,
    pub rollout_blocks: usize,
    pub backprop_mode: TttBackpropMode,
    pub backprop_truncate_blocks: usize,
    pub freeze_pretrained: bool,
}

fn is_default_target_mode(mode: &TttTargetMode) -> bool {
    *mode == TttTargetMode::default()
}

impl Default for TttEncoderConfig {
    fn default() -> Self {
        Self {
            layer_placement: TttLayerPlacement::FirstLast,
            layers: Vec::new(),
            predictor_layers: Vec::new(),
            chunk_tokens: 128,
            ttt_lr: 0.05,
            use_projection: true,
            conv_kernel: 3,
            memory_update: TttMemoryUpdateSource::SelfHidden,
            supervision: TttSupervisionMode::FinalTeacher,
            hybrid_final_steps: 1,
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
        ensure!(
            self.hybrid_final_steps > 0 || self.supervision != TttSupervisionMode::Hybrid,
            "ttt.hybrid_final_steps must be nonzero when ttt.supervision=hybrid"
        );
        for layer in self.resolved_layers(config) {
            ensure!(
                layer < config.encoder.depth.max(1),
                "ttt layer {layer} is outside encoder depth {}",
                config.encoder.depth.max(1)
            );
        }
        ensure!(
            self.predictor_layers.is_empty(),
            "TTT predictor-layer adapters are not implemented yet; use encoder layer placement"
        );
        Ok(())
    }

    pub fn resolved_layers(&self, config: &VJepaConfig) -> Vec<usize> {
        let depth = config.encoder.depth.max(1);
        let mut layers =
            if !self.layers.is_empty() || self.layer_placement == TttLayerPlacement::Explicit {
                self.layers.clone()
            } else {
                placement_layers(self.layer_placement, depth)
            };
        layers.sort_unstable();
        layers.dedup();
        layers
    }

    pub fn capture_layers(&self, config: &VJepaConfig) -> Vec<usize> {
        let mut layers = config.encoder.hierarchical_layers();
        if self.supervision.requires_layer_targets() {
            layers.extend(self.resolved_layers(config));
        }
        layers.sort_unstable();
        layers.dedup();
        layers
    }

    pub fn train_supervision_for_step(
        &self,
        step_index: usize,
        max_steps: usize,
    ) -> TttSupervisionMode {
        match self.supervision {
            TttSupervisionMode::Hybrid => {
                let final_steps = self.hybrid_final_steps.min(max_steps.max(1));
                if step_index + final_steps >= max_steps {
                    TttSupervisionMode::FinalTeacher
                } else {
                    TttSupervisionMode::LayerLocalTeacher
                }
            }
            mode => mode,
        }
    }
}

impl TttSupervisionMode {
    pub fn requires_layer_targets(self) -> bool {
        matches!(self, Self::LayerLocalTeacher | Self::Hybrid)
    }
}

fn placement_layers(placement: TttLayerPlacement, depth: usize) -> Vec<usize> {
    match placement {
        TttLayerPlacement::Explicit => Vec::new(),
        TttLayerPlacement::First => vec![0],
        TttLayerPlacement::Middle => vec![depth / 2],
        TttLayerPlacement::Last => vec![depth.saturating_sub(1)],
        TttLayerPlacement::FirstLast => vec![0, depth.saturating_sub(1)],
        TttLayerPlacement::Thirds => (1..=3)
            .map(|part| (part * depth).div_ceil(3).saturating_sub(1))
            .collect(),
    }
}
