use crate::VJepaConfig;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TttTargetMode {
    #[default]
    TeacherFinal,
    SelfHidden,
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
pub enum TttMemoryDynamics {
    #[default]
    Ema,
    MemoryAlibi,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TttInsertionMode {
    #[default]
    Adapter,
    InPlaceMlp,
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
    pub insertion: TttInsertionMode,
    pub layer_placement: TttLayerPlacement,
    pub layers: Vec<usize>,
    pub predictor_layers: Vec<usize>,
    pub chunk_tokens: usize,
    pub ttt_lr: f32,
    pub use_projection: bool,
    pub conv_kernel: usize,
    pub memory_update: TttMemoryUpdateSource,
    pub memory_dynamics: TttMemoryDynamics,
    pub memory_alibi_half_lives: Vec<usize>,
    pub memory_alibi_read_weights: Vec<f32>,
    pub memory_alibi_update_weights: Vec<f32>,
    pub memory_clip_rms: f32,
    pub supervision: TttSupervisionMode,
    pub hybrid_final_steps: usize,
    #[serde(default, skip_serializing_if = "is_default_target_mode")]
    pub target: TttTargetMode,
    pub rollout_blocks: usize,
    pub rollout_chunk_frames: usize,
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
            insertion: TttInsertionMode::Adapter,
            layer_placement: TttLayerPlacement::FirstLast,
            layers: Vec::new(),
            predictor_layers: Vec::new(),
            chunk_tokens: 128,
            ttt_lr: 0.05,
            use_projection: true,
            conv_kernel: 3,
            memory_update: TttMemoryUpdateSource::SelfHidden,
            memory_dynamics: TttMemoryDynamics::Ema,
            memory_alibi_half_lives: default_memory_alibi_half_lives(),
            memory_alibi_read_weights: default_memory_alibi_read_weights(),
            memory_alibi_update_weights: default_memory_alibi_update_weights(),
            memory_clip_rms: 0.0,
            supervision: TttSupervisionMode::FinalTeacher,
            hybrid_final_steps: 1,
            target: TttTargetMode::TeacherFinal,
            rollout_blocks: 1,
            rollout_chunk_frames: 16,
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
            self.memory_clip_rms.is_finite() && self.memory_clip_rms >= 0.0,
            "ttt.memory_clip_rms must be finite and non-negative"
        );
        if self.memory_dynamics == TttMemoryDynamics::MemoryAlibi {
            let half_lives = self.resolved_memory_alibi_half_lives();
            ensure!(
                !half_lives.is_empty(),
                "ttt.memory_alibi_half_lives must be non-empty when memory_dynamics=memory_alibi"
            );
            ensure!(
                half_lives.iter().all(|&half_life| half_life > 0),
                "ttt.memory_alibi_half_lives must contain positive values"
            );
            validate_memory_weights(
                &self.resolved_memory_alibi_read_weights(),
                "ttt.memory_alibi_read_weights",
            )?;
            validate_memory_weights(
                &self.resolved_memory_alibi_update_weights(),
                "ttt.memory_alibi_update_weights",
            )?;
            ensure!(
                self.resolved_memory_alibi_read_weights().len() == half_lives.len(),
                "ttt.memory_alibi_read_weights length must match memory_alibi_half_lives"
            );
            ensure!(
                self.resolved_memory_alibi_update_weights().len() == half_lives.len(),
                "ttt.memory_alibi_update_weights length must match memory_alibi_half_lives"
            );
        }
        ensure!(
            self.backprop_truncate_blocks > 0,
            "ttt.backprop_truncate_blocks must be nonzero"
        );
        ensure!(
            self.rollout_chunk_frames > 0,
            "ttt.rollout_chunk_frames must be nonzero"
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
        for layer in self.resolved_predictor_layers(config) {
            ensure!(
                layer < config.predictor.depth.max(1),
                "ttt predictor layer {layer} is outside predictor depth {}",
                config.predictor.depth.max(1)
            );
        }
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

    pub fn resolved_predictor_layers(&self, _config: &VJepaConfig) -> Vec<usize> {
        let mut layers = self.predictor_layers.clone();
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

    pub fn memory_bank_count(&self) -> usize {
        match self.memory_dynamics {
            TttMemoryDynamics::Ema => 1,
            TttMemoryDynamics::MemoryAlibi => self.resolved_memory_alibi_half_lives().len().max(1),
        }
    }

    pub fn resolved_memory_alibi_half_lives(&self) -> Vec<usize> {
        if self.memory_alibi_half_lives.is_empty() {
            default_memory_alibi_half_lives()
        } else {
            self.memory_alibi_half_lives.clone()
        }
    }

    pub fn resolved_memory_alibi_read_weights(&self) -> Vec<f32> {
        resolve_memory_weights(
            &self.memory_alibi_read_weights,
            &default_memory_alibi_read_weights(),
            self.resolved_memory_alibi_half_lives().len(),
        )
    }

    pub fn resolved_memory_alibi_update_weights(&self) -> Vec<f32> {
        resolve_memory_weights(
            &self.memory_alibi_update_weights,
            &default_memory_alibi_update_weights(),
            self.resolved_memory_alibi_half_lives().len(),
        )
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

fn default_memory_alibi_half_lives() -> Vec<usize> {
    vec![8, 64, 512]
}

fn default_memory_alibi_read_weights() -> Vec<f32> {
    vec![0.45, 0.35, 0.20]
}

fn default_memory_alibi_update_weights() -> Vec<f32> {
    vec![1.0, 1.0, 1.0]
}

fn resolve_memory_weights(values: &[f32], defaults: &[f32], len: usize) -> Vec<f32> {
    let source = if values.is_empty() { defaults } else { values };
    if source.len() == len {
        normalize_memory_weights(source)
    } else if len == defaults.len() {
        normalize_memory_weights(defaults)
    } else {
        vec![1.0 / len.max(1) as f32; len.max(1)]
    }
}

fn normalize_memory_weights(values: &[f32]) -> Vec<f32> {
    let positive = values
        .iter()
        .map(|value| value.max(0.0))
        .collect::<Vec<_>>();
    let sum = positive.iter().sum::<f32>();
    if sum <= f32::EPSILON {
        return vec![1.0 / positive.len().max(1) as f32; positive.len().max(1)];
    }
    positive.into_iter().map(|value| value / sum).collect()
}

fn validate_memory_weights(values: &[f32], name: &str) -> Result<()> {
    ensure!(
        !values.is_empty(),
        "{name} must resolve to at least one value"
    );
    ensure!(
        values
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0),
        "{name} must contain finite non-negative values"
    );
    ensure!(
        values.iter().any(|value| *value > 0.0),
        "{name} must contain at least one positive value"
    );
    Ok(())
}
