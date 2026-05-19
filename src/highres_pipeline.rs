#[cfg(any(feature = "sparse-patchify-wgpu", feature = "sparse-patchify-cuda"))]
use crate::SparsePatchifyBatchPlan;
use crate::{
    FeaturePcaConfig, FeaturePcaProjector, FeaturePcaUpdateConfig, FeaturePcaUpdateScheduler,
    InterframeJepaFeatureMemory, InterframeJepaFeatureMemoryConfig,
    InterframeJepaFeatureMemoryOutput, PatchDiffRefreshState, SparseMaskBatch, SparseTokenMask,
    TokenGridShape, TttState, VJepa2_1Model, VJepaConfig, VJepaEncoderOutput, VJepaTttModel,
    apply_token_mask, jepa_feature_tokens_to_nchw,
};
use anyhow::{Context, Result, bail, ensure};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_anyup::{AnyUp, AnyUpImageGrid};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::fmt;

#[cfg(not(target_arch = "wasm32"))]
type PipelineInstant = std::time::Instant;
#[cfg(target_arch = "wasm32")]
type PipelineInstant = f64;

#[cfg(not(target_arch = "wasm32"))]
fn pipeline_now() -> PipelineInstant {
    std::time::Instant::now()
}

#[cfg(target_arch = "wasm32")]
fn pipeline_now() -> PipelineInstant {
    web_sys::window()
        .and_then(|window| window.performance())
        .map(|performance| performance.now())
        .unwrap_or_else(js_sys::Date::now)
}

#[cfg(not(target_arch = "wasm32"))]
fn pipeline_delta_us(now: PipelineInstant, previous: PipelineInstant) -> u64 {
    micros_u64(now.duration_since(previous).as_micros())
}

#[cfg(target_arch = "wasm32")]
fn pipeline_delta_us(now: PipelineInstant, previous: PipelineInstant) -> u64 {
    let elapsed_ms = now - previous;
    if elapsed_ms.is_finite() && elapsed_ms > 0.0 {
        micros_u64((elapsed_ms * 1000.0) as u128)
    } else {
        0
    }
}

fn pipeline_elapsed_us(start: PipelineInstant) -> u64 {
    pipeline_delta_us(pipeline_now(), start)
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SparseJepaAnyUpPcaMeasurementConfig {
    pub enabled: bool,
    pub sync_backend: bool,
}

impl SparseJepaAnyUpPcaMeasurementConfig {
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            sync_backend: false,
        }
    }

    pub const fn enabled() -> Self {
        Self {
            enabled: true,
            sync_backend: false,
        }
    }

    pub const fn enabled_with_backend_sync() -> Self {
        Self {
            enabled: true,
            sync_backend: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TttRuntimeCollapseGuardAction {
    None,
    #[default]
    Decay,
    Reset,
    FreezeUpdates,
}

impl TttRuntimeCollapseGuardAction {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Decay => "decay",
            Self::Reset => "reset",
            Self::FreezeUpdates => "freeze-updates",
        }
    }
}

impl fmt::Display for TttRuntimeCollapseGuardAction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl std::str::FromStr for TttRuntimeCollapseGuardAction {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "decay" => Ok(Self::Decay),
            "reset" => Ok(Self::Reset),
            "freeze-updates" | "freeze_updates" => Ok(Self::FreezeUpdates),
            other => bail!(
                "unknown TTT runtime collapse guard action `{other}`; expected none, decay, reset, or freeze-updates"
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TttRuntimeStateConfig {
    /// Apply TTT fast-memory update policy during image-stream inference.
    pub enabled: bool,
    /// Update fast weights from the current hidden state. Disable for frozen-memory diagnostics.
    pub update_fast_weight: bool,
    /// Multiplicative fast-memory decay applied after each processed frame.
    pub state_decay_per_frame: f64,
    /// Periodic reset cadence in processed frames. Zero disables interval resets.
    pub reset_interval_frames: u64,
    /// Sample token/state diagnostics every N processed frames. Zero disables diagnostics.
    pub metrics_interval_frames: u64,
    /// Enable sampled collapse guard diagnostics and mitigation.
    pub collapse_guard_enabled: bool,
    /// Action taken when the sampled rollout looks collapsed or numerically unstable.
    pub collapse_guard_action: TttRuntimeCollapseGuardAction,
    /// Extra fast-memory decay factor used by the decay collapse guard action.
    pub collapse_guard_decay: f64,
    /// Trigger guard when per-token spatial standard deviation RMS falls below this value.
    pub min_token_spatial_std_rms: f64,
    /// Trigger guard when sampled mean pairwise token cosine exceeds this value.
    pub max_mean_pairwise_token_cosine: f64,
    /// Trigger guard when fast-weight RMS exceeds this value. Non-positive disables this check.
    pub max_state_fast_weight_rms: f64,
}

impl Default for TttRuntimeStateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            update_fast_weight: true,
            state_decay_per_frame: 0.998_1,
            reset_interval_frames: 64,
            metrics_interval_frames: 0,
            collapse_guard_enabled: false,
            collapse_guard_action: TttRuntimeCollapseGuardAction::Decay,
            collapse_guard_decay: 0.25,
            min_token_spatial_std_rms: 1.0e-3,
            max_mean_pairwise_token_cosine: 0.995,
            max_state_fast_weight_rms: 0.0,
        }
    }
}

impl TttRuntimeStateConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            update_fast_weight: false,
            collapse_guard_enabled: false,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.state_decay_per_frame.is_finite()
                && (0.0..=1.0).contains(&self.state_decay_per_frame),
            "TTT runtime state decay must be finite and in [0, 1]"
        );
        ensure!(
            self.collapse_guard_decay.is_finite()
                && (0.0..=1.0).contains(&self.collapse_guard_decay),
            "TTT runtime collapse guard decay must be finite and in [0, 1]"
        );
        ensure!(
            self.min_token_spatial_std_rms.is_finite() && self.min_token_spatial_std_rms >= 0.0,
            "TTT runtime min token spatial std must be finite and non-negative"
        );
        ensure!(
            self.max_mean_pairwise_token_cosine.is_finite()
                && (-1.0..=1.0).contains(&self.max_mean_pairwise_token_cosine),
            "TTT runtime max mean pairwise token cosine must be finite and in [-1, 1]"
        );
        ensure!(
            self.max_state_fast_weight_rms.is_finite(),
            "TTT runtime max state fast-weight RMS must be finite"
        );
        ensure!(
            !self.collapse_guard_enabled || self.metrics_interval_frames > 0,
            "TTT runtime collapse guard requires metrics_interval_frames > 0"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct FeatureTokenStabilityMetrics {
    pub measured: bool,
    pub token_spatial_std_rms: f64,
    pub token_norm_mean: f64,
    pub token_norm_std: f64,
    pub mean_pairwise_token_cosine: f64,
    pub collapse_score: f64,
    pub sampled_tokens: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TttRuntimeStateMetrics {
    pub enabled: bool,
    pub update_fast_weight: bool,
    pub frame_index: u64,
    pub frames_since_reset: u64,
    pub reset_count: u64,
    pub decay_count: u64,
    pub collapse_guard_triggers: u64,
    pub state_decay_applied: bool,
    pub collapse_guard_triggered: bool,
    pub collapse_guard_action: Option<TttRuntimeCollapseGuardAction>,
    pub state_fast_weight_rms: Option<f64>,
    pub token_stability: Option<FeatureTokenStabilityMetrics>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseJepaAnyUpPcaEncodePath {
    #[default]
    DensePatchEmbed,
    SparsePatchify,
}

impl SparseJepaAnyUpPcaEncodePath {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DensePatchEmbed => "dense-patch",
            Self::SparsePatchify => "sparse-patchify",
        }
    }
}

impl fmt::Display for SparseJepaAnyUpPcaEncodePath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureFrameNode {
    LowResPca,
    HighResAnyUpPca,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FeatureFrameRequest {
    pub low_res_pca: bool,
    pub high_res_pca: bool,
    pub high_res_features: bool,
}

impl FeatureFrameRequest {
    pub const fn none() -> Self {
        Self {
            low_res_pca: false,
            high_res_pca: false,
            high_res_features: false,
        }
    }

    pub const fn low_res() -> Self {
        Self {
            low_res_pca: true,
            high_res_pca: false,
            high_res_features: false,
        }
    }

    pub const fn high_res_pca() -> Self {
        Self {
            low_res_pca: false,
            high_res_pca: true,
            high_res_features: false,
        }
    }

    pub const fn high_res_features() -> Self {
        Self {
            low_res_pca: false,
            high_res_pca: false,
            high_res_features: true,
        }
    }

    pub const fn full_pca() -> Self {
        Self {
            low_res_pca: true,
            high_res_pca: true,
            high_res_features: false,
        }
    }

    pub const fn high_res() -> Self {
        Self {
            low_res_pca: false,
            high_res_pca: true,
            high_res_features: true,
        }
    }

    pub const fn full() -> Self {
        Self {
            low_res_pca: true,
            high_res_pca: true,
            high_res_features: true,
        }
    }

    pub const fn includes(self, node: FeatureFrameNode) -> bool {
        match node {
            FeatureFrameNode::LowResPca => self.low_res_pca,
            FeatureFrameNode::HighResAnyUpPca => self.high_res_pca,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FeatureFrameSchedule {
    pub low_res_pca_every: Option<u64>,
    pub high_res_pca_every: Option<u64>,
}

impl Default for FeatureFrameSchedule {
    fn default() -> Self {
        Self {
            low_res_pca_every: Some(1),
            high_res_pca_every: None,
        }
    }
}

impl FeatureFrameSchedule {
    pub const fn every_frame_full() -> Self {
        Self {
            low_res_pca_every: Some(1),
            high_res_pca_every: Some(1),
        }
    }

    pub const fn low_res_every_frame() -> Self {
        Self {
            low_res_pca_every: Some(1),
            high_res_pca_every: None,
        }
    }

    pub const fn disabled() -> Self {
        Self {
            low_res_pca_every: None,
            high_res_pca_every: None,
        }
    }

    pub fn request_for(&self, ids: &[SparseJepaAnyUpPcaFrameId]) -> FeatureFrameRequest {
        FeatureFrameRequest {
            low_res_pca: scheduled_for_any(ids, self.low_res_pca_every),
            high_res_pca: scheduled_for_any(ids, self.high_res_pca_every),
            high_res_features: false,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SparseJepaAnyUpPcaStageMetrics {
    pub measured: bool,
    pub sync_backend: bool,
    pub encode_path: SparseJepaAnyUpPcaEncodePath,
    pub frame_count: usize,
    pub dense_tokens_per_frame: usize,
    /// Backward-compatible sparse width for cache writes.
    pub sparse_width: usize,
    pub valid_sparse_tokens: usize,
    /// Fixed row width used for cache writes.
    pub write_width: usize,
    /// Non-padding cache-write tokens across the batch.
    pub valid_write_tokens: usize,
    /// Fixed row width supplied to the encoder.
    pub encode_width: usize,
    /// Non-padding encoder tokens across the batch.
    pub valid_encode_tokens: usize,
    pub output_pixels: usize,
    pub encode_us: u64,
    pub cache_update_us: u64,
    pub token_view_us: u64,
    pub anyup_context_us: u64,
    pub anyup_decode_us: u64,
    pub low_res_pca_project_us: u64,
    pub pca_update_us: u64,
    pub pca_online_us: u64,
    pub pca_project_us: u64,
    pub pca_sample_window_frames: usize,
    pub pca_sample_frames: usize,
    pub pca_update_applied: bool,
    pub pca_update_tokens: usize,
    pub ttt_runtime: TttRuntimeStateMetrics,
    pub total_us: u64,
}

impl SparseJepaAnyUpPcaStageMetrics {
    fn set_encode_mask<B: Backend>(&mut self, mask: &SparseMaskBatch<B>) {
        self.encode_width = mask.len();
        self.valid_encode_tokens = mask.valid_token_count();
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SparseJepaAnyUpPcaPipelineConfig {
    pub memory: InterframeJepaFeatureMemoryConfig,
    pub pca: FeaturePcaConfig,
    pub pca_update: FeaturePcaUpdateConfig,
    pub anyup_q_chunk_size: Option<usize>,
    pub update_pca_online: bool,
    pub measurement: SparseJepaAnyUpPcaMeasurementConfig,
    pub ttt_runtime: TttRuntimeStateConfig,
}

impl Default for SparseJepaAnyUpPcaPipelineConfig {
    fn default() -> Self {
        Self {
            memory: InterframeJepaFeatureMemoryConfig::default(),
            pca: FeaturePcaConfig::default(),
            pca_update: FeaturePcaUpdateConfig::disabled(),
            anyup_q_chunk_size: Some(16),
            update_pca_online: false,
            measurement: SparseJepaAnyUpPcaMeasurementConfig::disabled(),
            ttt_runtime: TttRuntimeStateConfig::default(),
        }
    }
}

impl SparseJepaAnyUpPcaPipelineConfig {
    pub fn validate(&self) -> Result<()> {
        self.pca.validate()?;
        self.pca_update.validate()?;
        self.effective_pca_update().validate()?;
        ensure!(
            self.anyup_q_chunk_size.is_none_or(|chunk| chunk > 0),
            "AnyUp q chunk size must be positive when set"
        );
        self.ttt_runtime.validate()?;
        Ok(())
    }

    pub fn effective_pca_update(&self) -> FeaturePcaUpdateConfig {
        if self.pca_update.enabled() || !self.update_pca_online {
            self.pca_update.clone()
        } else {
            FeaturePcaUpdateConfig::rolling_low_res_every(1)
        }
    }
}

#[derive(Debug)]
pub struct SparseJepaAnyUpPcaOutput<B: Backend> {
    pub encoded: VJepaEncoderOutput<B>,
    pub token_cache: InterframeJepaFeatureMemoryOutput<B>,
    pub low_res_features: Tensor<B, 4>,
    pub high_res_features: Tensor<B, 4>,
    pub pca_display: Tensor<B, 4>,
    pub mask: SparseTokenMask,
}

#[derive(Debug)]
pub struct SparseJepaAnyUpPcaStepBatchOutput<B: Backend> {
    pub encoded: VJepaEncoderOutput<B>,
    pub token_cache: InterframeJepaFeatureMemoryOutput<B>,
    pub low_res_features: Tensor<B, 4>,
    pub high_res_features: Tensor<B, 4>,
    pub pca_display: Tensor<B, 4>,
    pub mask: SparseMaskBatch<B>,
}

#[derive(Debug)]
pub struct LowResFrameArtifacts<B: Backend> {
    pub features: Tensor<B, 4>,
    pub pca_display: Option<Tensor<B, 4>>,
}

#[derive(Debug)]
pub struct HighResFrameArtifacts<B: Backend> {
    pub features: Option<Tensor<B, 4>>,
    pub pca_display: Option<Tensor<B, 4>>,
}

#[derive(Debug)]
pub struct FeatureFrameBatch<B: Backend> {
    pub encoded: VJepaEncoderOutput<B>,
    pub token_cache: InterframeJepaFeatureMemoryOutput<B>,
    pub low_res: LowResFrameArtifacts<B>,
    pub high_res: Option<HighResFrameArtifacts<B>>,
    pub mask: SparseMaskBatch<B>,
}

impl<B: Backend> FeatureFrameBatch<B> {
    pub fn has_low_res_pca(&self) -> bool {
        self.low_res.pca_display.is_some()
    }

    pub fn has_high_res_pca(&self) -> bool {
        self.high_res
            .as_ref()
            .is_some_and(|high_res| high_res.pca_display.is_some())
    }

    pub fn has_high_res_features(&self) -> bool {
        self.high_res
            .as_ref()
            .is_some_and(|high_res| high_res.features.is_some())
    }

    fn into_full_output(self) -> Result<SparseJepaAnyUpPcaStepBatchOutput<B>> {
        let Some(high_res) = self.high_res else {
            bail!("full high-res output was not requested for this frame batch");
        };
        Ok(SparseJepaAnyUpPcaStepBatchOutput {
            encoded: self.encoded,
            token_cache: self.token_cache,
            low_res_features: self.low_res.features,
            high_res_features: high_res
                .features
                .ok_or_else(|| anyhow::anyhow!("high-res features were not requested"))?,
            pca_display: high_res
                .pca_display
                .ok_or_else(|| anyhow::anyhow!("high-res PCA display was not requested"))?,
            mask: self.mask,
        })
    }
}

#[derive(Debug)]
pub struct MeasuredFeatureFrameBatch<B: Backend> {
    pub output: FeatureFrameBatch<B>,
    pub metrics: SparseJepaAnyUpPcaStageMetrics,
}

#[derive(Debug)]
pub struct SparseJepaAnyUpPcaMeasuredOutput<B: Backend> {
    pub output: SparseJepaAnyUpPcaOutput<B>,
    pub metrics: SparseJepaAnyUpPcaStageMetrics,
}

#[derive(Debug)]
pub struct SparseJepaAnyUpPcaMeasuredBatchOutput<B: Backend> {
    pub output: SparseJepaAnyUpPcaStepBatchOutput<B>,
    pub metrics: SparseJepaAnyUpPcaStageMetrics,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureFrameJepaEncoderKind {
    #[default]
    Base,
    Ttt,
}

pub enum FeatureFrameJepaEncoder<B: Backend> {
    Base(Box<VJepa2_1Model<B>>),
    Ttt {
        model: Box<VJepaTttModel<B>>,
        state: TttState<B>,
        runtime: TttRuntimeStateTracker,
    },
}

#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct TttRuntimeStateTracker {
    frame_index: u64,
    frames_since_reset: u64,
    reset_count: u64,
    decay_count: u64,
    collapse_guard_triggers: u64,
    update_fast_weight: bool,
}

impl<B: Backend> FeatureFrameJepaEncoder<B> {
    pub fn base(model: VJepa2_1Model<B>) -> Self {
        Self::Base(Box::new(model))
    }

    pub fn ttt(model: VJepaTttModel<B>) -> Self {
        let state = model.fresh_state();
        Self::Ttt {
            model: Box::new(model),
            state,
            runtime: TttRuntimeStateTracker {
                update_fast_weight: true,
                ..TttRuntimeStateTracker::default()
            },
        }
    }

    pub fn kind(&self) -> FeatureFrameJepaEncoderKind {
        match self {
            Self::Base(_) => FeatureFrameJepaEncoderKind::Base,
            Self::Ttt { .. } => FeatureFrameJepaEncoderKind::Ttt,
        }
    }

    pub fn config(&self) -> &VJepaConfig {
        match self {
            Self::Base(model) => model.config(),
            Self::Ttt { model, .. } => model.config(),
        }
    }

    pub fn base_model(&self) -> Option<&VJepa2_1Model<B>> {
        match self {
            Self::Base(model) => Some(model.as_ref()),
            Self::Ttt { .. } => None,
        }
    }

    pub fn reset_state(&mut self) {
        if let Self::Ttt {
            model,
            state,
            runtime,
        } = self
        {
            *state = model.fresh_state();
            *runtime = TttRuntimeStateTracker {
                update_fast_weight: true,
                ..TttRuntimeStateTracker::default()
            };
        }
    }

    pub fn encode_image_batch(
        &mut self,
        image: Tensor<B, 4>,
        mask: SparseMaskBatch<B>,
    ) -> Result<VJepaEncoderOutput<B>> {
        match self {
            Self::Base(model) => model.encode_image_batch(image, mask),
            Self::Ttt { model, state, .. } => {
                model.encode_image_batch_with_state(image, mask, None, state)
            }
        }
    }

    pub fn encode_image_batch_with_runtime(
        &mut self,
        image: Tensor<B, 4>,
        mask: SparseMaskBatch<B>,
        runtime_config: &TttRuntimeStateConfig,
    ) -> Result<(VJepaEncoderOutput<B>, TttRuntimeStateMetrics)> {
        match self {
            Self::Base(model) => Ok((
                model.encode_image_batch(image, mask)?,
                TttRuntimeStateMetrics::default(),
            )),
            Self::Ttt {
                model,
                state,
                runtime,
            } => encode_ttt_image_batch_with_runtime(
                model,
                state,
                runtime,
                image,
                mask,
                runtime_config,
            ),
        }
    }
}

fn encode_ttt_image_batch_with_runtime<B: Backend>(
    model: &VJepaTttModel<B>,
    state: &mut TttState<B>,
    runtime: &mut TttRuntimeStateTracker,
    image: Tensor<B, 4>,
    mask: SparseMaskBatch<B>,
    runtime_config: &TttRuntimeStateConfig,
) -> Result<(VJepaEncoderOutput<B>, TttRuntimeStateMetrics)> {
    let batch = image.shape().dims::<4>()[0] as u64;
    let update_fast_weight = begin_ttt_runtime_step(model, state, runtime, runtime_config)?;
    let output = model.encode_image_batch_with_state_options(
        image,
        mask,
        None,
        state,
        update_fast_weight,
    )?;
    let metrics = finish_ttt_runtime_step(
        model,
        state,
        runtime,
        &output,
        batch,
        runtime_config,
        update_fast_weight,
    )?;
    Ok((output, metrics))
}

fn begin_ttt_runtime_step<B: Backend>(
    model: &VJepaTttModel<B>,
    state: &mut TttState<B>,
    runtime: &mut TttRuntimeStateTracker,
    runtime_config: &TttRuntimeStateConfig,
) -> Result<bool> {
    runtime_config.validate()?;
    if runtime_config.enabled
        && runtime_config.reset_interval_frames > 0
        && runtime.frames_since_reset >= runtime_config.reset_interval_frames
    {
        *state = model.fresh_state();
        runtime.frames_since_reset = 0;
        runtime.reset_count = runtime.reset_count.saturating_add(1);
        runtime.update_fast_weight = true;
    }

    Ok(runtime_config.enabled && runtime_config.update_fast_weight && runtime.update_fast_weight)
}

fn finish_ttt_runtime_step<B: Backend>(
    model: &VJepaTttModel<B>,
    state: &mut TttState<B>,
    runtime: &mut TttRuntimeStateTracker,
    output: &VJepaEncoderOutput<B>,
    batch: u64,
    runtime_config: &TttRuntimeStateConfig,
    update_fast_weight: bool,
) -> Result<TttRuntimeStateMetrics> {
    runtime.frame_index = runtime.frame_index.saturating_add(batch);
    runtime.frames_since_reset = runtime.frames_since_reset.saturating_add(batch);

    let mut metrics = TttRuntimeStateMetrics {
        enabled: runtime_config.enabled,
        update_fast_weight,
        frame_index: runtime.frame_index,
        frames_since_reset: runtime.frames_since_reset,
        reset_count: runtime.reset_count,
        decay_count: runtime.decay_count,
        collapse_guard_triggers: runtime.collapse_guard_triggers,
        ..TttRuntimeStateMetrics::default()
    };

    if runtime_config.enabled && runtime_config.state_decay_per_frame < 1.0 {
        let decay = runtime_config.state_decay_per_frame.powf(batch as f64);
        state.decay(decay);
        runtime.decay_count = runtime.decay_count.saturating_add(batch);
        metrics.decay_count = runtime.decay_count;
        metrics.state_decay_applied = true;
    }

    let should_sample = runtime_config.enabled
        && runtime_config.metrics_interval_frames > 0
        && runtime.frame_index % runtime_config.metrics_interval_frames < batch;
    if should_sample && !cfg!(target_arch = "wasm32") {
        let token_stability = measure_feature_token_stability(output.tokens.clone())?;
        let state_rms = measure_ttt_state_fast_weight_rms(state)?;
        let collapsed = runtime_config.collapse_guard_enabled
            && (token_stability.token_spatial_std_rms < runtime_config.min_token_spatial_std_rms
                || token_stability.mean_pairwise_token_cosine
                    > runtime_config.max_mean_pairwise_token_cosine
                || runtime_config.max_state_fast_weight_rms > 0.0
                    && state_rms.is_some_and(|rms| rms > runtime_config.max_state_fast_weight_rms));
        metrics.token_stability = Some(token_stability);
        metrics.state_fast_weight_rms = state_rms;
        if collapsed {
            metrics.collapse_guard_triggered = true;
            metrics.collapse_guard_action = Some(runtime_config.collapse_guard_action);
            runtime.collapse_guard_triggers = runtime.collapse_guard_triggers.saturating_add(1);
            metrics.collapse_guard_triggers = runtime.collapse_guard_triggers;
            match runtime_config.collapse_guard_action {
                TttRuntimeCollapseGuardAction::None => {}
                TttRuntimeCollapseGuardAction::Decay => {
                    state.decay(runtime_config.collapse_guard_decay);
                    runtime.decay_count = runtime.decay_count.saturating_add(1);
                    metrics.decay_count = runtime.decay_count;
                    metrics.state_decay_applied = true;
                    runtime.update_fast_weight = true;
                }
                TttRuntimeCollapseGuardAction::Reset => {
                    *state = model.fresh_state();
                    runtime.frames_since_reset = 0;
                    runtime.reset_count = runtime.reset_count.saturating_add(1);
                    runtime.update_fast_weight = true;
                    metrics.frames_since_reset = 0;
                    metrics.reset_count = runtime.reset_count;
                }
                TttRuntimeCollapseGuardAction::FreezeUpdates => {
                    runtime.update_fast_weight = false;
                }
            }
        }
    }

    Ok(metrics)
}

#[cfg(feature = "sparse-patchify-wgpu")]
impl FeatureFrameJepaEncoder<burn_flex_gmm::wgpu::DefaultWgpuBackend> {
    fn encode_image_sparse_patchify_wgpu_batch_with_runtime(
        &mut self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        patchify_plan: &SparsePatchifyBatchPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        runtime_config: &TttRuntimeStateConfig,
    ) -> Result<(
        VJepaEncoderOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        TttRuntimeStateMetrics,
    )> {
        match self {
            Self::Base(model) => Ok((
                model.encode_image_sparse_patchify_wgpu_batch(image, patchify_plan)?,
                TttRuntimeStateMetrics::default(),
            )),
            Self::Ttt {
                model,
                state,
                runtime,
            } => {
                let Some(mask) = patchify_plan.mask.uniform_mask() else {
                    bail!(
                        "TTT sparse patchify currently requires a uniform sparse mask batch; group variable masks or use dense patch embed"
                    );
                };
                let [batch, channels, height, width] = image.shape().dims::<4>();
                ensure!(
                    batch == patchify_plan.batch,
                    "image batch does not match sparse patchify batch plan"
                );
                let update_fast_weight =
                    begin_ttt_runtime_step(model, state, runtime, runtime_config)?;
                let output = model.forward_single_frame_rollout_sparse_patchify_wgpu_options(
                    image.reshape([batch, channels, 1, height, width]),
                    mask,
                    None,
                    state,
                    update_fast_weight,
                )?;
                let metrics = finish_ttt_runtime_step(
                    model,
                    state,
                    runtime,
                    &output,
                    batch as u64,
                    runtime_config,
                    update_fast_weight,
                )?;
                Ok((output, metrics))
            }
        }
    }
}

#[cfg(all(
    feature = "sparse-patchify-wgpu",
    any(not(target_arch = "wasm32"), feature = "wasm-fusion")
))]
impl FeatureFrameJepaEncoder<burn::backend::Wgpu<f32, i32>> {
    fn encode_image_sparse_patchify_wgpu_fusion_batch_with_runtime(
        &mut self,
        image: Tensor<burn::backend::Wgpu<f32, i32>, 4>,
        patchify_plan: &SparsePatchifyBatchPlan<burn::backend::Wgpu<f32, i32>>,
        runtime_config: &TttRuntimeStateConfig,
    ) -> Result<(
        VJepaEncoderOutput<burn::backend::Wgpu<f32, i32>>,
        TttRuntimeStateMetrics,
    )> {
        match self {
            Self::Base(model) => Ok((
                model.encode_image_sparse_patchify_wgpu_fusion_batch(image, patchify_plan)?,
                TttRuntimeStateMetrics::default(),
            )),
            Self::Ttt {
                model,
                state,
                runtime,
            } => {
                let batch = image.shape().dims::<4>()[0];
                let update_fast_weight =
                    begin_ttt_runtime_step(model, state, runtime, runtime_config)?;
                let output = model.forward_image_sparse_patchify_wgpu_fusion_batch_state_options(
                    image,
                    patchify_plan,
                    None,
                    state,
                    update_fast_weight,
                )?;
                let metrics = finish_ttt_runtime_step(
                    model,
                    state,
                    runtime,
                    &output,
                    batch as u64,
                    runtime_config,
                    update_fast_weight,
                )?;
                Ok((output, metrics))
            }
        }
    }
}

pub struct SparseJepaAnyUpPcaPipeline<B: Backend> {
    encoder: FeatureFrameJepaEncoder<B>,
    anyup: AnyUp<B>,
    anyup_image_grid: AnyUpImageGrid<B>,
    token_memory: InterframeJepaFeatureMemory<B>,
    pca: FeaturePcaProjector<B>,
    pca_update_scheduler: FeaturePcaUpdateScheduler,
    pca_samples: FeaturePcaSampleBuffer<B>,
    patch_diff_refresh_state: PatchDiffRefreshState,
    config: SparseJepaAnyUpPcaPipelineConfig,
    batch: usize,
    image_size: [usize; 2],
    grid: TokenGridShape,
    device: B::Device,
}

impl<B: Backend> SparseJepaAnyUpPcaPipeline<B> {
    pub fn new(
        jepa: VJepa2_1Model<B>,
        anyup: AnyUp<B>,
        jepa_config: &VJepaConfig,
        config: SparseJepaAnyUpPcaPipelineConfig,
        batch: usize,
        image_size: [usize; 2],
        device: &B::Device,
    ) -> Result<Self> {
        Self::new_with_encoder(
            FeatureFrameJepaEncoder::base(jepa),
            anyup,
            jepa_config,
            config,
            batch,
            image_size,
            device,
        )
    }

    pub fn new_with_encoder(
        encoder: FeatureFrameJepaEncoder<B>,
        anyup: AnyUp<B>,
        jepa_config: &VJepaConfig,
        config: SparseJepaAnyUpPcaPipelineConfig,
        batch: usize,
        image_size: [usize; 2],
        device: &B::Device,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(batch > 0, "pipeline batch must be nonzero");
        ensure!(
            encoder.config().encoder.embed_dim == jepa_config.encoder.embed_dim
                && encoder.config().patch_size == jepa_config.patch_size,
            "feature-frame encoder config must match the pipeline V-JEPA config"
        );
        ensure!(
            image_size[0] > 0 && image_size[1] > 0,
            "pipeline image size must be nonzero"
        );
        ensure!(
            image_size[0].is_multiple_of(jepa_config.patch_size.max(1))
                && image_size[1].is_multiple_of(jepa_config.patch_size.max(1)),
            "image size must be divisible by V-JEPA patch size"
        );
        let grid = TokenGridShape::new(
            1,
            image_size[0] / jepa_config.patch_size.max(1),
            image_size[1] / jepa_config.patch_size.max(1),
        );
        let token_memory = InterframeJepaFeatureMemory::new(
            config.memory,
            batch,
            grid,
            jepa_config.encoder.embed_dim,
            device,
        )?;
        let pca = FeaturePcaProjector::identity(
            jepa_config.encoder.embed_dim,
            config.pca.clone(),
            device,
        )?;
        let effective_pca_update = config.effective_pca_update();
        let pca_update_scheduler = FeaturePcaUpdateScheduler::new(effective_pca_update.clone())?;
        let pca_samples = FeaturePcaSampleBuffer::new(effective_pca_update.sample_window_frames)?;
        let anyup_image_grid = anyup.prepare_image_grid(image_size, device);
        Ok(Self {
            encoder,
            anyup,
            anyup_image_grid,
            token_memory,
            pca,
            pca_update_scheduler,
            pca_samples,
            patch_diff_refresh_state: PatchDiffRefreshState::default(),
            config,
            batch,
            image_size,
            grid,
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &SparseJepaAnyUpPcaPipelineConfig {
        &self.config
    }

    pub fn encoder(&self) -> &FeatureFrameJepaEncoder<B> {
        &self.encoder
    }

    pub fn encoder_kind(&self) -> FeatureFrameJepaEncoderKind {
        self.encoder.kind()
    }

    pub fn jepa(&self) -> Option<&VJepa2_1Model<B>> {
        self.encoder.base_model()
    }

    pub fn anyup(&self) -> &AnyUp<B> {
        &self.anyup
    }

    pub fn token_memory(&self) -> &InterframeJepaFeatureMemory<B> {
        &self.token_memory
    }

    pub fn token_memory_mut(&mut self) -> &mut InterframeJepaFeatureMemory<B> {
        &mut self.token_memory
    }

    pub fn pca(&self) -> &FeaturePcaProjector<B> {
        &self.pca
    }

    pub fn pca_mut(&mut self) -> &mut FeaturePcaProjector<B> {
        &mut self.pca
    }

    pub fn pca_update_scheduler(&self) -> &FeaturePcaUpdateScheduler {
        &self.pca_update_scheduler
    }

    pub fn patch_diff_refresh_state(&self) -> &PatchDiffRefreshState {
        &self.patch_diff_refresh_state
    }

    pub fn patch_diff_refresh_state_mut(&mut self) -> &mut PatchDiffRefreshState {
        &mut self.patch_diff_refresh_state
    }

    pub fn grid(&self) -> TokenGridShape {
        self.grid
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn image_size(&self) -> [usize; 2] {
        self.image_size
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }

    pub fn reset(&mut self) {
        self.encoder.reset_state();
        self.token_memory.reset();
        self.pca_update_scheduler.reset();
        self.pca_samples.reset();
        self.patch_diff_refresh_state.reset();
    }

    pub fn reset_visualization_state(&mut self) -> Result<()> {
        self.reset();
        let effective_pca_update = self.config.effective_pca_update();
        let feature_dim = self.pca.feature_dim();
        let pca_config = self.config.pca.clone();
        self.pca = FeaturePcaProjector::identity(feature_dim, pca_config, &self.device)?;
        self.pca_update_scheduler = FeaturePcaUpdateScheduler::new(effective_pca_update.clone())?;
        self.pca_samples = FeaturePcaSampleBuffer::new(effective_pca_update.sample_window_frames)?;
        Ok(())
    }

    pub fn set_pca_update_config(&mut self, pca_update: FeaturePcaUpdateConfig) -> Result<()> {
        pca_update.validate()?;
        self.config.pca_update = pca_update;
        let effective_pca_update = self.config.effective_pca_update();
        self.pca_update_scheduler = FeaturePcaUpdateScheduler::new(effective_pca_update.clone())?;
        self.pca_samples = FeaturePcaSampleBuffer::new(effective_pca_update.sample_window_frames)?;
        Ok(())
    }

    pub fn step_image_keep_ratio(
        &mut self,
        image: Tensor<B, 4>,
        keep_ratio: f32,
    ) -> Result<SparseJepaAnyUpPcaOutput<B>> {
        let mask = SparseTokenMask::from_keep_ratio(self.grid.len(), keep_ratio);
        self.step_image_with_mask(image, &mask)
    }

    pub fn step_image_with_mask(
        &mut self,
        image: Tensor<B, 4>,
        mask: &SparseTokenMask,
    ) -> Result<SparseJepaAnyUpPcaOutput<B>> {
        Ok(self.step_image_with_mask_measured(image, mask)?.output)
    }

    pub fn step_image_with_mask_measured(
        &mut self,
        image: Tensor<B, 4>,
        mask: &SparseTokenMask,
    ) -> Result<SparseJepaAnyUpPcaMeasuredOutput<B>> {
        let batch_mask = SparseMaskBatch::uniform(mask.clone(), self.batch, &self.device)?;
        let measured =
            self.step_image_with_mask_batch_measured(image, batch_mask, self.config.measurement)?;
        let batch_output = measured.output;
        Ok(SparseJepaAnyUpPcaMeasuredOutput {
            output: SparseJepaAnyUpPcaOutput {
                encoded: batch_output.encoded,
                token_cache: batch_output.token_cache,
                low_res_features: batch_output.low_res_features,
                high_res_features: batch_output.high_res_features,
                pca_display: batch_output.pca_display,
                mask: mask.clone(),
            },
            metrics: measured.metrics,
        })
    }

    pub fn step_image_with_mask_nodes_measured(
        &mut self,
        image: Tensor<B, 4>,
        mask: &SparseTokenMask,
        request: FeatureFrameRequest,
    ) -> Result<MeasuredFeatureFrameBatch<B>> {
        let batch_mask = SparseMaskBatch::uniform(mask.clone(), self.batch, &self.device)?;
        self.step_image_with_mask_batch_nodes_measured(
            image,
            batch_mask,
            request,
            self.config.measurement,
        )
    }

    pub fn step_image_with_encode_write_masks_nodes_measured(
        &mut self,
        image: Tensor<B, 4>,
        encode_mask: &SparseTokenMask,
        write_mask: &SparseTokenMask,
        request: FeatureFrameRequest,
    ) -> Result<MeasuredFeatureFrameBatch<B>> {
        if encode_mask == write_mask {
            return self.step_image_with_mask_nodes_measured(image, write_mask, request);
        }
        let encode_mask = SparseMaskBatch::uniform(encode_mask.clone(), self.batch, &self.device)?;
        let write_mask = SparseMaskBatch::uniform(write_mask.clone(), self.batch, &self.device)?;
        self.step_image_with_encode_write_mask_batch_nodes_measured(
            image,
            encode_mask,
            write_mask,
            request,
            self.config.measurement,
        )
    }

    pub fn step_image_with_mask_batch(
        &mut self,
        image: Tensor<B, 4>,
        mask: SparseMaskBatch<B>,
    ) -> Result<SparseJepaAnyUpPcaStepBatchOutput<B>> {
        Ok(self
            .step_image_with_mask_batch_measured(
                image,
                mask,
                SparseJepaAnyUpPcaMeasurementConfig::disabled(),
            )?
            .output)
    }

    fn validate_batch_step_input(
        &self,
        image: &Tensor<B, 4>,
        mask: &SparseMaskBatch<B>,
    ) -> Result<[usize; 4]> {
        ensure!(
            mask.dense_len() == self.grid.len(),
            "sparse token mask dense length must match pipeline grid"
        );
        ensure!(
            !mask.is_empty(),
            "sparse token mask must include at least one token"
        );
        ensure!(
            !mask.is_ragged(),
            "high-res pipeline requires uniform or fixed-width mask batches; group variable-width masks before this stage"
        );
        let [batch, channels, height, width] = image.shape().dims::<4>();
        ensure!(batch == self.batch, "image batch does not match pipeline");
        ensure!(channels == 3, "pipeline expects RGB image input");
        ensure!(
            [height, width] == self.image_size,
            "image spatial size does not match pipeline"
        );
        Ok([batch, channels, height, width])
    }

    fn initial_stage_metrics(
        &self,
        batch: usize,
        height: usize,
        width: usize,
        mask: &SparseMaskBatch<B>,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
        encode_path: SparseJepaAnyUpPcaEncodePath,
    ) -> SparseJepaAnyUpPcaStageMetrics {
        SparseJepaAnyUpPcaStageMetrics {
            measured: measurement.enabled,
            sync_backend: measurement.sync_backend,
            encode_path,
            frame_count: batch,
            dense_tokens_per_frame: self.grid.len(),
            sparse_width: mask.len(),
            valid_sparse_tokens: mask.valid_token_count(),
            write_width: mask.len(),
            valid_write_tokens: mask.valid_token_count(),
            encode_width: mask.len(),
            valid_encode_tokens: mask.valid_token_count(),
            output_pixels: batch * height * width,
            ..SparseJepaAnyUpPcaStageMetrics::default()
        }
    }

    fn finish_encoded_batch_nodes(
        &mut self,
        image: Tensor<B, 4>,
        mask: SparseMaskBatch<B>,
        encoded: VJepaEncoderOutput<B>,
        request: FeatureFrameRequest,
        mut timer: StageTimer,
        mut metrics: SparseJepaAnyUpPcaStageMetrics,
    ) -> Result<MeasuredFeatureFrameBatch<B>> {
        let token_cache = if mask.is_dense_ordered() {
            self.token_memory
                .update_dense_ordered_tokens(encoded.tokens.clone(), encoded.grid)?
        } else {
            self.token_memory.update_tokens(
                encoded.tokens.clone(),
                encoded.token_indices.clone(),
                encoded.grid,
            )?
        };
        metrics.cache_update_us = timer.mark::<B>(&self.device)?;
        self.finish_encoded_batch_nodes_with_cache(
            image,
            mask,
            encoded,
            token_cache,
            request,
            timer,
            metrics,
        )
    }

    fn finish_encoded_batch_nodes_with_cache(
        &mut self,
        image: Tensor<B, 4>,
        mask: SparseMaskBatch<B>,
        encoded: VJepaEncoderOutput<B>,
        token_cache: InterframeJepaFeatureMemoryOutput<B>,
        request: FeatureFrameRequest,
        mut timer: StageTimer,
        mut metrics: SparseJepaAnyUpPcaStageMetrics,
    ) -> Result<MeasuredFeatureFrameBatch<B>> {
        let low_res_features =
            jepa_feature_tokens_to_nchw(token_cache.features.clone(), self.grid)?;
        metrics.token_view_us = timer.mark::<B>(&self.device)?;

        let pca_update = self
            .pca_update_scheduler
            .observe_batch(metrics.frame_count, self.grid.len());
        if self.pca_update_scheduler.config().enabled() {
            self.pca_samples.push(
                token_cache.features.clone(),
                token_cache.observed.clone(),
                metrics.frame_count,
            )?;
            metrics.pca_sample_window_frames =
                self.pca_update_scheduler.config().sample_window_frames;
            metrics.pca_sample_frames = self.pca_samples.frame_count();
        }
        if pca_update.update {
            let iterations = self.pca_update_scheduler.config().iterations_per_update;
            let Some(pca_samples) = self.pca_samples.snapshot() else {
                bail!("PCA update requested before any samples were buffered");
            };
            self.pca.update_rolling_masked_tokens_iterations(
                pca_samples.features,
                pca_samples.observed,
                iterations,
            )?;
            metrics.pca_update_us = timer.mark::<B>(&self.device)?;
            metrics.pca_online_us = metrics.pca_update_us;
            metrics.pca_update_applied = true;
            metrics.pca_sample_frames = pca_samples.frame_count;
            metrics.pca_update_tokens = pca_samples.frame_count * self.grid.len();
        }

        let mut low_res_pca_components = None;
        let low_res_pca_display = if request.low_res_pca {
            let pca_components = self.pca.project_nchw(low_res_features.clone())?;
            let pca = self.pca.display_nchw(pca_components.clone())?;
            metrics.low_res_pca_project_us = timer.mark::<B>(&self.device)?;
            low_res_pca_components = Some(pca_components);
            Some(pca)
        } else {
            None
        };

        let high_res = if request.high_res_pca || request.high_res_features {
            let context = self.anyup.prepare_image_context_with_grid(
                image,
                &self.anyup_image_grid,
                Some(self.image_size),
                [self.grid.height, self.grid.width],
            );
            metrics.anyup_context_us = timer.mark::<B>(&self.device)?;

            let (features, pca_display) = if request.high_res_features {
                let features = self.anyup.upsample_with_context(
                    &context,
                    low_res_features.clone(),
                    self.config.anyup_q_chunk_size,
                );
                metrics.anyup_decode_us = timer.mark::<B>(&self.device)?;
                let pca_display = if request.high_res_pca {
                    let pca_display = self.pca.project_nchw_display(features.clone())?;
                    metrics.pca_project_us = timer.mark::<B>(&self.device)?;
                    Some(pca_display)
                } else {
                    None
                };
                (Some(features), pca_display)
            } else {
                let pca_values = if let Some(pca_components) = low_res_pca_components {
                    pca_components
                } else {
                    let pca_values = self.pca.project_nchw(low_res_features.clone())?;
                    metrics.pca_project_us = metrics
                        .pca_project_us
                        .saturating_add(timer.mark::<B>(&self.device)?);
                    pca_values
                };
                let pca_values = self.anyup.upsample_values_with_context(
                    &context,
                    low_res_features.clone(),
                    pca_values,
                    self.config.anyup_q_chunk_size,
                );
                metrics.anyup_decode_us = timer.mark::<B>(&self.device)?;
                let pca_display = self.pca.display_nchw(pca_values)?;
                metrics.pca_project_us = metrics
                    .pca_project_us
                    .saturating_add(timer.mark::<B>(&self.device)?);
                (None, Some(pca_display))
            };
            Some(HighResFrameArtifacts {
                features,
                pca_display,
            })
        } else {
            None
        };

        metrics.total_us = timer.total_us();
        Ok(MeasuredFeatureFrameBatch {
            output: FeatureFrameBatch {
                encoded,
                token_cache,
                low_res: LowResFrameArtifacts {
                    features: low_res_features,
                    pca_display: low_res_pca_display,
                },
                high_res,
                mask,
            },
            metrics,
        })
    }

    fn finish_encoded_batch_step(
        &mut self,
        image: Tensor<B, 4>,
        mask: SparseMaskBatch<B>,
        encoded: VJepaEncoderOutput<B>,
        timer: StageTimer,
        metrics: SparseJepaAnyUpPcaStageMetrics,
    ) -> Result<SparseJepaAnyUpPcaMeasuredBatchOutput<B>> {
        let measured = self.finish_encoded_batch_nodes(
            image,
            mask,
            encoded,
            FeatureFrameRequest::high_res(),
            timer,
            metrics,
        )?;
        Ok(SparseJepaAnyUpPcaMeasuredBatchOutput {
            output: measured.output.into_full_output()?,
            metrics: measured.metrics,
        })
    }

    pub fn step_image_with_mask_batch_nodes_measured(
        &mut self,
        image: Tensor<B, 4>,
        mask: SparseMaskBatch<B>,
        request: FeatureFrameRequest,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<MeasuredFeatureFrameBatch<B>> {
        let [batch, _channels, height, width] = self.validate_batch_step_input(&image, &mask)?;
        let mut metrics = self.initial_stage_metrics(
            batch,
            height,
            width,
            &mask,
            measurement,
            SparseJepaAnyUpPcaEncodePath::DensePatchEmbed,
        );
        let mut timer = StageTimer::new(measurement);

        let (encoded, ttt_runtime) = self.encoder.encode_image_batch_with_runtime(
            image.clone(),
            mask.clone(),
            &self.config.ttt_runtime,
        )?;
        metrics.ttt_runtime = ttt_runtime;
        metrics.encode_us = timer.mark::<B>(&self.device)?;
        self.finish_encoded_batch_nodes(image, mask, encoded, request, timer, metrics)
    }

    pub fn step_image_with_encode_write_mask_batch_nodes_measured(
        &mut self,
        image: Tensor<B, 4>,
        encode_mask: SparseMaskBatch<B>,
        write_mask: SparseMaskBatch<B>,
        request: FeatureFrameRequest,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<MeasuredFeatureFrameBatch<B>> {
        let [batch, _channels, height, width] =
            self.validate_batch_step_input(&image, &encode_mask)?;
        ensure!(
            write_mask.batch() == batch && write_mask.dense_len() == encode_mask.dense_len(),
            "write mask batch and dense length must match the encode mask"
        );
        let mut metrics = self.initial_stage_metrics(
            batch,
            height,
            width,
            &write_mask,
            measurement,
            SparseJepaAnyUpPcaEncodePath::DensePatchEmbed,
        );
        metrics.set_encode_mask(&encode_mask);
        let mut timer = StageTimer::new(measurement);

        let (encoded, ttt_runtime) = self.encoder.encode_image_batch_with_runtime(
            image.clone(),
            encode_mask.clone(),
            &self.config.ttt_runtime,
        )?;
        metrics.ttt_runtime = ttt_runtime;
        metrics.encode_us = timer.mark::<B>(&self.device)?;
        let encoded =
            restrict_encoded_to_write_mask(encoded, &encode_mask, &write_mask, &self.device)?;
        self.finish_encoded_batch_nodes(image, write_mask, encoded, request, timer, metrics)
    }

    pub fn step_image_with_mask_batch_measured(
        &mut self,
        image: Tensor<B, 4>,
        mask: SparseMaskBatch<B>,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<SparseJepaAnyUpPcaMeasuredBatchOutput<B>> {
        let [batch, _channels, height, width] = self.validate_batch_step_input(&image, &mask)?;
        let mut metrics = self.initial_stage_metrics(
            batch,
            height,
            width,
            &mask,
            measurement,
            SparseJepaAnyUpPcaEncodePath::DensePatchEmbed,
        );
        let mut timer = StageTimer::new(measurement);

        let (encoded, ttt_runtime) = self.encoder.encode_image_batch_with_runtime(
            image.clone(),
            mask.clone(),
            &self.config.ttt_runtime,
        )?;
        metrics.ttt_runtime = ttt_runtime;
        metrics.encode_us = timer.mark::<B>(&self.device)?;
        self.finish_encoded_batch_step(image, mask, encoded, timer, metrics)
    }
}

fn restrict_encoded_to_write_mask<B: Backend>(
    encoded: VJepaEncoderOutput<B>,
    encode_mask: &SparseMaskBatch<B>,
    write_mask: &SparseMaskBatch<B>,
    device: &B::Device,
) -> Result<VJepaEncoderOutput<B>> {
    ensure!(
        encode_mask.batch() == write_mask.batch()
            && encode_mask.dense_len() == write_mask.dense_len(),
        "encode and write masks must have matching batch and dense lengths"
    );
    ensure!(
        !encode_mask.is_ragged() && !write_mask.is_ragged(),
        "separate encode/write masks currently require uniform or fixed-width mask batches"
    );
    let [batch, encode_width, _embed_dim] = encoded.tokens.shape().dims::<3>();
    ensure!(
        batch == encode_mask.batch() && encode_width == encode_mask.len(),
        "encoded token shape must match the encode mask"
    );

    let encode_rows = encode_mask.rows();
    let write_rows = write_mask.rows();
    let write_width = write_mask.len();
    ensure!(
        write_rows.iter().all(|row| row.len() == write_width),
        "write mask rows must have a fixed width"
    );

    let mut position_rows = Vec::with_capacity(batch);
    for (row_index, (encode_row, write_row)) in
        encode_rows.iter().zip(write_rows.iter()).enumerate()
    {
        ensure!(
            encode_row.len() == encode_width,
            "encode mask row {row_index} width does not match encoded tokens"
        );
        let mut positions = Vec::with_capacity(write_width);
        for &token_index in write_row {
            let Ok(position) = encode_row.binary_search(&token_index) else {
                bail!(
                    "write mask token {token_index} at batch row {row_index} is absent from the encode mask"
                );
            };
            positions.push(position);
        }
        position_rows.push(positions);
    }

    let positions = SparseMaskBatch::from_rows(position_rows, encode_width, device)?.indices();
    let tokens = apply_token_mask(encoded.tokens, positions.clone());
    let hierarchical = encoded
        .hierarchical
        .into_iter()
        .map(|tokens| apply_token_mask(tokens, positions.clone()))
        .collect();
    Ok(VJepaEncoderOutput {
        tokens,
        hierarchical,
        captured_layers: encoded.captured_layers,
        token_indices: write_mask.indices(),
        grid: encoded.grid,
    })
}

#[cfg(feature = "sparse-patchify-wgpu")]
impl SparseJepaAnyUpPcaPipeline<burn_flex_gmm::wgpu::DefaultWgpuBackend> {
    pub fn step_image_with_mask_sparse_patchify_wgpu(
        &mut self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        mask: &SparseTokenMask,
    ) -> Result<SparseJepaAnyUpPcaOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        Ok(self
            .step_image_with_mask_sparse_patchify_wgpu_measured(image, mask)?
            .output)
    }

    pub fn step_image_with_mask_sparse_patchify_wgpu_measured(
        &mut self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        mask: &SparseTokenMask,
    ) -> Result<SparseJepaAnyUpPcaMeasuredOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        let batch_mask = SparseMaskBatch::uniform(mask.clone(), self.batch, &self.device)?;
        let measured = self.step_image_with_mask_batch_sparse_patchify_wgpu_measured(
            image,
            batch_mask,
            self.config.measurement,
        )?;
        let batch_output = measured.output;
        Ok(SparseJepaAnyUpPcaMeasuredOutput {
            output: SparseJepaAnyUpPcaOutput {
                encoded: batch_output.encoded,
                token_cache: batch_output.token_cache,
                low_res_features: batch_output.low_res_features,
                high_res_features: batch_output.high_res_features,
                pca_display: batch_output.pca_display,
                mask: mask.clone(),
            },
            metrics: measured.metrics,
        })
    }

    pub fn step_image_with_mask_batch_sparse_patchify_wgpu(
        &mut self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        mask: SparseMaskBatch<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    ) -> Result<SparseJepaAnyUpPcaStepBatchOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        Ok(self
            .step_image_with_mask_batch_sparse_patchify_wgpu_measured(
                image,
                mask,
                SparseJepaAnyUpPcaMeasurementConfig::disabled(),
            )?
            .output)
    }

    pub fn step_image_with_mask_batch_sparse_patchify_wgpu_measured(
        &mut self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        mask: SparseMaskBatch<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<SparseJepaAnyUpPcaMeasuredBatchOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>>
    {
        let patchify_plan = SparsePatchifyBatchPlan::new(mask.clone(), self.grid, &self.device)?;
        self.step_image_with_sparse_patchify_plan_wgpu_measured(image, &patchify_plan, measurement)
    }

    pub fn step_image_with_sparse_patchify_plan_wgpu_measured(
        &mut self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        patchify_plan: &SparsePatchifyBatchPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<SparseJepaAnyUpPcaMeasuredBatchOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>>
    {
        let measured = self.step_image_with_sparse_patchify_plan_wgpu_nodes_measured(
            image,
            patchify_plan,
            FeatureFrameRequest::high_res(),
            measurement,
        )?;
        Ok(SparseJepaAnyUpPcaMeasuredBatchOutput {
            output: measured.output.into_full_output()?,
            metrics: measured.metrics,
        })
    }

    pub fn step_image_with_sparse_patchify_plan_wgpu_nodes_measured(
        &mut self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        patchify_plan: &SparsePatchifyBatchPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        request: FeatureFrameRequest,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<MeasuredFeatureFrameBatch<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        ensure!(
            patchify_plan.grid == self.grid && patchify_plan.batch == self.batch,
            "sparse patchify plan must match the high-res pipeline grid and batch"
        );
        let mask = patchify_plan.mask.clone();
        let [batch, _channels, height, width] = self.validate_batch_step_input(&image, &mask)?;
        let mut metrics = self.initial_stage_metrics(
            batch,
            height,
            width,
            &mask,
            measurement,
            SparseJepaAnyUpPcaEncodePath::SparsePatchify,
        );
        let mut timer = StageTimer::new(measurement);
        let (encoded, ttt_runtime) = self
            .encoder
            .encode_image_sparse_patchify_wgpu_batch_with_runtime(
                image.clone(),
                patchify_plan,
                &self.config.ttt_runtime,
            )?;
        metrics.ttt_runtime = ttt_runtime;
        metrics.encode_us = timer.mark::<burn_flex_gmm::wgpu::DefaultWgpuBackend>(&self.device)?;
        let token_cache = if mask.is_dense_ordered() {
            self.token_memory
                .update_dense_ordered_tokens(encoded.tokens.clone(), encoded.grid)?
        } else {
            self.token_memory.update_tokens_tiled_assign_wgpu_raw(
                encoded.tokens.clone(),
                encoded.token_indices.clone(),
                encoded.grid,
            )?
        };
        metrics.cache_update_us =
            timer.mark::<burn_flex_gmm::wgpu::DefaultWgpuBackend>(&self.device)?;
        self.finish_encoded_batch_nodes_with_cache(
            image,
            mask,
            encoded,
            token_cache,
            request,
            timer,
            metrics,
        )
    }

    pub fn step_image_with_sparse_patchify_plan_wgpu_nodes_measured_with_write_mask(
        &mut self,
        image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
        patchify_plan: &SparsePatchifyBatchPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        write_mask: SparseMaskBatch<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        request: FeatureFrameRequest,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<MeasuredFeatureFrameBatch<burn_flex_gmm::wgpu::DefaultWgpuBackend>> {
        ensure!(
            patchify_plan.grid == self.grid && patchify_plan.batch == self.batch,
            "sparse patchify plan must match the high-res pipeline grid and batch"
        );
        let encode_mask = patchify_plan.mask.clone();
        let [batch, _channels, height, width] =
            self.validate_batch_step_input(&image, &encode_mask)?;
        ensure!(
            write_mask.batch() == batch && write_mask.dense_len() == encode_mask.dense_len(),
            "write mask batch and dense length must match the sparse patchify plan"
        );
        let mut metrics = self.initial_stage_metrics(
            batch,
            height,
            width,
            &write_mask,
            measurement,
            SparseJepaAnyUpPcaEncodePath::SparsePatchify,
        );
        metrics.set_encode_mask(&encode_mask);
        let mut timer = StageTimer::new(measurement);
        let (encoded, ttt_runtime) = self
            .encoder
            .encode_image_sparse_patchify_wgpu_batch_with_runtime(
                image.clone(),
                patchify_plan,
                &self.config.ttt_runtime,
            )?;
        metrics.ttt_runtime = ttt_runtime;
        metrics.encode_us = timer.mark::<burn_flex_gmm::wgpu::DefaultWgpuBackend>(&self.device)?;
        let encoded =
            restrict_encoded_to_write_mask(encoded, &encode_mask, &write_mask, &self.device)?;
        let token_cache = if write_mask.is_dense_ordered() {
            self.token_memory
                .update_dense_ordered_tokens(encoded.tokens.clone(), encoded.grid)?
        } else {
            self.token_memory.update_tokens_tiled_assign_wgpu_raw(
                encoded.tokens.clone(),
                encoded.token_indices.clone(),
                encoded.grid,
            )?
        };
        metrics.cache_update_us =
            timer.mark::<burn_flex_gmm::wgpu::DefaultWgpuBackend>(&self.device)?;
        self.finish_encoded_batch_nodes_with_cache(
            image,
            write_mask,
            encoded,
            token_cache,
            request,
            timer,
            metrics,
        )
    }
}

#[cfg(all(
    feature = "sparse-patchify-wgpu",
    any(not(target_arch = "wasm32"), feature = "wasm-fusion")
))]
impl SparseJepaAnyUpPcaPipeline<burn::backend::Wgpu<f32, i32>> {
    pub fn step_image_with_sparse_patchify_plan_wgpu_nodes_measured(
        &mut self,
        image: Tensor<burn::backend::Wgpu<f32, i32>, 4>,
        patchify_plan: &SparsePatchifyBatchPlan<burn::backend::Wgpu<f32, i32>>,
        request: FeatureFrameRequest,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<MeasuredFeatureFrameBatch<burn::backend::Wgpu<f32, i32>>> {
        ensure!(
            patchify_plan.grid == self.grid && patchify_plan.batch == self.batch,
            "sparse patchify plan must match the high-res pipeline grid and batch"
        );
        let mask = patchify_plan.mask.clone();
        let [batch, _channels, height, width] = self.validate_batch_step_input(&image, &mask)?;
        let mut metrics = self.initial_stage_metrics(
            batch,
            height,
            width,
            &mask,
            measurement,
            SparseJepaAnyUpPcaEncodePath::SparsePatchify,
        );
        let mut timer = StageTimer::new(measurement);
        let (encoded, ttt_runtime) = self
            .encoder
            .encode_image_sparse_patchify_wgpu_fusion_batch_with_runtime(
                image.clone(),
                patchify_plan,
                &self.config.ttt_runtime,
            )?;
        metrics.ttt_runtime = ttt_runtime;
        metrics.encode_us = timer.mark::<burn::backend::Wgpu<f32, i32>>(&self.device)?;
        let token_cache = if mask.is_dense_ordered() {
            self.token_memory
                .update_dense_ordered_tokens(encoded.tokens.clone(), encoded.grid)?
        } else {
            self.token_memory.update_tokens_tiled_assign_wgpu(
                encoded.tokens.clone(),
                encoded.token_indices.clone(),
                encoded.grid,
            )?
        };
        metrics.cache_update_us = timer.mark::<burn::backend::Wgpu<f32, i32>>(&self.device)?;
        self.finish_encoded_batch_nodes_with_cache(
            image,
            mask,
            encoded,
            token_cache,
            request,
            timer,
            metrics,
        )
    }

    pub fn step_image_with_sparse_patchify_plan_wgpu_nodes_measured_with_write_mask(
        &mut self,
        image: Tensor<burn::backend::Wgpu<f32, i32>, 4>,
        patchify_plan: &SparsePatchifyBatchPlan<burn::backend::Wgpu<f32, i32>>,
        write_mask: SparseMaskBatch<burn::backend::Wgpu<f32, i32>>,
        request: FeatureFrameRequest,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<MeasuredFeatureFrameBatch<burn::backend::Wgpu<f32, i32>>> {
        ensure!(
            patchify_plan.grid == self.grid && patchify_plan.batch == self.batch,
            "sparse patchify plan must match the high-res pipeline grid and batch"
        );
        let encode_mask = patchify_plan.mask.clone();
        let [batch, _channels, height, width] =
            self.validate_batch_step_input(&image, &encode_mask)?;
        ensure!(
            write_mask.batch() == batch && write_mask.dense_len() == encode_mask.dense_len(),
            "write mask batch and dense length must match the sparse patchify plan"
        );
        let mut metrics = self.initial_stage_metrics(
            batch,
            height,
            width,
            &write_mask,
            measurement,
            SparseJepaAnyUpPcaEncodePath::SparsePatchify,
        );
        metrics.set_encode_mask(&encode_mask);
        let mut timer = StageTimer::new(measurement);
        let (encoded, ttt_runtime) = self
            .encoder
            .encode_image_sparse_patchify_wgpu_fusion_batch_with_runtime(
                image.clone(),
                patchify_plan,
                &self.config.ttt_runtime,
            )?;
        metrics.ttt_runtime = ttt_runtime;
        metrics.encode_us = timer.mark::<burn::backend::Wgpu<f32, i32>>(&self.device)?;
        let encoded =
            restrict_encoded_to_write_mask(encoded, &encode_mask, &write_mask, &self.device)?;
        let token_cache = if write_mask.is_dense_ordered() {
            self.token_memory
                .update_dense_ordered_tokens(encoded.tokens.clone(), encoded.grid)?
        } else {
            self.token_memory.update_tokens_tiled_assign_wgpu(
                encoded.tokens.clone(),
                encoded.token_indices.clone(),
                encoded.grid,
            )?
        };
        metrics.cache_update_us = timer.mark::<burn::backend::Wgpu<f32, i32>>(&self.device)?;
        self.finish_encoded_batch_nodes_with_cache(
            image,
            write_mask,
            encoded,
            token_cache,
            request,
            timer,
            metrics,
        )
    }
}

#[cfg(feature = "sparse-patchify-cuda")]
impl SparseJepaAnyUpPcaPipeline<burn_flex_gmm::cuda::DefaultCudaBackend> {
    pub fn step_image_with_mask_sparse_patchify_cuda(
        &mut self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        mask: &SparseTokenMask,
    ) -> Result<SparseJepaAnyUpPcaOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        Ok(self
            .step_image_with_mask_sparse_patchify_cuda_measured(image, mask)?
            .output)
    }

    pub fn step_image_with_mask_sparse_patchify_cuda_measured(
        &mut self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        mask: &SparseTokenMask,
    ) -> Result<SparseJepaAnyUpPcaMeasuredOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        let batch_mask = SparseMaskBatch::uniform(mask.clone(), self.batch, &self.device)?;
        let measured = self.step_image_with_mask_batch_sparse_patchify_cuda_measured(
            image,
            batch_mask,
            self.config.measurement,
        )?;
        let batch_output = measured.output;
        Ok(SparseJepaAnyUpPcaMeasuredOutput {
            output: SparseJepaAnyUpPcaOutput {
                encoded: batch_output.encoded,
                token_cache: batch_output.token_cache,
                low_res_features: batch_output.low_res_features,
                high_res_features: batch_output.high_res_features,
                pca_display: batch_output.pca_display,
                mask: mask.clone(),
            },
            metrics: measured.metrics,
        })
    }

    pub fn step_image_with_mask_batch_sparse_patchify_cuda(
        &mut self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        mask: SparseMaskBatch<burn_flex_gmm::cuda::DefaultCudaBackend>,
    ) -> Result<SparseJepaAnyUpPcaStepBatchOutput<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        Ok(self
            .step_image_with_mask_batch_sparse_patchify_cuda_measured(
                image,
                mask,
                SparseJepaAnyUpPcaMeasurementConfig::disabled(),
            )?
            .output)
    }

    pub fn step_image_with_mask_batch_sparse_patchify_cuda_measured(
        &mut self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        mask: SparseMaskBatch<burn_flex_gmm::cuda::DefaultCudaBackend>,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<SparseJepaAnyUpPcaMeasuredBatchOutput<burn_flex_gmm::cuda::DefaultCudaBackend>>
    {
        let patchify_plan = SparsePatchifyBatchPlan::new(mask.clone(), self.grid, &self.device)?;
        self.step_image_with_sparse_patchify_plan_cuda_measured(image, &patchify_plan, measurement)
    }

    pub fn step_image_with_sparse_patchify_plan_cuda_measured(
        &mut self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        patchify_plan: &SparsePatchifyBatchPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<SparseJepaAnyUpPcaMeasuredBatchOutput<burn_flex_gmm::cuda::DefaultCudaBackend>>
    {
        let measured = self.step_image_with_sparse_patchify_plan_cuda_nodes_measured(
            image,
            patchify_plan,
            FeatureFrameRequest::high_res(),
            measurement,
        )?;
        Ok(SparseJepaAnyUpPcaMeasuredBatchOutput {
            output: measured.output.into_full_output()?,
            metrics: measured.metrics,
        })
    }

    pub fn step_image_with_sparse_patchify_plan_cuda_nodes_measured(
        &mut self,
        image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
        patchify_plan: &SparsePatchifyBatchPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
        request: FeatureFrameRequest,
        measurement: SparseJepaAnyUpPcaMeasurementConfig,
    ) -> Result<MeasuredFeatureFrameBatch<burn_flex_gmm::cuda::DefaultCudaBackend>> {
        ensure!(
            patchify_plan.grid == self.grid && patchify_plan.batch == self.batch,
            "sparse patchify plan must match the high-res pipeline grid and batch"
        );
        let mask = patchify_plan.mask.clone();
        let [batch, _channels, height, width] = self.validate_batch_step_input(&image, &mask)?;
        let mut metrics = self.initial_stage_metrics(
            batch,
            height,
            width,
            &mask,
            measurement,
            SparseJepaAnyUpPcaEncodePath::SparsePatchify,
        );
        let mut timer = StageTimer::new(measurement);
        let encoded = self
            .encoder
            .base_model()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "sparse-patchify CUDA high-res path currently requires the base V-JEPA encoder; use the dense-patch TTT path for trained TTT inference"
                )
            })?
            .encode_image_sparse_patchify_cuda_batch(image.clone(), patchify_plan)?;
        metrics.encode_us = timer.mark::<burn_flex_gmm::cuda::DefaultCudaBackend>(&self.device)?;
        let token_cache = if mask.is_dense_ordered() {
            self.token_memory
                .update_dense_ordered_tokens(encoded.tokens.clone(), encoded.grid)?
        } else {
            self.token_memory.update_tokens_tiled_assign_cuda_raw(
                encoded.tokens.clone(),
                encoded.token_indices.clone(),
                encoded.grid,
            )?
        };
        metrics.cache_update_us =
            timer.mark::<burn_flex_gmm::cuda::DefaultCudaBackend>(&self.device)?;
        self.finish_encoded_batch_nodes_with_cache(
            image,
            mask,
            encoded,
            token_cache,
            request,
            timer,
            metrics,
        )
    }
}

struct FeaturePcaSamples<B: Backend> {
    features: Tensor<B, 3>,
    observed: Tensor<B, 2>,
    frame_count: usize,
}

struct FeaturePcaSampleEntry<B: Backend> {
    features: Tensor<B, 3>,
    observed: Tensor<B, 2>,
    frame_count: usize,
}

struct FeaturePcaSampleBuffer<B: Backend> {
    window_frames: usize,
    frame_count: usize,
    entries: VecDeque<FeaturePcaSampleEntry<B>>,
}

impl<B: Backend> FeaturePcaSampleBuffer<B> {
    fn new(window_frames: usize) -> Result<Self> {
        ensure!(window_frames > 0, "PCA sample window must be nonzero");
        Ok(Self {
            window_frames,
            frame_count: 0,
            entries: VecDeque::new(),
        })
    }

    fn reset(&mut self) {
        self.frame_count = 0;
        self.entries.clear();
    }

    fn frame_count(&self) -> usize {
        self.frame_count
    }

    fn push(
        &mut self,
        features: Tensor<B, 3>,
        observed: Tensor<B, 2>,
        frame_count: usize,
    ) -> Result<()> {
        let [feature_batch, _, _] = features.shape().dims::<3>();
        let [observed_batch, _] = observed.shape().dims::<2>();
        ensure!(
            frame_count > 0 && feature_batch == frame_count && observed_batch == frame_count,
            "PCA sample batch must match frame count"
        );
        self.entries.push_back(FeaturePcaSampleEntry {
            features,
            observed,
            frame_count,
        });
        self.frame_count = self.frame_count.saturating_add(frame_count);
        self.trim();
        Ok(())
    }

    fn snapshot(&self) -> Option<FeaturePcaSamples<B>> {
        if self.entries.is_empty() {
            return None;
        }
        Some(FeaturePcaSamples {
            features: Tensor::cat(
                self.entries
                    .iter()
                    .map(|entry| entry.features.clone())
                    .collect::<Vec<_>>(),
                0,
            ),
            observed: Tensor::cat(
                self.entries
                    .iter()
                    .map(|entry| entry.observed.clone())
                    .collect::<Vec<_>>(),
                0,
            ),
            frame_count: self.frame_count,
        })
    }

    fn trim(&mut self) {
        while self.frame_count > self.window_frames {
            let excess = self.frame_count - self.window_frames;
            let Some(front_count) = self.entries.front().map(|entry| entry.frame_count) else {
                self.frame_count = 0;
                return;
            };
            if front_count <= excess {
                self.frame_count -= front_count;
                self.entries.pop_front();
                continue;
            }

            let front = self
                .entries
                .front_mut()
                .expect("front entry exists after previous check");
            let [_, tokens, dim] = front.features.shape().dims::<3>();
            front.features =
                front
                    .features
                    .clone()
                    .slice([excess..front.frame_count, 0..tokens, 0..dim]);
            front.observed = front
                .observed
                .clone()
                .slice([excess..front.frame_count, 0..tokens]);
            front.frame_count -= excess;
            self.frame_count -= excess;
        }
    }
}

struct StageTimer {
    measurement: SparseJepaAnyUpPcaMeasurementConfig,
    start: Option<PipelineInstant>,
    last: Option<PipelineInstant>,
}

impl StageTimer {
    fn new(measurement: SparseJepaAnyUpPcaMeasurementConfig) -> Self {
        let now = measurement.enabled.then(pipeline_now);
        Self {
            measurement,
            start: now,
            last: now,
        }
    }

    fn mark<B: Backend>(&mut self, device: &B::Device) -> Result<u64> {
        if !self.measurement.enabled {
            return Ok(0);
        }
        if self.measurement.sync_backend {
            B::sync(device)?;
        }
        let now = pipeline_now();
        let elapsed = self
            .last
            .replace(now)
            .map(|last| pipeline_delta_us(now, last))
            .unwrap_or(0);
        Ok(elapsed)
    }

    fn total_us(&self) -> u64 {
        self.start.map(pipeline_elapsed_us).unwrap_or(0)
    }
}

pub fn measure_feature_token_stability<B: Backend>(
    tokens: Tensor<B, 3>,
) -> Result<FeatureTokenStabilityMetrics> {
    let [_batch, token_count, dim] = tokens.shape().dims::<3>();
    if token_count == 0 || dim == 0 {
        return Ok(FeatureTokenStabilityMetrics::default());
    }
    let values = tokens
        .to_data()
        .to_vec::<f32>()
        .context("read token stability diagnostics")?;
    measure_feature_token_stability_values(&values, token_count, dim)
}

fn measure_ttt_state_fast_weight_rms<B: Backend>(state: &TttState<B>) -> Result<Option<f64>> {
    let mut sum_sq = 0.0f64;
    let mut count = 0usize;
    for layer in &state.layers {
        if let Some(weight) = &layer.fast_weight {
            let values = weight
                .clone()
                .to_data()
                .to_vec::<f32>()
                .context("read TTT fast-weight diagnostics")?;
            count += values.len();
            for value in values {
                let value = value as f64;
                sum_sq += value * value;
            }
        }
        if let Some(weight) = &layer.fast_weight_banks {
            let values = weight
                .clone()
                .to_data()
                .to_vec::<f32>()
                .context("read TTT banked fast-weight diagnostics")?;
            count += values.len();
            for value in values {
                let value = value as f64;
                sum_sq += value * value;
            }
        }
    }
    Ok((count > 0).then(|| (sum_sq / count as f64).sqrt()))
}

fn measure_feature_token_stability_values(
    values: &[f32],
    token_count: usize,
    dim: usize,
) -> Result<FeatureTokenStabilityMetrics> {
    ensure!(
        token_count > 0 && dim > 0 && values.len().is_multiple_of(token_count * dim),
        "token stability diagnostics require a non-empty [batch, tokens, dim] buffer"
    );
    let batch = values.len() / (token_count * dim);
    let mut spatial_var_sum = 0.0;
    for b in 0..batch {
        let base = b * token_count * dim;
        for d in 0..dim {
            let mut mean = 0.0;
            for t in 0..token_count {
                mean += values[base + t * dim + d] as f64;
            }
            mean /= token_count as f64;
            let mut var = 0.0;
            for t in 0..token_count {
                let diff = values[base + t * dim + d] as f64 - mean;
                var += diff * diff;
            }
            spatial_var_sum += var / token_count as f64;
        }
    }
    let token_spatial_std_rms = (spatial_var_sum / (batch * dim) as f64).sqrt();

    let token_total = batch * token_count;
    let mut norms = Vec::with_capacity(token_total);
    for chunk in values.chunks_exact(dim) {
        let norm_sq = chunk
            .iter()
            .map(|value| {
                let value = *value as f64;
                value * value
            })
            .sum::<f64>();
        norms.push(norm_sq.sqrt());
    }
    let token_norm_mean = norms.iter().sum::<f64>() / token_total as f64;
    let token_norm_std = (norms
        .iter()
        .map(|norm| {
            let diff = *norm - token_norm_mean;
            diff * diff
        })
        .sum::<f64>()
        / token_total as f64)
        .sqrt();

    let sample_tokens = token_count.min(64);
    let mut cosine_sum = 0.0;
    let mut cosine_count = 0usize;
    for b in 0..batch {
        let base = b * token_count * dim;
        for left in 0..sample_tokens {
            for right in left + 1..sample_tokens {
                let mut dot = 0.0;
                let mut left_norm = 0.0;
                let mut right_norm = 0.0;
                for d in 0..dim {
                    let l = values[base + left * dim + d] as f64;
                    let r = values[base + right * dim + d] as f64;
                    dot += l * r;
                    left_norm += l * l;
                    right_norm += r * r;
                }
                let denom = left_norm.sqrt() * right_norm.sqrt();
                if denom > 1.0e-12 {
                    cosine_sum += dot / denom;
                    cosine_count += 1;
                }
            }
        }
    }
    let mean_pairwise_token_cosine = if cosine_count == 0 {
        0.0
    } else {
        cosine_sum / cosine_count as f64
    };
    let relative_spread = (token_spatial_std_rms / (token_norm_mean + 1.0e-12)).clamp(0.0, 1.0);
    let collapse_score =
        (1.0 - relative_spread) * mean_pairwise_token_cosine.max(0.0).clamp(0.0, 1.0);

    Ok(FeatureTokenStabilityMetrics {
        measured: true,
        token_spatial_std_rms,
        token_norm_mean,
        token_norm_std,
        mean_pairwise_token_cosine,
        collapse_score,
        sampled_tokens: sample_tokens,
    })
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(default)]
pub struct SparseJepaAnyUpPcaFrameId {
    pub stream_id: u64,
    pub sequence: u64,
    pub capture_time_nanos: u64,
}

pub struct SparseJepaAnyUpPcaFrameInput<B: Backend> {
    pub id: SparseJepaAnyUpPcaFrameId,
    pub image: Tensor<B, 4>,
    pub mask: SparseTokenMask,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseJepaAnyUpPcaBackpressurePolicy {
    #[default]
    RejectNewest,
    DropOldest,
    OverwriteNewest,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SparseJepaAnyUpPcaStreamConfig {
    pub queue_capacity: usize,
    pub batch_size: usize,
    pub backpressure: SparseJepaAnyUpPcaBackpressurePolicy,
    pub schedule: FeatureFrameSchedule,
    pub require_monotonic_stream_sequences: bool,
    pub measurement: SparseJepaAnyUpPcaMeasurementConfig,
}

impl Default for SparseJepaAnyUpPcaStreamConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 2,
            batch_size: 1,
            backpressure: SparseJepaAnyUpPcaBackpressurePolicy::RejectNewest,
            schedule: FeatureFrameSchedule::default(),
            require_monotonic_stream_sequences: true,
            measurement: SparseJepaAnyUpPcaMeasurementConfig::disabled(),
        }
    }
}

impl SparseJepaAnyUpPcaStreamConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.batch_size > 0, "stream batch size must be nonzero");
        ensure!(
            self.queue_capacity >= self.batch_size,
            "stream queue capacity must be at least the batch size"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct SparseJepaAnyUpPcaQueueReport {
    pub accepted: bool,
    pub dropped_frame: Option<SparseJepaAnyUpPcaFrameId>,
    pub overwritten_frame: Option<SparseJepaAnyUpPcaFrameId>,
    pub queued_frames: usize,
    pub capacity: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct SparseJepaAnyUpPcaQueuedFrameTiming {
    pub id: SparseJepaAnyUpPcaFrameId,
    pub queue_wait_us: u64,
}

#[derive(Debug)]
pub struct SparseJepaAnyUpPcaStreamBatchOutput<B: Backend> {
    pub frame_ids: Vec<SparseJepaAnyUpPcaFrameId>,
    pub frame_timings: Vec<SparseJepaAnyUpPcaQueuedFrameTiming>,
    pub output: SparseJepaAnyUpPcaStepBatchOutput<B>,
    pub metrics: SparseJepaAnyUpPcaStageMetrics,
    pub queued_after: usize,
    pub dropped_frames_total: usize,
}

#[derive(Debug)]
pub struct FeatureFrameStreamOutput<B: Backend> {
    pub frame_ids: Vec<SparseJepaAnyUpPcaFrameId>,
    pub frame_timings: Vec<SparseJepaAnyUpPcaQueuedFrameTiming>,
    pub output: FeatureFrameBatch<B>,
    pub metrics: SparseJepaAnyUpPcaStageMetrics,
    pub request: FeatureFrameRequest,
    pub queued_after: usize,
    pub dropped_frames_total: usize,
    pub overwritten_frames_total: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct SparseJepaAnyUpPcaStreamStats {
    pub queued_frames: usize,
    pub dropped_frames: usize,
    pub overwritten_frames: usize,
    pub emitted_batches: usize,
    pub emitted_frames: usize,
}

pub struct SparseJepaAnyUpPcaStream<B: Backend> {
    pipeline: SparseJepaAnyUpPcaPipeline<B>,
    config: SparseJepaAnyUpPcaStreamConfig,
    pending: VecDeque<QueuedFrame<B>>,
    last_seen_by_stream: BTreeMap<u64, u64>,
    cached_mask_batch: Option<CachedSparseMaskBatch<B>>,
    #[cfg(any(feature = "sparse-patchify-wgpu", feature = "sparse-patchify-cuda"))]
    cached_sparse_patchify_plan: Option<CachedSparsePatchifyBatchPlan<B>>,
    dropped_frames: usize,
    overwritten_frames: usize,
    emitted_batches: usize,
    emitted_frames: usize,
}

struct QueuedFrame<B: Backend> {
    input: SparseJepaAnyUpPcaFrameInput<B>,
    queued_at: PipelineInstant,
}

struct DequeuedBatch<B: Backend> {
    ids: Vec<SparseJepaAnyUpPcaFrameId>,
    timings: Vec<SparseJepaAnyUpPcaQueuedFrameTiming>,
    image_batch: Tensor<B, 4>,
    mask_batch: SparseMaskBatch<B>,
}

struct CachedSparseMaskBatch<B: Backend> {
    dense_len: usize,
    rows: Vec<Vec<usize>>,
    mask: SparseMaskBatch<B>,
}

#[cfg(any(feature = "sparse-patchify-wgpu", feature = "sparse-patchify-cuda"))]
struct CachedSparsePatchifyBatchPlan<B: Backend> {
    grid: TokenGridShape,
    rows: Vec<Vec<usize>>,
    plan: SparsePatchifyBatchPlan<B>,
}

impl<B: Backend> SparseJepaAnyUpPcaStream<B> {
    pub fn new(
        pipeline: SparseJepaAnyUpPcaPipeline<B>,
        config: SparseJepaAnyUpPcaStreamConfig,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(
            pipeline.batch() == config.batch_size,
            "stream batch size must match the pipeline batch"
        );
        Ok(Self {
            pipeline,
            config,
            pending: VecDeque::new(),
            last_seen_by_stream: BTreeMap::new(),
            cached_mask_batch: None,
            #[cfg(any(feature = "sparse-patchify-wgpu", feature = "sparse-patchify-cuda"))]
            cached_sparse_patchify_plan: None,
            dropped_frames: 0,
            overwritten_frames: 0,
            emitted_batches: 0,
            emitted_frames: 0,
        })
    }

    pub fn pipeline(&self) -> &SparseJepaAnyUpPcaPipeline<B> {
        &self.pipeline
    }

    pub fn pipeline_mut(&mut self) -> &mut SparseJepaAnyUpPcaPipeline<B> {
        &mut self.pipeline
    }

    pub fn config(&self) -> &SparseJepaAnyUpPcaStreamConfig {
        &self.config
    }

    pub fn queued_frames(&self) -> usize {
        self.pending.len()
    }

    pub fn dropped_frames(&self) -> usize {
        self.dropped_frames
    }

    pub fn can_process_batch(&self) -> bool {
        self.pending.len() >= self.config.batch_size
    }

    pub fn stats(&self) -> SparseJepaAnyUpPcaStreamStats {
        SparseJepaAnyUpPcaStreamStats {
            queued_frames: self.pending.len(),
            dropped_frames: self.dropped_frames,
            overwritten_frames: self.overwritten_frames,
            emitted_batches: self.emitted_batches,
            emitted_frames: self.emitted_frames,
        }
    }

    pub fn enqueue(
        &mut self,
        input: SparseJepaAnyUpPcaFrameInput<B>,
    ) -> Result<SparseJepaAnyUpPcaQueueReport> {
        ensure!(
            input.mask.dense_len() == self.pipeline.grid().len(),
            "frame sparse mask dense length must match pipeline grid"
        );
        ensure!(
            input.image.shape().dims::<4>()[0] == 1,
            "stream frame inputs must have batch size 1"
        );
        if self.config.require_monotonic_stream_sequences
            && let Some(&last) = self.last_seen_by_stream.get(&input.id.stream_id)
        {
            ensure!(
                input.id.sequence > last,
                "frame sequence must increase monotonically per stream"
            );
        }

        let mut dropped_frame = None;
        let mut overwritten_frame = None;
        if self.pending.len() >= self.config.queue_capacity {
            match self.config.backpressure {
                SparseJepaAnyUpPcaBackpressurePolicy::RejectNewest => {
                    bail!("high-res pipeline queue is full; apply backpressure or use drop_oldest");
                }
                SparseJepaAnyUpPcaBackpressurePolicy::DropOldest => {
                    let dropped = self
                        .pending
                        .pop_front()
                        .expect("queue is nonempty when full");
                    dropped_frame = Some(dropped.input.id);
                    self.dropped_frames += 1;
                }
                SparseJepaAnyUpPcaBackpressurePolicy::OverwriteNewest => {
                    let overwritten = self
                        .pending
                        .pop_back()
                        .expect("queue is nonempty when full");
                    overwritten_frame = Some(overwritten.input.id);
                    dropped_frame = overwritten_frame;
                    self.dropped_frames += 1;
                    self.overwritten_frames += 1;
                }
            }
        }

        self.last_seen_by_stream
            .insert(input.id.stream_id, input.id.sequence);
        self.pending.push_back(QueuedFrame {
            input,
            queued_at: pipeline_now(),
        });
        Ok(SparseJepaAnyUpPcaQueueReport {
            accepted: true,
            dropped_frame,
            overwritten_frame,
            queued_frames: self.pending.len(),
            capacity: self.config.queue_capacity,
        })
    }

    pub fn process_next_ready(&mut self) -> Result<Option<SparseJepaAnyUpPcaStreamBatchOutput<B>>> {
        let Some(dequeued) = self.dequeue_ready_batch()? else {
            return Ok(None);
        };
        let measured = self.pipeline.step_image_with_mask_batch_measured(
            dequeued.image_batch,
            dequeued.mask_batch,
            self.config.measurement,
        )?;
        Ok(Some(self.finish_dequeued_batch(
            dequeued.ids,
            dequeued.timings,
            measured,
        )))
    }

    pub fn process_next_ready_nodes(&mut self) -> Result<Option<FeatureFrameStreamOutput<B>>> {
        let Some(dequeued) = self.dequeue_ready_batch()? else {
            return Ok(None);
        };
        let request = self.config.schedule.request_for(&dequeued.ids);
        let measured = self.pipeline.step_image_with_mask_batch_nodes_measured(
            dequeued.image_batch,
            dequeued.mask_batch,
            request,
            self.config.measurement,
        )?;
        Ok(Some(self.finish_dequeued_nodes(
            dequeued.ids,
            dequeued.timings,
            request,
            measured,
        )))
    }

    fn dequeue_ready_batch(&mut self) -> Result<Option<DequeuedBatch<B>>> {
        if self.pending.len() < self.config.batch_size {
            return Ok(None);
        }
        let sparse_width = self
            .pending
            .front()
            .map(|frame| frame.input.mask.len())
            .unwrap_or(0);
        ensure!(
            self.pending
                .iter()
                .take(self.config.batch_size)
                .all(|frame| frame.input.mask.len() == sparse_width),
            "front in-flight batch has variable sparse mask widths; use batch_size=1 or group masks by token budget"
        );

        let now = pipeline_now();
        let mut ids = Vec::with_capacity(self.config.batch_size);
        let mut timings = Vec::with_capacity(self.config.batch_size);
        let mut images = Vec::with_capacity(self.config.batch_size);
        let mut rows = Vec::with_capacity(self.config.batch_size);
        for _ in 0..self.config.batch_size {
            let queued = self
                .pending
                .pop_front()
                .expect("ready batch has enough queued frames");
            ids.push(queued.input.id);
            timings.push(SparseJepaAnyUpPcaQueuedFrameTiming {
                id: queued.input.id,
                queue_wait_us: pipeline_delta_us(now, queued.queued_at),
            });
            rows.push(queued.input.mask.indices().to_vec());
            images.push(queued.input.image);
        }

        let image_batch = Tensor::cat(images, 0);
        let dense_len = self.pipeline.grid().len();
        let device = image_batch.device();
        let mask_batch = self.cached_sparse_mask_batch(rows, dense_len, &device)?;
        ensure!(
            !mask_batch.is_ragged(),
            "stream produced a ragged sparse mask batch; use fixed token budgets for batched in-flight processing"
        );
        Ok(Some(DequeuedBatch {
            ids,
            timings,
            image_batch,
            mask_batch,
        }))
    }

    fn finish_dequeued_batch(
        &mut self,
        ids: Vec<SparseJepaAnyUpPcaFrameId>,
        timings: Vec<SparseJepaAnyUpPcaQueuedFrameTiming>,
        measured: SparseJepaAnyUpPcaMeasuredBatchOutput<B>,
    ) -> SparseJepaAnyUpPcaStreamBatchOutput<B> {
        self.emitted_batches += 1;
        self.emitted_frames += ids.len();
        SparseJepaAnyUpPcaStreamBatchOutput {
            frame_ids: ids,
            frame_timings: timings,
            output: measured.output,
            metrics: measured.metrics,
            queued_after: self.pending.len(),
            dropped_frames_total: self.dropped_frames,
        }
    }

    fn finish_dequeued_nodes(
        &mut self,
        ids: Vec<SparseJepaAnyUpPcaFrameId>,
        timings: Vec<SparseJepaAnyUpPcaQueuedFrameTiming>,
        request: FeatureFrameRequest,
        measured: MeasuredFeatureFrameBatch<B>,
    ) -> FeatureFrameStreamOutput<B> {
        self.emitted_batches += 1;
        self.emitted_frames += ids.len();
        FeatureFrameStreamOutput {
            frame_ids: ids,
            frame_timings: timings,
            output: measured.output,
            metrics: measured.metrics,
            request,
            queued_after: self.pending.len(),
            dropped_frames_total: self.dropped_frames,
            overwritten_frames_total: self.overwritten_frames,
        }
    }

    fn cached_sparse_mask_batch(
        &mut self,
        rows: Vec<Vec<usize>>,
        dense_len: usize,
        device: &B::Device,
    ) -> Result<SparseMaskBatch<B>> {
        let reuse = self
            .cached_mask_batch
            .as_ref()
            .is_some_and(|cached| cached.dense_len == dense_len && cached.rows == rows);
        if !reuse {
            let mask = SparseMaskBatch::from_rows(rows, dense_len, device)?;
            self.cached_mask_batch = Some(CachedSparseMaskBatch {
                dense_len,
                rows: mask.rows(),
                mask,
            });
        }
        Ok(self
            .cached_mask_batch
            .as_ref()
            .expect("sparse mask batch cache is initialized")
            .mask
            .clone())
    }

    #[cfg(any(feature = "sparse-patchify-wgpu", feature = "sparse-patchify-cuda"))]
    fn cached_sparse_patchify_batch_plan(
        &mut self,
        mask: &SparseMaskBatch<B>,
        device: &B::Device,
    ) -> Result<SparsePatchifyBatchPlan<B>> {
        let grid = self.pipeline.grid();
        let rows = mask.rows();
        let reuse = self
            .cached_sparse_patchify_plan
            .as_ref()
            .is_some_and(|cached| cached.grid == grid && cached.rows == rows);
        if !reuse {
            self.cached_sparse_patchify_plan = Some(CachedSparsePatchifyBatchPlan {
                grid,
                rows,
                plan: SparsePatchifyBatchPlan::new(mask.clone(), grid, device)?,
            });
        }
        Ok(self
            .cached_sparse_patchify_plan
            .as_ref()
            .expect("sparse patchify plan cache is initialized")
            .plan
            .clone())
    }

    pub fn process_all_ready(&mut self) -> Result<Vec<SparseJepaAnyUpPcaStreamBatchOutput<B>>> {
        let mut outputs = Vec::new();
        while self.can_process_batch() {
            if let Some(output) = self.process_next_ready()? {
                outputs.push(output);
            }
        }
        Ok(outputs)
    }

    pub fn process_all_ready_nodes(&mut self) -> Result<Vec<FeatureFrameStreamOutput<B>>> {
        let mut outputs = Vec::new();
        while self.can_process_batch() {
            if let Some(output) = self.process_next_ready_nodes()? {
                outputs.push(output);
            }
        }
        Ok(outputs)
    }
}

#[cfg(feature = "sparse-patchify-wgpu")]
impl SparseJepaAnyUpPcaStream<burn_flex_gmm::wgpu::DefaultWgpuBackend> {
    pub fn process_next_ready_sparse_patchify_wgpu(
        &mut self,
    ) -> Result<Option<SparseJepaAnyUpPcaStreamBatchOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>>>
    {
        let Some(dequeued) = self.dequeue_ready_batch()? else {
            return Ok(None);
        };
        let device = dequeued.image_batch.device();
        let patchify_plan =
            self.cached_sparse_patchify_batch_plan(&dequeued.mask_batch, &device)?;
        let measured = self
            .pipeline
            .step_image_with_sparse_patchify_plan_wgpu_measured(
                dequeued.image_batch,
                &patchify_plan,
                self.config.measurement,
            )?;
        Ok(Some(self.finish_dequeued_batch(
            dequeued.ids,
            dequeued.timings,
            measured,
        )))
    }

    pub fn process_next_ready_sparse_patchify_wgpu_nodes(
        &mut self,
    ) -> Result<Option<FeatureFrameStreamOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>>> {
        let Some(dequeued) = self.dequeue_ready_batch()? else {
            return Ok(None);
        };
        let request = self.config.schedule.request_for(&dequeued.ids);
        let device = dequeued.image_batch.device();
        let patchify_plan =
            self.cached_sparse_patchify_batch_plan(&dequeued.mask_batch, &device)?;
        let measured = self
            .pipeline
            .step_image_with_sparse_patchify_plan_wgpu_nodes_measured(
                dequeued.image_batch,
                &patchify_plan,
                request,
                self.config.measurement,
            )?;
        Ok(Some(self.finish_dequeued_nodes(
            dequeued.ids,
            dequeued.timings,
            request,
            measured,
        )))
    }

    pub fn process_all_ready_sparse_patchify_wgpu(
        &mut self,
    ) -> Result<Vec<SparseJepaAnyUpPcaStreamBatchOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>>>
    {
        let mut outputs = Vec::new();
        while self.can_process_batch() {
            if let Some(output) = self.process_next_ready_sparse_patchify_wgpu()? {
                outputs.push(output);
            }
        }
        Ok(outputs)
    }

    pub fn process_all_ready_sparse_patchify_wgpu_nodes(
        &mut self,
    ) -> Result<Vec<FeatureFrameStreamOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>>> {
        let mut outputs = Vec::new();
        while self.can_process_batch() {
            if let Some(output) = self.process_next_ready_sparse_patchify_wgpu_nodes()? {
                outputs.push(output);
            }
        }
        Ok(outputs)
    }
}

#[cfg(feature = "sparse-patchify-cuda")]
impl SparseJepaAnyUpPcaStream<burn_flex_gmm::cuda::DefaultCudaBackend> {
    pub fn process_next_ready_sparse_patchify_cuda(
        &mut self,
    ) -> Result<Option<SparseJepaAnyUpPcaStreamBatchOutput<burn_flex_gmm::cuda::DefaultCudaBackend>>>
    {
        let Some(dequeued) = self.dequeue_ready_batch()? else {
            return Ok(None);
        };
        let device = dequeued.image_batch.device();
        let patchify_plan =
            self.cached_sparse_patchify_batch_plan(&dequeued.mask_batch, &device)?;
        let measured = self
            .pipeline
            .step_image_with_sparse_patchify_plan_cuda_measured(
                dequeued.image_batch,
                &patchify_plan,
                self.config.measurement,
            )?;
        Ok(Some(self.finish_dequeued_batch(
            dequeued.ids,
            dequeued.timings,
            measured,
        )))
    }

    pub fn process_next_ready_sparse_patchify_cuda_nodes(
        &mut self,
    ) -> Result<Option<FeatureFrameStreamOutput<burn_flex_gmm::cuda::DefaultCudaBackend>>> {
        let Some(dequeued) = self.dequeue_ready_batch()? else {
            return Ok(None);
        };
        let request = self.config.schedule.request_for(&dequeued.ids);
        let device = dequeued.image_batch.device();
        let patchify_plan =
            self.cached_sparse_patchify_batch_plan(&dequeued.mask_batch, &device)?;
        let measured = self
            .pipeline
            .step_image_with_sparse_patchify_plan_cuda_nodes_measured(
                dequeued.image_batch,
                &patchify_plan,
                request,
                self.config.measurement,
            )?;
        Ok(Some(self.finish_dequeued_nodes(
            dequeued.ids,
            dequeued.timings,
            request,
            measured,
        )))
    }

    pub fn process_all_ready_sparse_patchify_cuda(
        &mut self,
    ) -> Result<Vec<SparseJepaAnyUpPcaStreamBatchOutput<burn_flex_gmm::cuda::DefaultCudaBackend>>>
    {
        let mut outputs = Vec::new();
        while self.can_process_batch() {
            if let Some(output) = self.process_next_ready_sparse_patchify_cuda()? {
                outputs.push(output);
            }
        }
        Ok(outputs)
    }

    pub fn process_all_ready_sparse_patchify_cuda_nodes(
        &mut self,
    ) -> Result<Vec<FeatureFrameStreamOutput<burn_flex_gmm::cuda::DefaultCudaBackend>>> {
        let mut outputs = Vec::new();
        while self.can_process_batch() {
            if let Some(output) = self.process_next_ready_sparse_patchify_cuda_nodes()? {
                outputs.push(output);
            }
        }
        Ok(outputs)
    }
}

pub type FeatureFrameMeasureConfig = SparseJepaAnyUpPcaMeasurementConfig;
pub type FeatureFrameEncodePath = SparseJepaAnyUpPcaEncodePath;
pub type FeatureFrameMetrics = SparseJepaAnyUpPcaStageMetrics;
pub type FeatureFramePipelineConfig = SparseJepaAnyUpPcaPipelineConfig;
pub type FeatureFramePipeline<B> = SparseJepaAnyUpPcaPipeline<B>;
pub type FrameId = SparseJepaAnyUpPcaFrameId;
pub type FeatureFrameInput<B> = SparseJepaAnyUpPcaFrameInput<B>;
pub type FrameQueuePolicy = SparseJepaAnyUpPcaBackpressurePolicy;
pub type FrameStreamConfig = SparseJepaAnyUpPcaStreamConfig;
pub type FrameQueueReport = SparseJepaAnyUpPcaQueueReport;
pub type FrameQueueTiming = SparseJepaAnyUpPcaQueuedFrameTiming;
pub type FeatureFrameStream<B> = SparseJepaAnyUpPcaStream<B>;
pub type FeatureFrameStreamStats = SparseJepaAnyUpPcaStreamStats;

fn scheduled_for_any(ids: &[SparseJepaAnyUpPcaFrameId], every: Option<u64>) -> bool {
    let Some(every) = every.filter(|&every| every > 0) else {
        return false;
    };
    ids.iter().any(|id| id.sequence % every == 0)
}

fn micros_u64(value: u128) -> u64 {
    value.min(u64::MAX as u128) as u64
}
