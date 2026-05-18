#![recursion_limit = "512"]

use std::{collections::VecDeque, path::PathBuf};

use anyhow::{Context, Result, bail};
#[cfg(not(target_arch = "wasm32"))]
use bevy::tasks::{AsyncComputeTaskPool, block_on, futures_lite::future};
use bevy::{
    app::AppExit,
    prelude::*,
    render::{
        RenderPlugin,
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
    },
    tasks::Task,
    ui::{RelativeCursorPosition, widget::ImageNode},
    window::PrimaryWindow,
};
use bevy_burn::{BevyBurnBridgePlugin, BurnDevice};
use burn::tensor::{
    Tensor, TensorData,
    backend::Backend,
    module::interpolate,
    ops::{InterpolateMode, InterpolateOptions},
};
use burn_jepa::{
    AnyUp, AnyUpImageGrid, FeatureFrameBatch, FeatureFrameMetrics, FeatureFramePipeline,
    FeatureFramePipelineConfig, FeatureFrameRequest, FeatureFrameSparseEncodeMode,
    FeaturePcaProjector, FrameId, HighResFrameArtifacts, LowResFrameArtifacts, SparseTokenMask,
    TokenGridShape, VJEPA_IMAGE_MEAN, VJEPA_IMAGE_STD, VJepaConfig, shape_prewarm_masks,
};
#[cfg(feature = "sparse-patchify-wgpu")]
use burn_jepa::{SparseMaskBatch, SparsePatchifyBatchPlan};
use image::{ImageReader, RgbaImage, imageops::FilterType};

mod config;
mod display;
mod mask;
mod metrics;
mod model_loading;
pub mod platform;

#[cfg(test)]
use burn_jepa::{
    FeaturePcaUpdateConfig, bucket_sparse_mask, center_prior_mask, finalize_patch_diff_mask,
    finalize_patch_diff_masks,
};
pub use config::{
    BevyJepaAnyUpModelPackageProfile, BevyJepaConfig, BevyJepaDisplayTransfer, BevyJepaEncodePath,
    BevyJepaEncoderSource, BevyJepaFrameSource, BevyJepaMaskSource, BevyJepaModelPackageProfile,
    BevyJepaSparseEncodeMode, DEFAULT_ANYUP_CHECKPOINT_PATH, DEFAULT_ANYUP_CHUNK_SIZE,
    DEFAULT_ANYUP_MODEL_MANIFEST_PATH, DEFAULT_ANYUP_PACKAGE_DIR,
    DEFAULT_BOOTSTRAP_CONTEXT_DENSITY, DEFAULT_CAMERA_FPS, DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH, DEFAULT_CONTEXT_DENSITY, DEFAULT_HIGH_RES_PCA_EVERY, DEFAULT_IMAGE_SIZE,
    DEFAULT_MIN_CONTEXT_DENSITY, DEFAULT_MODEL_MANIFEST_PATH, DEFAULT_MODEL_PACKAGE_DIR,
    DEFAULT_PATCH_DIFF_AGE_REFRESH_INTERVAL_FRAMES, DEFAULT_PATCH_DIFF_AGE_REFRESH_MAX_DENSITY,
    DEFAULT_PATCH_DIFF_BLUE_NOISE_REFRESH_DENSITY, DEFAULT_PATCH_DIFF_DENSE_FALLBACK_DENSITY,
    DEFAULT_PATCH_DIFF_QUALITY, DEFAULT_PATCH_DIFF_REFRESH_ENABLED,
    DEFAULT_PATCH_DIFF_REFRESH_MAX_DENSITY, DEFAULT_PATCH_DIFF_SUBTHRESHOLD_DECAY,
    DEFAULT_PATCH_DIFF_SUBTHRESHOLD_MAX_DENSITY, DEFAULT_PATCH_DIFF_SUBTHRESHOLD_TRIGGER,
    DEFAULT_PATCH_DIFF_THRESHOLD, DEFAULT_PCA_MIN_SAMPLE_FRAMES, DEFAULT_PCA_SAMPLE_WINDOW_FRAMES,
    DEFAULT_PCA_UPDATE_EVERY, DEFAULT_PCA_UPDATE_ITERATIONS, DEFAULT_PREWARM_SHAPE_BUCKETS,
    DEFAULT_SPARSE_MASK_BUCKET_TOKENS, DEFAULT_TTT_MODEL_PATH, DEFAULT_VJEPA21_CHECKPOINT_DIR,
    DEFAULT_VJEPA21_CONFIG_PATH, DEFAULT_VJEPA21_WEIGHTS_NAME, FeatureFrameViewerConfig,
    MIN_PIPELINE_IMAGE_SIZE, PIPELINE_IMAGE_SIZE_MULTIPLE, PatchDiffRefreshConfig,
    default_anyup_model_manifest_path_for_profile, default_model_manifest_path_for_profile,
};
use display::{
    HighResPanelData, InputPanelData, JepaPanelTextures, StagePanelData,
    apply_high_res_panel_to_world, apply_input_panel_to_world, apply_stage_panels_to_world,
    clear_completed_gpu_uploads,
};
#[cfg(test)]
use mask::run_sparse_mask_node;
use mask::run_sparse_mask_node_with_refresh_state;
pub use metrics::{BevyJepaMetrics, BevyJepaStepOutput};
use metrics::{MetricFrameContext, bevy_metrics_from_stage, metrics_source_status};
#[cfg(test)]
use metrics::{format_metrics_line, format_metrics_waiting_line};
use model_loading::{effective_anyup_weights, load_viewer_anyup, load_viewer_encoder};
#[cfg(test)]
use model_loading::{
    effective_ttt_model_path, resolve_repo_relative_path, tiny_viewer_model_config,
};

pub type JepaBevyBackend = burn::backend::WebGpu<f32, i32>;
pub type JepaBevyDevice = burn::backend::wgpu::WgpuDevice;

const UI_MARGIN_PX: f32 = 12.0;
const METRIC_ROW_HEIGHT: f32 = 24.0;
const CONTROL_PANEL_WIDTH_PX: f32 = 480.0;
const CONTROL_TOP_PX: f32 = UI_MARGIN_PX + METRIC_ROW_HEIGHT + 8.0;
const CONTROL_ROW_GAP_PX: f32 = 8.0;
const CONTROL_BUTTON_HEIGHT_PX: f32 = 28.0;
const CONTROL_LABEL_WIDTH_PX: f32 = 128.0;
const CONTROL_SLIDER_WIDTH_PX: f32 = 206.0;
const CONTROL_SLIDER_HEIGHT_PX: f32 = 18.0;
const CONTROL_SLIDER_UPDATE_EPSILON: f32 = 0.001;
const METRICS_TOP_WIDTH_PX: f32 = 560.0;
const METRICS_STAGE_HEIGHT_PX: f32 = 138.0;
const METRICS_GRAPH_BARS: usize = 64;
const METRICS_GRAPH_HEIGHT_PX: f32 = 52.0;
const DENSE_PIPELINE_FALLBACK_DENSITY: f32 = 0.0;

#[cfg(not(target_arch = "wasm32"))]
type ViewerInstant = std::time::Instant;
#[cfg(target_arch = "wasm32")]
type ViewerInstant = f64;

#[cfg(not(target_arch = "wasm32"))]
fn viewer_now() -> ViewerInstant {
    std::time::Instant::now()
}

#[cfg(target_arch = "wasm32")]
fn viewer_now() -> ViewerInstant {
    web_sys::window()
        .and_then(|window| window.performance())
        .map(|performance| performance.now())
        .unwrap_or_else(js_sys::Date::now)
}

#[cfg(not(target_arch = "wasm32"))]
fn viewer_seconds_since(now: ViewerInstant, previous: ViewerInstant) -> f64 {
    now.duration_since(previous).as_secs_f64()
}

#[cfg(target_arch = "wasm32")]
fn viewer_seconds_since(now: ViewerInstant, previous: ViewerInstant) -> f64 {
    (now - previous) / 1000.0
}

#[cfg(not(target_arch = "wasm32"))]
fn viewer_elapsed_us(start: ViewerInstant) -> u64 {
    micros_u64(start.elapsed().as_micros())
}

#[cfg(target_arch = "wasm32")]
fn viewer_elapsed_us(start: ViewerInstant) -> u64 {
    let elapsed_ms = viewer_now() - start;
    if elapsed_ms.is_finite() && elapsed_ms > 0.0 {
        micros_u64((elapsed_ms * 1000.0) as u128)
    } else {
        0
    }
}
const PANEL_LABEL_ROW_HEIGHT: f32 = 34.0;

pub fn log(message: &str) {
    #[cfg(target_arch = "wasm32")]
    web_sys::console::log_1(&message.into());

    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("{message}");
}

pub fn warn(message: &str) {
    #[cfg(target_arch = "wasm32")]
    web_sys::console::warn_1(&message.into());

    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("warning: {message}");
}

pub fn error(message: &str) {
    #[cfg(target_arch = "wasm32")]
    web_sys::console::error_1(&message.into());

    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("error: {message}");
}

fn sync_measurements_enabled(config: &BevyJepaConfig) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = config;
        false
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        config.sync_measurements
    }
}

fn measurement_config(config: &BevyJepaConfig) -> burn_jepa::SparseJepaAnyUpPcaMeasurementConfig {
    let mut measurement = config.measurement_config();
    measurement.sync_backend = sync_measurements_enabled(config);
    measurement
}

fn sync_bevy_backend(device: &JepaBevyDevice) -> Result<()> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = device;
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        JepaBevyBackend::sync(device)?;
        Ok(())
    }
}

#[derive(Resource, Default)]
struct JepaRuntime {
    pipeline: Option<FeatureFramePipeline<JepaBevyBackend>>,
    model_config: Option<VJepaConfig>,
    pipeline_signature: Option<RuntimePipelineSignature>,
    pipeline_grid: Option<TokenGridShape>,
    pipeline_patch_size: Option<usize>,
    active_task: Option<Task<JepaAsyncTaskOutput>>,
    pending_stage: Option<PendingStageFrame>,
    high_res_runtime: Option<AnyUpHighResRuntime>,
    high_res_signature: Option<RuntimePipelineSignature>,
    high_res_task: Option<Task<HighResAsyncTaskOutput>>,
    pending_high_res: Option<HighResFrameInput>,
    prev_image: Option<Tensor<JepaBevyBackend, 4>>,
    prev_rgba: Option<RgbaImage>,
    prev_stage_image: Option<Tensor<JepaBevyBackend, 4>>,
    prev_stage_rgba: Option<RgbaImage>,
    static_frame: Option<CachedStaticFrame>,
    frame_index: u64,
    input_frames_seen: u64,
    completed_frames: u64,
    high_res_frames: u64,
    latest_input_sequence: u64,
    dropped_frames: usize,
    overwritten_frames: usize,
    stale_completions: usize,
    last_input_at: Option<ViewerInstant>,
    last_completion_at: Option<ViewerInstant>,
    last_high_res_completion_at: Option<ViewerInstant>,
    input_fps: f64,
    low_res_fps: f64,
    high_res_fps: f64,
    last_high_res_anyup_context_us: u64,
    last_high_res_anyup_decode_us: u64,
    last_high_res_pca_us: u64,
    last_high_res_display_tensor_us: u64,
    last_error: Option<String>,
    last_logged_error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RuntimePipelineSignature {
    encoder_source: BevyJepaEncoderSource,
    encode_path: BevyJepaEncodePath,
    model_manifest_path: Option<PathBuf>,
    model_cache_dir: Option<PathBuf>,
    model_profile: BevyJepaModelPackageProfile,
    model_base_url: String,
    model_auto_download: bool,
    ttt_model_path: Option<PathBuf>,
    jepa_checkpoint_dir: Option<PathBuf>,
    jepa_config_path: Option<PathBuf>,
    jepa_weights_name: String,
    image_size: usize,
    anyup_weights: Option<PathBuf>,
    anyup_model_manifest_path: Option<PathBuf>,
    anyup_model_cache_dir: Option<PathBuf>,
    anyup_model_profile: BevyJepaAnyUpModelPackageProfile,
    anyup_model_base_url: String,
    anyup_model_auto_download: bool,
    anyup_attention_mode: burn_jepa::AnyUpAttentionMode,
    anyup_q_chunk_size: usize,
    pca_update_every: u64,
    pca_sample_window_frames: usize,
    pca_min_sample_frames: usize,
    pca_update_iterations: usize,
    sparse_encode_mode: BevyJepaSparseEncodeMode,
    patch_diff_threshold_bits: u32,
    context_density_bits: u32,
    min_context_density_bits: u32,
    bootstrap_context_density_bits: u32,
    patch_diff_dense_fallback_bits: u32,
    patch_diff_refresh_enabled: bool,
    patch_diff_subthreshold_enabled: bool,
    patch_diff_subthreshold_decay_bits: u32,
    patch_diff_subthreshold_gain_bits: u32,
    patch_diff_subthreshold_trigger_bits: u32,
    patch_diff_subthreshold_max_density_bits: u32,
    patch_diff_age_refresh_enabled: bool,
    patch_diff_age_refresh_interval_frames: u64,
    patch_diff_age_refresh_max_density_bits: u32,
    patch_diff_blue_noise_enabled: bool,
    patch_diff_blue_noise_refresh_density_bits: u32,
    patch_diff_blue_noise_seed: u64,
    patch_diff_refresh_max_density_bits: u32,
    sparse_mask_bucket_tokens: usize,
    prewarm_shape_buckets: bool,
    ttt_runtime_enabled: bool,
    ttt_update_fast_weight: bool,
    ttt_state_decay_per_frame_bits: u64,
    ttt_reset_interval_frames: u64,
    ttt_metrics_interval_frames: u64,
    ttt_collapse_guard_enabled: bool,
    ttt_collapse_guard_action: burn_jepa::TttRuntimeCollapseGuardAction,
    ttt_collapse_guard_decay_bits: u64,
}

impl RuntimePipelineSignature {
    fn new(config: &BevyJepaConfig, image_size: usize) -> Self {
        Self {
            encoder_source: config.encoder_source,
            encode_path: config.encode_path,
            model_manifest_path: config.model_manifest_path.clone(),
            model_cache_dir: config.model_cache_dir.clone(),
            model_profile: config.model_profile,
            model_base_url: config.model_base_url.clone(),
            model_auto_download: config.model_auto_download,
            ttt_model_path: config.ttt_model_path.clone(),
            jepa_checkpoint_dir: config.jepa_checkpoint_dir.clone(),
            jepa_config_path: config.jepa_config_path.clone(),
            jepa_weights_name: config.jepa_weights_name.clone(),
            image_size,
            anyup_weights: effective_anyup_weights(config),
            anyup_model_manifest_path: config.anyup_model_manifest_path.clone(),
            anyup_model_cache_dir: config.anyup_model_cache_dir.clone(),
            anyup_model_profile: config.anyup_model_profile,
            anyup_model_base_url: config.anyup_model_base_url.clone(),
            anyup_model_auto_download: config.anyup_model_auto_download,
            anyup_attention_mode: config.anyup_attention_mode,
            anyup_q_chunk_size: config.anyup_q_chunk_size,
            pca_update_every: config.pca_update_every,
            pca_sample_window_frames: config.pca_sample_window_frames,
            pca_min_sample_frames: config.pca_min_sample_frames,
            pca_update_iterations: config.pca_update_iterations,
            sparse_encode_mode: config.sparse_encode_mode,
            patch_diff_threshold_bits: config.patch_diff_threshold.to_bits(),
            context_density_bits: config.context_density.to_bits(),
            min_context_density_bits: config.min_context_density.to_bits(),
            bootstrap_context_density_bits: config.bootstrap_context_density.to_bits(),
            patch_diff_dense_fallback_bits: config.patch_diff_dense_fallback_density.to_bits(),
            patch_diff_refresh_enabled: config.patch_diff_refresh.enabled,
            patch_diff_subthreshold_enabled: config.patch_diff_refresh.subthreshold_enabled,
            patch_diff_subthreshold_decay_bits: config
                .patch_diff_refresh
                .subthreshold_decay
                .to_bits(),
            patch_diff_subthreshold_gain_bits: config
                .patch_diff_refresh
                .subthreshold_gain
                .to_bits(),
            patch_diff_subthreshold_trigger_bits: config
                .patch_diff_refresh
                .subthreshold_trigger
                .to_bits(),
            patch_diff_subthreshold_max_density_bits: config
                .patch_diff_refresh
                .subthreshold_max_density
                .to_bits(),
            patch_diff_age_refresh_enabled: config.patch_diff_refresh.age_refresh_enabled,
            patch_diff_age_refresh_interval_frames: config
                .patch_diff_refresh
                .age_refresh_interval_frames,
            patch_diff_age_refresh_max_density_bits: config
                .patch_diff_refresh
                .age_refresh_max_density
                .to_bits(),
            patch_diff_blue_noise_enabled: config.patch_diff_refresh.blue_noise_enabled,
            patch_diff_blue_noise_refresh_density_bits: config
                .patch_diff_refresh
                .blue_noise_refresh_density
                .to_bits(),
            patch_diff_blue_noise_seed: config.patch_diff_refresh.blue_noise_seed,
            patch_diff_refresh_max_density_bits: config
                .patch_diff_refresh
                .max_extra_density
                .to_bits(),
            sparse_mask_bucket_tokens: config.sparse_mask_bucket_tokens,
            prewarm_shape_buckets: config.prewarm_shape_buckets,
            ttt_runtime_enabled: config.ttt_runtime.enabled,
            ttt_update_fast_weight: config.ttt_runtime.update_fast_weight,
            ttt_state_decay_per_frame_bits: config.ttt_runtime.state_decay_per_frame.to_bits(),
            ttt_reset_interval_frames: config.ttt_runtime.reset_interval_frames,
            ttt_metrics_interval_frames: config.ttt_runtime.metrics_interval_frames,
            ttt_collapse_guard_enabled: config.ttt_runtime.collapse_guard_enabled,
            ttt_collapse_guard_action: config.ttt_runtime.collapse_guard_action,
            ttt_collapse_guard_decay_bits: config.ttt_runtime.collapse_guard_decay.to_bits(),
        }
    }
}

struct CachedStaticFrame {
    path: Option<PathBuf>,
    image_size: usize,
    rgba: RgbaImage,
    image: Tensor<JepaBevyBackend, 4>,
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
struct AnyUpHighResRuntime {
    anyup: AnyUp<JepaBevyBackend>,
    anyup_image_grid: AnyUpImageGrid<JepaBevyBackend>,
    image_size: [usize; 2],
    grid: TokenGridShape,
    q_chunk_size: usize,
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
struct HighResFrameInput {
    signature: RuntimePipelineSignature,
    id: FrameId,
    image: Tensor<JepaBevyBackend, 4>,
    low_res_features: Tensor<JepaBevyBackend, 4>,
    pca: FeaturePcaProjector<JepaBevyBackend>,
    display_transfer: BevyJepaDisplayTransfer,
    sync_measurements: bool,
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
struct HighResAsyncTaskOutput {
    signature: RuntimePipelineSignature,
    runtime: AnyUpHighResRuntime,
    result: Result<HighResProcessedFrame, String>,
}

struct HighResProcessedFrame {
    id: FrameId,
    panels: HighResPanelData,
    anyup_context_us: u64,
    anyup_decode_us: u64,
    high_res_pca_us: u64,
    display_tensor_us: u64,
    total_us: u64,
}

pub struct BevyJepaHeadlessPipeline {
    config: BevyJepaConfig,
    runtime: JepaRuntime,
    device: JepaBevyDevice,
}

#[derive(Clone, Copy)]
enum BevyJepaStepMode {
    StageOnly,
    DisplayPanels(FeatureFrameRequest),
    StageRequest(FeatureFrameRequest),
}

impl BevyJepaHeadlessPipeline {
    pub fn new(config: BevyJepaConfig, device: JepaBevyDevice) -> Self {
        Self {
            config,
            runtime: JepaRuntime::default(),
            device,
        }
    }

    pub fn step_stage_only(&mut self) -> Result<BevyJepaStepOutput> {
        self.step_internal(BevyJepaStepMode::StageOnly)
    }

    pub fn step_with_display_panels(&mut self) -> Result<BevyJepaStepOutput> {
        self.step_with_display_request(FeatureFrameRequest::full_pca())
    }

    pub fn step_with_display_request(
        &mut self,
        request: FeatureFrameRequest,
    ) -> Result<BevyJepaStepOutput> {
        self.step_internal(BevyJepaStepMode::DisplayPanels(request))
    }

    pub fn step_with_stage_request(
        &mut self,
        request: FeatureFrameRequest,
    ) -> Result<BevyJepaStepOutput> {
        self.step_internal(BevyJepaStepMode::StageRequest(request))
    }

    fn step_internal(&mut self, mode: BevyJepaStepMode) -> Result<BevyJepaStepOutput> {
        let Some(processed) =
            process_runtime_frame(&self.config, &mut self.runtime, &self.device, mode)?
        else {
            anyhow::bail!("camera frame is not ready");
        };
        Ok(BevyJepaStepOutput {
            metrics: processed.metrics,
        })
    }
}

#[derive(Component)]
struct MetricsOverlayRoot {
    visible_display: Display,
}

#[derive(Component)]
struct MetricsStageGrid;

#[derive(Component)]
struct MetricValueText {
    kind: MetricValueKind,
}

#[derive(Component)]
struct MetricGraphBar {
    index: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MetricValueKind {
    Status,
    Model,
    Grid,
    Tokens,
    EncodeTokens,
    Queue,
    Error,
    FpsInput,
    FpsLow,
    FpsHigh,
    FpsRolling,
    StageInputSource,
    StageInputQueue,
    StageMaskPolicy,
    StageMaskWrite,
    StageMaskEncode,
    StageEncodeLatency,
    StageEncodePath,
    StageTttStability,
    StageCacheLatency,
    StageTokenViewLatency,
    StageLowResPcaLatency,
    StagePcaBasis,
    StageAnyUpLatency,
    StageDisplayLatency,
}

#[derive(Resource, Default)]
struct MetricsRollingState {
    samples: VecDeque<MetricRollingSample>,
    last_sample_key: Option<u64>,
}

#[derive(Clone, Copy)]
struct MetricRollingSample {
    fps: f64,
    viewer_us: u64,
}

#[derive(Resource, Default)]
struct JepaControlsState {
    expanded: bool,
}

#[derive(Resource, Default)]
struct JepaActiveSlider {
    entity: Option<Entity>,
}

#[derive(Component)]
struct ControlsPanel;

#[derive(Component)]
struct ControlsSummaryText;

#[derive(Component)]
struct ControlsHelpText;

#[derive(Component)]
struct JepaControlHelp {
    text: &'static str,
}

#[derive(Component)]
struct HighResPanelElement;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum JepaControlAction {
    TogglePanel,
    ModelTtt,
    ModelBase,
    PipelineSparse,
    PipelineDense,
    Resolution256,
    Resolution512,
    AnyUpOff,
    AnyUpEvery8,
    AnyUpEvery1,
    AnyUpEfficientLocal,
    AnyUpUpstreamMasked,
    PatchRefresh,
    SubthresholdRefresh,
    AgeRefresh,
    BlueNoiseRefresh,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum JepaControlReset {
    None,
    Visual,
    Rebuild,
}

#[derive(Component)]
struct JepaControlButton {
    action: JepaControlAction,
}

#[derive(Component)]
struct JepaControlButtonText {
    action: JepaControlAction,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum JepaControlSliderKind {
    PatchDiffThreshold,
    ContextDensity,
    MinContextDensity,
    DenseFallbackDensity,
    SubthresholdTrigger,
    AgeIntervalFrames,
    BlueNoiseDensity,
}

#[derive(Component)]
struct JepaControlSlider {
    kind: JepaControlSliderKind,
}

#[derive(Component)]
struct JepaControlSliderFill {
    kind: JepaControlSliderKind,
}

#[derive(Component)]
struct JepaControlSliderValueText {
    kind: JepaControlSliderKind,
}

pub struct BevyJepaPlugin;

impl Plugin for BevyJepaPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<JepaPanelTextures>()
            .init_resource::<JepaRuntime>()
            .init_resource::<BevyJepaMetrics>()
            .init_resource::<MetricsRollingState>()
            .init_resource::<JepaControlsState>()
            .init_resource::<JepaActiveSlider>()
            .add_systems(Startup, setup_metrics_overlay)
            .add_systems(Startup, setup_controls_ui)
            .add_systems(Update, setup_ui)
            .add_systems(Update, process_jepa_frame)
            .add_systems(Update, update_metrics_overlay)
            .add_systems(Update, update_controls_ui)
            .add_systems(Update, control_button_interactions)
            .add_systems(Update, control_slider_interactions)
            .add_systems(Update, update_panel_layout)
            .add_systems(Update, fit_visualization_node)
            .add_systems(Update, keyboard_controls)
            .add_systems(Update, clear_completed_gpu_uploads);
    }
}

pub fn viewer_app(config: BevyJepaConfig) -> App {
    let mut app = App::new();
    app.insert_resource(config)
        .add_plugins(
            DefaultPlugins
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                        features: WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                        ..default()
                    })),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "burn_jepa sparse feature viewer".to_string(),
                        canvas: Some("#bevy".to_string()),
                        fit_canvas_to_parent: true,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(BevyBurnBridgePlugin::<JepaBevyBackend>::default())
        .add_plugins(BevyJepaPlugin);
    app
}

pub fn run_app(config: BevyJepaConfig) -> AppExit {
    let exit = viewer_app(config).run();

    #[cfg(not(target_arch = "wasm32"))]
    if let Some(sender) = platform::camera::APP_RUN_SENDER.get() {
        let _ = sender.send(());
    }

    exit
}

pub fn run_once() -> Result<BevyJepaMetrics> {
    let device = JepaBevyDevice::default();
    let config = BevyJepaConfig {
        encoder_source: BevyJepaEncoderSource::TinyTest,
        source: BevyJepaFrameSource::SyntheticLocalMotion,
        ..BevyJepaConfig::default()
    };
    let mut runtime = JepaRuntime::default();
    Ok(process_runtime_frame(
        &config,
        &mut runtime,
        &device,
        BevyJepaStepMode::DisplayPanels(FeatureFrameRequest::full_pca()),
    )?
    .expect("synthetic source always produces a frame")
    .metrics)
}

impl JepaRuntime {
    fn ensure_pipeline(
        &mut self,
        config: &BevyJepaConfig,
        device: &JepaBevyDevice,
    ) -> Result<(&mut FeatureFramePipeline<JepaBevyBackend>, VJepaConfig)> {
        let image_size = config.pipeline_image_size();
        let signature = RuntimePipelineSignature::new(config, image_size);
        let needs_init = self
            .pipeline_signature
            .as_ref()
            .is_none_or(|current| current != &signature);
        if needs_init {
            let (encoder, model_config) = load_viewer_encoder(config, image_size, device)?;
            let anyup = load_viewer_anyup(config, device)?;
            let pipeline_config = FeatureFramePipelineConfig {
                anyup_q_chunk_size: Some(config.anyup_q_chunk_size.max(1)),
                update_pca_online: false,
                pca_update: config.pca_update_config(),
                measurement: measurement_config(config),
                ttt_runtime: config.ttt_runtime,
                ..FeatureFramePipelineConfig::default()
            };
            self.pipeline = Some(FeatureFramePipeline::<JepaBevyBackend>::new_with_encoder(
                encoder,
                anyup,
                &model_config,
                pipeline_config,
                1,
                [image_size, image_size],
                device,
            )?);
            if config.prewarm_shape_buckets {
                let pipeline = self.pipeline.as_mut().expect("pipeline initialized");
                match prewarm_feature_frame_shapes(config, pipeline, image_size, device) {
                    Ok(Some(report)) => log(&format!(
                        "bevy_jepa: prewarmed sparse token widths {:?} in {:.1} ms",
                        report.token_widths,
                        micros_to_ms(report.total_us)
                    )),
                    Ok(None) => {}
                    Err(error) => {
                        let _ = pipeline.reset_visualization_state();
                        log(&format!(
                            "bevy_jepa: sparse shape prewarm skipped after error: {error:#}"
                        ));
                    }
                }
            }
            let pipeline = self.pipeline.as_ref().expect("pipeline initialized");
            self.pipeline_grid = Some(pipeline.grid());
            self.pipeline_patch_size = Some(model_config.patch_size);
            self.model_config = Some(model_config);
            self.pipeline_signature = Some(signature);
            self.pending_stage = None;
            self.high_res_runtime = None;
            self.high_res_signature = None;
            self.high_res_task = None;
            self.pending_high_res = None;
            self.prev_image = None;
            self.prev_rgba = None;
            self.prev_stage_image = None;
            self.prev_stage_rgba = None;
            self.frame_index = 0;
            self.last_high_res_completion_at = None;
            self.last_high_res_anyup_context_us = 0;
            self.last_high_res_anyup_decode_us = 0;
            self.last_high_res_pca_us = 0;
            self.last_high_res_display_tensor_us = 0;
        }
        let model_config = self
            .model_config
            .clone()
            .expect("model config initialized with pipeline");
        let pipeline = self.pipeline.as_mut().expect("pipeline initialized");
        Ok((pipeline, model_config))
    }

    fn take_pipeline(
        &mut self,
        config: &BevyJepaConfig,
        device: &JepaBevyDevice,
    ) -> Result<(
        FeatureFramePipeline<JepaBevyBackend>,
        VJepaConfig,
        RuntimePipelineSignature,
    )> {
        let image_size = config.pipeline_image_size();
        let signature = RuntimePipelineSignature::new(config, image_size);
        let _ = self.ensure_pipeline(config, device)?;
        let model_config = self
            .model_config
            .clone()
            .expect("model config initialized with pipeline");
        let pipeline = self.pipeline.take().expect("pipeline initialized");
        Ok((pipeline, model_config, signature))
    }

    fn ensure_high_res_runtime(
        &mut self,
        config: &BevyJepaConfig,
        device: &JepaBevyDevice,
    ) -> Result<()> {
        let image_size = config.pipeline_image_size();
        let signature = RuntimePipelineSignature::new(config, image_size);
        let needs_init = self
            .high_res_signature
            .as_ref()
            .is_none_or(|current| current != &signature);
        if !needs_init {
            return Ok(());
        }
        let Some(model_config) = self.model_config.as_ref() else {
            bail!("cannot initialize AnyUp worker before JEPA pipeline model config");
        };
        let anyup = load_viewer_anyup(config, device)?;
        let image_size = [image_size, image_size];
        let grid = TokenGridShape::new(
            1,
            image_size[0] / model_config.patch_size.max(1),
            image_size[1] / model_config.patch_size.max(1),
        );
        self.high_res_runtime = Some(AnyUpHighResRuntime {
            anyup_image_grid: anyup.prepare_image_grid(image_size, device),
            anyup,
            image_size,
            grid,
            q_chunk_size: config.anyup_q_chunk_size.max(1),
        });
        self.high_res_signature = Some(signature);
        self.high_res_task = None;
        self.pending_high_res = None;
        Ok(())
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fn enqueue_high_res_frame(
        &mut self,
        config: &BevyJepaConfig,
        input: HighResFrameInput,
        device: &JepaBevyDevice,
    ) -> Result<()> {
        self.ensure_high_res_runtime(config, device)?;
        if self.high_res_task.is_none() {
            #[cfg(target_arch = "wasm32")]
            {
                let _ = input;
                self.dropped_frames = self.dropped_frames.saturating_add(1);
                bail!(
                    "AnyUp high-res worker is disabled in the browser build; set high-res cadence to off"
                );
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                let runtime = self
                    .high_res_runtime
                    .take()
                    .expect("high-res AnyUp runtime initialized");
                self.high_res_task = Some(spawn_high_res_task(runtime, input));
            }
        } else if self.pending_high_res.replace(input).is_some() {
            self.dropped_frames = self.dropped_frames.saturating_add(1);
            self.overwritten_frames = self.overwritten_frames.saturating_add(1);
        }
        Ok(())
    }

    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    fn run_high_res_frame_inline(
        &mut self,
        config: &BevyJepaConfig,
        input: HighResFrameInput,
        device: &JepaBevyDevice,
    ) -> Result<HighResProcessedFrame> {
        self.ensure_high_res_runtime(config, device)?;
        let runtime = self
            .high_res_runtime
            .as_mut()
            .expect("high-res AnyUp runtime initialized");
        run_high_res_anyup_step(runtime, input)
    }

    fn record_input_frame(&mut self, sequence: u64) {
        let now = viewer_now();
        if let Some(previous) = self.last_input_at {
            let seconds = viewer_seconds_since(now, previous);
            if seconds.is_finite() && seconds > 0.0 {
                self.input_fps = 1.0 / seconds;
            }
        }
        self.last_input_at = Some(now);
        self.input_frames_seen = self.input_frames_seen.saturating_add(1);
        self.latest_input_sequence = sequence;
    }

    fn record_completion(&mut self, high_res_updated: bool) {
        let now = viewer_now();
        if let Some(previous) = self.last_completion_at {
            let seconds = viewer_seconds_since(now, previous);
            if seconds.is_finite() && seconds > 0.0 {
                self.low_res_fps = 1.0 / seconds;
            }
        }
        self.last_completion_at = Some(now);
        self.completed_frames = self.completed_frames.saturating_add(1);
        if high_res_updated {
            self.record_high_res_completion();
        }
    }

    fn record_high_res_completion(&mut self) {
        let now = viewer_now();
        if let Some(previous) = self.last_high_res_completion_at {
            let seconds = viewer_seconds_since(now, previous);
            if seconds.is_finite() && seconds > 0.0 {
                self.high_res_fps = 1.0 / seconds;
            }
        }
        self.last_high_res_completion_at = Some(now);
        self.high_res_frames = self.high_res_frames.saturating_add(1);
    }

    fn apply_high_res_timings(&mut self, processed: &HighResProcessedFrame) {
        self.last_high_res_anyup_context_us = processed.anyup_context_us;
        self.last_high_res_anyup_decode_us = processed.anyup_decode_us;
        self.last_high_res_pca_us = processed.high_res_pca_us;
        self.last_high_res_display_tensor_us = processed.display_tensor_us;
    }

    fn apply_runtime_counts(&self, metrics: &mut BevyJepaMetrics) {
        metrics.input_frames_seen = self.input_frames_seen;
        metrics.input_frame_index = self.latest_input_sequence;
        metrics.completed_frames = self.completed_frames;
        metrics.high_res_frames = self.high_res_frames;
        metrics.input_fps = self.input_fps;
        metrics.low_res_fps = self.low_res_fps;
        metrics.high_res_fps = self.high_res_fps;
        metrics.in_flight_frames = usize::from(self.active_task.is_some())
            + usize::from(self.pending_stage.is_some())
            + usize::from(self.high_res_task.is_some())
            + usize::from(self.pending_high_res.is_some());
        metrics.queue_dropped_frames = self.dropped_frames;
        metrics.queue_overwritten_frames = self.overwritten_frames;
        metrics.stale_completions = self.stale_completions;
        if self.last_high_res_anyup_decode_us > 0 || self.last_high_res_pca_us > 0 {
            metrics.anyup_context_us = self.last_high_res_anyup_context_us;
            metrics.anyup_decode_us = self.last_high_res_anyup_decode_us;
            metrics.high_res_pca_us = self.last_high_res_pca_us;
            metrics.display_tensor_us = metrics
                .display_tensor_us
                .max(self.last_high_res_display_tensor_us);
        }
    }

    fn set_error(&mut self, context: &str, error: String) {
        let log_key = format!("{context}: {error}");
        if self.last_logged_error.as_deref() != Some(log_key.as_str()) {
            crate::error(&format!("bevy_jepa {log_key}"));
            self.last_logged_error = Some(log_key);
        }
        self.last_error = Some(error);
    }

    fn clear_error(&mut self) {
        self.last_error = None;
    }

    fn reset_visual_state(&mut self) {
        if let Some(pipeline) = self.pipeline.as_mut()
            && let Err(err) = pipeline.reset_visualization_state()
        {
            warn(&format!("bevy_jepa visual reset skipped: {err:#}"));
        }
        self.pending_stage = None;
        self.pending_high_res = None;
        self.prev_image = None;
        self.prev_rgba = None;
        self.prev_stage_image = None;
        self.prev_stage_rgba = None;
        self.last_error = None;
    }

    fn rebuild_pipeline_state(&mut self) {
        self.active_task = None;
        self.high_res_task = None;
        self.pipeline = None;
        self.model_config = None;
        self.pipeline_signature = None;
        self.pipeline_grid = None;
        self.pipeline_patch_size = None;
        self.high_res_runtime = None;
        self.high_res_signature = None;
        self.pending_stage = None;
        self.pending_high_res = None;
        self.prev_image = None;
        self.prev_rgba = None;
        self.prev_stage_image = None;
        self.prev_stage_rgba = None;
        self.frame_index = 0;
        self.last_high_res_completion_at = None;
        self.last_high_res_anyup_context_us = 0;
        self.last_high_res_anyup_decode_us = 0;
        self.last_high_res_pca_us = 0;
        self.last_high_res_display_tensor_us = 0;
        self.last_error = None;
    }
}

fn high_res_panel_enabled(config: &BevyJepaConfig) -> bool {
    config.high_res_pca_every > 0
}

fn visible_panel_count(config: &BevyJepaConfig) -> usize {
    if high_res_panel_enabled(config) { 4 } else { 3 }
}

fn visible_panel_count_u16(config: &BevyJepaConfig) -> u16 {
    visible_panel_count(config) as u16
}

fn setup_ui(
    mut commands: Commands,
    config: Res<BevyJepaConfig>,
    mut texture: ResMut<JepaPanelTextures>,
    mut images: ResMut<Assets<Image>>,
    burn_device: Option<Res<BurnDevice>>,
) {
    if texture.root_entity.is_some() {
        return;
    }
    if burn_device
        .as_ref()
        .and_then(|device| device.device())
        .is_none()
    {
        return;
    }

    texture.input_image = images.add(empty_panel_image(1, 1));
    texture.mask_image = images.add(empty_panel_image(1, 1));
    texture.low_res_image = images.add(empty_panel_image(1, 1));
    texture.high_res_image = images.add(empty_panel_image(1, 1));

    let mut root = commands.spawn(Node {
        position_type: PositionType::Absolute,
        display: Display::Grid,
        width: Val::Percent(100.0),
        height: Val::Percent(100.0),
        align_items: AlignItems::Center,
        justify_items: JustifyItems::Center,
        grid_template_columns: RepeatedGridTrack::flex(visible_panel_count_u16(&config), 1.0),
        grid_template_rows: vec![GridTrack::px(PANEL_LABEL_ROW_HEIGHT), GridTrack::flex(1.0)],
        ..default()
    });
    let root_entity = root.id();
    let mut input_entity = None;
    let mut mask_entity = None;
    let mut low_res_entity = None;
    let mut high_res_entity = None;
    root.with_children(|builder| {
        input_entity = Some(spawn_panel_image(builder, texture.input_image.clone()));
        mask_entity = Some(spawn_panel_image(builder, texture.mask_image.clone()));
        low_res_entity = Some(spawn_panel_image(builder, texture.low_res_image.clone()));
        high_res_entity = Some(spawn_high_res_panel_image(
            builder,
            texture.high_res_image.clone(),
            high_res_panel_enabled(&config),
        ));

        spawn_panel_label(builder, "Input", false, true);
        spawn_panel_label(builder, "Sparse mask", false, true);
        spawn_panel_label(builder, "Token PCA", false, true);
        spawn_panel_label(builder, "AnyUp PCA", true, high_res_panel_enabled(&config));
    });

    texture.input_entity = input_entity;
    texture.mask_entity = mask_entity;
    texture.low_res_entity = low_res_entity;
    texture.high_res_entity = high_res_entity;
    texture.root_entity = Some(root_entity);
    commands
        .entity(root_entity)
        .insert(Name::new("bevy_jepa_panels"));
    commands.spawn(Camera2d);
}

fn setup_metrics_overlay(mut commands: Commands, config: Res<BevyJepaConfig>) {
    commands
        .spawn((
            MetricsOverlayRoot {
                visible_display: Display::Flex,
            },
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(UI_MARGIN_PX),
                left: Val::Px(UI_MARGIN_PX),
                width: Val::Px(METRICS_TOP_WIDTH_PX),
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(8.0),
                align_items: AlignItems::Stretch,
                ..default()
            },
            ZIndex(4),
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    width: Val::Px(318.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(4.0),
                    padding: UiRect::all(Val::Px(8.0)),
                    border: UiRect::all(Val::Px(1.0)),
                    ..default()
                },
                BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.18)),
                BackgroundColor(Color::srgba(0.025, 0.028, 0.032, 0.78)),
            ))
            .with_children(|card| {
                spawn_metric_card_title(card, "Pipeline");
                spawn_metric_value_line(card, MetricValueKind::Status);
                spawn_metric_value_line(card, MetricValueKind::Model);
                spawn_metric_value_line(card, MetricValueKind::Grid);
                spawn_metric_value_line(card, MetricValueKind::Tokens);
                spawn_metric_value_line(card, MetricValueKind::EncodeTokens);
                spawn_metric_value_line(card, MetricValueKind::Queue);
                spawn_metric_value_line(card, MetricValueKind::Error);
            });

            root.spawn((
                Node {
                    width: Val::Px(226.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(4.0),
                    padding: UiRect::all(Val::Px(8.0)),
                    border: UiRect::all(Val::Px(1.0)),
                    ..default()
                },
                BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.18)),
                BackgroundColor(Color::srgba(0.025, 0.028, 0.032, 0.78)),
            ))
            .with_children(|card| {
                spawn_metric_card_title(card, "Throughput");
                card.spawn(Node {
                    height: Val::Px(METRICS_GRAPH_HEIGHT_PX),
                    width: Val::Percent(100.0),
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::FlexEnd,
                    column_gap: Val::Px(1.0),
                    padding: UiRect::horizontal(Val::Px(1.0)),
                    ..default()
                })
                .with_children(|graph| {
                    for index in 0..METRICS_GRAPH_BARS {
                        graph.spawn((
                            MetricGraphBar { index },
                            Node {
                                width: Val::Px(2.0),
                                height: Val::Px(1.0),
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.22, 0.44, 0.68)),
                        ));
                    }
                });
                spawn_metric_value_line(card, MetricValueKind::FpsRolling);
                spawn_metric_value_line(card, MetricValueKind::FpsInput);
                spawn_metric_value_line(card, MetricValueKind::FpsLow);
                spawn_metric_value_line(card, MetricValueKind::FpsHigh);
            });
        });

    commands
        .spawn((
            MetricsOverlayRoot {
                visible_display: Display::Grid,
            },
            MetricsStageGrid,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(UI_MARGIN_PX),
                right: Val::Px(UI_MARGIN_PX),
                bottom: Val::Px(UI_MARGIN_PX),
                height: Val::Px(METRICS_STAGE_HEIGHT_PX),
                display: Display::Grid,
                grid_template_columns: RepeatedGridTrack::flex(
                    visible_panel_count_u16(&config),
                    1.0,
                ),
                column_gap: Val::Px(8.0),
                ..default()
            },
            ZIndex(3),
        ))
        .with_children(|grid| {
            spawn_stage_metrics_card(
                grid,
                "Input",
                &[
                    MetricValueKind::StageInputSource,
                    MetricValueKind::FpsInput,
                    MetricValueKind::StageInputQueue,
                ],
                false,
            );
            spawn_stage_metrics_card(
                grid,
                "Mask + JEPA",
                &[
                    MetricValueKind::StageMaskPolicy,
                    MetricValueKind::StageMaskWrite,
                    MetricValueKind::StageMaskEncode,
                    MetricValueKind::StageEncodePath,
                    MetricValueKind::StageEncodeLatency,
                    MetricValueKind::StageTttStability,
                ],
                false,
            );
            spawn_stage_metrics_card(
                grid,
                "Token Cache + PCA",
                &[
                    MetricValueKind::StageCacheLatency,
                    MetricValueKind::StageTokenViewLatency,
                    MetricValueKind::StageLowResPcaLatency,
                    MetricValueKind::StagePcaBasis,
                ],
                false,
            );
            spawn_stage_metrics_card(
                grid,
                "AnyUp + Display",
                &[
                    MetricValueKind::StageAnyUpLatency,
                    MetricValueKind::StageDisplayLatency,
                    MetricValueKind::FpsHigh,
                ],
                true,
            );
        });
}

fn spawn_stage_metrics_card(
    builder: &mut ChildSpawnerCommands<'_>,
    title: &'static str,
    fields: &[MetricValueKind],
    high_res: bool,
) {
    let mut entity = builder.spawn((
        Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(4.0),
            padding: UiRect::all(Val::Px(8.0)),
            border: UiRect::all(Val::Px(1.0)),
            overflow: Overflow::clip(),
            ..default()
        },
        BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.18)),
        BackgroundColor(Color::srgba(0.025, 0.028, 0.032, 0.74)),
    ));
    if high_res {
        entity.insert(HighResPanelElement);
    }
    entity.with_children(|card| {
        spawn_metric_card_title(card, title);
        for &kind in fields {
            spawn_metric_value_line(card, kind);
        }
    });
}

fn spawn_metric_card_title(builder: &mut ChildSpawnerCommands<'_>, title: &'static str) {
    builder.spawn((
        Text(title.to_string()),
        TextFont {
            font_size: bevy::text::FontSize::Px(12.0),
            ..default()
        },
        TextColor(Color::srgb(0.72, 0.78, 0.86)),
        Node {
            height: Val::Px(16.0),
            overflow: Overflow::clip(),
            ..default()
        },
    ));
}

fn spawn_metric_value_line(builder: &mut ChildSpawnerCommands<'_>, kind: MetricValueKind) {
    builder.spawn((
        MetricValueText { kind },
        Text(String::new()),
        TextFont {
            font_size: bevy::text::FontSize::Px(11.0),
            ..default()
        },
        TextColor(Color::srgb(0.92, 0.94, 0.96)),
        Node {
            height: Val::Px(16.0),
            overflow: Overflow::clip(),
            ..default()
        },
    ));
}

fn setup_controls_ui(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(CONTROL_TOP_PX),
                right: Val::Px(UI_MARGIN_PX),
                width: Val::Px(CONTROL_PANEL_WIDTH_PX),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(CONTROL_ROW_GAP_PX),
                align_items: AlignItems::Stretch,
                ..default()
            },
            ZIndex(5),
        ))
        .with_children(|builder| {
            spawn_control_button(builder, JepaControlAction::TogglePanel, "Settings");
            builder
                .spawn((
                    ControlsPanel,
                    Node {
                        display: Display::None,
                        flex_direction: FlexDirection::Column,
                        row_gap: Val::Px(CONTROL_ROW_GAP_PX),
                        padding: UiRect::all(Val::Px(8.0)),
                        border: UiRect::all(Val::Px(1.0)),
                        ..default()
                    },
                    BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.20)),
                    BackgroundColor(Color::srgba(0.025, 0.028, 0.032, 0.86)),
                ))
                .with_children(|panel| {
                    panel.spawn((
                        ControlsSummaryText,
                        Text(String::new()),
                        TextFont {
                            font_size: bevy::text::FontSize::Px(12.0),
                            ..default()
                        },
                        TextColor(Color::srgb(0.90, 0.92, 0.95)),
                        Node {
                            min_height: Val::Px(34.0),
                            ..default()
                        },
                    ));
                    panel.spawn((
                        ControlsHelpText,
                        Text(default_controls_help().to_string()),
                        TextFont {
                            font_size: bevy::text::FontSize::Px(11.0),
                            ..default()
                        },
                        TextColor(Color::srgb(0.66, 0.72, 0.80)),
                        Node {
                            min_height: Val::Px(30.0),
                            overflow: Overflow::clip(),
                            ..default()
                        },
                    ));
                    spawn_control_section_title(
                        panel,
                        "Model",
                        "Choose the encoder package and the token input mode.",
                    );
                    spawn_control_row(
                        panel,
                        "Model package",
                        &[
                            (JepaControlAction::ModelTtt, "TTT"),
                            (JepaControlAction::ModelBase, "Base"),
                        ],
                    );
                    spawn_control_row(
                        panel,
                        "JEPA input",
                        &[
                            (JepaControlAction::PipelineSparse, "Sparse"),
                            (JepaControlAction::PipelineDense, "Dense"),
                        ],
                    );
                    spawn_control_row(
                        panel,
                        "Resolution",
                        &[
                            (JepaControlAction::Resolution256, "256"),
                            (JepaControlAction::Resolution512, "512"),
                        ],
                    );
                    spawn_control_section_title(
                        panel,
                        "High-Resolution Decode",
                        "AnyUp runs off the low-res cache and may be decimated.",
                    );
                    spawn_control_row(
                        panel,
                        "AnyUp cadence",
                        &[
                            (JepaControlAction::AnyUpOff, "Off"),
                            (JepaControlAction::AnyUpEvery8, "1/8"),
                            (JepaControlAction::AnyUpEvery1, "1/1"),
                        ],
                    );
                    spawn_control_row(
                        panel,
                        "Attention",
                        &[
                            (JepaControlAction::AnyUpEfficientLocal, "Local"),
                            (JepaControlAction::AnyUpUpstreamMasked, "Masked"),
                        ],
                    );
                    spawn_control_section_title(
                        panel,
                        "Patch-Diff Mask",
                        "Threshold selects changed patches; refresh prevents stale tokens.",
                    );
                    spawn_slider_row(
                        panel,
                        "Diff threshold",
                        JepaControlSliderKind::PatchDiffThreshold,
                    );
                    spawn_slider_row(panel, "Max density", JepaControlSliderKind::ContextDensity);
                    spawn_slider_row(
                        panel,
                        "Min density",
                        JepaControlSliderKind::MinContextDensity,
                    );
                    spawn_slider_row(
                        panel,
                        "Dense cutoff",
                        JepaControlSliderKind::DenseFallbackDensity,
                    );
                    spawn_control_row(
                        panel,
                        "Refresh",
                        &[
                            (JepaControlAction::PatchRefresh, "Enable"),
                            (JepaControlAction::SubthresholdRefresh, "Subthr"),
                            (JepaControlAction::AgeRefresh, "Age"),
                            (JepaControlAction::BlueNoiseRefresh, "Blue"),
                        ],
                    );
                    spawn_slider_row(
                        panel,
                        "Subthr trigger",
                        JepaControlSliderKind::SubthresholdTrigger,
                    );
                    spawn_slider_row(
                        panel,
                        "Age interval",
                        JepaControlSliderKind::AgeIntervalFrames,
                    );
                    spawn_slider_row(
                        panel,
                        "Blue density",
                        JepaControlSliderKind::BlueNoiseDensity,
                    );
                });
        });
}

fn spawn_control_section_title(
    builder: &mut ChildSpawnerCommands<'_>,
    title: &'static str,
    description: &'static str,
) {
    builder
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(1.0),
            margin: UiRect::top(Val::Px(4.0)),
            ..default()
        })
        .with_children(|section| {
            section.spawn((
                Text(title.to_string()),
                TextFont {
                    font_size: bevy::text::FontSize::Px(12.0),
                    ..default()
                },
                TextColor(Color::srgb(0.78, 0.84, 0.92)),
                Node {
                    height: Val::Px(15.0),
                    overflow: Overflow::clip(),
                    ..default()
                },
            ));
            section.spawn((
                Text(description.to_string()),
                TextFont {
                    font_size: bevy::text::FontSize::Px(10.5),
                    ..default()
                },
                TextColor(Color::srgb(0.56, 0.62, 0.70)),
                Node {
                    min_height: Val::Px(14.0),
                    overflow: Overflow::clip(),
                    ..default()
                },
            ));
        });
}

fn spawn_control_row(
    builder: &mut ChildSpawnerCommands<'_>,
    label: &'static str,
    actions: &[(JepaControlAction, &'static str)],
) {
    builder
        .spawn(Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(CONTROL_ROW_GAP_PX),
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text(label.to_string()),
                TextFont {
                    font_size: bevy::text::FontSize::Px(12.0),
                    ..default()
                },
                TextColor(Color::srgb(0.72, 0.75, 0.80)),
                Node {
                    width: Val::Px(CONTROL_LABEL_WIDTH_PX),
                    ..default()
                },
            ));
            for &(action, text) in actions {
                spawn_control_button(row, action, text);
            }
        });
}

fn spawn_slider_row(
    builder: &mut ChildSpawnerCommands<'_>,
    label: &'static str,
    kind: JepaControlSliderKind,
) {
    builder
        .spawn(Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(CONTROL_ROW_GAP_PX),
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text(label.to_string()),
                TextFont {
                    font_size: bevy::text::FontSize::Px(12.0),
                    ..default()
                },
                TextColor(Color::srgb(0.72, 0.75, 0.80)),
                Node {
                    width: Val::Px(CONTROL_LABEL_WIDTH_PX),
                    ..default()
                },
            ));
            row.spawn((
                Button,
                Interaction::None,
                RelativeCursorPosition::default(),
                JepaControlSlider { kind },
                JepaControlHelp {
                    text: slider_help_text(kind),
                },
                Node {
                    width: Val::Px(CONTROL_SLIDER_WIDTH_PX),
                    height: Val::Px(CONTROL_SLIDER_HEIGHT_PX),
                    border: UiRect::all(Val::Px(1.0)),
                    overflow: Overflow::clip(),
                    align_items: AlignItems::Stretch,
                    ..default()
                },
                BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.22)),
                BackgroundColor(Color::srgb(0.08, 0.09, 0.11)),
            ))
            .with_child((
                JepaControlSliderFill { kind },
                Node {
                    width: Val::Percent(0.0),
                    height: Val::Percent(100.0),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.27, 0.45, 0.72)),
            ));
            row.spawn((
                JepaControlSliderValueText { kind },
                Text(String::new()),
                TextFont {
                    font_size: bevy::text::FontSize::Px(12.0),
                    ..default()
                },
                TextColor(Color::srgb(0.90, 0.92, 0.95)),
                Node {
                    width: Val::Px(74.0),
                    ..default()
                },
            ));
        });
}

fn spawn_control_button(
    builder: &mut ChildSpawnerCommands<'_>,
    action: JepaControlAction,
    text: &'static str,
) {
    let is_toggle = action == JepaControlAction::TogglePanel;
    builder
        .spawn((
            Button,
            Interaction::None,
            JepaControlButton { action },
            JepaControlHelp {
                text: control_help_text(action),
            },
            Node {
                height: Val::Px(CONTROL_BUTTON_HEIGHT_PX),
                width: if is_toggle { Val::Px(104.0) } else { Val::Auto },
                min_width: Val::Px(if is_toggle { 104.0 } else { 58.0 }),
                padding: UiRect::horizontal(Val::Px(8.0)),
                align_self: if is_toggle {
                    AlignSelf::FlexEnd
                } else {
                    AlignSelf::Auto
                },
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(Color::srgb(0.14, 0.16, 0.20)),
        ))
        .with_child((
            JepaControlButtonText { action },
            Text(text.to_string()),
            TextFont {
                font_size: bevy::text::FontSize::Px(12.0),
                ..default()
            },
            TextColor(Color::WHITE),
        ));
}

fn process_jepa_frame(world: &mut World) {
    let config = world.resource::<BevyJepaConfig>().clone();
    let Some(device) = world
        .get_resource::<BurnDevice>()
        .and_then(|device| device.device())
        .cloned()
    else {
        return;
    };

    let transfer = config.display_transfer;
    let high_res_completed = {
        let mut runtime = world.resource_mut::<JepaRuntime>();
        poll_high_res_task(&config, &mut runtime)
    };
    if let Some(completed) = high_res_completed {
        apply_high_res_completion_to_world(world, transfer, completed);
    }

    let completed = {
        let mut runtime = world.resource_mut::<JepaRuntime>();
        poll_jepa_task(&config, &mut runtime, &device)
    };
    if let Some(completed) = completed {
        apply_jepa_completion_to_world(world, &config, &device, transfer, completed);
    }

    let result = {
        let mut runtime = world.resource_mut::<JepaRuntime>();
        process_runtime_source_frame(&config, &mut runtime, &device)
    };

    match result {
        Ok(Some(input)) => {
            let inline_stage = input.stage;
            apply_input_panel_to_world(world, input.panel, transfer);
            let mut metrics = world.resource::<BevyJepaMetrics>().clone();
            {
                let runtime = world.resource::<JepaRuntime>();
                runtime.apply_runtime_counts(&mut metrics);
            }
            metrics.encoder_source = config.encoder_source;
            metrics.input_frame_index = input.sequence;
            metrics.frame_source = input.source;
            metrics.camera_frame_received = input.camera_frame_received;
            metrics.mask_source = config.mask_source;
            metrics.display_transfer = config.display_transfer;
            world.resource_mut::<JepaRuntime>().clear_error();
            *world.resource_mut::<BevyJepaMetrics>() = metrics;
            if let Some(completed) = inline_stage {
                apply_jepa_completion_to_world(world, &config, &device, transfer, completed);
            }
        }
        Ok(None) => {
            world.resource_mut::<JepaRuntime>().clear_error();
            let mut metrics = world.resource::<BevyJepaMetrics>().clone();
            if !metrics.frame_ready {
                metrics.encoder_source = config.encoder_source;
                metrics.frame_source = config.source;
                metrics.camera_frame_received = false;
                metrics.mask_source = config.mask_source;
                metrics.display_transfer = config.display_transfer;
            }
            {
                let runtime = world.resource::<JepaRuntime>();
                runtime.apply_runtime_counts(&mut metrics);
            }
            metrics.last_error = None;
            *world.resource_mut::<BevyJepaMetrics>() = metrics;
        }
        Err(err) => {
            let err = err.to_string();
            {
                let mut runtime = world.resource_mut::<JepaRuntime>();
                runtime.set_error("source frame", err.clone());
            }
            world.resource_mut::<BevyJepaMetrics>().last_error = Some(err);
        }
    }
}

fn apply_jepa_completion_to_world(
    world: &mut World,
    config: &BevyJepaConfig,
    device: &JepaBevyDevice,
    transfer: BevyJepaDisplayTransfer,
    completed: Result<StageProcessedFrame, String>,
) {
    #[cfg(target_arch = "wasm32")]
    let mut high_res_completed = None;
    #[cfg(not(target_arch = "wasm32"))]
    let high_res_completed: Option<Result<HighResProcessedFrame, String>> = None;
    match completed {
        Ok(processed) => {
            let mut metrics = processed.metrics.clone();
            {
                let mut runtime = world.resource_mut::<JepaRuntime>();
                runtime.record_completion(processed.high_res_updated);
                let mut enqueue_error = None;
                if let Some(input) = processed.high_res_input {
                    #[cfg(target_arch = "wasm32")]
                    match runtime.run_high_res_frame_inline(config, input, device) {
                        Ok(processed) => {
                            high_res_completed = Some(Ok(processed));
                        }
                        Err(err) => {
                            enqueue_error = Some(err.to_string());
                        }
                    }

                    #[cfg(not(target_arch = "wasm32"))]
                    if let Err(err) = runtime.enqueue_high_res_frame(config, input, device) {
                        enqueue_error = Some(err.to_string());
                    }
                }
                if let Some(err) = enqueue_error.as_ref() {
                    runtime.set_error("AnyUp enqueue", err.clone());
                } else {
                    runtime.clear_error();
                }
                metrics.last_error = enqueue_error;
                runtime.apply_runtime_counts(&mut metrics);
            }
            apply_stage_panels_to_world(world, processed.panels, transfer);
            *world.resource_mut::<BevyJepaMetrics>() = metrics;
            if let Some(completed) = high_res_completed {
                apply_high_res_completion_to_world(world, transfer, completed);
            }
        }
        Err(err) => {
            {
                let mut runtime = world.resource_mut::<JepaRuntime>();
                runtime.set_error("JEPA worker", err.clone());
            }
            world.resource_mut::<BevyJepaMetrics>().last_error = Some(err);
        }
    }
}

fn apply_high_res_completion_to_world(
    world: &mut World,
    transfer: BevyJepaDisplayTransfer,
    completed: Result<HighResProcessedFrame, String>,
) {
    match completed {
        Ok(processed) => {
            let mut metrics = world.resource::<BevyJepaMetrics>().clone();
            {
                let mut runtime = world.resource_mut::<JepaRuntime>();
                runtime.record_high_res_completion();
                runtime.apply_high_res_timings(&processed);
                runtime.clear_error();
                runtime.apply_runtime_counts(&mut metrics);
            }
            metrics.frame_index = processed.id.sequence;
            metrics.frame_ready = true;
            metrics.last_error = None;
            metrics.viewer_total_us = processed.total_us;
            apply_high_res_panel_to_world(world, processed.panels, transfer);
            *world.resource_mut::<BevyJepaMetrics>() = metrics;
        }
        Err(err) => {
            {
                let mut runtime = world.resource_mut::<JepaRuntime>();
                runtime.set_error("AnyUp worker", err.clone());
            }
            world.resource_mut::<BevyJepaMetrics>().last_error = Some(err);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn poll_jepa_task(
    config: &BevyJepaConfig,
    runtime: &mut JepaRuntime,
    device: &JepaBevyDevice,
) -> Option<Result<StageProcessedFrame, String>> {
    let task = runtime.active_task.as_mut()?;
    let output = block_on(future::poll_once(task))?;
    runtime.active_task = None;
    finish_jepa_task_output(config, runtime, device, output)
}

#[cfg(target_arch = "wasm32")]
fn poll_jepa_task(
    _config: &BevyJepaConfig,
    _runtime: &mut JepaRuntime,
    _device: &JepaBevyDevice,
) -> Option<Result<StageProcessedFrame, String>> {
    None
}

fn finish_jepa_task_output(
    config: &BevyJepaConfig,
    runtime: &mut JepaRuntime,
    device: &JepaBevyDevice,
    output: JepaAsyncTaskOutput,
) -> Option<Result<StageProcessedFrame, String>> {
    let current_signature = RuntimePipelineSignature::new(config, config.pipeline_image_size());
    if output.signature == current_signature {
        let pipeline = output.pipeline;
        #[cfg(target_arch = "wasm32")]
        {
            let _ = device;
            runtime.pending_stage = None;
            runtime.pipeline = Some(pipeline);
        }

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(pending) = runtime.pending_stage.take() {
            if pending.signature == current_signature {
                let Some(model_config) = runtime.model_config.clone() else {
                    runtime.pipeline = Some(pipeline);
                    return Some(Err(
                        "cannot admit pending JEPA frame without model config".to_string()
                    ));
                };
                let image =
                    match pending_stage_image(&pending, config.pipeline_image_size(), device) {
                        Ok(image) => image,
                        Err(err) => return Some(Err(err.to_string())),
                    };
                match spawn_admitted_jepa_task(
                    pending.config,
                    pending.signature,
                    pipeline,
                    image.clone(),
                    pending.rgba.as_ref(),
                    runtime.prev_stage_image.as_ref(),
                    runtime.prev_stage_rgba.as_ref(),
                    &model_config,
                    pending.id,
                    pending.grid,
                    pending.patch_size,
                    pending.frame_source,
                    pending.camera_frame_received,
                    pending.request,
                ) {
                    Ok(task) => {
                        runtime.prev_stage_image = Some(image);
                        runtime.prev_stage_rgba = pending.rgba;
                        runtime.active_task = Some(task);
                    }
                    Err(err) => {
                        return Some(Err(err.to_string()));
                    }
                }
            } else {
                runtime.stale_completions = runtime.stale_completions.saturating_add(1);
                runtime.pipeline = Some(pipeline);
            }
        } else {
            runtime.pipeline = Some(pipeline);
        }
        Some(output.result)
    } else {
        runtime.stale_completions = runtime.stale_completions.saturating_add(1);
        runtime.pending_stage = None;
        Some(Err(
            "discarded stale JEPA completion after pipeline config changed".to_string(),
        ))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn poll_high_res_task(
    config: &BevyJepaConfig,
    runtime: &mut JepaRuntime,
) -> Option<Result<HighResProcessedFrame, String>> {
    let task = runtime.high_res_task.as_mut()?;
    let output = block_on(future::poll_once(task))?;
    runtime.high_res_task = None;
    let current_signature = RuntimePipelineSignature::new(config, config.pipeline_image_size());
    if output.signature == current_signature {
        runtime.high_res_runtime = Some(output.runtime);
        if let Some(pending) = runtime.pending_high_res.take() {
            if pending.signature == current_signature {
                let high_res_runtime = runtime
                    .high_res_runtime
                    .take()
                    .expect("high-res AnyUp runtime initialized");
                runtime.high_res_task = Some(spawn_high_res_task(high_res_runtime, pending));
            } else {
                runtime.stale_completions = runtime.stale_completions.saturating_add(1);
            }
        }
        Some(output.result)
    } else {
        runtime.stale_completions = runtime.stale_completions.saturating_add(1);
        runtime.pending_high_res = None;
        Some(Err(
            "discarded stale AnyUp completion after pipeline config changed".to_string(),
        ))
    }
}

#[cfg(target_arch = "wasm32")]
fn poll_high_res_task(
    _config: &BevyJepaConfig,
    _runtime: &mut JepaRuntime,
) -> Option<Result<HighResProcessedFrame, String>> {
    None
}

fn process_runtime_source_frame(
    config: &BevyJepaConfig,
    runtime: &mut JepaRuntime,
    device: &JepaBevyDevice,
) -> Result<Option<SourcePreviewFrame>> {
    let frame_index = runtime.frame_index;
    let image_size = config.pipeline_image_size();
    let Some(source) = poll_source_input_node(config, runtime, frame_index, image_size, device)?
    else {
        return Ok(None);
    };
    runtime.record_input_frame(frame_index);
    let input_panel =
        source_input_panel_from_input_source(config, &source, [image_size, image_size])?;
    let frame_source = source.source;
    let camera_frame_received = source.camera_frame_received;
    let source_rgba = source.rgba;
    #[cfg(target_arch = "wasm32")]
    let mut stage = None;
    #[cfg(not(target_arch = "wasm32"))]
    let stage = None;
    if runtime.active_task.is_none() {
        let (pipeline, model_config, signature) = runtime.take_pipeline(config, device)?;
        let image = source_stage_image(
            source.image.as_ref(),
            source_rgba.as_ref(),
            image_size,
            device,
        )?;
        let grid = pipeline.grid();
        let id = FrameId {
            stream_id: 0,
            sequence: frame_index,
            capture_time_nanos: frame_index.saturating_mul(16_666_667),
        };
        let request = stage_request_for_frame(config, frame_index);
        #[cfg(target_arch = "wasm32")]
        {
            let output = run_admitted_jepa_task_inline(
                config.clone(),
                signature,
                pipeline,
                image.clone(),
                source_rgba.as_ref(),
                runtime.prev_stage_image.as_ref(),
                runtime.prev_stage_rgba.as_ref(),
                &model_config,
                id,
                grid,
                model_config.patch_size,
                frame_source,
                camera_frame_received,
                request,
            )?;
            stage = finish_jepa_task_output(config, runtime, device, output);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            runtime.active_task = Some(spawn_admitted_jepa_task(
                config.clone(),
                signature,
                pipeline,
                image.clone(),
                source_rgba.as_ref(),
                runtime.prev_stage_image.as_ref(),
                runtime.prev_stage_rgba.as_ref(),
                &model_config,
                id,
                grid,
                model_config.patch_size,
                frame_source,
                camera_frame_received,
                request,
            )?);
        }
        runtime.prev_stage_image = Some(image.clone());
        runtime.prev_stage_rgba = source_rgba.clone();
    } else if runtime.pipeline_signature.as_ref()
        == Some(&RuntimePipelineSignature::new(config, image_size))
    {
        if let (Some(grid), Some(patch_size), Some(signature)) = (
            runtime.pipeline_grid,
            runtime.pipeline_patch_size,
            runtime.pipeline_signature.clone(),
        ) {
            let id = FrameId {
                stream_id: 0,
                sequence: frame_index,
                capture_time_nanos: frame_index.saturating_mul(16_666_667),
            };
            let request = stage_request_for_frame(config, frame_index);
            let pending = PendingStageFrame {
                config: config.clone(),
                signature,
                image: source.image.clone(),
                rgba: source_rgba.clone(),
                id,
                grid,
                patch_size,
                frame_source,
                camera_frame_received,
                request,
            };
            if runtime.pending_stage.replace(pending).is_some() {
                runtime.dropped_frames = runtime.dropped_frames.saturating_add(1);
                runtime.overwritten_frames = runtime.overwritten_frames.saturating_add(1);
            }
        } else {
            runtime.dropped_frames = runtime.dropped_frames.saturating_add(1);
        }
    } else {
        runtime.dropped_frames = runtime.dropped_frames.saturating_add(1);
        runtime.overwritten_frames = runtime.overwritten_frames.saturating_add(1);
        runtime.pending_stage = None;
    }
    runtime.prev_image = source.image;
    runtime.prev_rgba = source_rgba;
    runtime.frame_index = runtime.frame_index.saturating_add(1);
    Ok(Some(SourcePreviewFrame {
        panel: input_panel,
        sequence: frame_index,
        source: frame_source,
        camera_frame_received,
        stage,
    }))
}

fn run_jepa_task_output(mut input: AdmittedJepaTaskInput) -> JepaAsyncTaskOutput {
    let result = run_stage_pipeline_step(
        &input.config,
        &mut input.pipeline,
        input.image,
        &input.write_mask,
        &input.encode_mask,
        input.id,
        input.grid,
        input.patch_size,
        input.frame_source,
        input.camera_frame_received,
        input.request,
    )
    .map_err(|err| err.to_string());
    JepaAsyncTaskOutput {
        signature: input.signature,
        pipeline: input.pipeline,
        result,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_jepa_task(input: AdmittedJepaTaskInput) -> Task<JepaAsyncTaskOutput> {
    AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::new)
        .spawn(async move { run_jepa_task_output(input) })
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_high_res_task(
    mut runtime: AnyUpHighResRuntime,
    input: HighResFrameInput,
) -> Task<HighResAsyncTaskOutput> {
    AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::new).spawn(async move {
        let signature = input.signature.clone();
        let result = run_high_res_anyup_step(&mut runtime, input).map_err(|err| err.to_string());
        HighResAsyncTaskOutput {
            signature,
            runtime,
            result,
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn admitted_jepa_task_input(
    config: BevyJepaConfig,
    signature: RuntimePipelineSignature,
    mut pipeline: FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    rgba: Option<&RgbaImage>,
    prev_stage_image: Option<&Tensor<JepaBevyBackend, 4>>,
    prev_stage_rgba: Option<&RgbaImage>,
    model_config: &VJepaConfig,
    id: FrameId,
    grid: TokenGridShape,
    patch_size: usize,
    frame_source: BevyJepaFrameSource,
    camera_frame_received: bool,
    request: FeatureFrameRequest,
) -> Result<AdmittedJepaTaskInput> {
    let masks = run_sparse_mask_node_with_refresh_state(
        &config,
        prev_stage_image,
        prev_stage_rgba,
        rgba,
        &image,
        model_config,
        grid,
        Some(pipeline.patch_diff_refresh_state_mut()),
    )?;
    Ok(AdmittedJepaTaskInput {
        config,
        signature,
        pipeline,
        image,
        write_mask: masks.write_mask,
        encode_mask: masks.encode_mask,
        id,
        grid,
        patch_size,
        frame_source,
        camera_frame_received,
        request,
    })
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn spawn_admitted_jepa_task(
    config: BevyJepaConfig,
    signature: RuntimePipelineSignature,
    pipeline: FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    rgba: Option<&RgbaImage>,
    prev_stage_image: Option<&Tensor<JepaBevyBackend, 4>>,
    prev_stage_rgba: Option<&RgbaImage>,
    model_config: &VJepaConfig,
    id: FrameId,
    grid: TokenGridShape,
    patch_size: usize,
    frame_source: BevyJepaFrameSource,
    camera_frame_received: bool,
    request: FeatureFrameRequest,
) -> Result<Task<JepaAsyncTaskOutput>> {
    Ok(spawn_jepa_task(admitted_jepa_task_input(
        config,
        signature,
        pipeline,
        image,
        rgba,
        prev_stage_image,
        prev_stage_rgba,
        model_config,
        id,
        grid,
        patch_size,
        frame_source,
        camera_frame_received,
        request,
    )?))
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
fn run_admitted_jepa_task_inline(
    config: BevyJepaConfig,
    signature: RuntimePipelineSignature,
    pipeline: FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    rgba: Option<&RgbaImage>,
    prev_stage_image: Option<&Tensor<JepaBevyBackend, 4>>,
    prev_stage_rgba: Option<&RgbaImage>,
    model_config: &VJepaConfig,
    id: FrameId,
    grid: TokenGridShape,
    patch_size: usize,
    frame_source: BevyJepaFrameSource,
    camera_frame_received: bool,
    request: FeatureFrameRequest,
) -> Result<JepaAsyncTaskOutput> {
    Ok(run_jepa_task_output(admitted_jepa_task_input(
        config,
        signature,
        pipeline,
        image,
        rgba,
        prev_stage_image,
        prev_stage_rgba,
        model_config,
        id,
        grid,
        patch_size,
        frame_source,
        camera_frame_received,
        request,
    )?))
}

fn process_runtime_frame(
    config: &BevyJepaConfig,
    runtime: &mut JepaRuntime,
    device: &JepaBevyDevice,
    mode: BevyJepaStepMode,
) -> Result<Option<ProcessedFrame>> {
    let frame_index = runtime.frame_index;
    let image_size = config.pipeline_image_size();
    let Some(source) = poll_source_frame_node(config, runtime, frame_index, image_size, device)?
    else {
        return Ok(None);
    };
    let image = source.image;
    let prev_image = runtime.prev_image.clone();
    let prev_rgba = runtime.prev_rgba.clone();
    let (pipeline, model_config) = runtime.ensure_pipeline(config, device)?;
    let grid = pipeline.grid();
    let masks = run_sparse_mask_node_with_refresh_state(
        config,
        prev_image.as_ref(),
        prev_rgba.as_ref(),
        source.rgba.as_ref(),
        &image,
        &model_config,
        grid,
        Some(pipeline.patch_diff_refresh_state_mut()),
    )?;
    let id = FrameId {
        stream_id: 0,
        sequence: frame_index,
        capture_time_nanos: frame_index.saturating_mul(16_666_667),
    };
    let processed = match mode {
        BevyJepaStepMode::StageOnly => run_stage_step_metrics(
            config,
            pipeline,
            image.clone(),
            &masks.write_mask,
            &masks.encode_mask,
            id,
            grid,
            model_config.patch_size,
            source.source,
            source.camera_frame_received,
        ),
        BevyJepaStepMode::StageRequest(request) => run_stage_step_metrics_with_request(
            config,
            pipeline,
            image.clone(),
            &masks.write_mask,
            &masks.encode_mask,
            id,
            grid,
            model_config.patch_size,
            source.source,
            source.camera_frame_received,
            request,
        ),
        BevyJepaStepMode::DisplayPanels(request) => run_pipeline_step(
            config,
            pipeline,
            image.clone(),
            &masks.write_mask,
            &masks.encode_mask,
            id,
            grid,
            model_config.patch_size,
            source.source,
            source.camera_frame_received,
            request,
        ),
    };
    runtime.prev_image = Some(image);
    runtime.prev_rgba = source.rgba;
    runtime.frame_index += 1;
    processed.map(Some)
}

struct ProcessedFrame {
    metrics: BevyJepaMetrics,
}

struct SourceFrameNodeOutput {
    image: Tensor<JepaBevyBackend, 4>,
    rgba: Option<RgbaImage>,
    source: BevyJepaFrameSource,
    camera_frame_received: bool,
}

struct SourceInputNodeOutput {
    image: Option<Tensor<JepaBevyBackend, 4>>,
    rgba: Option<RgbaImage>,
    source: BevyJepaFrameSource,
    camera_frame_received: bool,
}

struct SourcePreviewFrame {
    panel: InputPanelData,
    sequence: u64,
    source: BevyJepaFrameSource,
    camera_frame_received: bool,
    stage: Option<Result<StageProcessedFrame, String>>,
}

struct StageFrame {
    output: FeatureFrameBatch<JepaBevyBackend>,
    metrics: FeatureFrameMetrics,
    wall_us: u64,
}

struct StageProcessedFrame {
    metrics: BevyJepaMetrics,
    panels: StagePanelData,
    high_res_updated: bool,
    high_res_input: Option<HighResFrameInput>,
}

fn stage_request_for_frame(config: &BevyJepaConfig, frame_index: u64) -> FeatureFrameRequest {
    let _ = (config, frame_index);
    FeatureFrameRequest::low_res()
}

fn high_res_scheduled_for_frame(config: &BevyJepaConfig, frame_index: u64) -> bool {
    config.high_res_pca_every > 0 && frame_index.is_multiple_of(config.high_res_pca_every)
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
struct PendingStageFrame {
    config: BevyJepaConfig,
    signature: RuntimePipelineSignature,
    image: Option<Tensor<JepaBevyBackend, 4>>,
    rgba: Option<RgbaImage>,
    id: FrameId,
    grid: TokenGridShape,
    patch_size: usize,
    frame_source: BevyJepaFrameSource,
    camera_frame_received: bool,
    request: FeatureFrameRequest,
}

struct AdmittedJepaTaskInput {
    config: BevyJepaConfig,
    signature: RuntimePipelineSignature,
    pipeline: FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    write_mask: SparseTokenMask,
    encode_mask: SparseTokenMask,
    id: FrameId,
    grid: TokenGridShape,
    patch_size: usize,
    frame_source: BevyJepaFrameSource,
    camera_frame_received: bool,
    request: FeatureFrameRequest,
}

struct JepaAsyncTaskOutput {
    signature: RuntimePipelineSignature,
    pipeline: FeatureFramePipeline<JepaBevyBackend>,
    result: Result<StageProcessedFrame, String>,
}

fn run_stage_step_with_config_and_request(
    config: &BevyJepaConfig,
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    write_mask: &SparseTokenMask,
    encode_mask: &SparseTokenMask,
    request: FeatureFrameRequest,
) -> Result<StageFrame> {
    let wall_start = viewer_now();
    let measured =
        run_feature_frame_pipeline(config, pipeline, image, write_mask, encode_mask, request)?;
    Ok(StageFrame {
        output: measured.output,
        metrics: measured.metrics,
        wall_us: viewer_elapsed_us(wall_start),
    })
}

fn run_feature_frame_pipeline(
    config: &BevyJepaConfig,
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    write_mask: &SparseTokenMask,
    encode_mask: &SparseTokenMask,
    request: FeatureFrameRequest,
) -> Result<burn_jepa::MeasuredFeatureFrameBatch<JepaBevyBackend>> {
    match config.encode_path {
        BevyJepaEncodePath::DensePatchEmbed => pipeline
            .step_image_with_encode_write_masks_nodes_measured(
                image,
                encode_mask,
                write_mask,
                request,
            ),
        BevyJepaEncodePath::Auto => {
            run_feature_frame_pipeline_auto(pipeline, image, write_mask, encode_mask, request)
        }
        BevyJepaEncodePath::SparsePatchify => run_feature_frame_pipeline_sparse_patchify(
            pipeline,
            image,
            write_mask,
            encode_mask,
            request,
        ),
    }
}

fn run_feature_frame_pipeline_auto(
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    write_mask: &SparseTokenMask,
    encode_mask: &SparseTokenMask,
    request: FeatureFrameRequest,
) -> Result<burn_jepa::MeasuredFeatureFrameBatch<JepaBevyBackend>> {
    #[cfg(feature = "sparse-patchify-wgpu")]
    {
        if encode_mask.is_dense_ordered() {
            pipeline.step_image_with_encode_write_masks_nodes_measured(
                image,
                encode_mask,
                write_mask,
                request,
            )
        } else {
            run_feature_frame_pipeline_sparse_patchify(
                pipeline,
                image,
                write_mask,
                encode_mask,
                request,
            )
        }
    }

    #[cfg(not(feature = "sparse-patchify-wgpu"))]
    {
        pipeline.step_image_with_encode_write_masks_nodes_measured(
            image,
            encode_mask,
            write_mask,
            request,
        )
    }
}

#[derive(Clone, Debug)]
struct ShapePrewarmReport {
    token_widths: Vec<usize>,
    total_us: u64,
}

fn prewarm_feature_frame_shapes(
    config: &BevyJepaConfig,
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image_size: usize,
    device: &JepaBevyDevice,
) -> Result<Option<ShapePrewarmReport>> {
    if !config.uses_bucketed_sparse_encode() || !config.prewarm_shape_buckets {
        return Ok(None);
    }
    if config.encoder_source == BevyJepaEncoderSource::TinyTest {
        return Ok(None);
    }
    let grid = pipeline.grid();
    if grid.len() < 256 {
        return Ok(None);
    }
    let masks = shape_prewarm_masks(grid, config);
    if masks.is_empty() {
        return Ok(None);
    }

    let image = synthetic_image_tensor(0, image_size, device);
    let started = viewer_now();
    let mut token_widths = Vec::with_capacity(masks.len());
    for mask in masks {
        let measured = run_feature_frame_pipeline(
            config,
            pipeline,
            image.clone(),
            &mask,
            &mask,
            FeatureFrameRequest::low_res(),
        )?;
        token_widths.push(measured.metrics.sparse_width);
    }
    sync_bevy_backend(device)?;
    let total_us = viewer_elapsed_us(started);
    pipeline.reset_visualization_state()?;
    Ok(Some(ShapePrewarmReport {
        token_widths,
        total_us,
    }))
}

#[cfg(feature = "sparse-patchify-wgpu")]
fn run_feature_frame_pipeline_sparse_patchify(
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    write_mask: &SparseTokenMask,
    encode_mask: &SparseTokenMask,
    request: FeatureFrameRequest,
) -> Result<burn_jepa::MeasuredFeatureFrameBatch<JepaBevyBackend>> {
    let batch_mask =
        SparseMaskBatch::uniform(encode_mask.clone(), pipeline.batch(), pipeline.device())?;
    let patchify_plan =
        SparsePatchifyBatchPlan::new(batch_mask, pipeline.grid(), pipeline.device())?;
    if write_mask == encode_mask {
        pipeline.step_image_with_sparse_patchify_plan_wgpu_nodes_measured(
            image,
            &patchify_plan,
            request,
            pipeline.config().measurement,
        )
    } else {
        let write_batch_mask =
            SparseMaskBatch::uniform(write_mask.clone(), pipeline.batch(), pipeline.device())?;
        pipeline.step_image_with_sparse_patchify_plan_wgpu_nodes_measured_with_write_mask(
            image,
            &patchify_plan,
            write_batch_mask,
            request,
            pipeline.config().measurement,
        )
    }
}

#[cfg(not(feature = "sparse-patchify-wgpu"))]
fn run_feature_frame_pipeline_sparse_patchify(
    _pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    _image: Tensor<JepaBevyBackend, 4>,
    _write_mask: &SparseTokenMask,
    _encode_mask: &SparseTokenMask,
    _request: FeatureFrameRequest,
) -> Result<burn_jepa::MeasuredFeatureFrameBatch<JepaBevyBackend>> {
    bail!("Bevy sparse patchify requires building bevy_jepa with --features sparse-patchify-wgpu")
}

#[allow(clippy::too_many_arguments)]
fn run_stage_step_metrics(
    config: &BevyJepaConfig,
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    write_mask: &SparseTokenMask,
    encode_mask: &SparseTokenMask,
    id: FrameId,
    grid: TokenGridShape,
    patch_size: usize,
    frame_source: BevyJepaFrameSource,
    camera_frame_received: bool,
) -> Result<ProcessedFrame> {
    run_stage_step_metrics_with_request(
        config,
        pipeline,
        image,
        write_mask,
        encode_mask,
        id,
        grid,
        patch_size,
        frame_source,
        camera_frame_received,
        FeatureFrameRequest::full_pca(),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_stage_step_metrics_with_request(
    config: &BevyJepaConfig,
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    write_mask: &SparseTokenMask,
    encode_mask: &SparseTokenMask,
    id: FrameId,
    grid: TokenGridShape,
    patch_size: usize,
    frame_source: BevyJepaFrameSource,
    camera_frame_received: bool,
    request: FeatureFrameRequest,
) -> Result<ProcessedFrame> {
    let stage = run_stage_step_with_config_and_request(
        config,
        pipeline,
        image,
        write_mask,
        encode_mask,
        request,
    )?;
    let frame = MetricFrameContext {
        frame_index: id.sequence,
        encoder_source: config.encoder_source,
        frame_source,
        camera_frame_received,
        mask_source: config.mask_source,
        display_transfer: config.display_transfer,
        context_tokens: write_mask.len(),
        dense_tokens: write_mask.dense_len(),
        grid,
        patch_size,
    };
    let metrics = bevy_metrics_from_stage(frame, stage.metrics, 0, stage.wall_us);
    Ok(ProcessedFrame { metrics })
}

#[allow(clippy::too_many_arguments)]
fn run_pipeline_step(
    config: &BevyJepaConfig,
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    write_mask: &SparseTokenMask,
    encode_mask: &SparseTokenMask,
    id: FrameId,
    grid: TokenGridShape,
    patch_size: usize,
    frame_source: BevyJepaFrameSource,
    camera_frame_received: bool,
    request: FeatureFrameRequest,
) -> Result<ProcessedFrame> {
    let stage = run_stage_step_with_config_and_request(
        config,
        pipeline,
        image.clone(),
        write_mask,
        encode_mask,
        request,
    )?;
    let metrics = stage.metrics;
    let output = stage.output;
    let image_size = pipeline.image_size();
    let display_start = viewer_now();
    let low_res_pca = low_res_pca_or_features(output.low_res)?;
    let high_res_pca = high_res_pca_or_low_res(output.high_res, low_res_pca.clone())?;
    let input_rgba = nchw_to_rgba_tensor(image)?;
    let mask_rgba = sparse_mask_to_rgba_tensor::<JepaBevyBackend>(
        write_mask,
        pipeline.grid(),
        image_size,
        &input_rgba.device(),
    )?;
    let low_res_rgba = nchw_to_rgba_tensor(resize_nchw(low_res_pca, image_size))?;
    let high_res_rgba = nchw_to_rgba_tensor(resize_nchw(high_res_pca, image_size))?;
    let display_device = input_rgba.device();
    match config.display_transfer {
        BevyJepaDisplayTransfer::Gpu => {
            let _gpu_display_tensors = (input_rgba, mask_rgba, low_res_rgba, high_res_rgba);
            sync_bevy_backend(&display_device)?;
        }
        BevyJepaDisplayTransfer::Cpu => {
            let _host_display_buffers = (
                tensor_rgba_to_host(input_rgba)?,
                tensor_rgba_to_host(mask_rgba)?,
                tensor_rgba_to_host(low_res_rgba)?,
                tensor_rgba_to_host(high_res_rgba)?,
            );
        }
    }
    if config.display_transfer == BevyJepaDisplayTransfer::Cpu && sync_measurements_enabled(config)
    {
        sync_bevy_backend(&display_device)?;
    }
    let display_tensor_us = viewer_elapsed_us(display_start);
    let viewer_total_us = stage.wall_us.saturating_add(display_tensor_us);
    let frame = MetricFrameContext {
        frame_index: id.sequence,
        encoder_source: config.encoder_source,
        frame_source,
        camera_frame_received,
        mask_source: config.mask_source,
        context_tokens: write_mask.len(),
        dense_tokens: write_mask.dense_len(),
        display_transfer: config.display_transfer,
        grid,
        patch_size,
    };
    Ok(ProcessedFrame {
        metrics: bevy_metrics_from_stage(frame, metrics, display_tensor_us, viewer_total_us),
    })
}

#[allow(clippy::too_many_arguments)]
fn run_stage_pipeline_step(
    config: &BevyJepaConfig,
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    write_mask: &SparseTokenMask,
    encode_mask: &SparseTokenMask,
    id: FrameId,
    grid: TokenGridShape,
    patch_size: usize,
    frame_source: BevyJepaFrameSource,
    camera_frame_received: bool,
    request: FeatureFrameRequest,
) -> Result<StageProcessedFrame> {
    let stage = run_stage_step_with_config_and_request(
        config,
        pipeline,
        image.clone(),
        write_mask,
        encode_mask,
        request,
    )?;
    let metrics = stage.metrics;
    let output = stage.output;
    let image_size = pipeline.image_size();
    let display_start = viewer_now();
    let high_res_input =
        high_res_scheduled_for_frame(config, id.sequence).then(|| HighResFrameInput {
            signature: RuntimePipelineSignature::new(config, config.pipeline_image_size()),
            id,
            image: image.clone(),
            low_res_features: output.low_res.features.clone(),
            pca: pipeline.pca().clone(),
            display_transfer: config.display_transfer,
            sync_measurements: sync_measurements_enabled(config),
        });
    let low_res_pca = low_res_pca_or_features(output.low_res)?;
    let high_res_pca = output.high_res.and_then(|high_res| high_res.pca_display);
    let mask_rgba = sparse_mask_to_rgba_tensor::<JepaBevyBackend>(
        write_mask,
        pipeline.grid(),
        image_size,
        &image.device(),
    )?;
    let low_res_rgba = nchw_to_rgba_tensor(resize_nchw(low_res_pca, image_size))?;
    let high_res_updated = high_res_pca.is_some();
    let high_res_rgba = high_res_pca
        .map(|pca| nchw_to_rgba_tensor(resize_nchw(pca, image_size)))
        .transpose()?;
    if sync_measurements_enabled(config) {
        sync_bevy_backend(&image.device())?;
    }
    let display_tensor_us = viewer_elapsed_us(display_start);
    let viewer_total_us = stage.wall_us.saturating_add(display_tensor_us);
    let frame = MetricFrameContext {
        frame_index: id.sequence,
        encoder_source: config.encoder_source,
        frame_source,
        camera_frame_received,
        mask_source: config.mask_source,
        display_transfer: config.display_transfer,
        context_tokens: write_mask.len(),
        dense_tokens: write_mask.dense_len(),
        grid,
        patch_size,
    };
    let metrics = bevy_metrics_from_stage(frame, metrics, display_tensor_us, viewer_total_us);
    let panels = match config.display_transfer {
        BevyJepaDisplayTransfer::Gpu => StagePanelData::Tensor {
            width: image_size[1] as u32,
            height: image_size[0] as u32,
            mask_rgba,
            low_res_rgba,
            high_res_rgba,
        },
        BevyJepaDisplayTransfer::Cpu => StagePanelData::Host {
            width: image_size[1] as u32,
            height: image_size[0] as u32,
            mask_rgba: tensor_rgba_to_host(mask_rgba)?,
            low_res_rgba: tensor_rgba_to_host(low_res_rgba)?,
            high_res_rgba: high_res_rgba.map(tensor_rgba_to_host).transpose()?,
        },
    };
    Ok(StageProcessedFrame {
        metrics,
        panels,
        high_res_updated,
        high_res_input,
    })
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn run_high_res_anyup_step(
    runtime: &mut AnyUpHighResRuntime,
    input: HighResFrameInput,
) -> Result<HighResProcessedFrame> {
    let total_start = viewer_now();
    let device = input.image.device();

    let context_start = viewer_now();
    let context = runtime.anyup.prepare_image_context_with_grid(
        input.image,
        &runtime.anyup_image_grid,
        Some(runtime.image_size),
        [runtime.grid.height, runtime.grid.width],
    );
    if input.sync_measurements {
        sync_bevy_backend(&device)?;
    }
    let anyup_context_us = viewer_elapsed_us(context_start);

    let pca_start = viewer_now();
    let pca_values = input.pca.project_nchw(input.low_res_features.clone())?;
    if input.sync_measurements {
        sync_bevy_backend(&device)?;
    }
    let low_res_pca_us = viewer_elapsed_us(pca_start);

    let decode_start = viewer_now();
    let pca_values = runtime.anyup.upsample_values_with_context(
        &context,
        input.low_res_features,
        pca_values,
        Some(runtime.q_chunk_size),
    );
    if input.sync_measurements {
        sync_bevy_backend(&device)?;
    }
    let anyup_decode_us = viewer_elapsed_us(decode_start);

    let display_pca_start = viewer_now();
    let high_res_pca = input.pca.display_nchw(pca_values)?;
    if input.sync_measurements {
        sync_bevy_backend(&device)?;
    }
    let high_res_pca_us = low_res_pca_us.saturating_add(viewer_elapsed_us(display_pca_start));

    let display_start = viewer_now();
    let high_res_rgba = nchw_to_rgba_tensor(resize_nchw(high_res_pca, runtime.image_size))?;
    if input.sync_measurements {
        sync_bevy_backend(&device)?;
    }
    let display_tensor_us = viewer_elapsed_us(display_start);
    let panels = match input.display_transfer {
        BevyJepaDisplayTransfer::Gpu => HighResPanelData::Tensor {
            width: runtime.image_size[1] as u32,
            height: runtime.image_size[0] as u32,
            high_res_rgba,
        },
        BevyJepaDisplayTransfer::Cpu => HighResPanelData::Host {
            width: runtime.image_size[1] as u32,
            height: runtime.image_size[0] as u32,
            high_res_rgba: tensor_rgba_to_host(high_res_rgba)?,
        },
    };

    Ok(HighResProcessedFrame {
        id: input.id,
        panels,
        anyup_context_us,
        anyup_decode_us,
        high_res_pca_us,
        display_tensor_us,
        total_us: viewer_elapsed_us(total_start),
    })
}

fn source_input_panel_from_input_source(
    config: &BevyJepaConfig,
    source: &SourceInputNodeOutput,
    image_size: [usize; 2],
) -> Result<InputPanelData> {
    if let Some(rgba) = &source.rgba {
        let mut input_rgba = rgba.as_raw().clone();
        let expected = image_size[0] * image_size[1] * 4;
        if input_rgba.len() != expected {
            input_rgba.resize(expected, 0);
        }
        return Ok(InputPanelData::Host {
            width: image_size[1] as u32,
            height: image_size[0] as u32,
            input_rgba,
        });
    }
    let Some(image) = source.image.as_ref() else {
        bail!("source frame has neither RGBA preview data nor a tensor preview");
    };
    source_input_panel(config, image.clone(), image_size)
}

fn source_input_panel(
    config: &BevyJepaConfig,
    image: Tensor<JepaBevyBackend, 4>,
    image_size: [usize; 2],
) -> Result<InputPanelData> {
    let input_rgba = nchw_to_rgba_tensor(image)?;
    Ok(match config.display_transfer {
        BevyJepaDisplayTransfer::Gpu => InputPanelData::Tensor {
            width: image_size[1] as u32,
            height: image_size[0] as u32,
            input_rgba,
        },
        BevyJepaDisplayTransfer::Cpu => InputPanelData::Host {
            width: image_size[1] as u32,
            height: image_size[0] as u32,
            input_rgba: tensor_rgba_to_host(input_rgba)?,
        },
    })
}

fn low_res_pca_or_features(
    low_res: LowResFrameArtifacts<JepaBevyBackend>,
) -> Result<Tensor<JepaBevyBackend, 4>> {
    if let Some(pca) = low_res.pca_display {
        return Ok(pca);
    }
    let [batch, channels, height, width] = low_res.features.shape().dims::<4>();
    anyhow::ensure!(batch == 1, "low-res display expects batch size 1");
    anyhow::ensure!(
        channels >= 3,
        "low-res feature fallback requires at least three channels"
    );
    anyhow::ensure!(
        height > 0 && width > 0,
        "low-res feature fallback requires a nonempty spatial grid"
    );
    Ok(low_res.features.slice_dim(1, 0..3))
}

fn high_res_pca_or_low_res(
    high_res: Option<HighResFrameArtifacts<JepaBevyBackend>>,
    low_res_pca: Tensor<JepaBevyBackend, 4>,
) -> Result<Tensor<JepaBevyBackend, 4>> {
    Ok(high_res
        .and_then(|high_res| high_res.pca_display)
        .unwrap_or(low_res_pca))
}

fn poll_source_frame_node(
    config: &BevyJepaConfig,
    runtime: &mut JepaRuntime,
    frame_index: u64,
    image_size: usize,
    device: &JepaBevyDevice,
) -> Result<Option<SourceFrameNodeOutput>> {
    match config.source {
        BevyJepaFrameSource::SyntheticLocalMotion => Ok(Some(SourceFrameNodeOutput {
            image: synthetic_image_tensor(frame_index, image_size, device),
            rgba: None,
            source: BevyJepaFrameSource::SyntheticLocalMotion,
            camera_frame_received: false,
        })),
        BevyJepaFrameSource::StaticImage => Ok(Some(SourceFrameNodeOutput {
            image: static_image_tensor(config, runtime, image_size, device)?,
            rgba: runtime
                .static_frame
                .as_ref()
                .map(|cached| cached.rgba.clone()),
            source: BevyJepaFrameSource::StaticImage,
            camera_frame_received: false,
        })),
        BevyJepaFrameSource::Camera => {
            if let Some(frame) = receive_frame() {
                return source_frame_from_rgba(
                    frame,
                    image_size,
                    device,
                    BevyJepaFrameSource::Camera,
                    true,
                )
                .map(Some);
            }

            Ok(None)
        }
    }
}

fn poll_source_input_node(
    config: &BevyJepaConfig,
    runtime: &mut JepaRuntime,
    frame_index: u64,
    image_size: usize,
    device: &JepaBevyDevice,
) -> Result<Option<SourceInputNodeOutput>> {
    match config.source {
        BevyJepaFrameSource::SyntheticLocalMotion => Ok(Some(SourceInputNodeOutput {
            image: Some(synthetic_image_tensor(frame_index, image_size, device)),
            rgba: None,
            source: BevyJepaFrameSource::SyntheticLocalMotion,
            camera_frame_received: false,
        })),
        BevyJepaFrameSource::StaticImage => {
            let image = static_image_tensor(config, runtime, image_size, device)?;
            Ok(Some(SourceInputNodeOutput {
                image: Some(image),
                rgba: runtime
                    .static_frame
                    .as_ref()
                    .map(|cached| cached.rgba.clone()),
                source: BevyJepaFrameSource::StaticImage,
                camera_frame_received: false,
            }))
        }
        BevyJepaFrameSource::Camera => {
            let Some(frame) = receive_frame() else {
                return Ok(None);
            };
            Ok(Some(SourceInputNodeOutput {
                image: None,
                rgba: Some(resize_source_rgba(frame, image_size)),
                source: BevyJepaFrameSource::Camera,
                camera_frame_received: true,
            }))
        }
    }
}

fn source_stage_image(
    image: Option<&Tensor<JepaBevyBackend, 4>>,
    rgba: Option<&RgbaImage>,
    image_size: usize,
    device: &JepaBevyDevice,
) -> Result<Tensor<JepaBevyBackend, 4>> {
    if let Some(image) = image {
        return Ok(image.clone());
    }
    let Some(rgba) = rgba else {
        bail!("source frame has neither a stage tensor nor an RGBA frame");
    };
    rgba_image_to_tensor(rgba.clone(), image_size, device)
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn pending_stage_image(
    pending: &PendingStageFrame,
    image_size: usize,
    device: &JepaBevyDevice,
) -> Result<Tensor<JepaBevyBackend, 4>> {
    source_stage_image(
        pending.image.as_ref(),
        pending.rgba.as_ref(),
        image_size,
        device,
    )
}

fn receive_frame() -> Option<RgbaImage> {
    platform::camera::receive_image()
}

fn static_image_tensor(
    config: &BevyJepaConfig,
    runtime: &mut JepaRuntime,
    image_size: usize,
    device: &JepaBevyDevice,
) -> Result<Tensor<JepaBevyBackend, 4>> {
    let path = config.image_path.clone();
    let needs_reload = runtime.static_frame.as_ref().is_none_or(|cached| {
        cached.image_size != image_size || cached.path.as_ref() != path.as_ref()
    });
    if needs_reload {
        let frame = resize_source_rgba(load_static_rgba(path.as_ref(), image_size)?, image_size);
        let image = rgba_image_to_tensor(frame.clone(), image_size, device)?;
        runtime.static_frame = Some(CachedStaticFrame {
            path,
            image_size,
            rgba: frame,
            image,
        });
    }
    Ok(runtime
        .static_frame
        .as_ref()
        .expect("static frame initialized")
        .image
        .clone())
}

fn source_frame_from_rgba(
    frame: RgbaImage,
    image_size: usize,
    device: &JepaBevyDevice,
    source: BevyJepaFrameSource,
    camera_frame_received: bool,
) -> Result<SourceFrameNodeOutput> {
    let rgba = resize_source_rgba(frame, image_size);
    Ok(SourceFrameNodeOutput {
        image: rgba_image_to_tensor(rgba.clone(), image_size, device)?,
        rgba: Some(rgba),
        source,
        camera_frame_received,
    })
}

fn load_static_rgba(path: Option<&PathBuf>, image_size: usize) -> Result<RgbaImage> {
    if let Some(path) = path {
        return Ok(ImageReader::open(path)
            .with_context(|| format!("open static JEPA frame `{}`", path.display()))?
            .decode()
            .with_context(|| format!("decode static JEPA frame `{}`", path.display()))?
            .to_rgba8());
    }
    Ok(generated_static_rgba(image_size as u32, image_size as u32))
}

fn generated_static_rgba(width: u32, height: u32) -> RgbaImage {
    let width = width.max(1);
    let height = height.max(1);
    let mut rgba = vec![0; width as usize * height as usize * 4];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let offset = (y * width as usize + x) * 4;
            let checker = ((x >> 4) + (y >> 4)) & 1;
            rgba[offset] = ((x as f32 / width.max(1) as f32) * 220.0 + checker as f32 * 24.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            rgba[offset + 1] = ((y as f32 / height.max(1) as f32) * 220.0 + 20.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            rgba[offset + 2] = if checker == 0 { 92 } else { 230 };
            rgba[offset + 3] = 255;
        }
    }
    RgbaImage::from_raw(width, height, rgba).unwrap_or_else(|| RgbaImage::new(width, height))
}

fn rgba_image_to_tensor(
    frame: RgbaImage,
    image_size: usize,
    device: &JepaBevyDevice,
) -> Result<Tensor<JepaBevyBackend, 4>> {
    let frame = resize_source_rgba(frame, image_size);
    let height = frame.height() as usize;
    let width = frame.width() as usize;
    let raw = frame.as_raw();
    let mut values = vec![0.0; 3 * height * width];
    for y in 0..height {
        for x in 0..width {
            let pixel = (y * width + x) * 4;
            let index = y * width + x;
            values[index] = normalize_model_rgb_channel(raw[pixel] as f32 / 255.0, 0);
            values[height * width + index] =
                normalize_model_rgb_channel(raw[pixel + 1] as f32 / 255.0, 1);
            values[2 * height * width + index] =
                normalize_model_rgb_channel(raw[pixel + 2] as f32 / 255.0, 2);
        }
    }
    Ok(Tensor::<JepaBevyBackend, 4>::from_data(
        TensorData::new(values, [1, 3, height, width]),
        device,
    ))
}

fn resize_source_rgba(frame: RgbaImage, image_size: usize) -> RgbaImage {
    let size = image_size.max(1) as u32;
    if frame.width() == size && frame.height() == size {
        frame
    } else {
        let crop = center_square_crop_rgba(&frame);
        if crop.width() == size && crop.height() == size {
            crop
        } else {
            image::imageops::resize(&crop, size, size, FilterType::Triangle)
        }
    }
}

fn center_square_crop_rgba(frame: &RgbaImage) -> RgbaImage {
    let width = frame.width().max(1);
    let height = frame.height().max(1);
    let side = width.min(height);
    let x = width.saturating_sub(side) / 2;
    let y = height.saturating_sub(side) / 2;
    image::imageops::crop_imm(frame, x, y, side, side).to_image()
}

fn synthetic_image_tensor(
    frame_index: u64,
    image_size: usize,
    device: &JepaBevyDevice,
) -> Tensor<JepaBevyBackend, 4> {
    let height = image_size;
    let width = image_size;
    let phase = frame_index as f32 * 0.035;
    let cx = (phase.sin() * 0.32 + 0.5) * width as f32;
    let cy = (phase.cos() * 0.27 + 0.5) * height as f32;
    let sigma = (image_size as f32 * 0.13).max(1.0);
    let mut values = vec![0.0; 3 * height * width];
    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let blob = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
            let base =
                (x as f32 / width.max(1) as f32 * 0.45) + (y as f32 / height.max(1) as f32 * 0.25);
            let index = y * width + x;
            values[index] = normalize_model_rgb_channel((base + blob * 0.6).clamp(0.0, 1.0), 0);
            values[height * width + index] =
                normalize_model_rgb_channel(((1.0 - base) * 0.55 + blob * 0.35).clamp(0.0, 1.0), 1);
            values[2 * height * width + index] = normalize_model_rgb_channel(
                ((phase.sin() * 0.15 + 0.25) + blob * 0.5).clamp(0.0, 1.0),
                2,
            );
        }
    }
    Tensor::<JepaBevyBackend, 4>::from_data(TensorData::new(values, [1, 3, height, width]), device)
}

fn normalize_model_rgb_channel(value: f32, channel: usize) -> f32 {
    (value - VJEPA_IMAGE_MEAN[channel]) / VJEPA_IMAGE_STD[channel]
}

fn nchw_to_rgba_tensor(tensor: Tensor<JepaBevyBackend, 4>) -> Result<Tensor<JepaBevyBackend, 3>> {
    let [batch, channels, height, width] = tensor.shape().dims::<4>();
    anyhow::ensure!(batch == 1, "display tensor must have batch size 1");
    anyhow::ensure!(
        channels >= 3,
        "display tensor must have at least 3 channels"
    );
    let device = tensor.device();
    let rgb = tensor
        .slice_dim(1, 0..3)
        .permute([0, 2, 3, 1])
        .reshape([height, width, 3]);
    let alpha = Tensor::<JepaBevyBackend, 3>::ones([height, width, 1], &device);
    Ok(Tensor::cat(vec![rgb, alpha], 2))
}

fn resize_nchw(
    tensor: Tensor<JepaBevyBackend, 4>,
    output_size: [usize; 2],
) -> Tensor<JepaBevyBackend, 4> {
    let [_, _, height, width] = tensor.shape().dims::<4>();
    if [height, width] == output_size {
        tensor
    } else {
        interpolate(
            tensor,
            output_size,
            InterpolateOptions::new(InterpolateMode::Nearest),
        )
    }
}

fn sparse_mask_to_rgba_tensor<B: Backend>(
    mask: &SparseTokenMask,
    grid: TokenGridShape,
    image_size: [usize; 2],
    device: &B::Device,
) -> Result<Tensor<B, 3>> {
    let height = image_size[0].max(1);
    let width = image_size[1].max(1);
    let patch_h = (height / grid.height.max(1)).max(1);
    let patch_w = (width / grid.width.max(1)).max(1);
    let mut rgba = vec![0.06_f32; height * width * 4];
    for pixel in rgba.chunks_exact_mut(4) {
        pixel[3] = 1.0;
    }
    for &token in mask.indices() {
        let row = (token % grid.tokens_per_frame()) / grid.width;
        let col = token % grid.width;
        for y in row * patch_h..((row + 1) * patch_h).min(height) {
            for x in col * patch_w..((col + 1) * patch_w).min(width) {
                let offset = (y * width + x) * 4;
                rgba[offset] = 0.16;
                rgba[offset + 1] = 0.86;
                rgba[offset + 2] = 0.58;
            }
        }
    }
    Ok(Tensor::<B, 3>::from_data(
        TensorData::new(rgba, [height, width, 4]),
        device,
    ))
}

fn tensor_rgba_to_host(tensor: Tensor<JepaBevyBackend, 3>) -> Result<Vec<u8>> {
    let values = tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("read display tensor: {err}"))?;
    Ok(values
        .into_iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect())
}

fn update_metrics_overlay(
    config: Res<BevyJepaConfig>,
    metrics: Res<BevyJepaMetrics>,
    runtime: Res<JepaRuntime>,
    mut history: ResMut<MetricsRollingState>,
    mut metric_nodes: ParamSet<(
        Query<(&MetricsOverlayRoot, &mut Node)>,
        Query<(&MetricGraphBar, &mut Node, &mut BackgroundColor)>,
    )>,
    mut values: Query<(&MetricValueText, &mut Text)>,
) {
    if !metrics.is_changed() && !runtime.is_changed() && !config.is_changed() {
        return;
    }
    publish_wasm_metrics(&metrics);
    observe_metrics_history(&mut history, &metrics);
    for (root, mut node) in &mut metric_nodes.p0() {
        node.display = if config.show_metrics {
            root.visible_display
        } else {
            Display::None
        };
    }
    if !config.show_metrics {
        return;
    }
    if let Some(error) = runtime.last_error.as_ref().or(metrics.last_error.as_ref()) {
        publish_wasm_error(error);
    }
    for (value, mut text) in &mut values {
        **text = metric_value_text(&config, &metrics, &runtime, &history, value.kind);
    }
    for (bar, mut node, mut color) in &mut metric_nodes.p1() {
        let normalized = metric_graph_normalized(&history, bar.index, config.camera_fps as f64);
        node.height = Val::Px((METRICS_GRAPH_HEIGHT_PX * normalized).max(1.0));
        *color = BackgroundColor(metric_graph_color(normalized));
    }
}

fn observe_metrics_history(history: &mut MetricsRollingState, metrics: &BevyJepaMetrics) {
    if !metrics.frame_ready {
        return;
    }
    let sample_key = if metrics.completed_frames > 0 {
        metrics.completed_frames
    } else {
        metrics.frame_index
    };
    if history.last_sample_key == Some(sample_key) {
        return;
    }
    history.last_sample_key = Some(sample_key);
    let fps = if metrics.low_res_fps > 0.0 {
        metrics.low_res_fps
    } else {
        metrics.viewer_fps()
    };
    history.samples.push_back(MetricRollingSample {
        fps: fps.max(0.0),
        viewer_us: metrics.viewer_total_us,
    });
    while history.samples.len() > METRICS_GRAPH_BARS {
        history.samples.pop_front();
    }
}

fn metric_value_text(
    config: &BevyJepaConfig,
    metrics: &BevyJepaMetrics,
    runtime: &JepaRuntime,
    history: &MetricsRollingState,
    kind: MetricValueKind,
) -> String {
    match kind {
        MetricValueKind::Status => {
            format!("Status   {}", metrics_source_status(config, metrics))
        }
        MetricValueKind::Model => {
            format!(
                "Model    {} / {}",
                config.model_profile, metrics.encoder_source
            )
        }
        MetricValueKind::Grid => format!(
            "Grid     {}x{} @ {}px  p{}",
            metrics.grid_height,
            metrics.grid_width,
            config.pipeline_image_size(),
            metrics.patch_size,
        ),
        MetricValueKind::Tokens => format!(
            "Write    {}  {}",
            format_metric_token_ratio(metric_write_tokens(metrics), metrics.dense_tokens),
            metric_write_label(metrics),
        ),
        MetricValueKind::EncodeTokens => format!(
            "Encode   {}  {}",
            format_metric_token_ratio(metric_encode_tokens(metrics), metrics.dense_tokens),
            metric_encode_label(metrics),
        ),
        MetricValueKind::Queue => format!(
            "Queue    active {}  overwritten {}  stale {}",
            metrics.in_flight_frames, metrics.queue_overwritten_frames, metrics.stale_completions
        ),
        MetricValueKind::Error => runtime
            .last_error
            .as_ref()
            .or(metrics.last_error.as_ref())
            .map(|error| format!("Health   {}", truncate_metric_text(error, 42)))
            .unwrap_or_else(|| "Health   ok".to_string()),
        MetricValueKind::FpsInput => format!("Input    {}", format_metric_fps(metrics.input_fps)),
        MetricValueKind::FpsLow => format!("Low-res  {}", format_metric_fps(metrics.low_res_fps)),
        MetricValueKind::FpsHigh => format!("AnyUp    {}", format_metric_fps(metrics.high_res_fps)),
        MetricValueKind::FpsRolling => format!(
            "Rolling  {}  {}",
            format_metric_fps(rolling_fps(history)),
            format_metric_ms(rolling_viewer_us(history))
        ),
        MetricValueKind::StageInputSource => format!(
            "Source   {} ({})",
            metrics.frame_source,
            metrics_source_status(config, metrics)
        ),
        MetricValueKind::StageInputQueue => format!(
            "Queue    active {}  ovw {}  stale {}",
            metrics.in_flight_frames, metrics.queue_overwritten_frames, metrics.stale_completions
        ),
        MetricValueKind::StageMaskPolicy => {
            format!(
                "Policy   {}  refresh {}",
                metrics.mask_source,
                on_off(config.patch_diff_refresh.enabled)
            )
        }
        MetricValueKind::StageMaskWrite => format!(
            "Write    {}  {}",
            format_metric_token_ratio(metric_write_tokens(metrics), metrics.dense_tokens),
            metric_write_label(metrics),
        ),
        MetricValueKind::StageMaskEncode => format!(
            "Encode   {}  {}",
            format_metric_token_ratio(metric_encode_tokens(metrics), metrics.dense_tokens),
            metric_encode_label(metrics),
        ),
        MetricValueKind::StageEncodeLatency => {
            format!(
                "JEPA     {}  total {}",
                format_metric_ms(metrics.encode_us),
                format_metric_ms(metrics.total_us)
            )
        }
        MetricValueKind::StageEncodePath => format!(
            "Route    {} / {}",
            metrics.encode_path, config.sparse_encode_mode
        ),
        MetricValueKind::StageTttStability => format_ttt_stability_line(config, metrics),
        MetricValueKind::StageCacheLatency => {
            format!("Cache    {}", format_metric_ms(metrics.cache_update_us))
        }
        MetricValueKind::StageTokenViewLatency => {
            format!("Grid view {}", format_metric_ms(metrics.token_view_us))
        }
        MetricValueKind::StageLowResPcaLatency => {
            format!("Project  {}", format_metric_ms(metrics.low_res_pca_us))
        }
        MetricValueKind::StagePcaBasis => format_pca_basis_line(config, metrics),
        MetricValueKind::StageAnyUpLatency => format!(
            "Decode   ctx {}  up {}",
            format_metric_ms(metrics.anyup_context_us),
            format_metric_ms(metrics.anyup_decode_us)
        ),
        MetricValueKind::StageDisplayLatency => {
            format!("Display  {}", format_metric_ms(metrics.display_tensor_us))
        }
    }
}

fn metric_write_tokens(metrics: &BevyJepaMetrics) -> usize {
    let stage = &metrics.stage_metrics;
    if stage.valid_write_tokens > 0 {
        stage.valid_write_tokens
    } else if stage.write_width > 0 {
        stage.write_width
    } else {
        metrics.context_tokens
    }
}

fn metric_encode_tokens(metrics: &BevyJepaMetrics) -> usize {
    let stage = &metrics.stage_metrics;
    if stage.valid_encode_tokens > 0 {
        stage.valid_encode_tokens
    } else if stage.encode_width > 0 {
        stage.encode_width
    } else {
        metric_write_tokens(metrics)
    }
}

fn metric_write_label(metrics: &BevyJepaMetrics) -> &'static str {
    if metric_write_tokens(metrics) >= metrics.dense_tokens.max(1) {
        "dense"
    } else {
        "cache mask"
    }
}

fn metric_encode_label(metrics: &BevyJepaMetrics) -> &'static str {
    let write = metric_write_tokens(metrics);
    let encode = metric_encode_tokens(metrics);
    if encode >= metrics.dense_tokens.max(1) {
        "dense"
    } else if encode > write {
        "bucketed"
    } else {
        "exact"
    }
}

fn format_metric_token_ratio(tokens: usize, dense_tokens: usize) -> String {
    let dense_tokens = dense_tokens.max(1);
    format!(
        "{tokens}/{dense_tokens} {}",
        format_metric_percent_compact(tokens as f64 / dense_tokens as f64)
    )
}

fn format_ttt_stability_line(config: &BevyJepaConfig, metrics: &BevyJepaMetrics) -> String {
    if metrics.encoder_source != BevyJepaEncoderSource::TrainedTtt {
        return "TTT      n/a for base".to_string();
    }
    if !config.ttt_runtime.enabled {
        return "TTT      memory off".to_string();
    }
    if let (Some(collapse), Some(cosine)) = (
        metrics.ttt_collapse_score,
        metrics.ttt_mean_pairwise_token_cosine,
    ) {
        return format!(
            "TTT      col {collapse:.2}  cos {cosine:.2}  g{}",
            metrics.ttt_collapse_guard_triggers
        );
    }
    if config.ttt_runtime.metrics_interval_frames == 0 {
        return format!(
            "TTT      mem on  diag off  g{}",
            metrics.ttt_collapse_guard_triggers
        );
    }
    format!(
        "TTT      waiting {}f  g{}",
        config.ttt_runtime.metrics_interval_frames, metrics.ttt_collapse_guard_triggers
    )
}

fn format_pca_basis_line(config: &BevyJepaConfig, metrics: &BevyJepaMetrics) -> String {
    if metrics.pca_sample_window_frames == 0 {
        return "Basis   off".to_string();
    }
    let status = if metrics.pca_update_applied {
        "fit"
    } else {
        "cached"
    };
    format!(
        "Basis   {status} {}  {}/{} @{}f",
        format_metric_ms(metrics.pca_update_us),
        metrics.pca_sample_frames,
        metrics.pca_sample_window_frames,
        config.pca_update_every.max(1)
    )
}

fn rolling_fps(history: &MetricsRollingState) -> f64 {
    if history.samples.is_empty() {
        return 0.0;
    }
    history.samples.iter().map(|sample| sample.fps).sum::<f64>() / history.samples.len() as f64
}

fn rolling_viewer_us(history: &MetricsRollingState) -> u64 {
    if history.samples.is_empty() {
        return 0;
    }
    let sum = history
        .samples
        .iter()
        .map(|sample| sample.viewer_us as u128)
        .sum::<u128>();
    micros_u64(sum / history.samples.len() as u128)
}

fn metric_graph_normalized(
    history: &MetricsRollingState,
    bar_index: usize,
    target_fps: f64,
) -> f32 {
    let len = history.samples.len();
    if len == 0 || bar_index >= METRICS_GRAPH_BARS {
        return 0.0;
    }
    let leading_empty = METRICS_GRAPH_BARS.saturating_sub(len);
    if bar_index < leading_empty {
        return 0.0;
    }
    let sample_index = bar_index - leading_empty;
    let Some(sample) = history.samples.get(sample_index) else {
        return 0.0;
    };
    let target_fps = target_fps.max(1.0);
    (sample.fps / target_fps).clamp(0.0, 1.0) as f32
}

fn metric_graph_color(normalized: f32) -> Color {
    if normalized < 0.34 {
        Color::srgb(0.74, 0.20, 0.18)
    } else if normalized < 0.67 {
        Color::srgb(0.78, 0.55, 0.18)
    } else {
        Color::srgb(0.18, 0.58, 0.36)
    }
}

fn format_metric_fps(value: f64) -> String {
    format!("{value:>5.1} fps")
}

fn format_metric_ms(value_us: u64) -> String {
    format!("{:>6.2} ms", micros_to_ms(value_us))
}

fn format_metric_percent_compact(value: f64) -> String {
    format!("{:.1}%", value * 100.0)
}

fn truncate_metric_text(value: &str, max_chars: usize) -> String {
    let mut output = String::with_capacity(max_chars);
    for (index, ch) in value.chars().enumerate() {
        if index >= max_chars {
            output.push_str("...");
            return output;
        }
        output.push(ch);
    }
    output
}

#[cfg(target_arch = "wasm32")]
fn publish_wasm_error(error: &str) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let _ = js_sys::Reflect::set(
        window.as_ref(),
        &"__jepaLastError".into(),
        &wasm_bindgen::JsValue::from_str(error),
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn publish_wasm_error(_error: &str) {}

#[cfg(target_arch = "wasm32")]
fn publish_wasm_metrics(metrics: &BevyJepaMetrics) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let object = js_sys::Object::new();
    let fields = [
        ("frameIndex", metrics.frame_index as f64),
        ("inputFrameIndex", metrics.input_frame_index as f64),
        ("inputFramesSeen", metrics.input_frames_seen as f64),
        ("completedFrames", metrics.completed_frames as f64),
        ("contextTokens", metrics.context_tokens as f64),
        ("denseTokens", metrics.dense_tokens as f64),
        ("gridHeight", metrics.grid_height as f64),
        ("gridWidth", metrics.grid_width as f64),
        ("patchSize", metrics.patch_size as f64),
        ("encodeUs", metrics.encode_us as f64),
        ("cacheUpdateUs", metrics.cache_update_us as f64),
        ("lowResPcaUs", metrics.low_res_pca_us as f64),
        ("totalUs", metrics.total_us as f64),
        ("viewerTotalUs", metrics.viewer_total_us as f64),
        (
            "tttCollapseScore",
            metrics.ttt_collapse_score.unwrap_or(0.0),
        ),
        (
            "tttPairwiseCosine",
            metrics.ttt_mean_pairwise_token_cosine.unwrap_or(0.0),
        ),
        (
            "tttCollapseGuardTriggers",
            metrics.ttt_collapse_guard_triggers as f64,
        ),
    ];
    for (key, value) in fields {
        let _ = js_sys::Reflect::set(&object, &key.into(), &value.into());
    }
    let _ = js_sys::Reflect::set(
        &object,
        &"encoderSource".into(),
        &metrics.encoder_source.to_string().into(),
    );
    let _ = js_sys::Reflect::set(
        &object,
        &"maskSource".into(),
        &metrics.mask_source.to_string().into(),
    );
    let _ = js_sys::Reflect::set(&window, &"__jepaPipelineMetrics".into(), object.as_ref());
}

#[cfg(not(target_arch = "wasm32"))]
fn publish_wasm_metrics(_metrics: &BevyJepaMetrics) {}

fn control_button_interactions(
    mut interactions: Query<
        (&Interaction, &JepaControlButton),
        (Changed<Interaction>, With<Button>),
    >,
    mut controls: ResMut<JepaControlsState>,
    mut config: ResMut<BevyJepaConfig>,
    mut runtime: ResMut<JepaRuntime>,
) {
    for (interaction, button) in &mut interactions {
        if *interaction != Interaction::Pressed {
            continue;
        }
        if button.action == JepaControlAction::TogglePanel {
            controls.expanded = !controls.expanded;
            continue;
        }
        match apply_control_action(&mut config, button.action) {
            JepaControlReset::None => {}
            JepaControlReset::Visual => runtime.reset_visual_state(),
            JepaControlReset::Rebuild => runtime.rebuild_pipeline_state(),
        }
    }
}

fn control_slider_interactions(
    mouse: Res<ButtonInput<MouseButton>>,
    mut active_slider: ResMut<JepaActiveSlider>,
    sliders: Query<
        (
            Entity,
            &Interaction,
            &RelativeCursorPosition,
            &JepaControlSlider,
        ),
        With<Button>,
    >,
    mut config: ResMut<BevyJepaConfig>,
    mut runtime: ResMut<JepaRuntime>,
) {
    if !mouse.pressed(MouseButton::Left) {
        active_slider.entity = None;
        return;
    }

    if active_slider.entity.is_none() {
        for (entity, interaction, cursor, _) in &sliders {
            if *interaction == Interaction::Pressed && cursor.cursor_over() {
                active_slider.entity = Some(entity);
                break;
            }
        }
    }

    let Some(entity) = active_slider.entity else {
        return;
    };
    let Ok((_, _, cursor, slider)) = sliders.get(entity) else {
        active_slider.entity = None;
        return;
    };
    let Some(relative) = cursor.normalized else {
        return;
    };
    match apply_control_slider_value_if_changed(
        &mut config,
        slider.kind,
        slider_normalized_from_relative_x(relative.x),
    ) {
        JepaControlReset::None => {}
        JepaControlReset::Visual => runtime.reset_visual_state(),
        JepaControlReset::Rebuild => runtime.rebuild_pipeline_state(),
    }
}

fn apply_control_slider_value_if_changed(
    config: &mut BevyJepaConfig,
    kind: JepaControlSliderKind,
    normalized: f32,
) -> JepaControlReset {
    if (slider_normalized_value(config, kind) - normalized).abs() <= CONTROL_SLIDER_UPDATE_EPSILON {
        return JepaControlReset::None;
    }
    apply_control_slider_value(config, kind, normalized)
}

fn slider_normalized_from_relative_x(relative_x: f32) -> f32 {
    if !relative_x.is_finite() {
        return 0.0;
    }
    (relative_x + 0.5).clamp(0.0, 1.0)
}

fn update_controls_ui(
    config: Res<BevyJepaConfig>,
    controls: Res<JepaControlsState>,
    mut panels: Query<&mut Node, (With<ControlsPanel>, Without<JepaControlSliderFill>)>,
    mut summaries: Query<
        &mut Text,
        (
            With<ControlsSummaryText>,
            Without<ControlsHelpText>,
            Without<JepaControlButtonText>,
            Without<JepaControlSliderValueText>,
        ),
    >,
    mut help_texts: Query<
        &mut Text,
        (
            With<ControlsHelpText>,
            Without<ControlsSummaryText>,
            Without<JepaControlButtonText>,
            Without<JepaControlSliderValueText>,
        ),
    >,
    mut buttons: Query<(&JepaControlButton, &Interaction, &mut BackgroundColor), With<Button>>,
    hovered_help: Query<(&Interaction, &JepaControlHelp), With<Button>>,
    mut labels: Query<
        (&JepaControlButtonText, &mut Text),
        (
            Without<ControlsSummaryText>,
            Without<JepaControlSliderValueText>,
        ),
    >,
    mut slider_fills: Query<(&JepaControlSliderFill, &mut Node), Without<ControlsPanel>>,
    mut slider_values: Query<
        (&JepaControlSliderValueText, &mut Text),
        (Without<ControlsSummaryText>, Without<JepaControlButtonText>),
    >,
) {
    for mut panel in &mut panels {
        panel.display = if controls.expanded {
            Display::Flex
        } else {
            Display::None
        };
    }
    let summary = controls_summary(&config);
    for mut text in &mut summaries {
        **text = summary.clone();
    }
    let help = hovered_control_help(&hovered_help).unwrap_or_else(|| default_controls_help());
    for mut text in &mut help_texts {
        **text = help.to_string();
    }
    for (button, interaction, mut color) in &mut buttons {
        *color = BackgroundColor(control_button_color(
            button.action,
            control_button_active(&config, button.action, &controls),
            *interaction,
        ));
    }
    for (label, mut text) in &mut labels {
        **text = control_button_label(&config, label.action).to_string();
    }
    for (fill, mut node) in &mut slider_fills {
        node.width = Val::Percent(slider_normalized_value(&config, fill.kind) * 100.0);
    }
    for (value, mut text) in &mut slider_values {
        **text = slider_value_label(&config, value.kind);
    }
}

fn update_panel_layout(
    config: Res<BevyJepaConfig>,
    texture: Res<JepaPanelTextures>,
    mut nodes: ParamSet<(
        Query<&mut Node>,
        Query<&mut Node, With<HighResPanelElement>>,
        Query<&mut Node, With<MetricsStageGrid>>,
    )>,
) {
    let visible_count = visible_panel_count(&config);
    if let Some(root) = texture.root_entity
        && let Ok(mut node) = nodes.p0().get_mut(root)
    {
        node.grid_template_columns = RepeatedGridTrack::flex(visible_count as u16, 1.0);
    }
    let mut stage_grids = nodes.p2();
    for mut node in &mut stage_grids {
        node.grid_template_columns = RepeatedGridTrack::flex(visible_count as u16, 1.0);
    }
    let high_res_display = if high_res_panel_enabled(&config) {
        Display::Flex
    } else {
        Display::None
    };
    let mut high_res_nodes = nodes.p1();
    for mut node in &mut high_res_nodes {
        node.display = high_res_display;
    }
}

fn apply_control_action(
    config: &mut BevyJepaConfig,
    action: JepaControlAction,
) -> JepaControlReset {
    match action {
        JepaControlAction::TogglePanel => JepaControlReset::None,
        JepaControlAction::ModelTtt => {
            set_model_profile(config, BevyJepaModelPackageProfile::Vjepa21Ttt);
            JepaControlReset::Rebuild
        }
        JepaControlAction::ModelBase => {
            set_model_profile(config, BevyJepaModelPackageProfile::Vjepa21Base);
            JepaControlReset::Rebuild
        }
        JepaControlAction::PipelineSparse => {
            set_sparse_pipeline_preset(config);
            JepaControlReset::Rebuild
        }
        JepaControlAction::PipelineDense => {
            set_dense_pipeline_preset(config);
            JepaControlReset::Rebuild
        }
        JepaControlAction::Resolution256 => {
            config.image_size = 256;
            JepaControlReset::Rebuild
        }
        JepaControlAction::Resolution512 => {
            config.image_size = 512;
            JepaControlReset::Rebuild
        }
        JepaControlAction::AnyUpOff => {
            config.high_res_pca_every = 0;
            JepaControlReset::Rebuild
        }
        JepaControlAction::AnyUpEvery8 => {
            config.high_res_pca_every = 8;
            JepaControlReset::Rebuild
        }
        JepaControlAction::AnyUpEvery1 => {
            config.high_res_pca_every = 1;
            JepaControlReset::Rebuild
        }
        JepaControlAction::AnyUpEfficientLocal => {
            config.anyup_attention_mode = burn_jepa::AnyUpAttentionMode::EfficientLocal;
            JepaControlReset::Rebuild
        }
        JepaControlAction::AnyUpUpstreamMasked => {
            config.anyup_attention_mode = burn_jepa::AnyUpAttentionMode::UpstreamMasked;
            JepaControlReset::Rebuild
        }
        JepaControlAction::PatchRefresh => {
            config.patch_diff_refresh.enabled = !config.patch_diff_refresh.enabled;
            JepaControlReset::Visual
        }
        JepaControlAction::SubthresholdRefresh => {
            config.patch_diff_refresh.subthreshold_enabled =
                !config.patch_diff_refresh.subthreshold_enabled;
            JepaControlReset::Visual
        }
        JepaControlAction::AgeRefresh => {
            config.patch_diff_refresh.age_refresh_enabled =
                !config.patch_diff_refresh.age_refresh_enabled;
            JepaControlReset::Visual
        }
        JepaControlAction::BlueNoiseRefresh => {
            config.patch_diff_refresh.blue_noise_enabled =
                !config.patch_diff_refresh.blue_noise_enabled;
            JepaControlReset::Visual
        }
    }
}

fn apply_control_slider_value(
    config: &mut BevyJepaConfig,
    kind: JepaControlSliderKind,
    normalized: f32,
) -> JepaControlReset {
    match kind {
        JepaControlSliderKind::PatchDiffThreshold => {
            config.patch_diff_threshold = slider_lerp(0.0, 0.20, normalized);
            JepaControlReset::Visual
        }
        JepaControlSliderKind::ContextDensity => {
            config.context_density = slider_lerp(0.01, 1.0, normalized);
            config.min_context_density = config.min_context_density.min(config.context_density);
            JepaControlReset::Visual
        }
        JepaControlSliderKind::MinContextDensity => {
            config.min_context_density = slider_lerp(0.0, 1.0, normalized);
            config.context_density = config.context_density.max(config.min_context_density);
            JepaControlReset::Visual
        }
        JepaControlSliderKind::DenseFallbackDensity => {
            config.patch_diff_dense_fallback_density = slider_lerp(0.0, 1.0, normalized);
            JepaControlReset::Visual
        }
        JepaControlSliderKind::SubthresholdTrigger => {
            config.patch_diff_refresh.subthreshold_trigger = slider_lerp(0.1, 4.0, normalized);
            JepaControlReset::Visual
        }
        JepaControlSliderKind::AgeIntervalFrames => {
            config.patch_diff_refresh.age_refresh_interval_frames =
                slider_lerp(1.0, 300.0, normalized).round() as u64;
            JepaControlReset::Visual
        }
        JepaControlSliderKind::BlueNoiseDensity => {
            config.patch_diff_refresh.blue_noise_refresh_density =
                slider_lerp(0.0, 0.05, normalized);
            JepaControlReset::Visual
        }
    }
}

fn set_model_profile(config: &mut BevyJepaConfig, profile: BevyJepaModelPackageProfile) {
    config.model_profile = profile;
    config.model_base_url = burn_jepa::burn_jepa_model_profile_base_url(profile);
    config.model_manifest_path = None;
    config.ttt_model_path = None;
    match profile {
        BevyJepaModelPackageProfile::Vjepa21Base => {
            config.encoder_source = BevyJepaEncoderSource::BaseCheckpoint;
        }
        BevyJepaModelPackageProfile::Vjepa21Ttt => {
            config.encoder_source = BevyJepaEncoderSource::TrainedTtt;
        }
    }
}

fn set_sparse_pipeline_preset(config: &mut BevyJepaConfig) {
    let defaults = FeatureFrameViewerConfig::default();
    config.encode_path = BevyJepaEncodePath::Auto;
    config.context_density = defaults.context_density;
    config.min_context_density = defaults.min_context_density;
    config.bootstrap_context_density = defaults.bootstrap_context_density;
    config.patch_diff_threshold = defaults.patch_diff_threshold;
    config.patch_diff_dense_fallback_density = defaults.patch_diff_dense_fallback_density;
    config.sparse_encode_mode = FeatureFrameSparseEncodeMode::BucketedContext;
    config.patch_diff_refresh = defaults.patch_diff_refresh;
}

fn set_dense_pipeline_preset(config: &mut BevyJepaConfig) {
    config.encode_path = BevyJepaEncodePath::DensePatchEmbed;
    config.context_density = 1.0;
    config.min_context_density = 1.0;
    config.bootstrap_context_density = 1.0;
    config.patch_diff_threshold = 0.0;
    config.patch_diff_dense_fallback_density = DENSE_PIPELINE_FALLBACK_DENSITY;
    config.sparse_encode_mode = FeatureFrameSparseEncodeMode::Exact;
    config.patch_diff_refresh.enabled = false;
}

fn dense_pipeline_enabled(config: &BevyJepaConfig) -> bool {
    config.encode_path == BevyJepaEncodePath::DensePatchEmbed
        && config.context_density >= 1.0
        && config.min_context_density >= 1.0
        && config.bootstrap_context_density >= 1.0
        && config.patch_diff_threshold <= f32::EPSILON
        && config.patch_diff_dense_fallback_density <= f32::EPSILON
}

fn slider_lerp(min: f32, max: f32, normalized: f32) -> f32 {
    min + (max - min) * normalized.clamp(0.0, 1.0)
}

fn slider_unlerp(min: f32, max: f32, value: f32) -> f32 {
    if (max - min).abs() <= f32::EPSILON {
        return 0.0;
    }
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

fn slider_normalized_value(config: &BevyJepaConfig, kind: JepaControlSliderKind) -> f32 {
    match kind {
        JepaControlSliderKind::PatchDiffThreshold => {
            slider_unlerp(0.0, 0.20, config.patch_diff_threshold)
        }
        JepaControlSliderKind::ContextDensity => slider_unlerp(0.01, 1.0, config.context_density),
        JepaControlSliderKind::MinContextDensity => {
            slider_unlerp(0.0, 1.0, config.min_context_density)
        }
        JepaControlSliderKind::DenseFallbackDensity => {
            slider_unlerp(0.0, 1.0, config.patch_diff_dense_fallback_density)
        }
        JepaControlSliderKind::SubthresholdTrigger => {
            slider_unlerp(0.1, 4.0, config.patch_diff_refresh.subthreshold_trigger)
        }
        JepaControlSliderKind::AgeIntervalFrames => slider_unlerp(
            1.0,
            300.0,
            config.patch_diff_refresh.age_refresh_interval_frames as f32,
        ),
        JepaControlSliderKind::BlueNoiseDensity => slider_unlerp(
            0.0,
            0.05,
            config.patch_diff_refresh.blue_noise_refresh_density,
        ),
    }
}

fn slider_value_label(config: &BevyJepaConfig, kind: JepaControlSliderKind) -> String {
    match kind {
        JepaControlSliderKind::PatchDiffThreshold => {
            format!("{:.3}", config.patch_diff_threshold)
        }
        JepaControlSliderKind::ContextDensity => {
            format!("{:.0}%", config.context_density * 100.0)
        }
        JepaControlSliderKind::MinContextDensity => {
            format!("{:.0}%", config.min_context_density * 100.0)
        }
        JepaControlSliderKind::DenseFallbackDensity => {
            format!("{:.0}%", config.patch_diff_dense_fallback_density * 100.0)
        }
        JepaControlSliderKind::SubthresholdTrigger => {
            format!("{:.1}", config.patch_diff_refresh.subthreshold_trigger)
        }
        JepaControlSliderKind::AgeIntervalFrames => {
            format!("{}", config.patch_diff_refresh.age_refresh_interval_frames)
        }
        JepaControlSliderKind::BlueNoiseDensity => {
            format!(
                "{:.1}%",
                config.patch_diff_refresh.blue_noise_refresh_density * 100.0
            )
        }
    }
}

fn controls_summary(config: &BevyJepaConfig) -> String {
    format!(
        "{} | {} {}px | patch thr {:.3} ({:>4.1}% quality) | refresh {} / sub {} / age {} / blue {} | AnyUp {} {}",
        config.model_profile,
        if dense_pipeline_enabled(config) {
            "dense"
        } else {
            "sparse"
        },
        config.pipeline_image_size(),
        config.patch_diff_quality() * 100.0,
        config.patch_diff_threshold,
        on_off(config.patch_diff_refresh.enabled),
        on_off(config.patch_diff_refresh.subthreshold_enabled),
        on_off(config.patch_diff_refresh.age_refresh_enabled),
        on_off(config.patch_diff_refresh.blue_noise_enabled),
        if config.high_res_pca_every == 0 {
            "off".to_string()
        } else {
            format!("1/{}", config.high_res_pca_every)
        },
        config.anyup_attention_mode
    )
}

fn default_controls_help() -> &'static str {
    "Hover a control for details. Model, resolution, dense/sparse, and AnyUp mode rebuild the pipeline; mask sliders update the visual state."
}

fn hovered_control_help(
    query: &Query<(&Interaction, &JepaControlHelp), With<Button>>,
) -> Option<&'static str> {
    query.iter().find_map(|(interaction, help)| {
        matches!(interaction, Interaction::Hovered | Interaction::Pressed).then_some(help.text)
    })
}

fn control_help_text(action: JepaControlAction) -> &'static str {
    match action {
        JepaControlAction::TogglePanel => "Open or close the runtime settings panel.",
        JepaControlAction::ModelTtt => {
            "Use the sparse temporal TTT V-JEPA 2.1 package. This is the deployment path."
        }
        JepaControlAction::ModelBase => {
            "Use the base V-JEPA 2.1 encoder without TTT memory. Useful as a stability baseline."
        }
        JepaControlAction::PipelineSparse => {
            "Encode the patch-diff context mask and update only selected token-cache slots."
        }
        JepaControlAction::PipelineDense => {
            "Run full-frame dense JEPA and overwrite the whole token cache each processed frame."
        }
        JepaControlAction::Resolution256 => "Run the JEPA pipeline at 256x256 input resolution.",
        JepaControlAction::Resolution512 => {
            "Run the JEPA pipeline at 512x512. Better spatial detail, substantially more latency."
        }
        JepaControlAction::AnyUpOff => {
            "Disable high-resolution AnyUp decode. Low-res token PCA remains active."
        }
        JepaControlAction::AnyUpEvery8 => {
            "Run AnyUp on every eighth processed low-res frame to avoid stalling the core path."
        }
        JepaControlAction::AnyUpEvery1 => {
            "Run AnyUp on every processed frame. This is expensive and may lag on GPU."
        }
        JepaControlAction::AnyUpEfficientLocal => {
            "Use the portable efficient local-window AnyUp attention implementation."
        }
        JepaControlAction::AnyUpUpstreamMasked => {
            "Use the upstream-style masked attention variant for comparison."
        }
        JepaControlAction::PatchRefresh => {
            "Enable extra refresh patches beyond direct threshold hits to reduce stale tokens."
        }
        JepaControlAction::SubthresholdRefresh => {
            "Accumulate weak patch differences over time so slow semantic motion still refreshes."
        }
        JepaControlAction::AgeRefresh => {
            "Periodically refresh old token-cache slots even when patch difference is low."
        }
        JepaControlAction::BlueNoiseRefresh => {
            "Add a low-density spatially distributed refresh pattern to avoid clustered drift."
        }
    }
}

fn slider_help_text(kind: JepaControlSliderKind) -> &'static str {
    match kind {
        JepaControlSliderKind::PatchDiffThreshold => {
            "Patch difference threshold. Lower values admit more changed patches."
        }
        JepaControlSliderKind::ContextDensity => {
            "Maximum context density used to cap patch-diff masks before JEPA encoding."
        }
        JepaControlSliderKind::MinContextDensity => {
            "Minimum context density. Keep this low unless you need a guaranteed refresh floor."
        }
        JepaControlSliderKind::DenseFallbackDensity => {
            "Route very dense sparse masks to dense JEPA when sparse overhead would dominate."
        }
        JepaControlSliderKind::SubthresholdTrigger => {
            "Accumulated weak-diff score required before a subthreshold patch refreshes."
        }
        JepaControlSliderKind::AgeIntervalFrames => {
            "Maximum token age before eligible age-priority refresh."
        }
        JepaControlSliderKind::BlueNoiseDensity => "Extra blue-noise refresh density per frame.",
    }
}

fn control_button_label(config: &BevyJepaConfig, action: JepaControlAction) -> String {
    match action {
        JepaControlAction::TogglePanel => "Settings".to_string(),
        JepaControlAction::PatchRefresh => {
            format!("refresh {}", on_off(config.patch_diff_refresh.enabled))
        }
        JepaControlAction::SubthresholdRefresh => {
            format!(
                "subthr {}",
                on_off(config.patch_diff_refresh.subthreshold_enabled)
            )
        }
        JepaControlAction::AgeRefresh => {
            format!(
                "age {}",
                on_off(config.patch_diff_refresh.age_refresh_enabled)
            )
        }
        JepaControlAction::BlueNoiseRefresh => {
            format!(
                "blue {}",
                on_off(config.patch_diff_refresh.blue_noise_enabled)
            )
        }
        JepaControlAction::ModelTtt => "TTT".to_string(),
        JepaControlAction::ModelBase => "base".to_string(),
        JepaControlAction::PipelineSparse => "sparse".to_string(),
        JepaControlAction::PipelineDense => "dense".to_string(),
        JepaControlAction::Resolution256 => "256".to_string(),
        JepaControlAction::Resolution512 => "512".to_string(),
        JepaControlAction::AnyUpOff => "off".to_string(),
        JepaControlAction::AnyUpEvery8 => "1/8".to_string(),
        JepaControlAction::AnyUpEvery1 => "1/1".to_string(),
        JepaControlAction::AnyUpEfficientLocal => "local".to_string(),
        JepaControlAction::AnyUpUpstreamMasked => "masked".to_string(),
    }
}

fn control_button_active(
    config: &BevyJepaConfig,
    action: JepaControlAction,
    controls: &JepaControlsState,
) -> bool {
    match action {
        JepaControlAction::TogglePanel => controls.expanded,
        JepaControlAction::ModelTtt => {
            config.model_profile == BevyJepaModelPackageProfile::Vjepa21Ttt
                && config.encoder_source == BevyJepaEncoderSource::TrainedTtt
        }
        JepaControlAction::ModelBase => {
            config.model_profile == BevyJepaModelPackageProfile::Vjepa21Base
                || config.encoder_source == BevyJepaEncoderSource::BaseCheckpoint
        }
        JepaControlAction::PipelineSparse => !dense_pipeline_enabled(config),
        JepaControlAction::PipelineDense => dense_pipeline_enabled(config),
        JepaControlAction::Resolution256 => config.pipeline_image_size() == 256,
        JepaControlAction::Resolution512 => config.pipeline_image_size() == 512,
        JepaControlAction::AnyUpOff => config.high_res_pca_every == 0,
        JepaControlAction::AnyUpEvery8 => config.high_res_pca_every == 8,
        JepaControlAction::AnyUpEvery1 => config.high_res_pca_every == 1,
        JepaControlAction::AnyUpEfficientLocal => {
            config.anyup_attention_mode == burn_jepa::AnyUpAttentionMode::EfficientLocal
        }
        JepaControlAction::AnyUpUpstreamMasked => {
            config.anyup_attention_mode == burn_jepa::AnyUpAttentionMode::UpstreamMasked
        }
        JepaControlAction::PatchRefresh => config.patch_diff_refresh.enabled,
        JepaControlAction::SubthresholdRefresh => config.patch_diff_refresh.subthreshold_enabled,
        JepaControlAction::AgeRefresh => config.patch_diff_refresh.age_refresh_enabled,
        JepaControlAction::BlueNoiseRefresh => config.patch_diff_refresh.blue_noise_enabled,
    }
}

fn control_button_color(
    action: JepaControlAction,
    active: bool,
    interaction: Interaction,
) -> Color {
    if interaction == Interaction::Pressed {
        return Color::srgb(0.20, 0.30, 0.42);
    }
    if active {
        return Color::srgb(0.13, 0.30, 0.22);
    }
    if interaction == Interaction::Hovered {
        return Color::srgb(0.20, 0.23, 0.28);
    }
    if action == JepaControlAction::TogglePanel {
        Color::srgb(0.16, 0.18, 0.23)
    } else {
        Color::srgb(0.11, 0.13, 0.17)
    }
}

const fn on_off(enabled: bool) -> &'static str {
    if enabled { "on" } else { "off" }
}

fn keyboard_controls(
    mut config: ResMut<BevyJepaConfig>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut runtime: ResMut<JepaRuntime>,
    mut controls: ResMut<JepaControlsState>,
) {
    if keyboard.just_pressed(KeyCode::KeyC) {
        controls.expanded = !controls.expanded;
    }
    if keyboard.just_pressed(KeyCode::Space) {
        let next = config.mask_source.next();
        if next != config.mask_source {
            config.mask_source = next;
            runtime.reset_visual_state();
        }
    }
}

fn fit_visualization_node(
    config: Res<BevyJepaConfig>,
    texture: Res<JepaPanelTextures>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut nodes: Query<&mut Node>,
) {
    let Some(entity) = texture.root_entity else {
        return;
    };
    let Some(window) = windows.iter().next() else {
        return;
    };
    let Ok(mut node) = nodes.get_mut(entity) else {
        return;
    };
    let reserved_top = if config.show_metrics {
        UI_MARGIN_PX * 2.0 + METRIC_ROW_HEIGHT
    } else {
        0.0
    };
    let available_width = window.resolution.width().max(1.0);
    let available_height = (window.resolution.height().max(1.0) - reserved_top).max(1.0);
    let source_aspect = (texture.width.max(1) as f32 / texture.height.max(1) as f32)
        * visible_panel_count(&config) as f32;
    let available_image_height = (available_height - PANEL_LABEL_ROW_HEIGHT).max(1.0);
    let window_aspect = available_width / available_image_height;
    let (display_width, display_height) = if window_aspect > source_aspect {
        let height = available_image_height;
        (height * source_aspect, height)
    } else {
        let width = available_width;
        (width, width / source_aspect)
    };
    let total_height = display_height + PANEL_LABEL_ROW_HEIGHT;
    node.width = Val::Px(display_width.max(1.0));
    node.height = Val::Px(total_height.max(1.0));
    node.left = Val::Px(((available_width - display_width) * 0.5).max(0.0));
    node.top = Val::Px(reserved_top + ((available_height - total_height) * 0.5).max(0.0));
}

fn spawn_panel_image(builder: &mut ChildSpawnerCommands<'_>, image: Handle<Image>) -> Entity {
    builder
        .spawn((
            ImageNode::new(image).with_mode(NodeImageMode::Stretch),
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                grid_row: GridPlacement::start(2),
                ..default()
            },
        ))
        .id()
}

fn spawn_high_res_panel_image(
    builder: &mut ChildSpawnerCommands<'_>,
    image: Handle<Image>,
    visible: bool,
) -> Entity {
    let display = if visible {
        Display::Flex
    } else {
        Display::None
    };
    builder
        .spawn((
            HighResPanelElement,
            ImageNode::new(image).with_mode(NodeImageMode::Stretch),
            Node {
                display,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                grid_row: GridPlacement::start(2),
                ..default()
            },
        ))
        .id()
}

fn spawn_panel_label(
    builder: &mut ChildSpawnerCommands<'_>,
    label: &'static str,
    high_res: bool,
    visible: bool,
) {
    let display = if visible {
        Display::Flex
    } else {
        Display::None
    };
    let mut entity = builder.spawn((
        Text(label.to_string()),
        TextFont {
            font_size: bevy::text::FontSize::Px(20.0),
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            display,
            grid_row: GridPlacement::start(1),
            align_self: AlignSelf::Center,
            justify_self: JustifySelf::Center,
            padding: UiRect::horizontal(Val::Px(8.0)),
            ..default()
        },
    ));
    if high_res {
        entity.insert(HighResPanelElement);
    }
}

fn empty_panel_image(width: u32, height: u32) -> Image {
    use bevy::{
        asset::RenderAssetUsages,
        image::ImageSampler,
        render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
    };
    let mut image = Image::new(
        Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vec![0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    image.sampler = ImageSampler::nearest();
    image
}

fn micros_to_ms(us: u64) -> f64 {
    us as f64 / 1_000.0
}

fn micros_u64(value: u128) -> u64 {
    value.min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests;
