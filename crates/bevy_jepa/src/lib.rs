#![recursion_limit = "512"]

use std::{env, path::PathBuf, time::Instant};

use anyhow::{Context, Result, bail};
use bevy::{
    app::AppExit,
    prelude::*,
    render::{
        RenderPlugin,
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
    },
    tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future},
    ui::widget::ImageNode,
    window::PrimaryWindow,
};
use bevy_burn::{BevyBurnBridgePlugin, BurnDevice};
use burn::tensor::{
    Tensor, TensorData,
    backend::Backend,
    module::interpolate,
    ops::{InterpolateMode, InterpolateOptions},
};
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
};
use burn_jepa::{
    AnyUp, AnyUpConfig, AnyUpLoadOptions, FeatureFrameBatch, FeatureFrameEncodePath,
    FeatureFrameJepaEncoder, FeatureFrameMetrics, FeatureFramePipeline, FeatureFramePipelineConfig,
    FeatureFrameRequest, FeaturePcaUpdateConfig, FrameId, HighResFrameArtifacts,
    LowResFrameArtifacts, SparseJepaPatchDiffSparsityConfig, SparseTokenMask, TokenGridShape,
    TttBackpropMode, TttEncoderConfig, TttLayerPlacement, TttMemoryUpdateSource,
    TttSupervisionMode, VJepa2_1Model, VJepaConfig, VJepaLoadOptions, VJepaTttModel,
    coords_to_token_index, load_config_from_hf_dir, patch_diff_context_mask_from_scores,
    patch_diff_context_mask_from_video,
};
#[cfg(feature = "sparse-patchify-wgpu")]
use burn_jepa::{SparseMaskBatch, SparsePatchifyBatchPlan};
use image::{ImageReader, RgbaImage, imageops::FilterType};

mod config;
mod display;
pub mod platform;

pub use config::{
    BevyJepaConfig, BevyJepaDisplayTransfer, BevyJepaEncodePath, BevyJepaEncoderSource,
    BevyJepaFrameSource, BevyJepaMaskSource, DEFAULT_ANYUP_CHUNK_SIZE,
    DEFAULT_BOOTSTRAP_CONTEXT_DENSITY, DEFAULT_CAMERA_FPS, DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH, DEFAULT_CONTEXT_DENSITY, DEFAULT_HIGH_RES_PCA_EVERY, DEFAULT_IMAGE_SIZE,
    DEFAULT_MIN_CONTEXT_DENSITY, DEFAULT_PATCH_DIFF_QUALITY, DEFAULT_PATCH_DIFF_THRESHOLD,
    DEFAULT_PCA_UPDATE_EVERY, DEFAULT_TTT_MODEL_PATH, DEFAULT_VJEPA21_CHECKPOINT_DIR,
    DEFAULT_VJEPA21_CONFIG_PATH, DEFAULT_VJEPA21_WEIGHTS_NAME, MIN_PIPELINE_IMAGE_SIZE,
    PIPELINE_IMAGE_SIZE_MULTIPLE,
};
use display::{
    InputPanelData, JepaPanelTextures, StagePanelData, apply_input_panel_to_world,
    apply_stage_panels_to_world, clear_completed_gpu_uploads,
};

pub type JepaBevyBackend = burn::backend::WebGpu<f32, i32>;
pub type JepaBevyDevice = burn::backend::wgpu::WgpuDevice;

const UI_MARGIN_PX: f32 = 12.0;
const METRIC_ROW_HEIGHT: f32 = 24.0;
const PANEL_LABEL_ROW_HEIGHT: f32 = 34.0;

pub fn log(message: &str) {
    #[cfg(target_arch = "wasm32")]
    web_sys::console::log_1(&message.into());

    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("{message}");
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
    last_input_at: Option<Instant>,
    last_completion_at: Option<Instant>,
    input_fps: f64,
    low_res_fps: f64,
    high_res_fps: f64,
    last_error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RuntimePipelineSignature {
    encoder_source: BevyJepaEncoderSource,
    encode_path: BevyJepaEncodePath,
    ttt_model_path: Option<PathBuf>,
    jepa_checkpoint_dir: Option<PathBuf>,
    jepa_config_path: Option<PathBuf>,
    jepa_weights_name: String,
    image_size: usize,
    anyup_weights: Option<PathBuf>,
    anyup_attention_mode: burn_jepa::AnyUpAttentionMode,
    anyup_q_chunk_size: usize,
    pca_update_every: u64,
}

impl RuntimePipelineSignature {
    fn new(config: &BevyJepaConfig, image_size: usize) -> Self {
        Self {
            encoder_source: config.encoder_source,
            encode_path: config.encode_path,
            ttt_model_path: config.ttt_model_path.clone(),
            jepa_checkpoint_dir: config.jepa_checkpoint_dir.clone(),
            jepa_config_path: config.jepa_config_path.clone(),
            jepa_weights_name: config.jepa_weights_name.clone(),
            image_size,
            anyup_weights: config.anyup_weights.clone(),
            anyup_attention_mode: config.anyup_attention_mode,
            anyup_q_chunk_size: config.anyup_q_chunk_size,
            pca_update_every: config.pca_update_every,
        }
    }
}

struct CachedStaticFrame {
    path: Option<PathBuf>,
    image_size: usize,
    rgba: RgbaImage,
    image: Tensor<JepaBevyBackend, 4>,
}

#[derive(Resource, Clone, Debug, Default)]
pub struct BevyJepaMetrics {
    pub frame_index: u64,
    pub frame_ready: bool,
    pub encoder_source: BevyJepaEncoderSource,
    pub encode_path: FeatureFrameEncodePath,
    pub frame_source: BevyJepaFrameSource,
    pub camera_frame_received: bool,
    pub mask_source: BevyJepaMaskSource,
    pub display_transfer: BevyJepaDisplayTransfer,
    pub context_tokens: usize,
    pub dense_tokens: usize,
    pub grid_height: usize,
    pub grid_width: usize,
    pub patch_size: usize,
    pub stage_metrics: FeatureFrameMetrics,
    pub encode_us: u64,
    pub cache_update_us: u64,
    pub token_view_us: u64,
    pub anyup_context_us: u64,
    pub anyup_decode_us: u64,
    pub low_res_pca_us: u64,
    pub pca_update_us: u64,
    pub pca_sample_window_frames: usize,
    pub pca_sample_frames: usize,
    pub high_res_pca_us: u64,
    pub display_tensor_us: u64,
    pub total_us: u64,
    pub viewer_total_us: u64,
    pub pca_update_applied: bool,
    pub input_frame_index: u64,
    pub input_frames_seen: u64,
    pub completed_frames: u64,
    pub high_res_frames: u64,
    pub input_fps: f64,
    pub low_res_fps: f64,
    pub high_res_fps: f64,
    pub in_flight_frames: usize,
    pub queue_dropped_frames: usize,
    pub queue_overwritten_frames: usize,
    pub stale_completions: usize,
    pub last_error: Option<String>,
}

impl BevyJepaMetrics {
    pub fn density(&self) -> f64 {
        if self.dense_tokens == 0 {
            0.0
        } else {
            self.context_tokens as f64 / self.dense_tokens as f64
        }
    }

    pub fn fps(&self) -> f64 {
        self.viewer_fps()
    }

    pub fn core_fps(&self) -> f64 {
        if self.total_us == 0 {
            0.0
        } else {
            1_000_000.0 / self.total_us as f64
        }
    }

    pub fn viewer_fps(&self) -> f64 {
        if self.viewer_total_us == 0 {
            self.core_fps()
        } else {
            1_000_000.0 / self.viewer_total_us as f64
        }
    }

    pub fn aligns_with_stage_metrics(&self) -> bool {
        self.encode_us == self.stage_metrics.encode_us
            && self.cache_update_us == self.stage_metrics.cache_update_us
            && self.token_view_us == self.stage_metrics.token_view_us
            && self.anyup_context_us == self.stage_metrics.anyup_context_us
            && self.anyup_decode_us == self.stage_metrics.anyup_decode_us
            && self.low_res_pca_us == self.stage_metrics.low_res_pca_project_us
            && self.pca_update_us == self.stage_metrics.pca_update_us
            && self.pca_sample_window_frames == self.stage_metrics.pca_sample_window_frames
            && self.pca_sample_frames == self.stage_metrics.pca_sample_frames
            && self.high_res_pca_us == self.stage_metrics.pca_project_us
            && self.total_us == self.stage_metrics.total_us
            && self.encode_path == self.stage_metrics.encode_path
            && self.context_tokens == self.stage_metrics.sparse_width
            && self.dense_tokens == self.stage_metrics.dense_tokens_per_frame
            && self.pca_update_applied == self.stage_metrics.pca_update_applied
    }
}

#[derive(Clone, Debug)]
pub struct BevyJepaStepOutput {
    pub metrics: BevyJepaMetrics,
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
struct MetricsText;

pub struct BevyJepaPlugin;

impl Plugin for BevyJepaPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<JepaPanelTextures>()
            .init_resource::<JepaRuntime>()
            .init_resource::<BevyJepaMetrics>()
            .add_systems(Startup, setup_metrics_overlay)
            .add_systems(Update, setup_ui)
            .add_systems(Update, process_jepa_frame)
            .add_systems(Update, update_metrics_overlay)
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

fn load_viewer_encoder(
    config: &BevyJepaConfig,
    image_size: usize,
    device: &JepaBevyDevice,
) -> Result<(FeatureFrameJepaEncoder<JepaBevyBackend>, VJepaConfig)> {
    match config.encoder_source {
        BevyJepaEncoderSource::TinyTest => {
            let model_config = tiny_viewer_model_config(image_size);
            let jepa = VJepa2_1Model::<JepaBevyBackend>::new(&model_config, device);
            Ok((FeatureFrameJepaEncoder::base(jepa), model_config))
        }
        BevyJepaEncoderSource::BaseCheckpoint => {
            let (jepa, mut model_config) = load_base_checkpoint_model(config, image_size, device)?;
            model_config.image_size = image_size;
            Ok((FeatureFrameJepaEncoder::base(jepa), model_config))
        }
        BevyJepaEncoderSource::TrainedTtt => {
            let ttt_model_path = effective_ttt_model_path(config)?;
            if !ttt_model_path.exists() {
                bail!(
                    "trained TTT JEPA encoder checkpoint `{}` does not exist; pass --ttt-model or set BURN_JEPA_TTT_MODEL",
                    ttt_model_path.display()
                );
            }
            let model_config = viewer_model_config(config, image_size)?;
            let base = VJepa2_1Model::<JepaBevyBackend>::new(&model_config, device);
            let ttt_config = production_ttt_config();
            let ttt = VJepaTttModel::from_model(base, ttt_config, device)?
                .load_file(
                    ttt_model_path.clone(),
                    &NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
                    device,
                )
                .with_context(|| {
                    format!(
                        "load trained TTT JEPA encoder `{}`",
                        ttt_model_path.display()
                    )
                })?;
            Ok((FeatureFrameJepaEncoder::ttt(ttt), model_config))
        }
    }
}

fn load_base_checkpoint_model(
    config: &BevyJepaConfig,
    image_size: usize,
    device: &JepaBevyDevice,
) -> Result<(VJepa2_1Model<JepaBevyBackend>, VJepaConfig)> {
    if let Some(checkpoint_dir) = &config.jepa_checkpoint_dir {
        let checkpoint_dir = resolve_repo_relative_path(checkpoint_dir);
        let mut options = VJepaLoadOptions::default();
        options.weights_name = config.jepa_weights_name.clone();
        let (model, model_config, _report) = options
            .load_model(&checkpoint_dir, device)
            .with_context(|| {
                format!("load V-JEPA 2.1 checkpoint `{}`", checkpoint_dir.display())
            })?;
        return Ok((model, model_config));
    }

    let model_config = viewer_model_config(config, image_size)?;
    Ok((
        VJepa2_1Model::<JepaBevyBackend>::new(&model_config, device),
        model_config,
    ))
}

fn viewer_model_config(config: &BevyJepaConfig, image_size: usize) -> Result<VJepaConfig> {
    let mut model_config = if let Some(config_path) = &config.jepa_config_path {
        let config_path = resolve_repo_relative_path(config_path);
        if config_path.exists() {
            VJepaConfig::from_json_file(&config_path)
                .with_context(|| format!("load V-JEPA 2.1 config `{}`", config_path.display()))?
        } else if config.encoder_source == BevyJepaEncoderSource::TrainedTtt {
            bail!(
                "V-JEPA 2.1 config `{}` does not exist; pass --jepa-config or --encoder-source tiny-test",
                config_path.display()
            );
        } else {
            VJepaConfig::default()
        }
    } else if let Some(checkpoint_dir) = &config.jepa_checkpoint_dir {
        let checkpoint_dir = resolve_repo_relative_path(checkpoint_dir);
        load_config_from_hf_dir(&checkpoint_dir, &VJepaLoadOptions::default().config_name)
            .with_context(|| {
                format!("load V-JEPA 2.1 config from `{}`", checkpoint_dir.display())
            })?
    } else {
        VJepaConfig::default()
    };
    model_config.image_size = image_size;
    model_config.num_frames = 2;
    model_config.tubelet_size = 2;
    Ok(model_config)
}

fn tiny_viewer_model_config(image_size: usize) -> VJepaConfig {
    let mut model_config = VJepaConfig::tiny_for_tests();
    model_config.image_size = image_size;
    model_config.num_frames = 2;
    model_config.tubelet_size = 2;
    model_config
}

fn effective_ttt_model_path(config: &BevyJepaConfig) -> Result<PathBuf> {
    if let Some(path) = env::var_os("BURN_JEPA_TTT_MODEL") {
        return Ok(resolve_repo_relative_path(PathBuf::from(path)));
    }
    config
        .ttt_model_path
        .as_ref()
        .map(resolve_repo_relative_path)
        .ok_or_else(|| {
            anyhow::anyhow!("trained TTT JEPA encoder requires --ttt-model or BURN_JEPA_TTT_MODEL")
        })
}

fn resolve_repo_relative_path(path: impl Into<PathBuf>) -> PathBuf {
    let path = path.into();
    if path.is_absolute() || path.exists() {
        return path;
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let Some(workspace_root) = manifest_dir.parent().and_then(|parent| parent.parent()) else {
        return path;
    };
    let candidate = workspace_root.join(&path);
    if candidate.exists() { candidate } else { path }
}

fn production_ttt_config() -> TttEncoderConfig {
    TttEncoderConfig {
        layer_placement: TttLayerPlacement::Explicit,
        layers: vec![3, 7, 11],
        predictor_layers: Vec::new(),
        chunk_tokens: 196,
        ttt_lr: 0.003,
        use_projection: true,
        conv_kernel: 3,
        memory_update: TttMemoryUpdateSource::SelfHidden,
        supervision: TttSupervisionMode::FinalTeacher,
        hybrid_final_steps: 1,
        rollout_blocks: 2,
        backprop_mode: TttBackpropMode::FinalFeature,
        backprop_truncate_blocks: 1,
        freeze_pretrained: true,
        ..TttEncoderConfig::default()
    }
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
            let mut anyup_config = if config.anyup_weights.is_some() {
                AnyUpConfig::default()
            } else {
                AnyUpConfig::tiny_for_tests()
            }
            .with_attention_mode(config.anyup_attention_mode);
            anyup_config.input_dim = 3;
            let mut anyup = AnyUp::<JepaBevyBackend>::new(anyup_config, device)
                .context("initialize AnyUp viewer model")?;
            if let Some(path) = &config.anyup_weights {
                AnyUpLoadOptions::default()
                    .load_into(&mut anyup, path, device)
                    .with_context(|| format!("load AnyUp viewer weights `{}`", path.display()))?;
            }
            let pipeline_config = FeatureFramePipelineConfig {
                anyup_q_chunk_size: Some(config.anyup_q_chunk_size.max(1)),
                update_pca_online: false,
                pca_update: FeaturePcaUpdateConfig::rolling_low_res_every(
                    config.pca_update_every.max(1),
                ),
                measurement: if config.measure_stages {
                    burn_jepa::FeatureFrameMeasureConfig {
                        enabled: true,
                        sync_backend: config.sync_measurements,
                    }
                } else {
                    burn_jepa::FeatureFrameMeasureConfig::disabled()
                },
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
            let pipeline = self.pipeline.as_ref().expect("pipeline initialized");
            self.pipeline_grid = Some(pipeline.grid());
            self.pipeline_patch_size = Some(model_config.patch_size);
            self.model_config = Some(model_config);
            self.pipeline_signature = Some(signature);
            self.pending_stage = None;
            self.prev_image = None;
            self.prev_rgba = None;
            self.prev_stage_image = None;
            self.prev_stage_rgba = None;
            self.frame_index = 0;
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

    fn record_input_frame(&mut self, sequence: u64) {
        let now = Instant::now();
        if let Some(previous) = self.last_input_at {
            let seconds = now.duration_since(previous).as_secs_f64();
            if seconds.is_finite() && seconds > 0.0 {
                self.input_fps = 1.0 / seconds;
            }
        }
        self.last_input_at = Some(now);
        self.input_frames_seen = self.input_frames_seen.saturating_add(1);
        self.latest_input_sequence = sequence;
    }

    fn record_completion(&mut self, high_res_updated: bool) {
        let now = Instant::now();
        if let Some(previous) = self.last_completion_at {
            let seconds = now.duration_since(previous).as_secs_f64();
            if seconds.is_finite() && seconds > 0.0 {
                self.low_res_fps = 1.0 / seconds;
                if high_res_updated {
                    self.high_res_fps = 1.0 / seconds;
                }
            }
        }
        self.last_completion_at = Some(now);
        self.completed_frames = self.completed_frames.saturating_add(1);
        if high_res_updated {
            self.high_res_frames = self.high_res_frames.saturating_add(1);
        }
    }

    fn apply_runtime_counts(&self, metrics: &mut BevyJepaMetrics) {
        metrics.input_frames_seen = self.input_frames_seen;
        metrics.input_frame_index = self.latest_input_sequence;
        metrics.completed_frames = self.completed_frames;
        metrics.high_res_frames = self.high_res_frames;
        metrics.input_fps = self.input_fps;
        metrics.low_res_fps = self.low_res_fps;
        metrics.high_res_fps = self.high_res_fps;
        metrics.in_flight_frames =
            usize::from(self.active_task.is_some()) + usize::from(self.pending_stage.is_some());
        metrics.queue_dropped_frames = self.dropped_frames;
        metrics.queue_overwritten_frames = self.overwritten_frames;
        metrics.stale_completions = self.stale_completions;
    }
}

fn setup_ui(
    mut commands: Commands,
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
        grid_template_columns: RepeatedGridTrack::flex(4, 1.0),
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
        high_res_entity = Some(spawn_panel_image(builder, texture.high_res_image.clone()));

        for label in ["Input", "Sparse mask", "Token PCA", "AnyUp PCA"] {
            builder.spawn((
                Text(label.to_string()),
                TextFont {
                    font_size: bevy::text::FontSize::Px(20.0),
                    ..default()
                },
                TextColor(Color::WHITE),
                Node {
                    grid_row: GridPlacement::start(1),
                    align_self: AlignSelf::Center,
                    justify_self: JustifySelf::Center,
                    padding: UiRect::horizontal(Val::Px(8.0)),
                    ..default()
                },
            ));
        }
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

fn setup_metrics_overlay(mut commands: Commands) {
    commands
        .spawn((
            Text("jepa: ".to_string()),
            TextFont {
                font_size: bevy::text::FontSize::Px(14.0),
                ..default()
            },
            TextColor(Color::WHITE),
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(UI_MARGIN_PX),
                left: Val::Px(UI_MARGIN_PX),
                right: Val::Px(UI_MARGIN_PX),
                height: Val::Px(METRIC_ROW_HEIGHT),
                overflow: Overflow::clip(),
                ..default()
            },
            ZIndex(2),
        ))
        .with_child((
            MetricsText,
            TextColor(Color::srgb(1.0, 0.84, 0.0)),
            TextFont {
                font_size: bevy::text::FontSize::Px(14.0),
                ..default()
            },
            TextSpan::new(format_metrics_waiting_line()),
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
    let completed = {
        let mut runtime = world.resource_mut::<JepaRuntime>();
        poll_jepa_task(&config, &mut runtime, &device)
    };
    if let Some(completed) = completed {
        match completed {
            Ok(processed) => {
                let mut metrics = processed.metrics.clone();
                {
                    let mut runtime = world.resource_mut::<JepaRuntime>();
                    runtime.record_completion(processed.high_res_updated);
                    runtime.last_error = None;
                    runtime.apply_runtime_counts(&mut metrics);
                }
                apply_stage_panels_to_world(world, processed.panels, transfer);
                *world.resource_mut::<BevyJepaMetrics>() = metrics;
            }
            Err(err) => {
                world.resource_mut::<JepaRuntime>().last_error = Some(err.clone());
                world.resource_mut::<BevyJepaMetrics>().last_error = Some(err);
            }
        }
    }

    let result = {
        let mut runtime = world.resource_mut::<JepaRuntime>();
        process_runtime_source_frame(&config, &mut runtime, &device)
    };

    match result {
        Ok(Some(input)) => {
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
            world.resource_mut::<JepaRuntime>().last_error = None;
            *world.resource_mut::<BevyJepaMetrics>() = metrics;
        }
        Ok(None) => {
            world.resource_mut::<JepaRuntime>().last_error = None;
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
            world.resource_mut::<JepaRuntime>().last_error = Some(err.to_string());
            world.resource_mut::<BevyJepaMetrics>().last_error = Some(err.to_string());
        }
    }
}

fn poll_jepa_task(
    config: &BevyJepaConfig,
    runtime: &mut JepaRuntime,
    device: &JepaBevyDevice,
) -> Option<Result<StageProcessedFrame, String>> {
    let task = runtime.active_task.as_mut()?;
    let output = block_on(future::poll_once(task))?;
    runtime.active_task = None;
    let current_signature = RuntimePipelineSignature::new(config, config.pipeline_image_size());
    if output.signature == current_signature {
        let pipeline = output.pipeline;
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
    }))
}

fn spawn_jepa_task(
    config: BevyJepaConfig,
    signature: RuntimePipelineSignature,
    mut pipeline: FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    mask: SparseTokenMask,
    id: FrameId,
    grid: TokenGridShape,
    patch_size: usize,
    frame_source: BevyJepaFrameSource,
    camera_frame_received: bool,
    request: FeatureFrameRequest,
) -> Task<JepaAsyncTaskOutput> {
    AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::new).spawn(async move {
        let result = run_stage_pipeline_step(
            &config,
            &mut pipeline,
            image,
            &mask,
            id,
            grid,
            patch_size,
            frame_source,
            camera_frame_received,
            request,
        )
        .map_err(|err| err.to_string());
        JepaAsyncTaskOutput {
            signature,
            pipeline,
            result,
        }
    })
}

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
    let mask = run_sparse_mask_node(
        &config,
        prev_stage_image,
        prev_stage_rgba,
        rgba,
        &image,
        model_config,
        grid,
    )?
    .mask;
    Ok(spawn_jepa_task(
        config,
        signature,
        pipeline,
        image,
        mask,
        id,
        grid,
        patch_size,
        frame_source,
        camera_frame_received,
        request,
    ))
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
    let mask = run_sparse_mask_node(
        config,
        prev_image.as_ref(),
        prev_rgba.as_ref(),
        source.rgba.as_ref(),
        &image,
        &model_config,
        grid,
    )?
    .mask;
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
            &mask,
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
            &mask,
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
            &mask,
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
}

struct SparseMaskNodeOutput {
    mask: SparseTokenMask,
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
}

fn stage_request_for_frame(config: &BevyJepaConfig, frame_index: u64) -> FeatureFrameRequest {
    if config.high_res_pca_every > 0 && frame_index % config.high_res_pca_every == 0 {
        FeatureFrameRequest::full_pca()
    } else {
        FeatureFrameRequest::low_res()
    }
}

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

struct JepaAsyncTaskOutput {
    signature: RuntimePipelineSignature,
    pipeline: FeatureFramePipeline<JepaBevyBackend>,
    result: Result<StageProcessedFrame, String>,
}

#[derive(Clone, Copy)]
struct MetricFrameContext {
    frame_index: u64,
    encoder_source: BevyJepaEncoderSource,
    frame_source: BevyJepaFrameSource,
    camera_frame_received: bool,
    mask_source: BevyJepaMaskSource,
    display_transfer: BevyJepaDisplayTransfer,
    context_tokens: usize,
    dense_tokens: usize,
    grid: TokenGridShape,
    patch_size: usize,
}

fn bevy_metrics_from_stage(
    frame: MetricFrameContext,
    stage_metrics: FeatureFrameMetrics,
    display_tensor_us: u64,
    viewer_total_us: u64,
) -> BevyJepaMetrics {
    BevyJepaMetrics {
        frame_index: frame.frame_index,
        frame_ready: true,
        encoder_source: frame.encoder_source,
        encode_path: stage_metrics.encode_path,
        frame_source: frame.frame_source,
        camera_frame_received: frame.camera_frame_received,
        mask_source: frame.mask_source,
        display_transfer: frame.display_transfer,
        context_tokens: frame.context_tokens,
        dense_tokens: frame.dense_tokens,
        grid_height: frame.grid.height,
        grid_width: frame.grid.width,
        patch_size: frame.patch_size,
        encode_us: stage_metrics.encode_us,
        cache_update_us: stage_metrics.cache_update_us,
        token_view_us: stage_metrics.token_view_us,
        anyup_context_us: stage_metrics.anyup_context_us,
        anyup_decode_us: stage_metrics.anyup_decode_us,
        low_res_pca_us: stage_metrics.low_res_pca_project_us,
        pca_update_us: stage_metrics.pca_update_us,
        pca_sample_window_frames: stage_metrics.pca_sample_window_frames,
        pca_sample_frames: stage_metrics.pca_sample_frames,
        high_res_pca_us: stage_metrics.pca_project_us,
        display_tensor_us,
        total_us: stage_metrics.total_us,
        viewer_total_us,
        pca_update_applied: stage_metrics.pca_update_applied,
        input_frame_index: frame.frame_index,
        input_frames_seen: 0,
        completed_frames: 0,
        high_res_frames: u64::from(stage_metrics.anyup_decode_us > 0),
        input_fps: 0.0,
        low_res_fps: 0.0,
        high_res_fps: 0.0,
        in_flight_frames: 0,
        queue_dropped_frames: 0,
        queue_overwritten_frames: 0,
        stale_completions: 0,
        last_error: None,
        stage_metrics,
    }
}

fn run_stage_step_with_config_and_request(
    config: &BevyJepaConfig,
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    mask: &SparseTokenMask,
    request: FeatureFrameRequest,
) -> Result<StageFrame> {
    let wall_start = Instant::now();
    let measured = run_feature_frame_pipeline(config, pipeline, image, mask, request)?;
    Ok(StageFrame {
        output: measured.output,
        metrics: measured.metrics,
        wall_us: micros_u64(wall_start.elapsed().as_micros()),
    })
}

fn run_feature_frame_pipeline(
    config: &BevyJepaConfig,
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    mask: &SparseTokenMask,
    request: FeatureFrameRequest,
) -> Result<burn_jepa::MeasuredFeatureFrameBatch<JepaBevyBackend>> {
    match config.encode_path {
        BevyJepaEncodePath::DensePatchEmbed => {
            pipeline.step_image_with_mask_nodes_measured(image, mask, request)
        }
        BevyJepaEncodePath::Auto => run_feature_frame_pipeline_auto(pipeline, image, mask, request),
        BevyJepaEncodePath::SparsePatchify => {
            run_feature_frame_pipeline_sparse_patchify(pipeline, image, mask, request)
        }
    }
}

fn run_feature_frame_pipeline_auto(
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    mask: &SparseTokenMask,
    request: FeatureFrameRequest,
) -> Result<burn_jepa::MeasuredFeatureFrameBatch<JepaBevyBackend>> {
    #[cfg(feature = "sparse-patchify-wgpu")]
    {
        run_feature_frame_pipeline_sparse_patchify(pipeline, image, mask, request)
    }

    #[cfg(not(feature = "sparse-patchify-wgpu"))]
    {
        pipeline.step_image_with_mask_nodes_measured(image, mask, request)
    }
}

#[cfg(feature = "sparse-patchify-wgpu")]
fn run_feature_frame_pipeline_sparse_patchify(
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    mask: &SparseTokenMask,
    request: FeatureFrameRequest,
) -> Result<burn_jepa::MeasuredFeatureFrameBatch<JepaBevyBackend>> {
    let batch_mask = SparseMaskBatch::uniform(mask.clone(), pipeline.batch(), pipeline.device())?;
    let patchify_plan =
        SparsePatchifyBatchPlan::new(batch_mask, pipeline.grid(), pipeline.device())?;
    pipeline.step_image_with_sparse_patchify_plan_wgpu_nodes_measured(
        image,
        &patchify_plan,
        request,
        pipeline.config().measurement,
    )
}

#[cfg(not(feature = "sparse-patchify-wgpu"))]
fn run_feature_frame_pipeline_sparse_patchify(
    _pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    _image: Tensor<JepaBevyBackend, 4>,
    _mask: &SparseTokenMask,
    _request: FeatureFrameRequest,
) -> Result<burn_jepa::MeasuredFeatureFrameBatch<JepaBevyBackend>> {
    bail!("Bevy sparse patchify requires building bevy_jepa with --features sparse-patchify-wgpu")
}

fn run_stage_step_metrics(
    config: &BevyJepaConfig,
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    mask: &SparseTokenMask,
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
        mask,
        id,
        grid,
        patch_size,
        frame_source,
        camera_frame_received,
        FeatureFrameRequest::full_pca(),
    )
}

fn run_stage_step_metrics_with_request(
    config: &BevyJepaConfig,
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    mask: &SparseTokenMask,
    id: FrameId,
    grid: TokenGridShape,
    patch_size: usize,
    frame_source: BevyJepaFrameSource,
    camera_frame_received: bool,
    request: FeatureFrameRequest,
) -> Result<ProcessedFrame> {
    let stage = run_stage_step_with_config_and_request(config, pipeline, image, mask, request)?;
    let frame = MetricFrameContext {
        frame_index: id.sequence,
        encoder_source: config.encoder_source,
        frame_source,
        camera_frame_received,
        mask_source: config.mask_source,
        display_transfer: config.display_transfer,
        context_tokens: mask.len(),
        dense_tokens: mask.dense_len(),
        grid,
        patch_size,
    };
    let metrics = bevy_metrics_from_stage(frame, stage.metrics, 0, stage.wall_us);
    Ok(ProcessedFrame { metrics })
}

fn run_pipeline_step(
    config: &BevyJepaConfig,
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    mask: &SparseTokenMask,
    id: FrameId,
    grid: TokenGridShape,
    patch_size: usize,
    frame_source: BevyJepaFrameSource,
    camera_frame_received: bool,
    request: FeatureFrameRequest,
) -> Result<ProcessedFrame> {
    let stage =
        run_stage_step_with_config_and_request(config, pipeline, image.clone(), mask, request)?;
    let metrics = stage.metrics;
    let output = stage.output;
    let image_size = pipeline.image_size();
    let display_start = Instant::now();
    let low_res_pca = low_res_pca_or_features(output.low_res)?;
    let high_res_pca = high_res_pca_or_low_res(output.high_res, low_res_pca.clone())?;
    let input_rgba = nchw_to_rgba_tensor(image)?;
    let mask_rgba = sparse_mask_to_rgba_tensor::<JepaBevyBackend>(
        mask,
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
            JepaBevyBackend::sync(&display_device)?;
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
    if config.display_transfer == BevyJepaDisplayTransfer::Cpu && config.sync_measurements {
        JepaBevyBackend::sync(&display_device)?;
    }
    let display_tensor_us = micros_u64(display_start.elapsed().as_micros());
    let viewer_total_us = stage.wall_us.saturating_add(display_tensor_us);
    let frame = MetricFrameContext {
        frame_index: id.sequence,
        encoder_source: config.encoder_source,
        frame_source,
        camera_frame_received,
        mask_source: config.mask_source,
        context_tokens: mask.len(),
        dense_tokens: mask.dense_len(),
        display_transfer: config.display_transfer,
        grid,
        patch_size,
    };
    Ok(ProcessedFrame {
        metrics: bevy_metrics_from_stage(frame, metrics, display_tensor_us, viewer_total_us),
    })
}

fn run_stage_pipeline_step(
    config: &BevyJepaConfig,
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    mask: &SparseTokenMask,
    id: FrameId,
    grid: TokenGridShape,
    patch_size: usize,
    frame_source: BevyJepaFrameSource,
    camera_frame_received: bool,
    request: FeatureFrameRequest,
) -> Result<StageProcessedFrame> {
    let stage =
        run_stage_step_with_config_and_request(config, pipeline, image.clone(), mask, request)?;
    let metrics = stage.metrics;
    let output = stage.output;
    let image_size = pipeline.image_size();
    let display_start = Instant::now();
    let low_res_pca = low_res_pca_or_features(output.low_res)?;
    let high_res_pca = output.high_res.and_then(|high_res| high_res.pca_display);
    let mask_rgba = sparse_mask_to_rgba_tensor::<JepaBevyBackend>(
        mask,
        pipeline.grid(),
        image_size,
        &image.device(),
    )?;
    let low_res_rgba = nchw_to_rgba_tensor(resize_nchw(low_res_pca, image_size))?;
    let high_res_updated = high_res_pca.is_some();
    let high_res_rgba = high_res_pca
        .map(|pca| nchw_to_rgba_tensor(resize_nchw(pca, image_size)))
        .transpose()?;
    if config.sync_measurements {
        JepaBevyBackend::sync(&image.device())?;
    }
    let display_tensor_us = micros_u64(display_start.elapsed().as_micros());
    let viewer_total_us = stage.wall_us.saturating_add(display_tensor_us);
    let frame = MetricFrameContext {
        frame_index: id.sequence,
        encoder_source: config.encoder_source,
        frame_source,
        camera_frame_received,
        mask_source: config.mask_source,
        display_transfer: config.display_transfer,
        context_tokens: mask.len(),
        dense_tokens: mask.dense_len(),
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

fn run_sparse_mask_node(
    config: &BevyJepaConfig,
    prev_image: Option<&Tensor<JepaBevyBackend, 4>>,
    prev_rgba: Option<&RgbaImage>,
    rgba: Option<&RgbaImage>,
    image: &Tensor<JepaBevyBackend, 4>,
    model_config: &VJepaConfig,
    grid: TokenGridShape,
) -> Result<SparseMaskNodeOutput> {
    let mask = match config.mask_source {
        BevyJepaMaskSource::Autogaze => anyhow::bail!(
            "AutoGaze mask source requires a loaded model-backed AutoGaze node; \
             this viewer will not synthesize AutoGaze masks. Use --mask-source patch-diff \
             or wire a real burn_autogaze pipeline into this graph."
        ),
        BevyJepaMaskSource::PatchDiff => patch_diff_mask(
            prev_image,
            prev_rgba,
            rgba,
            image,
            model_config,
            grid,
            config,
        ),
    }?;
    Ok(SparseMaskNodeOutput { mask })
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
            values[index] = raw[pixel] as f32 / 255.0;
            values[height * width + index] = raw[pixel + 1] as f32 / 255.0;
            values[2 * height * width + index] = raw[pixel + 2] as f32 / 255.0;
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

fn patch_diff_mask(
    prev_image: Option<&Tensor<JepaBevyBackend, 4>>,
    prev_rgba: Option<&RgbaImage>,
    rgba: Option<&RgbaImage>,
    image: &Tensor<JepaBevyBackend, 4>,
    model_config: &VJepaConfig,
    grid: TokenGridShape,
    config: &BevyJepaConfig,
) -> Result<SparseTokenMask> {
    let Some(prev_image) = prev_image else {
        return center_prior_mask(grid, config.bootstrap_context_tokens(grid.len()));
    };
    if let (Some(prev_rgba), Some(rgba)) = (prev_rgba, rgba) {
        return patch_diff_mask_from_rgba(prev_rgba, rgba, model_config, grid, config);
    }
    let video = Tensor::cat(
        vec![
            prev_image.clone().reshape([
                1,
                3,
                1,
                image.shape().dims::<4>()[2],
                image.shape().dims::<4>()[3],
            ]),
            image.clone().reshape([
                1,
                3,
                1,
                image.shape().dims::<4>()[2],
                image.shape().dims::<4>()[3],
            ]),
        ],
        2,
    );
    let config = patch_diff_sparsity_config(config, grid);
    patch_diff_context_mask_from_video(&video, model_config, grid, &config)
}

fn patch_diff_mask_from_rgba(
    prev: &RgbaImage,
    current: &RgbaImage,
    model_config: &VJepaConfig,
    grid: TokenGridShape,
    config: &BevyJepaConfig,
) -> Result<SparseTokenMask> {
    anyhow::ensure!(
        grid.depth == 1,
        "RGBA patch-diff mask expects a single-frame token grid"
    );
    anyhow::ensure!(
        prev.dimensions() == current.dimensions(),
        "RGBA patch-diff frames must have matching dimensions"
    );
    let patch_size = model_config.patch_size.max(1);
    let height = current.height() as usize;
    let width = current.width() as usize;
    anyhow::ensure!(
        height == grid.height * patch_size && width == grid.width * patch_size,
        "RGBA patch-diff frame size must match the V-JEPA patch grid"
    );
    let prev = prev.as_raw();
    let current = current.as_raw();
    let mut scores = vec![0.0f32; grid.len()];
    for row in 0..grid.height {
        for col in 0..grid.width {
            let mut diff_sum = 0.0f32;
            for y in row * patch_size..(row + 1) * patch_size {
                for x in col * patch_size..(col + 1) * patch_size {
                    let offset = (y * width + x) * 4;
                    for channel in 0..3 {
                        diff_sum += (current[offset + channel] as f32
                            - prev[offset + channel] as f32)
                            .abs()
                            / 255.0;
                    }
                }
            }
            let denom = (3 * patch_size * patch_size) as f32;
            scores[coords_to_token_index(0, row, col, grid)] = diff_sum / denom;
        }
    }
    patch_diff_context_mask_from_scores(scores, grid, &patch_diff_sparsity_config(config, grid))
}

fn patch_diff_sparsity_config(
    config: &BevyJepaConfig,
    grid: TokenGridShape,
) -> SparseJepaPatchDiffSparsityConfig {
    let max_context_tokens = config.context_tokens(grid.len());
    let min_context_tokens = config
        .min_context_tokens(grid.len())
        .min(max_context_tokens);
    let target_tokens = grid.len().saturating_sub(max_context_tokens).max(1);
    SparseJepaPatchDiffSparsityConfig::adaptive_threshold(
        config.patch_diff_threshold,
        min_context_tokens,
        max_context_tokens,
        target_tokens,
    )
}

fn center_prior_mask(grid: TokenGridShape, context_tokens: usize) -> Result<SparseTokenMask> {
    let center_row = grid.height.saturating_sub(1) as f32 * 0.5;
    let center_col = grid.width.saturating_sub(1) as f32 * 0.5;
    let mut scores = Vec::with_capacity(grid.len());
    for row in 0..grid.height {
        for col in 0..grid.width {
            let dr = row as f32 - center_row;
            let dc = col as f32 - center_col;
            let dist = dr * dr + dc * dc;
            scores.push((coords_to_token_index(0, row, col, grid), dist));
        }
    }
    scores.sort_by(|left, right| {
        left.1
            .partial_cmp(&right.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    SparseTokenMask::new(
        scores
            .into_iter()
            .take(context_tokens.max(1).min(grid.len()))
            .map(|(index, _)| index)
            .collect(),
        grid.len(),
    )
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
            values[index] = (base + blob * 0.6).clamp(0.0, 1.0);
            values[height * width + index] = ((1.0 - base) * 0.55 + blob * 0.35).clamp(0.0, 1.0);
            values[2 * height * width + index] =
                ((phase.sin() * 0.15 + 0.25) + blob * 0.5).clamp(0.0, 1.0);
        }
    }
    Tensor::<JepaBevyBackend, 4>::from_data(TensorData::new(values, [1, 3, height, width]), device)
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
    mut query: Query<&mut TextSpan, With<MetricsText>>,
) {
    if !config.show_metrics || (!metrics.is_changed() && !runtime.is_changed()) {
        return;
    }
    let text = if let Some(error) = runtime.last_error.as_ref().or(metrics.last_error.as_ref()) {
        format!(
            "status:{:<12} error:{:<64}",
            "error",
            truncate_metric_text(error, 64)
        )
    } else {
        format_metrics_line(&config, &metrics)
    };
    for mut span in &mut query {
        **span = text.clone();
    }
}

fn format_metrics_waiting_line() -> String {
    format_metrics_line(&BevyJepaConfig::default(), &BevyJepaMetrics::default())
}

fn format_metrics_line(config: &BevyJepaConfig, metrics: &BevyJepaMetrics) -> String {
    let source = metrics_source_status(config, metrics);
    format!(
        "src:{:<18} model:{:<13} enc:{:<15} mask:{:<10} seq:{:>5}/{:<5} grid:{:>3}x{:<3} p:{:<2} sparse:{:>5.1}% fps:{:>5.1}/{:>5.1}/{:>5.1} infl:{:>1} drop:{:>4} ovw:{:>4} view:{:>7.2}ms core:{:>7.2}ms disp:{:>6.2}ms enc:{:>6.2}ms cache:{:>6.2}ms any:{:>6.2}/{:>6.2}ms pca:{:>6.2}/{:>6.2}ms upd:{:<3} {:>3}/{:<3}f",
        source,
        metrics.encoder_source,
        metrics.encode_path,
        metrics.mask_source,
        metrics.input_frame_index,
        metrics.frame_index,
        metrics.grid_height,
        metrics.grid_width,
        metrics.patch_size,
        metrics.density() * 100.0,
        metrics.input_fps,
        metrics.low_res_fps,
        metrics.high_res_fps,
        metrics.in_flight_frames,
        metrics.queue_dropped_frames,
        metrics.queue_overwritten_frames,
        micros_to_ms(metrics.viewer_total_us),
        micros_to_ms(metrics.total_us),
        micros_to_ms(metrics.display_tensor_us),
        micros_to_ms(metrics.encode_us),
        micros_to_ms(metrics.cache_update_us),
        micros_to_ms(metrics.anyup_context_us),
        micros_to_ms(metrics.anyup_decode_us),
        micros_to_ms(metrics.low_res_pca_us),
        micros_to_ms(metrics.high_res_pca_us),
        if metrics.pca_update_applied {
            "yes"
        } else {
            "no"
        },
        metrics.pca_sample_frames,
        metrics.pca_sample_window_frames,
    )
}

fn metrics_source_status(config: &BevyJepaConfig, metrics: &BevyJepaMetrics) -> &'static str {
    if !metrics.frame_ready {
        return match config.source {
            BevyJepaFrameSource::Camera => "camera:wait",
            BevyJepaFrameSource::StaticImage => "static:wait",
            BevyJepaFrameSource::SyntheticLocalMotion => "synthetic:wait",
        };
    }

    match metrics.frame_source {
        BevyJepaFrameSource::Camera => "camera:live",
        BevyJepaFrameSource::StaticImage => "static:ready",
        BevyJepaFrameSource::SyntheticLocalMotion => "synthetic:ready",
    }
}

fn truncate_metric_text(value: &str, max_chars: usize) -> String {
    let total = value.chars().count();
    if total <= max_chars {
        return value.to_string();
    }
    let keep = max_chars.saturating_sub(3);
    let mut output = String::new();
    for (index, ch) in value.chars().enumerate() {
        if index >= keep {
            break;
        }
        output.push(ch);
    }
    output.push_str("...");
    output
}

fn keyboard_controls(
    mut config: ResMut<BevyJepaConfig>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut runtime: ResMut<JepaRuntime>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        let next = config.mask_source.next();
        if next != config.mask_source {
            config.mask_source = next;
            runtime.prev_image = None;
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
    let source_aspect = (texture.width.max(1) as f32 / texture.height.max(1) as f32) * 4.0;
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
mod tests {
    use super::*;
    use burn_jepa::FeatureFrameJepaEncoderKind;

    fn tiny_viewer_config() -> BevyJepaConfig {
        BevyJepaConfig {
            encoder_source: BevyJepaEncoderSource::TinyTest,
            ttt_model_path: None,
            jepa_checkpoint_dir: None,
            jepa_config_path: None,
            ..BevyJepaConfig::default()
        }
    }

    fn values4(tensor: Tensor<JepaBevyBackend, 4>) -> Vec<f32> {
        tensor.to_data().to_vec::<f32>().expect("tensor values")
    }

    #[test]
    fn center_prior_mask_keeps_requested_density() {
        let grid = TokenGridShape::new(1, 4, 4);
        let mask = center_prior_mask(grid, 5).expect("mask");
        assert_eq!(mask.dense_len(), 16);
        assert_eq!(mask.len(), 5);
    }

    #[test]
    fn synthetic_source_uses_model_sized_tensor() {
        let device = JepaBevyDevice::default();
        let image = synthetic_image_tensor(0, 64, &device);
        assert_eq!(image.shape().dims::<4>(), [1, 3, 64, 64]);
    }

    #[test]
    fn default_source_is_camera() {
        assert_eq!(
            BevyJepaConfig::default().source,
            BevyJepaFrameSource::Camera
        );
    }

    #[test]
    fn default_mask_source_is_patch_diff() {
        assert_eq!(
            BevyJepaConfig::default().encoder_source,
            BevyJepaEncoderSource::TrainedTtt
        );
        assert_eq!(
            BevyJepaConfig::default().encode_path,
            BevyJepaEncodePath::Auto
        );
        assert_eq!(
            BevyJepaConfig::default().mask_source,
            BevyJepaMaskSource::PatchDiff
        );
        assert_eq!(BevyJepaConfig::default().image_size, DEFAULT_IMAGE_SIZE);
        assert_eq!(
            BevyJepaConfig::default().pipeline_image_size(),
            DEFAULT_IMAGE_SIZE
        );
        assert_eq!(BevyJepaConfig::default().context_density, 1.0);
        assert_eq!(
            BevyJepaConfig::default().min_context_density,
            DEFAULT_MIN_CONTEXT_DENSITY
        );
        assert_eq!(BevyJepaConfig::default().min_context_density, 0.0);
        assert!(
            (BevyJepaConfig::default().patch_diff_quality() - DEFAULT_PATCH_DIFF_QUALITY).abs()
                <= f32::EPSILON
        );
        assert_eq!(BevyJepaConfig::default().bootstrap_context_density, 1.0);
        assert_eq!(
            BevyJepaConfig::default().pca_update_every,
            DEFAULT_PCA_UPDATE_EVERY
        );
        assert_eq!(
            BevyJepaConfig::default().high_res_pca_every,
            DEFAULT_HIGH_RES_PCA_EVERY
        );
        assert_eq!(
            stage_request_for_frame(&BevyJepaConfig::default(), 0),
            FeatureFrameRequest::full_pca()
        );
        assert_eq!(
            stage_request_for_frame(&BevyJepaConfig::default(), 1),
            FeatureFrameRequest::low_res()
        );
        assert_eq!(
            BevyJepaMaskSource::PatchDiff.next(),
            BevyJepaMaskSource::PatchDiff
        );
    }

    #[test]
    fn viewer_pipeline_promotes_small_image_requests_to_minimum_resolution() {
        let config = BevyJepaConfig {
            image_size: 64,
            ..BevyJepaConfig::default()
        };
        assert_eq!(config.pipeline_image_size(), MIN_PIPELINE_IMAGE_SIZE);
    }

    #[test]
    fn viewer_pipeline_accepts_trained_256_resolution() {
        let config = BevyJepaConfig {
            image_size: MIN_PIPELINE_IMAGE_SIZE,
            ..BevyJepaConfig::default()
        };
        assert_eq!(config.pipeline_image_size(), 256);
    }

    #[test]
    fn default_patch_diff_quality_is_threshold_not_static_sparsity() {
        let config = BevyJepaConfig::default();
        assert_eq!(config.min_context_tokens(256), 1);
        assert_eq!(config.min_context_tokens(1024), 1);
        assert!((config.patch_diff_threshold - 0.15).abs() <= f32::EPSILON);
        assert!((config.patch_diff_quality() - 0.85).abs() <= f32::EPSILON);
    }

    #[test]
    fn default_patch_diff_static_frame_keeps_dynamic_minimum_only() {
        let device = JepaBevyDevice::default();
        let config = BevyJepaConfig {
            source: BevyJepaFrameSource::Camera,
            mask_source: BevyJepaMaskSource::PatchDiff,
            ..BevyJepaConfig::default()
        };
        let mut model_config = VJepaConfig::tiny_for_tests();
        model_config.image_size = 64;
        model_config.num_frames = 2;
        model_config.tubelet_size = 2;
        model_config.patch_size = 16;
        let grid = TokenGridShape::new(1, 4, 4);
        let previous_rgba = RgbaImage::new(64, 64);
        let current_rgba = previous_rgba.clone();
        let previous = rgba_image_to_tensor(previous_rgba.clone(), 64, &device).expect("prev");
        let current = rgba_image_to_tensor(current_rgba.clone(), 64, &device).expect("current");

        let output = run_sparse_mask_node(
            &config,
            Some(&previous),
            Some(&previous_rgba),
            Some(&current_rgba),
            &current,
            &model_config,
            grid,
        )
        .expect("static patch-diff mask");

        assert_eq!(output.mask.dense_len(), grid.len());
        assert_eq!(output.mask.len(), 1);
        assert!(
            output.mask.len() < (grid.len() as f32 * DEFAULT_PATCH_DIFF_QUALITY).round() as usize,
            "patch-diff quality must not be interpreted as a fixed sparsity floor"
        );
    }

    #[test]
    fn low_res_feature_fallback_preserves_nchw_spatial_grid() {
        let device = JepaBevyDevice::default();
        let features = Tensor::<JepaBevyBackend, 4>::from_data(
            TensorData::new(
                vec![
                    1.0, 2.0, 3.0, 4.0, //
                    10.0, 20.0, 30.0, 40.0, //
                    -1.0, -2.0, -3.0, -4.0, //
                    99.0, 98.0, 97.0, 96.0,
                ],
                [1, 4, 2, 2],
            ),
            &device,
        );

        let display = low_res_pca_or_features(LowResFrameArtifacts {
            features,
            pca_display: None,
        })
        .expect("low-res display fallback");

        assert_eq!(display.shape().dims::<4>(), [1, 3, 2, 2]);
        assert_eq!(
            values4(display),
            vec![
                1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, -1.0, -2.0, -3.0, -4.0
            ]
        );
    }

    #[test]
    fn low_res_display_resize_preserves_patch_grid_colors() {
        let device = JepaBevyDevice::default();
        let low_res = Tensor::<JepaBevyBackend, 4>::from_data(
            TensorData::new(
                vec![
                    1.0, 0.0, 0.0, 1.0, //
                    0.0, 1.0, 0.0, 1.0, //
                    0.0, 0.0, 1.0, 1.0,
                ],
                [1, 3, 2, 2],
            ),
            &device,
        );

        let resized = resize_nchw(low_res, [32, 32]);
        let rgba = tensor_rgba_to_host(nchw_to_rgba_tensor(resized).expect("rgba tensor"))
            .expect("host rgba");
        let sample = |x: usize, y: usize| {
            let offset = (y * 32 + x) * 4;
            [rgba[offset], rgba[offset + 1], rgba[offset + 2]]
        };

        assert_eq!(sample(4, 4), [255, 0, 0]);
        assert_eq!(sample(28, 4), [0, 255, 0]);
        assert_eq!(sample(4, 28), [0, 0, 255]);
        assert_eq!(sample(28, 28), [255, 255, 255]);
    }

    #[test]
    fn viewer_pipeline_rounds_image_requests_to_patch_multiple() {
        let config = BevyJepaConfig {
            image_size: MIN_PIPELINE_IMAGE_SIZE + 1,
            ..BevyJepaConfig::default()
        };
        assert_eq!(
            config.pipeline_image_size() % PIPELINE_IMAGE_SIZE_MULTIPLE,
            0
        );
        assert!(config.pipeline_image_size() > MIN_PIPELINE_IMAGE_SIZE);
    }

    #[test]
    fn camera_source_waits_without_generating_synthetic_warmup() {
        let device = JepaBevyDevice::default();
        let config = BevyJepaConfig {
            source: BevyJepaFrameSource::Camera,
            ..BevyJepaConfig::default()
        };
        let mut pipeline = BevyJepaHeadlessPipeline::new(config, device);
        let err = pipeline
            .step_stage_only()
            .expect_err("camera source should wait for a real frame");
        assert!(err.to_string().contains("camera frame is not ready"));
    }

    #[test]
    fn camera_source_without_frame_does_not_initialize_pipeline() {
        let device = JepaBevyDevice::default();
        let config = BevyJepaConfig {
            source: BevyJepaFrameSource::Camera,
            ..BevyJepaConfig::default()
        };
        let mut runtime = JepaRuntime::default();
        let processed =
            process_runtime_frame(&config, &mut runtime, &device, BevyJepaStepMode::StageOnly)
                .expect("camera wait should not be an error inside the Bevy schedule");
        assert!(processed.is_none());
        assert_eq!(runtime.frame_index, 0);
        assert!(runtime.pipeline.is_none());
        assert!(runtime.prev_image.is_none());
    }

    #[test]
    fn source_node_keeps_latest_pending_stage_frame_while_worker_runs() {
        let device = JepaBevyDevice::default();
        let config = BevyJepaConfig {
            source: BevyJepaFrameSource::SyntheticLocalMotion,
            high_res_pca_every: 8,
            ..tiny_viewer_config()
        };
        let mut runtime = JepaRuntime::default();

        let first = process_runtime_source_frame(&config, &mut runtime, &device)
            .expect("first source frame")
            .expect("synthetic source");
        assert_eq!(first.source, BevyJepaFrameSource::SyntheticLocalMotion);
        assert_eq!(first.sequence, 0);
        assert!(runtime.active_task.is_some());
        assert!(runtime.pending_stage.is_none());
        assert!(runtime.prev_stage_image.is_some());
        assert_eq!(runtime.input_frames_seen, 1);

        process_runtime_source_frame(&config, &mut runtime, &device)
            .expect("second source frame")
            .expect("synthetic source");
        assert!(runtime.active_task.is_some());
        assert_eq!(
            runtime
                .pending_stage
                .as_ref()
                .map(|pending| pending.id.sequence),
            Some(1)
        );
        assert_eq!(runtime.dropped_frames, 0);
        assert_eq!(runtime.overwritten_frames, 0);

        process_runtime_source_frame(&config, &mut runtime, &device)
            .expect("third source frame")
            .expect("synthetic source");
        assert_eq!(
            runtime
                .pending_stage
                .as_ref()
                .map(|pending| pending.id.sequence),
            Some(2)
        );
        assert_eq!(runtime.input_frames_seen, 3);
        assert_eq!(runtime.dropped_frames, 1);
        assert_eq!(runtime.overwritten_frames, 1);

        let mut metrics = BevyJepaMetrics::default();
        runtime.apply_runtime_counts(&mut metrics);
        assert_eq!(metrics.in_flight_frames, 2);
        assert_eq!(metrics.queue_dropped_frames, 1);
        assert_eq!(metrics.queue_overwritten_frames, 1);
        assert_eq!(metrics.input_frame_index, 2);
    }

    #[test]
    fn autogaze_mask_source_requires_real_model_node() {
        let device = JepaBevyDevice::default();
        let mut pipeline = BevyJepaHeadlessPipeline::new(
            BevyJepaConfig {
                source: BevyJepaFrameSource::SyntheticLocalMotion,
                mask_source: BevyJepaMaskSource::Autogaze,
                ..tiny_viewer_config()
            },
            device,
        );
        let err = pipeline
            .step_stage_only()
            .expect_err("fake AutoGaze masks must not run");
        assert!(
            err.to_string()
                .contains("loaded model-backed AutoGaze node")
        );
    }

    #[test]
    fn patch_diff_mask_selects_changed_camera_patch() {
        let device = JepaBevyDevice::default();
        let config = BevyJepaConfig {
            source: BevyJepaFrameSource::Camera,
            mask_source: BevyJepaMaskSource::PatchDiff,
            context_density: 1.0 / 16.0,
            patch_diff_threshold: 0.01,
            ..BevyJepaConfig::default()
        };
        let mut model_config = VJepaConfig::tiny_for_tests();
        model_config.image_size = 64;
        model_config.num_frames = 2;
        model_config.tubelet_size = 2;
        model_config.patch_size = 16;
        let grid = TokenGridShape::new(1, 4, 4);
        let previous_rgba = RgbaImage::new(64, 64);
        let current_rgba =
            rgba_with_patches(64, 64, &[(2, 1)], 16, image::Rgba([255, 255, 255, 255]));
        let previous = rgba_image_to_tensor(previous_rgba.clone(), 64, &device).expect("prev");
        let current = rgba_image_to_tensor(current_rgba.clone(), 64, &device).expect("current");

        let output = run_sparse_mask_node(
            &config,
            Some(&previous),
            Some(&previous_rgba),
            Some(&current_rgba),
            &current,
            &model_config,
            grid,
        )
        .expect("patch-diff mask");
        assert_eq!(output.mask.len(), 1);
        assert_eq!(output.mask.dense_len(), grid.len());
        assert_eq!(
            output.mask.indices(),
            &[coords_to_token_index(0, 2, 1, grid)]
        );
    }

    #[test]
    fn patch_diff_mask_includes_all_patches_above_threshold() {
        let device = JepaBevyDevice::default();
        let config = BevyJepaConfig {
            source: BevyJepaFrameSource::Camera,
            mask_source: BevyJepaMaskSource::PatchDiff,
            context_density: 1.0 / 16.0,
            patch_diff_threshold: 0.01,
            ..BevyJepaConfig::default()
        };
        let mut model_config = VJepaConfig::tiny_for_tests();
        model_config.image_size = 64;
        model_config.num_frames = 2;
        model_config.tubelet_size = 2;
        model_config.patch_size = 16;
        let grid = TokenGridShape::new(1, 4, 4);
        let changed = [(0, 0), (1, 3), (2, 1), (3, 2)];
        let previous_rgba = RgbaImage::new(64, 64);
        let current_rgba =
            rgba_with_patches(64, 64, &changed, 16, image::Rgba([255, 255, 255, 255]));
        let previous = rgba_image_to_tensor(previous_rgba.clone(), 64, &device).expect("prev");
        let current = rgba_image_to_tensor(current_rgba.clone(), 64, &device).expect("current");

        let output = run_sparse_mask_node(
            &config,
            Some(&previous),
            Some(&previous_rgba),
            Some(&current_rgba),
            &current,
            &model_config,
            grid,
        )
        .expect("patch-diff mask");

        assert_eq!(
            output.mask.len(),
            changed.len(),
            "adaptive patch-diff thresholding must not top-k cap changed patches"
        );
        assert_eq!(
            output.mask.indices(),
            &[
                coords_to_token_index(0, 0, 0, grid),
                coords_to_token_index(0, 1, 3, grid),
                coords_to_token_index(0, 2, 1, grid),
                coords_to_token_index(0, 3, 2, grid),
            ]
        );
    }

    #[test]
    fn patch_diff_mask_uses_adaptive_density_for_changed_patches() {
        let device = JepaBevyDevice::default();
        let config = BevyJepaConfig {
            source: BevyJepaFrameSource::Camera,
            mask_source: BevyJepaMaskSource::PatchDiff,
            context_density: 1.0,
            patch_diff_threshold: 0.01,
            ..BevyJepaConfig::default()
        };
        let mut model_config = VJepaConfig::tiny_for_tests();
        model_config.image_size = 64;
        model_config.num_frames = 2;
        model_config.tubelet_size = 2;
        model_config.patch_size = 16;
        let grid = TokenGridShape::new(1, 4, 4);
        let previous_rgba = RgbaImage::new(64, 64);
        let current_rgba = rgba_with_patches(
            64,
            64,
            &[(0, 0), (3, 2)],
            16,
            image::Rgba([255, 255, 255, 255]),
        );
        let previous = rgba_image_to_tensor(previous_rgba.clone(), 64, &device).expect("prev");
        let current = rgba_image_to_tensor(current_rgba.clone(), 64, &device).expect("current");

        let output = run_sparse_mask_node(
            &config,
            Some(&previous),
            Some(&previous_rgba),
            Some(&current_rgba),
            &current,
            &model_config,
            grid,
        )
        .expect("patch-diff mask");
        assert_eq!(output.mask.len(), 2);
        assert_eq!(
            output.mask.indices(),
            &[
                coords_to_token_index(0, 0, 0, grid),
                coords_to_token_index(0, 3, 2, grid),
            ]
        );
    }

    #[test]
    fn patch_diff_first_frame_bootstraps_dense_token_cache() {
        let device = JepaBevyDevice::default();
        let config = BevyJepaConfig {
            bootstrap_context_density: 1.0,
            ..BevyJepaConfig::default()
        };
        let mut model_config = VJepaConfig::tiny_for_tests();
        model_config.image_size = 64;
        model_config.num_frames = 2;
        model_config.tubelet_size = 2;
        model_config.patch_size = 16;
        let grid = TokenGridShape::new(1, 4, 4);
        let current = rgba_image_to_tensor(RgbaImage::new(64, 64), 64, &device).expect("current");

        let output = run_sparse_mask_node(&config, None, None, None, &current, &model_config, grid)
            .expect("bootstrap mask");
        assert_eq!(output.mask.len(), grid.len());
    }

    #[test]
    fn rgba_camera_frame_converts_to_model_sized_tensor() {
        let device = JepaBevyDevice::default();
        let frame = RgbaImage::from_pixel(4, 2, image::Rgba([128, 64, 32, 255]));
        let tensor = rgba_image_to_tensor(frame, 64, &device).expect("rgba tensor");
        assert_eq!(tensor.shape().dims::<4>(), [1, 3, 64, 64]);
    }

    #[test]
    fn rgba_camera_preprocess_center_crops_before_resizing() {
        let mut frame = RgbaImage::new(4, 2);
        for y in 0..2 {
            frame.put_pixel(0, y, image::Rgba([255, 0, 0, 255]));
            frame.put_pixel(1, y, image::Rgba([0, 255, 0, 255]));
            frame.put_pixel(2, y, image::Rgba([0, 0, 255, 255]));
            frame.put_pixel(3, y, image::Rgba([255, 0, 0, 255]));
        }

        let cropped = resize_source_rgba(frame, 2);

        assert_eq!(cropped.dimensions(), (2, 2));
        assert_eq!(*cropped.get_pixel(0, 0), image::Rgba([0, 255, 0, 255]));
        assert_eq!(*cropped.get_pixel(1, 0), image::Rgba([0, 0, 255, 255]));
        assert_eq!(*cropped.get_pixel(0, 1), image::Rgba([0, 255, 0, 255]));
        assert_eq!(*cropped.get_pixel(1, 1), image::Rgba([0, 0, 255, 255]));
    }

    #[test]
    fn frame_source_parses_camera_aliases() {
        assert_eq!(
            "webcam".parse::<BevyJepaFrameSource>().expect("webcam"),
            BevyJepaFrameSource::Camera
        );
        assert_eq!(
            "image".parse::<BevyJepaFrameSource>().expect("image"),
            BevyJepaFrameSource::StaticImage
        );
    }

    #[test]
    fn metrics_overlay_line_uses_stable_field_widths() {
        let config = BevyJepaConfig {
            source: BevyJepaFrameSource::SyntheticLocalMotion,
            ..tiny_viewer_config()
        };
        let mut first = BevyJepaMetrics {
            frame_ready: true,
            frame_source: BevyJepaFrameSource::SyntheticLocalMotion,
            mask_source: BevyJepaMaskSource::PatchDiff,
            context_tokens: 1,
            dense_tokens: 16,
            viewer_total_us: 9_000,
            total_us: 8_000,
            ..BevyJepaMetrics::default()
        };
        let mut second = first.clone();
        second.context_tokens = 16;
        second.viewer_total_us = 123_450;
        second.total_us = 98_760;
        second.anyup_decode_us = 12_345;
        second.pca_sample_frames = 16;
        second.pca_sample_window_frames = 16;
        second.pca_update_applied = true;

        assert_eq!(
            format_metrics_line(&config, &first).len(),
            format_metrics_line(&config, &second).len()
        );
        first.frame_ready = false;
        assert_eq!(
            format_metrics_waiting_line().len(),
            format_metrics_line(&config, &first).len()
        );
    }

    #[test]
    fn headless_metrics_align_with_raw_stage_metrics() {
        let device = JepaBevyDevice::default();
        let mut pipeline = BevyJepaHeadlessPipeline::new(
            BevyJepaConfig {
                source: BevyJepaFrameSource::SyntheticLocalMotion,
                ..tiny_viewer_config()
            },
            device,
        );

        let core = pipeline.step_stage_only().expect("stage-only viewer step");
        assert!(core.metrics.aligns_with_stage_metrics());
        assert_eq!(core.metrics.display_tensor_us, 0);
        assert_eq!(
            core.metrics.context_tokens,
            core.metrics.stage_metrics.sparse_width
        );
        assert_eq!(
            core.metrics.dense_tokens,
            core.metrics.stage_metrics.dense_tokens_per_frame
        );
        assert_eq!(core.metrics.grid_height, DEFAULT_IMAGE_SIZE / 16);
        assert_eq!(core.metrics.grid_width, DEFAULT_IMAGE_SIZE / 16);
        assert_eq!(core.metrics.patch_size, 16);
        assert_eq!(core.metrics.dense_tokens, 256);
        assert_eq!(core.metrics.encoder_source, BevyJepaEncoderSource::TinyTest);

        let display = pipeline
            .step_with_display_panels()
            .expect("display viewer step");
        assert!(display.metrics.aligns_with_stage_metrics());
        assert_eq!(
            display.metrics.display_transfer,
            BevyJepaDisplayTransfer::Gpu
        );
        assert!(display.metrics.viewer_total_us >= display.metrics.total_us);
        assert!(display.metrics.viewer_total_us >= display.metrics.display_tensor_us);
    }

    #[test]
    fn headless_stage_request_can_measure_low_res_only_path() {
        let device = JepaBevyDevice::default();
        let mut pipeline = BevyJepaHeadlessPipeline::new(
            BevyJepaConfig {
                source: BevyJepaFrameSource::SyntheticLocalMotion,
                ..tiny_viewer_config()
            },
            device,
        );

        let output = pipeline
            .step_with_stage_request(FeatureFrameRequest::low_res())
            .expect("low-res-only viewer step");

        assert!(output.metrics.aligns_with_stage_metrics());
        assert_eq!(output.metrics.anyup_decode_us, 0);
        assert_eq!(output.metrics.high_res_pca_us, 0);
        assert!(output.metrics.low_res_pca_us > 0 || !output.metrics.stage_metrics.measured);
    }

    #[test]
    fn highres_pipeline_can_run_ttt_encoder_branch() {
        let device = JepaBevyDevice::default();
        let model_config = tiny_viewer_model_config(32);
        let base = VJepa2_1Model::<JepaBevyBackend>::new(&model_config, &device);
        let ttt = VJepaTttModel::from_model(base, TttEncoderConfig::default(), &device)
            .expect("TTT model");
        let mut anyup_config = AnyUpConfig::tiny_for_tests();
        anyup_config.input_dim = 3;
        let anyup = AnyUp::<JepaBevyBackend>::new(anyup_config, &device).expect("AnyUp");
        let mut pipeline = FeatureFramePipeline::<JepaBevyBackend>::new_with_encoder(
            FeatureFrameJepaEncoder::ttt(ttt),
            anyup,
            &model_config,
            FeatureFramePipelineConfig::default(),
            1,
            [32, 32],
            &device,
        )
        .expect("TTT feature-frame pipeline");
        assert_eq!(pipeline.encoder_kind(), FeatureFrameJepaEncoderKind::Ttt);

        let image = synthetic_image_tensor(0, 32, &device);
        let mask = SparseTokenMask::all(pipeline.grid().len());
        let output = pipeline
            .step_image_with_mask_nodes_measured(image, &mask, FeatureFrameRequest::low_res())
            .expect("TTT pipeline step");
        assert_eq!(output.output.encoded.grid, pipeline.grid());
        assert_eq!(
            output.output.encoded.tokens.shape().dims::<3>()[1],
            pipeline.grid().len()
        );
    }

    #[cfg(feature = "sparse-patchify-wgpu")]
    #[test]
    fn highres_pipeline_can_run_ttt_sparse_patchify_branch() {
        let device = JepaBevyDevice::default();
        let model_config = tiny_viewer_model_config(32);
        let base = VJepa2_1Model::<JepaBevyBackend>::new(&model_config, &device);
        let ttt = VJepaTttModel::from_model(base, TttEncoderConfig::default(), &device)
            .expect("TTT model");
        let mut anyup_config = AnyUpConfig::tiny_for_tests();
        anyup_config.input_dim = 3;
        let anyup = AnyUp::<JepaBevyBackend>::new(anyup_config, &device).expect("AnyUp");
        let mut pipeline = FeatureFramePipeline::<JepaBevyBackend>::new_with_encoder(
            FeatureFrameJepaEncoder::ttt(ttt),
            anyup,
            &model_config,
            FeatureFramePipelineConfig::default(),
            1,
            [32, 32],
            &device,
        )
        .expect("TTT feature-frame pipeline");

        let image = synthetic_image_tensor(0, 32, &device);
        let mask = SparseTokenMask::all(pipeline.grid().len());
        let batch_mask = SparseMaskBatch::uniform(mask, 1, pipeline.device()).expect("mask batch");
        let patchify_plan =
            SparsePatchifyBatchPlan::new(batch_mask, pipeline.grid(), pipeline.device())
                .expect("patchify plan");
        let output = pipeline
            .step_image_with_sparse_patchify_plan_wgpu_nodes_measured(
                image,
                &patchify_plan,
                FeatureFrameRequest::low_res(),
                pipeline.config().measurement,
            )
            .expect("TTT sparse patchify pipeline step");
        assert_eq!(
            output.metrics.encode_path,
            FeatureFrameEncodePath::SparsePatchify
        );
        assert_eq!(output.output.encoded.grid, pipeline.grid());
        assert_eq!(
            output.output.encoded.tokens.shape().dims::<3>()[1],
            pipeline.grid().len()
        );
    }

    #[test]
    #[ignore = "loads the local 433 MiB production TTT checkpoint"]
    fn default_trained_ttt_artifact_initializes_viewer_encoder() {
        let config = BevyJepaConfig {
            source: BevyJepaFrameSource::SyntheticLocalMotion,
            image_size: MIN_PIPELINE_IMAGE_SIZE,
            ..BevyJepaConfig::default()
        };
        let path = effective_ttt_model_path(&config).expect("default trained TTT path");
        if !path.exists() {
            eprintln!(
                "skipping: trained TTT checkpoint is missing at {}",
                path.display()
            );
            return;
        }
        let device = JepaBevyDevice::default();
        let (encoder, model_config) =
            load_viewer_encoder(&config, config.pipeline_image_size(), &device)
                .expect("load default trained TTT encoder");
        assert_eq!(encoder.kind(), FeatureFrameJepaEncoderKind::Ttt);
        assert_eq!(model_config.model_type, "vjepa2_1");
        assert_eq!(model_config.encoder.embed_dim, 768);
    }

    #[test]
    #[ignore = "loads the local production TTT checkpoint and runs a WebGPU forward step"]
    fn default_trained_ttt_pipeline_runs_core_step() {
        let config = BevyJepaConfig {
            source: BevyJepaFrameSource::SyntheticLocalMotion,
            image_size: MIN_PIPELINE_IMAGE_SIZE,
            ..BevyJepaConfig::default()
        };
        let path = effective_ttt_model_path(&config).expect("default trained TTT path");
        if !path.exists() {
            eprintln!(
                "skipping: trained TTT checkpoint is missing at {}",
                path.display()
            );
            return;
        }
        let device = JepaBevyDevice::default();
        let mut pipeline = BevyJepaHeadlessPipeline::new(config, device);
        let output = pipeline
            .step_stage_only()
            .expect("trained TTT viewer stage-only step");
        assert_eq!(
            output.metrics.encoder_source,
            BevyJepaEncoderSource::TrainedTtt
        );
        assert_eq!(output.metrics.grid_height, MIN_PIPELINE_IMAGE_SIZE / 16);
        assert_eq!(output.metrics.grid_width, MIN_PIPELINE_IMAGE_SIZE / 16);
        assert_eq!(output.metrics.dense_tokens, 256);
    }

    fn rgba_with_patches(
        width: u32,
        height: u32,
        patches: &[(usize, usize)],
        patch_size: usize,
        color: image::Rgba<u8>,
    ) -> RgbaImage {
        let mut image = RgbaImage::new(width, height);
        for &(patch_row, patch_col) in patches {
            let row_start = patch_row * patch_size;
            let col_start = patch_col * patch_size;
            for y in row_start..(row_start + patch_size).min(height as usize) {
                for x in col_start..(col_start + patch_size).min(width as usize) {
                    image.put_pixel(x as u32, y as u32, color);
                }
            }
        }
        image
    }
}
