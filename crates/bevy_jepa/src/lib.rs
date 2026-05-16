#![recursion_limit = "512"]

use std::time::Instant;

use anyhow::{Context, Result};
use bevy::{
    app::AppExit,
    prelude::*,
    render::{
        RenderPlugin,
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
    },
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
use burn_jepa::{
    AnyUp, AnyUpConfig, FeatureFrameBatch, FeatureFrameMetrics, FeatureFramePipeline,
    FeatureFramePipelineConfig, FeatureFrameRequest, FeaturePcaUpdateConfig, FrameId,
    HighResFrameArtifacts, LowResFrameArtifacts, SparseJepaPatchDiffSparsityConfig,
    SparseTokenMask, TokenGridShape, VJepa2_1Model, VJepaConfig, coords_to_token_index,
    patch_diff_context_mask_from_video,
};

mod config;
mod display;

pub use config::{
    BevyJepaConfig, BevyJepaDisplayTransfer, BevyJepaMaskSource, DEFAULT_ANYUP_CHUNK_SIZE,
    DEFAULT_CONTEXT_DENSITY, DEFAULT_IMAGE_SIZE, DEFAULT_PATCH_DIFF_THRESHOLD,
    DEFAULT_PCA_UPDATE_EVERY,
};
use display::{
    JepaHostPanels, JepaPanelTextures, JepaTensorPanels, PanelImageData, apply_panels_to_world,
    clear_completed_gpu_uploads,
};

pub type JepaBevyBackend = burn::backend::WebGpu<f32, i32>;
pub type JepaBevyDevice = burn::backend::wgpu::WgpuDevice;

const UI_MARGIN_PX: f32 = 12.0;
const METRIC_ROW_HEIGHT: f32 = 24.0;
const PANEL_LABEL_ROW_HEIGHT: f32 = 34.0;

#[derive(Resource, Default)]
struct JepaRuntime {
    pipeline: Option<FeatureFramePipeline<JepaBevyBackend>>,
    model_config: Option<VJepaConfig>,
    prev_image: Option<Tensor<JepaBevyBackend, 4>>,
    frame_index: u64,
    last_error: Option<String>,
}

#[derive(Resource, Clone, Debug, Default)]
pub struct BevyJepaMetrics {
    pub frame_index: u64,
    pub mask_source: BevyJepaMaskSource,
    pub display_transfer: BevyJepaDisplayTransfer,
    pub context_tokens: usize,
    pub dense_tokens: usize,
    pub stage_metrics: FeatureFrameMetrics,
    pub encode_us: u64,
    pub cache_update_us: u64,
    pub token_view_us: u64,
    pub anyup_context_us: u64,
    pub anyup_decode_us: u64,
    pub low_res_pca_us: u64,
    pub pca_update_us: u64,
    pub high_res_pca_us: u64,
    pub display_tensor_us: u64,
    pub total_us: u64,
    pub viewer_total_us: u64,
    pub pca_update_applied: bool,
    pub queue_dropped_frames: usize,
    pub queue_overwritten_frames: usize,
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
            && self.high_res_pca_us == self.stage_metrics.pca_project_us
            && self.total_us == self.stage_metrics.total_us
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
    CoreOnly,
    DisplayPanels,
}

impl BevyJepaHeadlessPipeline {
    pub fn new(config: BevyJepaConfig, device: JepaBevyDevice) -> Self {
        Self {
            config,
            runtime: JepaRuntime::default(),
            device,
        }
    }

    pub fn step_core_only(&mut self) -> Result<BevyJepaStepOutput> {
        self.step_internal(BevyJepaStepMode::CoreOnly)
    }

    pub fn step_with_display_panels(&mut self) -> Result<BevyJepaStepOutput> {
        self.step_internal(BevyJepaStepMode::DisplayPanels)
    }

    fn step_internal(&mut self, mode: BevyJepaStepMode) -> Result<BevyJepaStepOutput> {
        let processed = process_runtime_frame(&self.config, &mut self.runtime, &self.device, mode)?;
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
    viewer_app(config).run()
}

pub fn run_once() -> Result<BevyJepaMetrics> {
    let device = JepaBevyDevice::default();
    let config = BevyJepaConfig::default();
    let mut runtime = JepaRuntime::default();
    Ok(process_runtime_frame(
        &config,
        &mut runtime,
        &device,
        BevyJepaStepMode::DisplayPanels,
    )?
    .metrics)
}

impl JepaRuntime {
    fn ensure_pipeline(
        &mut self,
        config: &BevyJepaConfig,
        device: &JepaBevyDevice,
    ) -> Result<(&mut FeatureFramePipeline<JepaBevyBackend>, VJepaConfig)> {
        let image_size = normalized_image_size(config.image_size);
        let needs_init = self
            .model_config
            .as_ref()
            .is_none_or(|model| model.image_size != image_size);
        if needs_init {
            let mut model_config = VJepaConfig::tiny_for_tests();
            model_config.image_size = image_size;
            model_config.num_frames = 2;
            model_config.tubelet_size = 2;
            let jepa = VJepa2_1Model::<JepaBevyBackend>::new(&model_config, device);
            let anyup = AnyUp::<JepaBevyBackend>::new(AnyUpConfig::tiny_for_tests(), device)
                .context("initialize tiny AnyUp viewer model")?;
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
            self.pipeline = Some(FeatureFramePipeline::<JepaBevyBackend>::new(
                jepa,
                anyup,
                &model_config,
                pipeline_config,
                1,
                [image_size, image_size],
                device,
            )?);
            self.model_config = Some(model_config);
            self.prev_image = None;
            self.frame_index = 0;
        }
        let model_config = self
            .model_config
            .clone()
            .expect("model config initialized with pipeline");
        let pipeline = self.pipeline.as_mut().expect("pipeline initialized");
        Ok((pipeline, model_config))
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
                font_size: bevy::text::FontSize::Px(18.0),
                ..default()
            },
            TextColor(Color::WHITE),
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(UI_MARGIN_PX),
                left: Val::Px(UI_MARGIN_PX),
                ..default()
            },
            ZIndex(2),
        ))
        .with_child((
            MetricsText,
            TextColor(Color::srgb(1.0, 0.84, 0.0)),
            TextFont {
                font_size: bevy::text::FontSize::Px(18.0),
                ..default()
            },
            TextSpan::new("waiting for WebGPU"),
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
    let result = {
        let mut runtime = world.resource_mut::<JepaRuntime>();
        process_runtime_frame(
            &config,
            &mut runtime,
            &device,
            BevyJepaStepMode::DisplayPanels,
        )
    };

    match result {
        Ok(processed) => {
            world.resource_mut::<JepaRuntime>().last_error = None;
            *world.resource_mut::<BevyJepaMetrics>() = processed.metrics.clone();
            apply_panels_to_world(world, processed.panels, transfer);
        }
        Err(err) => {
            world.resource_mut::<JepaRuntime>().last_error = Some(err.to_string());
            world.resource_mut::<BevyJepaMetrics>().last_error = Some(err.to_string());
        }
    }
}

fn process_runtime_frame(
    config: &BevyJepaConfig,
    runtime: &mut JepaRuntime,
    device: &JepaBevyDevice,
    mode: BevyJepaStepMode,
) -> Result<ProcessedFrame> {
    let frame_index = runtime.frame_index;
    let image_size = normalized_image_size(config.image_size);
    let image = synthetic_image_tensor(frame_index, image_size, device);
    let prev_image = runtime.prev_image.clone();
    let (pipeline, model_config) = runtime.ensure_pipeline(config, device)?;
    let grid = pipeline.grid();
    let context_tokens = config.context_tokens(grid.len());
    let mask = match config.mask_source {
        BevyJepaMaskSource::Autogaze => autogaze_like_mask(frame_index, grid, context_tokens),
        BevyJepaMaskSource::PatchDiff => {
            patch_diff_mask(prev_image.as_ref(), &image, &model_config, grid, config)
        }
    }?;
    let id = FrameId {
        stream_id: 0,
        sequence: frame_index,
        capture_time_nanos: frame_index.saturating_mul(16_666_667),
    };
    let processed = match mode {
        BevyJepaStepMode::CoreOnly => {
            run_core_step_metrics(config, pipeline, image.clone(), &mask, id)
        }
        BevyJepaStepMode::DisplayPanels => {
            run_pipeline_step(config, pipeline, image.clone(), &mask, id)
        }
    };
    runtime.prev_image = Some(image);
    runtime.frame_index += 1;
    processed
}

struct ProcessedFrame {
    metrics: BevyJepaMetrics,
    panels: PanelImageData,
}

struct CoreFrame {
    output: FeatureFrameBatch<JepaBevyBackend>,
    metrics: FeatureFrameMetrics,
    wall_us: u64,
}

#[derive(Clone, Copy)]
struct MetricFrameContext {
    frame_index: u64,
    mask_source: BevyJepaMaskSource,
    display_transfer: BevyJepaDisplayTransfer,
    context_tokens: usize,
    dense_tokens: usize,
}

fn bevy_metrics_from_stage(
    frame: MetricFrameContext,
    stage_metrics: FeatureFrameMetrics,
    display_tensor_us: u64,
    viewer_total_us: u64,
) -> BevyJepaMetrics {
    BevyJepaMetrics {
        frame_index: frame.frame_index,
        mask_source: frame.mask_source,
        display_transfer: frame.display_transfer,
        context_tokens: frame.context_tokens,
        dense_tokens: frame.dense_tokens,
        encode_us: stage_metrics.encode_us,
        cache_update_us: stage_metrics.cache_update_us,
        token_view_us: stage_metrics.token_view_us,
        anyup_context_us: stage_metrics.anyup_context_us,
        anyup_decode_us: stage_metrics.anyup_decode_us,
        low_res_pca_us: stage_metrics.low_res_pca_project_us,
        pca_update_us: stage_metrics.pca_update_us,
        high_res_pca_us: stage_metrics.pca_project_us,
        display_tensor_us,
        total_us: stage_metrics.total_us,
        viewer_total_us,
        pca_update_applied: stage_metrics.pca_update_applied,
        queue_dropped_frames: 0,
        queue_overwritten_frames: 0,
        last_error: None,
        stage_metrics,
    }
}

fn run_core_step(
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    mask: &SparseTokenMask,
) -> Result<CoreFrame> {
    let wall_start = Instant::now();
    let measured = pipeline.step_image_with_mask_nodes_measured(
        image,
        mask,
        FeatureFrameRequest::full_pca(),
    )?;
    Ok(CoreFrame {
        output: measured.output,
        metrics: measured.metrics,
        wall_us: micros_u64(wall_start.elapsed().as_micros()),
    })
}

fn run_core_step_metrics(
    config: &BevyJepaConfig,
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    mask: &SparseTokenMask,
    id: FrameId,
) -> Result<ProcessedFrame> {
    let core = run_core_step(pipeline, image, mask)?;
    let frame = MetricFrameContext {
        frame_index: id.sequence,
        mask_source: config.mask_source,
        display_transfer: config.display_transfer,
        context_tokens: mask.len(),
        dense_tokens: mask.dense_len(),
    };
    let metrics = bevy_metrics_from_stage(frame, core.metrics, 0, core.wall_us);
    Ok(ProcessedFrame {
        metrics,
        panels: empty_panel_data(),
    })
}

fn run_pipeline_step(
    config: &BevyJepaConfig,
    pipeline: &mut FeatureFramePipeline<JepaBevyBackend>,
    image: Tensor<JepaBevyBackend, 4>,
    mask: &SparseTokenMask,
    id: FrameId,
) -> Result<ProcessedFrame> {
    let core = run_core_step(pipeline, image.clone(), mask)?;
    let metrics = core.metrics;
    let output = core.output;
    let image_size = pipeline.image_size();
    let display_start = Instant::now();
    let low_res_pca = low_res_pca_or_features(output.low_res, image_size)?;
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
    if config.sync_measurements {
        JepaBevyBackend::sync(&input_rgba.device())?;
    }
    let display_tensor_us = micros_u64(display_start.elapsed().as_micros());
    let viewer_total_us = core.wall_us.saturating_add(display_tensor_us);
    let frame = MetricFrameContext {
        frame_index: id.sequence,
        mask_source: config.mask_source,
        context_tokens: mask.len(),
        dense_tokens: mask.dense_len(),
        display_transfer: config.display_transfer,
    };
    let latest = bevy_metrics_from_stage(frame, metrics, display_tensor_us, viewer_total_us);
    let panels = match config.display_transfer {
        BevyJepaDisplayTransfer::Gpu => PanelImageData::TensorPanels(Box::new(JepaTensorPanels {
            width: image_size[1] as u32,
            height: image_size[0] as u32,
            input_rgba,
            mask_rgba,
            low_res_rgba,
            high_res_rgba,
        })),
        BevyJepaDisplayTransfer::Cpu => PanelImageData::HostPanels(Box::new(JepaHostPanels {
            width: image_size[1] as u32,
            height: image_size[0] as u32,
            input_rgba: tensor_rgba_to_host(input_rgba)?,
            mask_rgba: tensor_rgba_to_host(mask_rgba)?,
            low_res_rgba: tensor_rgba_to_host(low_res_rgba)?,
            high_res_rgba: tensor_rgba_to_host(high_res_rgba)?,
        })),
    };
    Ok(ProcessedFrame {
        metrics: latest,
        panels,
    })
}

fn empty_panel_data() -> PanelImageData {
    PanelImageData::HostPanels(Box::new(JepaHostPanels {
        width: 1,
        height: 1,
        input_rgba: vec![0, 0, 0, 255],
        mask_rgba: vec![0, 0, 0, 255],
        low_res_rgba: vec![0, 0, 0, 255],
        high_res_rgba: vec![0, 0, 0, 255],
    }))
}

fn low_res_pca_or_features(
    low_res: LowResFrameArtifacts<JepaBevyBackend>,
    image_size: [usize; 2],
) -> Result<Tensor<JepaBevyBackend, 4>> {
    if let Some(pca) = low_res.pca_display {
        return Ok(pca);
    }
    Ok(low_res
        .features
        .slice_dim(1, 0..3)
        .reshape([1, 3, image_size[0] / 16, image_size[1] / 16]))
}

fn high_res_pca_or_low_res(
    high_res: Option<HighResFrameArtifacts<JepaBevyBackend>>,
    low_res_pca: Tensor<JepaBevyBackend, 4>,
) -> Result<Tensor<JepaBevyBackend, 4>> {
    Ok(high_res
        .and_then(|high_res| high_res.pca_display)
        .unwrap_or(low_res_pca))
}

fn patch_diff_mask(
    prev_image: Option<&Tensor<JepaBevyBackend, 4>>,
    image: &Tensor<JepaBevyBackend, 4>,
    model_config: &VJepaConfig,
    grid: TokenGridShape,
    config: &BevyJepaConfig,
) -> Result<SparseTokenMask> {
    let Some(prev_image) = prev_image else {
        return autogaze_like_mask(0, grid, config.context_tokens(grid.len()));
    };
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
    let context_tokens = config.context_tokens(grid.len());
    let target_tokens = grid.len().saturating_sub(context_tokens).max(1);
    let config = SparseJepaPatchDiffSparsityConfig::new(
        config.patch_diff_threshold,
        context_tokens,
        target_tokens,
    );
    patch_diff_context_mask_from_video(&video, model_config, grid, &config)
}

fn autogaze_like_mask(
    frame_index: u64,
    grid: TokenGridShape,
    context_tokens: usize,
) -> Result<SparseTokenMask> {
    let phase = frame_index as f32 * 0.045;
    let center_row = ((phase.sin() * 0.38 + 0.5) * grid.height.saturating_sub(1) as f32).round();
    let center_col = ((phase.cos() * 0.38 + 0.5) * grid.width.saturating_sub(1) as f32).round();
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
        format!("runtime error: {error}")
    } else {
        format!(
            "{} | {:.1}% sparse | {:.1} fps | viewer {:.2} ms | core {:.2} | display {:.2} | encode {:.2} | cache {:.2} | anyup {:.2}+{:.2} | pca {:.2}+{:.2} | pca_update {}",
            metrics.mask_source,
            metrics.density() * 100.0,
            metrics.fps(),
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
        )
    };
    for mut span in &mut query {
        **span = text.clone();
    }
}

fn keyboard_controls(
    mut config: ResMut<BevyJepaConfig>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut runtime: ResMut<JepaRuntime>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        config.mask_source = config.mask_source.next();
        runtime.prev_image = None;
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

fn normalized_image_size(size: usize) -> usize {
    let size = size.max(32);
    size.div_ceil(16) * 16
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

    #[test]
    fn autogaze_like_mask_keeps_requested_density() {
        let grid = TokenGridShape::new(1, 4, 4);
        let mask = autogaze_like_mask(7, grid, 5).expect("mask");
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
    fn headless_metrics_align_with_raw_stage_metrics() {
        let device = JepaBevyDevice::default();
        let mut pipeline = BevyJepaHeadlessPipeline::new(BevyJepaConfig::default(), device);

        let core = pipeline.step_core_only().expect("core viewer step");
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
}
