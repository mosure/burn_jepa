use anyhow::{Context, Result};
#[cfg(not(target_arch = "wasm32"))]
use burn::prelude::Backend;
use burn_jepa::{
    AnyUp, AnyUpConfig, FeatureFrameMeasureConfig, FeatureFrameMetrics, FeatureFramePipeline,
    FeatureFramePipelineConfig, FeatureFrameRequest, SparseTokenMask, VJepa2_1Model, VJepaConfig,
};
use serde::{Deserialize, Serialize};

use crate::{JepaBevyBackend, JepaBevyDevice, synthetic_image_tensor};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PipelinePerfConfig {
    pub image_sizes: Vec<usize>,
    pub densities: Vec<f32>,
    pub warmups: usize,
    pub reps: usize,
}

impl Default for PipelinePerfConfig {
    fn default() -> Self {
        Self {
            image_sizes: vec![256, 512],
            densities: vec![0.10, 0.25, 0.50, 1.0],
            warmups: 4,
            reps: 16,
        }
    }
}

impl PipelinePerfConfig {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            !self.image_sizes.is_empty(),
            "image_sizes must not be empty"
        );
        anyhow::ensure!(!self.densities.is_empty(), "densities must not be empty");
        anyhow::ensure!(self.reps > 0, "reps must be nonzero");
        for &image_size in &self.image_sizes {
            anyhow::ensure!(image_size > 0, "image_size must be nonzero");
            anyhow::ensure!(
                image_size.is_multiple_of(16),
                "image_size {image_size} must be divisible by the tiny test patch size"
            );
        }
        for &density in &self.densities {
            anyhow::ensure!(
                density.is_finite() && density > 0.0 && density <= 1.0,
                "density {density} must be in (0, 1]"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelinePerfReport {
    pub runtime: String,
    pub backend: String,
    pub fusion: bool,
    pub flush: String,
    pub config: PipelinePerfConfig,
    pub rows: Vec<PipelinePerfRow>,
}

impl PipelinePerfReport {
    pub fn markdown(&self) -> String {
        let mut text = String::new();
        text.push_str(
            "| runtime | image | density | tokens | encode path | encode ms | cache ms | wall ms | p95 wall ms | encoder Mtok/s | cache Mtok/s | step fps |\n",
        );
        text.push_str("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|\n");
        for row in &self.rows {
            text.push_str(&format!(
                "| {} | {} | {:.0}% | {}/{} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.2} | {:.2} | {:.1} |\n",
                self.runtime,
                row.image_size,
                row.density * 100.0,
                row.context_tokens,
                row.dense_tokens,
                row.encode_path,
                row.encode_mean_us / 1000.0,
                row.cache_update_mean_us / 1000.0,
                row.wall_mean_us / 1000.0,
                row.wall_p95_us as f64 / 1000.0,
                row.encoder_mtokens_per_sec,
                row.cache_mtokens_per_sec,
                row.step_fps,
            ));
        }
        text
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelinePerfRow {
    pub image_size: usize,
    pub grid_height: usize,
    pub grid_width: usize,
    pub dense_tokens: usize,
    pub context_tokens: usize,
    pub density: f32,
    pub encode_path: String,
    pub encode_mean_us: f64,
    pub encode_p50_us: u64,
    pub encode_p95_us: u64,
    pub cache_update_mean_us: f64,
    pub cache_update_p50_us: u64,
    pub cache_update_p95_us: u64,
    pub total_mean_us: f64,
    pub total_p50_us: u64,
    pub total_p95_us: u64,
    pub wall_mean_us: f64,
    pub wall_p50_us: u64,
    pub wall_p95_us: u64,
    pub encoder_mtokens_per_sec: f64,
    pub cache_mtokens_per_sec: f64,
    pub step_fps: f64,
}

#[derive(Clone, Debug, Default)]
struct MetricSamples {
    encode_us: Vec<u64>,
    cache_update_us: Vec<u64>,
    total_us: Vec<u64>,
    wall_us: Vec<u64>,
    last_metrics: Option<FeatureFrameMetrics>,
}

impl MetricSamples {
    fn push(&mut self, metrics: FeatureFrameMetrics, wall_us: u64) {
        self.encode_us.push(metrics.encode_us);
        self.cache_update_us.push(metrics.cache_update_us);
        self.total_us.push(metrics.total_us);
        self.wall_us.push(wall_us);
        self.last_metrics = Some(metrics);
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_native_perf_matrix(config: PipelinePerfConfig) -> Result<PipelinePerfReport> {
    config.validate()?;
    let device = JepaBevyDevice::default();
    let rows = run_perf_rows_sync(&config, &device)?;
    Ok(PipelinePerfReport {
        runtime: "native-wgpu".to_string(),
        backend: "burn-webgpu".to_string(),
        fusion: burn_fusion_enabled(),
        flush: "backend-sync".to_string(),
        config,
        rows,
    })
}

#[cfg(target_arch = "wasm32")]
pub async fn run_wasm_perf_matrix(config: PipelinePerfConfig) -> Result<PipelinePerfReport> {
    config.validate()?;
    let device = init_wasm_webgpu_device().await;
    let rows = run_perf_rows_async(&config, &device).await?;
    Ok(PipelinePerfReport {
        runtime: "wasm-webgpu".to_string(),
        backend: "burn-webgpu".to_string(),
        fusion: burn_fusion_enabled(),
        flush: "1-scalar-async-readback".to_string(),
        config,
        rows,
    })
}

fn burn_fusion_enabled() -> bool {
    cfg!(any(not(target_arch = "wasm32"), feature = "wasm-fusion"))
}

#[cfg(target_arch = "wasm32")]
async fn init_wasm_webgpu_device() -> JepaBevyDevice {
    use burn::backend::wgpu::{WgpuDevice, graphics::WebGpu, init_device, init_setup_async};

    let requested = WgpuDevice::default();
    let setup = init_setup_async::<WebGpu>(&requested, wasm_runtime_options()).await;
    init_device(setup, wasm_runtime_options())
}

#[cfg(target_arch = "wasm32")]
fn wasm_runtime_options() -> burn::backend::wgpu::RuntimeOptions {
    let mut options = burn::backend::wgpu::RuntimeOptions::default();
    if burn_fusion_enabled() {
        options.memory_config = burn::backend::wgpu::MemoryConfiguration::ExclusivePages;
    }
    options
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn benchmark_tiny_pipeline_json(warmups: usize, reps: usize) -> js_sys::Promise {
    wasm_bindgen_futures::future_to_promise(async move {
        let config = PipelinePerfConfig {
            warmups,
            reps,
            ..PipelinePerfConfig::default()
        };
        match run_wasm_perf_matrix(config).await {
            Ok(report) => serde_json::to_string(&report)
                .map(|json| wasm_bindgen::JsValue::from_str(&json))
                .map_err(|err| wasm_bindgen::JsValue::from_str(&err.to_string())),
            Err(err) => Err(wasm_bindgen::JsValue::from_str(&format!("{err:?}"))),
        }
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn run_perf_rows_sync(
    config: &PipelinePerfConfig,
    device: &JepaBevyDevice,
) -> Result<Vec<PipelinePerfRow>> {
    let mut rows = Vec::new();
    for &image_size in &config.image_sizes {
        let mut pipeline = new_tiny_pipeline(image_size, true, device)?;
        let dense_tokens = pipeline.grid().len();
        for &density in &config.densities {
            let mask = mask_for_density(dense_tokens, density);
            let mut samples = MetricSamples::default();
            for iteration in 0..(config.warmups + config.reps) {
                let image = synthetic_image_tensor(iteration as u64, image_size, device);
                let started = std::time::Instant::now();
                let measured = pipeline
                    .step_image_with_mask_nodes_measured(image, &mask, FeatureFrameRequest::none())
                    .context("run native perf pipeline step")?;
                JepaBevyBackend::sync(device).context("sync native perf backend")?;
                let wall_us = micros_u64(started.elapsed().as_micros());
                if iteration >= config.warmups {
                    samples.push(measured.metrics, wall_us);
                }
            }
            rows.push(row_from_samples(image_size, dense_tokens, &samples)?);
        }
    }
    Ok(rows)
}

#[cfg(target_arch = "wasm32")]
async fn run_perf_rows_async(
    config: &PipelinePerfConfig,
    device: &JepaBevyDevice,
) -> Result<Vec<PipelinePerfRow>> {
    let mut rows = Vec::new();
    for &image_size in &config.image_sizes {
        let mut pipeline = new_tiny_pipeline(image_size, false, device)?;
        let dense_tokens = pipeline.grid().len();
        for &density in &config.densities {
            let mask = mask_for_density(dense_tokens, density);
            let mut samples = MetricSamples::default();
            for iteration in 0..(config.warmups + config.reps) {
                let image = synthetic_image_tensor(iteration as u64, image_size, device);
                let started = wasm_now();
                let measured = pipeline
                    .step_image_with_mask_nodes_measured(image, &mask, FeatureFrameRequest::none())
                    .context("run wasm perf pipeline step")?;
                flush_measured_output(&measured).await?;
                let wall_us = micros_u64(((wasm_now() - started) * 1000.0).max(0.0) as u128);
                if iteration >= config.warmups {
                    samples.push(measured.metrics, wall_us);
                }
            }
            rows.push(row_from_samples(image_size, dense_tokens, &samples)?);
        }
    }
    Ok(rows)
}

fn new_tiny_pipeline(
    image_size: usize,
    sync_backend: bool,
    device: &JepaBevyDevice,
) -> Result<FeatureFramePipeline<JepaBevyBackend>> {
    let mut model_config = VJepaConfig::tiny_for_tests();
    model_config.image_size = image_size;
    model_config.num_frames = 2;
    model_config.tubelet_size = 2;
    let jepa = VJepa2_1Model::<JepaBevyBackend>::new(&model_config, device);
    let mut anyup_config = AnyUpConfig::tiny_for_tests();
    anyup_config.input_dim = 3;
    let anyup =
        AnyUp::<JepaBevyBackend>::new(anyup_config, device).context("initialize tiny AnyUp")?;
    FeatureFramePipeline::<JepaBevyBackend>::new(
        jepa,
        anyup,
        &model_config,
        FeatureFramePipelineConfig {
            measurement: FeatureFrameMeasureConfig {
                enabled: true,
                sync_backend,
            },
            ..FeatureFramePipelineConfig::default()
        },
        1,
        [image_size, image_size],
        device,
    )
    .context("initialize tiny feature-frame pipeline")
}

fn mask_for_density(dense_tokens: usize, density: f32) -> SparseTokenMask {
    if density >= 0.999 {
        SparseTokenMask::all(dense_tokens)
    } else {
        let keep = ((dense_tokens as f32) * density).ceil() as usize;
        SparseTokenMask::evenly_spaced(dense_tokens, keep.max(1).min(dense_tokens))
    }
}

fn row_from_samples(
    image_size: usize,
    dense_tokens: usize,
    samples: &MetricSamples,
) -> Result<PipelinePerfRow> {
    let metrics = samples
        .last_metrics
        .as_ref()
        .context("perf row has no samples")?;
    let context_tokens = metrics.valid_encode_tokens.max(1);
    let encode_mean_us = mean(&samples.encode_us);
    let cache_update_mean_us = mean(&samples.cache_update_us);
    let total_mean_us = mean(&samples.total_us);
    let wall_mean_us = mean(&samples.wall_us);
    Ok(PipelinePerfRow {
        image_size,
        grid_height: image_size / 16,
        grid_width: image_size / 16,
        dense_tokens,
        context_tokens,
        density: context_tokens as f32 / dense_tokens.max(1) as f32,
        encode_path: metrics.encode_path.as_str().to_string(),
        encode_mean_us,
        encode_p50_us: percentile(&samples.encode_us, 0.50),
        encode_p95_us: percentile(&samples.encode_us, 0.95),
        cache_update_mean_us,
        cache_update_p50_us: percentile(&samples.cache_update_us, 0.50),
        cache_update_p95_us: percentile(&samples.cache_update_us, 0.95),
        total_mean_us,
        total_p50_us: percentile(&samples.total_us, 0.50),
        total_p95_us: percentile(&samples.total_us, 0.95),
        wall_mean_us,
        wall_p50_us: percentile(&samples.wall_us, 0.50),
        wall_p95_us: percentile(&samples.wall_us, 0.95),
        encoder_mtokens_per_sec: tokens_per_second(context_tokens, encode_mean_us) / 1.0e6,
        cache_mtokens_per_sec: tokens_per_second(context_tokens, cache_update_mean_us) / 1.0e6,
        step_fps: if wall_mean_us > 0.0 {
            1.0e6 / wall_mean_us
        } else {
            0.0
        },
    })
}

fn mean(values: &[u64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<u64>() as f64 / values.len() as f64
}

fn percentile(values: &[u64], quantile: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let index = ((sorted.len() - 1) as f64 * quantile.clamp(0.0, 1.0)).round() as usize;
    sorted[index]
}

fn tokens_per_second(tokens: usize, micros: f64) -> f64 {
    if micros > 0.0 {
        tokens as f64 * 1.0e6 / micros
    } else {
        0.0
    }
}

fn micros_u64(micros: u128) -> u64 {
    micros.min(u128::from(u64::MAX)) as u64
}

#[cfg(target_arch = "wasm32")]
async fn flush_measured_output(
    measured: &burn_jepa::MeasuredFeatureFrameBatch<JepaBevyBackend>,
) -> Result<()> {
    measured
        .output
        .token_cache
        .features
        .clone()
        .slice([0..1, 0..1, 0..1])
        .into_data_async()
        .await
        .context("flush wasm perf output tensor")?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn wasm_now() -> f64 {
    web_sys::window()
        .and_then(|window| window.performance())
        .map(|performance| performance.now())
        .unwrap_or_else(js_sys::Date::now)
}
