#![cfg(not(target_arch = "wasm32"))]

use std::{
    env, fs,
    path::PathBuf,
    process::Command,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use bevy_jepa::{
    BevyJepaConfig, BevyJepaDisplayTransfer, BevyJepaEncodePath, BevyJepaEncoderSource,
    BevyJepaFrameSource, BevyJepaHeadlessPipeline, BevyJepaMaskSource, BevyJepaModelPackageProfile,
    BevyJepaSparseEncodeMode, FeatureFrameViewerConfig, JepaBevyBackend, JepaBevyDevice, platform,
};
use burn::tensor::backend::Backend;
use burn_jepa::{FeatureFrameRequest, PatchDiffRefreshConfig};
use image::{Rgba, RgbaImage};
use serde::Serialize;

const PATCH_SIZE: usize = 16;
const DEFAULT_FRAMES: usize = 24;
const DEFAULT_WARMUP: usize = 4;
const DEFAULT_SPARSE_DENSITY: f32 = 0.30;
const DEFAULT_THRESHOLD: f32 = 0.03;
const DEFAULT_SAMPLE_MS: u64 = 50;

fn main() -> Result<()> {
    let args = Args::parse()?;
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("create {}", args.output_dir.display()))?;
    let summary = run_case(&args)?;
    let label = args.case_label();
    let json_path = args.output_dir.join(format!("{label}.json"));
    let csv_path = args.output_dir.join(format!("{label}.csv"));
    fs::write(&json_path, serde_json::to_string_pretty(&summary)?)
        .with_context(|| format!("write {}", json_path.display()))?;
    fs::write(&csv_path, summary.to_csv())
        .with_context(|| format!("write {}", csv_path.display()))?;
    println!("{}", summary.to_markdown_row());
    println!("wrote {}", json_path.display());
    println!("wrote {}", csv_path.display());
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ModelKind {
    Base,
    Ttt,
}

impl ModelKind {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "base" | "vjepa" | "vjepa2_1_base" => Ok(Self::Base),
            "ttt" | "trained-ttt" | "vjepa2_1_ttt" => Ok(Self::Ttt),
            other => bail!("unsupported --model `{other}`; expected base or ttt"),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Ttt => "ttt",
        }
    }

    const fn encoder_source(self) -> BevyJepaEncoderSource {
        match self {
            Self::Base => BevyJepaEncoderSource::BaseCheckpoint,
            Self::Ttt => BevyJepaEncoderSource::TrainedTtt,
        }
    }

    const fn profile(self) -> BevyJepaModelPackageProfile {
        match self {
            Self::Base => BevyJepaModelPackageProfile::Vjepa21Base,
            Self::Ttt => BevyJepaModelPackageProfile::Vjepa21Ttt,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum InputMode {
    Dense,
    Sparse,
}

impl InputMode {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "dense" | "full" | "full-frame" => Ok(Self::Dense),
            "sparse" | "patch-diff" | "patchdiff" => Ok(Self::Sparse),
            other => bail!("unsupported --mode `{other}`; expected dense or sparse"),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Sparse => "sparse",
        }
    }
}

#[derive(Debug)]
struct Args {
    model: ModelKind,
    mode: InputMode,
    resolution: usize,
    frames: usize,
    warmup: usize,
    sparse_density: f32,
    threshold: f32,
    encode_route: Option<BevyJepaEncodePath>,
    display_panels: bool,
    sample_ms: u64,
    sync_measurements: bool,
    allow_download: bool,
    output_dir: PathBuf,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self {
            model: ModelKind::Ttt,
            mode: InputMode::Sparse,
            resolution: 256,
            frames: DEFAULT_FRAMES,
            warmup: DEFAULT_WARMUP,
            sparse_density: DEFAULT_SPARSE_DENSITY,
            threshold: DEFAULT_THRESHOLD,
            encode_route: None,
            display_panels: false,
            sample_ms: DEFAULT_SAMPLE_MS,
            sync_measurements: false,
            allow_download: false,
            output_dir: PathBuf::from("target/burn-jepa-gpu-inference"),
        };
        let mut values = env::args().skip(1);
        while let Some(flag) = values.next() {
            match flag.as_str() {
                "--model" => args.model = ModelKind::parse(&next_value(&mut values, &flag)?)?,
                "--mode" => args.mode = InputMode::parse(&next_value(&mut values, &flag)?)?,
                "--resolution" | "--image-size" => {
                    args.resolution = parse_value(&next_value(&mut values, &flag)?, &flag)?
                }
                "--frames" => args.frames = parse_value(&next_value(&mut values, &flag)?, &flag)?,
                "--warmup" => args.warmup = parse_value(&next_value(&mut values, &flag)?, &flag)?,
                "--sparse-density" => {
                    args.sparse_density = parse_value(&next_value(&mut values, &flag)?, &flag)?
                }
                "--threshold" | "--patch-diff-threshold" => {
                    args.threshold = parse_value(&next_value(&mut values, &flag)?, &flag)?
                }
                "--encode-route" | "--encode-path" => {
                    args.encode_route = Some(parse_value(&next_value(&mut values, &flag)?, &flag)?)
                }
                "--display-panels" => args.display_panels = true,
                "--sample-ms" => {
                    args.sample_ms = parse_value(&next_value(&mut values, &flag)?, &flag)?
                }
                "--sync-measurements" => args.sync_measurements = true,
                "--allow-download" => args.allow_download = true,
                "--output-dir" => args.output_dir = PathBuf::from(next_value(&mut values, &flag)?),
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => bail!("unknown argument `{other}`; pass --help for usage"),
            }
        }
        args.resolution = args.resolution.max(PATCH_SIZE).div_ceil(PATCH_SIZE) * PATCH_SIZE;
        args.frames = args.frames.max(1);
        args.sparse_density = args.sparse_density.clamp(0.0, 1.0);
        args.threshold = args.threshold.clamp(0.0, 1.0);
        args.sample_ms = args.sample_ms.max(10);
        Ok(args)
    }

    fn case_label(&self) -> String {
        format!(
            "{}_{}_{}_{}",
            self.model.as_str(),
            self.mode.as_str(),
            self.encode_route
                .map(|route| route.as_str())
                .unwrap_or("auto"),
            self.resolution
        )
    }
}

fn print_help() {
    println!(
        "gpu_inference_case --model base|ttt --mode dense|sparse --resolution 256|512 [--encode-route auto|dense-patch|sparse-patchify] [--display-panels] [--frames N] [--warmup N] [--sparse-density D] [--threshold T] [--sync-measurements] [--allow-download]"
    );
}

fn next_value(values: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    values
        .next()
        .with_context(|| format!("missing value after {flag}"))
}

fn parse_value<T: std::str::FromStr>(value: &str, flag: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("parse {flag} value `{value}`: {err}"))
}

fn run_case(args: &Args) -> Result<CaseSummary> {
    let sender = ensure_camera_sender();
    let cold_baseline = read_gpu_sample().ok();
    let device = JepaBevyDevice::default();
    let config = case_config(args);
    let mut pipeline = BevyJepaHeadlessPipeline::new(config, device.clone());

    for frame in 0..args.warmup {
        send_frame(
            &sender,
            motion_frame(args.resolution, args.sparse_density, frame as u64),
        )?;
        let _ = step_pipeline(&mut pipeline, args.display_panels)?;
        JepaBevyBackend::sync(&device)?;
    }

    let inference_baseline = read_gpu_sample().ok();
    let sampler = GpuSampler::start(args.sample_ms);
    let wall = Instant::now();
    let mut frames = Vec::with_capacity(args.frames);
    for frame in args.warmup..args.warmup + args.frames {
        send_frame(
            &sender,
            motion_frame(args.resolution, args.sparse_density, frame as u64),
        )?;
        let started = Instant::now();
        let output = step_pipeline(&mut pipeline, args.display_panels)?;
        JepaBevyBackend::sync(&device)?;
        frames.push(FrameTiming::from_metrics(
            started.elapsed(),
            &output.metrics,
        ));
    }
    let elapsed_ms = wall.elapsed().as_secs_f64() * 1000.0;
    let gpu = sampler.stop();
    Ok(CaseSummary::new(
        args,
        cold_baseline,
        inference_baseline,
        gpu,
        frames,
        elapsed_ms,
    ))
}

fn step_pipeline(
    pipeline: &mut BevyJepaHeadlessPipeline,
    display_panels: bool,
) -> Result<bevy_jepa::BevyJepaStepOutput> {
    if display_panels {
        pipeline.step_with_display_request(FeatureFrameRequest::low_res())
    } else {
        pipeline.step_with_stage_request(FeatureFrameRequest::low_res())
    }
}

fn case_config(args: &Args) -> BevyJepaConfig {
    let mut pipeline = FeatureFrameViewerConfig {
        image_size: args.resolution,
        high_res_pca_every: 0,
        measure_stages: true,
        sync_measurements: args.sync_measurements,
        prewarm_shape_buckets: args.mode == InputMode::Sparse,
        ..FeatureFrameViewerConfig::default()
    };
    match args.mode {
        InputMode::Dense => {
            pipeline.encode_path = BevyJepaEncodePath::DensePatchEmbed;
            pipeline.context_density = 1.0;
            pipeline.min_context_density = 1.0;
            pipeline.bootstrap_context_density = 1.0;
            pipeline.patch_diff_threshold = 0.0;
            pipeline.patch_diff_dense_fallback_density = 0.0;
            pipeline.sparse_encode_mode = BevyJepaSparseEncodeMode::Exact;
            pipeline.patch_diff_refresh = PatchDiffRefreshConfig::disabled();
        }
        InputMode::Sparse => {
            pipeline.encode_path = args.encode_route.unwrap_or(BevyJepaEncodePath::Auto);
            pipeline.context_density = 1.0;
            pipeline.min_context_density = 0.0;
            pipeline.bootstrap_context_density = 1.0;
            pipeline.patch_diff_threshold = args.threshold;
            pipeline.patch_diff_dense_fallback_density = 0.60;
            pipeline.sparse_encode_mode = BevyJepaSparseEncodeMode::BucketedContext;
            pipeline.patch_diff_refresh = PatchDiffRefreshConfig::disabled();
        }
    }
    BevyJepaConfig {
        encoder_source: args.model.encoder_source(),
        model_profile: args.model.profile(),
        model_base_url: burn_jepa::burn_jepa_model_profile_base_url(args.model.profile()),
        model_auto_download: args.allow_download,
        ttt_model_path: None,
        jepa_checkpoint_dir: None,
        jepa_config_path: None,
        source: BevyJepaFrameSource::Camera,
        mask_source: BevyJepaMaskSource::PatchDiff,
        display_transfer: BevyJepaDisplayTransfer::Gpu,
        pipeline,
        show_metrics: false,
        ..BevyJepaConfig::default()
    }
}

fn ensure_camera_sender() -> mpsc::SyncSender<RgbaImage> {
    if let Some(sender) = platform::camera::SAMPLE_SENDER.get() {
        return sender.clone();
    }
    let (sender, receiver) = mpsc::sync_channel(1);
    let _ = platform::camera::SAMPLE_RECEIVER.set(Arc::new(Mutex::new(receiver)));
    let _ = platform::camera::SAMPLE_SENDER.set(sender.clone());
    sender
}

fn send_frame(sender: &mpsc::SyncSender<RgbaImage>, frame: RgbaImage) -> Result<()> {
    match sender.try_send(frame) {
        Ok(()) => Ok(()),
        Err(mpsc::TrySendError::Full(frame)) => {
            while platform::camera::receive_image().is_some() {}
            sender
                .try_send(frame)
                .map_err(|err| anyhow::anyhow!("send synthetic camera frame: {err}"))
        }
        Err(mpsc::TrySendError::Disconnected(_)) => bail!("synthetic camera receiver disconnected"),
    }
}

fn motion_frame(image_size: usize, active_density: f32, frame_index: u64) -> RgbaImage {
    let grid = image_size / PATCH_SIZE;
    let dense_tokens = grid * grid;
    let active_tokens = ((dense_tokens as f32) * active_density).round() as usize;
    let active = active_patch_set(dense_tokens, active_tokens.min(dense_tokens), 0);
    let mut image = RgbaImage::from_pixel(
        image_size as u32,
        image_size as u32,
        Rgba([96, 96, 96, 255]),
    );
    for token in active {
        let row = token / grid;
        let col = token % grid;
        let hash = mix64(token as u64 ^ frame_index.rotate_left(17));
        let flip = if frame_index.is_multiple_of(2) { 0 } else { 96 };
        let rgba = Rgba([
            (48u8.saturating_add((hash & 0x7f) as u8)) ^ flip,
            (64u8.saturating_add(((hash >> 11) & 0x7f) as u8)) ^ flip,
            (80u8.saturating_add(((hash >> 23) & 0x7f) as u8)) ^ flip,
            255,
        ]);
        for y in row * PATCH_SIZE..(row + 1) * PATCH_SIZE {
            for x in col * PATCH_SIZE..(col + 1) * PATCH_SIZE {
                image.put_pixel(x as u32, y as u32, rgba);
            }
        }
    }
    image
}

fn active_patch_set(dense_tokens: usize, active_tokens: usize, seed: u64) -> Vec<usize> {
    let mut ranked = (0..dense_tokens)
        .map(|index| {
            (
                mix64(index as u64 ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
                index,
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable_by_key(|(score, index)| (*score, *index));
    ranked
        .into_iter()
        .take(active_tokens)
        .map(|(_, index)| index)
        .collect()
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Clone, Copy, Debug, Serialize)]
struct GpuSample {
    timestamp_ms: u128,
    gpu_util_percent: f64,
    memory_util_percent: f64,
    memory_used_mib: f64,
    power_w: f64,
}

struct GpuSampler {
    running: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<GpuSample>>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl GpuSampler {
    fn start(sample_ms: u64) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let samples = Arc::new(Mutex::new(Vec::new()));
        let thread_running = running.clone();
        let thread_samples = samples.clone();
        let handle = thread::spawn(move || {
            while thread_running.load(Ordering::Relaxed) {
                if let Ok(sample) = read_gpu_sample()
                    && let Ok(mut samples) = thread_samples.lock()
                {
                    samples.push(sample);
                }
                thread::sleep(Duration::from_millis(sample_ms));
            }
        });
        Self {
            running,
            samples,
            handle: Some(handle),
        }
    }

    fn stop(mut self) -> Vec<GpuSample> {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.samples
            .lock()
            .map(|samples| samples.clone())
            .unwrap_or_default()
    }
}

fn read_gpu_sample() -> Result<GpuSample> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .context("run nvidia-smi")?;
    if !output.status.success() {
        bail!("nvidia-smi exited with {}", output.status);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .find(|line| !line.trim().is_empty())
        .context("nvidia-smi returned no GPU rows")?;
    let values = line
        .split(',')
        .map(|value| value.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("parse nvidia-smi row `{line}`"))?;
    if values.len() < 4 {
        bail!("nvidia-smi row has {} fields, expected 4", values.len());
    }
    Ok(GpuSample {
        timestamp_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
        gpu_util_percent: values[0],
        memory_util_percent: values[1],
        memory_used_mib: values[2],
        power_w: values[3],
    })
}

#[derive(Clone, Debug, Serialize)]
struct FrameTiming {
    outer_ms: f64,
    viewer_ms: f64,
    encode_ms: f64,
    cache_ms: f64,
    low_res_pca_ms: f64,
    pca_update_ms: f64,
    display_ms: f64,
    write_density: f64,
    encode_density: f64,
    write_tokens: usize,
    encode_tokens: usize,
    dense_tokens: usize,
}

impl FrameTiming {
    fn from_metrics(elapsed: Duration, metrics: &bevy_jepa::BevyJepaMetrics) -> Self {
        let dense_tokens = metrics.dense_tokens.max(1);
        let encode_tokens = metrics
            .stage_metrics
            .encode_width
            .max(metrics.stage_metrics.sparse_width)
            .max(metrics.context_tokens);
        Self {
            outer_ms: elapsed.as_secs_f64() * 1000.0,
            viewer_ms: us_to_ms(metrics.viewer_total_us),
            encode_ms: us_to_ms(metrics.encode_us),
            cache_ms: us_to_ms(metrics.cache_update_us),
            low_res_pca_ms: us_to_ms(metrics.low_res_pca_us),
            pca_update_ms: us_to_ms(metrics.pca_update_us),
            display_ms: us_to_ms(metrics.display_tensor_us),
            write_density: metrics.context_tokens as f64 / dense_tokens as f64,
            encode_density: encode_tokens as f64 / dense_tokens as f64,
            write_tokens: metrics.context_tokens,
            encode_tokens,
            dense_tokens,
        }
    }
}

#[derive(Debug, Serialize)]
struct CaseSummary {
    model: ModelKind,
    mode: InputMode,
    encode_route: String,
    resolution: usize,
    frames: usize,
    warmup: usize,
    sparse_density_requested: f32,
    threshold: f32,
    sync_measurements: bool,
    display_panels: bool,
    elapsed_ms: f64,
    fps: f64,
    mean_outer_ms: f64,
    p50_outer_ms: f64,
    p95_outer_ms: f64,
    mean_encode_ms: f64,
    mean_cache_ms: f64,
    mean_low_res_pca_ms: f64,
    mean_pca_update_ms: f64,
    mean_display_ms: f64,
    mean_write_density: f64,
    mean_encode_density: f64,
    mean_write_tokens: f64,
    mean_encode_tokens: f64,
    dense_tokens: usize,
    gpu_sample_count: usize,
    gpu_cold_baseline: Option<GpuSample>,
    gpu_inference_baseline: Option<GpuSample>,
    gpu_mean_util_percent: f64,
    gpu_p95_util_percent: f64,
    gpu_mean_memory_util_percent: f64,
    gpu_peak_memory_mib: f64,
    gpu_peak_memory_delta_from_cold_mib: Option<f64>,
    gpu_peak_memory_delta_from_inference_mib: Option<f64>,
    gpu_mean_power_w: f64,
    gpu_peak_power_w: f64,
}

impl CaseSummary {
    fn new(
        args: &Args,
        cold_baseline: Option<GpuSample>,
        inference_baseline: Option<GpuSample>,
        gpu: Vec<GpuSample>,
        frames: Vec<FrameTiming>,
        elapsed_ms: f64,
    ) -> Self {
        let fps = if elapsed_ms > 0.0 {
            frames.len() as f64 / (elapsed_ms / 1000.0)
        } else {
            0.0
        };
        let peak_memory = max_f64(gpu.iter().map(|sample| sample.memory_used_mib));
        let peak_delta_from_cold = cold_baseline.map(|sample| peak_memory - sample.memory_used_mib);
        let peak_delta_from_inference =
            inference_baseline.map(|sample| peak_memory - sample.memory_used_mib);
        Self {
            model: args.model,
            mode: args.mode,
            encode_route: args
                .encode_route
                .unwrap_or(match args.mode {
                    InputMode::Dense => BevyJepaEncodePath::DensePatchEmbed,
                    InputMode::Sparse => BevyJepaEncodePath::Auto,
                })
                .as_str()
                .to_string(),
            resolution: args.resolution,
            frames: frames.len(),
            warmup: args.warmup,
            sparse_density_requested: args.sparse_density,
            threshold: args.threshold,
            sync_measurements: args.sync_measurements,
            display_panels: args.display_panels,
            elapsed_ms,
            fps,
            mean_outer_ms: mean(frames.iter().map(|frame| frame.outer_ms)),
            p50_outer_ms: percentile_f64(frames.iter().map(|frame| frame.outer_ms), 0.50),
            p95_outer_ms: percentile_f64(frames.iter().map(|frame| frame.outer_ms), 0.95),
            mean_encode_ms: mean(frames.iter().map(|frame| frame.encode_ms)),
            mean_cache_ms: mean(frames.iter().map(|frame| frame.cache_ms)),
            mean_low_res_pca_ms: mean(frames.iter().map(|frame| frame.low_res_pca_ms)),
            mean_pca_update_ms: mean(frames.iter().map(|frame| frame.pca_update_ms)),
            mean_display_ms: mean(frames.iter().map(|frame| frame.display_ms)),
            mean_write_density: mean(frames.iter().map(|frame| frame.write_density)),
            mean_encode_density: mean(frames.iter().map(|frame| frame.encode_density)),
            mean_write_tokens: mean(frames.iter().map(|frame| frame.write_tokens as f64)),
            mean_encode_tokens: mean(frames.iter().map(|frame| frame.encode_tokens as f64)),
            dense_tokens: frames.first().map(|frame| frame.dense_tokens).unwrap_or(0),
            gpu_sample_count: gpu.len(),
            gpu_cold_baseline: cold_baseline,
            gpu_inference_baseline: inference_baseline,
            gpu_mean_util_percent: mean(gpu.iter().map(|sample| sample.gpu_util_percent)),
            gpu_p95_util_percent: percentile_f64(
                gpu.iter().map(|sample| sample.gpu_util_percent),
                0.95,
            ),
            gpu_mean_memory_util_percent: mean(gpu.iter().map(|sample| sample.memory_util_percent)),
            gpu_peak_memory_mib: peak_memory,
            gpu_peak_memory_delta_from_cold_mib: peak_delta_from_cold,
            gpu_peak_memory_delta_from_inference_mib: peak_delta_from_inference,
            gpu_mean_power_w: mean(gpu.iter().map(|sample| sample.power_w)),
            gpu_peak_power_w: max_f64(gpu.iter().map(|sample| sample.power_w)),
        }
    }

    fn to_csv(&self) -> String {
        format!(
            "model,mode,encode_route,resolution,display_panels,frames,fps,mean_outer_ms,p95_outer_ms,mean_encode_ms,mean_cache_ms,mean_low_res_pca_ms,mean_pca_update_ms,mean_write_density,mean_encode_density,mean_write_tokens,mean_encode_tokens,dense_tokens,gpu_samples,gpu_mean_util_percent,gpu_p95_util_percent,gpu_mean_memory_util_percent,gpu_peak_memory_mib,gpu_peak_memory_delta_mib,gpu_mean_power_w,gpu_peak_power_w\n{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.4},{:.4},{:.1},{:.1},{},{},{:.2},{:.2},{:.2},{:.1},{:.1},{:.2},{:.2}\n",
            self.model.as_str(),
            self.mode.as_str(),
            self.encode_route,
            self.resolution,
            self.display_panels,
            self.frames,
            self.fps,
            self.mean_outer_ms,
            self.p95_outer_ms,
            self.mean_encode_ms,
            self.mean_cache_ms,
            self.mean_low_res_pca_ms,
            self.mean_pca_update_ms,
            self.mean_write_density,
            self.mean_encode_density,
            self.mean_write_tokens,
            self.mean_encode_tokens,
            self.dense_tokens,
            self.gpu_sample_count,
            self.gpu_mean_util_percent,
            self.gpu_p95_util_percent,
            self.gpu_mean_memory_util_percent,
            self.gpu_peak_memory_mib,
            self.gpu_peak_memory_delta_from_cold_mib.unwrap_or(0.0),
            self.gpu_mean_power_w,
            self.gpu_peak_power_w,
        )
    }

    fn to_markdown_row(&self) -> String {
        format!(
            "| {} | {} | {} | {} | {:.1}% / {:.1}% | {:.2} | {:.2} | {:.1}% | {:.0} MiB ({:+.0}) | {:.1} W |",
            self.model.as_str(),
            self.mode.as_str(),
            self.encode_route,
            self.resolution,
            self.mean_write_density * 100.0,
            self.mean_encode_density * 100.0,
            self.mean_outer_ms,
            self.fps,
            self.gpu_mean_util_percent,
            self.gpu_peak_memory_mib,
            self.gpu_peak_memory_delta_from_cold_mib.unwrap_or(0.0),
            self.gpu_mean_power_w,
        )
    }
}

fn us_to_ms(value: u64) -> f64 {
    value as f64 / 1000.0
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value;
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

fn max_f64(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0, f64::max)
}

fn percentile_f64(values: impl Iterator<Item = f64>, p: f64) -> f64 {
    let mut values = values.collect::<Vec<_>>();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let index = (((values.len() - 1) as f64) * p.clamp(0.0, 1.0)).round() as usize;
    values[index]
}
