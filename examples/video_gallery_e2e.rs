use anyhow::{Context, Result, bail, ensure};
use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use burn_jepa::{
    AnyUp, AnyUpConfig, FeatureFrameMeasureConfig, FeatureFramePipeline,
    FeatureFramePipelineConfig, FeatureFrameRequest, FeaturePcaProjector, FeaturePcaUpdateConfig,
    SparseTokenMask, TokenGridShape, VJepa2_1Model, VJepaConfig,
};
use clap::Parser;
use image::{ImageReader, RgbImage, imageops::FilterType};
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

type GalleryBackend = NdArray<f32>;

#[derive(Parser, Debug)]
#[command(about = "Generate an E2E Burn sparse JEPA/AnyUp/PCA video gallery")]
struct Args {
    #[arg(long, default_value = "target/burn-jepa-video-gallery")]
    output: PathBuf,
    #[arg(long)]
    dataset_root: PathBuf,
    #[arg(long, default_value_t = 16)]
    samples: usize,
    #[arg(long, default_value_t = 40)]
    frames: usize,
    #[arg(long, default_value_t = 1)]
    stride: usize,
    #[arg(long, default_value_t = 10)]
    fps: usize,
    #[arg(long, default_value_t = 224)]
    image_size: usize,
    #[arg(long, default_value_t = false)]
    force: bool,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct GalleryConfig {
    config_id: &'static str,
    encoder_path: &'static str,
    mask_policy: MaskPolicy,
    density: f32,
    quality: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum MaskPolicy {
    FullFrame,
    AutogazeStream,
    PatchDiff,
}

#[derive(Clone, Debug, Serialize)]
struct SampleWindow {
    sample_id: String,
    clip_name: String,
    clip_dir: String,
    dataset_split: String,
    start_frame: usize,
    frame_count: usize,
    stride: usize,
}

struct HostFrame {
    rgb: Vec<u8>,
    gray: Vec<f32>,
    tensor: Tensor<GalleryBackend, 4>,
}

#[derive(Clone, Debug, Default, Serialize)]
struct RenderStats {
    observed_tokens: usize,
    dense_tokens: usize,
    updated_tokens: usize,
    frames: usize,
    observed_density: f32,
    mask_ms: f64,
    pipeline_ms: f64,
    encode_ms: f64,
    cache_update_ms: f64,
    anyup_ms: f64,
    low_res_pca_ms: f64,
    high_res_pca_ms: f64,
    pca_update_ms: f64,
    mask_video_ms: f64,
    low_video_ms: f64,
    high_video_ms: f64,
}

#[derive(Debug, Serialize)]
struct ConfigResult {
    videos: BTreeMap<&'static str, String>,
    stats: RenderStats,
}

#[derive(Debug, Serialize)]
struct SampleResult {
    #[serde(flatten)]
    window: SampleWindow,
    input_video: String,
    configs: BTreeMap<&'static str, ConfigResult>,
}

#[derive(Debug, Serialize)]
struct Manifest {
    summary: ManifestSummary,
    configs: Vec<GalleryConfig>,
    samples: Vec<SampleResult>,
}

#[derive(Debug, Serialize)]
struct ManifestSummary {
    dataset: &'static str,
    dataset_url: &'static str,
    sample_count: usize,
    config_count: usize,
    mp4_count: usize,
    image_size: usize,
    token_grid: usize,
    frames_per_sample: usize,
    fps: usize,
    feature_source: &'static str,
    mask_source: &'static str,
    burn_pipeline: &'static str,
    burn_backend: &'static str,
    model_source: &'static str,
    pipeline_note: &'static str,
    generated_at_unix: u64,
}

const DATASET_URL: &str = "http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz";

const CONFIGS: [GalleryConfig; 7] = [
    GalleryConfig {
        config_id: "dense_100",
        encoder_path: "dense_token_context",
        mask_policy: MaskPolicy::FullFrame,
        density: 1.0,
        quality: "full",
    },
    GalleryConfig {
        config_id: "autogaze_50",
        encoder_path: "sparse_token_context",
        mask_policy: MaskPolicy::AutogazeStream,
        density: 0.50,
        quality: "high",
    },
    GalleryConfig {
        config_id: "autogaze_25",
        encoder_path: "sparse_token_context",
        mask_policy: MaskPolicy::AutogazeStream,
        density: 0.25,
        quality: "medium",
    },
    GalleryConfig {
        config_id: "autogaze_10",
        encoder_path: "sparse_token_context",
        mask_policy: MaskPolicy::AutogazeStream,
        density: 0.10,
        quality: "low",
    },
    GalleryConfig {
        config_id: "patchdiff_50",
        encoder_path: "sparse_token_context",
        mask_policy: MaskPolicy::PatchDiff,
        density: 0.50,
        quality: "high",
    },
    GalleryConfig {
        config_id: "patchdiff_25",
        encoder_path: "sparse_token_context",
        mask_policy: MaskPolicy::PatchDiff,
        density: 0.25,
        quality: "medium",
    },
    GalleryConfig {
        config_id: "patchdiff_10",
        encoder_path: "sparse_token_context",
        mask_policy: MaskPolicy::PatchDiff,
        density: 0.10,
        quality: "low",
    },
];

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}

fn run(args: Args) -> Result<()> {
    ensure!(args.samples > 0, "sample count must be nonzero");
    ensure!(args.frames > 0, "frames per sample must be nonzero");
    ensure!(args.stride > 0, "frame stride must be nonzero");
    ensure!(args.fps > 0, "fps must be nonzero");
    ensure!(
        args.image_size.is_multiple_of(16),
        "image size must be divisible by V-JEPA patch size"
    );

    if args.force {
        fs::remove_dir_all(args.output.join("videos")).ok();
        fs::remove_file(args.output.join("index.html")).ok();
        fs::remove_file(args.output.join("manifest.json")).ok();
    }
    fs::create_dir_all(&args.output)?;

    let clips = discover_clips(&args.dataset_root, args.frames, args.stride)?;
    let samples = choose_samples(
        &args.dataset_root,
        clips,
        args.samples,
        args.frames,
        args.stride,
    )?;
    let device = Default::default();
    let mut model_config = VJepaConfig::tiny_for_tests();
    model_config.image_size = args.image_size;
    model_config.num_frames = 2;
    model_config.tubelet_size = 2;
    let grid = TokenGridShape::new(
        1,
        args.image_size / model_config.patch_size,
        args.image_size / model_config.patch_size,
    );
    let jepa = VJepa2_1Model::<GalleryBackend>::new(&model_config, &device);
    let anyup = AnyUp::<GalleryBackend>::new(AnyUpConfig::tiny_for_tests(), &device)?;
    let pipeline_config = FeatureFramePipelineConfig {
        anyup_q_chunk_size: Some(16),
        pca_update: FeaturePcaUpdateConfig::rolling_low_res_every(1),
        measurement: FeatureFrameMeasureConfig::enabled(),
        ..FeatureFramePipelineConfig::default()
    };
    let mut pipeline = FeatureFramePipeline::<GalleryBackend>::new(
        jepa,
        anyup,
        &model_config,
        pipeline_config,
        1,
        [args.image_size, args.image_size],
        &device,
    )?;

    let mut manifest = Manifest {
        summary: ManifestSummary {
            dataset: "UCSD Anomaly/Pedestrian dataset",
            dataset_url: DATASET_URL,
            sample_count: samples.len(),
            config_count: CONFIGS.len(),
            mp4_count: samples.len() * (1 + CONFIGS.len() * 3),
            image_size: args.image_size,
            token_grid: grid.height,
            frames_per_sample: args.frames,
            fps: args.fps,
            feature_source: "burn_jepa_vjepa2_1_feature_frame_pipeline",
            mask_source: "burn_jepa_gallery_masks_full_autogaze_stream_patchdiff",
            burn_pipeline: "FeatureFramePipeline: image -> sparse mask -> V-JEPA encoder -> interframe feature cache -> PCA -> AnyUp high-res PCA",
            burn_backend: "ndarray",
            model_source: "tiny_for_tests_untrained",
            pipeline_note: "Low/high PCA artifacts are generated by the Burn FeatureFramePipeline e2e path (ndarray backend, tiny untrained V-JEPA/AnyUp module config): V-JEPA encode, sparse feature-cache update, low-res PCA, AnyUp upsample, and high-res PCA.",
            generated_at_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        },
        configs: CONFIGS.to_vec(),
        samples: Vec::new(),
    };

    println!(
        "dataset={} clips={} samples={} configs={} grid={}x{}",
        args.dataset_root.display(),
        discover_clip_count(&args.dataset_root)?,
        samples.len(),
        CONFIGS.len(),
        grid.height,
        grid.width
    );

    for (sample_index, sample) in samples.into_iter().enumerate() {
        println!(
            "[{:02}/{:02}] {} {}",
            sample_index + 1,
            args.samples,
            sample.sample_id,
            sample.clip_name
        );
        let frames = load_sample_frames(&sample, args.image_size, &device)?;
        let input_video =
            write_input_video(&args.output, &sample, &frames, args.fps, args.image_size)?;
        let mut sample_result = SampleResult {
            window: sample.clone(),
            input_video,
            configs: BTreeMap::new(),
        };
        for config in CONFIGS {
            reset_pipeline_state(&mut pipeline, &model_config, &device)?;
            let result = render_config(
                &args.output,
                &sample,
                &frames,
                config,
                &mut pipeline,
                grid,
                args.fps,
                args.image_size,
            )?;
            sample_result.configs.insert(config.config_id, result);
        }
        manifest.samples.push(sample_result);
        write_manifest(&args.output, &manifest)?;
    }
    write_manifest(&args.output, &manifest)?;
    write_html(&args.output, &manifest)?;
    println!("wrote {}", args.output.join("index.html").display());
    println!("wrote {}", args.output.join("manifest.json").display());
    Ok(())
}

fn reset_pipeline_state(
    pipeline: &mut FeatureFramePipeline<GalleryBackend>,
    model_config: &VJepaConfig,
    device: &<GalleryBackend as burn::tensor::backend::BackendTypes>::Device,
) -> Result<()> {
    pipeline.reset();
    let pca_config = pipeline.config().pca.clone();
    *pipeline.pca_mut() = FeaturePcaProjector::<GalleryBackend>::identity(
        model_config.encoder.embed_dim,
        pca_config,
        device,
    )?;
    Ok(())
}

fn discover_clip_count(dataset_root: &Path) -> Result<usize> {
    Ok(discover_clips(dataset_root, 1, 1)?.len())
}

fn discover_clips(dataset_root: &Path, frames: usize, stride: usize) -> Result<Vec<PathBuf>> {
    ensure!(
        dataset_root.exists(),
        "dataset root does not exist: {}",
        dataset_root.display()
    );
    let required = (frames - 1) * stride + 1;
    let mut clips = Vec::new();
    visit_dirs(dataset_root, &mut |path| {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            return Ok(());
        };
        if name.ends_with("_gt") || !(name.starts_with("Train") || name.starts_with("Test")) {
            return Ok(());
        }
        let count = frame_paths(path)?.len();
        if count >= required {
            clips.push(path.to_path_buf());
        }
        Ok(())
    })?;
    clips.sort();
    ensure!(
        !clips.is_empty(),
        "no eligible clips with at least {required} frames under {}",
        dataset_root.display()
    );
    Ok(clips)
}

fn visit_dirs(dir: &Path, f: &mut impl FnMut(&Path) -> Result<()>) -> Result<()> {
    if dir.is_dir() {
        f(dir)?;
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, f)?;
            }
        }
    }
    Ok(())
}

fn choose_samples(
    dataset_root: &Path,
    clips: Vec<PathBuf>,
    sample_count: usize,
    frame_count: usize,
    stride: usize,
) -> Result<Vec<SampleWindow>> {
    ensure!(
        clips.len() >= sample_count,
        "only {} clips are eligible; requested {sample_count}",
        clips.len()
    );
    let required = (frame_count - 1) * stride + 1;
    let step = (clips.len() / sample_count).max(1);
    let mut samples = Vec::with_capacity(sample_count);
    for index in 0..sample_count {
        let clip = clips[(index * step).min(clips.len() - 1)].clone();
        let frames = frame_paths(&clip)?;
        let max_start = frames.len().saturating_sub(required);
        let start_index = if max_start == 0 {
            0
        } else {
            ((index % 4) * max_start + 1) / 3
        };
        let start_frame = frame_number(&frames[start_index]);
        let clip_name = clip
            .strip_prefix(dataset_root)
            .unwrap_or(&clip)
            .to_string_lossy()
            .replace('\\', "/");
        let parts = clip_name.split('/').collect::<Vec<_>>();
        let dataset_split = if parts.len() >= 2 {
            format!("{}/{}", parts[0], parts[1])
        } else {
            String::new()
        };
        samples.push(SampleWindow {
            sample_id: format!("sample_{index:02}"),
            clip_name,
            clip_dir: clip.to_string_lossy().to_string(),
            dataset_split,
            start_frame,
            frame_count,
            stride,
        });
    }
    Ok(samples)
}

fn frame_paths(clip: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = fs::read_dir(clip)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| {
                    ext.eq_ignore_ascii_case("tif") || ext.eq_ignore_ascii_case("tiff")
                })
        })
        .collect::<Vec<_>>();
    paths.sort_by(|left, right| {
        frame_number(left)
            .cmp(&frame_number(right))
            .then_with(|| left.cmp(right))
    });
    Ok(paths)
}

fn frame_number(path: &Path) -> usize {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| stem.parse::<usize>().ok())
        .unwrap_or(0)
}

fn load_sample_frames(
    sample: &SampleWindow,
    image_size: usize,
    device: &<GalleryBackend as burn::tensor::backend::BackendTypes>::Device,
) -> Result<Vec<HostFrame>> {
    let clip = Path::new(&sample.clip_dir);
    let mut frames = Vec::with_capacity(sample.frame_count);
    for offset in 0..sample.frame_count {
        let frame_number = sample.start_frame + offset * sample.stride;
        let path = clip.join(format!("{frame_number:03}.tif"));
        frames.push(
            load_frame(&path, image_size, device).with_context(|| {
                format!("load frame {} for {}", path.display(), sample.sample_id)
            })?,
        );
    }
    Ok(frames)
}

fn load_frame(
    path: &Path,
    image_size: usize,
    device: &<GalleryBackend as burn::tensor::backend::BackendTypes>::Device,
) -> Result<HostFrame> {
    let image = ImageReader::open(path)?.decode()?.to_rgb8();
    let resized = image::imageops::resize(
        &image,
        image_size as u32,
        image_size as u32,
        FilterType::Triangle,
    );
    let rgb = resized.into_raw();
    let mut gray = Vec::with_capacity(image_size * image_size);
    for pixel in rgb.chunks_exact(3) {
        gray.push(
            (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) / 255.0,
        );
    }
    let mut values = vec![0.0_f32; 3 * image_size * image_size];
    let pixels = image_size * image_size;
    for index in 0..pixels {
        values[index] = rgb[index * 3] as f32 / 255.0;
        values[pixels + index] = rgb[index * 3 + 1] as f32 / 255.0;
        values[pixels * 2 + index] = rgb[index * 3 + 2] as f32 / 255.0;
    }
    let tensor = Tensor::<GalleryBackend, 4>::from_data(
        TensorData::new(values, [1, 3, image_size, image_size]),
        device,
    );
    Ok(HostFrame { rgb, gray, tensor })
}

fn render_config(
    output: &Path,
    sample: &SampleWindow,
    frames: &[HostFrame],
    config: GalleryConfig,
    pipeline: &mut FeatureFramePipeline<GalleryBackend>,
    grid: TokenGridShape,
    fps: usize,
    image_size: usize,
) -> Result<ConfigResult> {
    let mut stats = RenderStats {
        frames: frames.len(),
        ..RenderStats::default()
    };
    let mut mask_frames = Vec::with_capacity(frames.len());
    let mut low_frames = Vec::with_capacity(frames.len());
    let mut high_frames = Vec::with_capacity(frames.len());
    let mut previous: Option<&HostFrame> = None;
    for (frame_index, frame) in frames.iter().enumerate() {
        let mask_start = std::time::Instant::now();
        let mask = frame_mask(config, frame_index, frame, previous, grid, image_size)?;
        stats.mask_ms += mask_start.elapsed().as_secs_f64() * 1000.0;
        stats.observed_tokens += mask.len();
        stats.updated_tokens += mask.len();
        stats.dense_tokens += mask.dense_len();
        mask_frames.push(mask_overlay(&frame.rgb, &mask, grid, image_size));

        let measured = pipeline.step_image_with_mask_nodes_measured(
            frame.tensor.clone(),
            &mask,
            FeatureFrameRequest::full_pca(),
        )?;
        stats.pipeline_ms += measured.metrics.total_us as f64 / 1000.0;
        stats.encode_ms += measured.metrics.encode_us as f64 / 1000.0;
        stats.cache_update_ms += measured.metrics.cache_update_us as f64 / 1000.0;
        stats.anyup_ms +=
            (measured.metrics.anyup_context_us + measured.metrics.anyup_decode_us) as f64 / 1000.0;
        stats.low_res_pca_ms += measured.metrics.low_res_pca_project_us as f64 / 1000.0;
        stats.high_res_pca_ms += measured.metrics.pca_project_us as f64 / 1000.0;
        stats.pca_update_ms += measured.metrics.pca_update_us as f64 / 1000.0;

        let low = measured
            .output
            .low_res
            .pca_display
            .context("low-res PCA output missing from pipeline")?;
        let high = measured
            .output
            .high_res
            .and_then(|high| high.pca_display)
            .context("high-res PCA output missing from pipeline")?;
        low_frames.push(tensor_nchw_rgb8(low, [image_size, image_size], true)?);
        high_frames.push(tensor_nchw_rgb8(high, [image_size, image_size], false)?);
        previous = Some(frame);
    }
    stats.observed_density = if stats.dense_tokens == 0 {
        0.0
    } else {
        stats.observed_tokens as f32 / stats.dense_tokens as f32
    };

    let config_dir = output
        .join("videos")
        .join(&sample.sample_id)
        .join(config.config_id);
    let mask_path = config_dir.join("mask.mp4");
    let low_path = config_dir.join("low_pca.mp4");
    let high_path = config_dir.join("high_pca.mp4");
    stats.mask_video_ms = write_mp4(&mask_path, &mask_frames, image_size, image_size, fps)?;
    stats.low_video_ms = write_mp4(&low_path, &low_frames, image_size, image_size, fps)?;
    stats.high_video_ms = write_mp4(&high_path, &high_frames, image_size, image_size, fps)?;
    let mut videos = BTreeMap::new();
    videos.insert("mask", relpath(&mask_path, output));
    videos.insert("low_pca", relpath(&low_path, output));
    videos.insert("high_pca", relpath(&high_path, output));
    Ok(ConfigResult { videos, stats })
}

fn frame_mask(
    config: GalleryConfig,
    frame_index: usize,
    frame: &HostFrame,
    previous: Option<&HostFrame>,
    grid: TokenGridShape,
    image_size: usize,
) -> Result<SparseTokenMask> {
    match config.mask_policy {
        MaskPolicy::FullFrame => Ok(SparseTokenMask::all(grid.len())),
        MaskPolicy::PatchDiff => {
            let scores = patch_diff_scores(frame, previous, grid, image_size);
            topk_mask(&scores, config.density, grid.len())
        }
        MaskPolicy::AutogazeStream => {
            let scores = autogaze_stream_scores(frame_index, frame, previous, grid, image_size);
            topk_mask(&scores, config.density, grid.len())
        }
    }
}

fn patch_diff_scores(
    frame: &HostFrame,
    previous: Option<&HostFrame>,
    grid: TokenGridShape,
    image_size: usize,
) -> Vec<f32> {
    let Some(previous) = previous else {
        return center_prior_scores(grid);
    };
    let patch = image_size / grid.height.max(1);
    let mut scores = vec![0.0_f32; grid.len()];
    for row in 0..grid.height {
        for col in 0..grid.width {
            let mut sum = 0.0;
            for y in row * patch..((row + 1) * patch).min(image_size) {
                for x in col * patch..((col + 1) * patch).min(image_size) {
                    let index = y * image_size + x;
                    sum += (frame.gray[index] - previous.gray[index]).abs();
                }
            }
            scores[row * grid.width + col] = sum / (patch * patch).max(1) as f32;
        }
    }
    scores
}

fn center_prior_scores(grid: TokenGridShape) -> Vec<f32> {
    let center_row = (grid.height.saturating_sub(1)) as f32 * 0.5;
    let center_col = (grid.width.saturating_sub(1)) as f32 * 0.5;
    let sigma = (grid.height.max(grid.width) as f32 * 0.28).max(1.0);
    (0..grid.height)
        .flat_map(|row| {
            (0..grid.width).map(move |col| {
                let dr = row as f32 - center_row;
                let dc = col as f32 - center_col;
                (-(dr * dr + dc * dc) / (2.0 * sigma * sigma)).exp()
            })
        })
        .collect()
}

fn autogaze_stream_scores(
    frame_index: usize,
    frame: &HostFrame,
    previous: Option<&HostFrame>,
    grid: TokenGridShape,
    image_size: usize,
) -> Vec<f32> {
    let motion = normalize01(&patch_diff_scores(frame, previous, grid, image_size));
    let center = normalize01(&center_prior_scores(grid));
    let phase = frame_index as f32 * 0.09;
    let gaze_row = (phase.sin() * 0.32 + 0.5) * grid.height.saturating_sub(1) as f32;
    let gaze_col = ((phase * 0.77).cos() * 0.32 + 0.5) * grid.width.saturating_sub(1) as f32;
    let sigma = (grid.height.max(grid.width) as f32 * 0.25).max(1.0);
    let gaze = normalize01(
        &(0..grid.height)
            .flat_map(|row| {
                (0..grid.width).map(move |col| {
                    let dr = row as f32 - gaze_row;
                    let dc = col as f32 - gaze_col;
                    (-(dr * dr + dc * dc) / (2.0 * sigma * sigma)).exp()
                })
            })
            .collect::<Vec<_>>(),
    );
    motion
        .iter()
        .zip(gaze.iter())
        .zip(center.iter())
        .map(|((motion, gaze), center)| 0.52 * motion + 0.34 * gaze + 0.14 * center)
        .collect()
}

fn normalize01(values: &[f32]) -> Vec<f32> {
    let lo = values
        .iter()
        .copied()
        .fold(f32::INFINITY, |acc, value| acc.min(value));
    let hi = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
    if hi <= lo + 1.0e-8 {
        return vec![0.0; values.len()];
    }
    values
        .iter()
        .map(|value| (value - lo) / (hi - lo))
        .collect()
}

fn topk_mask(scores: &[f32], density: f32, dense_len: usize) -> Result<SparseTokenMask> {
    let keep = ((dense_len as f32) * density.clamp(0.0, 1.0)).round() as usize;
    let keep = keep.max(1).min(dense_len);
    let mut ranked = scores
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f32)>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    SparseTokenMask::new(
        ranked
            .into_iter()
            .take(keep)
            .map(|(index, _)| index)
            .collect(),
        dense_len,
    )
}

fn tensor_nchw_rgb8(
    tensor: Tensor<GalleryBackend, 4>,
    output_size: [usize; 2],
    nearest_resize: bool,
) -> Result<Vec<u8>> {
    let [batch, channels, height, width] = tensor.shape().dims::<4>();
    ensure!(batch == 1, "display tensor batch must be one");
    ensure!(
        channels >= 3,
        "display tensor must have at least three channels"
    );
    let values = tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("read display tensor: {err}"))?;
    let mut rgb = vec![0_u8; height * width * 3];
    let plane = height * width;
    for y in 0..height {
        for x in 0..width {
            let src = y * width + x;
            let dst = src * 3;
            rgb[dst] = to_u8(values[src]);
            rgb[dst + 1] = to_u8(values[plane + src]);
            rgb[dst + 2] = to_u8(values[plane * 2 + src]);
        }
    }
    if [height, width] == output_size {
        return Ok(rgb);
    }
    let image = RgbImage::from_raw(width as u32, height as u32, rgb)
        .context("construct low-res RGB image")?;
    let filter = if nearest_resize {
        FilterType::Nearest
    } else {
        FilterType::CatmullRom
    };
    Ok(
        image::imageops::resize(&image, output_size[1] as u32, output_size[0] as u32, filter)
            .into_raw(),
    )
}

fn to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn mask_overlay(
    rgb: &[u8],
    mask: &SparseTokenMask,
    grid: TokenGridShape,
    image_size: usize,
) -> Vec<u8> {
    let mut out = rgb
        .chunks_exact(3)
        .flat_map(|px| {
            [
                (px[0] as f32 * 0.42).round() as u8,
                (px[1] as f32 * 0.42).round() as u8,
                (px[2] as f32 * 0.42).round() as u8,
            ]
        })
        .collect::<Vec<_>>();
    let patch_h = image_size / grid.height.max(1);
    let patch_w = image_size / grid.width.max(1);
    let mut active = vec![false; grid.len()];
    for &token in mask.indices() {
        active[token] = true;
    }
    for row in 0..grid.height {
        for col in 0..grid.width {
            let token = row * grid.width + col;
            let color = if active[token] {
                [24, 211, 178]
            } else {
                [50, 60, 72]
            };
            let y0 = row * patch_h;
            let y1 = if row + 1 == grid.height {
                image_size
            } else {
                (row + 1) * patch_h
            };
            let x0 = col * patch_w;
            let x1 = if col + 1 == grid.width {
                image_size
            } else {
                (col + 1) * patch_w
            };
            if active[token] {
                for y in y0..y1 {
                    for x in x0..x1 {
                        let offset = (y * image_size + x) * 3;
                        out[offset] = ((out[offset] as f32 * 0.36) + 24.0 * 0.64) as u8;
                        out[offset + 1] = ((out[offset + 1] as f32 * 0.36) + 211.0 * 0.64) as u8;
                        out[offset + 2] = ((out[offset + 2] as f32 * 0.36) + 178.0 * 0.64) as u8;
                    }
                }
            }
            for x in x0..x1 {
                set_rgb(&mut out, image_size, y0, x, color);
                set_rgb(&mut out, image_size, y1.saturating_sub(1), x, color);
            }
            for y in y0..y1 {
                set_rgb(&mut out, image_size, y, x0, color);
                set_rgb(&mut out, image_size, y, x1.saturating_sub(1), color);
            }
        }
    }
    out
}

fn set_rgb(out: &mut [u8], width: usize, y: usize, x: usize, color: [u8; 3]) {
    let offset = (y * width + x) * 3;
    out[offset..offset + 3].copy_from_slice(&color);
}

fn write_input_video(
    output: &Path,
    sample: &SampleWindow,
    frames: &[HostFrame],
    fps: usize,
    image_size: usize,
) -> Result<String> {
    let path = output
        .join("videos")
        .join(&sample.sample_id)
        .join("input.mp4");
    let input = frames
        .iter()
        .map(|frame| frame.rgb.clone())
        .collect::<Vec<_>>();
    write_mp4(&path, &input, image_size, image_size, fps)?;
    Ok(relpath(&path, output))
}

fn write_mp4(
    path: &Path,
    frames: &[Vec<u8>],
    width: usize,
    height: usize,
    fps: usize,
) -> Result<f64> {
    ensure!(
        !frames.is_empty(),
        "cannot write empty video {}",
        path.display()
    );
    fs::create_dir_all(path.parent().context("video path has no parent")?)?;
    let start = std::time::Instant::now();
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{width}x{height}"),
            "-r",
            &fps.to_string(),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ])
        .arg(path)
        .stdin(Stdio::piped())
        .spawn()
        .context("spawn ffmpeg")?;
    {
        let stdin = child.stdin.as_mut().context("open ffmpeg stdin")?;
        for frame in frames {
            ensure!(
                frame.len() == width * height * 3,
                "frame size mismatch while writing {}",
                path.display()
            );
            stdin.write_all(frame)?;
        }
    }
    let status = child.wait()?;
    if !status.success() {
        bail!("ffmpeg failed for {} with {status}", path.display());
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0)
}

fn relpath(path: &Path, base: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn write_manifest(output: &Path, manifest: &Manifest) -> Result<()> {
    fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(manifest)?,
    )?;
    Ok(())
}

fn write_html(output: &Path, manifest: &Manifest) -> Result<()> {
    let mut body = String::new();
    for sample in &manifest.samples {
        let mut configs = String::new();
        for config in &manifest.configs {
            let result = sample
                .configs
                .get(config.config_id)
                .with_context(|| format!("missing config {}", config.config_id))?;
            configs.push_str(&format!(
                r#"
<article class="config">
  <h3><span>{}</span><span class="pill">{} / {:?}</span></h3>
  <div class="videos">
    <figure><video controls loop muted preload="metadata" src="{}"></video><figcaption>sparse token mask</figcaption></figure>
    <figure><video controls loop muted preload="metadata" src="{}"></video><figcaption>low-res token-cache PCA</figcaption></figure>
    <figure><video controls loop muted preload="metadata" src="{}"></video><figcaption>high-res AnyUp PCA</figcaption></figure>
  </div>
  <div class="metrics">
    <span>density {:.3}</span><span>mask {:.1} ms</span><span>pipeline {:.1} ms</span><span>encode {:.1} ms</span><span>anyup {:.1} ms</span>
  </div>
</article>
"#,
                escape(config.config_id),
                escape(config.encoder_path),
                config.mask_policy,
                escape(result.videos.get("mask").map(String::as_str).unwrap_or_default()),
                escape(result.videos.get("low_pca").map(String::as_str).unwrap_or_default()),
                escape(result.videos.get("high_pca").map(String::as_str).unwrap_or_default()),
                result.stats.observed_density,
                result.stats.mask_ms,
                result.stats.pipeline_ms,
                result.stats.encode_ms,
                result.stats.anyup_ms,
            ));
        }
        body.push_str(&format!(
            r#"
<section class="sample">
  <div class="sample-head">
    <video controls loop muted preload="metadata" src="{}"></video>
    <div class="meta">
      <h2>{}: {}</h2>
      <p>Input clip and artifacts were generated through the Burn sparse feature-frame pipeline. Sparse configs update only selected token positions before the interframe feature cache, PCA projection, and AnyUp high-res display.</p>
      <dl>
        <dt>split</dt><dd>{}</dd>
        <dt>start</dt><dd>{}</dd>
        <dt>frames</dt><dd>{}</dd>
        <dt>stride</dt><dd>{}</dd>
      </dl>
    </div>
  </div>
  <div class="config-grid">{}</div>
</section>
"#,
            escape(&sample.input_video),
            escape(&sample.window.sample_id),
            escape(&sample.window.clip_name),
            escape(&sample.window.dataset_split),
            sample.window.start_frame,
            sample.window.frame_count,
            sample.window.stride,
            configs,
        ));
    }

    let summary = &manifest.summary;
    let page = format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Burn JEPA E2E Sparse Feature PCA Gallery</title>
  <style>{}</style>
</head>
<body>
  <header>
    <h1>Burn JEPA E2E sparse feature PCA gallery</h1>
    <p>{}. Dense and sparse configs are rendered by the actual Burn feature-frame pipeline. {}</p>
    <div class="summary">
      <div class="stat"><strong>{}</strong><span>samples</span></div>
      <div class="stat"><strong>{}</strong><span>configs per sample</span></div>
      <div class="stat"><strong>{}</strong><span>MP4 artifacts</span></div>
      <div class="stat"><strong>{} px</strong><span>render size</span></div>
      <div class="stat"><strong>{}x{}</strong><span>token grid</span></div>
    </div>
  </header>
  <main>{}</main>
</body>
</html>
"#,
        css(),
        escape(summary.dataset),
        escape(summary.pipeline_note),
        summary.sample_count,
        summary.config_count,
        summary.mp4_count,
        summary.image_size,
        summary.token_grid,
        summary.token_grid,
        body,
    );
    fs::write(output.join("index.html"), page)?;
    Ok(())
}

fn css() -> &'static str {
    r#"
:root { color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; background: #0b0d10; color: #eef2f6; }
header { position: sticky; top: 0; z-index: 2; background: rgba(11,13,16,0.96); border-bottom: 1px solid #28313c; padding: 16px 22px; }
h1 { margin: 0 0 6px; font-size: 22px; line-height: 1.2; }
p { margin: 0; color: #aeb9c7; line-height: 1.45; }
main { padding: 20px; display: grid; gap: 22px; }
.summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-top: 12px; }
.stat { border: 1px solid #28313c; border-radius: 8px; padding: 10px 12px; background: #121720; }
.stat strong { display: block; font-size: 18px; color: #fff; }
.sample { border: 1px solid #28313c; border-radius: 8px; overflow: hidden; background: #11161d; }
.sample-head { display: grid; grid-template-columns: minmax(180px, 280px) 1fr; gap: 16px; padding: 14px; border-bottom: 1px solid #28313c; }
.sample-head video { width: 100%; border-radius: 6px; background: #050607; }
.meta { display: grid; gap: 8px; align-content: start; }
.meta h2 { margin: 0; font-size: 18px; }
.meta dl { display: grid; grid-template-columns: auto 1fr; gap: 5px 12px; margin: 0; color: #c7d0dc; }
.meta dt { color: #7f8da0; }
.config-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; padding: 14px; }
.config { border: 1px solid #26303a; border-radius: 8px; background: #0c1118; overflow: hidden; }
.config h3 { margin: 0; padding: 10px 12px; font-size: 14px; border-bottom: 1px solid #26303a; display: flex; justify-content: space-between; gap: 8px; }
.pill { font-size: 12px; color: #cbd5e1; border: 1px solid #3a4655; padding: 1px 7px; border-radius: 999px; white-space: nowrap; }
.videos { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; padding: 10px; }
figure { margin: 0; display: grid; gap: 6px; }
figcaption { color: #94a3b8; font-size: 12px; }
video { display: block; max-width: 100%; background: #050607; }
.metrics { padding: 0 10px 10px; color: #aeb9c7; font-size: 12px; display: flex; flex-wrap: wrap; gap: 6px; }
.metrics span { border: 1px solid #26303a; border-radius: 999px; padding: 2px 7px; }
@media (max-width: 720px) {
  header { position: static; }
  main { padding: 12px; }
  .sample-head { grid-template-columns: 1fr; }
  .videos { grid-template-columns: 1fr; }
}
"#
}

fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
