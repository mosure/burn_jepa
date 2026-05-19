#![cfg_attr(
    not(any(
        feature = "ndarray",
        feature = "wgpu",
        feature = "webgpu",
        feature = "cuda"
    )),
    allow(dead_code, unused_imports, unused_variables)
)]

use anyhow::{Context, Result, bail, ensure};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_jepa::{
    AnyUp, AnyUpConfig, BurnJepaPackageModelKind, BurnJepaPipelinePackageManifest,
    BurnJepaReconstructionPackageManifest, FeatureFrameJepaEncoder, FeatureFrameMeasureConfig,
    FeatureFramePipeline, FeatureFramePipelineConfig, FeatureFrameRequest, FeatureFrameSparseMasks,
    FeatureFrameViewerConfig, FeaturePcaUpdateConfig, SparseTokenMask, TokenGridShape,
    VJEPA_IMAGE_MEAN, VJEPA_IMAGE_STD, center_prior_mask, finalize_patch_diff_masks,
    load_jepa_reconstruction_burnpack_parts, load_ttt_burnpack_parts, load_vjepa_burnpack_parts,
    patch_diff_context_mask_from_scores, patch_diff_scores_from_rgba, read_parts_manifest,
    reconstruction_psnr_scalar, resolve_package_manifest_entry_path, resolve_part_entry_path,
    shape_prewarm_masks,
};
use clap::Parser;
use image::{ImageReader, RgbImage, imageops::FilterType};
use serde::Serialize;
use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::str::FromStr;

#[derive(Parser, Debug)]
#[command(about = "Measure JEPA reconstruction PSNR from the E2E sparse low-res feature cache")]
struct Args {
    #[arg(long, default_value = "target/burn_jepa/vjepa2_1_base/manifest.json")]
    jepa_manifest: PathBuf,
    #[arg(
        long,
        default_value = "target/burn_jepa_reconstruction/low_res_v1/manifest.json"
    )]
    reconstruction_manifest: PathBuf,
    #[arg(
        long,
        default_value = "target/burn_jepa_reconstruction_train_frames/video_023"
    )]
    frame_dir: PathBuf,
    #[arg(long = "image")]
    images: Vec<PathBuf>,
    #[arg(long, default_value_t = 256)]
    image_size: usize,
    #[arg(long, default_value_t = 32)]
    frames: usize,
    #[arg(long, default_value_t = 10)]
    fps: usize,
    #[arg(
        long,
        default_value = "dense,patch-diff,patch-diff-keyframe",
        value_delimiter = ','
    )]
    modes: Vec<EvalMode>,
    #[arg(long, default_value_t = 0.03)]
    patch_diff_threshold: f32,
    #[arg(long, default_value_t = 1)]
    patch_diff_dilation: usize,
    #[arg(long, default_value_t = 1.0)]
    context_density: f32,
    #[arg(long, default_value_t = 0.0)]
    min_context_density: f32,
    #[arg(long, default_value_t = 1.0)]
    bootstrap_context_density: f32,
    #[arg(long, default_value_t = 0.60)]
    dense_fallback_density: f32,
    #[arg(long, default_value_t = 16)]
    dense_keyframe_every: usize,
    #[arg(long, default_value_t = false)]
    exact_sparse_encode: bool,
    #[arg(long, default_value_t = false)]
    sync_measurements: bool,
    #[arg(long, default_value_t = false)]
    no_prewarm: bool,
    #[arg(long, default_value = "target/burn-jepa-reconstruction-e2e")]
    output: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum EvalMode {
    Dense,
    PatchDiff,
    PatchDiffKeyframe,
}

impl EvalMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::PatchDiff => "patch_diff",
            Self::PatchDiffKeyframe => "patch_diff_keyframe",
        }
    }
}

impl fmt::Display for EvalMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for EvalMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "dense" | "full" | "full-frame" | "full_frame" => Ok(Self::Dense),
            "patch-diff" | "patch_diff" | "sparse" => Ok(Self::PatchDiff),
            "patch-diff-keyframe" | "patch_diff_keyframe" | "keyframe" => {
                Ok(Self::PatchDiffKeyframe)
            }
            other => bail!(
                "unknown reconstruction E2E mode `{other}`; expected dense, patch-diff, or patch-diff-keyframe"
            ),
        }
    }
}

#[derive(Clone)]
struct HostFrame<B: Backend> {
    path: PathBuf,
    rgb: Vec<u8>,
    rgba: Vec<u8>,
    input: Tensor<B, 4>,
    target: Tensor<B, 4>,
}

#[derive(Clone, Debug, Serialize)]
struct FrameMetric {
    mode: EvalMode,
    frame_index: usize,
    source: String,
    write_tokens: usize,
    encode_tokens: usize,
    dense_tokens: usize,
    write_density: f32,
    encode_density: f32,
    psnr_db: f64,
    encode_ms: f64,
    cache_update_ms: f64,
    token_view_ms: f64,
    pipeline_ms: f64,
    reconstruction_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
struct ModeSummary {
    mode: EvalMode,
    frames: usize,
    mean_psnr_db: f64,
    min_psnr_db: f64,
    p05_psnr_db: f64,
    mean_write_density: f32,
    mean_encode_density: f32,
    mean_pipeline_ms: f64,
    mean_encode_ms: f64,
    mean_cache_update_ms: f64,
    mean_reconstruction_ms: f64,
    reconstruction_video: String,
    mask_video: String,
}

#[derive(Debug, Serialize)]
struct E2eReport {
    backend: String,
    jepa_manifest: String,
    reconstruction_manifest: String,
    image_size: usize,
    grid: ReportGrid,
    frame_count: usize,
    input_video: String,
    modes: Vec<ModeSummary>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct ReportGrid {
    depth: usize,
    height: usize,
    width: usize,
}

impl From<TokenGridShape> for ReportGrid {
    fn from(value: TokenGridShape) -> Self {
        Self {
            depth: value.depth,
            height: value.height,
            width: value.width,
        }
    }
}

#[allow(unreachable_code)]
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.image_size > 0 && args.image_size.is_multiple_of(16),
        "--image-size must be a positive multiple of 16"
    );

    #[cfg(feature = "cuda")]
    {
        if let Err(reason) =
            burn_jepa::runtime::cuda_runtime_preflight(burn_jepa::runtime::CUDA_TRAIN_FORCE_ENV)
        {
            eprintln!("skipping cuda reconstruction E2E backend: {reason}");
        } else {
            return run::<burn::backend::Cuda<f32, i32>>("cuda", args, Default::default());
        }
    }

    #[cfg(all(any(feature = "wgpu", feature = "webgpu"), not(target_arch = "wasm32")))]
    {
        return run::<burn::backend::Wgpu<f32, i32>>("wgpu", args, Default::default());
    }

    #[cfg(feature = "ndarray")]
    {
        return run::<burn::backend::NdArray<f32>>("ndarray", args, Default::default());
    }

    bail!("enable at least one backend feature: ndarray, wgpu/webgpu, or cuda")
}

fn run<B>(backend: &str, args: Args, device: B::Device) -> Result<()>
where
    B: Backend,
    B::Device: Clone,
{
    fs::create_dir_all(&args.output)
        .with_context(|| format!("create output dir {}", args.output.display()))?;
    let frame_paths = resolve_frame_paths(&args)?;
    ensure!(!frame_paths.is_empty(), "no input frames found");

    eprintln!(
        "loading {} frames at {}px on {backend}",
        frame_paths.len(),
        args.image_size
    );
    let frames = frame_paths
        .iter()
        .map(|path| load_frame::<B>(path, args.image_size, &device))
        .collect::<Result<Vec<_>>>()?;

    let jepa_manifest = read_jepa_manifest(&args.jepa_manifest)?;
    let reconstruction_manifest = read_reconstruction_manifest(&args.reconstruction_manifest)?;
    ensure!(
        jepa_manifest.jepa_config.encoder.embed_dim
            == reconstruction_manifest.reconstruction_config.input_dim,
        "V-JEPA embed dim {} does not match reconstruction input dim {}",
        jepa_manifest.jepa_config.encoder.embed_dim,
        reconstruction_manifest.reconstruction_config.input_dim
    );

    let grid = TokenGridShape::new(
        1,
        args.image_size / jepa_manifest.jepa_config.patch_size.max(1),
        args.image_size / jepa_manifest.jepa_config.patch_size.max(1),
    );
    ensure!(
        grid.height == grid.width,
        "reconstruction E2E example expects square token grid"
    );

    let encoder = load_encoder::<B>(&args.jepa_manifest, &jepa_manifest, &device)?;
    let parts = read_package_parts(
        &args.reconstruction_manifest,
        &reconstruction_manifest.parts_manifest,
    )?;
    let (decoder, reconstruction_apply) = load_jepa_reconstruction_burnpack_parts::<B>(
        &reconstruction_manifest.reconstruction_config,
        &parts,
        &device,
    )?;
    ensure!(
        reconstruction_apply.errors.is_empty() && !reconstruction_apply.applied.is_empty(),
        "reconstruction burnpack load failed: {:?}",
        reconstruction_apply.errors
    );

    let anyup = AnyUp::new(AnyUpConfig::tiny_for_tests(), &device)?;
    let mut pipeline_config = FeatureFramePipelineConfig {
        measurement: FeatureFrameMeasureConfig {
            enabled: true,
            sync_backend: args.sync_measurements,
        },
        pca_update: FeaturePcaUpdateConfig::disabled(),
        update_pca_online: false,
        ..FeatureFramePipelineConfig::default()
    };
    pipeline_config.ttt_runtime.enabled =
        matches!(jepa_manifest.model_kind, BurnJepaPackageModelKind::Ttt);
    let mut pipeline = FeatureFramePipeline::<B>::new_with_encoder(
        encoder,
        anyup,
        &jepa_manifest.jepa_config,
        pipeline_config,
        1,
        [args.image_size, args.image_size],
        &device,
    )?;

    let input_video = args.output.join("input.mp4");
    write_mp4(
        &input_video,
        &frames
            .iter()
            .map(|frame| frame.rgb.clone())
            .collect::<Vec<_>>(),
        args.image_size,
        args.image_size,
        args.fps,
    )?;

    let mut viewer_config = FeatureFrameViewerConfig {
        image_size: args.image_size,
        patch_diff_threshold: args.patch_diff_threshold,
        patch_diff_dilation_tiles: args.patch_diff_dilation,
        context_density: args.context_density,
        min_context_density: args.min_context_density,
        bootstrap_context_density: args.bootstrap_context_density,
        patch_diff_dense_fallback_density: args.dense_fallback_density,
        sparse_encode_mode: if args.exact_sparse_encode {
            burn_jepa::FeatureFrameSparseEncodeMode::Exact
        } else {
            burn_jepa::FeatureFrameSparseEncodeMode::BucketedContext
        },
        ..FeatureFrameViewerConfig::default()
    };
    viewer_config.measure_stages = true;
    viewer_config.sync_measurements = args.sync_measurements;

    let mut all_metrics = Vec::new();
    let mut summaries = Vec::new();
    if !args.no_prewarm {
        prewarm_decoder(
            &decoder,
            reconstruction_manifest.reconstruction_config.input_dim,
            grid,
            args.image_size,
            &device,
        )?;
    }
    for mode in &args.modes {
        if !args.no_prewarm {
            prewarm_pipeline(&mut pipeline, &frames[0], grid, &viewer_config)?;
        }
        pipeline.reset();
        let result = run_mode(
            &args,
            *mode,
            &frames,
            grid,
            &viewer_config,
            &mut pipeline,
            &decoder,
        )?;
        all_metrics.extend(result.metrics);
        summaries.push(result.summary);
    }

    write_metrics_csv(&args.output.join("frame_metrics.csv"), &all_metrics)?;
    let report = E2eReport {
        backend: backend.to_string(),
        jepa_manifest: args.jepa_manifest.to_string_lossy().to_string(),
        reconstruction_manifest: args.reconstruction_manifest.to_string_lossy().to_string(),
        image_size: args.image_size,
        grid: grid.into(),
        frame_count: frames.len(),
        input_video: relpath(&input_video, &args.output),
        modes: summaries,
    };
    fs::write(
        args.output.join("summary.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    write_markdown_report(&args.output.join("report.md"), &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

struct ModeResult {
    metrics: Vec<FrameMetric>,
    summary: ModeSummary,
}

fn prewarm_pipeline<B: Backend>(
    pipeline: &mut FeatureFramePipeline<B>,
    frame: &HostFrame<B>,
    grid: TokenGridShape,
    viewer_config: &FeatureFrameViewerConfig,
) -> Result<()> {
    let mut masks = shape_prewarm_masks(grid, viewer_config);
    masks.push(SparseTokenMask::all(grid.len()));
    masks.sort_by_key(|mask| (mask.len(), mask.is_dense_ordered()));
    masks.dedup_by(|left, right| {
        left.len() == right.len() && left.is_dense_ordered() == right.is_dense_ordered()
    });
    for mask in masks {
        let _ = pipeline.step_image_with_mask_nodes_measured(
            frame.input.clone(),
            &mask,
            FeatureFrameRequest::none(),
        )?;
    }
    pipeline.reset();
    Ok(())
}

fn prewarm_decoder<B: Backend>(
    decoder: &burn_jepa::JepaReconstructionDecoder<B>,
    input_dim: usize,
    grid: TokenGridShape,
    image_size: usize,
    device: &B::Device,
) -> Result<()> {
    let features = Tensor::<B, 4>::zeros([1, input_dim, grid.height, grid.width], device);
    let output = decoder.forward_to_size(features, [image_size, image_size]);
    let _ = output
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("read reconstruction prewarm output: {err:?}"))?;
    Ok(())
}

fn run_mode<B: Backend>(
    args: &Args,
    mode: EvalMode,
    frames: &[HostFrame<B>],
    grid: TokenGridShape,
    viewer_config: &FeatureFrameViewerConfig,
    pipeline: &mut FeatureFramePipeline<B>,
    decoder: &burn_jepa::JepaReconstructionDecoder<B>,
) -> Result<ModeResult> {
    let mut metrics = Vec::with_capacity(frames.len());
    let mut recon_frames = Vec::with_capacity(frames.len());
    let mut mask_frames = Vec::with_capacity(frames.len());
    let mut previous: Option<&HostFrame<B>> = None;

    for (frame_index, frame) in frames.iter().enumerate() {
        let masks = frame_masks(
            mode,
            frame_index,
            frame,
            previous,
            grid,
            viewer_config,
            args.dense_keyframe_every,
            args.image_size,
        )?;
        mask_frames.push(mask_overlay(
            &frame.rgb,
            &masks.write_mask,
            grid,
            args.image_size,
        ));

        let measured = if masks.write_mask == masks.encode_mask {
            pipeline.step_image_with_mask_nodes_measured(
                frame.input.clone(),
                &masks.write_mask,
                FeatureFrameRequest::none(),
            )?
        } else {
            pipeline.step_image_with_encode_write_masks_nodes_measured(
                frame.input.clone(),
                &masks.encode_mask,
                &masks.write_mask,
                FeatureFrameRequest::none(),
            )?
        };

        let reconstruction_start = std::time::Instant::now();
        let reconstruction = decoder.forward_to_size(
            measured.output.low_res.features.clone(),
            [args.image_size, args.image_size],
        );
        let psnr_db = reconstruction_psnr_scalar(reconstruction.clone(), frame.target.clone(), 1.0)
            .context("read reconstruction PSNR scalar")?;
        let reconstruction_ms = reconstruction_start.elapsed().as_secs_f64() * 1000.0;
        recon_frames.push(tensor_nchw_rgb8(
            reconstruction,
            [args.image_size, args.image_size],
        )?);

        let dense_tokens = masks.write_mask.dense_len();
        metrics.push(FrameMetric {
            mode,
            frame_index,
            source: frame.path.to_string_lossy().to_string(),
            write_tokens: masks.write_mask.len(),
            encode_tokens: masks.encode_mask.len(),
            dense_tokens,
            write_density: masks.write_mask.len() as f32 / dense_tokens.max(1) as f32,
            encode_density: masks.encode_mask.len() as f32 / dense_tokens.max(1) as f32,
            psnr_db,
            encode_ms: measured.metrics.encode_us as f64 / 1000.0,
            cache_update_ms: measured.metrics.cache_update_us as f64 / 1000.0,
            token_view_ms: measured.metrics.token_view_us as f64 / 1000.0,
            pipeline_ms: measured.metrics.total_us as f64 / 1000.0,
            reconstruction_ms,
        });
        previous = Some(frame);
    }

    let mode_dir = args.output.join(mode.as_str());
    let reconstruction_video = mode_dir.join("reconstruction.mp4");
    let mask_video = mode_dir.join("mask.mp4");
    write_mp4(
        &reconstruction_video,
        &recon_frames,
        args.image_size,
        args.image_size,
        args.fps,
    )?;
    write_mp4(
        &mask_video,
        &mask_frames,
        args.image_size,
        args.image_size,
        args.fps,
    )?;
    let summary = summarize_mode(
        mode,
        &metrics,
        relpath(&reconstruction_video, &args.output),
        relpath(&mask_video, &args.output),
    );
    Ok(ModeResult { metrics, summary })
}

fn frame_masks<B: Backend>(
    mode: EvalMode,
    frame_index: usize,
    frame: &HostFrame<B>,
    previous: Option<&HostFrame<B>>,
    grid: TokenGridShape,
    viewer_config: &FeatureFrameViewerConfig,
    dense_keyframe_every: usize,
    image_size: usize,
) -> Result<FeatureFrameSparseMasks> {
    if mode == EvalMode::Dense
        || (mode == EvalMode::PatchDiffKeyframe
            && dense_keyframe_every > 0
            && frame_index.is_multiple_of(dense_keyframe_every))
    {
        return Ok(FeatureFrameSparseMasks::same(SparseTokenMask::all(
            grid.len(),
        )));
    }
    let base_mask = if let Some(previous) = previous {
        let scores = patch_diff_scores_from_rgba(
            &previous.rgba,
            &frame.rgba,
            image_size,
            image_size / grid.width.max(1),
            grid,
        )?;
        patch_diff_context_mask_from_scores(
            scores,
            grid,
            &viewer_config.patch_diff_sparsity_config(grid),
        )?
    } else {
        center_prior_mask(grid, viewer_config.bootstrap_context_tokens(grid.len()))?
    };
    Ok(finalize_patch_diff_masks(base_mask, grid, viewer_config))
}

fn load_encoder<B: Backend>(
    manifest_path: &Path,
    manifest: &BurnJepaPipelinePackageManifest,
    device: &B::Device,
) -> Result<FeatureFrameJepaEncoder<B>> {
    let parts = read_package_parts(manifest_path, &manifest.parts_manifest)?;
    match manifest.model_kind {
        BurnJepaPackageModelKind::Base => {
            let (model, report) =
                load_vjepa_burnpack_parts::<B>(&manifest.jepa_config, &parts, device)?;
            ensure!(
                report.errors.is_empty() && !report.applied.is_empty(),
                "V-JEPA burnpack load failed: {:?}",
                report.errors
            );
            Ok(FeatureFrameJepaEncoder::base(model))
        }
        BurnJepaPackageModelKind::Ttt => {
            let ttt_config = manifest
                .ttt_config
                .clone()
                .context("TTT package manifest missing ttt_config")?;
            let (model, report) =
                load_ttt_burnpack_parts::<B>(&manifest.jepa_config, ttt_config, &parts, device)?;
            ensure!(
                report.errors.is_empty() && !report.applied.is_empty(),
                "TTT V-JEPA burnpack load failed: {:?}",
                report.errors
            );
            Ok(FeatureFrameJepaEncoder::ttt(model))
        }
    }
}

fn read_jepa_manifest(path: &Path) -> Result<BurnJepaPipelinePackageManifest> {
    let json = fs::read_to_string(path)
        .with_context(|| format!("read V-JEPA package manifest {}", path.display()))?;
    BurnJepaPipelinePackageManifest::from_json_str(&json)
        .with_context(|| format!("parse V-JEPA package manifest {}", path.display()))
}

fn read_reconstruction_manifest(path: &Path) -> Result<BurnJepaReconstructionPackageManifest> {
    let json = fs::read_to_string(path)
        .with_context(|| format!("read reconstruction package manifest {}", path.display()))?;
    BurnJepaReconstructionPackageManifest::from_json_str(&json)
        .with_context(|| format!("parse reconstruction package manifest {}", path.display()))
}

fn read_package_parts(manifest_path: &Path, parts_manifest_entry: &str) -> Result<Vec<Vec<u8>>> {
    let parts_manifest_path =
        resolve_package_manifest_entry_path(manifest_path, parts_manifest_entry)?;
    let parts_manifest = read_parts_manifest(&parts_manifest_path)?;
    ensure!(
        !parts_manifest.parts.is_empty(),
        "parts manifest {} has no parts",
        parts_manifest_path.display()
    );
    parts_manifest
        .parts
        .iter()
        .map(|part| {
            let path = resolve_part_entry_path(&parts_manifest_path, &part.path)?;
            fs::read(&path).with_context(|| format!("read burnpack part {}", path.display()))
        })
        .collect()
}

fn resolve_frame_paths(args: &Args) -> Result<Vec<PathBuf>> {
    let mut paths = if args.images.is_empty() {
        ensure!(
            args.frame_dir.exists(),
            "frame dir does not exist: {}",
            args.frame_dir.display()
        );
        let mut paths = fs::read_dir(&args.frame_dir)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| {
                        ext.eq_ignore_ascii_case("jpg")
                            || ext.eq_ignore_ascii_case("jpeg")
                            || ext.eq_ignore_ascii_case("png")
                    })
            })
            .collect::<Vec<_>>();
        paths.sort_by(|left, right| {
            frame_sort_key(left)
                .cmp(&frame_sort_key(right))
                .then(left.cmp(right))
        });
        paths
    } else {
        args.images.clone()
    };
    if args.frames > 0 {
        paths.truncate(args.frames);
    }
    Ok(paths)
}

fn frame_sort_key(path: &Path) -> usize {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| {
            let digits = stem
                .chars()
                .filter(|ch| ch.is_ascii_digit())
                .collect::<String>();
            digits.parse::<usize>().ok()
        })
        .unwrap_or(0)
}

fn load_frame<B: Backend>(
    path: &Path,
    image_size: usize,
    device: &B::Device,
) -> Result<HostFrame<B>> {
    let image = ImageReader::open(path)?.decode()?.to_rgb8();
    let (width, height) = image.dimensions();
    let crop = width.min(height);
    let left = (width - crop) / 2;
    let top = (height - crop) / 2;
    let cropped = image::imageops::crop_imm(&image, left, top, crop, crop).to_image();
    let resized = image::imageops::resize(
        &cropped,
        image_size as u32,
        image_size as u32,
        FilterType::Triangle,
    );
    let rgb = resized.into_raw();
    let mut rgba = Vec::with_capacity(image_size * image_size * 4);
    for pixel in rgb.chunks_exact(3) {
        rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
    }
    let pixels = image_size * image_size;
    let mut target_values = vec![0.0_f32; 3 * pixels];
    let mut input_values = vec![0.0_f32; 3 * pixels];
    for index in 0..pixels {
        for channel in 0..3 {
            let value = rgb[index * 3 + channel] as f32 / 255.0;
            target_values[channel * pixels + index] = value;
            input_values[channel * pixels + index] =
                (value - VJEPA_IMAGE_MEAN[channel]) / VJEPA_IMAGE_STD[channel];
        }
    }
    let target = Tensor::<B, 4>::from_data(
        TensorData::new(target_values, [1, 3, image_size, image_size]),
        device,
    );
    let input = Tensor::<B, 4>::from_data(
        TensorData::new(input_values, [1, 3, image_size, image_size]),
        device,
    );
    Ok(HostFrame {
        path: path.to_path_buf(),
        rgb,
        rgba,
        input,
        target,
    })
}

fn summarize_mode(
    mode: EvalMode,
    metrics: &[FrameMetric],
    reconstruction_video: String,
    mask_video: String,
) -> ModeSummary {
    let frames = metrics.len();
    let mut psnr = metrics
        .iter()
        .map(|metric| metric.psnr_db)
        .collect::<Vec<_>>();
    psnr.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let mean = |values: Vec<f64>| {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    };
    let mean_psnr_db = mean(metrics.iter().map(|metric| metric.psnr_db).collect());
    let p05_index = ((frames as f32 * 0.05).floor() as usize).min(frames.saturating_sub(1));
    ModeSummary {
        mode,
        frames,
        mean_psnr_db,
        min_psnr_db: psnr.first().copied().unwrap_or(0.0),
        p05_psnr_db: psnr.get(p05_index).copied().unwrap_or(0.0),
        mean_write_density: mean(
            metrics
                .iter()
                .map(|metric| metric.write_density as f64)
                .collect(),
        ) as f32,
        mean_encode_density: mean(
            metrics
                .iter()
                .map(|metric| metric.encode_density as f64)
                .collect(),
        ) as f32,
        mean_pipeline_ms: mean(metrics.iter().map(|metric| metric.pipeline_ms).collect()),
        mean_encode_ms: mean(metrics.iter().map(|metric| metric.encode_ms).collect()),
        mean_cache_update_ms: mean(
            metrics
                .iter()
                .map(|metric| metric.cache_update_ms)
                .collect(),
        ),
        mean_reconstruction_ms: mean(
            metrics
                .iter()
                .map(|metric| metric.reconstruction_ms)
                .collect(),
        ),
        reconstruction_video,
        mask_video,
    }
}

fn tensor_nchw_rgb8<B: Backend>(tensor: Tensor<B, 4>, output_size: [usize; 2]) -> Result<Vec<u8>> {
    let [batch, channels, height, width] = tensor.shape().dims::<4>();
    ensure!(batch == 1, "RGB tensor batch must be one");
    ensure!(
        channels >= 3,
        "RGB tensor must have at least three channels"
    );
    let values = tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("read RGB tensor values: {err:?}"))?;
    let plane = height * width;
    let mut rgb = vec![0_u8; plane * 3];
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
        .context("construct resized RGB image")?;
    Ok(image::imageops::resize(
        &image,
        output_size[1] as u32,
        output_size[0] as u32,
        FilterType::Triangle,
    )
    .into_raw())
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
                (px[0] as f32 * 0.45).round() as u8,
                (px[1] as f32 * 0.45).round() as u8,
                (px[2] as f32 * 0.45).round() as u8,
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
                [45, 55, 65]
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
                        out[offset] = ((out[offset] as f32 * 0.35) + 24.0 * 0.65) as u8;
                        out[offset + 1] = ((out[offset + 1] as f32 * 0.35) + 211.0 * 0.65) as u8;
                        out[offset + 2] = ((out[offset + 2] as f32 * 0.35) + 178.0 * 0.65) as u8;
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

fn write_mp4(
    path: &Path,
    frames: &[Vec<u8>],
    width: usize,
    height: usize,
    fps: usize,
) -> Result<()> {
    ensure!(
        !frames.is_empty(),
        "cannot write empty video {}",
        path.display()
    );
    fs::create_dir_all(path.parent().context("video path has no parent")?)?;
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
    ensure!(
        status.success(),
        "ffmpeg failed for {} with {status}",
        path.display()
    );
    Ok(())
}

fn write_metrics_csv(path: &Path, metrics: &[FrameMetric]) -> Result<()> {
    let mut text = String::from(
        "mode,frame_index,source,write_tokens,encode_tokens,dense_tokens,write_density,encode_density,psnr_db,encode_ms,cache_update_ms,token_view_ms,pipeline_ms,reconstruction_ms\n",
    );
    for metric in metrics {
        text.push_str(&format!(
            "{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            metric.mode,
            metric.frame_index,
            csv_escape(&metric.source),
            metric.write_tokens,
            metric.encode_tokens,
            metric.dense_tokens,
            metric.write_density,
            metric.encode_density,
            metric.psnr_db,
            metric.encode_ms,
            metric.cache_update_ms,
            metric.token_view_ms,
            metric.pipeline_ms,
            metric.reconstruction_ms,
        ));
    }
    fs::write(path, text).with_context(|| format!("write CSV {}", path.display()))
}

fn write_markdown_report(path: &Path, report: &E2eReport) -> Result<()> {
    let mut text = String::new();
    text.push_str("# reconstruction e2e report\n\n");
    text.push_str(&format!(
        "- backend: `{}`\n- image size: `{}`\n- token grid: `{}x{}`\n- frames: `{}`\n- input: `{}`\n\n",
        report.backend,
        report.image_size,
        report.grid.height,
        report.grid.width,
        report.frame_count,
        report.input_video,
    ));
    text.push_str("| mode | mean psnr | p05 psnr | min psnr | write density | encode density | pipeline ms | recon ms |\n");
    text.push_str("|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for mode in &report.modes {
        text.push_str(&format!(
            "| {} | {:.2} | {:.2} | {:.2} | {:.1}% | {:.1}% | {:.2} | {:.2} |\n",
            mode.mode,
            mode.mean_psnr_db,
            mode.p05_psnr_db,
            mode.min_psnr_db,
            mode.mean_write_density * 100.0,
            mode.mean_encode_density * 100.0,
            mode.mean_pipeline_ms,
            mode.mean_reconstruction_ms,
        ));
    }
    text.push_str("\n## videos\n\n");
    for mode in &report.modes {
        text.push_str(&format!(
            "- `{}`: `{}`, `{}`\n",
            mode.mode, mode.reconstruction_video, mode.mask_video
        ));
    }
    fs::write(path, text).with_context(|| format!("write report {}", path.display()))
}

fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn relpath(path: &Path, base: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}
