use crate::{
    BurnJepaModelBootstrapConfig, BurnJepaModelProfile, BurnJepaPackageModelKind,
    BurnJepaPipelinePackageManifest, BurnJepaReconstructionModelProfile,
    BurnJepaReconstructionPackageManifest, DEFAULT_BURN_JEPA_MODEL_BASE_URL,
    DEFAULT_BURN_JEPA_RECONSTRUCTION_MODEL_BASE_URL, JepaReconstructionArchitecture,
    JepaReconstructionConfig, JepaReconstructionDecoder, JepaReconstructionOutputActivation,
    VJepa2_1Model, VJepaRgbaVideoShape, jepa_feature_tokens_to_nchw,
    reconstruction_color_moment_loss, reconstruction_gradient_mse, reconstruction_l1,
    reconstruction_mse, reconstruction_psnr_scalar,
};
use anyhow::{Context, Result, bail, ensure};
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};
use image::imageops::FilterType;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png"];
pub const DEFAULT_RECONSTRUCTION_DEVICE_CACHE_MAX_MIB: usize = 4096;
const TRAIN_EVAL_MAX_SAMPLES: usize = 16;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconstructionFeatureSource {
    /// Match the live feature-frame pipeline: one image through the image patch embed path.
    #[default]
    Image,
    /// Legacy target: repeated still frames through the temporal/video patch embed path.
    Video,
}

#[derive(Clone, Debug)]
pub struct ReconstructionTrainingOptions {
    pub backend: crate::JepaTrainBackend,
    pub jepa_manifest: Option<PathBuf>,
    pub jepa_model_profile: BurnJepaModelProfile,
    pub jepa_model_base_url: String,
    pub jepa_manifest_url: Option<String>,
    pub jepa_cache_dir: Option<PathBuf>,
    pub image_paths: Vec<PathBuf>,
    pub image_dirs: Vec<PathBuf>,
    pub image_size: usize,
    pub frames: usize,
    pub feature_source: ReconstructionFeatureSource,
    pub max_samples: usize,
    pub val_split: f32,
    pub steps: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub l1_loss_weight: f64,
    pub gradient_loss_weight: f64,
    pub color_loss_weight: f64,
    pub hidden_dim: usize,
    pub reconstruction_architecture: JepaReconstructionArchitecture,
    pub min_channels: usize,
    pub residual_blocks_per_scale: usize,
    pub convnext_expansion: usize,
    pub residual_scale: f64,
    pub output_activation: JepaReconstructionOutputActivation,
    pub device_cache_max_mib: usize,
    pub log_interval: usize,
    pub seed: u64,
    pub output: PathBuf,
    pub shard_mib: u64,
    pub overwrite_shards: bool,
    pub reconstruction_model_profile: BurnJepaReconstructionModelProfile,
    pub reconstruction_model_base_url: String,
    pub deploy_dir: PathBuf,
    pub overwrite_deploy: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReconstructionTrainingReport {
    pub backend: String,
    pub sample_count: usize,
    pub train_samples: usize,
    pub val_samples: usize,
    pub train_eval_samples: usize,
    pub val_eval_samples: usize,
    pub image_size: usize,
    pub frames: usize,
    pub feature_source: ReconstructionFeatureSource,
    pub grid: [usize; 2],
    pub input_dim: usize,
    pub reconstruction_architecture: JepaReconstructionArchitecture,
    pub hidden_dim: usize,
    pub min_channels: usize,
    pub residual_blocks_per_scale: usize,
    pub convnext_expansion: usize,
    pub residual_scale: f64,
    pub output_activation: JepaReconstructionOutputActivation,
    pub steps: usize,
    pub batch_size: usize,
    pub data_cache_mode: String,
    pub device_cache_max_mib: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub l1_loss_weight: f64,
    pub gradient_loss_weight: f64,
    pub color_loss_weight: f64,
    pub seed: u64,
    pub train_loss_initial: Option<f64>,
    pub train_loss_final: Option<f64>,
    pub train_psnr_final: Option<f64>,
    pub train_gradient_loss_final: Option<f64>,
    pub train_color_loss_final: Option<f64>,
    pub val_loss_initial: Option<f64>,
    pub val_loss_final: Option<f64>,
    pub val_psnr_final: Option<f64>,
    pub val_gradient_loss_final: Option<f64>,
    pub val_color_loss_final: Option<f64>,
    pub feature_extract_ms: u128,
    pub train_ms: u128,
    pub burnpack: PathBuf,
    pub package_manifest: PathBuf,
    pub parts_manifest: PathBuf,
    pub part_count: usize,
    pub total_bytes: u64,
    pub deploy_manifest: PathBuf,
    pub deploy_dir: PathBuf,
    pub model_base_url: String,
    pub record_dtype: Option<String>,
    pub burnpack_dtype_counts: std::collections::BTreeMap<String, usize>,
}

struct ReconstructionSample {
    feature_values: Vec<f32>,
    target_values: Vec<f32>,
}

struct ReconstructionTrainingTensors<B: Backend> {
    features: Tensor<B, 4>,
    targets: Tensor<B, 4>,
}

#[derive(Clone, Copy, Debug)]
struct ReconstructionLossWeights {
    l1: f64,
    gradient: f64,
    color: f64,
}

enum ReconstructionTrainingData<B: Backend> {
    Device(ReconstructionTrainingTensors<B>),
    Host(Vec<ReconstructionSample>),
}

impl<B: Backend> ReconstructionTrainingData<B> {
    fn cache_mode(&self) -> &'static str {
        match self {
            Self::Device(_) => "device",
            Self::Host(_) => "host_stream",
        }
    }
}

pub fn train_reconstruction_bpk(
    options: ReconstructionTrainingOptions,
) -> Result<ReconstructionTrainingReport> {
    match options.backend {
        crate::JepaTrainBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                crate::runtime::cuda_runtime_preflight(crate::runtime::CUDA_TRAIN_FORCE_ENV)
                    .map_err(|reason| anyhow::anyhow!("cuda backend unavailable: {reason}"))?;
                let device = Default::default();
                train_reconstruction_bpk_backend::<
                    burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>,
                >(options, &device, "cuda")
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = options;
                bail!("cuda backend requested but the cuda feature is not enabled")
            }
        }
        crate::JepaTrainBackend::NdArray => {
            #[cfg(feature = "ndarray")]
            {
                let device = Default::default();
                train_reconstruction_bpk_backend::<
                    burn::backend::Autodiff<burn::backend::NdArray<f32>>,
                >(options, &device, "ndarray")
            }
            #[cfg(not(feature = "ndarray"))]
            {
                let _ = options;
                bail!("ndarray backend requested but the ndarray feature is not enabled")
            }
        }
        crate::JepaTrainBackend::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                let device = Default::default();
                train_reconstruction_bpk_backend::<
                    burn::backend::Autodiff<burn::backend::Wgpu<f32, i32>>,
                >(options, &device, "wgpu")
            }
            #[cfg(not(feature = "wgpu"))]
            {
                let _ = options;
                bail!("wgpu backend requested but the wgpu feature is not enabled")
            }
        }
        crate::JepaTrainBackend::WebGpu => {
            #[cfg(feature = "webgpu")]
            {
                let device = Default::default();
                train_reconstruction_bpk_backend::<
                    burn::backend::Autodiff<burn::backend::WebGpu<f32, i32>>,
                >(options, &device, "webgpu")
            }
            #[cfg(not(feature = "webgpu"))]
            {
                let _ = options;
                bail!("webgpu backend requested but the webgpu feature is not enabled")
            }
        }
        other => bail!(
            "reconstruction training currently supports cuda, wgpu, webgpu, or ndarray backends, got {:?}",
            other
        ),
    }
}

fn train_reconstruction_bpk_backend<B: AutodiffBackend>(
    options: ReconstructionTrainingOptions,
    device: &B::Device,
    backend_name: &'static str,
) -> Result<ReconstructionTrainingReport> {
    validate_options(&options)?;
    let image_paths = collect_image_paths(&options)?;
    ensure!(
        !image_paths.is_empty(),
        "train-reconstruction-bpk found no jpg/jpeg/png images"
    );
    let (jepa, mut jepa_config) = load_jepa_for_training::<B::InnerBackend>(&options, device)?;
    jepa_config.image_size = options.image_size;
    jepa_config.num_frames = options.frames;
    let input_dim = jepa_config.encoder.embed_dim;
    let patch_size = jepa_config.patch_size.max(1);
    ensure!(
        options.image_size.is_multiple_of(patch_size),
        "--image-size {} must be divisible by V-JEPA patch size {}",
        options.image_size,
        patch_size
    );
    let grid = [
        options.image_size / patch_size,
        options.image_size / patch_size,
    ];

    let extract_start = Instant::now();
    let samples = extract_samples::<B::InnerBackend>(
        &jepa,
        &image_paths,
        options.max_samples,
        options.image_size,
        options.frames,
        options.feature_source,
        input_dim,
        grid,
        device,
    )?;
    drop(jepa);
    let feature_extract_ms = extract_start.elapsed().as_millis();
    let split = split_indices(samples.len(), options.val_split, options.seed)?;
    let train_eval = eval_indices(&split.train, TRAIN_EVAL_MAX_SAMPLES);
    let val_eval = eval_indices(&split.val, TRAIN_EVAL_MAX_SAMPLES);
    let sample_count = samples.len();
    let reconstruction_config = JepaReconstructionConfig {
        architecture: options.reconstruction_architecture,
        input_dim,
        hidden_dim: options.hidden_dim,
        min_channels: options.min_channels,
        patch_size,
        residual_blocks_per_scale: options.residual_blocks_per_scale,
        convnext_expansion: options.convnext_expansion,
        residual_scale: options.residual_scale,
        output_activation: options.output_activation,
        ..JepaReconstructionConfig::default()
    };
    let data = build_training_data::<B>(
        samples,
        &reconstruction_config,
        grid,
        options.image_size,
        options.device_cache_max_mib,
        device,
    );
    let loss_weights = ReconstructionLossWeights {
        l1: options.l1_loss_weight,
        gradient: options.gradient_loss_weight,
        color: options.color_loss_weight,
    };
    eprintln!("reconstruction training data cache: {}", data.cache_mode());

    let mut decoder = JepaReconstructionDecoder::<B>::new(reconstruction_config.clone(), device)
        .context("initialize reconstruction decoder")?;
    let mut optim = AdamWConfig::new()
        .with_weight_decay(options.weight_decay as f32)
        .init();

    let val_initial = evaluate_decoder(
        &decoder,
        &data,
        &val_eval,
        &reconstruction_config,
        grid,
        options.image_size,
        options.batch_size,
        device,
    )?;
    let train_initial = evaluate_decoder(
        &decoder,
        &data,
        &train_eval,
        &reconstruction_config,
        grid,
        options.image_size,
        options.batch_size,
        device,
    )?;
    let train_start = Instant::now();
    let mut last_train_loss = train_initial.loss;
    let mut order = split.train.clone();
    shuffle_indices(&mut order, options.seed ^ 0x6d2b_79f5);
    for step in 0..options.steps {
        if step % order.len().max(1) == 0 {
            shuffle_indices(&mut order, options.seed.wrapping_add(step as u64));
        }
        let batch = cyclic_batch(&order, step * options.batch_size, options.batch_size);
        let (features, target) = tensor_batch::<B>(
            &data,
            &batch,
            &reconstruction_config,
            grid,
            options.image_size,
            device,
        );
        let output = decoder.forward_to_size(features, [options.image_size, options.image_size]);
        let loss = reconstruction_training_loss(output, target, loss_weights);
        if options.log_interval > 0
            && ((step + 1) % options.log_interval == 0 || step + 1 == options.steps)
            && let Some(loss_value) = tensor_scalar(loss.clone().detach())
        {
            last_train_loss = Some(loss_value);
            eprintln!(
                "reconstruction step {}/{} loss {:.6}",
                step + 1,
                options.steps,
                loss_value
            );
        }
        let grads = GradientsParams::from_grads(loss.backward(), &decoder);
        decoder = optim.step(options.learning_rate, decoder, grads);
    }
    let train_ms = train_start.elapsed().as_millis();
    let train_final = evaluate_decoder(
        &decoder,
        &data,
        &train_eval,
        &reconstruction_config,
        grid,
        options.image_size,
        options.batch_size,
        device,
    )?;
    let val_final = evaluate_decoder(
        &decoder,
        &data,
        &val_eval,
        &reconstruction_config,
        grid,
        options.image_size,
        options.batch_size,
        device,
    )?;
    let decoder = decoder.valid();

    write_reconstruction_package(
        decoder,
        reconstruction_config,
        ReconstructionPackageWriteInput {
            options,
            backend_name,
            sample_count,
            train_samples: split.train.len(),
            val_samples: split.val.len(),
            train_eval_samples: train_eval.len(),
            val_eval_samples: val_eval.len(),
            grid,
            data_cache_mode: data.cache_mode(),
            train_initial,
            train_final,
            val_initial,
            val_final,
            last_train_loss,
            feature_extract_ms,
            train_ms,
        },
    )
}

struct ReconstructionPackageWriteInput {
    options: ReconstructionTrainingOptions,
    backend_name: &'static str,
    sample_count: usize,
    train_samples: usize,
    val_samples: usize,
    train_eval_samples: usize,
    val_eval_samples: usize,
    grid: [usize; 2],
    data_cache_mode: &'static str,
    train_initial: EvalMetrics,
    train_final: EvalMetrics,
    val_initial: EvalMetrics,
    val_final: EvalMetrics,
    last_train_loss: Option<f64>,
    feature_extract_ms: u128,
    train_ms: u128,
}

fn write_reconstruction_package<B: Backend>(
    decoder: JepaReconstructionDecoder<B>,
    reconstruction_config: JepaReconstructionConfig,
    input: ReconstructionPackageWriteInput,
) -> Result<ReconstructionTrainingReport> {
    let ReconstructionPackageWriteInput {
        options,
        backend_name,
        sample_count,
        train_samples,
        val_samples,
        train_eval_samples,
        val_eval_samples,
        grid,
        data_cache_mode,
        train_initial,
        train_final,
        val_initial,
        val_final,
        last_train_loss,
        feature_extract_ms,
        train_ms,
    } = input;

    std::fs::create_dir_all(options.output.parent().unwrap_or_else(|| Path::new(".")))
        .with_context(|| format!("create output directory for {}", options.output.display()))?;
    let output = options.output.with_extension("bpk");
    crate::save_jepa_reconstruction_burnpack(&decoder.no_grad(), &output)?;
    let burnpack_dtype_counts = crate::burnpack_dtype_counts(&output)?;
    ensure_export_burnpack_is_f16(&burnpack_dtype_counts)?;
    let max_part_bytes = options
        .shard_mib
        .max(1)
        .checked_mul(1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("--shard-mib overflow"))?;
    let parts =
        crate::write_burnpack_parts_for_browser(&output, max_part_bytes, options.overwrite_shards)?;
    let model_base_url = resolve_reconstruction_model_profile_base_url(
        options.reconstruction_model_profile,
        options.reconstruction_model_base_url,
    );
    let manifest = BurnJepaReconstructionPackageManifest {
        record_dtype: Some("f16".to_string()),
        reconstruction_config,
        model_base_url,
        ..BurnJepaReconstructionPackageManifest::default()
    }
    .with_burnpack_paths(&output);
    let manifest_path = output
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("manifest.json");
    crate::write_jepa_reconstruction_package_manifest(&manifest_path, &manifest)?;
    let deploy_bundle = crate::write_burn_jepa_reconstruction_model_deploy_bundle(
        &manifest_path,
        &options.deploy_dir,
        options.overwrite_deploy,
    )?;
    let report = ReconstructionTrainingReport {
        backend: backend_name.to_string(),
        sample_count,
        train_samples,
        val_samples,
        train_eval_samples,
        val_eval_samples,
        image_size: options.image_size,
        frames: options.frames,
        feature_source: options.feature_source,
        grid,
        input_dim: manifest.reconstruction_config.input_dim,
        reconstruction_architecture: manifest.reconstruction_config.architecture,
        hidden_dim: manifest.reconstruction_config.hidden_dim,
        min_channels: manifest.reconstruction_config.min_channels,
        residual_blocks_per_scale: manifest.reconstruction_config.residual_blocks_per_scale,
        convnext_expansion: manifest.reconstruction_config.convnext_expansion,
        residual_scale: manifest.reconstruction_config.residual_scale,
        output_activation: manifest.reconstruction_config.output_activation,
        steps: options.steps,
        batch_size: options.batch_size,
        data_cache_mode: data_cache_mode.to_string(),
        device_cache_max_mib: options.device_cache_max_mib,
        learning_rate: options.learning_rate,
        weight_decay: options.weight_decay,
        l1_loss_weight: options.l1_loss_weight,
        gradient_loss_weight: options.gradient_loss_weight,
        color_loss_weight: options.color_loss_weight,
        seed: options.seed,
        train_loss_initial: train_initial.loss,
        train_loss_final: train_final.loss.or(last_train_loss),
        train_psnr_final: train_final.psnr,
        train_gradient_loss_final: train_final.gradient_loss,
        train_color_loss_final: train_final.color_loss,
        val_loss_initial: val_initial.loss,
        val_loss_final: val_final.loss,
        val_psnr_final: val_final.psnr,
        val_gradient_loss_final: val_final.gradient_loss,
        val_color_loss_final: val_final.color_loss,
        feature_extract_ms,
        train_ms,
        burnpack: output.clone(),
        package_manifest: manifest_path.clone(),
        parts_manifest: parts.manifest_path.clone(),
        part_count: parts.part_paths.len(),
        total_bytes: parts.total_bytes,
        deploy_manifest: deploy_bundle.manifest_path,
        deploy_dir: deploy_bundle.output_dir,
        model_base_url: manifest.model_base_url,
        record_dtype: manifest.record_dtype,
        burnpack_dtype_counts,
    };
    let report_path = manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("training-report.json");
    std::fs::write(&report_path, serde_json::to_string_pretty(&report)?).with_context(|| {
        format!(
            "write reconstruction training report {}",
            report_path.display()
        )
    })?;
    Ok(report)
}

#[derive(Clone)]
struct SplitIndices {
    train: Vec<usize>,
    val: Vec<usize>,
}

#[derive(Clone, Copy)]
struct EvalMetrics {
    loss: Option<f64>,
    psnr: Option<f64>,
    gradient_loss: Option<f64>,
    color_loss: Option<f64>,
}

fn validate_options(options: &ReconstructionTrainingOptions) -> Result<()> {
    ensure!(options.image_size >= 16, "--image-size must be at least 16");
    if options.feature_source == ReconstructionFeatureSource::Video {
        ensure!(
            options.frames >= 2 && options.frames.is_multiple_of(2),
            "--frames must be an even value >= 2 for V-JEPA video feature extraction"
        );
    } else {
        ensure!(options.frames > 0, "--frames must be nonzero");
    }
    ensure!(options.max_samples > 0, "--max-samples must be nonzero");
    ensure!(options.steps > 0, "--steps must be nonzero");
    ensure!(options.batch_size > 0, "--batch-size must be nonzero");
    ensure!(options.hidden_dim > 0, "--hidden-dim must be nonzero");
    ensure!(options.min_channels > 0, "--min-channels must be nonzero");
    ensure!(
        options.convnext_expansion > 0,
        "--convnext-expansion must be nonzero"
    );
    ensure!(
        options.val_split.is_finite() && (0.0..1.0).contains(&options.val_split),
        "--val-split must be in [0, 1)"
    );
    ensure!(
        options.learning_rate.is_finite() && options.learning_rate > 0.0,
        "--lr must be positive and finite"
    );
    ensure!(
        options.weight_decay.is_finite() && options.weight_decay >= 0.0,
        "--weight-decay must be finite and non-negative"
    );
    ensure!(
        options.l1_loss_weight.is_finite() && options.l1_loss_weight >= 0.0,
        "--lambda-l1 must be finite and non-negative"
    );
    ensure!(
        options.gradient_loss_weight.is_finite() && options.gradient_loss_weight >= 0.0,
        "--lambda-gradient must be finite and non-negative"
    );
    ensure!(
        options.color_loss_weight.is_finite() && options.color_loss_weight >= 0.0,
        "--lambda-color must be finite and non-negative"
    );
    ensure!(
        options.residual_scale.is_finite(),
        "--residual-scale must be finite"
    );
    Ok(())
}

fn collect_image_paths(options: &ReconstructionTrainingOptions) -> Result<Vec<PathBuf>> {
    let mut paths = options.image_paths.clone();
    for dir in &options.image_dirs {
        collect_image_dir(dir, &mut paths)?;
    }
    paths.sort();
    paths.dedup();
    if paths.len() > options.max_samples {
        let mut indices = (0..paths.len()).collect::<Vec<_>>();
        shuffle_indices(&mut indices, options.seed);
        indices.truncate(options.max_samples);
        indices.sort_unstable();
        paths = indices
            .into_iter()
            .map(|index| paths[index].clone())
            .collect();
    }
    Ok(paths)
}

fn collect_image_dir(dir: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    ensure!(
        dir.exists(),
        "image directory {} does not exist",
        dir.display()
    );
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("read image dir {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_image_dir(&path, paths)?;
        } else if is_supported_image(&path) {
            paths.push(path);
        }
    }
    Ok(())
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let ext = ext.to_ascii_lowercase();
            IMAGE_EXTENSIONS.contains(&ext.as_str())
        })
        .unwrap_or(false)
}

fn split_indices(len: usize, val_split: f32, seed: u64) -> Result<SplitIndices> {
    ensure!(
        len >= 2,
        "reconstruction training requires at least two samples"
    );
    let mut indices = (0..len).collect::<Vec<_>>();
    shuffle_indices(&mut indices, seed ^ 0xa5a5_5a5a);
    let val_count = ((len as f32) * val_split).round() as usize;
    let val_count = val_count.clamp(1, len - 1);
    let val = indices[..val_count].to_vec();
    let train = indices[val_count..].to_vec();
    Ok(SplitIndices { train, val })
}

fn eval_indices(indices: &[usize], max_samples: usize) -> Vec<usize> {
    indices.iter().copied().take(max_samples.max(1)).collect()
}

fn load_jepa_for_training<B: Backend>(
    options: &ReconstructionTrainingOptions,
    device: &B::Device,
) -> Result<(VJepa2_1Model<B>, crate::VJepaConfig)> {
    let package = if let Some(manifest_path) = &options.jepa_manifest {
        local_jepa_package(manifest_path)?
    } else {
        let model_base_url = resolve_jepa_model_profile_base_url(
            options.jepa_model_profile,
            &options.jepa_model_base_url,
        );
        let bootstrap = BurnJepaModelBootstrapConfig {
            cache_root: options.jepa_cache_dir.clone(),
            model_profile: options.jepa_model_profile,
            model_base_url,
            manifest_url: options.jepa_manifest_url.clone(),
        };
        crate::resolve_or_bootstrap_burn_jepa_model_package_with_config_and_progress(
            &bootstrap,
            |message| eprintln!("{message}"),
        )?
    };
    let manifest_json = std::fs::read_to_string(&package.manifest_path).with_context(|| {
        format!(
            "read JEPA package manifest {}",
            package.manifest_path.display()
        )
    })?;
    let mut manifest = BurnJepaPipelinePackageManifest::from_json_str(&manifest_json)?;
    ensure!(
        manifest.model_kind == BurnJepaPackageModelKind::Base,
        "train-reconstruction-bpk currently expects a base V-JEPA package, got {:?}",
        manifest.model_kind
    );
    manifest.jepa_config.image_size = options.image_size;
    manifest.jepa_config.num_frames = options.frames;
    let parts = package
        .part_paths
        .iter()
        .map(std::fs::read)
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("read JEPA burnpack shards")?;
    let (model, report) =
        crate::load_vjepa_burnpack_parts::<B>(&manifest.jepa_config, &parts, device)?;
    ensure!(
        report.errors.is_empty(),
        "V-JEPA burnpack apply reported errors: {:?}",
        report.errors
    );
    ensure!(
        !report.applied.is_empty(),
        "V-JEPA burnpack package did not apply any tensors"
    );
    Ok((model, manifest.jepa_config))
}

fn local_jepa_package(path: &Path) -> Result<crate::BurnJepaModelPackageFiles> {
    let manifest_json = std::fs::read_to_string(path)
        .with_context(|| format!("read JEPA manifest {}", path.display()))?;
    let manifest = BurnJepaPipelinePackageManifest::from_json_str(&manifest_json)?;
    let parts_manifest_path =
        crate::resolve_package_manifest_entry_path(path, &manifest.parts_manifest)?;
    let parts_manifest = crate::read_parts_manifest(&parts_manifest_path)?;
    let part_paths = parts_manifest
        .parts
        .iter()
        .map(|part| crate::resolve_part_entry_path(&parts_manifest_path, &part.path))
        .collect::<Result<Vec<_>>>()?;
    Ok(crate::BurnJepaModelPackageFiles {
        cache_root: path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
        manifest_path: path.to_path_buf(),
        parts_manifest_path,
        part_paths,
        total_bytes: parts_manifest.total_bytes,
        model_base_url: manifest.model_base_url,
    })
}

fn extract_samples<B: Backend>(
    jepa: &VJepa2_1Model<B>,
    paths: &[PathBuf],
    max_samples: usize,
    image_size: usize,
    frames: usize,
    feature_source: ReconstructionFeatureSource,
    input_dim: usize,
    grid: [usize; 2],
    device: &B::Device,
) -> Result<Vec<ReconstructionSample>> {
    let mut samples = Vec::with_capacity(paths.len().min(max_samples));
    for (sample_index, path) in paths.iter().take(max_samples).enumerate() {
        let target_values = load_target_chw(path, image_size)
            .with_context(|| format!("load reconstruction target image {}", path.display()))?;
        let output = match feature_source {
            ReconstructionFeatureSource::Image => {
                let image = target_chw_to_model_image::<B>(&target_values, image_size, device);
                jepa.encode_image(image, None)
            }
            ReconstructionFeatureSource::Video => {
                let rgba = target_chw_to_repeated_rgba(&target_values, image_size, frames);
                let video = crate::rgba_video_to_tensor::<B>(
                    &rgba,
                    VJepaRgbaVideoShape::new(1, frames, image_size, image_size),
                    device,
                )?;
                jepa.encode_video(video, None)
            }
        };
        let features = jepa_feature_tokens_to_nchw(output.tokens.detach(), output.grid)?;
        let dims = features.shape().dims::<4>();
        ensure!(
            dims == [1, input_dim, grid[0], grid[1]],
            "unexpected V-JEPA feature shape for {}: {:?}, expected {:?}",
            path.display(),
            dims,
            [1, input_dim, grid[0], grid[1]]
        );
        let feature_values = features
            .into_data()
            .to_vec::<f32>()
            .map_err(|err| anyhow::anyhow!("read JEPA feature values: {err:?}"))?;
        samples.push(ReconstructionSample {
            feature_values,
            target_values,
        });
        if (sample_index + 1).is_multiple_of(25) || sample_index + 1 == paths.len().min(max_samples)
        {
            eprintln!(
                "extracted reconstruction sample {}/{}",
                sample_index + 1,
                paths.len().min(max_samples)
            );
        }
    }
    Ok(samples)
}

fn load_target_chw(path: &Path, image_size: usize) -> Result<Vec<f32>> {
    let image = image::open(path)?.to_rgb8();
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
    let mut values = Vec::with_capacity(3 * image_size * image_size);
    for channel in 0..3 {
        for y in 0..image_size {
            for x in 0..image_size {
                values.push(resized.get_pixel(x as u32, y as u32)[channel] as f32 / 255.0);
            }
        }
    }
    Ok(values)
}

fn target_chw_to_repeated_rgba(target: &[f32], image_size: usize, frames: usize) -> Vec<u8> {
    let pixels = image_size * image_size;
    let mut rgba = Vec::with_capacity(frames * pixels * 4);
    for _frame in 0..frames {
        for pixel in 0..pixels {
            let r = target[pixel];
            let g = target[pixels + pixel];
            let b = target[pixels * 2 + pixel];
            rgba.push(float_to_u8(r));
            rgba.push(float_to_u8(g));
            rgba.push(float_to_u8(b));
            rgba.push(255);
        }
    }
    rgba
}

fn target_chw_to_model_image<B: Backend>(
    target: &[f32],
    image_size: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let pixels = image_size * image_size;
    let mut values = Vec::with_capacity(3 * pixels);
    for channel in 0..3 {
        for pixel in 0..pixels {
            let value = target[channel * pixels + pixel];
            values
                .push((value - crate::VJEPA_IMAGE_MEAN[channel]) / crate::VJEPA_IMAGE_STD[channel]);
        }
    }
    Tensor::<B, 4>::from_data(
        TensorData::new(values, [1, 3, image_size, image_size]),
        device,
    )
}

fn float_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn build_training_data<B: AutodiffBackend>(
    samples: Vec<ReconstructionSample>,
    config: &JepaReconstructionConfig,
    grid: [usize; 2],
    image_size: usize,
    device_cache_max_mib: usize,
    device: &B::Device,
) -> ReconstructionTrainingData<B> {
    let feature_bytes = samples.len() * config.input_dim * grid[0] * grid[1] * size_of::<f32>();
    let target_bytes = samples.len() * 3 * image_size * image_size * size_of::<f32>();
    let max_tensor_bytes = device_cache_max_mib.max(1) * 1024 * 1024;
    if feature_bytes > max_tensor_bytes || target_bytes > max_tensor_bytes {
        return ReconstructionTrainingData::Host(samples);
    }
    let per_sample = config.input_dim * grid[0] * grid[1];
    let mut feature_values = Vec::with_capacity(samples.len() * per_sample);
    for sample in &samples {
        feature_values.extend_from_slice(&sample.feature_values);
    }
    let features = Tensor::<B, 4>::from_data(
        TensorData::new(
            feature_values,
            [samples.len(), config.input_dim, grid[0], grid[1]],
        ),
        device,
    );
    let per_sample = 3 * image_size * image_size;
    let mut target_values = Vec::with_capacity(samples.len() * per_sample);
    for sample in &samples {
        target_values.extend_from_slice(&sample.target_values);
    }
    let targets = Tensor::<B, 4>::from_data(
        TensorData::new(target_values, [samples.len(), 3, image_size, image_size]),
        device,
    );
    ReconstructionTrainingData::Device(ReconstructionTrainingTensors { features, targets })
}

fn tensor_batch<B: AutodiffBackend>(
    data: &ReconstructionTrainingData<B>,
    batch: &[usize],
    config: &JepaReconstructionConfig,
    grid: [usize; 2],
    image_size: usize,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    match data {
        ReconstructionTrainingData::Device(dataset) => {
            let indices = batch.iter().map(|&index| index as i64).collect::<Vec<_>>();
            let indices = Tensor::<B, 1, Int>::from_data(indices.as_slice(), device);
            (
                dataset.features.clone().select(0, indices.clone()),
                dataset.targets.clone().select(0, indices),
            )
        }
        ReconstructionTrainingData::Host(samples) => {
            let features = feature_batch::<B>(samples, batch, config, grid, device);
            let target = target_batch::<B>(samples, batch, image_size, device);
            (features, target)
        }
    }
}

fn feature_batch<B: AutodiffBackend>(
    samples: &[ReconstructionSample],
    batch: &[usize],
    config: &JepaReconstructionConfig,
    grid: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 4> {
    let per_sample = config.input_dim * grid[0] * grid[1];
    let mut values = Vec::with_capacity(batch.len() * per_sample);
    for &index in batch {
        values.extend_from_slice(&samples[index].feature_values);
    }
    Tensor::<B, 4>::from_data(
        TensorData::new(values, [batch.len(), config.input_dim, grid[0], grid[1]]),
        device,
    )
}

fn target_batch<B: AutodiffBackend>(
    samples: &[ReconstructionSample],
    batch: &[usize],
    image_size: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let per_sample = 3 * image_size * image_size;
    let mut values = Vec::with_capacity(batch.len() * per_sample);
    for &index in batch {
        values.extend_from_slice(&samples[index].target_values);
    }
    Tensor::<B, 4>::from_data(
        TensorData::new(values, [batch.len(), 3, image_size, image_size]),
        device,
    )
}

fn cyclic_batch(indices: &[usize], offset: usize, batch_size: usize) -> Vec<usize> {
    (0..batch_size)
        .map(|i| indices[(offset + i) % indices.len()])
        .collect()
}

fn evaluate_decoder<B: AutodiffBackend>(
    decoder: &JepaReconstructionDecoder<B>,
    data: &ReconstructionTrainingData<B>,
    indices: &[usize],
    config: &JepaReconstructionConfig,
    grid: [usize; 2],
    image_size: usize,
    batch_size: usize,
    device: &B::Device,
) -> Result<EvalMetrics> {
    if indices.is_empty() {
        return Ok(EvalMetrics {
            loss: None,
            psnr: None,
            gradient_loss: None,
            color_loss: None,
        });
    }
    let mut mse_sum = 0.0;
    let mut psnr_sum = 0.0;
    let mut gradient_sum = 0.0;
    let mut color_sum = 0.0;
    let mut batches = 0usize;
    for batch in indices.chunks(batch_size.max(1)) {
        let (features, target) = tensor_batch::<B>(data, batch, config, grid, image_size, device);
        let output = decoder
            .forward_to_size(features, [image_size, image_size])
            .detach();
        let target = target.detach();
        let mse = tensor_scalar(reconstruction_mse(output.clone(), target.clone()));
        let psnr = reconstruction_psnr_scalar(output.clone(), target.clone(), 1.0);
        let gradient = tensor_scalar(reconstruction_gradient_mse(output.clone(), target.clone()));
        let color = tensor_scalar(reconstruction_color_moment_loss(output, target));
        if let (Some(mse), Some(psnr), Some(gradient), Some(color)) = (mse, psnr, gradient, color) {
            mse_sum += mse;
            psnr_sum += psnr;
            gradient_sum += gradient;
            color_sum += color;
            batches += 1;
        }
    }
    Ok(EvalMetrics {
        loss: (batches > 0).then_some(mse_sum / batches as f64),
        psnr: (batches > 0).then_some(psnr_sum / batches as f64),
        gradient_loss: (batches > 0).then_some(gradient_sum / batches as f64),
        color_loss: (batches > 0).then_some(color_sum / batches as f64),
    })
}

fn reconstruction_training_loss<B: Backend>(
    output: Tensor<B, 4>,
    target: Tensor<B, 4>,
    weights: ReconstructionLossWeights,
) -> Tensor<B, 1> {
    let mut loss = reconstruction_mse(output.clone(), target.clone());
    if weights.l1 > 0.0 {
        loss = loss + reconstruction_l1(output.clone(), target.clone()).mul_scalar(weights.l1);
    }
    if weights.gradient > 0.0 {
        loss = loss
            + reconstruction_gradient_mse(output.clone(), target.clone())
                .mul_scalar(weights.gradient);
    }
    if weights.color > 0.0 {
        loss = loss + reconstruction_color_moment_loss(output, target).mul_scalar(weights.color);
    }
    loss
}

fn tensor_scalar<B: Backend>(tensor: Tensor<B, 1>) -> Option<f64> {
    tensor
        .to_data()
        .to_vec::<f32>()
        .ok()
        .and_then(|values| values.first().copied())
        .map(f64::from)
}

fn shuffle_indices(indices: &mut [usize], seed: u64) {
    let mut state = seed.max(1);
    for i in (1..indices.len()).rev() {
        state = splitmix64(state);
        indices.swap(i, (state as usize) % (i + 1));
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn resolve_jepa_model_profile_base_url(
    profile: BurnJepaModelProfile,
    model_base_url: &str,
) -> String {
    if model_base_url == DEFAULT_BURN_JEPA_MODEL_BASE_URL {
        crate::burn_jepa_model_profile_base_url(profile)
    } else {
        model_base_url.to_string()
    }
}

fn resolve_reconstruction_model_profile_base_url(
    profile: BurnJepaReconstructionModelProfile,
    model_base_url: String,
) -> String {
    if model_base_url == DEFAULT_BURN_JEPA_RECONSTRUCTION_MODEL_BASE_URL {
        crate::burn_jepa_reconstruction_model_profile_base_url(profile)
    } else {
        model_base_url
    }
}

fn ensure_export_burnpack_is_f16(counts: &std::collections::BTreeMap<String, usize>) -> Result<()> {
    ensure!(
        counts.get("F16").copied().unwrap_or(0) > 0 && counts.get("F32").copied().unwrap_or(0) == 0,
        "exported reconstruction burnpack must be f16-only, got {:?}",
        counts
    );
    Ok(())
}
