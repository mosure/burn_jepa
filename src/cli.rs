#![cfg_attr(
    not(any(
        feature = "ndarray",
        feature = "flex",
        feature = "dispatch",
        feature = "wgpu",
        feature = "webgpu",
        feature = "cuda"
    )),
    allow(dead_code, unused_imports, unused_variables)
)]

#[cfg(feature = "cuda")]
use crate::runtime::{CUDA_TRAIN_FORCE_ENV, cuda_runtime_preflight};
use crate::{
    BurnJepaTrainConfig, DenseJepaTrainingReport, ExperimentConfig, ExperimentRunReport,
    JepaTrainBackend, TttEvalReport, TttTrainingReport, analyze_experiment,
    evaluate_ttt_base_sparse, evaluate_ttt_model_file, prepare_experiment_data, run_experiment,
    train_dense_jepa, train_ttt_distillation, write_experiment_plan,
};
#[cfg(feature = "dispatch")]
use crate::{JepaDispatchBackend, TrainingLoopConfig};
#[cfg(feature = "ndarray")]
use anyhow::ensure;
use anyhow::{Result, bail};
use burn::tensor::backend::AutodiffBackend;
use clap::{Parser, Subcommand};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "burn-jepa")]
#[command(about = "Burn-native V-JEPA 2.1 training and evaluation utilities")]
pub struct BurnJepaCli {
    #[command(subcommand)]
    pub command: BurnJepaCommand,
}

#[derive(Debug, Subcommand)]
pub enum BurnJepaCommand {
    Experiment {
        #[command(subcommand)]
        command: ExperimentCommand,
    },
    TrainTtt {
        #[arg(short, long)]
        config: PathBuf,
    },
    EvalTtt {
        #[arg(short, long)]
        config: PathBuf,
        #[arg(short, long, required_unless_present = "base_sparse")]
        model: Option<PathBuf>,
        #[arg(long)]
        base_sparse: bool,
        #[arg(long)]
        steps: Option<usize>,
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long)]
        full_grid: bool,
        #[arg(long)]
        no_full_grid: bool,
    },
    TrainJepa {
        #[arg(short, long)]
        config: PathBuf,
    },
    BenchTtt {
        #[arg(short, long)]
        config: PathBuf,
        #[arg(long)]
        steps: Option<usize>,
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long)]
        eval_steps: Option<usize>,
        #[arg(long)]
        eval_batch_size: Option<usize>,
    },
    ExportBpk {
        #[arg(short, long)]
        config: PathBuf,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 20)]
        shard_mib: u64,
        #[arg(long, default_value_t = false)]
        overwrite_shards: bool,
        #[arg(
            long,
            visible_alias = "model-name",
            help = "CDN/cache profile route. Defaults to base for base exports and TTT for TTT exports."
        )]
        model_profile: Option<crate::BurnJepaModelProfile>,
        #[arg(long, default_value = crate::DEFAULT_BURN_JEPA_MODEL_BASE_URL)]
        model_base_url: String,
        #[arg(
            long,
            help = "Optional clean CDN/upload directory containing manifest.json, parts manifest, and shard files only."
        )]
        deploy_dir: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        overwrite_deploy: bool,
        #[arg(
            long,
            default_value_t = false,
            help = "Permit exporting the tiny test model when no checkpoint/config is configured."
        )]
        allow_tiny_model: bool,
    },
    CacheModel {
        #[arg(long, visible_alias = "model-name", default_value_t = crate::BurnJepaModelProfile::default())]
        model_profile: crate::BurnJepaModelProfile,
        #[arg(long, default_value = crate::DEFAULT_BURN_JEPA_MODEL_BASE_URL)]
        model_base_url: String,
        #[arg(long)]
        manifest_url: Option<String>,
        #[arg(
            long,
            help = "Exact local cache directory. Defaults to ~/.burn_jepa/models/burn_jepa/{model_profile}."
        )]
        cache_dir: Option<PathBuf>,
    },
    VerifyBpk {
        #[arg(
            long,
            help = "Local burn_jepa package manifest. If omitted, cache/download from --model-base-url."
        )]
        manifest: Option<PathBuf>,
        #[arg(long, visible_alias = "model-name", default_value_t = crate::BurnJepaModelProfile::default())]
        model_profile: crate::BurnJepaModelProfile,
        #[arg(long, default_value = crate::DEFAULT_BURN_JEPA_MODEL_BASE_URL)]
        model_base_url: String,
        #[arg(long)]
        manifest_url: Option<String>,
        #[arg(
            long,
            help = "Exact local cache directory. Defaults to ~/.burn_jepa/models/burn_jepa/{model_profile}."
        )]
        cache_dir: Option<PathBuf>,
        #[arg(
            long,
            help = "Optional original V-JEPA checkpoint directory to compare against."
        )]
        checkpoint_dir: Option<PathBuf>,
        #[arg(long)]
        weights_name: Option<String>,
        #[arg(long, default_value_t = 32)]
        image_size: usize,
        #[arg(long, default_value_t = 4)]
        frames: usize,
        #[arg(long, default_value_t = 0.25)]
        max_abs_tol: f32,
        #[arg(long, default_value_t = 0.05)]
        mean_abs_tol: f32,
    },
    BundleBpkDeploy {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = false)]
        overwrite: bool,
    },
    ExportAnyupBpk {
        #[arg(long, default_value = crate::DEFAULT_BURN_ANYUP_CHECKPOINT_PATH)]
        weights: PathBuf,
        #[arg(
            short,
            long,
            default_value = "target/burn_anyup-build/anyup_multi_backbone/anyup.bpk"
        )]
        output: PathBuf,
        #[arg(long, default_value_t = 20)]
        shard_mib: u64,
        #[arg(long, default_value_t = false)]
        overwrite_shards: bool,
        #[arg(long, visible_alias = "anyup-model-name", default_value_t = crate::BurnAnyUpModelProfile::default())]
        model_profile: crate::BurnAnyUpModelProfile,
        #[arg(long, default_value = crate::DEFAULT_BURN_ANYUP_MODEL_BASE_URL)]
        model_base_url: String,
        #[arg(
            long,
            default_value = "target/burn_anyup/anyup_multi_backbone",
            help = "Clean CDN/upload directory containing manifest.json, parts manifest, and shard files only."
        )]
        deploy_dir: PathBuf,
        #[arg(long, default_value_t = false)]
        overwrite_deploy: bool,
        #[arg(
            long,
            default_value_t = false,
            help = "Permit exporting a randomly initialized AnyUp if --weights does not exist."
        )]
        allow_random_model: bool,
    },
    CacheAnyup {
        #[arg(long, visible_alias = "anyup-model-name", default_value_t = crate::BurnAnyUpModelProfile::default())]
        model_profile: crate::BurnAnyUpModelProfile,
        #[arg(long, default_value = crate::DEFAULT_BURN_ANYUP_MODEL_BASE_URL)]
        model_base_url: String,
        #[arg(long)]
        manifest_url: Option<String>,
        #[arg(
            long,
            help = "Exact local cache directory. Defaults to ~/.burn_jepa/models/burn_anyup/{model_profile}."
        )]
        cache_dir: Option<PathBuf>,
    },
    VerifyAnyupBpk {
        #[arg(
            long,
            help = "Local burn_anyup package manifest. If omitted, cache/download from --model-base-url."
        )]
        manifest: Option<PathBuf>,
        #[arg(long, visible_alias = "anyup-model-name", default_value_t = crate::BurnAnyUpModelProfile::default())]
        model_profile: crate::BurnAnyUpModelProfile,
        #[arg(long, default_value = crate::DEFAULT_BURN_ANYUP_MODEL_BASE_URL)]
        model_base_url: String,
        #[arg(long)]
        manifest_url: Option<String>,
        #[arg(
            long,
            help = "Exact local cache directory. Defaults to ~/.burn_jepa/models/burn_anyup/{model_profile}."
        )]
        cache_dir: Option<PathBuf>,
        #[arg(long, default_value_t = 32)]
        image_size: usize,
    },
    BundleAnyupBpkDeploy {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = false)]
        overwrite: bool,
    },
    ExportReconstructionBpk {
        #[arg(
            short,
            long,
            default_value = "target/burn_jepa_reconstruction-build/low_res_v1/jepa_reconstruction.bpk"
        )]
        output: PathBuf,
        #[arg(long, default_value_t = 20)]
        shard_mib: u64,
        #[arg(long, default_value_t = false)]
        overwrite_shards: bool,
        #[arg(long, visible_alias = "reconstruction-model-name", default_value_t = crate::BurnJepaReconstructionModelProfile::default())]
        model_profile: crate::BurnJepaReconstructionModelProfile,
        #[arg(long, default_value = crate::DEFAULT_BURN_JEPA_RECONSTRUCTION_MODEL_BASE_URL)]
        model_base_url: String,
        #[arg(
            long,
            default_value = "target/burn_jepa_reconstruction/low_res_v1",
            help = "Clean CDN/upload directory containing manifest.json, parts manifest, and shard files only."
        )]
        deploy_dir: PathBuf,
        #[arg(long, default_value_t = false)]
        overwrite_deploy: bool,
        #[arg(long, default_value_t = crate::JepaReconstructionConfig::default().input_dim)]
        input_dim: usize,
        #[arg(long, default_value_t = crate::JepaReconstructionConfig::default().hidden_dim)]
        hidden_dim: usize,
        #[arg(long, default_value_t = crate::JepaReconstructionConfig::default().patch_size)]
        patch_size: usize,
    },
    TrainReconstructionBpk {
        #[arg(long, default_value = "cuda")]
        backend: String,
        #[arg(long, help = "Local burn_jepa V-JEPA base package manifest.json.")]
        jepa_manifest: Option<PathBuf>,
        #[arg(long, visible_alias = "model-name", default_value_t = crate::BurnJepaModelProfile::Vjepa21Base)]
        jepa_model_profile: crate::BurnJepaModelProfile,
        #[arg(long, default_value = crate::DEFAULT_BURN_JEPA_MODEL_BASE_URL)]
        jepa_model_base_url: String,
        #[arg(long)]
        jepa_manifest_url: Option<String>,
        #[arg(long)]
        jepa_cache_dir: Option<PathBuf>,
        #[arg(long = "image")]
        images: Vec<PathBuf>,
        #[arg(long = "image-dir")]
        image_dirs: Vec<PathBuf>,
        #[arg(long, default_value_t = 512)]
        image_size: usize,
        #[arg(long, default_value_t = 2)]
        frames: usize,
        #[arg(long, default_value = "image")]
        feature_source: String,
        #[arg(long, default_value_t = 512)]
        max_samples: usize,
        #[arg(long, default_value_t = 0.10)]
        val_split: f32,
        #[arg(long, default_value_t = 12000)]
        steps: usize,
        #[arg(long, default_value_t = 4)]
        batch_size: usize,
        #[arg(long, default_value_t = 4.0e-4)]
        lr: f64,
        #[arg(long, default_value_t = 1.0e-4)]
        weight_decay: f64,
        #[arg(long, default_value_t = 0.0)]
        lambda_l1: f64,
        #[arg(long, default_value_t = 0.0)]
        lambda_gradient: f64,
        #[arg(long, default_value_t = 0.0)]
        lambda_color: f64,
        #[arg(long, default_value_t = 512)]
        hidden_dim: usize,
        #[arg(long, default_value = "patch-conv")]
        reconstruction_architecture: String,
        #[arg(long, default_value_t = crate::JepaReconstructionConfig::default().min_channels)]
        min_channels: usize,
        #[arg(long, default_value_t = 2)]
        residual_blocks_per_scale: usize,
        #[arg(long, default_value_t = crate::JepaReconstructionConfig::default().convnext_expansion)]
        convnext_expansion: usize,
        #[arg(long, default_value_t = crate::JepaReconstructionConfig::default().residual_scale)]
        residual_scale: f64,
        #[arg(long, default_value = "sigmoid")]
        output_activation: String,
        #[arg(long, default_value_t = crate::reconstruction_training::DEFAULT_RECONSTRUCTION_DEVICE_CACHE_MAX_MIB)]
        device_cache_max_mib: usize,
        #[arg(long, default_value_t = 50)]
        log_interval: usize,
        #[arg(long, default_value_t = 0x5EED)]
        seed: u64,
        #[arg(
            short,
            long,
            default_value = "target/burn_jepa_reconstruction-build/low_res_v1/jepa_reconstruction.bpk"
        )]
        output: PathBuf,
        #[arg(long, default_value_t = 20)]
        shard_mib: u64,
        #[arg(long, default_value_t = false)]
        overwrite_shards: bool,
        #[arg(long, visible_alias = "reconstruction-model-name", default_value_t = crate::BurnJepaReconstructionModelProfile::default())]
        reconstruction_model_profile: crate::BurnJepaReconstructionModelProfile,
        #[arg(long, default_value = crate::DEFAULT_BURN_JEPA_RECONSTRUCTION_MODEL_BASE_URL)]
        reconstruction_model_base_url: String,
        #[arg(
            long,
            default_value = "target/burn_jepa_reconstruction/low_res_v1",
            help = "Clean CDN/upload directory containing manifest.json, parts manifest, and shard files only."
        )]
        deploy_dir: PathBuf,
        #[arg(long, default_value_t = false)]
        overwrite_deploy: bool,
    },
    CacheReconstruction {
        #[arg(long, visible_alias = "reconstruction-model-name", default_value_t = crate::BurnJepaReconstructionModelProfile::default())]
        model_profile: crate::BurnJepaReconstructionModelProfile,
        #[arg(long, default_value = crate::DEFAULT_BURN_JEPA_RECONSTRUCTION_MODEL_BASE_URL)]
        model_base_url: String,
        #[arg(long)]
        manifest_url: Option<String>,
        #[arg(
            long,
            help = "Exact local cache directory. Defaults to ~/.burn_jepa/models/burn_jepa_reconstruction/{model_profile}."
        )]
        cache_dir: Option<PathBuf>,
    },
    VerifyReconstructionBpk {
        #[arg(
            long,
            help = "Local burn_jepa_reconstruction package manifest. If omitted, cache/download from --model-base-url."
        )]
        manifest: Option<PathBuf>,
        #[arg(long, visible_alias = "reconstruction-model-name", default_value_t = crate::BurnJepaReconstructionModelProfile::default())]
        model_profile: crate::BurnJepaReconstructionModelProfile,
        #[arg(long, default_value = crate::DEFAULT_BURN_JEPA_RECONSTRUCTION_MODEL_BASE_URL)]
        model_base_url: String,
        #[arg(long)]
        manifest_url: Option<String>,
        #[arg(
            long,
            help = "Exact local cache directory. Defaults to ~/.burn_jepa/models/burn_jepa_reconstruction/{model_profile}."
        )]
        cache_dir: Option<PathBuf>,
        #[arg(long, default_value_t = 64)]
        image_size: usize,
    },
    BundleReconstructionBpkDeploy {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = false)]
        overwrite: bool,
    },
    PrintConfig,
    PrintExperimentConfig,
}

#[derive(Debug, Subcommand)]
pub enum ExperimentCommand {
    Plan {
        #[arg(short, long)]
        config: PathBuf,
    },
    PrepareData {
        #[arg(short, long)]
        config: PathBuf,
    },
    Run {
        #[arg(short, long)]
        config: PathBuf,
    },
    Analyze {
        #[arg(long)]
        run_dir: PathBuf,
    },
}

pub fn main() -> Result<()> {
    let cli = BurnJepaCli::parse();
    run(cli)
}

pub fn run(cli: BurnJepaCli) -> Result<()> {
    match cli.command {
        BurnJepaCommand::Experiment { command } => run_experiment_command(command),
        BurnJepaCommand::TrainTtt { config } => {
            let config = BurnJepaTrainConfig::from_toml_file(config)?;
            let report = dispatch_ttt(&config)?;
            print_json(&report)
        }
        BurnJepaCommand::EvalTtt {
            config,
            model,
            base_sparse,
            steps,
            batch_size,
            full_grid,
            no_full_grid,
        } => {
            let mut config = BurnJepaTrainConfig::from_toml_file(config)?;
            if let Some(batch_size) = batch_size {
                config.training.eval_batch_size = Some(batch_size);
            }
            if full_grid && no_full_grid {
                bail!("--full-grid and --no-full-grid cannot be used together");
            }
            if full_grid {
                config.training.eval_full_grid = true;
            }
            if no_full_grid {
                config.training.eval_full_grid = false;
            }
            let steps = steps.unwrap_or(config.training.eval_steps.max(1));
            let report = dispatch_ttt_eval(&config, model, base_sparse, steps)?;
            print_json(&report)
        }
        BurnJepaCommand::TrainJepa { config } => {
            let config = BurnJepaTrainConfig::from_toml_file(config)?;
            let report = dispatch_dense(&config)?;
            print_json(&report)
        }
        BurnJepaCommand::BenchTtt {
            config,
            steps,
            batch_size,
            eval_steps,
            eval_batch_size,
        } => {
            let mut config = BurnJepaTrainConfig::from_toml_file(config)?;
            if let Some(steps) = steps {
                config.training.max_steps = steps;
                config.training.lr_schedule =
                    config.training.lr_schedule.clamped_to_max_steps(steps);
            }
            if let Some(batch_size) = batch_size {
                config.training.batch_size = batch_size.max(1);
            }
            if let Some(eval_steps) = eval_steps {
                config.training.eval_steps = eval_steps;
            }
            if let Some(eval_batch_size) = eval_batch_size {
                config.training.eval_batch_size = Some(eval_batch_size);
            }
            config.model.save_model = false;
            let report = dispatch_ttt(&config)?;
            print_json(&report)
        }
        BurnJepaCommand::ExportBpk {
            config,
            model,
            output,
            shard_mib,
            overwrite_shards,
            model_profile,
            model_base_url,
            deploy_dir,
            overwrite_deploy,
            allow_tiny_model,
        } => dispatch_export_bpk(
            config,
            model,
            output,
            shard_mib,
            overwrite_shards,
            model_profile,
            model_base_url,
            deploy_dir,
            overwrite_deploy,
            allow_tiny_model,
        ),
        BurnJepaCommand::CacheModel {
            model_profile,
            model_base_url,
            manifest_url,
            cache_dir,
        } => {
            let model_base_url = resolve_model_profile_base_url(model_profile, model_base_url);
            let config = crate::BurnJepaModelBootstrapConfig {
                cache_root: cache_dir,
                model_profile,
                model_base_url,
                manifest_url,
            };
            let report =
                crate::resolve_or_bootstrap_burn_jepa_model_package_with_config_and_progress(
                    &config,
                    |message| eprintln!("{message}"),
                )?;
            print_json(&report)
        }
        BurnJepaCommand::VerifyBpk {
            manifest,
            model_profile,
            model_base_url,
            manifest_url,
            cache_dir,
            checkpoint_dir,
            weights_name,
            image_size,
            frames,
            max_abs_tol,
            mean_abs_tol,
        } => dispatch_verify_bpk(
            manifest,
            model_profile,
            model_base_url,
            manifest_url,
            cache_dir,
            checkpoint_dir,
            weights_name,
            image_size,
            frames,
            max_abs_tol,
            mean_abs_tol,
        ),
        BurnJepaCommand::BundleBpkDeploy {
            manifest,
            output,
            overwrite,
        } => {
            let report = crate::write_burn_jepa_model_deploy_bundle(manifest, output, overwrite)?;
            print_json(&report)
        }
        BurnJepaCommand::ExportAnyupBpk {
            weights,
            output,
            shard_mib,
            overwrite_shards,
            model_profile,
            model_base_url,
            deploy_dir,
            overwrite_deploy,
            allow_random_model,
        } => dispatch_export_anyup_bpk(
            weights,
            output,
            shard_mib,
            overwrite_shards,
            model_profile,
            model_base_url,
            deploy_dir,
            overwrite_deploy,
            allow_random_model,
        ),
        BurnJepaCommand::CacheAnyup {
            model_profile,
            model_base_url,
            manifest_url,
            cache_dir,
        } => {
            let model_base_url =
                resolve_anyup_model_profile_base_url(model_profile, model_base_url);
            let config = crate::BurnAnyUpModelBootstrapConfig {
                cache_root: cache_dir,
                model_profile,
                model_base_url,
                manifest_url,
            };
            let report =
                crate::resolve_or_bootstrap_burn_anyup_model_package_with_config_and_progress(
                    &config,
                    |message| eprintln!("{message}"),
                )?;
            print_json(&report)
        }
        BurnJepaCommand::VerifyAnyupBpk {
            manifest,
            model_profile,
            model_base_url,
            manifest_url,
            cache_dir,
            image_size,
        } => dispatch_verify_anyup_bpk(
            manifest,
            model_profile,
            model_base_url,
            manifest_url,
            cache_dir,
            image_size,
        ),
        BurnJepaCommand::BundleAnyupBpkDeploy {
            manifest,
            output,
            overwrite,
        } => {
            let report = crate::write_burn_anyup_model_deploy_bundle(manifest, output, overwrite)?;
            print_json(&report)
        }
        BurnJepaCommand::ExportReconstructionBpk {
            output,
            shard_mib,
            overwrite_shards,
            model_profile,
            model_base_url,
            deploy_dir,
            overwrite_deploy,
            input_dim,
            hidden_dim,
            patch_size,
        } => dispatch_export_reconstruction_bpk(
            output,
            shard_mib,
            overwrite_shards,
            model_profile,
            model_base_url,
            deploy_dir,
            overwrite_deploy,
            input_dim,
            hidden_dim,
            patch_size,
        ),
        BurnJepaCommand::TrainReconstructionBpk {
            backend,
            jepa_manifest,
            jepa_model_profile,
            jepa_model_base_url,
            jepa_manifest_url,
            jepa_cache_dir,
            images,
            image_dirs,
            image_size,
            frames,
            feature_source,
            max_samples,
            val_split,
            steps,
            batch_size,
            lr,
            weight_decay,
            lambda_l1,
            lambda_gradient,
            lambda_color,
            hidden_dim,
            reconstruction_architecture,
            min_channels,
            residual_blocks_per_scale,
            convnext_expansion,
            residual_scale,
            output_activation,
            device_cache_max_mib,
            log_interval,
            seed,
            output,
            shard_mib,
            overwrite_shards,
            reconstruction_model_profile,
            reconstruction_model_base_url,
            deploy_dir,
            overwrite_deploy,
        } => {
            let backend = parse_reconstruction_backend(&backend)?;
            let reconstruction_architecture =
                parse_reconstruction_architecture(&reconstruction_architecture)?;
            let output_activation = parse_reconstruction_output_activation(&output_activation)?;
            let feature_source = parse_reconstruction_feature_source(&feature_source)?;
            let report = crate::reconstruction_training::train_reconstruction_bpk(
                crate::reconstruction_training::ReconstructionTrainingOptions {
                    backend,
                    jepa_manifest,
                    jepa_model_profile,
                    jepa_model_base_url,
                    jepa_manifest_url,
                    jepa_cache_dir,
                    image_paths: images,
                    image_dirs,
                    image_size,
                    frames,
                    feature_source,
                    max_samples,
                    val_split,
                    steps,
                    batch_size,
                    learning_rate: lr,
                    weight_decay,
                    l1_loss_weight: lambda_l1,
                    gradient_loss_weight: lambda_gradient,
                    color_loss_weight: lambda_color,
                    hidden_dim,
                    reconstruction_architecture,
                    min_channels,
                    residual_blocks_per_scale,
                    convnext_expansion,
                    residual_scale,
                    output_activation,
                    device_cache_max_mib,
                    log_interval,
                    seed,
                    output,
                    shard_mib,
                    overwrite_shards,
                    reconstruction_model_profile,
                    reconstruction_model_base_url,
                    deploy_dir,
                    overwrite_deploy,
                },
            )?;
            print_json(&report)
        }
        BurnJepaCommand::CacheReconstruction {
            model_profile,
            model_base_url,
            manifest_url,
            cache_dir,
        } => {
            let model_base_url =
                resolve_reconstruction_model_profile_base_url(model_profile, model_base_url);
            let config = crate::BurnJepaReconstructionModelBootstrapConfig {
                cache_root: cache_dir,
                model_profile,
                model_base_url,
                manifest_url,
            };
            let report =
                crate::resolve_or_bootstrap_burn_jepa_reconstruction_model_package_with_config_and_progress(
                    &config,
                    |message| eprintln!("{message}"),
                )?;
            print_json(&report)
        }
        BurnJepaCommand::VerifyReconstructionBpk {
            manifest,
            model_profile,
            model_base_url,
            manifest_url,
            cache_dir,
            image_size,
        } => dispatch_verify_reconstruction_bpk(
            manifest,
            model_profile,
            model_base_url,
            manifest_url,
            cache_dir,
            image_size,
        ),
        BurnJepaCommand::BundleReconstructionBpkDeploy {
            manifest,
            output,
            overwrite,
        } => {
            let report = crate::write_burn_jepa_reconstruction_model_deploy_bundle(
                manifest, output, overwrite,
            )?;
            print_json(&report)
        }
        BurnJepaCommand::PrintConfig => {
            let config = BurnJepaTrainConfig::default();
            println!("{}", config.to_toml_string()?);
            Ok(())
        }
        BurnJepaCommand::PrintExperimentConfig => {
            let config = ExperimentConfig::default();
            println!("{}", config.to_toml_string()?);
            Ok(())
        }
    }
}

fn resolve_model_profile_base_url(
    model_profile: crate::BurnJepaModelProfile,
    model_base_url: String,
) -> String {
    if model_base_url == crate::DEFAULT_BURN_JEPA_MODEL_BASE_URL {
        crate::burn_jepa_model_profile_base_url(model_profile)
    } else {
        model_base_url
    }
}

fn resolve_anyup_model_profile_base_url(
    model_profile: crate::BurnAnyUpModelProfile,
    model_base_url: String,
) -> String {
    if model_base_url == crate::DEFAULT_BURN_ANYUP_MODEL_BASE_URL {
        crate::burn_anyup_model_profile_base_url(model_profile)
    } else {
        model_base_url
    }
}

fn resolve_reconstruction_model_profile_base_url(
    model_profile: crate::BurnJepaReconstructionModelProfile,
    model_base_url: String,
) -> String {
    if model_base_url == crate::DEFAULT_BURN_JEPA_RECONSTRUCTION_MODEL_BASE_URL {
        crate::burn_jepa_reconstruction_model_profile_base_url(model_profile)
    } else {
        model_base_url
    }
}

fn parse_reconstruction_backend(value: &str) -> Result<JepaTrainBackend> {
    match value.trim().to_ascii_lowercase().as_str() {
        "cuda" => Ok(JepaTrainBackend::Cuda),
        "wgpu" => Ok(JepaTrainBackend::Wgpu),
        "webgpu" => Ok(JepaTrainBackend::WebGpu),
        "ndarray" | "cpu" => Ok(JepaTrainBackend::NdArray),
        other => bail!(
            "unsupported reconstruction backend `{other}`; expected cuda, wgpu, webgpu, or ndarray"
        ),
    }
}

fn parse_reconstruction_architecture(value: &str) -> Result<crate::JepaReconstructionArchitecture> {
    match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "residual-uniform" | "uniform" | "legacy" => {
            Ok(crate::JepaReconstructionArchitecture::ResidualUniform)
        }
        "pyramid-convnext" | "convnext-pyramid" | "pyramid" | "convnext" => {
            Ok(crate::JepaReconstructionArchitecture::PyramidConvnext)
        }
        "patch-linear" | "patchlinear" | "patch" => {
            Ok(crate::JepaReconstructionArchitecture::PatchLinear)
        }
        "patch-conv" | "patchconv" | "token-conv" => {
            Ok(crate::JepaReconstructionArchitecture::PatchConv)
        }
        other => bail!(
            "unsupported reconstruction architecture `{other}`; expected residual-uniform, pyramid-convnext, patch-linear, or patch-conv"
        ),
    }
}

fn parse_reconstruction_feature_source(
    value: &str,
) -> Result<crate::reconstruction_training::ReconstructionFeatureSource> {
    match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "image" | "single-frame" | "single" | "runtime" | "feature-frame" => {
            Ok(crate::reconstruction_training::ReconstructionFeatureSource::Image)
        }
        "video" | "temporal" | "tubelet" | "legacy" => {
            Ok(crate::reconstruction_training::ReconstructionFeatureSource::Video)
        }
        other => {
            bail!("unsupported reconstruction feature source `{other}`; expected image or video")
        }
    }
}

fn parse_reconstruction_output_activation(
    value: &str,
) -> Result<crate::JepaReconstructionOutputActivation> {
    match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "sigmoid" => Ok(crate::JepaReconstructionOutputActivation::Sigmoid),
        "tanh01" | "tanh-01" | "tanh" => Ok(crate::JepaReconstructionOutputActivation::Tanh01),
        "none" | "linear" | "raw" => Ok(crate::JepaReconstructionOutputActivation::None),
        other => bail!(
            "unsupported reconstruction output activation `{other}`; expected sigmoid, tanh01, or none"
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_verify_bpk(
    manifest: Option<PathBuf>,
    model_profile: crate::BurnJepaModelProfile,
    model_base_url: String,
    manifest_url: Option<String>,
    cache_dir: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
    weights_name: Option<String>,
    image_size: usize,
    frames: usize,
    max_abs_tol: f32,
    mean_abs_tol: f32,
) -> Result<()> {
    #[cfg(feature = "ndarray")]
    {
        verify_bpk_ndarray(
            manifest,
            model_profile,
            model_base_url,
            manifest_url,
            cache_dir,
            checkpoint_dir,
            weights_name,
            image_size,
            frames,
            max_abs_tol,
            mean_abs_tol,
        )
    }
    #[cfg(not(feature = "ndarray"))]
    {
        let _ = (
            manifest,
            model_profile,
            model_base_url,
            manifest_url,
            cache_dir,
            checkpoint_dir,
            weights_name,
            image_size,
            frames,
            max_abs_tol,
            mean_abs_tol,
        );
        bail!("verify-bpk requires the ndarray feature so native numerical checks can run on CPU")
    }
}

#[cfg(feature = "ndarray")]
#[derive(Debug, Serialize)]
struct BpkVerifyReport {
    manifest_path: PathBuf,
    parts_manifest_path: PathBuf,
    part_count: usize,
    total_bytes: u64,
    model_kind: crate::BurnJepaPackageModelKind,
    record_dtype: Option<String>,
    burnpack_dtype_counts: std::collections::BTreeMap<String, usize>,
    runtime_dtype_counts: std::collections::BTreeMap<String, usize>,
    apply_applied: usize,
    apply_missing: usize,
    apply_skipped: usize,
    apply_unused: usize,
    apply_errors: usize,
    output_shape: Vec<usize>,
    grid: [usize; 3],
    sample_count: usize,
    sample_mean: f32,
    sample_min: f32,
    sample_max: f32,
    checkpoint_parity: Option<BpkCheckpointParityReport>,
    load_path: &'static str,
}

#[cfg(feature = "ndarray")]
#[derive(Debug, Serialize)]
struct BpkCheckpointParityReport {
    checkpoint_dir: PathBuf,
    checkpoint_applied: usize,
    checkpoint_missing: usize,
    checkpoint_skipped: usize,
    checkpoint_errors: usize,
    max_abs: f32,
    mean_abs: f32,
    rmse: f32,
    cosine: f32,
    within_tolerance: bool,
}

#[cfg(feature = "ndarray")]
enum NativeBpkModel {
    Base(Box<crate::VJepa2_1Model<burn::backend::NdArray<f32>>>),
    Ttt(Box<crate::VJepaTttModel<burn::backend::NdArray<f32>>>),
}

#[cfg(feature = "ndarray")]
#[allow(clippy::too_many_arguments)]
fn verify_bpk_ndarray(
    manifest_path: Option<PathBuf>,
    model_profile: crate::BurnJepaModelProfile,
    model_base_url: String,
    manifest_url: Option<String>,
    cache_dir: Option<PathBuf>,
    checkpoint_dir: Option<PathBuf>,
    weights_name: Option<String>,
    image_size: usize,
    frames: usize,
    max_abs_tol: f32,
    mean_abs_tol: f32,
) -> Result<()> {
    ensure!(image_size > 0, "--image-size must be nonzero");
    ensure!(frames > 0, "--frames must be nonzero");

    type B = burn::backend::NdArray<f32>;

    let package = if let Some(manifest_path) = manifest_path {
        let manifest_json = std::fs::read_to_string(&manifest_path)?;
        let manifest = crate::BurnJepaPipelinePackageManifest::from_json_str(&manifest_json)?;
        let parts_manifest_path =
            crate::resolve_package_manifest_entry_path(&manifest_path, &manifest.parts_manifest)?;
        let parts_manifest = crate::read_parts_manifest(&parts_manifest_path)?;
        let part_paths = parts_manifest
            .parts
            .iter()
            .map(|part| crate::resolve_part_entry_path(&parts_manifest_path, &part.path))
            .collect::<Result<Vec<_>>>()?;
        Some(crate::BurnJepaModelPackageFiles {
            cache_root: manifest_path
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .to_path_buf(),
            manifest_path,
            parts_manifest_path,
            part_paths,
            total_bytes: parts_manifest.total_bytes,
            model_base_url: manifest.model_base_url,
        })
    } else {
        let model_base_url = resolve_model_profile_base_url(model_profile, model_base_url);
        let config = crate::BurnJepaModelBootstrapConfig {
            cache_root: cache_dir,
            model_profile,
            model_base_url,
            manifest_url,
        };
        Some(
            crate::resolve_or_bootstrap_burn_jepa_model_package_with_config_and_progress(
                &config,
                |message| eprintln!("{message}"),
            )?,
        )
    }
    .expect("package files");

    let manifest_json = std::fs::read_to_string(&package.manifest_path)?;
    let manifest = crate::BurnJepaPipelinePackageManifest::from_json_str(&manifest_json)?;
    let parts = package
        .part_paths
        .iter()
        .map(std::fs::read)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let device = Default::default();
    let (model, apply_result, runtime_dtype_counts) = match manifest.model_kind {
        crate::BurnJepaPackageModelKind::Base => {
            let (model, result) =
                crate::load_vjepa_burnpack_parts::<B>(&manifest.jepa_config, &parts, &device)?;
            let dtype_counts = crate::module_dtype_counts::<B, _>(&model);
            (NativeBpkModel::Base(Box::new(model)), result, dtype_counts)
        }
        crate::BurnJepaPackageModelKind::Ttt => {
            let ttt_config = manifest
                .ttt_config
                .clone()
                .ok_or_else(|| anyhow::anyhow!("TTT package manifest is missing ttt_config"))?;
            let (model, result) = crate::load_ttt_burnpack_parts::<B>(
                &manifest.jepa_config,
                ttt_config,
                &parts,
                &device,
            )?;
            let dtype_counts = crate::module_dtype_counts::<B, _>(&model);
            (NativeBpkModel::Ttt(Box::new(model)), result, dtype_counts)
        }
    };
    ensure!(
        apply_result.errors.is_empty(),
        "burnpack apply reported errors: {:?}",
        apply_result.errors
    );
    ensure!(
        runtime_dtype_counts.get("F16").copied().unwrap_or(0) == 0,
        "runtime model still contains F16 tensors after load: {:?}",
        runtime_dtype_counts
    );

    let video = verification_video::<B>(image_size, frames, &device)?;
    let output = encode_native_bpk_model(&model, video)?;
    let [batch, tokens, dim] = output.tokens.shape().dims::<3>();
    let values = output
        .tokens
        .clone()
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("read BPK output values: {err:?}"))?;
    let (sample_min, sample_max, sample_mean) = summarize_f32(&values);

    let checkpoint_parity = if let Some(checkpoint_dir) = checkpoint_dir {
        ensure!(
            manifest.model_kind == crate::BurnJepaPackageModelKind::Base,
            "checkpoint parity currently expects a base V-JEPA package"
        );
        let mut options = crate::VJepaLoadOptions::default();
        if let Some(weights_name) = weights_name {
            options.weights_name = weights_name;
        }
        let (checkpoint_model, _config, checkpoint_report) =
            options.load_model::<B>(&checkpoint_dir, &device)?;
        ensure!(
            checkpoint_report.errors.is_empty(),
            "checkpoint import reported errors: {:?}",
            checkpoint_report.errors
        );
        let checkpoint_output = checkpoint_model
            .encode_video(verification_video::<B>(image_size, frames, &device)?, None);
        let checkpoint_values = checkpoint_output
            .tokens
            .into_data()
            .to_vec::<f32>()
            .map_err(|err| anyhow::anyhow!("read checkpoint output values: {err:?}"))?;
        ensure!(
            checkpoint_values.len() == values.len(),
            "checkpoint/BPK output length mismatch: {} vs {}",
            checkpoint_values.len(),
            values.len()
        );
        let metrics = compare_f32(&values, &checkpoint_values);
        let within_tolerance = metrics.max_abs <= max_abs_tol && metrics.mean_abs <= mean_abs_tol;
        ensure!(
            within_tolerance,
            "BPK checkpoint parity exceeded tolerance: max_abs={} mean_abs={} tolerances=({}, {})",
            metrics.max_abs,
            metrics.mean_abs,
            max_abs_tol,
            mean_abs_tol
        );
        Some(BpkCheckpointParityReport {
            checkpoint_dir,
            checkpoint_applied: checkpoint_report.applied.len(),
            checkpoint_missing: checkpoint_report.missing.len(),
            checkpoint_skipped: checkpoint_report.skipped.len(),
            checkpoint_errors: checkpoint_report.errors.len(),
            max_abs: metrics.max_abs,
            mean_abs: metrics.mean_abs,
            rmse: metrics.rmse,
            cosine: metrics.cosine,
            within_tolerance,
        })
    } else {
        None
    };

    let burnpack_dtype_counts = crate::burnpack_parts_dtype_counts(&package.parts_manifest_path)?;
    ensure!(
        burnpack_dtype_counts.get("F16").copied().unwrap_or(0) > 0
            && burnpack_dtype_counts.get("F32").copied().unwrap_or(0) == 0,
        "deployment burnpack parts are not f16-only: {:?}",
        burnpack_dtype_counts
    );

    print_json(&BpkVerifyReport {
        manifest_path: package.manifest_path,
        parts_manifest_path: package.parts_manifest_path,
        part_count: package.part_paths.len(),
        total_bytes: package.total_bytes,
        model_kind: manifest.model_kind,
        record_dtype: manifest.record_dtype,
        burnpack_dtype_counts,
        runtime_dtype_counts,
        apply_applied: apply_result.applied.len(),
        apply_missing: apply_result.missing.len(),
        apply_skipped: apply_result.skipped.len(),
        apply_unused: apply_result.unused.len(),
        apply_errors: apply_result.errors.len(),
        output_shape: vec![batch, tokens, dim],
        grid: [output.grid.depth, output.grid.height, output.grid.width],
        sample_count: values.len(),
        sample_mean,
        sample_min,
        sample_max,
        checkpoint_parity,
        load_path: "burn_store::BurnpackStore + ModuleSnapshot::load_from clean init",
    })
}

#[cfg(feature = "ndarray")]
fn encode_native_bpk_model(
    model: &NativeBpkModel,
    video: burn::tensor::Tensor<burn::backend::NdArray<f32>, 5>,
) -> Result<crate::VJepaEncoderOutput<burn::backend::NdArray<f32>>> {
    match model {
        NativeBpkModel::Base(model) => Ok(model.encode_video(video, None)),
        NativeBpkModel::Ttt(model) => model.encode_video(video, None),
    }
}

#[cfg(feature = "ndarray")]
fn verification_video<B: burn::tensor::backend::Backend>(
    image_size: usize,
    frames: usize,
    device: &B::Device,
) -> Result<burn::tensor::Tensor<B, 5>> {
    let shape = crate::VJepaRgbaVideoShape::new(1, frames, image_size, image_size);
    let mut rgba = vec![0u8; shape.num_bytes()];
    for index in (0..rgba.len()).step_by(4) {
        let pixel = index / 4;
        let frame = pixel / (image_size * image_size);
        let spatial = pixel % (image_size * image_size);
        let y = spatial / image_size;
        let x = spatial % image_size;
        rgba[index] = ((x * 255 / image_size.max(1)) ^ (frame * 13)) as u8;
        rgba[index + 1] = ((y * 255 / image_size.max(1)) ^ (frame * 29)) as u8;
        rgba[index + 2] = ((x + y + frame * 17) % 256) as u8;
        rgba[index + 3] = 255;
    }
    crate::rgba_video_to_tensor::<B>(&rgba, shape, device)
}

#[cfg(feature = "ndarray")]
#[derive(Clone, Copy, Debug)]
struct CompareF32Metrics {
    max_abs: f32,
    mean_abs: f32,
    rmse: f32,
    cosine: f32,
}

#[cfg(feature = "ndarray")]
fn compare_f32(a: &[f32], b: &[f32]) -> CompareF32Metrics {
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&a, &b) in a.iter().zip(b) {
        let diff = (a - b).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff as f64;
        sum_sq += (diff as f64) * (diff as f64);
        dot += (a as f64) * (b as f64);
        norm_a += (a as f64) * (a as f64);
        norm_b += (b as f64) * (b as f64);
    }
    let len = a.len().max(1) as f64;
    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(f64::EPSILON);
    CompareF32Metrics {
        max_abs,
        mean_abs: (sum_abs / len) as f32,
        rmse: (sum_sq / len).sqrt() as f32,
        cosine: (dot / denom) as f32,
    }
}

#[cfg(feature = "ndarray")]
fn summarize_f32(values: &[f32]) -> (f32, f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &value in values {
        min = min.min(value);
        max = max.max(value);
        sum += value as f64;
    }
    (min, max, (sum / values.len() as f64) as f32)
}

fn dispatch_export_bpk(
    config: PathBuf,
    model: Option<PathBuf>,
    output: PathBuf,
    shard_mib: u64,
    overwrite_shards: bool,
    model_profile: Option<crate::BurnJepaModelProfile>,
    model_base_url: String,
    deploy_dir: Option<PathBuf>,
    overwrite_deploy: bool,
    allow_tiny_model: bool,
) -> Result<()> {
    #[cfg(feature = "ndarray")]
    {
        export_bpk_ndarray(
            config,
            model,
            output,
            shard_mib,
            overwrite_shards,
            model_profile,
            model_base_url,
            deploy_dir,
            overwrite_deploy,
            allow_tiny_model,
        )
    }
    #[cfg(not(feature = "ndarray"))]
    {
        let _ = (
            config,
            model,
            output,
            shard_mib,
            overwrite_shards,
            model_profile,
            model_base_url,
            deploy_dir,
            overwrite_deploy,
            allow_tiny_model,
        );
        bail!("export-bpk requires the ndarray feature so checkpoint import can run on CPU")
    }
}

#[cfg(feature = "ndarray")]
fn export_bpk_ndarray(
    config_path: PathBuf,
    model_path: Option<PathBuf>,
    output: PathBuf,
    shard_mib: u64,
    overwrite_shards: bool,
    model_profile: Option<crate::BurnJepaModelProfile>,
    model_base_url: String,
    deploy_dir: Option<PathBuf>,
    overwrite_deploy: bool,
    allow_tiny_model: bool,
) -> Result<()> {
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};

    let config = BurnJepaTrainConfig::from_toml_file(config_path)?;
    if !allow_tiny_model
        && config.model.checkpoint_dir.is_none()
        && config.model.config_path.is_none()
    {
        bail!(
            "export-bpk would export the tiny test V-JEPA model because the config has no model.checkpoint_dir or model.config_path; pass --allow-tiny-model only for smoke artifacts"
        );
    }
    let device = Default::default();
    let model_config = if let Some(path) = &config.model.config_path {
        crate::VJepaConfig::from_json_file(path)?
    } else if let Some(checkpoint_dir) = &config.model.checkpoint_dir {
        crate::load_config_from_hf_dir(
            checkpoint_dir,
            &crate::VJepaLoadOptions::default().config_name,
        )?
    } else {
        crate::VJepaConfig::tiny_for_tests()
    };
    let output = output.with_extension("bpk");
    let mut checkpoint_load_report = None;
    let mut checkpoint_source = None;
    let ttt_model_path = model_path.or(config.model.ttt_checkpoint_path.clone());
    let package_model_kind = if ttt_model_path.is_some() {
        crate::BurnJepaPackageModelKind::Ttt
    } else {
        crate::BurnJepaPackageModelKind::Base
    };
    let model_profile = model_profile.unwrap_or_else(|| {
        if ttt_model_path.is_some() {
            crate::BurnJepaModelProfile::Vjepa21Ttt
        } else {
            crate::BurnJepaModelProfile::Vjepa21Base
        }
    });
    ensure!(
        model_profile.model_kind() == package_model_kind,
        "--model-profile {} maps to a {} package, but this export config produces a {} package",
        model_profile,
        model_profile.model_kind().as_str(),
        package_model_kind.as_str()
    );
    let model_base_url = resolve_model_profile_base_url(model_profile, model_base_url);
    let package_manifest = if let Some(model_path) = ttt_model_path {
        let base = if let Some(checkpoint_dir) = &config.model.checkpoint_dir {
            let mut options = crate::VJepaLoadOptions::default();
            if let Some(weights_name) = &config.model.weights_name {
                options.weights_name = weights_name.clone();
            }
            let (model, _config, report) =
                options.load_model::<burn::backend::NdArray<f32>>(checkpoint_dir, &device)?;
            ensure_export_load_report_ok(&report)?;
            checkpoint_load_report = Some(report);
            checkpoint_source = Some(checkpoint_dir.clone());
            model
        } else {
            crate::VJepa2_1Model::<burn::backend::NdArray<f32>>::new(&model_config, &device)
        };
        use anyhow::Context as _;
        let ttt = crate::VJepaTttModel::from_model(base, config.ttt.clone(), &device)?
            .load_file(
                model_path.clone(),
                &NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
                &device,
            )
            .with_context(|| format!("load TTT model {}", model_path.display()))?;
        crate::save_ttt_burnpack(&ttt.no_grad(), &output)?;
        crate::BurnJepaPipelinePackageManifest {
            model_kind: crate::BurnJepaPackageModelKind::Ttt,
            record_dtype: Some("f16".to_string()),
            jepa_config: model_config,
            ttt_config: Some(config.ttt.clone()),
            model_base_url,
            ..crate::BurnJepaPipelinePackageManifest::default()
        }
        .with_burnpack_paths(&output)
    } else {
        let base = if let Some(checkpoint_dir) = &config.model.checkpoint_dir {
            let mut options = crate::VJepaLoadOptions::default();
            if let Some(weights_name) = &config.model.weights_name {
                options.weights_name = weights_name.clone();
            }
            let (model, _config, report) =
                options.load_model::<burn::backend::NdArray<f32>>(checkpoint_dir, &device)?;
            ensure_export_load_report_ok(&report)?;
            checkpoint_load_report = Some(report);
            checkpoint_source = Some(checkpoint_dir.clone());
            model
        } else {
            crate::VJepa2_1Model::<burn::backend::NdArray<f32>>::new(&model_config, &device)
        };
        crate::save_vjepa_burnpack(&base.no_grad(), &output)?;
        crate::BurnJepaPipelinePackageManifest {
            model_kind: crate::BurnJepaPackageModelKind::Base,
            record_dtype: Some("f16".to_string()),
            jepa_config: model_config,
            ttt_config: None,
            model_base_url,
            ..crate::BurnJepaPipelinePackageManifest::default()
        }
        .with_burnpack_paths(&output)
    };
    let burnpack_dtype_counts = crate::burnpack_dtype_counts(&output)?;
    ensure_export_burnpack_is_f16(&burnpack_dtype_counts)?;
    let max_part_bytes = shard_mib
        .max(1)
        .checked_mul(1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("--shard-mib overflow"))?;
    let parts = crate::write_burnpack_parts_for_browser(&output, max_part_bytes, overwrite_shards)?;
    let manifest_path = output
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join("manifest.json");
    crate::write_pipeline_package_manifest(&manifest_path, &package_manifest)?;
    let deploy_bundle = deploy_dir
        .map(|dir| {
            crate::write_burn_jepa_model_deploy_bundle(&manifest_path, dir, overwrite_deploy)
        })
        .transpose()?;
    print_json(&serde_json::json!({
        "burnpack": output,
        "package_manifest": manifest_path,
        "parts_manifest": parts.manifest_path,
        "parts": parts.part_paths,
        "total_bytes": parts.total_bytes,
        "record_dtype": package_manifest.record_dtype.clone(),
        "burnpack_dtype_counts": burnpack_dtype_counts,
        "model_base_url": package_manifest.model_base_url.clone(),
        "model_profile": model_profile.as_str(),
        "checkpoint_source": checkpoint_source,
        "checkpoint_load_report": export_load_report_json(checkpoint_load_report.as_ref()),
        "deploy_bundle": deploy_bundle,
    }))
}

#[cfg(feature = "ndarray")]
fn ensure_export_burnpack_is_f16(
    dtype_counts: &std::collections::BTreeMap<String, usize>,
) -> Result<()> {
    if dtype_counts.get("F16").copied().unwrap_or(0) == 0 {
        bail!("exported burnpack did not contain any F16 tensors");
    }
    if dtype_counts.get("F32").copied().unwrap_or(0) > 0 {
        bail!(
            "exported burnpack still contains F32 tensors: {:?}",
            dtype_counts
        );
    }
    Ok(())
}

#[cfg(feature = "ndarray")]
fn ensure_export_load_report_ok(report: &crate::LoadReport) -> Result<()> {
    if !report.errors.is_empty() {
        bail!(
            "V-JEPA checkpoint import reported tensor errors: {}",
            report.errors.join("; ")
        );
    }
    if report.applied.is_empty() {
        bail!("V-JEPA checkpoint import did not apply any tensors");
    }
    Ok(())
}

#[cfg(feature = "ndarray")]
fn export_load_report_json(report: Option<&crate::LoadReport>) -> serde_json::Value {
    match report {
        Some(report) => serde_json::json!({
            "applied": report.applied.len(),
            "missing": report.missing.len(),
            "skipped": report.skipped.len(),
            "errors": report.errors.len(),
            "missing_examples": report.missing.iter().take(8).collect::<Vec<_>>(),
            "skipped_examples": report.skipped.iter().take(8).collect::<Vec<_>>(),
        }),
        None => serde_json::Value::Null,
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_export_anyup_bpk(
    weights: PathBuf,
    output: PathBuf,
    shard_mib: u64,
    overwrite_shards: bool,
    model_profile: crate::BurnAnyUpModelProfile,
    model_base_url: String,
    deploy_dir: PathBuf,
    overwrite_deploy: bool,
    allow_random_model: bool,
) -> Result<()> {
    #[cfg(feature = "ndarray")]
    {
        export_anyup_bpk_ndarray(
            weights,
            output,
            shard_mib,
            overwrite_shards,
            model_profile,
            model_base_url,
            deploy_dir,
            overwrite_deploy,
            allow_random_model,
        )
    }
    #[cfg(not(feature = "ndarray"))]
    {
        let _ = (
            weights,
            output,
            shard_mib,
            overwrite_shards,
            model_profile,
            model_base_url,
            deploy_dir,
            overwrite_deploy,
            allow_random_model,
        );
        bail!(
            "export-anyup-bpk requires the ndarray feature so AnyUp checkpoint import can run on CPU"
        )
    }
}

#[cfg(feature = "ndarray")]
#[allow(clippy::too_many_arguments)]
fn export_anyup_bpk_ndarray(
    weights: PathBuf,
    output: PathBuf,
    shard_mib: u64,
    overwrite_shards: bool,
    model_profile: crate::BurnAnyUpModelProfile,
    model_base_url: String,
    deploy_dir: PathBuf,
    overwrite_deploy: bool,
    allow_random_model: bool,
) -> Result<()> {
    use burn::module::Module;

    type B = burn::backend::NdArray<f32>;

    let device = Default::default();
    let anyup_config = crate::AnyUpConfig {
        input_dim: 3,
        ..Default::default()
    };
    let mut anyup = crate::AnyUp::<B>::new(anyup_config.clone(), &device)?;
    let mut load_report = None;
    if weights.exists() {
        let report = crate::AnyUpLoadOptions::default()
            .load_into(&mut anyup, &weights, &device)
            .map_err(|err| anyhow::anyhow!("load AnyUp weights {}: {err}", weights.display()))?;
        ensure_anyup_load_report_ok(&report)?;
        load_report = Some(report);
    } else if !allow_random_model {
        bail!(
            "AnyUp weights `{}` do not exist; pass --allow-random-model only for smoke artifacts",
            weights.display()
        );
    }

    let output = output.with_extension("bpk");
    crate::save_anyup_burnpack(&anyup.no_grad(), &output)?;
    let burnpack_dtype_counts = crate::burnpack_dtype_counts(&output)?;
    ensure_export_burnpack_is_f16(&burnpack_dtype_counts)?;
    let max_part_bytes = shard_mib
        .max(1)
        .checked_mul(1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("--shard-mib overflow"))?;
    let parts = crate::write_burnpack_parts_for_browser(&output, max_part_bytes, overwrite_shards)?;
    let model_base_url = resolve_anyup_model_profile_base_url(model_profile, model_base_url);
    let manifest = crate::BurnAnyUpPackageManifest {
        record_dtype: Some("f16".to_string()),
        anyup_config,
        model_base_url,
        ..crate::BurnAnyUpPackageManifest::default()
    }
    .with_burnpack_paths(&output);
    let manifest_path = output
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join("manifest.json");
    crate::write_anyup_package_manifest(&manifest_path, &manifest)?;
    let deploy_bundle =
        crate::write_burn_anyup_model_deploy_bundle(&manifest_path, deploy_dir, overwrite_deploy)?;

    print_json(&serde_json::json!({
        "burnpack": output,
        "package_manifest": manifest_path,
        "parts_manifest": parts.manifest_path,
        "parts": parts.part_paths,
        "total_bytes": parts.total_bytes,
        "record_dtype": manifest.record_dtype.clone(),
        "burnpack_dtype_counts": burnpack_dtype_counts,
        "model_base_url": manifest.model_base_url.clone(),
        "model_profile": model_profile.as_str(),
        "weights": weights,
        "load_report": anyup_load_report_json(load_report.as_ref()),
        "deploy_bundle": deploy_bundle,
    }))
}

fn dispatch_verify_anyup_bpk(
    manifest: Option<PathBuf>,
    model_profile: crate::BurnAnyUpModelProfile,
    model_base_url: String,
    manifest_url: Option<String>,
    cache_dir: Option<PathBuf>,
    image_size: usize,
) -> Result<()> {
    #[cfg(feature = "ndarray")]
    {
        verify_anyup_bpk_ndarray(
            manifest,
            model_profile,
            model_base_url,
            manifest_url,
            cache_dir,
            image_size,
        )
    }
    #[cfg(not(feature = "ndarray"))]
    {
        let _ = (
            manifest,
            model_profile,
            model_base_url,
            manifest_url,
            cache_dir,
            image_size,
        );
        bail!(
            "verify-anyup-bpk requires the ndarray feature so native numerical checks can run on CPU"
        )
    }
}

#[cfg(feature = "ndarray")]
#[derive(Debug, Serialize)]
struct AnyUpBpkVerifyReport {
    manifest_path: PathBuf,
    parts_manifest_path: PathBuf,
    part_count: usize,
    total_bytes: u64,
    record_dtype: Option<String>,
    burnpack_dtype_counts: std::collections::BTreeMap<String, usize>,
    runtime_dtype_counts: std::collections::BTreeMap<String, usize>,
    apply_applied: usize,
    apply_missing: usize,
    apply_skipped: usize,
    apply_unused: usize,
    apply_errors: usize,
    output_shape: Vec<usize>,
    sample_count: usize,
    sample_mean: f32,
    sample_min: f32,
    sample_max: f32,
    load_path: &'static str,
}

#[cfg(feature = "ndarray")]
fn verify_anyup_bpk_ndarray(
    manifest_path: Option<PathBuf>,
    model_profile: crate::BurnAnyUpModelProfile,
    model_base_url: String,
    manifest_url: Option<String>,
    cache_dir: Option<PathBuf>,
    image_size: usize,
) -> Result<()> {
    ensure!(image_size >= 8, "--image-size must be at least 8");
    type B = burn::backend::NdArray<f32>;

    let package = if let Some(manifest_path) = manifest_path {
        let manifest_json = std::fs::read_to_string(&manifest_path)?;
        let manifest = crate::BurnAnyUpPackageManifest::from_json_str(&manifest_json)?;
        let parts_manifest_path =
            crate::resolve_package_manifest_entry_path(&manifest_path, &manifest.parts_manifest)?;
        let parts_manifest = crate::read_parts_manifest(&parts_manifest_path)?;
        let part_paths = parts_manifest
            .parts
            .iter()
            .map(|part| crate::resolve_part_entry_path(&parts_manifest_path, &part.path))
            .collect::<Result<Vec<_>>>()?;
        crate::BurnAnyUpModelPackageFiles {
            cache_root: manifest_path
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .to_path_buf(),
            manifest_path,
            parts_manifest_path,
            part_paths,
            total_bytes: parts_manifest.total_bytes,
            model_base_url: manifest.model_base_url,
        }
    } else {
        let model_base_url = resolve_anyup_model_profile_base_url(model_profile, model_base_url);
        let config = crate::BurnAnyUpModelBootstrapConfig {
            cache_root: cache_dir,
            model_profile,
            model_base_url,
            manifest_url,
        };
        crate::resolve_or_bootstrap_burn_anyup_model_package_with_config_and_progress(
            &config,
            |message| eprintln!("{message}"),
        )?
    };

    let manifest_json = std::fs::read_to_string(&package.manifest_path)?;
    let manifest = crate::BurnAnyUpPackageManifest::from_json_str(&manifest_json)?;
    let parts = package
        .part_paths
        .iter()
        .map(std::fs::read)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let device = Default::default();
    let (model, apply_result) =
        crate::load_anyup_burnpack_parts::<B>(&manifest.anyup_config, &parts, &device)?;
    ensure!(
        apply_result.errors.is_empty(),
        "burn_anyup apply reported errors: {:?}",
        apply_result.errors
    );
    ensure!(
        !apply_result.applied.is_empty(),
        "burn_anyup package did not apply any tensors"
    );
    let runtime_dtype_counts = crate::module_dtype_counts::<B, _>(&model);
    ensure!(
        runtime_dtype_counts.get("F16").copied().unwrap_or(0) == 0,
        "runtime AnyUp model still contains F16 tensors after load: {:?}",
        runtime_dtype_counts
    );

    let image = burn::tensor::Tensor::<B, 4>::ones([1, 3, image_size, image_size], &device);
    let low = (image_size / 8).max(2);
    let features =
        burn::tensor::Tensor::<B, 4>::ones([1, manifest.anyup_config.qk_dim, low, low], &device);
    let output = model.forward(image, features, Some([image_size, image_size]), Some(16));
    let [batch, channels, height, width] = output.shape().dims::<4>();
    let values = output
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("read AnyUp BPK output values: {err:?}"))?;
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "AnyUp BPK output contains non-finite values"
    );
    let (sample_min, sample_max, sample_mean) = summarize_f32(&values);
    let burnpack_dtype_counts = crate::burnpack_parts_dtype_counts(&package.parts_manifest_path)?;
    ensure!(
        burnpack_dtype_counts.get("F16").copied().unwrap_or(0) > 0
            && burnpack_dtype_counts.get("F32").copied().unwrap_or(0) == 0,
        "deployment burn_anyup burnpack parts are not f16-only: {:?}",
        burnpack_dtype_counts
    );

    print_json(&AnyUpBpkVerifyReport {
        manifest_path: package.manifest_path,
        parts_manifest_path: package.parts_manifest_path,
        part_count: package.part_paths.len(),
        total_bytes: package.total_bytes,
        record_dtype: manifest.record_dtype,
        burnpack_dtype_counts,
        runtime_dtype_counts,
        apply_applied: apply_result.applied.len(),
        apply_missing: apply_result.missing.len(),
        apply_skipped: apply_result.skipped.len(),
        apply_unused: apply_result.unused.len(),
        apply_errors: apply_result.errors.len(),
        output_shape: vec![batch, channels, height, width],
        sample_count: values.len(),
        sample_mean,
        sample_min,
        sample_max,
        load_path: "burn_store::BurnpackStore + ModuleSnapshot::load_from clean init",
    })
}

#[cfg(feature = "ndarray")]
fn ensure_anyup_load_report_ok(report: &crate::AnyUpLoadReport) -> Result<()> {
    if !report.errors.is_empty() {
        bail!(
            "AnyUp checkpoint import reported tensor errors: {}",
            report.errors.join("; ")
        );
    }
    if report.applied.is_empty() {
        bail!("AnyUp checkpoint import did not apply any tensors");
    }
    Ok(())
}

#[cfg(feature = "ndarray")]
fn anyup_load_report_json(report: Option<&crate::AnyUpLoadReport>) -> serde_json::Value {
    match report {
        Some(report) => serde_json::json!({
            "applied": report.applied.len(),
            "missing": report.missing.len(),
            "skipped": report.skipped.len(),
            "errors": report.errors.len(),
            "missing_examples": report.missing.iter().take(8).collect::<Vec<_>>(),
            "skipped_examples": report.skipped.iter().take(8).collect::<Vec<_>>(),
        }),
        None => serde_json::Value::Null,
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_export_reconstruction_bpk(
    output: PathBuf,
    shard_mib: u64,
    overwrite_shards: bool,
    model_profile: crate::BurnJepaReconstructionModelProfile,
    model_base_url: String,
    deploy_dir: PathBuf,
    overwrite_deploy: bool,
    input_dim: usize,
    hidden_dim: usize,
    patch_size: usize,
) -> Result<()> {
    #[cfg(feature = "ndarray")]
    {
        export_reconstruction_bpk_ndarray(
            output,
            shard_mib,
            overwrite_shards,
            model_profile,
            model_base_url,
            deploy_dir,
            overwrite_deploy,
            input_dim,
            hidden_dim,
            patch_size,
        )
    }
    #[cfg(not(feature = "ndarray"))]
    {
        let _ = (
            output,
            shard_mib,
            overwrite_shards,
            model_profile,
            model_base_url,
            deploy_dir,
            overwrite_deploy,
            input_dim,
            hidden_dim,
            patch_size,
        );
        bail!("export-reconstruction-bpk requires the ndarray feature")
    }
}

#[cfg(feature = "ndarray")]
#[allow(clippy::too_many_arguments)]
fn export_reconstruction_bpk_ndarray(
    output: PathBuf,
    shard_mib: u64,
    overwrite_shards: bool,
    model_profile: crate::BurnJepaReconstructionModelProfile,
    model_base_url: String,
    deploy_dir: PathBuf,
    overwrite_deploy: bool,
    input_dim: usize,
    hidden_dim: usize,
    patch_size: usize,
) -> Result<()> {
    use burn::module::Module;

    type B = burn::backend::NdArray<f32>;

    let device = Default::default();
    let reconstruction_config = crate::JepaReconstructionConfig {
        input_dim,
        hidden_dim,
        patch_size,
        ..crate::JepaReconstructionConfig::default()
    };
    let decoder =
        crate::JepaReconstructionDecoder::<B>::new(reconstruction_config.clone(), &device)?;
    let output = output.with_extension("bpk");
    crate::save_jepa_reconstruction_burnpack(&decoder.no_grad(), &output)?;
    let burnpack_dtype_counts = crate::burnpack_dtype_counts(&output)?;
    ensure_export_burnpack_is_f16(&burnpack_dtype_counts)?;
    let max_part_bytes = shard_mib
        .max(1)
        .checked_mul(1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("--shard-mib overflow"))?;
    let parts = crate::write_burnpack_parts_for_browser(&output, max_part_bytes, overwrite_shards)?;
    let model_base_url =
        resolve_reconstruction_model_profile_base_url(model_profile, model_base_url);
    let manifest = crate::BurnJepaReconstructionPackageManifest {
        record_dtype: Some("f16".to_string()),
        reconstruction_config,
        model_base_url,
        ..crate::BurnJepaReconstructionPackageManifest::default()
    }
    .with_burnpack_paths(&output);
    let manifest_path = output
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join("manifest.json");
    crate::write_jepa_reconstruction_package_manifest(&manifest_path, &manifest)?;
    let deploy_bundle = crate::write_burn_jepa_reconstruction_model_deploy_bundle(
        &manifest_path,
        deploy_dir,
        overwrite_deploy,
    )?;

    print_json(&serde_json::json!({
        "burnpack": output,
        "package_manifest": manifest_path,
        "parts_manifest": parts.manifest_path,
        "parts": parts.part_paths,
        "total_bytes": parts.total_bytes,
        "record_dtype": manifest.record_dtype.clone(),
        "burnpack_dtype_counts": burnpack_dtype_counts,
        "model_base_url": manifest.model_base_url.clone(),
        "model_profile": model_profile.as_str(),
        "deploy_bundle": deploy_bundle,
    }))
}

fn dispatch_verify_reconstruction_bpk(
    manifest: Option<PathBuf>,
    model_profile: crate::BurnJepaReconstructionModelProfile,
    model_base_url: String,
    manifest_url: Option<String>,
    cache_dir: Option<PathBuf>,
    image_size: usize,
) -> Result<()> {
    #[cfg(feature = "ndarray")]
    {
        verify_reconstruction_bpk_ndarray(
            manifest,
            model_profile,
            model_base_url,
            manifest_url,
            cache_dir,
            image_size,
        )
    }
    #[cfg(not(feature = "ndarray"))]
    {
        let _ = (
            manifest,
            model_profile,
            model_base_url,
            manifest_url,
            cache_dir,
            image_size,
        );
        bail!("verify-reconstruction-bpk requires the ndarray feature")
    }
}

#[cfg(feature = "ndarray")]
#[derive(Debug, Serialize)]
struct ReconstructionBpkVerifyReport {
    manifest_path: PathBuf,
    parts_manifest_path: PathBuf,
    part_count: usize,
    total_bytes: u64,
    record_dtype: Option<String>,
    burnpack_dtype_counts: std::collections::BTreeMap<String, usize>,
    runtime_dtype_counts: std::collections::BTreeMap<String, usize>,
    apply_applied: usize,
    apply_missing: usize,
    apply_skipped: usize,
    apply_unused: usize,
    apply_errors: usize,
    output_shape: Vec<usize>,
    psnr_db: Option<f64>,
    sample_count: usize,
    sample_mean: f32,
    sample_min: f32,
    sample_max: f32,
    load_path: &'static str,
}

#[cfg(feature = "ndarray")]
fn verify_reconstruction_bpk_ndarray(
    manifest_path: Option<PathBuf>,
    model_profile: crate::BurnJepaReconstructionModelProfile,
    model_base_url: String,
    manifest_url: Option<String>,
    cache_dir: Option<PathBuf>,
    image_size: usize,
) -> Result<()> {
    ensure!(image_size >= 16, "--image-size must be at least 16");
    type B = burn::backend::NdArray<f32>;

    let package = if let Some(manifest_path) = manifest_path {
        let manifest_json = std::fs::read_to_string(&manifest_path)?;
        let manifest = crate::BurnJepaReconstructionPackageManifest::from_json_str(&manifest_json)?;
        let parts_manifest_path =
            crate::resolve_package_manifest_entry_path(&manifest_path, &manifest.parts_manifest)?;
        let parts_manifest = crate::read_parts_manifest(&parts_manifest_path)?;
        let part_paths = parts_manifest
            .parts
            .iter()
            .map(|part| crate::resolve_part_entry_path(&parts_manifest_path, &part.path))
            .collect::<Result<Vec<_>>>()?;
        crate::BurnJepaReconstructionModelPackageFiles {
            cache_root: manifest_path
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .to_path_buf(),
            manifest_path,
            parts_manifest_path,
            part_paths,
            total_bytes: parts_manifest.total_bytes,
            model_base_url: manifest.model_base_url,
        }
    } else {
        let model_base_url =
            resolve_reconstruction_model_profile_base_url(model_profile, model_base_url);
        let config = crate::BurnJepaReconstructionModelBootstrapConfig {
            cache_root: cache_dir,
            model_profile,
            model_base_url,
            manifest_url,
        };
        crate::resolve_or_bootstrap_burn_jepa_reconstruction_model_package_with_config_and_progress(
            &config,
            |message| eprintln!("{message}"),
        )?
    };

    let manifest_json = std::fs::read_to_string(&package.manifest_path)?;
    let manifest = crate::BurnJepaReconstructionPackageManifest::from_json_str(&manifest_json)?;
    let parts = package
        .part_paths
        .iter()
        .map(std::fs::read)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let device = Default::default();
    let (decoder, apply_result) = crate::load_jepa_reconstruction_burnpack_parts::<B>(
        &manifest.reconstruction_config,
        &parts,
        &device,
    )?;
    ensure!(
        apply_result.errors.is_empty(),
        "burn_jepa_reconstruction apply reported errors: {:?}",
        apply_result.errors
    );
    ensure!(
        !apply_result.applied.is_empty(),
        "burn_jepa_reconstruction package did not apply any tensors"
    );
    let runtime_dtype_counts = crate::module_dtype_counts::<B, _>(&decoder);
    ensure!(
        runtime_dtype_counts.get("F16").copied().unwrap_or(0) == 0,
        "runtime reconstruction model still contains F16 tensors after load: {:?}",
        runtime_dtype_counts
    );

    let grid = (image_size / manifest.reconstruction_config.patch_size.max(1)).max(1);
    let features = burn::tensor::Tensor::<B, 4>::ones(
        [1, manifest.reconstruction_config.input_dim, grid, grid],
        &device,
    );
    let target =
        burn::tensor::Tensor::<B, 4>::ones([1, 3, image_size, image_size], &device).mul_scalar(0.5);
    let output = decoder.forward_to_size(features, [image_size, image_size]);
    let psnr_db = crate::reconstruction_psnr_scalar(output.clone(), target, 1.0);
    let [batch, channels, height, width] = output.shape().dims::<4>();
    let values = output
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("read reconstruction BPK output values: {err:?}"))?;
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "reconstruction BPK output contains non-finite values"
    );
    let (sample_min, sample_max, sample_mean) = summarize_f32(&values);
    let burnpack_dtype_counts = crate::burnpack_parts_dtype_counts(&package.parts_manifest_path)?;
    ensure!(
        burnpack_dtype_counts.get("F16").copied().unwrap_or(0) > 0
            && burnpack_dtype_counts.get("F32").copied().unwrap_or(0) == 0,
        "deployment reconstruction burnpack parts are not f16-only: {:?}",
        burnpack_dtype_counts
    );

    print_json(&ReconstructionBpkVerifyReport {
        manifest_path: package.manifest_path,
        parts_manifest_path: package.parts_manifest_path,
        part_count: package.part_paths.len(),
        total_bytes: package.total_bytes,
        record_dtype: manifest.record_dtype,
        burnpack_dtype_counts,
        runtime_dtype_counts,
        apply_applied: apply_result.applied.len(),
        apply_missing: apply_result.missing.len(),
        apply_skipped: apply_result.skipped.len(),
        apply_unused: apply_result.unused.len(),
        apply_errors: apply_result.errors.len(),
        output_shape: vec![batch, channels, height, width],
        psnr_db,
        sample_count: values.len(),
        sample_mean,
        sample_min,
        sample_max,
        load_path: "burn_store::BurnpackStore + ModuleSnapshot::load_from clean init",
    })
}

fn run_experiment_command(command: ExperimentCommand) -> Result<()> {
    match command {
        ExperimentCommand::Plan { config } => {
            let config = ExperimentConfig::from_toml_file(config)?;
            print_json(&write_experiment_plan(&config)?)
        }
        ExperimentCommand::PrepareData { config } => {
            let config = ExperimentConfig::from_toml_file(config)?;
            print_json(&prepare_experiment_data(&config)?)
        }
        ExperimentCommand::Run { config } => {
            let config = ExperimentConfig::from_toml_file(config)?;
            let report = dispatch_experiment(&config)?;
            print_json(&report)
        }
        ExperimentCommand::Analyze { run_dir } => print_json(&analyze_experiment(run_dir)?),
    }
}

fn dispatch_experiment(config: &ExperimentConfig) -> Result<ExperimentRunReport> {
    match config.base.training.backend {
        JepaTrainBackend::NdArray => {
            #[cfg(feature = "ndarray")]
            {
                let device = Default::default();
                run_experiment::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>(
                    config, &device,
                )
            }
            #[cfg(not(feature = "ndarray"))]
            {
                bail!("ndarray backend requested but the ndarray feature is not enabled")
            }
        }
        JepaTrainBackend::Flex => {
            #[cfg(feature = "flex")]
            {
                let device = Default::default();
                run_experiment::<burn::backend::Autodiff<burn::backend::Flex<f32, i32>>>(
                    config, &device,
                )
            }
            #[cfg(not(feature = "flex"))]
            {
                bail!("flex backend requested but the flex feature is not enabled")
            }
        }
        JepaTrainBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                cuda_runtime_preflight(CUDA_TRAIN_FORCE_ENV)
                    .map_err(|reason| anyhow::anyhow!("cuda backend unavailable: {reason}"))?;
                let device = Default::default();
                run_experiment::<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>(
                    config, &device,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("cuda backend requested but the cuda feature is not enabled")
            }
        }
        JepaTrainBackend::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                let device = Default::default();
                #[cfg(feature = "sparse-patchify-wgpu")]
                if wants_frozen_sparse_patchify_backend(&config.base) {
                    return run_experiment::<
                        burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
                    >(config, &device);
                }
                run_experiment::<burn::backend::Autodiff<burn::backend::Wgpu<f32, i32>>>(
                    config, &device,
                )
            }
            #[cfg(not(feature = "wgpu"))]
            {
                bail!("wgpu backend requested but the wgpu feature is not enabled")
            }
        }
        JepaTrainBackend::WebGpu => {
            #[cfg(feature = "webgpu")]
            {
                let device = Default::default();
                run_experiment::<burn::backend::Autodiff<burn::backend::WebGpu<f32, i32>>>(
                    config, &device,
                )
            }
            #[cfg(not(feature = "webgpu"))]
            {
                bail!("webgpu backend requested but the webgpu feature is not enabled")
            }
        }
        JepaTrainBackend::Dispatch => {
            #[cfg(feature = "dispatch")]
            {
                let device = dispatch_autodiff_device(&config.base.training)?;
                run_experiment::<burn::Dispatch>(config, &device)
            }
            #[cfg(not(feature = "dispatch"))]
            {
                bail!("dispatch backend requested but the dispatch feature is not enabled")
            }
        }
    }
}

fn dispatch_ttt(config: &BurnJepaTrainConfig) -> Result<TttTrainingReport> {
    match config.training.backend {
        JepaTrainBackend::NdArray => {
            #[cfg(feature = "ndarray")]
            {
                run_ttt::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>(config)
            }
            #[cfg(not(feature = "ndarray"))]
            {
                bail!("ndarray backend requested but the ndarray feature is not enabled")
            }
        }
        JepaTrainBackend::Flex => {
            #[cfg(feature = "flex")]
            {
                run_ttt::<burn::backend::Autodiff<burn::backend::Flex<f32, i32>>>(config)
            }
            #[cfg(not(feature = "flex"))]
            {
                bail!("flex backend requested but the flex feature is not enabled")
            }
        }
        JepaTrainBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                cuda_runtime_preflight(CUDA_TRAIN_FORCE_ENV)
                    .map_err(|reason| anyhow::anyhow!("cuda backend unavailable: {reason}"))?;
                run_ttt::<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>(config)
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("cuda backend requested but the cuda feature is not enabled")
            }
        }
        JepaTrainBackend::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                #[cfg(feature = "sparse-patchify-wgpu")]
                if wants_frozen_sparse_patchify_backend(config) {
                    return run_ttt::<
                        burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
                    >(config);
                }
                run_ttt::<burn::backend::Autodiff<burn::backend::Wgpu<f32, i32>>>(config)
            }
            #[cfg(not(feature = "wgpu"))]
            {
                bail!("wgpu backend requested but the wgpu feature is not enabled")
            }
        }
        JepaTrainBackend::WebGpu => {
            #[cfg(feature = "webgpu")]
            {
                run_ttt::<burn::backend::Autodiff<burn::backend::WebGpu<f32, i32>>>(config)
            }
            #[cfg(not(feature = "webgpu"))]
            {
                bail!("webgpu backend requested but the webgpu feature is not enabled")
            }
        }
        JepaTrainBackend::Dispatch => {
            #[cfg(feature = "dispatch")]
            {
                let device = dispatch_autodiff_device(&config.training)?;
                train_ttt_distillation::<burn::Dispatch>(config, &device)
            }
            #[cfg(not(feature = "dispatch"))]
            {
                bail!("dispatch backend requested but the dispatch feature is not enabled")
            }
        }
    }
}

fn dispatch_ttt_eval(
    config: &BurnJepaTrainConfig,
    model: Option<PathBuf>,
    base_sparse: bool,
    steps: usize,
) -> Result<TttEvalReport> {
    if base_sparse && model.is_some() {
        bail!(
            "--base-sparse evaluates zero-init/base sparse V-JEPA and cannot be combined with --model"
        );
    }
    if !base_sparse && model.is_none() {
        bail!("eval-ttt requires --model unless --base-sparse is set");
    }
    match config.training.backend {
        JepaTrainBackend::NdArray => {
            #[cfg(feature = "ndarray")]
            {
                run_ttt_eval::<burn::backend::NdArray<f32>>(config, model, base_sparse, steps)
            }
            #[cfg(not(feature = "ndarray"))]
            {
                bail!("ndarray backend requested but the ndarray feature is not enabled")
            }
        }
        JepaTrainBackend::Flex => {
            #[cfg(feature = "flex")]
            {
                run_ttt_eval::<burn::backend::Flex<f32, i32>>(config, model, base_sparse, steps)
            }
            #[cfg(not(feature = "flex"))]
            {
                bail!("flex backend requested but the flex feature is not enabled")
            }
        }
        JepaTrainBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                cuda_runtime_preflight(CUDA_TRAIN_FORCE_ENV)
                    .map_err(|reason| anyhow::anyhow!("cuda backend unavailable: {reason}"))?;
                run_ttt_eval::<burn::backend::Cuda<f32, i32>>(config, model, base_sparse, steps)
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("cuda backend requested but the cuda feature is not enabled")
            }
        }
        JepaTrainBackend::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                run_ttt_eval::<burn::backend::Wgpu<f32, i32>>(config, model, base_sparse, steps)
            }
            #[cfg(not(feature = "wgpu"))]
            {
                bail!("wgpu backend requested but the wgpu feature is not enabled")
            }
        }
        JepaTrainBackend::WebGpu => {
            #[cfg(feature = "webgpu")]
            {
                run_ttt_eval::<burn::backend::WebGpu<f32, i32>>(config, model, base_sparse, steps)
            }
            #[cfg(not(feature = "webgpu"))]
            {
                bail!("webgpu backend requested but the webgpu feature is not enabled")
            }
        }
        JepaTrainBackend::Dispatch => {
            #[cfg(feature = "dispatch")]
            {
                let device = dispatch_inner_device(config.training.dispatch_backend)?;
                if base_sparse {
                    evaluate_ttt_base_sparse::<burn::Dispatch>(config, &device, steps)
                } else {
                    evaluate_ttt_model_file::<burn::Dispatch>(
                        config,
                        model.expect("model checked above"),
                        &device,
                        steps,
                    )
                }
            }
            #[cfg(not(feature = "dispatch"))]
            {
                bail!("dispatch backend requested but the dispatch feature is not enabled")
            }
        }
    }
}

fn dispatch_dense(config: &BurnJepaTrainConfig) -> Result<DenseJepaTrainingReport> {
    match config.training.backend {
        JepaTrainBackend::NdArray => {
            #[cfg(feature = "ndarray")]
            {
                run_dense::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>(config)
            }
            #[cfg(not(feature = "ndarray"))]
            {
                bail!("ndarray backend requested but the ndarray feature is not enabled")
            }
        }
        JepaTrainBackend::Flex => {
            #[cfg(feature = "flex")]
            {
                run_dense::<burn::backend::Autodiff<burn::backend::Flex<f32, i32>>>(config)
            }
            #[cfg(not(feature = "flex"))]
            {
                bail!("flex backend requested but the flex feature is not enabled")
            }
        }
        JepaTrainBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                cuda_runtime_preflight(CUDA_TRAIN_FORCE_ENV)
                    .map_err(|reason| anyhow::anyhow!("cuda backend unavailable: {reason}"))?;
                run_dense::<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>(config)
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("cuda backend requested but the cuda feature is not enabled")
            }
        }
        JepaTrainBackend::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                run_dense::<burn::backend::Autodiff<burn::backend::Wgpu<f32, i32>>>(config)
            }
            #[cfg(not(feature = "wgpu"))]
            {
                bail!("wgpu backend requested but the wgpu feature is not enabled")
            }
        }
        JepaTrainBackend::WebGpu => {
            #[cfg(feature = "webgpu")]
            {
                run_dense::<burn::backend::Autodiff<burn::backend::WebGpu<f32, i32>>>(config)
            }
            #[cfg(not(feature = "webgpu"))]
            {
                bail!("webgpu backend requested but the webgpu feature is not enabled")
            }
        }
        JepaTrainBackend::Dispatch => {
            #[cfg(feature = "dispatch")]
            {
                let device = dispatch_autodiff_device(&config.training)?;
                train_dense_jepa::<burn::Dispatch>(config, &device)
            }
            #[cfg(not(feature = "dispatch"))]
            {
                bail!("dispatch backend requested but the dispatch feature is not enabled")
            }
        }
    }
}

#[cfg(feature = "dispatch")]
fn dispatch_autodiff_device(training: &TrainingLoopConfig) -> Result<burn::DispatchDevice> {
    Ok(burn::DispatchDevice::autodiff(dispatch_inner_device(
        training.dispatch_backend,
    )?))
}

#[cfg(feature = "dispatch")]
fn dispatch_inner_device(target: JepaDispatchBackend) -> Result<burn::DispatchDevice> {
    match target {
        JepaDispatchBackend::Auto => dispatch_auto_device(),
        JepaDispatchBackend::NdArray => {
            #[cfg(feature = "ndarray")]
            {
                Ok(burn::DispatchDevice::NdArray(Default::default()))
            }
            #[cfg(not(feature = "ndarray"))]
            {
                bail!("dispatch ndarray requested but the ndarray feature is not enabled")
            }
        }
        JepaDispatchBackend::Flex => {
            #[cfg(feature = "flex")]
            {
                Ok(burn::DispatchDevice::Flex(Default::default()))
            }
            #[cfg(not(feature = "flex"))]
            {
                bail!("dispatch flex requested but the flex feature is not enabled")
            }
        }
        JepaDispatchBackend::Wgpu | JepaDispatchBackend::WebGpu => {
            #[cfg(any(feature = "wgpu", feature = "webgpu"))]
            {
                Ok(burn::DispatchDevice::Wgpu(Default::default()))
            }
            #[cfg(not(any(feature = "wgpu", feature = "webgpu")))]
            {
                bail!("dispatch wgpu requested but neither the wgpu nor webgpu feature is enabled")
            }
        }
        JepaDispatchBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                cuda_runtime_preflight(CUDA_TRAIN_FORCE_ENV)
                    .map_err(|reason| anyhow::anyhow!("cuda backend unavailable: {reason}"))?;
                Ok(burn::DispatchDevice::Cuda(Default::default()))
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("dispatch cuda requested but the cuda feature is not enabled")
            }
        }
    }
}

#[cfg(feature = "dispatch")]
fn dispatch_auto_device() -> Result<burn::DispatchDevice> {
    let mut device = None;
    #[cfg(feature = "cuda")]
    {
        if cuda_runtime_preflight(CUDA_TRAIN_FORCE_ENV).is_ok() {
            device = Some(burn::DispatchDevice::Cuda(Default::default()));
        }
    }
    #[cfg(any(feature = "wgpu", feature = "webgpu"))]
    {
        if device.is_none() {
            device = Some(burn::DispatchDevice::Wgpu(Default::default()));
        }
    }
    #[cfg(feature = "flex")]
    {
        if device.is_none() {
            device = Some(burn::DispatchDevice::Flex(Default::default()));
        }
    }
    #[cfg(feature = "ndarray")]
    {
        if device.is_none() {
            device = Some(burn::DispatchDevice::NdArray(Default::default()));
        }
    }
    device.ok_or_else(|| {
        anyhow::anyhow!(
            "dispatch backend requested but no concrete dispatch backend feature is enabled"
        )
    })
}

fn run_ttt<B: crate::TttSparsePatchifyTrainingBackend>(
    config: &BurnJepaTrainConfig,
) -> Result<TttTrainingReport>
where
    B::Device: Default,
{
    let device = Default::default();
    train_ttt_distillation::<B>(config, &device)
}

fn run_ttt_eval<B: crate::TttSparsePatchifyBackend>(
    config: &BurnJepaTrainConfig,
    model: Option<PathBuf>,
    base_sparse: bool,
    steps: usize,
) -> Result<TttEvalReport>
where
    B::Device: Default,
{
    let device = Default::default();
    if base_sparse {
        evaluate_ttt_base_sparse::<B>(config, &device, steps)
    } else {
        evaluate_ttt_model_file::<B>(config, model.expect("model checked above"), &device, steps)
    }
}

#[cfg(feature = "sparse-patchify-wgpu")]
fn wants_frozen_sparse_patchify_backend(config: &BurnJepaTrainConfig) -> bool {
    match config.training.sparse_patchify_training {
        crate::TttSparsePatchifyTrainingMode::FrozenSparsePatchify => true,
        crate::TttSparsePatchifyTrainingMode::Auto => {
            config.ttt.freeze_pretrained
                && config
                    .training
                    .use_sparse_rollout(config.loss.predictor_loss_weight)
        }
        crate::TttSparsePatchifyTrainingMode::DensePatchEmbed => false,
    }
}

fn run_dense<B: AutodiffBackend>(config: &BurnJepaTrainConfig) -> Result<DenseJepaTrainingReport>
where
    B::Device: Default,
{
    let device = Default::default();
    train_dense_jepa::<B>(config, &device)
}

fn print_json<T: Serialize>(value: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}
