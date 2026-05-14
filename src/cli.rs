#[cfg(feature = "cuda")]
use crate::runtime::{CUDA_TRAIN_FORCE_ENV, cuda_runtime_preflight};
use crate::{
    BurnJepaTrainConfig, DenseJepaTrainingReport, ExperimentConfig, ExperimentRunReport,
    JepaTrainBackend, TttEvalReport, TttTrainingReport, analyze_experiment,
    evaluate_ttt_model_file, prepare_experiment_data, run_experiment, train_dense_jepa,
    train_ttt_distillation, write_experiment_plan,
};
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
        #[arg(short, long)]
        model: PathBuf,
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
            let report = dispatch_ttt_eval(&config, model, steps)?;
            print_json(&report)
        }
        BurnJepaCommand::TrainJepa { config } => {
            let config = BurnJepaTrainConfig::from_toml_file(config)?;
            let report = dispatch_dense(&config)?;
            print_json(&report)
        }
        BurnJepaCommand::BenchTtt { config, steps } => {
            let mut config = BurnJepaTrainConfig::from_toml_file(config)?;
            if let Some(steps) = steps {
                config.training.max_steps = steps;
            }
            config.model.save_model = false;
            let report = dispatch_ttt(&config)?;
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
    }
}

fn dispatch_ttt_eval(
    config: &BurnJepaTrainConfig,
    model: PathBuf,
    steps: usize,
) -> Result<TttEvalReport> {
    match config.training.backend {
        JepaTrainBackend::NdArray => {
            #[cfg(feature = "ndarray")]
            {
                run_ttt_eval::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>(
                    config, model, steps,
                )
            }
            #[cfg(not(feature = "ndarray"))]
            {
                bail!("ndarray backend requested but the ndarray feature is not enabled")
            }
        }
        JepaTrainBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                cuda_runtime_preflight(CUDA_TRAIN_FORCE_ENV)
                    .map_err(|reason| anyhow::anyhow!("cuda backend unavailable: {reason}"))?;
                run_ttt_eval::<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>(
                    config, model, steps,
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
                #[cfg(feature = "sparse-patchify-wgpu")]
                if wants_frozen_sparse_patchify_backend(config) {
                    return run_ttt_eval::<
                        burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
                    >(config, model, steps);
                }
                run_ttt_eval::<burn::backend::Autodiff<burn::backend::Wgpu<f32, i32>>>(
                    config, model, steps,
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
                run_ttt_eval::<burn::backend::Autodiff<burn::backend::WebGpu<f32, i32>>>(
                    config, model, steps,
                )
            }
            #[cfg(not(feature = "webgpu"))]
            {
                bail!("webgpu backend requested but the webgpu feature is not enabled")
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
    }
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

fn run_ttt_eval<B: crate::TttSparsePatchifyTrainingBackend>(
    config: &BurnJepaTrainConfig,
    model: PathBuf,
    steps: usize,
) -> Result<TttEvalReport>
where
    B::Device: Default,
{
    let device = Default::default();
    evaluate_ttt_model_file::<B>(config, model, &device, steps)
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
