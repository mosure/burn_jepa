use super::batch::load_training_batch;
use super::config::BurnJepaTrainConfig;
use super::model_io::load_student_model;
use super::report::{
    DenseJepaTrainingReport, TrainingLossSummary, samples_per_second, save_training_report,
    tensor_scalar,
};
use crate::{SparseTokenMask, VJepa2_1Model, dataset_from_config};
use anyhow::{Context, Result};
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};
use std::fs;
use std::time::Instant;

#[derive(Debug)]
pub struct DensePredictiveLoss<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub predictions: Tensor<B, 3>,
    pub targets: Tensor<B, 3>,
}

#[derive(Debug)]
pub struct VJepaTrainingBatch<B: Backend> {
    pub video: Tensor<B, 5>,
    pub context_mask: SparseTokenMask,
    pub target_mask: SparseTokenMask,
}

pub fn dense_predictive_loss<B: Backend>(
    predictions: Tensor<B, 3>,
    targets: Tensor<B, 3>,
) -> DensePredictiveLoss<B> {
    let loss = (predictions.clone() - targets.clone())
        .powf_scalar(2.0)
        .mean();
    DensePredictiveLoss {
        loss,
        predictions,
        targets,
    }
}

impl<B: AutodiffBackend> VJepa2_1Model<B> {
    pub fn training_loss(&self, batch: VJepaTrainingBatch<B>) -> Result<DensePredictiveLoss<B>> {
        let dense =
            self.predict_dense_targets(batch.video, &batch.context_mask, &batch.target_mask)?;
        Ok(dense_predictive_loss(dense.predictions, dense.targets))
    }
}

pub fn train_dense_jepa<B: AutodiffBackend>(
    config: &BurnJepaTrainConfig,
    device: &B::Device,
) -> Result<DenseJepaTrainingReport> {
    config.validate_common()?;
    let start = Instant::now();
    fs::create_dir_all(&config.model.output_dir)
        .with_context(|| format!("create {}", config.model.output_dir.display()))?;
    let mut model = load_student_model::<B>(config, device)?;
    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.training.weight_decay)
        .init::<B, VJepa2_1Model<B>>();
    let dataset = dataset_from_config(&config.dataset, true)?;
    let mut final_loss = 0.0;
    for step in 0..config.training.max_steps {
        let batch = load_training_batch::<B>(
            dataset.as_ref(),
            &config.dataset,
            model.config(),
            device,
            step * config.training.batch_size,
            config.training.batch_size,
        )?;
        let (context_mask, target_mask) = config.training.resolve_masks_with_metadata(
            &batch.student,
            model.config(),
            &batch.metadata,
        )?;
        let loss = model.training_loss(VJepaTrainingBatch {
            video: batch.student,
            context_mask,
            target_mask,
        })?;
        final_loss = tensor_scalar(loss.loss.clone().detach())?;
        let grads = GradientsParams::from_grads(loss.loss.backward(), &model);
        model = optim.step(config.training.learning_rate_for_step(step), model, grads);
    }
    let elapsed_ms = start.elapsed().as_millis();
    let model_path = if config.model.save_model {
        let path = config.model.output_dir.join("dense-jepa-model");
        model
            .clone()
            .save_file(
                path.clone(),
                &NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
            )
            .context("save dense JEPA model")?;
        Some(path.with_extension("mpk"))
    } else {
        None
    };
    let samples = config.training.max_steps * config.training.batch_size;
    let report_path = save_training_report(
        &config.model.output_dir,
        "dense-jepa-report.json",
        config.training.max_steps,
        samples,
        TrainingLossSummary::dense(final_loss),
        elapsed_ms,
        model_path.clone(),
    )?;
    Ok(DenseJepaTrainingReport {
        steps: config.training.max_steps,
        samples,
        final_loss,
        elapsed_ms,
        samples_per_second: samples_per_second(samples, elapsed_ms),
        model_path,
        report_path,
    })
}
