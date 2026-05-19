use crate::{
    JepaReconstructionConfig, JepaReconstructionDecoder, reconstruction_color_moment_loss,
    reconstruction_gradient_mse, reconstruction_l1, reconstruction_mse,
};
use anyhow::Result;
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

#[derive(Clone, Debug)]
pub struct JepaReconstructionTrainConfig {
    pub decoder: JepaReconstructionConfig,
    pub steps: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub l1_loss_weight: f64,
    pub gradient_loss_weight: f64,
    pub color_loss_weight: f64,
    pub log_interval: usize,
}

impl Default for JepaReconstructionTrainConfig {
    fn default() -> Self {
        Self {
            decoder: JepaReconstructionConfig::default(),
            steps: 500,
            learning_rate: 1.0e-3,
            weight_decay: 1.0e-4,
            l1_loss_weight: 0.02,
            gradient_loss_weight: 0.05,
            color_loss_weight: 0.02,
            log_interval: 50,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct JepaReconstructionFitReport {
    pub initial_loss: Option<f64>,
    pub final_loss: Option<f64>,
    pub best_loss: Option<f64>,
    pub steps: usize,
}

impl JepaReconstructionFitReport {
    fn record(&mut self, step: usize, loss: f64) {
        if self.initial_loss.is_none() {
            self.initial_loss = Some(loss);
        }
        self.final_loss = Some(loss);
        self.best_loss = Some(self.best_loss.map_or(loss, |best| best.min(loss)));
        self.steps = step;
    }
}

pub fn fit_reconstruction_decoder<B: AutodiffBackend>(
    config: JepaReconstructionTrainConfig,
    features: Tensor<B, 4>,
    target: Tensor<B, 4>,
    device: &B::Device,
) -> Result<(
    JepaReconstructionDecoder<B::InnerBackend>,
    JepaReconstructionFitReport,
)> {
    let mut model = JepaReconstructionDecoder::<B>::new(config.decoder.clone(), device)?;
    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.weight_decay as f32)
        .init();
    let mut report = JepaReconstructionFitReport::default();
    for step in 0..config.steps {
        let target_dims = target.shape().dims::<4>();
        let output = model.forward_to_size(features.clone(), [target_dims[2], target_dims[3]]);
        let loss = reconstruction_training_loss(
            output,
            target.clone(),
            config.l1_loss_weight,
            config.gradient_loss_weight,
            config.color_loss_weight,
        );
        let should_read = step == 0
            || step + 1 == config.steps
            || (config.log_interval > 0 && (step + 1) % config.log_interval == 0);
        if should_read && let Some(loss_value) = tensor_scalar(loss.clone().detach()) {
            report.record(step + 1, loss_value);
        }
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(config.learning_rate, model, grads);
    }
    Ok((model.valid(), report))
}

fn reconstruction_training_loss<B: burn::tensor::backend::Backend>(
    output: Tensor<B, 4>,
    target: Tensor<B, 4>,
    l1_weight: f64,
    gradient_weight: f64,
    color_weight: f64,
) -> Tensor<B, 1> {
    let mut loss = reconstruction_mse(output.clone(), target.clone());
    if l1_weight > 0.0 {
        loss = loss + reconstruction_l1(output.clone(), target.clone()).mul_scalar(l1_weight);
    }
    if gradient_weight > 0.0 {
        loss = loss
            + reconstruction_gradient_mse(output.clone(), target.clone())
                .mul_scalar(gradient_weight);
    }
    if color_weight > 0.0 {
        loss = loss + reconstruction_color_moment_loss(output, target).mul_scalar(color_weight);
    }
    loss
}

fn tensor_scalar<B: burn::tensor::backend::Backend>(tensor: Tensor<B, 1>) -> Option<f64> {
    tensor
        .to_data()
        .to_vec::<f32>()
        .ok()
        .and_then(|values| values.first().copied())
        .map(f64::from)
}
