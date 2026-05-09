use crate::{SparseTokenMask, VJepa2_1Model};
use anyhow::Result;
use burn::tensor::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};

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
