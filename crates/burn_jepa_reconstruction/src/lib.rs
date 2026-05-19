mod config;
mod metrics;
mod model;
mod train;

pub use config::{JepaReconstructionConfig, JepaReconstructionOutputActivation};
pub use metrics::{reconstruction_mse, reconstruction_psnr, reconstruction_psnr_scalar};
pub use model::{JepaReconstructionDecoder, JepaReconstructionUpBlock};
pub use train::{
    JepaReconstructionFitReport, JepaReconstructionTrainConfig, fit_reconstruction_decoder,
};

#[cfg(feature = "ndarray")]
pub type NdArrayJepaReconstructionDecoder = JepaReconstructionDecoder<burn::backend::NdArray<f32>>;
