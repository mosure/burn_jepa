mod batch;
mod config;
mod dense;
mod mask;
mod model_io;
mod report;
mod ttt;

pub use config::{
    BurnJepaTrainConfig, JepaTrainBackend, TrainModelConfig, TrainingBatchingMode,
    TrainingLoopConfig, TttDistillationConfig, TttSparsePatchifyTrainingMode, TttSparseRolloutMode,
};
pub use dense::{DensePredictiveLoss, VJepaTrainingBatch, dense_predictive_loss, train_dense_jepa};
pub use mask::{
    TrainingAutogazeTokenSource, TrainingImageTokenGrid, TrainingMaskConfig,
    center_prior_frame_tokens,
};
pub use report::{
    DenseJepaTrainingReport, TttBackpropMetrics, TttDomainEvalMetric, TttEvalReport,
    TttRolloutMetrics, TttRolloutReportMode, TttTrainingReport,
};
pub use ttt::{
    TttDistillationLoss, TttSparsePatchifyTrainingBackend, evaluate_ttt_distillation,
    evaluate_ttt_model_file, train_ttt_distillation,
};
