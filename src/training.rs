mod batch;
mod config;
mod dense;
mod mask;
mod model_io;
mod report;
mod ttt;

pub use config::{
    BurnJepaTrainConfig, JepaDispatchBackend, JepaTrainBackend, LearningRateScheduleConfig,
    LearningRateScheduleStats, TrainModelConfig, TrainingBatchingMode, TrainingLoopConfig,
    TttDenseSampleTrainingConfig, TttDistillationConfig, TttLatentRegularizationConfig,
    TttSequenceCurriculumConfig, TttSparsePatchifyTrainingMode, TttSparseRolloutMode,
    TttStreamTrainingConfig,
};
pub use dense::{DensePredictiveLoss, VJepaTrainingBatch, dense_predictive_loss, train_dense_jepa};
pub use mask::{
    TrainingAutogazeTokenSource, TrainingImageTokenGrid, TrainingMaskConfig,
    center_prior_frame_tokens,
};
pub use report::{
    DenseJepaTrainingReport, TttBackpropMetrics, TttDenseSampleMetrics, TttDomainEvalMetric,
    TttEvalModelKind, TttEvalReport, TttLatentRegularizationMetrics, TttLayerUtilizationMetric,
    TttLongRolloutMetrics, TttLongRolloutSegmentMetric, TttLongRolloutStreamMetric,
    TttRolloutMetrics, TttRolloutReportMode, TttStreamStepKind, TttStreamTrainingMetrics,
    TttTargetSupervisionMetrics, TttTemporalDiagnosticMetrics, TttTemporalSegmentMetric,
    TttTemporalSegmentMetrics, TttTrainingReport, TttUtilizationMetrics,
};
pub use ttt::{
    TttDistillationLoss, TttSparsePatchifyBackend, TttSparsePatchifyTrainingBackend,
    evaluate_ttt_base_sparse, evaluate_ttt_distillation, evaluate_ttt_model_file,
    train_ttt_distillation,
};
