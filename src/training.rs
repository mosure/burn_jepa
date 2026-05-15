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
    TttDistillationConfig, TttSequenceCurriculumConfig, TttSparsePatchifyTrainingMode,
    TttSparseRolloutMode, TttStreamTrainingConfig,
};
pub use dense::{DensePredictiveLoss, VJepaTrainingBatch, dense_predictive_loss, train_dense_jepa};
pub use mask::{
    TrainingAutogazeTokenSource, TrainingImageTokenGrid, TrainingMaskConfig,
    center_prior_frame_tokens,
};
pub use report::{
    DenseJepaTrainingReport, TttBackpropMetrics, TttDomainEvalMetric, TttEvalReport,
    TttLayerUtilizationMetric, TttRolloutMetrics, TttRolloutReportMode, TttStreamStepKind,
    TttStreamTrainingMetrics, TttTargetSupervisionMetrics, TttTemporalDiagnosticMetrics,
    TttTemporalSegmentMetric, TttTemporalSegmentMetrics, TttTrainingReport, TttUtilizationMetrics,
};
pub use ttt::{
    TttDistillationLoss, TttSparsePatchifyTrainingBackend, evaluate_ttt_distillation,
    evaluate_ttt_model_file, train_ttt_distillation,
};
