#[cfg(feature = "autogaze")]
mod autogaze;
#[cfg(not(target_arch = "wasm32"))]
pub mod cli;
mod config;
pub mod dataset;
pub mod experiment;
mod model;
mod nodes;
mod pipeline;
mod positional;
mod quantization;
pub mod runtime;
mod safetensors_io;
mod sparse_patchify;
mod temporal;
mod tokens;
pub mod training;
mod ttt;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod wasm;

#[cfg(feature = "autogaze")]
pub use autogaze::{
    AutogazeFrameTokenPairs, AutogazeSparseJepaMasks, AutogazeSparseJepaProjection,
    AutogazeSparseJepaProjectionConfig, AutogazeSparseJepaWindowConfig,
    AutogazeSparseJepaWindowPlan, autogaze_frame_token_pairs, autogaze_frame_tokens,
    autogaze_image_token_grid, autogaze_sparse_context_tokens, autogaze_sparse_generation_budget,
    autogaze_sparse_top_k_for_context, autogaze_sparse_top_k_for_context_with_overfetch,
    generate_autogaze_streaming_with_budget, project_autogaze_generated_masks,
    project_autogaze_generated_tokens,
};
pub use config::{
    VJepaConfig, VJepaEncoderConfig, VJepaModelVariant, VJepaPredictorConfig, VJepaPreprocessConfig,
};
pub use dataset::{
    JepaDataset, JepaDatasetConfig, JepaDatasetKind, JepaManifestRow, JepaSample, JepaSampleKind,
    JepaSampleMetadata, JepaTensorBatch, ManifestJepaDataset, SyntheticJepaDataset,
    dataset_from_config, load_jepa_tensor_batch, synthetic_video,
};
pub use experiment::{
    ExperimentConfig, ExperimentDataConfig, ExperimentDataReport, ExperimentMaskPolicy,
    ExperimentModelVariant, ExperimentPlanReport, ExperimentRunReport, ExperimentSuccessCriteria,
    ExperimentTrial, ExperimentTrialReport, ExperimentTrialStatus, ExperimentTrialTiming,
    ExperimentTttLayerSet, analyze_experiment, prepare_experiment_data, run_experiment,
    write_experiment_plan,
};
pub use model::{
    DensePredictionOutput, PatchEmbed2d, PatchEmbed3d, SparseEncoderBatchPlan, SparseEncoderPlan,
    SparsePredictorPlan, SparseVJepaForwardOutput, TokenSequencePosition, TransformerBlock,
    VJepa2_1Model, VJepaEncoder, VJepaEncoderOutput, VJepaMlp, VJepaPredictor,
    VJepaPredictorOutput, VJepaSelfAttention,
};
pub use nodes::{
    FnOutputNode, RgbaVideoInput, SparseJepaAutogazeSparsityConfig, SparseJepaInputNode,
    SparseJepaOutputNode, SparseJepaPacket, SparseJepaPatchDiffSparsityConfig,
    SparseJepaSparsityDriverConfig, SparseJepaTensorPipeline, SparseJepaTensorPipelineConfig,
    TensorVideoInput, VecOutputNode, empty_rgb_video_shape, resolve_sparsity_driver_masks,
};
pub use pipeline::{
    VJEPA_IMAGE_MEAN, VJEPA_IMAGE_STD, VJEPA_RESCALE_FACTOR, VJepaEmbedOutput, VJepaPipeline,
    VJepaRgbaVideoShape, VJepaVideoShape, ensure_model_sized_video, rgba_video_to_tensor,
};
pub use positional::{
    SparsePosition, coords_to_token_index, get_1d_sincos_pos_embed, get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed, sparse_3d_sincos_pos_embed, token_index_to_coords,
};
pub use quantization::{
    QuantizationMode, QuantizedTensorData, symmetric_dequantize, symmetric_quantize,
};
pub use safetensors_io::{
    LoadReport, VJepaLoadOptions, checkpoint_tensor_prefixes, default_hf_snapshot_dir,
    load_config_from_hf_dir,
};
pub use sparse_patchify::{
    SparseImageTokenGrid, SparsePatchRect, SparsePatchifyBatchPlan, SparsePatchifyPlan,
    sparse_mask_from_frame_rects, sparse_mask_from_frame_token_indices,
    sparse_mask_from_frame_token_pairs, video_token_grid,
};
pub use temporal::{
    TemporalSparseJepaConfig, TemporalSparseJepaOutput, TemporalSparseJepaState,
    TemporalSparseJepaStream, TemporalSparseJepaStreamConfig, TemporalSparseJepaStreamOutput,
    TemporalSparseMaskConfig, TemporalSparseMaskOutput, TemporalSparseMaskState,
    TemporalSparsePredictorInput,
};
pub use tokens::{
    SparseMaskBatch, SparseTokenMask, SparseVideoTokens, TokenGridShape, apply_mask_batch,
    apply_token_mask, complement_indices, dense_token_indices, make_context_target_masks,
    repeat_token_indices, target_mask_from_context,
};
pub use training::{
    BurnJepaTrainConfig, DenseJepaTrainingReport, DensePredictiveLoss, JepaTrainBackend,
    TrainModelConfig, TrainingAutogazeTokenSource, TrainingBatchingMode, TrainingImageTokenGrid,
    TrainingLoopConfig, TrainingMaskConfig, TttBackpropMetrics, TttDistillationConfig,
    TttDistillationLoss, TttDomainEvalMetric, TttEvalReport, TttLayerUtilizationMetric,
    TttRolloutMetrics, TttRolloutReportMode, TttSparsePatchifyTrainingBackend,
    TttSparsePatchifyTrainingMode, TttSparseRolloutMode, TttTargetSupervisionMetrics,
    TttTemporalDiagnosticMetrics, TttTrainingReport, TttUtilizationMetrics, VJepaTrainingBatch,
    center_prior_frame_tokens, dense_predictive_loss, evaluate_ttt_distillation,
    evaluate_ttt_model_file, train_dense_jepa, train_ttt_distillation,
};
pub use ttt::{
    TttBackpropMode, TttEncoderConfig, TttLayerPlacement, TttLayerState, TttState,
    TttStateResetMode, TttTargetMode, VJepaTttEncoder, VJepaTttLayer, VJepaTttLayerProbe,
    VJepaTttLayerProbeRecord, VJepaTttModel,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use wasm::*;

#[cfg(feature = "ndarray")]
pub type NdArrayVJepaModel = VJepa2_1Model<burn::backend::NdArray<f32>>;

#[cfg(feature = "ndarray")]
pub type NdArrayVJepaPipeline = VJepaPipeline<burn::backend::NdArray<f32>>;

#[cfg(feature = "cuda")]
pub type CudaVJepaModel = VJepa2_1Model<burn::backend::Cuda<f32, i32>>;

#[cfg(feature = "cuda")]
pub type CudaVJepaPipeline = VJepaPipeline<burn::backend::Cuda<f32, i32>>;

#[cfg(any(feature = "wgpu", feature = "webgpu"))]
pub type WgpuVJepaModel = VJepa2_1Model<burn::backend::Wgpu<f32, i32>>;

#[cfg(any(feature = "wgpu", feature = "webgpu"))]
pub type WgpuVJepaPipeline = VJepaPipeline<burn::backend::Wgpu<f32, i32>>;

#[cfg(feature = "sparse-patchify-wgpu")]
pub type SparsePatchifyWgpuVJepaModel = VJepa2_1Model<burn_flex_gmm::wgpu::DefaultWgpuBackend>;

#[cfg(feature = "sparse-patchify-wgpu")]
pub type SparsePatchifyWgpuVJepaPipeline = VJepaPipeline<burn_flex_gmm::wgpu::DefaultWgpuBackend>;

#[cfg(feature = "sparse-patchify-cuda")]
pub type SparsePatchifyCudaVJepaModel = VJepa2_1Model<burn_flex_gmm::cuda::DefaultCudaBackend>;

#[cfg(feature = "sparse-patchify-cuda")]
pub type SparsePatchifyCudaVJepaPipeline = VJepaPipeline<burn_flex_gmm::cuda::DefaultCudaBackend>;

#[cfg(feature = "webgpu")]
pub type WebGpuVJepaModel = VJepa2_1Model<burn::backend::WebGpu<f32, i32>>;

#[cfg(feature = "webgpu")]
pub type WebGpuVJepaPipeline = VJepaPipeline<burn::backend::WebGpu<f32, i32>>;
