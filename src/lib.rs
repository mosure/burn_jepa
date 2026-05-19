#![allow(clippy::too_many_arguments, clippy::type_complexity)]

#[cfg(feature = "autogaze")]
mod autogaze;
#[cfg(not(target_arch = "wasm32"))]
pub mod cli;
mod config;
pub mod dataset;
pub mod experiment;
mod feature_memory;
mod highres_pipeline;
mod model;
mod model_package;
mod nodes;
mod pca;
mod pipeline;
mod positional;
mod quantization;
#[cfg(not(target_arch = "wasm32"))]
mod reconstruction_training;
pub mod runtime;
mod safetensors_io;
#[cfg(any(
    feature = "sparse-feature-memory-wgpu",
    feature = "sparse-feature-memory-cuda"
))]
mod sparse_feature_memory;
mod sparse_patchify;
mod temporal;
mod tokens;
pub mod training;
mod ttt;
pub mod viewer;
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
pub use burn_anyup::{
    AnyUp, AnyUpAttentionMode, AnyUpConfig, AnyUpImageContext, AnyUpImageGrid, AnyUpLoadOptions,
    AnyUpLoadReport,
};
pub use burn_jepa_reconstruction::{
    JepaReconstructionArchitecture, JepaReconstructionConfig, JepaReconstructionDecoder,
    JepaReconstructionFitReport, JepaReconstructionOutputActivation,
    JepaReconstructionPyramidStage, JepaReconstructionTokenBlock, JepaReconstructionTrainConfig,
    JepaReconstructionUpBlock, fit_reconstruction_decoder, reconstruction_color_moment_loss,
    reconstruction_gradient_mse, reconstruction_l1, reconstruction_mse, reconstruction_psnr,
    reconstruction_psnr_scalar,
};
pub use burn_store::ApplyResult as BurnStoreApplyResult;
pub use config::{
    VJepaConfig, VJepaEncoderConfig, VJepaModelVariant, VJepaPredictorConfig, VJepaPreprocessConfig,
};
pub use dataset::{
    JepaDataset, JepaDatasetConfig, JepaDatasetKind, JepaDatasetRepeatMode, JepaManifestRow,
    JepaSample, JepaSampleKind, JepaSampleMetadata, JepaTensorBatch, ManifestJepaDataset,
    SyntheticJepaDataset, dataset_from_config, load_jepa_tensor_batch, synthetic_video,
};
pub use experiment::{
    ExperimentConfig, ExperimentDataConfig, ExperimentDataReport, ExperimentMaskPolicy,
    ExperimentModelVariant, ExperimentPlanReport, ExperimentRunReport, ExperimentSuccessCriteria,
    ExperimentTrial, ExperimentTrialReport, ExperimentTrialStatus, ExperimentTrialTiming,
    ExperimentTttLayerSet, analyze_experiment, prepare_experiment_data, run_experiment,
    write_experiment_plan,
};
pub use feature_memory::{
    InterframeJepaFeatureAgeMode, InterframeJepaFeatureMemory, InterframeJepaFeatureMemoryConfig,
    InterframeJepaFeatureMemoryOutput, InterframeJepaFeatureUpdateMode,
    jepa_feature_tokens_to_nchw,
};
pub use highres_pipeline::{
    FeatureFrameBatch, FeatureFrameEncodePath, FeatureFrameInput, FeatureFrameJepaEncoder,
    FeatureFrameJepaEncoderKind, FeatureFrameMeasureConfig, FeatureFrameMetrics, FeatureFrameNode,
    FeatureFramePipeline, FeatureFramePipelineConfig, FeatureFrameRequest, FeatureFrameSchedule,
    FeatureFrameStream, FeatureFrameStreamOutput, FeatureFrameStreamStats,
    FeatureTokenStabilityMetrics, FrameId, FrameQueuePolicy, FrameQueueReport, FrameQueueTiming,
    FrameStreamConfig, HighResFrameArtifacts, LowResFrameArtifacts, MeasuredFeatureFrameBatch,
    SparseJepaAnyUpPcaBackpressurePolicy, SparseJepaAnyUpPcaEncodePath, SparseJepaAnyUpPcaFrameId,
    SparseJepaAnyUpPcaFrameInput, SparseJepaAnyUpPcaMeasuredBatchOutput,
    SparseJepaAnyUpPcaMeasuredOutput, SparseJepaAnyUpPcaMeasurementConfig,
    SparseJepaAnyUpPcaOutput, SparseJepaAnyUpPcaPipeline, SparseJepaAnyUpPcaPipelineConfig,
    SparseJepaAnyUpPcaQueueReport, SparseJepaAnyUpPcaQueuedFrameTiming,
    SparseJepaAnyUpPcaStageMetrics, SparseJepaAnyUpPcaStepBatchOutput, SparseJepaAnyUpPcaStream,
    SparseJepaAnyUpPcaStreamBatchOutput, SparseJepaAnyUpPcaStreamConfig,
    SparseJepaAnyUpPcaStreamStats, TttRuntimeCollapseGuardAction, TttRuntimeStateConfig,
    TttRuntimeStateMetrics, measure_feature_token_stability,
};
pub use model::{
    DensePredictionOutput, PatchEmbed2d, PatchEmbed3d, SparseEncoderBatchPlan, SparseEncoderPlan,
    SparsePredictorPlan, SparseVJepaForwardOutput, TokenSequencePosition, TransformerBlock,
    VJepa2_1Model, VJepaEncoder, VJepaEncoderOutput, VJepaMlp, VJepaPredictor,
    VJepaPredictorOutput, VJepaSelfAttention,
};
pub use model_package::{
    BurnAnyUpModelBootstrapConfig, BurnAnyUpModelDeployBundleReport, BurnAnyUpModelPackageFiles,
    BurnAnyUpModelProfile, BurnAnyUpPackageManifest, BurnJepaModelBootstrapConfig,
    BurnJepaModelDeployBundleReport, BurnJepaModelPackageFiles, BurnJepaModelProfile,
    BurnJepaPackageModelKind, BurnJepaPipelinePackageManifest,
    BurnJepaReconstructionModelBootstrapConfig, BurnJepaReconstructionModelDeployBundleReport,
    BurnJepaReconstructionModelPackageFiles, BurnJepaReconstructionModelProfile,
    BurnJepaReconstructionPackageManifest, BurnpackPartEntry, BurnpackPartsManifest,
    BurnpackPartsReport, DEFAULT_BURN_ANYUP_CHECKPOINT_PATH, DEFAULT_BURN_ANYUP_MODEL_BASE_URL,
    DEFAULT_BURN_ANYUP_MODEL_CACHE_SUBDIR, DEFAULT_BURN_ANYUP_MODEL_ROOT_URL,
    DEFAULT_BURN_JEPA_MODEL_BASE_URL, DEFAULT_BURN_JEPA_MODEL_CACHE_ROOT_DIR,
    DEFAULT_BURN_JEPA_MODEL_CACHE_SUBDIR, DEFAULT_BURN_JEPA_MODEL_ROOT_URL,
    DEFAULT_BURN_JEPA_RECONSTRUCTION_MODEL_BASE_URL,
    DEFAULT_BURN_JEPA_RECONSTRUCTION_MODEL_CACHE_SUBDIR,
    DEFAULT_BURN_JEPA_RECONSTRUCTION_MODEL_ROOT_URL, DEFAULT_BURNPACK_SHARD_MAX_BYTES,
    apply_burnpack_parts, burn_anyup_model_profile_base_url, burn_jepa_model_profile_base_url,
    burn_jepa_reconstruction_model_profile_base_url, burnpack_dtype_counts,
    burnpack_parts_dtype_counts, burnpack_parts_manifest_path, load_anyup_burnpack_parts,
    load_jepa_reconstruction_burnpack_parts, load_ttt_burnpack, load_ttt_burnpack_parts,
    load_vjepa_burnpack, load_vjepa_burnpack_parts, manifest_has_all_parts, module_dtype_counts,
    read_parts_manifest, resolve_package_manifest_entry_path, resolve_part_entry_path,
    save_anyup_burnpack, save_jepa_reconstruction_burnpack, save_module_burnpack,
    save_ttt_burnpack, save_vjepa_burnpack, write_anyup_package_manifest,
    write_burn_anyup_model_deploy_bundle, write_burn_jepa_model_deploy_bundle,
    write_burn_jepa_reconstruction_model_deploy_bundle, write_burnpack_parts_for_browser,
    write_jepa_reconstruction_package_manifest, write_pipeline_package_manifest,
};
#[cfg(not(target_arch = "wasm32"))]
pub use model_package::{
    burn_anyup_model_package_cache_complete, burn_jepa_model_package_cache_complete,
    burn_jepa_reconstruction_model_package_cache_complete, default_burn_anyup_model_cache_root,
    default_burn_anyup_model_cache_root_with_config, default_burn_jepa_model_cache_root,
    default_burn_jepa_model_cache_root_with_config,
    default_burn_jepa_reconstruction_model_cache_root,
    default_burn_jepa_reconstruction_model_cache_root_with_config,
    resolve_or_bootstrap_burn_anyup_model_package,
    resolve_or_bootstrap_burn_anyup_model_package_with_config,
    resolve_or_bootstrap_burn_anyup_model_package_with_config_and_progress,
    resolve_or_bootstrap_burn_jepa_model_package,
    resolve_or_bootstrap_burn_jepa_model_package_with_config,
    resolve_or_bootstrap_burn_jepa_model_package_with_config_and_progress,
    resolve_or_bootstrap_burn_jepa_reconstruction_model_package,
    resolve_or_bootstrap_burn_jepa_reconstruction_model_package_with_config,
    resolve_or_bootstrap_burn_jepa_reconstruction_model_package_with_config_and_progress,
};
pub use nodes::{
    FnOutputNode, RgbaVideoInput, SparseJepaAutogazeSparsityConfig, SparseJepaInputNode,
    SparseJepaOutputNode, SparseJepaPacket, SparseJepaPatchDiffSparsityConfig,
    SparseJepaSparsityDriverConfig, SparseJepaTensorPipeline, SparseJepaTensorPipelineConfig,
    TensorVideoInput, VecOutputNode, empty_rgb_video_shape, patch_diff_context_mask_from_scores,
    patch_diff_context_mask_from_video, patch_diff_token_scores, resolve_sparsity_driver_masks,
};
pub use pca::{
    FeaturePcaConfig, FeaturePcaDisplayMode, FeaturePcaProjector, FeaturePcaUpdateConfig,
    FeaturePcaUpdateDecision, FeaturePcaUpdateMode, FeaturePcaUpdateScheduler,
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
    BurnJepaTrainConfig, DenseJepaTrainingReport, DensePredictiveLoss, JepaDispatchBackend,
    JepaTrainBackend, LearningRateScheduleConfig, LearningRateScheduleStats, TrainModelConfig,
    TrainingAutogazeTokenSource, TrainingBatchingMode, TrainingImageTokenGrid, TrainingLoopConfig,
    TrainingMaskConfig, TttBackpropMetrics, TttBestCheckpointSelection, TttDenseSampleMetrics,
    TttDenseSampleTrainingConfig, TttDistillationConfig, TttDistillationLoss, TttDomainEvalMetric,
    TttEvalModelKind, TttEvalReport, TttLatentRegularizationConfig, TttLatentRegularizationMetrics,
    TttLayerUtilizationMetric, TttLongRolloutMetrics, TttLongRolloutSegmentMetric,
    TttLongRolloutStreamMetric, TttRolloutMetrics, TttRolloutReportMode,
    TttSequenceCurriculumConfig, TttSparsePatchifyBackend, TttSparsePatchifyTrainingBackend,
    TttSparsePatchifyTrainingMode, TttSparseRolloutMode, TttStreamStepKind,
    TttStreamTrainingConfig, TttStreamTrainingMetrics, TttTargetSupervisionMetrics,
    TttTemporalDiagnosticMetrics, TttTemporalSegmentMetric, TttTemporalSegmentMetrics,
    TttTrainingReport, TttUtilizationMetrics, VJepaTrainingBatch, center_prior_frame_tokens,
    dense_predictive_loss, evaluate_ttt_base_sparse, evaluate_ttt_distillation,
    evaluate_ttt_model_file, train_dense_jepa, train_ttt_distillation,
};
pub use ttt::{
    TttBackpropMode, TttEncoderConfig, TttInsertionMode, TttLayerPlacement, TttLayerState,
    TttMemoryDynamics, TttMemoryUpdateSource, TttState, TttStateResetMode, TttSupervisionMode,
    TttTargetMode, VJepaInPlaceTttMlp, VJepaTttEncoder, VJepaTttLayer, VJepaTttLayerProbe,
    VJepaTttLayerProbeRecord, VJepaTttModel,
};
pub use viewer::{
    DEFAULT_ANYUP_CHUNK_SIZE, DEFAULT_BOOTSTRAP_CONTEXT_DENSITY, DEFAULT_CONTEXT_DENSITY,
    DEFAULT_HIGH_RES_PCA_EVERY, DEFAULT_IMAGE_SIZE, DEFAULT_MIN_CONTEXT_DENSITY,
    DEFAULT_PATCH_DIFF_AGE_REFRESH_INTERVAL_FRAMES, DEFAULT_PATCH_DIFF_AGE_REFRESH_MAX_DENSITY,
    DEFAULT_PATCH_DIFF_BLUE_NOISE_REFRESH_DENSITY, DEFAULT_PATCH_DIFF_DENSE_FALLBACK_DENSITY,
    DEFAULT_PATCH_DIFF_DILATION_TILES, DEFAULT_PATCH_DIFF_QUALITY,
    DEFAULT_PATCH_DIFF_REFRESH_ENABLED, DEFAULT_PATCH_DIFF_REFRESH_MAX_DENSITY,
    DEFAULT_PATCH_DIFF_SUBTHRESHOLD_DECAY, DEFAULT_PATCH_DIFF_SUBTHRESHOLD_MAX_DENSITY,
    DEFAULT_PATCH_DIFF_SUBTHRESHOLD_TRIGGER, DEFAULT_PATCH_DIFF_THRESHOLD,
    DEFAULT_PCA_MIN_SAMPLE_FRAMES, DEFAULT_PCA_SAMPLE_WINDOW_FRAMES, DEFAULT_PCA_UPDATE_EVERY,
    DEFAULT_PCA_UPDATE_ITERATIONS, DEFAULT_PREWARM_SHAPE_BUCKETS,
    DEFAULT_SPARSE_MASK_BUCKET_DENSITIES, DEFAULT_SPARSE_MASK_BUCKET_TOKENS,
    FeatureFrameEncodeRoute, FeatureFrameSparseEncodeMode, FeatureFrameSparseMasks,
    FeatureFrameViewerConfig, MIN_PIPELINE_IMAGE_SIZE, PIPELINE_IMAGE_SIZE_MULTIPLE,
    PatchDiffRefreshConfig, PatchDiffRefreshState, RgbaPatchDiffFrameStats, bucket_sparse_mask,
    bucket_sparse_mask_with_config, center_prior_mask, finalize_patch_diff_mask,
    finalize_patch_diff_masks, patch_diff_can_use_dense_fast_path, patch_diff_dense_fallback,
    patch_diff_sampled_dense_fast_path_from_rgba, patch_diff_scores_from_rgba,
    patch_diff_sparsity_config, patch_diff_threshold_from_quality, rgba_patch_diff_frame_stats,
    shape_prewarm_masks, sparse_mask_bucket_widths,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use wasm::*;

#[cfg(feature = "ndarray")]
pub type NdArrayVJepaModel = VJepa2_1Model<burn::backend::NdArray<f32>>;

#[cfg(feature = "ndarray")]
pub type NdArrayVJepaPipeline = VJepaPipeline<burn::backend::NdArray<f32>>;

#[cfg(feature = "flex")]
pub type FlexVJepaModel = VJepa2_1Model<burn::backend::Flex<f32, i32>>;

#[cfg(feature = "flex")]
pub type FlexVJepaPipeline = VJepaPipeline<burn::backend::Flex<f32, i32>>;

#[cfg(feature = "dispatch")]
pub type DispatchVJepaModel = VJepa2_1Model<burn::Dispatch>;

#[cfg(feature = "dispatch")]
pub type DispatchVJepaPipeline = VJepaPipeline<burn::Dispatch>;

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
