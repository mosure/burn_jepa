mod config;
mod model;
mod nodes;
mod pipeline;
mod positional;
mod quantization;
mod safetensors_io;
mod sparse_patchify;
mod temporal;
mod tokens;
mod training;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod wasm;

pub use config::{
    VJepaConfig, VJepaEncoderConfig, VJepaModelVariant, VJepaPredictorConfig, VJepaPreprocessConfig,
};
pub use model::{
    DensePredictionOutput, PatchEmbed2d, PatchEmbed3d, SparsePredictorPlan,
    SparseVJepaForwardOutput, TokenSequencePosition, TransformerBlock, VJepa2_1Model, VJepaEncoder,
    VJepaEncoderOutput, VJepaMlp, VJepaPredictor, VJepaPredictorOutput, VJepaSelfAttention,
};
pub use nodes::{
    FnOutputNode, RgbaVideoInput, SparseJepaInputNode, SparseJepaOutputNode, SparseJepaPacket,
    SparseJepaTensorPipeline, SparseJepaTensorPipelineConfig, TensorVideoInput, VecOutputNode,
    empty_rgb_video_shape,
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
    SparseImageTokenGrid, SparsePatchRect, SparsePatchifyPlan, sparse_mask_from_frame_rects,
    sparse_mask_from_frame_token_indices, video_token_grid,
};
pub use temporal::{
    TemporalSparseJepaConfig, TemporalSparseJepaOutput, TemporalSparseJepaState,
    TemporalSparseMaskConfig, TemporalSparseMaskOutput, TemporalSparseMaskState,
};
pub use tokens::{
    SparseTokenMask, SparseVideoTokens, TokenGridShape, apply_token_mask, complement_indices,
    dense_token_indices, make_context_target_masks, repeat_token_indices,
};
pub use training::{DensePredictiveLoss, VJepaTrainingBatch, dense_predictive_loss};
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

#[cfg(feature = "webgpu")]
pub type WebGpuVJepaModel = VJepa2_1Model<burn::backend::WebGpu<f32, i32>>;

#[cfg(feature = "webgpu")]
pub type WebGpuVJepaPipeline = VJepaPipeline<burn::backend::WebGpu<f32, i32>>;
