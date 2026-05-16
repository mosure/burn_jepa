mod attention;
mod config;
mod layers;
mod loading;
mod model;
mod rope;
mod sparse;
mod tensor_ops;

pub use attention::{EfficientCrossAttention, EfficientCrossAttentionBlock};
pub use config::AnyUpConfig;
pub use layers::{AnyUpConvEncoder, AnyUpFeatureEncoder, AnyUpResBlock, LearnedFeatureUnification};
pub use loading::{AnyUpLoadOptions, AnyUpLoadReport};
pub use model::{AnyUp, AnyUpImageContext, AnyUpImageGrid};
pub use rope::{AnyUpRoPE, rotate_half};
pub use sparse::{
    AnyUpHighResFeatureMemory, AnyUpHighResFeatureMemoryConfig, AnyUpHighResFeatureMemoryOutput,
    AnyUpSparseFeatureAgeMode, AnyUpSparseFeatureMemoryWriteMode, AnyUpSparseFeatureUpdateMode,
    AnyUpSparseOutput, AnyUpSparseOutputPlan, sparse_low_features_to_nchw,
};

#[cfg(feature = "ndarray")]
pub type NdArrayAnyUp = AnyUp<burn::backend::NdArray<f32>>;

#[cfg(feature = "flex")]
pub type FlexAnyUp = AnyUp<burn::backend::Flex<f32, i32>>;

#[cfg(feature = "dispatch")]
pub type DispatchAnyUp = AnyUp<burn::Dispatch>;

#[cfg(feature = "cuda")]
pub type CudaAnyUp = AnyUp<burn::backend::Cuda<f32, i32>>;

#[cfg(feature = "cuda")]
pub type CudaF16AnyUp = AnyUp<burn::backend::Cuda<burn::tensor::f16, i32>>;

#[cfg(feature = "cuda")]
pub type CudaBf16AnyUp = AnyUp<burn::backend::Cuda<burn::tensor::bf16, i32>>;

#[cfg(any(feature = "wgpu", feature = "webgpu"))]
pub type WgpuAnyUp = AnyUp<burn::backend::Wgpu<f32, i32>>;

#[cfg(any(feature = "wgpu", feature = "webgpu"))]
pub type WgpuF16AnyUp = AnyUp<burn::backend::Wgpu<burn::tensor::f16, i32>>;

#[cfg(any(feature = "wgpu", feature = "webgpu"))]
pub type WgpuBf16AnyUp = AnyUp<burn::backend::Wgpu<burn::tensor::bf16, i32>>;

#[cfg(feature = "webgpu")]
pub type WebGpuAnyUp = AnyUp<burn::backend::WebGpu<f32, i32>>;

#[cfg(feature = "webgpu")]
pub type WebGpuF16AnyUp = AnyUp<burn::backend::WebGpu<burn::tensor::f16, i32>>;

#[cfg(feature = "webgpu")]
pub type WebGpuBf16AnyUp = AnyUp<burn::backend::WebGpu<burn::tensor::bf16, i32>>;
