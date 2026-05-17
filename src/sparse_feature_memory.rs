use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

#[derive(Debug)]
pub(crate) struct SparseFeatureMemoryAssignOutput<B: Backend> {
    pub features: Tensor<B, 3>,
    pub observed: Tensor<B, 2>,
    pub age_frames: Tensor<B, 2>,
}

#[cfg(any(
    feature = "sparse-feature-memory-wgpu",
    feature = "sparse-feature-memory-cuda"
))]
mod cube {
    use burn::tensor::{DType, Int, Shape, Tensor as BurnTensor, TensorPrimitive};
    use burn_cubecl::cubecl;
    use burn_cubecl::cubecl::std::tensor::layout::linear::LinearView;
    use burn_cubecl::cubecl::{calculate_cube_count_elemwise, prelude::*};
    use burn_cubecl::tensor::CubeTensor;

    #[cube(launch_unchecked, address_type = "dynamic")]
    pub(super) fn copy_feature_memory_kernel(
        features: &LinearView<f32>,
        observed: &LinearView<f32>,
        age_frames: &LinearView<f32>,
        output_features: &mut LinearView<f32, ReadWrite>,
        output_observed: &mut LinearView<f32, ReadWrite>,
        output_age_frames: &mut LinearView<f32, ReadWrite>,
        feature_elements: usize,
        metadata_elements: usize,
        age_observed: u32,
    ) {
        let idx = ABSOLUTE_POS;
        if idx >= feature_elements + metadata_elements {
            terminate!();
        }
        if idx < feature_elements {
            output_features[idx] = features[idx];
            terminate!();
        }
        let meta_idx = idx - feature_elements;
        if meta_idx >= metadata_elements {
            terminate!();
        }
        let observed_value = observed[meta_idx];
        output_observed[meta_idx] = observed_value;
        let mut age_value = age_frames[meta_idx];
        if age_observed > 0 {
            age_value += observed_value;
        }
        output_age_frames[meta_idx] = age_value;
    }

    #[cube(launch_unchecked, address_type = "dynamic")]
    pub(super) fn sparse_assign_feature_memory_kernel(
        tokens: &LinearView<f32>,
        token_indices: &LinearView<i32>,
        output_features: &mut LinearView<f32, ReadWrite>,
        output_observed: &mut LinearView<f32, ReadWrite>,
        output_age_frames: &mut LinearView<f32, ReadWrite>,
        sparse_feature_elements: usize,
        sparse_metadata_elements: usize,
        sparse_len: usize,
        dense_tokens: usize,
        embed_dim: usize,
    ) {
        let idx = ABSOLUTE_POS;
        if idx >= sparse_feature_elements + sparse_metadata_elements {
            terminate!();
        }
        if idx < sparse_feature_elements {
            let row_stride = sparse_len * embed_dim;
            let batch = idx / row_stride;
            let local = idx % row_stride;
            let sparse_index = local / embed_dim;
            let channel = local % embed_dim;
            let dense_index_i = token_indices[batch * sparse_len + sparse_index];
            if dense_index_i < 0 {
                terminate!();
            }
            let dense_index = usize::cast_from(dense_index_i);
            if dense_index >= dense_tokens {
                terminate!();
            }
            let out_index = (batch * dense_tokens + dense_index) * embed_dim + channel;
            output_features[out_index] = tokens[idx];
            terminate!();
        }

        let meta_idx = idx - sparse_feature_elements;
        if meta_idx >= sparse_metadata_elements {
            terminate!();
        }
        let batch = meta_idx / sparse_len;
        let sparse_index = meta_idx % sparse_len;
        let dense_index_i = token_indices[batch * sparse_len + sparse_index];
        if dense_index_i < 0 {
            terminate!();
        }
        let dense_index = usize::cast_from(dense_index_i);
        if dense_index >= dense_tokens {
            terminate!();
        }
        let out_index = batch * dense_tokens + dense_index;
        output_observed[out_index] = 1.0;
        output_age_frames[out_index] = 0.0;
    }

    pub(super) fn sparse_feature_memory_assign_latest_raw<
        R: burn_cubecl::CubeRuntime,
        BT: burn_cubecl::BoolElement,
    >(
        features: BurnTensor<burn_cubecl::CubeBackend<R, f32, i32, BT>, 3>,
        observed: BurnTensor<burn_cubecl::CubeBackend<R, f32, i32, BT>, 2>,
        age_frames: BurnTensor<burn_cubecl::CubeBackend<R, f32, i32, BT>, 2>,
        tokens: BurnTensor<burn_cubecl::CubeBackend<R, f32, i32, BT>, 3>,
        token_indices: BurnTensor<burn_cubecl::CubeBackend<R, f32, i32, BT>, 2, Int>,
        age_observed: bool,
    ) -> Result<
        super::SparseFeatureMemoryAssignOutput<burn_cubecl::CubeBackend<R, f32, i32, BT>>,
        String,
    > {
        let [batch, dense_tokens, embed_dim] = features.dims();
        let [observed_batch, observed_tokens] = observed.dims();
        let [age_batch, age_tokens] = age_frames.dims();
        if observed_batch != batch
            || observed_tokens != dense_tokens
            || age_batch != batch
            || age_tokens != dense_tokens
        {
            return Err(format!(
                "feature memory metadata dims mismatch: features=[{batch},{dense_tokens},{embed_dim}] observed=[{observed_batch},{observed_tokens}] age=[{age_batch},{age_tokens}]"
            ));
        }
        let [token_batch, sparse_len, token_dim] = tokens.dims();
        let [index_batch, index_len] = token_indices.dims();
        if token_batch != batch || token_dim != embed_dim {
            return Err(format!(
                "feature memory token dims mismatch: tokens=[{token_batch},{sparse_len},{token_dim}] expected=[{batch},sparse,{embed_dim}]"
            ));
        }
        if index_batch != batch || index_len != sparse_len {
            return Err(format!(
                "feature memory index dims mismatch: indices=[{index_batch},{index_len}] expected=[{batch},{sparse_len}]"
            ));
        }
        if sparse_len == 0 {
            return Err("feature memory sparse assignment requires at least one token".to_string());
        }

        let feature_elements = batch
            .checked_mul(dense_tokens)
            .and_then(|value| value.checked_mul(embed_dim))
            .ok_or_else(|| "feature memory feature element count overflow".to_string())?;
        let metadata_elements = batch
            .checked_mul(dense_tokens)
            .ok_or_else(|| "feature memory metadata element count overflow".to_string())?;
        let sparse_feature_elements = batch
            .checked_mul(sparse_len)
            .and_then(|value| value.checked_mul(embed_dim))
            .ok_or_else(|| "feature memory sparse feature element count overflow".to_string())?;
        let sparse_metadata_elements = batch
            .checked_mul(sparse_len)
            .ok_or_else(|| "feature memory sparse metadata element count overflow".to_string())?;

        let features_p = features.into_primitive().tensor();
        let observed_p = observed.into_primitive().tensor();
        let age_frames_p = age_frames.into_primitive().tensor();
        let tokens_p = tokens.into_primitive().tensor();
        let token_indices_p = token_indices.reshape([batch * sparse_len]).into_primitive();

        let feature_bytes = feature_elements
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| "feature memory feature byte size overflow".to_string())?;
        let metadata_bytes = metadata_elements
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| "feature memory metadata byte size overflow".to_string())?;
        let output_features = CubeTensor::new_contiguous(
            features_p.client.clone(),
            features_p.device.clone(),
            Shape::new([batch, dense_tokens, embed_dim]),
            features_p.client.empty(feature_bytes),
            DType::F32,
        );
        let output_observed = CubeTensor::new_contiguous(
            features_p.client.clone(),
            features_p.device.clone(),
            Shape::new([batch, dense_tokens]),
            features_p.client.empty(metadata_bytes),
            DType::F32,
        );
        let output_age_frames = CubeTensor::new_contiguous(
            features_p.client.clone(),
            features_p.device.clone(),
            Shape::new([batch, dense_tokens]),
            features_p.client.empty(metadata_bytes),
            DType::F32,
        );

        let copy_elements = feature_elements
            .checked_add(metadata_elements)
            .ok_or_else(|| "feature memory copy element count overflow".to_string())?;
        let assign_elements = sparse_feature_elements
            .checked_add(sparse_metadata_elements)
            .ok_or_else(|| "feature memory sparse assign element count overflow".to_string())?;
        let copy_cube_dim = CubeDim::new(&features_p.client, copy_elements);
        let assign_cube_dim = CubeDim::new(&features_p.client, assign_elements);
        let copy_cube_count =
            calculate_cube_count_elemwise(&features_p.client, copy_elements, copy_cube_dim);
        let assign_cube_count =
            calculate_cube_count_elemwise(&features_p.client, assign_elements, assign_cube_dim);
        let client = features_p.client.clone();
        let address_type = [
            features_p.required_address_type(),
            observed_p.required_address_type(),
            age_frames_p.required_address_type(),
            tokens_p.required_address_type(),
            token_indices_p.required_address_type(),
            output_features.required_address_type(),
            output_observed.required_address_type(),
            output_age_frames.required_address_type(),
        ]
        .into_iter()
        .max()
        .unwrap_or_default();

        unsafe {
            copy_feature_memory_kernel::launch_unchecked::<R>(
                &client,
                copy_cube_count,
                copy_cube_dim,
                address_type,
                features_p.into_linear_view(),
                observed_p.into_linear_view(),
                age_frames_p.into_linear_view(),
                output_features.clone().into_linear_view(),
                output_observed.clone().into_linear_view(),
                output_age_frames.clone().into_linear_view(),
                feature_elements,
                metadata_elements,
                if age_observed { 1u32 } else { 0u32 },
            );
            sparse_assign_feature_memory_kernel::launch_unchecked::<R>(
                &client,
                assign_cube_count,
                assign_cube_dim,
                address_type,
                tokens_p.into_linear_view(),
                token_indices_p.into_linear_view(),
                output_features.clone().into_linear_view(),
                output_observed.clone().into_linear_view(),
                output_age_frames.clone().into_linear_view(),
                sparse_feature_elements,
                sparse_metadata_elements,
                sparse_len,
                dense_tokens,
                embed_dim,
            );
        }

        Ok(super::SparseFeatureMemoryAssignOutput {
            features: BurnTensor::from_primitive(TensorPrimitive::Float(output_features)),
            observed: BurnTensor::from_primitive(TensorPrimitive::Float(output_observed)),
            age_frames: BurnTensor::from_primitive(TensorPrimitive::Float(output_age_frames)),
        })
    }
}

#[cfg(feature = "sparse-feature-memory-wgpu")]
type RawWgpuBackend = burn_cubecl::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;

#[cfg(feature = "sparse-feature-memory-cuda")]
type RawCudaBackend =
    burn_cubecl::CubeBackend<burn_cubecl::cubecl::cuda::CudaRuntime, f32, i32, u8>;

#[cfg(feature = "sparse-feature-memory-wgpu")]
pub(crate) fn sparse_feature_memory_assign_latest_wgpu_raw(
    features: Tensor<RawWgpuBackend, 3>,
    observed: Tensor<RawWgpuBackend, 2>,
    age_frames: Tensor<RawWgpuBackend, 2>,
    tokens: Tensor<RawWgpuBackend, 3>,
    token_indices: Tensor<RawWgpuBackend, 2, Int>,
    age_observed: bool,
) -> Result<SparseFeatureMemoryAssignOutput<RawWgpuBackend>, String> {
    cube::sparse_feature_memory_assign_latest_raw::<burn_wgpu::WgpuRuntime, u32>(
        features,
        observed,
        age_frames,
        tokens,
        token_indices,
        age_observed,
    )
}

#[cfg(feature = "sparse-feature-memory-cuda")]
pub(crate) fn sparse_feature_memory_assign_latest_cuda_raw(
    features: Tensor<RawCudaBackend, 3>,
    observed: Tensor<RawCudaBackend, 2>,
    age_frames: Tensor<RawCudaBackend, 2>,
    tokens: Tensor<RawCudaBackend, 3>,
    token_indices: Tensor<RawCudaBackend, 2, Int>,
    age_observed: bool,
) -> Result<SparseFeatureMemoryAssignOutput<RawCudaBackend>, String> {
    cube::sparse_feature_memory_assign_latest_raw::<burn_cubecl::cubecl::cuda::CudaRuntime, u8>(
        features,
        observed,
        age_frames,
        tokens,
        token_indices,
        age_observed,
    )
}

#[cfg(feature = "sparse-feature-memory-wgpu")]
pub(crate) fn sparse_feature_memory_assign_latest_wgpu_fusion(
    features: Tensor<burn::backend::Wgpu<f32, i32>, 3>,
    observed: Tensor<burn::backend::Wgpu<f32, i32>, 2>,
    age_frames: Tensor<burn::backend::Wgpu<f32, i32>, 2>,
    tokens: Tensor<burn::backend::Wgpu<f32, i32>, 3>,
    token_indices: Tensor<burn::backend::Wgpu<f32, i32>, 2, Int>,
    age_observed: bool,
) -> SparseFeatureMemoryAssignOutput<burn::backend::Wgpu<f32, i32>> {
    use burn::tensor::Tensor as BurnTensor;
    use burn_backend::{DType, Shape, TensorPrimitive};
    use burn_fusion::stream::{Operation, OperationStreams};
    use burn_ir::{CustomOpIr, OperationIr, OperationOutput, TensorIr, TensorStatus};

    type RawBackend = RawWgpuBackend;

    #[derive(Debug)]
    struct SparseFeatureMemoryAssignWgpuFusionOp {
        age_observed: bool,
        desc: CustomOpIr,
    }

    impl Operation<<RawBackend as burn_fusion::FusionBackend>::FusionRuntime>
        for SparseFeatureMemoryAssignWgpuFusionOp
    {
        fn execute(
            &self,
            handles: &mut burn_ir::HandleContainer<
                burn_fusion::FusionHandle<
                    <RawBackend as burn_fusion::FusionBackend>::FusionRuntime,
                >,
            >,
        ) {
            let (inputs, outputs) = self.desc.as_fixed::<5, 3>();
            let features = BurnTensor::<RawBackend, 3>::from_primitive(TensorPrimitive::Float(
                handles.get_float_tensor::<RawBackend>(&inputs[0]),
            ));
            let observed = BurnTensor::<RawBackend, 2>::from_primitive(TensorPrimitive::Float(
                handles.get_float_tensor::<RawBackend>(&inputs[1]),
            ));
            let age_frames = BurnTensor::<RawBackend, 2>::from_primitive(TensorPrimitive::Float(
                handles.get_float_tensor::<RawBackend>(&inputs[2]),
            ));
            let tokens = BurnTensor::<RawBackend, 3>::from_primitive(TensorPrimitive::Float(
                handles.get_float_tensor::<RawBackend>(&inputs[3]),
            ));
            let token_indices = BurnTensor::<RawBackend, 2, Int>::from_primitive(
                handles.get_int_tensor::<RawBackend>(&inputs[4]),
            );
            let output =
                crate::sparse_feature_memory::sparse_feature_memory_assign_latest_wgpu_raw(
                    features,
                    observed,
                    age_frames,
                    tokens,
                    token_indices,
                    self.age_observed,
                )
                .expect("sparse feature memory WGPU fusion op failed");
            handles.register_float_tensor::<RawBackend>(
                &outputs[0].id,
                output.features.into_primitive().tensor(),
            );
            handles.register_float_tensor::<RawBackend>(
                &outputs[1].id,
                output.observed.into_primitive().tensor(),
            );
            handles.register_float_tensor::<RawBackend>(
                &outputs[2].id,
                output.age_frames.into_primitive().tensor(),
            );
        }
    }

    let [batch, dense_tokens, embed_dim] = features.shape().dims::<3>();
    let features = features.into_primitive().tensor();
    let observed = observed.into_primitive().tensor();
    let age_frames = age_frames.into_primitive().tensor();
    let tokens = tokens.into_primitive().tensor();
    let token_indices = token_indices.into_primitive();
    let client = features.client.clone();
    let streams =
        OperationStreams::with_inputs([&features, &observed, &age_frames, &tokens, &token_indices]);
    let inputs = [
        features.into_ir(),
        observed.into_ir(),
        age_frames.into_ir(),
        tokens.into_ir(),
        token_indices.into_ir(),
    ];
    let outputs = [
        TensorIr {
            status: TensorStatus::NotInit,
            shape: Shape::new([batch, dense_tokens, embed_dim]),
            id: client.create_empty_handle(),
            dtype: DType::F32,
        },
        TensorIr {
            status: TensorStatus::NotInit,
            shape: Shape::new([batch, dense_tokens]),
            id: client.create_empty_handle(),
            dtype: DType::F32,
        },
        TensorIr {
            status: TensorStatus::NotInit,
            shape: Shape::new([batch, dense_tokens]),
            id: client.create_empty_handle(),
            dtype: DType::F32,
        },
    ];
    let desc = CustomOpIr::new(
        "burn_jepa::sparse_feature_memory_assign_latest_wgpu",
        &inputs,
        &outputs,
    );
    let [features_out, observed_out, age_frames_out] = client
        .register(
            streams,
            OperationIr::Custom(desc.clone()),
            SparseFeatureMemoryAssignWgpuFusionOp { age_observed, desc },
        )
        .outputs();
    SparseFeatureMemoryAssignOutput {
        features: Tensor::from_primitive(TensorPrimitive::Float(features_out)),
        observed: Tensor::from_primitive(TensorPrimitive::Float(observed_out)),
        age_frames: Tensor::from_primitive(TensorPrimitive::Float(age_frames_out)),
    }
}

#[cfg(feature = "sparse-feature-memory-cuda")]
pub(crate) fn sparse_feature_memory_assign_latest_cuda_fusion(
    features: Tensor<burn::backend::Cuda<f32, i32>, 3>,
    observed: Tensor<burn::backend::Cuda<f32, i32>, 2>,
    age_frames: Tensor<burn::backend::Cuda<f32, i32>, 2>,
    tokens: Tensor<burn::backend::Cuda<f32, i32>, 3>,
    token_indices: Tensor<burn::backend::Cuda<f32, i32>, 2, Int>,
    age_observed: bool,
) -> SparseFeatureMemoryAssignOutput<burn::backend::Cuda<f32, i32>> {
    use burn::tensor::Tensor as BurnTensor;
    use burn_backend::{DType, Shape, TensorPrimitive};
    use burn_fusion::stream::{Operation, OperationStreams};
    use burn_ir::{CustomOpIr, OperationIr, OperationOutput, TensorIr, TensorStatus};

    type RawBackend = RawCudaBackend;

    #[derive(Debug)]
    struct SparseFeatureMemoryAssignCudaFusionOp {
        age_observed: bool,
        desc: CustomOpIr,
    }

    impl Operation<<RawBackend as burn_fusion::FusionBackend>::FusionRuntime>
        for SparseFeatureMemoryAssignCudaFusionOp
    {
        fn execute(
            &self,
            handles: &mut burn_ir::HandleContainer<
                burn_fusion::FusionHandle<
                    <RawBackend as burn_fusion::FusionBackend>::FusionRuntime,
                >,
            >,
        ) {
            let (inputs, outputs) = self.desc.as_fixed::<5, 3>();
            let features = BurnTensor::<RawBackend, 3>::from_primitive(TensorPrimitive::Float(
                handles.get_float_tensor::<RawBackend>(&inputs[0]),
            ));
            let observed = BurnTensor::<RawBackend, 2>::from_primitive(TensorPrimitive::Float(
                handles.get_float_tensor::<RawBackend>(&inputs[1]),
            ));
            let age_frames = BurnTensor::<RawBackend, 2>::from_primitive(TensorPrimitive::Float(
                handles.get_float_tensor::<RawBackend>(&inputs[2]),
            ));
            let tokens = BurnTensor::<RawBackend, 3>::from_primitive(TensorPrimitive::Float(
                handles.get_float_tensor::<RawBackend>(&inputs[3]),
            ));
            let token_indices = BurnTensor::<RawBackend, 2, Int>::from_primitive(
                handles.get_int_tensor::<RawBackend>(&inputs[4]),
            );
            let output =
                crate::sparse_feature_memory::sparse_feature_memory_assign_latest_cuda_raw(
                    features,
                    observed,
                    age_frames,
                    tokens,
                    token_indices,
                    self.age_observed,
                )
                .expect("sparse feature memory CUDA fusion op failed");
            handles.register_float_tensor::<RawBackend>(
                &outputs[0].id,
                output.features.into_primitive().tensor(),
            );
            handles.register_float_tensor::<RawBackend>(
                &outputs[1].id,
                output.observed.into_primitive().tensor(),
            );
            handles.register_float_tensor::<RawBackend>(
                &outputs[2].id,
                output.age_frames.into_primitive().tensor(),
            );
        }
    }

    let [batch, dense_tokens, embed_dim] = features.shape().dims::<3>();
    let features = features.into_primitive().tensor();
    let observed = observed.into_primitive().tensor();
    let age_frames = age_frames.into_primitive().tensor();
    let tokens = tokens.into_primitive().tensor();
    let token_indices = token_indices.into_primitive();
    let client = features.client.clone();
    let streams =
        OperationStreams::with_inputs([&features, &observed, &age_frames, &tokens, &token_indices]);
    let inputs = [
        features.into_ir(),
        observed.into_ir(),
        age_frames.into_ir(),
        tokens.into_ir(),
        token_indices.into_ir(),
    ];
    let outputs = [
        TensorIr {
            status: TensorStatus::NotInit,
            shape: Shape::new([batch, dense_tokens, embed_dim]),
            id: client.create_empty_handle(),
            dtype: DType::F32,
        },
        TensorIr {
            status: TensorStatus::NotInit,
            shape: Shape::new([batch, dense_tokens]),
            id: client.create_empty_handle(),
            dtype: DType::F32,
        },
        TensorIr {
            status: TensorStatus::NotInit,
            shape: Shape::new([batch, dense_tokens]),
            id: client.create_empty_handle(),
            dtype: DType::F32,
        },
    ];
    let desc = CustomOpIr::new(
        "burn_jepa::sparse_feature_memory_assign_latest_cuda",
        &inputs,
        &outputs,
    );
    let [features_out, observed_out, age_frames_out] = client
        .register(
            streams,
            OperationIr::Custom(desc.clone()),
            SparseFeatureMemoryAssignCudaFusionOp { age_observed, desc },
        )
        .outputs();
    SparseFeatureMemoryAssignOutput {
        features: Tensor::from_primitive(TensorPrimitive::Float(features_out)),
        observed: Tensor::from_primitive(TensorPrimitive::Float(observed_out)),
        age_frames: Tensor::from_primitive(TensorPrimitive::Float(age_frames_out)),
    }
}
