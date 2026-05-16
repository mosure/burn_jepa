#![cfg(feature = "webgpu")]

use burn::tensor::{Int, Tensor, TensorData};
use burn_anyup::{
    AnyUpHighResFeatureMemory, AnyUpHighResFeatureMemoryConfig, sparse_low_features_to_nchw,
};

type B = burn::backend::WebGpu<f32, i32>;

#[test]
fn high_res_sparse_feature_memory_updates_all_selected_pixels_webgpu() {
    let device = Default::default();
    let mut memory = AnyUpHighResFeatureMemory::<B>::new(
        AnyUpHighResFeatureMemoryConfig::default(),
        1,
        [2, 3],
        2,
        &device,
    )
    .expect("AnyUp high-res memory");

    let first = memory
        .update_tokens(
            tensor3(&[10.0, 11.0, 20.0, 21.0], [1, 2, 2], &device),
            indices(&[0, 4], [1, 2], &device),
        )
        .expect("first sparse update");
    assert_eq!(
        values3(first.features),
        vec![
            10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 21.0, 0.0, 0.0
        ]
    );

    let second = memory
        .update_tokens(
            tensor3(&[30.0, 31.0, 40.0, 41.0], [1, 2, 2], &device),
            indices(&[2, 5], [1, 2], &device),
        )
        .expect("second sparse update");
    assert_eq!(
        values3(second.features),
        vec![
            10.0, 11.0, 0.0, 0.0, 30.0, 31.0, 0.0, 0.0, 20.0, 21.0, 40.0, 41.0
        ]
    );
    assert_eq!(values2(second.observed), vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
}

#[test]
fn sparse_low_features_to_nchw_writes_all_sparse_positions_webgpu() {
    let device = Default::default();
    let output = sparse_low_features_to_nchw(
        tensor3(&[1.0, 10.0, 3.0, 30.0, 5.0, 50.0], [1, 3, 2], &device),
        indices(&[0, 2, 4], [1, 3], &device),
        [2, 3],
        &device,
    )
    .expect("sparse low features");

    assert_eq!(
        values4(output),
        vec![
            1.0, 0.0, 3.0, 0.0, 5.0, 0.0, //
            10.0, 0.0, 30.0, 0.0, 50.0, 0.0,
        ]
    );
}

fn tensor3(
    values: &[f32],
    shape: [usize; 3],
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(TensorData::new(values.to_vec(), shape), device)
}

fn indices(
    values: &[i64],
    shape: [usize; 2],
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2, Int> {
    Tensor::<B, 2, Int>::from_data(TensorData::new(values.to_vec(), shape), device)
}

fn values4(tensor: Tensor<B, 4>) -> Vec<f32> {
    tensor.to_data().to_vec::<f32>().expect("tensor values")
}

fn values3(tensor: Tensor<B, 3>) -> Vec<f32> {
    tensor.to_data().to_vec::<f32>().expect("tensor values")
}

fn values2(tensor: Tensor<B, 2>) -> Vec<f32> {
    tensor.to_data().to_vec::<f32>().expect("tensor values")
}
