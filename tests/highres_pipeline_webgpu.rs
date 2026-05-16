#![cfg(feature = "webgpu")]

use burn::tensor::{Int, Tensor, TensorData};
use burn_jepa::{
    AnyUp, AnyUpConfig, FeatureFrameRequest, SparseJepaAnyUpPcaPipeline,
    SparseJepaAnyUpPcaPipelineConfig, SparseTokenMask, VJepa2_1Model, VJepaConfig,
};

type B = burn::backend::WebGpu<f32, i32>;

#[test]
fn dense_tiny_jepa_pipeline_updates_every_low_res_cache_token_webgpu() {
    let device = Default::default();
    let model_config = VJepaConfig::tiny_for_tests();
    let mut pipeline = tiny_pipeline(&device, &model_config);
    let mask = SparseTokenMask::all(pipeline.grid().len());

    let output = pipeline
        .step_image_with_mask_nodes_measured(
            gradient_image(&model_config, &device),
            &mask,
            FeatureFrameRequest::low_res(),
        )
        .expect("dense WebGPU pipeline")
        .output;

    assert_eq!(values_i64(output.encoded.token_indices), vec![0, 1, 2, 3]);
    assert_eq!(
        values2(output.token_cache.observed),
        vec![1.0; pipeline.grid().len()]
    );
    assert_all_token_norms_nonzero(
        &values3(output.token_cache.features),
        pipeline.grid().len(),
        32,
    );
}

#[test]
fn sparse_tiny_jepa_pipeline_accumulates_low_res_cache_tokens_webgpu() {
    let device = Default::default();
    let model_config = VJepaConfig::tiny_for_tests();
    let mut pipeline = tiny_pipeline(&device, &model_config);
    let first_mask = SparseTokenMask::new(vec![0, 2], pipeline.grid().len()).expect("first mask");
    let second_mask = SparseTokenMask::new(vec![1, 3], pipeline.grid().len()).expect("second mask");

    let first = pipeline
        .step_image_with_mask_nodes_measured(
            gradient_image(&model_config, &device),
            &first_mask,
            FeatureFrameRequest::low_res(),
        )
        .expect("first sparse WebGPU pipeline")
        .output;
    assert_eq!(values_i64(first.encoded.token_indices), vec![0, 2]);
    assert_eq!(
        values2(first.token_cache.observed),
        vec![1.0, 0.0, 1.0, 0.0]
    );

    let second = pipeline
        .step_image_with_mask_nodes_measured(
            shifted_gradient_image(&model_config, &device),
            &second_mask,
            FeatureFrameRequest::low_res(),
        )
        .expect("second sparse WebGPU pipeline")
        .output;
    assert_eq!(values_i64(second.encoded.token_indices), vec![1, 3]);
    assert_eq!(
        values2(second.token_cache.observed),
        vec![1.0; pipeline.grid().len()]
    );
    assert_all_token_norms_nonzero(
        &values3(second.token_cache.features),
        pipeline.grid().len(),
        32,
    );
}

fn tiny_pipeline(
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
    model_config: &VJepaConfig,
) -> SparseJepaAnyUpPcaPipeline<B> {
    let jepa = VJepa2_1Model::<B>::new(model_config, device);
    let anyup = AnyUp::<B>::new(AnyUpConfig::tiny_for_tests(), device).expect("AnyUp");
    SparseJepaAnyUpPcaPipeline::<B>::new(
        jepa,
        anyup,
        model_config,
        SparseJepaAnyUpPcaPipelineConfig {
            anyup_q_chunk_size: Some(1),
            ..SparseJepaAnyUpPcaPipelineConfig::default()
        },
        1,
        [model_config.image_size, model_config.image_size],
        device,
    )
    .expect("pipeline")
}

fn gradient_image(
    model_config: &VJepaConfig,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 4> {
    image_with_shift(model_config, 0, 0, device)
}

fn shifted_gradient_image(
    model_config: &VJepaConfig,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 4> {
    image_with_shift(
        model_config,
        model_config.image_size / 4,
        model_config.image_size / 3,
        device,
    )
}

fn image_with_shift(
    model_config: &VJepaConfig,
    shift_x: usize,
    shift_y: usize,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 4> {
    let height = model_config.image_size;
    let width = model_config.image_size;
    let mut values = vec![0.0; 3 * height * width];
    for y in 0..height {
        for x in 0..width {
            let shifted_x = (x + shift_x) % width.max(1);
            let shifted_y = (y + shift_y) % height.max(1);
            let index = y * width + x;
            values[index] = shifted_x as f32 / width.max(1) as f32;
            values[height * width + index] = shifted_y as f32 / height.max(1) as f32;
            values[2 * height * width + index] =
                (shifted_x + shifted_y) as f32 / (height + width).max(1) as f32;
        }
    }
    Tensor::<B, 4>::from_data(TensorData::new(values, [1, 3, height, width]), device)
}

fn assert_all_token_norms_nonzero(values: &[f32], token_count: usize, dim: usize) {
    assert_eq!(values.len(), token_count * dim);
    for token in 0..token_count {
        let norm = values[token * dim..(token + 1) * dim]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!(
            norm > 1.0e-4,
            "token {token} should have been written, norm={norm}, values={values:?}"
        );
    }
}

fn values3(tensor: Tensor<B, 3>) -> Vec<f32> {
    tensor.to_data().to_vec::<f32>().expect("tensor values")
}

fn values2(tensor: Tensor<B, 2>) -> Vec<f32> {
    tensor.to_data().to_vec::<f32>().expect("tensor values")
}

fn values_i64<const D: usize>(tensor: Tensor<B, D, Int>) -> Vec<i64> {
    tensor
        .to_data()
        .to_vec::<i32>()
        .expect("integer tensor values")
        .into_iter()
        .map(i64::from)
        .collect()
}
