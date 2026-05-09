#![cfg(feature = "sparse-patchify-wgpu")]

use burn::tensor::Tensor;
use burn::tensor::backend::BackendTypes;
use burn_jepa::{
    SparseImageTokenGrid, SparsePatchifyPlan, SparseTokenMask, TemporalSparseJepaStream,
    TemporalSparseJepaStreamConfig, VJepa2_1Model, VJepaConfig,
};

type B = burn_flex_gmm::wgpu::DefaultWgpuBackend;

#[test]
fn wgpu_sparse_patchify_matches_dense_encoder_on_selected_tokens() {
    let device = <B as BackendTypes>::Device::default();
    let config = VJepaConfig::tiny_for_tests();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let grid = config.token_grid();
    let mask = SparseTokenMask::new(vec![0, 3, 7], grid.len()).expect("mask");
    let plan = SparsePatchifyPlan::<B>::new(mask.clone(), grid, 1, &device).expect("plan");
    let values =
        (0..config.in_channels * config.num_frames * config.image_size * config.image_size)
            .map(|idx| (idx as f32).sin() * 0.01)
            .collect::<Vec<_>>();
    let video = Tensor::<B, 1>::from_floats(values.as_slice(), &device).reshape([
        1,
        config.in_channels,
        config.num_frames,
        config.image_size,
        config.image_size,
    ]);

    let dense = model
        .encode_video(video.clone(), Some(&mask))
        .tokens
        .to_data();
    let sparse = model
        .encode_video_sparse_patchify_wgpu(video, &plan)
        .expect("sparse patchify encode")
        .tokens
        .to_data();
    let dense = dense.as_slice::<f32>().expect("dense f32");
    let sparse = sparse.as_slice::<f32>().expect("sparse f32");
    assert_eq!(dense.len(), sparse.len());
    for (idx, (lhs, rhs)) in dense.iter().zip(sparse.iter()).enumerate() {
        let diff = (lhs - rhs).abs();
        assert!(
            diff <= 5.0e-4,
            "sparse patchify encoder drift at {idx}: dense={lhs} sparse={rhs} diff={diff}"
        );
    }
}

#[test]
fn wgpu_temporal_stream_sparse_patchify_matches_dense_masked_stream() {
    let device = <B as BackendTypes>::Device::default();
    let config = VJepaConfig::tiny_for_tests();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let stream_config = TemporalSparseJepaStreamConfig::new(4, 2, SparseImageTokenGrid::new(2, 2))
        .with_keyframe_interval(4);
    let frame_tokens = vec![vec![0], vec![1], vec![2], vec![3]];
    let values =
        (0..config.in_channels * config.num_frames * config.image_size * config.image_size)
            .map(|idx| (idx as f32).cos() * 0.01)
            .collect::<Vec<_>>();
    let video = Tensor::<B, 1>::from_floats(values.as_slice(), &device).reshape([
        1,
        config.in_channels,
        config.num_frames,
        config.image_size,
        config.image_size,
    ]);
    let mut dense_stream = TemporalSparseJepaStream::<B>::new(stream_config);
    let mut sparse_stream = TemporalSparseJepaStream::<B>::new(stream_config);

    let dense = dense_stream
        .forward_frame_tokens(&model, video.clone(), &frame_tokens, 0)
        .expect("dense masked stream");
    let sparse = sparse_stream
        .forward_frame_tokens_sparse_patchify_wgpu(&model, video.clone(), &frame_tokens, 0)
        .expect("sparse patchify stream");
    let sparse_reused = sparse_stream
        .forward_frame_tokens_sparse_patchify_wgpu(&model, video, &frame_tokens, 0)
        .expect("sparse patchify stream reused");

    assert_eq!(
        dense.masks.context_mask.indices(),
        sparse.masks.context_mask.indices()
    );
    assert_eq!(
        dense.masks.target_mask.indices(),
        sparse.masks.target_mask.indices()
    );
    assert_close(
        &dense.context.tokens.to_data(),
        &sparse.context.tokens.to_data(),
        "context",
    );
    assert_close(
        &dense.temporal.predictor.target_predictions.to_data(),
        &sparse.temporal.predictor.target_predictions.to_data(),
        "predictor",
    );
    assert!(!sparse.reused_patchify_plan);
    assert!(sparse_reused.reused_patchify_plan);
    assert!(sparse_reused.temporal.reused_predictor_plan);
}

fn assert_close(left: &burn::tensor::TensorData, right: &burn::tensor::TensorData, label: &str) {
    let left = left.as_slice::<f32>().expect("left f32");
    let right = right.as_slice::<f32>().expect("right f32");
    assert_eq!(left.len(), right.len(), "{label} lengths differ");
    for (idx, (lhs, rhs)) in left.iter().zip(right).enumerate() {
        let diff = (lhs - rhs).abs();
        assert!(
            diff <= 5.0e-4,
            "{label} drift at {idx}: dense={lhs} sparse={rhs} diff={diff}"
        );
    }
}
