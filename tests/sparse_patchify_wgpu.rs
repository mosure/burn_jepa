#![cfg(feature = "sparse-patchify-wgpu")]

use burn::tensor::Tensor;
use burn::tensor::backend::BackendTypes;
use burn_jepa::{
    AnyUp, AnyUpConfig, FeatureFrameRequest, SparseImageTokenGrid, SparseJepaAnyUpPcaEncodePath,
    SparseJepaAnyUpPcaMeasurementConfig, SparseJepaAnyUpPcaPipeline,
    SparseJepaAnyUpPcaPipelineConfig, SparseMaskBatch, SparsePatchifyBatchPlan, SparsePatchifyPlan,
    SparseTokenMask, TemporalSparseJepaStream, TemporalSparseJepaStreamConfig, TttEncoderConfig,
    VJepa2_1Model, VJepaConfig, VJepaTttModel, apply_token_mask,
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
fn wgpu_sparse_image_patchify_matches_dense_image_encoder_on_selected_tokens() {
    let device = <B as BackendTypes>::Device::default();
    let config = VJepaConfig::tiny_for_tests();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let grid = burn_jepa::TokenGridShape::new(1, config.grid_height(), config.grid_width());
    let mask = SparseTokenMask::new(vec![0, 3], grid.len()).expect("mask");
    let plan = SparsePatchifyPlan::<B>::new(mask.clone(), grid, 1, &device).expect("plan");
    let values = (0..config.in_channels * config.image_size * config.image_size)
        .map(|idx| (idx as f32).sin() * 0.01)
        .collect::<Vec<_>>();
    let image = Tensor::<B, 1>::from_floats(values.as_slice(), &device).reshape([
        1,
        config.in_channels,
        config.image_size,
        config.image_size,
    ]);

    let dense = model
        .encode_image(image.clone(), Some(&mask))
        .tokens
        .to_data();
    let sparse = model
        .encode_image_sparse_patchify_wgpu(image, &plan)
        .expect("sparse image patchify encode")
        .tokens
        .to_data();
    assert_close(&dense, &sparse, "wgpu sparse image patchify encoder");
}

#[test]
fn wgpu_highres_pipeline_uses_sparse_image_patchify_encode_path() {
    let device = <B as BackendTypes>::Device::default();
    let config = VJepaConfig::tiny_for_tests();
    let jepa = VJepa2_1Model::<B>::new(&config, &device);
    let anyup = AnyUp::<B>::new(AnyUpConfig::tiny_for_tests(), &device).expect("anyup");
    let mut pipeline = SparseJepaAnyUpPcaPipeline::<B>::new(
        jepa,
        anyup,
        &config,
        SparseJepaAnyUpPcaPipelineConfig {
            anyup_q_chunk_size: Some(1),
            measurement: SparseJepaAnyUpPcaMeasurementConfig::enabled(),
            ..SparseJepaAnyUpPcaPipelineConfig::default()
        },
        1,
        [config.image_size, config.image_size],
        &device,
    )
    .expect("pipeline");
    let image = Tensor::<B, 4>::ones([1, 3, config.image_size, config.image_size], &device);
    let mask = SparseTokenMask::new(vec![0, 3], pipeline.grid().len()).expect("mask");
    let mask_batch = SparseMaskBatch::uniform(mask, 1, &device).expect("mask batch");
    let patchify_plan =
        SparsePatchifyBatchPlan::new(mask_batch, pipeline.grid(), &device).expect("patchify plan");

    let measured = pipeline
        .step_image_with_sparse_patchify_plan_wgpu_nodes_measured(
            image,
            &patchify_plan,
            FeatureFrameRequest::none(),
            SparseJepaAnyUpPcaMeasurementConfig::enabled(),
        )
        .expect("sparse patchify high-res step");

    assert_eq!(
        measured.metrics.encode_path,
        SparseJepaAnyUpPcaEncodePath::SparsePatchify
    );
    assert_eq!(
        measured.output.encoded.tokens.shape().dims::<3>(),
        [1, 2, 32]
    );
    assert_eq!(
        measured.output.low_res.features.shape().dims::<4>(),
        [1, 32, config.grid_height(), config.grid_width()]
    );
    assert!(!measured.output.has_low_res_pca());
    assert!(!measured.output.has_high_res_pca());
}

#[test]
fn wgpu_ttt_sparse_image_patchify_matches_dense_patch_embed_on_selected_tokens() {
    let device = <B as BackendTypes>::Device::default();
    let config = VJepaConfig::tiny_for_tests();
    let base = VJepa2_1Model::<B>::new(&config, &device);
    let model =
        VJepaTttModel::from_model(base, TttEncoderConfig::default(), &device).expect("TTT model");
    let frame_grid = burn_jepa::TokenGridShape::new(1, config.grid_height(), config.grid_width());
    let mask = SparseTokenMask::new(vec![0, 3], frame_grid.len()).expect("mask");
    let plan = SparsePatchifyPlan::<B>::new(mask.clone(), frame_grid, 1, &device).expect("plan");
    let values = (0..config.in_channels * config.image_size * config.image_size)
        .map(|idx| (idx as f32).cos() * 0.01)
        .collect::<Vec<_>>();
    let image = Tensor::<B, 1>::from_floats(values.as_slice(), &device).reshape([
        1,
        config.in_channels,
        config.image_size,
        config.image_size,
    ]);

    let dense = model
        .encoder
        .base
        .image_patch_embed
        .forward(image.clone().reshape([
            1,
            config.in_channels,
            1,
            config.image_size,
            config.image_size,
        ]));
    let dense = apply_token_mask(dense, mask.to_tensor::<B>(1, &device)).to_data();
    let sparse = model
        .encoder
        .sparse_patchify_image_wgpu(image, &plan)
        .expect("sparse image patchify")
        .to_data();
    assert_close(&dense, &sparse, "TTT sparse image patchify");
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
    assert!(!sparse.reused_encoder_plan);
    assert!(sparse_reused.reused_patchify_plan);
    assert!(sparse_reused.reused_encoder_plan);
    assert!(sparse_reused.temporal.reused_predictor_plan);
}

#[test]
fn wgpu_temporal_stream_accepts_precomputed_masks_and_reuses_sparse_plan() {
    let device = <B as BackendTypes>::Device::default();
    let config = VJepaConfig::tiny_for_tests();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let stream_config = TemporalSparseJepaStreamConfig::new(1, 1, SparseImageTokenGrid::new(1, 1))
        .with_keyframe_interval(4);
    let grid = config.token_grid();
    let context_mask = SparseTokenMask::new(vec![0, 2, 5, 7], grid.len()).expect("context");
    let target_mask = SparseTokenMask::new(vec![1, 3, 4, 6], grid.len()).expect("target");
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
    let mut dense_stream = TemporalSparseJepaStream::<B>::new(stream_config);
    let mut sparse_stream = TemporalSparseJepaStream::<B>::new(stream_config);

    let dense = dense_stream
        .forward_masks(
            &model,
            video.clone(),
            context_mask.clone(),
            target_mask.clone(),
            0,
        )
        .expect("dense precomputed-mask stream");
    let sparse = sparse_stream
        .forward_masks_sparse_patchify_wgpu(
            &model,
            video.clone(),
            context_mask.clone(),
            target_mask.clone(),
            0,
        )
        .expect("sparse patchify precomputed-mask stream");
    let sparse_reused = sparse_stream
        .forward_masks_sparse_patchify_wgpu(&model, video, context_mask, target_mask, 0)
        .expect("reused sparse patchify precomputed-mask stream");

    assert_close(
        &dense.context.tokens.to_data(),
        &sparse.context.tokens.to_data(),
        "precomputed context",
    );
    assert_close(
        &dense.temporal.predictor.target_predictions.to_data(),
        &sparse.temporal.predictor.target_predictions.to_data(),
        "precomputed predictor",
    );
    assert!(!sparse.reused_patchify_plan);
    assert!(!sparse.reused_encoder_plan);
    assert!(sparse_reused.reused_patchify_plan);
    assert!(sparse_reused.reused_encoder_plan);
    assert!(sparse_reused.temporal.reused_predictor_plan);
}

#[test]
fn wgpu_temporal_stream_dense_keyframe_prediction_is_opt_in() {
    let device = <B as BackendTypes>::Device::default();
    let config = VJepaConfig::tiny_for_tests();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let stream_config = TemporalSparseJepaStreamConfig::new(4, 2, SparseImageTokenGrid::new(2, 2))
        .with_keyframe_interval(2)
        .with_dense_keyframe_prediction(true);
    let frame_tokens = vec![vec![0], vec![1], vec![2], vec![3]];
    let video = Tensor::<B, 5>::zeros(
        [
            1,
            config.in_channels,
            config.num_frames,
            config.image_size,
            config.image_size,
        ],
        &device,
    );
    let mut stream = TemporalSparseJepaStream::<B>::new(stream_config);

    let keyframe = stream
        .forward_frame_tokens_sparse_patchify_wgpu(&model, video.clone(), &frame_tokens, 0)
        .expect("sparse patchify keyframe");
    let update = stream
        .forward_frame_tokens_sparse_patchify_wgpu(&model, video, &frame_tokens, 0)
        .expect("sparse patchify update");

    let dense_prediction = keyframe
        .dense_keyframe_prediction
        .as_ref()
        .expect("dense keyframe prediction");
    assert_eq!(
        dense_prediction.predictions.shape().dims::<3>()[1],
        keyframe.masks.target_mask.len()
    );
    assert_eq!(
        dense_prediction.targets.shape().dims::<3>()[1],
        keyframe.masks.target_mask.len()
    );
    assert!(update.dense_keyframe_prediction.is_none());
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
