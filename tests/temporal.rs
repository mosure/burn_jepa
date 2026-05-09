use burn::tensor::backend::BackendTypes;
use burn::tensor::{Tensor, TensorData};
use burn_jepa::{
    SparseImageTokenGrid, SparseTokenMask, TemporalSparseJepaConfig, TemporalSparseJepaState,
    TemporalSparseJepaStream, TemporalSparseJepaStreamConfig, TemporalSparseMaskConfig,
    TemporalSparseMaskState, TokenGridShape, VJepa2_1Model, VJepaConfig,
};

type B = burn::backend::NdArray<f32>;

#[test]
fn temporal_state_reuses_predictor_plan_for_stable_masks() {
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let (context, target) = masks(&config);
    let mut state = TemporalSparseJepaState::<B>::new(
        TemporalSparseJepaConfig::default().with_keyframe_interval(8),
    );
    assert!(state.next_is_keyframe());

    let first = state
        .forward_predictor(
            &config,
            &model.predictor,
            feature_tensor(context.len(), config.encoder.embed_dim, 0.0, &device),
            &context,
            &target,
            config.token_grid(),
            0,
        )
        .expect("first temporal forward");
    let second = state
        .forward_predictor(
            &config,
            &model.predictor,
            feature_tensor(context.len(), config.encoder.embed_dim, 1.0, &device),
            &context,
            &target,
            config.token_grid(),
            0,
        )
        .expect("second temporal forward");

    assert!(first.keyframe);
    assert!(!first.reused_predictor_plan);
    assert!(!state.next_is_keyframe());
    assert!(!second.keyframe);
    assert!(second.reused_predictor_plan);
    assert_eq!(
        second.predictor.target_predictions.shape().dims::<3>()[1],
        target.len()
    );
}

#[test]
fn temporal_state_blends_sparse_features_between_keyframes() {
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let (context, target) = masks(&config);
    let mut state = TemporalSparseJepaState::<B>::new(
        TemporalSparseJepaConfig::default()
            .with_keyframe_interval(8)
            .with_feature_blend(0.25),
    );

    state
        .forward_predictor(
            &config,
            &model.predictor,
            feature_tensor(context.len(), config.encoder.embed_dim, 0.0, &device),
            &context,
            &target,
            config.token_grid(),
            0,
        )
        .expect("prime temporal state");
    let output = state
        .forward_predictor(
            &config,
            &model.predictor,
            feature_tensor(context.len(), config.encoder.embed_dim, 1.0, &device),
            &context,
            &target,
            config.token_grid(),
            0,
        )
        .expect("blended temporal forward");

    let values = output
        .features
        .to_data()
        .to_vec::<f32>()
        .expect("feature values");
    assert!(
        values.iter().all(|value| (*value - 0.25).abs() < 1.0e-6),
        "unexpected blended feature values: {values:?}"
    );
}

#[test]
fn temporal_state_rebuilds_predictor_plan_when_masks_change() {
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let (context, target) = masks(&config);
    let shifted_context =
        SparseTokenMask::new(vec![1, 3, 5, 7], config.num_patches()).expect("shifted context");
    let shifted_target =
        SparseTokenMask::new(vec![0, 2, 4, 6], config.num_patches()).expect("shifted target");
    let mut state = TemporalSparseJepaState::<B>::new(
        TemporalSparseJepaConfig::default().with_keyframe_interval(8),
    );

    state
        .forward_predictor(
            &config,
            &model.predictor,
            feature_tensor(context.len(), config.encoder.embed_dim, 0.0, &device),
            &context,
            &target,
            config.token_grid(),
            0,
        )
        .expect("prime temporal state");
    let output = state
        .forward_predictor(
            &config,
            &model.predictor,
            feature_tensor(
                shifted_context.len(),
                config.encoder.embed_dim,
                1.0,
                &device,
            ),
            &shifted_context,
            &shifted_target,
            config.token_grid(),
            0,
        )
        .expect("changed-mask temporal forward");

    assert!(!output.reused_predictor_plan);
}

#[test]
fn temporal_mask_state_projects_sparse_image_tokens_and_marks_keyframes() {
    let grid = TokenGridShape::new(2, 4, 4);
    let image_grid = SparseImageTokenGrid::new(2, 2);
    let frame_tokens = vec![vec![0], vec![], vec![3], vec![]];
    let mut state =
        TemporalSparseMaskState::new(TemporalSparseMaskConfig::new(6, 4).with_keyframe_interval(2));

    let first = state
        .next_from_frame_tokens(grid, 2, image_grid, &frame_tokens)
        .expect("first sparse masks");
    let second = state
        .next_from_frame_tokens(grid, 2, image_grid, &frame_tokens)
        .expect("second sparse masks");

    assert!(first.keyframe);
    assert!(!second.keyframe);
    assert_eq!(first.context_mask.len(), 6);
    assert_eq!(first.target_mask.len(), 4);
    assert_eq!(first.context_mask.dense_len(), grid.len());
    assert_eq!(first.target_mask.dense_len(), grid.len());
    for index in first.context_mask.indices() {
        assert!(!first.target_mask.indices().contains(index));
    }
}

#[test]
fn temporal_mask_state_rejects_all_context_masks() {
    let grid = TokenGridShape::new(1, 2, 2);
    let image_grid = SparseImageTokenGrid::new(1, 1);
    let frame_tokens = vec![vec![0]];
    let mut state = TemporalSparseMaskState::new(TemporalSparseMaskConfig::new(4, 1));

    let err = state
        .next_from_frame_tokens(grid, 1, image_grid, &frame_tokens)
        .expect_err("all-context mask should fail");

    assert!(
        err.to_string().contains("target token"),
        "unexpected error: {err:#}"
    );
}

#[test]
fn temporal_sparse_video_step_projects_encodes_and_reuses_plan() {
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let image_grid = SparseImageTokenGrid::new(2, 2);
    let frame_tokens = vec![vec![0], vec![1], vec![2], vec![3]];
    let mut mask_state =
        TemporalSparseMaskState::new(TemporalSparseMaskConfig::new(4, 2).with_keyframe_interval(2));
    let mut jepa_state = TemporalSparseJepaState::<B>::new(
        TemporalSparseJepaConfig::default().with_keyframe_interval(2),
    );
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

    let first_masks = mask_state
        .next_from_frame_tokens(
            config.token_grid(),
            config.tubelet_size,
            image_grid,
            &frame_tokens,
        )
        .expect("first projected masks");
    let first_context = model.encode_video(video.clone(), Some(&first_masks.context_mask));
    let first = jepa_state
        .forward_predictor(
            &config,
            &model.predictor,
            first_context.tokens,
            &first_masks.context_mask,
            &first_masks.target_mask,
            first_context.grid,
            0,
        )
        .expect("first sparse temporal video step");

    let second_masks = mask_state
        .next_from_frame_tokens(
            config.token_grid(),
            config.tubelet_size,
            image_grid,
            &frame_tokens,
        )
        .expect("second projected masks");
    let second_context = model.encode_video(video, Some(&second_masks.context_mask));
    let second = jepa_state
        .forward_predictor(
            &config,
            &model.predictor,
            second_context.tokens,
            &second_masks.context_mask,
            &second_masks.target_mask,
            second_context.grid,
            0,
        )
        .expect("second sparse temporal video step");

    assert!(first_masks.keyframe);
    assert!(first.keyframe);
    assert!(!first.reused_predictor_plan);
    assert!(!second_masks.keyframe);
    assert!(!second.keyframe);
    assert!(second.reused_predictor_plan);
    assert_eq!(
        first_masks.context_mask.indices(),
        second_masks.context_mask.indices()
    );
    assert_eq!(
        first_masks.target_mask.indices(),
        second_masks.target_mask.indices()
    );
    assert_eq!(
        second.predictor.target_predictions.shape().dims::<3>()[1],
        second_masks.target_mask.len()
    );
    assert_eq!(
        second.features.shape().dims::<3>()[1],
        second_masks.context_mask.len()
    );
}

#[test]
fn temporal_stream_projects_encodes_predicts_and_resets() {
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let frame_tokens = vec![vec![0], vec![1], vec![2], vec![3]];
    let mut stream = TemporalSparseJepaStream::<B>::new(
        TemporalSparseJepaStreamConfig::new(4, 2, SparseImageTokenGrid::new(2, 2))
            .with_keyframe_interval(2),
    );
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

    let first = stream
        .forward_frame_tokens(&model, video.clone(), &frame_tokens, 0)
        .expect("first temporal stream step");
    let second = stream
        .forward_frame_tokens(&model, video.clone(), &frame_tokens, 0)
        .expect("second temporal stream step");
    stream.reset();
    let reset = stream
        .forward_frame_tokens(&model, video, &frame_tokens, 0)
        .expect("reset temporal stream step");

    assert!(first.masks.keyframe);
    assert!(first.temporal.keyframe);
    assert!(!first.temporal.reused_predictor_plan);
    assert!(!second.masks.keyframe);
    assert!(!second.temporal.keyframe);
    assert!(second.temporal.reused_predictor_plan);
    assert!(reset.masks.keyframe);
    assert!(reset.temporal.keyframe);
    assert!(!reset.temporal.reused_predictor_plan);
    assert_eq!(first.context.tokens.shape().dims::<3>()[1], 4);
    assert_eq!(
        second
            .temporal
            .predictor
            .target_predictions
            .shape()
            .dims::<3>()[1],
        2
    );
}

#[test]
fn temporal_stream_can_refresh_dense_keyframes() {
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let frame_tokens = vec![vec![0], vec![1], vec![2], vec![3]];
    let mut stream = TemporalSparseJepaStream::<B>::new(
        TemporalSparseJepaStreamConfig::new(4, 2, SparseImageTokenGrid::new(2, 2))
            .with_keyframe_interval(2)
            .with_dense_keyframe_refresh(true),
    );
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

    let keyframe = stream
        .forward_frame_tokens(&model, video.clone(), &frame_tokens, 0)
        .expect("keyframe stream step");
    let sparse_update = stream
        .forward_frame_tokens(&model, video, &frame_tokens, 0)
        .expect("sparse stream step");

    let dense_keyframe = keyframe
        .dense_keyframe
        .as_ref()
        .expect("dense keyframe refresh");
    assert_eq!(dense_keyframe.grid, config.token_grid());
    assert_eq!(
        dense_keyframe.tokens.shape().dims::<3>()[1],
        config.num_patches()
    );
    assert!(sparse_update.dense_keyframe.is_none());
}

fn masks(config: &VJepaConfig) -> (SparseTokenMask, SparseTokenMask) {
    (
        SparseTokenMask::new(vec![0, 2, 5, 7], config.num_patches()).expect("context"),
        SparseTokenMask::new(vec![1, 3, 4, 6], config.num_patches()).expect("target"),
    )
}

fn feature_tensor(
    tokens: usize,
    dim: usize,
    value: f32,
    device: &<B as BackendTypes>::Device,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(
        TensorData::new(vec![value; tokens * dim], [1, tokens, dim]),
        device,
    )
}
