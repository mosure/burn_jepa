use burn::tensor::backend::BackendTypes;
use burn::tensor::{Tensor, TensorData};
use burn_jepa::{
    SparseImageTokenGrid, SparseTokenMask, TemporalSparseJepaConfig, TemporalSparseJepaState,
    TemporalSparseJepaStream, TemporalSparseJepaStreamConfig, TemporalSparseMaskConfig,
    TemporalSparseMaskState, TemporalSparsePredictorInput, TokenGridShape, VJepa2_1Model,
    VJepaConfig,
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
        .forward_predictor(temporal_input(
            &model,
            feature_tensor(context.len(), config.encoder.embed_dim, 0.0, &device),
            &context,
            &target,
            config.token_grid(),
        ))
        .expect("first temporal forward");
    let second = state
        .forward_predictor(temporal_input(
            &model,
            feature_tensor(context.len(), config.encoder.embed_dim, 1.0, &device),
            &context,
            &target,
            config.token_grid(),
        ))
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
        .forward_predictor(temporal_input(
            &model,
            feature_tensor(context.len(), config.encoder.embed_dim, 0.0, &device),
            &context,
            &target,
            config.token_grid(),
        ))
        .expect("prime temporal state");
    let output = state
        .forward_predictor(temporal_input(
            &model,
            feature_tensor(context.len(), config.encoder.embed_dim, 1.0, &device),
            &context,
            &target,
            config.token_grid(),
        ))
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
        .forward_predictor(temporal_input(
            &model,
            feature_tensor(context.len(), config.encoder.embed_dim, 0.0, &device),
            &context,
            &target,
            config.token_grid(),
        ))
        .expect("prime temporal state");
    let output = state
        .forward_predictor(temporal_input(
            &model,
            feature_tensor(
                shifted_context.len(),
                config.encoder.embed_dim,
                1.0,
                &device,
            ),
            &shifted_context,
            &shifted_target,
            config.token_grid(),
        ))
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
        .forward_predictor(temporal_input(
            &model,
            first_context.tokens,
            &first_masks.context_mask,
            &first_masks.target_mask,
            first_context.grid,
        ))
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
        .forward_predictor(temporal_input(
            &model,
            second_context.tokens,
            &second_masks.context_mask,
            &second_masks.target_mask,
            second_context.grid,
        ))
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
    assert!(!first.reused_encoder_plan);
    assert!(!first.reused_patchify_plan);
    assert!(!second.masks.keyframe);
    assert!(!second.temporal.keyframe);
    assert!(second.temporal.reused_predictor_plan);
    assert!(second.reused_encoder_plan);
    assert!(!second.reused_patchify_plan);
    assert!(reset.masks.keyframe);
    assert!(reset.temporal.keyframe);
    assert!(!reset.temporal.reused_predictor_plan);
    assert!(!reset.reused_encoder_plan);
    assert!(!reset.reused_patchify_plan);
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
fn temporal_stream_accepts_precomputed_masks_without_frame_projection() {
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let (context_mask, target_mask) = masks(&config);
    let mut stream = TemporalSparseJepaStream::<B>::new(
        TemporalSparseJepaStreamConfig::new(1, 1, SparseImageTokenGrid::new(1, 1))
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
        .forward_masks(
            &model,
            video.clone(),
            context_mask.clone(),
            target_mask.clone(),
            0,
        )
        .expect("first precomputed-mask stream step");
    let second = stream
        .forward_masks(&model, video, context_mask.clone(), target_mask.clone(), 0)
        .expect("second precomputed-mask stream step");

    assert!(first.masks.keyframe);
    assert!(!first.temporal.reused_predictor_plan);
    assert!(!first.reused_encoder_plan);
    assert_eq!(first.masks.context_mask.indices(), context_mask.indices());
    assert_eq!(first.masks.target_mask.indices(), target_mask.indices());
    assert!(!second.masks.keyframe);
    assert!(second.temporal.reused_predictor_plan);
    assert!(second.reused_encoder_plan);
    assert_eq!(
        second.context.tokens.shape().dims::<3>()[1],
        context_mask.len()
    );
    assert_eq!(
        second
            .temporal
            .predictor
            .target_predictions
            .shape()
            .dims::<3>()[1],
        target_mask.len()
    );
}

#[test]
fn temporal_stream_rejects_overlapping_precomputed_masks() {
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let (context_mask, _) = masks(&config);
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
    let mut stream = TemporalSparseJepaStream::<B>::new(
        TemporalSparseJepaStreamConfig::new(1, 1, SparseImageTokenGrid::new(1, 1))
            .with_keyframe_interval(2),
    );

    let err = stream
        .forward_masks(&model, video, context_mask.clone(), context_mask, 0)
        .expect_err("overlapping precomputed masks should fail");

    assert!(
        err.to_string()
            .contains("temporal context and target masks must not overlap")
    );
    assert!(stream.next_is_keyframe());
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
    assert!(keyframe.dense_keyframe_prediction.is_none());
    assert!(sparse_update.dense_keyframe.is_none());
    assert!(sparse_update.dense_keyframe_prediction.is_none());
}

#[test]
fn temporal_stream_can_refresh_dense_keyframe_predictions() {
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let frame_tokens = vec![vec![0], vec![1], vec![2], vec![3]];
    let mut stream = TemporalSparseJepaStream::<B>::new(
        TemporalSparseJepaStreamConfig::new(4, 2, SparseImageTokenGrid::new(2, 2))
            .with_keyframe_interval(2)
            .with_dense_keyframe_prediction(true),
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
    assert_eq!(
        dense_prediction.target_indices.shape().dims::<2>(),
        [1, keyframe.masks.target_mask.len()]
    );
    assert!(keyframe.dense_keyframe.is_none());
    assert!(sparse_update.dense_keyframe_prediction.is_none());
}

#[test]
fn temporal_stream_accepts_tubelet_sized_next_frame_windows() {
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let frame_tokens = vec![vec![0], vec![1]];
    let rolling_grid = TokenGridShape::new(
        1,
        config.image_size / config.patch_size,
        config.image_size / config.patch_size,
    );
    let mut stream = TemporalSparseJepaStream::<B>::new(
        TemporalSparseJepaStreamConfig::new(2, 1, SparseImageTokenGrid::new(2, 2))
            .with_keyframe_interval(2)
            .with_dense_keyframe_refresh(true),
    );
    let first_window = Tensor::<B, 5>::zeros(
        [
            1,
            config.in_channels,
            config.tubelet_size,
            config.image_size,
            config.image_size,
        ],
        &device,
    );
    let second_window = Tensor::<B, 5>::ones(
        [
            1,
            config.in_channels,
            config.tubelet_size,
            config.image_size,
            config.image_size,
        ],
        &device,
    );

    let keyframe = stream
        .forward_frame_tokens(&model, first_window, &frame_tokens, 0)
        .expect("tubelet-sized keyframe step");
    let update = stream
        .forward_frame_tokens(&model, second_window, &frame_tokens, 0)
        .expect("tubelet-sized sparse update");

    assert_eq!(keyframe.context.grid, rolling_grid);
    assert_eq!(keyframe.masks.context_mask.dense_len(), rolling_grid.len());
    assert_eq!(keyframe.context.tokens.shape().dims::<3>()[1], 2);
    assert!(keyframe.temporal.keyframe);
    assert!(keyframe.dense_keyframe.is_some());
    assert!(keyframe.dense_keyframe_prediction.is_none());
    assert_eq!(
        keyframe
            .dense_keyframe
            .as_ref()
            .expect("dense keyframe")
            .grid,
        rolling_grid
    );
    assert!(!update.temporal.keyframe);
    assert!(update.temporal.reused_predictor_plan);
    assert!(update.dense_keyframe.is_none());
    assert!(update.dense_keyframe_prediction.is_none());
    assert!(stream.next_is_keyframe());
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

fn temporal_input<'a>(
    model: &'a VJepa2_1Model<B>,
    context_tokens: Tensor<B, 3>,
    context_mask: &'a SparseTokenMask,
    target_mask: &'a SparseTokenMask,
    grid: TokenGridShape,
) -> TemporalSparsePredictorInput<'a, B> {
    TemporalSparsePredictorInput {
        config: model.config(),
        predictor: &model.predictor,
        context_tokens,
        context_mask,
        target_mask,
        grid,
        mask_index: 0,
    }
}
