#![cfg(feature = "sparse-patchify-cuda")]

use burn::tensor::Tensor;
use burn::tensor::backend::BackendTypes;
use burn_jepa::{
    SparseImageTokenGrid, SparsePatchifyPlan, SparseTokenMask, TemporalSparseJepaStream,
    TemporalSparseJepaStreamConfig, TttEncoderConfig, VJepa2_1Model, VJepaConfig, VJepaTttModel,
    apply_token_mask,
};
use std::sync::{Mutex, OnceLock};

type B = burn_flex_gmm::cuda::DefaultCudaBackend;
const CUDA_PARITY_TOLERANCE: f32 = 2.0e-3;

fn cuda_sparse_patchify_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn cuda_tiny_config() -> VJepaConfig {
    let mut config = VJepaConfig::tiny_for_tests();
    config.encoder.depth = 1;
    config.predictor.depth = 1;
    config
}

fn cuda_sparse_patchify_smoke_enabled() -> bool {
    if std::env::var("BURN_JEPA_RUN_CUDA_SPARSE_PATCHIFY")
        .ok()
        .as_deref()
        == Some("1")
    {
        #[cfg(feature = "cuda")]
        burn_jepa::runtime::cuda_runtime_preflight(burn_jepa::runtime::CUDA_TRAIN_FORCE_ENV)
            .expect("CUDA sparse patchify preflight");
        true
    } else {
        eprintln!("skipping CUDA sparse patchify smoke; set BURN_JEPA_RUN_CUDA_SPARSE_PATCHIFY=1");
        false
    }
}

#[test]
fn cuda_sparse_patchify_matches_dense_encoder_on_selected_tokens() {
    if !cuda_sparse_patchify_smoke_enabled() {
        return;
    }
    let _guard = cuda_sparse_patchify_test_lock()
        .lock()
        .expect("CUDA sparse patchify test lock");

    let device = <B as BackendTypes>::Device::default();
    let config = cuda_tiny_config();
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
        .encode_video_sparse_patchify_cuda(video, &plan)
        .expect("sparse patchify encode")
        .tokens
        .to_data();
    assert_close(&dense, &sparse, "cuda sparse patchify encoder");
}

#[test]
fn cuda_ttt_sparse_image_patchify_matches_dense_patch_embed_on_selected_tokens() {
    if !cuda_sparse_patchify_smoke_enabled() {
        return;
    }
    let _guard = cuda_sparse_patchify_test_lock()
        .lock()
        .expect("CUDA sparse patchify test lock");

    let device = <B as BackendTypes>::Device::default();
    let config = cuda_tiny_config();
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
        .sparse_patchify_image_cuda(image, &plan)
        .expect("sparse image patchify")
        .to_data();
    assert_close(&dense, &sparse, "cuda TTT sparse image patchify");
}

#[test]
fn cuda_temporal_stream_sparse_patchify_matches_dense_masked_stream() {
    if !cuda_sparse_patchify_smoke_enabled() {
        return;
    }
    let _guard = cuda_sparse_patchify_test_lock()
        .lock()
        .expect("CUDA sparse patchify test lock");

    let device = <B as BackendTypes>::Device::default();
    let config = cuda_tiny_config();
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
        .forward_frame_tokens_sparse_patchify_cuda(&model, video.clone(), &frame_tokens, 0)
        .expect("sparse patchify stream");
    let sparse_reused = sparse_stream
        .forward_frame_tokens_sparse_patchify_cuda(&model, video, &frame_tokens, 0)
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

fn assert_close(left: &burn::tensor::TensorData, right: &burn::tensor::TensorData, label: &str) {
    let left = left.as_slice::<f32>().expect("left f32");
    let right = right.as_slice::<f32>().expect("right f32");
    assert_eq!(left.len(), right.len(), "{label} lengths differ");
    for (idx, (lhs, rhs)) in left.iter().zip(right).enumerate() {
        let diff = (lhs - rhs).abs();
        assert!(
            diff <= CUDA_PARITY_TOLERANCE,
            "{label} drift at {idx}: dense={lhs} sparse={rhs} diff={diff}"
        );
    }
}
