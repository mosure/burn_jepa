#![cfg(feature = "sparse-patchify-wgpu")]

use burn::tensor::Tensor;
use burn::tensor::backend::BackendTypes;
use burn_jepa::{SparsePatchifyPlan, SparseTokenMask, VJepa2_1Model, VJepaConfig};

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
