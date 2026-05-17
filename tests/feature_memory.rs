use burn::tensor::{Int, Tensor, TensorData};
use burn_jepa::{
    InterframeJepaFeatureMemory, InterframeJepaFeatureMemoryConfig,
    InterframeJepaFeatureUpdateMode, SparseTokenMask, TokenGridShape, VJepaEncoderOutput,
};

type B = burn::backend::NdArray<f32>;

#[test]
fn sparse_updates_accumulate_dense_features_observed_mask_and_age() {
    let device = Default::default();
    let grid = TokenGridShape::new(1, 2, 2);
    let mut memory = InterframeJepaFeatureMemory::<B>::new(
        InterframeJepaFeatureMemoryConfig::default(),
        1,
        grid,
        2,
        &device,
    )
    .expect("feature memory");

    let first = memory
        .update_tokens(
            tensor3(&[10.0, 11.0, 20.0, 21.0], [1, 2, 2], &device),
            indices(&[0, 2], [1, 2], &device),
            grid,
        )
        .expect("first update");
    assert_eq!(first.step, 1);
    assert_eq!(first.updated_tokens, 2);
    assert_close(
        &values3(first.features),
        &[10.0, 11.0, 0.0, 0.0, 20.0, 21.0, 0.0, 0.0],
    );
    assert_close(&values2(first.observed), &[1.0, 0.0, 1.0, 0.0]);
    assert_close(&values2(first.age_frames), &[0.0, 0.0, 0.0, 0.0]);

    let second = memory
        .update_tokens(
            tensor3(&[30.0, 31.0], [1, 1, 2], &device),
            indices(&[1], [1, 1], &device),
            grid,
        )
        .expect("second update");
    assert_eq!(second.step, 2);
    assert_close(
        &values3(second.features),
        &[10.0, 11.0, 30.0, 31.0, 20.0, 21.0, 0.0, 0.0],
    );
    assert_close(&values2(second.observed), &[1.0, 1.0, 1.0, 0.0]);
    assert_close(&values2(second.age_frames), &[1.0, 0.0, 1.0, 0.0]);

    let third = memory
        .update_tokens(
            tensor3(&[99.0, 98.0], [1, 1, 2], &device),
            indices(&[0], [1, 1], &device),
            grid,
        )
        .expect("third update");
    assert_close(
        &values3(third.features),
        &[99.0, 98.0, 30.0, 31.0, 20.0, 21.0, 0.0, 0.0],
    );
    assert_close(&values2(third.age_frames), &[0.0, 1.0, 2.0, 0.0]);
}

#[test]
fn sparse_updates_write_by_token_index_not_sparse_position() {
    let device = Default::default();
    let grid = TokenGridShape::new(1, 3, 3);
    let mut memory = InterframeJepaFeatureMemory::<B>::new(
        InterframeJepaFeatureMemoryConfig::default(),
        1,
        grid,
        1,
        &device,
    )
    .expect("feature memory");

    let first = memory
        .update_tokens(
            tensor3(&[8.0, 0.0, 4.0], [1, 3, 1], &device),
            indices(&[8, 0, 4], [1, 3], &device),
            grid,
        )
        .expect("first sparse write");
    assert_close(
        &values3(first.features),
        &[0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 8.0],
    );
    assert_close(
        &values2(first.observed),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );

    let second = memory
        .update_tokens(
            tensor3(&[2.0, 18.0], [1, 2, 1], &device),
            indices(&[2, 8], [1, 2], &device),
            grid,
        )
        .expect("second sparse write");
    assert_close(
        &values3(second.features),
        &[0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 0.0, 18.0],
    );
    assert_close(
        &values2(second.observed),
        &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
    assert_close(
        &values2(second.age_frames),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    );
}

#[test]
fn sparse_updates_eventually_fill_spatial_cache_over_time() {
    let device = Default::default();
    let grid = TokenGridShape::new(1, 2, 2);
    let mut memory = InterframeJepaFeatureMemory::<B>::new(
        InterframeJepaFeatureMemoryConfig::default(),
        1,
        grid,
        1,
        &device,
    )
    .expect("feature memory");

    for token in 0..grid.len() {
        memory
            .update_tokens(
                tensor3(&[token as f32 + 1.0], [1, 1, 1], &device),
                indices(&[token as i64], [1, 1], &device),
                grid,
            )
            .expect("single-token sparse update");
    }

    let output = memory.snapshot();
    assert_close(&values2(output.observed), &[1.0, 1.0, 1.0, 1.0]);
    assert_close(&values3(output.features), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn ema_updates_assign_first_observation_then_blend_repeated_tokens() {
    let device = Default::default();
    let grid = TokenGridShape::new(1, 1, 2);
    let mut memory = InterframeJepaFeatureMemory::<B>::new(
        InterframeJepaFeatureMemoryConfig {
            update_mode: InterframeJepaFeatureUpdateMode::Ema { alpha: 0.25 },
            ..InterframeJepaFeatureMemoryConfig::default()
        },
        1,
        grid,
        1,
        &device,
    )
    .expect("feature memory");

    memory
        .update_tokens(
            tensor3(&[8.0], [1, 1, 1], &device),
            indices(&[0], [1, 1], &device),
            grid,
        )
        .expect("first update");
    let output = memory
        .update_tokens(
            tensor3(&[12.0], [1, 1, 1], &device),
            indices(&[0], [1, 1], &device),
            grid,
        )
        .expect("second update");

    assert_close(&values3(output.features), &[9.0, 0.0]);
}

#[test]
fn dense_ordered_update_matches_full_sparse_update_without_scatter() {
    let device = Default::default();
    let grid = TokenGridShape::new(1, 2, 2);
    let config = InterframeJepaFeatureMemoryConfig {
        update_mode: InterframeJepaFeatureUpdateMode::Ema { alpha: 0.25 },
        ..InterframeJepaFeatureMemoryConfig::default()
    };
    let mut dense_memory =
        InterframeJepaFeatureMemory::<B>::new(config, 1, grid, 1, &device).expect("dense memory");
    let mut sparse_memory =
        InterframeJepaFeatureMemory::<B>::new(config, 1, grid, 1, &device).expect("sparse memory");
    let all_indices = indices(&[0, 1, 2, 3], [1, 4], &device);

    for values in [&[1.0, 2.0, 3.0, 4.0][..], &[5.0, 6.0, 7.0, 8.0][..]] {
        let tokens = tensor3(values, [1, 4, 1], &device);
        let dense = dense_memory
            .update_dense_ordered_tokens(tokens.clone(), grid)
            .expect("dense ordered update");
        let sparse = sparse_memory
            .update_tokens(tokens, all_indices.clone(), grid)
            .expect("full sparse update");

        assert_eq!(dense.updated_tokens, sparse.updated_tokens);
        assert_close(&values3(dense.features), &values3(sparse.features));
        assert_close(&values2(dense.observed), &values2(sparse.observed));
        assert_close(&values2(dense.age_frames), &values2(sparse.age_frames));
    }
}

#[test]
fn high_density_sparse_update_preserves_unwritten_512_grid_tokens() {
    let device = Default::default();
    let grid = TokenGridShape::new(1, 32, 32);
    let embed_dim = 4;
    let dense_tokens = grid.len();
    let keep = (dense_tokens * 98).div_ceil(100);
    let mut memory = InterframeJepaFeatureMemory::<B>::new(
        InterframeJepaFeatureMemoryConfig::default(),
        1,
        grid,
        embed_dim,
        &device,
    )
    .expect("feature memory");

    let initial_values = token_values(dense_tokens, embed_dim, 1000.0);
    memory
        .update_dense_ordered_tokens(
            tensor3(&initial_values, [1, dense_tokens, embed_dim], &device),
            grid,
        )
        .expect("dense prime");

    let sparse_indices = (0..keep).map(|index| index as i64).collect::<Vec<_>>();
    let updated_values = token_values(keep, embed_dim, 2000.0);
    let output = memory
        .update_tokens(
            tensor3(&updated_values, [1, keep, embed_dim], &device),
            indices(&sparse_indices, [1, keep], &device),
            grid,
        )
        .expect("high-density sparse update");

    let features = values3(output.features);
    let observed = values2(output.observed);
    let age = values2(output.age_frames);
    assert_eq!(output.updated_tokens, keep);
    assert_close(&features[0..embed_dim], &updated_values[0..embed_dim]);
    assert_close(
        &features[(keep - 1) * embed_dim..keep * embed_dim],
        &updated_values[(keep - 1) * embed_dim..keep * embed_dim],
    );
    assert_close(
        &features[keep * embed_dim..(keep + 1) * embed_dim],
        &initial_values[keep * embed_dim..(keep + 1) * embed_dim],
    );
    assert!(observed.iter().all(|value| (*value - 1.0).abs() < 1.0e-5));
    assert_eq!(age[0], 0.0);
    assert_eq!(age[keep - 1], 0.0);
    assert_eq!(age[keep], 1.0);
}

#[test]
fn masked_and_encoder_output_conveniences_update_the_same_canvas() {
    let device = Default::default();
    let grid = TokenGridShape::new(1, 2, 2);
    let mut memory = InterframeJepaFeatureMemory::<B>::new(
        InterframeJepaFeatureMemoryConfig::default(),
        1,
        grid,
        2,
        &device,
    )
    .expect("feature memory");
    let mask = SparseTokenMask::new(vec![0, 3], grid.len()).expect("mask");

    memory
        .update_masked_tokens(
            tensor3(&[1.0, 2.0, 7.0, 8.0], [1, 2, 2], &device),
            &mask,
            grid,
        )
        .expect("masked update");
    let output = VJepaEncoderOutput {
        tokens: tensor3(&[5.0, 6.0], [1, 1, 2], &device),
        hierarchical: Vec::new(),
        captured_layers: Vec::new(),
        token_indices: indices(&[2], [1, 1], &device),
        grid,
    };
    let snapshot = memory
        .update_from_encoder_output(output)
        .expect("encoder output update");

    assert_close(
        &values3(snapshot.features),
        &[1.0, 2.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0],
    );
}

#[test]
fn reset_rows_and_reset_clear_memory_without_reallocation_readbacks() {
    let device = Default::default();
    let grid = TokenGridShape::new(1, 1, 3);
    let mut memory = InterframeJepaFeatureMemory::<B>::new(
        InterframeJepaFeatureMemoryConfig::default(),
        2,
        grid,
        1,
        &device,
    )
    .expect("feature memory");

    memory
        .update_tokens(
            tensor3(&[1.0, 2.0, 3.0, 4.0], [2, 2, 1], &device),
            indices(&[0, 1, 1, 2], [2, 2], &device),
            grid,
        )
        .expect("update");
    memory
        .reset_rows(Tensor::<B, 1, Int>::from_data(
            TensorData::new(vec![1i64], [1]),
            &device,
        ))
        .expect("device row reset");

    let snapshot = memory.snapshot();
    assert_close(&values3(snapshot.features), &[1.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
    assert_close(&values2(snapshot.observed), &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

    memory.reset_row_indices(&[0]).expect("host row reset");
    let snapshot = memory.snapshot();
    assert_close(&values3(snapshot.features), &[0.0; 6]);
    assert_close(&values2(snapshot.observed), &[0.0; 6]);

    memory.reset();
    assert_eq!(memory.step(), 0);
    assert_close(&values2(memory.snapshot().age_frames), &[0.0; 6]);
}

#[test]
fn update_rejects_incompatible_shapes() {
    let device = Default::default();
    let grid = TokenGridShape::new(1, 2, 2);
    let mut memory = InterframeJepaFeatureMemory::<B>::new(
        InterframeJepaFeatureMemoryConfig::default(),
        1,
        grid,
        2,
        &device,
    )
    .expect("feature memory");

    let err = memory
        .update_tokens(
            tensor3(&[1.0, 2.0], [1, 1, 2], &device),
            indices(&[0], [1, 1], &device),
            TokenGridShape::new(1, 1, 4),
        )
        .expect_err("grid mismatch should fail");
    assert!(
        err.to_string().contains("grid"),
        "unexpected error: {err:#}"
    );

    let err = memory
        .update_tokens(
            tensor3(&[1.0, 2.0], [1, 1, 2], &device),
            indices(&[0, 1], [1, 2], &device),
            grid,
        )
        .expect_err("index shape mismatch should fail");
    assert!(
        err.to_string().contains("index shape"),
        "unexpected error: {err:#}"
    );
}

#[test]
fn sparse_update_hot_path_has_no_host_readbacks_or_tensordata_construction() {
    let source = include_str!("../src/feature_memory.rs");
    let start = source.find("pub fn update_tokens").expect("update_tokens");
    let end = source[start..]
        .find("pub fn reset")
        .map(|offset| start + offset)
        .expect("reset");
    let hot_path = &source[start..end];

    for marker in [".to_data(", ".into_data(", "TensorData::new"] {
        assert!(
            !hot_path.contains(marker),
            "feature memory sparse update hot path should not contain {marker}"
        );
    }
    assert!(!hot_path.contains(".scatter_nd("));
    assert!(hot_path.contains(".scatter("));
    assert!(hot_path.contains("feature_indices"));
    assert!(hot_path.contains("observed_values"));
    assert!(hot_path.contains("age_values"));
    assert!(hot_path.contains("IndexingUpdateOp::Add"));
}

#[test]
fn dense_ordered_update_hot_path_avoids_sparse_gather_and_scatter() {
    let source = include_str!("../src/feature_memory.rs");
    let start = source
        .find("pub fn update_dense_ordered_tokens")
        .expect("update_dense_ordered_tokens");
    let end = source[start..]
        .find("pub fn reset")
        .map(|offset| start + offset)
        .expect("reset");
    let hot_path = &source[start..end];

    for marker in [
        ".to_data(",
        ".into_data(",
        "TensorData::new",
        ".gather(",
        ".scatter(",
        ".scatter_nd(",
        "Tensor::<B, 2>::ones",
        "Tensor::<B, 2>::zeros",
    ] {
        assert!(
            !hot_path.contains(marker),
            "dense ordered update hot path should not contain {marker}"
        );
    }
    assert!(hot_path.contains("dense_metadata_values"));
}

#[test]
fn tiled_sparse_assign_kernel_is_backend_gated_and_device_resident() {
    let source = include_str!("../src/sparse_feature_memory.rs");
    assert!(source.contains("copy_feature_memory_kernel"));
    assert!(source.contains("sparse_assign_feature_memory_kernel"));
    assert!(source.contains("sparse-feature-memory-wgpu"));
    assert!(source.contains("sparse-feature-memory-cuda"));
    assert!(source.contains("launch_unchecked::<R>"));
    for marker in [".to_data(", ".into_data(", "TensorData::new", ".scatter("] {
        assert!(
            !source.contains(marker),
            "backend sparse assign kernel should not use portable/host hot-path marker {marker}"
        );
    }
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

fn token_values(token_count: usize, embed_dim: usize, offset: f32) -> Vec<f32> {
    (0..token_count)
        .flat_map(|token| {
            (0..embed_dim).map(move |channel| offset + token as f32 * 10.0 + channel as f32)
        })
        .collect()
}

fn values3(tensor: Tensor<B, 3>) -> Vec<f32> {
    tensor.to_data().to_vec::<f32>().expect("tensor values")
}

fn values2(tensor: Tensor<B, 2>) -> Vec<f32> {
    tensor.to_data().to_vec::<f32>().expect("tensor values")
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "value count mismatch");
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual_value - expected_value).abs() < 1.0e-5,
            "value {index}: expected {expected_value}, got {actual_value}; full actual: {actual:?}"
        );
    }
}
