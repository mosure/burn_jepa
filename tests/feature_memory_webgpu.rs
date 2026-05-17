#![cfg(feature = "webgpu")]

use burn::tensor::{Int, Tensor, TensorData};
use burn_jepa::{
    InterframeJepaFeatureMemory, InterframeJepaFeatureMemoryConfig, SparseTokenMask, TokenGridShape,
};

type B = burn::backend::WebGpu<f32, i32>;

#[test]
fn dense_full_grid_update_overwrites_every_spatial_token_webgpu() {
    let device = Default::default();
    let grid = TokenGridShape::new(1, 2, 3);
    let mut memory = InterframeJepaFeatureMemory::<B>::new(
        InterframeJepaFeatureMemoryConfig::default(),
        1,
        grid,
        3,
        &device,
    )
    .expect("feature memory");
    let values = (0..grid.len())
        .flat_map(|token| {
            [
                token as f32 + 1.0,
                token as f32 + 11.0,
                token as f32 + 101.0,
            ]
        })
        .collect::<Vec<_>>();
    let tokens =
        Tensor::<B, 3>::from_data(TensorData::new(values.clone(), [1, grid.len(), 3]), &device);
    let mask = SparseTokenMask::all(grid.len());

    let output = memory
        .update_masked_tokens(tokens, &mask, grid)
        .expect("dense update");

    assert_eq!(output.updated_tokens, grid.len());
    assert_eq!(values3(output.features), values);
    assert_eq!(values2(output.observed), vec![1.0; grid.len()]);
    assert_eq!(values2(output.age_frames), vec![0.0; grid.len()]);
}

#[test]
fn sparse_updates_overwrite_only_selected_spatial_tokens_webgpu() {
    let device = Default::default();
    let grid = TokenGridShape::new(1, 2, 3);
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
            tensor3(&[1.0, 10.0, 3.0, 30.0], [1, 2, 2], &device),
            indices(&[0, 2], [1, 2], &device),
            grid,
        )
        .expect("first sparse update");
    assert_eq!(
        values3(first.features),
        vec![1.0, 10.0, 0.0, 0.0, 3.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );

    let second = memory
        .update_tokens(
            tensor3(&[4.0, 40.0, 6.0, 60.0], [1, 2, 2], &device),
            indices(&[2, 5], [1, 2], &device),
            grid,
        )
        .expect("second sparse update");
    assert_eq!(
        values3(second.features),
        vec![
            1.0, 10.0, 0.0, 0.0, 4.0, 40.0, 0.0, 0.0, 0.0, 0.0, 6.0, 60.0
        ]
    );
    assert_eq!(values2(second.observed), vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
}

#[cfg(feature = "sparse-feature-memory-wgpu")]
#[test]
fn tiled_sparse_assign_matches_portable_sparse_update_webgpu() {
    let device = Default::default();
    let grid = TokenGridShape::new(1, 2, 4);
    let mut portable = InterframeJepaFeatureMemory::<B>::new(
        InterframeJepaFeatureMemoryConfig::default(),
        1,
        grid,
        3,
        &device,
    )
    .expect("portable memory");
    let mut tiled = InterframeJepaFeatureMemory::<B>::new(
        InterframeJepaFeatureMemoryConfig::default(),
        1,
        grid,
        3,
        &device,
    )
    .expect("tiled memory");

    for (values, selected) in [
        (
            vec![1.0, 2.0, 3.0, 20.0, 21.0, 22.0, 70.0, 71.0, 72.0],
            vec![0, 2, 7],
        ),
        (
            vec![30.0, 31.0, 32.0, 40.0, 41.0, 42.0, 50.0, 51.0, 52.0],
            vec![2, 4, 5],
        ),
    ] {
        let tokens = tensor3(values.as_slice(), [1, selected.len(), 3], &device);
        let token_indices = indices(
            selected
                .iter()
                .copied()
                .map(i64::from)
                .collect::<Vec<_>>()
                .as_slice(),
            [1, selected.len()],
            &device,
        );
        portable
            .update_tokens(tokens.clone(), token_indices.clone(), grid)
            .expect("portable update");
        tiled
            .update_tokens_tiled_assign_wgpu(tokens, token_indices, grid)
            .expect("tiled update");
    }

    let portable = portable.snapshot();
    let tiled = tiled.snapshot();
    assert_eq!(values3(tiled.features), values3(portable.features));
    assert_eq!(values2(tiled.observed), values2(portable.observed));
    assert_eq!(values2(tiled.age_frames), values2(portable.age_frames));
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

fn values3(tensor: Tensor<B, 3>) -> Vec<f32> {
    tensor.to_data().to_vec::<f32>().expect("tensor values")
}

fn values2(tensor: Tensor<B, 2>) -> Vec<f32> {
    tensor.to_data().to_vec::<f32>().expect("tensor values")
}
