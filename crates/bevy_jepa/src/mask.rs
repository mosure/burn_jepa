use anyhow::{Result, bail};
use burn::tensor::Tensor;
use burn_jepa::{
    FeatureFrameSparseMasks, SparseTokenMask, TokenGridShape, VJepaConfig, center_prior_mask,
    finalize_patch_diff_masks, patch_diff_can_use_dense_fast_path,
    patch_diff_context_mask_from_scores, patch_diff_context_mask_from_video,
    patch_diff_sampled_dense_fast_path_from_rgba, patch_diff_scores_from_rgba,
    patch_diff_sparsity_config,
};
use image::RgbaImage;

use crate::{BevyJepaConfig, BevyJepaMaskSource, JepaBevyBackend};

pub(super) type SparseMaskNodeOutput = FeatureFrameSparseMasks;

pub(super) fn run_sparse_mask_node(
    config: &BevyJepaConfig,
    prev_image: Option<&Tensor<JepaBevyBackend, 4>>,
    prev_rgba: Option<&RgbaImage>,
    rgba: Option<&RgbaImage>,
    image: &Tensor<JepaBevyBackend, 4>,
    model_config: &VJepaConfig,
    grid: TokenGridShape,
) -> Result<SparseMaskNodeOutput> {
    match config.mask_source {
        BevyJepaMaskSource::Autogaze => bail!(
            "AutoGaze mask source requires a loaded model-backed AutoGaze node; \
             this viewer will not synthesize AutoGaze masks. Use --mask-source patch-diff \
             or wire a real burn_autogaze pipeline into this graph."
        ),
        BevyJepaMaskSource::PatchDiff => patch_diff_mask(
            prev_image,
            prev_rgba,
            rgba,
            image,
            model_config,
            grid,
            config,
        ),
    }
}

fn patch_diff_mask(
    prev_image: Option<&Tensor<JepaBevyBackend, 4>>,
    prev_rgba: Option<&RgbaImage>,
    rgba: Option<&RgbaImage>,
    image: &Tensor<JepaBevyBackend, 4>,
    model_config: &VJepaConfig,
    grid: TokenGridShape,
    config: &BevyJepaConfig,
) -> Result<SparseMaskNodeOutput> {
    let Some(prev_image) = prev_image else {
        let mask = center_prior_mask(grid, config.bootstrap_context_tokens(grid.len()))?;
        return Ok(finalize_patch_diff_masks(mask, grid, config));
    };
    if let (Some(prev_rgba), Some(rgba)) = (prev_rgba, rgba) {
        return patch_diff_mask_from_rgba(prev_rgba, rgba, model_config, grid, config);
    }
    let video = Tensor::cat(
        vec![
            prev_image.clone().reshape([
                1,
                3,
                1,
                image.shape().dims::<4>()[2],
                image.shape().dims::<4>()[3],
            ]),
            image.clone().reshape([
                1,
                3,
                1,
                image.shape().dims::<4>()[2],
                image.shape().dims::<4>()[3],
            ]),
        ],
        2,
    );
    let sparsity = patch_diff_sparsity_config(config, grid);
    if patch_diff_can_use_dense_fast_path(&sparsity, grid) {
        return Ok(FeatureFrameSparseMasks::same(SparseTokenMask::all(
            grid.len(),
        )));
    }
    let mask = patch_diff_context_mask_from_video(&video, model_config, grid, &sparsity)?;
    Ok(finalize_patch_diff_masks(mask, grid, config))
}

fn patch_diff_mask_from_rgba(
    prev: &RgbaImage,
    current: &RgbaImage,
    model_config: &VJepaConfig,
    grid: TokenGridShape,
    config: &BevyJepaConfig,
) -> Result<SparseMaskNodeOutput> {
    anyhow::ensure!(
        grid.depth == 1,
        "RGBA patch-diff mask expects a single-frame token grid"
    );
    anyhow::ensure!(
        prev.dimensions() == current.dimensions(),
        "RGBA patch-diff frames must have matching dimensions"
    );
    let patch_size = model_config.patch_size.max(1);
    let height = current.height() as usize;
    let width = current.width() as usize;
    anyhow::ensure!(
        height == grid.height * patch_size && width == grid.width * patch_size,
        "RGBA patch-diff frame size must match the V-JEPA patch grid"
    );
    let prev = prev.as_raw();
    let current = current.as_raw();
    let sparsity = patch_diff_sparsity_config(config, grid);
    if patch_diff_can_use_dense_fast_path(&sparsity, grid) {
        return Ok(FeatureFrameSparseMasks::same(SparseTokenMask::all(
            grid.len(),
        )));
    }
    if patch_diff_sampled_dense_fast_path_from_rgba(
        prev,
        current,
        width,
        patch_size,
        grid,
        &sparsity,
        config.patch_diff_dense_fallback_density,
    ) {
        return Ok(FeatureFrameSparseMasks::same(SparseTokenMask::all(
            grid.len(),
        )));
    }
    let scores = patch_diff_scores_from_rgba(prev, current, width, patch_size, grid)?;
    let mask = patch_diff_context_mask_from_scores(scores, grid, &sparsity)?;
    Ok(finalize_patch_diff_masks(mask, grid, config))
}
