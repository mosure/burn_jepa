use anyhow::{Result, bail};
use burn::tensor::Tensor;
use burn_jepa::{
    FeatureFrameSparseMasks, PatchDiffRefreshState, SparseTokenMask, TokenGridShape, VJepaConfig,
    center_prior_mask, finalize_patch_diff_masks, patch_diff_can_use_dense_fast_path,
    patch_diff_context_mask_from_video, patch_diff_masks_from_scores,
    patch_diff_sampled_dense_fast_path_from_rgba, patch_diff_scores_from_rgba,
    patch_diff_sparsity_config,
};
use image::RgbaImage;

use crate::{BevyJepaConfig, BevyJepaMaskSource, JepaBevyBackend};

pub(super) type SparseMaskNodeOutput = FeatureFrameSparseMasks;

#[cfg(test)]
pub(super) fn run_sparse_mask_node(
    config: &BevyJepaConfig,
    prev_image: Option<&Tensor<JepaBevyBackend, 4>>,
    prev_rgba: Option<&RgbaImage>,
    rgba: Option<&RgbaImage>,
    image: &Tensor<JepaBevyBackend, 4>,
    model_config: &VJepaConfig,
    grid: TokenGridShape,
) -> Result<SparseMaskNodeOutput> {
    run_sparse_mask_node_with_refresh_state(
        config,
        prev_image,
        prev_rgba,
        rgba,
        image,
        model_config,
        grid,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_sparse_mask_node_with_refresh_state(
    config: &BevyJepaConfig,
    prev_image: Option<&Tensor<JepaBevyBackend, 4>>,
    prev_rgba: Option<&RgbaImage>,
    rgba: Option<&RgbaImage>,
    image: &Tensor<JepaBevyBackend, 4>,
    model_config: &VJepaConfig,
    grid: TokenGridShape,
    refresh_state: Option<&mut PatchDiffRefreshState>,
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
            refresh_state,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn patch_diff_mask(
    prev_image: Option<&Tensor<JepaBevyBackend, 4>>,
    prev_rgba: Option<&RgbaImage>,
    rgba: Option<&RgbaImage>,
    image: &Tensor<JepaBevyBackend, 4>,
    model_config: &VJepaConfig,
    grid: TokenGridShape,
    config: &BevyJepaConfig,
    refresh_state: Option<&mut PatchDiffRefreshState>,
) -> Result<SparseMaskNodeOutput> {
    let Some(prev_image) = prev_image else {
        let mask = center_prior_mask(grid, config.bootstrap_context_tokens(grid.len()))?;
        let masks = finalize_patch_diff_masks(mask, grid, config);
        observe_patch_diff_write(refresh_state, &masks, grid);
        return Ok(masks);
    };
    if let (Some(prev_rgba), Some(rgba)) = (prev_rgba, rgba) {
        return patch_diff_mask_from_rgba(
            prev_rgba,
            rgba,
            model_config,
            grid,
            config,
            refresh_state,
        );
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
        let masks = FeatureFrameSparseMasks::same(SparseTokenMask::all(grid.len()));
        observe_patch_diff_write(refresh_state, &masks, grid);
        return Ok(masks);
    }
    let mask = patch_diff_context_mask_from_video(&video, model_config, grid, &sparsity)?;
    let masks = finalize_patch_diff_masks(mask, grid, config);
    observe_patch_diff_write(refresh_state, &masks, grid);
    Ok(masks)
}

fn patch_diff_mask_from_rgba(
    prev: &RgbaImage,
    current: &RgbaImage,
    model_config: &VJepaConfig,
    grid: TokenGridShape,
    config: &BevyJepaConfig,
    refresh_state: Option<&mut PatchDiffRefreshState>,
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
        let masks = FeatureFrameSparseMasks::same(SparseTokenMask::all(grid.len()));
        observe_patch_diff_write(refresh_state, &masks, grid);
        return Ok(masks);
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
        let masks = FeatureFrameSparseMasks::same(SparseTokenMask::all(grid.len()));
        observe_patch_diff_write(refresh_state, &masks, grid);
        return Ok(masks);
    }
    let scores = patch_diff_scores_from_rgba(prev, current, width, patch_size, grid)?;
    if let Some(refresh_state) = refresh_state {
        refresh_state.masks_from_scores(scores, grid, &sparsity, config)
    } else {
        patch_diff_masks_from_scores(scores, grid, &sparsity, config)
    }
}

fn observe_patch_diff_write(
    refresh_state: Option<&mut PatchDiffRefreshState>,
    masks: &FeatureFrameSparseMasks,
    grid: TokenGridShape,
) {
    if let Some(refresh_state) = refresh_state {
        refresh_state.observe_write_mask(&masks.write_mask, grid);
    }
}
