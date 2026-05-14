use super::step::{TttPatchifyKind, TttRolloutKind};
use crate::{SparseTokenMask, VJepaConfig, training::config::BurnJepaTrainConfig};

use crate::training::report::{
    TttMaskMetrics, TttMemoryMetrics, TttRolloutMetrics, TttRolloutReportMode,
};

pub(super) fn mask_metrics_from_masks(
    context: &SparseTokenMask,
    target: &SparseTokenMask,
) -> TttMaskMetrics {
    let dense_tokens = context.dense_len().max(1);
    TttMaskMetrics {
        context_tokens: context.len(),
        target_tokens: target.len(),
        dense_tokens,
        context_density: context.len() as f32 / dense_tokens as f32,
        target_density: target.len() as f32 / dense_tokens as f32,
    }
}

pub(super) fn ttt_memory_metrics(
    config: &BurnJepaTrainConfig,
    model: &VJepaConfig,
) -> TttMemoryMetrics {
    ttt_memory_metrics_for_batch_size(config, model, config.training.batch_size)
}

pub(super) fn ttt_memory_metrics_for_batch_size(
    config: &BurnJepaTrainConfig,
    model: &VJepaConfig,
    batch_size: usize,
) -> TttMemoryMetrics {
    let mut layers = config.ttt.layers.clone();
    layers.sort_unstable();
    layers.dedup();
    let embed_dim = model.encoder.embed_dim.max(1);
    let layer_count = layers.len();
    let batch_size = batch_size.max(1);
    let fast_weight_elements = layer_count * batch_size * embed_dim * embed_dim;
    let per_layer_params = embed_dim * embed_dim
        + embed_dim * config.ttt.conv_kernel.max(1)
        + usize::from(config.ttt.use_projection) * embed_dim * embed_dim;
    let trainable_param_elements = per_layer_params * layer_count;
    TttMemoryMetrics {
        layers,
        embed_dim,
        batch_size,
        chunk_tokens: config.ttt.chunk_tokens.max(1),
        ttt_lr: config.ttt.ttt_lr,
        fast_weight_elements,
        fast_weight_bytes_f32: fast_weight_elements * std::mem::size_of::<f32>(),
        trainable_param_elements,
        trainable_param_bytes_f32: trainable_param_elements * std::mem::size_of::<f32>(),
        adam_state_bytes_f32: trainable_param_elements * std::mem::size_of::<f32>() * 2,
    }
}

pub(super) fn rollout_metrics(
    model: &VJepaConfig,
    rollout: TttRolloutKind,
    mask: Option<&TttMaskMetrics>,
    full_grid_eval: bool,
    observed_dense_tokens: Option<usize>,
    patchify: TttPatchifyKind,
) -> TttRolloutMetrics {
    let dense_tokens = mask
        .map(|mask| mask.dense_tokens)
        .or(observed_dense_tokens)
        .unwrap_or_else(|| model.num_patches())
        .max(1);
    let student_tokens = match rollout {
        TttRolloutKind::Dense => dense_tokens,
        TttRolloutKind::SparseContext => {
            mask.map(|mask| mask.context_tokens).unwrap_or(dense_tokens)
        }
        TttRolloutKind::SparseTarget => mask.map(|mask| mask.target_tokens).unwrap_or(dense_tokens),
    };
    TttRolloutMetrics {
        mode: match rollout {
            TttRolloutKind::Dense => TttRolloutReportMode::Dense,
            TttRolloutKind::SparseContext => TttRolloutReportMode::SparseContext,
            TttRolloutKind::SparseTarget => TttRolloutReportMode::SparseTarget,
        },
        dense_tokens,
        student_tokens,
        student_token_density: student_tokens as f32 / dense_tokens as f32,
        full_grid_eval,
        autodiff_sparse_patchify: patchify == TttPatchifyKind::FrozenSparsePatchify,
    }
}
