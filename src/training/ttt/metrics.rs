use super::step::{TttPatchifyKind, TttRolloutKind};
use crate::training::config::BurnJepaTrainConfig;
use crate::{
    SparseMaskBatch, TttMemoryUpdateSource, TttSupervisionMode, VJepaConfig,
    VJepaTttLayerProbeRecord, VJepaTttModel,
};
use anyhow::Result;
use burn::optim::GradientsParams;
use burn::tensor::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};

use crate::training::report::{
    TttLayerUtilizationMetric, TttMaskMetrics, TttMemoryMetrics, TttRolloutMetrics,
    TttRolloutReportMode, TttTargetSupervisionMetrics, TttUtilizationMetrics, tensor_scalar,
};

pub(super) fn mask_metrics_from_batches<B: Backend>(
    context: &SparseMaskBatch<B>,
    target: &SparseMaskBatch<B>,
) -> TttMaskMetrics {
    let context_rows = context.rows();
    let target_rows = target.rows();
    let context_lengths = context_rows.iter().map(Vec::len).collect::<Vec<_>>();
    let target_lengths = target_rows.iter().map(Vec::len).collect::<Vec<_>>();
    let context_stats = token_length_stats(&context_lengths);
    let target_stats = token_length_stats(&target_lengths);
    let dense_tokens = context.dense_len().max(1);
    TttMaskMetrics {
        context_tokens: context_stats.max,
        target_tokens: target_stats.max,
        context_min_tokens: context_stats.min,
        context_max_tokens: context_stats.max,
        context_mean_tokens: context_stats.mean,
        target_min_tokens: target_stats.min,
        target_max_tokens: target_stats.max,
        target_mean_tokens: target_stats.mean,
        dense_tokens,
        context_density: context_stats.mean / dense_tokens as f32,
        target_density: target_stats.mean / dense_tokens as f32,
    }
}

struct TokenLengthStats {
    min: usize,
    max: usize,
    mean: f32,
}

fn token_length_stats(lengths: &[usize]) -> TokenLengthStats {
    let min = lengths.iter().copied().min().unwrap_or(0);
    let max = lengths.iter().copied().max().unwrap_or(0);
    let mean = if lengths.is_empty() {
        0.0
    } else {
        lengths.iter().sum::<usize>() as f32 / lengths.len() as f32
    };
    TokenLengthStats { min, max, mean }
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
    let layers = config.ttt.resolved_layers(model);
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

pub(super) fn target_supervision_metrics(
    config: &BurnJepaTrainConfig,
) -> TttTargetSupervisionMetrics {
    let train_adapter_target = match config.ttt.memory_update {
        TttMemoryUpdateSource::SelfHidden => "self_hidden_detached",
        TttMemoryUpdateSource::TeacherForcedDiagnostic => "teacher_final_encoder_tokens",
    };
    let layer_alignment = match config.ttt.supervision {
        TttSupervisionMode::FinalTeacher => "final_teacher_loss",
        TttSupervisionMode::LayerLocalTeacher => "same_depth_layer_teacher_loss",
        TttSupervisionMode::Hybrid => "layer_local_pretrain_then_final_teacher_finetune",
    };
    TttTargetSupervisionMetrics {
        mode: config.ttt.target,
        memory_update: config.ttt.memory_update,
        supervision: config.ttt.supervision,
        hybrid_final_steps: config.ttt.hybrid_final_steps,
        train_adapter_target,
        deploy_adapter_target: "self_hidden_detached",
        layer_alignment,
        teacher_forced_eval: config.ttt.memory_update
            == TttMemoryUpdateSource::TeacherForcedDiagnostic,
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct TttLayerGradientRms {
    pub ttt_layer: usize,
    pub target_proj_grad_rms: Option<f64>,
    pub temporal_conv_grad_rms: Option<f64>,
    pub out_proj_grad_rms: Option<f64>,
}

#[derive(Default)]
struct UtilizationAccumulator {
    encoder_layer: usize,
    ttt_layer: usize,
    samples: usize,
    hidden_rms: f64,
    memory_read_rms: f64,
    adapter_delta_rms: f64,
    fast_weight_rms: f64,
    fast_update_rms: f64,
}

pub(super) fn ttt_utilization_metrics<B: Backend>(
    config: &BurnJepaTrainConfig,
    model: &VJepaTttModel<B>,
    probes: Vec<VJepaTttLayerProbeRecord<B>>,
    samples: usize,
) -> Result<TttUtilizationMetrics> {
    let layers = config.ttt.resolved_layers(model.config());
    let mut accumulators = layers
        .iter()
        .enumerate()
        .map(|(ttt_layer, &encoder_layer)| {
            let mut acc = UtilizationAccumulator::default();
            acc.encoder_layer = encoder_layer;
            acc.ttt_layer = ttt_layer;
            acc
        })
        .collect::<Vec<_>>();

    for record in probes {
        if let Some(acc) = accumulators.get_mut(record.ttt_layer) {
            let hidden_rms = tensor_rms(record.probe.hidden)?;
            let memory_read_rms = tensor_rms(record.probe.memory_read)?;
            let adapter_delta_rms = tensor_rms(record.probe.adapter_delta)?;
            let fast_weight_rms = tensor_rms(record.probe.fast_weight_after.clone())?;
            let fast_update_rms =
                tensor_rms(record.probe.fast_weight_after - record.probe.fast_weight_before)?;
            acc.encoder_layer = record.encoder_layer;
            acc.samples += samples.max(1);
            acc.hidden_rms += hidden_rms;
            acc.memory_read_rms += memory_read_rms;
            acc.adapter_delta_rms += adapter_delta_rms;
            acc.fast_weight_rms += fast_weight_rms;
            acc.fast_update_rms += fast_update_rms;
        }
    }

    let metrics = accumulators
        .into_iter()
        .map(|acc| {
            let divisor = acc.samples.max(1) as f64 / samples.max(1) as f64;
            let layer = &model.encoder.ttt_layers[acc.ttt_layer];
            let target_proj_param_rms = layer
                .target_proj
                .as_ref()
                .map(|proj| tensor_rms(proj.weight.val()))
                .transpose()?;
            let temporal_conv_param_rms = tensor_rms(layer.temporal_conv.weight.val())?;
            let out_proj_param_rms = tensor_rms(layer.out_proj.weight.val())?;
            let hidden_rms = acc.hidden_rms / divisor.max(1.0);
            let adapter_delta_rms = acc.adapter_delta_rms / divisor.max(1.0);
            Ok(TttLayerUtilizationMetric {
                encoder_layer: acc.encoder_layer,
                ttt_layer: acc.ttt_layer,
                samples: acc.samples,
                hidden_rms,
                memory_read_rms: acc.memory_read_rms / divisor.max(1.0),
                adapter_delta_rms,
                adapter_delta_to_hidden: adapter_delta_rms / hidden_rms.max(1.0e-12),
                fast_weight_rms: acc.fast_weight_rms / divisor.max(1.0),
                fast_update_rms: acc.fast_update_rms / divisor.max(1.0),
                target_proj_param_rms,
                temporal_conv_param_rms,
                out_proj_param_rms,
                target_proj_grad_rms: None,
                temporal_conv_grad_rms: None,
                out_proj_grad_rms: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(TttUtilizationMetrics {
        samples,
        layers: metrics,
    })
}

pub(super) fn ttt_gradient_metrics<B: AutodiffBackend>(
    config: &BurnJepaTrainConfig,
    model: &VJepaTttModel<B>,
    grads: &GradientsParams,
) -> Result<Vec<TttLayerGradientRms>> {
    let layers = config.ttt.resolved_layers(model.config());
    layers
        .iter()
        .enumerate()
        .map(|(ttt_layer, _)| {
            let layer = &model.encoder.ttt_layers[ttt_layer];
            let target_proj_grad_rms = layer
                .target_proj
                .as_ref()
                .and_then(|proj| grads.get::<B::InnerBackend, 2>(proj.weight.id))
                .map(tensor_rms)
                .transpose()?;
            let temporal_conv_grad_rms = grads
                .get::<B::InnerBackend, 3>(layer.temporal_conv.weight.id)
                .map(tensor_rms)
                .transpose()?;
            let out_proj_grad_rms = grads
                .get::<B::InnerBackend, 2>(layer.out_proj.weight.id)
                .map(tensor_rms)
                .transpose()?;
            Ok(TttLayerGradientRms {
                ttt_layer,
                target_proj_grad_rms,
                temporal_conv_grad_rms,
                out_proj_grad_rms,
            })
        })
        .collect()
}

pub(super) fn merge_gradient_metrics(
    utilization: &mut TttUtilizationMetrics,
    gradients: &[TttLayerGradientRms],
) {
    for layer in &mut utilization.layers {
        if let Some(grad) = gradients
            .iter()
            .find(|grad| grad.ttt_layer == layer.ttt_layer)
        {
            layer.target_proj_grad_rms = grad.target_proj_grad_rms;
            layer.temporal_conv_grad_rms = grad.temporal_conv_grad_rms;
            layer.out_proj_grad_rms = grad.out_proj_grad_rms;
        }
    }
}

fn tensor_rms<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Result<f64> {
    let scalar = tensor_scalar(tensor.powf_scalar(2.0).mean().detach())?;
    Ok(scalar.max(0.0).sqrt())
}
