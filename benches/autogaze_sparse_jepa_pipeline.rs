use std::{
    collections::BTreeSet,
    env, fs,
    hint::black_box,
    panic::{self, AssertUnwindSafe},
    path::PathBuf,
    time::{Duration, Instant},
};

#[cfg(feature = "webgpu")]
use std::sync::OnceLock;

use burn::{
    module::{Module, ModuleMapper, Param},
    tensor::{
        Tensor, TensorData,
        backend::{Backend, BackendTypes},
    },
};
use burn_autogaze::{
    AutoGazeConfig, AutoGazeInferenceMode, AutoGazePipeline, ConnectorConfig, FixationBounds,
    FrameFixationTrace, GazeDecoderConfig, GazeModelConfig, NativeAutoGazeModel, VisionModelConfig,
};
use burn_jepa::{
    SparsePatchRect, SparsePatchifyPlan, SparsePredictorPlan, SparseTokenMask, TokenGridShape,
    VJepa2_1Model, VJepaConfig, VJepaEncoderConfig, VJepaModelVariant, VJepaPredictorConfig,
};

type JepaBackend = burn_flex_gmm::wgpu::DefaultWgpuBackend;

const BATCH: usize = 1;
const CHANNELS: usize = 3;
const FRAMES: usize = 4;
const PATCH_SIZE: usize = 16;
const TUBELET_SIZE: usize = 2;
const AUTOGAZE_MODEL_INPUT: usize = 224;
const AUTOGAZE_CONNECTOR_TOKENS: usize = 196;
const TARGET_TOKENS: usize = 64;
const DENSITIES: &[f32] = &[0.01, 0.05, 0.10, 0.25];
const RESOLUTIONS: &[Resolution] = &[
    Resolution {
        name: "224x224",
        width: 224,
        height: 224,
    },
    Resolution {
        name: "384x384",
        width: 384,
        height: 384,
    },
    Resolution {
        name: "720p",
        width: 1280,
        height: 720,
    },
    Resolution {
        name: "1080p16pad",
        width: 1920,
        height: 1088,
    },
];

#[derive(Clone, Copy, Debug)]
struct Resolution {
    name: &'static str,
    width: usize,
    height: usize,
}

#[derive(Clone, Debug)]
struct BenchRow {
    autogaze_backend: &'static str,
    jepa_backend: &'static str,
    resolution: &'static str,
    width: usize,
    height: usize,
    frames: usize,
    dense_tokens: usize,
    target_density: f32,
    actual_density: f32,
    autogaze_top_k: usize,
    context_tokens: usize,
    target_tokens: usize,
    autogaze_trace_ms: f64,
    sparse_project_plan_ms: f64,
    dense_patchify_ms: f64,
    sparse_patchify_ms: f64,
    sparse_encoder_ms: f64,
    predictor_ms: f64,
    sparse_jepa_ms: f64,
    e2e_pipeline_ms: f64,
    clips_per_sec: f64,
    frames_per_sec: f64,
}

fn main() {
    let reps = env_usize("BURN_JEPA_PIPELINE_BENCH_REPS", 3);
    let warmups = env_usize("BURN_JEPA_PIPELINE_BENCH_WARMUPS", 1);
    let include_1080 = env_bool("BURN_JEPA_PIPELINE_BENCH_1080P", true);
    let backend_filter = backend_filter();
    let resolutions = RESOLUTIONS
        .iter()
        .copied()
        .filter(|case| include_1080 || case.name != "1080p16pad")
        .collect::<Vec<_>>();
    let mut rows = Vec::new();

    println!("{}", BenchRow::csv_header());

    if backend_enabled(&backend_filter, "ndarray") {
        append_backend_rows(
            &mut rows,
            run_optional("autogaze-ndarray", || {
                run_autogaze_backend::<burn::backend::NdArray<f32>>(
                    "autogaze-ndarray",
                    <burn::backend::NdArray<f32> as BackendTypes>::Device::default(),
                    &resolutions,
                    warmups,
                    reps,
                )
            }),
        );
    }

    #[cfg(feature = "webgpu")]
    if backend_enabled(&backend_filter, "webgpu")
        && let Some(device) = webgpu_device()
    {
        append_backend_rows(
            &mut rows,
            run_optional("autogaze-webgpu", || {
                run_autogaze_backend::<burn::backend::WebGpu<f32, i32>>(
                    "autogaze-webgpu",
                    device,
                    &resolutions,
                    warmups,
                    reps,
                )
            }),
        );
    }

    #[cfg(feature = "cuda")]
    if backend_enabled(&backend_filter, "cuda") {
        append_backend_rows(
            &mut rows,
            run_optional("autogaze-cuda", || {
                run_autogaze_backend::<burn::backend::Cuda<f32, i32>>(
                    "autogaze-cuda",
                    burn::backend::cuda::CudaDevice::default(),
                    &resolutions,
                    warmups,
                    reps,
                )
            }),
        );
    }

    let out = bench_output_path();
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).expect("create benchmark output directory");
    }
    let csv = csv_string(&rows);
    fs::write(&out, csv).expect("write benchmark CSV");
    eprintln!("wrote {}", out.display());
}

fn append_backend_rows(rows: &mut Vec<BenchRow>, backend_rows: Option<Vec<BenchRow>>) {
    if let Some(mut backend_rows) = backend_rows {
        rows.append(&mut backend_rows);
    }
}

fn run_autogaze_backend<B>(
    autogaze_backend: &'static str,
    autogaze_device: B::Device,
    resolutions: &[Resolution],
    warmups: usize,
    reps: usize,
) -> Vec<BenchRow>
where
    B: Backend,
    B::Device: Clone,
{
    let autogaze = deterministic_autogaze_pipeline::<B>(&autogaze_device);
    let jepa_device = <JepaBackend as BackendTypes>::Device::default();
    let mut rows = Vec::new();

    for &resolution in resolutions {
        let grid = TokenGridShape::new(
            FRAMES / TUBELET_SIZE,
            resolution.height / PATCH_SIZE,
            resolution.width / PATCH_SIZE,
        );
        let jepa_config = jepa_config();
        let jepa = VJepa2_1Model::<JepaBackend>::new(&jepa_config, &jepa_device);
        let ag_values = deterministic_autogaze_values(resolution);
        let jepa_values = deterministic_jepa_values(resolution);
        let ag_video = Tensor::<B, 5>::from_data(
            TensorData::new(
                ag_values,
                [BATCH, FRAMES, CHANNELS, resolution.height, resolution.width],
            ),
            &autogaze_device,
        );
        let jepa_video = Tensor::<JepaBackend, 5>::from_data(
            TensorData::new(
                jepa_values,
                [BATCH, CHANNELS, FRAMES, resolution.height, resolution.width],
            ),
            &jepa_device,
        );
        <JepaBackend as Backend>::sync(&jepa_device).expect("jepa input upload sync");

        let dense_patchify_ms = measure_ms(warmups, reps, || {
            let tokens = jepa.encoder.patch_embed.forward(jepa_video.clone());
            let dims = tokens.shape().dims::<3>();
            black_box(dims);
            <JepaBackend as Backend>::sync(&jepa_device).expect("dense patchify sync");
            dims[1]
        });

        for &target_density in DENSITIES {
            let dense_tokens = grid.len();
            let target_context_tokens = ((dense_tokens as f32) * target_density).ceil() as usize;
            let target_context_tokens = target_context_tokens.max(1).min(dense_tokens);
            let autogaze_top_k = density_top_k(grid, target_density);

            let traces = autogaze.trace_video_with_mode(
                ag_video.clone(),
                autogaze_top_k,
                AutoGazeInferenceMode::ResizeToModelInput,
            );
            let context_mask = context_mask_from_autogaze(&traces, grid, target_context_tokens)
                .expect("autogaze sparse context mask");
            let target_mask =
                target_mask_for_context(&context_mask, TARGET_TOKENS.min(dense_tokens));
            let context_plan = SparsePatchifyPlan::<JepaBackend>::new(
                context_mask.clone(),
                grid,
                BATCH,
                &jepa_device,
            )
            .expect("context patchify plan");
            let predictor_plan = SparsePredictorPlan::<JepaBackend>::new(
                &jepa_config,
                context_mask.clone(),
                target_mask.clone(),
                grid,
                BATCH,
                &jepa_device,
            )
            .expect("predictor plan");
            <B as Backend>::sync(&autogaze_device).expect("autogaze sync");
            <JepaBackend as Backend>::sync(&jepa_device).expect("plan sync");

            let autogaze_trace_ms = measure_ms(warmups, reps, || {
                let traces = autogaze.trace_video_with_mode(
                    ag_video.clone(),
                    autogaze_top_k,
                    AutoGazeInferenceMode::ResizeToModelInput,
                );
                black_box(trace_point_count(&traces));
                <B as Backend>::sync(&autogaze_device).expect("autogaze trace sync");
                traces.len()
            });
            let sparse_project_plan_ms = measure_ms(warmups, reps, || {
                let context_mask = context_mask_from_autogaze(&traces, grid, target_context_tokens)
                    .expect("project context mask");
                let target_mask =
                    target_mask_for_context(&context_mask, TARGET_TOKENS.min(dense_tokens));
                let context_plan = SparsePatchifyPlan::<JepaBackend>::new(
                    context_mask.clone(),
                    grid,
                    BATCH,
                    &jepa_device,
                )
                .expect("context plan");
                let predictor_plan = SparsePredictorPlan::<JepaBackend>::new(
                    &jepa_config,
                    context_mask,
                    target_mask,
                    grid,
                    BATCH,
                    &jepa_device,
                )
                .expect("predictor plan");
                black_box(context_plan.output_rows());
                black_box(predictor_plan.sequence_indices.shape());
                <JepaBackend as Backend>::sync(&jepa_device).expect("project plan sync");
                context_plan.output_rows()
            });
            let sparse_patchify_ms = measure_ms(warmups, reps, || {
                let tokens =
                    sparse_patchify_tokens(&jepa, &jepa_config, jepa_video.clone(), &context_plan)
                        .expect("sparse patchify tokens");
                let dims = tokens.shape().dims::<3>();
                black_box(dims);
                <JepaBackend as Backend>::sync(&jepa_device).expect("sparse patchify sync");
                dims[1]
            });
            let context_tokens =
                sparse_patchify_tokens(&jepa, &jepa_config, jepa_video.clone(), &context_plan)
                    .expect("precompute sparse tokens");
            <JepaBackend as Backend>::sync(&jepa_device).expect("precompute sparse tokens sync");

            let sparse_encoder_ms = measure_ms(warmups, reps, || {
                let encoded = jepa.encoder.forward_sparse_tokens(
                    context_tokens.clone(),
                    BATCH,
                    grid,
                    context_mask.indices(),
                    true,
                );
                let dims = encoded.tokens.shape().dims::<3>();
                black_box(dims);
                <JepaBackend as Backend>::sync(&jepa_device).expect("sparse encoder sync");
                dims[1]
            });
            let encoded_context = jepa.encoder.forward_sparse_tokens(
                context_tokens,
                BATCH,
                grid,
                context_mask.indices(),
                true,
            );
            <JepaBackend as Backend>::sync(&jepa_device).expect("precompute encoded context sync");

            let predictor_ms = measure_ms(warmups, reps, || {
                let output = jepa
                    .predictor
                    .forward_sparse_with_plan(encoded_context.tokens.clone(), &predictor_plan, 0)
                    .expect("predictor");
                let dims = output.target_predictions.shape().dims::<3>();
                black_box(dims);
                <JepaBackend as Backend>::sync(&jepa_device).expect("predictor sync");
                dims[1]
            });
            let sparse_jepa_ms = measure_ms(warmups, reps, || {
                run_sparse_jepa_once(
                    &jepa,
                    &jepa_config,
                    jepa_video.clone(),
                    &context_plan,
                    &predictor_plan,
                )
                .expect("sparse jepa");
                <JepaBackend as Backend>::sync(&jepa_device).expect("sparse jepa sync");
            });
            let e2e_pipeline_ms = measure_ms(warmups, reps, || {
                let traces = autogaze.trace_video_with_mode(
                    ag_video.clone(),
                    autogaze_top_k,
                    AutoGazeInferenceMode::ResizeToModelInput,
                );
                let context_mask = context_mask_from_autogaze(&traces, grid, target_context_tokens)
                    .expect("e2e context mask");
                let target_mask =
                    target_mask_for_context(&context_mask, TARGET_TOKENS.min(dense_tokens));
                let context_plan = SparsePatchifyPlan::<JepaBackend>::new(
                    context_mask.clone(),
                    grid,
                    BATCH,
                    &jepa_device,
                )
                .expect("e2e context plan");
                let predictor_plan = SparsePredictorPlan::<JepaBackend>::new(
                    &jepa_config,
                    context_mask,
                    target_mask,
                    grid,
                    BATCH,
                    &jepa_device,
                )
                .expect("e2e predictor plan");
                run_sparse_jepa_once(
                    &jepa,
                    &jepa_config,
                    jepa_video.clone(),
                    &context_plan,
                    &predictor_plan,
                )
                .expect("e2e sparse jepa");
                <B as Backend>::sync(&autogaze_device).expect("e2e autogaze sync");
                <JepaBackend as Backend>::sync(&jepa_device).expect("e2e jepa sync");
            });
            let clips_per_sec = 1000.0 / e2e_pipeline_ms.max(f64::EPSILON);
            let frames_per_sec = clips_per_sec * FRAMES as f64;

            let row = BenchRow {
                autogaze_backend,
                jepa_backend: "sparse-patchify-wgpu",
                resolution: resolution.name,
                width: resolution.width,
                height: resolution.height,
                frames: FRAMES,
                dense_tokens,
                target_density,
                actual_density: context_mask.len() as f32 / dense_tokens as f32,
                autogaze_top_k,
                context_tokens: context_mask.len(),
                target_tokens: target_mask.len(),
                autogaze_trace_ms,
                sparse_project_plan_ms,
                dense_patchify_ms,
                sparse_patchify_ms,
                sparse_encoder_ms,
                predictor_ms,
                sparse_jepa_ms,
                e2e_pipeline_ms,
                clips_per_sec,
                frames_per_sec,
            };
            println!("{}", row.to_csv());
            rows.push(row);
        }
    }

    rows
}

fn run_sparse_jepa_once(
    jepa: &VJepa2_1Model<JepaBackend>,
    config: &VJepaConfig,
    video: Tensor<JepaBackend, 5>,
    context_plan: &SparsePatchifyPlan<JepaBackend>,
    predictor_plan: &SparsePredictorPlan<JepaBackend>,
) -> anyhow::Result<()> {
    let tokens = sparse_patchify_tokens(jepa, config, video, context_plan)?;
    let encoded = jepa.encoder.forward_sparse_tokens(
        tokens,
        BATCH,
        context_plan.grid,
        context_plan.mask.indices(),
        true,
    );
    let output = jepa
        .predictor
        .forward_sparse_with_plan(encoded.tokens, predictor_plan, 0)?;
    black_box(output.target_predictions.shape().dims::<3>());
    Ok(())
}

fn measure_ms<T>(warmups: usize, reps: usize, mut f: impl FnMut() -> T) -> f64 {
    for _ in 0..warmups {
        black_box(f());
    }
    let mut samples = Vec::with_capacity(reps.max(1));
    for _ in 0..reps.max(1) {
        let start = Instant::now();
        black_box(f());
        samples.push(start.elapsed());
    }
    median_ms(&mut samples)
}

fn median_ms(samples: &mut [Duration]) -> f64 {
    samples.sort_unstable();
    let mid = samples.len() / 2;
    samples[mid].as_secs_f64() * 1000.0
}

fn density_top_k(grid: TokenGridShape, density: f32) -> usize {
    let per_tubelet = grid.tokens_per_frame().max(1);
    ((per_tubelet as f32) * density).ceil().clamp(1.0, 32.0) as usize
}

fn context_mask_from_autogaze(
    traces: &[FrameFixationTrace],
    grid: TokenGridShape,
    min_keep_tokens: usize,
) -> anyhow::Result<SparseTokenMask> {
    let target = min_keep_tokens.max(1).min(grid.len());
    let rects = trace_rects(traces, FRAMES);
    let mut selected = Vec::with_capacity(target);
    let mut keep = vec![false; grid.len()];

    for tubelet in 0..grid.depth {
        let start = tubelet * TUBELET_SIZE;
        let end = ((tubelet + 1) * TUBELET_SIZE).min(rects.len());
        for frame_rects in &rects[start..end] {
            for rect in frame_rects {
                push_rect_tokens(*rect, tubelet, grid, target, &mut keep, &mut selected);
                if selected.len() >= target {
                    return SparseTokenMask::new(selected, grid.len());
                }
            }
        }
    }

    for index in SparseTokenMask::evenly_spaced(grid.len(), target)
        .indices()
        .iter()
        .copied()
    {
        push_sparse_index(index, target, &mut keep, &mut selected);
        if selected.len() >= target {
            return SparseTokenMask::new(selected, grid.len());
        }
    }
    for index in 0..grid.len() {
        push_sparse_index(index, target, &mut keep, &mut selected);
        if selected.len() >= target {
            break;
        }
    }
    SparseTokenMask::new(selected, grid.len())
}

fn trace_rects(traces: &[FrameFixationTrace], frames: usize) -> Vec<Vec<SparsePatchRect>> {
    let trace = traces.first();
    (0..frames)
        .map(|frame_idx| {
            trace
                .and_then(|trace| trace.frames.get(frame_idx))
                .map(|set| {
                    set.points
                        .iter()
                        .map(|point| bounds_to_rect(point.bounds()))
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect()
}

fn bounds_to_rect(bounds: FixationBounds) -> SparsePatchRect {
    SparsePatchRect::new(bounds.x_min, bounds.y_min, bounds.x_max, bounds.y_max)
}

fn push_rect_tokens(
    rect: SparsePatchRect,
    tubelet: usize,
    grid: TokenGridShape,
    target: usize,
    keep: &mut [bool],
    selected: &mut Vec<usize>,
) {
    let Some((row_start, row_end, col_start, col_end)) = rect_patch_bounds(rect, grid) else {
        return;
    };
    for row in row_start..=row_end {
        for col in col_start..=col_end {
            let index = tubelet * grid.tokens_per_frame() + row * grid.width + col;
            push_sparse_index(index, target, keep, selected);
            if selected.len() >= target {
                return;
            }
        }
    }
}

fn push_sparse_index(index: usize, target: usize, keep: &mut [bool], selected: &mut Vec<usize>) {
    if selected.len() >= target || index >= keep.len() || keep[index] {
        return;
    }
    keep[index] = true;
    selected.push(index);
}

fn rect_patch_bounds(
    rect: SparsePatchRect,
    grid: TokenGridShape,
) -> Option<(usize, usize, usize, usize)> {
    let x0 = rect.x0.min(rect.x1).clamp(0.0, 1.0);
    let y0 = rect.y0.min(rect.y1).clamp(0.0, 1.0);
    let x1 = rect.x0.max(rect.x1).clamp(0.0, 1.0);
    let y1 = rect.y0.max(rect.y1).clamp(0.0, 1.0);
    if x1 <= x0 || y1 <= y0 || grid.height == 0 || grid.width == 0 {
        return None;
    }
    let col_start = ((x0 * grid.width as f32).floor() as usize).min(grid.width - 1);
    let row_start = ((y0 * grid.height as f32).floor() as usize).min(grid.height - 1);
    let col_end = ((x1 * grid.width as f32).ceil() as usize)
        .saturating_sub(1)
        .min(grid.width - 1);
    let row_end = ((y1 * grid.height as f32).ceil() as usize)
        .saturating_sub(1)
        .min(grid.height - 1);
    Some((row_start, row_end, col_start, col_end))
}

fn target_mask_for_context(context: &SparseTokenMask, target_keep: usize) -> SparseTokenMask {
    let dense_len = context.dense_len();
    let context_set = context.indices().iter().copied().collect::<BTreeSet<_>>();
    let mut target = SparseTokenMask::evenly_spaced(dense_len, target_keep)
        .indices()
        .iter()
        .copied()
        .filter(|index| !context_set.contains(index))
        .collect::<Vec<_>>();
    if target.len() < target_keep.min(dense_len.saturating_sub(context.len()).max(1)) {
        for index in 0..dense_len {
            if !context_set.contains(&index) && !target.contains(&index) {
                target.push(index);
                if target.len() >= target_keep {
                    break;
                }
            }
        }
    }
    SparseTokenMask::new(target, dense_len).expect("target mask")
}

fn sparse_patchify_tokens(
    model: &VJepa2_1Model<JepaBackend>,
    config: &VJepaConfig,
    video: Tensor<JepaBackend, 5>,
    plan: &SparsePatchifyPlan<JepaBackend>,
) -> anyhow::Result<Tensor<JepaBackend, 3>> {
    let [batch, channels, frames, height, width] = video.shape().dims::<5>();
    let patchify_config = burn_flex_gmm::SparsePatchify3dConfig {
        in_channels: channels,
        out_channels: config.encoder.embed_dim,
        frames,
        height,
        width,
        tubelet_size: config.tubelet_size,
        patch_h: config.patch_size,
        patch_w: config.patch_size,
    };
    let bias = model
        .encoder
        .patch_embed
        .proj
        .bias
        .as_ref()
        .map(|bias| bias.val())
        .unwrap_or_else(|| {
            Tensor::<JepaBackend, 1>::zeros([config.encoder.embed_dim], &video.device())
        });
    let rows = plan.output_rows();
    let tokens = burn_flex_gmm::wgpu::sparse_patchify3d_forward_wgpu(
        &patchify_config,
        video,
        plan.coords.clone(),
        model.encoder.patch_embed.proj.weight.val(),
        bias,
    )
    .map_err(anyhow::Error::msg)?
    .reshape([batch, rows / batch, config.encoder.embed_dim]);
    Ok(tokens)
}

fn deterministic_autogaze_pipeline<B: Backend>(device: &B::Device) -> AutoGazePipeline<B> {
    let config = autogaze_config();
    let mut mapper = DeterministicParamMapper { cursor: 0 };
    let model = NativeAutoGazeModel::new(&config, device).map(&mut mapper);
    AutoGazePipeline::new(model)
        .with_max_gaze_tokens_each_frame(32)
        .with_task_loss_requirement(None)
}

fn autogaze_config() -> AutoGazeConfig {
    let hidden = 8;
    let heads = 2;
    AutoGazeConfig {
        scales: "224".to_string(),
        max_num_frames: FRAMES,
        num_vision_tokens_each_frame: AUTOGAZE_CONNECTOR_TOKENS,
        gaze_model_config: GazeModelConfig {
            input_img_size: AUTOGAZE_MODEL_INPUT,
            num_vision_tokens_each_frame: AUTOGAZE_CONNECTOR_TOKENS,
            attn_mode: "sdpa".to_string(),
            vision_model_config: VisionModelConfig {
                hidden_dim: hidden,
                out_dim: hidden,
                depth: 1,
                kernel_size: PATCH_SIZE,
                temporal_patch_size: 1,
                trunk_temporal_kernel_size: 3,
                trunk_spatial_kernel_size: 1,
            },
            connector_config: ConnectorConfig {
                hidden_dim: hidden,
                num_tokens: AUTOGAZE_CONNECTOR_TOKENS,
            },
            gaze_decoder_config: GazeDecoderConfig {
                vocab_size: AUTOGAZE_CONNECTOR_TOKENS + 1,
                hidden_size: hidden,
                intermediate_size: hidden * 2,
                num_hidden_layers: 1,
                num_attention_heads: heads,
                num_key_value_heads: heads,
                max_position_embeddings: 512,
                bos_token_id: 0,
                eos_token_id: AUTOGAZE_CONNECTOR_TOKENS as i64,
                head_dim: hidden / heads,
                num_multi_token_pred: 2,
                ..GazeDecoderConfig::default()
            },
        },
        ..AutoGazeConfig::default()
    }
}

fn jepa_config() -> VJepaConfig {
    VJepaConfig {
        model_type: "vjepa2_1_pipeline_bench".to_string(),
        image_size: 384,
        patch_size: PATCH_SIZE,
        num_frames: FRAMES,
        tubelet_size: TUBELET_SIZE,
        in_channels: CHANNELS,
        encoder: VJepaEncoderConfig {
            embed_dim: 64,
            depth: 1,
            num_heads: 4,
            mlp_ratio: 2.0,
            layer_norm_eps: 1.0e-6,
            use_rope: true,
            interpolate_rope: true,
            modality_embedding: true,
            n_output_distillation: 1,
        },
        predictor: VJepaPredictorConfig {
            embed_dim: 48,
            depth: 1,
            num_heads: 4,
            mlp_ratio: 2.0,
            num_mask_tokens: 2,
            output_dim: Some(64),
            return_all_tokens: false,
            layer_norm_eps: 1.0e-6,
            use_rope: true,
        },
        preprocess: Default::default(),
        variant: VJepaModelVariant::VitBase384,
    }
}

fn deterministic_autogaze_values(resolution: Resolution) -> Vec<f32> {
    let mut values =
        Vec::with_capacity(BATCH * FRAMES * CHANNELS * resolution.height * resolution.width);
    for _batch in 0..BATCH {
        for frame in 0..FRAMES {
            for channel in 0..CHANNELS {
                for y in 0..resolution.height {
                    for x in 0..resolution.width {
                        values.push(pixel_value(frame, channel, y, x));
                    }
                }
            }
        }
    }
    values
}

fn deterministic_jepa_values(resolution: Resolution) -> Vec<f32> {
    let mut values =
        Vec::with_capacity(BATCH * CHANNELS * FRAMES * resolution.height * resolution.width);
    for _batch in 0..BATCH {
        for channel in 0..CHANNELS {
            for frame in 0..FRAMES {
                for y in 0..resolution.height {
                    for x in 0..resolution.width {
                        values.push(pixel_value(frame, channel, y, x));
                    }
                }
            }
        }
    }
    values
}

fn pixel_value(frame: usize, channel: usize, y: usize, x: usize) -> f32 {
    let value = (x * 13 + y * 17 + frame * 29 + channel * 31) % 251;
    (value as f32 / 125.0) - 1.0
}

fn trace_point_count(traces: &[FrameFixationTrace]) -> usize {
    traces
        .iter()
        .flat_map(|trace| trace.frames.iter())
        .map(|set| set.points.len())
        .sum()
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| !matches!(value.as_str(), "0" | "false" | "False" | "FALSE"))
        .unwrap_or(default)
}

fn backend_filter() -> Vec<String> {
    env::var("BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS")
        .unwrap_or_else(|_| "ndarray,webgpu,cuda".to_string())
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_ascii_lowercase())
        .collect()
}

fn backend_enabled(filter: &[String], backend: &str) -> bool {
    filter
        .iter()
        .any(|value| value == "all" || value == backend)
}

fn bench_output_path() -> PathBuf {
    env::var_os("BURN_JEPA_PIPELINE_BENCH_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/autogaze_sparse_jepa_pipeline_bench.csv"))
}

fn csv_string(rows: &[BenchRow]) -> String {
    let mut out = String::from(BenchRow::csv_header());
    out.push('\n');
    for row in rows {
        out.push_str(&row.to_csv());
        out.push('\n');
    }
    out
}

impl BenchRow {
    fn csv_header() -> &'static str {
        "autogaze_backend,jepa_backend,resolution,width,height,frames,dense_tokens,target_density,actual_density,autogaze_top_k,context_tokens,target_tokens,autogaze_trace_ms,sparse_project_plan_ms,dense_patchify_ms,sparse_patchify_ms,sparse_encoder_ms,predictor_ms,sparse_jepa_ms,e2e_pipeline_ms,clips_per_sec,frames_per_sec"
    }

    fn to_csv(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{:.4},{:.4},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2},{:.2}",
            self.autogaze_backend,
            self.jepa_backend,
            self.resolution,
            self.width,
            self.height,
            self.frames,
            self.dense_tokens,
            self.target_density,
            self.actual_density,
            self.autogaze_top_k,
            self.context_tokens,
            self.target_tokens,
            self.autogaze_trace_ms,
            self.sparse_project_plan_ms,
            self.dense_patchify_ms,
            self.sparse_patchify_ms,
            self.sparse_encoder_ms,
            self.predictor_ms,
            self.sparse_jepa_ms,
            self.e2e_pipeline_ms,
            self.clips_per_sec,
            self.frames_per_sec
        )
    }
}

#[cfg(feature = "webgpu")]
fn webgpu_device() -> Option<burn::backend::wgpu::WgpuDevice> {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    let device = burn::backend::wgpu::WgpuDevice::default();
    match INIT.get_or_init(|| {
        match panic::catch_unwind(AssertUnwindSafe(|| {
            burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::AutoGraphicsApi>(
                &device,
                Default::default(),
            );
        })) {
            Ok(()) => Ok(()),
            Err(payload) => {
                let reason = panic_payload_to_string(payload);
                if reason.to_ascii_lowercase().contains("already initialized") {
                    Ok(())
                } else {
                    Err(reason)
                }
            }
        }
    }) {
        Ok(()) => Some(device),
        Err(reason) if is_unavailable_backend_reason(reason) => {
            eprintln!("skipping autogaze-webgpu benchmark: {reason}");
            None
        }
        Err(reason) => panic!("autogaze-webgpu benchmark initialization failed: {reason}"),
    }
}

fn run_optional<T>(name: &str, test: impl FnOnce() -> T) -> Option<T> {
    match panic::catch_unwind(AssertUnwindSafe(test)) {
        Ok(value) => Some(value),
        Err(payload) => {
            let reason = panic_payload_to_string(payload);
            if is_unavailable_backend_reason(&reason) {
                eprintln!("skipping {name} benchmark: {reason}");
                None
            } else {
                panic!("{name} benchmark setup failed: {reason}");
            }
        }
    }
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(message) => (*message).into(),
            Err(_) => "panic without string payload".into(),
        },
    }
}

fn is_unavailable_backend_reason(reason: &str) -> bool {
    let lower = reason.to_ascii_lowercase();
    [
        "no adapter",
        "no possible adapter",
        "no suitable adapter",
        "adapter not found",
        "backend is not available",
        "backend unavailable",
        "cuda driver",
        "driver version is insufficient",
        "failed to initialize cuda",
        "could not initialize cuda",
        "libcuda",
        "no cuda",
        "not supported on this system",
        "webgpu",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

#[derive(Default)]
struct DeterministicParamMapper {
    cursor: usize,
}

impl<B: Backend> ModuleMapper<B> for DeterministicParamMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let tensor = param.val();
        let dims = tensor.shape().dims::<D>();
        let len = dims.iter().product::<usize>();
        let start = self.cursor;
        self.cursor += len;
        let values = (0..len)
            .map(|idx| (((start + idx) % 97) as f32 - 48.0) * 0.002)
            .collect::<Vec<_>>();
        Param::from_tensor(Tensor::from_data(
            TensorData::new(values, dims),
            &tensor.device(),
        ))
    }
}
