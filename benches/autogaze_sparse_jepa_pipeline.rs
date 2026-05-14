use std::{
    env, fs,
    hint::black_box,
    panic::{self, AssertUnwindSafe},
    path::PathBuf,
    time::{Duration, Instant},
};

#[cfg(feature = "cuda")]
use std::{path::Path, process::Command};

#[cfg(any(feature = "cuda", feature = "webgpu"))]
use std::sync::OnceLock;

use burn::{
    module::{Module, ModuleMapper, Param},
    tensor::{
        Tensor, TensorData,
        backend::{Backend, BackendTypes},
    },
};
use burn_autogaze::{
    AutoGazeConfig, AutoGazeGenerateOutput, AutoGazeInferenceMode, AutoGazePipeline,
    AutoGazeStreamingCache, ConnectorConfig, GazeDecoderConfig, GazeModelConfig,
    NativeAutoGazeModel, VisionModelConfig,
};
use burn_jepa::{
    AutogazeSparseJepaWindowConfig, SparsePatchifyPlan, SparsePredictorPlan,
    TemporalSparseJepaStream, TemporalSparseJepaStreamOutput, VJepa2_1Model, VJepaConfig,
    VJepaEncoderConfig, VJepaModelVariant, VJepaPredictorConfig, autogaze_image_token_grid,
};

#[cfg(not(any(feature = "sparse-patchify-wgpu", feature = "sparse-patchify-cuda")))]
compile_error!(
    "autogaze_sparse_jepa_pipeline requires sparse-patchify-wgpu or sparse-patchify-cuda"
);

trait SparsePatchifyBenchBackend: Backend {
    const JEPA_BACKEND_LABEL: &'static str;

    fn sparse_patchify_tokens(
        model: &VJepa2_1Model<Self>,
        video: Tensor<Self, 5>,
        plan: &SparsePatchifyPlan<Self>,
    ) -> anyhow::Result<Tensor<Self, 3>>;
}

trait SparsePatchifyBenchStream<B: SparsePatchifyBenchBackend> {
    fn forward_frame_tokens_sparse_patchify_bench(
        &mut self,
        model: &VJepa2_1Model<B>,
        video: Tensor<B, 5>,
        frame_tokens: &[Vec<usize>],
        mask_index: usize,
    ) -> anyhow::Result<TemporalSparseJepaStreamOutput<B>>;

    fn forward_masks_sparse_patchify_bench(
        &mut self,
        model: &VJepa2_1Model<B>,
        video: Tensor<B, 5>,
        context_mask: burn_jepa::SparseTokenMask,
        target_mask: burn_jepa::SparseTokenMask,
        mask_index: usize,
    ) -> anyhow::Result<TemporalSparseJepaStreamOutput<B>>;
}

#[cfg(feature = "sparse-patchify-wgpu")]
impl SparsePatchifyBenchBackend for burn_flex_gmm::wgpu::DefaultWgpuBackend {
    const JEPA_BACKEND_LABEL: &'static str = "sparse-patchify-wgpu";

    fn sparse_patchify_tokens(
        model: &VJepa2_1Model<Self>,
        video: Tensor<Self, 5>,
        plan: &SparsePatchifyPlan<Self>,
    ) -> anyhow::Result<Tensor<Self, 3>> {
        model.encoder.sparse_patchify_video_wgpu(video, plan)
    }
}

#[cfg(feature = "sparse-patchify-wgpu")]
impl SparsePatchifyBenchStream<burn_flex_gmm::wgpu::DefaultWgpuBackend>
    for TemporalSparseJepaStream<burn_flex_gmm::wgpu::DefaultWgpuBackend>
{
    fn forward_frame_tokens_sparse_patchify_bench(
        &mut self,
        model: &VJepa2_1Model<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        video: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 5>,
        frame_tokens: &[Vec<usize>],
        mask_index: usize,
    ) -> anyhow::Result<TemporalSparseJepaStreamOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>>
    {
        self.forward_frame_tokens_sparse_patchify_wgpu(model, video, frame_tokens, mask_index)
    }

    fn forward_masks_sparse_patchify_bench(
        &mut self,
        model: &VJepa2_1Model<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
        video: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 5>,
        context_mask: burn_jepa::SparseTokenMask,
        target_mask: burn_jepa::SparseTokenMask,
        mask_index: usize,
    ) -> anyhow::Result<TemporalSparseJepaStreamOutput<burn_flex_gmm::wgpu::DefaultWgpuBackend>>
    {
        self.forward_masks_sparse_patchify_wgpu(model, video, context_mask, target_mask, mask_index)
    }
}

#[cfg(feature = "sparse-patchify-cuda")]
impl SparsePatchifyBenchBackend for burn_flex_gmm::cuda::DefaultCudaBackend {
    const JEPA_BACKEND_LABEL: &'static str = "sparse-patchify-cuda";

    fn sparse_patchify_tokens(
        model: &VJepa2_1Model<Self>,
        video: Tensor<Self, 5>,
        plan: &SparsePatchifyPlan<Self>,
    ) -> anyhow::Result<Tensor<Self, 3>> {
        model.encoder.sparse_patchify_video_cuda(video, plan)
    }
}

#[cfg(feature = "sparse-patchify-cuda")]
impl SparsePatchifyBenchStream<burn_flex_gmm::cuda::DefaultCudaBackend>
    for TemporalSparseJepaStream<burn_flex_gmm::cuda::DefaultCudaBackend>
{
    fn forward_frame_tokens_sparse_patchify_bench(
        &mut self,
        model: &VJepa2_1Model<burn_flex_gmm::cuda::DefaultCudaBackend>,
        video: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 5>,
        frame_tokens: &[Vec<usize>],
        mask_index: usize,
    ) -> anyhow::Result<TemporalSparseJepaStreamOutput<burn_flex_gmm::cuda::DefaultCudaBackend>>
    {
        self.forward_frame_tokens_sparse_patchify_cuda(model, video, frame_tokens, mask_index)
    }

    fn forward_masks_sparse_patchify_bench(
        &mut self,
        model: &VJepa2_1Model<burn_flex_gmm::cuda::DefaultCudaBackend>,
        video: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 5>,
        context_mask: burn_jepa::SparseTokenMask,
        target_mask: burn_jepa::SparseTokenMask,
        mask_index: usize,
    ) -> anyhow::Result<TemporalSparseJepaStreamOutput<burn_flex_gmm::cuda::DefaultCudaBackend>>
    {
        self.forward_masks_sparse_patchify_cuda(model, video, context_mask, target_mask, mask_index)
    }
}

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
struct BenchConfig {
    trace: BenchTraceConfig,
    dense_patchify: bool,
    progress: bool,
}

impl BenchConfig {
    fn from_env() -> Self {
        Self {
            trace: BenchTraceConfig::from_env(),
            dense_patchify: env_bool("BURN_JEPA_PIPELINE_BENCH_DENSE_PATCHIFY", true),
            progress: env_bool("BURN_JEPA_PIPELINE_BENCH_PROGRESS", false),
        }
    }

    fn progress(self, message: impl AsRef<str>) {
        if self.progress {
            eprintln!("{}", message.as_ref());
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct BenchTraceConfig {
    mode: BenchTraceMode,
}

impl BenchTraceConfig {
    const ENV: &'static str = "BURN_JEPA_PIPELINE_BENCH_TRACE";

    fn from_env() -> Self {
        let enabled = env_bool(Self::ENV, false);
        Self {
            mode: if enabled {
                BenchTraceMode::DecodedFixations
            } else {
                BenchTraceMode::Disabled
            },
        }
    }

    #[inline]
    fn enabled(self) -> bool {
        self.mode == BenchTraceMode::DecodedFixations
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum BenchTraceMode {
    #[default]
    Disabled,
    DecodedFixations,
}

fn measure_autogaze_trace_ms<B>(
    autogaze: &AutoGazePipeline<B>,
    video: &Tensor<B, 5>,
    top_k: usize,
    device: &B::Device,
    warmups: usize,
    reps: usize,
) -> f64
where
    B: Backend,
{
    measure_ms(warmups, reps, || {
        let traces = autogaze.trace_video_with_mode(
            video.clone(),
            top_k,
            AutoGazeInferenceMode::ResizeToModelInput,
        );
        black_box(trace_point_count(&traces));
        <B as Backend>::sync(device).expect("autogaze trace sync");
        traces.len()
    })
}

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
    autogaze_generate_ms: f64,
    rolling_autogaze_generate_ms: f64,
    rolling_autogaze_streaming_generate_ms: f64,
    autogaze_trace_ms: f64,
    sparse_project_ms: f64,
    sparse_mask_project_ms: f64,
    sparse_plan_ms: f64,
    sparse_project_plan_ms: f64,
    dense_patchify_ms: f64,
    sparse_patchify_ms: f64,
    sparse_encoder_ms: f64,
    predictor_ms: f64,
    sparse_jepa_ms: f64,
    temporal_stream_ms: f64,
    temporal_mask_stream_ms: f64,
    rolling_temporal_stream_ms: f64,
    rolling_temporal_mask_stream_ms: f64,
    e2e_pipeline_ms: f64,
    temporal_e2e_pipeline_ms: f64,
    temporal_mask_e2e_pipeline_ms: f64,
    rolling_temporal_e2e_pipeline_ms: f64,
    rolling_mask_temporal_e2e_pipeline_ms: f64,
    rolling_streaming_temporal_e2e_pipeline_ms: f64,
    clips_per_sec: f64,
    frames_per_sec: f64,
    temporal_clips_per_sec: f64,
    temporal_mask_clips_per_sec: f64,
    temporal_frames_per_sec: f64,
    rolling_temporal_frames_per_sec: f64,
    rolling_mask_temporal_frames_per_sec: f64,
    rolling_streaming_temporal_frames_per_sec: f64,
}

fn main() {
    let reps = env_usize("BURN_JEPA_PIPELINE_BENCH_REPS", 3);
    let warmups = env_usize("BURN_JEPA_PIPELINE_BENCH_WARMUPS", 1);
    let include_1080 = env_bool("BURN_JEPA_PIPELINE_BENCH_1080P", true);
    let bench_config = BenchConfig::from_env();
    let backend_filter = backend_filter();
    let jepa_backend_filter = jepa_backend_filter();
    let resolution_filter = name_filter("BURN_JEPA_PIPELINE_BENCH_RESOLUTIONS");
    let densities = density_cases();
    let resolutions = RESOLUTIONS
        .iter()
        .copied()
        .filter(|case| include_1080 || case.name != "1080p16pad")
        .filter(|case| name_enabled(&resolution_filter, case.name))
        .collect::<Vec<_>>();
    let mut rows = Vec::new();

    println!("{}", BenchRow::csv_header());

    if backend_enabled(&backend_filter, "ndarray") {
        append_autogaze_backend_rows::<burn::backend::NdArray<f32>>(
            &mut rows,
            "autogaze-ndarray",
            <burn::backend::NdArray<f32> as BackendTypes>::Device::default(),
            &jepa_backend_filter,
            &resolutions,
            &densities,
            bench_config,
            warmups,
            reps,
        );
    }

    #[cfg(feature = "webgpu")]
    if backend_enabled(&backend_filter, "webgpu")
        && let Some(device) = webgpu_device()
    {
        append_autogaze_backend_rows::<burn::backend::WebGpu<f32, i32>>(
            &mut rows,
            "autogaze-webgpu",
            device,
            &jepa_backend_filter,
            &resolutions,
            &densities,
            bench_config,
            warmups,
            reps,
        );
    }

    #[cfg(feature = "cuda")]
    if backend_enabled(&backend_filter, "cuda") {
        if let Err(reason) = cuda_runtime_preflight() {
            eprintln!("skipping autogaze-cuda benchmark: {reason}");
        } else {
            append_autogaze_backend_rows::<burn::backend::Cuda<f32, i32>>(
                &mut rows,
                "autogaze-cuda",
                burn::backend::cuda::CudaDevice::default(),
                &jepa_backend_filter,
                &resolutions,
                &densities,
                bench_config,
                warmups,
                reps,
            );
        }
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

fn append_autogaze_backend_rows<B>(
    rows: &mut Vec<BenchRow>,
    autogaze_backend: &'static str,
    autogaze_device: B::Device,
    jepa_backend_filter: &[String],
    resolutions: &[Resolution],
    densities: &[f32],
    bench_config: BenchConfig,
    warmups: usize,
    reps: usize,
) where
    B: Backend,
    B::Device: Clone,
{
    #[cfg(feature = "sparse-patchify-wgpu")]
    if backend_enabled(
        jepa_backend_filter,
        <burn_flex_gmm::wgpu::DefaultWgpuBackend as SparsePatchifyBenchBackend>::JEPA_BACKEND_LABEL,
    ) {
        let name = format!(
            "{autogaze_backend}+{}",
            <burn_flex_gmm::wgpu::DefaultWgpuBackend as SparsePatchifyBenchBackend>::JEPA_BACKEND_LABEL
        );
        append_backend_rows(
            rows,
            run_optional(&name, || {
                run_autogaze_backend::<B, burn_flex_gmm::wgpu::DefaultWgpuBackend>(
                    autogaze_backend,
                    autogaze_device.clone(),
                    resolutions,
                    densities,
                    bench_config,
                    warmups,
                    reps,
                )
            }),
        );
    }

    #[cfg(feature = "sparse-patchify-cuda")]
    if backend_enabled(
        jepa_backend_filter,
        <burn_flex_gmm::cuda::DefaultCudaBackend as SparsePatchifyBenchBackend>::JEPA_BACKEND_LABEL,
    ) {
        #[cfg(feature = "cuda")]
        if let Err(reason) = cuda_runtime_preflight() {
            eprintln!("skipping {autogaze_backend}+sparse-patchify-cuda benchmark: {reason}");
            return;
        }

        let name = format!(
            "{autogaze_backend}+{}",
            <burn_flex_gmm::cuda::DefaultCudaBackend as SparsePatchifyBenchBackend>::JEPA_BACKEND_LABEL
        );
        append_backend_rows(
            rows,
            run_optional(&name, || {
                run_autogaze_backend::<B, burn_flex_gmm::cuda::DefaultCudaBackend>(
                    autogaze_backend,
                    autogaze_device.clone(),
                    resolutions,
                    densities,
                    bench_config,
                    warmups,
                    reps,
                )
            }),
        );
    }
}

fn run_autogaze_backend<B, J>(
    autogaze_backend: &'static str,
    autogaze_device: B::Device,
    resolutions: &[Resolution],
    densities: &[f32],
    bench_config: BenchConfig,
    warmups: usize,
    reps: usize,
) -> Vec<BenchRow>
where
    B: Backend,
    B::Device: Clone,
    J: SparsePatchifyBenchBackend,
    J::Device: Default + Clone,
    TemporalSparseJepaStream<J>: SparsePatchifyBenchStream<J>,
{
    let autogaze = deterministic_autogaze_pipeline::<B>(&autogaze_device);
    let jepa_device = <J as BackendTypes>::Device::default();
    let mut rows = Vec::new();

    for &resolution in resolutions {
        let autogaze_image_grid = autogaze_image_token_grid(AUTOGAZE_CONNECTOR_TOKENS);
        let jepa_config = jepa_config();
        let jepa = VJepa2_1Model::<J>::new(&jepa_config, &jepa_device);
        let ag_values = deterministic_autogaze_values(resolution, FRAMES);
        let jepa_values = deterministic_jepa_values(resolution, FRAMES);
        let rolling_ag_values = deterministic_autogaze_values(resolution, TUBELET_SIZE);
        let rolling_jepa_values = deterministic_jepa_values(resolution, TUBELET_SIZE);
        let ag_video = Tensor::<B, 5>::from_data(
            TensorData::new(
                ag_values,
                [BATCH, FRAMES, CHANNELS, resolution.height, resolution.width],
            ),
            &autogaze_device,
        );
        let jepa_video = Tensor::<J, 5>::from_data(
            TensorData::new(
                jepa_values,
                [BATCH, CHANNELS, FRAMES, resolution.height, resolution.width],
            ),
            &jepa_device,
        );
        let rolling_ag_video = Tensor::<B, 5>::from_data(
            TensorData::new(
                rolling_ag_values,
                [
                    BATCH,
                    TUBELET_SIZE,
                    CHANNELS,
                    resolution.height,
                    resolution.width,
                ],
            ),
            &autogaze_device,
        );
        let rolling_jepa_video = Tensor::<J, 5>::from_data(
            TensorData::new(
                rolling_jepa_values,
                [
                    BATCH,
                    CHANNELS,
                    TUBELET_SIZE,
                    resolution.height,
                    resolution.width,
                ],
            ),
            &jepa_device,
        );
        <J as Backend>::sync(&jepa_device).expect("jepa input upload sync");

        bench_config.progress(format!(
            "{autogaze_backend}+{} {} dense patchify",
            J::JEPA_BACKEND_LABEL,
            resolution.name
        ));
        let dense_patchify_ms = if bench_config.dense_patchify {
            measure_ms(warmups, reps, || {
                let tokens = jepa.encoder.patch_embed.forward(jepa_video.clone());
                let dims = tokens.shape().dims::<3>();
                black_box(dims);
                <J as Backend>::sync(&jepa_device).expect("dense patchify sync");
                dims[1]
            })
        } else {
            0.0
        };

        for &target_density in densities {
            bench_config.progress(format!(
                "{autogaze_backend}+{} {} density={target_density:.4}",
                J::JEPA_BACKEND_LABEL,
                resolution.name
            ));
            let clip_plan = AutogazeSparseJepaWindowConfig::new(
                FRAMES,
                TUBELET_SIZE,
                PATCH_SIZE,
                resolution.height,
                resolution.width,
                AUTOGAZE_CONNECTOR_TOKENS,
                target_density,
                TARGET_TOKENS,
                autogaze.max_gaze_tokens_each_frame(),
            )
            .with_image_grid(autogaze_image_grid)
            .build()
            .expect("clip sparse window plan");
            let dense_tokens = clip_plan.grid.len();

            let generated = clip_plan.generate(&autogaze, ag_video.clone());
            let projection = clip_plan
                .project_generated_tokens(&generated)
                .expect("autogaze sparse context mask");
            let frame_tokens = projection.frame_tokens.clone();
            let context_mask = projection.context_mask;
            let target_mask = projection.target_mask;
            let context_plan = SparsePatchifyPlan::<J>::new(
                context_mask.clone(),
                clip_plan.grid,
                BATCH,
                &jepa_device,
            )
            .expect("context patchify plan");
            let predictor_plan = SparsePredictorPlan::<J>::new(
                &jepa_config,
                context_mask.clone(),
                target_mask.clone(),
                clip_plan.grid,
                BATCH,
                &jepa_device,
            )
            .expect("predictor plan");
            <B as Backend>::sync(&autogaze_device).expect("autogaze sync");
            <J as Backend>::sync(&jepa_device).expect("plan sync");

            let autogaze_generate_ms = measure_ms(warmups, reps, || {
                let generated = clip_plan.generate(&autogaze, ag_video.clone());
                black_box(generated_token_count(&generated));
                <B as Backend>::sync(&autogaze_device).expect("autogaze generate sync");
                generated.num_gazing_each_frame.len()
            });
            let autogaze_trace_ms = if bench_config.trace.enabled() {
                measure_autogaze_trace_ms(
                    &autogaze,
                    &ag_video,
                    clip_plan.top_k,
                    &autogaze_device,
                    warmups,
                    reps,
                )
            } else {
                0.0
            };
            let sparse_project_ms = measure_ms(warmups, reps, || {
                let projection = clip_plan
                    .project_generated_tokens(&generated)
                    .expect("project context mask");
                black_box(projection.frame_tokens.len());
                black_box(projection.context_mask.len());
                projection.target_mask.len()
            });
            let sparse_mask_project_ms = measure_ms(warmups, reps, || {
                let masks = clip_plan
                    .project_generated_masks(&generated)
                    .expect("project context masks");
                black_box(masks.context_mask.len());
                masks.target_mask.len()
            });
            let sparse_plan_ms = measure_ms(warmups, reps, || {
                let context_plan = SparsePatchifyPlan::<J>::new(
                    context_mask.clone(),
                    clip_plan.grid,
                    BATCH,
                    &jepa_device,
                )
                .expect("context plan");
                let predictor_plan = SparsePredictorPlan::<J>::new(
                    &jepa_config,
                    context_mask.clone(),
                    target_mask.clone(),
                    clip_plan.grid,
                    BATCH,
                    &jepa_device,
                )
                .expect("predictor plan");
                black_box(context_plan.output_rows());
                black_box(predictor_plan.sequence_indices.shape());
                <J as Backend>::sync(&jepa_device).expect("plan sync");
                context_plan.output_rows()
            });
            let sparse_project_plan_ms = measure_ms(warmups, reps, || {
                let projection = clip_plan
                    .project_generated_tokens(&generated)
                    .expect("project context mask");
                let context_mask = projection.context_mask;
                let target_mask = projection.target_mask;
                let context_plan = SparsePatchifyPlan::<J>::new(
                    context_mask.clone(),
                    clip_plan.grid,
                    BATCH,
                    &jepa_device,
                )
                .expect("context plan");
                let predictor_plan = SparsePredictorPlan::<J>::new(
                    &jepa_config,
                    context_mask,
                    target_mask,
                    clip_plan.grid,
                    BATCH,
                    &jepa_device,
                )
                .expect("predictor plan");
                black_box(context_plan.output_rows());
                black_box(predictor_plan.sequence_indices.shape());
                <J as Backend>::sync(&jepa_device).expect("project plan sync");
                context_plan.output_rows()
            });
            let sparse_patchify_ms = measure_ms(warmups, reps, || {
                let tokens = sparse_patchify_tokens(&jepa, jepa_video.clone(), &context_plan)
                    .expect("sparse patchify tokens");
                let dims = tokens.shape().dims::<3>();
                black_box(dims);
                <J as Backend>::sync(&jepa_device).expect("sparse patchify sync");
                dims[1]
            });
            let context_tokens = sparse_patchify_tokens(&jepa, jepa_video.clone(), &context_plan)
                .expect("precompute sparse tokens");
            <J as Backend>::sync(&jepa_device).expect("precompute sparse tokens sync");

            let sparse_encoder_ms = measure_ms(warmups, reps, || {
                let encoded = jepa.encoder.forward_sparse_tokens(
                    context_tokens.clone(),
                    BATCH,
                    clip_plan.grid,
                    context_mask.indices(),
                    true,
                );
                let dims = encoded.tokens.shape().dims::<3>();
                black_box(dims);
                <J as Backend>::sync(&jepa_device).expect("sparse encoder sync");
                dims[1]
            });
            let encoded_context = jepa.encoder.forward_sparse_tokens(
                context_tokens,
                BATCH,
                clip_plan.grid,
                context_mask.indices(),
                true,
            );
            <J as Backend>::sync(&jepa_device).expect("precompute encoded context sync");

            let predictor_ms = measure_ms(warmups, reps, || {
                let output = jepa
                    .predictor
                    .forward_sparse_with_plan(encoded_context.tokens.clone(), &predictor_plan, 0)
                    .expect("predictor");
                let dims = output.target_predictions.shape().dims::<3>();
                black_box(dims);
                <J as Backend>::sync(&jepa_device).expect("predictor sync");
                dims[1]
            });
            let sparse_jepa_ms = measure_ms(warmups, reps, || {
                run_sparse_jepa_once(&jepa, jepa_video.clone(), &context_plan, &predictor_plan)
                    .expect("sparse jepa");
                <J as Backend>::sync(&jepa_device).expect("sparse jepa sync");
            });
            let stream_config = clip_plan.stream.with_keyframe_interval(16);
            let mut temporal_stream = TemporalSparseJepaStream::<J>::new(stream_config);
            temporal_stream
                .forward_frame_tokens_sparse_patchify_bench(
                    &jepa,
                    jepa_video.clone(),
                    &frame_tokens,
                    0,
                )
                .expect("prime temporal stream");
            <J as Backend>::sync(&jepa_device).expect("prime temporal stream sync");
            let temporal_stream_ms = measure_ms(warmups, reps, || {
                let output = temporal_stream
                    .forward_frame_tokens_sparse_patchify_bench(
                        &jepa,
                        jepa_video.clone(),
                        &frame_tokens,
                        0,
                    )
                    .expect("temporal stream");
                black_box(output.reused_encoder_plan);
                black_box(output.reused_patchify_plan);
                black_box(output.temporal.reused_predictor_plan);
                <J as Backend>::sync(&jepa_device).expect("temporal stream sync");
            });
            let mut temporal_mask_stream =
                TemporalSparseJepaStream::<J>::new(clip_plan.stream.with_keyframe_interval(16));
            temporal_mask_stream
                .forward_masks_sparse_patchify_bench(
                    &jepa,
                    jepa_video.clone(),
                    context_mask.clone(),
                    target_mask.clone(),
                    0,
                )
                .expect("prime temporal mask stream");
            <J as Backend>::sync(&jepa_device).expect("prime temporal mask stream sync");
            let temporal_mask_stream_ms = measure_ms(warmups, reps, || {
                let output = temporal_mask_stream
                    .forward_masks_sparse_patchify_bench(
                        &jepa,
                        jepa_video.clone(),
                        context_mask.clone(),
                        target_mask.clone(),
                        0,
                    )
                    .expect("temporal mask stream");
                black_box(output.reused_encoder_plan);
                black_box(output.reused_patchify_plan);
                black_box(output.temporal.reused_predictor_plan);
                <J as Backend>::sync(&jepa_device).expect("temporal mask stream sync");
            });
            let rolling_plan = AutogazeSparseJepaWindowConfig::new(
                TUBELET_SIZE,
                TUBELET_SIZE,
                PATCH_SIZE,
                resolution.height,
                resolution.width,
                AUTOGAZE_CONNECTOR_TOKENS,
                target_density,
                TARGET_TOKENS,
                autogaze.max_gaze_tokens_each_frame(),
            )
            .with_image_grid(autogaze_image_grid)
            .build()
            .expect("rolling sparse window plan");
            let rolling_generated = rolling_plan.generate(&autogaze, rolling_ag_video.clone());
            let rolling_projection = rolling_plan
                .project_generated_tokens(&rolling_generated)
                .expect("rolling autogaze sparse context mask");
            let rolling_frame_tokens = rolling_projection.frame_tokens;
            let rolling_context_mask = rolling_projection.context_mask;
            let rolling_target_mask = rolling_projection.target_mask;
            let rolling_stream_config = rolling_plan.stream.with_keyframe_interval(16);
            let mut rolling_temporal_stream =
                TemporalSparseJepaStream::<J>::new(rolling_stream_config);
            rolling_temporal_stream
                .forward_frame_tokens_sparse_patchify_bench(
                    &jepa,
                    rolling_jepa_video.clone(),
                    &rolling_frame_tokens,
                    0,
                )
                .expect("prime rolling temporal stream");
            <J as Backend>::sync(&jepa_device).expect("prime rolling stream sync");
            let rolling_temporal_stream_ms = measure_ms(warmups, reps, || {
                let output = rolling_temporal_stream
                    .forward_frame_tokens_sparse_patchify_bench(
                        &jepa,
                        rolling_jepa_video.clone(),
                        &rolling_frame_tokens,
                        0,
                    )
                    .expect("rolling temporal stream");
                black_box(output.reused_encoder_plan);
                black_box(output.reused_patchify_plan);
                black_box(output.temporal.reused_predictor_plan);
                <J as Backend>::sync(&jepa_device).expect("rolling stream sync");
            });
            let mut rolling_temporal_mask_stream =
                TemporalSparseJepaStream::<J>::new(rolling_plan.stream.with_keyframe_interval(16));
            rolling_temporal_mask_stream
                .forward_masks_sparse_patchify_bench(
                    &jepa,
                    rolling_jepa_video.clone(),
                    rolling_context_mask.clone(),
                    rolling_target_mask.clone(),
                    0,
                )
                .expect("prime rolling temporal mask stream");
            <J as Backend>::sync(&jepa_device).expect("prime rolling mask stream sync");
            let rolling_temporal_mask_stream_ms = measure_ms(warmups, reps, || {
                let output = rolling_temporal_mask_stream
                    .forward_masks_sparse_patchify_bench(
                        &jepa,
                        rolling_jepa_video.clone(),
                        rolling_context_mask.clone(),
                        rolling_target_mask.clone(),
                        0,
                    )
                    .expect("rolling temporal mask stream");
                black_box(output.reused_encoder_plan);
                black_box(output.reused_patchify_plan);
                black_box(output.temporal.reused_predictor_plan);
                <J as Backend>::sync(&jepa_device).expect("rolling mask stream sync");
            });
            let rolling_autogaze_generate_ms = measure_ms(warmups, reps, || {
                let generated = rolling_plan.generate(&autogaze, rolling_ag_video.clone());
                black_box(generated_token_count(&generated));
                <B as Backend>::sync(&autogaze_device).expect("rolling autogaze generate sync");
                generated.num_gazing_each_frame.len()
            });
            let mut rolling_autogaze_cache = AutoGazeStreamingCache::<B>::new(FRAMES);
            rolling_plan.generate_streaming(
                &autogaze,
                rolling_ag_video.clone(),
                &mut rolling_autogaze_cache,
            );
            <B as Backend>::sync(&autogaze_device)
                .expect("prime rolling autogaze streaming cache sync");
            let rolling_autogaze_streaming_generate_ms = measure_ms(warmups, reps, || {
                let generated = rolling_plan.generate_streaming(
                    &autogaze,
                    rolling_ag_video.clone(),
                    &mut rolling_autogaze_cache,
                );
                black_box(generated_token_count(&generated));
                <B as Backend>::sync(&autogaze_device)
                    .expect("rolling streaming autogaze generate sync");
                generated.num_gazing_each_frame.len()
            });
            let e2e_pipeline_ms = measure_ms(warmups, reps, || {
                let generated = clip_plan.generate(&autogaze, ag_video.clone());
                let projection = clip_plan
                    .project_generated_tokens(&generated)
                    .expect("e2e context mask");
                let context_mask = projection.context_mask;
                let target_mask = projection.target_mask;
                let context_plan = SparsePatchifyPlan::<J>::new(
                    context_mask.clone(),
                    clip_plan.grid,
                    BATCH,
                    &jepa_device,
                )
                .expect("e2e context plan");
                let predictor_plan = SparsePredictorPlan::<J>::new(
                    &jepa_config,
                    context_mask,
                    target_mask,
                    clip_plan.grid,
                    BATCH,
                    &jepa_device,
                )
                .expect("e2e predictor plan");
                run_sparse_jepa_once(&jepa, jepa_video.clone(), &context_plan, &predictor_plan)
                    .expect("e2e sparse jepa");
                <B as Backend>::sync(&autogaze_device).expect("e2e autogaze sync");
                <J as Backend>::sync(&jepa_device).expect("e2e jepa sync");
            });
            let mut temporal_e2e_stream = TemporalSparseJepaStream::<J>::new(stream_config);
            temporal_e2e_stream
                .forward_frame_tokens_sparse_patchify_bench(
                    &jepa,
                    jepa_video.clone(),
                    &frame_tokens,
                    0,
                )
                .expect("prime e2e temporal stream");
            <J as Backend>::sync(&jepa_device).expect("prime e2e temporal stream sync");
            let temporal_e2e_pipeline_ms = measure_ms(warmups, reps, || {
                let generated = clip_plan.generate(&autogaze, ag_video.clone());
                let projection = clip_plan
                    .project_generated_tokens(&generated)
                    .expect("e2e temporal projection");
                let output = temporal_e2e_stream
                    .forward_frame_tokens_sparse_patchify_bench(
                        &jepa,
                        jepa_video.clone(),
                        &projection.frame_tokens,
                        0,
                    )
                    .expect("e2e temporal stream");
                black_box(output.reused_encoder_plan);
                black_box(output.reused_patchify_plan);
                black_box(output.temporal.reused_predictor_plan);
                <B as Backend>::sync(&autogaze_device).expect("temporal e2e autogaze sync");
                <J as Backend>::sync(&jepa_device).expect("temporal e2e jepa sync");
            });
            let mut temporal_mask_e2e_stream = TemporalSparseJepaStream::<J>::new(stream_config);
            temporal_mask_e2e_stream
                .forward_masks_sparse_patchify_bench(
                    &jepa,
                    jepa_video.clone(),
                    context_mask.clone(),
                    target_mask.clone(),
                    0,
                )
                .expect("prime e2e temporal mask stream");
            <J as Backend>::sync(&jepa_device).expect("prime e2e temporal mask stream sync");
            let temporal_mask_e2e_pipeline_ms = measure_ms(warmups, reps, || {
                let generated = clip_plan.generate(&autogaze, ag_video.clone());
                let masks = clip_plan
                    .project_generated_masks(&generated)
                    .expect("e2e temporal masks");
                let output = temporal_mask_e2e_stream
                    .forward_masks_sparse_patchify_bench(
                        &jepa,
                        jepa_video.clone(),
                        masks.context_mask,
                        masks.target_mask,
                        0,
                    )
                    .expect("e2e temporal mask stream");
                black_box(output.reused_encoder_plan);
                black_box(output.reused_patchify_plan);
                black_box(output.temporal.reused_predictor_plan);
                <B as Backend>::sync(&autogaze_device).expect("temporal mask e2e autogaze sync");
                <J as Backend>::sync(&jepa_device).expect("temporal mask e2e jepa sync");
            });
            let mut rolling_temporal_e2e_stream =
                TemporalSparseJepaStream::<J>::new(rolling_stream_config);
            rolling_temporal_e2e_stream
                .forward_frame_tokens_sparse_patchify_bench(
                    &jepa,
                    rolling_jepa_video.clone(),
                    &rolling_frame_tokens,
                    0,
                )
                .expect("prime rolling e2e stream");
            <J as Backend>::sync(&jepa_device).expect("prime rolling e2e stream sync");
            let rolling_temporal_e2e_pipeline_ms = measure_ms(warmups, reps, || {
                let generated = rolling_plan.generate(&autogaze, rolling_ag_video.clone());
                let projection = rolling_plan
                    .project_generated_tokens(&generated)
                    .expect("rolling e2e temporal projection");
                let output = rolling_temporal_e2e_stream
                    .forward_frame_tokens_sparse_patchify_bench(
                        &jepa,
                        rolling_jepa_video.clone(),
                        &projection.frame_tokens,
                        0,
                    )
                    .expect("rolling e2e temporal stream");
                black_box(output.reused_encoder_plan);
                black_box(output.reused_patchify_plan);
                black_box(output.temporal.reused_predictor_plan);
                <B as Backend>::sync(&autogaze_device).expect("rolling e2e autogaze sync");
                <J as Backend>::sync(&jepa_device).expect("rolling e2e jepa sync");
            });
            let mut rolling_mask_temporal_e2e_stream =
                TemporalSparseJepaStream::<J>::new(rolling_stream_config);
            rolling_mask_temporal_e2e_stream
                .forward_masks_sparse_patchify_bench(
                    &jepa,
                    rolling_jepa_video.clone(),
                    rolling_context_mask.clone(),
                    rolling_target_mask.clone(),
                    0,
                )
                .expect("prime rolling mask e2e stream");
            <J as Backend>::sync(&jepa_device).expect("prime rolling mask e2e stream sync");
            let rolling_mask_temporal_e2e_pipeline_ms = measure_ms(warmups, reps, || {
                let generated = rolling_plan.generate(&autogaze, rolling_ag_video.clone());
                let masks = rolling_plan
                    .project_generated_masks(&generated)
                    .expect("rolling e2e temporal masks");
                let output = rolling_mask_temporal_e2e_stream
                    .forward_masks_sparse_patchify_bench(
                        &jepa,
                        rolling_jepa_video.clone(),
                        masks.context_mask,
                        masks.target_mask,
                        0,
                    )
                    .expect("rolling mask e2e temporal stream");
                black_box(output.reused_encoder_plan);
                black_box(output.reused_patchify_plan);
                black_box(output.temporal.reused_predictor_plan);
                <B as Backend>::sync(&autogaze_device).expect("rolling mask e2e autogaze sync");
                <J as Backend>::sync(&jepa_device).expect("rolling mask e2e jepa sync");
            });
            let mut rolling_streaming_autogaze_e2e_cache = AutoGazeStreamingCache::<B>::new(FRAMES);
            rolling_plan.generate_streaming(
                &autogaze,
                rolling_ag_video.clone(),
                &mut rolling_streaming_autogaze_e2e_cache,
            );
            let mut rolling_streaming_temporal_e2e_stream =
                TemporalSparseJepaStream::<J>::new(rolling_stream_config);
            rolling_streaming_temporal_e2e_stream
                .forward_masks_sparse_patchify_bench(
                    &jepa,
                    rolling_jepa_video.clone(),
                    rolling_context_mask.clone(),
                    rolling_target_mask.clone(),
                    0,
                )
                .expect("prime rolling streaming e2e stream");
            <B as Backend>::sync(&autogaze_device)
                .expect("prime rolling streaming e2e autogaze sync");
            <J as Backend>::sync(&jepa_device).expect("prime rolling streaming e2e jepa sync");
            let rolling_streaming_temporal_e2e_pipeline_ms = measure_ms(warmups, reps, || {
                let generated = rolling_plan.generate_streaming(
                    &autogaze,
                    rolling_ag_video.clone(),
                    &mut rolling_streaming_autogaze_e2e_cache,
                );
                let masks = rolling_plan
                    .project_generated_masks(&generated)
                    .expect("rolling streaming e2e temporal masks");
                let output = rolling_streaming_temporal_e2e_stream
                    .forward_masks_sparse_patchify_bench(
                        &jepa,
                        rolling_jepa_video.clone(),
                        masks.context_mask,
                        masks.target_mask,
                        0,
                    )
                    .expect("rolling streaming e2e temporal stream");
                black_box(output.reused_encoder_plan);
                black_box(output.reused_patchify_plan);
                black_box(output.temporal.reused_predictor_plan);
                <B as Backend>::sync(&autogaze_device)
                    .expect("rolling streaming e2e autogaze sync");
                <J as Backend>::sync(&jepa_device).expect("rolling streaming e2e jepa sync");
            });
            let clips_per_sec = 1000.0 / e2e_pipeline_ms.max(f64::EPSILON);
            let frames_per_sec = clips_per_sec * FRAMES as f64;
            let temporal_clips_per_sec = 1000.0 / temporal_e2e_pipeline_ms.max(f64::EPSILON);
            let temporal_mask_clips_per_sec =
                1000.0 / temporal_mask_e2e_pipeline_ms.max(f64::EPSILON);
            let temporal_frames_per_sec = temporal_clips_per_sec * FRAMES as f64;
            let rolling_temporal_frames_per_sec =
                (1000.0 / rolling_temporal_e2e_pipeline_ms.max(f64::EPSILON)) * TUBELET_SIZE as f64;
            let rolling_mask_temporal_frames_per_sec = (1000.0
                / rolling_mask_temporal_e2e_pipeline_ms.max(f64::EPSILON))
                * TUBELET_SIZE as f64;
            let rolling_streaming_temporal_frames_per_sec = (1000.0
                / rolling_streaming_temporal_e2e_pipeline_ms.max(f64::EPSILON))
                * TUBELET_SIZE as f64;

            let row = BenchRow {
                autogaze_backend,
                jepa_backend: J::JEPA_BACKEND_LABEL,
                resolution: resolution.name,
                width: resolution.width,
                height: resolution.height,
                frames: FRAMES,
                dense_tokens,
                target_density,
                actual_density: context_mask.len() as f32 / dense_tokens as f32,
                autogaze_top_k: clip_plan.top_k,
                context_tokens: context_mask.len(),
                target_tokens: target_mask.len(),
                autogaze_generate_ms,
                rolling_autogaze_generate_ms,
                rolling_autogaze_streaming_generate_ms,
                autogaze_trace_ms,
                sparse_project_ms,
                sparse_mask_project_ms,
                sparse_plan_ms,
                sparse_project_plan_ms,
                dense_patchify_ms,
                sparse_patchify_ms,
                sparse_encoder_ms,
                predictor_ms,
                sparse_jepa_ms,
                temporal_stream_ms,
                temporal_mask_stream_ms,
                rolling_temporal_stream_ms,
                rolling_temporal_mask_stream_ms,
                e2e_pipeline_ms,
                temporal_e2e_pipeline_ms,
                temporal_mask_e2e_pipeline_ms,
                rolling_temporal_e2e_pipeline_ms,
                rolling_mask_temporal_e2e_pipeline_ms,
                rolling_streaming_temporal_e2e_pipeline_ms,
                clips_per_sec,
                frames_per_sec,
                temporal_clips_per_sec,
                temporal_mask_clips_per_sec,
                temporal_frames_per_sec,
                rolling_temporal_frames_per_sec,
                rolling_mask_temporal_frames_per_sec,
                rolling_streaming_temporal_frames_per_sec,
            };
            println!("{}", row.to_csv());
            rows.push(row);
        }
    }

    rows
}

fn run_sparse_jepa_once<J>(
    jepa: &VJepa2_1Model<J>,
    video: Tensor<J, 5>,
    context_plan: &SparsePatchifyPlan<J>,
    predictor_plan: &SparsePredictorPlan<J>,
) -> anyhow::Result<()>
where
    J: SparsePatchifyBenchBackend,
{
    let tokens = sparse_patchify_tokens(jepa, video, context_plan)?;
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

fn sparse_patchify_tokens<J>(
    model: &VJepa2_1Model<J>,
    video: Tensor<J, 5>,
    plan: &SparsePatchifyPlan<J>,
) -> anyhow::Result<Tensor<J, 3>>
where
    J: SparsePatchifyBenchBackend,
{
    J::sparse_patchify_tokens(model, video, plan)
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

fn deterministic_autogaze_values(resolution: Resolution, frames: usize) -> Vec<f32> {
    let mut values =
        Vec::with_capacity(BATCH * frames * CHANNELS * resolution.height * resolution.width);
    for _batch in 0..BATCH {
        for frame in 0..frames {
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

fn deterministic_jepa_values(resolution: Resolution, frames: usize) -> Vec<f32> {
    let mut values =
        Vec::with_capacity(BATCH * CHANNELS * frames * resolution.height * resolution.width);
    for _batch in 0..BATCH {
        for channel in 0..CHANNELS {
            for frame in 0..frames {
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

fn trace_point_count(traces: &[burn_autogaze::FrameFixationTrace]) -> usize {
    traces
        .iter()
        .flat_map(|trace| trace.frames.iter())
        .map(|set| set.points.len())
        .sum()
}

fn generated_token_count(generated: &AutoGazeGenerateOutput) -> usize {
    generated
        .if_padded_gazing
        .iter()
        .flat_map(|flags| flags.iter())
        .filter(|&&padded| !padded)
        .count()
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

fn jepa_backend_filter() -> Vec<String> {
    env::var("BURN_JEPA_PIPELINE_JEPA_BACKENDS")
        .unwrap_or_else(|_| "all".to_string())
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_ascii_lowercase())
        .collect()
}

fn name_filter(name: &str) -> Vec<String> {
    env::var(name)
        .unwrap_or_else(|_| "all".to_string())
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_ascii_lowercase())
        .collect()
}

fn density_cases() -> Vec<f32> {
    let Ok(value) = env::var("BURN_JEPA_PIPELINE_BENCH_DENSITIES") else {
        return DENSITIES.to_vec();
    };
    let densities = value
        .split(',')
        .filter_map(|value| value.trim().parse::<f32>().ok())
        .filter(|density| density.is_finite() && *density > 0.0)
        .map(|density| density.min(1.0))
        .collect::<Vec<_>>();
    if densities.is_empty() {
        DENSITIES.to_vec()
    } else {
        densities
    }
}

fn backend_enabled(filter: &[String], backend: &str) -> bool {
    name_enabled(filter, backend)
}

fn name_enabled(filter: &[String], name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    filter.iter().any(|value| value == "all" || value == &name)
}

#[cfg(feature = "cuda")]
fn cuda_runtime_preflight() -> Result<(), String> {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        if env_bool("BURN_JEPA_PIPELINE_CUDA_FORCE", false) {
            return Ok(());
        }
        if env::var("CUDA_VISIBLE_DEVICES")
            .ok()
            .is_some_and(|value| {
                let value = value.trim();
                value.is_empty() || matches!(value, "-1" | "none" | "None" | "NONE")
            })
        {
            return Err(
                "CUDA_VISIBLE_DEVICES disables CUDA; set BURN_JEPA_PIPELINE_CUDA_FORCE=1 to try anyway"
                    .to_string(),
            );
        }
        let nvidia_smi = nvidia_smi_summary();
        if cfg!(target_os = "linux") && !cuda_device_nodes_visible() {
            return Err(cuda_missing_device_nodes_reason(nvidia_smi.as_ref()));
        }
        nvidia_smi.map(|_| ())
    })
    .clone()
}

#[cfg(feature = "cuda")]
fn cuda_device_nodes_visible() -> bool {
    Path::new("/dev/nvidiactl").exists() || Path::new("/dev/nvidia0").exists()
}

#[cfg(feature = "cuda")]
fn cuda_missing_device_nodes_reason(nvidia_smi: Result<&String, &String>) -> String {
    let mut reason = String::from("no /dev/nvidia* device nodes");
    match nvidia_smi {
        Ok(summary) if !summary.is_empty() => {
            reason.push_str("; nvidia-smi -L sees ");
            reason.push_str(summary);
        }
        Err(error) => {
            reason.push_str("; nvidia-smi -L probe failed: ");
            reason.push_str(error);
        }
        _ => {}
    }
    if Path::new("/proc/driver/nvidia/version").exists() {
        reason.push_str("; /proc/driver/nvidia is visible");
    }
    reason.push_str(
        "; CUDA runtime cannot open a device without NVIDIA character devices; set BURN_JEPA_PIPELINE_CUDA_FORCE=1 to try anyway",
    );
    reason
}

#[cfg(feature = "cuda")]
fn nvidia_smi_summary() -> Result<String, String> {
    match Command::new("nvidia-smi").arg("-L").output() {
        Ok(output) if output.status.success() && !output.stdout.is_empty() => {
            Ok(String::from_utf8_lossy(&output.stdout)
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" "))
        }
        Ok(output) if output.status.success() => {
            Err("nvidia-smi -L returned no CUDA devices".into())
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("nvidia-smi -L failed: {}", stderr.trim()))
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(err) => Err(format!("failed to run nvidia-smi -L: {err}")),
    }
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
        "autogaze_backend,jepa_backend,resolution,width,height,frames,dense_tokens,target_density,actual_density,autogaze_top_k,context_tokens,target_tokens,autogaze_generate_ms,rolling_autogaze_generate_ms,rolling_autogaze_streaming_generate_ms,autogaze_trace_ms,sparse_project_ms,sparse_mask_project_ms,sparse_plan_ms,sparse_project_plan_ms,dense_patchify_ms,sparse_patchify_ms,sparse_encoder_ms,predictor_ms,sparse_jepa_ms,temporal_stream_ms,temporal_mask_stream_ms,rolling_temporal_stream_ms,rolling_temporal_mask_stream_ms,e2e_pipeline_ms,temporal_e2e_pipeline_ms,temporal_mask_e2e_pipeline_ms,rolling_temporal_e2e_pipeline_ms,rolling_mask_temporal_e2e_pipeline_ms,rolling_streaming_temporal_e2e_pipeline_ms,clips_per_sec,frames_per_sec,temporal_clips_per_sec,temporal_mask_clips_per_sec,temporal_frames_per_sec,rolling_temporal_frames_per_sec,rolling_mask_temporal_frames_per_sec,rolling_streaming_temporal_frames_per_sec"
    }

    fn to_csv(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{:.4},{:.4},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
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
            self.autogaze_generate_ms,
            self.rolling_autogaze_generate_ms,
            self.rolling_autogaze_streaming_generate_ms,
            self.autogaze_trace_ms,
            self.sparse_project_ms,
            self.sparse_mask_project_ms,
            self.sparse_plan_ms,
            self.sparse_project_plan_ms,
            self.dense_patchify_ms,
            self.sparse_patchify_ms,
            self.sparse_encoder_ms,
            self.predictor_ms,
            self.sparse_jepa_ms,
            self.temporal_stream_ms,
            self.temporal_mask_stream_ms,
            self.rolling_temporal_stream_ms,
            self.rolling_temporal_mask_stream_ms,
            self.e2e_pipeline_ms,
            self.temporal_e2e_pipeline_ms,
            self.temporal_mask_e2e_pipeline_ms,
            self.rolling_temporal_e2e_pipeline_ms,
            self.rolling_mask_temporal_e2e_pipeline_ms,
            self.rolling_streaming_temporal_e2e_pipeline_ms,
            self.clips_per_sec,
            self.frames_per_sec,
            self.temporal_clips_per_sec,
            self.temporal_mask_clips_per_sec,
            self.temporal_frames_per_sec,
            self.rolling_temporal_frames_per_sec,
            self.rolling_mask_temporal_frames_per_sec,
            self.rolling_streaming_temporal_frames_per_sec
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
            if let Some(skip_reason) = optional_backend_skip_reason(name, &reason) {
                eprintln!("skipping {name} benchmark: {skip_reason}");
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

fn optional_backend_skip_reason(name: &str, reason: &str) -> Option<String> {
    if is_unavailable_backend_reason(reason) {
        return Some(reason.to_string());
    }

    let lower = reason.to_ascii_lowercase();
    if name.contains("cuda") && lower.contains("recverror") {
        return Some(format!(
            "{reason}; CUDA worker thread failed before returning results, which usually means the CUDA runtime could not initialize a device. Check for a preceding CUDA driver error and verify /dev/nvidia* device nodes are visible."
        ));
    }

    None
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
