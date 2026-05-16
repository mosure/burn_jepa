#![cfg_attr(
    not(any(
        feature = "ndarray",
        feature = "wgpu",
        feature = "webgpu",
        feature = "cuda",
        feature = "sparse-patchify-wgpu",
        feature = "sparse-patchify-cuda"
    )),
    allow(dead_code, unused_imports, unused_variables)
)]

use anyhow::Result;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_jepa::{
    AnyUp, AnyUpConfig, FeatureFrameInput, FeatureFrameMeasureConfig, FeatureFrameMetrics,
    FeatureFramePipeline, FeatureFramePipelineConfig, FeatureFrameRequest, FeatureFrameSchedule,
    FeatureFrameStream, FeaturePcaUpdateConfig, FeaturePcaUpdateMode, FrameId, FrameQueuePolicy,
    FrameStreamConfig, SparseMaskBatch, SparseTokenMask, VJepa2_1Model, VJepaConfig,
};

#[cfg(any(feature = "sparse-patchify-wgpu", feature = "sparse-patchify-cuda"))]
use burn_jepa::SparsePatchifyBatchPlan;

#[derive(Clone, Copy)]
struct BreakdownCase {
    label: &'static str,
    image_hw: usize,
    batch: usize,
    density: f32,
}

const CASES: [BreakdownCase; 2] = [
    BreakdownCase {
        label: "tiny32_sparse50",
        image_hw: 32,
        batch: 1,
        density: 0.50,
    },
    BreakdownCase {
        label: "viz224_sparse20",
        image_hw: 224,
        batch: 1,
        density: 0.20,
    },
];

fn main() -> Result<()> {
    let iters = env_usize("BURN_JEPA_BREAKDOWN_ITERS", 5);
    let warmup = env_usize("BURN_JEPA_BREAKDOWN_WARMUP", 2);
    let stream_frames = env_usize("BURN_JEPA_STREAM_FRAMES", 32);
    let stream_burst = env_usize("BURN_JEPA_STREAM_BURST", 4).max(1);
    let high_every = env_usize("BURN_JEPA_STREAM_HIGH_EVERY", 8).max(1) as u64;

    #[cfg(feature = "ndarray")]
    run_dense_backend::<burn::backend::NdArray<f32>, _>(
        "ndarray",
        || Default::default(),
        iters,
        warmup,
        stream_frames,
        stream_burst,
        high_every,
    )?;

    #[cfg(all(any(feature = "wgpu", feature = "webgpu"), not(feature = "cuda")))]
    run_dense_backend::<burn::backend::Wgpu<f32, i32>, _>(
        "wgpu_dense_patch_embed",
        || Default::default(),
        iters,
        warmup,
        stream_frames,
        stream_burst,
        high_every,
    )?;

    #[cfg(feature = "sparse-patchify-wgpu")]
    run_sparse_patchify_wgpu(iters, warmup, stream_frames, stream_burst, high_every)?;

    #[cfg(feature = "cuda")]
    {
        if let Err(reason) =
            burn_jepa::runtime::cuda_runtime_preflight(burn_jepa::runtime::CUDA_TRAIN_FORCE_ENV)
        {
            eprintln!("skipping cuda_dense_patch_embed: {reason}");
        } else {
            run_dense_backend::<burn::backend::Cuda<f32, i32>, _>(
                "cuda_dense_patch_embed",
                || Default::default(),
                iters,
                warmup,
                stream_frames,
                stream_burst,
                high_every,
            )?;
        }
    }

    #[cfg(feature = "sparse-patchify-cuda")]
    {
        if let Err(reason) =
            burn_jepa::runtime::cuda_runtime_preflight(burn_jepa::runtime::CUDA_TRAIN_FORCE_ENV)
        {
            eprintln!("skipping cuda_sparse_patchify: {reason}");
        } else {
            run_sparse_patchify_cuda(iters, warmup, stream_frames, stream_burst, high_every)?;
        }
    }

    Ok(())
}

fn run_dense_backend<B, MakeDevice>(
    name: &str,
    make_device: MakeDevice,
    iters: usize,
    warmup: usize,
    stream_frames: usize,
    stream_burst: usize,
    high_every: u64,
) -> Result<()>
where
    B: Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    println!("\nbackend: {name}");
    print_direct_header();
    for case in CASES {
        let device = make_device();
        let mut pipeline = make_pipeline::<B>(&case, &device)?;
        let mask = sparse_mask(&pipeline, case.density);
        let mask_batch = mask_batch::<B>(&pipeline, &mask, &device)?;
        let image = image::<B>(&case, &device);

        run_direct_nodes(
            name,
            case,
            "dense_patch_embed",
            &mut pipeline,
            image,
            mask_batch,
            iters,
            warmup,
        )?;
    }

    print_stream_header();
    for case in CASES {
        let device = make_device();
        let pipeline = make_pipeline::<B>(&case, &device)?;
        run_dense_stream(
            name,
            case,
            "dense_patch_embed",
            pipeline,
            &device,
            stream_frames,
            stream_burst,
            high_every,
        )?;
    }
    Ok(())
}

#[cfg(feature = "sparse-patchify-wgpu")]
fn run_sparse_patchify_wgpu(
    iters: usize,
    warmup: usize,
    stream_frames: usize,
    stream_burst: usize,
    high_every: u64,
) -> Result<()> {
    type B = burn_flex_gmm::wgpu::DefaultWgpuBackend;
    let name = "wgpu_sparse_patchify";
    println!("\nbackend: {name}");
    print_direct_header();
    for case in CASES {
        let device = Default::default();
        let mut pipeline = make_pipeline::<B>(&case, &device)?;
        let mask = sparse_mask(&pipeline, case.density);
        let mask_batch = mask_batch::<B>(&pipeline, &mask, &device)?;
        let plan = SparsePatchifyBatchPlan::new(mask_batch, pipeline.grid(), &device)?;
        let image = image::<B>(&case, &device);
        run_direct_sparse_wgpu(name, case, &mut pipeline, image, plan, iters, warmup)?;
    }

    print_stream_header();
    for case in CASES {
        let device = Default::default();
        let pipeline = make_pipeline::<B>(&case, &device)?;
        run_sparse_wgpu_stream(
            name,
            case,
            pipeline,
            stream_frames,
            stream_burst,
            high_every,
        )?;
    }
    Ok(())
}

#[cfg(feature = "sparse-patchify-cuda")]
fn run_sparse_patchify_cuda(
    iters: usize,
    warmup: usize,
    stream_frames: usize,
    stream_burst: usize,
    high_every: u64,
) -> Result<()> {
    type B = burn_flex_gmm::cuda::DefaultCudaBackend;
    let name = "cuda_sparse_patchify";
    println!("\nbackend: {name}");
    print_direct_header();
    for case in CASES {
        let device = Default::default();
        let mut pipeline = make_pipeline::<B>(&case, &device)?;
        let mask = sparse_mask(&pipeline, case.density);
        let mask_batch = mask_batch::<B>(&pipeline, &mask, &device)?;
        let plan = SparsePatchifyBatchPlan::new(mask_batch, pipeline.grid(), &device)?;
        let image = image::<B>(&case, &device);
        run_direct_sparse_cuda(name, case, &mut pipeline, image, plan, iters, warmup)?;
    }

    print_stream_header();
    for case in CASES {
        let device = Default::default();
        let pipeline = make_pipeline::<B>(&case, &device)?;
        run_sparse_cuda_stream(
            name,
            case,
            pipeline,
            stream_frames,
            stream_burst,
            high_every,
        )?;
    }
    Ok(())
}

fn make_pipeline<B: Backend>(
    case: &BreakdownCase,
    device: &B::Device,
) -> Result<FeatureFramePipeline<B>> {
    let mut config = VJepaConfig::tiny_for_tests();
    config.image_size = case.image_hw;
    let jepa = VJepa2_1Model::<B>::new(&config, device);
    let anyup = AnyUp::<B>::new(AnyUpConfig::tiny_for_tests(), device)?;
    FeatureFramePipeline::<B>::new(
        jepa,
        anyup,
        &config,
        FeatureFramePipelineConfig {
            anyup_q_chunk_size: env_q_chunk("BURN_JEPA_ANYUP_Q_CHUNK")
                .unwrap_or(FeatureFramePipelineConfig::default().anyup_q_chunk_size),
            pca_update: env_pca_update_config(),
            ..FeatureFramePipelineConfig::default()
        },
        case.batch,
        [case.image_hw, case.image_hw],
        device,
    )
}

fn sparse_mask<B: Backend>(pipeline: &FeatureFramePipeline<B>, density: f32) -> SparseTokenMask {
    let keep = ((pipeline.grid().len() as f32) * density)
        .ceil()
        .max(1.0)
        .min(pipeline.grid().len() as f32) as usize;
    SparseTokenMask::evenly_spaced(pipeline.grid().len(), keep)
}

fn mask_batch<B: Backend>(
    pipeline: &FeatureFramePipeline<B>,
    mask: &SparseTokenMask,
    device: &B::Device,
) -> Result<SparseMaskBatch<B>> {
    let rows = (0..pipeline.batch())
        .map(|_| mask.indices().to_vec())
        .collect::<Vec<_>>();
    SparseMaskBatch::from_rows(rows, pipeline.grid().len(), device)
}

fn image<B: Backend>(case: &BreakdownCase, device: &B::Device) -> Tensor<B, 4> {
    Tensor::<B, 4>::ones([case.batch, 3, case.image_hw, case.image_hw], device)
}

fn frame_input<B: Backend>(
    sequence: u64,
    mask: SparseTokenMask,
    image: Tensor<B, 4>,
) -> FeatureFrameInput<B> {
    FeatureFrameInput {
        id: FrameId {
            stream_id: 0,
            sequence,
            capture_time_nanos: sequence,
        },
        image,
        mask,
    }
}

fn run_direct_nodes<B: Backend>(
    backend: &str,
    case: BreakdownCase,
    path: &str,
    pipeline: &mut FeatureFramePipeline<B>,
    image: Tensor<B, 4>,
    mask_batch: SparseMaskBatch<B>,
    iters: usize,
    warmup: usize,
) -> Result<()> {
    for (request_name, request) in [
        ("low_res", FeatureFrameRequest::low_res()),
        ("low_high_pca", FeatureFrameRequest::full_pca()),
        ("low_high_features", FeatureFrameRequest::full()),
    ] {
        for _ in 0..warmup {
            let measured = pipeline.step_image_with_mask_batch_nodes_measured(
                image.clone(),
                mask_batch.clone(),
                request,
                FeatureFrameMeasureConfig::enabled_with_backend_sync(),
            )?;
            B::sync(pipeline.device())?;
            std::hint::black_box(measured.output.has_high_res_pca());
        }
        let mut stages = StageSums::default();
        for _ in 0..iters {
            let measured = pipeline.step_image_with_mask_batch_nodes_measured(
                image.clone(),
                mask_batch.clone(),
                request,
                FeatureFrameMeasureConfig::enabled_with_backend_sync(),
            )?;
            B::sync(pipeline.device())?;
            stages.add(&measured.metrics);
            std::hint::black_box(measured.output.has_low_res_pca());
            std::hint::black_box(measured.output.has_high_res_pca());
        }
        print_direct_row(
            backend,
            case,
            path,
            request_name,
            mask_batch.len(),
            pipeline.grid().len(),
            &stages,
            iters,
        );
    }
    Ok(())
}

#[cfg(feature = "sparse-patchify-wgpu")]
fn run_direct_sparse_wgpu(
    backend: &str,
    case: BreakdownCase,
    pipeline: &mut FeatureFramePipeline<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    image: Tensor<burn_flex_gmm::wgpu::DefaultWgpuBackend, 4>,
    plan: SparsePatchifyBatchPlan<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    iters: usize,
    warmup: usize,
) -> Result<()> {
    for (request_name, request) in [
        ("low_res", FeatureFrameRequest::low_res()),
        ("low_high_pca", FeatureFrameRequest::full_pca()),
        ("low_high_features", FeatureFrameRequest::full()),
    ] {
        for _ in 0..warmup {
            let measured = pipeline.step_image_with_sparse_patchify_plan_wgpu_nodes_measured(
                image.clone(),
                &plan,
                request,
                FeatureFrameMeasureConfig::enabled_with_backend_sync(),
            )?;
            burn_flex_gmm::wgpu::DefaultWgpuBackend::sync(pipeline.device())?;
            std::hint::black_box(measured.output.has_high_res_pca());
        }
        let mut stages = StageSums::default();
        for _ in 0..iters {
            let measured = pipeline.step_image_with_sparse_patchify_plan_wgpu_nodes_measured(
                image.clone(),
                &plan,
                request,
                FeatureFrameMeasureConfig::enabled_with_backend_sync(),
            )?;
            burn_flex_gmm::wgpu::DefaultWgpuBackend::sync(pipeline.device())?;
            stages.add(&measured.metrics);
            std::hint::black_box(measured.output.has_low_res_pca());
            std::hint::black_box(measured.output.has_high_res_pca());
        }
        print_direct_row(
            backend,
            case,
            "sparse_patchify",
            request_name,
            plan.mask.len(),
            pipeline.grid().len(),
            &stages,
            iters,
        );
    }
    Ok(())
}

#[cfg(feature = "sparse-patchify-cuda")]
fn run_direct_sparse_cuda(
    backend: &str,
    case: BreakdownCase,
    pipeline: &mut FeatureFramePipeline<burn_flex_gmm::cuda::DefaultCudaBackend>,
    image: Tensor<burn_flex_gmm::cuda::DefaultCudaBackend, 4>,
    plan: SparsePatchifyBatchPlan<burn_flex_gmm::cuda::DefaultCudaBackend>,
    iters: usize,
    warmup: usize,
) -> Result<()> {
    for (request_name, request) in [
        ("low_res", FeatureFrameRequest::low_res()),
        ("low_high_pca", FeatureFrameRequest::full_pca()),
        ("low_high_features", FeatureFrameRequest::full()),
    ] {
        for _ in 0..warmup {
            let measured = pipeline.step_image_with_sparse_patchify_plan_cuda_nodes_measured(
                image.clone(),
                &plan,
                request,
                FeatureFrameMeasureConfig::enabled_with_backend_sync(),
            )?;
            burn_flex_gmm::cuda::DefaultCudaBackend::sync(pipeline.device())?;
            std::hint::black_box(measured.output.has_high_res_pca());
        }
        let mut stages = StageSums::default();
        for _ in 0..iters {
            let measured = pipeline.step_image_with_sparse_patchify_plan_cuda_nodes_measured(
                image.clone(),
                &plan,
                request,
                FeatureFrameMeasureConfig::enabled_with_backend_sync(),
            )?;
            burn_flex_gmm::cuda::DefaultCudaBackend::sync(pipeline.device())?;
            stages.add(&measured.metrics);
            std::hint::black_box(measured.output.has_low_res_pca());
            std::hint::black_box(measured.output.has_high_res_pca());
        }
        print_direct_row(
            backend,
            case,
            "sparse_patchify",
            request_name,
            plan.mask.len(),
            pipeline.grid().len(),
            &stages,
            iters,
        );
    }
    Ok(())
}

fn run_dense_stream<B: Backend>(
    backend: &str,
    case: BreakdownCase,
    path: &str,
    pipeline: FeatureFramePipeline<B>,
    device: &B::Device,
    stream_frames: usize,
    stream_burst: usize,
    high_every: u64,
) -> Result<()> {
    let mut stream = make_stream(pipeline, high_every)?;
    drive_stream(
        backend,
        case,
        path,
        device,
        &mut stream,
        |stream| stream.process_next_ready_nodes(),
        stream_frames,
        stream_burst,
        high_every,
    )
}

#[cfg(feature = "sparse-patchify-wgpu")]
fn run_sparse_wgpu_stream(
    backend: &str,
    case: BreakdownCase,
    pipeline: FeatureFramePipeline<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    stream_frames: usize,
    stream_burst: usize,
    high_every: u64,
) -> Result<()> {
    let device = pipeline.device().clone();
    let mut stream = make_stream(pipeline, high_every)?;
    drive_stream(
        backend,
        case,
        "sparse_patchify",
        &device,
        &mut stream,
        |stream| stream.process_next_ready_sparse_patchify_wgpu_nodes(),
        stream_frames,
        stream_burst,
        high_every,
    )
}

#[cfg(feature = "sparse-patchify-cuda")]
fn run_sparse_cuda_stream(
    backend: &str,
    case: BreakdownCase,
    pipeline: FeatureFramePipeline<burn_flex_gmm::cuda::DefaultCudaBackend>,
    stream_frames: usize,
    stream_burst: usize,
    high_every: u64,
) -> Result<()> {
    let device = pipeline.device().clone();
    let mut stream = make_stream(pipeline, high_every)?;
    drive_stream(
        backend,
        case,
        "sparse_patchify",
        &device,
        &mut stream,
        |stream| stream.process_next_ready_sparse_patchify_cuda_nodes(),
        stream_frames,
        stream_burst,
        high_every,
    )
}

fn make_stream<B: Backend>(
    pipeline: FeatureFramePipeline<B>,
    high_every: u64,
) -> Result<FeatureFrameStream<B>> {
    FeatureFrameStream::new(
        pipeline,
        FrameStreamConfig {
            queue_capacity: 1,
            batch_size: 1,
            backpressure: FrameQueuePolicy::OverwriteNewest,
            schedule: FeatureFrameSchedule {
                low_res_pca_every: Some(1),
                high_res_pca_every: Some(high_every),
            },
            measurement: FeatureFrameMeasureConfig::enabled_with_backend_sync(),
            ..FrameStreamConfig::default()
        },
    )
}

fn drive_stream<B, Process>(
    backend: &str,
    case: BreakdownCase,
    path: &str,
    device: &B::Device,
    stream: &mut FeatureFrameStream<B>,
    mut process: Process,
    stream_frames: usize,
    stream_burst: usize,
    high_every: u64,
) -> Result<()>
where
    B: Backend,
    Process:
        FnMut(&mut FeatureFrameStream<B>) -> Result<Option<burn_jepa::FeatureFrameStreamOutput<B>>>,
{
    let mask = sparse_mask(stream.pipeline(), case.density);
    let keep = mask.len();
    let image = Tensor::<B, 4>::ones([1, 3, case.image_hw, case.image_hw], device);
    let mut summary = StreamSummary::default();
    let mut sequence = 1u64;
    while sequence <= stream_frames as u64 {
        for _ in 0..stream_burst {
            if sequence > stream_frames as u64 {
                break;
            }
            let report = stream.enqueue(frame_input(sequence, mask.clone(), image.clone()))?;
            summary.input_frames += 1;
            if report.dropped_frame.is_some() {
                summary.dropped_on_enqueue += 1;
            }
            if report.overwritten_frame.is_some() {
                summary.overwritten_on_enqueue += 1;
            }
            sequence += 1;
        }
        if let Some(output) = process(stream)? {
            B::sync(device)?;
            summary.add_output(&output);
            std::hint::black_box(output.output.has_low_res_pca());
            std::hint::black_box(output.output.has_high_res_pca());
        }
    }
    while stream.can_process_batch() {
        if let Some(output) = process(stream)? {
            B::sync(device)?;
            summary.add_output(&output);
        }
    }
    let stats = stream.stats();
    print_stream_row(
        backend,
        case,
        path,
        keep,
        stream.pipeline().grid().len(),
        high_every,
        stream_burst,
        &summary,
        stats.dropped_frames,
        stats.overwritten_frames,
    );
    Ok(())
}

fn print_direct_header() {
    println!(
        "| backend | case | path | request | sparse tokens | avg total | encode | cache update | token view | pca update | low-res pca | anyup context | anyup decode | high-res pca | e2e fps |"
    );
    println!("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
}

fn print_stream_header() {
    println!(
        "| backend | case | path | inputs | emitted | low outputs | high outputs | queue dropped | overwritten | burst | high every | avg total | encode | pca update | low-res pca | anyup decode | e2e emitted fps | low-res fps | high-res fps |"
    );
    println!(
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    );
}

fn print_direct_row(
    backend: &str,
    case: BreakdownCase,
    path: &str,
    request_name: &str,
    keep: usize,
    dense: usize,
    stages: &StageSums,
    iters: usize,
) {
    let avg = stages.avg(iters);
    let fps = fps(case.batch, avg.total_us);
    println!(
        "| {backend} | {} | {path} | {request_name} | {keep} / {dense} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.1} |",
        case.label,
        fmt_ms(avg.total_us),
        fmt_ms(avg.encode_us),
        fmt_ms(avg.cache_update_us),
        fmt_ms(avg.token_view_us),
        fmt_ms(avg.pca_update_us),
        fmt_ms(avg.low_res_pca_project_us),
        fmt_ms(avg.anyup_context_us),
        fmt_ms(avg.anyup_decode_us),
        fmt_ms(avg.pca_project_us),
        fps,
    );
}

#[allow(clippy::too_many_arguments)]
fn print_stream_row(
    backend: &str,
    case: BreakdownCase,
    path: &str,
    keep: usize,
    dense: usize,
    high_every: u64,
    stream_burst: usize,
    summary: &StreamSummary,
    dropped_total: usize,
    overwritten_total: usize,
) {
    let avg_total = summary.total.avg_us(summary.emitted_batches);
    let encode_avg = summary.encode.avg_nonzero_us();
    let pca_update_avg = summary.pca_update.avg_nonzero_us();
    let low_pca_avg = summary.low_res_pca.avg_nonzero_us();
    let anyup_decode_avg = summary.anyup_decode.avg_nonzero_us();
    println!(
        "| {backend} | {} | {path} ({keep}/{dense}) | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.1} | {:.1} | {:.1} |",
        case.label,
        summary.input_frames,
        summary.emitted_frames,
        summary.low_res_outputs,
        summary.high_res_outputs,
        dropped_total.max(summary.dropped_on_enqueue),
        overwritten_total.max(summary.overwritten_on_enqueue),
        stream_burst,
        high_every,
        fmt_ms(avg_total),
        fmt_ms(encode_avg),
        fmt_ms(pca_update_avg),
        fmt_ms(low_pca_avg),
        fmt_ms(anyup_decode_avg),
        summary.emitted_fps(),
        summary.low_res_fps(),
        summary.high_res_fps(),
    );
}

#[derive(Default)]
struct StreamSummary {
    input_frames: usize,
    emitted_batches: usize,
    emitted_frames: usize,
    low_res_outputs: usize,
    high_res_outputs: usize,
    dropped_on_enqueue: usize,
    overwritten_on_enqueue: usize,
    total: StageCounter,
    encode: StageCounter,
    pca_update: StageCounter,
    low_res_pca: StageCounter,
    anyup_decode: StageCounter,
}

impl StreamSummary {
    fn add_output<B: Backend>(&mut self, output: &burn_jepa::FeatureFrameStreamOutput<B>) {
        self.emitted_batches += 1;
        self.emitted_frames += output.frame_ids.len();
        self.low_res_outputs += usize::from(output.request.low_res_pca);
        self.high_res_outputs += usize::from(output.request.high_res_pca);
        self.total.add(output.metrics.total_us);
        self.encode.add(output.metrics.encode_us);
        self.pca_update.add(output.metrics.pca_update_us);
        self.low_res_pca.add(output.metrics.low_res_pca_project_us);
        self.anyup_decode.add(output.metrics.anyup_decode_us);
    }

    fn emitted_fps(&self) -> f64 {
        self.total.fps(self.emitted_frames)
    }

    fn low_res_fps(&self) -> f64 {
        self.low_res_pca.fps(self.low_res_outputs)
    }

    fn high_res_fps(&self) -> f64 {
        self.anyup_decode.fps(self.high_res_outputs)
    }
}

#[derive(Default)]
struct StageCounter {
    total_us: u128,
    nonzero_count: usize,
}

impl StageCounter {
    fn add(&mut self, us: u64) {
        self.total_us += us as u128;
        if us > 0 {
            self.nonzero_count += 1;
        }
    }

    fn avg_us(&self, count: usize) -> u64 {
        avg_u64(self.total_us, count.max(1) as u128)
    }

    fn avg_nonzero_us(&self) -> u64 {
        self.avg_us(self.nonzero_count.max(1))
    }

    fn fps(&self, frames: usize) -> f64 {
        if self.total_us == 0 {
            0.0
        } else {
            frames as f64 * 1_000_000.0 / self.total_us as f64
        }
    }
}

#[derive(Default)]
struct StageSums {
    encode_us: u128,
    cache_update_us: u128,
    token_view_us: u128,
    anyup_context_us: u128,
    anyup_decode_us: u128,
    low_res_pca_project_us: u128,
    pca_update_us: u128,
    pca_online_us: u128,
    pca_project_us: u128,
    total_us: u128,
}

impl StageSums {
    fn add(&mut self, metrics: &FeatureFrameMetrics) {
        self.encode_us += metrics.encode_us as u128;
        self.cache_update_us += metrics.cache_update_us as u128;
        self.token_view_us += metrics.token_view_us as u128;
        self.anyup_context_us += metrics.anyup_context_us as u128;
        self.anyup_decode_us += metrics.anyup_decode_us as u128;
        self.low_res_pca_project_us += metrics.low_res_pca_project_us as u128;
        self.pca_update_us += metrics.pca_update_us as u128;
        self.pca_online_us += metrics.pca_online_us as u128;
        self.pca_project_us += metrics.pca_project_us as u128;
        self.total_us += metrics.total_us as u128;
    }

    fn avg(&self, iters: usize) -> FeatureFrameMetrics {
        let iters = iters.max(1) as u128;
        FeatureFrameMetrics {
            encode_us: avg_u64(self.encode_us, iters),
            cache_update_us: avg_u64(self.cache_update_us, iters),
            token_view_us: avg_u64(self.token_view_us, iters),
            anyup_context_us: avg_u64(self.anyup_context_us, iters),
            anyup_decode_us: avg_u64(self.anyup_decode_us, iters),
            low_res_pca_project_us: avg_u64(self.low_res_pca_project_us, iters),
            pca_update_us: avg_u64(self.pca_update_us, iters),
            pca_online_us: avg_u64(self.pca_online_us, iters),
            pca_project_us: avg_u64(self.pca_project_us, iters),
            total_us: avg_u64(self.total_us, iters),
            ..FeatureFrameMetrics::default()
        }
    }
}

fn avg_u64(total: u128, count: u128) -> u64 {
    (total / count.max(1)).min(u64::MAX as u128) as u64
}

fn fps(frames: usize, us: u64) -> f64 {
    if us == 0 {
        0.0
    } else {
        frames as f64 * 1_000_000.0 / us as f64
    }
}

fn fmt_ms(us: u64) -> String {
    format!("{:.3} ms", us as f64 / 1000.0)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_q_chunk(name: &str) -> Option<Option<usize>> {
    std::env::var(name).ok().and_then(|value| {
        let trimmed = value.trim();
        if trimmed.eq_ignore_ascii_case("none") || trimmed == "0" {
            Some(None)
        } else {
            trimmed.parse::<usize>().ok().map(Some)
        }
    })
}

fn env_pca_update_config() -> FeaturePcaUpdateConfig {
    let every = env_usize("BURN_JEPA_PCA_UPDATE_EVERY", 0);
    if every == 0 {
        FeaturePcaUpdateConfig::disabled()
    } else {
        let sample_window_frames = env_usize("BURN_JEPA_PCA_SAMPLE_WINDOW", every.max(2)).max(1);
        FeaturePcaUpdateConfig {
            mode: FeaturePcaUpdateMode::RollingOja,
            every_n_frames: every as u64,
            warmup_frames: env_usize("BURN_JEPA_PCA_UPDATE_WARMUP", 0) as u64,
            min_tokens_per_update: env_usize("BURN_JEPA_PCA_UPDATE_MIN_TOKENS", 1),
            iterations_per_update: env_usize("BURN_JEPA_PCA_UPDATE_ITERS", 1),
            sample_window_frames,
            min_sample_frames: env_usize("BURN_JEPA_PCA_MIN_SAMPLE_FRAMES", sample_window_frames),
        }
    }
}
