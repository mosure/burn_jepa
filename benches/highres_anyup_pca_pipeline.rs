use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_jepa::{
    AnyUp, AnyUpConfig, FeatureFrameRequest, FeatureFrameViewerConfig, FeaturePcaConfig,
    FeaturePcaDisplayMode, FeaturePcaProjector, FeaturePcaUpdateConfig,
    InterframeJepaFeatureMemory, InterframeJepaFeatureMemoryConfig, PatchDiffRefreshConfig,
    PatchDiffRefreshState, SparseJepaAnyUpPcaFrameId, SparseJepaAnyUpPcaFrameInput,
    SparseJepaAnyUpPcaMeasurementConfig, SparseJepaAnyUpPcaPipeline,
    SparseJepaAnyUpPcaPipelineConfig, SparseJepaAnyUpPcaStream, SparseJepaAnyUpPcaStreamConfig,
    SparseTokenMask, TokenGridShape, VJepa2_1Model, VJepaConfig, jepa_feature_tokens_to_nchw,
    patch_diff_sparsity_config,
};
#[cfg(any(feature = "sparse-patchify-wgpu", feature = "sparse-patchify-cuda"))]
use burn_jepa::{SparseMaskBatch, SparsePatchifyBatchPlan};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

#[derive(Clone, Copy)]
struct HighResCase {
    label: &'static str,
    image_hw: usize,
    grid_hw: usize,
    embed_dim: usize,
    batch: usize,
    q_chunk_size: Option<usize>,
    anyup: AnyUpBenchConfig,
    large: bool,
}

#[derive(Clone, Copy)]
enum AnyUpBenchConfig {
    Tiny,
    Default,
}

#[derive(Clone, Copy)]
struct E2ePipelineCase {
    label: &'static str,
    image_hw: usize,
    density: f32,
    q_chunk_size: Option<usize>,
    pca_update_every: Option<u64>,
    measurement: SparseJepaAnyUpPcaMeasurementConfig,
}

fn bench_1024_enabled() -> bool {
    std::env::var("BURN_JEPA_BENCH_1024").ok().as_deref() == Some("1")
}

fn viewer_e2e_pipeline_cases() -> Vec<E2ePipelineCase> {
    let mut cases = vec![
        E2ePipelineCase {
            label: "tiny32_sparse50",
            image_hw: 32,
            density: 0.50,
            q_chunk_size: Some(1),
            pca_update_every: None,
            measurement: SparseJepaAnyUpPcaMeasurementConfig::disabled(),
        },
        E2ePipelineCase {
            label: "viewer256_sparse100",
            image_hw: 256,
            density: 1.0,
            q_chunk_size: Some(16),
            pca_update_every: Some(16),
            measurement: SparseJepaAnyUpPcaMeasurementConfig::enabled(),
        },
        E2ePipelineCase {
            label: "viewer512_sparse100",
            image_hw: 512,
            density: 1.0,
            q_chunk_size: Some(16),
            pca_update_every: Some(16),
            measurement: SparseJepaAnyUpPcaMeasurementConfig::enabled(),
        },
    ];
    if bench_1024_enabled() {
        cases.push(E2ePipelineCase {
            label: "viewer1024_sparse100",
            image_hw: 1024,
            density: 1.0,
            q_chunk_size: Some(16),
            pca_update_every: Some(16),
            measurement: SparseJepaAnyUpPcaMeasurementConfig::enabled(),
        });
    }
    cases
}

fn viewer_cache_sweep_image_sizes() -> Vec<usize> {
    let mut sizes = vec![256, 512];
    if bench_1024_enabled() {
        sizes.push(1024);
    }
    sizes
}

const CASES: [HighResCase; 4] = [
    HighResCase {
        label: "tiny32_grid2_c32",
        image_hw: 32,
        grid_hw: 2,
        embed_dim: 32,
        batch: 1,
        q_chunk_size: Some(1),
        anyup: AnyUpBenchConfig::Tiny,
        large: false,
    },
    HighResCase {
        label: "viz224_grid14_c128",
        image_hw: 224,
        grid_hw: 14,
        embed_dim: 128,
        batch: 1,
        q_chunk_size: Some(16),
        anyup: AnyUpBenchConfig::Tiny,
        large: false,
    },
    HighResCase {
        label: "vjepa224_grid14_c768",
        image_hw: 224,
        grid_hw: 14,
        embed_dim: 768,
        batch: 1,
        q_chunk_size: Some(16),
        anyup: AnyUpBenchConfig::Default,
        large: true,
    },
    HighResCase {
        label: "vjepa384_grid24_c768",
        image_hw: 384,
        grid_hw: 24,
        embed_dim: 768,
        batch: 1,
        q_chunk_size: Some(16),
        anyup: AnyUpBenchConfig::Default,
        large: true,
    },
];

fn include_large_cases() -> bool {
    std::env::var("BURN_JEPA_HIGHRES_BENCH_LARGE").is_ok_and(|value| value != "0")
}

fn bench_patch_diff_refresh_policy(c: &mut Criterion) {
    let mut group = c.benchmark_group("patch_diff_refresh_policy");
    for grid_hw in [16usize, 32] {
        let grid = TokenGridShape::new(1, grid_hw, grid_hw);
        group.throughput(Throughput::Elements((grid.len() * 32) as u64));
        for (label, refresh) in [
            ("instant_only", PatchDiffRefreshConfig::disabled()),
            (
                "subthreshold_av1_like",
                PatchDiffRefreshConfig {
                    subthreshold_decay: 0.92,
                    subthreshold_trigger: 1.0,
                    subthreshold_max_density: 0.08,
                    age_refresh_enabled: false,
                    blue_noise_enabled: false,
                    max_extra_density: 0.08,
                    ..PatchDiffRefreshConfig::default()
                },
            ),
            (
                "subthreshold_age_blue_noise",
                PatchDiffRefreshConfig::default(),
            ),
        ] {
            group.bench_function(format!("grid{grid_hw}_{label}"), |bench| {
                bench.iter_batched(
                    || {
                        let config = FeatureFrameViewerConfig {
                            context_density: 0.30,
                            min_context_density: 0.0,
                            patch_diff_threshold: 0.03,
                            patch_diff_refresh: refresh,
                            ..FeatureFrameViewerConfig::default()
                        };
                        let sparsity = patch_diff_sparsity_config(&config, grid);
                        let state = PatchDiffRefreshState::default();
                        let scores = vec![0.0f32; grid.len()];
                        (config, sparsity, state, scores)
                    },
                    |(config, sparsity, mut state, mut scores)| {
                        for frame in 0..32usize {
                            for (index, score) in scores.iter_mut().enumerate() {
                                let phase = (index + frame * 7) % 97;
                                *score = if phase < 3 {
                                    0.05
                                } else if phase < 23 {
                                    0.012
                                } else {
                                    0.0
                                };
                            }
                            let masks = state
                                .masks_from_scores(
                                    black_box(scores.clone()),
                                    grid,
                                    &sparsity,
                                    &config,
                                )
                                .expect("patch-diff refresh mask");
                            black_box(masks.write_mask.len());
                        }
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

fn anyup_config(case: AnyUpBenchConfig) -> AnyUpConfig {
    match case {
        AnyUpBenchConfig::Tiny => AnyUpConfig::tiny_for_tests(),
        AnyUpBenchConfig::Default => AnyUpConfig::default(),
    }
}

fn token_indices<B: Backend>(
    batch: usize,
    dense_tokens: usize,
    keep: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let values = (0..batch)
        .flat_map(|row| {
            (0..keep).map(move |index| ((index * dense_tokens / keep) + row) % dense_tokens)
        })
        .map(|index| index as i64)
        .collect::<Vec<_>>();
    Tensor::<B, 2, Int>::from_data(TensorData::new(values, [batch, keep]), device)
}

fn bench_pca_projection<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut group = c.benchmark_group(format!("highres_pca_project_{backend_name}"));
    for case in CASES {
        if case.large && !include_large_cases() {
            continue;
        }
        group.throughput(Throughput::Elements(
            (case.batch * case.image_hw * case.image_hw) as u64,
        ));
        for (mode_label, display_mode) in [
            ("semantic_rgb", FeaturePcaDisplayMode::SemanticRgb),
            ("signed_unit", FeaturePcaDisplayMode::SignedUnit),
        ] {
            group.bench_function(format!("{}_{}", case.label, mode_label), |bench| {
                bench.iter_batched(
                    || {
                        let device = make_device();
                        let projector = FeaturePcaProjector::<B>::identity(
                            case.embed_dim,
                            FeaturePcaConfig {
                                display_mode,
                                ..FeaturePcaConfig::default()
                            },
                            &device,
                        )
                        .expect("pca projector");
                        let features = Tensor::<B, 4>::ones(
                            [case.batch, case.embed_dim, case.image_hw, case.image_hw],
                            &device,
                        );
                        (device, projector, features)
                    },
                    |(device, projector, features)| {
                        let projected = projector
                            .project_nchw_display(black_box(features))
                            .expect("pca project");
                        B::sync(&device).expect("sync PCA backend");
                        black_box(projected);
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

fn bench_semantic_pca_stats_update<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut group = c.benchmark_group(format!("highres_semantic_pca_stats_update_{backend_name}"));
    for case in CASES {
        if case.large && !include_large_cases() {
            continue;
        }
        group.throughput(Throughput::Elements((case.grid_hw * case.grid_hw) as u64));
        group.bench_function(case.label, |bench| {
            bench.iter_batched(
                || {
                    let device = make_device();
                    let projector = FeaturePcaProjector::<B>::identity(
                        case.embed_dim,
                        FeaturePcaConfig {
                            display_mode: FeaturePcaDisplayMode::SemanticRgb,
                            online_learning_rate: 0.05,
                            mean_momentum: 0.05,
                            display_momentum: 0.05,
                            ..FeaturePcaConfig::default()
                        },
                        &device,
                    )
                    .expect("pca projector");
                    let token_count = case.grid_hw * case.grid_hw;
                    let tokens = Tensor::<B, 3>::ones([1, token_count, case.embed_dim], &device);
                    let observed = Tensor::<B, 2>::ones([1, token_count], &device);
                    (device, projector, tokens, observed)
                },
                |(device, mut projector, tokens, observed)| {
                    projector
                        .update_rolling_masked_tokens(black_box(tokens), black_box(observed))
                        .expect("semantic PCA stats update");
                    B::sync(&device).expect("sync semantic PCA update backend");
                    black_box((projector.components(), projector.display_spread()));
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_pca_basis_update<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut group = c.benchmark_group(format!("highres_pca_basis_update_{backend_name}"));
    for case in CASES {
        if case.large && !include_large_cases() {
            continue;
        }
        for batch in [1usize, 4, 8] {
            group.throughput(Throughput::Elements(
                (batch * case.grid_hw * case.grid_hw) as u64,
            ));
            group.bench_function(format!("{}_batch{batch}", case.label), |bench| {
                bench.iter_batched(
                    || {
                        let device = make_device();
                        let projector = FeaturePcaProjector::<B>::identity(
                            case.embed_dim,
                            FeaturePcaConfig {
                                online_learning_rate: 0.05,
                                mean_momentum: 0.05,
                                ..FeaturePcaConfig::default()
                            },
                            &device,
                        )
                        .expect("pca projector");
                        let token_count = case.grid_hw * case.grid_hw;
                        let low_res_tokens =
                            Tensor::<B, 3>::ones([batch, token_count, case.embed_dim], &device);
                        let observed = Tensor::<B, 2>::ones([batch, token_count], &device);
                        (device, projector, low_res_tokens, observed)
                    },
                    |(device, mut projector, low_res_tokens, observed)| {
                        projector
                            .update_rolling_masked_tokens(
                                black_box(low_res_tokens),
                                black_box(observed),
                            )
                            .expect("rolling PCA update");
                        B::sync(&device).expect("sync PCA update backend");
                        black_box(projector.components());
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

fn bench_anyup_from_token_cache<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut group = c.benchmark_group(format!("highres_anyup_from_token_cache_{backend_name}"));
    for case in CASES {
        if case.large && !include_large_cases() {
            continue;
        }
        group.throughput(Throughput::Elements(
            (case.batch * case.image_hw * case.image_hw) as u64,
        ));
        group.bench_function(case.label, |bench| {
            bench.iter_batched(
                || {
                    let device = make_device();
                    let grid = TokenGridShape::new(1, case.grid_hw, case.grid_hw);
                    let anyup = AnyUp::<B>::new(anyup_config(case.anyup), &device).expect("anyup");
                    let image = Tensor::<B, 4>::ones(
                        [case.batch, 3, case.image_hw, case.image_hw],
                        &device,
                    );
                    let context = anyup.prepare_image_context(
                        image,
                        Some([case.image_hw, case.image_hw]),
                        [grid.height, grid.width],
                    );
                    let tokens =
                        Tensor::<B, 3>::ones([case.batch, grid.len(), case.embed_dim], &device);
                    let low_res =
                        jepa_feature_tokens_to_nchw(tokens, grid).expect("low-res features");
                    (device, anyup, context, low_res)
                },
                |(device, anyup, context, low_res)| {
                    let output = anyup.upsample_with_context(
                        &context,
                        black_box(low_res),
                        case.q_chunk_size,
                    );
                    B::sync(&device).expect("sync AnyUp backend");
                    black_box(output);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_sparse_update_to_anyup_pca<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut group = c.benchmark_group(format!("highres_sparse_cache_anyup_pca_{backend_name}"));
    let case = CASES[1];
    let grid = TokenGridShape::new(1, case.grid_hw, case.grid_hw);
    for density in [0.1f32, 0.25, 0.5, 1.0] {
        let keep = ((grid.len() as f32) * density).ceil() as usize;
        group.throughput(Throughput::Elements(
            (case.batch * case.image_hw * case.image_hw) as u64,
        ));
        group.bench_function(format!("{}_density_{density:.2}", case.label), |bench| {
            bench.iter_batched(
                || {
                    let device = make_device();
                    let memory = InterframeJepaFeatureMemory::<B>::new(
                        InterframeJepaFeatureMemoryConfig::default(),
                        case.batch,
                        grid,
                        case.embed_dim,
                        &device,
                    )
                    .expect("memory");
                    let indices = token_indices::<B>(case.batch, grid.len(), keep, &device);
                    let tokens = Tensor::<B, 3>::ones([case.batch, keep, case.embed_dim], &device);
                    let anyup = AnyUp::<B>::new(anyup_config(case.anyup), &device).expect("anyup");
                    let image = Tensor::<B, 4>::ones(
                        [case.batch, 3, case.image_hw, case.image_hw],
                        &device,
                    );
                    let context = anyup.prepare_image_context(
                        image,
                        Some([case.image_hw, case.image_hw]),
                        [grid.height, grid.width],
                    );
                    let projector = FeaturePcaProjector::<B>::identity(
                        case.embed_dim,
                        FeaturePcaConfig::default(),
                        &device,
                    )
                    .expect("pca");
                    (device, memory, indices, tokens, anyup, context, projector)
                },
                |(device, mut memory, indices, tokens, anyup, context, projector)| {
                    let cache = memory
                        .update_tokens(black_box(tokens), black_box(indices), grid)
                        .expect("sparse cache update");
                    let low_res = cache.features_nchw().expect("low-res features");
                    let high_res = anyup.upsample_with_context(
                        &context,
                        black_box(low_res),
                        case.q_chunk_size,
                    );
                    let display = projector
                        .project_nchw_display(black_box(high_res))
                        .expect("pca display");
                    B::sync(&device).expect("sync high-res backend");
                    black_box(display);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_tiny_e2e_pipeline_step<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut group = c.benchmark_group(format!("highres_sparse_jepa_anyup_pca_e2e_{backend_name}"));
    for case in viewer_e2e_pipeline_cases() {
        group.throughput(Throughput::Elements((case.image_hw * case.image_hw) as u64));
        group.bench_function(case.label, |bench| {
            bench.iter_batched(
                || {
                    let device = make_device();
                    let mut model_config = VJepaConfig::tiny_for_tests();
                    model_config.image_size = case.image_hw;
                    model_config.num_frames = 2;
                    model_config.tubelet_size = 2;
                    let jepa = VJepa2_1Model::<B>::new(&model_config, &device);
                    let anyup =
                        AnyUp::<B>::new(AnyUpConfig::tiny_for_tests(), &device).expect("anyup");
                    let pipeline = SparseJepaAnyUpPcaPipeline::<B>::new(
                        jepa,
                        anyup,
                        &model_config,
                        SparseJepaAnyUpPcaPipelineConfig {
                            anyup_q_chunk_size: case.q_chunk_size,
                            pca_update: case
                                .pca_update_every
                                .map(FeaturePcaUpdateConfig::rolling_low_res_every)
                                .unwrap_or_else(FeaturePcaUpdateConfig::disabled),
                            measurement: case.measurement,
                            ..SparseJepaAnyUpPcaPipelineConfig::default()
                        },
                        1,
                        [model_config.image_size, model_config.image_size],
                        &device,
                    )
                    .expect("pipeline");
                    let image = Tensor::<B, 4>::ones(
                        [1, 3, model_config.image_size, model_config.image_size],
                        &device,
                    );
                    let keep = ((pipeline.grid().len() as f32) * case.density).ceil() as usize;
                    let mask = SparseTokenMask::evenly_spaced(pipeline.grid().len(), keep);
                    (device, pipeline, image, mask)
                },
                |(device, mut pipeline, image, mask)| {
                    let output = pipeline
                        .step_image_with_mask(black_box(image), black_box(&mask))
                        .expect("pipeline step");
                    B::sync(&device).expect("sync e2e backend");
                    black_box(output.pca_display);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_tiny_jepa_cache_density_sweep<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut group = c.benchmark_group(format!("highres_jepa_cache_density_sweep_{backend_name}"));
    for image_hw in viewer_cache_sweep_image_sizes() {
        for density in [0.50f32, 0.75, 0.85, 0.90, 0.95, 0.98, 1.0] {
            group.throughput(Throughput::Elements((image_hw * image_hw) as u64));
            group.bench_function(format!("tiny{image_hw}_density_{density:.2}"), |bench| {
                bench.iter_batched(
                    || {
                        let device = make_device();
                        let mut model_config = VJepaConfig::tiny_for_tests();
                        model_config.image_size = image_hw;
                        model_config.num_frames = 2;
                        model_config.tubelet_size = 2;
                        let jepa = VJepa2_1Model::<B>::new(&model_config, &device);
                        let anyup =
                            AnyUp::<B>::new(AnyUpConfig::tiny_for_tests(), &device).expect("anyup");
                        let pipeline = SparseJepaAnyUpPcaPipeline::<B>::new(
                            jepa,
                            anyup,
                            &model_config,
                            SparseJepaAnyUpPcaPipelineConfig {
                                pca_update: FeaturePcaUpdateConfig::disabled(),
                                measurement: SparseJepaAnyUpPcaMeasurementConfig::disabled(),
                                ..SparseJepaAnyUpPcaPipelineConfig::default()
                            },
                            1,
                            [model_config.image_size, model_config.image_size],
                            &device,
                        )
                        .expect("pipeline");
                        let image = Tensor::<B, 4>::ones(
                            [1, 3, model_config.image_size, model_config.image_size],
                            &device,
                        );
                        let keep = ((pipeline.grid().len() as f32) * density).ceil() as usize;
                        let mask = SparseTokenMask::evenly_spaced(pipeline.grid().len(), keep);
                        (device, pipeline, image, mask)
                    },
                    |(device, mut pipeline, image, mask)| {
                        let output = pipeline
                            .step_image_with_mask_nodes_measured(
                                black_box(image),
                                black_box(&mask),
                                FeatureFrameRequest::none(),
                            )
                            .expect("JEPA cache density sweep step");
                        B::sync(&device).expect("sync JEPA cache density backend");
                        black_box(output.output.token_cache.features);
                        black_box(output.metrics);
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

#[cfg(feature = "sparse-patchify-wgpu")]
fn bench_tiny_e2e_pipeline_step_sparse_patchify_wgpu(c: &mut Criterion) {
    type B = burn_flex_gmm::wgpu::DefaultWgpuBackend;
    let mut group = c.benchmark_group("highres_sparse_patchify_jepa_anyup_pca_e2e_wgpu");
    let model_config = VJepaConfig::tiny_for_tests();
    group.throughput(Throughput::Elements(
        (model_config.image_size * model_config.image_size) as u64,
    ));
    group.bench_function("tiny32_sparse50", |bench| {
        bench.iter_batched(
            || {
                let device = Default::default();
                let jepa = VJepa2_1Model::<B>::new(&model_config, &device);
                let anyup = AnyUp::<B>::new(AnyUpConfig::tiny_for_tests(), &device).expect("anyup");
                let pipeline = SparseJepaAnyUpPcaPipeline::<B>::new(
                    jepa,
                    anyup,
                    &model_config,
                    SparseJepaAnyUpPcaPipelineConfig::default(),
                    1,
                    [model_config.image_size, model_config.image_size],
                    &device,
                )
                .expect("pipeline");
                let image = Tensor::<B, 4>::ones(
                    [1, 3, model_config.image_size, model_config.image_size],
                    &device,
                );
                let mask = SparseTokenMask::evenly_spaced(pipeline.grid().len(), 2);
                let mask_batch = SparseMaskBatch::uniform(mask, 1, &device).expect("mask batch");
                let patchify_plan =
                    SparsePatchifyBatchPlan::new(mask_batch, pipeline.grid(), &device)
                        .expect("patchify plan");
                (device, pipeline, image, patchify_plan)
            },
            |(device, mut pipeline, image, patchify_plan)| {
                let output = pipeline
                    .step_image_with_sparse_patchify_plan_wgpu_measured(
                        black_box(image),
                        black_box(&patchify_plan),
                        SparseJepaAnyUpPcaMeasurementConfig::disabled(),
                    )
                    .expect("sparse patchify pipeline step");
                B::sync(&device).expect("sync sparse patchify WGPU backend");
                black_box(output.output.pca_display);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

#[cfg(not(feature = "sparse-patchify-wgpu"))]
fn bench_tiny_e2e_pipeline_step_sparse_patchify_wgpu(_c: &mut Criterion) {}

#[cfg(feature = "sparse-patchify-cuda")]
fn bench_tiny_e2e_pipeline_step_sparse_patchify_cuda(c: &mut Criterion) {
    if let Err(reason) =
        burn_jepa::runtime::cuda_runtime_preflight(burn_jepa::runtime::CUDA_TRAIN_FORCE_ENV)
    {
        eprintln!("skipping highres sparse patchify CUDA bench: {reason}");
        return;
    }
    type B = burn_flex_gmm::cuda::DefaultCudaBackend;
    let mut group = c.benchmark_group("highres_sparse_patchify_jepa_anyup_pca_e2e_cuda");
    let model_config = VJepaConfig::tiny_for_tests();
    group.throughput(Throughput::Elements(
        (model_config.image_size * model_config.image_size) as u64,
    ));
    group.bench_function("tiny32_sparse50", |bench| {
        bench.iter_batched(
            || {
                let device = Default::default();
                let jepa = VJepa2_1Model::<B>::new(&model_config, &device);
                let anyup = AnyUp::<B>::new(AnyUpConfig::tiny_for_tests(), &device).expect("anyup");
                let pipeline = SparseJepaAnyUpPcaPipeline::<B>::new(
                    jepa,
                    anyup,
                    &model_config,
                    SparseJepaAnyUpPcaPipelineConfig::default(),
                    1,
                    [model_config.image_size, model_config.image_size],
                    &device,
                )
                .expect("pipeline");
                let image = Tensor::<B, 4>::ones(
                    [1, 3, model_config.image_size, model_config.image_size],
                    &device,
                );
                let mask = SparseTokenMask::evenly_spaced(pipeline.grid().len(), 2);
                let mask_batch = SparseMaskBatch::uniform(mask, 1, &device).expect("mask batch");
                let patchify_plan =
                    SparsePatchifyBatchPlan::new(mask_batch, pipeline.grid(), &device)
                        .expect("patchify plan");
                (device, pipeline, image, patchify_plan)
            },
            |(device, mut pipeline, image, patchify_plan)| {
                let output = pipeline
                    .step_image_with_sparse_patchify_plan_cuda_measured(
                        black_box(image),
                        black_box(&patchify_plan),
                        SparseJepaAnyUpPcaMeasurementConfig::disabled(),
                    )
                    .expect("sparse patchify pipeline step");
                B::sync(&device).expect("sync sparse patchify CUDA backend");
                black_box(output.output.pca_display);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

#[cfg(not(feature = "sparse-patchify-cuda"))]
fn bench_tiny_e2e_pipeline_step_sparse_patchify_cuda(_c: &mut Criterion) {}

fn bench_inflight_stream_batches<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let model_config = VJepaConfig::tiny_for_tests();
    let mut group = c.benchmark_group(format!("highres_inflight_stream_{backend_name}"));
    for batch_size in [1usize, 2, 4] {
        group.throughput(Throughput::Elements(
            (batch_size * model_config.image_size * model_config.image_size) as u64,
        ));
        group.bench_function(format!("tiny32_batch{batch_size}_sparse50"), |bench| {
            bench.iter_batched(
                || {
                    let device = make_device();
                    let jepa = VJepa2_1Model::<B>::new(&model_config, &device);
                    let anyup =
                        AnyUp::<B>::new(AnyUpConfig::tiny_for_tests(), &device).expect("anyup");
                    let pipeline = SparseJepaAnyUpPcaPipeline::<B>::new(
                        jepa,
                        anyup,
                        &model_config,
                        SparseJepaAnyUpPcaPipelineConfig::default(),
                        batch_size,
                        [model_config.image_size, model_config.image_size],
                        &device,
                    )
                    .expect("pipeline");
                    let grid = pipeline.grid();
                    let mut stream = SparseJepaAnyUpPcaStream::new(
                        pipeline,
                        SparseJepaAnyUpPcaStreamConfig {
                            queue_capacity: batch_size * 2,
                            batch_size,
                            measurement: SparseJepaAnyUpPcaMeasurementConfig::disabled(),
                            ..SparseJepaAnyUpPcaStreamConfig::default()
                        },
                    )
                    .expect("stream");
                    enqueue_tiny_stream_batch(
                        &mut stream,
                        &device,
                        &model_config,
                        grid,
                        batch_size,
                        0,
                    );
                    (device, stream)
                },
                |(device, mut stream)| {
                    let output = stream
                        .process_next_ready()
                        .expect("stream process")
                        .expect("ready output");
                    B::sync(&device).expect("sync stream backend");
                    black_box(output.output.pca_display);
                    black_box(output.frame_ids);
                    black_box(output.metrics);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_inflight_stream_cached_masks<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let model_config = VJepaConfig::tiny_for_tests();
    let mut group = c.benchmark_group(format!(
        "highres_inflight_stream_cached_mask_{backend_name}"
    ));
    for batch_size in [1usize, 2, 4] {
        group.throughput(Throughput::Elements(
            (batch_size * model_config.image_size * model_config.image_size) as u64,
        ));
        group.bench_function(format!("tiny32_batch{batch_size}_sparse50"), |bench| {
            bench.iter_batched(
                || {
                    let device = make_device();
                    let jepa = VJepa2_1Model::<B>::new(&model_config, &device);
                    let anyup =
                        AnyUp::<B>::new(AnyUpConfig::tiny_for_tests(), &device).expect("anyup");
                    let pipeline = SparseJepaAnyUpPcaPipeline::<B>::new(
                        jepa,
                        anyup,
                        &model_config,
                        SparseJepaAnyUpPcaPipelineConfig::default(),
                        batch_size,
                        [model_config.image_size, model_config.image_size],
                        &device,
                    )
                    .expect("pipeline");
                    let grid = pipeline.grid();
                    let mut stream = SparseJepaAnyUpPcaStream::new(
                        pipeline,
                        SparseJepaAnyUpPcaStreamConfig {
                            queue_capacity: batch_size * 2,
                            batch_size,
                            measurement: SparseJepaAnyUpPcaMeasurementConfig::disabled(),
                            ..SparseJepaAnyUpPcaStreamConfig::default()
                        },
                    )
                    .expect("stream");
                    enqueue_tiny_stream_batch(
                        &mut stream,
                        &device,
                        &model_config,
                        grid,
                        batch_size,
                        0,
                    );
                    let warm = stream
                        .process_next_ready()
                        .expect("warm cached mask stream")
                        .expect("warm cached mask output");
                    B::sync(&device).expect("sync cached mask warmup");
                    black_box(warm.metrics);
                    enqueue_tiny_stream_batch(
                        &mut stream,
                        &device,
                        &model_config,
                        grid,
                        batch_size,
                        batch_size as u64,
                    );
                    (device, stream)
                },
                |(device, mut stream)| {
                    let output = stream
                        .process_next_ready()
                        .expect("cached mask stream process")
                        .expect("ready cached mask output");
                    B::sync(&device).expect("sync cached mask stream backend");
                    black_box(output.output.pca_display);
                    black_box(output.frame_ids);
                    black_box(output.metrics);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn enqueue_tiny_stream_batch<B: Backend>(
    stream: &mut SparseJepaAnyUpPcaStream<B>,
    device: &B::Device,
    model_config: &VJepaConfig,
    grid: TokenGridShape,
    batch_size: usize,
    sequence_start: u64,
) {
    for row in 0..batch_size {
        let sequence = sequence_start + row as u64;
        stream
            .enqueue(SparseJepaAnyUpPcaFrameInput {
                id: SparseJepaAnyUpPcaFrameId {
                    stream_id: 0,
                    sequence,
                    capture_time_nanos: sequence,
                },
                image: Tensor::<B, 4>::ones(
                    [1, 3, model_config.image_size, model_config.image_size],
                    device,
                ),
                mask: SparseTokenMask::evenly_spaced(grid.len(), 2),
            })
            .expect("enqueue tiny stream frame");
    }
}

fn bench_backend<B, MakeDevice>(c: &mut Criterion, backend_name: &str, make_device: MakeDevice)
where
    B: Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    bench_pca_projection::<B, _>(c, backend_name, make_device);
    bench_semantic_pca_stats_update::<B, _>(c, backend_name, make_device);
    bench_pca_basis_update::<B, _>(c, backend_name, make_device);
    bench_anyup_from_token_cache::<B, _>(c, backend_name, make_device);
    bench_sparse_update_to_anyup_pca::<B, _>(c, backend_name, make_device);
    bench_tiny_jepa_cache_density_sweep::<B, _>(c, backend_name, make_device);
    bench_tiny_e2e_pipeline_step::<B, _>(c, backend_name, make_device);
    bench_inflight_stream_batches::<B, _>(c, backend_name, make_device);
    bench_inflight_stream_cached_masks::<B, _>(c, backend_name, make_device);
}

#[cfg(feature = "ndarray")]
fn highres_pipeline_ndarray(c: &mut Criterion) {
    bench_backend::<burn::backend::NdArray<f32>, _>(c, "ndarray", Default::default);
}

#[cfg(not(feature = "ndarray"))]
fn highres_pipeline_ndarray(_c: &mut Criterion) {}

#[cfg(feature = "flex")]
fn highres_pipeline_flex(c: &mut Criterion) {
    bench_backend::<burn::backend::Flex<f32, i32>, _>(c, "flex", Default::default);
}

#[cfg(not(feature = "flex"))]
fn highres_pipeline_flex(_c: &mut Criterion) {}

#[cfg(all(feature = "dispatch", feature = "ndarray"))]
fn highres_pipeline_dispatch_ndarray(c: &mut Criterion) {
    bench_backend::<burn::Dispatch, _>(c, "dispatch_ndarray", || {
        burn::DispatchDevice::NdArray(Default::default())
    });
}

#[cfg(not(all(feature = "dispatch", feature = "ndarray")))]
fn highres_pipeline_dispatch_ndarray(_c: &mut Criterion) {}

#[cfg(feature = "cuda")]
fn highres_pipeline_cuda(c: &mut Criterion) {
    if let Err(reason) =
        burn_jepa::runtime::cuda_runtime_preflight(burn_jepa::runtime::CUDA_TRAIN_FORCE_ENV)
    {
        eprintln!("skipping highres_pipeline_cuda: {reason}");
        return;
    }
    bench_backend::<burn::backend::Cuda<f32, i32>, _>(c, "cuda", Default::default);
}

#[cfg(not(feature = "cuda"))]
fn highres_pipeline_cuda(_c: &mut Criterion) {}

#[cfg(any(feature = "wgpu", feature = "webgpu"))]
fn highres_pipeline_wgpu(c: &mut Criterion) {
    bench_backend::<burn::backend::Wgpu<f32, i32>, _>(c, "wgpu", Default::default);
}

#[cfg(not(any(feature = "wgpu", feature = "webgpu")))]
fn highres_pipeline_wgpu(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_patch_diff_refresh_policy,
    highres_pipeline_ndarray,
    highres_pipeline_flex,
    highres_pipeline_dispatch_ndarray,
    highres_pipeline_cuda,
    highres_pipeline_wgpu,
    bench_tiny_e2e_pipeline_step_sparse_patchify_wgpu,
    bench_tiny_e2e_pipeline_step_sparse_patchify_cuda
);

criterion_main!(benches);
