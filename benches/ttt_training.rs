use burn::tensor::Tensor;
use burn_jepa::{
    SparseMaskBatch, SparseTokenMask, TttEncoderConfig, TttSparsePatchifyTrainingBackend,
    VJepa2_1Model, VJepaTttModel, apply_mask_batch, synthetic_video,
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

#[derive(Clone, Copy)]
struct DensityCase {
    label: &'static str,
    density: f32,
}

const SPARSITY_DENSITY_CASES: [DensityCase; 3] = [
    DensityCase {
        label: "10pct",
        density: 0.10,
    },
    DensityCase {
        label: "50pct",
        density: 0.50,
    },
    DensityCase {
        label: "100pct",
        density: 1.00,
    },
];

fn rollout_mask(config: &burn_jepa::VJepaConfig) -> SparseTokenMask {
    let dense = config.num_patches();
    let keep = (dense / 2).max(1);
    SparseTokenMask::evenly_spaced(dense, keep)
}

fn synthetic_video_batch<B: burn::tensor::backend::Backend>(
    config: &burn_jepa::VJepaConfig,
    batch_size: usize,
    device: &B::Device,
) -> Tensor<B, 5> {
    let videos = (0..batch_size)
        .map(|index| {
            synthetic_video::<B>(
                index,
                config.in_channels,
                4,
                config.image_size,
                config.image_size,
                device,
            )
        })
        .collect::<Vec<_>>();
    Tensor::cat(videos, 0)
}

fn training_step_bench_config() -> burn_jepa::VJepaConfig {
    let mut config = burn_jepa::VJepaConfig::tiny_for_tests();
    config.image_size = 64;
    config
}

fn keep_count_for_density(dense_tokens: usize, density: f32) -> usize {
    ((dense_tokens as f32) * density)
        .ceil()
        .max(1.0)
        .min(dense_tokens as f32) as usize
}

fn shifted_evenly_spaced_indices(
    dense_tokens: usize,
    keep_tokens: usize,
    shift: usize,
) -> Vec<usize> {
    let mut row = (0..keep_tokens)
        .map(|index| {
            let base = index * dense_tokens / keep_tokens;
            (base + shift) % dense_tokens
        })
        .collect::<Vec<_>>();
    row.sort_unstable();
    row
}

fn density_rows(dense_tokens: usize, batch_size: usize, density: f32) -> Vec<Vec<usize>> {
    let keep_tokens = keep_count_for_density(dense_tokens, density);
    (0..batch_size)
        .map(|index| shifted_evenly_spaced_indices(dense_tokens, keep_tokens, index))
        .collect()
}

fn bench_ttt_training_step_matrix<B>(c: &mut Criterion, backend_name: &str, batch_sizes: &[usize])
where
    B: burn::tensor::backend::AutodiffBackend,
{
    use burn::module::Module;
    use burn::optim::{AdamWConfig, GradientsParams, Optimizer};

    let mut group = c.benchmark_group(format!("ttt_training_step_{backend_name}"));
    for &batch_size in batch_sizes {
        let device = Default::default();
        let config = training_step_bench_config();
        let teacher = VJepa2_1Model::<B>::new(&config, &device).no_grad();
        let student = VJepaTttModel::from_model(
            VJepa2_1Model::<B>::new(&config, &device),
            TttEncoderConfig {
                layers: vec![0],
                chunk_tokens: 2,
                ..TttEncoderConfig::default()
            },
            &device,
        )
        .expect("ttt model");
        let video = synthetic_video_batch::<B>(&config, batch_size, &device);
        let teacher_tokens = teacher.encode_video(video.clone(), None).tokens.detach();
        let mut model = Some(student);
        let mut optim = AdamWConfig::new().init::<B, VJepaTttModel<B>>();

        group.bench_function(format!("dense_b{batch_size}"), |bench| {
            bench.iter(|| {
                let current = model.take().expect("model available");
                let mut state = current.fresh_state();
                let student = current
                    .forward_single_frame_rollout(
                        black_box(video.clone()),
                        Some(teacher_tokens.clone()),
                        &mut state,
                    )
                    .expect("dense rollout");
                let loss = (student.tokens - teacher_tokens.clone())
                    .powf_scalar(2.0)
                    .mean();
                let grads = GradientsParams::from_grads(loss.backward(), &current);
                let next = optim.step(1.0e-3, current, grads);
                model = Some(next);
            });
        });

        let mask = SparseMaskBatch::<B>::from_rows(
            density_rows(config.num_patches(), batch_size, 0.5),
            config.num_patches(),
            &device,
        )
        .expect("fixed-width sparse mask batch");
        let sparse_teacher_tokens = apply_mask_batch(teacher_tokens.clone(), &mask);
        let mut sparse_model = Some(
            VJepaTttModel::from_model(
                VJepa2_1Model::<B>::new(&config, &device),
                TttEncoderConfig {
                    layers: vec![0],
                    chunk_tokens: 2,
                    ..TttEncoderConfig::default()
                },
                &device,
            )
            .expect("sparse ttt model"),
        );
        let mut sparse_optim = AdamWConfig::new().init::<B, VJepaTttModel<B>>();
        group.bench_function(format!("fixed_width_sparse_b{batch_size}"), |bench| {
            bench.iter(|| {
                let current = sparse_model.take().expect("model available");
                let mut state = current.fresh_state();
                let student = current
                    .forward_single_frame_rollout_sparse_batch(
                        black_box(video.clone()),
                        &mask,
                        Some(teacher_tokens.clone()),
                        &mut state,
                    )
                    .expect("fixed-width sparse rollout");
                let loss = (student.tokens - sparse_teacher_tokens.clone())
                    .powf_scalar(2.0)
                    .mean();
                let grads = GradientsParams::from_grads(loss.backward(), &current);
                let next = sparse_optim.step(1.0e-3, current, grads);
                sparse_model = Some(next);
            });
        });
    }
    group.finish();
}

fn bench_ttt_sparsity_training_step_matrix<B>(
    c: &mut Criterion,
    backend_name: &str,
    batch_sizes: &[usize],
) where
    B: burn::tensor::backend::AutodiffBackend,
{
    use burn::module::Module;
    use burn::optim::{AdamWConfig, GradientsParams, Optimizer};

    let mut group = c.benchmark_group(format!("ttt_sparsity_training_step_{backend_name}"));
    for &batch_size in batch_sizes {
        group.throughput(Throughput::Elements(batch_size as u64));

        let device = Default::default();
        let config = training_step_bench_config();
        let dense_tokens = config.num_patches();
        let video = synthetic_video_batch::<B>(&config, batch_size, &device);
        let teacher = VJepa2_1Model::<B>::new(&config, &device).no_grad();
        let teacher_tokens = teacher.encode_video(video.clone(), None).tokens.detach();

        for case in SPARSITY_DENSITY_CASES {
            let keep_tokens = keep_count_for_density(dense_tokens, case.density);
            let mask = SparseMaskBatch::<B>::from_rows(
                density_rows(dense_tokens, batch_size, case.density),
                dense_tokens,
                &device,
            )
            .expect("density sparse mask batch");
            let sparse_teacher_tokens = apply_mask_batch(teacher_tokens.clone(), &mask);
            let mut sparse_model = Some(
                VJepaTttModel::from_model(
                    VJepa2_1Model::<B>::new(&config, &device),
                    TttEncoderConfig {
                        layers: vec![0],
                        chunk_tokens: 2,
                        ..TttEncoderConfig::default()
                    },
                    &device,
                )
                .expect("sparse ttt model"),
            );
            let mut sparse_optim = AdamWConfig::new().init::<B, VJepaTttModel<B>>();
            group.bench_function(
                format!(
                    "density_{}_sparse_b{batch_size}_tokens{keep_tokens}_of{dense_tokens}",
                    case.label
                ),
                |bench| {
                    bench.iter(|| {
                        let current = sparse_model.take().expect("model available");
                        let mut state = current.fresh_state();
                        let student = current
                            .forward_single_frame_rollout_sparse_batch(
                                black_box(video.clone()),
                                &mask,
                                Some(teacher_tokens.clone()),
                                &mut state,
                            )
                            .expect("sparse rollout");
                        let loss = (student.tokens - sparse_teacher_tokens.clone())
                            .powf_scalar(2.0)
                            .mean();
                        let grads = GradientsParams::from_grads(loss.backward(), &current);
                        let next = sparse_optim.step(1.0e-3, current, grads);
                        sparse_model = Some(next);
                    });
                },
            );
        }

        let mut dense_model = Some(
            VJepaTttModel::from_model(
                VJepa2_1Model::<B>::new(&config, &device),
                TttEncoderConfig {
                    layers: vec![0],
                    chunk_tokens: 2,
                    ..TttEncoderConfig::default()
                },
                &device,
            )
            .expect("dense ttt model"),
        );
        let mut dense_optim = AdamWConfig::new().init::<B, VJepaTttModel<B>>();
        group.bench_function(
            format!("density_100pct_dense_b{batch_size}_tokens{dense_tokens}_of{dense_tokens}"),
            |bench| {
                bench.iter(|| {
                    let current = dense_model.take().expect("model available");
                    let mut state = current.fresh_state();
                    let student = current
                        .forward_single_frame_rollout(
                            black_box(video.clone()),
                            Some(teacher_tokens.clone()),
                            &mut state,
                        )
                        .expect("dense rollout");
                    let loss = (student.tokens - teacher_tokens.clone())
                        .powf_scalar(2.0)
                        .mean();
                    let grads = GradientsParams::from_grads(loss.backward(), &current);
                    let next = dense_optim.step(1.0e-3, current, grads);
                    dense_model = Some(next);
                });
            },
        );
    }
    group.finish();
}

fn bench_ttt_sparse_patchify_sparsity_training_step_matrix<B>(
    c: &mut Criterion,
    backend_name: &str,
    batch_sizes: &[usize],
) where
    B: TttSparsePatchifyTrainingBackend,
{
    use burn::module::Module;
    use burn::optim::{AdamWConfig, GradientsParams, Optimizer};

    if !B::frozen_sparse_patchify_batch_supported() {
        eprintln!(
            "skipping ttt_sparse_patchify_sparsity_training_step_{backend_name}: batched frozen sparse patchify is not supported"
        );
        return;
    }

    let mut group = c.benchmark_group(format!(
        "ttt_sparse_patchify_sparsity_training_step_{backend_name}"
    ));
    for &batch_size in batch_sizes {
        group.throughput(Throughput::Elements(batch_size as u64));

        let device = Default::default();
        let config = training_step_bench_config();
        let dense_tokens = config.num_patches();
        let video = synthetic_video_batch::<B>(&config, batch_size, &device);
        let teacher = VJepa2_1Model::<B>::new(&config, &device).no_grad();
        let teacher_tokens = teacher.encode_video(video.clone(), None).tokens.detach();

        for case in SPARSITY_DENSITY_CASES {
            let keep_tokens = keep_count_for_density(dense_tokens, case.density);
            let mask = SparseMaskBatch::<B>::from_rows(
                density_rows(dense_tokens, batch_size, case.density),
                dense_tokens,
                &device,
            )
            .expect("density sparse mask batch");
            let sparse_teacher_tokens = apply_mask_batch(teacher_tokens.clone(), &mask);
            let mut sparse_model = Some(
                VJepaTttModel::from_model(
                    VJepa2_1Model::<B>::new(&config, &device),
                    TttEncoderConfig {
                        layers: vec![0],
                        chunk_tokens: 2,
                        freeze_pretrained: true,
                        ..TttEncoderConfig::default()
                    },
                    &device,
                )
                .expect("sparse patchify ttt model"),
            );
            let mut sparse_optim = AdamWConfig::new().init::<B, VJepaTttModel<B>>();
            group.bench_function(
                format!(
                    "density_{}_sparse_patchify_b{batch_size}_tokens{keep_tokens}_of{dense_tokens}",
                    case.label
                ),
                |bench| {
                    bench.iter(|| {
                        let current = sparse_model.take().expect("model available");
                        let mut state = current.fresh_state();
                        let student =
                            <B as TttSparsePatchifyTrainingBackend>::student_frozen_sparse_patchify_rollout_batch(
                                &current,
                                black_box(video.clone()),
                                &mask,
                                Some(teacher_tokens.clone()),
                                &mut state,
                            )
                            .expect("frozen sparse patchify rollout");
                        let loss = (student.tokens - sparse_teacher_tokens.clone())
                            .powf_scalar(2.0)
                            .mean();
                        let grads = GradientsParams::from_grads(loss.backward(), &current);
                        let next = sparse_optim.step(1.0e-3, current, grads);
                        sparse_model = Some(next);
                    });
                },
            );
        }

        let mut dense_model = Some(
            VJepaTttModel::from_model(
                VJepa2_1Model::<B>::new(&config, &device),
                TttEncoderConfig {
                    layers: vec![0],
                    chunk_tokens: 2,
                    freeze_pretrained: true,
                    ..TttEncoderConfig::default()
                },
                &device,
            )
            .expect("dense ttt model"),
        );
        let mut dense_optim = AdamWConfig::new().init::<B, VJepaTttModel<B>>();
        group.bench_function(
            format!("density_100pct_dense_b{batch_size}_tokens{dense_tokens}_of{dense_tokens}"),
            |bench| {
                bench.iter(|| {
                    let current = dense_model.take().expect("model available");
                    let mut state = current.fresh_state();
                    let student = current
                        .forward_single_frame_rollout(
                            black_box(video.clone()),
                            Some(teacher_tokens.clone()),
                            &mut state,
                        )
                        .expect("dense rollout");
                    let loss = (student.tokens - teacher_tokens.clone())
                        .powf_scalar(2.0)
                        .mean();
                    let grads = GradientsParams::from_grads(loss.backward(), &current);
                    let next = dense_optim.step(1.0e-3, current, grads);
                    dense_model = Some(next);
                });
            },
        );
    }
    group.finish();
}

fn ttt_single_frame_rollout_ndarray(c: &mut Criterion) {
    type B = burn::backend::NdArray<f32>;
    let device = Default::default();
    let config = burn_jepa::VJepaConfig::tiny_for_tests();
    let teacher = VJepa2_1Model::<B>::new(&config, &device);
    let student = VJepaTttModel::from_model(
        VJepa2_1Model::<B>::new(&config, &device),
        TttEncoderConfig {
            layers: vec![0],
            chunk_tokens: 2,
            ..TttEncoderConfig::default()
        },
        &device,
    )
    .expect("ttt model");
    let video = synthetic_video::<B>(0, config.in_channels, 4, 32, 32, &device);
    let teacher_tokens = teacher.encode_video(video.clone(), None).tokens.detach();

    c.bench_function("ttt_single_frame_rollout_ndarray", |bench| {
        bench.iter(|| {
            let mut state = student.fresh_state();
            let output = student
                .forward_single_frame_rollout(
                    black_box(video.clone()),
                    Some(teacher_tokens.clone()),
                    &mut state,
                )
                .expect("rollout");
            black_box(output.tokens);
        });
    });
}

fn ttt_sparse_single_frame_rollout_ndarray(c: &mut Criterion) {
    type B = burn::backend::NdArray<f32>;
    let device = Default::default();
    let config = burn_jepa::VJepaConfig::tiny_for_tests();
    let teacher = VJepa2_1Model::<B>::new(&config, &device);
    let student = VJepaTttModel::from_model(
        VJepa2_1Model::<B>::new(&config, &device),
        TttEncoderConfig {
            layers: vec![0],
            chunk_tokens: 2,
            ..TttEncoderConfig::default()
        },
        &device,
    )
    .expect("ttt model");
    let mask = rollout_mask(&config);
    let video = synthetic_video::<B>(0, config.in_channels, 4, 32, 32, &device);
    let teacher_tokens = teacher.encode_video(video.clone(), None).tokens.detach();

    c.bench_function("ttt_sparse_single_frame_rollout_ndarray_50pct", |bench| {
        bench.iter(|| {
            let mut state = student.fresh_state();
            let output = student
                .forward_single_frame_rollout_sparse(
                    black_box(video.clone()),
                    &mask,
                    Some(teacher_tokens.clone()),
                    &mut state,
                )
                .expect("sparse rollout");
            black_box(output.tokens);
        });
    });
}

fn ttt_fixed_width_sparse_single_frame_rollout_ndarray(c: &mut Criterion) {
    type B = burn::backend::NdArray<f32>;
    let device = Default::default();
    let config = burn_jepa::VJepaConfig::tiny_for_tests();
    let teacher = VJepa2_1Model::<B>::new(&config, &device);
    let student = VJepaTttModel::from_model(
        VJepa2_1Model::<B>::new(&config, &device),
        TttEncoderConfig {
            layers: vec![0],
            chunk_tokens: 2,
            ..TttEncoderConfig::default()
        },
        &device,
    )
    .expect("ttt model");
    let video = Tensor::cat(
        vec![
            synthetic_video::<B>(0, config.in_channels, 4, 32, 32, &device),
            synthetic_video::<B>(1, config.in_channels, 4, 32, 32, &device),
        ],
        0,
    );
    let teacher_tokens = teacher.encode_video(video.clone(), None).tokens.detach();
    let mask = SparseMaskBatch::<B>::from_rows(
        vec![vec![0, 1, 4, 5], vec![2, 3, 6, 7]],
        config.num_patches(),
        &device,
    )
    .expect("fixed-width mask batch");

    c.bench_function(
        "ttt_fixed_width_sparse_single_frame_rollout_ndarray_b2_50pct",
        |bench| {
            bench.iter(|| {
                let mut state = student.fresh_state();
                let output = student
                    .forward_single_frame_rollout_sparse_batch(
                        black_box(video.clone()),
                        &mask,
                        Some(teacher_tokens.clone()),
                        &mut state,
                    )
                    .expect("fixed-width sparse rollout");
                black_box(output.tokens);
            });
        },
    );
}

fn ttt_training_step_matrix_ndarray(c: &mut Criterion) {
    bench_ttt_training_step_matrix::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>(
        c,
        "ndarray",
        &[1, 2, 4, 8],
    );
}

fn ttt_sparsity_training_step_matrix_ndarray(c: &mut Criterion) {
    bench_ttt_sparsity_training_step_matrix::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>(
        c,
        "ndarray",
        &[1, 2, 4, 8],
    );
}

#[cfg(feature = "cuda")]
fn ttt_training_step_matrix_cuda(c: &mut Criterion) {
    if let Err(reason) =
        burn_jepa::runtime::cuda_runtime_preflight(burn_jepa::runtime::CUDA_TRAIN_FORCE_ENV)
    {
        eprintln!("skipping ttt_training_step_matrix_cuda: {reason}");
        return;
    }
    bench_ttt_training_step_matrix::<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>(
        c,
        "cuda",
        &[1, 2, 4],
    );
}

#[cfg(not(feature = "cuda"))]
fn ttt_training_step_matrix_cuda(_c: &mut Criterion) {}

#[cfg(feature = "cuda")]
fn ttt_sparsity_training_step_matrix_cuda(c: &mut Criterion) {
    if let Err(reason) =
        burn_jepa::runtime::cuda_runtime_preflight(burn_jepa::runtime::CUDA_TRAIN_FORCE_ENV)
    {
        eprintln!("skipping ttt_sparsity_training_step_matrix_cuda: {reason}");
        return;
    }
    bench_ttt_sparsity_training_step_matrix::<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>(
        c,
        "cuda",
        &[1, 2, 4],
    );
}

#[cfg(not(feature = "cuda"))]
fn ttt_sparsity_training_step_matrix_cuda(_c: &mut Criterion) {}

#[cfg(all(feature = "cuda", feature = "sparse-patchify-cuda"))]
fn ttt_sparse_patchify_sparsity_training_step_matrix_cuda(c: &mut Criterion) {
    if let Err(reason) =
        burn_jepa::runtime::cuda_runtime_preflight(burn_jepa::runtime::CUDA_TRAIN_FORCE_ENV)
    {
        eprintln!("skipping ttt_sparse_patchify_sparsity_training_step_cuda: {reason}");
        return;
    }
    bench_ttt_sparse_patchify_sparsity_training_step_matrix::<
        burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>,
    >(c, "cuda", &[1, 2, 4]);
}

#[cfg(not(all(feature = "cuda", feature = "sparse-patchify-cuda")))]
fn ttt_sparse_patchify_sparsity_training_step_matrix_cuda(_c: &mut Criterion) {}

#[cfg(feature = "wgpu")]
fn ttt_training_step_matrix_wgpu(c: &mut Criterion) {
    bench_ttt_training_step_matrix::<burn::backend::Autodiff<burn::backend::Wgpu<f32, i32>>>(
        c,
        "wgpu",
        &[1, 2, 4],
    );
}

#[cfg(not(feature = "wgpu"))]
fn ttt_training_step_matrix_wgpu(_c: &mut Criterion) {}

#[cfg(feature = "wgpu")]
fn ttt_sparsity_training_step_matrix_wgpu(c: &mut Criterion) {
    bench_ttt_sparsity_training_step_matrix::<burn::backend::Autodiff<burn::backend::Wgpu<f32, i32>>>(
        c,
        "wgpu",
        &[1, 2, 4],
    );
}

#[cfg(not(feature = "wgpu"))]
fn ttt_sparsity_training_step_matrix_wgpu(_c: &mut Criterion) {}

#[cfg(feature = "sparse-patchify-wgpu")]
fn ttt_sparse_patchify_sparsity_training_step_matrix_wgpu(c: &mut Criterion) {
    bench_ttt_sparse_patchify_sparsity_training_step_matrix::<
        burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>,
    >(c, "wgpu", &[1, 2, 4]);
}

#[cfg(not(feature = "sparse-patchify-wgpu"))]
fn ttt_sparse_patchify_sparsity_training_step_matrix_wgpu(_c: &mut Criterion) {}

#[cfg(feature = "webgpu")]
fn ttt_training_step_matrix_webgpu(c: &mut Criterion) {
    bench_ttt_training_step_matrix::<burn::backend::Autodiff<burn::backend::WebGpu<f32, i32>>>(
        c,
        "webgpu",
        &[1, 2, 4],
    );
}

#[cfg(not(feature = "webgpu"))]
fn ttt_training_step_matrix_webgpu(_c: &mut Criterion) {}

#[cfg(feature = "webgpu")]
fn ttt_sparsity_training_step_matrix_webgpu(c: &mut Criterion) {
    bench_ttt_sparsity_training_step_matrix::<
        burn::backend::Autodiff<burn::backend::WebGpu<f32, i32>>,
    >(c, "webgpu", &[1, 2, 4]);
}

#[cfg(not(feature = "webgpu"))]
fn ttt_sparsity_training_step_matrix_webgpu(_c: &mut Criterion) {}

#[cfg(feature = "cuda")]
fn ttt_single_frame_rollout_cuda(c: &mut Criterion) {
    if let Err(reason) =
        burn_jepa::runtime::cuda_runtime_preflight(burn_jepa::runtime::CUDA_TRAIN_FORCE_ENV)
    {
        eprintln!("skipping ttt_single_frame_rollout_cuda: {reason}");
        return;
    }
    type B = burn::backend::Cuda<f32, i32>;
    let device = Default::default();
    let config = burn_jepa::VJepaConfig::tiny_for_tests();
    let teacher = VJepa2_1Model::<B>::new(&config, &device);
    let student = VJepaTttModel::from_model(
        VJepa2_1Model::<B>::new(&config, &device),
        TttEncoderConfig {
            layers: vec![0],
            chunk_tokens: 2,
            ..TttEncoderConfig::default()
        },
        &device,
    )
    .expect("ttt model");
    let video = synthetic_video::<B>(0, config.in_channels, 4, 32, 32, &device);
    let teacher_tokens = teacher.encode_video(video.clone(), None).tokens.detach();

    c.bench_function("ttt_single_frame_rollout_cuda", |bench| {
        bench.iter(|| {
            let mut state = student.fresh_state();
            let output = student
                .forward_single_frame_rollout(
                    black_box(video.clone()),
                    Some(teacher_tokens.clone()),
                    &mut state,
                )
                .expect("rollout");
            black_box(output.tokens);
        });
    });
}

#[cfg(not(feature = "cuda"))]
fn ttt_single_frame_rollout_cuda(_c: &mut Criterion) {}

#[cfg(feature = "sparse-patchify-wgpu")]
fn ttt_sparse_patchify_single_frame_rollout_wgpu(c: &mut Criterion) {
    type B = burn_flex_gmm::wgpu::DefaultWgpuBackend;
    let device = Default::default();
    let config = burn_jepa::VJepaConfig::tiny_for_tests();
    let teacher = VJepa2_1Model::<B>::new(&config, &device);
    let student = VJepaTttModel::from_model(
        VJepa2_1Model::<B>::new(&config, &device),
        TttEncoderConfig {
            layers: vec![0],
            chunk_tokens: 2,
            ..TttEncoderConfig::default()
        },
        &device,
    )
    .expect("ttt model");
    let mask = rollout_mask(&config);
    let video = synthetic_video::<B>(0, config.in_channels, 4, 32, 32, &device);
    let teacher_tokens = teacher.encode_video(video.clone(), None).tokens.detach();

    c.bench_function(
        "ttt_sparse_patchify_single_frame_rollout_wgpu_50pct",
        |bench| {
            bench.iter(|| {
                let mut state = student.fresh_state();
                let output = student
                    .forward_single_frame_rollout_sparse_patchify_wgpu(
                        black_box(video.clone()),
                        &mask,
                        Some(teacher_tokens.clone()),
                        &mut state,
                    )
                    .expect("wgpu sparse patchify rollout");
                black_box(output.tokens);
            });
        },
    );
}

#[cfg(not(feature = "sparse-patchify-wgpu"))]
fn ttt_sparse_patchify_single_frame_rollout_wgpu(_c: &mut Criterion) {}

#[cfg(feature = "sparse-patchify-cuda")]
fn ttt_sparse_patchify_single_frame_rollout_cuda(c: &mut Criterion) {
    if let Err(reason) =
        burn_jepa::runtime::cuda_runtime_preflight(burn_jepa::runtime::CUDA_TRAIN_FORCE_ENV)
    {
        eprintln!("skipping ttt_sparse_patchify_single_frame_rollout_cuda: {reason}");
        return;
    }
    type B = burn_flex_gmm::cuda::DefaultCudaBackend;
    let device = Default::default();
    let config = burn_jepa::VJepaConfig::tiny_for_tests();
    let teacher = VJepa2_1Model::<B>::new(&config, &device);
    let student = VJepaTttModel::from_model(
        VJepa2_1Model::<B>::new(&config, &device),
        TttEncoderConfig {
            layers: vec![0],
            chunk_tokens: 2,
            ..TttEncoderConfig::default()
        },
        &device,
    )
    .expect("ttt model");
    let mask = rollout_mask(&config);
    let video = synthetic_video::<B>(0, config.in_channels, 4, 32, 32, &device);
    let teacher_tokens = teacher.encode_video(video.clone(), None).tokens.detach();

    c.bench_function(
        "ttt_sparse_patchify_single_frame_rollout_cuda_50pct",
        |bench| {
            bench.iter(|| {
                let mut state = student.fresh_state();
                let output = student
                    .forward_single_frame_rollout_sparse_patchify_cuda(
                        black_box(video.clone()),
                        &mask,
                        Some(teacher_tokens.clone()),
                        &mut state,
                    )
                    .expect("cuda sparse patchify rollout");
                black_box(output.tokens);
            });
        },
    );
}

#[cfg(not(feature = "sparse-patchify-cuda"))]
fn ttt_sparse_patchify_single_frame_rollout_cuda(_c: &mut Criterion) {}

criterion_group!(
    benches,
    ttt_single_frame_rollout_ndarray,
    ttt_sparse_single_frame_rollout_ndarray,
    ttt_fixed_width_sparse_single_frame_rollout_ndarray,
    ttt_training_step_matrix_ndarray,
    ttt_sparsity_training_step_matrix_ndarray,
    ttt_training_step_matrix_cuda,
    ttt_sparsity_training_step_matrix_cuda,
    ttt_sparse_patchify_sparsity_training_step_matrix_cuda,
    ttt_training_step_matrix_wgpu,
    ttt_sparsity_training_step_matrix_wgpu,
    ttt_sparse_patchify_sparsity_training_step_matrix_wgpu,
    ttt_training_step_matrix_webgpu,
    ttt_sparsity_training_step_matrix_webgpu,
    ttt_single_frame_rollout_cuda,
    ttt_sparse_patchify_single_frame_rollout_wgpu,
    ttt_sparse_patchify_single_frame_rollout_cuda
);

criterion_main!(benches);
