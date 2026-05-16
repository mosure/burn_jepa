use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_jepa::{InterframeJepaFeatureMemory, InterframeJepaFeatureMemoryConfig, TokenGridShape};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

#[derive(Clone, Copy)]
struct FeatureMemoryCase {
    label: &'static str,
    grid: TokenGridShape,
    embed_dim: usize,
    batch: usize,
}

#[derive(Clone, Copy)]
struct DensityCase {
    label: &'static str,
    density: f32,
}

const FEATURE_MEMORY_CASES: [FeatureMemoryCase; 4] = [
    FeatureMemoryCase {
        label: "tiny_stream",
        grid: TokenGridShape::new(2, 4, 4),
        embed_dim: 128,
        batch: 1,
    },
    FeatureMemoryCase {
        label: "vjepa224_b1",
        grid: TokenGridShape::new(8, 14, 14),
        embed_dim: 768,
        batch: 1,
    },
    FeatureMemoryCase {
        label: "vjepa224_b4",
        grid: TokenGridShape::new(8, 14, 14),
        embed_dim: 768,
        batch: 4,
    },
    FeatureMemoryCase {
        label: "vjepa384_b1",
        grid: TokenGridShape::new(8, 24, 24),
        embed_dim: 768,
        batch: 1,
    },
];

const DENSITY_CASES: [DensityCase; 6] = [
    DensityCase {
        label: "1pct",
        density: 0.01,
    },
    DensityCase {
        label: "5pct",
        density: 0.05,
    },
    DensityCase {
        label: "10pct",
        density: 0.10,
    },
    DensityCase {
        label: "25pct",
        density: 0.25,
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

fn keep_count(dense_tokens: usize, density: f32) -> usize {
    ((dense_tokens as f32) * density)
        .ceil()
        .max(1.0)
        .min(dense_tokens as f32) as usize
}

fn shifted_indices(dense_tokens: usize, keep: usize, row: usize) -> Vec<usize> {
    let mut indices = (0..keep)
        .map(|index| {
            let base = index * dense_tokens / keep;
            (base + row) % dense_tokens
        })
        .collect::<Vec<_>>();
    indices.sort_unstable();
    indices
}

fn token_indices<B: Backend>(
    batch: usize,
    dense_tokens: usize,
    keep: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let values = (0..batch)
        .flat_map(|row| {
            shifted_indices(dense_tokens, keep, row)
                .into_iter()
                .map(|index| index as i64)
        })
        .collect::<Vec<_>>();
    Tensor::<B, 2, Int>::from_data(TensorData::new(values, [batch, keep]), device)
}

fn bench_feature_memory_cached_sparse_update<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut group = c.benchmark_group(format!("feature_memory_cached_update_{backend_name}"));
    for case in FEATURE_MEMORY_CASES {
        let dense_tokens = case.grid.len();
        for density in DENSITY_CASES {
            let keep = keep_count(dense_tokens, density.density);
            group.throughput(Throughput::Elements((case.batch * keep) as u64));
            group.bench_function(
                format!(
                    "{}_density_{}_b{}_tokens{}_of{}",
                    case.label, density.label, case.batch, keep, dense_tokens
                ),
                |bench| {
                    bench.iter_batched(
                        || {
                            let device = make_device();
                            let indices =
                                token_indices::<B>(case.batch, dense_tokens, keep, &device);
                            let tokens =
                                Tensor::<B, 3>::ones([case.batch, keep, case.embed_dim], &device);
                            let mut memory = InterframeJepaFeatureMemory::<B>::new(
                                InterframeJepaFeatureMemoryConfig::default(),
                                case.batch,
                                case.grid,
                                case.embed_dim,
                                &device,
                            )
                            .expect("feature memory");
                            memory
                                .update_tokens(tokens.clone(), indices.clone(), case.grid)
                                .expect("prime cached plan");
                            (device, memory, tokens, indices)
                        },
                        |(device, mut memory, tokens, indices)| {
                            let output = memory
                                .update_tokens(black_box(tokens), black_box(indices), case.grid)
                                .expect("sparse update");
                            B::sync(&device).expect("sync feature memory backend");
                            black_box(output.features);
                            black_box(output.observed);
                            black_box(output.age_frames);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    group.finish();
}

fn bench_feature_memory_plan_build_update<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut group = c.benchmark_group(format!("feature_memory_plan_build_update_{backend_name}"));
    let case = FEATURE_MEMORY_CASES
        .iter()
        .copied()
        .find(|case| case.label == "vjepa224_b1")
        .expect("vjepa224 case");
    let dense_tokens = case.grid.len();
    for density in DENSITY_CASES {
        let keep = keep_count(dense_tokens, density.density);
        group.throughput(Throughput::Elements(keep as u64));
        group.bench_function(
            format!(
                "{}_density_{}_tokens{}_of{}",
                case.label, density.label, keep, dense_tokens
            ),
            |bench| {
                bench.iter_batched(
                    || {
                        let device = make_device();
                        let indices = token_indices::<B>(case.batch, dense_tokens, keep, &device);
                        let tokens =
                            Tensor::<B, 3>::ones([case.batch, keep, case.embed_dim], &device);
                        let memory = InterframeJepaFeatureMemory::<B>::new(
                            InterframeJepaFeatureMemoryConfig::default(),
                            case.batch,
                            case.grid,
                            case.embed_dim,
                            &device,
                        )
                        .expect("feature memory");
                        (device, memory, tokens, indices)
                    },
                    |(device, mut memory, tokens, indices)| {
                        let output = memory
                            .update_tokens(black_box(tokens), black_box(indices), case.grid)
                            .expect("sparse update");
                        B::sync(&device).expect("sync feature memory backend");
                        black_box(output.features);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_feature_memory_row_reset<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut group = c.benchmark_group(format!("feature_memory_row_reset_{backend_name}"));
    let case = FeatureMemoryCase {
        label: "vjepa224_b4",
        grid: TokenGridShape::new(8, 14, 14),
        embed_dim: 768,
        batch: 4,
    };
    let dense_tokens = case.grid.len();
    let keep = keep_count(dense_tokens, 0.25);
    group.throughput(Throughput::Elements(dense_tokens as u64));
    group.bench_function(
        format!("{}_one_row_dense_tokens{}", case.label, dense_tokens),
        |bench| {
            bench.iter_batched(
                || {
                    let device = make_device();
                    let indices = token_indices::<B>(case.batch, dense_tokens, keep, &device);
                    let tokens = Tensor::<B, 3>::ones([case.batch, keep, case.embed_dim], &device);
                    let mut memory = InterframeJepaFeatureMemory::<B>::new(
                        InterframeJepaFeatureMemoryConfig::default(),
                        case.batch,
                        case.grid,
                        case.embed_dim,
                        &device,
                    )
                    .expect("feature memory");
                    memory
                        .update_tokens(tokens, indices, case.grid)
                        .expect("prime memory");
                    let rows =
                        Tensor::<B, 1, Int>::from_data(TensorData::new(vec![2i64], [1]), &device);
                    (device, memory, rows)
                },
                |(device, mut memory, rows)| {
                    memory.reset_rows(black_box(rows)).expect("row reset");
                    B::sync(&device).expect("sync feature memory backend");
                    black_box(memory.snapshot().features);
                },
                BatchSize::SmallInput,
            );
        },
    );
    group.finish();
}

fn bench_feature_memory_backend<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    bench_feature_memory_cached_sparse_update::<B, _>(c, backend_name, make_device);
    bench_feature_memory_plan_build_update::<B, _>(c, backend_name, make_device);
    bench_feature_memory_row_reset::<B, _>(c, backend_name, make_device);
}

#[cfg(feature = "ndarray")]
fn feature_memory_ndarray(c: &mut Criterion) {
    bench_feature_memory_backend::<burn::backend::NdArray<f32>, _>(c, "ndarray", || {
        Default::default()
    });
}

#[cfg(not(feature = "ndarray"))]
fn feature_memory_ndarray(_c: &mut Criterion) {}

#[cfg(feature = "flex")]
fn feature_memory_flex(c: &mut Criterion) {
    bench_feature_memory_backend::<burn::backend::Flex<f32, i32>, _>(c, "flex", || {
        Default::default()
    });
}

#[cfg(not(feature = "flex"))]
fn feature_memory_flex(_c: &mut Criterion) {}

#[cfg(all(feature = "dispatch", feature = "ndarray"))]
fn feature_memory_dispatch_ndarray(c: &mut Criterion) {
    bench_feature_memory_backend::<burn::Dispatch, _>(c, "dispatch_ndarray", || {
        burn::DispatchDevice::NdArray(Default::default())
    });
}

#[cfg(not(all(feature = "dispatch", feature = "ndarray")))]
fn feature_memory_dispatch_ndarray(_c: &mut Criterion) {}

#[cfg(all(feature = "dispatch", feature = "flex"))]
fn feature_memory_dispatch_flex(c: &mut Criterion) {
    bench_feature_memory_backend::<burn::Dispatch, _>(c, "dispatch_flex", || {
        burn::DispatchDevice::Flex(Default::default())
    });
}

#[cfg(not(all(feature = "dispatch", feature = "flex")))]
fn feature_memory_dispatch_flex(_c: &mut Criterion) {}

#[cfg(feature = "cuda")]
fn feature_memory_cuda(c: &mut Criterion) {
    if let Err(reason) =
        burn_jepa::runtime::cuda_runtime_preflight(burn_jepa::runtime::CUDA_TRAIN_FORCE_ENV)
    {
        eprintln!("skipping feature_memory_cuda: {reason}");
        return;
    }
    bench_feature_memory_backend::<burn::backend::Cuda<f32, i32>, _>(c, "cuda", || {
        Default::default()
    });
}

#[cfg(not(feature = "cuda"))]
fn feature_memory_cuda(_c: &mut Criterion) {}

#[cfg(all(feature = "dispatch", feature = "cuda"))]
fn feature_memory_dispatch_cuda(c: &mut Criterion) {
    if let Err(reason) =
        burn_jepa::runtime::cuda_runtime_preflight(burn_jepa::runtime::CUDA_TRAIN_FORCE_ENV)
    {
        eprintln!("skipping feature_memory_dispatch_cuda: {reason}");
        return;
    }
    bench_feature_memory_backend::<burn::Dispatch, _>(c, "dispatch_cuda", || {
        burn::DispatchDevice::Cuda(Default::default())
    });
}

#[cfg(not(all(feature = "dispatch", feature = "cuda")))]
fn feature_memory_dispatch_cuda(_c: &mut Criterion) {}

#[cfg(feature = "wgpu")]
fn feature_memory_wgpu(c: &mut Criterion) {
    bench_feature_memory_backend::<burn::backend::Wgpu<f32, i32>, _>(c, "wgpu", || {
        Default::default()
    });
}

#[cfg(not(feature = "wgpu"))]
fn feature_memory_wgpu(_c: &mut Criterion) {}

#[cfg(feature = "webgpu")]
fn feature_memory_webgpu(c: &mut Criterion) {
    bench_feature_memory_backend::<burn::backend::WebGpu<f32, i32>, _>(c, "webgpu", || {
        Default::default()
    });
}

#[cfg(not(feature = "webgpu"))]
fn feature_memory_webgpu(_c: &mut Criterion) {}

#[cfg(all(feature = "dispatch", any(feature = "wgpu", feature = "webgpu")))]
fn feature_memory_dispatch_wgpu(c: &mut Criterion) {
    bench_feature_memory_backend::<burn::Dispatch, _>(c, "dispatch_wgpu", || {
        burn::DispatchDevice::Wgpu(Default::default())
    });
}

#[cfg(not(all(feature = "dispatch", any(feature = "wgpu", feature = "webgpu"))))]
fn feature_memory_dispatch_wgpu(_c: &mut Criterion) {}

criterion_group!(
    benches,
    feature_memory_ndarray,
    feature_memory_flex,
    feature_memory_dispatch_ndarray,
    feature_memory_dispatch_flex,
    feature_memory_cuda,
    feature_memory_dispatch_cuda,
    feature_memory_wgpu,
    feature_memory_webgpu,
    feature_memory_dispatch_wgpu
);

criterion_main!(benches);
