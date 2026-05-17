use burn::tensor::DType;
use burn::tensor::module::adaptive_avg_pool2d;
use burn::tensor::{Int, Tensor, TensorData};
use burn_anyup::{
    AnyUp, AnyUpConfig, AnyUpHighResFeatureMemory, AnyUpHighResFeatureMemoryConfig,
    AnyUpSparseOutputPlan, EfficientCrossAttentionBlock,
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

#[derive(Clone, Copy)]
struct AnyUpBenchCase {
    label: &'static str,
    image_hw: usize,
    feature_hw: usize,
    feature_dim: usize,
    batch: usize,
    q_chunk_size: Option<usize>,
    config: AnyUpBenchConfig,
    large: bool,
}

#[derive(Clone, Copy)]
enum AnyUpBenchConfig {
    Tiny,
    Default,
}

#[derive(Clone, Copy)]
struct SparseDensityCase {
    label: &'static str,
    density: f32,
}

const CASES: [AnyUpBenchCase; 11] = [
    AnyUpBenchCase {
        label: "tiny_4x",
        image_hw: 64,
        feature_hw: 16,
        feature_dim: 32,
        batch: 1,
        q_chunk_size: Some(4),
        config: AnyUpBenchConfig::Tiny,
        large: false,
    },
    AnyUpBenchCase {
        label: "image224_feat16_c64",
        image_hw: 224,
        feature_hw: 16,
        feature_dim: 64,
        batch: 1,
        q_chunk_size: Some(4),
        config: AnyUpBenchConfig::Tiny,
        large: false,
    },
    AnyUpBenchCase {
        label: "image224_feat16_c64_nochunk",
        image_hw: 224,
        feature_hw: 16,
        feature_dim: 64,
        batch: 1,
        q_chunk_size: None,
        config: AnyUpBenchConfig::Tiny,
        large: false,
    },
    AnyUpBenchCase {
        label: "image224_feat32_c64_b2",
        image_hw: 224,
        feature_hw: 32,
        feature_dim: 64,
        batch: 2,
        q_chunk_size: Some(4),
        config: AnyUpBenchConfig::Tiny,
        large: false,
    },
    AnyUpBenchCase {
        label: "jepa224_grid14_c768_chunk2",
        image_hw: 224,
        feature_hw: 14,
        feature_dim: 768,
        batch: 1,
        q_chunk_size: Some(2),
        config: AnyUpBenchConfig::Default,
        large: true,
    },
    AnyUpBenchCase {
        label: "jepa224_grid14_c768_chunk14",
        image_hw: 224,
        feature_hw: 14,
        feature_dim: 768,
        batch: 1,
        q_chunk_size: Some(14),
        config: AnyUpBenchConfig::Default,
        large: true,
    },
    AnyUpBenchCase {
        label: "jepa224_grid14_c768_nochunk",
        image_hw: 224,
        feature_hw: 14,
        feature_dim: 768,
        batch: 1,
        q_chunk_size: None,
        config: AnyUpBenchConfig::Default,
        large: true,
    },
    AnyUpBenchCase {
        label: "jepa384_grid24_c768_chunk2",
        image_hw: 384,
        feature_hw: 24,
        feature_dim: 768,
        batch: 1,
        q_chunk_size: Some(2),
        config: AnyUpBenchConfig::Default,
        large: true,
    },
    AnyUpBenchCase {
        label: "jepa384_grid24_c768_chunk8",
        image_hw: 384,
        feature_hw: 24,
        feature_dim: 768,
        batch: 1,
        q_chunk_size: Some(8),
        config: AnyUpBenchConfig::Default,
        large: true,
    },
    AnyUpBenchCase {
        label: "jepa384_grid24_c768_chunk24",
        image_hw: 384,
        feature_hw: 24,
        feature_dim: 768,
        batch: 1,
        q_chunk_size: Some(24),
        config: AnyUpBenchConfig::Default,
        large: true,
    },
    AnyUpBenchCase {
        label: "jepa384_grid24_c1024_chunk8",
        image_hw: 384,
        feature_hw: 24,
        feature_dim: 1024,
        batch: 1,
        q_chunk_size: Some(8),
        config: AnyUpBenchConfig::Default,
        large: true,
    },
];

const SPARSE_DENSITIES: [SparseDensityCase; 5] = [
    SparseDensityCase {
        label: "1pct",
        density: 0.01,
    },
    SparseDensityCase {
        label: "5pct",
        density: 0.05,
    },
    SparseDensityCase {
        label: "10pct",
        density: 0.10,
    },
    SparseDensityCase {
        label: "25pct",
        density: 0.25,
    },
    SparseDensityCase {
        label: "100pct",
        density: 1.00,
    },
];

fn sparse_keep(dense_len: usize, density: f32) -> usize {
    ((dense_len as f32) * density)
        .ceil()
        .max(1.0)
        .min(dense_len as f32) as usize
}

fn evenly_spaced_indices(dense_len: usize, keep: usize) -> Vec<usize> {
    let keep = keep.max(1).min(dense_len.max(1));
    if keep == dense_len {
        return (0..dense_len).collect();
    }
    let last = dense_len.saturating_sub(1);
    (0..keep)
        .map(|index| ((index * last) + (keep / 2)) / keep)
        .collect()
}

fn repeated_indices<B: burn::tensor::backend::Backend>(
    indices: &[usize],
    batch: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            (0..batch)
                .flat_map(|_| indices.iter().map(|&index| index as i64))
                .collect::<Vec<_>>(),
            [batch, indices.len()],
        ),
        device,
    )
}

fn should_run_sparse_case(case: AnyUpBenchCase) -> bool {
    !case.label.contains("chunk14") && !case.label.contains("nochunk")
}

fn bench_anyup_forward<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: burn::tensor::backend::Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut group = c.benchmark_group(format!("anyup_forward_{backend_name}"));
    for case in CASES {
        if case.large && std::env::var("BURN_ANYUP_BENCH_LARGE").ok().as_deref() != Some("1") {
            continue;
        }
        group.throughput(Throughput::Elements(
            (case.batch * case.image_hw * case.image_hw) as u64,
        ));
        let device = make_device();
        let model = AnyUp::<B>::new(case.config.config(), &device).expect("AnyUp model");
        let image = Tensor::<B, 4>::ones([case.batch, 3, case.image_hw, case.image_hw], &device);
        let features = Tensor::<B, 4>::ones(
            [
                case.batch,
                case.feature_dim,
                case.feature_hw,
                case.feature_hw,
            ],
            &device,
        );
        group.bench_function(
            format!(
                "{}_b{}_img{}_feat{}_c{}_chunk{:?}",
                case.label,
                case.batch,
                case.image_hw,
                case.feature_hw,
                case.feature_dim,
                case.q_chunk_size
            ),
            |bench| {
                bench.iter(|| {
                    let output = model.forward(
                        black_box(image.clone()),
                        black_box(features.clone()),
                        None,
                        case.q_chunk_size,
                    );
                    B::sync(&device).expect("sync AnyUp backend");
                    black_box(output);
                });
            },
        );
    }
    group.finish();
}

fn bench_anyup_attention<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: burn::tensor::backend::Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut group = c.benchmark_group(format!("anyup_attention_{backend_name}"));
    for case in CASES {
        if !case.large || std::env::var("BURN_ANYUP_BENCH_LARGE").ok().as_deref() != Some("1") {
            continue;
        }
        group.throughput(Throughput::Elements(
            (case.batch * case.image_hw * case.image_hw) as u64,
        ));
        let device = make_device();
        let config = case.config.config();
        let block = EfficientCrossAttentionBlock::<B>::new(
            config.qk_dim,
            config.num_heads,
            config.window_ratio,
            config.rms_norm_eps,
            &device,
        );
        let q = Tensor::<B, 4>::ones(
            [case.batch, config.qk_dim, case.image_hw, case.image_hw],
            &device,
        );
        let k = Tensor::<B, 4>::ones(
            [case.batch, config.qk_dim, case.feature_hw, case.feature_hw],
            &device,
        );
        let v = Tensor::<B, 4>::ones(
            [
                case.batch,
                case.feature_dim,
                case.feature_hw,
                case.feature_hw,
            ],
            &device,
        );
        group.bench_function(
            format!(
                "{}_b{}_img{}_feat{}_c{}_chunk{:?}",
                case.label,
                case.batch,
                case.image_hw,
                case.feature_hw,
                case.feature_dim,
                case.q_chunk_size
            ),
            |bench| {
                bench.iter(|| {
                    let output = block.forward(
                        black_box(q.clone()),
                        black_box(k.clone()),
                        black_box(v.clone()),
                        case.q_chunk_size,
                    );
                    B::sync(&device).expect("sync AnyUp backend");
                    black_box(output);
                });
            },
        );
    }
    group.finish();
}

fn bench_anyup_stages<B, MakeDevice>(c: &mut Criterion, backend_name: &str, make_device: MakeDevice)
where
    B: burn::tensor::backend::Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut encode_group = c.benchmark_group(format!("anyup_encode_{backend_name}"));
    for case in CASES {
        if !case.large || std::env::var("BURN_ANYUP_BENCH_LARGE").ok().as_deref() != Some("1") {
            continue;
        }
        encode_group.throughput(Throughput::Elements(
            (case.batch * case.image_hw * case.image_hw) as u64,
        ));
        let device = make_device();
        let model = AnyUp::<B>::new(case.config.config(), &device).expect("AnyUp model");
        let image = Tensor::<B, 4>::ones([case.batch, 3, case.image_hw, case.image_hw], &device);
        let grid = model.prepare_image_grid([case.image_hw, case.image_hw], &device);
        B::sync(&device).expect("sync AnyUp backend");
        encode_group.bench_function(
            format!("{}_b{}_img{}", case.label, case.batch, case.image_hw),
            |bench| {
                bench.iter(|| {
                    let encoded = model.encode_image(black_box(image.clone()));
                    B::sync(&device).expect("sync AnyUp backend");
                    black_box(encoded);
                });
            },
        );
        encode_group.bench_function(
            format!(
                "{}_b{}_img{}_cached_grid",
                case.label, case.batch, case.image_hw
            ),
            |bench| {
                bench.iter(|| {
                    let encoded =
                        model.encode_image_with_grid(black_box(image.clone()), black_box(&grid));
                    B::sync(&device).expect("sync AnyUp backend");
                    black_box(encoded);
                });
            },
        );
    }
    encode_group.finish();

    let mut upsample_group = c.benchmark_group(format!("anyup_upsample_{backend_name}"));
    for case in CASES {
        if !case.large || std::env::var("BURN_ANYUP_BENCH_LARGE").ok().as_deref() != Some("1") {
            continue;
        }
        upsample_group.throughput(Throughput::Elements(
            (case.batch * case.image_hw * case.image_hw) as u64,
        ));
        let device = make_device();
        let model = AnyUp::<B>::new(case.config.config(), &device).expect("AnyUp model");
        let image = Tensor::<B, 4>::ones([case.batch, 3, case.image_hw, case.image_hw], &device);
        let features = Tensor::<B, 4>::ones(
            [
                case.batch,
                case.feature_dim,
                case.feature_hw,
                case.feature_hw,
            ],
            &device,
        );
        let encoded = model.encode_image(image);
        B::sync(&device).expect("sync AnyUp backend");
        upsample_group.bench_function(
            format!(
                "{}_b{}_img{}_feat{}_c{}_chunk{:?}",
                case.label,
                case.batch,
                case.image_hw,
                case.feature_hw,
                case.feature_dim,
                case.q_chunk_size
            ),
            |bench| {
                bench.iter(|| {
                    let output = model.upsample(
                        black_box(encoded.clone()),
                        black_box(features.clone()),
                        [case.image_hw, case.image_hw],
                        case.q_chunk_size,
                    );
                    B::sync(&device).expect("sync AnyUp backend");
                    black_box(output);
                });
            },
        );
    }
    upsample_group.finish();
}

fn bench_anyup_upsample_parts<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: burn::tensor::backend::Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut group = c.benchmark_group(format!("anyup_upsample_parts_{backend_name}"));
    for case in CASES {
        if !case.large
            || case.label.contains("chunk14")
            || case.label.contains("nochunk")
            || std::env::var("BURN_ANYUP_BENCH_LARGE").ok().as_deref() != Some("1")
        {
            continue;
        }
        group.throughput(Throughput::Elements(
            (case.batch * case.image_hw * case.image_hw) as u64,
        ));
        let device = make_device();
        let model = AnyUp::<B>::new(case.config.config(), &device).expect("AnyUp model");
        let image = Tensor::<B, 4>::ones([case.batch, 3, case.image_hw, case.image_hw], &device);
        let encoded = model.encode_image(image);
        let features = Tensor::<B, 4>::ones(
            [
                case.batch,
                case.feature_dim,
                case.feature_hw,
                case.feature_hw,
            ],
            &device,
        );
        let key_image = adaptive_avg_pool2d(
            model.key_encoder.forward(encoded.clone()),
            [case.feature_hw, case.feature_hw],
        );
        let key_features = model.key_features_encoder.forward(features.clone());
        B::sync(&device).expect("sync AnyUp backend");

        group.bench_function(format!("{}_query_encoder", case.label), |bench| {
            bench.iter(|| {
                let q = model.query_encoder.forward(black_box(encoded.clone()));
                B::sync(&device).expect("sync AnyUp backend");
                black_box(q);
            });
        });
        group.bench_function(format!("{}_key_encoder_pool", case.label), |bench| {
            bench.iter(|| {
                let k = adaptive_avg_pool2d(
                    model.key_encoder.forward(black_box(encoded.clone())),
                    [case.feature_hw, case.feature_hw],
                );
                B::sync(&device).expect("sync AnyUp backend");
                black_box(k);
            });
        });
        group.bench_function(format!("{}_key_features_encoder", case.label), |bench| {
            bench.iter(|| {
                let k = model
                    .key_features_encoder
                    .forward(black_box(features.clone()));
                B::sync(&device).expect("sync AnyUp backend");
                black_box(k);
            });
        });
        group.bench_function(format!("{}_aggregation", case.label), |bench| {
            bench.iter(|| {
                let k = model.aggregation.forward(Tensor::cat(
                    vec![
                        black_box(key_image.clone()),
                        black_box(key_features.clone()),
                    ],
                    1,
                ));
                B::sync(&device).expect("sync AnyUp backend");
                black_box(k);
            });
        });
    }
    group.finish();
}

fn bench_anyup_context<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: burn::tensor::backend::Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut prepare_group = c.benchmark_group(format!("anyup_prepare_context_{backend_name}"));
    for case in CASES {
        if !case.large
            || case.label.contains("chunk14")
            || case.label.contains("nochunk")
            || std::env::var("BURN_ANYUP_BENCH_LARGE").ok().as_deref() != Some("1")
        {
            continue;
        }
        prepare_group.throughput(Throughput::Elements(
            (case.batch * case.image_hw * case.image_hw) as u64,
        ));
        let device = make_device();
        let model = AnyUp::<B>::new(case.config.config(), &device).expect("AnyUp model");
        let image = Tensor::<B, 4>::ones([case.batch, 3, case.image_hw, case.image_hw], &device);
        let grid = model.prepare_image_grid([case.image_hw, case.image_hw], &device);
        B::sync(&device).expect("sync AnyUp backend");
        prepare_group.bench_function(format!("{}_prepare", case.label), |bench| {
            bench.iter(|| {
                let context = model.prepare_image_context(
                    black_box(image.clone()),
                    Some([case.image_hw, case.image_hw]),
                    [case.feature_hw, case.feature_hw],
                );
                B::sync(&device).expect("sync AnyUp backend");
                black_box(context);
            });
        });
        prepare_group.bench_function(format!("{}_prepare_cached_grid", case.label), |bench| {
            bench.iter(|| {
                let context = model.prepare_image_context_with_grid(
                    black_box(image.clone()),
                    black_box(&grid),
                    Some([case.image_hw, case.image_hw]),
                    [case.feature_hw, case.feature_hw],
                );
                B::sync(&device).expect("sync AnyUp backend");
                black_box(context);
            });
        });
    }
    prepare_group.finish();

    let mut decode_group = c.benchmark_group(format!("anyup_context_decode_{backend_name}"));
    for case in CASES {
        if !case.large || std::env::var("BURN_ANYUP_BENCH_LARGE").ok().as_deref() != Some("1") {
            continue;
        }
        decode_group.throughput(Throughput::Elements(
            (case.batch * case.image_hw * case.image_hw) as u64,
        ));
        let device = make_device();
        let model = AnyUp::<B>::new(case.config.config(), &device).expect("AnyUp model");
        let image = Tensor::<B, 4>::ones([case.batch, 3, case.image_hw, case.image_hw], &device);
        let features = Tensor::<B, 4>::ones(
            [
                case.batch,
                case.feature_dim,
                case.feature_hw,
                case.feature_hw,
            ],
            &device,
        );
        let context = model.prepare_image_context(
            image,
            Some([case.image_hw, case.image_hw]),
            [case.feature_hw, case.feature_hw],
        );
        B::sync(&device).expect("sync AnyUp backend");
        decode_group.bench_function(format!("{}_decode", case.label), |bench| {
            bench.iter(|| {
                let output = model.upsample_with_context(
                    black_box(&context),
                    black_box(features.clone()),
                    case.q_chunk_size,
                );
                B::sync(&device).expect("sync AnyUp backend");
                black_box(output);
            });
        });
    }
    decode_group.finish();
}

fn bench_anyup_sparse_context<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: burn::tensor::backend::Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut group = c.benchmark_group(format!("anyup_sparse_context_decode_{backend_name}"));
    for case in CASES {
        if !should_run_sparse_case(case)
            || (case.large && std::env::var("BURN_ANYUP_BENCH_LARGE").ok().as_deref() != Some("1"))
        {
            continue;
        }
        let dense_len = case.image_hw * case.image_hw;
        for density in SPARSE_DENSITIES {
            let keep = sparse_keep(dense_len, density.density);
            group.throughput(Throughput::Elements((case.batch * keep) as u64));
            let device = make_device();
            let model = AnyUp::<B>::new(case.config.config(), &device).expect("AnyUp model");
            let image =
                Tensor::<B, 4>::ones([case.batch, 3, case.image_hw, case.image_hw], &device);
            let features = Tensor::<B, 4>::ones(
                [
                    case.batch,
                    case.feature_dim,
                    case.feature_hw,
                    case.feature_hw,
                ],
                &device,
            );
            let context = model.prepare_image_context(
                image,
                Some([case.image_hw, case.image_hw]),
                [case.feature_hw, case.feature_hw],
            );
            let plan = AnyUpSparseOutputPlan::<B>::new(
                evenly_spaced_indices(dense_len, keep),
                [case.image_hw, case.image_hw],
                [case.feature_hw, case.feature_hw],
                case.batch,
                case.config.config().window_ratio,
                &device,
            )
            .expect("sparse AnyUp plan");
            B::sync(&device).expect("sync AnyUp backend");
            group.bench_function(
                format!(
                    "{}_density_{}_b{}_pixels{}_of{}_c{}",
                    case.label, density.label, case.batch, keep, dense_len, case.feature_dim
                ),
                |bench| {
                    bench.iter(|| {
                        let output = model
                            .upsample_sparse_with_context(
                                black_box(&context),
                                black_box(features.clone()),
                                black_box(&plan),
                            )
                            .expect("sparse AnyUp decode");
                        B::sync(&device).expect("sync AnyUp backend");
                        black_box(output.features);
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_anyup_sparse_low_feature_context<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: burn::tensor::backend::Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut group = c.benchmark_group(format!("anyup_sparse_low_feature_decode_{backend_name}"));
    for case in CASES {
        if !should_run_sparse_case(case)
            || (case.large && std::env::var("BURN_ANYUP_BENCH_LARGE").ok().as_deref() != Some("1"))
        {
            continue;
        }
        let high_dense_len = case.image_hw * case.image_hw;
        let high_keep = sparse_keep(high_dense_len, 0.10);
        let low_dense_len = case.feature_hw * case.feature_hw;
        for density in SPARSE_DENSITIES {
            let low_keep = sparse_keep(low_dense_len, density.density);
            group.throughput(Throughput::Elements((case.batch * high_keep) as u64));
            let device = make_device();
            let model = AnyUp::<B>::new(case.config.config(), &device).expect("AnyUp model");
            let image =
                Tensor::<B, 4>::ones([case.batch, 3, case.image_hw, case.image_hw], &device);
            let context = model.prepare_image_context(
                image,
                Some([case.image_hw, case.image_hw]),
                [case.feature_hw, case.feature_hw],
            );
            let sparse_features =
                Tensor::<B, 3>::ones([case.batch, low_keep, case.feature_dim], &device);
            let low_indices = repeated_indices::<B>(
                &evenly_spaced_indices(low_dense_len, low_keep),
                case.batch,
                &device,
            );
            let plan = AnyUpSparseOutputPlan::<B>::new(
                evenly_spaced_indices(high_dense_len, high_keep),
                [case.image_hw, case.image_hw],
                [case.feature_hw, case.feature_hw],
                case.batch,
                case.config.config().window_ratio,
                &device,
            )
            .expect("sparse AnyUp plan");
            B::sync(&device).expect("sync AnyUp backend");
            group.bench_function(
                format!(
                    "{}_low_density_{}_high10pct_b{}_low{}_of{}_high{}_of{}",
                    case.label,
                    density.label,
                    case.batch,
                    low_keep,
                    low_dense_len,
                    high_keep,
                    high_dense_len
                ),
                |bench| {
                    bench.iter(|| {
                        let output = model
                            .upsample_sparse_low_features_with_context(
                                black_box(&context),
                                black_box(sparse_features.clone()),
                                black_box(low_indices.clone()),
                                black_box(&plan),
                            )
                            .expect("sparse low feature AnyUp decode");
                        B::sync(&device).expect("sync AnyUp backend");
                        black_box(output.features);
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_anyup_highres_update<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: burn::tensor::backend::Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut group = c.benchmark_group(format!("anyup_sparse_highres_update_{backend_name}"));
    for case in CASES {
        if !should_run_sparse_case(case)
            || (case.large && std::env::var("BURN_ANYUP_BENCH_LARGE").ok().as_deref() != Some("1"))
        {
            continue;
        }
        let dense_len = case.image_hw * case.image_hw;
        for density in SPARSE_DENSITIES {
            let keep = sparse_keep(dense_len, density.density);
            group.throughput(Throughput::Elements((case.batch * keep) as u64));
            let device = make_device();
            let mut memory = AnyUpHighResFeatureMemory::<B>::new(
                AnyUpHighResFeatureMemoryConfig::default(),
                case.batch,
                [case.image_hw, case.image_hw],
                case.feature_dim,
                &device,
            )
            .expect("AnyUp high-res feature memory");
            let tokens = Tensor::<B, 3>::ones([case.batch, keep, case.feature_dim], &device);
            let indices =
                repeated_indices::<B>(&evenly_spaced_indices(dense_len, keep), case.batch, &device);
            memory
                .update_tokens(tokens.clone(), indices.clone())
                .expect("prime sparse high-res update plan");
            B::sync(&device).expect("sync AnyUp backend");
            group.bench_function(
                format!(
                    "{}_density_{}_b{}_pixels{}_of{}_c{}",
                    case.label, density.label, case.batch, keep, dense_len, case.feature_dim
                ),
                |bench| {
                    bench.iter(|| {
                        let output = memory
                            .update_tokens(black_box(tokens.clone()), black_box(indices.clone()))
                            .expect("sparse high-res update");
                        B::sync(&device).expect("sync AnyUp backend");
                        black_box(output.features);
                    });
                },
            );
        }
    }
    group.finish();
}

#[allow(dead_code)]
fn bench_anyup_low_precision<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    dtype: DType,
    make_device: MakeDevice,
) where
    B: burn::tensor::backend::Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    if std::env::var("BURN_ANYUP_BENCH_LOW_PRECISION")
        .ok()
        .as_deref()
        != Some("1")
    {
        return;
    }

    let device = make_device();
    if !B::supports_dtype(&device, dtype) {
        eprintln!("skipping {backend_name}: backend does not report {dtype:?} support");
        return;
    }

    bench_anyup_forward::<B, _>(c, backend_name, make_device);
    bench_anyup_attention::<B, _>(c, backend_name, make_device);
    bench_anyup_stages::<B, _>(c, backend_name, make_device);
    bench_anyup_upsample_parts::<B, _>(c, backend_name, make_device);
    bench_anyup_context::<B, _>(c, backend_name, make_device);
    bench_anyup_sparse_context::<B, _>(c, backend_name, make_device);
    bench_anyup_sparse_low_feature_context::<B, _>(c, backend_name, make_device);
    bench_anyup_highres_update::<B, _>(c, backend_name, make_device);
}

impl AnyUpBenchConfig {
    fn config(self) -> AnyUpConfig {
        match self {
            AnyUpBenchConfig::Tiny => AnyUpConfig::tiny_for_tests(),
            AnyUpBenchConfig::Default => AnyUpConfig::default(),
        }
    }
}

#[cfg(feature = "ndarray")]
fn anyup_forward_ndarray(c: &mut Criterion) {
    bench_anyup_forward::<burn::backend::NdArray<f32>, _>(c, "ndarray", Default::default);
    bench_anyup_attention::<burn::backend::NdArray<f32>, _>(c, "ndarray", Default::default);
    bench_anyup_stages::<burn::backend::NdArray<f32>, _>(c, "ndarray", Default::default);
    bench_anyup_upsample_parts::<burn::backend::NdArray<f32>, _>(c, "ndarray", Default::default);
    bench_anyup_context::<burn::backend::NdArray<f32>, _>(c, "ndarray", Default::default);
    bench_anyup_sparse_context::<burn::backend::NdArray<f32>, _>(c, "ndarray", Default::default);
    bench_anyup_sparse_low_feature_context::<burn::backend::NdArray<f32>, _>(
        c,
        "ndarray",
        Default::default,
    );
    bench_anyup_highres_update::<burn::backend::NdArray<f32>, _>(c, "ndarray", Default::default);
}

#[cfg(not(feature = "ndarray"))]
fn anyup_forward_ndarray(_c: &mut Criterion) {}

#[cfg(feature = "flex")]
fn anyup_forward_flex(c: &mut Criterion) {
    bench_anyup_forward::<burn::backend::Flex<f32, i32>, _>(c, "flex", Default::default);
    bench_anyup_attention::<burn::backend::Flex<f32, i32>, _>(c, "flex", Default::default);
    bench_anyup_stages::<burn::backend::Flex<f32, i32>, _>(c, "flex", Default::default);
    bench_anyup_upsample_parts::<burn::backend::Flex<f32, i32>, _>(c, "flex", Default::default);
    bench_anyup_context::<burn::backend::Flex<f32, i32>, _>(c, "flex", Default::default);
    bench_anyup_sparse_context::<burn::backend::Flex<f32, i32>, _>(c, "flex", Default::default);
    bench_anyup_sparse_low_feature_context::<burn::backend::Flex<f32, i32>, _>(
        c,
        "flex",
        Default::default,
    );
    bench_anyup_highres_update::<burn::backend::Flex<f32, i32>, _>(c, "flex", Default::default);
}

#[cfg(not(feature = "flex"))]
fn anyup_forward_flex(_c: &mut Criterion) {}

#[cfg(all(feature = "dispatch", feature = "flex"))]
fn anyup_forward_dispatch_flex(c: &mut Criterion) {
    bench_anyup_forward::<burn::Dispatch, _>(c, "dispatch_flex", || {
        burn::DispatchDevice::Flex(Default::default())
    });
    bench_anyup_attention::<burn::Dispatch, _>(c, "dispatch_flex", || {
        burn::DispatchDevice::Flex(Default::default())
    });
    bench_anyup_stages::<burn::Dispatch, _>(c, "dispatch_flex", || {
        burn::DispatchDevice::Flex(Default::default())
    });
    bench_anyup_upsample_parts::<burn::Dispatch, _>(c, "dispatch_flex", || {
        burn::DispatchDevice::Flex(Default::default())
    });
    bench_anyup_context::<burn::Dispatch, _>(c, "dispatch_flex", || {
        burn::DispatchDevice::Flex(Default::default())
    });
    bench_anyup_sparse_context::<burn::Dispatch, _>(c, "dispatch_flex", || {
        burn::DispatchDevice::Flex(Default::default())
    });
    bench_anyup_sparse_low_feature_context::<burn::Dispatch, _>(c, "dispatch_flex", || {
        burn::DispatchDevice::Flex(Default::default())
    });
    bench_anyup_highres_update::<burn::Dispatch, _>(c, "dispatch_flex", || {
        burn::DispatchDevice::Flex(Default::default())
    });
}

#[cfg(not(all(feature = "dispatch", feature = "flex")))]
fn anyup_forward_dispatch_flex(_c: &mut Criterion) {}

#[cfg(feature = "webgpu")]
fn anyup_forward_webgpu(c: &mut Criterion) {
    bench_anyup_forward::<burn::backend::WebGpu<f32, i32>, _>(c, "webgpu", Default::default);
    bench_anyup_attention::<burn::backend::WebGpu<f32, i32>, _>(c, "webgpu", Default::default);
    bench_anyup_stages::<burn::backend::WebGpu<f32, i32>, _>(c, "webgpu", Default::default);
    bench_anyup_upsample_parts::<burn::backend::WebGpu<f32, i32>, _>(c, "webgpu", Default::default);
    bench_anyup_context::<burn::backend::WebGpu<f32, i32>, _>(c, "webgpu", Default::default);
    bench_anyup_sparse_context::<burn::backend::WebGpu<f32, i32>, _>(c, "webgpu", Default::default);
    bench_anyup_sparse_low_feature_context::<burn::backend::WebGpu<f32, i32>, _>(
        c,
        "webgpu",
        Default::default,
    );
    bench_anyup_highres_update::<burn::backend::WebGpu<f32, i32>, _>(c, "webgpu", Default::default);
}

#[cfg(not(feature = "webgpu"))]
fn anyup_forward_webgpu(_c: &mut Criterion) {}

#[cfg(feature = "webgpu")]
fn anyup_forward_webgpu_f16(c: &mut Criterion) {
    bench_anyup_low_precision::<burn::backend::WebGpu<burn::tensor::f16, i32>, _>(
        c,
        "webgpu_f16",
        DType::F16,
        Default::default,
    );
}

#[cfg(not(feature = "webgpu"))]
fn anyup_forward_webgpu_f16(_c: &mut Criterion) {}

#[cfg(feature = "webgpu")]
fn anyup_forward_webgpu_bf16(c: &mut Criterion) {
    bench_anyup_low_precision::<burn::backend::WebGpu<burn::tensor::bf16, i32>, _>(
        c,
        "webgpu_bf16",
        DType::BF16,
        Default::default,
    );
}

#[cfg(not(feature = "webgpu"))]
fn anyup_forward_webgpu_bf16(_c: &mut Criterion) {}

#[cfg(feature = "wgpu")]
fn anyup_forward_wgpu(c: &mut Criterion) {
    bench_anyup_forward::<burn::backend::Wgpu<f32, i32>, _>(c, "wgpu", Default::default);
    bench_anyup_attention::<burn::backend::Wgpu<f32, i32>, _>(c, "wgpu", Default::default);
    bench_anyup_stages::<burn::backend::Wgpu<f32, i32>, _>(c, "wgpu", Default::default);
    bench_anyup_upsample_parts::<burn::backend::Wgpu<f32, i32>, _>(c, "wgpu", Default::default);
    bench_anyup_context::<burn::backend::Wgpu<f32, i32>, _>(c, "wgpu", Default::default);
    bench_anyup_sparse_context::<burn::backend::Wgpu<f32, i32>, _>(c, "wgpu", Default::default);
    bench_anyup_sparse_low_feature_context::<burn::backend::Wgpu<f32, i32>, _>(
        c,
        "wgpu",
        Default::default,
    );
    bench_anyup_highres_update::<burn::backend::Wgpu<f32, i32>, _>(c, "wgpu", Default::default);
}

#[cfg(not(feature = "wgpu"))]
fn anyup_forward_wgpu(_c: &mut Criterion) {}

#[cfg(feature = "cuda")]
fn anyup_forward_cuda(c: &mut Criterion) {
    bench_anyup_forward::<burn::backend::Cuda<f32, i32>, _>(c, "cuda", Default::default);
    bench_anyup_attention::<burn::backend::Cuda<f32, i32>, _>(c, "cuda", Default::default);
    bench_anyup_stages::<burn::backend::Cuda<f32, i32>, _>(c, "cuda", Default::default);
    bench_anyup_upsample_parts::<burn::backend::Cuda<f32, i32>, _>(c, "cuda", Default::default);
    bench_anyup_context::<burn::backend::Cuda<f32, i32>, _>(c, "cuda", Default::default);
    bench_anyup_sparse_context::<burn::backend::Cuda<f32, i32>, _>(c, "cuda", Default::default);
    bench_anyup_sparse_low_feature_context::<burn::backend::Cuda<f32, i32>, _>(
        c,
        "cuda",
        Default::default,
    );
    bench_anyup_highres_update::<burn::backend::Cuda<f32, i32>, _>(c, "cuda", Default::default);
}

#[cfg(not(feature = "cuda"))]
fn anyup_forward_cuda(_c: &mut Criterion) {}

#[cfg(feature = "cuda")]
fn anyup_forward_cuda_f16(c: &mut Criterion) {
    bench_anyup_low_precision::<burn::backend::Cuda<burn::tensor::f16, i32>, _>(
        c,
        "cuda_f16",
        DType::F16,
        Default::default,
    );
}

#[cfg(not(feature = "cuda"))]
fn anyup_forward_cuda_f16(_c: &mut Criterion) {}

#[cfg(feature = "cuda")]
fn anyup_forward_cuda_bf16(c: &mut Criterion) {
    bench_anyup_low_precision::<burn::backend::Cuda<burn::tensor::bf16, i32>, _>(
        c,
        "cuda_bf16",
        DType::BF16,
        Default::default,
    );
}

#[cfg(not(feature = "cuda"))]
fn anyup_forward_cuda_bf16(_c: &mut Criterion) {}

criterion_group!(
    benches,
    anyup_forward_ndarray,
    anyup_forward_flex,
    anyup_forward_dispatch_flex,
    anyup_forward_webgpu,
    anyup_forward_webgpu_f16,
    anyup_forward_webgpu_bf16,
    anyup_forward_wgpu,
    anyup_forward_cuda,
    anyup_forward_cuda_f16,
    anyup_forward_cuda_bf16
);
criterion_main!(benches);
