use burn::tensor::Tensor;
use burn_jepa::{SparsePredictorPlan, SparseTokenMask, VJepa2_1Model, VJepaConfig};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::collections::BTreeSet;

type B = burn::backend::NdArray<f32>;

fn bench_sparse_forward(c: &mut Criterion) {
    let device = Default::default();
    let config = bench_config();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let grid = config.token_grid();
    let (context, target) = sparse_pair(grid.len(), 16, 8);

    c.bench_function("sparse_vjepa_tiny_forward_ndarray", |b| {
        b.iter_batched(
            || Tensor::<B, 5>::zeros([1, 3, 4, 64, 64], &device),
            |video| {
                model
                    .predict_dense_targets(video, &context, &target)
                    .expect("forward")
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_sparse_predictor_hot_path(c: &mut Criterion) {
    let device = Default::default();
    let config = bench_config();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let grid = config.token_grid();
    let video = Tensor::<B, 5>::zeros([1, 3, 4, 64, 64], &device);

    let mut group = c.benchmark_group("sparse_predictor_hot_path_ndarray");
    for context_keep in [8, 16, 24] {
        let (context, target) = sparse_pair(grid.len(), context_keep, 8);
        let context_tokens = model.encode_video(video.clone(), Some(&context)).tokens;
        let plan =
            SparsePredictorPlan::new(&config, context, target, grid, 1, &device).expect("plan");
        let sequence_tokens = context_keep + 8;
        group.throughput(Throughput::Elements(sequence_tokens as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{sequence_tokens}_sequence_tokens")),
            &context_tokens,
            |b, context_tokens| {
                b.iter_batched(
                    || context_tokens.clone(),
                    |tokens| {
                        model
                            .predictor
                            .forward_sparse_with_plan(tokens, &plan, 0)
                            .expect("predictor")
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_config() -> VJepaConfig {
    let mut config = VJepaConfig::tiny_for_tests();
    config.image_size = 64;
    config.num_frames = 4;
    config
}

fn sparse_pair(
    dense_len: usize,
    context_keep: usize,
    target_keep: usize,
) -> (SparseTokenMask, SparseTokenMask) {
    let context = SparseTokenMask::evenly_spaced(dense_len, context_keep);
    let context_set = context.indices().iter().copied().collect::<BTreeSet<_>>();
    let target = (0..dense_len)
        .filter(|index| !context_set.contains(index))
        .take(target_keep)
        .collect::<Vec<_>>();
    (
        context,
        SparseTokenMask::new(target, dense_len).expect("target mask"),
    )
}

criterion_group!(
    benches,
    bench_sparse_forward,
    bench_sparse_predictor_hot_path
);
criterion_main!(benches);
