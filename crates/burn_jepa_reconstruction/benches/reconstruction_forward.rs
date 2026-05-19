use burn::tensor::{Distribution, Tensor, backend::Backend};
use burn_jepa_reconstruction::{JepaReconstructionConfig, JepaReconstructionDecoder};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

#[derive(Clone, Copy)]
struct ReconstructionBenchCase {
    label: &'static str,
    image_hw: usize,
    grid_hw: usize,
    feature_dim: usize,
    hidden_dim: usize,
}

const CASES: [ReconstructionBenchCase; 5] = [
    ReconstructionBenchCase {
        label: "jepa256_grid16_c768_h128",
        image_hw: 256,
        grid_hw: 16,
        feature_dim: 768,
        hidden_dim: 128,
    },
    ReconstructionBenchCase {
        label: "jepa384_grid24_c768_h128",
        image_hw: 384,
        grid_hw: 24,
        feature_dim: 768,
        hidden_dim: 128,
    },
    ReconstructionBenchCase {
        label: "jepa512_grid32_c768_h128",
        image_hw: 512,
        grid_hw: 32,
        feature_dim: 768,
        hidden_dim: 128,
    },
    ReconstructionBenchCase {
        label: "jepa1024_grid64_c768_h128",
        image_hw: 1024,
        grid_hw: 64,
        feature_dim: 768,
        hidden_dim: 128,
    },
    ReconstructionBenchCase {
        label: "tiny64_grid4_c32",
        image_hw: 64,
        grid_hw: 4,
        feature_dim: 32,
        hidden_dim: 32,
    },
];

fn bench_reconstruction_forward<B, MakeDevice>(
    c: &mut Criterion,
    backend_name: &str,
    make_device: MakeDevice,
) where
    B: Backend,
    MakeDevice: Fn() -> B::Device + Copy,
{
    let mut group = c.benchmark_group(format!("jepa_reconstruction_forward_{backend_name}"));
    for case in CASES {
        if case.image_hw == 1024
            && std::env::var("BURN_JEPA_RECONSTRUCTION_BENCH_1024")
                .ok()
                .as_deref()
                != Some("1")
        {
            continue;
        }
        if (384..1024).contains(&case.image_hw)
            && std::env::var("BURN_JEPA_RECONSTRUCTION_BENCH_LARGE")
                .ok()
                .as_deref()
                != Some("1")
        {
            continue;
        }
        let device = make_device();
        let config = JepaReconstructionConfig {
            input_dim: case.feature_dim,
            hidden_dim: case.hidden_dim,
            patch_size: case.image_hw / case.grid_hw,
            ..JepaReconstructionConfig::default()
        };
        let decoder =
            JepaReconstructionDecoder::<B>::new(config, &device).expect("reconstruction decoder");
        let features = Tensor::<B, 4>::random(
            [1, case.feature_dim, case.grid_hw, case.grid_hw],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        group.throughput(Throughput::Elements((case.image_hw * case.image_hw) as u64));
        group.bench_function(case.label, |bench| {
            bench.iter(|| {
                let output =
                    decoder.forward_to_size(features.clone(), [case.image_hw, case.image_hw]);
                B::sync(&device).expect("sync reconstruction backend");
                black_box(output)
            });
        });
    }
    group.finish();
}

#[cfg(feature = "ndarray")]
fn bench_ndarray(c: &mut Criterion) {
    bench_reconstruction_forward::<burn::backend::NdArray<f32>, _>(c, "ndarray", Default::default);
}

#[cfg(feature = "webgpu")]
fn bench_webgpu(c: &mut Criterion) {
    bench_reconstruction_forward::<burn::backend::WebGpu<f32, i32>, _>(
        c,
        "webgpu",
        Default::default,
    );
}

#[cfg(all(feature = "ndarray", feature = "webgpu"))]
criterion_group!(benches, bench_ndarray, bench_webgpu);
#[cfg(all(feature = "ndarray", not(feature = "webgpu")))]
criterion_group!(benches, bench_ndarray);
#[cfg(all(feature = "webgpu", not(feature = "ndarray")))]
criterion_group!(benches, bench_webgpu);
#[cfg(not(any(feature = "ndarray", feature = "webgpu")))]
fn bench_no_backend(_c: &mut Criterion) {}
#[cfg(not(any(feature = "ndarray", feature = "webgpu")))]
criterion_group!(benches, bench_no_backend);
criterion_main!(benches);
