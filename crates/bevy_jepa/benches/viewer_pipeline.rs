use std::hint::black_box;

use bevy_jepa::{
    BevyJepaConfig, BevyJepaDisplayTransfer, BevyJepaHeadlessPipeline, BevyJepaMaskSource,
    JepaBevyBackend, JepaBevyDevice,
};
use burn::tensor::backend::Backend;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

const VIEWER_IMAGE_SIZE: usize = 64;
const VIEWER_CONTEXT_DENSITY: f32 = 0.25;

fn viewer_config(
    mask_source: BevyJepaMaskSource,
    display_transfer: BevyJepaDisplayTransfer,
) -> BevyJepaConfig {
    BevyJepaConfig {
        mask_source,
        display_transfer,
        image_size: VIEWER_IMAGE_SIZE,
        context_density: VIEWER_CONTEXT_DENSITY,
        measure_stages: true,
        sync_measurements: false,
        ..BevyJepaConfig::default()
    }
}

fn prepare_pipeline(
    mask_source: BevyJepaMaskSource,
    display_transfer: BevyJepaDisplayTransfer,
) -> (JepaBevyDevice, BevyJepaHeadlessPipeline) {
    let device = JepaBevyDevice::default();
    let mut pipeline =
        BevyJepaHeadlessPipeline::new(viewer_config(mask_source, display_transfer), device.clone());
    if mask_source == BevyJepaMaskSource::PatchDiff {
        pipeline
            .step_core_only()
            .expect("seed patch-diff previous frame");
    }
    (device, pipeline)
}

fn bench_viewer_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("bevy_jepa_viewer_pipeline_wgpu");
    group.throughput(Throughput::Elements(
        (VIEWER_IMAGE_SIZE * VIEWER_IMAGE_SIZE) as u64,
    ));

    for mask_source in [BevyJepaMaskSource::Autogaze, BevyJepaMaskSource::PatchDiff] {
        group.bench_function(format!("{mask_source}_core_only"), |bench| {
            bench.iter_batched(
                || prepare_pipeline(mask_source, BevyJepaDisplayTransfer::Gpu),
                |(device, mut pipeline)| {
                    let output = pipeline.step_core_only().expect("core viewer step");
                    JepaBevyBackend::sync(&device).expect("sync viewer backend");
                    assert!(output.metrics.aligns_with_stage_metrics());
                    assert_eq!(output.metrics.display_tensor_us, 0);
                    black_box(output.metrics);
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function(format!("{mask_source}_gpu_panels"), |bench| {
            bench.iter_batched(
                || prepare_pipeline(mask_source, BevyJepaDisplayTransfer::Gpu),
                |(device, mut pipeline)| {
                    let output = pipeline
                        .step_with_display_panels()
                        .expect("GPU display viewer step");
                    JepaBevyBackend::sync(&device).expect("sync viewer backend");
                    assert!(output.metrics.aligns_with_stage_metrics());
                    assert_eq!(
                        output.metrics.display_transfer,
                        BevyJepaDisplayTransfer::Gpu
                    );
                    assert!(output.metrics.viewer_total_us >= output.metrics.total_us);
                    black_box(output.metrics);
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function(format!("{mask_source}_cpu_panels"), |bench| {
            bench.iter_batched(
                || prepare_pipeline(mask_source, BevyJepaDisplayTransfer::Cpu),
                |(device, mut pipeline)| {
                    let output = pipeline
                        .step_with_display_panels()
                        .expect("CPU display viewer step");
                    JepaBevyBackend::sync(&device).expect("sync viewer backend");
                    assert!(output.metrics.aligns_with_stage_metrics());
                    assert_eq!(
                        output.metrics.display_transfer,
                        BevyJepaDisplayTransfer::Cpu
                    );
                    assert!(output.metrics.viewer_total_us >= output.metrics.total_us);
                    black_box(output.metrics);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_viewer_pipeline);
criterion_main!(benches);
