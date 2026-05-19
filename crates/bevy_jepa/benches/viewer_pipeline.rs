use std::hint::black_box;

use bevy_jepa::{
    BevyJepaConfig, BevyJepaDisplayTransfer, BevyJepaEncoderSource, BevyJepaFrameSource,
    BevyJepaHeadlessPipeline, BevyJepaMaskSource, DEFAULT_IMAGE_SIZE, FeatureFrameViewerConfig,
    JepaBevyBackend, JepaBevyDevice,
};
use burn::tensor::backend::Backend;
use burn_jepa::{
    BurnJepaReconstructionPackageManifest, FeatureFrameRequest, JepaReconstructionConfig,
    JepaReconstructionDecoder, VJepaConfig, save_jepa_reconstruction_burnpack,
    write_burnpack_parts_for_browser, write_jepa_reconstruction_package_manifest,
};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use std::path::PathBuf;
use tempfile::TempDir;

const VIEWER_CONTEXT_DENSITY: f32 = 1.0;

#[derive(Clone, Copy)]
struct ViewerMaskCase {
    label: &'static str,
    source: BevyJepaMaskSource,
    patch_diff_threshold: f32,
}

const VIEWER_MASK_CASES: [ViewerMaskCase; 2] = [
    ViewerMaskCase {
        label: "patch_diff_t003",
        source: BevyJepaMaskSource::PatchDiff,
        patch_diff_threshold: 0.03,
    },
    ViewerMaskCase {
        label: "patch_diff_t000",
        source: BevyJepaMaskSource::PatchDiff,
        patch_diff_threshold: 0.0,
    },
];

#[derive(Clone, Copy)]
enum ViewerBenchLane {
    Stage {
        name: &'static str,
        request: FeatureFrameRequest,
    },
    Reconstruction {
        name: &'static str,
        request: FeatureFrameRequest,
    },
    Display {
        name: &'static str,
        request: FeatureFrameRequest,
        transfer: BevyJepaDisplayTransfer,
    },
}

const VIEWER_BENCH_LANES: [ViewerBenchLane; 6] = [
    ViewerBenchLane::Stage {
        name: "low_res_cache_update",
        request: FeatureFrameRequest::none(),
    },
    ViewerBenchLane::Stage {
        name: "pca_projection",
        request: FeatureFrameRequest::low_res(),
    },
    ViewerBenchLane::Stage {
        name: "full_anyup_decode",
        request: FeatureFrameRequest::high_res_features(),
    },
    ViewerBenchLane::Reconstruction {
        name: "reconstruction_decode",
        request: FeatureFrameRequest::low_res(),
    },
    ViewerBenchLane::Display {
        name: "display_upload_gpu",
        request: FeatureFrameRequest::low_res(),
        transfer: BevyJepaDisplayTransfer::Gpu,
    },
    ViewerBenchLane::Display {
        name: "display_upload_cpu",
        request: FeatureFrameRequest::low_res(),
        transfer: BevyJepaDisplayTransfer::Cpu,
    },
];

fn viewer_config(
    image_size: usize,
    mask_source: BevyJepaMaskSource,
    patch_diff_threshold: f32,
    display_transfer: BevyJepaDisplayTransfer,
) -> BevyJepaConfig {
    BevyJepaConfig {
        encoder_source: BevyJepaEncoderSource::TinyTest,
        ttt_model_path: None,
        jepa_checkpoint_dir: None,
        jepa_config_path: None,
        source: BevyJepaFrameSource::SyntheticLocalMotion,
        mask_source,
        display_transfer,
        pipeline: FeatureFrameViewerConfig {
            image_size,
            context_density: VIEWER_CONTEXT_DENSITY,
            min_context_density: 1.0,
            bootstrap_context_density: 1.0,
            patch_diff_threshold,
            measure_stages: true,
            sync_measurements: true,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    }
}

fn prepare_pipeline(
    image_size: usize,
    mask_source: BevyJepaMaskSource,
    patch_diff_threshold: f32,
    display_transfer: BevyJepaDisplayTransfer,
) -> (JepaBevyDevice, BevyJepaHeadlessPipeline) {
    let device = JepaBevyDevice::default();
    let mut pipeline = BevyJepaHeadlessPipeline::new(
        viewer_config(
            image_size,
            mask_source,
            patch_diff_threshold,
            display_transfer,
        ),
        device.clone(),
    );
    if mask_source == BevyJepaMaskSource::PatchDiff {
        pipeline
            .step_stage_only()
            .expect("seed patch-diff previous frame");
    }
    (device, pipeline)
}

fn prepare_reconstruction_pipeline(
    image_size: usize,
    mask_source: BevyJepaMaskSource,
    patch_diff_threshold: f32,
    reconstruction_model_manifest_path: PathBuf,
) -> (JepaBevyDevice, BevyJepaHeadlessPipeline) {
    let device = JepaBevyDevice::default();
    let mut config = viewer_config(
        image_size,
        mask_source,
        patch_diff_threshold,
        BevyJepaDisplayTransfer::Gpu,
    );
    config.reconstruction_every = 1;
    config.high_res_pca_every = 0;
    config.sync_measurements = false;
    config.reconstruction_model_manifest_path = Some(reconstruction_model_manifest_path);
    config.reconstruction_model_auto_download = false;
    let mut pipeline = BevyJepaHeadlessPipeline::new(config, device.clone());
    if mask_source == BevyJepaMaskSource::PatchDiff {
        pipeline
            .step_stage_only()
            .expect("seed patch-diff previous frame");
    }
    (device, pipeline)
}

fn write_tiny_reconstruction_package(device: &JepaBevyDevice) -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("temp reconstruction package dir");
    let config = JepaReconstructionConfig {
        input_dim: VJepaConfig::tiny_for_tests().encoder.embed_dim,
        patch_size: VJepaConfig::tiny_for_tests().patch_size,
        ..JepaReconstructionConfig::tiny_for_tests()
    };
    let decoder = JepaReconstructionDecoder::<JepaBevyBackend>::new(config.clone(), device)
        .expect("tiny reconstruction decoder");
    let burnpack = dir.path().join("jepa_reconstruction.bpk");
    save_jepa_reconstruction_burnpack(&decoder, &burnpack).expect("save tiny reconstruction bpk");
    write_burnpack_parts_for_browser(&burnpack, 1024 * 1024, true)
        .expect("write tiny reconstruction bpk parts");
    let manifest = BurnJepaReconstructionPackageManifest {
        record_dtype: Some("f16".to_string()),
        reconstruction_config: config,
        model_base_url: "http://127.0.0.1/reconstruction".to_string(),
        ..BurnJepaReconstructionPackageManifest::default()
    }
    .with_burnpack_paths(&burnpack);
    let manifest_path = dir.path().join("manifest.json");
    write_jepa_reconstruction_package_manifest(&manifest_path, &manifest)
        .expect("write tiny reconstruction manifest");
    (dir, manifest_path)
}

fn bench_viewer_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("bevy_jepa_viewer_pipeline_wgpu");
    let package_device = JepaBevyDevice::default();
    let (_reconstruction_package_dir, reconstruction_model_manifest_path) =
        write_tiny_reconstruction_package(&package_device);

    for image_size in viewer_image_sizes() {
        group.throughput(Throughput::Elements((image_size * image_size) as u64));
        for mask_case in VIEWER_MASK_CASES {
            for lane in VIEWER_BENCH_LANES {
                match lane {
                    ViewerBenchLane::Stage { name, request } => {
                        group.bench_function(
                            format!("{}_{}_{}", mask_case.label, image_size, name),
                            |bench| {
                                bench.iter_batched(
                                    || {
                                        prepare_pipeline(
                                            image_size,
                                            mask_case.source,
                                            mask_case.patch_diff_threshold,
                                            BevyJepaDisplayTransfer::Gpu,
                                        )
                                    },
                                    |(device, mut pipeline)| {
                                        let output = pipeline
                                            .step_with_stage_request(request)
                                            .expect("viewer stage step");
                                        JepaBevyBackend::sync(&device)
                                            .expect("sync viewer backend");
                                        assert!(output.metrics.aligns_with_stage_metrics());
                                        assert_eq!(output.metrics.display_tensor_us, 0);
                                        match name {
                                            "low_res_cache_update" => {
                                                assert_eq!(output.metrics.low_res_pca_us, 0);
                                                assert_eq!(output.metrics.anyup_decode_us, 0);
                                                assert_eq!(output.metrics.high_res_pca_us, 0);
                                            }
                                            "pca_projection" => {
                                                assert_eq!(output.metrics.anyup_decode_us, 0);
                                                assert_eq!(output.metrics.high_res_pca_us, 0);
                                            }
                                            "full_anyup_decode" => {
                                                assert!(
                                                    output
                                                        .metrics
                                                        .stage_metrics
                                                        .has_high_res_work()
                                                );
                                            }
                                            _ => {}
                                        }
                                        black_box(output.metrics);
                                    },
                                    BatchSize::SmallInput,
                                );
                            },
                        );
                    }
                    ViewerBenchLane::Display {
                        name,
                        request,
                        transfer,
                    } => {
                        group.bench_function(
                            format!("{}_{}_{}", mask_case.label, image_size, name),
                            |bench| {
                                bench.iter_batched(
                                    || {
                                        prepare_pipeline(
                                            image_size,
                                            mask_case.source,
                                            mask_case.patch_diff_threshold,
                                            transfer,
                                        )
                                    },
                                    |(device, mut pipeline)| {
                                        let output = pipeline
                                            .step_with_display_request(request)
                                            .expect("display viewer step");
                                        JepaBevyBackend::sync(&device)
                                            .expect("sync viewer backend");
                                        assert!(output.metrics.aligns_with_stage_metrics());
                                        assert_eq!(output.metrics.display_transfer, transfer);
                                        assert!(
                                            output.metrics.viewer_total_us
                                                >= output.metrics.total_us
                                        );
                                        black_box(output.metrics);
                                    },
                                    BatchSize::SmallInput,
                                );
                            },
                        );
                    }
                    ViewerBenchLane::Reconstruction { name, request } => {
                        group.bench_function(
                            format!("{}_{}_{}", mask_case.label, image_size, name),
                            |bench| {
                                bench.iter_batched(
                                    || {
                                        prepare_reconstruction_pipeline(
                                            image_size,
                                            mask_case.source,
                                            mask_case.patch_diff_threshold,
                                            reconstruction_model_manifest_path.clone(),
                                        )
                                    },
                                    |(device, mut pipeline)| {
                                        let output = pipeline
                                            .step_with_reconstruction_request(request)
                                            .expect("reconstruction viewer step");
                                        JepaBevyBackend::sync(&device)
                                            .expect("sync viewer backend");
                                        assert!(output.metrics.aligns_with_stage_metrics());
                                        assert!(output.metrics.reconstruction_decode_us > 0);
                                        assert_eq!(output.metrics.reconstruction_frames, 1);
                                        assert_eq!(output.metrics.anyup_decode_us, 0);
                                        assert!(output.metrics.reconstruction_psnr_db.is_none());
                                        black_box(output.metrics);
                                    },
                                    BatchSize::SmallInput,
                                );
                            },
                        );
                    }
                }
            }
        }
    }

    group.finish();
}

trait ViewerStageMetricsExt {
    fn has_high_res_work(&self) -> bool;
}

impl ViewerStageMetricsExt for burn_jepa::FeatureFrameMetrics {
    fn has_high_res_work(&self) -> bool {
        self.anyup_decode_us > 0 || self.anyup_context_us > 0 || self.pca_project_us > 0
    }
}

criterion_group!(benches, bench_viewer_pipeline);
criterion_main!(benches);

fn viewer_image_sizes() -> Vec<usize> {
    let mut sizes = vec![256, DEFAULT_IMAGE_SIZE];
    if std::env::var("BURN_JEPA_BENCH_1024").ok().as_deref() == Some("1") {
        sizes.push(1024);
    }
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}
