#[cfg(feature = "cuda")]
#[test]
fn cuda_training_preflight_reports_unavailable_runtime_without_initializing_cuda() {
    let result =
        burn_jepa::runtime::cuda_runtime_preflight(burn_jepa::runtime::CUDA_TRAIN_FORCE_ENV);
    if cfg!(target_os = "linux") && !std::path::Path::new("/dev/nvidiactl").exists() {
        let reason = result.expect_err("missing CUDA device nodes should preflight as unavailable");
        assert!(
            reason.contains("CUDA runtime cannot open a device without NVIDIA character devices"),
            "unexpected CUDA preflight reason: {reason}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_ttt_training_smoke_runs_when_requested() {
    if std::env::var("BURN_JEPA_RUN_GPU_TRAINING_SMOKE")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("skipping CUDA runtime smoke; set BURN_JEPA_RUN_GPU_TRAINING_SMOKE=1");
        return;
    }
    burn_jepa::runtime::cuda_runtime_preflight(burn_jepa::runtime::CUDA_TRAIN_FORCE_ENV)
        .expect("CUDA runtime preflight");
    type B = burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>;
    let device = Default::default();
    let temp = tempfile::tempdir().expect("tempdir");
    let mut config = burn_jepa::BurnJepaTrainConfig::default();
    config.model.save_model = false;
    config.model.output_dir = temp.path().join("cuda-ttt-train");
    config.training.backend = burn_jepa::JepaTrainBackend::Cuda;
    config.training.max_steps = 2;
    config.training.batch_size = 2;
    config.training.learning_rate = 5.0e-3;
    config.dataset.synthetic_len = 1;
    let report =
        burn_jepa::train_ttt_distillation::<B>(&config, &device).expect("cuda TTT runtime smoke");

    assert_ttt_smoke_report_is_numerically_stable(&report, 2, 4);
}

#[cfg(all(feature = "cuda", feature = "sparse-patchify-cuda"))]
#[test]
fn cuda_ttt_training_sparse_patchify_smoke_runs_when_requested() {
    if std::env::var("BURN_JEPA_RUN_GPU_TRAINING_SMOKE")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!(
            "skipping CUDA sparse patchify training smoke; set BURN_JEPA_RUN_GPU_TRAINING_SMOKE=1"
        );
        return;
    }
    burn_jepa::runtime::cuda_runtime_preflight(burn_jepa::runtime::CUDA_TRAIN_FORCE_ENV)
        .expect("CUDA runtime preflight");
    type B = burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>;
    let device = Default::default();
    let temp = tempfile::tempdir().expect("tempdir");
    let config = sparse_patchify_smoke_config(
        burn_jepa::JepaTrainBackend::Cuda,
        temp.path().join("cuda-ttt-sparse-patchify-train"),
    );
    let report = burn_jepa::train_ttt_distillation::<B>(&config, &device)
        .expect("cuda sparse patchify TTT runtime smoke");

    assert_ttt_smoke_report_is_numerically_stable(&report, 1, 1);
    assert!(report.rollout.frozen_sparse_patchify);
}

#[cfg(feature = "webgpu")]
#[test]
fn webgpu_ttt_training_smoke_runs_when_requested() {
    if std::env::var("BURN_JEPA_RUN_GPU_TRAINING_SMOKE")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("skipping WebGPU runtime smoke; set BURN_JEPA_RUN_GPU_TRAINING_SMOKE=1");
        return;
    }
    type B = burn::backend::Autodiff<burn::backend::WebGpu<f32, i32>>;
    let device = Default::default();
    let temp = tempfile::tempdir().expect("tempdir");
    let mut config = burn_jepa::BurnJepaTrainConfig::default();
    config.model.save_model = false;
    config.model.output_dir = temp.path().join("webgpu-ttt-train");
    config.training.backend = burn_jepa::JepaTrainBackend::WebGpu;
    config.training.max_steps = 2;
    config.training.batch_size = 2;
    config.training.learning_rate = 5.0e-3;
    config.dataset.synthetic_len = 1;
    let report =
        burn_jepa::train_ttt_distillation::<B>(&config, &device).expect("webgpu TTT runtime smoke");

    assert_ttt_smoke_report_is_numerically_stable(&report, 2, 4);
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_ttt_training_smoke_runs_when_requested() {
    if std::env::var("BURN_JEPA_RUN_GPU_TRAINING_SMOKE")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("skipping WGPU runtime smoke; set BURN_JEPA_RUN_GPU_TRAINING_SMOKE=1");
        return;
    }
    type B = burn::backend::Autodiff<burn::backend::Wgpu<f32, i32>>;
    let device = Default::default();
    let temp = tempfile::tempdir().expect("tempdir");
    let mut config = burn_jepa::BurnJepaTrainConfig::default();
    config.model.save_model = false;
    config.model.output_dir = temp.path().join("wgpu-ttt-train");
    config.training.backend = burn_jepa::JepaTrainBackend::Wgpu;
    config.training.max_steps = 2;
    config.training.batch_size = 2;
    config.training.learning_rate = 5.0e-3;
    config.dataset.synthetic_len = 1;
    let report =
        burn_jepa::train_ttt_distillation::<B>(&config, &device).expect("wgpu TTT runtime smoke");

    assert_ttt_smoke_report_is_numerically_stable(&report, 2, 4);
}

#[cfg(all(feature = "wgpu", feature = "sparse-patchify-wgpu"))]
#[test]
fn wgpu_ttt_training_sparse_patchify_smoke_runs_when_requested() {
    if std::env::var("BURN_JEPA_RUN_GPU_TRAINING_SMOKE")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!(
            "skipping WGPU sparse patchify training smoke; set BURN_JEPA_RUN_GPU_TRAINING_SMOKE=1"
        );
        return;
    }
    type B = burn::backend::Autodiff<burn_flex_gmm::wgpu::DefaultWgpuBackend>;
    let device = Default::default();
    let temp = tempfile::tempdir().expect("tempdir");
    let config = sparse_patchify_smoke_config(
        burn_jepa::JepaTrainBackend::Wgpu,
        temp.path().join("wgpu-ttt-sparse-patchify-train"),
    );
    let report = burn_jepa::train_ttt_distillation::<B>(&config, &device)
        .expect("wgpu sparse patchify TTT runtime smoke");

    assert_ttt_smoke_report_is_numerically_stable(&report, 1, 1);
    assert!(report.rollout.frozen_sparse_patchify);
}

#[cfg(any(
    all(feature = "cuda", feature = "sparse-patchify-cuda"),
    all(feature = "wgpu", feature = "sparse-patchify-wgpu")
))]
fn sparse_patchify_smoke_config(
    backend: burn_jepa::JepaTrainBackend,
    output_dir: std::path::PathBuf,
) -> burn_jepa::BurnJepaTrainConfig {
    let mut config = burn_jepa::BurnJepaTrainConfig::default();
    config.model.save_model = false;
    config.model.output_dir = output_dir;
    config.training.backend = backend;
    config.training.max_steps = 1;
    config.training.batch_size = 1;
    config.training.learning_rate = 3.0e-3;
    config.training.sparse_rollout = burn_jepa::TttSparseRolloutMode::TargetMask;
    config.training.sparse_patchify_training =
        burn_jepa::TttSparsePatchifyTrainingMode::FrozenSparsePatchify;
    config.training.loss_trace_interval = 0;
    config.training.mask = Some(burn_jepa::TrainingMaskConfig::PrecomputedMasks {
        context_indices: vec![0, 2, 5, 7],
        target_indices: vec![1, 3],
    });
    config.dataset.synthetic_len = 1;
    config
}

#[cfg(any(feature = "cuda", feature = "webgpu", feature = "wgpu"))]
fn assert_ttt_smoke_report_is_numerically_stable(
    report: &burn_jepa::TttTrainingReport,
    steps: usize,
    samples: usize,
) {
    assert_eq!(report.steps, steps);
    assert_eq!(report.samples, samples);
    assert!(report.initial_loss.is_finite());
    assert!(report.best_loss.is_finite());
    assert!(report.final_loss.is_finite());
    assert!(
        report.best_loss <= report.initial_loss,
        "best loss should not exceed initial loss: initial={} best={}",
        report.initial_loss,
        report.best_loss
    );
    assert!(
        report.final_loss <= report.initial_loss * 1.10,
        "TTT smoke should not diverge: initial={} final={}",
        report.initial_loss,
        report.final_loss
    );
}
