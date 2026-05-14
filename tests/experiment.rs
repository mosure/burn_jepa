use burn_jepa::{
    ExperimentConfig, ExperimentMaskPolicy, ExperimentModelVariant, JepaDatasetKind,
    JepaSampleKind, prepare_experiment_data, run_experiment, write_experiment_plan,
};

type AB = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

#[test]
fn experiment_config_plans_trial_matrix() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut config = ExperimentConfig::default();
    config.output_dir = temp.path().join("experiment");
    config.seeds = vec![7, 11];
    config.densities = vec![0.05, 0.10];
    config.model_variants = vec![ExperimentModelVariant::SingleFrameNoTtt];
    config.mask_policies = vec![
        ExperimentMaskPolicy::FullFrame,
        ExperimentMaskPolicy::RandomSparse,
    ];

    let report = write_experiment_plan(&config).expect("write plan");
    assert_eq!(report.trial_count, 8);
    assert!(report.run_manifest.exists());
    assert!(report.planned_trials.exists());
}

#[test]
fn default_experiment_config_covers_full_synthetic_gate_matrix() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut config = ExperimentConfig::default();
    config.output_dir = temp.path().join("experiment");

    let report = write_experiment_plan(&config).expect("write default plan");
    assert_eq!(report.trial_count, 96);
    assert!(config.base.loss.predictor_loss_weight > 0.0);
    assert!(
        config
            .model_variants
            .contains(&ExperimentModelVariant::Teacher3dReference)
    );
    assert!(
        config
            .model_variants
            .contains(&ExperimentModelVariant::TttSelfHidden)
    );
    assert!(
        config
            .mask_policies
            .contains(&ExperimentMaskPolicy::AutogazeSparse)
    );
    assert!(
        config
            .mask_policies
            .contains(&ExperimentMaskPolicy::PatchDiff)
    );
    assert!(
        config
            .mask_policies
            .contains(&ExperimentMaskPolicy::PrecomputedMasks)
    );
}

#[test]
fn experiment_prepare_data_splits_frame_directories_by_clip() {
    let temp = tempfile::tempdir().expect("tempdir");
    let input = temp.path().join("frames");
    let clip_a = input.join("clip_a");
    let clip_b = input.join("clip_b");
    std::fs::create_dir_all(&clip_a).expect("clip_a");
    std::fs::create_dir_all(&clip_b).expect("clip_b");
    for (clip, color) in [
        (&clip_a, image::Rgb([255, 0, 0])),
        (&clip_b, image::Rgb([0, 0, 255])),
    ] {
        for index in 0..3 {
            image::RgbImage::from_pixel(2, 2, color)
                .save(clip.join(format!("{index:03}.png")))
                .expect("frame");
        }
    }

    let mut config = ExperimentConfig::default();
    config.data.input = Some(input);
    config.data.output_dir = temp.path().join("out");
    config.data.train_manifest = temp.path().join("out/train.jsonl");
    config.data.eval_manifest = temp.path().join("out/eval.jsonl");
    config.data.window_frames = 2;
    config.data.window_stride = 2;
    config.data.eval_ratio = 0.5;

    let report = prepare_experiment_data(&config).expect("prepare data");
    assert_eq!(report.clips, 2);
    assert!(report.train_rows > 0);
    assert!(report.eval_rows > 0);
    assert!(report.train_manifest.exists());
    assert!(report.eval_manifest.exists());
}

#[test]
fn experiment_prepare_data_records_domain_labels() {
    let temp = tempfile::tempdir().expect("tempdir");
    let input = temp.path().join("frames");
    let clip = input.join("nature").join("clip_a");
    std::fs::create_dir_all(&clip).expect("clip");
    for index in 0..3 {
        image::RgbImage::from_pixel(2, 2, image::Rgb([0, 255, 0]))
            .save(clip.join(format!("{index:03}.png")))
            .expect("frame");
    }

    let mut config = ExperimentConfig::default();
    config.data.input = Some(input);
    config.data.output_dir = temp.path().join("out");
    config.data.train_manifest = temp.path().join("out/train.jsonl");
    config.data.eval_manifest = temp.path().join("out/eval.jsonl");
    config.data.window_frames = 2;
    config.data.window_stride = 2;
    config.data.domain_from_parent = true;

    let report = prepare_experiment_data(&config).expect("prepare data");
    assert_eq!(report.domains, vec!["nature".to_string()]);
    let text = std::fs::read_to_string(report.train_manifest).expect("read train manifest");
    assert!(text.contains("\"domain\":\"nature\""));
}

#[test]
fn experiment_run_smoke_writes_summary_analysis_and_csv() {
    let device = Default::default();
    let temp = tempfile::tempdir().expect("tempdir");
    let mut config = ExperimentConfig::default();
    config.output_dir = temp.path().join("run");
    config.base.model.output_dir = temp.path().join("train");
    config.base.model.save_model = false;
    config.base.dataset.kind = JepaDatasetKind::Synthetic;
    config.base.dataset.sample_kind = JepaSampleKind::Video;
    config.base.dataset.synthetic_len = 2;
    config.base.training.max_steps = 1;
    config.base.training.eval_steps = 1;
    config.base.training.batch_size = 1;
    config.model_variants = vec![
        ExperimentModelVariant::SingleFrameNoTtt,
        ExperimentModelVariant::TttTeacherFinal,
    ];
    config.mask_policies = vec![ExperimentMaskPolicy::FullFrame];
    config.densities = vec![0.05];

    let report = run_experiment::<AB>(&config, &device).expect("run experiment");
    assert_eq!(report.trial_count, 2);
    assert_eq!(report.completed_trials, 2);
    assert_eq!(report.failed_trials, 0);
    assert!(report.success_criteria.all_trials_completed);
    assert!(report.success_criteria.mask_loss_enabled);
    assert!(!report.success_criteria.full_model_matrix);
    assert!(!report.success_criteria.full_mask_matrix);
    assert!(report.summary_path.exists());
    assert!(report.analysis_path.exists());
    assert!(report.csv_path.exists());
    assert!(
        report
            .trials
            .iter()
            .any(|trial| trial.eval_loss.is_some() || trial.train_final_loss.is_some())
    );
    assert!(report.trials.iter().any(|trial| trial.timing.train_ms > 0));
}
