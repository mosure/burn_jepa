use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::{Tensor, TensorData};
use burn_jepa::{
    BurnJepaTrainConfig, JepaDatasetConfig, JepaSample, JepaSampleMetadata, SparseMaskBatch,
    SparseTokenMask, TttBackpropMode, TttEncoderConfig, TttLayerPlacement, TttLayerState,
    TttMemoryUpdateSource, TttRolloutReportMode, TttSparsePatchifyTrainingMode,
    TttSparseRolloutMode, TttSupervisionMode, TttTargetMode, VJepa2_1Model, VJepaTttLayer,
    VJepaTttModel, load_jepa_tensor_batch, synthetic_video, train_dense_jepa,
    train_ttt_distillation,
};

type B = burn::backend::NdArray<f32>;
type AB = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

#[test]
fn ttt_default_layer_placement_is_first_last() {
    let config = TttEncoderConfig::default();
    let model = burn_jepa::VJepaConfig::default();
    assert_eq!(config.layer_placement, TttLayerPlacement::FirstLast);
    assert_eq!(
        config.resolved_layers(&model),
        vec![0, model.encoder.depth.saturating_sub(1)]
    );
}

#[test]
fn ttt_training_config_round_trips_through_public_training_namespace() {
    let default_toml = burn_jepa::training::BurnJepaTrainConfig::default()
        .to_toml_string()
        .expect("serialize default config");
    assert!(
        !default_toml.contains("target ="),
        "legacy ttt.target should stay out of default print-config output"
    );

    let mut config = burn_jepa::training::BurnJepaTrainConfig::default();
    config.ttt.target = TttTargetMode::SelfHidden;
    config.ttt.memory_update = TttMemoryUpdateSource::TeacherForcedDiagnostic;
    config.ttt.supervision = TttSupervisionMode::Hybrid;
    config.ttt.hybrid_final_steps = 2;
    config.ttt.backprop_mode = TttBackpropMode::LayerLocal;
    config.training.max_steps = 3;
    config.training.batch_size = 2;
    config.loss.predictor_loss_weight = 0.25;

    let toml = config.to_toml_string().expect("serialize config");
    assert!(toml.contains("[ttt]"));
    assert!(toml.contains("target = \"self_hidden\""));
    assert!(toml.contains("memory_update = \"teacher_forced_diagnostic\""));
    assert!(toml.contains("supervision = \"hybrid\""));
    assert!(toml.contains("hybrid_final_steps = 2"));
    assert!(toml.contains("backprop_mode = \"layer_local\""));
    assert!(toml.contains("[loss]"));

    let parsed: burn_jepa::training::BurnJepaTrainConfig =
        toml::from_str(&toml).expect("parse config");
    parsed.validate_for_ttt().expect("valid parsed TTT config");
    let _model_config: burn_jepa::TrainModelConfig = parsed.model.clone();
    let _loop_config: burn_jepa::TrainingLoopConfig = parsed.training.clone();
    let _loss_config: burn_jepa::TttDistillationConfig = parsed.loss.clone();

    assert_eq!(parsed.ttt.target, TttTargetMode::SelfHidden);
    assert_eq!(
        parsed.ttt.memory_update,
        TttMemoryUpdateSource::TeacherForcedDiagnostic
    );
    assert_eq!(parsed.ttt.supervision, TttSupervisionMode::Hybrid);
    assert_eq!(parsed.ttt.hybrid_final_steps, 2);
    assert_eq!(parsed.ttt.backprop_mode, TttBackpropMode::LayerLocal);
    assert_eq!(parsed.training.max_steps, 3);
    assert_eq!(parsed.training.batch_size, 2);
    assert_eq!(parsed.loss.predictor_loss_weight, 0.25);
}

#[test]
fn ttt_sparse_rollout_config_round_trips() {
    let mut config = BurnJepaTrainConfig::default();
    config.training.mask = Some(burn_jepa::TrainingMaskConfig::PrecomputedMasks {
        context_indices: vec![0, 2, 5, 7],
        target_indices: vec![1, 3],
    });
    config.training.sparse_rollout = TttSparseRolloutMode::TargetMask;
    config.training.sparse_patchify_training = TttSparsePatchifyTrainingMode::FrozenSparsePatchify;

    let toml = config.to_toml_string().expect("serialize config");
    assert!(toml.contains("sparse_rollout = \"target_mask\""));
    assert!(toml.contains("sparse_patchify_training = \"frozen_sparse_patchify\""));

    let parsed: BurnJepaTrainConfig = toml::from_str(&toml).expect("parse config");
    parsed
        .validate_for_ttt()
        .expect("valid sparse rollout config");
    assert_eq!(
        parsed.training.sparse_rollout,
        TttSparseRolloutMode::TargetMask
    );
    assert_eq!(
        parsed.training.sparse_patchify_training,
        TttSparsePatchifyTrainingMode::FrozenSparsePatchify
    );
}

#[test]
fn ttt_context_sparse_rollout_config_round_trips() {
    let mut config = BurnJepaTrainConfig::default();
    config.training.mask = Some(burn_jepa::TrainingMaskConfig::PrecomputedMasks {
        context_indices: vec![0, 2, 5, 7],
        target_indices: vec![1, 3],
    });
    config.training.sparse_rollout = TttSparseRolloutMode::ContextMask;
    config.training.sparse_patchify_training = TttSparsePatchifyTrainingMode::FrozenSparsePatchify;

    let toml = config.to_toml_string().expect("serialize config");
    assert!(toml.contains("sparse_rollout = \"context_mask\""));

    let parsed: BurnJepaTrainConfig = toml::from_str(&toml).expect("parse config");
    parsed
        .validate_for_ttt()
        .expect("valid context sparse rollout config");
    assert_eq!(
        parsed.training.sparse_rollout,
        TttSparseRolloutMode::ContextMask
    );
}

#[test]
fn ttt_sparse_rollout_config_rejects_incompatible_modes() {
    let mut missing_mask = BurnJepaTrainConfig::default();
    missing_mask.training.sparse_rollout = TttSparseRolloutMode::TargetMask;
    assert!(
        missing_mask.validate_for_ttt().is_err(),
        "target-mask sparse rollout should require an explicit training mask"
    );

    let mut predictor_loss = BurnJepaTrainConfig::default();
    predictor_loss.training.mask = Some(burn_jepa::TrainingMaskConfig::PrecomputedMasks {
        context_indices: vec![0, 2, 5, 7],
        target_indices: vec![1, 3],
    });
    predictor_loss.training.sparse_rollout = TttSparseRolloutMode::TargetMask;
    predictor_loss.loss.predictor_loss_weight = 0.25;
    assert!(
        predictor_loss.validate_for_ttt().is_err(),
        "target-mask sparse rollout should not be accepted with predictor auxiliary loss"
    );

    let mut unfrozen = BurnJepaTrainConfig::default();
    unfrozen.training.mask = Some(burn_jepa::TrainingMaskConfig::PrecomputedMasks {
        context_indices: vec![0, 2, 5, 7],
        target_indices: vec![1, 3],
    });
    unfrozen.training.sparse_rollout = TttSparseRolloutMode::TargetMask;
    unfrozen.training.sparse_patchify_training =
        TttSparsePatchifyTrainingMode::FrozenSparsePatchify;
    unfrozen.ttt.freeze_pretrained = false;
    assert!(
        unfrozen.validate_for_ttt().is_err(),
        "frozen sparse patchify should require frozen pretrained weights"
    );
}

#[test]
fn training_mask_config_round_trips_and_preserves_legacy_keep_ratio() {
    let mut legacy = BurnJepaTrainConfig::default();
    legacy.training.context_keep_ratio = 0.5;
    assert_eq!(
        legacy.training.mask_config(),
        burn_jepa::TrainingMaskConfig::KeepRatio {
            context_keep_ratio: 0.5
        }
    );

    let mut config = BurnJepaTrainConfig::default();
    let mask = burn_jepa::TrainingMaskConfig::AutogazeSparse {
        image_grid: burn_jepa::TrainingImageTokenGrid::new(2, 2),
        context_tokens: 4,
        target_tokens: 2,
        source: burn_jepa::TrainingAutogazeTokenSource::default(),
        frame_tokens: None,
        dilation: 0,
    };
    config.training.mask = Some(mask.clone());

    let toml = config.to_toml_string().expect("serialize config");
    assert!(toml.contains("[training.mask]"));
    assert!(toml.contains("kind = \"autogaze_sparse\""));
    assert!(toml.contains("[training.mask.image_grid]"));
    assert!(!toml.contains("frame_tokens"));

    let parsed: BurnJepaTrainConfig = toml::from_str(&toml).expect("parse config");
    parsed.validate_for_ttt().expect("valid parsed TTT config");
    assert_eq!(parsed.training.mask, Some(mask));
}

#[test]
fn legacy_autogaze_frame_tokens_config_still_parses() {
    let toml = r#"
[training]
max_steps = 1
batch_size = 1

[training.mask]
kind = "autogaze_sparse"
frame_tokens = [[0, 3], [1], [2], [0]]
context_tokens = 4
target_tokens = 2

[training.mask.image_grid]
height = 2
width = 2
"#;
    let parsed: BurnJepaTrainConfig = toml::from_str(toml).expect("parse legacy config");
    let Some(burn_jepa::TrainingMaskConfig::AutogazeSparse {
        frame_tokens: Some(frame_tokens),
        ..
    }) = parsed.training.mask
    else {
        panic!("legacy frame_tokens should be preserved");
    };
    assert_eq!(frame_tokens, vec![vec![0, 3], vec![1], vec![2], vec![0]]);
}

#[test]
fn training_mask_config_resolves_supported_policies() {
    let device = Default::default();
    let model = burn_jepa::VJepaConfig::tiny_for_tests();
    let video = synthetic_video::<B>(0, model.in_channels, 4, 32, 32, &device);
    let policies = [
        burn_jepa::TrainingMaskConfig::KeepRatio {
            context_keep_ratio: 0.5,
        },
        burn_jepa::TrainingMaskConfig::FullFrame { target_tokens: 2 },
        burn_jepa::TrainingMaskConfig::AutogazeSparse {
            image_grid: burn_jepa::TrainingImageTokenGrid::new(2, 2),
            context_tokens: 4,
            target_tokens: 2,
            source: burn_jepa::TrainingAutogazeTokenSource::FrameTokens {
                frame_tokens: vec![vec![0, 3], vec![1], vec![2], vec![0]],
            },
            frame_tokens: None,
            dilation: 0,
        },
        burn_jepa::TrainingMaskConfig::PatchDiff {
            threshold: 0.0,
            context_tokens: 4,
            target_tokens: 2,
            dilation: 0,
        },
        burn_jepa::TrainingMaskConfig::PrecomputedMasks {
            context_indices: vec![0, 2, 5, 7],
            target_indices: vec![1, 3],
        },
    ];

    for policy in policies {
        let mut training = burn_jepa::TrainingLoopConfig::default();
        training.mask = Some(policy);
        let (context, target) = training
            .resolve_masks(&video, &model)
            .expect("resolve training mask policy");
        assert_eq!(context.dense_len(), model.num_patches());
        assert_eq!(target.dense_len(), model.num_patches());
        assert!(!context.is_empty());
        assert!(!target.is_empty());
        for index in target.indices() {
            assert!(!context.indices().contains(index));
        }
    }
}

#[test]
fn manifest_precomputed_masks_resolve_from_batch_metadata() {
    let device = Default::default();
    let model = burn_jepa::VJepaConfig::tiny_for_tests();
    let video = synthetic_video::<B>(0, model.in_channels, 4, 32, 32, &device);
    let mut training = burn_jepa::TrainingLoopConfig::default();
    training.mask = Some(burn_jepa::TrainingMaskConfig::ManifestPrecomputedMasks);
    let metadata = vec![JepaSampleMetadata {
        precomputed_context_indices: Some(vec![0, 2, 5, 7]),
        precomputed_target_indices: Some(vec![1, 3]),
        ..JepaSampleMetadata::default()
    }];

    let (context, target) = training
        .resolve_masks_with_metadata(&video, &model, &metadata)
        .expect("resolve manifest masks");

    assert_eq!(context.indices(), &[0, 2, 5, 7]);
    assert_eq!(target.indices(), &[1, 3]);
}

#[test]
fn manifest_precomputed_masks_reject_mixed_batch_masks() {
    let device = Default::default();
    let model = burn_jepa::VJepaConfig::tiny_for_tests();
    let video = Tensor::cat(
        vec![
            synthetic_video::<B>(0, model.in_channels, 4, 32, 32, &device),
            synthetic_video::<B>(1, model.in_channels, 4, 32, 32, &device),
        ],
        0,
    );
    let mut training = burn_jepa::TrainingLoopConfig::default();
    training.mask = Some(burn_jepa::TrainingMaskConfig::ManifestPrecomputedMasks);
    let metadata = vec![
        JepaSampleMetadata {
            precomputed_context_indices: Some(vec![0, 2, 5, 7]),
            precomputed_target_indices: Some(vec![1, 3]),
            ..JepaSampleMetadata::default()
        },
        JepaSampleMetadata {
            precomputed_context_indices: Some(vec![0, 4, 5, 7]),
            precomputed_target_indices: Some(vec![1, 3]),
            ..JepaSampleMetadata::default()
        },
    ];

    let error = format!(
        "{:#}",
        training
            .resolve_masks_with_metadata(&video, &model, &metadata)
            .expect_err("mixed masks should be rejected")
    );

    assert!(error.contains("batch_size=1"));
}

#[test]
fn sparse_mask_batch_represents_ragged_rows_with_valid_mask() {
    let device = Default::default();
    let mask = SparseMaskBatch::<B>::from_rows(vec![vec![0, 2, 4], vec![1]], 6, &device)
        .expect("ragged mask");

    assert!(mask.is_ragged());
    assert_eq!(mask.batch(), 2);
    assert_eq!(mask.len(), 3);
    assert_eq!(mask.valid_token_count(), 4);
    assert_eq!(mask.rows(), vec![vec![0, 2, 4], vec![1]]);
    assert_eq!(mask.padded_rows(), vec![vec![0, 2, 4], vec![1, 1, 1]]);

    let valid = mask
        .valid_token_mask(&device)
        .expect("ragged valid mask")
        .into_data()
        .to_vec::<f32>()
        .expect("valid mask values");
    assert_eq!(valid, vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
}

#[test]
fn ttt_zero_initialized_adapter_preserves_video_encoder_output() {
    let device = Default::default();
    let layer = VJepaTttLayer::<B>::new(
        4,
        &TttEncoderConfig {
            chunk_tokens: 2,
            ..TttEncoderConfig::default()
        },
        &device,
    );
    let input = Tensor::<B, 3>::from_data(
        TensorData::new(
            vec![
                0.0, 0.1, 0.2, 0.3, //
                0.4, 0.5, 0.6, 0.7, //
                0.8, 0.9, 1.0, 1.1, //
                1.2, 1.3, 1.4, 1.5,
            ],
            [1, 4, 4],
        ),
        &device,
    );
    let mut state = TttLayerState::empty();
    let output = layer
        .forward(input.clone(), None, &mut state)
        .into_data()
        .to_vec::<f32>()
        .expect("output values");
    let input = input.into_data().to_vec::<f32>().expect("input values");
    let max_diff = input
        .iter()
        .zip(output.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1.0e-5,
        "zero-initialized TTT adapter should preserve input, diff={max_diff}"
    );
    assert!(state.fast_weight.is_some());
}

#[test]
fn ttt_model_zero_init_matches_pretrained_video_encoder_and_stays_stable() {
    let device = Default::default();
    let model_config = burn_jepa::VJepaConfig::tiny_for_tests();
    let base = VJepa2_1Model::<B>::new(&model_config, &device);
    let student = VJepaTttModel::from_model(
        base,
        TttEncoderConfig {
            chunk_tokens: 4,
            ..TttEncoderConfig::default()
        },
        &device,
    )
    .expect("TTT wrapped model");
    let video = synthetic_video::<B>(0, model_config.in_channels, 4, 32, 32, &device);
    let expected = student
        .encoder
        .base
        .forward_video(video.clone(), None)
        .tokens;

    let direct = student
        .encode_video(video.clone(), None)
        .expect("TTT video encode")
        .tokens;
    assert_tensor_close(
        "zero-init TTT video encode should match pretrained/base encoder",
        expected.clone(),
        direct,
        1.0e-5,
    );

    let mut state = student.fresh_state();
    let first = student
        .encoder
        .forward_video_with_state(video.clone(), None, Some(expected.clone()), &mut state)
        .expect("stateful TTT video encode")
        .tokens;
    assert!(
        state.layers.iter().all(|layer| layer.fast_weight.is_some()),
        "stateful TTT pass should update fast-weight state"
    );
    assert_tensor_close(
        "zero-init TTT stateful pass should match pretrained/base encoder",
        expected.clone(),
        first,
        1.0e-5,
    );

    let second = student
        .encoder
        .forward_video_with_state(video, None, Some(expected.clone()), &mut state)
        .expect("second stateful TTT video encode")
        .tokens;
    assert_tensor_close(
        "zero-init TTT output should stay stable after fast-weight updates",
        expected,
        second,
        1.0e-5,
    );
}

#[test]
fn ttt_distillation_training_smoke_improves_tiny_loss() {
    let device = Default::default();
    let mut config = BurnJepaTrainConfig::default();
    config.model.save_model = false;
    let temp = tempfile::tempdir().expect("tempdir");
    config.model.output_dir = temp.path().join("ttt-train");
    config.training.max_steps = 3;
    config.training.batch_size = 2;
    config.training.eval_steps = 1;
    config.training.eval_utilization_diagnostics = true;
    config.training.eval_temporal_diagnostics = true;
    config.training.learning_rate = 5.0e-3;
    config.dataset.synthetic_len = 1;
    let report = train_ttt_distillation::<AB>(&config, &device).expect("training smoke");

    assert_eq!(report.steps, 3);
    assert_eq!(report.samples, 6);
    assert_eq!(report.loss_trace.len(), 3);
    assert!(report.initial_loss.is_finite());
    assert!(report.best_loss.is_finite());
    assert!(report.final_loss.is_finite());
    assert!(report.pre_train_eval_loss.is_some());
    assert!(report.pre_train_eval_cosine.is_some());
    assert!(report.eval_loss.is_some());
    assert!(report.eval_cosine.is_some());
    assert_eq!(report.target_supervision.mode, TttTargetMode::TeacherFinal);
    assert_eq!(
        report.target_supervision.memory_update,
        TttMemoryUpdateSource::SelfHidden
    );
    assert_eq!(
        report.target_supervision.supervision,
        TttSupervisionMode::FinalTeacher
    );
    assert!(report.teacher_forced_eval_loss.is_none());
    assert!(report.teacher_forcing_loss_gap.is_none());
    let utilization = report
        .utilization
        .as_ref()
        .expect("TTT eval should report utilization probes");
    assert_eq!(utilization.layers.len(), report.memory.layers.len());
    assert!(utilization.layers.iter().all(|layer| {
        layer.hidden_rms.is_finite()
            && layer.memory_read_rms.is_finite()
            && layer.adapter_delta_rms.is_finite()
            && layer.fast_weight_rms.is_finite()
            && layer.fast_update_rms.is_finite()
            && layer.temporal_conv_param_rms.is_finite()
            && layer.out_proj_param_rms.is_finite()
            && (layer.target_proj_grad_rms.is_some()
                || layer.temporal_conv_grad_rms.is_some()
                || layer.out_proj_grad_rms.is_some())
    }));
    let temporal = report
        .temporal_diagnostics
        .as_ref()
        .expect("TTT eval should report temporal diagnostics");
    assert!(temporal.reset_each_frame_loss.is_some_and(f64::is_finite));
    assert!(temporal.reverse_order_loss.is_some_and(f64::is_finite));
    assert!(temporal.shuffle_order_loss.is_some_and(f64::is_finite));
    assert!(temporal.freeze_fast_update_loss.is_some_and(f64::is_finite));
    assert!(
        report.best_loss < report.initial_loss,
        "tiny training run should improve at least one step: initial={} best={} final={}",
        report.initial_loss,
        report.best_loss,
        report.final_loss
    );
    assert!(
        report.final_loss <= report.initial_loss * 1.01,
        "tiny training run should not diverge: initial={} final={}",
        report.initial_loss,
        report.final_loss
    );
    assert!(report.report_path.exists());
}

#[test]
fn ttt_sparse_rollout_training_smoke_uses_target_mask() {
    let device = Default::default();
    let mut config = BurnJepaTrainConfig::default();
    config.model.save_model = false;
    let temp = tempfile::tempdir().expect("tempdir");
    config.model.output_dir = temp.path().join("ttt-sparse-rollout-train");
    config.training.max_steps = 1;
    config.training.batch_size = 1;
    config.training.eval_steps = 1;
    config.training.eval_utilization_diagnostics = true;
    config.training.sparse_rollout = TttSparseRolloutMode::TargetMask;
    config.training.mask = Some(burn_jepa::TrainingMaskConfig::PrecomputedMasks {
        context_indices: vec![0, 2, 5, 7],
        target_indices: vec![1, 3],
    });
    config.dataset.synthetic_len = 1;
    let report = train_ttt_distillation::<AB>(&config, &device).expect("sparse rollout smoke");

    assert_eq!(report.steps, 1);
    assert!(report.mask.is_some());
    assert_eq!(report.rollout.mode, TttRolloutReportMode::SparseTarget);
    assert_eq!(report.rollout.dense_tokens, 8);
    assert_eq!(report.rollout.student_tokens, 2);
    assert!(!report.rollout.autodiff_sparse_patchify);
    assert!(report.final_loss.is_finite());
    assert!(report.eval_loss.is_some_and(f64::is_finite));
    assert!(report.teacher_forced_eval_loss.is_none());
    assert!(report.teacher_forcing_cosine_gap.is_none());
    assert!(report.eval_full_loss.is_some_and(f64::is_finite));
}

#[test]
fn ttt_teacher_forced_eval_is_explicitly_opt_in() {
    let device = Default::default();
    let mut config = BurnJepaTrainConfig::default();
    config.model.save_model = false;
    let temp = tempfile::tempdir().expect("tempdir");
    config.model.output_dir = temp.path().join("ttt-teacher-forced-diagnostic");
    config.ttt.memory_update = TttMemoryUpdateSource::TeacherForcedDiagnostic;
    config.training.max_steps = 1;
    config.training.batch_size = 1;
    config.training.eval_steps = 1;
    config.dataset.synthetic_len = 1;

    let report = train_ttt_distillation::<AB>(&config, &device).expect("teacher-forced smoke");

    assert_eq!(
        report.target_supervision.memory_update,
        TttMemoryUpdateSource::TeacherForcedDiagnostic
    );
    assert!(report.target_supervision.teacher_forced_eval);
    assert!(report.teacher_forced_eval_loss.is_some_and(f64::is_finite));
    assert!(report.teacher_forcing_loss_gap.is_some_and(f64::is_finite));
}

#[test]
fn ttt_layer_local_supervision_trains_against_same_depth_teacher_features() {
    let device = Default::default();
    let mut config = BurnJepaTrainConfig::default();
    config.model.save_model = false;
    let temp = tempfile::tempdir().expect("tempdir");
    config.model.output_dir = temp.path().join("ttt-layer-local");
    config.ttt.layer_placement = TttLayerPlacement::First;
    config.ttt.supervision = TttSupervisionMode::LayerLocalTeacher;
    config.training.max_steps = 1;
    config.training.batch_size = 1;
    config.training.eval_steps = 1;
    config.dataset.synthetic_len = 1;

    let report = train_ttt_distillation::<AB>(&config, &device).expect("layer-local smoke");

    assert_eq!(
        report.target_supervision.supervision,
        TttSupervisionMode::LayerLocalTeacher
    );
    assert_eq!(
        report.target_supervision.layer_alignment,
        "same_depth_layer_teacher_loss"
    );
    assert!(report.final_loss.is_finite());
    assert!(report.eval_loss.is_some_and(f64::is_finite));
    assert!(report.teacher_forced_eval_loss.is_none());
}

#[test]
fn ttt_hybrid_supervision_runs_layer_local_then_final_teacher_steps() {
    let device = Default::default();
    let mut config = BurnJepaTrainConfig::default();
    config.model.save_model = false;
    let temp = tempfile::tempdir().expect("tempdir");
    config.model.output_dir = temp.path().join("ttt-hybrid");
    config.ttt.layer_placement = TttLayerPlacement::First;
    config.ttt.supervision = TttSupervisionMode::Hybrid;
    config.ttt.hybrid_final_steps = 1;
    config.training.max_steps = 2;
    config.training.batch_size = 1;
    config.training.eval_steps = 1;
    config.dataset.synthetic_len = 1;

    let report = train_ttt_distillation::<AB>(&config, &device).expect("hybrid smoke");

    assert_eq!(
        report.target_supervision.supervision,
        TttSupervisionMode::Hybrid
    );
    assert_eq!(
        report.target_supervision.layer_alignment,
        "layer_local_pretrain_then_final_teacher_finetune"
    );
    assert_eq!(report.target_supervision.hybrid_final_steps, 1);
    assert_eq!(report.loss_trace.len(), 2);
    assert!(report.final_loss.is_finite());
    assert!(report.eval_loss.is_some_and(f64::is_finite));
}

#[test]
fn ttt_manifest_fixed_width_masks_train_with_batch_size_two() {
    let device = Default::default();
    let temp = tempfile::tempdir().expect("tempdir");
    let frame_dir = temp.path().join("frames");
    std::fs::create_dir_all(&frame_dir).expect("frame dir");
    let mut frame_paths = Vec::new();
    for frame in 0..4 {
        let path = frame_dir.join(format!("frame-{frame}.png"));
        let image = image::RgbImage::from_fn(32, 32, |x, y| {
            image::Rgb([
                ((x + y + frame) % 255) as u8,
                ((x * 3 + frame) % 255) as u8,
                ((y * 5 + frame) % 255) as u8,
            ])
        });
        image.save(&path).expect("save frame");
        frame_paths.push(path);
    }
    let manifest = temp.path().join("manifest.jsonl");
    let frame_paths = frame_paths
        .iter()
        .map(|path| path.to_string_lossy().to_string())
        .collect::<Vec<_>>();
    let rows = [
        serde_json::json!({
            "clip_id": "a",
            "frames": frame_paths.clone(),
            "precomputed_context_indices": [0, 1, 4, 5],
            "precomputed_target_indices": [2, 6]
        }),
        serde_json::json!({
            "clip_id": "b",
            "frames": frame_paths.clone(),
            "precomputed_context_indices": [2, 3, 6, 7],
            "precomputed_target_indices": [0, 4]
        }),
    ];
    std::fs::write(
        &manifest,
        rows.iter()
            .map(serde_json::Value::to_string)
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .expect("write manifest");

    let mut config = BurnJepaTrainConfig::default();
    config.model.save_model = false;
    config.model.output_dir = temp.path().join("ttt-fixed-width-batch");
    config.dataset.kind = burn_jepa::JepaDatasetKind::Manifest;
    config.dataset.sample_kind = burn_jepa::JepaSampleKind::Video;
    config.dataset.train_manifest = Some(manifest.clone());
    config.dataset.eval_manifest = Some(manifest);
    config.dataset.synthetic_len = 2;
    config.training.max_steps = 1;
    config.training.batch_size = 2;
    config.training.eval_steps = 1;
    config.training.eval_batch_size = Some(2);
    config.training.batching = burn_jepa::TrainingBatchingMode::FixedWidthMasks;
    config.training.sparse_rollout = TttSparseRolloutMode::ContextMask;
    config.training.mask = Some(burn_jepa::TrainingMaskConfig::ManifestPrecomputedMasks);

    let report = train_ttt_distillation::<AB>(&config, &device)
        .expect("fixed-width manifest TTT training smoke");

    assert_eq!(report.steps, 1);
    assert_eq!(report.samples, 2);
    assert_eq!(report.rollout.mode, TttRolloutReportMode::SparseContext);
    assert_eq!(report.rollout.student_tokens, 4);
    assert!(report.final_loss.is_finite());
    assert!(report.eval_loss.is_some_and(f64::is_finite));
}

#[test]
fn ttt_manifest_ragged_masks_train_with_batch_size_two() {
    let device = Default::default();
    let temp = tempfile::tempdir().expect("tempdir");
    let frame_dir = temp.path().join("frames");
    std::fs::create_dir_all(&frame_dir).expect("frame dir");
    let mut frame_paths = Vec::new();
    for frame in 0..4 {
        let path = frame_dir.join(format!("frame-{frame}.png"));
        let image = image::RgbImage::from_fn(32, 32, |x, y| {
            image::Rgb([
                ((x + y + frame * 7) % 255) as u8,
                ((x * 3 + frame) % 255) as u8,
                ((y * 5 + frame * 11) % 255) as u8,
            ])
        });
        image.save(&path).expect("save frame");
        frame_paths.push(path);
    }
    let manifest = temp.path().join("manifest-ragged.jsonl");
    let frame_paths = frame_paths
        .iter()
        .map(|path| path.to_string_lossy().to_string())
        .collect::<Vec<_>>();
    let rows = [
        serde_json::json!({
            "clip_id": "a",
            "frames": frame_paths.clone(),
            "precomputed_context_indices": [0, 1, 4, 5],
            "precomputed_target_indices": [2, 6]
        }),
        serde_json::json!({
            "clip_id": "b",
            "frames": frame_paths.clone(),
            "precomputed_context_indices": [2, 6, 7],
            "precomputed_target_indices": [0]
        }),
    ];
    std::fs::write(
        &manifest,
        rows.iter()
            .map(serde_json::Value::to_string)
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .expect("write manifest");

    let mut config = BurnJepaTrainConfig::default();
    config.model.save_model = false;
    config.model.output_dir = temp.path().join("ttt-ragged-batch");
    config.dataset.kind = burn_jepa::JepaDatasetKind::Manifest;
    config.dataset.sample_kind = burn_jepa::JepaSampleKind::Video;
    config.dataset.train_manifest = Some(manifest.clone());
    config.dataset.eval_manifest = Some(manifest);
    config.dataset.synthetic_len = 2;
    config.training.max_steps = 1;
    config.training.batch_size = 2;
    config.training.eval_steps = 1;
    config.training.eval_batch_size = Some(2);
    config.training.sparse_rollout = TttSparseRolloutMode::ContextMask;
    config.training.mask = Some(burn_jepa::TrainingMaskConfig::ManifestPrecomputedMasks);

    let report =
        train_ttt_distillation::<AB>(&config, &device).expect("ragged manifest TTT training smoke");

    assert_eq!(report.steps, 1);
    assert_eq!(report.samples, 2);
    assert_eq!(report.rollout.mode, TttRolloutReportMode::SparseContext);
    assert_eq!(report.rollout.student_tokens, 4);
    let mask = report.mask.expect("ragged mask metrics");
    assert_eq!(mask.context_min_tokens, 3);
    assert_eq!(mask.context_max_tokens, 4);
    assert_eq!(mask.target_min_tokens, 1);
    assert_eq!(mask.target_max_tokens, 2);
    assert!((mask.context_mean_tokens - 3.5).abs() < f32::EPSILON);
    assert!((mask.target_mean_tokens - 1.5).abs() < f32::EPSILON);
    assert!((mask.context_density - 0.4375).abs() < f32::EPSILON);
    assert!((mask.target_density - 0.1875).abs() < f32::EPSILON);
    assert!(report.final_loss.is_finite());
    assert!(report.eval_loss.is_some_and(f64::is_finite));
}

#[test]
fn ttt_training_rejects_forced_sparse_patchify_on_unsupported_backend() {
    let device = Default::default();
    let mut config = BurnJepaTrainConfig::default();
    config.model.save_model = false;
    let temp = tempfile::tempdir().expect("tempdir");
    config.model.output_dir = temp.path().join("ttt-unsupported-sparse-patchify");
    config.training.max_steps = 1;
    config.training.batch_size = 1;
    config.training.sparse_rollout = TttSparseRolloutMode::TargetMask;
    config.training.sparse_patchify_training = TttSparsePatchifyTrainingMode::FrozenSparsePatchify;
    config.training.mask = Some(burn_jepa::TrainingMaskConfig::PrecomputedMasks {
        context_indices: vec![0, 2, 5, 7],
        target_indices: vec![1, 3],
    });
    config.dataset.synthetic_len = 1;

    let error =
        train_ttt_distillation::<AB>(&config, &device).expect_err("unsupported sparse patchify");
    assert!(
        error
            .to_string()
            .contains("frozen sparse patchify is not available"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn ttt_training_can_disable_per_step_loss_trace_sync() {
    let device = Default::default();
    let mut config = BurnJepaTrainConfig::default();
    config.model.save_model = false;
    let temp = tempfile::tempdir().expect("tempdir");
    config.model.output_dir = temp.path().join("ttt-no-trace");
    config.training.max_steps = 2;
    config.training.batch_size = 1;
    config.training.loss_trace_interval = 0;
    config.dataset.synthetic_len = 1;
    let report = train_ttt_distillation::<AB>(&config, &device).expect("training no trace");

    assert!(report.loss_trace.is_empty());
    assert!(report.final_loss.is_finite());
    assert_eq!(report.initial_loss, report.final_loss);
    assert_eq!(report.best_loss, report.final_loss);
    assert_eq!(report.rollout.mode, TttRolloutReportMode::Dense);
}

#[test]
fn dense_training_accepts_full_frame_mask_config() {
    let device = Default::default();
    let mut config = BurnJepaTrainConfig::default();
    config.model.save_model = false;
    let temp = tempfile::tempdir().expect("tempdir");
    config.model.output_dir = temp.path().join("dense-train-mask");
    config.training.max_steps = 1;
    config.training.batch_size = 2;
    config.training.mask = Some(burn_jepa::TrainingMaskConfig::FullFrame { target_tokens: 2 });
    config.dataset.synthetic_len = 1;
    let report = train_dense_jepa::<AB>(&config, &device).expect("dense training smoke");

    assert_eq!(report.steps, 1);
    assert_eq!(report.samples, 2);
    assert!(report.final_loss.is_finite());
    assert!(report.report_path.exists());
}

#[test]
fn ttt_training_supports_self_hidden_target_predictor_loss_and_reload() {
    let device = Default::default();
    let mut config = BurnJepaTrainConfig::default();
    let temp = tempfile::tempdir().expect("tempdir");
    config.model.output_dir = temp.path().join("ttt-train-reload");
    config.model.save_model = true;
    config.training.max_steps = 1;
    config.training.batch_size = 2;
    config.training.mask = Some(burn_jepa::TrainingMaskConfig::PrecomputedMasks {
        context_indices: vec![0, 2, 5, 7],
        target_indices: vec![1, 3],
    });
    config.loss.predictor_loss_weight = 0.25;
    config.ttt.target = TttTargetMode::SelfHidden;
    config.training.eval_steps = 1;
    config.training.eval_utilization_diagnostics = true;
    config.training.eval_temporal_diagnostics = true;
    let report = train_ttt_distillation::<AB>(&config, &device).expect("training smoke");
    assert_eq!(report.target_supervision.mode, TttTargetMode::SelfHidden);
    assert!(report.eval_loss.is_some_and(f64::is_finite));
    assert!(report.teacher_forced_eval_loss.is_none());
    assert!(report.teacher_forcing_loss_gap.is_none());
    let model_path = report.model_path.expect("saved TTT model path");
    assert!(
        model_path.exists(),
        "saved model should exist at {model_path:?}"
    );

    let model_config = burn_jepa::VJepaConfig::tiny_for_tests();
    let base = VJepa2_1Model::<AB>::new(&model_config, &device);
    let loaded = VJepaTttModel::from_model(base, config.ttt.clone(), &device)
        .expect("fresh TTT model")
        .load_file(
            model_path,
            &NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
            &device,
        )
        .expect("reload saved TTT model");
    let video = synthetic_video::<AB>(0, model_config.in_channels, 4, 32, 32, &device);
    let mut state = loaded.fresh_state();
    let output = loaded
        .forward_single_frame_rollout(video, None, &mut state)
        .expect("loaded rollout");

    assert_eq!(
        output.tokens.shape().dims::<3>(),
        [1, model_config.num_patches(), 32]
    );
}

#[test]
fn ttt_sparse_single_frame_rollout_returns_only_masked_video_tokens() {
    let device = Default::default();
    let model_config = burn_jepa::VJepaConfig::tiny_for_tests();
    let base = VJepa2_1Model::<B>::new(&model_config, &device);
    let student = VJepaTttModel::from_model(
        base,
        TttEncoderConfig {
            chunk_tokens: 4,
            ..TttEncoderConfig::default()
        },
        &device,
    )
    .expect("TTT wrapped model");
    let video = synthetic_video::<B>(0, model_config.in_channels, 4, 32, 32, &device);
    let teacher_tokens = student
        .encoder
        .base
        .forward_video(video.clone(), None)
        .tokens
        .detach();
    let mask = SparseTokenMask::new(vec![1, 2, 6], model_config.num_patches()).expect("mask");
    let mut state = student.fresh_state();
    let output = student
        .forward_single_frame_rollout_sparse(video, &mask, Some(teacher_tokens), &mut state)
        .expect("sparse rollout");

    assert_eq!(
        output.tokens.shape().dims::<3>(),
        [1, mask.len(), model_config.encoder.embed_dim]
    );
    assert_eq!(output.grid, model_config.token_grid());
    let indices = output
        .token_indices
        .into_data()
        .to_vec::<i64>()
        .expect("indices");
    assert_eq!(indices, vec![1, 2, 6]);
}

#[test]
fn image_manifest_loader_preserves_bcthw_layout() {
    let device = Default::default();
    let temp = tempfile::tempdir().expect("tempdir");
    let frame0 = temp.path().join("frame0.png");
    let frame1 = temp.path().join("frame1.png");
    image::RgbImage::from_pixel(2, 2, image::Rgb([255, 0, 0]))
        .save(&frame0)
        .expect("frame0");
    image::RgbImage::from_pixel(2, 2, image::Rgb([0, 128, 255]))
        .save(&frame1)
        .expect("frame1");

    let dataset = JepaDatasetConfig {
        image_size: 2,
        frames: 2,
        ..JepaDatasetConfig::default()
    };
    let model = burn_jepa::VJepaConfig {
        image_size: 2,
        patch_size: 1,
        tubelet_size: 2,
        ..burn_jepa::VJepaConfig::tiny_for_tests()
    };
    let batch = load_jepa_tensor_batch::<B>(
        &JepaSample::Video {
            frames: vec![frame0, frame1],
            metadata: Default::default(),
        },
        &dataset,
        &model,
        &device,
    )
    .expect("load batch");
    let values = batch
        .student
        .into_data()
        .to_vec::<f32>()
        .expect("tensor values");

    assert_eq!(values.len(), 24);
    assert_eq!(&values[0..4], &[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(&values[4..8], &[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(&values[8..12], &[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(&values[20..24], &[1.0, 1.0, 1.0, 1.0]);
    for value in &values[12..16] {
        assert!((*value - 128.0 / 255.0).abs() < 1.0e-6);
    }
}

fn assert_tensor_close<const D: usize>(
    label: &str,
    expected: Tensor<B, D>,
    actual: Tensor<B, D>,
    tolerance: f32,
) {
    assert_eq!(
        expected.shape(),
        actual.shape(),
        "{label}: tensor shapes differ"
    );
    let shape = expected.shape();
    let expected = tensor_values(expected);
    let actual = tensor_values(actual);
    let max_diff = expected
        .iter()
        .zip(actual.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff <= tolerance,
        "{label}: max_diff={max_diff} tolerance={tolerance} shape={shape:?}"
    );
}

fn tensor_values<const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().expect("tensor values")
}
