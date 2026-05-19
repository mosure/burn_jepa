#![allow(clippy::field_reassign_with_default)]

use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::{Int, Tensor, TensorData};
use burn_jepa::{
    BurnJepaTrainConfig, JepaDatasetConfig, JepaSample, JepaSampleMetadata, SparseMaskBatch,
    SparseTokenMask, TttBackpropMode, TttBestCheckpointSelection, TttEncoderConfig,
    TttEvalModelKind, TttInsertionMode, TttLayerPlacement, TttLayerState, TttMemoryDynamics,
    TttMemoryUpdateSource, TttPretrainedTrainScope, TttRolloutReportMode,
    TttSparsePatchifyTrainingMode, TttSparseRolloutMode, TttState, TttStreamStepKind,
    TttSupervisionMode, TttTargetMode, VJepa2_1Model, VJepaTttLayer, VJepaTttModel,
    apply_token_mask, evaluate_ttt_base_sparse, evaluate_ttt_model_file, load_jepa_tensor_batch,
    synthetic_video, train_dense_jepa, train_ttt_distillation,
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
    config.ttt.insertion = TttInsertionMode::InPlaceMlp;
    config.ttt.memory_update = TttMemoryUpdateSource::TeacherForcedDiagnostic;
    config.ttt.supervision = TttSupervisionMode::Hybrid;
    config.ttt.hybrid_final_steps = 2;
    config.ttt.backprop_mode = TttBackpropMode::LayerLocal;
    config.training.max_steps = 3;
    config.training.batch_size = 2;
    config.training.gradient_clip_norm = 0.25;
    config.training.lr_schedule = burn_jepa::LearningRateScheduleConfig::LinearWarmupCosine {
        warmup_steps: 1,
        min_learning_rate: 1.0e-5,
    };
    config.loss.predictor_loss_weight = 0.25;
    config.loss.latent_regularization.weight = 1.0e-4;
    config.loss.latent_regularization.covariance_weight = 0.25;
    config.loss.latent_regularization.covariance_sketch_dim = 8;

    let toml = config.to_toml_string().expect("serialize config");
    assert!(toml.contains("[ttt]"));
    assert!(toml.contains("insertion = \"in_place_mlp\""));
    assert!(toml.contains("target = \"self_hidden\""));
    assert!(toml.contains("memory_update = \"teacher_forced_diagnostic\""));
    assert!(toml.contains("supervision = \"hybrid\""));
    assert!(toml.contains("hybrid_final_steps = 2"));
    assert!(toml.contains("backprop_mode = \"layer_local\""));
    assert!(toml.contains("save_best_model = true"));
    assert!(toml.contains("best_checkpoint_selection = \"deploy_rollout\""));
    assert!(toml.contains("gradient_clip_norm = 0.25"));
    assert!(toml.contains("[training.lr_schedule]"));
    assert!(toml.contains("kind = \"linear_warmup_cosine\""));
    assert!(toml.contains("[loss]"));
    assert!(toml.contains("[loss.latent_regularization]"));

    let parsed: burn_jepa::training::BurnJepaTrainConfig =
        toml::from_str(&toml).expect("parse config");
    parsed.validate_for_ttt().expect("valid parsed TTT config");
    let _model_config: burn_jepa::TrainModelConfig = parsed.model.clone();
    let _loop_config: burn_jepa::TrainingLoopConfig = parsed.training.clone();
    let _loss_config: burn_jepa::TttDistillationConfig = parsed.loss.clone();

    assert_eq!(parsed.ttt.target, TttTargetMode::SelfHidden);
    assert_eq!(parsed.ttt.insertion, TttInsertionMode::InPlaceMlp);
    assert_eq!(
        parsed.ttt.memory_update,
        TttMemoryUpdateSource::TeacherForcedDiagnostic
    );
    assert_eq!(parsed.ttt.memory_dynamics, TttMemoryDynamics::Ema);
    assert_eq!(parsed.ttt.supervision, TttSupervisionMode::Hybrid);
    assert_eq!(parsed.ttt.hybrid_final_steps, 2);
    assert_eq!(parsed.ttt.backprop_mode, TttBackpropMode::LayerLocal);
    assert_eq!(parsed.training.max_steps, 3);
    assert_eq!(parsed.training.batch_size, 2);
    assert_eq!(parsed.training.gradient_clip_norm, 0.25);
    assert!(parsed.model.save_best_model);
    assert_eq!(
        parsed.model.best_checkpoint_selection,
        TttBestCheckpointSelection::DeployRollout
    );
    assert!(matches!(
        parsed.training.lr_schedule,
        burn_jepa::LearningRateScheduleConfig::LinearWarmupCosine { .. }
    ));
    assert_eq!(parsed.loss.predictor_loss_weight, 0.25);
    assert_eq!(parsed.loss.latent_regularization.weight, 1.0e-4);
    assert_eq!(parsed.loss.latent_regularization.covariance_weight, 0.25);
    assert_eq!(parsed.loss.latent_regularization.covariance_sketch_dim, 8);
}

#[test]
fn ttt_strict_in_place_config_is_explicit_and_rejects_memory_alibi() {
    let mut config = BurnJepaTrainConfig::default();
    config.ttt.insertion = TttInsertionMode::InPlaceMlpStrict;
    config.ttt.memory_dynamics = TttMemoryDynamics::Ema;
    config.ttt.layer_placement = TttLayerPlacement::Explicit;
    config.ttt.layers = vec![0];
    config.training.teacher_window_frames = config.dataset.frames;

    let toml = config.to_toml_string().expect("serialize strict config");
    assert!(toml.contains("insertion = \"in_place_mlp_strict\""));
    assert!(toml.contains("teacher_window_frames"));
    let parsed: BurnJepaTrainConfig = toml::from_str(&toml).expect("parse strict config");
    parsed
        .validate_for_ttt()
        .expect("strict in-place EMA config should validate");

    let mut invalid = parsed.clone();
    invalid.ttt.memory_dynamics = TttMemoryDynamics::MemoryAlibi;
    let error = invalid
        .validate_for_ttt()
        .expect_err("strict in-place should reject Memory-ALiBi");
    assert!(
        error
            .to_string()
            .contains("in_place_mlp_strict implements the paper-style"),
        "{error:?}"
    );
}

#[test]
fn rolling_teacher_window_requires_current_sample_window() {
    let mut config = BurnJepaTrainConfig::default();
    config.ttt.layer_placement = TttLayerPlacement::Explicit;
    config.ttt.layers = vec![0];
    config.dataset.frames = 16;
    config.training.teacher_window_frames = 8;

    let error = config
        .validate_for_ttt()
        .expect_err("teacher window should match dataset frame window");
    assert!(
        error
            .to_string()
            .contains("teacher_window_frames currently formalizes rolling teacher windows"),
        "{error:?}"
    );

    config.training.teacher_window_frames = 16;
    config
        .validate_for_ttt()
        .expect("matching rolling teacher window should validate");
}

#[test]
fn ttt_memory_alibi_config_round_trips_and_resolves_banks() {
    let mut config = burn_jepa::training::BurnJepaTrainConfig::default();
    config.ttt.memory_dynamics = TttMemoryDynamics::MemoryAlibi;
    config.ttt.memory_alibi_half_lives = vec![4, 32, 256];
    config.ttt.memory_alibi_read_weights = vec![2.0, 1.0, 1.0];
    config.ttt.memory_alibi_update_weights = vec![1.0, 0.5, 0.25];
    config.ttt.memory_clip_rms = 8.0;
    config.training.max_steps = 2;

    let toml = config.to_toml_string().expect("serialize config");
    assert!(toml.contains("memory_dynamics = \"memory_alibi\""));
    assert!(toml.contains("memory_alibi_half_lives"));

    let parsed: burn_jepa::training::BurnJepaTrainConfig =
        toml::from_str(&toml).expect("parse config");
    parsed
        .validate_for_ttt()
        .expect("memory-alibi config validates");
    assert_eq!(parsed.ttt.memory_bank_count(), 3);
    assert_eq!(
        parsed.ttt.resolved_memory_alibi_half_lives(),
        vec![4, 32, 256]
    );
    assert_eq!(
        parsed.ttt.resolved_memory_alibi_read_weights(),
        vec![0.5, 0.25, 0.25]
    );
}

#[test]
fn learning_rate_schedule_warmup_cosine_reaches_floor() {
    let mut config = BurnJepaTrainConfig::default();
    config.training.max_steps = 5;
    config.training.learning_rate = 1.0e-3;
    config.training.lr_schedule = burn_jepa::LearningRateScheduleConfig::LinearWarmupCosine {
        warmup_steps: 2,
        min_learning_rate: 1.0e-4,
    };
    config.validate_for_ttt().expect("valid scheduled config");

    let stats = config.training.learning_rate_stats();
    assert!((config.training.learning_rate_for_step(0) - 5.0e-4).abs() < 1.0e-12);
    assert!((config.training.learning_rate_for_step(1) - 1.0e-3).abs() < 1.0e-12);
    assert!((stats.final_learning_rate - 1.0e-4).abs() < 1.0e-12);
    assert_eq!(stats.base_learning_rate, 1.0e-3);

    let clamped = config.training.lr_schedule.clamped_to_max_steps(1);
    clamped
        .validate(1, config.training.learning_rate)
        .expect("clamped schedule stays valid for short bench overrides");
}

#[test]
fn production_ttt_configs_are_encoder_only_and_scheduled() {
    let stage1: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-stage1-adapter-cuda.toml"
    ))
    .expect("parse stage1 production config");
    let stream: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-stage1-stream-tbptt-cuda.toml"
    ))
    .expect("parse stream production config");
    let stream_eval: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-stream-eval-cuda.toml"
    ))
    .expect("parse stream eval production config");
    let stable_stream: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-stage1-stream-tbptt-stable-cuda.toml"
    ))
    .expect("parse stable stream production config");
    let verified_stream: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-stage1-stream-tbptt-verified-cuda.toml"
    ))
    .expect("parse verified stream production config");
    let sigreg_stream: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-stage1-stream-tbptt-sigreg-cuda.toml"
    ))
    .expect("parse SIGReg stream production config");
    let carry_forever_alibi: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-stage1-stream-tbptt-carry-forever-alibi-cuda.toml"
    ))
    .expect("parse carry-forever Memory-ALiBi production config");
    let verified_lowlr_stream: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-stage1-stream-tbptt-verified-lowlr-cuda.toml"
    ))
    .expect("parse verified low-lr stream production config");
    let stable_eval: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-stream-eval-stable-cuda.toml"
    ))
    .expect("parse stable stream eval config");
    let dense_eval: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-dense-eval-stable-cuda.toml"
    ))
    .expect("parse stable dense eval config");
    let long_eval: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-long-rollout-eval-cuda.toml"
    ))
    .expect("parse long rollout eval config");
    let long_dense_eval: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-long-rollout-dense-eval-cuda.toml"
    ))
    .expect("parse dense long rollout eval config");
    let verylong_eval: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-long-rollout-verylong-cuda.toml"
    ))
    .expect("parse very long rollout eval config");
    let base_sparse_verylong_eval: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-base-sparse-long-rollout-verylong-cuda.toml"
    ))
    .expect("parse base sparse very long rollout eval config");
    let reset_window_eval: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-long-rollout-reset-window-cuda.toml"
    ))
    .expect("parse reset-window long rollout eval config");
    let sigreg_long_eval: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-long-rollout-sigreg-cuda.toml"
    ))
    .expect("parse SIGReg long rollout eval config");
    let carry_forever_cactus_eval: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-long-rollout-carry-forever-alibi-cactus-64x-cuda.toml"
    ))
    .expect("parse carry-forever cactus eval config");
    let carry_forever_adversarial_eval: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-long-rollout-carry-forever-alibi-adversarial-8x-cuda.toml"
    ))
    .expect("parse carry-forever adversarial eval config");
    let stage2: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-stage2-unfrozen-low-lr-cuda.toml"
    ))
    .expect("parse stage2 production config");
    let stage2_norms: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-stage2-norms-low-lr-cuda.toml"
    ))
    .expect("parse stage2 norm-only production config");
    let stage2_last2: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-stage2-last2-low-lr-cuda.toml"
    ))
    .expect("parse stage2 last-2-block production config");
    let image2video_norms: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-image2video-stage2-norms-low-lr-cuda.toml"
    ))
    .expect("parse image-to-video norm-only baseline production config");
    let image2video_last2: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-image2video-stage2-last2-low-lr-cuda.toml"
    ))
    .expect("parse image-to-video last-2-block baseline production config");
    let strict_inplace: BurnJepaTrainConfig = toml::from_str(include_str!(
        "../configs/production/vjepa21-ttt-stage1-stream-tbptt-inplace-mlp-strict-thirds-cuda.toml"
    ))
    .expect("parse strict in-place production ablation config");

    assert_eq!(stage1.ttt.layers, vec![3, 7, 11]);
    assert!(stage1.dataset.image_size >= 256);
    assert!(stage1.ttt.predictor_layers.is_empty());
    assert!(stage1.ttt.freeze_pretrained);
    assert_eq!(
        stage1.training.sparse_patchify_training,
        TttSparsePatchifyTrainingMode::FrozenSparsePatchify
    );
    assert!(matches!(
        stage1.training.lr_schedule,
        burn_jepa::LearningRateScheduleConfig::LinearWarmupCosine { .. }
    ));

    assert_eq!(stream.ttt.layers, vec![3, 7, 11]);
    assert!(stream.ttt.predictor_layers.is_empty());
    assert!(stream.ttt.freeze_pretrained);
    assert!(stream.training.stream.enabled);
    assert!(stream.training.stream.curriculum.enabled);
    assert_eq!(stream.training.stream.reset_interval_for_step(0), 1);
    assert_eq!(
        stream
            .training
            .stream
            .reset_interval_for_step(stream.training.max_steps - 1),
        4
    );
    assert_eq!(
        stream.ttt.supervision,
        burn_jepa::TttSupervisionMode::FinalTeacher
    );
    assert_eq!(stream.training.batch_size, 4);
    assert!(stream.training.prefetch_batches);
    assert!(!stream.training.cache_teacher_tokens);
    assert_eq!(stream.training.teacher_cache_max_entries, 0);
    assert!(stream.training.dense_samples.enabled);
    assert_eq!(stream.training.dense_samples.warmup_steps, 128);
    assert_eq!(stream.training.dense_samples.interval_steps, 16);
    assert_eq!(stream.training.stream.state_regularization_width, 64);
    assert!(stream.dataset.image_size >= 256);
    assert_eq!(
        stream.training.batching,
        burn_jepa::TrainingBatchingMode::PackedStreams
    );
    assert!(stable_stream.training.dense_samples.enabled);
    assert_eq!(stable_stream.training.dense_samples.warmup_steps, 128);
    assert_eq!(stable_stream.training.dense_samples.interval_steps, 16);
    assert!(
        stable_stream
            .model
            .output_dir
            .ends_with("stage1-stream-tbptt-stable")
    );
    assert!(verified_stream.training.dense_samples.enabled);
    assert_eq!(verified_stream.training.dense_samples.warmup_steps, 48);
    assert_eq!(verified_stream.training.dense_samples.interval_steps, 16);
    assert_eq!(verified_stream.training.max_steps, 160);
    assert_eq!(
        verified_stream.training.stream.reset_interval_for_step(0),
        1
    );
    assert_eq!(
        verified_stream
            .training
            .stream
            .reset_interval_for_step(verified_stream.training.max_steps - 1),
        4
    );
    assert!(
        verified_stream
            .model
            .output_dir
            .ends_with("stage1-stream-tbptt-verified")
    );
    assert!(sigreg_stream.loss.latent_regularization.active());
    assert_eq!(sigreg_stream.loss.latent_regularization.weight, 1.0e-5);
    assert_eq!(
        sigreg_stream.training.max_steps,
        verified_stream.training.max_steps
    );
    assert_eq!(
        sigreg_stream.training.sparse_rollout,
        verified_stream.training.sparse_rollout
    );
    assert!(
        sigreg_stream
            .model
            .output_dir
            .ends_with("stage1-stream-tbptt-sigreg")
    );
    assert_eq!(
        carry_forever_alibi.ttt.memory_dynamics,
        TttMemoryDynamics::MemoryAlibi
    );
    assert!(carry_forever_alibi.model.save_best_model);
    assert_eq!(
        carry_forever_alibi.model.best_checkpoint_selection,
        TttBestCheckpointSelection::DeployRollout
    );
    assert_eq!(carry_forever_alibi.training.gradient_clip_norm, 0.5);
    assert_eq!(carry_forever_alibi.ttt.memory_bank_count(), 3);
    assert!(carry_forever_alibi.training.stream.enabled);
    assert_eq!(carry_forever_alibi.training.stream.reset_interval_steps, 0);
    assert!(!carry_forever_alibi.training.stream.reset_on_scene_change);
    assert!(
        !carry_forever_alibi
            .training
            .stream
            .reset_on_non_monotonic_start
    );
    assert_eq!(carry_forever_alibi.training.stream.state_decay, 1.0);
    assert!(
        carry_forever_alibi
            .model
            .output_dir
            .ends_with("stage1-stream-tbptt-carry-forever-alibi")
    );
    assert!(verified_lowlr_stream.model.ttt_checkpoint_path.is_some());
    assert!(verified_lowlr_stream.training.learning_rate < verified_stream.training.learning_rate);
    assert_eq!(verified_lowlr_stream.training.max_steps, 96);
    assert!(
        verified_lowlr_stream
            .model
            .output_dir
            .ends_with("stage1-stream-tbptt-verified-lowlr")
    );

    assert!(stream_eval.training.stream.enabled);
    assert!(stream_eval.dataset.image_size >= 256);
    assert_eq!(stream_eval.training.effective_eval_batch_size(), 4);
    assert_eq!(
        stream_eval.training.batching,
        burn_jepa::TrainingBatchingMode::PackedStreams
    );
    assert_eq!(stream_eval.training.stream.reset_interval_steps, 4);
    assert!(!stream_eval.training.stream.curriculum.enabled);
    assert!(stable_eval.training.eval_temporal_diagnostics);
    assert!(stable_eval.training.eval_utilization_diagnostics);
    assert_eq!(stable_eval.training.effective_eval_batch_size(), 4);
    assert_eq!(
        dense_eval.training.sparse_rollout,
        burn_jepa::TttSparseRolloutMode::Dense
    );
    assert!(dense_eval.training.eval_full_grid);
    assert_eq!(
        long_eval.training.batching,
        burn_jepa::TrainingBatchingMode::Sequential
    );
    assert_eq!(long_eval.training.effective_eval_batch_size(), 1);
    assert_eq!(long_eval.training.stream.reset_interval_steps, 0);
    assert!(long_eval.training.eval_temporal_diagnostics);
    assert_eq!(
        long_dense_eval.training.sparse_rollout,
        burn_jepa::TttSparseRolloutMode::Dense
    );
    assert_eq!(
        long_dense_eval.training.batching,
        burn_jepa::TrainingBatchingMode::Sequential
    );
    assert!(long_dense_eval.training.eval_full_grid);
    assert_eq!(
        verylong_eval.training.batching,
        burn_jepa::TrainingBatchingMode::Sequential
    );
    assert_eq!(verylong_eval.training.effective_eval_batch_size(), 1);
    assert_eq!(verylong_eval.training.eval_steps, 164);
    assert_eq!(verylong_eval.training.stream.reset_interval_steps, 0);
    assert!(!verylong_eval.training.eval_temporal_diagnostics);
    assert_eq!(
        base_sparse_verylong_eval.training.sparse_rollout,
        verylong_eval.training.sparse_rollout
    );
    assert_eq!(
        base_sparse_verylong_eval.training.sparse_patchify_training,
        verylong_eval.training.sparse_patchify_training
    );
    assert_eq!(
        base_sparse_verylong_eval
            .training
            .effective_eval_batch_size(),
        verylong_eval.training.effective_eval_batch_size()
    );
    assert!(
        base_sparse_verylong_eval
            .model
            .output_dir
            .ends_with("long-rollout-base-sparse")
    );
    assert_eq!(reset_window_eval.training.stream.reset_interval_steps, 1);
    assert_eq!(
        reset_window_eval.training.eval_steps,
        verylong_eval.training.eval_steps
    );
    assert_eq!(
        reset_window_eval.training.sparse_rollout,
        verylong_eval.training.sparse_rollout
    );
    assert!(sigreg_long_eval.loss.latent_regularization.active());
    assert_eq!(
        sigreg_long_eval.training.batching,
        verylong_eval.training.batching
    );
    assert_eq!(
        sigreg_long_eval.training.effective_eval_batch_size(),
        verylong_eval.training.effective_eval_batch_size()
    );
    assert_eq!(
        sigreg_long_eval.training.stream.reset_interval_steps,
        verylong_eval.training.stream.reset_interval_steps
    );
    assert_eq!(
        carry_forever_cactus_eval.ttt.memory_dynamics,
        TttMemoryDynamics::MemoryAlibi
    );
    assert_eq!(
        carry_forever_cactus_eval
            .training
            .stream
            .reset_interval_steps,
        0
    );
    assert!(
        !carry_forever_cactus_eval
            .training
            .stream
            .reset_on_scene_change
    );
    assert_eq!(carry_forever_cactus_eval.training.eval_steps, 1088);
    assert!(
        carry_forever_cactus_eval
            .training
            .eval_feature_stability_diagnostics
    );
    assert_eq!(
        carry_forever_adversarial_eval.dataset.repeat_mode,
        burn_jepa::JepaDatasetRepeatMode::AdversarialStitchedStream
    );
    assert_eq!(
        carry_forever_adversarial_eval
            .training
            .stream
            .reset_interval_steps,
        0
    );
    assert!(
        carry_forever_adversarial_eval
            .training
            .eval_feature_stability_diagnostics
    );

    assert_eq!(stage2.ttt.layers, vec![3, 7, 11]);
    assert!(stage2.dataset.image_size >= 256);
    assert!(stage2.ttt.predictor_layers.is_empty());
    assert!(!stage2.ttt.freeze_pretrained);
    assert_eq!(
        stage2.training.sparse_patchify_training,
        TttSparsePatchifyTrainingMode::DensePatchEmbed
    );
    assert!(stage2.training.learning_rate < stage1.training.learning_rate);
    assert!(matches!(
        stage2.training.lr_schedule,
        burn_jepa::LearningRateScheduleConfig::LinearWarmupCosine { .. }
    ));
    assert_eq!(
        stage2.ttt.pretrained_train_scope,
        TttPretrainedTrainScope::All
    );
    assert_eq!(
        stage2_norms.ttt.pretrained_train_scope,
        TttPretrainedTrainScope::Norms
    );
    assert_eq!(
        stage2_norms.training.sparse_patchify_training,
        TttSparsePatchifyTrainingMode::FrozenSparsePatchify
    );
    assert_eq!(
        stage2_last2.ttt.pretrained_train_scope,
        TttPretrainedTrainScope::LastNBlocks
    );
    assert_eq!(stage2_last2.ttt.pretrained_train_last_n_blocks, 2);
    assert_eq!(
        stage2_last2.training.sparse_patchify_training,
        TttSparsePatchifyTrainingMode::FrozenSparsePatchify
    );
    assert!(image2video_norms.ttt.layers.is_empty());
    assert!(image2video_norms.ttt.predictor_layers.is_empty());
    assert!(image2video_norms.model.ttt_checkpoint_path.is_none());
    assert_eq!(
        image2video_norms.ttt.pretrained_train_scope,
        TttPretrainedTrainScope::Norms
    );
    assert_eq!(
        image2video_norms.training.sparse_patchify_training,
        TttSparsePatchifyTrainingMode::FrozenSparsePatchify
    );
    assert!(image2video_last2.ttt.layers.is_empty());
    assert!(image2video_last2.ttt.predictor_layers.is_empty());
    assert!(image2video_last2.model.ttt_checkpoint_path.is_none());
    assert_eq!(
        image2video_last2.ttt.pretrained_train_scope,
        TttPretrainedTrainScope::LastNBlocks
    );
    assert_eq!(image2video_last2.ttt.pretrained_train_last_n_blocks, 2);
    assert_eq!(
        strict_inplace.ttt.insertion,
        TttInsertionMode::InPlaceMlpStrict
    );
    assert_eq!(strict_inplace.ttt.memory_dynamics, TttMemoryDynamics::Ema);
    assert_eq!(strict_inplace.training.teacher_window_frames, 16);
    assert_eq!(strict_inplace.dataset.frames, 16);
    assert!(strict_inplace.training.stream.enabled);
    assert_eq!(
        strict_inplace.training.sparse_patchify_training,
        TttSparsePatchifyTrainingMode::FrozenSparsePatchify
    );
}

#[test]
fn ttt_dispatch_backend_config_round_trips() {
    let default_toml = BurnJepaTrainConfig::default()
        .to_toml_string()
        .expect("serialize default config");
    assert!(
        !default_toml.contains("dispatch_backend"),
        "dispatch backend selector should stay out of default configs unless dispatch is selected"
    );

    let mut config = BurnJepaTrainConfig::default();
    config.training.backend = burn_jepa::JepaTrainBackend::Dispatch;
    config.training.dispatch_backend = burn_jepa::JepaDispatchBackend::Flex;

    let toml = config.to_toml_string().expect("serialize dispatch config");
    assert!(toml.contains("backend = \"dispatch\""));
    assert!(toml.contains("dispatch_backend = \"flex\""));

    let parsed: BurnJepaTrainConfig = toml::from_str(&toml).expect("parse dispatch config");
    parsed.validate_for_ttt().expect("valid dispatch config");
    assert_eq!(
        parsed.training.backend,
        burn_jepa::JepaTrainBackend::Dispatch
    );
    assert_eq!(
        parsed.training.dispatch_backend,
        burn_jepa::JepaDispatchBackend::Flex
    );
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
fn ttt_stream_training_config_round_trips_and_validates() {
    let mut config = BurnJepaTrainConfig::default();
    config.training.batch_size = 1;
    config.training.stream.enabled = true;
    config.training.stream.state_decay = 0.95;
    config.training.stream.state_l2_weight = 1.0e-6;
    config.training.stream.update_l2_weight = 2.0e-6;
    config.training.stream.state_regularization_width = 64;
    config.training.stream.reset_interval_steps = 4;
    config.training.stream.curriculum.enabled = true;
    config
        .training
        .stream
        .curriculum
        .initial_reset_interval_steps = 1;
    config.training.stream.curriculum.final_reset_interval_steps = 4;
    config.training.stream.curriculum.warmup_steps = 8;

    let toml = config.to_toml_string().expect("serialize config");
    assert!(toml.contains("[training.stream]"));
    assert!(toml.contains("[training.stream.curriculum]"));
    assert!(toml.contains("enabled = true"));
    assert!(toml.contains("state_decay = 0.95"));
    assert!(toml.contains("state_l2_weight = "));
    assert!(toml.contains("update_l2_weight = "));
    assert!(toml.contains("state_regularization_width = 64"));
    assert!(toml.contains("reset_interval_steps = 4"));

    let parsed: BurnJepaTrainConfig = toml::from_str(&toml).expect("parse stream config");
    parsed.validate_for_ttt().expect("valid stream config");
    assert!(parsed.training.stream.enabled);
    assert!(parsed.training.stream.detach_between_steps);
    assert_eq!(parsed.training.stream.reset_interval_steps, 4);
    assert!(parsed.training.stream.curriculum.enabled);
    assert_eq!(
        parsed
            .training
            .stream
            .curriculum
            .initial_reset_interval_steps,
        1
    );
    assert_eq!(
        parsed.training.stream.curriculum.final_reset_interval_steps,
        4
    );
    assert_eq!(parsed.training.stream.curriculum.warmup_steps, 8);
    assert_eq!(parsed.training.stream.reset_interval_for_step(0), 1);
    assert_eq!(parsed.training.stream.reset_interval_for_step(7), 4);
    assert_eq!(parsed.training.stream.state_l2_weight, 1.0e-6);
    assert_eq!(parsed.training.stream.update_l2_weight, 2.0e-6);
    assert_eq!(parsed.training.stream.state_regularization_width, 64);

    let mut packed = parsed;
    packed.training.batch_size = 2;
    packed.training.eval_steps = 1;
    packed.training.eval_batch_size = Some(2);
    packed.training.batching = burn_jepa::TrainingBatchingMode::PackedStreams;
    packed
        .validate_for_ttt()
        .expect("stream mode should accept packed multi-stream batches");

    let toml = packed
        .to_toml_string()
        .expect("serialize packed stream config");
    assert!(toml.contains("batching = \"packed_streams\""));
    let parsed: BurnJepaTrainConfig = toml::from_str(&toml).expect("parse packed stream config");
    assert_eq!(
        parsed.training.batching,
        burn_jepa::TrainingBatchingMode::PackedStreams
    );
}

#[test]
fn ttt_predictor_layers_require_predictor_loss() {
    let mut config = BurnJepaTrainConfig::default();
    config.ttt.layer_placement = TttLayerPlacement::Explicit;
    config.ttt.layers.clear();
    config.ttt.predictor_layers = vec![0];
    config.loss.predictor_loss_weight = 0.0;
    assert!(
        config.validate_for_ttt().is_err(),
        "predictor-only TTT should require predictor auxiliary loss"
    );

    config.loss.feature_loss_weight = 0.0;
    config.loss.predictor_loss_weight = 0.25;
    config
        .validate_for_ttt()
        .expect("predictor-only TTT should validate with predictor loss");
}

#[test]
fn no_ttt_image_to_video_distillation_requires_trainable_feature_path() {
    let mut frozen = BurnJepaTrainConfig::default();
    frozen.ttt.layer_placement = TttLayerPlacement::Explicit;
    frozen.ttt.layers.clear();
    frozen.ttt.predictor_layers.clear();
    assert!(
        frozen.validate_for_ttt().is_err(),
        "no-adapter TTT config should not silently train nothing"
    );

    let mut no_feature_loss = frozen.clone();
    no_feature_loss.ttt.pretrained_train_scope = TttPretrainedTrainScope::Norms;
    no_feature_loss.loss.feature_loss_weight = 0.0;
    no_feature_loss.loss.predictor_loss_weight = 0.25;
    assert!(
        no_feature_loss.validate_for_ttt().is_err(),
        "no-adapter image-to-video baseline should use the video-teacher feature loss"
    );

    let mut norms = frozen;
    norms.ttt.pretrained_train_scope = TttPretrainedTrainScope::Norms;
    norms
        .validate_for_ttt()
        .expect("trainable norm-only image-to-video baseline should validate");
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
fn ttt_pretrained_train_scope_validation_is_explicit() {
    let mut all_with_freeze = BurnJepaTrainConfig::default();
    all_with_freeze.ttt.freeze_pretrained = true;
    all_with_freeze.ttt.pretrained_train_scope = TttPretrainedTrainScope::All;
    assert!(
        all_with_freeze.validate_for_ttt().is_err(),
        "full pretrained finetuning should opt out of freeze_pretrained"
    );

    let mut last_n_missing_count = BurnJepaTrainConfig::default();
    last_n_missing_count.ttt.pretrained_train_scope = TttPretrainedTrainScope::LastNBlocks;
    last_n_missing_count.ttt.pretrained_train_last_n_blocks = 0;
    assert!(
        last_n_missing_count.validate_for_ttt().is_err(),
        "last_n_blocks should require an explicit block count"
    );

    let mut sparse_compatible_norms = BurnJepaTrainConfig::default();
    sparse_compatible_norms.training.mask = Some(burn_jepa::TrainingMaskConfig::PrecomputedMasks {
        context_indices: vec![0, 2, 5, 7],
        target_indices: vec![1, 3],
    });
    sparse_compatible_norms.training.sparse_rollout = TttSparseRolloutMode::TargetMask;
    sparse_compatible_norms.training.sparse_patchify_training =
        TttSparsePatchifyTrainingMode::FrozenSparsePatchify;
    sparse_compatible_norms.ttt.pretrained_train_scope = TttPretrainedTrainScope::Norms;
    sparse_compatible_norms
        .validate_for_ttt()
        .expect("norm-only tuning keeps the patchify boundary frozen");
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
        burn_jepa::TrainingMaskConfig::TemporalUniformSparse {
            context_tokens: 4,
            target_tokens: 2,
        },
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
fn temporal_uniform_sparse_mask_balances_context_across_tubelets() {
    let device = Default::default();
    let model = burn_jepa::VJepaConfig::tiny_for_tests();
    let video = synthetic_video::<B>(0, model.in_channels, 4, 32, 32, &device);
    let mut training = burn_jepa::TrainingLoopConfig::default();
    training.mask = Some(burn_jepa::TrainingMaskConfig::TemporalUniformSparse {
        context_tokens: 4,
        target_tokens: 2,
    });

    let (context, target) = training
        .resolve_masks(&video, &model)
        .expect("resolve temporal uniform sparse mask");
    let frame_tokens = 4;
    let counts = |mask: &SparseTokenMask| {
        let mut counts = vec![0usize; 2];
        for &index in mask.indices() {
            counts[index / frame_tokens] += 1;
        }
        counts
    };

    assert_eq!(counts(&context), vec![2, 2]);
    assert_eq!(counts(&target), vec![1, 1]);
    for index in target.indices() {
        assert!(!context.indices().contains(index));
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
fn ttt_memory_alibi_adapter_preserves_input_and_updates_banked_state() {
    let device = Default::default();
    let layer = VJepaTttLayer::<B>::new(
        4,
        &TttEncoderConfig {
            chunk_tokens: 2,
            memory_dynamics: TttMemoryDynamics::MemoryAlibi,
            memory_alibi_half_lives: vec![2, 8, 32],
            memory_alibi_read_weights: vec![0.5, 0.3, 0.2],
            memory_alibi_update_weights: vec![1.0, 1.0, 1.0],
            memory_clip_rms: 16.0,
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
        "zero-initialized Memory-ALiBi TTT adapter should preserve input, diff={max_diff}"
    );
    assert!(state.fast_weight.is_none());
    assert_eq!(
        state
            .fast_weight_banks
            .as_ref()
            .expect("banked fast weights")
            .shape()
            .dims::<4>(),
        [1, 3, 4, 4]
    );
}

#[test]
fn ttt_model_zero_init_matches_pretrained_video_encoder_and_stays_stable() {
    let device = Default::default();
    let model_config = burn_jepa::VJepaConfig::tiny_for_tests();
    let base = VJepa2_1Model::<B>::new(&model_config, &device);
    let student = VJepaTttModel::from_model(
        base,
        TttEncoderConfig {
            chunk_tokens: 8,
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
fn ttt_in_place_mlp_zero_init_reuses_pretrained_mlp_and_matches_base_encoder() {
    let device = Default::default();
    let model_config = burn_jepa::VJepaConfig::tiny_for_tests();
    let base = VJepa2_1Model::<B>::new(&model_config, &device);
    let student = VJepaTttModel::from_model(
        base,
        TttEncoderConfig {
            insertion: TttInsertionMode::InPlaceMlp,
            layer_placement: TttLayerPlacement::Explicit,
            layers: vec![0, 1],
            chunk_tokens: 8,
            ..TttEncoderConfig::default()
        },
        &device,
    )
    .expect("in-place TTT wrapped model");
    assert_eq!(
        student.encoder.insertion_mode(),
        TttInsertionMode::InPlaceMlp
    );
    assert_eq!(student.encoder.ttt_layer_indices(), &[0, 1]);
    assert!(student.encoder.ttt_layers.is_empty());
    assert_eq!(
        student
            .encoder
            .inplace_ttt_layers
            .as_ref()
            .expect("in-place layers")
            .len(),
        2
    );

    let video = synthetic_video::<B>(1, model_config.in_channels, 4, 32, 32, &device);
    let expected = student
        .encoder
        .base
        .forward_video(video.clone(), None)
        .tokens;
    let mut state = student.fresh_state();
    let actual = student
        .encoder
        .forward_video_with_state(video.clone(), None, Some(expected.clone()), &mut state)
        .expect("stateful in-place TTT encode")
        .tokens;

    assert_tensor_close(
        "zero-init in-place MLP TTT should preserve pretrained/base encoder",
        expected.clone(),
        actual,
        1.0e-5,
    );
    assert!(
        state.layers.iter().all(|layer| layer.fast_weight.is_some()),
        "in-place MLP TTT should store per-layer down-proj delta fast weights"
    );
    assert_eq!(
        state.layers[0]
            .fast_weight
            .as_ref()
            .expect("in-place fast weight delta")
            .shape()
            .dims::<3>(),
        [
            1,
            ((model_config.encoder.embed_dim as f32) * model_config.encoder.mlp_ratio).round()
                as usize,
            model_config.encoder.embed_dim,
        ]
    );

    let second = student
        .encoder
        .forward_video_with_state(video, None, Some(expected.clone()), &mut state)
        .expect("second in-place TTT encode")
        .tokens;
    assert_tensor_close(
        "zero-init in-place MLP TTT should stay stable after cached updates",
        expected,
        second,
        1.0e-5,
    );
}

#[test]
fn ttt_strict_in_place_mlp_first_pass_reuses_pretrained_mlp() {
    let device = Default::default();
    let model_config = burn_jepa::VJepaConfig::tiny_for_tests();
    let base = VJepa2_1Model::<B>::new(&model_config, &device);
    let student = VJepaTttModel::from_model(
        base,
        TttEncoderConfig {
            insertion: TttInsertionMode::InPlaceMlpStrict,
            layer_placement: TttLayerPlacement::Explicit,
            layers: vec![0, 1],
            chunk_tokens: 8,
            ttt_lr: 0.001,
            memory_dynamics: TttMemoryDynamics::Ema,
            ..TttEncoderConfig::default()
        },
        &device,
    )
    .expect("strict in-place TTT wrapped model");
    assert_eq!(
        student.encoder.insertion_mode(),
        TttInsertionMode::InPlaceMlpStrict
    );

    let video = synthetic_video::<B>(1, model_config.in_channels, 4, 32, 32, &device);
    let expected = student
        .encoder
        .base
        .forward_video(video.clone(), None)
        .tokens;
    let mut state = student.fresh_state();
    let actual = student
        .encoder
        .forward_video_with_state(video, None, Some(expected.clone()), &mut state)
        .expect("stateful strict in-place TTT encode")
        .tokens;

    assert_tensor_close(
        "strict in-place MLP TTT first pass should preserve pretrained/base encoder",
        expected,
        actual,
        1.0e-5,
    );
    assert!(
        state.layers.iter().all(|layer| layer.fast_weight.is_some()),
        "strict in-place MLP TTT should store per-layer down-proj delta fast weights"
    );
}

#[test]
fn ttt_chunked_dense_rollout_matches_sequential_recurrence() {
    let device = Default::default();
    let model_config = burn_jepa::VJepaConfig::tiny_for_tests();
    let student = VJepaTttModel::from_model(
        VJepa2_1Model::<B>::new(&model_config, &device),
        TttEncoderConfig {
            layers: vec![0, 1],
            chunk_tokens: 2,
            rollout_chunk_frames: 1,
            ..TttEncoderConfig::default()
        },
        &device,
    )
    .expect("TTT wrapped model");
    let video = synthetic_video::<B>(0, model_config.in_channels, 4, 32, 32, &device);
    let teacher = student
        .encoder
        .base
        .forward_video(video.clone(), None)
        .tokens
        .detach();

    let mut sequential_state = student.fresh_state();
    let sequential = student
        .forward_single_frame_rollout_with_chunk_frames(
            video.clone(),
            Some(teacher.clone()),
            &mut sequential_state,
            1,
        )
        .expect("sequential rollout");
    let mut chunked_state = student.fresh_state();
    let chunked = student
        .forward_single_frame_rollout_with_chunk_frames(video, Some(teacher), &mut chunked_state, 4)
        .expect("chunked rollout");

    assert_tensor_close(
        "chunked dense rollout tokens should match sequential rollout",
        sequential.tokens,
        chunked.tokens,
        1.0e-4,
    );
    assert_ttt_state_close(
        "chunked dense rollout state should match sequential rollout",
        &sequential_state,
        &chunked_state,
        1.0e-4,
    );
}

#[test]
fn ttt_memory_alibi_chunked_rollout_matches_sequential_recurrence() {
    let device = Default::default();
    let model_config = burn_jepa::VJepaConfig::tiny_for_tests();
    let student = VJepaTttModel::from_model(
        VJepa2_1Model::<B>::new(&model_config, &device),
        TttEncoderConfig {
            layers: vec![0, 1],
            chunk_tokens: 2,
            rollout_chunk_frames: 1,
            memory_dynamics: TttMemoryDynamics::MemoryAlibi,
            memory_alibi_half_lives: vec![2, 8, 32],
            memory_alibi_read_weights: vec![0.5, 0.3, 0.2],
            memory_alibi_update_weights: vec![1.0, 1.0, 1.0],
            ..TttEncoderConfig::default()
        },
        &device,
    )
    .expect("TTT wrapped model");
    let video = synthetic_video::<B>(0, model_config.in_channels, 4, 32, 32, &device);
    let teacher = student
        .encoder
        .base
        .forward_video(video.clone(), None)
        .tokens
        .detach();

    let mut sequential_state = student.fresh_state();
    let sequential = student
        .forward_single_frame_rollout_with_chunk_frames(
            video.clone(),
            Some(teacher.clone()),
            &mut sequential_state,
            1,
        )
        .expect("sequential rollout");
    let mut chunked_state = student.fresh_state();
    let chunked = student
        .forward_single_frame_rollout_with_chunk_frames(video, Some(teacher), &mut chunked_state, 4)
        .expect("chunked rollout");

    assert_tensor_close(
        "memory-alibi chunked rollout tokens should match sequential rollout",
        sequential.tokens,
        chunked.tokens,
        1.0e-4,
    );
    assert_ttt_state_close(
        "memory-alibi chunked rollout state should match sequential rollout",
        &sequential_state,
        &chunked_state,
        1.0e-4,
    );
}

#[test]
fn ttt_chunked_sparse_batch_rollout_matches_sequential_recurrence() {
    let device = Default::default();
    let model_config = burn_jepa::VJepaConfig::tiny_for_tests();
    let student = VJepaTttModel::from_model(
        VJepa2_1Model::<B>::new(&model_config, &device),
        TttEncoderConfig {
            layers: vec![0, 1],
            chunk_tokens: 2,
            rollout_chunk_frames: 1,
            ..TttEncoderConfig::default()
        },
        &device,
    )
    .expect("TTT wrapped model");
    let video = Tensor::cat(
        vec![
            synthetic_video::<B>(0, model_config.in_channels, 4, 32, 32, &device),
            synthetic_video::<B>(1, model_config.in_channels, 4, 32, 32, &device),
        ],
        0,
    );
    let teacher = student
        .encoder
        .base
        .forward_video(video.clone(), None)
        .tokens
        .detach();
    let mask = SparseMaskBatch::<B>::from_rows(
        vec![vec![0, 2, 4, 6], vec![1, 3, 5, 7]],
        model_config.num_patches(),
        &device,
    )
    .expect("fixed-width sparse mask batch");

    let mut sequential_state = student.fresh_state();
    let sequential = student
        .forward_single_frame_rollout_sparse_batch_with_chunk_frames(
            video.clone(),
            &mask,
            Some(teacher.clone()),
            &mut sequential_state,
            1,
        )
        .expect("sequential sparse rollout");
    let mut chunked_state = student.fresh_state();
    let chunked = student
        .forward_single_frame_rollout_sparse_batch_with_chunk_frames(
            video,
            &mask,
            Some(teacher),
            &mut chunked_state,
            4,
        )
        .expect("chunked sparse rollout");

    assert_tensor_close(
        "chunked sparse rollout tokens should match sequential rollout",
        sequential.tokens,
        chunked.tokens,
        1.0e-4,
    );
    assert_int_tensor_equal(
        "chunked sparse rollout token indices should match sequential rollout",
        sequential.token_indices,
        chunked.token_indices,
    );
    assert_ttt_state_close(
        "chunked sparse rollout state should match sequential rollout",
        &sequential_state,
        &chunked_state,
        1.0e-4,
    );
}

#[test]
fn ttt_predictor_layer_zero_init_matches_pretrained_predictor() {
    let device = Default::default();
    let model_config = burn_jepa::VJepaConfig::tiny_for_tests();
    let base = VJepa2_1Model::<B>::new(&model_config, &device);
    let video = synthetic_video::<B>(0, model_config.in_channels, 4, 32, 32, &device);
    let dense = base.encode_video(video, None);
    let context =
        SparseTokenMask::new(vec![0, 2, 5, 7], model_config.num_patches()).expect("context mask");
    let target = SparseTokenMask::new(vec![1, 3], model_config.num_patches()).expect("target mask");
    let context_tokens = apply_token_mask(dense.tokens.clone(), context.to_tensor(1, &device));
    let expected = base
        .predictor
        .forward_sparse(context_tokens.clone(), &context, &target, dense.grid, 0)
        .expect("base predictor")
        .target_predictions;

    let student = VJepaTttModel::from_model(
        base,
        TttEncoderConfig {
            layer_placement: TttLayerPlacement::Explicit,
            layers: Vec::new(),
            predictor_layers: vec![0],
            ..TttEncoderConfig::default()
        },
        &device,
    )
    .expect("TTT predictor model");
    assert_eq!(student.predictor_ttt_layer_indices(), &[0]);
    let actual = student
        .forward_predictor_sparse(context_tokens, &context, &target, dense.grid, 0)
        .expect("TTT predictor")
        .target_predictions;
    assert_tensor_close(
        "zero-init predictor TTT should match pretrained/base predictor",
        expected,
        actual,
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
    config.loss.latent_regularization.weight = 1.0e-5;
    config.loss.latent_regularization.covariance_weight = 0.25;
    config.loss.latent_regularization.covariance_sketch_dim = 4;
    config.dataset.synthetic_len = 1;
    let report = train_ttt_distillation::<AB>(&config, &device).expect("training smoke");

    assert_eq!(report.steps, 3);
    assert_eq!(report.samples, 6);
    assert_eq!(report.loss_trace.len(), 3);
    assert!(report.initial_loss.is_finite());
    assert!(report.best_loss.is_finite());
    assert!(report.final_loss.is_finite());
    assert!(report.pre_train_eval_loss.is_some());
    assert!(
        report
            .pre_train_eval_regularizer_loss
            .is_some_and(f64::is_finite)
    );
    assert!(report.pre_train_eval_cosine.is_some());
    assert!(report.eval_loss.is_some());
    assert!(report.eval_regularizer_loss.is_some_and(f64::is_finite));
    assert!(report.eval_cosine.is_some());
    assert!(report.latent_regularization.active);
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
    assert!(temporal.samples > 0);
    assert!(temporal.reset_each_frame_loss.is_some_and(f64::is_finite));
    assert!(temporal.reverse_order_loss.is_some_and(f64::is_finite));
    assert!(temporal.shuffle_order_loss.is_some_and(f64::is_finite));
    assert!(temporal.freeze_fast_update_loss.is_some_and(f64::is_finite));
    let segments = report
        .temporal_segments
        .as_ref()
        .expect("TTT eval should report temporal segment diagnostics");
    assert_eq!(segments.samples, temporal.samples);
    assert_eq!(segments.segments.len(), 3);
    assert!(
        segments
            .segments
            .iter()
            .all(|segment| segment.loss.is_finite() && segment.cosine.is_finite())
    );
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
fn ttt_in_place_mlp_distillation_training_smoke_runs() {
    let device = Default::default();
    let mut config = BurnJepaTrainConfig::default();
    config.ttt.insertion = TttInsertionMode::InPlaceMlp;
    config.ttt.layer_placement = TttLayerPlacement::Explicit;
    config.ttt.layers = vec![0, 1];
    config.ttt.chunk_tokens = 2;
    config.model.save_model = true;
    let temp = tempfile::tempdir().expect("tempdir");
    config.model.output_dir = temp.path().join("ttt-in-place-train");
    config.training.max_steps = 2;
    config.training.batch_size = 1;
    config.training.eval_steps = 1;
    config.training.learning_rate = 5.0e-3;
    config.dataset.synthetic_len = 2;

    let report =
        train_ttt_distillation::<AB>(&config, &device).expect("in-place MLP TTT training smoke");
    assert_eq!(report.steps, 2);
    assert_eq!(report.memory.layers, vec![0, 1]);
    assert!(report.final_loss.is_finite());
    assert!(report.eval_loss.is_some_and(f64::is_finite));
    assert!(report.memory.fast_weight_bytes_f32 > 0);
    assert!(report.memory.trainable_param_bytes_f32 > 0);

    let model_path = report.model_path.expect("saved in-place TTT model path");
    let model_config = burn_jepa::VJepaConfig::tiny_for_tests();
    let base = VJepa2_1Model::<AB>::new(&model_config, &device);
    let loaded = VJepaTttModel::from_model(base, config.ttt.clone(), &device)
        .expect("fresh in-place TTT model")
        .load_file(
            model_path,
            &NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
            &device,
        )
        .expect("reload saved in-place TTT model");
    assert_eq!(
        loaded.encoder.insertion_mode(),
        TttInsertionMode::InPlaceMlp
    );
}

#[test]
fn ttt_strict_in_place_mlp_distillation_training_smoke_runs() {
    let device = Default::default();
    let mut config = BurnJepaTrainConfig::default();
    config.ttt.insertion = TttInsertionMode::InPlaceMlpStrict;
    config.ttt.memory_dynamics = TttMemoryDynamics::Ema;
    config.ttt.layer_placement = TttLayerPlacement::Explicit;
    config.ttt.layers = vec![0, 1];
    config.ttt.chunk_tokens = 2;
    config.ttt.ttt_lr = 1.0e-3;
    config.model.save_model = false;
    let temp = tempfile::tempdir().expect("tempdir");
    config.model.output_dir = temp.path().join("ttt-strict-in-place-train");
    config.training.max_steps = 1;
    config.training.batch_size = 1;
    config.training.eval_steps = 1;
    config.training.learning_rate = 1.0e-3;
    config.dataset.synthetic_len = 1;

    let report = train_ttt_distillation::<AB>(&config, &device)
        .expect("strict in-place MLP TTT training smoke");
    assert_eq!(report.steps, 1);
    assert_eq!(report.memory.insertion, TttInsertionMode::InPlaceMlpStrict);
    assert!(report.final_loss.is_finite());
    assert!(report.eval_loss.is_some_and(f64::is_finite));
}

#[test]
fn ttt_stream_training_smoke_carries_and_decays_state() {
    let device = Default::default();
    let mut config = BurnJepaTrainConfig::default();
    config.model.save_model = false;
    let temp = tempfile::tempdir().expect("tempdir");
    config.model.output_dir = temp.path().join("ttt-stream-train");
    config.training.max_steps = 3;
    config.training.batch_size = 1;
    config.training.eval_steps = 1;
    config.training.learning_rate = 5.0e-3;
    config.training.stream.enabled = true;
    config.training.stream.state_decay = 0.9;
    config.training.stream.state_l2_weight = 1.0e-6;
    config.training.stream.update_l2_weight = 1.0e-6;
    config.dataset.synthetic_len = 1;

    let report = train_ttt_distillation::<AB>(&config, &device).expect("stream training smoke");

    assert_eq!(report.steps, 3);
    assert!(report.final_loss.is_finite());
    assert!(report.eval_loss.is_some_and(f64::is_finite));
    assert!(report.stream.enabled);
    assert!(report.stream.detach_between_steps);
    assert_eq!(report.stream.reset_steps, 1);
    assert_eq!(report.stream.carried_steps, 2);
    assert_eq!(report.stream.optimizer_steps, Some(3));
    assert_eq!(report.stream.reset_optimizer_steps, Some(1));
    assert_eq!(report.stream.carried_optimizer_steps, Some(2));
    assert_eq!(report.stream.detached_steps, 3);
    assert_eq!(report.stream.decayed_steps, 3);
    assert!(!report.stream.curriculum_enabled);
    assert_eq!(report.stream.final_effective_reset_interval_steps, 0);
    assert_eq!(report.stream.state_decay, 0.9);
    assert_eq!(report.stream.state_l2_weight, 1.0e-6);
    assert_eq!(report.stream.update_l2_weight, 1.0e-6);
    assert_eq!(report.loss_trace.len(), 3);
    assert_eq!(
        report.loss_trace[0].stream_step,
        Some(TttStreamStepKind::Reset)
    );
    assert_eq!(
        report.loss_trace[1].stream_step,
        Some(TttStreamStepKind::Carried)
    );
    assert_eq!(
        report.loss_trace[2].stream_step,
        Some(TttStreamStepKind::Carried)
    );
    assert_eq!(report.loss_trace[0].effective_reset_interval_steps, Some(0));
}

#[test]
fn ttt_stream_training_resets_on_manifest_clip_change() {
    let device = Default::default();
    let temp = tempfile::tempdir().expect("tempdir");
    let frame_dir = temp.path().join("frames");
    std::fs::create_dir_all(&frame_dir).expect("frame dir");
    let mut frame_paths = Vec::new();
    for frame in 0..6 {
        let path = frame_dir.join(format!("frame-{frame}.png"));
        let image = image::RgbImage::from_fn(32, 32, |x, y| {
            image::Rgb([
                ((x + y + frame) % 255) as u8,
                ((x * 3 + frame * 5) % 255) as u8,
                ((y * 7 + frame) % 255) as u8,
            ])
        });
        image.save(&path).expect("save frame");
        frame_paths.push(path);
    }
    let frames = |start: usize| {
        frame_paths[start..start + 4]
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect::<Vec<_>>()
    };
    let manifest = temp.path().join("stream-manifest.jsonl");
    let rows = [
        serde_json::json!({
            "clip_id": "a",
            "domain": "test",
            "start_frame": 0,
            "frames": frames(0),
            "precomputed_context_indices": [0, 1, 4, 5],
            "precomputed_target_indices": [2, 6]
        }),
        serde_json::json!({
            "clip_id": "a",
            "domain": "test",
            "start_frame": 1,
            "frames": frames(1),
            "precomputed_context_indices": [0, 1, 4, 5],
            "precomputed_target_indices": [2, 6]
        }),
        serde_json::json!({
            "clip_id": "b",
            "domain": "test",
            "start_frame": 0,
            "frames": frames(2),
            "precomputed_context_indices": [0, 1, 4, 5],
            "precomputed_target_indices": [2, 6]
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
    config.model.output_dir = temp.path().join("ttt-stream-manifest");
    config.dataset.kind = burn_jepa::JepaDatasetKind::Manifest;
    config.dataset.sample_kind = burn_jepa::JepaSampleKind::Video;
    config.dataset.train_manifest = Some(manifest.clone());
    config.dataset.eval_manifest = Some(manifest);
    config.dataset.synthetic_len = 3;
    config.training.max_steps = 3;
    config.training.batch_size = 1;
    config.training.eval_steps = 0;
    config.training.stream.enabled = true;
    config.training.stream.reset_interval_steps = 0;
    config.training.sparse_rollout = TttSparseRolloutMode::ContextMask;
    config.training.mask = Some(burn_jepa::TrainingMaskConfig::ManifestPrecomputedMasks);

    let report = train_ttt_distillation::<AB>(&config, &device).expect("stream manifest smoke");

    assert_eq!(report.stream.reset_steps, 2);
    assert_eq!(report.stream.carried_steps, 1);
    assert_eq!(report.stream.detached_steps, 3);
    assert_eq!(report.rollout.mode, TttRolloutReportMode::SparseContext);
    assert!(report.final_loss.is_finite());
}

#[test]
fn ttt_stream_training_packs_independent_manifest_streams() {
    let device = Default::default();
    let temp = tempfile::tempdir().expect("tempdir");
    let frame_dir = temp.path().join("packed-frames");
    std::fs::create_dir_all(&frame_dir).expect("frame dir");
    let mut frame_paths = Vec::new();
    for frame in 0..10 {
        let path = frame_dir.join(format!("frame-{frame}.png"));
        let image = image::RgbImage::from_fn(32, 32, |x, y| {
            image::Rgb([
                ((x + frame * 11) % 255) as u8,
                ((y * 2 + frame * 3) % 255) as u8,
                ((x + y + frame * 5) % 255) as u8,
            ])
        });
        image.save(&path).expect("save frame");
        frame_paths.push(path);
    }
    let frames = |start: usize| {
        frame_paths[start..start + 4]
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect::<Vec<_>>()
    };
    let manifest = temp.path().join("packed-stream-manifest.jsonl");
    let rows = [
        serde_json::json!({
            "clip_id": "a",
            "domain": "test",
            "start_frame": 0,
            "frames": frames(0),
            "precomputed_context_indices": [0, 1, 4, 5],
            "precomputed_target_indices": [2, 6]
        }),
        serde_json::json!({
            "clip_id": "a",
            "domain": "test",
            "start_frame": 1,
            "frames": frames(1),
            "precomputed_context_indices": [0, 1, 4, 5],
            "precomputed_target_indices": [2, 6]
        }),
        serde_json::json!({
            "clip_id": "b",
            "domain": "test",
            "start_frame": 0,
            "frames": frames(4),
            "precomputed_context_indices": [0, 3, 4, 7],
            "precomputed_target_indices": [1, 5]
        }),
        serde_json::json!({
            "clip_id": "b",
            "domain": "test",
            "start_frame": 1,
            "frames": frames(5),
            "precomputed_context_indices": [0, 3, 4, 7],
            "precomputed_target_indices": [1, 5]
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
    config.model.output_dir = temp.path().join("ttt-packed-stream-manifest");
    config.dataset.kind = burn_jepa::JepaDatasetKind::Manifest;
    config.dataset.sample_kind = burn_jepa::JepaSampleKind::Video;
    config.dataset.train_manifest = Some(manifest.clone());
    config.dataset.eval_manifest = Some(manifest);
    config.dataset.synthetic_len = 4;
    config.training.max_steps = 2;
    config.training.batch_size = 2;
    config.training.prefetch_batches = true;
    config.training.eval_steps = 0;
    config.training.learning_rate = 5.0e-3;
    config.training.batching = burn_jepa::TrainingBatchingMode::PackedStreams;
    config.training.stream.enabled = true;
    config.training.stream.reset_interval_steps = 0;
    config.training.stream.state_decay = 0.95;
    config.training.sparse_rollout = TttSparseRolloutMode::ContextMask;
    config.training.mask = Some(burn_jepa::TrainingMaskConfig::ManifestPrecomputedMasks);

    let report =
        train_ttt_distillation::<AB>(&config, &device).expect("packed stream training smoke");

    assert_eq!(report.steps, 2);
    assert_eq!(report.samples, 4);
    assert_eq!(report.stream.reset_steps, 2);
    assert_eq!(report.stream.carried_steps, 2);
    assert_eq!(report.stream.packed_batches, 2);
    assert_eq!(report.stream.max_packed_batch_size, 2);
    assert_eq!(report.stream.active_streams, 2);
    assert_eq!(report.stream.max_active_streams, 2);
    assert_eq!(report.stream.optimizer_steps, Some(2));
    assert_eq!(report.stream.reset_optimizer_steps, Some(1));
    assert_eq!(report.stream.carried_optimizer_steps, Some(1));
    assert_eq!(report.stream.mixed_optimizer_steps, Some(0));
    assert_eq!(report.stream.detached_steps, 4);
    assert_eq!(report.stream.decayed_steps, 4);
    assert_eq!(report.train_stage.data_ms, 0);
    assert_eq!(
        report.loss_trace[0].stream_step,
        Some(TttStreamStepKind::Reset)
    );
    assert_eq!(
        report.loss_trace[1].stream_step,
        Some(TttStreamStepKind::Carried)
    );
    assert!(report.final_loss.is_finite());
}

#[test]
fn ttt_stream_packed_batches_shrink_when_stream_count_is_smaller_than_batch_size() {
    let device = Default::default();
    let temp = tempfile::tempdir().expect("tempdir");
    let frame_dir = temp.path().join("partial-frames");
    std::fs::create_dir_all(&frame_dir).expect("frame dir");
    let mut frame_paths = Vec::new();
    for frame in 0..5 {
        let path = frame_dir.join(format!("frame-{frame}.png"));
        image::RgbImage::from_pixel(32, 32, image::Rgb([frame as u8, 3, 7]))
            .save(&path)
            .expect("save frame");
        frame_paths.push(path);
    }
    let frames = |start: usize| {
        frame_paths[start..start + 4]
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect::<Vec<_>>()
    };
    let manifest = temp.path().join("partial-stream-manifest.jsonl");
    let rows = [
        serde_json::json!({
            "clip_id": "solo",
            "domain": "test",
            "start_frame": 0,
            "frames": frames(0),
            "precomputed_context_indices": [0, 1, 4, 5],
            "precomputed_target_indices": [2, 6]
        }),
        serde_json::json!({
            "clip_id": "solo",
            "domain": "test",
            "start_frame": 1,
            "frames": frames(1),
            "precomputed_context_indices": [0, 1, 4, 5],
            "precomputed_target_indices": [2, 6]
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
    config.model.output_dir = temp.path().join("ttt-partial-packed-stream");
    config.dataset.kind = burn_jepa::JepaDatasetKind::Manifest;
    config.dataset.sample_kind = burn_jepa::JepaSampleKind::Video;
    config.dataset.train_manifest = Some(manifest.clone());
    config.dataset.eval_manifest = Some(manifest);
    config.dataset.synthetic_len = 2;
    config.training.max_steps = 2;
    config.training.batch_size = 4;
    config.training.eval_steps = 0;
    config.training.batching = burn_jepa::TrainingBatchingMode::PackedStreams;
    config.training.stream.enabled = true;
    config.training.stream.reset_interval_steps = 0;
    config.training.sparse_rollout = TttSparseRolloutMode::ContextMask;
    config.training.mask = Some(burn_jepa::TrainingMaskConfig::ManifestPrecomputedMasks);

    let report =
        train_ttt_distillation::<AB>(&config, &device).expect("partial packed stream training");

    assert_eq!(report.steps, 2);
    assert_eq!(
        report.samples, 2,
        "actual sample count should reflect partial packed batches"
    );
    assert_eq!(report.stream.packed_batches, 2);
    assert_eq!(report.stream.max_packed_batch_size, 1);
    assert_eq!(report.stream.active_streams, 1);
    assert_eq!(report.stream.reset_steps, 1);
    assert_eq!(report.stream.carried_steps, 1);
    assert_eq!(report.stream.detached_steps, 2);
    assert!(report.final_loss.is_finite());
}

#[test]
fn ttt_stream_training_rejects_duplicate_stream_rows_in_one_batch() {
    let device = Default::default();
    let temp = tempfile::tempdir().expect("tempdir");
    let frame_dir = temp.path().join("duplicate-frames");
    std::fs::create_dir_all(&frame_dir).expect("frame dir");
    let mut frame_paths = Vec::new();
    for frame in 0..5 {
        let path = frame_dir.join(format!("frame-{frame}.png"));
        image::RgbImage::from_pixel(32, 32, image::Rgb([frame as u8, 0, 0]))
            .save(&path)
            .expect("save frame");
        frame_paths.push(path);
    }
    let frames = |start: usize| {
        frame_paths[start..start + 4]
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect::<Vec<_>>()
    };
    let manifest = temp.path().join("duplicate-stream-manifest.jsonl");
    let rows = [
        serde_json::json!({
            "clip_id": "a",
            "domain": "test",
            "start_frame": 0,
            "frames": frames(0)
        }),
        serde_json::json!({
            "clip_id": "a",
            "domain": "test",
            "start_frame": 1,
            "frames": frames(1)
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
    config.model.output_dir = temp.path().join("ttt-duplicate-stream-manifest");
    config.dataset.kind = burn_jepa::JepaDatasetKind::Manifest;
    config.dataset.sample_kind = burn_jepa::JepaSampleKind::Video;
    config.dataset.train_manifest = Some(manifest.clone());
    config.dataset.eval_manifest = Some(manifest);
    config.dataset.synthetic_len = 2;
    config.training.max_steps = 1;
    config.training.batch_size = 2;
    config.training.eval_steps = 0;
    config.training.batching = burn_jepa::TrainingBatchingMode::Sequential;
    config.training.stream.enabled = true;

    let error = format!(
        "{:#}",
        train_ttt_distillation::<AB>(&config, &device)
            .expect_err("duplicate stream rows should be rejected")
    );
    assert!(
        error.contains("at most one window per stream key"),
        "unexpected duplicate-stream error: {error}"
    );
}

#[test]
fn ttt_stream_eval_carries_manifest_state_between_windows() {
    let device = Default::default();
    let temp = tempfile::tempdir().expect("tempdir");
    let frame_dir = temp.path().join("frames");
    std::fs::create_dir_all(&frame_dir).expect("frame dir");
    let mut frame_paths = Vec::new();
    for frame in 0..5 {
        let path = frame_dir.join(format!("frame-{frame}.png"));
        let image = image::RgbImage::from_fn(32, 32, |x, y| {
            image::Rgb([
                ((x + frame) % 255) as u8,
                ((y + frame * 3) % 255) as u8,
                ((x + y + frame * 7) % 255) as u8,
            ])
        });
        image.save(&path).expect("save frame");
        frame_paths.push(path);
    }
    let frames = |start: usize| {
        frame_paths[start..start + 4]
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect::<Vec<_>>()
    };
    let manifest = temp.path().join("stream-eval-manifest.jsonl");
    let rows = [
        serde_json::json!({
            "clip_id": "stream-a",
            "domain": "test",
            "start_frame": 0,
            "frames": frames(0)
        }),
        serde_json::json!({
            "clip_id": "stream-a",
            "domain": "test",
            "start_frame": 1,
            "frames": frames(1)
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
    config.model.output_dir = temp.path().join("ttt-stream-eval");
    config.model.save_model = true;
    config.dataset.kind = burn_jepa::JepaDatasetKind::Manifest;
    config.dataset.sample_kind = burn_jepa::JepaSampleKind::Video;
    config.dataset.train_manifest = Some(manifest.clone());
    config.dataset.eval_manifest = Some(manifest);
    config.dataset.synthetic_len = 2;
    config.training.max_steps = 1;
    config.training.batch_size = 1;
    config.training.eval_steps = 0;
    config.training.stream.enabled = true;
    config.training.stream.reset_interval_steps = 0;

    let train = train_ttt_distillation::<AB>(&config, &device).expect("stream eval train");
    let model_path = train.model_path.expect("saved stream eval model");
    let eval = evaluate_ttt_model_file::<AB>(&config, model_path, &device, 2).expect("stream eval");

    assert_eq!(eval.model_kind, TttEvalModelKind::Checkpoint);
    assert!(eval.model_path.is_some());
    assert!(eval.feature_stability.is_none());
    assert!(eval.stream.enabled);
    assert_eq!(eval.stream.reset_steps, 1);
    assert_eq!(eval.stream.carried_steps, 1);
    assert_eq!(eval.stream.detached_steps, 2);
    assert_eq!(eval.stream.optimizer_steps, None);
    let long_rollout = eval
        .long_rollout
        .as_ref()
        .expect("stream eval should report longitudinal rollout metrics");
    assert_eq!(long_rollout.samples, 2);
    assert_eq!(long_rollout.windows, 2);
    assert_eq!(long_rollout.streams, 1);
    assert_eq!(long_rollout.longest_stream_windows, 2);
    assert_eq!(long_rollout.longest_consecutive_windows, 2);
    assert_eq!(
        long_rollout
            .segments
            .iter()
            .map(|s| s.samples)
            .sum::<usize>(),
        2
    );
    assert!(
        long_rollout
            .late_minus_early_loss
            .is_some_and(f64::is_finite)
    );
    assert!(
        long_rollout
            .stream_segments
            .iter()
            .all(|stream| stream.loss.is_finite() && stream.cosine.is_finite())
    );
    assert!(eval.loss.is_finite());

    let base_eval =
        evaluate_ttt_base_sparse::<AB>(&config, &device, 2).expect("base sparse stream eval");
    assert_eq!(
        base_eval.model_kind,
        TttEvalModelKind::BaseSparseZeroInitTtt
    );
    assert!(base_eval.model_path.is_none());
    assert_eq!(base_eval.stream.reset_steps, 1);
    assert_eq!(base_eval.stream.carried_steps, 1);
    assert!(base_eval.loss.is_finite());

    config.training.eval_feature_stability_diagnostics = true;
    let stability_eval =
        evaluate_ttt_base_sparse::<AB>(&config, &device, 2).expect("feature stability eval");
    let stability = stability_eval
        .feature_stability
        .expect("feature stability should be opt-in");
    assert_eq!(stability.samples, 2);
    assert!(stability.spatial_std_rms.is_finite());
    assert!(stability.relative_spread.is_finite());
    assert!(stability.mean_pairwise_token_cosine.is_finite());
    assert!(stability.collapse_score.is_finite());
}

#[test]
fn ttt_stream_manifest_training_requires_identity_metadata() {
    let device = Default::default();
    let temp = tempfile::tempdir().expect("tempdir");
    let frame_dir = temp.path().join("frames");
    std::fs::create_dir_all(&frame_dir).expect("frame dir");
    let mut frame_paths = Vec::new();
    for frame in 0..4 {
        let path = frame_dir.join(format!("frame-{frame}.png"));
        image::RgbImage::from_pixel(32, 32, image::Rgb([frame as u8, 0, 0]))
            .save(&path)
            .expect("save frame");
        frame_paths.push(path.to_string_lossy().to_string());
    }
    let manifest = temp.path().join("missing-stream-metadata.jsonl");
    std::fs::write(
        &manifest,
        serde_json::json!({
            "frames": frame_paths
        })
        .to_string(),
    )
    .expect("write manifest");

    let mut config = BurnJepaTrainConfig::default();
    config.model.output_dir = temp.path().join("ttt-stream-missing-metadata");
    config.model.save_model = false;
    config.dataset.kind = burn_jepa::JepaDatasetKind::Manifest;
    config.dataset.sample_kind = burn_jepa::JepaSampleKind::Video;
    config.dataset.train_manifest = Some(manifest.clone());
    config.dataset.eval_manifest = Some(manifest);
    config.training.max_steps = 1;
    config.training.batch_size = 1;
    config.training.eval_steps = 0;
    config.training.stream.enabled = true;

    let error = format!(
        "{:#}",
        train_ttt_distillation::<AB>(&config, &device)
            .expect_err("stream manifest should require identity metadata")
    );
    assert!(
        error.contains("requires clip_id or source metadata"),
        "unexpected stream metadata validation error: {error}"
    );
}

#[test]
fn ttt_predictor_layers_train_with_predictor_loss_smoke() {
    let device = Default::default();
    let mut config = BurnJepaTrainConfig::default();
    let temp = tempfile::tempdir().expect("tempdir");
    config.model.output_dir = temp.path().join("ttt-predictor-train");
    config.model.save_model = false;
    config.training.max_steps = 2;
    config.training.batch_size = 2;
    config.training.eval_steps = 1;
    config.training.mask = Some(burn_jepa::TrainingMaskConfig::PrecomputedMasks {
        context_indices: vec![0, 2, 5, 7],
        target_indices: vec![1, 3],
    });
    config.loss.feature_loss_weight = 0.0;
    config.loss.predictor_loss_weight = 0.25;
    config.ttt.layer_placement = TttLayerPlacement::Explicit;
    config.ttt.layers.clear();
    config.ttt.predictor_layers = vec![0];

    let report = train_ttt_distillation::<AB>(&config, &device).expect("predictor TTT smoke");
    assert_eq!(report.memory.layers, Vec::<usize>::new());
    assert_eq!(report.memory.predictor_layers, vec![0]);
    assert!(report.initial_loss.is_finite());
    assert!(report.final_loss.is_finite());
    assert!(report.eval_loss.is_some_and(f64::is_finite));
    assert!(report.eval_feature_loss.is_some_and(f64::is_finite));
    assert!(report.eval_predictor_loss.is_some_and(f64::is_finite));
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
    assert!(!report.rollout.frozen_sparse_patchify);
    assert!(report.final_loss.is_finite());
    assert!(report.eval_loss.is_some_and(f64::is_finite));
    assert!(report.teacher_forced_eval_loss.is_none());
    assert!(report.teacher_forcing_cosine_gap.is_none());
    assert!(report.eval_full_loss.is_some_and(f64::is_finite));
}

#[test]
fn ttt_sparse_training_can_interleave_dense_full_token_samples() {
    let device = Default::default();
    let mut config = BurnJepaTrainConfig::default();
    config.model.save_model = false;
    let temp = tempfile::tempdir().expect("tempdir");
    config.model.output_dir = temp.path().join("ttt-dense-sample-mix");
    config.training.max_steps = 4;
    config.training.batch_size = 1;
    config.training.eval_steps = 0;
    config.training.learning_rate = 1.0e-3;
    config.training.sparse_rollout = TttSparseRolloutMode::ContextMask;
    config.training.mask = Some(burn_jepa::TrainingMaskConfig::PrecomputedMasks {
        context_indices: vec![0, 2, 5, 7],
        target_indices: vec![1, 3],
    });
    config.training.dense_samples.enabled = true;
    config.training.dense_samples.interval_steps = 2;
    config.dataset.synthetic_len = 1;

    let report = train_ttt_distillation::<AB>(&config, &device).expect("dense sample mix");

    assert_eq!(report.steps, 4);
    assert_eq!(report.dense_samples.dense_steps, 2);
    assert_eq!(report.dense_samples.sparse_steps, 2);
    assert_eq!(report.rollout.mode, TttRolloutReportMode::SparseContext);
    assert!(report.mask.is_some());
    assert!(report.final_loss.is_finite());
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
fn ttt_training_rejects_negative_gradient_clip_norm() {
    let mut config = BurnJepaTrainConfig::default();
    config.training.gradient_clip_norm = -1.0;
    let error = config
        .validate_for_ttt()
        .expect_err("negative gradient clip norm should be invalid");
    assert!(
        error.to_string().contains("gradient_clip_norm"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn ttt_training_saves_best_sampled_checkpoint() {
    let device = Default::default();
    let mut config = BurnJepaTrainConfig::default();
    let temp = tempfile::tempdir().expect("tempdir");
    config.model.output_dir = temp.path().join("ttt-best-checkpoint");
    config.model.save_model = true;
    config.model.save_best_model = true;
    config.training.max_steps = 2;
    config.training.batch_size = 1;
    config.training.loss_trace_interval = 1;
    config.training.gradient_clip_norm = 0.5;
    config.dataset.synthetic_len = 1;

    let report = train_ttt_distillation::<AB>(&config, &device).expect("training with best save");
    let final_model = report.model_path.expect("final TTT model path");
    let best_model = report.best_model_path.expect("best TTT model path");

    assert!(
        final_model.exists(),
        "final model missing at {final_model:?}"
    );
    assert!(best_model.exists(), "best model missing at {best_model:?}");
    assert_eq!(
        best_model.file_name().and_then(|name| name.to_str()),
        Some("ttt-model-best.mpk")
    );
    assert!(
        report
            .best_model_step
            .is_some_and(|step| (1..=2).contains(&step))
    );
    assert_eq!(report.gradient_clip_norm, 0.5);
}

#[test]
fn ttt_training_best_checkpoint_skips_dense_warmup_by_default() {
    let device = Default::default();
    let mut config = BurnJepaTrainConfig::default();
    let temp = tempfile::tempdir().expect("tempdir");
    config.model.output_dir = temp.path().join("ttt-best-deploy-rollout");
    config.model.save_model = true;
    config.model.save_best_model = true;
    config.model.best_checkpoint_selection = TttBestCheckpointSelection::DeployRollout;
    config.training.max_steps = 2;
    config.training.batch_size = 1;
    config.training.loss_trace_interval = 1;
    config.training.dense_samples.enabled = true;
    config.training.dense_samples.warmup_steps = 1;
    config.training.dense_samples.interval_steps = 0;
    config.training.mask = Some(burn_jepa::TrainingMaskConfig::PrecomputedMasks {
        context_indices: vec![0, 2, 5, 7],
        target_indices: vec![1, 3],
    });
    config.training.sparse_rollout = TttSparseRolloutMode::TargetMask;
    config.training.sparse_patchify_training = TttSparsePatchifyTrainingMode::DensePatchEmbed;
    config.dataset.synthetic_len = 1;

    let report =
        train_ttt_distillation::<AB>(&config, &device).expect("training with dense warmup");

    assert_eq!(
        report.best_checkpoint_selection,
        TttBestCheckpointSelection::DeployRollout
    );
    assert_eq!(
        report.best_model_step,
        Some(2),
        "default best checkpoint should track deploy sparse rollout, not dense warmup"
    );
    assert_eq!(report.dense_samples.dense_steps, 1);
    assert_eq!(report.dense_samples.sparse_steps, 1);
    assert!(report.best_checkpoint_loss.is_some());
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

fn assert_int_tensor_equal<const D: usize>(
    label: &str,
    expected: Tensor<B, D, Int>,
    actual: Tensor<B, D, Int>,
) {
    assert_eq!(
        expected.shape(),
        actual.shape(),
        "{label}: tensor shapes differ"
    );
    let shape = expected.shape();
    let expected = expected.into_data().to_vec::<i64>().expect("tensor values");
    let actual = actual.into_data().to_vec::<i64>().expect("tensor values");
    assert_eq!(
        expected, actual,
        "{label}: tensor values differ shape={shape:?}"
    );
}

fn assert_ttt_state_close(
    label: &str,
    expected: &TttState<B>,
    actual: &TttState<B>,
    tolerance: f32,
) {
    assert_eq!(
        expected.layers.len(),
        actual.layers.len(),
        "{label}: state layer count differs"
    );
    for (layer_index, (expected, actual)) in
        expected.layers.iter().zip(actual.layers.iter()).enumerate()
    {
        match (&expected.fast_weight, &actual.fast_weight) {
            (Some(expected), Some(actual)) => assert_tensor_close(
                &format!("{label} layer {layer_index}"),
                expected.clone(),
                actual.clone(),
                tolerance,
            ),
            (None, None) => {}
            _ => panic!("{label}: fast-weight presence differs at layer {layer_index}"),
        }
        match (&expected.fast_weight_banks, &actual.fast_weight_banks) {
            (Some(expected), Some(actual)) => assert_tensor_close(
                &format!("{label} layer {layer_index} banks"),
                expected.clone(),
                actual.clone(),
                tolerance,
            ),
            (None, None) => {}
            _ => panic!("{label}: banked fast-weight presence differs at layer {layer_index}"),
        }
    }
}

fn tensor_values<const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().expect("tensor values")
}
