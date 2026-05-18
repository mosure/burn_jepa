use super::*;
use burn_jepa::{
    AnyUpConfig, BurnJepaPackageModelKind, BurnJepaPipelinePackageManifest, FeatureFrameEncodePath,
    FeatureFrameJepaEncoder, FeatureFrameJepaEncoderKind, PatchDiffRefreshState, TttEncoderConfig,
    VJepa2_1Model, VJepaTttModel, coords_to_token_index, write_burnpack_parts_for_browser,
    write_pipeline_package_manifest,
};

fn tiny_viewer_config() -> BevyJepaConfig {
    BevyJepaConfig {
        encoder_source: BevyJepaEncoderSource::TinyTest,
        ttt_model_path: None,
        jepa_checkpoint_dir: None,
        jepa_config_path: None,
        ..BevyJepaConfig::default()
    }
}

fn values4(tensor: Tensor<JepaBevyBackend, 4>) -> Vec<f32> {
    tensor.to_data().to_vec::<f32>().expect("tensor values")
}

#[test]
fn center_prior_mask_keeps_requested_density() {
    let grid = TokenGridShape::new(1, 4, 4);
    let mask = center_prior_mask(grid, 5).expect("mask");
    assert_eq!(mask.dense_len(), 16);
    assert_eq!(mask.len(), 5);
}

#[test]
fn synthetic_source_uses_model_sized_tensor() {
    let device = JepaBevyDevice::default();
    let image = synthetic_image_tensor(0, 64, &device);
    assert_eq!(image.shape().dims::<4>(), [1, 3, 64, 64]);
}

#[test]
fn default_source_is_camera() {
    assert_eq!(
        BevyJepaConfig::default().source,
        BevyJepaFrameSource::Camera
    );
}

#[test]
fn default_mask_source_is_patch_diff() {
    assert_eq!(
        BevyJepaConfig::default().encoder_source,
        BevyJepaEncoderSource::TrainedTtt
    );
    assert_eq!(
        BevyJepaConfig::default().encode_path,
        BevyJepaEncodePath::Auto
    );
    assert_eq!(
        BevyJepaConfig::default().mask_source,
        BevyJepaMaskSource::PatchDiff
    );
    assert!(DEFAULT_MODEL_MANIFEST_PATH.ends_with("vjepa2_1_ttt/manifest.json"));
    assert_eq!(
        default_model_manifest_path_for_profile(BevyJepaModelPackageProfile::Vjepa21Base),
        std::path::PathBuf::from("target/burn-jepa-web/model/vjepa2_1_base/manifest.json")
    );
    assert!(DEFAULT_ANYUP_MODEL_MANIFEST_PATH.ends_with("anyup_multi_backbone/manifest.json"));
    assert_eq!(
        default_anyup_model_manifest_path_for_profile(
            BevyJepaAnyUpModelPackageProfile::AnyupMultiBackbone
        ),
        std::path::PathBuf::from("target/burn_anyup/anyup_multi_backbone/manifest.json")
    );
    assert!(DEFAULT_TTT_MODEL_PATH.contains("burn-jepa-production-final"));
    assert!(BevyJepaConfig::default().model_manifest_path.is_none());
    assert!(BevyJepaConfig::default().model_cache_dir.is_none());
    assert_eq!(
        BevyJepaConfig::default().model_profile,
        BevyJepaModelPackageProfile::Vjepa21Ttt
    );
    assert_eq!(
        BevyJepaConfig::default().model_base_url,
        burn_jepa::burn_jepa_model_profile_base_url(BevyJepaModelPackageProfile::Vjepa21Ttt)
    );
    assert!(BevyJepaConfig::default().model_auto_download);
    assert!(
        BevyJepaConfig::default()
            .anyup_model_manifest_path
            .is_none()
    );
    assert!(BevyJepaConfig::default().anyup_model_cache_dir.is_none());
    assert_eq!(
        BevyJepaConfig::default().anyup_model_profile,
        BevyJepaAnyUpModelPackageProfile::AnyupMultiBackbone
    );
    assert_eq!(
        BevyJepaConfig::default().anyup_model_base_url,
        burn_jepa::burn_anyup_model_profile_base_url(
            BevyJepaAnyUpModelPackageProfile::AnyupMultiBackbone
        )
    );
    assert!(BevyJepaConfig::default().anyup_model_auto_download);
    assert!(BevyJepaConfig::default().ttt_model_path.is_none());
    assert!(DEFAULT_VJEPA21_CHECKPOINT_DIR.starts_with("~/"));
    assert!(DEFAULT_VJEPA21_CONFIG_PATH.starts_with("~/"));
    assert_eq!(BevyJepaConfig::default().image_size, DEFAULT_IMAGE_SIZE);
    assert_eq!(
        BevyJepaConfig::default().pipeline_image_size(),
        DEFAULT_IMAGE_SIZE
    );
    assert_eq!(BevyJepaConfig::default().context_density, 1.0);
    assert_eq!(
        BevyJepaConfig::default().min_context_density,
        DEFAULT_MIN_CONTEXT_DENSITY
    );
    assert_eq!(BevyJepaConfig::default().min_context_density, 0.0);
    assert!(
        (BevyJepaConfig::default().patch_diff_quality() - DEFAULT_PATCH_DIFF_QUALITY).abs()
            <= f32::EPSILON
    );
    assert_eq!(BevyJepaConfig::default().bootstrap_context_density, 1.0);
    assert_eq!(
        BevyJepaConfig::default().pca_update_every,
        DEFAULT_PCA_UPDATE_EVERY
    );
    assert_eq!(
        BevyJepaConfig::default().pca_sample_window_frames,
        DEFAULT_PCA_SAMPLE_WINDOW_FRAMES
    );
    assert_eq!(
        BevyJepaConfig::default().pca_min_sample_frames,
        DEFAULT_PCA_MIN_SAMPLE_FRAMES
    );
    assert_eq!(
        BevyJepaConfig::default().pca_update_iterations,
        DEFAULT_PCA_UPDATE_ITERATIONS
    );
    assert_eq!(
        BevyJepaConfig::default().sparse_encode_mode,
        BevyJepaSparseEncodeMode::BucketedContext
    );
    assert!(BevyJepaConfig::default().prewarm_shape_buckets);
    assert_eq!(
        BevyJepaConfig::default().high_res_pca_every,
        DEFAULT_HIGH_RES_PCA_EVERY
    );
    assert_eq!(
        stage_request_for_frame(&BevyJepaConfig::default(), 0),
        FeatureFrameRequest::low_res()
    );
    assert_eq!(
        stage_request_for_frame(&BevyJepaConfig::default(), 1),
        FeatureFrameRequest::low_res()
    );
    let high_res_config = BevyJepaConfig {
        pipeline: FeatureFrameViewerConfig {
            high_res_pca_every: 8,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    assert_eq!(
        stage_request_for_frame(&high_res_config, 0),
        FeatureFrameRequest::low_res()
    );
    assert!(high_res_scheduled_for_frame(&high_res_config, 0));
    assert_eq!(
        stage_request_for_frame(&high_res_config, 1),
        FeatureFrameRequest::low_res()
    );
    assert!(!high_res_scheduled_for_frame(&high_res_config, 1));
    assert_eq!(
        stage_request_for_frame(&high_res_config, 8),
        FeatureFrameRequest::low_res()
    );
    assert!(high_res_scheduled_for_frame(&high_res_config, 8));
    assert_eq!(
        BevyJepaMaskSource::PatchDiff.next(),
        BevyJepaMaskSource::PatchDiff
    );
}

#[test]
fn anyup_panel_visibility_follows_high_res_cadence() {
    let mut config = BevyJepaConfig {
        pipeline: FeatureFrameViewerConfig {
            high_res_pca_every: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(!high_res_panel_enabled(&config));
    assert_eq!(visible_panel_count(&config), 3);

    config.high_res_pca_every = 8;
    assert!(high_res_panel_enabled(&config));
    assert_eq!(visible_panel_count(&config), 4);
}

#[test]
fn control_actions_switch_model_profiles_and_resolution() {
    let mut config = BevyJepaConfig::default();

    assert_eq!(
        apply_control_action(&mut config, JepaControlAction::ModelBase),
        JepaControlReset::Rebuild
    );
    assert_eq!(config.encoder_source, BevyJepaEncoderSource::BaseCheckpoint);
    assert_eq!(
        config.model_profile,
        BevyJepaModelPackageProfile::Vjepa21Base
    );
    assert!(config.model_manifest_path.is_none());
    assert!(config.ttt_model_path.is_none());
    assert!(config.model_base_url.ends_with("/vjepa2_1_base"));

    assert_eq!(
        apply_control_action(&mut config, JepaControlAction::ModelTtt),
        JepaControlReset::Rebuild
    );
    assert_eq!(config.encoder_source, BevyJepaEncoderSource::TrainedTtt);
    assert_eq!(
        config.model_profile,
        BevyJepaModelPackageProfile::Vjepa21Ttt
    );
    assert!(config.model_base_url.ends_with("/vjepa2_1_ttt"));

    assert_eq!(
        apply_control_action(&mut config, JepaControlAction::Resolution256),
        JepaControlReset::Rebuild
    );
    assert_eq!(config.pipeline_image_size(), 256);
    assert_eq!(
        apply_control_action(&mut config, JepaControlAction::Resolution512),
        JepaControlReset::Rebuild
    );
    assert_eq!(config.pipeline_image_size(), 512);
}

#[test]
fn control_actions_update_patch_diff_refresh_without_model_rebuild() {
    let mut config = BevyJepaConfig::default();
    let threshold = config.patch_diff_threshold;
    let subthreshold = config.patch_diff_refresh.subthreshold_enabled;
    let age = config.patch_diff_refresh.age_refresh_enabled;
    let blue = config.patch_diff_refresh.blue_noise_enabled;

    assert_eq!(
        apply_control_slider_value(&mut config, JepaControlSliderKind::PatchDiffThreshold, 1.0),
        JepaControlReset::Visual
    );
    assert!(config.patch_diff_threshold > threshold);
    assert_eq!(
        apply_control_slider_value(&mut config, JepaControlSliderKind::PatchDiffThreshold, 0.5),
        JepaControlReset::Visual
    );

    assert_eq!(
        apply_control_action(&mut config, JepaControlAction::SubthresholdRefresh),
        JepaControlReset::Visual
    );
    assert_eq!(
        config.patch_diff_refresh.subthreshold_enabled,
        !subthreshold
    );
    assert_eq!(
        apply_control_action(&mut config, JepaControlAction::AgeRefresh),
        JepaControlReset::Visual
    );
    assert_eq!(config.patch_diff_refresh.age_refresh_enabled, !age);
    assert_eq!(
        apply_control_action(&mut config, JepaControlAction::BlueNoiseRefresh),
        JepaControlReset::Visual
    );
    assert_eq!(config.patch_diff_refresh.blue_noise_enabled, !blue);
}

#[test]
fn control_actions_toggle_anyup_panel_cadence() {
    let mut config = BevyJepaConfig::default();

    assert_eq!(
        apply_control_action(&mut config, JepaControlAction::AnyUpEvery8),
        JepaControlReset::Rebuild
    );
    assert_eq!(config.high_res_pca_every, 8);
    assert_eq!(visible_panel_count(&config), 4);

    assert_eq!(
        apply_control_action(&mut config, JepaControlAction::AnyUpOff),
        JepaControlReset::Rebuild
    );
    assert_eq!(config.high_res_pca_every, 0);
    assert_eq!(visible_panel_count(&config), 3);
}

#[test]
fn control_actions_toggle_pca_basis_training_without_rebuild() {
    let mut config = BevyJepaConfig::default();

    assert!(config.pca_update_config().enabled());
    assert_eq!(
        apply_control_action(&mut config, JepaControlAction::PcaLock),
        JepaControlReset::PcaConfig
    );
    assert_eq!(config.pca_update_every, 0);
    assert!(!config.pca_update_config().enabled());
    assert!(control_button_active(
        &config,
        JepaControlAction::PcaLock,
        &JepaControlsState::default()
    ));

    assert_eq!(
        apply_control_action(&mut config, JepaControlAction::PcaTrain),
        JepaControlReset::PcaConfig
    );
    assert_eq!(config.pca_update_every, DEFAULT_PCA_UPDATE_EVERY.max(1));
    assert!(config.pca_update_config().enabled());
}

#[test]
fn pca_lock_applies_to_pipeline_returning_from_active_task() {
    let device = JepaBevyDevice::default();
    let mut config = tiny_viewer_config();
    let image_size = config.pipeline_image_size();
    let active_pca_update = config.pca_update_config();
    let model_config = tiny_viewer_model_config(image_size);
    let jepa = VJepa2_1Model::<JepaBevyBackend>::new(&model_config, &device);
    let mut anyup_config = AnyUpConfig::tiny_for_tests();
    anyup_config.input_dim = 3;
    let anyup = AnyUp::<JepaBevyBackend>::new(anyup_config, &device).expect("AnyUp");
    let pipeline = FeatureFramePipeline::<JepaBevyBackend>::new(
        jepa,
        anyup,
        &model_config,
        FeatureFramePipelineConfig {
            pca_update: config.pca_update_config(),
            ..FeatureFramePipelineConfig::default()
        },
        1,
        [image_size, image_size],
        &device,
    )
    .expect("pipeline");
    assert!(pipeline.config().pca_update.enabled());
    assert!(pipeline.pca_update_scheduler().config().enabled());

    let signature = RuntimePipelineSignature::new(&config, image_size);
    let mut runtime = JepaRuntime {
        pipeline_signature: Some(signature.clone()),
        ..JepaRuntime::default()
    };

    config.pca_update_every = 0;
    runtime
        .apply_pca_update_config(&config)
        .expect("missing active pipeline should defer cleanly");

    let completed = finish_jepa_task_output(
        &config,
        &mut runtime,
        &device,
        JepaAsyncTaskOutput {
            signature,
            pca_update: active_pca_update,
            pipeline,
            result: Err("sentinel".to_string()),
        },
    );
    assert!(
        completed.is_none(),
        "stale PCA completion should be suppressed instead of repainting old colors"
    );

    let returned = runtime.pipeline.as_ref().expect("returned pipeline");
    assert!(!returned.config().pca_update.enabled());
    assert!(!returned.pca_update_scheduler().config().enabled());
    assert_eq!(runtime.stale_completions, 1);
}

#[test]
fn control_tabs_are_view_state_not_pipeline_rebuilds() {
    let config = BevyJepaConfig::default();
    let mut controls = JepaControlsState::default();

    assert_eq!(controls.tab, JepaControlsTab::Pipeline);
    assert_eq!(
        control_tab_for_action(JepaControlAction::TabMask),
        Some(JepaControlsTab::Mask)
    );
    let mut tab_config = BevyJepaConfig::default();
    assert_eq!(
        apply_control_action(&mut tab_config, JepaControlAction::TabMask),
        JepaControlReset::None
    );
    assert!(control_button_active(
        &config,
        JepaControlAction::TabPipeline,
        &controls
    ));
    assert!(!control_button_active(
        &config,
        JepaControlAction::TabMask,
        &controls
    ));

    controls.tab = JepaControlsTab::Mask;
    assert!(control_button_active(
        &config,
        JepaControlAction::TabMask,
        &controls
    ));
    assert_eq!(
        control_button_label(&config, JepaControlAction::TabPca),
        "PCA"
    );
    assert!(control_help_text(JepaControlAction::TabAnyUp).contains("AnyUp"));
}

#[test]
fn control_actions_switch_anyup_attention_mode() {
    let mut config = BevyJepaConfig::default();

    assert_eq!(
        apply_control_action(&mut config, JepaControlAction::AnyUpUpstreamMasked),
        JepaControlReset::Rebuild
    );
    assert_eq!(
        config.anyup_attention_mode,
        burn_jepa::AnyUpAttentionMode::UpstreamMasked
    );
    assert!(control_button_active(
        &config,
        JepaControlAction::AnyUpUpstreamMasked,
        &JepaControlsState::default()
    ));

    assert_eq!(
        apply_control_action(&mut config, JepaControlAction::AnyUpEfficientLocal),
        JepaControlReset::Rebuild
    );
    assert_eq!(
        config.anyup_attention_mode,
        burn_jepa::AnyUpAttentionMode::EfficientLocal
    );
}

#[test]
fn control_actions_switch_dense_and_sparse_pipeline_presets() {
    let mut config = BevyJepaConfig::default();

    assert_eq!(
        apply_control_action(&mut config, JepaControlAction::PipelineDense),
        JepaControlReset::Rebuild
    );
    assert!(dense_pipeline_enabled(&config));
    assert_eq!(config.encode_path, BevyJepaEncodePath::DensePatchEmbed);
    assert_eq!(config.context_density, 1.0);
    assert_eq!(config.min_context_density, 1.0);
    assert_eq!(config.bootstrap_context_density, 1.0);
    assert_eq!(config.patch_diff_threshold, 0.0);

    assert_eq!(
        apply_control_action(&mut config, JepaControlAction::PipelineSparse),
        JepaControlReset::Rebuild
    );
    assert!(!dense_pipeline_enabled(&config));
    assert_eq!(config.encode_path, BevyJepaEncodePath::Auto);
    assert_eq!(
        config.sparse_encode_mode,
        BevyJepaSparseEncodeMode::BucketedContext
    );
}

#[test]
fn control_actions_preserve_pca_settings_across_pipeline_changes() {
    let mut config = BevyJepaConfig::default();
    config.pca_update_every = 23;
    config.pca_sample_window_frames = 41;
    config.pca_min_sample_frames = 7;
    config.pca_update_iterations = 5;
    let pca = PcaControlSettings::capture(&config);

    for action in [
        JepaControlAction::TogglePanel,
        JepaControlAction::TabPipeline,
        JepaControlAction::TabMask,
        JepaControlAction::TabPca,
        JepaControlAction::TabAnyUp,
        JepaControlAction::ModelBase,
        JepaControlAction::ModelTtt,
        JepaControlAction::PipelineDense,
        JepaControlAction::PipelineSparse,
        JepaControlAction::Resolution256,
        JepaControlAction::Resolution512,
        JepaControlAction::AnyUpOff,
        JepaControlAction::AnyUpEvery8,
        JepaControlAction::AnyUpEvery1,
        JepaControlAction::AnyUpEfficientLocal,
        JepaControlAction::AnyUpUpstreamMasked,
        JepaControlAction::PatchRefresh,
        JepaControlAction::SubthresholdRefresh,
        JepaControlAction::AgeRefresh,
        JepaControlAction::BlueNoiseRefresh,
    ] {
        apply_control_action(&mut config, action);
        assert_eq!(
            PcaControlSettings::capture(&config),
            pca,
            "{action:?} should not reset PCA settings"
        );
    }
}

#[test]
fn control_sliders_preserve_pca_settings_for_non_pca_fields() {
    let mut config = BevyJepaConfig::default();
    config.pca_update_every = 17;
    config.pca_sample_window_frames = 37;
    config.pca_min_sample_frames = 11;
    config.pca_update_iterations = 6;
    let pca = PcaControlSettings::capture(&config);

    for kind in [
        JepaControlSliderKind::PatchDiffThreshold,
        JepaControlSliderKind::ContextDensity,
        JepaControlSliderKind::MinContextDensity,
        JepaControlSliderKind::DenseFallbackDensity,
        JepaControlSliderKind::SubthresholdTrigger,
        JepaControlSliderKind::AgeIntervalFrames,
        JepaControlSliderKind::BlueNoiseDensity,
    ] {
        apply_control_slider_value(&mut config, kind, 0.75);
        assert_eq!(
            PcaControlSettings::capture(&config),
            pca,
            "{kind:?} should not reset PCA settings"
        );
    }
}

#[test]
fn control_sliders_update_numeric_pipeline_fields() {
    let mut config = BevyJepaConfig::default();

    assert_eq!(
        apply_control_slider_value(&mut config, JepaControlSliderKind::PatchDiffThreshold, 0.5),
        JepaControlReset::Visual
    );
    assert!((config.patch_diff_threshold - 0.10).abs() <= 1.0e-6);

    apply_control_slider_value(&mut config, JepaControlSliderKind::MinContextDensity, 0.25);
    assert!((config.min_context_density - 0.25).abs() <= 1.0e-6);
    apply_control_slider_value(&mut config, JepaControlSliderKind::ContextDensity, 0.0);
    assert!(config.context_density >= config.min_context_density);

    apply_control_slider_value(&mut config, JepaControlSliderKind::AgeIntervalFrames, 1.0);
    assert_eq!(config.patch_diff_refresh.age_refresh_interval_frames, 300);

    assert_eq!(
        apply_control_slider_value(&mut config, JepaControlSliderKind::PcaUpdateEvery, 0.0),
        JepaControlReset::PcaConfig
    );
    assert_eq!(config.pca_update_every, 0);
    apply_control_slider_value(
        &mut config,
        JepaControlSliderKind::PcaSampleWindowFrames,
        0.0,
    );
    assert_eq!(config.pca_sample_window_frames, 2);
    apply_control_slider_value(&mut config, JepaControlSliderKind::PcaMinSampleFrames, 1.0);
    assert!(config.pca_min_sample_frames <= config.pca_sample_window_frames);
    apply_control_slider_value(&mut config, JepaControlSliderKind::PcaUpdateIterations, 1.0);
    assert_eq!(config.pca_update_iterations, 12);
}

#[test]
fn control_slider_relative_coordinates_map_to_unit_range() {
    assert_eq!(slider_normalized_from_relative_x(-0.75), 0.0);
    assert_eq!(slider_normalized_from_relative_x(-0.5), 0.0);
    assert!((slider_normalized_from_relative_x(0.0) - 0.5).abs() <= f32::EPSILON);
    assert_eq!(slider_normalized_from_relative_x(0.5), 1.0);
    assert_eq!(slider_normalized_from_relative_x(0.75), 1.0);
}

#[test]
fn control_slider_ignores_tiny_drag_jitter() {
    let mut config = BevyJepaConfig::default();
    let current = slider_normalized_value(&config, JepaControlSliderKind::PatchDiffThreshold);

    assert_eq!(
        apply_control_slider_value_if_changed(
            &mut config,
            JepaControlSliderKind::PatchDiffThreshold,
            current + CONTROL_SLIDER_UPDATE_EPSILON * 0.5,
        ),
        JepaControlReset::None
    );
}

#[test]
fn controls_ui_system_has_disjoint_text_queries() {
    let mut app = App::new();
    app.insert_resource(BevyJepaConfig::default())
        .init_resource::<JepaControlsState>()
        .add_systems(Update, update_controls_ui);

    app.update();
}

#[test]
fn controls_ui_shows_only_selected_tab_panel() {
    let mut app = App::new();
    app.insert_resource(BevyJepaConfig::default())
        .insert_resource(JepaControlsState {
            expanded: true,
            tab: JepaControlsTab::Mask,
        })
        .add_systems(Startup, setup_controls_ui)
        .add_systems(Update, update_controls_ui);

    app.update();
    assert_visible_controls_tab(&mut app, JepaControlsTab::Mask);

    app.world_mut().resource_mut::<JepaControlsState>().tab = JepaControlsTab::Pca;
    app.update();
    assert_visible_controls_tab(&mut app, JepaControlsTab::Pca);
}

fn assert_visible_controls_tab(app: &mut App, active: JepaControlsTab) {
    let mut query = app.world_mut().query::<(&ControlsTabPanel, &Node)>();
    let mut count = 0;
    for (panel, node) in query.iter(app.world()) {
        count += 1;
        let expected = if panel.tab == active {
            Display::Flex
        } else {
            Display::None
        };
        assert_eq!(node.display, expected, "tab {:?}", panel.tab);
    }
    assert_eq!(count, 4);
}

#[test]
fn trained_ttt_missing_explicit_package_reports_bpk_path() {
    let device = JepaBevyDevice::default();
    let temp = tempfile::tempdir().expect("tempdir");
    let missing_manifest = temp.path().join("missing-manifest.json");
    let config = BevyJepaConfig {
        encoder_source: BevyJepaEncoderSource::TrainedTtt,
        model_manifest_path: Some(missing_manifest.clone()),
        model_auto_download: false,
        ttt_model_path: None,
        jepa_checkpoint_dir: None,
        jepa_config_path: None,
        ..BevyJepaConfig::default()
    };

    let err = match load_viewer_encoder(&config, 32, &device) {
        Ok(_) => panic!("trained TTT should require a package or explicit checkpoint"),
        Err(err) => err,
    };
    let message = err.to_string();
    assert!(message.contains("burn_jepa package manifest"));
    assert!(message.contains(&missing_manifest.display().to_string()));
    assert!(message.contains("--model-manifest"));
    assert!(!message.contains(DEFAULT_TTT_MODEL_PATH));
}

#[test]
fn trained_ttt_encoder_loads_from_sharded_bpk_manifest() {
    let device = JepaBevyDevice::default();
    let image_size = 32;
    let model_config = tiny_viewer_model_config(image_size);
    let ttt_config = TttEncoderConfig::default();
    let base = VJepa2_1Model::<JepaBevyBackend>::new(&model_config, &device);
    let ttt = VJepaTttModel::from_model(base, ttt_config.clone(), &device).expect("tiny TTT model");
    let temp = tempfile::tempdir().expect("tempdir");
    let burnpack = temp.path().join("jepa_ttt.bpk");
    burn_jepa::save_ttt_burnpack(&ttt, &burnpack).expect("save TTT bpk");
    let parts = write_burnpack_parts_for_browser(&burnpack, 1024, true).expect("write sharded bpk");
    assert!(
        parts.part_paths.len() > 1,
        "small shard size should exercise multi-part package loading"
    );
    let manifest = BurnJepaPipelinePackageManifest {
        model_kind: BurnJepaPackageModelKind::Ttt,
        record_dtype: Some("f16".to_string()),
        jepa_config: model_config.clone(),
        ttt_config: Some(ttt_config),
        ..BurnJepaPipelinePackageManifest::default()
    }
    .with_burnpack_paths(&burnpack);
    let manifest_path = temp.path().join("manifest.json");
    write_pipeline_package_manifest(&manifest_path, &manifest).expect("write package manifest");
    let config = BevyJepaConfig {
        encoder_source: BevyJepaEncoderSource::TrainedTtt,
        model_manifest_path: Some(manifest_path),
        ttt_model_path: None,
        jepa_checkpoint_dir: None,
        jepa_config_path: None,
        pipeline: FeatureFrameViewerConfig {
            image_size,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };

    let (encoder, loaded_config) =
        load_viewer_encoder(&config, image_size, &device).expect("load package encoder");
    assert_eq!(encoder.kind(), FeatureFrameJepaEncoderKind::Ttt);
    assert_eq!(loaded_config.image_size, image_size);
}

#[test]
fn default_viewer_pca_uses_smooth_rolling_updates() {
    let config = BevyJepaConfig::default();
    let pca_update = config.pca_update_config();

    assert_eq!(pca_update.every_n_frames, 1);
    assert_eq!(pca_update.sample_window_frames, 16);
    assert_eq!(pca_update.min_sample_frames, 2);
    assert_eq!(
        pca_update.iterations_per_update,
        DEFAULT_PCA_UPDATE_ITERATIONS
    );

    let mut locked = config;
    locked.pca_update_every = 0;
    assert!(!locked.pca_update_config().enabled());
}

#[test]
fn default_headless_low_res_pca_updates_after_short_warmup() {
    let device = JepaBevyDevice::default();
    let mut pipeline = BevyJepaHeadlessPipeline::new(
        BevyJepaConfig {
            source: BevyJepaFrameSource::SyntheticLocalMotion,
            ..tiny_viewer_config()
        },
        device,
    );

    let first = pipeline
        .step_with_stage_request(FeatureFrameRequest::low_res())
        .expect("first low-res stage");
    assert!(!first.metrics.pca_update_applied);
    assert_eq!(first.metrics.pca_sample_frames, 1);

    let second = pipeline
        .step_with_stage_request(FeatureFrameRequest::low_res())
        .expect("second low-res stage");
    assert!(second.metrics.pca_update_applied);
    assert_eq!(second.metrics.pca_sample_frames, 2);

    let third = pipeline
        .step_with_stage_request(FeatureFrameRequest::low_res())
        .expect("third low-res stage");
    assert!(third.metrics.pca_update_applied);
    assert_eq!(third.metrics.pca_sample_frames, 3);

    let mut later_metrics = third.metrics;
    for _ in 3..18 {
        later_metrics = pipeline
            .step_with_stage_request(FeatureFrameRequest::low_res())
            .expect("later low-res stage")
            .metrics;
    }
    assert!(later_metrics.pca_update_applied);
    assert_eq!(later_metrics.pca_sample_frames, 16);
}

#[test]
fn camera_base_dense_default_keeps_anyup_off_hot_path() {
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::Camera,
        encoder_source: BevyJepaEncoderSource::BaseCheckpoint,
        mask_source: BevyJepaMaskSource::PatchDiff,
        pipeline: FeatureFrameViewerConfig {
            encode_path: BevyJepaEncodePath::DensePatchEmbed,
            patch_diff_threshold: 0.0,
            context_density: 1.0,
            min_context_density: 1.0,
            bootstrap_context_density: 1.0,
            image_size: 256,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    assert_eq!(config.high_res_pca_every, 0);
    assert_eq!(
        stage_request_for_frame(&config, 0),
        FeatureFrameRequest::low_res()
    );
    assert_eq!(
        stage_request_for_frame(&config, 128),
        FeatureFrameRequest::low_res()
    );
}

#[test]
fn anyup_weights_auto_discovery_uses_default_checkpoint_for_real_encoder_only() {
    let default_path = resolve_repo_relative_path(DEFAULT_ANYUP_CHECKPOINT_PATH);
    if default_path.exists() {
        assert_eq!(
            effective_anyup_weights(&BevyJepaConfig::default()),
            Some(default_path)
        );
    }
    assert_eq!(effective_anyup_weights(&tiny_viewer_config()), None);
}

#[test]
fn viewer_pipeline_promotes_small_image_requests_to_minimum_resolution() {
    let config = BevyJepaConfig {
        pipeline: FeatureFrameViewerConfig {
            image_size: 64,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    assert_eq!(config.pipeline_image_size(), MIN_PIPELINE_IMAGE_SIZE);
}

#[test]
fn viewer_pipeline_accepts_trained_256_resolution() {
    let config = BevyJepaConfig {
        pipeline: FeatureFrameViewerConfig {
            image_size: MIN_PIPELINE_IMAGE_SIZE,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    assert_eq!(config.pipeline_image_size(), 256);
}

#[test]
fn default_patch_diff_quality_is_threshold_not_static_sparsity() {
    let config = BevyJepaConfig::default();
    assert_eq!(config.min_context_tokens(256), 1);
    assert_eq!(config.min_context_tokens(1024), 1);
    assert!((config.patch_diff_threshold - 0.03).abs() <= f32::EPSILON);
    assert!((config.patch_diff_quality() - 0.97).abs() <= f32::EPSILON);
    assert!(
        (config.patch_diff_dense_fallback_density - DEFAULT_PATCH_DIFF_DENSE_FALLBACK_DENSITY)
            .abs()
            <= f32::EPSILON
    );
}

#[test]
fn default_patch_diff_static_frame_keeps_dynamic_minimum_only() {
    let device = JepaBevyDevice::default();
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::Camera,
        mask_source: BevyJepaMaskSource::PatchDiff,
        ..BevyJepaConfig::default()
    };
    let mut model_config = VJepaConfig::tiny_for_tests();
    model_config.image_size = 64;
    model_config.num_frames = 2;
    model_config.tubelet_size = 2;
    model_config.patch_size = 16;
    let grid = TokenGridShape::new(1, 4, 4);
    let previous_rgba = RgbaImage::new(64, 64);
    let current_rgba = previous_rgba.clone();
    let previous = rgba_image_to_tensor(previous_rgba.clone(), 64, &device).expect("prev");
    let current = rgba_image_to_tensor(current_rgba.clone(), 64, &device).expect("current");

    let output = run_sparse_mask_node(
        &config,
        Some(&previous),
        Some(&previous_rgba),
        Some(&current_rgba),
        &current,
        &model_config,
        grid,
    )
    .expect("static patch-diff mask");

    assert_eq!(output.write_mask.dense_len(), grid.len());
    assert_eq!(output.write_mask.len(), 1);
    assert!(
        output.write_mask.len() < (grid.len() as f32 * DEFAULT_PATCH_DIFF_QUALITY).round() as usize,
        "patch-diff quality must not be interpreted as a fixed sparsity floor"
    );
}

#[test]
fn default_patch_diff_quality_keeps_subtle_motion_patches() {
    let device = JepaBevyDevice::default();
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::Camera,
        mask_source: BevyJepaMaskSource::PatchDiff,
        pipeline: FeatureFrameViewerConfig {
            context_density: 1.0,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    let mut model_config = VJepaConfig::tiny_for_tests();
    model_config.image_size = 64;
    model_config.num_frames = 2;
    model_config.tubelet_size = 2;
    model_config.patch_size = 16;
    let grid = TokenGridShape::new(1, 4, 4);
    let changed = [(0, 0), (2, 3), (3, 1)];
    let previous_rgba = RgbaImage::new(64, 64);
    let current_rgba = rgba_with_patches(64, 64, &changed, 16, image::Rgba([20, 20, 20, 255]));
    let previous = rgba_image_to_tensor(previous_rgba.clone(), 64, &device).expect("prev");
    let current = rgba_image_to_tensor(current_rgba.clone(), 64, &device).expect("current");

    let output = run_sparse_mask_node(
        &config,
        Some(&previous),
        Some(&previous_rgba),
        Some(&current_rgba),
        &current,
        &model_config,
        grid,
    )
    .expect("patch-diff mask");

    assert_eq!(output.write_mask.len(), changed.len());
    assert_eq!(
        output.write_mask.indices(),
        &[
            coords_to_token_index(0, 0, 0, grid),
            coords_to_token_index(0, 2, 3, grid),
            coords_to_token_index(0, 3, 1, grid),
        ]
    );
}

#[test]
fn patch_diff_global_lighting_shift_does_not_fill_the_frame() {
    let device = JepaBevyDevice::default();
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::Camera,
        mask_source: BevyJepaMaskSource::PatchDiff,
        pipeline: FeatureFrameViewerConfig {
            context_density: 1.0,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    let mut model_config = VJepaConfig::tiny_for_tests();
    model_config.image_size = 64;
    model_config.num_frames = 2;
    model_config.tubelet_size = 2;
    model_config.patch_size = 16;
    let grid = TokenGridShape::new(1, 4, 4);
    let previous_rgba = RgbaImage::from_pixel(64, 64, image::Rgba([80, 80, 80, 255]));
    let current_rgba = RgbaImage::from_pixel(64, 64, image::Rgba([110, 110, 110, 255]));
    let previous = rgba_image_to_tensor(previous_rgba.clone(), 64, &device).expect("prev");
    let current = rgba_image_to_tensor(current_rgba.clone(), 64, &device).expect("current");

    let output = run_sparse_mask_node(
        &config,
        Some(&previous),
        Some(&previous_rgba),
        Some(&current_rgba),
        &current,
        &model_config,
        grid,
    )
    .expect("patch-diff mask");

    assert_eq!(
        output.write_mask.len(),
        1,
        "uniform exposure shifts should not be treated as full-frame motion"
    );
}

#[test]
fn patch_diff_relative_luma_detects_dark_region_motion() {
    let device = JepaBevyDevice::default();
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::Camera,
        mask_source: BevyJepaMaskSource::PatchDiff,
        pipeline: FeatureFrameViewerConfig {
            context_density: 1.0,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    let mut model_config = VJepaConfig::tiny_for_tests();
    model_config.image_size = 64;
    model_config.num_frames = 2;
    model_config.tubelet_size = 2;
    model_config.patch_size = 16;
    let grid = TokenGridShape::new(1, 4, 4);
    let previous_rgba = RgbaImage::from_pixel(64, 64, image::Rgba([20, 20, 20, 255]));
    let current_rgba = rgba_with_base_and_patches(
        64,
        64,
        image::Rgba([20, 20, 20, 255]),
        &[(1, 2)],
        16,
        image::Rgba([30, 30, 30, 255]),
    );
    let previous = rgba_image_to_tensor(previous_rgba.clone(), 64, &device).expect("prev");
    let current = rgba_image_to_tensor(current_rgba.clone(), 64, &device).expect("current");

    let output = run_sparse_mask_node(
        &config,
        Some(&previous),
        Some(&previous_rgba),
        Some(&current_rgba),
        &current,
        &model_config,
        grid,
    )
    .expect("patch-diff mask");

    assert_eq!(output.write_mask.len(), 1);
    assert_eq!(
        output.write_mask.indices(),
        &[coords_to_token_index(0, 1, 2, grid)]
    );
}

#[test]
fn low_res_feature_fallback_preserves_nchw_spatial_grid() {
    let device = JepaBevyDevice::default();
    let features = Tensor::<JepaBevyBackend, 4>::from_data(
        TensorData::new(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                10.0, 20.0, 30.0, 40.0, //
                -1.0, -2.0, -3.0, -4.0, //
                99.0, 98.0, 97.0, 96.0,
            ],
            [1, 4, 2, 2],
        ),
        &device,
    );

    let display = low_res_pca_or_features(LowResFrameArtifacts {
        features,
        pca_display: None,
    })
    .expect("low-res display fallback");

    assert_eq!(display.shape().dims::<4>(), [1, 3, 2, 2]);
    assert_eq!(
        values4(display),
        vec![
            1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, -1.0, -2.0, -3.0, -4.0
        ]
    );
}

#[test]
fn low_res_display_resize_preserves_patch_grid_colors() {
    let device = JepaBevyDevice::default();
    let low_res = Tensor::<JepaBevyBackend, 4>::from_data(
        TensorData::new(
            vec![
                1.0, 0.0, 0.0, 1.0, //
                0.0, 1.0, 0.0, 1.0, //
                0.0, 0.0, 1.0, 1.0,
            ],
            [1, 3, 2, 2],
        ),
        &device,
    );

    let resized = resize_nchw(low_res, [32, 32]);
    let rgba =
        tensor_rgba_to_host(nchw_to_rgba_tensor(resized).expect("rgba tensor")).expect("host rgba");
    let sample = |x: usize, y: usize| {
        let offset = (y * 32 + x) * 4;
        [rgba[offset], rgba[offset + 1], rgba[offset + 2]]
    };

    assert_eq!(sample(4, 4), [255, 0, 0]);
    assert_eq!(sample(28, 4), [0, 255, 0]);
    assert_eq!(sample(4, 28), [0, 0, 255]);
    assert_eq!(sample(28, 28), [255, 255, 255]);
}

#[test]
fn low_res_pca_panel_uses_multiple_rgb_colors_after_default_warmup() {
    let device = JepaBevyDevice::default();
    let image_size = 32;
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::SyntheticLocalMotion,
        display_transfer: BevyJepaDisplayTransfer::Cpu,
        ..tiny_viewer_config()
    };
    let model_config = tiny_viewer_model_config(image_size);
    let jepa = VJepa2_1Model::<JepaBevyBackend>::new(&model_config, &device);
    let mut anyup_config = AnyUpConfig::tiny_for_tests();
    anyup_config.input_dim = 3;
    let anyup = AnyUp::<JepaBevyBackend>::new(anyup_config, &device).expect("AnyUp");
    let mut pipeline = FeatureFramePipeline::<JepaBevyBackend>::new(
        jepa,
        anyup,
        &model_config,
        FeatureFramePipelineConfig {
            pca_update: config.pca_update_config(),
            measurement: burn_jepa::FeatureFrameMeasureConfig::enabled(),
            ..FeatureFramePipelineConfig::default()
        },
        1,
        [image_size, image_size],
        &device,
    )
    .expect("pipeline");
    let grid = pipeline.grid();
    let mask = SparseTokenMask::all(grid.len());

    for sequence in 0..2 {
        let processed = run_stage_pipeline_step(
            &config,
            &mut pipeline,
            synthetic_image_tensor(sequence, image_size, &device),
            &mask,
            &mask,
            FrameId {
                stream_id: 0,
                sequence,
                capture_time_nanos: sequence.saturating_mul(16_666_667),
            },
            grid,
            model_config.patch_size,
            BevyJepaFrameSource::SyntheticLocalMotion,
            false,
            FeatureFrameRequest::low_res(),
        )
        .expect("low-res stage");

        if sequence == 1 {
            assert!(processed.metrics.pca_update_applied);
            let StagePanelData::Host {
                low_res_rgba,
                width,
                height,
                ..
            } = processed.panels
            else {
                panic!("CPU display transfer should expose host panel bytes");
            };
            let colors = panel_patch_center_colors(
                &low_res_rgba,
                width as usize,
                height as usize,
                grid.height,
                grid.width,
            );
            assert!(
                unique_rgb_color_count(&colors, 12) >= 3,
                "low-res PCA panel should not collapse to a mono-color grid, got {colors:?}"
            );
        }
    }
}

#[test]
fn viewer_pipeline_rounds_image_requests_to_patch_multiple() {
    let config = BevyJepaConfig {
        pipeline: FeatureFrameViewerConfig {
            image_size: MIN_PIPELINE_IMAGE_SIZE + 1,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    assert_eq!(
        config.pipeline_image_size() % PIPELINE_IMAGE_SIZE_MULTIPLE,
        0
    );
    assert!(config.pipeline_image_size() > MIN_PIPELINE_IMAGE_SIZE);
}

#[test]
fn camera_source_waits_without_generating_synthetic_warmup() {
    let device = JepaBevyDevice::default();
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::Camera,
        ..BevyJepaConfig::default()
    };
    let mut pipeline = BevyJepaHeadlessPipeline::new(config, device);
    let err = pipeline
        .step_stage_only()
        .expect_err("camera source should wait for a real frame");
    assert!(err.to_string().contains("camera frame is not ready"));
}

#[test]
fn camera_source_without_frame_does_not_initialize_pipeline() {
    let device = JepaBevyDevice::default();
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::Camera,
        ..BevyJepaConfig::default()
    };
    let mut runtime = JepaRuntime::default();
    let processed =
        process_runtime_frame(&config, &mut runtime, &device, BevyJepaStepMode::StageOnly)
            .expect("camera wait should not be an error inside the Bevy schedule");
    assert!(processed.is_none());
    assert_eq!(runtime.frame_index, 0);
    assert!(runtime.pipeline.is_none());
    assert!(runtime.prev_image.is_none());
}

#[test]
fn source_node_keeps_latest_pending_stage_frame_while_worker_runs() {
    let device = JepaBevyDevice::default();
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::SyntheticLocalMotion,
        pipeline: FeatureFrameViewerConfig {
            high_res_pca_every: 8,
            ..FeatureFrameViewerConfig::default()
        },
        ..tiny_viewer_config()
    };
    let mut runtime = JepaRuntime::default();

    let first = process_runtime_source_frame(&config, &mut runtime, &device)
        .expect("first source frame")
        .expect("synthetic source");
    assert_eq!(first.source, BevyJepaFrameSource::SyntheticLocalMotion);
    assert_eq!(first.sequence, 0);
    assert!(runtime.active_task.is_some());
    assert!(runtime.pending_stage.is_none());
    assert!(runtime.prev_stage_image.is_some());
    assert_eq!(runtime.input_frames_seen, 1);

    process_runtime_source_frame(&config, &mut runtime, &device)
        .expect("second source frame")
        .expect("synthetic source");
    assert!(runtime.active_task.is_some());
    assert_eq!(
        runtime
            .pending_stage
            .as_ref()
            .map(|pending| pending.id.sequence),
        Some(1)
    );
    assert_eq!(runtime.dropped_frames, 0);
    assert_eq!(runtime.overwritten_frames, 0);

    process_runtime_source_frame(&config, &mut runtime, &device)
        .expect("third source frame")
        .expect("synthetic source");
    assert_eq!(
        runtime
            .pending_stage
            .as_ref()
            .map(|pending| pending.id.sequence),
        Some(2)
    );
    assert_eq!(runtime.input_frames_seen, 3);
    assert_eq!(runtime.dropped_frames, 1);
    assert_eq!(runtime.overwritten_frames, 1);

    let mut metrics = BevyJepaMetrics::default();
    runtime.apply_runtime_counts(&mut metrics);
    assert_eq!(metrics.in_flight_frames, 2);
    assert_eq!(metrics.queue_dropped_frames, 1);
    assert_eq!(metrics.queue_overwritten_frames, 1);
    assert_eq!(metrics.input_frame_index, 2);
}

#[test]
fn autogaze_mask_source_requires_real_model_node() {
    let device = JepaBevyDevice::default();
    let mut pipeline = BevyJepaHeadlessPipeline::new(
        BevyJepaConfig {
            source: BevyJepaFrameSource::SyntheticLocalMotion,
            mask_source: BevyJepaMaskSource::Autogaze,
            ..tiny_viewer_config()
        },
        device,
    );
    let err = pipeline
        .step_stage_only()
        .expect_err("fake AutoGaze masks must not run");
    assert!(
        err.to_string()
            .contains("loaded model-backed AutoGaze node")
    );
}

#[test]
fn patch_diff_mask_selects_changed_camera_patch() {
    let device = JepaBevyDevice::default();
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::Camera,
        mask_source: BevyJepaMaskSource::PatchDiff,
        pipeline: FeatureFrameViewerConfig {
            context_density: 1.0 / 16.0,
            patch_diff_threshold: 0.01,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    let mut model_config = VJepaConfig::tiny_for_tests();
    model_config.image_size = 64;
    model_config.num_frames = 2;
    model_config.tubelet_size = 2;
    model_config.patch_size = 16;
    let grid = TokenGridShape::new(1, 4, 4);
    let previous_rgba = RgbaImage::new(64, 64);
    let current_rgba = rgba_with_patches(64, 64, &[(2, 1)], 16, image::Rgba([255, 255, 255, 255]));
    let previous = rgba_image_to_tensor(previous_rgba.clone(), 64, &device).expect("prev");
    let current = rgba_image_to_tensor(current_rgba.clone(), 64, &device).expect("current");

    let output = run_sparse_mask_node(
        &config,
        Some(&previous),
        Some(&previous_rgba),
        Some(&current_rgba),
        &current,
        &model_config,
        grid,
    )
    .expect("patch-diff mask");
    assert_eq!(output.write_mask.len(), 1);
    assert_eq!(output.write_mask.dense_len(), grid.len());
    assert_eq!(
        output.write_mask.indices(),
        &[coords_to_token_index(0, 2, 1, grid)]
    );
}

#[test]
fn patch_diff_mask_includes_all_patches_above_threshold() {
    let device = JepaBevyDevice::default();
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::Camera,
        mask_source: BevyJepaMaskSource::PatchDiff,
        pipeline: FeatureFrameViewerConfig {
            context_density: 1.0 / 16.0,
            patch_diff_threshold: 0.01,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    let mut model_config = VJepaConfig::tiny_for_tests();
    model_config.image_size = 64;
    model_config.num_frames = 2;
    model_config.tubelet_size = 2;
    model_config.patch_size = 16;
    let grid = TokenGridShape::new(1, 4, 4);
    let changed = [(0, 0), (1, 3), (2, 1), (3, 2)];
    let previous_rgba = RgbaImage::new(64, 64);
    let current_rgba = rgba_with_patches(64, 64, &changed, 16, image::Rgba([255, 255, 255, 255]));
    let previous = rgba_image_to_tensor(previous_rgba.clone(), 64, &device).expect("prev");
    let current = rgba_image_to_tensor(current_rgba.clone(), 64, &device).expect("current");

    let output = run_sparse_mask_node(
        &config,
        Some(&previous),
        Some(&previous_rgba),
        Some(&current_rgba),
        &current,
        &model_config,
        grid,
    )
    .expect("patch-diff mask");

    assert_eq!(
        output.write_mask.len(),
        changed.len(),
        "adaptive patch-diff thresholding must not top-k cap changed patches"
    );
    assert_eq!(
        output.write_mask.indices(),
        &[
            coords_to_token_index(0, 0, 0, grid),
            coords_to_token_index(0, 1, 3, grid),
            coords_to_token_index(0, 2, 1, grid),
            coords_to_token_index(0, 3, 2, grid),
        ]
    );
}

#[test]
fn patch_diff_refresh_state_promotes_subthreshold_camera_motion() {
    let device = JepaBevyDevice::default();
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::Camera,
        mask_source: BevyJepaMaskSource::PatchDiff,
        pipeline: FeatureFrameViewerConfig {
            context_density: 0.25,
            patch_diff_threshold: 0.10,
            patch_diff_dense_fallback_density: 1.0,
            sparse_encode_mode: BevyJepaSparseEncodeMode::Exact,
            patch_diff_refresh: PatchDiffRefreshConfig {
                subthreshold_decay: 1.0,
                subthreshold_trigger: 1.0,
                subthreshold_max_density: 0.25,
                age_refresh_enabled: false,
                blue_noise_enabled: false,
                max_extra_density: 0.25,
                ..PatchDiffRefreshConfig::default()
            },
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    let mut model_config = VJepaConfig::tiny_for_tests();
    model_config.image_size = 64;
    model_config.num_frames = 2;
    model_config.tubelet_size = 2;
    model_config.patch_size = 16;
    let grid = TokenGridShape::new(1, 4, 4);
    let distractor = coords_to_token_index(0, 0, 0, grid);
    let slow = coords_to_token_index(0, 2, 1, grid);
    let mut refresh_state = PatchDiffRefreshState::default();

    let frames = [
        rgba_with_two_patch_levels(64, 64, (0, 0), 0, (2, 1), 0, 16),
        rgba_with_two_patch_levels(64, 64, (0, 0), 20, (2, 1), 10, 16),
        rgba_with_two_patch_levels(64, 64, (0, 0), 40, (2, 1), 20, 16),
        rgba_with_two_patch_levels(64, 64, (0, 0), 60, (2, 1), 30, 16),
    ];

    let mut latest = None;
    for pair in frames.windows(2).take(2) {
        let previous = rgba_image_to_tensor(pair[0].clone(), 64, &device).expect("prev");
        let current = rgba_image_to_tensor(pair[1].clone(), 64, &device).expect("current");
        latest = Some(
            run_sparse_mask_node_with_refresh_state(
                &config,
                Some(&previous),
                Some(&pair[0]),
                Some(&pair[1]),
                &current,
                &model_config,
                grid,
                Some(&mut refresh_state),
            )
            .expect("patch-diff mask"),
        );
    }
    let second = latest.expect("second mask");
    assert!(second.write_mask.indices().contains(&distractor));
    assert!(!second.write_mask.indices().contains(&slow));

    let previous = rgba_image_to_tensor(frames[2].clone(), 64, &device).expect("prev");
    let current = rgba_image_to_tensor(frames[3].clone(), 64, &device).expect("current");
    let third = run_sparse_mask_node_with_refresh_state(
        &config,
        Some(&previous),
        Some(&frames[2]),
        Some(&frames[3]),
        &current,
        &model_config,
        grid,
        Some(&mut refresh_state),
    )
    .expect("patch-diff mask");

    assert!(
        third.write_mask.indices().contains(&slow),
        "stateful refresh should include repeated below-threshold camera motion"
    );
}

#[test]
fn patch_diff_mask_uses_adaptive_density_for_changed_patches() {
    let device = JepaBevyDevice::default();
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::Camera,
        mask_source: BevyJepaMaskSource::PatchDiff,
        pipeline: FeatureFrameViewerConfig {
            context_density: 1.0,
            patch_diff_threshold: 0.01,
            patch_diff_dense_fallback_density: 0.98,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    let mut model_config = VJepaConfig::tiny_for_tests();
    model_config.image_size = 64;
    model_config.num_frames = 2;
    model_config.tubelet_size = 2;
    model_config.patch_size = 16;
    let grid = TokenGridShape::new(1, 4, 4);
    let previous_rgba = RgbaImage::new(64, 64);
    let current_rgba = rgba_with_patches(
        64,
        64,
        &[(0, 0), (3, 2)],
        16,
        image::Rgba([255, 255, 255, 255]),
    );
    let previous = rgba_image_to_tensor(previous_rgba.clone(), 64, &device).expect("prev");
    let current = rgba_image_to_tensor(current_rgba.clone(), 64, &device).expect("current");

    let output = run_sparse_mask_node(
        &config,
        Some(&previous),
        Some(&previous_rgba),
        Some(&current_rgba),
        &current,
        &model_config,
        grid,
    )
    .expect("patch-diff mask");
    assert_eq!(output.write_mask.len(), 2);
    assert_eq!(
        output.write_mask.indices(),
        &[
            coords_to_token_index(0, 0, 0, grid),
            coords_to_token_index(0, 3, 2, grid),
        ]
    );
}

#[test]
fn patch_diff_zero_threshold_promotes_to_dense_without_sparse_work() {
    let device = JepaBevyDevice::default();
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::Camera,
        mask_source: BevyJepaMaskSource::PatchDiff,
        pipeline: FeatureFrameViewerConfig {
            context_density: 1.0,
            min_context_density: 0.01,
            patch_diff_threshold: 0.0,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    let mut model_config = VJepaConfig::tiny_for_tests();
    model_config.image_size = 64;
    model_config.num_frames = 2;
    model_config.tubelet_size = 2;
    model_config.patch_size = 16;
    let grid = TokenGridShape::new(1, 4, 4);
    let previous_rgba = RgbaImage::new(64, 64);
    let current_rgba = RgbaImage::new(64, 64);
    let previous = rgba_image_to_tensor(previous_rgba.clone(), 64, &device).expect("prev");
    let current = rgba_image_to_tensor(current_rgba.clone(), 64, &device).expect("current");

    let output = run_sparse_mask_node(
        &config,
        Some(&previous),
        Some(&previous_rgba),
        Some(&current_rgba),
        &current,
        &model_config,
        grid,
    )
    .expect("patch-diff mask");

    assert_eq!(
        output.write_mask.indices(),
        SparseTokenMask::all(grid.len()).indices()
    );
}

#[test]
fn patch_diff_high_density_promotes_to_dense_ordered_mask() {
    let device = JepaBevyDevice::default();
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::Camera,
        mask_source: BevyJepaMaskSource::PatchDiff,
        pipeline: FeatureFrameViewerConfig {
            context_density: 1.0,
            patch_diff_threshold: 0.01,
            patch_diff_dense_fallback_density: 0.98,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    let mut model_config = VJepaConfig::tiny_for_tests();
    model_config.image_size = 64;
    model_config.num_frames = 2;
    model_config.tubelet_size = 2;
    model_config.patch_size = 16;
    let grid = TokenGridShape::new(1, 4, 4);
    let changed = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 0),
        (3, 1),
    ];
    let previous_rgba = RgbaImage::new(64, 64);
    let current_rgba = rgba_with_patches(64, 64, &changed, 16, image::Rgba([255, 255, 255, 255]));
    let previous = rgba_image_to_tensor(previous_rgba.clone(), 64, &device).expect("prev");
    let current = rgba_image_to_tensor(current_rgba.clone(), 64, &device).expect("current");

    let output = run_sparse_mask_node(
        &config,
        Some(&previous),
        Some(&previous_rgba),
        Some(&current_rgba),
        &current,
        &model_config,
        grid,
    )
    .expect("patch-diff mask");

    assert_eq!(
        output.write_mask.indices(),
        SparseTokenMask::all(grid.len()).indices()
    );
}

#[test]
fn patch_diff_below_dense_fallback_cutoff_remains_sparse() {
    let device = JepaBevyDevice::default();
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::Camera,
        mask_source: BevyJepaMaskSource::PatchDiff,
        pipeline: FeatureFrameViewerConfig {
            context_density: 1.0,
            patch_diff_threshold: 0.01,
            patch_diff_dense_fallback_density: 0.98,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    let mut model_config = VJepaConfig::tiny_for_tests();
    model_config.image_size = 64;
    model_config.num_frames = 2;
    model_config.tubelet_size = 2;
    model_config.patch_size = 16;
    let grid = TokenGridShape::new(1, 4, 4);
    let changed = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 0),
    ];
    let previous_rgba = RgbaImage::new(64, 64);
    let current_rgba = rgba_with_patches(64, 64, &changed, 16, image::Rgba([255, 255, 255, 255]));
    let previous = rgba_image_to_tensor(previous_rgba.clone(), 64, &device).expect("prev");
    let current = rgba_image_to_tensor(current_rgba.clone(), 64, &device).expect("current");

    let output = run_sparse_mask_node(
        &config,
        Some(&previous),
        Some(&previous_rgba),
        Some(&current_rgba),
        &current,
        &model_config,
        grid,
    )
    .expect("patch-diff mask");

    assert_eq!(output.write_mask.len(), changed.len());
    assert_ne!(
        output.write_mask.indices(),
        SparseTokenMask::all(grid.len()).indices()
    );
}

#[test]
fn sparse_mask_bucket_preserves_changed_tokens_and_limits_shape_churn() {
    let grid = TokenGridShape::new(1, 32, 32);
    let changed = SparseTokenMask::new((0..103).collect(), grid.len()).expect("mask");

    let bucketed = bucket_sparse_mask(changed.clone(), grid, DEFAULT_SPARSE_MASK_BUCKET_TOKENS);

    assert_eq!(bucketed.len(), 256);
    for index in changed.indices() {
        assert!(bucketed.indices().contains(index));
    }
    assert!(!bucketed.is_dense_ordered());
}

#[test]
fn sparse_mask_bucket_padding_does_not_become_write_mask() {
    let grid = TokenGridShape::new(1, 32, 32);
    let changed = SparseTokenMask::new(vec![coords_to_token_index(0, 15, 16, grid)], grid.len())
        .expect("mask");
    let config = BevyJepaConfig {
        pipeline: FeatureFrameViewerConfig {
            sparse_encode_mode: BevyJepaSparseEncodeMode::BucketedContext,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };

    let masks = finalize_patch_diff_masks(changed.clone(), grid, &config);

    assert_eq!(masks.write_mask, changed);
    assert_eq!(masks.encode_mask.len(), 102);
    assert!(
        masks
            .encode_mask
            .indices()
            .contains(&coords_to_token_index(0, 15, 16, grid))
    );
}

#[test]
fn default_sparse_encode_mode_keeps_writes_exact_and_buckets_encode() {
    let grid = TokenGridShape::new(1, 32, 32);
    let changed = SparseTokenMask::new(vec![coords_to_token_index(0, 15, 16, grid)], grid.len())
        .expect("mask");
    let config = BevyJepaConfig::default();

    let masks = finalize_patch_diff_masks(changed.clone(), grid, &config);

    assert_eq!(masks.write_mask, changed);
    assert_eq!(masks.encode_mask.len(), 102);
    assert!(
        masks
            .encode_mask
            .indices()
            .contains(&coords_to_token_index(0, 15, 16, grid))
    );
}

#[test]
fn sparse_mask_bucket_can_trigger_dense_fallback_near_cutoff() {
    let grid = TokenGridShape::new(1, 32, 32);
    let config = BevyJepaConfig {
        pipeline: FeatureFrameViewerConfig {
            sparse_encode_mode: BevyJepaSparseEncodeMode::BucketedContext,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    let changed = SparseTokenMask::new((0..1004).collect(), grid.len()).expect("mask");

    let finalized = finalize_patch_diff_mask(changed, grid, &config);

    assert!(finalized.is_dense_ordered());
}

#[test]
fn shape_prewarm_masks_cover_bucket_widths_once() {
    let grid = TokenGridShape::new(1, 32, 32);
    let config = BevyJepaConfig {
        pipeline: FeatureFrameViewerConfig {
            sparse_encode_mode: BevyJepaSparseEncodeMode::BucketedContext,
            prewarm_shape_buckets: true,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };

    let masks = shape_prewarm_masks(grid, &config);
    let widths: Vec<_> = masks.iter().map(SparseTokenMask::len).collect();

    assert_eq!(widths, vec![102, 256, 512, 1024]);
    assert!(masks.last().expect("dense mask").is_dense_ordered());
}

#[test]
fn empty_sparse_bucket_density_list_uses_legacy_bucket_token_width() {
    let grid = TokenGridShape::new(1, 32, 32);
    let config = BevyJepaConfig {
        pipeline: FeatureFrameViewerConfig {
            sparse_encode_mode: BevyJepaSparseEncodeMode::BucketedContext,
            sparse_mask_bucket_densities: Vec::new(),
            prewarm_shape_buckets: true,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };

    let masks = shape_prewarm_masks(grid, &config);
    let widths: Vec<_> = masks.iter().map(SparseTokenMask::len).collect();

    assert_eq!(widths, vec![256, 512, 1024]);
}

#[test]
fn shape_prewarm_masks_are_empty_for_exact_sparse_encode_mode() {
    let grid = TokenGridShape::new(1, 32, 32);
    let config = BevyJepaConfig {
        pipeline: FeatureFrameViewerConfig {
            sparse_encode_mode: BevyJepaSparseEncodeMode::Exact,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };

    assert!(shape_prewarm_masks(grid, &config).is_empty());
}

#[test]
fn patch_diff_first_frame_bootstraps_dense_token_cache() {
    let device = JepaBevyDevice::default();
    let config = BevyJepaConfig {
        pipeline: FeatureFrameViewerConfig {
            bootstrap_context_density: 1.0,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    let mut model_config = VJepaConfig::tiny_for_tests();
    model_config.image_size = 64;
    model_config.num_frames = 2;
    model_config.tubelet_size = 2;
    model_config.patch_size = 16;
    let grid = TokenGridShape::new(1, 4, 4);
    let current = rgba_image_to_tensor(RgbaImage::new(64, 64), 64, &device).expect("current");

    let output = run_sparse_mask_node(&config, None, None, None, &current, &model_config, grid)
        .expect("bootstrap mask");
    assert_eq!(output.write_mask.len(), grid.len());
}

#[test]
fn rgba_camera_frame_converts_to_model_sized_tensor() {
    let device = JepaBevyDevice::default();
    let frame = RgbaImage::from_pixel(4, 2, image::Rgba([128, 64, 32, 255]));
    let tensor = rgba_image_to_tensor(frame, 64, &device).expect("rgba tensor");
    assert_eq!(tensor.shape().dims::<4>(), [1, 3, 64, 64]);
    let values = values4(tensor);
    let pixels = 64 * 64;
    assert!((values[0] - normalize_model_rgb_channel(128.0 / 255.0, 0)).abs() <= 1.0e-5);
    assert!((values[pixels] - normalize_model_rgb_channel(64.0 / 255.0, 1)).abs() <= 1.0e-5);
    assert!((values[2 * pixels] - normalize_model_rgb_channel(32.0 / 255.0, 2)).abs() <= 1.0e-5);
}

#[test]
fn rgba_camera_preprocess_center_crops_before_resizing() {
    let mut frame = RgbaImage::new(4, 2);
    for y in 0..2 {
        frame.put_pixel(0, y, image::Rgba([255, 0, 0, 255]));
        frame.put_pixel(1, y, image::Rgba([0, 255, 0, 255]));
        frame.put_pixel(2, y, image::Rgba([0, 0, 255, 255]));
        frame.put_pixel(3, y, image::Rgba([255, 0, 0, 255]));
    }

    let cropped = resize_source_rgba(frame, 2);

    assert_eq!(cropped.dimensions(), (2, 2));
    assert_eq!(*cropped.get_pixel(0, 0), image::Rgba([0, 255, 0, 255]));
    assert_eq!(*cropped.get_pixel(1, 0), image::Rgba([0, 0, 255, 255]));
    assert_eq!(*cropped.get_pixel(0, 1), image::Rgba([0, 255, 0, 255]));
    assert_eq!(*cropped.get_pixel(1, 1), image::Rgba([0, 0, 255, 255]));
}

#[test]
fn frame_source_parses_camera_aliases() {
    assert_eq!(
        "webcam".parse::<BevyJepaFrameSource>().expect("webcam"),
        BevyJepaFrameSource::Camera
    );
    assert_eq!(
        "image".parse::<BevyJepaFrameSource>().expect("image"),
        BevyJepaFrameSource::StaticImage
    );
}

#[test]
fn metrics_overlay_line_uses_stable_field_widths() {
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::SyntheticLocalMotion,
        ..tiny_viewer_config()
    };
    let mut first = BevyJepaMetrics {
        frame_ready: true,
        frame_source: BevyJepaFrameSource::SyntheticLocalMotion,
        mask_source: BevyJepaMaskSource::PatchDiff,
        context_tokens: 1,
        dense_tokens: 16,
        viewer_total_us: 9_000,
        total_us: 8_000,
        ..BevyJepaMetrics::default()
    };
    let mut second = first.clone();
    second.context_tokens = 16;
    second.viewer_total_us = 123_450;
    second.total_us = 98_760;
    second.anyup_decode_us = 12_345;
    second.pca_sample_frames = 16;
    second.pca_sample_window_frames = 16;
    second.pca_update_applied = true;

    assert_eq!(
        format_metrics_line(&config, &first).len(),
        format_metrics_line(&config, &second).len()
    );
    second.last_error = Some(
        "camera frame is not ready; this full diagnostic belongs in the console, not the overlay"
            .to_string(),
    );
    let error_line = format_metrics_line(&config, &second);
    assert!(!error_line.contains("camera frame is not ready"));
    assert!(!error_line.contains("error:"));
    assert_eq!(format_metrics_line(&config, &first).len(), error_line.len());
    first.frame_ready = false;
    assert_eq!(
        format_metrics_waiting_line().len(),
        format_metrics_line(&config, &first).len()
    );
}

#[test]
fn metrics_overlay_updates_structured_fields_and_graph() {
    let mut app = App::new();
    app.insert_resource(BevyJepaConfig {
        source: BevyJepaFrameSource::SyntheticLocalMotion,
        show_metrics: true,
        ..tiny_viewer_config()
    })
    .insert_resource(BevyJepaMetrics {
        frame_ready: true,
        frame_index: 1,
        input_frame_index: 2,
        completed_frames: 1,
        frame_source: BevyJepaFrameSource::SyntheticLocalMotion,
        mask_source: BevyJepaMaskSource::PatchDiff,
        context_tokens: 64,
        dense_tokens: 256,
        grid_height: 16,
        grid_width: 16,
        patch_size: 16,
        input_fps: 60.0,
        low_res_fps: 31.0,
        high_res_fps: 7.5,
        viewer_total_us: 33_000,
        encode_us: 2_100,
        cache_update_us: 300,
        token_view_us: 120,
        low_res_pca_us: 450,
        pca_update_us: 90,
        pca_sample_frames: 8,
        pca_sample_window_frames: 16,
        pca_update_events: 3,
        pca_update_fps: 2.0,
        ..BevyJepaMetrics::default()
    })
    .init_resource::<JepaRuntime>()
    .init_resource::<MetricsRollingState>()
    .add_systems(Startup, setup_metrics_overlay)
    .add_systems(Update, update_metrics_overlay);

    app.update();
    for frame in 2..5 {
        {
            let mut metrics = app.world_mut().resource_mut::<BevyJepaMetrics>();
            metrics.frame_index = frame;
            metrics.completed_frames = frame;
            metrics.low_res_fps = 30.0 + frame as f64;
            metrics.viewer_total_us = 33_000 - frame * 1_000;
        }
        app.update();
    }

    let world = app.world_mut();
    let mut text_query = world.query::<(&MetricValueText, &Text)>();
    let rendered = text_query
        .iter(world)
        .map(|(_, text)| text.0.clone())
        .collect::<Vec<_>>();
    assert!(
        rendered
            .iter()
            .any(|line| line.contains("Write") && line.contains("64/256"))
    );
    assert!(
        rendered
            .iter()
            .any(|line| line.contains("Encode") && line.contains("64/256"))
    );
    assert!(rendered.iter().any(|line| line.contains("JEPA")));
    assert!(rendered.iter().any(|line| line.contains("2.10 ms")));
    assert!(rendered.iter().any(|line| line.contains("Cache")));
    assert!(rendered.iter().any(|line| line.contains("Grid view")));
    assert!(!rendered.iter().any(|line| line.contains("Frames   in")));
    assert!(rendered.iter().any(|line| line.contains("Rolling")));

    let mut bar_query = world.query::<(&MetricGraphBar, &Node)>();
    assert!(
        bar_query
            .iter(world)
            .any(|(_, node)| matches!(node.height, Val::Px(height) if height > 1.0))
    );
}

#[test]
fn metrics_text_clarifies_bucketed_encode_and_ttt_diagnostics() {
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::SyntheticLocalMotion,
        encoder_source: BevyJepaEncoderSource::TrainedTtt,
        ..tiny_viewer_config()
    };
    let metrics = BevyJepaMetrics {
        frame_ready: true,
        encoder_source: BevyJepaEncoderSource::TrainedTtt,
        context_tokens: 12,
        dense_tokens: 1024,
        pca_sample_frames: 16,
        pca_sample_window_frames: 16,
        pca_update_events: 3,
        pca_update_fps: 2.0,
        stage_metrics: FeatureFrameMetrics {
            write_width: 12,
            valid_write_tokens: 12,
            encode_width: 256,
            valid_encode_tokens: 256,
            pca_sample_frames: 16,
            pca_sample_window_frames: 16,
            ..FeatureFrameMetrics::default()
        },
        ..BevyJepaMetrics::default()
    };
    let runtime = JepaRuntime::default();
    let history = MetricsRollingState::default();

    let write = metric_value_text(
        &config,
        &metrics,
        &runtime,
        &history,
        MetricValueKind::Tokens,
    );
    let encode = metric_value_text(
        &config,
        &metrics,
        &runtime,
        &history,
        MetricValueKind::EncodeTokens,
    );
    let ttt = metric_value_text(
        &config,
        &metrics,
        &runtime,
        &history,
        MetricValueKind::StageTttStability,
    );
    let basis = metric_value_text(
        &config,
        &metrics,
        &runtime,
        &history,
        MetricValueKind::StagePcaBasis,
    );
    let updates = metric_value_text(
        &config,
        &metrics,
        &runtime,
        &history,
        MetricValueKind::StagePcaUpdates,
    );

    assert!(write.contains("12/1024"));
    assert!(write.contains("cache mask"));
    assert!(encode.contains("256/1024"));
    assert!(encode.contains("bucketed"));
    assert!(ttt.contains("diag off"));
    assert!(basis.contains("cached"));
    assert!(basis.contains(&format!("@{}f", DEFAULT_PCA_UPDATE_EVERY)));
    assert!(updates.contains("2.0 fps"));
    assert!(updates.contains("3 total"));
}

#[test]
fn metrics_status_shows_runtime_autotune_message() {
    let config = BevyJepaConfig::default();
    let metrics = BevyJepaMetrics::default();
    let runtime = JepaRuntime {
        status_message: Some(shape_prewarm_status_message(
            "autotuning",
            &[256, 512, 1024],
        )),
        ..JepaRuntime::default()
    };
    let history = MetricsRollingState::default();

    let status = metric_value_text(
        &config,
        &metrics,
        &runtime,
        &history,
        MetricValueKind::Status,
    );

    assert!(status.contains("autotuning sparse token widths"));
    assert!(status.contains("256"));
    assert!(status.contains("1024"));
}

#[test]
fn controls_help_text_documents_dense_sparse_and_threshold_controls() {
    assert!(default_controls_help().contains("Hover"));
    assert!(default_controls_help().contains("tabs"));
    assert!(control_help_text(JepaControlAction::TabMask).contains("patch-diff"));
    assert!(control_help_text(JepaControlAction::PipelineSparse).contains("patch-diff"));
    assert!(control_help_text(JepaControlAction::PipelineDense).contains("full-frame"));
    assert!(control_help_text(JepaControlAction::PcaLock).contains("fixed"));
    assert!(slider_help_text(JepaControlSliderKind::PatchDiffThreshold).contains("Lower"));
    assert!(slider_help_text(JepaControlSliderKind::DenseFallbackDensity).contains("dense JEPA"));
    assert!(slider_help_text(JepaControlSliderKind::PcaUpdateEvery).contains("Zero locks"));
}

#[test]
fn headless_metrics_align_with_raw_stage_metrics() {
    let device = JepaBevyDevice::default();
    let mut pipeline = BevyJepaHeadlessPipeline::new(
        BevyJepaConfig {
            source: BevyJepaFrameSource::SyntheticLocalMotion,
            ..tiny_viewer_config()
        },
        device,
    );

    let core = pipeline.step_stage_only().expect("stage-only viewer step");
    assert!(core.metrics.aligns_with_stage_metrics());
    assert_eq!(core.metrics.display_tensor_us, 0);
    assert_eq!(
        core.metrics.context_tokens,
        core.metrics.stage_metrics.sparse_width
    );
    assert_eq!(
        core.metrics.dense_tokens,
        core.metrics.stage_metrics.dense_tokens_per_frame
    );
    assert_eq!(core.metrics.grid_height, DEFAULT_IMAGE_SIZE / 16);
    assert_eq!(core.metrics.grid_width, DEFAULT_IMAGE_SIZE / 16);
    assert_eq!(core.metrics.patch_size, 16);
    assert_eq!(
        core.metrics.dense_tokens,
        (DEFAULT_IMAGE_SIZE / 16) * (DEFAULT_IMAGE_SIZE / 16)
    );
    assert_eq!(core.metrics.encoder_source, BevyJepaEncoderSource::TinyTest);

    let display = pipeline
        .step_with_display_panels()
        .expect("display viewer step");
    assert!(display.metrics.aligns_with_stage_metrics());
    assert_eq!(
        display.metrics.display_transfer,
        BevyJepaDisplayTransfer::Gpu
    );
    assert!(display.metrics.viewer_total_us >= display.metrics.total_us);
    assert!(display.metrics.viewer_total_us >= display.metrics.display_tensor_us);
}

#[test]
fn headless_stage_request_can_measure_low_res_only_path() {
    let device = JepaBevyDevice::default();
    let mut pipeline = BevyJepaHeadlessPipeline::new(
        BevyJepaConfig {
            source: BevyJepaFrameSource::SyntheticLocalMotion,
            ..tiny_viewer_config()
        },
        device,
    );

    let output = pipeline
        .step_with_stage_request(FeatureFrameRequest::low_res())
        .expect("low-res-only viewer step");

    assert!(output.metrics.aligns_with_stage_metrics());
    assert_eq!(output.metrics.anyup_decode_us, 0);
    assert_eq!(output.metrics.high_res_pca_us, 0);
    assert!(output.metrics.low_res_pca_us > 0 || !output.metrics.stage_metrics.measured);
}

#[test]
fn scheduled_high_res_does_not_run_inline_with_low_res_stage() {
    let device = JepaBevyDevice::default();
    let image_size = 32;
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::SyntheticLocalMotion,
        pipeline: FeatureFrameViewerConfig {
            high_res_pca_every: 1,
            measure_stages: true,
            ..FeatureFrameViewerConfig::default()
        },
        ..tiny_viewer_config()
    };
    let model_config = tiny_viewer_model_config(image_size);
    let jepa = VJepa2_1Model::<JepaBevyBackend>::new(&model_config, &device);
    let mut anyup_config = AnyUpConfig::tiny_for_tests();
    anyup_config.input_dim = 3;
    let anyup = AnyUp::<JepaBevyBackend>::new(anyup_config, &device).expect("AnyUp");
    let mut pipeline = FeatureFramePipeline::<JepaBevyBackend>::new(
        jepa,
        anyup,
        &model_config,
        FeatureFramePipelineConfig {
            measurement: burn_jepa::FeatureFrameMeasureConfig::enabled(),
            ..FeatureFramePipelineConfig::default()
        },
        1,
        [image_size, image_size],
        &device,
    )
    .expect("pipeline");
    let grid = pipeline.grid();
    let image = synthetic_image_tensor(0, image_size, &device);
    let mask = SparseTokenMask::all(grid.len());
    let processed = run_stage_pipeline_step(
        &config,
        &mut pipeline,
        image,
        &mask,
        &mask,
        FrameId {
            stream_id: 0,
            sequence: 0,
            capture_time_nanos: 0,
        },
        grid,
        model_config.patch_size,
        BevyJepaFrameSource::SyntheticLocalMotion,
        false,
        stage_request_for_frame(&config, 0),
    )
    .expect("low-res stage");

    assert!(processed.high_res_input.is_some());
    assert!(!processed.high_res_updated);
    assert_eq!(processed.metrics.anyup_context_us, 0);
    assert_eq!(processed.metrics.anyup_decode_us, 0);
    assert_eq!(processed.metrics.high_res_pca_us, 0);
    match processed.panels {
        StagePanelData::Tensor { high_res_rgba, .. } => assert!(high_res_rgba.is_none()),
        StagePanelData::Host { high_res_rgba, .. } => assert!(high_res_rgba.is_none()),
    }
    JepaBevyBackend::sync(&device).expect("sync backend before dropping scheduled tensors");
}

#[test]
fn highres_pipeline_can_run_ttt_encoder_branch() {
    let device = JepaBevyDevice::default();
    let model_config = tiny_viewer_model_config(32);
    let base = VJepa2_1Model::<JepaBevyBackend>::new(&model_config, &device);
    let ttt =
        VJepaTttModel::from_model(base, TttEncoderConfig::default(), &device).expect("TTT model");
    let mut anyup_config = AnyUpConfig::tiny_for_tests();
    anyup_config.input_dim = 3;
    let anyup = AnyUp::<JepaBevyBackend>::new(anyup_config, &device).expect("AnyUp");
    let mut pipeline = FeatureFramePipeline::<JepaBevyBackend>::new_with_encoder(
        FeatureFrameJepaEncoder::ttt(ttt),
        anyup,
        &model_config,
        FeatureFramePipelineConfig::default(),
        1,
        [32, 32],
        &device,
    )
    .expect("TTT feature-frame pipeline");
    assert_eq!(pipeline.encoder_kind(), FeatureFrameJepaEncoderKind::Ttt);

    let image = synthetic_image_tensor(0, 32, &device);
    let mask = SparseTokenMask::all(pipeline.grid().len());
    let output = pipeline
        .step_image_with_mask_nodes_measured(image, &mask, FeatureFrameRequest::low_res())
        .expect("TTT pipeline step");
    assert_eq!(output.output.encoded.grid, pipeline.grid());
    assert_eq!(
        output.output.encoded.tokens.shape().dims::<3>()[1],
        pipeline.grid().len()
    );
}

#[cfg(feature = "sparse-patchify-wgpu")]
#[test]
fn highres_pipeline_can_run_ttt_sparse_patchify_branch() {
    let device = JepaBevyDevice::default();
    let model_config = tiny_viewer_model_config(32);
    let base = VJepa2_1Model::<JepaBevyBackend>::new(&model_config, &device);
    let ttt =
        VJepaTttModel::from_model(base, TttEncoderConfig::default(), &device).expect("TTT model");
    let mut anyup_config = AnyUpConfig::tiny_for_tests();
    anyup_config.input_dim = 3;
    let anyup = AnyUp::<JepaBevyBackend>::new(anyup_config, &device).expect("AnyUp");
    let mut pipeline = FeatureFramePipeline::<JepaBevyBackend>::new_with_encoder(
        FeatureFrameJepaEncoder::ttt(ttt),
        anyup,
        &model_config,
        FeatureFramePipelineConfig::default(),
        1,
        [32, 32],
        &device,
    )
    .expect("TTT feature-frame pipeline");

    let image = synthetic_image_tensor(0, 32, &device);
    let mask = SparseTokenMask::all(pipeline.grid().len());
    let batch_mask = SparseMaskBatch::uniform(mask, 1, pipeline.device()).expect("mask batch");
    let patchify_plan =
        SparsePatchifyBatchPlan::new(batch_mask, pipeline.grid(), pipeline.device())
            .expect("patchify plan");
    let output = pipeline
        .step_image_with_sparse_patchify_plan_wgpu_nodes_measured(
            image,
            &patchify_plan,
            FeatureFrameRequest::low_res(),
            pipeline.config().measurement,
        )
        .expect("TTT sparse patchify pipeline step");
    assert_eq!(
        output.metrics.encode_path,
        FeatureFrameEncodePath::SparsePatchify
    );
    assert_eq!(output.output.encoded.grid, pipeline.grid());
    assert_eq!(
        output.output.encoded.tokens.shape().dims::<3>()[1],
        pipeline.grid().len()
    );
}

#[test]
#[ignore = "loads the local 433 MiB production TTT checkpoint"]
fn default_trained_ttt_artifact_initializes_viewer_encoder() {
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::SyntheticLocalMotion,
        pipeline: FeatureFrameViewerConfig {
            image_size: MIN_PIPELINE_IMAGE_SIZE,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    let path = effective_ttt_model_path(&config).expect("default trained TTT path");
    if !path.exists() {
        eprintln!(
            "skipping: trained TTT checkpoint is missing at {}",
            path.display()
        );
        return;
    }
    let device = JepaBevyDevice::default();
    let (encoder, model_config) =
        load_viewer_encoder(&config, config.pipeline_image_size(), &device)
            .expect("load default trained TTT encoder");
    assert_eq!(encoder.kind(), FeatureFrameJepaEncoderKind::Ttt);
    assert_eq!(model_config.model_type, "vjepa2_1");
    assert_eq!(model_config.encoder.embed_dim, 768);
}

#[test]
#[ignore = "loads the local production TTT checkpoint and runs a WebGPU forward step"]
fn default_trained_ttt_pipeline_runs_core_step() {
    let config = BevyJepaConfig {
        source: BevyJepaFrameSource::SyntheticLocalMotion,
        pipeline: FeatureFrameViewerConfig {
            image_size: MIN_PIPELINE_IMAGE_SIZE,
            ..FeatureFrameViewerConfig::default()
        },
        ..BevyJepaConfig::default()
    };
    let path = effective_ttt_model_path(&config).expect("default trained TTT path");
    if !path.exists() {
        eprintln!(
            "skipping: trained TTT checkpoint is missing at {}",
            path.display()
        );
        return;
    }
    let device = JepaBevyDevice::default();
    let mut pipeline = BevyJepaHeadlessPipeline::new(config, device);
    let output = pipeline
        .step_stage_only()
        .expect("trained TTT viewer stage-only step");
    assert_eq!(
        output.metrics.encoder_source,
        BevyJepaEncoderSource::TrainedTtt
    );
    assert_eq!(output.metrics.grid_height, MIN_PIPELINE_IMAGE_SIZE / 16);
    assert_eq!(output.metrics.grid_width, MIN_PIPELINE_IMAGE_SIZE / 16);
    assert_eq!(output.metrics.dense_tokens, 256);
}

#[test]
#[ignore = "loads the local production TTT checkpoint and prints a density/timing sweep"]
fn default_trained_ttt_density_sweep_reports_stage_metrics() {
    let mut config = BevyJepaConfig {
        source: BevyJepaFrameSource::SyntheticLocalMotion,
        pipeline: FeatureFrameViewerConfig {
            image_size: DEFAULT_IMAGE_SIZE,
            measure_stages: true,
            sync_measurements: true,
            high_res_pca_every: 0,
            ..FeatureFrameViewerConfig::default()
        },
        anyup_weights: None,
        ..BevyJepaConfig::default()
    };
    let path = effective_ttt_model_path(&config).expect("default trained TTT path");
    if !path.exists() {
        eprintln!(
            "skipping: trained TTT checkpoint is missing at {}",
            path.display()
        );
        return;
    }
    config.anyup_weights = None;
    let device = JepaBevyDevice::default();
    let image_size = config.pipeline_image_size();
    let (encoder, model_config) =
        load_viewer_encoder(&config, image_size, &device).expect("load trained TTT encoder");
    let mut anyup_config = AnyUpConfig::tiny_for_tests();
    anyup_config.input_dim = 3;
    let anyup = AnyUp::<JepaBevyBackend>::new(anyup_config, &device).expect("AnyUp");
    let mut pipeline = FeatureFramePipeline::<JepaBevyBackend>::new_with_encoder(
        encoder,
        anyup,
        &model_config,
        FeatureFramePipelineConfig {
            pca_update: FeaturePcaUpdateConfig::rolling_low_res_every(
                config.pca_update_every.max(1),
            ),
            measurement: burn_jepa::FeatureFrameMeasureConfig {
                enabled: true,
                sync_backend: true,
            },
            ..FeatureFramePipelineConfig::default()
        },
        1,
        [image_size, image_size],
        &device,
    )
    .expect("pipeline");
    let grid = pipeline.grid();
    let image = synthetic_image_tensor(0, image_size, &device);
    if let Some(report) = prewarm_feature_frame_shapes(&config, &mut pipeline, image_size, &device)
        .expect("prewarm shape buckets")
    {
        eprintln!(
            "prewarm,widths={:?},total_ms={:.3}",
            report.token_widths,
            micros_to_ms(report.total_us)
        );
    }

    eprintln!(
        "pass,density,tokens,encode_ms,cache_ms,pca_update_ms,low_res_pca_ms,total_ms,wall_ms,encode_path,pca_update"
    );
    for pass in ["cold", "warm"] {
        for density in [0.10f32, 0.30, 0.50, 0.75, 0.85, 0.95, 0.98, 1.0] {
            let keep = ((grid.len() as f32) * density)
                .ceil()
                .min(grid.len() as f32) as usize;
            let raw_mask = if keep == grid.len() {
                SparseTokenMask::all(grid.len())
            } else {
                SparseTokenMask::evenly_spaced(grid.len(), keep)
            };
            let mask = finalize_patch_diff_mask(raw_mask, grid, &config);
            let stage = run_stage_step_with_config_and_request(
                &config,
                &mut pipeline,
                image.clone(),
                &mask,
                &mask,
                FeatureFrameRequest::low_res(),
            )
            .expect("density step");
            eprintln!(
                "{pass},{density:.2},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:?},{}",
                mask.len(),
                micros_to_ms(stage.metrics.encode_us),
                micros_to_ms(stage.metrics.cache_update_us),
                micros_to_ms(stage.metrics.pca_update_us),
                micros_to_ms(stage.metrics.low_res_pca_project_us),
                micros_to_ms(stage.metrics.total_us),
                micros_to_ms(stage.wall_us),
                stage.metrics.encode_path,
                stage.metrics.pca_update_applied
            );
        }
    }
}

fn rgba_with_patches(
    width: u32,
    height: u32,
    patches: &[(usize, usize)],
    patch_size: usize,
    color: image::Rgba<u8>,
) -> RgbaImage {
    rgba_with_base_and_patches(
        width,
        height,
        image::Rgba([0, 0, 0, 0]),
        patches,
        patch_size,
        color,
    )
}

fn panel_patch_center_colors(
    rgba: &[u8],
    width: usize,
    height: usize,
    grid_height: usize,
    grid_width: usize,
) -> Vec<[u8; 3]> {
    let patch_h = (height / grid_height.max(1)).max(1);
    let patch_w = (width / grid_width.max(1)).max(1);
    let mut colors = Vec::with_capacity(grid_height * grid_width);
    for row in 0..grid_height {
        for col in 0..grid_width {
            let y = (row * patch_h + patch_h / 2).min(height.saturating_sub(1));
            let x = (col * patch_w + patch_w / 2).min(width.saturating_sub(1));
            let offset = (y * width + x) * 4;
            colors.push([rgba[offset], rgba[offset + 1], rgba[offset + 2]]);
        }
    }
    colors
}

fn unique_rgb_color_count(colors: &[[u8; 3]], min_distance: u8) -> usize {
    let threshold = u32::from(min_distance).pow(2);
    colors
        .iter()
        .enumerate()
        .filter(|(index, color)| {
            colors[..*index]
                .iter()
                .all(|seen| rgb_distance_sq(seen, color) > threshold)
        })
        .count()
}

fn rgb_distance_sq(left: &[u8; 3], right: &[u8; 3]) -> u32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = i32::from(*left) - i32::from(*right);
            (delta * delta) as u32
        })
        .sum()
}

fn rgba_with_base_and_patches(
    width: u32,
    height: u32,
    base: image::Rgba<u8>,
    patches: &[(usize, usize)],
    patch_size: usize,
    color: image::Rgba<u8>,
) -> RgbaImage {
    let mut image = RgbaImage::from_pixel(width, height, base);
    for &(patch_row, patch_col) in patches {
        let row_start = patch_row * patch_size;
        let col_start = patch_col * patch_size;
        for y in row_start..(row_start + patch_size).min(height as usize) {
            for x in col_start..(col_start + patch_size).min(width as usize) {
                image.put_pixel(x as u32, y as u32, color);
            }
        }
    }
    image
}

fn rgba_with_two_patch_levels(
    width: u32,
    height: u32,
    first_patch: (usize, usize),
    first_level: u8,
    second_patch: (usize, usize),
    second_level: u8,
    patch_size: usize,
) -> RgbaImage {
    let mut image = RgbaImage::new(width, height);
    paint_patch(
        &mut image,
        first_patch,
        patch_size,
        image::Rgba([first_level, first_level, first_level, 255]),
    );
    paint_patch(
        &mut image,
        second_patch,
        patch_size,
        image::Rgba([second_level, second_level, second_level, 255]),
    );
    image
}

fn paint_patch(
    image: &mut RgbaImage,
    patch: (usize, usize),
    patch_size: usize,
    color: image::Rgba<u8>,
) {
    let row_start = patch.0 * patch_size;
    let col_start = patch.1 * patch_size;
    for y in row_start..(row_start + patch_size).min(image.height() as usize) {
        for x in col_start..(col_start + patch_size).min(image.width() as usize) {
            image.put_pixel(x as u32, y as u32, color);
        }
    }
}
