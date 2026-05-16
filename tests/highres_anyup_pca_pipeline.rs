use burn::tensor::{Tensor, TensorData};
use burn_jepa::{
    AnyUp, AnyUpConfig, FeatureFrameRequest, FeatureFrameSchedule, FeaturePcaConfig,
    FeaturePcaProjector, FeaturePcaUpdateConfig, FeaturePcaUpdateMode, FeaturePcaUpdateScheduler,
    FrameQueuePolicy, InterframeJepaFeatureMemory, InterframeJepaFeatureMemoryConfig,
    SparseJepaAnyUpPcaBackpressurePolicy, SparseJepaAnyUpPcaEncodePath, SparseJepaAnyUpPcaFrameId,
    SparseJepaAnyUpPcaFrameInput, SparseJepaAnyUpPcaMeasurementConfig, SparseJepaAnyUpPcaPipeline,
    SparseJepaAnyUpPcaPipelineConfig, SparseJepaAnyUpPcaStream, SparseJepaAnyUpPcaStreamConfig,
    SparseMaskBatch, SparseTokenMask, TokenGridShape, VJepa2_1Model, VJepaConfig,
    jepa_feature_tokens_to_nchw,
};

type B = burn::backend::NdArray<f32>;

#[test]
fn pca_identity_projects_tokens_and_display_values_without_host_statistics() {
    let device = Default::default();
    let projector = FeaturePcaProjector::<B>::identity(4, FeaturePcaConfig::default(), &device)
        .expect("pca projector");
    let tokens = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0], [1, 2, 4]),
        &device,
    );

    let projected = projector.project_tokens(tokens.clone()).expect("project");
    assert_eq!(projected.shape().dims::<3>(), [1, 2, 3]);
    assert_close(
        &values3(projected),
        &[1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        1.0e-5,
    );

    let display = projector
        .project_tokens_display(tokens)
        .expect("display project");
    let values = values3(display);
    assert!(values.iter().all(|value| (0.0..=1.0).contains(value)));
}

#[test]
fn pca_online_update_preserves_projector_shapes() {
    let device = Default::default();
    let mut projector = FeaturePcaProjector::<B>::identity(
        5,
        FeaturePcaConfig {
            online_learning_rate: 0.2,
            mean_momentum: 0.5,
            ..FeaturePcaConfig::default()
        },
        &device,
    )
    .expect("pca projector");
    let tokens = Tensor::<B, 3>::ones([2, 6, 5], &device);

    projector
        .update_online_tokens(tokens.clone())
        .expect("online update");
    let projected = projector.project_tokens(tokens).expect("project");

    assert_eq!(projector.components().shape().dims::<3>(), [1, 5, 3]);
    assert_eq!(projector.mean().shape().dims::<3>(), [1, 1, 5]);
    assert_eq!(projected.shape().dims::<3>(), [2, 6, 3]);
}

#[test]
fn rolling_pca_update_keeps_basis_stable_and_orthogonal() {
    let device = Default::default();
    let mut projector = FeaturePcaProjector::<B>::identity(
        5,
        FeaturePcaConfig {
            online_learning_rate: 0.25,
            mean_momentum: 0.4,
            ..FeaturePcaConfig::default()
        },
        &device,
    )
    .expect("pca projector");
    let tokens = Tensor::<B, 3>::from_data(
        TensorData::new(
            vec![
                3.0, 0.2, 0.1, 0.0, 0.0, 2.0, 1.0, 0.1, 0.0, 0.0, 1.0, 0.5, 2.0, 0.1, 0.0, -2.0,
                -0.7, 0.2, 1.0, 0.0,
            ],
            [1, 4, 5],
        ),
        &device,
    );

    projector
        .update_rolling_tokens_iterations(tokens, 4)
        .expect("rolling update");
    let components = values3(projector.components());

    for channel in 0..3 {
        let norm = component_dot(&components, 5, channel, channel).sqrt();
        assert!(
            (norm - 1.0).abs() <= 1.0e-3,
            "component {channel} norm should stay near one, got {norm}"
        );
    }
    for left in 0..3 {
        for right in left + 1..3 {
            let dot = component_dot(&components, 5, left, right).abs();
            assert!(
                dot <= 1.0e-3,
                "components {left}/{right} should stay orthogonal, dot={dot}"
            );
        }
    }
}

#[test]
fn rolling_pca_masked_update_ignores_unobserved_cache_slots() {
    let device = Default::default();
    let config = FeaturePcaConfig {
        online_learning_rate: 0.25,
        mean_momentum: 0.4,
        ..FeaturePcaConfig::default()
    };
    let mut masked =
        FeaturePcaProjector::<B>::identity(4, config.clone(), &device).expect("masked projector");
    let mut observed_only =
        FeaturePcaProjector::<B>::identity(4, config, &device).expect("observed projector");
    let observed_tokens = Tensor::<B, 3>::from_data(
        TensorData::new(vec![3.0, 0.5, 0.0, 0.0, -2.0, 1.0, 0.25, 0.0], [1, 2, 4]),
        &device,
    );
    let full_cache = Tensor::<B, 3>::from_data(
        TensorData::new(
            vec![
                3.0, 0.5, 0.0, 0.0, -2.0, 1.0, 0.25, 0.0, 100.0, 100.0, 100.0, 100.0, -50.0, -50.0,
                -50.0, -50.0,
            ],
            [1, 4, 4],
        ),
        &device,
    );
    let weights =
        Tensor::<B, 2>::from_data(TensorData::new(vec![1.0, 1.0, 0.0, 0.0], [1, 4]), &device);

    masked
        .update_rolling_masked_tokens(full_cache, weights)
        .expect("masked update");
    observed_only
        .update_rolling_tokens(observed_tokens)
        .expect("observed update");

    assert_close(
        &values3(masked.components()),
        &values3(observed_only.components()),
        1.0e-4,
    );
    assert_close(
        &values3(masked.mean()),
        &values3(observed_only.mean()),
        1.0e-4,
    );
}

#[test]
fn pca_update_scheduler_updates_on_configured_cadence() {
    let mut scheduler = FeaturePcaUpdateScheduler::new(FeaturePcaUpdateConfig {
        mode: FeaturePcaUpdateMode::RollingOja,
        every_n_frames: 2,
        warmup_frames: 1,
        min_tokens_per_update: 2,
        iterations_per_update: 1,
    })
    .expect("scheduler");

    assert!(!scheduler.observe_batch(1, 2).update);
    assert!(!scheduler.observe_batch(1, 1).update);
    assert!(scheduler.observe_batch(1, 2).update);
    assert_eq!(scheduler.update_count(), 1);
    assert!(!scheduler.observe_batch(1, 2).update);
    assert!(scheduler.observe_batch(1, 2).update);
    assert_eq!(scheduler.observed_frames(), 5);
}

#[test]
fn pipeline_pca_update_node_is_scheduled_independently_from_display_nodes() {
    let device = Default::default();
    let model_config = VJepaConfig::tiny_for_tests();
    let mut pipeline = tiny_pipeline(
        1,
        SparseJepaAnyUpPcaPipelineConfig {
            anyup_q_chunk_size: Some(1),
            pca_update: FeaturePcaUpdateConfig::rolling_low_res_every(2),
            ..SparseJepaAnyUpPcaPipelineConfig::default()
        },
        &device,
        &model_config,
    );
    let image = Tensor::<B, 4>::ones(
        [1, 3, model_config.image_size, model_config.image_size],
        &device,
    );
    let mask = SparseTokenMask::new(vec![0, 3], pipeline.grid().len()).expect("mask");

    let first = pipeline
        .step_image_with_mask_nodes_measured(image.clone(), &mask, FeatureFrameRequest::low_res())
        .expect("first step");
    assert!(!first.metrics.pca_update_applied);
    assert_eq!(first.metrics.pca_update_tokens, 0);

    let second = pipeline
        .step_image_with_mask_nodes_measured(image, &mask, FeatureFrameRequest::none())
        .expect("second step");
    assert!(second.metrics.pca_update_applied);
    assert_eq!(second.metrics.pca_update_tokens, pipeline.grid().len());
    assert_eq!(pipeline.pca_update_scheduler().update_count(), 1);
    assert!(!second.output.has_low_res_pca());
    assert!(!second.output.has_high_res_pca());
}

#[test]
fn token_feature_cache_converts_to_low_res_nchw_for_anyup() {
    let device = Default::default();
    let grid = TokenGridShape::new(1, 2, 2);
    let features = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0], [1, 4, 2]),
        &device,
    );

    let nchw = jepa_feature_tokens_to_nchw(features, grid).expect("nchw");

    assert_eq!(nchw.shape().dims::<4>(), [1, 2, 2, 2]);
    assert_close(
        &values4(nchw),
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        1.0e-5,
    );
}

#[test]
fn sparse_jepa_anyup_pca_pipeline_runs_end_to_end_on_tiny_config() {
    let device = Default::default();
    let model_config = VJepaConfig::tiny_for_tests();
    let jepa = VJepa2_1Model::<B>::new(&model_config, &device);
    let anyup = AnyUp::<B>::new(AnyUpConfig::tiny_for_tests(), &device).expect("anyup");
    let mut pipeline = SparseJepaAnyUpPcaPipeline::<B>::new(
        jepa,
        anyup,
        &model_config,
        SparseJepaAnyUpPcaPipelineConfig {
            anyup_q_chunk_size: Some(1),
            update_pca_online: true,
            ..SparseJepaAnyUpPcaPipelineConfig::default()
        },
        1,
        [model_config.image_size, model_config.image_size],
        &device,
    )
    .expect("pipeline");
    let image = Tensor::<B, 4>::ones(
        [1, 3, model_config.image_size, model_config.image_size],
        &device,
    );
    let mask = SparseTokenMask::new(vec![0, 3], pipeline.grid().len()).expect("mask");

    let output = pipeline
        .step_image_with_mask(image, &mask)
        .expect("pipeline step");

    assert_eq!(output.mask, mask);
    assert_eq!(output.encoded.tokens.shape().dims::<3>(), [1, 2, 32]);
    assert_eq!(output.token_cache.updated_tokens, 2);
    assert_eq!(output.low_res_features.shape().dims::<4>(), [1, 32, 2, 2]);
    assert_eq!(
        output.high_res_features.shape().dims::<4>(),
        [1, 32, model_config.image_size, model_config.image_size]
    );
    assert_eq!(
        output.pca_display.shape().dims::<4>(),
        [1, 3, model_config.image_size, model_config.image_size]
    );
}

#[test]
fn batched_pipeline_supports_fixed_width_per_frame_masks_and_stage_metrics() {
    let device = Default::default();
    let model_config = VJepaConfig::tiny_for_tests();
    let mut pipeline = tiny_pipeline(
        2,
        SparseJepaAnyUpPcaPipelineConfig {
            anyup_q_chunk_size: Some(1),
            ..SparseJepaAnyUpPcaPipelineConfig::default()
        },
        &device,
        &model_config,
    );
    let image = Tensor::<B, 4>::ones(
        [2, 3, model_config.image_size, model_config.image_size],
        &device,
    );
    let mask =
        SparseMaskBatch::from_rows(vec![vec![0, 3], vec![1, 2]], pipeline.grid().len(), &device)
            .expect("fixed-width masks");

    let measured = pipeline
        .step_image_with_mask_batch_measured(
            image,
            mask,
            SparseJepaAnyUpPcaMeasurementConfig::enabled(),
        )
        .expect("batched pipeline step");

    assert_eq!(
        measured.output.encoded.tokens.shape().dims::<3>(),
        [2, 2, 32]
    );
    assert_eq!(measured.output.token_cache.updated_tokens, 4);
    assert_eq!(
        measured.output.pca_display.shape().dims::<4>(),
        [2, 3, model_config.image_size, model_config.image_size]
    );
    assert!(measured.metrics.measured);
    assert_eq!(
        measured.metrics.encode_path,
        SparseJepaAnyUpPcaEncodePath::DensePatchEmbed
    );
    assert_eq!(measured.metrics.frame_count, 2);
    assert_eq!(measured.metrics.sparse_width, 2);
    assert_eq!(measured.metrics.valid_sparse_tokens, 4);
    assert_eq!(
        measured.metrics.output_pixels,
        2 * model_config.image_size * model_config.image_size
    );
}

#[test]
fn frame_node_requests_emit_low_and_high_res_artifacts_independently() {
    let device = Default::default();
    let model_config = VJepaConfig::tiny_for_tests();
    let mut pipeline = tiny_pipeline(
        1,
        SparseJepaAnyUpPcaPipelineConfig {
            anyup_q_chunk_size: Some(1),
            ..SparseJepaAnyUpPcaPipelineConfig::default()
        },
        &device,
        &model_config,
    );
    let image = Tensor::<B, 4>::ones(
        [1, 3, model_config.image_size, model_config.image_size],
        &device,
    );
    let mask = SparseTokenMask::new(vec![0, 3], pipeline.grid().len()).expect("mask");

    let low = pipeline
        .step_image_with_mask_nodes_measured(image.clone(), &mask, FeatureFrameRequest::low_res())
        .expect("low-res nodes");
    assert!(low.output.has_low_res_pca());
    assert!(!low.output.has_high_res_pca());
    assert_eq!(
        low.output
            .low_res
            .pca_display
            .as_ref()
            .expect("low-res PCA")
            .shape()
            .dims::<4>(),
        [1, 3, 2, 2]
    );
    assert_eq!(low.metrics.anyup_decode_us, 0);

    let high = pipeline
        .step_image_with_mask_nodes_measured(image, &mask, FeatureFrameRequest::high_res())
        .expect("high-res nodes");
    assert!(!high.output.has_low_res_pca());
    assert!(high.output.has_high_res_pca());
    assert_eq!(
        high.output
            .high_res
            .as_ref()
            .expect("high-res")
            .pca_display
            .as_ref()
            .expect("high-res PCA")
            .shape()
            .dims::<4>(),
        [1, 3, model_config.image_size, model_config.image_size]
    );
}

#[test]
fn high_res_pca_only_matches_full_feature_decode_display() {
    let device = Default::default();
    let model_config = VJepaConfig::tiny_for_tests();
    let mut pipeline = tiny_pipeline(
        1,
        SparseJepaAnyUpPcaPipelineConfig {
            anyup_q_chunk_size: Some(1),
            ..SparseJepaAnyUpPcaPipelineConfig::default()
        },
        &device,
        &model_config,
    );
    let image = Tensor::<B, 4>::ones(
        [1, 3, model_config.image_size, model_config.image_size],
        &device,
    );
    let mask = SparseTokenMask::new(vec![0, 3], pipeline.grid().len()).expect("mask");

    let pca_only = pipeline
        .step_image_with_mask_nodes_measured(image.clone(), &mask, FeatureFrameRequest::full_pca())
        .expect("PCA-only high-res nodes");
    assert!(pca_only.output.has_high_res_pca());
    assert!(!pca_only.output.has_high_res_features());
    let pca_only_display = pca_only
        .output
        .high_res
        .expect("high-res PCA")
        .pca_display
        .expect("PCA display");

    pipeline.reset();
    let full = pipeline
        .step_image_with_mask_nodes_measured(image, &mask, FeatureFrameRequest::full())
        .expect("full high-res nodes");
    assert!(full.output.has_high_res_features());
    let full_display = full
        .output
        .high_res
        .expect("high-res full")
        .pca_display
        .expect("PCA display");

    assert_close(&values4(pca_only_display), &values4(full_display), 1.0e-4);
}

#[test]
fn inflight_stream_batches_frames_in_order_and_reports_queue_wait() {
    let device = Default::default();
    let model_config = VJepaConfig::tiny_for_tests();
    let pipeline = tiny_pipeline(
        2,
        SparseJepaAnyUpPcaPipelineConfig {
            anyup_q_chunk_size: Some(1),
            ..SparseJepaAnyUpPcaPipelineConfig::default()
        },
        &device,
        &model_config,
    );
    let grid = pipeline.grid();
    let mut stream = SparseJepaAnyUpPcaStream::new(
        pipeline,
        SparseJepaAnyUpPcaStreamConfig {
            queue_capacity: 4,
            batch_size: 2,
            measurement: SparseJepaAnyUpPcaMeasurementConfig::enabled(),
            ..SparseJepaAnyUpPcaStreamConfig::default()
        },
    )
    .expect("stream");

    stream
        .enqueue(frame_input(7, 0, vec![0, 3], grid, &device, &model_config))
        .expect("enqueue first");
    stream
        .enqueue(frame_input(7, 1, vec![1, 2], grid, &device, &model_config))
        .expect("enqueue second");

    let output = stream
        .process_next_ready()
        .expect("process")
        .expect("ready batch");

    assert_eq!(
        output.frame_ids,
        vec![
            SparseJepaAnyUpPcaFrameId {
                stream_id: 7,
                sequence: 0,
                capture_time_nanos: 0,
            },
            SparseJepaAnyUpPcaFrameId {
                stream_id: 7,
                sequence: 1,
                capture_time_nanos: 1,
            },
        ]
    );
    assert_eq!(output.frame_timings.len(), 2);
    assert_eq!(output.queued_after, 0);
    assert_eq!(output.metrics.frame_count, 2);
    assert_eq!(stream.stats().emitted_frames, 2);
}

#[test]
fn inflight_stream_backpressure_rejects_or_drops_before_queue_buildup() {
    let device = Default::default();
    let model_config = VJepaConfig::tiny_for_tests();
    let reject_pipeline = tiny_pipeline(
        2,
        SparseJepaAnyUpPcaPipelineConfig::default(),
        &device,
        &model_config,
    );
    let grid = reject_pipeline.grid();
    let mut reject_stream = SparseJepaAnyUpPcaStream::new(
        reject_pipeline,
        SparseJepaAnyUpPcaStreamConfig {
            queue_capacity: 2,
            batch_size: 2,
            ..SparseJepaAnyUpPcaStreamConfig::default()
        },
    )
    .expect("reject stream");
    reject_stream
        .enqueue(frame_input(0, 0, vec![0, 3], grid, &device, &model_config))
        .expect("enqueue 0");
    reject_stream
        .enqueue(frame_input(0, 1, vec![0, 3], grid, &device, &model_config))
        .expect("enqueue 1");
    let err = reject_stream
        .enqueue(frame_input(0, 2, vec![0, 3], grid, &device, &model_config))
        .expect_err("full queue should reject newest");
    assert!(err.to_string().contains("queue is full"));

    let drop_pipeline = tiny_pipeline(
        2,
        SparseJepaAnyUpPcaPipelineConfig::default(),
        &device,
        &model_config,
    );
    let mut drop_stream = SparseJepaAnyUpPcaStream::new(
        drop_pipeline,
        SparseJepaAnyUpPcaStreamConfig {
            queue_capacity: 2,
            batch_size: 2,
            backpressure: SparseJepaAnyUpPcaBackpressurePolicy::DropOldest,
            ..SparseJepaAnyUpPcaStreamConfig::default()
        },
    )
    .expect("drop stream");
    drop_stream
        .enqueue(frame_input(0, 0, vec![0, 3], grid, &device, &model_config))
        .expect("enqueue 0");
    drop_stream
        .enqueue(frame_input(0, 1, vec![0, 3], grid, &device, &model_config))
        .expect("enqueue 1");
    let report = drop_stream
        .enqueue(frame_input(0, 2, vec![0, 3], grid, &device, &model_config))
        .expect("drop oldest");
    assert_eq!(report.dropped_frame.expect("dropped").sequence, 0);

    let output = drop_stream
        .process_next_ready()
        .expect("process")
        .expect("ready");
    assert_eq!(
        output
            .frame_ids
            .iter()
            .map(|id| id.sequence)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(output.dropped_frames_total, 1);
}

#[test]
fn inflight_stream_can_overwrite_the_latest_queued_frame() {
    let device = Default::default();
    let model_config = VJepaConfig::tiny_for_tests();
    let pipeline = tiny_pipeline(
        1,
        SparseJepaAnyUpPcaPipelineConfig::default(),
        &device,
        &model_config,
    );
    let grid = pipeline.grid();
    let mut stream = SparseJepaAnyUpPcaStream::new(
        pipeline,
        SparseJepaAnyUpPcaStreamConfig {
            queue_capacity: 1,
            batch_size: 1,
            backpressure: FrameQueuePolicy::OverwriteNewest,
            ..SparseJepaAnyUpPcaStreamConfig::default()
        },
    )
    .expect("overwrite stream");

    stream
        .enqueue(frame_input(0, 0, vec![0, 3], grid, &device, &model_config))
        .expect("enqueue first");
    let report = stream
        .enqueue(frame_input(0, 1, vec![0, 3], grid, &device, &model_config))
        .expect("overwrite queued");
    assert_eq!(report.queued_frames, 1);
    assert_eq!(report.overwritten_frame.expect("overwritten").sequence, 0);
    assert_eq!(stream.stats().overwritten_frames, 1);

    let output = stream
        .process_next_ready()
        .expect("process")
        .expect("ready");
    assert_eq!(output.frame_ids[0].sequence, 1);
    assert_eq!(output.dropped_frames_total, 1);
}

#[test]
fn frame_stream_schedule_emits_low_and_high_res_nodes_at_different_rates() {
    let device = Default::default();
    let model_config = VJepaConfig::tiny_for_tests();
    let pipeline = tiny_pipeline(
        1,
        SparseJepaAnyUpPcaPipelineConfig {
            anyup_q_chunk_size: Some(1),
            ..SparseJepaAnyUpPcaPipelineConfig::default()
        },
        &device,
        &model_config,
    );
    let grid = pipeline.grid();
    let mut stream = SparseJepaAnyUpPcaStream::new(
        pipeline,
        SparseJepaAnyUpPcaStreamConfig {
            queue_capacity: 2,
            batch_size: 1,
            schedule: FeatureFrameSchedule {
                low_res_pca_every: Some(1),
                high_res_pca_every: Some(2),
            },
            ..SparseJepaAnyUpPcaStreamConfig::default()
        },
    )
    .expect("scheduled stream");

    stream
        .enqueue(frame_input(0, 0, vec![0, 3], grid, &device, &model_config))
        .expect("enqueue 0");
    let first = stream
        .process_next_ready_nodes()
        .expect("process first")
        .expect("first output");
    assert!(first.output.has_low_res_pca());
    assert!(first.output.has_high_res_pca());

    stream
        .enqueue(frame_input(0, 1, vec![0, 3], grid, &device, &model_config))
        .expect("enqueue 1");
    let second = stream
        .process_next_ready_nodes()
        .expect("process second")
        .expect("second output");
    assert!(second.output.has_low_res_pca());
    assert!(!second.output.has_high_res_pca());
}

#[test]
fn inflight_stream_rejects_sequence_rewind_and_variable_width_front_batch() {
    let device = Default::default();
    let model_config = VJepaConfig::tiny_for_tests();
    let pipeline = tiny_pipeline(
        2,
        SparseJepaAnyUpPcaPipelineConfig::default(),
        &device,
        &model_config,
    );
    let grid = pipeline.grid();
    let mut stream = SparseJepaAnyUpPcaStream::new(
        pipeline,
        SparseJepaAnyUpPcaStreamConfig {
            queue_capacity: 4,
            batch_size: 2,
            ..SparseJepaAnyUpPcaStreamConfig::default()
        },
    )
    .expect("stream");
    stream
        .enqueue(frame_input(0, 2, vec![0, 3], grid, &device, &model_config))
        .expect("enqueue");
    let err = stream
        .enqueue(frame_input(0, 1, vec![0, 3], grid, &device, &model_config))
        .expect_err("sequence rewind should fail");
    assert!(err.to_string().contains("monotonically"));

    stream
        .enqueue(frame_input(1, 0, vec![1], grid, &device, &model_config))
        .expect("enqueue variable width");
    let err = stream
        .process_next_ready()
        .expect_err("variable-width front batch should fail");
    assert!(err.to_string().contains("variable sparse mask widths"));
}

#[test]
fn highres_pipeline_step_keeps_hot_path_device_resident() {
    let source = include_str!("../src/highres_pipeline.rs");
    let start = source
        .find("fn finish_encoded_batch_step")
        .expect("finish_encoded_batch_step");
    let end = source[start..]
        .find("struct StageTimer")
        .map(|offset| start + offset)
        .expect("stage timer");
    let hot_path = &source[start..end];

    for marker in [".to_data(", ".into_data(", "TensorData::new"] {
        assert!(
            !hot_path.contains(marker),
            "high-res sparse pipeline hot path should not contain {marker}"
        );
    }
}

#[test]
fn feature_memory_output_exposes_nchw_view() {
    let device = Default::default();
    let grid = TokenGridShape::new(1, 1, 2);
    let mut memory = InterframeJepaFeatureMemory::<B>::new(
        InterframeJepaFeatureMemoryConfig::default(),
        1,
        grid,
        2,
        &device,
    )
    .expect("memory");
    let mask = SparseTokenMask::all(grid.len());
    let output = memory
        .update_masked_tokens(
            Tensor::<B, 3>::ones([1, grid.len(), 2], &device),
            &mask,
            grid,
        )
        .expect("memory update");

    assert_eq!(
        output.features_nchw().expect("nchw").shape().dims::<4>(),
        [1, 2, 1, 2]
    );
}

fn values3(tensor: Tensor<B, 3>) -> Vec<f32> {
    tensor.to_data().to_vec::<f32>().expect("tensor values")
}

fn values4(tensor: Tensor<B, 4>) -> Vec<f32> {
    tensor.to_data().to_vec::<f32>().expect("tensor values")
}

fn component_dot(values: &[f32], feature_dim: usize, left: usize, right: usize) -> f32 {
    let output_channels = 3;
    (0..feature_dim)
        .map(|feature| {
            values[feature * output_channels + left] * values[feature * output_channels + right]
        })
        .sum()
}

fn tiny_pipeline(
    batch: usize,
    config: SparseJepaAnyUpPcaPipelineConfig,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
    model_config: &VJepaConfig,
) -> SparseJepaAnyUpPcaPipeline<B> {
    let jepa = VJepa2_1Model::<B>::new(model_config, device);
    let anyup = AnyUp::<B>::new(AnyUpConfig::tiny_for_tests(), device).expect("anyup");
    SparseJepaAnyUpPcaPipeline::<B>::new(
        jepa,
        anyup,
        model_config,
        config,
        batch,
        [model_config.image_size, model_config.image_size],
        device,
    )
    .expect("pipeline")
}

fn frame_input(
    stream_id: u64,
    sequence: u64,
    indices: Vec<usize>,
    grid: TokenGridShape,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
    model_config: &VJepaConfig,
) -> SparseJepaAnyUpPcaFrameInput<B> {
    SparseJepaAnyUpPcaFrameInput {
        id: SparseJepaAnyUpPcaFrameId {
            stream_id,
            sequence,
            capture_time_nanos: sequence,
        },
        image: Tensor::<B, 4>::ones(
            [1, 3, model_config.image_size, model_config.image_size],
            device,
        ),
        mask: SparseTokenMask::new(indices, grid.len()).expect("mask"),
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "value count mismatch");
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual_value - expected_value).abs() <= tolerance,
            "value {index}: expected {expected_value}, got {actual_value}; actual={actual:?}"
        );
    }
}
