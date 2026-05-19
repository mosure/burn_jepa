use std::collections::BTreeSet;

#[derive(Debug)]
struct BenchRow {
    backend: String,
    resolution: String,
    density: String,
    context_tokens: usize,
    temporal_stream_ms: f64,
    rolling_stream_ms: f64,
    temporal_e2e_ms: f64,
    rolling_e2e_ms: f64,
    e2e_fps: f64,
    rolling_fps: f64,
    trace_ms: f64,
}

#[test]
fn e2e_benchmark_report_has_required_matrix_and_trace_off_rows() {
    let report = include_str!("../docs/e2e-benchmark-results.md");
    let rows = parse_benchmark_rows(report);
    assert_eq!(
        rows.len(),
        36,
        "expected ndarray, webgpu, and cuda 3x4 matrices"
    );

    let expected_backends = ["ndarray", "webgpu", "cuda"];
    let expected_resolutions = ["224x224", "384x384", "720p"];
    let expected_densities = ["0.0100", "0.0500", "0.1000", "0.2500"];

    let observed = rows
        .iter()
        .map(|row| {
            (
                row.backend.as_str(),
                row.resolution.as_str(),
                row.density.as_str(),
            )
        })
        .collect::<BTreeSet<_>>();

    for backend in expected_backends {
        for resolution in expected_resolutions {
            for density in expected_densities {
                assert!(
                    observed.contains(&(backend, resolution, density)),
                    "missing {backend} {resolution} density={density} row"
                );
            }
        }
    }

    for row in rows {
        assert!(
            row.context_tokens > 0,
            "context tokens must be positive: {row:?}"
        );
        assert!(
            row.temporal_stream_ms > 0.0,
            "temporal stream timing must be positive: {row:?}"
        );
        assert!(
            row.rolling_stream_ms > 0.0,
            "rolling stream timing must be positive: {row:?}"
        );
        assert!(
            row.temporal_e2e_ms >= row.temporal_stream_ms,
            "E2E timing should include the stream timing: {row:?}"
        );
        assert!(
            row.rolling_e2e_ms >= row.rolling_stream_ms,
            "rolling E2E timing should include the rolling stream timing: {row:?}"
        );
        assert!(
            row.rolling_e2e_ms < row.temporal_e2e_ms,
            "rolling window should be lower latency than the 4-frame clip path: {row:?}"
        );
        assert!(row.e2e_fps > 0.0, "FPS must be positive: {row:?}");
        assert!(
            row.rolling_fps > 0.0,
            "rolling FPS must be positive: {row:?}"
        );
        assert_eq!(
            row.trace_ms, 0.0,
            "checked-in E2E report should be trace-disabled: {row:?}"
        );
    }
}

#[test]
fn cuda_benchmark_path_documents_runtime_and_rejects_header_only_csv() {
    let report = include_str!("../docs/e2e-benchmark-results.md");
    assert!(report.contains("## CUDA Status"));
    assert!(report.contains("CUDA is now the fastest measured E2E lane"));
    assert!(report.contains("## Pure CUDA Sparse Patchify Smoke"));
    assert!(report.contains("sparse-patchify-cuda | 224x224 | 0.0500"));
    assert!(report.contains("CUDA training smoke completed successfully"));
    assert!(report.contains("96/96 CUDA trials"));
    assert!(report.contains("header-only CSV rejection"));
    assert!(report.contains("RTX PRO 6000 Blackwell Workstation Edition"));

    let runbook = include_str!("../docs/cuda-benchmark.md");
    assert!(runbook.contains("nvidia-smi -L"));
    assert!(runbook.contains("nvidia-smi` alone is not sufficient evidence"));
    assert!(runbook.contains("nvidia-smi -L sees"));
    assert!(runbook.contains("probe\nfailure details"));
    assert!(runbook.contains("CUDA runtime cannot open a device without NVIDIA character devices"));
    assert!(runbook.contains("The CSV has data rows, not just the header."));
    assert!(
        runbook.contains("cargo check --no-default-features --features cuda,sparse-patchify-cuda")
    );
    assert!(runbook.contains("BURN_JEPA_PIPELINE_BENCH_DENSE_PATCHIFY=0"));
    assert!(runbook.contains("autogaze_trace_ms` is `0.000`"));

    let workflow_template = include_str!("../docs/workflows/cuda-benchmark.yml");
    assert!(workflow_template.contains("BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS: cuda"));
    assert!(workflow_template.contains("BURN_JEPA_PIPELINE_JEPA_BACKENDS: sparse-patchify-cuda"));
    assert!(
        workflow_template
            .contains("cargo check --no-default-features --features cuda,sparse-patchify-cuda")
    );
    assert!(workflow_template.contains("BURN_JEPA_PIPELINE_BENCH_TRACE"));
    assert!(workflow_template.contains("CUDA benchmark produced no data rows"));
    assert!(workflow_template.contains("if [ \"$rows\" -le 1 ]; then"));
}

#[test]
fn benchmark_trace_config_is_opt_in_and_disabled_path_avoids_tensor_clone() {
    let bench = include_str!("../benches/autogaze_sparse_jepa_pipeline.rs");
    let compact = compact_source(bench);

    assert!(compact.contains("constENV:&'staticstr=\"BURN_JEPA_PIPELINE_BENCH_TRACE\";"));
    assert!(
        compact.contains("env_bool(Self::ENV,false)"),
        "trace collection should be disabled unless the benchmark config opts in"
    );
    assert!(
        compact.contains("video:&Tensor<B,5>"),
        "trace measurement should borrow the video so disabled tracing does not clone tensors"
    );
    assert!(
        compact.contains("letautogaze_trace_ms=ifbench_config.trace.enabled(){measure_autogaze_trace_ms(&autogaze,&ag_video,"),
        "trace measurement should only be called from an enabled config branch"
    );
    let trace_branch = compact
        .find("letautogaze_trace_ms=ifbench_config.trace.enabled(){")
        .expect("trace config branch");
    let disabled_branch = compact[trace_branch..]
        .find("}else{0.0};")
        .map(|offset| trace_branch + offset)
        .expect("disabled trace branch");
    let trace_helper = compact
        .find("fnmeasure_autogaze_trace_ms")
        .expect("trace measurement helper");
    let trace_decoder = compact[trace_helper..]
        .find("trace_video_with_mode(")
        .map(|offset| trace_helper + offset)
        .expect("trace decoder call");
    let eager_clone = compact[trace_helper..]
        .find("video.clone()")
        .map(|offset| trace_helper + offset)
        .expect("trace tensor clone");
    assert!(
        trace_decoder < trace_branch && eager_clone < trace_branch,
        "trace decoder work should only live in the helper that the disabled branch skips"
    );
    assert!(
        trace_branch < disabled_branch,
        "trace config branch should have an explicit zero-cost disabled arm"
    );
    assert!(
        compact.contains("measure_autogaze_trace_ms(&autogaze,&ag_video,"),
        "the call site should pass a borrowed tensor"
    );
    assert!(
        !compact.contains("measure_autogaze_trace_ms(&autogaze,ag_video.clone(),"),
        "disabled trace config must not pay for an eager tensor clone at the call site"
    );
}

#[test]
fn ttt_training_benchmark_has_sparse_density_training_step_matrix() {
    let bench = include_str!("../benches/ttt_training.rs");
    let compact = compact_source(bench);

    assert!(bench.contains("SPARSITY_DENSITY_CASES"));
    assert!(bench.contains("label: \"10pct\""));
    assert!(bench.contains("density: 0.10"));
    assert!(bench.contains("label: \"50pct\""));
    assert!(bench.contains("density: 0.50"));
    assert!(bench.contains("label: \"100pct\""));
    assert!(bench.contains("density: 1.00"));
    assert!(
        bench.contains("ttt_sparsity_training_step_"),
        "TTT Criterion benches should expose an explicit sparsity sweep group"
    );
    assert!(
        bench.contains("dense_seq_b{batch_size}")
            && bench.contains("dense_chunked_b{batch_size}")
            && bench.contains("fixed_width_sparse_seq_b{batch_size}")
            && bench.contains("fixed_width_sparse_chunked_b{batch_size}"),
        "TTT training-step benches should compare sequential vs chunked recurrent rollout"
    );
    assert!(
        bench.contains("Throughput::Elements"),
        "Criterion output should include sample-throughput context"
    );
    assert!(
        bench.contains("density_{}_sparse_b{batch_size}_tokens{keep_tokens}_of{dense_tokens}"),
        "sparse rows should encode density, batch size, and token count in the benchmark id"
    );
    assert!(
        bench.contains("density_100pct_dense_b{batch_size}_tokens{dense_tokens}_of{dense_tokens}"),
        "matrix should include a normal dense full-token baseline"
    );
    assert!(
        compact.contains("forward_single_frame_rollout_sparse_batch("),
        "sparsity sweep should exercise fixed-width per-sample sparse TTT rollout"
    );
    assert!(
        bench.contains("ttt_sparse_patchify_sparsity_training_step_"),
        "TTT Criterion benches should expose a flex-gmm sparse-patchify training-step group"
    );
    assert!(
        compact.contains("student_frozen_sparse_patchify_rollout_batch("),
        "sparse-patchify sweep should skip dense patch embedding in the sparse training rows"
    );
    assert!(
        bench.contains(
            "density_{}_sparse_patchify_b{batch_size}_tokens{keep_tokens}_of{dense_tokens}"
        ),
        "sparse-patchify rows should encode density, batch size, and token count in the benchmark id"
    );
    assert!(
        compact.contains("forward_single_frame_rollout("),
        "sparsity sweep should include the dense TTT rollout baseline"
    );
    assert!(
        compact.contains("loss.backward()"),
        "matrix should measure a full training step, not only forward latency"
    );
    assert!(
        compact.contains("sparse_optim.step(") && compact.contains("dense_optim.step("),
        "matrix should include optimizer updates for sparse and dense rows"
    );
    assert!(
        bench.contains("TBPTT_STREAM_CASES"),
        "TTT benches should expose a TBPTT/reset/decay sweep"
    );
    assert!(bench.contains("tbptt_carry4_decay0_97"));
    assert!(bench.contains("tbptt_carry4_decay0_90"));
    assert!(
        bench.contains("ttt_training_step_matrix_flex")
            && bench.contains("ttt_sparsity_training_step_matrix_flex"),
        "TTT Criterion benches should include Burn 0.21 flex backend lanes"
    );
    assert!(
        bench.contains("ttt_training_step_matrix_dispatch_flex")
            && bench.contains("ttt_training_step_matrix_dispatch_wgpu")
            && bench.contains("ttt_training_step_matrix_dispatch_cuda"),
        "TTT Criterion benches should include Burn 0.21 dispatch backend lanes"
    );
    assert!(
        compact.contains("DispatchDevice::autodiff("),
        "dispatch benches should select concrete runtime devices through DispatchDevice"
    );
    assert!(
        bench.contains("for batch_size in [1usize, 4]"),
        "TBPTT bench should sweep single-stream and packed multi-stream batches"
    );
    assert!(
        bench.contains("ttt_tbptt_training_step_"),
        "Criterion benches should include a TBPTT training-step group"
    );
    assert!(
        compact.contains("state.detach();") && compact.contains("state.decay(case.state_decay);"),
        "TBPTT bench should include detach and decay overhead in the measured step"
    );
}

#[test]
fn cargo_features_expose_burn_021_flex_and_dispatch_lanes() {
    let manifest = include_str!("../Cargo.toml");

    assert!(
        manifest.contains(
            "flex = [\"burn/flex\", \"burn_anyup/flex\", \"burn_jepa_reconstruction/flex\"]"
        ),
        "root flex feature should cover all Burn-backed model crates"
    );
    assert!(
        manifest.contains(
            "dispatch = [\"burn/dispatch\", \"burn_anyup/dispatch\", \"burn_jepa_reconstruction/dispatch\"]"
        ),
        "root dispatch feature should cover all Burn-backed model crates"
    );
    assert!(
        manifest.contains("features = [\"autodiff\", \"fusion\", \"std\"]"),
        "Burn dependency should keep fusion enabled for GPU/backend benches"
    );
    assert!(
        manifest.contains("required-features = [\"ndarray\"]"),
        "ndarray-only benches should stay feature-gated so flex/dispatch bench builds are narrow"
    );
}

#[test]
fn feature_memory_benchmark_covers_sparse_update_density_matrix() {
    let bench = include_str!("../benches/feature_memory.rs");
    let compact = compact_source(bench);

    assert!(bench.contains("FEATURE_MEMORY_CASES"));
    assert!(bench.contains("vjepa224_b4"));
    assert!(bench.contains("vjepa384_b1"));
    assert!(bench.contains("label: \"1pct\""));
    assert!(bench.contains("label: \"50pct\""));
    assert!(bench.contains("label: \"30pct\""));
    assert!(bench.contains("label: \"100pct\""));
    assert!(
        bench.contains("feature_memory_cached_update_"),
        "feature memory benches should measure cached sparse update latency"
    );
    assert!(
        bench.contains("feature_memory_plan_build_update_"),
        "feature memory benches should expose first-update plan build overhead"
    );
    assert!(
        bench.contains("feature_memory_dense_ordered_update_"),
        "feature memory benches should compare high-density sparse updates to the dense ordered path"
    );
    assert!(
        bench.contains("feature_memory_row_reset_"),
        "feature memory benches should measure packed-stream row resets"
    );
    assert!(
        bench.contains("feature_memory_tiled_sparse_assign_wgpu")
            && bench.contains("feature_memory_tiled_sparse_assign_cuda"),
        "feature memory benches should include backend-specific tiled sparse assignment lanes"
    );
    assert!(
        compact.contains(".update_tokens(black_box(tokens),black_box(indices),"),
        "bench should exercise sparse feature update with caller-owned sparse tokens and indices"
    );
    assert!(
        compact.contains(".update_dense_ordered_tokens(black_box(tokens),case.grid)"),
        "bench should exercise the full-token dense ordered cache update path"
    );
    assert!(
        compact.contains("Throughput::Elements((case.batch*keep)asu64)"),
        "cached sparse-update rows should report sparse token throughput"
    );
    assert!(
        bench.contains("feature_memory_flex")
            && bench.contains("feature_memory_dispatch_flex")
            && bench.contains("feature_memory_dispatch_wgpu")
            && bench.contains("feature_memory_dispatch_cuda"),
        "feature memory benches should include flex and dispatch backend lanes"
    );

    let docs = include_str!("../docs/interframe-feature-memory.md");
    assert!(docs.contains("InterframeJepaFeatureMemory"));
    assert!(docs.contains("features`, `observed`, and"));
    assert!(docs.contains("scatter(..., Add)"));
    assert!(docs.contains("tiled sparse assignment"));
    assert!(docs.contains("cargo bench --bench feature_memory"));

    let manifest = include_str!("../Cargo.toml");
    assert!(manifest.contains("name = \"feature_memory\""));
    assert!(manifest.contains("sparse-feature-memory-wgpu"));
    assert!(manifest.contains("sparse-feature-memory-cuda"));
}

#[test]
fn highres_anyup_pca_pipeline_has_modular_bench_and_docs() {
    let bench = include_str!("../benches/highres_anyup_pca_pipeline.rs");
    let compact = bench.split_whitespace().collect::<String>();

    assert!(
        bench.contains("highres_pca_project_"),
        "high-res bench should isolate PCA display projection"
    );
    assert!(
        bench.contains("highres_anyup_from_token_cache_"),
        "high-res bench should isolate AnyUp from the token cache"
    );
    assert!(
        bench.contains("highres_sparse_cache_anyup_pca_"),
        "high-res bench should measure sparse cache update plus AnyUp plus PCA"
    );
    assert!(
        bench.contains("highres_sparse_jepa_anyup_pca_e2e_"),
        "high-res bench should include an end-to-end sparse JEPA path"
    );
    assert!(
        bench.contains("highres_jepa_cache_density_sweep_") && bench.contains("0.98"),
        "high-res bench should measure the near-dense JEPA cache crossover"
    );
    assert!(
        bench.contains("highres_sparse_patchify_jepa_anyup_pca_e2e_wgpu"),
        "high-res bench should include the WGPU flex-gmm sparse patchify E2E path"
    );
    assert!(
        bench.contains("highres_sparse_patchify_jepa_anyup_pca_e2e_cuda"),
        "high-res bench should include the CUDA flex-gmm sparse patchify E2E path"
    );
    assert!(
        bench.contains("highres_inflight_stream_"),
        "high-res bench should include bounded in-flight stream batching"
    );
    assert!(
        bench.contains("highres_inflight_stream_cached_mask_"),
        "high-res bench should isolate cached sparse-mask stream batching"
    );
    assert!(
        bench.contains("for batch_size in [1usize, 2, 4]"),
        "in-flight stream bench should sweep multiple frame batch sizes"
    );
    assert!(
        compact.contains("jepa_feature_tokens_to_nchw(tokens,grid)"),
        "bench should use the shared token-cache NCHW view helper"
    );
    assert!(
        compact.contains(".project_nchw_display(black_box(high_res))"),
        "bench should measure the shared PCA display projector"
    );
    assert!(
        bench.contains("BURN_JEPA_HIGHRES_BENCH_LARGE"),
        "bench should keep large JEPA-like cases opt-in"
    );
    let pipeline = include_str!("../src/highres_pipeline.rs");
    assert!(
        pipeline.contains("cached_sparse_mask_batch") && pipeline.contains("CachedSparseMaskBatch"),
        "in-flight stream should cache repeated sparse mask batches"
    );

    let docs = include_str!("../docs/highres-anyup-pca-pipeline.md");
    assert!(docs.contains("FeatureFramePipeline"));
    assert!(docs.contains("FeatureFrameStream"));
    assert!(docs.contains("FeatureFrameRequest"));
    assert!(docs.contains("FeatureFrameSchedule"));
    assert!(docs.contains("backpressure"));
    assert!(docs.contains("overwrite_newest"));
    assert!(docs.contains("monotonic per-stream sequence"));
    assert!(docs.contains("FeaturePcaProjector"));
    assert!(docs.contains("backend-to-host reads"));
    assert!(docs.contains("step_image_with_mask_batch_nodes_measured"));
    assert!(docs.contains("process_next_ready_nodes"));
    assert!(docs.contains("step_image_with_mask_sparse_patchify_wgpu"));
    assert!(docs.contains("step_image_with_mask_sparse_patchify_cuda"));
    assert!(docs.contains("step_image_with_sparse_patchify_plan_wgpu_measured"));
    assert!(docs.contains("SparsePatchifyBatchPlan"));
    assert!(docs.contains("encode_path"));
    assert!(docs.contains("BURN_JEPA_HIGHRES_BENCH_LARGE=1"));

    let manifest = include_str!("../Cargo.toml");
    assert!(manifest.contains("name = \"highres_anyup_pca_pipeline\""));
}

#[test]
fn bevy_viewer_benchmark_aligns_with_raw_pipeline_metrics() {
    let viewer_bench = include_str!("../crates/bevy_jepa/benches/viewer_pipeline.rs");
    let fps_stability = include_str!("../crates/bevy_jepa/examples/fps_stability.rs");
    let viewer_config = include_str!("../crates/bevy_jepa/src/config.rs");
    let viewer_lib = include_str!("../crates/bevy_jepa/src/lib.rs");
    let viewer_mask = include_str!("../crates/bevy_jepa/src/mask.rs");
    let viewer_metrics = include_str!("../crates/bevy_jepa/src/metrics.rs");
    let viewer_platform = include_str!("../crates/bevy_jepa/src/platform.rs");
    let viewer_docs = include_str!("../crates/bevy_jepa/README.md");
    let viewer_policy = include_str!("../src/viewer.rs");
    let raw_bench = include_str!("../benches/highres_anyup_pca_pipeline.rs");
    let highres_docs = include_str!("../docs/highres-anyup-pca-pipeline.md");
    let preprint = include_str!("../docs/papers/vjepa21_ttt_sparse_temporal_preprint.tex");
    let root_manifest = include_str!("../Cargo.toml");
    let manifest = include_str!("../crates/bevy_jepa/Cargo.toml");
    let pca = include_str!("../src/pca.rs");
    let pca_runtime = pca.split("#[cfg(test)]").next().unwrap_or(pca);
    let pca_bench = include_str!("../benches/highres_anyup_pca_pipeline.rs");
    let workflow = include_str!("../.github/workflows/test.yml");
    let deploy_workflow = include_str!("../.github/workflows/deploy-pages.yml");
    let safetensors = include_str!("../src/safetensors_io.rs");
    let experiment = include_str!("../src/experiment.rs");

    assert!(viewer_bench.contains("bevy_jepa_viewer_pipeline_wgpu"));
    assert!(viewer_bench.contains("low_res_cache_update"));
    assert!(viewer_bench.contains("pca_projection"));
    assert!(viewer_bench.contains("full_anyup_decode"));
    assert!(viewer_bench.contains("display_upload_gpu"));
    assert!(viewer_bench.contains("display_upload_cpu"));
    assert!(viewer_bench.contains("patch_diff_t003"));
    assert!(viewer_bench.contains("patch_diff_t000"));
    assert!(viewer_bench.contains("patch_diff_threshold: 0.03"));
    assert!(viewer_bench.contains("patch_diff_threshold: 0.0"));
    assert!(viewer_bench.contains("sync_measurements: true"));
    assert!(viewer_bench.contains("step_with_display_request"));
    assert!(viewer_bench.contains("step_with_stage_request"));
    assert!(viewer_bench.contains("aligns_with_stage_metrics"));
    assert!(viewer_bench.contains("BevyJepaMaskSource::PatchDiff"));
    assert!(
        !viewer_bench.contains("BevyJepaMaskSource::Autogaze"),
        "viewer bench must not time the reserved AutoGaze mode unless it is backed by a real model node"
    );
    assert!(viewer_bench.contains("mask_case.label"));
    assert!(viewer_policy.contains("DEFAULT_PATCH_DIFF_DENSE_FALLBACK_DENSITY"));
    assert!(viewer_policy.contains("pub struct FeatureFrameViewerConfig"));
    assert!(viewer_policy.contains("pub fn patch_diff_dense_fallback"));
    assert!(viewer_policy.contains("pub fn bucket_sparse_mask"));
    assert!(viewer_policy.contains("pub fn finalize_patch_diff_masks"));
    assert!(viewer_policy.contains("pub fn patch_diff_sampled_dense_fast_path_from_rgba"));
    assert!(viewer_policy.contains("pub fn patch_diff_scores_from_rgba"));
    assert!(viewer_policy.contains("pub fn patch_diff_can_use_dense_fast_path"));
    assert!(viewer_mask.contains("patch_diff_sampled_dense_fast_path_from_rgba"));
    assert!(viewer_mask.contains("patch_diff_scores_from_rgba"));
    assert!(viewer_mask.contains("patch_diff_can_use_dense_fast_path"));
    assert!(viewer_mask.contains("patch_diff_dense_fallback_density"));
    assert!(!viewer_mask.contains("fn patch_diff_dense_fallback"));
    assert!(!viewer_mask.contains("fn bucket_sparse_mask"));
    assert!(viewer_lib.contains("DEFAULT_SPARSE_MASK_BUCKET_TOKENS"));
    assert!(viewer_metrics.contains("pub stage_metrics: FeatureFrameMetrics"));
    assert!(viewer_metrics.contains("pub encode_path: FeatureFrameEncodePath"));
    assert!(viewer_metrics.contains("pub fn aligns_with_stage_metrics(&self) -> bool"));
    assert!(viewer_lib.contains("pub struct BevyJepaHeadlessPipeline"));
    assert!(viewer_lib.contains("pub fn step_stage_only(&mut self)"));
    assert!(viewer_lib.contains("pub fn step_with_display_panels(&mut self)"));
    assert!(viewer_lib.contains("pub fn step_with_display_request"));
    assert!(viewer_lib.contains("pub fn step_with_stage_request"));
    assert!(viewer_lib.contains("stage_request_for_frame"));
    assert!(viewer_lib.contains("run_feature_frame_pipeline_sparse_patchify"));
    assert!(viewer_lib.contains("pub mod platform"));
    assert!(viewer_lib.contains("AsyncComputeTaskPool"));
    assert!(viewer_lib.contains("pending_stage"));
    assert!(viewer_lib.contains("pending_stage_image"));
    assert!(viewer_lib.contains("prev_stage_image"));
    assert!(viewer_lib.contains("source_stage_image"));
    assert!(viewer_lib.contains("sparse_mask_to_rgba_tensor"));
    assert!(viewer_lib.contains("apply_input_panel_to_world"));
    assert!(viewer_lib.contains("apply_stage_panels_to_world"));
    assert!(viewer_lib.contains("FeatureFrameRequest::low_res()"));
    assert!(viewer_lib.contains("queue_overwritten_frames"));
    assert!(viewer_platform.contains("frame_input"));
    assert!(viewer_platform.contains("native_camera_thread_with_request"));
    assert!(viewer_config.contains("pub enum BevyJepaFrameSource"));
    assert!(viewer_config.contains("pub type BevyJepaEncodePath = FeatureFrameEncodeRoute"));
    assert!(
        viewer_config.contains("pub type BevyJepaSparseEncodeMode = FeatureFrameSparseEncodeMode")
    );
    assert!(viewer_config.contains("pub pipeline: FeatureFrameViewerConfig"));
    assert!(viewer_config.contains("impl Deref for BevyJepaConfig"));
    assert!(viewer_policy.contains("MIN_PIPELINE_IMAGE_SIZE: usize = 256"));
    assert!(viewer_policy.contains("DEFAULT_IMAGE_SIZE: usize = 512"));
    assert!(viewer_config.contains("target/burn-jepa-web/model/vjepa2_1_ttt/manifest.json"));
    assert!(viewer_docs.contains("BURN_JEPA_MODEL_MANIFEST"));
    assert!(viewer_config.contains("~/.cache/burn_jepa/vjepa2_1_vitb_dist_vitG_384"));
    assert!(viewer_policy.contains("DEFAULT_HIGH_RES_PCA_EVERY: u64 = 0"));
    assert!(viewer_policy.contains("DEFAULT_PCA_UPDATE_EVERY: u64 = 1"));
    assert!(viewer_policy.contains("DEFAULT_PCA_SAMPLE_WINDOW_FRAMES: usize = 16"));
    assert!(viewer_policy.contains("DEFAULT_PCA_MIN_SAMPLE_FRAMES: usize = 2"));
    assert!(viewer_policy.contains("DEFAULT_PATCH_DIFF_QUALITY: f32 = 0.97"));
    assert!(viewer_policy.contains("DEFAULT_SPARSE_MASK_BUCKET_TOKENS: usize = 256"));
    assert!(viewer_policy.contains("DEFAULT_MIN_CONTEXT_DENSITY: f32 = 0.0"));
    assert!(viewer_policy.contains("pub enum FeatureFrameSparseEncodeMode"));
    assert!(viewer_policy.contains("Exact"));
    assert!(viewer_policy.contains("pub fn pipeline_image_size(&self) -> usize"));
    assert!(viewer_policy.contains("pub fn patch_diff_quality(&self) -> f32"));
    assert!(
        viewer_policy
            .contains("DEFAULT_PATCH_DIFF_THRESHOLD: f32 = 1.0 - DEFAULT_PATCH_DIFF_QUALITY")
    );
    assert!(viewer_lib.contains("DEFAULT_PATCH_DIFF_QUALITY"));
    assert!(viewer_policy.contains("DEFAULT_PREWARM_SHAPE_BUCKETS: bool = true"));
    assert!(viewer_lib.contains("prewarm_feature_frame_shapes"));
    assert!(fps_stability.contains("fps-stability-summary.csv"));
    assert!(fps_stability.contains("unique_encode_widths"));
    assert!(fps_stability.contains("p95_outer_ms"));
    assert!(viewer_lib.contains("not(feature = \"wasm-fusion\")"));
    assert!(viewer_lib.contains("CubeBackend<burn::backend::wgpu::WgpuRuntime"));
    assert!(
        !root_manifest.contains("wgpu = [\"burn/wgpu\", \"burn-store/wgpu\""),
        "wasm sparse WGPU must not pull burn-store/wgpu because that enables burn-wgpu default autotune"
    );
    assert!(
        root_manifest.contains("burn_flex_gmm = { version = \"0.21.2\""),
        "burn_flex_gmm 0.21.2 keeps wasm sparse patchify off burn-wgpu fusion/default"
    );
    assert!(manifest.contains("default = [\"sparse-patchify-wgpu\"]"));
    assert!(
        manifest.contains("sparse-patchify-wgpu = [\"burn_jepa/sparse-patchify-wgpu\"]"),
        "bevy_jepa sparse patchify must not enable browser Burn fusion implicitly"
    );
    assert!(
        manifest.contains(
            "bevy_burn = { path = \"../bevy_burn\", default-features = false, features = [\"fusion\"] }"
        ),
        "native bevy_jepa builds should still enable Bevy/Burn fusion through target-specific dependencies"
    );
    assert!(
        viewer_bench.contains("fn viewer_image_sizes() -> Vec<usize>")
            && viewer_bench.contains("BURN_JEPA_BENCH_1024")
    );
    assert!(viewer_docs.contains("--source camera"));
    assert!(viewer_docs.contains("default `--high-res-pca-every 0`"));
    assert!(viewer_docs.contains("separate AnyUp worker"));
    assert!(viewer_docs.contains("latest-frame overwrite slot"));
    assert!(viewer_docs.contains("--pca-update-every 1"));
    assert!(viewer_docs.contains("16-frame sample window"));
    assert!(viewer_docs.contains("quality `0.97`, threshold `0.03`"));
    assert!(viewer_docs.contains("quality value only changes the threshold"));
    assert!(viewer_docs.contains("patch-diff-dense-fallback-density 0.60"));
    assert!(viewer_docs.contains("shape-stable bucketed sparse encode"));
    assert!(viewer_docs.contains("bucketed-context"));
    assert!(viewer_docs.contains("FeatureFrameViewerConfig"));
    assert!(viewer_docs.contains("fixed 97%"));
    assert!(viewer_docs.contains("uniform global RGB/luma shifts"));
    assert!(
        raw_bench.contains("viewer256_sparse100") && raw_bench.contains("viewer512_sparse100"),
        "raw high-res E2E bench should include both supported V-JEPA 2.1 viewer resolutions"
    );
    assert!(
        raw_bench.contains("FeaturePcaUpdateConfig::rolling_low_res_every")
            && raw_bench.contains("SparseJepaAnyUpPcaMeasurementConfig::enabled()"),
        "raw viewer row should mirror Bevy's rolling PCA and stage-measurement config"
    );
    assert!(pca.contains("pub enum FeaturePcaDisplayMode"));
    assert!(pca.contains("SemanticRgb"));
    assert!(pca.contains("display_center"));
    assert!(pca.contains("display_spread"));
    assert!(
        !pca_runtime.contains("into_data()") && !pca_runtime.contains("to_vec::<"),
        "PCA visualization must not require host readback for display statistics"
    );
    assert!(pca_bench.contains("highres_semantic_pca_stats_update"));
    assert!(pca_bench.contains("FeaturePcaDisplayMode::SemanticRgb"));
    assert!(
        viewer_docs.contains("highres_sparse_jepa_anyup_pca_e2e_wgpu/viewer512_sparse100")
            && highres_docs.contains("viewer256_sparse100")
            && highres_docs.contains("viewer512_sparse100"),
        "docs should tell users how to compare Bevy wrapper and raw pipeline rows"
    );
    assert!(preprint.contains("\\usepackage{pgfplots}"));
    assert!(preprint.contains("\\label{fig:sparsity-latency}"));
    assert!(preprint.contains("(85,3.218)"));
    assert!(preprint.contains("100\\% point is the dense ordered baseline"));
    assert!(workflow.contains("cargo fmt --all -- --check"));
    assert!(
        workflow
            .contains("cargo test -p burn_jepa --locked --no-default-features --features ndarray")
    );
    assert!(workflow.contains("cargo test -p burn_anyup -p bevy_burn --locked"));
    assert!(workflow.contains("cargo test --locked --test benchmark_report"));
    assert!(workflow.contains("cargo package -p burn_anyup --locked"));
    assert!(workflow.contains("cargo check -p bevy_jepa --locked --features sparse-patchify-wgpu"));
    assert!(
        deploy_workflow
            .contains("cargo build -p bevy_jepa --target wasm32-unknown-unknown --release"),
        "Pages should build the default sparse-patchify wasm viewer lane"
    );
    for source in [viewer_config, safetensors, experiment] {
        assert!(
            !source.contains("/home/mosure"),
            "source defaults must not bake in machine-specific absolute cache paths"
        );
    }
    assert!(manifest.contains("name = \"viewer_pipeline\""));
}

#[test]
fn e2e_benchmark_reuses_library_projection_and_patchify_core() {
    let bench = include_str!("../benches/autogaze_sparse_jepa_pipeline.rs");

    assert!(
        bench.contains("project_generated_tokens"),
        "E2E bench should use the sparse window plan AutoGaze projection helper"
    );
    assert!(
        bench.contains("project_generated_masks"),
        "E2E bench should use the sparse window plan direct mask projection helper"
    );
    let autogaze = include_str!("../src/autogaze.rs");
    assert!(
        autogaze.contains("pub fn autogaze_frame_tokens"),
        "library should keep generated-token readout centralized"
    );
    assert!(
        autogaze.contains("pub fn autogaze_frame_token_pairs"),
        "library should expose generated-token iteration without frame-token allocation"
    );
    assert!(
        autogaze.contains("pub fn project_autogaze_generated_masks"),
        "library should expose direct generated-token-to-mask projection"
    );
    assert!(
        bench.contains("sparse_patchify_video_wgpu"),
        "E2E bench should use the model-owned sparse patchify helper"
    );
    assert!(
        bench.contains("sparse_patchify_video_cuda"),
        "E2E bench should expose the CUDA sparse patchify helper"
    );
    assert!(
        bench.contains("BURN_JEPA_PIPELINE_JEPA_BACKENDS"),
        "E2E bench should allow selecting the sparse JEPA backend independently"
    );
    assert!(
        !bench.contains("compile_error!"),
        "E2E bench should skip cleanly when all-targets checks compile it without sparse-patchify features"
    );
    assert!(
        autogaze.contains("autogaze_sparse_top_k_for_context"),
        "library should keep density-aware AutoGaze top-k selection centralized"
    );
    assert!(
        bench.contains("AutogazeSparseJepaWindowConfig"),
        "E2E bench should reuse the library sparse window planner"
    );
    assert!(
        autogaze.contains("autogaze_sparse_generation_budget"),
        "sparse window planning should centralize AutoGaze generation budget selection"
    );
    assert!(
        bench.contains(".generate_streaming("),
        "E2E bench should measure streaming AutoGaze generation through the shared sparse window plan"
    );
    assert!(
        autogaze.contains("pub fn generate_streaming"),
        "sparse window planning should expose the shared streaming AutoGaze generation helper"
    );
    assert!(
        !bench.contains("SparsePatchify3dConfig {"),
        "E2E bench should not duplicate sparse patchify kernel setup"
    );
    assert!(
        !bench.contains("fn generated_frame_tokens"),
        "E2E bench should not keep local AutoGaze token projection logic"
    );
    assert!(
        !bench.contains(".max_gaze_tokens_each_frame().max("),
        "E2E bench should not force sparse rows to generate the maximum AutoGaze token budget"
    );
    assert!(
        !bench.contains("fn density_top_k"),
        "E2E bench should not estimate per-frame AutoGaze top-k from dense V-JEPA tokens directly"
    );

    let manifest = include_str!("../Cargo.toml");
    assert!(manifest.contains("burn_autogaze = { version = \"0.21.6\""));
    assert!(manifest.contains("burn_flex_gmm = { version = \"0.21.2\""));
    assert!(manifest.contains("sparse-patchify-cuda"));
    assert!(manifest.contains("autogaze-webgpu"));
    assert!(manifest.contains("autogaze-cuda"));

    let report = include_str!("../docs/e2e-benchmark-results.md");
    assert!(report.contains("## 720p Stage Metrics"));
    assert!(report.contains("Mask project"));
    assert!(report.contains("Rolling mask stream"));
    assert!(report.contains("default top-k overfetch is\nnow 1.0"));
    assert!(report.contains("Rolling AG cached"));
    assert!(report.contains("Exact-budget top-k"));
}

#[test]
fn benchmark_cuda_channel_failure_reports_runtime_diagnostic() {
    let bench = include_str!("../benches/autogaze_sparse_jepa_pipeline.rs");

    assert!(
        bench.contains("fn optional_backend_skip_reason"),
        "benchmark should centralize optional backend skip handling"
    );
    assert!(
        bench.contains("CUDA worker thread failed before returning results"),
        "forced CUDA runtime failures should not collapse to an opaque RecvError"
    );
    assert!(
        bench.contains("fn cuda_missing_device_nodes_reason"),
        "CUDA preflight should explain driver-visible but device-node-hidden states"
    );
    assert!(
        bench.contains("nvidia-smi -L sees"),
        "CUDA preflight should report when NVML can see a GPU"
    );
    assert!(
        bench.contains("verify /dev/nvidia* device nodes are visible"),
        "CUDA skip diagnostic should point at the runtime/device-node blocker"
    );

    let runbook = include_str!("../docs/cuda-benchmark.md");
    assert!(runbook.contains("CUDA runtime cannot open a device without NVIDIA character devices"));
    assert!(runbook.contains("CSV has data rows, not just the header"));
}

fn parse_benchmark_rows(report: &str) -> Vec<BenchRow> {
    report
        .lines()
        .filter(|line| {
            line.starts_with("| ndarray |")
                || line.starts_with("| webgpu |")
                || line.starts_with("| cuda |")
        })
        .filter(|line| {
            line.contains("| 224x224 |")
                || line.contains("| 384x384 |")
                || line.contains("| 720p |")
        })
        .map(parse_benchmark_row)
        .collect()
}

fn compact_source(source: &str) -> String {
    source.split_whitespace().collect()
}

fn parse_benchmark_row(line: &str) -> BenchRow {
    let cols = line
        .trim_matches('|')
        .split('|')
        .map(str::trim)
        .collect::<Vec<_>>();
    assert_eq!(
        cols.len(),
        11,
        "unexpected benchmark table row shape: {line}"
    );

    BenchRow {
        backend: cols[0].to_string(),
        resolution: cols[1].to_string(),
        density: cols[2].to_string(),
        context_tokens: cols[3].parse().expect("context tokens"),
        temporal_stream_ms: cols[4].parse().expect("temporal stream ms"),
        rolling_stream_ms: cols[5].parse().expect("rolling stream ms"),
        temporal_e2e_ms: cols[6].parse().expect("temporal e2e ms"),
        rolling_e2e_ms: cols[7].parse().expect("rolling e2e ms"),
        e2e_fps: cols[8].parse().expect("e2e fps"),
        rolling_fps: cols[9].parse().expect("rolling fps"),
        trace_ms: cols[10].parse().expect("trace ms"),
    }
}
