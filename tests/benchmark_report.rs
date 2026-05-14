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
    assert!(runbook.contains("BURN_JEPA_PIPELINE_BENCH_DENSE_PATCHIFY=0"));
    assert!(runbook.contains("autogaze_trace_ms` is `0.000`"));

    let workflow_template = include_str!("../docs/workflows/cuda-benchmark.yml");
    assert!(workflow_template.contains("BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS: cuda"));
    assert!(workflow_template.contains("BURN_JEPA_PIPELINE_JEPA_BACKENDS: sparse-patchify-cuda"));
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
    assert!(manifest.contains("burn_autogaze = { version = \"0.21.5\""));
    assert!(manifest.contains("burn_flex_gmm = { version = \"0.21.1\""));
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
