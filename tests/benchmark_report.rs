use std::collections::BTreeSet;

#[derive(Debug)]
struct BenchRow {
    backend: String,
    resolution: String,
    density: String,
    context_tokens: usize,
    temporal_stream_ms: f64,
    temporal_e2e_ms: f64,
    e2e_fps: f64,
    trace_ms: f64,
}

#[test]
fn e2e_benchmark_report_has_required_matrix_and_trace_off_rows() {
    let report = include_str!("../docs/e2e-benchmark-results.md");
    let rows = parse_benchmark_rows(report);
    assert_eq!(rows.len(), 24, "expected ndarray and webgpu 3x4 matrices");

    let expected_backends = ["ndarray", "webgpu"];
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
            row.temporal_e2e_ms >= row.temporal_stream_ms,
            "E2E timing should include the stream timing: {row:?}"
        );
        assert!(row.e2e_fps > 0.0, "FPS must be positive: {row:?}");
        assert_eq!(
            row.trace_ms, 0.0,
            "checked-in E2E report should be trace-disabled: {row:?}"
        );
    }
}

#[test]
fn cuda_benchmark_path_documents_blocker_and_rejects_header_only_csv() {
    let report = include_str!("../docs/e2e-benchmark-results.md");
    assert!(report.contains("## CUDA Status"));
    assert!(report.contains("no defensible CUDA\nFPS rows from this environment"));
    assert!(report.contains("skipping autogaze-cuda benchmark"));

    let runbook = include_str!("../docs/cuda-benchmark.md");
    assert!(runbook.contains("nvidia-smi -L"));
    assert!(runbook.contains("The CSV has data rows, not just the header."));
    assert!(runbook.contains("autogaze_trace_ms` is `0.000`"));

    let workflow_template = include_str!("../docs/workflows/cuda-benchmark.yml");
    assert!(workflow_template.contains("BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS: cuda"));
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
        compact.contains("ifself.disabled(){return0.0;}"),
        "disabled tracing should return before entering the decoder/timing path"
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

fn parse_benchmark_rows(report: &str) -> Vec<BenchRow> {
    report
        .lines()
        .filter(|line| line.starts_with("| ndarray |") || line.starts_with("| webgpu |"))
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
        8,
        "unexpected benchmark table row shape: {line}"
    );

    BenchRow {
        backend: cols[0].to_string(),
        resolution: cols[1].to_string(),
        density: cols[2].to_string(),
        context_tokens: cols[3].parse().expect("context tokens"),
        temporal_stream_ms: cols[4].parse().expect("temporal stream ms"),
        temporal_e2e_ms: cols[5].parse().expect("temporal e2e ms"),
        e2e_fps: cols[6].parse().expect("e2e fps"),
        trace_ms: cols[7].parse().expect("trace ms"),
    }
}
