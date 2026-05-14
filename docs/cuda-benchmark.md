# CUDA E2E Benchmark Runbook

The CUDA throughput gate needs a runner with visible NVIDIA device nodes and a
working driver. CPU-only CI can compile the CUDA feature set, but it cannot
produce defensible FPS rows.

When workflow-file write permission is available, copy
`docs/workflows/cuda-benchmark.yml` to `.github/workflows/cuda-benchmark.yml`.
That template defines a manual `cuda benchmark` GitHub Actions workflow for a
self-hosted CUDA runner. The default runner labels are:

```json
["self-hosted","linux","x64","cuda"]
```

The workflow runs:

```sh
cargo check --no-default-features --features cuda

BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS=cuda \
BURN_JEPA_PIPELINE_JEPA_BACKENDS=sparse-patchify-cuda \
BURN_JEPA_PIPELINE_BENCH_TRACE=0 \
BURN_JEPA_PIPELINE_BENCH_DENSE_PATCHIFY=0 \
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,autogaze-ndarray,autogaze-cuda,sparse-patchify-cuda,cuda
```

The workflow uploads `target/cuda-benchmark/autogaze_sparse_jepa_cuda.csv` as
the `cuda-e2e-benchmark` artifact and fails if the CSV has only the header row.
Trace decoding remains disabled by default; set the workflow `trace` input to
`true` only when measuring the decoded fixation-trace path.

For a local CUDA machine, the equivalent command is:

```sh
BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS=cuda \
BURN_JEPA_PIPELINE_JEPA_BACKENDS=sparse-patchify-cuda \
BURN_JEPA_PIPELINE_BENCH_REPS=3 \
BURN_JEPA_PIPELINE_BENCH_WARMUPS=1 \
BURN_JEPA_PIPELINE_BENCH_1080P=false \
BURN_JEPA_PIPELINE_BENCH_TRACE=0 \
BURN_JEPA_PIPELINE_BENCH_DENSE_PATCHIFY=0 \
BURN_JEPA_PIPELINE_BENCH_OUT=target/cuda-benchmark/autogaze_sparse_jepa_cuda.csv \
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,autogaze-ndarray,autogaze-cuda,sparse-patchify-cuda,cuda
```

Before accepting the result, verify that:

- `nvidia-smi -L` lists at least one device.
- `/dev/nvidiactl` or `/dev/nvidia0` exists on Linux.
- The benchmark process appears in `nvidia-smi` while a live run is active.
- A forced smoke does not fail with
  `DriverError(CUDA_ERROR_NO_DEVICE, "no CUDA-capable device is detected")`.
- The CSV has data rows, not just the header.
- `jepa_backend` is `sparse-patchify-cuda` for end-to-end CUDA sparse patchify.
- `autogaze_trace_ms` is `0.000` when trace is disabled.
- `temporal_frames_per_sec` is reported for every requested resolution and
  density row.

`nvidia-smi` alone is not sufficient evidence in sandboxed environments: NVML
can see a GPU while the CUDA runtime still cannot open a device. In that case
the benchmark should keep reporting no defensible CUDA FPS rows. The benchmark
preflight prints both sides of that state when possible:
`/proc/driver/nvidia is visible`, optional `nvidia-smi -L sees ...` or probe
failure details, and
`CUDA runtime cannot open a device without NVIDIA character devices`.

The latest local run on 2026-05-13 used an RTX PRO 6000 Blackwell Workstation
Edition and produced data rows in
`target/codex-cuda-live/autogaze_sparse_jepa_cuda_trace_off.csv`; see
`docs/e2e-benchmark-results.md` for the checked-in CUDA matrix.
