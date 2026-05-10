# E2E Pipeline Benchmark Results

Date: 2026-05-09

Benchmark data source: local run from this checked-in benchmark revision.

This report measures the current AutoGaze -> sparse V-JEPA temporal stream path.
The benchmark feeds deterministic video through AutoGaze token generation,
projects generated sparse image tokens into V-JEPA tubelet tokens, runs sparse
patchification with `burn_flex_gmm`, encodes the sparse context, and runs the
cached temporal predictor. It reports both the 4-frame clip path and a
2-frame rolling/tubelet window path for next-frame-style updates. Trace decoding
is disabled for all rows.

Commands:

```sh
BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS=ndarray \
BURN_JEPA_PIPELINE_BENCH_REPS=1 \
BURN_JEPA_PIPELINE_BENCH_WARMUPS=0 \
BURN_JEPA_PIPELINE_BENCH_1080P=false \
BURN_JEPA_PIPELINE_BENCH_TRACE=0 \
BURN_JEPA_PIPELINE_BENCH_OUT=target/codex-rolling-bench/autogaze_sparse_jepa_ndarray_rolling_trace_off.csv \
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,sparse-patchify-wgpu

BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS=webgpu \
BURN_JEPA_PIPELINE_BENCH_REPS=1 \
BURN_JEPA_PIPELINE_BENCH_WARMUPS=0 \
BURN_JEPA_PIPELINE_BENCH_1080P=false \
BURN_JEPA_PIPELINE_BENCH_TRACE=0 \
BURN_JEPA_PIPELINE_BENCH_OUT=target/codex-rolling-bench/autogaze_sparse_jepa_webgpu_rolling_trace_off.csv \
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,webgpu,sparse-patchify-wgpu

BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS=cuda \
BURN_JEPA_PIPELINE_BENCH_REPS=1 \
BURN_JEPA_PIPELINE_BENCH_WARMUPS=0 \
BURN_JEPA_PIPELINE_BENCH_1080P=false \
BURN_JEPA_PIPELINE_BENCH_TRACE=0 \
BURN_JEPA_PIPELINE_BENCH_OUT=target/codex-cuda-bench-check/autogaze_sparse_jepa_cuda_trace_off.csv \
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,sparse-patchify-wgpu,cuda
```

## Results

`clip_stream_ms` is the hot cached stream call for a 4-frame clip.
`rolling_stream_ms` is the same cached stream call with a 2-frame tubelet window.
`clip_e2e_ms` and `rolling_e2e_ms` include AutoGaze token generation plus the
temporal stream. FPS is frames per second for each path's input window.

| Backend | Resolution | Density | Context tokens | Clip stream ms | Rolling stream ms | Clip E2E ms | Rolling E2E ms | Clip fps | Rolling fps | Trace ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ndarray | 224x224 | 0.0100 | 4 | 6.596 | 7.001 | 30.158 | 17.714 | 132.64 | 112.91 | 0.000 |
| ndarray | 224x224 | 0.0500 | 20 | 7.586 | 6.874 | 31.395 | 17.796 | 127.41 | 112.39 | 0.000 |
| ndarray | 224x224 | 0.1000 | 40 | 8.603 | 7.714 | 32.953 | 17.655 | 121.38 | 113.28 | 0.000 |
| ndarray | 224x224 | 0.2500 | 98 | 10.800 | 9.010 | 36.685 | 19.044 | 109.04 | 105.02 | 0.000 |
| ndarray | 384x384 | 0.0100 | 12 | 7.454 | 7.431 | 56.716 | 29.579 | 70.53 | 67.62 | 0.000 |
| ndarray | 384x384 | 0.0500 | 58 | 8.654 | 8.001 | 56.172 | 31.237 | 71.21 | 64.03 | 0.000 |
| ndarray | 384x384 | 0.1000 | 116 | 11.176 | 9.195 | 58.164 | 31.264 | 68.77 | 63.97 | 0.000 |
| ndarray | 384x384 | 0.2500 | 288 | 16.929 | 12.937 | 68.358 | 35.621 | 58.52 | 56.15 | 0.000 |
| ndarray | 720p | 0.0100 | 72 | 9.503 | 8.363 | 75.168 | 33.437 | 53.21 | 59.81 | 0.000 |
| ndarray | 720p | 0.0500 | 360 | 25.005 | 14.392 | 87.753 | 37.508 | 45.58 | 53.32 | 0.000 |
| ndarray | 720p | 0.1000 | 720 | 36.535 | 19.418 | 102.974 | 43.887 | 38.84 | 45.57 | 0.000 |
| ndarray | 720p | 0.2500 | 1800 | 232.481 | 74.375 | 261.551 | 98.564 | 15.29 | 20.29 | 0.000 |
| webgpu | 224x224 | 0.0100 | 4 | 9.273 | 12.914 | 514.608 | 217.762 | 7.77 | 9.18 | 0.000 |
| webgpu | 224x224 | 0.0500 | 20 | 9.081 | 8.022 | 398.848 | 192.157 | 10.03 | 10.41 | 0.000 |
| webgpu | 224x224 | 0.1000 | 40 | 8.881 | 8.430 | 373.824 | 181.743 | 10.70 | 11.00 | 0.000 |
| webgpu | 224x224 | 0.2500 | 98 | 10.537 | 9.507 | 330.391 | 158.924 | 12.11 | 12.58 | 0.000 |
| webgpu | 384x384 | 0.0100 | 12 | 7.721 | 7.242 | 335.037 | 163.667 | 11.94 | 12.22 | 0.000 |
| webgpu | 384x384 | 0.0500 | 58 | 10.106 | 8.181 | 320.036 | 160.054 | 12.50 | 12.50 | 0.000 |
| webgpu | 384x384 | 0.1000 | 116 | 10.847 | 8.594 | 327.100 | 156.316 | 12.23 | 12.79 | 0.000 |
| webgpu | 384x384 | 0.2500 | 288 | 16.482 | 11.397 | 334.479 | 165.883 | 11.96 | 12.06 | 0.000 |
| webgpu | 720p | 0.0100 | 72 | 9.150 | 8.765 | 330.306 | 166.022 | 12.11 | 12.05 | 0.000 |
| webgpu | 720p | 0.0500 | 360 | 18.549 | 13.239 | 344.517 | 169.967 | 11.61 | 11.77 | 0.000 |
| webgpu | 720p | 0.1000 | 720 | 39.589 | 18.592 | 493.887 | 181.064 | 8.10 | 11.05 | 0.000 |
| webgpu | 720p | 0.2500 | 1800 | 116.237 | 42.803 | 431.033 | 200.370 | 9.28 | 9.98 | 0.000 |

## Interpretation

The sparse V-JEPA temporal stream is similar between ndarray and WebGPU for the
small sparse-token counts tested here. Rolling-window E2E latency is lower than
the 4-frame clip latency because AutoGaze and sparse patchification process only
one V-JEPA tubelet window. At 720p/25% density, ndarray drops from 261.551 ms
per 4-frame clip to 98.564 ms per 2-frame rolling window, and WebGPU drops from
431.033 ms to 200.370 ms. Single-repetition rows are noisy, but the rolling
columns expose the intended next-frame update path separately from full-clip
latency.

WebGPU E2E throughput is still lower because the AutoGaze generation stage is
much slower on this stack: hundreds of milliseconds per 4-frame clip for WebGPU
versus tens of milliseconds for ndarray in the same benchmark matrix. For this
pipeline and hardware state, backend launch/synchronization and AutoGaze WebGPU
runtime overhead dominate the small sparse V-JEPA workload.

This explains why ndarray can report higher E2E FPS than WebGPU even though the
V-JEPA sparse patchification path is on WGPU: the E2E number includes AutoGaze,
device synchronization, and short sparse-token kernels, not only large dense
GPU-friendly matrix work.

## CUDA Status

The CUDA feature and benchmark target compile:

```sh
cargo check --no-default-features --features cuda
cargo check --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,sparse-patchify-wgpu,cuda
```

Runtime CUDA measurement is blocked on this machine. `nvidia-smi -L` reports a
driver-visible GPU:

```text
GPU 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (UUID: GPU-343b002b-d5c6-2d9c-9fea-9ab5a52d0879)
```

No `/dev/nvidia*` device nodes are visible. The CUDA benchmark selector compiled
and skips CUDA at runtime during preflight unless forced. The preflight now
distinguishes driver/procfs visibility from CUDA runtime device-node access,
and includes `nvidia-smi -L` output when the benchmark process can query it:

```text
skipping autogaze-cuda benchmark: no /dev/nvidia* device nodes; nvidia-smi -L probe failed: nvidia-smi -L failed: ; /proc/driver/nvidia is visible; CUDA runtime cannot open a device without NVIDIA character devices; set BURN_JEPA_PIPELINE_CUDA_FORCE=1 to try anyway
```

Forcing the benchmark past preflight with `BURN_JEPA_PIPELINE_CUDA_FORCE=1`
still does not produce data rows because CubeCL CUDA cannot initialize a
CUDA-capable device:

```text
DriverError(CUDA_ERROR_NO_DEVICE, "no CUDA-capable device is detected")
skipping autogaze-cuda benchmark: called `Result::unwrap()` on an `Err` value: RecvError; CUDA worker thread failed before returning results, which usually means the CUDA runtime could not initialize a device. Check for a preceding CUDA driver error and verify /dev/nvidia* device nodes are visible.
```

The emitted CUDA CSV contains only the header, so there are no defensible CUDA
FPS rows from this environment. Set `BURN_JEPA_PIPELINE_CUDA_FORCE=1` only when
CUDA is known to be available despite the default Linux device-node or
`nvidia-smi -L` preflight checks.
