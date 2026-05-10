# E2E Pipeline Benchmark Results

Date: 2026-05-09

Benchmark data commit: `9a8ee119b101d4de26754ad9ac20b235c8a98f73`

This report measures the current AutoGaze -> sparse V-JEPA temporal stream path.
The benchmark feeds deterministic video through AutoGaze token generation,
projects generated sparse image tokens into V-JEPA tubelet tokens, runs sparse
patchification with `burn_flex_gmm`, encodes the sparse context, and runs the
cached temporal predictor. Trace decoding is disabled for all rows.

Commands:

```sh
BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS=ndarray \
BURN_JEPA_PIPELINE_BENCH_REPS=1 \
BURN_JEPA_PIPELINE_BENCH_WARMUPS=0 \
BURN_JEPA_PIPELINE_BENCH_1080P=false \
BURN_JEPA_PIPELINE_BENCH_TRACE=0 \
BURN_JEPA_PIPELINE_BENCH_OUT=target/codex-hf-parity/autogaze_sparse_jepa_ndarray_trace_off.csv \
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,sparse-patchify-wgpu

BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS=webgpu \
BURN_JEPA_PIPELINE_BENCH_REPS=1 \
BURN_JEPA_PIPELINE_BENCH_WARMUPS=0 \
BURN_JEPA_PIPELINE_BENCH_1080P=false \
BURN_JEPA_PIPELINE_BENCH_TRACE=0 \
BURN_JEPA_PIPELINE_BENCH_OUT=target/codex-hf-parity/autogaze_sparse_jepa_webgpu_trace_off.csv \
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

`temporal_stream_ms` is the hot temporal stream call with cached plans. It
includes sparse patchify, sparse encoder, and cached predictor, but not AutoGaze.
`temporal_e2e_ms` and `fps` include AutoGaze token generation plus the temporal
stream. FPS is frames per second for 4-frame clips.

| Backend | Resolution | Density | Context tokens | Temporal stream ms | Temporal E2E ms | E2E fps | Trace ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| ndarray | 224x224 | 0.0100 | 4 | 6.251 | 28.975 | 138.05 | 0.000 |
| ndarray | 224x224 | 0.0500 | 20 | 6.766 | 30.218 | 132.37 | 0.000 |
| ndarray | 224x224 | 0.1000 | 40 | 9.734 | 31.737 | 126.04 | 0.000 |
| ndarray | 224x224 | 0.2500 | 98 | 11.060 | 33.376 | 119.85 | 0.000 |
| ndarray | 384x384 | 0.0100 | 12 | 6.320 | 54.835 | 72.95 | 0.000 |
| ndarray | 384x384 | 0.0500 | 58 | 7.586 | 54.869 | 72.90 | 0.000 |
| ndarray | 384x384 | 0.1000 | 116 | 9.436 | 57.967 | 69.01 | 0.000 |
| ndarray | 384x384 | 0.2500 | 288 | 15.275 | 62.890 | 63.60 | 0.000 |
| ndarray | 720p | 0.0100 | 72 | 9.404 | 73.605 | 54.34 | 0.000 |
| ndarray | 720p | 0.0500 | 360 | 18.021 | 83.209 | 48.07 | 0.000 |
| ndarray | 720p | 0.1000 | 720 | 37.624 | 100.434 | 39.83 | 0.000 |
| ndarray | 720p | 0.2500 | 1800 | 113.191 | 169.449 | 23.61 | 0.000 |
| webgpu | 224x224 | 0.0100 | 4 | 7.089 | 319.453 | 12.52 | 0.000 |
| webgpu | 224x224 | 0.0500 | 20 | 7.698 | 313.181 | 12.77 | 0.000 |
| webgpu | 224x224 | 0.1000 | 40 | 8.905 | 332.627 | 12.03 | 0.000 |
| webgpu | 224x224 | 0.2500 | 98 | 11.875 | 358.502 | 11.16 | 0.000 |
| webgpu | 384x384 | 0.0100 | 12 | 7.134 | 303.928 | 13.16 | 0.000 |
| webgpu | 384x384 | 0.0500 | 58 | 8.313 | 327.402 | 12.22 | 0.000 |
| webgpu | 384x384 | 0.1000 | 116 | 10.235 | 324.010 | 12.35 | 0.000 |
| webgpu | 384x384 | 0.2500 | 288 | 16.447 | 346.431 | 11.55 | 0.000 |
| webgpu | 720p | 0.0100 | 72 | 10.447 | 386.749 | 10.34 | 0.000 |
| webgpu | 720p | 0.0500 | 360 | 19.668 | 322.102 | 12.42 | 0.000 |
| webgpu | 720p | 0.1000 | 720 | 39.491 | 383.620 | 10.43 | 0.000 |
| webgpu | 720p | 0.2500 | 1800 | 112.432 | 420.002 | 9.52 | 0.000 |

## Interpretation

The sparse V-JEPA temporal stream is similar between ndarray and WebGPU for the
small sparse-token counts tested here. WebGPU E2E throughput is lower because
the AutoGaze generation stage is much slower on this stack: roughly 290-352 ms
per 4-frame clip for WebGPU versus 23-71 ms for ndarray in the same benchmark
matrix. For this pipeline and hardware state, backend launch/synchronization and
AutoGaze WebGPU runtime overhead dominate the small sparse V-JEPA workload.

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
