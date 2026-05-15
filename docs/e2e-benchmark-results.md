# E2E Pipeline Benchmark Results

Date: 2026-05-13

Benchmark data source: local warmup/3-repetition median run from this checked-in
benchmark revision.

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
BURN_JEPA_PIPELINE_BENCH_REPS=3 \
BURN_JEPA_PIPELINE_BENCH_WARMUPS=1 \
BURN_JEPA_PIPELINE_BENCH_1080P=false \
BURN_JEPA_PIPELINE_BENCH_TRACE=0 \
BURN_JEPA_PIPELINE_BENCH_OUT=target/codex-opt-bench/autogaze_sparse_jepa_ndarray_masks.csv \
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,autogaze-ndarray,sparse-patchify-wgpu

BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS=webgpu \
BURN_JEPA_PIPELINE_BENCH_REPS=3 \
BURN_JEPA_PIPELINE_BENCH_WARMUPS=1 \
BURN_JEPA_PIPELINE_BENCH_1080P=false \
BURN_JEPA_PIPELINE_BENCH_TRACE=0 \
BURN_JEPA_PIPELINE_BENCH_OUT=target/codex-opt-bench/autogaze_sparse_jepa_webgpu_masks.csv \
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,webgpu,autogaze-ndarray,autogaze-webgpu,sparse-patchify-wgpu

BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS=webgpu \
BURN_JEPA_PIPELINE_BENCH_RESOLUTIONS=720p \
BURN_JEPA_PIPELINE_BENCH_DENSITIES=0.25 \
BURN_JEPA_PIPELINE_BENCH_REPS=3 \
BURN_JEPA_PIPELINE_BENCH_WARMUPS=1 \
BURN_JEPA_PIPELINE_BENCH_1080P=false \
BURN_JEPA_PIPELINE_BENCH_TRACE=0 \
BURN_JEPA_PIPELINE_BENCH_OUT=target/codex-opt-bench/autogaze_sparse_jepa_webgpu_720p25_plan.csv \
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,webgpu,autogaze-ndarray,autogaze-webgpu,sparse-patchify-wgpu

BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS=cuda \
BURN_JEPA_PIPELINE_JEPA_BACKENDS=sparse-patchify-cuda \
BURN_JEPA_PIPELINE_BENCH_REPS=3 \
BURN_JEPA_PIPELINE_BENCH_WARMUPS=1 \
BURN_JEPA_PIPELINE_BENCH_1080P=false \
BURN_JEPA_PIPELINE_BENCH_TRACE=0 \
BURN_JEPA_PIPELINE_BENCH_DENSE_PATCHIFY=0 \
BURN_JEPA_PIPELINE_BENCH_OUT=target/codex-cuda-live/autogaze_sparse_jepa_cuda_trace_off.csv \
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,autogaze-ndarray,autogaze-cuda,sparse-patchify-cuda,cuda
```

## Results

`clip_stream_ms` is the hot cached stream call for a 4-frame clip using
precomputed sparse masks. `rolling_stream_ms` is the same cached stream call
with a 2-frame tubelet window. `clip_e2e_ms` and `rolling_e2e_ms` include
AutoGaze token generation, direct generated-token-to-mask projection, and the
temporal stream. FPS is frames per second for each path's input window.

| Backend | Resolution | Density | Context tokens | Clip stream ms | Rolling stream ms | Clip E2E ms | Rolling E2E ms | Clip fps | Rolling fps | Trace ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ndarray | 224x224 | 0.0100 | 4 | 6.400 | 5.929 | 20.814 | 12.216 | 192.16 | 163.72 | 0.000 |
| ndarray | 224x224 | 0.0500 | 20 | 6.490 | 6.401 | 22.827 | 13.997 | 175.24 | 142.88 | 0.000 |
| ndarray | 224x224 | 0.1000 | 40 | 7.070 | 7.133 | 24.584 | 14.637 | 162.72 | 136.64 | 0.000 |
| ndarray | 224x224 | 0.2500 | 98 | 9.532 | 8.137 | 30.685 | 16.928 | 130.36 | 118.15 | 0.000 |
| ndarray | 384x384 | 0.0100 | 12 | 6.626 | 6.272 | 45.373 | 25.133 | 88.16 | 79.58 | 0.000 |
| ndarray | 384x384 | 0.0500 | 58 | 8.169 | 7.892 | 48.780 | 26.084 | 82.00 | 76.67 | 0.000 |
| ndarray | 384x384 | 0.1000 | 116 | 9.559 | 7.993 | 51.621 | 27.203 | 77.48 | 73.52 | 0.000 |
| ndarray | 384x384 | 0.2500 | 288 | 15.896 | 10.991 | 60.336 | 32.448 | 66.28 | 61.64 | 0.000 |
| ndarray | 720p | 0.0100 | 72 | 8.906 | 6.975 | 63.024 | 27.547 | 63.48 | 72.60 | 0.000 |
| ndarray | 720p | 0.0500 | 360 | 18.374 | 11.897 | 73.455 | 32.661 | 54.44 | 61.24 | 0.000 |
| ndarray | 720p | 0.1000 | 720 | 34.044 | 18.320 | 92.113 | 40.199 | 43.44 | 49.75 | 0.000 |
| ndarray | 720p | 0.2500 | 1800 | 114.995 | 46.540 | 180.906 | 68.952 | 22.12 | 29.01 | 0.000 |
| webgpu | 224x224 | 0.0100 | 4 | 18.523 | 10.579 | 54.513 | 33.824 | 73.36 | 59.13 | 0.000 |
| webgpu | 224x224 | 0.0500 | 20 | 8.261 | 8.253 | 438.984 | 271.061 | 9.12 | 7.38 | 0.000 |
| webgpu | 224x224 | 0.1000 | 40 | 35.729 | 46.186 | 163.579 | 111.022 | 24.44 | 18.01 | 0.000 |
| webgpu | 224x224 | 0.2500 | 98 | 14.258 | 10.879 | 246.931 | 125.557 | 16.20 | 15.93 | 0.000 |
| webgpu | 384x384 | 0.0100 | 12 | 7.703 | 7.463 | 33.917 | 20.063 | 117.92 | 99.69 | 0.000 |
| webgpu | 384x384 | 0.0500 | 58 | 9.398 | 7.824 | 63.846 | 35.948 | 62.64 | 55.64 | 0.000 |
| webgpu | 384x384 | 0.1000 | 116 | 14.730 | 9.235 | 91.736 | 47.398 | 43.60 | 42.20 | 0.000 |
| webgpu | 384x384 | 0.2500 | 288 | 15.854 | 11.367 | 213.540 | 103.442 | 18.72 | 19.33 | 0.000 |
| webgpu | 720p | 0.0100 | 72 | 9.424 | 7.616 | 38.094 | 23.713 | 105.00 | 84.34 | 0.000 |
| webgpu | 720p | 0.0500 | 360 | 18.269 | 12.073 | 78.684 | 42.127 | 50.84 | 47.48 | 0.000 |
| webgpu | 720p | 0.1000 | 720 | 38.234 | 18.568 | 122.481 | 60.452 | 32.64 | 33.08 | 0.000 |
| webgpu | 720p | 0.2500 | 1800 | 118.912 | 42.204 | 341.841 | 145.744 | 11.72 | 13.72 | 0.000 |
| cuda | 224x224 | 0.0100 | 4 | 1.392 | 1.245 | 5.085 | 3.918 | 786.67 | 510.41 | 0.000 |
| cuda | 224x224 | 0.0500 | 20 | 1.201 | 1.236 | 11.012 | 6.769 | 363.23 | 295.46 | 0.000 |
| cuda | 224x224 | 0.1000 | 40 | 1.216 | 1.229 | 18.750 | 9.554 | 213.33 | 209.34 | 0.000 |
| cuda | 224x224 | 0.2500 | 98 | 1.367 | 1.694 | 41.778 | 21.141 | 95.74 | 94.60 | 0.000 |
| cuda | 384x384 | 0.0100 | 12 | 1.239 | 1.261 | 5.397 | 4.074 | 741.14 | 490.90 | 0.000 |
| cuda | 384x384 | 0.0500 | 58 | 1.240 | 2.141 | 12.841 | 8.311 | 311.50 | 240.63 | 0.000 |
| cuda | 384x384 | 0.1000 | 116 | 1.510 | 1.745 | 17.313 | 9.131 | 231.04 | 219.03 | 0.000 |
| cuda | 384x384 | 0.2500 | 288 | 1.517 | 1.288 | 42.739 | 21.895 | 93.59 | 91.35 | 0.000 |
| cuda | 720p | 0.0100 | 72 | 1.267 | 1.248 | 5.002 | 3.937 | 799.68 | 507.99 | 0.000 |
| cuda | 720p | 0.0500 | 360 | 1.535 | 1.309 | 10.990 | 8.366 | 363.97 | 239.07 | 0.000 |
| cuda | 720p | 0.1000 | 720 | 1.370 | 1.551 | 18.247 | 9.387 | 219.21 | 213.05 | 0.000 |
| cuda | 720p | 0.2500 | 1800 | 2.519 | 1.527 | 45.504 | 22.112 | 87.90 | 90.45 | 0.000 |

## 720p Stage Metrics

All values are median milliseconds except the two FPS columns. `mask project`
uses the direct generated-token-to-mask path and avoids materializing
`Vec<Vec<usize>>` frame tokens. `rolling AG cached` and `rolling cached E2E`
use `AutoGazeStreamingCache`; in this native synthetic run they are reported for
diagnosis but are not faster than direct rolling generation.

| Backend | Density | Top-k | AG gen | Rolling AG | Rolling AG cached | Project | Mask project | Plan | Sparse patchify | Encoder | Predictor | Clip stream | Clip mask stream | Rolling stream | Rolling mask stream | Rolling mask E2E | Rolling cached E2E | Rolling mask fps | Rolling cached fps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ndarray | 0.0100 | 1 | 54.228 | 19.321 | 21.255 | 0.002 | 0.002 | 0.127 | 1.675 | 3.572 | 4.268 | 8.807 | 8.906 | 7.040 | 6.975 | 27.547 | 29.626 | 72.60 | 67.51 |
| ndarray | 0.0500 | 5 | 53.864 | 20.150 | 22.063 | 0.005 | 0.004 | 0.218 | 6.908 | 6.827 | 6.383 | 18.262 | 18.374 | 12.393 | 11.897 | 32.661 | 35.417 | 61.24 | 56.47 |
| ndarray | 0.1000 | 10 | 54.682 | 20.431 | 23.022 | 0.009 | 0.008 | 0.169 | 13.493 | 10.629 | 12.322 | 35.567 | 34.044 | 18.759 | 18.320 | 40.199 | 42.757 | 49.75 | 46.78 |
| ndarray | 0.2500 | 25 | 58.875 | 21.986 | 25.802 | 0.021 | 0.020 | 0.241 | 32.217 | 42.507 | 40.098 | 116.738 | 114.995 | 45.174 | 46.540 | 68.952 | 72.981 | 29.01 | 27.40 |
| webgpu | 0.0100 | 1 | 28.767 | 14.135 | 17.202 | 0.002 | 0.002 | 0.035 | 1.691 | 3.830 | 4.015 | 9.250 | 9.424 | 7.910 | 7.616 | 23.713 | 26.372 | 84.34 | 75.84 |
| webgpu | 0.0500 | 5 | 58.481 | 30.497 | 32.707 | 0.008 | 0.007 | 0.074 | 7.023 | 6.131 | 6.690 | 19.291 | 18.269 | 12.311 | 12.073 | 42.127 | 45.598 | 47.48 | 43.86 |
| webgpu | 0.1000 | 10 | 88.032 | 41.474 | 46.945 | 0.016 | 0.014 | 0.155 | 13.760 | 11.275 | 12.334 | 36.266 | 38.234 | 19.148 | 18.568 | 60.452 | 65.980 | 33.08 | 30.31 |
| webgpu | 0.2500 | 25 | 198.487 | 92.697 | 107.994 | 0.024 | 0.023 | 0.166 | 33.391 | 37.329 | 40.456 | 106.193 | 118.912 | 47.451 | 42.204 | 145.744 | 167.004 | 13.72 | 11.98 |
| cuda | 0.0100 | 1 | 3.197 | 1.699 | 1.706 | 0.001 | 0.001 | 0.049 | 0.420 | 0.848 | 0.727 | 1.255 | 1.267 | 1.825 | 1.248 | 3.937 | 3.971 | 507.99 | 503.68 |
| cuda | 0.0500 | 5 | 9.822 | 4.616 | 5.991 | 0.004 | 0.004 | 0.062 | 0.268 | 0.764 | 1.268 | 1.401 | 1.535 | 1.302 | 1.309 | 8.366 | 7.597 | 239.07 | 263.26 |
| cuda | 0.1000 | 10 | 17.506 | 7.437 | 7.109 | 0.008 | 0.008 | 0.152 | 0.342 | 1.314 | 0.808 | 1.466 | 1.370 | 1.299 | 1.551 | 9.387 | 9.452 | 213.05 | 211.61 |
| cuda | 0.2500 | 25 | 39.844 | 19.315 | 19.515 | 0.021 | 0.019 | 0.363 | 0.726 | 1.487 | 1.237 | 2.485 | 2.519 | 1.625 | 1.527 | 22.112 | 24.533 | 90.45 | 81.52 |

## Interpretation

Terminology note: `webgpu` and `wgpu` are separate Burn feature/backend
surfaces, but they are the same WebGPU/wgpu backend family for paper-level
analysis. In this document, `WebGPU` usually means Burn's default
`burn::backend::WebGpu` lane, while `sparse-patchify-wgpu` means the native WGPU
`burn_flex_gmm` sparse-patchify kernel lane. Do not treat those rows as
independent hardware configurations.

The sparse V-JEPA temporal stream is similar between ndarray and WebGPU for the
small sparse-token counts tested here. The benchmark uses
`AutogazeSparseJepaWindowConfig`, which treats AutoGaze top-k as a per-frame
budget and accounts for the fact that one AutoGaze image token fans out to
multiple V-JEPA patches at higher resolutions. The default top-k overfetch is
now 1.0, so the 720p rows use 1, 5, 10, and 25 AutoGaze tokens per frame for
1%, 5%, 10%, and 25% context density instead of over-generating to the
32-token cap.

The dominant E2E bottleneck is still AutoGaze generation. At 720p/25%, WebGPU
spends 198.487 ms in 4-frame AutoGaze generation and 92.697 ms in direct
rolling AutoGaze generation; direct mask projection and plan construction
together stay below 0.189 ms. Sparse patchification scales with density, but is
still 33.391 ms at 720p/25%, below AutoGaze plus encoder/predictor cost.
Exact-budget top-k remains the main generation-side control; the remaining
WebGPU slowdown is token-loop launch/readback overhead inside AutoGaze.
After consolidating generation/projection on `AutogazeSparseJepaWindowPlan`, a
targeted 720p/25% WebGPU rerun measured 195.005 ms 4-frame AutoGaze generation,
91.093 ms rolling AutoGaze generation, 0.019 ms direct mask projection, 0.301 ms
plan construction, 136.900 ms rolling mask E2E, and 14.61 rolling-mask FPS.

This pass removed duplicate projection from the optimized stream path. The
frame-token stream remains measured, but E2E rows now use direct
generated-token-to-mask projection and `forward_masks_sparse_patchify_wgpu`.
For WebGPU at 720p/25%, the rolling stream call drops from 47.451 ms with
frame-token projection inside the stream to 42.204 ms with precomputed masks.
For ndarray at 720p/10%, the 4-frame mask stream is 34.044 ms versus
35.567 ms for the frame-token stream.

CUDA is now the fastest measured E2E lane on this host. The full matrix above
was captured before the dedicated CUDA `burn_flex_gmm` sparse patchify backend
landed, so those CUDA rows are the older CUDA AutoGaze plus WGPU
sparse-patchify/V-JEPA lane. At 720p/25%, that lane reports 45.504 ms for the
4-frame table E2E path, 43.749 ms for the direct-mask 4-frame E2E path, and
22.112 ms for the rolling mask path, versus WebGPU's 341.841 ms and
145.744 ms. At 720p/1%, it reports 5.002 ms and 3.937 ms.

The benchmark now has an independent JEPA backend selector:
`BURN_JEPA_PIPELINE_JEPA_BACKENDS=sparse-patchify-cuda` routes sparse
patchification and V-JEPA through the CUDA `burn_flex_gmm` kernel path, while
`sparse-patchify-wgpu` preserves the older mixed lane for comparison.

## Pure CUDA Sparse Patchify Smoke

After adding the CUDA `burn_flex_gmm` sparse patchify backend, a warmed single
row was run with dense patchify comparison disabled so the measured path stays
on the sparse CUDA hot path. This uses published `burn_flex_gmm` 0.21.1, which
adds the `cuda-kernel` feature:

```sh
BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS=cuda \
BURN_JEPA_PIPELINE_JEPA_BACKENDS=sparse-patchify-cuda \
BURN_JEPA_PIPELINE_BENCH_RESOLUTIONS=224x224 \
BURN_JEPA_PIPELINE_BENCH_DENSITIES=0.05 \
BURN_JEPA_PIPELINE_BENCH_WARMUPS=1 \
BURN_JEPA_PIPELINE_BENCH_REPS=1 \
BURN_JEPA_PIPELINE_BENCH_DENSE_PATCHIFY=0 \
BURN_JEPA_PIPELINE_BENCH_OUT=target/autogaze_sparse_jepa_pipeline_cuda_smoke_warmed.csv \
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,autogaze-cuda,cuda,sparse-patchify-cuda
```

| AutoGaze backend | JEPA backend | Resolution | Density | Context tokens | Sparse patchify ms | Encoder ms | Predictor ms | Sparse JEPA ms | Clip E2E ms | Rolling mask E2E ms | Clip fps | Rolling mask fps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| autogaze-cuda | sparse-patchify-cuda | 224x224 | 0.0500 | 20 | 0.204 | 0.933 | 0.896 | 1.272 | 12.381 | 7.539 | 323.07 | 265.30 |

The first no-warmup run of the same row showed inflated standalone CUDA segment
timings from first-use kernel setup, while the warmed run above kept sparse
patchify under a quarter millisecond. A 720p/25% stress row was started but
stopped after several minutes of first-use execution without producing a sample;
use the opt-in kernel parity tests below for isolated patchify correctness and
repeat high-resolution benchmarking after CUDA kernels are warm.

WebGPU still wins the lowest-density 720p 4-frame path against ndarray:
38.094 ms versus ndarray's 63.024 ms. The rolling 720p/1% mask path is
23.713 ms on WebGPU and 27.547 ms on ndarray. At higher densities, ndarray can
still report higher E2E FPS than WebGPU because the current AutoGaze WebGPU path
is a small autoregressive decoder loop. Each generated token block requires
short GPU launches and a logits/task readback to choose the next token block, so
launch/synchronization/readback overhead dominates before the sparse V-JEPA
kernels become large enough to amortize it. This is an AutoGaze generation
bottleneck, not evidence that dense CPU patchification is beating the WGPU
sparse patchify path.

## CUDA Status

The CUDA feature, runtime smoke, TTT training path, and E2E benchmark path now
run on this host with visible NVIDIA device nodes:

```sh
nvidia-smi -L
cargo check --no-default-features --features cuda
cargo check --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,autogaze-ndarray,autogaze-cuda,sparse-patchify-cuda,cuda
BURN_JEPA_RUN_CUDA_SPARSE_PATCHIFY=1 \
cargo test --test sparse_patchify_cuda \
  --no-default-features --features ndarray,cuda,sparse-patchify-cuda -- --nocapture --test-threads=1
BURN_JEPA_RUN_GPU_TRAINING_SMOKE=1 \
cargo test --no-default-features --features ndarray,cuda \
  --test gpu_training_smoke cuda_ttt_training_smoke_runs_when_requested -- --nocapture
```

The local device probe reports:

```text
GPU 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (UUID: GPU-343b002b-d5c6-2d9c-9fea-9ab5a52d0879)
```

The CUDA training smoke completed successfully. The expanded synthetic TTT
experiment matrix also completed 96/96 CUDA trials across all model variants,
mask policies, and four density settings, with sparse predictor mask loss
enabled. A separate first-use single-row run confirmed that the CUDA benchmark
process was resident on the GPU, but its standalone encoder/predictor timings
included first-use kernel setup. The checked-in table therefore uses the warmed
3-repetition run.

The benchmark still keeps the runtime preflight and header-only CSV rejection:
`nvidia-smi` visibility alone is not treated as sufficient evidence in
sandboxed environments, and `BURN_JEPA_PIPELINE_CUDA_FORCE=1` should only be
used when the CUDA runtime is known to be accessible despite the default Linux
device-node or `nvidia-smi -L` checks.
