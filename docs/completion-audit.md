# Completion Audit

Date: 2026-05-09

Objective: support an efficient and numerically checked
`burn_autogaze` -> sparse image token projection -> sparse V-JEPA 2.1 temporal
pipeline with stable next-frame updates, optional dense keyframes, benchmark
evidence, and clear backend/runtime status.

## Checklist

| Requirement | Artifact | Evidence | Status |
|---|---|---|---|
| Sparse image tokens from AutoGaze map into V-JEPA tubelet tokens. | `src/sparse_patchify.rs`, `sparse_mask_from_frame_token_indices` | Unit tests cover token projection, fill-to-count behavior, and partial stream windows. | Covered |
| Masked pixels are not patchified on the WGPU sparse path. | `src/model.rs`, `src/temporal.rs`, `forward_frame_tokens_sparse_patchify_wgpu` | `tests/sparse_patchify_wgpu.rs` compares sparse patchify encoder output against dense masked encoder output. | Covered |
| Temporal sparse stream supports next-frame updates instead of waiting for full 16-frame flushes. | `TemporalSparseJepaStream`, `TemporalSparseJepaState`, `TemporalSparseMaskState` | `tests/temporal.rs` covers projection, precomputed-mask entrypoints, keyframe counters, cached predictor-plan reuse, reset, sparse stream calls, and tubelet-sized rolling windows that are shorter than `config.num_frames`. | Covered |
| Dense keyframe refresh is available but not mandatory on sparse update steps. | `TemporalSparseJepaStreamConfig::with_dense_keyframe_refresh` | `temporal_stream_can_refresh_dense_keyframes` asserts dense output is returned on keyframes and absent on sparse update steps. | Covered |
| KV cache expectations are represented correctly. | README model notes | README documents that this is predictor/feature caching, not causal KV caching, because V-JEPA attention is bidirectional. | Covered |
| Stable sparse-context updates can skip host mask projection. | `TemporalSparseJepaStream::forward_masks`, `forward_masks_sparse_patchify_wgpu` | `tests/temporal.rs` verifies direct precomputed masks preserve keyframe cadence and predictor-plan reuse; `tests/sparse_patchify_wgpu.rs` verifies the WGPU sparse-patchify direct-mask path matches the dense masked stream and reuses sparse patchify/predictor plans. A focused Criterion run measured `cached_plan_from_frame_tokens_32_sequence_tokens` at 8.5135-8.5740 ms and `cached_plan_from_precomputed_masks_32_sequence_tokens` at 8.4181-8.4998 ms. | Covered |
| Sparse hot paths avoid backend-to-host readbacks. | `tests/numerical_parity.rs` | `sparse_forward_hot_path_has_no_backend_readbacks` scans production `src/model.rs` and `src/temporal.rs` for `to_data`/`into_data` markers. | Covered |
| Trace decoding is opt-in and disabled hot path has no trace work. | `BenchTraceConfig` in `benches/autogaze_sparse_jepa_pipeline.rs`, `tests/benchmark_report.rs` | Trace-disabled benchmark rows report `autogaze_trace_ms=0.000`; the config branch skips the trace helper, so disabled runs do not clone the video tensor or call `trace_video_with_mode`. | Covered |
| Tiny Burn sparse forward is numerically checked against an independent implementation. | `tests/fixtures/vjepa_tiny_parity.py`, `tests/numerical_parity.rs` | `tiny_sparse_forward_matches_independent_torch_fixture` passes within `5e-4`. | Covered |
| Transformers-style Hugging Face V-JEPA2 loader and forward path are checked. | `tests/fixtures/vjepa_hf_tiny_parity.py`, `tests/numerical_parity.rs` | HF tiny parity passed with prediction diff `1.49e-8` and target diff `7.45e-8`. | Covered |
| Real V-JEPA 2.1 checkpoint loading is strict and real-weight forward parity is available. | `VJepaLoadOptions`, `tests/fixtures/vjepa_hf_real_micro_forward.py` | Local HF checkpoint strict load passed with `applied=456 missing=0 skipped=0 errors=0`; micro parity passed with prediction diff `2.00e-5`, target diff `4.17e-5`. | Covered |
| E2E throughput is measured across image resolutions and sparsity densities. | `benches/autogaze_sparse_jepa_pipeline.rs`, `docs/e2e-benchmark-results.md` | Checked-in table covers ndarray and WebGPU for 224x224, 384x384, 720p and densities 1%, 5%, 10%, 25%, with both 4-frame clip and 2-frame rolling/tubelet window latency. | Covered |
| Checked-in benchmark evidence has regression coverage. | `tests/benchmark_report.rs` | The integration test parses the E2E report, asserts the ndarray/WebGPU resolution-density matrix, verifies trace-off rows, checks rolling-window latency is below 4-frame clip latency, and checks the CUDA runbook/template reject header-only CSV output. | Covered |
| WebGPU and ndarray performance behavior is explained. | `docs/e2e-benchmark-results.md` | Report notes sparse V-JEPA stream is similar, but WebGPU E2E is slower because AutoGaze WebGPU generation dominates. | Covered |
| CUDA feature support compiles. | Cargo features and benchmark target | `cargo check --no-default-features --features cuda` and CUDA E2E bench target check passed. | Covered |
| CUDA E2E FPS is measured. | `docs/e2e-benchmark-results.md`, `docs/cuda-benchmark.md`, `docs/workflows/cuda-benchmark.yml` | This host has no usable CUDA runtime for Burn/CubeCL: `nvidia-smi -L` can see an RTX PRO 6000, but no `/dev/nvidia*` nodes are visible and a forced CUDA benchmark fails with `CUDA_ERROR_NO_DEVICE`, emitting only a header CSV. The benchmark preflight now reports when `nvidia-smi` and `/proc/driver/nvidia` are visible but CUDA character devices are missing, and expands the CUDA `RecvError` skip into an explicit CUDA worker/runtime diagnostic. A manual self-hosted CUDA benchmark workflow template now fails if the emitted CSV has no data rows; publishing it under `.github/workflows/` requires workflow-file write permission. | Blocked |
| Static page shell and workflow status are honest. | `crates/bevy_burn_jepa/www`, `README.md`, `web/README.md`, `crates/bevy_burn_jepa/README.md` | The static shell remains checked in, but the Pages badge is removed and docs note that the deploy workflow is disabled remotely because GitHub reports Pages is unavailable for this repository plan. | Covered |
| Package remains publishable. | Cargo package manifest | `cargo package --allow-dirty` passed with docs included. | Covered |

## Verification Commands

The following commands were run during the final audit:

```sh
cargo test --no-default-features --features ndarray
cargo test --test temporal --no-default-features --features ndarray
cargo test --test sparse_patchify_wgpu --no-default-features --features sparse-patchify-wgpu -- --nocapture
cargo test --test numerical_parity sparse_forward_hot_path_has_no_backend_readbacks \
  --no-default-features --features ndarray
BURN_JEPA_VJEPA21_CHECKPOINT_DIR=/home/mosure/.cache/huggingface/hub/models--facebook--vjepa2-vitl-fpc64-256/snapshots/b3c1679b7c34d3255ef3547f27c7b226aefab26f \
BURN_JEPA_VJEPA21_FORWARD_PARITY=1 \
cargo test --test numerical_parity real_vjepa_checkpoint_loads_when_fixture_is_set \
  --no-default-features --features ndarray -- --ignored --nocapture
cargo bench --bench sparse_pipeline temporal_sparse_stream_hot_path_ndarray \
  --no-default-features --features ndarray -- --sample-size 10
cargo check --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,sparse-patchify-wgpu,cuda
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,sparse-patchify-wgpu,cuda
BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS=cuda \
BURN_JEPA_PIPELINE_BENCH_REPS=1 \
BURN_JEPA_PIPELINE_BENCH_WARMUPS=0 \
BURN_JEPA_PIPELINE_BENCH_1080P=false \
BURN_JEPA_PIPELINE_BENCH_TRACE=0 \
BURN_JEPA_PIPELINE_CUDA_FORCE=1 \
BURN_JEPA_PIPELINE_BENCH_OUT=target/codex-cuda-live/autogaze_sparse_jepa_cuda_trace_off.csv \
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,sparse-patchify-wgpu,cuda
cargo test --test benchmark_report --no-default-features --features ndarray
cargo package --allow-dirty
```

## Current Status

The implementation, tests, parity fixtures, packaging, and ndarray/WebGPU E2E
benchmarks are aligned with the objective. The only incomplete item is real CUDA
runtime throughput. The code now preflights CUDA and reports a clean skip when
the runtime is unavailable; a forced local run reached CubeCL CUDA but failed
with `CUDA_ERROR_NO_DEVICE` and produced a header-only CSV. Actual CUDA FPS
needs a machine where both the driver and CUDA device nodes/runtime are visible
to the process. Use `docs/cuda-benchmark.md` locally, or install the
`docs/workflows/cuda-benchmark.yml` template as a manual `cuda benchmark`
workflow, to produce the missing CUDA FPS artifact on that host.
