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
| Temporal sparse stream supports next-frame updates instead of waiting for full 16-frame flushes. | `TemporalSparseJepaStream`, `TemporalSparseJepaState`, `TemporalSparseMaskState` | `tests/temporal.rs` covers projection, keyframe counters, cached predictor-plan reuse, reset, and sparse stream calls. | Covered |
| Dense keyframe refresh is available but not mandatory on sparse update steps. | `TemporalSparseJepaStreamConfig::with_dense_keyframe_refresh` | `temporal_stream_can_refresh_dense_keyframes` asserts dense output is returned on keyframes and absent on sparse update steps. | Covered |
| KV cache expectations are represented correctly. | README model notes | README documents that this is predictor/feature caching, not causal KV caching, because V-JEPA attention is bidirectional. | Covered |
| Sparse hot paths avoid backend-to-host readbacks. | `tests/numerical_parity.rs` | `sparse_forward_hot_path_has_no_backend_readbacks` scans production `src/model.rs` and `src/temporal.rs` for `to_data`/`into_data` markers. | Covered |
| Trace decoding is opt-in and disabled hot path has no trace work. | `BenchTraceConfig` in `benches/autogaze_sparse_jepa_pipeline.rs` | Trace-disabled benchmark rows report `autogaze_trace_ms=0.000`; disabled code path returns before calling `trace_video_with_mode`. | Covered |
| Tiny Burn sparse forward is numerically checked against an independent implementation. | `tests/fixtures/vjepa_tiny_parity.py`, `tests/numerical_parity.rs` | `tiny_sparse_forward_matches_independent_torch_fixture` passes within `5e-4`. | Covered |
| Transformers-style Hugging Face V-JEPA2 loader and forward path are checked. | `tests/fixtures/vjepa_hf_tiny_parity.py`, `tests/numerical_parity.rs` | HF tiny parity passed with prediction diff `1.49e-8` and target diff `7.45e-8`. | Covered |
| Real V-JEPA 2.1 checkpoint loading is strict and real-weight forward parity is available. | `VJepaLoadOptions`, `tests/fixtures/vjepa_hf_real_micro_forward.py` | Local HF checkpoint strict load passed with `applied=456 missing=0 skipped=0 errors=0`; micro parity passed with prediction diff `2.00e-5`, target diff `4.17e-5`. | Covered |
| E2E throughput is measured across image resolutions and sparsity densities. | `benches/autogaze_sparse_jepa_pipeline.rs`, `docs/e2e-benchmark-results.md` | Checked-in table covers ndarray and WebGPU for 224x224, 384x384, 720p and densities 1%, 5%, 10%, 25%. | Covered |
| WebGPU and ndarray performance behavior is explained. | `docs/e2e-benchmark-results.md` | Report notes sparse V-JEPA stream is similar, but WebGPU E2E is slower because AutoGaze WebGPU generation dominates. | Covered |
| CUDA feature support compiles. | Cargo features and benchmark target | `cargo check --no-default-features --features cuda` and CUDA E2E bench target check passed. | Covered |
| CUDA E2E FPS is measured. | `docs/e2e-benchmark-results.md`, `docs/cuda-benchmark.md`, `docs/workflows/cuda-benchmark.yml` | This host has no usable CUDA runtime: no `/dev/nvidia*`, `nvidia-smi` cannot communicate with the driver, and the benchmark preflight skips CUDA. A manual self-hosted CUDA benchmark workflow template now fails if the emitted CSV has no data rows; publishing it under `.github/workflows/` requires workflow-file write permission. | Blocked |
| Package remains publishable. | Cargo package manifest | `cargo package --allow-dirty` passed with docs included. | Covered |

## Verification Commands

The following commands were run during the final audit:

```sh
cargo test --no-default-features --features ndarray
cargo test --test sparse_patchify_wgpu --no-default-features --features sparse-patchify-wgpu -- --nocapture
BURN_JEPA_VJEPA21_CHECKPOINT_DIR=/home/mosure/.cache/huggingface/hub/models--facebook--vjepa2-vitl-fpc64-256/snapshots/b3c1679b7c34d3255ef3547f27c7b226aefab26f \
BURN_JEPA_VJEPA21_FORWARD_PARITY=1 \
cargo test --test numerical_parity real_vjepa_checkpoint_loads_when_fixture_is_set \
  --no-default-features --features ndarray -- --ignored --nocapture
cargo check --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,sparse-patchify-wgpu,cuda
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,sparse-patchify-wgpu,cuda
cargo package --allow-dirty
```

## Current Status

The implementation, tests, parity fixtures, packaging, and ndarray/WebGPU E2E
benchmarks are aligned with the objective. The only incomplete item is real CUDA
runtime throughput. The code now preflights CUDA and reports a clean skip when
the runtime is unavailable; actual CUDA FPS needs a machine with a visible CUDA
driver and device nodes. Use `docs/cuda-benchmark.md` locally, or install the
`docs/workflows/cuda-benchmark.yml` template as a manual `cuda benchmark`
workflow, to produce the missing CUDA FPS artifact on that host.
