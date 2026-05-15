# Completion Audit

Date: 2026-05-13

Objective: support an efficient and numerically checked
`burn_autogaze` -> sparse image token projection -> sparse V-JEPA 2.1 temporal
pipeline with stable next-frame updates, optional dense keyframes, benchmark
evidence, and clear backend/runtime status.

## Checklist

| Requirement | Artifact | Evidence | Status |
|---|---|---|---|
| Sparse image tokens from AutoGaze map into V-JEPA tubelet tokens. | `src/autogaze.rs`, `src/sparse_patchify.rs`, `project_autogaze_generated_tokens`, `project_autogaze_generated_masks`, `sparse_mask_from_frame_token_pairs` | Unit tests cover generated-token projection without trace decoding, direct generated-token-to-mask projection, token projection, fill-to-count behavior, and partial stream windows. | Covered |
| Masked pixels are not patchified on the WGPU sparse path. | `src/model.rs`, `src/temporal.rs`, `VJepaEncoder::sparse_patchify_video_wgpu`, `forward_frame_tokens_sparse_patchify_wgpu` | `tests/sparse_patchify_wgpu.rs` compares sparse patchify encoder output against dense masked encoder output. | Covered |
| Temporal sparse stream supports next-frame updates instead of waiting for full 16-frame flushes. | `TemporalSparseJepaStream`, `TemporalSparseJepaState`, `TemporalSparseMaskState` | `tests/temporal.rs` covers projection, precomputed-mask entrypoints, keyframe counters, cached predictor-plan reuse, reset, sparse stream calls, and tubelet-sized rolling windows that are shorter than `config.num_frames`. | Covered |
| Dense keyframe refresh is available but not mandatory on sparse update steps. | `TemporalSparseJepaStreamConfig::with_dense_keyframe_refresh` | `temporal_stream_can_refresh_dense_keyframes` asserts dense output is returned on keyframes and absent on sparse update steps. | Covered |
| Full dense prediction on keyframes is opt-in. | `TemporalSparseJepaStreamConfig::with_dense_keyframe_prediction` | `temporal_stream_can_refresh_dense_keyframe_predictions` and `wgpu_temporal_stream_dense_keyframe_prediction_is_opt_in` assert dense prediction/target output is produced on keyframes only and remains disabled by default. | Covered |
| KV cache expectations are represented correctly. | README model notes | README documents that this is predictor/feature caching, not causal KV caching, because V-JEPA attention is bidirectional. | Covered |
| Stable sparse-context updates can skip repeated host mask projection. | `TemporalSparseJepaStream::forward_masks`, `forward_masks_sparse_patchify_wgpu`, `project_autogaze_generated_masks` | `tests/temporal.rs` verifies direct precomputed masks preserve keyframe cadence and predictor-plan reuse; `tests/sparse_patchify_wgpu.rs` verifies the WGPU sparse-patchify direct-mask path matches the dense masked stream and reuses sparse patchify, sparse encoder, and predictor plans. The E2E report now compares frame-token stream latency with precomputed-mask stream latency and uses the mask path for optimized E2E rows. | Covered |
| Sparse hot paths avoid backend-to-host readbacks. | `tests/numerical_parity.rs` | `sparse_forward_hot_path_has_no_backend_readbacks` scans production `src/model.rs` and `src/temporal.rs` for `to_data`/`into_data` markers. | Covered |
| Stable sparse hot paths avoid repeated positional tensor creation. | `SparseEncoderPlan`, `SparsePredictorPlan`, `TemporalSparseJepaStream` | `cached_sparse_forward_paths_do_not_rebuild_position_tensors` checks that cached encoder/predictor/WGPU stream hot paths do not rebuild `TensorData` or call `position_tensor`. | Covered |
| Trace decoding is opt-in and disabled hot path has no trace work. | `BenchTraceConfig` in `benches/autogaze_sparse_jepa_pipeline.rs`, `tests/benchmark_report.rs` | Trace-disabled benchmark rows report `autogaze_trace_ms=0.000`; the config branch skips the trace helper, so disabled runs do not clone the video tensor or call `trace_video_with_mode`. | Covered |
| Tiny Burn sparse forward is numerically checked against an independent implementation. | `tests/fixtures/vjepa_tiny_parity.py`, `tests/numerical_parity.rs` | `tiny_sparse_forward_matches_independent_torch_fixture` passes within `5e-4`. | Covered |
| Transformers-style Hugging Face V-JEPA2 loader and forward path are checked. | `tests/fixtures/vjepa_hf_tiny_parity.py`, `tests/numerical_parity.rs` | HF tiny parity passed with prediction diff `1.49e-8` and target diff `7.45e-8`. | Covered |
| Real V-JEPA 2.1 checkpoint loading is strict and real-weight forward parity is available where a reference fixture is present. | `VJepaLoadOptions`, `tests/fixtures/vjepa_hf_real_micro_forward.py`, `tests/fixtures/vjepa21_torchhub_real_micro_forward.py`, `tests/numerical_parity.rs` | HF-compatible fixtures run one-token micro parity against Transformers. The local official Meta `.pt` fixture strict-loads through the upstream 2.1 adapter with `applied=312 missing=0 skipped=0 errors=0` and matches `torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_base_384")` on micro and multi-tubelet 3x4-grid cases, including masked encoder context tokens. Latest max abs diffs: micro context `1.109e-5`, prediction `1.812e-5`, target `1.034e-5`; multi-grid context `1.335e-5`, prediction `1.597e-5`, target `2.921e-5`. | Covered |
| E2E throughput is measured across image resolutions and sparsity densities. | `benches/autogaze_sparse_jepa_pipeline.rs`, `docs/e2e-benchmark-results.md` | Checked-in warmup/3-rep median table covers ndarray, WebGPU, and CUDA AutoGaze for 224x224, 384x384, 720p and densities 1%, 5%, 10%, 25%, with 4-frame clip, 2-frame rolling/tubelet window, and streaming-cache diagnostic latency. | Covered |
| Checked-in benchmark evidence has regression coverage. | `tests/benchmark_report.rs` | The integration test parses the E2E report, asserts the ndarray/WebGPU/CUDA resolution-density matrix, verifies trace-off rows, checks rolling-window latency is below 4-frame clip latency, and checks the CUDA runbook/template reject header-only CSV output. | Covered |
| WebGPU and ndarray performance behavior is explained. | `docs/e2e-benchmark-results.md` | Report includes 720p stage metrics, direct mask projection metrics, and identifies AutoGaze generation as the bottleneck; low-density 720p WebGPU remains faster than ndarray while higher-density behavior is dominated by AutoGaze's short autoregressive GPU/readback loop. | Covered |
| CUDA feature support compiles. | Cargo features and benchmark target | `cargo check --no-default-features --features cuda` and CUDA E2E bench target check passed. | Covered |
| CUDA E2E FPS is measured. | `docs/e2e-benchmark-results.md`, `docs/cuda-benchmark.md`, `docs/workflows/cuda-benchmark.yml` | Local CUDA access now works. `nvidia-smi -L` sees the RTX PRO 6000, the CUDA TTT runtime smoke passed, and the CUDA E2E bench emitted rows for 224x224, 384x384, and 720p across 1%, 5%, 10%, and 25% density. The benchmark preflight and workflow template still reject header-only CSV output for CPU-only or sandboxed runners. | Covered |
| Static page shell and workflow status are honest. | `crates/bevy_burn_jepa/www`, `README.md`, `web/README.md`, `crates/bevy_burn_jepa/README.md` | The static shell remains checked in, but the Pages badge is removed and docs note that the deploy workflow is disabled remotely because GitHub reports Pages is unavailable for this repository plan. | Covered |
| Package remains publishable. | Cargo package manifest | `cargo package --allow-dirty` passed with docs included. | Covered |

## Verification Commands

The following commands were run during the final audit:

```sh
cargo test --no-default-features --features ndarray
cargo test --lib --no-default-features --features ndarray,autogaze-ndarray
cargo test --test temporal --no-default-features --features ndarray
cargo test --test sparse_patchify_wgpu --no-default-features --features sparse-patchify-wgpu -- --nocapture
cargo test --test numerical_parity sparse_forward_hot_path_has_no_backend_readbacks \
  --no-default-features --features ndarray
cargo test --test numerical_parity cached_sparse_forward_paths_do_not_rebuild_position_tensors \
  --no-default-features --features ndarray
BURN_JEPA_VJEPA21_CHECKPOINT_DIR=/home/mosure/.cache/burn_jepa/vjepa2_1_vitb_dist_vitG_384 \
BURN_JEPA_VJEPA21_WEIGHTS=model.pt \
BURN_JEPA_VJEPA21_FORWARD_PARITY=1 \
cargo test --test numerical_parity real_vjepa_checkpoint_loads_when_fixture_is_set \
  --no-default-features --features ndarray -- --ignored --nocapture
cargo bench --bench sparse_pipeline temporal_sparse_stream_hot_path_ndarray \
  --no-default-features --features ndarray -- --sample-size 10
cargo check --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,autogaze-ndarray,autogaze-cuda,sparse-patchify-wgpu,sparse-patchify-cuda,cuda
BURN_JEPA_RUN_CUDA_SPARSE_PATCHIFY=1 \
cargo test --test sparse_patchify_cuda \
  --no-default-features --features ndarray,cuda,sparse-patchify-cuda -- --nocapture --test-threads=1
BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS=ndarray \
BURN_JEPA_PIPELINE_BENCH_REPS=3 \
BURN_JEPA_PIPELINE_BENCH_WARMUPS=1 \
BURN_JEPA_PIPELINE_BENCH_1080P=false \
BURN_JEPA_PIPELINE_BENCH_TRACE=0 \
BURN_JEPA_PIPELINE_BENCH_OUT=target/codex-cleanup-bench/autogaze_sparse_jepa_ndarray_steady_trace_off.csv \
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,autogaze-ndarray,sparse-patchify-wgpu
BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS=webgpu \
BURN_JEPA_PIPELINE_BENCH_REPS=3 \
BURN_JEPA_PIPELINE_BENCH_WARMUPS=1 \
BURN_JEPA_PIPELINE_BENCH_1080P=false \
BURN_JEPA_PIPELINE_BENCH_TRACE=0 \
BURN_JEPA_PIPELINE_BENCH_OUT=target/codex-cleanup-bench/autogaze_sparse_jepa_webgpu_steady_trace_off.csv \
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,webgpu,autogaze-ndarray,autogaze-webgpu,sparse-patchify-wgpu
BURN_JEPA_RUN_GPU_TRAINING_SMOKE=1 \
cargo test --no-default-features --features ndarray,cuda \
  --test gpu_training_smoke cuda_ttt_training_smoke_runs_when_requested -- --nocapture
BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS=cuda \
BURN_JEPA_PIPELINE_JEPA_BACKENDS=sparse-patchify-cuda \
BURN_JEPA_PIPELINE_BENCH_REPS=3 \
BURN_JEPA_PIPELINE_BENCH_WARMUPS=1 \
BURN_JEPA_PIPELINE_BENCH_1080P=false \
BURN_JEPA_PIPELINE_BENCH_TRACE=0 \
BURN_JEPA_PIPELINE_BENCH_OUT=target/codex-cuda-live/autogaze_sparse_jepa_cuda_trace_off.csv \
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,autogaze-ndarray,autogaze-cuda,sparse-patchify-cuda,cuda
cargo test --test benchmark_report --no-default-features --features ndarray
cargo package --allow-dirty
```

## Current Status

The implementation, tests, parity fixtures, packaging, and ndarray/WebGPU/CUDA
E2E benchmarks are aligned with the objective. CUDA is no longer blocked on this
host: the opt-in runtime training smoke passes and the warmed CUDA E2E run
produces data rows. The remaining release operational item is still external to
the crate code: install `docs/workflows/cuda-benchmark.yml` under
`.github/workflows/` if repository workflow-file permissions become available,
so the same CUDA artifact gate can run on a self-hosted runner.
