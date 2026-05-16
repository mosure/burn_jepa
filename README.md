# burn_jepa

[![test](https://github.com/mosure/burn_jepa/workflows/test/badge.svg)](https://github.com/mosure/burn_jepa/actions?query=workflow%3Atest)

Burn-native sparse-token V-JEPA 2.1 inference and training primitives.

The crate mirrors the shape of `burn_autogaze`, but the model surface is the
Meta V-JEPA 2.1 encoder/predictor recipe:

- dense RGB video or image grid to patch/tubelet tokens
- sparse context and target token masks without expanding work back to the full grid
- sparse 3D positional encoding for `[frame, row, col]` token coordinates
- sparse image-token to V-JEPA tubelet-token projection for AutoGaze-style masks
- persistent interframe dense feature memory with sparse device-side updates
- a workspace `burn_anyup` crate for AnyUp efficient local-window feature upsampling
- ViT encoder, predictor, dense predictive loss, and backend-neutral Burn modules
- safetensors loading path with PyTorch-to-Burn weight layout adaptation
- simple symmetric int8 quantization helpers for checkpoint/tooling experiments
- ndarray, WebGPU/WGPU, CUDA, and wasm feature gates
- CI, a wasm GitHub Pages viewer, benchmarks, and a `bevy_jepa` example crate

## Workspace Layout

- `src/` contains the publishable `burn_jepa` library and the canonical
  `burn-jepa` CLI only.
- `crates/burn_anyup/` contains the AnyUp Burn module, loader, tests, and
  benchmarks.
- `crates/bevy_jepa/` contains the native/wasm Bevy viewer app.
- `examples/` contains runnable diagnostics and artifact generators, including
  the E2E video-gallery renderer.
- `benches/` and `tests/` contain the root crate performance and correctness
  surfaces.
- `tools/` contains thin host-side wrappers for dataset/download orchestration;
  heavy Burn work should live in Rust examples, benches, tests, or crates.

## Usage

```rust,no_run
use burn::backend::NdArray;
use burn_jepa::{
    VJepaConfig, VJepaPipeline, VJepaVideoShape, make_context_target_masks,
};

type B = NdArray<f32>;

let device = Default::default();
let config = VJepaConfig::tiny_for_tests();
let pipeline = VJepaPipeline::<B>::random(config.clone(), &device);

let shape = VJepaVideoShape::new(1, 3, 4, 32, 32);
let frames = vec![0.0; shape.num_values()];
let video = VJepaPipeline::<B>::tensor_from_frames(&frames, shape, &device)?;

let grid = config.token_grid();
let (context, target) = make_context_target_masks(grid, 0.5);
let output = pipeline
    .model()
    .predict_dense_targets(video, &context, &target)?;

assert_eq!(output.predictions.shape().dims::<3>(), [1, target.len(), 32]);
# Ok::<(), anyhow::Error>(())
```

## Model Notes

The default config follows the upstream V-JEPA 2.1 public recipe: 384px inputs,
16px patches, 64 frames, tubelet size 2, RoPE-style sparse position handling,
modality embeddings, and hierarchical encoder outputs. For tests and examples,
`VJepaConfig::tiny_for_tests()` keeps the same data flow with a small model.

For repeated sparse predictor calls, build a `SparsePredictorPlan` once and use
`VJepaPredictor::forward_sparse_with_plan`. The plan stores backend gather
tensors, sequence indices, sorted sparse positions, and RoPE sin/cos tensors, so
the predictor hot path does not reconstruct masks or read tensors back to the
CPU.

For repeated sparse encoder calls, build a `SparseEncoderPlan` once and use
`VJepaEncoder::forward_sparse_tokens_with_plan` or
`VJepaEncoder::forward_video_with_plan`. The temporal stream does this
internally, so stable masks reuse backend token-index and positional tensors
instead of recreating RoPE/positional tensors on every sparse update.

For video streams, `TemporalSparseJepaState` adds a small runtime cache around
the sparse predictor. It reuses the predictor plan while sparse context/target
masks are stable and can optionally blend sparse context features between
keyframes for temporally stable outputs. The default `feature_blend = 1.0`
preserves exact per-frame sparse features; lower values opt into EMA-style
stability without adding backend-to-host reads. `next_is_keyframe` lets callers
run a dense/keyframe path at a fixed interval and sparse next-frame updates
between keyframes. This is deliberately a predictor/feature cache rather than a
transformer KV cache: V-JEPA attention is bidirectional, so causal KV reuse would
not be numerically equivalent to the full model.

`TemporalSparseJepaStream` composes the stream hot path: project sparse
per-frame image tokens, encode only the sparse V-JEPA context tokens, run the
cached sparse predictor, and keep mask/predictor keyframe counters aligned. Use
it for AutoGaze-style video loops where the dense/full path runs on keyframes
and sparse updates run between keyframes. Set
`with_dense_keyframe_refresh(true)` when the caller also wants full-grid encoder
features returned on keyframe steps; the default keeps keyframes sparse-only so
inter-frame updates do not pay for dense patchification.
Set `with_dense_keyframe_prediction(true)` when the caller also wants the dense
V-JEPA prediction/target path for the sparse context/target masks on keyframes;
this is opt-in so normal sparse updates do not clone the video or run dense
target encoding.
If the upstream pipeline already has V-JEPA context/target masks, call
`forward_masks` directly to skip per-frame image-token mask projection while
preserving the same internal keyframe cadence and cached predictor plan.

The stream call is window based and does not require `config.num_frames` frames.
For lowest latency, pass a rolling window as short as one V-JEPA tubelet
(`tubelet_size` frames) plus the sparse image-token ids for that same window on
each new frame. Use a full clip/keyframe refresh when exact full-window V-JEPA
context is required; tubelet-sized rolling updates are the low-latency sparse
path, not a causal KV-cache equivalent.

On the `sparse-patchify-wgpu` backend,
`forward_frame_tokens_sparse_patchify_wgpu` routes the stream context path
through `burn_flex_gmm` sparse patchification, so masked-out patches are not
patchified before the encoder. The stream caches the sparse patchify plan while
the context mask/grid/batch stay stable, and also caches the sparse encoder
plan, avoiding repeated backend index and positional tensor creation. The
generic `forward_frame_tokens` method remains backend-neutral and uses the dense
patch embed followed by token masking, but still reuses sparse encoder and
predictor plans when masks are stable.
Use `forward_masks_sparse_patchify_wgpu` for the lowest-overhead stable-mask
path when the caller can reuse precomputed V-JEPA masks.

`InterframeJepaFeatureMemory` keeps a full dense V-JEPA token-feature canvas
updated from sparse encoder observations. It returns dense features plus
`observed` and `age_frames` metadata so downstream dense or memory tasks can
distinguish fresh, stale, and never-observed token positions. Updates use cached
scatter indices and stay device-side; the host row-reset helper is separate from
the sparse update hot path. See `docs/interframe-feature-memory.md` for API
details and sparse-update benchmark commands.

`FeatureFramePipeline` composes the current high-resolution visualization
path: image tensor, caller-owned sparse token mask, sparse V-JEPA image encoder,
sparse update into `InterframeJepaFeatureMemory`, dense AnyUp full-resolution
feature decode, and tensor-side PCA display projection. `FeaturePcaProjector`
supports fixed components plus an explicit rolling PCA update node driven by
`FeaturePcaUpdateConfig`; the updater consumes accumulated low-resolution batch
features on its own cadence so display emission does not force basis updates.
The pipeline step avoids backend-to-host reads; display readback belongs at the
UI boundary. See `docs/highres-anyup-pca-pipeline.md` and
`benches/highres_anyup_pca_pipeline.rs` for the API and modular timing matrix.
The backend-neutral high-res entry point uses dense patch embedding followed by
sparse encoder tokens; `sparse-patchify-wgpu` and `sparse-patchify-cuda` expose
flex-gmm image patchify entry points that skip masked-out image patches before
the encoder. `FeatureFrameRequest` and `FeatureFrameSchedule` let callers emit
low-res PCA and high-res AnyUp/PCA at separate rates from the same sparse token
cache update. The Bevy viewer follows the production-friendly default of
updating low-res token-cache PCA every processed stage frame while decimating
the expensive high-res AnyUp/PCA stage.
For camera-style loops, `FeatureFrameStream` adds bounded in-flight
queueing, fixed-width sparse-mask batching, per-stage metrics, queue-wait
timing, monotonic frame sequencing, and reject/drop/overwrite backpressure
policies.

For AutoGaze-style sparse inputs, enable the optional `autogaze-*` feature and
use `project_autogaze_generated_tokens` to turn `burn_autogaze` generated token
ids into per-frame sparse image tokens, V-JEPA context masks, and target masks.
Use `project_autogaze_generated_masks` when downstream code only needs V-JEPA
context/target masks; it streams over generated token ids directly and skips the
intermediate per-frame token allocation. This keeps the sparse-patch path
independent of decoded fixation traces.

`SparseJepaTensorPipelineConfig` selects sparse input behavior through
`SparseJepaSparsityDriverConfig`:

- `FullFrame { target_tokens }` runs a dense/full-frame JEPA input policy with
  an evenly spaced target holdout.
- `AutogazeSparse(...)` projects per-frame AutoGaze image-token ids into V-JEPA
  tubelet context masks and derives target masks from the complement.
- `PatchDiff(...)` selects context patches from a simple frame-difference
  threshold, then fills any remaining context budget from the strongest patch
  deltas.
- `PrecomputedMasks { ... }` accepts caller-owned V-JEPA context/target masks
  for hot paths that already projected masks upstream.

The patch-diff driver necessarily reads the input tensor to score patches, so it
is intended as a heuristic/fallback mode. AutoGaze and precomputed-mask modes are
the preferred device-resident sparse pipeline entry points when the upstream
stream already has sparse tokens or masks.

Checkpoint loading expects a directory with `config.json` and `model.safetensors`:

```rust,no_run
use burn::backend::NdArray;
use burn_jepa::VJepaPipeline;

let device = Default::default();
let pipeline = VJepaPipeline::<NdArray<f32>>::load("/path/to/vjepa2_1", &device)?;
# Ok::<(), anyhow::Error>(())
```

Set `VJepaLoadOptions::weights_name` to a `.pt` / `.pth` file to load a
PyTorch checkpoint directly. The loader applies V-JEPA 2.1 prefix remaps for
`ema_encoder`, `target_encoder`, `encoder.module.backbone`, and
`predictor.module.backbone` by default, and safetensors loads use the
PyTorch-to-Burn tensor-layout adapter.

## Training CLI

The `burn-jepa` binary exposes the shared training core used by tests and
benchmarks:

```sh
cargo run --bin burn-jepa -- print-config > train.toml
cargo run --bin burn-jepa -- train-ttt --config train.toml
cargo run --bin burn-jepa -- eval-ttt --config train.toml --model ttt-model.mpk --batch-size 16 --no-full-grid
cargo run --bin burn-jepa -- train-jepa --config train.toml
cargo run --bin burn-jepa -- bench-ttt --config train.toml --steps 10
cargo run --bin burn-jepa -- print-experiment-config > experiment.toml
cargo run --bin burn-jepa -- experiment run --config experiment.toml
```

`train-ttt` loads an optional pretrained V-JEPA checkpoint, inserts
zero-initialized TTT adapter layers into the encoder, and trains single-frame
rollouts to match the teacher 3D/tubelet encoder output. Existing V-JEPA weights
are frozen by default (`ttt.freeze_pretrained = true`), so optimizer updates
target the added TTT modules unless the config explicitly opts into full
finetuning. The TTT state is chunked by `ttt.chunk_tokens`, updated in sequence,
and detached every `ttt.rollout_blocks` tubelets for block-rollout training.
`ttt.memory_update = "self_hidden"` is the deployable default: adapter fast
weights update from detached current hidden states. Set
`ttt.memory_update = "teacher_forced_diagnostic"` only for privileged
teacher-forced diagnostics. `ttt.supervision` controls the loss objective:
`"final_teacher"` matches final 3D teacher tokens, `"layer_local_teacher"`
matches same-depth teacher features at each TTT insertion point, and `"hybrid"`
runs layer-local training until the last `ttt.hybrid_final_steps`, then
finetunes against the final teacher. `ttt.predictor_layers` is also available
for opt-in auxiliary predictor-head ablations, but the default production path
keeps TTT in the encoder because predictor adapters do not change deployed
encoder features.
Training/eval reports keep deployable student inference separate from
teacher-forced diagnostics: `eval_loss`/`eval_cosine` are free-run metrics,
while `teacher_forced_eval_*` and `teacher_forcing_*_gap` quantify privileged
teacher-target adaptation. Reports also include per-layer TTT utilization
metrics such as fast-weight update RMS, adapter residual RMS, memory-read RMS,
parameter RMS, final-step gradient RMS, and state-reset/frozen-update temporal
diagnostics, including reverse-order and deterministic shuffle-order probes.
Those deeper probes are opt-in via `training.eval_utilization_diagnostics` and
`training.eval_temporal_diagnostics` so large eval/benchmark runs keep the
production sparse rollout cost unless the probes are explicitly requested.
Set `loss.predictor_loss_weight > 0` to train the normal sparse JEPA predictor
objective alongside feature distillation.
The legacy `ttt.target` field is still accepted for older config files, but new
configs should use `ttt.memory_update` and `ttt.supervision`.
The default TTT placement is `ttt.layer_placement = "first_last"`, which
resolves to the first and final encoder blocks. Local real-checkpoint CUDA
ablation selected it as the best short-run speed/quality default; use
`"thirds"` for the higher-capacity `[3, 7, 11]` ViT-B preset in longer
quality-focused runs.

Set `model.ttt_checkpoint_path` to continue adapter training from a saved
`ttt-model.mpk` while still loading the pretrained V-JEPA checkpoint for frozen
base/teacher weights. `eval-ttt` accepts `--batch-size`, `--full-grid`, and
`--no-full-grid`; full-grid comparison is useful for parity checks, while
`--no-full-grid` keeps sparse eval on the production rollout path.

Training sparse masks are configured through `training.mask`, with the legacy
`training.context_keep_ratio` kept as the default fallback for existing configs.
Supported mask policies are `keep_ratio`, `full_frame`, `autogaze_sparse`,
`patch_diff`, `precomputed_masks`, and `manifest_precomputed_masks`; dense JEPA
training and the TTT predictor auxiliary loss both resolve through the same
sparse-driver path as inference.
For GPU-resident sparse training, prefer `autogaze_sparse` or
`manifest_precomputed_masks` generated offline from AutoGaze; `patch_diff` is a
host-scored heuristic. Manifest-precomputed masks are per window; set
`training.batching = "group_uniform_masks"` for identical masks or
`training.batching = "fixed_width_masks"` for equal-width per-sample masks.
For long-form TBPTT over multiple videos, set
`training.batching = "packed_streams"` so each batch row owns an independent
carried TTT state keyed by manifest `clip_id`/`source`.
Variable-width ragged masks are accepted in TTT training/eval. The rollout
groups samples by per-tubelet token-count shape, runs exact-token
encoder/sparse-patchify calls per bucket, and pads only the returned tensors so
loss/reporting can ignore invalid positions with a valid-token mask.
Use `ttt.supervision = "hybrid"` to train TTT across early encoder layers
without paying full frozen-tail backward cost on every step. The training loop
uses a layer-local early-exit phase for those steps and restores full-encoder
free-run eval afterward. `training.cache_teacher_tokens = true` caches detached
final and layer-local teacher features for repeat windows.

Datasets can be synthetic smoke data or JSONL manifests. Video rows accept
either explicit frame paths or a frame directory:

```json
{"frames":["clip000/000.png","clip000/001.png"],"teacher_frames":["clip000_teacher/000.png","clip000_teacher/001.png"]}
{"frame_dir":"clip001","teacher_frame_dir":"clip001_teacher"}
{"image":"single_frame.png"}
```

Manifest paths are resolved relative to the manifest file. Loaded tensors use
Burn's `[batch, channels, frames, height, width]` layout so image, video, and
paired-video datasets feed the same training functions. The loader rounds
`dataset.frames` to a multiple of `tubelet_size`, rounds `dataset.image_size` to
a multiple of `patch_size`, applies `dataset.stride`, and pads short clips by
repeating the last frame.
See [docs/ttt-training.md](docs/ttt-training.md) for the training protocol,
block-rollout behavior, manifest details, and the experiment harness. Experiment
runs write `run-manifest.json`, `planned-trials.json`, `experiment-summary.json`,
`trial-metrics.csv`, and `ablation-summary.md` under the configured output
directory. Set `require_real_checkpoint = true` and `require_real_dataset = true`
when running open-set experiments so missing fixtures cannot be mistaken for
real results.

GPU runtime smoke:

```sh
BURN_JEPA_RUN_GPU_TRAINING_SMOKE=1 \
cargo test --test gpu_training_smoke webgpu_ttt_training_smoke_runs_when_requested -- --nocapture

BURN_JEPA_RUN_GPU_TRAINING_SMOKE=1 \
cargo test --no-default-features --features ndarray,wgpu \
  --test gpu_training_smoke wgpu_ttt_training_smoke_runs_when_requested -- --nocapture

cargo test --no-default-features --features ndarray,cuda \
  --test gpu_training_smoke cuda_training_preflight_reports_unavailable_runtime_without_initializing_cuda -- --nocapture

BURN_JEPA_RUN_GPU_TRAINING_SMOKE=1 \
cargo test --no-default-features --features ndarray,cuda \
  --test gpu_training_smoke cuda_ttt_training_smoke_runs_when_requested -- --nocapture
```

CUDA training dispatch preflights runtime access before constructing Burn's CUDA
backend; on Linux, `nvidia-smi` alone is not enough if `/dev/nvidia*` device
nodes are hidden from the process.

## Benchmarks

```sh
cargo bench --bench sparse_pipeline --no-default-features --features ndarray
```

The benchmark includes an end-to-end sparse forward and a predictor-only hot path
that reuses `SparsePredictorPlan`. On the local ndarray backend, a short run with
10 Criterion samples measured:

- `sparse_vjepa_tiny_forward_ndarray`: 17.043 ms to 17.062 ms
- `sparse_predictor_hot_path_ndarray/16_sequence_tokens`: 172.03 us to 172.86 us
- `sparse_predictor_hot_path_ndarray/24_sequence_tokens`: 223.40 us to 223.95 us
- `sparse_predictor_hot_path_ndarray/32_sequence_tokens`: 271.37 us to 273.84 us
- `temporal_sparse_predictor_hot_path_ndarray/cached_plan_32_sequence_tokens`: 273.75 us to 274.55 us
- `temporal_sparse_mask_projection_720p`: 8.6548 us to 8.9288 us
- `temporal_sparse_stream_hot_path_ndarray/cached_plan_from_frame_tokens_32_sequence_tokens`: 8.5135 ms to 8.5740 ms
- `temporal_sparse_stream_hot_path_ndarray/cached_plan_from_precomputed_masks_32_sequence_tokens`: 8.4181 ms to 8.4998 ms

The AutoGaze -> sparse V-JEPA pipeline bench projects sparse masks directly from
AutoGaze generated token ids. Trace collection is disabled in the benchmark
config by default; set `BURN_JEPA_PIPELINE_BENCH_TRACE=1` to opt into decoded
fixation traces and include the extra trace path timing. With tracing disabled,
the bench sets `autogaze_trace_ms=0.000` without calling the trace helper,
cloning the input tensor, allocating trace samples, or entering AutoGaze's trace
decoder.
The AutoGaze generation budget uses `AutogazeSparseJepaWindowConfig`, so top-k
scales with the projected sparse V-JEPA context budget instead of expanding
every sparse row to AutoGaze's maximum token cap. The default overfetch is 1.0;
raise `with_top_k_overfetch` only when a downstream quality target needs extra
AutoGaze candidates. This matters for WebGPU E2E throughput because AutoGaze
generation is an autoregressive loop with short launches and token-selection
readbacks.
The CSV includes both one-shot sparse pipeline timing and cached
`TemporalSparseJepaStream` timing (`temporal_stream_ms`,
`temporal_mask_stream_ms`, `rolling_temporal_stream_ms`,
`rolling_temporal_mask_stream_ms`, `temporal_e2e_pipeline_ms`,
`temporal_mask_e2e_pipeline_ms`, `rolling_temporal_e2e_pipeline_ms`,
`rolling_mask_temporal_e2e_pipeline_ms`,
`rolling_streaming_temporal_e2e_pipeline_ms`, and temporal FPS columns) plus
stage metrics for AutoGaze generation, token projection, direct mask projection,
plan construction, sparse patchification, encoder, and predictor segments.
See [docs/e2e-benchmark-results.md](docs/e2e-benchmark-results.md) for the
latest checked-in ndarray/WebGPU/CUDA E2E throughput table and CUDA runtime
status.
See [docs/cuda-benchmark.md](docs/cuda-benchmark.md) for the manual CUDA
benchmark workflow and local CUDA runbook.
See [docs/completion-audit.md](docs/completion-audit.md) for the current
prompt-to-artifact checklist.

The TTT rollout benchmark measures the same single-frame recurrent path used by
`train-ttt` without reading tensors back to the host in the hot path:

```sh
cargo bench --bench ttt_training -- --sample-size 10 --measurement-time 1 --warm-up-time 1
```

Burn 0.21 backend lanes are feature-gated independently. The crate keeps Burn
fusion enabled for GPU-capable backends. Use `flex` for the new portable CPU
backend, and `dispatch` with one or more concrete backend features to measure
Burn's runtime backend switch:

```sh
cargo bench --bench ttt_training \
  --no-default-features --features flex \
  -- ttt_training_step_flex/dense_b1 --sample-size 10 --measurement-time 1 --warm-up-time 0.2

cargo bench --bench ttt_training \
  --no-default-features --features dispatch,flex \
  -- ttt_training_step_dispatch_flex/dense_b1 --sample-size 10 --measurement-time 1 --warm-up-time 0.2

cargo bench --bench ttt_training \
  --no-default-features --features dispatch,webgpu \
  -- ttt_training_step_dispatch_wgpu/dense_b1 --sample-size 10 --measurement-time 1 --warm-up-time 0.2
```

The `ttt_sparsity_training_step_*` Criterion groups are the training-step
sparsity matrix. They sweep 10%, 50%, and 100% sparse token input, plus the
normal dense 100% baseline, and each sample includes student rollout, loss,
backward, and the optimizer step:

```sh
cargo bench --bench ttt_training \
  --no-default-features --features ndarray \
  -- ttt_sparsity_training_step_ndarray --sample-size 10 --measurement-time 1 --warm-up-time 1

BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo bench --bench ttt_training \
  --no-default-features --features ndarray,cuda \
  -- ttt_sparsity_training_step_cuda --sample-size 10 --measurement-time 1 --warm-up-time 1

cargo bench --bench ttt_training \
  --no-default-features --features ndarray,wgpu,sparse-patchify-wgpu \
  -- ttt_sparse_patchify_sparsity_training_step_wgpu --sample-size 10 --measurement-time 1 --warm-up-time 0.2

BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo bench --bench ttt_training \
  --no-default-features --features ndarray,cuda,sparse-patchify-cuda \
  -- ttt_sparse_patchify_sparsity_training_step_cuda --sample-size 10 --measurement-time 1 --warm-up-time 0.2
```

Use `ttt_sparsity_training_step_*` to isolate sparse-token transformer/backward
scaling. Use `ttt_sparse_patchify_sparsity_training_step_*` for the
adapter-training pixel-skip path; sparse rows use flex-gmm sparse patchify
instead of dense Burn patch embedding followed by gather.

On the local ndarray backend this tiny smoke benchmark measured
`ttt_single_frame_rollout_ndarray` at 4.7869 ms to 4.8055 ms.

Example trace-disabled E2E commands:

```sh
BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS=ndarray \
BURN_JEPA_PIPELINE_BENCH_TRACE=0 \
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,autogaze-ndarray,sparse-patchify-wgpu

BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS=webgpu \
BURN_JEPA_PIPELINE_BENCH_RESOLUTIONS=720p \
BURN_JEPA_PIPELINE_BENCH_DENSITIES=0.25 \
BURN_JEPA_PIPELINE_BENCH_TRACE=0 \
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,webgpu,autogaze-ndarray,autogaze-webgpu,sparse-patchify-wgpu

BURN_JEPA_PIPELINE_AUTOGAZE_BACKENDS=cuda \
BURN_JEPA_PIPELINE_JEPA_BACKENDS=sparse-patchify-cuda \
BURN_JEPA_PIPELINE_BENCH_RESOLUTIONS=224x224 \
BURN_JEPA_PIPELINE_BENCH_DENSITIES=0.05 \
BURN_JEPA_PIPELINE_BENCH_TRACE=0 \
BURN_JEPA_PIPELINE_BENCH_DENSE_PATCHIFY=0 \
cargo bench --bench autogaze_sparse_jepa_pipeline \
  --no-default-features --features ndarray,autogaze-cuda,cuda,sparse-patchify-cuda
```

## Correctness

The default test suite covers sparse mask behavior, dense target prediction
shapes, preprocessing, quantization, and the Bevy smoke path. It also runs
`tests/numerical_parity.rs`, which saves a tiny Burn V-JEPA 2.1 model to
safetensors, executes the Burn sparse target path, executes an independent
PyTorch fixture from those saved weights, and compares predictor and target
encoder outputs within `5e-4` max absolute error. The same test module also
round-trips the tiny model through `VJepaLoadOptions::load_model` with strict
missing-tensor checks. A second fixture creates a tiny Hugging Face
`VJEPA2Model`, saves Transformers-style safetensors, loads them through the Burn
loader, and compares dense-encoder predictor outputs against the installed
Transformers implementation. The Python fixtures are skipped only when the
required Python packages are unavailable.

The checked-in parity fixture validates the sparse Burn implementation against
an independent PyTorch implementation with synthetic tiny weights. Real Meta
V-JEPA 2.1 checkpoint parity should be run with a local checkpoint fixture before
claiming production weight parity. `tests/numerical_parity.rs` includes an
env-gated loader smoke for that fixture. The loader maps Transformers-style
VJEPA2 configs, fuses HF query/key/value tensors into the Burn `qkv` layout, and
strict-loads official upstream `.pt` / `.pth` checkpoints that use
`ema_encoder`, `target_encoder`, `encoder`, and `predictor` top-level modules:

```sh
BURN_JEPA_VJEPA21_CHECKPOINT_DIR=/path/to/vjepa2_1 \
BURN_JEPA_VJEPA21_WEIGHTS=model.pt \
BURN_JEPA_VJEPA21_FORWARD_PARITY=1 \
cargo test --test numerical_parity real_vjepa_checkpoint_loads_when_fixture_is_set -- --ignored
```

Set `BURN_JEPA_VJEPA21_FORWARD_SMOKE=1` to also run a sparse forward smoke after
loading. Set `BURN_JEPA_VJEPA21_FORWARD_PARITY=1` to compare real-checkpoint
forwards against the installed Hugging Face `VJEPA2Model` for HF-compatible
fixtures or against the official `facebookresearch/vjepa2` torch.hub V-JEPA 2.1
entrypoints for Meta `.pt` fixtures. The torch.hub fixture now checks both a
small sparse 2x2 case and a multi-tubelet 3x4 grid case, including official
masked-encoder context tokens, predictor targets, and predictor outputs.
Official `.pt` checkpoints store singleton modality tensors in a rank that
differs from this crate's persisted Burn layout; the loader reshapes the
encoder/predictor modality embeddings into the Burn module layout and keeps the
extra Burn predictor mask-token slots at their zero initialization when the
upstream checkpoint does not provide them. Hugging Face safetensors fixtures
slice and load the combined predictor mask-token tensor.
CUDA pipeline throughput is exposed by the benchmark harness, but it requires a
CUDA-capable device at runtime.
Set `BURN_JEPA_RUN_CUDA_SPARSE_PATCHIFY=1` to run the opt-in CUDA sparse
patchify parity smoke against dense masked V-JEPA output.

## Bevy Example

```sh
cargo run -p bevy_jepa
cargo run -p bevy_jepa -- --source static --image-path /path/to/frame.png
cargo run -p bevy_jepa -- --mask-source patch-diff --image-size 512
cargo run -p bevy_jepa -- --mask-source patch-diff --image-size 256
cargo run -p bevy_jepa -- --source camera --anyup-weights /path/to/anyup_multi_backbone.pth --anyup-attention-mode upstream-masked
cargo run -p bevy_jepa -- --encoder-source tiny-test --source synthetic-local-motion

cd crates/bevy_jepa
npm run serve
```

The Bevy viewer uses the shared `bevy_burn` WebGPU device path to render input
frames, sparse masks, low-resolution JEPA token-cache PCA, and high-resolution
AnyUp PCA as GPU texture uploads. The viewer pipeline runs at a minimum
256x256 JEPA input resolution, defaults to 256x256 sparse encoding, and rounds
larger requests to a multiple of the 16px V-JEPA patch size. Native camera input
uses a latest-frame overwrite queue, center-crops the camera frame to preserve
aspect ratio, and resizes the crop to the configured JEPA input size. The wasm
page uses `getUserMedia` plus the exported
`frame_input(...)` bridge. Native runs default to the trained encoder-only TTT
V-JEPA 2.1 checkpoint at
`target/burn-jepa-production-final/stage1-stream-tbptt/ttt-model.mpk`; pass
`--ttt-model`, set `BURN_JEPA_TTT_MODEL`, or use `--encoder-source
base-checkpoint`/`tiny-test` for explicit alternatives.

Input preview and stage processing are decoupled. The camera/input panel is
updated immediately from the latest source frame, while the sparse JEPA ->
feature-cache -> AnyUp/PCA stage runs on a single async worker with a one-frame
overwrite queue. When AnyUp or high-res PCA is slower than the camera, the
latest pending stage frame replaces the previous pending one; the overlay
reports input FPS, low-res FPS, high-res FPS, in-flight frames, drops, and
overwrites so backpressure is visible instead of silently stalling the app.
Camera preview frames remain center-cropped RGBA until they are admitted into
the stage worker, so pending preview updates do not build Burn tensors or run
patch-diff scoring. The sparse-mask panel is the completed stage write map:
the token positions shown are the positions overwritten in the low-resolution
feature cache for that same stage frame.

The viewer defaults to patch-diff masks so the sparse
mask node is driven by the current camera/static frame. Patch-diff uses
threshold-driven adaptive density by default: every patch above the threshold is
updated, `--min-context-density` is only a near-static fallback floor, and
`--bootstrap-context-density` controls the first frame token-cache fill.
`--patch-diff-quality Q` mirrors the AutoGaze viewer mapping by setting the
patch-diff threshold to `1 - Q`; it does not impose a fixed 85% token density.
`--mask-source autogaze` requires a real
model-backed AutoGaze node and fails clearly instead of generating a synthetic
moving center prior. The GitHub Pages workflow builds the wasm target and
publishes `crates/bevy_jepa/www`. If no AnyUp checkpoint is provided, the viewer
uses the tiny untrained AnyUp test module; use
`--anyup-weights` with `--anyup-attention-mode upstream-masked` for exact
upstream Python AnyUp parity. PCA display follows the V-JEPA 2.1 dense-feature
visualization protocol by mapping the first three rolling PCA components of
observed patch features to RGB with device-resident normalization.
