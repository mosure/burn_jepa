# burn_jepa

[![test](https://github.com/mosure/burn_jepa/workflows/test/badge.svg)](https://github.com/mosure/burn_jepa/actions?query=workflow%3Atest)
[![deploy github pages](https://github.com/mosure/burn_jepa/workflows/deploy%20github%20pages/badge.svg)](https://github.com/mosure/burn_jepa/actions?query=workflow%3A%22deploy+github+pages%22)

Burn-native sparse-token V-JEPA 2.1 inference and training primitives.

The crate mirrors the shape of `burn_autogaze`, but the model surface is the
Meta V-JEPA 2.1 encoder/predictor recipe:

- dense RGB video or image grid to patch/tubelet tokens
- sparse context and target token masks without expanding work back to the full grid
- sparse 3D positional encoding for `[frame, row, col]` token coordinates
- sparse image-token to V-JEPA tubelet-token projection for AutoGaze-style masks
- ViT encoder, predictor, dense predictive loss, and backend-neutral Burn modules
- safetensors loading path with PyTorch-to-Burn weight layout adaptation
- simple symmetric int8 quantization helpers for checkpoint/tooling experiments
- ndarray, WebGPU/WGPU, CUDA, and wasm feature gates
- CI, docs/page workflow, benchmarks, and a `bevy_burn_jepa` example crate

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

For AutoGaze-style sparse inputs, use `sparse_mask_from_frame_token_indices` with
the source `SparseImageTokenGrid` to project per-frame sparse image tokens into
the V-JEPA tubelet grid. This keeps the sparse-patch path independent of decoded
fixation traces.

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
`ema_encoder`, `target_encoder`, and `predictor` by default, and safetensors
loads use the PyTorch-to-Burn tensor-layout adapter.

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
- `temporal_sparse_stream_hot_path_ndarray/cached_plan_32_sequence_tokens`: 8.7408 ms to 8.7889 ms

The AutoGaze -> sparse V-JEPA pipeline bench projects sparse masks directly from
AutoGaze generated token ids. Decoded fixation traces are opt-in for diagnostics:
set `BURN_JEPA_PIPELINE_BENCH_TRACE=1` to include the extra trace path timing.

## Correctness

The default test suite covers sparse mask behavior, dense target prediction
shapes, preprocessing, quantization, and the Bevy smoke path. It also runs
`tests/numerical_parity.rs`, which saves a tiny Burn V-JEPA 2.1 model to
safetensors, executes the Burn sparse target path, executes an independent
PyTorch fixture from those saved weights, and compares predictor and target
encoder outputs within `5e-4` max absolute error. The same test module also
round-trips the tiny model through `VJepaLoadOptions::load_model` with strict
missing-tensor checks. The PyTorch fixture is skipped only when `python3` cannot
import `torch` and `safetensors`.

The checked-in parity fixture validates the sparse Burn implementation against
an independent PyTorch implementation with synthetic tiny weights. Real Meta
V-JEPA 2.1 checkpoint parity should be run with a local checkpoint fixture before
claiming production weight parity. `tests/numerical_parity.rs` includes an
env-gated loader smoke for that fixture:

```sh
BURN_JEPA_VJEPA21_CHECKPOINT_DIR=/path/to/vjepa2_1 \
BURN_JEPA_VJEPA21_WEIGHTS=model.safetensors \
cargo test --test numerical_parity real_vjepa_checkpoint_loads_when_fixture_is_set -- --ignored
```

Set `BURN_JEPA_VJEPA21_FORWARD_SMOKE=1` to also run a sparse forward smoke after
loading. CUDA pipeline throughput is exposed by the benchmark harness, but it
requires a CUDA-capable device at runtime.

## Bevy Example

```sh
cargo run -p bevy_burn_jepa

cd crates/bevy_burn_jepa
npm run serve
```

The native example renders a small live Bevy window and exercises the same Burn
pipeline shape used by the crate tests. The wasm page is intentionally static by
default so GitHub Pages can smoke-test the bundled demo shell without requiring
large model weights.
