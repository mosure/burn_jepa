# burn_jepa

[![test](https://github.com/mosure/burn_jepa/workflows/test/badge.svg)](https://github.com/mosure/burn_jepa/actions?query=workflow%3Atest)
[![deploy github pages](https://github.com/mosure/burn_jepa/workflows/deploy%20github%20pages/badge.svg)](https://github.com/mosure/burn_jepa/actions?query=workflow%3A%22deploy+github+pages%22)

Burn-native sparse-token V-JEPA 2.1 inference and training primitives.

The crate mirrors the shape of `burn_autogaze`, but the model surface is the
Meta V-JEPA 2.1 encoder/predictor recipe:

- dense RGB video or image grid to patch/tubelet tokens
- sparse context and target token masks without expanding work back to the full grid
- sparse 3D positional encoding for `[frame, row, col]` token coordinates
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

- `sparse_vjepa_tiny_forward_ndarray`: 16.25 ms to 16.46 ms
- `sparse_predictor_hot_path_ndarray/16_sequence_tokens`: 170.63 us to 171.96 us
- `sparse_predictor_hot_path_ndarray/24_sequence_tokens`: 216.15 us to 217.89 us
- `sparse_predictor_hot_path_ndarray/32_sequence_tokens`: 265.98 us to 270.69 us

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
