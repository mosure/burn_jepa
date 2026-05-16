# burn_anyup

Burn-native implementation of AnyUp universal feature upsampling.

This crate ports AnyUp feature upsampling and keeps the public input contract
from upstream AnyUp:

- high-resolution normalized image tensor `[B, 3, H, W]`
- low-resolution feature tensor `[B, C, h, w]`
- output upsampled features `[B, C, H', W']`

```rust,no_run
use burn::backend::NdArray;
use burn::tensor::Tensor;
use burn_anyup::{AnyUp, AnyUpConfig};

type B = NdArray<f32>;

let device = Default::default();
let model = AnyUp::<B>::new(AnyUpConfig::default(), &device)?;
let image = Tensor::<B, 4>::zeros([1, 3, 224, 224], &device);
let features = Tensor::<B, 4>::zeros([1, 384, 16, 16], &device);
let upsampled = model.forward(image, features, None, Some(4));

assert_eq!(upsampled.shape().dims::<4>(), [1, 384, 224, 224]);
# Ok::<(), anyhow::Error>(())
```

`q_chunk_size` trades latency for memory while preserving the same attention
result. In `efficient_local` mode it follows upstream NATTEN's low-resolution
row chunk convention; in `upstream_masked` mode it follows upstream
`CrossAttentionBlock`'s query-token chunk convention.

## Attention Modes

`AnyUpConfig::default()` uses `AnyUpAttentionMode::EfficientLocal`, the portable
Burn equivalent of upstream's `use_natten=True` conversion path: q/k are
projected with 1x1 convolutions, k/v are nearest-resized to the query
resolution, and attention is computed over a local dilated 2D window. It does
not require the NATTEN package or custom CUDA kernels.

Use `AnyUpConfig::upstream_masked()` or
`AnyUpConfig::default().with_attention_mode(AnyUpAttentionMode::UpstreamMasked)`
when exact parity with upstream Python's default `use_natten=False` path is
required. That path implements the original masked multi-head attention over
low-resolution feature tokens and is intended for correctness checks and
reference-quality runs rather than the real-time viewer path.

For repeated feature upsampling against the same image, prepare an exact image
context once and reuse it:

```rust,no_run
use burn::backend::WebGpu;
use burn::tensor::Tensor;
use burn_anyup::{AnyUp, AnyUpConfig};

type B = WebGpu<f32, i32>;

let device = Default::default();
let model = AnyUp::<B>::new(AnyUpConfig::default(), &device)?;
let image = Tensor::<B, 4>::zeros([1, 3, 224, 224], &device);
let features = Tensor::<B, 4>::zeros([1, 768, 14, 14], &device);
let context = model.prepare_image_context(image, Some([224, 224]), [14, 14]);
let upsampled = model.upsample_with_context(&context, features, Some(2));

assert_eq!(upsampled.shape().dims::<4>(), [1, 768, 224, 224]);
# Ok::<(), anyhow::Error>(())
```

`prepare_image_context` is numerically identical to the normal forward path; it
only exposes the image-dependent query and image-key work so callers can amortize
it across multiple feature maps.

For streaming paths with a fixed image resolution, cache the RoPE coordinate
grid as well. This avoids rebuilding and uploading the `[H*W, 2]` coordinate
tensor every frame while preserving the exact same encoded image result:

```rust,no_run
use burn::backend::WebGpu;
use burn::tensor::Tensor;
use burn_anyup::{AnyUp, AnyUpConfig};

type B = WebGpu<f32, i32>;

let device = Default::default();
let model = AnyUp::<B>::new(AnyUpConfig::default(), &device)?;
let grid = model.prepare_image_grid([224, 224], &device);
let image = Tensor::<B, 4>::zeros([1, 3, 224, 224], &device);
let context = model.prepare_image_context_with_grid(image, &grid, Some([224, 224]), [14, 14]);
# Ok::<(), anyhow::Error>(())
```

## Sparse AnyUp

Sparse pipelines can decode only selected high-resolution feature positions and
apply them as sparse updates to a persistent high-resolution feature memory:

```rust,no_run
use burn::backend::WebGpu;
use burn::tensor::Tensor;
use burn_anyup::{
    AnyUp, AnyUpConfig, AnyUpHighResFeatureMemory, AnyUpHighResFeatureMemoryConfig,
    AnyUpSparseOutputPlan,
};

type B = WebGpu<f32, i32>;

let device = Default::default();
let model = AnyUp::<B>::new(AnyUpConfig::default(), &device)?;
let image = Tensor::<B, 4>::zeros([1, 3, 224, 224], &device);
let low_features = Tensor::<B, 4>::zeros([1, 768, 14, 14], &device);
let context = model.prepare_image_context(image, Some([224, 224]), [14, 14]);
let plan = AnyUpSparseOutputPlan::<B>::new(
    vec![0, 224 * 112 + 112, 224 * 224 - 1],
    [224, 224],
    [14, 14],
    1,
    model.config.window_ratio,
    &device,
)?;

let sparse = model.upsample_sparse_with_context(&context, low_features, &plan)?;
let mut memory = AnyUpHighResFeatureMemory::<B>::new(
    AnyUpHighResFeatureMemoryConfig::default(),
    1,
    [224, 224],
    768,
    &device,
)?;
let snapshot = memory.update_sparse_output(sparse)?;

assert_eq!(snapshot.updated_tokens, 3);
# Ok::<(), anyhow::Error>(())
```

`upsample_sparse_with_context` returns sparse high-resolution features
`[B, K, C]` plus the dense output indices `[B, K]`; it does not materialize
`[B, C, H, W]`. `upsample_sparse_low_features_with_context` accepts sparse
low-resolution encoder tokens `[B, K_low, C]` and low-grid indices, scatters
them into the small low-resolution grid, and then performs sparse high-resolution
decode. This is the path intended for sparse JEPA encoder output feeding a
high-resolution feature canvas.

`AnyUpHighResFeatureMemory` stores a persistent `[B, H*W, C]` canvas plus
observed/age metadata and supports reset/snapshot APIs. Its default write mode
is backend-aware: WebGPU/WGPU/ndarray use fast `scatter_nd` slice assignment,
while CUDA uses a scatter-add delta fallback to avoid the current Burn CUDA
`scatter_nd` illegal-address failure on this update pattern.

The sparse path avoids dense high-resolution decode/output and supports sparse
low-resolution inputs. It still uses the image-dependent AnyUp query encoder for
the full image context; reuse `prepare_image_context` when multiple sparse
updates share the same frame.

## Weights

`AnyUpLoadOptions` loads Burn-compatible safetensors and upstream-style PyTorch
state dicts:

```rust,no_run
use burn::backend::NdArray;
use burn_anyup::{AnyUp, AnyUpAttentionMode, AnyUpConfig, AnyUpLoadOptions};

let device = Default::default();
let config = AnyUpConfig::default().with_attention_mode(AnyUpAttentionMode::UpstreamMasked);
let mut model = AnyUp::<NdArray<f32>>::new(config, &device)?;
let report = AnyUpLoadOptions::default().load_into(&mut model, "anyup_paper.pth", &device)?;
println!("loaded {} tensors", report.applied.len());
# Ok::<(), anyhow::Error>(())
```

Raw upstream paper checkpoints store q/k projection weights in
`cross_decode.cross_attn.attention.in_proj_*`. The loader splits those fused
PyTorch tensors into the Burn `q_proj` and `k_proj` parameters. With
`UpstreamMasked`, this matches upstream Python's default paper path; with
`EfficientLocal`, it matches upstream's `use_natten=True` conversion behavior.

## Validation

The test fixture `tests/fixtures/anyup_tiny_parity.py` builds a deterministic
PyTorch AnyUp-compatible model, writes efficient and fused upstream-style
checkpoints, and compares Burn forward output numerically against both PyTorch
attention modes.
An ignored real-checkpoint parity test downloads or reads the published
`anyup_multi_backbone.pth` checkpoint, splits its fused paper attention weights,
and checks Burn output against both the upstream-masked reference path and the
efficient-local conversion path.

Run:

```sh
cargo test -p burn_anyup --no-default-features --features ndarray
cargo bench -p burn_anyup --bench anyup_forward --no-default-features --features ndarray
BURN_ANYUP_DOWNLOAD_REAL=1 cargo test -p burn_anyup --no-default-features --features ndarray \
  real_multi_backbone_checkpoint_matches_torch -- --ignored --nocapture
BURN_ANYUP_BENCH_LARGE=1 cargo bench -p burn_anyup --bench anyup_forward \
  --no-default-features --features webgpu -- anyup_forward_webgpu/jepa384
BURN_ANYUP_BENCH_LARGE=1 BURN_ANYUP_BENCH_LOW_PRECISION=1 cargo bench -p burn_anyup \
  --bench anyup_forward --no-default-features --features cuda -- cuda_f16/jepa224
BURN_ANYUP_BENCH_LARGE=1 cargo bench -p burn_anyup --bench anyup_forward \
  --no-default-features --features webgpu -- 'anyup_sparse_(context_decode|low_feature_decode|highres_update)_webgpu/jepa224'
```

The CUDA, WebGPU/WGPU, Flex, and Dispatch backend feature flags mirror the rest
of the workspace.

## Local Benchmark Snapshot

These numbers were measured on this workstation with Criterion's short
`--warm-up-time 0.1 --measurement-time 0.2 --sample-size 10` lane. JEPA-like
cases use `AnyUpConfig::default()` (`qk_dim=128`, `num_heads=4`) and dense
upsampled feature outputs. The encoder path uses a pointwise-linear fast path
for exact 1x1 convolutions; 3x3/reflect-padded convolutions still use Burn's
portable convolution operator.

| Backend | Lane | Case | Mean time | Throughput |
|---|---|---:|---:|---:|
| ndarray | full forward | 64px, grid 16, C=32 | 60.27 ms | 67.96 Kelem/s |
| ndarray | full forward | 224px, grid 14, C=768 | 7.01 s | 7.16 Kelem/s |
| ndarray | full forward | 384px, grid 24, C=768 | 33.87 s | 4.35 Kelem/s |
| WebGPU | full forward | 224px, grid 14, C=768 | 158.57 ms | 316.44 Kelem/s |
| WebGPU | image encode | 224px | 47.31 ms | 1.06 Melem/s |
| WebGPU | query encoder | 224px | 46.74 ms | 1.07 Melem/s |
| WebGPU | key encoder + pool | 224px, grid 14 | 47.09 ms | 1.07 Melem/s |
| WebGPU | context decode | 224px, grid 14, C=768 | 19.42 ms | 2.58 Melem/s |
| WebGPU f16 | full forward | 224px, grid 14, C=768 | 160.83 ms | 311.99 Kelem/s |
| WebGPU f16 | context decode | 224px, grid 14, C=768 | 19.85 ms | 2.53 Melem/s |
| CUDA | full forward | 224px, grid 14, C=768 | 161.37 ms | 310.94 Kelem/s |
| CUDA | context decode | 224px, grid 14, C=768 | 17.87 ms | 2.81 Melem/s |
| CUDA f16 | full forward | 224px, grid 14, C=768 | 152.07 ms | 329.95 Kelem/s |
| CUDA bf16 | full forward | 224px, grid 14, C=768 | 156.78 ms | 320.05 Kelem/s |
| WebGPU | full forward | 384px, grid 24, C=768 | 826.81 ms | 178.34 Kelem/s |
| WebGPU | full forward | 384px, grid 24, C=1024 | 944.72 ms | 156.08 Kelem/s |
| WebGPU | prepare image context | 384px, grid 24 | 748.11 ms | 197.11 Kelem/s |
| WebGPU | context decode | 384px, grid 24, C=768 | 80.13 ms | 1.84 Melem/s |
| WebGPU | context decode | 384px, grid 24, C=1024 | 101.67 ms | 1.45 Melem/s |

Sparse 224px/C768 WebGPU lanes with `BURN_ANYUP_BENCH_LARGE=1`:

| Lane | Density | Mean time |
|---|---:|---:|
| sparse context decode | 1% output pixels | 13.54 ms |
| sparse context decode | 5% output pixels | 12.71 ms |
| sparse context decode | 10% output pixels | 12.81 ms |
| sparse context decode | 25% output pixels | 14.06 ms |
| sparse context decode | 100% output pixels | 20.89 ms |
| sparse low-feature decode | 1% low tokens, 10% output pixels | 13.55 ms |
| sparse low-feature decode | 5% low tokens, 10% output pixels | 13.55 ms |
| sparse low-feature decode | 10% low tokens, 10% output pixels | 13.19 ms |
| sparse low-feature decode | 25% low tokens, 10% output pixels | 13.00 ms |
| sparse low-feature decode | 100% low tokens, 10% output pixels | 12.66 ms |
| high-res sparse memory update | 1% output pixels | 1.11 ms |
| high-res sparse memory update | 5% output pixels | 1.00 ms |
| high-res sparse memory update | 10% output pixels | 1.17 ms |
| high-res sparse memory update | 25% output pixels | 1.21 ms |
| high-res sparse memory update | 100% output pixels | 1.39 ms |

CUDA high-res sparse memory update at 224px/C768/10% output pixels measured
4.20 ms with the CUDA-safe scatter-add delta write mode.

The exact attention/decode lane is real-time at 224px once the image context is
prepared. Full exact forward is still not real-time in portable Burn tensor ops:
the high-resolution image encoder plus query/key image encoders dominate runtime.
Low precision on the current CUDA/WebGPU stack is not enough by itself, so the
next full-forward milestone is a backend-specific fused GroupNorm/SILU/1x1
encoder block that avoids repeated high-resolution intermediate tensors and
kernel launches.
