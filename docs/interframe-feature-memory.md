# Interframe JEPA Feature Memory

`InterframeJepaFeatureMemory` is a device-resident dense feature canvas for sparse
video pipelines. It keeps one feature vector per V-JEPA token position and
updates only the sparse positions observed in the current frame/window.

The intended use is the AutoGaze to sparse V-JEPA path:

1. AutoGaze or another sparsity driver chooses visible patch/tubelet tokens.
2. V-JEPA encodes only those sparse context tokens.
3. `InterframeJepaFeatureMemory` scatters the sparse encoder output into a
   persistent dense token grid.
4. Downstream dense-memory consumers read `features`, `observed`, and
   `age_frames` without forcing a CPU readback in the update path.

## API

```rust,no_run
use burn::backend::NdArray;
use burn::tensor::Tensor;
use burn_jepa::{
    InterframeJepaFeatureMemory, InterframeJepaFeatureMemoryConfig,
    SparseTokenMask, TokenGridShape,
};

type B = NdArray<f32>;

let device = Default::default();
let grid = TokenGridShape::new(8, 14, 14);
let mut memory = InterframeJepaFeatureMemory::<B>::new(
    InterframeJepaFeatureMemoryConfig::default(),
    1,
    grid,
    768,
    &device,
)?;

let mask = SparseTokenMask::evenly_spaced(grid.len(), 128);
let sparse_features = Tensor::<B, 3>::zeros([1, mask.len(), 768], &device);
let output = memory.update_masked_tokens(sparse_features, &mask, grid)?;

assert_eq!(output.features.shape().dims::<3>(), [1, grid.len(), 768]);
# Ok::<(), anyhow::Error>(())
```

`update_from_encoder_output` accepts `VJepaEncoderOutput` directly when the
sparse encoder output already owns the token indices.

## Semantics

- `features`: full dense token grid, shape `[batch, dense_tokens, embed_dim]`.
- `observed`: dense validity mask, shape `[batch, dense_tokens]`, with `1.0`
  after a token position has been observed at least once.
- `age_frames`: dense age counter, shape `[batch, dense_tokens]`, incremented
  only for already-observed tokens and reset to zero when a token is updated.
- `AssignLatest`: default update mode; repeated observations overwrite the
  previous feature.
- `Ema { alpha }`: optional smoothing mode; the first observation is assigned
  directly, then repeated observations blend with the prior memory value.

Sparse updates use cached row-index tensors plus `scatter_nd(..., Assign)`.
The update path does not call `to_data`, `into_data`, or construct host
`TensorData`; the host-side `reset_row_indices` helper is intentionally outside
the sparse update path.

`reset` clears the whole canvas. `reset_rows` clears selected batch rows from a
device index tensor, which is the preferred API for packed multi-stream video
serving. `reset_row_indices` is a host convenience for control-plane resets.

The current encoder-output convenience assumes each row has unique token
indices. For variable-width ragged masks, use grouped/fixed-width sparse batches
or carry a validity mask before writing into the dense memory.

## Benchmarks

The Criterion bench measures cached sparse updates, first-update plan build
cost, and row resets across token densities:

```sh
cargo bench --bench feature_memory --no-default-features --features ndarray
cargo bench --bench feature_memory --no-default-features --features flex
cargo bench --bench feature_memory --no-default-features --features dispatch,flex
cargo bench --bench feature_memory --no-default-features --features cuda
cargo bench --bench feature_memory --no-default-features --features webgpu
```

Benchmark rows encode the grid/resolution proxy, density, batch size, and sparse
token count, for example
`vjepa224_b4_density_10pct_b4_tokens157_of1568`.
