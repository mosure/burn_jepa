# burn_jepa_reconstruction

Small Burn-native decoder for inspecting sparse V-JEPA token-cache artifacts as
reconstructed RGB images.

Input is a low-res token feature grid `[batch, dim, grid_h, grid_w]`; output is
an image-like tensor `[batch, 3, h, w]`. The crate is intentionally modular so
`burn_jepa` and `bevy_jepa` can train, export, load, or disable it independently
from V-JEPA and AnyUp.

The decoder is a compact `1x1` projection plus nearest-upsample residual
blocks. It is not a replacement for AnyUp: it is a diagnostic low-res token
decoder used to reveal stale sparse-cache writes, collapsed features, or
temporal artifacts in a way that PCA can hide.

```rust
use burn::tensor::Tensor;
use burn_jepa_reconstruction::{
    JepaReconstructionConfig, JepaReconstructionDecoder, reconstruction_psnr_scalar,
};

type B = burn::backend::NdArray<f32>;

let device = Default::default();
let config = JepaReconstructionConfig::default();
let decoder = JepaReconstructionDecoder::<B>::new(config.clone(), &device)?;
let features = Tensor::<B, 4>::ones([1, config.input_dim, 24, 24], &device);
let rgb = decoder.forward(features);
```

Training is exposed through `fit_reconstruction_decoder` for modular/offline
experiments. The expected dataset item is `(low_res_jepa_features, target_rgb)`
where `target_rgb` is the denormalized square image crop in `[0, 1]`. The
parent `burn-jepa` CLI can train and shard f16 `.bpk` bundles for native and
wasm deployment. Use `train-reconstruction-bpk` for viewer-quality weights;
`export-reconstruction-bpk` only creates an untrained loader-smoke package.

```bash
cargo run --release --no-default-features --features wgpu --bin burn-jepa -- train-reconstruction-bpk \
  --backend wgpu \
  --jepa-manifest target/burn-jepa-web/model/vjepa2_1_base/manifest.json \
  --image-dir target/burn-jepa-vjepa21-ttt-ablation/data/frames \
  --image-size 512 \
  --frames 2 \
  --max-samples 64 \
  --steps 400 \
  --batch-size 4 \
  --lr 1e-4 \
  --lambda-l1 0.02 \
  --lambda-gradient 0.05 \
  --lambda-color 0.02 \
  --output target/burn_jepa_reconstruction-build/low_res_v1/jepa_reconstruction.bpk \
  --hidden-dim 128 \
  --shard-mib 20 \
  --deploy-dir target/burn_jepa_reconstruction/low_res_v1 \
  --overwrite-shards \
  --overwrite-deploy

cargo run --no-default-features --features ndarray --bin burn-jepa -- verify-reconstruction-bpk \
  --manifest target/burn_jepa_reconstruction/low_res_v1/manifest.json \
  --image-size 384
```

For benchmarks:

```bash
cargo bench -p burn_jepa_reconstruction --no-default-features --features ndarray
cargo bench -p burn_jepa_reconstruction --no-default-features --features webgpu --bench reconstruction_forward
BURN_JEPA_RECONSTRUCTION_BENCH_LARGE=1 cargo bench -p burn_jepa_reconstruction --no-default-features --features webgpu --bench reconstruction_forward
BURN_JEPA_RECONSTRUCTION_BENCH_1024=1 cargo bench -p burn_jepa_reconstruction --no-default-features --features webgpu --bench reconstruction_forward -- jepa1024
```

The WebGPU benchmark syncs the backend inside the timed loop, so the reported
latency is completed decoder work rather than command enqueue time. Bevy's live
pipeline reports reconstruction PSNR only when synchronized measurements are
enabled; the default reconstruction hot path avoids scalar host reads.
