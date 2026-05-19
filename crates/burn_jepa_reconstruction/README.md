# burn_jepa_reconstruction

Small Burn-native decoder for inspecting sparse V-JEPA token-cache artifacts as
reconstructed RGB images.

Input is a low-res token feature grid `[batch, dim, grid_h, grid_w]`; output is
an image-like tensor `[batch, 3, h, w]`. The crate is intentionally modular so
`burn_jepa` and `bevy_jepa` can train, export, load, or disable it independently
from V-JEPA and AnyUp.

The recommended decoder is `patch_conv`: residual convs on the low-resolution
JEPA token grid followed by a learned patch projection. This avoids expensive
full-resolution hidden-channel convolutions while still giving each decoded
patch local token context. Legacy `residual_uniform` packages still load, but
new viewer-quality packages should prefer `patch_conv`.

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
  --image-size 256 \
  --frames 1 \
  --feature-source image \
  --max-samples 2048 \
  --steps 12000 \
  --batch-size 4 \
  --lr 4e-4 \
  --reconstruction-architecture patch-conv \
  --output target/burn_jepa_reconstruction-build/low_res_v1/jepa_reconstruction.bpk \
  --hidden-dim 512 \
  --residual-blocks-per-scale 4 \
  --shard-mib 20 \
  --deploy-dir target/burn_jepa_reconstruction/low_res_v1 \
  --overwrite-shards \
  --overwrite-deploy

cargo run --no-default-features --features ndarray --bin burn-jepa -- verify-reconstruction-bpk \
  --manifest target/burn_jepa_reconstruction/low_res_v1/manifest.json \
  --image-size 384
```

Use `--feature-source image` for Bevy/live sparse-cache reconstruction. The
legacy `--feature-source video --frames 2` target trains against repeated-frame
temporal V-JEPA features and is not compatible with the single-frame
feature-cache path used by the viewer.

For E2E PSNR from the actual low-res cache path:

```bash
cargo run --release --example reconstruction_e2e --no-default-features --features cuda -- \
  --jepa-manifest target/burn_jepa/vjepa2_1_base/manifest.json \
  --reconstruction-manifest target/burn_jepa_reconstruction/low_res_v1/manifest.json \
  --frame-dir target/burn_jepa_reconstruction_train_frames/video_023 \
  --image-size 256 \
  --frames 32
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

Current caveat: this decoder is trained from dense single-frame V-JEPA base
encoder tokens. Reconstructions from sparse persistent caches or TTT features
are useful diagnostics, but their PSNR is only directly comparable when the
active encoder and cache freshness match the training feature distribution.
