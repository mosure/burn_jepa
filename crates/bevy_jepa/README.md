# bevy_jepa

Bevy viewer for the `burn_jepa` sparse feature pipeline. It uses the same
`bevy_burn` device-sharing pattern as `bevy_burn_autogaze`, so Burn WebGPU
tensors can be uploaded directly into Bevy textures.

```bash
cargo run -p bevy_jepa -- --mask-source autogaze
cargo run -p bevy_jepa -- --mask-source patch-diff --image-size 128
```

The app renders four stage panels:

- input frame
- sparse token mask
- low-resolution JEPA token-cache PCA
- high-resolution AnyUp PCA

Press space to switch between AutoGaze-style sparse token masks and patch-diff
sparse token masks.

Patch-diff defaults to threshold-gated selection because the default 64px viewer
grid is tiny and this is faster than launching backend top-k. Set
`--patch-diff-threshold 0` when you specifically want top-k selection with a
smaller host readback for larger grids.

## Benchmarks

The viewer exposes a headless Criterion lane so its stage metrics can be
compared against the raw high-resolution pipeline benches without involving the
Bevy render schedule:

```bash
cargo bench -p bevy_jepa --bench viewer_pipeline -- --sample-size 10
cargo bench --bench highres_anyup_pca_pipeline --features webgpu -- highres_sparse_jepa_anyup_pca_e2e_wgpu/viewer64_sparse25
```

`*_core_only` measures the shared JEPA -> feature cache -> AnyUp -> PCA path.
`*_gpu_panels` adds the GPU display tensor preparation used by the Bevy app, and
`*_cpu_panels` makes backend-to-host display transfer cost explicit.
