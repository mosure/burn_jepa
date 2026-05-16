# bevy_jepa

Bevy viewer for the `burn_jepa` sparse feature pipeline. It uses the same
`bevy_burn` device-sharing pattern as `bevy_burn_autogaze`, so Burn WebGPU
tensors can be uploaded directly into Bevy textures.

```bash
cargo run -p bevy_jepa
cargo run -p bevy_jepa -- --source static --image-path /path/to/frame.png
cargo run -p bevy_jepa -- --source synthetic-local-motion --mask-source patch-diff --image-size 512
cargo run -p bevy_jepa -- --source synthetic-local-motion --mask-source patch-diff --image-size 256
cargo run -p bevy_jepa -- --source camera --anyup-weights /path/to/anyup_multi_backbone.pth --anyup-attention-mode upstream-masked
cargo run -p bevy_jepa -- --encoder-source tiny-test --source synthetic-local-motion
```

The default source is the camera. Synthetic/local-motion input is only used when
`--source synthetic-local-motion` is selected explicitly; camera mode waits for a
real camera frame instead of feeding generated warmup imagery into the pipeline.
The default JEPA encoder source is the trained encoder-only TTT V-JEPA 2.1
checkpoint at
`target/burn-jepa-production-final/stage1-stream-tbptt/ttt-model.mpk`; override
it with `--ttt-model` or `BURN_JEPA_TTT_MODEL`. Use
`--encoder-source tiny-test` only for local wiring smoke tests, and
`--encoder-source base-checkpoint` when you intentionally want the frozen base
V-JEPA 2.1 encoder without TTT state.
The default sparse mask source is patch-diff because it is image-driven and does
not require loading a separate AutoGaze model. Patch-diff is adaptive by
default: every patch whose score is at or above the threshold is updated,
`--min-context-density` is only a fallback floor for near-static frames, and
`--bootstrap-context-density` controls the first frame cache fill.
`--patch-diff-quality Q` mirrors `bevy_burn_autogaze` by setting the patch-diff
threshold to `1 - Q`; the no-arg default is quality `0.85`, threshold `0.15`.
The quality value only changes the threshold; it does not impose a fixed 85%
token density.
The pipeline image size is at least 256x256, defaults to 256x256 sparse
encoding, and is rounded up to a multiple of the 16px V-JEPA patch size. The
default token grid is 16x16; `--image-size 512` uses the larger 32x32-grid path.

The app renders four stage panels:

- input frame
- sparse token mask
- low-resolution JEPA token-cache PCA
- high-resolution AnyUp PCA

Without `--anyup-weights`, the viewer uses the tiny untrained AnyUp test module
so the high-resolution panel validates pipeline wiring rather than pretrained
feature quality. Use an upstream checkpoint plus `--anyup-attention-mode
upstream-masked` for exact parity with upstream Python's default AnyUp path, or
`efficient-local` for the portable NATTEN-style path used by the real-time
pipeline.

`--mask-source autogaze` is reserved for a real model-backed AutoGaze node. The
viewer now fails clearly instead of synthesizing an AutoGaze-looking moving
center prior, so any "autogaze" output must come from `burn_autogaze` rather
than from generated test motion.

The PCA basis update is decoupled from display emission. By default the viewer
updates the rolling low-resolution PCA basis every 4 frames using a 4-frame
sample window and multi-iteration updates, so stable features across time define
the color space without spending several seconds on the cold-start identity
basis. PCA display uses the V-JEPA 2.1 dense-feature visualization
protocol: the first three PCA components of observed patch features are mapped
to RGB with rolling, device-resident normalization so colors remain semantically
stable across sparse updates.

Native camera input follows the same one-frame overwrite queue as
`bevy_burn_autogaze`: the capture thread keeps only the latest RGBA frame, and
the viewer center-crops that frame to preserve aspect ratio before resizing it
into the configured JEPA input size. The shared sparse JEPA -> feature cache ->
AnyUp -> PCA pipeline then runs on the square crop. The wasm page
uses `navigator.mediaDevices.getUserMedia` and forwards frames through the
exported `frame_input(...)` function; `?source=static` uses generated or
`?image-url=...` frames without requesting a webcam.

The Bevy schedule keeps input preview separate from stage processing. Camera
frames update the input panel as soon as they arrive; JEPA/AnyUp/PCA work runs
on Bevy's async compute pool with one active stage task and one latest pending
stage frame. If the stage worker is still busy, a newer input frame overwrites
the pending stage frame instead of letting the queue grow. The overlay reports
input, low-res, and high-res FPS separately, plus in-flight, dropped, and
overwritten stage-frame counts. `--high-res-pca-every N` keeps low-res token
cache PCA available every processed stage frame while emitting the slower AnyUp
high-res PCA panel at a lower rate. The default is `8`; set it to `1` for
full-rate high-res AnyUp or `0` to disable high-res AnyUp/PCA emission.

For camera input, preview frames stay as center-cropped RGBA until they are
actually admitted into the sparse JEPA stage. The pending slot therefore does
not run patch-diff scoring or build Burn tensors on every Bevy update. Patch
diff is computed against the previous admitted stage frame, and the sparse-mask
panel visualizes that admitted stage mask. The high-resolution pipeline tests
validate that this mask is the same token set passed through `encoded.token_indices`
and into the cache scatter, so the panel marks the cache positions overwritten
by that stage rather than the newest camera preview frame.

Patch-diff defaults to threshold-gated selection with dynamic density. If only a
few JEPA patches cross the threshold, only those patches are updated except for
the independent `--min-context-density` floor; if the whole frame changes, the
mask can expand to the full token grid. `--context-density` is retained for
legacy fixed-budget patch-diff configs, but the Bevy adaptive threshold path
does not top-k cap tokens that pass the threshold.

`--encode-path auto` is the default. With the `sparse-patchify-wgpu` feature and
the V-JEPA or trained TTT encoder, auto routes image encoding through the
flex-gmm sparse patchify path so masked patches are skipped before the encoder.
Use `--encode-path dense-patch` to force the portable dense-patch-embed plus
sparse-token path, or `--encode-path sparse-patchify` when you want the app to
fail clearly if sparse patchify is unavailable for the selected build.

## Benchmarks

The viewer exposes a headless Criterion lane so its stage metrics can be
compared against the raw high-resolution pipeline benches without involving the
Bevy render schedule:

```bash
cargo bench -p bevy_jepa --bench viewer_pipeline -- --sample-size 10
cargo bench --bench highres_anyup_pca_pipeline --features webgpu -- highres_sparse_jepa_anyup_pca_e2e_wgpu/viewer512_sparse100
```

`*_low_res_cache_update` measures sparse mask generation, sparse JEPA encode,
and token-cache update without PCA display emission.
`*_pca_projection` adds low-resolution PCA projection using the cached rolling
PCA basis.
`*_full_anyup_decode` measures dense high-resolution AnyUp feature decoding
without display upload.
`*_display_upload_gpu` adds GPU display tensor preparation used by the Bevy app,
and `*_display_upload_cpu` makes backend-to-host display transfer cost explicit.
