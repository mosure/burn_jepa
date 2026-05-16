# Sparse JEPA to AnyUp PCA Pipeline

This pipeline is the image-visualization path for persistent sparse JEPA token
features:

1. Build or receive a `SparseTokenMask` for the current image.
2. Encode the selected image tokens with `VJepa2_1Model::encode_image` or,
   when `sparse-patchify-wgpu` / `sparse-patchify-cuda` is enabled, the
   flex-gmm image sparse patchify path.
3. Scatter the sparse encoder output into `InterframeJepaFeatureMemory`.
4. Reshape the dense token cache to a low-resolution NCHW feature grid.
5. Run `AnyUp::upsample_with_context` to recover full-resolution feature
   tensors.
6. Project high-dimensional features to display channels with
   `FeaturePcaProjector`.

The preferred API names are the concise frame pipeline aliases:
`FeatureFramePipeline`, `FeatureFrameStream`, `FeatureFrameRequest`, and
`FeatureFrameSchedule`. The older `SparseJepaAnyUpPca*` names remain as
compatibility aliases/wrappers for the original full-output path.

`FeatureFramePipeline` owns the composition above. The hot path accepts a
caller-owned sparse mask and image tensor, then returns node artifacts:

- `encoded`: sparse JEPA encoder output for the observed tokens.
- `token_cache`: dense token memory plus `observed` and `age_frames` metadata.
- `low_res.features`: dense token-cache view as `[B, C, grid_h, grid_w]`.
- `low_res.pca_display`: optional low-resolution PCA display tensor.
- `high_res.features`: optional AnyUp output as `[B, C, H, W]`.
- `high_res.pca_display`: optional full-resolution PCA display tensor.

The high-resolution display path has a dedicated PCA-only fast path. When
`FeatureFrameRequest::high_res_pca()` or the stream schedule requests
high-resolution PCA without `high_res_features`, the pipeline projects the
low-resolution JEPA token cache to PCA components first, then runs AnyUp over
only those display channels. This preserves the displayed image because AnyUp's
attention weights are independent of value channels, while avoiding
materialization of a full `[B, C, H, W]` high-resolution JEPA feature tensor.
Use `FeatureFrameRequest::high_res()` or `FeatureFrameRequest::full()` when
downstream code needs the dense high-resolution feature tensor itself.

The sparse update and PCA projection stay in tensor ops. There are no
backend-to-host reads in `FeatureFramePipeline::step_image_with_mask`;
host conversion belongs at the UI/display boundary.

## Stage Control

`FeatureFramePipeline` exposes both single-mask and fixed-width batch
entry points. The backend-neutral methods use dense patch embedding followed by
sparse token selection, while the sparse patchify methods skip masked-out image
patches before the encoder:

- `step_image_with_mask` for one uniform sparse mask across the pipeline batch.
- `step_image_with_mask_batch` for fixed-width per-frame masks with the same
  token budget per row.
- `step_image_with_mask_measured` and
  `step_image_with_mask_batch_measured` for opt-in stage timing.
- `step_image_with_mask_nodes_measured` and
  `step_image_with_mask_batch_nodes_measured` for optional low-res PCA and
  high-res AnyUp/PCA artifacts via `FeatureFrameRequest`.
- `step_image_with_mask_sparse_patchify_wgpu` /
  `step_image_with_mask_batch_sparse_patchify_wgpu` for WGPU flex-gmm sparse
  image patchify.
- `step_image_with_mask_sparse_patchify_cuda` /
  `step_image_with_mask_batch_sparse_patchify_cuda` for CUDA flex-gmm sparse
  image patchify.
- `step_image_with_sparse_patchify_plan_wgpu_measured` /
  `step_image_with_sparse_patchify_plan_cuda_measured` when the caller already
  owns a reusable `SparsePatchifyBatchPlan` and wants to avoid rebuilding
  coordinate tensors on stable masks.

`FeatureFrameMetrics` reports frame count, dense token count, sparse width,
valid sparse-token count, output pixels, `encode_path`, and per-stage
microsecond durations for sparse encode, cache update, token-cache view, rolling
PCA basis update, low-res PCA projection, AnyUp image context, AnyUp decode, and
high-res PCA projection. `pca_update_applied` and `pca_update_tokens` identify
whether that batch actually updated the display basis.
`FeatureFrameMeasureConfig::disabled()` is the default; backend sync
for true GPU wall-clock timing is an explicit opt-in through
`enabled_with_backend_sync()`.

Ragged masks are rejected by this hot path. For real per-frame masks, keep the
sparse token budget fixed, group frames by token budget, or run a batch-size-1
lane. This avoids hidden padded-token writes into the feature cache.

## In-Flight Stream

`FeatureFrameStream` is the bounded queue/orchestrator for camera-style
loops. It keeps model execution separate from flow control while handling:

- bounded queue capacity and backpressure (`reject_newest`, `drop_oldest`, or
  `overwrite_newest`);
- monotonic per-stream sequence validation;
- FIFO batch formation with fixed-width sparse masks;
- ordered `frame_ids` on output batches;
- queue-wait timing per frame; and
- stream stats for queued, dropped, emitted-batch, and emitted-frame counts.

The stream batches frames by concatenating `[1, 3, H, W]` tensors into
`[B, 3, H, W]` and building a `SparseMaskBatch` from the queued mask rows. It
does not reorder frames to improve grouping, because preserving output sequence
is more important for display correctness. If the front batch has variable mask
widths, the stream fails clearly so callers can reduce batch size or group by
token budget before enqueueing. The sparse patchify stream entry points cache
the last fixed-width `SparsePatchifyBatchPlan`, so stable masks reuse the device
coordinate tensor instead of rebuilding it every frame.

Use `process_next_ready_nodes` for stage-rate control. `FeatureFrameSchedule`
turns frame ids into per-batch `FeatureFrameRequest`s, so a stream can emit
low-res PCA every frame while only emitting full AnyUp/PCA every N frames. A
payload can therefore contain neither optional display artifact, only low-res
PCA, only high-res AnyUp/PCA, or both. The full legacy `process_next_ready`
method still returns high-res AnyUp/PCA every processed batch.

## PCA

`FeaturePcaProjector` supports two basis modes:

- Fixed components, including the identity initializer used by tests and smoke
  pipelines.
- Rolling Oja-style component updates through `update_rolling_tokens` or
  `update_rolling_nchw`.

`FeaturePcaUpdateConfig` controls whether and how often the pipeline updates the
PCA basis. This update node is independent from `FeatureFrameSchedule`: a frame
can update the rolling PCA basis without emitting either low-res or high-res
display artifacts, and a display artifact can use the last stable basis without
forcing an update. Updates consume the accumulated low-resolution token-feature
batch `[B, C, grid_h, grid_w]`, maintain a moving mean, nudge components toward
the observed covariance directions, then normalize and orthogonalize the basis.
Because the basis is rolled forward instead of recomputed from scratch, signs and
axes remain stable across frames, which reduces PCA color flicker. This is meant
for live visualization and domain adaptation of the display basis, not as a
replacement for an offline PCA fit on a large feature corpus.

The legacy `update_pca_online` config flag maps to
`FeaturePcaUpdateConfig::rolling_low_res_every(1)` for compatibility. New code
should prefer the explicit `pca_update` config. The pipeline update path uses
`InterframeJepaFeatureMemoryOutput::observed` as tensor-side weights, so
never-observed cache slots do not bias the rolling PCA basis toward zero.

Display normalization uses a bounded tensor transform instead of per-frame
min/max host readbacks. This keeps the render path predictable on WGPU/CUDA and
avoids synchronizing the backend just to colorize a frame.

## Benchmarks

Run the modular high-resolution pipeline benches with:

```sh
cargo bench --bench highres_anyup_pca_pipeline --no-default-features --features ndarray
cargo bench --bench highres_anyup_pca_pipeline --no-default-features --features wgpu
cargo bench --bench highres_anyup_pca_pipeline --no-default-features --features cuda
cargo bench --bench highres_anyup_pca_pipeline --no-default-features --features sparse-patchify-wgpu
cargo bench --bench highres_anyup_pca_pipeline --no-default-features --features sparse-patchify-cuda
```

By default the bench matrix runs small and mid-sized visualization cases. Set
`BURN_JEPA_HIGHRES_BENCH_LARGE=1` to include JEPA-like 224/384px cases with
768-dimensional token features:

```sh
BURN_JEPA_HIGHRES_BENCH_LARGE=1 cargo bench --bench highres_anyup_pca_pipeline --no-default-features --features wgpu
```

The raw E2E matrix includes a `viewer64_sparse25` row that mirrors the default
`bevy_jepa` viewer configuration. Compare it with the headless Bevy wrapper
bench to separate shared pipeline cost from display tensor preparation:

```sh
cargo bench -p bevy_jepa --bench viewer_pipeline -- --sample-size 10
cargo bench --bench highres_anyup_pca_pipeline --no-default-features --features webgpu -- highres_sparse_jepa_anyup_pca_e2e_wgpu/viewer64_sparse25
```

For stage-by-stage e2e latency and queue overwrite/drop accounting, use the
breakdown example:

```sh
BURN_JEPA_BREAKDOWN_ITERS=5 BURN_JEPA_BREAKDOWN_WARMUP=2 \
BURN_JEPA_STREAM_FRAMES=32 BURN_JEPA_STREAM_BURST=4 BURN_JEPA_STREAM_HIGH_EVERY=8 \
cargo run --no-default-features --features sparse-patchify-wgpu --example highres_breakdown
```

`BURN_JEPA_ANYUP_Q_CHUNK` overrides the AnyUp query chunk size for this report.
`FeatureFramePipelineConfig` defaults to a GPU-friendly chunk of 16 low-res
query rows. The GPU path benefits from larger chunks because it reduces
repeated small attention launches. CPU/ndarray can prefer smaller chunks because
the larger intermediate attention tensors are less cache-friendly.
Set `BURN_JEPA_PCA_UPDATE_EVERY=N` on the breakdown example to enable the
rolling low-res PCA update node every N input frames; `BURN_JEPA_PCA_UPDATE_ITERS`,
`BURN_JEPA_PCA_UPDATE_WARMUP`, and `BURN_JEPA_PCA_UPDATE_MIN_TOKENS` tune the
update work and cadence.

The matrix splits the path into:

- PCA projection of dense high-resolution features.
- Rolling PCA basis update from accumulated low-resolution batch features.
- AnyUp from a dense token-feature cache.
- Sparse token cache update plus AnyUp plus PCA.
- Tiny end-to-end sparse JEPA encoder plus cache plus AnyUp plus PCA.
- Tiny end-to-end flex-gmm sparse patchify JEPA plus cache plus AnyUp plus PCA
  when sparse patchify features are enabled.
- In-flight stream batching for batch sizes 1, 2, and 4.
- Cached-mask in-flight stream batching for repeated fixed-width sparse masks.

Use the split timings to distinguish sparse-cache update cost from AnyUp dense
full-frame decode cost. In this flow, AnyUp intentionally produces all high-res
tokens for display, so sparsity reduces JEPA encoder and token-cache work while
the dense full-resolution upsample remains proportional to output resolution.
The stream cache avoids rebuilding device-side sparse index tensors when the
next batch uses the same fixed token budget and token rows, which is the common
case for viewer and grouped real-mask pipelines.
