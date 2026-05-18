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
Use `FeatureFrameRequest::high_res_features()`,
`FeatureFrameRequest::high_res()`, or `FeatureFrameRequest::full()` when
downstream code needs the dense high-resolution feature tensor itself.

AnyUp quality depends on the supplied AnyUp module. Tiny test configs are useful
for checking control flow and device residency, but they are randomly
initialized and can produce overly smooth high-resolution PCA displays. Use a
loaded upstream AnyUp checkpoint and `AnyUpAttentionMode::UpstreamMasked` for
exact upstream Python parity, or `EfficientLocal` for the portable NATTEN-style
path used in performance runs.

README and paper-gallery visuals should be generated from real package
manifests, not the tiny smoke modules. The gallery example supports
`--model-manifest` for the V-JEPA 2.1 package, `--anyup-model-manifest` for the
AnyUp package, and `--config` to render a single representative lane. The
current README frame was generated from `patchdiff_50` at 256px with
`AnyUpAttentionMode::UpstreamMasked`, so low-res PCA comes from real V-JEPA 2.1
token-cache features and high-res PCA comes from the same PCA components after
the real AnyUp upsampler. The high-res display should be texture-aligned but
smoother than the token grid because AnyUp is still a dense full-resolution
upsampling stage from low-resolution patch features.

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
whether that batch actually updated the display basis; `pca_sample_frames` and
`pca_sample_window_frames` report the rolling frame window used for that update.
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
method still returns high-res AnyUp/PCA every processed batch. The default node
schedule is low-res-only; set `high_res_pca_every` explicitly when a stream
should spend stage-worker time on AnyUp.

The Bevy viewer uses the same node separation but keeps the live input preview
outside the stage worker. Source frames update the input panel immediately; the
JEPA/cache/PCA low-res stage owns one active async task plus one latest pending
frame. New camera frames overwrite that pending stage frame when the worker is
busy, so low-res work cannot build an unbounded queue. High-res AnyUp is not
run inline with this stage: scheduled high-res frames are copied from completed
low-res cache snapshots into a separate AnyUp task, also with a latest-frame
overwrite slot. Bevy metrics report input/low-res/high-res FPS and
drop/overwrite counts alongside the raw `FeatureFrameMetrics`.
For camera sources, pending frames stay as resized RGBA until the worker admits
them. The worker then converts the admitted frame to a Burn tensor, computes
patch-diff against the previous admitted stage frame, runs JEPA, and renders the
mask panel from the admitted sparse mask. The pipeline tests assert that this
mask matches `encoded.token_indices` and the cache scatter positions. This makes
the displayed mask a cache write map for the completed stage frame, not a
speculative mask for a newer preview frame.
High-motion patch-diff frames that select much of the token grid are promoted to
a dense ordered mask. The feature memory then uses dense assignment rather than
high-density sparse gather/scatter, and the encoder avoids exploring many
near-dense sparse widths. The viewer default cutoff is `0.60`, based on the
latest 256/512px WGPU viewer stability sweep: exact sparse widths are steady
once shapes are warm, but live high-density jitter can still trigger first-use
shape/autotune stalls, and dense full-grid inference is already competitive in
that regime. The Bevy RGBA patch-diff node also uses this cutoff for a sampled
high-motion precheck, so dense frames can bypass full per-patch scoring before
they enter JEPA. By default the Bevy viewer uses bucketed sparse encode with
exact cache writes: the displayed mask remains the cache-write mask, while the
encoder context is widened to stable token buckets. The default bucket list is
10%, 25%, and 50% of the current token grid, followed by the dense width;
the shared library config can use an empty density list for the legacy fixed
token step, while the Bevy CLI exposes that as `--legacy-sparse-mask-buckets`.
This preserves every threshold-selected patch but adds real extra context
tokens, so it is an approximate performance mode, not dummy padding.
`--sparse-encode-mode exact` restores one-to-one encode/write masks for
experiments.
`--prewarm-shape-buckets` is enabled by default and runs those bucket widths
during pipeline initialization, then resets encoder/cache/PCA state before
admitting live frames.
Patch-diff refresh is stateful but still bounded by the same context budget:
subthreshold residuals accumulate over time, old token positions receive
age-priority refreshes, and a deterministic blue-noise probe covers quiet
regions. These refresh modes are meant to reduce semantic drift in the
interframe cache when a patch changes slowly enough that no single frame crosses
the threshold. They never replace above-threshold motion patches, and dense
fallback still runs after refresh selection.

The current WGPU sparse-vs-dense crossover should be read at the full
JEPA+cache level, not from cache writes alone. A focused tiny JEPA+cache sweep
measured these mean latencies:

| Input density | 256px JEPA+cache | 512px JEPA+cache |
| ---: | ---: | ---: |
| 50% | 1.986 ms | 3.260 ms |
| 75% | 1.887 ms | 3.156 ms |
| 85% | 2.061 ms | 3.218 ms |
| 90% | 2.057 ms | 4.113 ms |
| 95% | 1.969 ms | 3.289 ms |
| 98% | 2.049 ms | 3.970 ms |
| 100% dense ordered | 1.864 ms | 4.292 ms |

The feature-cache-only sweep still favors dense ordered assignment for
near-full writes, especially at 512px, while the full Bevy viewer path adds
patch-diff, rolling PCA, display upload, and WGPU shape-specialization effects.
That is why the raw bench still includes near-dense rows, but the production
Bevy fallback is lower (`0.60`) for smoother live camera behavior.

The headless Bevy FPS-stability sweep exercises camera-like RGBA frames through
the real patch-diff path and writes CSV/Markdown artifacts:

```sh
BURN_JEPA_FPS_STABILITY_FRAMES=16 BURN_JEPA_FPS_STABILITY_WARMUP=4 \
cargo run -p bevy_jepa --features sparse-patchify-wgpu --example fps_stability
```

Latest WGPU sparse-patchify run:

| Mode | Resolution | Threshold | Dynamics | Write density | Encode density | Unique encode widths | p95 outer ms | Max outer ms |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| bucketed256 | 512 | 0.03 | static | 0.1% | 25.0% | 1 | 24.14 | 24.21 |
| bucketed256 | 512 | 0.03 | stable_10 | 10.0% | 25.0% | 1 | 24.59 | 25.26 |
| bucketed256 | 512 | 0.03 | stable_30 | 30.0% | 50.0% | 1 | 24.49 | 26.30 |
| bucketed256 | 512 | 0.03 | stable_60 | 100.0% | 100.0% | 1 | 20.89 | 22.01 |
| bucketed256 | 512 | 0.03 | jitter_60 | 100.0% | 100.0% | 1 | 22.80 | 25.21 |
| bucketed256 | 512 | 0.03 | low_contrast_60 | 33.5% | 50.0% | 1 | 24.96 | 26.42 |
| bucketed256 | 256 | 0.03 | low_contrast_60 | 34.5% | 50.0% | 1 | 9.34 | 9.46 |

The diagnostic exact mode remains available, but it is no longer the Bevy
default. In the same run, exact low-contrast variable-width masks at 256px had
10 unique encode widths and one 1.665 s max frame from first-use WGPU shape
specialization. Bucketed encode preserved the exact write mask, used one encode
width, and kept the row below 9.5 ms.

## PCA

`FeaturePcaProjector` follows the V-JEPA 2.1 feature-map visualization protocol:
compute PCA on dense patch features and map the first three principal components
to RGB. The default `semantic_rgb` display mode is intended for these
semantically coherent V-JEPA 2.1 features: it fits the PCA basis from a rolling
multi-frame sample of observed token-cache features, projects low- or high-res
features with that stable basis, and normalizes RGB channels with rolling
projected-feature statistics instead of per-frame min/max.

The projector supports two basis modes:

- Fixed components, including the identity initializer used by tests and smoke
  pipelines.
- Rolling Oja-style component updates through `update_rolling_tokens` or
  `update_rolling_nchw`.

`FeaturePcaUpdateConfig` controls whether and how often the pipeline updates the
PCA basis. This update node is independent from `FeatureFrameSchedule`: a frame
can update the rolling PCA basis without emitting either low-res or high-res
display artifacts, and a display artifact can use the last stable basis without
forcing an update. Updates consume a rolling device-resident window of
low-resolution token-cache snapshots. The Bevy viewer default performs an early
two-frame warmup update, then updates every processed low-res frame while
sampling from a 16-frame window, so live PCA is fit from temporal context instead
of a single cache snapshot. Each scheduled update runs configurable
orthogonalized Oja/power iterations, maintains a moving mean, nudges components
toward the observed covariance directions, aligns component signs with the
previous basis, then normalizes the basis. Because the basis is rolled forward
instead of recomputed from scratch, signs and axes remain stable across frames,
which reduces PCA color flicker. This is meant for live visualization and domain
adaptation of the display basis, not as a replacement for an offline PCA fit on a
large feature corpus.

The legacy `update_pca_online` config flag maps to
`FeaturePcaUpdateConfig::rolling_low_res_every(1)` for compatibility. New code
should prefer the explicit `pca_update` config when it needs independent cadence,
window, warmup, and iteration control. The pipeline update path uses
`InterframeJepaFeatureMemoryOutput::observed` as tensor-side weights, so
never-observed cache slots do not bias the rolling PCA basis toward zero.

Display normalization is device-resident. `semantic_rgb` keeps rolling
per-component center/spread statistics from the same observed token samples used
to update the PCA basis, then softly clips projected z-scores into RGB. This
keeps the color mapping robust to sparse-cache holes and outliers while avoiding
host readbacks or per-frame min/max flicker. `signed_unit` remains available as a
simple bounded signed projection mode for debugging.

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

The raw E2E matrix includes `viewer256_sparse100` and `viewer512_sparse100`
rows for the V-JEPA 2.1 trained-resolution viewer paths. The Bevy app defaults
to the 512x512 sparse-encoding path with a 32x32 token grid; 256x256 remains
available as the smaller 16x16-grid path. Compare these rows with the headless
Bevy wrapper bench to separate shared pipeline cost from display tensor
preparation:

```sh
cargo bench -p bevy_jepa --bench viewer_pipeline -- --sample-size 10
cargo bench --bench highres_anyup_pca_pipeline --no-default-features --features webgpu -- highres_sparse_jepa_anyup_pca_e2e_wgpu/viewer512_sparse100
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
`BURN_JEPA_PCA_UPDATE_WARMUP`, `BURN_JEPA_PCA_UPDATE_MIN_TOKENS`,
`BURN_JEPA_PCA_SAMPLE_WINDOW`, and `BURN_JEPA_PCA_MIN_SAMPLE_FRAMES` tune the
update work, cadence, and sample window.

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
