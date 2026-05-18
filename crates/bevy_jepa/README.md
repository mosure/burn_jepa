# bevy_jepa

Bevy viewer for the `burn_jepa` sparse feature pipeline. It uses the same
`bevy_burn` device-sharing pattern as `bevy_burn_autogaze`, so Burn WebGPU
tensors can be uploaded directly into Bevy textures.

The Bevy crate is intentionally a thin app wrapper. Shared live-pipeline policy
such as `FeatureFrameViewerConfig`, patch-diff thresholding, dense fallback,
bucketed sparse encode masks, PCA update cadence, and shape prewarm masks lives
in `burn_jepa`; `bevy_jepa` owns camera/static input, Bevy scheduling, UI, and
texture upload only.

```bash
cargo run -p bevy_jepa
cargo run -p bevy_jepa -- --source static --image-path /path/to/frame.png
cargo run -p bevy_jepa -- --source synthetic-local-motion --mask-source patch-diff
cargo run -p bevy_jepa -- --source synthetic-local-motion --mask-source patch-diff --image-size 256
cargo run -p bevy_jepa -- --source camera --anyup-weights /path/to/anyup_multi_backbone.pth --anyup-attention-mode upstream-masked
cargo run -p bevy_jepa -- --encoder-source tiny-test --source synthetic-local-motion
```

Install from git main with the package name as Cargo's positional crate
argument; `cargo install --package` is not available on all Cargo versions:

```bash
cargo install --git https://github.com/mosure/burn_jepa.git --branch main bevy_jepa --locked --force
```

The default source is the camera. Synthetic/local-motion input is only used when
`--source synthetic-local-motion` is selected explicitly; camera mode waits for a
real camera frame instead of feeding generated warmup imagery into the pipeline.
The default JEPA encoder source is trained encoder-only TTT V-JEPA 2.1, loaded
from a sharded `.bpk` package when `--model-manifest`,
`BURN_JEPA_MODEL_MANIFEST`, or
`target/burn-jepa-web/model/vjepa2_1_ttt/manifest.json` is available. A legacy
local `.mpk` is only used when explicitly passed with `--ttt-model` or
`BURN_JEPA_TTT_MODEL`. Use `--model-profile vjepa2_1_base` to switch to the
base f16 CDN package, `--encoder-source tiny-test` only for local wiring smoke
tests, and `--encoder-source base-checkpoint` when you intentionally want the
frozen base V-JEPA 2.1 encoder without TTT state.
Base-checkpoint mode defaults to
`~/.cache/burn_jepa/vjepa2_1_vitb_dist_vitG_384`; use
`--jepa-checkpoint-dir` and `--jepa-config` when the official checkpoint lives
elsewhere.
The default sparse mask source is patch-diff because it is image-driven and does
not require loading a separate AutoGaze model. Patch-diff is adaptive by
default: every patch whose score is at or above the threshold is updated,
`--min-context-density` is only a fallback floor for near-static frames, and
`--bootstrap-context-density` controls the first frame cache fill.
`--patch-diff-quality Q` mirrors `bevy_burn_autogaze` by setting the patch-diff
threshold to `1 - Q`; the no-arg default is quality `0.97`, threshold `0.03`.
The quality value only changes the threshold; it does not impose a fixed 97%
token density. The camera RGBA path removes uniform global RGB/luma shifts
before scoring and also includes relative-luma/chroma terms, so the threshold is
less tied to the scene's average brightness.
When patch-diff activates much of the token grid, the mask is promoted to a
dense ordered mask so the JEPA feature cache can use its dense update path
instead of paying sparse gather/scatter overhead for a high-density sparse
write. The default `--patch-diff-dense-fallback-density 0.60` keeps low- and
medium-density adaptive motion sparse and routes high-motion frames through the
dense ordered path. The latest 512px WGPU viewer stability sweep showed that
exact sparse widths are steady once warm, but live high-density jitter can still
trigger first-use shape/autotune stalls. Before running full per-patch scoring,
the camera RGBA path samples the patch grid and takes this dense path early when
the sampled frame is already above the cutoff.
The viewer defaults to shape-stable bucketed sparse encode with exact cache
writes (`bucketed-context`): threshold-selected patches still define the
low-res cache overwrite mask, while the encoder context is widened into stable
GPU buckets. The default buckets are grid-relative densities
`--sparse-mask-bucket-densities 0.10,0.25,0.50`, with dense as the final
fallback. `--legacy-sparse-mask-buckets` restores the fixed-width
`--sparse-mask-bucket-tokens 256` stepping. This never drops threshold-selected
patches, but it adds real extra context tokens, so use `--sparse-encode-mode
exact` when an experiment needs encode tokens to match the displayed write mask
exactly.
`--prewarm-shape-buckets` is enabled by default to move bucket specializations
to startup; use `--no-prewarm-shape-buckets` to disable it.
The pipeline image size is at least 256x256, defaults to 512x512 sparse
encoding, and is rounded up to a multiple of the 16px V-JEPA patch size. The
default token grid is 32x32; `--image-size 256` uses the smaller 16x16-grid
path.

The default view renders three stage panels:

- input frame
- sparse token mask
- low-resolution JEPA token-cache PCA

High-resolution AnyUp PCA is hidden when `--high-res-pca-every 0` (the
default) and appears as a fourth panel when AnyUp is enabled. Open the compact
`controls` submenu, or press `C`, to switch TTT/base model packages, 256/512
input size, AnyUp cadence and attention mode, patch-diff threshold, and bounded
refresh modes without crowding the default viewer.

The viewer preprocesses camera/static frames with the same ImageNet
mean/std normalization expected by V-JEPA and upstream AnyUp. When high-res PCA
is enabled, AnyUp prefers the sharded f16 `.bpk` package path; the legacy
`--anyup-weights` / `BURN_ANYUP_WEIGHTS` checkpoint path remains a fallback for
local parity work. Use `--anyup-attention-mode upstream-masked` for exact parity
with upstream Python's default AnyUp path, or `efficient-local` for the portable
NATTEN-style path used by the real-time pipeline.
If high-res PCA looks nearly uniform, first verify that the viewer loaded a real
AnyUp package or checkpoint; `--encoder-source tiny-test`, missing AnyUp
manifests, and tiny smoke modules are useful for wiring tests but are not
representative visualizations.

`--mask-source autogaze` is reserved for a real model-backed AutoGaze node. The
viewer now fails clearly instead of synthesizing an AutoGaze-looking moving
center prior, so any "autogaze" output must come from `burn_autogaze` rather
than from generated test motion.

The PCA basis update is decoupled from display emission. By default the viewer
fits the rolling low-resolution PCA basis after a two-frame warmup and then on
every processed low-res frame, using a 16-frame sample window. Stable features
across time define the color space without spending several seconds on the
cold-start identity basis, while sign-stable updates reduce PCA color flicker.
PCA display uses
the V-JEPA 2.1 dense-feature visualization
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
Patch-diff mask refresh is enabled by default for the live cache: slow
subthreshold changes accumulate, old token positions are age-refreshed, and a
small deterministic blue-noise refresh probes quiet regions. The extra writes
are capped by unused context budget, so high-motion threshold hits still win.
Use `--no-patch-diff-refresh` for legacy instantaneous masks.
Model loading prefers sharded `.bpk` package manifests. Exported packages store
floating-point records as f16 for deployment size, and the native/wasm loaders
upcast those records into the active backend dtype. Native JEPA runs check
`--model-manifest`, `BURN_JEPA_MODEL_MANIFEST`,
`target/burn-jepa-web/model/{model_profile}/manifest.json`, then an
auto-downloaded cache under `~/.burn_jepa/models/burn_jepa/{model_profile}`
before accepting a legacy explicit `--ttt-model ...mpk` override. The native
cache downloads from the same profile route wasm uses. The default is
`https://aberration.technology/model/burn_jepa/vjepa2_1_ttt/manifest.json`;
`--model-profile vjepa2_1_base` or `?model-profile=vjepa2_1_base` switches to
`https://aberration.technology/model/burn_jepa/vjepa2_1_base/manifest.json`.
Use `--model-base-url`, `--model-cache-dir`, or `--no-model-download` to control
that path; `BURN_JEPA_MODEL_PROFILE` / `BURN_JEPA_MODEL_NAME` select the native
auto-cache profile. Wasm accepts `?model-profile=vjepa2_1_base`,
`?model-base=http://127.0.0.1:8091` for a local directory containing
`manifest.json`, or `?model-manifest=...` for a specific manifest URL.
`?load-model=false` selects the tiny test encoder and skips all model shard
fetches. `?preload-only=true` checks shard fetching without starting the Bevy
app. Model shards are not included in the GitHub Pages artifact.

When `--high-res-pca-every` is positive, AnyUp also uses a sharded
`burn_anyup` package. Native runs check `--anyup-model-manifest`,
`BURN_ANYUP_MODEL_MANIFEST`,
`target/burn_anyup/{anyup_model_profile}/manifest.json`, then
`~/.burn_jepa/models/burn_anyup/{anyup_model_profile}`. The default route is
`https://aberration.technology/model/burn_anyup/anyup_multi_backbone/manifest.json`.
Use `--anyup-model-base-url`, `--anyup-model-cache-dir`,
`--no-anyup-model-download`, or wasm query params `?anyup-model-base=...` and
`?anyup-model-manifest=...` to override it.

```bash
cargo run --bin burn-jepa -- export-bpk \
  --config ../../configs/deploy/vjepa21-base-bpk-export.toml \
  --output ../../target/burn-jepa-web/model/vjepa2_1_base/jepa.bpk \
  --shard-mib 20 \
  --model-profile vjepa2_1_base \
  --deploy-dir ../../target/burn-jepa-cdn-upload/vjepa2_1_base \
  --overwrite-shards \
  --overwrite-deploy
cargo run --bin burn-jepa -- export-bpk \
  --config ../../configs/deploy/vjepa21-ttt-bpk-export.toml \
  --output ../../target/burn-jepa-web/model/vjepa2_1_ttt/jepa_ttt.bpk \
  --shard-mib 20 \
  --model-profile vjepa2_1_ttt \
  --deploy-dir ../../target/burn-jepa-cdn-upload/vjepa2_1_ttt \
  --overwrite-shards \
  --overwrite-deploy
cargo run --no-default-features --features ndarray --bin burn-jepa -- export-anyup-bpk \
  --weights ../../target/burn-anyup-checkpoints/anyup_multi_backbone.pth \
  --output ../../target/burn_anyup-build/anyup_multi_backbone/anyup.bpk \
  --shard-mib 20 \
  --model-profile anyup_multi_backbone \
  --deploy-dir ../../target/burn_anyup/anyup_multi_backbone \
  --overwrite-shards \
  --overwrite-deploy
python3 -m http.server 8091 -d ../../target/burn-jepa-web/model
npm run build:wasm
npm run serve
# open http://127.0.0.1:8080/?model-base=http://127.0.0.1:8091/vjepa2_1_ttt&source=static
# native auto-cache: cargo run -p bevy_jepa -- --model-base-url http://127.0.0.1:8091/vjepa2_1_ttt
# native explicit: cargo run -p bevy_jepa -- --model-manifest ../../target/burn-jepa-web/model/vjepa2_1_ttt/manifest.json
```

For a small local package/inference smoke:

```bash
cargo run --no-default-features --features ndarray --bin burn-jepa -- export-bpk \
  --config ../../configs/wasm/tiny-bpk-export.toml \
  --output ../../target/burn-jepa-wasm-model/jepa.bpk \
  --shard-mib 1 \
  --overwrite-shards \
  --deploy-dir ../../target/burn-jepa-wasm-model-upload \
  --overwrite-deploy \
  --allow-tiny-model
cargo build --release --target wasm32-unknown-unknown --no-default-features --features wasm
mkdir -p ../../target/burn-jepa-wasm-api/out
wasm-bindgen --target web --out-dir ../../target/burn-jepa-wasm-api/out \
  --out-name burn_jepa ../../target/wasm32-unknown-unknown/release/burn_jepa.wasm
npm run test:wasm-api
BURN_JEPA_WASM_MODEL_MANIFEST_URL=https://aberration.technology/model/burn_jepa/vjepa2_1_ttt/manifest.json npm run test:wasm-api
```

The Bevy schedule keeps input preview separate from stage processing. Camera
frames update the input panel as soon as they arrive; JEPA/cache/PCA work runs
on Bevy's async compute pool with one active low-res stage task and one latest
pending stage frame. If the low-res stage worker is still busy, a newer input
frame overwrites the pending stage frame instead of letting the queue grow. The
overlay reports input, low-res, and high-res FPS separately, plus in-flight,
dropped, and overwritten stage-frame counts. `--high-res-pca-every N` keeps
low-res token-cache PCA available every processed stage frame. Positive values
send completed low-res cache snapshots to a separate AnyUp worker with its own
latest-frame overwrite slot. The default `--high-res-pca-every 0` means AnyUp is
opt-in, so it cannot stall the camera -> mask -> JEPA -> low-res cache path.
The low-res PCA basis is adaptive by default: `--pca-update-every 1` performs
an early two-frame warmup fit, then updates the rolling Oja basis every processed
low-res frame while sampling from a 16-frame device-resident window. Use
`--pca-update-every 0` to lock the current basis, or
`--pca-sample-window-frames`, `--pca-min-sample-frames`, and
`--pca-update-iterations` to trade color stability against adaptation speed and
update cost.

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
mask can expand to the full token grid. Near-full masks are intentionally
promoted to the dense ordered path because high-density sparse shape churn and
scatter can be slower than dense assignment on GPU backends. The default cutoff
is `0.60`; use
`--patch-diff-dense-fallback-density 1.0` to promote only exactly full masks, or
lower it after measuring a backend where sparse shape churn is worse than dense
full-grid inference. A sampled high-motion precheck uses the same cutoff to
skip full patch-diff scoring when the frame is already clearly near-dense.
`--context-density` is retained for legacy fixed-budget patch-diff configs, but
the Bevy adaptive threshold path does not top-k cap tokens that pass the
threshold.

`--encode-path auto` is the default. `bevy_jepa` enables flex-gmm WGPU sparse
patchify by default, including the matching fused Burn-to-Bevy texture bridge
needed by the WGPU kernel stack. Auto routes non-dense masks through sparse
patchify while dense ordered masks stay on the dense path. Bucketed-context
sparse encode is also the default; use `--sparse-encode-mode exact` when stable
token-width buckets are not worth the extra real context tokens. Use
`--encode-path dense-patch` to force the portable dense-patch-embed plus
sparse-token path, or `--encode-path sparse-patchify` when you want the app to
force sparse patchify for diagnostics. Build with `--no-default-features` only
when you explicitly want the portable non-flex-gmm path.

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
