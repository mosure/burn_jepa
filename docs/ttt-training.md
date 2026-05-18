# TTT Training Protocol

The TTT path trains recurrent fast-weight updates to make single-frame JEPA
rollouts approximate the pretrained 3D/tubelet V-JEPA encoder.

## Model Shape

- Teacher: the loaded V-JEPA 2.1 encoder runs the normal 3D patch/tubelet path.
- Student: the same V-JEPA encoder receives one frame at a time through the
  image patch path. `ttt.insertion = "adapter"` inserts the historical
  zero-init residual `VJepaTttLayer` after configured transformer blocks.
  `ttt.insertion = "in_place_mlp"` instead wraps the selected block MLP and
  reuses its pretrained `fc2`/down-projection as the base fast weight.
- Fast state: adapter mode keeps `[batch, dim, dim]` fast-weight tensors.
  In-place MLP mode keeps a delta over the existing MLP down-projection with
  shape `[batch, mlp_hidden, dim]`, or banked variants of that shape when
  `memory_dynamics = "memory_alibi"`.
- Initialization: adapter mode zero-initializes the output projection, so the
  residual path starts as a no-op. In-place MLP mode zero-initializes the
  temporal update generator, so selected MLPs initially match the pretrained
  encoder exactly while still allocating a per-context down-projection delta.
- Memory update source: `ttt.memory_update = "self_hidden"` is the deployable
  default and updates fast weights from detached current hidden states.
  `ttt.memory_update = "teacher_forced_diagnostic"` is privileged and should be
  used only when measuring the teacher-forced gap.
- Supervision mode: `ttt.supervision = "final_teacher"` matches final 3D
  teacher tubelet features, `ttt.supervision = "layer_local_teacher"` matches
  same-depth teacher features at each configured TTT layer, and
  `ttt.supervision = "hybrid"` runs layer-local training before a shorter final
  teacher finetune controlled by `ttt.hybrid_final_steps`.

The default remains `adapter` for checkpoint compatibility with the existing
SC-TTT runs. The new `in_place_mlp` mode exists to directly ablate the
In-Place TTT design: selected MLP down-projections are reused as the base fast
weight, while the learned update machinery remains small and zero-effect at
initialization. More MLP layers can be converted by increasing `ttt.layers`,
but memory scales with `mlp_hidden * dim` per selected layer rather than
`dim * dim`.

## Code Organization

- `src/ttt/config.rs`: insertion mode, layer placement, rollout, memory-update source,
  supervision mode, backprop-mode, and freeze config.
- `src/ttt/state.rs`: per-layer fast-weight state and detach behavior.
- `src/ttt/layer.rs`: zero-init adapter TTT and in-place MLP fast-weight update.
- `src/ttt/encoder.rs`: V-JEPA encoder wrapper and single-frame rollout path.
- `src/ttt/model.rs`: pretrained/base model wrapping plus sparse predictor
  entrypoints.
- `src/training/config.rs`: CLI/file training config, backend selection, and
  validation.
- `src/training/ttt/mod.rs`: TTT distillation orchestration.
- `src/training/ttt/step.rs`: mask resolution, rollout selection, and
  teacher/student forward plumbing.
- `src/training/ttt/loss.rs`: feature and predictor distillation losses plus
  eval cosine helpers.
- `src/training/ttt/eval.rs`: free-run evaluation loop, explicit
  teacher-forced diagnostics, temporal ablations, and full-grid comparison pass.
- `src/training/ttt/metrics.rs`: TTT memory, mask, target-supervision, and
  per-layer utilization report metrics.
- `src/training/dense.rs`: normal dense JEPA training loop.
- `src/training/batch.rs`, `model_io.rs`, and `report.rs`: shared batch loading,
  checkpoint resolution, and report serialization helpers.

The crate exposes both root-level reexports such as `BurnJepaTrainConfig` and a
public `burn_jepa::training` namespace for callers that prefer explicit
training imports.

## Loss

For each training sample:

1. Load student and teacher video tensors in `[B, C, T, H, W]` layout.
2. Run the teacher video through the 3D encoder and detach final plus optional
   same-depth layer tokens.
3. Roll the student over single frames, updating TTT state frame by frame.
4. Compare the student to the selected teacher objective with feature MSE.
5. Backpropagate through the student rollout and update the configured trainable
   modules with AdamW.

The default config freezes pretrained V-JEPA weights and updates only the added
TTT modules. Set `ttt.freeze_pretrained = false` for full finetuning.
Set `model.ttt_checkpoint_path` to resume/continue adapter training from a
saved `ttt-model.mpk` while still resolving pretrained V-JEPA weights from
`model.checkpoint_dir`.
Set `training.lr_schedule.kind = "linear_warmup_cosine"` for long CUDA runs so
the adapter stage can warm up from a small first update and decay toward a
floor after the useful high-LR phase. Reports include `lr_schedule` and
`lr_stats` so production artifacts record the actual first, final, min, and max
learning rates.
Set `loss.predictor_loss_weight > 0` to add the normal sparse JEPA predictor
loss on top of feature distillation; the context/target masks come from
`training.mask` when configured, otherwise the legacy
`training.context_keep_ratio` field is used. The TTT training report records
`initial_loss`, `best_loss`, and `final_loss`; smoke tests assert finite losses
and a tiny synthetic convergence step.

## Evaluation Semantics

`eval_loss` and `eval_cosine` are deploy-style free-run metrics: the student
rollout does not receive teacher tokens as adapter update targets. When
`ttt.memory_update = "teacher_forced_diagnostic"`, reports also include
`teacher_forced_eval_loss`, `teacher_forced_eval_cosine`, and the
`teacher_forcing_*_gap` fields. Those teacher-forced fields are diagnostics for
how much privileged teacher-target adaptation helps; they are not production
student-inference metrics.

Reports include `target_supervision` to make memory updates and supervision
explicit. `memory_update` describes what updates fast weights, while
`supervision` describes the loss objective. This avoids conflating
teacher-forced diagnostics with deployable sparse student inference.

## Production Staging

Use a staged schedule for production candidates:

1. Generate real AutoGaze context/target masks into the manifests.
2. Train encoder-only TTT adapters with frozen pretrained V-JEPA weights,
   sparse context rollout, and frozen sparse patchify.
3. Evaluate free-run sparse quality, temporal diagnostics, utilization metrics,
   and cross-domain slices.
4. Start a short low-LR unfrozen continuation only after the frozen adapter run
   plateaus cleanly.

The unfrozen continuation is an ablation, not the default first run. It may
help reduce residual teacher-student mismatch after adapter saturation, but it
uses dense autodiff patch embedding during training and can damage pretrained
feature geometry if the learning rate or duration is too aggressive. Keep the
stage-1 frozen adapter checkpoint as the deployable sparse baseline.

## Stream and TBPTT Training

Long video training can carry TTT fast-weight state across adjacent manifest
windows. Enable this with `[training.stream]`:

```toml
[training.stream]
enabled = true
detach_between_steps = true
reset_on_clip_change = true
reset_on_scene_change = true
reset_on_non_monotonic_start = true
reset_interval_steps = 16
state_decay = 0.97
state_l2_weight = 0.000001
update_l2_weight = 0.00001
state_regularization_width = 64

[training.stream.curriculum]
enabled = true
initial_reset_interval_steps = 1
final_reset_interval_steps = 16
warmup_steps = 512
```

For single-stream debugging, use `training.batch_size = 1` with
`training.batching = "sequential"`. For production stream training, use
`training.batching = "packed_streams"` and set `training.batch_size` to the
number of independent stream lanes to train per step. The packed loader groups
manifest rows by `clip_id`/`source`, emits at most one window from each stream
per batch, and advances each stream independently across steps. The carried TTT
state is stored per stream key, then packed into the batch tensor for the
forward/backward step and unpacked back into per-stream state afterward. This
keeps the TTT memory device-resident and avoids the old single-stream
`batch_size = 1` throughput limit. If a shard exposes fewer streams than
`training.batch_size`, the loader emits a smaller valid batch instead of
duplicating a stream lane, and the report `samples` field records the actual
number of trained windows. Runtime validation still rejects duplicate stream
keys in one batch because `A0,A1` for the same stream requires sequential state
updates inside the batch.

The loader uses manifest metadata (`clip_id`, `domain`, `source`, and
`start_frame`) to reset state at new streams, non-monotonic windows, and
scheduled reset intervals. When `reset_on_scene_change = true`, repeated or
stitched manifests may additionally provide `original_stream`; this lets eval
or a deployment scene-cut detector reset state when a logical stream jumps to a
different scene while still measuring one continuous output stream. Manifest
stream rows must include `start_frame` plus either `clip_id` or `source`;
anonymous manifest streams are rejected so unrelated windows cannot silently
share state.
`detach_between_steps = true` gives a TBPTT boundary between windows; the
carried fast weights remain device tensors, but the previous window graph is not
retained. `state_decay` applies a scheduled stability decay after each step,
while `state_l2_weight` and `update_l2_weight` add device-side regularization
terms to the training loss. `state_regularization_width = 0` regularizes the
full fast-weight matrix; production CUDA configs use a sampled width of `64` so
long-horizon stabilization does not add a full extra backward path through every
`768 x 768` fast-weight matrix on each step.

Every stream window still receives a normal optimizer step. The curriculum does
not run hidden no-grad warmup rollouts: steps with a fresh/reset state train the
adapter to initialize from zero context, and carried steps train stability after
the TBPTT boundary. Reports expose this directly through
`stream.reset_optimizer_steps`, `stream.carried_optimizer_steps`,
`stream.mixed_optimizer_steps`, and traced loss rows tagged with
`stream_step = "reset"`, `"carried"`, or `"mixed"` when
`training.loss_trace_interval` is enabled. Set `loss_trace_interval = 0` for
throughput runs to avoid extra scalar readbacks.

## Latent Gaussian Regularization

The TTT loss supports an opt-in LeJEPA/SIGReg-style latent regularizer:

```toml
[loss.latent_regularization]
weight = 0.00001
mean_weight = 1.0
variance_weight = 1.0
covariance_weight = 0.25
target_variance = 1.0
covariance_sketch_dim = 64
```

The penalty is applied to the same aligned student tokens used by the feature
loss.  It is mask-aware, excludes padded ragged tokens, and uses a cheap
adjacent-feature covariance sketch rather than a dense covariance matrix.  Keep
the weight small for V-JEPA 2.1 teacher distillation: the 2026-05-18 ablation
selected `1e-5` because it was non-regressive, but the full 164-window eval was
effectively neutral relative to the unregularized selected checkpoint. Treat it
as a stability hook and diagnostic surface, not as a substitute for state
decay, update regularization, dense stabilization samples, or long-rollout
eval.

The sequence curriculum ramps the scheduled reset interval from short horizon
to longer carried-state blocks. The current production gate uses 16-frame
windows and `final_reset_interval_steps = 16`, so the trained deployment
horizon is 256 frames before the next scheduled refresh. Unbounded carried
state is not assumed: repeated-stream stress tests showed drift without a
reset, while the reset16 gate remained flat over 136 consecutive same-source
windows. Training reports include `stream.carried_steps`,
`stream.reset_steps`, `stream.detached_steps`, `stream.decayed_steps`, and the
effective final reset interval so experiment artifacts show whether the run
actually trained with persistent state. Packed reports also include
`stream.active_streams`, `stream.max_active_streams`, `stream.packed_batches`,
and `stream.max_packed_batch_size` so throughput results can be interpreted as
single-stream or multi-stream TBPTT.

The same stream config is honored by `eval-ttt`. With stream enabled, eval
carries free-run TTT state across adjacent manifest windows or packed stream
lanes, reports the same stream counters, and still runs teacher-forced/full-grid
diagnostics in fresh diagnostic states. Use
`configs/production/vjepa21-ttt-stream-eval-cuda.toml` for the production
long-form eval shape.

Long-run stress configs can repeat or stitch manifests without generating new
JSONL files. Set `dataset.repeat_count > 1` and choose
`repeat_mode = "continuous_streams"` for same-source carry,
`"stitched_stream"` for scene-switch recovery, or
`"adversarial_stitched_stream"` for worst-case scene ordering. `sample_limit`
can isolate the first N rows of a manifest, for example the current cactus-only
8x repeat gate.

## Carry-Forever Memory-ALiBi

The carry-forever line is separate from the reset16 production fallback. Its
goal is one persistent TTT state per stream with no hard reset after stream
creation:

```toml
[ttt]
memory_dynamics = "memory_alibi"
memory_alibi_half_lives = [8, 64, 512]
memory_alibi_read_weights = [0.45, 0.35, 0.20]
memory_alibi_update_weights = [1.0, 1.0, 1.0]
memory_clip_rms = 16.0

[training.stream]
enabled = true
detach_between_steps = true
reset_on_scene_change = false
reset_on_non_monotonic_start = false
reset_interval_steps = 0
state_decay = 1.0
```

This does not modify V-JEPA 2.1 attention or positional encoding. The ALiBi
idea is applied only to the added TTT memory: each TTT layer keeps multiple
fast-weight banks with log-spaced half-lives, reads a weighted mixture of those
banks, and updates each bank with its own decay. The state object persists
forever, while the banks provide time-scale-aware forgetting without an
external reset. Default `ttt.memory_dynamics = "ema"` preserves the historical
single fast-weight matrix and old checkpoint shape.

Use
`configs/production/vjepa21-ttt-stage1-stream-tbptt-carry-forever-alibi-cuda.toml`
to train from the latest reset16 checkpoint into the no-reset Memory-ALiBi
candidate. Use the matching long-rollout configs with `carry-forever-alibi` in
the filename for 1024+ window same-scene and adversarial stitched no-reset
gates. A carry-forever checkpoint should not replace the reset16 fallback until
it beats base sparse V-JEPA, the current no-reset EMA TTT path, and the reset16
checkpoint on the documented no-hard-reset gates.

The live image pipeline now has an explicit deployment-side TTT runtime policy
(`TttRuntimeStateConfig`) instead of carrying fast weights forever. The default
viewer policy updates fast weights, applies per-frame decay, and resets at the
trained rollout horizon without host reads. Native token/state stability probes
and collapse-guard actions are opt-in diagnostics (`metrics_interval_frames > 0`
plus `collapse_guard_enabled = true`) so normal deployment does not add periodic
tensor readbacks. This mirrors the training intent: fresh/reset windows teach
zero-state initialization, carried TBPTT windows teach stability, and runtime
decay/reset keeps arbitrary-length streams from drifting outside the trained
horizon. WebAssembly builds keep the decay and reset policy but skip synchronous
host-read diagnostics in the hot path.

The legacy `ttt.target` field still deserializes for old configs, but
`print-config` omits its default value; new configs should use
`ttt.memory_update` and `ttt.supervision`.

The eval report also records per-layer `utilization` probes:
`hidden_rms`, `memory_read_rms`, `adapter_delta_rms`,
`adapter_delta_to_hidden`, `fast_weight_rms`, `fast_update_rms`, trainable
parameter RMS, and the final-step gradient RMS when available. Temporal
diagnostics compare free-run output with reset-each-frame, reset-each-tubelet,
reverse-order, deterministic shuffle-order, and frozen-fast-update rollouts to
show whether the adapter is using temporal state rather than only acting as a
static residual.

These deep probes are opt-in because they add extra rollout passes on the first
eval batch. Set `training.eval_utilization_diagnostics = true` for per-layer
probe metrics and `training.eval_temporal_diagnostics = true` for the
reset/order/frozen-update ablations. Free-run, teacher-forced, and gap metrics
are always reported when `eval_steps > 0`.

## Mask Config

Training uses the same sparse mask resolver as the inference pipeline. The
serializable `training.mask` enum supports:

- `keep_ratio`: contiguous context tokens plus complement targets.
- `full_frame`: full-grid input with an evenly spaced target holdout.
- `autogaze_sparse`: AutoGaze-like image-token selection projected into V-JEPA
  tubelet context masks. The normal config is compact and uses a deterministic
  center-prior source; explicit per-frame token lists are an advanced override.
- `patch_diff`: a device-scored frame-difference heuristic. It keeps
  frame-difference and patch reduction on the active backend, then reads back
  only the patch score vector needed to build the host `SparseTokenMask`.
- `precomputed_masks`: explicit V-JEPA context and target token indices.
- `manifest_precomputed_masks`: per-window V-JEPA context and target indices
  read from the dataset manifest. Use this for real AutoGaze masks generated
  offline by `experiment prepare-data`.

If `training.mask` is omitted, old configs remain valid and resolve to:

```toml
[training]
context_keep_ratio = 0.75
```

Explicit mask configs live under `[training.mask]`:

```toml
[training.mask]
kind = "full_frame"
target_tokens = 16
```

```toml
[training.mask]
kind = "autogaze_sparse"
context_tokens = 32
target_tokens = 8
dilation = 1

[training.mask.image_grid]
height = 2
width = 2
```

```toml
[training.mask]
kind = "autogaze_sparse"
context_tokens = 32
target_tokens = 8

[training.mask.image_grid]
height = 2
width = 2

[training.mask.source]
kind = "frame_tokens"
frame_tokens = [[0, 3], [1], [2], [0]]
```

```toml
[training.mask]
kind = "precomputed_masks"
context_indices = [0, 2, 5, 7]
target_indices = [1, 3]
```

```toml
[training.mask]
kind = "manifest_precomputed_masks"
```

The dense JEPA loop uses `training.mask` for its predictor objective. The TTT
loop uses `training.mask` for the primary feature-distillation target when a
mask is configured, and also reports full-grid eval loss/cosine so sparse
training can be checked for dense-token regression. `training.sparse_rollout`
controls whether the student rollout itself is dense or sparse:

- `auto`: use target-mask sparse rollout when `training.mask` is configured and
  `loss.predictor_loss_weight = 0`.
- `dense`: always keep the dense single-frame rollout.
- `context_mask`: sparse rollout over the configured context mask. This matches
  the production AutoGaze sparse-input path.
- `target_mask`: force target-mask sparse rollout; this requires
  `training.mask` and is incompatible with predictor auxiliary loss.

Use `training.dense_samples` when a sparse stream run should also receive
all-token TTT distillation against the 3D/tubelet teacher:

```toml
[training.dense_samples]
enabled = true
warmup_steps = 128
interval_steps = 16
```

Dense-sample steps ignore `training.mask`, run the dense single-frame rollout,
and compute feature distillation over every token. Other steps keep the normal
configured sparse rollout, including frozen sparse patchify when available.
This is intentionally separate from `training.mask.kind = "full_frame"`:
`full_frame` still creates a JEPA holdout target mask, while dense-sample steps
train true full-token TTT behavior.

Set
`loss.predictor_loss_weight > 0` to add the normal sparse predictor auxiliary
loss. `training.sparse_patchify_training` controls the patch-embed boundary used
by sparse TTT training:

- `auto`: use the frozen sparse patchify bridge when the selected backend and
  enabled features support it.
- `dense_patch_embed`: always use Burn's normal dense patch embedding, then
  gather sparse tokens.
- `frozen_sparse_patchify`: require the bridge and fail if it is unavailable.

The bridge is currently available for WGPU with `sparse-patchify-wgpu` and CUDA
with `sparse-patchify-cuda`. In training it runs frozen sparse patchify at the
sparse token boundary so adapter-only TTT gradients still train the TTT/memory
layers while the frozen patch embedding does not receive gradients. Evaluation
can use the same frozen sparse patchify route without the autodiff wrapper.
This requires `ttt.freeze_pretrained = true`. Training and eval reports include
`rollout.frozen_sparse_patchify` so benchmark artifacts show which path was
actually used.

Training reports include `rollout.mode`, `rollout.student_tokens`,
`rollout.student_token_density`, `rollout.frozen_sparse_patchify`, and
`dense_samples.{dense_steps,sparse_steps}` so experiment artifacts show whether
a run actually trained dense rollout, target-mask sparse rollout, mixed dense
samples, or frozen sparse patchify rollout. Set
`training.loss_trace_interval = 0` for throughput-oriented GPU runs to avoid the
per-step scalar loss readback; the final loss is still reported, but `loss_trace`
is left empty.

Real AutoGaze masks are generated as a manifest preprocessing step:

```toml
[data.autogaze_masks]
checkpoint_dir = "/home/mosure/.cache/huggingface/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a"
backend = "cuda"
streaming = true
context_density = 0.2
target_density = 0.05
max_gaze_tokens_each_frame = 32
top_k_overfetch = 1.25
```

This writes `precomputed_context_indices` and `precomputed_target_indices` into
the train/eval manifests. With `streaming = true`, mask preparation keeps an
`AutoGazeStreamingCache` per `(domain, clip_id/source)` stream, resets at split
boundaries and non-monotonic `start_frame`s, and therefore matches downstream
online AutoGaze deployment more closely than independent per-window generation.
With `streaming = false`, each manifest window is generated independently,
which is useful only for isolated ablations. Real masks can now batch in three
modes:

```toml
[training]
batching = "sequential"          # legacy order
batching = "group_uniform_masks" # group identical context/target masks
batching = "fixed_width_masks"   # group equal-width per-sample masks
batching = "packed_streams"      # one window per manifest stream per batch
```

`fixed_width_masks` supports different mask indices per sample when each row has
the same context/target token count. The encoder and loss gather from batched
`[batch, tokens]` index tensors, and the CUDA sparse-patchify bridge can consume
fixed-width per-sample coordinate plans.

`packed_streams` is the long-form TBPTT mode. It groups manifest rows by stream
identity, selects one window from each stream per optimizer step, and hands the
training loop a packed batch whose TTT state rows are carried independently.
Use it with `training.stream.enabled = true` when a sorted manifest would
otherwise create unsafe `A0,A1` duplicate-stream batches. Uneven stream counts
or worker shards are handled by partial packed batches; uneven window counts
wrap per stream and reset through the non-monotonic `start_frame` rule.

Ragged per-sample masks are accepted for TTT training/eval. Internally the
rollout groups samples by per-tubelet token-count shape, runs exact-token
transformer calls for each bucket, then pads only the returned tensors. Padding
does not enter attention or fast-weight updates; the feature loss and cosine
diagnostics use a valid-token mask so padded positions do not affect metrics.
Ragged sparse patchify uses the same bucketed strategy on CUDA/WGPU, so
masked-out pixels are still skipped, with one sparse patchify call per bucket.

## Block Rollout

`ttt.rollout_blocks` controls truncated rollout training. A value of `1`
detaches state after every produced tubelet block; higher values keep gradients
through more temporal blocks before detaching. This keeps long clips trainable
without forcing the entire stream history into one autodiff graph.

`ttt.rollout_chunk_frames` controls the recurrence-preserving chunked rollout
scheduler. The default `16` batches patch embedding and transformer block work
across adjacent single-frame updates, but still applies every TTT fast-weight
update in frame order. Set it to `1` to force the old per-frame execution path
for debugging or exact performance ablations. CUDA sparse-patchify training uses
the same chunk scheduler for fixed-width/uniform masks; ragged masks still route
through bucketed exact-token batches.

`ttt.backprop_mode` makes the backward/runtime tradeoff explicit:

- `final_feature`: default full final-feature distillation objective.
- `truncated_final`: same objective, but uses
  `ttt.backprop_truncate_blocks` for the rollout detach cadence.
- `layer_local`: early-exit execution used by
  `ttt.supervision = "layer_local_teacher"` and the pre-finetune part of
  `"hybrid"`. It stops after the last configured TTT layer for that phase and
  compares same-depth teacher features instead of final teacher tokens.

For multi-layer TTT across the encoder, prefer `ttt.supervision = "hybrid"`.
The training loop uses layer-local early-exit steps first, then switches to
full final-feature passthrough for the last `ttt.hybrid_final_steps`. Eval and
model-file evaluation force full-encoder free-run rollout so layer-local
training speedups do not hide deploy-time quality regressions.

Set `training.prefetch_batches = true` for manifest runs to decode the next CPU
batch while the current GPU step is executing. The training report separates
`data_ms`, `prefetch_wait_ms`, and `host_to_device_ms` so genuine runtime gaps do
not get misattributed to JEPA compute. `training.cache_teacher_tokens = true`
caches detached final and layer-local teacher features inside a run, bounded by
`training.teacher_cache_max_entries`; leave it disabled for one-pass production
training unless repeated windows actually produce cache hits. Reports include
cache hit/miss/eviction counts in train/eval stage metrics.

## TTT Layer Placement

`ttt.insertion` controls the fast-weight architecture. Use `adapter` to keep
the existing residual TTT adapters. Use `in_place_mlp` to convert selected
existing encoder MLPs into In-Place-style adaptive MLPs that reuse the
pretrained down-projection as the base fast weight. `ttt.layer_placement`
controls which encoder layers are selected. Supported placements are `first`,
`middle`, `last`,
`first_last`, `thirds`, and `explicit`. `explicit` uses `ttt.layers` directly.
The default is `first_last`, which resolves to `[0, 11]` for a 12-layer ViT-B
encoder. It was selected as the smoke/training default because the real V-JEPA
2.1 CUDA ablation below matched the best held-out sparse loss while avoiding
the much larger backward cost of the three-adapter preset. Use `thirds` for the
higher-capacity `[3, 7, 11]` ViT-B preset when longer quality-focused runs can
afford the extra backward time.

For the in-place MLP ablation, compare at least matched `thirds` layers
(`[3, 7, 11]`) against `adapter` before trying higher coverage such as
`[1, 3, 5, 7, 9, 11]`. The latter is plausible because it reuses existing MLP
projections rather than adding separate residual blocks, but it also increases
fast-state memory by the MLP ratio and can increase backward cost.

Bounded real-checkpoint CUDA smoke results from 2026-05-18 used the V-JEPA 2.1
ViT-B checkpoint, 256px real manifest windows, manifest AutoGaze masks, and a
16-step budget unless noted:

| Insertion | Layers | Steps | Eval loss | Eval cosine | Samples/sec | Fast memory | Trainable |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapter | 3/7/11 | 16 | 0.3961 | 0.8301 | 0.1715 | 40.5 MiB | 13.5 MiB |
| in_place_mlp | 3/7/11 | 16 | 0.3981 | 0.8293 | 0.0901 | 162.0 MiB | 6.8 MiB |
| in_place_mlp | 1/3/5/7/9/11 | 4 | 0.4248 | 0.8197 | 0.0885 | 324.0 MiB | 13.6 MiB |

This is not a promotion-quality training run, but it is enough to avoid a bad
assumption: in-place MLP is architecturally sane and uses fewer trainable helper
parameters at matched layer count, but its per-context fast state is larger and
current Burn autodiff backward is slower. Keep `adapter` as the production
default until a longer in-place run beats it on held-out free-run quality at
matched wall-clock or an analytical/fused backward path changes the cost curve.

Production training should keep `ttt.predictor_layers = []`. The predictor
remains available as the normal JEPA prediction head for parity and auxiliary
loss experiments, but recurrent TTT adapters belong in the encoder for this
project. The deployed artifact is the sparse temporal encoder; predictor-side
TTT does not improve the encoder's sparse patch/token update path and violates
the goal of shipping a compact per-frame sparse encoder student.

## Dataset Modes

`dataset.kind = "synthetic"` is intended for smoke tests and CI. Manifest mode
supports:

- `image`: a single image row, reshaped to one frame.
- `frames` or `frame_dir`: a video row.
- `teacher_frames` or `teacher_frame_dir`: optional paired teacher video.

When no teacher path is provided, the student video is reused as the teacher
input. Paths are resolved relative to the manifest file.

## Dataset Requirements

- Inputs are decoded to RGB and resized into `[batch, channels, frames, height,
  width]` tensors.
- `dataset.frames` is rounded up to a multiple of `config.tubelet_size`; short
  manifests are padded by repeating the last selected frame.
- `dataset.image_size` is rounded up to a multiple of `config.patch_size`.
- `dataset.stride` subsamples manifest frames before padding.
- For pretrained V-JEPA 2.1 checkpoints, use the checkpoint/config resolution
  unless training a deliberately tiny smoke model.
- For TTT distillation, paired teacher clips should be temporally aligned with
  student clips. If the teacher path is omitted, the same clip is used for both.
- For robust temporal learning, clips should contain at least several tubelets;
  one-tubelet examples are smoke tests, not useful TTT training data.

## Commands

```sh
cargo run --bin burn-jepa -- print-config > train.toml
cargo run --bin burn-jepa -- train-ttt --config train.toml
cargo run --bin burn-jepa -- eval-ttt --config train.toml --model ttt-model.mpk --batch-size 16 --no-full-grid
cargo run --bin burn-jepa -- bench-ttt --config train.toml --steps 10
cargo bench --bench ttt_training
```

Set `training.backend` to `nd_array`, `flex`, `wgpu`, `web_gpu`, `cuda`, or
`dispatch` and enable the matching Cargo feature. For Burn 0.21 dispatch runs,
set `training.backend = "dispatch"` and optionally set
`training.dispatch_backend` to `auto`, `flex`, `nd_array`, `web_gpu`, `wgpu`, or
`cuda`. `auto` chooses the first enabled/available device in CUDA, WGPU, Flex,
NdArray order, skipping CUDA when the runtime preflight cannot open a device.
Use `eval-ttt --no-full-grid` for production sparse-rollout throughput. Use
`--full-grid` for slower parity diagnostics that run the sparse rollout and an
additional dense student rollout for full-token loss/cosine.
The primary eval fields remain free-run in both modes; teacher-forced results
are only exposed through the explicit `teacher_forced_*` diagnostic fields.
The deeper utilization and temporal ablations are controlled by
`training.eval_utilization_diagnostics` and
`training.eval_temporal_diagnostics`; leave them disabled for throughput
benchmarks and large ablation matrices unless those probes are the measurement
target.

The Criterion TTT bench includes an explicit sparse-token training-step matrix:

```sh
cargo bench --bench ttt_training \
  --no-default-features --features ndarray \
  -- ttt_sparsity_training_step_ndarray --sample-size 10 --measurement-time 1 --warm-up-time 1

cargo bench --bench ttt_training \
  --no-default-features --features flex \
  -- ttt_sparsity_training_step_flex --sample-size 10 --measurement-time 1 --warm-up-time 1

cargo bench --bench ttt_training \
  --no-default-features --features dispatch,flex \
  -- ttt_sparsity_training_step_dispatch_flex --sample-size 10 --measurement-time 1 --warm-up-time 1

cargo bench --bench ttt_training \
  --no-default-features --features dispatch,webgpu \
  -- ttt_sparsity_training_step_dispatch_wgpu --sample-size 10 --measurement-time 1 --warm-up-time 1

cargo bench --bench ttt_training \
  --no-default-features --features ndarray,wgpu \
  -- ttt_sparsity_training_step_wgpu --sample-size 10 --measurement-time 1 --warm-up-time 1

BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo bench --bench ttt_training \
  --no-default-features --features ndarray,cuda \
  -- ttt_sparsity_training_step_cuda --sample-size 10 --measurement-time 1 --warm-up-time 1

cargo bench --bench ttt_training \
  --no-default-features --features ndarray,wgpu,sparse-patchify-wgpu \
  -- ttt_sparse_patchify_sparsity_training_step_wgpu --sample-size 10 --measurement-time 1 --warm-up-time 0.2

BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo bench --bench ttt_training \
  --no-default-features --features ndarray,cuda,sparse-patchify-cuda \
  -- ttt_sparse_patchify_sparsity_training_step_cuda --sample-size 10 --measurement-time 1 --warm-up-time 0.2
```

Each `ttt_sparsity_training_step_*` sample includes sparse or dense student
rollout, feature loss, backward, and AdamW. The sparse rows use fixed-width
per-sample `SparseMaskBatch` inputs at 10%, 50%, and 100% token density. The
extra `density_100pct_dense_*` row is the normal full-token baseline, while
`density_100pct_sparse_*` isolates sparse-wrapper overhead when no tokens are
actually skipped. The benchmark does not read scalar losses back to the host in
the hot path.

The `ttt_sparse_patchify_sparsity_training_step_*` groups measure the same full
training-step surface, but route sparse rows through the flex-gmm frozen
sparse-patchify bridge. Those rows skip dense image patch embedding before the
TTT encoder and keep the benchmark on the pixel-skip path used by sparse
adapter-only training when pretrained JEPA weights are frozen.

The `ttt_tbptt_training_step_*` groups isolate TBPTT stream overhead. They
measure the same one-window forward, loss, backward, and AdamW step, then add
the state bookkeeping used by stream training: optional carry, detach, scheduled
reset, and decay. This is the benchmark to check the expectation that TBPTT
should run close to ordinary TTT training because it does not backpropagate
through prior windows.

```bash
cargo bench --bench ttt_training \
  --no-default-features --features ndarray \
  -- ttt_tbptt_training_step_ndarray --sample-size 10 --measurement-time 1 --warm-up-time 0.2

BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo bench --bench ttt_training \
  --no-default-features --features ndarray,cuda \
  -- ttt_tbptt_training_step_cuda --sample-size 10 --measurement-time 1 --warm-up-time 0.2
```

Local short Criterion TBPTT smoke from 2026-05-15, using the same tiny 64px
fixture, one 50% sparse mask row, and full forward+backward+AdamW:

| Backend | Case | Median step | Notes |
|---|---|---:|---|
| ndarray | no stream, fresh state | 10.321 ms | Non-TBPTT baseline. |
| ndarray | TBPTT reset every step, decay 1.00 | 10.470 ms | Fresh-state curriculum step with optimizer update. |
| ndarray | TBPTT carry 4, decay 1.00 | 10.475 ms | Carry+detach without decay. |
| ndarray | TBPTT carry 4, decay 0.97 | 10.511 ms | Production-style decay. |
| ndarray | TBPTT carry 4, decay 0.97, packed b4 | 49.529 ms | Live packed multi-stream sanity run; 80.8 samples/sec mean throughput. |
| ndarray | TBPTT carry 4, decay 0.90 | 10.479 ms | Stronger decay sweep point. |
| CUDA | TBPTT reset every step, decay 1.00 | 15.696 ms | WGPU/CUDA first-use noise makes the fresh no-stream row unstable in short runs. |
| CUDA | TBPTT carry 4, decay 1.00 | 14.980 ms | No statistically significant change from reset-only TBPTT. |
| CUDA | TBPTT carry 4, decay 0.97 | 15.913 ms | Production-style decay. |
| CUDA | TBPTT carry 4, decay 0.90 | 13.695 ms | No statistically significant change in this short run. |

Interpretation: on ndarray, single-stream TBPTT state carry+detach+decay adds
about 1.5--1.9% over the fresh-state training step. The packed b4 row exercises
the multi-stream state pack/unpack path and lands at about 4.7x the b1 median
for 4x the samples in this tiny CPU fixture; it is a sanity row, not a tuned CPU
throughput target. On CUDA, the short run is noisy but all stable single-stream
TBPTT rows stay in the same 13.7--15.9 ms band. This supports the intended
implementation model: long-form training cost is still dominated by the current
window's forward/backward/optimizer work, not by copying or decaying the carried
TTT memory.

Local short Criterion smoke from 2026-05-14, using a tiny 64px fixture with 32
dense tokens and `--sample-size 10 --measurement-time 1 --warm-up-time 0.2`:

| Backend | Batch | 10% sparse | 50% sparse | 100% sparse | 100% dense | 10% vs dense |
|---|---:|---:|---:|---:|---:|---:|
| ndarray | 1 | 9.633 ms | 10.211 ms | 10.830 ms | 10.836 ms | 11.1% faster |
| ndarray | 2 | 18.418 ms | 19.175 ms | 20.039 ms | 20.082 ms | 8.3% faster |
| ndarray | 4 | 35.308 ms | 36.676 ms | 37.861 ms | 38.109 ms | 7.4% faster |
| ndarray | 8 | 68.756 ms | 71.575 ms | 73.748 ms | 74.324 ms | 7.5% faster |
| WGPU | 1 | 16.157 ms | 18.652 ms | 21.820 ms | 20.996 ms | 23.0% faster |
| WGPU | 2 | 14.712 ms | 18.652 ms | 21.557 ms | 19.321 ms | 23.9% faster |
| WGPU | 4 | 15.814 ms | 18.121 ms | 20.258 ms | 19.781 ms | 20.1% faster |
| WebGPU | 1 | 16.990 ms | 20.866 ms | 23.048 ms | 21.322 ms | 20.3% faster |
| WebGPU | 2 | 13.989 ms | 18.678 ms | 23.234 ms | 22.074 ms | 36.6% faster |
| WebGPU | 4 | 15.504 ms | 18.751 ms | 22.818 ms | 22.696 ms | 31.7% faster |
| CUDA | 1 | 34.516 ms | 18.221 ms | 21.663 ms | 19.408 ms | noisy/outlier |
| CUDA | 2 | 15.068 ms | 18.425 ms | 21.974 ms | 18.798 ms | 19.8% faster |
| CUDA | 4 | 14.627 ms | 18.267 ms | 21.281 ms | 20.809 ms | 29.7% faster |

Interpretation: the sparse TTT path now shows the expected latency ordering in
the training step, especially on WGPU/WebGPU and CUDA batch 2+. The gain is not
proportional to token count because this Criterion lane still uses Burn autodiff
dense image patch embedding before sparse token gather, and the timed step
includes adapter backward plus AdamW state updates. Teacher-token precompute is
outside the timed loop. It is therefore a clean TTT forward+backward density
sweep, not a full AutoGaze/flex-gmm pixel-skip E2E replacement.

Pixel-skip sparse-patchify training-step smoke from the same fixture, using
`--sample-size 10 --measurement-time 1 --warm-up-time 0.2`:

| Backend | Batch | 10% sparse patchify | 50% sparse patchify | 100% sparse patchify | 100% dense | 10% vs dense |
|---|---:|---:|---:|---:|---:|---:|
| WGPU | 1 | 16.408 ms | 12.107 ms | 14.263 ms | 14.420 ms | noisy/outlier |
| WGPU | 2 | 8.842 ms | 11.793 ms | 13.599 ms | 14.022 ms | 36.9% faster |
| WGPU | 4 | 8.790 ms | 11.853 ms | 13.512 ms | 13.957 ms | 37.0% faster |
| CUDA | 1 | 33.360 ms | 18.519 ms | 20.076 ms | 21.215 ms | noisy/outlier |
| CUDA | 2 | 15.316 ms | 19.627 ms | 20.579 ms | 22.666 ms | 32.4% faster |
| CUDA | 4 | 15.085 ms | 18.050 ms | 21.669 ms | 19.835 ms | 23.9% faster |

Interpretation: the sparse-patchify lane closes the earlier benchmark gap. It
now measures sparse pixel patchification, sparse encoder rollout, feature loss,
backward, and AdamW in one timed sample. The remaining non-proportional scaling
is expected because transformer dispatch, adapter backward, AdamW, and fixed
kernel overhead remain in the timed loop; 100% sparse-patchify rows isolate the
wrapper/kernel overhead when no pixels are skipped.

## Experiment Harness

The `experiment` CLI runs the TTT direction as a reproducible trial matrix. It
plans trials, optionally builds frame-window manifests from local extracted
video frames, runs baseline/TTT variants, and writes JSON/CSV/Markdown analysis:

```sh
cargo run --bin burn-jepa -- print-experiment-config > experiment.toml
cargo run --bin burn-jepa -- experiment plan --config experiment.toml
cargo run --bin burn-jepa -- experiment prepare-data --config experiment.toml
cargo run --bin burn-jepa -- experiment run --config experiment.toml
cargo run --bin burn-jepa -- experiment analyze --run-dir target/burn-jepa-experiments
```

The default experiment is a synthetic tiny smoke. Set
`require_real_checkpoint = true` and `require_real_dataset = true` for open-set
experiments so a missing V-JEPA checkpoint or real train manifest fails before
training. The default matrix covers all model variants
(`teacher3d_reference`, `single_frame_no_ttt`, `ttt_teacher_final`,
`ttt_self_hidden`), all mask policies (`full_frame`, `keep_ratio`,
`random_sparse`, `patch_diff`, `autogaze_sparse`, `precomputed_masks`), four
densities (`0.01`, `0.05`, `0.10`, `0.25`), and enables the sparse predictor
mask objective with `base.loss.predictor_loss_weight = 0.25`.

`autogaze_sparse` in the synthetic training matrix uses a deterministic
center-biased sparse projection shaped like AutoGaze output. For real
AutoGaze-vs-patch-diff conclusions, set `require_real_dataset = true` and feed
per-window AutoGaze masks through manifests or run the existing AutoGaze E2E
benchmark lane alongside the training report.

Local synthetic matrix results from 2026-05-14:

| Backend | Trials | Full matrix | Loss-improved TTT trials | Cosine-improved TTT trials | Runtime |
|---|---:|---:|---:|---:|---:|
| NdArray | 96/96 | yes | 37/48 | 39/48 | 1.914 s |
| CUDA | 96/96 | yes | 34/48 | 34/48 | 420.211 s |
| WebGPU | 96/96 | yes | 30/48 | 29/48 | 10.205 s |

The CUDA matrix includes first-use kernel compilation and dispatch stalls in
the early sparse-predictor policies. After warm-up, most two-step tiny TTT
trials ran in roughly 50-80 ms on this machine; treat the full CUDA matrix time
as cold-runtime validation, not steady-state throughput.

TTT layer-placement ablations from 2026-05-14:

| Run | Checkpoint/Data | Layer Set | Layers | Variant | Trials | Free-run loss | Free-run cosine | Teacher-forced loss | Teacher-forced cosine | Train time | Samples/sec |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `real-cuda-224-corrected` | V-JEPA 2.1 ViT-B + real video windows | `encoder_first_last` | `[0, 11]` | `ttt_teacher_final` | 1 | 0.6170 | 0.7575 | 0.3791 | 0.8445 | 160.865 s | 0.050 |
| `real-cuda-224-corrected` | V-JEPA 2.1 ViT-B + real video windows | `encoder_last` | `[11]` | `ttt_teacher_final` | 1 | 0.6634 | 0.7524 | 0.3863 | 0.8418 | 13.123 s | 0.610 |
| `real-cuda-224-previous` | V-JEPA 2.1 ViT-B + real video windows | `encoder_thirds` | `[3, 7, 11]` | `ttt_teacher_final` | 1 | not rerun | not rerun | 0.3800 | 0.8442 | 176.536 s | 0.045 |
| `ndarray-depth4-confirm` | synthetic tiny depth-4 | `encoder_thirds` | `[1, 2, 3]` | `ttt_teacher_final` | 4 | 1.1586 | 0.4207 | n/a | n/a | synthetic | synthetic |
| `ndarray-depth4-confirm` | synthetic tiny depth-4 | `encoder_first_last` | `[0, 3]` | `ttt_teacher_final` | 4 | 1.3336 | 0.3332 | n/a | n/a | synthetic | synthetic |
| `ndarray-depth4-confirm` | synthetic tiny depth-4 | `encoder_last` | `[3]` | `ttt_teacher_final` | 4 | 1.4493 | 0.2753 | n/a | n/a | synthetic | synthetic |

The corrected real-checkpoint rerun shows why teacher-forced metrics must stay
separate: `first_last` has free-run loss `0.6170`, while the old teacher-forced
diagnostic is `0.3791`. `first_last` remains the best corrected free-run smoke
among completed real rows, but its backward pass is the current bottleneck. The
`thirds` corrected free-run row was intentionally not completed in the smoke
rerun because the three-adapter backward path exceeded the interactive smoke
budget; treat the previous `thirds` row as teacher-forced-only historical
context until a dedicated longer run refreshes it. The synthetic depth-4
confirmation still prefers `thirds`, which is why it remains the documented
high-capacity preset.

Predictor-side TTT ablations are intentionally omitted from production
analysis. They are not the architecture we intend to ship: the sparse temporal
student is an encoder module, and adding recurrent state to the predictor does
not make the deployed sparse encoder more efficient or more temporally aware.

Real-checkpoint CUDA mask/memory ablations from 2026-05-14 used the published
V-JEPA 2.1 ViT-B checkpoint fixture under
`/home/mosure/.cache/burn_jepa/vjepa2_1_vitb_dist_vitG_384` and 57 train / 16
eval open-set video windows:

| Run | Grid | TTT layers | Trials | TTT improved | Fast memory | Trainable adapter |
|---|---:|---:|---:|---:|---:|---:|
| `mask-memory-cuda-112-v2` | 392 tokens | 3 | 24/24 | 8/8 matched masks | 6.75 MiB | 13.53 MiB |
| `mask-memory-cuda-224` | 1568 tokens | 1 | 12/12 | 4/4 matched masks | 2.25 MiB | 4.51 MiB |

For the 112px three-layer run, matched `ttt_self_hidden` improved held-out
target-mask loss on every policy/density. At 20% context density, target-mask
loss/cosine were: random sparse `0.1851 / 0.9200`, full-frame holdout
`0.1885 / 0.9182`, AutoGaze sparse `0.1952 / 0.9150`, and patch diff
`0.1970 / 0.9142`. Full-grid eval stayed improved over the single-frame
baseline for every mask, with random sparse best in that run.

For the 224px last-layer confirmation at 20% context density, full-frame holdout
had the best target-mask loss/cosine (`0.1729 / 0.9274`). Among sparse policies,
AutoGaze and patch diff were close: AutoGaze `0.1803 / 0.9238`, patch diff
`0.1798 / 0.9235`, random sparse `0.1960 / 0.9173`. AutoGaze had near-zero mask
resolution time in this fixture, while patch diff added 86 ms to single-frame
eval mask scoring and 13 ms to TTT eval mask scoring at 224px because it reads
video data back for host-side scoring.

These runs validate that the TTT module is useful for quality in this setup.
Sparse rollout reduces the TTT student sequence and loss/eval target.
CUDA/WGPU flex-gmm sparse image patchify is wired for inference rollout and for
adapter-only TTT training through the frozen sparse patchify bridge, so masked
pixels can be skipped before patch embedding when pretrained V-JEPA weights are
frozen. Full-grid eval remains intentionally opt-in because it performs a second
dense student rollout for diagnostic parity metrics.

Production sparse-context continuation from 2026-05-14 used 91 clips, 585 train
windows, and 164 held-out eval windows. It resumed from
`target/burn-jepa-production-ttt/autogaze-sparse-224-context-1024-trainonly/ttt-model.mpk`
and trained another 1024 CUDA sparse-patchify steps at 20% context density. The
held-out sparse-context eval improved from loss/cosine `0.3167 / 0.8684` before
continuation to `0.2802 / 0.8839` after continuation. Sparse-only eval with
`eval_batch_size = 16` measured `5.62` samples/sec on the new adapter; the
32-window full-grid diagnostic measured sparse loss/cosine `0.2854 / 0.8819`
and full-grid loss/cosine `0.2358 / 0.9025`.

Real-AutoGaze cross-domain pilot from 2026-05-14 used 19 clips across `cisco`,
`nature`, and `screen`, generating 83 masked windows from the local
`nvidia/AutoGaze` checkpoint. On the 20-window held-out eval split, dense
single-frame no-TTT measured loss/cosine `0.2734 / 0.8849`. The current
512-step real-mask continuation checkpoint at
`target/burn-jepa-real-autogaze-cross-domain/real-autogaze-context-continue-512/ttt-model.mpk`
measured sparse free-run loss/cosine `0.2444 / 0.8969` at `0.95` samples/sec
with `--batch-size 4 --no-full-grid`; the slower full-grid diagnostic measured
full loss/cosine `0.2088 / 0.9129` at `0.73` samples/sec. Split training timings
from the larger continuation runs showed backward dominates optimizer time, so
the next throughput work is still reducing autodiff backward through the sparse
TTT rollout rather than swapping AdamW.

Sparse rollout smoke after this change:

- WGPU 64px dense TTT, 6 steps, batch 2: final loss `1.5396`, eval full cosine
  `0.2690`, `0.82` samples/sec, student forward `26 ms`, backward/optim
  `6701 ms`.
- WGPU 64px patch-diff sparse target rollout, 6 steps, batch 2: final loss
  `2.0063`, eval target cosine `0.0094`, eval full cosine `0.1119`,
  `1.02` samples/sec. Patch-diff mask scoring measured `51 ms` over train and
  `1 ms` over eval.
- WGPU 64px AutoGaze sparse target rollout, 6 steps, batch 2: final loss
  `2.2713`, eval target cosine `-0.1110`, eval full cosine `0.0247`,
  `0.68` samples/sec, mask scoring `0 ms`.
- CUDA 64px dense TTT, 6 steps, batch 2: final loss `1.5254`, eval full cosine
  `0.2671`; CUDA autodiff backward/optim dominated at `169656 ms`, so it is not
  a training-throughput win in this tiny setup.
- CUDA 64px patch-diff and AutoGaze sparse target rollouts were run as one-step
  smokes. Both completed with finite losses; patch diff measured `29 ms` mask
  scoring and AutoGaze measured `0 ms`.

TTT rollout criterion smoke:

- ndarray dense rollout: `4.67 ms`.
- ndarray sparse-token rollout at 50% target mask: `4.65 ms`.
- WGPU flex-gmm sparse image patchify rollout at 50% target mask: `4.21-4.80 ms`.
- CUDA dense rollout: `4.06-4.44 ms`.
- CUDA flex-gmm sparse image patchify rollout at 50% target mask: `2.88-3.27 ms`.

Runtime smoke tests:

```sh
BURN_JEPA_RUN_GPU_TRAINING_SMOKE=1 \
cargo test --test gpu_training_smoke webgpu_ttt_training_smoke_runs_when_requested -- --nocapture

BURN_JEPA_RUN_GPU_TRAINING_SMOKE=1 \
cargo test --no-default-features --features ndarray,wgpu \
  --test gpu_training_smoke wgpu_ttt_training_smoke_runs_when_requested -- --nocapture

cargo test --no-default-features --features ndarray,cuda \
  --test gpu_training_smoke cuda_training_preflight_reports_unavailable_runtime_without_initializing_cuda -- --nocapture

BURN_JEPA_RUN_GPU_TRAINING_SMOKE=1 \
cargo test --no-default-features --features ndarray,cuda \
  --test gpu_training_smoke cuda_ttt_training_smoke_runs_when_requested -- --nocapture
```

CUDA dispatch preflights Linux NVIDIA character devices before constructing a
Burn CUDA backend. Set `BURN_JEPA_TRAIN_CUDA_FORCE=1` only on a machine where
CUDA is known to be available despite the default preflight. On the 2026-05-13
local RTX PRO 6000 run, the opt-in CUDA TTT smoke completed the two-step
numerical stability check successfully.
