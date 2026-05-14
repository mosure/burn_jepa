# TTT Training Protocol

The TTT path trains an added recurrent adapter to make single-frame JEPA
rollouts approximate the pretrained 3D/tubelet V-JEPA encoder.

## Model Shape

- Teacher: the loaded V-JEPA 2.1 encoder runs the normal 3D patch/tubelet path.
- Student: the same V-JEPA encoder receives one frame at a time through the
  image patch path, with `VJepaTttLayer` adapters inserted after configured
  transformer blocks.
- Adapter state: each TTT layer keeps a `[batch, dim, dim]` fast-weight tensor.
  Tokens are processed in `ttt.chunk_tokens` chunks so updates can roll through
  a frame/window without materializing a dense temporal block.
- Initialization: adapter output projection is zero-initialized, so inserting a
  layer starts as a no-op residual path. The temporal target projection starts as
  an identity-style depthwise 1D filter plus optional identity linear projection.
- Target mode: `ttt.target = "teacher_final"` trains fast weights from detached
  final teacher tubelet features, while `ttt.target = "self_hidden"` updates
  from the current hidden states. `teacher_final` is a cross-depth distillation
  target when adapters are placed before the final block; reports label it as
  `cross_depth_final_teacher_to_all_ttt_layers`.

This is intentionally a Burn-native adapter instead of a literal mutation of an
existing dense matrix. In-Place TTT's LLM recipe updates fast weights in the MLP
down-projection and chunks long sequences. The JEPA adaptation keeps the same
compatibility constraints: preserve pretrained weights, add/update only a small
fast-weight path by default, and roll chunks through a sequence without external
memory modules.

## Code Organization

- `src/ttt/config.rs`: adapter placement, rollout, target-mode,
  backprop-mode, and freeze config.
- `src/ttt/state.rs`: per-layer fast-weight state and detach behavior.
- `src/ttt/layer.rs`: zero-init TTT adapter layer and fast-weight update.
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
2. Run the teacher video through the 3D encoder and detach the final tubelet
   tokens.
3. Roll the student over single frames, updating TTT state frame by frame.
4. Compare the collected student tubelet tokens to the teacher tubelet tokens
   with feature MSE.
5. Backpropagate through the student rollout and update the configured trainable
   modules with AdamW.

The default config freezes pretrained V-JEPA weights and updates only the added
TTT modules. Set `ttt.freeze_pretrained = false` for full finetuning.
Set `model.ttt_checkpoint_path` to resume/continue adapter training from a
saved `ttt-model.mpk` while still resolving pretrained V-JEPA weights from
`model.checkpoint_dir`.
Set `loss.predictor_loss_weight > 0` to add the normal sparse JEPA predictor
loss on top of feature distillation; the context/target masks come from
`training.mask` when configured, otherwise the legacy
`training.context_keep_ratio` field is used. The TTT training report records
`initial_loss`, `best_loss`, and `final_loss`; smoke tests assert finite losses
and a tiny synthetic convergence step.

## Evaluation Semantics

`eval_loss` and `eval_cosine` are deploy-style free-run metrics: the student
rollout does not receive teacher tokens as adapter update targets. For
`ttt.target = "teacher_final"`, reports also include
`teacher_forced_eval_loss`, `teacher_forced_eval_cosine`, and the
`teacher_forcing_*_gap` fields. Those teacher-forced fields are diagnostics for
how much privileged teacher-target adaptation helps; they are not production
student-inference metrics.

Reports include `target_supervision` to make the adapter target explicit. In
`teacher_final` mode the training target is final teacher tokens for every TTT
insertion point, which is cross-depth matching for early adapters. In
`self_hidden` mode the train and deploy update target is the detached current
hidden state.

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
with `sparse-patchify-cuda`. It runs flex-gmm sparse patchify on the backend's
inner non-autodiff tensor, then re-enters autodiff at the sparse token boundary.
This is intended for adapter-only TTT training and requires
`ttt.freeze_pretrained = true`; gradients still train the TTT/memory layers, but
the frozen patch embedding does not receive gradients. Training reports include
`rollout.autodiff_sparse_patchify` so benchmark artifacts show which path was
actually used.

Training reports include `rollout.mode`, `rollout.student_tokens`,
`rollout.student_token_density`, and `rollout.autodiff_sparse_patchify` so
experiment artifacts show whether a run actually trained dense rollout,
target-mask sparse rollout, or frozen sparse patchify rollout. Set
`training.loss_trace_interval = 0` for throughput-oriented GPU runs to avoid the
per-step scalar loss readback; the final loss is still reported, but `loss_trace`
is left empty.

Real AutoGaze masks are generated as a manifest preprocessing step:

```toml
[data.autogaze_masks]
checkpoint_dir = "/home/mosure/.cache/huggingface/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a"
backend = "cuda"
context_density = 0.2
target_density = 0.05
max_gaze_tokens_each_frame = 32
top_k_overfetch = 1.25
```

This writes `precomputed_context_indices` and `precomputed_target_indices` into
the train/eval manifests. Real masks can now batch in three modes:

```toml
[training]
batching = "sequential"          # legacy order
batching = "group_uniform_masks" # group identical context/target masks
batching = "fixed_width_masks"   # group equal-width per-sample masks
```

`fixed_width_masks` supports different mask indices per sample when each row has
the same context/target token count. The encoder and loss gather from batched
`[batch, tokens]` index tensors, and the CUDA sparse-patchify bridge can consume
fixed-width per-sample coordinate plans. Ragged per-sample masks remain the
future fallback for variable token counts inside the same batch; until then,
bucket data by token count or keep `batch_size = 1` for fully ragged masks.

## Block Rollout

`ttt.rollout_blocks` controls truncated rollout training. A value of `1`
detaches state after every produced tubelet block; higher values keep gradients
through more temporal blocks before detaching. This keeps long clips trainable
without forcing the entire stream history into one autodiff graph.

`ttt.backprop_mode` makes the backward/runtime tradeoff explicit:

- `final_feature`: default full final-feature distillation objective.
- `truncated_final`: same objective, but uses
  `ttt.backprop_truncate_blocks` for the rollout detach cadence.
- `layer_local`: experimental early-exit objective that stops after the last
  configured TTT layer. This intentionally shortens the frozen tail in the
  backward graph and should be compared against `final_feature` for quality.

Set `training.cache_teacher_tokens = true` to cache detached teacher features
inside a run. Reports include `teacher_cache_hits` and
`teacher_cache_misses` in train/eval stage metrics.

## TTT Layer Placement

`ttt.layer_placement` controls where recurrent TTT adapters are inserted in the
JEPA encoder. Supported placements are `first`, `middle`, `last`,
`first_last`, `thirds`, and `explicit`. `explicit` uses `ttt.layers` directly.
The default is `first_last`, which resolves to `[0, 11]` for a 12-layer ViT-B
encoder. It was selected as the smoke/training default because the real V-JEPA
2.1 CUDA ablation below matched the best held-out sparse loss while avoiding
the much larger backward cost of the three-adapter preset. Use `thirds` for the
higher-capacity `[3, 7, 11]` ViT-B preset when longer quality-focused runs can
afford the extra backward time.

`ttt.predictor_layers` is reserved in the config schema, but predictor-layer TTT
adapters are intentionally rejected today. The recurrent TTT state is trained to
make the single-frame encoder behave like the temporal 3D encoder; the JEPA
predictor is still trained through the normal sparse predictor loss.

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

Set `training.backend` to `nd_array`, `wgpu`, `web_gpu`, or `cuda` and enable
the matching Cargo feature for GPU training/bench dispatch.
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
single-frame no-TTT measured loss/cosine `0.2734 / 0.8849`; the prior TTT
adapter measured `0.2657 / 0.8885`; and a 512-step real-mask continuation
measured `0.2531 / 0.8938`. Split training timings showed `backward_ms = 657s`
and `optimizer_ms = 19.5s`, so the bottleneck is autodiff backward through the
sparse TTT rollout rather than AdamW optimizer updates.

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
