# Production Sparse TTT Training Runbook

This runbook is the launch path for the next production-grade V-JEPA 2.1
sparse temporal student. It keeps TTT modules encoder-only.

## Current Answer

The current production candidate is the stage2 norm-only continuation:
`target/burn-jepa-production-final-256/stage2-norms-low-lr/ttt-model.mpk`.
It uses real per-window AutoGaze masks, encoder-only TTT adapters, frozen
sparse patchify, Memory-ALiBi carried state, and a tiny encoder LayerNorm
finetune. It beat the previous stage1 Memory-ALiBi checkpoint on both
same-stream and adversarial long-rollout gates.

Do not start with unfrozen V-JEPA weights. The safe order is:

1. Generate real AutoGaze masks into manifests.
2. Train encoder TTT adapters with pretrained V-JEPA frozen and sparse
   patchification enabled.
3. Evaluate free-run sparse quality, temporal diagnostics, and cross-domain
   slices.
4. Continue with norm-only LayerNorm finetuning when stage 1 plateaus.
5. Treat last-block or full unfrozen training as upper-bound ablations only.

The unfrozen stage is useful to test whether a small amount of encoder
finetuning removes residual teacher-student mismatch. It is also the highest
risk stage: it gives up the frozen sparse-patchify training bridge, uses dense
autodiff patch embedding during training, and can degrade pretrained feature
geometry if the learning rate is too high.

The hybrid layer-local pretrain path remains useful as an efficiency ablation,
but the production candidate currently favors full final-teacher passthrough.
That is slower, but it directly optimizes the deployable sparse student against
the temporal V-JEPA teacher and avoided the batch-4 layer-local instability seen
in the latest packed launch smoke.

## Configs

- `configs/production/vjepa21-autogaze-mask-data.toml` prepares real AutoGaze
  context/target masks and writes manifest rows with
  `precomputed_context_indices` and `precomputed_target_indices`. It derives
  manifest `domain` labels from clip ID prefixes so eval reports can slice by
  source when the corpus is laid out as `frames/<domain>_<clip>`. The production
  config uses `data.autogaze_masks.streaming = true`, so mask preparation keeps
  AutoGaze state per clip/source stream and resets at train/eval split
  boundaries or non-monotonic `start_frame`s.
- `configs/production/vjepa21-ttt-stage1-stream-tbptt-carry-forever-alibi-cuda.toml`
  is the previous carry-forever base candidate. It continues from the reset16
  `longstable` checkpoint, disables hard stream resets and stream-level decay,
  and uses three Memory-ALiBi fast-weight banks with half-lives
  `[8, 64, 512]`.
- `configs/production/vjepa21-ttt-stage1-stream-tbptt-inplace-mlp-thirds-cuda.toml`
  and
  `configs/production/vjepa21-ttt-stage1-stream-tbptt-inplace-mlp-every-other-cuda.toml`
  are In-Place TTT ablation configs. They reuse selected encoder MLP
  down-projections as the TTT fast weight. They are not the production default:
  the first CUDA smoke showed correct zero-init behavior but worse backward
  cost than adapter TTT at matched thirds layers.
- `configs/production/vjepa21-ttt-stage1-stream-tbptt-longstable-cuda.toml` is
  the bounded-horizon fallback. It continues from the SIGReg checkpoint with a
  reset curriculum ending at 16 windows, state decay, and runtime reset/refresh
  gates.
- `configs/production/vjepa21-ttt-stage1-stream-tbptt-cuda.toml` and
  `configs/production/vjepa21-ttt-stage1-stream-tbptt-verified-cuda.toml` are
  retained as unregularized baselines.
- `configs/production/vjepa21-ttt-long-rollout-cactus-repeat-reset16-cuda.toml`
  is the same-scene repeat gate. It uses `sample_limit = 17`,
  `repeat_count = 8`, and `repeat_mode = "continuous_streams"` to force a
  single source past the previous 33-window evidence horizon.
- `configs/production/vjepa21-ttt-long-rollout-stitched-reset-4x-cuda.toml`
  is the scene-switch recovery gate. It forces multiple clips into one logical
  stream and relies on `training.stream.reset_on_scene_change = true`.
- `configs/production/vjepa21-ttt-long-rollout-adversarial-stitch-reset-4x-cuda.toml`
  is the adversarial-order recovery gate. It should be stable, but it is not a
  quality win because every window is a scene switch and the state resets each
  time.
- `configs/production/vjepa21-ttt-long-rollout-sigreg-cuda.toml` is retained as
  the comparable full-manifest sequential long-rollout eval for the earlier
  SIGReg checkpoint. Do not use the packed training config for long-rollout
  claims; packed stream order shortens contiguous carried segments.
- `configs/production/vjepa21-ttt-stage1-adapter-cuda.toml` is retained as a
  single-window adapter baseline. Use it for quick comparisons, not as the
  long-rollout production candidate.
- `configs/production/vjepa21-ttt-stage2-norms-low-lr-cuda.toml` continues from
  the carry-forever checkpoint with only encoder LayerNorms trainable in
  addition to the TTT modules. This is the current promoted production
  candidate because it preserves frozen sparse patchify and improves
  long-rollout quality.
- `configs/production/vjepa21-ttt-stage2-last2-low-lr-cuda.toml` trains the last
  two encoder blocks plus TTT modules. It keeps patch embedding frozen but is
  much heavier than norm-only.
- `configs/production/vjepa21-image2video-stage2-norms-low-lr-cuda.toml` is the
  matched no-TTT control for the norm-only continuation. It trains only encoder
  LayerNorms against the same video-teacher feature target with
  `ttt.layers = []`.
- `configs/production/vjepa21-image2video-stage2-last2-low-lr-cuda.toml` is the
  matched no-TTT control for late-block static image-to-video finetuning. Use
  it to distinguish recurrent TTT gains from static image-path capacity.
- `configs/production/vjepa21-ttt-stage2-unfrozen-low-lr-cuda.toml` continues
  from the promoted stage2-norms checkpoint with `freeze_pretrained = false`
  and a much lower scheduled LR. Treat this as an upper-bound ablation, not the
  default.
- `configs/production/vjepa21-ttt-final-eval-cuda.toml` evaluates a saved
  stage-1 or stage-2 model with utilization and temporal diagnostics.
- `configs/production/vjepa21-ttt-stream-eval-fast-cuda.toml` is the lean
  deployment-style streamed eval gate. Use it for routine checkpoint
  comparisons before enabling expensive diagnostics.

## Commands

Generate real AutoGaze masks:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,autogaze-cuda,sparse-patchify-cuda \
  --bin burn-jepa -- experiment prepare-data \
  --config configs/production/vjepa21-autogaze-mask-data.toml
```

Train the encoder-only sparse TTT adapter:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- train-ttt \
  --config configs/production/vjepa21-ttt-stage1-stream-tbptt-carry-forever-alibi-cuda.toml
```

Run a launch smoke without the expensive pre/post eval pass:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- bench-ttt \
  --config configs/production/vjepa21-ttt-stage1-stream-tbptt-sigreg-cuda.toml \
  --steps 1 --eval-steps 0
```

Evaluate the promoted stage2-norms checkpoint:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config configs/production/vjepa21-ttt-long-rollout-carry-forever-alibi-cactus-64x-cuda.toml \
  --model target/burn-jepa-production-final-256/stage2-norms-low-lr/ttt-model.mpk \
  --steps 1088 --batch-size 1 --no-full-grid
```

Evaluate the saved best sampled deploy-rollout checkpoint as a diagnostic
  candidate, but do not promote it unless the long-rollout gates beat the final
  checkpoint:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config configs/production/vjepa21-ttt-long-rollout-carry-forever-alibi-cactus-64x-cuda.toml \
  --model target/burn-jepa-production-final-256/stage2-norms-low-lr/ttt-model-best.mpk \
  --steps 1088 --batch-size 1 --no-full-grid
```

Evaluate adversarial no-reset scene-switch recovery:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config configs/production/vjepa21-ttt-long-rollout-carry-forever-alibi-adversarial-8x-cuda.toml \
  --model target/burn-jepa-production-final-256/stage2-norms-low-lr/ttt-model.mpk \
  --steps 512 --batch-size 1 --no-full-grid
```

Evaluate the bounded-horizon reset16 fallback:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config configs/production/vjepa21-ttt-long-rollout-cactus-repeat-reset16-cuda.toml \
  --model target/burn-jepa-production-final-256/stage1-stream-tbptt-longstable/ttt-model.mpk \
  --steps 136 --batch-size 1 --no-full-grid
```

Run the slower diagnostic eval after the fast streamed gate improves:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config configs/production/vjepa21-ttt-final-eval-cuda.toml \
  --model target/burn-jepa-production-final-256/stage1-stream-tbptt/ttt-model.mpk \
  --steps 11 --batch-size 1 --no-full-grid
```

Norm-only low-LR continuation, current production path:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- train-ttt \
  --config configs/production/vjepa21-ttt-stage2-norms-low-lr-cuda.toml
```

Optional last-block low-LR continuation:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- train-ttt \
  --config configs/production/vjepa21-ttt-stage2-last2-low-lr-cuda.toml
```

Optional full low-LR unfrozen upper-bound continuation from the promoted
stage2-norms checkpoint:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- train-ttt \
  --config configs/production/vjepa21-ttt-stage2-unfrozen-low-lr-cuda.toml
```

Evaluate the unfrozen continuation on the sparse CUDA eval path:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config configs/production/vjepa21-ttt-final-eval-cuda.toml \
  --model target/burn-jepa-production-final-256/stage2-unfrozen-low-lr/ttt-model.mpk \
  --steps 11 --batch-size 16 --no-full-grid
```

## Gates

Stage 1 is ready to continue while all of these remain true:

- Free-run sparse eval loss improves against zero-init and the prior best
  production checkpoint.
- Cosine improves or remains flat while loss improves.
- `reset_each_frame`, `reverse_order`, `shuffle_order`, and
  `freeze_fast_update` diagnostics show the adapter is using temporal memory.
- Per-layer `adapter_delta_to_hidden` is nonzero but not exploding.
- Train throughput is stable and no periodic checkpoint regresses sharply.
- A saved best sampled checkpoint is evaluated against the final checkpoint on
  the long-rollout gates; promotion is based on long-rollout eval, not mixed
  training loss alone.
- Feature-stability diagnostics stay non-collapsed: token spread should remain
  materially nonzero and late rollout segments must not converge to one feature
  value.

Start the low-LR unfrozen stage only after stage 1 plateaus. A practical
plateau criterion is no best-loss improvement over the last 10-20% of the run
and no eval improvement after a saved checkpoint. Stop the unfrozen stage if
free-run sparse eval worsens by more than 2%, cosine drops by more than 0.01,
or temporal diagnostics indicate the adapter becomes a static residual path.

## Remaining Work

- Scale data beyond the current local frame corpus and keep domain labels in
  the manifest so eval can report domain slices.
- Use real AutoGaze masks for the primary run; center-prior masks are only a
  proxy.
- Refresh exact upstream parity for any V-JEPA 2.1 large checkpoint before
  claiming large-model support.
- Reduce backward/optimizer cost. The current final-run configs are ready, but
  training throughput is still the bottleneck.
