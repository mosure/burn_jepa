# Production Sparse TTT Training Runbook

This runbook is the launch path for the next production-grade V-JEPA 2.1
sparse temporal student. It keeps TTT modules encoder-only.

## Current Answer

It is worth running a longer job, but only after switching from the center-prior
AutoGaze proxy to real per-window AutoGaze masks and using a staged training
gate. The existing 1024-step official V-JEPA 2.1 run already shows the
direction is viable: sparse free-run loss improved from `0.4544` to `0.3021`
and cosine from `0.8076` to `0.8746`. The unresolved question is
generalization and scale, not basic wiring.

Do not start with unfrozen V-JEPA weights. The safe order is:

1. Generate real AutoGaze masks into manifests.
2. Train encoder TTT adapters with pretrained V-JEPA frozen and sparse
   patchification enabled.
3. Evaluate free-run sparse quality, temporal diagnostics, and cross-domain
   slices.
4. Only if stage 1 plateaus cleanly, continue with a short, low-LR unfrozen
   stage as an ablation.

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
- `configs/production/vjepa21-ttt-stage1-stream-tbptt-cuda.toml` is the main
  production run: frozen pretrained V-JEPA, encoder TTT layers `[3, 7, 11]`,
  sparse context rollout, frozen sparse patchify, real manifest masks,
  packed-stream TBPTT batches, state decay, final-teacher supervision, and
  warmup/cosine LR. It uses batch 2 and a lower LR than the single-window
  baseline because the batch-4 hybrid/layer-local launch diverged at the first
  partial checkpoint.
- `configs/production/vjepa21-ttt-stage1-adapter-cuda.toml` is retained as a
  single-window adapter baseline. Use it for quick comparisons, not as the
  long-rollout production candidate.
- `configs/production/vjepa21-ttt-stage2-unfrozen-low-lr-cuda.toml` continues
  from the stage-1 checkpoint with `freeze_pretrained = false` and a much lower
  scheduled LR. Treat this as a gated ablation, not the default.
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
  --config configs/production/vjepa21-ttt-stage1-stream-tbptt-cuda.toml
```

Run a launch smoke without the expensive pre/post eval pass:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- bench-ttt \
  --config configs/production/vjepa21-ttt-stage1-stream-tbptt-cuda.toml \
  --steps 1 --eval-steps 0
```

Evaluate the stage-1 checkpoint:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config configs/production/vjepa21-ttt-stream-eval-fast-cuda.toml \
  --model target/burn-jepa-production-final/stage1-stream-tbptt/ttt-model.mpk \
  --steps 16 --batch-size 2 --no-full-grid
```

Run the slower diagnostic eval after the fast streamed gate improves:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config configs/production/vjepa21-ttt-final-eval-cuda.toml \
  --model target/burn-jepa-production-final/stage1-stream-tbptt/ttt-model.mpk \
  --steps 11 --batch-size 1 --no-full-grid
```

Optional low-LR unfrozen continuation:

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
  --model target/burn-jepa-production-final/stage2-unfrozen-low-lr/ttt-model.mpk \
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
