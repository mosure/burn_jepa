# Sparse TTT Production Status

This note records the current production candidate gate for sparse temporal
V-JEPA 2.1 adapters. It is intentionally narrower than the training protocol
doc: it names the checkpoint, eval command, measured behavior, and remaining
external parity/data requirements.

## Candidate

- Base V-JEPA fixture:
  `/home/mosure/.cache/burn_jepa/vjepa2_1_vitb_dist_vitG_384/model.pt`
- TTT adapter:
  `target/burn-jepa-production-final/stage1-stream-tbptt/ttt-model.mpk`
- Sparse policy: AutoGaze-style sparse context masks, 314 / 1568 context
  tokens, 79 / 1568 target tokens.
- Eval split: 164 held-out open-set windows from
  `target/burn-jepa-production-final/data/eval-real-autogaze.jsonl`.

## Loader Gate

The official Meta `.pt` fixture uses top-level `ema_encoder` and `predictor`
modules with nested `module.backbone.*` parameter names. The Burn loader maps
those prefixes directly, loads the official V-JEPA 2.1 encoder/predictor
modality embeddings, keeps predictor mask tokens zero-initialized, and keeps
strict missing checks for everything else.

```sh
BURN_JEPA_VJEPA21_CHECKPOINT_DIR=/home/mosure/.cache/burn_jepa/vjepa2_1_vitb_dist_vitG_384 \
BURN_JEPA_VJEPA21_WEIGHTS=model.pt \
BURN_JEPA_VJEPA21_FORWARD_PARITY=1 \
cargo test --no-default-features --features ndarray \
  --test numerical_parity real_vjepa_checkpoint_loads_when_fixture_is_set \
  -- --ignored --nocapture
```

Current result:

- `applied=312 missing=0 skipped=0 errors=0`
- Official torch.hub reference:
  `torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_base_384", pretrained=False, num_frames=16)`
- Micro sparse encoder/predictor parity: context max abs diff `1.109e-5`,
  prediction max abs diff `1.812e-5`, target max abs diff `1.034e-5`.
- Multi-tubelet 3x4-grid sparse encoder/predictor parity: context max abs diff
  `1.335e-5`, prediction max abs diff `1.597e-5`, target max abs diff
  `2.921e-5`.

## Held-Out Eval

Sparse production rollout:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config configs/production/vjepa21-ttt-final-eval-cuda.toml \
  --model target/burn-jepa-production-final/stage1-stream-tbptt/ttt-model.mpk \
  --steps 11 --batch-size 1 --no-full-grid
```

Result:

- Sparse free-run loss/cosine: `0.3790 / 0.8402` over 11 held-out windows.
- Throughput: `0.079` samples/sec with utilization and temporal diagnostics
  enabled.
- Stage time: teacher `4494 ms`, student `27527 ms`, loss `42 ms`.
- The diagnostic probes show the TTT memory is active: freezing fast updates
  worsens loss to `0.4688`, resetting every frame worsens loss to `0.4688`,
  and reversing frame order worsens loss to `0.5359`.
- Keep this diagnostic at `eval_batch_size = 1`; larger batches with
  utilization probes have previously hit cubecl CUDA allocator pressure.

Streamed 16-frame eval:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config configs/production/vjepa21-ttt-stream-eval-fast-cuda.toml \
  --model target/burn-jepa-production-final/stage1-stream-tbptt/ttt-model.mpk \
  --steps 16 --batch-size 2 --no-full-grid
```

Result:

- Sparse streamed free-run loss/cosine: `0.2825 / 0.8794` over 32 held-out
  windows.
- Throughput: `0.805` samples/sec.
- Domain slices: `cisco` loss/cosine `0.2685 / 0.8850` over 26 windows;
  `mixed` loss/cosine `0.3429 / 0.8552` over 6 windows.
- Stream state: `13` carried windows, `19` reset windows, `32`
  detached/decayed windows, reset interval `4`.
- This is the preferred deployment-style eval. The diagnostic config disables
  temporal probes so it measures the free-run sparse path without repeated
  ablation rollouts.

Training behavior:

- 1024 CUDA steps, 2048 samples, 20.0% sparse-context density.
- The current production config uses `training.batching = "packed_streams"`
  with `batch_size = 2`, final-teacher supervision, and LR `2.5e-6`.
- Sequence-length curriculum: reset every window initially, ramping to 4
  windows per stream segment by step 512.
- Carried TTT state on `1217 / 2048` windows; all windows detached across the
  TBPTT boundary and decayed by `0.97`.
- Reset/fresh-state windows are not no-grad warmups: every stream window gets a
  normal optimizer step, so the same run trains both zero-state initialization
  and carried long-form stability.
- Optimizer-step mix: `276` reset, `469` carried, `279` mixed.
- Initial loss `0.4076`, best loss `0.2126`, final loss `0.2589`.
- Throughput `0.871` samples/sec.
- Runtime bottleneck remains backward/optimizer: `2082.7 s` of `2350.1 s`
  train elapsed. Teacher cache behavior was `672` misses then `352` hits.
- A batch-4 hybrid/layer-local packed launch was stopped at step 512 because
  loss rose from `1.9740` to `5.8752`; the production config was made more
  conservative after that failure.

Long-rollout checks:

| Eval | Loss | Cosine | Late-early loss | Prior late-early loss |
|---|---:|---:|---:|---:|
| 32-frame real stratified AutoGaze | 0.3603 | 0.8493 | -0.0342 | +0.2500 |
| 64-frame real AutoGaze | 0.3846 | 0.8439 | +0.0474 | +3.8152 |
| 64-frame balanced temporal sparse | 0.3841 | 0.8433 | -0.0201 | +3.7623 |

## Production Verdict

The direction is viable: exact official V-JEPA 2.1 torch.hub parity, sparse
rollout, sparse patchify, adapter training, checkpoint reload, and CUDA/WebGPU
smokes all work together. The fresh official-2.1 adapter gives a clear
long-horizon stability gain over the previous 16-frame-trained adapter while
preserving the deployable free-run sparse path.

It is a stronger production candidate, but still not a final production model.
The remaining gates are scale and speed: train on a larger, more diverse real
AutoGaze-mask corpus, run broader cross-domain eval, and reduce sparse TTT
backward/optimizer cost.

The next launch path is tracked in `docs/production-training-runbook.md`. Use
the stage-1 frozen adapter run as the production candidate path. The stage-2
low-LR unfrozen config is a gated ablation after plateau, not the default first
run.
