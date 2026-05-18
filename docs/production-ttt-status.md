# Sparse TTT Production Status

This note records the current production candidate gate for sparse temporal
V-JEPA 2.1 adapters. It is intentionally narrower than the training protocol
doc: it names the checkpoint, eval command, measured behavior, and remaining
external parity/data requirements.

## Candidate

- Base V-JEPA fixture:
  `/home/mosure/.cache/burn_jepa/vjepa2_1_vitb_dist_vitG_384/model.pt`
- TTT adapter:
  `target/burn-jepa-production-final-256/stage1-stream-tbptt-longstable/ttt-model.mpk`
- Sparse policy: AutoGaze-style sparse context masks, 410 / 2048 context
  tokens, 103 / 2048 target tokens at 256px / 16 frames.
- Eval split: 164 held-out open-set windows from
  `target/burn-jepa-production-final-256/data/eval-real-autogaze.jsonl`.

## 2026-05-18 Long-Rollout Stability Gate

The previous full-manifest eval was not enough to justify an
arbitrary-length claim because its longest contiguous stream was only 33
windows.  The new stress configs add explicit dataset repeat/stitch modes:

- `repeat_mode = "continuous_streams"` repeats each stream with monotonic
  `start_frame`s so carried state can run past the original manifest horizon.
- `repeat_mode = "stitched_stream"` forces multiple clips into one logical
  stream, exposing scene-switch contamination.
- `repeat_mode = "adversarial_stitched_stream"` alternates the manifest order
  while keeping one logical stream.
- `sample_limit` can isolate one stream for same-scene repeat tests.

Unbounded carried state failed the new gate.  On an 8x repeated cactus stream
with 136 consecutive windows and no scene switches, the SIGReg checkpoint
drifted from first-quarter loss/cosine `0.3786 / 0.8489` to final-quarter
`0.5558 / 0.7831`; late-minus-early loss was `+0.1772`.  Forced scene
stitching also failed without recovery: 64 stitched windows degraded from
`0.3274 / 0.8669` to `0.5399 / 0.7876`.

The production policy is therefore bounded-horizon state, not unbounded memory:
reset/refresh every 16 windows and reset immediately when the scene identity
changes.  A continuation trained from the SIGReg checkpoint with a reset
curriculum ending at 16 windows produced the current candidate:

```sh
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- train-ttt \
  --config configs/production/vjepa21-ttt-stage1-stream-tbptt-longstable-cuda.toml
```

Result:

- Saved model:
  `target/burn-jepa-production-final-256/stage1-stream-tbptt-longstable/ttt-model.mpk`.
- 128 optimizer steps, 256 samples, final reset horizon 16 windows.
- Loss trace: initial `0.1950`, best `0.1653`, final `0.2479`.
- Runtime: `303.4 s`, `0.845 samples/s`; backward+optimizer remained dominant
  at `241.7 s`.

Stress evals with the longstable checkpoint:

| Gate | Windows | Resets | Loss | Cosine | Late-early loss |
|---|---:|---:|---:|---:|---:|
| Same-stream cactus repeat, reset16 | 136 | 9 | 0.3270 | 0.8666 | +0.0049 |
| Scene-stitched stream, scene reset | 64 | 11 | 0.2896 | 0.8783 | -0.0675 |
| Adversarial stitched stream, scene reset | 64 | 64 | 0.4209 | 0.8215 | -0.0338 |

This resolves the old paper caveat in a narrower, production-relevant sense:
arbitrary-duration streams are stable when the runtime uses the trained
bounded-horizon refresh policy.  It does not show that a single fast state can
be carried forever without reset; that mode is explicitly measured as a
failure case.

## 2026-05-18 LeJEPA-Style Stability Regularization Update

LeJEPA-style Gaussian latent regularization is now implemented as an opt-in
`[loss.latent_regularization]` block.  The term penalizes student-token mean,
variance, and an adjacent-feature covariance sketch.  It is intentionally small
for V-JEPA 2.1 distillation because the student must still match the fixed
teacher representation.

Matched 64-step CUDA ablation:

```sh
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- train-ttt \
  --config configs/production/vjepa21-ttt-stage1-stream-tbptt-sigreg-ablation-low-cuda.toml
```

| Run | Reg weight | 64-step feature eval | Cosine | Regularizer loss | Train seconds |
|---|---:|---:|---:|---:|---:|
| off | 0 | 0.3478696 | 0.8516861 | n/a | 218.3 |
| low | 1e-5 | 0.3478668 | 0.8516874 | 5.82e-6 | 222.5 |
| high | 5e-5 | 0.3478670 | 0.8516872 | 2.91e-5 | 220.7 |

The low weight was selected because it was non-regressive and slightly best on
the short held-out subset.  The result is effectively neutral, not a material
quality gain.

Selected 160-step run:

```sh
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- train-ttt \
  --config configs/production/vjepa21-ttt-stage1-stream-tbptt-sigreg-cuda.toml
```

Result:

- Saved model:
  `target/burn-jepa-production-final-256/stage1-stream-tbptt-sigreg/ttt-model.mpk`.
- 160 optimizer steps, 320 samples, 20.0% sparse context density.
- Total loss trace: initial `0.24117`, best `0.18285`, final `0.31107`.
- Runtime remained backward dominated: `408.9 s` backward plus `7.2 s`
  optimizer out of `494.2 s` train elapsed.
- Regularizer loss-stage overhead stayed small: `75 ms` total loss time over
  160 steps.

Comparable full sequential long-rollout eval:

```sh
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config configs/production/vjepa21-ttt-long-rollout-sigreg-cuda.toml \
  --model target/burn-jepa-production-final-256/stage1-stream-tbptt-sigreg/ttt-model.mpk \
  --steps 164 --batch-size 1 --no-full-grid
```

Result:

- SIGReg SC-TTT sparse persistent state: feature loss/cosine
  `0.3208972 / 0.8674483`.
- Previous selected SC-TTT sparse persistent state:
  `0.3208981 / 0.8674482`.
- Base sparse V-JEPA 2.1 remains `0.4273 / 0.8187`.
- Late-minus-early drift remains essentially unchanged:
  `+0.08379` loss and `-0.03002` cosine.
- Eval took `1394.6 s`; student rollout dominated with `1365.6 s`.

The regularizer is therefore a safe stability hook and reporting surface, but
the present data does not show a meaningful stability breakthrough over the
state decay/update penalties already in the selected recipe.

## 2026-05-17 Verification Update

The previous `stage1-stream-tbptt` output directory did not contain a saved
TTT checkpoint. A bounded CUDA verification run trained a fresh checkpoint with
packed-stream TBPTT, sparse context rollout, dense stabilization samples, state
decay, and state/update regularization:

```sh
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- train-ttt \
  --config configs/production/vjepa21-ttt-stage1-stream-tbptt-verified-cuda.toml
```

Training result:

- Saved model:
  `target/burn-jepa-production-final-256/stage1-stream-tbptt-verified/ttt-model.mpk`.
- 160 optimizer steps, 320 samples, 20.0% sparse context density.
- Loss trace was noisy: initial `0.2412`, best `0.1828`, final `0.3111`.
- Stream state was exercised: `211` carried windows, `109` reset windows,
  reset interval ramped to `4`, and all states were detached/decayed.
- Runtime remained backward dominated: `407.8 s` backward plus `6.9 s`
  optimizer out of `489.7 s` train elapsed.

Sequential carried long-rollout sparse eval is the primary coherence gate:

```sh
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config configs/production/vjepa21-ttt-long-rollout-eval-cuda.toml \
  --model target/burn-jepa-production-final-256/stage1-stream-tbptt-verified/ttt-model.mpk \
  --steps 24
```

Historical result for the previous selected checkpoint:

- Sparse free-run loss/cosine: `0.3334 / 0.8629` over 24 sequential windows.
- Stream state was actually carried: `21` carried windows, `3` reset windows.
- Persistent TTT state beat reset/frozen updates:
  reset-each-frame `0.4639 / 0.8047`, freeze-fast-update `0.4639 / 0.8047`.
- Temporal order mattered: reverse-order loss/cosine `0.5894 / 0.7520`.
- Late segment did not show collapse: segment 0 loss/cosine
  `0.3229 / 0.8673`; segment 1 `0.3551 / 0.8536`.

Full-manifest longitudinal eval is stricter and is the current evidence gate
for the direct sparse ablation:

```sh
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config configs/production/vjepa21-base-sparse-long-rollout-verylong-cuda.toml \
  --base-sparse \
  --steps 164

cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config configs/production/vjepa21-ttt-long-rollout-verylong-cuda.toml \
  --model target/burn-jepa-production-final-256/stage1-stream-tbptt-verified/ttt-model.mpk \
  --steps 164

cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config configs/production/vjepa21-ttt-long-rollout-reset-window-cuda.toml \
  --model target/burn-jepa-production-final-256/stage1-stream-tbptt-verified/ttt-model.mpk \
  --steps 164
```

Result:

- Base sparse V-JEPA 2.1: `0.4273 / 0.8187` loss/cosine over 164 windows.
- SC-TTT sparse persistent state: `0.3209 / 0.8674`.
- Reset-each-window baseline: `0.3990 / 0.8309`.
- SC-TTT improved over base sparse by `0.1064` loss and `0.0487` cosine.
- SC-TTT improved over reset-each-window by `0.0781` loss and `0.0366` cosine.
- The held-out manifest has 19 streams; the longest contiguous stream is only
  33 windows, so this is not a very-long single-stream proof.
- The final 41-window segment degraded relative to the first:
  `+0.0838` loss and `-0.0300` cosine. This keeps arbitrary-length rollout
  stability as an open production gate.
- The eval harness is slow because it recomputes the V-JEPA teacher for every
  window and the student rollout dominates wall time: persistent eval took
  `1246.3 s` at `0.132 samples/s` with `1219.4 s` in student forward. The
  base sparse run took `1977.8 s`; treat these as eval-harness timings, not
  deploy throughput numbers.

Dense full-token carried eval also completed without feature collapse when the
expensive temporal probes were disabled:

```sh
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config configs/production/vjepa21-ttt-long-rollout-dense-eval-cuda.toml \
  --model target/burn-jepa-production-final-256/stage1-stream-tbptt-verified/ttt-model.mpk \
  --steps 16
```

Result: dense free-run loss/cosine `0.3946 / 0.8448`, with `15` carried
windows and `1` reset window.

A low-LR continuation was tested and rejected as the selected model. It improved
the sampled training tail (`0.3083 -> 0.2555`) but worsened carried sparse eval
to `0.3977 / 0.8410`.  The current selected checkpoint is the later
non-regressive SIGReg-stabilized run:
`stage1-stream-tbptt-sigreg/ttt-model.mpk`.

Dense full-grid temporal diagnostics currently hit a CubeCL CUDA allocation
panic while materializing diagnostic cosine tensors. Dense primary eval is
available; dense temporal ablation diagnostics should be made streaming or
chunked before using them as a production gate.

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
  --model target/burn-jepa-production-final-256/stage1-stream-tbptt/ttt-model.mpk \
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
  --model target/burn-jepa-production-final-256/stage1-stream-tbptt/ttt-model.mpk \
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
