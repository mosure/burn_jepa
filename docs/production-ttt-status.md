# Sparse TTT Production Status

This note records the current production candidate gate for sparse temporal
V-JEPA 2.1 adapters. It is intentionally narrower than the training protocol
doc: it names the checkpoint, eval command, measured behavior, and remaining
external parity/data requirements.

## Candidate

- Base V-JEPA fixture:
  `/home/mosure/.cache/burn_jepa/vjepa2_1_vitb_dist_vitG_384/model.pt`
- TTT adapter:
  `target/burn-jepa-real-autogaze-cross-domain/real-autogaze-context-continue-512/ttt-model.mpk`
- Sparse policy: real AutoGaze precomputed context masks, 314 / 1568 context
  tokens, 79 / 1568 target tokens.
- Eval split: 20 held-out windows across `cisco`, `mixed`, and `screen`.

## Loader Gate

The official Meta `.pt` fixture uses flattened upstream names such as
`encoder.module.backbone.blocks.*` and `predictor.module.backbone.*`. The Burn
loader maps those prefixes directly, filters only the known rank-incompatible
singleton modality / predictor mask-token parameters, and keeps strict missing
checks for everything else.

```sh
BURN_JEPA_VJEPA21_CHECKPOINT_DIR=/home/mosure/.cache/burn_jepa/vjepa2_1_vitb_dist_vitG_384 \
BURN_JEPA_VJEPA21_WEIGHTS=model.pt \
BURN_JEPA_VJEPA21_FORWARD_PARITY=1 \
cargo test --no-default-features --features ndarray \
  --test numerical_parity real_vjepa_checkpoint_loads_when_fixture_is_set \
  -- --ignored --nocapture
```

Current result:

- `applied=308 missing=0 skipped=0 errors=0`
- HF micro parity is skipped for this fixture because it is not saved as
  `model.safetensors` or `pytorch_model.bin`.
- The fallback Burn real-weight micro forward smoke verifies finite predictor
  and target outputs.

Exact upstream numerical parity still requires either the Meta reference
inference path or an HF-compatible real checkpoint fixture.

## Held-Out Eval

Sparse production rollout:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config target/burn-jepa-real-autogaze-cross-domain/configs/eval-real-autogaze-new.toml \
  --model target/burn-jepa-real-autogaze-cross-domain/real-autogaze-context-continue-512/ttt-model.mpk \
  --steps 5 --batch-size 4 --no-full-grid
```

Result:

- Sparse free-run loss/cosine: `0.2444 / 0.8969`
- Throughput: `0.95` samples/sec
- Stage time: teacher `4273 ms`, student `12627 ms`, loss `409 ms`
- Domain split:
  - `cisco`: `0.2594 / 0.8910`
  - `mixed`: `0.2101 / 0.9112`
  - `screen`: `0.2465 / 0.8956`

Full-grid diagnostic:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config target/burn-jepa-real-autogaze-cross-domain/configs/eval-real-autogaze-new-full10.toml \
  --model target/burn-jepa-real-autogaze-cross-domain/real-autogaze-context-continue-512/ttt-model.mpk \
  --steps 5 --batch-size 4 --full-grid
```

Result:

- Sparse target loss/cosine: `0.2444 / 0.8969`
- Full-grid loss/cosine: `0.2088 / 0.9129`
- Throughput: `0.73` samples/sec

The comparable single-frame no-TTT baseline reports sparse loss/cosine
`0.2734 / 0.8849`, so the current TTT adapter improves held-out sparse loss by
about `10.6%` and cosine by `0.0120`.

## Production Verdict

The direction is viable: real AutoGaze sparse masks, real V-JEPA 2.1 weights,
ragged sparse rollout, and adapter checkpoint reload all work together on CUDA.
The adapter gives a measurable held-out quality gain over the dense single-frame
baseline while preserving the deployable free-run sparse path.

It is not yet a final production model. The remaining gates are larger
open-set training/eval, exact upstream numerical parity against a real reference
fixture, and throughput work on the sparse TTT backward/runtime path.
