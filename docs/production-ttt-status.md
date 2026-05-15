# Sparse TTT Production Status

This note records the current production candidate gate for sparse temporal
V-JEPA 2.1 adapters. It is intentionally narrower than the training protocol
doc: it names the checkpoint, eval command, measured behavior, and remaining
external parity/data requirements.

## Candidate

- Base V-JEPA fixture:
  `/home/mosure/.cache/burn_jepa/vjepa2_1_vitb_dist_vitG_384/model.pt`
- TTT adapter:
  `target/burn-jepa-production-ttt-vjepa21-official/autogaze-sparse-224-context-1024/ttt-model.mpk`
- Sparse policy: AutoGaze-style sparse context masks, 314 / 1568 context
  tokens, 78 / 1568 target tokens.
- Eval split: 164 held-out open-set windows from
  `target/burn-jepa-production-ttt-large/data/eval.jsonl`.

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
- Micro parity: prediction max abs diff `1.006e-4`, target max abs diff
  `1.034e-5`.

## Held-Out Eval

Sparse production rollout:

```sh
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config target/burn-jepa-production-ttt-vjepa21-official/configs/autogaze-sparse-224-context-eval-trained-fast.toml \
  --model target/burn-jepa-production-ttt-vjepa21-official/autogaze-sparse-224-context-1024/ttt-model.mpk \
  --steps 11 --batch-size 16 --no-full-grid
```

Result:

- Sparse free-run loss/cosine: `0.3021 / 0.8746`
- Throughput: `5.91` samples/sec over 164 windows
- Stage time: teacher `6828 ms`, student `12542 ms`, loss `687 ms`
- Zero-init same-loader baseline: `0.4544 / 0.8076`
- Loss reduction vs zero-init: `33.5%`; cosine gain: `+0.0670`

Full-grid diagnostic:

```sh
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- eval-ttt \
  --config target/burn-jepa-production-ttt-vjepa21-official/configs/autogaze-sparse-224-context-eval-trained-full32.toml \
  --model target/burn-jepa-production-ttt-vjepa21-official/autogaze-sparse-224-context-1024/ttt-model.mpk \
  --steps 2 --batch-size 16 --full-grid
```

Result:

- Sparse target loss/cosine: `0.3028 / 0.8746`
- Full-grid loss/cosine: `0.2552 / 0.8948`
- Throughput: `1.23` samples/sec over 32 windows

Training behavior:

- 1024 CUDA steps, 1024 samples, 20.0% sparse-context density.
- Initial loss `0.3672`, best loss `0.1482`, final loss `0.2383`.
- Throughput `0.448` samples/sec.
- Runtime bottleneck remains backward/optimizer: `2070.8 s` of `2286.3 s`
  elapsed.

## Production Verdict

The direction is viable: exact official V-JEPA 2.1 torch.hub parity, sparse
rollout, sparse patchify, adapter training, checkpoint reload, and CUDA/WebGPU
smokes all work together. The fresh official-2.1 adapter gives a clear
held-out quality gain over a same-loader zero-init TTT baseline while preserving
the deployable free-run sparse path.

It is not yet a final production model. The remaining gate is throughput and
scale: train on a larger, more diverse real AutoGaze-mask corpus, run
cross-domain eval, and reduce sparse TTT backward/optimizer cost.
