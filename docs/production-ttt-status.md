# Sparse TTT Production Status

This note records the current production candidate gate for sparse temporal
V-JEPA 2.1 adapters. It is intentionally narrower than the training protocol
doc: it names the checkpoint, eval command, measured behavior, and remaining
external parity/data requirements.

## Candidate

- Base V-JEPA fixture:
  `/home/mosure/.cache/burn_jepa/vjepa2_1_vitb_dist_vitG_384/model.pt`
- TTT adapter:
  `target/burn-jepa-production-final-256/stage2-norms-low-lr/ttt-model.mpk`
- Stage2 train-loss-best checkpoint, retained for audit:
  `target/burn-jepa-production-final-256/stage2-norms-low-lr/ttt-model-best.mpk`
- Sparse policy: AutoGaze-style sparse context masks, 410 / 2048 context
  tokens, 103 / 2048 target tokens at 256px / 16 frames.
- Eval split: 164 held-out open-set windows from
  `target/burn-jepa-production-final-256/data/eval-real-autogaze.jsonl`.

In-Place TTT support is implemented as an ablation, but the active candidate is
still the SC-TTT adapter/Memory-ALiBi model above. A bounded real-checkpoint
CUDA smoke found `in_place_mlp` thirds had similar initial quality but about 4x
fast state memory and slower backward/optimizer throughput than adapter thirds,
so it is not promoted without a longer positive quality result or a
lower-overhead backward path. The stricter paper-conformance lane is
`in_place_mlp_strict`; it uses causal target generation, single `fc2`
fast-weight state, and apply-then-update full-chunk updates. Memory-ALiBi
remains labeled as a local SC-TTT extension, not an In-Place TTT claim.

## 2026-05-19 Image-Student to Video-Teacher Alignment

The deployed sparse TTT student starts from image patch tokens and the teacher
target is produced by V-JEPA 2.1 video/tubelet tokens. The alignment risk is
real: adapter-only TTT has to learn both temporal recurrence and any static
image-token to video-token distribution shift.

Matched 8-window CUDA cactus evals against the video teacher:

| Lane | Student tokens | Loss | Cosine | Late-early loss |
|---|---:|---:|---:|---:|
| Dense image V-JEPA 2.1 baseline | 2048 / 2048 | 0.3288 | 0.8636 | -0.0254 |
| Sparse image V-JEPA 2.1 baseline | 410 / 2048 | 0.4631 | 0.8048 | +0.0010 |
| Sparse Memory-ALiBi TTT | 410 / 2048 | 0.3410 | 0.8576 | -0.0828 |

Interpretation: sparse Memory-ALiBi TTT closes most of the sparse image-token
gap to the dense image baseline, but it still does not beat dense image tokens
against the video teacher on this small matched slice. That is the right signal
for staged trainable-capacity ablations; it is not evidence to fully unfreeze
patch embedding by default.

New train-scope controls:

- `ttt.pretrained_train_scope = "frozen"`: adapter-only baseline.
- `ttt.pretrained_train_scope = "norms"`: train encoder LayerNorms plus TTT.
- `ttt.pretrained_train_scope = "last_n_blocks"`: train the last N encoder
  blocks plus TTT.
- `ttt.pretrained_train_scope = "all"`: full low-LR upper-bound ablation;
  requires `ttt.freeze_pretrained = false`.

Short CUDA continuation probes now include no-TTT image-to-video controls. The
no-TTT controls start from the base V-JEPA 2.1 image pathway, train the same
video-teacher feature loss, and leave `ttt.layers = []`; this prevents the
comparison from assuming that image and video features already align.

| Lane | Steps | Dense / sparse train steps | Trainable params | Pre eval loss/cos | Post eval loss/cos | Samples/s |
|---|---:|---:|---:|---:|---:|---:|
| No-TTT image-to-video, norms | 96 | 65 / 31 | 43K | 0.3981 / 0.8293 | 0.3954 / 0.8304 | 2.82 |
| No-TTT image-to-video, last 2 blocks | 96 | 65 / 31 | 14.18M | 0.3981 / 0.8293 | 0.3967 / 0.8299 | 1.87 |
| TTT + norms, from Memory-ALiBi best | 96 | 65 / 31 | 3.59M | 0.3557 / 0.8477 | 0.3493 / 0.8504 | 0.76 |
| TTT + norms, from Memory-ALiBi best | 192 | 66 / 126 | 3.59M | 0.3557 / 0.8477 | 0.3465 / 0.8516 | 0.81 |
| TTT + last 2 blocks, short sparse probe | 32 | 9 / 23 | 17.73M | 0.3814 / 0.8384 | 0.3791 / 0.8394 | not comparable |
| TTT + last 2 blocks, lower LR short sparse probe | 32 | 9 / 23 | 17.73M | 0.3814 / 0.8384 | 0.3809 / 0.8386 | not comparable |

Interpretation: the short-budget no-TTT controls do not explain the TTT result.
Training only the image pathway, even with 14.18M late-block parameters, barely
moved the sparse image-to-video feature loss on this eval slice. The recurrent
TTT checkpoint starts substantially closer to the video teacher and norm-only
continuation improves it slightly. This does not prove a larger static
finetune could never work, but it makes TTT recurrence the current best
quality path and puts training throughput, not static capacity, at the top of
the optimization list.

The norm-only TTT continuation is promoted over the prior stage1
Memory-ALiBi checkpoint. It adds only `43,008` pretrained trainable parameters,
preserves frozen sparse patchify, and improves both matched short eval and
long-rollout gates. The final checkpoint is promoted over the step-448
train-loss-best checkpoint because it is slightly better on the same-stream
long-rollout gate. The last-block probes are not promoted because they add
`14,181,888` pretrained trainable parameters, cost more backward time, and did
not produce a proportionate token-space gain in the short run.

Full stage2-norms training result:

- 512 steps, 1024 samples, 20.0% sparse context density.
- Mixed training loss: initial `0.1931`, best deploy-rollout sample `0.1649`,
  final `0.2549`.
- Saved final model:
  `target/burn-jepa-production-final-256/stage2-norms-low-lr/ttt-model.mpk`.
- Saved train-loss-best model:
  `target/burn-jepa-production-final-256/stage2-norms-low-lr/ttt-model-best.mpk`
  at step `448`.
- Runtime: `993.6 s`, `1.03 samples/s`.
- Backward/optimizer remains the bottleneck:
  `750.1 s` backward plus `3.3 s` optimizer.
- Stream mix: `952` carried windows, `72` reset windows, no stream decay, no
  periodic reset.

Long-rollout promotion gates:

| Lane | Windows | Runtime resets | Loss | Cosine | Late-early loss |
|---|---:|---:|---:|---:|---:|
| Stage1 Memory-ALiBi final, same-stream cactus repeat | 1088 | 1 | 0.3273 | 0.8646 | -0.0009 |
| Stage2-norms train-loss-best, same-stream cactus repeat | 1088 | 1 | 0.2803 | 0.8847 | -0.0009 |
| Stage2-norms final, same-stream cactus repeat | 1088 | 1 | 0.2794 | 0.8851 | -0.0009 |
| Stage1 Memory-ALiBi final, adversarial stitched stream | 512 | 1 | 0.2850 | 0.8807 | -0.0229 |
| Stage2-norms final, adversarial stitched stream | 512 | 1 | 0.2573 | 0.8924 | -0.0193 |

Feature-stability diagnostics still do not show token collapse:

| Gate | Relative spread | Mean pairwise token cosine | Collapse score |
|---|---:|---:|---:|
| Same-stream cactus repeat, stage2 final | 0.4163 | 0.8255 | 0.4819 |
| Adversarial stitched stream, stage2 final | 0.4301 | 0.8135 | 0.4639 |

Artifacts:

- `target/burn-jepa-production-final-256/long-rollout-dense-eval/ttt-eval-report.json`
- `target/burn-jepa-production-final-256/long-rollout-base-sparse/ttt-eval-report.json`
- `target/burn-jepa-production-final-256/long-rollout-carry-forever-alibi-cactus-64x/ttt-eval-report.json`
- `target/burn-jepa-production-final-256/image2video-stage2-norms-low-lr/ttt-report.json`
- `target/burn-jepa-production-final-256/image2video-stage2-last2-low-lr/ttt-report.json`
- `target/burn-jepa-production-final-256/stage2-norms-low-lr/ttt-report.json`
- `target/burn-jepa-production-final-256/long-rollout-carry-forever-alibi-cactus-64x/ttt-eval-report-stage2-norms-best.json`
- `target/burn-jepa-production-final-256/long-rollout-carry-forever-alibi-cactus-64x/ttt-eval-report-stage2-norms-final.json`
- `target/burn-jepa-production-final-256/long-rollout-carry-forever-alibi-adversarial-8x/ttt-eval-report-stage2-norms-final.json`
- `target/burn-jepa-production-final-256/stage2-last2-low-lr-short-sparse/ttt-report.json`
- `target/burn-jepa-production-final-256/stage2-last2-low-lr-short-sparse-lr1e-7/ttt-report.json`

## 2026-05-18 Memory-ALiBi Candidate Gate

This is the previous active candidate, retained as the stage2 starting point
and audit baseline. It was trained as a continuation from the reset16
`longstable` checkpoint, but disables hard runtime resets: no clip-change
reset, no scene-change reset, no non-monotonic reset, no periodic reset, and no
stream-level state decay. The added TTT fast memory uses three ALiBi-style
banks with half-lives `[8, 64, 512]` windows.

Training command:

```sh
BURN_JEPA_TRAIN_CUDA_FORCE=1 \
cargo run --no-default-features --features cuda,sparse-patchify-cuda \
  --bin burn-jepa -- train-ttt \
  --config configs/production/vjepa21-ttt-stage1-stream-tbptt-carry-forever-alibi-cuda.toml
```

Training result after the stability-selection pass:

- Saved model:
  `target/burn-jepa-production-final-256/stage1-stream-tbptt-carry-forever-alibi/ttt-model.mpk`.
- Saved best sampled deploy-rollout checkpoint:
  `target/burn-jepa-production-final-256/stage1-stream-tbptt-carry-forever-alibi/ttt-model-best.mpk`
  at step `448`.
- 512 optimizer steps, 1024 samples, 20.0% sparse context density.
- Mixed loss trace: initial `0.2324`, best `0.2110`, final `0.2770`.
- Best checkpoint loss, excluding dense warmup/checkpoint samples:
  `0.2110`.
- Runtime: `927.6 s`, `1.104 samples/s`.
- Backward/optimizer remains the training bottleneck:
  `751.4 s` backward plus `20.7 s` optimizer.
- Stream mix: `952` carried windows, `72` reset windows, no stream decay, no
  periodic reset.
- Dense stabilization samples remained enabled: `71` dense steps and `441`
  sparse steps.
- Gradient clipping is enabled at norm `0.5`.

The training instability diagnosis was a checkpointing and selection issue,
not simply an LR issue. Two 64-step CUDA probes at `5e-7` and `2.5e-7`
learning rate with gradient clipping reproduced the same dense-to-sparse loss
transition, so lowering LR alone did not fix the apparent tail regression. The
training loop now keeps raw mixed loss visible while selecting best checkpoints
only from deploy-rollout samples by default.

This pass hardens stability and selection; it is not a quality upgrade over
the previous Memory-ALiBi eval artifact, whose same-stream cactus loss was
`0.3237`.

Matched CUDA evals:

| Lane | Windows | Runtime resets | Loss | Cosine | Late-early loss |
|---|---:|---:|---:|---:|---:|
| Base sparse V-JEPA 2.1, same-stream cactus repeat | 1088 | 1 | 0.4799 | 0.7989 | 0.0000 |
| Reset16 EMA TTT, same-stream cactus repeat | 136 | 9 | 0.3270 | 0.8666 | +0.0049 |
| Memory-ALiBi TTT final, same-stream cactus repeat | 1088 | 1 | 0.3273 | 0.8646 | -0.0009 |
| Memory-ALiBi TTT train-loss-best, same-stream cactus repeat | 1088 | 1 | 0.3289 | 0.8639 | -0.0009 |
| Reset-scene EMA TTT, adversarial stitched stream | 64 | 64 | 0.4209 | 0.8215 | -0.0338 |
| Memory-ALiBi TTT final, adversarial stitched stream | 512 | 1 | 0.2850 | 0.8807 | -0.0229 |

The important change is not just aggregate loss.  The 1088-window same-stream
run carried state for `1087` consecutive windows and stayed flat across the
four 272-window segments (`0.3279 -> 0.3271` loss).  The adversarial stitched
run carried through `502` scene switches without explicit reset and improved
from first to last quarter (`0.2967 -> 0.2739` loss).

Feature-stability diagnostics did not show token collapse:

| Gate | Relative spread | Mean pairwise token cosine | Collapse score |
|---|---:|---:|---:|
| Same-stream cactus repeat, final | 0.4573 | 0.7891 | 0.4283 |
| Adversarial stitched stream, final | 0.4620 | 0.7836 | 0.4219 |

The train-loss-best checkpoint is kept because it proves the checkpoint policy
works, but it is not the promoted deployment checkpoint: the long-rollout eval
gate selected the final checkpoint for this run. Future production promotion
should always choose from saved checkpoints using the long-rollout gate, not
raw mixed training loss alone.

Promotion decision: Memory-ALiBi replaces the reset16 `longstable` checkpoint as
the current candidate for carry-forever sparse temporal V-JEPA 2.1.  The
reset16 checkpoint remains a fallback for bounded-horizon deployments because it
uses one-third of the fast-memory bytes, but it is no longer the best
long-rollout candidate. The deploy/export config now points at this checkpoint
and includes the Memory-ALiBi TTT shape.

Artifact reports:

- `target/burn-jepa-production-final-256/stage1-stream-tbptt-carry-forever-alibi/ttt-report.json`
- `target/burn-jepa-production-final-256/long-rollout-carry-forever-alibi-cactus-64x/ttt-eval-report-final.json`
- `target/burn-jepa-production-final-256/long-rollout-carry-forever-alibi-cactus-64x/ttt-eval-report-best.json`
- `target/burn-jepa-production-final-256/long-rollout-carry-forever-alibi-cactus-64x/ttt-eval-report-base-sparse.json`
- `target/burn-jepa-production-final-256/long-rollout-carry-forever-alibi-adversarial-8x/ttt-eval-report-final.json`
- `target/burn-jepa-production-final-256/long-rollout-cactus-repeat-reset16/ttt-eval-report-longstable-reset16.json`
- `target/burn-jepa-production-final-256/long-rollout-adversarial-stitch-reset-4x/ttt-eval-report-longstable-reset-scene.json`

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

Before Memory-ALiBi, the production policy was bounded-horizon state:
reset/refresh every 16 windows and reset immediately when the scene identity
changes.  A continuation trained from the SIGReg checkpoint with a reset
curriculum ending at 16 windows produced the previous candidate:

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
- Runtime: `287.1 s`, `0.892 samples/s`; backward+optimizer remained dominant
  at `230.2 s`.

Stress evals with the longstable checkpoint:

| Gate | Windows | Resets | Loss | Cosine | Late-early loss |
|---|---:|---:|---:|---:|---:|
| Same-stream cactus repeat, reset16 | 136 | 9 | 0.3270 | 0.8666 | +0.0049 |
| Scene-stitched stream, scene reset | 64 | 11 | 0.2896 | 0.8783 | -0.0675 |
| Adversarial stitched stream, scene reset | 64 | 64 | 0.4209 | 0.8215 | -0.0338 |

Matched 164-window sparse long-rollout comparison:

| Lane | Carried windows | Resets | Loss | Cosine | Samples/s |
|---|---:|---:|---:|---:|---:|
| Base sparse V-JEPA 2.1 | 145 | 19 | 0.4273 | 0.8187 | 3.28 |
| TTT reset each window | 0 | 164 | 0.3965 | 0.8319 | 3.57 |
| TTT carried state | 145 | 19 | 0.3140 | 0.8694 | 3.64 |

This resolved the old paper caveat in a narrower, production-relevant sense:
arbitrary-duration streams are stable when the runtime uses the trained
bounded-horizon refresh policy.  It does not show that a single fast state can
be carried forever without reset; that mode is explicitly measured as a
failure case for the earlier EMA TTT memory.

Current Memory-ALiBi deployment bundle:

- Upload directory:
  `target/burn-jepa-cdn-upload/vjepa2_1_ttt`.
- Manifest URL after upload:
  `https://aberration.technology/model/burn_jepa/vjepa2_1_ttt/manifest.json`.
- Record dtype: f16-only, 12 shards, 226,540,288 total shard bytes.
- Verified with `burn-jepa verify-bpk --manifest ...`; the clean
  `burn_store::BurnpackStore + ModuleSnapshot::load_from` path applied all 329
  tensors with no missing, skipped, unused, or errored records.
- Package manifest includes `ttt_config.memory_dynamics = "memory_alibi"`,
  half-lives `[8, 64, 512]`, and `memory_clip_rms = 16.0`.

## Carry-Forever Workstream

The no-hard-reset workstream now has a separate Memory-ALiBi TTT mode. It keeps
the frozen V-JEPA 2.1 attention/position path unchanged and applies the
ALiBi-style recency prior only to the added TTT fast memory. Each TTT layer uses
three log-spaced fast-weight banks by default (`8`, `64`, and `512` window
half-lives), with no external stream reset or stream-level decay in the new
carry-forever configs.

Entry points:

- Train:
  `configs/production/vjepa21-ttt-stage1-stream-tbptt-carry-forever-alibi-cuda.toml`.
- Same-scene 1024+ window eval:
  `configs/production/vjepa21-ttt-long-rollout-carry-forever-alibi-cactus-64x-cuda.toml`.
- Adversarial stitched no-reset eval:
  `configs/production/vjepa21-ttt-long-rollout-carry-forever-alibi-adversarial-8x-cuda.toml`.

This workstream has passed the first carry-forever promotion gate and is now
the active candidate. Keep the reset16 checkpoint as the bounded-horizon
fallback until the Memory-ALiBi package has broader cross-domain eval beyond the
same-stream and adversarial stitched gates above.

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
