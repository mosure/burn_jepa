# burn_jepa examples

Examples are runnable workflows and artifact generators that exercise the
library without expanding the package CLI surface. The root crate keeps only the
canonical `burn-jepa` binary under `src/bin/`.

## High-resolution pipeline breakdown

```bash
cargo run --example highres_breakdown --no-default-features --features ndarray
```

## E2E video gallery

```bash
python3 tools/video_feature_gallery.py --download --samples 16 --frames 40 --image-size 224 --force
```

The Python wrapper downloads/extracts the UCSD pedestrian frame dataset and then
delegates rendering to:

```bash
cargo run --example video_gallery_e2e --no-default-features --features ndarray -- \
  --dataset-root target/burn-jepa-video-gallery/source/UCSD_Anomaly_Dataset.v1p2 \
  --samples 16 \
  --frames 40 \
  --image-size 224 \
  --model-manifest target/burn-jepa-web/model/vjepa2_1_base/manifest.json \
  --anyup-model-manifest target/burn_anyup/anyup_multi_backbone/manifest.json \
  --anyup-attention-mode upstream-masked \
  --force
```

Use real V-JEPA and AnyUp package manifests for README or paper assets. Omitting
`--model-manifest` or `--anyup-model-manifest` is only for fast path smoke
tests; the tiny untrained modules validate Burn pipeline wiring rather than
pretrained feature or AnyUp visual quality.

For a quick README-frame regeneration from one sparse lane:

```bash
python3 tools/video_feature_gallery.py \
  --dataset-root target/burn-jepa-video-gallery/source/UCSD_Anomaly_Dataset.v1p2 \
  --output target/burn-jepa-video-gallery-readme-real \
  --samples 1 \
  --frames 4 \
  --image-size 256 \
  --config patchdiff_50 \
  --model-manifest target/burn-jepa-web/model/vjepa2_1_base/manifest.json \
  --anyup-model-manifest target/burn_anyup/anyup_multi_backbone/manifest.json \
  --anyup-attention-mode upstream-masked \
  --pca-update-every 1 \
  --force
```
