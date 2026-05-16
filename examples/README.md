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
  --force
```
