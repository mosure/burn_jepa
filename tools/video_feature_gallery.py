#!/usr/bin/env python3
"""Download UCSD pedestrian clips and run the Burn E2E gallery example."""

from __future__ import annotations

import argparse
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlretrieve


DATA_URL = "http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz"
DATASET_DIR = "UCSD_Anomaly_Dataset.v1p2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("target/burn-jepa-video-gallery"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--anyup-model-manifest", type=Path)
    parser.add_argument("--anyup-weights", type=Path)
    parser.add_argument("--anyup-attention-mode", default="efficient-local")
    parser.add_argument(
        "--config",
        action="append",
        help="render one gallery config id; may be passed multiple times or comma-separated",
    )
    parser.add_argument("--pca-update-every", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-url", default=DATA_URL)
    return parser.parse_args()


def ensure_dataset(args: argparse.Namespace) -> Path:
    if args.dataset_root is not None:
        dataset = args.dataset_root.expanduser().resolve()
        if not dataset.exists():
            raise FileNotFoundError(f"dataset root does not exist: {dataset}")
        return dataset

    source_dir = args.output.resolve() / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    dataset = source_dir / DATASET_DIR
    if dataset.exists():
        return dataset

    archive = source_dir / "UCSD_Anomaly_Dataset.tar.gz"
    if not archive.exists():
        if not args.download:
            raise FileNotFoundError(
                f"dataset missing at {dataset}; rerun with --download or pass --dataset-root"
            )
        print(f"downloading dataset: {args.data_url}", flush=True)
        urlretrieve(args.data_url, archive)

    print(f"extracting dataset: {archive}", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(source_dir)
    if not dataset.exists():
        raise FileNotFoundError(f"expected extracted dataset at {dataset}")
    return dataset


def main() -> int:
    args = parse_args()
    dataset = ensure_dataset(args)
    command = [
        "cargo",
        "run",
        "--example",
        "video_gallery_e2e",
        "--no-default-features",
        "--features",
        "ndarray",
        "--",
        "--output",
        str(args.output.resolve()),
        "--dataset-root",
        str(dataset),
        "--samples",
        str(args.samples),
        "--frames",
        str(args.frames),
        "--stride",
        str(args.stride),
        "--fps",
        str(args.fps),
        "--image-size",
        str(args.image_size),
        "--anyup-attention-mode",
        args.anyup_attention_mode,
        "--pca-update-every",
        str(args.pca_update_every),
    ]
    if args.model_manifest is not None:
        command.extend(["--model-manifest", str(args.model_manifest.expanduser().resolve())])
    if args.anyup_model_manifest is not None:
        command.extend(
            ["--anyup-model-manifest", str(args.anyup_model_manifest.expanduser().resolve())]
        )
    if args.anyup_weights is not None:
        command.extend(["--anyup-weights", str(args.anyup_weights.expanduser().resolve())])
    for config in args.config or []:
        command.extend(["--config", config])
    if args.force:
        command.append("--force")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
