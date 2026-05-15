import json
import os
import sys
from pathlib import Path

import torch


def clean_backbone_key(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def load_official_vjepa21(checkpoint_dir: Path, weights_name: str, model_name: str, num_frames: int):
    encoder, predictor = torch.hub.load(
        "facebookresearch/vjepa2",
        model_name,
        pretrained=False,
        num_frames=num_frames,
        trust_repo=True,
    )
    checkpoint = torch.load(checkpoint_dir / weights_name, map_location="cpu")
    encoder_state = checkpoint.get("ema_encoder") or checkpoint.get("encoder")
    if encoder_state is None:
        raise RuntimeError("checkpoint is missing ema_encoder/encoder state")
    encoder.load_state_dict(clean_backbone_key(encoder_state), strict=True)
    predictor.load_state_dict(clean_backbone_key(checkpoint["predictor"]), strict=True)
    encoder.eval()
    predictor.eval()
    return encoder, predictor


def deterministic_video(config: dict) -> torch.Tensor:
    frames = int(config.get("tubelet_size", 2))
    channels = int(config.get("in_channels", 3))
    patch = int(config.get("patch_size", 16))
    height = patch * 2
    width = patch * 2
    size = channels * frames * height * width
    values = torch.arange(size, dtype=torch.float32)
    values = ((values % 31) - 15) / 23.0
    return values.reshape(1, channels, frames, height, width)


def main() -> None:
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: vjepa21_torchhub_real_micro_forward.py "
            "<checkpoint_dir> <weights_name> <model_name> <num_frames> <output.json>"
        )

    os.environ.setdefault("TORCH_HOME", str(Path.home() / ".cache" / "torch"))
    torch.set_num_threads(1)
    checkpoint_dir = Path(sys.argv[1])
    weights_name = sys.argv[2]
    model_name = sys.argv[3]
    num_frames = int(sys.argv[4])
    output_path = Path(sys.argv[5])
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    encoder, predictor = load_official_vjepa21(checkpoint_dir, weights_name, model_name, num_frames)

    context = torch.tensor([[0, 2]], dtype=torch.long)
    target = torch.tensor([[1, 3]], dtype=torch.long)
    with torch.no_grad():
        dense = encoder(deterministic_video(config))
        context_tokens = dense[:, context[0], :]
        predictions, _ = predictor(
            context_tokens,
            [context],
            [target],
            mask_index=int(os.environ.get("BURN_JEPA_VJEPA21_MASK_INDEX", "1")),
        )
        targets = dense[:, target[0], :]

    payload = {
        "predictions": predictions.contiguous().flatten().tolist(),
        "targets": targets.contiguous().flatten().tolist(),
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
