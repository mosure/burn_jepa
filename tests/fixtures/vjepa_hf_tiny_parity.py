import json
import sys
from pathlib import Path

import torch
from transformers import VJEPA2Config, VJEPA2Model


def deterministic_fill(model: VJEPA2Model) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            offset = (sum(name.encode("utf-8")) % 97) - 48
            values = torch.arange(param.numel(), dtype=torch.float32).reshape(param.shape)
            param.copy_(((values % 89) - 44 + offset) / 700.0)


def deterministic_video(config: VJEPA2Config) -> torch.Tensor:
    size = 1 * config.frames_per_clip * config.in_chans * config.crop_size * config.crop_size
    values = torch.arange(size, dtype=torch.float32)
    values = ((values % 31) - 15) / 23.0
    return values.reshape(1, config.frames_per_clip, config.in_chans, config.crop_size, config.crop_size)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: vjepa_hf_tiny_parity.py <model_dir> <output.json>")

    model_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    config = VJEPA2Config(
        architectures=["VJEPA2Model"],
        crop_size=16,
        frames_per_clip=2,
        patch_size=8,
        tubelet_size=1,
        hidden_size=24,
        in_chans=3,
        num_attention_heads=3,
        num_hidden_layers=1,
        mlp_ratio=2.0,
        pred_hidden_size=24,
        pred_num_attention_heads=3,
        pred_num_hidden_layers=1,
        pred_num_mask_tokens=2,
        pred_zero_init_mask_tokens=False,
        pred_mlp_ratio=2.0,
        layer_norm_eps=1e-6,
    )
    model = VJEPA2Model(config)
    model.config._attn_implementation = "eager"
    model.eval()
    deterministic_fill(model)
    model.save_pretrained(model_dir, safe_serialization=True)

    context = torch.arange(8, dtype=torch.long).reshape(1, 8)
    target = torch.tensor([[1, 3, 4, 6]], dtype=torch.long)
    video = deterministic_video(config)
    with torch.no_grad():
        output = model(
            video,
            context_mask=[context],
            target_mask=[target],
            output_attentions=False,
            output_hidden_states=False,
        )

    payload = {
        "predictions": output.predictor_output.last_hidden_state.flatten().tolist(),
        "targets": output.predictor_output.target_hidden_state.flatten().tolist(),
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
