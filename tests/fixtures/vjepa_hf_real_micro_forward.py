import json
import os
import sys
from pathlib import Path

import torch
from transformers import VJEPA2Model


def deterministic_video(model: VJEPA2Model) -> torch.Tensor:
    config = model.config
    frames = config.tubelet_size
    height = config.patch_size
    width = config.patch_size
    size = 1 * frames * config.in_chans * height * width
    values = torch.arange(size, dtype=torch.float32)
    values = ((values % 31) - 15) / 23.0
    return values.reshape(1, frames, config.in_chans, height, width)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: vjepa_hf_real_micro_forward.py <checkpoint_dir> <output.json>")

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    checkpoint_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    model = VJEPA2Model.from_pretrained(checkpoint_dir, local_files_only=True)
    model.config._attn_implementation = "eager"
    model.eval()

    mask = torch.tensor([[0]], dtype=torch.long)
    with torch.no_grad():
        output = model(
            deterministic_video(model),
            context_mask=[mask],
            target_mask=[mask],
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
