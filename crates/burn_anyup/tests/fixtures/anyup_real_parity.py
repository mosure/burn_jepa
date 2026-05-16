#!/usr/bin/env python3
import importlib.util
import json
import sys
from pathlib import Path

import torch


def load_reference_module():
    path = Path(__file__).with_name("anyup_tiny_parity.py")
    spec = importlib.util.spec_from_file_location("anyup_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.QK = 128
    module.HEADS = 4
    module.GROUPS = 8
    module.WINDOW_RATIO = 0.1
    module.RMS_EPS = torch.finfo(torch.float32).eps
    return module


def efficient_state_dict(raw):
    state = dict(raw)
    q_w, k_w, _ = raw["cross_decode.cross_attn.attention.in_proj_weight"].chunk(3)
    q_b, k_b, _ = raw["cross_decode.cross_attn.attention.in_proj_bias"].chunk(3)
    state["cross_decode.cross_attn.q_proj.weight"] = q_w.reshape(128, 128, 1, 1).contiguous()
    state["cross_decode.cross_attn.q_proj.bias"] = q_b.contiguous()
    state["cross_decode.cross_attn.k_proj.weight"] = k_w.reshape(128, 128, 1, 1).contiguous()
    state["cross_decode.cross_attn.k_proj.bias"] = k_b.contiguous()
    state["cross_decode.norm_q.weight"] = raw["cross_decode.cross_attn.norm_q.weight"]
    state["cross_decode.norm_k.weight"] = raw["cross_decode.cross_attn.norm_k.weight"]
    return state


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: anyup_real_parity.py <checkpoint.pth> <out-dir>")
    checkpoint = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    reference = load_reference_module()
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = efficient_state_dict(raw["state_dict"] if "state_dict" in raw else raw)

    torch.manual_seed(20260515)
    image = torch.linspace(-1.0, 1.0, 1 * 3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32)
    features = torch.linspace(-0.75, 0.75, 1 * 768 * 4 * 4, dtype=torch.float32).reshape(1, 768, 4, 4)
    with torch.no_grad():
        output = reference.forward(image, features, state, q_chunk_size=2)
        chunked = reference.forward(image, features, state, q_chunk_size=1)

    (out / "fixture.json").write_text(json.dumps({
        "image": image.flatten().tolist(),
        "image_shape": list(image.shape),
        "features": features.flatten().tolist(),
        "features_shape": list(features.shape),
        "output": output.flatten().tolist(),
        "chunked_output": chunked.flatten().tolist(),
        "output_shape": list(output.shape),
    }))


if __name__ == "__main__":
    main()
