#!/usr/bin/env python3
import json
import math
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


BATCH = 1
CHANNELS = 3
FRAMES = 2
IMAGE_SIZE = 16
PATCH_SIZE = 8
TUBELET_SIZE = 1
ENCODER_DIM = 24
ENCODER_HEADS = 3
PREDICTOR_DIM = 24
PREDICTOR_HEADS = 3
MLP_RATIO = 2.0
EPS = 1.0e-6
GRID = (2, 2, 2)
CONTEXT = [0, 2, 5, 7]
TARGET = [1, 3, 4, 6]


def linear(x, weight, bias):
    if weight.shape[0] == x.shape[-1]:
        y = x.matmul(weight)
    elif weight.shape[1] == x.shape[-1]:
        y = x.matmul(weight.t())
    else:
        raise RuntimeError(f"linear shape mismatch: x={x.shape} weight={weight.shape}")
    return y if bias is None else y + bias


def layer_norm(x, gamma, beta):
    return F.layer_norm(x, (x.shape[-1],), gamma, beta, EPS)


def token_index_to_coords(index):
    tokens_per_frame = GRID[1] * GRID[2]
    frame = index // tokens_per_frame
    rem = index - frame * tokens_per_frame
    row = rem // GRID[2]
    col = rem - row * GRID[2]
    return frame, row, col


def append_rotary_axis(sin, cos, dim, pos):
    half = dim // 2
    for i in range(half):
        omega = 1.0 / math.pow(10000.0, i / max(half, 1))
        angle = pos * omega
        s = math.sin(angle)
        c = math.cos(angle)
        sin.extend([s, s])
        cos.extend([c, c])


def rotary_sin_cos(indices, head_dim):
    axis_dim = 2 * ((head_dim // 3) // 2)
    sin = []
    cos = []
    for index in indices:
        frame, row, col = token_index_to_coords(index)
        append_rotary_axis(sin, cos, axis_dim, float(frame))
        append_rotary_axis(sin, cos, axis_dim, float(row))
        append_rotary_axis(sin, cos, axis_dim, float(col))
        used = axis_dim * 3
        sin.extend([0.0] * (head_dim - used))
        cos.extend([1.0] * (head_dim - used))
    sin = torch.tensor(sin, dtype=torch.float32).reshape(1, len(indices), head_dim)
    cos = torch.tensor(cos, dtype=torch.float32).reshape(1, len(indices), head_dim)
    return sin.repeat(BATCH, 1, 1), cos.repeat(BATCH, 1, 1)


def rotate_half_pairs(x):
    b, h, n, d = x.shape
    paired = x.reshape(b, h, n, d // 2, 2)
    first = paired[..., 0:1]
    second = paired[..., 1:2]
    return torch.cat((-second, first), dim=-1).reshape(b, h, n, d)


def apply_rotary(x, indices, head_dim):
    sin, cos = rotary_sin_cos(indices, head_dim)
    sin = sin.unsqueeze(1)
    cos = cos.unsqueeze(1)
    return (x * cos) + (rotate_half_pairs(x) * sin)


def block(x, tensors, prefix, heads, dim, indices):
    head_dim = dim // heads
    norm = layer_norm(
        x,
        tensors[f"{prefix}.norm1.gamma"],
        tensors[f"{prefix}.norm1.beta"],
    )
    qkv = linear(
        norm,
        tensors[f"{prefix}.attn.qkv.weight"],
        tensors[f"{prefix}.attn.qkv.bias"],
    )
    q = qkv[..., :dim].reshape(BATCH, -1, heads, head_dim).permute(0, 2, 1, 3)
    k = qkv[..., dim : dim * 2].reshape(BATCH, -1, heads, head_dim).permute(0, 2, 1, 3)
    v = qkv[..., dim * 2 : dim * 3].reshape(BATCH, -1, heads, head_dim).permute(0, 2, 1, 3)
    q = apply_rotary(q, indices, head_dim)
    k = apply_rotary(k, indices, head_dim)
    attn = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(BATCH, -1, dim)
    out = linear(out, tensors[f"{prefix}.attn.proj.weight"], tensors[f"{prefix}.attn.proj.bias"])
    x = x + out
    norm = layer_norm(
        x,
        tensors[f"{prefix}.norm2.gamma"],
        tensors[f"{prefix}.norm2.beta"],
    )
    hidden = int(round(dim * MLP_RATIO))
    mlp = linear(norm, tensors[f"{prefix}.mlp.fc1.weight"], tensors[f"{prefix}.mlp.fc1.bias"])
    if mlp.shape[-1] != hidden:
        raise RuntimeError(f"unexpected mlp hidden dim: {mlp.shape[-1]} != {hidden}")
    mlp = linear(F.gelu(mlp), tensors[f"{prefix}.mlp.fc2.weight"], tensors[f"{prefix}.mlp.fc2.bias"])
    return x + mlp


def encode(video, tensors, indices):
    x = F.conv3d(
        video,
        tensors["encoder.patch_embed.proj.weight"],
        tensors["encoder.patch_embed.proj.bias"],
        stride=(TUBELET_SIZE, PATCH_SIZE, PATCH_SIZE),
    )
    b, dim, depth, height, width = x.shape
    if (b, dim, depth, height, width) != (BATCH, ENCODER_DIM, 2, 2, 2):
        raise RuntimeError(f"unexpected patch shape: {x.shape}")
    x = x.reshape(b, dim, depth * height * width).transpose(1, 2)
    x = x[:, indices, :]
    x = block(x, tensors, "encoder.blocks.0", ENCODER_HEADS, ENCODER_DIM, indices)
    return layer_norm(
        x,
        tensors["encoder.norms_block.0.gamma"],
        tensors["encoder.norms_block.0.beta"],
    )


def predictor(context_tokens, tensors):
    context = linear(
        context_tokens,
        tensors["predictor.predictor_embed.weight"],
        tensors["predictor.predictor_embed.bias"],
    )
    target = tensors["predictor.mask_tokens.0"].reshape(1, 1, PREDICTOR_DIM).repeat(
        BATCH, len(TARGET), 1
    )
    sequence = torch.cat((context, target), dim=1)
    merged = CONTEXT + TARGET
    sort_order = sorted(range(len(merged)), key=lambda i: merged[i])
    reverse_order = [0] * len(sort_order)
    for sorted_pos, original_pos in enumerate(sort_order):
        reverse_order[original_pos] = sorted_pos
    sorted_indices = [merged[i] for i in sort_order]
    sequence = sequence[:, sort_order, :]
    sequence = block(
        sequence,
        tensors,
        "predictor.blocks.0",
        PREDICTOR_HEADS,
        PREDICTOR_DIM,
        sorted_indices,
    )
    sequence = layer_norm(sequence, tensors["predictor.norm.gamma"], tensors["predictor.norm.beta"])
    sequence = sequence[:, reverse_order, :]
    targets = sequence[:, len(CONTEXT) : len(CONTEXT) + len(TARGET), :]
    return linear(
        targets,
        tensors["predictor.target_proj.weight"],
        tensors["predictor.target_proj.bias"],
    )


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: vjepa_tiny_parity.py <weights.safetensors> <output.json>")
    weights_path, output_path = sys.argv[1:]
    tensors = {key: value.float() for key, value in load_file(weights_path).items()}
    count = BATCH * CHANNELS * FRAMES * IMAGE_SIZE * IMAGE_SIZE
    values = [((i % 29) - 14) / 31.0 for i in range(count)]
    video = torch.tensor(values, dtype=torch.float32).reshape(
        BATCH, CHANNELS, FRAMES, IMAGE_SIZE, IMAGE_SIZE
    )
    context_tokens = encode(video, tensors, CONTEXT)
    target_tokens = encode(video, tensors, TARGET)
    predictions = predictor(context_tokens, tensors)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "predictions": predictions.reshape(-1).tolist(),
                "targets": target_tokens.reshape(-1).tolist(),
            },
            f,
        )


if __name__ == "__main__":
    main()
