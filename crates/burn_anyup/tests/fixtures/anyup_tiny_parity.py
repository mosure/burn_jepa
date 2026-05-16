#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


QK = 8
HEADS = 2
GROUPS = 2
GN_EPS = 1.0e-5
RMS_EPS = torch.finfo(torch.float32).eps
WINDOW_RATIO = 0.4


def randn(*shape):
    return torch.randn(*shape, dtype=torch.float32) * 0.05


def rope_freqs(dim=QK, theta=100.0):
    freqs_1d = theta ** torch.linspace(0, -1, dim // 4)
    freqs_1d = torch.cat([freqs_1d, freqs_1d])
    freqs_2d = torch.zeros(2, dim)
    freqs_2d[0, : dim // 2] = freqs_1d
    freqs_2d[1, -dim // 2 :] = freqs_1d
    return freqs_2d * 2 * math.pi


def add_resblock(state, prefix, channels=QK):
    state[f"{prefix}.block.0.weight"] = torch.randn(channels) * 0.02 + 1.0
    state[f"{prefix}.block.0.bias"] = torch.randn(channels) * 0.02
    state[f"{prefix}.block.2.weight"] = randn(channels, channels, 1, 1)
    state[f"{prefix}.block.3.weight"] = torch.randn(channels) * 0.02 + 1.0
    state[f"{prefix}.block.3.bias"] = torch.randn(channels) * 0.02
    state[f"{prefix}.block.5.weight"] = randn(channels, channels, 1, 1)


def add_conv_encoder(state, prefix, in_channels, pre_kernel):
    state[f"{prefix}.0.weight"] = randn(QK, in_channels, pre_kernel, pre_kernel)
    add_resblock(state, f"{prefix}.1")
    add_resblock(state, f"{prefix}.2")


def add_state():
    torch.manual_seed(1234)
    state = {}
    add_conv_encoder(state, "image_encoder", 3, 1)
    add_conv_encoder(state, "key_encoder", QK, 1)
    add_conv_encoder(state, "query_encoder", QK, 1)
    state["key_features_encoder.0.basis"] = randn(QK, 1, 3, 3)
    add_resblock(state, "key_features_encoder.1")
    add_resblock(state, "key_features_encoder.2")
    add_conv_encoder(state, "aggregation", 2 * QK, 3)
    state["cross_decode.conv2d.weight"] = randn(QK, QK, 3, 3)
    state["cross_decode.norm_q.weight"] = torch.randn(QK) * 0.02 + 1.0
    state["cross_decode.norm_k.weight"] = torch.randn(QK) * 0.02 + 1.0
    state["cross_decode.cross_attn.q_proj.weight"] = randn(QK, QK, 1, 1)
    state["cross_decode.cross_attn.q_proj.bias"] = randn(QK)
    state["cross_decode.cross_attn.k_proj.weight"] = randn(QK, QK, 1, 1)
    state["cross_decode.cross_attn.k_proj.bias"] = randn(QK)
    state["rope.freqs"] = rope_freqs()
    return state


def conv_reflect(x, weight):
    k = weight.shape[-1]
    p = k // 2
    if p:
        x = F.pad(x, (p, p, p, p), mode="reflect")
    return F.conv2d(x, weight)


def resblock(x, state, prefix):
    shortcut = x
    y = F.group_norm(
        x,
        GROUPS,
        state[f"{prefix}.block.0.weight"],
        state[f"{prefix}.block.0.bias"],
        eps=GN_EPS,
    )
    y = F.silu(y)
    y = conv_reflect(y, state[f"{prefix}.block.2.weight"])
    y = F.group_norm(
        y,
        GROUPS,
        state[f"{prefix}.block.3.weight"],
        state[f"{prefix}.block.3.bias"],
        eps=GN_EPS,
    )
    y = F.silu(y)
    y = conv_reflect(y, state[f"{prefix}.block.5.weight"])
    return y + shortcut


def conv_encoder(x, state, prefix):
    x = conv_reflect(x, state[f"{prefix}.0.weight"])
    x = resblock(x, state, f"{prefix}.1")
    x = resblock(x, state, f"{prefix}.2")
    return x


def lfu(features, basis):
    b, c, h, w = features.shape
    k = basis.shape[-1]
    p = k // 2
    x = F.pad(features, (p, p, p, p), value=0)
    x = F.conv2d(x, basis.repeat(c, 1, 1, 1), groups=c)
    mask = torch.ones(1, 1, h, w, dtype=x.dtype)
    denom = F.conv2d(
        F.pad(mask, (p, p, p, p), value=0),
        torch.ones(1, 1, k, k, dtype=x.dtype),
    )
    x = (x / denom).view(b, basis.shape[0], c, h, w)
    return F.softmax(x, dim=1).mean(dim=2)


def feature_encoder(features, state):
    features = F.normalize(features, dim=1)
    x = lfu(features, state["key_features_encoder.0.basis"])
    x = resblock(x, state, "key_features_encoder.1")
    x = resblock(x, state, "key_features_encoder.2")
    return x


def coords(h, w):
    y = torch.linspace(0, 1, h)
    x = torch.linspace(0, 1, w)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([yy, xx], dim=-1).view(1, h * w, 2)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope(x, freqs):
    angle = coords(int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))) @ freqs
    return x * angle.cos() + rotate_half(x) * angle.sin()


def flatten(x):
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def unflatten(x, h, w):
    b, _, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


def rms_norm(x, weight):
    return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + RMS_EPS) * weight


def odd(value):
    return value if value % 2 == 1 else value + 1


def natten_window(hq, wq, hk, wk):
    dilation_h = max(1, hq // hk)
    dilation_w = max(1, wq // wk)
    if 0 < WINDOW_RATIO < 0.5:
        kernel_h = odd(max(3, round(2 * WINDOW_RATIO * hk)))
        kernel_w = odd(max(3, round(2 * WINDOW_RATIO * wk)))
    else:
        kernel_h = odd(hk)
        kernel_w = odd(wk)
    return kernel_h, kernel_w, dilation_h, dilation_w


def efficient_local_attention(q, k, v, q_chunk_size=None):
    b, qk_dim, hq, wq = q.shape
    _, _, hk, wk = k.shape
    value_dim = v.shape[1]
    head_dim = qk_dim // HEADS
    kernel_h, kernel_w, dilation_h, dilation_w = natten_window(hq, wq, hk, wk)
    radius_h = kernel_h // 2
    radius_w = kernel_w // 2
    pad_h = radius_h * dilation_h
    pad_w = radius_w * dilation_w

    k = F.interpolate(k, size=(hq, wq), mode="nearest-exact")
    v = F.interpolate(v, size=(hq, wq), mode="nearest-exact")
    q = q.reshape(b, HEADS, head_dim, hq, wq)
    k = k.reshape(b * HEADS, head_dim, hq, wq)
    k = F.pad(k, (pad_w, pad_w, pad_h, pad_h), value=0)
    v = F.pad(v, (pad_w, pad_w, pad_h, pad_h), value=0)
    valid = F.pad(torch.ones(1, 1, hq, wq), (pad_w, pad_w, pad_h, pad_h), value=0)

    rows_per_chunk = hq if q_chunk_size is None else max(1, q_chunk_size * max(1, hq // hk))
    outputs = []
    for start_row in range(0, hq, rows_per_chunk):
        end_row = min(hq, start_row + rows_per_chunk)
        chunk_h = end_row - start_row
        q_chunk = q[:, :, :, start_row:end_row, :]
        logits = []
        values = []
        for ky in range(kernel_h):
            for kx in range(kernel_w):
                row_start = start_row + pad_h + (ky - radius_h) * dilation_h
                col_start = pad_w + (kx - radius_w) * dilation_w
                k_shift = k[:, :, row_start:row_start + chunk_h, col_start:col_start + wq]
                k_shift = k_shift.reshape(b, HEADS, head_dim, chunk_h, wq)
                mask = valid[:, :, row_start:row_start + chunk_h, col_start:col_start + wq]
                score = (q_chunk * k_shift).sum(dim=2) * (head_dim ** -0.5)
                score = score[:, :, None] + (1 - mask[:, :, None]) * -1.0e9
                logits.append(score)
                values.append(v[:, :, row_start:row_start + chunk_h, col_start:col_start + wq])
        weights = torch.softmax(torch.cat(logits, dim=2), dim=2).mean(dim=1)
        out = torch.zeros(b, value_dim, chunk_h, wq)
        for offset, value in enumerate(values):
            out = out + value * weights[:, offset:offset + 1]
        outputs.append(out)
    return torch.cat(outputs, dim=2)


def attention_mask(hq, wq, hk, wk):
    values = []
    for row in range(hq):
        for col in range(wq):
            row_pos = (row + 0.5) / hq
            col_pos = (col + 0.5) / wq
            row_start = math.floor(max(0.0, row_pos - WINDOW_RATIO) * hk)
            row_end = math.ceil(min(1.0, row_pos + WINDOW_RATIO) * hk)
            col_start = math.floor(max(0.0, col_pos - WINDOW_RATIO) * wk)
            col_end = math.ceil(min(1.0, col_pos + WINDOW_RATIO) * wk)
            for key_row in range(hk):
                for key_col in range(wk):
                    allowed = row_start <= key_row < row_end and col_start <= key_col < col_end
                    values.append(0.0 if allowed else -1.0e9)
    return torch.tensor(values, dtype=torch.float32).view(hq * wq, hk * wk)


def upstream_masked_attention(q, k, v, q_chunk_size=None):
    b, q_tokens, qk_dim = q.shape
    k_tokens = k.shape[1]
    head_dim = qk_dim // HEADS
    q = q.view(b, q_tokens, HEADS, head_dim).transpose(1, 2)
    k = k.view(b, k_tokens, HEADS, head_dim).transpose(1, 2)
    k_t = k.transpose(-2, -1)
    chunk_size = q_tokens if q_chunk_size is None else max(1, q_chunk_size)
    hq = int(math.sqrt(q_tokens))
    wq = q_tokens // hq
    hk = int(math.sqrt(k_tokens))
    wk = k_tokens // hk
    mask = attention_mask(hq, wq, hk, wk) if WINDOW_RATIO > 0 else None
    outputs = []
    for start in range(0, q_tokens, chunk_size):
        end = min(q_tokens, start + chunk_size)
        logits = (q[:, :, start:end] @ k_t) * (head_dim ** -0.5)
        if mask is not None:
            logits = logits + mask[start:end][None, None]
        attn = torch.softmax(logits, dim=-1).mean(dim=1)
        outputs.append(attn @ v)
    return torch.cat(outputs, dim=1)


def cross_decode(q, k, v, state, q_chunk_size=None, mode="efficient"):
    q = F.conv2d(q, state["cross_decode.conv2d.weight"], padding=1)
    b, _, hq, wq = q.shape
    _, _, hk, wk = k.shape
    q = rms_norm(flatten(q), state["cross_decode.norm_q.weight"])
    k = rms_norm(flatten(k), state["cross_decode.norm_k.weight"])
    q = F.conv2d(
        unflatten(q, hq, wq),
        state["cross_decode.cross_attn.q_proj.weight"],
        state["cross_decode.cross_attn.q_proj.bias"],
    )
    k = F.conv2d(
        unflatten(k, hk, wk),
        state["cross_decode.cross_attn.k_proj.weight"],
        state["cross_decode.cross_attn.k_proj.bias"],
    )
    if mode == "efficient":
        return efficient_local_attention(q, k, v, q_chunk_size)
    q = flatten(q)
    k = flatten(k)
    v = flatten(v)
    features = upstream_masked_attention(q, k, v, q_chunk_size)
    return unflatten(features, hq, wq)


def forward(image, features, state, q_chunk_size=None, mode="efficient"):
    enc = conv_encoder(image, state, "image_encoder")
    b, c, h, w = enc.shape
    enc = rope(flatten(enc), state["rope.freqs"])
    enc = unflatten(enc, h, w)
    q = F.adaptive_avg_pool2d(conv_encoder(enc, state, "query_encoder"), image.shape[-2:])
    k_img = F.adaptive_avg_pool2d(conv_encoder(enc, state, "key_encoder"), features.shape[-2:])
    k_feat = feature_encoder(features, state)
    k = conv_encoder(torch.cat([k_img, k_feat], dim=1), state, "aggregation")
    return cross_decode(q, k, features, state, q_chunk_size=q_chunk_size, mode=mode)


def fused_state_dict(state):
    fused = {}
    for key, value in state.items():
        if key.startswith("cross_decode.cross_attn.q_proj") or key.startswith("cross_decode.cross_attn.k_proj"):
            continue
        key = key.replace("cross_decode.norm_q.weight", "cross_decode.cross_attn.norm_q.weight")
        key = key.replace("cross_decode.norm_k.weight", "cross_decode.cross_attn.norm_k.weight")
        fused[key] = value
    fused["cross_decode.cross_attn.attention.in_proj_weight"] = torch.cat(
        [
            state["cross_decode.cross_attn.q_proj.weight"].flatten(1),
            state["cross_decode.cross_attn.k_proj.weight"].flatten(1),
            randn(QK, QK),
        ],
        dim=0,
    )
    fused["cross_decode.cross_attn.attention.in_proj_bias"] = torch.cat(
        [
            state["cross_decode.cross_attn.q_proj.bias"],
            state["cross_decode.cross_attn.k_proj.bias"],
            randn(QK),
        ],
        dim=0,
    )
    return fused


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: anyup_tiny_parity.py <out-dir>")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    state = add_state()
    image = torch.linspace(-1.0, 1.0, 1 * 3 * 8 * 8, dtype=torch.float32).view(1, 3, 8, 8)
    features = torch.linspace(-0.5, 0.5, 1 * 5 * 2 * 2, dtype=torch.float32).view(1, 5, 2, 2)
    output = forward(image, features, state, q_chunk_size=None)
    chunked = forward(image, features, state, q_chunk_size=1)
    paper_output = forward(image, features, state, q_chunk_size=None, mode="paper")
    paper_chunked = forward(image, features, state, q_chunk_size=1, mode="paper")
    save_file(state, str(out / "efficient.safetensors"))
    fused = fused_state_dict(state)
    save_file(fused, str(out / "paper_fused.safetensors"))
    torch.save(fused, out / "paper_fused.pth")
    (out / "fixture.json").write_text(
        json.dumps(
            {
                "image": image.flatten().tolist(),
                "image_shape": list(image.shape),
                "features": features.flatten().tolist(),
                "features_shape": list(features.shape),
                "output": output.flatten().tolist(),
                "chunked_output": chunked.flatten().tolist(),
                "paper_output": paper_output.flatten().tolist(),
                "paper_chunked_output": paper_chunked.flatten().tolist(),
                "output_shape": list(output.shape),
            }
        )
    )


if __name__ == "__main__":
    main()
