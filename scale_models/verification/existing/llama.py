"""Independent FP64 and production fixed-point cached Llama PyTorch graphs.

The fixed reference models arithmetic, not known graph bugs (in-place o_proj,
finite causal masking). Those must fail comparison. Invalid constants in the
saved arithmetic raise ArithmeticFault rather than being silently corrected.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .arithmetic import (
    ACC,
    DATA,
    ArithmeticFault,
    exact_matmul,
    exp_codes,
    sqrt_codes,
    trunc_div,
    wrap,
)


class Weights:
    def __init__(self, folder, shapes):
        self.folder, self.shapes = Path(folder), shapes

    def get(self, name, layer=None):
        shape = self.shapes[name]
        array = np.memmap(
            self.folder / (name + ".bin"), dtype="<i2", mode="r", shape=shape
        )
        if layer is not None:
            array = array[layer]
        return torch.from_numpy(np.array(array, dtype=np.int64))


class FP64Ops:
    def linear(self, x, w):
        return x @ w.T

    def rmsnorm(self, x, gamma):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * gamma

    def rope(self, x, start, heads, head_dim=128):
        paired = x.reshape(x.shape[0], heads, head_dim // 2, 2)
        f = 500000.0 ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
        angles = torch.arange(start, start + len(x), dtype=torch.float64)[:, None] * f
        c, s = angles.cos()[:, None], angles.sin()[:, None]
        return torch.stack(
            (
                paired[..., 0] * c - paired[..., 1] * s,
                paired[..., 0] * s + paired[..., 1] * c,
            ),
            -1,
        ).reshape_as(x)

    def attention(self, q, k, v, start):
        head_dim = 128
        heads = q.shape[1] // head_dim
        kv_heads = k.shape[1] // head_dim
        q = q.reshape(len(q), heads, head_dim).transpose(0, 1)
        k = (
            k.reshape(len(k), kv_heads, head_dim)
            .repeat_interleave(heads // kv_heads, 1)
            .transpose(0, 1)
        )
        v = (
            v.reshape(len(v), kv_heads, head_dim)
            .repeat_interleave(heads // kv_heads, 1)
            .transpose(0, 1)
        )
        scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
        mask = (
            torch.arange(k.shape[1])[None]
            > torch.arange(start, start + q.shape[1])[:, None]
        )
        scores.masked_fill_(mask[None], -torch.inf)
        return (torch.softmax(scores, -1) @ v).transpose(0, 1).reshape(q.shape[1], -1)

    def add(self, a, b):
        return a + b

    def swiglu(self, gate, up):
        return F.silu(gate) * up


class FixedOps:
    def __init__(self, tile_in=128, hidden_chunk=128, vendor_root=None):
        self.tile_in, self.hidden_chunk = tile_in, hidden_chunk
        self.vendor_root = vendor_root
        self.rope_provider = None

    def linear(self, x, w):
        out = torch.zeros((len(x), len(w)), dtype=torch.int64)
        for i in range(0, x.shape[1], self.tile_in):
            # Products of data_t values converted to acc_t are exact at F=22.
            acc = wrap(
                (out << 11)
                + exact_matmul(
                    x[:, i : i + self.tile_in], w[:, i : i + self.tile_in].T
                ),
                32,
            )
            out = wrap(acc >> 11)
        return out

    def rmsnorm(self, x, gamma):
        divisor = int(ACC.quantize(x.shape[1]))
        if divisor == 0:
            raise ArithmeticFault(
                f"RMSNorm: (ap_fixed<32,10>){x.shape[1]} is zero; cannot divide"
            )
        sums = torch.zeros(len(x), dtype=torch.int64)
        for i in range(0, x.shape[1], self.hidden_chunk):
            v = x[:, i : i + self.hidden_chunk]
            acc = wrap((sums << 11) + (v * v).sum(-1), 32)
            sums = wrap(acc >> 11)
        mean = trunc_div(sums << 33, divisor) + int(ACC.quantize(1e-5))
        denominator = sqrt_codes(mean, 22)
        inverse = wrap(
            trunc_div(torch.full_like(denominator, 1 << 44), denominator) >> 11
        )
        return wrap((x * gamma * inverse[:, None]) >> 22)

    def rope(self, x, start, heads, head_dim=128):
        # Share only the opaque vendor primitive coefficients, never model
        # outputs. FP64 computes its own coefficients from real mathematics.
        if self.rope_provider is None:
            from .vendor_math import VendorRope

            self.rope_provider = VendorRope(self.vendor_root)
        table = self.rope_provider.coefficients(start, len(x), head_dim)
        c, s = table[:, :, 0][:, None], table[:, :, 1][:, None]
        paired = x.reshape(len(x), heads, head_dim // 2, 2)
        a, b = paired[..., 0], paired[..., 1]
        return wrap(
            torch.stack(((a * c - b * s) >> 11, (a * s + b * c) >> 11), -1)
        ).reshape_as(x)

    def attention(self, q, k, v, start):
        # Preserve the actual constant cast. At Q16.5, 128 wraps to zero.
        denom = sqrt_codes(DATA.quantize(128))
        if int(denom) == 0:
            raise ArithmeticFault("attention: hls::sqrt((ap_fixed<16,5>)128) is zero")
        scale = wrap(trunc_div(DATA.scale**2, denom))
        heads, kv_heads = q.shape[1] // 128, k.shape[1] // 128
        outputs = []
        for h in range(heads):
            kh = h // (heads // kv_heads)
            scores = wrap(
                (
                    wrap(
                        exact_matmul(
                            q[:, h * 128 : (h + 1) * 128],
                            k[:, kh * 128 : (kh + 1) * 128].T,
                        ),
                        32,
                    )
                    * scale
                )
                >> 22
            )
            rows = []
            for r in range(len(q)):
                visible = start + r + 1
                local = scores[r, :visible]
                weights = wrap(exp_codes(local - local.max()))
                denominator = wrap(weights.sum())
                values = v[:visible, kh * 128 : (kh + 1) * 128]
                context = wrap(((weights[:, None] * values) >> 11).sum(0))
                rows.append(wrap(trunc_div(context * 2048, denominator)))
            outputs.append(torch.stack(rows))
        return torch.cat(outputs, 1)

    def add(self, a, b):
        return wrap(a + b)

    def swiglu(self, gate, up):
        # Unary minus widens data_t to W=17,I=6; exp has that same type.
        denominator = exp_codes(-gate) + 2048
        silu = wrap(trunc_div(gate * 2048, denominator))
        return wrap(silu * up >> 11)


class LlamaReference:
    def __init__(self, project, weights, fixed):
        self.project, self.weights, self.fixed = project, weights, fixed
        self.ops = (
            FixedOps(
                project.tile_in,
                project.hidden_chunk,
                getattr(project, "vendor_root", None),
            )
            if fixed
            else FP64Ops()
        )
        shape = project.ports["DRAM_k_cache"]
        self.layers, _, self.kv_heads, self.head_dim = shape
        self.hidden = project.ports["DRAM_embedding"][1]
        self.q_heads = self.hidden // self.head_dim
        self.vocab = project.ports["DRAM_embedding"][0]
        self.keys = torch.zeros(shape, dtype=torch.int64 if fixed else torch.float64)
        self.values = torch.zeros_like(self.keys)
        self.position = 0

    def weight(self, name, layer=None):
        w = self.weights.get("DRAM_" + name, layer)
        return w if self.fixed else w.double() / 2048

    @torch.inference_mode()
    def forward(self, tokens, start, checkpoint):
        if (
            start != self.position
            or not len(tokens)
            or start + len(tokens) > self.project.max_ctx
        ):
            raise ValueError("invalid reference cache position")
        if any(t < 0 or t >= self.vocab for t in tokens):
            raise ValueError("token outside vocabulary")
        x = self.weight("embedding")[tokens]
        for layer in range(self.layers):
            norm = self.ops.rmsnorm(x, self.weight("attn_norm", layer))
            q = self.ops.linear(norm, self.weight("q_proj", layer))
            k = self.ops.linear(norm, self.weight("k_proj", layer))
            v = self.ops.linear(norm, self.weight("v_proj", layer))
            q = self.ops.rope(q, start, self.q_heads, self.head_dim)
            k = self.ops.rope(k, start, self.kv_heads, self.head_dim)
            self.keys[layer, start : start + len(x)] = k.reshape(
                len(x), self.kv_heads, self.head_dim
            )
            self.values[layer, start : start + len(x)] = v.reshape(
                len(x), self.kv_heads, self.head_dim
            )
            context = self.ops.attention(
                q,
                self.keys[layer, : start + len(x)].reshape(
                    -1, self.kv_heads * self.head_dim
                ),
                self.values[layer, : start + len(x)].reshape(
                    -1, self.kv_heads * self.head_dim
                ),
                start,
            )
            mid = self.ops.add(
                x, self.ops.linear(context, self.weight("o_proj", layer))
            )
            norm = self.ops.rmsnorm(mid, self.weight("ffn_norm", layer))
            gate = self.ops.linear(norm, self.weight("gate_proj", layer))
            up = self.ops.linear(norm, self.weight("up_proj", layer))
            x = self.ops.add(
                mid,
                self.ops.linear(
                    self.ops.swiglu(gate, up), self.weight("down_proj", layer)
                ),
            )
        norm = self.ops.rmsnorm(x, self.weight("final_norm"))
        logits = self.ops.linear(norm, self.weight("lm_head"))
        checkpoint("logits", logits)
        checkpoint("k_cache", self.keys[:, : start + len(tokens)])
        checkpoint("v_cache", self.values[:, : start + len(tokens)])
        self.position += len(tokens)
        return logits
