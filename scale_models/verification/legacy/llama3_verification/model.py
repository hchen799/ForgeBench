"""PyTorch mathematical and exact scaled-integer Llama 3 references."""
import math
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from .codegen import tables


def round_shift(value, bits):
    return torch.div(value + (1 << (bits - 1)), 1 << bits, rounding_mode="floor")


def round_ratio(numerator, denominator):
    quotient = torch.div(numerator, denominator, rounding_mode="floor")
    remainder = numerator - quotient * denominator
    return quotient + (remainder * 2 >= denominator).to(torch.int64)


class FixedOps:
    def __init__(self, config, rope):
        self.c = config
        self.rope_table = torch.from_numpy(np.array(rope, dtype=np.int64, copy=True))
        exp, silu = tables()
        self.exp_table = torch.tensor(exp, dtype=torch.int64)
        self.silu_table = torch.tensor(silu, dtype=torch.int64)
        self.saturations = Counter()
        self.name = ""

    def commit(self, value):
        self.saturations[self.name] += int(((value < -32768) | (value > 32767)).sum())
        return value.clamp(-32768, 32767)

    def linear(self, x, weight):
        # Signed 16-bit products summed over <=32768 columns have absolute
        # sum <=2^45, so all FP64 products/prefix sums are exact integers.
        result = (x.double() @ weight.T).to(torch.int64)
        return self.commit(round_shift(result, 11))

    def rmsnorm(self, x, gamma):
        sums = (x * x).sum(-1)
        means = torch.div(sums, self.c.hidden, rounding_mode="floor") + 42
        inverse = []
        for mean in means.tolist():
            value = mean << 22
            root = math.isqrt(value)
            if value - root * root > root:
                root += 1
            inverse.append((1 << 44) // root)
        inverse = torch.tensor(inverse, dtype=torch.int64)
        return self.commit(round_shift(x * gamma.to(torch.int64) * inverse[:, None], 33))

    def rope(self, x, start, heads):
        rows = x.shape[0]
        paired = x.reshape(rows, heads, self.c.head_dim // 2, 2)
        table = self.rope_table[start:start + rows].reshape(rows, 1, self.c.head_dim // 2, 2)
        cosine, sine = table[..., 0], table[..., 1]
        a, b = paired[..., 0], paired[..., 1]
        output = torch.stack((a * cosine - b * sine, a * sine + b * cosine), dim=-1)
        return self.commit(round_shift(output, 11)).reshape(rows, -1)

    def attention(self, q, keys, values, start):
        c, rows = self.c, q.shape[0]
        k = keys.reshape(-1, c.kv_heads, c.head_dim)
        v = values.reshape(-1, c.kv_heads, c.head_dim)
        queries = q.reshape(rows, c.q_heads, c.head_dim)
        outputs = torch.empty_like(queries)
        scale = math.floor((1 << 22) / math.sqrt(c.head_dim) + .5)
        # Process a whole query-head matrix at once. Keys are repeated logically
        # through indexing, rather than materializing a full repeated cache.
        for head in range(c.q_heads):
            kh = head // (c.q_heads // c.kv_heads)
            scores = (queries[:, head].double() @ k[:, kh].double().T).to(torch.int64)
            scores = round_shift(scores * scale, 33)
            for row in range(rows):
                visible = start + row + 1
                local = self.commit(scores[row, :visible])
                probabilities = self.exp_table[local.max() - local]
                denominator = probabilities.sum()
                # Bound: 8192 * 2^22 * 32768 = 2^50, still exact in FP64.
                numerator = (probabilities.double() @ v[:visible, kh].double()).to(torch.int64)
                outputs[row, head] = self.commit(round_ratio(numerator, denominator))
        return outputs.reshape(rows, c.hidden)

    def add(self, x, y):
        return self.commit(x + y)

    def silu(self, x):
        return self.silu_table[x + 32768]

    def multiply(self, x, y):
        return self.commit(round_shift(x * y, 11))


class FloatOps:
    def __init__(self, config):
        self.c = config
        self.name = ""

    def linear(self, x, weight):
        return (x @ weight.T) / 2048.0

    def rmsnorm(self, x, gamma):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.c.epsilon) * (gamma / 2048.0)

    def rope(self, x, start, heads):
        rows = x.shape[0]
        frequencies = self.c.rope_theta ** (-torch.arange(0, self.c.head_dim, 2, dtype=torch.float64) / self.c.head_dim)
        angles = torch.arange(start, start + rows, dtype=torch.float64)[:, None] * frequencies[None]
        cosine, sine = angles.cos()[:, None], angles.sin()[:, None]
        paired = x.reshape(rows, heads, self.c.head_dim // 2, 2)
        a, b = paired[..., 0], paired[..., 1]
        return torch.stack((a * cosine - b * sine, a * sine + b * cosine), dim=-1).reshape(rows, -1)

    def attention(self, q, keys, values, start):
        c, rows = self.c, q.shape[0]
        queries = q.reshape(rows, c.q_heads, c.head_dim).transpose(0, 1)
        k = keys.reshape(-1, c.kv_heads, c.head_dim).repeat_interleave(c.q_heads // c.kv_heads, dim=1).transpose(0, 1)
        v = values.reshape(-1, c.kv_heads, c.head_dim).repeat_interleave(c.q_heads // c.kv_heads, dim=1).transpose(0, 1)
        scores = queries @ k.transpose(-1, -2) / math.sqrt(c.head_dim)
        mask = torch.arange(keys.shape[0])[None, :] > torch.arange(start, start + rows)[:, None]
        probabilities = torch.softmax(scores.masked_fill(mask[None], float("-inf")), dim=-1)
        return (probabilities @ v).transpose(0, 1).reshape(rows, c.hidden)

    def add(self, x, y):
        return x + y

    def silu(self, x):
        return F.silu(x)

    def multiply(self, x, y):
        return x * y


class LlamaReference:
    """Single-batch cached transformer, with independent caches for each oracle."""
    def __init__(self, config, weights, fixed=False, rope=None):
        self.c, self.weights, self.fixed = config, weights, fixed
        self.ops = FixedOps(config, rope) if fixed else FloatOps(config)
        dtype = torch.int16 if fixed else torch.float64
        self.keys = torch.zeros((config.layers, config.max_ctx, config.kv_dim), dtype=dtype)
        self.values = torch.zeros_like(self.keys)
        self.position = 0

    @torch.inference_mode()
    def forward(self, tokens, start, call, checkpoint):
        c, ops = self.c, self.ops
        rows = len(tokens)
        if start != self.position or not 1 <= rows <= c.prefill_tile or start + rows > c.max_ctx:
            raise ValueError("invalid sequence/cache position")
        if np.any(tokens < 0) or np.any(tokens >= c.vocab):
            raise ValueError("token ID outside vocabulary")
        prefix = f"call{call:04d}."

        def emit(name, value, essential=False):
            if not torch.isfinite(value).all():
                raise ValueError(f"non-finite tensor: {name}")
            checkpoint(prefix + name, value, essential)
            return value

        def operation(name, method, *args, essential=False):
            ops.name = prefix + name
            return emit(name, getattr(ops, method)(*args), essential)

        x = self.weights.embedding(tokens)
        if not self.fixed:
            x = x.double() / 2048.0
        emit("embedding", x)
        for layer in range(c.layers):
            name = f"layer{layer:02d}."
            weight_name = f"layers.{layer}."
            weight = lambda suffix: self.weights.tensor(weight_name + suffix)
            norm = operation(name + "attn_norm", "rmsnorm", x, weight("attn_norm"))
            q = operation(name + "q_proj", "linear", norm, weight("q_proj"))
            k = operation(name + "k_proj", "linear", norm, weight("k_proj"))
            v = operation(name + "v_proj", "linear", norm, weight("v_proj"))
            q = operation(name + "q_rope", "rope", q, start, c.q_heads)
            k = operation(name + "k_rope", "rope", k, start, c.kv_heads)
            self.keys[layer, start:start + rows] = k
            self.values[layer, start:start + rows] = v
            emit(name + "k_cache_update", self.keys[layer, start:start + rows].to(k.dtype), True)
            emit(name + "v_cache_update", self.values[layer, start:start + rows].to(v.dtype), True)
            context = operation(name + "context", "attention", q, self.keys[layer, :start + rows], self.values[layer, :start + rows], start)
            projected = operation(name + "o_proj", "linear", context, weight("o_proj"))
            mid = operation(name + "attn_residual", "add", x, projected)
            norm = operation(name + "ffn_norm", "rmsnorm", mid, weight("ffn_norm"))
            gate = operation(name + "gate_proj", "linear", norm, weight("gate_proj"))
            up = operation(name + "up_proj", "linear", norm, weight("up_proj"))
            gate = operation(name + "silu", "silu", gate)
            ffn = operation(name + "swiglu", "multiply", gate, up)
            down = operation(name + "down_proj", "linear", ffn, weight("down_proj"))
            x = operation(name + "out", "add", mid, down, essential=True)
        norm = operation("final_norm", "rmsnorm", x, self.weights.tensor("final_norm"), essential=True)
        logits = operation("logits", "linear", norm, self.weights.tensor("lm_head"), essential=True)
        self.position += rows
        return logits
