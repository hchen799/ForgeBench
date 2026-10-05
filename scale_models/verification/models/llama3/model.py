"""PyTorch mathematical and exact scaled-integer Llama 3 references."""
import math
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from .codegen import tables
from ...precision import objects, clamp_integer, round_shift, exact_dot, rounded_ratio


class FixedOps:
    def __init__(self, config, rope):
        self.c, self.p = config, config.precision
        self.rope_table = torch.from_numpy(np.array(rope, dtype=np.int64, copy=True))
        exp, silu = tables(config)
        self.exp_table = torch.tensor(exp, dtype=torch.int64)
        self.silu_table = torch.tensor(silu, dtype=torch.int64)
        self.saturations = Counter()
        self.name = ""

    def commit(self, value):
        bad = (value < self.p.minimum) | (value > self.p.maximum)
        self.saturations[self.name] += int(np.count_nonzero(bad))
        return clamp_integer(value, self.p.minimum, self.p.maximum)

    def dot(self, a, b):
        if self.c.word_bits <= 16:
            return (a.double() @ b.double()).long()
        return exact_dot(a, b)

    def linear(self, x, weight):
        return self.commit(round_shift(self.dot(x, weight.T), self.c.frac))

    def rmsnorm(self, x, gamma):
        f = self.c.frac
        epsilon = max(1, math.floor(self.c.epsilon * (1 << (2 * f)) + .5))
        inverse = []
        for row in x.tolist():
            mean = sum(int(v) ** 2 for v in row) // self.c.hidden + epsilon
            value = mean << (2 * f)
            root = math.isqrt(value)
            root += value - root * root > root
            inverse.append((1 << (4 * f)) // root)
        if self.c.word_bits <= 16 and f <= 11:
            product = x * gamma.long() * torch.tensor(inverse, dtype=torch.int64)[:, None]
        else:
            product = objects(x) * objects(gamma.long()) * np.array(inverse, dtype=object)[:, None]
        return self.commit(round_shift(product, 3 * f))

    def rope(self, x, start, heads):
        rows = x.shape[0]
        paired = x.reshape(rows, heads, self.c.head_dim // 2, 2)
        table = self.rope_table[start:start + rows].reshape(rows, 1, self.c.head_dim // 2, 2)
        a, b = paired[..., 0], paired[..., 1]
        cosine, sine = table[..., 0], table[..., 1]
        if self.c.word_bits <= 16:
            output = torch.stack((a * cosine - b * sine, a * sine + b * cosine), dim=-1)
        else:
            a, b, cosine, sine = map(objects, (a, b, cosine, sine))
            output = np.stack((a * cosine - b * sine, a * sine + b * cosine), axis=-1)
        return self.commit(round_shift(output, self.c.frac)).reshape(rows, -1)

    def lookup(self, x, table, offset=0):
        f = self.c.frac
        if f <= 11:
            return table[x * (1 << (11 - f)) + offset]
        unit = 1 << (f - 11)
        index = torch.div(x, unit, rounding_mode="floor")
        remainder = x - index * unit
        value = table[index + offset] * (unit - remainder) + table[index + offset + 1] * remainder
        return round_shift(value, f - 11)

    def exp(self, delta):
        bound = 32 * self.p.scale
        values = self.lookup(delta.clamp(0, bound - 1), self.exp_table)
        return torch.where(delta >= bound, 0, values)

    def attention(self, q, keys, values, start):
        c, rows = self.c, q.shape[0]
        k = keys.reshape(-1, c.kv_heads, c.head_dim)
        v = values.reshape(-1, c.kv_heads, c.head_dim)
        queries = q.reshape(rows, c.q_heads, c.head_dim)
        outputs = torch.empty_like(queries)
        scale = math.floor((1 << (2 * c.frac)) / math.sqrt(c.head_dim) + .5)
        for head in range(c.q_heads):
            kh = head // (c.q_heads // c.kv_heads)
            scores = self.dot(queries[:, head], k[:, kh].T)
            if c.word_bits > 16 or c.frac > 11:
                scores = objects(scores)
            scores = round_shift(scores * scale, 3 * c.frac)
            for row in range(rows):
                visible = start + row + 1
                local = self.commit(scores[row, :visible])
                probabilities = self.exp(local.max() - local)
                denominator = int(probabilities.sum())
                if c.word_bits <= 16 and c.exp_frac <= 22:
                    numerator = (probabilities.double() @ v[:visible, kh].double()).long()
                else:
                    numerator = exact_dot(probabilities[None], v[:visible, kh])[0]
                outputs[row, head] = self.commit(rounded_ratio(numerator, denominator))
        return outputs.reshape(rows, c.hidden)

    def add(self, x, y):
        return self.commit(x + y)

    def silu(self, x):
        bound = 32 * self.p.scale
        result = self.lookup(x.clamp(-bound, bound - 1), self.silu_table, 65536)
        return torch.where(x >= bound, x, torch.where(x <= -bound, 0, result))

    def multiply(self, x, y):
        # Individual signed-int32 products fit int64; shifts are overflow-safe.
        return self.commit(round_shift(x * y, self.c.frac))


class FloatOps:
    def __init__(self, config):
        self.c = config
        self.name = ""

    def linear(self, x, weight):
        return (x @ weight.T) / self.c.precision.scale

    def rmsnorm(self, x, gamma):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.c.epsilon) * (gamma / self.c.precision.scale)

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
        dtype = config.precision.torch_dtype if fixed else torch.float64
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
            x = x.double() / self.c.precision.scale
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
