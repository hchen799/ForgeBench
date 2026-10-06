"""Independent integer-code PyTorch operators for production arithmetic v2.

Q16.5 storage remains AP_TRN/AP_WRAP. Reductions persist at Q64.42 (F=22);
division floors, sqrt rounds nearest, and exp uses a mathematical ROM grid.
No generated C++ output or dedicated verification kernel is imported.
"""
import math

import torch
import torch.nn.functional as F

from .arithmetic import DATA, exact_matmul, sqrt_codes, wrap
from .llama import FixedOps as LegacyLlamaOps

FRAC = 22
SCALE = 1 << FRAC
EPSILON = math.floor(1e-5 * SCALE)


def divide(a, b):
    if torch.any(torch.as_tensor(b) == 0):
        raise ValueError("production v2 division by zero")
    return torch.div(a, b, rounding_mode="floor")


def exp_negative(x):
    table = torch.tensor(
        [math.floor(math.exp(-i / 256) * SCALE + 0.5) for i in range(4097)],
        dtype=torch.int64,
    )
    coordinate = -x.clamp(-16 * SCALE, 0) * 256
    index = (coordinate >> FRAC).clamp_max(4095)
    fraction = coordinate - (index << FRAC)
    values = (table[index] * (SCALE - fraction) + table[index + 1] * fraction) >> FRAC
    return torch.where(x <= -16 * SCALE, 0, values)


class ResNetOps:
    def conv(self, x, w, stride=1, padding=0):
        # Integer operands/products/reductions stay below 2**53, so FP64
        # convolution is an exact integer reduction, not a floating reference.
        bound = int(x.abs().max()) * int(w.abs().max()) * math.prod(w.shape[1:])
        if bound >= 2**53:
            raise ValueError("convolution exceeds exact FP64 integer bound")
        sums = F.conv2d(x[None].double(), w.double(), stride=stride, padding=padding)[0]
        return wrap(sums.long() >> 11)

    def bn(self, x, p):
        gamma, beta, mean, var = [v[:, None, None] for v in p]
        denom = sqrt_codes((var << 11) + EPSILON, FRAC)
        norm = divide((x - mean) << 33, denom)
        return wrap((gamma * norm >> 22) + beta)

    def add(self, a, b):
        return wrap(a + b)

    def gap(self, x):
        return wrap(divide(x.sum((1, 2)), x.shape[1] * x.shape[2]))

    def fc(self, x, w):
        return wrap(exact_matmul(w, x[:, None])[:, 0] >> 11)


class LlamaOps(LegacyLlamaOps):
    def linear(self, x, w):
        return wrap(exact_matmul(x, w.T) >> 11)

    def rmsnorm(self, x, gamma):
        mean = divide((x * x).sum(-1), x.shape[1]) + EPSILON
        denominator = sqrt_codes(mean, FRAC)
        inverse = divide(torch.full_like(denominator, 1 << 44), denominator)
        return wrap((x * gamma * inverse[:, None]) >> 33)

    def attention(self, q, k, v, start):
        heads, kv_heads = q.shape[1] // 128, k.shape[1] // 128
        scale = math.floor(SCALE / math.sqrt(128))
        outputs = []
        for h in range(heads):
            kh = h // (heads // kv_heads)
            scores = (
                exact_matmul(
                    q[:, h * 128 : (h + 1) * 128], k[:, kh * 128 : (kh + 1) * 128].T
                )
                * scale
            ) >> FRAC
            rows = []
            for r in range(len(q)):
                visible = start + r + 1
                local = scores[r, :visible]
                weights = exp_negative(local - local.max())
                values = v[:visible, kh * 128 : (kh + 1) * 128]
                context = ((weights[:, None] * values) >> 11).sum(0)
                rows.append(wrap(divide(context << FRAC, weights.sum()) >> 11))
            outputs.append(torch.stack(rows))
        return torch.cat(outputs, 1)

    def swiglu(self, gate, up):
        x = gate << 11
        e = exp_negative(-x.abs())
        numerator = torch.where(x < 0, (x * e) >> FRAC, x)
        silu = wrap(divide(numerator << FRAC, SCALE + e) >> 11)
        return wrap((silu * up) >> 11)
