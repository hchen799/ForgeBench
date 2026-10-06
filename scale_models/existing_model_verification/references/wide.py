"""Independent AP_TRN/AP_WRAP PyTorch reference for Q32.10 production v2.

The emitter uses an 80-bit accumulator with 44 fractional bits. Python integer
intermediates prevent int64 overflow; large reductions use exact FP64 GEMMs on
16-bit integer limbs, never a rounded FP64 result as the fixed-point oracle.
"""
import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from .arithmetic import Format, wrap
from .llama import FP64Ops, LlamaReference
from .resnet import (
    forward as resnet_forward,
    depth_from_variant,
    FP64Ops as ResNetFP64,
)
from .vendor_math import VendorRope

DATA = Format(32, 10)
FRAC = 44
SCALE = 1 << FRAC
EPSILON = math.floor(1e-5 * SCALE)


def objects(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=object)


def tensor(x):
    return torch.from_numpy(np.asarray(x, dtype=np.int64))


def cast(x, shift=0):
    return tensor((((objects(x) >> shift) + (1 << 31)) % (1 << 32)) - (1 << 31))


def sqrt_nearest(x):
    array = objects(x)
    result = []
    for value in array.flat:
        if value < 0:
            raise ValueError("negative square root")
        n = int(value) << FRAC
        r = math.isqrt(n)
        result.append(r + int(n - r * r > r))
    return np.asarray(result, dtype=object).reshape(array.shape)


def exact_reduction(a, b, operation):
    """Four GEMMs/convolutions of integer limbs; every sum is below 2**53.

    Signed high limbs and unsigned low limbs reconstruct the exact integer
    result, even when that result requires more than 64 bits.
    """
    al, ah = a & 65535, a >> 16
    bl, bh = b & 65535, b >> 16
    result = None
    for left, right, shift in ((ah, bh, 32), (ah, bl, 16), (al, bh, 16), (al, bl, 0)):
        sums = operation(left.double(), right.double()).long()
        term = objects(sums) << shift
        result = term if result is None else result + term
    return result


def dot(a, b):
    if a.shape[-1] * 65535**2 >= 2**53:
        raise ValueError("limb GEMM exceeds exact integer reduction bound")
    return exact_reduction(a, b, lambda x, y: x @ y)


@lru_cache(maxsize=1)
def exp_table():
    return np.asarray(
        [math.floor(math.exp(-i / 256) * SCALE + 0.5) for i in range(4097)],
        dtype=object,
    )


def exp_negative(x):
    x = objects(x)
    coordinate = -np.clip(x, -16 * SCALE, 0) * 256
    index = np.asarray(coordinate >> FRAC, dtype=np.int64).clip(0, 4095)
    fraction = coordinate - (index.astype(object) << FRAC)
    table = exp_table()
    value = (table[index] * (SCALE - fraction) + table[index + 1] * fraction) >> FRAC
    return np.where(x <= -16 * SCALE, 0, value)


class ResNetOps:
    def conv(self, x, w, stride=1, padding=0):
        if math.prod(w.shape[1:]) * 65535**2 >= 2**53:
            raise ValueError("limb convolution exceeds exact integer reduction bound")
        sums = exact_reduction(
            x, w, lambda a, b: F.conv2d(a[None], b, stride=stride, padding=padding)[0]
        )
        return cast(sums, 22)

    def bn(self, x, p):
        gamma, beta, mean, var = [objects(v[:, None, None]) for v in p]
        denom = sqrt_nearest((var << 22) + EPSILON)
        norm = ((objects(x) - mean) << 66) // denom
        return cast((gamma * norm >> FRAC) + beta)

    def add(self, a, b):
        return wrap(a + b, 32)

    def gap(self, x):
        return cast(objects(x).sum((1, 2)) // (x.shape[1] * x.shape[2]))

    def fc(self, x, w):
        return cast(dot(w, x[:, None])[:, 0], 22)


class Rope32(VendorRope):
    @lru_cache(maxsize=8)
    def coefficients(self, start, rows, head_dim=128):
        values = np.empty((rows, head_dim // 2, 2), dtype=np.float64)
        frequency = [
            np.float32(self.pow(500000.0, -d / head_dim)) for d in range(0, head_dim, 2)
        ]
        for r in range(rows):
            for d, freq in enumerate(frequency):
                angle = float(np.float32(start + r) * freq)
                values[r, d] = self.cos(angle), self.sin(angle)
        return DATA.quantize(values)

    def provenance(self):
        return dict(super().provenance(), coefficient_format=DATA.describe())


class LlamaOps:
    def __init__(self, tile_in=128, hidden_chunk=128, vendor_root=None):
        self.tile_in, self.hidden_chunk, self.vendor_root = (
            tile_in,
            hidden_chunk,
            vendor_root,
        )
        self.rope_provider = None

    def linear(self, x, w):
        return cast(dot(x, w.T), 22)

    def rmsnorm(self, x, gamma):
        a = objects(x)
        mean = (a * a).sum(-1) // x.shape[1] + EPSILON
        inv = (1 << 88) // sqrt_nearest(mean)
        return cast(a * objects(gamma) * inv[:, None], 66)

    def rope(self, x, start, heads, head_dim=128):
        if self.rope_provider is None:
            self.rope_provider = Rope32(self.vendor_root)
        coefficients = self.rope_provider.coefficients(start, len(x), head_dim)
        c, s = coefficients[:, :, 0][:, None], coefficients[:, :, 1][:, None]
        pairs = x.reshape(len(x), heads, head_dim // 2, 2)
        a, b = pairs[..., 0], pairs[..., 1]
        return torch.stack(
            (
                cast(objects(a) * objects(c) - objects(b) * objects(s), 22),
                cast(objects(a) * objects(s) + objects(b) * objects(c), 22),
            ),
            -1,
        ).reshape_as(x)

    def attention(self, q, k, v, start):
        heads, kv_heads = q.shape[1] // 128, k.shape[1] // 128
        scale = math.floor(SCALE / math.sqrt(128))
        outputs = []
        for h in range(heads):
            kh = h // (heads // kv_heads)
            scores = (
                dot(q[:, h * 128 : (h + 1) * 128], k[:, kh * 128 : (kh + 1) * 128].T)
                * scale
            ) >> FRAC
            rows = []
            for r in range(len(q)):
                visible = start + r + 1
                local = scores[r, :visible]
                weights = exp_negative(local - local.max())
                values = objects(v[:visible, kh * 128 : (kh + 1) * 128])
                context = ((weights[:, None] * values) >> 22).sum(0)
                rows.append(cast((context << FRAC) // weights.sum(), 22))
            outputs.append(torch.stack(rows))
        return torch.cat(outputs, 1)

    def swiglu(self, gate, up):
        x = objects(gate) << 22
        e = exp_negative(-np.abs(x))
        numerator = np.where(x < 0, (x * e) >> FRAC, x)
        silu = cast((numerator << FRAC) // (SCALE + e), 22)
        return cast(objects(silu) * objects(up), 22)

    def add(self, a, b):
        return wrap(a + b, 32)


class ResNetReference32:
    def __init__(self, variant, fixed, version=2):
        if version != 2:
            raise ValueError("Q32.10 requires the production v2 contract")
        self.fixed = fixed
        self.depth = depth_from_variant(variant)
        self.ops = ResNetOps() if fixed else ResNetFP64()

    def run(self, tensors, checkpoint):
        values = {
            n: (v.long() if self.fixed else v.double() / DATA.scale)
            for n, v in tensors.items()
        }
        return resnet_forward(values, self.ops, checkpoint, self.depth)


class LlamaReference32(LlamaReference):
    def __init__(self, project, weights, fixed):
        super().__init__(project, weights, fixed)
        self.ops = (
            LlamaOps(
                project.tile_in,
                project.hidden_chunk,
                getattr(project, "vendor_root", None),
            )
            if fixed
            else FP64Ops()
        )

    def weight(self, name, layer=None):
        w = self.weights.get("DRAM_" + name, layer)
        return w if self.fixed else w.double() / DATA.scale
