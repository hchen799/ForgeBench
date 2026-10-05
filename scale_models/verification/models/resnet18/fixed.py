"""Vitis 2024.1 arithmetic for the current tiled emitter, including known defects.

Public tensors contain configured signed fixed-point codes, not real values.
Accumulator codes retain twice the storage fractional bits. Wide software temporaries do not change the modeled
hardware widths. The floating reference lives separately in model.py.
"""
import math
from collections import Counter

import torch
import numpy as np
from .config import Config
from ...precision import FixedFormat, objects, clamp_integer, exact_dot, round_shift
import torch.nn.functional as F

def quantize(x, precision=None):
    if not torch.isfinite(x).all():
        raise ValueError("cannot quantize non-finite values")
    return (precision or FixedFormat()).quantize(x)


def trunc_div(x, denominator):
    if isinstance(x, torch.Tensor):
        return torch.div(x, denominator, rounding_mode="trunc")
    n, d = objects(x), objects(denominator)
    return np.where(n >= 0, n // d, -((-n) // d))


def sqrt_codes(variance, precision=None):
    """Positive variance square root rounded to twice the storage fraction bits.

    Matches the generated wide integer helper, including a minimum one-code
    epsilon. At <16,5> this also matches the installed vendor square root.
    """
    if (variance < 0).any():
        raise ValueError("negative BN variance")
    p = precision or FixedFormat()
    f = p.fractional_bits
    epsilon = max(1, math.floor(1e-5 * (1 << (2 * f)) + .5))
    values = []
    for v in variance.reshape(-1).tolist():
        radicand = (v * p.scale + epsilon) << (2 * f)
        root = math.isqrt(radicand)
        if 4 * radicand >= (2 * root + 1) ** 2:
            root += 1
        values.append(root)
    return torch.tensor(values, dtype=torch.int64).reshape(variance.shape)


class FixedOps:
    def __init__(self, guard=256, fast=True, config=None):
        self.c = (config or Config(guard=guard)).validate()
        self.p = self.c.precision
        self.acc_scale = 1 << (2 * self.p.fractional_bits)
        self.acc_min = -(1 << (self.c.acc_word_bits - 1))
        self.acc_max = (1 << (self.c.acc_word_bits - 1)) - 1
        self.guard = self.c.guard * self.acc_scale
        self.fast = fast
        self.name = ""
        self.events = []
        self.stats = {}

    def count(self, key, count):
        self.stats.setdefault(self.name, Counter())[key] += int(count)

    def clamp(self, x, low, high, key):
        self.count(key, int(np.count_nonzero((x < low) | (x > high))))
        return clamp_integer(x, low, high)

    def acc(self, x):
        return self.clamp(x, self.acc_min, self.acc_max, "accumulator_saturations")

    def data(self, x):
        return self.clamp(x, self.p.minimum, self.p.maximum, "storage_saturations")

    def shift(self, x, shift):
        for _ in range(max(shift, 0)):
            x = self.acc(objects(x) * 2 if self.c.acc_word_bits == 64 else x * 2)
        for _ in range(max(-shift, 0)):
            # Division retains the accumulator fraction, truncated towards zero.
            x = trunc_div(x, 2)
        return x

    def commit(self, x, exp=0, output_shift=0):
        return self.data(round_shift(self.shift(x, exp - output_shift), self.p.fractional_bits))

    def renorm(self, x, exp):
        while min(self.acc_max, max(abs(int(x.min())), abs(int(x.max())))) > self.guard:
            x = trunc_div(x, 2)
            exp += 1
            self.count("renormalizations", 1)
        return x, exp

    def reduction(self, acc, weights, patches):
        if self.fast:
            if self.p.word_bits <= 16:
                bound = weights.abs() @ patches.abs()
                safe = bool((acc.double().abs() + bound <= self.acc_max).all())
            else:
                bound = int(weights.abs().max()) * int(patches.abs().max()) * weights.shape[1]
                safe = max(abs(int(acc.min())), abs(int(acc.max()))) + bound <= self.acc_max
            if safe:
                self.count("fast_chunks", 1)
                product = exact_dot(weights, patches)
                value = acc + product if isinstance(product, torch.Tensor) else objects(acc) + product
                return clamp_integer(value, self.acc_min, self.acc_max)
        self.count("sequential_chunks", 1)
        w, p = weights.long(), patches.long()
        for k in range(w.shape[1]):
            if self.c.acc_word_bits == 64:
                value = objects(acc) + objects(w[:, k:k + 1]) * objects(p[k:k + 1, :])
            else:
                value = acc + w[:, k:k + 1] * p[k:k + 1, :]
            acc = self.acc(value)
        return acc

    def conv(self, x, weights, stride=1, padding=0, output_shift=0):
        co_total, ci_total, kernel, _ = weights.shape
        h = (x.shape[1] + 2 * padding - kernel) // stride + 1
        w = (x.shape[2] + 2 * padding - kernel) // stride + 1
        patches = F.unfold(x.to(torch.float64).unsqueeze(0), kernel, padding=padding, stride=stride)[0]
        wf = weights.to(torch.float64).reshape(co_total, -1)
        out = torch.empty((co_total, h, w), dtype=torch.int64)
        for co in range(0, co_total, self.c.tile_c):
            nc = min(self.c.tile_c, co_total - co)
            for row in range(0, h, self.c.tile_h):
                nh = min(self.c.tile_h, h - row)
                for col in range(0, w, self.c.tile_w):
                    nw = min(self.c.tile_w, w - col)
                    indices = (torch.arange(row, row + nh)[:, None] * w + torch.arange(col, col + nw)).flatten()
                    tile = patches[:, indices]
                    acc = torch.zeros((nc, nh * nw), dtype=torch.int64)
                    exp = 0
                    for ci in range(0, ci_total, self.c.tile_c):
                        start, stop = ci * kernel * kernel, min(ci + self.c.tile_c, ci_total) * kernel * kernel
                        # Deliberately reproduce the emitter: incoming products
                        # are NOT scaled when exp is nonzero (known defect).
                        acc = self.reduction(acc, wf[co:co + nc, start:stop], tile[start:stop])
                        acc, exp = self.renorm(acc, exp)
                        self.events.append([self.name, co, row, col, ci, exp])
                    out[co:co + nc, row:row + nh, col:col + nw] = self.commit(acc, exp, output_shift).reshape(nc, nh, nw)
        return out

    def bn(self, x, params):
        gamma, beta, mean, variance = params
        f = self.p.fractional_bits
        denominator = sqrt_codes(variance, self.p)[:, None, None]
        if self.p.word_bits <= 16 and f <= 11:
            numerator = (x - mean[:, None, None]) * (1 << (3 * f))
            norm = self.acc(trunc_div(numerator, denominator))
            affine = gamma[:, None, None] * norm + beta[:, None, None] * (1 << (2 * f))
        else:
            numerator = objects(x - mean[:, None, None]) * (1 << (3 * f))
            norm = self.acc(trunc_div(numerator, denominator))
            affine = objects(gamma[:, None, None]) * objects(norm) + objects(beta[:, None, None]) * (1 << (2 * f))
        return self.data(round_shift(affine, 2 * f))

    def relu(self, x):
        return x.clamp_min(0)

    def pool(self, x):
        # Explicit zero padding matters for negative-valued operator tests.
        return F.max_pool2d(F.pad(x.to(torch.float64).unsqueeze(0), (1, 1, 1, 1)), 3, 2)[0].to(torch.int64)

    def add(self, x, skip):
        return self.data(x + skip)

    def gap(self, x):
        channels, h, w = x.shape
        out = torch.empty(channels, dtype=torch.int64)
        for co in range(0, channels, self.c.tile_c):
            tile = x[co:co + self.c.tile_c].reshape(-1, h * w) * self.p.scale
            acc = torch.zeros(tile.shape[0], dtype=torch.int64)
            for pos in range(h * w):
                acc = self.acc(acc + tile[:, pos])
            acc, exp = self.renorm(acc, 0)
            self.events.append([self.name, co, 0, 0, -1, exp])
            value = trunc_div(self.shift(acc, exp), h * w)
            out[co:co + tile.shape[0]] = self.commit(value)
        return out

    def fc(self, x, weights, output_shift=0):
        out = torch.empty(weights.shape[0], dtype=torch.int64)
        wf = weights.to(torch.float64)
        patches = x.to(torch.float64)[:, None]
        for co in range(0, weights.shape[0], self.c.tile_c):
            nc = min(self.c.tile_c, weights.shape[0] - co)
            acc = torch.zeros((nc, 1), dtype=torch.int64)
            exp = 0
            for ci in range(0, x.numel(), self.c.tile_c):
                acc = self.reduction(acc, wf[co:co + nc, ci:ci + self.c.tile_c], patches[ci:ci + self.c.tile_c])
                acc, exp = self.renorm(acc, exp)
                self.events.append([self.name, co, 0, 0, ci, exp])
            out[co:co + nc] = self.commit(acc, exp, output_shift).flatten()
        return out
