"""Vitis 2024.1 arithmetic for the current tiled emitter, including known defects.

Public tensors contain signed Q5.11 *codes*, not floating-point values. Accumulator
codes have 22 fractional bits. Wide software temporaries do not change the modeled
hardware widths. The floating reference lives separately in model.py.
"""
import math
from collections import Counter

import torch
import torch.nn.functional as F

DATA_SCALE = 1 << 11
ACC_SCALE = 1 << 22
DATA_MIN, DATA_MAX = -(1 << 15), (1 << 15) - 1
ACC_MIN, ACC_MAX = -(1 << 31), (1 << 31) - 1


def quantize(x):
    if not torch.isfinite(x).all():
        raise ValueError("cannot quantize non-finite values")
    return torch.floor(x.to(torch.float64) * DATA_SCALE + 0.5).clamp(DATA_MIN, DATA_MAX).to(torch.int64)


def round_shift(x, bits):
    """AP_RND assignment: nearest, ties towards positive infinity."""
    return torch.div(x + (1 << (bits - 1)), 1 << bits, rounding_mode="floor")


def trunc_div(x, denominator):
    return torch.div(x, denominator, rounding_mode="trunc")


def sqrt_codes(variance):
    """hls::sqrt(ap_fixed<33,11>) for nonnegative Q5.11 variance + Q10.22 eps.

    The installed bit-by-bit implementation rounds its positive result to 22
    fractional bits. Integer square roots and midpoint comparisons avoid host
    sqrt rounding differences.
    """
    if (variance < 0).any():
        raise ValueError("negative BN variance")
    values = []
    for v in variance.reshape(-1).tolist():
        radicand = (v * 2048 + 42) << 22  # Q10.22(1e-5) has code 42.
        root = math.isqrt(radicand)
        if 4 * radicand >= (2 * root + 1) ** 2:
            root += 1
        values.append(root)
    return torch.tensor(values, dtype=torch.int64).reshape(variance.shape)


class FixedOps:
    def __init__(self, guard=256, fast=True):
        if not 0 < guard < 512:
            raise ValueError("renormalization guard must be between 0 and 512")
        self.guard = guard * ACC_SCALE
        self.fast = fast
        self.name = ""
        self.events = []
        self.stats = {}

    def count(self, key, count):
        self.stats.setdefault(self.name, Counter())[key] += int(count)

    def clamp(self, x, low, high, key):
        self.count(key, ((x < low) | (x > high)).sum().item())
        return x.clamp(low, high)

    def acc(self, x):
        return self.clamp(x, ACC_MIN, ACC_MAX, "accumulator_saturations")

    def data(self, x):
        return self.clamp(x, DATA_MIN, DATA_MAX, "storage_saturations")

    def shift(self, x, shift):
        for _ in range(max(shift, 0)):
            x = self.acc(x * 2)
        for _ in range(max(-shift, 0)):
            # ap_fixed division returns F=22, already truncated towards zero.
            x = trunc_div(x, 2)
        return x

    def commit(self, x, exp=0, output_shift=0):
        return self.data(round_shift(self.shift(x, exp - output_shift), 11))

    def renorm(self, x, exp):
        while x.abs().max().item() > self.guard:
            x = trunc_div(x, 2)
            exp += 1
            self.count("renormalizations", 1)
        return x, exp

    def reduction(self, acc, weights, patches):
        """[OC, K] @ [K, pixels], saturating each MAC when required.

        Each input is a signed 16-bit code. K <= 1152 (147 in the stem),
        so every integer product and absolute sum fits exactly in float64.
        abs(acc) + sum(abs(products)) bounds *all* sequential prefixes.
        """
        if self.fast:
            bound = weights.abs() @ patches.abs()
            if (acc.abs().to(torch.float64) + bound <= ACC_MAX).all():
                self.count("fast_chunks", 1)
                return acc + (weights @ patches).to(torch.int64)
        self.count("sequential_chunks", 1)
        w, p = weights.to(torch.int64), patches.to(torch.int64)
        for k in range(w.shape[1]):
            acc = self.acc(acc + w[:, k:k + 1] * p[k:k + 1, :])
        return acc

    def conv(self, x, weights, stride=1, padding=0, output_shift=0):
        co_total, ci_total, kernel, _ = weights.shape
        h = (x.shape[1] + 2 * padding - kernel) // stride + 1
        w = (x.shape[2] + 2 * padding - kernel) // stride + 1
        patches = F.unfold(x.to(torch.float64).unsqueeze(0), kernel, padding=padding, stride=stride)[0]
        wf = weights.to(torch.float64).reshape(co_total, -1)
        out = torch.empty((co_total, h, w), dtype=torch.int64)
        for co in range(0, co_total, 128):
            nc = min(128, co_total - co)
            for row in range(0, h, 14):
                nh = min(14, h - row)
                for col in range(0, w, 14):
                    nw = min(14, w - col)
                    indices = (torch.arange(row, row + nh)[:, None] * w + torch.arange(col, col + nw)).flatten()
                    tile = patches[:, indices]
                    acc = torch.zeros((nc, nh * nw), dtype=torch.int64)
                    exp = 0
                    for ci in range(0, ci_total, 128):
                        start, stop = ci * kernel * kernel, min(ci + 128, ci_total) * kernel * kernel
                        # Deliberately reproduce the emitter: incoming products
                        # are NOT scaled when exp is nonzero (known defect).
                        acc = self.reduction(acc, wf[co:co + nc, start:stop], tile[start:stop])
                        acc, exp = self.renorm(acc, exp)
                        self.events.append([self.name, co, row, col, ci, exp])
                    out[co:co + nc, row:row + nh, col:col + nw] = self.commit(acc, exp, output_shift).reshape(nc, nh, nw)
        return out

    def bn(self, x, params):
        gamma, beta, mean, variance = params
        denominator = sqrt_codes(variance)[:, None, None]
        numerator = (x - mean[:, None, None]) * (1 << 33)
        norm = self.acc(trunc_div(numerator, denominator))
        # gamma*norm+beta has a wide expression; no intermediate acc cast.
        affine = gamma[:, None, None] * norm + beta[:, None, None] * (1 << 22)
        return self.data(round_shift(affine, 22))

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
        for co in range(0, channels, 128):
            tile = x[co:co + 128].reshape(-1, h * w) * 2048
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
        for co in range(0, weights.shape[0], 128):
            nc = min(128, weights.shape[0] - co)
            acc = torch.zeros((nc, 1), dtype=torch.int64)
            exp = 0
            for ci in range(0, x.numel(), 128):
                acc = self.reduction(acc, wf[co:co + nc, ci:ci + 128], patches[ci:ci + 128])
                acc, exp = self.renorm(acc, exp)
                self.events.append([self.name, co, 0, 0, ci, exp])
            out[co:co + nc] = self.commit(acc, exp, output_shift).flatten()
        return out
