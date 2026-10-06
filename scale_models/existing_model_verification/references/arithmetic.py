"""Independent integer-code arithmetic for production AP_TRN/AP_WRAP designs.

This deliberately does not import the dedicated generators' precision helpers.
Division truncates toward zero at the expression's fractional width; assignment
to a narrower fractional width floors. These are different operations.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch


class ArithmeticFault(ValueError):
    """The declared fixed-point computation has an invalid operation."""


def wrap(x, bits=16):
    return (x + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))


def trunc_div(n, d):
    n, d = torch.broadcast_tensors(
        torch.as_tensor(n, dtype=torch.int64), torch.as_tensor(d, dtype=torch.int64)
    )
    if torch.any(d == 0):
        raise ArithmeticFault("fixed-point division by zero")
    return torch.div(n, d, rounding_mode="trunc")


def sqrt_codes(x, fractional=11):
    """Vendor fixed sqrt rounds to nearest; negative arguments return zero.

    Integer square roots avoid FP64 rounding at integer-code boundaries. Only
    the installed library's supported I<=34,F<=32 domain is used by this adapter.
    """
    x = torch.as_tensor(x, dtype=torch.int64)
    result = []
    for value in x.reshape(-1).tolist():
        radicand = max(value, 0) << fractional
        root = math.isqrt(radicand)
        root += 4 * radicand >= (2 * root + 1) ** 2
        result.append(root)
    return torch.tensor(result, dtype=torch.int64).reshape(x.shape)


@dataclass(frozen=True)
class Format:
    word: int = 16
    integer: int = 5

    @property
    def frac(self):
        return self.word - self.integer

    @property
    def scale(self):
        return 1 << self.frac

    def quantize(self, real):
        real = torch.as_tensor(real, dtype=torch.float64)
        if not torch.isfinite(real).all():
            raise ArithmeticFault("nonfinite input")
        return wrap(torch.floor(real * self.scale).to(torch.int64), self.word)

    def describe(self):
        return dict(
            word_bits=self.word,
            integer_bits=self.integer,
            rounding="AP_TRN",
            overflow="AP_WRAP",
            fractional_bits=self.frac,
        )


DATA = Format()
ACC = Format(32, 10)


def product_sum(x, weight, chunk=32):
    """Sum separately truncated products, as in data_t += data_t * data_t.

    AP_WRAP addition is associative modulo 2**16, so final wrapping is exact
    even when we sum a vector of already-truncated products in int64. In
    particular this is NOT floor(matmul(x, w)/scale).
    x: [reduction, locations], weight: [outputs, reduction].
    """
    output = torch.zeros((weight.shape[0], x.shape[1]), dtype=torch.int64)
    for co in range(0, weight.shape[0], 16):
        local = output[co : co + 16]
        for ci in range(0, x.shape[0], chunk):
            p = weight[co : co + 16, ci : ci + chunk, None] * x[None, ci : ci + chunk]
            local += torch.bitwise_right_shift(p, DATA.frac).sum(1)
    return wrap(output)


def exact_matmul(a, b):
    """Use FP64 GEMM only after proving every integer reduction is exact."""
    bound = int(a.abs().max()) * int(b.abs().max()) * a.shape[-1]
    if bound > 2**52:
        raise ArithmeticFault("integer GEMM exceeds proven FP64 exactness bound")
    return (a.double() @ b.double()).to(torch.int64)


def exp_codes(x, word=17, integer=6):
    """Integer evaluation of Vitis's F=11 exp range reduction.

    Small tables are derived from the mathematical exponential, with the
    library's stated table precisions (nearest for tables, truncation for
    intermediates). This never calls the accelerator or a C++ oracle.
    """
    if word - integer != 11 or not 2 <= integer <= 12:
        raise ArithmeticFault("unsupported fixed exp format")
    x = torch.as_tensor(x, dtype=torch.int64)
    lsb = x & 31
    mid = (x >> 5) & 31
    coarse = ((x >> 10) & 15) | ((x < 0).long() << 4)
    # The low correction is rounded to 24 fractional bits, the middle table
    # to 25, and the coarse table to 14. Test all input codes against Vitis.
    low_table = torch.tensor(
        [
            math.floor((math.expm1(i / 2048) - i / 2048) * 2**24 + 0.5)
            for i in range(32)
        ],
        dtype=torch.int64,
    )
    mid_table = torch.tensor(
        [math.floor(math.expm1(i / 64) * 2**25 + 0.5) for i in range(32)],
        dtype=torch.int64,
    )
    coarse_table = torch.tensor(
        [
            math.floor(math.exp((i if i < 16 else i - 32) / 2) * 2**14 + 0.5)
            for i in range(32)
        ],
        dtype=torch.int64,
    )
    low = (lsb << 13) + low_table[lsb]  # F=24
    middle = mid_table[mid]  # F=25
    correction = ((middle * low) >> 24) + (low << 1) + middle
    big = coarse_table[coarse]
    value = ((((big * correction) >> 25) + big) % (1 << 25)) >> 3  # F=11
    value = torch.where(x < -8 * 2048, 0, value)
    value = torch.where(x > int(7.625 * 2048), (1 << (word - 1)) - 1, value)
    return value.clamp(0, (1 << (word - 1)) - 1)
