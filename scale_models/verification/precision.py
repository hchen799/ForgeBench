"""Explicit signed fixed-point formats and overflow-safe exact tensor helpers."""
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class FixedFormat:
    word_bits: int = 16
    integer_bits: int = 5
    rounding: str = "nearest_ties_positive"
    overflow: str = "saturate"

    def __post_init__(self):
        if not (isinstance(self.word_bits, int) and isinstance(self.integer_bits, int)):
            raise ValueError("fixed-point widths must be integers")
        if not 8 <= self.word_bits <= 32 or not 2 <= self.integer_bits < self.word_bits:
            raise ValueError("supported storage: 8 <= W <= 32 and 2 <= I < W (I includes sign)")
        if self.fractional_bits > 24:
            raise ValueError("at most 24 fractional bits are supported by the verified wide datapath")
        if self.rounding != "nearest_ties_positive" or self.overflow != "saturate":
            raise ValueError("supported arithmetic modes: nearest_ties_positive and saturate")

    @property
    def fractional_bits(self):
        return self.word_bits - self.integer_bits

    @property
    def scale(self):
        return 1 << self.fractional_bits

    @property
    def minimum(self):
        return -(1 << (self.word_bits - 1))

    @property
    def maximum(self):
        return (1 << (self.word_bits - 1)) - 1

    @property
    def dtype(self):
        return "<i2" if self.word_bits <= 16 else "<i4"

    @property
    def itemsize(self):
        return np.dtype(self.dtype).itemsize

    @property
    def torch_dtype(self):
        return torch.int16 if self.word_bits <= 16 else torch.int32

    @property
    def cpp_type(self):
        return "int16_t" if self.word_bits <= 16 else "int32_t"

    def describe(self):
        return dict(**asdict(self), fractional_bits=self.fractional_bits,
                    storage_dtype=self.dtype, storage_bytes=self.itemsize,
                    scale=self.scale, lsb=1 / self.scale,
                    real_minimum=self.minimum / self.scale,
                    real_maximum=self.maximum / self.scale)

    def quantize(self, x):
        if not torch.isfinite(x).all():
            raise ValueError("non-finite quantization input")
        return torch.floor(x.double() * self.scale + .5).clamp(self.minimum, self.maximum).long()


def objects(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=object)


def clamp_integer(x, minimum, maximum):
    if isinstance(x, torch.Tensor):
        return x.clamp(minimum, maximum)
    # Without object dtype, a positive Python scalar in [2^63,2^64) is
    # inferred as uint64; combining it with signed limits can promote to
    # FP64 and round INT64_MAX up to 2^63 before the final int64 cast.
    return torch.from_numpy(np.asarray(np.clip(objects(x), minimum, maximum), dtype=np.int64))


def round_shift(x, bits):
    """Round a possibly wider-than-int64 integer tensor, with signed ties up."""
    if bits == 0:
        return x
    if bits < 0:
        return objects(x) * (1 << -bits)
    if isinstance(x, torch.Tensor) and bits < 63 and int(x.max()) <= (1 << 63) - 1 - (1 << (bits - 1)):
        return torch.div(x + (1 << (bits - 1)), 1 << bits, rounding_mode="floor")
    return (objects(x) + (1 << (bits - 1))) // (1 << bits)


def exact_dot(a, b):
    """Exact integer matrix product, including 32-bit extremes and wide sums.

    FP64 is used only where a conservative bound proves exactness. Otherwise
    signed integers are decomposed into base-2^15 limbs; each limb GEMM is
    exact, and Python integers reconstruct results without int64 overflow.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("exact_dot requires compatible matrices")
    if a.shape[1] > 32768:
        raise ValueError("exact_dot reduction exceeds the verified limb bound")
    a, b = a.double(), b.double()
    if any(not torch.isfinite(x).all() or (x != torch.floor(x)).any() or (x.abs() > 1 << 31).any() for x in (a, b)):
        raise ValueError("exact_dot operands must be integer codes of magnitude <= 2^31")
    bound = int(a.abs().max()) * int(b.abs().max()) * a.shape[1]
    if bound <= 1 << 52:
        return (a @ b).to(torch.int64)
    base = 1 << 15
    ah, bh = torch.floor(a / base), torch.floor(b / base)
    al, bl = a - ah * base, b - bh * base
    parts = [(al @ bl).long(), (ah @ bl).long(), (al @ bh).long(), (ah @ bh).long()]
    return objects(parts[0]) + (objects(parts[1]) + objects(parts[2])) * base + objects(parts[3]) * base**2


def rounded_ratio(numerator, denominator):
    n, d = objects(numerator), objects(denominator)
    if np.any(d <= 0):
        raise ValueError("division requires a positive denominator")
    return (2 * n + d) // (2 * d)
