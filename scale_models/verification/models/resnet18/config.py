"""Supported ResNet-18 graph with configurable tiles, shifts, and arithmetic."""
from dataclasses import asdict, dataclass, field

import generate_tiled_resnet18 as gen
from ...precision import FixedFormat


@dataclass
class Config:
    word_bits: int = 16
    integer_bits: int = 5
    acc_word_bits: int = None
    acc_integer_bits: int = None
    tile_c: int = 128
    tile_h: int = 14
    tile_w: int = 14
    guard: int = None
    shifts: dict = field(default_factory=dict)

    @property
    def precision(self):
        return FixedFormat(self.word_bits, self.integer_bits)

    def validate(self):
        p = self.precision
        if self.acc_word_bits is None:
            self.acc_word_bits = 2 * p.word_bits
        if self.acc_integer_bits is None:
            self.acc_integer_bits = self.acc_word_bits - 2 * p.fractional_bits
        for name in ("acc_word_bits", "acc_integer_bits", "tile_c", "tile_h", "tile_w"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 8 <= self.acc_word_bits <= 64 or not 2 <= self.acc_integer_bits < self.acc_word_bits:
            raise ValueError("ResNet accumulator requires 8 <= W <= 64 and 2 <= I < W")
        if self.acc_word_bits - self.acc_integer_bits != 2 * p.fractional_bits:
            raise ValueError("ResNet accumulator must retain exactly twice the storage fractional bits")
        if not 4 <= self.tile_c <= 256 or not 1 <= min(self.tile_h, self.tile_w) <= max(self.tile_h, self.tile_w) <= 28:
            raise ValueError("require tile_c in [4,256] and tile_h/tile_w in [1,28]")
        if self.guard is None:
            self.guard = min(256, 1 << (self.acc_integer_bits - 2))
        if not isinstance(self.guard, int) or not 0 < self.guard < 1 << (self.acc_integer_bits - 1):
            raise ValueError("renormalization guard must fit the positive accumulator range")
        names = gen.build_shift_names(gen.build_blocks())
        if set(self.shifts) - set(names):
            raise ValueError("unknown ResNet output-shift names")
        self.shifts = {name: self.shifts.get(name, 0) for name in names}
        if any(not isinstance(v, int) or abs(v) > 31 for v in self.shifts.values()):
            raise ValueError("output shifts must be integers in [-31,31]")
        return self

    def to_dict(self):
        return asdict(self)
