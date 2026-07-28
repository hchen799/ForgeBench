"""IO helpers shared by the ForgeBench golden-reference verification flow.

The generated testbench (`tb_top.cpp`) flattens every array row-major and reads
inputs with `fscanf("%f")`, which is whitespace-agnostic. To keep the golden
reference bit-for-bit consistent with what the HLS run consumes, we read the
exact same `.txt` files and reshape them row-major, and we emit golden files in
the same one-value-per-line float format.
"""
import numpy as np


def prod(dims):
    out = 1
    for d in dims:
        out *= d
    return out


def load_dram_txt(path, dims):
    """Load a DRAM `.txt` file into a float32 array shaped row-major by `dims`."""
    flat = np.loadtxt(path, dtype=np.float32).reshape(-1)
    n = prod(dims)
    if flat.size != n:
        raise ValueError(
            f"{path}: expected {n} values for dims {dims}, got {flat.size}"
        )
    return flat.reshape(dims).astype(np.float32)


def write_golden(path, arr):
    """Write `arr` (any shape) to `path`, row-major, one float per line.

    Uses `%.9g` -- the shortest format that round-trips float32 exactly. The
    earlier `%.6f` put a ~5e-7 *absolute* quantization floor on every comparison,
    which dominated the reported errors: identity ops (relu on positive inputs,
    dropout) reported max_abs=5.07e-07 purely from golden-file rounding, and ops
    with small outputs (gelu near zero, elementwise_mult) reported max_rel in the
    1e-3 range for the same reason. The testbench reads with `fscanf("%f")`,
    which accepts exponent notation, so this is backward compatible.
    """
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    with open(path, "w") as f:
        f.write("\n".join(f"{v:.9g}" for v in flat))
