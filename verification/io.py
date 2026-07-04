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

    Matches `%f` printf formatting (6 decimal places) used by the testbench dump
    so the file is human-comparable; the testbench reads it with `fscanf("%f")`.
    """
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    with open(path, "w") as f:
        f.write("\n".join(f"{v:.6f}" for v in flat))
