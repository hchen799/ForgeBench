"""Working precision of the NumPy goldens.

Float verification uses float32 (the design's own precision, the historical default). Fixed-point error characterization
needs float64: with a float32 golden the reference's own rounding (~1e-7 relative) is comparable to the quantization step of
wide types such as <32,10> (2^-22 ~ 2.4e-7) and would contaminate the measurement. Modules read `fp.FP` at call time, so
`set_precision(np.float64)` switches every golden function without changing any formula.
"""
import numpy as np

FP = np.float32


def set_precision(dtype):
    global FP
    FP = np.dtype(dtype).type
    return FP
