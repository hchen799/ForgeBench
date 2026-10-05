"""ap_fixed emulation and error metrics for fixed-point operator verification.

`quantize` reproduces what the generated testbench does to each input (`array[i] = (data_t)temp`, `temp` a double):
the conversion double -> ap_fixed<W,I,Q,O>. Supported modes: Q in {AP_TRN (default), AP_RND}, O in {AP_WRAP (default), AP_SAT}.
These are validated bit-for-bit against the real Vitis ap_fixed headers by verification/tests/test_fixedpoint.py.

  AP_TRN   truncate toward -infinity (floor)            AP_WRAP  two's-complement wrap into [-2^(I-1), 2^(I-1))
  AP_RND   round to nearest, ties toward +infinity      AP_SAT   clamp to [-2^(I-1), 2^(I-1) - 2^-F]
"""
import re

import numpy as np

_DT = re.compile(r"^\s*ap_fixed\s*<\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(AP_\w+)\s*(?:,\s*(AP_\w+)\s*)?)?>\s*$")


class DType:
    def __init__(self, text):
        t = text.strip()
        self.text = t
        if t == "float":
            self.kind, self.W, self.I, self.F, self.q, self.o = "float", 32, 0, 0, None, None
            return
        m = _DT.match(t)
        if not m:
            raise ValueError(f"unsupported data type {text!r}: expected 'float' or 'ap_fixed<W,I[,AP_TRN|AP_RND[,AP_WRAP|AP_SAT]]>'")
        self.kind = "fixed"
        self.W, self.I = int(m.group(1)), int(m.group(2))
        self.F = self.W - self.I
        self.q, self.o = m.group(3) or "AP_TRN", m.group(4) or "AP_WRAP"
        if self.q not in ("AP_TRN", "AP_RND"):
            raise ValueError(f"unsupported quantization mode {self.q}")
        if self.o not in ("AP_WRAP", "AP_SAT"):
            raise ValueError(f"unsupported overflow mode {self.o}")
        if self.W > 52:
            raise ValueError("W > 52 cannot be emulated exactly in float64")

    @property
    def lsb(self):
        return 2.0 ** -self.F

    @property
    def lo(self):
        return -(2.0 ** (self.I - 1))

    @property
    def hi(self):
        return 2.0 ** (self.I - 1) - self.lsb

    def tag(self):
        """Short filesystem/column-safe name, e.g. ap_fixed_16_5 or ap_fixed_16_5_RND_SAT."""
        if self.kind == "float":
            return "float"
        extra = "" if (self.q, self.o) == ("AP_TRN", "AP_WRAP") else f"_{self.q[3:]}_{self.o[3:]}"
        return f"ap_fixed_{self.W}_{self.I}{extra}"


def quantize(x, dt):
    """double array -> the double value of the ap_fixed it converts to (exact)."""
    x = np.asarray(x, dtype=np.float64)
    if dt.kind == "float":
        return x.astype(np.float32).astype(np.float64)
    scaled = x * (2.0 ** dt.F)                      # exact: power-of-two scale
    n = np.floor(scaled) if dt.q == "AP_TRN" else np.floor(scaled + 0.5)
    lo, hi = -(2.0 ** (dt.W - 1)), 2.0 ** (dt.W - 1) - 1
    if dt.o == "AP_SAT":
        n = np.clip(n, lo, hi)
    else:
        span = 2.0 ** dt.W
        n = np.mod(n - lo, span) + lo               # wrap into [lo, hi]
    return n * dt.lsb


def error_metrics(out, ref, atol, rtol=0.0):
    """Error of `out` against `ref` (float64). `atol/rtol`: tolerance for the within-tolerance fraction (a reported number, not a verdict)."""
    out, ref = np.asarray(out, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    err = out - ref
    ae = np.abs(err)
    p_sig, p_err = float(np.sum(ref * ref)), float(np.sum(err * err))
    return {
        "max_abs_err": float(ae.max()) if ae.size else 0.0,
        "mean_abs_err": float(ae.mean()) if ae.size else 0.0,
        "rmse": float(np.sqrt(np.mean(err * err))) if ae.size else 0.0,
        "sqnr_db": float("inf") if p_err == 0.0 else (float("-inf") if p_sig == 0.0 else 10.0 * np.log10(p_sig / p_err)),
        "rel_l2": float(np.sqrt(p_err / p_sig)) if p_sig > 0 else (0.0 if p_err == 0 else float("inf")),
        "frac_within_tol": float(np.mean(ae <= atol + rtol * np.abs(ref))) if ae.size else 1.0,
        "n_elements": int(ae.size),
    }
