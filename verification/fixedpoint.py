"""ap_fixed emulation and error metrics for fixed-point operator verification (import after the repo root is on sys.path).

`quantize` reproduces what the generated testbench does to each input (`array[i] = (data_t)temp`, `temp` a double):
the conversion double -> ap_fixed<W,I,Q,O>. Supported modes: Q in {AP_TRN (default), AP_RND}, O in {AP_WRAP (default), AP_SAT}.
These are validated bit-for-bit against the real Vitis ap_fixed headers by verification/tests/test_fixedpoint.py.

  AP_TRN   truncate toward -infinity (floor)            AP_WRAP  two's-complement wrap into [-2^(I-1), 2^(I-1))
  AP_RND   round to nearest, ties toward +infinity      AP_SAT   clamp to [-2^(I-1), 2^(I-1) - 2^-F]
"""
import numpy as np

from backends.base import parse_fixed


class DType:
    """A data type spelled as in a config's `data_type`: 'float', a generic `fixed<W,I[,quant[,overflow]]>` (default round +
    saturate) or a raw `ap_fixed<W,I[,AP_*[,AP_*]]>` (default truncate + wrap). Parsing is `backends.base.parse_fixed`, the same
    function the backends use to lower the type, so the verification and the generated design agree on what the string means."""

    def __init__(self, text):
        t = text.strip()
        self.text = t
        if t == "float":
            self.kind, self.W, self.I, self.F, self.q, self.o = "float", 32, 0, 0, None, None
            return
        fx = parse_fixed(t)
        if fx is None:
            raise ValueError(f"unsupported data type {text!r}: expected 'float', 'fixed<W,I[,trn|rnd[,wrap|sat]]>' or 'ap_fixed<...>'")
        self.kind = "fixed"
        self.W, self.I, qq, oo = fx
        self.F = self.W - self.I
        self.q, self.o = f"AP_{qq.upper()}", f"AP_{oo.upper()}"
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
        """Filesystem/column-safe name, e.g. fixed_16_5_rnd_sat."""
        return "float" if self.kind == "float" else f"fixed_{self.W}_{self.I}_{self.q[3:].lower()}_{self.o[3:].lower()}"


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
