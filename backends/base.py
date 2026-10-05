"""Tool-backend abstraction for the ForgeBench generators (review concern E1).

Every generated design used to be Vitis-specific in four places: the `ap_fixed`
types, the `#pragma HLS ...` directives, the `m_axi` interface pragmas, and
`run_hls.tcl`. A `ToolBackend` owns all four, so the same JSON config can emit a
Vitis project or a Catapult project.

Registry style mirrors `analysis/extractors/base.py` so both halves of the repo
select a tool the same way.

Note on loop emission: Vitis puts `#pragma HLS unroll factor=N` *inside* the loop
body; Catapult wants `#pragma hls_unroll N` *immediately before* the `for`. A hook
that only returns a pragma string cannot express that difference, which is why the
backend owns `open_loop()` — the whole loop opening — rather than just the pragma.
"""
import re

_REGISTRY = {}


def register(cls):
    _REGISTRY[cls.name] = cls
    return cls


def get_backend(name):
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown tool backend {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]()


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------

_FIXED_RE = re.compile(r"^\s*(?:ap_|ac_)?fixed\s*<\s*(\d+)\s*,\s*(-?\d+)\s*(?:,[^>]*)?>\s*$")


# ---- fixed-point format with explicit quantization / overflow modes -------------------------------------------------
# Generic spelling:  fixed<W,I>  fixed<W,I,rnd>  fixed<W,I,rnd,sat>   (modes: trn|rnd, wrap|sat; case-insensitive)
# Tool spellings:    ap_fixed<W,I[,AP_RND[,AP_SAT]]>   ac_fixed<W,I,true,AC_RND,AC_SAT>
# Defaults: a spelling without modes -- generic `fixed<W,I>` or a tool spelling `ap_fixed<W,I>` -- means the tool defaults
# (truncate + wrap). Other modes are never implied; they must be written explicitly, e.g. fixed<16,5,rnd,sat>.
# NOTE: Vitis hls_math.h (exp, sqrt, tanh, sin, cos) only supports the default modes, so designs that call those functions fail to
# compile with rnd/sat in Vitis; ForgeBench does not work around that.
GENERIC_DEFAULT_MODES = ("trn", "wrap")
TOOL_DEFAULT_MODES = ("trn", "wrap")
_MODE_Q = {"trn": "trn", "ap_trn": "trn", "ac_trn": "trn", "rnd": "rnd", "ap_rnd": "rnd", "ac_rnd": "rnd"}
_MODE_O = {"wrap": "wrap", "ap_wrap": "wrap", "ac_wrap": "wrap", "sat": "sat", "ap_sat": "sat", "ac_sat": "sat"}
_FIXED_ANY_RE = re.compile(r"^\s*(ap_|ac_)?fixed\s*<\s*(\d+)\s*,\s*(-?\d+)\s*(?:,\s*([^>]*))?>\s*$")


def parse_fixed(data_type):
    """-> (W, I, quant, overflow) with quant in {trn,rnd}, overflow in {wrap,sat}; None if not a fixed-point spelling.
    Unsupported modes raise ValueError (they must not be silently ignored)."""
    m = _FIXED_ANY_RE.match(data_type.strip())
    if not m:
        return None
    prefix, w, i, rest = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    args = [a.strip().lower() for a in rest.split(",")] if rest and rest.strip() else []
    if prefix == "ac_" and args and args[0] in ("true", "false"):
        if args[0] == "false":
            raise ValueError(f"unsigned ac_fixed is not supported: {data_type!r}")
        args = args[1:]
    q, o = GENERIC_DEFAULT_MODES if prefix is None else TOOL_DEFAULT_MODES
    if len(args) > 2:
        raise ValueError(f"too many mode arguments in {data_type!r}")
    if len(args) >= 1:
        if args[0] not in _MODE_Q:
            raise ValueError(f"unsupported quantization mode {args[0]!r} in {data_type!r} (supported: trn, rnd)")
        q = _MODE_Q[args[0]]
    if len(args) == 2:
        if args[1] not in _MODE_O:
            raise ValueError(f"unsupported overflow mode {args[1]!r} in {data_type!r} (supported: wrap, sat)")
        o = _MODE_O[args[1]]
    return w, i, q, o


def normalize_data_type(data_type):
    """Canonicalize a config `data_type` into a tool-neutral form.

    Accepts the legacy Vitis spellings (`ap_fixed<16, 5>` — what all 3840 gemm
    configs currently contain), the Catapult spelling, the neutral spelling, and
    plain scalars. Returns ``("fixed", W, I)`` or ``("scalar", name, None)``.
    """
    dt = data_type.strip()
    m = _FIXED_RE.match(dt)
    if m:
        return ("fixed", int(m.group(1)), int(m.group(2)))
    return ("scalar", dt, None)


class ToolBackend:
    """Emission policy for one HLS tool.

    Subclasses must not change *what* the generators compute, only how it is
    spelled. The Vitis subclass is required to reproduce the pre-refactor output
    byte-for-byte; see the regression gate in the plan.
    """

    name = None

    # -- types -------------------------------------------------------------

    def type_decl(self, data_type):
        """The C++ type as it appears in generated declarations."""
        raise NotImplementedError

    def type_suffix(self, data_type):
        """Sanitized form used to build unique function names."""
        raise NotImplementedError

    # -- includes ----------------------------------------------------------

    def includes_top_cpp(self):
        raise NotImplementedError

    def includes_top_h(self):
        raise NotImplementedError

    def includes_tb(self):
        raise NotImplementedError

    # -- pragmas -----------------------------------------------------------

    def open_loop(self, header, factor):
        """Return the full loop opening, unroll directive included.

        `header` is the bare `for (...)` text with no trailing brace.
        """
        raise NotImplementedError

    def array_partition(self, var, factor, dim):
        """Return the in-body array-partition directive for `var`, or ""."""
        raise NotImplementedError

    def interface(self, drams):
        """Return the top-level interface pragma lines (may be empty)."""
        raise NotImplementedError

    # -- build script ------------------------------------------------------

    def emit_script(self, drams, target, clock_period, tasks, output_filename):
        raise NotImplementedError

    def run_cmd(self):
        """argv used to invoke the tool inside a design directory."""
        raise NotImplementedError


def effective_pragma_factor(requested_factor, concrete_dim):
    """Clamp an unroll/partition factor to the loop bound it applies to.

    Lifted from `scale_models/generate_code.py` (the only place in the repo where
    pragma factors were already funnelled through a helper) so every backend can
    share it. Not applied on the Vitis path — see `VitisBackend.open_loop`.
    """
    if requested_factor <= 1 or concrete_dim <= 1:
        return 1
    return min(requested_factor, concrete_dim)
