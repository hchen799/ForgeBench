"""Typed storage and arithmetic for the production JSON-to-HLS emitter.

No model graph or verification kernel is generated here. Buffer roles select
integer controls and persistent wide reductions; the ordered JSON ops remain
the source of the accelerator's schedule.

Arithmetic is set from the design JSON (all fields optional; without them the
emitted text is unchanged):

    "data_type":    storage type data_t, e.g. "ap_fixed<16,5>" or "fixed<16,5,rnd,sat>" (docs/DATA_TYPES.md)
    "acc_type":     the accumulator acc_t, e.g. "fixed<64,42>"; "acc_rounding" (trn|rnd), "acc_overflow" (wrap|sat)
                    (default: the production wide accumulator derived from data_type, truncate + wrap)
    "math":         "table" (default: ROM exp, integer-code division and square root on acc_t) or
                    "hls" (Vitis hls::exp / hls::sqrt on "math_type", default "fixed<48,16>"; division in acc_t)
    brams/drams[*].data_type: "acc" | "data" | "int32" | any type spelling; overrides the role-derived buffer type
"""
import copy
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backends import current as _backend          # noqa: E402
from backends.base import parse_fixed             # noqa: E402


REVISION = 2
DEFAULT_MATH_TYPE = "fixed<48,16>"     # hls::exp / hls::sqrt return 0 for ap_fixed with I > 34 (e.g. <64,42>); <48,16> is accurate


def c_type(spelling):
    """Design-JSON type spelling -> Vitis C++ type (raw ap_* passes through unchanged)."""
    return _backend().type_decl(spelling)


def arithmetic(config):
    """Arithmetic fields of a design JSON -> dict used by numeric_support (None values = production defaults)."""
    math_impl = config.get("math", "table")
    if math_impl not in ("table", "hls"):
        raise ValueError(f"math must be 'table' or 'hls', got {math_impl!r}")
    acc = None
    if any(config.get(k) is not None for k in ("acc_type", "acc_rounding", "acc_overflow")):
        if config.get("acc_type") is None:
            raise ValueError("acc_rounding/acc_overflow need an acc_type")
        spec = parse_fixed(config["acc_type"])
        if spec is None:
            raise ValueError(f"unsupported acc_type {config['acc_type']!r}")
        w, i, q, o = spec
        q = (config.get("acc_rounding") or q).lower()
        o = (config.get("acc_overflow") or o).lower()
        if q not in ("trn", "rnd") or o not in ("wrap", "sat"):
            raise ValueError(f"acc modes must be trn|rnd and wrap|sat, got {q}, {o}")
        acc = (w, i, q, o)
    return {"acc": acc, "math": math_impl, "math_type": config.get("math_type", DEFAULT_MATH_TYPE)}


_ROLE_TYPES = {"acc": "acc_t", "data": "data_t", "int32": "int32_t"}


def buffer_type_name(spelling):
    """Per-buffer data_type -> (C name used in the code, typedef line or None)."""
    if spelling in _ROLE_TYPES:
        return _ROLE_TYPES[spelling], None
    ctype = c_type(spelling)
    tag = re.sub(r"[^0-9A-Za-z]+", "_", ctype).strip("_")
    return f"t_{tag}", f"typedef {ctype} t_{tag};"


def buffer_typedefs(buffers):
    """typedef lines for explicitly typed buffers (empty for role-typed designs)."""
    lines = []
    for b in buffers:
        if "data_type" in b:
            line = buffer_type_name(b["data_type"])[1]
            if line and line not in lines:
                lines.append(line)
    return lines


def annotate_storage(brams, drams, ops):
    brams, drams = copy.deepcopy(brams), copy.deepcopy(drams)
    buffers = {b["name"]: b for b in brams + drams}
    operations = ops.values() if isinstance(ops, dict) else (v for _, v in ops)
    operations = list(operations)
    used = {arg for op in operations for arg in op.get("args", ())}
    unread = [
        b["name"]
        for b in drams
        if (b["name"].startswith(("DRAM_w_", "DRAM_bn_")) or b["name"] == "DRAM_fc")
        and b["name"] not in used
    ]
    if unread and "DRAM_w_s1_b0_1" in buffers:
        raise ValueError(
            "unread model parameters; regenerate the JSON with auto_generate_json.py: "
            + ", ".join(unread)
        )
    wide_arguments = {
        "conv_tile": (2,),
        "linear_tile": (2,),
        "rmsnorm_accumulate_tile": (1,),
        "rmsnorm_finalize_rows": (0, 1),
        "rmsnorm_apply_tile": (2,),
        "init_rowmax_tile": (0,),
        "attention_score_tile": (2,),
        "attention_rowmax_tile": (0, 1),
        "attention_softmax_context_tile": (0, 2, 3, 4),
        "attention_finalize_tile": (0, 1),
        "avgpool_accumulate_tile": (1,),
        "avgpool_finalize_tile": (0, 1),
    }

    def assign(name, dtype):
        if name not in buffers:
            raise ValueError("expected named buffer: " + name)
        prior = buffers[name].get("dtype")
        if prior is not None and prior != dtype:
            raise ValueError(f"{name}: expected {dtype}, got {prior}")
        buffers[name]["dtype"] = dtype

    for op in operations:
        for index in wide_arguments.get(op["func_name"], ()):
            assign(op["args"][index], "acc_t")
        if op["func_name"] in ("embedding_lookup_chunk", "embedding_lookup_tile"):
            assign(op["args"][0], "int32_t")
    for name in ("DRAM_prefill_len", "DRAM_decode_pos"):
        if name in buffers:
            assign(name, "int32_t")
    for op in operations:
        if (
            op["func_name"] == "load"
            and buffers.get(op["args"][0], {}).get("dtype") == "int32_t"
        ):
            assign(op["args"][1], "int32_t")
    for b in buffers.values():                      # an explicit per-buffer data_type overrides the role-derived type
        if "data_type" in b:
            name = buffer_type_name(b["data_type"])[0]
            if name == "data_t":
                b.pop("dtype", None)
            else:
                b["dtype"] = name
    for b in buffers.values():
        if b.get("dtype", "data_t") not in ("data_t", "acc_t", "int32_t") and not b["dtype"].startswith("t_"):
            raise ValueError("unsupported buffer dtype: " + str(b))
    return brams, drams


def specialize(code, name, op, storage, shapes=None):
    """Specialize array parameter types, keeping the production operator body."""
    pattern = r"\bvoid\s+" + re.escape(name) + r"\s*\((.*?)\)\s*\{"
    match = re.search(pattern, code, re.S)
    if match is None:
        raise ValueError("cannot locate operator signature: " + name)
    parameters = match[1].split(",")
    if len(parameters) != len(op["args"]):
        raise ValueError(
            f'{name}: {len(parameters)} parameters but {len(op["args"])} arguments'
        )
    types = []
    for i, (parameter, arg) in enumerate(zip(parameters, op["args"])):
        if re.search(r"\bdata_t\s+\w+\s*\[", parameter):
            base = re.match(r"\w+", arg)
            if shapes is not None:
                if base is None or base[0] not in shapes:
                    raise ValueError(f"{name}: undeclared array argument {arg}")
                expected = tuple(map(int, re.findall(r"\[(\d+)\]", parameter)))
                actual = tuple(shapes[base[0]])[arg.count("[") :]
                if len(actual) != len(expected) or actual[1:] != expected[1:]:
                    raise ValueError(
                        f"{name}: {arg} has shape {actual}, needs {expected}"
                    )
                # The first C++ array extent is only a pointer annotation;
                # runtime valid counts/layer_idx determine how much is used.
            dtype = storage.get(base[0], "data_t") if base else "data_t"
            parameters[i] = re.sub(r"\bdata_t\b", dtype, parameter)
            types.append(dtype)
    if all(t == "data_t" for t in types):
        return code, name
    replacement = name + "__" + "_".join(types)
    code = code[: match.start(1)] + ",".join(parameters) + code[match.end(1) :]
    return re.sub(r"\b" + re.escape(name) + r"\b", replacement, code), replacement


def accumulator_format(data_type):
    match = re.fullmatch(r"ap_fixed\s*<\s*(\d+)\s*,\s*(\d+)(?:\s*,[^>]+)?>", data_type)
    if match is None:
        if data_type in ("float", "double"):
            return None
        raise ValueError("unsupported production arithmetic: " + data_type)
    word, integer = map(int, match.groups())
    if not 1 <= integer < word <= 32:
        raise ValueError("production fixed arithmetic requires 1 <= I < W <= 32")
    fractional = 2 * (word - integer)
    width = max(64, 2 * word + 16)
    return width, width - fractional, fractional


def numeric_support(data_type, arith=None):
    arith = arith or {"acc": None, "math": "table", "math_type": DEFAULT_MATH_TYPE}
    fmt = accumulator_format(data_type)
    if fmt is None:
        return """typedef double acc_t;
static acc_t fb_sqrt(acc_t x) { return hls::sqrt(x); }
static acc_t fb_div(acc_t a, acc_t b) { assert(b != 0); return a / b; }
static acc_t fb_exp_negative(acc_t x) { return hls::exp(x); }
static const acc_t FB_EPS = 1e-5;
static const acc_t FB_MIN_SCORE = -1e30;
"""
    width, integer, frac = fmt
    acc_modes = "AP_TRN,AP_WRAP"
    if arith["acc"] is not None:
        width, integer, q, o = arith["acc"]
        frac = width - integer
        if not 1 <= integer < width or frac > 62:
            raise ValueError("acc_type needs 1 <= I < W and at most 62 fractional bits")
        acc_modes = f"AP_{q.upper()},AP_{o.upper()}"
    if arith["math"] == "hls":
        return _hls_math_support(width, integer, frac, acc_modes, arith["math_type"])
    table = [math.floor(math.exp(-i / 256) * (1 << frac) + 0.5) for i in range(4097)]
    entries = ",\n".join(
        ", ".join(str(v) + "ULL" for v in table[i : i + 16])
        for i in range(0, len(table), 16)
    )
    return f"""// Production arithmetic revision {REVISION}: persistent wide sums.
typedef ap_fixed<{width},{integer},{acc_modes}> acc_t;
static const int FB_ACC_BITS = {width};
static const int FB_ACC_FRAC = {frac};
static acc_t fb_from_code(ap_int<{width}> code) {{
    acc_t result; result.range({width-1},0) = code; return result;
}}
static ap_int<{width}> fb_code(acc_t x) {{ return x.range({width-1},0); }}
static acc_t fb_div(acc_t a, acc_t b) {{
    assert(b != 0);
    ap_int<{2*width}> numerator = fb_code(a);
    numerator <<= {frac};
    ap_int<{2*width}> denominator = fb_code(b);
    ap_int<{2*width}> q = numerator / denominator;
    if ((numerator % denominator) != 0 && ((numerator < 0) != (denominator < 0))) --q;
    return fb_from_code(q);
}}
static acc_t fb_sqrt(acc_t x) {{
    assert(x >= 0);
    ap_uint<{2*width}> value = fb_code(x);
    value <<= {frac};
    ap_uint<{2*width}> remainder = value, root = 0;
    ap_uint<{2*width}> bit = ap_uint<{2*width}>(1) << {2*width-2};
    for (int i=0; i<{width}; ++i) {{
        if (remainder >= root + bit) {{ remainder -= root + bit; root = (root >> 1) + bit; }}
        else root >>= 1;
        bit >>= 2;
    }}
    if (value - root * root > root) ++root;
    return fb_from_code(root);
}}
static const acc_t FB_EPS = fb_from_code({max(1, math.floor(1e-5 * (1 << frac)))}ULL);
static const acc_t FB_MIN_SCORE = -(acc_t(1) << {integer-2});
// exp(x), x<=0: 1/256 grid, nearest ROM coefficients and linear interpolation.
// Below -16 the omitted value is < 1.13e-7. No floating-point runtime exp.
static acc_t fb_exp_negative(acc_t x) {{
    if (x >= 0) return acc_t(1);
    if (x <= -16) return acc_t(0);
    static const unsigned long long table[4097] = {{
{entries}
    }};
    ap_uint<{2*width}> coordinate = -fb_code(x);
    coordinate <<= 8;
    unsigned index = (coordinate >> {frac}).to_uint();
    ap_uint<{frac}> fraction = coordinate;
    ap_uint<{2*width}> value = ap_uint<{2*width}>(table[index]) * ((ap_uint<{2*width}>(1) << {frac}) - fraction)
                                + ap_uint<{2*width}>(table[index+1]) * fraction;
    return fb_from_code(value >> {frac});
}}
"""


def _hls_math_support(width, integer, frac, acc_modes, math_type):
    """acc_t with Vitis library math: exp/sqrt on math_t (default modes; Vitis hls_math only accepts truncate + wrap), division in acc_t."""
    spec = parse_fixed(math_type)
    if spec is None:
        raise ValueError(f"unsupported math_type {math_type!r}")
    mw, mi = spec[0], spec[1]
    return f"""// Production arithmetic revision {REVISION}: persistent wide sums, Vitis library math.
typedef ap_fixed<{width},{integer},{acc_modes}> acc_t;
typedef ap_fixed<{mw},{mi},AP_TRN,AP_WRAP> math_t;
static const int FB_ACC_BITS = {width};
static const int FB_ACC_FRAC = {frac};
static acc_t fb_from_code(ap_int<{width}> code) {{
    acc_t result; result.range({width-1},0) = code; return result;
}}
static acc_t fb_div(acc_t a, acc_t b) {{ assert(b != 0); return acc_t(a / b); }}
static acc_t fb_sqrt(acc_t x) {{ assert(x >= 0); return acc_t(hls::sqrt(math_t(x))); }}
static acc_t fb_exp_negative(acc_t x) {{ return acc_t(hls::exp(math_t(x))); }}
static const acc_t FB_EPS = fb_from_code({max(1, math.floor(1e-5 * (1 << frac)))}ULL);
static const acc_t FB_MIN_SCORE = -(acc_t(1) << {integer-2});
"""
