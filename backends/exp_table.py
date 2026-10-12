"""Lookup-table exp for operators that call hls::exp / hls::tanh (op option `exp_impl`).

An op dict may carry `"exp_impl": "table"`; without it (or with "hls") the operator is generated exactly as before (hls::exp / hls::tanh).
With "table" the op's function is renamed `<name>_exptable` and its exp/tanh calls go through a 4,097-entry ROM:

  exp(-t), t in [0, 16): 1/256 grid, nearest ROM entries, linear interpolation; exp(-t) = 0 for t >= 16 (omitted value < 1.13e-7)
  exp(x), x > 0:         1 / exp(-x)
  tanh(x):               sign(x) (1 - exp(-2|x|)) / (1 + exp(-2|x|))

Same grid, cut-off and interpolation as `fb_exp_negative` in scale_models/production_types.py (the full-scale models), written for any
argument type: the table and interpolation run in fixed internal types, the result is converted to the caller's type. Generic
(user-defined) tables are future work. Ops that take the option: activation (sigmoid, tanh, elu, selu, gelu, swish, softmax), mha, swa.
"""
import math
import re

MODES = ("hls", "table")
SUFFIX = "_exptable"
GUARD = "FB_EXP_TABLE"


def mode(op_info):
    m = op_info.get("exp_impl", "hls")
    if m not in MODES:
        raise ValueError(f"exp_impl must be one of {MODES}, got {m!r}")
    return m


def func_name(name, op_info):
    """Name of the op's function (call sites use this)."""
    return name + SUFFIX if mode(op_info) == "table" else name


def _rom():
    vals = [repr(math.exp(-i / 256)) for i in range(4097)]
    return ",\n".join("    " + ", ".join(vals[i:i + 8]) for i in range(0, len(vals), 8))


def support_code():
    return f"""
#ifndef {GUARD}
#define {GUARD}
// exp lookup table (op option "exp_impl": "table"; backends/exp_table.py): exp(-t) for t in [0,16) on a 1/256 grid, linear interpolation.
typedef ap_ufixed<45,13,AP_TRN,AP_SAT> fb_exp_arg_t;
typedef ap_ufixed<34,2> fb_exp_val_t;
static const ap_ufixed<32,1> FB_EXP_ROM[4097] = {{
{_rom()}
}};
static fb_exp_val_t fb_exp_neg_t(fb_exp_arg_t t) {{      // exp(-t), t >= 0
    if (t >= 16) return 0;
    ap_ufixed<53,13> c = t;
    c <<= 8;
    unsigned k = c.to_uint();
    ap_ufixed<40,0> f = c;                                  // fractional part
    ap_ufixed<41,1> one = 1;
    return fb_exp_val_t(FB_EXP_ROM[k] * (one - f) + FB_EXP_ROM[k + 1] * f);
}}
template <typename T> static T fb_exp(T x) {{
    if (x <= T(0)) return T(fb_exp_neg_t(fb_exp_arg_t(-x)));
    fb_exp_val_t e = fb_exp_neg_t(fb_exp_arg_t(x));
    if (e == 0) e = FB_EXP_ROM[4096];
    ap_ufixed<64,32,AP_TRN,AP_SAT> r = ap_ufixed<66,34>(1) / e;
    return T(r);
}}
template <typename T> static T fb_tanh(T x) {{
    bool neg = x < T(0);
    fb_exp_arg_t a = neg ? fb_exp_arg_t(-x) : fb_exp_arg_t(x);
    fb_exp_val_t e = fb_exp_neg_t(fb_exp_arg_t(a * 2));
    ap_ufixed<70,4> num = ap_ufixed<70,4>(1) - e;
    ap_ufixed<36,2> r = num / (fb_exp_val_t(1) + e);
    return neg ? T(-r) : T(r);
}}
#endif
"""


def apply(code, name, op_info):
    """-> (code, name): unchanged unless the op has exp_impl == "table"."""
    if mode(op_info) == "hls":
        return code, name
    if "hls::exp(" not in code and "hls::tanh(" not in code:
        raise ValueError(f"exp_impl 'table' on {op_info.get('func_name')} ({name}): the operator calls neither hls::exp nor hls::tanh")
    new = name + SUFFIX
    code = re.sub(r"\b%s\b" % re.escape(name), new, code)
    code = code.replace("hls::exp(", "fb_exp(").replace("hls::tanh(", "fb_tanh(")
    return support_code() + code, new
