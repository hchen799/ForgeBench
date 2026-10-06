"""Typed storage and arithmetic for the production JSON-to-HLS emitter.

No model graph or verification kernel is generated here. Buffer roles select
integer controls and persistent wide reductions; the ordered JSON ops remain
the source of the accelerator's schedule.
"""
import copy
import math
import re


REVISION = 2


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
    for b in buffers.values():
        if b.get("dtype", "data_t") not in ("data_t", "acc_t", "int32_t"):
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


def numeric_support(data_type):
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
    table = [math.floor(math.exp(-i / 256) * (1 << frac) + 0.5) for i in range(4097)]
    entries = ",\n".join(
        ", ".join(str(v) + "ULL" for v in table[i : i + 16])
        for i in range(0, len(table), 16)
    )
    return f"""// Production arithmetic revision {REVISION}: persistent wide sums.
typedef ap_fixed<{width},{integer},AP_TRN,AP_WRAP> acc_t;
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
