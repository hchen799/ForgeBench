"""Generate the per-operator variant configs for the verification suite.

For each core operator whose *computation* depends on parameters, we emit one
config per compute-affecting combination, named ``<operator>__<variant>.json``
so verify_operators.py groups them under that operator and counts variants.

Compute-affecting axes (inline/unroll/loop-order are excluded -- they change the
hardware schedule, not the numerical result):
  - activation : the activation function (13 for gemm/llm, 15 for conv)
  - gemm/vmm/mmv/dot_product : bias {False, True}
  - conv       : bias {False, True} x grouped {conv2d, group_conv2d}
  - mha        : rope {False, True} x grouping {dense (groups==heads), grouped}
  - swa        : rope {False, True}
Operators with no compute-affecting parameter (batchnorm, maxpool,
adaptive_avgpool, matrix_add, elementwise_mult, layernorm, rmsnorm, dropout,
matmul) keep their single base config.

Run:  python -m verification.gen_variants        # writes into op_configs/<domain>/
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OP = os.path.join(HERE, "op_configs")

ACTS_13 = ["relu", "leaky_relu", "prelu", "rrelu", "thresholded_relu", "relu6",
           "sigmoid", "tanh", "elu", "selu", "gelu", "swish", "softmax"]
ACTS_CONV = ACTS_13 + ["hardsigmoid", "hardswish"]


def _load(domain, name):
    with open(os.path.join(OP, domain, name)) as f:
        return json.load(f)


def _write(domain, name, cfg):
    path = os.path.join(OP, domain, name)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return name


def _act_op_key(cfg):
    for k, v in cfg["ops"].items():
        if v.get("func_name") == "activation":
            return k
    raise KeyError("no activation op")


def gen_activation(domain, acts):
    base = _load(domain, "activation.json")
    key = _act_op_key(base)
    written = []
    for fn in acts:
        cfg = copy.deepcopy(base)
        cfg["ops"][key]["func_info"][1] = fn
        written.append(_write(domain, f"activation__{fn}.json", cfg))
    return written


def _find_op(cfg, func_name):
    for k, v in cfg["ops"].items():
        if v.get("func_name") == func_name:
            return k, v
    raise KeyError(func_name)


def gen_bias(domain, base_name, func_name):
    """gemm/vmm/mmv/dot_product: bias is func_info[-2] (…, with_bias, inline)."""
    base = _load(domain, base_name)
    op = os.path.splitext(base_name)[0]
    key, v = _find_op(base, func_name)
    written = []
    for bias in (False, True):
        cfg = copy.deepcopy(base)
        cfg["ops"][key]["func_info"][-2] = bias
        written.append(_write(domain, f"{op}__bias{int(bias)}.json", cfg))
    return written


def gen_conv(groups=2):
    """conv variants over dense/grouped x bias.

    func_info = [template, "conv2d"|"group_conv2d", with_bias]. For grouped conv
    the emitted C takes an extra ``int groups`` parameter, which the generator
    passes by appending the groups count to the op's args list (mirroring
    conv/auto_generate_json.py). The kernel stays [C_OUT][C_IN][K][K]; only the
    block-diagonal slice per group is used.
    """
    base = _load("conv", "conv.json")
    key, v = _find_op(base, "conv")
    # Base args are [in, weight, bias, out]; the bias BRAM is only passed when the
    # conv has a bias parameter (arg-count must match the emitted C signature).
    in_a, w_a, bias_a, out_a = v["args"]
    written = []
    for grouped in (False, True):
        for bias in (False, True):
            cfg = copy.deepcopy(base)
            op = cfg["ops"][key]
            op["func_info"][1] = "group_conv2d" if grouped else "conv2d"
            op["func_info"][2] = bias
            op["args"] = [in_a, w_a, bias_a, out_a] if bias else [in_a, w_a, out_a]
            if grouped:
                op["args"].append(groups)  # trailing int groups count
            tag = f"{'grouped' if grouped else 'dense'}_bias{int(bias)}"
            written.append(_write("conv", f"conv__{tag}.json", cfg))
    return written


def gen_attention(domain, base_name, func_name, do_grouping):
    """mha/swa: func_info = [template, use_rope]; groups is args[5] (a string)."""
    base = _load(domain, base_name)
    op = os.path.splitext(base_name)[0]
    key, v = _find_op(base, func_name)
    # dims = [S, DIN, NH, HD]; args[5] = groups count
    NH = base["ops"][key]["dims"][2]
    written = []
    rope_opts = (False, True)
    if do_grouping:
        # dense = groups == NH (one head per group); grouped = a proper divisor > 1
        grouped_g = 2 if NH % 2 == 0 and NH > 2 else max(1, NH // 2)
        group_opts = [("dense", NH), ("grouped", grouped_g)]
    else:
        group_opts = [(None, int(v["args"][5]))]
    for rope in rope_opts:
        for gname, g in group_opts:
            cfg = copy.deepcopy(base)
            cfg["ops"][key]["func_info"][1] = rope
            cfg["ops"][key]["args"][5] = str(g)
            parts = [f"rope{int(rope)}"] + ([gname] if gname else [])
            written.append(_write(domain, f"{op}__{'_'.join(parts)}.json", cfg))
    return written


def main():
    written = []
    # activations
    written += gen_activation("gemm", ACTS_13)
    written += gen_activation("llm", ACTS_13)
    written += gen_activation("conv", ACTS_CONV)
    # gemm-family bias
    written += gen_bias("gemm", "gemm.json", "gemm")
    written += gen_bias("gemm", "vmm.json", "vmm")
    written += gen_bias("gemm", "mmv.json", "mmv")
    written += gen_bias("gemm", "dot_product.json", "dot_product")
    # conv bias x grouped
    written += gen_conv()
    # attention
    written += gen_attention("llm", "mha.json", "mha", do_grouping=True)
    written += gen_attention("llm", "swa.json", "swa", do_grouping=False)
    print(f"wrote {len(written)} variant configs")
    for w in written:
        print("  ", w)


if __name__ == "__main__":
    main()
