
import itertools
import os

def conv_output_size(input_size, kernel_size, stride, padding):
    """
    Calculate the output feature map size for a convolutional layer.
    """
    return (input_size - kernel_size + 2 * padding) // stride + 1

def generate_config_text(
    C_IN, H_IN, W_IN, C_OUT, K,
    unroll_factor_cin, unroll_factor_cout,
    DATA_TYPE, need_bias,
    PAD, STRIDE,
    conv_type, activations, groups=None
):
    """
    Returns a string of the JSON config in exactly the format requested,
    computing H_OUT and W_OUT automatically and using conv_type (and groups if needed)
    in the conv_1 op.
    """
    H_OUT = conv_output_size(H_IN, K, STRIDE, PAD)
    W_OUT = conv_output_size(W_IN, K, STRIDE, PAD)
    input_partition_factor = unroll_factor_cin
    kernel_partition_factor1 = unroll_factor_cout
    kernel_partition_factor2 = unroll_factor_cin
    bias_partition_factor = unroll_factor_cout
    output_partition_factor = unroll_factor_cout

    # 1) Build the lines for brams
    brams = [
        {"name": "BRAM_image_input",        "dims": [C_IN, H_IN, W_IN]},
        {"name": "BRAM_conv_weight",        "dims": [C_OUT, C_IN, K, K]},
        {"name": "BRAM_conv_bias",          "dims": [C_OUT]},
        {"name": "BRAM_batch_norm_weights", "dims": [4, C_OUT]},
        {"name": "BRAM_buffer_1",           "dims": [C_OUT, H_OUT, W_OUT]},
        {"name": "BRAM_buffer_2",           "dims": [C_OUT, H_OUT, W_OUT]}
    ]
    brams_lines = []
    for i, b in enumerate(brams):
        comma = "," if i < len(brams) - 1 else ""
        brams_lines.append(
            f'        {{"name": "{b["name"]}", "dims": [{", ".join(str(x) for x in b["dims"])}]}}{comma}'
        )
    brams_str = "\n".join(brams_lines)

    # 2) Build the lines for drams
    drams = [
        {"name": "DRAM_image_input",        "dims": [C_IN, H_IN, W_IN],     "bundle": "mem1"},
        {"name": "DRAM_conv_weight",        "dims": [C_OUT, C_IN, K, K],   "bundle": "mem1"},
        {"name": "DRAM_conv_bias",          "dims": [C_OUT],               "bundle": "mem1"},
        {"name": "DRAM_batch_norm_weights", "dims": [4, C_OUT],            "bundle": "mem1"},
        {"name": "DRAM_image_output",       "dims": [C_OUT, H_OUT, W_OUT], "bundle": "mem2"}
    ]
    drams_lines = []
    for i, d in enumerate(drams):
        comma = "," if i < len(drams) - 1 else ""
        dims_str = "[" + ", ".join(str(x) for x in d["dims"]) + "]"
        drams_lines.append(
            f'        {{"name": "{d["name"]}", "dims": {dims_str}, "bundle": "{d["bundle"]}"}}{comma}'
        )
    drams_str = "\n".join(drams_lines)

    # 3) Build ops. Utility functions:
    def one_line_list(lst):
        return "[" + ", ".join(str(v) for v in lst) + "]"

    def quoted_list(lst):
        return "[" + ", ".join(f'"{item}"' for item in lst) + "]"

    # Define ops in order. Update conv_1 per conv_type.
    # The conv/group_conv function only has a bias parameter when need_bias is set,
    # so the bias BRAM must be omitted from the call args otherwise (arg-count match).
    conv_args = ["BRAM_image_input", "BRAM_conv_weight"]
    if need_bias:
        conv_args.append("BRAM_conv_bias")
    conv_args.append("BRAM_buffer_1")
    if conv_type == "group_conv2d":
        conv_args.append(groups)

    ops_order = [
        ("load_1", {
            "func_name": "load",
            "dims": [C_IN, H_IN, W_IN],
            "args": ["DRAM_image_input", "BRAM_image_input"]
        }),
        ("load_2", {
            "func_name": "load",
            "dims": [C_OUT, C_IN, K, K],
            "args": ["DRAM_conv_weight", "BRAM_conv_weight"]
        }),
        ("load_3", {
            "func_name": "load",
            "dims": [C_OUT],
            "args": ["DRAM_conv_bias", "BRAM_conv_bias"]
        }),
        ("load_4", {
            "func_name": "load",
            "dims": [4, C_OUT],
            "args": ["DRAM_batch_norm_weights", "BRAM_batch_norm_weights"]
        }),
        ("conv_1", {
            "func_name": "conv",
            "dims": [
                C_IN, C_OUT, H_IN, W_IN, H_OUT, W_OUT, K,
                PAD, STRIDE,
                input_partition_factor, kernel_partition_factor1, kernel_partition_factor2,
                bias_partition_factor, output_partition_factor,
                unroll_factor_cin, unroll_factor_cout
            ],
            "func_info": ["conv_template.cpp", conv_type, need_bias],
            "args": conv_args
        }),
        ("batchnorm_1", {
            "func_name": "batchnorm",
            "dims": [C_OUT, H_OUT, W_OUT],
            "func_info": ["batch_norm_template.cpp", 1e-5],
            "args": ["BRAM_buffer_1", "BRAM_batch_norm_weights", "BRAM_buffer_2"]
        }),
        ("activation_1", {
            "func_name": "activation",
            "dims": [C_OUT, H_OUT, W_OUT],
            "func_info": ["activations_template.cpp", activations],
            "args": ["BRAM_buffer_2", "BRAM_buffer_1"]
        }),
        ("store", {
            "func_name": "store",
            "dims": [C_OUT, H_OUT, W_OUT],
            "args": ["BRAM_buffer_1", "DRAM_image_output"]
        })
    ]

    ops_lines = []
    for i, (op_name, op_data) in enumerate(ops_order):
        comma = "," if i < len(ops_order) - 1 else ""
        func_info_line = ""
        if "func_info" in op_data:
            fi_list = []
            for item in op_data["func_info"]:
                if isinstance(item, bool):
                    fi_list.append("true" if item else "false")
                elif isinstance(item, float):
                    fi_list.append(str(item))
                else:
                    fi_list.append(f'"{item}"')
            fi_str = "[" + ", ".join(fi_list) + "]"
            func_info_line = f'        "func_info": {fi_str},\n'
        dims_str = one_line_list(op_data["dims"])
        args_str = quoted_list(op_data["args"]) if conv_type != "group_conv2d" else "[" + ", ".join(f'"{a}"' if not isinstance(a, int) else str(a) for a in op_data["args"]) + "]"
        if func_info_line:
            op_block = (
f'''{{
        "func_name": "{op_data["func_name"]}",
        "dims": {dims_str},
{func_info_line}        "args": {args_str}
    }}'''
            )
        else:
            op_block = (
f'''{{
        "func_name": "{op_data["func_name"]}",
        "dims": {dims_str},
        "args": {args_str}
    }}'''
            )
        ops_lines.append(
            f'        "{op_name}": {op_block}{comma}'
        )
    ops_str = "\n".join(ops_lines)

    text = f'''{{
    "brams": [
{brams_str}
    ],
    "drams": [
{drams_str}
    ],
    "ops": {{
{ops_str}
    }},
    "output_dram_names": ["DRAM_image_output"],
    "FPGA_name": "xczu9eg-ffvb1156-2-e",
    "clock_period": 10,
    "task": ["csynth"],
    "data_type": "{DATA_TYPE}",
    "top_func_name": "top"
}}'''
    return text

# ---- Sweep specification (single source of truth; imported by manifest/ and paper_artifacts/) ----
SWEEP = {
    "C_IN": [16, 32],
    "H_IN": [56, 28],
    "W_IN": [56, 28],
    "C_OUT": [16, 32],
    "K": [1, 3],
    "unroll_cin": [1, 4, 8],
    "unroll_cout": [1, 4, 8],
    "pad": [1],
    "stride": [1],
    "with_bias": [True, False],
    "activation": ['relu', 'sigmoid', 'tanh'],
    "conv_type": ["conv2d", "group_conv2d"],
    "groups": [2, 4],  # only used when conv_type is "group_conv2d"
    "data_type": ["ap_fixed<16,5>"],
}


def iter_params():
    """Yield one dict of sweep parameters per design, in generation order."""
    for (c_in, h_in, w_in, c_out, k, uf_cin, uf_cout, pad, stride, act, bias) in itertools.product(
            SWEEP["C_IN"], SWEEP["H_IN"], SWEEP["W_IN"], SWEEP["C_OUT"], SWEEP["K"],
            SWEEP["unroll_cin"], SWEEP["unroll_cout"], SWEEP["pad"], SWEEP["stride"],
            SWEEP["activation"], SWEEP["with_bias"]):
        for conv_type in SWEEP["conv_type"]:
            for groups in (SWEEP["groups"] if conv_type == "group_conv2d" else [None]):
                for dtype in SWEEP["data_type"]:
                    yield {"C_IN": c_in, "H_IN": h_in, "W_IN": w_in, "C_OUT": c_out, "K": k,
                           "unroll_cin": uf_cin, "unroll_cout": uf_cout, "pad": pad, "stride": stride,
                           "with_bias": bias, "conv_type": conv_type, "groups": groups,
                           "activation": act, "data_type": dtype}


def config_stem(p):
    dtype = p["data_type"].replace('<', '_').replace('>', '_').replace(',', '_')
    g = f"_GROUPS{p['groups']}" if p["conv_type"] == "group_conv2d" else ""
    return (f"config_CIN{p['C_IN']}_HIN{p['H_IN']}_WIN{p['W_IN']}_COUT{p['C_OUT']}_K{p['K']}_"
            f"UFCIN{p['unroll_cin']}_UFCOU{p['unroll_cout']}_PAD{p['pad']}_STRIDE{p['stride']}_BIAS{p['with_bias']}_"
            f"{p['conv_type']}{g}_ACTIVATION{p['activation']}_{dtype}")


def build_config_text(p):
    return generate_config_text(
        p["C_IN"], p["H_IN"], p["W_IN"], p["C_OUT"], p["K"], p["unroll_cin"], p["unroll_cout"],
        p["data_type"], p["with_bias"], p["pad"], p["stride"], p["conv_type"], p["activation"], p["groups"])


def main():
    output_dir = "auto_generated_configs"
    os.makedirs(output_dir, exist_ok=True)
    n = 0
    for p in iter_params():
        filepath = os.path.join(output_dir, config_stem(p) + ".json")
        with open(filepath, "w") as f:
            f.write(build_config_text(p))
        print(f"Generated {filepath}")
        n += 1
    print("Total number of combos:", n)

if __name__ == "__main__":
    main()
