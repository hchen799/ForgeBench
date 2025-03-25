
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
    conv_type, groups=None
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
    conv_args = (["BRAM_image_input", "BRAM_conv_weight", "BRAM_conv_bias", "BRAM_buffer_1", groups]
                 if conv_type == "group_conv2d" else
                 ["BRAM_image_input", "BRAM_conv_weight", "BRAM_conv_bias", "BRAM_buffer_1"])

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
            "func_info": ["activations_template.cpp", "relu"],
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
        args_str = quoted_list(op_data["args"]) if conv_type != "conv2d_group" else "[" + ", ".join(f'"{a}"' if not isinstance(a, int) else str(a) for a in op_data["args"]) + "]"
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
    "task": ["csim", "csynth", "cosim", "export_ip"],
    "data_type": "{DATA_TYPE}",
    "top_func_name": "top"
}}'''
    return text

def main():
    # Define parameter ranges (adjust as needed)
    C_IN_list              = [16, 32]
    H_IN_list              = [56, 28]
    W_IN_list              = [56, 28]
    C_OUT_list             = [16, 32]
    K_list                 = [1, 3]
    unroll_factor_cin_list = [1, 4]
    unroll_factor_cout_list= [1, 4]
    PAD_list               = [1]
    STRIDE_list            = [1]
    need_bias_list         = [True, False]

    # New sweeping parameters
    conv_type_list = ["conv2d", "group_conv2d"]
    groups_list    = [2, 4]  # Only used when conv_type is "conv2d_group"

    # Data type list
    data_type_list = ["ap_fixed<16,5>"]

    output_dir = "auto_generated_json"
    os.makedirs(output_dir, exist_ok=True)

    # Create base combinations for parameters except conv_type and groups
    base_combos = list(itertools.product(
        C_IN_list,
        H_IN_list,
        W_IN_list,
        C_OUT_list,
        K_list,
        unroll_factor_cin_list,
        unroll_factor_cout_list,
        PAD_list,
        STRIDE_list,
        need_bias_list
    ))

    # Now iterate over the base combos, conv_type, (groups if needed), and data_type.
    for (C_IN, H_IN, W_IN, C_OUT, K,
         unroll_factor_cin, unroll_factor_cout, PAD, STRIDE, need_bias) in base_combos:
        for conv_type in conv_type_list:
            if conv_type == "conv2d_group":
                for groups in groups_list:
                    for DATA_TYPE in data_type_list:
                        config_text = generate_config_text(
                            C_IN, H_IN, W_IN, C_OUT, K,
                            unroll_factor_cin, unroll_factor_cout,
                            DATA_TYPE, need_bias,
                            PAD, STRIDE,
                            conv_type, groups
                        )
                        safe_dtype = DATA_TYPE.replace('<','_').replace('>','_').replace(',','_')
                        filename = (
                            f"config_CIN{C_IN}_HIN{H_IN}_WIN{W_IN}_COUT{C_OUT}_K{K}_"
                            f"UFCIN{unroll_factor_cin}_UFCOU{unroll_factor_cout}_PAD{PAD}_STRIDE{STRIDE}_BIAS{need_bias}_"
                            f"{conv_type}_GROUPS{groups}_{safe_dtype}.json"
                        )
                        filepath = os.path.join(output_dir, filename)
                        with open(filepath, "w") as f:
                            f.write(config_text)
                        print(f"Generated {filepath}")
            else:
                groups = None
                for DATA_TYPE in data_type_list:
                    config_text = generate_config_text(
                        C_IN, H_IN, W_IN, C_OUT, K,
                        unroll_factor_cin, unroll_factor_cout,
                        DATA_TYPE, need_bias,
                        PAD, STRIDE,
                        conv_type, groups
                    )
                    safe_dtype = DATA_TYPE.replace('<','_').replace('>','_').replace(',','_')
                    filename = (
                        f"config_CIN{C_IN}_HIN{H_IN}_WIN{W_IN}_COUT{C_OUT}_K{K}_"
                        f"UFCIN{unroll_factor_cin}_UFCOU{unroll_factor_cout}_PAD{PAD}_STRIDE{STRIDE}_BIAS{need_bias}_"
                        f"{conv_type}_{safe_dtype}.json"
                    )
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, "w") as f:
                        f.write(config_text)
                    print(f"Generated {filepath}")

    # Compute total number of combos:
    # For each base combo, we have: 
    #   - 1 possibility if conv_type is "conv2d"
    #   - len(groups_list) possibilities if conv_type is "conv2d_group"
    # Total conv-type possibilities = 1 + len(groups_list)
    # And then multiplied by len(data_type_list)
    total_conv_variants = 1 + len(groups_list)
    total_combos = len(base_combos) * total_conv_variants * len(data_type_list)
    print("Total number of combos:", total_combos)

if __name__ == "__main__":
    main()
