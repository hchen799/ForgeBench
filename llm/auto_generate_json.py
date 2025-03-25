import itertools
import os

# Computing xABy + C

# option 1
# Compute t = xA, s = yB
# compute ts + C

# option 2
# Compue T = AB
# compute s = Ty + C
# compute xs

# x -- 1 X M
# A -- M X K
# B -- K X N
# y -- N X 1

def generate_config_text(
        SEQ_LEN, DIM_IN, NUM_HEADS, HEAD_DIM, NUM_GROUPS,
        DATA_TYPE, with_rope=False, norm_type='layer_norm', with_dropout=True, dropout_prob=0.1, seed=47, norm_eps=1e-2
):
    """
    Returns a string of the JSON config of an Attentoin/Norm block with the given parameters.
    """
    DIM_OUT = NUM_HEADS * HEAD_DIM

    # 1) Build the lines for brams
    brams = [
        {"name": "BRAM_attn_input",        "dims": [SEQ_LEN, DIM_IN]},
        {"name": "BRAM_WQ",        "dims": [DIM_IN, DIM_IN]},
        {"name": "BRAM_WK",        "dims": [DIM_IN, DIM_IN]},
        {"name": "BRAM_WV",        "dims": [DIM_IN, DIM_IN]},
        {"name": "BRAM_attn_output",        "dims": [SEQ_LEN, DIM_OUT]},
        {"name": "BRAM_norm_output",        "dims": [SEQ_LEN, DIM_OUT]},
    ]

    if with_dropout:
        brams.extend([
            {"name": "BRAM_dropout",        "dims": [SEQ_LEN, DIM_OUT]},
        ])
    
    brams_lines = []
    for i, b in enumerate(brams):
        comma = "," if i < len(brams) - 1 else ""
        brams_lines.append(
            f'        {{"name": "{b["name"]}", "dims": [{", ".join(str(x) for x in b["dims"])}]}}{comma}'
        )
    brams_str = "\n".join(brams_lines)

    # 2) Build the lines for drams
    drams = [
        {"name": "DRAM_attn_input",        "dims": [SEQ_LEN, DIM_IN], "bundle": "mem1"},
        {"name": "DRAM_WQ",        "dims": [DIM_IN, DIM_IN], "bundle": "mem2"},
        {"name": "DRAM_WK",        "dims": [DIM_IN, DIM_IN], "bundle": "mem3"},
        {"name": "DRAM_WV",        "dims": [DIM_IN, DIM_IN], "bundle": "mem4"},
        {"name": "DRAM_norm_output",        "dims": [SEQ_LEN, DIM_OUT], "bundle": "mem6"}
    ]

    drams_lines = []
    for i, d in enumerate(drams):
        comma = "," if i < len(drams) - 1 else ""
        dims_str = "[" + ", ".join(str(x) for x in d["dims"]) + "]"
        drams_lines.append(
            f'{{"name": "{d["name"]}", "dims": {dims_str}, "bundle": "{d["bundle"]}"}}{comma}'
        )
    drams_str = "\n".join(drams_lines)

    # 3) Build the lines for ops
    def one_line_list(lst):
        return "[" + ", ".join(str(v) for v in lst) + "]"

    def quoted_list(lst):
        return "[" + ", ".join(f'"{item}"' for item in lst) + "]"
    
    ops_order = [
        ("load_1", {"func_name": "load", "dims": [SEQ_LEN, DIM_IN], "args": ["DRAM_attn_input", "BRAM_attn_input"]}),
        ("load_2", {"func_name": "load", "dims": [DIM_IN, DIM_IN], "args": ["DRAM_WQ", "BRAM_WQ"]}),
        ("load_3", {"func_name": "load", "dims": [DIM_IN, DIM_IN], "args": ["DRAM_WK", "BRAM_WK"]}),
        ("load_4", {"func_name": "load", "dims": [DIM_IN, DIM_IN], "args": ["DRAM_WV", "BRAM_WV"]}),
        
        ("attn", {
            "func_name": "mha", 
            "dims": [SEQ_LEN, DIM_IN, NUM_HEADS, HEAD_DIM], 
            "args": ["BRAM_attn_input", "BRAM_WQ", "BRAM_WK", "BRAM_WV", "BRAM_attn_output", f"{NUM_GROUPS}"],
            "func_info": ["grouped_mha_rope_template.cpp", with_rope]
        }),
    ]

    input_bram = "BRAM_attn_output" 
    if with_dropout:
        ops_order.extend([
            ("dropout", {
                "func_name": "dropout", 
                "dims": [SEQ_LEN, DIM_OUT], 
                "args": ["BRAM_attn_output", "BRAM_dropout", f"{dropout_prob}", f"{seed}"],
                "func_info": ["dropout_template.cpp"]
            }),
        ])
        input_bram = "BRAM_dropout"
    
    if norm_type == 'layer_norm':
        ops_order.extend([
            ("norm", {
                "func_name": "layernorm", 
                "dims": [SEQ_LEN, DIM_OUT, norm_eps], 
                "args": [input_bram, "BRAM_norm_output"],
                "func_info": ["layer_norm_template.cpp"]
            }),
        ])
    elif norm_type == 'rms_norm':
        ops_order.extend([
            ("rms_norm", {
                "func_name": "rmsnorm", 
                "dims": [SEQ_LEN, DIM_OUT, norm_eps], 
                "args": [input_bram, "BRAM_norm_output"],
                "func_info": ["rms_norm_template.cpp"]
            }),
        ])
    
    ops_order.extend([
        ("store_1", {"func_name": "store", "dims": [SEQ_LEN, DIM_OUT], "args": ["BRAM_norm_output", "DRAM_norm_output"]}),
    ])

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
                elif isinstance(item, list):
                    fi_list.append(quoted_list(item))
                else:
                    fi_list.append(f'"{item}"')
            fi_str = "[" + ", ".join(fi_list) + "]"
            func_info_line = f'        "func_info": {fi_str},\n'
        dims_str = one_line_list(op_data["dims"])
        args_str = quoted_list(op_data["args"])
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
    "output_dram_names": ["DRAM_norm_output"],
    "FPGA_name": "xczu9eg-ffvb1156-2-e",
    "clock_period": 10,
    "task": ["csim", "csynth", "cosim", "export_ip"],
    "data_type": "{DATA_TYPE}",
    "top_func_name": "top"
}}'''
    return text


def main():
    # Define parameter ranges (adjust as needed)
    vals_seq_len = [8, 16, 32]
    vals_dim_in = [128, 256, 512]
    vals_num_heads = [8, 16, 32]
    vals_head_dim = [16, 32, 64]
    vals_num_groups = [1, 4]
    vals_with_rope = [True, False]
    vals_norm_type = ['layer_norm', 'rms_norm']
    vals_with_dropout = [True, False]
    vals_dropout_prob = [0.1, 0.5]

    # Static parameters
    data_type_list = ["ap_fixed<16,5>"]
    # seed_list = [47]    

    combinations = itertools.product(
        vals_seq_len, vals_dim_in, vals_num_heads, vals_head_dim, vals_num_groups,
        vals_with_rope, vals_norm_type, vals_with_dropout, data_type_list
    )

    output_dir = "auto_generated_configs"
    os.makedirs(output_dir, exist_ok=True)

    # Now iterate over the base combos, conv_type, (groups if needed), and data_type.
    for (seq, d_in, heads, d_head, groups, with_rope, norm, with_dropout, data_type) in combinations:
        if with_dropout:
            for dropout_prob in vals_dropout_prob:
                config_text = generate_config_text(
                    seq, d_in, heads, d_head, groups,
                    data_type_list[0], with_rope, norm, with_dropout, dropout_prob
                )
                naming_dtype = data_type.replace('<','_').replace('>','_').replace(',','_')
                filename = (
                    f"ATTN_config_{seq}_{d_in}_{heads}_{d_head}_{groups}_"
                    f"{norm}_{with_rope}_DROP{with_dropout}_{dropout_prob}_"
                    f"{naming_dtype}.json"
                )
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "w") as f:
                    f.write(config_text)
                print(f"Generated {filepath}")
        else:
            config_text = generate_config_text(
                seq, d_in, heads, d_head, groups,
                data_type_list[0], with_rope, norm, with_dropout
            )
            naming_dtype = data_type.replace('<','_').replace('>','_').replace(',','_')
            filename = (
                f"ATTN_config_{seq}_{d_in}_{heads}_{d_head}_{groups}_"
                f"{norm}_ROPE_{with_rope}_DROP_{with_dropout}_"
                f"{naming_dtype}.json"
            )
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                f.write(config_text)
            print(f"Generated {filepath}")

    num_combos = len(vals_seq_len) * len(vals_dim_in) * len(vals_num_heads) * len(vals_head_dim) * len(vals_num_groups) * len(vals_with_rope) * len(vals_norm_type) * len(data_type_list)
    num_combos *= (len(vals_dropout_prob) + 1)
    print("Total number of combos:", num_combos)

if __name__ == "__main__":
    main()
