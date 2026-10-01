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
        DATA_TYPE, with_rope=False, norm_type='layer_norm', with_dropout=True, dropout_prob=0.1, seed=47, norm_eps=1e-2,
        hd_unroll=None
):
    """
    Returns a string of the JSON config of an Attentoin/Norm block with the given parameters.
    """
    DIM_OUT = NUM_HEADS * HEAD_DIM
    # W_q/W_k/W_v are [DIM_OUT][DIM_IN] (see grouped_mha_rope_template.cpp). Declaring them
    # [DIM_IN][DIM_IN] made the kernel read out of bounds whenever NUM_HEADS*HEAD_DIM > DIM_IN.

    # 1) Build the lines for brams
    brams = [
        {"name": "BRAM_attn_input",        "dims": [SEQ_LEN, DIM_IN]},
        {"name": "BRAM_WQ",        "dims": [DIM_OUT, DIM_IN]},
        {"name": "BRAM_WK",        "dims": [DIM_OUT, DIM_IN]},
        {"name": "BRAM_WV",        "dims": [DIM_OUT, DIM_IN]},
        {"name": "BRAM_attn_output",        "dims": [SEQ_LEN, DIM_OUT]},
        {"name": "BRAM_norm_output",        "dims": [SEQ_LEN, DIM_OUT]},
    ]

    if with_dropout:
        brams.extend([
            {"name": "BRAM_dropout",        "dims": [SEQ_LEN, DIM_OUT]},
        ])

    # Norm scale/shift params: layer_norm needs gamma+beta, rms_norm needs gamma only.
    brams.append({"name": "BRAM_norm_gamma", "dims": [DIM_OUT]})
    if norm_type == 'layer_norm':
        brams.append({"name": "BRAM_norm_beta", "dims": [DIM_OUT]})

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
        {"name": "DRAM_WQ",        "dims": [DIM_OUT, DIM_IN], "bundle": "mem2"},
        {"name": "DRAM_WK",        "dims": [DIM_OUT, DIM_IN], "bundle": "mem3"},
        {"name": "DRAM_WV",        "dims": [DIM_OUT, DIM_IN], "bundle": "mem4"},
        {"name": "DRAM_norm_output",        "dims": [SEQ_LEN, DIM_OUT], "bundle": "mem6"}
    ]

    # Off-chip storage for the norm scale/shift parameters (loaded into BRAM below).
    drams.append({"name": "DRAM_norm_gamma", "dims": [DIM_OUT], "bundle": "mem5"})
    if norm_type == 'layer_norm':
        drams.append({"name": "DRAM_norm_beta", "dims": [DIM_OUT], "bundle": "mem7"})

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
        ("load_2", {"func_name": "load", "dims": [DIM_OUT, DIM_IN], "args": ["DRAM_WQ", "BRAM_WQ"]}),
        ("load_3", {"func_name": "load", "dims": [DIM_OUT, DIM_IN], "args": ["DRAM_WK", "BRAM_WK"]}),
        ("load_4", {"func_name": "load", "dims": [DIM_OUT, DIM_IN], "args": ["DRAM_WV", "BRAM_WV"]}),
        
        ("attn", {
            "func_name": "mha", 
            "dims": [SEQ_LEN, DIM_IN, NUM_HEADS, HEAD_DIM], 
            "args": ["BRAM_attn_input", "BRAM_WQ", "BRAM_WK", "BRAM_WV", "BRAM_attn_output", f"{NUM_GROUPS}"],
            "func_info": ["grouped_mha_rope_template.cpp", with_rope] + ([] if hd_unroll is None else [hd_unroll])
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
    
    # Load the norm scale/shift params from DRAM before the norm op.
    ops_order.append(
        ("load_norm_gamma", {"func_name": "load", "dims": [DIM_OUT],
                             "args": ["DRAM_norm_gamma", "BRAM_norm_gamma"]})
    )
    if norm_type == 'layer_norm':
        ops_order.append(
            ("load_norm_beta", {"func_name": "load", "dims": [DIM_OUT],
                                "args": ["DRAM_norm_beta", "BRAM_norm_beta"]})
        )
        ops_order.extend([
            ("norm", {
                "func_name": "layernorm",
                "dims": [SEQ_LEN, DIM_OUT, norm_eps],
                "args": [input_bram, "BRAM_norm_gamma", "BRAM_norm_beta", "BRAM_norm_output"],
                "func_info": ["layer_norm_template.cpp"]
            }),
        ])
    elif norm_type == 'rms_norm':
        ops_order.extend([
            ("rms_norm", {
                "func_name": "rmsnorm",
                "dims": [SEQ_LEN, DIM_OUT, norm_eps],
                "args": [input_bram, "BRAM_norm_gamma", "BRAM_norm_output"],
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
                elif isinstance(item, (int, float)):
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
    "task": ["csynth"],
    "data_type": "{DATA_TYPE}",
    "top_func_name": "top"
}}'''
    return text


# ---- Sweep specification (single source of truth; imported by manifest/ and paper_artifacts/) ----
# hd_unroll: head-dimension unroll/partition factor of the attention kernel (see
# generate_grouped_mha_code). It replaces the former dropout setting, which is an identity at
# inference and therefore did not change the design (see docs/revision_r2/CHANGELOG_R2.md).
SWEEP = {
    "seq_len": [8, 16, 32],
    "dim_in": [128, 256, 512],
    "num_heads": [8, 16, 32],
    "head_dim": [16, 32, 64],
    "num_groups": [1, 2, 4],
    "with_rope": [True, False],
    "norm_type": ['layer_norm', 'rms_norm'],
    "hd_unroll": [1, 2, 4, 8],
    "data_type": ["ap_fixed<16,5>"],
}


def iter_params():
    """Yield one dict of sweep parameters per design, in generation order."""
    for (seq, d_in, heads, d_head, groups, rope, norm, uhd, dtype) in itertools.product(
            SWEEP["seq_len"], SWEEP["dim_in"], SWEEP["num_heads"], SWEEP["head_dim"], SWEEP["num_groups"],
            SWEEP["with_rope"], SWEEP["norm_type"], SWEEP["hd_unroll"], SWEEP["data_type"]):
        yield {"seq_len": seq, "dim_in": d_in, "num_heads": heads, "head_dim": d_head, "num_groups": groups,
               "with_rope": rope, "norm_type": norm, "hd_unroll": uhd, "data_type": dtype}


def config_stem(p):
    dtype = p["data_type"].replace('<', '_').replace('>', '_').replace(',', '_')
    return (f"ATTN_config_S{p['seq_len']}_D{p['dim_in']}_H{p['num_heads']}_HD{p['head_dim']}_G{p['num_groups']}_"
            f"{p['norm_type']}_ROPE{p['with_rope']}_UHD{p['hd_unroll']}_{dtype}")


def build_config_text(p):
    return generate_config_text(
        p["seq_len"], p["dim_in"], p["num_heads"], p["head_dim"], p["num_groups"], p["data_type"],
        p["with_rope"], p["norm_type"], with_dropout=False, hd_unroll=p["hd_unroll"])


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
    expected = 1
    for v in SWEEP.values():
        expected *= len(v)
    assert n == expected
    print("Total number of combos:", n)

if __name__ == "__main__":
    main()
