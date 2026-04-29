import argparse
import itertools
import os

# Generating a LLAMA 3.1B Implementation
# 32 Decoder Layers

# Each Decoder Layer
###### RMS Norm 
###### Grouped-Query Attention
###### Residual Add
###### RMSNorm
###### SWIGLU
###### Residual Add

def generate_llama_decode_block (mod_name, bram_in,tmp_bram_1, tmp_bram_2, tmp_bram_3,
bram_rms_norm_weights_1, bram_weights_q, bram_weights_k, bram_weights_v, bram_rms_norm_weights_2,
bram_mm_weights_1, bram_mm_weights_2, bram_mm_weights_3, bram_out,
SEQ_LEN,
):

    DIM_IN = 4096
    NUM_HEADS = 32
    HEAD_DIM = 128
    WITH_ROPE = True
    NUM_GROUPS = 8

    SWIGLU_DIM = 14336

    NORM_EPS = 1e-6

    rms_norm_1 = (f"rms1_{mod_name}", {
                "func_name": "rmsnorm", 
                "dims": [SEQ_LEN, DIM_IN, NORM_EPS], 
                "args": [bram_in, bram_rms_norm_weights_1, bram_in],
                "func_info": ["rms_norm_template.cpp"]
        })

    attn_block = (
        f"attn_{mod_name}", {
                "func_name": "mha", 
                "dims": [SEQ_LEN, DIM_IN, NUM_HEADS, HEAD_DIM], 
                "args": [bram_in, bram_weights_q, bram_weights_k, bram_weights_v, tmp_bram_1, f"{NUM_GROUPS}"],
                "func_info": ["grouped_mha_rope_template.cpp", WITH_ROPE]
        })

    res_add = (
        f"res_add_{mod_name}", {
                "func_name": "matrix_add", 
                "dims": [SEQ_LEN, DIM_IN], 
                "args": [tmp_bram_1, bram_in, bram_in],
                "func_info": ["matrix_add_template.cpp"] 
        })

    rms_norm_2 = (
        f"rms2_{mod_name}", {
                "func_name": "rmsnorm", 
                "dims": [SEQ_LEN, DIM_IN, NORM_EPS], 
                "args": [bram_in, bram_rms_norm_weights_2, bram_in],
                "func_info": ["rms_norm_template.cpp"]
        })

    # SwiGlu MLP
    matmul_1 = ( 
        f"matmul1_{mod_name}", {
            "func_name": "matmul", 
            "dims": [SEQ_LEN, DIM_IN, SWIGLU_DIM], 
            "args": [bram_in, bram_mm_weights_1, tmp_bram_2],
            "func_info": ["matmul_template.cpp", False], 
        })

    matmul_2 = ( 
        f"matmul2_{mod_name}", {
            "func_name": "matmul", 
            "dims": [SEQ_LEN, DIM_IN, SWIGLU_DIM], 
            "args": [bram_in, bram_mm_weights_2, tmp_bram_3],
            "func_info": ["matmul_template.cpp", False], 
        })

    SiLU = (
        f"silu_{mod_name}", {
            "func_name": "activation", 
            "dims": [SEQ_LEN, SWIGLU_DIM], 
            "args": [tmp_bram_2, tmp_bram_2],
            "func_info": ["activation_template.cpp", "swish"], 
        })


    elementwise_mult = (
        f"emult_{mod_name}", {
            "func_name": "elementwise_mult", 
            "dims": [SEQ_LEN, SWIGLU_DIM], 
            "args": [tmp_bram_2, tmp_bram_3, tmp_bram_3],
            "func_info": ["elementwise_mult_template.cpp"], 
        })

    matmul_3 = (
        f"matmul3_{mod_name}", {
            "func_name": "matmul", 
            "dims": [SEQ_LEN, SWIGLU_DIM, DIM_IN], 
            "args": [tmp_bram_3, bram_mm_weights_3, tmp_bram_1],
            "func_info": ["matmul_template.cpp", False], 
        })

    res_add_2 = (
        f"res_add2_{mod_name}", {
                "func_name": "matrix_add",
                "dims": [SEQ_LEN, DIM_IN],
                "args": [tmp_bram_1, bram_in, bram_out],
                "func_info": ["matrix_add_template.cpp"] 
        })


    return [
            rms_norm_1,
            attn_block,
            res_add,
            rms_norm_2,
            matmul_1,
            SiLU,
            elementwise_mult,
            matmul_3,
            res_add_2
        ]

def generate_llama_architecture(seq_len=2048):
    """
    Generates the full architecture for LLAMA 3.1B decoder with 32 layers.
    Returns drams, brams, and ops for the complete 32-layer decoder.
    
    Args:
        seq_len (int): Sequence length (default: 2048)
    """
    SEQ_LEN = seq_len
    DIM_IN = 4096
    NUM_HEADS = 32
    HEAD_DIM = 128

    SWIGLU_DIM = 14336

    brams = [
        {"name": "BRAM_input_0", "dims": [SEQ_LEN, DIM_IN]},
        {"name": "BRAM_tmp_1", "dims": [SEQ_LEN, DIM_IN]},
        {"name": "BRAM_tmp_2", "dims": [SEQ_LEN, SWIGLU_DIM]},
        {"name": "BRAM_tmp_3", "dims": [SEQ_LEN, SWIGLU_DIM]},
    ]
    
    drams = [
        {"name": "DRAM_input", "dims": [SEQ_LEN, DIM_IN], "bundle": "mem_input"},
    ]

    for layer_idx in range(32):
        brams.extend([
            {"name": f"BRAM_input_{layer_idx + 1}", "dims": [SEQ_LEN, DIM_IN]},
            {"name": f"BRAM_rms_norm_w1_{layer_idx}", "dims": [DIM_IN]},
            {"name": f"BRAM_rms_norm_w2_{layer_idx}", "dims": [DIM_IN]},
            {"name": f"BRAM_wq_{layer_idx}", "dims": [DIM_IN, DIM_IN]},
            {"name": f"BRAM_wk_{layer_idx}", "dims": [DIM_IN, DIM_IN]},
            {"name": f"BRAM_wv_{layer_idx}", "dims": [DIM_IN, DIM_IN]},
            {"name": f"BRAM_mm_w1_{layer_idx}", "dims": [SWIGLU_DIM, DIM_IN]},
            {"name": f"BRAM_mm_w2_{layer_idx}", "dims": [SWIGLU_DIM, DIM_IN]},
            {"name": f"BRAM_mm_w3_{layer_idx}", "dims": [DIM_IN, SWIGLU_DIM]},
        ])

        drams.extend([
            {"name": f"DRAM_rms_norm_w1_{layer_idx}", "dims": [DIM_IN], "bundle": f"mem_rms_norm_w1_{layer_idx}"},
            {"name": f"DRAM_rms_norm_w2_{layer_idx}", "dims": [DIM_IN], "bundle": f"mem_rms_norm_w2_{layer_idx}"},
            {"name": f"DRAM_wq_{layer_idx}", "dims": [DIM_IN, DIM_IN], "bundle": f"mem_wq_{layer_idx}"},
            {"name": f"DRAM_wk_{layer_idx}", "dims": [DIM_IN, DIM_IN], "bundle": f"mem_wk_{layer_idx}"},
            {"name": f"DRAM_wv_{layer_idx}", "dims": [DIM_IN, DIM_IN], "bundle": f"mem_wv_{layer_idx}"},
            {"name": f"DRAM_mm_w1_{layer_idx}", "dims": [SWIGLU_DIM, DIM_IN], "bundle": f"mem_mm_w1_{layer_idx}"},
            {"name": f"DRAM_mm_w2_{layer_idx}", "dims": [SWIGLU_DIM, DIM_IN], "bundle": f"mem_mm_w2_{layer_idx}"},
            {"name": f"DRAM_mm_w3_{layer_idx}", "dims": [DIM_IN, SWIGLU_DIM], "bundle": f"mem_mm_w3_{layer_idx}"},
        ])
    
    ops = []

    # Load initial input
    ops.append((
        f"load_input",
        {"func_name": "load", "dims": [SEQ_LEN, DIM_IN], "args": ["DRAM_input", "BRAM_input_0"]}
    ))

    for layer_idx in range(32):
        ops.extend([
            (
                f"load_rms_norm_w1_{layer_idx}",
                {"func_name": "load", "dims": [DIM_IN], "args": [f"DRAM_rms_norm_w1_{layer_idx}", f"BRAM_rms_norm_w1_{layer_idx}"]}
            ),
            (
                f"load_rms_norm_w2_{layer_idx}",
                {"func_name": "load", "dims": [DIM_IN], "args": [f"DRAM_rms_norm_w2_{layer_idx}", f"BRAM_rms_norm_w2_{layer_idx}"]}
            ),
            (
                f"load_wq_{layer_idx}",
                {"func_name": "load", "dims": [DIM_IN, DIM_IN], "args": [f"DRAM_wq_{layer_idx}", f"BRAM_wq_{layer_idx}"]}
            ),
            (
                f"load_wk_{layer_idx}",
                {"func_name": "load", "dims": [DIM_IN, DIM_IN], "args": [f"DRAM_wk_{layer_idx}", f"BRAM_wk_{layer_idx}"]}
            ),
            (
                f"load_wv_{layer_idx}",
                {"func_name": "load", "dims": [DIM_IN, DIM_IN], "args": [f"DRAM_wv_{layer_idx}", f"BRAM_wv_{layer_idx}"]}
            ),
            (
                f"load_mm_w1_{layer_idx}",
                {"func_name": "load", "dims": [SWIGLU_DIM, DIM_IN], "args": [f"DRAM_mm_w1_{layer_idx}", f"BRAM_mm_w1_{layer_idx}"]}
            ),
            (
                f"load_mm_w2_{layer_idx}",
                {"func_name": "load", "dims": [SWIGLU_DIM, DIM_IN], "args": [f"DRAM_mm_w2_{layer_idx}", f"BRAM_mm_w2_{layer_idx}"]}
            ),
            (
                f"load_mm_w3_{layer_idx}",
                {"func_name": "load", "dims": [DIM_IN, SWIGLU_DIM], "args": [f"DRAM_mm_w3_{layer_idx}", f"BRAM_mm_w3_{layer_idx}"]}
            ),
        ])

    # Process 32 decoder layers
    for layer_idx in range(32):
        mod_name = f"layer_{layer_idx}"
        
        bram_in = f"BRAM_input_{layer_idx}"
        tmp_bram_1 = f"BRAM_tmp_1"
        tmp_bram_2 = f"BRAM_tmp_2"
        tmp_bram_3 = f"BRAM_tmp_3"
        bram_rms_norm_weights_1 = f"BRAM_rms_norm_w1_{layer_idx}"
        bram_rms_norm_weights_2 = f"BRAM_rms_norm_w2_{layer_idx}"
        bram_weights_q = f"BRAM_wq_{layer_idx}"
        bram_weights_k = f"BRAM_wk_{layer_idx}"
        bram_weights_v = f"BRAM_wv_{layer_idx}"
        bram_mm_weights_1 = f"BRAM_mm_w1_{layer_idx}"
        bram_mm_weights_2 = f"BRAM_mm_w2_{layer_idx}"
        bram_mm_weights_3 = f"BRAM_mm_w3_{layer_idx}"
        bram_out = f"BRAM_input_{layer_idx + 1}"

        # Generate decoder block using the function
        block_ops = generate_llama_decode_block(
            mod_name, bram_in, tmp_bram_1, tmp_bram_2, tmp_bram_3,
            bram_rms_norm_weights_1,
            bram_weights_q, bram_weights_k, bram_weights_v,
            bram_rms_norm_weights_2, bram_mm_weights_1, bram_mm_weights_2,
            bram_mm_weights_3, bram_out, SEQ_LEN
        )
        
        ops.extend(block_ops)

    # Store final output to DRAM
    drams.append({"name": "DRAM_output", "dims": [SEQ_LEN, DIM_IN], "bundle": "mem_output"})
    ops.append((f"store_output",
        {"func_name": "store", "dims": [SEQ_LEN, DIM_IN], "args": [f"BRAM_input_32", "DRAM_output"]}))

    return brams, drams, ops


def generate_llama_config_text(seq_len=2048, data_type="ap_fixed<16,5>"):
    """
    Returns a string of the JSON config of the LLAMA with the given parameters.
    """
    
    # brams, drams, ops = generate_llama_architecture_block_size(block_size, seq_len)
    brams, drams, ops = generate_llama_architecture(seq_len)
    # brams, drams, ops = generate_llama_architecture_single_block(seq_len)
    
    brams_lines = []
    for i, b in enumerate(brams):
        comma = "," if i < len(brams) - 1 else ""
        brams_lines.append(
            f'        {{"name": "{b["name"]}", "dims": [{", ".join(str(x) for x in b["dims"])}]}}{comma}'
        )
    brams_str = "\n".join(brams_lines)

    drams_lines = []
    for i, d in enumerate(drams):
        comma = "," if i < len(drams) - 1 else ""
        dims_str = "[" + ", ".join(str(x) for x in d["dims"]) + "]"
        drams_lines.append(
            f'{{"name": "{d["name"]}", "dims": {dims_str}, "bundle": "{d["bundle"]}"}}{comma}'
        )
    drams_str = "\n".join(drams_lines)

    def one_line_list(lst):
        return "[" + ", ".join(str(v) for v in lst) + "]"

    def quoted_list(lst):
        return "[" + ", ".join(f'"{item}"' for item in lst) + "]"

    ops_lines = []
    for i, x in enumerate(ops):
        op_name, op_data = x
        comma = "," if i < len(ops) - 1 else ""
        func_info_line = ""
        if "func_info" in op_data:
            fi_list = []
            for item in op_data["func_info"]:
                if isinstance(item, bool):
                    fi_list.append("true" if item else "0")
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
        "brams": [{brams_str}],
        "drams": [{drams_str}],
        "ops": {{{ops_str}}},
        "output_dram_names": ["DRAM_output"],
        "FPGA_name": "xczu9eg-ffvb1156-2-e",
        "clock_period": 10,
        "task": ["csynth"],
        "data_type": "{data_type}",
        "top_func_name": "top"
    }}'''
    return text

def generate_longformer_block(
    mod_name,
    bram_in,
    tmp_bram,
    
    tmp_bram_2,

    ln1_w_g,
    ln1_w_b,

    ln2_w_g,
    ln2_w_b,

    wq,
    wk,
    wv,
    mlp_w1,
    mlp_w2,
    bram_out,
    SEQ_LEN,
    DIM=768,
    NUM_HEADS=12,
    WINDOW=512,
):

    HEAD_DIM = DIM // NUM_HEADS
    NORM_EPS = 1e-6

    return [
        # LN 1
        (f"ln1_{mod_name}", {
            "func_name": "layernorm",
            "dims": [SEQ_LEN, DIM, NORM_EPS],
            "args": [bram_in, ln1_w_g, ln1_w_b, bram_in],
            "func_info": ["layer_norm_template.cpp"]
        }),

        # Sliding Window Attention
        (f"attn_{mod_name}", {
            "func_name": "swa",
            "dims": [SEQ_LEN, DIM, NUM_HEADS, HEAD_DIM],
            "args": [bram_in, wq, wk, wv, tmp_bram, WINDOW],
            "func_info": ["sliding_window_attention_template.cpp", False]
        }),

        # Residual
        (f"res1_{mod_name}", {
            "func_name": "matrix_add",
            "dims": [SEQ_LEN, DIM],
            "args": [tmp_bram, bram_in, bram_in],
            "func_info": ["matrix_add_template.cpp"] 

        }),

        # LN 2
        (f"ln2_{mod_name}", {
            "func_name": "layernorm",
            "dims": [SEQ_LEN, DIM, NORM_EPS],
            "args": [bram_in, ln2_w_g, ln2_w_b, bram_in],
            "func_info": ["layer_norm_template.cpp"]
        }),

        # MLP
        (f"mlp_fc1_{mod_name}", {
            "func_name": "matmul",
            "dims": [SEQ_LEN, DIM, 4 * DIM],
            "args": [bram_in, mlp_w1, tmp_bram_2],
            "func_info": ["matmul_template.cpp", False], 
        }),

        (f"gelu_{mod_name}", {
            "func_name": "activation",
            "dims": [SEQ_LEN, 4 * DIM],
            "args": [tmp_bram_2, tmp_bram_2],
            "func_info": ["activation_template.cpp", "gelu"]
        }),

        (f"mlp_fc2_{mod_name}", {
            "func_name": "matmul",
            "dims": [SEQ_LEN, 4 * DIM, DIM],
            "args": [tmp_bram_2, mlp_w2, tmp_bram],
            "func_info": ["matmul_template.cpp", False], 
        }),

        (f"res2_{mod_name}", {
            "func_name": "matrix_add",
            "dims": [SEQ_LEN, DIM],
            "args": [tmp_bram, bram_in, bram_out],
            "func_info": ["matrix_add_template.cpp"] 
        }),
    ]

def generate_longformer_architecture(seq_len=4096):
    DIM = 768
    MLP_DIM = 3072
    NUM_LAYERS = 12

    brams = [{"name": "BRAM_input_0", "dims": [seq_len, DIM]}]
    drams = [{"name": "DRAM_input", "dims": [seq_len, DIM], "bundle": "mem_input"}]

    for i in range(NUM_LAYERS):
        brams.extend([
            {"name": f"BRAM_input_{i+1}", "dims": [seq_len, DIM]},
            {"name": f"BRAM_tmp_{i}", "dims": [seq_len, DIM]},
            {"name": f"BRAM_tmp_2_{i}", "dims": [seq_len, 4 * DIM]},
            {"name": f"BRAM_ln1_w_g{i}", "dims": [DIM]},
            {"name": f"BRAM_ln1_w_b{i}", "dims": [DIM]},
            {"name": f"BRAM_ln2_w_g{i}", "dims": [DIM]},
            {"name": f"BRAM_ln2_w_b{i}", "dims": [DIM]},
            {"name": f"BRAM_wq_{i}", "dims": [DIM, DIM]},
            {"name": f"BRAM_wk_{i}", "dims": [DIM, DIM]},
            {"name": f"BRAM_wv_{i}", "dims": [DIM, DIM]},
            {"name": f"BRAM_mlp_w1_{i}", "dims": [MLP_DIM, DIM]},
            {"name": f"BRAM_mlp_w2_{i}", "dims": [DIM, MLP_DIM]},
        ])

        drams.extend([
            {"name": f"DRAM_ln1_w_g{i}", "dims": [DIM], "bundle": f"mem_ln1_w_g_{i}"},
            {"name": f"DRAM_ln1_w_b{i}", "dims": [DIM], "bundle": f"mem_ln1_w_b_{i}"},
            {"name": f"DRAM_ln2_w_g{i}", "dims": [DIM], "bundle": f"mem_ln2_w_g_{i}"},
            {"name": f"DRAM_ln2_w_b{i}", "dims": [DIM], "bundle": f"mem_ln2_w_b_{i}"},
            {"name": f"DRAM_wq_{i}", "dims": [DIM, DIM], "bundle": f"mem_wq_{i}"},
            {"name": f"DRAM_wk_{i}", "dims": [DIM, DIM], "bundle": f"mem_wk_{i}"},
            {"name": f"DRAM_wv_{i}", "dims": [DIM, DIM], "bundle": f"mem_wv_{i}"},
            {"name": f"DRAM_mlp_w1_{i}", "dims": [MLP_DIM, DIM], "bundle": f"mem_mlp1_{i}"},
            {"name": f"DRAM_mlp_w2_{i}", "dims": [DIM, MLP_DIM], "bundle": f"mem_mlp2_{i}"},
        ])

    ops = [(
        "load_input",
        {"func_name": "load", "dims": [seq_len, DIM], "args": ["DRAM_input", "BRAM_input_0"]}
    )]

    for i in range(NUM_LAYERS):
        ops.extend([
            (f"load_ln1_w_g{i}", {"func_name": "load", "dims": [DIM], "args": [f"DRAM_ln1_w_g{i}", f"BRAM_ln1_w_g{i}"]}),
            (f"load_ln1_w_b{i}", {"func_name": "load", "dims": [DIM], "args": [f"DRAM_ln1_w_b{i}", f"BRAM_ln1_w_b{i}"]}),
            (f"load_ln2_w_g{i}", {"func_name": "load", "dims": [DIM], "args": [f"DRAM_ln2_w_g{i}", f"BRAM_ln2_w_g{i}"]}),
            (f"load_ln2_w_b{i}", {"func_name": "load", "dims": [DIM], "args": [f"DRAM_ln2_w_b{i}", f"BRAM_ln2_w_b{i}"]}),
            (f"load_wq_{i}",  {"func_name": "load", "dims": [DIM, DIM], "args": [f"DRAM_wq_{i}", f"BRAM_wq_{i}"]}),
            (f"load_wk_{i}",  {"func_name": "load", "dims": [DIM, DIM], "args": [f"DRAM_wk_{i}", f"BRAM_wk_{i}"]}),
            (f"load_wv_{i}",  {"func_name": "load", "dims": [DIM, DIM], "args": [f"DRAM_wv_{i}", f"BRAM_wv_{i}"]}),
            (f"load_mlp1_{i}",{"func_name": "load", "dims": [MLP_DIM, DIM], "args": [f"DRAM_mlp_w1_{i}", f"BRAM_mlp_w1_{i}"]}),
            (f"load_mlp2_{i}",{"func_name": "load", "dims": [DIM, MLP_DIM], "args": [f"DRAM_mlp_w2_{i}", f"BRAM_mlp_w2_{i}"]}),
        ])
    
    for i in range(NUM_LAYERS):
        ops.extend(generate_longformer_block(
            f"layer{i}",
            f"BRAM_input_{i}",
            f"BRAM_tmp_{i}",
            f"BRAM_tmp_2_{i}",
            f"BRAM_ln1_w_g{i}",
            f"BRAM_ln1_w_b{i}",
            f"BRAM_ln2_w_g{i}",
            f"BRAM_ln2_w_b{i}",
            f"BRAM_wq_{i}",
            f"BRAM_wk_{i}",
            f"BRAM_wv_{i}",
            f"BRAM_mlp_w1_{i}",
            f"BRAM_mlp_w2_{i}",
            f"BRAM_input_{i+1}",
            seq_len
        ))

    drams.append({"name": "DRAM_output", "dims": [seq_len, DIM], "bundle": "mem_output"})
    ops.append(("store_output", {
        "func_name": "store",
        "dims": [seq_len, DIM],
        "args": [f"BRAM_input_{NUM_LAYERS}", "DRAM_output"]
    }))

    return brams, drams, ops

def generate_longformer_config_txt(seq_len=4096, data_type="ap_fixed<16,5>"):
    """
    Returns a string of the JSON config of the Longformer with the given parameters.
    """
    
    # brams, drams, ops = generate_llama_architecture_block_size(block_size, seq_len)
    brams, drams, ops = generate_longformer_architecture(seq_len)
    # brams, drams, ops = generate_llama_architecture_single_block(seq_len)
    
    brams_lines = []
    for i, b in enumerate(brams):
        comma = "," if i < len(brams) - 1 else ""
        brams_lines.append(
            f'        {{"name": "{b["name"]}", "dims": [{", ".join(str(x) for x in b["dims"])}]}}{comma}'
        )
    brams_str = "\n".join(brams_lines)

    drams_lines = []
    for i, d in enumerate(drams):
        comma = "," if i < len(drams) - 1 else ""
        dims_str = "[" + ", ".join(str(x) for x in d["dims"]) + "]"
        drams_lines.append(
            f'{{"name": "{d["name"]}", "dims": {dims_str}, "bundle": "{d["bundle"]}"}}{comma}'
        )
    drams_str = "\n".join(drams_lines)

    def one_line_list(lst):
        return "[" + ", ".join(str(v) for v in lst) + "]"

    def quoted_list(lst):
        return "[" + ", ".join(f'"{item}"' for item in lst) + "]"

    ops_lines = []
    for i, x in enumerate(ops):
        op_name, op_data = x
        comma = "," if i < len(ops) - 1 else ""
        func_info_line = ""
        if "func_info" in op_data:
            fi_list = []
            for item in op_data["func_info"]:
                if isinstance(item, bool):
                    fi_list.append("true" if item else "0")
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
        "brams": [{brams_str}],
        "drams": [{drams_str}],
        "ops": {{{ops_str}}},
        "output_dram_names": ["DRAM_output"],
        "FPGA_name": "xczu9eg-ffvb1156-2-e",
        "clock_period": 10,
        "task": ["csynth"],
        "data_type": "{data_type}",
        "top_func_name": "top"
    }}'''
    return text

def generate_vit_block(
    mod_name,
    bram_in,
    bram_tmp,
    bram_tmp_2,
    bram_ln1_w_g,
    bram_ln1_w_b,
    bram_ln2_w_g,
    bram_ln2_w_b,
    bram_wq,
    bram_wk,
    bram_wv,
    bram_wo,
    bram_mlp_w1,
    bram_mlp_w2,
    bram_out,
    TOKENS,
):
    DIM = 768
    HEADS = 12
    MLP_DIM = 3072
    EPS = 1e-6

    return [
        (f"ln1_{mod_name}", {
            "func_name": "layernorm",
            "dims": [TOKENS, DIM, EPS],
            "args": [bram_in, bram_ln1_w_g, bram_ln1_w_b, bram_in],
            "func_info": ["layer_norm_template.cpp"],

        }),

        (f"attn_{mod_name}", {
            "func_name": "mha",
            "dims": [TOKENS, DIM, HEADS, DIM // HEADS],
            "args": [bram_in, bram_wq, bram_wk, bram_wv, bram_tmp, f"{DIM//HEADS}"],
            "func_info": ["grouped_mha_rope_template.cpp", False]
        }),

        (f"res1_{mod_name}", {
            "func_name": "matrix_add",
            "dims": [TOKENS, DIM],
            "args": [bram_tmp, bram_in, bram_in],
            "func_info": ["matrix_add_template.cpp"],
        }),

        (f"ln2_{mod_name}", {
            "func_name": "layernorm",
            "dims": [TOKENS, DIM, EPS],
            "args": [bram_in, bram_ln2_w_g, bram_ln2_w_b, bram_in],
            "func_info": ["layer_norm_template.cpp"],
        }),

        (f"mlp1_{mod_name}", {
            "func_name": "matmul",
            "dims": [TOKENS, DIM, MLP_DIM],
            "args": [bram_in, bram_mlp_w1, bram_tmp_2],
            "func_info": ["matmul_template.cpp", False],
        }),

        (f"gelu_{mod_name}", {
            "func_name": "activation",
            "dims": [TOKENS, MLP_DIM],
            "args": [bram_tmp_2, bram_tmp_2],
            "func_info": ["activation_template.cpp", "gelu"],
        }),

        (f"mlp2_{mod_name}", {
            "func_name": "matmul",
            "dims": [TOKENS, MLP_DIM, DIM],
            "args": [bram_tmp_2, bram_mlp_w2, bram_tmp],
            "func_info": ["matmul_template.cpp", False],
        }),

        (f"res2_{mod_name}", {
            "func_name": "matrix_add",
            "dims": [TOKENS, DIM],
            "args": [bram_tmp, bram_in, bram_out],
            "func_info": ["matrix_add_template.cpp"] 
        }),
    ]

def generate_vit_base_architecture():
    IMG_H = IMG_W = 224
    PATCH = 16
    TOKENS = 196
    DIM = 768
    MLP_DIM = 3072
    NUM_LAYERS = 12

    brams = [
        {"name": "BRAM_tokens_0", "dims": [TOKENS, DIM]},
    ]

    drams = [
        {"name": "DRAM_tokens_0", "dims": [TOKENS, DIM], "bundle": "mem_token_input"},
    ]

    for i in range(NUM_LAYERS):
        brams.extend([
            {"name": f"BRAM_tokens_{i+1}", "dims": [TOKENS, DIM]},
            {"name": f"BRAM_tmp_{i}", "dims": [TOKENS, DIM]},
            {"name": f"BRAM_tmp_2_{i}", "dims": [TOKENS, MLP_DIM]},
            {"name": f"BRAM_ln1_w_g{i}", "dims": [DIM]},
            {"name": f"BRAM_ln1_w_b{i}", "dims": [DIM]},
            {"name": f"BRAM_ln2_w_g{i}", "dims": [DIM]},
            {"name": f"BRAM_ln2_w_b{i}", "dims": [DIM]},
            {"name": f"BRAM_wq_{i}", "dims": [DIM, DIM]},
            {"name": f"BRAM_wk_{i}", "dims": [DIM, DIM]},
            {"name": f"BRAM_wv_{i}", "dims": [DIM, DIM]},
            {"name": f"BRAM_wo_{i}", "dims": [DIM, DIM]},
            {"name": f"BRAM_mlp_w1_{i}", "dims": [MLP_DIM, DIM]},
            {"name": f"BRAM_mlp_w2_{i}", "dims": [DIM, MLP_DIM]},
        ])

        drams.extend([
            {"name": f"DRAM_ln1_w_g{i}", "dims": [DIM], "bundle": f"mem_ln1_{i}"},
            {"name": f"DRAM_ln1_w_b{i}", "dims": [DIM], "bundle": f"mem_ln1_{i}"},
            {"name": f"DRAM_ln2_w_g{i}", "dims": [DIM], "bundle": f"mem_ln2_{i}"},
            {"name": f"DRAM_ln2_w_b{i}", "dims": [DIM], "bundle": f"mem_ln2_{i}"},
            {"name": f"DRAM_wq_{i}", "dims": [DIM, DIM], "bundle": f"mem_wq_{i}"},
            {"name": f"DRAM_wk_{i}", "dims": [DIM, DIM], "bundle": f"mem_wk_{i}"},
            {"name": f"DRAM_wv_{i}", "dims": [DIM, DIM], "bundle": f"mem_wv_{i}"},
            {"name": f"DRAM_wo_{i}", "dims": [DIM, DIM], "bundle": f"mem_wo_{i}"},
            {"name": f"DRAM_mlp_w1_{i}", "dims": [MLP_DIM, DIM], "bundle": f"mem_mlp1_{i}"},
            {"name": f"DRAM_mlp_w2_{i}", "dims": [DIM, MLP_DIM], "bundle": f"mem_mlp2_{i}"},
        ])

    ops = [
        ("load_tokens", {
            "func_name": "load",
            "dims": [TOKENS, DIM],
            "args": ["BRAM_tokens_0", "DRAM_tokens_0"],
        }),
    ]

    for i in range(NUM_LAYERS):
        ops.extend([
            (f"load_ln1_w_g{i}", {"func_name": "load", "dims": [DIM], "args": [f"DRAM_ln1_w_g{i}", f"BRAM_ln1_w_g{i}"]}),
            (f"load_ln1_w_b{i}", {"func_name": "load", "dims": [DIM], "args": [f"DRAM_ln1_w_b{i}", f"BRAM_ln1_w_b{i}"]}),
            (f"load_ln2_w_g{i}", {"func_name": "load", "dims": [DIM], "args": [f"DRAM_ln2_w_g{i}", f"BRAM_ln2_w_g{i}"]}),
            (f"load_ln2_w_b{i}", {"func_name": "load", "dims": [DIM], "args": [f"DRAM_ln2_w_b{i}", f"BRAM_ln2_w_b{i}"]}),
            (f"load_wq_{i}", {"func_name": "load", "dims": [DIM, DIM], "args": [f"DRAM_wq_{i}", f"BRAM_wq_{i}"]}),
            (f"load_wk_{i}", {"func_name": "load", "dims": [DIM, DIM], "args": [f"DRAM_wk_{i}", f"BRAM_wk_{i}"]}),
            (f"load_wv_{i}", {"func_name": "load", "dims": [DIM, DIM], "args": [f"DRAM_wv_{i}", f"BRAM_wv_{i}"]}),
            (f"load_wo_{i}", {"func_name": "load", "dims": [DIM, DIM], "args": [f"DRAM_wo_{i}", f"BRAM_wo_{i}"]}),
            (f"load_mlp1_{i}", {"func_name": "load", "dims": [MLP_DIM, DIM], "args": [f"DRAM_mlp_w1_{i}", f"BRAM_mlp_w1_{i}"]}),
            (f"load_mlp2_{i}", {"func_name": "load", "dims": [DIM, MLP_DIM], "args": [f"DRAM_mlp_w2_{i}", f"BRAM_mlp_w2_{i}"]}),
        ])
    
    for i in range(NUM_LAYERS):
        ops.extend(generate_vit_block(
            f"layer{i}",
            f"BRAM_tokens_{i}",
            f"BRAM_tmp_{i}",
            f"BRAM_tmp_2_{i}",
            f"BRAM_ln1_w_g{i}",
            f"BRAM_ln1_w_b{i}",
            f"BRAM_ln2_w_g{i}",
            f"BRAM_ln2_w_b{i}",
            f"BRAM_wq_{i}",
            f"BRAM_wk_{i}",
            f"BRAM_wv_{i}",
            f"BRAM_wo_{i}",
            f"BRAM_mlp_w1_{i}",
            f"BRAM_mlp_w2_{i}",
            f"BRAM_tokens_{i+1}",
            TOKENS
        ))

    drams.append({"name": "DRAM_output", "dims": [TOKENS, DIM], "bundle": "mem_output"})
    ops.append(("store_output", {
        "func_name": "store",
        "dims": [TOKENS, DIM],
        "args": [f"BRAM_tokens_{NUM_LAYERS}", "DRAM_output"],
    }))

    return brams, drams, ops

def generate_vit_config_txt(data_type="ap_fixed<16,5>"):
    """
    Returns a string of the JSON config of the Longformer with the given parameters.
    """
    
    brams, drams, ops = generate_vit_base_architecture()
    
    brams_lines = []
    for i, b in enumerate(brams):
        comma = "," if i < len(brams) - 1 else ""
        brams_lines.append(
            f'        {{"name": "{b["name"]}", "dims": [{", ".join(str(x) for x in b["dims"])}]}}{comma}'
        )
    brams_str = "\n".join(brams_lines)

    drams_lines = []
    for i, d in enumerate(drams):
        comma = "," if i < len(drams) - 1 else ""
        dims_str = "[" + ", ".join(str(x) for x in d["dims"]) + "]"
        drams_lines.append(
            f'{{"name": "{d["name"]}", "dims": {dims_str}, "bundle": "{d["bundle"]}"}}{comma}'
        )
    drams_str = "\n".join(drams_lines)

    def one_line_list(lst):
        return "[" + ", ".join(str(v) for v in lst) + "]"

    def quoted_list(lst):
        return "[" + ", ".join(f'"{item}"' for item in lst) + "]"

    ops_lines = []
    for i, x in enumerate(ops):
        op_name, op_data = x
        comma = "," if i < len(ops) - 1 else ""
        func_info_line = ""
        if "func_info" in op_data:
            fi_list = []
            for item in op_data["func_info"]:
                if isinstance(item, bool):
                    fi_list.append("true" if item else "0")
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
        "brams": [{brams_str}],
        "drams": [{drams_str}],
        "ops": {{{ops_str}}},
        "output_dram_names": ["DRAM_output"],
        "FPGA_name": "xczu9eg-ffvb1156-2-e",
        "clock_period": 10,
        "task": ["csynth"],
        "data_type": "{data_type}",
        "top_func_name": "top"
    }}'''
    return text

DEFAULT_GLOBAL_CONV_CI_FACTORS = [8]
DEFAULT_GLOBAL_CONV_CO_FACTORS = [64]
DEFAULT_CONV_CI_FACTOR = DEFAULT_GLOBAL_CONV_CI_FACTORS[0]
DEFAULT_CONV_CO_FACTOR = DEFAULT_GLOBAL_CONV_CO_FACTORS[0]


def build_conv_factor_config(ci_factor=DEFAULT_CONV_CI_FACTOR, co_factor=DEFAULT_CONV_CO_FACTOR):
    ci_factor = int(ci_factor)
    co_factor = int(co_factor)
    if ci_factor <= 0:
        raise ValueError(f"ci_factor must be positive, got {ci_factor}.")
    if co_factor <= 0:
        raise ValueError(f"co_factor must be positive, got {co_factor}.")
    return {
        "ci_factor": ci_factor,
        "co_factor": co_factor,
        "input_partition_factor": ci_factor,
        "kernel_partition_factor1": co_factor,
        "kernel_partition_factor2": ci_factor,
        "bias_partition_factor": co_factor,
        "output_partition_factor": co_factor,
        "unroll_factor_cin": ci_factor,
        "unroll_factor_cout": co_factor,
    }


def normalize_conv_factor_config(conv_cfg=None):
    if conv_cfg is None:
        return build_conv_factor_config()
    return build_conv_factor_config(conv_cfg["ci_factor"], conv_cfg["co_factor"])


def is_default_conv_factor_config(conv_cfg=None):
    conv_cfg = normalize_conv_factor_config(conv_cfg)
    return (
        conv_cfg["ci_factor"] == DEFAULT_CONV_CI_FACTOR
        and conv_cfg["co_factor"] == DEFAULT_CONV_CO_FACTOR
    )


def get_conv_factor_suffix(conv_cfg=None, preserve_legacy_name=False):
    conv_cfg = normalize_conv_factor_config(conv_cfg)
    if preserve_legacy_name and is_default_conv_factor_config(conv_cfg):
        return ""
    return f"_ci{conv_cfg['ci_factor']}_co{conv_cfg['co_factor']}"


def build_conv_dims(c_in, c_out, h_in, w_in, h_out, w_out, kernel, pad, stride, conv_cfg=None):
    conv_cfg = normalize_conv_factor_config(conv_cfg)
    return [
        c_in, c_out, h_in, w_in, h_out, w_out, kernel, pad, stride,
        conv_cfg["input_partition_factor"],
        conv_cfg["kernel_partition_factor1"],
        conv_cfg["kernel_partition_factor2"],
        conv_cfg["bias_partition_factor"],
        conv_cfg["output_partition_factor"],
        conv_cfg["unroll_factor_cin"],
        conv_cfg["unroll_factor_cout"],
    ]


def build_conv_tile_dims(
    ic_tile,
    patch_h,
    patch_w,
    oc_tile,
    weight_ic,
    kernel,
    out_c,
    tile_h,
    tile_w,
    stride,
    conv_cfg=None,
):
    conv_cfg = normalize_conv_factor_config(conv_cfg)
    return [
        ic_tile,
        patch_h,
        patch_w,
        oc_tile,
        weight_ic,
        kernel,
        out_c,
        tile_h,
        tile_w,
        stride,
        conv_cfg["ci_factor"],
        conv_cfg["co_factor"],
    ]

def build_conv_tiled_dims(
    c_in,
    c_out,
    h_in,
    w_in,
    h_out,
    w_out,
    kernel,
    pad,
    stride,
    tile_c=128,
    tile_h=14,
    tile_w=14,
    ic_chunk=32,
    oc_chunk=128,
    ci_factor=DEFAULT_CONV_CI_FACTOR,
    co_factor=DEFAULT_CONV_CO_FACTOR,
):
    return [
        c_in, c_out, h_in, w_in, h_out, w_out, kernel, pad, stride,
        tile_c, tile_h, tile_w, ic_chunk, oc_chunk, ci_factor, co_factor,
    ]

def serialize_scale_model_config(brams, drams, ops, output_dram_names, data_type="ap_fixed<16,5>"):
    brams_lines = []
    for i, bram in enumerate(brams):
        comma = "," if i < len(brams) - 1 else ""
        dims_str = ", ".join(str(x) for x in bram["dims"])
        brams_lines.append(
            f'        {{"name": "{bram["name"]}", "dims": [{dims_str}]}}{comma}'
        )
    brams_str = "\n".join(brams_lines)

    drams_lines = []
    for i, dram in enumerate(drams):
        comma = "," if i < len(drams) - 1 else ""
        dims_str = "[" + ", ".join(str(x) for x in dram["dims"]) + "]"
        drams_lines.append(
            f'{{"name": "{dram["name"]}", "dims": {dims_str}, "bundle": "{dram["bundle"]}"}}{comma}'
        )
    drams_str = "\n".join(drams_lines)

    def quoted_list(items):
        return "[" + ", ".join(f'"{item}"' for item in items) + "]"

    def one_line_list(items):
        return "[" + ", ".join(str(item) for item in items) + "]"

    ops_lines = []
    for i, (op_name, op_data) in enumerate(ops):
        comma = "," if i < len(ops) - 1 else ""
        func_info_line = ""
        if "func_info" in op_data:
            func_info_items = []
            for item in op_data["func_info"]:
                if isinstance(item, bool):
                    func_info_items.append("true" if item else "0")
                elif isinstance(item, float):
                    func_info_items.append(str(item))
                elif isinstance(item, list):
                    func_info_items.append(quoted_list(item))
                else:
                    func_info_items.append(f'"{item}"')
            func_info_line = f'        "func_info": [{", ".join(func_info_items)}],\n'
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
        ops_lines.append(f'        "{op_name}": {op_block}{comma}')
    ops_str = "\n".join(ops_lines)

    return f'''{{
        "brams": [{brams_str}],
        "drams": [{drams_str}],
        "ops": {{{ops_str}}},
        "output_dram_names": {quoted_list(output_dram_names)},
        "FPGA_name": "xczu9eg-ffvb1156-2-e",
        "clock_period": 10,
        "task": ["csynth"],
        "data_type": "{data_type}",
        "top_func_name": "top"
    }}'''

def generate_resnet18_basic_block(
    stage_id,
    block_id,
    c_in,
    c_out,
    h_in,
    w_in,
    downsample,
    bram_in,
    bram_tmp1,
    bram_tmp2,
    bram_out,
    bram_skip=None,
):
    if downsample and bram_skip is None:
        raise ValueError("A downsample block requires a skip buffer.")

    brams = []
    drams = []
    ops = []

    stride = 2 if downsample else 1
    h_out = h_in // stride
    w_out = w_in // stride
    prefix = f"s{stage_id}_b{block_id}"

    brams.extend([
        {"name": f"BRAM_w_{prefix}_1", "dims": [c_out, c_in, 3, 3]},
        {"name": f"BRAM_bn_{prefix}_1", "dims": [4, c_out]},
        {"name": f"BRAM_w_{prefix}_2", "dims": [c_out, c_out, 3, 3]},
        {"name": f"BRAM_bn_{prefix}_2", "dims": [4, c_out]},
    ])
    drams.extend([
        {"name": f"DRAM_w_{prefix}_1", "dims": [c_out, c_in, 3, 3], "bundle": f"mem_w_{prefix}_1"},
        {"name": f"DRAM_bn_{prefix}_1", "dims": [4, c_out], "bundle": f"mem_bn_{prefix}_1"},
        {"name": f"DRAM_w_{prefix}_2", "dims": [c_out, c_out, 3, 3], "bundle": f"mem_w_{prefix}_2"},
        {"name": f"DRAM_bn_{prefix}_2", "dims": [4, c_out], "bundle": f"mem_bn_{prefix}_2"},
    ])

    ops.extend([
        (f"{prefix}_conv1", {
            "func_name": "conv",
            "dims": build_conv_dims(c_in, c_out, h_in, w_in, h_out, w_out, 3, 1, stride),
            "func_info": ["conv_template.cpp", "conv2d", False],
            "args": [bram_in, f"BRAM_w_{prefix}_1", bram_tmp1],
        }),
        (f"{prefix}_bn1", {
            "func_name": "batchnorm",
            "dims": [c_out, h_out, w_out],
            "func_info": ["batch_norm_template.cpp", False],
            "args": [bram_tmp1, f"BRAM_bn_{prefix}_1", bram_tmp1],
        }),
        (f"{prefix}_relu1", {
            "func_name": "activation_conv",
            "dims": [c_out, h_out, w_out],
            "func_info": ["activation_template_conv.cpp", "relu"],
            "args": [bram_tmp1, bram_tmp1],
        }),
        (f"{prefix}_conv2", {
            "func_name": "conv",
            "dims": build_conv_dims(c_out, c_out, h_out, w_out, h_out, w_out, 3, 1, 1),
            "func_info": ["conv_template.cpp", "conv2d", False],
            "args": [bram_tmp1, f"BRAM_w_{prefix}_2", bram_tmp2],
        }),
        (f"{prefix}_bn2", {
            "func_name": "batchnorm",
            "dims": [c_out, h_out, w_out],
            "func_info": ["batch_norm_template.cpp", False],
            "args": [bram_tmp2, f"BRAM_bn_{prefix}_2", bram_tmp2],
        }),
    ])

    skip_source = bram_in
    if downsample:
        brams.append({"name": f"BRAM_w_{prefix}_down", "dims": [c_out, c_in, 1, 1]})
        drams.append({"name": f"DRAM_w_{prefix}_down", "dims": [c_out, c_in, 1, 1], "bundle": f"mem_w_{prefix}_down"})
        ops.append((
            f"{prefix}_downsample", {
                "func_name": "conv",
                "dims": build_conv_dims(c_in, c_out, h_in, w_in, h_out, w_out, 1, 0, 2),
                "func_info": ["conv_template.cpp", "conv2d", False],
                "args": [bram_in, f"BRAM_w_{prefix}_down", bram_skip],
            }
        ))
        skip_source = bram_skip

    ops.extend([
        (f"{prefix}_residual_add", {
            "func_name": "matrix_add_conv",
            "dims": [c_out, h_out, w_out],
            "func_info": ["matrix_add_template_conv.cpp"],
            "args": [bram_tmp2, skip_source, bram_out],
        }),
        (f"{prefix}_relu2", {
            "func_name": "activation_conv",
            "dims": [c_out, h_out, w_out],
            "func_info": ["activation_template_conv.cpp", "relu"],
            "args": [bram_out, bram_out],
        }),
    ])

    return brams, drams, ops, h_out, w_out

def generate_resnet18_architecture():
    brams = [
        {"name": "BRAM_feat_input", "dims": [3, 224, 224]},
        {"name": "BRAM_feat_stem", "dims": [64, 112, 112]},
        {"name": "BRAM_feat_pool", "dims": [64, 56, 56]},
        {"name": "BRAM_feat_s1_0", "dims": [64, 56, 56]},
        {"name": "BRAM_feat_s1_1", "dims": [64, 56, 56]},
        {"name": "BRAM_feat_s1_tmp1", "dims": [64, 56, 56]},
        {"name": "BRAM_feat_s1_tmp2", "dims": [64, 56, 56]},
        {"name": "BRAM_feat_s2_0", "dims": [128, 28, 28]},
        {"name": "BRAM_feat_s2_1", "dims": [128, 28, 28]},
        {"name": "BRAM_feat_s2_tmp1", "dims": [128, 28, 28]},
        {"name": "BRAM_feat_s2_tmp2", "dims": [128, 28, 28]},
        {"name": "BRAM_feat_s2_skip", "dims": [128, 28, 28]},
        {"name": "BRAM_feat_s3_0", "dims": [256, 14, 14]},
        {"name": "BRAM_feat_s3_1", "dims": [256, 14, 14]},
        {"name": "BRAM_feat_s3_tmp1", "dims": [256, 14, 14]},
        {"name": "BRAM_feat_s3_tmp2", "dims": [256, 14, 14]},
        {"name": "BRAM_feat_s3_skip", "dims": [256, 14, 14]},
        {"name": "BRAM_feat_s4_0", "dims": [512, 7, 7]},
        {"name": "BRAM_feat_s4_1", "dims": [512, 7, 7]},
        {"name": "BRAM_feat_s4_tmp1", "dims": [512, 7, 7]},
        {"name": "BRAM_feat_s4_tmp2", "dims": [512, 7, 7]},
        {"name": "BRAM_feat_s4_skip", "dims": [512, 7, 7]},
        {"name": "BRAM_pool", "dims": [512, 1, 1]},
        {"name": "BRAM_out", "dims": [1000, 1, 1]},
        {"name": "BRAM_fc", "dims": [1000, 512, 1, 1]},
        {"name": "BRAM_w_stem", "dims": [64, 3, 7, 7]},
        {"name": "BRAM_bn_stem", "dims": [4, 64]},
    ]
    drams = [
        {"name": "DRAM_input", "dims": [3, 224, 224], "bundle": "mem_input"},
        {"name": "DRAM_w_stem", "dims": [64, 3, 7, 7], "bundle": "mem_w_stem"},
        {"name": "DRAM_bn_stem", "dims": [4, 64], "bundle": "mem_bn_stem"},
    ]
    ops = [
        ("load_input", {
            "func_name": "load",
            "dims": [3, 224, 224],
            "args": ["DRAM_input", "BRAM_feat_input"],
        }),
        ("stem_conv", {
            "func_name": "conv",
            "dims": build_conv_dims(3, 64, 224, 224, 112, 112, 7, 3, 2),
            "func_info": ["conv_template.cpp", "conv2d", False],
            "args": ["BRAM_feat_input", "BRAM_w_stem", "BRAM_feat_stem"],
        }),
        ("stem_bn", {
            "func_name": "batchnorm",
            "dims": [64, 112, 112],
            "func_info": ["batch_norm_template.cpp", False],
            "args": ["BRAM_feat_stem", "BRAM_bn_stem", "BRAM_feat_stem"],
        }),
        ("stem_relu", {
            "func_name": "activation_conv",
            "dims": [64, 112, 112],
            "func_info": ["activation_template_conv.cpp", "relu"],
            "args": ["BRAM_feat_stem", "BRAM_feat_stem"],
        }),
        ("stem_pool", {
            "func_name": "maxpool_pad",
            "dims": [64, 112, 112, 56, 56, 3, 3, 2, 2, 1, 1],
            "func_info": ["maxpool_pad_template.cpp"],
            "args": ["BRAM_feat_stem", "BRAM_feat_pool"],
        }),
    ]

    current_bram = "BRAM_feat_pool"
    c_in = 64
    h_in = 56
    w_in = 56
    stage_specs = [
        (1, 64, "BRAM_feat_s1_tmp1", "BRAM_feat_s1_tmp2", None),
        (2, 128, "BRAM_feat_s2_tmp1", "BRAM_feat_s2_tmp2", "BRAM_feat_s2_skip"),
        (3, 256, "BRAM_feat_s3_tmp1", "BRAM_feat_s3_tmp2", "BRAM_feat_s3_skip"),
        (4, 512, "BRAM_feat_s4_tmp1", "BRAM_feat_s4_tmp2", "BRAM_feat_s4_skip"),
    ]

    for stage_id, c_out, tmp1, tmp2, skip in stage_specs:
        for block_id in range(2):
            downsample = stage_id > 1 and block_id == 0
            out_bram = f"BRAM_feat_s{stage_id}_{block_id}"
            block_brams, block_drams, block_ops, h_out, w_out = generate_resnet18_basic_block(
                stage_id=stage_id,
                block_id=block_id,
                c_in=c_in,
                c_out=c_out,
                h_in=h_in,
                w_in=w_in,
                downsample=downsample,
                bram_in=current_bram,
                bram_tmp1=tmp1,
                bram_tmp2=tmp2,
                bram_out=out_bram,
                bram_skip=skip if downsample else None,
            )
            brams.extend(block_brams)
            drams.extend(block_drams)
            ops.extend(block_ops)
            current_bram = out_bram
            c_in = c_out
            h_in = h_out
            w_in = w_out

    drams.extend([
        {"name": "DRAM_out", "dims": [1000, 1, 1], "bundle": "mem_out"},
        {"name": "DRAM_fc", "dims": [1000, 512, 1, 1], "bundle": "mem_fc"},
    ])
    ops.extend([
        ("global_pool", {
            "func_name": "adaptive_avgpool",
            "dims": [512, 7, 7, 1, 1],
            "func_info": ["adaptive_avgpool_template.cpp"],
            "args": [current_bram, "BRAM_pool"],
        }),
        ("fc", {
            "func_name": "conv",
            "dims": build_conv_dims(512, 1000, 1, 1, 1, 1, 1, 0, 1),
            "func_info": ["conv_template.cpp", "conv2d", False],
            "args": ["BRAM_pool", "BRAM_fc", "BRAM_out"],
        }),
        ("store_output", {
            "func_name": "store",
            "dims": [1000, 1, 1],
            "args": ["BRAM_out", "DRAM_out"],
        }),
    ])

    return brams, drams, ops

def generate_resnet18_config_txt(data_type="ap_fixed<16,5>"):
    brams, drams, ops = generate_resnet18_architecture()
    return serialize_scale_model_config(
        brams=brams,
        drams=drams,
        ops=ops,
        output_dram_names=["DRAM_out"],
        data_type=data_type,
    )

DEFAULT_RESNET18_TILE_OC = 128
DEFAULT_RESNET18_TILE_IC = 32
DEFAULT_RESNET18_TILE_H = 14
DEFAULT_RESNET18_TILE_W = 14


def build_resnet18_tiled_config(
    tile_oc=DEFAULT_RESNET18_TILE_OC,
    tile_ic=DEFAULT_RESNET18_TILE_IC,
    tile_h=DEFAULT_RESNET18_TILE_H,
    tile_w=DEFAULT_RESNET18_TILE_W,
):
    values = {
        "tile_oc": int(tile_oc),
        "tile_ic": int(tile_ic),
        "tile_h": int(tile_h),
        "tile_w": int(tile_w),
    }
    for key, value in values.items():
        if value <= 0:
            raise ValueError(f"{key} must be positive, got {value}.")

    values.update({
        "gap_tile_h": min(values["tile_h"], 7),
        "gap_tile_w": min(values["tile_w"], 7),
        "patch_stem_h": (values["tile_h"] - 1) * 2 + 7,
        "patch_stem_w": (values["tile_w"] - 1) * 2 + 7,
        "patch_stride1_k1_h": values["tile_h"],
        "patch_stride1_k1_w": values["tile_w"],
        "patch_stride2_k3_h": (values["tile_h"] - 1) * 2 + 3,
        "patch_stride2_k3_w": (values["tile_w"] - 1) * 2 + 3,
        "patch_stride2_k1_h": (values["tile_h"] - 1) * 2 + 1,
        "patch_stride2_k1_w": (values["tile_w"] - 1) * 2 + 1,
        "patch_stride1_k3_h": values["tile_h"] + 2,
        "patch_stride1_k3_w": values["tile_w"] + 2,
        "is_default": (
            values["tile_oc"] == DEFAULT_RESNET18_TILE_OC
            and values["tile_ic"] == DEFAULT_RESNET18_TILE_IC
            and values["tile_h"] == DEFAULT_RESNET18_TILE_H
            and values["tile_w"] == DEFAULT_RESNET18_TILE_W
        ),
    })
    return values


def get_resnet18_tiled_filename(data_type, tiled_cfg):
    naming_dtype = data_type.replace('<', '_').replace('>', '_').replace(',', '_')
    if tiled_cfg["is_default"]:
        return f"RESNET18_TILED_config_{naming_dtype}.json"
    return (
        f"RESNET18_TILED_oc{tiled_cfg['tile_oc']}_ic{tiled_cfg['tile_ic']}"
        f"_h{tiled_cfg['tile_h']}_w{tiled_cfg['tile_w']}_config_{naming_dtype}.json"
    )


def make_loop_begin(name, var_name, start_expr, end_expr, step_expr, valid_name, tile_expr):
    return (
        name,
        {
            "func_name": "loop_begin",
            "dims": [],
            "args": [
                str(var_name),
                str(start_expr),
                str(end_expr),
                str(step_expr),
                str(valid_name),
                str(tile_expr),
            ],
        },
    )


def make_loop_end(name):
    return (
        name,
        {
            "func_name": "loop_end",
            "dims": [],
            "args": [],
        },
    )


def append_tiled_conv_ops(
    ops,
    prefix,
    tiled_cfg,
    conv_cfg,
    feat_in,
    weight_dram,
    feat_out,
    c_in,
    c_out,
    h_in,
    w_in,
    h_out,
    w_out,
    kernel,
    pad,
    stride,
    patch_bram,
    weight_bram,
    out_bram,
    out_tile_h,
    out_tile_w,
    bn_dram=None,
    relu_after=False,
):
    conv_cfg = normalize_conv_factor_config(conv_cfg)
    tile_oc = tiled_cfg["tile_oc"]
    tile_ic = tiled_cfg["tile_ic"]
    patch_h = (out_tile_h - 1) * stride + kernel
    patch_w = (out_tile_w - 1) * stride + kernel

    ops.extend([
        make_loop_begin(f"{prefix}_co_loop", "co_base", 0, c_out, tile_oc, "valid_co", tile_oc),
        make_loop_begin(f"{prefix}_oh_loop", "oh_base", 0, h_out, out_tile_h, "valid_oh", out_tile_h),
        make_loop_begin(f"{prefix}_ow_loop", "ow_base", 0, w_out, out_tile_w, "valid_ow", out_tile_w),
        (
            f"{prefix}_clear_out",
            {
                "func_name": "clear_tile",
                "dims": [tile_oc, out_tile_h, out_tile_w],
                "args": [out_bram],
            },
        ),
        make_loop_begin(f"{prefix}_ci_loop", "ci_base", 0, c_in, tile_ic, "valid_ci", tile_ic),
        (
            f"{prefix}_load_patch",
            {
                "func_name": "load_fmap_patch",
                "dims": [c_in, h_in, w_in, tile_ic, patch_h, patch_w, stride, pad],
                "args": [feat_in, patch_bram, "ci_base", "oh_base", "ow_base"],
            },
        ),
        (
            f"{prefix}_load_weight",
            {
                "func_name": "load_weight_tile",
                "dims": [c_out, c_in, kernel, tile_oc, tile_ic],
                "args": [weight_dram, weight_bram, "co_base", "ci_base"],
            },
        ),
        (
            f"{prefix}_conv",
            {
                "func_name": "conv_tile",
                "dims": build_conv_tile_dims(
                    tile_ic,
                    patch_h,
                    patch_w,
                    tile_oc,
                    tile_ic,
                    kernel,
                    tile_oc,
                    out_tile_h,
                    out_tile_w,
                    stride,
                    conv_cfg=conv_cfg,
                ),
                "args": [patch_bram, weight_bram, out_bram, "valid_co", "valid_ci", "valid_oh", "valid_ow"],
            },
        ),
        make_loop_end(f"{prefix}_ci_loop_end"),
    ])

    if bn_dram is not None:
        ops.extend([
            (
                f"{prefix}_load_bn",
                {
                    "func_name": "load_bn_tile",
                    "dims": [c_out, tile_oc],
                    "args": [bn_dram, "BRAM_bn_tile", "co_base"],
                },
            ),
            (
                f"{prefix}_batchnorm",
                {
                    "func_name": "batchnorm_tile",
                    "dims": [tile_oc, out_tile_h, out_tile_w],
                    "args": [out_bram, "BRAM_bn_tile", out_bram, "valid_co", "valid_oh", "valid_ow"],
                },
            ),
        ])

    if relu_after:
        ops.append(
            (
                f"{prefix}_relu",
                {
                    "func_name": "activation_tile",
                    "dims": [tile_oc, out_tile_h, out_tile_w],
                    "func_info": ["activation_tile", "relu"],
                    "args": [out_bram, out_bram, "valid_co", "valid_oh", "valid_ow"],
                },
            )
        )

    ops.extend([
        (
            f"{prefix}_store",
            {
                "func_name": "store_fmap_tile",
                "dims": [c_out, h_out, w_out, tile_oc, out_tile_h, out_tile_w],
                "args": [out_bram, feat_out, "co_base", "oh_base", "ow_base", "valid_co", "valid_oh", "valid_ow"],
            },
        ),
        make_loop_end(f"{prefix}_ow_loop_end"),
        make_loop_end(f"{prefix}_oh_loop_end"),
        make_loop_end(f"{prefix}_co_loop_end"),
    ])


def append_tiled_residual_add_ops(
    ops,
    prefix,
    tiled_cfg,
    feat_main,
    feat_skip,
    feat_out,
    c_out,
    h_out,
    w_out,
):
    tile_oc = tiled_cfg["tile_oc"]
    tile_h = tiled_cfg["tile_h"]
    tile_w = tiled_cfg["tile_w"]
    ops.extend([
        make_loop_begin(f"{prefix}_add_co_loop", "co_base", 0, c_out, tile_oc, "valid_co", tile_oc),
        make_loop_begin(f"{prefix}_add_oh_loop", "oh_base", 0, h_out, tile_h, "valid_oh", tile_h),
        make_loop_begin(f"{prefix}_add_ow_loop", "ow_base", 0, w_out, tile_w, "valid_ow", tile_w),
        (
            f"{prefix}_load_main_tile",
            {
                "func_name": "load_fmap_tile",
                "dims": [c_out, h_out, w_out, tile_oc, tile_h, tile_w],
                "args": [feat_main, "BRAM_out_tile", "co_base", "oh_base", "ow_base"],
            },
        ),
        (
            f"{prefix}_load_skip_tile",
            {
                "func_name": "load_fmap_tile",
                "dims": [c_out, h_out, w_out, tile_oc, tile_h, tile_w],
                "args": [feat_skip, "BRAM_skip_tile", "co_base", "oh_base", "ow_base"],
            },
        ),
        (
            f"{prefix}_add_tile",
            {
                "func_name": "matrix_add_tile",
                "dims": [tile_oc, tile_h, tile_w],
                "args": ["BRAM_out_tile", "BRAM_skip_tile", "BRAM_out_tile", "valid_co", "valid_oh", "valid_ow"],
            },
        ),
        (
            f"{prefix}_relu_tile",
            {
                "func_name": "activation_tile",
                "dims": [tile_oc, tile_h, tile_w],
                "func_info": ["activation_tile", "relu"],
                "args": ["BRAM_out_tile", "BRAM_out_tile", "valid_co", "valid_oh", "valid_ow"],
            },
        ),
        (
            f"{prefix}_store_tile",
            {
                "func_name": "store_fmap_tile",
                "dims": [c_out, h_out, w_out, tile_oc, tile_h, tile_w],
                "args": ["BRAM_out_tile", feat_out, "co_base", "oh_base", "ow_base", "valid_co", "valid_oh", "valid_ow"],
            },
        ),
        make_loop_end(f"{prefix}_add_ow_loop_end"),
        make_loop_end(f"{prefix}_add_oh_loop_end"),
        make_loop_end(f"{prefix}_add_co_loop_end"),
    ])


def append_tiled_maxpool_ops(ops, prefix, tiled_cfg, feat_in, feat_out, c_in, h_in, w_in, h_out, w_out):
    tile_oc = tiled_cfg["tile_oc"]
    tile_h = tiled_cfg["tile_h"]
    tile_w = tiled_cfg["tile_w"]
    patch_h = tiled_cfg["patch_stride2_k3_h"]
    patch_w = tiled_cfg["patch_stride2_k3_w"]
    ops.extend([
        make_loop_begin(f"{prefix}_co_loop", "co_base", 0, c_in, tile_oc, "valid_co", tile_oc),
        make_loop_begin(f"{prefix}_oh_loop", "oh_base", 0, h_out, tile_h, "valid_oh", tile_h),
        make_loop_begin(f"{prefix}_ow_loop", "ow_base", 0, w_out, tile_w, "valid_ow", tile_w),
        (
            f"{prefix}_load_patch",
            {
                "func_name": "load_fmap_patch",
                "dims": [c_in, h_in, w_in, tile_oc, patch_h, patch_w, 2, 1],
                "args": [feat_in, "BRAM_pool_patch", "co_base", "oh_base", "ow_base"],
            },
        ),
        (
            f"{prefix}_maxpool",
            {
                "func_name": "maxpool_tile",
                "dims": [tile_oc, patch_h, patch_w, tile_oc, tile_h, tile_w, 3, 3, 2, 2],
                "args": ["BRAM_pool_patch", "BRAM_out_tile", "valid_co", "valid_oh", "valid_ow"],
            },
        ),
        (
            f"{prefix}_store",
            {
                "func_name": "store_fmap_tile",
                "dims": [c_in, h_out, w_out, tile_oc, tile_h, tile_w],
                "args": ["BRAM_out_tile", feat_out, "co_base", "oh_base", "ow_base", "valid_co", "valid_oh", "valid_ow"],
            },
        ),
        make_loop_end(f"{prefix}_ow_loop_end"),
        make_loop_end(f"{prefix}_oh_loop_end"),
        make_loop_end(f"{prefix}_co_loop_end"),
    ])


def append_tiled_gap_ops(ops, prefix, tiled_cfg, feat_in, feat_out, c_in, h_in, w_in):
    tile_oc = tiled_cfg["tile_oc"]
    gap_tile_h = tiled_cfg["gap_tile_h"]
    gap_tile_w = tiled_cfg["gap_tile_w"]
    ops.extend([
        make_loop_begin(f"{prefix}_co_loop", "co_base", 0, c_in, tile_oc, "valid_co", tile_oc),
        (
            f"{prefix}_clear_acc",
            {
                "func_name": "clear_tile",
                "dims": [tile_oc, 1, 1],
                "args": ["BRAM_gap_out"],
            },
        ),
        make_loop_begin(f"{prefix}_oh_loop", "gh_base", 0, h_in, gap_tile_h, "valid_gh", gap_tile_h),
        make_loop_begin(f"{prefix}_ow_loop", "gw_base", 0, w_in, gap_tile_w, "valid_gw", gap_tile_w),
        (
            f"{prefix}_load_tile",
            {
                "func_name": "load_fmap_tile",
                "dims": [c_in, h_in, w_in, tile_oc, gap_tile_h, gap_tile_w],
                "args": [feat_in, "BRAM_gap_tile", "co_base", "gh_base", "gw_base"],
            },
        ),
        (
            f"{prefix}_accumulate",
            {
                "func_name": "avgpool_accumulate_tile",
                "dims": [tile_oc, gap_tile_h, gap_tile_w],
                "args": ["BRAM_gap_tile", "BRAM_gap_out", "valid_co", "valid_gh", "valid_gw"],
            },
        ),
        make_loop_end(f"{prefix}_ow_loop_end"),
        make_loop_end(f"{prefix}_oh_loop_end"),
        (
            f"{prefix}_finalize",
            {
                "func_name": "avgpool_finalize_tile",
                "dims": [tile_oc, h_in * w_in],
                "args": ["BRAM_gap_out", "BRAM_gap_out", "valid_co"],
            },
        ),
        (
            f"{prefix}_store",
            {
                "func_name": "store_fmap_tile",
                "dims": [c_in, 1, 1, tile_oc, 1, 1],
                "args": ["BRAM_gap_out", feat_out, "co_base", "0", "0", "valid_co", "1", "1"],
            },
        ),
        make_loop_end(f"{prefix}_co_loop_end"),
    ])


def generate_resnet18_tiled_basic_block(
    stage_id,
    block_id,
    tiled_cfg,
    c_in,
    c_out,
    h_in,
    w_in,
    downsample,
    feat_in,
    feat_mid,
    feat_out,
    feat_skip=None,
):
    if downsample and feat_skip is None:
        raise ValueError("A downsample tiled block requires a skip feature tensor.")

    prefix = f"s{stage_id}_b{block_id}"
    stride = 2 if downsample else 1
    h_out = h_in // stride
    w_out = w_in // stride

    drams = [
        {"name": f"DRAM_w_{prefix}_1", "dims": [c_out, c_in, 3, 3], "bundle": f"mem_w_{prefix}_1"},
        {"name": f"DRAM_bn_{prefix}_1", "dims": [4, c_out], "bundle": f"mem_bn_{prefix}_1"},
        {"name": f"DRAM_w_{prefix}_2", "dims": [c_out, c_out, 3, 3], "bundle": f"mem_w_{prefix}_2"},
        {"name": f"DRAM_bn_{prefix}_2", "dims": [4, c_out], "bundle": f"mem_bn_{prefix}_2"},
    ]

    ops = []
    append_tiled_conv_ops(
        ops=ops,
        prefix=f"{prefix}_conv1",
        tiled_cfg=tiled_cfg,
        feat_in=feat_in,
        weight_dram=f"DRAM_w_{prefix}_1",
        feat_out=feat_mid,
        c_in=c_in,
        c_out=c_out,
        h_in=h_in,
        w_in=w_in,
        h_out=h_out,
        w_out=w_out,
        kernel=3,
        pad=1,
        stride=stride,
        patch_bram="BRAM_in_patch_stride2_k3" if stride == 2 else "BRAM_in_patch_stride1_k3",
        weight_bram="BRAM_weight_tile_3",
        out_bram="BRAM_out_tile",
        out_tile_h=tiled_cfg["tile_h"],
        out_tile_w=tiled_cfg["tile_w"],
        bn_dram=f"DRAM_bn_{prefix}_1",
        relu_after=True,
    )
    append_tiled_conv_ops(
        ops=ops,
        prefix=f"{prefix}_conv2",
        tiled_cfg=tiled_cfg,
        feat_in=feat_mid,
        weight_dram=f"DRAM_w_{prefix}_2",
        feat_out=feat_out,
        c_in=c_out,
        c_out=c_out,
        h_in=h_out,
        w_in=w_out,
        h_out=h_out,
        w_out=w_out,
        kernel=3,
        pad=1,
        stride=1,
        patch_bram="BRAM_in_patch_stride1_k3",
        weight_bram="BRAM_weight_tile_3",
        out_bram="BRAM_out_tile",
        out_tile_h=tiled_cfg["tile_h"],
        out_tile_w=tiled_cfg["tile_w"],
        bn_dram=f"DRAM_bn_{prefix}_2",
        relu_after=False,
    )

    skip_source = feat_in
    if downsample:
        drams.append({"name": f"DRAM_w_{prefix}_down", "dims": [c_out, c_in, 1, 1], "bundle": f"mem_w_{prefix}_down"})
        append_tiled_conv_ops(
            ops=ops,
            prefix=f"{prefix}_down",
            tiled_cfg=tiled_cfg,
            feat_in=feat_in,
            weight_dram=f"DRAM_w_{prefix}_down",
            feat_out=feat_skip,
            c_in=c_in,
            c_out=c_out,
            h_in=h_in,
            w_in=w_in,
            h_out=h_out,
            w_out=w_out,
            kernel=1,
            pad=0,
            stride=2,
            patch_bram="BRAM_in_patch_stride2_k1",
            weight_bram="BRAM_weight_tile_1",
            out_bram="BRAM_out_tile",
            out_tile_h=tiled_cfg["tile_h"],
            out_tile_w=tiled_cfg["tile_w"],
            bn_dram=None,
            relu_after=False,
        )
        skip_source = feat_skip

    append_tiled_residual_add_ops(
        ops=ops,
        prefix=f"{prefix}_residual",
        tiled_cfg=tiled_cfg,
        feat_main=feat_out,
        feat_skip=skip_source,
        feat_out=feat_out,
        c_out=c_out,
        h_out=h_out,
        w_out=w_out,
    )

    return drams, ops, h_out, w_out


def generate_resnet18_tiled_architecture():
    return generate_resnet18_tiled_architecture_with_config(build_resnet18_tiled_config())


def generate_resnet18_tiled_architecture_with_config(tiled_cfg):
    tile_oc = tiled_cfg["tile_oc"]
    tile_ic = tiled_cfg["tile_ic"]
    tile_h = tiled_cfg["tile_h"]
    tile_w = tiled_cfg["tile_w"]
    brams = [
        {"name": "BRAM_in_patch_stem", "dims": [tile_ic, tiled_cfg["patch_stem_h"], tiled_cfg["patch_stem_w"]]},
        {"name": "BRAM_in_patch_stride1_k1", "dims": [tile_ic, tiled_cfg["patch_stride1_k1_h"], tiled_cfg["patch_stride1_k1_w"]]},
        {"name": "BRAM_in_patch_stride2_k3", "dims": [tile_ic, tiled_cfg["patch_stride2_k3_h"], tiled_cfg["patch_stride2_k3_w"]]},
        {"name": "BRAM_in_patch_stride2_k1", "dims": [tile_ic, tiled_cfg["patch_stride2_k1_h"], tiled_cfg["patch_stride2_k1_w"]]},
        {"name": "BRAM_in_patch_stride1_k3", "dims": [tile_ic, tiled_cfg["patch_stride1_k3_h"], tiled_cfg["patch_stride1_k3_w"]]},
        {"name": "BRAM_in_patch_1", "dims": [tile_ic, 1, 1]},
        {"name": "BRAM_pool_patch", "dims": [tile_oc, tiled_cfg["patch_stride2_k3_h"], tiled_cfg["patch_stride2_k3_w"]]},
        {"name": "BRAM_out_tile", "dims": [tile_oc, tile_h, tile_w]},
        {"name": "BRAM_skip_tile", "dims": [tile_oc, tile_h, tile_w]},
        {"name": "BRAM_weight_tile_7", "dims": [tile_oc, tile_ic, 7, 7]},
        {"name": "BRAM_weight_tile_3", "dims": [tile_oc, tile_ic, 3, 3]},
        {"name": "BRAM_weight_tile_1", "dims": [tile_oc, tile_ic, 1, 1]},
        {"name": "BRAM_bn_tile", "dims": [4, tile_oc]},
        {"name": "BRAM_gap_tile", "dims": [tile_oc, tiled_cfg["gap_tile_h"], tiled_cfg["gap_tile_w"]]},
        {"name": "BRAM_gap_out", "dims": [tile_oc, 1, 1]},
        {"name": "BRAM_fc_out", "dims": [tile_oc, 1, 1]},
    ]

    drams = [
        {"name": "DRAM_input", "dims": [3, 224, 224], "bundle": "mem_input"},
        {"name": "DRAM_w_stem", "dims": [64, 3, 7, 7], "bundle": "mem_w_stem"},
        {"name": "DRAM_bn_stem", "dims": [4, 64], "bundle": "mem_bn_stem"},
        {"name": "DRAM_stem_feat", "dims": [64, 112, 112], "bundle": "mem_stem_feat"},
        {"name": "DRAM_stem_pool", "dims": [64, 56, 56], "bundle": "mem_stem_pool"},
        {"name": "DRAM_s1_mid", "dims": [64, 56, 56], "bundle": "mem_s1_mid"},
        {"name": "DRAM_s1_a", "dims": [64, 56, 56], "bundle": "mem_s1_a"},
        {"name": "DRAM_s1_b", "dims": [64, 56, 56], "bundle": "mem_s1_b"},
        {"name": "DRAM_s2_mid", "dims": [128, 28, 28], "bundle": "mem_s2_mid"},
        {"name": "DRAM_s2_skip", "dims": [128, 28, 28], "bundle": "mem_s2_skip"},
        {"name": "DRAM_s2_a", "dims": [128, 28, 28], "bundle": "mem_s2_a"},
        {"name": "DRAM_s2_b", "dims": [128, 28, 28], "bundle": "mem_s2_b"},
        {"name": "DRAM_s3_mid", "dims": [256, 14, 14], "bundle": "mem_s3_mid"},
        {"name": "DRAM_s3_skip", "dims": [256, 14, 14], "bundle": "mem_s3_skip"},
        {"name": "DRAM_s3_a", "dims": [256, 14, 14], "bundle": "mem_s3_a"},
        {"name": "DRAM_s3_b", "dims": [256, 14, 14], "bundle": "mem_s3_b"},
        {"name": "DRAM_s4_mid", "dims": [512, 7, 7], "bundle": "mem_s4_mid"},
        {"name": "DRAM_s4_skip", "dims": [512, 7, 7], "bundle": "mem_s4_skip"},
        {"name": "DRAM_s4_a", "dims": [512, 7, 7], "bundle": "mem_s4_a"},
        {"name": "DRAM_s4_b", "dims": [512, 7, 7], "bundle": "mem_s4_b"},
        {"name": "DRAM_gap", "dims": [512, 1, 1], "bundle": "mem_gap"},
        {"name": "DRAM_fc", "dims": [1000, 512, 1, 1], "bundle": "mem_fc"},
        {"name": "DRAM_out", "dims": [1000, 1, 1], "bundle": "mem_out"},
    ]

    ops = []
    append_tiled_conv_ops(
        ops=ops,
        prefix="stem_conv",
        tiled_cfg=tiled_cfg,
        feat_in="DRAM_input",
        weight_dram="DRAM_w_stem",
        feat_out="DRAM_stem_feat",
        c_in=3,
        c_out=64,
        h_in=224,
        w_in=224,
        h_out=112,
        w_out=112,
        kernel=7,
        pad=3,
        stride=2,
        patch_bram="BRAM_in_patch_stem",
        weight_bram="BRAM_weight_tile_7",
        out_bram="BRAM_out_tile",
        out_tile_h=tile_h,
        out_tile_w=tile_w,
        bn_dram="DRAM_bn_stem",
        relu_after=True,
    )
    append_tiled_maxpool_ops(
        ops=ops,
        prefix="stem_pool",
        tiled_cfg=tiled_cfg,
        feat_in="DRAM_stem_feat",
        feat_out="DRAM_stem_pool",
        c_in=64,
        h_in=112,
        w_in=112,
        h_out=56,
        w_out=56,
    )

    current_feat = "DRAM_stem_pool"
    c_in = 64
    h_in = 56
    w_in = 56
    stage_buffers = {
        1: {"c_out": 64, "mid": "DRAM_s1_mid", "skip": None, "outs": ["DRAM_s1_a", "DRAM_s1_b"]},
        2: {"c_out": 128, "mid": "DRAM_s2_mid", "skip": "DRAM_s2_skip", "outs": ["DRAM_s2_a", "DRAM_s2_b"]},
        3: {"c_out": 256, "mid": "DRAM_s3_mid", "skip": "DRAM_s3_skip", "outs": ["DRAM_s3_a", "DRAM_s3_b"]},
        4: {"c_out": 512, "mid": "DRAM_s4_mid", "skip": "DRAM_s4_skip", "outs": ["DRAM_s4_a", "DRAM_s4_b"]},
    }

    for stage_id in [1, 2, 3, 4]:
        stage_info = stage_buffers[stage_id]
        for block_id in range(2):
            downsample = stage_id > 1 and block_id == 0
            block_drams, block_ops, h_out, w_out = generate_resnet18_tiled_basic_block(
                stage_id=stage_id,
                block_id=block_id,
                tiled_cfg=tiled_cfg,
                c_in=c_in,
                c_out=stage_info["c_out"],
                h_in=h_in,
                w_in=w_in,
                downsample=downsample,
                feat_in=current_feat,
                feat_mid=stage_info["mid"],
                feat_out=stage_info["outs"][block_id],
                feat_skip=stage_info["skip"],
            )
            drams.extend(block_drams)
            ops.extend(block_ops)
            current_feat = stage_info["outs"][block_id]
            c_in = stage_info["c_out"]
            h_in = h_out
            w_in = w_out

    append_tiled_gap_ops(
        ops=ops,
        prefix="global_pool",
        tiled_cfg=tiled_cfg,
        feat_in=current_feat,
        feat_out="DRAM_gap",
        c_in=512,
        h_in=7,
        w_in=7,
    )
    append_tiled_conv_ops(
        ops=ops,
        prefix="fc",
        tiled_cfg=tiled_cfg,
        feat_in="DRAM_gap",
        weight_dram="DRAM_fc",
        feat_out="DRAM_out",
        c_in=512,
        c_out=1000,
        h_in=1,
        w_in=1,
        h_out=1,
        w_out=1,
        kernel=1,
        pad=0,
        stride=1,
        patch_bram="BRAM_in_patch_1",
        weight_bram="BRAM_weight_tile_1",
        out_bram="BRAM_fc_out",
        out_tile_h=1,
        out_tile_w=1,
        bn_dram=None,
        relu_after=False,
    )

    return brams, drams, ops

def generate_resnet18_tiled_config_txt(data_type="ap_fixed<16,5>", tiled_cfg=None):
    if tiled_cfg is None:
        tiled_cfg = build_resnet18_tiled_config()
    brams, drams, ops = generate_resnet18_tiled_architecture_with_config(tiled_cfg)
    return serialize_scale_model_config(
        brams=brams,
        drams=drams,
        ops=ops,
        output_dram_names=["DRAM_out"],
        data_type=data_type,
    )


LLAMA3_8B_LAYERS = 32
LLAMA3_8B_HIDDEN = 4096
LLAMA3_8B_FFN = 14336
LLAMA3_8B_Q_HEADS = 32
LLAMA3_8B_KV_HEADS = 8
LLAMA3_8B_HEAD_DIM = 128
LLAMA3_8B_KV_DIM = LLAMA3_8B_KV_HEADS * LLAMA3_8B_HEAD_DIM
LLAMA3_8B_VOCAB = 128256
DEFAULT_LLAMA3_8B_CONTEXTS = [2048, 8192]


def build_llama3_8b_tile_config(tile_cfg=None):
    config = {
        "token_tile_prefill": 16,
        "token_tile_decode": 1,
        "hidden_chunk": 128,
        "in_chunk": 128,
        "max_out_chunk": 256,
        "ffn_chunk": 128,
        "k_token_tile": 128,
        "q_head_tile": 4,
    }
    if tile_cfg is not None:
        config.update(tile_cfg)

    for key in (
        "token_tile_prefill",
        "token_tile_decode",
        "hidden_chunk",
        "in_chunk",
        "max_out_chunk",
        "ffn_chunk",
        "k_token_tile",
        "q_head_tile",
    ):
        config[key] = int(config[key])
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive, got {config[key]}.")

    if LLAMA3_8B_Q_HEADS % config["q_head_tile"] != 0:
        raise ValueError(
            f"q_head_tile must divide {LLAMA3_8B_Q_HEADS}, got {config['q_head_tile']}."
        )
    if config["q_head_tile"] % 4 != 0:
        raise ValueError(
            f"q_head_tile must be a multiple of 4 to preserve GQA grouping, got {config['q_head_tile']}."
        )
    if config["max_out_chunk"] < config["in_chunk"]:
        raise ValueError(
            f"max_out_chunk must be >= in_chunk, got {config['max_out_chunk']} < {config['in_chunk']}."
        )
    if config["max_out_chunk"] < config["ffn_chunk"]:
        raise ValueError(
            f"max_out_chunk must be >= ffn_chunk, got {config['max_out_chunk']} < {config['ffn_chunk']}."
        )

    config["q_head_cols"] = config["q_head_tile"] * LLAMA3_8B_HEAD_DIM
    config["local_kv_heads"] = config["q_head_tile"] // 4
    config["local_kv_cols"] = config["local_kv_heads"] * LLAMA3_8B_HEAD_DIM
    config["is_default"] = (
        config["token_tile_prefill"] == 16
        and config["token_tile_decode"] == 1
        and config["hidden_chunk"] == 128
        and config["in_chunk"] == 128
        and config["max_out_chunk"] == 256
        and config["ffn_chunk"] == 128
        and config["k_token_tile"] == 128
        and config["q_head_tile"] == 4
    )
    return config


def build_llama3_8b_pragma_config(pragma_cfg=None):
    config = {
        "linear_in_factor": 1,
        "linear_out_factor": 1,
        "chunk_col_factor": 1,
        "attn_head_factor": 1,
        "attn_dim_factor": 1,
    }
    if pragma_cfg is not None:
        config.update(pragma_cfg)

    for key in (
        "linear_in_factor",
        "linear_out_factor",
        "chunk_col_factor",
        "attn_head_factor",
        "attn_dim_factor",
    ):
        config[key] = int(config[key])
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive, got {config[key]}.")

    config["is_default"] = all(
        config[key] == 1
        for key in (
            "linear_in_factor",
            "linear_out_factor",
            "chunk_col_factor",
            "attn_head_factor",
            "attn_dim_factor",
        )
    )
    return config


def get_llama3_8b_token_tile(seq_capacity, tile_cfg):
    return tile_cfg["token_tile_prefill"] if seq_capacity > 1 else tile_cfg["token_tile_decode"]


def parse_context_list(contexts_arg):
    contexts = []
    for item in contexts_arg.split(","):
        value = int(item.strip())
        if value <= 0:
            raise ValueError(f"context must be positive, got {value}")
        contexts.append(value)
    return contexts


def get_llama3_8b_tile_suffix(tile_cfg, preserve_legacy_name=False):
    if preserve_legacy_name and tile_cfg["is_default"]:
        return ""
    return (
        f"_tp{tile_cfg['token_tile_prefill']}"
        f"_td{tile_cfg['token_tile_decode']}"
        f"_hc{tile_cfg['hidden_chunk']}"
        f"_ic{tile_cfg['in_chunk']}"
        f"_oc{tile_cfg['max_out_chunk']}"
        f"_ffn{tile_cfg['ffn_chunk']}"
        f"_kt{tile_cfg['k_token_tile']}"
        f"_qh{tile_cfg['q_head_tile']}"
    )


def get_llama3_8b_pragma_suffix(pragma_cfg, preserve_legacy_name=False):
    if preserve_legacy_name and pragma_cfg["is_default"]:
        return ""
    return (
        f"_li{pragma_cfg['linear_in_factor']}"
        f"_lo{pragma_cfg['linear_out_factor']}"
        f"_cc{pragma_cfg['chunk_col_factor']}"
        f"_ah{pragma_cfg['attn_head_factor']}"
        f"_ad{pragma_cfg['attn_dim_factor']}"
    )


def get_llama3_8b_filename(mode, max_ctx, data_type, tile_cfg, pragma_cfg=None, preserve_legacy_name=False):
    pragma_cfg = build_llama3_8b_pragma_config(pragma_cfg)
    naming_dtype = data_type.replace('<', '_').replace('>', '_').replace(',', '_')
    return (
        f"LLAMA3_8B_{mode.upper()}_ctx{max_ctx}"
        f"{get_llama3_8b_tile_suffix(tile_cfg, preserve_legacy_name)}"
        f"{get_llama3_8b_pragma_suffix(pragma_cfg, preserve_legacy_name)}"
        f"_config_{naming_dtype}.json"
    )


def append_llama_embedding_ops(ops, prefix, seq_capacity, seq_end_expr, tile_cfg, pragma_cfg, token_dram, output_dram):
    tile_t = get_llama3_8b_token_tile(seq_capacity, tile_cfg)
    hidden_chunk = tile_cfg["hidden_chunk"]
    chunk_col_factor = pragma_cfg["chunk_col_factor"]
    ops.extend([
        make_loop_begin(f"{prefix}_t_loop", "t_base", 0, seq_end_expr, tile_t, "valid_t", tile_t),
        make_loop_begin(f"{prefix}_h_loop", "h_base", 0, LLAMA3_8B_HIDDEN, hidden_chunk, "valid_h", hidden_chunk),
        (
            f"{prefix}_lookup",
            {
                "func_name": "embedding_lookup_chunk",
                "dims": [seq_capacity, LLAMA3_8B_VOCAB, LLAMA3_8B_HIDDEN, tile_t, hidden_chunk, chunk_col_factor],
                "args": [token_dram, "DRAM_embedding", "BRAM_hidden_a", "t_base", "h_base", "valid_t", "valid_h"],
            },
        ),
        (
            f"{prefix}_store",
            {
                "func_name": "store_matrix_tile",
                "dims": [seq_capacity, LLAMA3_8B_HIDDEN, tile_t, hidden_chunk, chunk_col_factor],
                "args": ["BRAM_hidden_a", output_dram, "t_base", "h_base", "valid_t", "valid_h"],
            },
        ),
        make_loop_end(f"{prefix}_h_loop_end"),
        make_loop_end(f"{prefix}_t_loop_end"),
    ])


def append_llama_rmsnorm_ops(ops, prefix, seq_capacity, seq_end_expr, tile_cfg, pragma_cfg, input_dram, output_dram, gamma_dram, layer_idx=None):
    tile_t = get_llama3_8b_token_tile(seq_capacity, tile_cfg)
    hidden_chunk = tile_cfg["hidden_chunk"]
    chunk_col_factor = pragma_cfg["chunk_col_factor"]
    ops.extend([
        make_loop_begin(f"{prefix}_t_loop", "t_base", 0, seq_end_expr, tile_t, "valid_t", tile_t),
        (
            f"{prefix}_clear_sumsq",
            {
                "func_name": "clear_vector_tile",
                "dims": [tile_t],
                "args": ["BRAM_rms_sumsq"],
            },
        ),
        make_loop_begin(f"{prefix}_acc_h_loop", "h_base", 0, LLAMA3_8B_HIDDEN, hidden_chunk, "valid_h", hidden_chunk),
        (
            f"{prefix}_load_acc",
            {
                "func_name": "load_matrix_tile",
                "dims": [seq_capacity, LLAMA3_8B_HIDDEN, tile_t, hidden_chunk, chunk_col_factor],
                "args": [input_dram, "BRAM_hidden_a", "t_base", "h_base"],
            },
        ),
        (
            f"{prefix}_accumulate",
            {
                "func_name": "rmsnorm_accumulate_tile",
                "dims": [tile_t, hidden_chunk, chunk_col_factor],
                "args": ["BRAM_hidden_a", "BRAM_rms_sumsq", "valid_t", "valid_h"],
            },
        ),
        make_loop_end(f"{prefix}_acc_h_loop_end"),
        (
            f"{prefix}_finalize",
            {
                "func_name": "rmsnorm_finalize_rows",
                "dims": [tile_t, LLAMA3_8B_HIDDEN],
                "args": ["BRAM_rms_sumsq", "BRAM_rms_inv", "valid_t"],
            },
        ),
        make_loop_begin(f"{prefix}_apply_h_loop", "h_base", 0, LLAMA3_8B_HIDDEN, hidden_chunk, "valid_h", hidden_chunk),
        (
            f"{prefix}_load_apply",
            {
                "func_name": "load_matrix_tile",
                "dims": [seq_capacity, LLAMA3_8B_HIDDEN, tile_t, hidden_chunk, chunk_col_factor],
                "args": [input_dram, "BRAM_hidden_a", "t_base", "h_base"],
            },
        ),
    ])
    if layer_idx is None:
        ops.append(
            (
                f"{prefix}_load_gamma",
                {
                    "func_name": "load_vector_tile",
                    "dims": [LLAMA3_8B_HIDDEN, hidden_chunk, chunk_col_factor],
                    "args": [gamma_dram, "BRAM_gamma_chunk", "h_base"],
                },
            )
        )
    else:
        ops.append(
            (
                f"{prefix}_load_gamma",
                {
                    "func_name": "load_layer_vector_tile",
                    "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_HIDDEN, hidden_chunk, chunk_col_factor],
                    "args": [gamma_dram, "BRAM_gamma_chunk", str(layer_idx), "h_base"],
                },
            )
        )
    ops.extend([
        (
            f"{prefix}_apply",
            {
                "func_name": "rmsnorm_apply_tile",
                "dims": [tile_t, hidden_chunk, chunk_col_factor],
                "args": ["BRAM_hidden_a", "BRAM_gamma_chunk", "BRAM_rms_inv", "BRAM_hidden_b", "valid_t", "valid_h"],
            },
        ),
        (
            f"{prefix}_store",
            {
                "func_name": "store_matrix_tile",
                "dims": [seq_capacity, LLAMA3_8B_HIDDEN, tile_t, hidden_chunk, chunk_col_factor],
                "args": ["BRAM_hidden_b", output_dram, "t_base", "h_base", "valid_t", "valid_h"],
            },
        ),
        make_loop_end(f"{prefix}_apply_h_loop_end"),
        make_loop_end(f"{prefix}_t_loop_end"),
    ])


def append_llama_linear_ops(
    ops,
    prefix,
    seq_capacity,
    seq_end_expr,
    tile_t,
    input_dram,
    weight_dram,
    output_dram,
    input_dim,
    output_dim,
    out_chunk,
    tile_cfg,
    pragma_cfg,
    layer_idx=None,
):
    in_chunk = tile_cfg["in_chunk"]
    max_out_chunk = tile_cfg["max_out_chunk"]
    linear_in_factor = pragma_cfg["linear_in_factor"]
    linear_out_factor = pragma_cfg["linear_out_factor"]
    ops.extend([
        make_loop_begin(f"{prefix}_t_loop", "t_base", 0, seq_end_expr, tile_t, "valid_t", tile_t),
        make_loop_begin(f"{prefix}_o_loop", "o_base", 0, output_dim, out_chunk, "valid_o", out_chunk),
        (
            f"{prefix}_clear",
            {
                "func_name": "clear_matrix_tile",
                "dims": [tile_t, max_out_chunk, linear_out_factor],
                "args": ["BRAM_matrix_out"],
            },
        ),
        make_loop_begin(f"{prefix}_i_loop", "i_base", 0, input_dim, in_chunk, "valid_i", in_chunk),
        (
            f"{prefix}_load_in",
            {
                "func_name": "load_matrix_tile",
                "dims": [seq_capacity, input_dim, tile_t, in_chunk, linear_in_factor],
                "args": [input_dram, "BRAM_matrix_in", "t_base", "i_base"],
            },
        ),
    ])
    if layer_idx is None:
        ops.append(
            (
                f"{prefix}_load_w",
                {
                    "func_name": "load_weight_tile_2d",
                    "dims": [output_dim, input_dim, max_out_chunk, in_chunk, linear_out_factor, linear_in_factor],
                    "args": [weight_dram, "BRAM_weight_tile", "o_base", "i_base"],
                },
            )
        )
    else:
        ops.append(
            (
                f"{prefix}_load_w",
                {
                    "func_name": "load_weight_tile_layered",
                    "dims": [LLAMA3_8B_LAYERS, output_dim, input_dim, max_out_chunk, in_chunk, linear_out_factor, linear_in_factor],
                    "args": [weight_dram, "BRAM_weight_tile", str(layer_idx), "o_base", "i_base"],
                },
            )
        )
    ops.extend([
        (
            f"{prefix}_linear",
            {
                "func_name": "linear_tile",
                "dims": [tile_t, in_chunk, max_out_chunk, linear_in_factor, linear_out_factor],
                "args": ["BRAM_matrix_in", "BRAM_weight_tile", "BRAM_matrix_out", "valid_t", "valid_o", "valid_i"],
            },
        ),
        make_loop_end(f"{prefix}_i_loop_end"),
        (
            f"{prefix}_store",
            {
                "func_name": "store_matrix_tile",
                "dims": [seq_capacity, output_dim, tile_t, max_out_chunk, linear_out_factor],
                "args": ["BRAM_matrix_out", output_dram, "t_base", "o_base", "valid_t", "valid_o"],
            },
        ),
        make_loop_end(f"{prefix}_o_loop_end"),
        make_loop_end(f"{prefix}_t_loop_end"),
    ])


def append_llama_residual_add_ops(ops, prefix, seq_capacity, seq_end_expr, tile_cfg, pragma_cfg, lhs_dram, rhs_dram, output_dram):
    tile_t = get_llama3_8b_token_tile(seq_capacity, tile_cfg)
    hidden_chunk = tile_cfg["hidden_chunk"]
    chunk_col_factor = pragma_cfg["chunk_col_factor"]
    ops.extend([
        make_loop_begin(f"{prefix}_t_loop", "t_base", 0, seq_end_expr, tile_t, "valid_t", tile_t),
        make_loop_begin(f"{prefix}_h_loop", "h_base", 0, LLAMA3_8B_HIDDEN, hidden_chunk, "valid_h", hidden_chunk),
        (
            f"{prefix}_load_lhs",
            {
                "func_name": "load_matrix_tile",
                "dims": [seq_capacity, LLAMA3_8B_HIDDEN, tile_t, hidden_chunk, chunk_col_factor],
                "args": [lhs_dram, "BRAM_hidden_a", "t_base", "h_base"],
            },
        ),
        (
            f"{prefix}_load_rhs",
            {
                "func_name": "load_matrix_tile",
                "dims": [seq_capacity, LLAMA3_8B_HIDDEN, tile_t, hidden_chunk, chunk_col_factor],
                "args": [rhs_dram, "BRAM_hidden_b", "t_base", "h_base"],
            },
        ),
        (
            f"{prefix}_add",
            {
                "func_name": "matrix_add_tile_2d",
                "dims": [tile_t, hidden_chunk, chunk_col_factor],
                "args": ["BRAM_hidden_a", "BRAM_hidden_b", "BRAM_hidden_a", "valid_t", "valid_h"],
            },
        ),
        (
            f"{prefix}_store",
            {
                "func_name": "store_matrix_tile",
                "dims": [seq_capacity, LLAMA3_8B_HIDDEN, tile_t, hidden_chunk, chunk_col_factor],
                "args": ["BRAM_hidden_a", output_dram, "t_base", "h_base", "valid_t", "valid_h"],
            },
        ),
        make_loop_end(f"{prefix}_h_loop_end"),
        make_loop_end(f"{prefix}_t_loop_end"),
    ])


def append_llama_swiglu_ops(ops, prefix, seq_capacity, seq_end_expr, tile_cfg, pragma_cfg, gate_dram, up_dram, output_dram):
    tile_t = get_llama3_8b_token_tile(seq_capacity, tile_cfg)
    ffn_chunk = tile_cfg["ffn_chunk"]
    chunk_col_factor = pragma_cfg["chunk_col_factor"]
    ops.extend([
        make_loop_begin(f"{prefix}_t_loop", "t_base", 0, seq_end_expr, tile_t, "valid_t", tile_t),
        make_loop_begin(f"{prefix}_c_loop", "c_base", 0, LLAMA3_8B_FFN, ffn_chunk, "valid_c", ffn_chunk),
        (
            f"{prefix}_load_gate",
            {
                "func_name": "load_matrix_tile",
                "dims": [seq_capacity, LLAMA3_8B_FFN, tile_t, ffn_chunk, chunk_col_factor],
                "args": [gate_dram, "BRAM_gate_chunk", "t_base", "c_base"],
            },
        ),
        (
            f"{prefix}_load_up",
            {
                "func_name": "load_matrix_tile",
                "dims": [seq_capacity, LLAMA3_8B_FFN, tile_t, ffn_chunk, chunk_col_factor],
                "args": [up_dram, "BRAM_up_chunk", "t_base", "c_base"],
            },
        ),
        (
            f"{prefix}_silu",
            {
                "func_name": "activation_tile_2d",
                "dims": [tile_t, ffn_chunk, chunk_col_factor],
                "func_info": ["activation_tile_2d", "silu"],
                "args": ["BRAM_gate_chunk", "BRAM_gate_chunk", "valid_t", "valid_c"],
            },
        ),
        (
            f"{prefix}_mult",
            {
                "func_name": "elementwise_mult_tile_2d",
                "dims": [tile_t, ffn_chunk, chunk_col_factor],
                "args": ["BRAM_gate_chunk", "BRAM_up_chunk", "BRAM_ffn_chunk", "valid_t", "valid_c"],
            },
        ),
        (
            f"{prefix}_store",
            {
                "func_name": "store_matrix_tile",
                "dims": [seq_capacity, LLAMA3_8B_FFN, tile_t, ffn_chunk, chunk_col_factor],
                "args": ["BRAM_ffn_chunk", output_dram, "t_base", "c_base", "valid_t", "valid_c"],
            },
        ),
        make_loop_end(f"{prefix}_c_loop_end"),
        make_loop_end(f"{prefix}_t_loop_end"),
    ])


def append_llama_kv_store_ops(ops, prefix, src_rows, cache_ctx, seq_end_expr, tile_t, input_k_dram, input_v_dram, cache_base_expr, layer_idx):
    ops.extend([
        make_loop_begin(f"{prefix}_t_loop", "t_base", 0, seq_end_expr, tile_t, "valid_t", tile_t),
        (
            f"{prefix}_store_k",
            {
                "func_name": "kv_cache_store_tile",
                "dims": [src_rows, cache_ctx, LLAMA3_8B_KV_HEADS, LLAMA3_8B_HEAD_DIM, tile_t, LLAMA3_8B_KV_DIM],
                "args": [input_k_dram, "DRAM_k_cache", str(layer_idx), "t_base", cache_base_expr, "valid_t"],
            },
        ),
        (
            f"{prefix}_store_v",
            {
                "func_name": "kv_cache_store_tile",
                "dims": [src_rows, cache_ctx, LLAMA3_8B_KV_HEADS, LLAMA3_8B_HEAD_DIM, tile_t, LLAMA3_8B_KV_DIM],
                "args": [input_v_dram, "DRAM_v_cache", str(layer_idx), "t_base", cache_base_expr, "valid_t"],
            },
        ),
        make_loop_end(f"{prefix}_t_loop_end"),
    ])


def append_llama_attention_ops(
    ops,
    prefix,
    seq_capacity,
    cache_ctx,
    q_end_expr,
    k_end_expr,
    q_row_base_expr,
    q_index_base_expr,
    output_row_base_expr,
    q_dram,
    output_dram,
    tile_cfg,
    pragma_cfg,
    layer_idx,
):
    tile_t = get_llama3_8b_token_tile(seq_capacity, tile_cfg)
    q_cols = tile_cfg["q_head_cols"]
    k_tile = tile_cfg["k_token_tile"]
    q_head_tile = tile_cfg["q_head_tile"]
    local_kv_heads = tile_cfg["local_kv_heads"]
    local_kv_cols = tile_cfg["local_kv_cols"]
    attn_head_factor = pragma_cfg["attn_head_factor"]
    attn_dim_factor = pragma_cfg["attn_dim_factor"]
    ops.extend([
        make_loop_begin(f"{prefix}_t_loop", "t_base", 0, q_end_expr, tile_t, "valid_t", tile_t),
        make_loop_begin(f"{prefix}_qh_loop", "qh_base", 0, LLAMA3_8B_Q_HEADS, q_head_tile, "valid_qh", q_head_tile),
        (
            f"{prefix}_init_rowmax",
            {
                "func_name": "init_rowmax_tile",
                "dims": [tile_t, q_head_tile, attn_head_factor],
                "args": ["BRAM_rowmax"],
            },
        ),
        (
            f"{prefix}_clear_rowsum",
            {
                "func_name": "clear_matrix_tile",
                "dims": [tile_t, q_head_tile, attn_head_factor],
                "args": ["BRAM_rowsum"],
            },
        ),
        (
            f"{prefix}_clear_ctx",
            {
                "func_name": "clear_matrix_tile",
                "dims": [tile_t, q_cols, attn_dim_factor],
                "args": ["BRAM_ctx_tile"],
            },
        ),
        (
            f"{prefix}_load_q",
            {
                "func_name": "load_matrix_tile",
                "dims": [seq_capacity, LLAMA3_8B_HIDDEN, tile_t, q_cols, attn_dim_factor],
                "args": [q_dram, "BRAM_q_tile", q_row_base_expr, "(qh_base * 128)"],
            },
        ),
        make_loop_begin(f"{prefix}_k1_loop", "k_base", 0, k_end_expr, k_tile, "valid_k", k_tile),
        (
            f"{prefix}_load_k_pass1",
            {
                "func_name": "kv_cache_load_group_tile",
                "dims": [cache_ctx, LLAMA3_8B_KV_HEADS, LLAMA3_8B_HEAD_DIM, k_tile, local_kv_heads, attn_dim_factor],
                "args": ["DRAM_k_cache", "BRAM_k_tile", str(layer_idx), "(qh_base / 4)", "k_base", "valid_k"],
            },
        ),
        (
            f"{prefix}_score_pass1",
            {
                "func_name": "attention_score_tile",
                "dims": [tile_t, q_head_tile, k_tile, LLAMA3_8B_HEAD_DIM, local_kv_heads, attn_head_factor, attn_dim_factor],
                "args": ["BRAM_q_tile", "BRAM_k_tile", "BRAM_score_tile", "valid_t", "valid_k", q_index_base_expr, "k_base"],
            },
        ),
        (
            f"{prefix}_rowmax",
            {
                "func_name": "attention_rowmax_tile",
                "dims": [tile_t, q_head_tile, k_tile, attn_head_factor],
                "args": ["BRAM_score_tile", "BRAM_rowmax", "valid_t", "valid_k"],
            },
        ),
        make_loop_end(f"{prefix}_k1_loop_end"),
        make_loop_begin(f"{prefix}_k2_loop", "k_base", 0, k_end_expr, k_tile, "valid_k", k_tile),
        (
            f"{prefix}_load_k_pass2",
            {
                "func_name": "kv_cache_load_group_tile",
                "dims": [cache_ctx, LLAMA3_8B_KV_HEADS, LLAMA3_8B_HEAD_DIM, k_tile, local_kv_heads, attn_dim_factor],
                "args": ["DRAM_k_cache", "BRAM_k_tile", str(layer_idx), "(qh_base / 4)", "k_base", "valid_k"],
            },
        ),
        (
            f"{prefix}_load_v_pass2",
            {
                "func_name": "kv_cache_load_group_tile",
                "dims": [cache_ctx, LLAMA3_8B_KV_HEADS, LLAMA3_8B_HEAD_DIM, k_tile, local_kv_heads, attn_dim_factor],
                "args": ["DRAM_v_cache", "BRAM_v_tile", str(layer_idx), "(qh_base / 4)", "k_base", "valid_k"],
            },
        ),
        (
            f"{prefix}_score_pass2",
            {
                "func_name": "attention_score_tile",
                "dims": [tile_t, q_head_tile, k_tile, LLAMA3_8B_HEAD_DIM, local_kv_heads, attn_head_factor, attn_dim_factor],
                "args": ["BRAM_q_tile", "BRAM_k_tile", "BRAM_score_tile", "valid_t", "valid_k", q_index_base_expr, "k_base"],
            },
        ),
        (
            f"{prefix}_ctx_acc",
            {
                "func_name": "attention_softmax_context_tile",
                "dims": [tile_t, q_head_tile, k_tile, LLAMA3_8B_HEAD_DIM, local_kv_heads, attn_head_factor, attn_dim_factor],
                "args": ["BRAM_score_tile", "BRAM_v_tile", "BRAM_rowmax", "BRAM_rowsum", "BRAM_ctx_tile", "valid_t", "valid_k"],
            },
        ),
        make_loop_end(f"{prefix}_k2_loop_end"),
        (
            f"{prefix}_ctx_finalize",
            {
                "func_name": "attention_finalize_tile",
                "dims": [tile_t, q_head_tile, LLAMA3_8B_HEAD_DIM, attn_head_factor, attn_dim_factor],
                "args": ["BRAM_ctx_tile", "BRAM_rowsum", "valid_t"],
            },
        ),
        (
            f"{prefix}_store_ctx",
            {
                "func_name": "store_matrix_tile",
                "dims": [seq_capacity, LLAMA3_8B_HIDDEN, tile_t, q_cols, attn_dim_factor],
                "args": ["BRAM_ctx_tile", output_dram, output_row_base_expr, "(qh_base * 128)", "valid_t", str(q_cols)],
            },
        ),
        make_loop_end(f"{prefix}_qh_loop_end"),
        make_loop_end(f"{prefix}_t_loop_end"),
    ])


def append_llama_rope_ops(ops, prefix, seq_capacity, seq_end_expr, q_row_base_expr, q_index_base_expr, q_dram, k_dram, tile_cfg, pragma_cfg):
    tile_t = get_llama3_8b_token_tile(seq_capacity, tile_cfg)
    q_cols = tile_cfg["q_head_cols"]
    attn_head_factor = pragma_cfg["attn_head_factor"]
    attn_dim_factor = pragma_cfg["attn_dim_factor"]
    ops.extend([
        make_loop_begin(f"{prefix}_q_t_loop", "t_base", 0, seq_end_expr, tile_t, "valid_t", tile_t),
        make_loop_begin(f"{prefix}_q_h_loop", "qh_base", 0, LLAMA3_8B_Q_HEADS, tile_cfg["q_head_tile"], "valid_qh", tile_cfg["q_head_tile"]),
        (
            f"{prefix}_load_q",
            {
                "func_name": "load_matrix_tile",
                "dims": [seq_capacity, LLAMA3_8B_HIDDEN, tile_t, q_cols, attn_dim_factor],
                "args": [q_dram, "BRAM_q_tile", q_row_base_expr, "(qh_base * 128)"],
            },
        ),
        (
            f"{prefix}_rope_q",
            {
                "func_name": "apply_rope_tile",
                "dims": [tile_t, tile_cfg["q_head_tile"], LLAMA3_8B_HEAD_DIM, attn_head_factor, attn_dim_factor],
                "args": ["BRAM_q_tile", "BRAM_q_tile", q_index_base_expr, "qh_base", "valid_t"],
            },
        ),
        (
            f"{prefix}_store_q",
            {
                "func_name": "store_matrix_tile",
                "dims": [seq_capacity, LLAMA3_8B_HIDDEN, tile_t, q_cols, attn_dim_factor],
                "args": ["BRAM_q_tile", q_dram, q_row_base_expr, "(qh_base * 128)", "valid_t", str(q_cols)],
            },
        ),
        make_loop_end(f"{prefix}_q_h_loop_end"),
        make_loop_end(f"{prefix}_q_t_loop_end"),
        make_loop_begin(f"{prefix}_k_t_loop", "t_base", 0, seq_end_expr, tile_t, "valid_t", tile_t),
        make_loop_begin(f"{prefix}_k_h_loop", "kh_base", 0, LLAMA3_8B_KV_HEADS, 1, "valid_kh", 1),
        (
            f"{prefix}_load_k",
            {
                "func_name": "load_matrix_tile",
                "dims": [seq_capacity, LLAMA3_8B_KV_DIM, tile_t, LLAMA3_8B_HEAD_DIM, attn_dim_factor],
                "args": [k_dram, "BRAM_k_rope", "t_base", "(kh_base * 128)"],
            },
        ),
        (
            f"{prefix}_rope_k",
            {
                "func_name": "apply_rope_tile",
                "dims": [tile_t, 1, LLAMA3_8B_HEAD_DIM, attn_head_factor, attn_dim_factor],
                "args": ["BRAM_k_rope", "BRAM_k_rope", q_index_base_expr, "kh_base", "valid_t"],
            },
        ),
        (
            f"{prefix}_store_k",
            {
                "func_name": "store_matrix_tile",
                "dims": [seq_capacity, LLAMA3_8B_KV_DIM, tile_t, LLAMA3_8B_HEAD_DIM, attn_dim_factor],
                "args": ["BRAM_k_rope", k_dram, "t_base", "(kh_base * 128)", "valid_t", str(LLAMA3_8B_HEAD_DIM)],
            },
        ),
        make_loop_end(f"{prefix}_k_h_loop_end"),
        make_loop_end(f"{prefix}_k_t_loop_end"),
    ])


def generate_llama3_8b_prefill_architecture(max_ctx, tile_cfg=None, pragma_cfg=None):
    tile_cfg = build_llama3_8b_tile_config(tile_cfg)
    pragma_cfg = build_llama3_8b_pragma_config(pragma_cfg)
    tile_t = tile_cfg["token_tile_prefill"]
    hidden_chunk = tile_cfg["hidden_chunk"]
    q_cols = tile_cfg["q_head_cols"]
    local_kv_cols = tile_cfg["local_kv_cols"]
    max_out_chunk = tile_cfg["max_out_chunk"]

    brams = [
        {"name": "BRAM_prefill_len", "dims": [1]},
        {"name": "BRAM_gamma_chunk", "dims": [hidden_chunk]},
        {"name": "BRAM_hidden_a", "dims": [tile_t, hidden_chunk]},
        {"name": "BRAM_hidden_b", "dims": [tile_t, hidden_chunk]},
        {"name": "BRAM_rms_sumsq", "dims": [tile_t]},
        {"name": "BRAM_rms_inv", "dims": [tile_t]},
        {"name": "BRAM_matrix_in", "dims": [tile_t, tile_cfg["in_chunk"]]},
        {"name": "BRAM_matrix_out", "dims": [tile_t, max_out_chunk]},
        {"name": "BRAM_weight_tile", "dims": [max_out_chunk, tile_cfg["in_chunk"]]},
        {"name": "BRAM_gate_chunk", "dims": [tile_t, tile_cfg["ffn_chunk"]]},
        {"name": "BRAM_up_chunk", "dims": [tile_t, tile_cfg["ffn_chunk"]]},
        {"name": "BRAM_ffn_chunk", "dims": [tile_t, tile_cfg["ffn_chunk"]]},
        {"name": "BRAM_q_tile", "dims": [tile_t, q_cols]},
        {"name": "BRAM_k_rope", "dims": [tile_t, LLAMA3_8B_HEAD_DIM]},
        {"name": "BRAM_ctx_tile", "dims": [tile_t, q_cols]},
        {"name": "BRAM_k_tile", "dims": [tile_cfg["k_token_tile"], local_kv_cols]},
        {"name": "BRAM_v_tile", "dims": [tile_cfg["k_token_tile"], local_kv_cols]},
        {"name": "BRAM_score_tile", "dims": [tile_t, tile_cfg["q_head_tile"], tile_cfg["k_token_tile"]]},
        {"name": "BRAM_rowmax", "dims": [tile_t, tile_cfg["q_head_tile"]]},
        {"name": "BRAM_rowsum", "dims": [tile_t, tile_cfg["q_head_tile"]]},
    ]

    drams = [
        {"name": "DRAM_token_ids", "dims": [max_ctx], "bundle": "mem_token_ids"},
        {"name": "DRAM_prefill_len", "dims": [1], "bundle": "mem_prefill_len"},
        {"name": "DRAM_embedding", "dims": [LLAMA3_8B_VOCAB, LLAMA3_8B_HIDDEN], "bundle": "mem_embedding"},
        {"name": "DRAM_final_norm", "dims": [LLAMA3_8B_HIDDEN], "bundle": "mem_final_norm"},
        {"name": "DRAM_lm_head", "dims": [LLAMA3_8B_VOCAB, LLAMA3_8B_HIDDEN], "bundle": "mem_lm_head"},
        {"name": "DRAM_attn_norm", "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_HIDDEN], "bundle": "mem_attn_norm"},
        {"name": "DRAM_q_proj", "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_HIDDEN, LLAMA3_8B_HIDDEN], "bundle": "mem_q_proj"},
        {"name": "DRAM_k_proj", "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_KV_DIM, LLAMA3_8B_HIDDEN], "bundle": "mem_k_proj"},
        {"name": "DRAM_v_proj", "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_KV_DIM, LLAMA3_8B_HIDDEN], "bundle": "mem_v_proj"},
        {"name": "DRAM_o_proj", "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_HIDDEN, LLAMA3_8B_HIDDEN], "bundle": "mem_o_proj"},
        {"name": "DRAM_ffn_norm", "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_HIDDEN], "bundle": "mem_ffn_norm"},
        {"name": "DRAM_gate_proj", "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_FFN, LLAMA3_8B_HIDDEN], "bundle": "mem_gate_proj"},
        {"name": "DRAM_up_proj", "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_FFN, LLAMA3_8B_HIDDEN], "bundle": "mem_up_proj"},
        {"name": "DRAM_down_proj", "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_HIDDEN, LLAMA3_8B_FFN], "bundle": "mem_down_proj"},
        {"name": "DRAM_hidden_ping", "dims": [max_ctx, LLAMA3_8B_HIDDEN], "bundle": "mem_hidden_ping"},
        {"name": "DRAM_hidden_pong", "dims": [max_ctx, LLAMA3_8B_HIDDEN], "bundle": "mem_hidden_pong"},
        {"name": "DRAM_norm1", "dims": [max_ctx, LLAMA3_8B_HIDDEN], "bundle": "mem_norm1"},
        {"name": "DRAM_q", "dims": [max_ctx, LLAMA3_8B_HIDDEN], "bundle": "mem_q"},
        {"name": "DRAM_k", "dims": [max_ctx, LLAMA3_8B_KV_DIM], "bundle": "mem_k"},
        {"name": "DRAM_v", "dims": [max_ctx, LLAMA3_8B_KV_DIM], "bundle": "mem_v"},
        {"name": "DRAM_attn", "dims": [max_ctx, LLAMA3_8B_HIDDEN], "bundle": "mem_attn"},
        {"name": "DRAM_mid", "dims": [max_ctx, LLAMA3_8B_HIDDEN], "bundle": "mem_mid"},
        {"name": "DRAM_norm2", "dims": [max_ctx, LLAMA3_8B_HIDDEN], "bundle": "mem_norm2"},
        {"name": "DRAM_gate", "dims": [max_ctx, LLAMA3_8B_FFN], "bundle": "mem_gate"},
        {"name": "DRAM_up", "dims": [max_ctx, LLAMA3_8B_FFN], "bundle": "mem_up"},
        {"name": "DRAM_ffn", "dims": [max_ctx, LLAMA3_8B_FFN], "bundle": "mem_ffn"},
        {"name": "DRAM_ffn_out", "dims": [max_ctx, LLAMA3_8B_HIDDEN], "bundle": "mem_ffn_out"},
        {"name": "DRAM_k_cache", "dims": [LLAMA3_8B_LAYERS, max_ctx, LLAMA3_8B_KV_HEADS, LLAMA3_8B_HEAD_DIM], "bundle": "mem_k_cache"},
        {"name": "DRAM_v_cache", "dims": [LLAMA3_8B_LAYERS, max_ctx, LLAMA3_8B_KV_HEADS, LLAMA3_8B_HEAD_DIM], "bundle": "mem_v_cache"},
        {"name": "DRAM_logits", "dims": [max_ctx, LLAMA3_8B_VOCAB], "bundle": "mem_logits"},
    ]

    ops = [
        ("load_prefill_len", {"func_name": "load", "dims": [1], "args": ["DRAM_prefill_len", "BRAM_prefill_len"]}),
    ]
    append_llama_embedding_ops(ops, "embed", max_ctx, "(int)BRAM_prefill_len[0]", tile_cfg, pragma_cfg, "DRAM_token_ids", "DRAM_hidden_ping")

    current_hidden = "DRAM_hidden_ping"
    next_hidden = "DRAM_hidden_pong"
    for layer_idx in range(LLAMA3_8B_LAYERS):
        append_llama_rmsnorm_ops(ops, f"l{layer_idx}_attn_norm", max_ctx, "(int)BRAM_prefill_len[0]", tile_cfg, pragma_cfg, current_hidden, "DRAM_norm1", "DRAM_attn_norm", layer_idx)
        append_llama_linear_ops(ops, f"l{layer_idx}_qproj", max_ctx, "(int)BRAM_prefill_len[0]", tile_t, "DRAM_norm1", "DRAM_q_proj", "DRAM_q", LLAMA3_8B_HIDDEN, LLAMA3_8B_HIDDEN, max_out_chunk, tile_cfg, pragma_cfg, layer_idx)
        append_llama_linear_ops(ops, f"l{layer_idx}_kproj", max_ctx, "(int)BRAM_prefill_len[0]", tile_t, "DRAM_norm1", "DRAM_k_proj", "DRAM_k", LLAMA3_8B_HIDDEN, LLAMA3_8B_KV_DIM, max_out_chunk, tile_cfg, pragma_cfg, layer_idx)
        append_llama_linear_ops(ops, f"l{layer_idx}_vproj", max_ctx, "(int)BRAM_prefill_len[0]", tile_t, "DRAM_norm1", "DRAM_v_proj", "DRAM_v", LLAMA3_8B_HIDDEN, LLAMA3_8B_KV_DIM, max_out_chunk, tile_cfg, pragma_cfg, layer_idx)
        append_llama_rope_ops(ops, f"l{layer_idx}_rope", max_ctx, "(int)BRAM_prefill_len[0]", "t_base", "t_base", "DRAM_q", "DRAM_k", tile_cfg, pragma_cfg)
        append_llama_kv_store_ops(ops, f"l{layer_idx}_cache", max_ctx, max_ctx, "(int)BRAM_prefill_len[0]", tile_t, "DRAM_k", "DRAM_v", "t_base", layer_idx)
        append_llama_attention_ops(ops, f"l{layer_idx}_attn", max_ctx, max_ctx, "(int)BRAM_prefill_len[0]", "(int)BRAM_prefill_len[0]", "t_base", "t_base", "t_base", "DRAM_q", "DRAM_attn", tile_cfg, pragma_cfg, layer_idx)
        append_llama_linear_ops(ops, f"l{layer_idx}_oproj", max_ctx, "(int)BRAM_prefill_len[0]", tile_t, "DRAM_attn", "DRAM_o_proj", "DRAM_attn", LLAMA3_8B_HIDDEN, LLAMA3_8B_HIDDEN, max_out_chunk, tile_cfg, pragma_cfg, layer_idx)
        append_llama_residual_add_ops(ops, f"l{layer_idx}_res1", max_ctx, "(int)BRAM_prefill_len[0]", tile_cfg, pragma_cfg, current_hidden, "DRAM_attn", "DRAM_mid")
        append_llama_rmsnorm_ops(ops, f"l{layer_idx}_ffn_norm", max_ctx, "(int)BRAM_prefill_len[0]", tile_cfg, pragma_cfg, "DRAM_mid", "DRAM_norm2", "DRAM_ffn_norm", layer_idx)
        append_llama_linear_ops(ops, f"l{layer_idx}_gate", max_ctx, "(int)BRAM_prefill_len[0]", tile_t, "DRAM_norm2", "DRAM_gate_proj", "DRAM_gate", LLAMA3_8B_HIDDEN, LLAMA3_8B_FFN, tile_cfg["ffn_chunk"], tile_cfg, pragma_cfg, layer_idx)
        append_llama_linear_ops(ops, f"l{layer_idx}_up", max_ctx, "(int)BRAM_prefill_len[0]", tile_t, "DRAM_norm2", "DRAM_up_proj", "DRAM_up", LLAMA3_8B_HIDDEN, LLAMA3_8B_FFN, tile_cfg["ffn_chunk"], tile_cfg, pragma_cfg, layer_idx)
        append_llama_swiglu_ops(ops, f"l{layer_idx}_swiglu", max_ctx, "(int)BRAM_prefill_len[0]", tile_cfg, pragma_cfg, "DRAM_gate", "DRAM_up", "DRAM_ffn")
        append_llama_linear_ops(ops, f"l{layer_idx}_down", max_ctx, "(int)BRAM_prefill_len[0]", tile_t, "DRAM_ffn", "DRAM_down_proj", "DRAM_ffn_out", LLAMA3_8B_FFN, LLAMA3_8B_HIDDEN, max_out_chunk, tile_cfg, pragma_cfg, layer_idx)
        append_llama_residual_add_ops(ops, f"l{layer_idx}_res2", max_ctx, "(int)BRAM_prefill_len[0]", tile_cfg, pragma_cfg, "DRAM_mid", "DRAM_ffn_out", next_hidden)
        current_hidden, next_hidden = next_hidden, current_hidden

    append_llama_rmsnorm_ops(ops, "final_norm", max_ctx, "(int)BRAM_prefill_len[0]", tile_cfg, pragma_cfg, current_hidden, "DRAM_norm1", "DRAM_final_norm", None)
    append_llama_linear_ops(ops, "lm_head", max_ctx, "(int)BRAM_prefill_len[0]", tile_t, "DRAM_norm1", "DRAM_lm_head", "DRAM_logits", LLAMA3_8B_HIDDEN, LLAMA3_8B_VOCAB, max_out_chunk, tile_cfg, pragma_cfg, None)
    return brams, drams, ops


def generate_llama3_8b_decode_architecture(max_ctx, tile_cfg=None, pragma_cfg=None):
    tile_cfg = build_llama3_8b_tile_config(tile_cfg)
    pragma_cfg = build_llama3_8b_pragma_config(pragma_cfg)
    tile_t = tile_cfg["token_tile_decode"]
    hidden_chunk = tile_cfg["hidden_chunk"]
    q_cols = tile_cfg["q_head_cols"]
    local_kv_cols = tile_cfg["local_kv_cols"]
    max_out_chunk = tile_cfg["max_out_chunk"]

    brams = [
        {"name": "BRAM_decode_pos", "dims": [1]},
        {"name": "BRAM_gamma_chunk", "dims": [hidden_chunk]},
        {"name": "BRAM_hidden_a", "dims": [tile_t, hidden_chunk]},
        {"name": "BRAM_hidden_b", "dims": [tile_t, hidden_chunk]},
        {"name": "BRAM_rms_sumsq", "dims": [tile_t]},
        {"name": "BRAM_rms_inv", "dims": [tile_t]},
        {"name": "BRAM_matrix_in", "dims": [tile_t, tile_cfg["in_chunk"]]},
        {"name": "BRAM_matrix_out", "dims": [tile_t, max_out_chunk]},
        {"name": "BRAM_weight_tile", "dims": [max_out_chunk, tile_cfg["in_chunk"]]},
        {"name": "BRAM_gate_chunk", "dims": [tile_t, tile_cfg["ffn_chunk"]]},
        {"name": "BRAM_up_chunk", "dims": [tile_t, tile_cfg["ffn_chunk"]]},
        {"name": "BRAM_ffn_chunk", "dims": [tile_t, tile_cfg["ffn_chunk"]]},
        {"name": "BRAM_q_tile", "dims": [tile_t, q_cols]},
        {"name": "BRAM_k_rope", "dims": [tile_t, LLAMA3_8B_HEAD_DIM]},
        {"name": "BRAM_ctx_tile", "dims": [tile_t, q_cols]},
        {"name": "BRAM_k_tile", "dims": [tile_cfg["k_token_tile"], local_kv_cols]},
        {"name": "BRAM_v_tile", "dims": [tile_cfg["k_token_tile"], local_kv_cols]},
        {"name": "BRAM_score_tile", "dims": [tile_t, tile_cfg["q_head_tile"], tile_cfg["k_token_tile"]]},
        {"name": "BRAM_rowmax", "dims": [tile_t, tile_cfg["q_head_tile"]]},
        {"name": "BRAM_rowsum", "dims": [tile_t, tile_cfg["q_head_tile"]]},
    ]

    drams = [
        {"name": "DRAM_token_id", "dims": [1], "bundle": "mem_token_id"},
        {"name": "DRAM_decode_pos", "dims": [1], "bundle": "mem_decode_pos"},
        {"name": "DRAM_embedding", "dims": [LLAMA3_8B_VOCAB, LLAMA3_8B_HIDDEN], "bundle": "mem_embedding"},
        {"name": "DRAM_final_norm", "dims": [LLAMA3_8B_HIDDEN], "bundle": "mem_final_norm"},
        {"name": "DRAM_lm_head", "dims": [LLAMA3_8B_VOCAB, LLAMA3_8B_HIDDEN], "bundle": "mem_lm_head"},
        {"name": "DRAM_attn_norm", "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_HIDDEN], "bundle": "mem_attn_norm"},
        {"name": "DRAM_q_proj", "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_HIDDEN, LLAMA3_8B_HIDDEN], "bundle": "mem_q_proj"},
        {"name": "DRAM_k_proj", "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_KV_DIM, LLAMA3_8B_HIDDEN], "bundle": "mem_k_proj"},
        {"name": "DRAM_v_proj", "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_KV_DIM, LLAMA3_8B_HIDDEN], "bundle": "mem_v_proj"},
        {"name": "DRAM_o_proj", "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_HIDDEN, LLAMA3_8B_HIDDEN], "bundle": "mem_o_proj"},
        {"name": "DRAM_ffn_norm", "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_HIDDEN], "bundle": "mem_ffn_norm"},
        {"name": "DRAM_gate_proj", "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_FFN, LLAMA3_8B_HIDDEN], "bundle": "mem_gate_proj"},
        {"name": "DRAM_up_proj", "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_FFN, LLAMA3_8B_HIDDEN], "bundle": "mem_up_proj"},
        {"name": "DRAM_down_proj", "dims": [LLAMA3_8B_LAYERS, LLAMA3_8B_HIDDEN, LLAMA3_8B_FFN], "bundle": "mem_down_proj"},
        {"name": "DRAM_hidden_ping", "dims": [1, LLAMA3_8B_HIDDEN], "bundle": "mem_hidden_ping"},
        {"name": "DRAM_hidden_pong", "dims": [1, LLAMA3_8B_HIDDEN], "bundle": "mem_hidden_pong"},
        {"name": "DRAM_norm1", "dims": [1, LLAMA3_8B_HIDDEN], "bundle": "mem_norm1"},
        {"name": "DRAM_q", "dims": [1, LLAMA3_8B_HIDDEN], "bundle": "mem_q"},
        {"name": "DRAM_k", "dims": [1, LLAMA3_8B_KV_DIM], "bundle": "mem_k"},
        {"name": "DRAM_v", "dims": [1, LLAMA3_8B_KV_DIM], "bundle": "mem_v"},
        {"name": "DRAM_attn", "dims": [1, LLAMA3_8B_HIDDEN], "bundle": "mem_attn"},
        {"name": "DRAM_mid", "dims": [1, LLAMA3_8B_HIDDEN], "bundle": "mem_mid"},
        {"name": "DRAM_norm2", "dims": [1, LLAMA3_8B_HIDDEN], "bundle": "mem_norm2"},
        {"name": "DRAM_gate", "dims": [1, LLAMA3_8B_FFN], "bundle": "mem_gate"},
        {"name": "DRAM_up", "dims": [1, LLAMA3_8B_FFN], "bundle": "mem_up"},
        {"name": "DRAM_ffn", "dims": [1, LLAMA3_8B_FFN], "bundle": "mem_ffn"},
        {"name": "DRAM_ffn_out", "dims": [1, LLAMA3_8B_HIDDEN], "bundle": "mem_ffn_out"},
        {"name": "DRAM_k_cache", "dims": [LLAMA3_8B_LAYERS, max_ctx, LLAMA3_8B_KV_HEADS, LLAMA3_8B_HEAD_DIM], "bundle": "mem_k_cache"},
        {"name": "DRAM_v_cache", "dims": [LLAMA3_8B_LAYERS, max_ctx, LLAMA3_8B_KV_HEADS, LLAMA3_8B_HEAD_DIM], "bundle": "mem_v_cache"},
        {"name": "DRAM_logits_decode", "dims": [1, LLAMA3_8B_VOCAB], "bundle": "mem_logits"},
    ]

    ops = [
        ("load_decode_pos", {"func_name": "load", "dims": [1], "args": ["DRAM_decode_pos", "BRAM_decode_pos"]}),
    ]
    append_llama_embedding_ops(ops, "embed_decode", 1, "1", tile_cfg, pragma_cfg, "DRAM_token_id", "DRAM_hidden_ping")
    current_hidden = "DRAM_hidden_ping"
    next_hidden = "DRAM_hidden_pong"
    for layer_idx in range(LLAMA3_8B_LAYERS):
        append_llama_rmsnorm_ops(ops, f"d{layer_idx}_attn_norm", 1, "1", tile_cfg, pragma_cfg, current_hidden, "DRAM_norm1", "DRAM_attn_norm", layer_idx)
        append_llama_linear_ops(ops, f"d{layer_idx}_qproj", 1, "1", tile_t, "DRAM_norm1", "DRAM_q_proj", "DRAM_q", LLAMA3_8B_HIDDEN, LLAMA3_8B_HIDDEN, max_out_chunk, tile_cfg, pragma_cfg, layer_idx)
        append_llama_linear_ops(ops, f"d{layer_idx}_kproj", 1, "1", tile_t, "DRAM_norm1", "DRAM_k_proj", "DRAM_k", LLAMA3_8B_HIDDEN, LLAMA3_8B_KV_DIM, max_out_chunk, tile_cfg, pragma_cfg, layer_idx)
        append_llama_linear_ops(ops, f"d{layer_idx}_vproj", 1, "1", tile_t, "DRAM_norm1", "DRAM_v_proj", "DRAM_v", LLAMA3_8B_HIDDEN, LLAMA3_8B_KV_DIM, max_out_chunk, tile_cfg, pragma_cfg, layer_idx)
        append_llama_rope_ops(ops, f"d{layer_idx}_rope", 1, "1", "0", "(int)BRAM_decode_pos[0]", "DRAM_q", "DRAM_k", tile_cfg, pragma_cfg)
        append_llama_kv_store_ops(ops, f"d{layer_idx}_cache", 1, max_ctx, "1", tile_t, "DRAM_k", "DRAM_v", "(int)BRAM_decode_pos[0]", layer_idx)
        append_llama_attention_ops(ops, f"d{layer_idx}_attn", 1, max_ctx, "1", "((int)BRAM_decode_pos[0] + 1)", "0", "(int)BRAM_decode_pos[0]", "0", "DRAM_q", "DRAM_attn", tile_cfg, pragma_cfg, layer_idx)
        append_llama_linear_ops(ops, f"d{layer_idx}_oproj", 1, "1", tile_t, "DRAM_attn", "DRAM_o_proj", "DRAM_attn", LLAMA3_8B_HIDDEN, LLAMA3_8B_HIDDEN, max_out_chunk, tile_cfg, pragma_cfg, layer_idx)
        append_llama_residual_add_ops(ops, f"d{layer_idx}_res1", 1, "1", tile_cfg, pragma_cfg, current_hidden, "DRAM_attn", "DRAM_mid")
        append_llama_rmsnorm_ops(ops, f"d{layer_idx}_ffn_norm", 1, "1", tile_cfg, pragma_cfg, "DRAM_mid", "DRAM_norm2", "DRAM_ffn_norm", layer_idx)
        append_llama_linear_ops(ops, f"d{layer_idx}_gate", 1, "1", tile_t, "DRAM_norm2", "DRAM_gate_proj", "DRAM_gate", LLAMA3_8B_HIDDEN, LLAMA3_8B_FFN, tile_cfg["ffn_chunk"], tile_cfg, pragma_cfg, layer_idx)
        append_llama_linear_ops(ops, f"d{layer_idx}_up", 1, "1", tile_t, "DRAM_norm2", "DRAM_up_proj", "DRAM_up", LLAMA3_8B_HIDDEN, LLAMA3_8B_FFN, tile_cfg["ffn_chunk"], tile_cfg, pragma_cfg, layer_idx)
        append_llama_swiglu_ops(ops, f"d{layer_idx}_swiglu", 1, "1", tile_cfg, pragma_cfg, "DRAM_gate", "DRAM_up", "DRAM_ffn")
        append_llama_linear_ops(ops, f"d{layer_idx}_down", 1, "1", tile_t, "DRAM_ffn", "DRAM_down_proj", "DRAM_ffn_out", LLAMA3_8B_FFN, LLAMA3_8B_HIDDEN, max_out_chunk, tile_cfg, pragma_cfg, layer_idx)
        append_llama_residual_add_ops(ops, f"d{layer_idx}_res2", 1, "1", tile_cfg, pragma_cfg, "DRAM_mid", "DRAM_ffn_out", next_hidden)
        current_hidden, next_hidden = next_hidden, current_hidden

    append_llama_rmsnorm_ops(ops, "decode_final_norm", 1, "1", tile_cfg, pragma_cfg, current_hidden, "DRAM_norm1", "DRAM_final_norm", None)
    append_llama_linear_ops(ops, "decode_lm_head", 1, "1", tile_t, "DRAM_norm1", "DRAM_lm_head", "DRAM_logits_decode", LLAMA3_8B_HIDDEN, LLAMA3_8B_VOCAB, max_out_chunk, tile_cfg, pragma_cfg, None)
    return brams, drams, ops


def generate_llama3_8b_prefill_config_text(max_ctx, data_type="ap_fixed<16,5>", tile_cfg=None, pragma_cfg=None):
    brams, drams, ops = generate_llama3_8b_prefill_architecture(max_ctx, tile_cfg, pragma_cfg)
    return serialize_scale_model_config(brams, drams, ops, ["DRAM_logits"], data_type=data_type)


def generate_llama3_8b_decode_config_text(max_ctx, data_type="ap_fixed<16,5>", tile_cfg=None, pragma_cfg=None):
    brams, drams, ops = generate_llama3_8b_decode_architecture(max_ctx, tile_cfg, pragma_cfg)
    return serialize_scale_model_config(brams, drams, ops, ["DRAM_logits_decode"], data_type=data_type)


RESNET_FAMILY_SPECS = {
    18: {"block_type": "basic", "layers": [2, 2, 2, 2], "expansion": 1},
    34: {"block_type": "basic", "layers": [3, 4, 6, 3], "expansion": 1},
    50: {"block_type": "bottleneck", "layers": [3, 4, 6, 3], "expansion": 4},
    101: {"block_type": "bottleneck", "layers": [3, 4, 23, 3], "expansion": 4},
    152: {"block_type": "bottleneck", "layers": [3, 8, 36, 3], "expansion": 4},
}
RESNET_STAGE_WIDTHS = [64, 128, 256, 512]
DEFAULT_RESNET_DEPTHS = [18, 34, 50, 101, 152]


def get_resnet_spec(depth):
    depth = int(depth)
    if depth not in RESNET_FAMILY_SPECS:
        raise ValueError(f"Unsupported ResNet depth: {depth}")
    return RESNET_FAMILY_SPECS[depth]


def parse_resnet_depths(depths_arg):
    depths = []
    for item in str(depths_arg).split(","):
        item = item.strip()
        if not item:
            continue
        depth = int(item)
        get_resnet_spec(depth)
        depths.append(depth)
    if not depths:
        raise ValueError("No valid ResNet depths were provided.")
    return depths


def parse_positive_int_list(values_arg, arg_name):
    values = []
    for item in str(values_arg).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"{arg_name} must contain only positive integers, got {value}.")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"{arg_name} did not contain any valid integers.")
    return values


def get_resnet_filename(depth, data_type, conv_cfg=None, preserve_legacy_name=False):
    naming_dtype = data_type.replace('<', '_').replace('>', '_').replace(',', '_')
    return f"RESNET{depth}{get_conv_factor_suffix(conv_cfg, preserve_legacy_name)}_config_{naming_dtype}.json"


def get_resnet_tiled_filename(
    depth,
    data_type,
    tiled_cfg,
    conv_cfg=None,
    preserve_legacy_conv_name=False,
    preserve_legacy_tiled_name=False,
):
    naming_dtype = data_type.replace('<', '_').replace('>', '_').replace(',', '_')
    conv_suffix = get_conv_factor_suffix(conv_cfg, preserve_legacy_conv_name)
    if preserve_legacy_tiled_name and tiled_cfg["is_default"]:
        return f"RESNET{depth}_TILED{conv_suffix}_config_{naming_dtype}.json"
    return (
        f"RESNET{depth}_TILED_oc{tiled_cfg['tile_oc']}_ic{tiled_cfg['tile_ic']}"
        f"_h{tiled_cfg['tile_h']}_w{tiled_cfg['tile_w']}{conv_suffix}_config_{naming_dtype}.json"
    )


def get_mobilenetv2_filename(data_type, conv_cfg=None, preserve_legacy_name=False):
    naming_dtype = data_type.replace('<', '_').replace('>', '_').replace(',', '_')
    return f"MOBILENETV2{get_conv_factor_suffix(conv_cfg, preserve_legacy_name)}_config_{naming_dtype}.json"


def generate_resnet_basic_block(
    prefix,
    c_in,
    c_out,
    h_in,
    w_in,
    stride,
    projection,
    bram_in,
    bram_tmp1,
    bram_tmp2,
    bram_out,
    bram_skip=None,
    conv_cfg=None,
):
    if projection and bram_skip is None:
        raise ValueError("A projection block requires a skip buffer.")
    conv_cfg = normalize_conv_factor_config(conv_cfg)

    h_out = h_in // stride
    w_out = w_in // stride
    brams = [
        {"name": f"BRAM_w_{prefix}_1", "dims": [c_out, c_in, 3, 3]},
        {"name": f"BRAM_bn_{prefix}_1", "dims": [4, c_out]},
        {"name": f"BRAM_w_{prefix}_2", "dims": [c_out, c_out, 3, 3]},
        {"name": f"BRAM_bn_{prefix}_2", "dims": [4, c_out]},
    ]
    drams = [
        {"name": f"DRAM_w_{prefix}_1", "dims": [c_out, c_in, 3, 3], "bundle": f"mem_w_{prefix}_1"},
        {"name": f"DRAM_bn_{prefix}_1", "dims": [4, c_out], "bundle": f"mem_bn_{prefix}_1"},
        {"name": f"DRAM_w_{prefix}_2", "dims": [c_out, c_out, 3, 3], "bundle": f"mem_w_{prefix}_2"},
        {"name": f"DRAM_bn_{prefix}_2", "dims": [4, c_out], "bundle": f"mem_bn_{prefix}_2"},
    ]
    ops = [
        (f"{prefix}_conv1", {
            "func_name": "conv",
            "dims": build_conv_dims(c_in, c_out, h_in, w_in, h_out, w_out, 3, 1, stride, conv_cfg=conv_cfg),
            "func_info": ["conv_template.cpp", "conv2d", False],
            "args": [bram_in, f"BRAM_w_{prefix}_1", bram_tmp1],
        }),
        (f"{prefix}_bn1", {
            "func_name": "batchnorm",
            "dims": [c_out, h_out, w_out],
            "func_info": ["batch_norm_template.cpp", False],
            "args": [bram_tmp1, f"BRAM_bn_{prefix}_1", bram_tmp1],
        }),
        (f"{prefix}_relu1", {
            "func_name": "activation_conv",
            "dims": [c_out, h_out, w_out],
            "func_info": ["activation_template_conv.cpp", "relu"],
            "args": [bram_tmp1, bram_tmp1],
        }),
        (f"{prefix}_conv2", {
            "func_name": "conv",
            "dims": build_conv_dims(c_out, c_out, h_out, w_out, h_out, w_out, 3, 1, 1, conv_cfg=conv_cfg),
            "func_info": ["conv_template.cpp", "conv2d", False],
            "args": [bram_tmp1, f"BRAM_w_{prefix}_2", bram_tmp2],
        }),
        (f"{prefix}_bn2", {
            "func_name": "batchnorm",
            "dims": [c_out, h_out, w_out],
            "func_info": ["batch_norm_template.cpp", False],
            "args": [bram_tmp2, f"BRAM_bn_{prefix}_2", bram_tmp2],
        }),
    ]

    skip_source = bram_in
    if projection:
        brams.append({"name": f"BRAM_w_{prefix}_down", "dims": [c_out, c_in, 1, 1]})
        drams.append({"name": f"DRAM_w_{prefix}_down", "dims": [c_out, c_in, 1, 1], "bundle": f"mem_w_{prefix}_down"})
        ops.append((
            f"{prefix}_downsample",
            {
                "func_name": "conv",
                "dims": build_conv_dims(c_in, c_out, h_in, w_in, h_out, w_out, 1, 0, stride, conv_cfg=conv_cfg),
                "func_info": ["conv_template.cpp", "conv2d", False],
                "args": [bram_in, f"BRAM_w_{prefix}_down", bram_skip],
            }
        ))
        skip_source = bram_skip

    ops.extend([
        (f"{prefix}_residual_add", {
            "func_name": "matrix_add_conv",
            "dims": [c_out, h_out, w_out],
            "func_info": ["matrix_add_template_conv.cpp"],
            "args": [bram_tmp2, skip_source, bram_out],
        }),
        (f"{prefix}_relu2", {
            "func_name": "activation_conv",
            "dims": [c_out, h_out, w_out],
            "func_info": ["activation_template_conv.cpp", "relu"],
            "args": [bram_out, bram_out],
        }),
    ])
    return brams, drams, ops, h_out, w_out


def generate_resnet_bottleneck_block(
    prefix,
    c_in,
    bottleneck_c,
    h_in,
    w_in,
    stride,
    projection,
    bram_in,
    bram_tmp1,
    bram_tmp2,
    bram_tmp3,
    bram_out,
    bram_skip=None,
    conv_cfg=None,
):
    if projection and bram_skip is None:
        raise ValueError("A projection block requires a skip buffer.")
    conv_cfg = normalize_conv_factor_config(conv_cfg)

    c_out = bottleneck_c * 4
    h_out = h_in // stride
    w_out = w_in // stride
    brams = [
        {"name": f"BRAM_w_{prefix}_1", "dims": [bottleneck_c, c_in, 1, 1]},
        {"name": f"BRAM_bn_{prefix}_1", "dims": [4, bottleneck_c]},
        {"name": f"BRAM_w_{prefix}_2", "dims": [bottleneck_c, bottleneck_c, 3, 3]},
        {"name": f"BRAM_bn_{prefix}_2", "dims": [4, bottleneck_c]},
        {"name": f"BRAM_w_{prefix}_3", "dims": [c_out, bottleneck_c, 1, 1]},
        {"name": f"BRAM_bn_{prefix}_3", "dims": [4, c_out]},
    ]
    drams = [
        {"name": f"DRAM_w_{prefix}_1", "dims": [bottleneck_c, c_in, 1, 1], "bundle": f"mem_w_{prefix}_1"},
        {"name": f"DRAM_bn_{prefix}_1", "dims": [4, bottleneck_c], "bundle": f"mem_bn_{prefix}_1"},
        {"name": f"DRAM_w_{prefix}_2", "dims": [bottleneck_c, bottleneck_c, 3, 3], "bundle": f"mem_w_{prefix}_2"},
        {"name": f"DRAM_bn_{prefix}_2", "dims": [4, bottleneck_c], "bundle": f"mem_bn_{prefix}_2"},
        {"name": f"DRAM_w_{prefix}_3", "dims": [c_out, bottleneck_c, 1, 1], "bundle": f"mem_w_{prefix}_3"},
        {"name": f"DRAM_bn_{prefix}_3", "dims": [4, c_out], "bundle": f"mem_bn_{prefix}_3"},
    ]
    ops = [
        (f"{prefix}_conv1", {
            "func_name": "conv",
            "dims": build_conv_dims(c_in, bottleneck_c, h_in, w_in, h_in, w_in, 1, 0, 1, conv_cfg=conv_cfg),
            "func_info": ["conv_template.cpp", "conv2d", False],
            "args": [bram_in, f"BRAM_w_{prefix}_1", bram_tmp1],
        }),
        (f"{prefix}_bn1", {
            "func_name": "batchnorm",
            "dims": [bottleneck_c, h_in, w_in],
            "func_info": ["batch_norm_template.cpp", False],
            "args": [bram_tmp1, f"BRAM_bn_{prefix}_1", bram_tmp1],
        }),
        (f"{prefix}_relu1", {
            "func_name": "activation_conv",
            "dims": [bottleneck_c, h_in, w_in],
            "func_info": ["activation_template_conv.cpp", "relu"],
            "args": [bram_tmp1, bram_tmp1],
        }),
        (f"{prefix}_conv2", {
            "func_name": "conv",
            "dims": build_conv_dims(bottleneck_c, bottleneck_c, h_in, w_in, h_out, w_out, 3, 1, stride, conv_cfg=conv_cfg),
            "func_info": ["conv_template.cpp", "conv2d", False],
            "args": [bram_tmp1, f"BRAM_w_{prefix}_2", bram_tmp2],
        }),
        (f"{prefix}_bn2", {
            "func_name": "batchnorm",
            "dims": [bottleneck_c, h_out, w_out],
            "func_info": ["batch_norm_template.cpp", False],
            "args": [bram_tmp2, f"BRAM_bn_{prefix}_2", bram_tmp2],
        }),
        (f"{prefix}_relu2", {
            "func_name": "activation_conv",
            "dims": [bottleneck_c, h_out, w_out],
            "func_info": ["activation_template_conv.cpp", "relu"],
            "args": [bram_tmp2, bram_tmp2],
        }),
        (f"{prefix}_conv3", {
            "func_name": "conv",
            "dims": build_conv_dims(bottleneck_c, c_out, h_out, w_out, h_out, w_out, 1, 0, 1, conv_cfg=conv_cfg),
            "func_info": ["conv_template.cpp", "conv2d", False],
            "args": [bram_tmp2, f"BRAM_w_{prefix}_3", bram_tmp3],
        }),
        (f"{prefix}_bn3", {
            "func_name": "batchnorm",
            "dims": [c_out, h_out, w_out],
            "func_info": ["batch_norm_template.cpp", False],
            "args": [bram_tmp3, f"BRAM_bn_{prefix}_3", bram_tmp3],
        }),
    ]

    skip_source = bram_in
    if projection:
        brams.append({"name": f"BRAM_w_{prefix}_down", "dims": [c_out, c_in, 1, 1]})
        drams.append({"name": f"DRAM_w_{prefix}_down", "dims": [c_out, c_in, 1, 1], "bundle": f"mem_w_{prefix}_down"})
        ops.append((
            f"{prefix}_downsample",
            {
                "func_name": "conv",
                "dims": build_conv_dims(c_in, c_out, h_in, w_in, h_out, w_out, 1, 0, stride, conv_cfg=conv_cfg),
                "func_info": ["conv_template.cpp", "conv2d", False],
                "args": [bram_in, f"BRAM_w_{prefix}_down", bram_skip],
            }
        ))
        skip_source = bram_skip

    ops.extend([
        (f"{prefix}_residual_add", {
            "func_name": "matrix_add_conv",
            "dims": [c_out, h_out, w_out],
            "func_info": ["matrix_add_template_conv.cpp"],
            "args": [bram_tmp3, skip_source, bram_out],
        }),
        (f"{prefix}_relu3", {
            "func_name": "activation_conv",
            "dims": [c_out, h_out, w_out],
            "func_info": ["activation_template_conv.cpp", "relu"],
            "args": [bram_out, bram_out],
        }),
    ])
    return brams, drams, ops, h_out, w_out


def generate_resnet_architecture(depth, conv_cfg=None):
    conv_cfg = normalize_conv_factor_config(conv_cfg)
    spec = get_resnet_spec(depth)
    block_type = spec["block_type"]
    expansion = spec["expansion"]
    brams = [
        {"name": "BRAM_feat_input", "dims": [3, 224, 224]},
        {"name": "BRAM_feat_stem", "dims": [64, 112, 112]},
        {"name": "BRAM_feat_pool", "dims": [64, 56, 56]},
        {"name": "BRAM_w_stem", "dims": [64, 3, 7, 7]},
        {"name": "BRAM_bn_stem", "dims": [4, 64]},
    ]
    drams = [
        {"name": "DRAM_input", "dims": [3, 224, 224], "bundle": "mem_input"},
        {"name": "DRAM_w_stem", "dims": [64, 3, 7, 7], "bundle": "mem_w_stem"},
        {"name": "DRAM_bn_stem", "dims": [4, 64], "bundle": "mem_bn_stem"},
    ]
    ops = [
        ("load_input", {
            "func_name": "load",
            "dims": [3, 224, 224],
            "args": ["DRAM_input", "BRAM_feat_input"],
        }),
        ("stem_conv", {
            "func_name": "conv",
            "dims": build_conv_dims(3, 64, 224, 224, 112, 112, 7, 3, 2, conv_cfg=conv_cfg),
            "func_info": ["conv_template.cpp", "conv2d", False],
            "args": ["BRAM_feat_input", "BRAM_w_stem", "BRAM_feat_stem"],
        }),
        ("stem_bn", {
            "func_name": "batchnorm",
            "dims": [64, 112, 112],
            "func_info": ["batch_norm_template.cpp", False],
            "args": ["BRAM_feat_stem", "BRAM_bn_stem", "BRAM_feat_stem"],
        }),
        ("stem_relu", {
            "func_name": "activation_conv",
            "dims": [64, 112, 112],
            "func_info": ["activation_template_conv.cpp", "relu"],
            "args": ["BRAM_feat_stem", "BRAM_feat_stem"],
        }),
        ("stem_pool", {
            "func_name": "maxpool_pad",
            "dims": [64, 112, 112, 56, 56, 3, 3, 2, 2, 1, 1],
            "func_info": ["maxpool_pad_template.cpp"],
            "args": ["BRAM_feat_stem", "BRAM_feat_pool"],
        }),
    ]

    current_bram = "BRAM_feat_pool"
    c_in = 64
    h_in = 56
    w_in = 56

    for stage_id, (stage_width, num_blocks) in enumerate(zip(RESNET_STAGE_WIDTHS, spec["layers"]), start=1):
        stage_stride = 1 if stage_id == 1 else 2
        stage_h = h_in // stage_stride
        stage_w = w_in // stage_stride
        stage_out_c = stage_width * expansion

        brams.append({"name": f"BRAM_feat_s{stage_id}_tmp1", "dims": [stage_width, stage_h, stage_w]})
        brams.append({"name": f"BRAM_feat_s{stage_id}_tmp2", "dims": [stage_width, stage_h, stage_w]})
        if block_type == "bottleneck":
            brams.append({"name": f"BRAM_feat_s{stage_id}_tmp3", "dims": [stage_out_c, stage_h, stage_w]})
        if stage_id > 1 or block_type == "bottleneck":
            brams.append({"name": f"BRAM_feat_s{stage_id}_skip", "dims": [stage_out_c, stage_h, stage_w]})

        for block_id in range(num_blocks):
            prefix = f"s{stage_id}_b{block_id}"
            block_stride = stage_stride if block_id == 0 else 1
            projection = block_id == 0 and (block_stride != 1 or c_in != stage_out_c)
            out_bram = f"BRAM_feat_{prefix}"
            brams.append({"name": out_bram, "dims": [stage_out_c, stage_h, stage_w]})

            if block_type == "basic":
                block_brams, block_drams, block_ops, h_out, w_out = generate_resnet_basic_block(
                    prefix=prefix,
                    c_in=c_in,
                    c_out=stage_out_c,
                    h_in=h_in,
                    w_in=w_in,
                    stride=block_stride,
                    projection=projection,
                    bram_in=current_bram,
                    bram_tmp1=f"BRAM_feat_s{stage_id}_tmp1",
                    bram_tmp2=f"BRAM_feat_s{stage_id}_tmp2",
                    bram_out=out_bram,
                    bram_skip=f"BRAM_feat_s{stage_id}_skip" if projection else None,
                    conv_cfg=conv_cfg,
                )
            else:
                block_brams, block_drams, block_ops, h_out, w_out = generate_resnet_bottleneck_block(
                    prefix=prefix,
                    c_in=c_in,
                    bottleneck_c=stage_width,
                    h_in=h_in,
                    w_in=w_in,
                    stride=block_stride,
                    projection=projection,
                    bram_in=current_bram,
                    bram_tmp1=f"BRAM_feat_s{stage_id}_tmp1",
                    bram_tmp2=f"BRAM_feat_s{stage_id}_tmp2",
                    bram_tmp3=f"BRAM_feat_s{stage_id}_tmp3",
                    bram_out=out_bram,
                    bram_skip=f"BRAM_feat_s{stage_id}_skip" if projection else None,
                    conv_cfg=conv_cfg,
                )

            brams.extend(block_brams)
            drams.extend(block_drams)
            ops.extend(block_ops)
            current_bram = out_bram
            c_in = stage_out_c
            h_in = h_out
            w_in = w_out

    brams.extend([
        {"name": "BRAM_pool", "dims": [c_in, 1, 1]},
        {"name": "BRAM_out", "dims": [1000, 1, 1]},
        {"name": "BRAM_fc", "dims": [1000, c_in, 1, 1]},
    ])
    drams.extend([
        {"name": "DRAM_out", "dims": [1000, 1, 1], "bundle": "mem_out"},
        {"name": "DRAM_fc", "dims": [1000, c_in, 1, 1], "bundle": "mem_fc"},
    ])
    ops.extend([
        ("global_pool", {
            "func_name": "adaptive_avgpool",
            "dims": [c_in, h_in, w_in, 1, 1],
            "func_info": ["adaptive_avgpool_template.cpp"],
            "args": [current_bram, "BRAM_pool"],
        }),
        ("fc", {
            "func_name": "conv",
            "dims": build_conv_dims(c_in, 1000, 1, 1, 1, 1, 1, 0, 1, conv_cfg=conv_cfg),
            "func_info": ["conv_template.cpp", "conv2d", False],
            "args": ["BRAM_pool", "BRAM_fc", "BRAM_out"],
        }),
        ("store_output", {
            "func_name": "store",
            "dims": [1000, 1, 1],
            "args": ["BRAM_out", "DRAM_out"],
        }),
    ])
    return brams, drams, ops


def generate_resnet_config_txt(depth, data_type="ap_fixed<16,5>", conv_cfg=None):
    brams, drams, ops = generate_resnet_architecture(depth, conv_cfg=conv_cfg)
    return serialize_scale_model_config(
        brams=brams,
        drams=drams,
        ops=ops,
        output_dram_names=["DRAM_out"],
        data_type=data_type,
    )


def generate_resnet_tiled_basic_block(
    prefix,
    tiled_cfg,
    conv_cfg,
    c_in,
    c_out,
    h_in,
    w_in,
    stride,
    projection,
    feat_in,
    feat_mid,
    feat_out,
    feat_skip=None,
):
    if projection and feat_skip is None:
        raise ValueError("A projection block requires a skip feature tensor.")
    conv_cfg = normalize_conv_factor_config(conv_cfg)

    h_out = h_in // stride
    w_out = w_in // stride
    drams = [
        {"name": f"DRAM_w_{prefix}_1", "dims": [c_out, c_in, 3, 3], "bundle": f"mem_w_{prefix}_1"},
        {"name": f"DRAM_bn_{prefix}_1", "dims": [4, c_out], "bundle": f"mem_bn_{prefix}_1"},
        {"name": f"DRAM_w_{prefix}_2", "dims": [c_out, c_out, 3, 3], "bundle": f"mem_w_{prefix}_2"},
        {"name": f"DRAM_bn_{prefix}_2", "dims": [4, c_out], "bundle": f"mem_bn_{prefix}_2"},
    ]
    ops = []
    append_tiled_conv_ops(
        ops=ops,
        prefix=f"{prefix}_conv1",
        tiled_cfg=tiled_cfg,
        conv_cfg=conv_cfg,
        feat_in=feat_in,
        weight_dram=f"DRAM_w_{prefix}_1",
        feat_out=feat_mid,
        c_in=c_in,
        c_out=c_out,
        h_in=h_in,
        w_in=w_in,
        h_out=h_out,
        w_out=w_out,
        kernel=3,
        pad=1,
        stride=stride,
        patch_bram="BRAM_in_patch_stride2_k3" if stride == 2 else "BRAM_in_patch_stride1_k3",
        weight_bram="BRAM_weight_tile_3",
        out_bram="BRAM_out_tile",
        out_tile_h=tiled_cfg["tile_h"],
        out_tile_w=tiled_cfg["tile_w"],
        bn_dram=f"DRAM_bn_{prefix}_1",
        relu_after=True,
    )
    append_tiled_conv_ops(
        ops=ops,
        prefix=f"{prefix}_conv2",
        tiled_cfg=tiled_cfg,
        conv_cfg=conv_cfg,
        feat_in=feat_mid,
        weight_dram=f"DRAM_w_{prefix}_2",
        feat_out=feat_out,
        c_in=c_out,
        c_out=c_out,
        h_in=h_out,
        w_in=w_out,
        h_out=h_out,
        w_out=w_out,
        kernel=3,
        pad=1,
        stride=1,
        patch_bram="BRAM_in_patch_stride1_k3",
        weight_bram="BRAM_weight_tile_3",
        out_bram="BRAM_out_tile",
        out_tile_h=tiled_cfg["tile_h"],
        out_tile_w=tiled_cfg["tile_w"],
        bn_dram=f"DRAM_bn_{prefix}_2",
        relu_after=False,
    )

    skip_source = feat_in
    if projection:
        drams.append({"name": f"DRAM_w_{prefix}_down", "dims": [c_out, c_in, 1, 1], "bundle": f"mem_w_{prefix}_down"})
        append_tiled_conv_ops(
            ops=ops,
            prefix=f"{prefix}_down",
            tiled_cfg=tiled_cfg,
            conv_cfg=conv_cfg,
            feat_in=feat_in,
            weight_dram=f"DRAM_w_{prefix}_down",
            feat_out=feat_skip,
            c_in=c_in,
            c_out=c_out,
            h_in=h_in,
            w_in=w_in,
            h_out=h_out,
            w_out=w_out,
            kernel=1,
            pad=0,
            stride=stride,
            patch_bram="BRAM_in_patch_stride2_k1" if stride == 2 else "BRAM_in_patch_1",
            weight_bram="BRAM_weight_tile_1",
            out_bram="BRAM_out_tile",
            out_tile_h=tiled_cfg["tile_h"],
            out_tile_w=tiled_cfg["tile_w"],
            bn_dram=None,
            relu_after=False,
        )
        skip_source = feat_skip

    append_tiled_residual_add_ops(
        ops=ops,
        prefix=f"{prefix}_residual",
        tiled_cfg=tiled_cfg,
        feat_main=feat_out,
        feat_skip=skip_source,
        feat_out=feat_out,
        c_out=c_out,
        h_out=h_out,
        w_out=w_out,
    )
    return drams, ops, h_out, w_out


def generate_resnet_tiled_bottleneck_block(
    prefix,
    tiled_cfg,
    conv_cfg,
    c_in,
    bottleneck_c,
    h_in,
    w_in,
    stride,
    projection,
    feat_in,
    feat_mid1,
    feat_mid2,
    feat_out,
    feat_skip=None,
):
    if projection and feat_skip is None:
        raise ValueError("A projection block requires a skip feature tensor.")
    conv_cfg = normalize_conv_factor_config(conv_cfg)

    c_out = bottleneck_c * 4
    h_out = h_in // stride
    w_out = w_in // stride
    drams = [
        {"name": f"DRAM_w_{prefix}_1", "dims": [bottleneck_c, c_in, 1, 1], "bundle": f"mem_w_{prefix}_1"},
        {"name": f"DRAM_bn_{prefix}_1", "dims": [4, bottleneck_c], "bundle": f"mem_bn_{prefix}_1"},
        {"name": f"DRAM_w_{prefix}_2", "dims": [bottleneck_c, bottleneck_c, 3, 3], "bundle": f"mem_w_{prefix}_2"},
        {"name": f"DRAM_bn_{prefix}_2", "dims": [4, bottleneck_c], "bundle": f"mem_bn_{prefix}_2"},
        {"name": f"DRAM_w_{prefix}_3", "dims": [c_out, bottleneck_c, 1, 1], "bundle": f"mem_w_{prefix}_3"},
        {"name": f"DRAM_bn_{prefix}_3", "dims": [4, c_out], "bundle": f"mem_bn_{prefix}_3"},
    ]
    ops = []
    append_tiled_conv_ops(
        ops=ops,
        prefix=f"{prefix}_conv1",
        tiled_cfg=tiled_cfg,
        conv_cfg=conv_cfg,
        feat_in=feat_in,
        weight_dram=f"DRAM_w_{prefix}_1",
        feat_out=feat_mid1,
        c_in=c_in,
        c_out=bottleneck_c,
        h_in=h_in,
        w_in=w_in,
        h_out=h_in,
        w_out=w_in,
        kernel=1,
        pad=0,
        stride=1,
        patch_bram="BRAM_in_patch_stride1_k1",
        weight_bram="BRAM_weight_tile_1",
        out_bram="BRAM_out_tile",
        out_tile_h=tiled_cfg["tile_h"],
        out_tile_w=tiled_cfg["tile_w"],
        bn_dram=f"DRAM_bn_{prefix}_1",
        relu_after=True,
    )
    append_tiled_conv_ops(
        ops=ops,
        prefix=f"{prefix}_conv2",
        tiled_cfg=tiled_cfg,
        conv_cfg=conv_cfg,
        feat_in=feat_mid1,
        weight_dram=f"DRAM_w_{prefix}_2",
        feat_out=feat_mid2,
        c_in=bottleneck_c,
        c_out=bottleneck_c,
        h_in=h_in,
        w_in=w_in,
        h_out=h_out,
        w_out=w_out,
        kernel=3,
        pad=1,
        stride=stride,
        patch_bram="BRAM_in_patch_stride2_k3" if stride == 2 else "BRAM_in_patch_stride1_k3",
        weight_bram="BRAM_weight_tile_3",
        out_bram="BRAM_out_tile",
        out_tile_h=tiled_cfg["tile_h"],
        out_tile_w=tiled_cfg["tile_w"],
        bn_dram=f"DRAM_bn_{prefix}_2",
        relu_after=True,
    )
    append_tiled_conv_ops(
        ops=ops,
        prefix=f"{prefix}_conv3",
        tiled_cfg=tiled_cfg,
        conv_cfg=conv_cfg,
        feat_in=feat_mid2,
        weight_dram=f"DRAM_w_{prefix}_3",
        feat_out=feat_out,
        c_in=bottleneck_c,
        c_out=c_out,
        h_in=h_out,
        w_in=w_out,
        h_out=h_out,
        w_out=w_out,
        kernel=1,
        pad=0,
        stride=1,
        patch_bram="BRAM_in_patch_stride1_k1",
        weight_bram="BRAM_weight_tile_1",
        out_bram="BRAM_out_tile",
        out_tile_h=tiled_cfg["tile_h"],
        out_tile_w=tiled_cfg["tile_w"],
        bn_dram=f"DRAM_bn_{prefix}_3",
        relu_after=False,
    )

    skip_source = feat_in
    if projection:
        drams.append({"name": f"DRAM_w_{prefix}_down", "dims": [c_out, c_in, 1, 1], "bundle": f"mem_w_{prefix}_down"})
        append_tiled_conv_ops(
            ops=ops,
            prefix=f"{prefix}_down",
            tiled_cfg=tiled_cfg,
            conv_cfg=conv_cfg,
            feat_in=feat_in,
            weight_dram=f"DRAM_w_{prefix}_down",
            feat_out=feat_skip,
            c_in=c_in,
            c_out=c_out,
            h_in=h_in,
            w_in=w_in,
            h_out=h_out,
            w_out=w_out,
            kernel=1,
            pad=0,
            stride=stride,
            patch_bram="BRAM_in_patch_stride2_k1" if stride == 2 else "BRAM_in_patch_stride1_k1",
            weight_bram="BRAM_weight_tile_1",
            out_bram="BRAM_out_tile",
            out_tile_h=tiled_cfg["tile_h"],
            out_tile_w=tiled_cfg["tile_w"],
            bn_dram=None,
            relu_after=False,
        )
        skip_source = feat_skip

    append_tiled_residual_add_ops(
        ops=ops,
        prefix=f"{prefix}_residual",
        tiled_cfg=tiled_cfg,
        feat_main=feat_out,
        feat_skip=skip_source,
        feat_out=feat_out,
        c_out=c_out,
        h_out=h_out,
        w_out=w_out,
    )
    return drams, ops, h_out, w_out


def generate_resnet_tiled_architecture(depth, tiled_cfg=None, conv_cfg=None):
    if tiled_cfg is None:
        tiled_cfg = build_resnet18_tiled_config()
    conv_cfg = normalize_conv_factor_config(conv_cfg)

    spec = get_resnet_spec(depth)
    block_type = spec["block_type"]
    expansion = spec["expansion"]
    tile_oc = tiled_cfg["tile_oc"]
    tile_ic = tiled_cfg["tile_ic"]
    tile_h = tiled_cfg["tile_h"]
    tile_w = tiled_cfg["tile_w"]

    brams = [
        {"name": "BRAM_in_patch_stem", "dims": [tile_ic, tiled_cfg["patch_stem_h"], tiled_cfg["patch_stem_w"]]},
        {"name": "BRAM_in_patch_stride2_k3", "dims": [tile_ic, tiled_cfg["patch_stride2_k3_h"], tiled_cfg["patch_stride2_k3_w"]]},
        {"name": "BRAM_in_patch_stride2_k1", "dims": [tile_ic, tiled_cfg["patch_stride2_k1_h"], tiled_cfg["patch_stride2_k1_w"]]},
        {"name": "BRAM_in_patch_stride1_k3", "dims": [tile_ic, tiled_cfg["patch_stride1_k3_h"], tiled_cfg["patch_stride1_k3_w"]]},
        {"name": "BRAM_in_patch_1", "dims": [tile_ic, 1, 1]},
        {"name": "BRAM_pool_patch", "dims": [tile_oc, tiled_cfg["patch_stride2_k3_h"], tiled_cfg["patch_stride2_k3_w"]]},
        {"name": "BRAM_out_tile", "dims": [tile_oc, tile_h, tile_w]},
        {"name": "BRAM_skip_tile", "dims": [tile_oc, tile_h, tile_w]},
        {"name": "BRAM_weight_tile_7", "dims": [tile_oc, tile_ic, 7, 7]},
        {"name": "BRAM_weight_tile_3", "dims": [tile_oc, tile_ic, 3, 3]},
        {"name": "BRAM_weight_tile_1", "dims": [tile_oc, tile_ic, 1, 1]},
        {"name": "BRAM_bn_tile", "dims": [4, tile_oc]},
        {"name": "BRAM_gap_tile", "dims": [tile_oc, tiled_cfg["gap_tile_h"], tiled_cfg["gap_tile_w"]]},
        {"name": "BRAM_gap_out", "dims": [tile_oc, 1, 1]},
        {"name": "BRAM_fc_out", "dims": [tile_oc, 1, 1]},
    ]
    drams = [
        {"name": "DRAM_input", "dims": [3, 224, 224], "bundle": "mem_input"},
        {"name": "DRAM_w_stem", "dims": [64, 3, 7, 7], "bundle": "mem_w_stem"},
        {"name": "DRAM_bn_stem", "dims": [4, 64], "bundle": "mem_bn_stem"},
        {"name": "DRAM_stem_feat", "dims": [64, 112, 112], "bundle": "mem_stem_feat"},
        {"name": "DRAM_stem_pool", "dims": [64, 56, 56], "bundle": "mem_stem_pool"},
    ]
    ops = []
    append_tiled_conv_ops(
        ops=ops,
        prefix="stem_conv",
        tiled_cfg=tiled_cfg,
        conv_cfg=conv_cfg,
        feat_in="DRAM_input",
        weight_dram="DRAM_w_stem",
        feat_out="DRAM_stem_feat",
        c_in=3,
        c_out=64,
        h_in=224,
        w_in=224,
        h_out=112,
        w_out=112,
        kernel=7,
        pad=3,
        stride=2,
        patch_bram="BRAM_in_patch_stem",
        weight_bram="BRAM_weight_tile_7",
        out_bram="BRAM_out_tile",
        out_tile_h=tile_h,
        out_tile_w=tile_w,
        bn_dram="DRAM_bn_stem",
        relu_after=True,
    )
    append_tiled_maxpool_ops(
        ops=ops,
        prefix="stem_pool",
        tiled_cfg=tiled_cfg,
        feat_in="DRAM_stem_feat",
        feat_out="DRAM_stem_pool",
        c_in=64,
        h_in=112,
        w_in=112,
        h_out=56,
        w_out=56,
    )

    current_feat = "DRAM_stem_pool"
    c_in = 64
    h_in = 56
    w_in = 56
    for stage_id, (stage_width, num_blocks) in enumerate(zip(RESNET_STAGE_WIDTHS, spec["layers"]), start=1):
        stage_stride = 1 if stage_id == 1 else 2
        stage_h = h_in // stage_stride
        stage_w = w_in // stage_stride
        stage_out_c = stage_width * expansion
        if block_type == "basic":
            drams.append({"name": f"DRAM_s{stage_id}_mid1", "dims": [stage_out_c, stage_h, stage_w], "bundle": f"mem_s{stage_id}_mid1"})
        else:
            drams.append({"name": f"DRAM_s{stage_id}_mid1", "dims": [stage_width, stage_h, stage_w], "bundle": f"mem_s{stage_id}_mid1"})
            drams.append({"name": f"DRAM_s{stage_id}_mid2", "dims": [stage_width, stage_h, stage_w], "bundle": f"mem_s{stage_id}_mid2"})
        if stage_id > 1 or block_type == "bottleneck":
            drams.append({"name": f"DRAM_s{stage_id}_skip", "dims": [stage_out_c, stage_h, stage_w], "bundle": f"mem_s{stage_id}_skip"})
        for block_id in range(num_blocks):
            out_name = f"DRAM_s{stage_id}_b{block_id}"
            drams.append({"name": out_name, "dims": [stage_out_c, stage_h, stage_w], "bundle": f"mem_s{stage_id}_b{block_id}"})
            prefix = f"s{stage_id}_b{block_id}"
            block_stride = stage_stride if block_id == 0 else 1
            projection = block_id == 0 and (block_stride != 1 or c_in != stage_out_c)
            if block_type == "basic":
                block_drams, block_ops, h_out, w_out = generate_resnet_tiled_basic_block(
                    prefix=prefix,
                    tiled_cfg=tiled_cfg,
                    conv_cfg=conv_cfg,
                    c_in=c_in,
                    c_out=stage_out_c,
                    h_in=h_in,
                    w_in=w_in,
                    stride=block_stride,
                    projection=projection,
                    feat_in=current_feat,
                    feat_mid=f"DRAM_s{stage_id}_mid1",
                    feat_out=out_name,
                    feat_skip=f"DRAM_s{stage_id}_skip" if projection else None,
                )
            else:
                block_drams, block_ops, h_out, w_out = generate_resnet_tiled_bottleneck_block(
                    prefix=prefix,
                    tiled_cfg=tiled_cfg,
                    conv_cfg=conv_cfg,
                    c_in=c_in,
                    bottleneck_c=stage_width,
                    h_in=h_in,
                    w_in=w_in,
                    stride=block_stride,
                    projection=projection,
                    feat_in=current_feat,
                    feat_mid1=f"DRAM_s{stage_id}_mid1",
                    feat_mid2=f"DRAM_s{stage_id}_mid2",
                    feat_out=out_name,
                    feat_skip=f"DRAM_s{stage_id}_skip" if projection else None,
                )
            drams.extend(block_drams)
            ops.extend(block_ops)
            current_feat = out_name
            c_in = stage_out_c
            h_in = h_out
            w_in = w_out

    drams.extend([
        {"name": "DRAM_gap", "dims": [c_in, 1, 1], "bundle": "mem_gap"},
        {"name": "DRAM_fc", "dims": [1000, c_in, 1, 1], "bundle": "mem_fc"},
        {"name": "DRAM_out", "dims": [1000, 1, 1], "bundle": "mem_out"},
    ])
    append_tiled_gap_ops(
        ops=ops,
        prefix="global_pool",
        tiled_cfg=tiled_cfg,
        feat_in=current_feat,
        feat_out="DRAM_gap",
        c_in=c_in,
        h_in=h_in,
        w_in=w_in,
    )
    append_tiled_conv_ops(
        ops=ops,
        prefix="fc",
        tiled_cfg=tiled_cfg,
        conv_cfg=conv_cfg,
        feat_in="DRAM_gap",
        weight_dram="DRAM_fc",
        feat_out="DRAM_out",
        c_in=c_in,
        c_out=1000,
        h_in=1,
        w_in=1,
        h_out=1,
        w_out=1,
        kernel=1,
        pad=0,
        stride=1,
        patch_bram="BRAM_in_patch_1",
        weight_bram="BRAM_weight_tile_1",
        out_bram="BRAM_fc_out",
        out_tile_h=1,
        out_tile_w=1,
        bn_dram=None,
        relu_after=False,
    )
    return brams, drams, ops


def generate_resnet_tiled_config_txt(depth, data_type="ap_fixed<16,5>", tiled_cfg=None, conv_cfg=None):
    if tiled_cfg is None:
        tiled_cfg = build_resnet18_tiled_config()
    brams, drams, ops = generate_resnet_tiled_architecture(depth, tiled_cfg, conv_cfg=conv_cfg)
    return serialize_scale_model_config(
        brams=brams,
        drams=drams,
        ops=ops,
        output_dram_names=["DRAM_out"],
        data_type=data_type,
    )


def generate_resnet18_architecture():
    return generate_resnet_architecture(18)


def generate_resnet18_config_txt(data_type="ap_fixed<16,5>", conv_cfg=None):
    return generate_resnet_config_txt(18, data_type, conv_cfg=conv_cfg)


def get_resnet18_tiled_filename(
    data_type,
    tiled_cfg,
    conv_cfg=None,
    preserve_legacy_conv_name=False,
    preserve_legacy_tiled_name=False,
):
    return get_resnet_tiled_filename(
        18,
        data_type,
        tiled_cfg,
        conv_cfg=conv_cfg,
        preserve_legacy_conv_name=preserve_legacy_conv_name,
        preserve_legacy_tiled_name=preserve_legacy_tiled_name,
    )


def generate_resnet18_tiled_architecture(conv_cfg=None):
    return generate_resnet_tiled_architecture(18, conv_cfg=conv_cfg)


def generate_resnet18_tiled_architecture_with_config(tiled_cfg, conv_cfg=None):
    return generate_resnet_tiled_architecture(18, tiled_cfg, conv_cfg=conv_cfg)


def generate_resnet18_tiled_config_txt(data_type="ap_fixed<16,5>", tiled_cfg=None, conv_cfg=None):
    return generate_resnet_tiled_config_txt(18, data_type, tiled_cfg, conv_cfg=conv_cfg)

def generate_mobilenetv2_block(
    block_id,
    C_IN,
    C_OUT,
    H_IN,
    W_IN,
    stride,
    expansion,
    bram_ping,
    bram_pong,
    conv_cfg=None,
):
    """
    Generates BRAMs, DRAMs, and ops for a single MobileNet-V2 inverted residual block.
    Uses CHW layout.
    """

    brams = []
    drams = []
    ops = []

    conv_cfg = normalize_conv_factor_config(conv_cfg)
    EXP_C = C_IN * expansion
    H_OUT = H_IN // stride
    W_OUT = W_IN // stride

    # H_OUT = H_IN
    # W_OUT = W_IN

    feat_in = bram_ping
    feat_mid = bram_pong
    feat_out = bram_ping  # ping-pong swap back

    # -------------------------
    # Expansion 1x1 conv
    # -------------------------
    if expansion != 1:
        brams.extend([
            {"name": f"BRAM_w_exp_{block_id}", "dims": [EXP_C, C_IN, 1, 1]},
            {"name": f"BRAM_bn_exp_{block_id}", "dims": [4, EXP_C]},
        ])

        drams.extend([
            {"name": f"DRAM_w_exp_{block_id}", "dims": [EXP_C, C_IN, 1, 1],
             "bundle": f"mem_w_exp_{block_id}"},
            {"name": f"DRAM_bn_exp_{block_id}", "dims": [4, EXP_C],
             "bundle": f"mem_bn_exp_{block_id}"},
        ])

        ops.extend([
            ("exp_conv", {
                "func_name": "conv",
                "dims": build_conv_dims(C_IN, EXP_C, H_IN, W_IN, H_IN, W_IN, 1, 0, 1, conv_cfg=conv_cfg),
                "func_info": ["conv_template.cpp", "conv2d", False],
                "args": [feat_in, f"BRAM_w_exp_{block_id}", feat_mid]
            }),
            ("exp_bn", {
                "func_name": "batchnorm",
                "dims": [EXP_C, H_IN, W_IN],
                "func_info": ["batch_norm_template.cpp", False],
                "args": [feat_mid, f"BRAM_bn_exp_{block_id}", feat_mid]
            }),
            ("exp_relu", {
                "func_name": "activation_conv",
                "dims": [EXP_C, H_IN, W_IN],
                "func_info": ["activation_template_conv.cpp", "relu6"],
                "args": [feat_mid, feat_mid, 6]
            }),
        ])
    else:
        feat_mid = feat_in
        EXP_C = C_IN

    # -------------------------
    # Depthwise 3x3 conv
    # -------------------------
    brams.extend([
        {"name": f"BRAM_w_dw_{block_id}", "dims": [EXP_C, EXP_C, 3, 3]},
        {"name": f"BRAM_bn_dw_{block_id}", "dims": [4, EXP_C]},
    ])

    drams.extend([
        {"name": f"DRAM_w_dw_{block_id}", "dims": [EXP_C, EXP_C, 3, 3],
         "bundle": f"mem_w_dw_{block_id}"},
        {"name": f"DRAM_bn_dw_{block_id}", "dims": [4, EXP_C],
         "bundle": f"mem_bn_dw_{block_id}"},
    ])

    ops.extend([
        ("dw_conv", {
            "func_name": "conv",
            "dims": build_conv_dims(EXP_C, EXP_C, H_IN, W_IN, H_OUT, W_OUT, 3, 1, stride, conv_cfg=conv_cfg),
            "func_info": ["conv_template.cpp", "group_conv2d", False],
            "args": [feat_mid, f"BRAM_w_dw_{block_id}", feat_mid, EXP_C]
        }),
        ("dw_bn", {
            "func_name": "batchnorm",
            "dims": [EXP_C, H_OUT, W_OUT],
            "func_info": ["batch_norm_template.cpp", False],
            "args": [feat_mid, f"BRAM_bn_dw_{block_id}", feat_mid]
        }),
        ("dw_relu", {
            "func_name": "activation_conv",
            "dims": [EXP_C, H_OUT, W_OUT],
            "func_info": ["activation_template_conv.cpp", "relu6"],
            "args": [feat_mid, feat_mid, 6]
        }),
    ])

    # -------------------------
    # Projection 1x1 conv
    # -------------------------
    brams.extend([
        {"name": f"BRAM_w_proj_{block_id}", "dims": [C_OUT, EXP_C, 1, 1]},
        {"name": f"BRAM_bn_proj_{block_id}", "dims": [4, C_OUT]},
    ])

    drams.extend([
        {"name": f"DRAM_w_proj_{block_id}", "dims": [C_OUT, EXP_C, 1, 1],
         "bundle": f"mem_w_proj_{block_id}"},
        {"name": f"DRAM_bn_proj_{block_id}", "dims": [4, C_OUT],
         "bundle": f"mem_bn_proj_{block_id}"},
    ])

    ops.extend([
        ("proj_conv", {
            "func_name": "conv",
            "dims": build_conv_dims(EXP_C, C_OUT, H_OUT, W_OUT, H_OUT, W_OUT, 1, 0, 1, conv_cfg=conv_cfg),
            "func_info": ["conv_template.cpp", "conv2d", False],
            "args": [feat_mid, f"BRAM_w_proj_{block_id}", feat_out]
        }),
        ("proj_bn", {
            "func_name": "batchnorm",
            "dims": [C_OUT, H_OUT, W_OUT],
            "func_info": ["batch_norm_template.cpp", False],
            "args": [feat_out, f"BRAM_bn_proj_{block_id}", feat_out]
        }),
    ])

    # -------------------------
    # Residual
    # -------------------------
    if stride == 1 and C_IN == C_OUT:
        ops.append((
            "residual_add", {
                "func_name": "matrix_add_conv",
                "dims": [C_OUT, H_OUT, W_OUT],
                "func_info": ["matrix_add_template_conv.cpp"],
                "args": [feat_out, feat_in, feat_out]
            }
        ))

    return brams, drams, ops, C_OUT, H_OUT, W_OUT

def generate_mobilenetv2_architecture(conv_cfg=None):
    conv_cfg = normalize_conv_factor_config(conv_cfg)
    brams = []
    drams = []
    ops = []

    # -------------------------
    # Input
    # -------------------------
    brams.extend([
        {"name": "BRAM_feat_0", "dims": [3, 224, 224]},
        {"name": "BRAM_feat_1", "dims": [3, 112, 112]},
        {"name": "BRAM_feat_2", "dims": [3, 112, 112]},
    ])

    drams.append({
        "name": "DRAM_input",
        "dims": [3, 224, 224],
        "bundle": "mem_input"
    })

    ops.append((
        "load_input", {
            "func_name": "load",
            "dims": [3, 224, 224],
            "args": ["DRAM_input", "BRAM_feat_0"]
        }
    ))

    # -------------------------
    # Stem
    # -------------------------
    brams.extend([
        {"name": "BRAM_w_stem", "dims": [32, 3, 3, 3]},
        {"name": "BRAM_bn_stem", "dims": [4, 32]},
    ])

    drams.extend([
        {"name": "DRAM_w_stem", "dims": [32, 3, 3, 3], "bundle": "mem_w_stem"},
        {"name": "DRAM_bn_stem", "dims": [4, 32], "bundle": "mem_bn_stem"},
    ])

    ops.extend([
        ("stem_conv", {
            "func_name": "conv",
            "dims": build_conv_dims(3, 32, 224, 224, 112, 112, 3, 1, 2, conv_cfg=conv_cfg),
            "func_info": ["conv_template.cpp", "conv2d", False],
            "args": ["BRAM_feat_0", "BRAM_w_stem", "BRAM_feat_1"]
        }),
        ("stem_bn", {
            "func_name": "batchnorm",
            "dims": [32, 112, 112],
            "func_info": ["batch_norm_template.cpp", False],
            "args": ["BRAM_feat_1", "BRAM_bn_stem", "BRAM_feat_1"]
        }),
        ("stem_relu", {
            "func_name": "activation_conv",
            "dims": [32, 112, 112],
            "func_info": ["activation_template_conv.cpp", "relu6"],
            "args": ["BRAM_feat_1", "BRAM_feat_1", 6]
        }),
    ])

    # -------------------------
    # Inverted residual blocks
    # -------------------------
    cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),
        (6, 32, 3, 1),
        (6, 64, 4, 1),
        (6, 96, 3, 1),
        (6, 160, 3, 1),
        (6, 320, 1, 1),
    ]

    C, H, W = 32, 112, 112
    block_id = 0
    ping, pong = "BRAM_feat_1", "BRAM_feat_2"

    for t, c, n, s in cfg:
        for i in range(n):
            stride = s if i == 0 else 1
            b_bram, b_dram, b_ops, C, H, W = generate_mobilenetv2_block(
                block_id, C, c, H, W, stride, t, ping, pong, conv_cfg=conv_cfg
            )
            brams.extend(b_bram)
            drams.extend(b_dram)
            ops.extend(b_ops)
            ping, pong = pong, ping
            block_id += 1

    # -------------------------
    # Final conv
    # -------------------------
    
    brams.extend([
        {"name": "BRAM_w_final", "dims": [1280, C, 1, 1]},
        {"name": "BRAM_bn_final", "dims": [4, 1280]},
    ])

    drams.extend([
        {"name": "DRAM_w_final", "dims": [1280, C, 1, 1], "bundle": "mem_w_final"},
        {"name": "DRAM_bn_final", "dims": [4, 1280], "bundle": "mem_bn_final"},
    ])

    ops.extend([
        ("final_conv", {
            "func_name": "conv",
            "dims": build_conv_dims(C, 1280, H, W, H, W, 1, 0, 1, conv_cfg=conv_cfg),
            "func_info": ["conv_template.cpp", "conv2d", False],
            "args": [ping, "BRAM_w_final", pong]
        }),
        ("final_bn", {
            "func_name": "batchnorm",
            "dims": [1280, H, W],
            "func_info": ["batch_norm_template.cpp", False],
            "args": [pong, "BRAM_bn_final", pong]
        }),
        ("final_relu", {
            "func_name": "activation_conv",
            "dims": [1280, H, W],
            "func_info": ["activation_template_conv.cpp", "relu6"],
            "args": [pong, pong, 6]
        }),
    ])


    brams.extend([
        {"name": "BRAM_pool", "dims": [1280, 1, 1]},
        {"name": "BRAM_out", "dims": [1000, 1, 1]}
    ])

    drams.append({
        "name": "DRAM_out",
        "dims": [1000, 1, 1],
        "bundle": "mem_out"
    })
    # -------------------------
    # Global pool + classifier
    # -------------------------
    ops.append((
        "global_pool", {
            "func_name": "adaptive_avgpool",
            "dims": [1280, H, W, 1, 1],
            "args": [pong, "BRAM_pool"],
            "func_info": ["adaptive_avgpool_template.cpp"],
        }
    ))

    brams.append({"name": "BRAM_fc", "dims": [1000, 1280, 1, 1]})
    drams.append({"name": "DRAM_fc", "dims": [1000, 1280, 1, 1], "bundle": "mem_fc"})

    ops.append((
        "fc", {
            "func_name": "conv",
            "dims": build_conv_dims(1280, 1000, 1, 1, 1, 1, 1, 0, 1, conv_cfg=conv_cfg),
            "func_info": ["conv_template.cpp", "conv2d", False],
            "args": ["BRAM_pool", "BRAM_fc", "BRAM_out"]
        }
    ))

    # Store output
    ops.append((
        "store_output", {
            "func_name": "store",
            "dims": [1000, 1, 1],
            "args": ["BRAM_out", "DRAM_out"]
        }
    ))

    return brams, drams, ops

def generate_mobilenetv2_config_txt(data_type="ap_fixed<16,5>", conv_cfg=None):
    brams, drams, ops = generate_mobilenetv2_architecture(conv_cfg=conv_cfg)
    return serialize_scale_model_config(
        brams=brams,
        drams=drams,
        ops=ops,
        output_dram_names=["DRAM_out"],
        data_type=data_type,
    )



def parse_args():
    parser = argparse.ArgumentParser(description="Generate scale model JSON configs.")
    parser.add_argument("--resnet-depths", type=str, default="18,34,50,101,152")
    parser.add_argument("--llama3-8b-contexts", type=str, default="2048,8192")
    parser.add_argument("--llama3-8b-token-tile-prefill", type=str, default="16")
    parser.add_argument("--llama3-8b-token-tile-decode", type=str, default="1")
    parser.add_argument("--llama3-8b-hidden-chunk", type=str, default="128")
    parser.add_argument("--llama3-8b-in-chunk", type=str, default="128")
    parser.add_argument("--llama3-8b-max-out-chunk", type=str, default="256")
    parser.add_argument("--llama3-8b-ffn-chunk", type=str, default="128")
    parser.add_argument("--llama3-8b-k-token-tile", type=str, default="128")
    parser.add_argument("--llama3-8b-q-head-tile", type=str, default="4")
    parser.add_argument("--llama3-8b-linear-in-factors", type=str, default="1")
    parser.add_argument("--llama3-8b-linear-out-factors", type=str, default="1")
    parser.add_argument("--llama3-8b-chunk-col-factors", type=str, default="1")
    parser.add_argument("--llama3-8b-attn-head-factors", type=str, default="1")
    parser.add_argument("--llama3-8b-attn-dim-factors", type=str, default="1")
    parser.add_argument("--conv-ci-factors", type=str, default=",".join(str(x) for x in DEFAULT_GLOBAL_CONV_CI_FACTORS))
    parser.add_argument("--conv-co-factors", type=str, default=",".join(str(x) for x in DEFAULT_GLOBAL_CONV_CO_FACTORS))
    parser.add_argument("--resnet-tiled-oc", "--resnet18-tiled-oc", dest="resnet_tiled_oc", type=str, default=str(DEFAULT_RESNET18_TILE_OC))
    parser.add_argument("--resnet-tiled-ic", "--resnet18-tiled-ic", dest="resnet_tiled_ic", type=str, default=str(DEFAULT_RESNET18_TILE_IC))
    parser.add_argument("--resnet-tiled-h", "--resnet18-tiled-h", dest="resnet_tiled_h", type=str, default=str(DEFAULT_RESNET18_TILE_H))
    parser.add_argument("--resnet-tiled-w", "--resnet18-tiled-w", dest="resnet_tiled_w", type=str, default=str(DEFAULT_RESNET18_TILE_W))
    return parser.parse_args()


def main():
    args = parse_args()
    selected_resnet_depths = parse_resnet_depths(args.resnet_depths)
    llama3_8b_contexts = parse_context_list(args.llama3_8b_contexts)
    llama3_8b_token_tile_prefills = parse_positive_int_list(
        args.llama3_8b_token_tile_prefill,
        "--llama3-8b-token-tile-prefill",
    )
    llama3_8b_token_tile_decodes = parse_positive_int_list(
        args.llama3_8b_token_tile_decode,
        "--llama3-8b-token-tile-decode",
    )
    llama3_8b_hidden_chunks = parse_positive_int_list(
        args.llama3_8b_hidden_chunk,
        "--llama3-8b-hidden-chunk",
    )
    llama3_8b_in_chunks = parse_positive_int_list(
        args.llama3_8b_in_chunk,
        "--llama3-8b-in-chunk",
    )
    llama3_8b_max_out_chunks = parse_positive_int_list(
        args.llama3_8b_max_out_chunk,
        "--llama3-8b-max-out-chunk",
    )
    llama3_8b_ffn_chunks = parse_positive_int_list(
        args.llama3_8b_ffn_chunk,
        "--llama3-8b-ffn-chunk",
    )
    llama3_8b_k_token_tiles = parse_positive_int_list(
        args.llama3_8b_k_token_tile,
        "--llama3-8b-k-token-tile",
    )
    llama3_8b_q_head_tiles = parse_positive_int_list(
        args.llama3_8b_q_head_tile,
        "--llama3-8b-q-head-tile",
    )
    llama3_8b_linear_in_factors = parse_positive_int_list(
        args.llama3_8b_linear_in_factors,
        "--llama3-8b-linear-in-factors",
    )
    llama3_8b_linear_out_factors = parse_positive_int_list(
        args.llama3_8b_linear_out_factors,
        "--llama3-8b-linear-out-factors",
    )
    llama3_8b_chunk_col_factors = parse_positive_int_list(
        args.llama3_8b_chunk_col_factors,
        "--llama3-8b-chunk-col-factors",
    )
    llama3_8b_attn_head_factors = parse_positive_int_list(
        args.llama3_8b_attn_head_factors,
        "--llama3-8b-attn-head-factors",
    )
    llama3_8b_attn_dim_factors = parse_positive_int_list(
        args.llama3_8b_attn_dim_factors,
        "--llama3-8b-attn-dim-factors",
    )

    llama3_8b_tile_cfgs = []
    skipped_llama3_8b_tile_cfgs = []
    for (
        token_tile_prefill,
        token_tile_decode,
        hidden_chunk,
        in_chunk,
        max_out_chunk,
        ffn_chunk,
        k_token_tile,
        q_head_tile,
    ) in itertools.product(
        llama3_8b_token_tile_prefills,
        llama3_8b_token_tile_decodes,
        llama3_8b_hidden_chunks,
        llama3_8b_in_chunks,
        llama3_8b_max_out_chunks,
        llama3_8b_ffn_chunks,
        llama3_8b_k_token_tiles,
        llama3_8b_q_head_tiles,
    ):
        tile_cfg_candidate = {
            "token_tile_prefill": token_tile_prefill,
            "token_tile_decode": token_tile_decode,
            "hidden_chunk": hidden_chunk,
            "in_chunk": in_chunk,
            "max_out_chunk": max_out_chunk,
            "ffn_chunk": ffn_chunk,
            "k_token_tile": k_token_tile,
            "q_head_tile": q_head_tile,
        }
        try:
            llama3_8b_tile_cfgs.append(build_llama3_8b_tile_config(tile_cfg_candidate))
        except ValueError as exc:
            skipped_llama3_8b_tile_cfgs.append((tile_cfg_candidate, str(exc)))

    if not llama3_8b_tile_cfgs:
        raise ValueError(
            "No valid Llama 3 8B tile configurations were generated from the provided sweep values."
        )

    llama3_8b_pragma_cfgs = []
    skipped_llama3_8b_pragma_cfgs = []
    for (
        linear_in_factor,
        linear_out_factor,
        chunk_col_factor,
        attn_head_factor,
        attn_dim_factor,
    ) in itertools.product(
        llama3_8b_linear_in_factors,
        llama3_8b_linear_out_factors,
        llama3_8b_chunk_col_factors,
        llama3_8b_attn_head_factors,
        llama3_8b_attn_dim_factors,
    ):
        pragma_cfg_candidate = {
            "linear_in_factor": linear_in_factor,
            "linear_out_factor": linear_out_factor,
            "chunk_col_factor": chunk_col_factor,
            "attn_head_factor": attn_head_factor,
            "attn_dim_factor": attn_dim_factor,
        }
        try:
            llama3_8b_pragma_cfgs.append(build_llama3_8b_pragma_config(pragma_cfg_candidate))
        except ValueError as exc:
            skipped_llama3_8b_pragma_cfgs.append((pragma_cfg_candidate, str(exc)))

    if not llama3_8b_pragma_cfgs:
        raise ValueError(
            "No valid Llama 3 8B pragma configurations were generated from the provided sweep values."
        )

    preserve_legacy_llama_names = (
        len(llama3_8b_tile_cfgs) == 1
        and len(llama3_8b_pragma_cfgs) == 1
        and llama3_8b_tile_cfgs[0]["is_default"]
        and llama3_8b_pragma_cfgs[0]["is_default"]
    )
    conv_ci_factors = parse_positive_int_list(args.conv_ci_factors, "--conv-ci-factors")
    conv_co_factors = parse_positive_int_list(args.conv_co_factors, "--conv-co-factors")
    conv_factor_cfgs = [
        build_conv_factor_config(ci_factor, co_factor)
        for ci_factor, co_factor in itertools.product(conv_ci_factors, conv_co_factors)
    ]
    preserve_legacy_conv_names = (
        len(conv_factor_cfgs) == 1
        and is_default_conv_factor_config(conv_factor_cfgs[0])
    )
    resnet_tiled_ocs = parse_positive_int_list(args.resnet_tiled_oc, "--resnet-tiled-oc")
    resnet_tiled_ics = parse_positive_int_list(args.resnet_tiled_ic, "--resnet-tiled-ic")
    resnet_tiled_hs = parse_positive_int_list(args.resnet_tiled_h, "--resnet-tiled-h")
    resnet_tiled_ws = parse_positive_int_list(args.resnet_tiled_w, "--resnet-tiled-w")
    tiled_cfgs = [
        build_resnet18_tiled_config(tile_oc=tile_oc, tile_ic=tile_ic, tile_h=tile_h, tile_w=tile_w)
        for tile_oc, tile_ic, tile_h, tile_w in itertools.product(
            resnet_tiled_ocs,
            resnet_tiled_ics,
            resnet_tiled_hs,
            resnet_tiled_ws,
        )
    ]
    preserve_legacy_resnet_tiled_names = (
        len(tiled_cfgs) == 1
        and tiled_cfgs[0]["is_default"]
    )

    # Define parameter ranges (adjust as needed)

    # Static parameters
    seq_len = [2048]
    data_type_list = ["ap_fixed<16,5>",]
    # seed_list = [47]    

    combinations = itertools.product(
        seq_len, data_type_list, 
    )

    output_dir = "auto_generated_configs"
    os.makedirs(output_dir, exist_ok=True)

    # Now iterate over the base combos, conv_type, (groups if needed), and data_type.
    for (seq, data_type) in combinations:
        
        llama_config_text = generate_llama_config_text(
            seq_len=seq, 
            data_type=data_type
        )

        longformer_config_text = generate_longformer_config_txt(
            seq_len=seq, 
            data_type=data_type
        )

        vit_config_text = generate_vit_config_txt(
            data_type=data_type
        )

        naming_dtype = data_type.replace('<','_').replace('>','_').replace(',','_')
        
        llama_filename = (
            f"LLAMA3B_config_{seq}_{naming_dtype}.json"
        )
        llama_filepath = os.path.join(output_dir, llama_filename)
        with open(llama_filepath, "w") as f:
            f.write(llama_config_text)


        longformer_filename = (
            f"LONGFORMER_config_{seq}_{naming_dtype}.json"
        )
        longformer_filepath = os.path.join(output_dir, longformer_filename)
        with open(longformer_filepath, "w") as f:
            f.write(longformer_config_text)
        
        conv_filepaths = []
        for conv_cfg in conv_factor_cfgs:
            mobilenetv2_config_text = generate_mobilenetv2_config_txt(
                data_type=data_type,
                conv_cfg=conv_cfg,
            )
            mobilenetv2_filename = get_mobilenetv2_filename(
                data_type,
                conv_cfg=conv_cfg,
                preserve_legacy_name=preserve_legacy_conv_names,
            )
            mobilenetv2_filepath = os.path.join(output_dir, mobilenetv2_filename)
            with open(mobilenetv2_filepath, "w") as f:
                f.write(mobilenetv2_config_text)
            conv_filepaths.append(mobilenetv2_filepath)

            for depth in selected_resnet_depths:
                resnet_config_text = generate_resnet_config_txt(
                    depth,
                    data_type=data_type,
                    conv_cfg=conv_cfg,
                )
                resnet_filename = get_resnet_filename(
                    depth,
                    data_type,
                    conv_cfg=conv_cfg,
                    preserve_legacy_name=preserve_legacy_conv_names,
                )
                resnet_filepath = os.path.join(output_dir, resnet_filename)
                with open(resnet_filepath, "w") as f:
                    f.write(resnet_config_text)
                conv_filepaths.append(resnet_filepath)

                for tiled_cfg in tiled_cfgs:
                    resnet_tiled_config_text = generate_resnet_tiled_config_txt(
                        depth,
                        data_type=data_type,
                        tiled_cfg=tiled_cfg,
                        conv_cfg=conv_cfg,
                    )
                    resnet_tiled_filename = get_resnet_tiled_filename(
                        depth,
                        data_type,
                        tiled_cfg,
                        conv_cfg=conv_cfg,
                        preserve_legacy_conv_name=preserve_legacy_conv_names,
                        preserve_legacy_tiled_name=preserve_legacy_resnet_tiled_names,
                    )
                    resnet_tiled_filepath = os.path.join(output_dir, resnet_tiled_filename)
                    with open(resnet_tiled_filepath, "w") as f:
                        f.write(resnet_tiled_config_text)
                    conv_filepaths.append(resnet_tiled_filepath)

        llama3_8b_filepaths = []
        for llama3_8b_tile_cfg in llama3_8b_tile_cfgs:
            for llama3_8b_pragma_cfg in llama3_8b_pragma_cfgs:
                for ctx in llama3_8b_contexts:
                    prefill_config_text = generate_llama3_8b_prefill_config_text(
                        ctx,
                        data_type=data_type,
                        tile_cfg=llama3_8b_tile_cfg,
                        pragma_cfg=llama3_8b_pragma_cfg,
                    )
                    prefill_filename = get_llama3_8b_filename(
                        "prefill",
                        ctx,
                        data_type,
                        llama3_8b_tile_cfg,
                        pragma_cfg=llama3_8b_pragma_cfg,
                        preserve_legacy_name=preserve_legacy_llama_names,
                    )
                    prefill_filepath = os.path.join(output_dir, prefill_filename)
                    with open(prefill_filepath, "w") as f:
                        f.write(prefill_config_text)
                    llama3_8b_filepaths.append(prefill_filepath)

                    decode_config_text = generate_llama3_8b_decode_config_text(
                        ctx,
                        data_type=data_type,
                        tile_cfg=llama3_8b_tile_cfg,
                        pragma_cfg=llama3_8b_pragma_cfg,
                    )
                    decode_filename = get_llama3_8b_filename(
                        "decode",
                        ctx,
                        data_type,
                        llama3_8b_tile_cfg,
                        pragma_cfg=llama3_8b_pragma_cfg,
                        preserve_legacy_name=preserve_legacy_llama_names,
                    )
                    decode_filepath = os.path.join(output_dir, decode_filename)
                    with open(decode_filepath, "w") as f:
                        f.write(decode_config_text)
                    llama3_8b_filepaths.append(decode_filepath)

        vit_filename = (
            f"VIT_config_{naming_dtype}.json"
        )
        vit_filepath = os.path.join(output_dir, vit_filename)
        with open(vit_filepath, "w") as f:
            f.write(vit_config_text)
        
        print(f"Generated {llama_filepath}")
        print(f"Generated {longformer_filepath}")
        for conv_filepath in conv_filepaths:
            print(f"Generated {conv_filepath}")
        for llama3_8b_filepath in llama3_8b_filepaths:
            print(f"Generated {llama3_8b_filepath}")
        for skipped_tile_cfg, skip_reason in skipped_llama3_8b_tile_cfgs:
            print(f"Skipped Llama3 8B tile config {skipped_tile_cfg}: {skip_reason}")
        for skipped_pragma_cfg, skip_reason in skipped_llama3_8b_pragma_cfgs:
            print(f"Skipped Llama3 8B pragma config {skipped_pragma_cfg}: {skip_reason}")
        print(f"Generated {vit_filepath}")
        print(
            "Llama3 8B tile sweep summary: "
            f"candidates={len(llama3_8b_tile_cfgs) + len(skipped_llama3_8b_tile_cfgs)}, "
            f"valid={len(llama3_8b_tile_cfgs)}, "
            f"skipped={len(skipped_llama3_8b_tile_cfgs)}, "
            f"generated_jsons={len(llama3_8b_filepaths)}"
        )
        print(
            "Llama3 8B pragma sweep summary: "
            f"candidates={len(llama3_8b_pragma_cfgs) + len(skipped_llama3_8b_pragma_cfgs)}, "
            f"valid={len(llama3_8b_pragma_cfgs)}, "
            f"skipped={len(skipped_llama3_8b_pragma_cfgs)}"
        )

    num_combos = len(list(itertools.product(
        seq_len, data_type_list,
    )))
    print("Total number of combos:", num_combos)

if __name__ == "__main__":
    main()
