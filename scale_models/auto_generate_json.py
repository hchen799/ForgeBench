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

def generate_llama_decode_block (
    mod_name, 
    bram_in,

    tmp_bram_1,
    tmp_bram_2,
    tmp_bram_3,

    bram_rms_norm_weights_1,

    bram_weights_q,
    bram_weights_k,
    bram_weights_v,

    bram_rms_norm_weights_2,

    bram_mm_weights_1,
    bram_mm_weights_2,
    bram_mm_weights_3,

    bram_out,
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
        print(x)
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
        print(x)
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
        print(x)
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
):
    """
    Generates BRAMs, DRAMs, and ops for a single MobileNet-V2 inverted residual block.
    Uses CHW layout.
    """

    brams = []
    drams = []
    ops = []

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
                "dims": [C_IN, EXP_C, H_IN, W_IN, H_IN, W_IN, 1, 0, 1,
                         0, 8, 64, 8, 64, 64, 8, 64],
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
            "dims": [EXP_C, EXP_C, H_IN, W_IN, H_OUT, W_OUT, 3, 1, stride,
                     0, 8, 64, 8, 64, 64, 8, 64],
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
            "dims": [EXP_C, C_OUT, H_OUT, W_OUT, H_OUT, W_OUT, 1, 0, 1,
                     0, 8, 64, 8, 64, 64, 8, 64],
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

def generate_mobilenetv2_architecture():
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
            "dims": [3, 32, 224, 224, 112, 112, 3, 1, 2,
                     0, 8, 64, 8, 64, 64, 8, 64],
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
                block_id, C, c, H, W, stride, t, ping, pong
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
            "dims": [C, 1280, H, W, H, W, 1, 0, 1,
                     0, 8, 64, 8, 64, 64, 8, 64],
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
            "dims": [1280, 1000, 1, 1, 1, 1, 1, 0, 1,
                     0, 8, 64, 8, 64, 64, 8, 64],
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

def generate_mobilenetv2_config_txt(data_type="ap_fixed<16,5>"):
    """
    Returns a string of the JSON config of the Longformer with the given parameters.
    """
    
    brams, drams, ops = generate_mobilenetv2_architecture()
    
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
        print(x)
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



def main():
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

        mobilenetv2_config_text = generate_mobilenetv2_config_txt(
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
        
        mobilenetv2_filename = (
            f"MOBILENETV2_config_{naming_dtype}.json"
        )
        mobilenetv2_filepath = os.path.join(output_dir, mobilenetv2_filename)
        with open(mobilenetv2_filepath, "w") as f:
            f.write(mobilenetv2_config_text)

        vit_filename = (
            f"VIT_config_{naming_dtype}.json"
        )
        vit_filepath = os.path.join(output_dir, vit_filename)
        with open(vit_filepath, "w") as f:
            f.write(vit_config_text)
        
        print(f"Generated {llama_filepath}")
        print(f"Generated {longformer_filepath}")
        print(f"Generated {mobilenetv2_filepath}")
        print(f"Generated {vit_filepath}")

    num_combos = len(list(itertools.product(
        seq_len, data_type_list,
    )))
    print("Total number of combos:", num_combos)

if __name__ == "__main__":
    main()
