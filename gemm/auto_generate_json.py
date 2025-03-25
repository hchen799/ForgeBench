
import itertools
import os

# Computing xABy

# option 1
# ((xA)B)y

# option 2
# (x(AB))y

# option 3
# x((AB)y)

# option 4
# x(A(By))

# option 5
# (xA)(By)

def generate_config_text(
    M, K, N,
    unroll_M, unroll_K, unroll_N,
    order, 
    DATA_TYPE, intermediate_bias, inline, computation_order = "option_1",
):
    """
    Returns a string of the JSON config in exactly the format requested,
    computing H_OUT and W_OUT automatically and using conv_type (and groups if needed)
    in the conv_1 op.
    """

    if computation_order not in ["option_1", "option_2", "option_3", "option_4", "option_5"]:
        raise ValueError(f"Invalid computation_order_option: {computation_order}")
    
    order_vm = [f"{x}" for x in order if x in ['i', 'j']]
    order_gmm = [f"{x}"  for x in order if x in ['i', 'k', 'j']]

    # 1) Build the lines for brams
    brams = [
        {"name": "BRAM_x",        "dims": [M]},
        {"name": "BRAM_A",        "dims": [M, K]},
        {"name": "BRAM_B",        "dims": [K, N]},
        {"name": "BRAM_y",        "dims": [N]},
        {"name": "BRAM_bias",        "dims": [1]},
        {"name": "BRAM_result",        "dims": [1]},
    ]


    if computation_order == "option_1":
        brams.extend([
            {"name": "BRAM_xA_bias",        "dims": [K]},
            {"name": "BRAM_xA",        "dims": [K]},
            {"name": "BRAM_xAB_bias",        "dims": [N]},
            {"name": "BRAM_xAB",        "dims": [N]},
        ])
    elif computation_order == "option_2":
        brams.extend([
            {"name": "BRAM_gemm_bias",        "dims": [M, N]},
            {"name": "BRAM_gemm",        "dims": [M, N]},
            {"name": "BRAM_vmm_bias",        "dims": [N]},
            {"name": "BRAM_vmm",        "dims": [N]},
        ])
    elif computation_order == "option_3":
        brams.extend([
            {"name": "BRAM_gemm_bias",        "dims": [M, N]},
            {"name": "BRAM_gemm",        "dims": [M, N]},
            {"name": "BRAM_mmv_bias",        "dims": [M]},
            {"name": "BRAM_mmv",        "dims": [M]},
        ])
    elif computation_order == "option_4":
        brams.extend([
            {"name": "BRAM_By_bias",        "dims": [K]},
            {"name": "BRAM_By",        "dims": [K]},
            {"name": "BRAM_ABy_bias",        "dims": [M]},
            {"name": "BRAM_ABy",        "dims": [M]},
        ])
    elif computation_order == "option_5":
        brams.extend([
            {"name": "BRAM_xt_bias",        "dims": [K]},
            {"name": "BRAM_xt",        "dims": [K]},
            {"name": "BRAM_xt_bias",        "dims": [K]},
            {"name": "BRAM_yt",        "dims": [K]},
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
        {"name": "DRAM_x", "dims": [M], "bundle": "mem1"},
        {"name": "DRAM_A", "dims": [M, K], "bundle": "mem1"},
        {"name": "DRAM_B", "dims": [K, N], "bundle": "mem1"},
        {"name": "DRAM_y", "dims": [N], "bundle": "mem1"},
        {"name": "DRAM_bias", "dims": [1], "bundle": "mem1"},
        {"name": "DRAM_result", "dims": [1], "bundle": "mem2"},
    ]

    drams_lines = []
    for i, d in enumerate(drams):
        comma = "," if i < len(drams) - 1 else ""
        dims_str = "[" + ", ".join(str(x) for x in d["dims"]) + "]"
        drams_lines.append(
            f'{{"name": "{d["name"]}", "dims": {dims_str}, "bundle": "{d["bundle"]}"}}{comma}'
        )
    drams_str = "\n".join(drams_lines)

    # 3) Build ops. Utility functions:
    def one_line_list(lst):
        return "[" + ", ".join(str(v) for v in lst) + "]"

    def quoted_list(lst):
        return "[" + ", ".join(f'"{item}"' for item in lst) + "]"

    ops_order = [
        ("load_1", {"func_name": "load", "dims": [M], "args": ["DRAM_x", "BRAM_x"]}),
        ("load_2", {"func_name": "load", "dims": [M, K], "args": ["DRAM_A", "BRAM_A"]}),
        ("load_3", {"func_name": "load", "dims": [K, N], "args": ["DRAM_B", "BRAM_B"]}),
        ("load_4", {"func_name": "load", "dims": [N], "args": ["DRAM_y", "BRAM_y"]}),
        ("load_5", {"func_name": "load", "dims": [1], "args": ["DRAM_bias", "BRAM_bias"]}),
    ]
    if computation_order == "option_1":
        ops_order.extend([
            ("vmm_1", {
                "func_name": "vmm",
                "dims": [M, K],
                "args": ["BRAM_A", "BRAM_x", "BRAM_xA_bias", "BRAM_xA"],
                "func_info": [order_vm, [unroll_M, unroll_K], intermediate_bias, inline]
            }),

            ("vmm_2", {
                "func_name": "vmm",
                "dims": [K, N],
                "args": ["BRAM_B", "BRAM_y", "BRAM_xAB_bias", "BRAM_xAB"],
                "func_info": [order_vm, [unroll_K, unroll_N], intermediate_bias, inline]
            }),

            ("dot_1", {
                "func_name": "dot_product",
                "dims": [N],
                "args": ["BRAM_xA", "BRAM_mmv", "BRAM_bias", "BRAM_result"],
                "func_info": [unroll_N, intermediate_bias, inline]
            }),
        ])
    elif computation_order == "option_2":
        ops_order.extend([
            
            ("gemm_1", {
                "func_name": "gemm",
                "dims": [M, K, N],
                "args": ["BRAM_A", "BRAM_B", "BRAM_gemm_bias", "BRAM_gemm"],
                "func_info": [order_gmm, [unroll_M, unroll_K, unroll_N], intermediate_bias, inline]
            }),

            ("vmm_1", {
                "func_name": "vmm",
                "dims": [M, N],
                "args": ["BRAM_gemm", "BRAM_x", "BRAM_vmm_bias", "BRAM_vmm"],
                "func_info": [order_vm, [unroll_M, unroll_N], intermediate_bias, inline]
            }),

            ("dot_1", {
                "func_name": "dot_product",
                "dims": [N],
                "args": ["BRAM_x", "BRAM_vmm", "BRAM_bias", "BRAM_result"],
                "func_info": [unroll_N, intermediate_bias, inline]
            }),
        ])
    elif computation_order == "option_3":
        ops_order.extend([

            ("gemm_1", {
                "func_name": "gemm",
                "dims": [M, K, N],
                "args": ["BRAM_A", "BRAM_B", "BRAM_gemm_bias", "BRAM_gemm"],
                "func_info": [order_gmm, [unroll_M, unroll_K, unroll_N], intermediate_bias, inline]
            }),

            ("mmv_1", {
                "func_name": "mmv",
                "dims": [M, N],
                "args": ["BRAM_gemm", "BRAM_y", "BRAM_mmv_bias", "BRAM_mmv"],
                "func_info": [order_vm, [unroll_M, unroll_N], intermediate_bias, inline]
            }),

            ("dot_1", {
                "func_name": "dot_product",
                "dims": [M],
                "args": ["BRAM_x", "BRAM_mmv", "BRAM_bias", "BRAM_result"],
                "func_info": [unroll_M, intermediate_bias, inline]
            }),
        ])
    elif computation_order == "option_4":
        ops_order.extend([
            ("vmm_1", {
                "func_name": "vmm",
                "dims": [K, N],
                "args": ["BRAM_B", "BRAM_y", "BRAM_By_bias", "BRAM_By"],
                "func_info": [order_vm, [unroll_K, unroll_N], intermediate_bias, inline]
            }),

            ("vmm_2", {
                "func_name": "vmm",
                "dims": [M, K],
                "args": ["BRAM_A", "BRAM_By", "BRAM_ABy_bias", "BRAM_ABy"],
                "func_info": [order_vm, [unroll_M, unroll_K], intermediate_bias, inline]
            }),

            ("dot_1", {
                "func_name": "dot_product",
                "dims": [M],
                "args": ["BRAM_x", "BRAM_ABy", "BRAM_bias", "BRAM_result"],
                "func_info": [unroll_M, intermediate_bias, inline]
            }),
        ])
    elif computation_order == "option_5":
        ops_order.extend([
            ("load_1", {"func_name": "load", "dims": [M], "args": ["DRAM_x", "BRAM_x"]}),
            ("load_2", {"func_name": "load", "dims": [M, K], "args": ["DRAM_A", "BRAM_A"]}),
            ("load_3", {"func_name": "load", "dims": [K, N], "args": ["DRAM_B", "BRAM_B"]}),
            ("load_4", {"func_name": "load", "dims": [N], "args": ["DRAM_y", "BRAM_y"]}),
            ("load_5", {"func_name": "load", "dims": [1], "args": ["DRAM_bias", "BRAM_bias"]}),

            ("vmm_1", {
                "func_name": "vmm",
                "dims": [M, K],
                "args": ["BRAM_A", "BRAM_x", "BRAM_xt_bias", "BRAM_xt"],
                "func_info": [order_vm, [unroll_M, unroll_N], intermediate_bias, inline]
            }),

            ("vmm_2", {
                "func_name": "mmv",
                "dims": [K, N],
                "args": ["BRAM_B", "BRAM_y", "BRAM_yt_bias", "BRAM_yt"],
                "func_info": [order_vm, [unroll_K, unroll_N], intermediate_bias, inline]
            }),

            ("dot_1", {
                "func_name": "dot_product",
                "dims": [K],
                "args": ["BRAM_xt", "BRAM_yt", "BRAM_bias", "BRAM_result"],
                "func_info": [unroll_M, intermediate_bias, inline]
            }),
        ])
    
    ops_order.append(("store", {"func_name": "store", "dims": [1], "args": ["BRAM_result", "DRAM_result"]})),

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
    "output_dram_names": ["DRAM_result"],
    "FPGA_name": "xczu9eg-ffvb1156-2-e",
    "clock_period": 10,
    "task": ["csim", "csynth", "cosim", "export_ip"],
    "data_type": "{DATA_TYPE}",
    "top_func_name": "top"
}}'''
    return text

def main():
    # Define parameter ranges (adjust as needed)
    vals_M = [64, 128]
    vals_K = [64, 128]
    vals_N = [64, 128]
    vals_unroll_M = [1, 8]
    vals_unroll_K = [1, 8]
    vals_unroll_N = [1, 8]
    vals_order = [x for x in itertools.permutations(["i", "j", "k"])]
    comp_order_list = ["option_1", "option_2", "option_3", "option_4", "option_5"]

    # Static parameters
    need_bias_list = [False]
    inline_list = [True]
    data_type_list = ["ap_fixed<16,5>"]

    combinations = itertools.product(
        vals_M, vals_K, vals_N,
        vals_unroll_M, vals_unroll_K, vals_unroll_N,
        vals_order,
        comp_order_list,
        need_bias_list,
        inline_list,
        data_type_list
    )

    output_dir = "auto_generated_configs"
    os.makedirs(output_dir, exist_ok=True)

    # Now iterate over the base combos, conv_type, (groups if needed), and data_type.
    for (m, k, n, unroll_m, unroll_k, unroll_n, ord, comp_order, with_bias, inline, data_type) in combinations:
        config_text = generate_config_text(
            m, k, n,
            unroll_m, unroll_k, unroll_n,
            ord,
            data_type, with_bias, inline, comp_order
        )
        naming_dtype = data_type.replace('<','_').replace('>','_').replace(',','_')
        filename = (
            f"GEMM_config_M{m}_K{k}_N{n}_UM{unroll_m}_UK{unroll_k}_UN{unroll_n}_"
            f"{ord[0]}{ord[1]}{ord[2]}_BIAS_{with_bias}_INLINE_{inline}_COMPORDER_{comp_order}_"
            f"{naming_dtype}.json"
        )
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            f.write(config_text)
        print(f"Generated {filepath}")


    num_combos = len(vals_M) * len(vals_K) * len(vals_N) * len(vals_unroll_M) * len(vals_unroll_K) * len(vals_unroll_N) * len(vals_order) * len(comp_order_list) * len(need_bias_list) * len(inline_list) * len(data_type_list)
    print("Total number of combos:", num_combos)

if __name__ == "__main__":
    main()
