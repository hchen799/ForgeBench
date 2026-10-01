
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
    vm_orders=None,
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
    # Loop order of each vector-matrix op (vmm/mmv), by position in the option's op list.
    # Default (vm_orders=None): every vm op uses the i/j subsequence of `order` (original behaviour).
    # Options 1/4/5 have no gemm op, so `order` is meaningless there; pass vm_orders=("ij"|"ji", "ij"|"ji").
    if vm_orders is None:
        order_vm_1 = order_vm_2 = order_vm
    else:
        order_vm_1, order_vm_2 = list(vm_orders[0]), list(vm_orders[1])

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
            {"name": "BRAM_yt_bias",        "dims": [K]},
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
                "func_info": [order_vm_1, [unroll_M, unroll_K], intermediate_bias, inline]
            }),

            ("vmm_2", {
                "func_name": "vmm",
                "dims": [K, N],
                "args": ["BRAM_B", "BRAM_xA", "BRAM_xAB_bias", "BRAM_xAB"],
                "func_info": [order_vm_2, [unroll_K, unroll_N], intermediate_bias, inline]
            }),

            ("dot_1", {
                "func_name": "dot_product",
                "dims": [N],
                "args": ["BRAM_xAB", "BRAM_y", "BRAM_bias", "BRAM_result"],
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
                "func_info": [order_vm_1, [unroll_M, unroll_N], intermediate_bias, inline]
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
                "func_info": [order_vm_1, [unroll_M, unroll_N], intermediate_bias, inline]
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
                "func_info": [order_vm_1, [unroll_K, unroll_N], intermediate_bias, inline]
            }),

            ("vmm_2", {
                "func_name": "vmm",
                "dims": [M, K],
                "args": ["BRAM_A", "BRAM_By", "BRAM_ABy_bias", "BRAM_ABy"],
                "func_info": [order_vm_2, [unroll_M, unroll_K], intermediate_bias, inline]
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
                "func_info": [order_vm_1, [unroll_M, unroll_N], intermediate_bias, inline]
            }),

            ("vmm_2", {
                "func_name": "mmv",
                "dims": [K, N],
                "args": ["BRAM_B", "BRAM_y", "BRAM_yt_bias", "BRAM_yt"],
                "func_info": [order_vm_2, [unroll_K, unroll_N], intermediate_bias, inline]
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
    "task": ["csynth"],
    "data_type": "{DATA_TYPE}",
    "top_func_name": "top"
}}'''
    return text

# ---- Sweep specification (single source of truth; imported by manifest/ and paper_artifacts/) ----
# Loop orders are per-op and only exist where an op has them:
#   * options 2, 3 contain a gemm op: gemm_order is one of the 6 permutations of (i,j,k); the single
#     vector-matrix op uses the i/j subsequence of it (so vm_order_1 is derived, vm_order_2 is n/a).
#   * options 1, 4, 5 contain no gemm op (only vmm/mmv + dot_product): there is no 3-loop order, instead
#     each of the two vector-matrix ops independently picks "ij" or "ji" (4 combinations).
# The earlier sweep crossed all 6 permutations with every option, so options 1/4/5 contained 3 byte-identical
# copies of each distinct design (see docs/revision_r2/CHANGELOG_R2.md).
SWEEP = {
    "M": [64, 128],
    "K": [64, 128],
    "N": [64, 128],
    "unroll_M": [1, 8],
    "unroll_K": [1, 8],
    "unroll_N": [1, 8],
    "comp_order": ["option_1", "option_2", "option_3", "option_4", "option_5"],
    "gemm_order": ["".join(x) for x in itertools.permutations(["i", "j", "k"])],   # options 2, 3
    "vm_order": ["ij", "ji"],                                                       # per vm op, options 1, 4, 5
    "with_bias": [False, True],
    "inline": [True],
    "data_type": ["ap_fixed<16,5>"],
}
GEMM_OPTIONS = ("option_2", "option_3")


def _vm_of(order):
    return "".join(x for x in order if x in "ij")


def iter_params():
    """Yield one dict of sweep parameters per design, in generation order."""
    for (m, k, n, um, uk, un, comp, bias, inline, dtype) in itertools.product(
            SWEEP["M"], SWEEP["K"], SWEEP["N"], SWEEP["unroll_M"], SWEEP["unroll_K"], SWEEP["unroll_N"],
            SWEEP["comp_order"], SWEEP["with_bias"], SWEEP["inline"], SWEEP["data_type"]):
        if comp in GEMM_OPTIONS:
            orders = [(g, _vm_of(g), "") for g in SWEEP["gemm_order"]]
        else:
            orders = [("", v1, v2) for v1 in SWEEP["vm_order"] for v2 in SWEEP["vm_order"]]
        for (g, v1, v2) in orders:
            yield {"M": m, "K": k, "N": n, "unroll_M": um, "unroll_K": uk, "unroll_N": un,
                   "comp_order": comp, "gemm_order": g, "vm_order_1": v1, "vm_order_2": v2,
                   "with_bias": bias, "inline": inline, "data_type": dtype}


def config_stem(p):
    dtype = p["data_type"].replace('<', '_').replace('>', '_').replace(',', '_')
    return (f"GEMM_config_M{p['M']}_K{p['K']}_N{p['N']}_UM{p['unroll_M']}_UK{p['unroll_K']}_UN{p['unroll_N']}_"
            f"GORD{p['gemm_order'] or 'na'}_VM1{p['vm_order_1']}_VM2{p['vm_order_2'] or 'na'}_"
            f"BIAS_{p['with_bias']}_INLINE_{p['inline']}_COMPORDER_{p['comp_order']}_{dtype}")


def legacy_stem(p):
    """Design id this design had in the original (R1) sweep, or None if it did not exist there.

    R1 crossed 6 permutations with every option; for options 1/4/5 only the i/j subsequence mattered, so
    a design whose two vm orders agree has the same source as the R1 designs with the matching permutation
    (R1 'ijk' for ij/ij and 'jik' for ji/ji were the first-generated copies). Mixed (ij,ji) designs are new.
    """
    if p["comp_order"] in GEMM_OPTIONS:
        order = p["gemm_order"]
    elif p["vm_order_1"] == p["vm_order_2"]:
        order = {"ij": "ijk", "ji": "jik"}[p["vm_order_1"]]
    else:
        return None
    dtype = p["data_type"].replace('<', '_').replace('>', '_').replace(',', '_')
    return (f"GEMM_config_M{p['M']}_K{p['K']}_N{p['N']}_UM{p['unroll_M']}_UK{p['unroll_K']}_UN{p['unroll_N']}_"
            f"{order}_BIAS_{p['with_bias']}_INLINE_{p['inline']}_COMPORDER_{p['comp_order']}_{dtype}")


def build_config_text(p):
    if p["comp_order"] in GEMM_OPTIONS:
        return generate_config_text(
            p["M"], p["K"], p["N"], p["unroll_M"], p["unroll_K"], p["unroll_N"],
            tuple(p["gemm_order"]), p["data_type"], p["with_bias"], p["inline"], p["comp_order"])
    return generate_config_text(
        p["M"], p["K"], p["N"], p["unroll_M"], p["unroll_K"], p["unroll_N"],
        ("i", "j", "k"), p["data_type"], p["with_bias"], p["inline"], p["comp_order"],
        vm_orders=(p["vm_order_1"], p["vm_order_2"]))


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
