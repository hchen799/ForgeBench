# /// script
# dependencies = [
#     "pandas",
# ]
# ///

import os
import xml.etree.ElementTree as ET
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(BASE_DIR, "modularization_results.csv")
OUTPUT_TEX = os.path.join(BASE_DIR, "modularization_table.tex")

REPORT_REL = "project_1/solution1/syn/report"


def parse_average_utilization(xml_path):
    """Return average of (DSP%, LUT%) from an HLS csynth XML, or None if unavailable."""
    if not os.path.exists(xml_path):
        return None
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None
    root = tree.getroot()

    area_report = root.find("AreaEstimates")
    if area_report is None:
        return None
    resources = area_report.find("Resources")
    available = area_report.find("AvailableResources")
    if resources is None or available is None:
        return None

    try:
        dsp = int(resources.find("DSP").text)
        lut = int(resources.find("LUT").text)
        avail_dsp = int(available.find("DSP").text)
        avail_lut = int(available.find("LUT").text)
    except (AttributeError, ValueError):
        return None

    if not avail_dsp or not avail_lut:
        return None

    return ((dsp / avail_dsp) * 100 + (lut / avail_lut) * 100) / 2


def project_xml(category, project):
    return os.path.join(BASE_DIR, category, "hls_files", project, REPORT_REL, "csynth.xml")


def function_xml(category, project, function):
    return os.path.join(BASE_DIR, category, "hls_files", project, REPORT_REL, f"{function}_csynth.xml")


# Row-by-row description of the table. shared is a list of function names whose
# areas are summed for the "Shared" column.
ROWS = [
    {
        "suite": "GEMM", "case": r"Tiled GEMM $\dagger$", "category": "gemm",
        "programs": ["diff_dims_p1", "diff_dims_p2", "diff_dims_p3"],
        "module": "diff_dims_module_small",
        "shared": ["tiled_gemm"],
    },
    {
        "suite": "GEMM", "case": r"Vec/Mtx Mult - Dot $\dagger$ $\ast$", "category": "gemm",
        "programs": ["mult_op_p1", "mult_op_p2", "mult_op_p3"],
        "module": "mult_op_module_dot",
        "shared": ["dot_product"],
    },
    {
        "suite": "GEMM", "case": r"Vec/Mtx Mult - MMV $\dagger$ $\ast$", "category": "gemm",
        "programs": ["mult_op_p1", "mult_op_p2", "mult_op_p3"],
        "module": "mult_op_module_mmv",
        "shared": ["mmv_ij"],
    },
    {
        "suite": "GEMM", "case": r"Vec/Mtx Mult - GEMM $\dagger$ $\ast$", "category": "gemm",
        "programs": ["mult_op_p1", "mult_op_p2", "mult_op_p3"],
        "module": "mult_op_module_mm",
        "shared": ["gemm_ijk"],
    },
    {
        "suite": "GEMM", "case": r"i-j-k Orders $\ast$", "category": "gemm",
        "programs": ["diff_orders_p1", "diff_orders_p2", "diff_orders_p3"],
        "module": "diff_orders_module",
        "shared": ["common_gemm_ikj"],
    },
    {
        "suite": "GEMM", "case": r"Vector Transpose $\ast$", "category": "gemm",
        "programs": ["vec_mtx_p1", "vec_mtx_p2", None],
        "module": "vec_mtx_module",
        "shared": ["mmv_ij"],
    },
    {
        "suite": "DNN", "case": r"Activation Functions $\ddagger$ $\ast$", "category": "conv",
        "programs": ["activation_op1", "activation_op2", "activation_op3"],
        "module": "activation_module",
        "shared": ["compute_exp"],
    },
    {
        "suite": "DNN", "case": r"Tiled Convolution $\dagger$ $\ast$", "category": "conv",
        "programs": ["conv_A", "conv_B", "conv_C"],
        "module": "conv_module",
        "shared": ["conv_via_tiling_3x3"],
    },
    {
        "suite": "DNN", "case": r"DNN Blocks $\ddagger$", "category": "conv",
        "programs": ["conv_block_op1", "conv_block_op2", "conv_block_op3"],
        "module": "conv_block_module",
        "shared": ["conv_via_tiling_3x3_1", "batch_norm_tiled", "relu_tiled"],
    },
    {
        "suite": "LLM", "case": r"Tiled Attention $\dagger$ $\ast$", "category": "llm",
        "programs": ["tiled_attn_p1", "tiled_attn_p2", None],
        "module": "tiled_attn_module",
        "shared": ["grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_s"],
    },
    {
        "suite": "LLM", "case": r"Functional Attention $\ddagger$", "category": "llm",
        "programs": ["attn_breakdown_op1", "attn_breakdown_op2", None],
        "module": "attn_breakdown_module",
        "shared": [
            "matmul_large",
            "matmul_small_1",
            "calculate_scale",
            "softmax",
            "matmul_small_2",
        ],
    },
    {
        "suite": "LLM", "case": r"LLaMA/GPT Transformers $\ddagger$", "category": "llm",
        "programs": ["gpt_transformer_p1", "llama_transformer_p2", None],
        "module": "Llama_GPT_module",
        "shared": [
            "grouped_multihead_attention_8_32_8_4_rope_ap_fixed_16_5_s",
            "dropout_8_32_ap_fixed_16_5_s",
            "matrix_add_8_32_ap_fixed_16_5_s",
            "matmul_8_32_128_ap_fixed_16_5_s",
            "matmul_8_128_32_ap_fixed_16_5_s",
        ],
    },
]


def compute_row(row):
    cat = row["category"]
    p_utils = []
    for p in row["programs"]:
        if p is None:
            p_utils.append(None)
        else:
            p_utils.append(parse_average_utilization(project_xml(cat, p)))

    module_total = parse_average_utilization(project_xml(cat, row["module"]))

    shared_parts = [
        parse_average_utilization(function_xml(cat, row["module"], f))
        for f in row["shared"]
    ]
    if any(v is None for v in shared_parts):
        shared_total = None
    else:
        shared_total = sum(shared_parts)

    expected_progs = [p for p in row["programs"] if p is not None]
    available_p = [v for v in p_utils if v is not None]
    if available_p and len(available_p) == len(expected_progs):
        program_sum = sum(available_p)
    else:
        program_sum = None

    if shared_total is not None and module_total not in (None, 0):
        pct_reuse = (shared_total / module_total) * 100
    else:
        pct_reuse = None

    if program_sum not in (None, 0) and module_total is not None:
        pct_change = ((module_total - program_sum) / program_sum) * 100
    else:
        pct_change = None

    return {
        "Test Suite": row["suite"],
        "Test Case": row["case"],
        "P1": p_utils[0],
        "P2": p_utils[1],
        "P3": p_utils[2],
        "Total_Before": program_sum,
        "Shared": shared_total,
        "Pct_Reuse": pct_reuse,
        "Total_After": module_total,
        "Pct_Change": pct_change,
    }


def fmt(v, dp=2):
    if v is None:
        return "-"
    return f"{v:.{dp}f}"


def fmt_int(v):
    if v is None:
        return "-"
    return f"{v:.0f}"


def render_latex(rows):
    suite_counts = {}
    for r in rows:
        suite_counts[r["Test Suite"]] = suite_counts.get(r["Test Suite"], 0) + 1

    lines = []
    lines.append(r"\begin{table*}[!tbh]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{L{1.1cm}L{2.9cm}C{0.6cm}C{0.6cm}C{0.6cm}R{0.8cm}C{0.8cm}C{0.9cm}R{0.8cm}C{2.5cm}}")
    lines.append(r"\toprule")
    lines.append("")
    lines.append(r"    \multirow{2}{*}{\centering \textbf{Test Suite}} & ")
    lines.append(r"    \multirow{2}{*}{\centering \textbf{Test Case}} & ")
    lines.append(r"    \multicolumn{4}{L{3.3cm}}{\textbf{Resource Utilization Before Modularization}} & ")
    lines.append(r"    \multicolumn{3}{L{3.2cm}}{\textbf{Resource Utilization After Modularization}} &")
    lines.append(r"    \multirow{2}{2.5cm}{\textbf{\% Change in Resource Utilization}} \\ ")
    lines.append("")
    lines.append(r"    \cmidrule{3-6} ")
    lines.append(r"    \cmidrule{7-9}")
    lines.append("")
    lines.append(r"    & & P1 & P2 & P3 & \textbf{Total} & Shared & \textbf{\% Reuse} & \textbf{Total} & \\ ")
    lines.append("")
    lines.append(r"    \midrule")

    seen_suite = set()
    prev_suite = None
    for r in rows:
        suite = r["Test Suite"]
        if suite != prev_suite and prev_suite is not None:
            lines.append(r"    \midrule")
        if suite not in seen_suite:
            count = suite_counts[suite]
            suite_cell = r"\multirow{" + str(count) + r"}{*}{\centering \textbf{" + suite + r"}} & "
            seen_suite.add(suite)
        else:
            suite_cell = "    & "

        line = (
            "    " + suite_cell + r["Test Case"] + " & " +
            fmt(r["P1"]) + " & " +
            fmt(r["P2"]) + " & " +
            fmt(r["P3"]) + " & " +
            fmt(r["Total_Before"]) + " & " +
            fmt(r["Shared"]) + " & " +
            r"\textbf{" + fmt(r["Pct_Reuse"]) + "} & " +
            fmt(r["Total_After"]) + " & " +
            r"\textbf{" + fmt(r["Pct_Change"]) + r"} \\"
        )
        lines.append(line)
        prev_suite = suite

    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \caption{ Resource usage of hardware test cases, annotated according to the available reuse $\dagger$: tiling reuse, $\ddagger$: functional reuse, $\ast$: arithmetic reuse. }")
    lines.append(r"    \label{tab:modularization_results}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    rows = [compute_row(r) for r in ROWS]

    df = pd.DataFrame(rows)
    for col in ["P1", "P2", "P3", "Total_Before", "Shared", "Pct_Reuse", "Total_After", "Pct_Change"]:
        df[col] = df[col].apply(lambda v: round(v, 2) if v is not None else None)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}")

    tex = render_latex(rows)
    with open(OUTPUT_TEX, "w") as f:
        f.write(tex + "\n")
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
