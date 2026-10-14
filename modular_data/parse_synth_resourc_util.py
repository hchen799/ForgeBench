# /// script
# dependencies = [
#     "pandas",
# ]
# ///

import argparse
import os
import xml.etree.ElementTree as ET
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(BASE_DIR, "modularization_results.csv")
OUTPUT_TEX = os.path.join(BASE_DIR, "modularization_table.tex")

REPORT_REL = "project_1/solution1/syn/report"

# By default reports are read in place (<category>/hls_files/<project>/project_1/...), as in the R1 run.
# --reports-root DIR reads the lean layout written by analysis/run_prebuilt.py instead:
# DIR/<category>/<project>/project_1/...
REPORTS_ROOT = None


# Table 7 reports utilization as (LUT, DSP, BRAM) tuples of device percentages (R1 reported the mean of DSP% and LUT%, which hid
# BRAM), so a cell over 100% shows which resource exceeds the device; Shared / % Reuse / % Change print BRAM as a dot (LUT and DSP
# only). The CSV keeps all three for every column.
RESOURCES = [("LUT", "LUT"), ("DSP", "DSP"), ("BRAM", "BRAM_18K")]


def parse_utilization(xml_path):
    """Return (LUT%, DSP%, BRAM%) of the device from an HLS csynth XML, or None if unavailable."""
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
        used = [int(resources.find(tag).text) for _, tag in RESOURCES]
        avail = [int(available.find(tag).text) for _, tag in RESOURCES]
    except (AttributeError, ValueError):
        return None

    if not all(avail):
        return None

    return tuple(u / a * 100 for u, a in zip(used, avail))


def _report_dir(category, project):
    if REPORTS_ROOT:
        return os.path.join(REPORTS_ROOT, category, project, REPORT_REL)
    return os.path.join(BASE_DIR, category, "hls_files", project, REPORT_REL)


def project_xml(category, project):
    return os.path.join(_report_dir(category, project), "csynth.xml")


def function_xml(category, project, function):
    return os.path.join(_report_dir(category, project), f"{function}_csynth.xml")


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


def _sum(parts):
    """Element-wise sum of (LUT, DSP, BRAM) tuples; None if any part is missing."""
    if not parts or any(v is None for v in parts):
        return None
    return tuple(sum(v[k] for v in parts) for k in range(len(RESOURCES)))


def _ratio(num, den, offset=0.0):
    """Element-wise (num / den - offset) * 100; a resource with den == 0 gives None."""
    if num is None or den is None:
        return None
    return tuple(None if d == 0 else (n / d - offset) * 100 for n, d in zip(num, den))


def compute_row(row):
    cat = row["category"]
    p_utils = [None if p is None else parse_utilization(project_xml(cat, p)) for p in row["programs"]]
    module_total = parse_utilization(project_xml(cat, row["module"]))
    shared_total = _sum([parse_utilization(function_xml(cat, row["module"], f)) for f in row["shared"]])
    program_sum = _sum([v for p, v in zip(row["programs"], p_utils) if p is not None])

    return {
        "Test Suite": row["suite"],
        "Test Case": row["case"],
        "P1": p_utils[0],
        "P2": p_utils[1],
        "P3": p_utils[2],
        "Total_Before": program_sum,
        "Shared": shared_total,
        "Pct_Reuse": _ratio(shared_total, module_total),
        "Total_After": module_total,
        "Pct_Change": _ratio(module_total, program_sum, offset=1.0),
    }


COLUMNS = ["P1", "P2", "P3", "Total_Before", "Shared", "Pct_Reuse", "Total_After", "Pct_Change"]


def _num(x, pct):
    if x is None:
        return "-"
    if pct or abs(x) >= 10:
        return f"{x:.0f}"
    return f"{x:.1f}"


def fmt(v, pct=False, bram=True):
    """(LUT, DSP, BRAM) tuple as '(l, d, b)': utilization with one decimal below 10%, % reuse / % change as integers; '-' for a
    missing cell or resource (a resource the denominator does not use). bram=False prints the BRAM entry as a centered dot."""
    if v is None:
        return "-"
    vals = [_num(x, pct) for x in v]
    if not bram:
        vals[2] = r"$\cdot$"
    return "(" + ", ".join(vals) + ")"


def render_latex(rows):
    suite_counts = {}
    for r in rows:
        suite_counts[r["Test Suite"]] = suite_counts.get(r["Test Suite"], 0) + 1

    lines = []
    lines.append(r"\begin{table*}[!tbh]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{L{0.8cm}L{2.6cm}cccccccc}")
    lines.append(r"\toprule")
    lines.append("")
    lines.append(r"    \multirow{2}{*}{\centering \textbf{Test Suite}} & ")
    lines.append(r"    \multirow{2}{*}{\centering \textbf{Test Case}} & ")
    lines.append(r"    \multicolumn{4}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Resource Utilization Before Modularization\\(LUT, DSP, BRAM)\end{tabular}}} & ")
    lines.append(r"    \multicolumn{3}{c}{\textbf{\begin{tabular}[c]{@{}c@{}}Resource Utilization After Modularization\\(LUT, DSP, BRAM)\end{tabular}}} &")
    lines.append(r"    \multirow{2}{*}{\textbf{\% Change}} \\ ")
    lines.append("")
    lines.append(r"    \cmidrule(lr){3-6} ")
    lines.append(r"    \cmidrule(lr){7-9}")
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
            fmt(r["Shared"], bram=False) + " & " +
            r"\textbf{" + fmt(r["Pct_Reuse"], pct=True, bram=False) + "} & " +
            fmt(r["Total_After"]) + " & " +
            r"\textbf{" + fmt(r["Pct_Change"], pct=True, bram=False) + r"} \\"
        )
        lines.append(line)
        prev_suite = suite

    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \caption{ Resource usage of hardware test cases as (LUT, DSP, BRAM) percentages of the ZCU102 (C-synthesis estimates, \texttt{ap\_fixed<32,10>}), annotated according to the available reuse $\dagger$: tiling reuse, $\ddagger$: functional reuse, $\ast$: arithmetic reuse. Shared, \% Reuse and \% Change cover LUT and DSP only ($\cdot$): BRAM is dominated by each design's top-level buffers for inputs, weights and intermediate results, which are sized by the data rather than by the shared functions, so a BRAM reuse or change figure would reflect buffer sizing rather than function sharing. }")
    lines.append(r"    \label{tab:modularization_results}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    global REPORTS_ROOT, OUTPUT_CSV, OUTPUT_TEX
    ap = argparse.ArgumentParser(description="Recompute the modularization table (Table 7) from csynth reports")
    ap.add_argument("--reports-root", help="lean report root written by analysis/run_prebuilt.py (default: read in place)")
    ap.add_argument("--out", help="directory for modularization_results.csv/.tex (default: this directory)")
    args = ap.parse_args()
    if args.reports_root:
        REPORTS_ROOT = os.path.abspath(args.reports_root)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        OUTPUT_CSV = os.path.join(args.out, "modularization_results.csv")
        OUTPUT_TEX = os.path.join(args.out, "modularization_table.tex")
    rows = [compute_row(r) for r in ROWS]

    flat = []
    for r in rows:
        f = {"Test Suite": r["Test Suite"], "Test Case": r["Test Case"]}
        for col in COLUMNS:
            for k, (name, _) in enumerate(RESOURCES):
                v = r[col]
                f[f"{col}_{name}"] = None if v is None or v[k] is None else round(v[k], 2)
        flat.append(f)
    pd.DataFrame(flat).to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}")

    tex = render_latex(rows)
    with open(OUTPUT_TEX, "w") as f:
        f.write(tex + "\n")
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
