"""Regenerate the paper's data tables from committed/collected CSVs.

- table2: full-scale model resource min/max (paper Table 2) from a scale-models
  csynth CSV (default scale_models/hls_synth_utilization.csv).
- table3: benchmark-suite testcase counts (paper Table 3).
- table5: modularization %reuse / %change (paper Table 5) from
  modular_data/modularization_results.csv.

Each emits a normalized CSV and a LaTeX snippet under analysis/tables_out/.
"""
import argparse
import os

import pandas as pd

from analysis import io

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "tables_out")

# Paper Table 3 counts (documented). Regenerate from the actual generated suite to
# verify — see the count-discrepancy flag in analysis/README.md (README 6000+ vs
# paper 10,837).
PAPER_TABLE3 = [
    {"class": "GEMM", "base_design": "x . A . B . y", "n_testcases": 3335},
    {"class": "DNN", "base_design": "Conv-BatchNorm-Activation", "n_testcases": 4143},
    {"class": "LLM", "base_design": "Attention-Dropout-Norm", "n_testcases": 3359},
]


def _to_latex(df):
    """Minimal booktabs LaTeX tabular (no jinja2 dependency)."""
    cols = list(df.columns)
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    lines = [r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule",
             " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(esc(v) for v in row.tolist()) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def _write(df, name, index=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"{name}.csv")
    tex_path = os.path.join(OUT_DIR, f"{name}.tex")
    df.to_csv(csv_path, index=index)
    with open(tex_path, "w") as f:
        f.write(_to_latex(df.reset_index() if index else df))
    print(f"wrote {csv_path} and {tex_path}")
    return csv_path


def table2_fullscale_minmax(scale_csv=None):
    """Paper Table 2: per-model min/max LUT(x1000)/BRAM/DSP over config sweeps."""
    scale_csv = scale_csv or os.path.join(REPO_ROOT, "scale_models", "hls_synth_utilization.csv")
    df = io.load_legacy_csv(scale_csv)
    # Model family = Project name before "_config_".
    df["model"] = df["design"].str.split("_config_").str[0]
    g = df.groupby("model")
    out = pd.DataFrame({
        "LUT_min_k": (g["lut"].min() / 1000).round(0),
        "LUT_max_k": (g["lut"].max() / 1000).round(0),
        "BRAM_min": g["bram"].min(),
        "BRAM_max": g["bram"].max(),
        "DSP_min": g["dsp"].min(),
        "DSP_max": g["dsp"].max(),
    }).reset_index()
    n_models = out["model"].nunique()
    single_cfg = (g.size() == 1).all()
    if single_cfg:
        print("NOTE: sample scale-models CSV has one config per model, so min==max. "
              "Full min/max needs the server csynth sweep across tiling/unroll configs. [FLAG]")
    print(f"table2: {n_models} models")
    return _write(out, "table2_fullscale_minmax")


def table3_counts(configs_root=None):
    """Paper Table 3: testcase counts per domain.

    Counts *.json in each domain's generated-config dir if available; otherwise
    falls back to the paper-reported numbers with a flag.
    """
    rows = []
    all_found = True
    for entry in PAPER_TABLE3:
        domain_dir = {"GEMM": "gemm", "DNN": "conv", "LLM": "llm"}[entry["class"]]
        cfg_dir = configs_root or os.path.join(REPO_ROOT, domain_dir, "auto_generated_configs")
        if os.path.isdir(cfg_dir):
            n = sum(1 for f in os.listdir(cfg_dir) if f.endswith(".json"))
            src = "counted"
        else:
            n = entry["n_testcases"]
            src = "paper (generated suite not present)"
            all_found = False
        rows.append({"class": entry["class"], "base_design": entry["base_design"],
                     "n_testcases": n, "source": src})
    df = pd.DataFrame(rows)
    total = df["n_testcases"].sum()
    print(f"table3: total testcases = {total}")
    if not all_found:
        print("NOTE: counts are paper-reported (generated suites not in repo). "
              "Regenerate suites to verify; also reconcile README 6000+ vs paper 10,837. [FLAG]")
    return _write(df, "table3_counts")


def table5_modular_reuse(modular_csv=None):
    """Paper Table 5: modularization %reuse / %change (already tabulated)."""
    modular_csv = modular_csv or os.path.join(REPO_ROOT, "modular_data", "modularization_results.csv")
    df = pd.read_csv(modular_csv)
    print(f"table5: {len(df)} test cases across {df['Test Suite'].nunique()} suites")
    return _write(df, "table5_modular_reuse")


_TABLES = {
    "table2": table2_fullscale_minmax,
    "table3": table3_counts,
    "table5": table5_modular_reuse,
}


def main():
    ap = argparse.ArgumentParser(description="Regenerate ForgeBench paper tables")
    ap.add_argument("which", nargs="+", choices=sorted(_TABLES) + ["all"])
    args = ap.parse_args()
    which = sorted(_TABLES) if "all" in args.which else args.which
    for name in which:
        _TABLES[name]()


if __name__ == "__main__":
    main()
