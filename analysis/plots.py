"""Scatter plots for ForgeBench design-space coverage and PPA trade-offs.

- fig4_coverage: reproduces paper Figure 4 (BRAM/DSP/LUT util vs Cycles, plus
  BRAM-vs-DSP util), colored by domain. Faithful to the original
  plotting/analyze_report_data_new_plot_final.py (same panels, log-y, filters).
- resource_vs_latency: resource util vs end-to-end latency (ns).
- resource_vs_power: resource util vs total power (STUB+FLAG: needs impl power
  data; no-op with a clear message until metrics contain power_w).

By default reads the committed legacy coverage CSVs
(plotting/{gemm,conv,llm}_data). Pass --metrics <metrics.csv> to plot a unified
metrics file from analysis.collect instead.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis import io

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOMAIN_STYLE = {
    "gemm": {"color": "lightseagreen", "marker": "s", "label": "GEMM"},
    "conv": {"color": "coral", "marker": "s", "label": "CONV"},
    "llm": {"color": "mediumslateblue", "marker": "s", "label": "LLM"},
}

_DEFAULT_COVERAGE_DIRS = {
    os.path.join(REPO_ROOT, "plotting", "gemm_data"): "gemm",
    os.path.join(REPO_ROOT, "plotting", "conv_data"): "conv",
    os.path.join(REPO_ROOT, "plotting", "llm_data"): "llm",
}


def _load_default_coverage():
    return io.load_coverage_dirs(_DEFAULT_COVERAGE_DIRS)


def _filter(df, bram_min=0, bram_max=100, dsp_max=100, lut_max=100, cycles_max=1e9):
    d = df.copy()
    keep = pd.Series(True, index=d.index)
    if "bram_util" in d:
        keep &= d["bram_util"].between(bram_min, bram_max)
    if "dsp_util" in d:
        keep &= d["dsp_util"] <= dsp_max
    if "lut_util" in d:
        keep &= d["lut_util"] <= lut_max
    if "cycles" in d:
        keep &= d["cycles"] <= cycles_max
    return d[keep.fillna(False)]


def fig4_coverage(df=None, out=None):
    """4-panel design-space coverage figure (paper Fig 4)."""
    if df is None:
        df = _load_default_coverage()
    df = _filter(df)
    out = out or os.path.join(REPO_ROOT, "analysis", "figures", "fig4_coverage.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    panels = [  # (x_col, y_col, title, xlabel, ylabel, ylog)
        ("bram_util", "cycles", "BRAM Util vs Cycles", "BRAM Utilization (%)", "Cycles", True),
        ("dsp_util", "cycles", "DSP Util vs Cycles", "DSP Utilization (%)", "Cycles", True),
        ("lut_util", "cycles", "LUT Util vs Cycles", "LUT Utilization (%)", "Cycles", True),
        ("bram_util", "dsp_util", "BRAM vs DSP Util", "BRAM Utilization (%)", "DSP Utilization (%)", False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(36, 10))
    for ax, (xc, yc, title, xl, yl, ylog) in zip(axes, panels):
        for domain, style in DOMAIN_STYLE.items():
            sub = df[df["domain"] == domain]
            ax.scatter(sub[xc], sub[yc], marker=style["marker"], s=30,
                       c=style["color"], label=style["label"], alpha=0.7)
        ax.set_title(title, fontsize=32, fontweight="bold", pad=20)
        ax.set_xlabel(xl, fontsize=32, fontweight="bold")
        ax.set_ylabel(yl, fontsize=32, fontweight="bold")
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="both", labelsize=28)
        ax.set_xlim(left=(15 if "bram" in xc else -5), right=105)
        if ylog:
            ax.set_yscale("log")
        else:
            ax.set_ylim(-5, 105)

    handles, labels = axes[0].get_legend_handles_labels()
    for h in handles:
        h.set_alpha(1.0)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, fontsize=32, markerscale=6.0, frameon=True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({len(df)} points)")
    return out


def resource_vs_latency(df=None, out=None):
    """Resource utilization vs end-to-end latency (ns) per resource."""
    if df is None:
        df = _load_default_coverage()
    ycol = "latency_ns" if ("latency_ns" in df and df["latency_ns"].notna().any()) else "cycles"
    df = _filter(df)
    out = out or os.path.join(REPO_ROOT, "analysis", "figures", "resource_vs_latency.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(27, 9))
    for ax, rc in zip(axes, ["bram_util", "dsp_util", "lut_util"]):
        for domain, style in DOMAIN_STYLE.items():
            sub = df[df["domain"] == domain]
            ax.scatter(sub[rc], sub[ycol], marker="s", s=30, c=style["color"],
                       label=style["label"], alpha=0.7)
        ax.set_title(f"{rc.split('_')[0].upper()} Util vs {'Latency' if ycol=='latency_ns' else 'Cycles'}",
                     fontsize=26, fontweight="bold")
        ax.set_xlabel(f"{rc.split('_')[0].upper()} Utilization (%)", fontsize=24, fontweight="bold")
        ax.set_ylabel("Latency (ns)" if ycol == "latency_ns" else "Cycles", fontsize=24, fontweight="bold")
        ax.set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=22)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({len(df)} points, y={ycol})")
    return out


def resource_vs_power(df=None, out=None):
    """Resource utilization vs total power. STUB+FLAG: needs impl power data."""
    if df is None or "power_w" not in df or df["power_w"].notna().sum() == 0:
        print("resource_vs_power: SKIPPED — no power data available yet. "
              "Provide a Vitis impl report sample so analysis.extractors.vitis can "
              "populate power_w, then re-run. [TODO]")
        return None
    out = out or os.path.join(REPO_ROOT, "analysis", "figures", "resource_vs_power.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(27, 9))
    for ax, rc in zip(axes, ["bram_util", "dsp_util", "lut_util"]):
        for domain, style in DOMAIN_STYLE.items():
            sub = df[df["domain"] == domain]
            ax.scatter(sub[rc], sub["power_w"], marker="s", s=30, c=style["color"],
                       label=style["label"], alpha=0.7)
        ax.set_xlabel(f"{rc.split('_')[0].upper()} Utilization (%)", fontsize=24, fontweight="bold")
        ax.set_ylabel("Power (W)", fontsize=24, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=22)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


_PLOTS = {
    "fig4": fig4_coverage,
    "resource_vs_latency": resource_vs_latency,
    "resource_vs_power": resource_vs_power,
}


def main():
    ap = argparse.ArgumentParser(description="ForgeBench analysis plots")
    ap.add_argument("which", nargs="+", choices=sorted(_PLOTS) + ["all"])
    ap.add_argument("--metrics", default=None, help="unified metrics.csv (default: committed coverage CSVs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = io.read_metrics_csv(args.metrics) if args.metrics else None
    which = sorted(_PLOTS) if "all" in args.which else args.which
    for name in which:
        _PLOTS[name](df=df, out=args.out if len(which) == 1 else None)


if __name__ == "__main__":
    main()
