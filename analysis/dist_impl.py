"""Domain-grouped distribution plot from impl data (power + cycles).

Two panels — Power (W, linear) and Cycles (log) — each an overlapping filled KDE
per domain (GEMM cyan / CONV yellow / LLM red), matching the reference
distribution figures.

By default reads the unified impl metrics in analysis/results_impl/
(metrics_{gemm,conv,llm}.csv, which carry a populated `power_w` column); pass
--metrics <metrics.csv> for a single file. Use --feasible-only to drop designs
that don't fit the device (util > 100%). Skips with a clear message if no power
data is present.

  uv run python -m analysis.dist_impl
  uv run python -m analysis.dist_impl --feasible-only
  uv run python -m analysis.dist_impl --metrics <impl metrics.csv> --out fig.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import io
from analysis.density import DOMAIN_STYLE, density_panel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DEFAULT_DIR = os.path.join(REPO_ROOT, "analysis", "results_impl")

# (value_col, title, xlabel, log-x)
_PANELS = [
    ("power_w", "Power", "Total On-Chip Power (W)", False),
    ("cycles", "Cycles", "Latency (Cycles)", True),
]


def dist_impl(df=None, out=None, feasible=False):
    if df is None:
        df = io.read_metrics_dir(_DEFAULT_DIR)
    if feasible and not df.empty:
        df = io.feasible_only(df)
    if df.empty or "power_w" not in df.columns or df["power_w"].notna().sum() == 0:
        print("dist_impl: SKIPPED — no impl metrics with power_w. "
              "Pass an impl metrics.csv (analysis.collect --flow impl) once the "
              "impl reports are parsed, then re-run.")
        return None
    out = out or os.path.join(REPO_ROOT, "analysis", "figures", "dist_impl.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    fig, axes = plt.subplots(1, len(_PANELS), figsize=(6 * len(_PANELS), 6))
    for ax, (col, title, xlabel, log) in zip(axes, _PANELS):
        if col not in df.columns or df[col].notna().sum() == 0:
            ax.set_visible(False)
            continue
        density_panel(ax, df, col, log=log)
        ax.set_title(title, fontsize=22, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel("Density", fontsize=18)
        ax.tick_params(axis="both", labelsize=14)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(DOMAIN_STYLE),
               fontsize=18, frameon=True, bbox_to_anchor=(0.5, 1.08))
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({df['power_w'].notna().sum()} impl designs)")
    return out


def main():
    ap = argparse.ArgumentParser(description="impl domain distribution plot (power + cycles)")
    ap.add_argument("--metrics", default=None, help="impl metrics.csv with power_w (default: results_impl/*.csv)")
    ap.add_argument("--feasible-only", action="store_true", help="drop designs with util > 100%%")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = io.read_metrics_csv(args.metrics) if args.metrics else None
    dist_impl(df=df, out=args.out, feasible=args.feasible_only)


if __name__ == "__main__":
    main()
