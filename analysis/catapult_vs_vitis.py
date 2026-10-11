"""Catapult vs Vitis synthesis comparison on a sample of the GEMM sweep, both in ap_fixed<32,10> (truncate + wrap), same FPGA part
(xczu9eg-ffvb1156-2-e), 10 ns.

Population: GEMM sweep designs (gemm/auto_generate_json.py) with at most 2 of the 3 loops unrolled (unroll_M/K/N = 8 on at most two),
2,688 of 3,072; a seeded random sample of 1,000 of them. Each sampled config is re-typed to ap_fixed<32,10> and synthesized by
  Vitis HLS 2024.1  csynth                     (analysis.sweep_runner --tool vitis)
  Catapult 2026.2   go extract, ZU9EG library  (analysis.sweep_runner --tool catapult)
Both report pre-implementation estimates (latency in cycles, LUT, DSP, FF / register bits); neither went through Vivado.

  python -m analysis.catapult_vs_vitis sample  --out _sweeps/catapult_vs_vitis       # configs/ + sample.csv
  bash analysis/run_catapult_vs_vitis.sh                                             # both sweeps (64 jobs total)
  python -m analysis.catapult_vs_vitis collect --out _sweeps/catapult_vs_vitis       # metrics.csv (one row per design x tool)
  python -m analysis.catapult_vs_vitis plot    --out _sweeps/catapult_vs_vitis       # Fig-8-style density plot
"""
import argparse
import csv
import json
import os
import random
import sys
import xml.etree.ElementTree as ET

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED, N = 20261011, 1000
DATA_TYPE = "ap_fixed<32,10>"
TOOLS = ("vitis", "catapult")
TOOL_STYLE = {"vitis": {"color": "#1CC5D6", "label": "Vitis HLS 2024.1"}, "catapult": {"color": "#E8635A", "label": "Catapult 2026.2"}}


def eligible():
    sys.path.insert(0, os.path.join(REPO_ROOT, "gemm"))
    import auto_generate_json as g
    return g, [p for p in g.iter_params() if sum(p[k] > 1 for k in ("unroll_M", "unroll_K", "unroll_N")) <= 2]


def sample(out):
    g, pool = eligible()
    pick = sorted(random.Random(SEED).sample(range(len(pool)), N))
    cdir = os.path.join(out, "configs")
    os.makedirs(cdir, exist_ok=True)
    rows = []
    for i in pick:
        p = pool[i]
        src = g.config_stem(p)
        with open(os.path.join(REPO_ROOT, "gemm", "auto_generated_configs", src + ".json")) as f:
            cfg = json.load(f)
        cfg["data_type"] = DATA_TYPE
        stem = src.replace("ap_fixed_16_5_", "ap_fixed_32_10_")
        with open(os.path.join(cdir, stem + ".json"), "w") as f:
            json.dump(cfg, f, indent=4)
        rows.append({"design": stem, "source_config": src, **{k: p[k] for k in ("M", "K", "N", "unroll_M", "unroll_K", "unroll_N",
                                                                               "comp_order", "gemm_order", "with_bias")}})
    with open(os.path.join(out, "sample.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(pool)} eligible, sampled {len(rows)} (seed {SEED}) -> {cdir}")


def _vitis(d):
    p = os.path.join(d, "project_1", "solution1", "syn", "report", "csynth.xml")
    if not os.path.isfile(p):
        return None
    r = ET.parse(p).getroot()
    est = r.find("AreaEstimates/Resources")
    perf = r.find("PerformanceEstimates/SummaryOfOverallLatency")
    num = lambda n, t: int(n.findtext(t)) if n is not None and (n.findtext(t) or "").isdigit() else None
    return {"cycles": num(perf, "Worst-caseLatency"), "lut": num(est, "LUT"), "ff": num(est, "FF"), "dsp": num(est, "DSP"),
            "bram": num(est, "BRAM_18K"), "est_clock_ns": float(r.findtext("PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod"))}


def _catapult(d):
    p = os.path.join(d, "rtl.rpt")
    if not os.path.isfile(p):
        return None
    q = catapult_qor_file(p)
    return {"cycles": q.get("latency_cycles"), "lut": q.get("lut"), "ff": q.get("reg_bits"), "dsp": q.get("dsp"), "bram": None,
            "est_clock_ns": 10.0 - q["slack_ns"] if "slack_ns" in q else None}


def catapult_qor_file(path):
    """verification.cosim.catapult_qor on a single rtl.rpt (lean layout keeps the report at the design root)."""
    import tempfile, shutil
    from verification.cosim import catapult_qor
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(os.path.join(t, "p", "s"))
        shutil.copy2(path, os.path.join(t, "p", "s", "rtl.rpt"))
        return catapult_qor(t)


def collect(out):
    with open(os.path.join(out, "sample.csv"), newline="") as f:
        designs = [r["design"] for r in csv.DictReader(f)]
    rows = []
    for tool, parse in (("vitis", _vitis), ("catapult", _catapult)):
        st = {}
        sp = os.path.join(out, f"lean_{tool}", "status.csv")
        if os.path.isfile(sp):
            with open(sp, newline="") as f:
                for r in csv.DictReader(f):
                    st[r["design"]] = r
        for d in designs:
            s = st.get(d, {})
            m = parse(os.path.join(out, f"lean_{tool}", d)) if s.get("status") == "ok" else None
            rows.append({"design": d, "tool": tool, "status": s.get("status", "not_run"), "seconds": s.get("seconds", ""),
                         **(m or {k: "" for k in ("cycles", "lut", "ff", "dsp", "bram", "est_clock_ns")})})
    with open(os.path.join(out, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for tool in TOOLS:
        c = {}
        for r in rows:
            if r["tool"] == tool:
                c[r["status"]] = c.get(r["status"], 0) + 1
        print(tool, c)


def plot(out, fig_out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from analysis.density import density_panel
    df = pd.read_csv(os.path.join(out, "metrics.csv"))
    ok = df[df.status == "ok"]
    both = set(ok[ok.tool == "vitis"].design) & set(ok[ok.tool == "catapult"].design)
    ok = ok[ok.design.isin(both)].rename(columns={"tool": "domain"})
    panels = [("cycles", "Latency", "Latency (Cycles)", True), ("lut", "LUT", "LUTs", True), ("dsp", "DSP", "DSPs", True)]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6))
    for ax, (col, title, xlabel, log) in zip(axes, panels):
        d = ok[ok[col] > 0] if log else ok
        density_panel(ax, d, col, log=log, domain_style=TOOL_STYLE)
        ax.set_title(title, fontsize=22, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel("Density", fontsize=18)
        ax.tick_params(axis="both", labelsize=14)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=18, frameon=True, bbox_to_anchor=(0.5, 1.08))
    plt.tight_layout()
    fig_out = fig_out or os.path.join(out, "catapult_vs_vitis.png")
    plt.savefig(fig_out, dpi=300, bbox_inches="tight")
    plt.savefig(os.path.splitext(fig_out)[0] + ".pdf", bbox_inches="tight")
    print(f"wrote {fig_out} ({len(both)} designs synthesized by both tools)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "collect", "plot"])
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "_sweeps", "catapult_vs_vitis"))
    ap.add_argument("--fig")
    a = ap.parse_args()
    {"sample": lambda: sample(a.out), "collect": lambda: collect(a.out), "plot": lambda: plot(a.out, a.fig)}[a.cmd]()


if __name__ == "__main__":
    main()
