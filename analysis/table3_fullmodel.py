"""Paper Table 3 (full-scale models): one row per synthesized project, C-synthesis resources and C-simulation error.

Designs: ResNet-18/34/50/101/152 full at <32,10> round/saturate and tiled at <16,5> truncate/wrap with an 8 x 32 conv tile; Llama-3-8B prefill and
decode at KV cache 2048 and 8192, <32,10> round/saturate. Rows grouped as in the paper (Full / Tiled / Prefill / Decode); errors are the
maximum over seeds.

  Resources  LUT, DSP, BRAM_18K (count and % of the ZCU102) and latency (cycles) from csynth.xml:
             ResNets _sweeps/lean/fullmodel_csynth_r2_sched (new conv schedule), Llama _sweeps/lean/fullmodel_csynth_r2.
  C-sim      Vitis C simulation of the same sources (verify.py via verification/paper_runs/run_fullmodel.sh) against
             (a) the integer-code fixed-point reference: mismatching codes, summed over all seeds and observed tensors, and
             (b) the FP64 PyTorch model: final-logit max abs error and relative L2, min-max over seeds.
             ResNets: 10 seeds (full: results_paper/fullmodel_r2_sched; tiled: results_paper/fullmodel_r4_16_5_trn). Llama: 3 seeds of 4 prefill + 2 decode tokens with the prefill and
             decode projects run together (paper_results/fullmodel ctx2048, results_paper/fullmodel_ctx8192_r2); the prefill row
             reports the prefill call, the decode row the two decode calls.

    python3 -m analysis.table3_fullmodel   -> analysis/tables_out/table3_fullmodel.{csv,tex}
"""
import csv
import glob
import os
import re
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "analysis", "tables_out")
LEAN = {"resnet": os.path.join(REPO, "_sweeps", "lean", "fullmodel_csynth_r2_sched", "scale"),
        "tiled": os.path.join(REPO, "_sweeps", "lean", "fullmodel_csynth_r4_16_5_trn", "scale"),
        "llama": os.path.join(REPO, "_sweeps", "lean", "fullmodel_csynth_r2", "scale")}
VERIF = {"resnet": os.path.join(REPO, "verification", "results_paper", "fullmodel_r2_sched"),
         "tiled": os.path.join(REPO, "verification", "results_paper", "fullmodel_r4_16_5_trn"),
         2048: os.path.join(REPO, "verification", "paper_results", "fullmodel"),
         8192: os.path.join(REPO, "verification", "results_paper", "fullmodel_ctx8192_r2")}
REPORT = "project_1/solution1/syn/report/csynth.xml"
LOGIT = re.compile(r"^\s*(\S+)/logits: .*mismatches=(\d+) .*max_abs=([0-9.e+-]+) relative_l2=([0-9.e+-]+)")
MISMATCH = re.compile(r"mismatches=(\d+)")

# row groups as in the paper: (variant label, format, projects)
GROUPS = [
    ("Full", "<32,10>", [f"RESNET{d}_config_ap_fixed_32_10_" for d in (18, 34, 50, 101, 152)]),
    ("Tiled", "<16,5>", [f"RESNET{d}_TILED_ci8_co32_config_ap_fixed_16_5_" for d in (18, 34, 50, 101, 152)]),
    ("Prefill", "<32,10>", [f"LLAMA3_8B_PREFILL_ctx{c}_config_ap_fixed_32_10_" for c in (2048, 8192)]),
    ("Decode", "<32,10>", [f"LLAMA3_8B_DECODE_ctx{c}_config_ap_fixed_32_10_" for c in (2048, 8192)]),
]


def family(project):
    return "llama" if project.startswith("LLAMA") else "tiled" if "_TILED" in project else "resnet"


def resources(project, family):
    path = os.path.join(LEAN[family], project, REPORT)
    if not os.path.isfile(path):
        return None
    a = ET.parse(path).getroot()
    area = a.find("AreaEstimates")
    row = {}
    for name, tag in (("lut", "LUT"), ("dsp", "DSP"), ("bram", "BRAM_18K")):
        used, avail = int(area.find(f"Resources/{tag}").text), int(area.find(f"AvailableResources/{tag}").text)
        row[name], row[f"{name}_pct"] = used, used / avail * 100
    lat = a.find("PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency").text
    row["cycles"] = int(lat) if lat and lat.isdigit() else None
    return row


def summaries(run_dir_glob):
    out = []
    for d in sorted(glob.glob(run_dir_glob)):
        s = os.path.join(d, "summary.txt")
        if os.path.isfile(s):
            out.append(open(s).read())
    return out


def csim(project):
    """(seeds, all PASS, mismatching codes, [(max_abs, rel_l2) per seed]) for one project."""
    if project.startswith("RESNET"):
        texts = summaries(os.path.join(VERIF[family(project)], f"{project}seed*"))
        calls = None
    else:
        ctx = int(re.search(r"ctx(\d+)", project).group(1))
        texts = summaries(os.path.join(VERIF[ctx], f"LLAMA3_8B_ctx{ctx}_32_10_seed*"))
        calls = {"call0000"} if "PREFILL" in project else {"call0001", "call0002"}
    errs, ok, mism = [], True, 0
    for t in texts:
        ok &= t.splitlines()[0].strip() == "PASS"
        mism += sum(int(m) for m in MISMATCH.findall(t))
        seed = [(float(m[3]), float(m[4])) for m in (LOGIT.match(l) for l in t.splitlines()) if m and (calls is None or m[1] in calls)]
        errs.append((max(e[0] for e in seed), max(e[1] for e in seed)))
    return len(texts), ok, mism, errs


def model(project):
    m = re.match(r"RESNET(\d+)", project)
    return f"ResNet-{m[1]}" if m else "Llama-3-8B (" + re.search(r"ctx(\d+)", project)[1] + ")"


def sci(x):
    mant, e = f"{x:.1e}".split("e")
    return rf"${mant}\!\times\!10^{{{int(e)}}}$"


def scaled(x, div):
    """Count in units of div: >= 10 as an integer with thousands separators, 1-10 with one decimal, below 1 with two."""
    if x is None:
        return "--"
    v = x / div
    return f"{v:,.0f}" if v >= 10 else f"{v:.1f}" if v >= 1 else f"{v:.2f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for variant, fmt, projects in GROUPS:
        for p in projects:
            r = {"project": p, "model": model(p), "variant": variant, "format": fmt}
            res = resources(p, family(p))
            r.update(res or {})
            n, ok, mism, errs = csim(p) if summaries(os.path.join(VERIF.get(family(p), ""), "*")) or p.startswith("LLAMA") else (0, None, None, [])
            r.update(seeds=n, csim_pass_all=ok, mismatching_codes=mism, errs=errs,
                     seeds_fp64_pass=None,
                     max_abs=max((e[0] for e in errs), default=None), rel_l2=max((e[1] for e in errs), default=None))
            rows.append(r)

    cols = ["model", "variant", "format", "project", "lut", "lut_pct", "dsp", "dsp_pct", "bram", "bram_pct", "cycles", "seeds",
            "csim_pass_all", "mismatching_codes", "max_abs", "rel_l2"]
    with open(os.path.join(OUT, "table3_fullmodel.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    lines = [
        "% generated by analysis/table3_fullmodel.py; errors are the maximum over seeds (final logits vs FP64)",
        r"\begin{tabular}{ll ccc c cc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Variant}} & ",
        r"\multirow{2}{*}{\textbf{LUT $\times 10^3$}} & \multirow{2}{*}{\textbf{DSP $\times 10^3$}} & \multirow{2}{*}{\textbf{BRAM $\times 10^3$}} & ",
        r"\textbf{Latency} &",
        r"\multirow{2}{*}{\textbf{Max abs. err.}} & ",
        r"\multirow{2}{*}{\textbf{Rel. $L_2$}} \\",
        r"&  & ",
        r"&  &  &",
        r"\textbf{(cycles $\times 10^6$)} & ",
        r"&  \\ ",
        r"\midrule",
    ]
    for gi, (variant, fmt, projects) in enumerate(GROUPS):
        if gi:
            lines.append(r"\midrule")
        group = [r for r in rows if r["variant"] == variant]
        for i, r in enumerate(group):
            var_cell = (r"\multirow{" + str(len(group)) + r"}{*}{\begin{tabular}[c]{@{}c@{}}" + variant + r"\\\texttt{" + fmt + r"}\end{tabular}}"
                        if i == 0 else "")
            cyc = scaled(r.get("cycles"), 1e6)
            lines.append(" & ".join([r["model"], var_cell, scaled(r.get("lut"), 1e3), scaled(r.get("dsp"), 1e3),
                                     scaled(r.get("bram"), 1e3), cyc,
                                     sci(r["max_abs"]) if r["max_abs"] is not None else "--",
                                     sci(r["rel_l2"]) if r["rel_l2"] is not None else "--"]) + r" \\")
            lines.append("")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT, "table3_fullmodel.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}/table3_fullmodel.{{csv,tex}} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['variant']:8s} {r['model']:18s} LUT {scaled(r.get('lut'),1e3):>7}K DSP {scaled(r.get('dsp'),1e3):>6}K "
              f"BRAM {scaled(r.get('bram'),1e3):>6}K cyc {scaled(r.get('cycles'),1e6):>6}M seeds {r['seeds']} all-pass {r['csim_pass_all']} "
              f"mism {r['mismatching_codes']} maxabs {r['max_abs']} relL2 {r['rel_l2']}")


if __name__ == "__main__":
    main()
