"""Paper Table 3 (full-scale models): one row per synthesized project, C-synthesis resources and C-simulation error.

Designs: the 14 <32,10> round/saturate projects in scale_models/hls_files (ResNet-18/34/50/101/152 full and tiled; Llama-3-8B
prefill and decode at KV cache 2048 and 8192).

  Resources  LUT, DSP, BRAM_18K (count and % of the ZCU102) and latency (cycles) from csynth.xml:
             ResNets _sweeps/lean/fullmodel_csynth_r2_sched (new conv schedule), Llama _sweeps/lean/fullmodel_csynth_r2.
  C-sim      Vitis C simulation of the same sources (verify.py via verification/paper_runs/run_fullmodel.sh) against
             (a) the integer-code fixed-point reference: mismatching codes, summed over all seeds and observed tensors, and
             (b) the FP64 PyTorch model: final-logit max abs error and relative L2, min-max over seeds.
             ResNets: 10 seeds (results_paper/fullmodel_r2_sched). Llama: 3 seeds of 4 prefill + 2 decode tokens with the prefill and
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
        "llama": os.path.join(REPO, "_sweeps", "lean", "fullmodel_csynth_r2", "scale")}
VERIF = {"resnet": os.path.join(REPO, "verification", "results_paper", "fullmodel_r2_sched"),
         2048: os.path.join(REPO, "verification", "paper_results", "fullmodel"),
         8192: os.path.join(REPO, "verification", "results_paper", "fullmodel_ctx8192_r2")}
REPORT = "project_1/solution1/syn/report/csynth.xml"
LOGIT = re.compile(r"^\s*(\S+)/logits: .*mismatches=(\d+) .*max_abs=([0-9.e+-]+) relative_l2=([0-9.e+-]+)")
MISMATCH = re.compile(r"mismatches=(\d+)")

RESNETS = [f"RESNET{d}{t}_config_ap_fixed_32_10_" for d in (18, 34, 50, 101, 152) for t in ("", "_TILED")]
LLAMAS = [f"LLAMA3_8B_{p}_ctx{c}_config_ap_fixed_32_10_" for c in (2048, 8192) for p in ("PREFILL", "DECODE")]


def resources(project, family):
    a = ET.parse(os.path.join(LEAN[family], project, REPORT)).getroot()
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
        texts = summaries(os.path.join(VERIF["resnet"], f"{project}seed*"))
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


def label(project):
    m = re.match(r"RESNET(\d+)(_TILED)?_", project)
    if m:
        return f"ResNet-{m[1]}", "tiled" if m[2] else "full"
    m = re.match(r"LLAMA3_8B_(PREFILL|DECODE)_ctx(\d+)_", project)
    return f"Llama-3-8B ({m[2]})", m[1].lower()


def sci(x):
    mant, e = f"{x:.1e}".split("e")
    return rf"{mant}\!\times\!10^{{{int(e)}}}"


def rng(vals, f):
    lo, hi = min(vals), max(vals)
    return f"${f(lo)}$" if f(lo) == f(hi) else f"${f(lo)}$--${f(hi)}$"


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for p in RESNETS + LLAMAS:
        fam = "resnet" if p.startswith("RESNET") else "llama"
        r = {"project": p, **resources(p, fam)}
        r["model"], r["variant"] = label(p)
        n, ok, mism, errs = csim(p)
        r.update(seeds=n, csim_pass=ok, mismatching_codes=mism,
                 max_abs_min=min(e[0] for e in errs), max_abs_max=max(e[0] for e in errs),
                 rel_l2_min=min(e[1] for e in errs), rel_l2_max=max(e[1] for e in errs), errs=errs)
        rows.append(r)

    cols = ["model", "variant", "project", "lut", "lut_pct", "dsp", "dsp_pct", "bram", "bram_pct", "cycles", "seeds", "csim_pass",
            "mismatching_codes", "max_abs_min", "max_abs_max", "rel_l2_min", "rel_l2_max"]
    with open(os.path.join(OUT, "table3_fullmodel.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    pct = lambda n, p: f"{n:,} ({p:.0f}\\%)"
    lines = [
        "% generated by analysis/table3_fullmodel.py",
        r"\begin{tabular}{llrrrrccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Variant} & \textbf{LUT} & \textbf{DSP} & \textbf{BRAM} & \textbf{Latency (cycles)} & "
        r"\textbf{Mism. codes} & \textbf{Max abs. err.} & \textbf{Rel. $L_2$} \\",
        r"\midrule",
    ]
    prev = None
    for r in rows:
        if prev and prev.split(" ")[0] != r["model"].split(" ")[0]:
            lines.append(r"\midrule")
        model = r["model"] if r["model"] != prev else ""
        prev = r["model"]
        cyc = f"{r['cycles']:,}" if r["cycles"] is not None else "--"
        lines.append(" & ".join([model, r["variant"], pct(r["lut"], r["lut_pct"]), pct(r["dsp"], r["dsp_pct"]),
                                 pct(r["bram"], r["bram_pct"]), cyc, f"{r['mismatching_codes']:,}",
                                 rng([e[0] for e in r["errs"]], sci), rng([e[1] for e in r["errs"]], sci)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT, "table3_fullmodel.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}/table3_fullmodel.{{csv,tex}} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['model']:20s} {r['variant']:8s} LUT {r['lut']:>9,} ({r['lut_pct']:5.0f}%)  DSP {r['dsp']:>6,} ({r['dsp_pct']:4.0f}%)  "
              f"BRAM {r['bram']:>7,} ({r['bram_pct']:6.0f}%)  cyc {r['cycles']}  seeds {r['seeds']} pass {r['csim_pass']} "
              f"mism {r['mismatching_codes']}  maxabs {r['max_abs_min']:.1e}-{r['max_abs_max']:.1e}  relL2 {r['rel_l2_min']:.1e}-{r['rel_l2_max']:.1e}")


if __name__ == "__main__":
    main()
