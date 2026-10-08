"""Build the paper's verification tables (LaTeX) and the matching CSVs from the CSIM / full-model result folders.

    python verification/paper_runs/make_tables.py [--results verification/results_paper] [--fullmodel DIR] [--out verification/paper_runs]

Outputs (in --out):
  verif_ops.tex        tab:verif-ops-range  operators, columns A-D: largest faithful range r, max abs error and relative L2 error at r (N=100)
                       tab:verif-ops-pm1    operators, columns A-D: max abs and relative L2 error with every input in [-1, 1] (N=100)
  verif_modular.tex    tab:verif-mod-range / tab:verif-mod-pm1: the same for the Table 7 designs (18 generated, 8 hand-written), columns B-D
  verif_fullmodel.tex  tab:verif-fullmodel  full-scale models at <32,10>: seeds, fixed-reference code mismatches, FP64 logit errors
  verif_ops.csv, verif_modular.csv, verif_fullmodel.csv   every number above, one row per (design, column, experiment)

Aggregation over an operator's variants: range = minimum (conservative), errors = maximum. Relative L2 = ||out - golden|| / ||golden||
over all N trials. Float is characterised, not bounded: its range is the span over which it stays within the float tolerance (search
ceiling shown as ``cap''); its errors in the range table are measured at the design's nominal range (+-1).
"""
import argparse
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = {"A_16_5": r"\texttt{<16,5>}",
        "B_16_5_op24_8_acc32_10": r"mixed$^{\ddagger}$",
        "C_32_10": r"\texttt{<32,10>}",
        "D_float": "float"}
MIXED_NOTE = r"$^{\ddagger}$\texttt{<16,5>} data, \texttt{<32,10>} RND accumulator (operators: \texttt{<24,8>} operator storage)."
CAP = 600.0          # search ceilings (741.45) are shown as "cap"


# ----------------------------------------------------------------------------- helpers
def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def sci(x, digits=1):
    if x is None:
        return "--"
    if x != x or x in (float("inf"), float("-inf")):
        return r"$\infty$"
    if x == 0:
        return "$0$"
    m, e = f"{x:.{digits}e}".split("e")
    return f"${m}$" if int(e) == 0 else rf"${m}\!\times\!10^{{{int(e)}}}$"


def rng_text(r):
    if r is None:
        return "--"
    return "cap" if r >= CAP else (f"{r:.0f}" if r >= 100 else f"{r:.2g}")


def esc(s):
    return s.replace("_", r"\_")


def rows_of(res, name):
    """summary.csv of a run, else its per-variant rows/ (a run still in progress)."""
    p = os.path.join(res, name, "summary.csv")
    if os.path.isfile(p):
        return list(csv.DictReader(open(p)))
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(res, name, "rows", "*.json")))]


def agg(rows):
    """-> dict(range, max_abs, rel_l2, all_within, n) over variants (range min, errors max)."""
    ok = [r for r in rows if r.get("status") == "ok"]
    rs = [num(r["range_hi"]) for r in ok if num(r.get("range_hi")) is not None]
    ea = [num(r["max_abs_err"]) for r in ok if num(r.get("max_abs_err")) is not None]
    er = [x for x in (num(r.get("rel_l2")) for r in ok) if x is not None and x == x and x != float("inf")]   # inf: all-zero golden (e.g. thresholded_relu at +-1)
    within = all(num(r.get("trials_within_bound")) == num(r.get("n_trials")) for r in ok)
    return {"range": min(rs) if rs else None, "max_abs": max(ea) if ea else None, "rel_l2": max(er) if er else None,
            "all_within": within and len(ok) == len(rows), "n": min(int(num(r["n_trials"])) for r in ok) if ok else None,
            "missing": len(ok) < len(rows)}


def write_csv(path, recs):
    keys = sorted({k for r in recs for k in r}, key=lambda k: (k not in ("design", "column", "experiment"), k))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(recs)


def range_table(label, caption, groups, cols, wide=True):
    """groups: [(group title or None, [(row label, {col: agg_range})])]"""
    env = "table*" if wide else "table"
    L = [rf"\begin{{{env}}}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         r"\begin{tabular}{l" + "rrr" * len(cols) + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{COLS[c]}}}" for c in cols) + r" \\",
         "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(cols))),
         " & " + " & ".join(r"$r$ & max abs & rel.\ L2" for _ in cols) + r" \\", r"\midrule"]
    for title, rows in groups:
        if title:
            L.append(rf"\multicolumn{{{1 + 3 * len(cols)}}}{{l}}{{\textit{{{title}}}}} \\")
        for name, by in rows:
            cells = []
            for c in cols:
                a = by.get(c)
                if a is None:
                    cells += ["--"] * 3
                else:
                    cells += [rng_text(a["range"]) + (r"$^{*}$" if a["missing"] else ""), sci(a["max_abs"]), sci(a["rel_l2"])]
            L.append(esc(name) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    return "\n".join(L)


def pm1_table(label, caption, groups, cols):
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         r"\begin{tabular}{l" + "rr" * len(cols) + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{COLS[c]}}}" for c in cols) + r" \\",
         "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(cols))),
         " & " + " & ".join(r"max abs & rel.\ L2" for _ in cols) + r" \\", r"\midrule"]
    for title, rows in groups:
        if title:
            L.append(rf"\multicolumn{{{1 + 2 * len(cols)}}}{{l}}{{\textit{{{title}}}}} \\")
        for name, by in rows:
            cells = []
            for c in cols:
                a = by.get(c)
                if a is None:
                    cells += ["--"] * 2
                else:
                    cells += [sci(a["max_abs"]) + ("" if a["all_within"] else r"$^{\dagger}$"), sci(a["rel_l2"])]
            L.append(esc(name) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


PRE = "% Auto-generated by verification/paper_runs/make_tables.py -- do not edit by hand. Needs: booktabs.\n"


# ----------------------------------------------------------------------------- operators
def ops(res, out):
    cols = list(COLS)
    data = {(c, k): {} for c in cols for k in ("max_range", "pm1")}
    recs = []
    for c in cols:
        for k in ("max_range", "pm1"):
            for r in rows_of(res, f"{c}__{k}"):
                data[(c, k)].setdefault(r["operator"], []).append(r)
                recs.append({"design": r["design"], "column": c, "experiment": k, **{f: r.get(f) for f in (
                    "operator", "datatype", "status", "range_hi", "first_fail_range", "n_trials", "trials_within_bound", "max_abs_err",
                    "rel_l2", "rmse", "sqnr_db", "tol_abs", "notes")}})
    names = sorted({op for d in data.values() for op in d})
    rng = [(None, [(op, {c: agg(data[(c, "max_range")][op]) for c in cols if op in data[(c, "max_range")]}) for op in names])]
    pm1 = [(None, [(op, {c: agg(data[(c, "pm1")][op]) for c in cols if op in data[(c, "pm1")]}) for op in names])]
    cap_r = (r"Operator verification (C simulation, $N{=}100$ input samples per variant and range): the largest input range $\pm r$ over which "
             r"the operator stays within the format's error bound, and the maximum absolute and relative L2 error over the $N$ samples at "
             r"that range (range: minimum, errors: maximum over the operator's variants). Weights of matmul, MHA, SWA and conv are held at "
             r"$\pm0.1$, batchnorm parameters in $[0.25,1]$. Float: errors at $\pm1$; ``cap'': search ceiling. " + MIXED_NOTE)
    cap_p = (r"Operator verification with every input uniform in $[-1,1]$ (non-negative inputs: $[0,1]$), $N{=}100$ samples: maximum "
             r"absolute and relative L2 error (maximum over variants). $^{\dagger}$: some sample left the format's error bound. " + MIXED_NOTE)
    open(os.path.join(out, "verif_ops.tex"), "w").write(PRE + range_table("tab:verif-ops-range", cap_r, rng, cols) + "\n\n"
                                                         + pm1_table("tab:verif-ops-pm1", cap_p, pm1, cols) + "\n")
    write_csv(os.path.join(out, "verif_ops.csv"), recs)
    return len(names)


# ----------------------------------------------------------------------------- modular (Table 7) designs
def modular(res, out):
    cols = ["B_16_5_op24_8_acc32_10", "C_32_10", "D_float"]
    data, recs = {}, []
    for c in cols:
        for k in ("max_range", "pm1"):
            rows = {}
            for name in (f"{c}__table7_{k}", f"{c}__manual_{k}", f"{c}__manual_{k}__expfix"):    # later runs replace earlier rows
                for r in rows_of(res, name):
                    rows[r["design"]] = r
            for d, r in rows.items():
                data[(c, k, d)] = r
                recs.append({"design": d, "column": c, "experiment": k, **{f: r.get(f) for f in (
                    "operator", "datatype", "status", "range_hi", "first_fail_range", "n_trials", "trials_within_bound", "max_abs_err",
                    "rel_l2", "rmse", "sqnr_db", "tol_abs", "notes")}})
    designs = sorted({d for (_, _, d) in data})
    gen = [d for d in designs if not any(data.get((c, k, d), {}).get("operator") == "manual_design" for c in cols for k in ("max_range", "pm1"))]
    man = [d for d in designs if d not in gen]

    def grp(k):
        return [(t, [(d.split("/", 1)[1], {c: agg([data[(c, k, d)]]) for c in cols if (c, k, d) in data}) for d in ds])
                for t, ds in (("Generated programs (ForgeBench JSON)", gen), ("Hand-written designs", man)) if ds]
    note = (r"Per-design input constraints: weights $\pm\sqrt{3/\mathrm{fan\_in}}$ (gemm: the second operand), conv bias $\pm0.1$, "
            r"norm/batchnorm parameters $[0.25,1]$. conv\_block\_op1 ($\approx$35\,min per fixed-point sample) uses $N{=}10$ and no range search.")
    cap_r = (r"Modularization designs (Table~\ref{tab:modular}), whole designs end to end, C simulation, $N{=}100$: largest faithful input "
             r"range $\pm r$ and maximum absolute / relative L2 error at $r$. " + note + " " + MIXED_NOTE.replace(r" (operators: \texttt{<24,8>} operator storage)", ""))
    cap_p = (r"Modularization designs with every input in $[-1,1]$, $N{=}100$: maximum absolute and relative L2 error. "
             r"$^{\dagger}$: some sample left the format's error bound. " + note)
    open(os.path.join(out, "verif_modular.tex"), "w").write(PRE + range_table("tab:verif-mod-range", cap_r, grp("max_range"), cols) + "\n\n"
                                                             + pm1_table("tab:verif-mod-pm1", cap_p, grp("pm1"), cols) + "\n")
    write_csv(os.path.join(out, "verif_modular.csv"), recs)
    return len(designs)


# ----------------------------------------------------------------------------- full-scale models
def fullmodel(src, out):
    """src: run_fullmodel.sh output (one folder per project x seed with summary.json)."""
    runs = {}
    for p in sorted(glob.glob(os.path.join(src, "*", "summary.json"))):
        s = json.load(open(p))
        name = os.path.basename(os.path.dirname(p))
        model = name.split("_config")[0].split("_32_10")[0] if "LLAMA" not in name else "LLAMA3_8B_ctx2048"
        runs.setdefault(model, []).append(s)
    recs, L = [], []
    for model, ss in sorted(runs.items(), key=lambda kv: (kv[0].startswith("LLAMA"), int("".join(ch for ch in kv[0].split("_")[0] if ch.isdigit()) or 0), kv[0])):
        logit = [c for s in ss for k, case in s["cases"].items() for t, c in case.items() if t == "logits"]
        tens = [c for s in ss for case in s["cases"].values() for c in case.values()]
        r = {"design": model, "column": "C_32_10", "experiment": "fullmodel", "seeds": len(ss), "pass": sum(s["status"] == "PASS" for s in ss),
             "mismatching_codes": sum(c["mismatching_codes"] for c in tens), "tensors_per_seed": len(tens) // max(len(ss), 1),
             "logits_max_abs": max(c["max_abs"] for c in logit), "logits_rel_l2": max(c["relative_l2"] for c in logit),
             "worst_tensor_rel_l2": max(c["relative_l2"] for c in tens), "seed_list": " ".join(str(s["seed"]) for s in ss)}
        recs.append(r)
        nm = (f"ResNet-{''.join(ch for ch in model.split('_')[0] if ch.isdigit())}" + (" (tiled)" if "TILED" in model else "")) if model.startswith("RESNET") \
            else "Llama-3-8B (ctx 2048)"
        L.append(f"{nm} & {r['seeds']} & {r['pass']}/{r['seeds']} & {r['mismatching_codes']} & {sci(r['logits_max_abs'])} & "
                 f"{sci(r['logits_rel_l2'])} & {sci(r['worst_tensor_rel_l2'])} \\\\")
    cap = (r"Full-scale models at \texttt{<32,10>} (round, saturate), generated from the design JSON and run in Vitis HLS C simulation. "
           r"Each seed draws a synthetic parameter set and input: ResNet input uniform $\pm1$ ($3{\times}224{\times}224$), weights "
           r"$\pm\sqrt{3/\mathrm{fan\_in}}$, batchnorm $\gamma,\sigma^2\in[0.9,1.1]$, $\beta,\mu\in\pm0.05$; Llama embedding $\pm0.5$, norm "
           r"weights $[0.9,1.1]$, 4 prefill tokens and 2 decode steps. Fixed-point reference: an independent integer-code PyTorch model "
           r"of the same arithmetic (mismatching codes over every observed tensor); FP64 reference: PyTorch float64 graph (pass: relative "
           r"L2 $\leq 0.05$ at every observed tensor). Errors: maximum over seeds (Llama: over prefill and decode calls).")
    T = [r"\begin{table*}[t]", r"\centering", r"\small", rf"\caption{{{cap}}}", r"\label{tab:verif-fullmodel}",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Model & seeds & pass & code mismatches & logits max abs & logits rel.\ L2 & worst tensor rel.\ L2 \\", r"\midrule"] + L + \
        [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(out, "verif_fullmodel.tex"), "w").write(PRE + "\n".join(T) + "\n")
    write_csv(os.path.join(out, "verif_fullmodel.csv"), recs)
    return len(recs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(HERE, "..", "results_paper"))
    ap.add_argument("--fullmodel", help="run_fullmodel.sh output folder")
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    print("operators:", ops(a.results, a.out))
    print("modular designs:", modular(a.results, a.out))
    if a.fullmodel:
        print("full-scale models:", fullmodel(a.fullmodel, a.out))


if __name__ == "__main__":
    main()
