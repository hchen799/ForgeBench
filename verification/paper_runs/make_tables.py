"""Build the paper's verification tables (LaTeX) and the matching CSVs from the CSIM / full-model result folders.

    python verification/paper_runs/make_tables.py [--results verification/results_paper] [--fullmodel DIR] [--out verification/paper_runs]

Outputs (in --out):
  verif_ops.tex        tab:verif-ops-range  operators, columns A-D: largest faithful range r, max abs error and relative L2 error at r (N=100)
                       tab:verif-ops-pm1    operators, columns A-D: max abs and relative L2 error with every input in [-1, 1] (N=100)
  verif_modular.tex    tab:verif-mod-range / tab:verif-mod-pm1: the same for the Table 7 designs (18 generated, 8 hand-written), columns B-D
  verif_ops_paper.tex  tab:verification     the paper's operator table: one row per operator, max abs + relative L2 error at +-1, columns A-D
  verif_ops_paper_range.tex  tab:verification-range  per format the largest range r and the relative L2 error at r
  verif_ops_paper_both.tex   tab:verification-both   both: +-1 max abs + relative L2, and range r + relative L2 at r
  verif_cosim.tex      tab:verif-cosim      Vitis C/RTL co-simulation, one row per operator, columns C and D: variants passing, RTL != C elements, latency
  verif_catapult.tex   tab:verif-catapult   Catapult backend: gemm, vmm, mmv, dot (one row per operator), C sim (vs Vitis per seed)
  verif_fullmodel.tex  tab:verif-fullmodel  full-scale models at <32,10>: seeds, fixed-reference code mismatches, FP64 logit errors
  verif_ops.csv, verif_modular.csv, verif_cosim.csv, verif_catapult.csv, verif_fullmodel.csv   every number above, one row per (design, column, experiment)

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


def sci_e(x, digits=1):
    """Compact scientific notation for wide tables: 1.5e--3 (en dash: shorter than the math minus)."""
    if x is None:
        return "--"
    if x != x or x in (float("inf"), float("-inf")):
        return r"$\infty$"
    if x == 0:
        return "0"
    m, e = f"{x:.{digits}e}".split("e")
    return m if int(e) == 0 else f"{m}e" + (f"--{-int(e)}" if int(e) < 0 else str(int(e)))


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


LO_START = 0.0011        # range_search fixed_start (0.001) with float slack


def lo_note(groups):
    """Caption sentence for windows that exclude small inputs, naming the rows/columns."""
    hits = [f"\\texttt{{{esc(d.split('/', 1)[1])}}} ({COLS[c]}: $\\pm{lo:.3g}$--$\\pm{hi:.3g}$)" for _, rows in groups for name, by in rows
            for c, a in by.items() if a for d, lo, hi in a.get("lo") or []]
    return (r" $^{\S}$: faithful only above a lower edge too: " + "; ".join(hits) + ".") if hits else ""


def agg(rows):
    """-> dict(range, max_abs, rel_l2, all_within, n) over variants (range min, errors max)."""
    ok = [r for r in rows if r.get("status") == "ok"]
    rs = [num(r["range_hi"]) for r in ok if num(r.get("range_hi")) is not None]
    ea = [num(r["max_abs_err"]) for r in ok if num(r.get("max_abs_err")) is not None]
    er = [x for x in (num(r.get("rel_l2")) for r in ok) if x is not None and x == x and x != float("inf")]   # inf: all-zero golden (e.g. thresholded_relu at +-1)
    within = all(num(r.get("trials_within_bound")) == num(r.get("n_trials")) for r in ok)
    # a window that does not reach down to small inputs (e.g. <16,5> softmax over 32 entries: near-equal inputs make the exponential sum ~32,
    # past the format's +-16) -- its lower edge, else None
    lo = [(r["design"], num(r["range_lo"]), num(r["range_hi"])) for r in ok if (num(r.get("range_lo")) or 0) > LO_START]
    return {"range": min(rs) if rs else None, "lo": lo, "max_abs": max(ea) if ea else None, "rel_l2": max(er) if er else None,
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
                    cells += [rng_text(a["range"]) + (r"$^{*}$" if a["missing"] else "") + (r"$^{\S}$" if a.get("lo") else ""), sci(a["max_abs"]), sci(a["rel_l2"])]
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
                    "operator", "datatype", "status", "range_lo", "range_hi", "first_fail_range", "n_trials", "trials_within_bound", "max_abs_err",
                    "rel_l2", "rmse", "sqnr_db", "tol_abs", "notes")}})
    names = sorted({op for d in data.values() for op in d})
    rng = [(None, [(op, {c: agg(data[(c, "max_range")][op]) for c in cols if op in data[(c, "max_range")]}) for op in names])]
    pm1 = [(None, [(op, {c: agg(data[(c, "pm1")][op]) for c in cols if op in data[(c, "pm1")]}) for op in names])]
    cap_r = (r"Operator verification (C simulation, $N{=}100$ input samples per variant and range): the largest input range $\pm r$ over which "
             r"the operator stays within the format's error bound, and the maximum absolute and relative L2 error over the $N$ samples at "
             r"that range (range: minimum, errors: maximum over the operator's variants). Weights of matmul, MHA, SWA and conv are held at "
             r"$\pm0.1$, batchnorm parameters in $[0.25,1]$. Float: errors at $\pm1$; ``cap'': search ceiling. " + MIXED_NOTE + lo_note(rng))
    cap_p = (r"Operator verification with every input uniform in $[-1,1]$ (non-negative inputs: $[0,1]$), $N{=}100$ samples: maximum "
             r"absolute and relative L2 error (maximum over variants). $^{\dagger}$: some sample left the format's error bound. " + MIXED_NOTE)
    open(os.path.join(out, "verif_ops.tex"), "w").write(PRE + range_table("tab:verif-ops-range", cap_r, rng, cols) + "\n\n"
                                                         + pm1_table("tab:verif-ops-pm1", cap_p, pm1, cols) + "\n")
    write_csv(os.path.join(out, "verif_ops.csv"), recs)
    return len(names)


PAPER_OPS = [   # (domain, [(operator key, label, parameters)]) in the order of the paper's operator table
    ("GEMM", [("gemm", "GEMM", r"\{With Bias\}"), ("vmm", "Vec-Mtx Mult.", r"\{With Bias\}"), ("mmv", "Mtx-Vec Mult.", r"\{With Bias\}"),
              ("dot_product", "Dot Product", r"\{With Bias\}")]),
    ("DNN", [("conv", "Convolution", r"\{With Bias, With Grouping\}"), ("batchnorm", "BatchNorm", "---")]),
    ("LLM", [("mha", "Multi-head Attention", r"\{With Grouping, With RoPE\}"), ("swa", "Sliding-window Attn.", r"\{With RoPE\}"),
             ("matmul", "MatMul", "---"), ("layernorm", "LayerNorm", "---"), ("rmsnorm", "RMSNorm", "---")]),
    ("Helpers", [("activation", "Activation", r"\{Tensor Rank, Function\}"), ("matrix_add", "Matrix add", "---"),
                 ("elementwise_mult", "Element-wise Mult.", "---"), ("dropout", "Dropout", "---"), ("maxpool", "Max pool", "---"),
                 ("adaptive_avgpool", "Avg pool", "---")]),
]


OPS_PAPER_MODES = {   # mode -> (file, label, per-format sub-columns as (experiment, field))
    "pm1": ("verif_ops_paper.tex", "tab:verification", [("pm1", "abs"), ("pm1", "rel")]),
    "range": ("verif_ops_paper_range.tex", "tab:verification-range", [("max_range", "range"), ("max_range", "rel")]),
    "both": ("verif_ops_paper_both.tex", "tab:verification-both", [("pm1", "abs"), ("pm1", "rel"), ("max_range", "range"), ("max_range", "rel")]),
}
SHORT = {"gemm": "GEMM", "vmm": "Vec-Mtx", "mmv": "Mtx-Vec", "dot_product": "Dot Prod.", "conv": "Conv", "batchnorm": "BatchNorm",
         "mha": "MH Attn.", "swa": "SW Attn.", "matmul": "MatMul", "layernorm": "LayerNorm", "rmsnorm": "RMSNorm", "activation": "Act.",
         "matrix_add": "Mtx Add", "elementwise_mult": "Elem. Mult.", "dropout": "Dropout", "maxpool": "Max Pool", "adaptive_avgpool": "Avg Pool"}
ACT_FNS = {"relu": "ReLU", "relu6": "ReLU6", "leaky_relu": "Leaky ReLU", "prelu": "PReLU", "rrelu": "RReLU", "elu": "ELU", "selu": "SELU",
           "thresholded_relu": "Thresh.\\ ReLU", "gelu": "GELU", "swish": "Swish", "sigmoid": "Sigmoid", "hardsigmoid": "Hard sigmoid",
           "hardswish": "Hard swish", "tanh": "Tanh", "softmax": "Softmax"}   # compact table: one row per function, in this order
CAPTION_BOTH = (r"\newclaude{M1}{1}{C-simulation verification of the 57 operator variants in four data types (Sec.~\ref{sec:verification}). "
                r"$\pm1$: inputs uniform in $[-1,1]$. \emph{Max range}: largest faithful input range $\pm r$ (``cap'': search ceiling) and the "
                r"error at $r$. \emph{Abs.}: worst element-wise absolute error. \emph{Rel.}: worst relative L2 error "
                r"$\lVert\hat{y}-y\rVert_2/\lVert y\rVert_2$. $N$: variants per operator.")   # closing brace of \newclaude added in ops_paper
SUBHEAD = {("pm1", "abs"): r"\textbf{Max Abs.\ Error}", ("pm1", "rel"): r"\textbf{Rel.\ L2 Error}",
           ("max_range", "range"): r"\textbf{Range}", ("max_range", "rel"): r"\textbf{Rel.\ L2 Error}"}
SUBHEAD_BOTH = {("pm1", "abs"): r"\textbf{Abs.}", ("pm1", "rel"): r"\textbf{Rel.}",          # under a "+-1" / "Max range" header row
                ("max_range", "range"): r"$\boldsymbol{\pm r}$", ("max_range", "rel"): r"\textbf{Rel.}"}


def ops_paper(res, out, mode="pm1"):
    """The paper's operator table, one row per operator (maximum / minimum over its variants; Total = over all operators). Modes:
    pm1   -> verif_ops_paper.tex (tab:verification): per format max abs + relative L2 error with every input in [-1, 1] (__pm1 runs);
    range -> verif_ops_paper_range.tex (tab:verification-range): per format the largest faithful range +-r (minimum over variants) and the
             relative L2 error at r (__max_range runs; float: at +-1);
    both  -> verif_ops_paper_both.tex (tab:verification-both): the two side by side (4 columns per format, scaled to the text width)."""
    cols = list(COLS)
    fname, label, spec = OPS_PAPER_MODES[mode]
    exps = sorted({e for e, _ in spec})
    data = {(c, e): {} for c in cols for e in exps}
    for c in cols:
        for e in exps:
            for r in rows_of(res, f"{c}__{e}"):
                data[(c, e)].setdefault(r["operator"], []).append(r)
    known = {k for _, ops_ in PAPER_OPS for k, _, _ in ops_}
    missing = sorted({op for d in data.values() for op in d} - known)
    if missing:
        raise SystemExit(f"operators without a row in PAPER_OPS: {missing}")
    k = len(spec)
    compact = mode == "both"   # no Parameters column, short operator names, N = variants
    state = {"flag": False, "lo": []}

    def cells(c, pick, mark_lo=True):
        out_ = []
        for e, f in spec:
            rows = pick(data[(c, e)])
            a = agg(rows) if rows else None
            if a is None:
                out_.append("--")
                continue
            if f == "abs":
                state["flag"] |= not a["all_within"] and mark_lo
                out_.append(sci_e(a["max_abs"]) + ("" if a["all_within"] else r"$^{\dagger}$"))
            elif f == "rel":
                out_.append(sci_e(a["rel_l2"]))
            else:
                lo = bool(a.get("lo")) and mark_lo
                if lo:
                    state["lo"] += [(c, *h) for h in a["lo"]]
                out_.append(rng_text(a["range"]) + (r"$^{\S}$" if lo else ""))
        return out_

    def nvar(op):
        n = {len(data[(c, e)].get(op, [])) for c in cols for e in exps}
        return n.pop() if len(n) == 1 else "/".join(str(len(data[(c, exps[0])].get(op, []))) for c in cols)

    def act_fn(r):
        return r["design"].rsplit("__", 1)[1]

    def lines(op, lab):
        """-> [(row label, N, pick)]; compact: one row per activation function (its tensor-rank variants)."""
        if compact and op == "activation":
            fns = {act_fn(r) for c in cols for e in exps for r in data[(c, e)].get(op, [])}
            unknown = sorted(fns - set(ACT_FNS))
            if unknown:
                raise SystemExit(f"activation functions without a row in ACT_FNS: {unknown}")
            out_ = []
            for fn in (f for f in ACT_FNS if f in fns):
                pick = lambda d, fn=fn: [r for r in d.get(op, []) if act_fn(r) == fn]
                ns = {len(pick(data[(c, e)])) for c in cols for e in exps}
                out_.append((rf"Act.: {ACT_FNS[fn]}", ns.pop() if len(ns) == 1 else "/".join(str(len(pick(data[(c, exps[0])]))) for c in cols), pick))
            return out_
        n = nvar(op)
        n_txt = (r"$2{\times}15$" if compact else r"$2\times15$") if op == "activation" and n == 30 else n
        return [(SHORT[op] if compact else lab, n_txt if isinstance(n_txt, str) else n, lambda d, op=op: d.get(op, []))]

    body, total = [], 0
    for dom, ops_ in PAPER_OPS:
        body.append(r"\midrule")
        dom_lines = [(op, params, ln) for op, lab, params in ops_ for ln in lines(op, lab)]
        for i, (op, params, (lab, n, pick)) in enumerate(dom_lines):
            total += n if isinstance(n, int) else (nvar(op) if op == "activation" and not compact else 0)
            row = [x for c in cols for x in cells(c, pick)]
            first = rf"\multirow{{{len(dom_lines)}}}{{*}}{{{dom}}}" if i == 0 else ""
            body.append(f" {first} & {lab} & " + ("" if compact else f"{params} & ") + f"{n} & " + " & ".join(row) + r" \\")
    tot = [x for c in cols for x in cells(c, lambda d: [r for rows in d.values() for r in rows], mark_lo=False)]
    tot = [rf"\textbf{{{x}}}" if x != "--" else x for x in tot]
    n_ops = sum(len(o) for _, o in PAPER_OPS)
    body += [r"\midrule", rf"\textbf{{Total}} & \textbf{{{n_ops} " + ("ops" if compact else "operators") + "} & " + ("" if compact else "& ") + rf"\textbf{{{total}}} & " + " & ".join(tot) + r" \\"]

    consts = r"weights of GEMM-type, attention and convolution operators $\pm0.1$, batchnorm parameters in $[0.25,1]$"
    golden = (r"\new{3}{1}{Per-operator Vitis C-simulation verification against a NumPy float64 golden computed on the same inputs, "
              r"quantized to the design's format, so the error is the design's internal arithmetic only. ")
    ns = "100" if compact else "$N{=}100$"   # compact: N is the variant-count column
    rel = (r"\emph{Rel.}: relative L2 error " if compact else r"\emph{Rel.\ L2}: ") + r"$\lVert\hat{y}-y\rVert_2/\lVert y\rVert_2$ over all " + ns + " samples, worst variant. "
    absd = (r"\emph{Abs.}" if compact else r"\emph{Max Abs.\ Error}") + r": worst elementwise absolute error over all samples and variants. "
    pm1 = (r"$\pm1$: every input uniform in $[-1,1]$ (non-negative inputs: $[0,1]$; " + consts + r"), " + ns + " samples per variant. ")
    rng = (r"\emph{Range} $\pm r$: the largest input range (every input uniform in $[-r,r]$" + ("" if "pm1" in exps else "; " + consts) +
           r") over which every output stays within the format's error bound, found by a geometric search and confirmed on " + ns + " "
           r"samples (minimum over the operator's variants; ``cap'': the search ceiling, 741); its error is measured on " + ns + " fresh "
           r"samples at $r$ (float: at $\pm1$). ")
    bound = (r"Float outputs are within $r_\text{tol}=10^{-3}$, $a_\text{tol}=10^{-5}$ of the golden; fixed-point outputs within the format's "
             r"rounding-error bound" + (r" ($^{\dagger}$: some samples outside it)" if state["flag"] else "") + ". ")
    lo_txt = ""
    if state["lo"]:
        lo_txt = (r" $^{\S}$: faithful only above a lower edge too: " + "; ".join(
            f"\\texttt{{{esc(d.split('/', 1)[1])}}} ({COLS[c]}: $\\pm{lo:.3g}$--$\\pm{hi:.3g}$)" for c, d, lo, hi in state["lo"]) + ".")
    cap = golden + ("" if mode == "range" else pm1) + ("" if mode == "pm1" else rng) + ("" if mode == "range" else absd) + rel + bound + MIXED_NOTE + lo_txt + (r" $N$: variants per operator." if compact else "") + "}"
    if compact:   # the paper's caption (tab:verification-both), as given by the authors
        cap = CAPTION_BOTH + " " + MIXED_NOTE + "".join(x for x in (r" $^{\dagger}$: some samples outside the format's error bound." if state["flag"] else "", lo_txt) if x) + "}"
    sub = SUBHEAD_BOTH if mode == "both" else SUBHEAD
    c0 = 4 if compact else 5
    wide = True   # 12+ columns overflow a two-column page's \textwidth (measured: +-1 table 285 pt, range table 131 pt over)
    head = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{cap}}}", rf"\label{{{label}}}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
            r"\revfloat{"] + ([r"\resizebox{\textwidth}{!}{%"] if wide else []) + [
            (r"\begin{tabular}{llc" if compact else r"\begin{tabular}{L{1.1cm}L{2.6cm}lc") + "c" * k * len(cols) + "}", r"\toprule",
            (r"\textbf{Domain} & \textbf{Operator} & $\mathbf{N}$ & " if compact else
             r"\textbf{Domain} & \textbf{Operator} & \textbf{Parameters} & \textbf{Variants} & ")
            + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{{COLS[c]}}}" for c in cols) + r" \\",
            "".join(rf"\cmidrule(lr){{{c0 + k * i}-{c0 - 1 + k * (i + 1)}}}" for i in range(len(cols)))]
    if compact:   # +-1 (abs, rel) | max range (r, rel) under every format
        head += [" & & & " + " & ".join(r"\multicolumn{2}{c}{$\pm1$} & \multicolumn{2}{c}{max range}" for _ in cols) + r" \\",
                 "".join(rf"\cmidrule(lr){{{c0 + 2 * i}-{c0 + 2 * i + 1}}}" for i in range(2 * len(cols)))]
    head += [(" & & & " if compact else " & & & & ") + " & ".join(" & ".join(sub[x] for x in spec) for _ in cols) + r" \\"]
    foot = [r"\bottomrule", r"\end{tabular}"] + (["}"] if wide else []) + ["}",
            r"\footnotesize{\raggedright \emph{Activation} covers 15 supported functions (ReLU family, sigmoid, tanh, GELU, SiLU/Swish, softmax, "
            r"hard-sigmoid, hard-swish) across 2 primary tensor dimension variants in GEMM/DNN/LLM.}", r"\end{table*}"]
    needs = r"booktabs, multirow" + (", graphicx" if wide else "") + r", the paper's " + (r"\newclaude" if compact else r"\new") + r", \revfloat and L{} column type."
    open(os.path.join(out, fname), "w").write(PRE.replace("booktabs.", needs) + "\n".join(head + body + foot) + "\n")
    return total


# ----------------------------------------------------------------------------- co-simulation
COSIM_COLS = ["C_32_10", "D_float"]


def _cyc(x):
    x = int(x)
    return f"{x / 1e6:.1f}M" if x >= 1e6 else (f"{x / 1e3:.0f}k" if x >= 1e4 else str(x))


def cosim(res, out):
    """Vitis C/RTL co-simulation of every operator variant (__cosim runs): 10 looped input sets at +-1 per variant, RTL outputs compared
    with the C simulation element by element; latency from the RTL simulation. One row per operator (as the paper's operator table)."""
    data, recs = {}, []
    for c in COSIM_COLS:
        for r in rows_of(res, f"{c}__cosim"):
            data.setdefault(c, {}).setdefault(r["operator"], []).append(r)
            recs.append({"design": r["design"], "column": c, **{f: r.get(f) for f in (
                "operator", "datatype", "cosim_status", "cosim_trials", "cosim_c_vs_rtl_mismatches", "cosim_latency_min", "cosim_latency_avg",
                "cosim_latency_max", "notes")}})
    if not data:
        return 0

    def cells(rows):
        if not rows:
            return ["--"] * 3
        # Pass = completed with no C-vs-RTL mismatching element, so the mismatch count needs no column of its own
        ok = sum(r.get("cosim_status") == "ok" and not int(num(r.get("cosim_c_vs_rtl_mismatches")) or 0) for r in rows)
        lo = [num(r.get("cosim_latency_min")) for r in rows if num(r.get("cosim_latency_min")) is not None]
        hi = [num(r.get("cosim_latency_max")) for r in rows if num(r.get("cosim_latency_max")) is not None]
        return [f"{ok}/{len(rows)}", _cyc(min(lo)) if lo else "--", _cyc(max(hi)) if hi else "--"]

    body, n_all = [], 0
    for dom, ops_ in PAPER_OPS:
        body.append(r"\midrule")
        for i, (op, lab, params) in enumerate(ops_):
            n = max(len(data.get(c, {}).get(op, [])) for c in COSIM_COLS)
            n_all += n
            first = rf"\multirow{{{len(ops_)}}}{{*}}{{{dom}}}" if i == 0 else ""
            row = [x for c in COSIM_COLS for x in cells(data.get(c, {}).get(op, []))]
            body.append(f" {first} & {lab} & {params} & " + " & ".join(row) + r" \\")
    tot = [x for c in COSIM_COLS for x in cells([r for rows in data.get(c, {}).values() for r in rows])]
    body += [r"\midrule", r"\textbf{Total} & & & " + " & ".join(rf"\textbf{{{x}}}" for x in tot) + r" \\"]
    trials = sorted({r.get("cosim_trials") for c in data for rows in data[c].values() for r in rows})
    cap = (r"Vitis C/RTL co-simulation of every operator variant: the synthesized RTL simulated on " + "/".join(map(str, trials)) +
           r" input sets per variant (every input uniform in $[-1,1]$, the same designs and inputs as the $\pm1$ C simulation), outputs "
           r"compared element by element with the C simulation. \emph{Pass}: variants whose co-simulation completed with every output "
           r"element identical to the C simulation; \emph{Latency}: RTL-simulation latency in clock cycles (10\,ns), minimum and maximum over "
           r"the operator's variants and input sets.")
    T = [r"\begin{table}[t]", r"\centering", rf"\caption{{{cap}}}", r"\label{tab:verif-cosim}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lll" + "rrr" * len(COSIM_COLS) + "}", r"\toprule",
         r"\textbf{Domain} & \textbf{Operator} & \textbf{Parameters} & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{COLS[c]}}}" for c in COSIM_COLS) + r" \\",
         "".join(rf"\cmidrule(lr){{{4 + 3 * i}-{6 + 3 * i}}}" for i in range(len(COSIM_COLS))),
         " & & & " + " & ".join(r"& \multicolumn{2}{c}{\textbf{Latency}}" for _ in COSIM_COLS) + r" \\",
         "".join(rf"\cmidrule(lr){{{5 + 3 * i}-{6 + 3 * i}}}" for i in range(len(COSIM_COLS))),
         " & & & " + " & ".join(r"\textbf{Pass} & \textbf{Min} & \textbf{Max}" for _ in COSIM_COLS) + r" \\"] + body + \
        [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out, "verif_cosim.tex"), "w").write(PRE.replace("booktabs.", "booktabs, multirow.") + "\n".join(T) + "\n")
    write_csv(os.path.join(out, "verif_cosim.csv"), recs)
    return n_all


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
                    "operator", "datatype", "status", "range_lo", "range_hi", "first_fail_range", "n_trials", "trials_within_bound", "max_abs_err",
                    "rel_l2", "rmse", "sqnr_db", "tol_abs", "notes")}})
    designs = sorted({d for (_, _, d) in data})
    gen = [d for d in designs if not any(data.get((c, k, d), {}).get("operator") == "manual_design" for c in cols for k in ("max_range", "pm1"))]
    man = [d for d in designs if d not in gen]

    def grp(k):
        return [(t, [(d.split("/", 1)[1], {c: agg([data[(c, k, d)]]) for c in cols if (c, k, d) in data}) for d in ds])
                for t, ds in (("Generated programs (ForgeBench JSON)", gen), ("Hand-written designs", man)) if ds]
    note = (r"Per-design input constraints: weights $\pm\sqrt{3/\mathrm{fan\_in}}$ (gemm: the second operand), conv bias $\pm0.1$, "
            r"norm/batchnorm parameters $[0.25,1]$. conv\_block\_op1 (35--60\,min per fixed-point sample) uses $N{=}10$ and no range search.")
    cap_r = (r"Modularization designs (Table~\ref{tab:modular}), whole designs end to end, C simulation, $N{=}100$: largest faithful input "
             r"range $\pm r$ and maximum absolute / relative L2 error at $r$. " + note + " " + MIXED_NOTE.replace(r" (operators: \texttt{<24,8>} operator storage)", "")
             + lo_note(grp("max_range")))
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


def _trials(res, name, design, dt_tag):
    import gzip
    p = os.path.join(res, name, "trials", f"{design.split('/')[1]}__{design.split('/')[2]}__{dt_tag}.csv.gz")
    return list(csv.DictReader(gzip.open(p, "rt"))) if os.path.isfile(p) else None


def catapult(res, out):
    """Catapult backend check (Table-2 gemm, vmm, mmv, dot_product variants): SCVerify C sim at +-1 (N trials), compared per seed with the Vitis run of the
    same column. Synthesis results (reduced unroll) come from verification/catapult_synth.py, not this table."""
    recs, body = [], []
    for c in ("C_32_10", "D_float"):
        for r in sorted(rows_of(res, f"{c}__catapult_pm1"), key=lambda r: r["design"]):
            d = r["design"]
            tag = "float" if c == "D_float" else "fixed_32_10_trn_wrap"
            cat, vit = _trials(res, f"{c}__catapult_pm1", d, tag), _trials(res, f"{c}__pm1", d, tag)
            same = (sum(x["seed"] == y["seed"] and x["max_abs_err"] == y["max_abs_err"] and x["rmse"] == y["rmse"] for x, y in zip(cat, vit))
                    if cat and vit else None)
            rec = {"design": d, "column": c, "csim_status": r["status"], "csim_n": r["n_trials"], "csim_within": r["trials_within_bound"],
                   "max_abs_err": r["max_abs_err"], "rel_l2": r["rel_l2"], "same_as_vitis_per_seed": "" if same is None else same}
            recs.append(rec)
            body.append(f"\\texttt{{{esc(d.split('/', 2)[2])}}} & {COLS[c]} & {rec['csim_within']}/{rec['csim_n']} & {sci(num(rec['max_abs_err']))} & "
                        f"{sci(num(rec['rel_l2']))} & {'--' if same is None else same}/{rec['csim_n']} \\\\")
    if not recs:
        return 0
    write_csv(os.path.join(out, "verif_catapult.csv"), recs)
    # one row per operator (bias0 and bias1 together): worst Abs./Rel. over both variants; Eq. = every trial of both equals Vitis per seed
    body = []
    for op, label in (("gemm", "GEMM"), ("vmm", "Vec-Mtx Mult."), ("mmv", "Mtx-Vec Mult."), ("dot_product", "Dot Product")):
        cells = [label, r"\{With Bias\}"]
        for c in ("C_32_10", "D_float"):
            rs = [r for r in recs if r["column"] == c and r["design"].split("/")[2].rsplit("__", 1)[0] == op]
            if not rs:
                cells += ["--"] * 3
                continue
            eq = all(r["same_as_vitis_per_seed"] != "" and int(r["same_as_vitis_per_seed"]) == int(r["csim_n"]) for r in rs)
            ok = all(int(r["csim_within"]) == int(r["csim_n"]) for r in rs)
            cells += [sci_e(max(num(r["max_abs_err"]) for r in rs)) + ("" if ok else r"$^{\dagger}$"),
                      sci_e(max(num(r["rel_l2"]) for r in rs)), r"\cmark" if eq else r"\xmark"]
        body.append(" & ".join(cells) + r" \\")
    tex = ("% Generated from verification/paper_results/verif_catapult.csv (branch FPT_TRETS_2026, Catapult 2026.2). "
           "Needs: booktabs, \\cmark/\\xmark (e.g. pifont \\ding{51}/\\ding{55}).\n"
           r"\begin{table}[t]" "\n" r"\centering" "\n"
           r"\caption{Catapult C-SIM of GEMM operators. \emph{Abs.}: worst element-wise absolute error. \emph{Rel.}: worst relative L2 "
           r"error $\lVert\hat{y}-y\rVert_2/\lVert y\rVert_2$. $N$: variants per operator. \emph{Eq.} outputs are equal to Vitis counterpart.}" "\n"
           r"\label{tab:verif-catapult}" "\n" r"\footnotesize" "\n" r"\setlength{\tabcolsep}{3pt}" "\n"
           r"% \resizebox{\textwidth}{!}{%" "\n"
           r"\begin{tabular}{llrrrrrr}" "\n" r"\toprule" "\n"
           r"\textbf{Operator} & \textbf{Parameters} & \multicolumn{3}{c}{\texttt{<32,10>}} & \multicolumn{3}{c}{\texttt{float}} \\" "\n"
           r"& & \textbf{Abs.} & \textbf{Rel.} & \textbf{Eq.} & \textbf{Abs.} & \textbf{Rel.} & \textbf{Eq.} \\" "\n"
           r"\midrule" "\n" + "\n".join(body) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"% }" "\n" r"\end{table}" "\n")
    open(os.path.join(out, "verif_catapult.tex"), "w").write(tex)
    return len(recs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(HERE, "..", "results_paper"))
    ap.add_argument("--fullmodel", help="run_fullmodel.sh output folder")
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    print("operators:", ops(a.results, a.out))
    for mode in OPS_PAPER_MODES:
        print(f"operator table (paper format, {mode}), variants:", ops_paper(a.results, a.out, mode))
    print("modular designs:", modular(a.results, a.out))
    print("co-simulation variants:", cosim(a.results, a.out))
    print("catapult:", catapult(a.results, a.out))
    if a.fullmodel:
        print("full-scale models:", fullmodel(a.fullmodel, a.out))


if __name__ == "__main__":
    main()
