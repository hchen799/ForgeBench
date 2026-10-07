"""Build the verification tables of the paper from verification/results_paper/<config>/summary.csv.

    python verification/paper_runs/make_tables.py [results_dir] [out.tex]

Two tables (one tabular each, rows = operators, aggregated over the operator's variants):
  tab:verif-range   per format column: largest faithful input range (conservative: minimum over the operator's variants) and the maximum
                    absolute error over N trials at that range (maximum over variants).
  tab:verif-pm1     per format column: maximum absolute error with every input in [-1, 1]; a dagger marks a column in which some trial
                    left the format's error bound (overflow / wrap).
Float is characterised, not bounded: its range is the span over which the operator stays within the float tolerance (search ceiling shown
as ``cap''), and its error is measured at the design's nominal input range in the first table.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results_paper")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "verif_ops.tex")

COLS = [("A_16_5", r"\texttt{<16,5>} all"),
        ("B_16_5_op24_8_acc32_10", r"\texttt{<16,5>} I/O, \texttt{<24,8>} op, \texttt{<32,10>} acc"),
        ("C_32_10", r"\texttt{<32,10>} all"),
        ("D_float", "float")]
CAP = 600.0     # search ceilings (741.45 in the float and wide fixed-point searches) are shown as "cap"


def esc(s):
    return s.replace("_", r"\_")


def load(col, kind):
    path = os.path.join(RES, f"{col}__{kind}", "summary.csv")
    if not os.path.isfile(path):
        return None
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(r["operator"], []).append(r)
    return out


def f(x, digits=2):
    if x is None:
        return "--"
    if x == 0:
        return "0"
    m, e = f"{x:.{digits}e}".split("e")
    if int(e) == 0:
        return m
    return rf"{m}\!\times\!10^{{{int(e)}}}"


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def agg(rows, kind):
    """-> (range_text, err, flag). range: min over variants (conservative); err: max over variants."""
    rng, err, bad, missing = [], [], False, False
    for r in rows:
        if r["status"] != "ok":
            missing = True
            continue
        e = num(r["max_abs_err"])
        if e is not None:
            err.append(e)
        hi = num(r["range_hi"])
        if hi is not None:
            rng.append(hi)
        w, n = num(r["trials_within_bound"]), num(r["n_trials"])
        if w is not None and n is not None and w < n:
            bad = True
    return (min(rng) if rng else None), (max(err) if err else None), bad, missing


def cell_range(rows, is_float):
    rng, err, bad, missing = agg(rows, "range")
    if rng is None:
        return "none", "--"
    rt = r"cap" if rng >= CAP else (f"{rng:.0f}" if rng >= 100 else f"{rng:.2g}")
    return rt + (r"$^{*}$" if missing else ""), f"${f(err)}$" if err is not None else "--"


def table_range(data, ops):
    cols = r"l" + "rr" * len(COLS)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Functional verification (C simulation, $N{=}100$ uniform input samples): the largest input range $\pm r$ over which "
             r"the operator stays within the format's error bound, and the maximum absolute error over the $N$ samples at that range. "
             r"Ranges are the minimum, and errors the maximum, over the operator's variants. Weights of matmul, MHA, SWA and conv are held "
             r"at $\pm0.1$. Float: error at the design's nominal range; ``cap'' is the search ceiling. $^{*}$: some variant had no faithful range.}",
             r"\label{tab:verif-range}", r"\begin{tabular}{" + cols + "}", r"\toprule"]
    lines.append("Operator & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{name}}}" for _, name in COLS) + r" \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
    lines.append(" & " + " & ".join(r"$r$ & max abs err" for _ in COLS) + r" \\")
    lines.append(r"\midrule")
    for op in ops:
        cells = []
        for col, _ in COLS:
            d = data[col]
            if d is None or op not in d:
                cells += ["--", "--"]
            else:
                cells += list(cell_range(d[op], col.startswith("D_")))
        lines.append(esc(op) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def table_pm1(data, ops):
    cols = "l" + "r" * len(COLS)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Maximum absolute error over $N{=}100$ samples with every input uniform in $[-1,1]$ (inputs whose configured range is "
             r"non-negative: $[0,1]$), for all four formats. $^{\dagger}$: at least one sample left the format's error bound (overflow or wrap) "
             r"for some variant.}",
             r"\label{tab:verif-pm1}", r"\begin{tabular}{" + cols + "}", r"\toprule",
             "Operator & " + " & ".join(name for _, name in COLS) + r" \\", r"\midrule"]
    for op in ops:
        cells = []
        for col, _ in COLS:
            d = data[col]
            if d is None or op not in d:
                cells.append("--")
                continue
            _, err, bad, missing = agg(d[op], "pm1")
            c = f"${f(err)}$" if err is not None else "--"
            cells.append(c + (r"$^{\dagger}$" if bad else ""))
        lines.append(esc(op) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    rng = {c: load(c, "max_range") for c, _ in COLS}
    pm1 = {c: load(c, "pm1") for c, _ in COLS}
    ops = sorted({op for d in list(rng.values()) + list(pm1.values()) if d for op in d})
    preamble = ("% Auto-generated by verification/paper_runs/make_tables.py -- do not edit by hand.\n"
                "% Needs: booktabs (\\toprule, \\midrule, \\cmidrule).\n")
    with open(OUT, "w") as fh:
        fh.write(preamble + table_range(rng, ops) + "\n\n" + table_pm1(pm1, ops) + "\n")
    print(f"wrote {OUT} ({len(ops)} operators)")


if __name__ == "__main__":
    main()
