"""Operator verification across data types: float pass/fail and fixed-point error characterization (Vitis CSIM).

For every (operator variant, data type): generate the design, run N seeded CSIM trials (trial k uses seed 42+k; trial 0 is a full
`vitis_hls` run that builds `csim.exe`, trials 1.. re-run that exe), and compare the design's output with a **float64 NumPy golden
evaluated on the quantized inputs** (the inputs after the design's own `(data_t)` conversion, so the measured error is the
design's internal arithmetic and not input rounding). A second column, `*_raw`, measures against the golden on the unquantized
inputs.

  float          pass/fail with the established tolerance (rtol 1e-3, atol 1e-5 + 5e-5*max|ref|, as in the generated testbench)
  fixed<W,I,..>  error metrics only -- never labelled PASS/FAIL against the float tolerance
                 (max/mean abs, RMSE, SQNR dB, relative L2, fraction within the worst-case rounding bound, overflow)

Overflow: each fixed-point design is also built as a "wide-integer twin" with identical fractional bits but 16 more integer bits.
Truncation/rounding are then identical and only overflow differs, so an output element that differs between the design and its twin
was affected by overflow/saturation somewhere upstream (exact; no instrumentation).

Within-tolerance fraction: tol = 0.5*LSB*(L+1) for rounding (LSB*(L+1) for truncation), L = accumulation length: the worst-case
rounding error of an L-term accumulation. It is a reported diagnostic, not a verdict; non-linear ops (exp, sqrt, ...) are not covered
by this bound. L comes from the op's dims (approximate for conv/attention; see accum_length).

  python -m verification.fixed_runner --table2 --n 100 --jobs 32 --out verification/results_fixed/pilot
  python -m verification.fixed_runner gemm --configs gemm__bias1 --dtypes float "fixed<16,5>" --n 20
  python -m verification.fixed_runner --table2 --stress --dtypes "fixed<16,5>"      # overflow-provoking input range
"""
import argparse
import csv
import gzip
import json
import math
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from verification import fp
from verification import layout
from verification.csim_runner import BASE_SEED, csim_build_dir, refresh_inputs, rerun_csim_exe, run_vitis
from verification.fixedpoint import DType, error_metrics, quantize
from verification.golden_ref import compute_goldens
from verification.prepare_designs import generate_design

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAINS = ("gemm", "conv", "llm")
DEFAULT_DTYPES = ["float", "fixed<16,5>", "fixed<32,10>"]
STRESS_FACTOR = 4          # stress inputs span 4x the representable range of the format
TWIN_EXTRA_INT_BITS = 16
# float verdict tolerance (same rule as the generated testbench)
RTOL, ATOL, ATOL_SCALE = 1e-3, 1e-5, 5e-5

SUMMARY_COLS = ["design", "domain", "operator", "variant", "datatype", "rounding_overflow", "stress", "input_range",
                "n_trials", "seed_base", "golden", "accum_len", "tol", "trials_pass", "max_abs_err", "mean_abs_err", "rmse",
                "sqnr_db", "rel_l2", "frac_within_tol", "max_abs_err_raw", "rmse_raw", "overflow_trials", "overflow_elements",
                "n_elements", "total_s", "notes"]

_PRINT = threading.Lock()


def accum_length(op_func, dims):
    """Approximate length of the longest accumulation inside one output element (diagnostic only)."""
    d = list(dims)
    try:
        if op_func == "gemm":
            return d[1]
        if op_func == "vmm":
            return d[0]
        if op_func == "mmv":
            return d[1]
        if op_func == "dot_product":
            return d[0]
        if op_func == "conv":
            return d[0] * d[6] * d[6]                  # Cin * K * K (grouped conv: an upper bound)
        if op_func == "matmul":
            return d[1]
        if op_func in ("mha", "swa"):
            return max(d[0], d[3])                     # context sum over seq, score sum over head_dim
        if op_func in ("layernorm", "rmsnorm"):
            return d[1]
        if op_func == "activation":
            return max(d[-2:]) if len(d) >= 2 else 1   # softmax reduces over a row
    except (IndexError, TypeError):
        pass
    return 1


def op_under_test(config):
    ops = [o for o in config["ops"].values() if o["func_name"] not in ("load", "store")]
    return ops[0]


def input_range_for(config, dt, stress):
    """Per-DRAM input_range overrides for this run (None = leave the config's ranges alone)."""
    if not stress or dt.kind == "float":
        return None
    hi = STRESS_FACTOR * 2.0 ** (dt.I - 1)
    base = config.get("input_range", [0.0, 1.0])
    out = {"input_range": [0.0, hi] if base[0] >= 0 else [-hi, hi]}
    drams = json.loads(json.dumps(config["drams"]))
    for d in drams:                                    # keep the sign convention of per-DRAM overrides (e.g. variances >= 0)
        r = d.get("input_range")
        if r is not None:
            d["input_range"] = [0.0, hi] if r[0] >= 0 else [-hi, hi]
    out["drams"] = drams
    return out


def twin_type(dt):
    return f"fixed<{dt.W + TWIN_EXTRA_INT_BITS},{dt.I + TWIN_EXTRA_INT_BITS},{dt.q[3:].lower()},{dt.o[3:].lower()}>"


def read_outputs(build, names):
    parts = []
    for n in names:
        p = os.path.join(build, f"{n}_output.txt")
        if not os.path.isfile(p):
            raise RuntimeError(f"no output file {n}_output.txt in the CSIM build dir")
        parts.append(np.loadtxt(p, dtype=np.float64).reshape(-1))
    return np.concatenate(parts)


def prepare(domain, cfg_path, base, dt_text, overrides):
    run_dir, config = generate_design(domain, cfg_path, base, ["csim"], data_type=dt_text, config_overrides=overrides)
    rc, out = run_vitis(run_dir, log_name="vitis_trial0.log")
    build = csim_build_dir(run_dir)
    if build is None:
        raise RuntimeError(f"CSIM build failed (rc={rc}); see {run_dir}/vitis_trial0.log")
    return run_dir, build, config


def run_job(job, a):
    domain, stem, cfg_path, dt = job["domain"], job["stem"], job["cfg_path"], job["dt"]
    t0 = time.time()
    base = os.path.join(a.work, f"{dt.tag()}{'_stress' if a.stress else ''}", domain)
    os.makedirs(base, exist_ok=True)
    cfg0 = json.load(open(cfg_path))
    overrides = input_range_for(cfg0, dt, a.stress)
    notes = []
    row = {"design": f"ops/{domain}/{stem}", "domain": domain, "datatype": dt.text, "stress": "YES" if a.stress else "NO",
           "rounding_overflow": "" if dt.kind == "float" else f"{dt.q[3:]}/{dt.o[3:]}", "n_trials": a.n, "seed_base": BASE_SEED,
           "golden": "numpy float64 on quantized inputs" if dt.kind == "fixed" else "numpy float64 on float32 inputs"}
    operator, _, variant = stem.partition("__")
    row.update({"operator": operator, "variant": variant})
    run_dir = twin_dir = None
    try:
        run_dir, build, config = prepare(domain, cfg_path, base, dt.text, overrides)
        twin = None
        if dt.kind == "fixed":
            twin_base = base + "_twin"
            os.makedirs(twin_base, exist_ok=True)
            twin_dir, twin_build, _ = prepare(domain, cfg_path, twin_base, twin_type(dt), overrides)
        op = op_under_test(config)
        L = accum_length(op["func_name"], op.get("dims", []))
        tol = (0.5 if (dt.kind == "fixed" and dt.q == "AP_RND") else 1.0) * dt.lsb * (L + 1) if dt.kind == "fixed" else 0.0
        row["accum_len"], row["tol"] = L, ("" if dt.kind == "float" else f"{tol:.6g}")
        row["input_range"] = json.dumps(config.get("input_range"))
        names = config["output_dram_names"]
        qf = (lambda x: quantize(x, dt))
        acc = {"abs_max": 0.0, "abs_mean": 0.0, "mse": 0.0, "sig": 0.0, "err": 0.0, "frac": 0.0, "raw_abs_max": 0.0, "raw_mse": 0.0,
               "ovf_trials": 0, "ovf_elems": 0, "pass": 0, "n": 0}
        trials = []
        for k in range(a.n):
            seed = BASE_SEED + k
            if k > 0:
                refresh_inputs(domain, config, run_dir, seed)
                rc, _ = rerun_csim_exe(run_dir, build)
                if rc != 0:
                    raise RuntimeError(f"csim.exe failed on trial {k} (rc={rc})")
            g_q = np.concatenate([a_.reshape(-1) for a_ in (compute_goldens(config, run_dir, domain, input_transform=qf)[n] for n in names)])
            g_raw = np.concatenate([a_.reshape(-1) for a_ in (compute_goldens(config, run_dir, domain)[n] for n in names)]) if dt.kind == "fixed" else g_q
            out = read_outputs(build, names)
            m = error_metrics(out, g_q, atol=tol if dt.kind == "fixed" else 0.0)
            mr = error_metrics(out, g_raw, atol=0.0)
            ovf = 0
            if dt.kind == "fixed":
                for fn in os.listdir(run_dir):                       # identical inputs for the twin
                    if fn.endswith(".txt"):
                        shutil.copy2(os.path.join(run_dir, fn), os.path.join(twin_dir, fn))
                rc, _ = rerun_csim_exe(twin_dir, twin_build)
                if rc != 0:
                    raise RuntimeError(f"twin csim.exe failed on trial {k} (rc={rc})")
                ovf = int(np.sum(out != read_outputs(twin_build, names)))
            if dt.kind == "float":
                thr = ATOL + ATOL_SCALE * float(np.max(np.abs(g_q))) + RTOL * np.abs(g_q)
                acc["pass"] += int(np.all(np.abs(out - g_q) <= thr))
            acc["abs_max"] = max(acc["abs_max"], m["max_abs_err"])
            acc["abs_mean"] += m["mean_abs_err"]
            acc["mse"] += m["rmse"] ** 2
            err2 = (out - g_q) ** 2
            acc["err"] += float(err2.sum())
            acc["sig"] += float(np.sum(g_q * g_q))
            acc["frac"] += m["frac_within_tol"]
            acc["raw_abs_max"] = max(acc["raw_abs_max"], mr["max_abs_err"])
            acc["raw_mse"] += mr["rmse"] ** 2
            acc["ovf_trials"] += int(ovf > 0)
            acc["ovf_elems"] += ovf
            acc["n"] = m["n_elements"]
            trials.append({"seed": seed, "max_abs_err": m["max_abs_err"], "rmse": m["rmse"], "sqnr_db": m["sqnr_db"],
                           "rel_l2": m["rel_l2"], "frac_within_tol": m["frac_within_tol"], "overflow_elements": ovf})
        n = a.n
        row.update({
            "trials_pass": acc["pass"] if dt.kind == "float" else "",
            "max_abs_err": acc["abs_max"], "mean_abs_err": acc["abs_mean"] / n, "rmse": math.sqrt(acc["mse"] / n),
            "sqnr_db": (float("inf") if acc["err"] == 0 else 10 * math.log10(acc["sig"] / acc["err"]) if acc["sig"] > 0 else float("-inf")),
            "rel_l2": math.sqrt(acc["err"] / acc["sig"]) if acc["sig"] > 0 else float("inf"),
            "frac_within_tol": "" if dt.kind == "float" else acc["frac"] / n,
            "max_abs_err_raw": "" if dt.kind == "float" else acc["raw_abs_max"], "rmse_raw": "" if dt.kind == "float" else math.sqrt(acc["raw_mse"] / n),
            "overflow_trials": "" if dt.kind == "float" else acc["ovf_trials"], "overflow_elements": "" if dt.kind == "float" else acc["ovf_elems"],
            "n_elements": acc["n"]})
        tdir = os.path.join(a.out, "trials", f"{dt.tag()}{'_stress' if a.stress else ''}")
        os.makedirs(tdir, exist_ok=True)
        with gzip.open(os.path.join(tdir, f"{domain}__{stem}.csv.gz"), "wt") as f:
            w = csv.DictWriter(f, fieldnames=list(trials[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(trials)
    except Exception as e:                                     # one bad variant must not kill the campaign
        notes.append(f"ERROR: {e}")
    finally:
        if not a.keep:
            for d in (run_dir, twin_dir):
                if d:
                    shutil.rmtree(d, ignore_errors=True)
    row["total_s"] = f"{time.time() - t0:.1f}"
    row["notes"] = "; ".join(notes)
    with _PRINT:
        verdict = (f"pass {row.get('trials_pass')}/{a.n}" if dt.kind == "float" else f"ovf_trials {row.get('overflow_trials')}")
        print(f"  {domain}/{stem} [{dt.text}] max_abs={row.get('max_abs_err', '-')} {verdict} ({row['total_s']}s) {row['notes']}", flush=True)
    return row


def table2_stems():
    p = os.path.join(REPO_ROOT, "manifest", "designs", "ops.csv")
    with open(p, newline="") as f:
        return {r["design_id"] for r in csv.DictReader(f) if r["in_table2"] == "YES"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("domains", nargs="*", default=list(DOMAINS))
    ap.add_argument("--dtypes", nargs="+", default=DEFAULT_DTYPES)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--configs", nargs="*", help="operator-variant stems (e.g. gemm__bias1); default all")
    ap.add_argument("--table2", action="store_true", help="only the 57 variants counted in the paper's Table 2")
    ap.add_argument("--stress", action="store_true", help="overflow-provoking input range (fixed-point types only)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", help="scratch for generated designs (default <out>/_work)")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    a.work = os.path.abspath(a.work or os.path.join(a.out, "_work"))
    fp.set_precision(np.float64)       # goldens are float64 for every data type (set once: it is process-global and jobs are threads)
    dts = [DType(t) for t in a.dtypes]
    keep = table2_stems() if a.table2 else None

    jobs = []
    for domain in a.domains:
        for v in layout.variants(domain):
            p, stem = v["path"], v["stem"]
            if a.configs and stem not in a.configs:
                continue
            if keep is not None and f"ops/{domain}/{stem}" not in keep:
                continue
            for dt in dts:
                if a.stress and dt.kind == "float":
                    continue
                jobs.append({"domain": domain, "stem": stem, "cfg_path": p, "dt": dt})
    print(f"{len(jobs)} jobs ({len({(j['domain'], j['stem']) for j in jobs})} variants x {len(dts)} dtypes), n={a.n}, jobs={a.jobs}", flush=True)
    if a.dry_run:
        return
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(lambda j: run_job(j, a), jobs))
    name = f"summary_n{a.n}{'_stress' if a.stress else ''}.csv"
    with open(os.path.join(a.out, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["notes"]]
    print(f"DONE {len(rows)} rows in {(time.time() - t0) / 60:.1f} min -> {os.path.join(a.out, name)}  ({len(bad)} with errors)")
    shutil.rmtree(a.work, ignore_errors=True) if not a.keep else None


if __name__ == "__main__":
    main()
