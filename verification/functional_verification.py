"""ForgeBench functional-verification engine (shared by every operator; settings come from JSON).

    python -m verification.functional_verification --config verification/paper_verif.json            # everything reported in the paper
    python -m verification.functional_verification --operators gemm vmm --dtypes float "fixed<16,5>" --n 50
    python verification/operators/gemm/functional_verification.py                                    # one operator, its verif_config.json

For each (operator variant x data type) the engine

  1. generates the design and builds it once with Vitis CSIM (trials then re-run the built `csim.exe`; ~0.2 s each);
  2. finds the operating window with `range_search` (unless an explicit input range is configured): the contiguous range of input
     magnitudes over which every output element, in M seeded input sets, is within the error bound and nothing crashes;
  3. validates the window's top with N seeded trials (stepping down until all N pass) and reports max/mean abs error, RMSE, SQNR, ...;
  4. optionally runs a stress range (a multiple of the format's representable range) and reports how the operator fails;
  5. optionally runs CO-SIM (RTL) over K looped input sets and checks RTL == C bit-for-bit.

The golden is NumPy float64 evaluated on the *quantized* inputs (what the design receives after its `(data_t)` input conversion), so the
error measured is the design's arithmetic, not input rounding. float rows get PASS/FAIL against rtol/atol; fixed-point rows get error
metrics and a "faithful" verdict against a bound derived from the format (0.5*LSB*(L+stages) for rounding, LSB*(L+stages) for truncation, L the
accumulation length, plus `stages` dependent roundings and `peak_fraction` of the output's peak) unless explicit tolerances are configured.

Config (all keys optional; later files/overrides win; `extends` takes a path relative to the config file):
{
  "extends": "paper_verif.json",
  "operators": "table2" | ["gemm", "vmm"] | ["ops/gemm/gemm__bias1"],
  "datatypes": ["float", "fixed<16,5>", "fixed<32,10>"],          # 'fixed<W,I[,trn|rnd,wrap|sat]>' or 'ap_fixed<...>' (see docs/DATA_TYPES.md)
  "sim": ["csim", "cosim"],
  "n_trials": 100, "seed_base": 42, "cosim_trials": 20,
  "golden": {"precision": "float64"},
  "inputs": {"range": "auto" | "design" | [lo, hi] | {"DRAM_x": [lo, hi]}, "static_dir": null},
  "tolerance": {"mode": "format_bound" | "explicit", "atol": 1e-5, "rtol": 1e-3, "atol_scale": 5e-5, "stages": 4, "peak_fraction": 0.01},
  "range_search": {"enabled": true, "samples": 20, "factor": 1.4142, "float_span": [1e-3, 1e3], "fixed_start": 1e-3, "fixed_stop_hi_multiple": 2,
                   "bisect_steps": 4, "step_down": 1.19, "max_step_downs": 8},
  "stress": {"enabled": true, "factor": 4, "trials": 20}
}
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

from verification import fp, layout
from verification.csim_runner import BASE_SEED, csim_build_dir, refresh_inputs, rerun_csim_exe, run_vitis
from verification.fixedpoint import DType, error_metrics, quantize
from verification.fixed_runner import accum_length, op_under_test, read_outputs
from verification.golden_ref import compute_goldens
from verification.prepare_designs import generate_design
from verification.range_search import confirm_and_step_down, find_window, geometric_grid

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

BUILTIN_DEFAULTS = {
    "operators": "table2",
    "datatypes": ["float", "fixed<16,5>", "fixed<32,10>"],
    "sim": ["csim"],
    "n_trials": 100, "seed_base": BASE_SEED, "cosim_trials": 20,
    "golden": {"precision": "float64"},
    "inputs": {"range": "auto", "static_dir": None},
    "tolerance": {"mode": "format_bound", "atol": 1e-5, "rtol": 1e-3, "atol_scale": 5e-5, "stages": 4, "peak_fraction": 0.01},
    "range_search": {"enabled": True, "samples": 20, "factor": 2 ** 0.5, "float_span": [1e-3, 1e3], "fixed_start": 1e-3,
                     "fixed_stop_hi_multiple": 2, "bisect_steps": 4, "step_down": 1.19, "max_step_downs": 8},
    "stress": {"enabled": True, "factor": 4, "trials": 20},
}

SUMMARY_COLS = [
    "design", "domain", "operator", "variant", "datatype", "rounding_overflow", "sim_level", "status",
    "range_source", "range_lo", "range_hi", "first_fail_range", "failure_mode", "analytic_range", "run_range",
    "n_trials", "seed_base", "golden", "accum_len", "tolerance_kind", "tol_abs", "trials_within_bound", "n_elements",
    "max_abs_err", "mean_abs_err", "rmse", "sqnr_db", "rel_l2", "max_abs_err_raw",
    "stress_range", "stress_trials_within_bound", "stress_frac_outside", "stress_max_abs_err", "stress_crashes",
    "cosim_trials", "cosim_c_vs_rtl_mismatches", "cosim_latency_min", "cosim_latency_avg", "cosim_latency_max", "cosim_status",
    "total_s", "notes"]
_LOCK = threading.Lock()


# ----------------------------------------------------------------------------- config
def deep_merge(a, b):
    out = json.loads(json.dumps(a))
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path):
    cfg = {}
    if path:
        path = os.path.abspath(path)
        raw = json.load(open(path))
        if "extends" in raw:
            cfg = load_config(os.path.join(os.path.dirname(path), raw.pop("extends")))
        cfg = deep_merge(cfg, raw)
    return cfg


def resolve_config(path, overrides):
    cfg = deep_merge(BUILTIN_DEFAULTS, load_config(path))
    return deep_merge(cfg, overrides)


def select_variants(spec):
    """'table2' | list of operator names | list of ids -> layout variants."""
    allv = layout.variants()
    if spec == "all":
        return allv
    if spec == "table2":
        import csv as _csv
        with open(os.path.join(REPO_ROOT, "manifest", "designs", "ops.csv"), newline="") as f:
            ids = {r["design_id"] for r in _csv.DictReader(f) if r["in_table2"] == "YES"}
        return [v for v in allv if v["id"] in ids]
    spec = [spec] if isinstance(spec, str) else spec
    return [v for v in allv if v["id"] in spec or v["operator"] in spec or v["stem"] in spec]


# ----------------------------------------------------------------------------- bounds and ranges
def bound_for(cfg, dt, L):
    """-> (kind, fn(golden) -> per-element absolute tolerance array, scalar description)."""
    tol = cfg["tolerance"]
    if dt.kind == "float" or tol["mode"] == "explicit":
        def fn(g):
            return tol["atol"] + tol["atol_scale"] * float(np.max(np.abs(g))) + tol["rtol"] * np.abs(g)
        return ("explicit(atol,rtol)" if dt.kind == "fixed" else "float(rtol,atol_scale)"), fn, tol["rtol"]
    # operating-window criterion: rounding of an L-term accumulation plus `stages` further dependent roundings, plus a fraction of the output's
    # peak magnitude (covers division/library-math amplification). Overflow and garbage are orders of magnitude outside it.
    t = (0.5 if dt.q == "AP_RND" else 1.0) * dt.lsb * (L + tol["stages"])
    pf = tol["peak_fraction"]
    return (f"format_bound(c*LSB*(L+{tol['stages']})+{pf}*peak)", (lambda g: t + pf * float(np.max(np.abs(g)))), t)


def float_bound_no_atol(cfg):
    """For sweeps over tiny inputs the fixed atol would swamp the signal: keep only the relative and peak-scaled terms."""
    tol = cfg["tolerance"]
    return lambda g: tol["atol_scale"] * float(np.max(np.abs(g))) + tol["rtol"] * np.abs(g)


def ranged_config(config, mag):
    """Same design, inputs spanning +-mag (or [0, mag] where a DRAM's configured range is non-negative)."""
    c = json.loads(json.dumps(config))
    base = c.get("input_range", [0.0, 1.0])
    c["input_range"] = [0.0, mag] if base[0] >= 0 else [-mag, mag]
    for d in c["drams"]:
        r = d.get("input_range")
        if r is not None:
            d["input_range"] = [0.0, mag] if r[0] >= 0 else [-mag, mag]
    return c


def explicit_config(config, spec):
    c = json.loads(json.dumps(config))
    if isinstance(spec, list):
        c["input_range"] = list(spec)
        for d in c["drams"]:
            d.pop("input_range", None)
    elif isinstance(spec, dict):
        for d in c["drams"]:
            if d["name"] in spec:
                d["input_range"] = list(spec[d["name"]])
    return c


def analytic_range(op_func, dims, dt):
    """Rule-of-thumb ceiling for cross-checking the measured window (fixed-point only)."""
    if dt.kind == "float":
        return ""
    hi = 2.0 ** (dt.I - 1)
    L = accum_length(op_func, dims)
    if op_func in ("activation",):
        return round(0.9 * math.log(hi), 4)
    if op_func in ("maxpool", "matrix_add", "dropout", "elementwise_mult", "adaptive_avgpool"):
        return round(hi / 2, 4)
    return round(math.sqrt(0.5 * hi / max(L, 1)), 4)


# ----------------------------------------------------------------------------- one design x dtype
class Harness:
    """A built CSIM design that can be re-run on new input sets."""

    def __init__(self, domain, cfg_path, dt, work, seed_base):
        self.domain, self.dt, self.seed_base = domain, dt, seed_base
        self.run_dir, self.config = generate_design(domain, cfg_path, work, ["csim"], data_type=dt.text)
        rc, _ = run_vitis(self.run_dir, log_name="vitis_trial0.log")
        self.build = csim_build_dir(self.run_dir)
        if self.build is None:
            raise RuntimeError(f"CSIM build failed (rc={rc}); see {self.run_dir}/vitis_trial0.log")
        self.names = self.config["output_dram_names"]
        self.qf = (lambda x: quantize(x, dt))

    def run(self, cfg_ranged, seed, static_dir=None):
        """-> (out, golden_q, golden_raw) for one input set, or raises on a crash."""
        if static_dir:
            for d in self.config["drams"]:
                src = os.path.join(static_dir, f"{d['name']}.txt")
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(self.run_dir, f"{d['name']}.txt"))
        else:
            refresh_inputs(self.domain, cfg_ranged, self.run_dir, seed=seed)
        for n in self.names:                                    # a crashed run must not leave a stale output behind
            p = os.path.join(self.build, f"{n}_output.txt")
            if os.path.isfile(p):
                os.remove(p)
        rc, _ = rerun_csim_exe(self.run_dir, self.build)
        if rc != 0:
            raise CrashError(rc)
        out = read_outputs(self.build, self.names)
        g = compute_goldens(self.config, self.run_dir, self.domain, input_transform=self.qf)
        gq = np.concatenate([g[n].reshape(-1) for n in self.names])
        return out, gq

    def raw_golden(self):
        g = compute_goldens(self.config, self.run_dir, self.domain)
        return np.concatenate([g[n].reshape(-1) for n in self.names])

    def close(self, keep=False):
        if not keep:
            shutil.rmtree(self.run_dir, ignore_errors=True)


class CrashError(Exception):
    def __init__(self, rc):
        super().__init__(f"csim.exe exited with {rc}")
        self.rc = rc


def sample_ok(h, cfg_r, seed, boundfn):
    """One seeded input set at a range: (faithful?, max_abs, n_outside, failure_mode)."""
    try:
        out, g = h.run(cfg_r, seed)
    except CrashError as e:
        return False, float("inf"), -1, f"crash(rc={e.rc})"
    err = np.abs(out - g)
    tol = boundfn(g)
    outside = int(np.sum(err > tol))
    if not np.all(np.isfinite(out)):
        return False, float("inf"), outside, "non-finite output"
    return outside == 0, float(err.max()), outside, ("exceeds bound" if outside else "")


def verify_variant(v, dt, cfg, work, out_dir, keep=False):
    t0 = time.time()
    domain, stem, path = v["domain"], v["stem"], v["path"]
    row = {"design": v["id"], "domain": domain, "operator": v["operator"], "variant": "" if v["variant"] == "default" else v["variant"],
           "datatype": dt.text, "rounding_overflow": "" if dt.kind == "float" else f"{dt.q[3:]}/{dt.o[3:]}", "sim_level": "csim",
           "n_trials": cfg["n_trials"], "seed_base": cfg["seed_base"], "status": "ok", "notes": "",
           "golden": f"numpy {cfg['golden']['precision']} on " + ("quantized inputs" if dt.kind == "fixed" else "float32 inputs")}
    h = None
    try:
        base = os.path.join(work, f"{dt.tag()}")
        os.makedirs(base, exist_ok=True)
        h = Harness(domain, path, dt, base, cfg["seed_base"])
        op = op_under_test(h.config)
        L = accum_length(op["func_name"], op.get("dims", []))
        kind, boundfn, _ = bound_for(cfg, dt, L)
        row.update({"accum_len": L, "tolerance_kind": kind, "analytic_range": analytic_range(op["func_name"], op.get("dims", []), dt)})
        row["tol_abs"] = "" if dt.kind == "float" or cfg["tolerance"]["mode"] == "explicit" else f"{float(np.asarray(boundfn(np.zeros(1))).ravel()[0]):.6g}"
        rs, N = cfg["range_search"], cfg["n_trials"]
        static = cfg["inputs"].get("static_dir")
        rng_spec = cfg["inputs"]["range"]
        sweep_bound = float_bound_no_atol(cfg) if dt.kind == "float" else boundfn

        # ---- 1. input range ---------------------------------------------------------------------------------------------
        if static:
            run_cfg, row["range_source"], r_val = h.config, "static", None
            row["run_range"] = "static"
        elif rng_spec == "auto" and rs["enabled"]:
            if dt.kind == "float":
                grid = geometric_grid(rs["float_span"][0], rs["float_span"][1], rs["factor"])
            else:
                grid = geometric_grid(rs["fixed_start"], rs["fixed_stop_hi_multiple"] * 2.0 ** (dt.I - 1), rs["factor"])
            counter = [0]

            def faithful(r):
                cr = ranged_config(h.config, r)
                counter[0] += 1
                res = [sample_ok(h, cr, cfg["seed_base"] + 100000 + 977 * counter[0] + i, sweep_bound) for i in range(rs["samples"])]
                return all(x[0] for x in res)

            win = find_window(faithful, grid, bisect_steps=rs["bisect_steps"], stop_after_window=(dt.kind != "float"))
            row.update({"range_source": f"sweep({win['status']})", "range_lo": win["r_lo"], "range_hi": win["r_hi"],
                        "first_fail_range": win["first_fail_above"]})
            curve_path = os.path.join(out_dir, "sweep_curves")
            os.makedirs(curve_path, exist_ok=True)
            with open(os.path.join(curve_path, f"{domain}__{stem}__{dt.tag()}.csv"), "w") as f:
                f.write("range,faithful\n" + "\n".join(f"{r:.6g},{int(ok)}" for r, ok in win["curve"]) + "\n")
            if win["status"] == "none":
                row.update({"status": "no_faithful_range", "notes": "no grid point was faithful"})
                raise _Done()
            row["failure_mode"] = "" if win["first_fail_above"] is None else _mode(h, ranged_config(h.config, win["first_fail_above"]), cfg, sweep_bound)
            if dt.kind == "float":
                # float error scales with magnitude, so the headline is measured at the design's nominal range (the established
                # protocol); the sweep above documents the span over which the operator is faithful.
                run_cfg, r_val, row["run_range"] = h.config, None, "design"
            else:
                # ---- 2. validate the top of the window with N trials, stepping down on failure ---------------------------
                def confirm(r):
                    cr = ranged_config(h.config, r)
                    return all(sample_ok(h, cr, cfg["seed_base"] + i, sweep_bound)[0] for i in range(N))

                r_val, steps = confirm_and_step_down(confirm, win["r_hi"], rs["step_down"], rs["max_step_downs"])
                if r_val is None:
                    row.update({"status": "no_faithful_range", "notes": "window did not survive N-trial confirmation"})
                    raise _Done()
                row["range_hi"] = r_val
                row["run_range"] = f"+-{r_val:.6g}"
                run_cfg = ranged_config(h.config, r_val)
        elif isinstance(rng_spec, (list, dict)):
            run_cfg, row["range_source"], r_val = explicit_config(h.config, rng_spec), "explicit", None
            row["range_hi"] = row["run_range"] = json.dumps(rng_spec)
        else:
            run_cfg, row["range_source"], r_val = h.config, "design", None
            row["run_range"] = "design"

        # ---- 3. N trials at the validated range ---------------------------------------------------------------------------
        acc = {"max": 0.0, "abs_mean": 0.0, "mse": 0.0, "sig": 0.0, "err": 0.0, "raw_max": 0.0, "within": 0, "n": 0, "crash": 0}
        trials = []
        raw_ok = dt.kind == "fixed"
        N_eff = 1 if static else N
        for k in range(N_eff):
            seed = cfg["seed_base"] + k
            try:
                out, g = h.run(run_cfg, seed, static_dir=static)
            except CrashError:
                acc["crash"] += 1
                trials.append({"seed": seed, "max_abs_err": float("inf"), "within_bound": 0, "crash": 1})
                continue
            m = error_metrics(out, g, atol=0.0)
            within = int(np.all(np.abs(out - g) <= boundfn(g)))
            acc["within"] += within
            acc["max"] = max(acc["max"], m["max_abs_err"])
            acc["abs_mean"] += m["mean_abs_err"]
            acc["mse"] += m["rmse"] ** 2
            acc["err"] += float(((out - g) ** 2).sum())
            acc["sig"] += float((g * g).sum())
            acc["n"] = m["n_elements"]
            if raw_ok:
                acc["raw_max"] = max(acc["raw_max"], float(np.max(np.abs(out - h.raw_golden()))))
            trials.append({"seed": seed, "max_abs_err": m["max_abs_err"], "rmse": m["rmse"], "within_bound": within, "crash": 0})
        done = max(N_eff - acc["crash"], 1)
        row.update({"trials_within_bound": acc["within"], "n_elements": acc["n"], "max_abs_err": acc["max"],
                    "mean_abs_err": acc["abs_mean"] / done, "rmse": math.sqrt(acc["mse"] / done),
                    "sqnr_db": ("inf" if acc["err"] == 0 else 10 * math.log10(acc["sig"] / acc["err"]) if acc["sig"] > 0 else "-inf"),
                    "rel_l2": math.sqrt(acc["err"] / acc["sig"]) if acc["sig"] > 0 else "inf",
                    "max_abs_err_raw": acc["raw_max"] if raw_ok else ""})
        if acc["crash"]:
            row["notes"] += f"{acc['crash']} crashed trials; "
        tdir = os.path.join(out_dir, "trials")
        os.makedirs(tdir, exist_ok=True)
        with gzip.open(os.path.join(tdir, f"{domain}__{stem}__{dt.tag()}.csv.gz"), "wt") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for t in trials for k in t}), lineterminator="\n")
            w.writeheader()
            w.writerows(trials)

        # ---- 4. stress range (fixed-point only) ---------------------------------------------------------------------------
        st = cfg["stress"]
        if st["enabled"] and dt.kind == "fixed" and not static:
            hi = st["factor"] * 2.0 ** (dt.I - 1)
            cr = ranged_config(h.config, hi)
            ok_trials = crashes = outside = total = 0
            smax = 0.0
            for k in range(st["trials"]):
                good, mx, n_out, mode = sample_ok(h, cr, cfg["seed_base"] + 500000 + k, boundfn)
                if mode.startswith("crash"):
                    crashes += 1
                    continue
                ok_trials += int(good)
                outside += n_out
                total += acc["n"]
                smax = max(smax, mx)
            row.update({"stress_range": hi, "stress_trials_within_bound": ok_trials, "stress_frac_outside": (outside / total) if total else "",
                        "stress_max_abs_err": smax, "stress_crashes": crashes})

        # ---- 5. CO-SIM ----------------------------------------------------------------------------------------------------
        if "cosim" in cfg["sim"]:
            from verification.cosim import run_cosim
            ir = run_cfg.get("input_range")
            cs = run_cosim(domain, path, dt.text, cfg["cosim_trials"], os.path.join(work, f"{dt.tag()}_cosim"), input_range=ir, keep=keep)
            tr = cs["trials"]
            row.update({"cosim_status": cs["status"], "cosim_trials": len(tr),
                        "cosim_c_vs_rtl_mismatches": sum(t["c_vs_rtl_mismatch"] for t in tr)})
            if cs["latency"]:
                row.update(dict(zip(("cosim_latency_min", "cosim_latency_avg", "cosim_latency_max"), cs["latency"])))
            if cs["notes"]:
                row["notes"] += f"cosim: {cs['notes']}; "
    except _Done:
        pass
    except Exception as e:                      # one bad variant must not kill the campaign
        row["status"] = "error"
        row["notes"] += f"{type(e).__name__}: {e}"
    finally:
        if h:
            h.close(keep)
    row["total_s"] = f"{time.time() - t0:.1f}"
    with _LOCK:
        print(f"  {v['id']} [{dt.text}] {row['status']} range_hi={row.get('range_hi', '-')} max_abs={row.get('max_abs_err', '-')} "
              f"within={row.get('trials_within_bound', '-')}/{row['n_trials']} ({row['total_s']}s) {row['notes'][:80]}", flush=True)
    return row


class _Done(Exception):
    pass


def _mode(h, cfg_r, cfg, boundfn):
    """Failure mode just above the window (diagnostic text for the report)."""
    modes = {}
    for i in range(5):
        ok, _, n_out, mode = sample_ok(h, cfg_r, cfg["seed_base"] + 900000 + i, boundfn)
        if not ok:
            modes[mode.split("(")[0]] = modes.get(mode.split("(")[0], 0) + 1
    return max(modes, key=modes.get) if modes else ""


# ----------------------------------------------------------------------------- CLI
def main(default_config=None, default_operators=None, argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=default_config)
    ap.add_argument("--operators", nargs="*", default=default_operators)
    ap.add_argument("--dtypes", nargs="+")
    ap.add_argument("--n", type=int, dest="n_trials")
    ap.add_argument("--sim", nargs="+", choices=["csim", "cosim"])
    ap.add_argument("--range", dest="range_spec", help='"auto", "design", or JSON like [-1,1]')
    ap.add_argument("--no-stress", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    ov = {}
    if a.operators:
        ov["operators"] = a.operators
    if a.dtypes:
        ov["datatypes"] = a.dtypes
    if a.n_trials:
        ov["n_trials"] = a.n_trials
    if a.sim:
        ov["sim"] = a.sim
    if a.range_spec:
        ov["inputs"] = {"range": a.range_spec if a.range_spec in ("auto", "design") else json.loads(a.range_spec)}
    if a.no_stress:
        ov["stress"] = {"enabled": False}
    cfg = resolve_config(a.config, ov)
    out_dir = os.path.abspath(a.out)
    work = os.path.abspath(a.work or os.path.join(out_dir, "_work"))
    fp.set_precision(np.float64 if cfg["golden"]["precision"] == "float64" else np.float32)
    dts = [DType(t) for t in cfg["datatypes"]]
    variants = select_variants(cfg["operators"])
    jobs = [(v, dt) for v in variants for dt in dts]
    print(f"{len(jobs)} jobs ({len(variants)} variants x {len(dts)} dtypes); sim={cfg['sim']} n={cfg['n_trials']} "
          f"range={cfg['inputs']['range']} jobs={a.jobs}", flush=True)
    if a.dry_run:
        return
    os.makedirs(out_dir, exist_ok=True)
    json.dump(cfg, open(os.path.join(out_dir, "resolved_config.json"), "w"), indent=2)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(lambda j: verify_variant(j[0], j[1], cfg, work, out_dir, a.keep), jobs))
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"DONE {len(rows)} rows in {(time.time() - t0) / 60:.1f} min -> {os.path.join(out_dir, 'summary.csv')}; "
          f"{sum(1 for r in rows if r['status'] != 'ok')} not ok")
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
