"""N-input CSIM runner for the ForgeBench correctness harness.

The Vitis-free proxy (`verification.verify_operators --n K`) already verifies each
operator over K randomized input trials. This module does the same thing through
**real Vitis HLS CSIM**: for every config it generates the float design once, then
runs K trials, each with a fresh input set (`random.seed(42 + k)`, the same seeding
`gcc_selfcheck` uses) and a freshly recomputed numpy golden, and records the
testbench's `VERIFICATION:` verdict per trial.

Two trial modes, because the cost profile differs by an order of magnitude:

  full  Every trial is a complete `vitis_hls -f run_hls.tcl` invocation. Tool
        startup + project open + clang rebuild are paid K times. Most faithful,
        most expensive.
  exe   Trial 0 is a complete `vitis_hls` invocation; trials 1..K-1 re-execute the
        `csim.exe` that Vitis just built (in its own csim build directory, with
        refreshed inputs/goldens copied in). Same binary, same compiler, same
        object code that CSIM runs -- only the tool wrapper is skipped. Startup
        and rebuild are paid once, so per-trial cost collapses to the actual
        simulation.

Design generation is deliberately serialized: `gen_configs.run_hls_flow` writes
`<DRAM>.txt` into the *current directory* (the domain dir) before moving them into
the run dir, so two concurrent generations in one domain would clobber each
other's input files. Only the Vitis runs are parallelized (`--jobs`).

CLI:
  python -m verification.csim_runner --n 1                     # ops suite, all domains
  python -m verification.csim_runner --n 10 --mode exe --jobs 32
  python -m verification.csim_runner --suite designs --n 1     # whole-design suite
  python -m verification.csim_runner gemm --n 5 --jobs 8
"""
import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from verification.golden_ref import compute_goldens
from verification.io import write_golden
from verification.prepare_designs import generate_design, register_goldens_in_tcl

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAINS = ("gemm", "conv", "llm")

# Trial k uses BASE_SEED + k, so k=0 reproduces the historical single-input run.
BASE_SEED = 42

_PASS_RE = re.compile(r"VERIFICATION:\s*PASS\s*\(max_abs=([^,]+),\s*max_rel=([^,]+),\s*n=(\d+)\)")
_FAIL_RE = re.compile(r"VERIFICATION:\s*FAIL\s*\(max_abs=([^,]+),\s*max_rel=([^,]+),\s*n_mismatch=(\d+)/(\d+)\)")
_SKIP_RE = re.compile(r"VERIFICATION:\s*SKIP")

_GEN_CACHE = {}
# Guards the process-global cwd flip in refresh_inputs (see its docstring).
_CHDIR_LOCK = threading.Lock()


def domain_generator(domain):
    """Load `<domain>/generate_code.py` under a domain-unique module name.

    `importlib.import_module("generate_code")` would cache whichever domain was
    imported first and silently hand it back for the others -- fine only because
    the three `generate_dram_txt_files` bodies happen to be identical today.
    """
    if domain not in _GEN_CACHE:
        path = os.path.join(REPO_ROOT, domain, "generate_code.py")
        spec = importlib.util.spec_from_file_location(f"_fb_generate_code_{domain}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _GEN_CACHE[domain] = module
    return _GEN_CACHE[domain]


def refresh_inputs(domain, config, run_dir, seed):
    """Rewrite the design's DRAM input .txt files with a new seed, in place.

    `generate_dram_txt_files` writes `<NAME>.txt` into the *current directory*, so
    it has to be called under a chdir. `os.chdir` is process-global, not
    per-thread, so with --jobs > 1 two workers would otherwise write their inputs
    into each other's design directory. Everything else in the trial path uses
    absolute paths (or an explicit subprocess `cwd=`) and is unaffected by the
    cwd flip, so serializing just this call is sufficient -- and it costs
    milliseconds next to a multi-second CSIM.
    """
    gen = domain_generator(domain)
    low, high = config.get("input_range", [0.0, 1.0])
    with _CHDIR_LOCK:
        cwd = os.getcwd()
        os.chdir(run_dir)
        try:
            gen.generate_dram_txt_files(config["drams"], seed=seed, low=low, high=high)
        finally:
            os.chdir(cwd)


def write_goldens(config, run_dir, domain):
    goldens = compute_goldens(config, run_dir, domain)
    for name, arr in goldens.items():
        write_golden(os.path.join(run_dir, f"{name}.golden.txt"), arr)
    return goldens


def parse_verdict(text):
    """-> dict(verdict, max_abs, max_rel, n_mismatch, n_total) from tb output."""
    m = _PASS_RE.search(text)
    if m:
        return {"verdict": "PASS", "max_abs": float(m.group(1)), "max_rel": float(m.group(2)),
                "n_mismatch": 0, "n_total": int(m.group(3))}
    m = _FAIL_RE.search(text)
    if m:
        return {"verdict": "FAIL", "max_abs": float(m.group(1)), "max_rel": float(m.group(2)),
                "n_mismatch": int(m.group(3)), "n_total": int(m.group(4))}
    if _SKIP_RE.search(text):
        return {"verdict": "SKIP", "max_abs": 0.0, "max_rel": 0.0, "n_mismatch": 0, "n_total": 0}
    return {"verdict": "NONE", "max_abs": 0.0, "max_rel": 0.0, "n_mismatch": 0, "n_total": 0}


def run_vitis(run_dir, log_name="vitis_hls_run.log"):
    """Full `vitis_hls -f run_hls.tcl` in run_dir. -> (rc, combined output)."""
    cp = subprocess.run(["vitis_hls", "-f", "run_hls.tcl"], cwd=run_dir,
                        capture_output=True, text=True)
    out = (cp.stdout or "") + (cp.stderr or "")
    with open(os.path.join(run_dir, log_name), "w") as f:
        f.write(out)
    return cp.returncode, out


def csim_build_dir(run_dir):
    """Directory holding the csim.exe Vitis built, or None if CSIM never ran."""
    for project in sorted(os.listdir(run_dir)):
        build = os.path.join(run_dir, project, "solution1", "csim", "build")
        if os.path.isfile(os.path.join(build, "csim.exe")):
            return build
    return None


def rerun_csim_exe(run_dir, build):
    """Re-execute the Vitis-built csim.exe against the current inputs/goldens."""
    for name in os.listdir(run_dir):
        if name.endswith(".txt"):
            shutil.copy2(os.path.join(run_dir, name), os.path.join(build, name))
    cp = subprocess.run([os.path.join(build, "csim.exe")], cwd=build,
                        capture_output=True, text=True)
    return cp.returncode, (cp.stdout or "") + (cp.stderr or "")


def run_trials(domain, stem, run_dir, config, n_inputs, mode):
    """Run n_inputs CSIM trials on an already-prepared design. -> result dict."""
    result = {
        "design": stem, "domain": domain, "datatype": "float", "n_inputs": n_inputs,
        "mode": mode, "trials_passed": 0, "max_abs_err": 0.0, "max_rel_err": 0.0,
        "n_mismatch": 0, "n_total": 0, "trial0_s": 0.0, "mean_trial_s": 0.0,
        "total_s": 0.0, "notes": "",
    }
    build = None
    trial_times = []
    notes = []
    t_all = time.time()

    for k in range(n_inputs):
        if k > 0:
            # Trial 0 reuses the seed-42 inputs + goldens prepare_one already wrote.
            refresh_inputs(domain, config, run_dir, seed=BASE_SEED + k)
            write_goldens(config, run_dir, domain)

        t0 = time.time()
        if mode == "exe" and k > 0 and build:
            rc, out = rerun_csim_exe(run_dir, build)
        else:
            rc, out = run_vitis(run_dir, log_name=f"vitis_trial{k}.log")
            if mode == "exe" and build is None:
                build = csim_build_dir(run_dir)
                if build is None:
                    notes.append("no csim.exe found; every trial ran full vitis_hls")
        trial_times.append(time.time() - t0)

        v = parse_verdict(out)
        result["max_abs_err"] = max(result["max_abs_err"], v["max_abs"])
        result["max_rel_err"] = max(result["max_rel_err"], v["max_rel"])
        result["n_mismatch"] = max(result["n_mismatch"], v["n_mismatch"])
        result["n_total"] = max(result["n_total"], v["n_total"])
        if v["verdict"] == "PASS":
            result["trials_passed"] += 1
        else:
            notes.append(f"trial{k}={v['verdict']}(rc={rc})")
            if len(notes) > 5:
                notes = notes[:5] + ["..."]

    result["total_s"] = time.time() - t_all
    result["trial0_s"] = trial_times[0] if trial_times else 0.0
    if len(trial_times) > 1:
        result["mean_trial_s"] = sum(trial_times[1:]) / (len(trial_times) - 1)
    result["passed"] = result["trials_passed"] == n_inputs
    result["notes"] = "; ".join(notes)
    return result


def select_configs(cfg_dir):
    """All configs in cfg_dir, skipping a bare `<op>.json` when `<op>__*.json`
    variants exist (same rule as prepare_designs / verify_operators)."""
    names = sorted(f for f in os.listdir(cfg_dir) if f.endswith(".json"))
    ops_with_variants = {n.split("__", 1)[0] for n in names if "__" in n}
    return [os.path.join(cfg_dir, n) for n in names
            if "__" in n or os.path.splitext(n)[0] not in ops_with_variants]


def prepare_all(domain, cfg_paths, out_base, tasks):
    """Serially generate every design + its seed-42 golden. -> list of job dicts."""
    jobs = []
    for cfg in cfg_paths:
        stem = os.path.splitext(os.path.basename(cfg))[0]
        t0 = time.time()
        try:
            run_dir, config = generate_design(domain, cfg, out_base, tasks)
            goldens = write_goldens(config, run_dir, domain)
            register_goldens_in_tcl(run_dir, list(goldens))
            jobs.append({"domain": domain, "stem": stem, "run_dir": run_dir,
                         "config": config, "prepare_s": time.time() - t0})
        except Exception as e:  # noqa: BLE001 - one bad config must not stop the suite
            print(f"  FAILED to prepare {domain}/{stem}: {e}")
            jobs.append({"domain": domain, "stem": stem, "run_dir": None,
                         "config": None, "prepare_s": time.time() - t0, "error": str(e)})
    return jobs


FIELDS = ["design", "domain", "datatype", "n_inputs", "mode", "trials_passed",
          "max_abs_err", "max_rel_err", "n_mismatch", "n_total",
          "prepare_s", "trial0_s", "mean_trial_s", "total_s", "notes"]

# Error columns span many decades (1e-7 .. 1e-2); rounding them to a fixed number
# of decimal places writes 0.0 for every small-but-nonzero error.
_ERR_FIELDS = {"max_abs_err", "max_rel_err"}
_TIME_FIELDS = {"prepare_s", "trial0_s", "mean_trial_s", "total_s"}


def _fmt(key, value):
    if not isinstance(value, float):
        return value
    if key in _ERR_FIELDS:
        return f"{value:.6g}"
    if key in _TIME_FIELDS:
        return round(value, 3)
    return value


def main():
    ap = argparse.ArgumentParser(description="N-input Vitis CSIM verification run")
    ap.add_argument("domains", nargs="*", default=None,
                    help=f"domains to run (default: {' '.join(DOMAINS)})")
    ap.add_argument("--n", type=int, default=1, dest="n_inputs",
                    help="randomized input trials per design (seed 42+k). Default 1.")
    ap.add_argument("--mode", choices=["full", "exe"], default="full",
                    help="full: vitis_hls per trial. exe: vitis_hls once, then re-run "
                         "the Vitis-built csim.exe per trial (much faster).")
    ap.add_argument("--suite", choices=["ops", "designs"], default="ops",
                    help="ops: verification/op_configs/<domain>. designs: <domain>/test_case_configs.")
    ap.add_argument("--configs-dir", default=None, help="override the config directory")
    ap.add_argument("--configs", nargs="*", default=None, help="specific config stems")
    ap.add_argument("--jobs", type=int, default=1, help="parallel Vitis runs (default 1)")
    ap.add_argument("--task", default="csim", help="comma-separated Vitis tasks")
    ap.add_argument("--out", default=None, help="output base dir")
    ap.add_argument("--csv", default=None, help="results CSV path")
    args = ap.parse_args()

    domains = args.domains or list(DOMAINS)
    tasks = [t.strip() for t in args.task.split(",") if t.strip()]
    suffix = "ops" if args.suite == "ops" else "designs"
    out_root = os.path.abspath(args.out or os.path.join(REPO_ROOT, "verification", f"_csim_{suffix}"))
    csv_path = args.csv or os.path.join(
        REPO_ROOT, "verification", f"results_csim_{suffix}_n{args.n_inputs}.csv")

    print(f">>> CSIM run: domains={domains} n={args.n_inputs} mode={args.mode} "
          f"jobs={args.jobs} suite={args.suite} out={out_root}")

    all_jobs = []
    t_prep = time.time()
    for domain in domains:
        cfg_dir = args.configs_dir or (
            os.path.join(REPO_ROOT, "verification", "op_configs", domain) if args.suite == "ops"
            else os.path.join(REPO_ROOT, domain, "test_case_configs"))
        if args.configs:
            cfg_paths = [os.path.join(cfg_dir, c if c.endswith(".json") else c + ".json")
                         for c in args.configs]
        else:
            cfg_paths = select_configs(cfg_dir)
        out_base = os.path.join(out_root, domain)
        os.makedirs(out_base, exist_ok=True)
        print(f">>> preparing {domain}: {len(cfg_paths)} configs")
        all_jobs += prepare_all(domain, cfg_paths, out_base, tasks)
    prep_s = time.time() - t_prep
    runnable = [j for j in all_jobs if j["run_dir"]]
    print(f">>> prepared {len(runnable)}/{len(all_jobs)} designs in {prep_s:.1f}s; "
          f"starting CSIM ({args.jobs} parallel)")

    results = []
    t_run = time.time()

    def _one(job):
        r = run_trials(job["domain"], job["stem"], job["run_dir"], job["config"],
                       args.n_inputs, args.mode)
        r["prepare_s"] = round(job["prepare_s"], 2)
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['domain']}/{r['design']}: "
              f"trials={r['trials_passed']}/{r['n_inputs']} "
              f"max_abs={r['max_abs_err']:.3e} n={r['n_total']} "
              f"t={r['total_s']:.1f}s {r['notes']}", flush=True)
        return r

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(_one, runnable))
    else:
        results = [_one(j) for j in runnable]
    run_s = time.time() - t_run

    for job in all_jobs:
        if not job["run_dir"]:
            results.append({"design": job["stem"], "domain": job["domain"],
                            "datatype": "float", "n_inputs": args.n_inputs,
                            "mode": args.mode, "trials_passed": 0,
                            "notes": f"prepare failed: {job.get('error', '')}",
                            "passed": False, "prepare_s": round(job["prepare_s"], 2)})

    results.sort(key=lambda r: (r["domain"], r["design"]))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: _fmt(k, v) for k, v in r.items() if k in FIELDS})

    n_pass = sum(1 for r in results if r.get("passed"))
    print(f"\n==== {n_pass}/{len(results)} designs PASS at n={args.n_inputs} "
          f"({args.mode} mode) ====")
    print(f"prepare {prep_s:.1f}s + csim {run_s:.1f}s (wall, jobs={args.jobs}) "
          f"= {prep_s + run_s:.1f}s total")
    cpu_s = sum(r.get("total_s", 0.0) for r in results)
    print(f"serial CSIM time across designs: {cpu_s:.1f}s")

    # max_abs_err is the headline accuracy number: max_rel_err is dominated by
    # rare near-cancelling elements once inputs are signed, so it is not a
    # usable summary statistic. See README "Reading max_rel_err".
    errs = sorted(r["max_abs_err"] for r in results if "max_abs_err" in r)
    if errs:
        worst = max(results, key=lambda r: r.get("max_abs_err", 0.0))
        median = errs[len(errs) // 2]
        exact = sum(1 for e in errs if e == 0.0)
        print(f"max_abs_err: worst {worst['max_abs_err']:.3e} "
              f"({worst['domain']}/{worst['design']}), median {median:.3e}, "
              f"{exact}/{len(errs)} designs bit-exact")
    print(f"wrote {csv_path}")
    failed = [f"{r['domain']}/{r['design']}" for r in results if not r.get("passed")]
    if failed:
        print("FAILED: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
