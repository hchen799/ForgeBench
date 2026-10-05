"""C/RTL co-simulation of one operator variant over K seeded input sets inside a single Vitis run.

The generated testbench (`trials=K`) loops over per-trial inputs `<DRAM>.t<k>.txt` and writes `<out>_output.t<k>.txt`, so C-synthesis and
RTL elaboration are paid once for all K trials. Both CSIM and CO-SIM execute that same testbench (in `csim/build` and `sim/verilog`), so for
every trial we can compare

    C output  vs  RTL output   -> bit-exact agreement (the point of co-simulation)
    C output  vs  float64 golden on the quantized inputs   (same error metrics as the CSIM runs)

and read the RTL latency from the co-simulation report.
"""
import glob
import json
import os
import re
import shutil
import subprocess

import numpy as np

from verification.csim_runner import BASE_SEED, refresh_inputs
from verification.fixedpoint import DType, error_metrics, quantize
from verification.golden_ref import compute_goldens
from verification.prepare_designs import generate_design


def add_trial_files_to_tcl(run_dir, drams, k):
    path = os.path.join(run_dir, "run_hls.tcl")
    lines = open(path).read().split("\n")
    extra = [f"add_files -tb {d['name']}.t{t}.txt" for t in range(k) for d in drams]
    i = next(n for n, l in enumerate(lines) if l.startswith("open_solution"))
    lines[i:i] = extra + [""]
    open(path, "w").write("\n".join(lines))


def read_trial_outputs(directory, names, trial):
    parts = []
    for n in names:
        p = os.path.join(directory, f"{n}_output.t{trial}.txt")
        if not os.path.isfile(p):
            return None
        parts.append(np.loadtxt(p, dtype=np.float64).reshape(-1))
    return np.concatenate(parts)


def find_dir_with(run_dir, name):
    hits = glob.glob(os.path.join(run_dir, "**", name), recursive=True)
    return os.path.dirname(sorted(hits)[0]) if hits else None


def cosim_latency(run_dir):
    """(min, avg, max) RTL latency in cycles from the co-simulation report, or None."""
    for rpt in glob.glob(os.path.join(run_dir, "**", "sim", "report", "*_cosim.rpt"), recursive=True):
        txt = open(rpt).read()
        m = re.search(r"\|\s*Verilog\|\s*Pass\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|", txt) or re.search(r"Pass\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)", txt)
        if m:
            return tuple(int(x) for x in m.groups())
        return None
    return None


def run_cosim(domain, cfg_path, dtype_text, k, work, input_range=None, keep=False):
    """-> dict(status, notes, trials=[{trial, c_vs_rtl_mismatch, max_abs_err, ...}], latency)."""
    dt = DType(dtype_text)
    overrides = {"trials": k}
    if input_range is not None:
        overrides["input_range"] = input_range
    os.makedirs(work, exist_ok=True)
    run_dir, config = generate_design(domain, cfg_path, work, ["csim", "csynth", "cosim"], data_type=dtype_text, config_overrides=overrides)
    out = {"status": "ok", "notes": "", "trials": [], "latency": None, "run_dir": run_dir}
    try:
        names = config["output_dram_names"]
        qf = (lambda x: quantize(x, dt))
        goldens = []
        for t in range(k):
            refresh_inputs(domain, config, run_dir, seed=BASE_SEED + t)       # writes <DRAM>.txt for this seed
            for d in config["drams"]:
                shutil.copy2(os.path.join(run_dir, f"{d['name']}.txt"), os.path.join(run_dir, f"{d['name']}.t{t}.txt"))
            g = compute_goldens(config, run_dir, domain, input_transform=qf)
            goldens.append(np.concatenate([g[n].reshape(-1) for n in names]))
        add_trial_files_to_tcl(run_dir, config["drams"], k)
        cp = subprocess.run(["vitis_hls", "-f", "run_hls.tcl"], cwd=run_dir, capture_output=True, text=True)
        log = (cp.stdout or "") + (cp.stderr or "")
        open(os.path.join(run_dir, "vitis_cosim.log"), "w").write(log)
        # CSIM writes the C outputs in csim/build. In CO-SIM Vitis runs the testbench once more in sim/wrapc_pc with the RTL results
        # substituted for the DUT's outputs ("C post checking"), so that directory holds the RTL outputs (sim/wrapc holds a C run).
        hits = glob.glob(os.path.join(run_dir, "**", f"{names[0]}_output.t0.txt"), recursive=True)
        c_dir = next((os.path.dirname(p) for p in hits if os.sep + "csim" + os.sep in p), None)
        r_dir = next((os.path.dirname(p) for p in hits if os.sep + "wrapc_pc" + os.sep in p), None)
        if c_dir is None:
            out["status"], out["notes"] = "csim_failed", "no C outputs (CSIM did not run); see vitis_cosim.log"
            return out
        if r_dir is None:
            out["status"], out["notes"] = "cosim_failed", "no RTL outputs; see vitis_cosim.log"
            return out
        for t in range(k):
            c, r = read_trial_outputs(c_dir, names, t), read_trial_outputs(r_dir, names, t)
            if c is None or r is None:
                out["status"], out["notes"] = "cosim_incomplete", f"missing outputs for trial {t}"
                break
            m = error_metrics(c, goldens[t], atol=0.0)
            out["trials"].append({"trial": t, "c_vs_rtl_mismatch": int(np.sum(c != r)), "c_vs_rtl_max_abs": float(np.max(np.abs(c - r))),
                                  "max_abs_err": m["max_abs_err"], "rmse": m["rmse"], "n": m["n_elements"]})
        out["latency"] = cosim_latency(run_dir)
        if re.search(r"C/RTL co-simulation finished: FAIL", log):
            out["status"] = "cosim_testbench_fail"
    finally:
        if not keep:
            shutil.rmtree(run_dir, ignore_errors=True)
    return out
