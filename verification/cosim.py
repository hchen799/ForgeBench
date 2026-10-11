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


def run_cosim(domain, cfg_path, dtype_text, k, work, input_range=None, inputs_config=None, keep=False):
    """-> dict(status, notes, trials=[{trial, c_vs_rtl_mismatch, max_abs_err, ...}], latency).

    `inputs_config` (a design config whose `input_range` and per-DRAM `input_range`s are used to draw the trial inputs, e.g. the ranged
    config of the CSIM run) takes precedence over `input_range`."""
    dt = DType(dtype_text)
    overrides = {"trials": k}
    if inputs_config is not None and inputs_config.get("input_range") is not None:
        overrides["input_range"] = inputs_config["input_range"]
    elif input_range is not None:
        overrides["input_range"] = input_range
    os.makedirs(work, exist_ok=True)
    run_dir, config = generate_design(domain, cfg_path, work, ["csim", "csynth", "cosim"], data_type=dtype_text, config_overrides=overrides)
    out = {"status": "ok", "notes": "", "trials": [], "latency": None, "run_dir": run_dir}
    try:
        names = config["output_dram_names"]
        qf = (lambda x: quantize(x, dt))
        goldens = []
        if inputs_config is not None:                                         # per-DRAM ranges (e.g. weights fixed at +-0.1)
            ranges = {d["name"]: d for d in inputs_config["drams"]}
            for d in config["drams"]:
                for key in ("input_range", "fixed_range"):
                    if key in ranges.get(d["name"], {}):
                        d[key] = ranges[d["name"]][key]
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


# ----------------------------------------------------------------------------- Catapult (SCVerify)
C_OUT_DIR = "c_outputs"


def catapult_split_outputs(run_dir):
    """Insert, between the SCVerify C simulation and the RTL simulation, a Tcl step that moves the C outputs into `c_outputs/` (both
    runs execute the same testbench in the design directory and would otherwise write the same files)."""
    path = os.path.join(run_dir, "run_hls.tcl")
    lines = open(path).read().split("\n")
    i = next(n for n, l in enumerate(lines) if "Verify_orig_cxx_osci.mk" in l)
    lines[i + 1:i + 1] = [f"file mkdir {C_OUT_DIR}",
                          f"foreach f [glob -nocomplain *_output.t*.txt] {{ file rename -force $f {C_OUT_DIR}/$f }}"]
    open(path, "w").write("\n".join(lines))


def catapult_rtl_status(log):
    """SCVerify's own RTL-vs-C transaction comparison, from the Catapult transcript. -> (status, notes)."""
    if re.search(r"Simulation PASSED", log):
        return "ok", ""
    if re.search(r"Simulation FAILED|mismatch", log, re.I):
        return "cosim_testbench_fail", "SCVerify reported a failure/mismatch; see catapult_run.log"
    return "cosim_incomplete", "no SCVerify RTL verdict; see catapult_run.log"


def catapult_qor(run_dir):
    """Synthesis results from the solution's rtl.rpt: latency / throughput (cycles, design total), area score (post-assignment, library
    units), critical-path slack (ns); with an FPGA library also DSP / LUT / MUX_CARRY and register bits (Catapult estimates, before
    Vivado). -> dict (missing fields absent)."""
    q = {}
    hits = sorted(glob.glob(os.path.join(run_dir, "*", "*", "rtl.rpt")))
    if not hits:
        return q
    txt = open(hits[0]).read()
    m = re.search(r"Design Total:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", txt)
    if m:
        q["real_ops"], q["latency_cycles"], q["throughput_cycles"], q["reset_length"] = (int(x) for x in m.groups())
    m = re.search(r"Total Area Score:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", txt)
    if m:
        q["area_score"] = float(m.group(3))
    m = re.search(r"TOTAL AREA \(After Assignment\):\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", txt)
    if m:                                             # FPGA library only: Catapult's own pre-Vivado estimates
        q["dsp"], q["lut"], q["mux_carry"] = (int(round(float(x))) for x in m.groups()[1:])
    m = re.search(r"Register-to-Variable Mappings.*?\n\s+Total:\s+(\d+)", txt, re.S)
    if m:
        q["reg_bits"] = int(m.group(1))
    m = re.search(r"Slack:\s+(-?[\d.]+)", txt)
    if m:
        q["slack_ns"] = float(m.group(1))
    q["rtl_rpt"] = os.path.relpath(hits[0], run_dir)
    return q


def catapult_rtl_latency(log):
    """Latency (cycles) measured by the SCVerify monitor in the RTL simulation, or None."""
    p, lat = re.search(r"constant clock period ([\d.]+) ns", log), re.search(r"Monitor: Latency: ([\d.]+) ns", log)
    return int(round(float(lat.group(1)) / float(p.group(1)))) if p and lat else None


def run_cosim_catapult(domain, cfg_path, dtype_text, k, work, inputs_config=None, keep=False):
    """Catapult flow for one design over K seeded input sets in one run: SCVerify C simulation, synthesis to RTL (`go extract`), SCVerify
    RTL (Verilog, Questa) simulation of the same K-trial testbench. -> same dict as run_cosim, plus `synth_s` and `qor` (from the report).
    C vs RTL is compared per trial bit for bit; C vs the float64 golden gives the usual error metrics."""
    import time
    from verification.csim_runner import run_catapult
    dt = DType(dtype_text)
    overrides = {"trials": k, "tool": "catapult"}
    if inputs_config is not None and inputs_config.get("input_range") is not None:
        overrides["input_range"] = inputs_config["input_range"]
    os.makedirs(work, exist_ok=True)
    run_dir, config = generate_design(domain, cfg_path, work, ["csim", "csynth", "cosim"], data_type=dtype_text, config_overrides=overrides)
    out = {"status": "ok", "notes": "", "trials": [], "latency": None, "run_dir": run_dir}
    try:
        names = config["output_dram_names"]
        qf = (lambda x: quantize(x, dt))
        goldens = []
        if inputs_config is not None:
            ranges = {d["name"]: d for d in inputs_config["drams"]}
            for d in config["drams"]:
                for key in ("input_range", "fixed_range"):
                    if key in ranges.get(d["name"], {}):
                        d[key] = ranges[d["name"]][key]
        for t in range(k):
            refresh_inputs(domain, config, run_dir, seed=BASE_SEED + t)
            for d in config["drams"]:
                shutil.copy2(os.path.join(run_dir, f"{d['name']}.txt"), os.path.join(run_dir, f"{d['name']}.t{t}.txt"))
            g = compute_goldens(config, run_dir, domain, input_transform=qf)
            goldens.append(np.concatenate([g[n].reshape(-1) for n in names]))
        catapult_split_outputs(run_dir)
        t0 = time.time()
        rc, log = run_catapult(run_dir)
        out["synth_s"] = round(time.time() - t0, 1)
        c_dir = os.path.join(run_dir, C_OUT_DIR)
        if read_trial_outputs(c_dir, names, 0) is None:
            out["status"], out["notes"] = "csim_failed", f"no C outputs (rc={rc}); see catapult_run.log"
            return out
        r_hits = [p for p in glob.glob(os.path.join(run_dir, "**", f"{names[0]}_output.t0.txt"), recursive=True)
                  if os.sep + C_OUT_DIR + os.sep not in p]
        if not r_hits:
            out["status"], out["notes"] = "cosim_failed", f"no RTL outputs (rc={rc}); see catapult_run.log"
            return out
        r_dir = os.path.dirname(sorted(r_hits)[0])
        for t in range(k):
            c, r = read_trial_outputs(c_dir, names, t), read_trial_outputs(r_dir, names, t)
            if c is None or r is None:
                out["status"], out["notes"] = "cosim_incomplete", f"missing outputs for trial {t}"
                break
            m = error_metrics(c, goldens[t], atol=0.0)
            out["trials"].append({"trial": t, "c_vs_rtl_mismatch": int(np.sum(c != r)), "c_vs_rtl_max_abs": float(np.max(np.abs(c - r))),
                                  "max_abs_err": m["max_abs_err"], "rmse": m["rmse"], "n": m["n_elements"]})
        if out["status"] == "ok":
            out["status"], out["notes"] = catapult_rtl_status(log)
        lat = catapult_rtl_latency(log)
        out["latency"] = (lat, lat, lat) if lat is not None else None
        out["qor"] = catapult_qor(run_dir)
    finally:
        if not keep:
            shutil.rmtree(run_dir, ignore_errors=True)
    return out
