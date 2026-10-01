"""Run a csynth or impl sweep with bounded disk use.

For every design: generate it from its JSON config, run `vitis_hls -f run_hls.tcl`, copy the
"lean" files (sources + csynth.xml [+ export_impl.xml + power report for impl]) to --lean-out,
then delete the build tree. Peak disk is ~jobs x one build tree (csynth ~70 MB, impl ~140 MB),
not the whole sweep. Resumable: designs already recorded as ok in <lean-out>/status.csv are skipped.

  python -m analysis.sweep_runner --domain llm --flow csynth --jobs 48 \
      --configs-dir llm/auto_generated_configs --lean-out _sweeps/lean/llm_csynth
  python -m analysis.sweep_runner --domain llm --flow impl --jobs 32 --timeout 14400 \
      --configs-dir llm/auto_generated_configs --select manifest/llm_impl_selection.csv \
      --lean-out _sweeps/lean/llm_impl

Lean layout (what the release bundle / analysis.collect expect):
  <lean-out>/<design>/{top.cpp,top.h,tb_top.cpp,run_hls.tcl}
  <lean-out>/<design>/project_1/solution1/syn/report/csynth.xml
  <lean-out>/<design>/project_1/solution1/impl/report/verilog/export_impl.xml        (impl)
  <lean-out>/<design>/project_1/solution1/impl/verilog/project.runs/impl_1/*_power_routed.rpt  (impl)
  <lean-out>/<design>/vitis_tail.log                                                 (failures only)
"""
import argparse
import csv
import glob
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOL = os.path.join("project_1", "solution1")
CSYNTH_XML = os.path.join(SOL, "syn", "report", "csynth.xml")
IMPL_XML = os.path.join(SOL, "impl", "report", "verilog", "export_impl.xml")
IMPL_RPT_GLOB = os.path.join(SOL, "impl", "verilog", "project.runs", "impl_1", "*_power_routed.rpt")
SOURCES = ["top.cpp", "top.h", "tb_top.cpp", "run_hls.tcl"]
TASKS = {"csynth": ["csynth"], "impl": ["csynth", "export_ip"]}
STATUS_COLS = ["design", "status", "rc", "seconds", "finished_at"]


def _copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def run_one(design, cfg_path, a, tmpl_q):
    """Generate -> vitis_hls -> lean copy -> prune. Returns (status, rc, seconds)."""
    t0 = time.time()
    work = os.path.join(a.work_dir, design)
    lean = os.path.join(a.lean_out, design)
    tmpl = tmpl_q.get()  # per-thread cwd copy of the generator dir: gen_configs writes DRAM_*.txt in cwd
    # The sweep configs carry "task": ["csynth"] and gen_configs lets the config override the `task`
    # argument, so write a copy whose task is the one this flow needs (otherwise impl silently = csynth).
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["task"] = TASKS[a.flow]
    cfg_dir = os.path.join(a.work_dir, "_cfg")
    os.makedirs(cfg_dir, exist_ok=True)
    run_cfg = os.path.join(cfg_dir, design + ".json")
    with open(run_cfg, "w") as f:
        json.dump(cfg, f, indent=4)
    try:
        gen = ("import gen_configs; gen_configs.run_hls_flow("
               f"{run_cfg!r}, base_dir={os.path.abspath(a.work_dir)!r}, task={TASKS[a.flow]!r})")
        g = subprocess.run([sys.executable, "-c", gen], cwd=tmpl, capture_output=True, text=True)
        if g.returncode != 0 or not os.path.isfile(os.path.join(work, "run_hls.tcl")):
            return "gen_fail", g.returncode, time.time() - t0
    finally:
        tmpl_q.put(tmpl)
        if os.path.exists(run_cfg):
            os.remove(run_cfg)

    logp = os.path.join(work, "vitis_hls.log")
    status, rc = "ok", 0
    try:
        with open(logp, "wb") as lf:
            rc = subprocess.run(["vitis_hls", "-f", "run_hls.tcl"], cwd=work, stdout=lf,
                                stderr=subprocess.STDOUT, timeout=a.timeout).returncode
    except subprocess.TimeoutExpired:
        status, rc = "timeout", -9
        # the timed-out vitis_hls leaves vivado children behind; kill anything running in this work dir
        subprocess.run(["pkill", "-f", work], capture_output=True)
    need = [CSYNTH_XML] + ([IMPL_XML] if a.flow == "impl" else [])
    if status == "ok" and (rc != 0 or not all(os.path.isfile(os.path.join(work, n)) for n in need)):
        status = "fail" if rc != 0 else "invalid_report"

    os.makedirs(lean, exist_ok=True)
    for f in SOURCES:
        if os.path.isfile(os.path.join(work, f)):
            _copy(os.path.join(work, f), os.path.join(lean, f))
    for rel in [CSYNTH_XML, IMPL_XML]:
        if os.path.isfile(os.path.join(work, rel)):
            _copy(os.path.join(work, rel), os.path.join(lean, rel))
    for p in glob.glob(os.path.join(work, IMPL_RPT_GLOB)):
        _copy(p, os.path.join(lean, os.path.relpath(p, work)))
    if status != "ok" and os.path.isfile(logp):
        with open(logp, "rb") as lf:
            tail = lf.read().splitlines()[-100:]
        with open(os.path.join(lean, "vitis_tail.log"), "wb") as o:
            o.write(b"\n".join(tail) + b"\n")
    shutil.rmtree(work, ignore_errors=True)
    return status, rc, time.time() - t0


def load_done(status_csv):
    if not os.path.isfile(status_csv):
        return set()
    with open(status_csv, newline="") as f:
        return {r["design"] for r in csv.DictReader(f) if r["status"] == "ok"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, choices=["gemm", "conv", "llm"])
    ap.add_argument("--flow", required=True, choices=["csynth", "impl"])
    ap.add_argument("--configs-dir", required=True)
    ap.add_argument("--select", help="CSV with a `design` column (config stems); default = all configs")
    ap.add_argument("--lean-out", required=True)
    ap.add_argument("--work-dir", help="scratch for build trees (default <lean-out>/../_work_<basename>)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=7200, help="per-design seconds")
    ap.add_argument("--limit", type=int, help="only the first N designs (smoke test)")
    a = ap.parse_args()
    a.lean_out = os.path.abspath(a.lean_out)
    a.work_dir = os.path.abspath(a.work_dir or os.path.join(os.path.dirname(a.lean_out), "_work_" + os.path.basename(a.lean_out)))
    os.makedirs(a.lean_out, exist_ok=True)
    os.makedirs(a.work_dir, exist_ok=True)

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(a.configs_dir) if f.endswith(".json"))
    if a.select:
        with open(a.select, newline="") as f:
            want = {r["design"] for r in csv.DictReader(f)}
        missing = want - set(stems)
        if missing:
            sys.exit(f"{len(missing)} selected designs have no config, e.g. {sorted(missing)[:3]}")
        stems = [s for s in stems if s in want]
    status_csv = os.path.join(a.lean_out, "status.csv")
    done = load_done(status_csv)
    todo = [s for s in stems if s not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{a.domain}/{a.flow}: {len(stems)} designs, {len(done)} already ok, {len(todo)} to run, jobs={a.jobs}", flush=True)

    tmpl_q = queue.Queue()
    gen_src = os.path.join(REPO_ROOT, a.domain)
    for i in range(a.jobs):
        d = os.path.join(a.work_dir, f"_tmpl{i}")
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(gen_src, "*.py")) + glob.glob(os.path.join(gen_src, "*.cpp")):
            shutil.copy2(f, d)
        tmpl_q.put(d)

    new_file = not os.path.isfile(status_csv)
    lock = threading.Lock()
    sf = open(status_csv, "a", newline="")
    w = csv.writer(sf, lineterminator="\n")
    if new_file:
        w.writerow(STATUS_COLS)
    counts, t_start = {}, time.time()

    def job(stem):
        status, rc, secs = run_one(stem, os.path.join(a.configs_dir, stem + ".json"), a, tmpl_q)
        with lock:
            w.writerow([stem, status, rc, f"{secs:.0f}", time.strftime("%F %T")])
            sf.flush()
            counts[status] = counts.get(status, 0) + 1
            n = sum(counts.values())
            if n % 50 == 0 or status != "ok":
                print(f"[{time.strftime('%T')}] {n}/{len(todo)} {counts} elapsed={(time.time()-t_start)/60:.0f}m"
                      + ("" if status == "ok" else f"  {status}: {stem}"), flush=True)

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        list(ex.map(job, todo))
    sf.close()
    shutil.rmtree(a.work_dir, ignore_errors=True)
    print(f"DONE {counts} in {(time.time()-t_start)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
