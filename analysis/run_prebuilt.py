"""csynth for designs that already exist as directories (not generated from a config): the modularization suite.

Reads a design registry (default manifest/designs/modular.csv), copies each `hls_dir` to a scratch dir, runs
`vitis_hls -f run_hls.tcl` on a csynth-only copy of its TCL (csim/cosim/export lines are dropped), keeps the
"lean" reports, deletes the scratch tree. Same status/resume conventions as analysis/sweep_runner.py.

    # 1. look at the plan (no Vitis needed)
    python -m analysis.run_prebuilt --dry-run
    # 2. run (38 designs; each takes minutes)
    python -m analysis.run_prebuilt --jobs 8 --lean-out _sweeps/lean/modular_csynth
    # 3. recompute the Table 7 inputs from the new reports
    python modular_data/parse_synth_resourc_util.py --reports-root _sweeps/lean/modular_csynth --out _sweeps/modular_r2
    # 4. package for the release bundle (layout modular/<domain>/<name>/...)
    tar -czf checkpoints/r2/csynth_modular_lean.tar.gz -C _sweeps/lean/modular_csynth --transform 's,^,modular/,' .

Lean layout:  <lean-out>/<domain>/<name>/{top.cpp,top.h,tb_top.cpp,run_hls.tcl,vitis_tail.log (failures)}
              <lean-out>/<domain>/<name>/project_1/solution1/syn/report/{csynth.xml,csynth.rpt,<function>_csynth.xml,...}
The per-function `<function>_csynth.xml` files are needed: Table 7's "Shared" column sums the areas of the shared
functions inside each modularized design (see ROWS in modular_data/parse_synth_resourc_util.py).
"""
import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join("project_1", "solution1", "syn", "report")
SOURCES = ["top.cpp", "top.h", "tb_top.cpp", "run_hls.tcl"]
DROP = re.compile(r"^\s*(csim_design|cosim_design|export_design)\b")   # uncommented lines only
STATUS_COLS = ["design_id", "status", "rc", "seconds", "finished_at", "notes"]


def csynth_only_tcl(text):
    kept, dropped = [], []
    for line in text.splitlines():
        (dropped if DROP.match(line) else kept).append(line)
    if not any(l.strip() == "csynth_design" for l in kept):
        raise ValueError("TCL has no csynth_design")
    return "\n".join(kept) + "\n", dropped


def load_registry(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def shared_functions(cases_csv):
    """modularized design id -> list of shared function names (from modular_cases.csv)."""
    out = {}
    if os.path.isfile(cases_csv):
        with open(cases_csv, newline="") as f:
            for r in csv.DictReader(f):
                out.setdefault(r["modularized_design"], [])
                out[r["modularized_design"]] += [x for x in r["shared_functions"].split(";") if x]
    return out


def run_one(row, a, shared):
    t0 = time.time()
    domain, name = row["domain"], row["name"]
    src = os.path.join(REPO_ROOT, row["hls_dir"])
    work = os.path.join(a.work_dir, f"{domain}__{name}")
    lean = os.path.join(a.lean_out, domain, name)
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(src, work, ignore=shutil.ignore_patterns("project_1", "*.log", "*.jou"))
    tcl, _ = csynth_only_tcl(open(os.path.join(src, "run_hls.tcl")).read())
    with open(os.path.join(work, "run_hls.tcl"), "w") as f:
        f.write(tcl)

    logp = os.path.join(work, "vitis_hls.log")
    status, rc = "ok", 0
    try:
        with open(logp, "wb") as lf:
            rc = subprocess.run(["vitis_hls", "-f", "run_hls.tcl"], cwd=work, stdout=lf, stderr=subprocess.STDOUT,
                                timeout=a.timeout).returncode
    except subprocess.TimeoutExpired:
        status, rc = "timeout", -9
    rep = os.path.join(work, REPORT_DIR)
    notes = ""
    if status == "ok":
        if rc != 0:
            status = "fail"
        elif not os.path.isfile(os.path.join(rep, "csynth.xml")):
            status = "invalid_report"
        else:
            missing = [fn for fn in shared.get(row["design_id"], []) if not os.path.isfile(os.path.join(rep, f"{fn}_csynth.xml"))]
            if missing:            # design synthesized, but Table 7's "Shared" column cannot be computed
                status, notes = "missing_function_reports", ";".join(missing)

    os.makedirs(lean, exist_ok=True)
    for f in SOURCES:
        p = os.path.join(src, f)                       # original sources, not the stripped TCL
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(lean, f))
    if os.path.isdir(rep):
        dst = os.path.join(lean, REPORT_DIR)
        os.makedirs(dst, exist_ok=True)
        for p in glob.glob(os.path.join(rep, "*.xml")) + glob.glob(os.path.join(rep, "csynth.rpt")):
            shutil.copy2(p, dst)
    if status != "ok" and os.path.isfile(logp):
        with open(logp, "rb") as lf:
            tail = lf.read().splitlines()[-100:]
        with open(os.path.join(lean, "vitis_tail.log"), "wb") as o:
            o.write(b"\n".join(tail) + b"\n")
    shutil.rmtree(work, ignore_errors=True)
    return status, rc, time.time() - t0, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=os.path.join(REPO_ROOT, "manifest", "designs", "modular.csv"))
    ap.add_argument("--cases", default=os.path.join(REPO_ROOT, "manifest", "designs", "modular_cases.csv"))
    ap.add_argument("--lean-out", default=os.path.join(REPO_ROOT, "_sweeps", "lean", "modular_csynth"))
    ap.add_argument("--work-dir", help="scratch (default <lean-out>/../_work_<basename>)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=7200, help="per-design seconds")
    ap.add_argument("--only", nargs="*", help="substrings of design_id to run (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    a = ap.parse_args()
    a.lean_out = os.path.abspath(a.lean_out)
    a.work_dir = os.path.abspath(a.work_dir or os.path.join(os.path.dirname(a.lean_out), "_work_" + os.path.basename(a.lean_out)))

    rows = load_registry(a.registry)
    if a.only:
        rows = [r for r in rows if any(s in r["design_id"] for s in a.only)]
    shared = shared_functions(a.cases)
    status_csv = os.path.join(a.lean_out, "status.csv")
    done = set()
    if os.path.isfile(status_csv):
        with open(status_csv, newline="") as f:
            done = {r["design_id"] for r in csv.DictReader(f) if r["status"] == "ok"}
    todo = [r for r in rows if r["design_id"] not in done]

    if a.dry_run:
        print(f"{len(rows)} designs in {os.path.relpath(a.registry, REPO_ROOT)}, {len(done)} already ok, {len(todo)} to run")
        bad = 0
        for r in todo:
            d = os.path.join(REPO_ROOT, r["hls_dir"])
            try:
                _, dropped = csynth_only_tcl(open(os.path.join(d, "run_hls.tcl")).read())
                msg = f"drops {len(dropped)} line(s): {'; '.join(x.strip() for x in dropped)}" if dropped else "csynth-only already"
            except (OSError, ValueError) as e:
                msg, bad = f"ERROR {e}", bad + 1
            sf = len(shared.get(r["design_id"], []))
            print(f"  {r['design_id']:<46} role={r['role']:<19} tb={'y' if r['has_testbench'] == 'True' else 'n'}  "
                  f"shared fns to check={sf}  {msg}")
        print(f"scratch: {a.work_dir}\nlean out: {a.lean_out}\nvitis_hls: {shutil.which('vitis_hls') or 'NOT FOUND'}")
        sys.exit(1 if bad else 0)

    if not shutil.which("vitis_hls"):
        sys.exit("vitis_hls not on PATH")
    os.makedirs(a.lean_out, exist_ok=True)
    os.makedirs(a.work_dir, exist_ok=True)
    print(f"{len(rows)} designs, {len(done)} already ok, {len(todo)} to run, jobs={a.jobs}", flush=True)
    new = not os.path.isfile(status_csv)
    sf = open(status_csv, "a", newline="")
    w = csv.writer(sf, lineterminator="\n")
    if new:
        w.writerow(STATUS_COLS)
    counts, t_start, lock = {}, time.time(), threading.Lock()

    def job(row):
        status, rc, secs, notes = run_one(row, a, shared)
        with lock:
            w.writerow([row["design_id"], status, rc, f"{secs:.0f}", time.strftime("%F %T"), notes])
            sf.flush()
            counts[status] = counts.get(status, 0) + 1
            print(f"[{time.strftime('%T')}] {sum(counts.values())}/{len(todo)} {status:<14} {row['design_id']} ({secs:.0f}s) {notes}", flush=True)

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        list(ex.map(job, todo))
    sf.close()
    shutil.rmtree(a.work_dir, ignore_errors=True)
    print(f"DONE {counts} in {(time.time() - t_start) / 60:.0f} min")


if __name__ == "__main__":
    main()
