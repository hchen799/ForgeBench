"""Catapult synthesis of the Table-2 GEMM variants at reduced unroll factors (no RTL simulation).

The Table-2 GEMM rows use a full unroll [8,16,16]; Catapult 2026.2 had not finished scheduling them after ~19 h, so synthesis runs
on the same designs (16x64x32, loop order ijk, same formats) with smaller unroll factors. One Catapult run per design x format x unroll:
`go analyze` .. `go extract` (Nangate 45nm, 10 ns), QoR from rtl.rpt.

  python3 -m verification.catapult_synth --out verification/results_paper/catapult_synth [--unrolls 1,1,1 1,1,4 1,1,16] [--jobs 4]

The license environment is inherited (see verification/paper_runs/run_catapult.sh). -> <out>/summary.{csv,json}, one row per run.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from verification.cosim import catapult_qor
from verification.csim_runner import run_catapult
from verification.fixedpoint import DType
from verification.functional_verification import select_variants
from verification.prepare_designs import generate_design

DATATYPES = ["fixed<32,10>", "float"]
UNROLLS = [[1, 1, 1], [1, 1, 4], [1, 1, 16]]
FIELDS = ["design", "datatype", "unroll", "table2_unroll", "status", "synth_s", "latency_cycles", "throughput_cycles", "area_score",
          "slack_ns", "clock_ns", "notes"]


def gemm_op(cfg):
    return next(op for op in cfg["ops"].values() if op["func_name"] == "gemm")


def synth_one(v, dt_text, unroll, out_dir, keep):
    dt = DType(dt_text)
    with open(v["path"]) as f:
        cfg = json.load(f)
    op = gemm_op(cfg)
    row = {"design": v["stem"], "datatype": dt_text, "unroll": "x".join(map(str, unroll)),
           "table2_unroll": "x".join(map(str, op["func_info"][1])), "clock_ns": 10, "notes": ""}
    op["func_info"][1] = list(unroll)
    tag = f"{v['stem']}__u{'_'.join(map(str, unroll))}"
    work = os.path.join(out_dir, "_work", dt.tag())
    os.makedirs(work, exist_ok=True)
    cfg_path = os.path.join(work, f"{tag}.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    run_dir, _ = generate_design(v["domain"], cfg_path, work, ["csynth"], data_type=dt_text, config_overrides={"tool": "catapult"})
    t0 = time.time()
    rc, log = run_catapult(run_dir)
    row["synth_s"] = round(time.time() - t0, 1)
    q = catapult_qor(run_dir)
    row.update({k: q.get(k, "") for k in ("latency_cycles", "throughput_cycles", "area_score", "slack_ns")})
    errors = [l for l in log.splitlines() if l.startswith("# Error")]
    row["status"] = "ok" if rc == 0 and q.get("latency_cycles") is not None and not errors else "synth_failed"
    if row["status"] != "ok":
        row["notes"] = f"rc={rc}; " + "; ".join(errors[:3])
    keep_dir = os.path.join(out_dir, "reports", dt.tag(), tag)
    os.makedirs(keep_dir, exist_ok=True)
    for name in ("catapult_run.log", "run_hls.tcl", "top.cpp", "top.h"):
        if os.path.exists(os.path.join(run_dir, name)):
            shutil.copy2(os.path.join(run_dir, name), keep_dir)
    if q.get("rtl_rpt"):
        shutil.copy2(os.path.join(run_dir, q["rtl_rpt"]), keep_dir)
    if not keep:
        shutil.rmtree(run_dir, ignore_errors=True)
    print(f"{time.strftime('%F %T')} {tag} {dt_text}: {row['status']} {row['synth_s']}s lat={row['latency_cycles']} "
          f"area={row['area_score']} slack={row['slack_ns']}", flush=True)
    return row


def safe(v, dt_text, unroll, out_dir, keep):
    try:
        return synth_one(v, dt_text, unroll, out_dir, keep)
    except Exception as e:                            # one failed generation/run must not lose the other rows
        print(f"{v['stem']} {dt_text} {unroll}: {e!r}", flush=True)
        return {"design": v["stem"], "datatype": dt_text, "unroll": "x".join(map(str, unroll)), "status": "error", "notes": repr(e)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--unrolls", nargs="+", default=[",".join(map(str, u)) for u in UNROLLS])
    ap.add_argument("--datatypes", nargs="+", default=DATATYPES)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--keep", action="store_true", help="keep the Catapult project directories")
    a = ap.parse_args(argv)
    unrolls = [[int(x) for x in u.split(",")] for u in a.unrolls]
    a.out = os.path.abspath(a.out)                  # the generator runs with the domain dir as cwd
    os.makedirs(a.out, exist_ok=True)
    jobs = [(v, dt, u) for v in select_variants(["gemm"]) for dt in a.datatypes for u in unrolls]
    print(f"{len(jobs)} Catapult synthesis runs, jobs={a.jobs}", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(lambda j: safe(*j, a.out, a.keep), jobs))
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return 0 if all(r["status"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
