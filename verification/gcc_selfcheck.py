"""Local, Vitis-free self-check for the ForgeBench verification flow.

For each design config this:
  1. Generates the design with data_type="float", task=["csim"] via the domain's
     existing `gen_configs.run_hls_flow` (run in a subprocess with cwd=<domain> so
     the relative activation template path and DRAM-file moves behave exactly as in
     the normal flow).
  2. Builds the golden reference from the SAME generated input `.txt` files.
  3. Compiles the emitted top.cpp + tb_top.cpp with a stock g++ using the shim
     headers (no Vitis) and runs it.
  4. Cross-checks two ways:
       - Python: golden vs the testbench's `<DRAM>_output.txt` dump (authoritative
         check of the golden + generated C).
       - Parses the testbench's `VERIFICATION:` line if present (confirms the tb
         compare logic once that is wired in).

This validates the golden reference and the generated logic without Vitis. The
real CSIM/CO-SIM run happens on the server via run_all.sh.

CLI:
  python -m verification.gcc_selfcheck gemm                 # all test_case_configs
  python -m verification.gcc_selfcheck gemm mlp mult_op_p1  # specific configs
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

from verification.golden_ref import compute_goldens
from verification.io import write_golden

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIM_DIR = os.path.join(REPO_ROOT, "verification", "shim")

RTOL = 1e-3
ATOL = 1e-5

_VERIF_RE = re.compile(r"VERIFICATION:\s*(PASS|FAIL)")


def _domain_dir(domain):
    return os.path.join(REPO_ROOT, domain)


def _config_path(domain, name):
    if not name.endswith(".json"):
        name += ".json"
    return os.path.join(_domain_dir(domain), "test_case_configs", name)


def generate_float_design(domain, config_path, out_base):
    """Generate a float/csim design; return (run_dir, modified_config_dict)."""
    with open(config_path) as f:
        config = json.load(f)
    config["data_type"] = "float"
    config["task"] = ["csim"]

    stem = os.path.splitext(os.path.basename(config_path))[0]
    run_name = f"_vtmp_{stem}"
    domain_dir = _domain_dir(domain)
    tmp_cfg = os.path.join(domain_dir, f"{run_name}.json")
    with open(tmp_cfg, "w") as f:
        json.dump(config, f)

    try:
        code = (
            "import gen_configs; "
            f"gen_configs.run_hls_flow({os.path.basename(tmp_cfg)!r}, "
            f"base_dir={out_base!r}, task=['csim'])"
        )
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=domain_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        if os.path.exists(tmp_cfg):
            os.remove(tmp_cfg)

    return os.path.join(out_base, run_name), config


def compile_and_run(run_dir):
    """g++ compile top.cpp+tb_top.cpp with shim, run; return (rc, stdout)."""
    sim = os.path.join(run_dir, "sim")
    compile_cmd = [
        "g++", "-std=c++14", "-O1",
        f"-I{SHIM_DIR}", f"-I{run_dir}",
        "-o", sim,
        os.path.join(run_dir, "top.cpp"),
        os.path.join(run_dir, "tb_top.cpp"),
    ]
    cp = subprocess.run(compile_cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        return None, f"COMPILE ERROR:\n{cp.stderr}"
    rp = subprocess.run([sim], cwd=run_dir, capture_output=True, text=True)
    return rp.returncode, rp.stdout + rp.stderr


def python_compare(config, run_dir, domain):
    """Compare golden vs the tb's <DRAM>_output.txt dump. Returns per-output stats."""
    goldens = compute_goldens(config, run_dir, domain)
    for name, arr in goldens.items():
        write_golden(os.path.join(run_dir, f"{name}.golden.txt"), arr)

    stats = {}
    for name, golden in goldens.items():
        out_path = os.path.join(run_dir, f"{name}_output.txt")
        if not os.path.exists(out_path):
            stats[name] = {"error": "no output dump"}
            continue
        got = np.loadtxt(out_path, dtype=np.float32).reshape(-1)
        ref = np.asarray(golden, dtype=np.float32).reshape(-1)
        n = min(got.size, ref.size)
        got, ref = got[:n], ref[:n]
        abs_err = np.abs(got - ref)
        rel_err = abs_err / (np.abs(ref) + 1e-12)
        tol = ATOL + RTOL * np.abs(ref)
        # Non-finite C output (e.g. overflow to inf/nan) counts as a mismatch;
        # NaN comparisons are False, so count them explicitly.
        nonfinite = ~np.isfinite(got)
        n_mis = int(np.sum((abs_err > tol) | nonfinite))
        stats[name] = {
            "max_abs": float(abs_err.max()) if n else 0.0,
            "max_rel": float(rel_err.max()) if n else 0.0,
            "n_mismatch": n_mis,
            "n_total": int(n),
            "pass": n_mis == 0,
        }
    return stats


def run_one(domain, config_path, out_base):
    stem = os.path.splitext(os.path.basename(config_path))[0]
    run_dir, config = generate_float_design(domain, config_path, out_base)
    rc, out = compile_and_run(run_dir)
    stats = python_compare(config, run_dir, domain)

    tb_verdict = None
    m = _VERIF_RE.search(out or "")
    if m:
        tb_verdict = m.group(1)

    ok = rc == 0 and all(s.get("pass") for s in stats.values())
    print(f"\n=== {domain}/{stem} ===")
    if rc is None:
        print(out)  # compile error
        return False
    for name, s in stats.items():
        if "error" in s:
            print(f"  {name}: {s['error']}")
        else:
            print(
                f"  {name}: max_abs={s['max_abs']:.3e} max_rel={s['max_rel']:.3e} "
                f"mismatch={s['n_mismatch']}/{s['n_total']} "
                f"{'PASS' if s['pass'] else 'FAIL'}"
            )
    print(f"  g++ exit={rc}  tb_verdict={tb_verdict or 'n/a'}  -> {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Vitis-free golden self-check")
    ap.add_argument("domain", choices=["gemm", "conv", "llm"])
    ap.add_argument("configs", nargs="*", help="config names (default: all test_case_configs)")
    ap.add_argument("--out", default=None, help="output base dir (default: verification/_selfcheck/<domain>)")
    args = ap.parse_args()

    out_base = args.out or os.path.join(REPO_ROOT, "verification", "_selfcheck", args.domain)
    os.makedirs(out_base, exist_ok=True)

    if args.configs:
        cfg_paths = [_config_path(args.domain, c) for c in args.configs]
    else:
        cfg_dir = os.path.join(_domain_dir(args.domain), "test_case_configs")
        cfg_paths = sorted(
            os.path.join(cfg_dir, f) for f in os.listdir(cfg_dir) if f.endswith(".json")
        )

    results = {}
    for cfg in cfg_paths:
        stem = os.path.splitext(os.path.basename(cfg))[0]
        try:
            results[stem] = run_one(args.domain, cfg, out_base)
        except Exception as e:  # noqa: BLE001 - surface per-config failures, keep going
            print(f"\n=== {args.domain}/{stem} ===\n  EXCEPTION: {e}")
            results[stem] = False

    n_ok = sum(1 for v in results.values() if v)
    print(f"\n==== {args.domain}: {n_ok}/{len(results)} configs OK ====")
    failed = [k for k, v in results.items() if not v]
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
