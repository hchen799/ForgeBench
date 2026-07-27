"""Operator-level CSIM verification (local, Vitis-free proxy).

Runs every per-operator config in verification/op_configs/<domain> through the
same golden-oracle self-check as gcc_selfcheck (generate float design -> write
numpy golden -> g++ compile with shim -> run -> compare), then prints a
per-operator PASS/FAIL matrix and writes results_operators.csv. Exits nonzero if
any operator variant fails.

Each config isolates ONE core operator (load -> op -> store) so a failure points
directly at that operator's C-vs-numpy correctness. This is the local proxy for
the server CSIM flow (verification/run_operators.sh), and is a faithful stand-in
because the emitted C is data-type "float" and compiled with equivalent math.

VARIANTS: an operator whose computation depends on parameters (e.g. an activation
function choice, a bias flag, grouped vs. dense attention, RoPE on/off) is
verified across those variants. Each variant is a separate JSON config named
``<operator>__<variant>.json`` (double underscore) inside op_configs/<domain>/;
all variants of an operator are grouped under that operator in the report, and
the ``Variants`` count is how many passed.

INPUTS: each variant is checked over ``--n`` randomized input trials (seed 42+k),
comparing the full output tensor element-wise each trial. Default n=1 reproduces
the historical single-input result.

  uv run python -m verification.verify_operators                 # all domains, n=1
  uv run python -m verification.verify_operators llm --n 100     # llm, 100 inputs/variant
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

from verification import gcc_selfcheck

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OP_CONFIGS = os.path.join(REPO_ROOT, "verification", "op_configs")
DOMAINS = ["gemm", "conv", "llm"]


def _operator_of(config_stem):
    """The base operator name for a config. ``activation__gelu`` -> ``activation``;
    ``mha__grouped_rope`` -> ``mha``; a plain ``gemm`` -> ``gemm``."""
    return config_stem.split("__", 1)[0]


def run_domain(domain, out_base, n_inputs=1):
    """Return {operator: [variant_result, ...]} for one domain.

    Config selection: an operator with explicit ``<op>__<variant>.json`` files is
    verified across those variants only; its bare ``<op>.json`` base (the template
    the variants were derived from) is skipped so it is not double-counted.
    Operators with no variant files are verified via their single base config.
    """
    cfg_dir = os.path.join(OP_CONFIGS, domain)
    os.makedirs(out_base, exist_ok=True)
    all_cfgs = sorted(glob.glob(os.path.join(cfg_dir, "*.json")))
    ops_with_variants = {
        _operator_of(os.path.splitext(os.path.basename(c))[0])
        for c in all_cfgs if "__" in os.path.basename(c)
    }
    by_op = defaultdict(list)
    for cfg in all_cfgs:
        stem = os.path.splitext(os.path.basename(cfg))[0]
        op = _operator_of(stem)
        # Skip a bare base config when that operator has explicit variants.
        if "__" not in stem and op in ops_with_variants:
            continue
        try:
            res = gcc_selfcheck.run_one_multi(domain, cfg, out_base, n_inputs=n_inputs)
        except Exception as e:  # noqa: BLE001 - surface per-variant failures, keep going
            print(f"\n=== {domain}/{stem} ===\n  EXCEPTION: {e}")
            res = {"op": stem, "domain": domain, "n_inputs": n_inputs,
                   "elements": 0, "max_rel_err": float("nan"), "passed": False}
        res["variant"] = stem
        by_op[op].append(res)
    return by_op


def main():
    ap = argparse.ArgumentParser(description="Operator-level CSIM verification (local proxy)")
    ap.add_argument("domains", nargs="*", default=[],
                    help=f"domains to run, any of {DOMAINS} (default: all)")
    ap.add_argument("--n", type=int, default=1, dest="n_inputs",
                    help="randomized input trials per variant (default: 1)")
    ap.add_argument("--out", default=None, help="output base dir (default: verification/_selfcheck_ops)")
    ap.add_argument("--csv", default=None,
                    help="CSV path (default: verification/results_operators.csv)")
    args = ap.parse_args()

    bad = [d for d in args.domains if d not in DOMAINS]
    if bad:
        ap.error(f"invalid domain(s): {bad}; choose from {DOMAINS}")

    out_root = args.out or os.path.join(REPO_ROOT, "verification", "_selfcheck_ops")
    csv_path = args.csv or os.path.join(REPO_ROOT, "verification", "results_operators.csv")
    domains = args.domains if args.domains else DOMAINS

    all_results = {}
    for domain in domains:
        all_results[domain] = run_domain(domain, os.path.join(out_root, domain), args.n_inputs)

    # Per-operator summary: Variants (passed/total), worst-case rel err, elements.
    print("\n" + "=" * 60)
    print("OPERATOR CSIM VERIFICATION SUMMARY (C vs numpy)")
    print("=" * 60)
    rows = []
    total_ops = ok_ops = 0
    for domain in domains:
        print(f"\n[{domain}]")
        for op, variants in sorted(all_results[domain].items()):
            n_var = len(variants)
            n_pass = sum(1 for v in variants if v["passed"])
            worst = max((v["max_rel_err"] for v in variants
                         if v["max_rel_err"] == v["max_rel_err"]), default=float("nan"))
            elements = max((v["elements"] for v in variants), default=0)
            n_inputs = variants[0]["n_inputs"] if variants else args.n_inputs
            op_ok = n_pass == n_var
            total_ops += 1
            ok_ops += op_ok
            print(f"  {'PASS' if op_ok else 'FAIL'}  {op:18s} "
                  f"variants={n_pass}/{n_var}  inputs={n_inputs}  "
                  f"elements={elements}  max_rel={worst:.2e}")
            rows.append({
                "domain": domain, "operator": op,
                "variants_passed": n_pass, "variants_total": n_var,
                "n_inputs": n_inputs, "elements": elements,
                "max_rel_err": worst, "passed": op_ok,
            })

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "domain", "operator", "variants_passed", "variants_total",
            "n_inputs", "elements", "max_rel_err", "passed"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")
    print(f"==== {ok_ops}/{total_ops} operators fully verified ====")
    if ok_ops != total_ops:
        sys.exit(1)


if __name__ == "__main__":
    main()
