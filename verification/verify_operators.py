"""Operator-level CSIM verification (local, Vitis-free proxy).

Runs every per-operator config in verification/op_configs/<domain> through the
same golden-oracle self-check as gcc_selfcheck (generate float design -> write
numpy golden -> g++ compile with shim -> run -> compare), then prints a
per-operator PASS/FAIL matrix. Exits nonzero if any operator fails.

Each config isolates ONE core operator (load -> op -> store) so a failure points
directly at that operator's C-vs-numpy correctness. The operator set is the full
list emitted by each domain's generate_code.py. This is the local proxy for the
server CSIM flow (verification/run_operators.sh), and is a faithful stand-in
because the emitted C is data-type "float" and compiled with equivalent math.

  uv run python -m verification.verify_operators            # all domains
  uv run python -m verification.verify_operators llm        # one domain
"""
import argparse
import glob
import os
import sys

from verification import gcc_selfcheck

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OP_CONFIGS = os.path.join(REPO_ROOT, "verification", "op_configs")
DOMAINS = ["gemm", "conv", "llm"]


def run_domain(domain, out_base):
    cfg_dir = os.path.join(OP_CONFIGS, domain)
    os.makedirs(out_base, exist_ok=True)
    results = {}
    for cfg in sorted(glob.glob(os.path.join(cfg_dir, "*.json"))):
        op = os.path.splitext(os.path.basename(cfg))[0]
        try:
            results[op] = gcc_selfcheck.run_one(domain, cfg, out_base)
        except Exception as e:  # noqa: BLE001 - surface per-op failures, keep going
            print(f"\n=== {domain}/{op} ===\n  EXCEPTION: {e}")
            results[op] = False
    return results


def main():
    ap = argparse.ArgumentParser(description="Operator-level CSIM verification (local proxy)")
    ap.add_argument("domains", nargs="*", default=[],
                    help=f"domains to run, any of {DOMAINS} (default: all)")
    ap.add_argument("--out", default=None, help="output base dir (default: verification/_selfcheck_ops)")
    args = ap.parse_args()

    bad = [d for d in args.domains if d not in DOMAINS]
    if bad:
        ap.error(f"invalid domain(s): {bad}; choose from {DOMAINS}")

    out_root = args.out or os.path.join(REPO_ROOT, "verification", "_selfcheck_ops")
    domains = args.domains if args.domains else DOMAINS

    all_results = {}
    for domain in domains:
        all_results[domain] = run_domain(domain, os.path.join(out_root, domain))

    # Per-operator PASS/FAIL matrix.
    print("\n" + "=" * 52)
    print("OPERATOR CSIM VERIFICATION SUMMARY (C vs numpy)")
    print("=" * 52)
    total = n_ok = 0
    failures = []
    for domain in domains:
        print(f"\n[{domain}]")
        for op, ok in sorted(all_results[domain].items()):
            total += 1
            n_ok += bool(ok)
            print(f"  {'PASS' if ok else 'FAIL'}  {op}")
            if not ok:
                failures.append(f"{domain}/{op}")
    print(f"\n==== {n_ok}/{total} operators PASS ====")
    if failures:
        print("FAILED:", ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
