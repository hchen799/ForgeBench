"""CLI for shared vectors, two PyTorch references, Vitis CSIM, and comparison."""
import argparse
import math
import sys
from pathlib import Path

import torch

from tiled_resnet18_verification.io import prepare
from tiled_resnet18_verification.runner import compare, csim, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "reference", "csim", "compare", "run"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--run-dir", required=True, type=Path)
        sub.add_argument("--threads", type=int, default=8, help="CPU PyTorch threads (default: 8)")
        if name in ("prepare", "run"):
            sub.add_argument("--project-dir", required=True, type=Path)
            sub.add_argument("--seed", type=int, default=42)
        if name in ("reference", "run"):
            sub.add_argument("--sequential", action="store_true", help="Disable the proven-safe matrix reduction path")
        if name == "reference":
            sub.add_argument("--mode", choices=("both", "float", "fixed"), default="both")
        if name in ("csim", "run"):
            sub.add_argument("--vitis", default="vitis_hls")
            sub.add_argument("--cache-dir", type=Path)
            sub.add_argument("--rebuild", action="store_true")
        if name in ("compare", "run"):
            sub.add_argument("--float-atol", type=float)
            sub.add_argument("--float-rtol", type=float)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    atol, rtol = getattr(args, "float_atol", None), getattr(args, "float_rtol", None)
    if (atol is None) != (rtol is None) or any(v is not None and (not math.isfinite(v) or v < 0) for v in (atol, rtol)):
        parser.error("provide both nonnegative finite --float-atol and --float-rtol, or neither")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    try:
        if args.command in ("prepare", "run"):
            prepare(args.project_dir, args.run_dir, args.seed)
            print(f"Prepared seed {args.seed}: {args.run_dir}", flush=True)
        if args.command in ("reference", "run"):
            reference(args.run_dir, getattr(args, "mode", "both"), fast=not args.sequential)
        sim_code = 0
        if args.command in ("csim", "run"):
            sim_code = csim(args.run_dir, args.vitis, args.cache_dir, args.rebuild)
        if args.command in ("compare", "run"):
            return 0 if compare(args.run_dir, atol, rtol) else 1
        return 1 if sim_code else 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"VERIFICATION ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
