"""Compare generated Llama 3 HLS C++ with fixed-point and mathematical PyTorch."""
import argparse
from pathlib import Path

from llama3_verification.io import prepare_case, prepare_model
from llama3_verification.runner import compare, csim, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    model = commands.add_parser("model", help="create one reusable packed random-parameter model")
    model.add_argument("--project-dir", type=Path, required=True)
    model.add_argument("--model-dir", type=Path, required=True)
    model.add_argument("--seed", type=int, default=42)
    for name in ("prepare", "reference", "csim", "compare", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--run-dir", type=Path, required=True)
        if name in ("prepare", "run"):
            sub.add_argument("--model-dir", type=Path, required=True)
            sub.add_argument("--prefill", type=int, default=4)
            sub.add_argument("--decode", type=int, default=2)
            sub.add_argument("--seed", type=int, default=43)
            sub.add_argument("--trace", choices=("ops", "layers"), default="ops")
            sub.add_argument("--chunk", type=int)
        if name in ("reference", "run"):
            sub.add_argument("--threads", type=int, default=16)
            sub.add_argument("--cache-gib", type=float, default=64)
        if name in ("csim", "run"):
            sub.add_argument("--backend", choices=("vitis", "native"), default="vitis")
            sub.add_argument("--vitis", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "model":
            prepare_model(args.project_dir, args.model_dir, args.seed)
        if args.command in ("prepare", "run"):
            prepare_case(args.model_dir, args.run_dir, args.prefill, args.decode, args.seed, args.trace, args.chunk)
        if args.command in ("reference", "run"):
            reference(args.run_dir, args.threads, args.cache_gib)
        if args.command in ("csim", "run"):
            csim(args.run_dir, args.backend, args.vitis)
        if args.command in ("compare", "run"):
            compare(args.run_dir)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
