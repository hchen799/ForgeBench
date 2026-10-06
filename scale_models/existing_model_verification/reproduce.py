#!/usr/bin/env python3
"""Regenerate or verify the production JSON-generated HLS projects.

Run this file from any directory. Paths supplied on the command line are
resolved from the caller's working directory.
"""
import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCALE_MODELS = ROOT.parent
VERIFY = ROOT / "verify.py"

FORMATS = {
    "16_5": "ap_fixed<16,5>",
    "32_10": "ap_fixed<32,10>",
}

PROJECTS = {
    "resnet18-full": ("resnet18", "RESNET18", None),
    "resnet18-tiled": ("resnet18", "RESNET18_TILED", None),
    "resnet34-full": ("resnet34", "RESNET34", None),
    "resnet34-tiled": ("resnet34", "RESNET34_TILED", None),
    "resnet50-full": ("resnet50", "RESNET50", None),
    "resnet50-tiled": ("resnet50", "RESNET50_TILED", None),
    "resnet101-full": ("resnet101", "RESNET101", None),
    "resnet101-tiled": ("resnet101", "RESNET101_TILED", None),
    "resnet152-full": ("resnet152", "RESNET152", None),
    "resnet152-tiled": ("resnet152", "RESNET152_TILED", None),
    "llama3-8b-ctx2048": (
        "llama3",
        "LLAMA3_8B_PREFILL_ctx2048",
        "LLAMA3_8B_DECODE_ctx2048",
    ),
    "llama3-8b-ctx8192": (
        "llama3",
        "LLAMA3_8B_PREFILL_ctx8192",
        "LLAMA3_8B_DECODE_ctx8192",
    ),
    # Backward-compatible spelling used by the first verification campaign.
    "llama3-8b": (
        "llama3",
        "LLAMA3_8B_PREFILL_ctx2048",
        "LLAMA3_8B_DECODE_ctx2048",
    ),
}

DEFAULT_PROJECTS = tuple(name for name in PROJECTS if name != "llama3-8b")


def selected_precisions(value):
    return tuple(FORMATS) if value == "both" else (value,)


def project_name(stem, precision):
    return f"{stem}_config_ap_fixed_{precision}_"


def run(command, cwd=SCALE_MODELS, accepted=(0,)):
    print("+", " ".join(map(str, command)), flush=True)
    completed = subprocess.run([str(x) for x in command], cwd=cwd)
    if completed.returncode not in accepted:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed.returncode


def generate(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    precisions = selected_precisions(args.precision)
    run(
        [
            sys.executable,
            SCALE_MODELS / "auto_generate_json.py",
            "--data-types",
            *(FORMATS[p] for p in precisions),
        ]
    )
    sys.path.insert(0, str(SCALE_MODELS))
    import gen_configs

    for precision in precisions:
        for model in args.models:
            _, primary, secondary = PROJECTS[model]
            for stem in (primary, secondary):
                if stem is None:
                    continue
                config = (
                    SCALE_MODELS
                    / "auto_generated_configs"
                    / (project_name(stem, precision) + ".json")
                )
                print(f"+ gen_configs.run_hls_flow({config}, {output})", flush=True)
                gen_configs.run_hls_flow(str(config), str(output))


def verify(args):
    project_root = args.project_root.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    numerical_failures = []
    for precision in selected_precisions(args.precision):
        for model in args.models:
            family, primary, secondary = PROJECTS[model]
            command = [
                sys.executable,
                VERIFY,
                "--family",
                family,
                "--project",
                project_root / project_name(primary, precision),
                "--output",
                output_root / precision / model,
                "--seed",
                args.seed,
                "--threads",
                args.threads,
                "--timeout",
                args.timeout,
            ]
            if secondary:
                command.extend(
                    [
                        "--decode-project",
                        project_root / project_name(secondary, precision),
                        "--prefill",
                        args.prefill,
                        "--decode",
                        args.decode,
                    ]
                )
            if args.vitis:
                command.extend(["--vitis", args.vitis])
            if args.inspect_only:
                command.append("--inspect-only")
            returncode = run(command, accepted=(0, 1))
            if returncode == 1:
                numerical_failures.append(f"{precision}/{model}")
    if numerical_failures:
        print(
            "Completed with recorded FP64 acceptance failures: "
            + ", ".join(numerical_failures),
            flush=True,
        )


def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--precision", choices=("16_5", "32_10", "both"), default="both"
    )
    common.add_argument(
        "--models",
        nargs="+",
        choices=tuple(PROJECTS),
        default=list(DEFAULT_PROJECTS),
    )
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(required=True)
    generate_parser = commands.add_parser(
        "generate", parents=[common], help="run JSON -> gen_configs.run_hls_flow"
    )
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.set_defaults(function=generate)
    verify_parser = commands.add_parser(
        "verify", parents=[common], help="execute already generated production C++"
    )
    verify_parser.add_argument(
        "--project-root", type=Path, default=SCALE_MODELS / "hls_files"
    )
    verify_parser.add_argument("--output-dir", type=Path, required=True)
    verify_parser.add_argument("--seed", type=int, default=42)
    verify_parser.add_argument("--threads", type=int, default=8)
    verify_parser.add_argument("--timeout", type=int, default=14400)
    verify_parser.add_argument("--vitis", help="path to the vitis_hls executable")
    verify_parser.add_argument("--prefill", type=int, default=4)
    verify_parser.add_argument("--decode", type=int, default=2)
    verify_parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="check source contracts and interfaces without numerical execution",
    )
    verify_parser.set_defaults(function=verify)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
