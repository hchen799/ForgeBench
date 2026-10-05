#!/usr/bin/env python3
"""Verify saved production HLS against version-specific fixed and FP64 PyTorch.

Never generates/replaces top.cpp. Exit 0: both comparisons PASS; 1: numerical
FAIL; 2: invalid interface/arithmetic, tool failure, or incomplete verification.
"""
import argparse
import contextlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification.existing.project import Project, sha, write_json
from verification.existing.runner import (
    find_vitis,
    llama_preflight,
    run_llama,
    run_resnet,
)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for stream in self.streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("resnet18", "llama3"), required=True)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--decode-project", type=Path)
    parser.add_argument("--decode-config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--vitis")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--prefill", type=int, default=4)
    parser.add_argument("--decode", type=int, default=2)
    parser.add_argument(
        "--tokens",
        help="explicit comma-separated IDs; never implicitly narrowed to fit fixed point",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="save provenance and interface/arithmetic diagnostics without running models",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must be a new directory")
    if args.threads < 1 or args.timeout < 1 or args.seed < 0:
        parser.error("threads/timeout must be positive; seed must be nonnegative")
    output = args.output.resolve()
    output.mkdir(parents=True)
    report = dict(
        schema_version=1,
        status="INCOMPLETE",
        implementation_verdict="NOT_RUN",
        mathematical_verdict="NOT_RUN",
        seed=args.seed,
        threads=args.threads,
        thresholds=dict(max_absolute=0.1, relative_l2=0.01),
        execution="Vitis HLS C simulation",
        accelerator_executed=False,
        claim="saved production source; synthetic parameters; no trained accuracy or RTL claim",
    )
    projects = []
    start = time.monotonic()
    rc = 2
    with (output / "verification.log").open("w") as log, contextlib.redirect_stdout(
        Tee(sys.stdout, log)
    ), contextlib.redirect_stderr(Tee(sys.stderr, log)):
        try:
            import numpy as np
            import torch

            torch.set_num_threads(args.threads)
            report["versions"] = dict(
                python=sys.version, torch=torch.__version__, numpy=np.__version__
            )
            report["reference_sources"] = {
                str(p.relative_to(Path(__file__).parent)): sha(p)
                for p in sorted((Path(__file__).parent / "existing").iterdir())
                if p.suffix in (".py", ".json")
            }
            report["reference_sources"]["verify_existing.py"] = sha(__file__)
            for name in report["reference_sources"]:
                target = output / "reference_source" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(__file__).parent / name, target)
            project = Project(args.project, args.family, args.config)
            projects.append(project)
            project.snapshot(output / "source")
            report["projects"] = [project.manifest()]
            failures = []
            if args.family == "llama3":
                if not args.decode_project:
                    raise ValueError(
                        "--decode-project is required for the paired Llama campaign"
                    )
                decode = Project(args.decode_project, args.family, args.decode_config)
                projects.append(decode)
                decode.snapshot(output / "decode_source")
                report["projects"].append(decode.manifest())
                tokens, failures = llama_preflight(project, decode, args)
                report["schedule"] = dict(
                    prefill=args.prefill, decode=args.decode, tokens=tokens
                )
                report["reference_independence"] = dict(
                    fp64="independent PyTorch graph and real-valued primitives",
                    fixed="independent integer-code graph; RoPE shares Vitis pow/sin/cos primitive constants",
                    limitation="fixed RoPE checks do not independently validate vendor transcendental functions",
                )
            report["reference_pairs"] = {
                p.variant: dict(
                    fp64=f"{p.variant}-fp64-v{p.contract_version}",
                    fixed=f"{p.variant}-ap-trn-wrap-v{p.contract_version}",
                )
                for p in projects
            }
            report["preflight_failures"] = failures
            write_json(output / "manifest.json", report)
            if failures:
                report.update(
                    status="INVALID_DESIGN",
                    implementation_verdict="NOT_RUN",
                    mathematical_verdict="NOT_RUN",
                )
                print("INVALID_DESIGN:\n" + "\n".join(failures), flush=True)
            elif args.inspect_only:
                report["status"] = "INSPECTED_NOT_EXECUTED"
                rc = 0
            else:
                args.vitis = find_vitis(args.vitis)
                include = args.vitis.parent.parent / "include"
                library_files = [
                    include / "ap_fixed.h",
                    include / "etc/ap_fixed_base.h",
                    include / "hls_math.h",
                    include / "etc/hls_sqrt_apfixed.h",
                    include / "etc/hls_exp_apfixed.h",
                ]
                report["vendor_headers"] = {
                    str(p): sha(p) for p in library_files if p.exists()
                }
                result = (
                    run_resnet(project, output, args)
                    if args.family == "resnet18"
                    else run_llama(project, decode, output, args, tokens)
                )
                report.update(result)
                rows = [r for case in result["cases"].values() for r in case.values()]
                for kind in ("implementation_verdict", "mathematical_verdict"):
                    report[kind] = (
                        "PASS"
                        if rows and all(r[kind] == "PASS" for r in rows)
                        else "FAIL"
                    )
                report["status"] = (
                    "PASS"
                    if all(
                        report[k] == "PASS"
                        for k in ("implementation_verdict", "mathematical_verdict")
                    )
                    else "FAIL"
                )
                rc = 0 if report["status"] == "PASS" else 1
        except Exception as exc:
            report.update(status="ERROR", error=str(exc))
            traceback.print_exc()
        finally:
            try:
                for p in projects:
                    p.assert_unchanged()
                report["original_sources_unchanged"] = bool(projects)
                for name, digest in report.get("reference_sources", {}).items():
                    if sha(Path(__file__).parent / name) != digest:
                        raise ValueError(
                            "reference source changed during verification: " + name
                        )
                report["reference_sources_unchanged"] = True
            except Exception as exc:
                report.update(
                    status="ERROR", error=str(exc), original_sources_unchanged=False
                )
                rc = 2
            # Preserve failed simulations too; lack of comparison output must
            # not hide a compiler error or a signal from the actual binary.
            report["simulations"] = {
                str(p.parent.relative_to(output)): json.loads(p.read_text())
                for p in sorted(output.glob("*/simulation.json"))
            }
            report["accelerator_executed"] = bool(report["simulations"]) and all(
                s["executed"] for s in report["simulations"].values()
            )
            report["elapsed_seconds"] = time.monotonic() - start
            report["exit_code"] = rc
            write_json(output / "summary.json", report)
            summary = f"{report['status']}\nFixed-point comparison: {report['implementation_verdict']}\nFP64 comparison: {report['mathematical_verdict']}\n"
            if report.get("error"):
                summary += "Error: " + report["error"] + "\n"
            summary += "\n".join(report.get("preflight_failures", [])) + "\n"
            (output / "summary.txt").write_text(summary)
            print(summary, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
