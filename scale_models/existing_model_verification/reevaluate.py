#!/usr/bin/env python3
"""Re-evaluate recorded metrics using the current 5% FP64 acceptance rule.

This does not execute or replace an accelerator output. It updates verdicts
from the already recorded max-absolute and relative-L2 metrics and records the
change in the report and human-readable verification log.
"""
import argparse
import json
from pathlib import Path

from references.runner import (
    MAX_RELATIVE_L2_ERROR,
    ZERO_REFERENCE_MAX_ABSOLUTE_ERROR,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verdict(record):
    relative = record.get("relative_l2")
    passed = (
        record["max_abs"] <= ZERO_REFERENCE_MAX_ABSOLUTE_ERROR
        if relative is None
        else relative <= MAX_RELATIVE_L2_ERROR
    )
    record["mathematical_verdict"] = "PASS" if passed else "FAIL"


def final_lines(report):
    lines = [
        report["status"],
        "Fixed-point comparison: " + report["implementation_verdict"],
        "FP64 comparison: " + report["mathematical_verdict"],
        "",
        "Final layer output comparison:",
    ]
    schedule = report.get("schedule", {})
    for case_name, tensors in report.get("cases", {}).items():
        final = tensors.get("logits")
        if not final:
            continue
        shape = final.get("shape")
        if shape is None:
            rows = schedule.get("prefill", 1) if case_name == "call0000" else 1
            shape = [rows, 128256] if case_name.startswith("call") else [1000]
            final["shape"] = shape
        lines.append(
            f"  {case_name}/logits: shape={shape} "
            f"codes={final['implementation_verdict']} "
            f"mismatches={final['mismatching_codes']} "
            f"max_code_error={final['max_code_error']} "
            f"FP64={final['mathematical_verdict']} "
            f"max_abs={final['max_abs']:.12g} "
            f"relative_l2={final['relative_l2']!r} "
            f"relative_l2_limit={MAX_RELATIVE_L2_ERROR}"
        )
    return "\n".join(lines) + "\n"


def reevaluate(folder):
    summary_path = folder / "summary.json"
    report = json.loads(summary_path.read_text())
    previous = report.get("thresholds", {})
    rows = []
    for case_name, tensors in report.get("cases", {}).items():
        for record in tensors.values():
            verdict(record)
            rows.append(record)
        comparison = folder / ("case" if not case_name.startswith("call") else case_name) / "comparison.json"
        if comparison.exists():
            write_json(comparison, tensors)
    report["thresholds"] = {
        "relative_l2": MAX_RELATIVE_L2_ERROR,
        "zero_reference_max_absolute": ZERO_REFERENCE_MAX_ABSOLUTE_ERROR,
    }
    report["verdict_re_evaluation"] = {
        "metrics_reused": ["max_abs", "relative_l2"],
        "accelerator_rerun": False,
        "previous_thresholds": previous,
    }
    report["implementation_verdict"] = (
        "PASS" if rows and all(r["implementation_verdict"] == "PASS" for r in rows) else "FAIL"
    )
    report["mathematical_verdict"] = (
        "PASS" if rows and all(r["mathematical_verdict"] == "PASS" for r in rows) else "FAIL"
    )
    report["status"] = (
        "PASS"
        if report["implementation_verdict"] == report["mathematical_verdict"] == "PASS"
        else "FAIL"
    )
    report["exit_code"] = 0 if report["status"] == "PASS" else 1
    text = final_lines(report)
    write_json(summary_path, report)
    (folder / "summary.txt").write_text(text)
    with (folder / "verification.log").open("a") as log:
        log.write("\nVERDICT RE-EVALUATION FROM RECORDED METRICS\n" + text)
    print(folder, report["status"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for folder in args.runs:
        reevaluate(folder.resolve())


if __name__ == "__main__":
    main()
