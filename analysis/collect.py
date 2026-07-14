"""Collect PPA metrics from a results tree into a unified metrics.csv.

Walks a directory of generated designs, runs the selected tool's extractor over
each, and writes one row per design using the schema in analysis/schema.py.

Examples:
  python3 -m analysis.collect large_hls_files --tool vitis --flow csynth --domain gemm
  python3 -m analysis.collect impl_out --tool vitis --flow impl --out analysis/impl_metrics.csv
"""
import argparse
import csv

from analysis.extractors import get_extractor
from analysis.schema import CSV_COLUMNS


def collect(results_dir, tool="vitis", flow="csynth", domain="", datatype=""):
    ex = get_extractor(tool)
    rows = []
    for design_name, design_dir in ex.discover(results_dir, flow=flow):
        m = ex.parse(design_name, design_dir, flow=flow, domain=domain, datatype=datatype)
        if m is not None:
            rows.append(m.to_row())
    return rows


def write_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Collect PPA metrics into a unified CSV")
    ap.add_argument("results_dir")
    ap.add_argument("--tool", default="vitis")
    ap.add_argument("--flow", default="csynth", choices=["csynth", "impl"])
    ap.add_argument("--domain", default="")
    ap.add_argument("--datatype", default="")
    ap.add_argument("--out", default="analysis/metrics.csv")
    args = ap.parse_args()

    rows = collect(args.results_dir, args.tool, args.flow, args.domain, args.datatype)
    write_csv(rows, args.out)
    print(f"wrote {args.out}: {len(rows)} designs (tool={args.tool}, flow={args.flow})")


if __name__ == "__main__":
    main()
