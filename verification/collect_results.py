"""Aggregate ForgeBench verification results into a CSV for the response letter.

Scans a server output tree (as produced by prepare_designs.py + run_all.sh),
finds each design's testbench `VERIFICATION:` verdict (printed during Vitis CSIM,
captured in the Vitis logs), and the CSIM/CO-SIM status, and writes a CSV:

  design, domain, datatype, csim_pass, cosim_pass, max_abs_err, max_rel_err,
  n_mismatch, n_total, notes

CLI:  python -m verification.collect_results verification/_server --out verification/results.csv

The VERIFICATION line format emitted by the generated testbench:
  VERIFICATION: PASS (max_abs=%g, max_rel=%g, n=%ld)
  VERIFICATION: FAIL (max_abs=%g, max_rel=%g, n_mismatch=%ld/%ld)
  VERIFICATION: SKIP (no golden files found)
"""
import argparse
import csv
import os
import re

_PASS_RE = re.compile(r"VERIFICATION:\s*PASS\s*\(max_abs=([^,]+),\s*max_rel=([^,]+),\s*n=(\d+)\)")
_FAIL_RE = re.compile(r"VERIFICATION:\s*FAIL\s*\(max_abs=([^,]+),\s*max_rel=([^,]+),\s*n_mismatch=(\d+)/(\d+)\)")
_SKIP_RE = re.compile(r"VERIFICATION:\s*SKIP")
# Vitis csim success/cosim markers (best-effort; exact strings vary by version).
_CSIM_PASS_RE = re.compile(r"CSim done with 0 errors", re.IGNORECASE)
_COSIM_PASS_RE = re.compile(r"Cosim.*(Pass|PASS)")
_COSIM_FAIL_RE = re.compile(r"Cosim.*(Fail|FAIL)")


def _iter_text_files(design_dir):
    for root, _dirs, files in os.walk(design_dir):
        for fn in files:
            if fn.endswith((".log", ".rpt", ".txt")):
                yield os.path.join(root, fn)


def _scan_design(design_dir):
    """Return a result dict for one design directory."""
    verdict = None  # 'PASS'/'FAIL'/'SKIP'
    max_abs = max_rel = ""
    n_mismatch = n_total = ""
    csim_pass = ""
    cosim_pass = ""
    notes = []

    blob = ""
    for path in _iter_text_files(design_dir):
        try:
            with open(path, errors="ignore") as f:
                blob += f.read() + "\n"
        except OSError:
            continue

    m = _FAIL_RE.search(blob)
    if m:
        verdict = "FAIL"
        max_abs, max_rel, n_mismatch, n_total = m.group(1), m.group(2), m.group(3), m.group(4)
    else:
        m = _PASS_RE.search(blob)
        if m:
            verdict = "PASS"
            max_abs, max_rel, n_total = m.group(1), m.group(2), m.group(3)
            n_mismatch = "0"
        elif _SKIP_RE.search(blob):
            verdict = "SKIP"
            notes.append("no golden file found by testbench")

    if verdict == "PASS":
        csim_pass = "PASS"
    elif verdict == "FAIL":
        csim_pass = "FAIL"
    elif _CSIM_PASS_RE.search(blob):
        csim_pass = "PASS(no-verdict)"

    if _COSIM_FAIL_RE.search(blob):
        cosim_pass = "FAIL"
    elif _COSIM_PASS_RE.search(blob):
        cosim_pass = "PASS"

    if verdict is None:
        notes.append("no VERIFICATION line found (csim may not have run)")

    return {
        "csim_pass": csim_pass,
        "cosim_pass": cosim_pass,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "n_mismatch": n_mismatch,
        "n_total": n_total,
        "notes": "; ".join(notes),
    }


def collect(out_base):
    rows = []
    for domain in sorted(os.listdir(out_base)):
        domain_dir = os.path.join(out_base, domain)
        if not os.path.isdir(domain_dir):
            continue
        for entry in sorted(os.listdir(domain_dir)):
            design_dir = os.path.join(domain_dir, entry)
            if not os.path.isfile(os.path.join(design_dir, "run_hls.tcl")):
                continue
            design = entry[len("_vprep_"):] if entry.startswith("_vprep_") else entry
            res = _scan_design(design_dir)
            res.update({"design": design, "domain": domain, "datatype": "float"})
            rows.append(res)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Aggregate verification results to CSV")
    ap.add_argument("out_base", help="server output base dir (has gemm/ conv/ llm/ subdirs)")
    ap.add_argument("--out", default="verification/results.csv")
    args = ap.parse_args()

    rows = collect(args.out_base)
    cols = ["design", "domain", "datatype", "csim_pass", "cosim_pass",
            "max_abs_err", "max_rel_err", "n_mismatch", "n_total", "notes"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    n_pass = sum(1 for r in rows if r["csim_pass"] == "PASS")
    print(f"wrote {args.out}: {len(rows)} designs, {n_pass} csim PASS")


if __name__ == "__main__":
    main()
