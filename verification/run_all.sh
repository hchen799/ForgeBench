#!/usr/bin/env bash
# Turnkey ForgeBench functional-correctness run (server-side, needs Vitis HLS).
#
# What it does, per domain (gemm, conv, llm):
#   1. Generates every test_case_configs design in float with the requested task
#      (csim, or csim+cosim) and drops golden-reference files next to the inputs,
#      registering them as testbench files in run_hls.tcl.  [prepare_designs.py]
#   2. Runs `vitis_hls -f run_hls.tcl` in each generated design directory. The
#      generated testbench compares against the golden and prints a
#      `VERIFICATION:` verdict, returning nonzero on mismatch so CSIM surfaces it.
#   3. Aggregates all verdicts into verification/results.csv.  [collect_results.py]
#
# Usage:
#   bash verification/run_all.sh              # csim only (fast)
#   bash verification/run_all.sh csim,cosim   # csim + co-simulation (slow)
#   bash verification/run_all.sh csim verification/_server   # custom output dir
#
# Prereqs: `vitis_hls` on PATH, python3 with numpy. No g++ needed here (that is
# the local, Vitis-free self-check: `python3 -m verification.gcc_selfcheck <domain>`).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TASK="${1:-csim}"
OUTBASE="${2:-verification/_server}"

echo ">>> ForgeBench verification run: task=[$TASK] out=$OUTBASE"

for domain in gemm conv llm; do
  echo ">>> Preparing $domain designs + goldens"
  python3 -m verification.prepare_designs "$domain" --task "$TASK" --out "$OUTBASE/$domain"

  for d in "$OUTBASE/$domain"/*/; do
    [ -f "${d}run_hls.tcl" ] || continue
    echo ">>> vitis_hls: $d"
    ( cd "$d" && vitis_hls -f run_hls.tcl ) \
      || echo "!!! vitis_hls returned nonzero for $d (see its logs)"
  done
done

echo ">>> Aggregating results"
python3 -m verification.collect_results "$OUTBASE" --out verification/results.csv
echo ">>> Done. See verification/results.csv"
