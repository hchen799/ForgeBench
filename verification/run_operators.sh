#!/usr/bin/env bash
# Operator-level ForgeBench CSIM run (server-side, needs Vitis HLS).
#
# Same flow as run_all.sh, but drives the per-operator config suite in
# verification/op_configs/<domain> instead of the full test_case_configs. Each
# design isolates ONE core operator (load -> op -> store) and CSIMs it against
# its numpy golden, so a VERIFICATION: FAIL points directly at that operator.
#
# Usage:
#   bash verification/run_operators.sh              # csim only
#   bash verification/run_operators.sh csim,cosim   # csim + co-simulation
#   bash verification/run_operators.sh csim verification/_server_ops   # custom out
#
# Prereqs: vitis_hls on PATH, python3 with numpy. The local, Vitis-free proxy is
#   python3 -m verification.verify_operators
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TASK="${1:-csim}"
OUTBASE="${2:-verification/_server_ops}"

echo ">>> ForgeBench operator verification run: task=[$TASK] out=$OUTBASE"

for domain in gemm conv llm; do
  echo ">>> Preparing $domain operator designs + goldens"
  python3 -m verification.prepare_designs "$domain" \
    --task "$TASK" \
    --configs-dir "verification/op_configs/$domain" \
    --out "$OUTBASE/$domain"

  for d in "$OUTBASE/$domain"/*/; do
    [ -f "${d}run_hls.tcl" ] || continue
    echo ">>> vitis_hls: $d"
    ( cd "$d" && vitis_hls -f run_hls.tcl ) \
      || echo "!!! vitis_hls returned nonzero for $d (see its logs)"
  done
done

echo ">>> Aggregating results"
python3 -m verification.collect_results "$OUTBASE" --out verification/results_operators.csv
echo ">>> Done. See verification/results_operators.csv"
