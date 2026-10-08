#!/bin/bash
# Paper verification (CSIM), operators: 4 format columns x {largest faithful range, +-1 baseline}, N=100.
# Results -> verification/results_paper/<config>/ (summary.csv, rows/, configs/ = exact design JSON per experiment, trials/, sweep_curves/)
# Usage: JOBS=14 bash verification/paper_runs/run_all.sh [A B C D]
cd "$(dirname "$0")/../.."
JOBS=${JOBS:-14}
COLS=${*:-A B C D}
mkdir -p verification/results_paper
for col in $COLS; do
  for cfg in verification/paper_runs/${col}_*__max_range.json verification/paper_runs/${col}_*__pm1.json; do
    name=$(basename "$cfg" .json)
    python3 -m verification.functional_verification --config "$cfg" --jobs $JOBS --out verification/results_paper/$name > verification/results_paper/$name.log 2>&1 &
  done
done
wait
echo ALL_DONE
