#!/bin/bash
# Whole-design verification (CSIM) of the Table 7 generated programs: 4 columns x {largest range, +-1}. Results -> verification/results_paper/<cfg>/
cd "$(dirname "$0")/../.."
JOBS=${JOBS:-8}
mkdir -p verification/results_paper
for cfg in verification/paper_runs/*__table7_*.json; do
  name=$(basename "$cfg" .json)
  python3 -m verification.functional_verification --config "$cfg" --jobs $JOBS --out verification/results_paper/$name > verification/results_paper/$name.log 2>&1 &
done
wait
echo ALL_DONE
