#!/bin/bash
# Table 7 (modularization) designs, CSIM, N=100: the 18 generated programs (*__table7_{pm1,max_range}) and the 8 hand-written designs
# (*__manual_{pm1,max_range}; verification/manual_designs.py), columns B (mixed: <16,5> data, <32,10> RND acc), C (<32,10>), D (float).
# Results -> verification/results_paper/<config>/ (summary.csv, rows/, configs/, trials/, sweep_curves/)
#   JOBS=6 bash verification/paper_runs/run_table7.sh            # B C D
#   JOBS=6 bash verification/paper_runs/run_table7.sh C          # one column
cd "$(dirname "$0")/../.."
JOBS=${JOBS:-6}
COLS=${*:-B C D}
mkdir -p verification/results_paper
for col in $COLS; do
  for cfg in verification/paper_runs/${col}_*__table7_*.json verification/paper_runs/${col}_*__manual_*.json; do
    name=$(basename "$cfg" .json)
    python3 -m verification.functional_verification --config "$cfg" --jobs $JOBS --out verification/results_paper/$name > verification/results_paper/$name.log 2>&1 &
  done
done
wait
echo ALL_DONE
