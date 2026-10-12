#!/bin/bash
# Lookup-table exp (op option exp_impl=table, backends/exp_table.py): the 20 Table-2 variants that call exp/tanh, in the four paper
# columns, +-1 baseline and largest faithful range, N=100. -> verification/results_paper/<config>/
# Usage: JOBS=4 bash verification/paper_runs/run_exptable.sh
cd "$(dirname "$0")/../.."
JOBS=${JOBS:-4}
mkdir -p verification/results_paper
for cfg in verification/paper_runs/*__exptable_*.json; do
  name=$(basename "$cfg" .json)
  python3 -m verification.functional_verification --config "$cfg" --jobs $JOBS --out verification/results_paper/$name > verification/results_paper/$name.log 2>&1 &
done
wait
echo ALL_DONE
