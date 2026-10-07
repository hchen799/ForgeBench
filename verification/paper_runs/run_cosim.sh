#!/bin/bash
# Paper CO-SIM: columns A, B, D, then C (one after another, JOBS variants in parallel each). Results -> verification/results_paper/<col>__cosim/
# Run on an idle server. Measured cost: see verification/paper_runs/COSIM_TIMING.md.
#   JOBS=16 bash verification/paper_runs/run_cosim.sh            # all four columns
#   JOBS=16 bash verification/paper_runs/run_cosim.sh A_16_5      # one column
cd "$(dirname "$0")/../.."
JOBS=${JOBS:-16}
COLS=${@:-A_16_5 B_16_5_op24_8_acc32_10 D_float C_32_10}
mkdir -p verification/results_paper
for c in $COLS; do
  name=${c}__cosim
  echo "$(date '+%F %T') start $name"
  python3 -m verification.functional_verification --config verification/paper_runs/$name.json --jobs $JOBS \
      --out verification/results_paper/$name > verification/results_paper/$name.log 2>&1
  echo "$(date '+%F %T') done  $name (rc=$?)"
done
echo ALL_DONE
