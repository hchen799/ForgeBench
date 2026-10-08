#!/bin/bash
# Paper CO-SIM: all operators at +-1, 10 looped trials, columns C (<32,10>) and D (float) run concurrently (JOBS variants each).
# Results -> verification/results_paper/<col>__cosim/ (summary.csv: cosim_status, cosim_c_vs_rtl_mismatches, cosim_latency_*).
# Run on an idle server; measured cost: float conv ~16 h per 20 trials (one design), gemm ~50 min, <16,5> conv ~26 min.
#   JOBS=16 bash verification/paper_runs/run_cosim.sh                  # C_32_10 and D_float
#   JOBS=16 bash verification/paper_runs/run_cosim.sh C_32_10          # one column
cd "$(dirname "$0")/../.."
JOBS=${JOBS:-16}
COLS=${@:-C_32_10 D_float}
mkdir -p verification/results_paper
for c in $COLS; do
  name=${c}__cosim
  echo "$(date '+%F %T') start $name"
  python3 -m verification.functional_verification --config verification/paper_runs/$name.json --jobs $JOBS \
      --out verification/results_paper/$name > verification/results_paper/$name.log 2>&1 &
done
wait
echo "$(date '+%F %T') ALL_DONE"
