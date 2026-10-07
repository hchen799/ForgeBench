#!/bin/bash
# Paper verification (CSIM): 4 format columns x {largest faithful range, +-1 baseline}. Results -> verification/results_paper/<config>/
cd "$(dirname "$0")/../.."
JOBS=${JOBS:-14}
for cfg in verification/paper_runs/*__*.json; do
  name=$(basename "$cfg" .json)
  python3 -m verification.functional_verification --config "$cfg" --jobs $JOBS --out verification/results_paper/$name > verification/results_paper/$name.log 2>&1 &
done
wait
echo ALL_DONE
