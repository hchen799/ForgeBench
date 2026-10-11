#!/bin/bash
# Catapult vs Vitis synthesis on the 1,000-design GEMM sample (analysis/catapult_vs_vitis.py), both tools at once:
# 32 Vitis csynth jobs + 32 Catapult jobs (64 cores), 1 h per design per tool; resumable (designs already ok are skipped).
#   bash analysis/run_catapult_vs_vitis.sh [vitis_jobs] [catapult_jobs]
cd "$(dirname "$0")/.."
OUT=${OUT:-_sweeps/catapult_vs_vitis}
VJ=${1:-32}; CJ=${2:-32}
[ -d $OUT/configs ] || python3 -m analysis.catapult_vs_vitis sample --out $OUT
# Siemens licenses as in /tools/software/siemens/setup.csh (see verification/paper_runs/run_catapult.sh)
eval "$(grep -E 'setenv (SALT_LICENSE_SERVER|MGLS_LICENSE_FILE) ' /tools/software/siemens/setup.csh | sed 's/setenv \([A-Z_]*\) /export \1=/')"
export MGC_HOME=/tools/software/siemens/catapult/2026.2/Mgc_home
export PATH=$PATH:$MGC_HOME/bin
command -v vitis_hls >/dev/null || { echo "vitis_hls not on PATH"; exit 1; }
python3 -m analysis.sweep_runner --domain gemm --flow csynth --tool vitis --jobs $VJ --timeout 3600 \
    --configs-dir $OUT/configs --lean-out $OUT/lean_vitis ${LIMIT:+--limit $LIMIT} > $OUT/vitis.log 2>&1 &
python3 -m analysis.sweep_runner --domain gemm --flow csynth --tool catapult --jobs $CJ --timeout 3600 \
    --configs-dir $OUT/configs --lean-out $OUT/lean_catapult ${LIMIT:+--limit $LIMIT} > $OUT/catapult.log 2>&1 &
wait
python3 -m analysis.catapult_vs_vitis collect --out $OUT
echo "$(date '+%F %T') ALL_DONE"
