#!/bin/bash
# Conv impl with a preserved, seeded selection (replaces the July 896, whose sampler was not preserved): the 1,000 designs of
# manifest/conv_impl_selection.csv (analysis/impl_sampler.py conv --seed 20261014: all 576 with no unrolling + 53 from each of the
# 8 unrolled (unroll_cin, unroll_cout) groups), all run fresh into a new lean dir with the gemm/llm R2 settings (32 jobs, 4 h
# timeout). csynth is unchanged (_sweeps/lean/conv_csynth{,_r2b}). Resumable.
#   bash analysis/run_conv_impl_r3.sh
cd "$(dirname "$0")/.."
L=_sweeps/lean; G=_sweeps/logs
echo "$(date '+%F %T') start" >> $G/conv_r3.log
python3 -m analysis.sweep_runner --domain conv --flow impl --jobs ${IMPL_JOBS:-32} --timeout 14400 \
    --configs-dir conv/auto_generated_configs --select manifest/conv_impl_selection.csv \
    --lean-out $L/conv_impl_r3 >> $G/conv_impl_r3.log 2>&1
echo "$(date '+%F %T') impl finished" >> $G/conv_r3.log
