#!/bin/bash
# LLM sweep re-run after the 2026-10-05 constant fix (CHANGELOG_R2): csynth of all 3,888 designs, then impl of the 1,000 in
# manifest/llm_impl_selection.csv (same seeded selection as before). New lean dirs (*_r2b): the runner skips designs already ok in
# its lean dir, so the stale 2026-10-01/02 results in _sweeps/lean/llm_{csynth,impl} are kept and not reused. Resumable.
#   bash analysis/run_llm_r2b.sh
cd "$(dirname "$0")/.."
L=_sweeps/lean; G=_sweeps/logs
python3 -m analysis.sweep_runner --domain llm --flow csynth --jobs ${CSYNTH_JOBS:-64} \
    --configs-dir llm/auto_generated_configs --lean-out $L/llm_csynth_r2b >> $G/llm_csynth_r2b.log 2>&1
echo "$(date '+%F %T') csynth finished" >> $G/llm_r2b.log
python3 -m analysis.sweep_runner --domain llm --flow impl --jobs ${IMPL_JOBS:-32} --timeout 14400 \
    --configs-dir llm/auto_generated_configs --select manifest/llm_impl_selection.csv \
    --lean-out $L/llm_impl_r2b >> $G/llm_impl_r2b.log 2>&1
echo "$(date '+%F %T') impl finished" >> $G/llm_r2b.log
