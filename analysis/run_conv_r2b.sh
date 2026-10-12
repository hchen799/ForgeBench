#!/bin/bash
# Conv re-run after the tanh odd-symmetry fix (CHANGELOG_R2: Vitis hls::tanh(ap_fixed) returns tanh(|x|)): csynth of the 1,728
# activation=tanh sweep designs (manifest/conv_tanh_rerun_csynth.csv), then impl of the 256 of them in the July conv impl subset
# (manifest/conv_tanh_rerun_impl.csv). The other 3,456 conv designs (relu, sigmoid) are byte-identical and keep their July results.
# New lean dirs (*_r2b); configs regenerated first: byte-identical to conv/auto_generated_configs. Resumable.
#   bash analysis/run_conv_r2b.sh
cd "$(dirname "$0")/.."
L=_sweeps/lean; G=_sweeps/logs
echo "$(date '+%F %T') start" >> $G/conv_r2b.log
python3 -m analysis.sweep_runner --domain conv --flow csynth --jobs ${CSYNTH_JOBS:-64} \
    --configs-dir conv/auto_generated_configs --select manifest/conv_tanh_rerun_csynth.csv \
    --lean-out $L/conv_csynth_r2b >> $G/conv_csynth_r2b.log 2>&1
echo "$(date '+%F %T') csynth finished" >> $G/conv_r2b.log
python3 -m analysis.sweep_runner --domain conv --flow impl --jobs ${IMPL_JOBS:-32} --timeout 14400 \
    --configs-dir conv/auto_generated_configs --select manifest/conv_tanh_rerun_impl.csv \
    --lean-out $L/conv_impl_r2b >> $G/conv_impl_r2b.log 2>&1
echo "$(date '+%F %T') impl finished" >> $G/conv_r2b.log
