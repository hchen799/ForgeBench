#!/bin/bash
# Catapult backend check (Table-2 GEMM variants, columns C <32,10> and D float), Catapult 2026.2:
#   pm1    SCVerify C simulation, N=100 at +-1: same inputs, seeds, golden and bound as the Vitis {C,D}__pm1 runs
#   cosim  one Catapult run per design: SCVerify C sim + synthesis (go extract) + SCVerify RTL (Questa) sim, 10 looped trials, RTL vs C
# Results -> verification/results_paper/<col>__catapult_{pm1,cosim}/.
#   bash verification/paper_runs/run_catapult.sh [pm1] [cosim]
cd "$(dirname "$0")/../.."
# Siemens licenses as in /tools/software/siemens/setup.csh (MGLS_LICENSE_FILE from ~/.bashrc points at another server)
eval "$(grep -E 'setenv (SALT_LICENSE_SERVER|MGLS_LICENSE_FILE) ' /tools/software/siemens/setup.csh | sed 's/setenv \([A-Z_]*\) /export \1=/')"
export MGC_HOME=/tools/software/siemens/catapult/2026.2/Mgc_home
export PATH=$PATH:$MGC_HOME/bin          # appended: Mgc_home/bin also ships a python3
[ "$(dirname "$(readlink -f "$(command -v catapult)")")" = "$(readlink -f $MGC_HOME/bin)" ] || { echo "catapult on PATH is not 2026.2"; exit 1; }
EXPS=${@:-pm1 cosim}
mkdir -p verification/results_paper
for e in $EXPS; do
  for c in C_32_10 D_float; do
    name=${c}__catapult_$e
    echo "$(date '+%F %T') start $name"
    python3 -m verification.functional_verification --config verification/paper_runs/$name.json --jobs 2 \
        --out verification/results_paper/$name > verification/results_paper/$name.log 2>&1 &
  done
  wait
done
echo "$(date '+%F %T') ALL_DONE"
