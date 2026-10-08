#!/bin/bash
# Full-scale model verification (Vitis HLS C simulation of the generated accelerators) with
# scale_models/existing_model_verification/verify.py: every run compares the accelerator against (a) the integer-code fixed-point
# PyTorch reference (bit-exact: mismatching codes) and (b) the FP64 PyTorch graph (relative L2 <= 5% at every observed tensor).
# One seed = one synthetic parameter set + input (DRAM_input uniform +-1; weights +-sqrt(3/fan_in); BN gamma/var [0.9,1.1], beta/mean
# +-0.05; Llama: embedding +-0.5, norm weights [0.9,1.1], 4 prefill + 2 decode tokens).
#
#   SOURCES=<dir with RESNET*/LLAMA3_* project folders> OUT=<new results dir> [PY=python] [SEEDS="42 ... 51"] [LLAMA_SEEDS="42 43 44"] \
#       bash verification/paper_runs/run_fullmodel.sh
#
# Projects: the <32,10> (fixed<32,10,rnd,sat>) designs generated from the design JSON (scale_models/auto_generate_json.py ->
# gen_configs.run_hls_flow). A Llama run writes a ~33 GB parameter store; it is deleted after the run (its manifest and sha256s remain).
# Do not edit scale_models/existing_model_verification/references/contracts.json while this runs (verify.py checks it is unchanged).
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-python3}
SEEDS=${SEEDS:-"42 43 44 45 46 47 48 49 50 51"}
LLAMA_SEEDS=${LLAMA_SEEDS:-"42 43 44"}
PAR=${PAR:-3}
: "${SOURCES:?set SOURCES}" "${OUT:?set OUT}"
mkdir -p "$OUT"
LOG="$OUT/fullmodel_runs.txt"
cd "$REPO/scale_models"

resnet() {   # $1 project folder name, $2 seed
  local d=$1 s=$2 f o
  f=resnet$(echo "$d" | sed -E 's/RESNET([0-9]+).*/\1/')
  o="$OUT/${d}seed$s"
  rm -rf "$o"
  "$PY" existing_model_verification/verify.py --family "$f" --project "$SOURCES/$d" --seed "$s" --threads 8 --output "$o" > "$o.log" 2>&1
  echo "$d seed=$s $(head -n 1 "$o/summary.txt" 2>/dev/null) $(grep -o 'mismatches=[0-9]* .*relative_l2=[0-9.e-]*' "$o/summary.txt" 2>/dev/null | head -n 1)" >> "$LOG"
}
export -f resnet
export PY SOURCES OUT LOG

for s in $SEEDS; do
  for d in $(ls "$SOURCES" | grep '^RESNET' | grep '32_10'); do echo "$d $s"; done
done | xargs -P "$PAR" -n 2 bash -c 'resnet "$0" "$1"'

for s in $LLAMA_SEEDS; do
  o="$OUT/LLAMA3_8B_ctx2048_32_10_seed$s"
  rm -rf "$o"
  "$PY" existing_model_verification/verify.py --family llama3 \
      --project "$SOURCES/LLAMA3_8B_PREFILL_ctx2048_config_ap_fixed_32_10_" --decode-project "$SOURCES/LLAMA3_8B_DECODE_ctx2048_config_ap_fixed_32_10_" \
      --prefill 4 --decode 2 --seed "$s" --timeout 14400 --threads 16 --output "$o" > "$o.log" 2>&1
  echo "LLAMA3_8B_ctx2048_32_10 seed=$s $(head -n 1 "$o/summary.txt" 2>/dev/null)" >> "$LOG"
  rm -rf "$o/inputs"/*.bin
done
echo ALL_DONE >> "$LOG"
