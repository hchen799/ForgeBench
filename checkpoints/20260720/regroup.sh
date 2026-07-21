#!/bin/bash
# Reassemble the split checkpoint archives and (optionally) extract them.
#
# Large archives are committed as 75 MB `_part_NN` chunks because GitHub rejects
# any single file over 100 MB. This mirrors the ML_testsuite_part_* convention:
#   cat <name>_part_* > <name>.tar.gz && tar -xzf <name>.tar.gz
#
# Usage:
#   bash regroup.sh            # reassemble only
#   bash regroup.sh --extract  # reassemble, verify, and untar
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

EXTRACT=${1:-}

for base in csynth_conv_lean csynth_llm_lean; do
  if compgen -G "${base}_part_*" > /dev/null; then
    echo ">>> reassembling ${base}.tar.gz"
    cat "${base}"_part_* > "${base}.tar.gz"
  fi
done

echo ">>> verifying archives"
for f in csynth_gemm_lean.tar.gz csynth_conv_lean.tar.gz csynth_llm_lean.tar.gz \
         impl60_reports_lean.tar.gz; do
  [ -f "$f" ] || { echo "    (missing $f - skipping)"; continue; }
  if tar -tzf "$f" > /dev/null 2>&1; then
    echo "    OK  $f"
  else
    echo "    CORRUPT $f" >&2; exit 1
  fi
done

if [ "$EXTRACT" = "--extract" ]; then
  for f in csynth_*_lean.tar.gz impl60_reports_lean.tar.gz; do
    [ -f "$f" ] || continue
    echo ">>> extracting $f"
    tar -xzf "$f"
  done
fi

cat <<'EOF'

Contents:
  csynth_{gemm,conv,llm}_lean.tar.gz  per-design csynth.xml + HLS source
                                      (3840 / 5183 / 3888 designs)
  impl60_reports_lean.tar.gz          60 impl designs: export_impl.xml,
                                      csynth.xml, *_power_routed.rpt
                                      (post-P&R power / Fmax / timing)

Re-extract metrics to CSV with:
  python3 -m analysis.collect <dir> --tool vitis --flow csynth --domain <dm> --out <out.csv>
EOF
