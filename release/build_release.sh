#!/usr/bin/env bash
# Assemble the ForgeBench data release OUTSIDE git, then write SHA256SUMS and refresh
# release/bundle_files.csv. The human uploads release/_bundle/ to Zenodo + a GitHub Release.
#
# Usage: bash release/build_release.sh [BUNDLE_DIR]      (default: release/_bundle)
# Sources that do not exist are skipped with a warning (listed at the end), so the script
# can be re-run as more results (Catapult, HLSFactory, tool eval, full-model) appear.
# Override locations with env vars, e.g.  CKPT_DIR=/path/to/checkpoints/20260720
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:-release/_bundle}"
CKPT_DIR="${CKPT_DIR:-checkpoints/20260720}"
MISSING=()

rm -rf "$OUT"; mkdir -p "$OUT"/{full_model,logs}

have() { [ -e "$1" ] || { MISSING+=("$1"); return 1; }; }

# (i)-(iii) configs/ and reports/ trees as the manifest paths describe them (regenerated configs + lean csynth/impl/modular
# reports from the July and R2 archives named in manifest/_common.py), one tarball per domain, each with an inner SHA256SUMS.
python3 release/assemble_reports.py --out "$OUT" && rm -rf "$OUT/_tree"

# (iv) full-model configs + reports (Table 3): populated once workstream K lands.
# scale_csynth_r1_lean: July csynth reports of ResNet-18/34 (plain+tiled), saved before the build trees were deleted.
have "$CKPT_DIR/scale_csynth_r1_lean.tar.gz" && cp "$CKPT_DIR/scale_csynth_r1_lean.tar.gz" "$OUT/full_model/"
if have "scale_models/paper_configs"; then
  tar -czf "$OUT/full_model/paper_configs.tar.gz" -C scale_models paper_configs
fi

# (v)+(vi) verification / CO-SIM / Catapult / HLSFactory / tool-eval logs
for d in verification/results_fixed verification/e2e results/catapult integrations/hlsfactory/expected_output tool_eval; do
  if have "$d"; then tar -czf "$OUT/logs/$(echo "$d" | tr / _).tar.gz" "$d"; fi
done
have _run_logs && tar -czf "$OUT/logs/run_logs_r1_july.tar.gz" _run_logs
have verification/_csim_ops_n1000 && tar -czf "$OUT/logs/verification_csim_ops_n1000.tar.gz" verification/_csim_ops_n1000

( cd "$OUT" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS )
python3 release/make_manifest.py --out "${MANIFEST_OUT:-release/bundle_files.csv}" --category bundle "$OUT"

echo; echo "Bundle: $OUT  ($(du -sh "$OUT" | cut -f1))"
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "Skipped (not present yet):"; printf '  %s\n' "${MISSING[@]}"
fi
