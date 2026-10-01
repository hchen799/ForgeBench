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

rm -rf "$OUT"; mkdir -p "$OUT"/{configs,reports,full_model,logs}

have() { [ -e "$1" ] || { MISSING+=("$1"); return 1; }; }

# (i) all sweep JSON configs, one tarball per domain (gitignored */auto_generated_configs)
for d in gemm conv llm; do
  if have "$d/auto_generated_configs"; then
    tar -czf "$OUT/configs/sweep_configs_$d.tar.gz" -C "$d" auto_generated_configs
  fi
done

# (ii)+(iii) lean csynth / impl report archives. Split parts are re-joined so the bundle
# holds whole tarballs (Zenodo allows files up to 50 GB).
for base in csynth_gemm_lean csynth_conv_lean csynth_llm_lean; do
  if [ -f "$CKPT_DIR/$base.tar.gz" ]; then cp "$CKPT_DIR/$base.tar.gz" "$OUT/reports/"
  elif compgen -G "$CKPT_DIR/${base}_part_*" >/dev/null; then cat "$CKPT_DIR/${base}"_part_* > "$OUT/reports/$base.tar.gz"
  else MISSING+=("$CKPT_DIR/$base[.tar.gz|_part_*]"); fi
done
have checkpoints/r2/csynth_conv_rerun_lean.tar.gz && cp checkpoints/r2/csynth_conv_rerun_lean.tar.gz "$OUT/reports/"
for f in impl1000_reports_lean.tar.gz impl60_reports_lean.tar.gz; do
  have "$CKPT_DIR/$f" && cp "$CKPT_DIR/$f" "$OUT/reports/"
done

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
