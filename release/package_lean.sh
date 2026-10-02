#!/usr/bin/env bash
# Package a lean run directory (from analysis.sweep_runner / analysis.run_prebuilt) into a release archive
# with the member layout manifest/build_manifest.py indexes. status.csv and *.bak are left out (status.csv is
# copied to manifest/evidence/r2_runs/ instead).
#
#   release/package_lean.sh csynth <domain> <lean_dir> <out.tar.gz>   -> <domain>/large_hls_files/<design>/...
#   release/package_lean.sh impl   <domain> <lean_dir> <out.tar.gz>   -> <domain>__<design>/...
#   release/package_lean.sh modular_root - <lean_root> <out.tar.gz>   -> modular/<domain>/<name>/...   (lean_root has gemm/ conv/ llm/ subdirs)
set -euo pipefail
kind=$1; domain=$2; dir=$3; out=$4
case "$kind" in
  csynth)  tr="s,^\./,${domain}/large_hls_files/," ;;
  impl)    tr="s,^\./,${domain}__," ;;
  modular_root) tr="s,^\./,modular/," ;;
  *) echo "kind must be csynth|impl|modular_root" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$out")"
tar -czf "$out" -C "$dir" --exclude=status.csv --exclude='*.bak' --transform "$tr" .
echo "$out: $(tar -tzf "$out" | grep -c 'csynth.xml$') csynth.xml, $(tar -tzf "$out" | grep -c 'export_impl.xml$') export_impl.xml, $(du -h "$out" | cut -f1)"
