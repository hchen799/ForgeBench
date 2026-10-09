#!/bin/bash
# Copy the compact, reviewable part of the paper verification runs into the tracked folder verification/paper_results/:
# per run: summary.csv, resolved_config.json (run settings), configs/ (exact design JSON of every experiment), sweep_curves/ (range search),
# rows/; per-trial logs (trials/*.csv.gz) stay in verification/results_paper/ (release bundle). Full-scale models: summary.json/.txt and
# the input manifests (seeds, distributions, sha256 of every parameter tensor) of every run_fullmodel.sh run.
#   FULLMODEL=<run_fullmodel.sh OUT dir> bash verification/paper_runs/collect_results.sh
set -eu
cd "$(dirname "$0")/../.."
SRC=verification/results_paper
DST=verification/paper_results
mkdir -p "$DST"
for run in $SRC/*__max_range $SRC/*__pm1 $SRC/*__table7_* $SRC/*__manual_* $SRC/*__cosim; do
  [ -d "$run" ] || continue
  name=$(basename "$run")
  rm -rf "$DST/$name"; mkdir -p "$DST/$name"
  for f in summary.csv resolved_config.json configs sweep_curves rows; do
    [ -e "$run/$f" ] && cp -r "$run/$f" "$DST/$name/"
  done
  [ -f "$SRC/$name.log" ] && gzip -9c "$SRC/$name.log" > "$DST/$name/run.log.gz"
done
if [ -n "${FULLMODEL:-}" ]; then
  mkdir -p "$DST/fullmodel"
  cp "$FULLMODEL/fullmodel_runs.txt" "$DST/fullmodel/" 2>/dev/null || true
  for r in "$FULLMODEL"/*/; do
    n=$(basename "$r"); mkdir -p "$DST/fullmodel/$n"
    for f in summary.json summary.txt manifest.json inputs/manifest.json; do
      [ -f "$r/$f" ] || continue
      case $f in
        *.json) gzip -9c "$r/$f" > "$DST/fullmodel/$n/$(echo $f | tr / _).gz" ;;   # per-tensor errors; sha256 of every parameter tensor
        *) cp "$r/$f" "$DST/fullmodel/$n/" ;;
      esac
    done
  done
fi
python3 verification/paper_runs/make_tables.py --results "$SRC" ${FULLMODEL:+--fullmodel "$FULLMODEL"} --out "$DST"
du -sh "$DST"
