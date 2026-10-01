# ForgeBench design manifest

One row per generated design. A design is a JSON config plus the generator that turns it into HLS
source; the manifest records every sweep design, what happened to it at each stage, and where its
reports are in the release bundle. Results themselves (LUT/DSP/cycles/power/...) live in
`analysis/results_{csynth,impl}/metrics_<domain>.csv`, keyed by `design_id`.

    python manifest/build_manifest.py            # regenerate manifest/designs/*.csv (deterministic, ~15 s)
    python manifest/build_manifest.py --check-disk   # also verify config hashes against */auto_generated_configs

Parameters, names and config text come from `{gemm,conv,llm}/auto_generate_json.py`
(`iter_params`, `config_stem`, `build_config_text`), so the manifest cannot drift from the generator.

## `designs/{gemm,conv,llm}.csv` (sweep suites)

| Column group | Columns |
|---|---|
| identity | `design_id` (= config file stem = design directory name), `legacy_design_id` (id in the R1 sweep; its R1 results/reports are filed under it; blank if none), `domain`, `suite` (`sweep`) |
| parameters | one column per sweep parameter (see `SWEEP` in the generator), then `data_type` |
| files | `config_path` (repo-relative; also inside `configs/sweep_configs_<domain>.tar.gz`), `config_sha256`, `source_sha256` (hash of emitted `top.cpp`) |
| duplicates | `duplicate_of`: `design_id` of the first design with byte-identical `top.cpp`; empty if unique |
| stages | `generated`, `csynth_status`, `csynth_fail_reason`, `over_capacity`, `over_capacity_resources`, `impl_selected`, `impl_status`, `impl_fail_reason` |
| reports | `csynth_report_archive` + `csynth_report_path`, `impl_report_archive` + `impl_report_path`, `impl_power_report_path` (archive = file under `reports/` in the bundle; path = member inside it) |

Status values: `csynth_status` ∈ ok / fail / timeout / invalid_report / pending (not yet run);
`impl_status` ∈ ok / fail / timeout / not_selected / pending. `impl_selected` is true if the design was
chosen for place-and-route (the rule is documented per domain below).

`over_capacity` is a post-synthesis (csynth estimate) check against the ZCU102 (xczu9eg) totals:
BRAM_18K 1824, DSP 2520, LUT 274,080, FF 548,160; `over_capacity_resources` lists the offenders.

### Impl selection rule
- gemm, conv (July run): tiered by unroll factor. All no-unroll designs plus a random sample of each
  unrolled tier (gemm 820, conv 896). The original sampler is not in the repo; the selected set is
  reconstructed from the July results and is canonical (`impl_selected`).
- llm (R2): 1,000 of 3,888, 250 at random for each `hd_unroll` value, seed 20261001
  (`analysis/impl_sampler.py`).

### Duplicates
`duplicate_of` flags designs whose emitted `top.cpp` is byte-identical to an earlier design (they would be
separate designs with the same hardware). The R1 GEMM sweep had 1,536 such copies (loop order crossed with
options that have no gemm op); the R2 GEMM sweep removes them (see `docs/revision_r2/CHANGELOG_R2.md`), so
the expected value of `duplicate_of` is empty everywhere. conv: none. llm: all 3,888 sources are unique
(checked by generating them).

### Current state
- conv: complete (csynth 5,184/5,184; impl 896 selected, all ok). One design's July report was missing; it was
  re-run in R2 (`reports/csynth_conv_rerun_lean.tar.gz`).
- gemm: 3,072 designs (per-op loop orders, see changelog). 2,304 have July csynth results (via `legacy_design_id`);
  768 are `pending` (new designs). impl: 500 selected-and-run (499 ok, 1 Vivado segfault); no selection yet for the new 768.
- llm: all rows `pending` until the R2 sweeps (dropout replaced by `hd_unroll`, weight-shape fix) are collected;
  `impl_selected` is filled from `manifest/llm_impl_selection.csv`.

## Planned
`designs/{ops,cases,modular,fullmodel}.csv` (same core columns, suite-specific parameters) and
`verif/*.csv`: one row per `design_id` x datatype x sim level, pointing at `designs/` by `design_id`.

`evidence/` holds small logs backing statuses that have no report (e.g. July impl failures).
