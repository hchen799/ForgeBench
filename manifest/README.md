# ForgeBench design manifest

Quick lookup of what the paper's results are made of: every design behind a table or figure, what was run on it, and
where its reports are. The generator itself is the primary artifact (see the top-level README); this folder indexes the
designs it produced for the paper.

## Sweep designs: `designs/{gemm,conv,llm}.csv`

One row per design (Table 4; Figs 6-8, Table 5). Columns, in order:

| Column | Meaning |
|---|---|
| `design_id` | `gemm_0001`, `conv_0001`, `llm_0001`, ... Opaque and frozen for the release. |
| *sweep parameters* | One column per parameter that varies in that sweep (see the `SWEEP` dict in `<domain>/auto_generate_json.py`). Parameters with a single value are in `designs/fixed_parameters.csv`. |
| `generated` | `YES` if the design's HLS source was generated (it is regenerable from `config_path`). |
| `csynth` | `YES` (C-synthesis report present), `NO` (not run), `FAIL`. |
| `impl` | `YES` (place-and-route report present), `NO` (not in the implemented subset / not run), `FAIL`. |
| `config_path` | The JSON config that generates the design: `configs/<domain>/<design_id>.json`. |
| `csynth_report`, `impl_report`, `impl_power_report` | `reports/<domain>/<design_id>/{csynth.xml, export_impl.xml, power_routed.rpt}`; empty unless that stage is `YES`. |
| `over_capacity` | `YES` if the csynth estimate exceeds any ZCU102 limit (BRAM_18K 1,824; DSP 2,520; LUT 274,080; FF 548,160). The report shows which resource. Empty if csynth was not run. |
| `fail_reason` | Only for `FAIL` rows, prefixed with the stage, e.g. `impl: Vivado 2024.1.2 segfault in opt_design ...`. |

Paths are relative to the extracted release bundle: extract every `configs_*.tar.gz` and `reports_*.tar.gz` next to
each other and every path in these files resolves. (The archives exist only because Zenodo limits files per record.)
Metrics (LUT, DSP, cycles, power, Fmax, ...) are not repeated here: they are in
`analysis/results_{csynth,impl}/metrics_<domain>.csv`, keyed by `design_id`, and are re-derivable from the reports
(`python -m analysis.collect`).

### Which designs were implemented (`impl`)
- conv (July run): all 576 no-unroll designs plus a random sample of the unrolled tiers (896 total). The original sampler was
  not preserved; the implemented set is the canonical definition.
- gemm (R2): 1,000 = all 384 designs with no unrolling plus 88 chosen at random from each of the 7 unrolled
  (unroll_M, unroll_K, unroll_N) combinations; seed 20261002 (`analysis/impl_sampler.py`, list in `gemm_impl_selection.csv`).
- llm (R2): 1,000, 250 at random for each `hd_unroll` value; seed 20261001 (`llm_impl_selection.csv`).

## Other suites

| File | Rows | What |
|---|---|---|
| `designs/ops.csv` | 69 | Operator variants verified in `verification/` (the 17 operators of Table 2; gemm 21, conv 23, llm 25 variants). `paper_table2_row` maps each to its Table 2 row; `generator_function` and `golden_function` point at the source and the NumPy reference. A bare `<op>.json` is skipped when `<op>__*.json` variants exist. |
| `designs/modular.csv` | 38 | Design directories behind Table 7: `program` (P1-P3) and `modularized_design` (the paper's "Overall"; contains the shared function(s)). `construction`: generated (config in repo) / generated (config not in repo) / manual (all 12 modularized designs are hand-written and ship without configs, testbenches or inputs). `design_path` is the directory in this repo. Reports: `reports/modular/<domain>/<name>/` holds `csynth.xml` plus one `<function>_csynth.xml` per function (Table 7's "Shared" column sums these). |
| `designs/modular_cases.csv` | 12 | The 12 rows of Table 7: `reuse_types` (the paper's tiling/functional/arithmetic marks), programs, modularized design, shared functions. |

Planned: `designs/fullmodel.csv` (Table 3: ResNet-18/34/50/101/152, LLaMA-3.1-8B) and `verif/*.csv` (one row per design x
datatype x sim level, pointing at `designs/` by id).

Only designs the paper references are registered. Whole-design configs in `<domain>/test_case_configs/` that are not part
of Table 7 are shipped as examples (Table 1), not as manifest entries.

## Regenerating

```
python manifest/rekey_results.py          # after collecting a run: map run names to design ids in analysis/results_*
python manifest/build_manifest.py         # designs/{gemm,conv,llm}.csv (add --check-disk to verify configs on disk)
python manifest/build_registry.py         # designs/{ops,modular,modular_cases}.csv
python release/assemble_reports.py        # build configs/ + reports/ trees and tarballs; checks every manifest path exists
```
`internal/id_map.csv` maps each design id to the generator's parameter-encoded name (and the R1 name where the design
existed in R1). It freezes the numbering; `build_manifest.py` refuses to run if the generator order no longer matches it.
`evidence/` holds the small logs behind `FAIL` rows.
