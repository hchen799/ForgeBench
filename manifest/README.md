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

## Non-sweep registries (`python manifest/build_registry.py`)

`design_id = <suite>/<domain>/<name>`. A design is in exactly one file; modular designs that also have a
`test_case_configs` config are listed under `modular`, not `cases`. Everything is read from the repo
(configs, generator dispatch, golden dispatch, the `ROWS` table in `modular_data/parse_synth_resourc_util.py`).

| File | Rows | What |
|---|---|---|
| `designs/ops.csv` | 69 | Operator variants verified in `verification/` (17 distinct operators: gemm 21, conv 23, llm 25 variants). Columns: `operator`, `variant`, `op_func`, `op_dims`, `op_func_info`, `template`, `generator_function` (`<domain>/generate_code.py:fn`), `golden_function` (`verification/domains/<domain>.py:fn`), `input_range`, config path + hash. A bare `<op>.json` is skipped when `<op>__*.json` variants exist (10 superseded base configs). `data_type` is blank: the verification flow sets it at generation time. |
| `designs/cases.csv` | 32 | Whole-design configs in `<domain>/test_case_configs/` that are not modular (conv 27 incl. ResNet-18/50 and VGG-19 blocks, gemm 2, llm 3). `runnable=False` marks `conv_variable.json`, a symbolic template whose dims are variable names. `hls_dir` points at a generated design if one is tracked. |
| `designs/modular.csv` | 40 | Every design directory under `modular_data/*/hls_files/`. `role`: program / shared_module / unused (`gemm/mlp`, `gemm/diff_dims_module_large` are in no test case). `construction`: generated (config in repo), generated (config not in repo), or manual (all 12 shared modules are hand-written; they ship without configs, testbenches or inputs). `test_cases` lists the cases using the design (the `mult_op_p*` programs are shared by 3 cases). `csynth_report` is `missing` for all 40: no csynth report for any modular design exists in the repo or the R1 tarball. |
| `designs/modular_cases.csv` | 12 | The modularization test cases (6 gemm, 3 conv, 3 llm): programs, shared module, shared functions. Note the paper says 13 test cases; the table behind it (`modular_data/modularization_results.csv`) has 12 rows. |

`cases` + modular-with-config = 51 = all files in the three `test_case_configs/` dirs.

## Planned
`designs/fullmodel.csv` (ResNet-18..152, LLaMA-3.1-8B, reduced e2e models) and `verif/*.csv`: one row per
`design_id` x datatype x sim level, pointing at `designs/` by `design_id`.
`evidence/` holds small logs backing statuses that have no report (e.g. July impl failures).
