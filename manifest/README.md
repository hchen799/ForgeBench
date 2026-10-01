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

## What the manifest covers (scope = what the paper references)

| Paper item | Registry | Status |
|---|---|---|
| Table 4, Figs 6-7, Table 5 (sweep suites) | `designs/{gemm,conv,llm}.csv` | gemm/conv built; llm pending sweep |
| Table 2 (per-operator verification, 17 operators) | `designs/ops.csv` | built |
| Sec. 4.4, Table 7 (modularization) | `designs/modular.csv`, `designs/modular_cases.csv` | built; csynth reports missing |
| Table 3 (full models: ResNet-18/34/50/101/152, LLaMA-3.1-8B) | `designs/fullmodel.csv` | planned; needs the exact configs (Hanqiu) |
| Table 6 (tool support), Catapult backend (Fig. 5), HLSFactory (Fig. 4c/d) | selections of sweep design ids (`tool_eval/`, `manifest/catapult_selection.csv`); Fig. 4c/d uses `modular/llm/Llama_GPT_module` | planned |

Designs the paper does not reference are not registered (they stay in the repo and in git history): the
whole-design configs in `<domain>/test_case_configs/` outside Table 7 (ResNet/VGG blocks, `attention_op_p*`,
`testing_*`, ...) and two unused modular designs (`gemm/mlp`, `gemm/diff_dims_module_large`).

## Non-sweep registries (`python manifest/build_registry.py`)

`design_id = <suite>/<domain>/<name>`. Everything is read from the repo (configs, generator dispatch, golden
dispatch, the `ROWS` table in `modular_data/parse_synth_resourc_util.py`); only the Table 2 row labels are typed.

| File | Rows | What |
|---|---|---|
| `designs/ops.csv` | 69 | Operator variants verified in `verification/` (17 distinct operators = the 17 rows of Table 2; gemm 21, conv 23, llm 25 variants). Columns: `operator`, `paper_table2_row`, `variant`, `op_func`, `op_dims`, `op_func_info`, `template`, `generator_function` (`<domain>/generate_code.py:fn`), `golden_function` (`verification/domains/<domain>.py:fn`), `input_range`, config path + hash. A bare `<op>.json` is skipped when `<op>__*.json` variants exist (10 superseded base configs). `data_type` is blank: the verification flow sets it at generation time. Table 2 lists 57 variants; this registry has 69 (see `docs/revision_r2/PAPER_VS_REPO.md`). |
| `designs/modular.csv` | 38 | Design directories under `modular_data/*/hls_files/` used by Table 7: 26 `program`s (P1-P3) and 12 `modularized_design`s (the paper's "Overall" column; each contains the shared function(s)). `construction`: generated (config in repo, 18), generated (config not in repo, 8), manual (all 12 modularized designs are hand-written; they ship without configs, testbenches or inputs). `test_cases` lists the cases using the design (`mult_op_p*` are shared by 3 cases). `csynth_report` is `missing` for all: no csynth report for any modular design exists in the repo or the R1 tarball. |
| `designs/modular_cases.csv` | 12 | The 12 rows of Table 7 (gemm 6, conv 3, llm 3): `reuse_types` (the paper's †/‡/∗ marks: tiling / functional / arithmetic), programs, modularized design, shared functions. |

`evidence/` holds small logs backing statuses that have no report (e.g. July impl failures).
