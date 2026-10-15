# CHANGELOG_R2 — generator changes and the committed results they make stale

## 2026-10-15 — Conv impl re-sampled (seeded 1,000) and re-run; full-model ResNet re-synthesis finished
* **Conv impl selection** (`analysis/impl_sampler.py conv --seed 20261014`, `manifest/conv_impl_selection.csv`): all 576 designs
  with no unrolling + 53 at random from each of the 8 unrolled (unroll_cin, unroll_cout) groups = 1,000, as gemm. Replaces the
  July 896 (sampler not preserved). All 1,000 run fresh (`analysis/run_conv_impl_r3.sh` -> `_sweeps/lean/conv_impl_r3`, 32 jobs,
  4 h timeout, 248 min): 998 ok + 2 Vivado 2024.1 segfaults in impl, both ok on retry (first-pass status kept in
  `manifest/evidence/r2_runs/conv_impl_r3_status_first_pass.csv`). All 1,000 meet timing.
  The 607 designs also in the July set reproduce their July/r2b metrics (DSP/BRAM/cycles/power identical; one design's LUT
  +0.04% and Fmax +5.5%).
* `analysis/results_impl/metrics_conv.csv`: replaced by the 1,000 (was 896). Archive `checkpoints/r2/impl_conv_r3_lean.tar.gz`
  first in `manifest/_common.py` SOURCES (all 1,000 resolve to it); `JULY_IMPL_VALID` now empty. Manifest conv impl: YES 1,000
  (289 July-only designs now NO). Table 5 / Fig 8: DNN (1,000); impl total GEMM 999 + DNN 1,000 + LLM 962 = 2,961.
* **Full-model csynth** (`_sweeps/lean/fullmodel_csynth_r2_sched`): all 10 ResNet `<32,10>` new-schedule designs ok (ResNet-152
  plain 28.9 h, -101 15.5 h, -50 6.7 h; the others 18 min-3.4 h); with the 4 Llama designs (`_sweeps/lean/fullmodel_csynth_r2`)
  14/14. New-schedule sources (`top.cpp`, `resolved_config.json`: only the `"schedule": "pipeline_inner"` key on the conv ops
  differs) copied from `_sweeps/fullmodel_r2_sched/` into `scale_models/hls_files` (user decision); `top.h`, `tb_top.cpp`, `run_hls.tcl`
  unchanged. Their hashes are the ones registered in `contracts.json`.
* **Table 3** (`analysis/table3_fullmodel.py` -> `analysis/tables_out/table3_fullmodel.{csv,tex}`): one row per synthesized project
  (14): LUT / DSP / BRAM_18K (count and % of the ZCU102), latency (cycles; undefined for Llama, run-time loop bounds), C-sim
  mismatching codes vs the fixed-point reference and final-logit max abs error / relative L2 vs FP64 (min-max over seeds). Only
  `<32,10>` round/saturate designs are simulated (FP64 is the PyTorch reference, not a simulated design). All ResNets exceed the
  ZCU102 (full: 697-2,160% LUT, up to 11,645% BRAM; tiled: 133-174% BRAM, 277-544% LUT), as did the R1 `<16,5>` ResNets
  (ResNet-18 full 5.2M LUT / 5,427 BRAM18K); Llama prefill 104% LUT.

## 2026-10-14 — Table 7 stays at <32,10> (user decision), reported as (LUT, DSP, BRAM) tuples; three merged LLM designs fixed
* **Format**: Table 7 is `<32,10>` TRN/WRAP for every case, the format in which all 26 programs and all 32 merged-design x program checks
  pass (every row N/N within bound). `<16,5>` (`results_paper/A_16_5__table7r2_{pm1,merged}`, `_sweeps/lean/modular_csynth_16_5_r2`)
  is not used: `gpt_transformer_p1` 0/100 and `llama_transformer_p2` 0/100 (layer_norm variance of the residual wraps past 16, sqrt -> 0,
  divide by zero), `conv_block_op2` 78/100, `conv_block_op3` 63/100. Note: those rows carry status `ok` in summary.csv although trials
  fail; `trials_within_bound` / `n_trials` is the verdict.
* **Table 7 metric** (`modular_data/parse_synth_resourc_util.py`): ZCU102 percentages per resource instead of
  the mean of DSP% and LUT%: every cell one (LUT, DSP, BRAM) column; Shared / % Reuse / % Change print BRAM as a dot (LUT, DSP only);
  Total = element-wise sum, % Reuse and % Change per resource (`-` where the denominator resource is 0).
  R1's mean hid BRAM: DNN Blocks was already 162-259% of BRAM18K in R1 (whole 256x56x56 feature maps on chip; R1 op1 2,960, module 4,729
  BRAM18K) and is 246-628% at `<32,10>`. CSV columns are now `<column>_{LUT,DSP,BRAM}`.
* **Merged designs, manual source edits** (`modular_data/llm/hls_files/`), each to match its generated programs; found by the `<16,5>`
  merged check, all re-verified at `<32,10>` (100/100 per program) and re-synthesized (`_sweeps/lean/modular_csynth_32_10_r2`, old
  reports kept as `*.pre_castfix`; merged rows replaced in `results_paper/C_32_10__table7r2_merged`, old rows `rows_pre_castfix_*.json`):
  - `tiled_attn_module`: `1/sqrt((data_t)16)` -> constant 0.25 (16 wraps to -16 at `<16,5>`: divide by zero).
  - `attn_breakdown_module`: softmax subtracts the row max before `exp` (as the program); scale constant 0.5.
  - `Llama_GPT_module`: rms_norm `sum_sq / (data_t)DIM` -> `/ DIM` (32 wraps to 0 at `<16,5>`); attention scale constant 0.5.

## 2026-10-13 — Table 7 at <32,10>; Section 4.2 regenerated; full-model ResNet conv schedule (opt-in); Table 7 <16,5> run
* **Table 7** (`modular_data/modularization_{results.csv,table.tex}`): recomputed from all 38 designs synthesized at `<32,10>` TRN/WRAP
  (`_sweeps/lean/modular_csynth_32_10_r2`, conv_block_op3 with the PAD-0 config, LLaMA with swish). User decision: use `<32,10>` for now.
  Values are larger than the `<16,5>` R1 table (32-bit multipliers ~4 DSPs); several cells exceed 100% of the ZCU102 (csynth estimates:
  `mult_op_p1`, `diff_orders_p*`, `mult_op_module_mm` 8,192 DSP; `conv_block_op3`/`conv_block_module` 7,240/7,960 BRAM18K;
  `llama_transformer_p2`/`Llama_GPT_module` 506-523K LUT).
* **Table 7 at `<16,5>`** (TRN/WRAP, same programs/inputs/N): `verification/paper_runs/A_16_5__table7r2_pm1.json`; synthesis sources from
  `prepare_table7_synth.py --dtype "fixed<16,5>" --runs A_16_5__table7r2_pm1` (new options; defaults unchanged) -> `_sweeps/table7_16_5_r2`,
  lean reports `_sweeps/lean/modular_csynth_16_5_r2`; merged-design check `results_paper/A_16_5__table7r2_merged`.
* **Section 4.2**: conv tanh re-run collected (`analysis/results_{csynth,impl}/metrics_conv.csv`: 1,728 csynth + 256 impl rows replaced by
  id; LUT/FF +0.7-0.9%, DSP/BRAM/cycles unchanged); archives `checkpoints/r2/{csynth,impl}_conv_r2b_lean.tar.gz` listed first in
  `manifest/_common.py` SOURCES (conv names unchanged since July, first match wins); run status in `manifest/evidence/r2_runs/conv_*_r2b_status.csv`.
  `analysis/section42.py` (new) writes Figs 6-8 and Table 5 from the gemm/conv/llm metrics only; `dist_impl.py` gains the Fmax and LUT
  panels of R1 Fig. 8; `density.py` uses the R1 colours and the "DNN" label. Impl counts: GEMM 999, DNN 896, LLM 962 (2,857; R1 2,711).
* **Full-model ResNet csynth at `<32,10>` did not finish**: all 10 ResNets (`_sweeps/lean/fullmodel_csynth_r2`, stopped after ~24 h) sat
  in scheduling the 7x7 stem conv: Vitis pipelined the output-pixel loop with the K x K x C_IN x C_OUT window unrolled into one stage
  (plain: 7x7x3x64; tiled: 7x7 x 8x64 over run-time-bounded `valid_ic`/`valid_oc` loops). The 4 Llama designs finished.
  Fix (opt-in per op, `"schedule": "pipeline_inner"`; configs without it regenerate byte-identical, checked on all 10 ResNet projects):
  `conv2d` pipelines the kw loop when ci and co are fully unrolled (only the stem); `conv_tile` walks ci/co in 8 x 64 blocks with
  run-time block counts, pipelines the block loop (II=1) and sums each block's ci lanes into an `acc_t` before one update of the (acc_t)
  output tile. Accumulation is exact (acc_t), so results are unchanged by the order. Files: `scale_models/generate_code.py`,
  `scale_models/conv_template.cpp`. New sources in `_sweeps/fullmodel_r2_sched/` (not yet in `scale_models/hls_files`; the committed
  sources and results are untouched). Per-function csynth of every new/changed conv function (31 conv2d, 6 conv_tile): 3-12 min each, conv_tile II=1. Verified (`results_paper/fullmodel_r2_sched`, 10 seeds x 10 ResNets, all PASS, bit-exact; every per-seed summary line identical to the
  old sources' runs); sources registered in `contracts.json` (reference_version 2). Re-synthesis `_sweeps/lean/fullmodel_csynth_r2_sched` running.
* `contracts.json`: Llama-3-8B ctx8192 `<32,10>` prefill/decode sources registered (reference_version 2, same recipe as the verified
  ctx2048 pair) so `verify.py` accepts them; verification `results_paper/fullmodel_ctx8192_r2` (4 prefill + 2 decode, seeds 42-44): all PASS, 0 code mismatches, logits
  max abs vs FP64 <= 4.9e-6.

## 2026-10-12 — Table 7 merged designs verified against their programs; bugs fixed (manual source edits)
Files: `verification/merged_designs.py` (new), `modular_data/{gemm,conv,llm}/hls_files/*_module*/top.{cpp,h}`,
`gemm/test_case_configs/vec_mtx_p2.json`, `conv/test_case_configs/conv_block_op3.json`,
`verification/paper_runs/{C_32_10__table7r2_vecmtx_fix.json,C_32_10__table7r2_convblock3_fix.json,
C_32_10__table7r2_llamaswish.json,prepare_table7_synth.py}`, `llm/test_case_configs/llama_transformer_p2.json`.
* New check: one testbench per merged design drives the synthesized `top` with every program's inputs (each program's verified
  `<32,10>` design JSON and report seeds) and checks each program's outputs against that program's golden and bound.
* The first run found that most merged designs did not reproduce their programs. Manual edits (each restores the program's semantics):
  - `mult_op_module_dot`, `mult_op_module_mmv`, `vec_mtx_module`: in-place `transpose` / `transpose_32` overwrote half the matrix
    (`input[i][j] = input[j][i]` over the full square) -> swap across the diagonal.
  - `mult_op_module_dot`: program B stored `dot_out_B`, never written (the dot product wrote `dot_out`).
  - `mult_op_module_mm`, `mult_op_module_mmv`: program C copied 16 of the 64 vector elements into the matrix row.
  - `diff_dims_module_small`: `tiled_gemm` re-initialised the output tile with the bias for every reduction (j) tile, so only the last
    j tile was kept (P3 has one j tile and passed) -> initialise on `j_offset == 0`; `top.h` declared `top(` without `void`.
  - `tiled_attn_module`: head-chunk output copied with row/column swapped (`[c][i*64+r]` for r<8, c<64) -> `[r][i*64+c]`.
  - `attn_breakdown_module`: softmax looped `j < DIM` (32) over an 8x8 score row -> `j < SEQ_LENGTH`.
  - `Llama_GPT_module`: stochastic dropout -> identity (as the generator); `top`'s second LLaMA rms_norm used the first norm's
    weights; GPT `gelu` called `hls::tanh` on negative arguments (Vitis bug) -> the generator's odd-symmetry form.
* LLaMA MLP activation: `llama_transformer_p2.json` had been switched swish -> relu in d193924 (2025-03-17, with the MLP
  restructured to `(act(x W1) * x W1) W2`); a77dfbb first changed the merged design to relu to match. User decision: swish.
  The config's activation and the merged design's function are swish again (the generator's swish code; MLP structure kept).
  Re-verified (`results_paper/C_32_10__table7r2_llamaswish`, merged row replaced; relu row kept as
  `rows_relu_superseded_Llama_GPT_module.json`).
  - `activation_module`: shared `compute_exp` now uses the lookup-table exp (`backends/exp_table.py`), as its programs do.
  - `conv_block_module`: the fixes of 9477692 (made then only in the standalone `conv_block_op1-3`) were missing: `MAX_LOCAL_SIZE` 9 -> 16
    (14-row tiles), program C's 64->64 3x3 conv loaded its weights with K=1 (first tap only); plus the same `acc_t`/`math_t`
    accumulator and batchnorm types.
* Program config bug: `vec_mtx_p2.json` loaded 16 of the 64 vector elements (`load_2` dims [16]); the program and its golden both
  followed the config, so it passed while computing a truncated product. Fixed ([64]) and re-verified (100/100,
  `results_paper/C_32_10__table7r2_vecmtx_fix`). No other Table 7 config has a partial load/store.
* Program config bug: `conv_block_op3.json` (copied from `conv/test_case_configs/resnet50_block1.json`) gave both 1x1 convs
  PAD=1 with H_OUT=W_OUT=56, i.e. the output shifted by one pixel (cropped from 58x58). Program and golden both followed the config;
  the merged design (unpadded, as ResNet-50) did not match it. Fixed (PAD 0) and re-verified (`results_paper/C_32_10__table7r2_convblock3_fix`).
  The same PAD=1 1x1 convs are in all ten `resnet50_block*.json`; `resnet18_block{2,3}_downsample.json` declare an H_OUT that does
  not match K/PAD/STRIDE. None of these is used by a paper run; left unchanged.
* Stale: Table 7 synthesis of `vec_mtx_p2`, `conv_block_op3`, `llama_transformer_p2` and of every merged design (re-run from the fixed sources).

## 2026-10-12 — Catapult backend check extended to vmm, mmv, dot product
Files: `gemm/generate_code.py`, `verification/paper_runs/{C_32_10,D_float}__catapult_pm1.json`, `make_tables.py`.
* The accumulator initialisation of vmm, mmv, dot (and the inline gemm/mmv/vmm/dot call variants) now goes through the backend's `zero()`
  (Vitis `0`, Catapult `data_t(0)`; `ac_ieee_float` has no assignment from int). Vitis: `verification/regression_check.py`, 0 of 281 changed.
* Catapult SCVerify C sim, ±1, N=100, C and D: 16/16 variant×format pass, every trial identical to Vitis per seed (`verif_catapult.tex`).
* Stale: nothing.

## 2026-10-12 — lookup-table exp option (`exp_impl`)
Files: `backends/exp_table.py` (new), `{gemm,conv,llm}/generate_code.py` (hook in `generate_func_def` / call sites).
* An op may carry `"exp_impl": "table"` (default `"hls"`). The op's function is renamed `<name>_exptable` and its `hls::exp` / `hls::tanh`
  calls go through a 4,097-entry ROM of exp(-t), t in [0,16), 1/256 grid with linear interpolation (the `fb_exp_negative` scheme of
  `scale_models/production_types.py`); exp(x>0) = 1/exp(-x); tanh from exp(-2|x|). Ops: activation (sigmoid, tanh, elu, selu, gelu,
  swish, softmax), mha, swa; any other op with the option is an error. Generic user-defined tables: future work.
* Default designs are byte-identical (regression check: 0 of 273 sampled designs changed).
* Verification: `verification/paper_runs/run_exptable.sh` (the 20 Table-2 variants that use exp/tanh, columns A-D, +-1 and max range, N=100).
* Stale: nothing.

## 2026-10-11 — Catapult backend: SCVerify testbench, ac_ieee_float, FPGA target (Vitis output unchanged)
Files: `backends/base.py`, `backends/catapult.py`, `gemm/generate_code.py`; commits 8d69820, 9a2d9c5.
* New backend hooks `tb_cast`, `zero`, `tb_main`, `tb_call`, `tb_return` (Vitis defaults reproduce the old text: `verification/regression_check.py`,
  0 of 273 designs / testbenches changed).
* Catapult: float -> `ac_ieee_float<binary32>`; testbench via SCVerify macros and `fb_to_double`; GEMM accumulator init `data_t(0)`; script
  runs SCVerify C sim / `go extract` / SCVerify RTL sim per task; default library `mgc_Xilinx-ZYNQ-uplus-2_beh` for `xczu9eg-ffvb1156-2-e`
  (was Nangate 45 nm).
* Stale: nothing (no committed result used the Catapult backend before).

## 2026-10-11 — LLM sweep re-run started
csynth 3,888 + impl 1,000 (`analysis/run_llm_r2b.sh`, lean dirs `_sweeps/lean/llm_{csynth,impl}_r2b`); replaces the results marked stale on
2026-10-05 once collected. The stale lean trees `_sweeps/lean/llm_{csynth,impl}` are kept.

## 2026-10-01 — LLM sweep: dropout replaced by head-dim unroll knob; weight-shape fix; naming

Files: `llm/auto_generate_json.py`, `llm/generate_code.py`, `llm/grouped_mha_rope_template.cpp`.

1. **Dropout removed from the sweep.** Dropout is an identity at inference (fix 0f6ca97), so the
   {off, p=0.1, p=0.3, p=0.5} setting did not change the design (p=0.1 and p=0.3 had identical csynth
   metrics in all 972 groups; p=0.5 differed slightly). Replaced by `hd_unroll` ∈ {1,2,4,8}: unroll
   factor of the two head-dim loops of the attention kernel plus cyclic partitioning of Q/K/V along the
   feature dim. Sweep size unchanged: 3,888. Legacy path preserved: `func_info` without a third element
   emits no pragmas, so `llm/test_case_configs` and `verification/` variants regenerate byte-identically
   (checked on the 7 `test_case_configs`).
2. **Weight shape bug fixed.** `W_q/W_k/W_v` were declared `[DIM_IN][DIM_IN]` (BRAM and DRAM) but the
   kernel indexes `[DIM_OUT][DIM_IN]`, `DIM_OUT = heads*head_dim`. For 17 of 27 (dim_in, heads, head_dim)
   shape combos (`DIM_OUT > DIM_IN`) the kernel read out of bounds (AddressSanitizer:
   `global-buffer-overflow` at the `W_q` read); 4 more were over-allocated; only 6 were consistent.
   After the fix all 27 shapes are ASan-clean in g++ CSIM. This changes weight BRAM/DRAM sizes and the
   emitted `load_*` calls.
3. **Config names** are now `ATTN_config_S{seq}_D{dim}_H{heads}_HD{hd}_G{groups}_{norm}_ROPE{bool}_UHD{u}_{dtype}`
   (previously two inconsistent schemes depending on dropout).
4. `{gemm,conv,llm}/auto_generate_json.py` now expose a module-level `SWEEP` dict (single source of
   truth for the manifest and Table 4). GEMM/conv emitted configs are byte-identical to before (12 × all
   3,840 + 5,184 configs diffed).

**Stale as of this change (must be regenerated):** all LLM csynth results
(`analysis/results_csynth/metrics_llm.csv`, `checkpoints/20260720/csynth_llm_lean*`), all LLM impl results
(`analysis/results_impl/metrics_llm.csv`, LLM share of `impl1000_reports_lean`), every figure/table that
uses them (Figs 6–8, Table 5, counts in §4.2), and any LLM entry in the verification whole-design runs.
GEMM and conv results are unaffected.

## 2026-10-01 — GEMM sweep: per-op loop orders (set up, NOT yet run)

Files: `gemm/auto_generate_json.py` (`SWEEP`, `iter_params`, `config_stem`, `legacy_stem`, `generate_config_text(vm_orders=...)`).

The R1 sweep crossed all 6 loop-order permutations of (i,j,k) with every computation option. Only options 2
and 3 contain a `gemm` op (the only op that uses the 3-loop order); options 1, 4, 5 contain only
`vmm`/`mmv` + `dot_product`, which use just the i/j subsequence of the order. So for options 1/4/5 the 6
permutations collapsed to 2 distinct designs: 1,536 of 3,840 designs were byte-identical copies
(768 groups of 3, identical csynth metrics in all 768 groups).

New sweep (3,072 designs): options 2/3 keep the 6 gemm orders (their single vm op's order is derived, as
before); options 1/4/5 let each of their two vm ops independently pick `ij` or `ji` (4 combinations; ij vs ji
changes csynth results in 88–100% of matched pairs). Sweep total 3,072 + 5,184 + 3,888 = 12,144.

* New names: `..._UN{u}_GORD{ijk|na}_VM1{ij|ji}_VM2{ij|ji|na}_BIAS_...`. Manifest column `legacy_design_id`
  gives the R1 id. 2,304 designs map to R1 designs and generate byte-identical configs (checked: 0 differences
  against the R1 config files); 768 are new (the mixed (ij,ji)/(ji,ij) combinations of options 1/4/5).
* The 1,536 R1 designs that no longer exist are exactly the manifest's former `duplicate_of` rows.
* **To run (not yet run):** csynth of the 768 new designs (`manifest/designs/gemm.csv`, `csynth_status=pending`).
* **Impl:** the July GEMM impl sample (820) covers 628 distinct sources: 500 are kept designs directly, 128
  more kept designs have an impl result only through a byte-identical R1 copy (not adopted automatically).
  No impl selection has been made for the 768 new designs.
* Stale: nothing. Existing GEMM csynth/impl results stay valid for the 2,304 mapped designs; the metrics CSVs
  are still keyed by R1 ids until the 768 new results are collected and re-keyed.
* **Decision (2026-10-01):** after the LLM sweeps finish, GEMM impl is re-sampled and re-run (the July sample no
  longer matches the 3,072-design sweep).

## 2026-10-02 — R2 csynth complete for all sweeps
* GEMM: the 768 new designs synthesized (768/768 ok, 21 min at 80 jobs). `analysis/results_csynth/metrics_gemm.csv` is now keyed
  by the new design ids: 2,304 July rows re-keyed (R1 ids -> new ids; July values unchanged) + 768 new rows; the 1,536 rows of
  R1 duplicate designs were dropped from it (they remain in git history and in `checkpoints/20260720/csynth_gemm_lean.tar.gz`).
* LLM: csynth 3,888/3,888 ok; impl 999 ok + 1 Vivado 2024.1.2 opt_design segfault (reproduced on retry). 96 of 3,888 exceed ZCU102 BRAM
  (weight-shape fix enlarged the weight BRAMs).
* Modular: csynth 38/38 ok after rerunning `gemm/diff_orders_module` with an 8 h limit (it took 7,921 s). Table 7 recomputed from the
  new reports: 11 of 12 rows identical to R1; 'DNN Blocks' P2 (conv_block_op2) is 43.82 vs R1 21.83 (restoring the commented-out bias
  array_partition pragma does not change it: 43.95), so Total_Before 108.49 -> 130.48 and % Change -20.25 -> -33.69. Open.

## 2026-10-05 — R2 impl complete for all domains
* GEMM impl re-sampled (1,000 designs: all 384 no-unroll + 88 per unrolled (unroll_M, unroll_K, unroll_N) group, seed 20261002) and re-run
  (64 jobs, 161 min). First pass 993 ok / 7 fail; all 7 failures were the Vivado 2024.1.2 `opt_design` segfault. A single retry succeeded for
  6 of them (the crash is intermittent), 1 failed again -> final 999 ok / 1 fail. First-pass statuses are kept in
  `manifest/evidence/r2_runs/gemm_impl_status_first_pass.csv`. The July GEMM impl metrics are in `legacy/results_r1/`.
* Implemented subsets now: conv 896 (July sample, unchanged sweep), gemm 999 (+1 fail), llm 999 (+1 fail) = 2,894 designs, all meeting timing at 100 MHz.

## 2026-10-05 — activation functions in rank-2 templates; testbench precision (generator changes for verification)
Files: `gemm/2D_activations_template.cpp`, `llm/activation_template.cpp`, `{gemm,conv,llm}/generate_code.py` (`generate_activation_function`,
`generate_testbench_code`), `verification/gen_variants.py`, `verification/regression_check.py` (new).
1. **`hardsigmoid` and `hardswish` added to the rank-2 (gemm, llm) activation templates.** Table 2 claims 15 activation functions x 2 tensor ranks;
   only the conv (rank-3) template had all 15, the rank-2 templates had 13. Same definitions as the conv template (hardsigmoid = 0 for x<=-3, 1 for x>=3,
   else (x+3)/6; hardswish = x*hardsigmoid(x)). Additive: existing designs are unchanged (regression check: 0 of 279 sampled designs changed
   `top.cpp`/`top.h`/`run_hls.tcl`; the template header comment was deliberately left alone because it is copied into emitted `top.cpp`).
   Verification variants: 4 new configs (`activation__hardsigmoid|hardswish` for gemm and llm); Table 2 selection (`ops.csv: in_table2`) = 57.
2. **Generated testbench precision (all three domains).** Outputs were dumped with `fprintf("%f ", (float)v)` (6 decimals, float32) and inputs read
   with `float temp; fscanf("%f")`; neither can resolve wide fixed-point types (`<32,10>` step 2^-22 ~ 2.4e-7), so fixed-point comparisons through
   the stock testbench lost precision. Now: inputs read as `double` (`%lf`), outputs dumped as `(double)` with `%.17g` (exact for any ap_fixed up to 53 bits
   and for float). Testbench only: `tb_top.cpp` of every design changes (279 of 279 sampled), the synthesized files do not, so no csynth/impl result is affected.

## 2026-10-05 — generic fixed-point spelling with explicit rounding/overflow modes (mechanism only; sweep results unchanged)
Files: `backends/{base,vitis,catapult}.py` (`parse_fixed`), `{conv,llm}/gen_configs.py` (lower the type at config load),
`gemm/generate_code.py` (testbench typedef goes through the backend), `verification/tests/test_data_types.py`.
* New generic spelling `fixed<W,I[,quant[,overflow]]>` (quant: trn|rnd, overflow: wrap|sat). A generic `fixed<W,I>` means **round-to-nearest +
  saturate** (`AP_RND, AP_SAT` / `AC_RND, AC_SAT`), consistent with the full-model designs. Unsupported modes raise.
* Raw tool spellings are untouched: `ap_*` strings pass through the Vitis backend verbatim, and mode-less `ap_fixed<W,I>` keeps the tool defaults
  (truncate + wrap; Catapult `AC_TRN, AC_WRAP`). Regression check: 0 of 279 sampled designs changed `top.cpp`/`top.h`/`run_hls.tcl`.
* Not yet applied to the sweeps: the three `SWEEP` dicts still carry `ap_fixed<16,5>` (truncate + wrap), so existing csynth/impl results remain
  valid for what they are. Switching the sweeps to the generic default changes every design's arithmetic and requires re-running everything.

## 2026-10-05 (later) — correction: default fixed-point modes stay truncate + wrap; no math shim
The generic `fixed<W,I>` spelling added earlier the same day was first defined to default to round + saturate. That was reverted: Vitis `hls_math.h`
only supports fixed-point `exp/sqrt/tanh/sin/cos` for the default modes (see docs/DATA_TYPES.md), so a round+saturate default would break every
design that uses them. Final behaviour: a spelling without modes (`fixed<W,I>` or `ap_fixed<W,I>`) is truncate + wrap everywhere; round/saturate must be
requested explicitly (`fixed<W,I,rnd,sat>`) and then inherits Vitis' limitation. A C++ compatibility shim that made the math calls compile for other modes was
prototyped and removed. The paper should state that the full-scale models use RND/SAT and everything else the tool default. Regression check: 0 of 279 sampled
designs changed `top.cpp`/`top.h`/`run_hls.tcl`.

## 2026-10-05 (later) — activation templates: leaky_relu, prelu, gelu now build in fixed-point
Files: `gemm/2D_activations_template.cpp`, `conv/activations_template.cpp`, `llm/activation_template.cpp`.
Found by the fixed-point operator pilot: these three activations did not compile for *any* `ap_fixed` type (not a Vitis math limitation).
* `leaky_relu`, `prelu`: `cond ? input : alpha * input` mixed `ap_fixed<W,I>` with the wider product type (ambiguous conditional); the else branch is now cast
  `(data_t)(alpha * input)`.
* `gelu` (gemm, conv): `0.044715 * x_cube` and `0.5 * x * ...` multiplied a `double` literal with `ap_fixed` (ambiguous); constants are now `(data_t)`
  (the llm template already was).
Effect: the emitted `top.cpp` of the 8 sampled designs using these functions changes (regression check: 8 of 279, exactly those; no sweep or modular
design uses them). Float results are unchanged to within ~1e-7 (float verification 10/10 trials pass for all nine variants).

## 2026-10-05 (later) — dimension-derived constants no longer cast to `data_t` (LLM sweep results are STALE; re-run pending)
Files: `llm/grouped_mha_rope_template.cpp`, `llm/sliding_window_attention_template.cpp`, `llm/rms_norm_template.cpp`, `conv/adaptive_avgpool_template.cpp`,
`llm/generate_code.py` (passes `SCALE`).
Found by fixed-point operator verification: `(data_t)head_dim` / `(data_t)DIM` / `(data_t)count` do not fit `ap_fixed<16,5>` (range [-16,16): 16 -> -16,
32/64/128/... -> 0 under wrap), so the attention scale `1/sqrt((data_t)head_dim)` and RMSNorm's `/(data_t)DIM` divided by zero (CSIM: SIGFPE).
* attention (mha, swa): `const data_t scale = (data_t){SCALE};` with SCALE = repr(1/sqrt(HEAD_DIM)) computed at generation time.
* rmsnorm: `sum_sq / {DIM}` (integer divisor, as layernorm already did). adaptive_avgpool: `sum / count` (integer).
* Effect on the sweeps: **every LLM sweep design changes** (attention scale; half also RMSNorm); GEMM and conv sweep designs are byte-identical (regression
  check: 0 of the sampled gemm/conv sweep designs changed). Measured on one design (S16 D256 H16 HD32 G2, RMSNorm): as generated BRAM 179 / DSP 88 /
  FF 38,406 / LUT 38,426 / 2.59M cycles; fixed BRAM 435 / DSP 88 / FF 43,195 / LUT 44,035 / 2.46M cycles -- the broken constant let the tool optimize away
  part of the attention, so the previous LLM PPA was for a functionally invalid design.
* **Stale until re-run:** all LLM csynth (3,888) and LLM impl (1,000), Table 5's LLM column, the LLM parts of Figs 6-8 and the counts that use them.
  Decision (2026-10-05): do not re-run until all operator verification is finished (more template bugs may be found), then re-run once.
* Also affected, not changed here: hand-written modular LLM designs and `scale_models/` (own template copies) may contain the same constants.

## Per-operator accumulator parameters (verification phase)

Operators accept `acc_type` (e.g. `fixed<32,10>`, `float`), `acc_rounding` (`trn|rnd`) and `acc_overflow` (`wrap|sat`) in their op
JSON; unset means the design's `data_t` (generated text is byte-identical to before: `verification/regression_check.py` reports 0 changed
files). `data_t` stays design-wide. Implemented for layernorm, rmsnorm, batchnorm (epsilon/denominator in the accumulator type),
adaptive avgpool, llm matmul (per-row accumulator buffer), mha and swa (Q/K/V projections, scores, softmax sum, context), conv
(per-pixel accumulator buffer, dense and grouped) and the Python-built gemm/mmv/vmm/dot families (local `acc_out` copy, any loop order).
The verification engine takes an `op_params` block that is applied to the operator under test.
Effect check (`fixed<16,5>` storage, `fixed<32,10>` RND accumulator, N=10): layernorm/rmsnorm/matmul/conv/avgpool/gemm/dot max abs
error about 5e-4 (1 LSB); mha/swa 0.008-0.022 (range_hi 0.56-0.94).
Open: mmv/vmm max abs error (0.018 / 0.005) is identical with and without the accumulator, so it does not come from accumulation; not yet explained.
Stale: all LLM sweep results (unchanged from the earlier note).

## Per-operator `data_type`; tanh sign fix; accumulator fixes (verification phase, continued)

* **Per-operator `data_type`** (`backends/op_types.py`): an op JSON may carry `data_type` (same spellings as the design-wide one; unset =
  design `data_t`). The operator is generated with that storage type (accumulator defaults to it unless `acc_*` fields are set) and the call
  site converts operands at the boundary (copy-in of inputs, copy-out of the `output`/`out` operand). Typed gemm-family operators are emitted as functions
  (the inline form works on the design-typed BRAMs). Default designs are byte-identical. Checked: mha in `double` inside a float design passes
  to range 741 at ~1e-6 (kernel is exact; the float32 window end is float32 conditioning of the scores); a `fixed<32,10>` operator inside a
  `fixed<16,5>` design verifies for every operator family.
* **Bug found by verification: Vitis HLS 2024.1 `hls::tanh(ap_fixed)` returns tanh(|x|) for negative inputs** (reproduced with a bare call,
  `ap_fixed<32,10>`/`<16,5>`/`<32,8>`). The generated tanh and gelu (tanh inside) now use odd symmetry
  `x<0 ? -tanh(-x) : tanh(x)`. Positive-side library accuracy is 1-4 LSB; for `<32,10>` it breaks beyond |x|~24 (error up to 2.0 at 27.6).
  **27 conv sweep designs (tanh/gelu, all `ap_fixed_16_5`) change; their synthesis results are stale.** *(Correction 2026-10-12: 27 was the count in the 279-design regression sample. Every conv sweep design with `activation=tanh` changes: 1,728 of 5,184 csynth results and 256 of the 896 implemented conv designs are stale; checked against an archived July `top.cpp`, which calls `hls::tanh` directly.)* The committed modular/full-model
  designs that contain gelu (`llm/hls_files/*`, conv `hls_files`, `scale_models/`) carry the same defect for negative gelu arguments; not
  touched (scale_models waits on Hanqiu).
* Inline operators were missing the accumulator spec (`_set_acc` is now also called when generating calls). The earlier mmv/vmm error
  (0.018/0.005 at `<16,5>`) was truncation bias of a `data_t` accumulator over 64 terms (~0.5 LSB per add), not a kernel bug; with a wide
  accumulator it is 1 LSB.
* softmax sums use `{ACC}` in all three domains. llm/conv softmax have no max-subtraction (gemm has), so at `<16,5>` the sum of 128 exps
  overflows for every symmetric input range ("no faithful range") until an accumulator type is set (then the window is ln 16 ~ 2.8, exp storage).
* Window ends that are overflow, not bugs: gelu `<32,10>` ~7.9 (x^3 reaches 512); mha/swa limited by Q/K scores overflow.

## Softmax max-subtraction in the accumulator type; `fixed_ranges` in the verification engine

* With an accumulator spec, `scores - max` (mha, swa, gemm softmax) is computed in the accumulator type and `exp` is called on its
  default-mode counterpart (`ACCM`); the difference spans up to twice the score range and wrapped in `data_t` (first failure of mha at
  `<16,5>`, weights +-0.1, was this subtraction at |x|~3). Default designs byte-identical.
* After the fix mha (weights +-0.1, `fixed<32,10>` RND accumulator) fails at |x|~3.4-3.6; a float64 stage model puts the stored
  `scores` (a `data_t` array) past 16 at |x|~3.4 (max |scores| 14.6 at 3.2, 18.4 at 3.4), so the remaining limit is the storage format,
  addressed by a per-operator `data_type`, not by the accumulator. swa reaches ~11.
* Engine option `fixed_ranges` ({"weights": 0.1}): DRAMs whose name contains the key keep +-value while the range search scales the others.

## conv / llm softmax subtract the max (now consistent with gemm)

`conv/activations_template.cpp` (per spatial location, over channels) and `llm/activation_template.cpp` (per row) subtract the max before
`exp`, as the gemm softmax and the attention softmax already did (mathematically identical; every exp argument is <= 0, so exp cannot
overflow in fixed point). With an accumulator spec the subtraction is done in the accumulator type. Regression check: only softmax designs
change (2 of the sampled designs); **conv/llm softmax design results are stale.**
Effect at `<16,5>` (default `data_t` everywhere): llm softmax went from "no faithful range" to a window that covers the whole search range (23.2),
conv softmax from about 0.9 to 23.2 (max abs error 1.7e-3 and 7.8e-4 for conv and llm).
Storage `fixed<24,8>` + `fixed<32,10>` RND accumulator inside a `<16,5>` design (weights +-0.1): mha window 7.2, swa 23.2 (full search range).
