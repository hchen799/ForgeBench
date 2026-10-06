# CHANGELOG_R2 — generator changes and the committed results they make stale

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
  **27 conv sweep designs (tanh/gelu, all `ap_fixed_16_5`) change; their synthesis results are stale.** The committed modular/full-model
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
