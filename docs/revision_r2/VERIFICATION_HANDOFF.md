# Verification: status and plan (hand-off, 2026-10-07)

Branch `FPT_TRETS_2026`. Deadlines: code/results freeze 2026-10-21, resubmission 2026-10-30.
Standing rules: do not touch `scale_models/` layout until the consolidation is agreed; the LLM sweep re-run waits until verification is finished;
the user pushes (the agent only commits locally).

## 1. Operator-level verification (our side) — done / in place

**Engine** (`verification/functional_verification.py`): float64 golden on the quantized inputs; faithful-range search (N=20 samples per step), N=100 confirmation
with step-down, a one-step safety margin (`range_search.margin_steps`, x1/1.19), then N=100 reported trials on fresh seeds (`report_seed_offset`).
Tolerance `format_bound` = c*LSB*(L+stages) + peak_fraction. Float: PASS/FAIL; fixed point: characterization (max abs error headline, SQNR secondary).
Added this round: numeric `inputs.range`, `fixed_ranges_by_operator` (accepts +-mag or `[lo,hi]`), `range_search.caps` (per-variant range cap), `op_params`.

**Generator features it relies on** (defaults byte-identical, checked by `verification/regression_check.py`, 273 designs):
- per-operator accumulator: `acc_type`, `acc_rounding`, `acc_overflow`;
- per-operator `data_type` (unset = design-wide), with boundary conversion (`backends/op_types.py`);
- softmax/mha/swa subtract the max in the accumulator type; conv/llm softmax now subtract the max (as gemm);
- tanh/gelu odd-symmetry workaround for the Vitis `hls::tanh(ap_fixed)` negative-input bug.

**Paper runs** (`verification/paper_runs/`, outputs `verification/results_paper/`, not tracked): 4 columns x {largest range, +-1}, CSIM only, N=100, 57 Table-2 variants:
(A) `<16,5>` all; (B) `<16,5>` I/O + `<24,8>` op storage + `<32,10>` RND acc; (C) `<32,10>` all; (D) float.
`make_tables.py` -> `verif_ops.tex` (`tab:verif-range`, `tab:verif-pm1`); `verif_section.tex` is the draft text (error-bound sentence is a placeholder for the authors).
Settings: weights of matmul/mha/swa/conv fixed at +-0.1; batchnorm parameters in [0.25, 1] (variance floor); tanh range capped (16 in B, 24 in C).
Run with `run_all.sh`. NOTE: the B gemm-tanh row and the batchnorm rows of the largest-range summaries were patched from single-design re-runs (`results_paper/*_fix*`);
re-run the full largest-range pass if the paper numbers should come from one command (~1-1.5 h).

**Known findings / caveats**
- Vitis `hls::tanh` (fixed point): `<24,8>` crashes (SIGFPE) at isolated inputs from |x|~20; `<32,10>` returns the wrong sign at isolated inputs from |x|~26. Caps are a workaround, not a fix.
- `<16,5>` at +-1: gemm softmax (4/100 within bound), layernorm and rmsnorm (99/100) are flagged; mha and conv are fine with +-0.1 weights.
- Absolute errors at huge ranges (e.g. `<32,10>` leaky_relu ~600) are dominated by parameter quantization times magnitude; consider a relative-error column.
- Window edges are statistically soft; the margin step addresses it. A corner-input check (all +-r) was proposed but not implemented.
- Stale / not touched: committed gelu designs in `llm/hls_files/*`, conv `hls_files`, `scale_models/` (tanh negative-argument defect); 27 conv tanh/gelu and the conv/llm softmax sweep designs are stale until the sweep re-run.

## 2. Remaining plan

1. **CO-SIM** on the verification columns (A, B, D, then C), including a float case; measure idle-machine timing first (`verification/cosim.py`, `cosim_trials` = 20).
2. **Modular designs**: verify the per-operator modules as composed in multi-operator designs (my reading of "modular designs" — please confirm scope).
3. **Catapult**: run co-sim and verification on some Catapult GEMM designs (backend `catapult`; today only a syntax check exists, `verification/catapult_syntax_check.py`). Not yet scoped — needs the toolchain, a testbench path and a data-type mapping (`ac_fixed`).
4. Optional: relative-error column; corner-input check; re-run the full largest-range pass as a single command.
5. After verification is final: LLM sweep re-run (csynth 3,888 + impl 1,000), then open items (`conv_block_op2` Table 7 P2 discrepancy, `forgebench/` package move after Hanqiu, `fullmodel.csv`, counts funnel / Table 4, release dry run).

## 3. Hanqiu's work (merged as 07bb88c; `scale_models/existing_model_verification/`)

- Verifies the **saved** production `top.cpp`/`top.h` of full models (ResNet-18/34/50/101/152, full and tiled; Llama-3-8B ctx2048/8192) produced by the production flow
  (`auto_generate_json.py` -> `gen_configs.run_hls_flow`). It runs the C++ unchanged with its own testbench and compares against (a) an FP64 PyTorch graph and
  (b) an integer-code fixed-point PyTorch graph that mimics the C++ arithmetic. It does **not** use the gemm/conv/llm operator generators.
- Results (C-sim, synthetic weights, one seed): all 24 variants match the fixed-point reference bit-exactly. FP64 relative L2 (threshold 5%): ResNets 0.5-2.0% in `<16,5>`, ~2e-4..6e-4 % in `<32,10>`;
  Llama `<16,5>` fails (123-125%), `<32,10>` passes (~3e-4 %). Llama uses 4 prefill + 2 decode tokens only; the RoPE reference shares the Vitis sin/cos.
- Generator changes he made (these alter the full-model sources; `hls_files` regenerated, ~1M lines):
  graph fixes (Llama o-proj in/out aliasing, unread parameter ports now rejected, bottleneck `pre_downsample` buffer, `quantize_tile` op);
  `scale_models/production_types.py` — persistent wide accumulator buffers `acc_t` (for `<16,5>`: 64 bits with 22 fractional bits), table-based `exp`, custom sqrt/div, `production_revision` = 2;
  storage `data_t` stays plain `ap_fixed<16,5>` (truncate/wrap), not the RND/SAT the paper states for full-scale models.
- Not yet reviewed: whether the repo's committed full-model synthesis results (csynth/impl) were produced from the old or the regenerated sources.

## 4. Consolidation plan (proposal)

Goal: one arithmetic story. Hanqiu's models should be expressed with the same mechanisms as the operator library, then his tests re-run on the result.

1. Decide the paper's full-model numeric configuration (storage type incl. RND/SAT, accumulator type) and which sources the synthesis tables correspond to.
2. Try to recreate his arithmetic with our controls: per-operator `acc_type` / `acc_rounding` / `acc_overflow` and per-operator `data_type`.
   **Caveats to check first:** (a) his accumulators are *persistent buffers* carried across tile calls (typed per buffer by `annotate_storage`), whereas our `acc_type` is a local accumulator inside one operator call — if partial sums
   live in a BRAM between calls, the buffer itself must be wide (a per-buffer type, not a per-operator accumulator), so this likely needs a small extension; (b) his custom `exp` table, `fb_div`, `fb_sqrt` and `FB_EPS` have no counterpart in the operator library, so bit-exact agreement with his fixed-point reference may require those to be ported or his reference to be re-derived.
3. Regenerate a small subset first (ResNet-18 tiled and full, Llama ctx2048, in `<16,5>` and `<32,10>`), re-run his `verify.py` against the new sources, and compare per-tensor relative L2 and code mismatches with his current results.
4. If the subset reproduces (or the differences are understood), regenerate all 24, refresh `hls_files`, and decide the reported metrics (his relative L2 for full models, our max abs error for operators — keep them separate and labelled).
5. Only then touch the `scale_models/` layout and the `forgebench/` package move.
