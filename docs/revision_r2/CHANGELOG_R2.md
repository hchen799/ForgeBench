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
