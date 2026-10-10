# Verification: status and plan (hand-off, updated 2026-10-09)

Branch `FPT_TRETS_2026`. Deadlines: code/results freeze 2026-10-21, resubmission 2026-10-30.
Standing rules: do not touch `scale_models/` layout until the consolidation is agreed; the LLM sweep re-run waits until verification is finished;
the user pushes (the agent only commits locally).

## 0. Status 2026-10-09 (read first; supersedes §1–§2 where they differ)

**CSIM verification is complete; every run passes.** Results, tables and the exact design JSON of every experiment are committed in
`verification/paper_results/` (commit 6646b71): `verif_ops`, `verif_modular`, `verif_fullmodel` (`.tex` + `.csv`), per run `configs/` (design JSON,
seeds, ranges), `sweep_curves/`, `rows/`, `summary.csv`. How each experiment is defined and re-run: `verification/paper_runs/README.md`
(`run_all.sh` ops, `run_table7.sh` Table 7, `run_fullmodel.sh` full models, `run_cosim.sh` CO-SIM, `collect_results.sh` -> tables).

| Experiment | Formats | What was measured | Result |
|---|---|---|---|
| Operators: 57 Table-2 variants | A `<16,5>`, B mixed, C `<32,10>`, D float | largest passing range; max abs + relative L2 error there and at ±1; N=100 | 57/57 pass in every column |
| Table 7: 18 generated programs | B, C, D | same | 18/18 pass. Weights ±√(3/fan_in), conv bias ±0.1, norm parameters [0.25, 1] |
| Table 7: 8 hand-written designs (`verification/manual_designs.py`) | B, C, D | same; conv_block_op1 ±1 only, N=10 (35–60 min per fixed-point sample) | 8/8 pass. Designs now use `acc_t`/`math_t`; softmax subtracts the max; sigmoid/tanh only take `exp` of a negative number |
| Full models, `<32,10>` round/saturate (`verify.py`) | C | exact match with the integer-code fixed reference + FP64 relative L2 | ResNet-18/34/50/101/152, full + tiled, 10 seeds each: 100/100 pass; Llama-3-8B ctx2048, 3 seeds: pass. Zero code mismatches; logits relative L2 ≤ ~1e-6 |

Full-model inputs: input uniform ±1, weights ±√(3/fan_in), one seed = new parameter set + input. Float full models were not run (`verify.py` is
fixed-point only; recommended skip, not yet confirmed by the user). Full-model results in `/usr/scratch/awanna3/fb_fullmodel_verif/seeds_32_10`
(34 GB, summaries copied to `paper_results/fullmodel/`).

**Pending**
1. ~~Commit the `<32,10>` round/saturate full-model sources~~ **done 2026-10-10**: the 12 `*_32_10_` ctx2048/ResNet projects in
   `scale_models/hls_files/` are now the verified RND/SAT sources (`top.cpp`, `top.h`, `tb_top.cpp`, `resolved_config.json`; hashes registered in
   `contracts.json`, 5f74260). Full-model synthesis must be re-run from them. Llama ctx8192 `<32,10>` is still the old truncate/wrap source.
2. **CO-SIM**: `run_cosim.sh` prepared (C and D, all 57 ops, ±1, 10 looped trials), not started; needs an idle server. Float conv ~8 h.
3. **Catapult GEMM C-sim**: next (§0b).
4. Open questions: modular `_module` equivalence check (each merged design reproduces every P program) not done; Llama ctx8192 not verified;
   gpt/llama transformer programs stop at the same range in B and C (±9.3, ±4.5), probably the shared `<32,10>` accumulator in the norm, unconfirmed;
   hand-written conv blocks in C use a truncate/wrap `acc_t` (~1e-3 error), round/saturate would likely reduce it (re-run ~10 h).
5. After that: LLM sweep re-run (csynth 3,888 + impl 1,000), then the §2 item 5 open items.

## 0b. Starting the Catapult work

Scope (user, 2026-10-08): "a couple csim tests for the catapult generated gemm ops". The larger plan (96-design subset + synthesis, metrics table) is
`REVISION_R2_AGENT_BRIEF.md` §F; its fallback is `docs/CATAPULT_STATUS.md` describing exactly what exists.

What exists:
- `backends/catapult.py`: Catapult backend, wired into **gemm only**. `ap_fixed<W,I>` -> `ac_fixed<W,I,true,AC_TRN,AC_WRAP>` (modes follow the
  generic `fixed<W,I,rnd,sat>` spelling); its math header binds `exp/sqrt/tanh` to `ac_math` for `ac_fixed` and to `<cmath>` for float;
  `pow/sin/cos` are float-only.
- Selected per design by the config key `"tool": "catapult"` (`gemm/gen_configs.py::run_hls_flow`, `backends.set_current`).
- `verification/catapult_syntax_check.py`: generates designs with the Catapult backend and runs `g++ -fsyntax-only` against the `ac_*` headers.
  No Catapult run, no testbench run, no golden compare yet. `analysis/extractors/catapult.py` is a stub.
- Toolchain: `/tools/software/siemens/catapult/{2025.2.1,2026.2}`, `latest -> 2026.2` (`MGC_HOME` default
  `/tools/software/siemens/catapult/latest/Mgc_home`, headers in `shared/include`).
- The verification engine has no backend switch yet: `functional_verification.py` / `csim_runner.py` build Vitis projects only.

Suggested path: (1) generate a few gemm ops (e.g. `ops/gemm/gemm__bias1` and one unrolled variant) with `"tool": "catapult"` in float and
`<32,10>`; (2) compile the generated `top.cpp` + testbench with g++ against the `ac_*` headers (Catapult C-sim is plain C++, no license) and compare
with the same float64 golden and `format_bound` as the Vitis runs, ideally by adding a backend option to the engine so inputs, seeds and tables
are shared; (3) then, if time allows, `catapult -shell -f run_hls.tcl` for the same designs and record the version in `docs/TOOL_VERSIONS.md`.

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
5. Exp lookup table: Hanqiu's 4,097-entry `fb_exp_negative` (ROM + interpolation in `acc_t`) is part of the **implementation** (emitted into each `top.cpp`, used by activation and attention softmax; the golden uses true exp, his fixed reference models the table). Prefer the operator-library softmax (`hls::exp` on the accumulator type, max-subtracted) so synthesis reflects the library; first test whether `hls::exp` accepts his 64-bit `acc_t` (else use a narrower `ACCM`). Same check for his `fb_div`, `fb_sqrt`, `FB_EPS`.
6. Full-model synthesis (csynth/impl) is re-run only after the sources are final: consolidate -> regenerate -> re-run his verification -> synthesize.
7. Only then touch the `scale_models/` layout and the `forgebench/` package move.
