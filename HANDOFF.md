# ForgeBench Revision — Handoff to Claude Code (WSL)

**Purpose of this file:** everything a fresh Claude Code instance needs to continue the TRETS major-revision work on ForgeBench. Read this top-to-bottom first, then `VERIFICATION_FLOW_PLAN.md` (the detailed spec for the first coding task). This file is self-contained — it does not assume any prior conversation.

**How to use this in WSL:** `cd` into the repo, open Claude Code, and start with something like: *"Read HANDOFF.md and VERIFICATION_FLOW_PLAN.md, then implement the gemm verification slice."* Claude Code has native git + real background agents, which is why the work moved here from the desktop app.

---

## 1. The situation

ForgeBench is an HLS (High-Level Synthesis) benchmark generator. Paper "ForgeBench: A Machine Learning Benchmark Suite and Auto-Generation Framework for Next-Generation HLS Tools" (TRETS-2026-0091, FPT 2026 Journal Track) received a **Major Revision**. Authors: Andy Wanna (you, co-first), Hanqiu Chen (co-first), Cong "Callie" Hao (Georgia Tech). Remote: `github.com/hchen799/ForgeBench.git`.

**Deadline:** revised manuscript + point-by-point response letter due **2026-07-30**. Budget ~10 hrs/week.

**Two review headlines:** (1) sharpen novelty/positioning vs. existing benchmark/dataset frameworks; (2) strengthen evaluation. The full grouped review summary + 4-week action plan is in `ForgeBench_Review_Summary_and_Action_Plan.md` (in the repo root — concerns are coded A1–G4 and referenced throughout).

## 2. What's already been decided / done

- **Branch:** all revision code goes on `feature/correctness-harness` (already created off `main`). Do NOT commit to `main`.
- **First work item (chosen): the correctness harness (review concern C1).** ForgeBench's generated `tb_top.cpp` currently only *dumps* outputs — it never checks them against a reference. We're adding golden-reference verification with a stated tolerance. Full spec: `VERIFICATION_FLOW_PLAN.md`.
- **Key decision — datatype:** run the verification suite with `data_type: "float"` (IEEE-754 single precision), compare against a NumPy/PyTorch **float32** golden reference with a tight relative tolerance (`rtol≈1e-3, atol≈1e-5`, tune during bring-up). Rationale: the default `ap_fixed<16,5>` (~11 fractional bits) would force a large, hand-wavy tolerance; running in `float` means the only error is float reassociation between the HLS schedule and the reference. VERIFIED: `data_type` is a single JSON field threaded through the generators, and setting it to `"float"` produces valid single-precision C++ (`typedef float data_t`, `_float` function-name suffixes). NOTE: `ap_float<32>` is NOT a real Vitis type — use plain `float`.
- **Execution split:** Vitis HLS (and Catapult) run on a **remote server**. So: pure-Python + a local `g++` float compile validate the flow WITHOUT Vitis; actual CSIM/CO-SIM runs happen on the server. Make driver scripts turnkey so the server run is one command.

## 3. Remaining work items (priority order)

1. **Verification/correctness harness (concern C1)** — IN PROGRESS, spec ready in `VERIFICATION_FLOW_PLAN.md`. Suggested first slice: gemm domain only (golden_ref.py + testbench compare + g++ self-check), then expand to conv/llm.
2. **Post-implementation PPA (concern D2)** — place-and-route IS already supported in the flow (Vitis `export_design ... -flow impl`); most reported numbers are csynth-only for speed. Need to run P&R on a representative subset and scrape latency/power/Fmax/timing into a table. Server-side run.
3. **Repo reproducibility / open-sourcing (concerns F1/F2/F3)** — reorganize repo, write a manifest (design ID → params → results), one-command regen script, repro instructions + runtime/storage estimates, and a small runnable HLSFactory integration example. **Resolve the count discrepancy below.**
4. **Catapult HLS flow (concern E1)** — abstract the Vitis-specific layer (see §5) and add a Catapult backend to demonstrate cross-tool generality.
5. **Writing/citations (concerns A1–A4, B1–B5, D3–D6, G1–G4)** — reframe as "generator + generated suites" (not a dataset), cite the prior FCCM poster version, fix "full-scale" overclaim + the LUT-vs-device-capacity issue, tone/typo/citation cleanup. Authorial — keep for the human.

## 4. How the generator works (orientation)

Domain folders `gemm/`, `conv/`, `llm/` each have `generate_code.py` + `gen_configs.py`; `scale_models/` = full ResNet/LLaMA; `modular_data/` = modularization suite. Python 3.12, uv-managed (`pyproject.toml`, `uv.lock`); deps: joblib, matplotlib, pandas, tqdm.

A design = a JSON config: `data_type` (default `"ap_fixed<16, 5>"`), `top_func_name`, `clock_period`, `FPGA_name` (default `xczu9eg-ffvb1156-2-e` = ZCU102), `task` (list, e.g. `["csynth"]`), `brams` `[{name,dims}]`, `drams` `[{name,dims,bundle}]`, ordered `ops`, `output_dram_names`.
Each op: `{func_name, dims, args[, func_info]}`. Args convention `[input_A, input_B, bias, output]`.

`gen_configs.run_hls_flow(config_path, base_dir, task=[...])` emits per design: `top.cpp`, `top.h`, `tb_top.cpp`, one `DRAM_*.txt` per DRAM (inputs = `random.seed(42)` uniform [0,1)), `run_hls.tcl`. `run_hls_configs.py` runs `vitis_hls -f run_hls.tcl` in parallel.

**Op vocabulary** (from the dispatch in each `generate_code.py`):
- gemm: `load, store, gemm` (dims `[M,N,K]`, func_info `[loop_order, unroll_list, with_bias, inline]`), `vmm, mmv` (`[M,N]`), `dot_product` (`[M]`), `activation`.
- conv: `load, store, conv, batchnorm, activation, maxpool, adaptive_avgpool, matrix_add`.
- llm: `load, store, matmul, mha, swa, layernorm, rmsnorm, activation, dropout, matrix_add, elementwise_mult`.
- activations (union): relu, leaky_relu, prelu, rrelu, thresholded_relu, relu6, sigmoid, tanh/tanh_act, elu, selu, gelu, geglu, swish, softmax, hardsigmoid, hardswish.
**When implementing the golden reference, derive each op's math from the EMITTED C in a generated `top.cpp`, not from textbook formulas** (e.g. gelu tanh-approx vs exact, attention scaling). `load`/`store` are pure copies; `dropout` at inference is identity (confirm in C).

## 5. Vitis-specific surface (for the Catapult port, concern E1)

TCL generator `generate_full_tcl_file()` in each `generate_code.py` emits Vitis commands: `open_project -reset / set_top / add_files / add_files -tb / open_solution / set_part / create_clock / csim_design / csynth_design / cosim_design / export_design -format ip_catalog -flow impl`. Interfaces: `#pragma HLS interface m_axi port=... offset=slave bundle=...`. Compute pragmas: `#pragma HLS array_partition ... cyclic factor=`, `#pragma HLS unroll factor=`. Types: `ap_fixed<16,5>` from `ap_fixed.h` (sanitized into function-name suffixes via `replace_data_type()` which regex-strips `<>,` and spaces). These are the porting surface.

## 6. Environment gotchas (were painful on the desktop/OneDrive side — mostly moot in WSL, but FYI)

- On the OneDrive-mounted checkout, the working tree was CRLF while committed blobs are LF (`core.autocrlf=false`, no `.gitattributes`) → `git status` showed ALL ~671 files modified, but it was pure line-ending noise (`git -c core.autocrlf=input diff` was empty). **Recommended first commit in WSL:** add `.gitattributes` with `* text=auto eol=lf` and run `git add --renormalize .` as one dedicated commit, so real diffs stay clean. If your WSL clone is LF-native this may already be clean — check `git status` first.
- OneDrive also blocked deleting git `.lock` files from the sandbox; not an issue in WSL.
- **Count discrepancy to resolve (concern F1):** repo README claims 6000+ testcases (1920 GEMM / 2304 DNN / 1944 LLM); the paper claims 10,837 (3335 / 4143 / 3359). Reconcile before finalizing artifact/repro claims — reviewers explicitly asked whether all 10,837 are released.

## 7. Companion files in this repo
- `VERIFICATION_FLOW_PLAN.md` — detailed implementation brief for the verification flow (deliverables, acceptance criteria, op conventions).
- `ForgeBench_Review_Summary_and_Action_Plan.md` — grouped reviewer concerns (A1–G4) + 4-week plan.
- (The reviews themselves and the manuscript PDF are with the human author.)

## 8. Suggested first action in WSL
1. `git checkout feature/correctness-harness` (create from main if the branch didn't come over: `git checkout -b feature/correctness-harness main`).
2. `.gitattributes` + renormalize commit if `git status` is noisy.
3. Implement the gemm verification slice per `VERIFICATION_FLOW_PLAN.md`; validate with the local g++ float path (no Vitis needed); commit in small chunks.
4. Expand to conv/llm; then hand the turnkey server scripts to the human for the real CSIM/CO-SIM run.
