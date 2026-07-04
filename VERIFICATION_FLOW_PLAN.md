# ForgeBench Verification Flow — Implementation Brief

**Audience:** a coding agent implementing functional-correctness verification for ForgeBench.
**Branch:** `feature/correctness-harness` (already created off `main`).
**Motivation:** TRETS reviewers (esp. R1 #3) require functional-correctness evidence: per-design testbenches, a golden reference, a stated numerical tolerance, and CSIM/CO-SIM pass/fail for released designs. Today the generated `tb_top.cpp` only *dumps* outputs — it never compares against anything. This flow closes that gap.

---

## Design decision (already made — do not re-litigate)

Run the verification suite with **`data_type: "float"`** (IEEE-754 single precision), and compare HLS CSIM/CO-SIM output against a **NumPy/PyTorch float32** golden reference with a **tight relative tolerance**. Rationale: the default `ap_fixed<16,5>` has ~11 fractional bits, so quantization error would force a large, hand-wavy tolerance. Running in `float` means the only discrepancy is floating-point reassociation between the HLS schedule and the reference, which justifies a small, defensible `rtol`. `data_type` is a single JSON field already threaded through the generators, and setting it to `"float"` has been verified to produce valid single-precision C++ (`typedef float data_t`, `_float` function suffixes). Note: `ap_float<32>` is NOT a real Vitis type — use plain `float`.

This does not replace the fixed-point suites; it adds a **float verification pass** that establishes functional correctness of the generated logic. The paper can then state: correctness is verified in float via CSIM/CO-SIM against a PyTorch reference; the fixed-point suites reuse the identical logic with only the datatype changed.

---

## How the generator works (context you need)

Each domain folder (`gemm/`, `conv/`, `llm/`) has a `generate_code.py` and a `gen_configs.py`. A design is described by a JSON config with:

- `data_type` (default `"ap_fixed<16, 5>"`), `top_func_name`, `clock_period`, `FPGA_name`, `task` (list, e.g. `["csynth"]`).
- `brams`: on-chip arrays `[{"name","dims"}]`.
- `drams`: off-chip arrays `[{"name","dims","bundle"}]`, each backed by a seeded `.txt` file.
- `ops`: an **ordered** dict; each value is `{"func_name","dims","args"[,"func_info"]}`. `args` names source/dest arrays; convention for compute ops is `[input_A, input_B, bias, output]`.
- `output_dram_names`: which DRAMs the testbench writes out.

`gen_configs.run_hls_flow(config_path, base_dir, task=...)` generates, per design: `top.cpp`, `top.h`, `tb_top.cpp`, one `.txt` per DRAM (random uniform [0,1), `random.seed(42)` via `generate_dram_txt_files`), and `run_hls.tcl`. `run_hls_configs.py` runs `vitis_hls -f run_hls.tcl` in parallel over a directory of generated designs.

**Op vocabulary to reproduce in the reference** (from the `generate_operator_call` / `generate_func_def` dispatch in each `generate_code.py`):

- **gemm/**: `load`, `store`, `gemm` (dims `[M,N,K]`, `func_info=[loop_order, unroll_list, with_bias, inline]`), `vmm`, `mmv` (dims `[M,N]`), `dot_product` (dims `[M]`), `activation`.
- **conv/**: `load`, `store`, `conv`, `batchnorm`, `activation`, `maxpool`, `adaptive_avgpool`, `matrix_add`.
- **llm/**: `load`, `store`, `matmul`, `mha` (multi-head attention), `swa` (sliding-window attention), `layernorm`, `rmsnorm`, `activation`, `dropout`, `matrix_add`, `elementwise_mult`.
- **activation** names (union across domains): relu, leaky_relu, prelu, rrelu, thresholded_relu, relu6, sigmoid, tanh/tanh_act, elu, selu, gelu, geglu, swish, softmax, hardsigmoid, hardswish. Match the exact C math in each domain's activation generator (read the emitted C for the formula; do not assume canonical definitions — e.g. gelu tanh-approx vs exact).

`load`/`store` are pure copies (DRAM↔BRAM). `dropout` at inference is identity (verify in the emitted C). Read the generated `top.cpp` for any op whose math you are unsure of — the C is the ground truth the reference must match.

---

## Deliverables

### 1. Golden-reference interpreter — `verification/golden_ref.py`
A standalone module that:
- Takes a JSON config path + the directory holding that design's input `.txt` files.
- Loads each DRAM `.txt` into a NumPy array shaped by its `dims` (row-major, matching how `tb_top.cpp` flattens: nested `[d0][d1]...`).
- Walks `ops` **in insertion order**, maintaining a dict of named arrays (BRAM/DRAM). Implements each `func_name` above in NumPy (use PyTorch only if it materially simplifies `mha`/`swa`/`conv`; keep the dependency optional and gated). Honor `with_bias`, grouping, RoPE flags, normalization epsilon, etc., exactly as the emitted C does.
- Writes, for each name in `output_dram_names`, a golden file `<DRAM>.golden.txt` in the same one-value-per-line float format the testbench uses.
- Is importable and CLI-runnable: `python -m verification.golden_ref <config.json> <design_dir>`.

Correctness of the reference itself must be checked against the emitted C semantics, not against textbook formulas.

### 2. Testbench comparison — modify `generate_testbench_code` in each `generate_code.py`
Change the generated `tb_top.cpp` so that, for each output DRAM, it:
- Loads the golden file `<DRAM>.golden.txt` (skip compare gracefully if absent, to stay backward-compatible).
- Compares element-wise against the computed output with a tolerance rule: `pass if |a-b| <= atol + rtol*|b|`. Defaults: `rtol=1e-3`, `atol=1e-5` (tune during bring-up; report the final values). Make them overridable via `#define` at the top of the tb.
- Prints a machine-parseable verdict line, e.g. `VERIFICATION: PASS` / `VERIFICATION: FAIL (max_abs=..., max_rel=..., n_mismatch=.../N)`, and returns nonzero from `main()` on failure so CSIM/CO-SIM surface it.
Keep the existing output-dump behavior (write `<DRAM>_output.txt`) for debugging.

### 3. Config/task wiring — `verification/` driver
- A script that takes a set of representative configs (start with everything in each domain's `test_case_configs/`), rewrites `data_type` to `"float"` and `task` to `["csim"]` (and a `["csim","cosim"]` variant), generates the design via the existing `gen_configs` entry points, then invokes `golden_ref.py` to drop the `.golden.txt` files next to the generated inputs.
- Reuse the existing `random.seed(42)` input generation so golden and HLS see identical inputs. **Critical:** the golden reference must read the *same* `.txt` files the HLS run reads — generate inputs once, then both consume them.

### 4. Results aggregation — `verification/collect_results.py`
Parse the `vitis_hls` logs / `csim.log` for each design, extract the `VERIFICATION:` line + CSIM/CO-SIM status, and emit `verification/results.csv` with columns: `design, domain, datatype, csim_pass, cosim_pass, max_abs_err, max_rel_err, n_mismatch, n_total, notes`. This CSV feeds the point-by-point response letter and a paper table.

---

## Execution model (important — tools are remote)

Vitis HLS runs on an **external server the agent cannot reach**. So:
- The agent writes/edits all code and can run the **pure-Python** parts locally: generation, `golden_ref.py`, and a NumPy self-check (compile the emitted `top.cpp` with g++ in plain C++ using `float`, run it against the golden files — this validates the reference and the compare logic *without* Vitis).
- Actually running CSIM/CO-SIM and producing `results.csv` from real logs is done by the **user on the server**; the agent should make the driver scripts turnkey (`verification/run_all.sh`) and document the exact commands.

## Acceptance criteria
1. `golden_ref.py` reproduces every op in all three domains, validated against a native g++ `float` compile of the emitted `top.cpp` (no Vitis needed) to within `rtol=1e-3, atol=1e-5` on the `test_case_configs/` designs.
2. Generated `tb_top.cpp` performs the tolerance compare and returns nonzero on mismatch; verified via the g++ path.
3. `run_all.sh` + README section documents how the user runs CSIM/CO-SIM on the server and produces `verification/results.csv`.
4. No changes to default (`ap_fixed`) behavior when no golden file is present — backward compatible.
5. Work committed on `feature/correctness-harness` in reviewable commits; do NOT touch `main`.

## Constraints / gotchas
- Repo is OneDrive-synced: git lock files can get stuck; the working tree checks out CRLF while blobs are LF (cosmetic). Do a `.gitattributes` (`* text=auto eol=lf`) normalization commit FIRST if you touch many files, to keep diffs clean.
- Keep new deps minimal (repo currently uses numpy-via-pandas, matplotlib, tqdm, joblib). PyTorch should be optional, imported lazily only where it clearly helps.
- Match array flattening/row-major order between `golden_ref.py` and the C testbench exactly, or everything will spuriously mismatch.
