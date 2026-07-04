# ForgeBench functional-correctness verification

This directory adds functional-correctness verification to ForgeBench's generated
designs (reviewer concern **C1**). Previously the generated `tb_top.cpp` only
*dumped* outputs; it never checked them. Now each generated design is compared
against an **independent golden reference** with a stated tolerance, and CSIM/CO-SIM
surface a machine-parseable pass/fail verdict.

## Approach

- Designs are generated in **`data_type = "float"`** (IEEE-754 single precision) and
  compared against a **NumPy float32** golden reference at **`rtol = 1e-3`,
  `atol = 1e-5`** (`pass if |a - b| <= atol + rtol*|b|`). Running in float means the
  only expected discrepancy is floating-point reassociation between the HLS schedule
  and the reference, which justifies a tight tolerance. The default `ap_fixed<16,5>`
  suites reuse the *identical* generated logic with only the datatype changed.
- The golden is an **independent textbook oracle**: it implements the canonical math
  for each op (parameterized by real knobs like `with_bias`, layernorm `epsilon`,
  attention scaling, etc.), and it does **not** read the emitted C. This lets it
  catch bugs in the generated logic rather than mirroring them. Where the C
  deliberately implements a documented variant (e.g. gelu tanh-approximation), the
  golden matches that variant and the deviation is recorded in `FINDINGS.md`.

## Layout

| Path | Purpose |
|------|---------|
| `golden_ref.py` | CLI + dispatcher: reads a design config + its input `.txt` files, walks ops, writes `<DRAM>.golden.txt`. Auto-detects domain. |
| `domains/{gemm,conv,llm}.py` | Per-domain textbook golden implementations. |
| `activations.py` | Shared textbook activation oracles. |
| `io.py` | Row-major DRAM load / golden write helpers (match the testbench flatten). |
| `shim/{ap_fixed.h,hls_math.h}` | Tiny stubs so float designs compile with a stock `g++` (no Vitis) for the local self-check. |
| `gcc_selfcheck.py` | Local, **Vitis-free** validator: generate → `g++` compile → run → golden → compare. |
| `prepare_designs.py` | Server prep: generate float designs + drop goldens + register them in `run_hls.tcl`. |
| `run_all.sh` | Turnkey server driver: prepare → `vitis_hls` CSIM/CO-SIM → `results.csv`. |
| `collect_results.py` | Scrape Vitis logs → `results.csv`. |
| `FINDINGS.md`, `FINDINGS_*.md` | Bugs/variants surfaced by the harness. |

## Local self-check (no Vitis, no server)

Validates the golden reference and the generated logic on your machine using `g++`:

```bash
python3 -m verification.gcc_selfcheck gemm     # or conv, llm
```

For each `test_case_configs/*.json` it generates the float design, compiles the
emitted `top.cpp`+`tb_top.cpp` with the shim headers, runs it, builds the golden,
and cross-checks both in Python and via the testbench's `VERIFICATION:` line.

## Server run (Vitis HLS)

On the machine with `vitis_hls` on `PATH`:

```bash
bash verification/run_all.sh              # CSIM only (fast)
bash verification/run_all.sh csim,cosim   # CSIM + CO-SIM (slow)
```

This generates every design in float, drops golden files, runs `vitis_hls -f
run_hls.tcl` in each design directory (the testbench opens `<DRAM>.golden.txt`,
compares, prints `VERIFICATION: PASS/FAIL/SKIP`, and returns nonzero on mismatch),
then writes **`verification/results.csv`** with columns: `design, domain, datatype,
csim_pass, cosim_pass, max_abs_err, max_rel_err, n_mismatch, n_total, notes`.

## Backward compatibility

The testbench change is additive: if no `<DRAM>.golden.txt` is present it prints
`VERIFICATION: SKIP` and returns 0, so the existing `ap_fixed` dump-only flow is
unaffected.

## Tolerance

Default `rtol = 1e-3`, `atol = 1e-5`, overridable at compile time with
`-DVERIF_RTOL=... -DVERIF_ATOL=...`. See `FINDINGS.md` for any per-domain
adjustments and the reasoning behind them.
