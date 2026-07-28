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

## N randomized inputs under real CSIM

`csim_runner.py` verifies each design over **N randomized input trials** through
Vitis CSIM (the Vitis-free equivalent is `verify_operators --n`). Trial *k* uses
`random.seed(42 + k)`, so *k*=0 reproduces the historical single-input result, and
the numpy golden is recomputed from the new inputs before every trial.

```bash
python3 -m verification.csim_runner --n 1                      # all 69 op variants, 1 input
python3 -m verification.csim_runner --n 100 --mode exe --jobs 24
python3 -m verification.csim_runner --suite designs --n 1      # whole-design suite
python3 -m verification.csim_runner llm --n 10 --configs mha__rope1_dense
```

Two trial modes, because tool overhead dominates:

| mode | trial 0 | trials 1..N-1 | measured cost/trial |
|------|---------|---------------|---------------------|
| `full` (default) | `vitis_hls -f run_hls.tcl` | same, every trial | **~9.0 s** — startup + project open + clang rebuild are paid N times |
| `exe` | `vitis_hls -f run_hls.tcl` | re-run the `csim.exe` Vitis just built, with refreshed inputs/goldens | **~0.18 s** (median 0.05 s) |

`exe` mode runs the exact binary Vitis CSIM produced — same clang, same objects,
only the tool wrapper is skipped — and was cross-checked against `full` mode:
identical `max_rel` per seed. Use `full` for the headline result, `exe` when N is
large.

### Measured campaigns

Both over the full 69-config operator suite, run concurrently at 64 threads total
on a 128-core host:

| run | jobs | wall | serial CPU | result |
|-----|------|------|------------|--------|
| `--n 100 --mode full` | 48 | 30.9 min | 17.6 h | **69/69 designs, 6,900/6,900 trials PASS** |
| `--n 1000 --mode exe` | 16 | 28.0 min | 6.1 h | **69/69 designs, 69,000/69,000 trials PASS** |

Worst absolute error over all 75,900 trials: **2.4e-4** (`llm/mha`, on a tensor
whose elements are O(3-20)); median across designs **1.2e-7**; **26 of 69**
designs are bit-exact against the numpy golden.

> **`max_abs_err` is the headline accuracy metric**, not `max_rel_err`. With
> signed inputs the relative column is dominated by rare near-cancelling
> elements (`rel = abs_err / (|ref| + 1e-12)`), so it reaches absurd values --
> 2.1e5 for `conv__grouped_bias1` -- while the design passes and its worst
> absolute error is 2.3e-5. Judge accuracy by `max_abs_err` against the tensor
> scale. `max_rel_err` is still written to the CSV for completeness, but the
> runner's per-design log line and end-of-run summary (worst / median /
> bit-exact count) report `max_abs_err`.

`--jobs` parallelizes the Vitis runs. Design *generation* stays serial on purpose:
`gen_configs.run_hls_flow` writes `<DRAM>.txt` into the current directory before
moving it into the run dir, so concurrent generation within a domain would clobber
inputs. Generation is ~25 ms/design, so this costs nothing.

Results go to `results_csim_<suite>_n<N>.csv`, adding `n_inputs`, `mode`,
`trials_passed`, and per-trial timing columns (`trial0_s`, `mean_trial_s`) to the
usual error/verdict columns.

## Backward compatibility

The testbench change is additive: if no `<DRAM>.golden.txt` is present it prints
`VERIFICATION: SKIP` and returns 0, so the existing `ap_fixed` dump-only flow is
unaffected.

## Tolerance

An element passes when

```
|a - b| <= (atol + atol_scale * max_j|b_j|) + rtol * |b|
```

with `rtol = 1e-3`, `atol = 1e-5`, `atol_scale = 5e-5`, overridable at compile
time with `-DVERIF_RTOL=... -DVERIF_ATOL=... -DVERIF_ATOL_SCALE=...`.

The `atol_scale` term makes the absolute floor proportional to the output
tensor's peak magnitude. A reduction's float32 reassociation error is set by the
magnitude of the terms being summed, not by the magnitude of an individual
result, so an element that happens to cancel to near zero cannot be held to a
near-zero absolute tolerance. Once inputs span both signs this is not
hypothetical: `mha` produces a couple of outputs per run that are ~1000x below
the tensor's typical magnitude, while the error floor stays at ~1e-4 for a tensor
whose elements are O(3-20). With the scale term those elements are compared
against ~1e-3 instead of ~1e-5, which is still far tighter than `rtol` allows for
a typical element.

## Input ranges

Each config carries an `input_range` (`[low, high]`, default `[0, 1)`) that sets
the range of the generated DRAM `.txt` values; individual DRAMs may override it
with their own `input_range` key. The historical `[0, 1)` default is strictly
non-negative, so every sign-dependent branch (relu, leaky_relu, elu, prelu, ...)
and every clamp (relu6's cap, thresholded_relu's theta) was dead code under test
at any number of trials. The operator suite therefore uses:

| configs | range | why |
|---------|-------|-----|
| activations | `[-8, 8]` | reaches both sign branches and clears the clamping constants (cap 6, theta 1) |
| everything else | `[-1, 1]` | exercises sign handling without inviting catastrophic cancellation in long float accumulations |
| `batchnorm` `DRAM_weights` | `[0, 1)` | the array holds `variance`, which feeds `sqrt(var + eps)` -- negatives give NaN |

Widening the range matters more than raising N: more seeds from a one-sided
distribution resample the same orthant.

See `FINDINGS.md` for per-domain notes.
