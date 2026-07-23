# ForgeBench — Summary of Additions & Changes for Paper Drafting

**Purpose:** Hand-off notes for the agent drafting the paper update. Every item
below is *new or changed* work not yet reflected in the manuscript. Each section
tags the likely paper location (Contributions / Methodology / Results / Eval /
Artifact). Branch: `feature/correctness-harness`. Target device for HLS results:
**ZCU102 (`xczu9eg-ffvb1156-2-e`), Vitis HLS 2024.1**, 100 MHz (10 ns) clock.

---

## 1. Full-scale data collection — csynth (Results)

The C-synthesis sweep was re-run at full yield across all three domains. Final
committed metrics (`analysis/results_csynth/metrics_{gemm,conv,llm}.csv`):

| Domain | Designs (csynth) | Prior paper target |
|--------|-----------------:|-------------------:|
| gemm   | 3,840  | 3,335 |
| conv   | 5,183  | 4,143 |
| llm    | 3,888  | 3,359 |
| **Total** | **12,911** | 10,837 |

All exceed the originally reported targets. Each row carries the standard
resource/latency schema (`analysis/schema.py`, `CSV_COLUMNS`). Extractor:
`analysis/extractors/vitis.py` parsing `csynth.xml`.

**Paper impact:** update dataset-size claims and all csynth distribution figures.
The distribution/plotting code is in `analysis/dist_csynth.py`, `density.py`,
`plots.py`.

---

## 2. NEW: Post–place-and-route (impl) results (Contributions + Results)

A **stratified ~1000-design-per-domain implementation subset** was placed & routed
(`export_design -format ip_catalog -flow impl`), giving *real* post-P&R
power / Fmax / timing rather than csynth estimates. This is a new result class
the prior draft did not have.

Committed metrics (`analysis/results_impl/metrics_{gemm,conv,llm}.csv`):

| Domain | Designs (P&R) | Power        | Fmax         | Timing met | LUT range      |
|--------|--------------:|--------------|--------------|:----------:|----------------|
| gemm   | 817 | 0.63–1.15 W | 141–398 MHz | 817/817 | 1,054–16,578 |
| conv   | 896 | 0.69–0.85 W | 104–166 MHz | 896/896 | 4,043–9,494 |
| llm    | 998 | 0.91–1.70 W | 105–177 MHz | 998/998 | 14,085–46,857 |
| **Total** | **2,711** | | | **all met** | |

- Stratified sampler deliberately spreads the unroll-factor distribution
  (includes a controlled fraction of unroll-8 designs) so the tails are populated
  — gemm LUT spans ~16×.
- 5 of 2,716 staged designs did not finish P&R (3 gemm + 2 llm), almost certainly
  the heaviest unroll-8 designs hitting the 4 h per-design timeout (0.2%).
- Extraction reads `impl/report/verilog/export_impl.xml` (timing/resource) +
  `impl/verilog/.../*_power_routed.rpt` (power). Runtime: ~17 h wall at 32 parallel
  jobs.
- Plotting: `analysis/dist_impl.py`.

**Scale models (ResNet-18/34):** attempted at impl but **all timed out** — the
tiled designs are 4–20× over ZCU102 capacity (e.g. ResNet-18 ~5.24M LUT untiled,
~1.12M tiled, vs ~274K on device). csynth-level metrics were salvaged to
`analysis/results_impl/metrics_scale_csynth.csv`. **Paper impact:** frame these as
a scalability/limitation point, not an impl data point.

**Paper impact:** new section/figures on measured power–performance–area from
real P&R; contrast csynth-estimated vs post-route where designs overlap.

---

## 3. NEW: Multi-backend / tool-portability — Catapult HLS (Contributions + Methodology)

The generators were refactored behind a **`ToolBackend` abstraction** (`backends/`)
so the same JSON config can emit either a **Vitis** or a **Catapult** project.
This addresses a "single-tool" reviewer concern (labeled "review concern E1" in the
code) and is a genuine new contribution: ForgeBench is no longer Vitis-locked.

- `backends/base.py` — `ToolBackend` interface + registry (mirrors
  `analysis/extractors/base.py`). Owns the four tool-specific emission sites:
  **types** (`ap_fixed`/`ac_fixed`), **loop/unroll pragmas**, **array-partition
  pragmas**, **interface pragmas**, and the **build TCL**.
- `backends/vitis.py` — faithful transcription of pre-refactor emission;
  **byte-identical** to the old output (two quirks preserved on purpose:
  `#pragma HLS unroll factor=1` still emitted; `<hls_math.h>` included twice). This
  guarantees the committed metrics remain valid post-refactor.
- `backends/catapult.py` — Catapult backend targeting **Nangate 45nm** (ASIC flow,
  DesignCompiler RTL). Key translations documented in-file:
  - `ap_fixed<W,I>` → `ac_fixed<W,I,true,AC_TRN,AC_WRAP>`
  - unroll pragma moves *before* the loop (`#pragma hls_unroll N`)
  - `array_partition cyclic` → `hls_interleave` on the *declaration* (exact for
    dim=2 row-major, approximate for dim=1; parameter-level requests are dropped
    and reported via `dropped_partitions`)
  - no `m_axi`; sized array params map to memory interfaces natively
  - `hls::exp/tanh/sqrt` → **ac_math PWL** via a generated `forgebench_math.h`
    that re-provides `namespace hls` on top of ac_math (same trick as the
    verification shim), so templates and Vitis output are untouched.
- **Important caveats for the paper:**
  - ac_math functions are **piecewise-linear approximations, NOT bit-identical** to
    Vitis hls_math → designs verified on Catapult need looser tolerance or
    `data_type:"float"`.
  - `pow/sin/cos` (RoPE kernels) are **float-only** so far (ac_sincos_cordic needs
    kernel restructuring) — a documented TODO / limitation.
  - The backend is currently **wired into the `gemm` generator only**
    (`gemm/gen_configs.py` selects via `config.get("tool", tool)`;
    `gemm/generate_code.py` refactored, −405 lines). **conv / llm / scale_models
    generators are NOT yet ported.** State this scope honestly.
- `verification/catapult_syntax_check.py` — **license-free** validation: generates
  with `tool="catapult"` and compiles `top.cpp` against real Catapult
  `ac_fixed`/`ac_math` headers with stock g++ (`-fsyntax-only`). Catches
  type/math-binding errors without a Catapult license or synthesis run. (Does not
  check scheduling/area/pragmas — g++ ignores unknown pragmas.)

**Paper impact:** new "tool portability" contribution + methodology subsection;
be precise that Catapult is demonstrated for gemm with a license-free syntax gate,
full Catapult synthesis numbers are not yet collected.

---

## 4. NEW: Correctness / functional-verification harness (Contributions + Eval)

A verification harness was added establishing that generated designs are
*functionally correct*, not just synthesizable — a common benchmark-generator gap.
Three tiers, all comparing against a **numpy golden oracle** (rtol=1e-3, atol=1e-5,
`data_type:"float"`):

1. **g++ Vitis-free self-check** (`verification/gcc_selfcheck.py`) — generate float
   design → numpy golden → compile with a `hls_math.h` shim → run → compare. No
   Vitis license needed. Shim sources f-suffixed math from global `::` (GCC 11.5
   compat).
2. **Real Vitis CSIM** (`verification/run_all.sh`, `prepare_designs.py`,
   `collect_results.py`) — the true simulation flow; goldens registered into
   `run_hls.tcl` as `-tb` files; `VERIFICATION: PASS/FAIL` on golden compare.
3. **NEW per-operator CSIM suite** (`verification/op_configs/<domain>/*.json`,
   `verify_operators.py`, `run_operators.sh`) — 20 configs each isolating ONE core
   operator (load → op → store) so a failure points directly at that operator.
   Operator coverage: **gemm** (activation, dot_product, gemm, mmv, vmm) · **conv**
   (activation, adaptive_avgpool, batchnorm, conv, matrix_add, maxpool) · **llm**
   (activation, dropout, elementwise_mult, layernorm, matmul, matrix_add, mha,
   rmsnorm, swa).

Findings are written up in `verification/FINDINGS.md`, `FINDINGS_conv.md`,
`FINDINGS_llm.md` (ready to cite directly).

**Paper impact:** new "functional verification" contribution + an evaluation table
of per-operator/per-design PASS rates. Note the methodological point that CSIM
catches bugs the golden self-check cannot (see §5, vgg19_block1).

---

## 5. Bugs found & fixed *via* the verification harness (Eval — evidence the harness works)

These are concrete correctness bugs the harness surfaced and that were fixed — good
"the harness earns its keep" evidence:

| # | Domain / design | Bug | Caught by | Status |
|---|-----------------|-----|-----------|--------|
| 1 | gemm `mlp` | softmax **float overflow** in 2D activations | self-check | FIXED |
| 2 | llm (dropout) | dropout ran **train-time formula at inference** (non-identity) | self-check | FIXED |
| 3 | conv `vgg19_block3` | **buffer-size mismatch** (won't compile) | self-check | FIXED |
| 4 | conv `vgg19_block1` | maxpool **channel OOB write** (256 vs 128) | **Vitis CSIM only** | FIXED |
| 5 | gemm (8/14 configs) | bias array declared/loaded but **unused** (minor smell) | review | noted |

**Case 4 is methodologically important:** the golden self-check compares only
in-bounds elements and missed the out-of-bounds write; only real Vitis CSIM
(SIGSEGV) caught it. Use this to justify why the harness runs *both* tiers.

Additionally, generator-correctness fixes in the gemm codegen itself: `option_1`
now computes `result = ((xᵀA)B)·y` with correct vmm/dot operand wiring; `option_5`
bias BRAM fixed; and a C++ **array-parameter-decay name collision** (`T[M][N]`
decays to `T(*)[N]`, so two functions differing only in outer dim collided) was
fixed by baking `(order, M, N, bias)` into both definition and call names.

**Per-operator CSIM result (real Vitis CSIM, completed):
`20/20 operators PASS, 0 FAIL`** across gemm/conv/llm, all `data_type:"float"`,
max relative error ≤ 8.4e-3 (typically ~1e-4 or better), zero element mismatches.
Full per-operator table in `verification/results_operators.csv` — directly usable
as a paper evaluation table.

---

## 6. Infrastructure / artifact notes (Artifact appendix — not paper body)

- **Checkpoints** (`checkpoints/20260720/`): lean archives holding only the report
  files needed to re-extract metrics. `csynth_{gemm,conv,llm}_lean.tar.gz`
  (csynth.xml + source per design), `impl60_reports_lean.tar.gz`,
  `impl1000_reports_lean.tar.gz` (~2,711 P&R designs: export_impl.xml + csynth.xml
  + power_routed.rpt, 70 MB). Archives >100 MB are split into `_part_NN` chunks;
  `regroup.sh` reassembles (GitHub 100 MB/file limit).
- **`.autopilot` pruning**: each domain's `run_hls_configs.py` now deletes the
  ~25 MB/design Vitis build DB after each run (was the cause of a 170 GB llm dir).
- `tqdm`/`joblib` imports made optional in the runners (a missing module had
  silently produced zero results in an earlier full run).
- **Cleanup TODO (repo hygiene, flag to maintainer):** commit `afd74cc`
  accidentally included a `jgproject/` JasperGold formal-session directory (session
  logs, caches). It is not part of the artifact and should be `git rm`'d +
  gitignored.
- The 363 GB `_impl1000/` build tree is gitignored and safe to delete after the
  lean archive is committed (frees `/usr/scratch`).

---

## 7. Suggested paper-section mapping

- **Contributions:** (a) larger dataset (§1–2), (b) real post-P&R PPA (§2),
  (c) tool portability / Catapult backend (§3), (d) functional-verification
  harness (§4–5).
- **Methodology:** backend abstraction (§3), three-tier verification (§4).
- **Results/Eval:** csynth + impl distributions (§1–2), scale-model scalability
  limit (§2), per-operator verification pass table + bug case studies (§4–5).
- **Limitations:** Catapult wired to gemm only + PWL math non-bit-exactness (§3),
  scale models exceed ZCU102 (§2).
