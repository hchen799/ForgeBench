# Verification findings

Issues surfaced by the golden-reference correctness harness (see `verification/`).
The golden is an independent textbook oracle, so a mismatch means the generated C
and the canonical math disagree.

## Summary across domains

Local Vitis-free self-check (`python3 -m verification.gcc_selfcheck <domain>`),
real random inputs, `rtol=1e-3, atol=1e-5`:

| Domain | Pass | Non-passing (all genuine bugs / non-configs) |
|--------|------|-----------------------------------------------|
| gemm | 13/14 | `mlp` — softmax overflows to `nan` in float (see below) |
| conv | 28/30 | `conv_variable` (symbolic non-config), `vgg19_block3` (buffer-size bug) — see `FINDINGS_conv.md` |
| llm  | 7/7  | dropout-at-inference bug now fixed (was 5/7) — see `FINDINGS_llm.md` |

Per-domain detail: this file (gemm), `FINDINGS_conv.md`, `FINDINGS_llm.md`.

### Per-operator CSIM suite (`verification/op_configs/`)

In addition to the whole-design configs above, every **core operator** emitted by
each domain's `generate_code.py` now has an isolated single-operator config
(`load -> op -> store`) under `verification/op_configs/<domain>/`, so a failure
points directly at one operator's C-vs-numpy correctness rather than at a
composite design. Run the local proxy with
`python3 -m verification.verify_operators` (server: `bash verification/run_operators.sh`).

Operators covered (20 total): gemm — `gemm, vmm, mmv, dot_product, activation`;
conv — `conv, batchnorm, activation, maxpool, adaptive_avgpool, matrix_add`;
llm — `matmul, mha, swa, layernorm, rmsnorm, activation, dropout, matrix_add,
elementwise_mult`. (`load`/`store` are exercised by every config.)

Result (local proxy, random inputs, `rtol=1e-3, atol=1e-5`): **20/20 PASS**.
`llm/dropout` initially failed (256/256 elements mismatch, `max_rel≈1.0`) — the
emitted C ran its train-time LCG mask + inverse-keep scaling at inference while
the textbook counterpart is the identity — the same bug documented in
`FINDINGS_llm.md` §1, isolated to the operator and no longer masked by all-zero
inputs. **Now fixed** (`llm/dropout_template.cpp` emits an identity passthrough),
so the operator, both transformer designs, and the suite all pass. `swa` and
`adaptive_avgpool`, which no whole-design config exercised, both PASS.

**Bugs the harness caught** (all invisible to the old dump-only testbench):
train-time dropout at inference (llm) — **fixed**, `llm/dropout_template.cpp` now
emits an identity passthrough; softmax overflow (gemm/§1) and the `vgg19_block3`
buffer-size mismatch (conv) — **still open**, awaiting an author decision (fix the
generator vs. report as a known limitation). Two domains (conv, llm) also had
their DRAM inputs hard-coded to all-zeros (random generation commented out),
which would have made any functional check vacuous; restored to random.

**Caught by real Vitis CSIM, missed by the local self-check:** `vgg19_block1`
had a `maxpool` channel dim of 256 over 128-channel buffers — an out-of-bounds
write that `g++` tolerated (so the golden self-check reported PASS) but that
SIGSEGVs under Vitis CSIM. Fixed (256→128); CSIM now passes. See
`FINDINGS_conv.md` §3 — a concrete case where CSIM/CO-SIM adds coverage beyond
the Vitis-free oracle.

---

## 1. Softmax overflows in float (real bug) — gemm `mlp`

The generated softmax (`gemm/2D_activations_template.cpp`, SOFTMAX block) computes
`out[i][j] = exp(in[i][j])` with no max-subtraction, then normalizes by the row
sum. In the `mlp` design the pre-softmax activations reach ~770–1338, so
`exp(...)` overflows to `inf` in IEEE-754 float and the normalization yields `nan`
for **all** outputs. The old dump-only testbench never caught this.

- **Impact:** any design whose pre-softmax magnitudes exceed ~88 (float `exp`
  overflow point) produces `nan`. This also affects the released fixed-point suites
  (fixed-point saturates instead, giving wrong—but finite—results).
- **Proposed fix (recommended, not yet applied):** subtract the row max before
  `exp` in the SOFTMAX template — mathematically identical (softmax is
  shift-invariant), numerically stable, one-line change. Would make `mlp` pass.
- **Status:** flagged for author decision; harness currently reports `mlp` as FAIL
  (correct signal).

## 2. Bias array declared/loaded but unused (minor smell) — 8/14 gemm configs

`diff_orders_p{1,2,3}`, `mult_op_p{1,2,3}`, `vec_mtx_p{1,2}` each declare a bias
BRAM/DRAM and pass it as the 3rd arg to a gemm/vmm/mmv/dot op whose
`with_bias` flag is `False`. The codegen honors the flag and drops the bias, so
the bias array is allocated and (sometimes) loaded but never used. Pervasive, so
likely a config-template artifact rather than a per-design mistake — but it wastes
on-chip/off-chip storage and can mislead readers. The golden honors `with_bias`,
so these match. Worth a config-generator cleanup.

## Non-issues (modeled faithfully, documented for clarity)

- **Partial loads / zero-init BRAMs.** `load`/`store` copy exactly `prod(op_dims)`
  elements, which can be fewer than the declared array size (e.g. `vec_mtx_p2`
  loads 16 of a 64-length vector). Unwritten BRAM elements are zero (C globals).
  The golden models both, so these designs verify correctly.
