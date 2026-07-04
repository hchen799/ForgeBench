# Verification findings

Issues surfaced by the golden-reference correctness harness (see `verification/`).
The golden is an independent textbook oracle, so a mismatch means the generated C
and the canonical math disagree.

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
