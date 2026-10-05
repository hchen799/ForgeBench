# Result inventory and evidence

Results for the **saved production JSON-generated HLS** are recorded separately
in [existing/validation_summary.json](existing/validation_summary.json), with
[reproduction instructions](../existing/README.md). Their 18 verifier tests
pass, but the production designs do not pass both numerical comparisons.
The inventory and historical results below concern the dedicated workflow.

`validation_summary.json` contains selected completed results, exact metrics,
source/artifact locations and report/log SHA256 checksums. Paths are relative
to `scale_models`. `regression_tests.log` records the completed combined test
suite. Large inputs, references, C++ outputs and build files are in the
git-ignored `verification/runs/` directory on this machine.

Final combined regression result: **77 passed, zero failures, zero skips**.
This includes 45 new configurable/generic tests and the preserved 32-test suite.

## What the evidence establishes

### A. C/C++ versus fixed-point PyTorch: exact agreement

The results below concern raw integer-code equality at all selected checkpoints,
not FP64 numerical accuracy. The full
[fixed-point result table](../README.md#a-fixed-point-pytorch-comparison) lists
checkpoint counts, mismatching elements, maximum code errors and exact verdicts.

- New configurable ResNet-18: full-network Vitis C simulation at `<16,5>` and
  `<32,10>`, plus a `<24,8>` run with tile sizes 64/7/7 and a stem output shift.
  All 65 tensor checkpoints and shared-exponent events agree exactly.
- New configurable Llama: Vitis C simulation for a two-layer, 64-hidden,
  128-FFN, four-query-head/one-KV-head model at `<16,5>` and `<32,10>`, using
  30 prefill tokens and two decode calls. Every one of the 390 checkpoints agrees.
- New custom Llama: native C++ for hidden 60, FFN 131, three query heads,
  head dimension 20, vocabulary 259, and non-multiple 17/19 tiles, at `<24,8>`.
  The 31+2 case agrees at all 312 checkpoints.
- Preserved full Llama 3 8B: `<16,5>` short 4+2 and long 2048+2 runs were
  re-compared with source/input/reference integrity checks after moving the
  artifacts. These saved C++ outputs were **not re-simulated** as part of that
  re-comparison. The long run's eight SwiGLU saturations remain disclosed.
- Preserved ResNet seeds 42/43/44 and tiny Llama cache-boundary runs remain
  available. The seed-42 result was re-compared through the new CLI.

### B. Dequantized C/C++ versus FP64 PyTorch: numerical error

The separate [FP64 result table](../README.md#b-fp64-pytorch-comparison) reports
maximum absolute error, MAE, RMSE and maximum relative error across all logits.
These metrics use the same actual accelerator outputs as comparison A, converted
to real values using the configured fixed-point scale. Both references consume
the same saved quantized input and parameters.

All recorded nominal mathematical verdicts are **`DIAGNOSTIC_ONLY`**, not an
accuracy PASS. For example, full ResNet-18 `<16,5>` has zero integer mismatches
against its fixed reference, but maximum absolute logit error
`0.0006131635532236057` against FP64. Full ResNet-18 `<32,10>` also has zero
integer mismatches, with FP64 maximum absolute logit error
`3.2091310631088277e-7`. These statements answer different questions and should
not be combined into a single undifferentiated "PyTorch comparison passed" claim.

### Artifact provenance

`validated_*` directories are the final new architecture runs. Earlier
`review_*`, `configurable_smoke_*` and `*_probe` directories are retained
development evidence; some predate reference-helper fixes and therefore have
intentionally stale fingerprints. They are not used as current-adapter pass
evidence. `review_archive_*` directories are logged re-comparisons of the
unchanged frozen baseline, not development configurations.

## How to inspect or reproduce

The central [README](../README.md) has runnable commands. For each selected
new case, read in this order:

1. `summary.json`: read `implementation_verdict` / `integer_mismatches` for the
   fixed-point comparison, then `mathematical_verdict` / `logit_errors` for the
   FP64 comparison; `overall_verdict` is the combined policy result;
2. `resolved_config.json`: actual dimensions, precision and test schedule;
3. `verification.log`: comparison contract, data provenance and every tensor;
4. `comparison.json`: machine-readable per-checkpoint mismatch/error details;
5. `case/source/` (ResNet) or `model/source/` (Llama): actual compiled source;
6. golden `case/fixed/` / `case/float/` and actual `case/csim/` files.

The full historical 8B manifests still contain the earlier absolute model path.
The `verification_runs` compatibility symlink keeps that path valid locally.
Historical artifact documentation is preserved in
[Llama VALIDATION.md](../legacy/llama3_verification/VALIDATION.md) and
[ResNet README.md](../legacy/tiled_resnet18_verification/README.md).

The generic demo is recorded separately as **comparator plumbing only**. Its
`actual.npy` is copied from a golden tensor and must not be interpreted as an
HLS result. Negative tests deliberately introduce mismatches, bad checksums,
invalid shapes/dtypes/non-finite values, and failed mathematical tolerances.

## Directed arithmetic coverage

The new tests cover `<8,4>`, `<12,6>`, `<16,5>`, `<24,8>`, `<32,10>` and
`<32,8>` in selected quantization/Llama operator probes. Partial-tile end-to-end
native C++ runs cover `<12,6>`, `<16,5>`, `<24,8>` and `<32,10>`. Actual vendor
ResNet BN/square-root/shift/saturating-MAC probes cover `<12,6>`, `<24,8>` and
`<32,10>`. This is selected coverage, not an exhaustive model/precision matrix.

The 32-bit Llama probe also executes the synthesis-side `ap_int<128>` aliases
using the vendor's host simulation library. This checks arithmetic behavior;
it is **not** a new RTL synthesis result. Scalar and array cases near signed
int64 limits explicitly test clipping without accidental NumPy floating-point
promotion. Sums exceeding int64 are checked against Python integer arithmetic.

## Limits on conclusions

All mathematical verdicts in the nominal result inventory are diagnostic:
no accuracy tolerance was selected after observing the results. Exact agreement
means agreement with the implementation reference, including the preserved
[ResNet arithmetic defects](../legacy/tiled_resnet18_verification/FINDINGS.md).
Saturation counters come from exact-reference instrumentation, not physical
hardware event counters.

There are no new full-8B `<32,10>` results, no full-8B 8192-token execution,
no RTL cosimulation or board validation, and no pretrained accuracy claims.
Only the frozen full-size `<16,5>` Llama tops have the previously completed
HLS synthesis evidence. New configurations emit synthesis projects but require
their own synthesis/implementation validation.
