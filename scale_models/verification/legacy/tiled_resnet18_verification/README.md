# Tiled ResNet-18 verification

This workflow validates the dedicated `generate_tiled_resnet18.py` accelerator.
It uses two CPU PyTorch references and the real Vitis fixed-point C simulation:

* **Mathematical:** float64 operations on the same quantized input and parameters,
  with no intermediate quantization. This measures numerical error.
* **Implementation:** integer codes reproducing the current C++ arithmetic,
  including saturation, division, BN square root, and shared tile exponents.
  Every stored checkpoint and exponent event must match exactly.

The graph intentionally has no BN on the downsample shortcuts and no convolution
or classifier biases. It returns 1,000 logits, without softmax. It is not a
drop-in torchvision pretrained ResNet-18.

## Complete run

Run from `scale_models`, with Python 3.9+, PyTorch, NumPy, and `vitis_hls` on PATH:

```bash
python generate_tiled_resnet18.py --output-dir verification_runs/resnet18/design
python golden_tiled_resnet18.py run \
  --project-dir verification_runs/resnet18/design \
  --run-dir verification_runs/resnet18/seed42 --seed 42
```

The original generator command still writes to its original default directory.
The extra `--output-dir` lets validation leave an existing HLS project untouched.
New generated projects include `tb_top.cpp`, `verification_trace.h`, and
`run_csim.tcl`; the synthesis entry point remains `run_hls.tcl`.

For independent follow-up seeds, repeat `run` with `--seed 43` / `--seed 44` and
new run directories. The compiled C-simulation executable is reused from
`verification_runs/resnet18/.csim_cache` when source and tool identity match.
`--rebuild` forces a Vitis invocation. Do not run simultaneous builds in the same
cache directory; use separate `--cache-dir` values for independent builds.

Use `--threads N` to control PyTorch CPU threads (default 8). HLS simulation can
take substantially longer than the PyTorch reference. Its live progress is in
`csim/simulation.log`; each residual block announces a checkpoint.

## Individual steps

```bash
python golden_tiled_resnet18.py prepare \
  --project-dir verification_runs/resnet18/design \
  --run-dir verification_runs/resnet18/my_case --seed 42
python golden_tiled_resnet18.py reference --run-dir verification_runs/resnet18/my_case
python golden_tiled_resnet18.py csim --run-dir verification_runs/resnet18/my_case
python golden_tiled_resnet18.py compare --run-dir verification_runs/resnet18/my_case
```

`reference --mode float` and `reference --mode fixed` run one reference.
`--sequential` disables the exact, bound-checked matrix reduction acceleration;
it is intended for arithmetic debugging and is slower for complete networks.

`prepare` refuses to overwrite a populated directory. Reference results can be
recomputed in the existing case. A repeated `csim` archives the previous output
directory as `csim.previous.<timestamp>` before starting. All inputs and source
snapshots are checksummed, as are reference results and simulation outputs.
After editing the Python implementation, rerun `reference` before comparing.
After changing HLS sources or scales, prepare a new case.

## Random data and storage

One seeded CPU generator creates all input and parameter tensors in port order:

| Tensor | Uniform distribution before Q5.11 conversion |
|---|---|
| Input | `[-1, 1]` |
| Convolution weights | `[-sqrt(3/fan_in), sqrt(3/fan_in)]` |
| FC weights | `[-sqrt(3/512), sqrt(3/512)]` |
| BN gamma | `[0.9, 1.1]` |
| BN beta and mean | `[-0.05, 0.05]` |
| BN variance | `[0.9, 1.1]` |

`fan_in = Cin * Kh * Kw`. BN rows are gamma, beta, mean, variance. Both
references and C simulation read the saved quantized values, not independently
generated random streams. Versions and distributions are recorded in the
manifest; saved vectors, rather than seeds alone, define reproducibility across
library versions. Weights are for functional testing, not trained inference.

Each `.bin` contains signed little-endian int16 codes. A code represents
`code / 2048`, with range `[-16, 15.99951171875]`. C++ writes/reads the raw 16
bits explicitly; it does not serialize an `ap_fixed` object's host memory.
Activations are CHW, weights OIHW, BN `[4,C]`, and FC `[1000,512]`.
Only the 39 input/parameter tensors are loaded (about 22.6 MiB). Scratch and
output arrays start at zero. Trace exports respect the physical `[512][56][56]`
strides while writing only the valid logical region.

Accumulator codes have 22 fractional bits. Conversion into AP_RND types rounds
nearest with ties toward positive infinity. HLS division expressions already
truncate at their expression precision; they are not modeled as an ideal real
division followed by AP_RND. BN uses the actual quantized epsilon code 42,
fixed-point square root, normalization saturation, and a wide affine expression
before the final storage conversion.

## Reading results

* `manifest.json`: inputs, shapes, seed, distributions, checksums, scales, versions.
* `source/`: exact C++ and testbench snapshot.
* `float/`: float64 `.npy` checkpoints and readable `logits.txt`.
* `fixed/`: expected raw checkpoints, readable logits, exponent trace, statistics.
* `csim/`: actual raw checkpoints, readable logits, exponent trace, tool log.
* `report.json` / `report.txt`: exact comparison and numerical errors.

There are 65 tensor checkpoints: four stem outputs, seven outputs per identity
block, eight per downsample block, and the two head outputs. Exponent events
record operation, output-channel offset, row, column, input-channel offset, and
the shared exponent after the chunk's guard check.

`implementation_match: PASS` requires zero mismatched codes at **every**
checkpoint, matching exponent events, and a successful C-simulation exit.
Mathematical errors are diagnostic by default: maximum absolute error, MAE,
RMSE, and maximum relative error with a denominator floor of one storage LSB.
No loose numerical tolerance can make an implementation mismatch pass.

To add an explicit mathematical gate, provide both `--float-atol` and
`--float-rtol` to `run` or `compare`. They apply at every checkpoint using
`abs(actual - reference) <= atol + rtol * abs(reference)`. Thresholds are chosen
by the caller, not inferred from the observed result. Nominal saturation or
constant logits are flagged and cause a nonzero overall exit status.

Exit status is 0 for success, 1 for failed comparisons/simulation or nominal
warnings, and 2 for malformed inputs/configuration or other execution errors.
An exact implementation match does **not** establish mathematical correctness;
the current accelerator has the directed arithmetic issues in `FINDINGS.md`.

## Focused tests

```bash
python -m pytest -q tiled_resnet18_verification/tests
```

The C++ arithmetic probe includes the actual generated operators and compiles
with the installed Vitis headers using g++. Set `VITIS_HLS_INCLUDE` if headers
cannot be found relative to `vitis_hls`. Tests that need the real headers are
explicitly skipped when unavailable; a skip is not fixed-point validation.
The full workflow separately uses Vitis's own C-simulation compiler.

Validation on 2026-10-02 used PyTorch 2.2.0 and Vitis HLS 2024.1.2. The first
case ran through `vitis_hls`; subsequent cases reused that compiled executable:

| Seed | Exact checkpoints | Exponent trace | Maximum absolute logit error vs. float64 |
|---|---|---|---:|
| 42 | 65/65 matched | Matched | 0.000613164 |
| 43 | 65/65 matched | Matched | 0.000638318 |
| 44 | 65/65 matched | Matched | 0.000614963 |

All 1,000 logits matched the integer reference for each seed. None of these
nominal cases had storage/accumulator saturation or constant logits. Detailed
local results are under `verification_runs/resnet18/seed*/report.json`; generated
vectors, builds, and reports are ignored by git and can be reproduced with the
commands above. The focused suite contains 15 passing tests, including the
directed defects, raw I/O checks, reproducibility, and stale-data rejection.

Coverage includes the complete stored nonnegative variance domain for square
root, signed rounding/division, saturation, randomized BN parameters, tile-edge
convolutions with 129 input channels and both strides, and reproductions of the
known reduction defects. Both accelerated and sequential Python reductions are
checked against the C++ outputs.

No RTL co-simulation, board execution, trained-checkpoint importer, or automatic
scale calibration is included. Trace hooks compile out of synthesis and do not
change the top-level interface or arithmetic algorithms.
