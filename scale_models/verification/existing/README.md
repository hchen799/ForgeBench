# Verify saved production HLS against two PyTorch references

`verify_existing.py` tests the **existing** `top.cpp` and `top.h` in a
production `scale_models/hls_files/` directory. It does not call a model
generator, repair the DUT, change its arithmetic, or use a dedicated tiled
accelerator as a substitute. The original synthesis TCL and testbench are
not executed; a separate testbench supplies inputs and observes outputs.

Each supported source version is paired with:

1. An independently expressed **FP64 PyTorch mathematical graph**.
2. An **integer-code PyTorch fixed-point graph**, modeling the source's
   intermediate widths, casts, truncation, wrapping, and reduction behavior.

The same quantized input and parameter values feed all three executions.
The fixed reference is not an FP64 model followed by output quantization.
It does not repair arithmetic such as `(data_t)49 == -15`, nor reproduce graph
defects such as disconnected parameter ports to force an implementation PASS.
The mathematical reference retains the intended real-valued operations.

The FP64 reference uses independent PyTorch mathematics throughout. For the
fixed Llama reference, RoPE coefficients use the same Vitis float pow/sin/cos
primitives as C simulation because their rounding differs from PyTorch/libm.
This limited shared dependency is disclosed in the report; executed runs save
the library hashes and coefficient tables. The fixed graph remains separate,
but this RoPE comparison does not independently validate those vendor
transcendental functions. No accelerator outputs become reference values.

The earlier `verification/verify.py run` workflow uses dedicated generators
and different arithmetic. Its results do **not** validate these saved sources.

## Run

From `scale_models`, with Python 3.9+, PyTorch, NumPy, and Vitis HLS:

```bash
python verification/verify_existing.py \
  --family resnet18 \
  --project hls_files/RESNET18_TILED_config_ap_fixed_16_5_ \
  --output verification/runs/my_existing_resnet18_tiled

python verification/verify_existing.py \
  --family resnet18 \
  --project hls_files/RESNET18_config_ap_fixed_16_5_ \
  --output verification/runs/my_existing_resnet18_full

python verification/verify_existing.py \
  --family resnet50 \
  --project hls_files/RESNET50_config_ap_fixed_16_5_ \
  --output verification/runs/my_existing_resnet50_full

python verification/verify_existing.py \
  --family llama3 \
  --project hls_files/LLAMA3_8B_PREFILL_ctx2048_config_ap_fixed_16_5_ \
  --decode-project hls_files/LLAMA3_8B_DECODE_ctx2048_config_ap_fixed_16_5_ \
  --output verification/runs/my_existing_llama3
```

To generate both precisions through the same production entry point:

```bash
python auto_generate_json.py --data-types 'ap_fixed<16,5>' 'ap_fixed<32,10>'
```

Run `gen_configs.run_hls_flow` on the desired JSON as before. For Q32.10,
use the same verifier commands with project names ending in
`_config_ap_fixed_32_10_`, and a new output directory. The existing default
of `auto_generate_json.py` remains Q16.5. The data path is selected by the
header and exact source contract, never by a command-line override.

Every output directory must be new. Optional flags:

- `--config FILE` and `--decode-config FILE`: production JSONs, cross-checked
  against the corresponding saved headers. Otherwise matching files under
  `auto_generated_configs/` are detected; a saved project can also be verified
  without its original JSON.
  This checks configuration/interface consistency; it does not regenerate
  the source or prove equivalence of every JSON operation to the source.
- `--seed 42`, `--threads 8`, `--vitis /path/to/vitis_hls`, `--timeout 7200`.
- `--prefill 4 --decode 2`: Llama schedule, within the saved cache capacity.
- `--tokens 1,2,3,4,5,6`: explicit token IDs. The default draws IDs across the
  full vocabulary. Revision 2 transfers control ports as int32 values.
  Historical v1 sources still reject IDs that their data_t interface cannot represent.
- `--inspect-only`: snapshot the sources and check their interface/arithmetic
  without allocating parameters or running C simulation. This is not a PASS.

The current registered versions are full-buffer and tiled ResNet-18/50 and
the context-2048 Llama 3 8B prefill/decode sources in this checkout.
Revision 2 repairs are made in the original JSON-to-HLS production generator;
[repair details and measured results](../results/production_v2/README.md) distinguish
implementation agreement from FP64 closeness. Historical v1 contracts remain
registered for archived sources. Both `ap_fixed<16,5>` and `ap_fixed<32,10>` are registered, using
**AP_TRN/AP_WRAP**. The word length is read from the saved header; it controls
raw binary I/O, parameter quantization, fixed arithmetic, and FP64 scaling.
Q16.5 uses a 64-bit accumulator with 22 fractional bits; Q32.10 uses an
80-bit accumulator with 44 fractional bits. The wide PyTorch reference uses
exact integer limbs for GEMM/convolution and Python integers for intermediates
that exceed int64. Source-compatible precision is never inferred from a filename. Source hashes in
`contracts.json` bind a reference pair to an exact implementation. Changed
sources, other precisions, ResNet depths beyond 18/50, and other context/tile variants
must have their arithmetic and interface validated before adding a contract;
the script rejects them instead of guessing that an old reference applies.

The FP64 mathematical graph stays the same when changing storage precision;
its inputs are the exact parameter/input values stored by that version.
The fixed reference changes its storage and intermediate arithmetic to match
each version. Full-buffer and tiled implementations share the intended graph
but have separate source contracts and separate actual C++ executions.

| Reference component | Implementation |
|---|---|
| ResNet-18/50 mathematical graph and FP64 operations | [resnet.py](resnet.py) |
| Llama prefill/decode mathematical graph and FP64 operations | [llama.py](llama.py) |
| Repaired Q16.5 fixed operations | [repaired.py](repaired.py) |
| Q32.10 fixed operations and precision-specific reference classes | [wide.py](wide.py) |
| Exact source-to-reference version mapping | [contracts.json](contracts.json) |

## Verdicts and coverage

The script writes `summary.json`, `summary.txt`, `verification.log`, source
and reference-code snapshots, input manifests, both reference outputs, the
actual C++ outputs, and per-tensor `comparison.json` records.

- **Implementation PASS** requires zero raw integer-code mismatches.
- **Mathematical PASS** requires maximum absolute error <= 0.1 and relative
  L2 error <= 0.01 for every compared tensor. Relative L2 is
  `||actual/2**(W-I) - fp64||_2 / ||fp64||_2`. For a zero reference, only the
  absolute limit applies. These are the selected synthetic verification
  limits, not a trained-model accuracy standard.
- All values must be finite, tensor shapes and byte counts must match, and
  original source hashes must remain unchanged.
- Exit code **0** means both comparisons passed (or successful inspection,
  explicitly labelled `INSPECTED_NOT_EXECUTED`); **1** means numerical failure;
  **2** means invalid design/interface, unsupported version, tool failure, or
  incomplete execution. Missing results are never converted into a PASS.

ResNet-18 exports 12 persistent checkpoints without changing the DUT: stem
BN/ReLU output, maxpool output, eight residual-block outputs, global pool,
and all 1,000 logits. These total 1,757,672 codes. Full-buffer checkpoints are
observed through existing externally linked BRAM globals; tiled checkpoints
are observed through existing DRAM ports. ResNet-50 exports 20 checkpoints
(16 bottleneck blocks), totaling 6,525,928 codes. Its independent mathematical
graph preserves the declared production projection branch, which has no extra
projection BN; it does not claim equivalence to an unmodified torchvision
ResNet-50 with pretrained parameters. Overwritten intermediate values
are not claimed as observed checkpoints.

Llama has separate C++, fixed-reference, and FP64-reference cache state.
The runner transfers the actual C++ prefill cache to the C++ decode process;
each reference retains its own cache. Every emitted logit and populated cache
entry is selected for comparison. Invalid constants or unrepresentable
controls stop execution before allocating an 8B parameter store, with
`INVALID_DESIGN` and both comparisons `NOT_RUN`.

## Historical v1 findings (before the production repair)

The preserved [validation record](../results/existing/validation_summary.json)
contains the historical runs, source/reference hashes, per-checkpoint metrics, and
links to the simulation reports. The verifier regression suite passed all
18 tests. These records concern the saved production sources only.

The direct run of the saved tiled ResNet-18 (seed 42) matched the fixed
reference at every exported code. It failed the mathematical limits: maximum
logit error was **1.986021**, and relative logit L2 error was **1.424389**.
Per-product truncation accumulates error, and the stored pooling implementation
casts its divisor 49 to `data_t`, producing -15. Thus exact implementation
agreement alone does not establish closeness to the mathematical model.

The saved full-buffer ResNet-18 leaves its weight and BN BRAMs unloaded.
Its actual Vitis C-simulation binary raises **SIGFPE in the first BN**, where
the denominator is zero. Both references can execute with the supplied
parameters, but no valid C++ output exists to compare.

The saved Llama pair is rejected before a full-model run:

- Token IDs and positions use `data_t`, representing only integer IDs 0..15.
- RMSNorm casts 4096 to `ap_fixed<32,10>`, producing zero before division.
- Attention casts 128 to `ap_fixed<16,5>`, producing zero before square root
  and division.

Llama reference primitives and cache machinery are tested, but these findings
prevent claiming full-8B execution or numerical agreement for the saved pair.
Future fixes must be made through the production generation flow and receive
new source contracts and fresh results. None of these runs establishes RTL,
FPGA execution, pretrained-model accuracy, or equivalence to old synthesis
measurements for a changed design.

## Tests

```bash
cd verification
python -m pytest tests/test_existing.py tests/test_production.py tests/test_production_graph.py tests/test_production_wide.py -q
```

Tests exercise the actual Vitis headers/libraries: BN, signed division,
per-product conversion, pooling, partial linear/reduction tiles, RMSNorm,
SwiGLU, fixed square root and exponential across all 131,072 input codes of
the relevant 17-bit format, and RoPE across the saved 2048-position context.
Additional tests cover source/interface changes, corrupt inputs, separate
verdicts, nonfinite outputs, FP64 prefill/decode cache continuity, invalid Llama
preflight even with representable token IDs, inspection without a numerical
PASS, and vendor primitive loading in a fresh process.
