# Accelerator verification: ResNet-18, Llama 3, and exported tensors

This is the central verification directory. It contains the configurable HLS
C/C++ generators, PyTorch golden references, tensor comparison tools, tests,
and preserved earlier verification implementations and results.

**Two different questions are answered on every architecture run:**

1. **Implementation agreement:** Does actual compiled C/C++ produce exactly the
   same stored integer codes as the fixed-point PyTorch implementation reference?
   Every selected checkpoint must match; ResNet shared-exponent events must also match.
2. **Mathematical error:** How far are those C/C++ values from a mathematical
   FP64 PyTorch evaluation using the **same quantized input and parameters**?
   This is diagnostic unless you explicitly set numerical tolerances.

Neither result establishes trained-model accuracy. Parameters are reproducible
synthetic random values, not pretrained ResNet or Meta Llama weights. Vitis
C simulation executes the generated HLS C/C++ on the host, not on an FPGA.

## Results at a glance

Validated here with Python 3.9, PyTorch 2.2.0, NumPy 1.24.4 and Vitis HLS
2024.1.2. New configurable-path runs were completed on 2026-10-03.
All rows below have **zero integer mismatches**. Numerical errors are against FP64.

| Implementation / case | Storage `<W,I>` | Execution | Checkpoints | Max absolute logit error | Logit RMSE | Saturations |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Configurable full ResNet-18, seed 42 | `<16,5>` | Vitis C simulation | 65 | 6.13164e-4 | 2.13101e-4 | 0 |
| Configurable full ResNet-18, seed 42 | `<32,10>` | Vitis C simulation | 65 | 3.20913e-7 | 1.04816e-7 | 0 |
| Configurable full ResNet-18, custom tiles/shift, seed 43 | `<24,8>` | Vitis C simulation | 65 | 2.42109e-5 | 7.06171e-6 | 0 |
| Configurable tiny Llama, 30 prefill + 2 decode | `<16,5>` | Vitis C simulation | 390 | 5.47864e-3 | 1.06748e-3 | 0 |
| Configurable tiny Llama, 30 prefill + 2 decode | `<32,10>` | Vitis C simulation | 390 | 2.66215e-6 | 5.48875e-7 | 0 |
| Configurable custom Llama, 31 prefill + 2 decode | `<24,8>` | Native C++ | 312 | 1.52328e-4 | 2.99027e-5 | 0 |
| Preserved full Llama 3 8B, 4 prefill + 2 decode | `<16,5>` | Vitis C simulation | 1,737 | 1.50993e-2 | 2.87035e-3 | 0 |
| Preserved full Llama 3 8B, 2048 prefill + 2 decode | `<16,5>` | Vitis C simulation | 12,740 | 6.99633e-2 | 3.90948e-3 | 8 |

Each ResNet run compared **8,054,760 codes**, including all 1,000 logits.
The full-8B short and long cases compared **19,889,664** and **674,368,000**
codes respectively. The long case includes all **262,924,800 logits**, new
KV-cache entries, layer outputs, final normalization, and both decode calls.
Its eight saturation events were in SwiGLU; exact agreement does not make the
calculation saturation-free. The earlier tiny 8190+2 cache-boundary test also
passed, but that is **not** a full-8B 8192-token test.

The full-8B results belong to the frozen earlier `<16,5>` implementation and
were integrity-checked and re-compared after this reorganization. They are not
new full-8B runs of the configurable generator. **Full 8B at `<32,10>`, full 8B
at 8192 tokens, and RTL/board execution have not been validated.** See the
machine-readable [result inventory](results/validation_summary.json),
[coverage notes](results/README.md), and the detailed
[earlier Llama validation record](legacy/llama3_verification/VALIDATION.md).

## Quick start

Run these commands from `scale_models`. The entry point can also be called by
absolute path from another directory; command-line paths are relative to your
working directory. Every `--output` must be a **new directory**. Existing runs
are not overwritten. Dependencies are Python 3.9+, PyTorch, NumPy, a C++14
compiler for native Llama, and Vitis HLS for ResNet or the Vitis Llama backend.

Small Llama, no Vitis required:

```bash
python verification/verify.py run \
  --config verification/configs/llama_tiny_q16_5.json \
  --output verification/runs/my_llama16

python verification/verify.py run \
  --config verification/configs/llama_tiny_q32_10.json \
  --backend vitis \
  --output verification/runs/my_llama32
```

Full ResNet-18 at either requested precision:

```bash
python verification/verify.py run \
  --config verification/configs/resnet18_q16_5.json \
  --output verification/runs/my_resnet16

python verification/verify.py run \
  --config verification/configs/resnet18_q32_10.json \
  --output verification/runs/my_resnet32
```

If `vitis_hls` is not on PATH, append `--vitis /path/to/vitis_hls`.
The local installation at
`/tools/software/xilinx/ARCHIVE/Vitis_HLS/2024.1/bin/vitis_hls` is also detected
when present; other machines should supply their own installation.

Each `run` generates a new C/C++ design, saves random inputs/weights, runs both
PyTorch references, compiles and runs the actual C/C++, and compares all traces.
Changing precision changes **the generated datapath and saved data**, not just
the display format. It is not valid to reinterpret old 16-bit files as 32-bit.

Read the results without guessing which values are being compared:

```bash
cat verification/runs/my_resnet32/summary.json
less verification/runs/my_resnet32/verification.log
```

## Full 8B: short first, then long context

These commands exercise the **new configurable path**, not the archived run.
Use the same parameter store to avoid generating another full model:

```bash
python verification/verify.py run \
  --config verification/configs/llama3_8b_short_q16_5.json \
  --model-dir verification/runs/my_8b_model \
  --output verification/runs/my_8b_short

python verification/verify.py run \
  --config verification/configs/llama3_8b_long_q16_5.json \
  --model-dir verification/runs/my_8b_model \
  --output verification/runs/my_8b_long
```

The short configuration uses four prefill tokens and two one-token decode
calls. The long configuration uses 2048 prefill tokens and two decode calls,
with an 8192-position cache. Decode inputs are saved, deterministic token IDs,
**not** tokens sampled from predicted logits. This tests the shared KV cache
and position-dependent computation without divergent token-generation paths.

Use `--prefill N --decode M` to change the schedule; the total must fit
`model.max_ctx`. `test.chunk` controls how prefill is split into calls and
cannot exceed `model.prefill_tile`. For a long run, `trace: "layers"` saves
layer outputs, every new cache entry, final normalization and all logits;
`trace: "ops"` additionally saves intermediate operators. Neither mode samples
the vocabulary: every emitted logit is compared.

Full 8B has 8,030,261,248 stored parameters: approximately **14.96 GiB** at
16-bit and **29.92 GiB** at 32-bit. A complete FP64 matrix cache is another
approximately 59.83 GiB, plus activations/caches/traces/builds. `cache_gib` limits
the matrix cache, not total process memory. Long FP64/logit traces alone can
occupy several GiB. The earlier 2048+2 run took about 1292 seconds for both
references and 4127 seconds for the Vitis invocation on this server; these are
host runtimes, not FPGA latency. Wide-integer 32-bit reference execution may
be substantially slower. Start with the tiny examples when changing formats.

Parameter reuse checks configuration, seed, checksums, and generated source
identity. A different format/configuration requires a different model directory.
The new Llama generator uses continuous real random draws before quantization;
the frozen legacy generator sampled integer codes directly. The same seed does
not imply identical legacy/new parameter files.

## Configuration interface

Example:

```json
{
  "schema_version": 1,
  "family": "llama3",
  "model": {"profile": "tiny", "max_ctx": 64, "prefill_tile": 7},
  "precision": {
    "word_bits": 32,
    "integer_bits": 10,
    "rounding": "nearest_ties_positive",
    "overflow": "saturate"
  },
  "parameters": {"seed": 42},
  "test": {"seed": 43, "prefill": 61, "decode": 2, "chunk": 6, "trace": "ops"},
  "execution": {"backend": "vitis", "threads": 8, "cache_gib": 1},
  "comparison": {"float_atol": null, "float_rtol": null, "fail_on_saturation": false}
}
```

Unknown fields and unsupported modes are errors, not silently ignored.
Defaults are seed 42 for parameters, seed 43 for Llama tokens, 4+2 tokens,
operator tracing, Vitis, eight PyTorch threads, a 64-GiB matrix cache, and
diagnostic-only numerical error. Explicit example files override some defaults.
Supported command-line overrides are `--word-bits`, `--integer-bits`,
`--backend`, `--vitis`, `--threads`, `--prefill`, and `--decode`. The effective
values, including inferred accumulator widths, are saved in `resolved_config.json`.
That file is an audit record with derived fields, not an input-schema template.

### Llama variants

`model.profile` is a starting template (`tiny` or `llama3_8b`), not proof that
overridden dimensions are still a standard 8B model. The resolved dimensions
and packed parameter count are authoritative. Configurable fields are:

- `layers`, `hidden`, `ffn`, `q_heads`, `kv_heads`, `head_dim`, `vocab`;
- `max_ctx`, `prefill_tile`, `tile_in`, `tile_out`;
- `rope_theta` and `epsilon`.

Require `hidden = q_heads * head_dim`, `q_heads % kv_heads = 0`, even
`head_dim`, positive dimensions, `hidden/ffn <= 32768`, and
`prefill_tile <= max_ctx <= 32768`. The upper context bound is a configurable
datapath bound, **not** a claim that every such context was tested or that a
pretrained checkpoint supports it. Non-multiple tiles are supported. The
[custom example](configs/llama_custom_q24_8.json) uses hidden 60, FFN 131,
three query heads, one KV head, vocabulary 259, and tiles 17/19.

### ResNet variants

This adapter models the dedicated tiled ResNet-18 graph: one `3x224x224`
CHW input and 1,000 output logits, with no final softmax. As in the original
generator, downsample shortcuts have no BN, and convolution/classifier biases
are absent. It is not a drop-in torchvision pretrained ResNet-18.

Configurable fields are `tile_c` (4–256), `tile_h/tile_w` (1–28),
`acc_word_bits`, `acc_integer_bits`, `guard`, and the `shifts` dictionary.
Accumulator fractional bits must equal twice the storage fractional bits;
the default is `<2W,2I>`, up to 64 accumulator bits. `guard` defaults to
`min(256, 2^(acc_integer_bits-2))`. Output shifts are integers in [-31,31].
Valid names are those emitted by `generate_tiled_resnet18.build_shift_names`;
all are listed in a run's resolved configuration. Unspecified shifts are zero.
For example, `"shifts": {"SHIFT_STEM": 1}` scales that convolution output.
The [custom example](configs/resnet18_custom_q24_8.json) changes all three tile
sizes and the stem shift.

Other ResNet depths, input sizes and unrelated network graphs are not silently
accepted by this adapter. Use the generic exported-tensor interface below, or
implement an additional architecture adapter.

## Precision and arithmetic contract

`<W,I>` means **signed** fixed point, including the sign bit in `I`, with
`F = W-I` fractional bits. A raw integer code represents `code / 2^F`.
The supported domain is `8 <= W <= 32`, `2 <= I < W`, and `F <= 24`.
Logical widths need not be a multiple of eight. Only nearest rounding with
ties toward positive infinity (`AP_RND`) and saturation (`AP_SAT`) are
implemented. Wrapping, truncation modes, unsigned activations, 64-bit storage,
and floating-point accelerator outputs are not supported by this fixed-code
interface and are rejected.

| Storage format | Fraction bits | LSB | Real range | Binary container | Default ResNet accumulator |
| --- | ---: | ---: | --- | --- | --- |
| `<16,5>` | 11 | 0.00048828125 | [-16, 15.99951171875] | little-endian signed int16 | `<32,10>` |
| `<32,10>` | 22 | 2.384185791015625e-7 | [-512, 511.9999997615814] | little-endian signed int32 | `<64,20>` |
| `<24,8>` | 16 | 0.0000152587890625 | [-128, 127.99998474121094] | little-endian signed int32 | `<48,16>` |

For non-container widths, codes are sign-extended to the container width;
they are not densely packed. File sizes and logical code ranges matter.
Tokens remain little-endian int32 vocabulary indices, independent of activation
precision. Mathematical references use FP64; generic supplied floating
references may be FP16, FP32, or FP64, with metrics evaluated in FP64.

The fixed PyTorch reference is integer-code execution, not FP32 neural-network
execution with a final cast. At wider precisions it uses exact limb-decomposed
PyTorch FP64 matrix products and Python/NumPy arbitrary-width integer
reconstruction. The fast GEMM path is used only under a conservative integer
exactness bound; 32-bit products/sums must not be naively stored in int64 or
treated as always exactly representable in FP64. Directed tests include sums
larger than int64 and cancellation at int32 extrema.

Llama uses signed 64-bit accumulators for storage widths <=16 and F<=11,
otherwise signed 128-bit accumulators. Native C++ uses `__int128` for the
wide path; synthesis uses `ap_int<128>`. RMSNorm uses an unsigned 128-bit
integer square root. SiLU and negative exponential use a fixed 1/2048-spaced
lookup grid over [-32,32] and [0,32] respectively; higher fractional precision
uses integer linear interpolation. SiLU uses its asymptotic tails beyond that
range, and exponential weights beyond delta 32 are zero. The fixed reference
reproduces those approximations; FP64 uses the mathematical functions.

ResNet uses real vendor `ap_fixed`, including widened expression temporaries,
truncating division expressions, per-assignment saturation and shared tile
exponents. For wide BN arithmetic, the installed vendor square-root overload
returns zero when its input has more than 32 fractional bits. The configurable
generator therefore emits an exact wide integer square-root helper, retaining
the widened return type required by the subsequent division. This is essential
for `<32,10>` storage / `<64,20>` accumulation. The effective epsilon has a
minimum of one accumulator code. At `<16,5>` the helper matches the original
vendor result; historical projects are not modified.

**Known ResNet reduction issues are deliberately still modeled:** later chunks
do not compensate for an existing shared exponent, and accumulation may
saturate before the guard check. Pooling also has a saturating intermediate
sum. Bit-exact agreement validates the implementation reference, not the
mathematical correctness of these algorithms. See
[the directed arithmetic findings](legacy/tiled_resnet18_verification/FINDINGS.md).

## Random input and parameter generation

Saved tensors, checksums and source snapshots define a case; seeds alone do
not guarantee cross-version library reproducibility. All three executions
(C/C++, fixed reference, FP64 reference) consume the same saved quantized data.

| Tensor | Distribution before quantization |
| --- | --- |
| ResNet image | uniform [-1,1] |
| ResNet conv/FC weights | uniform +/-sqrt(3/fan_in) |
| ResNet BN gamma / variance | uniform [0.9,1.1] |
| ResNet BN beta / mean | uniform [-0.05,0.05] |
| Llama embeddings | uniform [-0.5,0.5] |
| Llama norm gamma | uniform [0.9,1.1] |
| Llama linear weights | uniform +/-sqrt(3/fan_in) |
| Llama output/down projections | preceding bound divided by sqrt(2*layers) |

Llama uses independent deterministic per-parameter random streams. Changing
precision while keeping the architecture and seed gives the same underlying
real draws before quantization. Token IDs are uniform within the vocabulary;
the first two IDs explicitly exercise the upper vocabulary boundary and ID 17
(or the largest available ID for a very small vocabulary). RoPE tables use the
configured theta and positions, then are quantized. BN variances must stay
positive after quantization. These ranges aim for useful nominal functional
tests; they are not a substitute for directed overflow tests or trained weights.

## Generic interface for unrelated accelerators

Use `compare-tensors` when you already have outputs from another accelerator
and its golden model. It does not infer the graph, generate that model, compile
the accelerator, or attest where supplied arrays came from. You supply three
arrays per checkpoint: actual fixed codes, expected fixed codes, and expected
mathematical floating values. Checkpoints may be intermediate tensors or just
the final output; the coverage claim applies only to what you supply.

Example manifest:

```json
{
  "schema_version": 1,
  "model": "my accelerator and test-case identifier",
  "precision": {"word_bits": 32, "integer_bits": 10},
  "checkpoints": [
    {
      "name": "output",
      "shape": [1, 1000],
      "actual": {"path": "actual.bin", "dtype": "<i4"},
      "fixed": {"path": "golden_fixed.npy"},
      "floating": {"path": "golden_float.npy"}
    }
  ]
}
```

```bash
python verification/verify.py compare-tensors \
  --manifest /path/to/manifest.json \
  --output verification/runs/my_external_accelerator
```

File paths are resolved relative to the manifest. Arrays are C-order; `.npy`
shapes must match exactly, and raw files must have exactly the declared byte
count. Raw floating arrays require an explicit dtype, such as `"<f8"`.
Fixed arrays require signed int16 containers when W<=16, otherwise signed
int32 containers. An explicit raw dtype declares its byte order; the architecture
adapters always export little-endian. Every input is checksummed and its actual
path/dtype/checksum logged; optional `sha256` fields enforce expected checksums.
Duplicate names, nonpositive shapes, out-of-range logical codes, non-finite
values, wrong dtypes and malformed files are rejected. Saturation is reported
as **unknown**, not zero: final exported values cannot establish whether an
intermediate operation clipped.

An executable comparator-only demonstration is included:

```bash
python verification/examples/make_generic_fixture.py \
  --word-bits 32 --integer-bits 10 \
  --output verification/runs/generic_demo_inputs
python verification/verify.py compare-tensors \
  --manifest verification/runs/generic_demo_inputs/manifest.json \
  --output verification/runs/generic_demo_comparison
```

The demo intentionally copies golden values into `actual.npy` and labels this
in its manifest/log. **It is not evidence of an accelerator passing.** Replace
the actual tensor with a real export for a real verification case.

## Logs, tolerances, and exit codes

The main `verification.log` is written while execution proceeds. It records the
command, UTC timestamps, library versions, effective model/precision, the two
comparison definitions, inputs/distributions/seeds/checksums, source identities,
phase timings and C-simulation outcome. Every compared checkpoint has its name,
shape, element count, exact mismatch count, maximum code/LSB error, maximum
absolute error, MAE, RMSE and maximum relative error. Mismatches include the
first differing index and actual/expected codes. JSON reports retain these
details for tooling. ResNet's simulator writes a separate live log at
`case/csim/simulation.log`, which is copied into the main log when it completes;
Llama streams simulator output into the main log and `case/csim.log`.

For actual dequantized values `a` and FP64 reference values `r`:

```text
absolute error = abs(a-r)
MAE            = mean(abs(a-r))
RMSE           = sqrt(mean((a-r)^2))
relative error = abs(a-r) / max(abs(r), one storage LSB)
```

Large relative errors near zero do not by themselves imply large absolute
errors. For example, the earlier ResNet numbers you saw—max_abs
0.0006131635532236057, MAE 0.00017244851009612928, RMSE
0.00021310109084550356—are **C++ `<16,5>` logits dequantized to real values
versus FP64 PyTorch**, not integer mismatch measurements. The corresponding
C++ versus fixed-reference comparison had zero mismatches.

To impose a mathematical gate, set both `comparison.float_atol` and
`comparison.float_rtol` in an architecture config, or both top-level fields in
a generic manifest. A value passes when
`abs(a-r) <= atol + rtol*abs(r)`. This applies at **every selected checkpoint**,
not only logits. Tolerances must be finite and nonnegative. Exact-code
agreement is still mandatory, even when numerical tolerances are loose.
Set `comparison.fail_on_saturation: true` to make an observed architecture
reference saturation fail the overall run; by default it is a warning.

Exit codes:

- `0`: exact comparison passed, and all explicitly requested policies passed;
- `1`: completed comparison failed exactness, numerical tolerances, or the
  requested no-saturation policy;
- `2`: invalid configuration/data, missing tools, stale provenance, compilation/
  execution failure, or another error preventing a complete comparison.

`summary.json` separates `implementation_verdict`, `mathematical_verdict`, and
`overall_verdict`. `DIAGNOSTIC_ONLY` means no mathematical acceptance threshold
was requested; it is not an accuracy pass. An error before comparison writes
`error.json` and a traceback to the log. Existing output-directory rejection
happens before opening a new log so previous data are left untouched.

## Recompare existing cases without rerunning C/C++

```bash
python verification/verify.py compare \
  --run-dir verification/runs/llama3_8b/long2048 \
  --output verification/runs/my_long_recomparison

python verification/verify.py compare \
  --run-dir verification/runs/validated_resnet_q32_10/case \
  --output verification/runs/my_resnet_recomparison
```

This rechecks manifests, source snapshots, parameter/input files, reference
fingerprints/checksums, and saved simulation provenance before comparing.
It does not rerun the accelerator. Old version-1 cases dispatch to the frozen
implementation; configurable version-2 cases dispatch to the current adapter.
New comparison reports are written only under the new output directory; a
symlink-based `saved_case_view/` keeps the saved case and its old reports unchanged.
Changing reference code makes existing current-adapter references stale: create
a fresh run rather than bypassing the checks. Frozen legacy Llama comparison
retains diagnostic-only FP64 semantics; use exported-tensor comparison if you
need a new tolerance policy for those tensors.

## Directory and artifact guide

```text
verification/
  verify.py                 unified run / compare / compare-tensors CLI
  precision.py              explicit formats and overflow-safe integer helpers
  reporting.py, generic.py  logging, common summaries, tensor-only comparison
  configs/                  runnable ResNet/Llama JSON examples
  models/llama3/            configurable generator, kernel, references, runner
  models/resnet18/          configurable adapter to the dedicated ResNet emitter
  tests/                    configurable arithmetic, C++, CLI and corruption tests
  examples/                 clearly labelled generic comparator fixture
  results/                  small reviewable result inventory and test record
  legacy/                   frozen original implementations, tests and findings
  runs/                     local vectors, builds, traces, logs and full reports
  run_legacy_llama3.sh       original suite orchestration, retained for compatibility
```

A new architecture job contains `verification.log`, `resolved_config.json`,
`timings.json`, `summary.json`, `comparison.json`, `design/` and `case/`.
Llama also creates `model/`, unless `--model-dir` supplies a shared location.
`design/` contains top/header/testbench/configuration and Vitis TCL files.
`case/fixed/` and `case/float/` are golden checkpoints; `case/csim/` contains
**actual C/C++ outputs**, never a copy of golden outputs. Llama model weights
are packed in `model/weights.bin`; ResNet ports are separate `case/inputs/`
files. Manifests describe layouts and SHA256 checksums.

The former `verification_runs/` directory now points to `verification/runs/`.
Former package names are compatibility symlinks into `verification/legacy/`,
and root `golden_tiled_*.py`/`verify_tiled_llama3.sh` remain compatibility
entry points. The original root generators retain their existing behavior.
For configurable precision, use **this new CLI**, not a legacy wrapper. Keeping
the old path aliases preserves absolute paths in the existing large manifests.
Moving the entire checkout to another machine requires regenerating cases or
an explicit manifest migration; do not edit checksums to suppress stale-data
errors. `runs/` is git-ignored; small `results/` records are separate so code
review does not require committing tens of GiB. Intermediate development probes
are retained locally but are not authoritative current-adapter validation runs.

## Regression tests and remaining scope

```bash
VITIS_HLS_INCLUDE=/path/to/Vitis_HLS/include \
  python -m pytest -q verification/tests \
  verification/legacy/llama3_verification/tests \
  verification/legacy/tiled_resnet18_verification/tests
```

The tests cover rounding/saturation, limb arithmetic and int32 extremes,
non-byte widths, generated C++ operators, actual vendor wide integer/fixed-point
operators, BN and square roots, causal/GQA/cache behavior, partial tiles,
corruption/stale-data rejection, generic manifest validation and log/exit-code
behavior. Vendor-header tests skip when headers are absent; a skip is not a
successful hardware-arithmetic validation. See [the test record](results/regression_tests.log).
The completed combined run has **77 passing tests with no failures or skips**.

The frozen full-size `<16,5>` Llama tops completed HLS synthesis (prefill and
decode), with resource/timing estimates documented in the earlier validation
record. That does not establish synthesis/timing closure for every new model or
precision. The configurable projects emit synthesis TCL scripts, but new
wide-format full-model RTL synthesis, RTL cosimulation, board deployment,
pretrained-checkpoint import, model-quality evaluation, performance tuning,
and automatic quantization calibration remain outside the completed checks.
