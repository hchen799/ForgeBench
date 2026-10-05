# Tiled Llama 3 8B: HLS C++ and PyTorch verification

This dedicated path generates `llama_prefill` and `llama_decode`, with shared
external-memory parameters and persistent KV caches. It does not modify the
older generic Llama generator in `generate_code.py`.

See [the validation record](VALIDATION.md) for measured results and hardware
verification limitations.

The full profile has 32 layers, hidden size 4096, FFN size 14336, 32 query heads,
8 KV heads, head dimension 128, and vocabulary size 128256: **8,030,261,248
parameters** including independent embedding and output matrices. It uses
bias-free projections, RMSNorm with epsilon 1e-5, adjacent-pair RoPE with theta
500000, grouped-query causal attention, and SwiGLU. Architecture conventions
follow the [official Meta Llama 3 implementation](https://github.com/meta-llama/llama3/blob/main/llama/model.py).

## Run the full short test, then the long test

Run from `ForgeBench/scale_models`. Use new, empty model/run directories; tools
refuse to overwrite an existing dataset or result.

For a single command that automatically gates the long run on short-run success:

```bash
bash verify_tiled_llama3.sh verification_runs/my_llama3_suite
```

Or run the stages explicitly:

```bash
python generate_tiled_llama3.py \
  --profile llama3_8b --max-ctx 8192 \
  --output-dir verification_runs/llama3_8b/design

python golden_tiled_llama3.py model \
  --project-dir verification_runs/llama3_8b/design \
  --model-dir verification_runs/llama3_8b/model --seed 42

python golden_tiled_llama3.py run \
  --model-dir verification_runs/llama3_8b/model \
  --run-dir verification_runs/llama3_8b/short \
  --prefill 4 --decode 2 --trace ops --threads 16

# Run this after the short test passes. Reuses exactly the same 8B parameters.
python golden_tiled_llama3.py run \
  --model-dir verification_runs/llama3_8b/model \
  --run-dir verification_runs/llama3_8b/long2048 \
  --prefill 2048 --decode 2 --trace layers --threads 16
```

`run` performs dataset preparation, both PyTorch references, **actual Vitis HLS
C simulation**, and the complete checkpoint comparison. `--vitis /path/to/vitis_hls`
selects an installation. It searches PATH and the local archived Vitis 2024.1
installation. `--backend native` explicitly selects a faster-to-build ordinary
G++ host simulation of the same C++ kernels; it is identified separately in the
report, and is not described as Vitis simulation.

The short test covers 6 token positions and 769,536 vocabulary logits. The long
test covers 2,050 positions and **262,924,800 logits**, not just the last token or
the top-1 result. Prefill is processed in chunks of at most 16 tokens, with the
KV cache retained across every chunk and both decode calls. `--trace layers`
compares all layer outputs, all newly written keys/values, final normalization,
and every vocabulary logit. `--trace ops` additionally compares each projection,
normalization, RoPE, attention, residual, and FFN intermediate.

Token IDs are deterministic random integers in `[0, 128255]`; the first is
128255 to exercise large vocabulary IDs. Decode uses predetermined tokens
(teacher-forced validation), **not** a free-running text sampler. Both
implementations see identical tokens, including when their floating-point
argmax predictions differ. No tokenizer or gated model download is needed.

Longer tests are configurable: `--prefill 4096 --decode 2`, or the cache-boundary
test `--prefill 8190 --decode 2`. The total must not exceed the generated
`--max-ctx` capacity. A 2048-token test does not establish correctness at every
possible position up to 8192. `--chunk` controls runtime prefill chunk size up
to the generated tile size and can exercise partial tiles.

## What is compared and what precision is used?

| Path | Arithmetic and purpose |
| --- | --- |
| Generated HLS C++ | Signed int16 codes representing Q5.11 weights, activations, caches, and logits; exact signed int64 linear products/reductions with 22 fractional bits; explicit nearest rounding with ties toward positive infinity and saturation |
| Fixed-point PyTorch | Independently implemented matching rounding, saturation, RMSNorm, RoPE, softmax, and SiLU; requires **zero integer-code mismatches** against C++ |
| Mathematical PyTorch | FP64 operations with true RMSNorm, trigonometric RoPE, softmax, and SiLU, using the same quantized input parameters; measures the effect of fixed-point arithmetic |

The fixed-point reference uses FP64 matrix multiplication of **integer codes**
for speed. This remains exact: a linear dot product has absolute sum at most
2^45 under the dimension constraints, below FP64's exact-integer limit 2^53.
Attention's weighted-value sum is at most 2^50 at 8192 positions. Conversion back
to int64 precedes the specified rounding. This is not a float model merely
rounded at the final output.

Q5.11 denotes 16 total bits with 5 integer bits **including sign**, range
`[-16, 15.99951171875]`, spacing `1/2048`. The storage format is little-endian
int16; decode to real values by dividing by 2048. Tokens are int32, positions
are integers, and all parameter/cache offsets are 64-bit. Accumulators are not
restricted to the activation format.

Nonlinear implementation details:

- RMSNorm sums integer squares, divides by hidden size, adds epsilon represented
  by 42 at Q22, uses a rounded integer square root, and a Q22 reciprocal.
- RoPE uses an external Q11 sine/cosine table for every absolute cache position.
- Attention excludes future positions completely; it does not use a small
  finite negative masking constant. Scores are Q5.11, followed by an exponential
  ROM indexed by the nonnegative difference from the maximum score. Exponentials
  use unsigned Q22; division of weighted values is rounded once at the output.
- SiLU is a 65,536-entry Q11 lookup table, covering every int16 input code.

`comparison.json` records exact mismatches and FP64 errors for every checkpoint.
`logit_errors` reports max absolute error, MAE, RMSE, and maximum relative error
with denominator floor `1/2048`. Relative errors near zero can be large even
when absolute errors are small. These mathematical errors are **diagnostic**:
no arbitrary accuracy threshold is used to claim pretrained-model accuracy.
Saturation counts are recorded separately; an exact match alone would not
establish that a chosen numeric format has adequate dynamic range.

## Random parameter ranges

The model is a full-size **synthetic numerical-validation model, not Meta's
pretrained checkpoint**. Every parameter is materialized; dense matrices are
not replaced with zero, diagonal, sparse, or low-rank stand-ins. Seeds are
stable per named tensor. Values are drawn uniformly on the Q11 grid:

- Token embeddings: `[-0.5, 0.5]`.
- RMSNorm scales: approximately `[0.9, 1.1]`.
- Ordinary projections and output head: `±sqrt(3/fan_in)`.
- Attention output and FFN down projections: the same fan-in bound divided by
  `sqrt(2 * number_of_layers)` to keep the 32-layer residual stack well scaled.

`model.json` records the actual inclusive integer-code bounds for each tensor.
This validates architecture, indexing, arithmetic, and cache behavior under
controlled random inputs. It does not establish language quality or acceptable
quantization error for pretrained weights; those need a separately calibrated
checkpoint-import and evaluation workflow.

## Files, resources, and separate stages

The shared `weights.bin` is about 14.96 GiB. Default CPU reference caching
reserves up to 64 GiB for decoded FP64 integer-code matrices; allow roughly
90 GiB host RAM for comfortable full-model verification including memory maps,
the two independent KV caches, and working buffers. `--cache-gib` limits that
cache, but a small limit increases repeated weight conversion substantially.
`--threads` controls PyTorch CPU threads. These are CPU references, independent
of CUDA availability. Full runs can be lengthy; the program prints per-call
progress. Allow at least 40 GiB free disk for the shared model and the two cases.

Each case contains `tokens.bin`, `case.txt`, `case.json`, `fixed/`, `float/`,
`reference.json`, the C++ source snapshot under `build/`, `csim/`, `csim.log`,
`csim.json`, and `comparison.json`. Parameter, source, input, and reference
hashes prevent accidental comparison of different datasets or stale references.

The stages can also be run separately:

```bash
python golden_tiled_llama3.py prepare --model-dir MODEL --run-dir CASE --prefill 4 --decode 2
python golden_tiled_llama3.py reference --run-dir CASE
python golden_tiled_llama3.py csim --run-dir CASE
python golden_tiled_llama3.py compare --run-dir CASE
```

`compare` can be rerun without simulating again. Do not edit reference scripts
between stages: reference provenance is checked. Keep failed cases for diagnosis
and use a fresh case directory for a corrected run.

## HLS interfaces and hardware scope

Generated `top.h` declares separate `llama_prefill` and `llama_decode` entry
points. They share the exact same weights, RoPE table, KV-cache layout, and
scratch layout. Call positions must advance contiguously for a sequence; the
host owns this protocol. Start a new sequence at position zero. Only newly
written cache rows are consumed, so clearing unused rows is not required by
the kernel. Return status 1 rejects invalid lengths/positions and status 2
rejects invalid token IDs before modifying outputs or caches.

AXI master interfaces connect parameters, tables, caches, scratch, and logits
to external memory; AXI-Lite controls pointers and scalar arguments. The matrix
engine stages 128x128 parameter tiles and accumulates all input chunks before
committing a saturated result. Input and output scratch buffers are distinct
where aliasing would overwrite inputs needed by later output tiles.

`run_prefill_hls.tcl` and `run_decode_hls.tcl` invoke synthesis separately.
Their ZU9EG part/10 ns clock are development defaults, **not a claim that a
particular board has enough external memory or meets a performance target**.
Weights alone require approximately 15 GiB; full-capacity int16 K/V storage
requires another 1 GiB. Select a suitable memory system and board before
deployment. C simulation does not establish RTL cosimulation, timing closure,
resource fit, board throughput, or on-board correctness. Trace code and mmap
testbench code are host-only and excluded from synthesis.

## Regression tests and a small smoke run

```bash
python -m pytest -q llama3_verification/tests
python generate_tiled_llama3.py --profile tiny --output-dir verification_runs/tiny/design
python golden_tiled_llama3.py model --project-dir verification_runs/tiny/design --model-dir verification_runs/tiny/model
python golden_tiled_llama3.py run --model-dir verification_runs/tiny/model --run-dir verification_runs/tiny/case --prefill 30 --decode 2 --threads 2
```

Tests cover full architecture/layout, invalid configurations, signed rounding,
nonlinear tables, zero RMSNorm input, grouped-query causal attention, chunked
prefill versus whole-sequence evaluation, decode/cache boundaries, integer
square roots, invalid C++ arguments, non-multiple tile dimensions, extreme
int16 values with intentional saturation, end-to-end native C++ trace comparison,
and deliberate input corruption. The tiny profile is only a debugging aid;
its results must not be presented as a full 8B validation.
