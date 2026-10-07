# Data types: how `data_type` in a JSON config becomes C++

The design's numeric type is the `data_type` field of its JSON config. The tool backend (`backends/`) lowers it to the tool's C++ type.

| `data_type` in the JSON | Vitis HLS | Catapult HLS | Meaning |
|---|---|---|---|
| `float` | `float` | `float` | IEEE-754 single precision |
| `ap_fixed<16,5>` (raw Vitis spelling) | `ap_fixed<16,5>` (verbatim) | `ac_fixed<16,5,true,AC_TRN,AC_WRAP>` | W=16, I=5, **truncate + wrap** (the tool default) |
| `fixed<16,5>` (generic) | `ap_fixed<16, 5, AP_TRN, AP_WRAP>` | `ac_fixed<16,5,true,AC_TRN,AC_WRAP>` | same: **truncate + wrap** |
| `fixed<16,5,rnd,sat>` | `ap_fixed<16, 5, AP_RND, AP_SAT>` | `ac_fixed<16,5,true,AC_RND,AC_SAT>` | round to nearest (ties toward +inf) + saturate |

## Reading `ap_fixed<W,I>`

`W` is the **total** number of bits and `I` is the number of **integer bits including the sign bit**; the fractional bits are `F = W - I`.
`ap_fixed<16,5>`: 1 sign bit, 4 magnitude bits, 11 fractional bits, so the range is [-2^(I-1), 2^(I-1)) = **[-16, 16)** and the step is 2^-11.
`ap_fixed<32,10>`: range [-512, 512), step 2^-22. (`ac_fixed<W,I,true,...>` uses the same convention.)

Consequence for generated code: **any compile-time constant derived from a dimension must be computed at generation time, not cast to `data_t`.**
Under the default wrap mode `(data_t)64` is 0 in `<16,5>` (64 = `1000000`, only the low 5 bits survive) and `(data_t)16` is -16. The attention
scale `1/sqrt(head_dim)` is therefore emitted as a literal, and dimension divisors are plain integers (see CHANGELOG_R2.md, 2026-10-05).

Rules:
* **No mode is ever implied.** Without explicit modes the type is truncate (toward -infinity) + wrap (two's complement), the tool default.
  A raw `ap_*` string is passed through to Vitis unchanged, so writing `ap_fixed<16,5>` never changes arithmetic.
* Modes: quantization `trn | rnd`, overflow `wrap | sat` (case-insensitive). Anything else is an error.
* The sweep suites (GEMM / DNN / LLM, Tables 4-5) and the modularization suite use `ap_fixed<16,5>`: truncate + wrap.
  The full-model designs (ResNet, LLaMA) use `ap_fixed<16,5,AP_RND,AP_SAT>` for storage and `ap_fixed<32,10,AP_RND,AP_SAT>` for accumulation.
* **Vitis limitation:** `hls_math.h` defines its fixed-point overloads (`exp`, `sqrt`, `tanh`, `sin`, `cos`) only for the default modes. A design
  that calls them (softmax, sigmoid, tanh, layernorm, RoPE, ...) does not compile with `rnd`/`sat` in Vitis ("call to 'exp' is ambiguous").
  ForgeBench does not work around this; the compile error is Vitis'. Some operator templates also mix `double` constants with fixed-point values
  (e.g. GELU's `0.044715 * x`), which is ambiguous for non-default modes.
* The numeric emulation used by verification (`verification/fixedpoint.py`) is validated bit-for-bit against the real Vitis `ap_fixed` conversion
  (`verification/tests/test_fixedpoint.py`) for both default and explicit modes.

## Full-model designs (`scale_models/`): arithmetic in the design JSON

The full-model generator (`scale_models/gen_configs.run_hls_flow`) reads the same `data_type` field plus optional arithmetic fields.
Without them the emitted C++ is unchanged (the production revision-2 arithmetic: truncate + wrap storage, a persistent wide accumulator
`acc_t` of `max(64, 2W+16)` bits with `2F` fractional bits, ROM `exp` and integer-code division / square root).

| field | values | effect |
|---|---|---|
| `data_type` | any spelling above, e.g. `fixed<16,5,rnd,sat>` | storage type `data_t` |
| `acc_type`, `acc_rounding`, `acc_overflow` | e.g. `fixed<64,42>`, `trn\|rnd`, `wrap\|sat` | the accumulator `acc_t` (same field names as the per-operator fields of the operator library) |
| `math` | `table` (default) \| `hls` | `table`: ROM `exp` with interpolation, integer-code `div`/`sqrt` on `acc_t`; `hls`: Vitis `hls::exp` / `hls::sqrt` on `math_type`, division in `acc_t` |
| `math_type` | default `fixed<48,16>` | type of the `hls::` math calls (always truncate + wrap: Vitis `hls_math` only accepts the default modes) |
| `brams[*].data_type`, `drams[*].data_type` | `acc` \| `data` \| `int32` \| any type spelling | per-buffer type; overrides the role-derived default (accumulating buffers are `acc_t`) |

Vitis caveat for `math: hls`: `hls::exp` / `hls::sqrt` compile for `ap_fixed<64,42>` but return 0 (integer part too wide); `<48,16>` and
`<56,24>` are accurate, `<32,10>` is accurate except that `exp(-20)` underflows to 0. With non-default modes (`AP_RND`, `AP_SAT`) they do not compile at all.

The fixed-point reference of the full-model verifier (`scale_models/existing_model_verification`) follows the storage modes of the saved
`data_t` typedef: every `(data_t)` conversion rounds half up under `AP_RND` and saturates under `AP_SAT`; `acc_t` stays truncate + wrap.
Generated sources are hash-locked; register new ones with `existing_model_verification/register_sources.py` before running `verify.py`.
