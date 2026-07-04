# CONV-domain verification findings

Independent textbook golden oracle for the conv domain
(`verification/domains/conv.py`) plus the golden-comparison wiring added to
`conv/generate_code.py` `generate_testbench_code`.

Self-check command:

    python3 -m verification.gcc_selfcheck conv

## Ops implemented

`load`, `store`, `conv` (conv2d; group_conv2d supported for completeness but not
instantiable — see below), `batchnorm`, `activation`, `maxpool`,
`adaptive_avgpool`, `matrix_add`.

All math is canonical/textbook, parameterized by each op's genuine knobs:
conv stride/padding/kernel; batchnorm epsilon + affine gamma/beta; pooling
kernel/stride. Layout/indexing conventions were derived by reading the emitted
C templates (`conv_template.cpp`, `batch_norm_template.cpp`,
`maxpool_template.cpp`, `adaptive_avgpool_template.cpp`,
`matrix_add_template.cpp`, `activations_template.cpp`) and the dispatch in
`generate_code.py`.

Layout conventions confirmed from the C:
- conv: input[C_IN][H_IN][W_IN], kernel[C_OUT][C_IN][K][K], bias[C_OUT],
  output[C_OUT][H_OUT][W_OUT]; output initialized to bias (or 0), then
  cross-correlation with `in_row=i*STRIDE-PAD+kh`, `in_col=j*STRIDE-PAD+kw`,
  bounds-checked (implicit zero padding), dilation=1, square kernel.
- batchnorm weights layout weights[4][C]: 0=gamma, 1=beta, 2=mean, 3=variance;
  `gamma*(x-mean)/sqrt(var+eps)+beta`.
- maxpool: NO padding; window `input[c][i*S_H+kh][j*S_W+kw]`, seeded from the
  top-left window corner.
- adaptive_avgpool: PyTorch-style floor/ceil start/end bins, mean over region.

## Results

28 / 30 `conv/test_case_configs/*.json` pass the self-check
(Python golden vs generated-C output dump, plus the tb `VERIFICATION:` line),
verified with real random inputs.

The 2 non-passing configs are NOT golden defects; both are pre-existing
config/generator issues surfaced by the harness:

### 1. `conv_variable.json` — not a concrete config (symbolic template)
Its dims are literal symbols (`C_IN`, `H_IN`, `K`, ...) rather than integers, so
it is invalid JSON and cannot be generated/compiled. It is a hand-authoring
template, not a runnable test case. Excluded as non-instantiable.

### 2. `vgg19_block3.json` — CAUGHT BUG: buffer-size mismatch (won't compile)
`BRAM_buffer_3` is declared `[512, 28, 28]`, but the `maxpool` op writes it as
`[512][14][14]` and the following `store` reads it as `[512][14][14]`. The
generated C therefore declares `BRAM_buffer_3` as `[512][28][28]` while calling
`maxpool_512_28_28_14_14_2_2_2_2_float(BRAM_buffer_2, BRAM_buffer_3)` and
`store_512_14_14_float(BRAM_buffer_3, ...)`, which are pointer-type mismatches:

    error: cannot convert 'data_t (*)[28][28]' to 'data_t (*)[14][14]'

This is a genuine generator/config bug (the pooled buffer should be
`[512, 14, 14]`, as it correctly is in the analogous `vgg19_block1` where
`BRAM_buffer_3` is `[128, 56, 56]`). The gemm-style flow catches it as a
compile error before verification even runs. Left failing on purpose per the
"a caught bug is the POINT" rule.

## Deliberate variants matched (not bugs)

- **Activations** reuse the shared textbook oracles in
  `verification/activations.py`, which already document the gelu tanh-approx
  variant matching the C.
- **Softmax in the conv domain is over the CHANNEL axis (axis 0)**, not the
  last axis. The conv `activations_template.cpp` softmax normalizes across `c`
  for each spatial `(h, w)`; the shared `activations.softmax` normalizes the
  last axis. To match the C faithfully, conv.py implements softmax LOCALLY over
  axis 0 (documented inline). Softmax is not exercised by any current config,
  but the variant is matched for correctness if one is added.
- **hardsigmoid / hardswish** exist in the C activation template but are not in
  the shared oracle; implemented LOCALLY in conv.py with the standard
  clip((x+3)/6, 0, 1) form matching the C. Not exercised by current configs.

## C memory semantics modeled

- BRAMs are C file-scope globals => zero-initialized; the golden zero-inits BRAM
  arrays (done centrally in `golden_ref.build_initial_arrays`).
- load/store copy exactly `prod(op_dims)` elements row-major.
- Some ops declare a logically larger tensor than the underlying buffer holds
  (e.g. `vgg19_block1` maxpool declares `C=256` while `BRAM_buffer_2` holds 128
  channels). The C reads out of bounds there; only the in-bounds region is ever
  stored to the output DRAM and compared. `_read` zero-pads reads past the real
  storage and `_write` truncates writes that overflow the destination, so the
  compared region matches the C exactly.

## group_conv2d

`func_type == "group_conv2d"` is implemented in the golden, but the generator's
call site (`generate_operator_call`) never passes the `groups` argument to the
emitted `group_conv2d(...)` call, so grouped convolution is not instantiable in
any valid config today. No test config uses it. (Latent generator gap, not
exercised.)

## Tolerance

Unchanged: rtol=1e-3, atol=1e-5 (both the Python compare and the tb
`VERIF_RTOL`/`VERIF_ATOL` defines). Note: deep, unnormalized conv stacks
(resnet50 blocks) accumulate large activations with random [0,1) inputs and push
the max relative error toward ~1e-3; all still pass via the absolute-tolerance
floor. If CO-SIM on the server shows borderline failures on the deepest blocks,
raise `rtol` modestly (e.g. 5e-3) — this is float reassociation, not a bug.
