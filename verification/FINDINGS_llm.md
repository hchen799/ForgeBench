# LLM domain — golden-reference verification findings

Scope: the `llm` domain of the ForgeBench functional-correctness harness. The
golden reference (`verification/domains/llm.py`) is an independent textbook oracle
in float32; it is compared against the emitted C at `rtol=1e-3, atol=1e-5`.

Ops implemented: `load`, `store`, `matmul`, `mha`, `swa`, `layernorm`, `rmsnorm`,
`activation`, `dropout`, `matrix_add`, `elementwise_mult`.

Self-check result: `python3 -m verification.gcc_selfcheck llm` -> **5/7 configs
PASS**. The two failures (`gpt_transformer_p1`, `llama_transformer_p2`) are the
only configs containing a `dropout` op and are caused by a real bug in the emitted
C (see Caught bug #1). Everything else matches to within ~1.9e-6 absolute.

Tolerance: unchanged at the harness default `rtol=1e-3, atol=1e-5`. No tolerance
relaxation was needed; passing configs clear it by 3+ orders of magnitude. Torch
was NOT used — the golden is pure NumPy.

---

## Caught bug #1 — dropout runs its train-time formula at inference (non-identity)

Evidence — `llm/dropout_template.cpp`:

    unsigned int r = lcg_rand(&seed);
    data_t rand_val = (data_t)r / (data_t)2147483647;
    if (rand_val < dropout_prob) { output[i][j] = (data_t)0; }
    else { output[i][j] = input[i][j] / (1 - dropout_prob); }

At inference, dropout must be the identity (or, for inverted dropout, already
folded away). The emitted design instead applies a stochastic LCG mask and an
inverse-keep `1/(1-p)` rescale for *every* forward pass. Consequences:

1. It is not the identity — with `p=0.5` it zeros ~half the activations and
   doubles the rest, corrupting the transformer output.
2. It is not even a valid *inference* dropout: the mask is applied unconditionally
   (there is no train/eval flag), so results are not the expected deterministic
   pass-through.

The textbook golden implements dropout as the identity (`_op_dropout`), so this
divergence surfaces as a verification FAIL on the two transformer configs
(`gpt_transformer_p1`, `llama_transformer_p2`, both `p=0.5`). This is the harness
working as intended: a genuine correctness bug is caught rather than hidden.

Note: with the *old* all-zero DRAM inputs (see Fix #1) `0/(1-p) == 0`, so the bug
was completely masked — the design "passed" only because it computed nothing.

Recommended fix (design side, not this harness): at inference emit an identity
copy, or make dropout a compile-time no-op.

---

## Fix #1 (applied) — DRAM inputs were hard-coded to all zeros

Evidence — `llm/generate_code.py::generate_dram_txt_files` previously wrote
`str(0)` for every element (the `random.random()` line was commented out), unlike
the `gemm`/`conv` domains which use `random.random()`.

With all-zero inputs the entire network collapses to zeros (Q=K=V=0, softmax is
uniform, norms output their bias, dropout of 0 is 0, etc.), so the golden check is
trivially satisfied and cannot catch any bug — it defeats the purpose of the
correctness harness (reviewer concern C1). Changed to `random.random()` to match
the other domains and give the oracle real signal. This change is what exposed
Caught bug #1.

---

## Deliberate variants matched (not bugs)

These are widely-used variants where the golden intentionally mirrors the C:

- **gelu — tanh approximation.** `activation_template.cpp` uses
  `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`, not the exact erf gelu. The
  shared `verification/activations.py` already implements this variant.
- **matmul layout.** `matmul_template.cpp` computes
  `output[i][j] += input[i][k]*weights[j][k]`, i.e. `out = input @ weights.T` with
  `weights` stored `[DIM_OUT][DIM_IN]`. The golden transposes accordingly.
- **Attention scale.** Both attention templates use `1/sqrt(head_dim)` — standard.
- **Softmax stability.** Both attention templates already subtract the row/window
  max before `exp` (unlike the gemm domain's earlier unstable softmax), so there
  is no NaN-overflow bug here. The golden also uses a max-subtracting softmax;
  softmax is shift-invariant so the two agree within float tolerance.
- **RoPE convention.** `theta = 10000^(-d/head_dim)` for `d = 0,2,4,...`, angle =
  `seq*theta`, rotating pairs `(h*head_dim+d, h*head_dim+d+1)`. The golden's
  `_apply_rope` replicates this exactly, and only rotates Q and K (not V), as the C
  does. Exercised via `llama_transformer_p2` (`use_rope=True`).
- **MHA grouping / partial head coverage.** The C computes heads as
  `g*heads_per_group + h` with `heads_per_group = num_heads/groups` (integer
  division). When `groups` does not divide `num_heads`, the trailing heads are
  never written and keep their zero-init value. The golden mirrors this
  (`heads_per_group = NH // groups`) rather than assuming all heads are covered.
- **No causal mask.** Neither attention template applies a causal mask (mha is
  full bidirectional; swa is a symmetric `[i-w, i+w]` window). The golden matches.

---

## Notes

- `swa` (sliding-window attention) is implemented per the vocabulary but is not
  exercised by any current `test_case_configs`.
- Shared shim (`verification/shim/hls_math.h`) already provides every math symbol
  the llm C needs (`exp`, `tanh`, `sqrt`, `pow`, `powf`, `sin`, `cos`); no shim
  change was required.
- In the gcc self-check, the tb prints `VERIFICATION: SKIP` on the very first run
  because the Python comparison writes the `.golden.txt` files *after* the binary
  runs; the Python cross-check (golden vs the tb's `_output.txt` dump) is the
  authoritative check there. On the server flow the golden files are generated
  before CSIM, so the tb's own `VERIFICATION:` verdict is meaningful.
