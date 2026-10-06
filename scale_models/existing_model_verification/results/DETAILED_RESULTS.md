# Production generator repair: revision 2

Both requested campaigns are complete. All measured variants match their
fixed-point references exactly. With preserved Q16.5 storage, ResNet-18 passes
FP64 comparison, while ResNet-50 and Llama-3-8B fail. The additional
[Q32.10 variants](q32_10/README.md) all pass FP64 comparison, including full
32-layer Llama prefill and two decode calls. The
[machine-readable summary](validation_summary.json) retains both precisions
and every failure instead of replacing the Q16.5 results.

The workflow remains:

```text
auto_generate_json.py -> auto_generated_configs/*.json
    -> gen_configs.run_hls_flow(...) -> hls_files/<config_name>/
    -> existing_model_verification/verify.py
```

No dedicated ResNet generator or verification Llama kernel template supplies
the DUT. The direct verifier snapshots and executes the saved production C++.
[Regeneration evidence](audits/regeneration.json) records byte-identical regeneration
of all five emitted files for sixteen saved production projects: ten repaired
Q16.5 projects and six additional Q32.10 projects. The fourteen local Q16.5
ResNet and Llama-3-8B JSON configurations are updated, and six Q32.10 JSONs
provide the additional measured variants. The dual-precision CLI was also
exercised in an isolated directory and produced 32 JSON configurations.
`auto_generated_configs/` is git-ignored by the repository; each regenerated
project also saves its JSON as `resolved_config.json` for reproducibility.
Generation of a design does not establish its numerical validation.

## Repairs in the production flow

- Full-buffer ResNet loads every weight and BN parameter. Bottleneck conv1
  uses an appropriately sized buffer before conv2 downsamples. Tiled bottlenecks
  also declare the missing stride-1, 1x1 patch buffer and pre-downsample storage.
- Conv/linear reductions persist in wide buffers across input tiles and
  quantize once at the layer boundary. Pooling uses a wide sum and divisor.
- Llama token IDs, lengths, and positions are `int32_t`. RMSNorm sums and
  reciprocal factors, attention scores, softmax sums, and contexts stay wide.
- `o_proj` writes to the already available `DRAM_norm1`, after Q/K/V have
  consumed it. This avoids overwriting attention inputs needed by later tiles.
- Causal softmax excludes future keys. SiLU uses the stable negative-exponent
  form, avoiding overflow in `exp(-x)` for negative inputs.
- Generation rejects missing/undersized array arguments and unread ResNet
  parameter ports. Old invalid JSONs must be regenerated from the repaired
  builder. `resolved_config.json` records the resolved storage types.

The original data/parameter storage remains **ap_fixed<16,5>, AP_TRN/AP_WRAP**.
The requested [additional Q32.10 campaign](q32_10/README.md) uses separate files.
The repaired Q16.5 designs use **ap_fixed<64,42>, AP_TRN/AP_WRAP** accumulators (22 fractional bits),
floor division, nearest integer square root, and a 1/256 exponential ROM with
linear interpolation on [-16,0]. BN/RMSNorm epsilon is nonzero at accumulator
precision. The fixed PyTorch operators model these choices explicitly; the
FP64 graph uses real-valued PyTorch operations. Fixed RoPE shares only the
vendor transcendental coefficients, as disclosed in each Llama report.

The entry points and filenames are preserved. This is **not an unchanged
hardware ABI**: Llama control ports become integer ports, and tiled bottleneck
ResNets acquire three pre-downsample scratch ports. Hosts must use regenerated
headers/JSONs. Changed arithmetic also requires fresh synthesis measurements.

## Full-model measurements: Q16.5

Seed 42, identical quantized synthetic parameters and inputs, Vitis HLS
2024.1.2 C simulation. Acceptance remains max absolute error <= 0.1 and
relative L2 error <= 1% at **every** observed tensor; fixed comparison requires
zero differing integer codes. Fixtures and limits were not tuned for a PASS.

| Production model | Observed codes | Fixed comparison | FP64 comparison | Logits max absolute | Logits relative L2 |
|---|---:|---|---|---:|---:|
| [ResNet-18 full](q16_5/resnet18_full.json) | 1,757,672 | PASS | PASS | 0.00698534 | 0.511272% |
| [ResNet-18 tiled](q16_5/resnet18_tiled.json) | 1,757,672 | PASS | PASS | 0.00698534 | 0.511272% |
| [ResNet-50 full](q16_5/resnet50_full.json) | 6,525,928 | PASS | FAIL | 0.01602118 | 1.241235% |
| [ResNet-50 tiled](q16_5/resnet50_tiled.json) | 6,525,928 | PASS | FAIL | 0.01602118 | 1.241235% |

[Full/tiled outputs are byte-identical](audits/full_tiled_agreement.json) at all
12 ResNet-18 and 20 ResNet-50 checkpoints. Five ResNet-50 checkpoints exceed
the 1% relative-L2 limit.
Exact implementation agreement does **not** make this a mathematical PASS;
the retained 16-bit arithmetic still fails that selected acceptance criterion.
The ResNet-50 reference follows the production bottleneck graph, including
its projection without an extra BN; this is not a pretrained torchvision test.

The full Llama-3-8B campaign runs all 32 layers, hidden size 4096, FFN 14336,
vocabulary 128256, and cache capacity 2048 with four prefill tokens and two
decode calls. [The completed 16/5 report](q16_5/llama3_8b.json) records **PASS** for all fixed
comparisons and **FAIL** for all FP64 comparisons. All 1,752,576 compared codes
match the independent fixed reference. Prefill logits have max absolute error
7.759451 and relative L2 error 123.293%; the two decode logit errors are
125.028% and 123.603% relative L2. These are real DUT results, not estimates
from the references. Cache capacity 2048 does not mean a 2048-token prompt
was simulated.

[The separate FP64 range audit](audits/llama_fp64_range_audit.json) finds the first
unrepresentable residual at zero-based layer 5: a minimum of -19.1649 falls
below Q16.5's lower limit -16. Subsequent layers contain many values outside
the storage range. This explains why fixed agreement cannot establish FP64
closeness for the preserved AP_WRAP format. The requested Q32.10 campaign
uses separate generated projects, reference arithmetic, inputs, and reports;
it does not replace or relabel these 16/5 failures.

[Cache continuity](audits/cache_continuity.json) is checked using hashes of the actual
C++ cache files passed between prefill and both decode calls, in both
precisions. Each mathematical/fixed reference maintains its own cache.

## Reproduce

From `scale_models`, run `python auto_generate_json.py` to recreate the
git-ignored JSON directory on a fresh checkout. The existing generation
command still works:

```python
import gen_configs
gen_configs.run_hls_flow(
    "auto_generated_configs/RESNET50_config_ap_fixed_16_5_.json", "hls_files"
)
```

Use the same call with `LLAMA3_8B_PREFILL_ctx2048_config_ap_fixed_16_5_.json`
or `LLAMA3_8B_DECODE_ctx2048_config_ap_fixed_16_5_.json`. This emits the project;
it does not itself invoke Vitis. For direct execution and both comparisons:

```bash
python existing_model_verification/verify.py --family resnet50 \
  --project hls_files/RESNET50_config_ap_fixed_16_5_ \
  --output /tmp/forgebench_new_resnet50

python existing_model_verification/verify.py --family llama3 \
  --project hls_files/LLAMA3_8B_PREFILL_ctx2048_config_ap_fixed_16_5_ \
  --decode-project hls_files/LLAMA3_8B_DECODE_ctx2048_config_ap_fixed_16_5_ \
  --prefill 4 --decode 2 --timeout 14400 \
  --output /tmp/forgebench_new_llama3
```

Each output directory must be new. Preserved execution logs are under
`../logs/csim/`; summaries here preserve source/reference hashes, input
manifests, all checkpoint metrics, and simulation provenance.
Historical failures remain in [the v1 record](historical_v1/validation_summary.json).

## Directed coverage and limits

The combined verification regression passed **87 tests** with both precisions
([log and test-source hashes](../logs/pytest/full_verification_87_tests.json)).
Run it from `scale_models` with
`python -m pytest verification/tests existing_model_verification/tests -q`.

The reduced production-graph integration test uses 17 prefill tokens and two
decode calls with token IDs above 15. It checks every logit and populated KV
entry, including prefill/decode cache continuity. Operator tests exercise
4096-wide RMSNorm, partial reduction tiles, strict causality, BN, and stable
SwiGLU with actual Vitis types and independent integer-code references.

[The helper synthesis report](synthesis/q16_5/numeric_top_csynth.rpt) confirms the new division,
sqrt, and exponential primitives synthesize. It is **not** full-model synthesis,
RTL cosimulation, FPGA execution, or a performance/accuracy benchmark. The
regenerated ResNet-34/101/152 projects and context-8192 JSONs are not covered by
the full-model numerical claims above.
