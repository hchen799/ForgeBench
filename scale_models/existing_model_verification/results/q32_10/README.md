# Q32.10 production verification

These are additional `ap_fixed<32,10>` versions generated through the same
JSON -> `gen_configs.run_hls_flow` production flow. The
[completed Q16.5 results](../DETAILED_RESULTS.md) remain separate, including their failures.
All measured Q32.10 variants pass both exact fixed-point comparison and the
unchanged FP64 error limits after actual Vitis HLS C simulation.

Data remains AP_TRN/AP_WRAP. Persistent accumulators use ap_fixed<80,36>
(44 fractional bits). Full-width data products remain exact before widening
into the accumulator. Fixed PyTorch uses four FP64 GEMMs/convolutions of
16-bit integer limbs with a proven exactness bound, plus Python integers for
wide intermediate products/division. FP64 uses ordinary real-valued PyTorch
operations. Neither reference reads DUT outputs.

The seed (42), underlying random draws, distributions, graph, token schedule,
and acceptance limits match the Q16.5 campaign. Each format quantizes those
same underlying draws to its declared precision, then shares its exact
quantized parameters between the DUT and both references. The
[complete input audit](../audits/cross_precision_inputs.json) checks all
8,067,850,624 input/parameter values across ResNet-18, ResNet-50, and Llama-3-8B:
every Q16.5 code equals its Q32.10 code shifted right by 11 bits. Fixtures are
not scaled to obtain PASS.

| Model | Fixed | FP64 | Logits max absolute error | Logits relative L2 |
|---|---|---|---:|---:|
| [ResNet-18 full](resnet18_full.json) | PASS | PASS | 7.10460e-6 | 1.91553e-6 |
| [ResNet-18 tiled](resnet18_tiled.json) | PASS | PASS | 7.10460e-6 | 1.91553e-6 |
| [ResNet-50 full](resnet50_full.json) | PASS | PASS | 1.78067e-5 | 5.07193e-6 |
| [ResNet-50 tiled](resnet50_tiled.json) | PASS | PASS | 1.78067e-5 | 5.07193e-6 |
| [Llama-3-8B prefill](llama3_8b.json) | PASS | PASS | 1.34250e-5 | 2.65945e-6 |
| Llama-3-8B decode 1 | PASS | PASS | 1.41819e-5 | 2.94283e-6 |
| Llama-3-8B decode 2 | PASS | PASS | 1.41496e-5 | 3.06626e-6 |

Relative L2 values in this table are ratios, not percentages. Every observed
tensor passed the unchanged max-absolute <= 0.1 and relative-L2 <= 0.01 limits.
All fixed comparisons contain zero code mismatches. Full/tiled outputs are
byte-identical at every checkpoint (12 for ResNet-18, 20 for ResNet-50), with
[output hashes recorded for both precisions](../audits/full_tiled_agreement.json).

The completed Llama campaign executes all 32 layers and 8,030,261,248
parameters (hidden 4096, FFN 14336, vocabulary 128256), with four prefill tokens
and two decode calls. All 1,752,576 compared logit/cache codes match exactly;
every populated KV entry and every emitted logit passes FP64 comparison.
The maximum relative L2 error across all observed tensors is 3.06626e-6
(0.000306626%). Source and reference-code integrity checks also pass.

[The cache continuity audit](../audits/cache_continuity.json) records byte-identical
transfers from actual C++ prefill output to decode 1 input, then from actual
decode 1 output to decode 2 input, in both precisions. Each PyTorch reference
retains its own cache state. Cache capacity 2048 does not mean that a
2048-token prompt was executed.

[The helper synthesis report](../synthesis/q32_10/numeric_top_csynth.rpt) confirms that the
80-bit division, square root, and exponential helpers synthesize. This check
does not establish synthesis or timing for an entire accelerator.

From `scale_models`:

```bash
python auto_generate_json.py --data-types 'ap_fixed<16,5>' 'ap_fixed<32,10>'
python - <<'GEN'
import gen_configs
for name in ('RESNET18', 'RESNET18_TILED', 'RESNET50', 'RESNET50_TILED',
             'LLAMA3_8B_PREFILL_ctx2048', 'LLAMA3_8B_DECODE_ctx2048'):
    gen_configs.run_hls_flow(
        f'auto_generated_configs/{name}_config_ap_fixed_32_10_.json', 'hls_files')
GEN
python existing_model_verification/verify.py --family llama3 \
  --project hls_files/LLAMA3_8B_PREFILL_ctx2048_config_ap_fixed_32_10_ \
  --decode-project hls_files/LLAMA3_8B_DECODE_ctx2048_config_ap_fixed_32_10_ \
  --prefill 4 --decode 2 --timeout 14400 \
  --output /tmp/forgebench_new_llama_q32_10
```

Use `--family resnet18` or `--family resnet50` and the corresponding saved
project for the other runs. The output directory must be new. These are
functional C-simulation checks with synthetic parameters, not pretrained
accuracy, full-model RTL cosimulation, or FPGA performance measurements.
