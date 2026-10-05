# Validation record

Host validation on 2026-10-02 used PyTorch 2.2.0, NumPy 1.24.4, and Vitis HLS
2024.1.2. The parameter seed was 42 and the token seed was 43. These are
synthetic random parameters, not a pretrained checkpoint.

On 2026-10-03, the complete long-case comparison was rerun from the saved
artifacts with source, parameter, input, and reference integrity checks; it
passed again. The combined 32-test regression suite also passed again.

## Completed functional checks

| Configuration | Prefill + decode | Vitis checkpoint comparisons | Integer mismatches | Saturations |
| --- | ---: | ---: | ---: | ---: |
| Full 8B, 8192-slot cache | 4 + 2 | 1,737 | 0 | 0 |
| Full 8B, 8192-slot cache | 2048 + 2 | 12,740 | 0 | 8 |
| Tiny, 32-slot cache | 30 + 2 | 390 | 0 | 0 |
| Tiny, 8192-slot cache | 8190 + 2 | 2,064 | 0 | 0 |

The full-size short case compared 19,889,664 raw integer codes, including all
769,536 logits. Against mathematical FP64, logit max absolute error was
0.015099260658835378, MAE 0.0022845966209983494, and RMSE
0.0028703466575113606.

The full 8B **2048 + 2** case passed: all **674,368,000 integer codes** across
12,740 checkpoints matched exactly, including every new KV-cache entry,
layer output, final normalization, and all **262,924,800 logits**. Both decode
calls passed using the complete prefill cache. Both PyTorch references completed
in 1291.95 seconds; the Vitis C-simulation invocation took 4126.96 seconds.
These are host runtime measurements, not FPGA latency estimates.

Across all long-case logits, C++ fixed point versus mathematical FP64 has
max absolute error 0.06996331252043593, MAE
0.0031098447654074944, and RMSE 0.003909475912172491. The relative L2 error
is 0.003896285403543128 (about 0.390%). Eight SwiGLU intermediate values
saturated; no other operations recorded saturation. Thus the implementation
is bit-exact, but the Q5.11 computation is not saturation-free for this long
case. Mathematical error is reported diagnostically, not used to claim
pretrained language-model accuracy. Full 8B execution at 8192 tokens has not
been tested; the 8192-position boundary result above uses the tiny profile.

Artifacts relative to `scale_models`:

- `verification_runs/llama3_8b/short/comparison.json`
- `verification_runs/llama3_8b/long2048/comparison.json`
- `verification_runs/llama3_tiny_v2/boundary/comparison.json`
- `verification_runs/llama3_tiny_ctx8192/boundary/comparison.json`

The first six token positions in the long and short runs match bit-for-bit
in both fixed-point PyTorch and actual C++ outputs. All 98 shared C++
checkpoints (including layer outputs and cache updates) agree. FP64 logits
differ by at most 1.14e-14. This checks that the short run's two decode
positions agree with the same positions processed inside a larger prefill chunk.

The 17 Llama regression tests and 15 existing ResNet regression tests passed
together (32 total). Directed C++ tests additionally exercise intentional
saturation, extreme int16 values, signed ties, integer square roots, cache
bounds, and non-multiple tile dimensions.

## Full-size HLS synthesis

Both generated full-size tops completed `csynth_design` and emitted Verilog
and VHDL using the development default ZU9EG part and a 10 ns target clock.

| Top | BRAM_18K | DSP | FF | LUT |
| --- | ---: | ---: | ---: | ---: |
| `llama_prefill` | 201 | 40 | 18,804 | 31,975 |
| `llama_decode` | 199 | 29 | 15,123 | 26,202 |

These are HLS estimates, not placed-and-routed utilization. Both reports give
7.300 ns estimated clock period with 2.70 ns uncertainty. Variable loop bounds
leave whole-kernel latency unspecified. HLS reports that not all loop scheduling
constraints are satisfied, and reports missed AXI burst opportunities. This is
a functional baseline, not a claim of optimized throughput or timing closure.
Generated parameter AXI addresses are 64-bit, sufficient for the packed
approximately 15 GiB model. Board memory capacity still needs separate design.

Reports:

- `verification_runs/llama3_8b/design/project_prefill/solution1/syn/report/llama_prefill_csynth.rpt`
- `verification_runs/llama3_8b/design/project_decode/solution1/syn/report/llama_decode_csynth.rpt`

No RTL cosimulation or FPGA-board execution has been performed. An auxiliary
tiny-profile synthesis was stopped during expensive automatic loop scheduling;
the completed synthesis results above are for the requested full-size model.
