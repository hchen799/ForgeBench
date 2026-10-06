# Existing production-model verification

This directory verifies C++ accelerators emitted by ForgeBench's production path:

```text
auto_generate_json.py -> auto_generated_configs/*.json
  -> gen_configs.run_hls_flow(...) -> hls_files/<project>/top.cpp
  -> existing_model_verification/verify.py
```

The verifier snapshots and executes the saved generated `top.cpp`. It never
substitutes a dedicated ResNet generator, a Llama verification template, or a
PyTorch implementation for the device under test.

## Results

All 24 executed accelerator variants match their precision-specific fixed-point
PyTorch references at every compared integer code. FP64 acceptance uses
relative L2 error <= **5%** at every observed tensor. Maximum absolute error is
reported as a diagnostic; only an all-zero FP64 tensor uses the documented 0.1
absolute fallback because relative L2 is undefined.

| Model and implementation | `ap_fixed<16,5>` logits relative L2 | `ap_fixed<32,10>` logits relative L2 |
|---|---:|---:|
| ResNet-18 full, tiled | PASS (0.5113%) | PASS (0.000192%) |
| ResNet-34 full, tiled | PASS (0.7649%) | PASS (0.000219%) |
| ResNet-50 full, tiled | PASS (1.2412%) | PASS (0.000507%) |
| ResNet-101 full, tiled | PASS (1.9427%) | PASS (0.000630%) |
| ResNet-152 full, tiled | PASS (2.0462%) | PASS (0.000612%) |
| Llama-3-8B ctx2048, 4 prefill + 2 decode | FAIL (123%–125%) | PASS (0.000266%–0.000307%) |
| Llama-3-8B ctx8192, 4 prefill + 2 decode | FAIL (123%–125%) | PASS (0.000266%–0.000307%) |

For Llama, `ctx2048` and `ctx8192` specify K/V-cache capacity and generated
interface dimensions. Both campaigns execute the same four prefill tokens and
two decode calls. They validate both generated cache layouts and the populated
prefix, but they are not 2048-token or 8192-token prompt runs. Since the same
seed, tokens, weights, and first six cache positions are used, their numerical
results are identical.

The `123%–125%` and `0.000266%–0.000307%` entries above summarize three
separate final-logit comparisons; they are ranges, not measurement
uncertainty. The exact results are:

| Context | Accelerator call | `ap_fixed<16,5>` logits relative L2 | `ap_fixed<32,10>` logits relative L2 |
|---|---|---:|---:|
| ctx2048 | 4-token prefill | 123.293061% (FAIL) | 0.000265945% (PASS) |
| ctx2048 | decode 1 | 125.028486% (FAIL) | 0.000294283% (PASS) |
| ctx2048 | decode 2 | 123.603168% (FAIL) | 0.000306626% (PASS) |
| ctx8192 | 4-token prefill | 123.293061% (FAIL) | 0.000265945% (PASS) |
| ctx8192 | decode 1 | 125.028486% (FAIL) | 0.000294283% (PASS) |
| ctx8192 | decode 2 | 123.603168% (FAIL) | 0.000306626% (PASS) |

Each row compares the final vocabulary logits emitted by that accelerator
call. The verifier also checks every populated K/V-cache entry. A run passes
only when every observed tensor satisfies the 5% FP64 threshold and every C++
code matches fixed-point PyTorch exactly.

See [detailed results](results/DETAILED_RESULTS.md), the
[machine-readable index](results/validation_summary.json), and the
[preserved logs](logs/README.md). Every verification log ends with final-logit
shape, code mismatches, maximum code error, maximum absolute error, relative L2,
the 5% threshold, and both verdicts.

The [expanded regeneration audit](results/audits/expanded_regeneration.json)
records byte-identical reproduction of all five emitted files for 16 newly
generated depth/context/precision projects.

## Reproduce generated HLS

Requirements are Python 3.9+, PyTorch, NumPy, Vitis HLS 2024.1.x, and sufficient
disk. A full Llama parameter store uses about 16 GB at Q16 and 32 GB at Q32.
Run from `scale_models`:

```bash
python existing_model_verification/reproduce.py generate \
  --precision both \
  --output-dir /tmp/forgebench_reproduced_hls
```

Select any subset with `--models`; supported names include all full/tiled
ResNets and both Llama capacities. For example:

```bash
python existing_model_verification/reproduce.py generate \
  --precision both \
  --models resnet34-full resnet101-tiled resnet152-full \
           llama3-8b-ctx2048 llama3-8b-ctx8192 \
  --output-dir /tmp/forgebench_reproduced_hls
```

The direct production API is unchanged:

```python
import gen_configs

gen_configs.run_hls_flow(
    "auto_generated_configs/RESNET101_config_ap_fixed_32_10_.json",
    "/tmp/forgebench_reproduced_hls",
)
```

## Execute already-generated C++

```bash
python existing_model_verification/reproduce.py verify \
  --precision both \
  --models resnet18-full resnet18-tiled \
           resnet34-full resnet34-tiled \
           resnet50-full resnet50-tiled \
           resnet101-full resnet101-tiled \
           resnet152-full resnet152-tiled \
  --project-root hls_files \
  --output-dir /tmp/forgebench_resnet_results

python existing_model_verification/reproduce.py verify \
  --precision both \
  --models llama3-8b-ctx2048 llama3-8b-ctx8192 \
  --project-root hls_files \
  --prefill 4 --decode 2 --timeout 14400 \
  --output-dir /tmp/forgebench_llama_results
```

Use `--inspect-only` for fast source-hash and interface validation. Inspection
is reported as `INSPECTED_NOT_EXECUTED` and never as a numerical pass. Output
directories must be new.

`reevaluate.py RUN_DIR...` applies the current acceptance rule to stored metrics
without rerunning C++; it annotates the report with `accelerator_rerun: false`.
This was used to update the earlier 1% reports after the requested threshold
change. The raw accelerator outputs and measured errors are unchanged.

## Tests and scope

```bash
python -m pytest existing_model_verification/tests -q
python -m pytest verification/tests existing_model_verification/tests -q
```

These results establish functional Vitis HLS C-simulation agreement for
deterministic synthetic parameters. They do not establish pretrained accuracy,
full-model RTL cosimulation, FPGA execution, or accelerator performance.
