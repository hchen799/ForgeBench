# Existing production-model verification

This directory verifies the C++ accelerators produced by ForgeBench's original
production path:

```text
auto_generate_json.py
    -> auto_generated_configs/*.json
    -> gen_configs.run_hls_flow(...)
    -> hls_files/<project>/top.cpp
    -> existing_model_verification/verify.py
```

The device under test is the saved `top.cpp` generated from JSON. The verifier
does not substitute `generate_tiled_resnet18.py`, `generate_tiled_llama3.py`, or
the templates under the older dedicated verification workflow.

## Recorded results

Every measured C++ model matches its precision-specific fixed-point PyTorch
reference at every observed integer code. FP64 acceptance requires every
observed tensor to have maximum absolute error at most 0.1 and relative L2
error at most 0.01.

| Production model | `ap_fixed<16,5>` | `ap_fixed<32,10>` |
|---|---:|---:|
| ResNet-18 full and tiled | PASS (logits relative L2 0.5113%) | PASS (0.000192%) |
| ResNet-50 full and tiled | FAIL (1.2412%) | PASS (0.000507%) |
| Llama-3-8B, 4-token prefill + 2 decode | FAIL (123%–125%) | PASS (0.000266%–0.000307%) |

The Q16.5 failures are retained as failures. Q32.10 is an additional generated
version and does not relabel the Q16.5 measurements. The FP64 graph remains the
same mathematical model across precisions. The fixed-point reference changes
its storage widths, intermediate widths, truncation, and wrap behavior to match
the corresponding C++ accelerator.

The Llama campaign covers all 32 layers, hidden size 4096, FFN size 14336,
vocabulary 128256, and 8,030,261,248 parameters. It executes four prefill
tokens and two one-token decode calls with a cache capacity of 2048. This is
not a 2048-token prompt run.

Detailed results are in [results/DETAILED_RESULTS.md](results/DETAILED_RESULTS.md).
The combined machine-readable index is
[results/validation_summary.json](results/validation_summary.json).

## Directory layout

```text
existing_model_verification/
├── verify.py                  direct verifier for a saved production project
├── reproduce.py               generation and verification command wrapper
├── references/                independent FP64 and fixed-point PyTorch models
├── tests/                     production-flow and arithmetic tests
├── results/
│   ├── q16_5/                 Q16.5 full-model summaries and failures
│   ├── q32_10/                Q32.10 full-model summaries
│   ├── audits/                regeneration, input, cache, and output audits
│   ├── synthesis/             numeric-helper synthesis reports
│   └── historical_v1/         failures found before the production repair
└── logs/
    ├── pytest/                preserved regression output
    └── csim/                  Vitis logs for all ten measured model variants
```

`references/resnet.py` and `references/llama.py` contain the mathematical FP64
graphs. `references/repaired.py` contains the Q16.5 fixed operations, and
`references/wide.py` contains the Q32.10 fixed operations. Exact source hashes
are bound to reference versions in `references/contracts.json`.

## Requirements

- Python 3.9 or newer
- PyTorch and NumPy
- Vitis HLS 2024.1.x on `PATH`, or `--vitis /path/to/vitis_hls`
- Enough disk for generated inputs and build artifacts

The full Llama input store is about 16 GB for Q16.5 and 32 GB for Q32.10.
Run outputs must use new directories; the verifier refuses to overwrite an
existing result.

Run the commands below from `scale_models`. `reproduce.py` itself also works
from another directory because it resolves the repository paths internally.

## 1. Inspect the checked-in models quickly

This validates registered source hashes, JSON/header interfaces, arithmetic
versions, and reference pairing without allocating 8B parameters or executing
C simulation:

```bash
python existing_model_verification/reproduce.py verify \
  --precision both \
  --project-root hls_files \
  --inspect-only \
  --output-dir /tmp/forgebench_existing_inspection
```

An inspection is labelled `INSPECTED_NOT_EXECUTED`; it is not a numerical PASS.

## 2. Recreate HLS from the original JSON workflow

Use a new output directory so the checked-in projects remain untouched:

```bash
python existing_model_verification/reproduce.py generate \
  --precision both \
  --output-dir /tmp/forgebench_reproduced_hls
```

This command runs `auto_generate_json.py` and then calls
`gen_configs.run_hls_flow` for ResNet-18 full/tiled, ResNet-50 full/tiled, and
the Llama prefill/decode pair. The generated projects contain
`resolved_config.json`, `top.cpp`, `top.h`, `tb_top.cpp`, and `run_hls.tcl`.

To reproduce one case:

```bash
python existing_model_verification/reproduce.py generate \
  --precision 32_10 \
  --models resnet50-tiled \
  --output-dir /tmp/forgebench_reproduced_hls
```

The direct original API remains valid:

```python
import gen_configs

gen_configs.run_hls_flow(
    "auto_generated_configs/RESNET50_config_ap_fixed_16_5_.json",
    "/tmp/forgebench_reproduced_hls",
)
```

## 3. Execute already generated production C++

Verify both ResNet depths and both implementations:

```bash
python existing_model_verification/reproduce.py verify \
  --precision both \
  --models resnet18-full resnet18-tiled resnet50-full resnet50-tiled \
  --project-root hls_files \
  --output-dir /tmp/forgebench_resnet_results
```

Verify full Llama-3-8B:

```bash
python existing_model_verification/reproduce.py verify \
  --precision both \
  --models llama3-8b \
  --project-root hls_files \
  --prefill 4 --decode 2 --timeout 14400 \
  --output-dir /tmp/forgebench_llama_results
```

To verify projects regenerated in step 2, change `--project-root` to
`/tmp/forgebench_reproduced_hls`. A verifier return code of 1 means the C++ run
completed but at least one numerical acceptance check failed. The wrapper
continues through expected Q16.5 failures and lists them at the end. Tool,
interface, incomplete-run, and integrity errors still stop reproduction.

## 4. Run tests

The production-model suite contains 42 tests:

```bash
python -m pytest existing_model_verification/tests -q
```

The recorded 87-test regression also included the configurable verification
framework that remains under `verification/`:

```bash
python -m pytest verification/tests existing_model_verification/tests -q
```

The preserved outputs are
[logs/pytest/production_tests.txt](logs/pytest/production_tests.txt) and
[logs/pytest/full_verification_87_tests.txt](logs/pytest/full_verification_87_tests.txt).
Vitis logs for each full-model execution are indexed under
[logs](logs/README.md).

## Scope of the claim

These are functional Vitis HLS C-simulation checks with deterministic synthetic
parameters. They establish agreement between the generated C++, an independent
fixed-point PyTorch implementation, and the stated FP64 error limits. They do
not establish pretrained-model accuracy, full-model RTL cosimulation, FPGA
execution, or accelerator performance. The helper synthesis reports cover only
the repaired numeric primitives, not an entire full-scale model.
