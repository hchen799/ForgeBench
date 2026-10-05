#!/usr/bin/env bash
# Full 8B validation: the long test starts only if the short test passes.
set -euo pipefail

if [[ "${1:-}" == "--help" || $# -gt 1 ]]; then
    echo "Usage: bash verify_tiled_llama3.sh [new-output-directory]"
    echo "Runs full Llama 3 8B: 4-token prefill + 2 decode, then 2048 + 2."
    echo "Requires Vitis HLS, PyTorch, about 90 GiB host RAM and 40 GiB disk."
    exit 0
fi

llama_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$llama_script_dir"
llama_output_dir="${1:-verification_runs/llama3_full_suite}"
if [[ -e "$llama_output_dir" ]]; then
    echo "Refusing to overwrite an existing output directory: $llama_output_dir" >&2
    exit 1
fi

python generate_tiled_llama3.py --profile llama3_8b --max-ctx 8192 \
    --output-dir "$llama_output_dir/design"
python golden_tiled_llama3.py model \
    --project-dir "$llama_output_dir/design" --model-dir "$llama_output_dir/model" --seed 42
python golden_tiled_llama3.py run \
    --model-dir "$llama_output_dir/model" --run-dir "$llama_output_dir/short" \
    --prefill 4 --decode 2 --trace ops --threads 16
python golden_tiled_llama3.py run \
    --model-dir "$llama_output_dir/model" --run-dir "$llama_output_dir/long2048" \
    --prefill 2048 --decode 2 --trace layers --threads 16
