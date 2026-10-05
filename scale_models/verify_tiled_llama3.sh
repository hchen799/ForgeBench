#!/usr/bin/env bash
# Compatibility launcher. See verification/README.md for configurable runs.
set -euo pipefail
bash "$(dirname -- "${BASH_SOURCE[0]}")/verification/run_legacy_llama3.sh" "$@"
