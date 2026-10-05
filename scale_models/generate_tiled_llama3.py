"""Generate full Llama 3 8B prefill/decode HLS kernels and C-simulation support."""
import argparse
from pathlib import Path

from llama3_verification.config import make_config
from llama3_verification.codegen import emit_project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("llama3_8b", "tiny"), default="llama3_8b")
    parser.add_argument("--max-ctx", type=int, help="KV capacity (default: 8192 for full 8B, 32 for tiny)")
    parser.add_argument("--prefill-tile", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    max_ctx = args.max_ctx
    if max_ctx is None and args.profile == "llama3_8b":
        max_ctx = 8192
    config = make_config(args.profile, max_ctx=max_ctx, prefill_tile=args.prefill_tile)
    destination = args.output_dir or Path(__file__).parent / "hls_files" / f"{args.profile}_tiled_ctx{config.max_ctx}"
    emit_project(destination, config)
    print(f"Generated {config.profile}: {destination}")


if __name__ == "__main__":
    main()
