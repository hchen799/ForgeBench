"""ForgeBench golden-reference interpreter.

Reads a design JSON config plus the directory holding that design's input DRAM
`.txt` files, walks the ops in insertion order computing a NumPy float32 golden
reference, and writes `<DRAM>.golden.txt` for each output DRAM next to the inputs.

The golden is an INDEPENDENT textbook oracle (see plan): it does not read the
emitted C. It reads the SAME input `.txt` files the HLS run consumes, so both see
identical `random.seed(42)` inputs.

CLI:  python -m verification.golden_ref <config.json> <design_dir> [--domain gemm|conv|llm]
"""
import argparse
import importlib
import json
import os

import numpy as np

from verification.io import load_dram_txt, prod, write_golden

# Ops unique to each domain, used to auto-detect the domain from a config.
_DOMAIN_MARKERS = {
    "gemm": {"gemm", "vmm", "mmv", "dot_product"},
    "conv": {"conv", "batchnorm", "maxpool", "adaptive_avgpool"},
    "llm": {"matmul", "mha", "swa", "layernorm", "rmsnorm", "elementwise_mult", "dropout"},
}


def load_config(path):
    with open(path) as f:
        return json.load(f)


def detect_domain(config):
    func_names = {op["func_name"] for op in config["ops"].values()}
    for domain, markers in _DOMAIN_MARKERS.items():
        if func_names & markers:
            return domain
    raise ValueError(
        f"cannot infer domain from ops {sorted(func_names)}; pass --domain explicitly"
    )


def build_initial_arrays(config, design_dir):
    """Build the initial name -> float32 ndarray state.

    DRAMs are loaded from their `.txt` files; BRAMs are zero-initialized to match
    the C, where they are file-scope globals (zero-init) that ops read/write.
    """
    arrays = {}
    for bram in config.get("brams", []):
        arrays[bram["name"]] = np.zeros(bram["dims"], dtype=np.float32)
    for dram in config["drams"]:
        path = os.path.join(design_dir, f"{dram['name']}.txt")
        arrays[dram["name"]] = load_dram_txt(path, dram["dims"])
    return arrays


def compute_goldens(config, design_dir, domain=None):
    """Run the golden reference and return {output_name: ndarray}."""
    if domain is None:
        domain = detect_domain(config)
    module = importlib.import_module(f"verification.domains.{domain}")
    arrays = build_initial_arrays(config, design_dir)
    module.run(config["ops"], arrays)
    return {name: arrays[name] for name in config["output_dram_names"]}


def main():
    ap = argparse.ArgumentParser(description="ForgeBench golden reference generator")
    ap.add_argument("config", help="path to the design JSON config")
    ap.add_argument("design_dir", help="directory holding the design's input .txt files")
    ap.add_argument("--domain", choices=sorted(_DOMAIN_MARKERS), default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    goldens = compute_goldens(config, args.design_dir, args.domain)
    for name, arr in goldens.items():
        out_path = os.path.join(args.design_dir, f"{name}.golden.txt")
        write_golden(out_path, arr)
        print(f"wrote {out_path}  shape={tuple(arr.shape)}")


if __name__ == "__main__":
    main()
