#!/usr/bin/env python3
"""Create a clearly labelled comparator DEMO, not an HLS validation result."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verification.precision import FixedFormat


def create(destination, word_bits=16, integer_bits=5):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    p = FixedFormat(word_bits, integer_bits)
    rng = torch.Generator().manual_seed(123)
    x = p.quantize(torch.rand((2, 3), generator=rng, dtype=torch.float64) - .5)
    w = p.quantize(torch.rand((3, 4), generator=rng, dtype=torch.float64) - .5)
    floating = (x.double() / p.scale) @ (w.double() / p.scale)
    fixed = p.quantize(floating).numpy().astype(p.dtype)
    # Deliberately synthetic copy. Replace actual.npy with REAL accelerator
    # outputs before treating this as accelerator verification.
    np.save(destination / "actual.npy", fixed)
    np.save(destination / "fixed.npy", fixed)
    np.save(destination / "floating.npy", floating.numpy())
    manifest = dict(schema_version=1, model="DEMO ONLY: 2x3 by 3x4 matrix product",
                    notes="actual.npy is copied from golden, NOT produced by C/C++ or hardware",
                    precision=dict(word_bits=word_bits, integer_bits=integer_bits),
                    checkpoints=[dict(name="matmul.output", shape=[2, 4],
                                      actual=dict(path="actual.npy"), fixed=dict(path="fixed.npy"),
                                      floating=dict(path="floating.npy"))])
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination / "manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--word-bits", type=int, default=16)
    parser.add_argument("--integer-bits", type=int, default=5)
    args = parser.parse_args()
    print(create(args.output, args.word_bits, args.integer_bits))
