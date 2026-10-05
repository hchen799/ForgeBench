#!/usr/bin/env python3
"""Run this operator's functional verification. Settings: verif_config.json next to this file (defaults: ../../paper_verif.json).

    python functional_verification.py --out results/                      # as configured
    python functional_verification.py --out results/ --dtypes float "fixed<16,5>" --n 20 --sim csim
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from verification.functional_verification import main  # noqa: E402

if __name__ == "__main__":
    main(default_config=os.path.join(HERE, "verif_config.json"), default_operators=["mha"])
