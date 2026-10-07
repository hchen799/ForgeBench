#!/usr/bin/env python3
"""Register generated HLS sources in references/contracts.json (as a revision of their variant).

The verifier refuses sources it has not seen (hash-locked contracts). After generating a project with
gen_configs.run_hls_flow -- e.g. a design JSON with different storage modes -- register it with the reference
arithmetic version it implements (2: production wide accumulators), then run verify.py; the verify run is what
validates the registration (fixed-point codes must match bit-exactly).

    python existing_model_verification/register_sources.py --family resnet18 --reference-version 2 PROJECT_DIR [...]
"""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from existing_model_verification.references import project as P  # noqa: E402

CONTRACTS = Path(P.__file__).with_name("contracts.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True)
    ap.add_argument("--reference-version", type=int, default=2)
    ap.add_argument("projects", nargs="+", type=Path)
    a = ap.parse_args(argv)
    contracts = json.loads(CONTRACTS.read_text())
    for proj in a.projects:
        try:
            P.Project(proj, a.family)
            print("already registered:", proj)
            continue
        except ValueError as e:
            m = re.match(r"unregistered HLS source version: (\S+);", str(e))
            if not m:
                raise
            variant = m[1]
        sources = {n: P.sha(proj / n) for n in ("top.cpp", "top.h")}
        entry = contracts["variants"].setdefault(variant, {"revisions": []})
        entry.setdefault("revisions", []).append({"reference_version": a.reference_version, "sources": sources})
        print(f"registered {proj} as {variant} (reference v{a.reference_version})")
    CONTRACTS.write_text(json.dumps(contracts, indent=2) + "\n")


if __name__ == "__main__":
    main()
