"""Seeded impl (place & route) design selection.

LLM (R2): 1,000 of the 3,888 sweep designs, stratified by the head-dim unroll knob
(`hd_unroll` in {1,2,4,8}): 250 per value, uniform random within each stratum.

  python -m analysis.impl_sampler llm --seed 20261001 --out manifest/llm_impl_selection.csv

The gemm/conv selections of the July run (tiered by unroll factor) are not reproduced here;
see workstream B6.
"""
import argparse
import csv
import os
import random
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def select_llm(configs_dir, seed, per_stratum=250):
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(configs_dir) if f.endswith(".json"))
    strata = {}
    for s in stems:
        strata.setdefault(int(re.search(r"_UHD(\d+)_", s).group(1)), []).append(s)
    rng = random.Random(seed)
    rows = []
    for u in sorted(strata):
        pop = strata[u]
        for d in sorted(rng.sample(pop, per_stratum)):
            rows.append({"design": d, "stratum": f"hd_unroll={u}", "stratum_population": len(pop), "seed": seed})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["llm"])
    ap.add_argument("--configs-dir", default=os.path.join(REPO_ROOT, "llm", "auto_generated_configs"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--per-stratum", type=int, default=250)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = select_llm(a.configs_dir, a.seed, a.per_stratum)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["design", "stratum", "stratum_population", "seed"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} designs -> {a.out}")


if __name__ == "__main__":
    main()
