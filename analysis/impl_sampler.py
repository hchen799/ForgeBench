"""Seeded impl (place & route) design selection.

LLM (R2): 1,000 of the 3,888 sweep designs, stratified by the head-dim unroll knob
(`hd_unroll` in {1,2,4,8}): 250 per value, uniform random within each stratum.

  python -m analysis.impl_sampler llm --seed 20261001 --out manifest/llm_impl_selection.csv

GEMM (R2): 1,000 of the 3,072 sweep designs. The unroll factors (unroll_M, unroll_K, unroll_N) in {1, 8}^3
give 8 groups of 384 designs. Every design with no unrolling (1/1/1) is taken (384), plus 88 chosen at random
from each of the 7 unrolled groups (616): 384 + 7 x 88 = 1,000.

  python -m analysis.impl_sampler gemm --seed 20261002 --out manifest/gemm_impl_selection.csv

The July gemm/conv selections (tiered by unroll factor, sampler not preserved) are not reproduced here.
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


def select_gemm(configs_dir, seed, per_group=88):
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(configs_dir) if f.endswith(".json"))
    groups = {}
    for s in stems:
        m = re.search(r"_UM(\d+)_UK(\d+)_UN(\d+)_", s)
        groups.setdefault(tuple(int(x) for x in m.groups()), []).append(s)
    rng = random.Random(seed)
    rows = []
    for key in sorted(groups):
        pop = groups[key]
        chosen = pop if key == (1, 1, 1) else rng.sample(pop, per_group)    # no unrolling: take all
        for d in sorted(chosen):
            rows.append({"design": d, "unroll_M": key[0], "unroll_K": key[1], "unroll_N": key[2], "seed": seed})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["llm", "gemm"])
    ap.add_argument("--configs-dir")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--per-stratum", type=int, default=250, help="llm: designs per hd_unroll value")
    ap.add_argument("--per-group", type=int, default=88, help="gemm: designs per unrolled group")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cdir = a.configs_dir or os.path.join(REPO_ROOT, a.domain, "auto_generated_configs")
    if a.domain == "llm":
        rows = select_llm(cdir, a.seed, a.per_stratum)
        cols = ["design", "stratum", "stratum_population", "seed"]
    else:
        rows = select_gemm(cdir, a.seed, a.per_group)
        cols = ["design", "unroll_M", "unroll_K", "unroll_N", "seed"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} designs -> {a.out}")


if __name__ == "__main__":
    main()
