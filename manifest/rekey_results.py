#!/usr/bin/env python3
"""Rewrite the `design` column of analysis/results_{csynth,impl}/metrics_<domain>.csv to public design ids.

Run outputs (analysis.collect) are keyed by the generator's long design names (or R1 names for July rows).
This maps them to `<domain>_<NNNN>` using the same table as the manifest. Idempotent (rows already keyed by id are kept).
Rows whose name matches no design in the current sweep are dropped and counted (e.g. R1 designs that no longer exist).

    python manifest/rekey_results.py [gemm conv llm]
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DOMAINS, REPO, design_table, check_id_map


def main():
    domains = sys.argv[1:] or DOMAINS
    for domain in domains:
        table = design_table(domain)
        check_id_map(domain, table)
        ids = {t["design_id"] for t in table}
        to_id = {t["stem"]: t["design_id"] for t in table}
        to_id.update({t["legacy"]: t["design_id"] for t in table if t["legacy"]})
        for kind in ("csynth", "impl"):
            path = os.path.join(REPO, "analysis", f"results_{kind}", f"metrics_{domain}.csv")
            if not os.path.isfile(path):
                continue
            with open(path, newline="") as f:
                rd = csv.DictReader(f)
                cols, rows = rd.fieldnames, list(rd)
            out, dropped, seen = [], 0, set()
            for r in rows:
                d = r["design"] if r["design"] in ids else to_id.get(r["design"])
                if d is None or d in seen:
                    dropped += 1
                    continue
                seen.add(d)
                r["design"] = d
                out.append(r)
            out.sort(key=lambda r: r["design"])
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
                w.writeheader()
                w.writerows(out)
            print(f"{kind}/{domain}: {len(out)} rows keyed by id ({dropped} dropped: not in the current sweep)")


if __name__ == "__main__":
    main()
