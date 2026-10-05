#!/usr/bin/env python3
"""Build the sweep design manifests: manifest/designs/{gemm,conv,llm}.csv.

One row per design:

    design_id | <sweep parameters> | generated | csynth | impl | config_path | csynth_report | impl_report |
    impl_power_report | over_capacity | fail_reason

* design_id            gemm_0001 ... (frozen by manifest/internal/id_map.csv; see manifest/_common.py)
* <sweep parameters>   one column per parameter that varies in the sweep (single-valued parameters are listed in
                       designs/fixed_parameters.csv)
* generated/csynth/impl  YES | NO | FAIL.  NO = not run (impl: not in the implemented subset).
* *_report / config_path  paths relative to the extracted release bundle: configs/<domain>/<id>.json and
                       reports/<domain>/<id>/{csynth.xml,export_impl.xml,power_routed.rpt}; filled only when the stage is YES
* over_capacity        YES if the csynth estimate exceeds any ZCU102 limit (BRAM_18K 1824, DSP 2520, LUT 274,080, FF 548,160)
* fail_reason          "impl: <reason>" for FAIL rows, else empty

Parameters and names come from the generators (`{domain}/auto_generate_json.py`), results from analysis/results_*/metrics_*.csv
(keyed by design id; run `python manifest/rekey_results.py` after collecting new runs) and the lean report archives.

    python manifest/build_manifest.py [gemm conv llm] [--check-disk] [--refreeze-ids]
"""
import argparse
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (DEVICE, DOMAINS, ID_MAP, JULY_IMPL_VALID, REPO, check_id_map, design_table, impl_failures,
                     index_sources, load_generator, locate, read_metrics, write_id_map)

OUT_DIR = os.path.join(REPO, "manifest", "designs")


def yn(flag):
    return "YES" if flag else "NO"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domains", nargs="*", default=DOMAINS)
    ap.add_argument("--check-disk", action="store_true", help="verify generated config text against */auto_generated_configs")
    ap.add_argument("--refreeze-ids", action="store_true", help="rewrite manifest/internal/id_map.csv (deliberate id change)")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    tables = {d: design_table(d) for d in a.domains}
    if a.refreeze_ids or not os.path.isfile(ID_MAP):
        write_id_map({d: design_table(d) for d in DOMAINS})
        print(f"wrote {os.path.relpath(ID_MAP, REPO)}")
    for d, t in tables.items():
        check_id_map(d, t)

    print("indexing report archives ...", flush=True)
    index = index_sources()
    failures = impl_failures()
    fixed_rows = []

    for domain in a.domains:
        gen, table = load_generator(domain), tables[domain]
        csynth = read_metrics("csynth", domain)
        impl = read_metrics("impl", domain)
        ids = {t["design_id"] for t in table}
        for name, rows in (("csynth", csynth), ("impl", impl)):
            stray = [k for k in rows if k not in ids]
            if stray:
                sys.exit(f"{domain}: results_{name}/metrics_{domain}.csv has rows not keyed by a design id (e.g. {stray[:2]}); "
                         "run python manifest/rekey_results.py")

        # parameter columns = parameters that vary; constants are documented separately
        names = list(table[0]["params"])
        values = {n: sorted({str(t["params"][n]) for t in table}) for n in names}
        var = [n for n in names if len(values[n]) > 1]
        fixed_rows += [[domain, n, values[n][0]] for n in names if len(values[n]) == 1]

        cols = (["design_id"] + var + ["generated", "csynth", "impl", "config_path", "csynth_report", "impl_report",
                                       "impl_power_report", "over_capacity", "fail_reason"])
        rows = []
        for t in table:
            did, p = t["design_id"], t["params"]
            if a.check_disk:
                with open(os.path.join(REPO, domain, "auto_generated_configs", t["stem"] + ".json"), "rb") as f:
                    assert f.read() == gen.build_config_text(p).encode(), f"disk config differs: {t['stem']}"
            loc = locate(index, domain, t)
            m, im = csynth.get(did), impl.get(did)
            gen_ok = "source" in loc
            cs_ok = m is not None
            if cs_ok and "csynth" not in loc:
                sys.exit(f"{did}: has csynth metrics but no csynth report in the archives")
            fail = failures.get((domain, t["stem"])) or failures.get((domain, t["key"]))
            im_ok = im is not None and ("impl" in loc)
            if im is not None and "impl" not in loc:
                sys.exit(f"{did}: has impl metrics but no impl report in the archives")
            impl_state = "YES" if im_ok else ("FAIL" if fail else "NO")
            over = ""
            if cs_ok:
                over = yn(any(float(m[k]) > DEVICE[k] for k in DEVICE))
            base = f"reports/{domain}/{did}"
            r = {"design_id": did, **{n: ("" if p[n] is None else p[n]) for n in var},
                 "generated": yn(gen_ok), "csynth": yn(cs_ok), "impl": impl_state,
                 "config_path": f"configs/{domain}/{did}.json",
                 "csynth_report": f"{base}/csynth.xml" if cs_ok else "",
                 "impl_report": f"{base}/export_impl.xml" if im_ok else "",
                 "impl_power_report": f"{base}/power_routed.rpt" if im_ok and "power" in loc else "",
                 "over_capacity": over,
                 "fail_reason": f"impl: {fail}" if impl_state == "FAIL" else ""}
            rows.append(r)
        out = os.path.join(OUT_DIR, f"{domain}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        c = lambda k, v: sum(1 for r in rows if r[k] == v)
        print(f"{domain}: {len(rows)} designs | generated {c('generated', 'YES')} | csynth YES {c('csynth', 'YES')} NO {c('csynth', 'NO')} "
              f"FAIL {c('csynth', 'FAIL')} | impl YES {c('impl', 'YES')} FAIL {c('impl', 'FAIL')} NO {c('impl', 'NO')} | "
              f"over_capacity {c('over_capacity', 'YES')} -> {os.path.relpath(out, REPO)}")

    if set(a.domains) == set(DOMAINS):
        with open(os.path.join(OUT_DIR, "fixed_parameters.csv"), "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["domain", "parameter", "value"])
            w.writerows(fixed_rows)


if __name__ == "__main__":
    main()
