#!/usr/bin/env python3
"""Assemble the release's configs/ and reports/ trees exactly as the manifest paths describe, then tar them.

    configs/<domain>/<design_id>.json
    reports/<domain>/<design_id>/{csynth.xml, export_impl.xml, power_routed.rpt}       (sweeps)
    reports/modular/<domain>/<name>/{csynth.xml, <function>_csynth.xml, csynth.rpt}    (modularization suite)

Inputs: manifest/designs/*.csv (which reports exist), the lean archives named in manifest/_common.py SOURCES,
checkpoints/r2/csynth_modular_lean.tar.gz, and the generators (config text is regenerated, not copied).
Outputs under --out (default release/_bundle): `_tree/` (loose files, for checking) and one tarball per domain
(`configs_<domain>.tar.gz`, `reports_<domain>.tar.gz`, `reports_modular.tar.gz`), each containing a SHA256SUMS of its
members (paths relative to the bundle root, so `sha256sum -c reports/<domain>/SHA256SUMS` works after extraction).
Finally every path in every manifest column is checked against the tree.

    python release/assemble_reports.py [--out DIR] [--domains gemm conv llm] [--no-tar]
"""
import argparse
import csv
import hashlib
import os
import shutil
import sys
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "manifest"))
from _common import DOMAINS, design_table, index_sources, load_generator, locate   # noqa: E402

MODULAR_ARCHIVE = os.path.join("checkpoints", "r2", "csynth_modular_lean.tar.gz")


def read_manifest(domain):
    with open(os.path.join(REPO, "manifest", "designs", f"{domain}.csv"), newline="") as f:
        return {r["design_id"]: r for r in csv.DictReader(f)}


def extract(plan, tree):
    """plan: {archive_local: {member: dest_rel}} -> stream each archive once."""
    n = 0
    for local, wanted in plan.items():
        with tarfile.open(os.path.join(REPO, local), "r:gz") as tf:
            for m in tf:
                dest = wanted.get(m.name)
                if dest and m.isfile():
                    p = os.path.join(tree, dest)
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "wb") as o:
                        shutil.copyfileobj(tf.extractfile(m), o)
                    n += 1
    return n


def tar_dir(tree, subdir, out_path):
    """tar `tree/subdir` into out_path with an inner SHA256SUMS (member hashes) at the archive root."""
    root = os.path.join(tree, subdir)
    sums = []
    for dp, _, files in sorted(os.walk(root)):
        for fn in sorted(files):
            full = os.path.join(dp, fn)
            h = hashlib.sha256(open(full, "rb").read()).hexdigest()
            sums.append(f"{h}  {os.path.relpath(full, tree)}")
    sums_path = os.path.join(tree, f".SHA256SUMS.{subdir.replace('/', '_')}")
    with open(sums_path, "w") as f:
        f.write("\n".join(sums) + "\n")
    with tarfile.open(out_path, "w:gz") as tf:
        tf.add(root, arcname=subdir)
        tf.add(sums_path, arcname=f"{subdir}/SHA256SUMS")      # lands at reports/<domain>/SHA256SUMS after extraction
    os.remove(sums_path)
    return len(sums)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "release", "_bundle"))
    ap.add_argument("--domains", nargs="*", default=DOMAINS)
    ap.add_argument("--no-tar", action="store_true")
    a = ap.parse_args()
    tree = os.path.join(a.out, "_tree")
    shutil.rmtree(tree, ignore_errors=True)
    os.makedirs(tree)

    index = index_sources()
    problems = []
    for domain in a.domains:
        manifest, table, gen = read_manifest(domain), design_table(domain), load_generator(domain)
        plan = {}
        for t in table:
            row = manifest[t["design_id"]]
            # configs: regenerated from the generator
            cp = os.path.join(tree, row["config_path"])
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            with open(cp, "w") as f:
                f.write(gen.build_config_text(t["params"]))
            loc = locate(index, domain, t)
            for col, kind in (("csynth_report", "csynth"), ("impl_report", "impl"), ("impl_power_report", "power")):
                if row[col]:
                    if kind not in loc:
                        problems.append(f"{t['design_id']}: manifest lists {col} but no source report found")
                        continue
                    local, member = loc[kind]
                    plan.setdefault(local, {})[member] = row[col]
        n = extract(plan, tree)
        print(f"{domain}: {len(table)} configs, {n} report files extracted", flush=True)

    # modular suite
    if os.path.isfile(os.path.join(REPO, MODULAR_ARCHIVE)):
        with open(os.path.join(REPO, "manifest", "designs", "modular.csv"), newline="") as f:
            mod = {r["design_id"]: r for r in csv.DictReader(f) if r["csynth"] == "YES"}
        n = 0
        with tarfile.open(os.path.join(REPO, MODULAR_ARCHIVE), "r:gz") as tf:
            for m in tf:
                parts = m.name.split("/")           # modular/<domain>/<name>/project_1/solution1/syn/report/<file>
                if m.isfile() and len(parts) == 8 and parts[3:7] == ["project_1", "solution1", "syn", "report"] \
                        and f"modular/{parts[1]}/{parts[2]}" in mod:
                    p = os.path.join(tree, "reports", "modular", parts[1], parts[2], parts[7])
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "wb") as o:
                        shutil.copyfileobj(tf.extractfile(m), o)
                    n += 1
        print(f"modular: {len(mod)} designs, {n} report files extracted")

    # every path the manifests claim must exist in the tree
    for domain in a.domains + ["modular"]:
        cols = ("config_path", "csynth_report", "impl_report", "impl_power_report") if domain in DOMAINS else ("csynth_report",)
        with open(os.path.join(REPO, "manifest", "designs", f"{domain}.csv"), newline="") as f:
            for r in csv.DictReader(f):
                for col in cols:
                    if r[col] and not os.path.isfile(os.path.join(tree, r[col])):
                        problems.append(f"{r['design_id']}: {col} {r[col]} missing from the assembled tree")
    if problems:
        print(f"{len(problems)} problems, e.g.:\n  " + "\n  ".join(problems[:8]))
        sys.exit(1)
    print("manifest paths: all present in the assembled tree")

    if not a.no_tar:
        for domain in a.domains:
            for kind in ("configs", "reports"):
                if os.path.isdir(os.path.join(tree, kind, domain)):
                    n = tar_dir(tree, f"{kind}/{domain}", os.path.join(a.out, f"{kind}_{domain}.tar.gz"))
                    print(f"{kind}_{domain}.tar.gz: {n} files, {os.path.getsize(os.path.join(a.out, f'{kind}_{domain}.tar.gz')) / 1e6:.0f} MB")
        if os.path.isdir(os.path.join(tree, "reports", "modular")):
            n = tar_dir(tree, "reports/modular", os.path.join(a.out, "reports_modular.tar.gz"))
            print(f"reports_modular.tar.gz: {n} files")


if __name__ == "__main__":
    main()
