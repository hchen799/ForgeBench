#!/usr/bin/env python3
"""Build the per-domain sweep design manifests: manifest/designs/{gemm,conv,llm}.csv.

One row per sweep design. Parameters and names come from the generators themselves
(`{domain}/auto_generate_json.py: iter_params / config_stem / build_config_text`), so the manifest
cannot drift from what the generator emits. Stage status comes from the result CSVs
(analysis/results_*) and from an index of the lean report archives (so every row says exactly
where its csynth.xml / export_impl.xml lives in the release bundle).

    python manifest/build_manifest.py            # all domains
    python manifest/build_manifest.py gemm conv
    python manifest/build_manifest.py --check-disk   # also verify config hashes against */auto_generated_configs

Deterministic: same inputs -> byte-identical CSVs.
"""
import argparse
import csv
import hashlib
import importlib.util
import os
import re
import sys
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "manifest", "designs")

# ZCU102 (xczu9eg-ffvb1156-2-e) totals; BRAM is in 18Kb blocks (as reported by csynth).
DEVICE = {"bram": 1824, "dsp": 2520, "lut": 274080, "ff": 548160}

# Parameter columns per domain, in manifest order (keys of iter_params()).
PARAMS = {
    "gemm": ["M", "K", "N", "unroll_M", "unroll_K", "unroll_N", "comp_order", "gemm_order", "vm_order_1", "vm_order_2",
             "with_bias", "inline"],
    "conv": ["C_IN", "H_IN", "W_IN", "C_OUT", "K", "unroll_cin", "unroll_cout", "pad", "stride", "with_bias",
             "conv_type", "groups", "activation"],
    "llm": ["seq_len", "dim_in", "num_heads", "head_dim", "num_groups", "with_rope", "norm_type", "hd_unroll"],
}

# Where the report archives live in the release bundle (reports/<name>) and where to read them locally.
# kind: csynth|impl ; layout: how the design id is embedded in member paths.
#   "<domain>/large_hls_files/<id>/..."  (csynth archives)      "<domain>__<id>/..."  (impl archive)
SOURCES = [
    dict(kind="csynth", domain="gemm", name="csynth_gemm_lean.tar.gz", local="checkpoints/20260720/csynth_gemm_lean.tar.gz"),
    dict(kind="csynth", domain="conv", name="csynth_conv_lean.tar.gz", local="checkpoints/20260720/csynth_conv_lean.tar.gz"),
    dict(kind="csynth", domain="conv", name="csynth_conv_rerun_lean.tar.gz", local="checkpoints/r2/csynth_conv_rerun_lean.tar.gz"),
    dict(kind="impl", domain=None, name="impl1000_reports_lean.tar.gz", local="checkpoints/20260720/impl1000_reports_lean.tar.gz"),
    # LLM sources are added once the R2 sweeps are packaged (July LLM results are stale: dropout/shape fix).
]
# July impl selections: gemm/conv designs staged for impl are the finished rows + the unfinished evidence rows.
IMPL_DOMAINS_WITH_JULY_SELECTION = {"gemm", "conv"}

CSYNTH_TAIL = "project_1/solution1/syn/report/csynth.xml"
IMPL_TAIL = "project_1/solution1/impl/report/verilog/export_impl.xml"
POWER_RE = re.compile(r"project_1/solution1/impl/verilog/project\.runs/impl_1/[^/]*_power_routed\.rpt$")


def load_generator(domain):
    path = os.path.join(REPO, domain, "auto_generate_json.py")
    spec = importlib.util.spec_from_file_location(f"_gen_{domain}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def index_archive(src):
    """Stream one tar.gz once. Returns {(domain, design): {csynth_path, impl_path, power_path, top_sha}}."""
    out = {}
    path = os.path.join(REPO, src["local"])
    if not os.path.isfile(path):
        print(f"warning: {src['local']} missing; its designs will show no report", file=sys.stderr)
        return out
    with tarfile.open(path, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = m.name.split("/")
            if "project_1" in parts:
                head = parts[:parts.index("project_1")]
            elif parts[-1] == "top.cpp":
                head = parts[:-1]
            else:
                continue
            design = head[-1]
            domain = src["domain"]
            if "__" in design and domain is None:          # impl layout: <domain>__<id>
                domain, design = design.split("__", 1)
            elif domain is None:
                continue
            rec = out.setdefault((domain, design), {})
            if m.name.endswith(CSYNTH_TAIL):
                rec["csynth_path"] = m.name
            elif m.name.endswith(IMPL_TAIL):
                rec["impl_path"] = m.name
            elif POWER_RE.search(m.name):
                rec["power_path"] = m.name
            elif parts[-1] == "top.cpp":
                rec["top_sha"] = hashlib.sha256(tf.extractfile(m).read()).hexdigest()
    return out


def read_csv(path):
    with open(path, newline="") as f:
        return {r["design"]: r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domains", nargs="*", default=["gemm", "conv", "llm"])
    ap.add_argument("--check-disk", action="store_true", help="verify config hashes against */auto_generated_configs")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("indexing report archives ...", flush=True)
    index = {}                      # (domain, design) -> dict of locations from each archive kind
    for src in SOURCES:
        for key, rec in index_archive(src).items():
            slot = index.setdefault(key, {})
            for k, v in rec.items():
                if k not in slot:                       # first archive that has it wins
                    slot[k] = v
                    if k.endswith("path"):
                        slot[k + "_archive"] = "reports/" + src["name"]
    unfinished = {}
    ev = os.path.join(REPO, "manifest", "evidence", "impl_r1_unfinished", "unfinished_impl_r1.csv")
    if os.path.isfile(ev):
        with open(ev, newline="") as f:
            for r in csv.DictReader(f):
                unfinished[(r["domain"], r["design"])] = r

    for domain in a.domains:
        gen = load_generator(domain)
        csynth = read_csv(os.path.join(REPO, "analysis", "results_csynth", f"metrics_{domain}.csv")) \
            if domain != "llm" else {}                  # July LLM results are stale (see CHANGELOG_R2)
        impl = read_csv(os.path.join(REPO, "analysis", "results_impl", f"metrics_{domain}.csv")) \
            if domain in IMPL_DOMAINS_WITH_JULY_SELECTION else {}
        llm_sel = None
        if domain == "llm":
            with open(os.path.join(REPO, "manifest", "llm_impl_selection.csv"), newline="") as f:
                llm_sel = {r["design"] for r in csv.DictReader(f)}
        cols = (["design_id", "legacy_design_id", "domain", "suite"] + PARAMS[domain] +
                ["data_type", "config_path", "config_sha256", "source_sha256", "duplicate_of",
                 "generated", "csynth_status", "csynth_fail_reason", "over_capacity", "over_capacity_resources",
                 "impl_selected", "impl_status", "impl_fail_reason",
                 "csynth_report_archive", "csynth_report_path",
                 "impl_report_archive", "impl_report_path", "impl_power_report_path"])
        rows, first_by_hash = [], {}
        for p in gen.iter_params():
            stem = gen.config_stem(p)
            cfg_text = gen.build_config_text(p)
            cfg_sha = hashlib.sha256(cfg_text.encode()).hexdigest()
            cfg_rel = f"{domain}/auto_generated_configs/{stem}.json"
            if a.check_disk:
                with open(os.path.join(REPO, cfg_rel), "rb") as f:
                    assert hashlib.sha256(f.read()).hexdigest() == cfg_sha, f"disk config differs: {cfg_rel}"
            # legacy_design_id: the id this design had in the R1 sweep (its R1 results/reports are filed under it)
            legacy = gen.legacy_stem(p) if hasattr(gen, "legacy_stem") else None
            key = legacy or stem
            is_new = domain == "llm" or (hasattr(gen, "legacy_stem") and legacy is None)   # no R1/July results exist
            loc = index.get((domain, key), {})
            m = csynth.get(key)
            r = {"design_id": stem, "legacy_design_id": legacy or "", "domain": domain, "suite": "sweep",
                 "data_type": p["data_type"], "config_path": cfg_rel, "config_sha256": cfg_sha}
            for k in PARAMS[domain]:
                v = p[k]
                r[k] = "" if v is None else v
            # source identity / duplicate detection (hash of emitted top.cpp)
            sha = loc.get("top_sha", "")
            r["source_sha256"] = sha
            r["generated"] = bool(sha)
            r["duplicate_of"] = ""
            if sha:
                if sha in first_by_hash:
                    r["duplicate_of"] = first_by_hash[sha]
                else:
                    first_by_hash[sha] = stem
            # csynth
            if m is not None and "csynth_path" in loc:
                r["csynth_status"], r["csynth_fail_reason"] = "ok", ""
                over = [k for k in DEVICE if float(m[k]) > DEVICE[k]]
                r["over_capacity"], r["over_capacity_resources"] = bool(over), "+".join(over)
            elif is_new:
                r["csynth_status"], r["csynth_fail_reason"] = "pending", "R2 sweep not yet collected"
                r["over_capacity"] = r["over_capacity_resources"] = ""
            else:
                r["csynth_status"], r["csynth_fail_reason"] = "fail", "unknown (no report in archive; log not retained)"
                r["over_capacity"] = r["over_capacity_resources"] = ""
            r["csynth_report_archive"] = loc.get("csynth_path_archive", "")
            r["csynth_report_path"] = loc.get("csynth_path", "")
            # impl
            uf = unfinished.get((domain, key))
            if domain == "llm":
                r["impl_selected"] = stem in llm_sel
            elif domain in IMPL_DOMAINS_WITH_JULY_SELECTION and not is_new:
                r["impl_selected"] = (key in impl) or (uf is not None)
            else:
                r["impl_selected"] = ""          # new design: selection not decided yet
            if key in impl and "impl_path" in loc:
                r["impl_status"], r["impl_fail_reason"] = "ok", ""
            elif uf is not None:
                r["impl_status"], r["impl_fail_reason"] = uf["impl_status"], uf["impl_fail_reason"]
            elif is_new and r["impl_selected"] in ("", True):
                r["impl_status"], r["impl_fail_reason"] = "pending", ""
            else:
                r["impl_status"], r["impl_fail_reason"] = "not_selected", ""
            r["impl_report_archive"] = loc.get("impl_path_archive", "")
            r["impl_report_path"] = loc.get("impl_path", "")
            r["impl_power_report_path"] = loc.get("power_path", "")
            rows.append(r)
        out = os.path.join(OUT_DIR, f"{domain}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        n = len(rows)
        cnt = lambda k, v: sum(1 for r in rows if r[k] == v)
        print(f"{domain}: {n} designs | generated {sum(1 for r in rows if r['generated'] is True)} | "
              f"csynth ok {cnt('csynth_status', 'ok')} fail {cnt('csynth_status', 'fail')} pending {cnt('csynth_status', 'pending')} | "
              f"over-capacity {cnt('over_capacity', True)} | impl selected {cnt('impl_selected', True)} "
              f"ok {cnt('impl_status', 'ok')} fail {cnt('impl_status', 'fail')} timeout {cnt('impl_status', 'timeout')} | "
              f"duplicate_of set {sum(1 for r in rows if r['duplicate_of'])} -> {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
