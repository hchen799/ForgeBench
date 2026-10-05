"""Shared helpers for manifest/build_manifest.py, manifest/rekey_results.py and release/assemble_reports.py.

Design ids
----------
Public design ids are `<domain>_<NNNN>` (gemm_0001 ...), assigned in generator order (`iter_params()` of
`<domain>/auto_generate_json.py`) and frozen for the release by `manifest/internal/id_map.csv`.
The long parameter-encoded names (`config_stem`) and the R1 names (`legacy_stem`) are internal working keys:
they name run directories, the lean archives and the raw result CSVs until `rekey_results.py` rewrites them.
"""
import csv
import hashlib
import importlib.util
import os
import re
import sys
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAINS = ["gemm", "conv", "llm"]
ID_MAP = os.path.join(REPO, "manifest", "internal", "id_map.csv")

# ZCU102 (xczu9eg-ffvb1156-2-e) totals; BRAM_18K as reported by csynth.
DEVICE = {"bram": 1824, "dsp": 2520, "lut": 274080, "ff": 548160}

# Lean report archives (local paths) and how design names appear inside them.
#   csynth archives: "<domain>/large_hls_files/<name>/project_1/..."        impl archives: "<domain>__<name>/project_1/..."
# `name` is the key used inside the archive: the R1 name for July archives, the current generator name for R2 ones
# (resolved in design_table: key = legacy_stem if the design existed in R1 else config_stem).
SOURCES = [
    dict(kind="csynth", domain="gemm", local="checkpoints/20260720/csynth_gemm_lean.tar.gz"),
    dict(kind="csynth", domain="conv", local="checkpoints/20260720/csynth_conv_lean.tar.gz"),
    dict(kind="csynth", domain="conv", local="checkpoints/r2/csynth_conv_rerun_lean.tar.gz"),
    dict(kind="csynth", domain="gemm", local="checkpoints/r2/csynth_gemm_new_lean.tar.gz"),
    dict(kind="csynth", domain="llm", local="checkpoints/r2/csynth_llm_lean.tar.gz"),
    dict(kind="impl", domain=None, local="checkpoints/20260720/impl1000_reports_lean.tar.gz"),   # conv is valid; gemm/llm members are stale
    dict(kind="impl", domain=None, local="checkpoints/r2/impl_llm_lean.tar.gz"),
    dict(kind="impl", domain=None, local="checkpoints/r2/impl_gemm_lean.tar.gz"),                  # R2 GEMM impl (when collected)
]
# Domains whose July impl results are still valid (gemm was re-sampled and re-run in R2; llm was redone).
JULY_IMPL_VALID = {"conv"}

CSYNTH_TAIL = "project_1/solution1/syn/report/csynth.xml"
IMPL_TAIL = "project_1/solution1/impl/report/verilog/export_impl.xml"
POWER_RE = re.compile(r"project_1/solution1/impl/verilog/project\.runs/impl_1/[^/]*_power_routed\.rpt$")


def load_generator(domain):
    path = os.path.join(REPO, domain, "auto_generate_json.py")
    spec = importlib.util.spec_from_file_location(f"_gen_{domain}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def design_table(domain):
    """[{design_id, stem, legacy, key, params}] in generator order. `key` = name the R1/R2 archives use."""
    gen = load_generator(domain)
    out = []
    for i, p in enumerate(gen.iter_params(), 1):
        stem = gen.config_stem(p)
        legacy = gen.legacy_stem(p) if hasattr(gen, "legacy_stem") else None
        out.append({"design_id": f"{domain}_{i:04d}", "stem": stem, "legacy": legacy or "", "key": legacy or stem,
                    "params": p, "config_text": None})
    return out


def check_id_map(domain, table):
    """The committed id map freezes the stem<->id assignment; fail loudly if the generator order changed."""
    if not os.path.isfile(ID_MAP):
        return
    with open(ID_MAP, newline="") as f:
        frozen = {r["design_id"]: r["generator_name"] for r in csv.DictReader(f) if r["domain"] == domain}
    if frozen:
        now = {t["design_id"]: t["stem"] for t in table}
        if frozen != now:
            bad = [k for k in now if frozen.get(k) != now[k]][:3]
            sys.exit(f"{domain}: generator order no longer matches manifest/internal/id_map.csv (e.g. {bad}); "
                     "ids are frozen - revert the sweep change or regenerate the map deliberately")


def write_id_map(tables):
    os.makedirs(os.path.dirname(ID_MAP), exist_ok=True)
    with open(ID_MAP, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["domain", "design_id", "generator_name", "r1_name"])
        for domain in DOMAINS:
            for t in tables.get(domain, []):
                w.writerow([domain, t["design_id"], t["stem"], t["legacy"]])


def index_sources():
    """Stream each lean archive once. Returns {(domain, key): {kind: (local_archive, member)}, 'top_sha': ...}."""
    index = {}
    for src in SOURCES:
        path = os.path.join(REPO, src["local"])
        if not os.path.isfile(path):
            continue
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
                key, domain = head[-1], src["domain"]
                if domain is None:                         # impl layout: <domain>__<name>
                    if "__" not in key:
                        continue
                    domain, key = key.split("__", 1)
                rec = index.setdefault((domain, key), {})
                loc = (src["local"], m.name)
                if m.name.endswith(CSYNTH_TAIL):
                    rec.setdefault("csynth", loc)
                elif m.name.endswith(IMPL_TAIL):
                    rec.setdefault("impl", loc)
                elif POWER_RE.search(m.name):
                    rec.setdefault("power", loc)
                elif parts[-1] == "top.cpp":
                    rec.setdefault("source", loc)
    return index


def locate(index, domain, t):
    """Report/source locations for one design, merged over the names archives may use: the current generator name
    (R2 archives) first, then the R1 name (July archives). Preferring the current name means a design re-run in R2 never
    resolves to its stale July member."""
    rec = {}
    for k in (t["stem"], t["legacy"]):
        for kind, loc in index.get((domain, k), {}).items():
            rec.setdefault(kind, loc)
    return rec


def impl_failures():
    """{(domain, design_name): reason} from manifest/evidence/**/unfinished_impl_*.csv.
    R1 gemm rows are ignored: the GEMM impl sample was redrawn in R2, so R1 outcomes do not apply to it."""
    out = {}
    root = os.path.join(REPO, "manifest", "evidence")
    for dp, _, files in sorted(os.walk(root)):
        for fn in sorted(files):
            if fn.startswith("unfinished_impl_") and fn.endswith(".csv"):
                with open(os.path.join(dp, fn), newline="") as f:
                    for r in csv.DictReader(f):
                        if r["domain"] == "gemm" and "impl_r1" in dp:
                            continue
                        out[(r["domain"], r["design"])] = r.get("impl_fail_reason") or r["impl_status"]
    return out


def read_metrics(kind, domain):
    path = os.path.join(REPO, "analysis", f"results_{kind}", f"metrics_{domain}.csv")
    if not os.path.isfile(path):
        return {}
    with open(path, newline="") as f:
        return {r["design"]: r for r in csv.DictReader(f)}
