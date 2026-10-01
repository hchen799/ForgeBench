#!/usr/bin/env python3
"""Build the non-sweep design registries: manifest/designs/{ops,cases,modular,modular_cases}.csv.

design_id = "<suite>/<domain>/<name>" (e.g. ops/gemm/dot_product__bias1, cases/conv/resnet18_block1,
modular/gemm/diff_dims_p1). A design appears in exactly one registry file: modular designs that also have a
`test_case_configs` config are listed in modular.csv (with that config_path), not in cases.csv.

    python manifest/build_registry.py

Sources of truth (nothing is hand-typed here):
  ops      verification/op_configs/<domain>/*.json  (same selection rule as verification/prepare_designs:
           a bare `<op>.json` is skipped when `<op>__*.json` variants exist), the generator dispatch in
           <domain>/generate_code.py and the golden dispatch in verification/domains/<domain>.py
  cases    <domain>/test_case_configs/*.json
  modular  modular_data/<category>/hls_files/*  +  modular_data/parse_synth_resourc_util.py:ROWS
           (ROWS defines each test case: its programs, its shared module and its shared functions)
"""
import ast
import csv
import hashlib
import importlib.util
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "manifest", "designs")
DOMAINS = ["gemm", "conv", "llm"]
CATEGORY = {"GEMM": "gemm", "DNN": "conv", "LLM": "llm"}


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def rel(path):
    return os.path.relpath(path, REPO).replace(os.sep, "/")


def compact(obj):
    return json.dumps(obj, separators=(",", ":"), sort_keys=False)


def write_csv(name, cols, rows):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def generator_dispatch(domain):
    """func_name -> generator function name, read from <domain>/generate_code.py:generate_func_def."""
    src = open(os.path.join(REPO, domain, "generate_code.py")).read()
    return dict(re.findall(r"op_info\['func_name'\] == '(\w+)':\s*\n\s*code_line, full_func_name = (\w+)\(", src))


def golden_dispatch(domain):
    """func_name -> golden function name, read from the _DISPATCH dict in verification/domains/<domain>.py."""
    tree = ast.parse(open(os.path.join(REPO, "verification", "domains", f"{domain}.py")).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_DISPATCH" for t in node.targets):
            return {k.value: v.id for k, v in zip(node.value.keys, node.value.values)}
    return {}


def template_of(op):
    fi = op.get("func_info") or []
    return fi[0] if fi and isinstance(fi[0], str) and fi[0].endswith(".cpp") else ""


def under_test(cfg):
    """The op(s) other than load/store (operator variants contain exactly one)."""
    return [(n, o) for n, o in cfg["ops"].items() if o["func_name"] not in ("load", "store")]


# ------------------------------------------------------------------ ops
def build_ops():
    rows = []
    for domain in DOMAINS:
        d = os.path.join(REPO, "verification", "op_configs", domain)
        names = sorted(f for f in os.listdir(d) if f.endswith(".json"))
        with_variants = {n.split("__", 1)[0] for n in names if "__" in n}
        gen, gold = generator_dispatch(domain), golden_dispatch(domain)
        for n in names:
            stem = n[:-5]
            if "__" not in stem and stem in with_variants:
                continue                                    # superseded base config
            path = os.path.join(d, n)
            cfg = json.load(open(path))
            ops = under_test(cfg)
            assert len(ops) == 1, (stem, [o[1]["func_name"] for o in ops])
            op = ops[0][1]
            fn = op["func_name"]
            rows.append({
                "design_id": f"ops/{domain}/{stem}", "domain": domain, "suite": "ops",
                "operator": stem.split("__", 1)[0], "variant": stem.split("__", 1)[1] if "__" in stem else "",
                "op_func": fn, "op_dims": compact(op.get("dims", [])), "op_func_info": compact(op.get("func_info", [])),
                "template": template_of(op),
                "generator_function": f"{domain}/generate_code.py:{gen.get(fn, '')}",
                "golden_function": f"verification/domains/{domain}.py:{gold.get(fn, '')}",
                "data_type": cfg.get("data_type", ""), "input_range": compact(cfg.get("input_range", "")),
                "config_path": rel(path), "config_sha256": sha256_file(path),
            })
    cols = ["design_id", "domain", "suite", "operator", "variant", "op_func", "op_dims", "op_func_info", "template",
            "generator_function", "golden_function", "data_type", "input_range", "config_path", "config_sha256"]
    return cols, rows


# ------------------------------------------------------------------ modular
def load_modular_rows():
    spec = importlib.util.spec_from_file_location("_modular_parse", os.path.join(REPO, "modular_data", "parse_synth_resourc_util.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ROWS


def slug(text):
    t = re.sub(r"\$\\\w+\$", "", text)               # latex footnote marks
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


def build_modular():
    cases = []
    in_cases = {}                                     # (category, design dir) -> [case_id]
    role = {}
    for r in load_modular_rows():
        cat = r["category"]
        case_id = f"modular_case/{cat}/{slug(r['case'])}"
        progs = [p for p in r["programs"] if p]
        for p in progs:
            in_cases.setdefault((cat, p), []).append(case_id)
            role[(cat, p)] = "program"
        in_cases.setdefault((cat, r["module"]), []).append(case_id)
        role[(cat, r["module"])] = "shared_module"
        cases.append({
            "case_id": case_id, "suite_name": r["suite"], "domain": cat,
            "case_name": re.sub(r"\s+", " ", re.sub(r"\$\\\w+\$", "", r["case"])).strip(),
            "programs": ";".join(f"modular/{cat}/{p}" for p in progs),
            "module": f"modular/{cat}/{r['module']}", "shared_functions": ";".join(r["shared"]),
        })
    designs = []
    for cat in DOMAINS:
        base = os.path.join(REPO, "modular_data", cat, "hls_files")
        cfg_dir = os.path.join(REPO, cat, "test_case_configs")
        for name in sorted(os.listdir(base)):
            ddir = os.path.join(base, name)
            if not os.path.isdir(ddir):
                continue
            files = set(os.listdir(ddir))
            cfg = os.path.join(cfg_dir, name + ".json")
            has_cfg = os.path.isfile(cfg)
            r_ = role.get((cat, name), "unused")
            if has_cfg:
                construction = "generated"
            elif r_ == "shared_module":
                construction = "manual"
            else:
                construction = "generated (config not in repo)"
            designs.append({
                "design_id": f"modular/{cat}/{name}", "domain": cat, "suite": "modular", "name": name, "role": r_,
                "test_cases": ";".join(in_cases.get((cat, name), [])),
                "construction": construction,
                "config_path": rel(cfg) if has_cfg else "", "config_sha256": sha256_file(cfg) if has_cfg else "",
                "hls_dir": rel(ddir), "source_sha256": sha256_file(os.path.join(ddir, "top.cpp")),
                "has_testbench": "tb_top.cpp" in files,
                "has_dram_inputs": any(f.startswith(("DRAM_", "BRAM_")) and f.endswith(".txt") for f in files),
                "csynth_report": "missing" if not os.path.isfile(os.path.join(ddir, "project_1", "solution1", "syn", "report", "csynth.xml")) else "present",
            })
    dcols = ["design_id", "domain", "suite", "name", "role", "test_cases", "construction", "config_path", "config_sha256",
             "hls_dir", "source_sha256", "has_testbench", "has_dram_inputs", "csynth_report"]
    ccols = ["case_id", "suite_name", "domain", "case_name", "programs", "module", "shared_functions"]
    return dcols, designs, ccols, cases


# ------------------------------------------------------------------ cases
def family(domain, name):
    if domain == "conv":
        m = re.match(r"(resnet\d+|vgg\d+)_", name)
        return m.group(1) if m else "conv"
    return {"gemm": "gemm", "llm": "attention"}[domain] if not re.match(r"(gpt|llama)", name) else "transformer"


def build_cases(modular_designs):
    modular_cfgs = {d["config_path"] for d in modular_designs if d["config_path"]}
    rows = []
    for domain in DOMAINS:
        d = os.path.join(REPO, domain, "test_case_configs")
        for n in sorted(f for f in os.listdir(d) if f.endswith(".json")):
            path = os.path.join(d, n)
            if rel(path) in modular_cfgs:
                continue                                  # registered under modular/
            name = n[:-5]
            hls = os.path.join(REPO, domain, "hls_files", name)
            try:
                cfg, runnable = json.load(open(path)), True
            except ValueError:                            # symbolic template (e.g. conv_variable.json: dims are variable names)
                cfg, runnable = {"ops": {}}, False
            funcs = [o["func_name"] for o in cfg["ops"].values() if o["func_name"] not in ("load", "store")]
            rows.append({
                "design_id": f"cases/{domain}/{name}", "domain": domain, "suite": "cases", "name": name,
                "family": family(domain, name), "runnable": runnable, "n_compute_ops": len(funcs), "compute_ops": ";".join(funcs),
                "data_type": cfg.get("data_type", ""),
                "config_path": rel(path), "config_sha256": sha256_file(path),
                "hls_dir": rel(hls) if os.path.isdir(hls) else "",
                "has_testbench": os.path.isfile(os.path.join(hls, "tb_top.cpp")),
            })
    cols = ["design_id", "domain", "suite", "name", "family", "runnable", "n_compute_ops", "compute_ops", "data_type",
            "config_path", "config_sha256", "hls_dir", "has_testbench"]
    return cols, rows


def main():
    cols, rows = build_ops()
    write_csv("ops.csv", cols, rows)
    dcols, designs, ccols, cases = build_modular()
    write_csv("modular.csv", dcols, designs)
    write_csv("modular_cases.csv", ccols, cases)
    cols, rows = build_cases(designs)
    write_csv("cases.csv", cols, rows)
    # consistency checks
    ids = [r["design_id"] for f in ("ops", "cases", "modular") for r in csv.DictReader(open(os.path.join(OUT, f + ".csv")))]
    assert len(ids) == len(set(ids)), "duplicate design_id across registries"
    print(f"total registered non-sweep designs: {len(ids)}")


if __name__ == "__main__":
    main()
