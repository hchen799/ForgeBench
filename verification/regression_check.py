"""Byte-compare regression check for generator changes.

Regenerates a fixed, seeded sample of designs in scratch directories (the repo is not written to) and records SHA-256 of
the emitted files. Run `--save` before a generator change and `--compare` after; the report separates
  design files   top.cpp, top.h, run_hls.tcl   (what Vitis synthesizes: must not change for existing designs)
  testbench      tb_top.cpp                    (reported separately; changes only when the testbench generator changes)
Sample per domain: SAMPLE sweep designs (seeded), every operator variant (verification/operators) and every
`<domain>/test_case_configs/*.json` that is valid JSON.

    python -m verification.regression_check --save  verification/_regression/before.json
    python -m verification.regression_check --compare verification/_regression/before.json
"""
import argparse
import glob
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAINS = ("gemm", "conv", "llm")
DESIGN_FILES = ("top.cpp", "top.h", "run_hls.tcl")
TB_FILES = ("tb_top.cpp",)
SAMPLE, SEED = 50, 20261005


def sample_configs(domain):
    out = {}
    cfgs = sorted(glob.glob(os.path.join(REPO, domain, "auto_generated_configs", "*.json")))
    for p in random.Random(SEED).sample(cfgs, min(SAMPLE, len(cfgs))):
        out["sweep/" + os.path.basename(p)] = p
    sys.path.insert(0, REPO)
    from verification import layout
    for v in layout.variants(domain):
        out["op/" + v["stem"] + ".json"] = v["path"]
    for p in sorted(glob.glob(os.path.join(REPO, domain, "test_case_configs", "*.json"))):
        try:
            json.load(open(p))
        except ValueError:
            continue
        out["case/" + os.path.basename(p)] = p
    return out


def generate(domain, configs, scratch):
    """Generate every config under scratch (a private copy of the domain dir). -> {key: {file: sha256}}"""
    work = os.path.join(scratch, domain)
    os.makedirs(work)
    for f in glob.glob(os.path.join(REPO, domain, "*.py")) + glob.glob(os.path.join(REPO, domain, "*.cpp")):
        shutil.copy2(f, work)
    manifest = os.path.join(work, "_keys.json")
    json.dump(configs, open(manifest, "w"))
    code = (
        "import json, gen_configs\n"
        f"cfgs = json.load(open({manifest!r}))\n"
        "for i, (k, p) in enumerate(sorted(cfgs.items())):\n"
        "    cfg = json.load(open(p)); cfg['task'] = ['csynth']\n"
        "    json.dump(cfg, open(f'_cfg_{i}.json', 'w'))\n"
        "    gen_configs.run_hls_flow(f'_cfg_{i}.json', 'out', task=['csynth'])\n"
    )
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-c", code], cwd=work, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{domain}: generation failed\n{r.stderr[-1500:]}")
    res = {}
    for i, (k, _) in enumerate(sorted(configs.items())):
        d = os.path.join(work, "out", f"_cfg_{i}")
        res[k] = {fn: hashlib.sha256(open(os.path.join(d, fn), "rb").read()).hexdigest()
                  for fn in DESIGN_FILES + TB_FILES if os.path.isfile(os.path.join(d, fn))}
    return res


def snapshot():
    snap = {}
    with tempfile.TemporaryDirectory(prefix="fb_regress_") as scratch:
        for d in DOMAINS:
            for k, v in generate(d, sample_configs(d), scratch).items():
                snap[f"{d}/{k}"] = v
    return snap


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save")
    g.add_argument("--compare")
    a = ap.parse_args()
    snap = snapshot()
    if a.save:
        os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
        json.dump(snap, open(a.save, "w"), indent=0, sort_keys=True)
        print(f"saved {len(snap)} designs -> {a.save}")
        return
    base = json.load(open(a.compare))
    only_base, only_new = sorted(set(base) - set(snap)), sorted(set(snap) - set(base))
    changed = {"design": [], "testbench": []}
    for k in sorted(set(base) & set(snap)):
        for fn in DESIGN_FILES:
            if base[k].get(fn) != snap[k].get(fn):
                changed["design"].append(f"{k}:{fn}")
        for fn in TB_FILES:
            if base[k].get(fn) != snap[k].get(fn):
                changed["testbench"].append(f"{k}:{fn}")
    print(f"compared {len(set(base) & set(snap))} designs ({len(only_base)} only in baseline, {len(only_new)} only now)")
    print(f"design files changed: {len(changed['design'])}   testbench files changed: {len(changed['testbench'])}")
    for kind in ("design", "testbench"):
        for x in changed[kind][:6]:
            print(f"  {kind}: {x}")
    sys.exit(1 if changed["design"] or only_base else 0)


if __name__ == "__main__":
    main()
