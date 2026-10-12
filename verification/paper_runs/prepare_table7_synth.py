"""Table 7 re-synthesis sources: every program and modularized design in the format they were verified in (<32,10>, truncate/wrap).

Table 7's designs were verified at mixed, <32,10> and float, and all pass at <32,10> (not all at <16,5>), so each P program and its
modularized ("Overall") design is synthesized at <32,10>, built the same way verification built them:

  generated programs (18)   gen_configs.run_hls_flow on the verified design JSON (verification/paper_results/C_32_10__table7_pm1/configs),
                            data_type fixed<32,10>, task csynth                      (verification.prepare_designs.generate_design)
  hand-written programs (8) modular_data/<domain>/hls_files/<name> with data_t/acc_t/math_t rebound to <32,10>, the verification testbench
                            and run_hls.tcl                                          (verification.manual_designs.prepare)
  modularized designs (12)  modular_data/<domain>/hls_files/<name> with the same typedef rebinding plus every spelled-out
                            ap_fixed<16,5> (copied generator code), own run_hls.tcl (not verified: the
                            equivalence check of each merged design against its programs is still open)

Writes <out>/<domain>/<name>/{top.cpp,top.h,tb_top.cpp,run_hls.tcl} and <out>/registry.csv (manifest/designs/modular.csv rows with hls_dir
pointing at <out>), the input of analysis.run_prebuilt:

  python3 -m verification.paper_runs.prepare_table7_synth --out _sweeps/table7_32_10
  python3 -m analysis.run_prebuilt --registry _sweeps/table7_32_10/registry.csv --lean-out _sweeps/lean/modular_csynth_32_10 --jobs 38
"""
import argparse
import csv
import glob
import os
import re
import shutil
import sys

from verification import manual_designs as M
from verification.prepare_designs import generate_design

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DTYPE = "fixed<32,10>"
GEN_CONFIGS = os.path.join(REPO, "verification", "paper_results", "C_32_10__table7_pm1", "configs")
KEEP = ("top.cpp", "top.h", "tb_top.cpp", "run_hls.tcl")
OLD_TYPE = re.compile(r"ap_fixed\s*<\s*16\s*,\s*5\s*>")


def rebind(src, dst):
    """Copy a hand-written design with its typedefs rebound to DTYPE (as manual_designs.prepare does), keeping its own tcl/testbench."""
    os.makedirs(dst, exist_ok=True)
    tys = M.types(DTYPE)
    for fn in os.listdir(src):
        p = os.path.join(src, fn)
        if not os.path.isfile(p) or fn.endswith((".log", ".jou")):
            continue
        if fn in ("top.cpp", "top.h"):
            text = open(p).read()
            for name, c in tys.items():
                text = M._typedef_re(name).sub(f"typedef {c} {name};", text)
            # merged designs copied generated functions with the type spelled out (ap_fixed<16, 5> in signatures/locals); function
            # names (..._ap_fixed_16_5_) are left alone: Table 7's Shared column looks the per-function reports up by name
            text = OLD_TYPE.sub(tys["data_t"], text)
            open(os.path.join(dst, fn), "w").write(text)
        else:
            shutil.copy2(p, dst)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    manual = {v["id"].split("/", 1)[1]: M.SPECS[v["id"].split("/", 1)[1]] for v in M.variants()}   # "<domain>/<name>" -> spec
    rows, how = [], {}
    with open(os.path.join(REPO, "manifest", "designs", "modular.csv"), newline="") as f:
        reg = list(csv.DictReader(f))
    for r in reg:
        dom, name = r["domain"], os.path.basename(r["design_path"])
        dst = os.path.join(out, dom, name)
        key = f"{dom}/{name}"
        if r["role"] == "program" and key in manual:
            M.prepare(manual[key], DTYPE, dst, tasks=("csynth",))
            how[key] = "hand-written program, verification sources"
        elif r["role"] == "program":
            cfg = os.path.join(GEN_CONFIGS, f"designs__{dom}__{name}__fixed_32_10_trn_wrap.json")
            if not os.path.isfile(cfg):
                sys.exit(f"no verified config for {key}: {cfg}")
            work = os.path.join(out, "_gen", dom)
            os.makedirs(work, exist_ok=True)
            run_dir, _ = generate_design(dom, cfg, work, ["csynth"], data_type=DTYPE)
            os.makedirs(dst, exist_ok=True)
            for fn in KEEP:
                if os.path.isfile(os.path.join(run_dir, fn)):
                    shutil.copy2(os.path.join(run_dir, fn), dst)
            how[key] = "generated from the verified design JSON"
        else:
            rebind(os.path.join(REPO, r["design_path"]), dst)
            how[key] = "modularized design, typedefs rebound (not verified)"
        rows.append({**r, "hls_dir": os.path.relpath(dst, REPO), "name": name, "source": how[key]})
    shutil.rmtree(os.path.join(out, "_gen"), ignore_errors=True)
    with open(os.path.join(out, "registry.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for k in sorted(how):
        print(f"{k:<40} {how[k]}")
    print(f"{len(rows)} designs -> {out}/registry.csv")


if __name__ == "__main__":
    main()
