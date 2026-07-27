"""Server-side design preparation for the ForgeBench verification flow.

For a domain, generate each design in float with the requested task list
(e.g. ["csim"] or ["csim","cosim"]), then compute and drop the golden files
next to the generated inputs, and register those golden files as testbench
files in run_hls.tcl so Vitis CSIM/CO-SIM can open them.

This does NOT run Vitis (that is run_all.sh's job) and does NOT need g++.
It is the turnkey "prepare everything the server run needs" step.

CLI:
  python -m verification.prepare_designs gemm --task csim --out verification/_server/gemm
  python -m verification.prepare_designs llm  --task csim,cosim
"""
import argparse
import json
import os
import subprocess
import sys

from verification.golden_ref import compute_goldens
from verification.io import write_golden

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _domain_dir(domain):
    return os.path.join(REPO_ROOT, domain)


def generate_design(domain, config_path, out_base, tasks):
    """Generate a float design with `tasks`; return (run_dir, config_dict)."""
    with open(config_path) as f:
        config = json.load(f)
    config["data_type"] = "float"
    config["task"] = list(tasks)

    stem = os.path.splitext(os.path.basename(config_path))[0]
    domain_dir = _domain_dir(domain)
    tmp_cfg = os.path.join(domain_dir, f"_vprep_{stem}.json")
    with open(tmp_cfg, "w") as f:
        json.dump(config, f)
    try:
        code = (
            "import gen_configs; "
            f"gen_configs.run_hls_flow({os.path.basename(tmp_cfg)!r}, "
            f"base_dir={out_base!r}, task={list(tasks)!r})"
        )
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=domain_dir, check=True, capture_output=True, text=True,
        )
    finally:
        if os.path.exists(tmp_cfg):
            os.remove(tmp_cfg)
    return os.path.join(out_base, f"_vprep_{stem}"), config


def register_goldens_in_tcl(run_dir, golden_names):
    """Append `add_files -tb <name>.golden.txt` so Vitis csim/cosim can open them."""
    tcl = os.path.join(run_dir, "run_hls.tcl")
    if not os.path.exists(tcl):
        return
    with open(tcl) as f:
        text = f.read()
    lines = []
    for name in golden_names:
        entry = f"add_files -tb {name}.golden.txt"
        if entry not in text:
            lines.append(entry)
    if not lines:
        return
    # Insert right after the last existing "add_files -tb" line.
    tcl_lines = text.splitlines()
    last_tb = max(i for i, l in enumerate(tcl_lines) if l.startswith("add_files -tb"))
    tcl_lines[last_tb + 1:last_tb + 1] = lines
    with open(tcl, "w") as f:
        f.write("\n".join(tcl_lines) + "\n")


def prepare_one(domain, config_path, out_base, tasks):
    run_dir, config = generate_design(domain, config_path, out_base, tasks)
    goldens = compute_goldens(config, run_dir, domain)
    for name, arr in goldens.items():
        write_golden(os.path.join(run_dir, f"{name}.golden.txt"), arr)
    register_goldens_in_tcl(run_dir, list(goldens))
    return run_dir, sorted(goldens)


def main():
    ap = argparse.ArgumentParser(description="Prepare float designs + goldens for the server run")
    ap.add_argument("domain", choices=["gemm", "conv", "llm"])
    ap.add_argument("--task", default="csim", help="comma-separated Vitis tasks, e.g. csim or csim,cosim")
    ap.add_argument("--out", default=None, help="output base dir (default verification/_server/<domain>)")
    ap.add_argument("--configs-dir", default=None,
                    help="directory of config JSONs (default: <domain>/test_case_configs)")
    ap.add_argument("configs", nargs="*", help="config names (default: all configs in --configs-dir)")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.task.split(",") if t.strip()]
    out_base = args.out or os.path.join(REPO_ROOT, "verification", "_server", args.domain)
    # generate_design runs the codegen subprocess with cwd=<domain>/, so a
    # relative out_base would resolve against that dir there but against the repo
    # root in compute_goldens/write_golden here. Anchor it absolutely up front.
    out_base = os.path.abspath(out_base)
    os.makedirs(out_base, exist_ok=True)

    cfg_dir = args.configs_dir or os.path.join(_domain_dir(args.domain), "test_case_configs")
    if args.configs:
        cfg_paths = [os.path.join(cfg_dir, c if c.endswith(".json") else c + ".json") for c in args.configs]
    else:
        names = sorted(f for f in os.listdir(cfg_dir) if f.endswith(".json"))
        # When an operator has explicit `<op>__<variant>.json` files, skip its bare
        # `<op>.json` base (the template the variants were derived from) so it is
        # not verified twice. Mirrors verification/verify_operators.py.
        ops_with_variants = {n.split("__", 1)[0] for n in names if "__" in n}
        names = [n for n in names
                 if "__" in n or os.path.splitext(n)[0] not in ops_with_variants]
        cfg_paths = [os.path.join(cfg_dir, n) for n in names]

    for cfg in cfg_paths:
        stem = os.path.splitext(os.path.basename(cfg))[0]
        try:
            run_dir, goldens = prepare_one(args.domain, cfg, out_base, tasks)
            print(f"prepared {args.domain}/{stem}: goldens={goldens} -> {run_dir}")
        except Exception as e:  # noqa: BLE001
            print(f"FAILED to prepare {args.domain}/{stem}: {e}")


if __name__ == "__main__":
    main()
