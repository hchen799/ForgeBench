"""Drive place-and-route (implementation) on the 4 base designs and collect PPA.

Paper's post-implementation PPA (concern D2) needs a real impl run (not just
csynth). This runs the three generation-case-study base designs
(GEMM x.A.B.y, DNN Conv-BatchNorm-Activation, LLM Attention-Dropout-Norm) plus a
ResNet design, with the Vitis `impl` flow on ZCU102, then parses resources,
latency, throughput, power, timing closure and Fmax via the Vitis extractor.

Impl PPA parsing (power/Fmax/timing/throughput) is fully implemented — see
analysis/extractors/vitis.py. Generation + TCL (with the `impl` task) are wired;
the actual `vitis_hls` run happens on the server, then `--parse-only` populates
every `PPAMetrics` field.

DESIGNS below are best-guess mappings — CONFIRM the exact case-study configs
(and the ResNet) or override with --config domain:path.
"""
import argparse
import json
import os
import subprocess
import sys

from analysis import collect as collect_mod

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# domain -> config path (relative to repo root). FLAG: confirm these are the exact
# three generation-case-study base designs + the intended ResNet.
DESIGNS = {
    "gemm": "gemm/test_case_configs/mlp.json",                 # x.A.B.y base [CONFIRM]
    "conv": "conv/test_case_configs/conv_A.json",              # Conv-BatchNorm-Activation base [CONFIRM]
    "llm": "llm/test_case_configs/attention_op_p1.json",       # Attention-Dropout-Norm base [CONFIRM]
    "resnet": "conv/test_case_configs/resnet18_block1.json",   # ResNet design [CONFIRM]
}

# domain key -> generator dir (resnet uses the conv generator)
_GEN_DIR = {"gemm": "gemm", "conv": "conv", "llm": "llm", "resnet": "conv"}

# "export_ip" is the generator's keyword that emits
# `export_design -format ip_catalog -flow impl`, i.e. the actual place-and-route
# (implementation) run that produces the impl reports.
DEFAULT_TASKS = ["csynth", "export_ip"]


def generate_impl_design(key, config_path, out_base, tasks):
    """Generate one design with the impl task list; return (run_dir, config)."""
    with open(config_path) as f:
        config = json.load(f)
    config["task"] = list(tasks)  # keep datatype as-is (ap_fixed by default, as in the paper)

    stem = os.path.splitext(os.path.basename(config_path))[0]
    run_name = f"{key}_{stem}"
    gen_dir = os.path.join(REPO_ROOT, _GEN_DIR[key])
    tmp_cfg = os.path.join(gen_dir, f"_vimpl_{run_name}.json")
    with open(tmp_cfg, "w") as f:
        json.dump(config, f)
    try:
        code = (
            "import gen_configs; "
            f"gen_configs.run_hls_flow({os.path.basename(tmp_cfg)!r}, "
            f"base_dir={out_base!r}, task={list(tasks)!r})"
        )
        subprocess.run([sys.executable, "-c", code], cwd=gen_dir,
                       check=True, capture_output=True, text=True)
    finally:
        if os.path.exists(tmp_cfg):
            os.remove(tmp_cfg)
    return os.path.join(out_base, f"_vimpl_{run_name}"), config


def main():
    ap = argparse.ArgumentParser(description="Generate + (server) run impl PPA on the 4 base designs")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "analysis", "_impl"))
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                    help="comma-separated Vitis tasks (default csynth,impl)")
    ap.add_argument("--config", action="append", default=[],
                    help="override design as domain:path (repeatable)")
    ap.add_argument("--parse-only", action="store_true",
                    help="skip generation; parse impl reports already in --out")
    args = ap.parse_args()

    designs = dict(DESIGNS)
    for spec in args.config:
        key, path = spec.split(":", 1)
        designs[key] = path
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    os.makedirs(args.out, exist_ok=True)

    if not args.parse_only:
        print("FLAG: confirm these are the intended case-study designs:")
        for key, path in designs.items():
            print(f"  {key}: {path}")
        for key, path in designs.items():
            abspath = os.path.join(REPO_ROOT, path)
            if not os.path.exists(abspath):
                print(f"  SKIP {key}: {path} not found")
                continue
            run_dir, _ = generate_impl_design(key, abspath, args.out, tasks)
            print(f"generated {key} -> {run_dir}")
        print("\nNext (on the Vitis server), for each generated design dir:")
        print("  (cd <design_dir> && vitis_hls -f run_hls.tcl)")
        print(f"then: python3 -m analysis.impl_runner --parse-only --out {args.out}")

    # Parse whatever impl reports exist (all PPA fields now populated).
    rows = collect_mod.collect(args.out, tool="vitis", flow="impl")
    out_csv = os.path.join(args.out, "impl_metrics.csv")
    if rows:
        collect_mod.write_csv(rows, out_csv)
        print(f"\nwrote {out_csv}: {len(rows)} impl designs "
              "(area, latency, throughput, power, Fmax, timing)")
    else:
        print("\nno impl reports found yet (run vitis_hls on the server first)")


if __name__ == "__main__":
    main()
